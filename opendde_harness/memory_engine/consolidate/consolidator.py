"""The host's readable long-term memory files, and the transactional store behind them.

``user.md`` (the profile) and ``episodes.md`` (the episodic log) are advertised
to the agent as ordinary workspace files, and every read and write of them goes
through :class:`MemoryStore`. Nothing here decides *what* to write: the prompts
that annotate a conversation and rewrite a profile section live in
:mod:`opendde_harness.memory_engine.host_backend`, which is the default
:class:`~opendde_harness.memory_engine.backend.MemoryBackend`. A workspace with a
plugin backend configured has that backend as its owner instead, and these two
files are then whatever their user maintains by hand.

What :class:`MemoryStore` owns is the shape of those files and the discipline of
changing them:

* the section-aware read the system prompt carries (:meth:`get_memory_context`);
* the episode append, locked and fsynced so a crashed writer's partial line
  cannot merge with the next record;
* the one-section profile splice, compare-and-set against the profile the caller
  read, so a rewrite computed from a stale profile is refused rather than
  dropping another writer's section;
* one advisory lock, shared by every process that writes the profile;
* atomic writes through a **unique** temporary file plus ``os.replace``, so an
  interrupted write can never leave a half-file where the profile was, and two
  writers can never collide on one temp name;
* a **versioned** state file for the store's cursors -- the extraction outbox's
  acknowledgement and each tag's refresh offset -- read and written only under
  the lock, with damaged or unrecognised state renamed aside rather than
  overwritten so it can be diagnosed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

from loguru import logger

from opendde_harness.utils.atomic_io import locked_append

_RELEVANCE_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _tokenize_for_relevance(text: str) -> set[str]:
    """Lowercased set of 2+ char alphanumeric/CJK runs. Lightweight stand-in
    for proper tokenization — Chinese runs become single multi-char tokens
    (e.g. a 3-character phrase is one token), English splits on whitespace + punctuation."""
    return {t.lower() for t in _RELEVANCE_TOKEN_RE.findall(text)}


def parse_user_md_sections(content: str) -> dict[str, str]:
    """Return ``{H2_heading_line: body}`` for every H2 section in ``content``.

    The H1 preamble (text before the first H2) is dropped. Body is the
    text between the heading and the next H2 (or EOF) with leading and
    trailing blank lines stripped. Order of insertion matches order in
    the source file.

    Public because ``user.md`` is a hand-maintained file: anything that wants
    to read it a section at a time parses it here, so the host only ever has
    one idea of what a section is.
    """
    lines = content.splitlines()
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip("\n")
            current = line.strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip("\n")
    return sections


def _score_section_relevance(query: str, heading: str, body: str) -> float:
    """Lexical-overlap relevance: how much query vocabulary appears in
    heading or body. Heading hits weighted 3x because section titles are
    short and intentional, e.g. asking about "Projects" should reliably
    pull '## Projects'."""
    q_tokens = _tokenize_for_relevance(query)
    if not q_tokens:
        return 0.0
    heading_hits = len(q_tokens & _tokenize_for_relevance(heading))
    body_hits = len(q_tokens & _tokenize_for_relevance(body))
    return heading_hits * 3.0 + body_hits


# ── Episodes, as the episodic log keeps them ───────────────────────────
#
# One episode is one line: ``[YYYY-MM-DD HH:MM] <summary> #tag #tag``. The tags
# are what makes the log more than a diary -- they are the trigger for a profile
# section refresh, and the index the refresh reads back.

_EPISODE_LINE_RE = re.compile(r"^\s*\[(\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2})\]\s+(.*?)\s*$")
_TAG_RE = re.compile(r"#([a-z][a-z0-9-]*)")


def parse_episode_line(line: str) -> tuple[str, str, list[str]] | None:
    """Split an ``episodes.md`` line into ``(timestamp, summary, tags)``.

    Returns None for lines that don't match the
    ``[YYYY-MM-DD HH:MM] <summary> #tag #tag`` shape. Tag tokens are
    stripped from the returned summary.
    """
    m = _EPISODE_LINE_RE.match(line)
    if not m:
        return None
    ts, body = m.group(1), m.group(2)
    tags = _TAG_RE.findall(body)
    summary = _TAG_RE.sub("", body).strip()
    return ts, summary, tags


# Code-layer guards.
#
# Each of these enforces a rule the annotation prompt already states but the
# model does not reliably follow at 30-day scale. Kept as pure functions so they
# can be unit-tested in isolation and reasoned about without the rest of
# MemoryStore.

#: Stored without the leading '#' since ``_TAG_RE`` already strips it when
#: parsing episode lines.
_PROCESS_TAGS: frozenset[str] = frozenset({"question", "habit", "answer"})
_VALID_CONFIDENCE: frozenset[str] = frozenset({"low", "medium", "high"})
_SRC_LINK_RE = re.compile(r"\[src:\s+episodes\.md\s+@\s+\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}\]")


def is_process_only_episode(line: str) -> bool:
    """True iff the episode's only tags are #question / #habit / #answer.

    The prompt declares these episodes invalid (a process tag must accompany a
    content tag). The model still emits them a small share of the time; this
    filter drops them at the write-back boundary so they never reach
    ``episodes.md`` or heat a tag that has no content to anchor a refresh to.

    Unparseable / untagged lines fall through (return False) so an unrelated
    freeform note is not suppressed.
    """
    parsed = parse_episode_line(line)
    if not parsed:
        return False
    _, _, tags = parsed
    if not tags:
        return False
    return {t.lower() for t in tags}.issubset(_PROCESS_TAGS)


def drop_bullets_without_src(body: str) -> tuple[str, int]:
    """Strip profile bullets that lack a ``[src: episodes.md @ ts]`` link.

    Non-bullet lines (blank, headings, prose) are preserved verbatim. Bullets
    without the evidence link are dropped -- the prompt mandates every profile
    bullet cite the episode it came from, and a bullet that cites nothing is the
    shape a speculation takes.

    Returns ``(cleaned_body, n_dropped)``.
    """
    kept: list[str] = []
    dropped = 0
    for line in body.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            kept.append(line)
            continue
        if _SRC_LINK_RE.search(line):
            kept.append(line)
        else:
            dropped += 1
    return "\n".join(kept), dropped


# ── Foresight, as user.md keeps it ─────────────────────────────────────
#
# Predictions persist to the ``## Foresight`` section. Bullet format (single
# line, mirroring the ``[src:]`` evidence link profile bullets carry):
#   - <prediction> (from <gen_ts>, window: <range>, confidence: <level>, src: episodes.md @ <ep_ts>)
# ``from`` is the wall time the prediction was emitted; ``src`` is the episode
# timestamp that triggered it.

FORESIGHT_HEADING = "## Foresight"
_FORESIGHT_BULLET_RE = re.compile(
    r"^-\s+(?P<prediction>.+?)\s+"
    r"\(from\s+(?P<gen_ts>[^,]+),\s+"
    r"window:\s+(?P<window>[^,]+),\s+"
    r"confidence:\s+(?P<confidence>[^,]+),\s+"
    r"src:\s+episodes\.md\s+@\s+(?P<src_ts>.+?)\)\s*$"
)
#: Tokens stripped before the semantic dedup of foresight predictions -- common
#: subjects / auxiliaries / framing words that carry no topical content.
_FORESIGHT_DEDUP_STOPWORDS: frozenset[str] = frozenset(
    {
        "user",
        "will",
        "may",
        "might",
        "likely",
        "again",
        "today",
        "tomorrow",
        "next",
        "this",
        "that",
        "the",
        "for",
        "with",
        "from",
        "and",
        "are",
        "has",
        "have",
        "continue",
        "recurring",
        "pattern",
        "habit",
    }
)
_FORESIGHT_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}|[\u4e00-\u9fff]{2,}")
#: Jaccard threshold for "same claim, reworded". 0.6 catches recurring-habit
#: clusters while leaving obviously distinct predictions alone.
_FORESIGHT_SEMANTIC_DUP_JACCARD: float = 0.6


def _stem_trailing_s(token: str) -> str:
    """Cheap plural -> singular: trim a trailing single 's' from tokens of five
    characters or more that do not end in 'ss'.

    Handles ``reminders`` -> ``reminder``, ``meetings`` -> ``meeting`` so the
    Jaccard comparison catches sibling forms without a real stemmer. Leaves
    ``boss``, ``class`` and -ing / -ed forms alone: the miss costs one skipped
    dedup, the alternative costs a morphology dependency.
    """
    if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _foresight_token_set(prediction: str) -> frozenset[str]:
    """Tokenize a prediction for the semantic-dedup comparison.

    English words of four characters or more plus CJK runs of two or more,
    lowercased, minus the framing words that carry no topical content. Each
    surviving token is s-stemmed so plural / singular siblings collapse.
    """
    raw = _FORESIGHT_TOKEN_RE.findall(prediction.lower())
    return frozenset(_stem_trailing_s(t) for t in raw if t not in _FORESIGHT_DEDUP_STOPWORDS)


def _is_semantic_duplicate_foresight(new_pred: str, existing_preds: list[str]) -> bool:
    """True iff ``new_pred`` overlaps an existing prediction by Jaccard >= threshold.

    The ``(prediction, src_ts)`` key is exact-string, so it lets through reworded
    re-emissions of the same claim from a different episode. This catches those.
    """
    new_tokens = _foresight_token_set(new_pred)
    if not new_tokens:
        return False
    for ex in existing_preds:
        ex_tokens = _foresight_token_set(ex)
        if not ex_tokens:
            continue
        union = new_tokens | ex_tokens
        if not union:
            continue
        if len(new_tokens & ex_tokens) / len(union) >= _FORESIGHT_SEMANTIC_DUP_JACCARD:
            return True
    return False


def _normalize_confidence(value: str) -> str:
    """Return ``value`` when it is one of {low, medium, high}, else ``'?'``.

    The model occasionally answers 'strong' / 'likely' / 'definite' instead of
    the prompt's enum; rendering '?' makes the deviation visible in ``user.md``
    rather than silently persisting a value nothing else understands.
    """
    v = (value or "").strip().lower()
    return v if v in _VALID_CONFIDENCE else "?"


def _format_foresight_bullet(entry: dict[str, Any], generation_ts: str) -> str:
    """Render one foresight dict as a single ``user.md`` bullet.

    Tolerates missing / blank fields (substituting ``?``) so partial output
    still leaves a record that can be reviewed later.
    """
    pred = (entry.get("prediction") or "").strip() or "?"
    window = (entry.get("window") or "").strip() or "?"
    confidence = _normalize_confidence(entry.get("confidence") or "")
    src_ts = (entry.get("src_ts") or "").strip() or "?"
    return f"- {pred} (from {generation_ts}, window: {window}, confidence: {confidence}, src: episodes.md @ {src_ts})"


# ── Splicing one H2 section ────────────────────────────────────────────


def _splice_h2_section(content: str, heading: str, new_body: str) -> str:
    """``content`` with the body of the H2 named by ``heading`` replaced.

    The body is everything from the line after the heading up to (but not
    including) the next H2, or EOF when the heading is the last one. The H1
    preamble and every other H2 are preserved byte for byte. A heading that is
    not found is appended as a fresh section at end of file.
    """
    lines = content.splitlines()
    target = heading.strip()

    h_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == target:
            h_idx = i
            break

    if h_idx is None:
        sep = "\n\n" if content and not content.endswith("\n\n") else ""
        return content.rstrip("\n") + sep + "\n" + target + "\n\n" + new_body.strip("\n") + "\n"

    next_h2 = None
    for i in range(h_idx + 1, len(lines)):
        if lines[i].startswith("## "):
            next_h2 = i
            break

    before = lines[: h_idx + 1]
    after = lines[next_h2:] if next_h2 is not None else []
    pieces = list(before) + [""] + new_body.strip("\n").splitlines() + [""]
    if after:
        pieces.extend(after)
    return "\n".join(pieces).rstrip("\n") + "\n"


def _splice_h2_section_at_end(content: str, heading: str, new_body: str) -> str:
    """Like :func:`_splice_h2_section`, but the named section ends up last.

    An absent section is appended; an existing one is removed from where it is
    and appended with the new body. Used for ``## Foresight``, which is written
    by one path only and reads as a footer rather than as part of the profile
    the user maintains.
    """
    lines = content.splitlines()
    target = heading.strip()

    h_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == target:
            h_idx = i
            break

    if h_idx is not None:
        next_h2 = None
        for i in range(h_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                next_h2 = i
                break
        lines = lines[:h_idx] + lines[next_h2:] if next_h2 is not None else lines[:h_idx]

    content_without = "\n".join(lines).rstrip("\n")
    body = new_body.strip("\n")
    if not content_without:
        return f"{target}\n\n{body}\n"
    return f"{content_without}\n\n{target}\n\n{body}\n"


def _ensure_foresight_at_end(content: str) -> str:
    """Move ``## Foresight`` to the end when a splice has buried it.

    Idempotent: returns ``content`` unchanged when the section is absent or
    already last.
    """
    sections = parse_user_md_sections(content)
    if FORESIGHT_HEADING not in sections:
        return content
    order = list(sections.keys())
    if order and order[-1] == FORESIGHT_HEADING:
        return content
    return _splice_h2_section_at_end(content, FORESIGHT_HEADING, sections[FORESIGHT_HEADING])


def _atomic_write_text(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` through a unique temp file in the same directory.

    The temp name is unique per write (``mkstemp``), so two writers -- in this
    process or another -- never truncate each other's half-written file the way
    a fixed ``.tmp`` sibling allows. The content is fsynced before the rename,
    so the rename publishes a complete file or nothing at all: a crash leaves
    the previous content exactly as it was, plus a stray temp file that no
    reader looks at. A failure this side of the rename unlinks its own temp and
    propagates -- a caller that is told nothing about a failed write would go
    on believing the profile it just wrote is on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class MemoryStore:
    """The scoped local-backend profile and episodic log, read transactionally.

    Two files under the instance's ``memory/host/<scope>``: ``profile/user.md`` (what
    :meth:`get_memory_context` puts in the prompt) and
    ``episodic/episodes.md`` (grep-searchable by the agent). Plus one
    versioned state file and shared lock in ``state/<scope>/memory``.
    """

    #: Bumped when the shape of the state file changes. A file written by a
    #: version this code does not know is preserved, not reinterpreted.
    STATE_VERSION = 1

    def __init__(
        self,
        workspace: Path,
        now_fn: Callable[[], datetime] | None = None,
        *,
        memory_dir: Path | None = None,
        state_dir: Path | None = None,
    ):
        from opendde_harness.config.paths import assert_storage_ready, get_workspace_storage

        storage = get_workspace_storage(workspace)
        assert_storage_ready(storage)
        content = memory_dir if memory_dir is not None else storage.host_memory
        self.state_dir = state_dir if state_dir is not None else storage.memory_state
        # Merely constructing the store must not initialize local memory in
        # external/off mode. Atomic writers create their directories on demand.
        self.memory_file = content / "profile" / "user.md"
        self.history_file = content / "episodic" / "episodes.md"
        self.memory_dir = self.memory_file.parent
        self.memory_lock_path = self.state_dir / "store.lock"
        self.state_file = self.state_dir / "state.json"
        self._now_fn = now_fn or datetime.now

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold an exclusive fcntl lock on user.md so concurrent writers
        across processes don't clobber each other.

        Usage:
            with memory.locked():
                cur = memory.read_long_term()
                memory.write_long_term(cur + "...")
        """
        yield from self._fcntl_locked(self.memory_lock_path)

    def _fcntl_locked(self, lock_path: Path) -> Iterator[None]:
        # Cross-platform advisory lock (portalocker): real serialization on
        # Windows too, instead of the previous win32 no-op that lost
        # concurrent writes to user.md.
        from opendde_harness.utils.portable_lock import file_lock

        with file_lock(lock_path):
            yield

    # ── The profile, as the prompt reads it ─────────────────────────────

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        """Replace user.md atomically. Raises if the write did not land."""
        _atomic_write_text(self.memory_file, content)

    # Section-aware read.
    _SECTION_READ_TOP_K = 2
    _NOTES_HEADING_PREFIX = "## Notes"

    def get_memory_context(
        self,
        current_message: str | None = None,
    ) -> str:
        """Return the memory block to embed in the agent's system prompt.

        ``current_message=None`` (or empty) → full user.md dump. Useful
        for cold-start sessions or when the agent is being pinged without
        a user query.

        ``current_message`` provided → parse user.md into H2 sections,
        score each by lexical overlap with the query, return top-K (2 by
        default) plus '## Notes' as catchall. Sections kept appear in
        their original file order so the prompt reads naturally.

        Falls back to full dump when section parsing yields nothing.
        """
        long_term = self.read_long_term()
        if not long_term:
            return ""
        if not current_message or not current_message.strip():
            return f"## Long-term Memory\n{long_term}"
        sections = parse_user_md_sections(long_term)
        if not sections:
            return f"## Long-term Memory\n{long_term}"
        selected = self._select_relevant_sections(
            current_message,
            sections,
            top_k=self._SECTION_READ_TOP_K,
        )
        if not selected:
            return f"## Long-term Memory\n{long_term}"
        body = "\n\n".join(f"{heading}\n\n{section_body}".rstrip() for heading, section_body in selected.items())
        return f"## Long-term Memory\n\n{body}\n"

    @classmethod
    def _select_relevant_sections(
        cls,
        query: str,
        sections: dict[str, str],
        top_k: int = 2,
    ) -> dict[str, str]:
        """Score each section, keep top-K of those with score > 0 plus
        '## Notes' as a catchall. Sections scoring 0 are NOT included as
        filler — otherwise tied-0 sections leak into the prompt.
        Returned dict preserves source file order for predictable
        rendering.
        """
        scored = [(heading, body, _score_section_relevance(query, heading, body)) for heading, body in sections.items()]
        scored.sort(key=lambda x: x[2], reverse=True)
        keep_keys: set[str] = {h for h, _, score in scored[:top_k] if score > 0}
        for heading in sections:
            if heading.startswith(cls._NOTES_HEADING_PREFIX):
                keep_keys.add(heading)
        return {h: b for h, b in sections.items() if h in keep_keys}

    # ── Versioned state ────────────────────────────────────────────────

    def read_state(self) -> dict[str, Any]:
        """Return the store's state, or ``{}`` when there is none to read.

        Absent and damaged are different things and are reported as such. An
        absent file is a store that has never written one. A file that does not
        parse, is not an object, or carries a ``version`` this code does not
        know is **preserved**: it is renamed to a unique
        ``state.json.damaged-<stamp>`` sibling and the move is logged, so the
        next write starts from empty without destroying the evidence. Resetting
        such a file to ``{}`` in place is what made an absent cursor and a
        corrupt one indistinguishable.
        """
        path = self.state_file
        if not path.exists():
            return {}
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            return self._preserve_damaged_state(f"unreadable ({exc.__class__.__name__})")
        if not isinstance(data, dict):
            return self._preserve_damaged_state(f"not an object ({type(data).__name__})")
        version = data.get("version")
        if version != self.STATE_VERSION:
            return self._preserve_damaged_state(f"version {version!r}, this build reads {self.STATE_VERSION}")
        return data

    def _preserve_damaged_state(self, why: str) -> dict[str, Any]:
        """Move the unusable state file aside and report an empty store."""
        path = self.state_file
        stamp = self._now_fn().strftime("%Y%m%d-%H%M%S")
        aside = path.with_name(f"{path.name}.damaged-{stamp}")
        n = 1
        while aside.exists():
            aside = path.with_name(f"{path.name}.damaged-{stamp}-{n}")
            n += 1
        try:
            path.rename(aside)
        except OSError:
            logger.warning(
                "memory state at {} is unusable ({}) and could not be moved aside; leaving it untouched",
                path,
                why,
            )
            return {}
        logger.warning(
            "memory state at {} is unusable ({}); moved to {} and starting from empty",
            path,
            why,
            aside,
        )
        return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        """Stamp the version and write atomically. Call under :meth:`locked`."""
        _atomic_write_text(
            self.state_file,
            json.dumps({**state, "version": self.STATE_VERSION}, indent=2, sort_keys=True, ensure_ascii=False),
        )

    def read_extraction_cursor(self) -> dict[str, Any]:
        """How far the extraction outbox has been acknowledged.

        ``{"seq": int, "turn_id": str, "acked_at": str}``, or ``{}`` before the
        first acknowledgement. The outbox drains in order, so one cursor says
        everything: every entry at or below ``seq`` is durably extracted, and
        every entry above it is still owed.
        """
        cursor = self.read_state().get("extraction_cursor")
        return dict(cursor) if isinstance(cursor, dict) else {}

    def commit_extraction_cursor(self, *, seq: int, turn_id: str) -> None:
        """Move the cursor to the last entry the backend confirmed.

        The read-modify-write runs under :meth:`locked` and lands through the
        atomic state write, so two processes draining at once cannot lose each
        other's progress -- the failure mode a lock-free cursor update has. The
        cursor never moves backwards: an out-of-order acknowledgement is
        ignored rather than replaying everything between.
        """
        with self.locked():
            state = self.read_state()
            current = state.get("extraction_cursor")
            if isinstance(current, dict) and int(current.get("seq", 0) or 0) >= seq:
                return
            self._write_state(
                {
                    **state,
                    "extraction_cursor": {
                        "seq": int(seq),
                        "turn_id": turn_id,
                        "acked_at": self._now_fn().isoformat(timespec="seconds"),
                    },
                }
            )

    # ── The episodic log ───────────────────────────────────────────────

    def append_episodes(self, lines: list[str]) -> int:
        """Append annotated episode lines to ``episodes.md``. Returns how many landed.

        One line per episode, appended as a contiguous block under the same
        cross-process discipline the outbox uses: the write is locked and
        fsynced, and a crashed writer's partial line cannot merge with the next
        record. Lines whose only tags are process tags are dropped here rather
        than in the caller, because this is the boundary past which a bad line is
        indexed and heats a tag.
        """
        keep = [line.strip() for line in lines if line and line.strip()]
        keep = [line for line in keep if not is_process_only_episode(line)]
        if not keep:
            return 0
        locked_append(self.history_file, keep)
        return len(keep)

    def count_tags(self) -> dict[str, int]:
        """Total occurrences of each tag across the whole of ``episodes.md``."""
        if not self.history_file.exists():
            return {}
        counts: dict[str, int] = {}
        for line in self.history_file.read_text(encoding="utf-8").splitlines():
            parsed = parse_episode_line(line)
            if not parsed:
                continue
            for tag in parsed[2]:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def recent_project_tags(self, *, days: int = 14, limit: int = 12) -> list[tuple[str, int]]:
        """Up to ``limit`` ``(project-tag, count)`` pairs from the last ``days``.

        Fed to the annotation prompt as "slugs you have already used", which is
        what keeps one project from being split across ``#project-x-cli`` /
        ``-docs`` / ``-release``. A tag nobody reuses never gets hot, and a
        project whose slug keeps changing never gets a profile section.
        """
        if not self.history_file.exists():
            return []
        cutoff = self._now_fn() - timedelta(days=days)
        counts: dict[str, int] = {}
        for line in self.history_file.read_text(encoding="utf-8").splitlines():
            parsed = parse_episode_line(line)
            if not parsed:
                continue
            ts, _, tags = parsed
            try:
                when = datetime.strptime(ts.replace("T", " "), "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if when < cutoff:
                continue
            for tag in tags:
                if tag.startswith("project-"):
                    counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]

    def episodes_for_tag(self, tag: str, max_episodes: int = 50) -> list[str]:
        """The most recent episode lines carrying ``tag``, oldest first.

        Read from the tail of the file, so a long log costs the same as a short
        one for the section being refreshed.
        """
        if not self.history_file.exists():
            return []
        matches: list[str] = []
        for line in reversed(self.history_file.read_text(encoding="utf-8").splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = parse_episode_line(stripped)
            if not parsed:
                continue
            if tag in parsed[2]:
                matches.append(stripped)
                if len(matches) >= max_episodes:
                    break
        matches.reverse()
        return matches

    def hot_tags(self, threshold: int) -> list[tuple[str, int, int]]:
        """Tags that have gained ``threshold`` episodes since their last refresh.

        Returns ``[(tag, current_count, previous_offset), ...]``, hottest delta
        first, so the tag with the most unfolded evidence rewrites its section
        before the others.
        """
        offsets = self.read_refresh_offsets()
        hot: list[tuple[str, int, int]] = []
        for tag, current in self.count_tags().items():
            previous = offsets.get(tag, 0)
            if current - previous >= threshold:
                hot.append((tag, current, previous))
        hot.sort(key=lambda row: row[1] - row[2], reverse=True)
        return hot

    # ── The profile, written one section at a time ─────────────────────

    def splice_section(self, heading: str, new_body: str, expected_prev: str) -> bool:
        """Replace one H2 section's body, but only if nobody else wrote meanwhile.

        Compare-and-set: the caller read the profile before spending a model
        call on it, and hands back what it read. If the file has changed since,
        nothing is written and False says so -- the rewrite was computed against
        a profile that no longer exists, and splicing it anyway would drop the
        other writer's section. The caller leaves its tag offset where it is, so
        the next round recomputes rather than loses the evidence.

        Every other H2 is preserved byte for byte; ``## Foresight`` is moved back
        to the end if the splice buried it.
        """
        with self.locked():
            current = self.read_long_term()
            if current != expected_prev:
                logger.info(
                    "MemoryStore: section {!r} not written -- the profile changed while it was being rewritten",
                    heading,
                )
                return False
            new_content = _ensure_foresight_at_end(_splice_h2_section(current, heading, new_body))
            if new_content != current:
                self.write_long_term(new_content)
            return True

    #: How many foresight bullets ``## Foresight`` keeps before the oldest drop off.
    FORESIGHT_MAX_KEEP = 20

    def append_foresight(self, foresights: list[dict[str, Any]], *, max_keep: int | None = None) -> int:
        """Persist predictions to ``## Foresight`` in ``user.md``. Returns how many landed.

        Deduped twice: exactly, on ``(prediction, src_ts)``, and semantically, on
        token overlap with what is already there -- the same claim reworded from
        a later episode is one prediction, not two. Capped FIFO at ``max_keep``
        so the section stays scannable. Written under the profile lock, through
        the atomic write, like every other profile change.
        """
        if not foresights:
            return 0
        cap = max_keep if max_keep is not None else self.FORESIGHT_MAX_KEEP
        gen_ts = self._now_fn().strftime("%Y-%m-%d %H:%M")

        with self.locked():
            current = self.read_long_term()
            sections = parse_user_md_sections(current)

            existing_bullets = [
                line.rstrip()
                for line in sections.get(FORESIGHT_HEADING, "").splitlines()
                if line.lstrip().startswith("-")
            ]
            existing_keys: set[tuple[str, str]] = set()
            existing_predictions: list[str] = []
            for line in existing_bullets:
                m = _FORESIGHT_BULLET_RE.match(line)
                if m:
                    pred_text = m.group("prediction").strip()
                    existing_keys.add((pred_text, m.group("src_ts").strip()))
                    existing_predictions.append(pred_text)

            new_bullets: list[str] = []
            semantic_skipped = 0
            for entry in foresights:
                pred = (entry.get("prediction") or "").strip()
                src_ts = (entry.get("src_ts") or "").strip()
                if not pred or (pred, src_ts) in existing_keys:
                    continue
                if _is_semantic_duplicate_foresight(pred, existing_predictions):
                    semantic_skipped += 1
                    continue
                new_bullets.append(_format_foresight_bullet(entry, gen_ts))
                existing_keys.add((pred, src_ts))
                existing_predictions.append(pred)
            if semantic_skipped:
                logger.info("foresight: skipped {} prediction(s) already made in other words", semantic_skipped)
            if not new_bullets:
                return 0

            bullets = existing_bullets + new_bullets
            if len(bullets) > cap:
                bullets = bullets[-cap:]
            new_content = _splice_h2_section_at_end(
                current or "# Long-term Memory\n",
                FORESIGHT_HEADING,
                "\n".join(bullets),
            )
            if new_content != current:
                self.write_long_term(new_content)
            return len(new_bullets)

    # ── Refresh cursors ────────────────────────────────────────────────

    def read_refresh_offsets(self) -> dict[str, int]:
        """Each tag's episode count as of its last profile refresh.

        ``{}`` for a store that has never refreshed, and for one whose state file
        was unreadable -- :meth:`read_state` has already moved that aside, and a
        tag treated as never refreshed folds its evidence in again rather than
        skipping it.
        """
        offsets = self.read_state().get("refresh_offsets")
        if not isinstance(offsets, dict):
            return {}
        out: dict[str, int] = {}
        for tag, count in offsets.items():
            if isinstance(tag, str) and isinstance(count, (int, float)) and not isinstance(count, bool):
                out[tag] = int(count)
        return out

    def commit_refresh_offset(self, tag: str, count: int) -> None:
        """Record that ``tag``'s section was refreshed with ``count`` episodes behind it.

        Lives in the same versioned state file as the extraction cursor, written
        under the same lock: a refresh in one process cannot lose another's
        progress, and an interrupted write cannot leave half an offset. Never
        moves backwards, so a stale round cannot re-open evidence already folded
        into the profile.
        """
        with self.locked():
            state = self.read_state()
            offsets = dict(self.read_refresh_offsets())
            if offsets.get(tag, 0) >= count:
                return
            offsets[tag] = int(count)
            self._write_state({**state, "refresh_offsets": offsets})
