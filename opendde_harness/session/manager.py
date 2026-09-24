"""Session management for conversation history."""

import copy
import json
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import COMPACTION_KEY, orphan_tool_results
from opendde_harness.session.legacy import read_records
from opendde_harness.utils.atomic_io import atomic_replace, locked_append
from opendde_harness.utils.helpers import ensure_dir, safe_filename

#: The ``_type`` of a journal record that is not a message: the turn and tool
#: lifecycle. Messages and lifecycle records share one append-only file so the
#: order between them is the file's order and needs no reconstruction.
LIFECYCLE_TYPE = "lifecycle"

#: A turn was accepted and its user message recorded; execution follows.
TURN_STARTED = "turn.started"
#: A turn reached its final assistant message with nothing left to run.
TURN_COMPLETED = "turn.completed"
#: A turn was cancelled. Written on the way out, before the cancellation is
#: re-raised, so a reload can tell a cancelled turn from a lost one.
TURN_INTERRUPTED = "turn.interrupted"
#: A tool that changes state outside this process is about to be invoked. Its
#: result may or may not exist; see :meth:`Session.uncertain_tool_calls`.
TOOL_STARTED = "tool.started"

_LIFECYCLE_KINDS = frozenset({TURN_STARTED, TURN_COMPLETED, TURN_INTERRUPTED, TOOL_STARTED})

#: In-memory only: which message a lifecycle record sits behind. Reconstructed
#: from file order on load and stripped again on write, so the file never
#: carries an index that could disagree with its own ordering.
_ANCHOR = "after"

_ID_LOCK = threading.Lock()
_ID_LAST_MS = 0
_ID_LAST_SEQ = 0


def new_record_id(now: datetime | None = None) -> str:
    """Mint a sortable, monotonic record id.

    26 lowercase hex characters in three fixed-width fields: a 48-bit
    millisecond clock, a 16-bit per-millisecond counter, and 40 random bits.
    Fixed widths are what make byte order agree with time order, so the journal
    can be read back in the order it was written without trusting a wall clock
    that a reader might not share.

    The counter is the reason two records written in the same millisecond still
    sort; it also carries the clock forward by a millisecond rather than
    repeating an id when more than 65,536 records land in one. The random tail
    keeps ids from two processes apart.
    """
    global _ID_LAST_MS, _ID_LAST_SEQ

    ms = int((now or datetime.now()).timestamp() * 1000)
    with _ID_LOCK:
        if ms > _ID_LAST_MS:
            _ID_LAST_MS, _ID_LAST_SEQ = ms, 0
        else:
            _ID_LAST_SEQ += 1
            if _ID_LAST_SEQ > 0xFFFF:
                _ID_LAST_MS += 1
                _ID_LAST_SEQ = 0
        stamp, seq = _ID_LAST_MS, _ID_LAST_SEQ
    return f"{stamp:012x}{seq:04x}{secrets.token_hex(5)}"


def unanswered_tool_calls(messages: list[dict[str, Any]]) -> set[str]:
    """Ids of tool calls in ``messages`` that no tool result answers.

    The journal is append-only, so an assistant message is written with every
    call the model asked for and the results land as they return. A turn that
    was cancelled between the two therefore leaves calls with no result, and
    every wire refuses a history carrying one.

    The reader reconciles rather than the writer inventing: this is the mirror
    of :func:`~opendde_harness.providers.base.orphan_tool_results`, which drops
    a result whose call a window cut away.
    """
    answered = {m.get("toolCallId") for m in messages if msg.is_tool_result(m)}
    asked = {call_id for m in messages for call_id in msg.tool_call_ids(m)}
    return asked - answered


def drop_unanswered_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``messages`` with every unanswered tool call removed.

    An assistant message left with neither text nor a surviving call is dropped
    with them: it said nothing and asked for nothing the history can show. The
    call is still in the journal, so what the model intended is not lost -- it
    is just not replayed as an instruction the model would answer twice.
    """
    unanswered = unanswered_tool_calls(messages)
    if not unanswered:
        return messages
    out: list[dict[str, Any]] = []
    for m in messages:
        calls = msg.tool_calls_of(m)
        if not calls:
            out.append(m)
            continue
        kept = [b for b in msg.blocks_of(m) if b.get("type") != msg.TOOL_CALL or b.get("id") not in unanswered]
        if len(kept) == len(msg.blocks_of(m)):
            out.append(m)
            continue
        if not any(b.get("type") == msg.TOOL_CALL for b in kept) and not any(
            b.get("type") == msg.TEXT and b.get("text") for b in kept
        ):
            continue
        out.append(msg.with_blocks(m, kept))
    return out


def new_chat_id(now: datetime | None = None) -> str:
    """Mint an opaque, sortable per-session chat_id: ``YYYYMMDD_HHMMSS_xxxxxx``.

    Sortable by value (timestamp prefix) and collision-safe (uuid suffix);
    channel-agnostic. Becomes the session key's chat_id segment and the JSONL
    filename stem.
    """
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


_AUTO_TITLE_MAX_CHARS = 40


def _derive_title(content: Any) -> str | None:
    """Derive an auto-title from message content: first non-empty line,
    whitespace collapsed, truncated to 40 characters. None when the
    content is not a usable string (e.g. structured multimodal parts)."""
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    collapsed = " ".join(stripped.splitlines()[0].split())
    return collapsed[:_AUTO_TITLE_MAX_CHARS] or None


def _first_user_auto_title(messages: list[dict[str, Any]]) -> str | None:
    for m in messages:
        if m.get("role") == "user":
            return _derive_title(m.get("content"))
    return None


def _record_line(record: dict[str, Any]) -> str:
    """One journal record as its JSONL line.

    The lifecycle anchor is dropped: file order already says where a record
    sits, and a stored index is one more thing that can disagree with it.
    """
    if record.get("_type") == LIFECYCLE_TYPE:
        record = {k: v for k, v in record.items() if k != _ANCHOR}
    return json.dumps(record, ensure_ascii=False)


def _interleave(
    messages: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    start_message: int,
    start_lifecycle: int,
) -> list[dict[str, Any]]:
    """Messages and lifecycle records from the given offsets, in write order.

    One merge over two append-only lists, ordered by each lifecycle record's
    anchor. Used for the whole journal and, with the persisted counts as
    offsets, for the tail one save appends -- so the file's order and the
    in-memory order are produced by the same rule.
    """
    out: list[dict[str, Any]] = []
    life = start_lifecycle
    total = len(lifecycle)

    def _due(upto: int) -> None:
        nonlocal life
        while life < total and lifecycle[life].get(_ANCHOR, 0) <= upto:
            out.append(lifecycle[life])
            life += 1

    _due(start_message)
    for index in range(start_message, len(messages)):
        out.append(messages[index])
        _due(index + 1)
    _due(len(messages))
    return out


@dataclass(frozen=True)
class SessionResolution:
    """Outcome of resolving a user-supplied session id to a full key.

    ``status`` is one of ``"resolved"`` / ``"ambiguous"`` / ``"not_found"``.
    ``key`` carries the full ``channel:chat_id`` when resolved; ``candidates``
    carries the matching full keys when ambiguous. The no-match case is reported
    as ``not_found`` so each caller decides its own tail — the agent
    ``--session`` path mints ``cli:<value>``, while a read-only export errors.
    """

    status: str
    key: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files
    #: How many times this key has been reset. ``/new`` and ``clear`` bump it,
    #: so a turn recorded after a reset can never be confused with one recorded
    #: before it -- the key is the same and the ids are fresh, and that pair
    #: alone does not say which conversation a record belongs to.
    generation: int = 0
    #: The turn and tool lifecycle, in append order. Each entry carries an
    #: in-memory ``after`` anchor: how many messages precede it. Kept apart from
    #: ``messages`` so every reader of the message projection -- the provider
    #: history, the resume wire, the message count -- is unaffected by it.
    lifecycle: list[dict[str, Any]] = field(default_factory=list)
    # Messages already on disk; save() appends only past this index.
    _persisted_count: int = field(default=0, repr=False)
    # Same, for lifecycle records.
    _persisted_lifecycle: int = field(default=0, repr=False)

    def set_title(self, title: str) -> None:
        """Set a human-given title.

        Clears the ``title_auto`` marker so fork inheritance treats the
        title as human even when the session was auto-named before. Every
        rename path must come through here, not assign metadata directly.
        """
        self.metadata["title"] = title
        self.metadata.pop("title_auto", None)

    def record(self, message: dict[str, Any]) -> None:
        """Append a message, stamping a wall-clock timestamp.

        The single choke point for session writes — every persistence path
        (the turn journal, ``journal.record_delivery``, a stored compaction
        marker) must come through here so no message lands unstamped. A
        caller-set ``timestamp`` is preserved, which is every message the model
        service answered: pi stamps its own. Milliseconds, because that is what
        pi's ``Message`` carries. Per-message ordering and turn grouping derive
        from append order and the ``role`` boundary, so no separate received_at
        / turn_id stamp is kept.
        """
        message.setdefault("timestamp", msg.now_ms())
        message.setdefault("id", new_record_id())
        self.messages.append(message)
        self.updated_at = datetime.now()

    def record_lifecycle(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Append one lifecycle record and return it.

        The turn journal's only writer. Stamped like a message (``id``,
        ``timestamp``) and anchored behind the messages recorded so far, which
        is what puts it in the right place in :meth:`records`.
        """
        record = {
            "_type": LIFECYCLE_TYPE,
            "kind": kind,
            "id": new_record_id(),
            "timestamp": datetime.now().isoformat(),
            **fields,
            _ANCHOR: len(self.messages),
        }
        self.lifecycle.append(record)
        self.updated_at = datetime.now()
        return record

    def records(self) -> list[dict[str, Any]]:
        """The journal: messages and lifecycle records in the order written.

        The durable view. :attr:`messages` is the projection a provider is sent;
        this is what actually happened, including the turns that did not finish.
        """
        return _interleave(self.messages, self.lifecycle, 0, 0)

    def turn_status(self) -> dict[str, str]:
        """``{turn_id: "completed" | "interrupted"}`` for every turn on record.

        A turn with a ``turn.started`` record and no completion of either kind
        is inferred interrupted: the process that was running it never came back
        to say so, which is the one case no writer can report for itself.
        """
        status: dict[str, str] = {}
        for record in self.lifecycle:
            turn_id = record.get("turn_id")
            if not isinstance(turn_id, str):
                continue
            kind = record.get("kind")
            if kind == TURN_STARTED:
                status.setdefault(turn_id, "interrupted")
            elif kind == TURN_COMPLETED:
                status[turn_id] = "completed"
            elif kind == TURN_INTERRUPTED:
                status[turn_id] = "interrupted"
        return status

    def uncertain_tool_calls(self) -> list[dict[str, Any]]:
        """The ``tool.started`` records no tool result answers.

        A state-changing tool was invoked and the journal never learned how it
        ended: the job may be running, may have finished, may never have begun.
        Reported so a reader can say so, and deliberately never re-executed --
        running it again is the one response that can do damage twice.
        """
        answered = {m.get("toolCallId") for m in self.messages if msg.is_tool_result(m)}
        return [
            record
            for record in self.lifecycle
            if record.get("kind") == TOOL_STARTED and record.get("call_id") not in answered
        ]

    @property
    def unpersisted(self) -> bool:
        """Whether anything recorded since the last save is still only in memory."""
        return len(self.messages) > self._persisted_count or len(self.lifecycle) > self._persisted_lifecycle

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a user turn.

        ``max_messages=0`` is no cap, not an empty history.
        """
        unconsolidated = self.messages[self.last_consolidated :]
        start = max(0, len(unconsolidated) - max_messages) if max_messages else 0
        # Never past a compaction marker. It is the model's own summary of
        # everything before it, so a cap that starts after one ships the tail
        # of a conversation with neither the summary nor the history it stood
        # in for -- the one slice that has less in it than either alternative.
        marker_at = next((i for i in reversed(range(len(unconsolidated))) if COMPACTION_KEY in unconsolidated[i]), None)
        if marker_at is not None:
            start = min(start, marker_at)
        sliced = unconsolidated[start:]

        # Drop leading non-user messages to avoid orphaned tool_result blocks.
        # A marker is as clean a start as a user message -- nothing before it
        # is sent -- with one exception: a call before it answered after it
        # leaves a result the cut has orphaned, and that result goes too.
        for i, m in enumerate(sliced):
            if m.get("role") == "user" or COMPACTION_KEY in m:
                sliced = sliced[i:]
                orphans = orphan_tool_results(sliced)
                if orphans:
                    sliced = [m for at, m in enumerate(sliced) if at not in orphans]
                break

        # The record is what the request carries, less the harness's own
        # bookkeeping: a pi message replays whole, thinking signatures and
        # native tool-call ids included, and ``compaction`` travels because it
        # is the boundary a backend that compacted the conversation itself
        # replays in place of everything before it. Dropped, the request falls
        # back to the local history and the compaction is paid for again every
        # eligible turn.
        out = [msg.wire_projection(m) for m in sliced]
        # A turn cancelled between an assistant's tool calls and their results
        # leaves calls nothing answers, and the journal keeps them rather than
        # inventing a result. They must not reach a request: a history with an
        # unanswered call is a 400 on every turn that follows.
        return drop_unanswered_tool_calls(out)

    def clear(self) -> None:
        """Clear all messages and start a new generation.

        The key survives the reset, so the generation is what tells the records
        written after it apart from the ones written before -- the fact an
        extraction job replayed from the outbox needs in order to say which
        conversation it belongs to.
        """
        self.messages = []
        self.lifecycle = []
        self.last_consolidated = 0
        self.generation += 1
        self.updated_at = datetime.now()

    def undo_last_turn(self, n: int = 1) -> int:
        """Drop the last ``n`` user-turn blocks from the unconsolidated tail.

        A turn starts at a ``role == "user"`` message and runs to the next
        user message (its assistant/tool followers inherit it). Only the
        unconsolidated tail (``messages[last_consolidated:]``) is eligible —
        content already summarized into MEMORY.md is never crossed. Returns
        the number of messages removed (0 when the tail has no user message).
        Persistence is the caller's job via ``SessionManager.save``.
        """
        if n < 1:
            return 0
        start = self.last_consolidated
        user_starts = [i for i in range(start, len(self.messages)) if self.messages[i].get("role") == "user"]
        if not user_starts:
            return 0
        cut_index = user_starts[-n] if n <= len(user_starts) else user_starts[0]
        removed = len(self.messages) - cut_index
        dropped_turns = {m.get("turn_id") for m in self.messages[cut_index:] if m.get("turn_id")}
        self.messages = self.messages[:cut_index]
        # The dropped turns' lifecycle goes with them. Left behind, a
        # ``turn.started`` whose messages no longer exist reads as a turn that
        # was interrupted, and the next resume would say so. By turn id, not by
        # anchor alone: one turn's completion and the next one's start sit behind
        # the same message, and only the id tells them apart.
        self.lifecycle = [
            r for r in self.lifecycle if r.get(_ANCHOR, 0) <= cut_index and r.get("turn_id") not in dropped_turns
        ]
        self.updated_at = datetime.now()
        return removed


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path, *, sessions_dir: Path | None = None):
        from opendde_harness.config.paths import assert_storage_ready, get_workspace_storage

        assert_storage_ready(get_workspace_storage(workspace))
        self.workspace = workspace
        self.sessions_dir = ensure_dir(
            sessions_dir if sessions_dir is not None else get_workspace_storage(workspace).sessions
        )
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session: sessions/{channel}/{chat_id}.jsonl."""
        channel, _, chat_id = key.partition(":")
        return self.sessions_dir / safe_filename(channel) / f"{safe_filename(chat_id)}.jsonl"

    @staticmethod
    def key_from_path(path: Path) -> str:
        """Best-effort reverse of the nested filename encoding for a session
        file: channel is the parent directory, chat_id is the stem.

        The on-disk ``_type:metadata`` key is authoritative when present and
        wins over this; callers use it only as the fallback for metadata-less
        files. ``safe_filename`` is non-invertible, so any character it folds
        to ``_`` (``/``, ``:``, ...) is not recovered here.
        """
        return f"{path.parent.name}:{path.stem}"

    def resolve_key(self, value: str) -> SessionResolution:
        """Resolve a session id to a full ``channel:chat_id`` key across channels.

        Shared resolution core for the agent ``--session`` path and session
        export:

        - a value containing ':' is already a full key -> resolved;
        - exactly one exact chat_id match across channels -> resolved;
        - exactly one prefix match -> resolved;
        - more than one match -> ambiguous (candidate full keys);
        - no match -> not_found.

        The no-match tail is reported as ``not_found``; callers decide whether to
        mint (agent ``--session``) or error (read-only export).
        """
        if ":" in value:
            return SessionResolution("resolved", key=value)
        sessions = self.list_sessions(channel=None)
        exact = [s for s in sessions if s["key"].partition(":")[2] == value]
        matches = exact or [s for s in sessions if s["key"].partition(":")[2].startswith(value)]
        if len(matches) > 1:
            return SessionResolution("ambiguous", candidates=tuple(s["key"] for s in matches))
        if matches:
            return SessionResolution("resolved", key=matches[0]["key"])
        return SessionResolution("not_found")

    def find_most_recent_chat_id(self, channel: str) -> str | None:
        """Return the chat_id of the most-recently-updated session on this
        channel, or None if no such session exists.

        Used by cron delivery at trigger time to auto-resolve where to
        forward ephemeral (cli / tui) reminders, so users don't need to
        know their own open_id / chat_id on the target channel.

        Reads each candidate file's metadata line (first line of the JSONL)
        to get the authoritative session key ``<channel>:<chat_id>`` and
        ``updated_at``; recency is decided by ``updated_at``, falling back
        to file mtime for files that lack it.

        Sessions with messages take priority: a freshly minted zero-message
        session (``sessions create``) must not hijack delivery away from the
        user's real last conversation. Empty sessions are only considered
        when the channel has no session with messages at all.
        """
        channel_dir = self.sessions_dir / safe_filename(channel)
        if not channel_dir.is_dir():
            return None

        best_chat_id: str | None = None
        best_updated = ""
        best_empty_chat_id: str | None = None
        best_empty_updated = ""
        for p in channel_dir.glob("*.jsonl"):
            meta, count = self._scan_file(p)
            if meta is None:
                continue
            key_val = meta.get("key", "")
            if ":" not in key_val:
                continue
            ch, chat_id = key_val.split(":", 1)
            if ch != channel or not chat_id:
                continue
            updated = meta.get("updated_at")
            if not isinstance(updated, str) or not updated:
                try:
                    updated = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                except OSError:
                    continue
            if count > 0:
                if updated > best_updated:
                    best_chat_id = chat_id
                    best_updated = updated
            elif updated > best_empty_updated:
                best_empty_chat_id = chat_id
                best_empty_updated = updated
        return best_chat_id if best_chat_id is not None else best_empty_chat_id

    @staticmethod
    def _scan_file(path: Path) -> tuple[dict[str, Any] | None, int]:
        """Single pass over a session file: return (last metadata record,
        message line count).

        One metadata record is appended per save, so the last reflects
        current state. Message lines are counted without keeping them in
        memory.
        """
        meta: dict[str, Any] | None = None
        count = 0
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    if data.get("_type") == "metadata":
                        meta = data
                    elif data.get("_type") != LIFECYCLE_TYPE:
                        count += 1
        except OSError:
            return None, 0
        return meta, count

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            lifecycle: list[dict[str, Any]] = []
            metadata = {}
            created_at = None
            last_consolidated = 0
            generation = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # Partial trailing line from a crashed append.
                        logger.debug("Skipping undecodable line in session {}", key)
                        continue

                    # Metadata records are appended per save; last one wins.
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                        generation = int(data.get("generation", 0) or 0)
                    elif data.get("_type") == LIFECYCLE_TYPE:
                        # The anchor is the file's own ordering, read back.
                        data[_ANCHOR] = len(messages)
                        lifecycle.append(data)
                    else:
                        # A file written before ids existed still loads, and
                        # gets ids here rather than on a rewrite: adding a
                        # field is not a reason to rewrite a conversation, and
                        # the append-only log has no place to put one anyway.
                        data.setdefault("id", new_record_id())
                        messages.append(data)

            # A file written before the records were pi messages is converted
            # here, in memory, for the same reason: the bytes on disk stay as
            # they were written (see ``session.legacy``).
            messages = read_records(messages)

            session = Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
                generation=generation,
                lifecycle=lifecycle,
            )
            session._persisted_count = len(messages)
            session._persisted_lifecycle = len(lifecycle)
            return session
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk.

        Appends a fresh metadata record plus the not-yet-persisted messages
        under a cross-process lock, so concurrent writers never lose each
        other's turns and a turn's messages stay contiguous. A shrunken
        message list (clear) rewrites the file atomically instead.

        An untitled session is auto-named here from its first user message
        (first line, collapsed whitespace, capped at 40 chars); a title set
        by the user is never overwritten. Forked children are excluded —
        their first user message names the fork point's ancestor, not the
        fork — so they stay untitled unless titled explicitly.
        """
        path = self._get_session_path(session.key)

        if not session.metadata.get("title") and not session.metadata.get("parent_session_id"):
            auto_title = _first_user_auto_title(session.messages)
            if auto_title:
                session.metadata["title"] = auto_title
                session.metadata["title_auto"] = True

        channel, _, chat_id = session.key.partition(":")
        reserved = {
            "source": None,
            "channel": channel,
            "chat_id": chat_id,
            "title": None,
            "parent_session_id": None,
        }
        session.metadata = {**reserved, **session.metadata}

        metadata_line = json.dumps(
            {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
                "generation": session.generation,
            },
            ensure_ascii=False,
        )

        if len(session.messages) < session._persisted_count:
            tail = _interleave(session.messages, session.lifecycle, 0, 0)
            lines = [metadata_line]
            lines += [_record_line(r) for r in tail]
            atomic_replace(path, "".join(line + "\n" for line in lines))
        else:
            tail = _interleave(
                session.messages,
                session.lifecycle,
                session._persisted_count,
                session._persisted_lifecycle,
            )
            lines = [metadata_line]
            lines += [_record_line(r) for r in tail]
            locked_append(path, lines)

        session._persisted_count = len(session.messages)
        session._persisted_lifecycle = len(session.lifecycle)
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def archive(self, key: str) -> Path | None:
        """Move a session's file aside so the key starts empty next time.

        ``/new`` closes a conversation; what was said in it is not what the
        user asked to discard. The file is preserved under
        ``sessions/_closed/<channel>/<chat_id>.<stamp>.jsonl`` -- one level
        deeper than the two-level glob that lists live sessions, so a closed
        conversation stops appearing as one -- and the stamp (plus a counter
        for the same second) means closing one key repeatedly never overwrites
        an earlier close.

        The cache entry goes with it, so the next :meth:`get_or_create` builds
        a genuinely fresh session rather than handing back the closed one.
        Returns the archive path, or ``None`` when the key had no file on disk.
        """
        path = self._get_session_path(key)
        self.invalidate(key)
        if not path.exists():
            return None
        channel, _, chat_id = key.partition(":")
        dest_dir = ensure_dir(self.sessions_dir / "_closed" / safe_filename(channel))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = safe_filename(chat_id)
        dest = dest_dir / f"{stem}.{stamp}.jsonl"
        n = 1
        while dest.exists():
            dest = dest_dir / f"{stem}.{stamp}-{n}.jsonl"
            n += 1
        path.rename(dest)
        logger.info("session {} closed; preserved at {}", key, dest)
        return dest

    def delete(self, key: str) -> bool:
        """Remove the session file and invalidate the cache entry.

        Returns True only if a file was actually removed; False if no file
        existed or the removal failed. Deleting an unknown key is a safe no-op.
        """
        path = self._get_session_path(key)
        self.invalidate(key)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("session.delete: failed to remove file for {}", key)
                return False
            return True
        return False

    def exists(self, key: str) -> bool:
        """Return True if the session has a file on disk (lazy sessions don't)."""
        return self._get_session_path(key).exists()

    def peek(self, key: str) -> "Session | None":
        """Return the cached session if present; else load from disk without caching.

        Callers that need read-only access to a session should use this instead
        of get_or_create, which would cache a fresh empty session for unknown keys.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        return self._load(key)

    def fork(self, source_key: str, *, title: str | None = None) -> "Session | None":
        """Fork ``source_key`` at its head into a new diverging child session.

        Full-copy semantics: the child is minted with a fresh chat_id on the
        source's channel, a deep copy of the source's messages, and
        ``parent_session_id`` set to the source key (the reserved lineage slot).
        The child inherits ``last_consolidated`` (so its active-context window
        matches the source at the fork point). The child is persisted eagerly.

        Only a human-given source title is inherited (as ``<title> (fork)``);
        a title stamped with the ``title_auto`` metadata marker (written by
        ``save`` when it auto-names) is not carried over, so children of
        never-explicitly-titled sessions stay untitled. A title without the
        marker counts as human, which keeps sessions saved before the marker
        existed inheriting as before.

        Returns the persisted child, or None when the source does not exist or
        has zero messages (a fork of an empty session has no value).
        """
        source = self.peek(source_key)
        if source is None or not source.messages:
            return None

        channel = source_key.partition(":")[0]
        child = Session(
            key=f"{channel}:{new_chat_id()}",
            messages=copy.deepcopy(source.messages),
            last_consolidated=source.last_consolidated,
            lifecycle=copy.deepcopy(source.lifecycle),
        )
        if title is not None:
            child.metadata["title"] = title
        else:
            parent_title = (source.metadata or {}).get("title")
            if parent_title and not (source.metadata or {}).get("title_auto"):
                child.metadata["title"] = f"{parent_title} (fork)"
        # A fork continues its parent's conversation, so it continues on its
        # parent's model. The caller re-points the live binding, but that lives
        # in memory only -- without carrying the record too, the fork drops to
        # the default the first time it is resumed in a new process.
        for slot in ("model", "provider"):
            inherited = (source.metadata or {}).get(slot)
            if inherited:
                child.metadata[slot] = inherited
        child.metadata["parent_session_id"] = source_key
        self.save(child)
        return child

    def flush(self, key: str) -> bool:
        """Save the cached session iff it has unpersisted messages.

        Uses the _persisted_count dirty check. Returns False only when a save
        was attempted and failed (the failure is swallowed); True otherwise,
        including the no-op cases (key not cached / no new messages).
        """
        cached = self._cache.get(key)
        if cached is None:
            return True
        if cached.unpersisted:
            try:
                self.save(cached)
            except Exception:
                logger.warning("flush: failed to persist session {}", key)
                return False
        return True

    def records(self, key: str) -> list[dict[str, Any]]:
        """The journal of ``key``: messages and lifecycle records, in write order.

        Read-only, through :meth:`peek`, so asking what happened in a session
        never mints one. Empty for a key with no file and no cache entry.
        """
        session = self.peek(key)
        return session.records() if session is not None else []

    def list_sessions(self, channel: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by channel.

        Each entry carries: key, created_at, updated_at, path, message_count.
        Sorted by updated_at descending. Each file is read in a single pass.
        """
        sessions = []

        for path in self.sessions_dir.glob("*/*.jsonl"):
            if channel is not None and path.parent.name != channel:
                continue
            data, message_count = self._scan_file(path)
            if data is None:
                continue
            key = data.get("key") or self.key_from_path(path)
            sessions.append(
                {
                    "key": key,
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "path": str(path),
                    "message_count": message_count,
                    "metadata": data.get("metadata", {}),
                }
            )

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
