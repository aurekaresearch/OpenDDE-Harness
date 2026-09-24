"""The extraction outbox: completed turns waiting for the memory backend.

A turn used to reach :meth:`MemoryBackend.store` from the turn itself, which
made a slow index a slow turn, and made a dropped write a turn the user would
never be able to recall with nothing on disk saying so. Committing the turn and
indexing it are two different promises and only the first belongs on the turn's
critical path.

So a completed turn is appended here -- one small file write -- and the turn is
done. A background drain hands entries to the backend in order and records how
far it got. Two files, and the split is the point:

* ``outbox.jsonl`` is append-only and ordered. It holds the work.
* the memory store's versioned state holds the acknowledgement cursor: the
  sequence number and turn id of the last entry the backend confirmed.

A crash between the store landing and the cursor moving replays that one entry.
That is the honest trade: at-most-once-more delivery of a turn the backend may
already hold, rather than a turn silently never indexed. Backends that chunk and
deduplicate are the ones this contract is written for.

Bounded in both directions. Past :attr:`MemoryOutbox.capacity` pending entries
the oldest are abandoned -- a service that far behind is not one to keep feeding
-- and each abandonment is counted so the host can say so once, out loud,
instead of logging it at shutdown. The file is rewritten to its pending tail
once the acknowledged prefix outgrows the same bound, so a long-lived workspace
does not accumulate every turn it ever indexed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
from opendde_harness.utils.atomic_io import atomic_replace, locked_append
from opendde_harness.utils.helpers import ensure_dir


@dataclass(frozen=True)
class OutboxEntry:
    """One completed turn awaiting durable extraction."""

    seq: int
    turn_id: str
    session: str
    generation: int
    messages: list[dict[str, Any]]
    queued_at: str
    closing: bool = False
    """This entry is a conversation closing (``/new``), not another turn in one.

    Recorded on the entry rather than held by whoever queued it: the reset it
    belongs to may be the last thing the process does, and the accumulator has to
    still know on the next run that there is nothing more coming for that session.
    """

    def as_line(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "turn_id": self.turn_id,
                "session": self.session,
                "generation": self.generation,
                "queued_at": self.queued_at,
                "closing": self.closing,
                "messages": self.messages,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_line(cls, data: dict[str, Any]) -> "OutboxEntry | None":
        try:
            return cls(
                seq=int(data["seq"]),
                turn_id=str(data["turn_id"]),
                session=str(data["session"]),
                generation=int(data.get("generation", 0) or 0),
                messages=list(data.get("messages") or []),
                queued_at=str(data.get("queued_at", "")),
                closing=bool(data.get("closing", False)),
            )
        except (KeyError, TypeError, ValueError):
            return None


class MemoryOutbox:
    """An ordered, bounded, append-only queue of turns to extract."""

    #: Pending entries past which the oldest are abandoned rather than queued
    #: behind a service that is not keeping up.
    capacity: int = 64

    def __init__(
        self,
        workspace: Path,
        store: MemoryStore,
        *,
        now_fn: Callable[[], datetime] | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self.path = ensure_dir(state_dir if state_dir is not None else store.state_dir) / "outbox.jsonl"
        self.store = store
        self._now_fn = now_fn or datetime.now
        #: Turns abandoned without being indexed, this process. Read once by the
        #: host to tell the user, and cleared by it.
        self.deferred = 0
        # The last parse, keyed by the file's size and mtime. Every write to this
        # file changes both -- an append grows it, a compaction shrinks it -- so a
        # stat that matches means the parse still stands. Without this, a turn
        # queued behind a wedged service re-parsed the whole queue, bodies and
        # all, on the way out.
        self._cache: tuple[tuple[int, int], list[OutboxEntry]] | None = None

    # ── Writing ────────────────────────────────────────────────────────

    def append(
        self,
        *,
        turn_id: str,
        session: str,
        generation: int,
        messages: list[dict[str, Any]],
        closing: bool = False,
    ) -> None:
        """Queue one completed turn. Returns as soon as the line is on disk."""
        if not messages:
            return
        entries = self._read()
        cursor = self._cursor()
        pending = [e for e in entries if e.seq > cursor]
        seq = (entries[-1].seq if entries else cursor) + 1
        entry = OutboxEntry(
            seq=seq,
            turn_id=turn_id,
            session=session,
            generation=generation,
            messages=messages,
            queued_at=self._now_fn().isoformat(timespec="seconds"),
            closing=closing,
        )
        locked_append(self.path, [entry.as_line()])
        self._cache = None

        if len(pending) + 1 > self.capacity:
            self._abandon(pending[: len(pending) + 1 - self.capacity])
        elif len(entries) - len(pending) > self.capacity:
            self._compact([*pending, entry])

    def _abandon(self, entries: list[OutboxEntry]) -> None:
        """Move the cursor past entries that will never be indexed."""
        if not entries:
            return
        last = entries[-1]
        self.deferred += len(entries)
        logger.warning(
            "memory outbox: abandoning {} unindexed turn(s) up to {}; the memory service is not keeping up",
            len(entries),
            last.turn_id,
        )
        self._commit(last)

    def _compact(self, pending: list[OutboxEntry]) -> None:
        """Rewrite the file as its pending tail, keeping sequence numbers."""
        atomic_replace(self.path, "".join(e.as_line() + "\n" for e in pending))
        self._cache = None

    # ── Reading and acknowledging ──────────────────────────────────────

    def pending(self) -> list[OutboxEntry]:
        """Entries the backend has not confirmed, oldest first."""
        cursor = self._cursor()
        return [e for e in self._read() if e.seq > cursor]

    def ack(self, entry: OutboxEntry) -> bool:
        """Record that the backend confirmed ``entry``. False if it did not stick.

        Ordered drain, so one cursor is the whole acknowledgement: everything at
        or below it landed. Written through the memory store's versioned state,
        under its lock -- the same transaction that guards the profile.

        A cursor that cannot be written is the crash this design is built to
        survive: the store landed and the record of it did not, so the entry
        replays. The caller stops draining rather than offering the same entry
        again in a loop -- the backend is fine, the workspace is not.
        """
        return self._commit(entry)

    def _commit(self, entry: OutboxEntry) -> bool:
        try:
            self.store.commit_extraction_cursor(seq=entry.seq, turn_id=entry.turn_id)
        except OSError:
            logger.exception("memory outbox: could not move the cursor past {}", entry.turn_id)
            return False
        return True

    def _cursor(self) -> int:
        return self.store.read_extraction_cursor().get("seq", 0)

    def _read(self) -> list[OutboxEntry]:
        try:
            stat = self.path.stat()
        except OSError:
            return []
        stamp = (stat.st_size, stat.st_mtime_ns)
        if self._cache is not None and self._cache[0] == stamp:
            return self._cache[1]
        out: list[OutboxEntry] = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("memory outbox: could not read {}", self.path)
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # A partial trailing line from a crashed append. The turn it
                # described is in the session log either way.
                logger.debug("memory outbox: skipping an undecodable line")
                continue
            entry = OutboxEntry.from_line(data) if isinstance(data, dict) else None
            if entry is not None:
                out.append(entry)
        out.sort(key=lambda e: e.seq)
        self._cache = (stamp, out)
        return out


__all__ = ["MemoryOutbox", "OutboxEntry"]
