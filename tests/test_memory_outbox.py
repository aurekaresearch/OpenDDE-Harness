"""The extraction outbox: indexing a turn is not the turn's business.

Two promises used to be one. A turn committed its conversation and handed it to
the memory backend in the same breath, so a slow index was a slow turn, and a
refused write was a turn nobody would ever recall with nothing on disk saying
so. Here the turn appends one line and is done; a background drain owes the rest.

What these tests pin down is the honest part of the trade: a crash between the
store landing and the acknowledgement replays that one turn, and never loses it.

The queue is also an accumulator, so the unit offered to the backend is a batch of
one session's turns rather than a turn. A short conversation is held until it is
worth a write; these tests use a wedged or refusing backend and a forced drain,
which is what makes a write happen on demand. The cadence itself is pinned down in
``test_memory_host_backend.py``.
"""

import asyncio
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.memory_engine import dispatch
from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
from opendde_harness.memory_engine.outbox import MemoryOutbox
from opendde_harness.providers.base import ErrorClassification, LLMProvider, LLMResponse
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from opendde_harness.spine.events import Notice, NoticeKind

MODEL = "primary/model"
KEY = "cli:t"

#: What a user is allowed to feel. The wedged backend below takes 30s.
_TURN_BOUND_S = 2.0


class _Provider(LLMProvider):
    """Answers in one call. The turn is about what happens after it."""

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return LLMResponse(content="done", finish_reason="stop")

    async def chat_with_retry(self, **kwargs):
        return LLMResponse(content="done", finish_reason="stop")

    def classify_error(self, exc):
        return ErrorClassification("no_model")

    def get_default_model(self):
        return MODEL


class Backend:
    """Records every ``store`` it is offered and answers ``landed``."""

    def __init__(self, *, landed: bool = True, delay: float = 0.0) -> None:
        self.landed = landed
        self.delay = delay
        self.calls: list[tuple[str, int]] = []
        self.entered = asyncio.Event()

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        return []

    async def store(self, session_id: str, messages: list[dict[str, Any]], *, metadata=None) -> bool:
        self.calls.append((session_id, len(messages)))
        self.entered.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.landed

    async def feedback(self, signals: dict[str, Any]) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _request() -> TurnRequest:
    channel, _, chat = KEY.partition(":")
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel=channel, chat_id=chat, sender_id="user", chat_type=ChatType.DM),
        text="what did the assay say",
        conversation=KEY,
    )


def _loop(tmp_path: Path, backend: Any = None) -> AgentLoop:
    return AgentLoop(_Provider(), tmp_path, AgentLoopSettings(model=MODEL), backend=backend)


async def _quiesce(loop: AgentLoop) -> None:
    """Stop the background drain so a wedged backend cannot hold the test."""
    task = loop._outbox_task
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _outbox(tmp_path: Path) -> MemoryOutbox:
    return MemoryOutbox(tmp_path, MemoryStore(tmp_path))


# ── The turn's critical path ───────────────────────────────────────────


async def test_a_wedged_memory_service_does_not_delay_the_answer(tmp_path):
    """The answer is committed; indexing it is a separate promise. A store that
    never returns used to hold the turn for five seconds and then four more."""
    backend = Backend(delay=30.0)
    loop = _loop(tmp_path, backend)

    started = time.monotonic()
    reply = await loop._process_message(_request())
    elapsed = time.monotonic() - started

    assert reply.content == "done"
    assert elapsed < _TURN_BOUND_S, f"the turn waited {elapsed:.1f}s on the memory service"
    # And the work is not lost: the outbox is holding it, unwritten, because one
    # short turn is not yet worth a write.
    assert [e.session for e in loop.outbox.pending()] == [KEY]

    # When it is written, the wedge is the drain's problem and nobody else's: the
    # teardown budget expires, the drain is taken away, and the turn is still owed.
    await loop.drain_backend_stores(timeout=0.2)
    await asyncio.wait_for(backend.entered.wait(), timeout=_TURN_BOUND_S)
    assert [e.session for e in loop.outbox.pending()] == [KEY], "unacknowledged, because nothing answered"
    await _quiesce(loop)


async def test_ten_turns_in_a_row_never_wait_for_the_queue_to_move(tmp_path):
    """The old bound on outstanding writes made the fifth turn wait for the
    first. A queue on disk has no such cliff."""
    backend = Backend(delay=30.0)
    loop = _loop(tmp_path, backend)

    started = time.monotonic()
    for _ in range(10):
        await loop._process_message(_request())
    elapsed = time.monotonic() - started

    assert elapsed < _TURN_BOUND_S, f"ten turns spent {elapsed:.1f}s on the memory service"
    assert len(loop.outbox.pending()) == 10
    await _quiesce(loop)


async def test_the_queued_entry_carries_the_turn_id_session_and_generation(tmp_path):
    backend = Backend(delay=30.0)
    loop = _loop(tmp_path, backend)
    loop.sessions.get_or_create(KEY).generation = 3

    await loop._process_message(_request())

    entry = loop.outbox.pending()[0]
    assert entry.session == KEY
    assert entry.generation == 3
    assert entry.turn_id
    assert [m["role"] for m in entry.messages] == ["user", "assistant"]
    assert all(m["turn_id"] == entry.turn_id for m in entry.messages)
    await _quiesce(loop)


# ── Crash between store and acknowledgement ────────────────────────────


async def test_a_crash_between_the_store_and_the_ack_replays_the_turn_once(tmp_path, monkeypatch):
    """The invariant the split buys: at most one more delivery of a turn the
    backend may already hold, and never a turn silently dropped."""
    backend = Backend()
    loop = _loop(tmp_path, backend)
    await loop._process_message(_request())

    # The store lands; the record of it does not. That is the crash.
    def _no_cursor(*_args, **_kwargs):
        raise OSError("read-only file system")

    original_commit = MemoryStore.commit_extraction_cursor
    monkeypatch.setattr(MemoryStore, "commit_extraction_cursor", _no_cursor)
    await loop.drain_backend_stores(timeout=5)

    assert len(backend.calls) == 1
    assert len(loop.outbox.pending()) == 1, "unacknowledged, so still owed"

    # The next process picks the entry up and this time the cursor sticks.
    monkeypatch.setattr(MemoryStore, "commit_extraction_cursor", original_commit)
    restarted = _loop(tmp_path, backend)
    await restarted.drain_backend_stores(timeout=5)

    assert len(backend.calls) == 2, "stored at most twice, never lost"
    assert restarted.outbox.pending() == []
    assert restarted.outbox.deferred == 0


async def test_a_queued_turn_outlives_the_process_that_queued_it(tmp_path):
    """No backend wired when the turn ran, so nothing drained. The turn is still
    owed, and a later process with a backend indexes it."""
    queued = _loop(tmp_path)
    # Touching the lazy outbox is what creates the file; the workspace is the same
    # directory the restarted loop below reads.
    queued.outbox.append(turn_id="t1", session=KEY, generation=0, messages=[{"role": "user", "content": "hi"}])

    backend = Backend()
    restarted = _loop(tmp_path, backend)
    await restarted.drain_backend_stores(timeout=5)

    assert backend.calls == [(KEY, 1)]
    assert restarted.outbox.pending() == []


async def test_entries_are_offered_in_the_order_they_were_queued(tmp_path):
    backend = Backend()
    loop = _loop(tmp_path, backend)
    for n in range(3):
        loop.outbox.append(turn_id=f"t{n}", session=f"cli:{n}", generation=0, messages=[{"role": "user", "content": n}])

    await loop.drain_backend_stores(timeout=5)

    assert [session for session, _ in backend.calls] == ["cli:0", "cli:1", "cli:2"]


# ── Failure, bounded, and said out loud ────────────────────────────────


async def test_a_store_that_reports_it_did_not_land_is_retried_then_abandoned(tmp_path, monkeypatch):
    """``store`` returns a bool and the old caller discarded it. A refused write
    is retried a bounded number of times, then given up on so the queue behind it
    can move.

    The retried unit is the batch: two turns of one conversation are one write, so
    a refusal costs three attempts and not six. What is counted as abandoned is
    still turns, because that is what the user will not be able to recall."""
    monkeypatch.setattr(dispatch, "STORE_RETRY_BACKOFF_S", 0.0)
    backend = Backend(landed=False)
    loop = _loop(tmp_path, backend)
    loop.outbox.append(turn_id="t1", session=KEY, generation=0, messages=[{"role": "user", "content": "hi"}])
    loop.outbox.append(turn_id="t2", session=KEY, generation=0, messages=[{"role": "user", "content": "again"}])

    await loop.drain_backend_stores(timeout=10)

    assert backend.calls == [(KEY, 2)] * dispatch.STORE_MAX_ATTEMPTS, "one batch of two turns, offered three times"
    assert loop.outbox.pending() == [], "a failing batch does not hold the queue"
    assert loop.outbox.deferred == 2


async def test_an_abandoned_turn_is_told_to_the_user_once(tmp_path, monkeypatch):
    """It used to be a warning in a log at shutdown, which is the one moment the
    user is not reading it."""
    monkeypatch.setattr(dispatch, "STORE_RETRY_BACKOFF_S", 0.0)
    backend = Backend(landed=False)
    loop = _loop(tmp_path, backend)
    loop.outbox.append(turn_id="t1", session=KEY, generation=0, messages=[{"role": "user", "content": "hi"}])
    await loop.drain_backend_stores(timeout=10)

    events: list[Any] = []

    async def emit(event):
        events.append(event)

    def drain():
        return []

    await loop._run_turn(_request(), emit, drain, stream=False)
    notices = [e for e in events if isinstance(e, Notice) and e.kind is NoticeKind.DELIVERY_FAILED]
    assert len(notices) == 1
    assert "memory service" in (notices[0].detail or "")

    events.clear()
    await loop._run_turn(_request(), emit, drain, stream=False)
    assert [e for e in events if isinstance(e, Notice) and e.kind is NoticeKind.DELIVERY_FAILED] == []


async def test_a_backend_that_raises_is_treated_as_a_write_that_did_not_land(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "STORE_RETRY_BACKOFF_S", 0.0)

    class Exploding(Backend):
        async def store(self, session_id, messages, *, metadata=None):
            self.calls.append((session_id, len(messages)))
            raise RuntimeError("memory service is down")

    backend = Exploding()
    loop = _loop(tmp_path, backend)
    loop.outbox.append(turn_id="t1", session=KEY, generation=0, messages=[{"role": "user", "content": "hi"}])

    await loop.drain_backend_stores(timeout=10)

    assert len(backend.calls) == dispatch.STORE_MAX_ATTEMPTS
    assert loop.outbox.deferred == 1


async def test_a_backend_that_answers_nothing_is_taken_at_its_word(tmp_path):
    """The Protocol returns a bool; an adapter returning None does not track
    landing, which is not the same as failing."""

    class Quiet(Backend):
        async def store(self, session_id, messages, *, metadata=None):
            self.calls.append((session_id, len(messages)))
            return None

    backend = Quiet()
    loop = _loop(tmp_path, backend)
    loop.outbox.append(turn_id="t1", session=KEY, generation=0, messages=[{"role": "user", "content": "hi"}])

    await loop.drain_backend_stores(timeout=5)

    assert len(backend.calls) == 1
    assert loop.outbox.pending() == []


# ── Bounds ─────────────────────────────────────────────────────────────


def test_the_queue_is_bounded_and_abandons_its_oldest_entries(tmp_path, monkeypatch):
    """A service this far behind is not one to keep feeding. The bound is on the
    queue, not on the turn."""
    outbox = _outbox(tmp_path)
    monkeypatch.setattr(type(outbox), "capacity", 3)

    for n in range(5):
        outbox.append(turn_id=f"t{n}", session=KEY, generation=0, messages=[{"role": "user", "content": n}])

    pending = outbox.pending()
    assert [e.turn_id for e in pending] == ["t2", "t3", "t4"]
    assert outbox.deferred == 2


def test_the_file_is_rewritten_to_its_pending_tail_once_the_acked_prefix_grows(tmp_path, monkeypatch):
    """A long-lived workspace does not accumulate every turn it ever indexed."""
    outbox = _outbox(tmp_path)
    monkeypatch.setattr(type(outbox), "capacity", 2)

    for n in range(6):
        outbox.append(turn_id=f"t{n}", session=KEY, generation=0, messages=[{"role": "user", "content": n}])
        for entry in outbox.pending():
            outbox.ack(entry)
    outbox.append(turn_id="last", session=KEY, generation=0, messages=[{"role": "user", "content": "x"}])

    lines = [line for line in outbox.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) <= 3, "the acknowledged prefix is not kept forever"
    assert [e.turn_id for e in outbox.pending()] == ["last"]


def test_an_empty_turn_is_not_queued(tmp_path):
    outbox = _outbox(tmp_path)
    outbox.append(turn_id="t1", session=KEY, generation=0, messages=[])
    assert not outbox.path.exists()


def test_a_partial_trailing_line_from_a_crashed_append_is_skipped(tmp_path):
    """The turn it described is in the session log either way."""
    outbox = _outbox(tmp_path)
    outbox.append(turn_id="t1", session=KEY, generation=0, messages=[{"role": "user", "content": "hi"}])
    with outbox.path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 2, "turn_id": "t2", "sess')

    assert [e.turn_id for e in outbox.pending()] == ["t1"]
