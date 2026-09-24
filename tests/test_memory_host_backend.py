"""The default memory writer: what a finished turn leaves in user.md and episodes.md.

The self-evolution loop is host-owned again, on the extraction outbox rather than
inside the turn. What has to hold:

* a turn is annotated into ``episodes.md`` with its tags, and the lines the
  vocabulary declares invalid never get there;
* a tag that has gained enough episodes rewrites **its own** profile section and
  leaves every other section byte for byte as it was;
* a rewrite computed from a profile that changed underneath it is refused, and
  the tag keeps its offset so the evidence is folded in next time instead of lost;
* state the store cannot read is set aside rather than reinterpreted;
* foresight is not asked for unless it is turned on;
* none of it is on the turn's critical path.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from opendde_harness.memory_engine import LifecycleContractTests, MemoryBackendContractTests, dispatch
from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
from opendde_harness.memory_engine.dispatch import ExtractionDispatcher
from opendde_harness.memory_engine.host_backend import REFRESH_HOT_TAG_THRESHOLD, HostMarkdownBackend
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import LLMResponse, ToolCallRequest
from opendde_harness.providers.binding import ModelBinding
from opendde_harness.utils.helpers import estimate_message_tokens

MODEL = "faux/annotator"

_PROFILE = """# Long-term Memory

## Identity

- **Role**: protein engineer [src: episodes.md @ 2026-01-02 09:00]

## Notes

- ships on Fridays [src: episodes.md @ 2026-01-02 09:00]
"""


class Model:
    """A provider that answers each call from a queued tool-call payload.

    Records the tool schema it was offered, so a test can assert what the model
    was asked for rather than only what it answered.
    """

    def __init__(self, answers: list[dict[str, Any] | None]) -> None:
        self._answers = list(answers)
        self.tools_offered: list[list[dict]] = []
        self.prompts: list[str] = []
        self.before_answer = None
        self.fail_next = False

    async def chat_with_retry(self, *, messages, tools=None, model=None, tool_choice=None, **_kw) -> LLMResponse:
        self.tools_offered.append(tools or [])
        self.prompts.append(messages[-1]["content"])
        if self.before_answer is not None:
            self.before_answer()
        if self.fail_next:
            return LLMResponse(content="the service is down", finish_reason="error")
        payload = self._answers.pop(0) if self._answers else None
        if payload is None:
            return LLMResponse(content="I would rather describe it in prose.", finish_reason="stop")
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="call_1", name="tool", arguments=payload)],
        )

    def get_default_model(self) -> str:
        return MODEL


def backend_for(tmp_path, answers, *, foresight: bool = False) -> tuple[HostMarkdownBackend, MemoryStore, Model]:
    store = MemoryStore(tmp_path)
    model = Model(answers)
    backend = HostMarkdownBackend(
        store,
        binding=ModelBinding(model, MODEL),
        enable_foresight=foresight,
    )
    return backend, store, model


def a_turn(said: str = "the CRLF2 assay is flaky at 37C", *, at: int = 1767258000000) -> list[dict[str, Any]]:
    """One turn, worth about 15 estimated tokens. Two turns that said different
    things are two turns: the batch folds copies of one message together, so a
    reply reused verbatim at the same millisecond would read as the same turn."""
    return [
        msg.user_message(said, timestamp=at),
        msg.assistant_message(f"re-ran {said} three times", timestamp=at + 60_000),
    ]


def turns(n: int) -> list[list[dict[str, Any]]]:
    """``n`` distinct turns, two minutes apart."""
    return [a_turn(f"turn number {i}", at=1767258000000 + i * 120_000) for i in range(n)]


def weight(batch: list[list[dict[str, Any]]]) -> int:
    """What the accumulator makes of these turns, by the estimator it uses.

    Read from the estimator rather than written down, so a test says "three turns
    reach the threshold and two do not" and keeps saying it if the estimator is
    ever retuned.
    """
    return sum(estimate_message_tokens(message) for turn in batch for message in turn)


def annotate_only(*episodes: str) -> dict[str, Any]:
    return {"episode_summary": list(episodes)}


def tool_names(offered: list[dict]) -> list[str]:
    return [entry["function"]["name"] for entry in offered]


def slots(offered: list[dict]) -> set[str]:
    return set(offered[0]["function"]["parameters"]["properties"])


# ── Annotation ─────────────────────────────────────────────────────────


async def test_a_finished_turn_is_annotated_into_the_episodic_log_with_its_tags(tmp_path):
    """The light path, and the one that runs every turn. The line is kept as the
    annotator wrote it -- stamp, summary, tags -- because the tags are the index
    the profile refresh reads back."""
    episode = "[2026-01-01 12:00] CRLF2 assay flaky at 37C; three reruns disagree #project-crlf2-assay #bug"
    backend, store, model = backend_for(tmp_path, [annotate_only(episode)])

    assert await backend.store("tui:main", a_turn()) is True

    assert store.history_file.read_text(encoding="utf-8").splitlines() == [episode]
    assert tool_names(model.tools_offered[0]) == ["annotate_conversation"]
    assert "the CRLF2 assay is flaky at 37C" in model.prompts[0], "the annotator reads what was actually said"


async def test_an_episode_that_only_says_how_the_user_was_talking_is_not_indexed(tmp_path):
    """``#question`` / ``#habit`` / ``#answer`` describe the interaction, not the
    work. The prompt forbids them standing alone; the model emits them anyway, and
    a tag with no content behind it heats up and rewrites a profile section from
    nothing."""
    good = "[2026-01-01 12:00] switched the buffer to HEPES #project-crlf2-assay #decision"
    backend, store, _ = backend_for(
        tmp_path,
        [annotate_only("[2026-01-01 12:05] user asked how to run it #question", good)],
    )

    await backend.store("tui:main", a_turn())

    assert store.history_file.read_text(encoding="utf-8").splitlines() == [good]


async def test_a_call_that_never_happened_leaves_the_turn_owed(tmp_path):
    """The outbox retries a ``False``. A failed call must therefore report one,
    rather than acknowledge a turn that was never indexed."""
    backend, store, model = backend_for(tmp_path, [])
    model.fail_next = True

    assert await backend.store("tui:main", a_turn()) is False
    assert not store.history_file.exists() or store.history_file.read_text(encoding="utf-8") == ""


async def test_a_model_that_answers_in_prose_is_not_asked_the_same_thing_twice(tmp_path):
    """There is no wire spelling for ``tool_choice: required`` on the model
    service, so the call is asked for by the prompt and a model can answer
    without making it. That is not an outage: offering the same prompt again buys
    the same non-answer, and reports a turn lost to a service that was fine."""
    backend, store, _ = backend_for(tmp_path, [None])

    assert await backend.store("tui:main", a_turn()) is True, "acknowledged, not retried"
    assert not store.history_file.exists() or store.history_file.read_text(encoding="utf-8") == ""


# ── The profile, one hot section at a time ─────────────────────────────


def heat_a_tag(store: MemoryStore, tag: str, n: int = REFRESH_HOT_TAG_THRESHOLD) -> None:
    store.append_episodes(
        [f"[2026-01-0{i + 1} 09:00] step {i} of the assay rebuild #{tag} #decision" for i in range(n)]
    )


async def test_a_hot_tag_rewrites_its_own_section_and_leaves_the_rest_byte_for_byte(tmp_path):
    """The heavy path. The splice replaces one H2 and nothing else -- the profile
    is mostly things this tag says nothing about, and a whole-file rewrite is how
    the released version lost them."""
    body = "- **Status**: rebuilt at pH 7.4 [src: episodes.md @ 2026-01-05 09:00]"
    backend, store, model = backend_for(
        tmp_path,
        [
            annotate_only("[2026-01-06 09:00] rebuilt the assay at pH 7.4 #project-crlf2-assay #decision"),
            {"section_heading": "## Projects", "section_body": body},
        ],
    )
    store.write_long_term(_PROFILE)
    heat_a_tag(store, "project-crlf2-assay", REFRESH_HOT_TAG_THRESHOLD - 1)

    await backend.store("tui:main", a_turn())

    profile = store.read_long_term()
    assert "## Projects" in profile and body in profile
    assert "- **Role**: protein engineer [src: episodes.md @ 2026-01-02 09:00]" in profile
    assert "- ships on Fridays [src: episodes.md @ 2026-01-02 09:00]" in profile
    assert tool_names(model.tools_offered[1]) == ["refresh_profile_section"]
    assert store.read_refresh_offsets()["project-crlf2-assay"] == REFRESH_HOT_TAG_THRESHOLD


async def test_a_tag_below_the_threshold_leaves_the_profile_untouched(tmp_path):
    """Episodes accumulate; the profile waits for evidence. One mention of a
    project is not a fact about the user."""
    backend, store, model = backend_for(
        tmp_path,
        [annotate_only("[2026-01-06 09:00] read the old assay notes #project-crlf2-assay #question")],
    )
    store.write_long_term(_PROFILE)

    await backend.store("tui:main", a_turn())

    assert store.read_long_term() == _PROFILE
    assert len(model.tools_offered) == 1, "no section rewrite was asked for"


async def test_a_profile_bullet_that_cites_no_episode_is_not_written(tmp_path):
    """Every profile bullet carries the episode it came from. A bullet that cites
    nothing is the shape a speculation takes, and the file is supposed to be
    checkable against the log."""
    backend, store, _ = backend_for(
        tmp_path,
        [
            annotate_only("[2026-01-06 09:00] rebuilt the assay #project-crlf2-assay #decision"),
            {
                "section_heading": "## Projects",
                "section_body": (
                    "- **Status**: rebuilt [src: episodes.md @ 2026-01-05 09:00]\n"
                    "- probably wants to publish this in the spring"
                ),
            },
        ],
    )
    heat_a_tag(store, "project-crlf2-assay", REFRESH_HOT_TAG_THRESHOLD - 1)

    await backend.store("tui:main", a_turn())

    profile = store.read_long_term()
    assert "- **Status**: rebuilt [src: episodes.md @ 2026-01-05 09:00]" in profile
    assert "publish this in the spring" not in profile


async def test_a_profile_that_changed_under_the_rewrite_keeps_the_other_writer(tmp_path):
    """Compare-and-set. The rewrite was computed from a profile that no longer
    exists, so it is refused: splicing it would drop whatever the other writer
    put there. The tag keeps its offset, so the evidence is folded in next round
    rather than lost."""
    concurrent = _PROFILE + "\n## Habits\n\n- runs the assay on Mondays [src: episodes.md @ 2026-01-04 08:00]\n"
    backend, store, model = backend_for(
        tmp_path,
        [
            annotate_only("[2026-01-06 09:00] rebuilt the assay #project-crlf2-assay #decision"),
            {
                "section_heading": "## Projects",
                "section_body": "- **Status**: rebuilt [src: episodes.md @ 2026-01-05 09:00]",
            },
        ],
    )
    store.write_long_term(_PROFILE)
    heat_a_tag(store, "project-crlf2-assay", REFRESH_HOT_TAG_THRESHOLD - 1)

    calls = {"n": 0}

    def other_writer() -> None:
        # Fires while the second call (the section rewrite) is in flight.
        calls["n"] += 1
        if calls["n"] == 2:
            store.write_long_term(concurrent)

    model.before_answer = other_writer

    await backend.store("tui:main", a_turn())

    assert store.read_long_term() == concurrent, "the other writer's profile stands"
    assert store.read_refresh_offsets() == {}, "and the tag is still owed a refresh"


async def test_state_the_store_cannot_read_is_set_aside_and_the_tags_read_as_owed(tmp_path):
    """An unreadable cursor file and an absent one are different things. Treating
    the tag as never refreshed folds its evidence in again; resetting the file in
    place would have destroyed the only evidence of what went wrong."""
    store = MemoryStore(tmp_path)
    store.commit_refresh_offset("project-crlf2-assay", 7)
    store.state_file.write_text("{not json at all", encoding="utf-8")

    assert store.read_refresh_offsets() == {}
    aside = sorted(p.name for p in store.state_file.parent.glob("state.json.damaged-*"))
    assert len(aside) == 1, f"the damaged state was not preserved: {aside}"


# ── Foresight ──────────────────────────────────────────────────────────


async def test_foresight_is_not_even_asked_for_unless_it_is_turned_on(tmp_path):
    """Off by default. With the flag off the annotation tool has one slot, so the
    model is not asked to predict anything and nothing can be written from a
    guess."""
    backend, store, model = backend_for(
        tmp_path,
        [annotate_only("[2026-01-06 09:00] rebuilt the assay #project-crlf2-assay #decision")],
    )

    await backend.store("tui:main", a_turn())

    assert slots(model.tools_offered[0]) == {"episode_summary"}
    assert "## Foresight" not in store.read_long_term()


async def test_foresight_turned_on_is_asked_for_and_kept_in_the_profile(tmp_path):
    """And with it on, the slot is there and the prediction lands in its own
    section, carrying the episode that triggered it."""
    backend, store, model = backend_for(
        tmp_path,
        [
            {
                "episode_summary": ["[2026-01-06 09:00] rebuilt the assay #project-crlf2-assay #decision"],
                "foresight_hint": [
                    {
                        "prediction": "User will re-run the CRLF2 assay after the buffer arrives",
                        "window": "2-3 days",
                        "confidence": "medium",
                        "src_ts": "2026-01-06 09:00",
                    }
                ],
            }
        ],
        foresight=True,
    )

    await backend.store("tui:main", a_turn())

    assert slots(model.tools_offered[0]) == {"episode_summary", "foresight_hint"}
    profile = store.read_long_term()
    assert "## Foresight" in profile
    assert "confidence: medium" in profile and "src: episodes.md @ 2026-01-06 09:00" in profile


async def test_the_same_prediction_in_other_words_is_not_kept_twice(tmp_path):
    """The exact ``(prediction, src_ts)`` key lets a reworded re-emission through;
    the section fills up with one claim said five ways."""
    store = MemoryStore(tmp_path)
    first = {
        "prediction": "User runs the CRLF2 assay every Saturday morning",
        "window": "recurring weekly",
        "confidence": "high",
        "src_ts": "2026-01-06 09:00",
    }
    assert store.append_foresight([first]) == 1
    reworded = {**first, "prediction": "User runs the CRLF2 assay Saturday mornings", "src_ts": "2026-01-13 09:00"}

    assert store.append_foresight([reworded]) == 0
    assert store.read_long_term().count("- User runs the CRLF2 assay") == 1


# ── Recall ─────────────────────────────────────────────────────────────


async def test_recall_answers_with_the_sections_the_query_matches(tmp_path):
    """The read side of the same file. The block is what the system prompt
    carries, so it comes back rendered rather than as fields to reassemble."""
    backend, store, _ = backend_for(tmp_path, [])
    store.write_long_term(_PROFILE)

    hits = await backend.recall("what is my role", user_id="default", top_k=5)

    assert len(hits) == 1
    assert "## Identity" in hits[0].text and "protein engineer" in hits[0].text
    assert hits[0].text.startswith("## Long-term Memory")


async def test_recall_has_nothing_for_the_agent_track(tmp_path):
    """A markdown profile holds what is true about the user. There is no case
    library behind it, so the skill-source call answers empty instead of
    returning the profile under the wrong track."""
    backend, store, _ = backend_for(tmp_path, [])
    store.write_long_term(_PROFILE)

    assert await backend.recall("anything", agent_id="default", top_k=5) == []
    assert await backend.recall("anything", user_id="u", agent_id="a", top_k=5) == []


async def test_recall_on_a_workspace_with_no_profile_is_empty(tmp_path):
    backend, _store, _ = backend_for(tmp_path, [])

    assert await backend.recall("anything", user_id="default", top_k=5) == []


# ── Where the work sits relative to the answer ─────────────────────────


async def test_the_outbox_drives_the_writer_and_the_turn_does_not_wait_for_it(tmp_path):
    """The dispatcher's contract, with the host writer behind it. Queuing is one
    appended line and no await; the annotation happens in the drain, and what the
    user typed is in ``episodes.md`` only after it."""
    episode = "[2026-01-06 09:00] rebuilt the assay at pH 7.4 #project-crlf2-assay #decision"
    backend, store, model = backend_for(tmp_path, [annotate_only(episode)])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    dispatcher.queue("tui:main", a_turn(), turn_id="t1", generation=0)

    assert [entry.turn_id for entry in dispatcher.outbox.pending()] == ["t1"], "queued, not yet written"
    assert not store.history_file.exists(), "and nothing was written while the turn was answering"
    assert model.prompts == [], "nor was a model asked anything on the turn's way out"

    await dispatcher.drain(timeout=5)

    assert store.history_file.read_text(encoding="utf-8").splitlines() == [episode]
    assert dispatcher.outbox.pending() == [], "the queue is empty rather than holding a turn it already wrote"
    assert store.read_extraction_cursor()["turn_id"] == "t1"


# ── Cadence: a turn is not a unit of memory ────────────────────────────


async def held(dispatcher: ExtractionDispatcher, *, budget: float = 2.0) -> None:
    """Let the drain run to its own end, or cancel it where it is holding.

    A drain that writes returns when the queue empties. One that is accumulating
    waits out its idle window, which is minutes, so the test takes it away rather
    than leaving a task alive after the loop it was started on is gone.
    """
    task = dispatcher.task
    if task is None or task.done():
        return
    try:
        await asyncio.wait_for(task, timeout=budget)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


async def test_turns_under_the_threshold_are_accumulated_and_cost_nothing(tmp_path):
    """The point of the batch. Three short turns are three lines in the queue and
    no model call at all: annotating each turn as it lands is a call per turn,
    which is the cost the released consolidator did not pay."""
    backend, store, model = backend_for(tmp_path, [annotate_only("[2026-01-06 09:00] x #project-x #decision")])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    for n, turn in enumerate(turns(3)):
        dispatcher.queue("tui:main", turn, turn_id=f"t{n}", generation=0)
    await held(dispatcher)

    assert model.prompts == [], "nothing was annotated"
    assert not store.history_file.exists(), "and nothing was written"
    assert len(dispatcher.outbox.pending()) == 3, "the turns are still owed, not lost"


async def test_crossing_the_threshold_annotates_the_accumulated_turns_at_once(tmp_path, monkeypatch):
    """One call, one batch, every queued turn in it. The chunk the annotator reads
    is a stretch of conversation, which is what its prompt is written for."""
    three = turns(3)
    assert weight(three[:2]) < weight(three), "the third turn is what crosses it"
    monkeypatch.setattr(dispatch, "BATCH_TOKEN_THRESHOLD", weight(three))
    episode = "[2026-01-06 09:00] three turns of assay work #project-crlf2-assay #decision"
    backend, store, model = backend_for(tmp_path, [annotate_only(episode)])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    for n, turn in enumerate(three):
        dispatcher.queue("tui:main", turn, turn_id=f"t{n}", generation=0)
    await held(dispatcher)

    assert len(model.prompts) == 1, "one annotation, not one per turn"
    for n in range(3):
        assert f"turn number {n}" in model.prompts[0], "and it covers every turn it was holding"
    assert store.history_file.read_text(encoding="utf-8").splitlines() == [episode]
    assert dispatcher.outbox.pending() == []


async def test_a_closed_conversation_writes_what_is_still_held(tmp_path):
    """``/new`` is the one moment there is certainly nothing more to add, so the
    remainder is written however little it weighs. Otherwise a conversation the
    user deliberately ended would sit in the queue waiting for turns that will
    never come."""
    episode = "[2026-01-06 09:00] closed the assay conversation #project-crlf2-assay #decision"
    backend, store, model = backend_for(tmp_path, [annotate_only(episode)])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    dispatcher.queue("tui:main", a_turn("first thing"), turn_id="t0", generation=0)
    await held(dispatcher)
    assert model.prompts == [], "held, as a short turn should be"

    dispatcher.queue("tui:main", a_turn("first thing"), turn_id="close", generation=0, closing=True)
    await held(dispatcher)

    assert len(model.prompts) == 1
    assert store.history_file.read_text(encoding="utf-8").splitlines() == [episode]
    assert dispatcher.outbox.pending() == []


async def test_the_closing_hand_off_does_not_annotate_the_same_turn_twice(tmp_path):
    """``/new`` queues the whole closed session, which overlaps every per-turn
    entry already waiting for it. The batch is the conversation, not the sum of
    the copies of it."""
    backend, store, model = backend_for(tmp_path, [annotate_only("[2026-01-06 09:00] x #project-x #decision")])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)
    first, second = a_turn("the first thing said"), a_turn("the second thing said")

    dispatcher.queue("tui:main", first, turn_id="t0", generation=0)
    dispatcher.queue("tui:main", second, turn_id="t1", generation=0)
    dispatcher.queue("tui:main", [*first, *second], turn_id="close", generation=0, closing=True)
    await held(dispatcher)

    assert len(model.prompts) == 1
    assert model.prompts[0].count("USER: the first thing said") == 1
    assert model.prompts[0].count("USER: the second thing said") == 1


async def test_a_turn_that_has_sat_long_enough_is_written_without_being_asked(tmp_path, monkeypatch):
    """The bound on how long a short conversation stays unremembered. Without it a
    user who says one thing and walks away has it in the queue until they come
    back, which may be never."""
    monkeypatch.setattr(dispatch, "BATCH_IDLE_S", 0.0)
    episode = "[2026-01-06 09:00] said one thing #project-crlf2-assay #question #decision"
    backend, store, model = backend_for(tmp_path, [annotate_only(episode)])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    dispatcher.queue("tui:main", a_turn("just the one thing"), turn_id="t0", generation=0)
    await held(dispatcher)

    assert len(model.prompts) == 1, "the idle window closed and the batch was written"
    assert store.history_file.read_text(encoding="utf-8").splitlines() == [episode]


async def test_two_conversations_are_annotated_as_two(tmp_path, monkeypatch):
    """One batch is one conversation. Folding two sessions into one slice would
    have the annotator summarize a conversation that never happened."""
    conversation = [a_turn("assay 0", at=1767258000000), a_turn("assay 1", at=1767258120000)]
    monkeypatch.setattr(dispatch, "BATCH_TOKEN_THRESHOLD", weight(conversation))
    backend, store, model = backend_for(
        tmp_path,
        [
            annotate_only("[2026-01-06 09:00] the assay #project-crlf2-assay #decision"),
            annotate_only("[2026-01-06 09:05] the paper #project-paper #decision"),
        ],
    )
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    for n in range(2):
        at = 1767258000000 + n * 120_000
        dispatcher.queue("tui:assay", a_turn(f"assay {n}", at=at), turn_id=f"a{n}", generation=0)
        dispatcher.queue("tui:paper", a_turn(f"paper {n}", at=at), turn_id=f"p{n}", generation=0)
    await held(dispatcher)

    assert len(model.prompts) == 2, "one call per conversation"
    assay = next(p for p in model.prompts if "assay 0" in p)
    assert "paper 0" not in assay, "and neither call was shown the other conversation"
    assert len(store.history_file.read_text(encoding="utf-8").splitlines()) == 2


async def test_a_writer_that_keeps_failing_is_abandoned_rather_than_retried_forever(tmp_path):
    """The bound the outbox puts on the promise. Three refusals is a writer that
    is failing, not busy, and holding the queue behind it would stop every later
    turn from being indexed."""
    backend, store, model = backend_for(tmp_path, [])
    model.fail_next = True
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=backend)

    dispatcher.queue("tui:main", a_turn(), turn_id="t1", generation=0)
    await dispatcher.drain(timeout=20)

    assert dispatcher.outbox.deferred == 1
    assert dispatcher.deferral_notice() is not None, "the user is told, once"


# ── Config ─────────────────────────────────────────────────────────────


def owner_for(config, provider):
    """The memory owner ``build_agent_loop`` would give this config.

    Read off the constructor rather than reimplemented, so the test is asking the
    one place that decides.
    """
    import opendde_harness.agent.loop.main as loop_main
    from opendde_harness.agent.loop.factory import build_agent_loop

    built: dict[str, Any] = {}

    class Loop:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.update(kwargs)

    original = loop_main.AgentLoop
    loop_main.AgentLoop = Loop
    try:
        build_agent_loop(config, provider=provider)
    finally:
        loop_main.AgentLoop = original
    return built["backend"]


def test_the_host_writer_is_what_a_config_with_no_backend_named_gets(tmp_path):
    """The default. ``memory.backend`` unset used to mean nothing wrote memory at
    all; it now means the host writes it."""
    from opendde_harness.agent.loop.factory import AgentLoopSettings, build_host_memory_backend
    from opendde_harness.config.schema import Config

    config = Config.model_validate({"agents": {"defaults": {"workspace": str(tmp_path)}}, "memory": {"backend": None}})
    assert config.memory.foresight is False, "and predictions stay off until asked for"

    backend = build_host_memory_backend(
        config,
        provider=Model([]),
        settings=AgentLoopSettings.from_config(config),
    )

    assert isinstance(backend, HostMarkdownBackend)
    assert backend.owns_profile is True


async def test_memory_turned_off_writes_nothing_and_asks_nothing(tmp_path, caplog):
    """``memory.backend: "off"`` is the only way to have no owner. Unset means the
    host writer, because that is what a config whose author never chose looks
    like, so turning extraction off has to be a word the user wrote.

    And it is not looked for among the plugins. A boot-time warning that no plugin
    provides ``"off"`` would be a complaint about a config that is exactly right.
    """
    import logging

    from opendde_harness.cli._plugin_stack import maybe_build_memory_backend
    from opendde_harness.config.schema import Config

    config = Config.model_validate({"agents": {"defaults": {"workspace": str(tmp_path)}}, "memory": {"backend": "off"}})
    with caplog.at_level(logging.WARNING, logger="opendde_harness.cli._plugin_stack"):
        assert maybe_build_memory_backend(tmp_path, config) is None
    assert caplog.records == [], f"turning memory off was complained about: {[r.message for r in caplog.records]}"

    store = MemoryStore(tmp_path)
    model = Model([annotate_only("[2026-01-06 09:00] x #project-x #decision")])
    dispatcher = ExtractionDispatcher(tmp_path, store, backend=owner_for(config, model))

    dispatcher.queue("tui:main", a_turn(), turn_id="t0", generation=0)
    await held(dispatcher)

    assert model.prompts == [], "no model call was made for memory"
    assert not store.history_file.exists(), "no episode was written"
    assert store.read_long_term() == "", "and no profile either"
    assert dispatcher.outbox.pending() == [], "nothing is even owed: there is nobody to owe it to"


def test_a_config_that_names_a_plugin_backend_does_not_also_get_the_host_writer(tmp_path):
    """One owner. Two writers on ``user.md`` is the state this whole seam exists
    to prevent, so a named backend the caller did not resolve is not stood in for
    either: the config declared an owner."""
    from opendde_harness.config.schema import Config

    config = Config.model_validate(
        {"agents": {"defaults": {"workspace": str(tmp_path)}}, "memory": {"backend": "longterm"}}
    )

    assert owner_for(config, Model([])) is None


# ── The transcript the annotator reads ─────────────────────────────────


@pytest.mark.parametrize(
    "message, expected",
    [
        (msg.user_message("hello", timestamp=1767258000000), "USER"),
        (msg.system_message("the prompt prefix"), None),
    ],
)
def test_the_transcript_carries_the_conversation_and_not_the_prompt_prefix(message, expected):
    """The system prefix is the same on every turn and would be most of what the
    annotator reads. What was said is the part that varies."""
    from opendde_harness.memory_engine.host_backend import _format_messages

    rendered = _format_messages([message])
    if expected is None:
        assert rendered == ""
    else:
        assert expected in rendered and "2026-01" in rendered


def test_the_transcript_names_the_tools_a_turn_called():
    """A turn's tool calls are what it did; a summary of the turn without them
    describes a conversation that did not happen."""
    from opendde_harness.memory_engine.host_backend import _format_messages

    called = msg.assistant_message(
        "reading the assay notes",
        tool_calls=[msg.tool_call_block("c1", "read", {})],
        timestamp=1767258060000,
    )

    rendered = _format_messages([called])

    assert "[tools: read]" in rendered
    assert "reading the assay notes" in rendered


def test_a_tool_calls_arguments_may_arrive_as_a_json_string():
    """Providers have handed both back. A stringified payload is an annotation to
    parse, not one to drop."""
    from opendde_harness.memory_engine.host_backend import _tool_args

    assert _tool_args(json.dumps({"episode_summary": ["x"]})) == {"episode_summary": ["x"]}
    assert _tool_args([{"episode_summary": []}]) == {"episode_summary": []}
    assert _tool_args("not json") is None


# ── What the prompt ends up carrying ───────────────────────────────────


def a_context(current_message: str):
    from opendde_harness.context_engine.base import AssemblyContext

    return AssemblyContext(
        session_key="tui:main",
        current_message=current_message,
        media=None,
        channel=None,
        chat_id=None,
        session_messages=[],
    )


class IndexBackend:
    """A backend that indexes beside ``user.md`` without owning it -- the plugin shape."""

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        from opendde_harness.memory_engine.backend import Memory

        return [Memory(text="the user prefers HEPES buffer", score=0.9)]

    async def store(self, session_id, messages, *, metadata=None):
        return True

    async def feedback(self, signals):
        return None

    async def start(self):
        return None

    async def stop(self):
        return None


async def test_the_profile_reaches_the_prompt_exactly_once_when_the_writer_owns_it(tmp_path):
    """The writer reads the file it writes. If the segment also dumped the store,
    the same sections would be in the system prompt twice -- paid for twice, and
    contradicting each other the moment one lane was filtered differently."""
    from opendde_harness.context_engine.segments.memory import MemorySegmentBuilder

    backend, store, _ = backend_for(tmp_path, [])
    store.write_long_term(_PROFILE)
    segment = await MemorySegmentBuilder(store, backend).build(a_context("what is my role"))

    assert segment.text.count("## Long-term Memory") == 1
    assert segment.text.count("protein engineer") == 1
    assert segment.text.startswith("# Memory")


async def test_an_external_backend_is_the_only_long_term_memory_source(tmp_path):
    """An old local profile must not duplicate or contradict external recall."""
    from opendde_harness.context_engine.segments.memory import MemorySegmentBuilder

    store = MemoryStore(tmp_path)
    store.write_long_term(_PROFILE)
    segment = await MemorySegmentBuilder(store, IndexBackend()).build(a_context("what is my role"))

    assert "protein engineer" not in segment.text
    assert "the user prefers HEPES buffer" in segment.text, "and so do the index's hits"
    assert segment.meta["memory_hits"] == 1


# ── The shipped conformance suite, run against the default backend ─────


class TestHostMarkdownBackendContract(MemoryBackendContractTests, LifecycleContractTests):
    """The default backend against the suite a plugin author subclasses.

    The host's own owner has to satisfy the same contract it asks plugins for, or
    the contract is describing something nothing in the tree implements.
    """

    async def make_backend(self):
        workspace = Path(tempfile.mkdtemp(prefix="opendde-host-memory-contract-"))
        return HostMarkdownBackend(
            MemoryStore(workspace),
            binding=ModelBinding(Model([annotate_only("[2026-01-01 12:00] noted #project-x #decision")] * 4), MODEL),
        )
