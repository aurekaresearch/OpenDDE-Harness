"""One render per turn, and the budget is measured on what it rendered.

The engine used to render a *second*, representative system prompt to size the
turn (with no recall hits and no router skills in it), then render the real one
and select history against the estimate of the other. These tests hold the
single render: ``reserved_system`` is the cost of the system message the request
carries, and the prefix a turn selects against is that turn's own.
"""

from __future__ import annotations

import asyncio

import pytest

from opendde_harness.context_engine.assembler import ContextAssembler
from opendde_harness.context_engine.base import AssemblyContext, Segment, TurnContext
from opendde_harness.context_engine.history_trimmer import ContextBudgetError, HistoryTrimmer
from opendde_harness.providers.base import COMPACTION_KEY, LLMProvider
from opendde_harness.utils.helpers import estimate_prompt_tokens
from tests import _messages as build

MODEL = "fake/model"
TOOLS = [
    {
        "type": "function",
        "function": {"name": "read", "description": "read a file", "parameters": {"type": "object"}},
    }
]


class _NoLLM(LLMProvider):
    """Any call is the failure: assembly makes none."""

    async def chat(self, messages, tools=None, model=None, **kwargs):
        raise AssertionError("assembly makes no model call")

    def get_default_model(self) -> str:
        return MODEL


class _Replays(_NoLLM):
    supports_compaction = True

    def replays_compaction(self, marker, model):
        return marker.get("provider") == "fake" and marker.get("model") == model


class _Text:
    """A builder that contributes a fixed block."""

    def __init__(self, name: str, order: int, text: str, meta: dict | None = None) -> None:
        self.name = name
        self.order = order
        self._text = text
        self._meta = meta or {}

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        return Segment(text=self._text, meta=dict(self._meta))


class _PerSession:
    """A builder whose block depends on the session, and waits for its peer."""

    name = "per_session"
    order = 1

    def __init__(self, blocks: dict[str, str], gate: asyncio.Event, arrived: asyncio.Event) -> None:
        self._blocks = blocks
        self._gate = gate
        self._arrived = arrived

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        self._arrived.set()
        await self._gate.wait()
        return Segment(text=self._blocks[ctx.session_key])


def _engine(
    builders: list,
    *,
    window: int | None,
    tools: list | None = None,
    provider: LLMProvider | None = None,
    protect: int = 3,
) -> ContextAssembler:
    def get_tools() -> list:
        return list(tools or [])

    history = HistoryTrimmer(provider or _NoLLM(), MODEL, get_tools, window, protect_first_n=protect)
    return ContextAssembler(builders, get_tools, history)


def _turns(n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        out.append(build.user(f"question {i} " * 40))
        out.append(build.assistant(f"answer {i} " * 40))
    return out


async def test_reserved_system_is_the_system_message_actually_sent() -> None:
    engine = _engine([_Text("identity", 1, "IDENTITY " * 50), _Text("memory", 4, "MEMORY " * 50)], window=200_000)

    assembled = await engine.assemble("tui:t", _turns(3), turn=TurnContext(current_message="next"))

    system = assembled.messages[0]
    assert system["role"] == "system"
    assert assembled.metadata["budget"]["reserved_system"] == estimate_prompt_tokens([system])
    # Both blocks are in the one render, in order, and nothing else rendered.
    assert system["content"].startswith("IDENTITY")
    assert "MEMORY" in system["content"]


async def test_reserved_tools_is_the_tool_schemas_actually_sent() -> None:
    engine = _engine([_Text("identity", 1, "IDENTITY")], window=200_000, tools=TOOLS)

    assembled = await engine.assemble("tui:t", _turns(2), turn=TurnContext(current_message="next"))

    assert assembled.metadata["budget"]["reserved_tools"] == estimate_prompt_tokens([], TOOLS)


async def test_the_available_history_is_the_window_less_what_the_prompt_holds() -> None:
    engine = _engine([_Text("identity", 1, "IDENTITY " * 50)], window=50_000, tools=TOOLS)

    assembled = await engine.assemble(
        "tui:t", _turns(2), turn=TurnContext(current_message="next", reserved_output=8_000)
    )

    budget = assembled.metadata["budget"]
    assert budget["context_length"] == 50_000 and budget["reserved_output"] == 8_000
    assert budget["available_history"] == 50_000 - 8_000 - budget["reserved_tools"] - budget["reserved_system"]


async def test_an_unknown_window_reports_unknown_and_trims_nothing() -> None:
    messages = _turns(40)
    engine = _engine([_Text("identity", 1, "IDENTITY")], window=None)

    assembled = await engine.assemble("tui:t", messages, turn=TurnContext(current_message="next"))

    assert assembled.metadata["budget"]["available_history"] is None
    assert assembled.metadata["budget"]["context_length"] is None
    assert len(assembled.messages) == len(messages) + 2, "every message rode out beside system and user"


async def test_a_window_the_reply_alone_fills_is_unknown_not_zero() -> None:
    """A model whose window is no larger than its output ceiling leaves nothing
    for history. Reported as unknown, not as a budget of zero -- which read as
    "every history overflows"."""
    engine = _engine([_Text("identity", 1, "IDENTITY")], window=8_192)

    assembled = await engine.assemble(
        "tui:t", _turns(2), turn=TurnContext(current_message="next", reserved_output=8_192)
    )

    assert assembled.metadata["budget"]["available_history"] is None


async def test_a_degenerate_budget_is_said_once_per_window() -> None:
    """The warning is per resolved window, not per turn: this runs every turn."""
    from loguru import logger as loguru_logger

    engine = _engine([_Text("identity", 1, "IDENTITY")], window=8_192)
    lines: list[str] = []
    sink = loguru_logger.add(lambda message: lines.append(str(message)), level="WARNING")
    try:
        for _ in range(3):
            await engine.assemble("tui:t", _turns(1), turn=TurnContext(current_message="next", reserved_output=8_192))
    finally:
        loguru_logger.remove(sink)

    assert sum("holds no history" in line for line in lines) == 1


async def test_a_turn_whose_history_cannot_be_dropped_far_enough_raises() -> None:
    """The protected head and the newest exchange are what will not fit, and
    they are not droppable. Refused locally rather than sent."""
    engine = _engine([_Text("identity", 1, "IDENTITY")], window=500)

    with pytest.raises(ContextBudgetError) as excinfo:
        await engine.assemble("tui:t", _turns(20), turn=TurnContext(current_message="next"))

    assert "/new" in str(excinfo.value)


async def test_a_prefix_too_large_for_the_window_is_sent_and_refused_upstream() -> None:
    """The other shape: nothing in the history is what does not fit. There is no
    turn to drop that would help, and a declared window can simply be wrong, so
    the provider's own refusal is the truth."""
    engine = _engine([_Text("identity", 1, "IDENTITY " * 2_000)], window=4_000)

    assembled = await engine.assemble("tui:t", _turns(20), turn=TurnContext(current_message="next"))

    # The minimum a turn means: the three protected user messages and the
    # newest exchange with the message it answers. Nothing is filled in beside
    # them, because there is no room to fill.
    assert assembled.metadata["history"]["included"] == 5
    assert assembled.metadata["history"]["dropped"] == 40 - 5
    assert HistoryTrimmer.structural_errors(assembled.messages) == []


async def test_the_identity_block_is_rendered_from_the_turn_not_from_config(tmp_path) -> None:
    """Rendering used to read config three times per turn for these three
    facts: the reply language, the model id and whether a memory backend is
    configured. A prompt that belongs to a turn which started earlier was then
    rendered against whatever config said at render time."""
    from opendde_harness.context_engine.segments.identity import IdentitySegmentBuilder

    engine = _engine([IdentitySegmentBuilder(tmp_path)], window=200_000)

    zh = await engine.assemble(
        "tui:t",
        _turns(1),
        turn=TurnContext(current_message="next", model="vendor/z", language="zh", long_term_memory=True),
    )
    en = await engine.assemble(
        "tui:t",
        _turns(1),
        turn=TurnContext(current_message="next", model=None, language="en", long_term_memory=False),
    )

    first, second = zh.messages[0]["content"], en.messages[0]["content"]
    assert str(tmp_path) in first
    assert "vendor/z" in first and "running on model" not in second
    assert "简体中文" in first and "简体中文" not in second
    assert "episodes.md" not in first and "episodes.md" not in second


async def test_two_sessions_assembling_at_once_keep_their_own_prefix() -> None:
    """The prefix used to be a field on a long-lived assembler, set before an
    await: a 1M-token research session and a short one overlapped there and
    sized their history against each other's overhead."""
    gate, arrived = asyncio.Event(), asyncio.Event()
    blocks = {"tui:big": "BIG " * 400, "tui:small": "SMALL"}
    engine = _engine([_PerSession(blocks, gate, arrived)], window=200_000)

    async def run(key: str):
        return await engine.assemble(key, _turns(2), turn=TurnContext(current_message=f"from {key}"))

    big = asyncio.create_task(run("tui:big"))
    small = asyncio.create_task(run("tui:small"))
    await arrived.wait()
    gate.set()
    big_assembled, small_assembled = await asyncio.gather(big, small)

    assert big_assembled.messages[0]["content"] == blocks["tui:big"]
    assert small_assembled.messages[0]["content"] == blocks["tui:small"]
    assert (
        big_assembled.metadata["budget"]["reserved_system"] > small_assembled.metadata["budget"]["reserved_system"]
    ), "each turn was sized against its own prefix"
    assert "from tui:big" in str(big_assembled.messages[-1]["content"])
    assert "from tui:small" in str(small_assembled.messages[-1]["content"])


async def test_a_history_the_backend_already_compacted_rides_out_whole() -> None:
    """Past a marker this model's own backend replays, the messages before it
    are not in the request. Their size must not cut the ones that are."""
    marker = build.marker({"provider": "fake", "model": MODEL, "items": [{"type": "opaque"}]})
    messages = [*_turns(40), marker, build.user("and now?")]
    engine = _engine([_Text("identity", 1, "IDENTITY")], window=20_000, provider=_Replays())

    assembled = await engine.assemble("tui:t", messages, turn=TurnContext(current_message="next"))

    assert any(COMPACTION_KEY in m for m in assembled.messages)
    assert assembled.metadata["history"]["dropped"] == 0, "nothing after the marker was dropped"
    assert assembled.include_indices == list(range(len(messages) - 2, len(messages)))


async def test_the_metadata_carries_every_builders_meta_and_the_engine_name() -> None:
    engine = _engine(
        [
            _Text("identity", 1, "IDENTITY"),
            _Text("skills", 6, "# Skills", meta={"available_skill_ids": ["local:a"]}),
        ],
        window=200_000,
    )

    assembled = await engine.assemble("tui:t", _turns(1), turn=TurnContext(current_message="next"))

    assert assembled.metadata["engine"] == "context_assembler"
    assert assembled.metadata["available_skill_ids"] == ["local:a"]
    assert assembled.metadata["history"]["included"] == 2


async def test_the_engine_follows_a_model_switch() -> None:
    engine = _engine([_Text("identity", 1, "IDENTITY")], window=200_000)

    engine.set_context_window(64_000)
    assert engine.history.context_window_tokens == 64_000
    engine.set_context_window(None)
    assert engine.history.context_window_tokens is None
