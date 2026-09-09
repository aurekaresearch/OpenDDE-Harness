"""The Curator's fast-path gate against a budget it cannot measure."""

import pytest

from opendde_harness.config.features import ContextConfig
from opendde_harness.context_engine.base import AssembledPrefix, AssemblyContext
from opendde_harness.context_engine.segments.curator import CuratorSegmentBuilder
from opendde_harness.memory_engine.base import TokenBudget


class _NeverCalled:
    """A provider the slow path would need; reaching it is the failure."""

    def __getattr__(self, name):
        raise AssertionError(f"provider.{name} must not be reached on the fast path")


class _NoLLM:
    """No token counter of its own, and any call is the failure."""

    def __getattr__(self, name):
        if name.startswith("chat"):
            raise AssertionError(f"provider.{name} must not be reached below the floor")
        raise AttributeError(name)


def _messages(n: int) -> list[dict]:
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"question {i} " * 50})
        out.append({"role": "assistant", "content": f"answer {i} " * 50})
    return out


def _ctx(messages, budget):
    return AssemblyContext(
        session_key="tui:t",
        current_message="next",
        media=None,
        channel="tui",
        chat_id="t",
        session_messages=messages,
        budget=budget,
        prefix=AssembledPrefix(system_prefix="sys", user_message={"role": "user", "content": "next"}, tool_defs=[]),
    )


@pytest.fixture
def builder(tmp_path):
    return CuratorSegmentBuilder(
        workspace=tmp_path,
        config=ContextConfig(),
        provider=_NeverCalled(),
        model="fake/model",
        context_window_tokens=None,
        get_tool_definitions=list,
    )


async def test_unknown_budget_takes_the_fast_path_however_long_the_history(builder):
    messages = _messages(40)
    budget = TokenBudget(
        context_length=None, reserved_output=1000, reserved_tools=0, reserved_system=0, available_history=None
    )

    seg = await builder.build(_ctx(messages, budget))

    assert seg.meta["path"] == "fast"
    assert seg.meta["threshold_tokens"] is None
    assert len(seg.history) == len(messages)


@pytest.mark.parametrize("available_history", [0, 1])
async def test_a_budget_nothing_fits_in_does_not_force_the_slow_path(builder, available_history):
    # window <= output ceiling once left available_history at 0, a threshold
    # of 0, and "history < 0" false for an empty history -- the LLM loop from
    # turn one, forever.
    budget = TokenBudget(
        context_length=8192,
        reserved_output=8192,
        reserved_tools=0,
        reserved_system=0,
        available_history=available_history,
    )

    seg = await builder.build(_ctx([], budget))

    assert seg.meta["path"] == "fast"
    assert seg.history == []


@pytest.mark.parametrize("available_history", [2, 50, 500])
async def test_a_budget_below_the_protected_head_takes_the_fallback_without_an_llm(tmp_path, available_history):
    # The band the <= 0 guard leaves: a little positive headroom, where the
    # slow path burned its steps producing exactly what the deterministic
    # fallback produces. Measured at 58 spare tokens: two messages, twelve
    # steps, every turn.
    builder = CuratorSegmentBuilder(
        workspace=tmp_path,
        config=ContextConfig(),
        provider=_NoLLM(),
        model="fake/model",
        context_window_tokens=40_000,
        get_tool_definitions=list,
    )
    messages = _messages(12)
    budget = TokenBudget(
        context_length=40_000,
        reserved_output=16_384,
        reserved_tools=0,
        reserved_system=0,
        available_history=available_history,
    )

    seg = await builder.build(_ctx(messages, budget))

    assert seg.meta["path"] == "fallback"
    assert seg.meta["reason"] == "below_floor"
    assert seg.meta["floor_tokens"] >= available_history
    # Trimmed, unlike the fast path, which ships everything -- but never past
    # the protected head: an empty history in the window is not an answer.
    assert 0 < len(seg.history) < len(messages)
    protected = ContextConfig().protect_first_n * 2
    assert [m["content"] for m in seg.history[:protected]] == [m["content"] for m in messages[:protected]]


async def test_above_the_floor_the_slow_path_is_attempted(tmp_path):
    class _Refusing(_NoLLM):
        async def chat_with_retry(self, *a, **k):
            raise RuntimeError("no curator model configured")

    builder = CuratorSegmentBuilder(
        workspace=tmp_path,
        config=ContextConfig(),
        provider=_Refusing(),
        model="fake/model",
        context_window_tokens=200_000,
        get_tool_definitions=list,
    )
    messages = _messages(40)
    budget = TokenBudget(
        context_length=200_000, reserved_output=16_384, reserved_tools=0, reserved_system=0, available_history=20_000
    )

    seg = await builder.build(_ctx(messages, budget))

    assert seg.meta["path"] == "fallback"
    assert seg.meta["reason"] == "slow_path_failed"


def test_unknown_budget_reports_no_compaction_trigger():
    budget = TokenBudget(
        context_length=None, reserved_output=1000, reserved_tools=0, reserved_system=0, available_history=None
    )

    assert budget.threshold is None
    assert not budget.known
    assert TokenBudget(100_000, 1000, 0, 0, 90_000).threshold == 67_500


async def test_a_slow_path_over_its_time_budget_falls_back_deterministically(tmp_path):
    import asyncio

    class _Stalling(_NoLLM):
        async def chat_with_retry(self, *a, **k):
            await asyncio.sleep(10)

    builder = CuratorSegmentBuilder(
        workspace=tmp_path,
        config=ContextConfig(curator_timeout_seconds=0.05),
        provider=_Stalling(),
        model="fake/model",
        context_window_tokens=200_000,
        get_tool_definitions=list,
    )
    messages = _messages(40)
    budget = TokenBudget(
        context_length=200_000, reserved_output=16_384, reserved_tools=0, reserved_system=0, available_history=20_000
    )

    seg = await builder.build(_ctx(messages, budget))

    assert seg.meta["path"] == "fallback"
    assert seg.meta["reason"] == "slow_path_timeout"


def test_the_curator_follows_the_agent_model_unless_pinned(tmp_path):
    following = CuratorSegmentBuilder(
        workspace=tmp_path,
        config=ContextConfig(),
        provider=_NoLLM(),
        model="fake/model",
        context_window_tokens=None,
        get_tool_definitions=list,
    )
    pinned = CuratorSegmentBuilder(
        workspace=tmp_path,
        config=ContextConfig(curator_model="fake/small"),
        provider=_NoLLM(),
        model="fake/model",
        context_window_tokens=None,
        get_tool_definitions=list,
    )

    assert following.curator_model == "fake/model"
    assert pinned.curator_model == "fake/small"
    following.set_provider(_NoLLM(), "fake/other")
    pinned.set_provider(_NoLLM(), "fake/other")
    assert following.curator_model == "fake/other"
    assert pinned.curator_model == "fake/small"
