from typing import Any

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.config.schema import Config, ProvidersConfig
from opendde_harness.providers.base import ErrorClassification, LLMProvider, LLMResponse, StreamDelta
from opendde_harness.providers.binding import ModelBinding
from opendde_harness.providers.model_id import row_for
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from opendde_harness.spine.events import Text
from tests._config import config as build_config
from tests._config import declared

PRIMARY = "primary/model"
FALLBACK = "fallback/model"
USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


class StreamProvider(LLMProvider):
    """Streams ``reply`` for any model except those whose next scripted
    outcome is an exception; ``script`` maps model -> list of exceptions to
    raise on successive calls (None = succeed)."""

    def __init__(self, script: dict[str, list[Exception | None]] | None = None, reply: str = "ok"):
        super().__init__()
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.reply = reply
        self.calls: list[str] = []
        self.raise_after_first_chunk = False

    async def chat(self, messages, tools=None, model=None, **kwargs):
        return LLMResponse(content=self.reply, finish_reason="stop", usage=dict(USAGE))

    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        self.calls.append(model)
        pending = self.script.get(model)
        exc = pending.pop(0) if pending else None
        if exc is not None and not self.raise_after_first_chunk:
            raise exc
        yield StreamDelta(content=self.reply)
        if exc is not None:
            raise exc
        yield StreamDelta(content=None, finish_reason="stop", usage=dict(USAGE))

    def classify_error(self, exc):
        return ErrorClassification("server", retryable=True) if "50" in str(exc) else ErrorClassification("no_model")

    def get_default_model(self):
        return PRIMARY


def _pinned(model: str, tokens: int, ceiling: int | None = None) -> ProvidersConfig:
    """A per-model window declaration, the only way a window is set by hand.

    It is a row in the model's own provider entry -- ``contextWindow``, and
    ``ceiling`` as ``maxTokens`` beside it -- so the declaration names the one
    model it is about. Both are the user's own words about their deployment,
    and for a provider that reports nothing -- every stub here -- they are the
    only tier that answers.

    The providers used here (``primary``, ``fallback``) are names pi does not
    ship, so each is written as a provider this config declares: an address and
    the protocol it speaks, which :func:`declared` pairs for us.
    """
    provider, _, bare = model.partition("/")
    row: dict[str, Any] = {"id": bare, "contextWindow": tokens}
    if ceiling is not None:
        row["maxTokens"] = ceiling
    return build_config(declared(provider, models=[row]), model=model).providers


def _loop(tmp_path, provider):
    return AgentLoop(provider, tmp_path, AgentLoopSettings(model=PRIMARY, providers=_pinned(PRIMARY, 4000)))


def test_the_loop_keeps_per_session_usage_in_memory_for_status(tmp_path):
    loop = _loop(tmp_path, StreamProvider())

    assert loop.strategies.get("usage_tracker") is loop.usage_tracker
    assert loop.usage_tracker.persist is False


async def test_a_stream_that_broke_after_delivering_says_why_it_stopped(tmp_path):
    """A drop this side of the model layer is the turn's answer, not a retry.

    Whether a failed call is run again is the model service's decision, under
    the budget the request carried; a generator that raised in this process is
    past that point, and the text already shown cannot be unshown. So the turn
    keeps it and says why it ended. The budget's details live in
    test_agent_loop_stream_retry.py.
    """
    provider = StreamProvider({PRIMARY: [RuntimeError("502"), None]})
    provider.raise_after_first_chunk = True
    loop = _loop(tmp_path, provider)
    tokens: list[str] = []

    async def on_token(text):
        tokens.append(text)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, PRIMARY, on_token_delta=on_token)

    assert provider.calls == [PRIMARY], "nothing in this process sends the call again"
    assert tokens == ["ok"]
    assert response.finish_reason == "error"
    assert response.content.startswith("ok")
    assert "Reply interrupted" in response.content


async def test_a_deterministic_failure_is_this_layers_answer(tmp_path):
    """Only a transient failure is spent on; a deterministic one comes straight
    back, whatever budget the session carries."""
    provider = StreamProvider({PRIMARY: [RuntimeError("no such model"), None]})
    loop = _loop(tmp_path, provider)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, PRIMARY)

    assert response.finish_reason == "error"
    assert not response.error_classification.retryable
    assert provider.calls == [PRIMARY]


def test_there_is_no_other_model_to_try():
    import inspect

    assert "fallback_models" not in inspect.signature(AgentLoop._llm_call_stream).parameters
    assert "fallback_models" not in inspect.signature(AgentLoop._run_agent_loop).parameters
    assert "fallback_models" not in inspect.signature(LLMProvider.chat_with_retry).parameters
    assert not hasattr(LLMProvider, "_CHAT_RETRY_DELAYS")


async def test_a_clean_stream_names_the_model_that_answered(tmp_path):
    provider = StreamProvider()
    loop = _loop(tmp_path, provider)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, PRIMARY)

    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert response.model == PRIMARY
    assert provider.calls == [PRIMARY]


def _request(text: str) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="cli", chat_id="t", sender_id="user", chat_type=ChatType.DM),
        text=text,
        conversation="cli:t",
    )


async def test_run_turn_reports_text_and_usage_on_outcome(tmp_path):
    provider = StreamProvider(reply="hello there")
    loop = _loop(tmp_path, provider)
    events = []

    async def emit(event):
        events.append(event)

    outcome = await loop.run_turn(_request("hi"), emit, lambda: [], stream=False)

    assert outcome.text == "hello there"
    assert outcome.explicit_reply is True
    assert outcome.usage.total_tokens == USAGE["total_tokens"]
    assert outcome.usage_detail["prompt_tokens"] == USAGE["prompt_tokens"]
    assert outcome.usage_detail["context_max"] == 4000
    assert [e.content for e in events if isinstance(e, Text)] == ["hello there"]
    session = loop.sessions.get_or_create("cli:t")
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    await loop.close_mcp()


async def test_slash_help_short_circuits_the_model(tmp_path):
    provider = StreamProvider()
    loop = _loop(tmp_path, provider)
    events = []

    async def emit(event):
        events.append(event)

    outcome = await loop.run_turn(_request("/help"), emit, lambda: [], stream=False)

    assert provider.calls == []
    assert outcome.text.startswith("ϒ OpenDDE Harness commands:")
    assert outcome.usage_detail == {}
    await loop.close_mcp()


def test_settings_from_config_mirrors_the_config_blocks():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"model": "openai/gpt-4o", "maxToolIterations": 7}},
            "tools": {"web": {"jinaApiKey": "j", "braveApiKey": "b"}, "disabledTools": ["exec"]},
            "skillForge": {"blocklist": ["x"]},
            "memory": {"userId": "u1"},
        }
    )
    settings = AgentLoopSettings.from_config(config)

    assert settings.model == "openai/gpt-4o"
    assert settings.max_iterations == 7
    assert settings.jina_api_key == "j"
    assert settings.brave_api_key == "b"
    assert settings.disabled_tools == ["exec"]
    assert settings.skill_forge.blocklist == ["x"]
    assert settings.memory.user_id == "u1"
    assert settings.empty_recovery.enabled is True


def test_unknown_window_is_carried_as_unknown(tmp_path, monkeypatch):
    from opendde_harness.providers import rates

    loop = AgentLoop(StreamProvider(), tmp_path, AgentLoopSettings(model=PRIMARY))

    assert loop.context_window_tokens is None
    assert loop.resolve_window().source == rates.SOURCE_UNKNOWN
    assert loop.context_engine.history.context_window_tokens is None, "and the selector does not trim"
    # Nothing knows the ceiling either -- no overlay declares one and this
    # provider reports none -- so nothing is held back. The request names no
    # ceiling, so there is nothing to hold back for.
    assert loop._reserved_output(None, PRIMARY) == 0


async def test_unknown_window_reports_no_context_max(tmp_path, monkeypatch):
    loop = AgentLoop(StreamProvider(reply="hi"), tmp_path, AgentLoopSettings(model=PRIMARY))

    async def emit(event):
        pass

    outcome = await loop.run_turn(_request("hi"), emit, lambda: [], stream=False)

    assert outcome.usage_detail["context_max"] == 0
    assert outcome.usage_detail["context_percent"] == 0
    await loop.close_mcp()


@pytest.mark.parametrize("window", [8192, 16384])
def test_a_window_no_larger_than_the_ceiling_reserves_the_whole_window(tmp_path, monkeypatch, window):
    """What the engine then makes of it -- a history budget of unknown rather
    than of zero -- is in tests/test_context_assembler.py."""
    loop = AgentLoop(
        StreamProvider(), tmp_path, AgentLoopSettings(model=PRIMARY, providers=_pinned(PRIMARY, window, window))
    )

    assert loop.context_window_tokens == window
    assert loop._reserved_output(window, PRIMARY) == window


def test_a_declared_row_sizes_the_window_per_model(tmp_path, monkeypatch):
    from opendde_harness.providers import rates

    provider = StreamProvider()
    loop = AgentLoop(provider, tmp_path, AgentLoopSettings(model=PRIMARY, providers=_pinned(PRIMARY, 8000)))

    assert loop.context_window_tokens == 8000
    assert loop.resolve_window().source == rates.SOURCE_DECLARED

    # A declaration is per model: switching to one with no row of its own does
    # not carry the number along, which is what the global pin did.
    loop.set_default_binding(ModelBinding(provider, FALLBACK))
    assert loop.context_window_tokens is None
    assert loop.context_engine.history.context_window_tokens is None


def test_settings_carry_every_providers_row():
    """The whole section travels, so any model a session switches to finds its row.

    Both kinds of entry: a provider this config declares, whose list *is* its
    catalogue, and one of pi's own, whose list only curates what the picker
    offers. A row is found by the model's qualified id and by nothing else --
    the gateway id below keeps its own slashes, which is why the lookup splits
    on the first one only.
    """
    config = build_config(
        {
            **declared("my-vllm", models=[{"id": "gpt-x", "contextWindow": 8000, "maxTokens": 512}]),
            "openrouter": {"apiKey": "k", "models": [{"id": "nano-gpt/tiny", "name": "Tiny"}]},
        },
        model="my-vllm/gpt-x",
    )

    providers = AgentLoopSettings.from_config(config).providers

    assert row_for(providers, "my-vllm/gpt-x").context_window == 8000
    assert row_for(providers, "my-vllm/gpt-x").max_tokens == 512
    assert row_for(providers, "openrouter/nano-gpt/tiny").name == "Tiny"


def test_loop_applies_settings(tmp_path):
    provider = StreamProvider()
    settings = AgentLoopSettings(
        model=FALLBACK, max_iterations=3, disabled_tools=["exec"], providers=_pinned(FALLBACK, 999)
    )
    loop = AgentLoop(provider, tmp_path, settings)

    assert loop.model == FALLBACK
    assert loop.max_iterations == 3
    assert loop.context_window_tokens == 999
    assert not loop.tools.has("exec")
    assert loop.tools.has("read")


@pytest.mark.parametrize(
    "policy,interactive,expected", [("never", True, False), ("always", False, True), ("interactive", False, False)]
)
def test_checkpoint_gate(policy, interactive, expected):
    assert AgentLoop._checkpoint_active(policy, interactive) is expected


def test_session_history_carries_reasoning_back_to_the_model():
    from opendde_harness.providers import messages as msg
    from opendde_harness.session.manager import Session
    from tests import _messages as build

    session = Session(key="tui:test")
    session.record(build.user("hi"))
    session.record(build.assistant(reasoning="weighing the options", calls=[("c1", "read", {})]))
    session.record(build.tool_result("c1", "read", "ok"))

    assistant = session.get_history()[1]

    # DeepSeek thinking mode rejects a request whose assistant turn comes back
    # without the reasoning it produced.
    assert msg.thinking_of(assistant) == "weighing the options"
    assert msg.tool_call_ids(assistant) == ["c1"]
    # The timestamp is pi's own field and travels; the record id is the
    # harness's and does not.
    assert isinstance(assistant["timestamp"], int)
    assert "id" not in assistant


def test_the_budget_reserves_the_models_own_ceiling_not_a_table_estimate(tmp_path):
    """The Codex login is reached under a bare slug. Sizing used to join that
    slug against a table, find no row and reserve an estimate where the model's
    own ceiling is 128000. The ceiling is now the provider's own answer about
    the model it is about to call, and no table is consulted at all."""

    class KnowsItsCeiling(StreamProvider):
        def max_output_tokens(self):
            return 128_000

    loop = AgentLoop(KnowsItsCeiling(), tmp_path, AgentLoopSettings(model="openai-codex/gpt-5.3-codex-spark"))

    assert loop._reserved_output(None, "openai-codex/gpt-5.3-codex-spark") == 128_000


async def test_the_loop_reports_cumulative_vendor_output_after_each_call(tmp_path):
    from opendde_harness.agent.loop.main import _reasoning_tokens_of

    assert _reasoning_tokens_of({"completion_tokens": 10}) == 0
    assert _reasoning_tokens_of({"completion_tokens_details": {"reasoning_tokens": 7}}) == 7

    class Counted(StreamProvider):
        async def chat_stream(self, messages, tools=None, model=None, **kwargs):
            yield StreamDelta(content="ok")
            yield StreamDelta(
                content=None,
                finish_reason="stop",
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 12,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            )

    loop = AgentLoop(Counted(), tmp_path, AgentLoopSettings(model=PRIMARY))
    seen = []

    async def on_usage(completion, reasoning, calls):
        seen.append((completion, reasoning, calls))

    async def on_token(text):
        pass

    await loop._run_agent_loop(
        [{"role": "user", "content": "hi"}], session_key="tui:t", on_token_delta=on_token, on_usage=on_usage
    )

    assert seen == [(12, 4, 1)]


@pytest.fixture
def skill_catalogue(tmp_path):
    """The ``# Skills`` segment as the factory wires it, over a synthetic
    registry, with every provider call the build makes counted."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from opendde_harness.config.features import ContextConfig, SkillForgeConfig, SkillForgeRouterConfig
    from opendde_harness.context_engine.base import AssemblyContext
    from opendde_harness.context_engine.factory import build_context_engine
    from opendde_harness.context_engine.segments.skills import SkillsSegmentBuilder
    from opendde_harness.memory_engine.skill_local.local_pool import LocalPool
    from opendde_harness.memory_engine.skill_local.types import SkillMeta

    class Counting(StreamProvider):
        """Any request at all is a failure of the deterministic path."""

        def __init__(self):
            super().__init__()
            self.requests: list[Any] = []

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.requests.append(messages)
            return await super().chat(messages, tools, model, **kwargs)

        async def chat_stream(self, messages, tools=None, model=None, **kwargs):
            self.requests.append(messages)
            async for delta in super().chat_stream(messages, tools, model, **kwargs):
                yield delta

    def make(count=2, *, extra_source=False, router_config=None, **config):
        # Skill 0 is the only one whose text matches the query below, so a
        # BM25-narrowed catalogue keeps it and drops the rest.
        metas = [
            SkillMeta(
                id=f"workspace/{i}",
                name="design" if i == 0 else f"skill_{i}",
                description="protein design" if i == 0 else "weather forecast",
                path=tmp_path / str(i) / "SKILL.md",
                content="Design proteins" if i == 0 else "Check the weather",
                source="workspace" if i == 0 else "builtin",
            )
            for i in range(count)
        ]

        def _get(name, source=None):
            return next(
                (m for m in metas if m.name == name and (source is None or m.source == source)),
                None,
            )

        registry = SimpleNamespace(
            list_all=lambda: metas,
            get=_get,
            # Stands in for the real requirement check: a declared ``requires``
            # is treated as unmet, which is what the availability filter reads.
            check_available=lambda name, source=None: not getattr(_get(name, source), "requires", None),
        )
        provider = Counting()
        pool = LocalPool(registry)
        from opendde_harness.config.paths import get_workspace_storage

        backend = SimpleNamespace(recall=AsyncMock(return_value=[])) if extra_source else None
        engine = build_context_engine(
            workspace=tmp_path,
            config=ContextConfig(),
            builder=SimpleNamespace(
                memory=SimpleNamespace(),
                skills=SimpleNamespace(pool=pool, registry=registry),
                storage=get_workspace_storage(tmp_path),
            ),
            provider=provider,
            model=PRIMARY,
            context_window_tokens=200_000,
            get_tool_definitions=lambda: [],
            backend=backend,
            skill_forge_config=SkillForgeConfig(**config),
            skill_forge_router_config=router_config or SkillForgeRouterConfig(),
        )
        segment = next(b for b in engine._builders if isinstance(b, SkillsSegmentBuilder))
        ctx = AssemblyContext("test", "protein design", None, None, None, [])
        return segment, ctx, provider, metas, pool

    return make


async def test_a_small_catalogue_advertises_every_skill_without_a_model_call(skill_catalogue):
    from opendde_harness.context_engine.segments.skills import FULL_CATALOGUE_MAX

    segment, ctx, provider, metas, _ = skill_catalogue(FULL_CATALOGUE_MAX)

    result = await segment.build(ctx)

    assert provider.requests == []
    assert result.meta["available_skill_ids"] == sorted(f"local/{m.name}" for m in metas)
    # Ids and one-line descriptions only; no body reaches the prompt.
    assert "protein design" in result.text and "Design proteins" not in result.text
    assert result.text.startswith("# Skills")


async def test_a_large_catalogue_advertises_the_bm25_top_k_without_a_model_call(skill_catalogue):
    from opendde_harness.context_engine.segments.skills import BM25_TOP_K, FULL_CATALOGUE_MAX

    segment, ctx, provider, _, _ = skill_catalogue(FULL_CATALOGUE_MAX + 1)

    result = await segment.build(ctx)

    assert provider.requests == []
    ids = result.meta["available_skill_ids"]
    assert ids[0] == "local/design" and 0 < len(ids) <= BM25_TOP_K


async def test_a_catalogue_reached_through_a_backend_is_narrowed_not_listed(skill_catalogue):
    """A memory source is a remote recall, so there is no complete pool to list."""
    from opendde_harness.context_engine.segments.skills import BM25_TOP_K

    segment, ctx, provider, _, _ = skill_catalogue(BM25_TOP_K + 3, extra_source=True)

    result = await segment.build(ctx)

    assert provider.requests == []
    assert result.meta["available_skill_ids"] == ["local/design"]


@pytest.mark.parametrize("off", ["forge", "router"])
async def test_either_master_switch_off_advertises_nothing_and_calls_nobody(skill_catalogue, off):
    from opendde_harness.config.features import SkillForgeRouterConfig

    segment, ctx, provider, _, _ = skill_catalogue(
        3,
        **({"enabled": False} if off == "forge" else {}),
        router_config=SkillForgeRouterConfig(enabled=False) if off == "router" else None,
    )

    result = await segment.build(ctx)

    assert provider.requests == []
    assert result.text == "" and result.meta["available_skill_ids"] == []


async def test_an_empty_message_costs_nothing_on_a_catalogue_too_large_to_list(skill_catalogue):
    from dataclasses import replace

    from opendde_harness.context_engine.segments.skills import FULL_CATALOGUE_MAX

    segment, ctx, provider, _, _ = skill_catalogue(FULL_CATALOGUE_MAX + 1)

    result = await segment.build(replace(ctx, current_message="  "))

    assert provider.requests == []
    assert result.text == "" and result.meta["available_skill_ids"] == []


async def test_the_catalogue_leaves_out_always_blocked_and_unrunnable_skills(skill_catalogue):
    segment, ctx, _, metas, _ = skill_catalogue(4, blocklist=["skill_2"])
    metas[1].always = True
    metas[3].requires = {"bins": ["a-binary-nobody-has"]}

    result = await segment.build(ctx)

    assert result.meta["available_skill_ids"] == ["local/design"]


async def test_a_skill_added_after_boot_joins_the_catalogue(skill_catalogue):
    from copy import deepcopy

    segment, ctx, _, metas, _ = skill_catalogue(2)
    added = deepcopy(metas[0])
    added.id, added.name, added.source = "external/new", "new", "external"
    metas.append(added)

    result = await segment.build(ctx)

    assert "local/new" in result.meta["available_skill_ids"]


async def test_two_skills_sharing_a_name_are_narrowed_rather_than_listed_ambiguously(skill_catalogue):
    """Both would render the same ``local/<name>``, which ``use_skill`` cannot split."""
    segment, ctx, provider, metas, pool = skill_catalogue(2)
    metas[1].name = "design"
    pool.rebuild_index()

    result = await segment.build(ctx)

    assert provider.requests == []
    assert result.meta["available_skill_ids"] == ["local/design"]


async def test_a_full_catalogue_is_ordered_independently_of_the_message(skill_catalogue):
    """The block sits in the cached prefix, so it must not move when the
    message does."""
    from dataclasses import replace

    segment, ctx, _, _, _ = skill_catalogue(6)

    first = await segment.build(ctx)
    second = await segment.build(replace(ctx, current_message="check the weather instead"))

    assert first.text == second.text


async def test_a_block_scalar_description_reaches_the_catalogue_as_one_line(tmp_path):
    """Most shipped SKILL.md files write ``description: |``. The description is
    the whole selection signal now, so storing the indicator instead of the
    text advertised half the catalogue as a literal ``|``."""
    from opendde_harness.config.paths import get_workspace_storage
    from opendde_harness.context_engine.base import AssemblyContext
    from opendde_harness.context_engine.segments.skills import SkillsSegmentBuilder
    from opendde_harness.memory_engine.skill_forge import LocalSkillCatalog, LocalSkillSource, SkillForgeRouter

    skill = get_workspace_storage(tmp_path / "workspace").skills / "epitopes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: epitopes\ndescription: |\n  Identify epitope residues\n  and hotspot coverage.\n---\n\nBody.\n"
    )
    (tmp_path / "builtin").mkdir()
    catalog = LocalSkillCatalog(
        tmp_path / "workspace", config=None, builtin_skills_dir=tmp_path / "builtin", start_watcher=False
    )

    assert catalog.registry.get("epitopes").description == "Identify epitope residues and hotspot coverage."

    router = SkillForgeRouter([LocalSkillSource(pool=catalog.pool, registry=catalog.registry)])
    result = await SkillsSegmentBuilder(router).build(AssemblyContext("s", "epitope hotspots", None, None, None, []))

    assert "local/epitopes" in result.meta["available_skill_ids"]
    assert "- **epitopes** [`local/epitopes`]: Identify epitope residues and hotspot coverage." in result.text


async def test_a_plain_chat_names_the_answering_model(tmp_path):
    provider = StreamProvider()
    response = await provider.chat_with_retry([{"role": "user", "content": "hi"}], model=None)

    assert response.model == PRIMARY


async def test_the_turns_session_is_marked_only_while_the_turn_runs(tmp_path):
    """Calls made on the turn's behalf see the turn's session; nothing sees it
    afterwards, so a nested turn or the next one starts from its own mark."""
    from opendde_harness.token_wise.base import CURRENT_SESSION_KEY

    class MarkingProvider(StreamProvider):
        def __init__(self):
            super().__init__()
            self.seen: list[str | None] = []

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.seen.append(CURRENT_SESSION_KEY.get())
            return await super().chat(messages, tools, model, **kwargs)

    provider = MarkingProvider()
    loop = _loop(tmp_path, provider)
    outer = CURRENT_SESSION_KEY.set("outer-scope")
    try:
        await loop.run_turn(_request("hi"), lambda event: _noop(), lambda: [], stream=False)

        assert provider.seen == ["cli:t"]
        assert CURRENT_SESSION_KEY.get() == "outer-scope"
    finally:
        CURRENT_SESSION_KEY.reset(outer)
        await loop.close_mcp()


async def _noop():
    return None
