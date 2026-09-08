import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.config.schema import Config
from opendde_harness.providers.base import ErrorClassification, LLMProvider, LLMResponse, StreamDelta
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from opendde_harness.spine.events import Text

PRIMARY = "primary/model"
FALLBACK = "fallback/model"
USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


class StreamProvider(LLMProvider):
    """Streams ``reply`` for any model except those whose next scripted
    outcome is an exception; ``script`` maps model -> list of exceptions to
    raise on successive calls (None = succeed)."""

    _CHAT_RETRY_DELAYS = (0,)

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

    def classify_error(self, exc=None, content=None):
        text = str(exc) if exc is not None else (content or "")
        if "50" in text:
            return ErrorClassification("server", retryable=True, should_fallback=True)
        return ErrorClassification("model_unavailable", should_fallback=True)

    def get_default_model(self):
        return PRIMARY


def _loop(tmp_path, provider):
    return AgentLoop(provider, tmp_path, AgentLoopSettings(model=PRIMARY, context_window_tokens=4000))


async def test_stream_falls_back_to_next_model(tmp_path):
    provider = StreamProvider({PRIMARY: [RuntimeError("model not found")]})
    loop = _loop(tmp_path, provider)
    tokens: list[str] = []

    async def on_token(text):
        tokens.append(text)

    response = await loop._llm_call_stream(
        [{"role": "user", "content": "hi"}], None, PRIMARY, fallback_models=[FALLBACK], on_token_delta=on_token
    )

    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert tokens == ["ok"]
    assert provider.calls == [PRIMARY, FALLBACK]


async def test_stream_retries_transient_error_before_first_chunk(tmp_path):
    provider = StreamProvider({PRIMARY: [RuntimeError("502"), None]})
    loop = _loop(tmp_path, provider)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, PRIMARY)

    assert response.finish_reason == "stop"
    assert provider.calls == [PRIMARY, PRIMARY]


async def test_stream_error_after_delivery_is_final(tmp_path):
    provider = StreamProvider({PRIMARY: [RuntimeError("cut")]})
    provider.raise_after_first_chunk = True
    loop = _loop(tmp_path, provider)
    tokens: list[str] = []

    async def on_token(text):
        tokens.append(text)

    response = await loop._llm_call_stream(
        [{"role": "user", "content": "hi"}], None, PRIMARY, fallback_models=[FALLBACK], on_token_delta=on_token
    )

    assert response.finish_reason == "error"
    assert response.error_classification.category == "model_unavailable"
    assert tokens == ["ok"]
    assert provider.calls == [PRIMARY]
    assert response.content.startswith("ok")
    assert "[Reply interrupted:" in response.content


async def test_stream_exhausted_chain_returns_error_response(tmp_path):
    provider = StreamProvider({PRIMARY: [RuntimeError("503"), RuntimeError("gone")], FALLBACK: [RuntimeError("gone too")]})
    loop = _loop(tmp_path, provider)

    response = await loop._llm_call_stream([{"role": "user", "content": "hi"}], None, PRIMARY, fallback_models=[FALLBACK])

    assert response.finish_reason == "error"
    assert "gone too" in (response.content or "")
    assert provider.calls == [PRIMARY, PRIMARY, FALLBACK]


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
            "agents": {"defaults": {"model": "openai/gpt-4o", "maxToolIterations": 7, "contextWindowTokens": 123}},
            "tools": {"web": {"jinaApiKey": "j", "search": {"apiKey": "b"}}, "disabledTools": ["exec"]},
            "skillForge": {"blocklist": ["x"]},
            "memory": {"userId": "u1"},
        }
    )
    settings = AgentLoopSettings.from_config(config)

    assert settings.model == "openai/gpt-4o"
    assert settings.max_iterations == 7
    assert settings.context_window_tokens == 123
    assert settings.brave_api_key == "b"
    assert settings.jina_api_key == "j"
    assert settings.disabled_tools == ["exec"]
    assert settings.skill_forge.blocklist == ["x"]
    assert settings.memory.user_id == "u1"
    assert settings.empty_recovery.enabled is True


def test_loop_applies_settings(tmp_path):
    provider = StreamProvider()
    settings = AgentLoopSettings(model=FALLBACK, max_iterations=3, disabled_tools=["exec"], context_window_tokens=999)
    loop = AgentLoop(provider, tmp_path, settings)

    assert loop.model == FALLBACK
    assert loop.max_iterations == 3
    assert loop.context_window_tokens == 999
    assert not loop.tools.has("exec")
    assert loop.tools.has("read_file")


@pytest.mark.parametrize("policy,interactive,expected", [("never", True, False), ("always", False, True), ("interactive", False, False)])
def test_checkpoint_gate(policy, interactive, expected):
    assert AgentLoop._checkpoint_active(policy, interactive) is expected


def test_session_history_carries_reasoning_back_to_the_model():
    from opendde_harness.session.manager import Session

    session = Session(key="tui:test")
    session.record({"role": "user", "content": "hi"})
    session.record(
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "weighing the options",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        }
    )
    session.record({"role": "tool", "tool_call_id": "c1", "content": "ok"})

    assistant = session.get_history()[1]

    # DeepSeek thinking mode rejects a request whose assistant turn comes back
    # without the reasoning it produced.
    assert assistant["reasoning_content"] == "weighing the options"
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert "timestamp" not in assistant
