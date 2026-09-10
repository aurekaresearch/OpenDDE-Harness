import os

import pytest

from opendde_harness.providers.litellm_provider import LiteLLMProvider

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {}}}}]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ZAI_API_KEY", "ZHIPUAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_api_key_is_not_exported_to_the_environment(clean_env):
    LiteLLMProvider(api_key="sk-a", default_model="anthropic/claude-sonnet-4-5", provider_name="anthropic")
    LiteLLMProvider(api_key="sk-o", default_model="openai/gpt-4o", provider_name="openai")

    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_env_extras_are_still_exported(clean_env):
    LiteLLMProvider(api_key="sk-z", default_model="zai/glm-4.5", provider_name="zai")

    assert os.environ.get("ZHIPUAI_API_KEY") == "sk-z"
    assert "ZAI_API_KEY" not in os.environ


def _kwargs(provider, *, responses, stream, tools=TOOLS, reasoning_effort=None):
    model = provider._resolve_model(provider.default_model)
    kwargs, fallback = provider._request_kwargs(
        provider.default_model,
        model,
        MESSAGES,
        tools,
        max_tokens=0,
        temperature=0.2,
        reasoning_effort=reasoning_effort,
        tool_choice=None,
        responses=responses,
        stream=stream,
    )
    assert fallback is None
    return kwargs


def test_chat_wire_kwargs(clean_env):
    provider = LiteLLMProvider(
        api_key="sk", api_base="http://gw", default_model="openai/gpt-5.5", extra_headers={"X": "1"}, wire="chat"
    )
    kwargs = _kwargs(provider, responses=False, stream=False, reasoning_effort="low")

    assert kwargs["messages"] == MESSAGES
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["max_tokens"] == 1
    assert kwargs["api_key"] == "sk"
    assert kwargs["api_base"] == "http://gw"
    assert kwargs["extra_headers"] == {"X": "1"}
    assert kwargs["reasoning_effort"] == "low"
    assert "stream" not in kwargs and "input" not in kwargs


def test_stream_kwargs_request_usage(clean_env):
    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4o", wire="chat")
    kwargs = _kwargs(provider, responses=False, stream=True)

    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


def test_responses_wire_kwargs(clean_env):
    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4o", wire="responses")
    kwargs = _kwargs(provider, responses=True, stream=True, reasoning_effort="high")

    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert kwargs["store"] is False
    assert kwargs["stream"] is True
    assert kwargs["max_output_tokens"] == 1
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["tools"] == [
        {"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {}}}
    ]
    assert kwargs["tool_choice"] == "auto"
    assert "messages" not in kwargs and "stream_options" not in kwargs


def test_streamed_thinking_blocks_keep_their_signature(clean_env):
    """Anthropic wants the original signed blocks back during a tool-use turn."""
    from opendde_harness.agent.loop.main import _merge_thinking_blocks

    blocks: list[dict] = []
    _merge_thinking_blocks(blocks, [{"type": "thinking", "thinking": "weighing "}])
    _merge_thinking_blocks(blocks, [{"type": "thinking", "thinking": "the options"}])
    _merge_thinking_blocks(blocks, [{"type": "thinking", "signature": "sig-abc"}])

    assert blocks == [{"type": "thinking", "thinking": "weighing the options", "signature": "sig-abc"}]


def test_streamed_tool_calls_keep_provider_signature_fields():
    from opendde_harness.agent.loop.main import _finalize_tool_calls, _merge_tool_call_fragments

    slots: list[dict] = []
    _merge_tool_call_fragments(
        slots,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "provider_specific_fields": {"thought_signature": "sig-1"},
                    "function": {"name": "read_file", "arguments": '{"path"'},
                }
            ]
        },
    )
    _merge_tool_call_fragments(slots, {"tool_calls": [{"index": 0, "function": {"arguments": ': "a.txt"}'}}]})

    call = _finalize_tool_calls(slots)[0]

    assert call.arguments == {"path": "a.txt"}
    assert call.provider_specific_fields == {"thought_signature": "sig-1"}
    assert call.to_openai_tool_call()["provider_specific_fields"] == {"thought_signature": "sig-1"}


def test_tool_call_ids_survive_except_where_a_backend_demands_short_ones():
    from opendde_harness.providers.litellm_provider import _tool_call_id

    # Gemini's thought signature travels inside the id LiteLLM hands over.
    assert _tool_call_id("call_abc123.thought_sig_xyz", "gemini/gemini-3-pro") == "call_abc123.thought_sig_xyz"
    assert len(_tool_call_id("call_abc123", "mistral/mistral-large-latest")) == 9
    assert len(_tool_call_id(None, "openai/gpt-5")) == 9


def test_qwen_gets_dashscope_thinking_switch_instead_of_an_effort(clean_env):
    """DashScope has no reasoning_effort; its depth travels as a thinking budget."""
    provider = LiteLLMProvider(api_key="sk", default_model="dashscope/qwen-plus", wire="chat")
    kwargs = _kwargs(provider, responses=False, stream=False, tools=None, reasoning_effort="high")

    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"] == {"enable_thinking": True, "thinking_budget": 16384}


def test_redacted_thinking_stays_its_own_block():
    from opendde_harness.agent.loop.main import _merge_thinking_blocks

    blocks: list[dict] = []
    _merge_thinking_blocks(blocks, [{"type": "thinking", "thinking": "first "}])
    _merge_thinking_blocks(blocks, [{"type": "redacted_thinking", "data": "opaque"}])
    _merge_thinking_blocks(blocks, [{"type": "thinking", "thinking": "second", "signature": "sig"}])

    assert blocks == [
        {"type": "thinking", "thinking": "first second", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]


def test_replayed_non_streaming_answer_keeps_tool_signatures():
    """A provider without real streaming is replayed as one delta; it must not lose fields."""
    import asyncio
    from typing import Any

    from opendde_harness.providers.base import LLMProvider, LLMResponse, ToolCallRequest

    class Replayed(LLMProvider):
        async def chat(self, **_: Any) -> LLMResponse:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "a.txt"},
                        provider_specific_fields={"thought_signature": "sig-1"},
                    )
                ],
            )

        def get_default_model(self) -> str:
            return "fake/model"

    async def collect() -> list[Any]:
        return [delta async for delta in Replayed().chat_stream(messages=MESSAGES)]

    deltas = asyncio.run(collect())
    call = deltas[0].tool_call_delta["tool_calls"][0]

    assert call["provider_specific_fields"] == {"thought_signature": "sig-1"}


def test_deepseek_always_gets_the_reasoning_key(clean_env):
    """Thinking mode refuses an assistant turn with no reasoning_content key at all."""
    provider = LiteLLMProvider(api_key="sk", default_model="deepseek/deepseek-v4-flash", wire="chat")
    history = [
        {"role": "user", "content": "status?"},
        # Recorded before the fix, and after a tool result the model returns no
        # reasoning of its own: the key was simply absent.
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [{"id": "c1", "function": {"name": "s", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "running"},
    ]

    kwargs, _ = provider._request_kwargs(
        provider.default_model,
        provider._resolve_model(provider.default_model),
        history,
        None,
        max_tokens=100,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
        responses=False,
        stream=False,
    )
    assistant = [m for m in kwargs["messages"] if m["role"] == "assistant"]

    assert all("reasoning_content" in m for m in assistant)


def test_other_vendors_keep_their_messages_untouched(clean_env):
    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-5", wire="chat")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    kwargs, _ = provider._request_kwargs(
        provider.default_model,
        provider._resolve_model(provider.default_model),
        history,
        None,
        max_tokens=100,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
        responses=False,
        stream=False,
    )

    assert all("reasoning_content" not in m for m in kwargs["messages"])


# ---------------------------------------------------------------------------
# Wire selection: declared per section and per model, never probed, and a
# wrong wire is reported as configuration rather than routed around.
# ---------------------------------------------------------------------------


class _Refused(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _refusing(status: int, message: str):
    async def call(**kwargs):
        raise _Refused(status, message)

    return call


def test_relay_defaults_to_chat_and_openai_to_responses():
    from opendde_harness.providers.registry import default_wire

    assert default_wire("custom") == "chat"
    assert default_wire("aihubmix") == "chat"
    assert default_wire("siliconflow") == "chat"
    assert default_wire("openai") == "responses"
    assert default_wire("a-vendor-with-no-spec") == "chat"


def test_wire_is_the_sections_unless_the_model_declares_its_own(clean_env):
    from opendde_harness.config.schema import ModelOverlay
    from opendde_harness.providers.wire import merge_key

    overlays = {merge_key("custom", "gpt-5.6-terra"): ModelOverlay(wire="responses")}
    provider = LiteLLMProvider(
        api_key="sk", api_base="http://relay", provider_name="custom", wire="chat", model_overlays=overlays
    )

    assert provider.wire_for("custom/gpt-5.6-terra") == "responses"
    assert provider.wire_for("custom/qwen3-max") == "chat"
    assert provider._uses_responses_api("custom/gpt-5.6-terra", "openai/gpt-5.6-terra")
    assert not provider._uses_responses_api("custom/qwen3-max", "openai/qwen3-max")


def test_responses_only_exists_on_the_openai_driver(clean_env):
    provider = LiteLLMProvider(api_key="sk", provider_name="deepseek", wire="responses")

    # Declared, but the id routes through another driver: ignored and said once.
    assert not provider._uses_responses_api("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash")


async def test_relay_without_responses_gets_a_configuration_error_not_a_model_hop(clean_env, monkeypatch):
    from opendde_harness.providers import litellm_provider as module

    calls = []

    async def refuse(**kwargs):
        calls.append("responses")
        raise _Refused(404, "404 page not found")

    monkeypatch.setattr(module, "aresponses", refuse)
    monkeypatch.setattr(module, "acompletion", _refusing(500, "must not be reached"))
    provider = LiteLLMProvider(api_key="sk", api_base="http://relay/v1", provider_name="custom", wire="responses")

    first = await provider.chat(MESSAGES, model="custom/gpt-5.6-terra")
    second = await provider.chat(MESSAGES, model="custom/gpt-5.6-terra")

    # A bare 404 is the same body for a missing /v1, a wire the endpoint does
    # not serve, and an unknown model: named as all three, base URL first.
    assert first.finish_reason == "error"
    assert first.error_classification.category == "endpoint_not_found"
    assert not first.error_classification.retryable and not first.error_classification.should_fallback
    assert "endpoint_not_found@custom" in first.content
    assert "without naming a route or a model" in first.content
    assert "providers.custom.apiBase; a missing /v1" in first.content
    assert 'providers.custom.wire to "chat"' in first.content
    assert "provider model set custom custom/gpt-5.6-terra --wire chat" in first.content
    # No learned downgrade: the second call goes out on the declared wire again.
    assert calls == ["responses", "responses"]
    assert second.error_classification.category == "endpoint_not_found"


async def test_a_route_named_in_the_body_is_the_wire_with_certainty(clean_env, monkeypatch):
    from opendde_harness.providers import litellm_provider as module

    monkeypatch.setattr(module, "aresponses", _refusing(404, "Cannot POST /v1/responses"))
    provider = LiteLLMProvider(api_key="sk", api_base="http://relay/v1", provider_name="custom", wire="responses")

    response = await provider.chat(MESSAGES, model="custom/gpt-5.6-terra")

    assert response.error_classification.category == "wire_mismatch"
    assert "http://relay/v1 does not serve the responses wire" in response.content
    assert "provider set custom --wire chat" in response.content


@pytest.mark.parametrize(
    "body",
    ["The model `gpt-9` does not exist", "OpenrouterException - No endpoints found for qwen/qwen3-max"],
)
async def test_a_404_naming_the_model_is_still_the_model(clean_env, monkeypatch, body):
    from opendde_harness.providers import litellm_provider as module

    monkeypatch.setattr(module, "aresponses", _refusing(404, body))
    provider = LiteLLMProvider(api_key="sk", provider_name="openai", wire="responses")

    response = await provider.chat(MESSAGES, model="openai/gpt-9")

    assert response.error_classification.category == "model_unavailable"


@pytest.mark.parametrize("body", ["404 page not found", '{"detail":"Not Found"}', "", "<html><body>404</body></html>"])
async def test_a_bare_404_on_the_chat_wire_names_all_three_causes_base_url_first(clean_env, monkeypatch, body):
    # nginx, Caddy and Go answer any unknown path this way, so a base URL
    # missing its /v1 produces it too; telling the user to change the wire
    # with certainty would send them past the real fault.
    from opendde_harness.providers import litellm_provider as module
    from opendde_harness.providers.base import EndpointNotFoundError

    monkeypatch.setattr(module, "acompletion", _refusing(404, body))
    provider = LiteLLMProvider(api_key="sk", api_base="http://relay", provider_name="custom", wire="chat")

    with pytest.raises(EndpointNotFoundError) as caught:
        async for _ in provider.chat_stream(MESSAGES, model="custom/qwen3-max"):
            pass

    text = str(caught.value)
    assert text.index("apiBase") < text.index("model id") < text.index("--wire responses")
    assert provider.classify_error(caught.value).category == "endpoint_not_found"


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (405, "Method Not Allowed"),
        (404, "Route POST:/v1/chat/completions not found"),
        (404, "Cannot POST /v1/chat/completions"),
    ],
)
async def test_streaming_reports_the_wire_too(clean_env, monkeypatch, status, body):
    from opendde_harness.providers import litellm_provider as module
    from opendde_harness.providers.base import WireMismatchError

    monkeypatch.setattr(module, "acompletion", _refusing(status, body))
    provider = LiteLLMProvider(api_key="sk", api_base="http://relay/v1", provider_name="custom", wire="chat")

    with pytest.raises(WireMismatchError) as caught:
        async for _ in provider.chat_stream(MESSAGES, model="custom/gpt-5.6-terra"):
            pass

    assert "does not serve the chat wire" in str(caught.value)
    assert 'wire to "responses"' in str(caught.value)
    assert provider.classify_error(caught.value).category == "wire_mismatch"


async def test_streaming_on_the_chat_wire_leaves_a_404_with_prose_alone(clean_env, monkeypatch):
    from opendde_harness.providers import litellm_provider as module

    monkeypatch.setattr(module, "acompletion", _refusing(404, "No endpoints found for qwen3-max"))
    provider = LiteLLMProvider(api_key="sk", api_base="http://relay/v1", provider_name="custom", wire="chat")

    with pytest.raises(_Refused):
        async for _ in provider.chat_stream(MESSAGES, model="custom/qwen3-max"):
            pass


def test_anthropic_gets_only_its_own_thinking_blocks_back():
    # A session that switched from Codex still carries that provider's
    # reasoning block on earlier assistant turns; sent to Anthropic verbatim it
    # is refused, since Anthropic accepts only the blocks it signed.
    messages = [
        {
            "role": "assistant",
            "content": "plan",
            "thinking_blocks": [
                {"type": "reasoning", "provider": "openai_codex", "thinking": "plan A", "items": []},
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            ],
        },
        {"role": "assistant", "content": "x", "thinking_blocks": [{"type": "reasoning", "provider": "openai_codex"}]},
    ]

    keep = LiteLLMProvider._keeps_thinking_blocks(
        "anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5", responses=False
    )
    clean = LiteLLMProvider._sanitize_messages(messages, keep_blocks=keep)

    assert clean[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "hmm", "signature": "sig"}]
    assert "thinking_blocks" not in clean[1]


def test_a_stream_reads_under_the_first_token_bound_and_a_call_under_the_wall_clock(clean_env):
    from opendde_harness.providers.base import GenerationSettings

    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4.1-mini", wire="chat")
    provider.generation = GenerationSettings(timeout=600, first_token_timeout=300, idle_timeout=120)

    assert _kwargs(provider, responses=False, stream=True)["timeout"] == 300
    assert _kwargs(provider, responses=False, stream=False)["timeout"] == 600


def test_a_responses_function_call_keeps_its_item_id_from_parse_to_replay(clean_env):
    """The id the backend pairs reasoning with must survive the history and
    the request sanitizer, or the replayed pair is refused."""
    from opendde_harness.utils.helpers import build_assistant_message

    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-5.5", wire="responses")
    response = provider._parse_responses_response(
        {
            "status": "completed",
            "output_text": "",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "{}"},
            ],
        }
    )
    assert response.tool_calls[0].id == "call_1|fc_1"

    history = [
        {"role": "user", "content": "hi"},
        build_assistant_message(
            "", [tc.to_openai_tool_call() for tc in response.tool_calls], None, response.thinking_blocks
        ),
        {"role": "tool", "tool_call_id": response.tool_calls[0].id, "content": "x"},
    ]
    kwargs, _ = provider._request_kwargs(
        provider.default_model,
        provider._resolve_model(provider.default_model),
        history,
        TOOLS,
        max_tokens=None,
        temperature=0.2,
        reasoning_effort=None,
        tool_choice=None,
        responses=True,
        stream=False,
    )

    kinds = [(item.get("type"), item.get("id"), item.get("call_id")) for item in kwargs["input"]]
    assert kinds[1:] == [
        ("reasoning", "rs_1", None),
        ("function_call", "fc_1", "call_1"),
        ("function_call_output", None, "call_1"),
    ]


def test_a_declared_output_ceiling_bounds_an_unpinned_request(clean_env):
    from opendde_harness.config.schema import ModelOverlay
    from opendde_harness.providers.wire import merge_key

    provider = LiteLLMProvider(
        api_key="sk",
        default_model="custom/local-model",
        wire="chat",
        provider_name="custom",
        model_overlays={merge_key("custom", "local-model"): ModelOverlay(maxOutputTokens=512)},
    )

    assert provider._declared_ceiling("custom/local-model", None) == 512
    assert provider._declared_ceiling("custom/local-model", 64) == 64
    assert provider._declared_ceiling("custom/local-model", 4096) == 512
    assert provider._declared_ceiling("custom/other-model", None) is None


async def test_the_first_token_budget_covers_opening_the_stream(clean_env, monkeypatch):
    import asyncio

    from opendde_harness.providers.base import GenerationSettings

    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4.1-mini", wire="chat")
    provider.generation = GenerationSettings(timeout=600, first_token_timeout=0.05, idle_timeout=600)

    async def slow_open(**kwargs):
        await asyncio.sleep(0.2)

    monkeypatch.setattr("opendde_harness.providers.litellm_provider.acompletion", slow_open)

    with pytest.raises(asyncio.TimeoutError):
        async for _ in provider.chat_stream([{"role": "user", "content": "hi"}]):
            pass


async def test_a_temperature_retry_spends_the_same_first_token_budget(clean_env, monkeypatch):
    import asyncio

    from opendde_harness.providers.base import GenerationSettings

    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4.1-mini", wire="chat")
    provider.generation = GenerationSettings(timeout=600, first_token_timeout=0.12, idle_timeout=600)
    attempts = []

    async def open_slowly(**kwargs):
        attempts.append(kwargs.get("temperature"))
        await asyncio.sleep(0.08)
        if len(attempts) == 1:
            raise RuntimeError("unsupported parameter: temperature does not support this model")
        await asyncio.sleep(0.08)

    monkeypatch.setattr("opendde_harness.providers.litellm_provider.acompletion", open_slowly)

    with pytest.raises(asyncio.TimeoutError):
        async for _ in provider.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert len(attempts) == 2


def test_a_models_overlay_effort_beats_the_global_default(clean_env):
    from opendde_harness.config.schema import ModelOverlay
    from opendde_harness.providers.base import GenerationSettings
    from opendde_harness.providers.wire import merge_key

    provider = LiteLLMProvider(
        api_key="sk",
        default_model="custom/deep-thinker",
        wire="chat",
        provider_name="custom",
        model_overlays={merge_key("custom", "deep-thinker"): ModelOverlay(reasoningEffort="high")},
    )
    provider.generation = GenerationSettings(reasoning_effort="low")

    assert provider.effort_for("custom/deep-thinker") == "high"
    assert provider.effort_for("custom/other") == "low"
    provider.generation = GenerationSettings()
    assert provider.effort_for("custom/other") is None
