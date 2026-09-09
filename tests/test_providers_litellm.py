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
        api_key="sk", api_base="http://gw", default_model="openai/gpt-4o", extra_headers={"X": "1"}, api_mode="chat"
    )
    kwargs = _kwargs(provider, responses=False, stream=False, reasoning_effort="low")

    assert kwargs["messages"] == MESSAGES
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["max_tokens"] == 1
    assert kwargs["api_key"] == "sk"
    assert kwargs["api_base"] == "http://gw"
    assert kwargs["extra_headers"] == {"X": "1"}
    assert kwargs["reasoning_effort"] == "low" and kwargs["drop_params"] is True
    assert "stream" not in kwargs and "input" not in kwargs


def test_stream_kwargs_request_usage(clean_env):
    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4o", api_mode="chat")
    kwargs = _kwargs(provider, responses=False, stream=True)

    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


def test_responses_wire_kwargs(clean_env):
    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-4o", api_mode="responses")
    kwargs = _kwargs(provider, responses=True, stream=True, reasoning_effort="high")

    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert kwargs["store"] is False
    assert kwargs["stream"] is True
    assert kwargs["max_output_tokens"] == 1
    assert kwargs["reasoning"] == {"effort": "high"}
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
    """DashScope has no reasoning_effort, and drop_params removed it silently."""
    provider = LiteLLMProvider(api_key="sk", default_model="dashscope/qwen3-max", api_mode="chat")
    kwargs = _kwargs(provider, responses=False, stream=False, tools=None, reasoning_effort="high")

    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"] == {"enable_thinking": True}


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
    provider = LiteLLMProvider(api_key="sk", default_model="deepseek/deepseek-v4-flash", api_mode="chat")
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
    provider = LiteLLMProvider(api_key="sk", default_model="openai/gpt-5", api_mode="chat")
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
