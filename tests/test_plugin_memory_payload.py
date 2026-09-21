"""Memory HTTP payloads preserve tool calls and isolate rejected writes."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from opendde_harness.plugin.memory.longterm.backend import LongTermMemoryBackend, ServiceState


@pytest.mark.parametrize("arguments", [{"sequence": "蛋白", "options": [1, True]}, {}, None, '{"n": 1}'])
def test_tool_calls_use_nested_function_and_json_arguments(arguments):
    messages = [
        {"role": "assistant", "content": [{"type": "toolCall", "id": "c1", "name": "fold", "arguments": arguments}]},
        {"role": "toolResult", "toolCallId": "c1", "content": "done"},
    ]
    payload = LongTermMemoryBackend._convert_messages(messages, agent_id="agent")
    call = payload[0]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["type"] == "function"
    assert "name" not in call
    assert call["function"]["name"] == "fold"
    expected = json.loads(arguments) if isinstance(arguments, str) else arguments or {}
    assert json.loads(call["function"]["arguments"]) == expected
    if isinstance(arguments, str):
        assert call["function"]["arguments"] == arguments
    assert payload[0]["content"] == ""
    assert payload[1]["tool_call_id"] == "c1"


def test_plain_messages_and_multiple_calls_are_preserved():
    payload = LongTermMemoryBackend._convert_messages(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "working"},
                    {"type": "toolCall", "id": "a", "name": "first", "arguments": {}},
                    {"type": "toolCall", "id": "b", "name": "second", "arguments": {"x": 2}},
                ],
            },
        ],
        agent_id="agent",
        user_id="user",
    )
    assert len(payload) == 2
    assert payload[0]["sender_id"] == "user"
    assert "tool_calls" not in payload[0]
    assert payload[1]["content"] == "working"
    assert [call["id"] for call in payload[1]["tool_calls"]] == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [422, 500])
async def test_rejected_payload_does_not_disable_subsequent_writes(status, caplog):
    response = httpx.Response(status, request=httpx.Request("POST", "http://memory/api/v2/memory/add"))
    error = httpx.HTTPStatusError("private input", request=response.request, response=response)
    adapter = SimpleNamespace(memorize=AsyncMock(side_effect=[error, None]))
    ctx = SimpleNamespace(
        config={},
        services=SimpleNamespace(agent_id="agent", user_id="user"),
        logger=logging.getLogger("test.memory.payload"),
    )
    backend = LongTermMemoryBackend(ctx, adapter=adapter)
    messages = [{"role": "user", "content": "hello"}]
    assert await backend.store("session", messages) is False
    assert backend._dropped_writes == 1
    if status == 422:
        assert backend._state is ServiceState.READY
        assert "invalid memory payload" in caplog.text
        assert "private input" not in caplog.text
        assert await backend.store("session", messages) is True
        assert adapter.memorize.await_count == 2
    else:
        assert backend._state is ServiceState.UNRESPONSIVE
