"""Tool-bearing turns retain the memory API's nested function wire contract."""

from __future__ import annotations

import json

import httpx
import pytest

from opendde_harness.plugin import PluginContext, ServiceLocator
from opendde_harness.plugin.memory.longterm.backend import LongTermMemoryBackend, _HttpMemoryAdapter
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.providers import messages as msg


def test_memory_conversion_keeps_tool_only_messages_and_json_arguments():
    arguments = {"target": "PD-L1", "description": "interface ΔG", "positions": [1, 2]}
    messages = [
        msg.assistant_message("", tool_calls=[msg.tool_call_block("call-1", "score", arguments)]),
        msg.tool_result_message("call-1", "score", "scored"),
    ]

    converted = LongTermMemoryBackend._convert_messages(messages, agent_id="design-agent")

    assert len(converted) == 2
    assert converted[0]["content"] == ""
    call = converted[0]["tool_calls"][0]
    assert set(call) == {"id", "type", "function"}
    assert call["id"] == "call-1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "score"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == arguments
    assert converted[1]["tool_call_id"] == call["id"]
    assert converted[1]["role"] == "tool"
    assert all(message["sender_id"] == "design-agent" for message in converted)


@pytest.mark.asyncio
async def test_design_memory_posts_valid_tool_calls_before_final_flush(tmp_path):
    requests = []
    target = "human_PD-L1_CD274_ectodomain"

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/add"):
            call = body["messages"][1]["tool_calls"][0]
            # The memory API requires function.name and JSON-string arguments;
            # a top-level name was rejected with HTTP 422 for every design case.
            if "function" not in call or not isinstance(call["function"].get("arguments"), str):
                return httpx.Response(422, json={"detail": "tool_calls.function is required"})
        return httpx.Response(200, json={"data": {"status": "extracted"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        backend = LongTermMemoryBackend(
            PluginContext(
                config={},
                services=ServiceLocator(workspace=tmp_path, user_id="user-1", agent_id="design-agent"),
            ),
            adapter=_HttpMemoryAdapter("http://memory.test", client=client),
        )
        memory = DesignMemory(backend, agent_id="design-agent")
        stored = await memory.record(
            "task-1",
            1,
            target,
            {"selected_skill_id": "protein-design-proposal", "candidate_ids": ["candidate-1"]},
            ["The candidate was scored."],
        )

    assert stored is True
    assert [path for path, _ in requests] == ["/api/v2/memory/add", "/api/v2/memory/flush"]
    add = requests[0][1]
    assert (add["session_id"], add["app_id"], add["project_id"]) == ("task-1", "protein-design", target)
    assistant, tool_result = add["messages"][1:3]
    call = assistant["tool_calls"][0]
    assert call["function"]["name"] == "protein-design-proposal"
    assert json.loads(call["function"]["arguments"]) == {
        "target": target,
        "cycle": 1,
        "applied_learned_skill_ids": [],
    }
    assert tool_result["tool_call_id"] == call["id"] == "design-cycle-1"
    assert json.loads(tool_result["content"])["candidate_ids"] == ["candidate-1"]
    assert backend._dropped_writes == 0
