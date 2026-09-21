"""Design memory uses compact queries and durable, deferred appends."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from opendde_harness.plugin.memory.longterm.backend import _HttpMemoryAdapter
from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases


@pytest.mark.asyncio
@pytest.mark.parametrize("design_type,region", [("antibody", "antibody CDR"), ("minibinder", "binder interface")])
async def test_recall_query_excludes_candidate_dump(design_type, region):
    class RecallComplete(Exception):
        pass

    memory = SimpleNamespace(
        retrieve=AsyncMock(return_value=[]),
        retrieve_skills=AsyncMock(side_effect=RecallComplete),
    )
    phases = ProteinDesignPhases(None, None, memory)
    best = SimpleNamespace(model_dump=lambda: pytest.fail("Do not serialize candidates for keyword recall"))
    config = SimpleNamespace(target="test-target", design_type=design_type, objective_key="loss")
    for cycle in (1, 99):
        with pytest.raises(RecallComplete):
            await phases.design_cycle(config, cycle, None, [{"sequence": "A" * 10000}], best)
    queries = [call.args[1] for call in memory.retrieve.call_args_list]
    assert queries[0] == queries[1]
    assert region in queries[0] and "loss" in queries[0]
    assert len(queries[0]) < 200
    assert memory.retrieve.call_args_list == memory.retrieve_skills.call_args_list


@pytest.mark.asyncio
@pytest.mark.parametrize("app_id", ["protein-design", "opendde_harness"])
@pytest.mark.parametrize("is_final", [False, True])
async def test_design_append_defers_extraction_until_flush(app_id, is_final):
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"data": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter = _HttpMemoryAdapter("http://memory", client=client)
        await adapter.memorize("run", [{"content": "case"}], app_id=app_id, project_id="target", is_final=is_final)
    assert requests[0][0] == "/api/v2/memory/add"
    assert requests[0][1].get("defer_extraction", False) is (app_id == "protein-design")
    assert len(requests) == (2 if is_final else 1)
    if is_final:
        assert requests[1] == (
            "/api/v2/memory/flush",
            {"session_id": "run", "app_id": app_id, "project_id": "target"},
        )
