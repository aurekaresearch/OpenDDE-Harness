"""Post-filter semantic validation shares the bounded structured-output repair loop."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.core.contracts import AnalyzeAgentOutput, Candidate
from opendde_harness.providers.base import LLMResponse


def ranking(ids=("a", "b"), ranks=(1, 2)):
    return json.dumps(
        {
            "strategy_summary": "Use refold evidence",
            "decisions": [
                {"candidate_id": candidate_id, "rank": rank, "rationale": "Evidence-backed order"}
                for candidate_id, rank in zip(ids, ranks, strict=True)
            ],
        }
    )


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def chat_with_retry(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def run(provider):
    phases = ProteinDesignPhases(
        OpenDDEHarnessStructuredSession(provider, "test"),
        compute=None,
        memory=None,
        catalog=SimpleNamespace(select=lambda _: []),
    )
    config = SimpleNamespace(
        metadata={},
        target="test",
        objective_key="loss",
        minimize=True,
        post_filter_top_k=1,
        cdr_regions={},
        cycles=1,
        reflection_interval=10,
        binder_chains={"D": "ACDE"},
        llm_model=None,
        llm_max_tokens=None,
    )
    return asyncio.run(
        phases.post_filter_run(
            config,
            [Candidate(candidate_id=name, sequence="ACDE") for name in ("a", "b")],
            AnalyzeAgentOutput(downstream_header="test"),
        )
    )


@pytest.mark.parametrize(
    "invalid, detail",
    [
        (ranking(("a", "a")), "duplicate IDs: ['a']"),
        (ranking(("a",), (1,)), "missing IDs: ['b']"),
        (ranking(("a", "unknown")), "unexpected IDs: ['unknown']"),
        (ranking(ranks=(1, 1)), "unique and contiguous"),
        (ranking(ranks=(1, 3)), "unique and contiguous"),
    ],
)
def test_invalid_ranking_is_repaired_with_feedback(invalid, detail):
    provider = Provider([LLMResponse(invalid), LLMResponse(ranking())])
    output = run(provider)
    assert [item.candidate_id for item in output.decisions] == ["a", "b"]
    assert len(provider.calls) == 2
    repair_messages = json.dumps(provider.calls[1]["messages"])
    assert detail in repair_messages
    assert "complete corrected ranking" in repair_messages
    assert len(provider.calls[1]["messages"]) > len(provider.calls[0]["messages"])


def test_valid_ranking_needs_only_one_attempt():
    provider = Provider([LLMResponse(ranking())])
    run(provider)
    assert len(provider.calls) == 1
    prompt = json.dumps(provider.calls[0]["messages"])
    assert "Authoritative selection objective" in prompt
    assert "commentary cannot override it" in prompt
    assert "Your explanations and" in AGENT_PROFILES[AgentRole.POST_FILTER].system_prompt
    assert "cannot override eligibility, order, or top_k" in AGENT_PROFILES[AgentRole.POST_FILTER].system_prompt


def test_invalid_rankings_exhaust_three_attempts():
    provider = Provider([LLMResponse(ranking(("a", "a")))] * 3)
    with pytest.raises(ValueError, match="missing IDs.*b.*duplicate IDs.*a"):
        run(provider)
    assert len(provider.calls) == 3


@pytest.mark.parametrize(
    "response, exception",
    [
        (LLMResponse("provider exhausted", finish_reason="error"), RuntimeError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
def test_transport_failure_and_cancellation_are_not_output_retries(response, exception):
    provider = Provider([response])
    with pytest.raises(exception):
        run(provider)
    assert len(provider.calls) == 1


def test_other_sessions_do_not_require_a_semantic_validator():
    provider = Provider([LLMResponse(ranking())])
    output = asyncio.run(
        OpenDDEHarnessStructuredSession(provider, "test").run(AGENT_PROFILES[AgentRole.POST_FILTER], "rank candidates")
    )
    assert len(output.decisions) == 2
