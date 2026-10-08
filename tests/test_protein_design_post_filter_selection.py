"""Bounded post-refold selection and auditable final decisions."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.core.contracts import (
    AnalyzeAgentOutput,
    Candidate,
    PostFilterAgentOutput,
    WorkflowConfig,
)
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.providers.base import LLMResponse


def candidate(index, **kwargs):
    return Candidate(
        candidate_id=f"c{index:03}",
        sequence="ACDEFGHIKL",
        objective=float(index),
        structure_path=f"{index}.pdb",
        **kwargs,
    )


def selection(ids):
    return PostFilterAgentOutput(
        strategy_summary="Interface support balanced against uncertainty",
        decisions=[
            {"candidate_id": name, "rank": rank, "rationale": "Supplied interface evidence"}
            for rank, name in enumerate(ids, 1)
        ],
    )


def test_top_k_subset_is_valid_without_ranking_unselected_candidates():
    selection(["c002"]).validate_ranking({"c001", "c002"}, top_k=1)


@pytest.mark.parametrize("ids", [["c001", "c001"], ["unknown"], ["c001", "c002"]])
def test_subset_rejects_duplicates_unknown_ids_and_wrong_count(ids):
    with pytest.raises(ValueError):
        selection(ids).validate_ranking({"c001", "c002"}, top_k=1)


@pytest.mark.parametrize("design_type", ["antibody", "minibinder"])
def test_large_pool_requests_only_top_k_with_bounded_evidence(design_type):
    class Session:
        async def run(self, profile, prompt, **kwargs):
            payload = json.loads(prompt)
            assert payload["rank_count"] == 20
            assert 20 <= len(payload["candidate_evidence"]) <= 80
            assert len(prompt) < 160_000
            assert "NaN" not in prompt and "Infinity" not in prompt
            result = selection([item["candidate_id"] for item in payload["candidate_evidence"][:20]])
            kwargs["output_validator"](result)
            return result

    config = WorkflowConfig(target="T", design_type=design_type, post_filter_top_k=20)
    candidates = [
        candidate(
            i,
            metrics={"iptm": i / 300, "ptm": float("nan")},
            metadata={
                "gate_evidence": {"raw_contacts": ["x" * 1000] * 1000},
                "quality_check": {"reasoning": "x" * 100_000},
            },
        )
        for i in range(287)
    ]
    phases = ProteinDesignPhases(Session(), None, None, catalog=SimpleNamespace(select=lambda _: []))
    output = asyncio.run(phases.post_filter_run(config, candidates, AnalyzeAgentOutput(downstream_header="test")))
    assert len(output.decisions) == 20
    assert len(candidates) == 287
    assert len(candidates[0].metadata["quality_check"]["reasoning"]) == 100_000


def test_policy_preserves_unselected_candidates_without_inventing_agent_ranks():
    candidates = [candidate(i) for i in range(100)]
    payload = DesignOrchestrator._post_filter_policy_selection(
        selection(["c000"]).model_dump(),
        eligible=candidates,
        rejected=[],
        terminal=candidates,
        config=WorkflowConfig(target="T", post_filter_top_k=1),
        post_refold_error=None,
    )
    assert payload["selected_candidate_ids"] == ["c000"]
    assert len(payload["candidates"]) == len(payload["decisions"]) == 100
    unselected = payload["decisions"][1:]
    assert all(item["hard_eligible"] and not item["pass_filter"] and item["rank"] is None for item in unselected)
    assert {item["selection_status"] for item in unselected} == {"not_selected", "not_shortlisted"}
    assert len(payload["shortlist"]["candidate_ids"]) <= 80


@pytest.mark.parametrize("candidates", [[], [candidate(0), candidate(0)]])
def test_invalid_input_fails_before_calling_provider(candidates):
    class Provider:
        async def chat_with_retry(self, **kwargs):
            raise AssertionError("invalid input must not reach provider")

    phases = ProteinDesignPhases(
        OpenDDEHarnessStructuredSession(Provider(), "test"), None, None, catalog=SimpleNamespace(select=lambda _: [])
    )
    with pytest.raises(ValueError, match="candidate"):
        asyncio.run(
            phases.post_filter_run(WorkflowConfig(target="T"), candidates, AnalyzeAgentOutput(downstream_header="test"))
        )


@pytest.mark.parametrize("max_tokens", [None, 32768])
def test_truncated_top_k_is_retried_not_accepted_as_complete(max_tokens):
    class Provider:
        def __init__(self):
            self.calls = 0

        async def chat_with_retry(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(selection(["c000"]).model_dump_json(), finish_reason="length")
            return LLMResponse(selection(["c001"]).model_dump_json())

    provider = Provider()
    phases = ProteinDesignPhases(
        OpenDDEHarnessStructuredSession(provider, "test"), None, None, catalog=SimpleNamespace(select=lambda _: [])
    )
    result = asyncio.run(
        phases.post_filter_run(
            WorkflowConfig(target="T", post_filter_top_k=1, llm_max_tokens=max_tokens),
            [candidate(0), candidate(1)],
            AnalyzeAgentOutput(downstream_header="test"),
        )
    )
    assert [item.candidate_id for item in result.decisions] == ["c001"]
    assert provider.calls == 2


def test_shortlist_keeps_metric_leaders_and_diverse_sequences_independent_of_input_order():
    from opendde_harness.plugin.protein_design.core.post_filter import shortlist_candidates

    candidates = [candidate(i) for i in range(287)]
    candidates[280].metrics = {"iptm": 0.99}
    candidates[281].metrics = {"binder_rmsd": 0.1}
    candidates[282].sequence = "YYYYYYYYYY"
    candidates[282].objective = None
    candidates[283].metrics = {"ipsae": float("nan"), "iptm": float("inf")}
    first = shortlist_candidates(candidates, top_k=20, minimize=True)
    second = shortlist_candidates(list(reversed(candidates)), top_k=20, minimize=True)
    ids = [item.candidate_id for item in first]
    assert len(ids) == len(set(ids)) == 80
    assert {"c000", "c280", "c281", "c282"} <= set(ids)
    assert ids == [item.candidate_id for item in second]


def test_shortlist_respects_objective_direction_and_never_returns_less_than_requested():
    from opendde_harness.plugin.protein_design.core.post_filter import shortlist_candidates

    candidates = [candidate(i) for i in range(287)]
    result = shortlist_candidates(candidates, top_k=100, minimize=False)
    assert len(result) == 100
    assert result[0].candidate_id == "c286"
    assert len(shortlist_candidates(candidates[:2], top_k=20, minimize=False)) == 2


@pytest.mark.parametrize("field", ["strengths", "risks"])
def test_explanations_cannot_hide_unbounded_text_in_lists(field):
    value = selection(["c001"]).model_dump()
    value["decisions"][0][field] = ["x" * 10000]
    with pytest.raises(ValueError):
        PostFilterAgentOutput.model_validate(value)


def terminal_selection(mode):
    """Exercise MPNN -> refold -> phase -> selection; only compute/LLM are fake."""
    from opendde_harness.plugin.protein_design.core.contracts import TaskSnapshot, TaskState
    from opendde_harness.plugin.protein_design.core.orchestrator import _ProgressState, _RunState
    from opendde_harness.tracing import trace

    stop = asyncio.Event()

    class Compute:
        updated = None

        async def generate_soluble_mpnn(self, request):
            return {
                "candidates": [
                    {"chains": {"B": aa + "CDEFGHIKL"}, "metadata": {"soluble_mpnn_scores": {"B": float(i)}}}
                    for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY" * 2)
                ]
            }

        async def submit_fold(self, request):
            self.candidates = request.candidates
            return SimpleNamespace(job_id="fold")

        async def wait_fold(self, *args, **kwargs):
            return SimpleNamespace(
                error=None,
                result={
                    "candidates": [
                        {
                            **item,
                            "objective": float(i),
                            "structure_path": "refold.pdb",
                            "metrics": {"iptm": 0.8},
                            "metadata": {"success": mode != "no_eligible"},
                        }
                        for i, item in enumerate(self.candidates)
                    ]
                },
            )

        async def pose_rmsd(self, request):
            return {"rmsd": 0.2}

        async def read_structure(self, *args, **kwargs):
            if mode == "stop_export" and getattr(self, "post_filter_finished", False):
                stop.set()
            raise FileNotFoundError("No physical GPU structure in this unit test")

        async def update_population(self, value):
            self.updated = value

    class Session:
        async def run(self, profile, prompt, **kwargs):
            assert mode != "no_eligible", "empty eligible pool must not reach the model"
            if mode in ("stop_error", "stop_return"):
                stop.set()
            if mode in ("error", "stop_error"):
                raise ValueError("invalid model output")
            payload = json.loads(prompt)
            result = selection([payload["candidate_evidence"][-1]["candidate_id"]])
            kwargs["output_validator"](result)
            compute.post_filter_finished = True
            return result

    compute = Compute()
    phases = ProteinDesignPhases(Session(), compute, None, catalog=SimpleNamespace(select=lambda _: []))
    config = WorkflowConfig(
        target="T",
        post_filter_enabled=True,
        post_filter_top_k=1,
        binder_chains={"B": "ACDEFGHIKL"},
        mutable_positions={"B": [0]},
    )
    state = _RunState(
        parents=[],
        snapshot=TaskSnapshot(task_id="t", status=TaskState.RUNNING, target="T", total_cycles=1),
        total_scored_candidates=1,
        search_history=[{**candidate(0).model_dump(), "gate_passed": True}],
    )
    orchestrator = DesignOrchestrator(compute, None, phases)
    orchestrator._progress_state = _ProgressState(task_id="t", total_cycles=1)
    with trace.span("test-terminal-selection") as span:
        result = asyncio.run(
            orchestrator._terminal_selection(
                state,
                "t",
                config,
                analysis=AnalyzeAgentOutput(downstream_header="test"),
                stop_event=stop,
                run_span=span,
            )
        )
    return result[1], compute.updated


@pytest.mark.parametrize("mode", ["stop_error", "stop_return", "stop_export"])
def test_stop_during_post_filter_never_publishes_completed_selection(mode):
    with pytest.raises((RuntimeError, ValueError), match="stopped|invalid model output"):
        terminal_selection(mode)


def test_terminal_agent_selection_and_fallback_are_distinct_and_auditable():
    agent, updated = terminal_selection("success")
    assert agent["mode"] == "agent"
    assert agent["selected_candidate_ids"] == ["c000__mpnn_04"]
    assert len(agent["decisions"]) == 4
    assert updated["final_selection"] == agent
    fallback, _ = terminal_selection("error")
    assert fallback["mode"] == "deterministic"
    assert fallback["selected_candidate_ids"] == ["c000__mpnn_01"]
    assert fallback["post_filter_error"] == "invalid model output"
    assert fallback["shortlist"]["candidate_ids"] == agent["shortlist"]["candidate_ids"]


def test_no_eligible_post_refold_candidates_skips_model_and_preserves_failures():
    result, _ = terminal_selection("no_eligible")
    assert result["mode"] == "failed"
    assert not result["post_filter_executed"]
    assert result["selected_candidate_ids"] == []
    assert len(result["candidates"]) == 4
    assert len(result["decisions"]) == 4
    assert all(not item["hard_eligible"] and item["rank"] is None for item in result["decisions"])


def test_deterministic_fallback_has_stable_contiguous_ranks_for_multiple_rejections():
    eligible = [candidate(0)]
    rejected = [candidate(i) for i in range(1, 4)]
    result = DesignOrchestrator._objective_final_selection(
        eligible=eligible,
        rejected=rejected,
        terminal=eligible + rejected,
        config=WorkflowConfig(target="T", post_filter_top_k=1),
        strategy_summary="Fallback",
    )
    assert [item["rank"] for item in result["decisions"]] == [1, 2, 3, 4]
