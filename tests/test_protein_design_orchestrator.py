"""Unit tests for protein-design orchestration, session budgets and shared predicates."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from opendde_harness.plugin.protein_design.core.constants import (
    CANONICAL_AMINO_ACIDS,
    is_canonical_sequence,
    is_materialized,
)
from opendde_harness.plugin.protein_design.core.contracts import (
    Candidate,
    JobResult,
    JobState,
    JobSubmission,
    PostFilterAgentOutput,
    QualityBatchOutput,
    WorkflowConfig,
)
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.plugin.protein_design.core.orchestrator import (
    POST_REFOLD_POOL_MULTIPLIER,
    DesignOrchestrator,
)

BINDER_SEQUENCE = "ACDEFGHIKL"


def make_config(**overrides: Any) -> WorkflowConfig:
    values: dict[str, Any] = {
        "target": "TGT",
        "cycles": 2,
        "candidates_per_cycle": 2,
        "objective_key": "loss",
        "minimize": True,
        "reflection_interval": 100,
        "initial_candidates": [
            {
                "candidate_id": "seed",
                "chains": {"D": BINDER_SEQUENCE},
                "sequence": BINDER_SEQUENCE,
                "metadata": {"chains": {"D": BINDER_SEQUENCE}},
            }
        ],
        "target_sequence": "MKTAYIAKQR",
        "target_chains": {"A": "MKTAYIAKQR"},
        "target_chain_ids": ["A"],
        "binder_chains": {"D": BINDER_SEQUENCE},
        "fixed_residues": {"D": []},
        "cdr_regions": {"D": [0, 1, 2]},
        "mutable_positions": {"D": [0, 1, 2]},
        "population_size": 8,
        "quality_check_enabled": False,
        "post_filter_enabled": True,
        "post_filter_top_k": 1,
    }
    values.update(overrides)
    return WorkflowConfig(**values)


def make_candidate(candidate_id: str, objective: float, *, gate: bool = True) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sequence=BINDER_SEQUENCE,
        objective=objective,
        metrics={"loss": objective, "gate_passed": 1.0 if gate else 0.0},
        structure_path=f"/tmp/{candidate_id}.cif",
        metadata={
            "chains": {"D": BINDER_SEQUENCE},
            "success": True,
            "gate_passed": gate,
            "post_refold_success": True,
            "post_refold_quality_passed": True,
        },
    )


class FakeCompute:
    """Minimal stand-in for ProteinDesignComputeClient."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.population_updates: list[dict[str, Any]] = []
        self.fold_batches: list[list[dict[str, Any]]] = []
        self._counter = 0

    async def submit_fold(self, request: Any) -> JobSubmission:
        self._counter += 1
        job_id = f"job-{self._counter}"
        payload = request.model_dump()
        self.fold_batches.append(payload["candidates"])
        scored = []
        for index, item in enumerate(payload["candidates"]):
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "success": True,
                    "gate_passed": True,
                    "chains": metadata.get("chains") or item.get("chains") or {"D": BINDER_SEQUENCE},
                }
            )
            objective = float(self._counter) + index / 100.0
            scored.append(
                {
                    "candidate_id": item.get("candidate_id") or f"c{self._counter}_{index}",
                    "sequence": item.get("sequence") or BINDER_SEQUENCE,
                    "objective": objective,
                    "metrics": {"loss": objective, "gate_passed": 1.0},
                    "structure_path": f"/tmp/{job_id}_{index}.cif",
                    "metadata": metadata,
                }
            )
        self.jobs[job_id] = {"candidates": scored}
        return JobSubmission(job_id=job_id, status=JobState.QUEUED)

    async def wait_fold(self, job_id: str, **_: Any) -> JobResult:
        return JobResult(job_id=job_id, status=JobState.SUCCEEDED, result=self.jobs[job_id])

    async def update_population(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.population_updates.append(payload)
        return {}

    async def pose_rmsd(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"available": False}

    async def read_structure(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}

    async def generate_soluble_mpnn(self, request: Any) -> dict[str, Any]:
        return {"candidates": []}


class FakePhases:
    def __init__(self) -> None:
        self.post_filter_calls: list[list[Candidate]] = []
        self.post_filter_output: PostFilterAgentOutput | Exception | None = None

    async def analyze_once(self, config: WorkflowConfig) -> dict[str, Any]:
        return {"summary": "analysis"}

    async def select_parent(self, config, cycle, candidates):
        return candidates[0]

    async def design_cycle(self, config, cycle, analysis, parents, best):
        from opendde_harness.plugin.protein_design.agents.phases import DesignCycleResult

        parent = parents[0]
        chains = parent.get("chains") or {"D": BINDER_SEQUENCE}
        fold_candidates = [
            {
                "candidate_id": f"c{cycle}_{index}",
                "sequence": "".join(chains.values()),
                "chains": dict(chains),
                "metadata": {"chains": dict(chains), "parent_id": parent.get("candidate_id")},
            }
            for index in range(config.candidates_per_cycle)
        ]
        return DesignCycleResult(
            fold_candidates=fold_candidates,
            selected_skill_id="cdr-point-mutation",
            memories=[],
        )

    async def quality_cycle(self, config, cycle, candidates, analysis) -> QualityBatchOutput:
        return QualityBatchOutput(results={})

    async def reflect_cycle(self, *args: Any, **kwargs: Any):
        raise AssertionError("reflection is disabled in these tests")

    async def post_filter_run(self, config, candidates, analysis) -> PostFilterAgentOutput:
        self.post_filter_calls.append(list(candidates))
        if isinstance(self.post_filter_output, Exception):
            raise self.post_filter_output
        if self.post_filter_output is not None:
            return self.post_filter_output
        return PostFilterAgentOutput(
            strategy_summary="fake",
            decisions=[
                {"candidate_id": candidate.candidate_id, "rank": rank, "rationale": "ok"}
                for rank, candidate in enumerate(candidates, start=1)
            ],
        )


def make_orchestrator(phases: FakePhases | None = None) -> tuple[DesignOrchestrator, FakeCompute, FakePhases]:
    compute = FakeCompute()
    phases = phases or FakePhases()
    orchestrator = DesignOrchestrator(
        compute, DesignMemory(None, agent_id="test-agent"), phases, fold_poll_interval=0.0
    )
    return orchestrator, compute, phases


# --- shared materialization predicate -------------------------------------------------


def test_canonical_alphabet_has_twenty_letters() -> None:
    assert len(CANONICAL_AMINO_ACIDS) == 20
    assert "X" not in CANONICAL_AMINO_ACIDS


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [(BINDER_SEQUENCE, True), (BINDER_SEQUENCE.lower(), True), ("ACDXFG", False), ("", False), (None, False)],
)
def test_is_canonical_sequence(sequence: Any, expected: bool) -> None:
    assert is_canonical_sequence(sequence) is expected


def test_is_materialized_accepts_candidate_and_raw_mapping() -> None:
    candidate = make_candidate("c1", 1.0)
    assert is_materialized(candidate) is True
    assert is_materialized({"chains": {"D": BINDER_SEQUENCE}}) is True
    assert is_materialized({"sequence": BINDER_SEQUENCE}) is True


def test_is_materialized_rejects_masked_sequences() -> None:
    masked = make_candidate("c1", 1.0)
    masked.metadata["chains"] = {"D": "ACDXFGHIKL"}
    assert is_materialized(masked) is False
    assert is_materialized({"chains": {"D": "ACDXFGHIKL"}}) is False
    assert is_materialized({"chains": {}, "sequence": ""}) is False


def test_is_materialized_requires_every_chain() -> None:
    assert is_materialized({"chains": {"D": BINDER_SEQUENCE, "E": "XXXX"}}) is False


# --- terminal refold pool -------------------------------------------------------------


def history_record(candidate_id: str, objective: float | None, gate: bool | None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "parent_id": None,
        "cycle": 0,
        "objective": objective,
        "objective_key": "loss",
        "minimize": True,
        "metrics": {},
        "mutations": [],
        "skill_id": "cdr-point-mutation",
        "gate_passed": gate,
        "gate_reason": None,
        "population_action": "retained",
        "sequence": BINDER_SEQUENCE,
        "chains": {"D": BINDER_SEQUENCE},
        "structure_path": f"/tmp/{candidate_id}.cif",
    }


def test_terminal_refold_pool_drops_unscored_and_gate_failures() -> None:
    config = make_config(post_filter_top_k=10)
    history = [
        history_record("kept", 1.0, True),
        history_record("gate_failed", 0.5, False),
        history_record("gate_unknown", 0.5, None),
        history_record("unscored", None, True),
    ]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    assert [item["candidate_id"] for item in pool] == ["kept"]


def test_terminal_refold_pool_is_capped_and_ordered_best_first() -> None:
    config = make_config(post_filter_top_k=2)
    history = [history_record(f"c{index}", float(100 - index), True) for index in range(50)]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    assert len(pool) == POST_REFOLD_POOL_MULTIPLIER * config.post_filter_top_k
    objectives = [item["objective"] for item in pool]
    assert objectives == sorted(objectives)
    assert objectives[0] == 51.0


def test_terminal_refold_pool_orders_by_objective_when_maximizing() -> None:
    config = make_config(post_filter_top_k=2, minimize=False)
    history = [history_record(f"c{index}", float(index), True) for index in range(50)]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    objectives = [item["objective"] for item in pool]
    assert objectives == sorted(objectives, reverse=True)


def test_terminal_refold_pool_deduplicates_by_candidate_id() -> None:
    config = make_config(post_filter_top_k=10)
    history = [history_record("dup", 1.0, True), history_record("dup", 2.0, True)]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    assert len(pool) == 1
    assert pool[0]["objective"] == 2.0


def test_terminal_refold_pool_marks_children_unrefolded() -> None:
    config = make_config(post_filter_top_k=1)
    pool = DesignOrchestrator._terminal_refold_pool([history_record("kept", 1.0, True)], config)
    assert pool[0]["metadata"]["post_refold_success"] is False
    assert pool[0]["metadata"]["chains"] == {"D": BINDER_SEQUENCE}


# --- PostFilter fallback --------------------------------------------------------------


def objective_fallback(eligible: list[Candidate], rejected: list[Candidate]) -> dict[str, Any]:
    return DesignOrchestrator._objective_final_selection(
        eligible=eligible,
        rejected=rejected,
        terminal=eligible + rejected,
        config=make_config(post_filter_top_k=1),
        strategy_summary="fallback",
        post_filter_error="PostFilter Agent must rank every eligible candidate exactly once",
    )


def test_objective_fallback_selects_instead_of_returning_empty() -> None:
    eligible = [make_candidate("worse", 2.0), make_candidate("better", 1.0)]
    payload = objective_fallback(eligible, [])
    assert payload["selected_candidate_ids"] == ["better"]
    assert payload["mode"] == "deterministic"
    assert payload["post_filter_error"]


def test_objective_fallback_ranks_rejected_candidates_last() -> None:
    eligible = [make_candidate("ok", 1.0)]
    rejected = [make_candidate("bad", 9.0, gate=False), make_candidate("also_bad", 0.0, gate=False)]
    payload = objective_fallback(eligible, rejected)
    ranks = {item["candidate_id"]: item["rank"] for item in payload["decisions"]}
    assert ranks == {"ok": 1, "bad": 2, "also_bad": 3}
    assert payload["selected_candidate_ids"] == ["ok"]


def test_failed_post_filter_selection_still_yields_nothing() -> None:
    """The empty payload stays reachable only when no candidate is eligible at all."""

    terminal = [make_candidate("bad", 9.0, gate=False)]
    payload = DesignOrchestrator._failed_post_filter_selection(terminal, [], None, "skipped")
    assert payload["selected_candidate_ids"] == []
    assert payload["mode"] == "failed"


def test_run_falls_back_to_objective_ordering_when_ranking_is_invalid() -> None:
    phases = FakePhases()
    phases.post_filter_output = PostFilterAgentOutput(
        strategy_summary="broken",
        decisions=[{"candidate_id": "does-not-exist", "rank": 1, "rationale": "wrong"}],
    )
    orchestrator, _compute, _ = make_orchestrator(phases)
    snapshot = asyncio.run(
        orchestrator.run(
            "task-fallback",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert snapshot.final_selection is not None
    assert snapshot.final_selection["selected_candidate_ids"]


def run_terminal_case(monkeypatch, candidates, *, output=None, quality=None, **config_overrides):
    from opendde_harness.plugin.protein_design.core import orchestrator as module

    class TerminalCompute(FakeCompute):
        async def submit_fold(self, request):
            submission = await super().submit_fold(request)
            if request.options.get("post_refold"):
                self.jobs[submission.job_id] = {"candidates": [item.model_dump(mode="json") for item in candidates]}
            return submission

    class TerminalPhases(FakePhases):
        def __init__(self):
            super().__init__()
            self.quality_calls = []

        async def quality_cycle(self, config, cycle, candidates, analysis):
            self.quality_calls.append((cycle, list(candidates)))
            if cycle == config.cycles and quality is not None:
                if isinstance(quality, Exception):
                    raise quality
                return quality
            return QualityBatchOutput(results={})

    async def prepare(compute, parents, *args):
        result = [item.model_copy(deep=True) for item in candidates]
        for candidate in result:
            candidate.metadata["post_mpnn_selected"] = True
        return result

    monkeypatch.setattr(module, "prepare_post_mpnn", prepare)
    compute = TerminalCompute()
    phases = TerminalPhases()
    phases.post_filter_output = output
    orchestrator = DesignOrchestrator(compute, DesignMemory(None, agent_id="test"), phases)
    snapshot = asyncio.run(
        orchestrator.run("terminal-policy", make_config(**config_overrides), stop_event=asyncio.Event(), adjustments={})
    )
    return snapshot, compute, phases


@pytest.mark.parametrize("gate_reason", ["cdr_contact_fraction_below_threshold", "cdr3_hotspot_contact_missing"])
def test_terminal_hard_gate_cannot_be_overridden_by_agent_praise(monkeypatch, gate_reason):
    bad = make_candidate("bad", -10.0, gate=False)
    bad.metadata["gate_evidence"] = {"reason": gate_reason}
    good = make_candidate("good", 1.0)
    praise = PostFilterAgentOutput(
        strategy_summary="Prefer failed gate",
        decisions=[
            {"candidate_id": "bad", "rank": 1, "rationale": "Excellent despite gate failure"},
            {"candidate_id": "good", "rank": 2, "rationale": "Second choice"},
        ],
    )
    snapshot, compute, phases = run_terminal_case(monkeypatch, [bad, good], output=praise)
    assert snapshot.status.value == "completed"
    assert [item.candidate_id for item in phases.post_filter_calls[0]] == ["good"]
    selection = snapshot.final_selection
    assert selection["selected_candidate_ids"] == ["good"]
    assert selection["mode"] == "deterministic"
    assert selection["post_filter_error"]
    rejected = next(item for item in selection["decisions"] if item["candidate_id"] == "bad")
    assert rejected["hard_eligible"] is False
    assert rejected["pass_filter"] is False
    assert gate_reason in rejected["rationale"]
    assert compute.population_updates[-1]["final_selection"] == selection


@pytest.mark.parametrize("minimize", [True, False])
@pytest.mark.parametrize("top_k", [1, 2])
def test_terminal_advisory_order_never_changes_objective_top_k(monkeypatch, minimize, top_k):
    candidates = [make_candidate("middle", 0.0), make_candidate("low", -2.0), make_candidate("high", 3.0)]
    ordered = sorted(candidates, key=lambda item: item.objective, reverse=not minimize)
    advice = PostFilterAgentOutput(
        strategy_summary="Reverse objective for diversity",
        decisions=[
            {"candidate_id": item.candidate_id, "rank": rank, "rationale": "Advisory tradeoff"}
            for rank, item in enumerate(reversed(ordered), start=1)
        ],
    )
    snapshot, _, _ = run_terminal_case(
        monkeypatch, candidates, output=advice, minimize=minimize, post_filter_top_k=top_k
    )
    selection = snapshot.final_selection
    expected = [item.candidate_id for item in ordered]
    assert selection["mode"] == "deterministic"
    assert selection["selected_candidate_ids"] == expected[:top_k]
    assert [item["candidate_id"] for item in selection["decisions"]] == expected
    assert [item["rank"] for item in selection["decisions"]] == [1, 2, 3]
    assert [item["advisory_rank"] for item in selection["decisions"]] == [3, 2, 1]
    assert selection["advisory_strategy_summary"] == advice.strategy_summary
    fallback = DesignOrchestrator._objective_final_selection(
        eligible=candidates,
        rejected=[],
        terminal=candidates,
        config=make_config(minimize=minimize, post_filter_top_k=top_k),
        strategy_summary="fallback",
    )
    assert fallback["selected_candidate_ids"] == selection["selected_candidate_ids"]


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "name,direction", [("rosetta_interface_dg", "minimize"), ("min_ipae", "minimize"), ("ipsae", "maximize")]
)
def test_terminal_missing_required_metric_cannot_be_selected(monkeypatch, metric, name, direction):
    candidate = make_candidate("unscored", -20.0)
    if metric is not None:
        candidate.metrics[name] = metric
    snapshot, _, phases = run_terminal_case(
        monkeypatch,
        [candidate],
        fold_options={"metric_loss_terms": {name: {"weight": 0.05, "direction": direction}}},
    )
    assert snapshot.status.value == "failed"
    assert snapshot.final_selection["mode"] == "failed"
    assert snapshot.final_selection["selected_candidate_ids"] == []
    assert snapshot.final_selection["decisions"][0]["hard_eligible"] is False
    assert "required loss metrics" in snapshot.final_candidates[0].metadata["post_refold_error"]
    assert phases.post_filter_calls == []


def test_terminal_quality_is_fresh_and_missing_or_failed_verdict_rejects(monkeypatch):
    candidates = [make_candidate(name, float(index)) for index, name in enumerate(("failed", "missing", "passed"))]
    for candidate in candidates:
        candidate.metrics["iptm"] = 0.9
        candidate.metadata["post_refold_quality_passed"] = True  # A stale verdict must not carry over.
    quality = QualityBatchOutput(
        results={
            name: {"reasoning": "Fresh assessment", "overall_risk_level": "low", "pass_check": passed}
            for name, passed in (("failed", False), ("passed", True))
        }
    )
    snapshot, _, phases = run_terminal_case(monkeypatch, candidates, quality=quality, quality_check_enabled=True)
    assert snapshot.status.value == "completed"
    assert snapshot.final_selection["selected_candidate_ids"] == ["passed"]
    assert [(cycle, [item.candidate_id for item in items]) for cycle, items in phases.quality_calls] == [
        (2, ["failed", "missing", "passed"])
    ]
    assert [item.metadata["post_refold_quality_passed"] for item in snapshot.final_candidates] == [False, False, True]


def test_terminal_quality_transport_failure_is_not_success(monkeypatch):
    candidate = make_candidate("quality-unavailable", 1.0)
    candidate.metrics["iptm"] = 0.9
    snapshot, _, phases = run_terminal_case(
        monkeypatch, [candidate], quality=RuntimeError("quality unavailable"), quality_check_enabled=True
    )
    assert snapshot.status.value == "failed"
    assert snapshot.error == "quality unavailable"
    assert snapshot.final_selection["selected_candidate_ids"] == []
    assert phases.post_filter_calls == []


@pytest.mark.parametrize("failure", ["gate", "scoring", "structure", "masked"])
def test_all_terminal_candidates_ineligible_marks_run_failed(monkeypatch, failure):
    candidate = make_candidate("failed", 1.0)
    if failure == "gate":
        candidate.metadata["gate_passed"] = False
    elif failure == "scoring":
        candidate.metadata["success"] = False
        candidate.objective = None
    elif failure == "structure":
        candidate.structure_path = None
    else:
        candidate.sequence = "XXXXXXXXXX"
        candidate.metadata["chains"] = {"D": candidate.sequence}
    snapshot, _, phases = run_terminal_case(monkeypatch, [candidate])
    assert snapshot.status.value == "failed"
    assert snapshot.error
    assert snapshot.final_selection["mode"] == "failed"
    assert snapshot.final_selection["selected_candidate_ids"] == []
    assert snapshot.final_selection["decisions"][0]["hard_eligible"] is False
    assert phases.post_filter_calls == []


def test_refold_never_inherits_parent_gate_or_quality_verdict():
    previous = make_candidate("candidate", 1.0)
    fresh = make_candidate("candidate", 2.0)
    fresh.metadata.pop("gate_passed")
    fresh.metrics.pop("gate_passed")
    merged = DesignOrchestrator._merge_post_refold_candidates([fresh], [previous])[0]
    assert merged.metadata["gate_passed"] is None
    assert merged.metadata["post_refold_quality_passed"] is False
    assert DesignOrchestrator._post_filter_eligible(merged) is False


def test_quality_rejection_reason_does_not_report_a_successful_geometry_gate():
    candidate = make_candidate("quality-rejected", 0.3)
    candidate.metadata.update(
        gate_evidence={"reason": "cdr_contact_fraction_passed"},
        post_refold_quality_passed=False,
        post_refold_quality={"pass_check": False, "reasoning": "Current candidate liability is High Risk"},
    )
    decision = DesignOrchestrator._hard_rejection_decision(candidate, 1)
    assert "quality_check_failed" in decision["rationale"]
    assert "Current candidate liability" in decision["rationale"]
    assert "cdr_contact_fraction_passed" not in decision["rationale"]


def test_terminal_metric_guard_never_inherits_parent_metrics():
    previous = make_candidate("candidate", -1.0)
    previous.metrics["rosetta_interface_dg"] = -40.0
    fresh = make_candidate("candidate", -2.0)
    merged = DesignOrchestrator._merge_post_refold_candidates([fresh], [previous])[0]
    config = make_config(
        fold_options={"metric_loss_terms": {"rosetta_interface_dg": {"weight": 0.05, "direction": "minimize"}}}
    )
    DesignOrchestrator._validate_terminal_metrics([merged], config)
    assert "rosetta_interface_dg" not in merged.metrics
    assert merged.metadata["post_refold_success"] is False
    assert "required loss metrics" in merged.metadata["post_refold_error"]


@pytest.mark.parametrize("weight,metric", [(0.05, -40.0), (0.0, None)])
def test_terminal_metric_guard_accepts_finite_required_or_disabled_metric(monkeypatch, weight, metric):
    candidate = make_candidate("valid", -2.0)
    if metric is not None:
        candidate.metrics["rosetta_interface_dg"] = metric
    snapshot, _, _ = run_terminal_case(
        monkeypatch,
        [candidate],
        fold_options={"metric_loss_terms": {"rosetta_interface_dg": {"weight": weight, "direction": "minimize"}}},
    )
    assert snapshot.status.value == "completed"
    assert snapshot.final_selection["selected_candidate_ids"] == ["valid"]


def test_terminal_quality_uses_same_threshold_and_skill_exemption_as_search(monkeypatch):
    low_confidence = make_candidate("below_trigger", -2.0)
    low_confidence.metrics["iptm"] = 0.6
    redesign = make_candidate("full_redesign", -1.0)
    redesign.metrics["iptm"] = 0.9
    redesign.metadata["skill_id"] = "cdr-full-redesign"
    snapshot, _, phases = run_terminal_case(
        monkeypatch, [low_confidence, redesign], quality_check_enabled=True, post_filter_top_k=2
    )
    assert phases.quality_calls == []
    assert snapshot.final_selection["selected_candidate_ids"] == ["below_trigger", "full_redesign"]


def test_disabled_terminal_selection_uses_actual_last_cycle_quality_admission(monkeypatch):
    orchestrator, compute, phases = make_orchestrator()
    submit = compute.submit_fold

    async def high_confidence(request):
        result = await submit(request)
        for item in compute.jobs[result.job_id]["candidates"]:
            item["metrics"]["iptm"] = 0.9
            item["metadata"]["post_refold_quality_passed"] = True  # Stale unrelated evidence cannot admit it.
        return result

    async def quality(config, cycle, candidates, analysis):
        return QualityBatchOutput(
            results={
                item.candidate_id: {
                    "reasoning": "Reject the lower objective for quality",
                    "overall_risk_level": "low",
                    "pass_check": item.candidate_id.endswith("_1"),
                }
                for item in candidates
            }
        )

    monkeypatch.setattr(compute, "submit_fold", high_confidence)
    monkeypatch.setattr(phases, "quality_cycle", quality)
    snapshot = asyncio.run(
        orchestrator.run(
            "quality-gated-without-refold",
            make_config(post_filter_enabled=False, quality_check_enabled=True),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert snapshot.final_selection["selected_candidate_ids"] == ["c1_1"]
    rejected = next(item for item in snapshot.final_selection["decisions"] if item["candidate_id"] == "c1_0")
    assert rejected["hard_eligible"] is False
    assert rejected["objective"] < snapshot.final_candidates[1].objective


# --- end-to-end run -------------------------------------------------------------------


def test_run_completes_and_reports_a_selection() -> None:
    orchestrator, compute, _ = make_orchestrator()
    snapshot = asyncio.run(
        orchestrator.run(
            "task-1",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert snapshot.status.value == "completed"
    assert snapshot.best_candidate is not None
    assert snapshot.final_selection is not None
    assert snapshot.final_selection["selected_candidate_ids"]
    assert compute.population_updates


def test_cycle_snapshots_carry_the_selected_skill() -> None:
    """The TUI renders TaskSnapshot.selected_skill, so the orchestrator must write it."""

    emitted: list[Any] = []
    orchestrator, _compute, _ = make_orchestrator()
    snapshot = asyncio.run(
        orchestrator.run(
            "task-skill",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
            on_progress=lambda item: emitted.append(item),
        )
    )
    running = [item for item in emitted if item.status.value == "running" and item.cycle > 0]
    assert running, "expected at least one in-cycle snapshot"
    assert all(item.selected_skill == "cdr-point-mutation" for item in running)
    assert snapshot.selected_skill is None


def test_run_stops_at_the_next_cycle_boundary() -> None:
    orchestrator, _compute, _ = make_orchestrator()
    stop_event = asyncio.Event()
    stop_event.set()
    snapshot = asyncio.run(
        orchestrator.run(
            "task-stop",
            make_config(post_filter_enabled=False),
            stop_event=stop_event,
            adjustments={},
        )
    )
    assert snapshot.status.value == "stopped"


def test_run_emits_progress_events_for_each_phase() -> None:
    events: list[str] = []
    compute = FakeCompute()
    orchestrator = DesignOrchestrator(
        compute,
        DesignMemory(None, agent_id="test-agent"),
        FakePhases(),
        fold_poll_interval=0.0,
        event_sink=lambda event: events.append(event.phase),
    )
    asyncio.run(
        orchestrator.run(
            "task-events",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert "analyze" in events
    assert "initial_fold" in events
    assert "final_visualization" in events


# --- session tool-turn budget ---------------------------------------------------------


class ToolCall:
    def __init__(self, name: str, arguments: dict[str, Any], call_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.id = call_id

    def to_pi_tool_call(self) -> dict[str, Any]:
        from opendde_harness.providers.messages import tool_call_block

        return tool_call_block(self.id, self.name, self.arguments)


class Response:
    def __init__(self, content: str = "", tool_calls: list[ToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = "stop"
        # A double, not the model service: no pi message to replay, so the
        # session builds the assistant turn from the fields beside it.
        self.pi_message = None


class SkillLoopProvider:
    """A provider that answers every turn with another use_skill call."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat_with_retry(self, **_: Any) -> Response:
        self.calls += 1
        return Response(tool_calls=[ToolCall("use_skill", {"skill_id": f"ghost-{self.calls}"}, f"t{self.calls}")])


def test_structured_session_terminates_on_a_persistent_use_skill_caller() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    provider = SkillLoopProvider()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=3)
    skills = (SkillDocument("only-skill", Path("/tmp/only-skill"), "content", ()),)

    with pytest.raises(RuntimeError, match="use_skill budget"):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert provider.calls == len(skills) + session._max_attempts + 1


def test_structured_session_budget_scales_with_available_skills() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    provider = SkillLoopProvider()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=2)
    skills = tuple(SkillDocument(f"skill-{index}", Path(f"/tmp/skill-{index}"), "content", ()) for index in range(3))

    with pytest.raises(RuntimeError, match="use_skill budget"):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert provider.calls == len(skills) + 2 + 1


def test_structured_session_emits_one_input_payload_per_run() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    payloads: list[Any] = []

    class Sink:
        def __call__(self, event: Any) -> None:
            if event.event_type.value == "agent":
                payloads.append(event.input_payload)

    provider = SkillLoopProvider()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=1, progress_sink=Sink())
    renders = 0
    original = session._system_message

    def counting_system_message(*args: Any, **kwargs: Any) -> str:
        nonlocal renders
        renders += 1
        return original(*args, **kwargs)

    session._system_message = counting_system_message
    skills = (SkillDocument("only-skill", Path("/tmp/only-skill"), "content", ()),)
    with pytest.raises(RuntimeError):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert renders == 2


def test_structured_session_widens_the_budget_when_the_model_runs_out_of_tokens() -> None:
    import json as json_module

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession

    class TruncatedThenAnswer:
        def __init__(self) -> None:
            self.budgets: list[int] = []
            self.requests: list[dict[str, Any]] = []

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.requests.append(kwargs)
            self.budgets.append(int(kwargs["max_tokens"]))
            if len(self.budgets) == 1:
                response = Response(content="")
                response.finish_reason = "length"
                return response
            return Response(content=json_module.dumps({"downstream_header": "epitope", "report": "ok"}))

    provider = TruncatedThenAnswer()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=2)
    result = asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt"))

    assert result.report == "ok"
    assert provider.budgets == [16384, 32768]
    # And no sampling temperature on either request. The agent profile carried
    # one and sent it on every call, so a design task launched from a Codex
    # conversation died on "Unsupported parameter: temperature". A temperature
    # is declared on the model's own row and read by the provider; no caller
    # names one.
    assert provider.requests, "the session called the provider"
    assert all("temperature" not in request for request in provider.requests), provider.requests


# --- config loading -------------------------------------------------------------------


def test_bundled_example_configs_still_normalize() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader

    for name in ("crlf2_quickstart.yaml", "cacng1_quickstart.yaml"):
        path = Path("docs/examples") / name
        if not path.is_file():
            pytest.skip(f"{path} is not present")
        config = WorkflowConfigLoader.config_from_path(str(path))
        assert config.target
        assert config.binder_chains
        assert math.isfinite(float(config.seed))


def test_a_yaml_llm_temperature_reaches_no_request(tmp_path) -> None:
    """`llm.temperature` in a task file is not a setting any more.

    It was read into ``WorkflowConfig.llm_temperature``, travelled as tool
    metadata and became a ``temperature=`` on every design call -- including
    calls to a Codex model, which answers "Unsupported parameter: temperature"
    and fails the turn. The key still loads so an existing file is not rejected,
    and nothing reads it: a sampling temperature belongs to the model's row in
    the harness config.
    """
    import json as json_module
    from pathlib import Path

    import yaml

    from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
    from opendde_harness.plugin.protein_design.tools.agent import ToolContext

    source = Path("docs/examples") / "crlf2_quickstart.yaml"
    if not source.is_file():
        pytest.skip(f"{source} is not present")
    data = yaml.safe_load(source.read_text())
    data["llm"] = {"model_name": "openai-codex/gpt-5.6-luna", "temperature": 0.7}
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(data), encoding="utf-8")

    config = WorkflowConfigLoader.config_from_path(str(task_file))

    assert config.llm_model == "openai-codex/gpt-5.6-luna", "the model is still a task setting"
    assert not hasattr(config, "llm_temperature"), "the field is gone, not merely unused"
    metadata = ProteinDesignPhases._tool_metadata(config)
    assert "llm_temperature" not in metadata

    class Recorder:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.requests.append(kwargs)
            return Response(content=json_module.dumps({"downstream_header": "epitope", "report": "ok"}))

    provider = Recorder()
    session = OpenDDEHarnessStructuredSession(provider, "fallback-model", max_attempts=1)
    result = asyncio.run(
        session.run(
            AGENT_PROFILES[AgentRole.ANALYZE],
            "prompt",
            tool_context=ToolContext(metadata=metadata),
        )
    )

    assert result.report == "ok"
    assert provider.requests[0]["model"] == "openai-codex/gpt-5.6-luna"
    assert "temperature" not in provider.requests[0], provider.requests[0]


# --- optional external services ---------------------------------------------------------


class ProtrekOutageCompute(FakeCompute):
    """A worker whose ProTrek call fails the way an egress-less container fails."""

    def __init__(self) -> None:
        super().__init__()
        self.protrek_calls = 0

    async def search_protrek_sequence(self, request: Any) -> Any:
        import httpx

        self.protrek_calls += 1
        raise httpx.ConnectTimeout("timed out")


def _analyze_session(compute: Any) -> tuple[Any, Any]:
    import json as json_module

    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.tools.agent import ProteinDesignToolRegistry

    class ProtrekThenAnswer:
        def __init__(self) -> None:
            self.calls = 0
            self.tool_messages: list[str] = []

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.calls += 1
            self.tool_messages.extend(
                str(message.get("content")) for message in kwargs.get("messages", []) if message.get("role") == "tool"
            )
            if self.calls == 1:
                return Response(tool_calls=[ToolCall("protrek_sequence_search", {"sequence": BINDER_SEQUENCE}, "t1")])
            return Response(content=json_module.dumps({"downstream_header": "epitope", "report": "no homologs"}))

    provider = ProtrekThenAnswer()
    session = OpenDDEHarnessStructuredSession(
        provider,
        "fake-model",
        tool_registry=ProteinDesignToolRegistry.for_compute(compute),
        max_attempts=3,
    )
    return provider, session


def test_run_completes_when_the_analysis_agent_hits_a_protrek_outage() -> None:
    from opendde_harness.plugin.protein_design.core.memory import DesignMemory

    compute = ProtrekOutageCompute()

    class ProtrekAnalysisPhases(FakePhases):
        async def analyze_once(self, config: WorkflowConfig) -> dict[str, Any]:
            _provider, session = _analyze_session(compute)
            from opendde_harness.plugin.protein_design.agents.profiles import (
                AGENT_PROFILES,
                AgentRole,
            )

            output = await session.run(AGENT_PROFILES[AgentRole.ANALYZE], "analyze the target")
            return {"summary": output.downstream_header}

    config = make_config(post_filter_enabled=False)
    orchestrator = DesignOrchestrator(
        compute,
        DesignMemory(None, agent_id="test-agent"),
        ProtrekAnalysisPhases(),
        fold_poll_interval=0.0,
    )
    snapshot = asyncio.run(orchestrator.run("task-protrek", config, stop_event=asyncio.Event(), adjustments={}))

    assert snapshot.status.value == "completed"
    assert not snapshot.failed_cycles
    assert not config.metadata.get("failed_cycles")
    assert compute.protrek_calls >= 1


def test_design_agent_returns_its_reasoning_with_the_tool_turn() -> None:
    """DeepSeek in thinking mode rejects a tool turn that comes back without its reasoning."""
    import json as json_module
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    sent: list[list[dict[str, Any]]] = []

    class ThinkingProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.calls += 1
            sent.append([dict(message) for message in kwargs["messages"]])
            if self.calls == 1:
                first = Response(tool_calls=[ToolCall("use_skill", {"skill_id": "only-skill"}, "t1")])
                first.reasoning_content = "deciding which skill to load"
                return first
            answer = Response(content=json_module.dumps({"downstream_header": "epitope", "report": "ok"}))
            answer.reasoning_content = "writing the report"
            return answer

    session = OpenDDEHarnessStructuredSession(ThinkingProvider(), "fake-model", max_attempts=2)
    skills = (SkillDocument("only-skill", Path("/tmp/only-skill"), "content", ()),)

    result = asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert result.report == "ok"
    from opendde_harness.providers.messages import thinking_of

    assistant = [message for message in sent[-1] if message["role"] == "assistant"]
    assert assistant and thinking_of(assistant[0]) == "deciding which skill to load"


@pytest.mark.parametrize("gate_passed", [True, False])
def test_initial_structure_is_published_before_first_design_failure(tmp_path, monkeypatch, gate_passed):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from opendde_harness.tracing import spans
    from opendde_harness.tracing.store import TraceStore

    monkeypatch.setenv("OPENDDE_HARNESS_TRACING", "1")
    monkeypatch.setattr(spans, "_store", TraceStore(tmp_path))
    compute = FakeCompute()
    original_wait = compute.wait_fold

    async def wait_fold(job_id, **kwargs):
        result = await original_wait(job_id, **kwargs)
        for candidate in result.result["candidates"]:
            candidate["metadata"]["gate_passed"] = gate_passed
            candidate["metrics"]["gate_passed"] = float(gate_passed)
        return result

    async def read_structure(*args, **kwargs):
        return SimpleNamespace(filename="seed.cif", format="cif", byte_count=10, text="data_seed\n")

    async def fail_design(*args, **kwargs):
        raise RuntimeError("first design failed")

    monkeypatch.setattr(compute, "wait_fold", wait_fold)
    monkeypatch.setattr(compute, "read_structure", read_structure)
    phases = FakePhases()
    monkeypatch.setattr(phases, "design_cycle", fail_design)
    orchestrator = DesignOrchestrator(compute, DesignMemory(None, agent_id="test"), phases)
    snapshots = []
    with pytest.raises(RuntimeError, match="first design failed"):
        asyncio.run(
            orchestrator.run(
                "task-initial",
                make_config(cycle_retry_limit=0, skip_failed_cycles=False),
                stop_event=asyncio.Event(),
                adjustments={},
                on_progress=snapshots.append,
            )
        )

    records = [json.loads(line) for line in (tmp_path / "logs/audit-spans.log").read_text().splitlines()]
    cycles = [record for record in records if record["name"] == "protein_design.cycle"]
    assert len(cycles) == 1
    attrs = cycles[0]["attributes"]
    assert attrs["protein_design.cycle_index"] == -1
    artifact = json.loads(Path(attrs["protein_design.cycle.artifact_path"]).read_text())
    candidate = artifact["candidates"][0]
    assert candidate["candidate_id"] == "seed"
    assert candidate["metrics"]["loss"] == 1.0
    structure = json.loads(Path(candidate["structure_artifact_path"]).read_text())
    assert structure["text"] == "data_seed\n"
    assert structure["target_chain_ids"] == ["A"]
    assert structure["binder_chain_ids"] == ["D"]
    assert artifact["population_candidate_ids"] == (["seed"] if gate_passed else [])
    assert artifact["admitted_candidate_ids"] == (["seed"] if gate_passed else [])
    assert snapshots[-1].cycle == 0
    assert (snapshots[-1].best_candidate is not None) == gate_passed
    if gate_passed:
        assert any(s.status.value == "running" and s.best_candidate is not None for s in snapshots)


@pytest.mark.parametrize("failure_phase", ["quality", "population", "memory", "initial_population"])
def test_fold_results_visible_before_downstream_failure(tmp_path, monkeypatch, failure_phase):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from opendde_harness.tracing import spans
    from opendde_harness.tracing.store import TraceStore

    monkeypatch.setenv("OPENDDE_HARNESS_TRACING", "1")
    monkeypatch.setattr(spans, "_store", TraceStore(tmp_path))
    compute, phases = FakeCompute(), FakePhases()
    memory = DesignMemory(None, agent_id="test")
    expected_cycle = -1 if failure_phase == "initial_population" else 0

    def visible():
        records = [json.loads(line) for line in (tmp_path / "logs/audit-spans.log").read_text().splitlines()]
        latest = {r["spanId"]: r for r in records if r["name"] == "protein_design.cycle"}
        matches = [r for r in latest.values() if r["attributes"]["protein_design.cycle_index"] == expected_cycle]
        assert len(matches) == 1
        artifact = json.loads(Path(matches[0]["attributes"]["protein_design.cycle.artifact_path"]).read_text())
        assert artifact["candidates"]
        for item in artifact["candidates"]:
            assert json.loads(Path(item["structure_artifact_path"]).read_text())["text"] == "data_fold\n"
        if failure_phase == "quality":
            assert "admitted_candidate_ids" not in artifact

    async def read_structure(*args, **kwargs):
        return SimpleNamespace(filename="fold.cif", format="cif", byte_count=10, text="data_fold\n")

    async def fail(*args, **kwargs):
        visible()  # Verify the live checkpoint, before the span exits on error.
        raise RuntimeError("downstream unavailable")

    original_update = compute.update_population

    async def update(payload):
        if payload["cycle"] == expected_cycle:
            await fail()
        return await original_update(payload)

    monkeypatch.setattr(compute, "read_structure", read_structure)
    if failure_phase == "quality":
        monkeypatch.setattr(phases, "quality_cycle", fail)
    elif failure_phase == "memory":
        monkeypatch.setattr(memory, "record", fail)
    else:
        monkeypatch.setattr(compute, "update_population", update)
    orchestrator = DesignOrchestrator(compute, memory, phases)
    if failure_phase == "quality":
        monkeypatch.setattr(orchestrator, "_needs_quality_check", lambda *args: True)
    if failure_phase == "memory":
        monkeypatch.setattr(orchestrator, "_case_triggers", lambda **kwargs: ["test"])
    with pytest.raises(RuntimeError, match="downstream unavailable"):
        asyncio.run(
            orchestrator.run(
                "early-fold",
                make_config(cycle_retry_limit=0, skip_failed_cycles=False),
                stop_event=asyncio.Event(),
                adjustments={},
            )
        )
    visible()


def test_each_completed_cycle_updates_same_trace_without_recapturing_structures(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from opendde_harness.tracing import spans
    from opendde_harness.tracing.store import TraceStore

    monkeypatch.setenv("OPENDDE_HARNESS_TRACING", "1")
    monkeypatch.setattr(spans, "_store", TraceStore(tmp_path))
    compute = FakeCompute()
    reads = []

    async def read_structure(*args, **kwargs):
        reads.append((args, kwargs))
        return SimpleNamespace(filename="fold.cif", format="cif", byte_count=10, text="data_fold\n")

    monkeypatch.setattr(compute, "read_structure", read_structure)
    orchestrator = DesignOrchestrator(compute, DesignMemory(None, agent_id="test"), FakePhases())
    asyncio.run(
        orchestrator.run(
            "all-cycles", make_config(post_filter_enabled=False), stop_event=asyncio.Event(), adjustments={}
        )
    )
    records = [json.loads(line) for line in (tmp_path / "logs/audit-spans.log").read_text().splitlines()]
    latest = {r["spanId"]: r for r in records if r["name"] == "protein_design.cycle"}
    assert sorted(r["attributes"]["protein_design.cycle_index"] for r in latest.values()) == [-1, 0, 1]
    assert len(reads) == 5  # one seed and two candidates in each of two rounds
    for r in latest.values():
        a = json.loads(Path(r["attributes"]["protein_design.cycle.artifact_path"]).read_text())
        assert "admitted_candidate_ids" in a


def test_post_refold_is_visible_before_pose_and_filter(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from opendde_harness.plugin.protein_design.core import orchestrator as module
    from opendde_harness.tracing import spans
    from opendde_harness.tracing.store import TraceStore

    monkeypatch.setenv("OPENDDE_HARNESS_TRACING", "1")
    monkeypatch.setattr(spans, "_store", TraceStore(tmp_path))
    compute = FakeCompute()

    async def prepare(compute, candidates, *args):
        for item in candidates:
            item.metadata["post_mpnn_selected"] = True
        return candidates

    async def read_structure(*args, **kwargs):
        return SimpleNamespace(filename="fold.cif", format="cif", byte_count=10, text="data_refold\n")

    checked = []

    async def pose(*args):
        records = [json.loads(line) for line in (tmp_path / "logs/audit-spans.log").read_text().splitlines()]
        r = next(r for r in records if r["attributes"].get("protein_design.phase") == "post_refold")
        a = json.loads(Path(r["attributes"]["protein_design.cycle.artifact_path"]).read_text())
        assert a["cycle"] == 2
        assert a["candidates"][0]["structure_artifact_path"]
        assert "admitted_candidate_ids" not in a
        checked.append(True)
        raise RuntimeError("pose unavailable")

    monkeypatch.setattr(module, "prepare_post_mpnn", prepare)
    monkeypatch.setattr(compute, "read_structure", read_structure)
    orchestrator = DesignOrchestrator(compute, DesignMemory(None, agent_id="test"), FakePhases())
    monkeypatch.setattr(orchestrator, "_post_refold_pose_evidence", pose)
    asyncio.run(orchestrator.run("refold", make_config(), stop_event=asyncio.Event(), adjustments={}))
    assert checked == [True]
