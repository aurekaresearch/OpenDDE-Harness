import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opendde_harness.plugin.protein_design.core.contracts import Candidate
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.plugin.protein_design.core.tracing import build_cycle_artifact
from opendde_harness.plugin.protein_design.servers.backends import fold, pyrosetta_analysis
from opendde_harness.plugin.protein_design.servers.harness import PythonProteinDesignHarness


@pytest.fixture
def harness(tmp_path, monkeypatch):
    class Predictor:
        def __init__(self, config):
            pass

        def predict_batch(self, sequences, output_dir):
            return [
                fold.FoldResult(
                    sequences={"A": "AAA", "B": "AAA"},
                    structure_path=[tmp_path / f"fold_{i}.pdb"],
                    confidence_path=[],
                    all_atom_confidence_path=tmp_path / "confidence.json",
                    iptm=0.8,
                    plddt=0.9,
                    backend="opendde",
                )
                for i in range(len(sequences))
            ]

    monkeypatch.setattr(fold, "StructurePredictor", Predictor)
    instance = PythonProteinDesignHarness(output_path=str(tmp_path))
    monkeypatch.setattr(instance, "_evaluate_gate", lambda *args: (True, {"cdr3_gate_passed": True}))
    monkeypatch.setattr(instance, "_score_esm", lambda payload: {"scores": [-2.0] * len(payload["sequences"])})
    return instance


def payload(**options):
    return {
        "task_id": "test",
        "candidates": [
            {"candidate_id": "first", "sequence": "AAA"},
            {"candidate_id": "second", "sequence": "AAA"},
        ],
        "options": {
            "execution_mode": "api",
            "objective_key": "iptm",
            "binder_chain_ids": ["B"],
            "target_chain_ids": ["A"],
            "target_chains": {"A": {"sequence": "AAA"}},
            "pyrosetta": {"enabled": True},
            **options,
        },
    }


def success(path="relaxed.pdb"):
    return pyrosetta_analysis.PyRosettaResult(
        status="success",
        relaxed_structure_path=path,
        metrics={
            key: (-20.0 if key == "rosetta_interface_dg" else 1.0) for key in pyrosetta_analysis.PYROSETTA_METRICS
        },
        provenance={"steps": ["FastRelax", "InterfaceAnalyzer"]},
        contact_residues=[
            {
                "chain_id": "B",
                "residue_index": 1,
                "pdb_residue_number": 102,
                "amino_acid": "A",
                "partner": "binder",
                "min_partner_distance": 3.0,
                "bound_score_reu": -3.0,
                "separated_score_reu": -2.0,
                "interface_dg_reu": -1.0,
            }
        ],
    )


def test_metrics_reach_candidates_and_cycle_trace(harness, monkeypatch):
    def analyze(structures, sequences, **kwargs):
        assert sequences == [{"A": "AAA", "B": "AAA"}] * 2
        assert kwargs["binder_chains"] == ["B"]
        assert len(structures) == 2
        return [success(), success()]

    monkeypatch.setattr(pyrosetta_analysis, "analyze_batch", analyze)
    rows = harness._fold(payload())["candidates"]
    candidate = Candidate.model_validate(rows[0])
    assert candidate.metrics["rosetta_interface_dg"] == -20
    assert candidate.metrics["iptm"] == 0.8
    assert candidate.structure_path.endswith("fold_0.pdb")
    assert candidate.metadata["pyrosetta"]["relaxed_structure_path"] == "relaxed.pdb"
    trace = build_cycle_artifact(
        task_id="test",
        target="target",
        cycle=1,
        objective_key="iptm",
        minimize=False,
        candidates=[candidate],
        cycle_best_id="first",
        global_best_id="first",
        known_candidate_ids={"first"},
        structure_artifacts={},
    )
    assert trace["candidates"][0]["metrics"]["rosetta_interface_dg"] == -20
    assert trace["candidates"][0]["metadata"]["pyrosetta"]["status"] == "success"
    assert trace["candidates"][0]["metadata"]["pyrosetta"]["contact_residues"][0]["bound_score_reu"] == -3.0


@pytest.mark.parametrize("policy,valid", [("fail", False), ("continue", True)])
def test_partial_analysis_failure_respects_policy(harness, monkeypatch, policy, valid):
    monkeypatch.setattr(
        pyrosetta_analysis,
        "analyze_batch",
        lambda *a, **kw: [pyrosetta_analysis.PyRosettaResult(status="failed", error="timed out"), success()],
    )
    first, second = harness._fold(payload(pyrosetta={"enabled": True, "on_failure": policy}))["candidates"]
    assert first["metadata"]["success"] is valid
    assert first["objective"] == (0.8 if valid else None)
    assert first["metadata"]["pyrosetta"]["error"] == "timed out"
    assert "rosetta_interface_dg" not in first["metrics"]
    assert second["metadata"]["success"]
    assert DesignOrchestrator._is_scored_candidate(Candidate.model_validate(first)) is valid


def test_all_analysis_failures_raise_actionable_batch_error(harness, monkeypatch):
    monkeypatch.setattr(
        pyrosetta_analysis,
        "analyze_batch",
        lambda *a, **kw: [
            pyrosetta_analysis.PyRosettaResult(status="failed", error="No module named pyrosetta") for _ in range(2)
        ],
    )
    with pytest.raises(RuntimeError, match="No module named pyrosetta"):
        harness._fold(payload())


def test_disabled_analysis_removes_stale_parent_provenance(harness):
    request = payload(pyrosetta={"enabled": False})
    request["candidates"][0]["metadata"] = {"pyrosetta": {"status": "success", "metrics": {"old": 1}}}
    first = harness._fold(request)["candidates"][0]
    assert "pyrosetta" not in first["metadata"]
    assert not any(key.startswith("rosetta_") for key in first["metrics"])


@pytest.mark.parametrize("post_refold", [False, True])
def test_raw_min_ipae_is_fresh_and_reaches_candidate_and_cycle(harness, monkeypatch, post_refold):
    from opendde_harness.plugin.protein_design.servers.backends import interchain_pae

    def measure(**kwargs):
        assert kwargs["binder_chains"] == ["B"]
        assert kwargs["target_chains"] == ["A"]
        assert kwargs["sequences"] == {"A": "AAA", "B": "AAA"}
        assert str(kwargs["confidence_path"]).endswith("confidence.json")
        return {"status": "success", "value": 2.75, "units": "angstrom"}

    monkeypatch.setattr(interchain_pae, "measure_min_interchain_pae", measure)
    request = payload(pyrosetta={"enabled": False}, post_refold=post_refold)
    request["candidates"][0].update(metrics={"min_ipae": -999}, metadata={"min_ipae": {"value": -999}})
    candidate = Candidate.model_validate(harness._fold(request)["candidates"][0])
    assert candidate.metrics["min_ipae"] == 2.75
    assert candidate.metadata["min_ipae"] == {"status": "success", "value": 2.75, "units": "angstrom"}
    artifact = build_cycle_artifact(
        task_id="test",
        target="target",
        cycle=0,
        objective_key="iptm",
        minimize=False,
        candidates=[candidate],
        cycle_best_id=candidate.candidate_id,
        global_best_id=candidate.candidate_id,
        known_candidate_ids={candidate.candidate_id},
        structure_artifacts={},
    )
    assert artifact["candidates"][0]["metrics"]["min_ipae"] == 2.75


def test_missing_raw_pae_omits_metric_and_clears_stale_parent_value(harness):
    request = payload(pyrosetta={"enabled": False})
    request["candidates"][0].update(
        metrics={"min_ipae": 0.1}, metadata={"min_ipae": {"status": "success", "value": 0.1}}
    )
    candidate = harness._fold(request)["candidates"][0]
    assert "min_ipae" not in candidate["metrics"]
    assert candidate["metadata"]["min_ipae"]["status"] == "unavailable"
    assert "FileNotFoundError" in candidate["metadata"]["min_ipae"]["reason"]
    assert candidate["metadata"]["success"] is True


@pytest.mark.parametrize("post_refold", [False, True])
@pytest.mark.parametrize("ipsae_value", [None, 0.0, 0.75])
def test_confidence_loss_reaches_trace_and_selection_with_explicit_ipsae_fallback(
    harness, monkeypatch, post_refold, ipsae_value
):
    from opendde_harness.plugin.protein_design.core.contracts import WorkflowConfig
    from opendde_harness.plugin.protein_design.servers.backends import interchain_pae, loss_confidence_scorer
    from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
        DEFAULT_LOSS_WEIGHTS,
        calculate_loss_objective,
    )

    result_type = fold.FoldResult
    monkeypatch.setattr(fold, "FoldResult", lambda **kwargs: result_type(ipsae=ipsae_value, **kwargs))
    values = iter([6.0, 12.0])
    monkeypatch.setattr(
        interchain_pae, "measure_min_interchain_pae", lambda **kwargs: {"status": "success", "value": next(values)}
    )
    weights = {name: 0.0 for name in DEFAULT_LOSS_WEIGHTS}
    weights["plddt"] = 1.0
    terms = {"min_ipae": {"direction": "minimize"}, "ipsae": {"direction": "maximize"}}
    combination = {
        "mode": "bounded_grouped",
        "calibration_id": "pipeline-test-fixture-only",
        "groups": {"confidence": {"budget": 1.0, "terms": ["plddt", "min_ipae", "ipsae"]}},
        "anchors": {
            "plddt": {"good": 0.0, "bad": 1.0},
            "min_ipae": {"good": 0.0, "bad": 20.0},
            "ipsae": {"good": 1.0, "bad": 0.0},
        },
    }

    def score(**kwargs):
        return calculate_loss_objective(
            {"plddt": 0.25},
            -2.0,
            weights=kwargs["loss_weights"],
            metric_values=kwargs["metric_values"],
            metric_terms=kwargs["metric_terms"],
            loss_combination=kwargs["loss_combination"],
        )

    monkeypatch.setattr(loss_confidence_scorer, "score_confidence_loss", score)
    request = payload(
        objective_key="loss",
        pyrosetta={"enabled": False},
        loss_weights=weights,
        metric_loss_terms=terms,
        loss_combination=combination,
        post_refold=post_refold,
    )
    for item in request["candidates"]:
        item.update(objective=-999, metrics={"loss": -999, "ipsae": 1.0, "min_ipae": 0.0})
        item["metadata"] = {"ipsae": {"status": "success", "value": 1.0}}
    rows = harness._fold(request)["candidates"]
    for row, min_ipae in zip(rows, [6.0, 12.0], strict=True):
        assert row["metadata"]["success"] is True
        raw_ipsae = ipsae_value if ipsae_value is not None else 0.0
        assert row["metrics"]["ipsae"] == raw_ipsae
        assert row["metrics"]["min_ipae"] == min_ipae
        assert row["objective"] == pytest.approx((0.25 + min_ipae / 20 + (1 - raw_ipsae)) / 3)
        assert row["metadata"]["loss"]["components"]["ipsae"]["raw"] == raw_ipsae
        provenance = row["metadata"]["ipsae"]
        assert provenance["status"] == ("unavailable" if ipsae_value is None else "success")
        assert provenance["value"] == ipsae_value
        assert ("fallback_value" in provenance) is (ipsae_value is None)
        if ipsae_value is None:
            assert provenance["fallback_value"] == 0.0
    assert rows[0]["metadata"]["loss"]["base_loss"] == rows[1]["metadata"]["loss"]["base_loss"]
    candidates = [Candidate.model_validate(row) for row in rows]
    artifact = build_cycle_artifact(
        task_id="test",
        target="target",
        cycle=1,
        objective_key="loss",
        minimize=True,
        candidates=candidates,
        cycle_best_id="first",
        global_best_id="first",
        known_candidate_ids={"first", "second"},
        structure_artifacts={},
    )
    assert artifact["candidates"][0]["metadata"]["ipsae"] == {
        key: value for key, value in rows[0]["metadata"]["ipsae"].items() if value is not None
    }
    assert artifact["candidates"][0]["metrics"]["loss"] == rows[0]["objective"]
    config = WorkflowConfig(target="target", objective_key="loss", minimize=True, post_filter_top_k=1)
    selected = DesignOrchestrator._objective_final_selection(
        eligible=list(reversed(candidates)),
        rejected=[],
        terminal=candidates,
        config=config,
        strategy_summary="Composite loss test",
    )
    assert selected["selected_candidate_ids"] == ["first"]


@pytest.mark.parametrize("post_refold", [False, True])
def test_required_min_ipae_remains_unavailable_not_zero(harness, monkeypatch, post_refold):
    from opendde_harness.plugin.protein_design.servers.backends import loss_confidence_scorer
    from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
        DEFAULT_LOSS_WEIGHTS,
        calculate_loss_objective,
    )

    def score(**kwargs):
        return calculate_loss_objective(
            {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS},
            -2.0,
            metric_values=kwargs["metric_values"],
            metric_terms=kwargs["metric_terms"],
        )

    monkeypatch.setattr(loss_confidence_scorer, "score_confidence_loss", score)
    with pytest.raises(RuntimeError, match="Missing required loss metric: min_ipae"):
        harness._fold(
            payload(
                objective_key="loss",
                pyrosetta={"enabled": False},
                post_refold=post_refold,
                metric_loss_terms={"min_ipae": {"direction": "minimize"}, "ipsae": {"direction": "maximize"}},
            )
        )


@pytest.mark.parametrize("bounded,post_refold", [(False, False), (True, False), (True, True)])
def test_required_metric_failure_cannot_continue_into_loss_ranking(harness, monkeypatch, bounded, post_refold):
    from opendde_harness.plugin.protein_design.servers.backends import loss_confidence_scorer
    from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
        DEFAULT_LOSS_WEIGHTS,
        calculate_loss_objective,
    )

    monkeypatch.setattr(
        pyrosetta_analysis,
        "analyze_batch",
        lambda *a, **kw: [pyrosetta_analysis.PyRosettaResult(status="failed", error="worker crashed"), success()],
    )

    def score(**kwargs):
        assert kwargs["structure_path"].endswith(".pdb")
        return calculate_loss_objective(
            {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS if key != "esm2"},
            -2,
            metric_values=kwargs["metric_values"],
            metric_terms=kwargs["metric_terms"],
            loss_combination=kwargs["loss_combination"],
        )

    monkeypatch.setattr(loss_confidence_scorer, "score_confidence_loss", score)
    combination = None
    if bounded:
        names = list(DEFAULT_LOSS_WEIGHTS) + ["rosetta_interface_dg"]
        combination = {
            "mode": "bounded_grouped",
            "calibration_id": "pipeline-test-fixture-only",
            "groups": {"all": {"budget": 1.0, "terms": names}},
            "anchors": {name: {"good": 0.0, "bad": 1.0} for name in names},
        }
        combination["anchors"].update(esm2={"good": 0.0, "bad": -5.0}, rosetta_interface_dg={"good": -50.0, "bad": 0.0})
    request = payload(
        objective_key="loss",
        pyrosetta={"enabled": True, "on_failure": "continue"},
        metric_loss_terms={"rosetta_interface_dg": {"direction": "minimize", "scale": 10.0}},
        loss_combination=combination,
        post_refold=post_refold,
    )
    for item in request["candidates"]:
        item.update(objective=-999, metrics={"loss": -999, "rosetta_interface_dg": -999})
        item["metadata"] = {"loss": {"formula_version": "stale", "loss": -999}}
    first, second = harness._fold(request)["candidates"]
    assert first["objective"] is None
    assert first["metadata"]["success"] is False
    assert "Missing required loss metric" in first["metadata"]["error"]
    assert "loss" not in first["metrics"]
    assert second["metrics"]["loss"] == second["objective"]
    assert second["metrics"]["rosetta_interface_dg"] == -20
    assert second["objective"] != -999
    assert second["metadata"]["loss"]["formula_version"] != "stale"
    if bounded:
        assert second["metadata"]["loss"]["formula_version"] == "bounded-fixed-grouped-v1"
        assert 0 <= second["objective"] <= 1
        assert (
            second["metadata"]["loss"]["legacy_loss_breakdown"]["components"]["rosetta_interface_dg"]["contribution"]
            == -2
        )
    else:
        assert second["metadata"]["loss"]["components"]["rosetta_interface_dg"]["contribution"] == -2


def test_post_refold_reruns_analysis_and_replaces_old_scores(harness, monkeypatch):
    outputs = []

    def analyze(*args, **kwargs):
        outputs.append(kwargs["output_dir"])
        assert "post_refold" in str(outputs[-1])
        return [success(str(outputs[-1] / "relaxed.pdb")), success()]

    monkeypatch.setattr(pyrosetta_analysis, "analyze_batch", analyze)
    request = payload(post_refold=True)
    request["candidates"][0]["metrics"] = {"rosetta_interface_dg": -1000}
    request["candidates"][0]["metadata"] = {"pyrosetta": {"status": "failed"}}
    for _ in range(2):
        first = harness._fold(request)["candidates"][0]
        assert first["metrics"]["rosetta_interface_dg"] == -20
        assert first["metadata"]["pyrosetta"]["status"] == "success"
    assert outputs[0] != outputs[1]


async def test_design_agent_receives_selected_parent_residue_scores_and_failed_sibling_feedback():
    from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
    from opendde_harness.plugin.protein_design.core.contracts import AnalyzeAgentOutput
    from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader

    config = WorkflowConfigLoader.config_from_path(
        str(Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml")
    )
    parent = Candidate(
        candidate_id="selected",
        sequence=next(iter(config.binder_chains.values())),
        metadata={"chains": config.binder_chains, "pyrosetta": success().model_dump()},
    )
    failed = Candidate(
        candidate_id="failed",
        sequence=parent.sequence,
        metadata={"pyrosetta": {"status": "failed", "error": "timed out", "contact_residues": []}},
    )
    feedback = DesignOrchestrator._search_feedback([failed], parent, [parent], config, None, {})
    assert json.loads(feedback)["candidate_pyrosetta_evidence"]["failed"]["error"] == "timed out"
    config.metadata["gate_feedback"] = feedback
    session = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("prompt captured")))
    memory = SimpleNamespace(retrieve=AsyncMock(return_value=[]), retrieve_skills=AsyncMock(return_value=[]))
    phases = ProteinDesignPhases(session, None, memory)
    with pytest.raises(RuntimeError, match="prompt captured"):
        await phases.design_cycle(config, 1, AnalyzeAgentOutput(downstream_header="test"), [parent.model_dump()], None)
    prompt = session.run.call_args.args[1]
    assert '"bound_score_reu": -3.0' in prompt
    assert '"residue_index": 1' in prompt
    assert "zero-based within `chain_id`" in prompt
    assert "timed out" in prompt
