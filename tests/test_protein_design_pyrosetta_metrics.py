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


def test_required_metric_failure_cannot_continue_into_loss_ranking(harness, monkeypatch):
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
        )

    monkeypatch.setattr(loss_confidence_scorer, "score_confidence_loss", score)
    first, second = harness._fold(
        payload(
            objective_key="loss",
            pyrosetta={"enabled": True, "on_failure": "continue"},
            metric_loss_terms={"rosetta_interface_dg": {"direction": "minimize", "scale": 10.0}},
        )
    )["candidates"]
    assert first["objective"] is None
    assert first["metadata"]["success"] is False
    assert "Missing required loss metric" in first["metadata"]["error"]
    assert "loss" not in first["metrics"]
    assert second["metrics"]["loss"] == second["objective"]
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
