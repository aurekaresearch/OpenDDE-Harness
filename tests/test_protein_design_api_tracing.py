"""API fold archives must remain readable through compute and tracing."""

import json
from pathlib import Path

import httpx

from opendde_harness.plugin.protein_design.core.contracts import Candidate
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.plugin.protein_design.core.tracing import attach_artifact, build_cycle_artifact
from opendde_harness.plugin.protein_design.servers.api import create_app
from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient
from opendde_harness.plugin.protein_design.servers.harness import PythonProteinDesignHarness
from opendde_harness.tracing import spans
from opendde_harness.tracing.trace import span


async def capture_fold_trace(run_dir: Path, trace_dir: Path, sequences=None):
    """Exercise the production result parser, compute read API and trace capture."""
    predictor = StructurePredictor(FoldConfig(execution_mode="api"))
    sequences = sequences or {"H": "EVQLVESGGG"}
    result = predictor._parse_opendde_batch_result(sequences, run_dir, 0)
    assert result.success
    assert result.all_atom_confidence_path.is_file()
    candidate = Candidate(
        candidate_id="api-candidate",
        sequence=sequences["H"],
        objective=result.iptm,
        metrics={"iptm": result.iptm, "plddt": result.plddt},
        structure_path=str(result.structure_path[0]),
        metadata={"fold": result.to_dict(), "chains": {"H": sequences["H"]}, "success": True},
    )
    task_id = "api-fold-smoke"
    compute = PythonProteinDesignHarness(output_path=str(run_dir.parent.parent))
    client = ProteinDesignComputeClient(
        "http://compute.test", transport=httpx.ASGITransport(app=create_app(harness=compute))
    )
    orchestrator = object.__new__(DesignOrchestrator)
    orchestrator._compute = client
    attributes = {
        "protein_design.task_id": task_id,
        "protein_design.target": "API gateway smoke",
        "protein_design.status": "completed",
        "protein_design.cycle": 0,
        "protein_design.total_cycles": 1,
        "protein_design.objective_key": "iptm",
        "protein_design.minimize": False,
    }
    try:
        with span("protein_design.run", attributes, session_key=task_id):
            with span("protein_design.cycle", {**attributes, "protein_design.cycle_index": 0}) as cycle:
                structures = await orchestrator._persist_candidate_structures(
                    cycle, [candidate], task_id=task_id, target_chain_ids=["A"], binder_chain_ids=["H"]
                )
                assert "structure_artifact_error" not in candidate.metadata
                assert candidate.candidate_id in structures
                payload = build_cycle_artifact(
                    task_id=task_id,
                    target="API gateway smoke",
                    cycle=0,
                    objective_key="iptm",
                    minimize=False,
                    candidates=[candidate],
                    cycle_best_id=candidate.candidate_id,
                    global_best_id=candidate.candidate_id,
                    known_candidate_ids={candidate.candidate_id},
                    structure_artifacts=structures,
                )
                artifact_path = attach_artifact(cycle, "protein_design.cycle", payload)
                assert artifact_path
    finally:
        await client.close()
    artifact = json.loads(Path(structures[candidate.candidate_id]).read_text())
    assert artifact["format"] == "cif"
    assert artifact["text"] == result.structure_path[0].read_text()
    assert artifact["byte_count"] > 0
    assert artifact["binder_chain_ids"] == ["H"]
    assert (trace_dir / "logs/audit-spans.log").is_file()
    return payload


async def test_api_archive_structure_reaches_tracing(tmp_path, monkeypatch):
    from opendde_harness.tracing.store import TraceStore

    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("OPENDDE_HARNESS_TRACING", "1")
    monkeypatch.setattr(spans, "_store", TraceStore(trace_dir))
    run_dir = tmp_path / "compute/api-fold-smoke/api_run"
    predictions = run_dir / "output/input_0000/input_0000/seed_0/predictions"
    predictions.mkdir(parents=True)
    (predictions / "input_0000_sample_0.cif").write_text("data_input_0000\n")
    (predictions / "input_0000_summary_confidence_sample_0.json").write_text(
        json.dumps({"iptm": 0.7, "ptm": 0.8, "plddt": 80, "ranking_score": 0.72})
    )
    (predictions / "input_0000_full_data_sample_0.json").write_text(json.dumps({"atom_plddt": [0.8]}))

    payload = await capture_fold_trace(run_dir, trace_dir)

    candidate = payload["candidates"][0]
    assert candidate["metrics"]["iptm"] == 0.7
    assert candidate["structure_artifact_path"].startswith(str(trace_dir))
