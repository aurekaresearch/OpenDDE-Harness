import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends import pyrosetta_analysis as analysis
from opendde_harness.plugin.protein_design.servers.backends import pyrosetta_worker as worker


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "false"},
        {"max_workers": 0},
        {"relax_repeats": 0},
        {"max_iter": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": -1},
        {"seed": 0},
        {"contact_distance": 0},
        {"contact_distance": float("nan")},
        {"on_failure": "ignore"},
        {"unknown": 1},
    ],
)
def test_analysis_rejects_invalid_configuration(options):
    with pytest.raises(ValidationError):
        analysis.PyRosettaConfig(**options)


def test_existing_workflow_defaults_and_new_options():
    path = Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml"
    old = WorkflowConfigLoader.config_from_path(str(path))
    assert "pyrosetta" not in old.fold_options
    data = old.metadata["source_config"]
    data["fold"]["pyrosetta"] = {"enabled": True, "max_workers": 2}
    new = WorkflowConfigLoader._normalize_config(data)
    assert new.fold_options["pyrosetta"]["relax_repeats"] == 5
    assert new.fold_options["pyrosetta"]["max_workers"] == 2
    assert new.fold_options["loss_weights"] == old.fold_options["loss_weights"]


@pytest.mark.parametrize("binder,target", [([], ["A"]), (["B"], ["B"]), (["BB"], ["A"]), (["_"], ["A"])])
def test_interface_rejects_ambiguous_chain_groups(binder, target):
    with pytest.raises(ValueError, match="PyRosetta requires"):
        analysis.interface_chains(binder, target)


def test_multichain_interface_and_affinity_limit(monkeypatch):
    assert analysis.interface_chains(["H", "L"], ["A", "B"]) == "HL_AB"
    monkeypatch.setattr(analysis.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    assert analysis.cpu_workers(8, 10) == 2
    assert analysis.cpu_workers(8, 1) == 1


def test_disabled_analysis_does_not_start_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "_analyze_one", lambda *args: pytest.fail("worker started"))
    results = analysis.analyze_batch(
        ["absent.pdb"],
        [{}],
        binder_chains=[],
        target_chains=[],
        config=analysis.PyRosettaConfig(),
        output_dir=tmp_path / "analysis",
    )
    assert results[0].status == "disabled"
    assert not (tmp_path / "analysis").exists()


def test_parallel_jobs_preserve_order_and_isolate_failures(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    monkeypatch.setattr(analysis, "cpu_workers", lambda *_: 2)

    def run(structure, sequences, interface, config, output_dir):
        barrier.wait(timeout=5)
        assert interface == "B_A"
        return analysis.PyRosettaResult(status="failed", error=structure)

    monkeypatch.setattr(analysis, "_analyze_one", run)
    results = analysis.analyze_batch(
        ["first.pdb", "second.pdb", None],
        [{}, {}, {}],
        binder_chains=["B"],
        target_chains=["A"],
        config=analysis.PyRosettaConfig(enabled=True),
        output_dir=tmp_path,
    )
    assert [result.error for result in results[:2]] == ["first.pdb", "second.pdb"]
    assert results[2].status == "skipped"
    assert all(result.elapsed_seconds > 0 for result in results[:2])


def _run_one(tmp_path):
    return analysis._analyze_one(
        "complex.pdb", {"A": "A", "B": "A"}, "B_A", analysis.PyRosettaConfig(enabled=True), tmp_path / "result"
    )


@pytest.mark.parametrize("failure", ["timeout", "crash", "malformed", "nonfinite"])
def test_worker_failures_do_not_publish_partial_metrics(tmp_path, monkeypatch, failure):
    def run(command, **kwargs):
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "crash":
            return SimpleNamespace(returncode=-11)
        output = Path(json.loads(kwargs["input"])["output_dir"])
        payload = {
            "status": "success",
            "metrics": {key: float("nan") for key in analysis.PYROSETTA_METRICS},
            "relaxed_structure_path": str(output / "relaxed.pdb"),
        }
        (output / "analysis.json").write_text("broken" if failure == "malformed" else json.dumps(payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(analysis.subprocess, "run", run)
    result = _run_one(tmp_path)
    assert result.status == "failed"
    assert result.error
    assert result.metrics == {}


@pytest.fixture
def fake_pyrosetta(monkeypatch):
    events = []
    pose = MagicMock()
    pose.total_residue.return_value = 2
    pose.pdb_info.return_value.chain.side_effect = lambda i: "AB"[i - 1]
    pose.residue.return_value.is_virtual_residue.return_value = False
    pose.residue.return_value.name1.return_value = "A"
    pose.residue.return_value.nheavyatoms.return_value = 1
    pose.residue.return_value.xyz.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    pose.pdb_info.return_value.number.side_effect = lambda i: i * 10
    pose.pdb_info.return_value.icode.return_value = " "
    scorefxn = MagicMock(return_value=-50.0)
    relax = MagicMock()
    relax.apply.side_effect = lambda _: events.append("relax")
    pose.remove_constraints.side_effect = lambda: events.append("remove_constraints")
    analyzer = MagicMock()
    analyzer.apply.side_effect = lambda _: events.append("analyze")
    analyzer.get_interface_delta_sasa.return_value = 1000.0
    analyzer.get_interface_dG.return_value = -20.0
    analyzer.get_all_data.return_value = SimpleNamespace(sc_value=0.65, interface_hbonds=5)
    analyzer.get_interface_delta_hbond_unsat.return_value = 2
    analyzer.get_num_interface_residues.return_value = 12
    analyzer.get_all_per_residue_data.return_value = SimpleNamespace(
        complexed_energy={1: -3.0, 2: -5.0}, separated_energy={1: -2.0, 2: -3.0}, dG={1: -1.0, 2: -2.0}
    )
    pyrosetta = MagicMock()
    pyrosetta.pose_from_file.return_value = pose
    pyrosetta.create_score_function.return_value = scorefxn
    pyrosetta.rosetta.protocols.relax.FastRelax.return_value = relax
    pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover.return_value = analyzer
    monkeypatch.setitem(sys.modules, "pyrosetta", pyrosetta)
    return pyrosetta, pose, relax, analyzer, events


def _request(tmp_path):
    return {
        "config": analysis.PyRosettaConfig(enabled=True).model_dump(),
        "structure_path": "complex.pdb",
        "output_dir": str(tmp_path),
        "sequences": {"A": "A", "B": "A"},
        "interface": "B_A",
    }


def test_fastrelax_precedes_interface_analysis_on_the_same_pose(tmp_path, fake_pyrosetta):
    pyrosetta, pose, relax, analyzer, events = fake_pyrosetta
    result = worker.relax_and_analyze(_request(tmp_path))
    assert events == ["relax", "remove_constraints", "analyze"]
    relax.apply.assert_called_once_with(pose)
    analyzer.apply.assert_called_once_with(pose)
    relax.constrain_relax_to_start_coords.assert_called_once_with(True)
    relax.max_iter.assert_called_once_with(200)
    pyrosetta.rosetta.core.kinematics.MoveMap.return_value.set_jump.assert_called_once_with(False)
    analyzer.set_pack_separated.assert_called_once_with(True)
    analyzer.set_pack_input.assert_called_once_with(False)
    partners = pyrosetta.rosetta.core.pose.DockingPartners.docking_partners_from_string
    partners.assert_called_once_with("B_A")
    analyzer.set_interface.assert_called_once_with(partners.return_value)
    assert result.metrics["rosetta_interface_dg_per_sasa"] == -2.0
    assert result.metrics["rosetta_interface_sc"] == 0.65
    assert set(result.metrics) == set(analysis.PYROSETTA_METRICS)
    pose.dump_pdb.assert_called_once_with(str(tmp_path / "relaxed.pdb"))


def test_failed_relaxation_never_runs_interface_analyzer(tmp_path, fake_pyrosetta):
    _, _, relax, analyzer, _ = fake_pyrosetta
    relax.apply.side_effect = RuntimeError("relax failed")
    with pytest.raises(RuntimeError, match="relax failed"):
        worker.relax_and_analyze(_request(tmp_path))
    analyzer.apply.assert_not_called()


def test_chain_mismatch_fails_before_relaxation(tmp_path, fake_pyrosetta):
    _, _, relax, _, _ = fake_pyrosetta
    request = _request(tmp_path)
    request["sequences"]["A"] = "AA"
    with pytest.raises(ValueError, match="chain sequences differ"):
        worker.relax_and_analyze(request)
    relax.apply.assert_not_called()


def test_absent_interface_fails_without_zero_metric_substitution(tmp_path, fake_pyrosetta):
    _, _, _, analyzer, _ = fake_pyrosetta
    analyzer.get_interface_delta_sasa.return_value = 0
    with pytest.raises(ValueError, match="no buried interface"):
        worker.relax_and_analyze(_request(tmp_path))


def test_contact_residue_scores_use_heavy_atoms_and_zero_based_chain_positions(fake_pyrosetta):
    _, pose, _, _, _ = fake_pyrosetta
    pose.total_residue.return_value = 4
    pose.pdb_info.return_value.chain.side_effect = lambda i: "BBAA"[i - 1]
    pose.pdb_info.return_value.icode.return_value = "A"
    residues = []
    for x in (100, 0, 5, 200):
        residue = MagicMock()
        residue.is_virtual_residue.return_value = False
        residue.name1.return_value = "W"
        residue.nheavyatoms.return_value = 1
        residue.xyz.side_effect = lambda atom, x=x: SimpleNamespace(x=float(x), y=0.0, z=0.0)
        residues.append(residue)
    pose.residue.side_effect = lambda i: residues[i - 1]
    pose.energies.return_value.residue_total_energy.side_effect = lambda i: -float(i)
    data = SimpleNamespace(
        complexed_energy={2: -2.0, 3: -3.0}, separated_energy={2: -1.0, 3: -1.0}, dG={2: -1.0, 3: -2.0}
    )
    scores = worker._contact_residue_scores(pose, "B_A", 5.0, data)
    assert [(row.chain_id, row.residue_index) for row in scores] == [("B", 1), ("A", 0)]
    assert [row.partner for row in scores] == ["binder", "target"]
    assert scores[0].pdb_residue_number == 20
    assert scores[0].insertion_code == "A"
    assert scores[0].bound_score_reu == -2.0
    assert scores[0].interface_dg_reu == -1.0
    assert scores[1].interface_dg_reu == -2.0
    assert all(row.min_partner_distance == 5.0 for row in scores)
    assert worker._contact_residue_scores(pose, "B_A", 4.9, data) == []
    data.complexed_energy[2] = float("nan")
    with pytest.raises(ValidationError):
        worker._contact_residue_scores(pose, "B_A", 5.0, data)
