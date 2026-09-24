"""Protein-design YAML validation summaries expose reviewed cycle policies."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from opendde_harness.cli import protein_design_commands as commands
from opendde_harness.plugin.protein_design.core.contracts import Placement
from opendde_harness.plugin.protein_design.core.detached import DetachedDesignTaskController, TaskFileStore


@pytest.mark.parametrize("degree", [3])
def test_placement_rejects_silently_unused_or_missing_devices(degree):
    with pytest.raises(ValueError, match="exactly cp_degree"):
        Placement(fold=[0, 1], cp_degree=degree)


def test_placement_default_does_not_silently_discard_extra_gpus():
    assert Placement(fold=[0, 1]).fold == [0, 1]


def test_validation_reports_candidate_parallelism_by_default():
    source = Path(__file__).parents[1] / "docs/examples/crlf2_scheduled.yaml"
    data = yaml.safe_load(source.read_text())
    data["compute"] = {"placement": {"fold": [0, 1, 2]}}
    workflow = commands.WorkflowConfigLoader._normalize_config(data)
    summary = commands._workflow_summary(workflow)
    assert summary["compute"]["fold_gpu_count"] == 3
    assert summary["compute"]["fold_parallelism"] == "candidate_parallel"
    assert summary["compute"]["candidate_data_parallelism"] is True


def test_validation_exposes_parallelism_and_external_hotspots():
    source = Path(__file__).parents[1] / "docs/examples/crlf2_scheduled.yaml"
    data = yaml.safe_load(source.read_text())
    data["compute"] = {"placement": {"fold": [0, 1], "cp_degree": 2, "esm": 0, "mpnn": 0}}
    chain = next(iter(data["target"]["chains"]))
    data["target"]["chains"][chain]["hotspots"] = [1, 39]
    workflow = commands.WorkflowConfigLoader._normalize_config(data)
    summary = commands._workflow_summary(workflow)
    assert summary["target_hotspots_1based"][chain] == [1, 39]
    assert summary["compute"]["fold_gpu_count"] == 2
    assert summary["compute"]["fold_parallelism"] == "context_parallel"
    assert summary["compute"]["candidate_data_parallelism"] is False


@pytest.fixture
def task_store(tmp_path, monkeypatch):
    store = TaskFileStore(tmp_path)
    controller = DetachedDesignTaskController({}, store=store)
    monkeypatch.setattr(commands, "DetachedDesignTaskController", lambda _: controller)
    return store


def test_status_reports_snapshot_in_text_and_json(task_store):
    task_id = "a" * 32
    task_store.task_dir(task_id).mkdir()
    snapshot = commands.TaskSnapshot(
        task_id=task_id,
        status="failed",
        target="example",
        cycle=2,
        total_cycles=5,
        phase="design",
        error="example failure",
    )
    task_store.write_snapshot(snapshot)
    runner = CliRunner()
    result = runner.invoke(commands.protein_design_app, ["status", "--task-id", task_id])
    assert result.exit_code == 0, result.output
    assert "cycle 2/5" in result.output and "example failure" in result.output
    result = runner.invoke(commands.protein_design_app, ["status", "--task-id", task_id, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == snapshot.model_dump(mode="json")
    result = runner.invoke(commands.protein_design_app, ["status", "--json"])
    assert json.loads(result.output) == [snapshot.model_dump(mode="json")]


def test_status_empty_list(task_store):
    runner = CliRunner()
    assert json.loads(runner.invoke(commands.protein_design_app, ["status", "--json"]).output) == []
    assert "No protein-design tasks" in runner.invoke(commands.protein_design_app, ["status"]).output


@pytest.mark.parametrize("task_id", ["b" * 32, "../../invalid"])
def test_status_missing_or_invalid_task_is_a_clean_error(task_store, task_id):
    result = CliRunner().invoke(commands.protein_design_app, ["status", "--task-id", task_id, "--json"])
    assert result.exit_code == 1
    assert "Protein-design status failed" in result.output
    assert "Traceback" not in result.output


def test_validate_displays_cycle_schedule(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "docs/examples/crlf2_scheduled.yaml"
    monkeypatch.setattr(commands, "_load_plugin_config", lambda _: {})
    result = CliRunner().invoke(commands.protein_design_app, ["validate", "--config", str(source), "--json"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["router_selection_strategy"] == "weighted"
    assert [stage["num_sequences"] for stage in summary["cycle_schedule"]] == [8, 4, 2]


def test_validate_rejects_overlapping_stages(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "docs/examples/crlf2_scheduled.yaml"
    data = yaml.safe_load(source.read_text())
    data["design"]["cycle_schedule"][1]["start_cycle"] = 3
    path = tmp_path / "overlap.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(commands, "_load_plugin_config", lambda _: {})
    result = CliRunner().invoke(commands.protein_design_app, ["validate", "--config", str(path), "--json"])
    assert result.exit_code == 1
    assert "overlap" in result.output
