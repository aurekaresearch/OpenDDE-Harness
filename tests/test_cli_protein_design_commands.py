"""Protein-design YAML validation summaries expose reviewed cycle policies."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from opendde_harness.cli import protein_design_commands as commands
from opendde_harness.plugin.protein_design.core.detached import DetachedDesignTaskController, TaskFileStore


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
