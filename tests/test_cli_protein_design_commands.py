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


@pytest.fixture
def saved_filter_task(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.core.contracts import Candidate, WorkflowConfig

    monkeypatch.setenv("OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT", str(tmp_path / "tasks"))
    store = TaskFileStore()
    task_id = "c" * 32
    candidate = Candidate(
        candidate_id="one",
        sequence="A",
        objective=1,
        structure_path="saved.cif",
        metadata={"post_refold_success": True},
    )
    workflow = WorkflowConfig(target="example")
    snapshot = commands.TaskSnapshot(
        task_id=task_id, status="completed", target="example", total_cycles=3, final_candidates=[candidate]
    )
    store.create(task_id, workflow, snapshot)
    monkeypatch.setattr(commands, "_load_plugin_config", lambda _: {})
    return store, task_id, workflow, candidate


def test_post_filter_cli_preserves_task_and_writes_separate_output(saved_filter_task, tmp_path, monkeypatch):
    store, task_id, workflow, candidate = saved_filter_task
    before = {p.name: p.read_bytes() for p in store.task_dir(task_id).iterdir()}

    async def run(config, candidates, analysis, config_path, plugin_config):
        assert config.post_filter_top_k == 1
        assert candidates == [candidate]
        assert "unavailable" in analysis.downstream_header
        return {"selected_candidate_ids": ["one"]}

    monkeypatch.setattr(commands, "_run_post_filter", run)
    output = tmp_path / "selection.json"
    args = ["post-filter", "--task-id", task_id, "--output", str(output), "--top-k", "1", "--json"]
    result = CliRunner().invoke(commands.protein_design_app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == json.loads(result.output)
    assert json.loads(result.output)["analysis_available"] is False
    assert {p.name: p.read_bytes() for p in store.task_dir(task_id).iterdir()} == before
    assert CliRunner().invoke(commands.protein_design_app, args).exit_code == 1


async def test_post_filter_runs_only_filter_and_closes_session(saved_filter_task, monkeypatch):
    from types import SimpleNamespace

    from opendde_harness.cli import _helpers
    from opendde_harness.plugin.protein_design.agents import session
    from opendde_harness.plugin.protein_design.core.contracts import AnalyzeAgentOutput, PostFilterAgentOutput

    store, task_id, workflow, candidate = saved_filter_task
    called = []
    monkeypatch.setattr(
        _helpers,
        "load_runtime_config",
        lambda _: SimpleNamespace(agents=SimpleNamespace(defaults=SimpleNamespace(model="test"))),
    )
    monkeypatch.setattr(_helpers, "make_provider", lambda _: object())

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, profile, prompt, **kwargs):
            called.append(profile.role.value)
            assert kwargs["tool_context"].compute is None
            result = PostFilterAgentOutput(
                strategy_summary="test",
                decisions=[{"candidate_id": "one", "rank": 1, "rationale": "test", "strengths": [], "risks": []}],
            )
            kwargs["output_validator"](result)
            return result

        async def close(self):
            called.append("closed")

    monkeypatch.setattr(session, "OpenDDEHarnessStructuredSession", Session)
    result = await commands._run_post_filter(
        workflow, [candidate], AnalyzeAgentOutput(downstream_header="saved"), None, {}
    )
    assert called == ["post_filter", "closed"]
    assert result["selected_candidate_ids"] == ["one"]


async def test_post_filter_rejects_unready_candidates_before_provider(saved_filter_task, monkeypatch):
    from opendde_harness.cli import _helpers
    from opendde_harness.plugin.protein_design.core.contracts import AnalyzeAgentOutput

    store, task_id, workflow, candidate = saved_filter_task
    monkeypatch.setattr(_helpers, "load_runtime_config", lambda _: pytest.fail("must not initialize provider"))
    analysis = AnalyzeAgentOutput(downstream_header="saved")
    with pytest.raises(ValueError, match="No eligible"):
        await commands._run_post_filter(workflow, [], analysis, None, {})
    with pytest.raises(ValueError, match="unique"):
        await commands._run_post_filter(workflow, [candidate, candidate], analysis, None, {})
    with pytest.raises(ValueError, match="No eligible"):
        await commands._run_post_filter(workflow, [candidate.model_copy(update={"metadata": {}})], analysis, None, {})


@pytest.mark.parametrize("mode", ["running", "bad_analysis", "agent_failure"])
def test_post_filter_errors_do_not_write_output(saved_filter_task, tmp_path, monkeypatch, mode):
    store, task_id, workflow, candidate = saved_filter_task
    output = tmp_path / "selection.json"
    args = ["post-filter", "--task-id", task_id, "--output", str(output)]
    if mode == "running":
        snapshot = store.read_snapshot(task_id, reconcile=False).model_copy(update={"status": "running"})
        store.write_snapshot(snapshot)
    elif mode == "bad_analysis":
        analysis = tmp_path / "analysis.json"
        analysis.write_text("{}")
        args.extend(["--analysis", str(analysis)])

    async def fail(*args):
        if mode != "agent_failure":
            pytest.fail("invalid inputs must not invoke the agent")
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(commands, "_run_post_filter", fail)
    result = CliRunner().invoke(commands.protein_design_app, args)
    assert result.exit_code == 1
    assert "post-filter failed" in result.output
    assert not output.exists()
