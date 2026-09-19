"""The detached design worker must import the installed package, not the launcher's cwd."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from opendde_harness.plugin.protein_design.core.detached import DetachedDesignTaskController, TaskFileStore


def test_the_worker_never_puts_the_launchers_cwd_on_its_path(tmp_path):
    """A TUI started inside another checkout once made the worker import that
    checkout's older package (``python -m`` prepends the cwd) and refuse the
    workflow the current CLI had written. ``-P`` is what prevents it."""
    controller = DetachedDesignTaskController({}, store=TaskFileStore(tmp_path), python_executable=sys.executable)
    argv = controller.worker_argv("task-1")
    assert argv[:3] == [sys.executable, "-P", "-m"]
    assert "opendde_harness.plugin.protein_design.servers.worker" in argv
    assert argv[argv.index("--task-id") + 1] == "task-1"


def test_detached_worker_inherits_the_launchers_explicit_config_path(tmp_path, monkeypatch):
    from opendde_harness.config import loader

    config_path = tmp_path / "custom" / "config.json"
    monkeypatch.setattr(loader, "_current_config_path", config_path)
    controller = DetachedDesignTaskController({}, store=TaskFileStore(tmp_path / "tasks"))
    argv = controller.worker_argv("task-1")
    assert argv[argv.index("--opendde-config") + 1] == str(config_path)


@pytest.mark.asyncio
async def test_detached_worker_loads_provider_and_compute_from_the_same_config(tmp_path, monkeypatch):
    from opendde_harness.cli import _helpers
    from opendde_harness.config import opendde_harness
    from opendde_harness.plugin.protein_design.servers import worker

    config_path = tmp_path / "custom-config.json"
    plugin_config = {"compute_url": "http://configured-worker:9876"}
    calls = []
    monkeypatch.setattr(worker.TaskFileStore, "read_workflow", lambda *a: object())
    monkeypatch.setattr(worker.TaskFileStore, "read_snapshot", lambda *a, **k: object())
    monkeypatch.setattr(_helpers, "load_runtime_config", lambda path, _: calls.append(("runtime", path)))

    def load_plugin(path):
        calls.append(("plugin", path))
        return SimpleNamespace(plugins=SimpleNamespace(config={"protein-design": plugin_config}))

    def select_pool(config):
        assert config == plugin_config
        raise RuntimeError("selected the configured pool")

    monkeypatch.setattr(opendde_harness, "load_opendde_harness_config", load_plugin)
    monkeypatch.setattr(worker.ComputePool, "from_config", select_pool)
    with pytest.raises(RuntimeError, match="selected the configured pool"):
        await worker.run_task("task-1", tmp_path / "tasks", config_path)
    assert calls == [("runtime", str(config_path)), ("plugin", config_path)]
