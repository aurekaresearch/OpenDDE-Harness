"""The detached worker closes its process-wide model service before its loop."""

from __future__ import annotations

import asyncio
import sys

import pytest

from opendde_harness.plugin.protein_design.servers import worker
from opendde_harness.providers import pi_service
from opendde_harness.providers.model_service import ModelService


@pytest.mark.parametrize("failed", [False, True])
def test_worker_main_closes_model_service_on_its_own_loop(tmp_path, monkeypatch, failed):
    child = tmp_path / "child.py"
    child.write_text("import sys\nsys.stdin.read()\n")
    service = ModelService(node=sys.executable, bundle=child)
    processes = []
    owner_loops = []
    config_path = tmp_path / "config.json"
    task_root = tmp_path / "tasks"

    async def run_task(task_id, root, config):
        assert (task_id, root, config) == ("task-1", task_root, config_path)
        await service.start()
        processes.append(service._proc)
        owner_loops.append(asyncio.get_running_loop())
        pi_service._service = service
        pi_service._service_loop = owner_loops[-1]
        if failed:
            raise RuntimeError("task startup failed")

    monkeypatch.setattr(pi_service, "_service", None)
    monkeypatch.setattr(pi_service, "_service_loop", None)
    monkeypatch.setattr(pi_service, "_fingerprint", None)
    monkeypatch.setattr(worker, "run_task", run_task)
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "--task-id", "task-1", "--task-root", str(task_root), "--opendde-config", str(config_path)],
    )

    if failed:
        with pytest.raises(RuntimeError, match="task startup failed"):
            worker.main()
    else:
        worker.main()

    assert processes[0].returncode == 0
    assert service.running is False
    assert owner_loops[0].is_closed()
    assert pi_service._service is None
    assert pi_service._service_loop is None
