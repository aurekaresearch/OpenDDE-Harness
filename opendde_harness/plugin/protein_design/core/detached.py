"""Persistent control plane for detached protein-design workers."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator, Mapping

from loguru import logger

from opendde_harness.plugin.protein_design.core.contracts import (
    TargetMsaSearchRequest,
    TaskSnapshot,
    TaskState,
    WorkflowConfig,
    normalize_workflow_adjustments,
)
from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.compute_pool import ComputePool

_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def default_task_root() -> Path:
    configured = os.environ.get("OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT")
    return Path(configured).expanduser() if configured else Path.home() / ".opendde_harness" / "protein_design"


class TaskFileStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or default_task_root()).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def task_dir(self, task_id: str) -> Path:
        if not _TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"invalid protein-design task ID: {task_id!r}")
        return self.root / task_id

    def create(
        self,
        task_id: str,
        workflow: WorkflowConfig,
        snapshot: TaskSnapshot,
    ) -> Path:
        directory = self.task_dir(task_id)
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._write_json(directory / "workflow.json", workflow.model_dump(mode="json"))
        self.write_snapshot(snapshot)
        self._write_json(
            directory / "control.json",
            {"revision": 0, "stop_requested": False, "adjustments": {}},
        )
        return directory

    def read_workflow(self, task_id: str) -> WorkflowConfig:
        path = self.task_dir(task_id) / "workflow.json"
        return WorkflowConfig.model_validate_json(path.read_text(encoding="utf-8"))

    def write_snapshot(self, snapshot: TaskSnapshot) -> None:
        self._write_json(
            self.task_dir(snapshot.task_id) / "snapshot.json",
            snapshot.model_dump(mode="json"),
        )

    def read_snapshot(self, task_id: str, *, reconcile: bool = True) -> TaskSnapshot:
        path = self.task_dir(task_id) / "snapshot.json"
        if not path.exists():
            raise KeyError(f"unknown protein-design task {task_id}")
        snapshot = TaskSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        return self._reconcile(snapshot) if reconcile else snapshot

    def list_snapshots(self, *, reconcile: bool = True) -> list[TaskSnapshot]:
        def modified_at(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        snapshots: list[TaskSnapshot] = []
        for path in sorted(self.root.glob("*/snapshot.json"), key=modified_at, reverse=True):
            try:
                snapshots.append(self.read_snapshot(path.parent.name, reconcile=reconcile))
            except (KeyError, OSError, ValueError):
                continue
        return snapshots

    def write_pid(self, task_id: str, pid: int) -> None:
        self._write_json(self.task_dir(task_id) / "worker.json", {"pid": int(pid)})

    def read_pid(self, task_id: str) -> int | None:
        path = self.task_dir(task_id) / "worker.json"
        if not path.exists():
            return None
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["pid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def read_control(self, task_id: str) -> dict[str, Any]:
        path = self.task_dir(task_id) / "control.json"
        if not path.exists():
            return {"revision": 0, "stop_requested": False, "adjustments": {}}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def acknowledge_adjustments(self, task_id: str, revision: int) -> None:
        """Clear consumed adjustments without overwriting a newer control write."""
        with self._control_lock(task_id):
            control = self.read_control(task_id)
            if int(control.get("revision", 0)) != revision:
                return
            self._write_json(
                self.task_dir(task_id) / "control.json",
                {
                    "revision": revision,
                    "stop_requested": bool(control.get("stop_requested", False)),
                    "adjustments": {},
                },
            )

    def adjust(self, task_id: str, params: Mapping[str, Any]) -> TaskSnapshot:
        snapshot = self.read_snapshot(task_id)
        if snapshot.status not in {TaskState.QUEUED, TaskState.RUNNING}:
            raise RuntimeError(f"protein-design task {task_id} is not running")
        normalized = normalize_workflow_adjustments(params)
        with self._control_lock(task_id):
            control = self.read_control(task_id)
            adjustments = dict(control.get("adjustments") or {})
            adjustments.update(normalized)
            self._write_json(
                self.task_dir(task_id) / "control.json",
                {
                    "revision": int(control.get("revision", 0)) + 1,
                    "stop_requested": bool(control.get("stop_requested", False)),
                    "adjustments": adjustments,
                },
            )
        snapshot = snapshot.model_copy(update={"pending_adjustments": adjustments})
        self.write_snapshot(snapshot)
        return snapshot

    def request_stop(self, task_id: str) -> TaskSnapshot:
        snapshot = self.read_snapshot(task_id)
        if snapshot.status not in {TaskState.QUEUED, TaskState.RUNNING}:
            return snapshot
        with self._control_lock(task_id):
            control = self.read_control(task_id)
            self._write_json(
                self.task_dir(task_id) / "control.json",
                {
                    "revision": int(control.get("revision", 0)) + 1,
                    "stop_requested": True,
                    "adjustments": dict(control.get("adjustments") or {}),
                },
            )
        return snapshot

    @contextmanager
    def _control_lock(self, task_id: str) -> Iterator[None]:
        path = self.task_dir(task_id) / ".control.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _reconcile(self, snapshot: TaskSnapshot) -> TaskSnapshot:
        if snapshot.status not in {TaskState.QUEUED, TaskState.RUNNING}:
            return snapshot
        pid = self.read_pid(snapshot.task_id)
        if pid is None or self._pid_running(pid):
            return snapshot
        failed = snapshot.model_copy(
            update={
                "status": TaskState.FAILED,
                "error": f"background protein-design worker {pid} exited unexpectedly",
            }
        )
        self.write_snapshot(failed)
        return failed

    @staticmethod
    def _pid_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class DetachedDesignTaskController:
    def __init__(
        self,
        plugin_config: Mapping[str, Any],
        *,
        store: TaskFileStore | None = None,
        python_executable: str | None = None,
        worker_module: str = "opendde_harness.plugin.protein_design.servers.worker",
    ) -> None:
        self._plugin_config = dict(plugin_config)
        self._store = store or TaskFileStore()
        self._python = python_executable or sys.executable
        self._worker_module = worker_module
        self._pool = ComputePool.from_config(self._plugin_config)

    def config_from_path(self, config_path: str) -> WorkflowConfig:
        return WorkflowConfigLoader.config_from_path(config_path, self._plugin_config)

    async def start(self, workflow: WorkflowConfig) -> TaskSnapshot:
        selection = await self._pool.select(workflow)
        bound = workflow.model_copy(
            update={
                "compute_url": selection.worker.url,
                "compute_worker_id": selection.worker.worker_id,
            }
        )
        task_id = uuid.uuid4().hex
        snapshot = TaskSnapshot(
            task_id=task_id,
            status=TaskState.QUEUED,
            target=bound.target,
            compute_url=bound.compute_url,
            compute_worker_id=bound.compute_worker_id,
            total_cycles=bound.cycles,
        )
        directory = self._store.create(task_id, bound, snapshot)
        log_handle = (directory / "worker.log").open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                [
                    self._python,
                    "-m",
                    self._worker_module,
                    "--task-id",
                    task_id,
                    "--task-root",
                    str(self._store.root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        except Exception as exc:
            failed = snapshot.model_copy(update={"status": TaskState.FAILED, "error": str(exc)})
            self._store.write_snapshot(failed)
            raise
        finally:
            log_handle.close()
        self._store.write_pid(task_id, process.pid)
        return snapshot

    def status(self, task_id: str | None = None) -> TaskSnapshot | list[TaskSnapshot]:
        if task_id:
            return self._store.read_snapshot(task_id, reconcile=True)
        return self._store.list_snapshots(reconcile=True)

    def minimize(self, task_id: str) -> bool:
        return self._store.read_workflow(task_id).minimize

    def adjust(self, task_id: str, params: dict[str, Any]) -> TaskSnapshot:
        return self._store.adjust(task_id, params)

    def stop(self, task_id: str) -> TaskSnapshot:
        return self._store.request_stop(task_id)

    async def top_candidates(self, task_id: str, top_k: int) -> dict[str, Any]:
        workflow = self._store.read_workflow(task_id)
        selection = await self._pool.select(workflow)
        return await selection.client.top_candidates(
            top_k,
            task_id=task_id,
            minimize=workflow.minimize,
        )

    async def search_target_msa(
        self,
        *,
        target_name: str,
        chain_id: str,
        sequence: str,
        compute_url: str | None = None,
        compute_worker_id: str | None = None,
        compute_profile: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        placement = WorkflowConfig(
            target=target_name,
            target_sequence=sequence,
            compute_url=compute_url,
            compute_worker_id=compute_worker_id,
            compute_profile=compute_profile,
        )
        selection = await self._pool.select(placement)
        result = await selection.client.search_target_msa(
            TargetMsaSearchRequest(
                target_name=target_name,
                chain_id=chain_id,
                sequence=sequence,
                force=force,
            )
        )
        return {
            **result.model_dump(mode="json"),
            "compute_url": selection.worker.url,
            "compute_worker_id": selection.worker.worker_id,
        }


class TaskLeaseHeartbeat:
    """Keep the on-demand compute container alive for a task, including between compute calls.

    The service treats an unexpired task lease as busy for its idle watchdog and
    for ``POST /shutdown {"if_idle": true}``. Refreshes run more often than the
    lease TTL; failures are logged and never interrupt the task. The lease is
    released when the task finishes, so the container can retire on schedule.
    """

    def __init__(self, client: Any, task_id: str, *, interval: float = 60.0) -> None:
        self._client = client
        self._task_id = task_id
        self._interval = interval
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            try:
                await self._client.refresh_task_lease(self._task_id)
            except Exception as exc:
                logger.warning("protein-design task {} lease heartbeat failed: {}", self._task_id, exc)
            await asyncio.sleep(self._interval)

    async def __aenter__(self) -> "TaskLeaseHeartbeat":
        self._task = asyncio.create_task(self._run(), name=f"protein-design-lease-{self._task_id}")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        try:
            await self._client.release_task_lease(self._task_id)
        except Exception as exc:
            logger.warning("protein-design task {} lease release failed: {}", self._task_id, exc)


def install_stop_signal_handlers(stop_callback) -> None:
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda _signum, _frame: stop_callback())


__all__ = [
    "DetachedDesignTaskController",
    "TaskFileStore",
    "TaskLeaseHeartbeat",
    "default_task_root",
    "install_stop_signal_handlers",
]
