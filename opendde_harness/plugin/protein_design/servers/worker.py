"""Detached protein-design worker entry point."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from pathlib import Path

from loguru import logger

from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.agents.skills import ProteinDesignSkillCatalog
from opendde_harness.plugin.protein_design.core.contracts import TaskSnapshot, TaskState, WorkflowConfig
from opendde_harness.plugin.protein_design.core.detached import (
    TaskFileStore,
    TaskLeaseHeartbeat,
    install_stop_signal_handlers,
)
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient
from opendde_harness.plugin.protein_design.servers.compute_pool import ComputePool
from opendde_harness.plugin.protein_design.tools.agent import ProteinDesignToolRegistry


def _with_compute_binding(snapshot: TaskSnapshot, workflow: WorkflowConfig) -> TaskSnapshot:
    """Keep the task's selected compute endpoint in every durable snapshot."""

    return snapshot.model_copy(
        update={
            "compute_url": workflow.compute_url,
            "compute_worker_id": workflow.compute_worker_id,
        }
    )


async def _watch_control(
    store: TaskFileStore,
    task_id: str,
    stop_event: asyncio.Event,
    adjustments: dict,
) -> None:
    revision = -1
    while not stop_event.is_set():
        control = store.read_control(task_id)
        current_revision = int(control.get("revision", 0))
        if current_revision != revision:
            revision = current_revision
            pending = dict(control.get("adjustments") or {})
            adjustments.update(pending)
            if pending:
                store.acknowledge_adjustments(task_id, revision)
            if control.get("stop_requested"):
                stop_event.set()
                return
        await asyncio.sleep(1.0)


async def run_task(task_id: str, task_root: Path) -> TaskSnapshot:
    from opendde_harness.cli._helpers import load_runtime_config, make_lazy_provider
    from opendde_harness.cli._plugin_stack import build_plugin_registry, maybe_build_memory_backend
    from opendde_harness.config.opendde_harness import load_opendde_harness_config

    store = TaskFileStore(task_root)
    workflow = store.read_workflow(task_id)
    snapshot = store.read_snapshot(task_id, reconcile=False)
    runtime_config = load_runtime_config(None, None)
    opendde_harness_config = load_opendde_harness_config()
    plugin_config = dict(opendde_harness_config.plugins.config.get("protein-design") or {})
    pool = ComputePool.from_config(plugin_config)
    selection = await pool.select(workflow)
    workflow = workflow.model_copy(
        update={
            "compute_url": selection.worker.url,
            "compute_worker_id": selection.worker.worker_id,
        }
    )
    provider = make_lazy_provider(runtime_config)
    registry = build_plugin_registry(opendde_harness_config)
    backend = maybe_build_memory_backend(
        runtime_config.workspace_path,
        opendde_harness_config,
        registry=registry,
    )
    if backend is not None:
        await backend.start()
    client: ProteinDesignComputeClient = selection.client
    session = OpenDDEHarnessStructuredSession(
        provider,
        str(runtime_config.agents.defaults.model),
        tool_registry=ProteinDesignToolRegistry.for_compute(client),
        max_attempts=int(plugin_config.get("structured_output_attempts", 3)),
    )
    memory = DesignMemory(
        backend,
        # The memory service assigns stored assistant/tool messages to the backend's
        # canonical agent owner.  Recall must use that same identity; a second
        # plugin-level default previously wrote under ``default`` and searched
        # under ``protein-design``, yielding an apparently healthy empty store.
        agent_id=str(getattr(backend, "agent_id", None) or plugin_config.get("memory_agent_id", "default")),
        app_id="protein-design",
        project_id=None,
    )
    phases = ProteinDesignPhases(
        session,
        client,
        memory,
        catalog=ProteinDesignSkillCatalog.builtin(),
    )
    orchestrator = DesignOrchestrator(
        client,
        memory,
        phases,
        fold_poll_interval=float(plugin_config.get("fold_poll_interval", 2.0)),
    )
    stop_event = asyncio.Event()
    adjustments: dict = {}
    install_stop_signal_handlers(stop_event.set)
    watcher = asyncio.create_task(
        _watch_control(store, task_id, stop_event, adjustments),
        name=f"protein-design-control-{task_id}",
    )
    running = _with_compute_binding(snapshot, workflow).model_copy(
        update={
            "status": TaskState.RUNNING,
            "error": None,
        }
    )
    store.write_snapshot(running)
    try:
        async with TaskLeaseHeartbeat(client, task_id):
            result = await orchestrator.run(
                task_id,
                workflow,
                stop_event=stop_event,
                adjustments=adjustments,
                on_progress=lambda progress: store.write_snapshot(
                    _with_compute_binding(progress, workflow)
                ),
            )
        result = _with_compute_binding(result, workflow)
        store.write_snapshot(result)
        return result
    except Exception as exc:
        logger.exception("protein-design task {} failed", task_id)
        failed = _with_compute_binding(
            store.read_snapshot(task_id, reconcile=False), workflow
        ).model_copy(
            update={"status": TaskState.FAILED, "error": str(exc)}
        )
        store.write_snapshot(failed)
        return failed
    finally:
        stop_event.set()
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        await client.close()
        with contextlib.suppress(Exception):
            await session.close()
        if backend is not None:
            with contextlib.suppress(Exception):
                await backend.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(run_task(args.task_id, args.task_root))
    except Exception as exc:
        logger.exception("protein-design worker {} failed during startup", args.task_id)
        store = TaskFileStore(args.task_root)
        with contextlib.suppress(Exception):
            snapshot = store.read_snapshot(args.task_id, reconcile=False)
            store.write_snapshot(snapshot.model_copy(update={"status": TaskState.FAILED, "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
