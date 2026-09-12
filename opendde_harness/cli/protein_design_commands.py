"""CLI commands for validating and launching protein-design tasks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from opendde_harness.plugin.protein_design.core.contracts import TaskSnapshot, WorkflowConfig
from opendde_harness.plugin.protein_design.core.detached import DetachedDesignTaskController
from opendde_harness.plugin.protein_design.core.preparation import (
    compute_summary,
    display_url,
    fold_summary,
    preparation_context,
)
from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader

protein_design_app = typer.Typer(
    help="Validate and launch detached protein-design tasks.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


def _load_plugin_config(config_path: Path | None) -> dict[str, Any]:
    from opendde_harness.cli._helpers import load_runtime_config
    from opendde_harness.config.opendde_harness import load_opendde_harness_config

    raw_path = str(config_path) if config_path else None
    load_runtime_config(raw_path)
    config = load_opendde_harness_config(config_path)
    if "protein-design" in config.plugins.disabled:
        raise RuntimeError("the protein-design plugin is disabled")
    return dict(config.plugins.config.get("protein-design") or {})


async def _launch(
    workflow: WorkflowConfig,
    plugin_config: dict[str, Any],
) -> TaskSnapshot:
    return await DetachedDesignTaskController(plugin_config).start(workflow)


def _workflow_summary(workflow: WorkflowConfig, plugin_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "target": workflow.target,
        "cycles": workflow.cycles,
        "candidates_per_cycle": workflow.candidates_per_cycle,
        "population_size": workflow.population_size,
        "router_selection_strategy": workflow.router_selection_strategy,
        "cycle_schedule": [stage.model_dump(exclude_none=True) for stage in workflow.cycle_schedule],
        "fold_backend": workflow.fold_backend,
        "objective_key": workflow.objective_key,
        "minimize": workflow.minimize,
        "binder_type": workflow.metadata.get("binder_type"),
        "binder_chain_ids": list(workflow.binder_chains),
        "target_chain_ids": list(workflow.target_chains),
        "mutable_residue_count": sum(len(positions) for positions in workflow.mutable_positions.values()),
        "fold": fold_summary(workflow.fold_options),
        "loss_weights": workflow.fold_options.get("loss_weights"),
        "compute": {
            **compute_summary(plugin_config or {}),
            "selection": (
                "explicit_url"
                if workflow.compute_url
                else "worker_id"
                if workflow.compute_worker_id
                else "profile"
                if workflow.compute_profile
                else compute_summary(plugin_config or {})["selection"]
            ),
            "requested_url": display_url(workflow.compute_url),
            "requested_worker_id": workflow.compute_worker_id,
            "requested_profile": workflow.compute_profile,
        },
        "post_filter_enabled": workflow.post_filter_enabled,
    }


@protein_design_app.command("context")
def show_context(
    repository_root: Path | None = typer.Option(None, "--repository-root", help="Actual source checkout, if known."),
    opendde_harness_config: Path | None = typer.Option(None, "--opendde-config", exists=True, dir_okay=False),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show safe preparation defaults and example paths without starting a task."""
    try:
        payload = preparation_context(
            _load_plugin_config(opendde_harness_config),
            str(repository_root) if repository_root else None,
        )
    except Exception as exc:
        error_console.print(
            f"[red]Cannot read design context ({type(exc).__name__}). Check the plugin configuration.[/red]"
        )
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        console.print_json(data=payload)


@protein_design_app.command("validate")
def validate_config(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Protein-design YAML configuration.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON.",
    ),
    opendde_harness_config: Path | None = typer.Option(
        None,
        "--opendde-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="OpenDDE Harness config.json path; defaults to ~/.opendde_harness/config.json.",
    ),
) -> None:
    """Validate a YAML configuration without starting a task."""

    try:
        plugin_config = _load_plugin_config(opendde_harness_config)
        workflow = WorkflowConfigLoader.config_from_path(str(config), plugin_config)
    except Exception as exc:
        error_console.print(f"[red]Invalid protein-design configuration:[/red] {exc}")
        raise typer.Exit(1) from exc

    summary = _workflow_summary(workflow, plugin_config)
    if json_output:
        typer.echo(json.dumps(summary, ensure_ascii=False))
        return
    console.print("[green]Protein-design configuration is valid.[/green]")
    for key, value in summary.items():
        console.print(f"[bold]{key}:[/bold] {value}")


@protein_design_app.command("start")
def start_task(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Reviewed protein-design YAML configuration.",
    ),
    opendde_harness_config: Path | None = typer.Option(
        None,
        "--opendde-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="OpenDDE Harness config.json path; defaults to ~/.opendde_harness/config.json.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON.",
    ),
) -> None:
    """Validate a YAML file and launch it as a detached task."""

    try:
        plugin_config = _load_plugin_config(opendde_harness_config)
        workflow = WorkflowConfigLoader.config_from_path(str(config), plugin_config)
        snapshot = asyncio.run(_launch(workflow, plugin_config))
    except Exception as exc:
        error_console.print(f"[red]Protein-design start failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    payload = snapshot.model_dump(mode="json")
    payload["fold_backend"] = workflow.fold_backend
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return
    console.print(f"[green]Started protein-design task {snapshot.task_id}.[/green]")
    console.print(f"[bold]target:[/bold] {snapshot.target}")
    console.print(f"[bold]cycles:[/bold] {snapshot.total_cycles}")
    console.print(f"[bold]fold_backend:[/bold] {workflow.fold_backend}")
    console.print(f"[bold]compute_url:[/bold] {snapshot.compute_url or 'unresolved'}")
