"""``ddeharness compute`` — prepare weights and runtime code, serve the compute API, or stop the local container."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import typer

compute_app = typer.Typer(help="Prepare, run or stop the Protein Design compute service.", no_args_is_help=True)


@compute_app.command("prepare")
def prepare(
    root: Optional[Path] = typer.Option(None, "--root", help="Harness tool weights directory (SolubleMPNN, ESM2); defaults to the configured or prepared location."),
    opendde_root_dir: Optional[Path] = typer.Option(None, "--opendde-root", help="OpenDDE data directory (checkpoint/, common/) for local folding; defaults to OPENDDE_ROOT_DIR or ~/.cache/opendde."),
    mode: Optional[str] = typer.Option(None, "--mode", help="OpenDDE mode: api or local. Local mode also prepares OpenDDE weights."),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="OpenDDE checkpoint filename; used in local mode only."),
    download_workers: int = typer.Option(2, "--download-workers", min=1, max=4),
    assets_only: bool = typer.Option(False, "--assets-only", help="Prepare weights only; skip runtime code."),
    code_only: bool = typer.Option(False, "--code-only", help="Prepare runtime code only; skip weights."),
    code_cache: Optional[Path] = typer.Option(None, "--code-cache", help="Versioned runtime code cache."),
    upstream_dir: Optional[Path] = typer.Option(None, "--upstream-dir", help="Reuse verified upstream Git checkouts for runtime code."),
    sources_only: Optional[Path] = typer.Option(
        None, "--sources-only", metavar="HARNESS_ROOT",
        help="Clone pinned upstream checkouts under a Harness source checkout; prepare nothing else.",
    ),
) -> None:
    """Download model weights and prepare runtime code for the configured compute setup."""
    from opendde_harness.cli import compute_assets
    from opendde_harness.cli.compute_code import prepare_runtime_code
    from opendde_harness.cli.onboard_compute import load_protein_design_config
    from opendde_harness.plugin.protein_design.core.asset_paths import DEFAULT_CHECKPOINT

    if assets_only + code_only + (sources_only is not None) > 1:
        raise typer.BadParameter("--assets-only, --code-only and --sources-only are mutually exclusive.")
    if mode not in {None, "api", "local"}:
        raise typer.BadParameter("Mode must be api or local.")
    config = load_protein_design_config()
    saved = config.get("compute_docker") or {}
    mode = mode or (config.get("fold_defaults") or {}).get("execution_mode") or "api"
    if mode not in {"api", "local"}:
        raise typer.BadParameter("Configured folding mode must be api or local.")
    if checkpoint and mode == "api":
        typer.echo("Warning: --checkpoint is ignored in api mode; OpenDDE weights are prepared for local folding only.", err=True)
    checkpoint = checkpoint or Path(saved.get("opendde_checkpoint") or DEFAULT_CHECKPOINT).name
    if mode == "local" and checkpoint not in compute_assets.CHECKPOINTS:
        raise typer.BadParameter("Unknown OpenDDE checkpoint filename; specify a published checkpoint with --checkpoint.")
    try:
        if sources_only is not None:
            compute_assets.prepare_sources(sources_only)
            return
        if not code_only:
            compute_assets.prepare(
                root or compute_assets.weights_root(saved),
                checkpoint,
                compute_assets.asset_state_path(),
                opendde_root=opendde_root_dir or compute_assets.opendde_root(saved),
                download_workers=download_workers,
                with_opendde=mode == "local",
            )
        if not assets_only:
            prepare_runtime_code(code_cache, upstream_dir=upstream_dir)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        typer.echo(f"Compute preparation failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@compute_app.command("serve")
def serve(
    device: str = typer.Option("auto", "--device", help="auto, cpu, or cuda"),
) -> None:
    """Run the compute API on this Linux host."""
    from opendde_harness.cli.compute_environment import check_local_platform, resolve_device
    from opendde_harness.plugin.protein_design.servers.api import run as serve_api

    try:
        check_local_platform()
        os.environ["OPENDDE_HARNESS_COMPUTE_DEVICE"] = resolve_device(device)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    serve_api()


@compute_app.command("stop")
def stop(
    force: bool = typer.Option(False, "--force", help="Stop even while jobs are running or queued."),
) -> None:
    """Stop the on-demand local compute container; refuses while jobs run unless --force."""
    import httpx

    from opendde_harness.cli.onboard_compute import ComputeSetupError, load_protein_design_config
    from opendde_harness.plugin.protein_design.servers import local_service

    state = local_service.running_instance()
    if state is None:
        typer.echo("The local compute service is not running.")
        return
    name = str(state["container"])
    token = str(load_protein_design_config().get("compute_token") or "") or None
    try:
        response = local_service.request_shutdown(str(state["url"]), token, if_idle=not force)
    except httpx.HTTPError as exc:
        typer.echo(f"Compute service {name} did not answer: {exc}", err=True)
        raise typer.Exit(1) from exc
    if response.status_code == 409:
        try:
            detail = response.json()
        except ValueError:
            detail = {}
        typer.echo(
            f"Compute service {name} is busy: {detail.get('running', '?')} running, {detail.get('queued', '?')} queued, "
            f"{detail.get('tasks', 0)} task leases. Use --force to stop anyway.",
            err=True,
        )
        raise typer.Exit(1)
    if response.status_code != 202:
        typer.echo(f"Compute service {name} refused to stop (HTTP {response.status_code}).", err=True)
        raise typer.Exit(1)
    local_service.clear_state()
    try:
        stopped = local_service.wait_until_stopped(name)
    except ComputeSetupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Compute container {name} stopped." if stopped else f"Compute container {name} is still stopping.")


__all__ = ["compute_app"]
