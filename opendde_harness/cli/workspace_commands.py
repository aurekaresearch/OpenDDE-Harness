"""Explicit migration of internal data out of the execution directory."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

workspace_app = typer.Typer(help="Inspect or migrate legacy workspace data.", no_args_is_help=True)


@workspace_app.command("migrate")
def migrate(
    workspace: Path | None = typer.Option(None, help="Legacy execution directory; defaults to configured workspace."),
    config: Path | None = typer.Option(None, help="Instance config file."),
    apply: bool = typer.Option(False, "--apply", help="Apply offline migration (default: preview only)."),
    rollback: Path | None = typer.Option(None, help="Roll back this manifest, only if no new data exists."),
) -> None:
    """Preview legacy data relocation, or explicitly apply/roll back offline."""
    from opendde_harness.config.loader import load_config, set_config_path
    from opendde_harness.config.paths import get_workspace_storage
    from opendde_harness.config.workspace_migration import (
        apply_migration,
        preview_migration,
        rollback_migration,
    )

    console = Console()
    if apply and rollback:
        raise typer.BadParameter("--apply and --rollback are mutually exclusive")
    if config is not None:
        set_config_path(config.expanduser().resolve())
    try:
        if rollback is not None:
            result = rollback_migration(rollback)
            console.print(f"Restored archived originals. Manifest: {result.manifest}", markup=False)
            return
        directory = (workspace or load_config().workspace_path).expanduser().resolve()
        plan = preview_migration(directory, get_workspace_storage(directory))
        if not apply:
            table = Table("Source", "Destination / archive", "Bytes", "Status")
            for row in plan.items:
                table.add_row(row["source"], row["destination"] or row["backup"], str(row["size"]), row["status"])
            console.print(table)
            console.print("Preview only. Stop clients, workers and memory services before --apply.")
            return
        result = apply_migration(plan)
        console.print(f"Managed files archived outside workspace. Manifest: {result.manifest}", markup=False)
        if result.requires_review:
            console.print(
                f"{len(result.requires_review)} archived memory/configuration files require review; NOT imported into EverOS."
            )
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"Migration refused or interrupted: {exc}", markup=False)
        raise typer.Exit(1) from exc
