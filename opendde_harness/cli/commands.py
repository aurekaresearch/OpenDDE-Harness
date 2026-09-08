"""OpenDDE Harness CLI entry-point.

This module wires together every top-level command and every subcommand
group. The actual implementations live in per-feature modules:

- Top-level commands (each exposes a ``register(app)`` function):
    - ``doctor``   → ``opendde_harness/cli/doctor_commands.py``
    - ``onboard``  → ``opendde_harness/cli/onboard_commands.py``
    - ``plugins``  → ``opendde_harness/cli/plugin_commands.py``
    - ``status``   → ``opendde_harness/cli/status_commands.py``
    - ``tracing``  → ``opendde_harness/cli/tracing_commands.py``
    - ``upgrade``  → ``opendde_harness/cli/upgrade_commands.py``
  ``compare`` is defined in this module.

- Subcommand groups (each exposes a typer ``*_app`` instance):
    - ``compute``        → ``opendde_harness/cli/compute_commands.py``
    - ``provider``       → ``opendde_harness/cli/provider_commands.py``
    - ``protein-design`` → ``opendde_harness/cli/protein_design_commands.py``
    - ``sessions``       → ``opendde_harness/cli/session_commands.py``
    - ``skill``          → ``opendde_harness/cli/skill_commands.py``
    - ``tui``            → ``opendde_harness/cli/tui_commands.py``

Shared helpers used across multiple command modules live in
``opendde_harness/cli/_helpers.py``.
"""

import os
import sys
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import typer
from rich.console import Console

from opendde_harness import __logo__, __version__

app = typer.Typer(
    name="ddeharness",
    help=f"{__logo__} OpenDDE Harness - Agent Framework",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} OpenDDE Harness v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True),
):
    """OpenDDE Harness - antibody design harness.

    Bare ``ddeharness`` (no subcommand) is ``ddeharness tui``: it routes
    through the same ``tui`` callback and enters the native TUI. On a first
    run without a usable provider that callback runs the terminal onboarding
    wizard before launching; ``ddeharness onboard`` reconfigures later.
    """
    if ctx.invoked_subcommand is not None:
        return
    from opendde_harness.cli.tui_commands import tui as _tui_entry

    # Delegate to the exact `ddeharness tui` callback so launch behavior is
    # identical for both entry points. Pass explicit plain defaults (the
    # function's typer.Option defaults are OptionInfo sentinels, only
    # resolved when typer drives the command).
    _tui_entry(
        ctx,
        check=False,
        dev=False,
        color=None,
        print_colors=False,
        preview_colors=False,
    )


# ============================================================================
# Top-level command registrations
# ============================================================================

from opendde_harness.cli import (
    doctor_commands,
    onboard_commands,
    plugin_commands,
    status_commands,
    tracing_commands,
    upgrade_commands,
)

onboard_commands.register(app)
status_commands.register(app)
doctor_commands.register(app)
plugin_commands.register(app)
tracing_commands.register(app)
upgrade_commands.register(app)


# ============================================================================
# Subcommand registrations
# ============================================================================

from opendde_harness.cli.compute_commands import compute_app
from opendde_harness.cli.protein_design_commands import protein_design_app
from opendde_harness.cli.provider_commands import provider_app
from opendde_harness.cli.skill_commands import skill_app

app.add_typer(compute_app, name="compute")
app.add_typer(provider_app, name="provider")
app.add_typer(protein_design_app, name="protein-design")
app.add_typer(skill_app, name="skill")


from opendde_harness.cli.tui_commands import tui_app

app.add_typer(tui_app, name="tui")

from opendde_harness.cli.session_commands import session_app

app.add_typer(session_app, name="sessions")

@app.command("compare")
def compare(
    legacy: Path = typer.Argument(..., exists=True, dir_okay=False),
    current: Path = typer.Argument(..., exists=True, dir_okay=False),
    top_k: int = typer.Option(20, "--top-k", min=1),
    maximize: bool = typer.Option(False, "--maximize"),
) -> None:
    """Compare two protein-design candidate populations."""
    from opendde_harness.plugin.protein_design.core.validation import compare_runs, load_candidates

    report = compare_runs(load_candidates(legacy), load_candidates(current), top_k=top_k, minimize=not maximize)
    typer.echo(report.model_dump_json(indent=2))


def run() -> None:
    """Console-script entry point."""
    from opendde_harness.config.loader import ConfigReadError, ConfigSchemaError
    from opendde_harness.providers.auth import MissingCredentialsError

    try:
        app()
    except MissingCredentialsError as exc:
        # The gate is decided in `providers.auth` because three entry points ask
        # it; printing and exiting is this one's idiom, so it happens here rather
        # than there. Rendered once for every command, like ConfigReadError.
        from opendde_harness.cli._helpers import console

        console.print(f"[red]Error: {exc.summary}.[/red]")
        if exc.remedy:
            console.print(exc.remedy)
        raise SystemExit(1) from exc
    except ConfigReadError as exc:
        # A config-write command (provider/onboard) hit an
        # unparseable config. The write layer already refused (file untouched);
        # surface it cleanly here, once, for every command instead of a traceback.
        from rich.console import Console

        Console(stderr=True).print(f"[red]✗[/red] {exc}")
        raise SystemExit(1) from exc
    except ConfigSchemaError as exc:
        # A key this release does not know. Nothing rewrites the file; the
        # message names the key and the remedy (`ddeharness onboard`).
        from rich.console import Console

        Console(stderr=True).print(f"[red]✗[/red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
