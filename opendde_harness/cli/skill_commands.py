"""Skill subcommands — owns the ``skill_app`` Typer instance.

Bundles all ``opendde skill ...`` subcommands:

Read-only inspection (registry-level):

- ``skill list``                — list skills visible to SkillForge
- ``skill get <name>``          — show one skill's metadata (and optionally body)

Lifecycle management:

- ``skill block <name>``        — add to skillForge.blocklist (refused everywhere)
- ``skill unblock <name>``      — remove from skillForge.blocklist

``commands.py`` imports :data:`skill_app` and registers it on the top-level
``app`` via ``app.add_typer(skill_app, name="skill")``.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

console = Console()


skill_app = typer.Typer(help="Inspect and manage SkillForge skills")


def _build_skill_service():
    from opendde_harness.config.loader import load_config
    from opendde_harness.memory_engine.skill_forge import LocalSkillCatalog

    config = load_config()
    workspace = config.workspace_path
    sf_cfg = _load_skill_forge_config() or getattr(config, "skill_forge", None)
    return LocalSkillCatalog(workspace, config=sf_cfg, start_watcher=False)


def _load_skill_forge_config():
    """The configured SkillForgeConfig, or ``None`` when the config cannot be read."""
    try:
        from opendde_harness.config.loader import load_config

        return load_config().skill_forge
    except Exception:
        return None


@skill_app.command("list")
def skill_list(
    source: str | None = typer.Option(
        None, "--source", "-s", help="Filter by source (workspace/builtin/memory/mirror/*)"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows shown"),
):
    """List skills visible to SkillForge."""
    svc = _build_skill_service()
    metas = svc.gather_all_skills()
    if source:
        metas = [m for m in metas if m.source == source]
    metas = metas[:limit]

    if not metas:
        console.print("[dim]No skills found.[/dim]")
        return

    from opendde_harness.memory_engine.skill_forge.catalog import is_blocked, normalize_blocklist

    blocked = normalize_blocklist(getattr(_load_skill_forge_config(), "blocklist", None))
    table = Table(title=f"Skills ({len(metas)})")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Description", overflow="fold")
    for m in metas:
        desc = (m.description or "")[:120]
        name_cell = Text(m.name)
        if is_blocked(blocked, m.name):
            name_cell.append(" [blocked]", style="bold red")
        table.add_row(name_cell, m.source, desc)
    console.print(table)


@skill_app.command("get")
def skill_get(
    name: str = typer.Argument(..., help="Skill name"),
    with_body: bool = typer.Option(False, "--with-body/--no-body", help="Include SKILL.md content"),
):
    """Show one skill's metadata (and optionally its body)."""
    svc = _build_skill_service()
    meta = svc.get_skill_metadata(name)
    if meta is None:
        console.print(f"[red]Skill not found: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]{name}[/bold cyan]")
    for k, v in meta.items():
        console.print(f"  [dim]{k}[/dim]: {v}")

    if with_body:
        body = svc.load_skill(name)
        if body:
            console.print("\n[bold]── SKILL.md ──[/bold]")
            console.print(Markdown(body))


@skill_app.command("block")
def skill_block(name: str = typer.Argument(..., help="Skill name / slug to refuse everywhere")):
    """Add a skill to skillForge.blocklist (dropped from the injection pool
    and refused by use_skill on the next agent start)."""
    from opendde_harness.config.update import set_skill_blocked

    blocklist = set_skill_blocked(name, True)
    console.print(f"[green]Blocked[/green] {name!r}. skillForge.blocklist = {blocklist}")
    console.print("[dim]Takes effect on the next agent/gateway start.[/dim]")


@skill_app.command("unblock")
def skill_unblock(name: str = typer.Argument(..., help="Skill name / slug to allow again")):
    """Remove a skill from skillForge.blocklist."""
    from opendde_harness.config.update import set_skill_blocked

    blocklist = set_skill_blocked(name, False)
    console.print(f"[green]Unblocked[/green] {name!r}. skillForge.blocklist = {blocklist}")
    console.print("[dim]Takes effect on the next agent/gateway start.[/dim]")


__all__ = ["skill_app"]
