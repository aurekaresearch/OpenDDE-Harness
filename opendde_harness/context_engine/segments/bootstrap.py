"""Segment 2 — bootstrap files (soul / agent / TOOLS). Host-owned."""

from __future__ import annotations

from pathlib import Path

from opendde_harness.context_engine.base import AssemblyContext, Segment
from opendde_harness.context_engine.segments import render


class BootstrapSegmentBuilder:
    name = "bootstrap"
    order = 2
    needs_prefix = False

    def __init__(self, workspace: Path, bootstrap_files: list[str] | None = None) -> None:
        self._workspace = workspace
        self._bootstrap_files = bootstrap_files

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        text = render.load_bootstrap_files(self._workspace, self._bootstrap_files)
        return Segment(text=text) if text else None
