"""Custom Hatchling build hook: conditionally package the prebuilt TUI bundle.

The TUI ships as a single self-contained esbuild bundle at ``ui-tui/dist/entry.js``.
We want a wheel to carry it so `pip`/`uv tool install` yields a working
`ddeharness tui` with no source checkout. But ``dist/`` is a build artifact and is
NOT committed (see .gitignore), so a clean checkout legitimately lacks it.

A static ``[tool.hatch.build.targets.wheel.force-include]`` entry would make
hatchling hard-fail with "Forced include not found" whenever ``dist/`` is
absent — which would break the ordinary developer flow (`git clone && uv sync`
with no prior `npm run build`). So instead we add the bundle to the wheel's
force-include map *only when it exists*, and emit a warning otherwise.

Release builds run ``npm ci && npm run build`` first (see
.github/workflows/release.yml), so the published wheel always carries the
bundle; dev builds without it simply fall back to the source tree at runtime
(see resolve_dist_entry() in opendde_harness/cli/tui_commands.py).
"""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        ui = Path(self.root) / "ui-tui"
        dist = ui / "dist"
        _refresh_bundle(ui, dist / "entry.js", self.app)
        if dist.is_dir() and (dist / "entry.js").is_file():
            # Map source -> path inside the wheel's `opendde` package.
            build_data.setdefault("force_include", {})[str(dist)] = "opendde_harness/ui-tui/dist"
        else:
            self.app.display_warning(
                "ui-tui/dist/entry.js not found — building WITHOUT the bundled TUI. "
                "`ddeharness tui` from this wheel will not work; run "
                "`npm --prefix ui-tui ci && npm --prefix ui-tui run build` before "
                "building a release wheel."
            )


def _refresh_bundle(ui: Path, entry: Path, app) -> None:
    """Rebuild the TUI bundle from a source tree when it is missing or stale.

    A wheel built from a checkout must not ship an entry.js older than the
    sources next to it; release builds run npm beforehand and are a no-op here.
    """
    import shutil
    import subprocess

    src = ui / "src"
    if not src.is_dir():
        return
    sources = [*src.rglob("*"), ui / "package.json", ui / "package-lock.json"]
    newest = max((f.stat().st_mtime for f in sources if f.is_file()), default=0.0)
    if entry.is_file() and entry.stat().st_mtime >= newest:
        return
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "ui-tui/dist/entry.js is missing or older than ui-tui/src and npm is not available; "
            "run `npm --prefix ui-tui ci && npm --prefix ui-tui run build` before building the wheel."
        )
    app.display_info("ui-tui/dist/entry.js is stale or missing; running `npm run build` in ui-tui/")
    if not (ui / "node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd=str(ui), check=True)
    subprocess.run([npm, "run", "build"], cwd=str(ui), check=True)
