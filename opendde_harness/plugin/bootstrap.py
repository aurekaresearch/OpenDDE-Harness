"""One-call plugin bootstrap helper.

Glues :class:`PluginDiscovery` and :class:`PluginRegistry` so callers
(CLI / AgentLoop construction) don't repeat the two-step dance.

Kept as a free function rather than a class so the dataflow stays
obviously linear: discover → activate → return. Callers wanting more
control instantiate the two pieces directly.
"""

from __future__ import annotations

from pathlib import Path

from opendde_harness.plugin.discover import PluginDiscovery
from opendde_harness.plugin.registry import PluginRegistry


def discovery_sources() -> dict:
    """The four discovery-source locations the host scans.

    - bundled — ``opendde_harness/plugin/`` inside the package.
    - user    — ``~/.opendde_harness/plugins/``.
    - project — ``./.opendde_harness/plugins/``.
    - entry_points — the ``opendde_harness.plugins`` group.
    """
    import opendde_harness

    return {
        "bundled_dir": Path(opendde_harness.__path__[0]) / "plugin",
        "user_dir": Path.home() / ".opendde_harness" / "plugins",
        "project_dir": Path.cwd() / ".opendde_harness" / "plugins",
        "entry_points_group": "opendde_harness.plugins",
    }


def assemble_plugin_registry(
    *,
    bundled_dir: Path | None = None,
    user_dir: Path | None = None,
    project_dir: Path | None = None,
    entry_points_group: str | None = "opendde_harness.plugins",
    disabled: frozenset[str] = frozenset(),
) -> PluginRegistry:
    """Discover all manifests, admit the enabled ones, return the registry.

    ``entry_points_group`` defaults to ``"opendde_harness.plugins"`` because
    that is the public group third-party plugins target in their
    ``pyproject.toml``. Pass ``None`` to suppress entry-point discovery
    entirely (tests do this to stay hermetic).
    """
    discovery = PluginDiscovery(
        bundled_dir=bundled_dir,
        user_dir=user_dir,
        project_dir=project_dir,
        entry_points_group=entry_points_group,
    )
    registry = PluginRegistry()
    registry.activate(discovery.discover(), disabled=disabled)
    return registry


__all__ = ["assemble_plugin_registry", "discovery_sources"]
