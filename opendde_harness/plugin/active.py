"""Process-wide access to plugin contributions for the generic runtime.

Prompt rendering, tool descriptions and skill discovery run in code that
never sees the registry the host assembled at boot (and on CLI paths and
in tests no host boots at all). They ask here instead: the first call
assembles a registry from the standard discovery sources, honouring the
user's ``plugins.disabled`` list, and every later call reuses it.
"""

from __future__ import annotations

import logging

from opendde_harness.plugin.bootstrap import assemble_plugin_registry, discovery_sources
from opendde_harness.plugin.registry import PluginError, PluginRegistry

logger = logging.getLogger(__name__)

_registry: PluginRegistry | None = None


def _disabled_plugins() -> frozenset[str]:
    try:
        from opendde_harness.config.opendde_harness import load_opendde_harness_config

        return frozenset(load_opendde_harness_config().plugins.disabled)
    except Exception:
        return frozenset()


def active_registry() -> PluginRegistry:
    """The activated registry, assembled on first use."""
    global _registry
    if _registry is None:
        try:
            _registry = assemble_plugin_registry(**discovery_sources(), disabled=_disabled_plugins())
        except PluginError as e:
            logger.warning("plugin activation failed (%s); continuing without plugin contributions", e)
            _registry = PluginRegistry()
    return _registry


def reset_active_registry() -> None:
    """Drop the cached registry so the next call re-discovers (tests)."""
    global _registry
    _registry = None


__all__ = ["active_registry", "reset_active_registry"]
