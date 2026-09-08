"""Plugin foundation.

PG-1 introduces the manifest schema + plugin context. Registry and
discovery follow in PG-2/PG-3. The first (and currently only) public
contribution point is ``memory_backends``; the schema is forward-
compatible so future contribution types land without breaking existing
manifests.

Two principles, both load-bearing:

1. **Manifests are pure data.** ``PluginManifest.from_toml_path`` only
   reads the TOML; no plugin code is imported until the registry asks
   the factory to build a backend. This keeps startup deterministic and
   audit-friendly.

2. **Factories are referenced by ``module.path:callable`` strings.**
   The registry imports the module and resolves the callable lazily —
   manifest parsing never triggers import-time side effects in the
   plugin's package.
"""

from __future__ import annotations

from opendde_harness.plugin.active import active_registry
from opendde_harness.plugin.bootstrap import assemble_plugin_registry, discovery_sources
from opendde_harness.plugin.context import PluginContext, ServiceLocator
from opendde_harness.plugin.discover import DiscoveredPlugin, PluginDiscovery, Source
from opendde_harness.plugin.manifest import (
    Contributes,
    MemoryBackendContribution,
    PluginManifest,
    PromptSegmentContribution,
    ToolContribution,
)
from opendde_harness.plugin.registry import (
    MemoryBackendFactory,
    PluginConflictError,
    PluginError,
    PluginFactoryImportError,
    PluginNotFoundError,
    PluginRegistry,
    ToolFactory,
)

__all__ = [
    "Contributes",
    "DiscoveredPlugin",
    "active_registry",
    "assemble_plugin_registry",
    "discovery_sources",
    "MemoryBackendContribution",
    "MemoryBackendFactory",
    "PluginConflictError",
    "PluginContext",
    "PluginDiscovery",
    "PluginError",
    "PluginFactoryImportError",
    "PluginManifest",
    "PluginNotFoundError",
    "PluginRegistry",
    "PromptSegmentContribution",
    "ServiceLocator",
    "Source",
    "ToolContribution",
    "ToolFactory",
]
