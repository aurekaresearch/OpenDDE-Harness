"""Long-term memory backend — bundled default plugin.

Implements the host's :class:`opendde_harness.memory_engine.MemoryBackend`
Protocol over a third-party memory library (served out of process over HTTP).
Discovered via ``opendde-harness-plugin.toml`` (bundled source);
``backend.make_backend`` is the factory the registry calls.

Every import of the library, and every literal it dictates, lives in
:mod:`opendde_harness.plugin.memory.longterm._library`; the rest of the tree
names the feature by its role only.

This module is kept import-cheap on purpose: PluginDiscovery touches it
during resource resolution, so it must NOT import ``backend`` (which
lazily pulls the heavy library). Import the backend explicitly from
:mod:`opendde_harness.plugin.memory.longterm.backend`.
"""

__version__ = "1.1.0"
