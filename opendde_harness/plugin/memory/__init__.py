"""Bundled memory-backend plugins.

Subpackages here are discovered by :class:`opendde_harness.plugin.PluginDiscovery`
via their ``opendde-harness-plugin.toml`` manifests (the bundled source). Keep
these packages' ``__init__`` modules empty/cheap: resource resolution
imports them during discovery, and a heavy import here would defeat the
manifest-only discovery guarantee.
"""
