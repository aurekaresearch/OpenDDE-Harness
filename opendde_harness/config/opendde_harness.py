"""Compatibility names for the single config root.

There is one root model, :class:`opendde_harness.config.schema.Config`, and one
loader, :func:`opendde_harness.config.loader.load_config`; the feature blocks
live in :mod:`opendde_harness.config.features`. The names below are kept so
existing imports from this module keep working.
"""

from opendde_harness.config.features import (
    CheckpointConfig,
    ContextConfig,
    LocalDirConfig,
    MemoryConfig,
    MemoryExtractionConfig,
    PluginsConfig,
    RuntimeConfig,
    SkillForgeConfig,
    SkillForgeRouterConfig,
    TracingConfig,
)
from opendde_harness.config.loader import load_config as load_opendde_harness_config
from opendde_harness.config.schema import Config as OpenDDEHarnessConfig

__all__ = [
    "CheckpointConfig",
    "ContextConfig",
    "MemoryExtractionConfig",
    "LocalDirConfig",
    "MemoryConfig",
    "OpenDDEHarnessConfig",
    "PluginsConfig",
    "RuntimeConfig",
    "SkillForgeConfig",
    "SkillForgeRouterConfig",
    "TracingConfig",
    "load_opendde_harness_config",
]
