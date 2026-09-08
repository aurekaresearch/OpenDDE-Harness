"""Configuration module for OpenDDE Harness.

One root model, :class:`Config`, parsed once from ``config.json`` by
:func:`load_config`. The feature blocks (context / SkillForge / runtime /
tracing / plugins / memory) are defined in
:mod:`opendde_harness.config.features` and composed into the root.
"""

from opendde_harness.config.features import ContextConfig, SkillForgeConfig
from opendde_harness.config.loader import get_config_path, load_config
from opendde_harness.config.paths import (
    get_cli_history_path,
    get_data_dir,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
)
from opendde_harness.config.schema import Config

OpenDDEHarnessConfig = Config
load_opendde_harness_config = load_config

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_runtime_subdir",
    "get_media_dir",
    "get_logs_dir",
    "get_workspace_path",
    "get_cli_history_path",
    "OpenDDEHarnessConfig",
    "load_opendde_harness_config",
    "ContextConfig",
    "SkillForgeConfig",
]
