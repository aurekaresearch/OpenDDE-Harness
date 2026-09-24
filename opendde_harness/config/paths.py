"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from opendde_harness.config.loader import get_config_path
from opendde_harness.utils.helpers import ensure_dir


def get_data_dir() -> Path:
    """Return the instance-level runtime data directory."""
    return ensure_dir(get_config_path().parent)


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory under the instance data dir."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory, optionally namespaced per channel."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cache_dir() -> Path:
    """Return the disposable, refetchable on-disk cache directory."""
    return get_runtime_subdir("cache")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path."""
    path = Path(workspace).expanduser() if workspace else Path.home() / ".opendde_harness" / "workspace"
    return ensure_dir(path)


@dataclass(frozen=True)
class WorkspaceStorage:
    """Instance-owned persistence, separate from the execution directory."""

    root: Path
    scope: str
    assistant: Path
    skills: Path
    sessions: Path
    exports: Path
    memory_state: Path
    checkpoint: Path
    host_memory: Path


def get_workspace_scope(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.expanduser().resolve()).encode()).hexdigest()


def assert_storage_ready(storage: WorkspaceStorage) -> None:
    """Never let a new reader or writer see a partially migrated instance."""
    if (storage.root / "migration.pending").exists():
        raise RuntimeError("workspace migration is unfinished; resume or roll back before starting Harness")


def get_workspace_storage(workspace: Path, *, data_dir: Path | None = None) -> WorkspaceStorage:
    """Resolve paths without creating files (including during migration preview)."""
    root = (data_dir if data_dir is not None else get_config_path().parent).expanduser().resolve()
    scope = get_workspace_scope(workspace)
    state = root / "state" / scope
    return WorkspaceStorage(
        root,
        scope,
        root / "assistant",
        root / "skills",
        root / "sessions" / scope,
        root / "exports" / scope,
        state / "memory",
        state / "checkpoint",
        root / "memory" / "host" / scope,
    )


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return Path.home() / ".opendde_harness" / "history" / "cli_history"
