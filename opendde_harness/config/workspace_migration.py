"""Explicit offline relocation of legacy Harness-owned workspace files.

A durable instance-wide marker fences clients until every file is verified.
The original byte copies remain in the private archive, including unclassified
memory. No memory backend or model is called by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from opendde_harness.config.paths import WorkspaceStorage, get_workspace_storage
from opendde_harness.memory_engine.consolidate.consolidator import _atomic_write_text
from opendde_harness.utils.portable_lock import file_lock


class MigrationError(RuntimeError):
    """Migration cannot safely proceed; source data is retained."""


@dataclass(frozen=True)
class MigrationPlan:
    workspace: Path
    storage: WorkspaceStorage
    items: tuple[dict, ...]


@dataclass(frozen=True)
class MigrationResult:
    manifest: Path
    requires_review: tuple[str, ...]


def _archive(storage: WorkspaceStorage) -> Path:
    return storage.root / "migrations" / storage.scope


def _no_symlink(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise MigrationError(f"symlink is not a safe migration path: {part}")


def _fingerprint(path: Path) -> dict:
    _no_symlink(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise MigrationError(f"not a regular file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size": info.st_size, "mode": stat.S_IMODE(info.st_mode)}


def _matches(path: Path, expected: dict) -> bool:
    return path.exists() and _fingerprint(path) == {k: expected[k] for k in ("sha256", "size", "mode")}


def _files(path: Path) -> list[Path]:
    _no_symlink(path)
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    result = []
    for item in sorted(path.rglob("*")):
        _no_symlink(item)
        if not item.is_dir():
            _fingerprint(item)
            result.append(item)
    return result


def _mappings(workspace: Path, storage: WorkspaceStorage):
    yield workspace / "sessions", storage.sessions, False
    yield workspace / "exports", storage.exports, False
    yield workspace / "skills", storage.skills, False
    yield workspace / "TOOLS.md", storage.assistant / "TOOLS.md", False
    yield workspace / "SOUL.md", storage.assistant / "soul.md", False
    yield workspace / "agent_memory/profile/soul.md", storage.assistant / "soul.md", False
    yield workspace / "agent_memory/profile/agent.md", storage.assistant / "agent.md", False
    yield workspace / ".opendde_harness/shadow.git", storage.checkpoint / "shadow.git", False
    # Operational state travels together, never through the local-profile tree.
    yield workspace / "user_memory/outbox.jsonl", storage.memory_state / "outbox.jsonl", False
    yield workspace / "user_memory/state.json", storage.memory_state / "state.json", False
    yield workspace / "user_memory/profile/user.md.lock", storage.memory_state / "store.lock", False
    # Unknown profile/procedural content is retained for human review, not
    # promoted to assistant instructions or silently fed to an external service.
    for relative in ("USER.md", "memory", "user_memory", "agent_memory"):
        yield workspace / relative, None, True


def preview_migration(workspace: Path, storage: WorkspaceStorage) -> MigrationPlan:
    workspace = workspace.expanduser().resolve()
    if storage.root == workspace or storage.root.is_relative_to(workspace):
        raise MigrationError("migration data root must be outside the workspace")
    _no_symlink(storage.root)
    items, seen, destinations = [], set(), {}
    for source_root, destination_root, review in _mappings(workspace, storage):
        for source in _files(source_root):
            if source in seen:
                continue
            seen.add(source)
            destination = None
            if destination_root is not None:
                destination = (
                    destination_root / source.relative_to(source_root) if source_root.is_dir() else destination_root
                )
                _no_symlink(destination)
            fingerprint = _fingerprint(source)
            status = "archive" if review else "copy"
            if destination is not None:
                previous = destinations.get(destination)
                if previous is not None and previous != fingerprint:
                    status = "conflict"
                elif destination.exists():
                    status = "identical" if _matches(destination, fingerprint) else "conflict"
                destinations[destination] = fingerprint
            items.append(
                {
                    "source": str(source),
                    "destination": str(destination) if destination else None,
                    "backup": str(_archive(storage) / "originals" / source.relative_to(workspace)),
                    "requires_review": review,
                    "status": status,
                    **fingerprint,
                }
            )
    return MigrationPlan(workspace, storage, tuple(items))


def assert_offline() -> None:
    """Conservatively reject other Python/Harness processes of this user.

    Older clients have no migration lease; a new lock alone cannot exclude
    them. Unknown platform/process visibility is therefore a refusal, not an
    assertion that nothing is running. Command contents are never reported.
    """
    if sys.platform != "linux" or not Path("/proc").is_dir():
        raise MigrationError("cannot verify offline process state on this platform")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            args = (process / "cmdline").read_bytes().split(b"\0")
            if not args or not args[0]:
                continue
            executable = Path(os.fsdecode(args[0])).name.lower()
            harness = any(b"opendde_harness" in arg or b"ddeharness" in arg for arg in args[:3])
            if executable.startswith(("python", "ddeharness", "opendde")) or harness:
                raise MigrationError(f"active writer/runtime PID {process.name}; stop it before migration")
        except FileNotFoundError:
            continue  # exited while enumerating
        except PermissionError as exc:
            raise MigrationError(f"cannot verify process PID {process.name}") from exc


def copy_verified(source: Path, target: Path) -> None:
    """Non-overwriting copy with permissions and byte verification."""
    _no_symlink(source)
    _no_symlink(target)
    expected = _fingerprint(source)
    if target.exists():
        if not _matches(target, expected):
            raise MigrationError(f"destination conflict: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Copy to a sibling, then link without replacement. A power loss cannot
    # expose an incomplete destination as a completed copy on the next run.
    import tempfile

    fd, name = tempfile.mkstemp(prefix=".migration-", dir=target.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output)
            output.flush()
            os.fsync(output.fileno())
        shutil.copystat(source, staged)
        if not _matches(staged, expected) or not _matches(source, expected):
            raise MigrationError(f"source changed while copying: {source}")
        os.link(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _save(path: Path, data: dict) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2) + "\n")
    path.chmod(0o600)


def _validate_manifest(data: dict, path: Path) -> WorkspaceStorage:
    if data.get("version") != 1:
        raise MigrationError("unsupported migration manifest")
    storage = get_workspace_storage(Path(data["workspace"]), data_dir=Path(data["root"]))
    if path != _archive(storage) / "manifest.json":
        raise MigrationError("manifest path does not match its instance and workspace")
    workspace = Path(data["workspace"])
    mappings = list(_mappings(workspace, storage))
    for row in data["items"]:
        source, backup = Path(row["source"]), Path(row["backup"])
        if not source.is_relative_to(workspace) or backup != _archive(storage) / "originals" / source.relative_to(
            workspace
        ):
            raise MigrationError("unsafe manifest source/archive path")
        destination = row["destination"]
        if not any(source.is_relative_to(old) for old, _, _ in mappings):
            raise MigrationError("unsafe manifest source")
        if destination is not None:
            allowed = False
            for old, new, _ in mappings:
                if new is not None and source.is_relative_to(old):
                    allowed |= Path(destination) == new / source.relative_to(old)
            if not allowed:
                raise MigrationError("unsafe manifest destination")
        for candidate in (source, backup, *([Path(destination)] if destination else [])):
            _no_symlink(candidate)
    return storage


def apply_migration(plan: MigrationPlan) -> MigrationResult:
    storage = plan.storage
    archive = _archive(storage)
    manifest = archive / "manifest.json"
    marker = storage.root / "migration.pending"
    with file_lock(storage.root / "migration.lock", blocking=False):
        assert_offline()
        if marker.exists() and marker.read_text().strip() != str(manifest):
            raise MigrationError("another workspace migration is unfinished")
        if manifest.exists():
            data = json.loads(manifest.read_text())
            _validate_manifest(data, manifest)
            if data["status"] == "rolled_back":
                raise MigrationError("this migration was rolled back; retain its archive and resolve it explicitly")
            if data["status"] == "rolling_back":
                raise MigrationError("rollback is unfinished; resume --rollback, not --apply")
        else:
            # Never act on stale preview metadata.
            current = preview_migration(plan.workspace, storage)
            if any(row["status"] == "conflict" for row in current.items):
                raise MigrationError("destination conflict; no source files moved")
            if not current.items:
                raise MigrationError("no legacy managed files to migrate")
            _no_symlink(archive)
            if archive.exists() and any(archive.iterdir()):
                raise MigrationError("nonempty archive without a manifest requires manual review")
            archive.mkdir(parents=True, exist_ok=True, mode=0o700)
            data = {
                "version": 1,
                "workspace": str(plan.workspace),
                "root": str(storage.root),
                "status": "copying",
                "items": list(current.items),
                "preexisting_shared": [
                    str(path) for root in (storage.assistant, storage.skills) for path in _files(root)
                ],
            }
            _save(manifest, data)
        _atomic_write_text(marker, str(manifest))
        # Stop new clients first; repeat the process scan to close the startup
        # race with an older client that has no marker awareness.
        assert_offline()
        if data["status"] != "complete":
            for row in data["items"]:
                source, backup = Path(row["source"]), Path(row["backup"])
                if source.exists() and not _matches(source, row):
                    raise MigrationError(f"source changed since preview: {source}")
                if not backup.exists():
                    copy_verified(source, backup)
                if not _matches(backup, row):
                    raise MigrationError(f"archive verification failed: {backup}")
            for row in data["items"]:
                if row["destination"]:
                    copy_verified(Path(row["backup"]), Path(row["destination"]))
            # Persist verification before removing any original. A resume can
            # finish archival after an interruption without replaying queues.
            data["status"] = "verified"
            _save(manifest, data)
            for row in data["items"]:
                source = Path(row["source"])
                if source.exists():
                    if not _matches(source, row):
                        raise MigrationError(f"source changed before archival: {source}")
                    source.unlink()
            data["status"] = "complete"
            _save(manifest, data)
        marker.unlink()
    return MigrationResult(manifest, tuple(row["source"] for row in data["items"] if row["requires_review"]))


def rollback_migration(manifest: Path) -> MigrationResult:
    manifest = manifest.expanduser().absolute()
    _no_symlink(manifest)
    data = json.loads(manifest.read_text())
    storage = _validate_manifest(data, manifest)
    marker = storage.root / "migration.pending"
    with file_lock(storage.root / "migration.lock", blocking=False):
        assert_offline()
        if marker.exists() and marker.read_text().strip() != str(manifest):
            raise MigrationError("another workspace migration is unfinished")
        # Any unrecorded runtime data means this is no longer an offline,
        # unchanged destination. Never overwrite a newly-used instance.
        known = {Path(row["destination"]) for row in data["items"] if row["destination"]}
        for root in (storage.sessions, storage.exports, storage.memory_state, storage.checkpoint, storage.host_memory):
            if any(path not in known for path in _files(root)):
                raise MigrationError("new destination data prevents rollback")
        shared = known | {Path(path) for path in data.get("preexisting_shared", [])}
        for root in (storage.assistant, storage.skills):
            if any(path not in shared for path in _files(root)):
                raise MigrationError("new shared assistant/skill data prevents rollback")
        for row in data["items"]:
            for name in ("source", "destination"):
                path = Path(row[name]) if row[name] else None
                if path is not None and path.exists() and not _matches(path, row):
                    raise MigrationError(f"changed {name} prevents rollback: {path}")
            if not _matches(Path(row["backup"]), row):
                raise MigrationError("missing or changed backup prevents rollback")
        _atomic_write_text(marker, str(manifest))
        assert_offline()
        data["status"] = "rolling_back"
        _save(manifest, data)
        for row in data["items"]:
            copy_verified(Path(row["backup"]), Path(row["source"]))
        for row in data["items"]:
            if row["destination"] and row["status"] != "identical":
                Path(row["destination"]).unlink(missing_ok=True)
        data["status"] = "rolled_back"
        _save(manifest, data)
        marker.unlink()
    return MigrationResult(manifest, ())


def warn_legacy_workspace(workspace: Path) -> None:
    """Startup diagnoses old locations without modifying their contents."""
    from loguru import logger

    names = ("sessions", "user_memory", "agent_memory", "skills", "TOOLS.md", "USER.md", "SOUL.md", "memory")
    if any((workspace / name).exists() for name in names):
        logger.warning("Legacy workspace data detected; preview with `ddeharness workspace migrate`. No files moved.")
