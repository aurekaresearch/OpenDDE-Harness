"""Prepare immutable mounted tool code from the installed Harness distribution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from rich.console import Console

from opendde_harness.cli._download import DownloadError, download_file, progress
from opendde_harness.cli.compute_assets import shared_asset_plan, source_revisions
from opendde_harness.cli.compute_environment import SUBPROCESS_TIMEOUT, environment_digest, load_environment
from opendde_harness.plugin.protein_design.core.asset_paths import harness_weights_path

MANIFEST = "runtime-code.json"
EXTERNAL_MARKER = '"""Pinned upstream tool sources prepared by ddeharness compute prepare."""\n'
REPOSITORIES = {
    "opendde": "aurekaresearch/OpenDDE",
    "ligandmpnn": "dauparas/LigandMPNN",
    "plip": "pharmai/plip",
}
IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv", "node_modules", "outputs", "model_params", "checkpoints"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def default_code_cache() -> Path:
    override = os.environ.get("OPENDDE_HARNESS_CODE_CACHE")
    return Path(override) if override else harness_weights_path() / "runtime-code"


def _files(root: Path) -> dict[str, str]:
    result = {}
    for directory, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in IGNORED_DIRECTORIES)
        for name in directories + sorted(names):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Runtime code cannot contain symlinks: {path}")
            if not path.is_file() or path.suffix == ".pyc" or path == root / MANIFEST:
                continue
            if path.suffix in WEIGHT_SUFFIXES:
                raise ValueError(f"Model weights must be stored outside runtime code: {path}")
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_code_identity(package_dir: Path | None = None) -> dict[str, Any]:
    package_dir = package_dir or Path(__file__).resolve().parents[1]
    try:
        version = importlib.metadata.version("opendde-harness")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0+source"
    environment = load_environment()
    revisions = source_revisions()
    identity = {
        "schema_version": 1,
        "harness_version": version,
        "harness_sha256": _digest(_files(package_dir)),
        "sources": {name: revisions[name.upper() + "_REV"] for name in REPOSITORIES},
        "environment_id": environment["id"],
        "environment_sha256": environment_digest(environment),
    }
    identity["id"] = _digest(identity)
    return identity


def runtime_code_dir(cache_root: Path | None = None) -> Path:
    """Where this release's managed runtime code lives, prepared or not.

    Managed code is not written into the config: its directory is derived from
    the release and the environment, so a caller that only wants to look (the
    doctor) must derive it the same way ``prepare_runtime_code`` does rather
    than read a ``package_root`` that managed deployments leave empty.
    """
    return _code_destination((cache_root or default_code_cache()).expanduser().resolve(), runtime_code_identity())


def _code_destination(cache_root: Path, identity: dict[str, Any]) -> Path:
    version = re.sub(r"[^A-Za-z0-9._-]", "_", identity["harness_version"])
    return cache_root / f"{version}-{identity['id'][:20]}"


def verify_runtime_code(root: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError(f"Runtime code directory must not be a symlink: {root}")
    if not (root / MANIFEST).is_file():
        raise ValueError(f"Runtime code is not prepared at {root or Path.cwd()}; run ddeharness compute prepare.")
    manifest = json.loads((root / MANIFEST).read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("identity"), dict):
        raise ValueError(f"Invalid runtime code manifest: {root}")
    identity = manifest.get("identity", {})
    payload = {key: value for key, value in identity.items() if key != "id"}
    if identity.get("schema_version") != 1 or identity.get("id") != _digest(payload):
        raise ValueError(f"Invalid runtime code identity: {root}")
    if expected is not None and identity != expected:
        raise ValueError(f"Runtime code identity differs: {root}; existing code was not overwritten")
    if manifest.get("files") != _files(root):
        raise ValueError(f"Runtime code changed or is incomplete: {root}; existing code was not overwritten")
    return identity


def _extract_source(
    stream: BinaryIO, destination: Path, *, strip_root: bool, advance: Callable[[int], None] | None = None
) -> None:
    destination.mkdir()
    total = 0
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name:
                raise ValueError(f"Unsafe source archive path: {member.name}")
            parts = path.parts[1:] if strip_root else path.parts
            if not parts or any(part in IGNORED_DIRECTORIES for part in parts) or path.suffix in WEIGHT_SUFFIXES:
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Source archive must contain regular files only: {member.name}")
            total += member.size
            if total > 512 * 1024 * 1024:
                raise ValueError("Source archive exceeds the runtime code size limit")
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable source archive member: {member.name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
            if advance is not None:
                advance(member.size)


def _download_source(name: str, revision: str, archive: Path, label: str, console: Console, position: str) -> None:
    url = f"https://codeload.github.com/{REPOSITORIES[name]}/tar.gz/{revision}"
    print(f"Preparing tool sources{f' ({position})' if position else ''}: {label} from {url}", flush=True)
    partial = archive.parent / f".{archive.name[: -len('.tar.gz')]}.part"
    try:
        download_file([url], partial, description=label, console=console, max_bytes=128 * 1024 * 1024, deadline_s=900)
    except DownloadError as exc:
        raise ValueError(
            f"Source download stalled or failed: {url} ({exc}). Set https_proxy for GitHub access, "
            "or pass --upstream-dir with local checkouts at the pinned revisions."
        ) from exc
    partial.replace(archive)


def _extract_archive(archive: Path, destination: Path, label: str, console: Console) -> None:
    with archive.open("rb") as stream, progress(console) as bar:
        task = bar.add_task(f"{label} extracting", total=None)
        _extract_source(stream, destination, strip_root=True, advance=lambda size: bar.update(task, advance=size))


def _prepare_upstream(
    name: str,
    revision: str,
    destination: Path,
    upstream_dir: Path | None,
    *,
    position: str = "",
    sources_dir: Path | None = None,
) -> None:
    if upstream_dir is not None:
        checkout = upstream_dir / name
        executable = shutil.which("git")
        if not executable:
            raise ValueError("Git is required only when reusing local upstream checkouts with --upstream-dir")
        git = [executable, "-C", str(checkout)]
        actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, timeout=SUBPROCESS_TIMEOUT).strip()
        dirty = subprocess.check_output(
            [*git, "status", "--porcelain", "--untracked-files=no"], text=True, timeout=SUBPROCESS_TIMEOUT
        ).strip()
        if actual != revision or dirty:
            raise ValueError(f"Upstream checkout differs from the pinned revision: {checkout}")
        with tempfile.TemporaryFile() as stream:
            subprocess.run(
                [*git, "archive", "--format=tar", revision], stdout=stream, check=True, timeout=SUBPROCESS_TIMEOUT
            )
            stream.seek(0)
            _extract_source(stream, destination, strip_root=False)
        return
    label = f"{name} ({revision[:12]})"
    console = Console()
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        sources_dir = sources_dir or Path(temporary)
        sources_dir.mkdir(parents=True, exist_ok=True)
        archive = sources_dir / f"{name}-{revision}.tar.gz"
        cached = archive.is_file() and archive.stat().st_size > 0
        if cached:
            print(f"Reusing cached source: {label}", flush=True)
        else:
            _download_source(name, revision, archive, label, console, position)
        try:
            _extract_archive(archive, destination, label, console)
        except (tarfile.TarError, EOFError):
            if not cached:
                raise
            # A cached archive that no longer extracts is corrupt; fetch it once more.
            shutil.rmtree(destination, ignore_errors=True)
            archive.unlink(missing_ok=True)
            _download_source(name, revision, archive, label, console, position)
            _extract_archive(archive, destination, label, console)


def _snapshots(cache_root: Path) -> list[Path]:
    snapshots = [
        path for path in cache_root.iterdir()
        if path.is_dir() and not path.is_symlink() and not path.name.startswith(".") and (path / MANIFEST).is_file()
    ]
    return sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True)


def _reuse_snapshot(name: str, revision: str, destination: Path, cache_root: Path) -> bool:
    prefix = f"external/{name}/"
    for snapshot in _snapshots(cache_root):
        source = snapshot / "external" / name
        try:
            manifest = json.loads((snapshot / MANIFEST).read_text())
            if manifest["identity"]["sources"].get(name) != revision:
                continue
            expected = {key[len(prefix):]: value for key, value in manifest["files"].items() if key.startswith(prefix)}
            if not expected or _files(source) != expected:
                continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        try:
            shutil.copytree(source, destination, copy_function=os.link)
        except OSError:
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)
        print(f"Reusing tool sources from {snapshot.name}: {name} ({revision[:12]})", flush=True)
        return True
    return False


def _referenced_snapshots() -> set[Path] | None:
    executable = shutil.which("docker")
    if not executable:
        return None
    try:
        containers = subprocess.check_output(
            [executable, "ps", "-aq", "--filter", "label=org.opendde-harness.code-id"],
            text=True, timeout=SUBPROCESS_TIMEOUT,
        ).split()
        if not containers:
            return set()
        mounts = subprocess.check_output(
            [executable, "inspect", "--format", "{{range .Mounts}}{{.Source}}{{println}}{{end}}", *containers],
            text=True, timeout=SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {Path(line.strip()) for line in mounts.splitlines() if line.strip()}


def prune_runtime_code(cache_root: Path, current: Path, *, keep: int = 2) -> list[Path]:
    """Remove snapshots that no container mounts, keeping ``current`` and the ``keep`` newest others."""
    referenced = _referenced_snapshots()
    if referenced is None:
        return []
    stale = [
        snapshot for snapshot in _snapshots(cache_root)
        if snapshot != current and not any(mount == snapshot or mount.is_relative_to(snapshot) for mount in referenced)
    ]
    removed = []
    for snapshot in stale[keep:]:
        try:
            shutil.rmtree(snapshot)
        except OSError as exc:
            print(f"Could not remove stale runtime code: {snapshot} ({exc})", flush=True)
            continue
        print(f"Removed stale runtime code: {snapshot}", flush=True)
        removed.append(snapshot)
    return removed


def prepare_runtime_code(
    cache_root: Path | None = None,
    *,
    upstream_dir: Path | None = None,
) -> Path:
    package_dir = Path(__file__).resolve().parents[1]
    identity = runtime_code_identity(package_dir)
    cache_root = (cache_root or default_code_cache()).expanduser().resolve()
    if cache_root in (Path(cache_root.anchor), Path.home().resolve()) or cache_root.is_relative_to(package_dir):
        raise ValueError("Choose a dedicated runtime code cache outside the installed package")
    destination = _code_destination(cache_root, identity)
    if destination.exists() or destination.is_symlink():
        verify_runtime_code(destination, identity)
        print(f"Reusing verified runtime code: {destination}", flush=True)
        return destination
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=cache_root) as temporary:
        staged = Path(temporary) / "code"
        staged.mkdir()
        for name in _files(package_dir):
            target = staged / "opendde_harness" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package_dir / name, target)
        if _digest(_files(staged / "opendde_harness")) != identity["harness_sha256"]:
            raise ValueError("Installed Harness changed during code preparation; retry after the upgrade completes")
        (staged / "external").mkdir()
        (staged / "external/__init__.py").write_text(EXTERNAL_MARKER)
        environment = load_environment()
        environment_path = staged / "opendde_harness/cli/compute_environment.json"
        if not environment_path.exists():
            environment_path.write_text(json.dumps(environment, indent=2) + "\n")
        revisions = source_revisions()
        revisions_path = staged / "opendde_harness/cli/source_versions.env"
        if not revisions_path.exists():
            revisions_path.write_text("".join(f"{name}={value}\n" for name, value in revisions.items()))
        checksums_path = staged / "opendde_harness/cli/model-checksums.sha256"
        if not checksums_path.exists():
            checksums_path.write_text("".join(f"{asset.sha256}  {asset.relative_path}\n" for asset in shared_asset_plan()))
        sources = identity["sources"]
        for index, (name, revision) in enumerate(sources.items(), 1):
            target = staged / "external" / name
            if upstream_dir is None and _reuse_snapshot(name, revision, target, cache_root):
                continue
            try:
                _prepare_upstream(
                    name, revision, target, upstream_dir,
                    position=f"{index} of {len(sources)}", sources_dir=cache_root / "sources",
                )
            except (tarfile.TarError, EOFError) as exc:
                raise ValueError(f"Invalid or incomplete source archive: {name}") from exc
        required = (
            "external/opendde/runner/inference.py", "external/ligandmpnn/model_utils.py",
            "external/ligandmpnn/data_utils.py", "external/plip/plip/plipcmd.py",
        )
        if not all((staged / name).is_file() for name in required):
            raise ValueError("Prepared tool sources are incomplete")
        manifest = {"identity": identity, "files": _files(staged)}
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            staged.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            verify_runtime_code(destination, identity)
    print(f"Runtime code ready: {destination}", flush=True)
    prune_runtime_code(cache_root, destination)
    return destination
