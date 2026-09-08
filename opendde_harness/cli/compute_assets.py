"""Prepare OpenDDE weights and common data for the prebuilt compute image."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from opendde_harness.cli._download import HF_ORIGIN, download_file, with_mirrors
from opendde_harness.cli.compute_environment import SUBPROCESS_TIMEOUT, load_environment
from opendde_harness.plugin.protein_design.core.asset_paths import (
    DEFAULT_CHECKPOINT,
    harness_weights_path,
    opendde_cache_path,
)

MODEL_REV = "eddd563ce96571f784012edd8f045181c8f8627d"
MODEL_ROOT = f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{MODEL_REV}"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_FILES = ("config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt")


@functools.cache
def source_revisions(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = Path(__file__).with_name("source_versions.env")
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / "docker/versions.env"
    revisions = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[A-Z][A-Z0-9_]*_REV", key)
            or not re.fullmatch(r"[0-9a-f]{40}", value)
            or key in revisions
        ):
            raise ValueError(f"Invalid or duplicate source revision in {path}: {key}")
        revisions[key] = value
    if not {"OPENDDE_REV", "LIGANDMPNN_REV", "PLIP_REV", "ESM_REV"} <= revisions.keys():
        raise ValueError(f"Missing required source revisions in {path}")
    return revisions


@dataclass(frozen=True)
class Asset:
    relative_path: str
    url: str
    sha256: str
    size: int | None = None
    mirrors: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[str, ...]:
        return (self.url, *self.mirrors)


SOLUBLE_MPNN_WEIGHTS = "soluble_mpnn/solublempnn_v_48_020.pt"
SOLUBLE_MPNN_SOURCES = (
    "https://ipd.graylab.jhu.edu/ligandmpnn/solublempnn_v_48_020.pt",
    f"{MODEL_ROOT}/{SOLUBLE_MPNN_WEIGHTS}",
    "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
)
ESM_SNAPSHOT_DIR = "huggingface/models--facebook--esm2_t33_650M_UR50D"


CHECKPOINTS = {
    "opendde.pt": (2625249069, "7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc"),
    "opendde_abag.pt": (2625271509, "5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d"),
}
COMMON = {
    "components.cif": (490777362, "bb31ae5cf6c8bc669924313077cb4231ee5ffefd3a20118cd14f3ec89f8bb6a5"),
    "components.cif.rdkit_mol.pkl": (142498117, "d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35"),
    "obsolete_to_successor.json": (86882, "2bc08348d0efba438c109bb27be6fa25b611d371c60b8a8da3de387a4a0698ad"),
    "release_date_cache.json": (12754898, "8b1ef12ddc01a0d5eb2d388c77ded91aa906eebce7440726c57b6f8d1a3ec142"),
}


def asset_state_path() -> Path:
    return Path(os.environ.get("OPENDDE_HARNESS_HOME", str(Path.home() / ".opendde_harness"))) / "compute-assets.json"


def prepared_defaults() -> dict[str, str]:
    try:
        state = json.loads(asset_state_path().read_text())
        if state.get("schema_version") != 2:
            return {}
        paths = state.get("paths", {})
        return {key: value for key, value in paths.items() if isinstance(value, str) and Path(value).is_absolute()}
    except (OSError, ValueError, AttributeError):
        return {}


def weights_root(saved: Mapping[str, Any] | None = None) -> Path:
    """Harness tool weights: saved deployment, then prepared assets, then OPENDDE_HARNESS_WEIGHTS_DIR or ~/.cache/opendde-harness."""
    value = (saved or {}).get("weights_dir") or prepared_defaults().get("weights_dir")
    root = Path(value).expanduser().resolve() if value else harness_weights_path()
    # The two trees are separate by design: OpenDDE's own checkpoint and common
    # files under OPENDDE_ROOT_DIR, the harness tool weights beside them. A
    # release from before that split recorded the OpenDDE root here, and every
    # prepare since downloaded a second 2.5 GB copy of the tool weights inside
    # it, which the container then mounted twice.
    return harness_weights_path() if root == opendde_root(saved) else root


def opendde_root(saved: Mapping[str, Any] | None = None) -> Path:
    """OpenDDE data: saved deployment, then prepared assets, then OPENDDE_ROOT_DIR or ~/.cache/opendde."""
    value = (saved or {}).get("opendde_data") or prepared_defaults().get("opendde_data")
    return Path(value).expanduser().resolve() if value else opendde_cache_path()


def asset_plan(checkpoint: str) -> list[Asset]:
    size, digest = CHECKPOINTS[checkpoint]
    assets = [Asset(f"checkpoint/{checkpoint}", f"{MODEL_ROOT}/{checkpoint}", digest, size)]
    assets.extend(
        Asset(f"common/{name}", f"{MODEL_ROOT}/common/{name}", digest, size) for name, (size, digest) in COMMON.items()
    )
    return assets


def shared_asset_plan() -> list[Asset]:
    checksums = Path(__file__).with_name("model-checksums.sha256")
    if not checksums.is_file():
        checksums = Path(__file__).resolve().parents[2] / "docker/model-checksums.sha256"
    revision = source_revisions()["ESM_REV"]
    prefix = f"huggingface/models--facebook--esm2_t33_650M_UR50D/snapshots/{revision}/"
    expected = {SOLUBLE_MPNN_WEIGHTS, *(prefix + name for name in ESM_FILES)}
    assets = []
    for line in checksums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name not in expected or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid shared model manifest entry: {name}")
        expected.remove(name)
        if name.startswith(prefix):
            assets.append(Asset(name, f"{HF_ORIGIN}/facebook/esm2_t33_650M_UR50D/resolve/{revision}/{Path(name).name}", digest))
        else:
            assets.append(Asset(name, SOLUBLE_MPNN_SOURCES[0], digest, mirrors=SOLUBLE_MPNN_SOURCES[1:]))
    if expected:
        raise ValueError(f"Shared model manifest is incomplete: {sorted(expected)}")
    return assets


def model_environment(weights_dir: str = "/weights", opendde_dir: str | None = None) -> dict[str, str]:
    weights = PurePosixPath(weights_dir)
    data = PurePosixPath(opendde_dir or weights_dir)
    return {
        "OPENDDE_ROOT_DIR": str(data),
        "STRUCTPRED_OPENDDE_ROOT_DIR": str(data),
        "STRUCTPRED_OPENDDE_COMMON_DIR": str(data / "common"),
        "STRUCTPRED_OPENDDE_CHECKPOINT_PATH": str(data / "checkpoint" / DEFAULT_CHECKPOINT),
        "HF_HOME": str(weights / "huggingface"),
        "HF_HUB_CACHE": str(weights / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ESM_MODEL_NAME": ESM_MODEL,
        "OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_WEIGHTS_PATH": str(weights / "soluble_mpnn/solublempnn_v_48_020.pt"),
    }


def _prepare_shared_models(root: Path) -> dict[str, str]:
    revision = source_revisions()["ESM_REV"]
    reference = root / "huggingface/models--facebook--esm2_t33_650M_UR50D/refs/main"
    if reference.is_symlink() or (reference.exists() and reference.read_text().strip() != revision):
        raise ValueError("This weights directory selects another ESM revision; choose a new directory instead of changing running models.")
    # A verified copy in an older root (the OpenDDE data root, which once held
    # these too) is copied in instead of downloaded again.
    previous = []
    for path in (opendde_cache_path(), Path.home() / ".cache/opendde"):
        if path.is_dir() and path.resolve() != root.resolve() and path.resolve() not in previous:
            previous.append(path.resolve())
    for asset in shared_asset_plan():
        reuse_asset(root / asset.relative_path, asset, [path / asset.relative_path for path in previous])
        download_asset(root, asset)
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(revision)
    license_path = root / "ESM-LICENSE.txt"
    if not license_path.exists():
        source = Path(__file__).with_name("esm-model-license.txt")
        if not source.is_file():
            source = Path(__file__).resolve().parents[2] / "docker/ESM-LICENSE.txt"
        with license_path.open("x") as handle:
            handle.write(source.read_text())
    return {"weights_dir": str(root)}


def valid_asset(path: Path, asset: Asset) -> bool:
    if not path.is_file() or (asset.size is not None and path.stat().st_size != asset.size):
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset.sha256


def inspect_assets(
    root: Path,
    *,
    opendde_root: Path | None = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    with_opendde: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    data = (opendde_root or root).expanduser().resolve()
    assets = [(root, asset) for asset in shared_asset_plan()]
    if with_opendde:
        assets.extend((data, asset) for asset in asset_plan(checkpoint))
    checks = []
    for base, asset in assets:
        path = base / asset.relative_path
        error = None
        try:
            if not path.is_file() or not path.stat().st_size:
                error = "missing or empty"
            elif asset.size is not None and path.stat().st_size != asset.size:
                error = "size mismatch"
            elif verify_hashes and not valid_asset(path, asset):
                error = "SHA256 mismatch"
        except OSError as exc:
            error = str(exc)
        checks.append({"path": str(path), "ok": error is None, "error": error})
    reference = root / "huggingface/models--facebook--esm2_t33_650M_UR50D/refs/main"
    try:
        matches = reference.read_text().strip() == source_revisions()["ESM_REV"]
    except OSError:
        matches = False
    checks.append({"path": str(reference), "ok": matches, "error": None if matches else "ESM revision mismatch or missing"})
    return {
        "ready": all(item["ok"] for item in checks),
        "root": str(root),
        "opendde_root": str(data) if with_opendde else None,
        "opendde": "required" if with_opendde else "not_required",
        "hashes_verified": verify_hashes,
        "files": checks,
    }


def weights_layout(
    root: Path, opendde_root: Path | None = None, *, with_opendde: bool, checkpoint: str = DEFAULT_CHECKPOINT, translate=None
) -> str:
    """Which root holds what: OpenDDE data (local fold mode only) and the Harness tool weights."""
    t = translate or (lambda en, zh: en)
    lines = []
    if with_opendde:
        lines.append(f"{t('OpenDDE data (OPENDDE_ROOT_DIR):', 'OpenDDE 数据（OPENDDE_ROOT_DIR）：')} {opendde_root or root}")
        lines.append(f"  checkpoint/{checkpoint}, common/")
    lines.append(f"{t('Harness tool weights (OPENDDE_HARNESS_WEIGHTS_DIR):', 'Harness 工具权重（OPENDDE_HARNESS_WEIGHTS_DIR）：')} {root}")
    lines.append(f"  SolubleMPNN: {SOLUBLE_MPNN_WEIGHTS}")
    lines.append(f"  ESM2: {ESM_SNAPSHOT_DIR}")
    return "\n".join(lines)


def download_asset(root: Path, asset: Asset) -> Path:
    destination = root / asset.relative_path
    if destination.exists() or destination.is_symlink():
        if valid_asset(destination, asset):
            print(f"Verified existing asset: {destination}", flush=True)
            return destination
        raise ValueError(
            f"Existing asset has a different identity: {destination}. It was not overwritten; use another asset directory or move it aside yourself."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{asset.sha256[:12]}.part")
    if partial.is_symlink():
        raise ValueError(f"Refusing a symlinked partial download: {partial}")
    print(f"Downloading {asset.relative_path} (partial transfers resume on rerun)", flush=True)
    download_file(
        with_mirrors(asset.sources), partial, sha256=asset.sha256, size=asset.size, description=asset.relative_path
    )
    partial.chmod(0o644)
    partial.replace(destination)
    return destination


def clone_repository(url: str, revision: str, destination: Path) -> None:
    executable = shutil.which("git")
    if not executable:
        raise ValueError("git is required to prepare source checkouts")

    def git(*args: str, cwd: Path = destination, timeout: int = SUBPROCESS_TIMEOUT) -> str:
        try:
            result = subprocess.run(
                [executable, "-C", str(cwd), *args], check=True, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"git {args[0]} failed for {url}: {exc.stderr.strip() or exc.returncode}") from exc
        return result.stdout.strip()

    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        if not (destination / ".git").exists():
            raise ValueError(
                f"Existing source is not a managed Git checkout: {destination}. It was not overwritten; use another asset directory."
            )
        if git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"Source revision differs or has local edits: {destination}. It was left unchanged.")
        print(f"Reusing pinned source: {destination} ({revision[:12]})", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clone-", dir=destination.parent) as temporary:
        checkout = Path(temporary) / "code"
        git(
            "clone", "-c", "http.version=HTTP/1.1", "--filter=blob:none", "--no-checkout", "--depth", "1",
            url, str(checkout), cwd=Path(temporary), timeout=300,
        )
        git("fetch", "--depth", "1", "origin", revision, cwd=checkout, timeout=300)
        git("checkout", "--detach", revision, cwd=checkout, timeout=300)
        actual = git("rev-parse", "HEAD", cwd=checkout)
        if actual != revision:
            raise ValueError(f"Unexpected source revision for {url}: {actual}")
        checkout.rename(destination)


def prepare_sources(root: Path) -> None:
    """Clone the pinned upstream checkouts under a Harness source checkout for development."""
    from opendde_harness.cli.compute_code import EXTERNAL_MARKER

    root = root.expanduser().resolve()
    if not (root / "opendde_harness").is_dir() or not (root / "docker/versions.env").is_file():
        raise ValueError("Source preparation requires a Harness source checkout")
    marker = root / "external/__init__.py"
    if not marker.is_file():
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(EXTERNAL_MARKER)
    revisions = source_revisions(root / "docker/versions.env")
    environment = load_environment(root / "docker/environment.json")
    for name, repository, revision in (
        ("opendde", "aurekaresearch/OpenDDE", revisions["OPENDDE_REV"]),
        ("ligandmpnn", "dauparas/LigandMPNN", revisions["LIGANDMPNN_REV"]),
        ("plip", "pharmai/plip", revisions["PLIP_REV"]),
        ("foldmason", "steineggerlab/foldmason", environment["foldmason_revision"]),
    ):
        clone_repository(f"https://github.com/{repository}.git", revision, root / "external" / name)
    print(f"Mounted codebase ready: {root}", flush=True)


def reuse_asset(destination: Path, asset: Asset, candidates: list[Path]) -> None:
    if destination.exists() or destination.is_symlink():
        return
    for candidate in candidates:
        if not valid_asset(candidate, asset):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(candidate, temporary)
            if not valid_asset(temporary, asset):
                raise ValueError(f"Asset changed while being copied: {candidate}")
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"Asset destination changed during preparation: {destination}")
            temporary.chmod(0o644)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Reused verified asset: {candidate} -> {destination}", flush=True)
        return


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _prepare_downloads(root: Path, checkpoint: str, workers: int, paths: dict[str, str]) -> None:
    plan = asset_plan(checkpoint)
    archived = root / ".legacy"
    previous_roots = sorted(archived.glob("opendde-*"), reverse=True)[:20] if archived.is_dir() else []

    def download(asset: Asset) -> None:
        is_checkpoint = asset.relative_path.startswith("checkpoint/")
        destination = (
            Path(paths["opendde_checkpoint"])
            if is_checkpoint
            else Path(paths["opendde_common"]) / Path(asset.relative_path).name
        )
        if is_checkpoint and destination.name not in CHECKPOINTS:
            if not destination.is_file() or not destination.stat().st_size or not os.access(destination, os.R_OK):
                raise ValueError(
                    f"Explicit custom checkpoint is missing or unreadable: {destination}; no download source was assumed."
                )
            print(
                f"Using explicit custom checkpoint: {destination}; released-model checksum is not asserted.", flush=True
            )
            return
        candidates = [
            previous / "weights/model.pt" if is_checkpoint else previous / "cache/common" / destination.name
            for previous in previous_roots
        ]
        reuse_asset(destination, asset, candidates)
        download_asset(destination.parent, Asset(destination.name, asset.url, asset.sha256, asset.size, asset.mirrors))

    print("Preparing OpenDDE checkpoint first (exclusive download stage).", flush=True)
    download(plan.pop(0))
    print(f"Preparing remaining assets with up to {workers} concurrent downloads.", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download, asset): asset.relative_path for asset in plan}
        for future in as_completed(jobs):
            name = jobs[future]
            try:
                future.result()
                print(f"Asset ready: {name}", flush=True)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                print(f"Asset failed: {name}: {exc}", file=sys.stderr, flush=True)
    if failures:
        raise ValueError(
            "Some compute assets failed; verified files and partial downloads were retained:\n"
            + "\n".join(sorted(failures))
        )


def prepare(
    root: Path,
    checkpoint: str,
    state_file: Path,
    *,
    opendde_root: Path | None = None,
    download_workers: int = 2,
    with_opendde: bool = True,
    paths: dict[str, str] | None = None,
) -> dict:
    import portalocker

    if download_workers not in (1, 2, 3, 4):
        raise ValueError("download_workers must be between 1 and 4")
    root = root.expanduser().resolve()
    data = (opendde_root or opendde_cache_path()).expanduser().resolve()
    for directory in (root, data):
        if directory == Path(directory.anchor) or directory == Path.home().resolve():
            raise ValueError("Choose dedicated asset directories, not a filesystem or home root.")
    root.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(root / ".prepare.lock"), timeout=1):
        print(
            weights_layout(root, data, with_opendde=with_opendde, checkpoint=checkpoint)
            + "\nPreparing inference assets only; large local search databases are excluded.",
            flush=True,
        )
        manifests: dict[Path, list[Asset]] = {root: shared_asset_plan()}
        if with_opendde:
            paths = {
                "opendde_data": str(data),
                "opendde_common": str(data / "common"),
                "opendde_checkpoint": str(data / "checkpoint" / checkpoint),
                **{key: value for key, value in (paths or {}).items() if key.startswith("opendde_")},
            }
            data.mkdir(parents=True, exist_ok=True)
            _prepare_downloads(data, checkpoint, download_workers, paths)
            manifests.setdefault(data, []).extend(asset_plan(checkpoint))
        else:
            paths = {}
        paths.update(_prepare_shared_models(root))
        # Every listed file was verified against its published digest above, so
        # the manifests can be rebuilt from the plan without rereading gigabytes.
        for base, plan in manifests.items():
            (base / "SHA256SUMS").write_text(
                "".join(f"{asset.sha256}  {asset.relative_path}\n" for asset in plan if (base / asset.relative_path).is_file())
            )
        state = {
            "schema_version": 2,
            "root": str(root),
            "opendde_root": str(data) if with_opendde else None,
            "checkpoint": Path(paths["opendde_checkpoint"]).name if with_opendde else None,
            "paths": paths,
            "with_opendde": with_opendde,
            "model_revision": MODEL_REV,
            "esm_revision": source_revisions()["ESM_REV"],
        }
        write_json(state_file, state)
        print(
            "Compute assets are ready. Docker, a compatible compute image, and GPU drivers are still required for local GPU execution.",
            flush=True,
        )
        return state
