"""Upstream-native locations shared by preparation and compute readiness."""

from __future__ import annotations

import os
from pathlib import Path

OPENDDE_COMMON_ASSETS = (
    "components.cif",
    "components.cif.rdkit_mol.pkl",
    "obsolete_to_successor.json",
    "release_date_cache.json",
)

#: Legacy installations default to antibody weights; new tasks resolve by mode
#: on the compute host, unless an explicit checkpoint was provided.
DEFAULT_CHECKPOINT = "opendde_abag.pt"
DESIGN_CHECKPOINTS = {"antibody": DEFAULT_CHECKPOINT, "minibinder": "opendde.pt"}


def resolve_checkpoint(design_type: str = "antibody", explicit: str | None = None) -> Path:
    """Resolve on the compute host; explicit/custom paths retain precedence."""
    if design_type not in DESIGN_CHECKPOINTS:
        raise ValueError(f"Unknown design type: {design_type}")
    configured = explicit or os.environ.get("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", "")
    if configured:
        path = Path(configured).expanduser()
        if explicit or path.name not in DESIGN_CHECKPOINTS.values():
            return path
        return path.with_name(DESIGN_CHECKPOINTS[design_type])
    root = Path(os.environ.get("STRUCTPRED_OPENDDE_ROOT_DIR") or opendde_cache_path()).expanduser()
    return root / "checkpoint" / DESIGN_CHECKPOINTS[design_type]


def checkpoint_status(path: Path, design_type: str) -> dict:
    """Lightweight runtime check; full digest verification belongs to preparation."""
    error = None
    try:
        if design_type == "minibinder" and "abag" in path.name.lower():
            error = "minibinder requires a general protein checkpoint, not antibody weights"
        elif not path.is_file() or not os.access(path, os.R_OK):
            error = "missing or unreadable"
        elif not path.stat().st_size:
            error = "empty file"
        else:
            with path.open("rb") as stream:
                stream.read(1)
    except OSError as exc:
        error = str(exc)
    return {"path": str(path), "ready": error is None, "error": error}


def opendde_cache_path() -> Path:
    """OpenDDE's own data root (checkpoint/, common/): OPENDDE_ROOT_DIR, else ~/.cache/opendde."""
    return Path(os.environ.get("OPENDDE_ROOT_DIR") or Path.home() / ".cache/opendde").expanduser().resolve()


def harness_weights_path() -> Path:
    """Harness tool weights root (soluble_mpnn/, huggingface/): OPENDDE_HARNESS_WEIGHTS_DIR, else ~/.cache/opendde-harness."""
    override = os.environ.get("OPENDDE_HARNESS_WEIGHTS_DIR")
    return Path(override or Path.home() / ".cache/opendde-harness").expanduser().resolve()


def huggingface_cache_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    hf_home = os.path.expandvars(os.path.expanduser(os.environ.get("HF_HOME", str(Path(cache_home) / "huggingface"))))
    cache = os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")))
    return Path(os.path.expandvars(os.path.expanduser(cache)))


def default_asset_paths(external: Path, checkpoint: str = DEFAULT_CHECKPOINT) -> dict[str, str]:
    external = external.expanduser().resolve()
    data = opendde_cache_path()
    return {
        "mpnn_weights": str(external / "ligandmpnn/model_params/solublempnn_v_48_020.pt"),
        "hf_cache": str(huggingface_cache_path()),
        "opendde_code": str(external / "opendde"),
        "opendde_data": str(data),
        "opendde_common": str(data / "common"),
        "opendde_checkpoint": str(data / "checkpoint" / checkpoint),
    }
