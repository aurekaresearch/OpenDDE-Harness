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

#: The folding checkpoint a fresh install prepares and mounts. This harness
#: designs antibodies, so the antibody-antigen weights are the default; the
#: general checkpoint stays available as ``--checkpoint opendde.pt``.
DEFAULT_CHECKPOINT = "opendde_abag.pt"


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
