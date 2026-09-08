"""Version contract shared by installed clients and tool environment images."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ENVIRONMENT_LABEL = "org.opendde-harness.environment"
ENVIRONMENT_SHA_LABEL = "org.opendde-harness.environment-sha256"
SUBPROCESS_TIMEOUT = 45

#: Where the tool environment is pulled from. The contract file below is hashed
#: byte for byte into every published image's label, so its own ``image`` field
#: records the reference the environment was sealed with and cannot be
#: re-pointed without invalidating images that are already published. The
#: registry the harness actually pulls from is decided here.
COMPUTE_IMAGE = "aurekaresearch/opendde-harness:v1"


def compute_image() -> str:
    """The image reference to pull, honouring an explicit override."""
    return os.environ.get("OPENDDE_HARNESS_COMPUTE_IMAGE", "").strip() or COMPUTE_IMAGE


def check_local_platform() -> None:
    if platform.system() != "Linux" or platform.machine().lower() != "x86_64":
        raise ValueError("Local compute is supported on Linux x86-64 only. Use a configured Linux compute service.")


def resolve_device(requested: str | None = None) -> str:
    device = str(requested or os.environ.get("OPENDDE_HARNESS_COMPUTE_DEVICE", "auto")).strip().lower()
    if device == "cpu":
        return device
    if device != "auto" and not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("Compute device must be auto, cpu, cuda, or cuda:N; this runtime supports Linux CPU/CUDA.")
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in the compute environment. Select CPU or fix GPU access.")
    if ":" in device and int(device.split(":", 1)[1]) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device is unavailable: {device}")
    return device


@functools.cache
def load_environment(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name("compute_environment.json")
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / "docker/environment.json"
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != 1 or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", spec.get("id", "")):
        raise ValueError(f"Invalid compute environment contract: {path}")
    if not spec.get("packages") or not spec.get("image") or spec.get("platform") != "linux/amd64":
        raise ValueError(f"Incomplete compute environment contract: {path}")
    return spec


def environment_digest(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_image_environment(image: dict[str, Any], spec: dict[str, Any]) -> None:
    labels = (image.get("Config") or {}).get("Labels") or {}
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise ValueError("The compute environment requires a Linux x86-64 image.")
    if labels.get(ENVIRONMENT_LABEL) != spec["id"] or labels.get(ENVIRONMENT_SHA_LABEL) != environment_digest(spec):
        raise ValueError(
            f"Incompatible compute environment: this code requires {spec['id']} ({compute_image()}). "
            "Select a matching environment image; no container was replaced or started."
        )


def check_runtime_environment(path: Path = Path("/opt/compute-environment.json")) -> dict[str, Any]:
    required = load_environment()
    actual = load_environment(path)
    if environment_digest(actual) != environment_digest(required):
        raise ValueError(f"Mounted code requires environment {required['id']}; the container provides {actual['id']}")
    check_local_platform()
    if f"{sys.version_info.major}.{sys.version_info.minor}" != required["python"]:
        raise ValueError("Python version does not match the required tool environment")
    for name, version in required["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Installed dependency differs from the required tool environment: {name}=={version}")
    import torch

    if torch.version.cuda != required["cuda"]:
        raise ValueError("PyTorch CUDA version does not match the required tool environment")
    executable = shutil.which("foldmason")
    if (
        not executable
        or subprocess.check_output([executable, "version"], text=True, timeout=SUBPROCESS_TIMEOUT).strip()
        != required["foldmason_revision"]
    ):
        raise ValueError("FoldMason differs from the required tool environment")
    return {"id": actual["id"], "sha256": environment_digest(actual)}
