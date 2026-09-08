"""Import-only checks; no model downloads or GPU allocations."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dependencies-only", action="store_true")
parser.add_argument("--environment", type=Path)
args = parser.parse_args()

if args.environment:
    spec = json.loads(args.environment.read_text())
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if os.environ.get("ENVIRONMENT_ID") != spec["id"] or os.environ.get("ENVIRONMENT_SHA256") != digest:
        raise SystemExit("Environment labels do not match the checked contract; build with docker/build.sh")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != spec["python"]:
        raise SystemExit("Python version does not match the environment contract")
    for name, expected in spec["packages"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise SystemExit(f"Environment dependency mismatch: {name}: {actual} != {expected}")
    import torch

    if torch.version.cuda != spec["cuda"]:
        raise SystemExit("PyTorch CUDA version does not match the environment contract")
    executable = shutil.which("foldmason")
    revision = subprocess.check_output([executable, "version"], text=True, timeout=60).strip() if executable else ""
    if revision != spec["foldmason_revision"]:
        raise SystemExit("FoldMason version does not match the environment contract")
    print(f"Environment contract verified: {spec['id']}: {digest}")

for name in ("pyrosetta", "pyright"):
    if importlib.util.find_spec(name) is not None or shutil.which(name):
        raise SystemExit(f"Excluded dependency present: {name}")

for name in (
    "torch",
    "transformers",
    "prody",
    "biotite",
    "rdkit",
):
    importlib.import_module(name)
    print(f"Import OK: {name}")

source_modules = (
    "opendde",
    "runner.inference",
    "runner.msa_search",
    "external.ligandmpnn.model_utils",
    "external.ligandmpnn.data_utils",
    "opendde_harness.plugin.protein_design.servers.api",
    "opendde_harness.plugin.protein_design.servers.backends.soluble_mpnn",
    "opendde_harness.plugin.protein_design.servers.backends.epitope_analysis",
    "opendde_harness.plugin.protein_design.servers.backends.epitope_batch",
    "opendde_harness.plugin.protein_design.servers.backends.evolution_analysis",
    "opendde_harness.plugin.protein_design.servers.backends.lineage_analysis",
    "opendde_harness.plugin.protein_design.servers.backends.plip_runner",
)
if args.dependencies_only:
    for name in ("opendde_harness", "opendde", "runner", "external", "plip"):
        if importlib.util.find_spec(name) is not None:
            raise SystemExit(f"Project source must be mounted, not installed in the image: {name}")
else:
    root = Path(os.environ.get("OPENDDE_HARNESS_PROJECT_ROOT", "/workspace")).resolve()
    for name in source_modules:
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(root):
            raise SystemExit(f"Project module loaded outside the mounted codebase: {name}: {location}")
        print(f"Mounted import OK: {name}: {location}")

commands = [["foldmason", "version"]]
if not args.dependencies_only:
    commands.append(["plipcmd.py", "--help"])
for command in commands:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=120)

print("Dependency checks passed; mounted models and GPU inference require separate verification.")
