"""Optional CPU analysis, isolated from the compute service's model runtimes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Stable candidate.metrics keys; units are also carried with analysis provenance.
PYROSETTA_METRICS = {
    "rosetta_total_score": "REU",
    "rosetta_interface_dg": "REU",
    "rosetta_interface_sasa": "angstrom^2",
    "rosetta_interface_dg_per_sasa": "100*REU/angstrom^2",
    "rosetta_interface_sc": "dimensionless",
    "rosetta_interface_hbonds": "count",
    "rosetta_interface_unsat_hbonds": "count",
    "rosetta_interface_residues": "count",
}


class PyRosettaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    enabled: bool = False
    max_workers: int = Field(default=4, ge=1, le=128)
    timeout_seconds: float = Field(default=900.0, gt=0)
    relax_repeats: int = Field(default=5, ge=1, le=100)
    max_iter: int = Field(default=200, ge=1)
    constrain_to_start: bool = True
    pack_separated: bool = True
    contact_distance: float = Field(default=5.0, gt=0, le=20)
    seed: int = Field(default=42, ge=1, le=2_147_483_647)
    on_failure: Literal["fail", "continue"] = "fail"


class ContactResidueScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    chain_id: str = Field(min_length=1, max_length=1)
    residue_index: int = Field(ge=0)
    pdb_residue_number: int
    insertion_code: str = Field(default="", max_length=1)
    amino_acid: str = Field(min_length=1, max_length=1)
    partner: Literal["binder", "target"]
    min_partner_distance: float = Field(ge=0)
    bound_score_reu: float
    separated_score_reu: float
    interface_dg_reu: float


class PyRosettaResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    status: Literal["success", "failed", "skipped", "disabled"]
    metrics: dict[str, float] = Field(default_factory=dict)
    contact_residues: list[ContactResidueScore] = Field(default_factory=list)
    error: str | None = None
    relaxed_structure_path: str | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metrics(self) -> PyRosettaResult:
        if self.status == "success":
            if set(self.metrics) != set(PYROSETTA_METRICS):
                raise ValueError("PyRosetta success requires the complete metric set")
            if not self.relaxed_structure_path:
                raise ValueError("PyRosetta success requires a relaxed structure")
            if self.metrics["rosetta_interface_sasa"] <= 0:
                raise ValueError("PyRosetta found no buried interface surface")
        elif self.metrics or self.contact_residues:
            raise ValueError("Unsuccessful PyRosetta analysis cannot publish metrics or residue scores")
        if self.status == "failed" and not self.error:
            raise ValueError("Failed PyRosetta analysis requires an error")
        return self


def interface_chains(binder: list[str], target: list[str]) -> str:
    chains = [*binder, *target]
    if not binder or not target or len(chains) != len(set(chains)):
        raise ValueError("PyRosetta requires nonempty, disjoint binder and target chain IDs")
    if any(len(chain) != 1 or not chain.isascii() or not chain.isalnum() for chain in chains):
        raise ValueError("PyRosetta requires single ASCII letter or digit chain IDs")
    return f"{''.join(binder)}_{''.join(target)}"


def cpu_workers(requested: int, jobs: int) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(requested, available, jobs))


def _analyze_one(
    structure: str,
    sequences: dict[str, str],
    interface: str,
    config: PyRosettaConfig,
    output_dir: Path,
) -> PyRosettaResult:
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        request = {
            "structure_path": structure,
            "sequences": sequences,
            "interface": interface,
            "config": config.model_dump(),
            "output_dir": str(output_dir),
        }
        environment = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        # Threads only supervise children; no PyRosetta objects cross process boundaries.
        with (output_dir / "worker.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "opendde_harness.plugin.protein_design.servers.backends.pyrosetta_worker"],
                input=json.dumps(request),
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_seconds,
                env=environment,
                check=False,
            )
        result_path = output_dir / "analysis.json"
        if completed.returncode != 0:
            raise RuntimeError(f"PyRosetta worker exited {completed.returncode}; see {output_dir / 'worker.log'}")
        result = PyRosettaResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.status not in {"success", "failed"}:
            raise ValueError("PyRosetta worker returned an invalid status")
        if result.status == "success" and (
            Path(result.relaxed_structure_path).resolve() != (output_dir / "relaxed.pdb").resolve()
            or not (output_dir / "relaxed.pdb").is_file()
        ):
            raise ValueError("PyRosetta worker did not write the expected relaxed structure")
        return result
    except subprocess.TimeoutExpired:
        return PyRosettaResult(status="failed", error=f"PyRosetta analysis timed out after {config.timeout_seconds:g}s")
    except Exception as exc:
        return PyRosettaResult(status="failed", error=f"{type(exc).__name__}: {exc}")


def analyze_batch(
    structures: list[str | None],
    sequences: list[dict[str, str]],
    *,
    binder_chains: list[str],
    target_chains: list[str],
    config: PyRosettaConfig,
    output_dir: Path,
) -> list[PyRosettaResult]:
    if len(structures) != len(sequences):
        raise ValueError("PyRosetta structure and sequence counts differ")
    if not config.enabled:
        return [PyRosettaResult(status="disabled") for _ in structures]
    interface = interface_chains(binder_chains, target_chains)

    def run(index: int) -> PyRosettaResult:
        if structures[index] is None:
            return PyRosettaResult(status="skipped", error="Fold failed or has no structure")
        started = time.monotonic()
        result = _analyze_one(structures[index], sequences[index], interface, config, output_dir / str(index))
        result.elapsed_seconds = time.monotonic() - started
        result.provenance = {
            **result.provenance,
            "interface": interface,
            "score_function": "ref2015",
            "config": config.model_dump(),
            "metric_units": PYROSETTA_METRICS,
            "source_structure_path": structures[index],
        }
        return result

    if not structures:
        return []
    with ThreadPoolExecutor(max_workers=cpu_workers(config.max_workers, len(structures))) as executor:
        return list(executor.map(run, range(len(structures))))
