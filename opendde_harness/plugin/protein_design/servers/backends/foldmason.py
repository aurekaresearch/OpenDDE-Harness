"""Deterministic, bounded FoldMason analysis for Reflection."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# FoldMason is an implementation detail of Reflection, not a scientific knob.
# Keeping these limits here gives every run the same bounded, reproducible policy.
MIN_STRUCTURES = 8
MIN_NEW_STRUCTURES = 12
MAX_STRUCTURES = 64
TIMEOUT_SECONDS = 90.0
MAX_REFLECTION_GAP = 3


def foldmason_available() -> bool:
    """Return whether the standalone FoldMason executable is available."""

    return shutil.which("foldmason") is not None


def run_structural_tree(
    candidates: list[dict[str, Any]],
    *,
    task_output: Path,
    cycle: int,
    binder_chain_ids: list[str],
    cdr_regions: Mapping[str, Sequence[int]],
    reflection_interval: int,
    current_parent_id: str | None,
    minimize: bool,
) -> dict[str, Any]:
    """Run or reuse a chain-specific FoldMason tree without failing Reflection."""

    analysis_root = task_output / "agent" / "evolutionary_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    eligible = _eligible_candidates(candidates)
    previous = _latest_manifest(analysis_root, before_cycle=cycle)
    eligible_ids = [str(item["candidate_id"]) for item in eligible]
    previous_ids = set(previous.get("eligible_candidate_ids") or []) if previous else set()
    new_count = len(set(eligible_ids) - previous_ids)
    gap = cycle - int(previous.get("cycle", cycle)) if previous else None

    decision = _schedule(
        eligible_count=len(eligible),
        new_count=new_count,
        reflection_gap=gap,
        reflection_interval=reflection_interval,
        has_cache=previous is not None,
    )
    if decision == "too_few_structures":
        return _status("skipped", decision, cycle, eligible, new_count)
    if decision == "cache_hit" and previous is not None:
        cached = dict(previous)
        cached.update(
            {
                "status": "cache_hit",
                "reason": "insufficient_new_structures",
                "requested_cycle": cycle,
                "eligible_structure_count": len(eligible),
                "new_structure_count": new_count,
            }
        )
        return cached
    if not foldmason_available():
        return _status("failed", "foldmason_not_installed", cycle, eligible, new_count)

    selected = _select_representatives(
        eligible,
        max_structures=MAX_STRUCTURES,
        current_parent_id=current_parent_id,
        cdr_regions=cdr_regions,
        minimize=minimize,
    )
    cycle_root = analysis_root / f"cycle_{cycle:04d}" / "foldmason"
    cycle_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    chains: dict[str, Any] = {}
    warnings: list[str] = []
    try:
        for chain_id in binder_chain_ids:
            chain_root = cycle_root / f"chain_{_safe_name(chain_id)}"
            input_root = chain_root / "inputs"
            input_root.mkdir(parents=True, exist_ok=True)
            inputs: list[Path] = []
            mapping: dict[str, str] = {}
            for index, candidate in enumerate(selected):
                candidate_id = str(candidate["candidate_id"])
                output = input_root / f"{index:04d}_{_safe_name(candidate_id)}.pdb"
                try:
                    _extract_chain(Path(str(candidate["structure_path"])), chain_id, output)
                except (OSError, RuntimeError, ValueError) as exc:
                    warnings.append(f"{candidate_id}/{chain_id}: {exc}")
                    continue
                inputs.append(output)
                mapping[output.stem] = candidate_id
            if len(inputs) < 2:
                warnings.append(f"chain {chain_id}: fewer than two readable structures")
                continue
            prefix = chain_root / "alignment"
            work = chain_root / "work"
            command = [
                "foldmason",
                "easy-msa",
                *(str(path) for path in inputs),
                str(prefix),
                str(work),
                "--threads",
                str(max(1, min(8, os.cpu_count() or 1))),
                "--input-format",
                "1",
                "-v",
                "1",
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                warnings.append(
                    f"chain {chain_id}: FoldMason exited {result.returncode}: {detail[-1000:]}"
                )
                continue
            tree_path = prefix.with_suffix(".nw")
            alignment_path = prefix.with_name(f"{prefix.name}_aa.fa")
            three_di_path = prefix.with_name(f"{prefix.name}_3di.fa")
            if not tree_path.is_file() or not alignment_path.is_file():
                warnings.append(f"chain {chain_id}: FoldMason output is incomplete")
                continue
            chains[chain_id] = {
                "structure_count": len(inputs),
                "tree_path": str(tree_path),
                "alignment_path": str(alignment_path),
                "three_di_alignment_path": str(three_di_path) if three_di_path.is_file() else None,
                "candidate_mapping": mapping,
            }
    except subprocess.TimeoutExpired as exc:
        warnings.append(f"FoldMason timed out after {exc.timeout} seconds")
    except OSError as exc:
        warnings.append(f"FoldMason execution failed: {exc}")

    status = "ran" if chains else "failed"
    manifest = {
        "status": status,
        "reason": decision,
        "cycle": cycle,
        "eligible_structure_count": len(eligible),
        "new_structure_count": new_count,
        "selected_candidate_ids": [str(item["candidate_id"]) for item in selected],
        "eligible_candidate_ids": eligible_ids,
        "chains": chains,
        "warnings": warnings,
        "runtime_seconds": round(time.monotonic() - started, 6),
        "foldmason_version": _foldmason_version(),
    }
    _atomic_json_write(cycle_root / "manifest.json", manifest)
    return manifest


def _schedule(
    *,
    eligible_count: int,
    new_count: int,
    reflection_gap: int | None,
    reflection_interval: int,
    has_cache: bool,
) -> str:
    if eligible_count < MIN_STRUCTURES:
        return "too_few_structures"
    if not has_cache:
        return "initial_build"
    if new_count >= MIN_NEW_STRUCTURES:
        return "enough_new_structures"
    maximum_cycle_gap = max(1, reflection_interval) * MAX_REFLECTION_GAP
    if reflection_gap is not None and reflection_gap >= maximum_cycle_gap:
        return "maximum_reflection_gap"
    return "cache_hit"


def _eligible_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_sequences: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        structure_path = str(candidate.get("structure_path") or "").strip()
        sequence = str(candidate.get("sequence") or "").strip().upper()
        if not candidate_id or not structure_path or not Path(structure_path).is_file():
            continue
        if sequence and sequence in seen_sequences:
            continue
        if sequence:
            seen_sequences.add(sequence)
        eligible.append(candidate)
    return eligible


def _select_representatives(
    candidates: list[dict[str, Any]],
    *,
    max_structures: int,
    current_parent_id: str | None,
    cdr_regions: Mapping[str, Sequence[int]],
    minimize: bool,
) -> list[dict[str, Any]]:
    by_id = {str(item["candidate_id"]): item for item in candidates}
    sequence_signatures = {
        candidate_id: _sequence_signature(item, cdr_regions)
        for candidate_id, item in by_id.items()
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(candidate: dict[str, Any] | None) -> None:
        if candidate is None or len(selected) >= max_structures:
            return
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate_id)

    retained = [
        item for item in candidates if str(item.get("population_action") or "").startswith("retained")
    ]
    for item in sorted(retained, key=lambda value: _objective_key(value, minimize)):
        add(item)

    scored = [item for item in candidates if _finite(item.get("objective"))]
    best = min(scored, key=lambda item: float(item["objective"])) if minimize and scored else (
        max(scored, key=lambda item: float(item["objective"])) if scored else None
    )
    for leaf in (best, by_id.get(str(current_parent_id or ""))):
        for item in _lineage(leaf, by_id):
            add(item)

    latest_cycle = max((int(item.get("cycle")) for item in candidates if isinstance(item.get("cycle"), int)), default=-1)
    for item in sorted(
        (candidate for candidate in candidates if candidate.get("cycle") == latest_cycle),
        key=lambda value: _objective_key(value, minimize),
    ):
        add(item)

    remaining = [item for item in candidates if str(item["candidate_id"]) not in selected_ids]
    minimum_distances = {
        str(item["candidate_id"]): min(
            _signature_distance(
                sequence_signatures[str(item["candidate_id"])],
                sequence_signatures[str(incumbent["candidate_id"])],
            )
            for incumbent in selected
        )
        for item in remaining
    } if selected else {}
    while remaining and len(selected) < max_structures:
        if not selected:
            choice_index, choice = min(
                enumerate(remaining),
                key=lambda pair: _objective_key(pair[1], minimize),
            )
        else:
            choice_index, choice = max(
                enumerate(remaining),
                key=lambda pair: (
                    minimum_distances[str(pair[1]["candidate_id"])],
                    -_objective_key(pair[1], minimize),
                    str(pair[1]["candidate_id"]),
                ),
            )
        add(choice)
        remaining.pop(choice_index)
        choice_signature = sequence_signatures[str(choice["candidate_id"])]
        for item in remaining:
            candidate_id = str(item["candidate_id"])
            distance = _signature_distance(
                sequence_signatures[candidate_id],
                choice_signature,
            )
            minimum_distances[candidate_id] = min(
                minimum_distances.get(candidate_id, distance),
                distance,
            )
    return selected


def _lineage(
    leaf: dict[str, Any] | None,
    by_id: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = leaf
    while current is not None:
        candidate_id = str(current["candidate_id"])
        if candidate_id in seen:
            break
        seen.add(candidate_id)
        path.append(current)
        current = by_id.get(str(current.get("parent_id") or ""))
    path.reverse()
    return path


def _sequence_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    cdr_regions: Mapping[str, Sequence[int]],
) -> float:
    return _signature_distance(
        _sequence_signature(left, cdr_regions),
        _sequence_signature(right, cdr_regions),
    )


def _sequence_signature(
    candidate: Mapping[str, Any],
    cdr_regions: Mapping[str, Sequence[int]],
) -> dict[str, str]:
    chains = _chains(candidate)
    signature: dict[str, str] = {}
    for chain, sequence in chains.items():
        positions = [int(value) for value in cdr_regions.get(chain, [])]
        if positions:
            sequence = "".join(sequence[index] for index in positions if index < len(sequence))
        signature[chain] = sequence
    return signature or {"": str(candidate.get("sequence") or "")}


def _signature_distance(left: Mapping[str, str], right: Mapping[str, str]) -> float:
    fragments = [
        (left.get(chain, ""), right.get(chain, ""))
        for chain in sorted(set(left) | set(right))
    ]
    return sum(_normalized_edit_distance(a, b) for a, b in fragments) / len(fragments)


def _chains(candidate: Mapping[str, Any]) -> dict[str, str]:
    chains = candidate.get("chains")
    if isinstance(chains, Mapping) and chains:
        return {str(chain): str(sequence) for chain, sequence in chains.items()}
    return {"": str(candidate.get("sequence") or "")}


def _normalized_edit_distance(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    return _edit_distance(left, right) / max(len(left), len(right), 1)


def _edit_distance(pattern: str, text: str) -> int:
    """Return exact Levenshtein distance using Myers' bit-vector algorithm."""
    if not pattern:
        return len(text)
    if not text:
        return len(pattern)
    if len(pattern) > len(text):
        pattern, text = text, pattern

    equality_masks: dict[str, int] = {}
    for index, character in enumerate(pattern):
        equality_masks[character] = equality_masks.get(character, 0) | (1 << index)

    positive = ~0
    negative = 0
    distance = len(pattern)
    highest_bit = 1 << (len(pattern) - 1)
    for character in text:
        equal = equality_masks.get(character, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & highest_bit:
            distance += 1
        elif negative_horizontal & highest_bit:
            distance -= 1
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return distance


def _extract_chain(source: Path, chain_id: str, output: Path) -> None:
    from biotite.structure import filter_amino_acids
    from biotite.structure.io import pdb, pdbx

    suffix = source.suffix.lower()
    if suffix == ".pdb":
        atoms = pdb.get_structure(pdb.PDBFile.read(str(source)), model=1)
    elif suffix in {".cif", ".mmcif"}:
        cif_class = getattr(pdbx, "CIFFile", None) or getattr(pdbx, "PDBxFile", None)
        if cif_class is None:
            raise RuntimeError("Biotite does not provide a CIF reader")
        atoms = pdbx.get_structure(cif_class.read(str(source)), model=1)
    else:
        raise ValueError(f"unsupported structure format: {source.suffix}")
    mask = (atoms.chain_id == chain_id) & filter_amino_acids(atoms)
    chain = atoms[mask]
    if chain.array_length() == 0:
        raise ValueError(f"binder chain {chain_id!r} is absent")
    chain.chain_id[:] = "A"
    output.parent.mkdir(parents=True, exist_ok=True)
    pdb_file = pdb.PDBFile()
    pdb.set_structure(pdb_file, chain)
    pdb_file.write(str(output))


def _latest_manifest(root: Path, *, before_cycle: int) -> dict[str, Any] | None:
    manifests = []
    for path in root.glob("cycle_*/foldmason/manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("status") == "ran" and int(value.get("cycle", -1)) < before_cycle:
            manifests.append(value)
    return max(manifests, key=lambda value: int(value["cycle"]), default=None)


def _status(
    status: str,
    reason: str,
    cycle: int,
    eligible: list[dict[str, Any]],
    new_count: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "cycle": cycle,
        "eligible_structure_count": len(eligible),
        "new_structure_count": new_count,
        "chains": {},
        "warnings": [],
        "runtime_seconds": 0.0,
    }


def _objective_key(candidate: Mapping[str, Any], minimize: bool) -> float:
    value = candidate.get("objective")
    if not _finite(value):
        return math.inf
    objective = float(value)
    return objective if minimize else -objective


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "candidate"


def _foldmason_version() -> str | None:
    executable = shutil.which("foldmason")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).strip() or None


def _atomic_json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
