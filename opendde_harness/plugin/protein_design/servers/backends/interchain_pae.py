"""Raw directional interchain PAE, independently of normalized design losses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFINITION = (
    "Minimum raw PAE over valid-frame entries of all configured binder-to-target and target-to-binder pairs; "
    "directions are not averaged; no normalization, contact cutoff, CDR or hotspot mask."
)


def _integer_vector(value: Any, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim != 1 or values.dtype.kind not in "iu" or any(isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return values.astype(np.int64)


def _structure_atom_residues(path: Path) -> list[tuple[str, str, str]]:
    """Read exact atom-file order without structure loaders filtering atoms."""
    if path.suffix.lower() in {".cif", ".mmcif"}:
        from biotite.structure.io.pdbx import CIFFile

        atom_site = CIFFile.read(path).block["atom_site"]
        if "pdbx_PDB_model_num" in atom_site and len(set(atom_site["pdbx_PDB_model_num"].as_array(str))) != 1:
            raise ValueError("PAE mapping requires one structure model")
        if "label_alt_id" in atom_site and set(atom_site["label_alt_id"].as_array(str)) - {".", "?", ""}:
            raise ValueError("Alternate atom locations make PAE atom ordering ambiguous")
        return list(
            zip(
                atom_site["label_asym_id"].as_array(str),
                atom_site["label_seq_id"].as_array(str),
                atom_site["label_comp_id"].as_array(str),
            )
        )
    if path.suffix.lower() == ".pdb":
        lines = path.read_text().splitlines()
        if sum(line.startswith("MODEL ") for line in lines) > 1:
            raise ValueError("PAE mapping requires one structure model")
        rows = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
        if any(len(line) < 27 or line[16].strip() for line in rows):
            raise ValueError("Malformed or alternate atom locations prevent PAE mapping")
        return [(line[21].strip(), line[22:27].strip(), line[17:20].strip()) for line in rows]
    raise ValueError("PAE mapping requires a PDB or mmCIF structure")


def _chain_matrix_indices(
    data: Mapping[str, Any], structure_path: Path, sequences: Mapping[str, str], size: int
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    from biotite.sequence import ProteinSequence

    atom_map = _integer_vector(data["atom_to_token_idx"], "atom_to_token_idx")
    asym_ids = _integer_vector(data["token_asym_id"], "token_asym_id")
    atoms = _structure_atom_residues(structure_path)
    if len(atoms) != len(atom_map):
        raise ValueError("atom_to_token_idx length does not match exact structure atom rows")
    if len(asym_ids) != size:
        raise ValueError("token_asym_id length does not match PAE matrix")
    if atom_map.size == 0 or np.any(atom_map < 0) or np.any(atom_map >= size):
        raise ValueError("atom_to_token_idx contains missing or out-of-range matrix indices")
    residue_to_index: dict[tuple[str, str], int] = {}
    index_to_residue: dict[int, tuple[str, str]] = {}
    residue_names: dict[tuple[str, str], str] = {}
    for (chain, residue, name), raw_index in zip(atoms, atom_map):
        if not chain or residue in {"", ".", "?"}:
            raise ValueError("Structure atom lacks an unambiguous protein chain/residue identity")
        key = (str(chain), str(residue))
        index = int(raw_index)
        if residue_to_index.setdefault(key, index) != index or index_to_residue.setdefault(index, key) != key:
            raise ValueError("PAE matrix indices and protein residues are not one-to-one")
        if residue_names.setdefault(key, str(name)) != name:
            raise ValueError("Conflicting residue names in structure atom mapping")
    if len(index_to_residue) != size:
        raise ValueError("Not every PAE matrix index is represented in the structure")
    observed: dict[str, str] = {}
    indices: dict[str, list[int]] = {}
    for (chain, residue), index in residue_to_index.items():
        name = residue_names[(chain, residue)]
        try:
            letter = ProteinSequence.convert_letter_3to1(name)
        except KeyError as exc:
            raise ValueError(f"Cannot map non-protein residue {name!r} to configured sequence") from exc
        observed[chain] = observed.get(chain, "") + letter
        indices.setdefault(chain, []).append(index)
    if observed != dict(sequences):
        raise ValueError("Structure chain IDs/sequences do not match configured protein chains")
    mapping: dict[str, dict[str, int]] = {}
    for chain, selected in indices.items():
        labels = np.unique(asym_ids[selected])
        if len(labels) != 1:
            raise ValueError(f"Structure chain {chain} maps to multiple confidence asym IDs")
        mapping[chain] = {"asym_id": int(labels[0]), "residue_count": len(selected)}
    if len({item["asym_id"] for item in mapping.values()}) != len(mapping):
        raise ValueError("Confidence asym ID spans multiple structure chains")
    return indices, mapping


def measure_min_interchain_pae(
    *,
    confidence_path: str | Path | None,
    structure_path: str | Path | None,
    sequences: Mapping[str, str],
    binder_chains: Sequence[str],
    target_chains: Sequence[str],
) -> dict[str, Any]:
    """Return a raw Å metric with provenance, or an explicit unavailable reason.

    Missing/invalid confidence never becomes zero. The mapping is proved from
    exact structure atom order, not guessed from dictionary order or chain length.
    """
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "units": "angstrom",
        "definition": DEFINITION,
        "binder_chains": list(binder_chains),
        "target_chains": list(target_chains),
        "confidence_path": str(confidence_path) if confidence_path else None,
        "structure_path": str(structure_path) if structure_path else None,
        "frame_policy": "Require token_has_frame=1 on both the PAE row and column",
        "axis_semantics": "binder_to_target uses binder rows/target columns; target_to_binder uses target rows/binder columns",
    }
    try:
        if not binder_chains or not target_chains or set(binder_chains) & set(target_chains):
            raise ValueError("Non-empty, disjoint configured binder and target chains are required")
        if len(set(binder_chains)) != len(binder_chains) or len(set(target_chains)) != len(target_chains):
            raise ValueError("Configured binder/target chain IDs must not be duplicated")
        if not (set(binder_chains) | set(target_chains)) <= set(sequences):
            raise ValueError("Configured binder/target chain IDs are absent from fold sequences")
        if not confidence_path or not structure_path:
            raise ValueError("Raw confidence and structure paths are required")
        data = json.loads(Path(confidence_path).read_text())
        if not isinstance(data, Mapping):
            raise ValueError("Raw confidence must be a JSON object")
        matrix_key = "token_pair_pae" if "token_pair_pae" in data else "pae"
        raw = np.asarray(data[matrix_key])
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1] or raw.dtype.kind not in "iuf":
            raise ValueError("Raw PAE must be a non-empty square numeric matrix")
        if any(isinstance(value, bool) for row in data[matrix_key] for value in row):
            raise ValueError("Raw PAE cannot contain boolean values")
        pae = raw.astype(float)
        if not np.all(np.isfinite(pae)) or np.any(pae < 0):
            raise ValueError("Raw PAE must contain only finite, non-negative angstrom values")
        frames = np.asarray(data["token_has_frame"])
        if (
            frames.shape != (pae.shape[0],)
            or frames.dtype.kind not in "biu"
            or not np.all((frames == 0) | (frames == 1))
        ):
            raise ValueError("token_has_frame must be one boolean/0-or-1 integer per PAE matrix index")
        indices, mapping = _chain_matrix_indices(data, Path(structure_path), sequences, pae.shape[0])
        for chain, selected in indices.items():
            mapping[chain]["valid_frame_count"] = int(np.count_nonzero(frames[selected]))
        binder = [index for chain in binder_chains for index in indices[chain] if frames[index]]
        target = [index for chain in target_chains for index in indices[chain] if frames[index]]
        if not binder or not target:
            raise ValueError("Both binder and target must contain at least one valid-frame PAE index")
        forward = float(pae[np.ix_(binder, target)].min())
        reverse = float(pae[np.ix_(target, binder)].min())
        evidence.update(
            status="success",
            value=min(forward, reverse),
            directional_minima={"binder_to_target": forward, "target_to_binder": reverse},
            matrix_shape=list(pae.shape),
            matrix_key=matrix_key,
            chain_mapping=mapping,
            mapping_method="Exact structure atom rows + atom_to_token_idx, checked against token_asym_id and sequences",
        )
    except Exception as exc:
        evidence["reason"] = f"{type(exc).__name__}: {exc}"
    return evidence
