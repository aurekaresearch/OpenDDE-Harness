"""Shared Biotite helpers for residue-level structure contacts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Set
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import load_structure, save_structure


def load_structure_model(path: str | Path) -> AtomArray:
    """Load the first model from a PDB or mmCIF file."""
    structure = load_structure(str(path), model=1)
    if isinstance(structure, AtomArrayStack):
        if structure.stack_depth() == 0:
            raise ValueError(f"Structure contains no models: {path}")
        structure = structure[0]
    if structure.array_length() == 0:
        raise ValueError(f"Structure contains no atoms: {path}")
    return structure


def heavy_atom_mask(atoms: AtomArray) -> np.ndarray:
    """Return a mask excluding hydrogen and deuterium atoms."""
    elements = np.char.upper(np.asarray(atoms.element, dtype=str))
    names = np.char.upper(np.asarray(atoms.atom_name, dtype=str))
    return np.where(
        elements != "",
        ~np.isin(elements, ("H", "D")),
        ~(np.char.startswith(names, "H") | np.char.startswith(names, "D")),
    )


def iter_protein_residues(
    atoms: AtomArray,
    chain_ids: Set[str] | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield atom slices for non-hetero residues containing a CA atom."""
    requested = None if chain_ids is None else {str(chain) for chain in chain_ids}
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        if requested is not None and str(atoms.chain_id[start]) not in requested:
            continue
        if bool(atoms.hetero[start]):
            continue
        if not np.any(atoms.atom_name[start:stop] == "CA"):
            continue
        yield int(start), int(stop)


def protein_atom_mask(
    atoms: AtomArray,
    chain_ids: Set[str] | None = None,
) -> np.ndarray:
    """Return an atom mask covering protein residues in selected chains."""
    mask = np.zeros(atoms.array_length(), dtype=bool)
    for start, stop in iter_protein_residues(atoms, chain_ids):
        mask[start:stop] = True
    return mask


def _protein_chain_sequences(atoms: AtomArray) -> dict[str, str]:
    """Return one-letter protein sequences keyed by structure chain ID."""
    residues: dict[str, list[str]] = {}
    for start, _stop in iter_protein_residues(atoms):
        chain_id = str(atoms.chain_id[start])
        try:
            residue = ProteinSequence.convert_letter_3to1(str(atoms.res_name[start]))
        except KeyError:
            residue = "X"
        residues.setdefault(chain_id, []).append(residue)
    return {chain_id: "".join(sequence) for chain_id, sequence in residues.items()}


def map_semantic_chain_ids(
    atoms: AtomArray,
    expected_sequences: Mapping[str, str],
) -> dict[str, str]:
    """Map configured semantic chain IDs to structure chain IDs by sequence."""
    observed = _protein_chain_sequences(atoms)
    mapping: dict[str, str] = {}
    used: set[str] = set()

    # Preserve already-correct chain IDs before resolving renamed chains.
    for semantic_id, sequence in expected_sequences.items():
        if observed.get(str(semantic_id)) == str(sequence):
            mapping[str(semantic_id)] = str(semantic_id)
            used.add(str(semantic_id))

    for semantic_id, sequence in expected_sequences.items():
        semantic_id = str(semantic_id)
        if semantic_id in mapping:
            continue
        matches = [
            structure_id
            for structure_id, observed_sequence in observed.items()
            if structure_id not in used and observed_sequence == str(sequence)
        ]
        if not matches:
            raise ValueError(
                f"Could not map configured chain {semantic_id!r} "
                f"(length {len(str(sequence))}) to predicted structure chains "
                f"{sorted(observed)}"
            )
        structure_id = matches[0]
        mapping[semantic_id] = structure_id
        used.add(structure_id)

    return mapping


def normalize_structure_chain_ids(
    structure_path: str | Path,
    expected_sequences: Mapping[str, str],
    output_path: str | Path,
) -> tuple[Path, dict[str, str]]:
    """Write a structure whose chain IDs match configured semantic chain IDs."""
    source = Path(structure_path)
    model = load_structure_model(source)
    mapping = map_semantic_chain_ids(model, expected_sequences)
    reverse_mapping = {structure_id: semantic_id for semantic_id, structure_id in mapping.items()}
    normalized = model.copy()
    normalized.chain_id = np.asarray([reverse_mapping.get(str(chain_id), str(chain_id)) for chain_id in model.chain_id])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_structure(destination, normalized)
    return destination, mapping
