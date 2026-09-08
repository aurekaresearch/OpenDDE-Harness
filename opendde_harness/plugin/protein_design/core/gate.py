"""Structure-based CDR/hotspot admission gate owned by OpenDDE Harness.

This is intentionally self-contained.  It uses Biotite to load the folded
structure and counts unique binder residues with any heavy atom within the
configured cutoff of an antigen heavy atom.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from pathlib import Path
from typing import Any

DEFAULT_HOTSPOT_CONTACT_CUTOFF_A = 5.0
CDR3_MIN_EPITOPE_CONTACTS = 1
DEFAULT_CDR_CONTACT_FRACTION_THRESHOLD = 0.5


def _hotspot_map(hotspots: Iterable[Any]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for hotspot in hotspots:
        if isinstance(hotspot, Mapping):
            chain = hotspot.get("chain")
            position = hotspot.get("position")
        else:
            chain = getattr(hotspot, "chain", None)
            position = getattr(hotspot, "position", None)
        if chain is not None and position is not None:
            grouped[str(chain)].append(int(position))
    return {chain: sorted(set(positions)) for chain, positions in grouped.items()}


def _load_structure_model(path: str | Path) -> Any:
    from biotite.structure import AtomArrayStack
    from biotite.structure.io import load_structure

    structure = load_structure(str(path), model=1)
    if isinstance(structure, AtomArrayStack):
        if structure.stack_depth() == 0:
            raise ValueError(f"Structure contains no models: {path}")
        structure = structure[0]
    if structure.array_length() == 0:
        raise ValueError(f"Structure contains no atoms: {path}")
    return structure


def _heavy_atom_mask(atoms: Any) -> Any:
    import numpy as np

    elements = np.char.upper(np.asarray(atoms.element, dtype=str))
    names = np.char.upper(np.asarray(atoms.atom_name, dtype=str))
    return np.where(
        elements != "",
        ~np.isin(elements, ("H", "D")),
        ~(np.char.startswith(names, "H") | np.char.startswith(names, "D")),
    )


def _iter_protein_residues(atoms: Any, chain_ids: Set[str] | None = None) -> Iterator[tuple[int, int]]:
    import biotite.structure as struc
    import numpy as np

    requested = None if chain_ids is None else {str(chain) for chain in chain_ids}
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        if requested is not None and str(atoms.chain_id[start]) not in requested:
            continue
        if bool(atoms.hetero[start]):
            continue
        if not np.any(atoms.atom_name[start:stop] == "CA"):
            continue
        yield int(start), int(stop)


def _protein_atom_mask(atoms: Any, chain_ids: Set[str]) -> Any:
    import numpy as np

    mask = np.zeros(atoms.array_length(), dtype=bool)
    for start, stop in _iter_protein_residues(atoms, chain_ids):
        mask[start:stop] = True
    return mask


def _residue_key(atoms: Any, index: int) -> tuple[str, int]:
    return str(atoms.chain_id[index]), int(atoms.res_id[index])


def _sequence_indices(atoms: Any, chain_ids: Sequence[str]) -> dict[tuple[str, int], int]:
    indices: dict[tuple[str, int], int] = {}
    positions: dict[str, int] = defaultdict(int)
    for start, _stop in _iter_protein_residues(atoms, set(chain_ids)):
        key = _residue_key(atoms, start)
        indices[key] = positions[key[0]]
        positions[key[0]] += 1
    return indices


def evaluate_hotspot_contact_map(
    structure_path: str | Path,
    binder_chains: Sequence[str],
    hotspots: Iterable[Any],
    *,
    target_chains: Sequence[str] | None = None,
    distance_cutoff: float = DEFAULT_HOTSPOT_CONTACT_CUTOFF_A,
    cdr3_positions: dict[str, Set[int]] | None = None,
    all_cdr_positions: dict[str, Set[int]] | None = None,
    cdr_contact_fraction_threshold: float = DEFAULT_CDR_CONTACT_FRACTION_THRESHOLD,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate OpenDDE Harness's CDR/hotspot admission policy."""
    import biotite.structure as struc
    import numpy as np

    if not 0.0 <= cdr_contact_fraction_threshold <= 1.0:
        raise ValueError("cdr_contact_fraction_threshold must be between 0 and 1")
    hotspot_map = _hotspot_map(hotspots)
    epitope_set = {
        (chain, position)
        for chain, positions in hotspot_map.items()
        for position in positions
    }
    coverage: dict[str, Any] = {
        "contacted_hotspots": set(),
        "missed_hotspots": set(epitope_set),
        "off_target_contacts": set(),
        "coverage_ratio": 0.0,
        "total_contacts": 0,
        "num_contacted_hotspots": 0,
        "num_missed_hotspots": len(epitope_set),
        "num_off_target": 0,
    }
    model = _load_structure_model(structure_path)
    binder_chain_set = {str(chain) for chain in binder_chains}
    binder_mask = _protein_atom_mask(model, binder_chain_set) & _heavy_atom_mask(model)
    binder_atoms = model[binder_mask]
    sequence_indices = _sequence_indices(model, binder_chains)
    if binder_atoms.array_length() == 0:
        coverage.update(
            {
                "missing_binder_chains": sorted(binder_chain_set),
                "distance_cutoff_a": float(distance_cutoff),
                "cdr3_gate_passed": False,
                "cdr_contact_fraction_gate_passed": False,
                "cdr_contact_fraction": 0.0,
            }
        )
        return False, coverage

    binder_search = struc.CellList(binder_atoms, cell_size=float(distance_cutoff))
    target_chain_set = (
        {str(chain) for chain in target_chains}
        if target_chains is not None
        else {chain for chain, _ in epitope_set}
    )
    contacted_hotspots: set[tuple[str, int]] = set()
    target_residues: set[tuple[str, int]] = set()
    contacted_binder_residues: set[tuple[str, int]] = set()
    all_contact_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    cdr_contact_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    cdr3_contact_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    cdr3_hotspot_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    hotspot_min_distances: dict[str, float] = {}
    binder_min_distances: dict[tuple[str, int], float] = {}
    contact_pairs: list[str] = []

    for start, stop in _iter_protein_residues(model, target_chain_set):
        target_key = _residue_key(model, start)
        residue = model[start:stop]
        residue_contacted = False
        min_distance: float | None = None
        for target_atom in residue[_heavy_atom_mask(residue)]:
            nearby_indices = binder_search.get_atoms(target_atom.coord, radius=float(distance_cutoff))
            for binder_index in np.asarray(nearby_indices, dtype=int):
                binder_atom = binder_atoms[int(binder_index)]
                distance = float(np.linalg.norm(target_atom.coord - binder_atom.coord))
                binder_key = _residue_key(binder_atoms, int(binder_index))
                current_binder_distance = binder_min_distances.get(binder_key)
                if current_binder_distance is None or distance < current_binder_distance:
                    binder_min_distances[binder_key] = distance
                residue_contacted = True
                min_distance = distance if min_distance is None else min(min_distance, distance)
                contacted_binder_residues.add(binder_key)
                pair = (binder_key, target_key)
                all_contact_pairs.add(pair)
                binder_position = sequence_indices.get(binder_key)
                if all_cdr_positions and binder_key[0] in all_cdr_positions and binder_position in all_cdr_positions[binder_key[0]]:
                    cdr_contact_pairs.add(pair)
                if cdr3_positions and binder_key[0] in cdr3_positions and binder_position in cdr3_positions[binder_key[0]]:
                    cdr3_contact_pairs.add(pair)
                    if target_key in epitope_set:
                        cdr3_hotspot_pairs.add(pair)
                if len(contact_pairs) < 24:
                    contact_pairs.append(f"{binder_key[0]}:{binder_key[1]}-{target_key[0]}:{target_key[1]}@{distance:.2f}A")
        if residue_contacted:
            target_residues.add(target_key)
            if target_key in epitope_set:
                contacted_hotspots.add(target_key)
                if min_distance is not None:
                    hotspot_min_distances[f"{target_key[0]}:{target_key[1]}"] = round(min_distance, 3)

    cdr3_gate_passed = not epitope_set or (
        bool(cdr3_positions) and len(cdr3_hotspot_pairs) >= CDR3_MIN_EPITOPE_CONTACTS
    )
    cdr3_contacted_hotspots = {
        target for _binder, target in cdr3_hotspot_pairs
    }
    binder_interface = contacted_binder_residues
    cdr_interface = {binder for binder, _target in cdr_contact_pairs}
    cdr_fraction = len(cdr_interface) / len(binder_interface) if binder_interface else 0.0
    cdr_fraction_passed = bool(all_cdr_positions) and bool(binder_interface) and cdr_fraction > cdr_contact_fraction_threshold
    framework_interface = binder_interface - cdr_interface
    coverage.update(
        {
            "contacted_hotspots": contacted_hotspots,
            "missed_hotspots": epitope_set - contacted_hotspots,
            "off_target_contacts": target_residues - epitope_set,
            "coverage_ratio": len(contacted_hotspots) / len(epitope_set) if epitope_set else 0.0,
            "total_contacts": len(target_residues),
            "num_contacted_hotspots": len(contacted_hotspots),
            "num_missed_hotspots": len(epitope_set - contacted_hotspots),
            "num_off_target": len(target_residues - epitope_set),
            "distance_cutoff_a": float(distance_cutoff),
            "contact_map_method": "heavy_atom_distance",
            "cdr_position_semantics": "zero_based_sequence_order",
            "epitope_gate_skipped": not epitope_set,
            "epitope_gate_skip_reason": "no_hotspots_configured" if not epitope_set else None,
            "contacted_binder_residues": contacted_binder_residues,
            "hotspot_min_distances_a": hotspot_min_distances,
            "contact_pairs": contact_pairs,
            "structure_path": str(structure_path),
            "cdr3_epitope_contacts": len(cdr3_hotspot_pairs),
            "cdr3_total_target_contacts": len(cdr3_contact_pairs),
            "cdr3_contacted_hotspots": cdr3_contacted_hotspots,
            "cdr_total_contacts": len(cdr_contact_pairs),
            "framework_total_contacts": len(all_contact_pairs - cdr_contact_pairs),
            "total_binder_contacts": len(all_contact_pairs),
            "binder_interface_residues": len(binder_interface),
            "cdr_interface_residues": len(cdr_interface),
            "framework_interface_residues": len(framework_interface),
            "n_contact_antibody": len(binder_interface),
            "n_contact_cdr": len(cdr_interface),
            "framework_contact_residue_ids": [
                {"chain": chain, "residue_id": residue_id, "sequence_position": sequence_indices.get((chain, residue_id))}
                | {
                    "distance_to_antigen_a": round(
                        binder_min_distances[(chain, residue_id)],
                        3,
                    )
                }
                for chain, residue_id in sorted(framework_interface)
            ],
            "cdr_contact_fraction": round(cdr_fraction, 4),
            "framework_contact_fraction": round(1.0 - cdr_fraction, 4),
            "cdr_contact_fraction_threshold": float(cdr_contact_fraction_threshold),
            "cdr3_gate_passed": cdr3_gate_passed,
            "cdr_contact_fraction_gate_passed": cdr_fraction_passed,
        }
    )
    return cdr3_gate_passed and cdr_fraction_passed, coverage


def target_aligned_binder_rmsd(
    reference_path: str | Path,
    mobile_path: str | Path,
    *,
    target_chain_ids: Sequence[str],
    binder_chain_ids: Sequence[str],
) -> dict[str, Any]:
    import numpy as np

    reference = _load_structure_model(reference_path)
    mobile = _load_structure_model(mobile_path)
    reference_target, mobile_target = _matched_ca_coordinates(
        reference,
        mobile,
        set(target_chain_ids),
    )
    if len(reference_target) < 3:
        raise ValueError("pose RMSD requires at least three matched target CA atoms")
    reference_center = reference_target.mean(axis=0)
    mobile_center = mobile_target.mean(axis=0)
    centered_reference = reference_target - reference_center
    centered_mobile = mobile_target - mobile_center
    u, _singular, vt = np.linalg.svd(centered_mobile.T @ centered_reference)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    reference_binder, mobile_binder = _matched_ca_coordinates(
        reference,
        mobile,
        set(binder_chain_ids),
    )
    if len(reference_binder) == 0:
        raise ValueError("pose RMSD requires matched binder CA atoms")
    aligned_mobile = (mobile_binder - mobile_center) @ rotation + reference_center
    rmsd = float(
        np.sqrt(np.mean(np.sum((aligned_mobile - reference_binder) ** 2, axis=1)))
    )
    return {
        "rmsd": rmsd,
        "matched_target_atoms": int(len(reference_target)),
        "matched_binder_atoms": int(len(reference_binder)),
        "reference_path": str(reference_path),
        "mobile_path": str(mobile_path),
    }


def _matched_ca_coordinates(
    reference: Any,
    mobile: Any,
    chain_ids: set[str],
) -> tuple[Any, Any]:
    import numpy as np

    def coordinates(atoms: Any) -> dict[tuple[str, int], Any]:
        mask = np.isin(atoms.chain_id, list(chain_ids)) & (atoms.atom_name == "CA")
        selected = atoms[mask]
        return {
            (str(selected.chain_id[index]), int(selected.res_id[index])): selected.coord[index]
            for index in range(selected.array_length())
        }

    reference_coordinates = coordinates(reference)
    mobile_coordinates = coordinates(mobile)
    keys = sorted(set(reference_coordinates) & set(mobile_coordinates))
    return (
        np.asarray([reference_coordinates[key] for key in keys], dtype=float),
        np.asarray([mobile_coordinates[key] for key in keys], dtype=float),
    )
