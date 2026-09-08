#!/usr/bin/env python3
"""
Epitope analysis backend - distance-based contact detection.

Identifies epitope residues and calculates hotspot coverage using simple
distance cutoffs. Fast, lightweight, no Docker dependency.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from scipy.spatial import cKDTree

CDRRegions = Dict[str, Dict[str, Set[int]]]


def _contiguous_regions(positions: List[int]) -> List[Set[int]]:
    """Split sorted 0-based positions into contiguous regions."""
    if not positions:
        return []
    regions: List[Set[int]] = []
    current = {positions[0]}
    previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            regions.append(current)
            current = set()
        current.add(position)
        previous = position
    regions.append(current)
    return regions


def _parse_yaml_positions(value: Any) -> Optional[List[int]]:
    """Parse the residue-list syntax accepted by ChainConfig."""
    if value is None or value == [] or value == "":
        return None
    tokens = value.split(",") if isinstance(value, str) else value
    if not isinstance(tokens, (list, tuple)):
        raise ValueError(f"Invalid YAML residue list: {value!r}")
    positions: Set[int] = set()
    for token in tokens:
        if isinstance(token, bool):
            raise ValueError(f"Invalid YAML residue token: {token!r}")
        if isinstance(token, int):
            start = end = token
        else:
            bounds = str(token).strip().split(":")
            if len(bounds) == 1:
                start = end = int(bounds[0])
            elif len(bounds) == 2:
                start, end = map(int, bounds)
            else:
                raise ValueError(f"Invalid YAML residue range: {token!r}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid YAML residue range: {token!r}")
        positions.update(range(start, end + 1))
    return sorted(positions)


def _binder_cdr_regions(binder: Dict[str, Any]) -> CDRRegions:
    """Derive named CDR regions from one YAML binder definition."""
    result: CDRRegions = {}
    binder_name = str(binder.get("name") or "unnamed")
    for chain_id, chain in (binder.get("chains") or {}).items():
        sequence = str(chain.get("sequence") or "")
        configured = {
            field: _parse_yaml_positions(chain.get(field))
            for field in ("fixed_residues", "designable_residues", "cdr_regions")
        }
        active = [field for field, value in configured.items() if value is not None]
        if not active:
            raise ValueError(f"Binder {binder_name} chain {chain_id} has no YAML CDR/designable configuration")
        if len(active) > 1:
            raise ValueError(f"Binder {binder_name} chain {chain_id} defines multiple residue specifications")
        field = active[0]
        if field == "fixed_residues":
            positions = sorted(set(range(len(sequence))) - set(configured[field] or []))
        else:
            positions = configured[field] or []
        if any(position >= len(sequence) for position in positions):
            raise ValueError(f"Binder {binder_name} chain {chain_id} has CDR positions outside its sequence")
        suffix = {"VH": "H", "VHH": "H", "VL": "L"}.get(chain.get("chain_type"), chain_id)
        result[chain_id] = {
            f"CDR{index}_{suffix}": region
            for index, region in enumerate(_contiguous_regions(sorted(set(positions))), start=1)
        }
    return result


def _canonical_cdr_regions(regions: CDRRegions) -> tuple:
    return tuple(
        (chain_id, name, tuple(sorted(positions)))
        for chain_id, chain_regions in sorted(regions.items())
        for name, positions in sorted(chain_regions.items())
    )


def load_cdr_regions_from_yaml(
    config_yaml: str | Path,
    binder_name: Optional[str] = None,
) -> CDRRegions:
    """Load binder CDR positions exclusively from the design YAML."""
    config_path = Path(config_yaml)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping")
    binders = list(payload.get("initial_binders") or [])
    if not binders:
        raise ValueError("YAML contains no initial_binders CDR configuration")

    if binder_name:
        binders = [binder for binder in binders if binder.get("name") == binder_name]
        if not binders:
            raise ValueError(f"Binder {binder_name!r} not found in YAML")

    configured = [_binder_cdr_regions(binder) for binder in binders]
    first = configured[0]
    if any(_canonical_cdr_regions(regions) != _canonical_cdr_regions(first) for regions in configured[1:]):
        raise ValueError("YAML binders have different CDR configurations; provide --binder_name")
    return first


def _query_contact_pairs(
    antibody_coords: np.ndarray,
    antigen_coords: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return atom contact pairs in dense-matrix row-major order."""
    if len(antibody_coords) == 0 or len(antigen_coords) == 0:
        empty = np.array([], dtype=int)
        return empty, empty
    antigen_tree = cKDTree(antigen_coords)
    neighbors = antigen_tree.query_ball_point(antibody_coords, r=cutoff)
    antibody_indices = []
    antigen_indices = []
    for antibody_index, matches in enumerate(neighbors):
        for antigen_index in sorted(matches):
            antibody_indices.append(antibody_index)
            antigen_indices.append(antigen_index)
    return (
        np.asarray(antibody_indices, dtype=int),
        np.asarray(antigen_indices, dtype=int),
    )


def _load_biotite_io():
    try:
        import biotite.structure.io.pdb as pdb
        import biotite.structure.io.pdbx as pdbx
    except ImportError as exc:
        raise RuntimeError(
            "biotite with PDB/PDBx structure IO is required. Install/upgrade with: pip install biotite"
        ) from exc

    return pdb, pdbx


def load_structure(structure_file: str):
    """Load structure from PDB or CIF file."""
    path = Path(structure_file)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")

    pdb, pdbx = _load_biotite_io()

    if path.suffix.lower() in [".cif", ".mmcif"]:
        cif_file = pdbx.CIFFile.read(str(path))
        structure = pdbx.get_structure(cif_file, model=1)
    else:  # Assume PDB
        pdb_file = pdb.PDBFile.read(str(path))
        structure = pdb.get_structure(pdb_file, model=1)

    return structure


def get_chain_atoms(structure, chain_ids: List[str], heavy_atoms_only: bool = True):
    """Extract atoms from specific chains."""
    chain_filter = np.isin(structure.chain_id, chain_ids)
    chain_atoms = structure[chain_filter]

    if heavy_atoms_only:
        # Exclude hydrogen atoms
        heavy_filter = ~np.isin(chain_atoms.element, ["H", "D"])
        chain_atoms = chain_atoms[heavy_filter]

    return chain_atoms


def find_contacts(antibody_atoms, antigen_atoms, cutoff: float = 4.5) -> Dict[str, Any]:
    """
    Find contacts between antibody and antigen.

    Returns:
        Dict with epitope residues and contact counts
    """
    epitope_residues = {}  # (chain, res_id) -> (res_name, contact_count)
    antibody_contacting_residues = set()  # (chain, res_id)

    ab_contact_indices, ag_contact_indices = _query_contact_pairs(antibody_atoms.coord, antigen_atoms.coord, cutoff)

    # Count contacts per antigen residue
    for ab_idx, ag_idx in zip(ab_contact_indices, ag_contact_indices):
        ag_chain = antigen_atoms.chain_id[ag_idx]
        ag_res_id = antigen_atoms.res_id[ag_idx]
        ag_res_name = antigen_atoms.res_name[ag_idx]

        key = (ag_chain, ag_res_id)
        if key not in epitope_residues:
            epitope_residues[key] = {
                "chain": ag_chain,
                "residue_id": int(ag_res_id),
                "residue_name": ag_res_name,
                "contacts": 0,
            }
        epitope_residues[key]["contacts"] += 1

        # Track antibody residues involved
        ab_chain = antibody_atoms.chain_id[ab_idx]
        ab_res_id = antibody_atoms.res_id[ab_idx]
        antibody_contacting_residues.add((ab_chain, ab_res_id))

    return {
        "epitope_residues": list(epitope_residues.values()),
        "antibody_contacting_residues": antibody_contacting_residues,
        "contact_pairs": (ab_contact_indices, ag_contact_indices),
    }


def parse_hotspots(hotspots_str: str) -> List[Tuple[str, int]]:
    """Parse hotspots string like 'A:45,A:52,B:67' into list of (chain, res_id)."""
    if not hotspots_str:
        return []

    hotspots = []
    for item in hotspots_str.split(","):
        item = item.strip()
        if ":" in item:
            chain, res_id = item.split(":")
            hotspots.append((chain.strip(), int(res_id.strip())))
        else:
            raise ValueError(f"Invalid hotspot format: {item}. Expected format: 'A:45'")

    return hotspots


def analyze_hotspot_coverage(epitope_residues: List[Dict], hotspots: List[Tuple[str, int]]) -> Dict[str, Any]:
    """Calculate hotspot coverage."""
    if not hotspots:
        return {"provided_hotspots": 0, "contacted_hotspots": 0, "hotspot_coverage": None, "missed_hotspots": []}

    epitope_set = {(r["chain"], r["residue_id"]) for r in epitope_residues}
    contacted_hotspots = [h for h in hotspots if h in epitope_set]
    missed_hotspots = [f"{h[0]}:{h[1]}" for h in hotspots if h not in epitope_set]

    coverage = len(contacted_hotspots) / len(hotspots) if hotspots else 0.0

    return {
        "provided_hotspots": len(hotspots),
        "contacted_hotspots": len(contacted_hotspots),
        "hotspot_coverage": round(coverage, 3),
        "missed_hotspots": missed_hotspots,
    }


def _atom_sequence_positions(atoms) -> np.ndarray:
    """Map each atom to its chain-local 0-based residue sequence position."""
    insertion_codes = getattr(atoms, "ins_code", np.full(len(atoms), ""))
    positions = np.empty(len(atoms), dtype=int)
    residue_positions: Dict[Tuple[str, int, str], int] = {}
    next_position: Dict[str, int] = {}
    for index, (chain, residue_id, insertion_code) in enumerate(zip(atoms.chain_id, atoms.res_id, insertion_codes)):
        chain_id = str(chain)
        key = (chain_id, int(residue_id), str(insertion_code))
        if key not in residue_positions:
            residue_positions[key] = next_position.get(chain_id, 0)
            next_position[chain_id] = residue_positions[key] + 1
        positions[index] = residue_positions[key]
    return positions


def analyze_cdr_contributions(
    antibody_atoms,
    antigen_atoms,
    cdr_regions: CDRRegions,
    contact_pairs: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, Dict[str, int]]:
    """Analyze contacts for YAML-defined 0-based CDR regions."""
    cdr_contributions: Dict[str, Dict[str, Any]] = {}
    cdr_by_position = {}
    for chain_id, chain_regions in cdr_regions.items():
        for cdr_name, positions in chain_regions.items():
            cdr_contributions[cdr_name] = {
                "contacted_residues": set(),
                "total_contacts": 0,
            }
            for position in positions:
                cdr_by_position[(chain_id, position)] = cdr_name

    sequence_positions = _atom_sequence_positions(antibody_atoms)
    antibody_indices, antigen_indices = contact_pairs
    for antibody_index, antigen_index in zip(antibody_indices, antigen_indices):
        cdr_name = cdr_by_position.get(
            (
                str(antibody_atoms.chain_id[antibody_index]),
                int(sequence_positions[antibody_index]),
            )
        )
        if cdr_name is None:
            continue
        contribution = cdr_contributions[cdr_name]
        contribution["total_contacts"] += 1
        contribution["contacted_residues"].add(
            (
                str(antigen_atoms.chain_id[antigen_index]),
                int(antigen_atoms.res_id[antigen_index]),
            )
        )

    return {
        cdr_name: {
            "residues_contacted": len(data["contacted_residues"]),
            "total_contacts": data["total_contacts"],
        }
        for cdr_name, data in cdr_contributions.items()
    }


def analyze_epitope(
    structure_file: str,
    antibody_chains: List[str],
    antigen_chains: List[str],
    cdr_regions: CDRRegions,
    cutoff: float = 4.5,
    hotspots: Optional[List[Tuple[str, int]]] = None,
    include_hydrogen: bool = False,
) -> Dict[str, Any]:
    """
    Main epitope analysis function.

    Args:
        structure_file: Path to PDB/CIF structure
        antibody_chains: List of antibody chain IDs (e.g., ['H', 'L'])
        antigen_chains: List of antigen chain IDs (e.g., ['A'])
        cutoff: Distance cutoff in Angstroms (default: 4.5)
        hotspots: Optional list of known hotspot residues as (chain, res_id)
        cdr_regions: YAML-derived chain-local 0-based CDR positions
        include_hydrogen: Include hydrogen atoms (default: False)

    Returns:
        Dictionary with epitope analysis results
    """
    # Load structure
    structure = load_structure(structure_file)

    # Extract antibody and antigen atoms
    antibody_atoms = get_chain_atoms(structure, antibody_chains, heavy_atoms_only=not include_hydrogen)
    antigen_atoms = get_chain_atoms(structure, antigen_chains, heavy_atoms_only=not include_hydrogen)

    if len(antibody_atoms) == 0:
        raise ValueError(f"No antibody atoms found in chains: {antibody_chains}")
    if len(antigen_atoms) == 0:
        raise ValueError(f"No antigen atoms found in chains: {antigen_chains}")

    # Find contacts
    contact_result = find_contacts(antibody_atoms, antigen_atoms, cutoff)
    epitope_residues = contact_result["epitope_residues"]

    # Sort by contact count
    epitope_residues.sort(key=lambda x: x["contacts"], reverse=True)

    # Calculate statistics
    total_contacts = sum(r["contacts"] for r in epitope_residues)
    epitope_size = len(epitope_residues)

    # Hotspot analysis
    hotspot_analysis = analyze_hotspot_coverage(epitope_residues, hotspots or [])

    # CDR contributions
    cdr_contributions = analyze_cdr_contributions(
        antibody_atoms,
        antigen_atoms,
        cdr_regions,
        contact_result["contact_pairs"],
    )

    result = {
        "structure": Path(structure_file).name,
        "antibody_chains": antibody_chains,
        "antigen_chains": antigen_chains,
        "cutoff": cutoff,
        "epitope_residues": epitope_residues,
        "epitope_size": epitope_size,
        "total_contacts": total_contacts,
        "hotspot_analysis": hotspot_analysis,
        "cdr_contributions": cdr_contributions,
        "cdr_source": "yaml",
        "cdr_regions": {
            chain_id: {name: sorted(positions) for name, positions in chain_regions.items()}
            for chain_id, chain_regions in cdr_regions.items()
        },
    }

    return result


def format_text_output(result: Dict[str, Any]) -> str:
    """Format result as human-readable text."""
    lines = [
        "=== Epitope Analysis ===",
        "",
        f"Structure: {result['structure']}",
        f"Antibody chains: {', '.join(result['antibody_chains'])}",
        f"Antigen chains: {', '.join(result['antigen_chains'])}",
        f"Distance cutoff: {result['cutoff']}Å",
        "",
        "Epitope Summary:",
        f"  Total epitope residues: {result['epitope_size']}",
        f"  Total contacts: {result['total_contacts']}",
    ]

    if result["epitope_size"] > 0:
        avg_contacts = result["total_contacts"] / result["epitope_size"]
        lines.append(f"  Average contacts per residue: {avg_contacts:.1f}")

    # Hotspot coverage
    hotspot = result["hotspot_analysis"]
    if hotspot["provided_hotspots"] > 0:
        lines.extend(
            [
                "",
                "Hotspot Coverage:",
                f"  Provided hotspots: {hotspot['provided_hotspots']}",
                f"  Contacted hotspots: {hotspot['contacted_hotspots']}",
                f"  Coverage: {hotspot['hotspot_coverage'] * 100:.1f}%",
            ]
        )
        if hotspot["missed_hotspots"]:
            lines.append(f"  Missed hotspots: {', '.join(hotspot['missed_hotspots'])}")

    # Top contacted residues
    if result["epitope_residues"]:
        lines.extend(["", "Top Contacted Residues:"])
        for res in result["epitope_residues"][:10]:
            lines.append(
                f"  {res['chain']}:{res['residue_id']:>4}  {res['residue_name']:>3}  {res['contacts']:>3} contacts"
            )

    # CDR contributions
    if result["cdr_contributions"]:
        lines.extend(["", "CDR Contributions:"])
        sorted_cdrs = sorted(result["cdr_contributions"].items(), key=lambda x: x[1]["total_contacts"], reverse=True)
        total = result["total_contacts"]
        for cdr, data in sorted_cdrs:
            pct = data["total_contacts"] / total * 100 if total > 0 else 0
            lines.append(
                f"  {cdr:8s}: {data['residues_contacted']:>2} residues, "
                f"{data['total_contacts']:>3} contacts ({pct:>5.1f}%)"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Epitope analysis using distance-based contact detection")
    parser.add_argument("--structure_file", required=True, help="Path to structure file (PDB or CIF)")
    parser.add_argument("--antibody_chains", required=True, help="Comma-separated antibody chain IDs (e.g., H,L)")
    parser.add_argument("--antigen_chains", required=True, help="Comma-separated antigen chain IDs (e.g., A)")
    parser.add_argument("--cutoff", type=float, default=4.5, help="Distance cutoff in Angstroms (default: 4.5)")
    parser.add_argument("--hotspots", help="Comma-separated known hotspots (e.g., A:45,A:52,A:67)")
    parser.add_argument(
        "--config_yaml", required=True, help="Design YAML containing the binder CDR/fixed-residue configuration"
    )
    parser.add_argument("--binder_name", help="Initial binder name when YAML binders use different CDR configurations")
    parser.add_argument("--include_hydrogen", action="store_true", help="Include hydrogen atoms in analysis")
    parser.add_argument("--format", choices=["json", "text"], default="json", help="Output format (default: json)")
    parser.add_argument("--output", help="Output file path (default: stdout)")

    args = parser.parse_args()

    try:
        # Parse chain IDs
        antibody_chains = [c.strip() for c in args.antibody_chains.split(",")]
        antigen_chains = [c.strip() for c in args.antigen_chains.split(",")]

        # Parse hotspots
        hotspots = parse_hotspots(args.hotspots) if args.hotspots else None
        cdr_regions = load_cdr_regions_from_yaml(args.config_yaml, binder_name=args.binder_name)

        # Run analysis
        result = analyze_epitope(
            structure_file=args.structure_file,
            antibody_chains=antibody_chains,
            antigen_chains=antigen_chains,
            cdr_regions=cdr_regions,
            cutoff=args.cutoff,
            hotspots=hotspots,
            include_hydrogen=args.include_hydrogen,
        )

        # Format output
        if args.format == "json":
            output_text = json.dumps(result, indent=2)
        else:
            output_text = format_text_output(result)

        # Write output
        if args.output:
            Path(args.output).write_text(output_text)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(output_text)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
