"""
PLIP output parser for extracting binding residue interactions.

Parses PLIP txt reports to extract binder-target residue pairs from
interaction tables (Hydrophobic, Hydrogen Bonds, Salt Bridges, Pi-stacking).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class BindingInteraction:
    """Single binding interaction between binder and target residues."""

    binder_chain: str
    binder_resnum: int
    binder_restype: str
    target_chain: str
    target_resnum: int
    target_restype: str
    interaction_type: str  # 'hydrophobic', 'hbond', 'saltbridge', 'pistack'
    distance: Optional[float] = None


@dataclass
class PLIPAnalysis:
    """Parsed PLIP analysis results for one structure."""

    structure_name: str
    interactions: List[BindingInteraction]
    binder_residues: Set[Tuple[str, int]]  # (chain, resnum)
    target_residues: Set[Tuple[str, int]]  # (chain, resnum)

    def get_target_residues_by_chain(self, chain_id: str) -> Set[int]:
        """Get target residue numbers for a specific chain."""
        return {resnum for ch, resnum in self.target_residues if ch == chain_id}

    def get_binder_residues_by_chain(self, chain_id: str) -> Set[int]:
        """Get binder residue numbers for a specific chain."""
        return {resnum for ch, resnum in self.binder_residues if ch == chain_id}


def _parse_interaction_table(
    lines: List[str],
    interaction_type: str,
    binder_chains: Set[str],
    target_chains: Set[str],
) -> List[BindingInteraction]:
    """Parse one interaction table section from PLIP output."""
    interactions = []
    header_index: Dict[str, int] = {}

    # Find table data rows (start with '|' and contain '|' separators)
    in_table = False
    for line in lines:
        line = line.strip()

        # Skip header and separator rows
        if line.startswith("+=") or line.startswith("+--"):
            in_table = True
            continue
        if not line.startswith("|") or not in_table:
            continue

        # Parse data row
        parts = [p.strip() for p in line.split("|")[1:-1]]  # Remove leading/trailing empty
        if len(parts) < 6:
            continue
        if parts[0].upper() == "RESNR":
            header_index = {name.strip().upper(): index for index, name in enumerate(parts)}
            continue

        try:
            # Common columns: RESNR, RESTYPE, RESCHAIN, RESNR_LIG, RESTYPE_LIG, RESCHAIN_LIG
            resnr = int(parts[header_index.get("RESNR", 0)])
            restype = parts[header_index.get("RESTYPE", 1)]
            reschain = parts[header_index.get("RESCHAIN", 2)]
            resnr_lig = int(parts[header_index.get("RESNR_LIG", 3)])
            restype_lig = parts[header_index.get("RESTYPE_LIG", 4)]
            reschain_lig = parts[header_index.get("RESCHAIN_LIG", 5)]

            distance_columns = {
                "hydrophobic": ("DIST",),
                "hbond": ("DIST_H-A", "DIST_D-A"),
                "saltbridge": ("DIST",),
                "pistack": ("CENTDIST", "DIST"),
                "pication": ("DIST",),
            }
            distance = None
            for column in distance_columns.get(interaction_type, ()):
                index = header_index.get(column)
                if index is None or index >= len(parts):
                    continue
                try:
                    distance = float(parts[index])
                    break
                except ValueError:
                    continue

            # Determine which is binder and which is target
            # PLIP uses PROT for protein (target) and LIG for ligand (binder in our case)
            if reschain in target_chains and reschain_lig in binder_chains:
                interactions.append(
                    BindingInteraction(
                        binder_chain=reschain_lig,
                        binder_resnum=resnr_lig,
                        binder_restype=restype_lig,
                        target_chain=reschain,
                        target_resnum=resnr,
                        target_restype=restype,
                        interaction_type=interaction_type,
                        distance=distance,
                    )
                )
            elif reschain in binder_chains and reschain_lig in target_chains:
                interactions.append(
                    BindingInteraction(
                        binder_chain=reschain,
                        binder_resnum=resnr,
                        binder_restype=restype,
                        target_chain=reschain_lig,
                        target_resnum=resnr_lig,
                        target_restype=restype_lig,
                        interaction_type=interaction_type,
                        distance=distance,
                    )
                )
        except (ValueError, IndexError):
            continue

    return interactions


def parse_plip_report(
    plip_file: Path,
    binder_chains: Set[str],
    target_chains: Set[str],
) -> PLIPAnalysis:
    """
    Parse PLIP txt report and extract binding interactions.

    Args:
        plip_file: Path to PLIP txt output file
        binder_chains: Set of chain IDs for the binder (antibody)
        target_chains: Set of chain IDs for the target (antigen)

    Returns:
        PLIPAnalysis with extracted interactions and residue sets
    """
    if not plip_file.exists():
        raise FileNotFoundError(f"PLIP file not found: {plip_file}")

    with open(plip_file, "r") as f:
        content = f.read()

    lines = content.split("\n")
    all_interactions = []

    # Parse each interaction type section
    interaction_sections = {
        "hydrophobic": r"\*\*Hydrophobic Interactions\*\*",
        "hbond": r"\*\*Hydrogen Bonds\*\*",
        "saltbridge": r"\*\*Salt Bridges\*\*",
        "pistack": r"\*\*pi-Stacking\*\*",
        "pication": r"\*\*pi-Cation Interactions\*\*",
    }

    for interaction_type, pattern in interaction_sections.items():
        for section_start, line in enumerate(lines):
            if not re.search(pattern, line):
                continue
            section_end = next(
                (i for i in range(section_start + 1, len(lines)) if re.match(r"\*\*[^*]+\*\*", lines[i])),
                len(lines),
            )
            all_interactions.extend(
                _parse_interaction_table(
                    lines[section_start:section_end],
                    interaction_type,
                    binder_chains,
                    target_chains,
                )
            )

    # Build residue sets
    binder_residues = {(i.binder_chain, i.binder_resnum) for i in all_interactions}
    target_residues = {(i.target_chain, i.target_resnum) for i in all_interactions}

    structure_name = plip_file.stem.replace("_plip", "")

    return PLIPAnalysis(
        structure_name=structure_name,
        interactions=all_interactions,
        binder_residues=binder_residues,
        target_residues=target_residues,
    )
