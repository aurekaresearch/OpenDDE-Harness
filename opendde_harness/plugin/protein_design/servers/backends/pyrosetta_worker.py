"""One fresh PyRosetta process per complex; invoked only by the compute backend."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from opendde_harness.plugin.protein_design.servers.backends.pyrosetta_analysis import (
    ContactResidueScore,
    PyRosettaConfig,
    PyRosettaResult,
)


def _input_pdb(source: Path, destination: Path) -> Path:
    if source.suffix.lower() == ".pdb":
        return source
    if source.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError("PyRosetta input must be PDB or mmCIF")
    from biotite.structure.io import save_structure

    from opendde_harness.plugin.protein_design.servers.backends.structure_contacts import load_structure_model

    atoms = load_structure_model(source)
    if any(len(str(chain)) != 1 for chain in set(atoms.chain_id)):
        raise ValueError("Cannot preserve multi-character mmCIF chain IDs in PyRosetta PDB input")
    save_structure(str(destination), atoms)
    return destination


def _check_pose(pose: Any, sequences: dict[str, str]) -> None:
    observed: dict[str, str] = {}
    for index in range(1, pose.total_residue() + 1):
        residue = pose.residue(index)
        if residue.is_virtual_residue():
            continue
        chain = pose.pdb_info().chain(index)
        observed[chain] = observed.get(chain, "") + residue.name1()
    if observed != sequences:
        raise ValueError("PyRosetta pose chain sequences differ from the folded candidate")


def relax_and_analyze(request: dict[str, Any]) -> PyRosettaResult:
    import pyrosetta

    config = PyRosettaConfig.model_validate(request["config"])
    output_dir = Path(request["output_dir"])
    pyrosetta.init(f"-mute all -constant_seed -jran {config.seed} -multithreading:total_threads 1")
    rosetta = pyrosetta.rosetta
    source = _input_pdb(Path(request["structure_path"]), output_dir / "input.pdb")
    pose = pyrosetta.pose_from_file(str(source))
    _check_pose(pose, request["sequences"])
    scorefxn = pyrosetta.create_score_function("ref2015")
    relax_scorefxn = scorefxn.clone()
    if config.constrain_to_start:
        relax_scorefxn.set_weight(rosetta.core.scoring.coordinate_constraint, 1.0)
    movemap = rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(False)
    relax = rosetta.protocols.relax.FastRelax(relax_scorefxn, config.relax_repeats)
    relax.set_movemap(movemap)
    relax.max_iter(config.max_iter)
    relax.constrain_relax_to_start_coords(config.constrain_to_start)
    relax.ramp_down_constraints(False)
    relax.apply(pose)
    # Coordinate restraints stabilize relaxation but must not affect interface energies.
    pose.remove_constraints()
    _check_pose(pose, request["sequences"])
    total_score = float(scorefxn(pose))
    analyzer = rosetta.protocols.analysis.InterfaceAnalyzerMover()
    partners = rosetta.core.pose.DockingPartners.docking_partners_from_string(request["interface"])
    analyzer.set_interface(partners)
    analyzer.set_scorefunction(scorefxn)
    analyzer.set_pack_input(False)
    analyzer.set_pack_separated(config.pack_separated)
    analyzer.set_compute_interface_energy(True)
    analyzer.set_compute_separated_sasa(True)
    analyzer.set_compute_interface_sc(True)
    analyzer.set_calc_hbond_sasaE(True)
    analyzer.set_compute_interface_delta_hbond_unsat(True)
    analyzer.apply(pose)
    data = analyzer.get_all_data()
    sasa = float(analyzer.get_interface_delta_sasa())
    if sasa <= 0:
        raise ValueError("PyRosetta found no buried interface surface")
    dg = float(analyzer.get_interface_dG())
    contact_residues = _contact_residue_scores(
        pose, request["interface"], config.contact_distance, analyzer.get_all_per_residue_data()
    )
    relaxed_path = output_dir / "relaxed.pdb"
    result = PyRosettaResult(
        status="success",
        relaxed_structure_path=str(relaxed_path),
        contact_residues=contact_residues,
        metrics={
            "rosetta_total_score": total_score,
            "rosetta_interface_dg": dg,
            "rosetta_interface_sasa": sasa,
            "rosetta_interface_dg_per_sasa": 100.0 * dg / sasa,
            "rosetta_interface_sc": float(data.sc_value),
            "rosetta_interface_hbonds": float(data.interface_hbonds),
            "rosetta_interface_unsat_hbonds": float(analyzer.get_interface_delta_hbond_unsat()),
            "rosetta_interface_residues": float(analyzer.get_num_interface_residues()),
        },
        provenance={
            "pyrosetta_version": str(rosetta.utility.Version.version()),
            "steps": ["FastRelax", "InterfaceAnalyzer"],
            "contact_definition": "minimum inter-partner heavy-atom distance <= contact_distance (angstrom)",
            "residue_index_base": 0,
            "contact_residue_count": len(contact_residues),
        },
    )
    pose.dump_pdb(str(relaxed_path))
    return result


def _contact_residue_scores(pose: Any, interface: str, cutoff: float, data: Any) -> list[ContactResidueScore]:
    import numpy as np

    binder, target = interface.split("_")
    residues = []
    chain_indices: dict[str, int] = {}
    for index in range(1, pose.total_residue() + 1):
        residue = pose.residue(index)
        if residue.is_virtual_residue():
            continue
        chain = pose.pdb_info().chain(index)
        position = chain_indices.get(chain, 0)
        chain_indices[chain] = position + 1
        if chain not in binder + target:
            continue
        xyz = [residue.xyz(atom) for atom in range(1, residue.nheavyatoms() + 1)]
        coords = np.asarray([(point.x, point.y, point.z) for point in xyz], dtype=float)
        residues.append((index, chain, position, "binder" if chain in binder else "target", coords))
    partner_atoms = {
        role: np.concatenate([row[4] for row in residues if row[3] == role]) for role in ("binder", "target")
    }
    scores = []
    for index, chain, position, role, coords in residues:
        other = partner_atoms["target" if role == "binder" else "binder"]
        # Bound intermediate arrays by one residue, not all atoms in the complex.
        distance = float(np.sqrt(np.min(np.sum((coords[:, None, :] - other[None, :, :]) ** 2, axis=-1))))
        if distance > cutoff:
            continue
        # IA scores every residue in the configured partners, not just its
        # neighbour-based interface set. Use its paired energies consistently.
        scores.append(
            ContactResidueScore(
                chain_id=chain,
                residue_index=position,
                pdb_residue_number=pose.pdb_info().number(index),
                insertion_code=pose.pdb_info().icode(index).strip(),
                amino_acid=pose.residue(index).name1(),
                partner=role,
                min_partner_distance=distance,
                bound_score_reu=float(data.complexed_energy[index]),
                separated_score_reu=float(data.separated_energy[index]),
                interface_dg_reu=float(data.dG[index]),
            )
        )
    return scores


def main() -> None:
    request = json.load(sys.stdin)
    try:
        result = relax_and_analyze(request)
    except Exception as exc:
        result = PyRosettaResult(status="failed", error=f"{type(exc).__name__}: {exc}")
    result.provenance.update(
        {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "worker_module_path": __file__,
            "pid": os.getpid(),
            "runtime_environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONPATH",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }
    )
    (Path(request["output_dir"]) / "analysis.json").write_text(result.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
