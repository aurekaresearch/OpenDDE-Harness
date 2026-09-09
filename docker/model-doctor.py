"""Verify local tools on CPU or CUDA and write an offline execution report."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import socket
import time
from pathlib import Path


def reject_network(*args, **kwargs):
    raise RuntimeError("Local tool verification must not access the network")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("api", "local"), default="api")
    parser.add_argument("--structure", type=Path, help="Small two-chain structure for full tool validation")
    parser.add_argument("--output", type=Path, required=True, help="Final JSON report in a fresh output directory")
    args = parser.parse_args()
    if args.mode == "local" and args.structure is None:
        parser.error("--structure is required for local inference validation")
    if args.output.parent.exists() and any(args.output.parent.iterdir()):
        raise ValueError("Use an empty verification output directory; existing artifacts are preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["OPENDDE_HARNESS_COMPUTE_DEVICE"] = args.device
    os.environ["OPENDDE_HARNESS_PROTEIN_FOLD_EXECUTION_MODE"] = args.mode
    from opendde_harness.cli.compute_assets import inspect_assets, model_environment, source_revisions
    from opendde_harness.cli.compute_environment import check_runtime_environment, resolve_device

    root = Path(os.environ.get("OPENDDE_HARNESS_WEIGHTS_DIR", "/weights")).resolve()
    opendde_root = Path(os.environ.get("OPENDDE_ROOT_DIR", "/opendde")).resolve()
    for key, value in model_environment(str(root), str(opendde_root)).items():
        os.environ.setdefault(key, value)
    socket.socket.connect = reject_network
    socket.create_connection = reject_network
    import torch

    import opendde_harness
    from opendde_harness.plugin.protein_design.servers.harness import PythonProteinDesignHarness

    code_root = Path(opendde_harness.__file__).resolve().parent.parent
    os.environ.setdefault("STRUCTPRED_OPENDDE_CODE_DIR", str(code_root / "external/opendde"))
    torch.set_num_threads(4)
    report = {"device": args.device, "mode": args.mode, "sources": source_revisions(), "checks": [], "network": "disabled"}
    harness = PythonProteinDesignHarness(output_path=str(args.output.parent))

    def invoke(operation, payload):
        return asyncio.run(harness.invoke(operation, payload))

    def require_device(name):
        resident = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        require(bool(resident) == (args.device == "cuda"), f"{name} did not run on {args.device}")

    def check(name, operation):
        started = time.monotonic()
        try:
            result = operation()
            report["checks"].append({"name": name, "status": "passed", "seconds": time.monotonic() - started, "result": result})
            print(f"PASS {name}", flush=True)
            return result
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed", "seconds": time.monotonic() - started, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {exc}", flush=True)
            return None

    def assets():
        result = inspect_assets(
            root,
            opendde_root=opendde_root,
            with_opendde=args.mode == "local",
            verify_hashes=True,
        )
        require(result["ready"], "Required model assets did not pass SHA256 verification")
        return result

    def esm():
        result = invoke("score_esm", {"sequences": ["MKVLWA"], "options": {"batch_size": 2}})
        require(len(result["scores"]) == 1 and math.isfinite(result["scores"][0]), "ESM returned an invalid score")
        require_device("ESM")
        return result

    try:
        check("environment", check_runtime_environment)
        check("device", lambda: resolve_device(args.device))
        check("weights", assets)
        check("esm2_scoring", esm)
        proposals = check("esm2_guided_generation", lambda: invoke("generate_esm2_guided", {
            "parent_id": "fixture", "parent_chains": {"B": "MKVLWA"}, "mutable_positions": {"B": [1, 2]},
            "min_llr": -100, "num_sequences": 2, "options": {"batch_size": 2},
        }))
        if proposals is not None and not proposals.get("result", {}).get("candidates"):
            report["checks"][-1]["status"] = "failed"
            report["checks"][-1]["error"] = "No ESM-guided candidates returned"
        if args.structure:
            import biotite.structure as struc
            from biotite.sequence import ProteinSequence
            from biotite.structure.io import save_structure

            from opendde_harness.plugin.protein_design.servers.backends.structure_contacts import load_structure_model

            fixture = args.output.parent / "fixture.cif"
            shutil.copyfile(args.structure, fixture)
            report["structure_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
            atoms = load_structure_model(fixture)
            chains = {}
            for chain in dict.fromkeys(atoms.chain_id):
                _ids, names = struc.get_residues(atoms[atoms.chain_id == chain])
                chains[str(chain)] = "".join(ProteinSequence.convert_letter_3to1(name) for name in names)
            target, binder = list(chains)[:2]
            mutable = list(range(len(chains[binder])))

            def mpnn():
                result = invoke("generate_soluble_mpnn", {
                    "parent_chains": chains, "structure_path": str(fixture), "num_sequences": 8,
                    "mutable_positions": [f"{binder}:{i}" for i in mutable],
                    "parameters": {"seed": 101, "temperature": 1.0, "wt_bias": 0},
                })
                require(len(result["candidates"]) == 8, "SolubleMPNN returned the wrong candidate count")
                require(all(len(item["chains"][binder]) == len(chains[binder]) and item["chains"][target] == chains[target] for item in result["candidates"]), "SolubleMPNN changed fixed chains or sequence lengths")
                require_device("SolubleMPNN")
                return result

            generated = check("soluble_mpnn_generation", mpnn)
            check("structure_read", lambda: invoke("structure_read", {"path": str(fixture)}))

            def rmsd():
                result = invoke("pose_rmsd", {"reference_path": str(fixture), "mobile_path": str(fixture), "target_chain_ids": [target], "binder_chain_ids": [binder]})
                require(result["rmsd"] < 1e-5, "Self-alignment RMSD is not zero")
                return result
            check("pose_rmsd", rmsd)
            check("epitope_analysis", lambda: invoke("epitope_analysis", {
                "structure_path": str(fixture), "antibody_chains": [binder], "antigen_chains": [target],
                "cdr_regions": {binder: {"CDR3": mutable}},
            }))

            def plip():
                result = invoke("structure_analysis", {"structure_paths": [str(fixture)], "candidate_names": ["fixture"], "binder_chain_ids": [binder], "target_chain_ids": [target]})
                require(result["available"], f"PLIP did not produce a valid report: {result}")
                return result
            check("plip_openbabel", plip)

            def tree():
                require(generated is not None, "SolubleMPNN candidates are unavailable")
                candidates = []
                for index, item in enumerate(generated["candidates"]):
                    copy = atoms[~atoms.hetero & ((atoms.atom_name == "N") | (atoms.atom_name == "CA") | (atoms.atom_name == "C") | (atoms.atom_name == "O"))].copy()
                    residue_ids, _names = struc.get_residues(copy[copy.chain_id == binder])
                    for residue, letter in zip(residue_ids, item["chains"][binder], strict=True):
                        copy.res_name[(copy.chain_id == binder) & (copy.res_id == residue)] = ProteinSequence.convert_letter_1to3(letter)
                    path = args.output.parent / f"variant_{index}.pdb"
                    save_structure(path, copy)
                    candidates.append({"candidate_id": f"variant_{index}", "sequence": item["chains"][binder], "structure_path": str(path), "objective": float(index), "cycle": 0})
                population = args.output.parent / "candidates.json"
                population.write_text(json.dumps(candidates))
                result = invoke("evolution_tree", {"candidates_json_path": str(population), "binder_chain_ids": [binder], "cdr_regions": {binder: mutable}})
                require(result["result"]["structural_analysis"]["status"] == "ran", f"FoldMason did not execute: {result}")
                for details in result["result"]["structural_analysis"]["chains"].values():
                    require(Path(details["tree_path"]).stat().st_size > 0 and Path(details["alignment_path"]).stat().st_size > 0, "FoldMason outputs are empty")
                return result
            check("foldmason_evolution", tree)

            if args.mode == "local":
                def fold():
                    result = invoke("fold", {"candidates": [{"candidate_id": "fixture", "sequence": chains[binder], "chains": chains}], "options": {
                        "execution_mode": "local", "device": args.device, "gpus": "none" if args.device == "cpu" else "0",
                        "use_msa": False, "use_templates": False, "enable_msa_search": False,
                        "seeds": [101], "diffusion_samples": 5, "diffusion_steps": 200, "recycling_cycles": 10,
                        "binder_chain_ids": [binder], "target_chain_ids": [target], "objective_key": "loss",
                        "fixed_residues": {binder: list(range(max(1, len(chains[binder]) // 2)))},
                        "esm2_options": {"batch_size": 2},
                    }})
                    require(len(result["candidates"]) == 1 and result["candidates"][0]["metadata"]["success"], f"OpenDDE prediction/scoring failed: {result}")
                    require(len(list((args.output.parent / "fold").rglob("*_sample_*.cif"))) == 5, "OpenDDE did not produce exactly five structures")
                    return result
                check("opendde_local_prediction_and_scoring", fold)
    except Exception as exc:
        report["checks"].append({"name": "fixture", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            harness.close()
        except Exception as exc:
            report["checks"].append({"name": "cleanup", "status": "failed", "error": str(exc)})
        report["passed"] = sum(item["status"] == "passed" for item in report["checks"])
        report["total"] = len(report["checks"])
        report["ok"] = report["passed"] == report["total"]
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"Report: {args.output} ({report['passed']}/{report['total']})", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
