#!/usr/bin/env python3
"""
Bounded structure interaction analysis runner.

Input: one or more structure files (.pdb/.cif/.mmcif)
Output: one PLIP interaction report txt per input structure

When multiple structures are passed, the inline PLIP invocations run
concurrently via a thread pool, capped by ``--max-parallel`` (default 4).
Each invocation writes its own report file; the script prints one absolute
output path per line in the same order as the input structures.
"""

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import biotite.structure.io as bsio

    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False


def _cif_to_pdb(cif_path: Path, output_dir: Optional[Path] = None) -> Path:
    if not HAS_BIOTITE:
        raise RuntimeError("biotite package is required for .cif/.mmcif conversion")

    if output_dir is None:
        output_dir = cif_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = output_dir / f"{cif_path.stem}.pdb"
    atom_array = bsio.load_structure(str(cif_path))
    bsio.save_structure(str(pdb_path), atom_array)
    return pdb_path


def prepare_structure(structure_path: str) -> tuple[Path, Optional[Path]]:
    input_path = Path(structure_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Structure file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".pdb":
        return input_path, None

    if suffix not in {".cif", ".mmcif"}:
        raise ValueError(f"Unsupported structure format: {input_path.suffix}")

    temp_dir = Path(tempfile.mkdtemp(prefix="plip_pdb_"))
    pdb_path = _cif_to_pdb(input_path, output_dir=temp_dir)
    return pdb_path, temp_dir


def resolve_default_output_dir(structure_path: str) -> Path:
    """
    Resolve default output directory as the run-specific agent dir.
    Preferred: .../outputs/<run_name>/agent
    Fallback:  <structure_parent>/agent
    """
    input_path = Path(structure_path).expanduser().resolve()
    parts = input_path.parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            run_root = Path(*parts[: idx + 2])
            return run_root / "agent"
    return input_path.parent / "agent"


def _normalize_chain_ids(
    binder_chain: Optional[str | Iterable[str]],
) -> list[str]:
    """Normalize CLI chain input into individual PLIP chain ids."""
    if binder_chain is None:
        return []

    raw_values = [binder_chain] if isinstance(binder_chain, str) else list(binder_chain)
    chain_ids: list[str] = []
    for value in raw_values:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                chain_ids.append(token)
    return chain_ids


def analyze_structure(
    structure_path: str,
    output_txt: Optional[str] = None,
    ligand_resname: Optional[str] = None,
    binder_chain: Optional[str | Iterable[str]] = None,
    timeout: int = 300,
) -> Path:
    pdb_path, temp_dir = prepare_structure(structure_path)

    if output_txt is None:
        output_txt_path = Path.cwd() / f"{Path(structure_path).stem}_interactions.txt"
    else:
        output_txt_path = Path(output_txt).expanduser().resolve()
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    plip_output_dir = Path(tempfile.mkdtemp(prefix="plip_output_"))

    plip_command = os.environ.get("PLIP_COMMAND")
    if plip_command and not Path(plip_command).is_file():
        raise RuntimeError(f"Configured PLIP_COMMAND does not exist: {plip_command}")
    plip_command = plip_command or shutil.which("plipcmd.py") or shutil.which("plipcmd")
    if plip_command is None:
        configured = Path("/usr/local/bin/plipcmd.py")
        if configured.is_file():
            plip_command = str(configured)
    if plip_command is None:
        raise RuntimeError("PLIP command not found. Run structure analysis inside the configured compute service.")

    cmd = [
        plip_command,
        "-f",
        str(pdb_path),
        "-o",
        str(plip_output_dir),
        "-t",
    ]

    chain_ids = _normalize_chain_ids(binder_chain)

    if ligand_resname:
        cmd.extend(["--ligand", ligand_resname])
    elif chain_ids:
        cmd.extend(["--inter", *chain_ids])

    try:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"PLIP analysis timed out after {timeout}s") from exc

        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"PLIP analysis failed: {error_text}")

        report_path = plip_output_dir / "report.txt"
        if not report_path.is_file():
            report_path = plip_output_dir / f"{pdb_path.stem}_report.txt"
        if not report_path.is_file():
            raise RuntimeError("PLIP exited without producing an interaction report")
        report_text = report_path.read_text(encoding="utf-8")

        output_txt_path.write_text(report_text, encoding="utf-8")
        return output_txt_path
    finally:
        shutil.rmtree(plip_output_dir, ignore_errors=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_output_path(
    structure_file: str,
    name: Optional[str],
    cycle_label: str,
    output_txt: Optional[str],
    output_dir: Optional[str],
) -> Path:
    final_output_txt = output_txt
    final_name = name or Path(structure_file).stem
    if final_output_txt is None and output_dir:
        return (Path(output_dir) / f"{cycle_label}_{final_name}_plip.txt").expanduser().resolve()
    if final_output_txt is None:
        out_dir = resolve_default_output_dir(structure_file)
        return (out_dir / f"{cycle_label}_{final_name}_plip.txt").resolve()
    return Path(final_output_txt).expanduser().resolve()


def _run_one_analysis(
    structure_file: str,
    output_path: Path,
    ligand_resname: Optional[str],
    binder_chain: Optional[Iterable[str]],
    timeout: int,
) -> Tuple[Path, Optional[str]]:
    """Run analyze_structure for a single input. Returns (output_path, error_or_None).

    On failure, the error report is written to ``output_path`` and a short error
    string is returned so the caller can summarise across all inputs without
    aborting the whole batch.
    """
    try:
        analyze_structure(
            structure_path=structure_file,
            output_txt=str(output_path),
            ligand_resname=ligand_resname,
            binder_chain=binder_chain,
            timeout=timeout,
        )
        return output_path, None
    except Exception as exc:  # noqa: BLE001
        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_text = f"PLIP analysis failed.\nError: {exc}\n\nTraceback:\n{traceback.format_exc()}"
        output_path.write_text(error_text, encoding="utf-8")
        return output_path, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one or more structure files and write PLIP interactions to txt. Multiple inputs run concurrently."
        )
    )
    parser.add_argument(
        "structure_file",
        nargs="*",
        help="Path(s) to .pdb/.cif/.mmcif complex structure(s)",
    )
    parser.add_argument(
        "--cyc-num",
        default=None,
        help="Cycle number used in output filename",
    )
    parser.add_argument(
        "--name",
        default=None,
        nargs="+",
        help=(
            "Name(s) used in output filename. When multiple structures are "
            "given, supply one --name per structure in matching order "
            "(default: each structure file stem)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-txt",
        default=None,
        help=(
            "Path to output txt file (single-input mode only). "
            "When multiple structures are given, outputs are auto-named."
        ),
    )
    parser.add_argument(
        "--ligand-resname",
        default=None,
        help="Optional ligand residue name for protein-ligand analysis",
    )
    parser.add_argument(
        "--binder-chain",
        default=None,
        nargs="+",
        help="Binder chain ID(s) for protein-protein analysis; accepts 'C D' or 'C,D'",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for generated interaction reports",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="PLIP timeout in seconds (per structure)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Max concurrent inline PLIP processes when multiple structures are given",
    )
    args = parser.parse_args()

    structure_files: List[str] = list(args.structure_file or [])
    if not structure_files:
        parser.error("At least one positional structure_file is required.")

    binder_chain = args.binder_chain
    cycle_label = args.cyc_num or "manual"

    names: List[Optional[str]]
    if args.name is None:
        names = [None] * len(structure_files)
    elif len(args.name) != len(structure_files):
        parser.error(f"--name count ({len(args.name)}) must match number of structures ({len(structure_files)})")
    else:
        names = list(args.name)

    if len(structure_files) > 1 and args.output_txt:
        parser.error("--output-txt is only valid when a single structure is given.")

    # Resolve all output paths up-front so we can print them in input order.
    output_paths: List[Path] = [
        _resolve_output_path(
            structure_file=sf,
            name=name,
            cycle_label=cycle_label,
            output_txt=args.output_txt if len(structure_files) == 1 else None,
            output_dir=args.output_dir,
        )
        for sf, name in zip(structure_files, names)
    ]

    n = len(structure_files)
    workers = max(1, min(args.max_parallel, n))
    failures: List[Tuple[str, str]] = []  # (structure_file, error)

    if n == 1 or workers == 1:
        for sf, out in zip(structure_files, output_paths):
            _, err = _run_one_analysis(
                sf,
                out,
                args.ligand_resname,
                binder_chain,
                args.timeout,
            )
            if err:
                failures.append((sf, err))
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _run_one_analysis,
                    sf,
                    out,
                    args.ligand_resname,
                    binder_chain,
                    args.timeout,
                ): sf
                for sf, out in zip(structure_files, output_paths)
            }
            for fut in cf.as_completed(futures):
                _, err = fut.result()
                if err:
                    failures.append((futures[fut], err))

    # One absolute path per input, in input order, so the caller can map them.
    for out in output_paths:
        print(str(out))

    if failures:
        sys.stderr.write(f"{len(failures)}/{n} structure(s) failed PLIP analysis:\n")
        for sf, err in failures:
            sys.stderr.write(f"  - {sf}: {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
