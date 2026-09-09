"""Bounded inline PLIP structure-analysis backend."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Union

from opendde_harness.plugin.protein_design.servers.backends.plip_parser import PLIPAnalysis, parse_plip_report

# Reflection must remain a bounded diagnostic.
_REFLECTION_PLIP_MAX_CANDIDATES = 3
_REFLECTION_PLIP_MAX_PARALLEL = 3
_REFLECTION_PLIP_TIMEOUT_SECONDS = 60
_PLIP_SUBPROCESS_GRACE_SECONDS = 15


def _normalize_chain_ids(
    chain_ids: Union[str, List[str]],
    argument_name: str,
) -> List[str]:
    """Normalize comma-, space-, or list-form chain identifiers."""
    values = [chain_ids] if isinstance(chain_ids, str) else chain_ids
    if not isinstance(values, list):
        raise ValueError(f"{argument_name} must be a string or list of strings")

    normalized: List[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{argument_name} must contain only strings")
        normalized.extend(part for part in value.replace(",", " ").split() if part)
    if not normalized:
        raise ValueError(f"{argument_name} must contain at least one chain identifier")
    return normalized


def _format_residues(residues: set[tuple[str, int]], limit: int = 12) -> str:
    ordered = sorted(residues)
    rendered = [f"{chain}:{resnum}" for chain, resnum in ordered[:limit]]
    if len(ordered) > limit:
        rendered.append(f"... (+{len(ordered) - limit} more)")
    return ", ".join(rendered) if rendered else "none"


def _format_contacts(analysis: PLIPAnalysis, limit: int = 8) -> str:
    contacts = []
    for interaction in analysis.interactions[:limit]:
        distance = f", {interaction.distance:.2f} A" if interaction.distance is not None else ""
        contacts.append(
            f"{interaction.binder_chain}:{interaction.binder_resnum}"
            f"-{interaction.binder_restype} -> "
            f"{interaction.target_chain}:{interaction.target_resnum}"
            f"-{interaction.target_restype} ({interaction.interaction_type}{distance})"
        )
    if len(analysis.interactions) > limit:
        contacts.append(f"... (+{len(analysis.interactions) - limit} more)")
    return "; ".join(contacts) if contacts else "none"


def _format_analysis_summary(
    candidate_name: str,
    report_path: Path,
    analysis: PLIPAnalysis,
) -> str:
    counts = Counter(interaction.interaction_type for interaction in analysis.interactions)
    return "\n".join(
        [
            f"Candidate: {candidate_name}",
            f"Report: {report_path}",
            "Interaction counts: "
            f"total={len(analysis.interactions)}, "
            f"hydrophobic={counts['hydrophobic']}, hbond={counts['hbond']}, "
            f"saltbridge={counts['saltbridge']}, pistack={counts['pistack']}, "
            f"pication={counts['pication']}",
            f"Binder residues: {_format_residues(analysis.binder_residues)}",
            f"Antigen residues: {_format_residues(analysis.target_residues)}",
            f"Representative contacts: {_format_contacts(analysis)}",
        ]
    )


def run_plip_structure_analysis(
    structure_files: List[str],
    cycle_num: int,
    candidate_names: List[str],
    binder_chain_ids: Union[str, List[str]],
    target_chain_ids: Union[str, List[str]],
    output_dir: str,
    max_parallel: int = _REFLECTION_PLIP_MAX_PARALLEL,
    timeout: int = _REFLECTION_PLIP_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Return availability and parsed evidence from bounded PLIP execution."""
    if not structure_files:
        raise ValueError("structure_files must contain at least one complex structure")
    if len(structure_files) != len(candidate_names):
        raise ValueError("structure_files and candidate_names must contain the same number of items")
    if len(structure_files) > _REFLECTION_PLIP_MAX_CANDIDATES:
        raise ValueError(f"structure analysis accepts at most {_REFLECTION_PLIP_MAX_CANDIDATES} candidates")

    binder_chains = _normalize_chain_ids(binder_chain_ids, "binder_chain_ids")
    target_chains = _normalize_chain_ids(target_chain_ids, "target_chain_ids")
    safe_parallel = max(1, min(int(max_parallel), len(structure_files)))
    safe_timeout = max(1, int(timeout))
    command = [
        sys.executable,
        "-m",
        "opendde_harness.plugin.protein_design.servers.backends.plip_runner",
        *structure_files,
        "--cyc-num",
        str(cycle_num),
        "--name",
        *candidate_names,
        "--binder-chain",
        *binder_chains,
        "--output_dir",
        output_dir,
        "--max-parallel",
        str(safe_parallel),
        "--timeout",
        str(safe_timeout),
    ]
    batches = (len(structure_files) + safe_parallel - 1) // safe_parallel
    wall_timeout = safe_timeout * batches + _PLIP_SUBPROCESS_GRACE_SECONDS
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "result": (
                "PLIP structure analysis unavailable: bounded runtime exceeded "
                f"{wall_timeout}s. Continue reflection using fold metrics and contact gates."
            ),
        }

    report_paths = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip().endswith("_plip.txt")]
    if len(report_paths) != len(candidate_names):
        return {
            "available": False,
            "result": (
                "PLIP structure analysis unavailable: unexpected report count "
                f"expected {len(candidate_names)}, received {len(report_paths)}. "
                "Continue reflection using fold metrics and contact gates."
            ),
        }

    summaries = [
        (
            "PLIP structure analysis completed."
            if completed.returncode == 0
            else "PLIP structure analysis completed with unavailable candidates."
        ),
        f"Candidates analyzed: {len(candidate_names)}.",
        "Use only the interaction evidence below for structural claims.",
    ]
    succeeded = 0
    for candidate_name, report_path in zip(candidate_names, report_paths):
        if report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except OSError as exc:
                summaries.append(f"Candidate: {candidate_name}\nPLIP status: unavailable ({type(exc).__name__}: {exc})")
                continue
            if report_text.startswith("PLIP analysis failed."):
                error_line = next(
                    (
                        line.removeprefix("Error: ").strip()
                        for line in report_text.splitlines()
                        if line.startswith("Error: ")
                    ),
                    "analysis failed",
                )
                summaries.append(f"Candidate: {candidate_name}\nPLIP status: unavailable ({error_line})")
                continue
        try:
            analysis = parse_plip_report(
                report_path,
                binder_chains=set(binder_chains),
                target_chains=set(target_chains),
            )
        except Exception as exc:
            summaries.append(
                f"Candidate: {candidate_name}\nReport: {report_path}\n"
                f"Parse status: failed ({type(exc).__name__}: {exc})"
            )
            continue
        summaries.append(_format_analysis_summary(candidate_name, report_path, analysis))
        succeeded += 1

    return {"available": succeeded > 0, "result": "\n\n".join(summaries)}
