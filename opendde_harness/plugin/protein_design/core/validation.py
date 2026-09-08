"""Distribution-level comparison for legacy and OpenDDE Harness design runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from opendde_harness.plugin.protein_design.core.constants import is_canonical_sequence


class RunQuality(BaseModel):
    candidate_count: int
    format_compliance: float
    fold_success_rate: float
    scored_rate: float
    unique_sequence_rate: float
    best_objective: float | None
    ranked_ids: list[str]


class ParityReport(BaseModel):
    legacy: RunQuality
    opendde_harness: RunQuality
    top_k_overlap: float
    objective_delta: float | None


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("candidates", value.get("population", []))
    if not isinstance(value, list):
        raise ValueError("candidate file must contain a list or an object with candidates")
    return [item for item in value if isinstance(item, dict)]


def summarize(candidates: list[dict[str, Any]], *, minimize: bool = True) -> RunQuality:
    count = len(candidates)
    valid = [item for item in candidates if _valid_candidate(item)]
    folded = [item for item in candidates if _fold_succeeded(item)]
    scored = [item for item in candidates if _objective(item) is not None]
    unique = {str(item.get("sequence", "")) for item in valid}
    ranked = sorted(
        scored,
        key=lambda item: _objective(item),
        reverse=not minimize,
    )
    return RunQuality(
        candidate_count=count,
        format_compliance=len(valid) / count if count else 0.0,
        fold_success_rate=len(folded) / count if count else 0.0,
        scored_rate=len(scored) / count if count else 0.0,
        unique_sequence_rate=len(unique) / len(valid) if valid else 0.0,
        best_objective=_objective(ranked[0]) if ranked else None,
        ranked_ids=[str(item.get("candidate_id") or item.get("name") or "") for item in ranked],
    )


def compare_runs(
    legacy_candidates: list[dict[str, Any]],
    opendde_harness_candidates: list[dict[str, Any]],
    *,
    top_k: int = 20,
    minimize: bool = True,
) -> ParityReport:
    legacy = summarize(legacy_candidates, minimize=minimize)
    opendde_harness = summarize(opendde_harness_candidates, minimize=minimize)
    legacy_top = set(legacy.ranked_ids[:top_k])
    opendde_harness_top = set(opendde_harness.ranked_ids[:top_k])
    denominator = max(1, min(top_k, len(legacy_top | opendde_harness_top)))
    delta = None
    if legacy.best_objective is not None and opendde_harness.best_objective is not None:
        delta = opendde_harness.best_objective - legacy.best_objective
    return ParityReport(
        legacy=legacy,
        opendde_harness=opendde_harness,
        top_k_overlap=len(legacy_top & opendde_harness_top) / denominator,
        objective_delta=delta,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare legacy and OpenDDE Harness protein-design populations")
    parser.add_argument("legacy")
    parser.add_argument("opendde_harness")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--maximize", action="store_true")
    args = parser.parse_args()
    report = compare_runs(
        load_candidates(args.legacy),
        load_candidates(args.opendde_harness),
        top_k=args.top_k,
        minimize=not args.maximize,
    )
    print(report.model_dump_json(indent=2))


def _valid_candidate(candidate: dict[str, Any]) -> bool:
    return is_canonical_sequence(candidate.get("sequence", ""))


def _fold_succeeded(candidate: dict[str, Any]) -> bool:
    metadata = candidate.get("metadata") or {}
    if "success" in metadata:
        return bool(metadata["success"])
    return bool(candidate.get("structure_path"))


def _objective(candidate: dict[str, Any]) -> float | None:
    value = candidate.get("objective")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
