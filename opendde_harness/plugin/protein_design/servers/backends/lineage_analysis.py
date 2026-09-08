#!/usr/bin/env python3

"""Objective-aware parent-child lineage analysis for OpenDDE Harness search history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_candidates(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text)
    if path.is_dir():
        for name in ("search_history.json", "population.json"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("candidates") or value.get("history") or []
    if not isinstance(value, list):
        raise ValueError("candidate input must be a JSON list or contain candidates/history")
    return [dict(item) for item in value if isinstance(item, dict)]


def _record(raw: dict[str, Any], metric: str) -> dict[str, Any] | None:
    metadata = raw.get("metadata") or {}
    metrics = raw.get("metrics") or {}
    candidate_id = str(raw.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("OpenDDE Harness lineage candidates require candidate_id")
    value = raw.get("objective") if metric == "objective" else metrics.get(metric)
    score = (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else None
    )
    return {
        "candidate_id": candidate_id,
        "parent_id": str(
            raw.get("parent_id")
            or metadata.get("parent_id")
            or ""
        ).strip() or None,
        "cycle": raw.get("cycle", metadata.get("cycle")),
        "score": score,
        "mutations": list(raw.get("mutations") or metadata.get("mutations") or []),
        "skill_id": raw.get("skill_id") or metadata.get("skill_id"),
    }


def _improvement(child: float, parent: float, minimize: bool) -> float:
    return parent - child if minimize else child - parent


def analyze_lineages(
    candidates: list[dict[str, Any]],
    metric: str = "objective",
    top_k: int = 10,
    min_lineage_length: int = 2,
    minimize: bool = True,
) -> dict[str, Any]:
    records = [record for raw in candidates if (record := _record(raw, metric))]
    by_id = {record["candidate_id"]: record for record in records}
    children: dict[str, list[str]] = {}
    for record in records:
        parent_id = record["parent_id"]
        if parent_id:
            children.setdefault(parent_id, []).append(record["candidate_id"])
    leaves = [record["candidate_id"] for record in records if record["candidate_id"] not in children]
    paths: list[dict[str, Any]] = []
    cycles: list[list[str]] = []
    for leaf in leaves:
        chain: list[dict[str, Any]] = []
        current: str | None = leaf
        visited: set[str] = set()
        cycle_detected = False
        while current and current in by_id:
            if current in visited:
                cycle_detected = True
                break
            visited.add(current)
            record = by_id[current]
            chain.append(record)
            current = record["parent_id"]
        if cycle_detected:
            cycles.append([item["candidate_id"] for item in reversed(chain)])
            continue
        chain.reverse()
        scored = [item for item in chain if item["score"] is not None]
        if len(chain) < min_lineage_length or len(scored) < 2:
            continue
        step_improvements = [
            _improvement(child["score"], parent["score"], minimize)
            for parent, child in zip(scored, scored[1:])
        ]
        net = _improvement(scored[-1]["score"], scored[0]["score"], minimize)
        paths.append(
            {
                "candidate_ids": [item["candidate_id"] for item in chain],
                "scores": [item["score"] for item in scored],
                "net_improvement": net,
                "improving_steps": sum(value > 0 for value in step_improvements),
                "declining_steps": sum(value < 0 for value in step_improvements),
                "consistent_improvement": bool(step_improvements)
                and all(value >= 0 for value in step_improvements),
                "leaf_id": leaf,
            }
        )
    paths.sort(key=lambda item: item["net_improvement"], reverse=True)
    root_ids = sorted(
        record["candidate_id"]
        for record in records
        if not record["parent_id"] or record["parent_id"] not in by_id
    )
    return {
        "metric": metric,
        "minimize": minimize,
        "candidate_count": len(records),
        "root_ids": root_ids,
        "lineage_count": len(paths),
        "improving_lineages": [item for item in paths if item["net_improvement"] > 0][:top_k],
        "declining_lineages": sorted(
            (item for item in paths if item["net_improvement"] < 0),
            key=lambda item: item["net_improvement"],
        )[:top_k],
        "cycle_errors": cycles,
        "unresolved_parent_count": sum(
            bool(record["parent_id"] and record["parent_id"] not in by_id)
            for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_json_path", required=True)
    parser.add_argument("--metric", default="objective")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_lineage_length", type=int, default=2)
    parser.add_argument("--maximize", action="store_true")
    args = parser.parse_args()
    result = analyze_lineages(
        load_candidates(args.candidates_json_path),
        metric=args.metric,
        top_k=args.top_k,
        min_lineage_length=args.min_lineage_length,
        minimize=not args.maximize,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
