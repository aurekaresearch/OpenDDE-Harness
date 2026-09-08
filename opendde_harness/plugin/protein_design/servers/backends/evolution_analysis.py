#!/usr/bin/env python3
"""Evolution analysis backend over an append-only design history."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .lineage_analysis import analyze_lineages


def _load_json(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text)
    if path.is_dir():
        for name in ("search_history.json", "population.json"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("history") or value.get("candidates") or []
    if not isinstance(value, list):
        raise ValueError("candidate input must be a JSON list or contain history/candidates")
    return [dict(item) for item in value if isinstance(item, dict)]


def _candidate_id(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("candidate_id") or "").strip()
    if not value:
        raise ValueError("OpenDDE Harness evolutionary candidates require candidate_id")
    return value


def _parent_id(candidate: dict[str, Any]) -> str | None:
    metadata = candidate.get("metadata") or {}
    value = candidate.get("parent_id") or metadata.get("parent_id")
    return str(value).strip() if value else None


def _score(candidate: dict[str, Any], metric: str) -> float | None:
    metrics = candidate.get("metrics") or {}
    value = (
        candidate.get("objective")
        if metric == "objective"
        else metrics.get(metric)
    )
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _mutations(candidate: dict[str, Any]) -> list[str]:
    metadata = candidate.get("metadata") or {}
    result = []
    for mutation in candidate.get("mutations") or metadata.get("mutations") or []:
        if isinstance(mutation, dict):
            chain = mutation.get("chain_id") or mutation.get("chain") or "?"
            position = mutation.get("position")
            before = mutation.get("from_aa") or "?"
            after = mutation.get("to_aa") or "?"
            result.append(f"{chain}:{position}:{before}>{after}")
        elif str(mutation).strip():
            result.append(str(mutation).strip())
    return result


def _improvement(child: float, parent: float, minimize: bool) -> float:
    return parent - child if minimize else child - parent


def mutation_analysis(
    candidates: list[dict[str, Any]],
    *,
    metric: str,
    minimize: bool,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    by_id = {_candidate_id(item): item for item in candidates if _candidate_id(item)}
    occurrences: Counter[str] = Counter()
    parent_origins: dict[str, set[str]] = defaultdict(set)
    improvements: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        child_score = _score(candidate, metric)
        parent = by_id.get(_parent_id(candidate) or "")
        parent_score = _score(parent, metric) if parent else None
        for mutation in set(_mutations(candidate)):
            occurrences[mutation] += 1
            parent_origins[mutation].add(_parent_id(candidate) or "root")
            if child_score is not None and parent_score is not None:
                improvements[mutation].append(
                    _improvement(child_score, parent_score, minimize)
                )
    rows = []
    for mutation, count in occurrences.items():
        deltas = improvements.get(mutation, [])
        rows.append(
            {
                "mutation": mutation,
                "candidate_count": count,
                "independent_parent_count": len(parent_origins[mutation]),
                "improved_count": sum(delta > 0 for delta in deltas),
                "mean_improvement": sum(deltas) / len(deltas) if deltas else None,
            }
        )
    rows.sort(
        key=lambda row: (
            row["independent_parent_count"],
            row["candidate_count"],
            row["mean_improvement"] if row["mean_improvement"] is not None else -math.inf,
        ),
        reverse=True,
    )
    return rows[:top_k]


def conservation_profile(path_text: str, top_k: int = 15) -> dict[str, Any]:
    sequences: list[str] = []
    current = ""
    for line in Path(path_text).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current:
                sequences.append(current)
            current = ""
        elif line:
            current += line
    if current:
        sequences.append(current)
    if not sequences:
        raise ValueError(f"alignment contains no sequences: {path_text}")
    length = max(map(len, sequences))
    positions = []
    for index in range(length):
        residues = [sequence[index] for sequence in sequences if index < len(sequence) and sequence[index] != "-"]
        if not residues:
            continue
        counts = Counter(residues)
        dominant, count = counts.most_common(1)[0]
        positions.append(
            {
                "alignment_position": index + 1,
                "dominant_residue": dominant,
                "dominant_fraction": count / len(residues),
            }
        )
    return {
        "sequence_count": len(sequences),
        "alignment_length": length,
        "most_conserved": sorted(
            positions, key=lambda item: item["dominant_fraction"], reverse=True
        )[:top_k],
        "most_variable": sorted(
            positions, key=lambda item: item["dominant_fraction"]
        )[:top_k],
    }


def newick_summary(path_text: str) -> dict[str, Any]:
    text = Path(path_text).read_text(encoding="utf-8").strip()
    if not text or text.count("(") != text.count(")"):
        raise ValueError(f"invalid Newick tree: {path_text}")
    labels = [
        match.strip().strip("'\"")
        for match in re.findall(r"(?:^|[(,])\s*([^():;,]+)", text)
        if match.strip()
    ]
    return {
        "leaf_count": len(labels),
        "sample_leaf_ids": labels[:20],
        "group_count": text.count("("),
    }


def analyze_evolution(
    candidates: list[dict[str, Any]],
    *,
    lineage_analysis: dict[str, Any],
    metric: str,
    minimize: bool,
    current_parent_id: str | None = None,
    tree_paths: dict[str, str] | None = None,
    alignment_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    trees = {}
    conservation = {}
    errors = []
    for chain, path in (tree_paths or {}).items():
        try:
            trees[chain] = newick_summary(path)
        except (OSError, ValueError) as exc:
            errors.append(f"tree {chain}: {exc}")
    for chain, path in (alignment_paths or {}).items():
        try:
            conservation[chain] = conservation_profile(path)
        except (OSError, ValueError) as exc:
            errors.append(f"alignment {chain}: {exc}")
    return {
        "source": "opendde-native",
        "metric": metric,
        "minimize": minimize,
        "candidate_count": len(candidates),
        "current_parent_id": current_parent_id,
        "lineage_analysis": lineage_analysis,
        "recurrent_mutations": mutation_analysis(
            candidates,
            metric=metric,
            minimize=minimize,
        ),
        "trees": trees,
        "conservation": conservation,
        "warnings": errors,
    }


def analyze_search_history(
    candidates: list[dict[str, Any]],
    *,
    metric: str,
    minimize: bool,
    current_parent_id: str | None = None,
    tree_paths: dict[str, str] | None = None,
    alignment_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the complete OpenDDE Harness evolution analysis through one public entry point."""
    lineage = analyze_lineages(
        candidates,
        metric=metric,
        minimize=minimize,
    )
    return analyze_evolution(
        candidates,
        lineage_analysis=lineage,
        metric=metric,
        minimize=minimize,
        current_parent_id=current_parent_id,
        tree_paths=tree_paths,
        alignment_paths=alignment_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_json_path", required=True)
    parser.add_argument("--metric", default="objective")
    parser.add_argument("--maximize", action="store_true")
    parser.add_argument("--current-parent-id")
    parser.add_argument("--tree-path", action="append", default=[])
    parser.add_argument("--alignment-path", action="append", default=[])
    args = parser.parse_args()
    candidates = _load_json(args.candidates_json_path)
    alignments = dict(item.split("=", 1) for item in args.alignment_path)
    trees = dict(item.split("=", 1) for item in args.tree_path)
    result = analyze_search_history(
        candidates,
        metric=args.metric,
        minimize=not args.maximize,
        current_parent_id=args.current_parent_id,
        tree_paths=trees,
        alignment_paths=alignments,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
