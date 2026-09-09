"""Canonical search-history records and bounded trajectory summaries."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any


def normalize_search_candidate(
    candidate: Mapping[str, Any],
    *,
    objective_key: str,
    minimize: bool,
    cycle: int | None = None,
    population_action: str | None = None,
) -> dict[str, Any]:
    metadata = dict(candidate.get("metadata") or {})
    metrics = {
        str(key): float(value)
        for key, value in (candidate.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    }
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("OpenDDE Harness search-history candidates require candidate_id")
    parent_id = str(candidate.get("parent_id") or metadata.get("parent_id") or "").strip() or None
    objective = candidate.get("objective")
    if objective is None:
        objective = metrics.get(objective_key)
    objective = (
        float(objective)
        if isinstance(objective, (int, float)) and not isinstance(objective, bool) and math.isfinite(float(objective))
        else None
    )
    inferred_cycle = cycle
    if inferred_cycle is None:
        value = metadata.get("cycle", candidate.get("cycle"))
        if isinstance(value, int):
            inferred_cycle = value
        else:
            match = re.match(r"c(\d+)", candidate_id)
            inferred_cycle = int(match.group(1)) if match else None
    action = str(
        population_action or metadata.get("population_action") or candidate.get("population_action") or "unknown"
    )
    gate_value = candidate.get(
        "gate_passed",
        metadata.get("gate_passed", metrics.get("gate_passed")),
    )
    chains = candidate.get("chains") or metadata.get("chains") or {}
    normalized_chains = (
        {str(chain): str(sequence) for chain, sequence in chains.items()} if isinstance(chains, Mapping) else {}
    )
    return {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "cycle": inferred_cycle,
        "objective": objective,
        "objective_key": objective_key,
        "minimize": bool(minimize),
        "metrics": metrics,
        "mutations": list(metadata.get("mutations") or candidate.get("mutations") or []),
        "skill_id": metadata.get("skill_id") or candidate.get("skill_id"),
        "gate_passed": bool(gate_value) if gate_value is not None else None,
        "gate_reason": candidate.get("gate_reason") or (metadata.get("gate_evidence") or {}).get("reason"),
        "population_action": action,
        "sequence": str(candidate.get("sequence") or ""),
        "chains": normalized_chains,
        "structure_path": str(candidate.get("structure_path") or "") or None,
    }


def improvement(child: float, parent: float, *, minimize: bool) -> float:
    return parent - child if minimize else child - parent


def _ensure_normalized(
    entry: Mapping[str, Any],
    *,
    objective_key: str,
    minimize: bool,
) -> Mapping[str, Any]:
    """Skip re-normalizing records this module already produced.

    The orchestrator stores normalized records and rebuilds the summary every
    cycle, so re-normalizing the whole history was the dominant per-cycle cost.
    """

    if entry.get("objective_key") == objective_key and entry.get("minimize") == bool(minimize):
        return entry
    return normalize_search_candidate(entry, objective_key=objective_key, minimize=minimize)


def build_search_trajectory_summary(
    entries: list[Mapping[str, Any]],
    *,
    objective_key: str,
    minimize: bool,
    max_cycles: int = 20,
) -> dict[str, Any]:
    normalized = [_ensure_normalized(entry, objective_key=objective_key, minimize=minimize) for entry in entries]
    normalized = [entry for entry in normalized if entry["candidate_id"]]
    by_id = {entry["candidate_id"]: entry for entry in normalized}
    scored = [entry for entry in normalized if entry["objective"] is not None]
    best = (
        min(scored, key=lambda entry: entry["objective"])
        if minimize and scored
        else max(scored, key=lambda entry: entry["objective"])
        if scored
        else None
    )
    actions = Counter(entry["population_action"] for entry in normalized)
    gate_reasons = Counter(
        str(entry["gate_reason"]) for entry in normalized if entry["gate_passed"] is False and entry["gate_reason"]
    )
    skills: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"candidate_count": 0, "improved_edges": 0, "mean_improvement": None}
    )
    skill_deltas: dict[str, list[float]] = defaultdict(list)
    improving_edges: list[dict[str, Any]] = []
    declining_edges: list[dict[str, Any]] = []
    for entry in normalized:
        skill = str(entry["skill_id"] or "unknown")
        skills[skill]["candidate_count"] += 1
        parent = by_id.get(entry["parent_id"] or "")
        if parent is None or entry["objective"] is None or parent["objective"] is None:
            continue
        delta = improvement(entry["objective"], parent["objective"], minimize=minimize)
        skill_deltas[skill].append(delta)
        edge = {
            "parent_id": parent["candidate_id"],
            "candidate_id": entry["candidate_id"],
            "cycle": entry["cycle"],
            "improvement": delta,
        }
        (improving_edges if delta > 0 else declining_edges).append(edge)
    for skill, deltas in skill_deltas.items():
        skills[skill]["improved_edges"] = sum(delta > 0 for delta in deltas)
        skills[skill]["mean_improvement"] = sum(deltas) / len(deltas)

    cycle_best: dict[int, dict[str, Any]] = {}
    for entry in scored:
        cycle = entry["cycle"]
        if cycle is None:
            continue
        current = cycle_best.get(cycle)
        if current is None or (
            entry["objective"] < current["objective"] if minimize else entry["objective"] > current["objective"]
        ):
            cycle_best[cycle] = entry
    recent_cycles = sorted(cycle_best)[-max_cycles:]
    return {
        "objective_key": objective_key,
        "minimize": minimize,
        "candidate_count": len(normalized),
        "scored_count": len(scored),
        "gate_passed_count": sum(entry["gate_passed"] is True for entry in normalized),
        "best": (
            {
                "candidate_id": best["candidate_id"],
                "cycle": best["cycle"],
                "objective": best["objective"],
            }
            if best
            else None
        ),
        "population_actions": dict(actions.most_common()),
        "recurring_gate_failures": dict(gate_reasons.most_common(10)),
        "skill_outcomes": dict(skills),
        "top_improving_edges": sorted(improving_edges, key=lambda edge: edge["improvement"], reverse=True)[:10],
        "top_declining_edges": sorted(declining_edges, key=lambda edge: edge["improvement"])[:10],
        "recent_cycle_best": [
            {
                "cycle": cycle,
                "candidate_id": cycle_best[cycle]["candidate_id"],
                "objective": cycle_best[cycle]["objective"],
            }
            for cycle in recent_cycles
        ],
    }
