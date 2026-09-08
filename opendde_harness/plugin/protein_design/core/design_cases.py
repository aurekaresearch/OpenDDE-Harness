"""Structured cross-run experience records for protein design."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from opendde_harness.plugin.protein_design.core.contracts import Candidate, WorkflowConfig

# Retrieval ranking weights for a design case.  Ordered so that beating the global
# best outranks a local improvement, and a gate failure costs more than the largest
# single gain, which keeps gate-failing cases below any gate-passing one.
QUALITY_SCORE_UNSCORED = 0.10
QUALITY_SCORE_BASE = 0.40
QUALITY_SCORE_IMPROVED = 0.15
QUALITY_SCORE_GLOBAL_BEST = 0.17
QUALITY_SCORE_GATE_PASSED = 0.08
QUALITY_SCORE_GATE_FAILED = -0.15
QUALITY_SCORE_RETAINED = 0.07
QUALITY_SCORE_NOT_RETAINED = -0.10
# Clamped away from 0 and 1 so no case is ever ranked as certain either way.
QUALITY_SCORE_MIN = 0.05
QUALITY_SCORE_MAX = 0.95
# One cycle's candidate batch; fewer supporting candidates than this is a weak sample.
QUALITY_HIGH_CONFIDENCE_EVIDENCE = 4


def build_design_case_v2(
    *,
    task_id: str,
    config: WorkflowConfig,
    cycle: int,
    triggers: list[str],
    candidates: list[Candidate],
    population: list[Candidate],
    population_size_before: int,
    population_actions: Mapping[str, str],
    admitted_ids: set[str],
    selected_parent: Mapping[str, Any] | None,
    previous_global_best: Candidate | None,
    best: Candidate | None,
    working_parent: Candidate | None,
    selected_skill_id: str | None,
    no_improvement_streak_before: int,
    no_improvement_streak: int,
) -> dict[str, Any]:
    scored = [candidate for candidate in candidates if _objective(candidate) is not None]
    parent_objective = _objective(selected_parent)
    previous_best_objective = _objective(previous_global_best)
    retained_ids = {
        candidate.candidate_id
        for candidate in candidates
        if population_actions.get(candidate.candidate_id) == "retained_constrained_elite"
    }
    gate_passed = [candidate for candidate in scored if _passes_gate(candidate)]
    retained = [
        candidate for candidate in scored if candidate.candidate_id in retained_ids
    ]
    cycle_admitted_ids = {
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id in admitted_ids
    }
    best_child = _best(retained or gate_passed or scored, minimize=config.minimize)
    rejection_summary = _rejection_summary(
        candidates,
        population_actions=population_actions,
        admitted_ids=admitted_ids,
    )
    component_deltas = _component_deltas(selected_parent, best_child)
    objective_delta_parent = _delta(_objective(best_child), parent_objective)
    objective_delta_global = _delta(_objective(best_child), previous_best_objective)
    child_improved_parent = _improved(
        _objective(best_child),
        parent_objective,
        minimize=config.minimize,
    )
    quality = _quality(
        candidates=candidates,
        scored=scored,
        gate_passed=gate_passed,
        retained_ids=retained_ids,
        parent_objective=parent_objective,
        minimize=config.minimize,
        global_best_improved="global_best_improvement" in triggers,
    )
    best_child_record = None
    if best_child is not None:
        best_child_record = {
            "candidate_id": best_child.candidate_id,
            "objective": _objective(best_child),
            "objective_delta_vs_parent": objective_delta_parent,
            "objective_delta_vs_global_best": objective_delta_global,
            "gate_passed": _passes_gate(best_child),
            "admitted": best_child.candidate_id in cycle_admitted_ids,
            "retained": best_child.candidate_id in retained_ids,
        }
    mutations = _mutations(best_child)
    design_scope = _design_scope(
        config,
        mutations,
        selected_skill_id=selected_skill_id,
    )
    recurring = config.metadata.get("recurring_offenders") or {}
    recurring_offenders = recurring_offender_ids(recurring)
    representative_ids = _representative_candidate_ids(
        candidates,
        best_child=best_child,
        retained_ids=retained_ids,
        minimize=config.minimize,
    )
    lesson = _lesson(
        selected_skill_id=selected_skill_id,
        mutations=mutations,
        best_child=best_child,
        parent_objective=parent_objective,
        objective_delta_parent=objective_delta_parent,
        retained_ids=retained_ids,
        triggers=triggers,
        verdict=quality["verdict"],
        rejection_summary=rejection_summary,
    )
    return {
        "memory_schema": "protein_design_case_v2",
        "identity": {
            "target": config.target,
            "task_id": task_id,
            "cycle": cycle,
            "triggers": list(triggers),
            "binder_type": _binder_type(config),
            "objective_key": config.objective_key,
            "minimize": config.minimize,
        },
        "state_before": {
            "selected_parent_id": _candidate_id(selected_parent),
            "parent_objective": parent_objective,
            "global_best_objective": previous_best_objective,
            "no_improvement_streak": no_improvement_streak_before,
            "population_size": population_size_before,
            "gate_state": _gate_state(selected_parent),
        },
        "action": {
            "primary_skill": selected_skill_id,
            "learned_skills": list(config.metadata.get("learned_skills_applied", [])),
            "mutations": mutations,
            "design_scope": design_scope,
        },
        "outcome": {
            "proposed_count": len(candidates),
            "scored_count": len(scored),
            "gate_passed_count": len(gate_passed),
            "admitted_count": len(cycle_admitted_ids),
            "retained_count": len(retained_ids),
            "best_child": best_child_record,
            "component_deltas": component_deltas,
            "rejection_summary": rejection_summary,
            "population_size": len(population),
            "global_best_id": best.candidate_id if best is not None else None,
            "global_best_objective": _objective(best),
            "no_improvement_streak": no_improvement_streak,
            "child_improved_parent": child_improved_parent,
        },
        "evidence": {
            "recurring_offenders": recurring_offenders,
            "structure_path": best_child.structure_path if best_child is not None else None,
            "parent_structure_path": _structure_path(selected_parent),
            "working_parent_id": (
                working_parent.candidate_id if working_parent is not None else None
            ),
            "representative_candidate_ids": representative_ids,
        },
        "lesson": lesson,
        "quality": quality,
    }


def _objective(candidate: Candidate | Mapping[str, Any] | None) -> float | None:
    if candidate is None:
        return None
    value = candidate.objective if isinstance(candidate, Candidate) else candidate.get("objective")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _candidate_id(candidate: Mapping[str, Any] | None) -> str | None:
    if candidate is None:
        return None
    value = str(candidate.get("candidate_id") or "").strip()
    return value or None


def _structure_path(candidate: Mapping[str, Any] | None) -> str | None:
    if candidate is None:
        return None
    value = str(candidate.get("structure_path") or "").strip()
    return value or None


def _metadata(candidate: Candidate | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if candidate is None:
        return {}
    value = candidate.metadata if isinstance(candidate, Candidate) else candidate.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _metrics(candidate: Candidate | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if candidate is None:
        return {}
    value = candidate.metrics if isinstance(candidate, Candidate) else candidate.get("metrics")
    return value if isinstance(value, Mapping) else {}


def _passes_gate(candidate: Candidate | Mapping[str, Any]) -> bool:
    metadata = _metadata(candidate)
    value = metadata.get("gate_passed", _metrics(candidate).get("gate_passed"))
    return bool(value)


def _gate_evidence(
    candidate: Candidate | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    evidence = _metadata(candidate).get("gate_evidence")
    return evidence if isinstance(evidence, Mapping) else {}


def _gate_state(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = _metadata(candidate)
    metrics = _metrics(candidate)
    evidence = _gate_evidence(candidate)
    has_gate_result = "gate_passed" in metadata or "gate_passed" in metrics
    passed = _passes_gate(candidate or {}) if has_gate_result else None
    reason = evidence.get("reason")
    reasons = evidence.get("rejection_reasons")
    rejection_reasons = (
        [str(item) for item in reasons]
        if isinstance(reasons, list)
        else [str(reason)] if reason and passed is False else []
    )
    fraction = evidence.get("cdr_contact_fraction")
    return {
        "cdr_contact_fraction": (
            float(fraction)
            if isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
            else None
        ),
        "passed": passed,
        "rejection_reasons": rejection_reasons,
    }


def _best(candidates: list[Candidate], *, minimize: bool) -> Candidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item.objective)) if minimize else max(
        candidates,
        key=lambda item: float(item.objective),
    )


def _delta(child: float | None, baseline: float | None) -> float | None:
    if child is None or baseline is None:
        return None
    return round(child - baseline, 8)


def _improved(child: float | None, parent: float | None, *, minimize: bool) -> bool | None:
    if child is None or parent is None:
        return None
    return child < parent if minimize else child > parent


def _loss_components(candidate: Candidate | Mapping[str, Any] | None) -> dict[str, float]:
    loss = _metadata(candidate).get("loss")
    if not isinstance(loss, Mapping):
        return {}
    result: dict[str, float] = {}
    components = loss.get("components")
    if isinstance(components, Mapping):
        for name, value in components.items():
            raw = value.get("raw") if isinstance(value, Mapping) else value
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
                result[str(name)] = float(raw)
    esm2 = loss.get("esm2_pll")
    if isinstance(esm2, (int, float)) and not isinstance(esm2, bool) and math.isfinite(float(esm2)):
        result["esm2"] = float(esm2)
    return result


def _component_deltas(
    parent: Mapping[str, Any] | None,
    child: Candidate | None,
) -> dict[str, float]:
    parent_components = _loss_components(parent)
    child_components = _loss_components(child)
    result = {}
    for name in sorted(set(parent_components) & set(child_components)):
        if name == "esm2":
            delta = child_components[name] - parent_components[name]
        else:
            delta = parent_components[name] - child_components[name]
        result[name] = round(delta, 8)
    if "i_ptm" not in result:
        parent_iptm = _metrics(parent).get("iptm")
        child_iptm = _metrics(child).get("iptm")
        if isinstance(parent_iptm, (int, float)) and isinstance(child_iptm, (int, float)):
            result["i_ptm"] = round(float(child_iptm) - float(parent_iptm), 8)
    return result


def _mutations(candidate: Candidate | None) -> list[dict[str, Any]]:
    if candidate is None:
        return []
    result = []
    for mutation in _metadata(candidate).get("mutations") or []:
        if not isinstance(mutation, Mapping):
            continue
        result.append(
            {
                "chain": mutation.get("chain") or mutation.get("chain_id"),
                "position": mutation.get("position"),
                "from": mutation.get("from") or mutation.get("from_aa"),
                "to": mutation.get("to") or mutation.get("to_aa"),
            }
        )
    return result


def _design_scope(
    config: WorkflowConfig,
    mutations: list[dict[str, Any]],
    *,
    selected_skill_id: str | None,
) -> list[str]:
    selected: list[tuple[str, int]] = []
    for mutation in mutations:
        chain = mutation.get("chain")
        position = mutation.get("position")
        if chain is not None and isinstance(position, int):
            selected.append((str(chain), position))
    selected_regions = {
        (chain, group_index)
        for chain, position in selected
        for group_index, group in enumerate(
            config.cdr_region_groups.get(chain, []),
            start=1,
        )
        if position in group
    }
    scopes = []
    multiple_chains = len(config.binder_chains) > 1
    for chain, groups in config.cdr_region_groups.items():
        for index, positions in enumerate(groups, start=1):
            include = (chain, index) in selected_regions
            if selected_skill_id == "cdr-full-redesign" and positions:
                include = True
            if include:
                scopes.append(f"{chain}:CDR{index}" if multiple_chains else f"CDR{index}")
    if scopes:
        return list(dict.fromkeys(scopes))
    if config.cdr_regions:
        return [str(chain) for chain, positions in config.cdr_regions.items() if positions]
    return []


def _rejection_summary(
    candidates: list[Candidate],
    *,
    population_actions: Mapping[str, str],
    admitted_ids: set[str],
) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for candidate in candidates:
        if _objective(candidate) is None:
            reason = _gate_evidence(candidate).get("reason")
            reasons[str(reason or "scoring_failed")] += 1
            continue
        if not _passes_gate(candidate):
            reason = _gate_evidence(candidate).get("reason")
            reasons[str(reason or "gate_failed")] += 1
            continue
        if candidate.candidate_id not in admitted_ids:
            reasons["quality_check_failed"] += 1
            continue
        action = population_actions.get(candidate.candidate_id)
        if action and action != "retained_constrained_elite":
            reasons[action] += 1
    return dict(sorted(reasons.items()))


def recurring_offender_ids(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []

    def count(item: tuple[Any, Any]) -> float:
        try:
            return float(item[1])
        except (TypeError, ValueError):
            return 0.0

    return [
        str(key)
        for key, _value in sorted(
            value.items(),
            key=lambda item: (-count(item), str(item[0])),
        )
    ]


def _representative_candidate_ids(
    candidates: list[Candidate],
    *,
    best_child: Candidate | None,
    retained_ids: set[str],
    minimize: bool,
) -> list[str]:
    representatives = []
    if best_child is not None:
        representatives.append(best_child.candidate_id)
    retained = [candidate for candidate in candidates if candidate.candidate_id in retained_ids]
    rejected = [candidate for candidate in candidates if candidate.candidate_id not in retained_ids]
    for group in (retained, rejected):
        scored = [candidate for candidate in group if _objective(candidate) is not None]
        representative = _best(scored, minimize=minimize)
        if representative is not None and representative.candidate_id not in representatives:
            representatives.append(representative.candidate_id)
    return representatives[:3]


def _quality(
    *,
    candidates: list[Candidate],
    scored: list[Candidate],
    gate_passed: list[Candidate],
    retained_ids: set[str],
    parent_objective: float | None,
    minimize: bool,
    global_best_improved: bool,
) -> dict[str, Any]:
    improved = [
        candidate
        for candidate in scored
        if _improved(_objective(candidate), parent_objective, minimize=minimize)
    ]
    supporting = [
        candidate
        for candidate in improved
        if candidate.candidate_id in retained_ids and _passes_gate(candidate)
    ]
    if not scored or not gate_passed:
        verdict = "failure"
    elif supporting or global_best_improved:
        verdict = "success"
    else:
        verdict = "mixed"
    if not scored:
        score = QUALITY_SCORE_UNSCORED
    else:
        score = QUALITY_SCORE_BASE
        score += QUALITY_SCORE_IMPROVED if improved else 0.0
        score += QUALITY_SCORE_GLOBAL_BEST if global_best_improved else 0.0
        score += QUALITY_SCORE_GATE_PASSED if gate_passed else QUALITY_SCORE_GATE_FAILED
        score += QUALITY_SCORE_RETAINED if retained_ids else QUALITY_SCORE_NOT_RETAINED
        score = min(QUALITY_SCORE_MAX, max(QUALITY_SCORE_MIN, score))
    if verdict == "success":
        evidence_count = len(supporting) or 1
    elif verdict == "failure":
        evidence_count = len(candidates) - len(gate_passed)
    else:
        evidence_count = len(scored)
    confidence = "high" if evidence_count >= QUALITY_HIGH_CONFIDENCE_EVIDENCE else "medium" if evidence_count else "low"
    return {
        "score": round(score, 4),
        "confidence": confidence,
        "evidence_count": evidence_count,
        "verdict": verdict,
    }


def _lesson(
    *,
    selected_skill_id: str | None,
    mutations: list[dict[str, Any]],
    best_child: Candidate | None,
    parent_objective: float | None,
    objective_delta_parent: float | None,
    retained_ids: set[str],
    triggers: list[str],
    verdict: str,
    rejection_summary: Mapping[str, int],
) -> dict[str, Any]:
    skill = selected_skill_id or "no primary skill"
    if best_child is None:
        summary = f"{skill} produced no scored child candidate."
    else:
        mutation_text = ", ".join(
            f"{item.get('chain')}{item.get('position')} {item.get('from')}>{item.get('to')}"
            for item in mutations
        ) or "the proposed CDR design"
        summary = (
            f"{skill} applied {mutation_text}; candidate {best_child.candidate_id} "
            f"changed the objective from {parent_objective} to {_objective(best_child)} "
            f"(delta {objective_delta_parent}), gate "
            f"{'passed' if _passes_gate(best_child) else 'failed'}, and was "
            f"{'retained' if best_child.candidate_id in retained_ids else 'not retained'}."
        )
    reusable_when = []
    if "stagnation_redesign" in triggers:
        reusable_when.append("the search is stagnant and needs a basin-changing redesign")
    if selected_skill_id == "cdr-point-mutation":
        reusable_when.append("local CDR refinement is appropriate for a comparable parent")
    if "mixed_gate_outcome" in triggers:
        reusable_when.append("similar proposals show mixed structural-gate outcomes")
    avoid_when = list(rejection_summary) if verdict != "success" else []
    return {
        "verdict": verdict,
        "summary": summary,
        "reusable_when": reusable_when,
        "avoid_when": avoid_when,
    }


def _binder_type(config: WorkflowConfig) -> str:
    configured = str(config.metadata.get("binder_type") or "").strip()
    if configured:
        return configured
    chain_count = len(config.binder_chains)
    if chain_count == 1:
        return "VHH"
    if chain_count == 2:
        return "VH/VL"
    return "multichain"
