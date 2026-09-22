"""Lossless records stay on disk; these projections are only for LLM context."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def bounded_design_memories(memories: list[str]) -> list[str]:
    """Bound recalled advisory excerpts without modifying stored memories."""
    result: list[str] = []
    seen: set[str] = set()
    remaining = 2000
    marker = " ... [excerpt truncated]"
    for memory in memories:
        text = memory.strip()
        key = " ".join(text.split())
        if not key or key in seen:
            continue
        seen.add(key)
        budget = min(600, remaining - bool(result))
        if budget <= len(marker) or len(result) == 4:
            break
        if len(text) > budget:
            text = text[: budget - len(marker)].rstrip() + marker
        remaining -= len(text) + bool(result)
        result.append(text)
    return result


def minibinder_prompt(config, **evidence) -> str:
    analysis = evidence.pop("analysis", None)
    if isinstance(analysis, dict):
        analysis = analysis.get("downstream_header") or analysis
    if "memories" in evidence:
        evidence["memories"] = bounded_design_memories(evidence["memories"])
    # The available skill catalog already lists these names once.
    evidence.pop("learned_skills", None)
    if "parent" in evidence and isinstance(evidence["parent"], dict):
        original_parent = evidence["parent"]
        parent = compact_candidate(original_parent, minibinder=True)
        if evidence.get("structure_path"):
            parent["structure_path"] = evidence.pop("structure_path")
        evidence["parent"] = parent
        if "population" in evidence:
            population = design_population(evidence.pop("population"), original_parent, {}, minibinder=True)
            evidence["other_population_candidates"] = [item for item in population if "sequence_ref" not in item]
    if "gate_feedback" in evidence:
        evidence["gate_feedback"] = compact_gate_feedback(evidence["gate_feedback"], minibinder=True)
    static = {
        "design_type": "minibinder",
        "target": config.target,
        "target_chains": config.target_chains,
        "hotspots": config.fold_options.get("target_hotspots", config.hotspots),
        "binder_chains": config.binder_chains,
        "mutable_positions": config.mutable_positions,
        "fixed_residues": config.fixed_residues,
        "objective_key": config.objective_key,
        "minimize": config.minimize,
    }
    if analysis is not None:
        static["analysis"] = analysis
    # Run controls usually remain unchanged across many parent selections.
    # Always use their current values: schedules may legitimately change them.
    for key in ("candidate_count", "mutation_budget"):
        if key in evidence:
            static[key] = evidence.pop(key)
    # Explicit ordering must not depend on the caller's keyword insertion order.
    # Slowly refreshed context precedes per-candidate evidence; cycle is last.
    dynamic_order = (
        "route",
        "memories",
        "reflection",
        "parent",
        "other_population_candidates",
        "candidates",
        "developability",
        "gate_feedback",
    )
    ordered = {key: evidence[key] for key in dynamic_order if key in evidence}
    ordered.update({key: evidence[key] for key in sorted(evidence) if key not in ordered and key != "cycle"})
    if "cycle" in evidence:
        ordered["cycle"] = evidence["cycle"]
    # Preserve the outer static-before-dynamic order; canonicalize nested maps.
    return json.dumps(
        {key: json.loads(context_json(value)) for key, value in {**static, **ordered}.items()},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _antibody_field(key: str) -> bool:
    return key.startswith(("cdr", "framework", "n_contact_cdr", "n_contact_antibody"))


def compact_gate(evidence: Mapping[str, Any], *, minibinder: bool = False) -> dict[str, Any]:
    fields = (
        "reason",
        "error",
        "gate_passed",
        "contact_map_method",
        "distance_cutoff_a",
        "coverage_ratio",
        "contacted_hotspots",
        "missed_hotspots",
        "num_contacted_hotspots",
        "num_missed_hotspots",
        "total_contacts",
        "total_binder_contacts",
        "binder_interface_residues",
        "epitope_gate_skipped",
        "epitope_gate_skip_reason",
        "hotspot_min_distances_a",
        "cdr_position_semantics",
        "cdr3_gate_passed",
        "cdr_contact_fraction_gate_passed",
        "cdr_contact_fraction",
        "cdr_contact_fraction_threshold",
        "framework_contact_fraction",
        "cdr3_epitope_contacts",
        "cdr3_total_target_contacts",
    )
    result = {key: evidence[key] for key in fields if key in evidence and not (minibinder and _antibody_field(key))}
    # Keep exact hotspot membership above. Bound illustrative contacts only, with
    # explicit counts so an omitted residue is never interpreted as no contact.
    for key in ("contacted_binder_residues", "contact_pairs", "framework_contact_residue_ids"):
        if minibinder and _antibody_field(key):
            continue
        value = evidence.get(key)
        if isinstance(value, (list, tuple)):
            result[key] = list(value[:24])
            result[f"{key}_count"] = len(value)
            if len(value) > 24:
                result[f"{key}_omitted"] = len(value) - 24
    return result


def compact_candidate(
    candidate: Mapping[str, Any], *, minibinder: bool = False, sequence: bool = True
) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    result = {
        key: candidate[key] for key in ("candidate_id", "id", "name", "objective", "structure_path") if key in candidate
    }
    metrics = candidate.get("metrics") or {}
    result["metrics"] = {
        key: value
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float, bool)) or value is None
        if not (minibinder and _antibody_field(key))
    }
    if sequence:
        chains = candidate.get("chains") or metadata.get("chains")
        if chains:
            result["chains"] = chains
        elif "sequence" in candidate:
            result["sequence"] = candidate["sequence"]
    for key in ("gate_passed", "success", "error", "skill_id", "mutations", "hypothesis", "strategy"):
        if key in metadata:
            result[key] = metadata[key]
        elif key in candidate:
            result[key] = candidate[key]
    result["gate_evidence"] = compact_gate(metadata.get("gate_evidence") or {}, minibinder=minibinder)
    return result


def _gate_feedback_evidence(evidence: Mapping[str, Any], *, minibinder: bool) -> dict[str, Any]:
    """Decision-level gate feedback, never atom contacts or structural records."""
    fields = (
        "gate_passed",
        "reason",
        "error",
        "coverage_ratio",
        "num_contacted_hotspots",
        "num_missed_hotspots",
        "num_off_target",
        "total_binder_contacts",
        "epitope_gate_skipped",
        "epitope_gate_skip_reason",
        "cdr3_gate_passed",
        "cdr_contact_fraction_gate_passed",
        "cdr_contact_fraction",
        "cdr_contact_fraction_threshold",
        "framework_contact_fraction",
    )
    result = {
        key: evidence[key]
        for key in fields
        if evidence.get(key) is not None and not (minibinder and _antibody_field(key))
    }
    for field, count in (("contacted_hotspots", "num_contacted_hotspots"), ("missed_hotspots", "num_missed_hotspots")):
        positions = evidence.get(field)
        if isinstance(positions, (list, tuple)):
            result.setdefault(count, len(positions))
            if field == "missed_hotspots" and positions:
                result[field] = list(positions[:6])
                if len(positions) > 6:
                    result["missed_hotspots_omitted"] = len(positions) - 6
                if evidence.get("hotspot_position_semantics"):
                    result["hotspot_position_semantics"] = evidence["hotspot_position_semantics"]
    return result


def compact_gate_feedback(value: Any, *, minibinder: bool = False) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    if not isinstance(value, Mapping):
        return value
    # Unknown feedback formats are retained rather than silently discarding evidence.
    if "candidate_gate_evidence" not in value:
        return value
    result = {key: value[key] for key in ("objective_key", "minimize", "population_size") if key in value}
    candidates = {}
    working = value.get("working_parent") or {}
    for field, candidate in (
        ("working_parent", working.get("candidate")),
        ("population_best", value.get("population_best")),
    ):
        if not candidate:
            result[field] = None
            continue
        identity = candidate.get("candidate_id") or candidate.get("id") or field
        result[field] = identity
        metadata = candidate.get("metadata") or {}
        evidence = metadata.get("gate_evidence") or metadata.get("hotspot_gate") or {}
        gate = _gate_feedback_evidence(evidence, minibinder=minibinder)
        passed = metadata.get("gate_passed", (candidate.get("metrics") or {}).get("gate_passed"))
        if passed is not None:
            gate.setdefault("gate_passed", passed)
        candidates[identity] = {"objective": candidate.get("objective"), "gate_evidence": gate}
    for identity, change in value.get("candidate_changes", {}).items():
        candidates.setdefault(identity, {}).update(
            {key: change[key] for key in ("objective", "delta", "status") if key in change}
        )
    for identity, evidence in value.get("candidate_gate_evidence", {}).items():
        candidates.setdefault(identity, {})["gate_evidence"] = _gate_feedback_evidence(
            evidence or {}, minibinder=minibinder
        )
    result["candidates"] = candidates
    if not minibinder:
        recurring = value.get("recurring_offenders") or {}
        ranked = sorted(recurring.items(), key=lambda item: (-item[1], item[0]))
        result["recurring_offenders"] = dict(ranked[:5])
        if len(ranked) > 5:
            result["recurring_offenders_omitted"] = len(ranked) - 5
        if ranked:
            result["recurring_offender_positions"] = (
                "chain:zero_based_sequence_position; fixed residues remain immutable"
            )
    return result


def context_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def reflection_candidates(candidates: list[dict[str, Any]], *, minibinder: bool = False) -> list[dict[str, Any]]:
    """Keep comparative evidence and tool paths, not nested backend records."""
    result = []
    for original in sorted(candidates, key=lambda item: item.get("candidate_id") or item.get("id") or ""):
        summary = compact_candidate(original, minibinder=minibinder)
        metadata = original.get("metadata") or {}
        for key in ("hypothesis", "strategy"):
            summary.pop(key, None)
        for key in ("parent_id", "cycle"):
            if key in metadata or key in original:
                summary[key] = metadata[key] if key in metadata else original[key]
        fold = metadata.get("fold") or {}
        for key in ("success", "error"):
            if key not in summary and key in fold:
                summary[key] = fold[key]
        gate = metadata.get("gate_evidence") or metadata.get("hotspot_gate") or {}
        summary["gate_evidence"] = compact_gate(gate, minibinder=minibinder)
        for key in ("hotspot_position_semantics", "num_off_target"):
            if key in gate:
                summary["gate_evidence"][key] = gate[key]
        loss = metadata.get("loss") or {}
        if isinstance(loss, Mapping) and isinstance(loss.get("loss_components"), Mapping):
            summary["loss_components"] = {
                key: value
                for key, value in loss["loss_components"].items()
                if isinstance(value, (int, float, bool)) or value is None
            }
        result.append(summary)
    return result


def design_population(
    candidates: list[dict[str, Any]], parent: dict[str, Any], feedback: Any, *, minibinder: bool = False
) -> list[dict[str, Any]]:
    """Comparison rows, not structural reports; full evidence remains on disk.

    Keep exact scores and non-parent sequences. Gate summaries retain outcomes,
    coverage and failure reasons, not atom contacts or per-residue distances.
    """
    result = reflection_candidates(candidates, minibinder=minibinder)
    parent_id = parent.get("candidate_id") or parent.get("id")
    gates = feedback.get("candidates", {}) if isinstance(feedback, Mapping) else {}
    for summary in result:
        identity = summary.get("candidate_id") or summary.get("id")
        if parent_id is not None and identity == parent_id:
            summary.pop("chains", None)
            summary.pop("sequence", None)
            summary["sequence_ref"] = "PARENT sequence context"
        existing = gates.get(identity, {})
        if summary.get("gate_evidence") and summary["gate_evidence"] == existing.get("gate_evidence"):
            summary.pop("gate_evidence")
            summary["gate_evidence_ref"] = {"section": "gate_feedback", "candidate_id": identity}
        elif "gate_evidence" in summary:
            evidence = summary["gate_evidence"]
            fields = (
                "reason",
                "error",
                "gate_passed",
                "coverage_ratio",
                "num_contacted_hotspots",
                "num_missed_hotspots",
                "num_off_target",
                "epitope_gate_skipped",
                "epitope_gate_skip_reason",
                "cdr3_gate_passed",
                "cdr_contact_fraction_gate_passed",
                "cdr_contact_fraction",
                "cdr_contact_fraction_threshold",
                "framework_contact_fraction",
            )
            summary["gate_evidence"] = {key: evidence[key] for key in fields if evidence.get(key) is not None}
            # Older records may have only membership lists. Count those without
            # interpreting absent evidence as zero measured contacts.
            for field, count in (
                ("contacted_hotspots", "num_contacted_hotspots"),
                ("missed_hotspots", "num_missed_hotspots"),
            ):
                if count not in summary["gate_evidence"] and isinstance(evidence.get(field), (list, tuple)):
                    summary["gate_evidence"][count] = len(evidence[field])
        for key in ("structure_path", "loss_components"):
            summary.pop(key, None)
        mutations = summary.pop("mutations", None)
        if mutations and not any(key in summary for key in ("chains", "sequence", "sequence_ref")):
            # Full redesign records include unchanged assignments, so do not
            # label this list as an observed sequence difference.
            summary["position_assignments"] = [
                [item["chain_id"], item["position"], item["to_aa"]]
                if isinstance(item, Mapping) and {"chain_id", "position", "to_aa"} <= item.keys()
                else item
                for item in mutations
            ]
        metrics = summary["metrics"]
        evidence = summary.get("gate_evidence") or (
            existing.get("gate_evidence", {}) if "gate_evidence_ref" in summary else {}
        )
        for key in list(metrics):
            if (key in evidence and metrics[key] == evidence[key]) or (key in summary and metrics[key] == summary[key]):
                metrics.pop(key)
        if summary.get("objective") is not None and metrics.get("loss") == summary["objective"]:
            metrics.pop("loss")
        for key in ("error", "success"):
            if summary.get(key) is None or (key == "success" and summary.get(key) is True):
                summary.pop(key, None)
        for key in ("gate_evidence", "metrics"):
            if not summary.get(key):
                summary.pop(key, None)
    return result


def quality_candidates(candidates: list[dict[str, Any]], *, minibinder: bool = False) -> list[dict[str, Any]]:
    """QC needs measured evidence, not execution records or proposal narratives."""
    result = []
    for candidate in sorted(candidates, key=lambda item: item["candidate_id"]):
        summary = compact_candidate(candidate, minibinder=minibinder)
        for key in ("structure_path", "skill_id", "mutations", "hypothesis", "strategy"):
            summary.pop(key, None)
        metadata = candidate.get("metadata") or {}
        fold = metadata.get("fold") or {}
        for key in ("success", "error"):
            if key not in summary and key in fold:
                summary[key] = fold[key]
        # Older records may expose the same gate under hotspot_gate only.
        gate = metadata.get("gate_evidence") or metadata.get("hotspot_gate") or {}
        summary["gate_evidence"] = compact_gate(gate, minibinder=minibinder)
        # Preserve interpretation of residue evidence and off-target concerns.
        for key in ("hotspot_position_semantics", "num_off_target"):
            if key in gate:
                summary["gate_evidence"][key] = gate[key]
        result.append(summary)
    return result
