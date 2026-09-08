"""Tracing payloads for protein-design runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from opendde_harness.plugin.protein_design.core.contracts import Candidate, TaskSnapshot, WorkflowConfig
from opendde_harness.tracing import spans as _spans

_CYCLE_SCHEMA = "protein_design.cycle.v1"
_MAX_IDENTIFIER_LENGTH = 256
_MAX_SEQUENCE_LENGTH = 4096
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_DEPTH = 4
_SENSITIVE_KEY_PARTS = ("api_key", "authorization", "password", "prompt", "secret", "token")


def attach_artifact(
    span: Any,
    key: str,
    payload: Any,
    *,
    label: str | None = None,
) -> str | None:
    try:
        artifact = _spans.persist_artifact(
            key,
            {"traceId": span.trace_id},
            payload,
            label=label or key,
        )
        span.set(_spans.artifact_attributes(key, artifact))
        path = artifact.get("path") if artifact else None
        return str(path) if path else None
    except Exception:
        return None


def build_cycle_artifact(
    *,
    task_id: str,
    target: str,
    cycle: int,
    objective_key: str,
    minimize: bool,
    candidates: list[Candidate],
    cycle_best_id: str | None,
    global_best_id: str | None,
    known_candidate_ids: set[str],
    structure_artifacts: Mapping[str, str],
    population_candidate_ids: set[str] | None = None,
    admitted_candidate_ids: set[str] | None = None,
    population_actions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": _CYCLE_SCHEMA,
        "task_id": _bounded_text(task_id, _MAX_IDENTIFIER_LENGTH),
        "target": _bounded_text(target, _MAX_IDENTIFIER_LENGTH),
        "cycle": int(cycle),
        "objective_key": _bounded_text(objective_key, _MAX_IDENTIFIER_LENGTH),
        "minimize": bool(minimize),
        "candidates": [
            _candidate_payload(candidate, known_candidate_ids, structure_artifacts) for candidate in candidates
        ],
        "cycle_best_candidate_id": _optional_identifier(cycle_best_id),
        "global_best_candidate_id": _optional_identifier(global_best_id),
    }
    if population_candidate_ids is not None:
        payload["population_candidate_ids"] = sorted(
            _bounded_text(value, _MAX_IDENTIFIER_LENGTH)
            for value in population_candidate_ids
        )
    if admitted_candidate_ids is not None:
        payload["admitted_candidate_ids"] = sorted(
            _bounded_text(value, _MAX_IDENTIFIER_LENGTH)
            for value in admitted_candidate_ids
        )
    if population_actions is not None:
        payload["population_actions"] = {
            _bounded_text(str(candidate_id), _MAX_IDENTIFIER_LENGTH): _bounded_text(
                str(action), 128
            )
            for candidate_id, action in population_actions.items()
        }
    return payload


def run_span_attributes(
    *, snapshot: TaskSnapshot, config: WorkflowConfig, phase: str | None = None
) -> dict[str, object]:
    best = snapshot.best_candidate
    return {
        "protein_design.task_id": snapshot.task_id,
        "protein_design.target": snapshot.target,
        "protein_design.compute_url": config.compute_url,
        "protein_design.compute_worker_id": config.compute_worker_id,
        "protein_design.status": snapshot.status.value,
        "protein_design.phase": phase or "",
        "protein_design.cycle": snapshot.cycle,
        "protein_design.total_cycles": snapshot.total_cycles,
        "protein_design.objective_key": config.objective_key,
        "protein_design.minimize": config.minimize,
        "protein_design.best_candidate_id": best.candidate_id if best else None,
        "protein_design.best_objective": _finite_number(best.objective) if best else None,
        "protein_design.error": snapshot.error,
    }


def cycle_span_attributes(
    *,
    task_id: str,
    cycle: int,
    candidates: list[Candidate],
    cycle_best: Candidate | None,
    global_best: Candidate | None,
) -> dict[str, object]:
    return {
        "protein_design.task_id": task_id,
        "protein_design.cycle_index": cycle,
        "protein_design.candidate_count": len(candidates),
        "protein_design.cycle_best_candidate_id": cycle_best.candidate_id if cycle_best else None,
        "protein_design.cycle_best_objective": _finite_number(cycle_best.objective) if cycle_best else None,
        "protein_design.global_best_candidate_id": global_best.candidate_id if global_best else None,
        "protein_design.global_best_objective": _finite_number(global_best.objective) if global_best else None,
    }


def _candidate_payload(
    candidate: Candidate,
    known_candidate_ids: set[str],
    structure_artifacts: Mapping[str, str],
) -> dict[str, object]:
    metadata = candidate.metadata
    raw_parent = metadata.get("parent_id")
    parent_id = str(raw_parent) if raw_parent is not None else None
    if parent_id not in known_candidate_ids:
        parent_id = None
    raw_skill = metadata.get("skill_id")
    skill_id = _optional_identifier(str(raw_skill)) if raw_skill is not None else None
    status = metadata.get("status")
    if status is None:
        status = "failed" if metadata.get("success") is False else "scored"
    return {
        "candidate_id": _bounded_text(candidate.candidate_id, _MAX_IDENTIFIER_LENGTH),
        "sequence": _bounded_text(candidate.sequence, _MAX_SEQUENCE_LENGTH),
        "objective": _finite_number(candidate.objective),
        "metrics": {
            str(key): number
            for key, value in candidate.metrics.items()
            if (number := _finite_number(value)) is not None
        },
        "parent_id": parent_id,
        "skill_id": skill_id,
        "status": _bounded_text(str(status), 64),
        "structure_artifact_path": structure_artifacts.get(candidate.candidate_id),
        "metadata": _safe_metadata(metadata),
    }


def _safe_metadata(metadata: Mapping[str, Any]) -> dict[str, object]:
    excluded = {"parent_id", "skill_id", "status"}
    result: dict[str, object] = {}
    for key, value in metadata.items():
        normalized_key = str(key)
        if normalized_key in excluded or _is_sensitive_key(normalized_key):
            continue
        safe_value = _safe_metadata_value(value, depth=1)
        if safe_value is not None:
            result[_bounded_text(normalized_key, _MAX_IDENTIFIER_LENGTH)] = safe_value
        if len(result) >= _MAX_METADATA_ITEMS:
            break
    return result


def _safe_metadata_value(value: Any, *, depth: int) -> object | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value)
    if isinstance(value, str):
        return _bounded_text(value, 512)
    if isinstance(value, (list, tuple)):
        if depth >= _MAX_METADATA_DEPTH:
            return None
        result = [
            safe
            for item in value[:_MAX_METADATA_ITEMS]
            if (safe := _safe_metadata_value(item, depth=depth + 1)) is not None
        ]
        return result
    if not isinstance(value, Mapping) or depth >= _MAX_METADATA_DEPTH:
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        normalized_key = str(key)
        if _is_sensitive_key(normalized_key):
            continue
        safe_item = _safe_metadata_value(item, depth=depth + 1)
        if safe_item is not None:
            result[_bounded_text(normalized_key, _MAX_IDENTIFIER_LENGTH)] = safe_item
        if len(result) >= _MAX_METADATA_ITEMS:
            break
    return result or None


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _optional_identifier(value: str | None) -> str | None:
    return _bounded_text(value, _MAX_IDENTIFIER_LENGTH) if value else None


def _bounded_text(value: str, limit: int) -> str:
    return str(value)[:limit]
