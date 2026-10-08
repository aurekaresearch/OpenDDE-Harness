"""Deterministic context reduction, not a replacement for agent selection."""

from __future__ import annotations

import math
from collections import deque

from opendde_harness.plugin.protein_design.core.contracts import Candidate


def shortlist_candidates(
    candidates: list[Candidate], *, top_k: int, minimize: bool, design_type: str = "antibody"
) -> list[Candidate]:
    """Round-robin metric leaders, preferring distinct sequences before filling.

    No weighted score is computed. Missing measurements do not become zero.
    The ID-ordered queue also gives candidates with sparse evidence a route in.
    At most max(80, top_k) records enter the model context; final order is the
    agent's decision, not the order of this shortlist.
    """
    if not candidates:
        raise ValueError("PostFilter requires at least one candidate")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("PostFilter candidate IDs must be unique")
    if top_k < 1:
        raise ValueError("PostFilter top_k must be positive")
    ordered = sorted(candidates, key=lambda item: item.candidate_id)
    limit = max(80, top_k)
    if len(ordered) <= limit:
        return ordered

    metrics = [
        ("iptm", False),
        ("ipsae", False),
        ("plddt", False),
        ("ptm", False),
        ("binder_rmsd", True),
        ("loglikelihood", False),
    ]
    if design_type != "minibinder":
        metrics.extend([("cdr_contact_fraction", False), ("framework_contact_fraction", True)])
    metrics.append(("objective", minimize))
    queues = []
    for name, lower in metrics:
        values = [(item, item.objective if name == "objective" else item.metrics.get(name)) for item in ordered]
        finite = [
            (item, float(value))
            for item, value in values
            if value is not None and not isinstance(value, bool) and math.isfinite(value)
        ]
        queues.append(
            [
                item
                for item, _ in sorted(finite, key=lambda pair: (pair[1] if lower else -pair[1], pair[0].candidate_id))
            ]
        )
    queues.append(ordered)
    selected: list[Candidate] = []
    seen: set[str] = set()
    for prefer_diversity in (True, False):
        pending = [deque(queue) for queue in queues]
        while any(pending) and len(selected) < limit:
            for queue in pending:
                while queue:
                    item = queue.popleft()
                    if item.candidate_id in seen:
                        continue
                    if prefer_diversity and any(_similar_sequence(item.sequence, other.sequence) for other in selected):
                        continue
                    selected.append(item)
                    seen.add(item.candidate_id)
                    break
                if len(selected) == limit:
                    break
    return selected


def _similar_sequence(left: str, right: str) -> bool:
    """Cheap, explicit shortlist heuristic; not an alignment or affinity model."""
    if not left or not right:
        return False
    return sum(a == b for a, b in zip(left, right)) / max(len(left), len(right)) >= 0.9


def shortlist_audit(candidates: list[Candidate], shortlist: list[Candidate]) -> dict:
    return {
        "method": "metric_round_robin_sequence_diversity",
        "eligible_count": len(candidates),
        "candidate_ids": [item.candidate_id for item in shortlist],
        "sequence_identity_threshold": 0.9,
        "description": "Metric leaders and ID-ordered coverage; prefer sequence diversity, then fill. "
        "Missing metrics remain unknown. Shortlist order is not a final ranking.",
    }
