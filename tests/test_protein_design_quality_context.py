"""QC projections preserve evidence without transmitting execution records."""

from copy import deepcopy

import pytest

from opendde_harness.plugin.protein_design.agents.prompt_context import context_json, quality_candidates


@pytest.mark.parametrize("minibinder", [False, True])
def test_quality_projection_preserves_evidence_and_input(minibinder):
    record = {
        "candidate_id": "b",
        "sequence": "ACDE",
        "objective": 0.4,
        "structure_path": "/unused/structure.cif",
        "metrics": {"iptm": 0.8, "missing": None, "cdr_contact_fraction": 0.5},
        "metadata": {
            "chains": {"B": "ACDE"},
            "gate_passed": False,
            "fold": {"success": False, "error": "missing output", "raw": "x" * 10000},
            "loss": {"configuration": "x" * 10000},
            "hotspot_gate": {
                "reason": "no_contact",
                "missed_hotspots": [["A", 3]],
                "num_off_target": 8,
                "cdr3_gate_passed": False,
            },
        },
    }
    records = [record, {**deepcopy(record), "candidate_id": "a"}]
    original = deepcopy(records)
    result = quality_candidates(records, minibinder=minibinder)
    assert records == original
    assert [r["candidate_id"] for r in result] == ["a", "b"]
    candidate = result[0]
    assert candidate["chains"] == {"B": "ACDE"}
    assert "sequence" not in candidate
    assert candidate["success"] is False and candidate["error"] == "missing output"
    assert candidate["metrics"]["missing"] is None
    assert candidate["gate_evidence"]["missed_hotspots"] == [["A", 3]]
    assert candidate["gate_evidence"]["num_off_target"] == 8
    assert ("cdr3_gate_passed" in candidate["gate_evidence"]) is not minibinder
    assert len(context_json(result)) < len(context_json(records)) / 10
    assert "structure_path" not in context_json(result)


def test_quality_projection_does_not_invent_success():
    result = quality_candidates([{"candidate_id": "unknown"}])[0]
    assert "success" not in result
    assert "gate_passed" not in result
