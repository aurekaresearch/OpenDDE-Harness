"""Design and reflection projections retain evidence without backend duplication."""

from copy import deepcopy

from opendde_harness.plugin.protein_design.agents.prompt_context import (
    compact_gate_feedback,
    context_json,
    design_population,
    reflection_candidates,
)


def test_comparison_projection_retains_lineage_failure_and_measurements():
    record = {
        "candidate_id": "child",
        "sequence": "ACDE",
        "objective": 2.0,
        "structure_path": "/real/child.cif",
        "metrics": {"iptm": 0.8, "missing": None},
        "metadata": {
            "parent_id": "parent",
            "mutations": [["B", 1, "C"]],
            "fold": {"success": False, "error": "partial output", "raw": "x" * 10000},
            "loss": {"loss_components": {"contact": 0.3}, "configuration": "x" * 10000},
            "gate_evidence": {
                "reason": "failed",
                "cdr3_gate_passed": False,
                "missed_hotspots": [["A", 5]],
                "contact_pairs": list(range(30)),
            },
        },
    }
    before = deepcopy(record)
    result = reflection_candidates([record])[0]
    assert result["parent_id"] == "parent"
    assert result["mutations"] == [["B", 1, "C"]]
    assert result["metrics"]["missing"] is None
    assert result["success"] is False and result["error"] == "partial output"
    assert result["structure_path"] == record["structure_path"]
    assert result["loss_components"] == {"contact": 0.3}
    assert result["gate_evidence"]["cdr3_gate_passed"] is False
    assert result["gate_evidence"]["contact_pairs_omitted"] == 6
    assert len(context_json(result)) < len(context_json(record)) / 10
    assert record == before
    mini = reflection_candidates([record], minibinder=True)[0]
    assert "cdr3_gate_passed" not in mini["gate_evidence"]
    assert mini["gate_evidence"]["missed_hotspots"] == [["A", 5]]


def test_design_references_only_matching_parent_and_gate():
    gate = {"reason": "failed", "cdr3_gate_passed": False}
    parent = {"candidate_id": "p", "sequence": "ACDE", "metadata": {"gate_evidence": gate}}
    other = {"candidate_id": "other", "sequence": "ACDF"}
    feedback = compact_gate_feedback({"candidate_gate_evidence": {"p": gate}})
    original = deepcopy([parent, other, feedback])
    result = {x["candidate_id"]: x for x in design_population([parent, other], parent, feedback)}
    assert result["p"]["sequence_ref"] == "PARENT sequence context"
    assert "sequence" not in result["p"]
    assert result["p"]["gate_evidence_ref"]["candidate_id"] == "p"
    assert result["other"]["sequence"] == "ACDF"
    assert [parent, other, feedback] == original
    mismatch = {"candidates": {"p": {"gate_evidence": {"reason": "different"}}}}
    assert "gate_evidence" in design_population([parent], parent, mismatch)[0]


def test_population_is_small_without_losing_failure_or_comparison_scores():
    parent = {
        "candidate_id": "p",
        "sequence": "ACDE",
        "objective": 1.9262415969,
        "structure_path": "/long/structure.cif",
        "metrics": {"loss": 1.9262415969, "iptm": 0.2685, "cdr_contact_fraction": 0.6, "gate_passed": 1.0},
        "metadata": {
            "gate_passed": True,
            "success": True,
            "error": None,
            "mutations": [{"chain_id": "B", "position": i, "to_aa": "A", "from_aa": None} for i in range(39)],
            "loss": {"loss_components": {"contact": 0.123}},
            "gate_evidence": {
                "reason": "contacted_hotspot",
                "cdr_contact_fraction": 0.6,
                "cdr_contact_fraction_threshold": 0.5,
                "cdr3_gate_passed": True,
                "contacted_hotspots": [["A", 1]],
                "missed_hotspots": [["A", 2], ["A", 3]],
                "coverage_ratio": 1 / 3,
                "contact_pairs": ["B:108-A:1@3.78A"] * 100,
                "hotspot_min_distances_a": {"A:1": 3.78},
                "framework_contact_residue_ids": [{"residue_id": i, "distance_to_antigen_a": 3.1} for i in range(20)],
            },
        },
    }
    before = deepcopy(parent)
    result = design_population([parent], parent, {})[0]
    assert parent == before
    assert result["objective"] == parent["objective"]
    assert result["metrics"] == {"iptm": 0.2685}
    assert result["gate_evidence"]["num_contacted_hotspots"] == 1
    assert result["gate_evidence"]["num_missed_hotspots"] == 2
    assert result["gate_evidence"]["cdr_contact_fraction_threshold"] == 0.5
    assert result["sequence_ref"] == "PARENT sequence context"
    for key in ("mutations", "structure_path", "loss_components", "success", "error"):
        assert key not in result
    assert "contact_pairs" not in context_json(result)
    assert len(context_json(result)) < len(context_json(reflection_candidates([parent]))) / 4
    failed = {
        "candidate_id": "failed",
        "metrics": {"iptm": None},
        "metadata": {
            "fold": {"success": False, "error": "fold failed"},
            "gate_evidence": {
                "cdr3_gate_passed": False,
                "epitope_gate_skipped": True,
                "epitope_gate_skip_reason": "missing structure",
            },
        },
    }
    failure = design_population([failed], parent, {})[0]
    assert failure["success"] is False and failure["error"] == "fold failed"
    assert failure["metrics"]["iptm"] is None
    assert failure["gate_evidence"]["cdr3_gate_passed"] is False
    assert "num_contacted_hotspots" not in failure["gate_evidence"]


def test_sequence_less_candidates_keep_compact_assignments_and_minibinder_gates():
    record = {
        "candidate_id": "other",
        "metadata": {
            "mutations": [{"chain_id": "B", "position": 3, "from_aa": None, "to_aa": "Y"}],
            "gate_evidence": {"cdr3_gate_passed": False, "coverage_ratio": 0.5},
        },
    }
    result = design_population([record], {"candidate_id": "parent"}, {}, minibinder=True)[0]
    assert result["position_assignments"] == [["B", 3, "Y"]]
    assert result["gate_evidence"] == {"coverage_ratio": 0.5}
