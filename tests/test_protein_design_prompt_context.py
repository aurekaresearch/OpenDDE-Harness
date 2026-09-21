"""Prompt projections keep scientific decisions without repeating raw records."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from opendde_harness.plugin.protein_design.agents.prompt_context import (
    compact_candidate,
    compact_gate_feedback,
    context_json,
    minibinder_prompt,
)


def test_design_prompt_modules_contain_only_literal_text_definitions():
    from opendde_harness.plugin.protein_design.prompts import antibody_design, minibinder

    for module in (antibody_design, minibinder):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.Expr):
                assert isinstance(ast.literal_eval(node.value), str)
            else:
                assert isinstance(node, ast.Assign)
                value = ast.literal_eval(node.value)
                assert isinstance(value, str) or (
                    isinstance(value, dict) and all(isinstance(v, str) for v in value.values())
                )


def candidate():
    return {
        "candidate_id": "parent",
        "sequence": "ACDE",
        "objective": 0.3,
        "metrics": {"iptm": 0.8, "cdr_contact_fraction": 0.4, "missing": None},
        "structure_path": "/real/parent.cif",
        "metadata": {
            "chains": {"B": "ACDE"},
            "gate_passed": False,
            "mutations": [{"chain_id": "B", "position": 2, "to_aa": "D"}],
            "raw_result": {"large_artifact": "x" * 10000},
            "gate_evidence": {
                "reason": "no_hotspot_contact",
                "missed_hotspots": [["A", 3]],
                "contacted_hotspots": [],
                "total_binder_contacts": 8,
                "cdr3_gate_passed": False,
                "framework_contact_fraction": 0.6,
                "framework_contact_residue_ids": list(range(40)),
                "contacted_binder_residues": list(range(40)),
            },
        },
    }


def feedback():
    parent = candidate()
    return {
        "objective_key": "loss",
        "minimize": True,
        "population_size": 2,
        "working_parent": {"candidate": parent, "violations": list(range(40))},
        "population_best": copy.deepcopy(parent),
        "candidate_changes": {
            "child": {"objective": 0.2, "parent_objective": 0.3, "delta": -0.1, "status": "improved"}
        },
        "candidate_gate_evidence": {
            "child": {"gate_passed": False, "reason": "no_hotspot_contact", "missed_hotspots": [["A", 3]]}
        },
        "recurring_offenders": {"B:2": 3},
    }


def test_gate_summary_preserves_decisions_and_deduplicates_without_mutating_records():
    original = feedback()
    before = copy.deepcopy(original)
    result = compact_gate_feedback(context_json(original))
    assert result["working_parent"] == result["population_best"] == "parent"
    assert set(result["candidates"]) == {"parent", "child"}
    assert result["candidates"]["child"]["delta"] == -0.1
    assert result["candidates"]["child"]["gate_evidence"]["gate_passed"] is False
    gate = result["candidates"]["parent"]["gate_evidence"]
    assert gate["missed_hotspots"] == [["A", 3]]
    assert "framework_contact_residue_ids" not in gate
    assert "contacted_binder_residues" not in gate
    assert "structure_path" not in result["candidates"]["parent"]
    assert "mutations" not in result["candidates"]["parent"]
    assert result["recurring_offenders"] == {"B:2": 3}
    assert len(context_json(result)) < len(context_json(original)) / 4
    assert original == before


def test_gate_feedback_bounds_sites_and_preserves_failed_skipped_and_unknown():
    value = feedback()
    value["candidate_gate_evidence"]["child"] = {
        "gate_passed": False,
        "cdr_contact_fraction": 0.49999,
        "cdr_contact_fraction_threshold": 0.5,
        "cdr_contact_fraction_gate_passed": False,
        "epitope_gate_skipped": True,
        "epitope_gate_skip_reason": "no structure",
        "missed_hotspots": [["A", i] for i in range(20)],
        "hotspot_position_semantics": "pdb_residue_id",
        "contact_pairs": ["raw contact"] * 100,
    }
    value["candidate_gate_evidence"]["unknown"] = {}
    value["recurring_offenders"] = {f"B:{i}": i for i in range(20)}
    original = copy.deepcopy(value)
    result = compact_gate_feedback(value)
    gate = result["candidates"]["child"]["gate_evidence"]
    assert gate["cdr_contact_fraction"] == 0.49999
    assert gate["cdr_contact_fraction_gate_passed"] is False
    assert gate["epitope_gate_skip_reason"] == "no structure"
    assert gate["num_missed_hotspots"] == 20
    assert len(gate["missed_hotspots"]) == 6
    assert gate["missed_hotspots_omitted"] == 14
    assert gate["hotspot_position_semantics"] == "pdb_residue_id"
    assert result["candidates"]["unknown"]["gate_evidence"] == {}
    assert len(result["recurring_offenders"]) == 5
    assert result["recurring_offenders_omitted"] == 15
    assert set(result["recurring_offenders"]) == {f"B:{i}" for i in range(15, 20)}
    assert "immutable" in result["recurring_offender_positions"]
    assert "raw contact" not in context_json(result)
    assert value == original


def test_minibinder_projection_removes_antibody_fields_not_interface_evidence():
    result = compact_gate_feedback(feedback(), minibinder=True)
    text = context_json(result)
    assert "cdr" not in text and "framework" not in text and "recurring_offenders" not in text
    assert "metrics" not in result["candidates"]["parent"]
    assert result["candidates"]["parent"]["gate_evidence"]["total_binder_contacts"] == 8
    assert result["candidates"]["parent"]["gate_evidence"]["gate_passed"] is False
    assert "gate_passed" not in compact_candidate({})
    assert compact_gate_feedback("unavailable") == "unavailable"
    assert compact_gate_feedback({"new_schema": "evidence"}) == {"new_schema": "evidence"}


def test_minibinder_static_prefix_and_single_parent_keep_other_candidates():
    config = SimpleNamespace(
        target="test",
        target_chains={"A": "AAAA"},
        fold_options={},
        hotspots=[],
        binder_chains={"B": "AAAA"},
        mutable_positions={"B": [2]},
        fixed_residues={"B": [0, 1, 3]},
        objective_key="loss",
        minimize=True,
    )
    parent = candidate()
    other = {**candidate(), "candidate_id": "other", "sequence": "ACEF"}
    inputs = {
        "parent": parent,
        "population": [copy.deepcopy(parent), other],
        "analysis": {"downstream_header": "stable analysis", "report": "long report"},
        "gate_feedback": context_json(feedback()),
        "cycle": 1,
        "candidate_count": 8,
        "mutation_budget": [1, 4],
        "route": "stable allowed skills",
        "reflection": "same reflection",
        "memories": ["retrieved evidence"],
        "learned_skills": ["advisory"],
    }
    before = copy.deepcopy(inputs)
    first = minibinder_prompt(config, **inputs)
    second = minibinder_prompt(config, **{**inputs, "parent": other, "cycle": 2})
    assert first.split(',"parent":')[0] == second.split(',"parent":')[0]
    result = json.loads(first)
    assert result["analysis"] == "stable analysis"
    assert result["parent"]["chains"] == {"B": "ACDE"}
    assert "sequence" not in result["parent"]
    assert [item["candidate_id"] for item in result["other_population_candidates"]] == ["other"]
    assert "population" not in result
    assert "cdr" not in first and "framework" not in first and "raw_result" not in first
    assert result["mutable_positions"] == config.mutable_positions
    assert inputs == before
    assert first == minibinder_prompt(config, **dict(reversed(list(inputs.items()))))
    keys = list(result)
    assert keys.index("analysis") < keys.index("candidate_count") < keys.index("route") < keys.index("parent")
    assert keys.index("memories") < keys.index("reflection")
    assert "learned_skills" not in result
    bounded = json.loads(minibinder_prompt(config, **{**inputs, "memories": [str(i) + "x" * 3000 for i in range(8)]}))
    assert len(bounded["memories"]) == 4
    assert len("\n".join(bounded["memories"])) <= 2000
    assert keys.index("mutation_budget") < keys.index("parent")
    assert keys.index("reflection") < keys.index("parent")
    assert keys[-1] == "cycle"
    changed = json.loads(minibinder_prompt(config, **{**inputs, "candidate_count": 4, "mutation_budget": [2, 3]}))
    assert changed["candidate_count"] == 4
    assert changed["mutation_budget"] == [2, 3]
