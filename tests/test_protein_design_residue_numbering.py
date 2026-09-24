"""Residue coordinates cross the user boundary once, not once per backend."""

import json
from types import SimpleNamespace

import pytest

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends.loss_confidence_scorer import _resolve_hotspot_tokens


@pytest.mark.parametrize("value,want", [(1, [0]), (4, [3]), ("1:4", [0, 1, 2, 3]), ([1, 4], [0, 3])])
def test_yaml_positions_become_internal_indices(value, want):
    assert WorkflowConfigLoader._parse_positions(value, 4) == want


@pytest.mark.parametrize("value", [0, -1, 5, "0:2", "2:5"])
def test_yaml_rejects_positions_outside_one_based_sequence(value):
    with pytest.raises(ValueError):
        WorkflowConfigLoader._parse_positions(value, 4)


def test_loss_consumes_internal_first_and_last_hotspot_indices():
    assert _resolve_hotspot_tokens({"A": [5, 7, 9]}, ["A"], {"A": [0, 2]}) == (
        [5, 9],
        {"A": [0, 2]},
        "configured_hotspots",
    )


@pytest.mark.parametrize("position", [-1, 3])
def test_loss_rejects_out_of_bounds_internal_hotspots(position):
    with pytest.raises(ValueError):
        _resolve_hotspot_tokens({"A": [5, 7, 9]}, ["A"], {"A": [position]})


@pytest.mark.parametrize("design_type", ["antibody", "minibinder"])
def test_gate_maps_sequence_index_to_offset_structure_residue(monkeypatch, design_type):
    import biotite.structure as struc
    import numpy as np

    from opendde_harness.plugin.protein_design.core import gate

    atoms = struc.AtomArray(3)
    atoms.chain_id = ["A", "A", "B"]
    atoms.res_id = [10, 20, 50]
    atoms.res_name = ["ALA"] * 3
    atoms.atom_name = ["CA"] * 3
    atoms.element = ["C"] * 3
    atoms.coord = np.array([[0, 0, 0], [100, 0, 0], [2, 0, 0]], dtype=float)
    monkeypatch.setattr(gate, "_load_structure_model", lambda _: atoms)
    _, evidence = gate.evaluate_hotspot_contact_map(
        "unused", ["B"], [{"chain": "A", "position": 0}], target_chains=["A"], design_type=design_type
    )
    assert evidence["num_contacted_hotspots"] == 1
    assert evidence["num_missed_hotspots"] == 0


def test_minibinder_prompt_displays_one_based_masks_without_changing_config():
    from opendde_harness.plugin.protein_design.agents.prompt_context import minibinder_prompt

    config = SimpleNamespace(
        target="test",
        target_chains={"A": {"sequence": "AC", "hotspots": [0, 1]}},
        fold_options={"target_hotspots": {"A": [0, 1]}},
        hotspots=[0, 1],
        binder_chains={"B": "AC"},
        mutable_positions={"B": [0]},
        fixed_residues={"B": [1]},
        objective_key="loss",
        minimize=True,
    )
    result = json.loads(minibinder_prompt(config))
    assert result["hotspots"] == {"A": [1, 2]}
    assert result["target_chains"]["A"]["hotspots"] == [1, 2]
    assert result["mutable_positions"] == {"B": [1]}
    assert result["fixed_residues"] == {"B": [2]}
    assert config.mutable_positions == {"B": [0]}


def test_llm_mutation_uses_first_one_based_position():
    from opendde_harness.plugin.protein_design.agents.policy import POINT_MUTATION_SKILL
    from opendde_harness.plugin.protein_design.agents.proposals import ProposalContext, ProposalExecutor

    context = ProposalContext("seed", {"B": "AC"}, {"B": [0]}, None, 1)
    proposal = ProposalExecutor(None)._materialize_llm(
        {"candidate_id": "first", "mutations": [["B", 1, "G"]]}, POINT_MUTATION_SKILL, context
    )
    assert proposal.chains == {"B": "GC"}
    assert proposal.mutations[0].position == 0


def test_tool_converts_llm_masks_and_backend_mutations_once():
    import asyncio

    from opendde_harness.plugin.protein_design.tools.agent import ProteinDesignToolRegistry, ToolContext

    class Compute:
        async def generate_esm2_guided(self, request):
            assert request.mutable_positions == {"B": [0, 1]}
            return {
                "available": True,
                "result": {"candidates": [{"mutations": [{"chain_id": "B", "position": 0, "to_aa": "G"}]}]},
            }

    result = asyncio.run(
        ProteinDesignToolRegistry.for_compute(Compute()).execute(
            "generate_esm2_guided",
            {"parent_id": "seed", "parent_chains": {"B": "AC"}, "mutable_positions": {"B": [1, 2]}},
            ToolContext(),
            allowed=["generate_esm2_guided"],
        )
    )
    assert result["result"]["candidates"][0]["mutations"][0]["position"] == 1


def test_candidate_prompt_projects_mutations_without_mutating_record():
    from opendde_harness.plugin.protein_design.agents.prompt_context import compact_candidate

    record = {"candidate_id": "a", "metadata": {"mutations": [{"chain_id": "B", "position": 0, "to_aa": "G"}]}}
    assert compact_candidate(record)["mutations"][0]["position"] == 1
    assert record["metadata"]["mutations"][0]["position"] == 0


def test_homolog_alternatives_use_human_sequence_positions():
    from opendde_harness.plugin.protein_design.servers.backends.protrek import _annotate_sequence_alignments

    result = _annotate_sequence_alignments("ACDEFGHIKLMNPQRSTVWY", [{"sequence": "ACDEFGHIKAMNPQRSTVWY"}])
    assert result[0]["query_aligned_alternatives"] == ["L10A"]


def test_memory_lesson_labels_first_residue_one():
    from opendde_harness.plugin.protein_design.core.contracts import Candidate
    from opendde_harness.plugin.protein_design.core.design_cases import _lesson

    result = _lesson(
        selected_skill_id="cdr-point-mutation",
        mutations=[{"chain": "B", "position": 0, "from": "A", "to": "G"}],
        best_child=Candidate(candidate_id="child", sequence="GC", objective=1),
        parent_objective=2,
        objective_delta_parent=-1,
        retained_ids={"child"},
        triggers=[],
        verdict="pass",
        rejection_summary={},
    )
    assert "B1 A>G" in result["summary"]


@pytest.mark.parametrize("index,label", [(0, "B:1:A>G"), (3, "B:4:A>G")])
def test_evolution_mutation_labels_are_one_based(index, label):
    from opendde_harness.plugin.protein_design.servers.backends.evolution_analysis import _mutations

    candidate = {"metadata": {"mutations": [{"chain_id": "B", "position": index, "from_aa": "A", "to_aa": "G"}]}}
    assert _mutations(candidate) == [label]


def test_standalone_epitope_yaml_parser_uses_same_one_based_boundary():
    from opendde_harness.plugin.protein_design.servers.backends.epitope_analysis import _parse_yaml_positions

    assert _parse_yaml_positions("1:4") == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        _parse_yaml_positions([0])


def test_epitope_cli_json_displays_one_based_cdr_positions(monkeypatch, capsys):
    from opendde_harness.plugin.protein_design.servers.backends import epitope_analysis

    monkeypatch.setattr(
        "sys.argv",
        [
            "epitope_analysis",
            "--structure_file",
            "test.pdb",
            "--antibody_chains",
            "B",
            "--antigen_chains",
            "A",
            "--config_yaml",
            "test.yaml",
        ],
    )
    monkeypatch.setattr(epitope_analysis, "load_cdr_regions_from_yaml", lambda *a, **k: {"B": {"CDR1": {0}}})
    monkeypatch.setattr(epitope_analysis, "analyze_epitope", lambda **k: {"cdr_regions": {"B": {"CDR1": [0]}}})
    epitope_analysis.main()
    assert json.loads(capsys.readouterr().out)["cdr_regions"] == {"B": {"CDR1": [1]}}
