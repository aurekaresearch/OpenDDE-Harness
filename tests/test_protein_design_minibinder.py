"""Existing mini binder optimization stays independent of antibody policies."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
from opendde_harness.plugin.protein_design.agents.policy import DesignRouteContext, route_design_skills
from opendde_harness.plugin.protein_design.agents.profiles import AgentRole, profile_for
from opendde_harness.plugin.protein_design.agents.proposals import (
    ProposalContext,
    ProposalExecutor,
    ProposalValidationError,
)
from opendde_harness.plugin.protein_design.core.contracts import AnalyzeAgentOutput, Candidate, DesignAgentOutput
from opendde_harness.plugin.protein_design.core.gate import evaluate_hotspot_contact_map
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.core.schedule import apply_cycle_schedule, cycle_parameters


def configuration():
    return {
        "target": {"name": "test", "chains": {"A": {"sequence": "ACDE", "hotspots": [0]}}},
        "initial_binders": [
            {
                "name": "seed",
                "chains": {
                    "B": {
                        "sequence": "ACDEFG",
                        "chain_type": "minibinder",
                        "designable_residues": "1:4",
                        "fixed_residues": [2],
                    }
                },
            }
        ],
        "design": {"type": "minibinder", "n_cycles": 2, "num_sequences": 1, "num_mutations": 1},
        "fold": {"execution_mode": "local", "checkpoint_path": "/weights/opendde.pt"},
    }


def test_minibinder_permissions_are_not_cdr_annotations():
    config = WorkflowConfigLoader._normalize_config(configuration())
    assert config.design_type == config.metadata["binder_type"] == "minibinder"
    assert config.mutable_positions == {"B": [1, 3, 4]}
    assert config.fixed_residues == {"B": [0, 2, 5]}
    assert config.cdr_regions == {"B": []}
    assert config.cdr_region_groups == {"B": []}
    assert config.fold_options["design_type"] == "minibinder"
    assert config.fold_options["checkpoint_path"] == "/weights/opendde.pt"
    assert config.model_validate_json(config.model_dump_json()).design_type == "minibinder"


@pytest.mark.parametrize(
    "field,value",
    [
        ("cdr_regions", [1]),
        ("sequence", "ACXXFG"),
        ("chain_type", "VHH"),
        ("designable_residues", None),
        ("designable_residues", []),
        ("designable_residues", [100]),
    ],
)
def test_invalid_minibinder_input_is_rejected(field, value):
    data = configuration()
    data["initial_binders"][0]["chains"]["B"][field] = value
    with pytest.raises(ValueError):
        WorkflowConfigLoader._normalize_config(data)


@pytest.mark.parametrize(
    "fold", [{"execution_mode": "api"}, {"checkpoint_path": "opendde_abag.pt"}, {"execution_mode": "local"}]
)
def test_minibinder_requires_explicit_general_checkpoint(fold):
    data = configuration()
    data["fold"] = fold
    with pytest.raises(ValueError, match="checkpoint"):
        WorkflowConfigLoader._normalize_config(data)


def test_antibody_default_and_minibinder_schedule_are_isolated():
    data = yaml.safe_load((Path(__file__).parents[1] / "docs/examples/crlf2_quickstart.yaml").read_text())
    assert WorkflowConfigLoader._normalize_config(data).design_type == "antibody"
    data = configuration()
    data["design"]["cycle_schedule"] = [
        {"start_cycle": 0, "end_cycle": 0, "router_skill_probabilities": {"minibinder-inverse-folding": 1}}
    ]
    config = WorkflowConfigLoader._normalize_config(data)
    defaults = cycle_parameters(config)
    apply_cycle_schedule(config, 0, defaults)
    assert set(config.skill_weights) == {"minibinder-point-mutation", "minibinder-inverse-folding"}
    data["design"]["cycle_schedule"][0]["router_skill_probabilities"] = {"cdr-point-mutation": 1}
    with pytest.raises(ValueError, match="design.type"):
        WorkflowConfigLoader._normalize_config(data)


def route(**kwargs):
    return route_design_skills(
        DesignRouteContext(
            parent_sequences={"B": "ACDEFG"},
            mutable_positions={"B": [1, 3]},
            population_size=1,
            design_type="minibinder",
            **kwargs,
        )
    )


def test_minibinder_route_checks_structure_and_disallows_antibody_skills():
    assert route().allowed_skill_ids == ("minibinder-point-mutation",)
    weights = {"minibinder-point-mutation": 0, "minibinder-inverse-folding": 1}
    with pytest.raises(ValueError, match="parent structure"):
        route(skill_weights=weights, inverse_folding_available=False)
    assert route(skill_weights=weights).backends == {"minibinder-inverse-folding": "inverse_folding"}
    with pytest.raises(ValueError, match="unsupported"):
        route(skill_weights={"cdr-point-mutation": 1})


@pytest.mark.parametrize("position,valid", [(1, True), (0, False), (9, False)])
def test_point_mutations_keep_fixed_residues_and_length(position, valid):
    output = DesignAgentOutput(
        skill_id="minibinder-point-mutation",
        candidates=[{"candidate_id": "child", "mutations": [{"chain_id": "B", "position": position, "to_aa": "W"}]}],
    )
    executor = ProposalExecutor(None)
    context = ProposalContext("seed", {"B": "ACDEFG"}, {"B": [1, 3]}, None, 1)
    if valid:
        result = asyncio.run(executor.execute(output, route(), context))
        assert result[0].chains == {"B": "AWDEFG"}
    else:
        with pytest.raises(ProposalValidationError):
            asyncio.run(executor.execute(output, route(), context))


@pytest.mark.parametrize(
    "distance,hotspots,expected",
    [
        (2, [], True),
        (20, [], False),
        (2, [{"chain": "A", "position": 0}], True),
        (2, [{"chain": "A", "position": 1}], False),
    ],
)
def test_minibinder_gate_uses_interface_not_cdrs(monkeypatch, distance, hotspots, expected):
    struc = pytest.importorskip("biotite.structure")
    import numpy as np

    import opendde_harness.plugin.protein_design.core.gate as gate

    atoms = struc.AtomArray(3)
    atoms.chain_id = ["A", "A", "B"]
    atoms.res_id = [10, 20, 50]
    atoms.res_name = ["ALA"] * 3
    atoms.atom_name = ["CA"] * 3
    atoms.element = ["C"] * 3
    atoms.coord = np.array([[0, 0, 0], [100, 0, 0], [distance, 0, 0]], dtype=float)
    monkeypatch.setattr(gate, "_load_structure_model", lambda _: atoms)
    passed, evidence = evaluate_hotspot_contact_map(
        "unused", ["B"], hotspots, target_chains=["A"], design_type="minibinder"
    )
    assert passed is expected
    assert evidence["total_binder_contacts"] == (1 if distance == 2 else 0)
    assert not any(key.startswith(("cdr", "framework")) for key in evidence)
    antibody_passed, _ = evaluate_hotspot_contact_map("unused", ["B"], [], target_chains=["A"])
    assert not antibody_passed


class Session:
    def __init__(self):
        self.calls = []

    async def run(self, profile, prompt, **kwargs):
        self.calls.append((profile, json.loads(prompt), kwargs))
        if profile.role == AgentRole.ANALYZE:
            return AnalyzeAgentOutput(downstream_header="Existing binder")
        if profile.role == AgentRole.DESIGN:
            return DesignAgentOutput(
                skill_id="minibinder-point-mutation",
                candidates=[
                    {
                        "candidate_id": "child",
                        "mutations": [{"chain_id": "B", "position": 1, "from_aa": "C", "to_aa": "W"}],
                    }
                ],
            )
        if profile.role == AgentRole.QUALITY:
            return profile.output_schema(
                results={
                    "child": {"reasoning": "Evidence incomplete", "overall_risk_level": "unknown", "pass_check": True}
                }
            )
        if profile.role == AgentRole.REFLECTION:
            return profile.output_schema(
                summary="Keep fixed positions", next_cycle_recommendations=["Inspect interface"]
            )
        if profile.role == AgentRole.POST_FILTER:
            result = profile.output_schema(
                strategy_summary="Evidence-based",
                decisions=[{"candidate_id": "child", "rank": 1, "rationale": "Interface support"}],
            )
            kwargs["output_validator"](result)
            return result
        raise AssertionError(profile.role)


def test_all_minibinder_agent_phases_and_postfilter_evidence():
    async def run():
        config = WorkflowConfigLoader._normalize_config(configuration())
        session = Session()
        compute = SimpleNamespace(developability=AsyncMock(side_effect=AssertionError("antibody QC must not run")))
        phases = ProteinDesignPhases(session, compute, DesignMemory(None, agent_id="test"))
        analysis = await phases.analyze_once(config)
        design = await phases.design_cycle(config, 0, analysis, config.initial_candidates, None)
        assert design.fold_candidates[0]["sequence"] == "AWDEFG"
        candidate = Candidate(
            candidate_id="child",
            sequence="AWDEFG",
            metrics={"iptm": 0.8},
            metadata={"gate_evidence": {"total_binder_contacts": 4}},
        )
        quality = await phases.quality_cycle(config, 0, [candidate], analysis)
        reflection = await phases.reflect_cycle(config, 0, [candidate], candidate, analysis, quality)
        assert "Inspect interface" in reflection.to_design_directives()
        await phases.post_filter_run(config, [candidate], analysis)
        evidence = session.calls[-1][1]["candidate_evidence"][0]
        assert "cdr_contact_fraction" not in evidence["missing_fields"]
        assert evidence["interface"]["total_binder_contacts"] == 4
        assert all(not profile.allowed_tools for profile, _, _ in session.calls)
        assert all(payload["design_type"] == "minibinder" for _, payload, _ in session.calls)
        compute.developability.assert_not_awaited()

    asyncio.run(run())


def test_checkpoint_override_is_per_task(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor

    checkpoint = tmp_path / "opendde.pt"
    checkpoint.touch()
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", "/wrong/opendde_abag.pt")
    predictor = object.__new__(StructurePredictor)
    predictor.config = FoldConfig(execution_mode="local", device="cpu", checkpoint_path=str(checkpoint))
    assert predictor._get_opendde_checkpoint_args() == (str(checkpoint), [])


def test_profiles_preserve_antibody_defaults():
    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES

    for role in AgentRole:
        assert profile_for(role, "antibody") is AGENT_PROFILES[role]
        assert "check_antibody_developability" not in profile_for(role, "minibinder").allowed_tools


def test_minibinder_point_mutation_budget_is_enforced():
    context = ProposalContext("seed", {"B": "ACDEFG"}, {"B": [1, 3]}, None, 1, mutation_count_bounds=(1, 1))
    output = DesignAgentOutput(
        skill_id="minibinder-point-mutation",
        candidates=[
            {
                "candidate_id": "child",
                "mutations": [
                    {"chain_id": "B", "position": 1, "to_aa": "W"},
                    {"chain_id": "B", "position": 3, "to_aa": "Y"},
                ],
            }
        ],
    )
    with pytest.raises(ProposalValidationError, match="budget"):
        asyncio.run(ProposalExecutor(None).execute(output, route(), context))


def test_minibinder_inverse_folding_preserves_mask():
    compute = SimpleNamespace(
        generate_soluble_mpnn=AsyncMock(
            return_value={"candidates": [{"candidate_id": "mpnn", "chains": {"B": "AWDEFG"}}]}
        )
    )
    context = ProposalContext("seed", {"B": "ACDEFG"}, {"B": [1, 3]}, "/parent.cif", 1)
    selected = route(skill_weights={"minibinder-point-mutation": 0, "minibinder-inverse-folding": 1})
    result = asyncio.run(
        ProposalExecutor(compute).execute(DesignAgentOutput(skill_id="minibinder-inverse-folding"), selected, context)
    )
    assert result[0].chains == {"B": "AWDEFG"}
    request = compute.generate_soluble_mpnn.call_args.args[0]
    assert request.mutable_positions == ["B:1", "B:3"]
    assert request.parent_chains == {"B": "ACDEFG"}


def test_minibinder_mock_workflow_reaches_final_refold_and_selection():
    from test_protein_design_orchestrator import FakeCompute

    from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator

    class Compute(FakeCompute):
        async def submit_fold(self, request):
            assert request.options["design_type"] == "minibinder"
            assert request.options["checkpoint_path"] == "/weights/opendde.pt"
            submission = await super().submit_fold(request)
            for item in self.jobs[submission.job_id]["candidates"]:
                item["metadata"]["chains"] = {"B": item["sequence"]}
            return submission

        async def generate_soluble_mpnn(self, request):
            assert request.mutable_positions == ["B:1", "B:3", "B:4"]
            samples = []
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
            for index in range(request.num_sequences):
                sequence = list(request.parent_chains["B"])
                sequence[1], sequence[3] = alphabet[index % 20], alphabet[index // 20]
                samples.append(
                    {"chains": {"B": "".join(sequence)}, "metadata": {"soluble_mpnn_scores": {"B": index / 100}}}
                )
            return {"candidates": samples}

    class WorkflowSession(Session):
        async def run(self, profile, prompt, **kwargs):
            if profile.role == AgentRole.POST_FILTER:
                payload = json.loads(prompt)
                output = profile.output_schema(
                    strategy_summary="mock ranking",
                    decisions=[
                        {"candidate_id": item["candidate_id"], "rank": index + 1, "rationale": "mock evidence"}
                        for index, item in enumerate(payload["candidate_evidence"])
                    ],
                )
                kwargs["output_validator"](output)
                return output
            return await super().run(profile, prompt, **kwargs)

    async def run():
        data = configuration()
        data["design"].update(
            n_cycles=1,
            enable_quality_check=False,
            reflection_interval=10,
            post_refold_filter={"enabled": True, "top_k": 2},
        )
        config = WorkflowConfigLoader._normalize_config(data)
        compute = Compute()
        memory = DesignMemory(None, agent_id="test")
        phases = ProteinDesignPhases(WorkflowSession(), compute, memory)
        snapshot = await DesignOrchestrator(compute, memory, phases).run(
            "mini-mock", config, stop_event=asyncio.Event(), adjustments={}
        )
        assert not snapshot.failed_cycles
        assert snapshot.final_candidates
        assert snapshot.final_selection is not None
        assert len(compute.fold_batches) >= 3
        for candidate in snapshot.final_candidates:
            assert len(candidate.sequence) == 6
            assert candidate.sequence[0] == "A" and candidate.sequence[2] == "D" and candidate.sequence[5] == "G"

    asyncio.run(run())
