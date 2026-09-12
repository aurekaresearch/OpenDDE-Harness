"""Cycle interval validation, boundary behavior, and reproducible router sampling."""

import asyncio
import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from test_protein_design_orchestrator import FakeCompute, FakePhases, make_config

from opendde_harness.plugin.protein_design.agents.router import (
    DesignRouteContext,
    route_design_skills,
    sample_design_skill,
)
from opendde_harness.plugin.protein_design.core.contracts import CycleDesignStage, WorkflowConfig
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.plugin.protein_design.core.population import ParentSampler
from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.core.schedule import apply_cycle_schedule, cycle_parameters


@pytest.mark.parametrize(
    "stage",
    [
        {"start_cycle": -1, "end_cycle": 1, "num_sequences": 2},
        {"start_cycle": 2, "end_cycle": 1, "num_sequences": 2},
        {"start_cycle": True, "end_cycle": 1, "num_sequences": 2},
        {"start_cycle": 0, "end_cycle": 1.5, "num_sequences": 2},
        {"start_cycle": 0, "end_cycle": 1, "num_sequences": 0},
        {"start_cycle": 0, "end_cycle": 1, "population_size": True},
        {"start_cycle": 0, "end_cycle": 1, "parent_fitness_temperature": float("nan")},
        {"start_cycle": 0, "end_cycle": 1, "parent_fitness_temperature": True},
        {"start_cycle": 0, "end_cycle": 1},
        {"start_cycle": 0, "end_cycle": 1, "router_skill_probabilities": {"typo": 1}},
        {"start_cycle": 0, "end_cycle": 1, "router_skill_probabilities": {"cdr-full-redesign": -1}},
        {"start_cycle": 0, "end_cycle": 1, "router_skill_probabilities": {"cdr-full-redesign": 0}},
        {"start_cycle": 0, "end_cycle": 1, "router_skill_probabilities": {"cdr-full-redesign": float("inf")}},
        {"start_cycle": 0, "end_cycle": 1, "router_skill_probabilities": {"cdr-full-redesign": True}},
        {"start_cycle": 0, "end_cycle": 1, "router_selection_strategy": "random"},
        {"start_cycle": 0, "end_cycle": 1, "num_sequences": 2, "typo": 1},
    ],
)
def test_invalid_stage_fails_before_launch(stage):
    with pytest.raises(ValidationError):
        CycleDesignStage.model_validate(stage)


@pytest.mark.parametrize("intervals", [[(0, 2), (2, 3)], [(2, 3), (0, 1)], [(0, 4)]])
def test_invalid_schedule_intervals_fail(intervals):
    with pytest.raises(ValidationError):
        WorkflowConfig(
            target="test",
            cycles=4,
            cycle_schedule=[dict(start_cycle=start, end_cycle=end, num_sequences=2) for start, end in intervals],
        )


def test_schedule_defaults_gaps_and_exact_weights_roundtrip():
    config = WorkflowConfig(
        target="test",
        cycles=5,
        candidates_per_cycle=8,
        population_size=20,
        skill_weights={"cdr-point-mutation": 1},
        esm2_available=False,
        cycle_schedule=[
            dict(
                start_cycle=0,
                end_cycle=1,
                num_sequences=16,
                population_size=40,
                router_skill_probabilities={"cdr-full-redesign": 1},
                parent_fitness_temperature=2,
            ),
            dict(start_cycle=3, end_cycle=3, num_sequences=2, router_selection_strategy="weighted"),
        ],
    )
    config = WorkflowConfig.model_validate_json(config.model_dump_json())
    defaults = cycle_parameters(config)
    original_defaults = copy.deepcopy(defaults)
    apply_cycle_schedule(config, 0, defaults)
    assert (config.candidates_per_cycle, config.population_size) == (16, 40)
    assert config.skill_weights["cdr-point-mutation"] == 0
    assert ParentSampler(config).temperature(0) == 2
    apply_cycle_schedule(config, 2, defaults)
    assert (config.candidates_per_cycle, config.population_size) == (8, 20)
    assert config.skill_weights == {"cdr-point-mutation": 1}
    assert config.parent_fitness_temperature is None
    assert not config.esm2_available
    apply_cycle_schedule(config, 3, defaults)
    assert config.candidates_per_cycle == 2
    assert config.router_selection_strategy == "weighted"
    assert config.population_size == 20
    apply_cycle_schedule(config, 4, defaults)
    assert config.router_selection_strategy == "agent"
    assert defaults == original_defaults


def test_yaml_loader_and_cli_summary_expose_schedule():
    from opendde_harness.cli.protein_design_commands import _workflow_summary

    data = yaml.safe_load((Path(__file__).parents[1] / "docs/examples/crlf2_quickstart.yaml").read_text())
    data["design"]["cycle_schedule"] = [dict(start_cycle=0, end_cycle=2, num_sequences=16)]
    config = WorkflowConfigLoader._normalize_config(data)
    assert _workflow_summary(config)["cycle_schedule"][0]["num_sequences"] == 16
    data["design"].update(parent_fitness_temperature=1, parent_fitness_temperature_end=0.1)
    with pytest.raises(ValueError, match="either"):
        WorkflowConfigLoader._normalize_config(data)


def test_weighted_router_is_reproducible_and_honors_legal_weights():
    context = DesignRouteContext(
        parent_sequences={"D": "CDEFGHIK"},
        mutable_positions={"D": list(range(8))},
        population_size=5,
        skill_weights={"cdr-point-mutation": 0.3, "cdr-full-redesign": 0.7},
    )
    route = route_design_skills(context)
    draws = [sample_design_skill(route, seed=42, cycle=i).selected_skill_id for i in range(1000)]
    assert 640 <= draws.count("cdr-full-redesign") <= 760
    assert sample_design_skill(route, seed=42, cycle=7) == sample_design_skill(route, seed=42, cycle=7)
    forced = route_design_skills(DesignRouteContext(**{**context.__dict__, "force_skill_id": "cdr-full-redesign"}))
    assert sample_design_skill(forced, seed=42, cycle=7).allowed_skill_ids == ("cdr-full-redesign",)
    masked = route_design_skills(DesignRouteContext(**{**context.__dict__, "parent_sequences": {"D": "XXXXXXXX"}}))
    assert sample_design_skill(masked, seed=42, cycle=7).allowed_skill_ids == ("cdr-full-redesign",)


def test_scheduled_run_changes_real_fold_batch_sizes_and_restores_base():
    class Phases(FakePhases):
        async def design_cycle(self, config, cycle, analysis, parents, best):
            result = await super().design_cycle(config, cycle, analysis, parents, best)
            exemplar = result.fold_candidates[0]
            result.fold_candidates.clear()
            for index in range(config.candidates_per_cycle):
                candidate = copy.deepcopy(exemplar)
                candidate["candidate_id"] = f"c{cycle}_{index}"
                sequence = "CDEFGHIKLMNPQRSTVWY"[(cycle * 4 + index) % 18] + exemplar["sequence"][1:]
                candidate.update(sequence=sequence, chains={"D": sequence})
                candidate["metadata"]["chains"] = {"D": sequence}
                result.fold_candidates.append(candidate)
            assert config.metadata["population_size"] <= config.population_size
            return result

    compute = FakeCompute()
    events = []
    config = make_config(
        cycles=5,
        cycle_retry_limit=0,
        skip_failed_cycles=False,
        candidates_per_cycle=2,
        population_size=8,
        post_filter_enabled=False,
        cycle_schedule=[
            dict(start_cycle=0, end_cycle=1, num_sequences=4, population_size=6, parent_fitness_temperature=2),
            dict(start_cycle=2, end_cycle=3, num_sequences=1, population_size=1, parent_fitness_temperature=0.1),
        ],
    )
    orchestrator = DesignOrchestrator(compute, DesignMemory(None, agent_id="test"), Phases(), event_sink=events.append)
    result = asyncio.run(orchestrator.run("schedule", config, stop_event=asyncio.Event(), adjustments={}))
    assert result.status.value == "completed"
    assert [len(batch) for batch in compute.fold_batches] == [1, 4, 4, 1, 1, 2]
    parameters = [e.output_payload for e in events if e.phase == "cycle_config" and e.output_payload]
    assert [p["num_sequences"] for p in parameters] == [4, 4, 1, 1, 2]
    assert [p["population_size"] for p in parameters] == [6, 6, 1, 1, 8]
    assert [p["parent_fitness_temperature"] for p in parameters[:4]] == [2, 2, 0.1, 0.1]
    assert not any(e.phase == "design_speculation" and e.cycle in (1, 3) for e in events)


def test_strategy_only_stage_is_valid():
    assert (
        CycleDesignStage(start_cycle=0, end_cycle=0, router_selection_strategy="weighted").router_selection_strategy
        == "weighted"
    )


def test_runtime_count_adjustment_persists_across_schedule_boundaries():
    config = make_config(
        cycles=3,
        post_filter_enabled=False,
        cycle_schedule=[
            dict(start_cycle=0, end_cycle=0, num_sequences=4),
            dict(start_cycle=1, end_cycle=2, num_sequences=1),
        ],
    )
    events = []
    orchestrator = DesignOrchestrator(
        FakeCompute(), DesignMemory(None, agent_id="test"), FakePhases(), event_sink=events.append
    )
    asyncio.run(orchestrator.run("adjusted", config, stop_event=asyncio.Event(), adjustments={"num_sequences": 3}))
    counts = [e.output_payload["num_sequences"] for e in events if e.phase == "cycle_config" and e.output_payload]
    assert counts == [3, 3, 3]


@pytest.mark.parametrize("name", ["crlf2_quickstart.yaml", "cacng1_quickstart.yaml"])
def test_removing_redundant_example_fields_preserves_effective_permissions(name):
    data = yaml.safe_load((Path(__file__).parents[1] / "docs/examples" / name).read_text())
    cleaned = WorkflowConfigLoader._normalize_config(data)
    explicit = copy.deepcopy(data)
    explicit["design"].update(
        optimization_metric="loss",
        cdr_contact_fraction_threshold=0.5,
        hotspot_contact_cutoff_a=5.0,
        post_refold_filter={"enabled": False},
    )
    for binder in explicit["initial_binders"]:
        for chain, value in binder["chains"].items():
            value["fixed_residues"] = cleaned.fixed_residues[chain]
    before = WorkflowConfigLoader._normalize_config(explicit)
    for key in (
        "fixed_residues",
        "mutable_positions",
        "cdr_regions",
        "cdr_region_groups",
        "objective_key",
        "minimize",
        "post_filter_enabled",
        "fold_options",
    ):
        assert getattr(before, key) == getattr(cleaned, key)


def test_reference_covers_every_accepted_yaml_key():
    reference = (Path(__file__).parents[1] / "docs/protein-design-yaml.md").read_text()
    for fields in (
        WorkflowConfigLoader._DESIGN_FIELDS,
        WorkflowConfigLoader._FOLD_FIELDS,
        WorkflowConfigLoader._STRUCTURED_FIELDS,
    ):
        for field in fields:
            assert f"`{field}`" in reference, field
