"""Resolve cycle overrides against the original task defaults, without carry-over."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from opendde_harness.plugin.protein_design.core.contracts import WorkflowConfig


def cycle_parameters(config: WorkflowConfig) -> dict[str, Any]:
    return {
        "num_sequences": config.candidates_per_cycle,
        "population_size": config.population_size,
        "router_skill_probabilities": deepcopy(config.skill_weights),
        "router_selection_strategy": config.router_selection_strategy,
        "esm2_available": config.esm2_available,
        "parent_fitness_temperature": config.parent_fitness_temperature,
    }


def stage_index(config: WorkflowConfig, cycle: int) -> int | None:
    return next(
        (index for index, stage in enumerate(config.cycle_schedule) if stage.start_cycle <= cycle <= stage.end_cycle),
        None,
    )


def apply_cycle_schedule(config: WorkflowConfig, cycle: int, defaults: dict[str, Any]) -> None:
    values = deepcopy(defaults)
    index = stage_index(config, cycle)
    if index is not None:
        values.update(
            config.cycle_schedule[index].model_dump(
                exclude={"start_cycle", "end_cycle"},
                exclude_none=True,
            )
        )
    config.candidates_per_cycle = values["num_sequences"]
    config.population_size = values["population_size"]
    weights = values["router_skill_probabilities"]
    if index is not None and config.cycle_schedule[index].router_skill_probabilities is not None:
        weights = {
            key: weights.get(key, 0.0)
            for key in ("cdr-point-mutation", "cdr-full-redesign", "antibody-inverse-folding", "esm2-guided-mutation")
        }
    config.skill_weights = weights
    config.router_selection_strategy = values["router_selection_strategy"]
    config.esm2_available = values["esm2_available"]
    config.parent_fitness_temperature = values["parent_fitness_temperature"]
    if config.skill_weights is not None:
        config.esm2_available = config.skill_weights.get("esm2-guided-mutation", 0.0) > 0
