"""Coarse legality boundary for Design Agent skill selection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Mapping

POINT_MUTATION_SKILL = "cdr-point-mutation"
FULL_REDESIGN_SKILL = "cdr-full-redesign"
INVERSE_FOLDING_SKILL = "antibody-inverse-folding"
ESM2_GUIDED_MUTATION_SKILL = "esm2-guided-mutation"

DESIGN_SKILLS = (
    POINT_MUTATION_SKILL,
    FULL_REDESIGN_SKILL,
    INVERSE_FOLDING_SKILL,
    ESM2_GUIDED_MUTATION_SKILL,
)

DEFAULT_SKILL_WEIGHTS: dict[str, float] = {
    POINT_MUTATION_SKILL: 1.0,
    FULL_REDESIGN_SKILL: 0.0,
    INVERSE_FOLDING_SKILL: 0.0,
    ESM2_GUIDED_MUTATION_SKILL: 0.0,
}

BOOTSTRAP_SKILLS = (FULL_REDESIGN_SKILL, INVERSE_FOLDING_SKILL)

_BACKENDS = {
    POINT_MUTATION_SKILL: "llm",
    FULL_REDESIGN_SKILL: "llm",
    INVERSE_FOLDING_SKILL: "inverse_folding",
    ESM2_GUIDED_MUTATION_SKILL: "esm2",
}


@dataclass(frozen=True)
class DesignRouteContext:
    parent_sequences: Mapping[str, str]
    mutable_positions: Mapping[str, tuple[int, ...] | list[int]]
    population_size: int
    inverse_folding_available: bool = True
    esm2_available: bool = True
    skill_weights: Mapping[str, float] | None = None
    force_skill_id: str | None = None
    low_quality_alanine_fraction: float = 0.60
    low_quality_min_mutable: int = 6


@dataclass(frozen=True)
class DesignSkillRoute:
    allowed_skill_ids: tuple[str, ...]
    weights: dict[str, float]
    reason: str
    selected_skill_id: str | None = None
    backends: dict[str, str] = field(default_factory=dict)

    def require_selected(self, skill_id: str | None) -> str:
        selected = str(skill_id or "").strip()
        if not selected:
            raise ValueError("Design Agent must select exactly one skill_id")
        if selected not in self.allowed_skill_ids:
            raise ValueError(f"Design skill {selected!r} is not allowed by this route")
        return selected

    def prompt_block(self) -> str:
        lines = [
            "=== ROUTER-ALLOWED PRIMARY SKILLS ===",
            "The Router checked runtime legality. Compare all listed skills and select exactly one.",
        ]
        for skill_id in self.allowed_skill_ids:
            lines.append(f"- `{skill_id}`: backend={self.backends[skill_id]}; prior={self.weights[skill_id]:.4f}")
        lines.extend(
            (
                f"- Router reason: {self.reason}.",
                "- Return the chosen ID once as top-level `skill_id`; do not compose skills.",
            )
        )
        return "\n".join(lines)


def _requires_bootstrap(context: DesignRouteContext) -> tuple[bool, str]:
    mutable_count = 0
    alanine_count = 0
    for chain_id, positions in context.mutable_positions.items():
        sequence = str(context.parent_sequences.get(chain_id, "")).upper()
        for position in positions:
            index = int(position)
            if index < 0 or index >= len(sequence):
                return True, "mutable_position_missing_from_parent"
            mutable_count += 1
            residue = sequence[index]
            if residue == "X":
                return True, "mutable_region_contains_masked_residues"
            alanine_count += residue == "A"
    if context.population_size <= 0:
        return True, "population_empty"
    if (
        mutable_count >= context.low_quality_min_mutable
        and mutable_count
        and alanine_count / mutable_count >= context.low_quality_alanine_fraction
    ):
        return True, "mutable_region_is_poly_alanine"
    return False, "parent_materialized"


def _normalized_weights(
    allowed: tuple[str, ...],
    configured: Mapping[str, float],
) -> dict[str, float]:
    values = {name: max(0.0, float(configured.get(name, 0.0))) for name in allowed}
    total = sum(values.values())
    if total <= 0:
        values = {name: 1.0 for name in allowed}
        total = float(len(values))
    return {name: value / total for name, value in values.items()}


def route_design_skills(context: DesignRouteContext) -> DesignSkillRoute:
    configured = dict(DEFAULT_SKILL_WEIGHTS)
    if context.skill_weights is not None:
        unknown = set(context.skill_weights) - set(DESIGN_SKILLS)
        if unknown:
            raise ValueError("unknown design skill: " + ", ".join(sorted(unknown)))
        configured.update({key: float(value) for key, value in context.skill_weights.items()})

    bootstrap, reason = _requires_bootstrap(context)
    capable = {
        POINT_MUTATION_SKILL,
        FULL_REDESIGN_SKILL,
        *(set((INVERSE_FOLDING_SKILL,)) if context.inverse_folding_available else set()),
        *(set((ESM2_GUIDED_MUTATION_SKILL,)) if context.esm2_available else set()),
    }
    if context.force_skill_id:
        if context.force_skill_id not in DESIGN_SKILLS:
            raise ValueError(f"unknown design skill: {context.force_skill_id}")
        if context.force_skill_id not in capable:
            raise ValueError(f"forced design skill is unavailable: {context.force_skill_id}")
        allowed = (context.force_skill_id,)
        reason = "configured_forced_skill"
    elif bootstrap:
        allowed = tuple(skill for skill in BOOTSTRAP_SKILLS if skill in capable and configured.get(skill, 0.0) > 0.0)
        # A masked or otherwise unusable parent must still be materialized. If
        # the Router menu disables every bootstrap helper, full redesign is the
        # only safe mandatory fallback; a zero-weight inverse-folding skill must
        # never be re-enabled implicitly.
        if not allowed:
            allowed = (FULL_REDESIGN_SKILL,)
            configured[FULL_REDESIGN_SKILL] = 1.0
    else:
        allowed = tuple(skill for skill in DESIGN_SKILLS if skill in capable and configured.get(skill, 0.0) > 0.0)
        if not allowed:
            allowed = (POINT_MUTATION_SKILL,)
            configured[POINT_MUTATION_SKILL] = 1.0

    weights = _normalized_weights(allowed, configured)
    return DesignSkillRoute(
        allowed_skill_ids=allowed,
        weights=weights,
        reason=reason,
        backends={skill: _BACKENDS[skill] for skill in allowed},
    )


def sample_design_skill(route: DesignSkillRoute, *, seed: int, cycle: int) -> DesignSkillRoute:
    """Choose one legal skill reproducibly; retries reuse the same cycle draw."""
    selected = random.Random(f"protein-design-router:{seed}:{cycle}").choices(
        route.allowed_skill_ids,
        weights=[route.weights[key] for key in route.allowed_skill_ids],
        k=1,
    )[0]
    return replace(
        route,
        allowed_skill_ids=(selected,),
        selected_skill_id=selected,
        weights={selected: 1.0},
        backends={selected: route.backends[selected]},
        reason=f"{route.reason}; weighted cycle draw selected {selected}",
    )
