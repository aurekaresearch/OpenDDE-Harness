"""Immutable agent profiles for OpenDDE Harness's protein-design workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel

from opendde_harness.plugin.protein_design.agents.reflection import ReflectOutput
from opendde_harness.plugin.protein_design.core.contracts import (
    AnalyzeAgentOutput,
    DesignAgentOutput,
    ParentSelectionOutput,
    PostFilterAgentOutput,
    QualityBatchOutput,
)
from opendde_harness.plugin.protein_design.prompts import (
    ANALYZE_SYSTEM_PROMPT,
    DESIGN_SYSTEM_PROMPT,
    PARENT_SELECTION_SYSTEM_PROMPT,
    POST_FILTER_AGENT_INSTRUCTIONS,
    QUALITY_CHECK_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
)

T = TypeVar("T", bound=BaseModel)


class AgentRole(StrEnum):
    ANALYZE = "analyze"
    DESIGN = "design"
    QUALITY = "quality"
    REFLECTION = "reflection"
    POST_FILTER = "post_filter"
    PARENT_SELECTION = "parent_selection"


@dataclass(frozen=True)
class AgentProfile(Generic[T]):
    role: AgentRole
    system_prompt: str
    output_schema: type[T]
    default_skills: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    max_tool_turns: int
    max_tokens: int = 8192
    temperature: float = 0.2
    tool_errors_are_fatal: bool = True
    tool_call_limits: tuple[tuple[str, int], ...] = ()


AGENT_PROFILES: dict[AgentRole, AgentProfile[BaseModel]] = {
    AgentRole.ANALYZE: AgentProfile(
        role=AgentRole.ANALYZE,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        output_schema=AnalyzeAgentOutput,
        default_skills=("epitope-analysis", "structure-analysis"),
        allowed_tools=("protrek_sequence_search", "protrek_structure_search"),
        max_tool_turns=4,
    ),
    AgentRole.DESIGN: AgentProfile(
        role=AgentRole.DESIGN,
        system_prompt=DESIGN_SYSTEM_PROMPT,
        output_schema=DesignAgentOutput,
        default_skills=(
            "cdr-point-mutation",
            "cdr-full-redesign",
            "antibody-inverse-folding",
            "esm2-guided-mutation",
        ),
        allowed_tools=(
            "generate_soluble_mpnn",
            "generate_esm2_guided",
        ),
        max_tool_turns=6,
    ),
    AgentRole.QUALITY: AgentProfile(
        role=AgentRole.QUALITY,
        system_prompt=QUALITY_CHECK_SYSTEM_PROMPT,
        output_schema=QualityBatchOutput,
        default_skills=("developability-filter",),
        allowed_tools=("check_antibody_developability",),
        max_tool_turns=3,
    ),
    AgentRole.REFLECTION: AgentProfile(
        role=AgentRole.REFLECTION,
        system_prompt=REFLECT_SYSTEM_PROMPT,
        output_schema=ReflectOutput,
        default_skills=(
            "evolutionary-analysis",
            "epitope-analysis",
            "structure-analysis",
        ),
        allowed_tools=(
            "analyze_current_cycle_structure",
            "analyze_evolution_tree",
            "analyze_epitope",
        ),
        max_tool_turns=6,
        tool_errors_are_fatal=False,
        tool_call_limits=(("analyze_evolution_tree", 1),),
    ),
    AgentRole.POST_FILTER: AgentProfile(
        role=AgentRole.POST_FILTER,
        system_prompt=POST_FILTER_AGENT_INSTRUCTIONS,
        output_schema=PostFilterAgentOutput,
        default_skills=("post-filter",),
        allowed_tools=(),
        max_tool_turns=1,
    ),
    AgentRole.PARENT_SELECTION: AgentProfile(
        role=AgentRole.PARENT_SELECTION,
        system_prompt=PARENT_SELECTION_SYSTEM_PROMPT,
        output_schema=ParentSelectionOutput,
        default_skills=(),
        allowed_tools=(),
        max_tool_turns=1,
    ),
}
