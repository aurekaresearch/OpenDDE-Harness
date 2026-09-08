"""Segment 4 — always-available skill catalog. Host-owned."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opendde_harness.context_engine.base import AssemblyContext, Segment
from opendde_harness.tracing import semconv, trace

if TYPE_CHECKING:
    from opendde_harness.memory_engine.skill_forge import LocalSkillCatalog


class ActiveSkillsSegmentBuilder:
    name = "active_skills"
    order = 4
    needs_prefix = False

    def __init__(self, skill_catalog: "LocalSkillCatalog") -> None:
        self._skills = skill_catalog

    @trace.instrument("skill.inject", kind="skill", detached=True, extract=semconv.skill_inject_active)
    async def build(self, ctx: AssemblyContext) -> Segment | None:
        always_skills = self._skills.get_always_skills()
        if not always_skills:
            return None
        lines = [
            "These skills are always available. Call `use_skill` with the exact "
            "qualified id before following a skill; bodies are not pre-injected."
        ]
        for skill in always_skills:
            # ``local`` is the public routing namespace for every on-disk
            # workspace/builtin/external skill. use_skill resolves the actual
            # registry layer through normal precedence.
            qid = f"memory/{skill.name}" if skill.source == "memory" else f"local/{skill.name}"
            line = f"- **{skill.name}** [`{qid}`]"
            if skill.description:
                line += f": {skill.description}"
            lines.append(line)
        return Segment(
            text="# Active Skills\n\n" + "\n\n".join(lines),
            meta={
                "injected_skill_ids": [],
                "available_skill_ids": [
                    f"memory/{skill.name}" if skill.source == "memory" else f"local/{skill.name}"
                    for skill in always_skills
                ],
            },
        )
