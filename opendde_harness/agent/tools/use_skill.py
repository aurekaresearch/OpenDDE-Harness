"""``use_skill`` — load a skill body (and bundled scripts) on demand.

The ``# Skills`` catalog advertises names, descriptions and qualified ids
only; the agent calls ``use_skill`` with the qualified id and the tool
resolves the skill dir through the registry (``local/<name>`` or
``memory/<id>``), returning the SKILL.md body plus the ``scripts/`` path
when the skill bundles resources. Nothing is downloaded: every skill this
tool can serve is already materialized on disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opendde_harness.agent.tools.base import Tool
from opendde_harness.memory_engine.skill_forge.loader import SkillLoader

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opendde_harness.memory_engine.skill_local.registry import SkillRegistry


class UseSkillTool(Tool):
    """Resolve a skill's SKILL.md and bundled scripts for ``bash``."""

    def __init__(
        self,
        registry: "SkillRegistry | None" = None,
        *,
        blocklist: "Iterable[str] | None" = None,
    ) -> None:
        self._loader = SkillLoader(registry, blocklist=blocklist)

    @property
    def name(self) -> str:
        return "use_skill"

    @property
    def description(self) -> str:
        return (
            "Load a skill before using it. Pass the exact qualified id from the "
            "'# Skills' catalog (e.g. 'local/x' or 'memory/x'). "
            "Returns the SKILL.md instructions and, when present, a scripts_dir "
            "for bundled resources. This call is required for pure-instruction "
            "skills too; skill bodies are not pre-injected into context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": (
                        "The skill's qualified id, exactly as shown in the "
                        "'# Skills' catalog brackets (e.g. 'local/my-skill')."
                    ),
                },
            },
            "required": ["skill_id"],
        }

    async def execute(self, skill_id: Any = None, **_: Any) -> str:
        try:
            return self._loader.load(skill_id).render()
        except ValueError as exc:
            return f"Error: {exc}"
