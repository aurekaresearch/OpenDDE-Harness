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
from opendde_harness.memory_engine.skill_forge.catalog import is_blocked, normalize_blocklist

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opendde_harness.memory_engine.skill_local.registry import SkillRegistry


def _split_qualified_id(skill_id: str) -> tuple[str, str]:
    """Split ``<source>/<native_id>``; a bare id is treated as ``local``."""
    source, sep, native = skill_id.partition("/")
    if not sep:
        return "local", source
    return source, native


class UseSkillTool(Tool):
    """Resolve a skill's SKILL.md and bundled scripts for ``exec``."""

    def __init__(
        self,
        registry: "SkillRegistry | None" = None,
        *,
        blocklist: "Iterable[str] | None" = None,
    ) -> None:
        self._registry = registry
        self._blocklist = normalize_blocklist(blocklist)

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
        if not skill_id or not isinstance(skill_id, str):
            return "Error: 'skill_id' is required — a skill's qualified id like 'local/<name>'."
        source, native = _split_qualified_id(skill_id)
        if is_blocked(self._blocklist, native):
            return f"Error: skill {native!r} is on the operator blocklist (skillForge.blocklist) and cannot be used."
        if source not in ("local", "memory"):
            return f"Error: unknown skill source {source!r} in {skill_id!r} (expected local or memory)."
        if self._registry is None:
            meta = None
        elif source == "local":
            # ``local`` is a logical SkillForge source, not a physical
            # SkillRegistry layer. Resolve through normal layer precedence.
            meta = self._registry.get(native)
        else:
            meta = self._registry.get(native, source=source) or self._registry.get(native)
        if meta is None:
            return (
                f"Error: no {source} skill {native!r} found on disk. If it is a "
                f"pure-instruction skill its body is already in your context."
            )
        scripts = meta.path.parent / "scripts"
        if scripts.is_dir():
            return f"## {meta.name}\nscripts_dir: {scripts}\ncached: true\n\n{meta.content}"
        return f"## {meta.name}\n(no bundled scripts — pure-instruction skill; follow the body)\n\n{meta.content}"
