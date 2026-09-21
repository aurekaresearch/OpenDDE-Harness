"""One skill-body resolver for host tools and scoped workflow agents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from opendde_harness.memory_engine.skill_forge.catalog import is_blocked, normalize_blocklist
from opendde_harness.memory_engine.skill_forge.documents import SkillDocument
from opendde_harness.memory_engine.skill_forge.refs import resolve_refs
from opendde_harness.memory_engine.skill_local.registry import SkillRegistry


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    qualified_id: str
    content: str
    skill_dir: Path | None

    def render(self) -> str:
        scripts = self.skill_dir / "scripts" if self.skill_dir else None
        resource = (
            f"scripts_dir: {scripts}\ncached: true"
            if scripts and scripts.is_dir()
            else "(no bundled scripts — pure-instruction skill; follow the body)"
        )
        return f"## {self.name}\n{resource}\n\n{self.content}"


class SkillLoader:
    def __init__(
        self,
        registry: SkillRegistry | None = None,
        *,
        documents: Iterable[SkillDocument] | None = None,
        blocklist: Iterable[str] | None = None,
    ) -> None:
        self._registry = registry
        self._blocklist = normalize_blocklist(blocklist)
        # None permits registry lookup; an empty snapshot permits nothing.
        self._documents = None if documents is None else tuple(documents)

    def load(self, skill_id: str) -> LoadedSkill:
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("'skill_id' is required — a skill's qualified id like 'local/<name>'.")
        source, sep, native = skill_id.partition("/")
        if not sep:
            source, native = "local", skill_id
        if is_blocked(self._blocklist, skill_id, native):
            raise ValueError(
                f"skill {skill_id!r} is on the operator blocklist (skillForge.blocklist) and cannot be used."
            )
        if self._documents is not None:
            # Resolve only the caller's allowlist, never the global registry.
            matches = [d for d in self._documents if skill_id in (d.name, d.qualified_id)]
            if len(matches) != 1:
                raise ValueError(f"unknown or Router-disallowed skill {skill_id!r}")
            doc = matches[0]
            name, qualified, body = doc.name, doc.qualified_id, doc.content
            directory = None if doc.source == "memory" else doc.path
            identifiers = (skill_id, qualified, name, doc.native_id)
        else:
            if source not in ("local", "memory"):
                raise ValueError(f"unknown skill source {source!r} in {skill_id!r} (expected local or memory).")
            meta = None
            if self._registry is not None:
                meta = self._registry.get(native) if source == "local" else self._registry.get(native, source=source)
            if meta is None:
                raise ValueError(f"no {source} skill {native!r} found on disk.")
            name, qualified, body, directory = meta.name, f"{source}/{native}", meta.content, meta.path.parent
            identifiers = (skill_id, qualified, name, native)
        if is_blocked(self._blocklist, *identifiers):
            raise ValueError(
                f"skill {skill_id!r} is on the operator blocklist (skillForge.blocklist) and cannot be used."
            )
        body, _ = resolve_refs(body, directory)
        return LoadedSkill(name, qualified, body, directory)
