"""Strict catalog for built-in and collision-safe learned protein-design skills."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BUILTIN_PROTEIN_DESIGN_SKILLS = (
    "cdr-point-mutation",
    "cdr-full-redesign",
    "antibody-inverse-folding",
    "esm2-guided-mutation",
    "evolutionary-analysis",
    "epitope-analysis",
    "structure-analysis",
    "developability-filter",
    "post-filter",
)

_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "cdr-point-mutation": ("SKILL.md", "agents/openai.yaml"),
    "cdr-full-redesign": ("SKILL.md", "agents/openai.yaml"),
    "antibody-inverse-folding": ("SKILL.md", "agents/openai.yaml"),
    "esm2-guided-mutation": ("SKILL.md", "agents/openai.yaml"),
    "evolutionary-analysis": ("SKILL.md",),
    "epitope-analysis": ("SKILL.md",),
    "structure-analysis": ("SKILL.md",),
    "developability-filter": ("SKILL.md",),
    "post-filter": ("SKILL.md",),
}


class MissingBuiltinSkillError(RuntimeError):
    pass


@dataclass(frozen=True)
class LearnedSkill:
    name: str
    content: str
    roles: tuple[str, ...]
    native_id: str
    score: float = 0.0


@dataclass(frozen=True)
class SkillDocument:
    name: str
    path: Path
    content: str
    missing_references: tuple[str, ...]
    source: str = "builtin"
    roles: tuple[str, ...] = ()
    native_id: str | None = None


class ProteinDesignSkillCatalog:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._documents: dict[str, SkillDocument] = {}
        for name in BUILTIN_PROTEIN_DESIGN_SKILLS:
            path = root / name
            missing = tuple(relative for relative in _REQUIRED_FILES[name] if not (path / relative).is_file())
            skill_file = path / "SKILL.md"
            content = skill_file.read_text(encoding="utf-8") if skill_file.is_file() else ""
            self._documents[name] = SkillDocument(name, path, content, missing)

    @classmethod
    def builtin(cls) -> "ProteinDesignSkillCatalog":
        root = Path(__file__).resolve().parents[1] / "skills"
        catalog = cls(root)
        incomplete = {
            name: document.missing_references
            for name, document in catalog._documents.items()
            if document.missing_references
        }
        if incomplete:
            details = "; ".join(f"{name}: {', '.join(paths)}" for name, paths in incomplete.items())
            raise MissingBuiltinSkillError(f"incomplete built-in protein-design skills: {details}")
        return catalog

    def names(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def documents(self) -> tuple[SkillDocument, ...]:
        return tuple(self._documents.values())

    def require(self, name: str) -> SkillDocument:
        try:
            return self._documents[name]
        except KeyError as exc:
            raise MissingBuiltinSkillError(f"unknown protein-design skill: {name}") from exc

    def select(self, names: Iterable[str]) -> tuple[SkillDocument, ...]:
        return tuple(self.require(name) for name in dict.fromkeys(names))

    def overlay_learned(
        self,
        learned: Iterable[LearnedSkill],
        *,
        max_skills: int = 8,
        max_chars: int = 12_000,
    ) -> "ProteinDesignSkillCatalog":
        merged = object.__new__(ProteinDesignSkillCatalog)
        merged._root = self._root
        merged._documents = dict(self._documents)
        accepted = 0
        for item in learned:
            name = self._safe_learned_name(item.name.strip(), item.native_id)
            if name in merged._documents:
                continue
            if not item.content.strip() or len(item.content) > max_chars or accepted >= max_skills:
                continue
            merged._documents[name] = SkillDocument(
                name=name,
                path=Path("memory") / item.native_id,
                content=item.content,
                missing_references=(),
                source="memory",
                roles=item.roles,
                native_id=item.native_id,
            )
            accepted += 1
        return merged

    @staticmethod
    def _safe_learned_name(name: str, native_id: str) -> str:
        """Map arbitrary learned-skill titles to stable, schema-safe local IDs."""
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            return name
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "memory"
        digest = hashlib.sha256(f"{native_id}\0{name}".encode()).hexdigest()[:10]
        return f"{slug[:48].rstrip('-')}-{digest}"

    def for_role(self, role: str, names: Iterable[str]) -> tuple[SkillDocument, ...]:
        selected: list[SkillDocument] = []
        for name in dict.fromkeys(names):
            document = self.require(name)
            if document.source == "builtin" or not document.roles or role in document.roles:
                selected.append(document)
        return tuple(selected)
