"""Immutable request-scoped skill catalog shared by workflow consumers.

Disk registries remain the source for host skills; this snapshot also holds
remote skill bodies without writing private memories into a global registry.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from opendde_harness.memory_engine.skill_forge.types import RouterHit


class SkillNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillDocument:
    name: str
    path: Path
    content: str
    missing_references: tuple[str, ...] = ()
    source: str = "builtin"
    roles: tuple[str, ...] = ()
    native_id: str | None = None

    @property
    def qualified_id(self) -> str:
        return f"memory/{self.native_id or self.name}" if self.source == "memory" else f"local/{self.name}"

    def as_hit(self) -> RouterHit:
        from opendde_harness.memory_engine.skill_local.registry import _parse_frontmatter

        frontmatter = _parse_frontmatter(self.content) or {}
        return RouterHit(
            self.qualified_id,
            self.name,
            self.content,
            0.0,
            meta={"source": self.source, "description": frontmatter.get("description", "")},
        )


class SkillDocumentCatalog:
    def __init__(self, documents: Iterable[SkillDocument] = ()) -> None:
        self._documents = {document.name: document for document in documents}

    def names(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def documents(self) -> tuple[SkillDocument, ...]:
        return tuple(self._documents.values())

    def require(self, name: str) -> SkillDocument:
        try:
            return self._documents[name]
        except KeyError as exc:
            raise SkillNotFoundError(f"unknown skill: {name}") from exc

    def select(self, names: Iterable[str]) -> tuple[SkillDocument, ...]:
        return tuple(self.require(name) for name in dict.fromkeys(names))

    def for_role(self, role: str, names: Iterable[str]) -> tuple[SkillDocument, ...]:
        return tuple(doc for doc in self.select(names) if doc.source == "builtin" or not doc.roles or role in doc.roles)
