"""SkillForge — the local skill pool, the multi-source router, and its fusion.

The package is intentionally scoped narrow — every public symbol here is
consumed by the ``# Skills`` segment and the ``use_skill`` tool, and by
nothing else.

Key design point repeated for newcomers reading top-down: the
:class:`SkillSource` Protocol is **host-internal**. Per the
project-wide design decision, sources are hardcoded (Local +
Memory) and not a public plugin contribution point. Third-party
extension of skill retrieval happens via :class:`MemoryBackend`
(``backend.recall(agent_id=...)``) — the MemorySkillSource
re-emits those hits as :class:`RouterHit` records.

Skill selection is deterministic: no LLM decides what the catalogue
advertises. A small catalogue is advertised whole; a large one is
narrowed by BM25 over the user's message. Bodies are never injected —
the agent loads one through ``use_skill``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendde_harness.memory_engine.skill_forge.catalog import LocalSkillCatalog
from opendde_harness.memory_engine.skill_forge.fusion import RRF_K, rrf_merge_weighted
from opendde_harness.memory_engine.skill_forge.local_source import LocalSkillSource
from opendde_harness.memory_engine.skill_forge.memory_source import MemorySkillSource
from opendde_harness.memory_engine.skill_forge.refs import resolve_refs
from opendde_harness.memory_engine.skill_forge.router import SkillForgeRouter
from opendde_harness.memory_engine.skill_forge.types import RouterHit, SkillSource


def __getattr__(name: str):
    # Lightweight skill documents are also used by compute-only installations.
    # Do not import the host's BM25/tokenizer stack until a local pool is needed.
    if name == "LocalSkillCatalog":
        from opendde_harness.memory_engine.skill_forge.catalog import LocalSkillCatalog

        return LocalSkillCatalog
    raise AttributeError(name)


__all__ = [
    "MemorySkillSource",
    "LocalSkillCatalog",
    "LocalSkillSource",
    "RRF_K",
    "RouterHit",
    "SkillForgeRouter",
    "SkillSource",
    "resolve_refs",
    "rrf_merge_weighted",
]
