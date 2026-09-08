"""SkillForgeRouter machinery — multi-source skill retrieval + RRF fusion.

SR-1 lands the type contract and the first concrete source
(:class:`LocalSkillSource`) plus the rendering helper
(:class:`LocalSkillCatalog`). SR-2 adds the router + weighted RRF;
SR-3/SR-4 add the Memory source. The package is intentionally
scoped narrow — every public symbol here is consumed by
:class:`DefaultContextEngine` (lands in CE-1) and not by anything else.

Key design point repeated for newcomers reading top-down: the
:class:`SkillSource` Protocol is **host-internal**. Per the
project-wide design decision, sources are hardcoded (Local +
Memory) and not a public plugin contribution point. Third-party
extension of skill retrieval happens via :class:`MemoryBackend`
(``backend.recall(agent_id=...)``) — the MemorySkillSource
re-emits those hits as :class:`RouterHit` records.
"""

from __future__ import annotations

from opendde_harness.memory_engine.skill_forge.catalog import LocalSkillCatalog
from opendde_harness.memory_engine.skill_forge.fusion import RRF_K, rrf_merge_weighted
from opendde_harness.memory_engine.skill_forge.gate import LLMGateFilter
from opendde_harness.memory_engine.skill_forge.local_source import LocalSkillSource
from opendde_harness.memory_engine.skill_forge.memory_source import MemorySkillSource
from opendde_harness.memory_engine.skill_forge.refs import resolve_refs
from opendde_harness.memory_engine.skill_forge.rewriter import (
    QueryRewriter,
    RewriteResult,
)
from opendde_harness.memory_engine.skill_forge.router import SkillForgeRouter
from opendde_harness.memory_engine.skill_forge.types import RouterHit, SkillSource

__all__ = [
    "MemorySkillSource",
    "LLMGateFilter",
    "LocalSkillCatalog",
    "LocalSkillSource",
    "QueryRewriter",
    "RRF_K",
    "RewriteResult",
    "RouterHit",
    "SkillForgeRouter",
    "SkillSource",
    "resolve_refs",
    "rrf_merge_weighted",
]
