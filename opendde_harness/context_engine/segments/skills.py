"""Segment 5 — ``# Skills``. SkillForge: rewriter → router → gate → render.

Four-step pipeline:

1. **Rewriter** (optional) — one LLM call that judges ``need_retrieval``
   and rewrites the query for skill routing. ``need_retrieval=False``
   short-circuits the rest of the build to an empty segment.
2. **Router fan-out** — :class:`SkillForgeRouter` queries Local +
   Memory in parallel and RRF-fuses to ``pool_size``.
3. **LLM gate** (optional) — one LLM call that picks 0..``max_select``
   hits from the pool. Empty result is a valid "inject nothing".
4. **Catalog render** — expose only names, descriptions, and qualified ids.
   The agent calls ``use_skill`` to load the selected body and any bundled
   resources. No skill instruction is pre-injected into the system prompt.

When rewriter / gate are not wired (provider missing, config off, etc.)
the pipeline degrades gracefully — both are independent and the segment
still produces a valid result with whatever is wired.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from opendde_harness.context_engine.base import AssemblyContext, Segment
from opendde_harness.context_engine.segments import render
from opendde_harness.memory_engine.skill_forge.catalog import is_blocked, normalize_blocklist
from opendde_harness.tracing import semconv, trace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opendde_harness.memory_engine.skill_forge import SkillForgeRouter
    from opendde_harness.memory_engine.skill_forge.gate import LLMGateFilter
    from opendde_harness.memory_engine.skill_forge.rewriter import QueryRewriter
    from opendde_harness.memory_engine.skill_forge.types import RouterHit
    from opendde_harness.providers.base import LLMProvider

log = logging.getLogger(__name__)


class SkillsSegmentBuilder:
    name = "skills"
    order = 5
    needs_prefix = False

    def __init__(
        self,
        router: "SkillForgeRouter | None",
        *,
        skill_top_k: int = 5,
        rewriter: "QueryRewriter | None" = None,
        gate: "LLMGateFilter | None" = None,
        gate_pool_size: int = 10,
        get_tool_definitions: "Any | None" = None,
        blocklist: "Iterable[str] | None" = None,
    ) -> None:
        self._router = router
        self._skill_top_k = skill_top_k
        self._rewriter = rewriter
        self._gate = gate
        # When the gate is active, the router selects ``gate_pool_size``
        # candidates (the gate then trims to ``max_select``). Without the
        # gate, ``skill_top_k`` controls direct injection size.
        self._pool_size = gate_pool_size if gate is not None else skill_top_k
        self._get_tool_definitions = get_tool_definitions
        self._blocklist = normalize_blocklist(blocklist)

    def set_provider(self, provider: "LLMProvider", model: str) -> None:
        """Hand a live ``/model`` switch down to the two LLM users in this
        segment. The router itself holds no provider."""
        if self._rewriter is not None:
            self._rewriter.set_provider(provider, model)
        if self._gate is not None:
            self._gate.set_provider(provider, model)

    @trace.instrument("skill.inject", kind="skill", detached=True, extract=semconv.skill_inject_skills)
    async def build(self, ctx: AssemblyContext) -> Segment | None:
        if self._router is None:
            return Segment(
                text="",
                meta={"injected_skill_ids": [], "skill_hits_by_source": {}},
            )

        query = ctx.current_message or ""

        # ── ① Rewriter ────────────────────────────────────────────────
        if self._rewriter is not None and query.strip():
            result = await self._rewriter.analyze(query)
            if not result.need_retrieval:
                return Segment(
                    text="",
                    meta={
                        "injected_skill_ids": [],
                        "skill_hits_by_source": {},
                        "rewriter_skipped": True,
                    },
                )
            if result.rewritten_query:
                query = result.rewritten_query

        # ── ② Router fan-out ─────────────────────────────────────────
        candidates = list(
            await self._router.select(
                query=query,
                history=ctx.session_messages,
                k=self._pool_size,
            )
        )

        # ── ②b Blocklist — hard drop across every source ──────────────
        if self._blocklist:
            kept: list["RouterHit"] = []
            for c in candidates:
                if is_blocked(self._blocklist, c.name, c.meta.get("skill_id")):
                    log.warning("dropping blocklisted skill from pool: %s", c.qualified_id)
                else:
                    kept.append(c)
            candidates = kept

        # ── ③ LLM gate ───────────────────────────────────────────────
        if self._gate is not None and candidates:
            tools = self._collect_tool_names()
            gated = await self._gate.filter(query, candidates, tools)
        else:
            gated = candidates[: self._skill_top_k]

        # ── ④ Render catalog ──────────────────────────────────────────
        body = render.render_router_skills(gated)
        meta: dict[str, Any] = {
            # No body was injected. Keep the old metadata key truthful so
            # memory feedback does not credit a merely advertised skill.
            "injected_skill_ids": [],
            "available_skill_ids": [h.qualified_id for h in gated if getattr(h, "qualified_id", None)],
            "skill_hits_by_source": dict(Counter((h.meta.get("source") or "?") for h in gated)),
        }
        text = f"# Skills\n\n{body}" if body else ""
        return Segment(text=text, meta=meta)

    def _collect_tool_names(self) -> list[str] | None:
        """Return tool names for the gate's hard-constraint block.

        ``get_tool_definitions`` is a callable injected at construction; when
        absent the gate runs without the tool-constraint hint (still
        works, just less aggressive at culling env-mismatched skills).
        """
        if self._get_tool_definitions is None:
            return None
        try:
            defs = self._get_tool_definitions()
        except Exception:
            return None
        names: list[str] = []
        for d in defs or []:
            if isinstance(d, dict):
                # OpenAI function-call schema → name lives under
                # ``function.name``; also accept a flat ``name``.
                fn = d.get("function") if isinstance(d.get("function"), dict) else None
                if fn and isinstance(fn.get("name"), str):
                    names.append(fn["name"])
                elif isinstance(d.get("name"), str):
                    names.append(d["name"])
        return names or None
