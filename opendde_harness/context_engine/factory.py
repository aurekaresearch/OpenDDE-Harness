"""Context engine factory — one engine.

There is a single :class:`ContextAssembler`. Per the context-builder
design it runs three lanes per turn (the prior ``legacy`` / ``curator`` /
``default`` split is gone):

- **Curator lane** — manifest build + fast / slow / fallback history
  selection + ``# Curator Working State``. Owns ``*history``.
- **Long-term memory lane** — ``backend.recall(user_id=...)`` (segment 3,
  ``# Memory``) and a :class:`SkillForgeRouter` over 1–3 sources (segment 5,
  ``# Skills``).
- **Host** — identity / bootstrap / always-skills, rendered by
  :class:`ContextBuilder`.

The SkillForgeRouter is assembled from up to two hardcoded sources:

- :class:`LocalSkillSource` — always; wraps the builder's existing
  ``LocalPool`` + ``SkillRegistry`` (no second disk scan).
- :class:`MemorySkillSource` — only when a ``backend`` is wired. Bridges
  ``backend.recall(agent_id=...)`` into the router.

With no ``backend`` the engine still constructs: the recall lane yields
``[]`` and the router runs Local-only, so the agent boots even when no
memory plugin is installed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from opendde_harness.agent.context import ContextBuilder
from opendde_harness.context_engine.assembler import ContextAssembler
from opendde_harness.context_engine.base import ContextEngine
from opendde_harness.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    SkillsSegmentBuilder,
)
from opendde_harness.context_engine.segments.curator import CuratorSegmentBuilder
from opendde_harness.providers.base import LLMProvider

if TYPE_CHECKING:
    from opendde_harness.config.opendde_harness import (
        ContextConfig,
        MemoryConfig,
        SkillForgeConfig,
        SkillForgeRouterConfig,
    )
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.memory_engine.skill_forge import (
        LLMGateFilter,
        QueryRewriter,
        SkillForgeRouter,
    )


def build_context_engine(
    *,
    workspace: Path,
    config: "ContextConfig",
    builder: ContextBuilder,
    provider: LLMProvider,
    model: str,
    context_window_tokens: int | None,
    get_tool_definitions: Callable[[], list[dict]],
    now_fn: Callable[[], datetime] | None = None,
    backend: "MemoryBackend | None" = None,
    memory_config: "MemoryConfig | None" = None,
    skill_forge_router_config: "SkillForgeRouterConfig | None" = None,
    skill_forge_config: "SkillForgeConfig | None" = None,
) -> ContextEngine:
    """Build the one :class:`ContextAssembler` from a flat SegmentBuilder list.

    ``config.engine`` is no longer a dispatch key — there is a single
    engine. The field is retained in :class:`ContextConfig` for config
    back-compat but is ignored here. ``builder`` is used only as the
    holder of the shared ``MemoryStore`` / ``LocalSkillCatalog`` until it
    is retired.
    """
    from opendde_harness.config.opendde_harness import (
        MemoryConfig as _MemoryConfig,
    )
    from opendde_harness.config.opendde_harness import (
        SkillForgeRouterConfig as _SkillForgeRouterConfig,
    )

    if memory_config is None:
        memory_config = _MemoryConfig()
    if skill_forge_router_config is None:
        skill_forge_router_config = _SkillForgeRouterConfig()

    router = _build_router(
        builder=builder,
        backend=backend,
        memory_config=memory_config,
        skill_forge_router_config=skill_forge_router_config,
    )

    rewriter, gate = _build_rewriter_and_gate(
        provider=provider,
        skill_forge_config=skill_forge_config,
        skill_forge_router_config=skill_forge_router_config,
    )

    builders = [
        IdentitySegmentBuilder(workspace),
        BootstrapSegmentBuilder(workspace),
        MemorySegmentBuilder(
            builder.memory,
            backend,
            user_id=memory_config.user_id,
            memory_top_k=memory_config.memory_top_k,
        ),
        ActiveSkillsSegmentBuilder(builder.skills),
        SkillsSegmentBuilder(
            router,
            skill_top_k=skill_forge_router_config.top_k,
            rewriter=rewriter,
            gate=gate,
            gate_pool_size=(
                int(getattr(skill_forge_config, "llm_gate_pool_size", 10)) if skill_forge_config is not None else 10
            ),
            get_tool_definitions=get_tool_definitions,
            blocklist=(getattr(skill_forge_config, "blocklist", None) if skill_forge_config is not None else None),
        ),
        CuratorSegmentBuilder(
            workspace=workspace,
            config=config,
            provider=provider,
            model=model,
            context_window_tokens=context_window_tokens,
            get_tool_definitions=get_tool_definitions,
            now_fn=now_fn,
        ),
    ]
    return ContextAssembler(builders, get_tool_definitions, now_fn=now_fn)


def _build_router(
    *,
    builder: ContextBuilder,
    backend: "MemoryBackend | None",
    memory_config: "MemoryConfig",
    skill_forge_router_config: "SkillForgeRouterConfig",
) -> "SkillForgeRouter":
    """Assemble the 1-to-2 source SkillForgeRouter for segment 5."""
    from opendde_harness.memory_engine.skill_forge import (
        LocalSkillSource,
        MemorySkillSource,
        SkillForgeRouter,
    )

    weights = skill_forge_router_config.weights or {}

    # ── Source 1: Local (always) ────────────────────────────────────
    # Reuse the builder's in-memory BM25 index / registry — no second
    # disk scan.
    local_source = LocalSkillSource(
        pool=builder.skills.pool,
        registry=builder.skills.registry,
    )
    if "local" in weights:
        local_source.weight = float(weights["local"])
    sources = [local_source]

    # ── Source 2: Memory (conditional on backend) ───────────────────
    if backend is not None:
        memory_source = MemorySkillSource(
            backend=backend,
            agent_id=memory_config.agent_id,
        )
        if "memory" in weights:
            memory_source.weight = float(weights["memory"])
        sources.append(memory_source)

    return SkillForgeRouter(
        sources=sources,
        over_fetch_factor=skill_forge_router_config.over_fetch_factor,
        dedup_by=skill_forge_router_config.dedup_by,
        rrf_k=skill_forge_router_config.rrf_k,
    )


def _build_rewriter_and_gate(
    *,
    provider: LLMProvider,
    skill_forge_config: "SkillForgeConfig | None",
    skill_forge_router_config: "SkillForgeRouterConfig",
) -> "tuple[QueryRewriter | None, LLMGateFilter | None]":
    """Construct the optional rewriter + gate from the parent SkillForge
    config. Both fall to ``None`` when their respective flag is off or
    no provider is wired — :class:`SkillsSegmentBuilder` then skips that
    stage.

    Per-stage isolation matters: gate-off + rewriter-on is a valid
    deployment (cheap retrieval, no LLM selector); rewriter-off + gate-on
    is also valid (always retrieve, then filter)."""
    if skill_forge_config is None or provider is None:
        return None, None

    from opendde_harness.memory_engine.skill_forge import LLMGateFilter, QueryRewriter

    rewriter: "QueryRewriter | None" = None
    if bool(getattr(skill_forge_config, "rewrite_enabled", False)):
        # No model: both helpers run on whatever the current provider serves by
        # default. Pinning the model the agent booted with outlived a live
        # ``/model`` switch, which then asked the new provider for the old
        # vendor's id ("LLM Provider NOT provided ... openai-codex/gpt-5.6-luna"
        # on a DeepSeek section) and dropped the stage to its fallback.
        rewriter = QueryRewriter(
            provider,
            max_tokens=int(getattr(skill_forge_config, "rewrite_max_tokens", 8192) or 8192),
        )

    gate: "LLMGateFilter | None" = None
    if bool(getattr(skill_forge_config, "llm_gate_enabled", False)):
        # ``legacy_top_k`` is the gate's failure-fallback size — must match
        # what the no-gate path renders (i.e. ``skill_top_k``) so a gate
        # outage doesn't change injection volume vs. having gate disabled.
        gate = LLMGateFilter(
            provider,
            max_select=int(getattr(skill_forge_config, "llm_gate_max_select", 2) or 2),
            legacy_top_k=int(skill_forge_router_config.top_k or 5),
            model=getattr(skill_forge_config, "llm_gate_model", None) or None,
            temperature=float(getattr(skill_forge_config, "llm_gate_temperature", 0.0)),
            max_tokens=int(getattr(skill_forge_config, "llm_gate_max_tokens", 8192) or 8192),
        )

    return rewriter, gate
