"""Context engine factory — one engine.

There is a single :class:`ContextAssembler`. It is assembled from:

- **The segment builders** — identity / bootstrap / project instructions /
  ``# Memory`` (``backend.recall(user_id=...)``) / always-skills /
  ``# Skills`` (a :class:`SkillForgeRouter` over one or two sources).
- **One history selector** — :class:`HistoryTrimmer`, deterministic, the only
  code path that decides which session messages reach the model.

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
from opendde_harness.context_engine.history_trimmer import HistoryTrimmer
from opendde_harness.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    ProjectInstructionsSegmentBuilder,
    SkillsSegmentBuilder,
)
from opendde_harness.providers.base import LLMProvider

if TYPE_CHECKING:
    from opendde_harness.config.opendde_harness import (
        ContextConfig,
        MemoryConfig,
        SkillForgeConfig,
        SkillForgeRouterConfig,
    )
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.memory_engine.skill_forge import SkillForgeRouter


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
) -> ContextAssembler:
    """Build the one :class:`ContextAssembler` from a flat SegmentBuilder list.

    ``builder`` is used only as the holder of the shared ``MemoryStore`` /
    ``LocalSkillCatalog`` until it is retired.
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

    # Either master switch off means no ``# Skills`` block at all: no router,
    # no catalogue, nothing advertised. ``None`` is what the segment reads as
    # "disabled", and it is the only thing that can disable it.
    skills_enabled = skill_forge_router_config.enabled and (
        skill_forge_config is None or bool(skill_forge_config.enabled)
    )
    router = (
        _build_router(
            builder=builder,
            backend=backend,
            memory_config=memory_config,
            skill_forge_router_config=skill_forge_router_config,
        )
        if skills_enabled
        else None
    )

    builders = [
        IdentitySegmentBuilder(workspace),
        BootstrapSegmentBuilder(workspace, assistant_dir=builder.storage.assistant),
        # The user's own AGENTS.md / ODH.md, read from the directory the tool
        # was launched in rather than from the workspace.
        ProjectInstructionsSegmentBuilder(workspace),
        MemorySegmentBuilder(
            builder.memory,
            backend,
            user_id=memory_config.user_id,
            memory_top_k=memory_config.memory_top_k,
        ),
        ActiveSkillsSegmentBuilder(builder.skills),
        SkillsSegmentBuilder(
            router,
            blocklist=(skill_forge_config.blocklist if skill_forge_config is not None else None),
        ),
    ]
    history = HistoryTrimmer(
        provider,
        model,
        get_tool_definitions,
        context_window_tokens,
        protect_first_n=config.protect_first_n,
    )
    return ContextAssembler(builders, get_tool_definitions, history, now_fn=now_fn)


def _build_router(
    *,
    builder: ContextBuilder,
    backend: "MemoryBackend | None",
    memory_config: "MemoryConfig",
    skill_forge_router_config: "SkillForgeRouterConfig",
) -> "SkillForgeRouter":
    """Assemble the one-or-two source SkillForgeRouter for segment 6."""
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
    # Only a backend with an agent-track store: the host's own markdown writer
    # has none, and a source that can only answer empty is a call per turn that
    # buys nothing.
    if backend is not None and getattr(backend, "agent_track", True):
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
