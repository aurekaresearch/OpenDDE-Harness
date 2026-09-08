"""Feature blocks of the OpenDDE Harness config: context, SkillForge, runtime,
tracing, plugins and memory.

Each block is a Pydantic model composed into the single root
:class:`opendde_harness.config.schema.Config`. Defaults are conservative:
every novel feature starts OFF so a fresh install behaves like the base
agent until features are enabled.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    """Accepts both camelCase and snake_case keys.

    ``extra='forbid'`` catches typos at startup: any key this release
    does not know is rejected, including keys retired releases wrote.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


# ---------------------------------------------------------------------------
# Feature 1 — Context Management (Curator)
# ---------------------------------------------------------------------------


class ContextConfig(_Base):
    """Context engine selection and tuning."""

    engine: str = "unified"
    """Deprecated — there is now a single :class:`ContextAssembler`.

    The historical ``"legacy"`` / ``"curator"`` / ``"default"`` split was
    collapsed: every turn runs the Curator history lane + the long-term
    memory recall / SkillForgeRouter lanes in one engine. The field is retained (as a
    free string) so existing YAML setting ``engine: legacy`` etc. still
    loads — the value is ignored by ``build_context_engine``.
    """

    # Curator history-lane knobs.
    fast_path_threshold: float = 0.60
    """Curator Fast Path cutoff. Below this % of budget → zero-LLM pass-through."""

    curator_model: str = "gemini-2.5-flash"
    """Model used by the Curator agent loop (Slow Path). Kept small & fast."""

    curator_timeout_seconds: float = 30.0
    """Max wall time for one Curator slow-path invocation before fallback."""

    relevance_decay: float = 0.95
    """Per-turn decay factor for non-recent message relevance."""

    relevance_reference_boost: float = 0.15
    """Boost applied when assistant response references older message content."""

    protect_first_n: int = 3
    """Number of head exchanges always preserved in context."""

    archive_dir: str = "memory/.curator/archive"
    """Relative path under workspace for lossless message archives."""


# ---------------------------------------------------------------------------
# Feature 4 — SkillForge
# ---------------------------------------------------------------------------
#
# SkillForge owns retrieval + execution + feedback emission. Evolution
# is handled by the long-term memory service's extraction pipeline.
#
# The config is intentionally kept flat. Component-level knobs
# (embedding model, BM25 parameters, RRF k, etc.) live in the
# scaffold dataclasses inside ``skill_forge/`` and stay at their
# defaults for now. Owners will promote individual fields here when
# they need user-facing knobs.


class MemoryExtractionConfig(_Base):
    """Long-term memory extraction pipeline configuration (``skillForge.memory``).

    When enabled, every completed user→agent turn is funneled into the
    memory service's pipeline that distills an AgentCase + zero-or-more
    SkillOps from it.
    """

    enabled: bool = False
    # Note: the per-turn tool-call gate (formerly min_tool_calls / min_messages
    # here) is now sourced from skill_forge.detect_min_tool_calls so the same
    # threshold drives any future auto-detect surface in addition to this
    # pipeline.
    # Number of similar existing skills shown to the skill_extractor
    # LLM as candidates for ``update``. 5 is enough — overlap between
    # turn-derived candidates above this rank is rare, and the prompt
    # budget for supporting_cases scales with this number.
    max_skills_top_k: int = 5
    # Confidence floor: skills falling below this after a downward
    # adjustment are soft-deleted on the spot.
    retire_confidence: float = 0.1
    # Skip the skill_extractor LLM call when ``case.quality_score`` is
    # below this floor. Low-quality distillations tend to produce noisy
    # / contradictory skills more often than reusable ones; the case is
    # still persisted (useful for retrieval / audit).
    min_quality_for_skill_extract: float = 0.2
    # 3-tier value gate placed before case extraction (in _flush_segment).
    # Only segments that pass at least one tier are extracted:
    #   Tier 1 (fast-pass): has_user_feedback AND >=2 user messages in segment
    #   Tier 2 (fast-pass): total tool_calls > complex_task_tool_call_threshold
    #   Tier 3 (cheap LLM): detect_llm asked whether trajectory is worth
    #                        learning from; false → skip, true → extract
    complex_task_tool_call_threshold: int = 20


class LocalDirConfig(_Base):
    """One local skill directory entry (R1)."""

    path: str
    """Absolute or ``~``-relative path. Expanded at startup."""

    enabled: bool = True
    """False → directory completely skipped."""

    name: str | None = None
    """Display name for logs. None → derived from path basename."""

    always_enabled: bool = True
    """False → skills from this dir with ``always: true`` are excluded
    from always injection (but still retrievable via select)."""


class SkillForgeConfig(_Base):
    """SkillForge configuration.

    ``enabled=True`` (default, R8) activates the SkillForge retrieval/
    injection pipeline. Set ``enabled=False`` to fall back to the
    pre-refactor behavior of handing the full skill directory to the LLM
    (component stubs that return empty lists also cause ``ContextBuilder``
    to fall back to the full directory automatically).

    Evolution is handled by the long-term memory
    extraction pipeline, configured via
    ``skill_forge.memory``. The LLM used by that pipeline
    is selected by ``skill_forge.evolve_model`` (falls back to the
    active agent model when unset).

    Other lifecycle fields (auto_detect / auto_evolve / retirement ...)
    are placeholders from the original spec. No local code reads them;
    preserved for now to avoid breaking user configs.
    """

    # --- Master switch + location ---
    enabled: bool = True
    """Master switch (R8: default True). Activates the SkillForge
    retrieval/injection pipeline."""

    blocklist: list[str] = Field(default_factory=list)
    """Skill names refused everywhere (config key ``skillForge.blocklist``):
    dropped from the injection pool for every source and refused by
    ``use_skill``. Matched case-insensitively against the skill's name,
    slug, and native id."""

    router: "SkillForgeRouterConfig" = Field(
        default_factory=lambda: SkillForgeRouterConfig(),
    )
    """Multi-source RRF routing policy (weights / over-fetch / dedup)
    — config key ``skillForge.router``. The
    router is a component of the SkillForge subsystem, so it nests here
    rather than living as a sibling top-level block. Forward-ref +
    ``model_rebuild`` (below): ``SkillForgeRouterConfig`` is defined later
    in this module."""

    local_dirs: list[LocalDirConfig] = Field(default_factory=list)
    """Local skill directories to mount (R1). List order = priority:
    later entries override earlier on name collision."""

    scan_max_depth: int = 5
    """Maximum directory depth when scanning for SKILL.md files (R2).
    Paths deeper than this below a layer root are silently skipped.
    Prevents unbounded filesystem walks on huge mirrors."""

    # --- Retrieval / reranker knobs ---
    embedding_model: str = "default"
    """Dense embedding model identifier used by the configured embedding
    service. Configure it to match the model used by the active retrieval
    source."""

    embedding_url: str = "http://localhost:1357"
    """Remote embedding service base URL.

    Retrieval calls ``POST <embedding_url>/embed``. Override this with
    ``REMOTE_EMBEDDING_URL`` or user config when using a hosted embedding
    service."""

    reranker_enabled: bool = True
    """Run a reranker pass after dense retrieval. On by default — adds
    200-500ms per query (cross-encoder GPU inference) but lifts mass-pool
    precision noticeably. Disable when latency matters more than ranking."""

    reranker_model: str = "default"
    """Reranker model label used for configuration and observability."""

    reranker_url: str = "http://localhost:1357"
    """Remote reranker service base URL.

    Reranking calls ``POST <reranker_url>/score`` with
    ``{"prompts": [...]}`` and reads ``{"scores": [...]}``. Override this
    with ``REMOTE_RERANKER_URL`` or user config when using a hosted reranker
    service."""

    embedding_api_key: str | None = None
    """Optional bearer token for the configured embedding service."""

    reranker_api_key: str | None = None
    """Optional bearer token for the configured reranker service."""

    embedding_dimensions: int | None = None
    """Request specific embedding dimensions (for models that support it)."""

    top_k: int = 5
    """Number of skills returned by ``select()``."""

    # --- Dual-pool fusion weights (R6) ---
    local_pool_top_k: int = 10
    """Candidate count from the local BM25 pool per query."""

    mass_pool_top_k: int = 10
    """Candidate count from the mass dense pool per query (post-rerank)."""

    local_weight: float = 1.3
    """RRF weight for local-pool candidates (mass is implicitly 1.0).
    Recommended range [1.2, 1.5]. Values < 1.0 or > 2.0 are rejected."""

    mass_reranker_overfetch: int = 20
    """When reranker is enabled, mass pool fetches this many candidates
    for rescoring, then truncates to ``mass_pool_top_k`` before RRF."""

    # --- Query rewrite knobs ---
    rewrite_enabled: bool = True
    """Enable a second retrieval path with LLM-rewritten queries."""

    rewrite_max_tokens: int = 8192
    """Output token budget for the rewriter LLM call. Defaults to 8192 to
    leave headroom for Qwen3-style reasoning traces (~3-4k tokens) on top
    of the actual rewrite output. The previous 1024 budget caused frequent
    finish_reason=length truncations with empty visible content, which
    surfaced as 'Failed to parse rewrite response as JSON' fallbacks."""

    # --- Deprecated presentation knobs ---------------------------------
    injection_mode: str = "catalog"
    """Deprecated compatibility field; selected skills are always rendered
    as a metadata-only catalog and loaded through ``use_skill``. Historical
    ``full_body`` / ``summary`` values are accepted when reading older user
    configs but no longer alter runtime behavior."""

    inject_max: int = 2
    """Deprecated compatibility field. ``llm_gate_max_select`` controls the
    catalog size; no skill body is inlined."""

    disable_always: bool = False
    """When True, ``get_always_skills()`` returns [] and select() filters
    out always:true skills. R8 default: False (always skills inject)."""

    always_max: int = 5
    """Max always skills injected per turn (R3). Exceeding this truncates
    by local_dirs list order + alphabetical, with a WARN listing dropped
    skill names."""

    # --- LLM gate selector (default-on, mirrors openspace select_skills_with_llm) ---
    llm_gate_enabled: bool = True
    """When ``True`` (default), ``select()`` resolves a pool of
    ``llm_gate_pool_size`` candidates after RRF merge, then asks an LLM to
    plan + filter down to ``llm_gate_max_select`` skills. Empty result is
    valid ("inject nothing"). Costs one LLM call per ``select()`` invocation
    but eliminates the ~30% noise-injection rate of pure-RRF top-K (Round D
    obs.: irrelevant skills polluting the prompt). Disable to skip the
    extra LLM call (rare; useful when LLM provider is unavailable)."""

    llm_gate_max_select: int = 2
    """Upper bound on skills advertised in the per-turn catalog."""

    llm_gate_pool_size: int = 10
    """Candidate pool size handed to the gate (after RRF). Aligned
    with RRF output size (local_pool_top_k + mass_pool_top_k dedupe)."""

    llm_gate_model: str | None = None
    """Optional model override for gate calls. ``None`` → use the
    provider's default chat model (typically the agent's main model)."""

    llm_gate_temperature: float = 0.0
    """Sampling temperature for gate calls. 0.0 for deterministic
    filtering. Reasoning models may need 0.6 to engage <think>."""

    llm_gate_max_tokens: int = 8192
    """Output token budget for the gate LLM call. Defaults to 8192 to
    leave headroom for Qwen3-style reasoning traces (~3-4k tokens) on top
    of the gate's JSON answer. The previous 4096 budget caused empty
    content (finish_reason=length) on the 27B model in ~50% of calls,
    forcing a legacy top-N fallback that returned 5 skills instead of
    the configured llm_gate_max_select."""

    # --- Evolver model ---
    evolve_model: str | None = None
    """LLM used by the long-term memory evolver for case
    distillation and skill rewrites. When ``None`` (default), the evolver
    falls back to the active agent model (``agents.defaults.model`` /
    provider default). Set explicitly to pin a stronger model for quality
    rewrites — e.g. ``"claude-opus-4-6"``."""

    # --- Detect / extraction gating (wired into memory extraction) ---
    detect_model: str = "gemini-2.5-flash"
    """LLM used for the cheap per-turn classification work — today that's
    the memory service's boundary detector (multi-turn task split). A
    smaller / faster model than ``evolve_model`` is intentional: boundary
    detection runs on every accumulated turn pair, while the heavier
    extractors only run when a segment is actually flushed."""

    detect_min_tool_calls: int = 3
    """Minimum tool calls in the current turn for it to enter the
    extraction pipeline at all. Coding work worth replaying almost always
    exercises ≥ this many tools; thinner turns get filtered before any
    LLM is invoked. Set to 0 to disable the gate."""

    # --- Legacy placeholders (not wired; see class docstring) ---
    stats_tracking: bool = True
    """Record per-skill invocation stats. Cheap, enables future features."""

    auto_detect: bool = False
    """End-of-session LLM check for new skill candidates."""

    auto_evolve: bool = False
    """Automatic skill improvement based on feedback. Requires auto_detect."""

    evolve_trigger_success_rate: float = 0.70
    """Evolution fires when success_rate drops below this over recent invocations."""

    evolve_trigger_min_invocations: int = 10
    """Don't evolve skills used fewer than this many times."""

    draft_first_activation: bool = True
    """New auto-created skills start as 'draft'; promoted to 'active' after first success."""

    retirement_idle_days: int = 90
    """Active skill unused for this long → deprecated."""

    # --- Long-term memory extraction pipeline ---
    memory: MemoryExtractionConfig = Field(default_factory=MemoryExtractionConfig)
    """Long-term memory extraction pipeline. Distinct from the
    SkillForge master switch above: the retrieval/injection path can be
    enabled (``skill_forge.enabled=True``) without extraction, and vice
    versa."""

    # --- Validators ---

    @model_validator(mode="before")
    @classmethod
    def _check_local_weight(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        lw = data.get("local_weight") or data.get("localWeight")
        if lw is not None:
            lw = float(lw)
            if lw < 1.0 or lw > 2.0:
                raise ValueError(f"local_weight={lw} out of valid range [1.0, 2.0]")
        return data


# ---------------------------------------------------------------------------
# CFG-1 — Plugin / Memory backend / SkillForgeRouter
# ---------------------------------------------------------------------------


class PluginsConfig(_Base):
    """Plugin-system top-level config.

    ``disabled`` is the user opt-out list keyed by plugin id (matches
    the ``id`` in ``opendde-harness-plugin.toml``). ``config`` is the per-
    plugin config slice the registry hands to each plugin's factory
    via :class:`PluginContext.config` — the host treats it as a
    free-form dict and never validates it, so each plugin is
    responsible for reading and defaulting its own keys.
    """

    disabled: list[str] = Field(default_factory=list)
    """Plugin ids the user opted out of (e.g. ``["long-term-memory"]``)."""

    config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-plugin configuration, keyed by plugin id. Each plugin's
    factory receives ``ctx.config = plugins.config.get(<id>, {})``."""


class MemoryConfig(_Base):
    """Which memory backend is active + per-track identity wiring.

    ``backend`` is the name of an activated ``memory_backend``
    contribution (set ``None`` to disable backend-driven memory and
    operate purely on opendde-core's MemoryStore + MemoryConsolidator).

    The two id fields are bare, backend-native strings. The host passes
    ``user_id`` for the user-track recall and ``agent_id`` for the
    agent-track recall (``backend.recall`` takes one XOR the other).
    The bundled long-term backend routes each to its matching store; flat backends (mem0 /
    MemOS / Letta) use ``user_id`` and return empty for the agent call.
    These two fields are the only place either id is configured. A
    backend receives them through ``ctx.services`` and must not read an
    id from its own config slice: a second place holding the same value
    lets a user edit one of them and silently split writes from reads,
    after which every stored memory is unrecallable with no warning.
    """

    backend: str | None = "longterm"
    """Activated backend contribution name. ``None`` disables the
    plugin-driven memory path; AgentLoop continues with opendde-core's
    MemoryStore alone."""

    user_id: str = "default"
    """Bare user identity passed as ``backend.recall(user_id=...)`` for
    the user-track recall channel inside ``ContextAssembler.assemble``."""

    agent_id: str = "default"
    """Bare agent identity passed as ``backend.recall(agent_id=...)`` by
    ``MemorySkillSource`` for agent-track skill recall."""

    memory_top_k: int = 5
    """Top-K passed to ``backend.recall(user_id=user_id)`` per turn for
    the ``# Recalled memory`` block."""


class SkillForgeRouterConfig(_Base):
    """Multi-source skill routing policy.

    Sources themselves are hardcoded (Local + Memory) per the
    project-wide design decision; this block tunes the weighted RRF
    and per-source plumbing.
    """

    enabled: bool = True
    """Master switch. ``False`` makes the host bypass SkillForgeRouter
    entirely (used by tests / restricted deployments)."""

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "local": 0.96,
            "memory": 0.9,
        },
    )
    """Per-source RRF weight. Higher = more rank mass when the same skill
    surfaces from multiple sources. Local highest (hand-curated); Memory
    lower (task-specific, auto-evolved).

    Only the ratios matter -- scaling both leaves the order unchanged.
    Read them together with ``rrf_k``: the spread has to stay well inside
    the rank ladder that ``rrf_k`` produces, or weight silently overrides
    rank and each source becomes a strict tier."""

    rrf_k: int = Field(default=10, ge=1)
    """RRF damping constant, mirroring ``skill_forge.fusion.RRF_K``.
    Lower = source-internal rank carries more weight relative to
    ``weights``; higher = flatter, so cross-source agreement and source
    identity dominate."""

    over_fetch_factor: int = 2
    """Each source is asked for ``top_k * factor`` hits before fusion
    narrows back to ``top_k``. Larger factors give better cross-source
    coverage at the cost of per-source query work."""

    dedup_by: Literal["name", "qualified_id"] = "name"
    """Cross-source dedup key for the RRF fusion. ``"name"`` collapses
    a same-named skill across sources into one slot; ``"qualified_id"``
    keeps them as separate entries (useful for telemetry experiments)."""

    top_k: int = 5
    """Final top-K returned from ``SkillForgeRouter.select``."""


# Resolve the forward-ref ``SkillForgeConfig.router: "SkillForgeRouterConfig"``
# now that ``SkillForgeRouterConfig`` exists in module scope.
SkillForgeConfig.model_rebuild()


# ---------------------------------------------------------------------------
# Feature 5 — Runtime Discipline
# ---------------------------------------------------------------------------


class CheckpointConfig(_Base):
    """Per-turn shadow-git checkpoint of the workspace.

    When active, the agent loop commits the workspace to an out-of-band
    shadow git repo at the end of each turn (covering both normal and
    max-iteration exits). This is the safety net behind Bug2: a truncated
    multi-file edit leaves a recoverable snapshot, and the next turn gets a
    recovery prompt listing what the interrupted turn changed.

    Activation is gated by ``policy`` and the AgentLoop's ``interactive``
    flag (set per call site by the CLI / TUI / gateway entry points):

    - ``"always"``     — active in every AgentLoop, including ``-m``
                          one-shot commands.
    - ``"interactive"`` — active only when constructed for a multi-turn
                          session (REPL, TUI, gateway). One-shot commands
                          have no "next turn" to inject recovery into, so
                          paying the snapshot cost there is wasted.
    - ``"never"``      — disabled entirely; loop is byte-identical to the
                          pre-Bug2 baseline (no commits, no interrupt
                          reclassification, no recovery injection).

    Default ``"interactive"`` matches mature competitors (Claude Code,
    Cursor) which transparently checkpoint long sessions while leaving
    one-shot batch invocations untouched.
    """

    policy: Literal["always", "interactive", "never"] = "interactive"
    """When the per-turn shadow-git snapshot is active. See class
    docstring for the interaction with the AgentLoop ``interactive`` flag."""

    shadow_dir: str = ".opendde_harness/shadow.git"
    """Shadow git-dir, relative to the workspace. The real workspace is the
    work-tree; the user's own ``.git`` is never touched."""


class RuntimeConfig(_Base):
    """Runtime discipline — the 5th feature pillar.

    Houses the opt-in runtime safety nets. Bug2 ships ``checkpoint``;
    later phases add ``journal`` / ``verifier`` / ``done_gate`` /
    ``loop_detection`` (Bug3, us) and ``session`` (Bug1, dev) as sibling
    sub-configs. All default off so the all-off baseline equals 68a3be7.
    """

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class TracingConfig(_Base):
    """Observability tracing (in-tree ``opendde_harness.tracing``).

    On by default; every ``opendde`` command auto-installs non-invasive
    instrumentation before any AgentLoop is built. ``OPENDDE_HARNESS_TRACING=0`` is an
    explicit env kill-switch that overrides this block. View captured traces
    with ``ddeharness tracing`` (or ``/tracing`` in the TUI).
    """

    enabled: bool = True
    port: int = 4318
    preview_len: int = 500
