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
# Feature 1 — Context Management
# ---------------------------------------------------------------------------


class ContextConfig(_Base):
    """Context engine tuning.

    Two settings: when a backend that compacts server-side should do so, and
    how much of the head of a conversation is never dropped. History selection
    itself is deterministic and has no knobs -- the policy is in
    :mod:`opendde_harness.context_engine.history_trimmer`.
    """

    server_compact_ratio: float = Field(default=0.8, ge=0.0, le=1.0, allow_inf_nan=False)
    """On a provider that compacts server-side (the Codex login), compact the
    session once a call's prompt reaches this share of the model's window; the
    backend's own opaque summary then replaces the earlier history on that
    model. 0 disables it.

    A share, so it is bounded at load: a negative one or a NaN compares false
    against every prompt size and asked the backend to compact on every single
    turn, and one above 1 (or an infinity) quietly made the trigger
    unreachable. Both read as a working setting.
    """

    protect_first_n: int = 3
    """How many of the conversation's first user messages are never dropped.

    The user's own framing of the task, which every later turn is judged
    against -- not the first few tool-heavy turns."""


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

    Which skills the agent is told about, and where they are read from.
    The choice itself takes no model call: a small catalogue is
    advertised whole, a large one is narrowed by BM25 over the user's
    message, and the agent loads a body with ``use_skill``.

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
    """Master switch. ``False`` drops the ``# Skills`` block entirely: no
    router is built, nothing is advertised, and configured ``local_dirs``
    are not mounted. ``skillForge.router.enabled=False`` does the same."""

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

    disable_always: bool = False
    """When True, ``get_always_skills()`` returns [] and the ``# Active
    Skills`` block is empty. Default False (always-skills are listed)."""

    always_max: int = 5
    """Max always skills injected per turn (R3). Exceeding this truncates
    by local_dirs list order + alphabetical, with a WARN listing dropped
    skill names."""

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

    ``backend`` names the activated ``memory_backend`` contribution that owns
    automatic durable extraction. Unset -- the default -- the owner is the host's
    own markdown writer, which needs no service installed: it annotates each
    completed turn into ``episodes.md`` and rewrites the ``user.md`` section
    behind a tag that has heated up. Naming a backend here replaces that writer
    rather than joining it; two writers on one profile is the state this setting
    exists to prevent.

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

    backend: str | None = None
    """Who owns automatic durable extraction. Three kinds of value:

    - unset (``None``), the default: the host's own writer
      (``HostMarkdownBackend``). The turns a session completes are annotated into
      ``episodes.md`` and the profile sections behind a hot tag are rewritten in
      ``user.md``. Nothing to install.
    - ``"off"``: no owner at all. Nothing is extracted, no model call is made for
      memory, and neither file is written; ``user.md`` is still read into the
      prompt exactly as whoever maintains it by hand left it.
    - any other value: the name of an activated ``memory_backend`` contribution,
      which replaces the host writer rather than joining it.

    ``"off"`` is a written word and not the absence of one on purpose. A config
    that says nothing about memory is a config whose author never chose, and that
    one gets the default owner."""

    foresight: bool = False
    """Ask the annotator for predictions as well as episodes, and keep them in
    ``user.md``'s ``## Foresight`` section.

    Off by default, and deliberately: with it off the annotation tool has one
    slot and the model is not asked to guess at all, which is both cheaper and
    the only version of this whose output is checkable against what was said.
    Only the host writer reads it; a plugin backend has its own policy."""

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
    entirely, which leaves the ``# Skills`` block out of the prompt (used
    by tests / restricted deployments). Same effect as
    ``skillForge.enabled=False``."""

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
    """Each source is asked for ``k * factor`` hits before fusion narrows back
    to ``k`` (the segment's own constant). Larger factors give better
    cross-source coverage at the cost of per-source query work."""

    dedup_by: Literal["name", "qualified_id"] = "name"
    """Cross-source dedup key for the RRF fusion. ``"name"`` collapses
    a same-named skill across sources into one slot; ``"qualified_id"``
    keeps them as separate entries (useful for telemetry experiments)."""


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

    shadow_dir: str = "shadow.git"
    """Shadow git-dir, relative to instance state/<scope>/checkpoint. The workspace is the
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
