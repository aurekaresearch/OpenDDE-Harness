"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack, aclosing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from opendde_harness import __logo__
from opendde_harness.agent.context import ContextBuilder
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.loop.failure_streak import (
    failure_class,
    is_hard_tool_failure,
    loop_break_nudge,
)
from opendde_harness.agent.loop.recovery import (
    POST_TOOL_NUDGE,
    RecoveryAction,
    classify_empty_response,
    is_only_think_debris,
    strip_think_blocks,
)
from opendde_harness.agent.subagent import SubagentManager
from opendde_harness.agent.tools.ask_user import AskUserTool
from opendde_harness.agent.tools.file_search import FindTool, GrepTool
from opendde_harness.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from opendde_harness.agent.tools.registry import ToolRegistry
from opendde_harness.agent.tools.shell import ExecTool
from opendde_harness.agent.tools.spawn import SpawnTool
from opendde_harness.agent.tools.web import WebFetchTool, WebSearchTool
from opendde_harness.memory_engine.base import TokenBudget
from opendde_harness.memory_engine.consolidate.consolidator import MemoryConsolidator
from opendde_harness.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    RunMeta,
    ToolCallRequest,
    format_llm_error,
    send_max_tokens,
)
from opendde_harness.providers.capabilities import image_placeholder_text, supports_image_tool_result, vision_verdict
from opendde_harness.providers.catalog import overlay_for
from opendde_harness.providers.rates import Resolved, resolve_context_window
from opendde_harness.providers.reasoning import split_orphan_think
from opendde_harness.providers.truncation import flag_truncation
from opendde_harness.sandbox import DirectExecutor, SandboxExecutor, SandboxInitError
from opendde_harness.session.manager import Session, SessionManager
from opendde_harness.spine.turn import Origin
from opendde_harness.tracing import semconv, trace
from opendde_harness.utils.helpers import estimate_prompt_tokens, is_image_part, is_inline_image

# How long a turn is willing to wait on plugin-side indexing before letting the
# write finish on its own. A budget, not a deadline: the task keeps running.
_STORE_TURN_BUDGET_S: float = 5.0
# Outstanding detached writes past which a turn waits for one to land, so a slow
# memory service cannot grow an unbounded queue behind a fast typist.
_STORE_MAX_INFLIGHT: int = 4
# Teardown's total budget for letting those writes finish.
_STORE_DRAIN_BUDGET_S: float = 15.0

_ABORTED_ACTION_REPLY = (
    "The operation was not completed, and no alternative method will be attempted. "
    "Would you like me to continue with the remaining parts of the task that do not "
    "require this operation?"
)

# NOTE: ``opendde_harness.context_engine`` is intentionally imported lazily (inside
# ``__init__`` and ``_assemble_context_messages``) to break a runtime
# import cycle: ``opendde_harness.agent.__init__`` eagerly loads AgentLoop,
# while ``opendde_harness.context_engine.curator`` imports ``ContextBuilder`` from
# ``opendde_harness.agent.context`` — a module-level top-down ``from
# opendde_harness.context_engine import ...`` here re-enters a partially-initialized
# package and raises ImportError on ``TurnContext``.

if TYPE_CHECKING:
    from opendde_harness.agent.hook import CompositeHook
    from opendde_harness.agent.tools.base import Tool
    from opendde_harness.context_engine import ContextEngine
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.providers.registry import ProviderSpec
    from opendde_harness.spine.runner import Drain, Emit
    from opendde_harness.spine.runner import TurnOutcome as SpineTurnOutcome
    from opendde_harness.spine.turn import TurnRequest
    from opendde_harness.token_wise.base import UsageSnapshot
    from opendde_harness.token_wise.registry import StrategyRegistry


@dataclass
class LoopOutcome:
    """Result of one ``_run_agent_loop`` pass beyond its text reply; the
    spine's ``LoopOutcome`` is what ``run_turn`` hands back.

    ``status`` distinguishes a normal completion from a max-iteration
    interruption or an LLM error — so the caller never mistakes "ran out of
    budget" for "done" (Bug2 / decision B). ``checkpoint_id`` and
    ``edited_files`` carry the shadow-git snapshot info used to build the
    next turn's recovery prompt.
    """

    status: str = "completed"  # "completed" | "interrupted" | "error"
    checkpoint_id: str | None = None
    edited_files: list[str] = field(default_factory=list)
    used_skill_ids: list[str] = field(default_factory=list)
    # The last model call's accounting: prompt/completion/total tokens,
    # cost_usd and context_max/context_used/context_percent.
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnReply:
    """What ``_process_message`` hands back. ``content`` is ``None`` for a
    silent turn (the message tool already delivered the reply)."""

    content: str | None
    media: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def _filter_qualified_ids(
    ids: list[str] | None,
    source_prefix: str,
) -> list[str]:
    """FB-1 helper: extract native ids from a list of qualified ids
    matching ``<source_prefix>/<native>``.

    Returns the bare native portion for each match (i.e. strips the
    ``"<source>/"`` prefix) so the receiving backend doesn't have to
    re-parse. Non-matching / unprefixed / malformed entries silently
    drop. ``None`` and empty inputs return ``[]``.
    """
    if not ids:
        return []
    needle = f"{source_prefix}/"
    out: list[str] = []
    for qid in ids:
        if not isinstance(qid, str):
            continue
        if qid.startswith(needle):
            native = qid[len(needle) :]
            if native:
                out.append(native)
    return out


# Asks the model for a best-effort wrap-up after the iteration budget is spent.
# Tools are withheld on this call, so the prompt must not invite another tool
# use or a question — there is no further turn to answer it.
_MAX_ITER_SYNTHESIS_PROMPT = (
    "You've used up the tool-calling budget for this turn, so no tools are "
    "available now. Using only what you've already gathered, give your best "
    "final answer: summarize what you accomplished, deliver any partial "
    "results, and briefly note what's left undone. Do not ask questions — "
    "there is no further turn to answer them. Reply in the same language as "
    "the user's request (this instruction is in English, but it is not the "
    "conversation language)."
)

# Returned only if the synthesis call itself fails — never leave the turn silent.
_MAX_ITER_STATIC_FALLBACK = (
    "I reached the maximum number of tool call iterations ({n}) without "
    "completing the task. You can try breaking the task into smaller steps."
)

# Origins whose turns skip the user-inbound hooks: a subagent result
# re-injection is not genuine user input.
_SKIP_USER_INBOUND_ORIGINS = frozenset({Origin.SUBAGENT})

# Origins whose reply skips the ``after_send`` chain: the subagent announce is
# system-originated and must not be modified on the way out.
_SKIP_AFTER_SEND_ORIGINS = frozenset({Origin.SUBAGENT})

# Marks the synthetic user message that carries images a transport cannot put in
# a tool result. Not persisted: the tool result above it already names the file
# path, so the only thing this message would add to the transcript is a user turn
# saying "[image]" that the user never sent -- misleading on resume and in
# session export. Deliberately a different key from ``_recovery_synthetic``:
# that one marks empty-response recovery scaffolding, and collapsing the two
# would make either meaning impossible to reason about separately.
_ATTACHED_IMAGE_KEY = "_attached_image"


def _strip_inline_images(content: list[Any]) -> list[Any]:
    """Replace inline base64 images with a text placeholder, for persistence.

    Images live for exactly the turn that produced them. Keeping the bytes would
    bloat the session JSONL by megabytes per picture, and every later turn would
    replay them to the model — paying for an image nobody asked about again.

    A *new* list is returned: the input is the live message the model is still
    working from this turn, and `_save_turn` only shallow-copies the entry, so
    mutating in place would pull the picture out from under the current request.
    """
    out: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            out.append(part)
            continue
        if is_inline_image(part):
            out.append({"type": "text", "text": "[image]"})
        else:
            out.append(part)
    return out


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the spine
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    # Max emergency context shrinks per turn before a context overflow is fatal.
    _MAX_COMPRESS_RETRIES = 2
    # Max image demotions per turn. One is enough: a refusal is deterministic for
    # the model, and the first retry also caches the verdict, so a second attempt
    # would mean the failure was never about images.
    _MAX_IMAGE_DEMOTE_RETRIES = 1
    # Most recent tool results kept intact when emergency-shrinking; older ones
    # are elided (their bodies are the bulk of mid-turn context growth).
    _SHRINK_KEEP_RECENT_TOOL_RESULTS = 3
    # Image-bearing messages kept intact when emergency-shrinking. Tighter than
    # the tool-result count because one image can cost 1568 tokens: the picture
    # the model is currently reasoning about is worth keeping, older ones are the
    # cheapest thing to give up.
    _SHRINK_KEEP_RECENT_IMAGES = 1
    # Tool-failure-loop break: nudge after the same tool fails deterministically
    # this many times running; cap the nudges per turn so it can't itself loop.
    _LOOP_BREAK_THRESHOLD = 2
    _LOOP_BREAK_MAX = 2

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        settings: AgentLoopSettings | None = None,
        *,
        session_manager: SessionManager | None = None,
        backend: "MemoryBackend | None" = None,
        plugin_tools: "list[Tool] | None" = None,
        interactive: bool = True,
        now_fn: Callable | None = None,
        hooks: "CompositeHook | None" = None,
        strategies: "StrategyRegistry | None" = None,
    ):
        """``settings`` carries everything read from config (see
        :class:`AgentLoopSettings`); the keyword arguments are the runtime
        collaborators a call site builds itself.

        ``backend`` is the plugin-provided MemoryBackend: with one wired the
        after-turn pipeline gains ``backend.store`` and ``backend.feedback``
        and the context engine gains the long-term memory recall lane. ``plugin_tools``
        are registered alongside the built-in tools. ``interactive`` gates the
        ``runtime.checkpoint`` policy (a one-shot ``-m`` turn has no next turn
        to recover into). ``now_fn`` is the fake-clock injection point for
        benchmark harnesses; it stamps persisted messages and the prompt's
        "Current Time" alike so the two stay in sync.
        """
        from opendde_harness.agent.hook import CompositeHook
        from opendde_harness.token_wise.registry import StrategyRegistry

        if settings is None:
            settings = AgentLoopSettings()
        self.settings = settings
        self.provider = provider
        self.workspace = workspace
        self.model = settings.model or provider.get_default_model()
        # Resolved lazily on the first tool result that carries an image. Keyed
        # by model, not a single flag: the loop is a long-lived singleton and
        # takes a per-call model (strategies rewrite it, and the model chain
        # falls back), so one model's verdict must not answer for another's.
        self._image_tool_result_ok: dict[str, bool] = {}
        self._vision_ok: dict[str, bool] = {}
        self.max_iterations = settings.max_iterations
        self._recovery_limits = settings.empty_recovery
        # allow_import=False: construction must not pay LiteLLM's import for a
        # window; the lazy provider's prewarm thread imports it and
        # `refresh_context_window` re-walks the ladder once it has, so only a
        # provider with no such callback has its answer settled here. None
        # means unknown -- no table lists the model and the user declared
        # nothing -- and every consumer treats that as "do not trim" rather
        # than as a number (see TokenBudget).
        self.context_window_tokens: int | None = None
        self.context_window_source: str = ""
        self._budget_warned_for: int | None = None
        self._apply_context_window(self.resolve_window(allow_import=False), settled=not hasattr(provider, "on_built"))
        self.brave_api_key = settings.brave_api_key
        self.jina_api_key = settings.jina_api_key
        self.web_proxy = settings.web_proxy
        self.exec_config = settings.exec_config
        self.restrict_to_workspace = settings.restrict_to_workspace
        # TokenWise strategies — empty registry acts as pure pass-through.
        self.strategies = strategies if strategies is not None else StrategyRegistry([])
        self._now_fn = now_fn or datetime.now

        self.backend: "MemoryBackend | None" = backend
        # Writes that outran their turn budget and are still running. Held so
        # teardown can drain them instead of dropping whatever was slowest.
        self._store_inflight: set[asyncio.Task] = set()
        # Writes refused because indexing never caught up. Reported at teardown:
        # a dropped write is a turn the user will not be able to recall.
        self._store_dropped = 0

        self.plugin_tools: "list[Tool]" = list(plugin_tools or [])

        # Per-turn stash for the ``injected_skill_ids`` the context engine
        # surfaces in ``AssembledContext.metadata``, populated inside
        # ``_assemble_context_messages`` so the after-turn feedback dispatcher
        # can read it without re-running selection. ``None`` means "use the
        # ``_collect_injected_skill_ids`` fallback path".
        self._last_injected_skill_ids: list[str] | None = None

        self.context = ContextBuilder(
            workspace,
            skill_forge_config=settings.skill_forge,
            llm_provider=provider,
            now_fn=now_fn,
        )
        self.sessions = session_manager or SessionManager(workspace)
        # Tool names to omit from the registry — applied after default-tool
        # registration and after MCP connect so it can blacklist either group.
        # Used by eval harnesses (e.g. BCP) that need a strict tool subset.
        self._disabled_tools = set(settings.disabled_tools)
        self._tool_search_config = settings.tool_search
        self.tools = ToolRegistry()

        # Context engine — the single ContextAssembler. Constructed here
        # (after self.tools) so the factory can capture
        # ``self.tools.get_definitions`` as a deferred callable; the registry
        # is filled by ``_register_default_tools`` later in this constructor.
        # Deferred import: see the module-level note about the import cycle.
        from opendde_harness.context_engine import build_context_engine

        self.context_config = settings.context
        self._skill_blocklist = list(getattr(settings.skill_forge, "blocklist", None) or [])
        self.context_engine: "ContextEngine" = build_context_engine(
            workspace=workspace,
            config=settings.context,
            builder=self.context,
            provider=provider,
            model=self.model,
            context_window_tokens=self.context_window_tokens,
            get_tool_definitions=self.tools.get_definitions,
            now_fn=now_fn,
            backend=backend,
            memory_config=settings.memory,
            skill_forge_router_config=settings.skill_forge.router if settings.skill_forge is not None else None,
            skill_forge_config=settings.skill_forge,
        )

        # Runtime discipline: the per-turn shadow-git checkpoint, gated by
        # (policy, interactive) — see ``_checkpoint_active``. When the gate is
        # closed the loop is byte-identical to baseline.
        self.runtime_config = settings.runtime
        self.interactive = interactive
        self._checkpoint = None
        if self._checkpoint_active(settings.runtime.checkpoint.policy, interactive):
            from opendde_harness.agent.loop.checkpoint import CheckpointService

            try:
                self._checkpoint = CheckpointService(
                    workspace,
                    shadow_dir=settings.runtime.checkpoint.shadow_dir,
                )
            except ValueError as exc:
                # Bad shadow_dir (e.g. ``../escape`` or absolute path) →
                # CheckpointService refuses to construct. Don't crash the
                # whole agent over a config typo; log and disable the
                # safety net so the turn still runs.
                logger.warning("runtime.checkpoint disabled — {}", exc)
        # session_key -> {"checkpoint_id", "files"} stashed when a turn is
        # interrupted (max-iter); consumed by the next turn's recovery prompt.
        self._pending_recovery: dict[str, dict] = {}

        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            model=self.model,
            brave_api_key=settings.brave_api_key,
            jina_api_key=settings.jina_api_key,
            web_proxy=settings.web_proxy,
            exec_config=settings.exec_config,
            restrict_to_workspace=settings.restrict_to_workspace,
            max_concurrent=settings.max_concurrent_subagents,
            max_spawns_per_hour=settings.max_subagent_spawns_per_hour,
        )

        self._executor: SandboxExecutor = DirectExecutor()
        self._executor_stack: AsyncExitStack | None = None
        self._executor_started: bool = False
        self._executor_start_lock = asyncio.Lock()

        self._running = False
        self._mcp_servers = dict(settings.mcp_servers)
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._processing_lock = asyncio.Lock()
        # Fired after every dispatched turn (success, error, or cancel).
        # Callbacks must be cheap and must not raise.
        self.on_turn_complete: list[Callable[[], None]] = []
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens or 0,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            now_fn=now_fn,
        )

        self._consolidation_tasks: set[asyncio.Task] = set()

        # A switch that arrives mid-turn is parked here until no turn is
        # running, so a turn in flight finishes on the provider it started
        # with. A depth counter, not a flag: OriginPools gates USER and
        # system origins on independent semaphores with no global cap
        # (spine/scheduler.py), so a user turn and a cron turn overlap on
        # this loop under the TUI defaults.
        self._pending_provider: tuple[LLMProvider, str] | None = None
        self._turns_in_flight = 0

        # ``self.subagents``, ``self.context_engine`` and
        # ``self.memory_consolidator`` were each handed ``provider`` earlier in
        # this constructor and hold their own reference; ``_adopt_provider`` is
        # what keeps them from outliving a live model switch. Add the call
        # there when adding another holder.

        # ``self.context.skills`` is the :class:`LocalSkillCatalog` for the
        # always-skills + ``# Skills`` render path; the SkillForgeRouter stack
        # (assembled in ``context_engine.factory``) owns retrieval.

        self.hooks: "CompositeHook" = CompositeHook()
        if hooks is not None:
            self.hooks.extend(hooks)

        self._register_default_tools()
        self._apply_disabled_tools()

        # LazyProvider defers the litellm import behind a background prewarm
        # thread (see providers.lazy); the window this constructor just
        # resolved above was answered with allow_import=False, so it can be
        # wrong until that import lands. Wiring the callback fixes it up in
        # place once the real provider is built -- a no-op for any other
        # provider, which has no ``on_built`` to set.
        if hasattr(provider, "on_built"):
            provider.on_built = self.refresh_context_window

    def _apply_disabled_tools(self) -> None:
        """Unregister tools whose names appear in ``tools.disabled_tools``.

        Run after :meth:`_register_default_tools` (here) and after MCP connect
        (see :meth:`_connect_mcp`) so the blacklist can cover either group.
        Silent on misses — eval configs commonly carry an over-broad list
        that's a no-op for tools that weren't registered in this build.
        """
        if not self._disabled_tools:
            return
        for name in list(self._disabled_tools):
            if self.tools.has(name):
                self.tools.unregister(name)

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        """Point the loop and everything it built at a new provider/model.

        ``config.set model`` builds a provider from the prospective config
        and hands it here. Assigning ``self.provider`` alone is not enough:
        the subagent manager, the context engine's LLM-backed segments and
        the consolidator each captured the provider handed to them in
        ``__init__``. Left behind, they keep calling the old endpoint for
        the rest of the process -- which is how switching away from a dead
        credential fixed the main loop while subagents and the skill
        rewriter/gate went on failing to authenticate.

        A switch that lands while any turn is running is parked rather than
        applied: the loop reads ``self.provider`` at call time (eight sites
        in this module, plus the context engine and consolidator
        underneath), so adopting mid-turn would relay one conversation
        across two vendors, and both call paths would turn the rejected
        request into ``finish_reason="error"`` content -- the turn reports a
        failure with no sign that its endpoint moved, which is not a
        diagnosis the user can act on.

        The park is the second line of defence, not the first: the RPC
        rejects a switch outright when the caller's own session has a turn
        in flight (``is_turn_active`` in ``tui_rpc.methods.config``). This
        covers what that guard cannot see -- a caller that passes no
        ``session_id``, and the proactive turns that run in their own lanes.
        Note the RPC still answers ``applied: True`` and the config file is
        already written, so a parked switch is applied on disk while the
        loop reports the old model until the last turn drains.

        Detached subagents are not covered by that park -- they outlive the
        turn that spawned them -- so ``SubagentManager`` snapshots instead.
        """
        if self._turns_in_flight:
            # The only trace of the window the docstring describes.
            logger.info("model switch to {} parked until {} running turn(s) drain", model, self._turns_in_flight)
            self._pending_provider = (provider, model)
            return
        self._adopt_provider(provider, model)

    def _adopt_provider(self, provider: LLMProvider, model: str) -> None:
        """Hand a provider to the loop and every subsystem holding the old one."""
        logger.info("adopting provider switch: model={}", model)
        self.provider = provider
        self.model = model
        # Cached per model id but computed from the provider, so a swap that
        # keeps the model id would keep serving the old transport's verdict.
        self._image_tool_result_ok.clear()
        self.subagents.set_provider(provider, model)
        self.context_engine.set_provider(provider, model)
        self.memory_consolidator.set_provider(provider, model)
        # Here rather than at the RPC call site: a parked switch adopts long
        # after that call returns, and the window must follow the pair that
        # was actually adopted, not the model the RPC saw.
        self.refresh_context_window()

    def _adopt_pending_provider(self) -> None:
        """Apply a parked switch. Callers must check that no turn is running."""
        pending = self._pending_provider
        if pending is None:
            return
        self._pending_provider = None
        self._adopt_provider(*pending)

    def resolve_window(self, model: str | None = None, *, allow_import: bool = True) -> Resolved:
        """Walk the window ladder for ``model`` (default: the loop's own).

        The model's own overlay is supplied from here so every caller -- construction, a ``/model`` switch, the per-turn
        usage report, the session banner -- answers the same question the same
        way. No tier reaches the network; ``allow_import=False`` additionally
        skips LiteLLM's import (see ``rates.resolve_context_window``).
        """
        model = model or self.model
        return resolve_context_window(
            model,
            overlay=overlay_for(self.settings.model_overlays, model),
            allow_import=allow_import,
        )

    def _apply_context_window(self, resolved: Resolved, *, settled: bool = True) -> None:
        """Adopt a resolution; say so when it is the final answer for this model.

        ``settled=False`` is construction ahead of a lazy provider's import,
        where an unknown may still become known and a warning would be noise.
        """
        self.context_window_tokens = resolved.tokens
        self.context_window_source = resolved.source
        if not settled:
            return
        if resolved.known:
            logger.info("context window for {}: {} tokens ({})", self.model, resolved.tokens, resolved.source)
        else:
            logger.warning(
                "context window for {} is unknown: no table lists it, so history is not trimmed; "
                "declare providers.<name>.modelOverlay.<id>.contextWindowTokens to size it",
                self.model,
            )

    def refresh_context_window(self) -> None:
        """Re-resolve ``context_window_tokens`` against the current ``self.model``.

        The ladder is re-walked so a ``/model`` switch picks up the new model's
        own window -- its overlay, then the tables -- instead of keeping the
        old one's.

        Also the callback ``LazyProvider.on_built`` fires from its prewarm
        thread, i.e. off the event loop -- safe because every write this
        method triggers, transitively through the consolidator and the
        context engine's builders, is a plain attribute assignment, and the
        GIL makes each one atomic.
        """
        # allow_import=False: a /model switch runs inside the running event
        # loop, so this must not block it on LiteLLM's import. Once the lazy
        # provider has imported it (the on_built callback lands here too) the
        # check is free and the LiteLLM tier answers.
        self._apply_context_window(self.resolve_window(allow_import=False))
        # Cascade into the builders that sized themselves against the window
        # at construction (the Curator's trimmer) and the consolidator --
        # both would otherwise keep budgeting against the pre-switch model's
        # window for the rest of the session. The consolidator's window is a
        # plain attribute (no setter of its own), set directly here; it takes
        # 0 for unknown and treats it as "off".
        self.context_engine.set_context_window(self.context_window_tokens)
        self.memory_consolidator.context_window_tokens = self.context_window_tokens or 0

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, GrepTool, FindTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
                executor=self._executor,
                extra_deny_patterns=self.exec_config.extra_deny_patterns,
            )
        )
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(api_key=self.jina_api_key, proxy=self.web_proxy))
        self.tools.register(SpawnTool(manager=self.subagents))
        # The QuestionBroker is a per-transport singleton, late-bound via
        # set_broker once the transport (TUI RPC server) exists.
        self.tools.register(AskUserTool())

        # Plugin-contributed tools (e.g. the memory plugin's ``understand_media``).
        # Registered last so a plugin can override a built-in by name if
        # it deliberately contributes the same name; ``_apply_disabled_tools``
        # still runs afterward and can strip any of them.
        for tool in self.plugin_tools:
            self.tools.register(tool)

        # ``use_skill`` is the single skill-body loading path: catalog
        # assembly advertises metadata only, the body is loaded on demand.
        skill_registry = getattr(
            getattr(self.context, "skills", None),
            "registry",
            None,
        )
        if skill_registry is not None:
            from opendde_harness.agent.tools.use_skill import UseSkillTool

            self.tools.register(UseSkillTool(registry=skill_registry, blocklist=self._skill_blocklist))

        # Progressive tool disclosure. Registered last so the catalog it
        # searches covers every built-in/plugin tool above; MCP tools join
        # later (registered in ``_connect_mcp``) and the strategy picks them up
        # since it re-reads the registry each turn.
        cfg = self._tool_search_config
        if cfg is not None and cfg.enabled:
            from opendde_harness.agent.tools.tool_search import (
                DEFAULT_ALWAYS_VISIBLE,
                ToolCallTool,
                ToolSearchController,
                ToolSearchStrategy,
                ToolSearchTool,
            )

            always = set(DEFAULT_ALWAYS_VISIBLE) | set(cfg.always_visible)
            self.tool_search_controller = ToolSearchController(
                self.tools,
                always_visible=always,
                search_result_limit=cfg.search_result_limit,
            )
            self.tools.register(ToolSearchTool(self.tool_search_controller))
            self.tools.register(ToolCallTool(self.tool_search_controller))
            # ``first=True``: filter the tool list before any strategy that
            # marks the final tool with ``cache_control`` (else the marked tool
            # may be filtered out and the breakpoint lost).
            self.strategies.register(
                ToolSearchStrategy(
                    self.tool_search_controller,
                    compaction_threshold=cfg.compaction_threshold,
                ),
                first=True,
            )

    def _supports_image_tool_result(self, model: str | None = None) -> bool:
        """Cached per model: resolving the LiteLLM target parses the model string,
        and this is asked once per tool call that returns an image.

        A ``False`` learned from a refused request (see ``should_drop_tool_images``)
        is written into the same cache, so a static table that guessed wrong stops
        costing a wasted call after the first one.
        """
        key = model or self.model
        if key not in self._image_tool_result_ok:
            self._image_tool_result_ok[key] = supports_image_tool_result(self.provider, key, self._spec_for(key))
            logger.debug(
                "image-in-tool-result support for {}: {}",
                key,
                self._image_tool_result_ok[key],
            )
        return self._image_tool_result_ok[key]

    @staticmethod
    def _spec_for(model: str) -> "ProviderSpec | None":
        from opendde_harness.providers.registry import find_by_model

        return find_by_model(model)

    # The tool that can read an attachment for a model that cannot see it.
    # Contributed by the long-term memory plugin, so absent on a default install.
    _DESCRIBE_TOOL = "understand_media"

    def _describe_tool_name(self) -> str | None:
        """The description tool's name if it is registered, else ``None``.

        Checked rather than assumed: the note that replaces a picture points at
        this tool, and pointing at one the model was never given is an
        instruction it cannot follow.
        """
        return self._DESCRIBE_TOOL if self.tools.get(self._DESCRIBE_TOOL) else None

    def _route_result_images(
        self,
        model_text: str,
        blocks: list[dict[str, Any]] | None,
        model: str,
    ) -> tuple[str, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """Decide how a tool result's pictures reach ``model``.

        Returns the text the tool result carries, the blocks to put *in* it, and
        the blocks to attach to a following user message. Exactly one of the last
        two is ever populated.

        Three outcomes, and the wording differs because the model's next move
        differs. No vision at all: nothing follows, so the note must not promise
        an attachment, and it names the description tool when one is registered.
        Vision and a transport that carries images in a ``role="tool"`` message:
        the blocks ride along untouched. Vision but a transport that cannot (every
        OpenAI-style Chat Completions endpoint -- image is excluded from the tool
        role at the schema level): the result carries text and the picture follows
        in a user message, the shape OpenClaw uses.

        A method rather than a branch inside the loop so it can be tested at all:
        the loop reaches this point only through a live provider and a real tool
        call, and the wrong choice here is silent -- the model answers about a
        picture it never received.
        """
        if not blocks:
            return model_text, blocks, None
        if not self._supports_vision(model):
            return image_placeholder_text(blocks, blind=True, describe_tool=self._describe_tool_name()), None, None
        if self._supports_image_tool_result(model):
            return model_text, blocks, None
        attach = [b for b in blocks if b.get("type") == "image_url"]
        return image_placeholder_text(blocks), None, attach

    def _supports_vision(self, model: str | None = None) -> bool:
        """Cached per model: whether this model can see a picture at all.

        Asked once per turn and once per tool result that returns an image, and
        the lookup joins the model string against the gateway catalogue -- same
        reason the sibling probe above is cached.

        Only a real verdict is cached. ``vision_verdict`` returns ``None`` while
        the catalog has no answer -- a cold install, before the background warm
        lands -- and that is optimism rather than knowledge: caching it would
        freeze the guess for the life of this loop, which is the life of the
        process, and the warm would then fill a table nothing re-reads. An
        unknown model is re-asked each turn, which costs a dict lookup.

        The verdict is the routed primary's. A fallback further down the chain is
        sent the same message list, so a vision-capable primary with a blind
        fallback hands the blind endpoint an image block it will refuse; that
        refusal classifies as fatal and stops the chain rather than answering
        blind. Pre-existing in the sibling probe too, and it needs the fallback
        chain to be assembled per candidate to fix properly.
        """
        key = model or self.model
        cached = self._vision_ok.get(key)
        if cached is not None:
            return cached

        verdict = vision_verdict(key, self._spec_for(key), self.provider)
        logger.debug("vision support for {}: {}", key, verdict)
        if verdict is None:
            return True
        self._vision_ok[key] = verdict
        return verdict

    # ── Context engine helpers ──────────────────────────────────────────

    def _context_messages_for_session(self, session: Session) -> list[dict[str, Any]]:
        """Return the candidate message view owned by the active context engine.

        Curator (``owns_compaction=True``) wants the full append-only log so
        it can decide what to archive itself; Legacy wants the post-consolidation
        slice to match the pre-Curator behavior exactly.
        """
        if self.context_engine.owns_compaction:
            return list(session.messages)
        return session.get_history(max_messages=0)

    def _make_token_budget(self, selected_skills: list[Any] | None = None) -> TokenBudget:
        """Compute a conservative per-turn prompt budget for the active engine."""
        # allow_import=False for the same reason construction passes it (see
        # __init__): this runs per turn on the loop's own thread, and it only
        # needs a number to reserve -- not the one a request will carry. The
        # fallback under-reserves at worst; the importing tier costs seconds.
        ceiling = send_max_tokens(
            getattr(self.provider, "generation", None),
            # The stored id, the same one the request's own bound resolves
            # under: the ladder knows every spelling a table files it as, and
            # the wire id names the driver -- "openai/" for a custom endpoint,
            # a bare slug for the Codex login -- which answers for the wrong
            # model or for none.
            self.model,
            overlay=overlay_for(self.settings.model_overlays, self.model),
            allow_import=False,
        )
        # The whole ceiling, not a share of it. Requests no longer name a
        # ceiling, so the one that applies is the model's own -- whatever the
        # vendor or LiteLLM's transformation fills in. Reserving less than that
        # hands out a prompt the reply cannot coexist with: measured on this
        # repo's default model, a share leaves the prompt 150000 of a 200000
        # window against a reply allowed 64000, and the sum is refused at
        # request time. `_emergency_shrink` only elides tool bodies, so a
        # history grown on conversation gets no retry from that refusal.
        #
        # A share would be right again only if the request carried one, which
        # is the trade the previous shape made and this one does not.
        window = self.context_window_tokens
        reserved_output = ceiling if window is None else min(ceiling, window)
        tool_tokens = estimate_prompt_tokens([], self.tools.get_definitions())
        system_prompt = self.context.build_system_prompt(selected_skills)
        system_tokens = estimate_prompt_tokens([{"role": "system", "content": system_prompt}])
        # Unknown stays unknown. Subtracting from a stand-in produced a budget
        # that nothing had measured, and the curator and trimmer acted on it.
        # A window the reservation alone fills -- a model whose window is no
        # larger than its output ceiling, which a declared overlay can now
        # describe -- leaves nothing for history; that is reported as unknown
        # too, not as a budget of zero, which the curator's gate read as "every
        # history overflows" and ran its LLM loop on an empty one.
        #
        # Two budgets, two questions: this is history-only headroom for the
        # curator's gate, while the trimmer bounds the whole prompt (its
        # estimate includes the system prompt and tool schemas) against
        # ``window - reserved_output``. Neither is the other's subtraction.
        available_history = None if window is None else window - reserved_output - tool_tokens - system_tokens
        if available_history is not None and available_history <= 0:
            # Once per resolved window, not per turn: this runs every turn.
            if self._budget_warned_for != window:
                self._budget_warned_for = window
                logger.warning(
                    "context window for {} ({} tokens) holds no history beside a {}-token reply, "
                    "{} tokens of tools and {} of system prompt; history is not trimmed",
                    self.model,
                    window,
                    reserved_output,
                    tool_tokens,
                    system_tokens,
                )
            available_history = None
        return TokenBudget(
            context_length=self.context_window_tokens,
            reserved_output=reserved_output,
            reserved_tools=tool_tokens,
            reserved_system=system_tokens,
            available_history=available_history,
        )

    async def _assemble_context_messages(
        self,
        *,
        session: Session,
        session_key: str,
        current_message: str,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        selected_skills: list[Any] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the active context engine for the main-agent message window.

        ``model`` is the id the request will actually reach (the router's pick,
        when there is one). It decides whether an attachment is inlined as a
        picture, so defaulting it to ``self.model`` would let the configured
        model answer for a routed one.
        """
        from opendde_harness.context_engine import TurnContext  # deferred — see module note

        # Phase A / Phase C tidy: reset the metadata stash BEFORE calling
        # the engine. If ``engine.assemble`` raises partway, the next
        # caller falls back to the legacy ``_collect_injected_skill_ids``
        # path rather than accidentally consuming a previous turn's
        # injected ids. Only successful assemble repopulates the stash.
        self._last_injected_skill_ids = None
        session_messages = self._context_messages_for_session(session)
        assembled = await self.context_engine.assemble(
            session_key,
            session_messages,
            self._make_token_budget(selected_skills),
            turn=TurnContext(
                current_message=current_message,
                media=media,
                can_see_images=self._supports_vision(model),
                describe_tool=self._describe_tool_name(),
                channel=channel,
                chat_id=chat_id,
                selected_skills=selected_skills,
            ),
        )
        # Stash the engine's injected_skill_ids so the after-turn
        # feedback dispatcher can read the source-qualified ids the
        # unified engine populates via SkillForgeRouter. If the key is absent
        # the stash stays None and _collect_injected_skill_ids falls back
        # to the SkillMeta-based path.
        meta_ids = assembled.metadata.get("injected_skill_ids") if assembled.metadata else None
        # An explicit empty list means the context engine advertised a catalog
        # but injected no bodies. Preserve that distinction; falling back to
        # the legacy selector here would falsely credit available skills as
        # injected ones.
        self._last_injected_skill_ids = list(meta_ids) if meta_ids is not None else None
        messages = assembled.messages
        self._inject_recovery_block(session_key, messages)
        return messages

    @staticmethod
    def _checkpoint_active(policy: str, interactive: bool) -> bool:
        """Resolve ``runtime.checkpoint.policy`` against the call-site's
        ``interactive`` signal. ``"interactive"`` (the default) skips the
        snapshot for one-shot ``-m`` invocations — those have no "next turn"
        to inject recovery into, so paying the snapshot cost there is just
        deadweight. ``"always"`` opts in regardless; ``"never"`` opts out
        regardless."""
        if policy == "never":
            return False
        if policy == "always":
            return True
        return interactive  # policy == "interactive"

    def _stash_recovery(self, session_key: str, outcome: "LoopOutcome") -> None:
        """Remember an interrupted turn's snapshot so the next turn in this
        session gets a recovery prompt. No-op unless checkpoint is enabled
        and the turn was actually interrupted with something to recover.

        Status filter is intentional: only ``"interrupted"`` triggers a
        recovery prompt. ``"error"`` turns still get a per-turn shadow
        commit (useful for audit), but they don't usually have a partial-
        edits trajectory to resume (provider 400 etc.) and surfacing
        "Files modified last turn" for them would be misleading.
        """
        if self._checkpoint is None or outcome.status != "interrupted":
            return
        if outcome.edited_files or outcome.checkpoint_id:
            self._pending_recovery[session_key] = {
                "checkpoint_id": outcome.checkpoint_id,
                "files": outcome.edited_files,
            }

    def _inject_recovery_block(self, session_key: str, messages: list[dict]) -> None:
        """Prepend a recovery notice to the current user message when the
        previous turn for this session was interrupted. Consumed once on
        successful injection; if the current message's content has an
        unexpected shape (None / dict / etc.) the pending entry is kept so
        a later assembly with a normal content can still inject it."""
        recovery = self._pending_recovery.get(session_key)
        if not recovery or not messages:
            return
        last = messages[-1]
        if last.get("role") != "user":
            # Last message isn't the user turn — keep the recovery pending so
            # the next assembly (which does end with the user message) injects it.
            return
        content = last.get("content")
        files = recovery.get("files") or []
        cid = recovery.get("checkpoint_id")
        lines = ["[Recovery — the previous turn was interrupted before finishing]"]
        if files:
            lines.append("Files modified last turn: " + ", ".join(files))
        if cid:
            lines.append(f"Checkpoint: {cid}")
        lines.append("Verify the current state of these files before continuing.")
        block = "\n".join(lines)
        # Mutate first, pop second — atomic from the caller's perspective. If
        # we can't safely write to ``content`` (unknown shape) the recovery
        # stays pending instead of being silently dropped on the floor.
        if isinstance(content, str):
            last["content"] = f"{block}\n\n{content}"
        elif isinstance(content, list):
            last["content"] = [{"type": "text", "text": block}] + content
        else:
            return  # unexpected content shape → keep pending
        self._pending_recovery.pop(session_key, None)

    @trace.instrument("memory.feedback", extract=semconv.memory_feedback)
    async def _dispatch_backend_feedback(
        self,
        session_key: str,
        injected_skill_ids: list[str] | None,
        used_skill_ids: list[str] | None = None,
    ) -> None:
        """FB-1: forward source-qualified skill-usage signals to
        :meth:`MemoryBackend.feedback`.

        Skill IDs surface with a ``<source>/<native_id>`` prefix
        (``local/git-resolver`` / ``mass/abc`` / ``memory/xyz``).
        Only the ``memory/`` prefix is forwarded — static libraries
        (``local`` / ``mass``) have no feedback channel; the dispatcher
        is silent for them (no warning, just skipped). Unprefixed legacy
        ids (e.g. raw skill names emitted by the pre-SkillForgeRouter
        ``SkillService.select`` path) are also skipped — they predate
        the qualified-id convention and there's no safe routing target.

        No-ops when:
        - ``self.backend is None`` (no plugin wired)
        - No qualified-id matches the ``memory/`` prefix
        - The injected + used lists are both empty / None

        Exceptions from :meth:`backend.feedback` are caught + logged.
        The host MUST NOT abort the after-turn pipeline because a
        plugin's feedback handler raised — feedback is best-effort
        telemetry, not load-bearing state.
        """
        if self.backend is None:
            return
        injected_native = _filter_qualified_ids(injected_skill_ids, "memory")
        used_native = _filter_qualified_ids(used_skill_ids, "memory")
        if not injected_native and not used_native:
            return
        signals = {
            "kind": "skill_usage",
            "session_id": session_key,
            "injected": injected_native,
            "used": used_native,
        }
        try:
            await self.backend.feedback(signals)
        except Exception:
            logger.exception(
                "backend.feedback failed for session {}; signals dropped",
                session_key,
            )

    @trace.instrument("memory.store", extract=semconv.memory_store)
    async def _dispatch_backend_store(
        self,
        session_key: str,
        messages_slice: list[dict],
    ) -> None:
        """AG-1: forward a turn's messages to the plugin :class:`MemoryBackend`.

        Third peer step in the after-turn pipeline alongside
        ``context_engine.after_turn`` (engine-side bookkeeping) and
        ``memory.maybe_consolidate`` (opendde-core compaction). When no
        backend was wired (``self.backend is None``), this is a no-op so
        legacy callsites that never registered a plugin behave
        identically to pre-AG-1.

        Exceptions raised by the backend are logged and swallowed —
        the AgentLoop's main pipeline must never abort because the
        plugin-side index failed; the turn is already saved to the
        session log and the host's MEMORY.md compaction will still run.
        """
        if self.backend is None:
            return
        if not messages_slice:
            return

        async def _store() -> None:
            try:
                await self.backend.store(session_key, messages_slice)  # type: ignore[union-attr]
            except Exception:
                logger.exception(
                    "backend.store failed for session {}; turn data preserved in session log, "
                    "plugin-side indexing skipped",
                    session_key,
                )

        task = asyncio.create_task(_store())
        self._store_inflight.add(task)
        task.add_done_callback(self._store_inflight.discard)
        # Deliberately not cancelled on timeout: the point is to stop *waiting*,
        # not to abandon the write. A turn that indexes quickly still does so
        # inline, which keeps ordering intact in the common case.
        await asyncio.wait({task}, timeout=_STORE_TURN_BUDGET_S)

        if len(self._store_inflight) > _STORE_MAX_INFLIGHT:
            # Backpressure rather than unbounded growth: a service slow enough
            # to accumulate this many outstanding writes is one whose queue
            # should stop growing, not one to keep feeding.
            #
            # Bounded by the turn's own budget, and that bound is the whole
            # point. Unbounded, this waited on the slowest outstanding write
            # instead -- and since flush_every_turns defaults to 1, every
            # interactive turn is a final flush carrying the six-minute
            # extraction budget, so reaching the cap stalled a turn for
            # minutes. That is the stall this method exists to remove.
            await asyncio.wait(set(self._store_inflight), timeout=_STORE_TURN_BUDGET_S)
            if len(self._store_inflight) > _STORE_MAX_INFLIGHT:
                # Still saturated. The queue is what gives, not the turn: this
                # write is dropped and said out loud at teardown, rather than
                # held open behind writes that are already over their time.
                task.cancel()
                self._store_inflight.discard(task)
                self._store_dropped += 1
                logger.warning(
                    "backend.store dropped for session {}: {} writes still in flight after {}s",
                    session_key,
                    len(self._store_inflight),
                    _STORE_TURN_BUDGET_S,
                )

    async def drain_backend_stores(self, timeout: float = _STORE_DRAIN_BUDGET_S) -> None:
        """Let detached writes finish before the process goes away.

        Writes that outran their turn budget are still in flight. Exiting on top
        of them loses exactly the turns that were slowest to index, which is a
        silent and biased kind of data loss.
        """
        pending = {t for t in self._store_inflight if not t.done()}
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        if self._store_dropped:
            logger.warning(
                "{} turn(s) were not indexed: the memory service never caught up",
                self._store_dropped,
            )

    def _collect_injected_skill_ids(
        self,
        selected: list[Any] | None,
    ) -> list[str]:
        """Combine selector top-K + always-skills into a deduplicated id list.

        ``selected`` is the :class:`SkillMeta` list returned by the
        retrieval selector for this turn (or ``None`` when the selector
        is disabled / returned empty). always-skills are pulled from
        :class:`LocalSkillCatalog` since they are unconditionally rendered
        regardless of the selector's output.

        Returns ids canonicalized to ``{source}/{stable_key}`` form.
        ``stable_key`` is whatever the source uses for unambiguous
        addressing — the sqlite ``skills.id`` for ``memory`` (which
        allows duplicate names) and the directory / display name
        elsewhere. Different SkillMeta producers populate ``meta.id``
        inconsistently (file registry: bare key or ``{source}/{key}``;
        sqlite store: ``{source}/{key}``); this function normalizes them
        to a single shape so the after-turn ``backend.feedback`` signal
        can route them uniformly.
        """
        skills_svc = getattr(self.context, "skills", None)
        if skills_svc is None:
            return []

        seen: set[str] = set()
        ids: list[str] = []

        def _add(meta: Any) -> None:
            src = getattr(meta, "source", None)
            mid = getattr(meta, "id", None)
            if not src or not mid:
                return
            canonical = mid if "/" in mid else f"{src}/{mid}"
            if canonical not in seen:
                seen.add(canonical)
                ids.append(canonical)

        def _add_raw_id(qid: str) -> None:
            if qid and qid not in seen:
                seen.add(qid)
                ids.append(qid)

        # Prefer the AssembledContext metadata the unified engine
        # populated from its SkillForgeRouter. Those ids are already
        # source-qualified (``local/x`` / ``mass/y`` / ``memory/z``) so
        # they bypass the SkillMeta-canonicalization path. Always-skills
        # get folded in afterwards because they live outside
        # SkillForgeRouter's selection.
        if self._last_injected_skill_ids is not None:
            for qid in self._last_injected_skill_ids:
                _add_raw_id(qid)
        else:
            for meta in selected or []:
                _add(meta)
        try:
            always = skills_svc.get_always_skills()
        except Exception:
            always = []
        for meta in always:
            _add(meta)
        return ids

    async def _start_executor(self) -> None:
        """Idempotent: start the sandbox executor once before first use."""
        async with self._executor_start_lock:
            if self._executor_started:
                return
            stack = AsyncExitStack()
            try:
                await stack.__aenter__()
                await stack.enter_async_context(self._executor)
            except Exception:
                await stack.aclose()
                raise
            self._executor_stack = stack
            self._executor_started = True

    async def close_executor(self) -> None:
        """Tear down the executor."""
        if self._executor_stack:
            try:
                await self._executor_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._executor_stack = None
        self._executor_started = False

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        # Set flag synchronously before the first await — asyncio is single-threaded so no
        # context switch occurs here; a lock is not needed for this mutual-exclusion pattern.
        self._mcp_connecting = True
        try:
            await self._start_executor()  # ensure executor is live before MCP servers connect
            from opendde_harness.agent.tools.mcp import connect_mcp_servers

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(
                self._mcp_servers,
                self.tools,
                self._mcp_stack,
                executor=self._executor,
            )
            # Re-apply blacklist: MCP servers may register tool names that
            # also appear in ``disabled_tools`` (e.g. ``mcp_<server>_search``).
            self._apply_disabled_tools()
            self._mcp_connected = True
            self._mcp_connecting = False
        except Exception:
            # Reset in-progress flag so a subsequent call can retry.
            self._mcp_connecting = False
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
            raise

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None, session_key: str | None = None
    ) -> None:
        """Update context for all tools that need routing info."""
        if (tool := self.tools.get("spawn")) and hasattr(tool, "set_context"):
            tool.set_context(channel, chat_id, session_key or f"{channel}:{chat_id}")

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content.

        Paired blocks are removed. What is left is then checked for being
        nothing but tag debris: when a backend inlines its reasoning and the
        turn is cut off inside it, content arrives as a lone closing tag with
        no opener to pair against, so the substitution above finds nothing and
        an eleven-character string reads as a real answer. Recovery is skipped
        and the tag is what the user sees.

        The check is on residue, not on vendor spellings -- it does not matter
        which prefix a backend picked. Text that merely mentions a tag keeps
        its other words and is returned untouched.
        """
        if not text:
            return None
        cleaned = strip_think_blocks(text)
        if is_only_think_debris(cleaned):
            return None
        return cleaned or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _build_usage_snapshot(response, model: str, session_key: str) -> "UsageSnapshot":
        """Build a UsageSnapshot from an LLMResponse for TokenWise after-hooks.

        Normalizes input_tokens to *fresh* (non-cached) prompt tokens. The
        ``prompt_tokens`` field has two conventions in the wild:
          - Anthropic native: fresh-only (cache_read/write are separate counts)
          - OpenRouter/LiteLLM: total (already includes cache_read + cache_write)
        We detect by inequality and subtract when needed so downstream code
        (pricing, telemetry) sees a single consistent semantics.
        """
        from opendde_harness.token_wise.base import UsageSnapshot
        from opendde_harness.token_wise.pricing import estimate_cost_usd

        usage = response.usage or {}
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        out_toks = int(usage.get("completion_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)

        # Normalize to fresh-only.
        if prompt_t >= cache_read + cache_write and (cache_read + cache_write) > 0:
            fresh = prompt_t - cache_read - cache_write
        else:
            fresh = prompt_t

        # Left as None for a plan-billed provider: the field is optional all the
        # way to the status bar, which renders it only when it is a number.
        cost = estimate_cost_usd(model, fresh, out_toks, cache_read, cache_write)
        return UsageSnapshot(
            model=model,
            input_tokens=fresh,
            output_tokens=out_toks,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            estimated_cost_usd=cost,
            session_key=session_key or None,
        )

    @trace.instrument("llm.call", extract=semconv.llm_call_stream)
    async def _llm_call_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        fallback_models: list[str] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Stream via ``provider.chat_stream`` with the retry and model-fallback
        ladder ``chat_with_retry`` gives the non-streaming path.

        Each model in ``[model, *fallback_models]`` gets the provider's retry
        ladder for ``retryable`` errors; a model exhausted with a
        ``should_fallback`` error hands over to the next one. Output that had
        already reached the callbacks does not end the ladder: a drop after
        the first token is the common failure, not a rare one, and the
        streamed text is discardable. ``on_retry(attempt, total, reason,
        discard)`` fires before each re-run so the outlet can start its live
        text over when ``discard`` says something had been shown.

        Exceptions from the stream are classified like the non-streaming
        path's, so the result is always an ``LLMResponse`` -- one with
        ``finish_reason="error"`` when the ladder is exhausted, carrying the
        last attempt's partial text.
        """
        from opendde_harness.providers import prompt_cache

        provider = self.provider
        delays = getattr(provider, "_CHAT_RETRY_DELAYS", LLMProvider._CHAT_RETRY_DELAYS)
        total_attempts = len(delays) + 1
        model_chain = [model, *(fallback_models or [])]
        can_serve = getattr(provider, "can_serve", None)
        response: LLMResponse | None = None

        async def _announce(attempt: int, reason: str, discard: bool) -> None:
            if on_retry is not None:
                await on_retry(attempt, total_attempts, reason, discard)

        for idx, current_model in enumerate(model_chain):
            if idx and can_serve is not None and not can_serve(current_model or ""):
                logger.warning(
                    "Skipping fallback model={} - this provider instance cannot serve it (wrong vendor)",
                    current_model,
                )
                continue
            if idx and not prompt_cache.accepts_cache_control(current_model or ""):
                messages, tools = prompt_cache.strip(messages, tools)

            for attempt in range(1, total_attempts + 1):
                response, delivered = await self._stream_once(
                    messages, tools, current_model, on_token_delta, on_reasoning_delta, on_tool_event, tool_choice
                )
                if response.finish_reason != "error":
                    return response
                classification = response.error_classification or provider.classify_error(content=response.content)
                response.error_classification = classification
                if not classification.retryable or attempt == total_attempts:
                    break
                delay = provider._jittered(delays[attempt - 1])
                logger.warning(
                    "LLM stream error [{}] (attempt {}/{}) model={}, delivered={}, retrying in {:.1f}s: {}",
                    classification.category,
                    attempt,
                    total_attempts,
                    current_model,
                    delivered,
                    delay,
                    (response.content or "")[:120],
                )
                await _announce(attempt + 1, classification.category, delivered)
                await asyncio.sleep(delay)

            has_next = idx + 1 < len(model_chain)
            if has_next and response.error_classification.should_fallback:
                logger.warning(
                    "LLM stream failed on model={} [{}], falling back to {}: {}",
                    current_model,
                    response.error_classification.category,
                    model_chain[idx + 1],
                    (response.content or "")[:120],
                )
                await _announce(
                    1, f"{response.error_classification.category}, switching to {model_chain[idx + 1]}", delivered
                )
                continue
            return response
        return response  # type: ignore[return-value]  # chain always non-empty

    async def _stream_once(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        on_token_delta: Callable[[str], Awaitable[None]] | None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[LLMResponse, bool]:
        """One streamed attempt, accumulated into an ``LLMResponse``.

        Each non-empty content chunk fires the callback; tool_call fragments
        are merged positionally; the final response object is shape-compatible
        with what ``chat()`` would have returned. The second value says whether
        anything reached a callback, which tells a retry whether the outlet
        has shown text that the re-run must replace.
        """
        content_buf: list[str] = []
        reasoning_buf: list[str] = []
        thinking_blocks: list[dict] = []
        tool_call_slots: list[dict[str, Any]] = []
        final_usage: dict[str, Any] | None = None
        had_error = False
        error_content: str | None = None
        error_classification: ErrorClassification | None = None
        upstream_finish_reason: str | None = None
        delivered = False

        def _error(content: str | None, classification: ErrorClassification | None) -> tuple[LLMResponse, bool]:
            return (
                LLMResponse(
                    content=content,
                    finish_reason="error",
                    error_classification=classification,
                    usage=final_usage or {},
                ),
                delivered,
            )

        # aclosing() guarantees the async generator (and its underlying stream)
        # is closed when an error unwinds the loop, so a stalled stream
        # terminates with a structured error instead of hanging or leaking the
        # connection.
        try:
            async with aclosing(
                self.provider.chat_stream(messages=messages, tools=tools, model=model, tool_choice=tool_choice)
            ) as stream:
                async for delta in stream:
                    if delta.finish_reason == "error":
                        # A non-streaming provider's chat() error, replayed
                        # through the fallback as its single terminal delta.
                        # Its content is the error text, not a token to render
                        # or accumulate -- surface it via error_classification
                        # instead of the normal success collation below.
                        had_error = True
                        error_content = delta.content
                        error_classification = delta.error_classification
                        if delta.usage is not None:
                            final_usage = delta.usage
                        continue
                    if delta.finish_reason:
                        upstream_finish_reason = delta.finish_reason
                    _merge_thinking_blocks(thinking_blocks, getattr(delta, "thinking_blocks", None))
                    reasoning_delta = getattr(delta, "reasoning_content", None)
                    if reasoning_delta:
                        reasoning_buf.append(reasoning_delta)
                        if on_reasoning_delta is not None:
                            delivered = True
                            await on_reasoning_delta(reasoning_delta)
                    if delta.content:
                        content_buf.append(delta.content)
                        if on_token_delta is not None:
                            delivered = True
                            await on_token_delta(delta.content)
                    if delta.builtin_tool_event and on_tool_event is not None:
                        delivered = True
                        event = delta.builtin_tool_event
                        await on_tool_event(str(event["phase"]), event)
                    if delta.tool_call_delta:
                        _merge_tool_call_fragments(
                            tool_call_slots,
                            delta.tool_call_delta,
                        )
                    if delta.usage is not None:
                        final_usage = delta.usage
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            classification = self.provider.classify_error(exc)
            head = getattr(self.provider, "provider_name", None) or None
            summary = format_llm_error(exc, classification, provider=head)
            if not delivered:
                return _error(summary, classification)
            # Part of the reply was shown. If the ladder gives up, this is
            # the turn's text, so say why it stopped instead of ending the
            # turn as if it were complete.
            partial = "".join(content_buf)
            logger.warning(
                "LLM stream interrupted after {} chars [{}]: {}",
                len(partial),
                getattr(classification, "category", "unknown"),
                summary[:200],
            )
            return _error(f'{partial}\n\n[Reply interrupted: {summary}. Send "continue" to resume.]', classification)

        if had_error:
            return _error(error_content, error_classification)

        tool_calls = _finalize_tool_calls(tool_call_slots)

        # No ceiling passed: the provider resolves its own after the loop has
        # handed over the request, so the number this turn carried is not
        # knowable here. Nothing is compared against it, so nothing is missing.
        sent_max_tokens, truncated = flag_truncation(
            finish_reason=upstream_finish_reason,
            usage=final_usage,
            tool_calls=tool_calls,
        )

        finish_reason = upstream_finish_reason or ("tool_calls" if tool_calls else "stop")

        content = "".join(content_buf)
        reasoning_content = "".join(reasoning_buf) or None
        # getattr because the loop accepts duck-typed providers (test stubs and
        # thin adapters implement just chat/chat_stream); absent means the
        # LLMProvider default, False.
        emits_unparsed = getattr(self.provider, "emits_unparsed_reasoning", None)
        if reasoning_content is None and emits_unparsed is not None and emits_unparsed():
            split_reasoning, content = split_orphan_think(content)
            reasoning_content = split_reasoning

        return (
            LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=final_usage or {},
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks or None,
                truncated=truncated,
                max_tokens=sent_max_tokens,
            ),
            delivered,
        )

    @classmethod
    def _emergency_shrink(cls, messages: list[dict]) -> tuple[list[dict], int]:
        """Elide the bodies of older tool-result messages to fit a tighter window.

        Mid-turn context overflow is almost always accumulated tool output, so
        replacing the content of all but the most recent few ``role="tool"``
        messages with a short placeholder frees the most tokens while keeping
        system / user / assistant reasoning intact. Deterministic, no extra LLM
        call. Returns ``(new_messages, num_elided)``; ``num_elided == 0`` means
        there was nothing worth eliding (caller should not bother retrying).
        """
        messages, elided = cls._elide_older_images(messages)

        placeholder = "[earlier tool output elided to fit the context window]"
        tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_idxs) <= cls._SHRINK_KEEP_RECENT_TOOL_RESULTS:
            return messages, elided
        elide = set(tool_idxs[: -cls._SHRINK_KEEP_RECENT_TOOL_RESULTS])
        shrunk: list[dict] = []
        for i, m in enumerate(messages):
            if i in elide and m.get("content") and m.get("content") != placeholder:
                clean = dict(m)
                clean["content"] = placeholder
                shrunk.append(clean)
                elided += 1
            else:
                shrunk.append(m)
        return shrunk, elided

    @staticmethod
    def _demote_tool_images(messages: list[dict]) -> tuple[list[dict], int]:
        """Move images out of tool results into a following user message.

        The recovery for an endpoint that refuses a picture in ``role="tool"``.
        Produces exactly the message list a ``False`` capability verdict would
        have built in the first place, so the retry lands on the already-tested
        placeholder path rather than inventing a third shape.

        A run of consecutive tool messages is one batch answering one assistant
        message, so the pictures pulled out of it are attached once after the last
        of them -- putting one between two tool results leaves a tool_call
        unanswered where the API checks the sequence (measured, see the batching
        comment in the tool loop).

        Returns ``(new_messages, num_demoted)``; ``0`` means no tool result
        carried an image, so the refusal was about something else and the caller
        should not retry.
        """
        out: list[dict] = []
        pending: list[dict] = []
        demoted = 0

        def flush() -> None:
            if pending:
                out.append({"role": "user", "content": list(pending), _ATTACHED_IMAGE_KEY: True})
                pending.clear()

        for m in messages:
            content = m.get("content")
            if m.get("role") != "tool":
                flush()
                out.append(m)
                continue
            if not isinstance(content, list):
                out.append(m)
                continue
            images = [p for p in content if is_image_part(p)]
            if not images:
                out.append(m)
                continue
            clean = dict(m)
            clean["content"] = image_placeholder_text(content)
            out.append(clean)
            pending.extend(images)
            demoted += len(images)
        flush()
        return out, demoted

    @classmethod
    def _elide_older_images(cls, messages: list[dict]) -> tuple[list[dict], int]:
        """Drop inline images from all but the most recent image-bearing message.

        Run before the tool-text pass because an image is by far the densest
        thing in the window -- one costs up to 1568 tokens, which is more than
        most tool outputs -- so dropping a stale picture buys more room than
        eliding several text results, and costs less of what the model still
        needs.

        Not restricted to ``role="tool"``: when the endpoint cannot carry an
        image in a tool result the picture is attached to a following ``user``
        message instead, and that message would otherwise be untouchable here.
        """
        bearing = [
            i
            for i, m in enumerate(messages)
            if isinstance(m.get("content"), list) and any(is_inline_image(p) for p in m["content"])
        ]
        if len(bearing) <= cls._SHRINK_KEEP_RECENT_IMAGES:
            return messages, 0

        target = set(bearing[: -cls._SHRINK_KEEP_RECENT_IMAGES] if cls._SHRINK_KEEP_RECENT_IMAGES else bearing)
        out: list[dict] = []
        elided = 0
        for i, m in enumerate(messages):
            if i not in target:
                out.append(m)
                continue
            clean = dict(m)
            # New list: the caller's messages may still be referenced elsewhere.
            clean["content"] = [
                {"type": "text", "text": "[image elided to fit the context window]"} if is_inline_image(p) else p
                for p in m["content"]
            ]
            out.append(clean)
            elided += 1
        return out, elided

    async def _synthesize_final_on_exhaustion(
        self,
        messages: list[dict],
        model: str | None,
        fallback_models: list[str] | None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
    ) -> str:
        """One tools-disabled LLM call to wrap up after the iteration budget runs out.

        Instead of returning a canned apology, ask the model to summarize what
        it accomplished and deliver its best partial answer. Tools are withheld
        (``tools=None``) so it cannot start another tool call — or an
        ``ask_user`` — at the cliff edge. Falls back to a static message if the
        call errors or comes back empty, so the turn is never left silent.

        When the turn caller wired streaming callbacks, this synthesized reply
        must stream too — otherwise it never reaches a streaming outlet: the
        run_turn boundary only emits a closing ``Text`` when nothing streamed,
        so a non-streamed wrap-up after an already-streamed turn gets dropped.
        """
        synth_messages = messages + [{"role": "user", "content": _MAX_ITER_SYNTHESIS_PROMPT}]
        # The tools are declared and then refused. Withholding them entirely is
        # what a "no more tool calls" turn should mean, but a budget can only be
        # exhausted by calling tools, so this history always carries tool_use and
        # tool_result blocks -- and Anthropic rejects a request holding those
        # with no tools defined. The wrap-up then failed and the turn ended on
        # the canned apology it exists to avoid.
        tool_defs = self.tools.get_definitions()
        try:
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=synth_messages,
                    tools=tool_defs,
                    model=model,
                    fallback_models=fallback_models,
                    on_token_delta=on_token_delta,
                    on_reasoning_delta=on_reasoning_delta,
                    tool_choice="none",
                    on_retry=on_retry,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=synth_messages,
                    tools=tool_defs,
                    tool_choice="none",
                    model=model,
                    fallback_models=fallback_models,
                )
            text = self._strip_think(response.content)
            if response.finish_reason != "error" and text:
                return text
            logger.warning(
                "Max-iter synthesis returned no usable content (finish_reason={})",
                response.finish_reason,
            )
        except Exception as exc:
            logger.warning("Max-iter synthesis call failed: {}", exc)
        fallback = _MAX_ITER_STATIC_FALLBACK.format(n=self.max_iterations)
        # The streamed-success path already delivered its text through
        # ``on_token_delta``; this fallback did not. Push it through the stream
        # too, or the run_turn boundary — which suppresses the closing ``Text``
        # once anything has streamed — would drop it on a streaming outlet.
        if on_token_delta is not None:
            await on_token_delta(fallback)
        return fallback

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session_key: str = "",
        model: str | None = None,
        fallback_models: list[str] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        on_episode_start: Callable[[int], Awaitable[None]] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        drain: Drain | None = None,
    ) -> tuple[str | None, list[str], list[dict], LoopOutcome]:
        """Run the agent iteration loop.

        ``drain``, when wired, is called at the top of each iteration to pull
        any user messages injected mid-turn (BusyPolicy.INJECT) and merge them
        as user turns before the next LLM call. ``session_key`` tags the
        TokenWise usage snapshots.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        used_skill_ids: list[str] = []
        effective_model = model or self.model
        usage: dict[str, Any] = {}
        turn_completion_tokens = 0
        turn_reasoning_tokens = 0

        # Bug2 / decision B — track whether the turn was a normal exit or a
        # max-iter interruption. ``status`` is the only piece read downstream
        # (used to label the shadow-git commit and stamp the ``LoopOutcome``).
        status = "completed"

        # Context-overflow recovery: bound the number of emergency shrinks so a
        # turn that overflows even after eliding can't loop forever.
        compress_retries = 0
        # Image-demotion recovery: bound per turn, same reason.
        image_demote_retries = 0
        # Tool-failure-loop break (#1b): track consecutive hard failures of the
        # same tool *with the same kind of error* across iterations; nudge once
        # per fresh streak, bounded/turn.
        loop_fail_key: tuple[str, str] | None = None
        loop_fail_streak = 0
        loop_nudges = 0
        # Empty-response recovery state, local to the turn — the AgentLoop is a
        # long-lived singleton shared across sessions, so per-instance counters
        # would leak across turns; resetting here gives clean per-turn budgets.
        prev_had_tool_calls = False
        post_tool_nudges = 0
        prefill_retries = 0
        empty_retries = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(
                "Iteration {}/{} model={}",
                iteration,
                self.max_iterations,
                effective_model,
            )

            # Mark the episode boundary (one per model call) so an outlet can
            # group this call's reasoning + text + tools into a single step.
            if on_episode_start is not None:
                await on_episode_start(iteration - 1)

            # Merge any INJECT-ed user messages (BusyPolicy.INJECT) before this
            # iteration's LLM call. Media-carrying injects keep their file
            # paths in the text so nothing is silently dropped.
            if drain is not None:
                for inj in drain():
                    inj_text = inj.text or ""
                    inj_paths = [m.path for m in inj.media]
                    if inj_paths:
                        prefix = inj_text + "\n" if inj_text else ""
                        inj_text = f"{prefix}[injected message; attached files: {', '.join(inj_paths)}]"
                    if inj_text:
                        messages.append({"role": "user", "content": inj_text})
                        logger.info("inject: merged a mid-turn user message")

            tool_defs = self.tools.get_definitions()

            # TokenWise before-hook: strategies may rewrite messages, tools,
            # or model (e.g. ToolSearchStrategy withholds tool schemas).
            call_messages, call_tools, call_model = await self.strategies.before_llm_call(
                messages,
                tool_defs,
                effective_model,
            )
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
                    fallback_models=fallback_models,
                    on_token_delta=on_token_delta,
                    on_reasoning_delta=on_reasoning_delta,
                    on_tool_event=on_tool_event,
                    on_retry=on_retry,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
                    fallback_models=fallback_models,
                )
            # TokenWise after-hook: strategies observe the response for
            # usage tracking, budget enforcement, etc. Errors are swallowed.
            usage_snapshot = self._build_usage_snapshot(response, call_model, session_key)
            await self.strategies.after_llm_call(
                {
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                },
                usage_snapshot,
            )
            # The final iteration's accounting reaches the outcome (the TUI
            # attaches it to message.complete). Wire-contract fields
            # (prompt_tokens / completion_tokens / total_tokens), not the
            # agent-internal snapshot with model / cache fields.
            if response.usage:
                prompt_tokens = int(response.usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(response.usage.get("completion_tokens", 0) or 0)
                turn_completion_tokens += completion_tokens
                turn_reasoning_tokens += _reasoning_tokens_of(response.usage)
                if on_usage is not None:
                    await on_usage(turn_completion_tokens, turn_reasoning_tokens, iteration)
                # The call's own model, which is not always ``self.model`` (a
                # strategy rewrote it, or the chain fell back). Unknown to every
                # table, 0 tells the UI to show its empty state rather than a
                # number that isn't this model's. Off the event loop only for
                # LiteLLM's import; no tier reaches the network.
                window = await asyncio.to_thread(self.resolve_window, call_model)
                context_max = window.tokens or 0
                context_used = prompt_tokens + completion_tokens
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(response.usage.get("total_tokens", 0) or 0),
                    "cost_usd": usage_snapshot.estimated_cost_usd,
                    "context_max": context_max,
                    "context_source": window.source,
                    "context_used": context_used,
                    "context_percent": round(100 * context_used / context_max) if context_max else 0,
                }

            # Context-window overflow recovery: the structured classifier flags
            # should_compress (a smaller window won't help, but eliding the bulk
            # of accumulated tool output will). Shrink in place and retry this
            # iteration instead of surfacing it as a fatal error. Bounded.
            cls_ = response.error_classification
            if (
                response.finish_reason == "error"
                and cls_ is not None
                and cls_.should_compress
                and compress_retries < self._MAX_COMPRESS_RETRIES
            ):
                shrunk, elided = self._emergency_shrink(messages)
                if elided > 0:
                    messages = shrunk
                    compress_retries += 1
                    iteration -= 1  # the overflowed call did no work; don't bill it
                    logger.warning(
                        "Context overflow; elided {} old tool result(s), retrying ({}/{})",
                        elided,
                        compress_retries,
                        self._MAX_COMPRESS_RETRIES,
                    )
                    continue

            # Image-in-tool-result refused: this endpoint takes a picture only in
            # a user message. Rebuild onto the placeholder path -- the shape a
            # False capability verdict would have produced -- and retry this
            # iteration. Also cache the verdict so the rest of the process stops
            # paying for the attempt: the static table in `capabilities` guessed
            # wrong, and this is how it self-corrects.
            if (
                response.finish_reason == "error"
                and cls_ is not None
                and cls_.should_drop_tool_images
                and image_demote_retries < self._MAX_IMAGE_DEMOTE_RETRIES
            ):
                demoted_messages, demoted = self._demote_tool_images(messages)
                if demoted > 0:
                    messages = demoted_messages
                    self._image_tool_result_ok[call_model or effective_model] = False
                    image_demote_retries += 1
                    iteration -= 1  # the refused call did no work; don't bill it
                    logger.warning(
                        "Endpoint refused {} image(s) in a tool result; moved them "
                        "to a user message and retrying ({}/{})",
                        demoted,
                        image_demote_retries,
                        self._MAX_IMAGE_DEMOTE_RETRIES,
                    )
                    continue

            if response.finish_reason == "error":
                # Before the tool branch: a provider can report a failure and
                # still carry the calls it had started, and dispatching those
                # ran tools on behalf of a response the backend had given up on
                # while the error itself was never logged, retried or shown.
                clean = self._strip_think(response.content)
                logger.error("LLM returned error: {}", (clean or "")[:200])
                final_content = clean or "Sorry, I encountered an error calling the AI model."
                status = "error"
                break

            if response.has_tool_calls:
                abort_action = False
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                # Images this transport cannot carry inside a tool result.
                # Collected across the whole batch and attached once *after* the
                # last tool result: a user message sitting between two tool
                # results leaves an assistant tool_call unanswered at the point
                # the API validates the sequence. Measured on gpt-4o, 2026-07-31,
                # two tool calls with the picture from the first:
                #   tool(c1), user, tool(c2) -> 400 "An assistant message with
                #     'tool_calls' must be followed by tool messages responding
                #     to each 'tool_call_id'"
                #   tool(c1), tool(c2), user -> 200, and the model named the
                #     image's colour
                # Anthropic accepts both, so this only bites on Chat Completions
                # -- which is the only transport that takes this path at all.
                pending_images: list[dict[str, Any]] = []
                tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call_index, tool_call in enumerate(response.tool_calls):
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    if on_tool_event is not None:
                        _tool = self.tools.get(tool_call.name)
                        await on_tool_event(
                            "start",
                            {
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                # Tool-authored call label; None -> UI derives one.
                                "display": _tool.display_call(tool_call.arguments) if _tool else None,
                            },
                        )
                    if tool_call.name == "exec":
                        exec_tool = self.tools.get("exec")
                        if isinstance(exec_tool, ExecTool):
                            exec_tool.set_tool_call_id(tool_call.id)
                    tool_t0 = time.monotonic()
                    result = await self.tools.execute(tool_call.name, tool_call.arguments, run_meta=tool_call.run_meta)
                    duration_ms = int((time.monotonic() - tool_t0) * 1000)
                    # The registry already unwrapped any ToolResult: `result` is
                    # the model-facing text, with the optional display string
                    # riding along on it (ToolOutput). The model always gets the
                    # model text; the UI preview prefers the display string.
                    model_text = str(result)
                    if tool_call.name == "use_skill" and not model_text.lstrip().lower().startswith("error:"):
                        skill_id = tool_call.arguments.get("skill_id")
                        if isinstance(skill_id, str) and "/" in skill_id and skill_id not in used_skill_ids:
                            used_skill_ids.append(skill_id)
                    display_src = getattr(result, "display_text", None) or model_text
                    # The log stays one line; the UI event keeps newlines so a
                    # tool that reports several items (e.g. ask_user's
                    # question -> answer pairs) renders one row each.
                    preview = display_src[:200]
                    logger.info(
                        "Tool result: {} duration={}ms result={}",
                        tool_call.name,
                        duration_ms,
                        preview.replace("\n", " ")[:200],
                    )
                    if on_tool_event is not None:
                        await on_tool_event(
                            "complete",
                            {
                                "tool_call_id": tool_call.id,
                                "result_preview": preview,
                                "truncated": len(display_src) > 200,
                            },
                        )
                    model_text, blocks, attach_blocks = self._route_result_images(
                        model_text, getattr(result, "blocks", None), call_model or effective_model
                    )
                    if blocks:
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, model_text, blocks
                        )
                    else:
                        # Keep the long-standing 4-arg call for text results so no
                        # existing caller or test double sees a signature change.
                        messages = self.context.add_tool_result(messages, tool_call.id, tool_call.name, model_text)
                    if attach_blocks:
                        pending_images.extend(attach_blocks)
                    if getattr(result, "abort_action", False):
                        abort_action = True
                        # A single assistant message may contain several parallel
                        # tool calls (for example ``rm`` followed by a Python
                        # fallback). Once policy terminates the action, none of
                        # the siblings may execute. We must nevertheless append
                        # one result for every advertised call id: OpenAI-style
                        # providers reject conversation history containing an
                        # assistant tool call without its matching tool result.
                        for skipped_call in response.tool_calls[tool_call_index + 1 :]:
                            messages = self.context.add_tool_result(
                                messages,
                                skipped_call.id,
                                skipped_call.name,
                                (
                                    "Error: Tool call was not executed because a prior safety "
                                    "decision terminated this action."
                                ),
                            )
                        break
                    # #1b Track consecutive same-tool deterministic failures
                    # (transient errors excluded — a retry would clear those).
                    if is_hard_tool_failure(model_text):
                        failure_key = (tool_call.name, failure_class(model_text))
                        if failure_key == loop_fail_key:
                            loop_fail_streak += 1
                        else:
                            loop_fail_key, loop_fail_streak = failure_key, 1
                    else:
                        loop_fail_key, loop_fail_streak = None, 0

                if abort_action:
                    # A normal tool result starts another model iteration. That
                    # is specifically unsafe here: the next plan can translate
                    # the rejected operation into an equivalent interpreter,
                    # script, or tool call. Finish the turn in runtime code and
                    # expose only the non-destructive continuation question.
                    # Streaming callers need the explicit callback because no
                    # final model response exists to generate token deltas.
                    messages = self.context.add_assistant_message(messages, _ABORTED_ACTION_REPLY)
                    final_content = _ABORTED_ACTION_REPLY
                    if on_token_delta is not None:
                        await on_token_delta(_ABORTED_ACTION_REPLY)
                    break

                # #1b Failure-loop break: the same tool failed deterministically
                # `threshold` times running → append a change-approach nudge to
                # the last tool result so the model stops repeating a dead call.
                if (
                    loop_fail_streak >= self._LOOP_BREAK_THRESHOLD
                    and loop_nudges < self._LOOP_BREAK_MAX
                    and messages
                    and messages[-1].get("role") == "tool"
                ):
                    loop_nudges += 1
                    messages[-1]["content"] = (
                        str(messages[-1].get("content", ""))
                        + "\n\n"
                        + loop_break_nudge(loop_fail_key[0], loop_fail_streak, loop_fail_key[1])
                    )
                    loop_fail_streak = 0  # fire once per fresh streak
                # After the nudge above, which needs the last message to still be
                # the tool result it appends to. Also after the abort_action
                # branch, which ends the turn in runtime code -- there is no
                # further model call to show a picture to, so an aborted action
                # deliberately drops it rather than leaving it dangling.
                if pending_images:
                    messages.append({"role": "user", "content": pending_images, _ATTACHED_IMAGE_KEY: True})
                prev_had_tool_calls = True
            else:
                # Error responses are handled above and never persisted to
                # session history -- they poison the context and cause
                # permanent 400 loops (#1303).
                clean = self._strip_think(response.content)
                # Empty-response recovery: an empty assistant turn would
                # otherwise break out here and surface a "no response to give"
                # dud. Try to recover before giving up. Synthetic scaffolding is
                # marked ``_recovery_synthetic`` and stripped before persistence
                # / extraction so it can't poison future context.
                action = classify_empty_response(
                    response,
                    clean,
                    prev_had_tool_calls=prev_had_tool_calls,
                    nudges_done=post_tool_nudges,
                    prefill_retries=prefill_retries,
                    empty_retries=empty_retries,
                    limits=self._recovery_limits,
                )
                if action is RecoveryAction.PREFILL:
                    prefill_retries += 1
                    logger.warning(
                        "empty-recovery: thinking-only prefill {}/{}",
                        prefill_retries,
                        self._recovery_limits.thinking_prefill_max_retries,
                    )
                    # Re-feed the model its own reasoning (not stripped) so it
                    # continues into the body. Marked synthetic → dropped before
                    # persistence/extraction. The provider's key allowlist decides
                    # what reaches the wire: everyone but Anthropic drops the
                    # thinking blocks, and Anthropic needs them.
                    messages = self.context.add_assistant_message(
                        messages,
                        response.content,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    messages[-1]["_recovery_synthetic"] = True
                    prev_had_tool_calls = False
                    continue
                if action is RecoveryAction.NUDGE:
                    post_tool_nudges += 1
                    logger.warning("empty-recovery: post-tool empty nudge")
                    # The (empty) assistant must sit between the tool result and
                    # the nudge — a bare tool→user sequence is a 400 on most APIs.
                    messages = self.context.add_assistant_message(messages, "(empty)")
                    messages[-1]["_recovery_synthetic"] = True
                    messages.append({"role": "user", "content": POST_TOOL_NUDGE, "_recovery_synthetic": True})
                    prev_had_tool_calls = False
                    continue
                if action is RecoveryAction.RETRY:
                    empty_retries += 1
                    logger.warning(
                        "empty-recovery: plain empty retry {}/{}",
                        empty_retries,
                        self._recovery_limits.empty_content_max_retries,
                    )
                    prev_had_tool_calls = False
                    continue

                messages = self.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached; synthesizing final answer", self.max_iterations)
            # Exhaustion is two orthogonal facts, not an either/or:
            #   1. The turn did NOT complete — tag it ``interrupted`` so the
            #      shadow-git checkpoint commit is labelled and the next turn's
            #      recovery prompt can surface the sha + edited files to resume.
            #   2. The user still deserves a useful reply NOW — so, checkpoint
            #      or not, synthesize a best-effort wrap-up (one tools-disabled
            #      call summarising what was done and what's left) instead of a
            #      canned apology. Synthesis falls back to a static message
            #      internally if the call fails, so the turn is never silent.
            status = "interrupted"
            final_content = await self._synthesize_final_on_exhaustion(
                messages,
                effective_model,
                fallback_models,
                on_token_delta=on_token_delta,
                on_reasoning_delta=on_reasoning_delta,
                on_retry=on_retry,
            )
            # Persist the wrap-up into history like any normal final reply.
            # Persistence downstream reads only the returned ``messages`` list,
            # so without this the synthesized answer reaches the user via the
            # stream yet never enters the conversation — the next turn (notably
            # an interrupted-turn resume) could not see what was summarized. The
            # synthesis prompt itself stays local to the helper, so only the
            # reply lands here.
            if final_content:
                messages = self.context.add_assistant_message(messages, final_content)

        # Drop transient empty-recovery scaffolding before persistence /
        # extraction / return — refactor/OpenDDE Harness's empty-recovery marks
        # synthetic nudge/prefill messages with ``_recovery_synthetic``; strip
        # them so they never persist. (Embedded ``_trigger_local_extraction``
        # was retired when the memory service was integrated; the after-turn pipeline —
        # ``context_engine.after_turn`` + ``backend.store`` in
        # ``_process_message`` — owns extraction now.)
        # Attached-image messages are dropped here for the same reason and at the
        # same point: the returned list feeds persistence, ``after_turn``
        # extraction and ``backend.store`` alike, so filtering once upstream of
        # all three is the only place that covers them.
        _transient = ("_recovery_synthetic", _ATTACHED_IMAGE_KEY)
        if any(any(m.get(k) for k in _transient) for m in messages):
            messages = [m for m in messages if not any(m.get(k) for k in _transient)]

        # Phase B-1 (memory service integration): embedded extraction (the
        # ``_trigger_local_extraction`` / ``SkillService.on_execution`` path)
        # was retired here in favor of the after-turn pipeline owned by the
        # caller — ``context_engine.after_turn`` + ``backend.store`` +
        # ``backend.feedback`` run from ``_process_message``. We surface
        # ``outcome.status`` so that pipeline can gate on completion later.

        outcome = LoopOutcome(status=status, used_skill_ids=used_skill_ids, usage=usage)
        if self._checkpoint is not None:
            # Per-turn snapshot: one commit covering all of this turn's edits,
            # for both normal and interrupted exits (matches Claude Code/Cursor
            # granularity). Best-effort — commit_turn never raises.
            label = f"turn {session_key or 'anon'} [{status}]"
            cid, changed = await self._checkpoint.commit_turn(label)
            outcome.checkpoint_id = cid
            if status == "interrupted":
                outcome.edited_files = changed

        return final_content, tools_used, messages, outcome

    async def run(self) -> None:
        """Bring the agent runtime up and stay alive.

        Turns arrive through the spine (``run_turn``); this coroutine no longer
        drains an inbound bus. It starts the executor / MCP, then
        idles on ``self._running`` so the gateway can gather it as a long-lived
        task and tear it down via ``stop()`` on shutdown.
        """
        self._running = True
        try:
            await self._start_executor()
            await self._connect_mcp()
        except SandboxInitError as exc:
            logger.error("Sandbox failed to start: {}", exc)
            await self.close_executor()
            self._running = False
            return
        except Exception:
            await self.close_executor()
            raise
        logger.info("Agent loop started")

        while self._running:
            await asyncio.sleep(1.0)

    @property
    def is_processing(self) -> bool:
        """True while a turn is being dispatched under the global lock."""
        return self._processing_lock.locked()

    def _notify_turn_complete(self) -> None:
        for callback in self.on_turn_complete:
            try:
                callback()
            except Exception:
                logger.exception("on_turn_complete callback failed")

    async def close_mcp(self) -> None:
        """Close MCP connections and the sandbox executor."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None
        self._mcp_connected = False  # reset so _connect_mcp() can reconnect after close
        self._mcp_connecting = False  # reset so a concurrent caller isn't permanently blocked
        await self.close_executor()  # always runs, even when no MCP servers are configured

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    @trace.instrument("session.turn", seed=semconv.turn_seed, on_open=semconv.turn_open, extract=semconv.turn)
    async def _process_message(
        self,
        req: TurnRequest,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        on_episode_start: Callable[[int], Awaitable[None]] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        origin: Origin | None = None,
        drain: Drain | None = None,
    ) -> TurnReply:
        """Process a single turn request and return its reply.

        ``reply.content`` is ``None`` for a silent turn (the message tool
        already sent, or a hook short-circuit chose to return None).
        ``origin`` is the spine TurnRequest's origin.
        """
        from opendde_harness.agent.hook import AgentHookContext

        channel = req.source.channel
        sender_id = req.source.sender_id
        chat_id = req.source.chat_id
        content = req.text
        metadata = dict(req.source.extras)
        media_paths = [m.path for m in req.media]
        msg_session_key = req.conversation or f"{channel}:{chat_id}"

        # ``before_user_inbound`` chain: observer hooks record the inbound,
        # short-circuit hooks halt processing and return their reply directly.
        # Skipped for subagent turns (by origin): a result re-injection is not
        # genuine user input.
        if len(self.hooks) > 0 and origin not in _SKIP_USER_INBOUND_ORIGINS:
            decision = await self.hooks.before_user_inbound(
                AgentHookContext(session_key=msg_session_key, turn_request=req)
            )
            if decision.short_circuit_result is not None:
                text, media = decision.short_circuit_result
                return TurnReply(content=text, media=list(media or []))

        preview = content[:80] + "..." if len(content) > 80 else content
        logger.info("Processing message from {}:{}: {}", channel, sender_id, preview)

        key = session_key or msg_session_key
        session = self.sessions.get_or_create(key)

        if (handled := await self._handle_slash_command(session, content)) is not None:
            return TurnReply(content=handled)
        if not self.context_engine.owns_compaction:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(channel, chat_id, metadata.get("message_id"), session_key=key)
        # ask_user keys by the true conversation_id (== the lane / gate key),
        # which is topic-aware (req.conversation), not just channel:chat_id.
        if (ask_tool := self.tools.get("ask_user")) and isinstance(ask_tool, AskUserTool):
            ask_tool.set_context(key)

        # Skill selection is the context engine's (SkillForgeRouter inside
        # ``assemble``); the injected ids come back in the assembled metadata.
        selected_skills = None
        initial_messages = await self._assemble_context_messages(
            session=session,
            session_key=key,
            current_message=content,
            media=media_paths if media_paths else None,
            channel=channel,
            chat_id=chat_id,
            selected_skills=selected_skills or None,
        )

        turn_start_idx = len(initial_messages) - 1
        final_content, _, all_msgs, outcome = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress,
            session_key=key,
            on_token_delta=on_token_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_tool_event=on_tool_event,
            on_episode_start=on_episode_start,
            on_retry=on_retry,
            on_usage=on_usage,
            drain=drain,
        )
        self._stash_recovery(key, outcome)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # ``after_send`` chain modifies the outbound text. Skipped for
        # system-originated turns (a subagent result) so their reply is not
        # rewritten on the way out.
        if len(self.hooks) > 0 and origin not in _SKIP_AFTER_SEND_ORIGINS:
            send_decision = await self.hooks.after_send(
                AgentHookContext(session_key=key, outbound_content=final_content)
            )
            if send_decision.modified_content is not None:
                final_content = send_decision.modified_content

        await self._after_turn(session, key, all_msgs[turn_start_idx:], final_content, selected_skills, outcome)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", channel, sender_id, preview)
        return TurnReply(content=final_content, usage=outcome.usage)

    async def _handle_slash_command(self, session: Session, content: str) -> str | None:
        """Reply text for a slash command, or ``None`` when ``content`` is not one."""
        cmd = content.strip().lower()
        if cmd == "/new":
            failed = "Memory archival failed, session not cleared. Please try again."
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return failed
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return failed
            session.clear()
            await asyncio.to_thread(self.sessions.save, session)
            self.sessions.invalidate(session.key)
            return "New session started."
        if cmd == "/help":
            return "\n".join(
                [
                    f"{__logo__} OpenDDE Harness commands:",
                    "/new — Start a new conversation",
                    "/stop — Stop the current task",
                    "/restart — Restart the bot",
                    "/help — Show available commands",
                ]
            )
        return None

    async def _after_turn(
        self,
        session: Session,
        key: str,
        turn_messages: list[dict],
        final_content: str,
        selected_skills: list[Any] | None,
        outcome: LoopOutcome,
    ) -> None:
        """Persist the turn and run the after-turn pipeline: engine bookkeeping,
        plugin-side indexing (``backend.store``), skill-usage feedback and
        markdown compaction."""
        self._save_turn(session, turn_messages, 0)
        await asyncio.to_thread(self.sessions.save, session)
        await self.context_engine.after_turn(key, {"final_content": final_content, "messages": turn_messages})
        await self._dispatch_backend_store(key, turn_messages)
        await self._dispatch_backend_feedback(
            key,
            self._collect_injected_skill_ids(selected_skills),
            outcome.used_skill_ids,
        )
        if not self.context_engine.owns_compaction:
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if entry.get("_recovery_synthetic"):
                continue  # #1a synthetic recovery nudge — never persist scaffolding
            if entry.get(_ATTACHED_IMAGE_KEY):
                # Already filtered upstream; kept because this is the last gate
                # before a write that cannot be undone, unlike the code above it.
                continue
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, list):
                # A multimodal tool result. Images must never reach the JSONL:
                # a single one adds megabytes that are then replayed on every
                # resume and re-fed to the model, and unlike a code bug that is
                # not revertible once written. The char cap below cannot catch
                # it either — it guards `str` content only.
                content = _strip_inline_images(content)
                entry["content"] = content
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = [
                        c
                        for c in _strip_inline_images(content)
                        if not (
                            isinstance(c, dict)
                            and c.get("type") == "text"
                            and isinstance(c.get("text"), str)
                            and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
                        )
                    ]
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", self._now_fn().isoformat())
            session.record(entry)
        session.updated_at = self._now_fn()

    async def run_turn(
        self,
        req: TurnRequest,
        emit: Emit,
        drain: Drain,
        *,
        stream: bool = True,
        inline_tool_stream: bool = False,
    ) -> "SpineTurnOutcome":
        """Turn boundary for a live ``/model`` switch; see ``_run_turn``.

        A parked switch is adopted here, before the turn reads
        ``self.provider`` for the first time, and the count kept for the
        duration is what parks the next one. Wrapping rather than
        snapshotting because the provider is read from ``self`` at eight
        sites in this module and by the context engine and consolidator
        underneath them -- one boundary covers all of them, a snapshot
        would have to be threaded through each.

        Both ends gate on zero, because turns overlap: a user turn and a
        proactive turn hold slots in separate pools. Adopting on the way in
        would otherwise land a switch parked for a turn that is still
        running, and clearing a flag on the way out would unpark it just as
        wrongly. Adopting again on the last exit is what keeps a park from
        outliving the turns it was waiting on.
        """
        if self._turns_in_flight == 0:
            self._adopt_pending_provider()
        self._turns_in_flight += 1
        try:
            return await self._run_turn(req, emit, drain, stream=stream, inline_tool_stream=inline_tool_stream)
        finally:
            self._turns_in_flight -= 1
            if self._turns_in_flight == 0:
                self._adopt_pending_provider()

    async def _run_turn(
        self,
        req: TurnRequest,
        emit: Emit,
        drain: Drain,
        *,
        stream: bool = True,
        inline_tool_stream: bool = False,
    ) -> "SpineTurnOutcome":
        """Spine-native turn entry: consume a TurnRequest, fan the agent's output
        onto the single ``emit``, return the spine's TurnOutcome.

        Named ``run_turn`` rather than ``run``: ``run`` is the runtime keep-alive
        (executor / MCP up, then idle). A spine runner calls the public
        ``run_turn`` to satisfy the TurnRunner protocol.

        ``stream`` is the assembly switch: a streaming outlet (TUI) wires it
        True so the reply goes out as StreamDelta and dissolves (no trailing
        Text); a non-streaming outlet (REPL) wires it False so the reply is one
        Text. It gates both LLM callbacks (the loop streams when either is
        wired) and the message-tool routing, so the whole reply travels one way.

        Exceptions propagate so the lane turns them into TurnFailed.

        The outcome carries the turn's full token accounting in
        ``usage_detail`` (cost / context, richer than the three-field
        ``usage``) and the reply text in ``text`` -- an observation copy, the
        reply itself goes out via ``emit``.

        ``drain`` pulls user messages injected mid-turn (BusyPolicy.INJECT); it
        is threaded into the agent loop and consumed at the top of each iteration.
        """
        from opendde_harness.spine.events import (
            EpisodeStart,
            MediaOut,
            Notice,
            NoticeKind,
            Reasoning,
            StreamDelta,
            Text,
            ToolEvent,
            ToolPhase,
            TurnRetry,
            TurnUsage,
            Usage,
        )
        from opendde_harness.spine.message import Media
        from opendde_harness.spine.runner import TurnOutcome

        cid = req.conversation or f"{req.source.channel}:{req.source.chat_id}"

        # Verbatim delivery (deliver_text — see TurnRequest). Persist before
        # emit, mirroring the normal turn's save-then-reply order, so a save
        # failure never leaves the user a delivered message no turn recorded.
        # The after-turn work a normal turn does is reduced to what a no-model
        # delivery needs: backend.store indexes the report; after_turn is a
        # no-op without a turn_id; consolidation is the curator's job on its
        # next assemble.
        if req.deliver_text is not None:
            session = self.sessions.get_or_create(cid)
            msg = {"role": "assistant", "content": req.deliver_text}
            self._save_turn(session, [msg], 0)
            await asyncio.to_thread(self.sessions.save, session)
            await self._dispatch_backend_store(cid, [msg])
            await emit(Text(content=req.deliver_text))
            return TurnOutcome(
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                explicit_reply=True,
                text=req.deliver_text,
            )

        streamed = False
        reply_text: str | None = None

        async def on_token(text: str) -> None:
            nonlocal streamed
            if not text:
                return
            streamed = True
            await emit(StreamDelta(delta=text))

        async def on_reasoning(text: str) -> None:
            if text:
                await emit(Reasoning(content=text))

        async def on_episode(index: int) -> None:
            await emit(EpisodeStart(index=index))

        async def on_retry(attempt: int, total: int, reason: str, discard: bool) -> None:
            nonlocal streamed
            if discard:
                streamed = False
            await emit(TurnRetry(attempt=attempt, total=total, reason=reason, discard=discard))

        async def on_usage(completion_tokens: int, reasoning_tokens: int, calls: int) -> None:
            await emit(TurnUsage(completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens, calls=calls))

        async def on_tool(phase: str, info: dict[str, Any]) -> None:
            if phase == "start":
                await emit(
                    ToolEvent(
                        phase=ToolPhase.START,
                        tool_call_id=info["tool_call_id"],
                        name=info["name"],
                        arguments=info["arguments"],
                        display=info.get("display"),
                    )
                )
            else:
                await emit(
                    ToolEvent(
                        phase=ToolPhase.COMPLETE,
                        tool_call_id=info["tool_call_id"],
                        result_preview=info["result_preview"],
                        truncated=info["truncated"],
                    )
                )

        async def on_progress(text: str, tool_hint: bool = False) -> None:
            # Keep the progress/tool-hint distinction so an outlet can gate each on
            # its own config flag (send_progress vs send_tool_hints) — tool-hint
            # text rides NoticeKind.TOOL_HINT, progress rides PROGRESS. Outlets
            # that don't render either eat both kinds anyway.
            if text:
                await emit(
                    Notice(
                        kind=NoticeKind.TOOL_HINT if tool_hint else NoticeKind.PROGRESS,
                        detail=text,
                    )
                )

        async def _emit_media(paths: list[str]) -> None:
            await emit(
                MediaOut(media=tuple(Media(path=p, mime="application/octet-stream", kind="file") for p in paths))
            )

        try:
            await self._start_executor()
            await self._connect_mcp()
            reply = await self._process_message(
                req,
                session_key=cid,
                on_progress=on_progress,
                on_token_delta=on_token if stream else None,
                on_reasoning_delta=on_reasoning if stream else None,
                on_tool_event=on_tool,
                on_episode_start=on_episode if stream else None,
                on_retry=on_retry,
                on_usage=on_usage if stream else None,
                origin=req.origin,
                drain=drain,
            )
        except Exception:
            await self.close_executor()
            raise

        # Single return->emit boundary. MediaOut is independent of the stream
        # and precedes Text.
        if reply.content is not None:
            if reply.media:
                await _emit_media(reply.media)
            if not streamed and reply.content:
                await emit(Text(content=reply.content))
            if reply.content:
                reply_text = reply.content

        usage = Usage(
            prompt_tokens=int(reply.usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(reply.usage.get("completion_tokens", 0) or 0),
            total_tokens=int(reply.usage.get("total_tokens", 0) or 0),
        )
        return TurnOutcome(
            usage=usage,
            explicit_reply=reply.content is not None,
            usage_detail=dict(reply.usage),
            text=reply_text,
        )


def _reasoning_tokens_of(usage: dict[str, Any]) -> int:
    """The reasoning share of a call's output, where the vendor reports one.

    OpenAI files it under ``completion_tokens_details.reasoning_tokens``;
    LiteLLM keeps that shape for every vendor that has an equivalent. Zero
    where nothing is reported, which is not the same as no reasoning.
    """
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens", 0) or 0)
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0


def _merge_thinking_blocks(blocks: list[dict], incoming: list[dict] | None) -> None:
    """Accumulate Anthropic thinking blocks arriving one delta at a time.

    The text arrives in fragments and the signature closes them, in a delta of
    its own carrying empty text, so the fragments fold into one block and the
    signature lands on it. Forwarding the raw per-delta list would send
    Anthropic dozens of unsigned scraps. Redacted blocks are opaque and stay
    separate entries, exactly as they arrived.
    """
    if not incoming:
        return
    for block in incoming:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "redacted_thinking":
            blocks.append(dict(block))
            continue
        target = next((item for item in blocks if item.get("type") != "redacted_thinking"), None)
        if target is None:
            target = {"type": "thinking"}
            blocks.append(target)
        for key, value in block.items():
            if key == "thinking" and isinstance(value, str):
                target["thinking"] = f"{target.get('thinking', '')}{value}"
            elif value not in (None, ""):
                target[key] = value


def _merge_tool_call_fragments(
    slots: list[dict[str, Any]],
    delta: dict[str, Any],
) -> None:
    """Merge a single chat_stream tool_call_delta into accumulator slots.

    Each slot follows the shape ``{id, function: {name, arguments_buf: [str]}}``.
    Per provider chunk semantics (OpenAI/LiteLLM): each tool call fragment
    carries an ``index`` field; ``id`` / ``function.name`` typically appear in
    the first fragment for that index, ``function.arguments`` is a JSON string
    streamed in pieces.

    Respects the ``index`` field so parallel multi-tool streams do not
    collapse into ``slots[0]``. Fragments without an ``index`` default to 0
    (single-tool case, backward-compatible).
    """
    incoming = delta.get("tool_calls") or []
    if not incoming:
        return
    for tc in incoming:
        idx = int(tc.get("index", 0) or 0)
        while len(slots) <= idx:
            slots.append(
                {"id": None, "function": {"name": None, "arguments_buf": []}, "fields": None, "fn_fields": None}
            )
        slot = slots[idx]
        if tc.get("id") and not slot["id"]:
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name") and not slot["function"]["name"]:
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments_buf"].append(fn["arguments"])
        # A provider that signs its tool calls (Gemini) sends the signature in
        # these fields, and the next request has to carry them back. Dropped
        # here, the streamed call went back unsigned while the non-streamed one
        # did not, which is the kind of difference nothing downstream can see.
        if tc.get("provider_specific_fields") and not slot.get("fields"):
            slot["fields"] = tc["provider_specific_fields"]
        if fn.get("provider_specific_fields") and not slot.get("fn_fields"):
            slot["fn_fields"] = fn["provider_specific_fields"]


def _finalize_tool_calls(slots: list[dict[str, Any]]) -> list[ToolCallRequest]:
    """Convert accumulator slots into final ToolCallRequest list.

    A slot whose arguments do not parse is flagged on that call, not on the
    turn: an unparseable blob is evidence about one call, and reducing it to
    "something in this turn failed" would leave the loop guessing which. An
    incomplete JSON blob is also the one piece of evidence that needs no
    cooperation from the backend, which matters where a backend reports a
    clean stop on a cut-off reply.
    """
    result: list[ToolCallRequest] = []
    for slot in slots:
        name = slot["function"]["name"]
        if not name:
            continue
        args_text = "".join(slot["function"]["arguments_buf"])
        repaired = False
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": args_text}
            repaired = True
        result.append(
            ToolCallRequest(
                id=slot["id"] or "",
                name=name,
                arguments=args,
                provider_specific_fields=slot.get("fields"),
                function_provider_specific_fields=slot.get("fn_fields"),
                run_meta=RunMeta(arguments_repaired=True) if repaired else None,
            )
        )
    return result
