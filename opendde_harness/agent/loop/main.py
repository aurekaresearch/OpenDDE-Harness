"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from opendde_harness import __logo__
from opendde_harness.agent.context import ContextBuilder
from opendde_harness.agent.loop import context_policy, model_call, tool_batch
from opendde_harness.agent.loop.accounting import (
    FinalCall,
    TurnAccounting,
    build_usage_snapshot,
    charge,
    reasoning_tokens_of,
)
from opendde_harness.agent.loop.bindings import SessionBindings
from opendde_harness.agent.loop.factory import (
    AgentLoopSettings,
    apply_disabled_tools,
    build_checkpoint_service,
    build_tool_registry,
    checkpoint_active,
)
from opendde_harness.agent.loop.failure_streak import FailureStreak
from opendde_harness.agent.loop.recovery import (
    POST_TOOL_NUDGE,
    RecoveryAction,
    RecoveryState,
    strip_think,
    synthesize_on_exhaustion,
)
from opendde_harness.agent.subagent import SubagentManager
from opendde_harness.agent.tools.ask_user import AskUserTool
from opendde_harness.memory_engine.dispatch import STORE_DRAIN_BUDGET_S, ExtractionDispatcher
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import (
    LLMProvider,
    LLMResponse,
)
from opendde_harness.providers.binding import (
    ModelBinding,
    use_binding,
)
from opendde_harness.providers.rates import Resolved
from opendde_harness.sandbox import DirectExecutor, SandboxExecutor, SandboxInitError
from opendde_harness.session.journal import TurnJournal, record_delivery, sanitized_record
from opendde_harness.session.manager import Session, SessionManager, new_record_id
from opendde_harness.spine.turn import Origin
from opendde_harness.tracing import semconv, trace

# A turn hands its completed work to the extraction outbox and is done: the
# answer is committed, and indexing it is a separate promise kept in the
# background. These are that background's bounds.
#
# Teardown's total budget for letting the drain finish what it is holding.
_STORE_DRAIN_BUDGET_S: float = 15.0
# What the turn will spend on skill-usage feedback, which is telemetry: it is
# best-effort in both directions, so it gets a budget rather than a turn.
_FEEDBACK_BUDGET_S: float = 1.0


_ABORTED_ACTION_REPLY = (
    "The operation was not completed, and no alternative method will be attempted. "
    "Would you like me to continue with the remaining parts of the task that do not "
    "require this operation?"
)

# NOTE: ``opendde_harness.context_engine`` is intentionally imported lazily (inside
# ``__init__`` and ``_assemble_context_messages``) to break a
# runtime import cycle: ``opendde_harness.agent.__init__`` eagerly loads AgentLoop,
# while ``opendde_harness.context_engine.factory`` imports ``ContextBuilder`` from
# ``opendde_harness.agent.context`` — a module-level top-down ``from
# opendde_harness.context_engine import ...`` here re-enters a partially-initialized
# package and raises ImportError.

if TYPE_CHECKING:
    from opendde_harness.agent.hook import CompositeHook
    from opendde_harness.agent.tools.base import Tool
    from opendde_harness.context_engine import ContextAssembler
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.memory_engine.outbox import MemoryOutbox
    from opendde_harness.providers.pool import ProviderPool
    from opendde_harness.spine.runner import Drain, Emit
    from opendde_harness.spine.runner import TurnOutcome as SpineTurnOutcome
    from opendde_harness.spine.turn import TurnRequest
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
    #: The turn's last billed call; None when no call reported any usage.
    final_call: FinalCall | None = None


@dataclass
class TurnReply:
    """What ``_process_message`` hands back. ``content`` is ``None`` for a
    silent turn (the message tool already delivered the reply)."""

    content: str | None
    media: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


# Origins whose turns skip the user-inbound hooks: a subagent result
# re-injection is not genuine user input.
_SKIP_USER_INBOUND_ORIGINS = frozenset({Origin.SUBAGENT})

# Origins whose reply skips the ``after_send`` chain: the subagent announce is
# system-originated and must not be modified on the way out.
_SKIP_AFTER_SEND_ORIGINS = frozenset({Origin.SUBAGENT})


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

    # Read by callers that report what was cut; owned by tool_batch.
    _TOOL_RESULT_MAX_CHARS = tool_batch.TOOL_RESULT_MAX_CHARS
    # Max emergency context shrinks per turn before a context overflow is fatal.
    _MAX_COMPRESS_RETRIES = 2
    # Read by callers that report what a shrink kept; owned by context_policy.
    _SHRINK_KEEP_RECENT_TOOL_RESULTS = context_policy.KEEP_RECENT_TOOL_RESULTS
    _SHRINK_KEEP_RECENT_IMAGES = context_policy.KEEP_RECENT_IMAGES
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
        provider_pool: "ProviderPool | None" = None,
    ):
        """``settings`` carries everything read from config (see
        :class:`AgentLoopSettings`); the keyword arguments are the runtime
        collaborators a call site builds itself.

        ``provider_pool`` is where a model id becomes a model id plus the
        credential for it: a per-session ``/model`` switch, a session restored
        with a stored model, and a subsystem pin all bind through it. Without
        one (a one-shot CLI turn, a test) every session runs on ``provider``.

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
        from opendde_harness.token_wise.usage_tracker import UsageTracker

        if settings is None:
            settings = AgentLoopSettings()
        self.settings = settings
        self.workspace = workspace
        from opendde_harness.config.paths import get_workspace_storage
        from opendde_harness.config.workspace_migration import warn_legacy_workspace

        self.storage = get_workspace_storage(workspace)
        warn_legacy_workspace(workspace)
        # The model a turn runs on is per session, so it cannot live in two
        # attributes on a process-wide loop. ``_default_binding`` is what a
        # session starts on; ``provider``/``model`` below read whichever
        # binding the running turn entered.
        self.sessions = session_manager or SessionManager(workspace, sessions_dir=self.storage.sessions)
        self._bindings = SessionBindings(
            ModelBinding(provider, settings.model or provider.get_default_model(), provider_name=settings.provider),
            providers=settings.providers,
            sessions=self.sessions,
            provider_pool=provider_pool,
        )
        self._bindings.on_rebind = self._forget_transport_verdicts
        # Resolved lazily on the first tool result that carries an image. Keyed
        # by model, not a single flag: the loop is a long-lived singleton and
        # takes a per-call model (strategies rewrite it, and the model chain
        # falls back), so one model's verdict must not answer for another's.
        self._vision_ok: dict[str, bool] = {}
        self.max_iterations = settings.max_iterations
        self._recovery_limits = settings.empty_recovery
        self.jina_api_key = settings.jina_api_key
        self.brave_api_key = settings.brave_api_key
        self.web_proxy = settings.web_proxy
        self.exec_config = settings.exec_config
        self.restrict_to_workspace = settings.restrict_to_workspace
        # TokenWise strategies — empty registry acts as pure pass-through.
        self.strategies = strategies if strategies is not None else StrategyRegistry([])
        # Per-session token and cost totals, read back by ``/status``. Memory
        # only: the figures belong to the running loop, and whether they also
        # go to a telemetry file is a separate decision.
        self.usage_tracker = UsageTracker(persist=False)
        self.strategies.register(self.usage_tracker)
        self._now_fn = now_fn or datetime.now

        self.backend: "MemoryBackend | None" = backend

        self.plugin_tools: "list[Tool]" = list(plugin_tools or [])

        self.context = ContextBuilder(
            workspace,
            storage=self.storage,
            skill_forge_config=settings.skill_forge,
            llm_provider=provider,
        )
        # What a committed turn is owed after it returns: a place in the memory
        # index, and the skill-usage signal beside it. Nothing here is on the turn's
        # critical path.
        self._dispatch = ExtractionDispatcher(workspace, self.context.memory, backend=backend, now_fn=now_fn)

        # Tool names to omit from the registry — applied at registration and
        # again after MCP connect so it can blacklist either group. Used by eval
        # harnesses (e.g. BCP) that need a strict tool subset.
        self._disabled_tools = set(settings.disabled_tools)

        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            model=self.model,
            jina_api_key=settings.jina_api_key,
            brave_api_key=settings.brave_api_key,
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

        # What the turn calls into, in registration order. ``self.context.skills``
        # is the :class:`LocalSkillCatalog` for the always-skills + ``# Skills``
        # render path; the SkillForgeRouter stack (assembled in
        # ``context_engine.factory``) owns retrieval.
        self.tools, self.tool_search_controller = build_tool_registry(
            workspace=workspace,
            settings=settings,
            executor=self._executor,
            subagents=self.subagents,
            skill_registry=getattr(getattr(self.context, "skills", None), "registry", None),
            plugin_tools=self.plugin_tools,
            strategies=self.strategies,
        )

        # Context engine — the single ContextAssembler. Constructed after the
        # registry so it can capture ``self.tools.get_definitions`` as a
        # deferred callable.
        # Deferred import: see the module-level note about the import cycle.
        from opendde_harness.context_engine import build_context_engine

        self.context_config = settings.context
        self.context_engine: "ContextAssembler" = build_context_engine(
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
        self._bindings.on_default_window = self.context_engine.set_context_window

        # Runtime discipline: the per-turn shadow-git checkpoint, gated by
        # (policy, interactive) — see ``checkpoint_active``. When the gate is
        # closed the loop is byte-identical to baseline.
        self.runtime_config = settings.runtime
        self.interactive = interactive
        self._checkpoint = build_checkpoint_service(settings, workspace, interactive)
        # session_key -> {"checkpoint_id", "files"} stashed when a turn is
        # interrupted (max-iter); consumed by the next turn's recovery prompt.
        self._pending_recovery: dict[str, dict] = {}

        self._running = False
        self._mcp_servers = dict(settings.mcp_servers)
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        # Tools registered per connected MCP server. Empty until the first turn
        # connects them; a server that failed stays absent.
        self._mcp_tool_counts: dict[str, int] = {}
        # ``self.subagents`` and ``self.context_engine`` were each handed
        # ``provider`` earlier in this constructor. Inside a turn they read the
        # turn's binding; the reference they hold is only the fallback for work
        # that runs outside one, and ``set_default_binding`` is what keeps that
        # fallback current. Add the call there when adding another holder.

        # ``self.context.skills`` is the :class:`LocalSkillCatalog` for the
        # always-skills + ``# Skills`` render path; the SkillForgeRouter stack
        # (assembled in ``context_engine.factory``) owns retrieval.

        self.hooks: "CompositeHook" = CompositeHook()
        if hooks is not None:
            self.hooks.extend(hooks)

    # ── Model identity and window: owned by SessionBindings ─────────────
    #
    # Properties rather than attributes: the model is per session, so there is
    # no single answer to cache on a process-wide loop. Outside a turn (startup,
    # a one-shot CLI call) each reads the configured default.

    @property
    def provider(self) -> LLMProvider:
        """The provider of the binding the running turn entered."""
        return self._bindings.provider

    @property
    def model(self) -> str:
        """The model id of the binding the running turn entered."""
        return self._bindings.model

    @property
    def context_window_tokens(self) -> int | None:
        """How much the running turn's model can hold; None is unknown."""
        return self._bindings.window

    @property
    def provider_pool(self) -> "ProviderPool | None":
        """Where a model id becomes a model id plus the credential for it."""
        return self._bindings.provider_pool

    @property
    def default_binding(self) -> ModelBinding:
        """What a session with no switch of its own runs on."""
        return self._bindings.default

    def binding_for_session(self, session_key: str) -> ModelBinding:
        """The binding this session runs on: its own switch, else the default."""
        return self._bindings.for_session(session_key)

    def session_model(self, session_key: str) -> str:
        """What to show this session's user, which is not the global default."""
        return self._bindings.model_for(session_key)

    def live_providers(self) -> list[LLMProvider]:
        """Every provider a turn can run on right now, once each."""
        return self._bindings.live_providers()

    def has_session_binding(self, session_key: str) -> bool:
        """Did this session switch, or is it just following the default?"""
        return self._bindings.has(session_key)

    def set_session_binding(self, session_key: str, binding: ModelBinding) -> None:
        """Switch one session, leaving every other session where it was."""
        self._bindings.set(session_key, binding)

    def clear_session_binding(self, session_key: str) -> None:
        """Drop a session's override so it follows the default again."""
        self._bindings.clear(session_key)

    def _forget_transport_verdicts(self) -> None:
        """Drop the capability verdicts a new provider may answer differently.

        The cache keys on a model id but is computed from the provider serving
        it, so a rebuild that keeps the id keeps the old endpoint's answer.
        Cleared wholesale rather than per binding: the loop now holds several
        providers at once, and the key does not say which answered.
        """
        self._vision_ok.clear()

    def set_default_binding(self, binding: ModelBinding) -> None:
        """Change what new sessions start on.

        Sessions that already switched keep their own binding; sessions that
        never did pick this up on their next turn. Subsystem fallbacks are
        re-pointed too, for the paths that run outside a turn and therefore have
        no binding to read, and the default window follows the model.
        """
        self._bindings.set_default(binding)
        self.subagents.set_provider(binding.provider, binding.model)
        self.context_engine.set_provider(binding.provider, binding.model)
        self.refresh_context_window()

    def resolve_window(self, model: str | None = None, binding: "ModelBinding | None" = None) -> Resolved:
        """Walk the window ladder for ``model`` (default: the loop's own).

        ``binding`` names the route that serves the model, for a caller asking
        about a session rather than about the running turn: outside a turn the
        provider tier would otherwise answer from the default binding.
        """
        return self._bindings.resolve_window(model, binding)

    def refresh_context_window(self) -> None:
        """Re-resolve the default window against the default binding's model."""
        self._bindings.refresh_default_window()

    def _note_window(self, model: str, resolved: Resolved) -> None:
        """Say once per model what its window resolved to, and size the turn by it."""
        self._bindings.note_window(model, resolved)

    def _tool_batch(self, journal: TurnJournal | None, on_tool_event) -> tool_batch.ToolBatch:
        """The executor for one assistant message's calls.

        Built per batch because the journal and the event sink are the turn's, not
        the loop's: the durable boundaries stay with whoever opened the journal.
        """
        return tool_batch.ToolBatch(
            tools=self.tools,
            context=self.context,
            route_images=self._route_result_images,
            on_tool_event=on_tool_event,
            journal=journal,
        )

    def _image_verdict(self) -> bool | None:
        """What is *known* about this model accepting a picture: True, False, unknown.

        The model layer's own answer -- the row it reports for the model it is
        about to call (``input: ["text","image"]``) -- because that is the only
        thing that has it. ``None`` is the load-bearing case: no row has been
        read yet, and a caller must not memoize the optimism that stands in for
        it.

        Being wrong optimistically is loud: the endpoint refuses the request and
        says so. Being wrong the other way is silent -- the picture never
        arrives and the model answers from the surrounding text as though it had
        seen one. That asymmetry is why unknown reads as yes.
        """
        modalities = self.provider.input_modalities()
        if not modalities:
            return None
        return "image" in modalities

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
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Decide how a tool result's pictures reach ``model``.

        Returns the text the tool result carries and the blocks to put in it.

        Two outcomes, and only one question separates them: whether the model
        can see a picture at all. It cannot -- the blocks are dropped and the
        text says so, naming the description tool when one is registered, since
        nothing follows that would carry the picture instead. It can -- the
        blocks ride along and the model layer decides where they go on the wire
        (pi's Chat Completions adapter moves them into a following user message
        itself; the Anthropic one leaves them in the tool result).

        A method rather than a branch inside the loop so it can be tested at all:
        the loop reaches this point only through a live provider and a real tool
        call, and the wrong choice here is silent -- the model answers about a
        picture it never received.
        """
        if not blocks or self._supports_vision(model):
            return model_text, blocks
        return tool_batch.image_placeholder_text(blocks, describe_tool=self._describe_tool_name()), None

    def _supports_vision(self, model: str | None = None) -> bool:
        """Cached per model: whether this model can see a picture at all.

        Asked once per turn and once per tool result that returns an image.

        Only a real verdict is cached. :meth:`_image_verdict` answers ``None``
        while the model layer has reported no row -- before the first call --
        and that is optimism rather than knowledge: caching it would freeze the
        guess for the life of this loop, which is the life of the process. An
        unanswered model is re-asked each turn, which costs a dict lookup.

        The verdict is the bound model's, which is the only one asked.
        """
        key = model or self.model
        cached = self._vision_ok.get(key)
        if cached is not None:
            return cached

        verdict = self._image_verdict()
        logger.debug("vision support for {}: {}", key, verdict)
        if verdict is None:
            return True
        self._vision_ok[key] = verdict
        return verdict

    # ── Context engine helpers ──────────────────────────────────────────

    def _context_messages_for_session(self, session: Session) -> list[dict[str, Any]]:
        """Return the candidate message view the context engine selects from.

        The full append-only log: the engine decides for itself how much of it
        fits. There is no second view to choose between any more -- the host
        markdown consolidator that used to cut the log down first is gone, and
        the engine has always been handed everything when it was absent.
        """
        return list(session.messages)

    async def _assemble_context_messages(
        self,
        *,
        session: Session,
        session_key: str,
        current_message: str,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ask the active context engine for the main-agent message window.

        ``model`` is the id the request will actually reach (the router's pick,
        when there is one). It decides whether an attachment is inlined as a
        picture and what the identity block says the model is, so defaulting it
        to ``self.model`` would let the configured model answer for a routed one.

        The turn's facts are passed rather than read from config inside
        rendering: the reply reservation, the model id, the reply language and
        whether a long-term memory backend is configured. One turn's prompt can
        then never be assembled from another turn's settings.
        """
        from opendde_harness.context_engine import TurnContext  # deferred — see module note

        session_messages = self._context_messages_for_session(session)
        request_model = model or self.model
        assembled = await self.context_engine.assemble(
            session_key,
            session_messages,
            turn=TurnContext(
                current_message=current_message,
                media=media,
                can_see_images=self._supports_vision(model),
                describe_tool=self._describe_tool_name(),
                channel=channel,
                chat_id=chat_id,
                model=request_model,
                language=self.settings.language,
                long_term_memory=self.settings.memory.backend is not None,
                reserved_output=self._reserved_output(self.context_window_tokens, request_model),
            ),
        )
        messages = assembled.messages
        self._inject_recovery_block(session_key, messages)
        return messages

    #: The checkpoint gate, resolved in the factory that builds the service.
    #: Kept under this name because it is a documented, directly tested rule.
    _checkpoint_active = staticmethod(checkpoint_active)

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
            last["content"] = [msg.text_block(block), *content]
        else:
            return  # unexpected content shape → keep pending
        self._pending_recovery.pop(session_key, None)

    # ── The background promise: owned by ExtractionDispatcher ───────────

    @property
    def outbox(self) -> "MemoryOutbox":
        """The extraction outbox, built on first use."""
        return self._dispatch.outbox

    @property
    def _outbox_task(self) -> asyncio.Task | None:
        """The drain, when one is running."""
        return self._dispatch.task

    def _queue_extraction(
        self,
        session_key: str,
        messages: list[dict],
        *,
        turn_id: str,
        generation: int,
        closing: bool = False,
    ) -> None:
        """Hand a committed turn to the outbox and let the drain pick it up.

        ``closing`` says this hand-off is a conversation ending rather than another
        turn in one. The dispatcher accumulates turns and writes them in batches,
        and a closed conversation is the one moment it knows nothing more is coming.
        """
        self._dispatch.queue(session_key, messages, turn_id=turn_id, generation=generation, closing=closing)

    def start_extraction_drain(self) -> None:
        """Take delivery of whatever a previous run left owed.

        The outbox outlives the process that filled it, so a host that has just
        brought a memory backend up asks for the drain rather than waiting for this
        session to produce a turn of its own.
        """
        self._dispatch.kick()

    async def drain_backend_stores(self, timeout: float = STORE_DRAIN_BUDGET_S) -> None:
        """Let the outbox drain before the process goes away."""
        await self._dispatch.drain(timeout)

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
            from opendde_harness.agent.tools.mcp import connect_mcp_servers, registered_tool_counts

            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            connected = await connect_mcp_servers(
                self._mcp_servers,
                self.tools,
                self._mcp_stack,
                executor=self._executor,
            )
            # Re-apply blacklist: MCP servers may register tool names that
            # also appear in ``disabled_tools`` (e.g. ``mcp_<server>_search``).
            apply_disabled_tools(self.tools, self._disabled_tools)
            # Counted after that, not before: what a server offered and what the
            # agent can actually call are different numbers whenever the
            # blacklist names one of its tools, and the count is read as the
            # second.
            self._mcp_tool_counts = registered_tool_counts(self.tools, connected)
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

    @property
    def mcp_servers(self) -> dict[str, Any]:
        """The MCP servers this loop is configured with, by name."""
        return self._mcp_servers

    @property
    def mcp_tool_counts(self) -> dict[str, int]:
        """Tools registered per connected MCP server. A configured server that
        has not connected yet, or failed, is absent."""
        return dict(self._mcp_tool_counts)

    def _set_tool_context(self, channel: str, chat_id: str, session_key: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if (tool := self.tools.get("spawn")) and hasattr(tool, "set_context"):
            tool.set_context(channel, chat_id, session_key or f"{channel}:{chat_id}")

    _strip_think = staticmethod(strip_think)

    #: The snapshot builder, kept under this name for the callers that price a
    #: call of their own the way a turn's own call is priced.
    _build_usage_snapshot = staticmethod(build_usage_snapshot)

    @trace.instrument("llm.call", extract=semconv.llm_call_stream)
    async def _llm_call_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """One streamed call on the turn's provider; see :mod:`model_call`."""
        return await model_call.llm_call_stream(
            self.provider,
            messages,
            tools,
            model,
            on_token_delta=on_token_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_tool_event=on_tool_event,
            tool_choice=tool_choice,
            on_retry=on_retry,
        )

    _emergency_shrink = staticmethod(context_policy.emergency_shrink)
    _elide_older_images = staticmethod(context_policy.elide_older_images)
    _cap_tool_result = staticmethod(tool_batch.cap_tool_result)
    _cap_tool_blocks = staticmethod(tool_batch.cap_tool_blocks)

    def _reserved_output(self, window: int | None, model: str) -> int:
        """Tokens to hold back for the reply, for the model the call goes to."""
        return context_policy.reserved_output(window, model, provider=self.provider, providers=self.settings.providers)

    def _fit_prompt(self, messages: list[dict], tools: list[dict] | None, model: str) -> list[dict]:
        """Elide older tool bodies before a call this model's window cannot hold.

        ``model`` is the one the call goes to, which a strategy may have changed
        from the turn's: the budget is that model's window and reply ceiling, not
        the turn's.
        """
        window = self.context_window_tokens if model == self.model else self.resolve_window(model).tokens
        limit = None if window is None else window - self._reserved_output(window, model)
        return context_policy.fit_prompt(messages, tools, model, provider=self.provider, limit=limit)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session_key: str = "",
        model: str | None = None,
        on_token_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_event: Callable[[str, dict], Awaitable[None]] | None = None,
        on_episode_start: Callable[[int], Awaitable[None]] | None = None,
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
        drain: Drain | None = None,
        journal: TurnJournal | None = None,
    ) -> tuple[str | None, list[dict], LoopOutcome]:
        """Run the agent iteration loop.

        ``drain``, when wired, is called at the top of each iteration to pull
        any user messages injected mid-turn (BusyPolicy.INJECT) and merge them
        as user turns before the next LLM call. ``session_key`` tags the
        TokenWise usage snapshots.

        ``journal`` is the turn's durable record. It is written forward -- the
        assistant message that asked for tools, then each result as it returns --
        so a turn cancelled or lost between two tool calls leaves what it already
        did. Without one (a test, a subagent turn) the loop records nothing and
        the caller persists at the end as before.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        used_skill_ids: list[str] = []
        effective_model = model or self.model
        # Every billed call of this turn is observed here exactly once: the
        # iterations below and the wrap-up after them. What the turn costs is not
        # what its last call cost -- a tool-using turn makes several billed calls,
        # and a single call's price used to reach the wire as the whole turn's.
        accounting = TurnAccounting(
            session_key=session_key,
            providers=self.settings.providers,
            provider=lambda: self.provider,
            resolve_window=self.resolve_window,
            note_window=self._note_window,
        )

        # Bug2 / decision B — track whether the turn was a normal exit or a
        # max-iter interruption. ``status`` is the only piece read downstream
        # (used to label the shadow-git commit and stamp the ``LoopOutcome``).
        status = "completed"

        # Context-overflow recovery: bound the number of emergency shrinks so a
        # turn that overflows even after eliding can't loop forever.
        compress_retries = 0
        # The two per-turn policies with budgets of their own: how many times the
        # same tool may fail the same way before the model is told to change
        # approach, and what an empty answer is worth recovering. Both are per turn
        # because the loop is shared across sessions.
        streak = FailureStreak(threshold=self._LOOP_BREAK_THRESHOLD, max_nudges=self._LOOP_BREAK_MAX)
        recovery = RecoveryState(limits=self._recovery_limits)

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
                        messages.append(msg.user_message(inj_text))
                        logger.info("inject: merged a mid-turn user message")

            tool_defs = self.tools.get_definitions()

            # TokenWise before-hook: strategies may rewrite messages, tools,
            # or model (e.g. ToolSearchStrategy withholds tool schemas).
            call_messages, call_tools, call_model = await self.strategies.before_llm_call(
                messages,
                tool_defs,
                effective_model,
            )
            # Fitted after the strategies, against what is actually sent: the
            # full catalog's schemas measured here elided history a request
            # without them had room for. The elision stays in the history when
            # the strategies passed it through unchanged.
            fitted = self._fit_prompt(call_messages, call_tools, call_model)
            if fitted is not call_messages:
                if call_messages is messages:
                    messages = fitted
                call_messages = fitted
            if on_token_delta is not None or on_reasoning_delta is not None:
                response = await self._llm_call_stream(
                    messages=call_messages,
                    tools=call_tools,
                    model=call_model,
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
                )
            # TokenWise after-hook: strategies observe the response for
            # usage tracking, budget enforcement, etc. Errors are swallowed.
            # The model that answered, as the backend reported it; usage and
            # the window are that model's.
            answered_model = response.model or call_model
            usage_snapshot = accounting.snapshot(response, answered_model)
            await self.strategies.after_llm_call(
                {
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                },
                usage_snapshot,
            )
            # One observation per call: the turn's sums, the wire fields the TUI
            # attaches to message.complete, and the receipt after-turn work reads.
            # ``call_messages`` is what was sent, which a strategy may have
            # rewritten; ``len(messages)`` is where the turn's own tail begins.
            if accounting.record(
                response,
                usage_snapshot,
                model=answered_model,
                sent=call_messages,
                tools=call_tools,
                history_len=len(messages),
            ):
                if on_usage is not None:
                    await on_usage(accounting.completion_tokens, accounting.reasoning_tokens, iteration)

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
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(tool_batch.tool_hint(response.tool_calls), tool_hint=True)

                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    [tc.to_pi_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                    pi_message=response.pi_message,
                )
                # Before the tools run, with every call the model asked for. A call
                # the journal never sees a result for is dropped from the next
                # request rather than answered twice -- see
                # ``drop_unanswered_tool_calls`` -- so recording the intent early
                # costs nothing and keeps the model's own reasoning with it.
                if journal is not None:
                    journal.flush(messages)

                # ``truncated`` is the one field that says the reply was cut at the
                # output limit, set by whichever route parsed the message. The batch
                # refuses every call of such a reply rather than dispatching calls
                # whose arguments may never have finished arriving.
                batch = await self._tool_batch(journal, on_tool_event).run(
                    response.tool_calls,
                    messages,
                    model=call_model or effective_model,
                    truncated=bool(response.truncated),
                )
                messages = batch.messages
                used_skill_ids.extend(sid for sid in batch.used_skill_ids if sid not in used_skill_ids)
                streak.observe(batch.failure)

                if batch.aborted:
                    # A normal tool result starts another model iteration. That is
                    # specifically unsafe here: the next plan can translate the
                    # rejected operation into an equivalent interpreter, script, or
                    # tool call. Finish the turn in runtime code and expose only the
                    # non-destructive continuation question. Streaming callers need
                    # the explicit callback because no final model response exists to
                    # generate token deltas.
                    messages = self.context.add_assistant_message(messages, _ABORTED_ACTION_REPLY)
                    final_content = _ABORTED_ACTION_REPLY
                    if on_token_delta is not None:
                        await on_token_delta(_ABORTED_ACTION_REPLY)
                    break

                # The same tool has failed the same way often enough to be a loop:
                # append a change-approach nudge to the last result so the model
                # stops repeating a dead call.
                nudge = streak.nudge() if messages and msg.is_tool_result(messages[-1]) else None
                if nudge is not None:
                    messages[-1] = msg.with_appended_text(messages[-1], "\n\n" + nudge)
                recovery.after_tool_calls = True
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
                action = recovery.decide(response, clean)
                if action is RecoveryAction.PREFILL:
                    logger.warning(
                        "empty-recovery: thinking-only prefill {}/{}",
                        recovery.prefills,
                        recovery.limits.thinking_prefill_max_retries,
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
                    recovery.after_tool_calls = False
                    continue
                if action is RecoveryAction.NUDGE:
                    logger.warning("empty-recovery: post-tool empty nudge")
                    # The (empty) assistant must sit between the tool result and
                    # the nudge — a bare tool→user sequence is a 400 on most APIs.
                    messages = self.context.add_assistant_message(messages, "(empty)")
                    messages[-1]["_recovery_synthetic"] = True
                    messages.append({**msg.user_message(POST_TOOL_NUDGE), "_recovery_synthetic": True})
                    recovery.after_tool_calls = False
                    continue
                if action is RecoveryAction.RETRY:
                    logger.warning(
                        "empty-recovery: plain empty retry {}/{}",
                        recovery.retries,
                        recovery.limits.empty_content_max_retries,
                    )
                    recovery.after_tool_calls = False
                    continue

                messages = self.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                    pi_message=response.pi_message,
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
            billed_before = accounting.billed_calls
            final_content, wrap_up_reply = await synthesize_on_exhaustion(
                messages,
                effective_model,
                stream=self._llm_call_stream,
                plain=self.provider.chat_with_retry,
                tool_defs=self.tools.get_definitions(),
                accounting=accounting,
                max_iterations=self.max_iterations,
                default_model=self.model,
                on_token_delta=on_token_delta,
                on_reasoning_delta=on_reasoning_delta,
                on_retry=on_retry,
            )
            # The wrap-up is the turn's last call, observed like the rest of them:
            # its price used to be the one a turn never mentioned, and its receipt
            # is what the after-turn compaction has to address -- left out, that
            # compaction summarised the request from before the last tool ran and
            # filed its marker behind the result, which the next turn on this
            # model then dropped.
            if accounting.billed_calls > billed_before:
                # The wrap-up is billed inside the turn but outside the
                # iteration's after-hook, which is what registers a call with the
                # session tracker. Told here the way the after-turn compaction
                # tells it, from the snapshot the turn already priced: without
                # this the turn's own total counted the one call the user
                # actually read and ``/status`` did not, so the footer and the
                # report disagreed by a whole call on every exhausted turn.
                if accounting.last_snapshot is not None:
                    self.usage_tracker.record_snapshot(accounting.last_snapshot)
                if on_usage is not None:
                    await on_usage(accounting.completion_tokens, accounting.reasoning_tokens, iteration)
            # Persist the wrap-up into history like any normal final reply.
            # Persistence downstream reads only the returned ``messages`` list,
            # so without this the synthesized answer reaches the user via the
            # stream yet never enters the conversation — the next turn (notably
            # an interrupted-turn resume) could not see what was summarized. The
            # synthesis prompt itself stays local to the helper, so only the
            # reply lands here.
            if final_content:
                # The wrap-up's own native message when it made one; the static
                # fallback is this loop's text and has none.
                messages = self.context.add_assistant_message(
                    messages, final_content, pi_message=getattr(wrap_up_reply, "pi_message", None)
                )

        # Everything recorded after the call whose numbers the outcome carries.
        # A compaction marker is filed behind all of it and stands for every
        # message before itself, so the compaction input has to include this
        # tail or the next turn replays a summary with a hole in it.
        last_call = accounting.last_call
        if last_call is not None:
            last_call.trailing = messages[last_call.length :]

        # Drop transient empty-recovery scaffolding before persistence /
        # extraction / return — refactor/OpenDDE Harness's empty-recovery marks
        # synthetic nudge/prefill messages with ``_recovery_synthetic``; strip
        # them so they never persist. (Embedded ``_trigger_local_extraction``
        # was retired when the memory service was integrated; the after-turn
        # pipeline — ``backend.store`` in ``_process_message`` — owns extraction
        # now.)
        if any(m.get("_recovery_synthetic") for m in messages):
            messages = [m for m in messages if not m.get("_recovery_synthetic")]
            if last_call is not None:
                last_call.trailing = [m for m in last_call.trailing if not m.get("_recovery_synthetic")]

        # Phase B-1 (memory service integration): embedded extraction (the
        # ``_trigger_local_extraction`` / ``SkillService.on_execution`` path)
        # was retired here in favor of the after-turn pipeline owned by the
        # caller — ``backend.store`` + ``backend.feedback`` run from
        # ``_process_message``. We surface
        # ``outcome.status`` so that pipeline can gate on completion later.

        outcome = LoopOutcome(
            status=status, used_skill_ids=used_skill_ids, usage=accounting.usage, final_call=last_call
        )
        if self._checkpoint is not None:
            # Per-turn snapshot: one commit covering all of this turn's edits,
            # for both normal and interrupted exits (matches Claude Code/Cursor
            # granularity). Best-effort — commit_turn never raises.
            label = f"turn {session_key or 'anon'} [{status}]"
            cid, changed = await self._checkpoint.commit_turn(label)
            outcome.checkpoint_id = cid
            if status == "interrupted":
                outcome.edited_files = changed

        return final_content, messages, outcome

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
        self._mcp_tool_counts = {}  # nothing is connected any more, so nothing is registered
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
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        on_retry: Callable[[int, int, str, bool], Awaitable[None]] | None = None,
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

        self._set_tool_context(channel, chat_id, session_key=key)
        # ask_user keys by the true conversation_id (== the lane / gate key),
        # which is topic-aware (req.conversation), not just channel:chat_id.
        if (ask_tool := self.tools.get("ask_user")) and isinstance(ask_tool, AskUserTool):
            ask_tool.set_context(key)

        # Skill selection is the context engine's (SkillForgeRouter inside
        # ``assemble``); the injected ids come back in the assembled metadata.
        initial_messages = await self._assemble_context_messages(
            session=session,
            session_key=key,
            current_message=content,
            media=media_paths if media_paths else None,
            channel=channel,
            chat_id=chat_id,
        )

        turn_start_idx = len(initial_messages) - 1
        # The turn's durable record opens before the first model call, with the
        # message that was accepted. Everything the turn goes on to complete is
        # appended as it completes, so what survives a cancellation is what
        # actually happened rather than nothing at all.
        journal = TurnJournal(
            self.sessions,
            session,
            sanitize=self._sanitized_record,
            start_index=turn_start_idx,
        )
        journal.open(initial_messages)
        try:
            final_content, all_msgs, outcome = await self._run_agent_loop(
                initial_messages,
                on_progress=on_progress,
                session_key=key,
                on_token_delta=on_token_delta,
                on_reasoning_delta=on_reasoning_delta,
                on_tool_event=on_tool_event,
                on_episode_start=on_episode_start,
                on_usage=on_usage,
                on_retry=on_retry,
                drain=drain,
                journal=journal,
            )
        except asyncio.CancelledError:
            # Synchronous on purpose: the task is being cancelled, so an await
            # here is the next thing to be cancelled. Everything the turn
            # finished is already on disk; this records only that it stopped.
            journal.interrupted()
            raise
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

        await self._after_turn(session, key, all_msgs, final_content, outcome, journal)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", channel, sender_id, preview)
        return TurnReply(content=final_content, usage=outcome.usage)

    async def _handle_slash_command(self, session: Session, content: str) -> str | None:
        """Reply text for a slash command, or ``None`` when ``content`` is not one."""
        cmd = content.strip().lower()
        if cmd == "/new":
            return await self._start_new_session(session)
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

    async def _start_new_session(self, session: Session) -> str:
        """``/new``: close this conversation, preserve it, and start an empty one.

        The plugin backend is the single owner of automatic durable extraction,
        so the turns this session completed are queued for it in the extraction
        outbox -- one appended line, committed before the reset. Nothing is
        waited on: a backend that is slow or wedged must not be able to hold the
        user inside a conversation they asked to leave, and a queued hand-off
        outlives the process that queued it.

        With no backend wired, nothing is extracted and nothing is recorded.
        The host used to summarize the closed tail into ``user.md`` itself,
        with one unsized LLM call that a long tool-heavy session could fail --
        and then ``/new`` refused to reset at all. That writer is gone;
        ``user.md`` is maintained by hand.

        The session's file is preserved rather than cleared in place, so what
        was said survives the reset whether or not anything indexed it. A move
        that fails is the one thing that stops the reset: clearing the session
        in place instead would destroy the conversation this is meant to keep.
        """
        key = session.key
        generation = session.generation
        self._queue_extraction(
            key,
            list(session.messages),
            turn_id=new_record_id(),
            generation=generation,
            closing=True,
        )
        try:
            await asyncio.to_thread(self.sessions.archive, key)
        except OSError:
            logger.exception("/new: could not close {}", key)
            return "Could not close this conversation, so nothing was cleared. The session log is untouched."
        # The fresh session carries the next generation. The key is unchanged, so
        # the generation is the only thing that tells a record written after the
        # reset from one written before it -- which is what an extraction job
        # replayed from the outbox needs in order to say which conversation it
        # came from.
        fresh = self.sessions.get_or_create(key)
        fresh.generation = generation + 1
        await asyncio.to_thread(self.sessions.save, fresh)
        return "New session started."

    async def _after_turn(
        self,
        session: Session,
        key: str,
        messages: list[dict],
        final_content: str,
        outcome: LoopOutcome,
        journal: TurnJournal,
    ) -> None:
        """Close the turn's journal and run the after-turn pipeline.

        ``messages`` is the loop's whole list, not the turn's tail: the journal
        holds the offset its turn starts at and slices for itself, which is the
        same list the loop handed it after every tool result.

        The order is the contract. The final assistant message and
        ``turn.completed`` land first, synchronously, before anything optional
        runs and before the caller acknowledges the turn: server-side compaction
        is a second network call on the whole prompt, and cancelled during it
        (Esc, a closed client) the finished turn -- an answer the user has
        already read -- used to be lost with it.

        What follows the commit is not the turn's business. Extraction goes to
        the outbox, which is one appended line, and the feedback signal gets a
        budget. Neither can hold the answer.
        """
        journal.close(messages, status=outcome.status)
        # The post-commit policies, in the order their names are read: compact the
        # backend's own copy of the prompt this turn filled, then hand the committed
        # records to extraction, then report the skills it used.
        await self._maybe_compact_remote(session, outcome, session_key=key)
        # The journal's own records, not the loop's list: the last request may
        # have elided tool bodies to fit the window, and what gets extracted
        # should be what was said.
        self._queue_extraction(
            key,
            journal.written,
            turn_id=journal.turn_id,
            generation=session.generation,
        )
        await self._dispatch.send_feedback(key, outcome.used_skill_ids)

    async def _maybe_compact_remote(self, session: Session, outcome: LoopOutcome, *, session_key: str = "") -> None:
        """Have a backend that compacts server-side do so once the prompt is large.

        The Codex login (like Codex itself) can hand the conversation back as
        one opaque item that stands for all of it, on the same model, at full
        fidelity -- no local summary, no trimming of what the model was working
        from. Everything here is the turn's last call as it was actually made:
        its model, its window, its tool catalogue and the messages it sent,
        which a routing strategy moves away from the session's binding.
        Triggered on that call's *reported* prompt size, not
        an estimate, once it reaches ``context.server_compact_ratio`` of that
        model's window.

        The marker is appended after this turn's own messages and stands for
        every message before it, so what the call sent plus everything recorded
        since -- ``FinalCall.trailing`` -- is what gets compacted. A summary
        built from less than that leaves a hole the next turn drops.

        The compaction is one billed call on the full prompt; a failure is
        logged and the session goes on as before.

        It is charged to the turn that caused it. This runs before
        ``message.complete`` is emitted, so the figure the status bar receives
        can include it -- and the session tracker behind ``/status`` has always
        counted it, so leaving it out of the turn's own total was the two
        disagreeing about the same call.
        """
        ratio = self.context_config.server_compact_ratio
        call = outcome.final_call
        if not ratio or call is None or not self.provider.supports_compaction:
            return
        # A call whose usage the vendor never stated, or whose model no table
        # knows, says nothing about how full the window is; compacting on it
        # bills a full-prompt request to act on a number nobody reported.
        if call.prompt_tokens <= 0 or call.window <= 0 or call.prompt_tokens < ratio * call.window:
            return
        prompt = [*call.messages, *call.trailing]
        try:
            marker, usage = await self.provider.compact(prompt, tools=call.tools, model=call.model)
        except Exception as exc:
            logger.warning("server-side compaction failed for {} ({}); history kept as is", call.model, exc)
            return
        compaction_response = LLMResponse(content="", usage=usage, model=call.model)
        # Priced at this call's own tier, like every other call of the turn, and
        # priced once: the session tracker and the turn's own total read the
        # same snapshot, which is what keeps ``/status`` and the footer saying
        # the same thing about the same call. A compaction the vendor reported
        # nothing for adds nothing: zero here would put a cost on the wire where
        # the turn had stated none, which on a plan-billed model reads as free
        # rather than as covered.
        snapshot = self._build_usage_snapshot(
            compaction_response, call.model, session_key, self.provider, self.settings.providers
        )
        self.usage_tracker.record_snapshot(snapshot)
        if usage:
            charge(outcome.usage, snapshot)
        marker.setdefault("timestamp", msg.to_ms(self._now_fn()))
        session.record(marker)
        # The turn's reported context describes the prompt this just replaced.
        # Said on the turn's own accounting, which is what reaches a status
        # line, so it can stop showing a percentage of a conversation that no
        # longer exists instead of holding the pre-compaction figure until the
        # next call happens to report.
        outcome.usage["context_compacted"] = True
        # Its own save: the turn is already on disk, and a marker that never
        # reached it would be replayed from memory and lost on the next load.
        await asyncio.to_thread(self.sessions.save, session)
        logger.info(
            "server-side compaction stored for {}: prompt was {} of {} tokens",
            call.model,
            call.prompt_tokens,
            call.window,
        )

    def _sanitized_record(self, message: dict) -> dict | None:
        """The durable form of one live message; see :func:`sanitized_record`."""
        return sanitized_record(message, now=self._now_fn)

    async def run_turn(
        self,
        req: TurnRequest,
        emit: Emit,
        drain: Drain,
        *,
        stream: bool = True,
    ) -> "SpineTurnOutcome":
        """Bind the turn to its session's model; see ``_run_turn`` for the turn.

        This is where a session's model becomes the one thing everything under
        the turn reads: the loop's own ``provider``/``model``, the context
        engine's LLM-backed segments, the consolidator, and anything the turn
        detaches (a subagent inherits the context it was created in). The
        session key is marked the same way, so a call made on the turn's behalf
        outside the loop -- reaching the provider but not the session -- is
        still accounted to it.

        Resolving once here is also what makes a mid-turn switch harmless
        without any parking. The binding is captured before the first read and
        held for the tree, so a switch that lands while this turn runs is
        simply not visible to it -- it takes effect on the session's next
        turn. Turns from other sessions run under their own binding
        concurrently, which is the point.

        The window is the bound model's; ``_note_window`` re-reads it for the
        model that actually answered, as the backend reported it.
        """
        from opendde_harness.token_wise.base import CURRENT_SESSION_KEY

        session_key = req.conversation or f"{req.source.channel}:{req.source.chat_id}"
        binding = self.binding_for_session(session_key)
        window = self.resolve_window(binding.model).tokens
        token = CURRENT_SESSION_KEY.set(session_key)
        try:
            with use_binding(binding, window=window):
                return await self._run_turn(req, emit, drain, stream=stream)
        finally:
            CURRENT_SESSION_KEY.reset(token)

    async def _run_turn(
        self,
        req: TurnRequest,
        emit: Emit,
        drain: Drain,
        *,
        stream: bool = True,
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
        from opendde_harness.agent.spine_runner import TurnEvents
        from opendde_harness.spine.events import NoticeKind, Usage
        from opendde_harness.spine.runner import TurnOutcome

        cid = req.conversation or f"{req.source.channel}:{req.source.chat_id}"
        events = TurnEvents(emit)

        # Verbatim delivery (deliver_text — see TurnRequest). Persist before
        # emit, mirroring the normal turn's save-then-reply order, so a save
        # failure never leaves the user a delivered message no turn recorded.
        # The after-turn work a normal turn does is reduced to what a no-model
        # delivery needs: backend.store indexes the report. There is no history
        # selection to run -- the next real turn selects from the log this just
        # appended to.
        if req.deliver_text is not None:
            session = self.sessions.get_or_create(cid)
            delivered = msg.assistant_message(req.deliver_text)
            before = len(session.messages)
            record_delivery(session, [delivered], 0, now=self._now_fn)
            await asyncio.to_thread(self.sessions.save, session)
            # Sliced from where this delivery started, not from the end: an empty
            # report records nothing, and a tail slice would then queue the
            # previous turn's last message for extraction a second time.
            self._queue_extraction(
                cid, list(session.messages[before:]), turn_id=new_record_id(), generation=session.generation
            )
            await events.text(req.deliver_text)
            return TurnOutcome(
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                explicit_reply=True,
                text=req.deliver_text,
            )

        # Said once, at the start of the first turn that follows the failed
        # restore: the session is not on the model its record names.
        pending = self._bindings.take_notice(cid)
        if pending is not None:
            await events.notice(NoticeKind.MODEL_FALLBACK, pending)

        # And once per session, whenever the background drain has given up on a
        # turn. The drain has no outlet of its own, so the next turn carries it.
        # ``DELIVERY_FAILED`` rather than a kind of its own: it is already on the
        # wire and already rendered, and what happened is that a completed turn
        # did not reach where it was going.
        deferred = self._dispatch.deferral_notice()
        if deferred is not None:
            await events.notice(NoticeKind.DELIVERY_FAILED, deferred)

        reply_text: str | None = None

        try:
            await self._start_executor()
            await self._connect_mcp()
            reply = await self._process_message(
                req,
                session_key=cid,
                on_progress=events.progress,
                on_token_delta=events.token if stream else None,
                on_reasoning_delta=events.reasoning if stream else None,
                on_tool_event=events.tool,
                on_episode_start=events.episode if stream else None,
                on_usage=events.usage if stream else None,
                on_retry=events.retry,
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
                await events.media(reply.media)
            if not events.streamed and reply.content:
                await events.text(reply.content)
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


#: The reasoning share of a call's output. Kept importable from here because the
#: turn's accounting is where it is read.
_reasoning_tokens_of = reasoning_tokens_of
