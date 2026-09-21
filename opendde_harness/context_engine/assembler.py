"""ContextAssembler — the one context engine.

One turn, one render:

1. every :class:`SegmentBuilder` runs concurrently; their ``text`` joins in
   ``order`` into the system prefix, their ``meta`` into the metadata;
2. the user message is built (a structural built-in — every turn has exactly
   one, so it is not a pluggable builder);
3. the budget is computed from *that* prefix, those tool definitions and that
   user message. There is no second, estimating render: ``reserved_system`` is
   the cost of the system message the request carries;
4. the history selector chooses what history fits beside them.

Tools are a side channel — passed to the LLM alongside ``messages`` and counted
in the budget, never rendered into a segment.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from opendde_harness.context_engine.base import (
    AssembledPrefix,
    AssemblyContext,
    SegmentBuilder,
    TurnContext,
)
from opendde_harness.context_engine.history_trimmer import ContextBudgetError, HistoryTrimmer
from opendde_harness.context_engine.segments import render
from opendde_harness.context_engine.types import AssembledContext, TokenBudget
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import COMPACTION_KEY
from opendde_harness.utils.helpers import estimate_prompt_tokens

if TYPE_CHECKING:
    from opendde_harness.providers.base import LLMProvider


class ContextAssembler:
    """Decides which messages reach the main agent's LLM each turn."""

    def __init__(
        self,
        builders: list[SegmentBuilder],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        history: HistoryTrimmer,
        now_fn: Callable[[], datetime] | None = None,
        *,
        include_runtime_context: bool = True,
    ) -> None:
        self._builders = sorted(builders, key=lambda b: b.order)
        self.get_tool_definitions = get_tool_definitions
        #: The one history-selector slot. Deterministic, and the only code path
        #: that decides which session messages reach the model.
        self.history = history
        self._now_fn = now_fn or datetime.now
        self._include_runtime_context = include_runtime_context
        # Windows already reported as holding no history, so the warning is
        # logged once per window rather than once per turn.
        self._budget_warned_for: int | None = None

    @property
    def name(self) -> str:
        return "context_assembler"

    def set_provider(self, provider: "LLMProvider", model: str) -> None:
        """Adopt the provider a live ``/model`` switch just built."""
        self.history.set_provider(provider, model)
        # Duck-typed on purpose: only the builders that actually call an LLM
        # implement it, and putting it on the SegmentBuilder protocol would
        # force an empty override onto every purely textual builder.
        for builder in self._builders:
            setter = getattr(builder, "set_provider", None)
            if callable(setter):
                setter(provider, model)

    def set_context_window(self, tokens: int | None) -> None:
        """Follow a ``/model`` switch: re-budget against the new window.

        ``None`` is an unknown window, which the selector treats as "do not
        trim".
        """
        self.history.context_window_tokens = tokens

    async def assemble(
        self,
        session_key: str,
        session_messages: list[dict[str, Any]],
        *,
        turn: TurnContext,
    ) -> AssembledContext:
        """Build the exact message list passed to the main agent's LLM.

        Raises :class:`~opendde_harness.context_engine.history_trimmer.ContextBudgetError`
        when the turn cannot be made to fit, rather than sending a request the
        window is known to refuse.
        """
        ctx = AssemblyContext(
            session_key=session_key,
            current_message=turn.current_message,
            media=turn.media,
            can_see_images=turn.can_see_images,
            describe_tool=turn.describe_tool,
            channel=turn.channel,
            chat_id=turn.chat_id,
            session_messages=session_messages,
            model=turn.model,
            language=turn.language,
            long_term_memory=turn.long_term_memory,
        )

        # ── The one render: independent builders, concurrent ─────────
        segments = await asyncio.gather(*[b.build(ctx) for b in self._builders])
        meta: dict[str, Any] = {}
        parts: list[tuple[int, str]] = []
        for builder, seg in zip(self._builders, segments):
            if seg is None:
                continue
            meta |= seg.meta
            if seg.text:
                parts.append((builder.order, seg.text))
        parts.sort(key=lambda t: t[0])
        system_prefix = "\n\n---\n\n".join(text for _, text in parts)

        prefix = AssembledPrefix(
            system_prefix=system_prefix,
            user_message=self._build_user(ctx),
            tool_defs=self.get_tool_definitions(),
        )
        budget = self._budget(prefix, turn.reserved_output)

        # Off the event loop: selection prices every candidate message, and the
        # loop is what the TUI's keystrokes and stream updates wait on.
        messages, outcome = await asyncio.to_thread(
            self.history.select,
            session_messages=session_messages,
            reserved_output=turn.reserved_output,
            build_messages=lambda history: self._build_messages(prefix, history),
        )
        for warning in outcome.warnings:
            logger.info("context: {}", warning)
        return AssembledContext(
            messages=messages,
            include_indices=outcome.included_ids,
            metadata=meta
            | {
                "engine": self.name,
                "budget": {
                    "context_length": budget.context_length,
                    "reserved_output": budget.reserved_output,
                    "reserved_tools": budget.reserved_tools,
                    "reserved_system": budget.reserved_system,
                    "available_history": budget.available_history,
                },
                "history": {
                    "estimated_tokens": outcome.estimated_tokens,
                    "max_prompt_tokens": outcome.max_prompt_tokens,
                    "included": len(outcome.included_ids),
                    "dropped": len(outcome.dropped_ids),
                    "excerpted": len(outcome.excerpted_ids),
                    "source": outcome.source,
                },
            },
        )

    # ------------------------------------------------------------------
    # Prompt composition
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(prefix: AssembledPrefix, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The full message list for a candidate history.

        The one composition rule, used for the request and for every estimate
        of it — so the number the selection is made against is the number the
        request costs.
        """
        return [
            msg.system_message(prefix.system_prefix),
            *_coalesce_assistant(history),
            prefix.user_message,
        ]

    def _build_user(self, ctx: AssemblyContext) -> dict[str, Any]:
        """The single structural user message: runtime context + content."""
        user_content = render.build_user_content(
            ctx.current_message,
            ctx.media,
            can_see_images=ctx.can_see_images,
            describe_tool=ctx.describe_tool,
        )
        if not self._include_runtime_context:
            return msg.user_message(user_content)
        runtime_ctx = render.build_runtime_context(self._now_fn, ctx.channel, ctx.chat_id)
        if isinstance(user_content, str):
            merged: Any = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [msg.text_block(runtime_ctx), *user_content]
        return msg.user_message(merged)

    async def validate_continuation(self, messages: list[dict[str, Any]], *, reserved_output: int) -> None:
        """Budget an in-flight tool/repair exchange without discarding its evidence.

        Unlike completed conversation history, loaded instructions and intermediate
        tool results may be prerequisites for the pending structured output. Treat
        the complete exchange as mandatory rather than silently excerpting it.
        """
        window = self.history.context_window_tokens
        if window is not None and reserved_output >= window:
            raise ContextBudgetError("Output reservation leaves no room for the in-flight context")
        _, outcome = await asyncio.to_thread(
            self.history.select,
            session_messages=[],
            reserved_output=reserved_output,
            build_messages=lambda _: messages,
        )
        if not outcome.ok:
            raise ContextBudgetError(
                f"In-flight context needs {outcome.estimated_tokens} tokens; "
                f"only {outcome.max_prompt_tokens} are available after reserving output"
            )

    def _budget(self, prefix: AssembledPrefix, reserved_output: int) -> TokenBudget:
        """This turn's budget, measured on the prompt this turn actually sends."""
        window = self.history.context_window_tokens
        tool_tokens = estimate_prompt_tokens([], prefix.tool_defs)
        system_tokens = estimate_prompt_tokens([msg.system_message(prefix.system_prefix)])
        # Unknown stays unknown. Subtracting from a stand-in produced a budget
        # that nothing had measured, and the selector acted on it. A window the
        # reservation alone fills -- a model whose window is no larger than its
        # output ceiling, which a declared row can now describe -- leaves
        # nothing for history; that is reported as unknown too, not as a budget
        # of zero.
        available_history = None if window is None else window - reserved_output - tool_tokens - system_tokens
        if available_history is not None and available_history <= 0:
            if self._budget_warned_for != window:
                self._budget_warned_for = window
                logger.warning(
                    "context window for {} ({} tokens) holds no history beside a {}-token reply, "
                    "{} tokens of tools and {} of system prompt",
                    self.history.model,
                    window,
                    reserved_output,
                    tool_tokens,
                    system_tokens,
                )
            available_history = None
        return TokenBudget(
            context_length=window,
            reserved_output=reserved_output,
            reserved_tools=tool_tokens,
            reserved_system=system_tokens,
            available_history=available_history,
        )


def _mergeable(message: dict[str, Any]) -> bool:
    """Whether this assistant message is plain text and nothing else.

    An assistant carrying ``toolCall`` blocks is always followed by its results,
    never by another assistant, and merging it would break tool-call adjacency.
    One carrying thinking is not merged either: the reasoning belongs to the
    turn that produced it, and a merge would drop one side's blocks -- including
    the signatures a thinking model is owed back.

    A compaction marker is an assistant message with text and no tool calls, so
    it matched on both sides before: merged into the answer before it, its
    ``compaction`` key went with the rest of the merged-in message and the
    boundary was gone a step before the request was built. It is a boundary, not
    text -- neither side of a merge.
    """
    return (
        msg.is_assistant(message)
        and COMPACTION_KEY not in message
        and all(block.get("type") == msg.TEXT for block in msg.blocks_of(message))
    )


def _coalesce_assistant(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent plain-assistant messages into one.

    A verbatim ``deliver_text`` turn records the bot's answer as an assistant
    message; it can land right after a prior assistant reply (the "on it" ack),
    leaving two adjacent assistant turns. Some providers reject consecutive
    same-role messages and nothing else in the pipeline merges them.
    """
    out: list[dict[str, Any]] = []
    for message in history:
        prev = out[-1] if out else None
        if prev is not None and _mergeable(prev) and _mergeable(message):
            out[-1] = msg.with_text(prev, f"{msg.text_of(prev)}\n\n{msg.text_of(message)}")
            continue
        out.append(message)
    return out


__all__ = ["ContextAssembler"]
