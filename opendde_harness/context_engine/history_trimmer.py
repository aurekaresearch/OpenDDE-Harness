"""History trimming — the Curator's contribution to ``*history``.

Extracted from :class:`CuratorAssembler` so the Curator and the unified
context engine share one implementation of the operations that decide
which session messages reach the model:

- **adjacency closure** (:meth:`canonical_ids`) — if a tool call is
  selected its result messages come along, and vice versa, so the
  provider never sees a dangling tool call / orphan result;
- **clean extraction** (:meth:`history_from_ids`) — project the selected
  session messages down to the provider-safe key set;
- **structural validation** (:meth:`structural_errors`) — verify every
  tool result has a parent assistant ``tool_calls`` and every call has a
  result;
- **budget trimming** (:meth:`trim`) — build the prompt, estimate its
  token cost, and drop the lowest-priority non-protected messages until
  it fits.

This is the *only* code path that selects ``*history``. Segment 6
(``# Curator Working State``) is rendered by :class:`ContextBuilder`
from the plan's working-state text — it is not this module's concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from opendde_harness.providers.base import LLMProvider
from opendde_harness.utils.helpers import estimate_prompt_tokens_chain

# Provider-safe message keys. Anything else on a session message
# (timestamps, internal ids, manifest annotations) is dropped before
# the dict reaches the LLM. reasoning_content / thinking_blocks must survive
# so multi-turn reasoning contracts (e.g. DeepSeek thinking mode) hold; the
# provider gate strips thinking_blocks for non-Anthropic targets downstream.
_ALLOWED_KEYS = {
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "name",
    "reasoning_content",
    "thinking_blocks",
}


@dataclass
class TrimOutcome:
    """Result of a :meth:`HistoryTrimmer.trim` call."""

    history: list[dict[str, Any]]
    included_ids: list[int]
    estimated_tokens: int
    max_prompt_tokens: int | None  # None: the window is unknown, nothing was enforced
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.max_prompt_tokens is None or self.estimated_tokens <= self.max_prompt_tokens

    @property
    def over_by(self) -> int:
        if self.max_prompt_tokens is None:
            return 0
        return max(0, self.estimated_tokens - self.max_prompt_tokens)


class HistoryTrimmer:
    """Shapes and budget-trims the session history into ``*history``."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        context_window_tokens: int | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.get_tool_definitions = get_tool_definitions
        self.context_window_tokens = context_window_tokens

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        """Adopt the provider a live ``/model`` switch just built, so token
        estimates keep matching the model actually being called."""
        self.provider = provider
        self.model = model

    # ------------------------------------------------------------------
    # Pure history-shaping helpers (no token estimation / no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_ids(messages: list[dict[str, Any]], ids: list[int]) -> list[int]:
        """Close ``ids`` over tool-call / tool-result adjacency.

        Returns the selected indices in order, trimmed so the sequence
        begins at a ``user`` message (so history never starts mid
        tool-exchange). Returns ``[]`` if no user message survives.
        """
        selected = {mid for mid in ids if isinstance(mid, int) and 0 <= mid < len(messages)}
        tool_parent_by_call, tool_result_by_call = HistoryTrimmer._tool_pairs(messages)

        changed = True
        while changed:
            changed = False
            for call_id, parent_idx in tool_parent_by_call.items():
                result_ids = tool_result_by_call.get(call_id, [])
                if parent_idx in selected:
                    for rid in result_ids:
                        if rid not in selected:
                            selected.add(rid)
                            changed = True
                if any(rid in selected for rid in result_ids) and parent_idx not in selected:
                    selected.add(parent_idx)
                    changed = True

        ordered = sorted(selected)
        for pos, mid in enumerate(ordered):
            if messages[mid].get("role") == "user":
                return ordered[pos:]
        return []

    @staticmethod
    def _tool_pairs(messages: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[int]]]:
        """Which assistant turn opened each tool call, and which results answer it."""
        parent_by_call: dict[str, int] = {}
        result_by_call: dict[str, list[int]] = {}
        for idx, message in enumerate(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for tc in message.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        parent_by_call[str(tc["id"])] = idx
            if message.get("role") == "tool" and message.get("tool_call_id"):
                result_by_call.setdefault(str(message["tool_call_id"]), []).append(idx)
        return parent_by_call, result_by_call

    @staticmethod
    def _tool_exchange(messages: list[dict[str, Any]], mid: int) -> set[int]:
        """Every message that cannot be kept without ``mid``, including itself.

        A tool exchange is indivisible on the wire: an assistant turn carrying
        tool_calls and the results answering them stand or fall together. Both
        providers refuse half of one -- "tool_use ids were found without
        tool_result blocks" and "messages with role 'tool' must be a response to
        a preceding message with 'tool_calls'".
        """
        parent_by_call, result_by_call = HistoryTrimmer._tool_pairs(messages)
        message = messages[mid]
        parent: int | None = None
        if message.get("role") == "assistant" and message.get("tool_calls"):
            parent = mid
        elif message.get("role") == "tool" and message.get("tool_call_id"):
            parent = parent_by_call.get(str(message["tool_call_id"]))
        if parent is None:
            return {mid}
        group = {mid, parent}
        for call_id, idx in parent_by_call.items():
            if idx == parent:
                group.update(result_by_call.get(call_id, []))
        return group

    @staticmethod
    def history_from_ids(messages: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
        """Project the selected messages down to provider-safe keys."""
        history: list[dict[str, Any]] = []
        for mid in ids:
            clean = {k: v for k, v in messages[mid].items() if k in _ALLOWED_KEYS}
            if clean.get("role"):
                history.append(clean)
        return history

    @staticmethod
    def structural_errors(messages: list[dict[str, Any]]) -> list[str]:
        """Tool-call closure validation over a built message list."""
        errors: list[str] = []
        open_calls: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        open_calls.add(str(tc["id"]))
            if msg.get("role") == "tool":
                call_id = str(msg.get("tool_call_id", ""))
                if call_id not in open_calls:
                    errors.append(f"tool result {call_id} has no parent assistant tool_call")
                else:
                    open_calls.remove(call_id)
        if open_calls:
            errors.append(f"assistant tool_calls missing results: {sorted(open_calls)}")
        return errors

    @staticmethod
    def _first_droppable(messages: list[dict[str, Any]], ids: list[int], protected_ids: set[int]) -> int | None:
        """Position of the first id whose whole exchange is unprotected, or None.

        Protected messages are never dropped, even to fit. Dropping them used
        to be the last resort, which in a budget smaller than the protected
        head shipped an empty history while reporting the prompt as fitting;
        a head a few hundred tokens over the reservation is a request the
        provider almost always accepts, since the reservation is the whole
        output ceiling, and one it refuses is handled by the loop's overflow
        recovery -- with the conversation still in it. Judged on the exchange
        rather than the message: a drop removes the whole exchange, so a tool
        result answering a protected assistant turn is protected with it.
        """
        for pos, mid in enumerate(ids):
            if mid in protected_ids:
                continue
            if HistoryTrimmer._tool_exchange(messages, mid) & protected_ids:
                continue
            return pos
        return None

    # ------------------------------------------------------------------
    # Budget-driven trimming
    # ------------------------------------------------------------------

    def trim(
        self,
        *,
        session_messages: list[dict[str, Any]],
        ids: list[int],
        protected_ids: set[int],
        reserved_output: int,
        build_messages: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], TrimOutcome]:
        """Close ``ids``, build, and drop until under budget.

        ``build_messages`` maps a history list to the full message list
        (system + history + user) — the caller owns prompt composition
        (segments, working state, router skills), the trimmer only owns
        history selection. Returns the final ``messages`` and a
        :class:`TrimOutcome`.
        """
        canon = self.canonical_ids(session_messages, ids)
        history = self.history_from_ids(session_messages, canon)
        messages = build_messages(history)

        estimated, source = estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            messages,
            self.get_tool_definitions(),
        )
        # No window, no trimming: dropping history to fit a number nobody
        # measured is worse than sending it all and letting the provider's
        # overflow error say so, which the loop handles by shrinking in place.
        max_prompt = (
            None if self.context_window_tokens is None else max(1, self.context_window_tokens - reserved_output)
        )
        warnings: list[str] = []
        trimmed_ids = list(canon)
        # Each pass drops at least one id, so the loop is bounded by the
        # history's length; the bound is written down so a future change to
        # the closure cannot make it spin. The re-estimate inside is cheap:
        # every message's count comes from the content cache, so a pass costs
        # a hash per message rather than an encode of the whole prompt.
        for _ in range(len(canon)):
            if max_prompt is None or estimated <= max_prompt or not trimmed_ids:
                break
            drop_idx = self._first_droppable(session_messages, trimmed_ids, protected_ids)
            if drop_idx is None:
                break
            dropped = trimmed_ids[drop_idx]
            # Drop the whole exchange and re-close: taking one message at a time
            # shipped an assistant tool_calls turn whose results had been
            # dropped, which the provider rejects outright.
            group = self._tool_exchange(session_messages, dropped)
            trimmed_ids = self.canonical_ids(session_messages, [mid for mid in trimmed_ids if mid not in group])
            warnings.append(f"dropped message {dropped} to fit budget")
            history = self.history_from_ids(session_messages, trimmed_ids)
            messages = build_messages(history)
            estimated, source = estimate_prompt_tokens_chain(
                self.provider,
                self.model,
                messages,
                self.get_tool_definitions(),
            )

        return messages, TrimOutcome(
            history=history,
            included_ids=trimmed_ids,
            estimated_tokens=estimated,
            max_prompt_tokens=max_prompt,
            source=source,
            warnings=warnings,
        )


__all__ = ["HistoryTrimmer", "TrimOutcome"]
