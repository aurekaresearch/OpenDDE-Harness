"""How each model family wants thinking switched on over the Chat Completions wire.

One declarative table, keyed on the vendor a model id names rather than on the
provider section it is served through: a relay fronts every family on one
OpenAI-compatible endpoint, and the request shape that turns DeepSeek's or
Z.ai's reasoning on is the vendor's, not the relay's. This used to be three
scattered checks (a DashScope switch, a DeepSeek reasoning-key rule, an
effort field sent to everyone else) that a relay-routed id fell through, so a
thinking model behind a relay answered without ever thinking, and the chain
of thought the next turn should have carried back was never produced.

The table is a port of pi's ``detectCompat`` (``packages/ai/src/api/
openai-completions.ts``), which is the most complete survey of these shapes
in the open, kept to the families this harness carries specs for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What a model id says about itself: these spellings name a thinking variant
#: outright, so thinking is switched on for them even with no effort chosen.
_THINKING_MARKERS = ("thinking", "reasoner", "-r1", "/r1", "r1-", "-think")


@dataclass(frozen=True)
class ChatCompat:
    """One family's request shape on the Chat Completions wire.

    ``thinking`` names how the family switches reasoning on:

    * ``deepseek`` / ``zai`` -- a ``thinking: {type: enabled}`` body field;
    * ``qwen`` -- DashScope's ``enable_thinking`` switch plus a
      ``thinking_budget`` sized from the level;
    * ``openrouter`` -- OpenRouter's nested ``reasoning: {effort}``, which it
      applies to every vendor it fronts, budget-only ones included;
    * ``effort`` -- OpenAI's top-level ``reasoning_effort``;
    * ``none`` -- the model decides for itself and takes no switch.

    ``supports_reasoning_effort`` is false for the vendors that reject the
    field outright (xAI, Z.ai, Moonshot, DashScope, MiniMax) rather than
    ignore it; those take a depth only where their own switch carries one.
    ``requires_reasoning_content`` marks a family whose thinking mode refuses
    an assistant turn that carries no ``reasoning_content`` key at all.
    """

    family: str
    thinking: str
    supports_reasoning_effort: bool = True
    requires_reasoning_content: bool = False


_DEFAULT = ChatCompat("openai", "effort")

#: (family, id substrings that name it, shape). First match wins; the
#: substrings are matched against the model id with any provider prefix and
#: against the endpoint's host, the way pi matches provider and baseUrl.
#: OpenRouter is first: it normalises every vendor's switch behind its own
#: ``reasoning`` object (an effort, or a token budget it derives from one for
#: Anthropic, Gemini and Qwen), so a vendor's own field must not be sent to
#: it even when the id names that vendor.
_FAMILIES: tuple[tuple[tuple[str, ...], ChatCompat], ...] = (
    (("openrouter/", "openrouter.ai"), ChatCompat("openrouter", "openrouter")),
    (("deepseek",), ChatCompat("deepseek", "deepseek", requires_reasoning_content=True)),
    (("glm", "zhipu", "zai/", "bigmodel"), ChatCompat("zai", "zai", supports_reasoning_effort=False)),
    (("qwen", "dashscope"), ChatCompat("qwen", "qwen", supports_reasoning_effort=False)),
    (("kimi", "moonshot"), ChatCompat("moonshot", "none", supports_reasoning_effort=False)),
    (("grok", "x.ai", "xai/"), ChatCompat("xai", "none", supports_reasoning_effort=False)),
    (("minimax",), ChatCompat("minimax", "none", supports_reasoning_effort=False)),
)


def detect_compat(model: str, *, resolved_model: str = "", api_base: str | None = None) -> ChatCompat:
    """The family shape for this model, from its id and where it is sent."""
    haystack = " ".join(part.lower() for part in (model, resolved_model, api_base or "") if part)
    for needles, compat in _FAMILIES:
        if any(needle in haystack for needle in needles):
            return compat
    return _DEFAULT


def names_thinking(model: str) -> bool:
    """Whether the id itself says this is a thinking variant."""
    tail = model.rsplit("/", 1)[-1].lower()
    return any(marker in tail for marker in _THINKING_MARKERS)


def can_reason(model: str) -> bool:
    """Whether this model reasons at all, so an effort is worth sending.

    The id first, then the bundled catalogue's ``reasoning`` flag; a model no
    table lists is assumed capable, since sending an effort a model ignores
    costs nothing while withholding one from a model that needed it is the
    defect this module exists to remove.
    """
    if names_thinking(model):
        return True
    from opendde_harness.providers.catalog import model_reasoning

    known = model_reasoning(model)
    return True if known is None else known


#: pi's DEFAULT_THINKING_BUDGETS: tokens of thinking per level for a family
#: that takes a budget rather than a level; ``xhigh`` and ``max`` clamp to
#: ``high`` (pi's clampReasoning).
_THINKING_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 16384, "max": 16384}
#: Tokens always left for the answer when thinking shares the response ceiling.
_MIN_ANSWER_TOKENS = 1024
#: The three-step ladder DeepSeek's ``reasoning_effort`` takes; the ends of
#: the vocabulary fold onto it.
_CHAT_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
#: The wire word for ``off`` on OpenAI's ladder (pi: ``thinkingLevelMap.off ?? "none"``).
OFF_EFFORT = "none"


def _takes_xhigh(model: str) -> bool:
    """pi's supportsOpenAiXhigh: the OpenAI generations that accept ``xhigh``."""
    tail = model.lower()
    return any(mark in tail for mark in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6-astra"))


def _takes_max(model: str) -> bool:
    """pi's supportsOpenAiMax: the OpenAI generations that accept ``max``."""
    tail = model.lower()
    return "gpt-5.6" in tail or "gpt-6-astra" in tail


def openai_effort(model: str, effort: str, *, codex: bool = False) -> str:
    """The level an OpenAI-style ``reasoning_effort`` takes for this model.

    pi's thinkingLevelMap, reduced to what the id says: ``max`` only where the
    generation accepts it, ``xhigh`` from gpt-5.2, both clamped down to the
    highest level the model takes; the Codex login maps ``minimal`` to ``low``
    on the same generations; ``off`` is spelled ``none`` on the wire.
    """
    if effort == "off":
        return OFF_EFFORT
    if effort == "max" and not _takes_max(model):
        effort = "xhigh"
    if effort == "xhigh" and not _takes_xhigh(model):
        effort = "high"
    if codex and effort == "minimal" and _takes_xhigh(model):
        effort = "low"
    return effort


def thinking_budget(effort: str, max_tokens: int | None) -> int | None:
    """Tokens of thinking for this level, kept under the response ceiling.

    Reasoning and the answer share ``max_tokens`` on the families that take a
    budget, so an uncapped level could spend the whole response on thinking
    and leave no answer and no tool call.
    """
    budget = _THINKING_BUDGETS.get(effort)
    if budget is None:
        return None
    if max_tokens:
        budget = min(budget, max(0, max_tokens - _MIN_ANSWER_TOKENS))
    return budget or None


def apply_thinking(
    kwargs: dict[str, Any],
    compat: ChatCompat,
    *,
    reasoning_effort: str | None,
    model: str,
    max_tokens: int | None = None,
) -> None:
    """Put the family's thinking switch into a Chat Completions request.

    The level vocabulary is pi's: ``off | minimal | low | medium | high |
    xhigh | max``. ``off`` switches thinking off explicitly; a level switches
    it on and names the depth where the family takes one (an effort for
    OpenAI and DeepSeek, a token budget for DashScope, on/off only for Z.ai).
    No level at all leaves the vendor's default -- a hybrid model stays as its
    vendor ships it -- except for an id that names a thinking variant
    outright, which is switched on: asked for by name, it must not answer as
    its non-thinking sibling.
    """
    off = reasoning_effort == "off"
    effort = reasoning_effort if reasoning_effort and not off and can_reason(model) else None
    enable = bool(effort) or (names_thinking(model) and not off)
    chat_effort = _CHAT_EFFORT.get(effort or "", effort)

    if compat.thinking == "deepseek":
        if enable:
            _merge_body(kwargs, {"thinking": {"type": "enabled"}})
        elif off:
            _merge_body(kwargs, {"thinking": {"type": "disabled"}})
        if effort and compat.supports_reasoning_effort:
            kwargs["reasoning_effort"] = chat_effort
    elif compat.thinking == "zai":
        if enable:
            _merge_body(kwargs, {"thinking": {"type": "enabled", "clear_thinking": False}})
        elif off:
            _merge_body(kwargs, {"thinking": {"type": "disabled"}})
    elif compat.thinking == "qwen":
        if enable:
            body: dict[str, Any] = {"enable_thinking": True}
            budget = thinking_budget(effort, max_tokens) if effort else None
            if budget:
                body["thinking_budget"] = budget
            _merge_body(kwargs, body)
        elif off:
            _merge_body(kwargs, {"enable_thinking": False})
    elif compat.thinking == "openrouter":
        # OpenRouter takes the whole vocabulary itself (none .. max) and
        # turns it into a budget for a model that only takes one.
        if effort:
            _merge_body(kwargs, {"reasoning": {"effort": effort}})
        elif off:
            _merge_body(kwargs, {"reasoning": {"effort": OFF_EFFORT}})
    elif compat.thinking == "effort":
        if effort:
            kwargs["reasoning_effort"] = openai_effort(model, effort)
        elif off:
            kwargs["reasoning_effort"] = OFF_EFFORT


def _merge_body(kwargs: dict[str, Any], fields: dict[str, Any]) -> None:
    """Add body fields under ``extra_body`` without displacing a user's own.

    A ``model_overrides`` entry may already have placed an ``extra_body`` in
    the request; its keys win on collision, since that channel is documented
    as the way to reverse a shipped default.
    """
    existing = kwargs.get("extra_body")
    kwargs["extra_body"] = {**fields, **existing} if isinstance(existing, dict) else dict(fields)
