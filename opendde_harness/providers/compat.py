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
    * ``qwen`` -- DashScope's ``enable_thinking`` body switch;
    * ``openrouter`` -- OpenRouter's nested ``reasoning: {effort}``;
    * ``effort`` -- OpenAI's top-level ``reasoning_effort``;
    * ``none`` -- the model decides for itself and takes no switch.

    ``supports_reasoning_effort`` is false for the vendors that reject the
    field outright (xAI, Moonshot, DashScope, MiniMax) rather than ignore it.
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
_FAMILIES: tuple[tuple[tuple[str, ...], ChatCompat], ...] = (
    (("deepseek",), ChatCompat("deepseek", "deepseek", requires_reasoning_content=True)),
    (("glm", "zhipu", "zai/", "bigmodel"), ChatCompat("zai", "zai")),
    (("qwen", "dashscope"), ChatCompat("qwen", "qwen", supports_reasoning_effort=False)),
    (("kimi", "moonshot"), ChatCompat("moonshot", "none", supports_reasoning_effort=False)),
    (("grok", "x.ai", "xai/"), ChatCompat("xai", "none", supports_reasoning_effort=False)),
    (("minimax",), ChatCompat("minimax", "none", supports_reasoning_effort=False)),
    (("openrouter/",), ChatCompat("openrouter", "openrouter")),
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


def apply_thinking(
    kwargs: dict[str, Any],
    compat: ChatCompat,
    *,
    reasoning_effort: str | None,
    model: str,
) -> None:
    """Put the family's thinking switch into a Chat Completions request.

    An effort switches thinking on and names the level where the family takes
    one. With no effort chosen the vendor's own default stands -- a hybrid
    model is left as its vendor ships it -- except for an id that names a
    thinking variant outright, which is switched on with no level: asked for
    by name, it must not answer as its non-thinking sibling.
    """
    effort = reasoning_effort if reasoning_effort and can_reason(model) else None
    enable = bool(effort) or names_thinking(model)

    if compat.thinking == "deepseek":
        if enable:
            _merge_body(kwargs, {"thinking": {"type": "enabled"}})
        if effort and compat.supports_reasoning_effort:
            kwargs["reasoning_effort"] = effort
    elif compat.thinking == "zai":
        if enable:
            _merge_body(kwargs, {"thinking": {"type": "enabled", "clear_thinking": False}})
        if effort and compat.supports_reasoning_effort:
            kwargs["reasoning_effort"] = effort
    elif compat.thinking == "qwen":
        if enable:
            _merge_body(kwargs, {"enable_thinking": True})
    elif compat.thinking == "openrouter":
        if effort:
            _merge_body(kwargs, {"reasoning": {"effort": effort}})
    elif compat.thinking == "effort":
        if effort:
            kwargs["reasoning_effort"] = effort


def _merge_body(kwargs: dict[str, Any], fields: dict[str, Any]) -> None:
    """Add body fields under ``extra_body`` without displacing a user's own.

    A ``model_overrides`` entry may already have placed an ``extra_body`` in
    the request; its keys win on collision, since that channel is documented
    as the way to reverse a shipped default.
    """
    existing = kwargs.get("extra_body")
    kwargs["extra_body"] = {**fields, **existing} if isinstance(existing, dict) else dict(fields)
