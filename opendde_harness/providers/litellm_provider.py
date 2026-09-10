"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import hashlib
import json
import os
import re
import secrets
import string
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

import json_repair
from loguru import logger

from opendde_harness.providers import prompt_cache
from opendde_harness.providers.base import (
    EndpointNotFoundError,
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    RunMeta,
    StreamDelta,
    ToolCallRequest,
    WireMismatchError,
    format_llm_error,
)
from opendde_harness.providers.catalog import overlay_for
from opendde_harness.providers.compat import apply_thinking, detect_compat, openai_effort
from opendde_harness.providers.litellm_setup import import_litellm
from opendde_harness.providers.prompt_cache import CACHE_CONTROL
from opendde_harness.providers.reasoning import split_orphan_think
from opendde_harness.providers.registry import (
    canonical_provider_name,
    find_by_keywords,
    find_by_model,
    find_by_name,
    find_gateway,
)
from opendde_harness.providers.responses_api import (
    PROVIDER_OPENAI,
    REASONING_BLOCK_TYPE,
    is_local_web_search_tool,
    join_tool_call_id,
    message_text,
    reasoning_block,
    rejects_hosted_web_search,
    responses_error_text,
    responses_incomplete_reason,
    responses_input,
    responses_tool_choice,
    responses_tools,
    responses_usage,
    selects_local_web_search,
    url_citations,
    value_of,
    web_search_action,
    web_search_preview,
)
from opendde_harness.providers.wire import wire_model

if TYPE_CHECKING:
    from opendde_harness.config.schema import ModelOverlay

litellm = import_litellm()
acompletion = litellm.acompletion
aresponses = litellm.aresponses

# LiteLLM's async logging worker (LoggingWorker) binds its queue to a single
# event loop. OpenDDE Harness runs each turn under a fresh loop (asyncio.run per call), so
# on the next turn the queue is reset and any pending ``Logging.async_*_handler``
# coroutine is dropped without being awaited. Python then prints a
# ``coroutine ... was never awaited`` RuntimeWarning that bleeds into the Ink TUI
# render. The dropped callback is LiteLLM's own success/failure logging, which
# OpenDDE Harness does not rely on. Scope the filter to LiteLLM's ``Logging`` handlers only
# -- a bare ``coroutine '.*'`` pattern would also hide genuine never-awaited bugs
# in OpenDDE Harness's own coroutines.
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'Logging\.async_.*' was never awaited",
    category=RuntimeWarning,
)

# Standard chat-completion message keys.
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"})
_ANTHROPIC_EXTRA_KEYS = frozenset({"thinking_blocks"})
# The block types Anthropic signed and accepts back. A session that switched
# from Codex carries its ``reasoning`` blocks in the same list, and Anthropic
# refuses a request with one in it; the strip is keyed on the target, so it
# has to filter by what the block is, not only on who is asked.
_ANTHROPIC_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
_ALNUM = string.ascii_letters + string.digits

# LiteLLM defaults to X-Title="liteLLM" / HTTP-Referer="https://litellm.ai" for OpenRouter
# requests, which would credit traffic to liteLLM instead of OpenDDE Harness on openrouter.ai/apps.
# Explicit headers here override those defaults; user-supplied extra_headers win over these.
_OPENROUTER_ATTRIBUTION: dict[str, str] = {
    "HTTP-Referer": "https://github.com/aurekaresearch/OpenDDE-Harness-beta",
    "X-Title": "OpenDDE Harness Agent",
    "X-OpenRouter-Title": "OpenDDE Harness Agent",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}


def _thinking_blocks(blocks: Any) -> list[dict] | None:
    """Anthropic thinking blocks from one chunk, as plain dicts.

    Kept verbatim: the blocks are signed, and Anthropic wants the originals
    back on the next request of a tool-use turn. Anything this cannot read as
    a mapping is dropped rather than guessed at.
    """
    if not blocks:
        return None
    out: list[dict] = []
    for block in blocks:
        if isinstance(block, dict):
            out.append(dict(block))
            continue
        dump = getattr(block, "model_dump", None)
        if dump is not None:
            try:
                out.append(dump())
                continue
            except Exception:
                pass
    return out or None


def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _tool_call_id(upstream: str | None, model: str) -> str:
    """The id to carry for one tool call.

    Upstream's id is kept: Gemini's thought signature travels inside it (LiteLLM
    appends it after a separator and strips it again on the way back), so
    minting a fresh id here silently discarded the signature for every
    Gemini tool call. Mistral is the exception that made this function exist --
    it accepts exactly nine alphanumerics -- so ids bound for it are still
    generated.
    """
    if not upstream or "mistral" in model.lower():
        return _short_tool_id()
    return upstream


def _merge_extra_body(kwargs: dict[str, Any], wire_extra_body: dict[str, Any]) -> None:
    """Merge the provider's built-in extra_body into kwargs instead of overwriting it.

    A model_overrides entry (see _apply_model_overrides) may have already placed
    a user extra_body dict in kwargs -- for example Qwen3's
    extra_body.chat_template_kwargs.enable_thinking. Assigning wire_extra_body
    over it would silently drop those keys. On a key collision, the user's
    value wins: everything wire_extra_body carries is a shipped default
    workaround (see capabilities._WIRE_OVERRIDES -- disabling OpenRouter's
    qwen reasoning mode is the whole table today), and model_overrides is
    documented as the channel that overrides shipped defaults, so a collision
    is the user deliberately reversing one.
    """
    existing = kwargs.get("extra_body")
    if isinstance(existing, dict):
        kwargs["extra_body"] = {**wire_extra_body, **existing}
    else:
        kwargs["extra_body"] = wire_extra_body


def session_affinity_headers() -> dict[str, str]:
    """Headers pinning one caller to one backend replica.

    Self-hosted OpenAI-compatible backends (vLLM and friends) route by this
    header, so a stable value per provider instance keeps prefix-cache hits warm.
    """
    return {"x-session-affinity": uuid.uuid4().hex}


def _rejects_temperature(exc: Exception) -> bool:
    """Whether an API error explicitly says ``temperature`` is unsupported."""
    message = str(exc).lower()
    names_temperature = "temperature" in message
    rejects_parameter = any(
        phrase in message
        for phrase in (
            "unsupported parameter",
            "parameter is not supported",
            "parameter not supported",
            "does not support",
        )
    )
    return names_temperature and rejects_parameter


# How a gateway says "no such route" when it does not use the status code
# alone. Only read beside a 400/404/405; a body naming the model is the model.
_UNKNOWN_ROUTE_PHRASES = (
    "invalid url",
    "unrecognized request url",
    "unknown request url",
    "unsupported endpoint",
    "cannot post",
    "no route",
    "route post",  # Fastify/Nest: "Route POST:/v1/responses not found"
    "not a valid endpoint",
    "method not allowed",
    "unsupported method",
)


_NOT_FOUND_FILLER = {"404", "page", "not", "found", "detail", "error", "message", "code", "status", "http", "null"}


def _plain(item: Any) -> dict[str, Any]:
    """A Responses output item as a plain dict, whatever object LiteLLM handed over."""
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    return {k: v for k, v in vars(item).items() if not k.startswith("_")} if hasattr(item, "__dict__") else {}


def _bare_not_found(message: str) -> bool:
    """Is this 404 body only the status restated, with nothing said about why?"""
    if not message.strip() or "<html" in message:
        return True
    # LiteLLM renders "<ExceptionName>: <Provider>Exception - <body>".
    body = message.rsplit(" - ", 1)[-1]
    words = set(re.findall(r"[a-z0-9]+", body.lower()))
    return bool(words) and words <= _NOT_FOUND_FILLER


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.

    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        disable_auto_cache_control: bool = False,
        extra_body: dict[str, Any] | None = None,
        model_overrides: dict[str, dict[str, Any]] | None = None,
        wire: Literal["responses", "chat"] = "chat",
        *,
        model_overlays: "dict[str, ModelOverlay] | None" = None,
        unparsed_reasoning: bool | None = None,
    ):
        super().__init__(api_key, api_base)
        # None: derive from the resolved spec, as emits_unparsed_reasoning always
        # did. An explicit bool overrides that derivation outright -- for a
        # caller that already knows the answer and for which the spec would
        # guess wrong, e.g. a per-model routing endpoint built with
        # provider_name="custom" for its api_base/api_key shape alone, not
        # because the backend behind it is a self-hosted inference server.
        self._unparsed_reasoning = unparsed_reasoning
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        # When a TokenStrategy handles cache_control placement upstream, turn
        # this on so the provider doesn't also stamp its own breakpoints on top.
        self.disable_auto_cache_control = disable_auto_cache_control
        # Provider-specific request body extras forwarded verbatim to LiteLLM.
        # Common use: OpenRouter routing affinity to keep prompt-cache hits warm,
        #   extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}
        self.extra_body = extra_body or {}
        # User-configured per-model parameter overrides; win over the registry's.
        self.model_overrides = model_overrides or {}
        # The wire this endpoint serves (see ``ProviderSpec.wire``), and the
        # per-model declarations that override it, keyed by ``wire.merge_key``.
        self.wire = wire
        self.model_overlays = model_overlays or {}
        self._wire_ignored: set[str] = set()
        # Some OpenAI-compatible gateways advertise an unknown model using the
        # generic OpenAI capability table. LiteLLM then claims that temperature
        # is supported even when the real backend rejects it. Learn only from an
        # explicit backend rejection and scope the result to endpoint + model.
        self._temperature_unsupported: set[tuple[str, str]] = set()
        self._hosted_web_search_unavailable: set[tuple[str, str]] = set()

        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        # Kept because the id alone cannot say where a request goes: a bare
        # `anthropic/claude-...` sent through this client reads as Anthropic's
        # wire, which is not the wire it will travel on.
        self._provider_name = provider_name or ""
        self._gateway = find_gateway(provider_name, api_key, api_base)
        if self._gateway and self._gateway.name == "openrouter":
            self.extra_headers = {**_OPENROUTER_ATTRIBUTION, **self.extra_headers}

        if api_key:
            self._export_env_extras(api_key, api_base, default_model)

        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True

    def _declared_ceiling(self, model: str, max_tokens: Any) -> int | None:
        """A caller's pin, else the output ceiling the model's overlay declares.

        The budget reserves the declared ceiling, so a request that carried no
        limit could run past the reservation trimming was made against.
        Nothing is volunteered for a model with no declaration: a guessed
        number above a vendor's real cap is a refused request.
        """
        overlay = overlay_for(self.model_overlays, model)
        declared = getattr(overlay, "max_output_tokens", None)
        if isinstance(max_tokens, int):
            return min(max_tokens, int(declared)) if declared else max_tokens
        return int(declared) if declared else None

    def wire_for(self, model: str) -> str:
        """The wire this model's request travels on: its overlay's say, else
        the section's. ``model`` is the stored id, which is what overlays key."""
        overlay = overlay_for(self.model_overlays, model)
        declared = getattr(overlay, "wire", None) if overlay is not None else None
        return declared or self.wire

    def _uses_responses_api(self, original_model: str, resolved_model: str) -> bool:
        """Whether this request travels over the OpenAI Responses wire.

        Only a request LiteLLM sends through its openai driver can: the other
        drivers each have one wire, and ``wire`` says nothing about them.
        """
        wire = self.wire_for(original_model)
        if wire == "responses" and not resolved_model.startswith("openai/"):
            if original_model not in self._wire_ignored:
                self._wire_ignored.add(original_model)
                logger.warning(
                    "wire=responses declared for {} has no effect: it routes as {} rather than through the "
                    "openai driver, which is the only one with a Responses wire",
                    original_model,
                    resolved_model,
                )
            return False
        return wire == "responses"

    def _wire_mismatch(self, exc: Exception, original_model: str, *, responses: bool) -> WireMismatchError | None:
        """The endpoint's refusal of the wire itself, named as such, or None.

        A relay that serves only Chat Completions answers ``/v1/responses`` with
        404, 405 or a 400 that spells out an unknown route. Left as it arrives,
        that 404 reads as "model not found" and the loop hops through fallback
        models over the same transport. Nothing here retries on the other wire:
        opencode and pi both treat a wrong wire as configuration to fix, and a
        silent downgrade would hide exactly what the user has to change.
        """
        status = self._extract_status_code(exc)
        msg = str(exc).lower()
        spelled_out = any(phrase in msg for phrase in _UNKNOWN_ROUTE_PHRASES)
        certain = status == 405 or (status in (400, 404) and spelled_out)
        # A 404 with prose -- "No endpoints found for x", "The model `x` does
        # not exist" -- is the model, and a typo behind a relay that does
        # serve this wire must not be reported as the wire. A 404 that says
        # nothing (a bare "404 page not found", an HTML page, an empty body)
        # is the same body for a base URL missing its /v1, for a wire the
        # endpoint does not serve, and for a model it does not know, so it
        # is reported as all three rather than as the one this code favours.
        if not certain and not (status == 404 and _bare_not_found(msg)):
            return None
        section = canonical_provider_name(self._provider_name) or (self._gateway.name if self._gateway else "")
        dashed = section.replace("_", "-") or "<provider>"
        wire, other = ("responses", "chat") if responses else ("chat", "responses")
        path = "/v1/responses" if responses else "/v1/chat/completions"
        where = self.api_base or "the configured endpoint"
        wire_fix = (
            f'set providers.{section}.wire to "{other}" (ddeharness provider set {dashed} --wire {other}), '
            f"or for this one model: ddeharness provider model set {dashed} {original_model} --wire {other}"
        )
        if certain:
            return WireMismatchError(
                f"{where} does not serve the {wire} wire for {original_model} (HTTP {status} on POST {path}); {wire_fix}",
                status_code=status,
            )
        return EndpointNotFoundError(
            f"{where} answered HTTP 404 to POST {path} for {original_model} without naming a route or a model. "
            f"Check the base URL first (providers.{section or '<provider>'}.apiBase; a missing /v1 gives exactly "
            f"this), then the model id; if the endpoint serves only the {other} wire, {wire_fix}",
            status_code=status,
        )

    def _hosted_web_search_capability_key(self, model: str) -> tuple[str, str]:
        return (str(self.api_base or ""), model)

    def _supports_hosted_web_search(self, model: str, *, responses: bool) -> bool:
        key = self._hosted_web_search_capability_key(model)
        if key in self._hosted_web_search_unavailable:
            return False
        if responses:
            return True
        from opendde_harness.providers.rates import _may_prompt

        if _may_prompt(model):
            # Three LiteLLM drivers answer a model lookup by starting a device
            # login: they print a code to stdout and block in a polling loop.
            # Asking here happens while a turn is being built, so a Copilot
            # session with no token froze mid-request. No hosted search is worth
            # that; the local tool still works.
            return False
        try:
            _, target, _, _ = litellm.get_llm_provider(model=model)
        except Exception:
            target = ""
        if target == "anthropic":
            return True
        try:
            return bool(litellm.supports_web_search(model, custom_llm_provider=target or None))
        except Exception:
            return False

    def _prepare_web_search(
        self,
        kwargs: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        model: str,
        *,
        responses: bool,
    ) -> tuple[list[dict[str, Any]], Any, bool] | None:
        if not tools or not any(is_local_web_search_tool(tool) for tool in tools):
            if tools:
                kwargs["tools"] = tools
            return None
        if not self._supports_hosted_web_search(model, responses=responses):
            kwargs["tools"] = tools
            return None

        original_choice = kwargs.get("tool_choice")
        remaining = [tool for tool in tools if not is_local_web_search_tool(tool)]
        sources_include_added = False
        if responses:
            kwargs["tools"] = [*remaining, {"type": "web_search"}]
            include = list(kwargs.get("include") or [])
            if "web_search_call.action.sources" not in include:
                include.append("web_search_call.action.sources")
                kwargs["include"] = include
                sources_include_added = True
            if selects_local_web_search(original_choice):
                kwargs["tool_choice"] = {"type": "web_search"}
        else:
            if remaining:
                kwargs["tools"] = remaining
            else:
                kwargs.pop("tools", None)
            kwargs["web_search_options"] = {}
            if selects_local_web_search(original_choice):
                kwargs["tool_choice"] = "auto"
        return tools, original_choice, sources_include_added

    def _restore_local_web_search(
        self,
        kwargs: dict[str, Any],
        fallback: tuple[list[dict[str, Any]], Any, bool],
        model: str,
    ) -> None:
        local_tools, original_choice, sources_include_added = fallback
        self._hosted_web_search_unavailable.add(self._hosted_web_search_capability_key(model))
        kwargs.pop("web_search_options", None)
        if sources_include_added:
            include = [value for value in (kwargs.get("include") or []) if value != "web_search_call.action.sources"]
            if include:
                kwargs["include"] = include
            else:
                kwargs.pop("include", None)
        kwargs["tools"] = local_tools
        if original_choice is None:
            kwargs.pop("tool_choice", None)
        else:
            kwargs["tool_choice"] = original_choice
        logger.warning(
            "Hosted web search unavailable for {} at {}; using the local web_search tool",
            model,
            self.api_base or "default endpoint",
        )

    async def _call_with_web_search_fallback(
        self,
        call: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any],
        local_fallback: tuple[list[dict[str, Any]], Any, bool] | None,
        model: str,
        *,
        timeout: Callable[[], float] | None = None,
    ) -> Any:
        try:
            return await self._call_with_temperature_compatibility(call, kwargs, timeout=timeout)
        except Exception as exc:
            if local_fallback is None or not rejects_hosted_web_search(exc):
                raise
            self._restore_local_web_search(kwargs, local_fallback, model)
            return await self._call_with_temperature_compatibility(call, kwargs, timeout=timeout)

    def _first_token_deadline(self) -> Callable[[], float]:
        """Seconds left of the first-token budget, counted from now.

        One budget spans opening the stream and its first event: a gateway
        that accepts the request and then says nothing is the case the bound
        exists for, and it was invisible to a cap that started at the pull.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.generation.first_token_timeout
        return lambda: max(0.001, deadline - loop.time())

    def _temperature_capability_key(self, kwargs: dict[str, Any]) -> tuple[str, str]:
        return (
            str(kwargs.get("api_base") or self.api_base or ""),
            str(kwargs.get("model") or self.default_model),
        )

    def _drop_rejected_temperature(
        self,
        exc: Exception,
        kwargs: dict[str, Any],
    ) -> bool:
        """Learn one explicit capability rejection and prepare a safe retry."""
        if "temperature" not in kwargs or not _rejects_temperature(exc):
            return False
        key = self._temperature_capability_key(kwargs)
        self._temperature_unsupported.add(key)
        kwargs.pop("temperature", None)
        logger.info(
            "Model {} at {} rejected temperature; retrying without it and "
            "remembering the capability for this provider instance",
            key[1],
            key[0] or "default endpoint",
        )
        return True

    async def _call_with_temperature_compatibility(
        self,
        call: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any],
        *,
        timeout: Callable[[], float] | None = None,
    ) -> Any:
        """Call LiteLLM, retrying only an explicit unsupported-temperature error.

        ``timeout`` returns the seconds still allowed, read again before the
        retry so a budget spanning several attempts is one budget.
        """
        request = dict(kwargs)
        if self._temperature_capability_key(request) in self._temperature_unsupported:
            request.pop("temperature", None)

        def cap() -> float:
            return self.generation.timeout if timeout is None else timeout()

        try:
            return await asyncio.wait_for(call(**request), cap())
        except Exception as exc:
            if not self._drop_rejected_temperature(exc, request):
                raise
            return await asyncio.wait_for(call(**request), cap())

    def _export_env_extras(self, api_key: str, api_base: str | None, model: str) -> None:
        """Export a spec's ``env_extras`` (vendor SDK variables LiteLLM reads
        from the environment). The API key itself travels on every call as an
        explicit kwarg, so it is never written to the environment: with several
        provider instances in one process the first would otherwise win and the
        rest inherit its key silently.

        Placeholders: ``{api_key}`` and ``{api_base}`` (the user's, falling
        back to ``spec.default_api_base``).
        """
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _strip_gateway_prefix(self, model: str) -> str:
        """Drop this gateway's own prefix, leaving the upstream vendor's id."""
        if not self._gateway:
            return model
        prefix = f"{self._gateway.model_prefix}/"
        return model[len(prefix) :] if model.startswith(prefix) else model

    def _resolve_model(self, model: str) -> str:
        """The id this request is sent under. See ``providers.wire``."""
        return wire_model(model, gateway=self._gateway)

    def wire_model_id(self, model: str) -> str:
        """See ``LLMProvider.wire_model_id``."""
        return self._resolve_model(model)

    def can_serve(self, model: str) -> bool:
        """See ``LLMProvider.can_serve``.

        A gateway instance answers for any model -- it is the one deciding
        which upstream vendor actually serves it, and its credentials are the
        gateway's own, not tied to one vendor.

        For a direct instance, this only vetoes the one case both sides are
        certain about: this instance's own provider_name resolves to a known,
        non-OAuth spec, the model resolves to a *different* known spec, and
        the two disagree -- that is one vendor's key answering for another
        vendor's model, rejected outright. Every other case is let through
        rather than guessed away here:
          - this instance's own identity does not resolve to a spec (empty
            provider_name, "auto", or a custom passthrough name LiteLLM
            recognizes natively but OpenDDE Harness has no ProviderSpec for, e.g.
            nebius/fireworks/together) -- there is nothing to compare against;
          - the resolved spec is OAuth-based (e.g. github_copilot): one OAuth
            grant can serve several upstream vendors, so a spec mismatch there
            says nothing about whether this instance can serve the model;
          - the model resolves to no spec at all (custom endpoints, bare ids
            only LiteLLM itself recognizes).
        In all of those, the model is not known to be wrong for this
        instance, so it fails loudly at the wire instead of being guessed
        away here.
        """
        if self._gateway is not None:
            return True
        mine = find_by_name(canonical_provider_name(self._provider_name))
        if mine is None or mine.is_oauth:
            return True
        theirs = find_by_model(model)
        if theirs is None:
            return True
        return theirs.name == mine.name

    def emits_unparsed_reasoning(self) -> bool:
        """See ``LLMProvider.emits_unparsed_reasoning``.

        ``self._unparsed_reasoning``, when set explicitly at construction, wins
        outright: it exists for a caller that already knows the answer and for
        which the spec-based guess below is wrong -- a per-model routing
        endpoint is built with ``provider_name="custom"`` for its api_base /
        api_key shape alone, not because the backend behind it is known to be a
        self-hosted inference server, so ``custom`` there would falsely claim
        every one of its responses.

        Otherwise, ``self._gateway`` already answers this for both shapes it
        can hold: a real network gateway (OpenRouter, AiHubMix) fronts one of
        the large hosted vendors below it, so a bare ``</think>`` in content
        is just content; the generic ``custom`` endpoint and a local spec
        (hosted_vllm, ollama_chat) *are* the self-hosted inference server this
        normalization exists for. When nothing was auto-detected, fall back to
        whatever spec ``provider_name`` resolves to.

        An identity that resolves to nothing answers False, the same reading
        ``can_serve`` settled on: an unresolved name says nothing about the
        backend, and several production constructors (the proactive planner,
        callers that build direct big-vendor connections with no
        ``provider_name`` at all -- guessing "self-hosted" there re-opens the
        false-positive cut on ordinary content this gate exists to close. A
        genuinely self-hosted backend is reached through ``custom`` or a
        local spec, which is where the parser-less sglang/vLLM shape comes
        from; a resolved direct big vendor (anthropic, openai, ...) never
        produces it behind its own API.
        """
        if self._unparsed_reasoning is not None:
            return self._unparsed_reasoning
        spec = self._gateway or find_by_name(canonical_provider_name(self._provider_name))
        return spec is not None and (spec.is_local or spec.name == "custom")

    def _supports_cache_control(self, model: str) -> bool:
        """Return True when this request may carry cache_control blocks.

        Decided by ``providers.prompt_cache``, which the token strategies ask too
        -- three copies of this question disagreed, and the one here could not
        have answered for the marks they place.

        The address falls back to the auto-detected gateway when no
        ``provider_name`` was given: several production constructors (the
        internal launch models) pass only an
        ``api_base``, and answering from the model id alone reads
        ``anthropic/claude-...`` as Anthropic's wire while the request actually
        travels through whatever gateway that base names -- a wire that may
        have nowhere honest to put the field.
        """
        from opendde_harness.providers.prompt_cache import accepts_cache_control

        addressed = self._provider_name or (self._gateway.name if self._gateway else "")
        return accepts_cache_control(model, addressed_to=addressed)

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected."""
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": CACHE_CONTROL}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": CACHE_CONTROL}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": CACHE_CONTROL}

        return new_messages, new_tools

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Layer per-model parameter overrides: registry defaults, config on top.

        Config supplies one parameter without discarding the rest of the
        registry's entry -- Kimi keeps its mandated temperature even when the
        user only wanted to set top_p.
        """
        model_lower = model.lower()
        # A gateway-routed id names the gateway and its upstream, not the vendor
        # whose quirks these defaults encode -- so match on keywords there.
        spec = find_by_keywords(self._strip_gateway_prefix(model)) if self._gateway else find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    break
        # Longest match wins, so "kimi-k2.5" beats a broad "kimi" regardless of
        # the order the entries happen to be written in.
        matches = [(p, o) for p, o in self.model_overrides.items() if p.lower() in model_lower]
        if matches:
            kwargs.update(max(matches, key=lambda item: len(item[0]))[1])

    @staticmethod
    def _keeps_thinking_blocks(
        original_model: str, resolved_model: str, *, responses: bool
    ) -> Callable[[dict], bool] | None:
        """Which ``thinking_blocks`` entries this target takes back, or None for none.

        Anthropic wants its own signed ``thinking`` blocks; the Responses wire
        wants the ``reasoning`` items its backend emitted. Each is refused the
        other's, and a session that switched models carries both in one list,
        so the filter is by what the block is rather than by who is asked.
        """
        if responses:
            return lambda block: block.get("type") == REASONING_BLOCK_TYPE and block.get("provider") == PROVIDER_OPENAI
        spec = find_by_model(original_model) or find_by_model(resolved_model)
        if (
            (spec and spec.name == "anthropic")
            or "claude" in original_model.lower()
            or resolved_model.startswith("anthropic/")
        ):
            return lambda block: block.get("type") in _ANTHROPIC_BLOCK_TYPES
        return None

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize tool_call_id to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    @staticmethod
    def _sanitize_messages(
        messages: list[dict[str, Any]],
        *,
        keep_blocks: Callable[[dict], bool] | None = None,
        require_reasoning_key: bool = False,
        normalize_ids: bool = True,
    ) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key.

        ``keep_blocks`` says which ``thinking_blocks`` entries the target takes
        back; None strips the key entirely. ``normalize_ids`` is off for the
        Responses wire: its ``call|item`` ids are the backend's own to pair,
        and the 9-char rewrite would sever the item id from the reasoning
        that references it.
        """
        allowed = _ALLOWED_MSG_KEYS | (_ANTHROPIC_EXTRA_KEYS if keep_blocks else frozenset())
        sanitized = LLMProvider._sanitize_request_messages(messages, allowed)
        if keep_blocks is not None:
            for clean in sanitized:
                blocks = clean.get("thinking_blocks")
                if isinstance(blocks, list):
                    kept = [b for b in blocks if isinstance(b, dict) and keep_blocks(b)]
                    if kept:
                        clean["thinking_blocks"] = kept
                    else:
                        clean.pop("thinking_blocks", None)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str) or not normalize_ids:
                return value
            return id_map.setdefault(value, LiteLLMProvider._normalize_tool_call_id(value))

        for clean in sanitized:
            # Keep assistant tool_calls[].id and tool tool_call_id in sync after
            # shortening, otherwise strict providers reject the broken linkage.
            if isinstance(clean.get("tool_calls"), list):
                normalized_tool_calls = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized_tool_calls.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized_tool_calls.append(tc_clean)
                clean["tool_calls"] = normalized_tool_calls

            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])

            # DeepSeek's thinking mode refuses a conversation whose assistant
            # turn arrives without the key at all, even when the model itself
            # produced no reasoning for that turn -- which it does after a tool
            # result. An empty value satisfies it, verified against the live
            # API, and it repairs turns recorded before this was understood.
            if require_reasoning_key and clean.get("role") == "assistant":
                clean.setdefault("reasoning_content", "")
        return sanitized

    def _cache_marked(
        self,
        original_model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        responses: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Messages and tools with prompt-cache breakpoints placed, or stripped
        when this wire cannot carry them. Responses performs provider-side
        prefix caching, so Chat cache-control annotations are not valid input
        items there."""
        if responses or not self._supports_cache_control(original_model):
            return prompt_cache.strip(messages, tools)
        if self.disable_auto_cache_control:
            return messages, tools
        return self._apply_cache_control(messages, tools)

    def _request_kwargs(
        self,
        original_model: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int | None,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        responses: bool,
        stream: bool,
    ) -> tuple[dict[str, Any], tuple[list[dict[str, Any]], Any, bool] | None]:
        """The LiteLLM call's keyword arguments for either wire, and the local
        web-search fallback ``_call_with_web_search_fallback`` restores when a
        hosted search is refused."""
        messages, tools = self._cache_marked(original_model, messages, tools, responses=responses)
        compat = detect_compat(original_model, resolved_model=model, api_base=self.api_base)
        messages = self._sanitize_messages(
            self._sanitize_empty_content(messages),
            keep_blocks=self._keeps_thinking_blocks(original_model, model, responses=responses),
            require_reasoning_key=compat.requires_reasoning_content,
            normalize_ids=not responses,
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            # Per-read httpx cap forwarded to the underlying client. A stream's
            # read timer resets on every byte, so the pulls below also bound
            # each gap with wait_for; a non-streamed call gets the wall-clock
            # cap around the whole await.
            "timeout": self.generation.first_token_timeout if stream else self.generation.timeout,
        }
        if stream:
            kwargs["stream"] = True
        # Never volunteered: a caller that wants a short answer pins one, and
        # a vendor that requires the field has a LiteLLM transformation that
        # supplies it. Clamped to at least 1 when present, since LiteLLM
        # rejects a zero or negative value outright.
        if responses:
            kwargs["input"] = responses_input(messages)
            kwargs["store"] = False
            # With nothing stored server-side, the chain of thought has to
            # travel with the history: asked for here, carried on the reply's
            # thinking_blocks, replayed by responses_input before the item it
            # produced. Without it the model re-plans from nothing at every
            # tool call.
            kwargs["include"] = ["reasoning.encrypted_content"]
            if max_tokens is not None:
                kwargs["max_output_tokens"] = max(1, max_tokens)
        else:
            kwargs["messages"] = messages
            if stream:
                # OpenAI-compatible providers only emit the trailing usage chunk
                # when usage is explicitly requested; without it the stream
                # carries no token counts and cost / context tracking sees zero.
                kwargs["stream_options"] = {"include_usage": True}
            if max_tokens is not None:
                kwargs["max_tokens"] = max(1, max_tokens)
            self._apply_model_overrides(model, kwargs)

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.extra_body:
            _merge_extra_body(kwargs, self.extra_body)
        if responses:
            # pi's shape: the level the model takes, with a summary so the
            # reasoning text streams back; ``off`` is spelled ``none``.
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": openai_effort(original_model, reasoning_effort), "summary": "auto"}
        else:
            apply_thinking(
                kwargs, compat, reasoning_effort=reasoning_effort, model=original_model, max_tokens=max_tokens
            )

        if responses:
            tools = responses_tools(tools)
            if tools:
                kwargs["tool_choice"] = responses_tool_choice(tool_choice) or "auto"
        elif tools:
            kwargs["tool_choice"] = tool_choice or "auto"
        return kwargs, self._prepare_web_search(kwargs, tools, model, responses=responses)

    def _error_response(self, exc: Exception) -> LLMResponse:
        """The error as content, classified while the exception is alive
        (status code + type) -- the retry/fallback layer reads that verdict."""
        classification = self.classify_error(exc)
        head = self._provider_name or (self._gateway.name if self._gateway else None)
        return LLMResponse(
            content=format_llm_error(exc, classification, provider=head),
            finish_reason="error",
            error_classification=classification,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)
        responses = self._uses_responses_api(original_model, model)
        kwargs, local_web_search_tools = self._request_kwargs(
            original_model,
            model,
            messages,
            tools,
            max_tokens=self._declared_ceiling(original_model, max_tokens),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            responses=responses,
            stream=False,
        )
        try:
            raw = await self._call_with_web_search_fallback(
                aresponses if responses else acompletion,
                kwargs,
                local_web_search_tools,
                model,
            )
            return self._parse_responses_response(raw) if responses else self._parse_response(raw)
        except Exception as e:
            return self._error_response(self._wire_mismatch(e, original_model, responses=responses) or e)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Streaming counterpart to chat().

        Yields one StreamDelta per non-empty chunk. Signature matches chat()
        so callers can swap providers transparently.

        Generation defaults resolve from ``self.generation`` the same way
        ``chat_with_retry`` does: literal defaults here would shadow the user's
        configuration, since the agent loop calls this with messages/tools/model
        only.
        """
        gen = getattr(self, "generation", None) or GenerationSettings()
        if max_tokens is self._SENTINEL:
            max_tokens = gen.max_tokens
        if temperature is self._SENTINEL:
            temperature = gen.temperature
        original_model = model or self.default_model
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.effort_for(original_model)
        model = self._resolve_model(original_model)
        responses = self._uses_responses_api(original_model, model)
        kwargs, local_web_search_tools = self._request_kwargs(
            original_model,
            model,
            messages,
            tools,
            max_tokens=self._declared_ceiling(original_model, max_tokens),
            temperature=float(temperature) if isinstance(temperature, (int, float)) else 0.7,
            reasoning_effort=reasoning_effort if isinstance(reasoning_effort, str) else None,
            tool_choice=tool_choice,
            responses=responses,
            stream=True,
        )

        if responses:
            try:
                async for delta in self._chat_responses_stream(kwargs, local_web_search_tools, model):
                    yield delta
            except Exception as exc:
                mismatch = self._wire_mismatch(exc, original_model, responses=True)
                if mismatch is None:
                    raise
                raise mismatch from exc
            return

        def _retry_without_breakpoints(exc: Exception) -> bool:
            """Learn the refusal and take the marks off, or say this is not one.

            The one retry this path takes. Restarting a *partially streamed* call
            is the problem that kept retry out of here, and this is not that: the
            refusal arrives before any chunk has been handed to the caller, so
            nothing has been said that would have to be unsaid. Without it the
            learned downgrade never reaches the surface that actually streams --
            the TUI, where the affected model answered 400 on every single turn.
            """
            if prompt_cache.is_suppressed(original_model) or not prompt_cache.is_rejection(exc):
                return False
            prompt_cache.suppress(original_model)
            kwargs["messages"], stripped = prompt_cache.strip(kwargs["messages"], kwargs.get("tools"))
            if stripped is not None:
                kwargs["tools"] = stripped
            return True

        remaining = self._first_token_deadline()

        async def _open():
            return (
                await self._call_with_web_search_fallback(
                    acompletion,
                    kwargs,
                    local_web_search_tools,
                    model,
                    timeout=remaining,
                )
            ).__aiter__()

        async def _close(target: Any) -> None:
            aclose = getattr(target, "aclose", None)
            if aclose is not None:
                await aclose()

        # Per-chunk idle cap: the timer resets on every chunk, so a long but
        # steadily-progressing generation is fine while a mid-stream stall (no
        # bytes for `timeout` seconds) raises TimeoutError instead of hanging.
        # Everything from the open onward sits inside the one try/finally, so the
        # underlying HTTP stream is closed deterministically on any exit -- a
        # first-chunk timeout included, which is the most likely one there is
        # (gateway queueing, cold start).
        # A chunk of None is a chunk, not the end of the stream. Pulling the
        # first one before the loop needs a value meaning "there was none", and
        # reusing None for it would let a provider that yields one truncate the
        # response silently -- which is not what the loop did before.
        done = object()

        stream: Any = None
        try:
            # The open and the first pull are one unit, and the `except` has to
            # cover both: an OpenAI-shaped route raises at the open, and a gateway
            # that defers the request until the first pull raises there instead.
            try:
                stream = await _open()
                first = await asyncio.wait_for(stream.__anext__(), remaining())
            except StopAsyncIteration:
                first = done
            except Exception as exc:
                hosted_search_retry = local_web_search_tools is not None and rejects_hosted_web_search(exc)
                if hosted_search_retry:
                    self._restore_local_web_search(
                        kwargs,
                        local_web_search_tools,
                        model,
                    )
                temperature_retry = False if hosted_search_retry else self._drop_rejected_temperature(exc, kwargs)
                if not hosted_search_retry and not temperature_retry and not _retry_without_breakpoints(exc):
                    mismatch = self._wire_mismatch(exc, original_model, responses=False)
                    if mismatch is not None:
                        raise mismatch from exc
                    raise
                # The refused stream is finished with; closing it before opening
                # the replacement keeps at most one live at a time. It is None
                # when the open itself was what failed.
                await _close(stream)
                stream = await _open()
                try:
                    first = await asyncio.wait_for(stream.__anext__(), remaining())
                except StopAsyncIteration:
                    first = done

            chunk = first
            while chunk is not done:
                delta = self._normalize_stream_chunk(chunk)
                if delta is not None:
                    yield delta
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), self.generation.idle_timeout)
                except StopAsyncIteration:
                    break
        finally:
            await _close(stream)

    async def _chat_responses_stream(
        self,
        kwargs: dict[str, Any],
        local_web_search_tools: tuple[list[dict[str, Any]], Any, bool] | None,
        model: str,
    ) -> AsyncIterator[StreamDelta]:
        """Stream Responses events while preserving OpenDDE Harness's Chat-shaped deltas."""
        remaining = self._first_token_deadline()
        stream = await self._call_with_web_search_fallback(
            aresponses,
            kwargs,
            local_web_search_tools,
            model,
            timeout=remaining,
        )
        iterator = stream.__aiter__()
        calls: dict[str, tuple[int, str, str]] = {}
        reasoning_items: list[dict[str, Any]] = []
        order: list[str] = []
        messages_out: dict[str, str] = {}
        web_searches: dict[str, Any] = {}
        started_web_searches: set[str] = set()
        streamed_citations: dict[str, str] = {}
        terminated = False
        gap = remaining()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(iterator.__anext__(), gap)
                except StopAsyncIteration:
                    break
                gap = self.generation.idle_timeout
                # Attribute access only: dumping the event would serialize the
                # usage LiteLLM mistyped (see litellm_setup) and warn each time.
                data = event
                event_type = value_of(data, "type", "")

                if event_type in {
                    "response.web_search_call.in_progress",
                    "response.web_search_call.searching",
                }:
                    item_id = str(value_of(data, "item_id", ""))
                    if item_id and item_id not in started_web_searches:
                        started_web_searches.add(item_id)
                        yield StreamDelta(
                            content=None,
                            builtin_tool_event={
                                "phase": "start",
                                "tool_call_id": item_id,
                                "name": "web_search",
                                "arguments": {},
                                "display": "searching the web",
                            },
                        )
                    continue
                if event_type == "response.output_text.annotation.added":
                    annotation = value_of(data, "annotation", {}) or {}
                    if value_of(annotation, "type", "") == "url_citation":
                        url = str(value_of(annotation, "url", "") or "").strip()
                        title = str(value_of(annotation, "title", "") or "").strip()
                        if url:
                            streamed_citations[url] = title
                    continue

                if event_type == "response.output_text.delta":
                    yield StreamDelta(content=value_of(data, "delta", ""))
                    continue
                if event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    yield StreamDelta(content=None, reasoning_content=value_of(data, "delta", ""))
                    continue
                if event_type == "response.output_item.added":
                    item = value_of(data, "item", {})
                    item_type = value_of(item, "type")
                    if item_type == "web_search_call":
                        item_id = str(value_of(item, "id", value_of(data, "item_id", "")))
                        if item_id:
                            web_searches[item_id] = item
                            if item_id not in started_web_searches:
                                arguments, display = web_search_action(item)
                                started_web_searches.add(item_id)
                                yield StreamDelta(
                                    content=None,
                                    builtin_tool_event={
                                        "phase": "start",
                                        "tool_call_id": item_id,
                                        "name": "web_search",
                                        "arguments": arguments,
                                        "display": display,
                                    },
                                )
                        continue
                    if item_type != "function_call":
                        continue
                    item_id = str(value_of(item, "id", ""))
                    call_id = str(value_of(item, "call_id", item_id))
                    name = str(value_of(item, "name", ""))
                    index = int(value_of(data, "output_index", 0) or 0)
                    # Both ids travel as one string: the call_id a result
                    # answers and the item id the backend pairs reasoning with.
                    call_id = join_tool_call_id(call_id, item_id)
                    calls[item_id] = (index, call_id, name)
                    yield StreamDelta(
                        content=None,
                        tool_call_delta={
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                    )
                    continue
                if event_type == "response.function_call_arguments.delta":
                    item_id = str(value_of(data, "item_id", ""))
                    index, call_id, name = calls.get(
                        item_id,
                        (int(value_of(data, "output_index", 0) or 0), "", ""),
                    )
                    yield StreamDelta(
                        content=None,
                        tool_call_delta={
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call_id or None,
                                    "type": "function",
                                    "function": {
                                        "name": name or None,
                                        "arguments": value_of(data, "delta", ""),
                                    },
                                }
                            ]
                        },
                    )
                    continue
                if event_type == "response.output_item.done":
                    item = value_of(data, "item", {})
                    item_type = value_of(item, "type", "")
                    item_id = str(value_of(item, "id", "") or "")
                    if item_id and item_type in ("reasoning", "message", "function_call"):
                        order.append(item_id)
                    if item_type == "reasoning":
                        reasoning_items.append(_plain(item))
                    elif item_type == "message" and item_id:
                        messages_out[item_id] = message_text(item)
                    elif item_type == "web_search_call":
                        item_id = str(value_of(item, "id", value_of(data, "item_id", "")))
                        if item_id:
                            web_searches[item_id] = item
                    continue
                if event_type == "response.completed":
                    terminated = True
                    response = value_of(data, "response", {})
                    block = reasoning_block(reasoning_items, order, messages_out, provider=PROVIDER_OPENAI)
                    if block is not None:
                        yield StreamDelta(content=None, thinking_blocks=[block])
                    citations = url_citations(response)
                    citations.extend(streamed_citations.items())
                    for item in value_of(response, "output", []) or []:
                        if value_of(item, "type", "") == "web_search_call":
                            item_id = str(value_of(item, "id", ""))
                            if item_id:
                                web_searches[item_id] = item
                    for item_id, item in web_searches.items():
                        if item_id not in started_web_searches:
                            arguments, display = web_search_action(item)
                            yield StreamDelta(
                                content=None,
                                builtin_tool_event={
                                    "phase": "start",
                                    "tool_call_id": item_id,
                                    "name": "web_search",
                                    "arguments": arguments,
                                    "display": display,
                                },
                            )
                        yield StreamDelta(
                            content=None,
                            builtin_tool_event={
                                "phase": "complete",
                                "tool_call_id": item_id,
                                "result_preview": web_search_preview(item, citations),
                                "truncated": False,
                            },
                        )
                    yield StreamDelta(
                        content=None,
                        usage=responses_usage(response),
                        finish_reason="tool_calls" if calls else "stop",
                    )
                    continue
                if event_type == "response.incomplete":
                    terminated = True
                    response = value_of(data, "response", {})
                    block = reasoning_block(reasoning_items, order, messages_out, provider=PROVIDER_OPENAI)
                    if block is not None:
                        yield StreamDelta(content=None, thinking_blocks=[block])
                    yield StreamDelta(
                        content=None,
                        usage=responses_usage(response),
                        finish_reason=responses_incomplete_reason(value_of(response, "incomplete_details", None)),
                    )
                    continue
                if event_type == "response.failed":
                    raise RuntimeError(f"Responses API failed: {responses_error_text(value_of(data, 'response', {}))}")
                if event_type == "error":
                    # A top-level error ends the run without a response object.
                    # Skipped as an unknown event, it left the stream finishing
                    # on whatever had already been yielded, which reads as a
                    # complete answer.
                    raise RuntimeError(f"Responses API error: {responses_error_text(data)}")
            if not terminated:
                # The connection closed with no terminal event: the answer stops
                # wherever the socket did. Without this the assembler defaulted
                # to "stop" (or "tool_calls"), so a call whose arguments never
                # arrived was dispatched with an empty object and the reply
                # looked finished.
                yield StreamDelta(content=None, finish_reason="length")
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    def _normalize_stream_chunk(self, chunk: Any) -> StreamDelta | None:
        """Normalize a raw provider chunk into a StreamDelta.

        Default: OpenAI shape — `chunk.choices[0].delta.content` (str | None),
        `delta.tool_calls` (list | None), and a final `chunk.usage` snapshot
        on the trailing chunk for some providers. Returns None when the chunk
        carries no content / tool_call / usage payload so callers can skip.
        """
        try:
            choices = getattr(chunk, "choices", None)
            if not choices:
                return None
            delta_obj = getattr(choices[0], "delta", None)
            if delta_obj is None:
                return None
            content = getattr(delta_obj, "content", None)
            tool_calls = getattr(delta_obj, "tool_calls", None)
            usage = getattr(chunk, "usage", None)
            reasoning_content = getattr(delta_obj, "reasoning_content", None) or None
            thinking_blocks = _thinking_blocks(getattr(delta_obj, "thinking_blocks", None))
            # Upstream states why it stopped only on the terminal chunk, which
            # otherwise carries no payload at all. Dropping that chunk (as the
            # emptiness check below used to) discards the one signal that says
            # the response was cut off at the output ceiling rather than
            # finished -- the difference between "the model is done" and "the
            # model was interrupted mid-token".
            finish_reason = getattr(choices[0], "finish_reason", None) or None

            tool_call_delta: dict[str, Any] | None = None
            if tool_calls:
                # Surface raw tool_call deltas as a list of dict snapshots so
                # downstream layers can re-assemble; intentionally light-touch
                # here (full tool-call accumulation is the consumer's job).
                tool_call_delta = {"tool_calls": [tc.model_dump() for tc in tool_calls]}

            usage_dict: dict[str, Any] | None = None
            if usage is not None:
                usage_dict = usage.model_dump(warnings=False) if hasattr(usage, "model_dump") else dict(usage)

            if (
                content is None
                and tool_call_delta is None
                and usage_dict is None
                and reasoning_content is None
                and thinking_blocks is None
                and finish_reason is None
            ):
                return None

            return StreamDelta(
                content=content,
                tool_call_delta=tool_call_delta,
                usage=usage_dict,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
                finish_reason=finish_reason,
            )
        except (AttributeError, IndexError):
            return None

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason

        # Some providers (e.g. GitHub Copilot) split content and tool_calls
        # across multiple choices. Merge them so tool_calls are not lost.
        raw_tool_calls = []
        for ch in response.choices:
            msg = ch.message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls.extend(msg.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and msg.content:
                content = msg.content

        if len(response.choices) > 1:
            logger.debug(
                "LiteLLM response has {} choices, merged {} tool_calls", len(response.choices), len(raw_tool_calls)
            )

        tool_calls = []
        for tc in raw_tool_calls:
            # Parse arguments from JSON string if needed. Strict first, so that
            # "this needed repairing" survives as a signal: an upstream cut mid
            # arguments arrives as an unclosed blob, and json_repair closes it
            # silently. Measured against openrouter, both Anthropic and OpenAI
            # backends send the raw fragment here, so this is the one locally
            # computable clue that the call was cut.
            args = tc.function.arguments
            repaired = False
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = json_repair.loads(args)
                    repaired = True

            provider_specific_fields = getattr(tc, "provider_specific_fields", None) or None
            function_provider_specific_fields = getattr(tc.function, "provider_specific_fields", None) or None

            tool_calls.append(
                ToolCallRequest(
                    id=_tool_call_id(
                        getattr(tc, "id", None), str(getattr(response, "model", "") or self.default_model)
                    ),
                    name=tc.function.name,
                    arguments=args,
                    provider_specific_fields=provider_specific_fields,
                    function_provider_specific_fields=function_provider_specific_fields,
                    run_meta=RunMeta(arguments_repaired=True) if repaired else None,
                )
            )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # Cache token extraction. LiteLLM normalizes these across providers
            # in different shapes depending on where the response came from:
            #   - Anthropic native:  usage.cache_read_input_tokens / cache_creation_input_tokens
            #   - LiteLLM internal:  usage._cache_read_input_tokens / _cache_creation_input_tokens
            #   - OpenAI-style:      usage.prompt_tokens_details.cached_tokens (read only)
            #   - OpenRouter:        usage.prompt_tokens_details.cached_tokens
            #                        usage.prompt_tokens_details.cache_write_tokens
            details = getattr(response.usage, "prompt_tokens_details", None)
            cache_read = (
                getattr(response.usage, "cache_read_input_tokens", None)
                or getattr(response.usage, "_cache_read_input_tokens", None)
                or (getattr(details, "cached_tokens", None) if details else None)
                or 0
            )
            cache_write = (
                getattr(response.usage, "cache_creation_input_tokens", None)
                or getattr(response.usage, "_cache_creation_input_tokens", None)
                or (getattr(details, "cache_write_tokens", None) if details else None)
                or 0
            )
            if cache_read:
                usage["cache_read_input_tokens"] = int(cache_read)
            if cache_write:
                usage["cache_creation_input_tokens"] = int(cache_write)

        reasoning_content = getattr(message, "reasoning_content", None) or None
        thinking_blocks = getattr(message, "thinking_blocks", None) or None

        if not reasoning_content and isinstance(content, str) and self.emits_unparsed_reasoning():
            split_reasoning, content = split_orphan_think(content)
            reasoning_content = split_reasoning or reasoning_content

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def _parse_responses_response(self, response: Any) -> LLMResponse:
        """Normalize Responses messages/function calls to OpenDDE Harness's contract."""
        tool_calls: list[ToolCallRequest] = []
        reasoning_parts: list[str] = []
        reasoning_items: list[dict[str, Any]] = []
        order: list[str] = []
        messages_out: dict[str, str] = {}
        for item in value_of(response, "output", []) or []:
            item_type = value_of(item, "type", "")
            item_id = str(value_of(item, "id", "") or "")
            if item_id and item_type in ("reasoning", "message", "function_call"):
                order.append(item_id)
            if item_type == "message" and item_id:
                messages_out[item_id] = message_text(item)
            if item_type == "function_call":
                raw_arguments = value_of(item, "arguments", "{}")
                repaired = False
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except Exception:
                    arguments = json_repair.loads(raw_arguments)
                    repaired = True
                tool_calls.append(
                    ToolCallRequest(
                        id=join_tool_call_id(str(value_of(item, "call_id", "") or _short_tool_id()), item_id),
                        name=str(value_of(item, "name", "")),
                        arguments=arguments if isinstance(arguments, dict) else {},
                        run_meta=RunMeta(arguments_repaired=True) if repaired else None,
                    )
                )
            elif item_type == "reasoning":
                reasoning_items.append(_plain(item))
                for summary in value_of(item, "summary", []) or []:
                    text = value_of(summary, "text", "")
                    if text:
                        reasoning_parts.append(str(text))

        # Status before tool calls. A failed or truncated response can still
        # carry a function_call, and reading the calls first reported it as a
        # normal tool turn: the retry ladder never saw the failure and the loop
        # executed tools the backend had already given up on.
        status = str(value_of(response, "status", "completed"))
        content = value_of(response, "output_text", "") or ""
        if status not in ("completed", "incomplete"):
            # failed, cancelled, queued, in_progress: none of them is an answer.
            # Reported as a normal turn, a cancelled run's tool calls were
            # dispatched and a queued one finished the turn with empty content.
            return LLMResponse(
                content=content or f"Responses API {status}: {responses_error_text(response)}",
                finish_reason="error",
                usage=responses_usage(response),
            )
        if status == "incomplete":
            finish_reason = responses_incomplete_reason(value_of(response, "incomplete_details", None))
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"
        block = reasoning_block(reasoning_items, order, messages_out, provider=PROVIDER_OPENAI)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=responses_usage(response),
            reasoning_content="\n".join(reasoning_parts) or None,
            thinking_blocks=[block] if block else None,
            truncated=finish_reason == "length",
        )

    @property
    def provider_name(self) -> str:
        """The config section this provider was built for, or ``""``.

        Read by callers deciding whether the model string is the operator's
        own naming (a ``custom`` gateway serves whatever its endpoint calls
        the model) -- see ``capabilities._model_id_is_caller_chosen``.
        """
        return self._provider_name

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
