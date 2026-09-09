"""Shared CLI helpers used by multiple top-level command modules.

Extracted from commands.py so that per-command modules
(``doctor_commands.py``, ``tui_commands.py``, ...) can import them
directly instead of going through lazy wrappers.

Function names drop the leading underscore: the file itself is marked
internal with the ``_helpers`` prefix, so members do not also need the
private-name convention.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from rich.console import Console

from opendde_harness.config.schema import Config

console = Console()


DEFAULT_PROBE_MESSAGE = "Hi! Say hello in one sentence."


def check_provider_credentials(config: Config) -> None:
    """Fail-fast when the configured provider is missing required credentials.

    Cheap (no litellm import), so it can run at startup even when the real
    provider is built lazily.

    Raises ``MissingCredentialsError`` rather than printing and exiting: three entry
    points call this, and only one of them is a terminal. Each renders the
    failure in its own idiom -- the CLI as a red line and exit 1, the TUI as an
    RPC error carrying the same sentence.

    What counts as configured is `providers.auth`, the same declaration routing
    and `provider list` consult. Deciding it here as well is what produced three
    verdicts on one config: a Gemini section holding only `api_key_list` read as
    configured in `provider list` and refused to start, and Azure with a key and
    no address was routed and displayed as configured yet rejected here.
    """
    from opendde_harness.providers.auth import MissingCredentialsError, credential_status
    from opendde_harness.providers.registry import find_by_model, split_model_id

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    if not provider_name:
        # Routing found no configured section, so name the provider the model id
        # points at rather than reporting on nothing.
        spec = find_by_model(model)
        provider_name = spec.name if spec else split_model_id(model)[0]
    if not provider_name:
        raise MissingCredentialsError(
            "no provider configured",
            # A command, not a config path: the old text pointed at
            # ~/.opendde_harness/config.json, the layout the CLI exists to hide.
            remedy=(
                "Run: ddeharness provider set <name> --api-key <key>, then ddeharness provider use <name>/<model>\n"
                "Or run `ddeharness onboard` for guided setup."
            ),
        )

    status = credential_status(provider_name, config.providers.get(provider_name), include_external=True)
    if status.ok:
        return

    # A first run fails this check while naming a provider the user never chose:
    # with nothing configured, routing falls back to the schema's default model,
    # whose vendor then gets reported as the thing to go fix. Sending someone who
    # only has an OpenRouter key to `provider set anthropic` is the wrong errand,
    # so answer the wizard instead. Both halves are required -- a user who picked
    # this model, or who has some other provider working, gets the specific
    # verdict, which for the OAuth families names a sign-in rather than a key.
    # Names come from the declared fields *and* the extras: an undeclared
    # provider key is a supported shape, and `ProvidersConfig.get` is the only
    # place allowed to resolve either kind, so route both through it rather than
    # reading `__dict__` -- which sees no extras and would call a user whose one
    # working credential lives there unconfigured.
    chose_a_model = config.agents.defaults.model != type(config.agents.defaults)().model
    configured = (*config.providers.__dict__, *(config.providers.model_extra or {}))
    if not chose_a_model and not any(
        credential_status(name, config.providers.get(name), include_external=True).ok for name in configured
    ):
        raise MissingCredentialsError(
            "no provider is configured yet -- run `ddeharness onboard` for guided setup",
            remedy="Already have a key? ddeharness provider set <name> --api-key <key>",
        )

    raise MissingCredentialsError(
        status.summary,
        provider=provider_name,
        remedy="Run `ddeharness onboard` for guided setup.",
    )


def make_provider(config: Config):
    """Create the appropriate LLM provider from config."""
    from opendde_harness.providers.auth import MissingCredentialsError
    from opendde_harness.providers.azure_openai_provider import AzureOpenAIProvider
    from opendde_harness.providers.base import GenerationSettings
    from opendde_harness.providers.openai_codex_provider import OpenAICodexProvider

    check_provider_credentials(config)

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    from opendde_harness.providers.registry import default_wire, endpoints_unsupported_reason, find_by_name

    spec = find_by_name(provider_name) if provider_name else None
    client = spec.client if spec else ""
    wire = (p.wire if p else None) or default_wire(provider_name)
    model_overlays = config.providers.model_overlays()

    if p and p.endpoints:
        reason = endpoints_unsupported_reason(provider_name)
        if reason:
            raise MissingCredentialsError(reason, provider=provider_name or "")

    if client == "codex":
        provider = OpenAICodexProvider(default_model=model)
    elif client == "minimax_oauth":
        from opendde_harness.providers.minimax_oauth_provider import MiniMaxOAuthProvider

        provider = MiniMaxOAuthProvider(
            region="global" if provider_name == "minimax_global" else "cn",
            default_model=model,
        )
    elif client == "azure":
        provider = AzureOpenAIProvider(
            api_key=p.effective_api_key,
            api_base=p.api_base,
            default_model=model,
            deployment=getattr(p, "deployment", "") or "",
            api_version=getattr(p, "api_version", "") or "2024-10-21",
        )
    else:
        from opendde_harness.providers.capabilities import wire_overrides
        from opendde_harness.providers.endpoints import provider_endpoints
        from opendde_harness.providers.litellm_provider import LiteLLMProvider

        eps = provider_endpoints(p) if p else []
        if len(eps) > 1:
            from opendde_harness.providers.endpoint_rotor import EndpointRotorProvider

            def make_inner(ep):
                return LiteLLMProvider(
                    api_key=ep.api_key,
                    # ``ep.api_base`` already carries the section's flat address
                    # when the endpoint named none of its own (see
                    # ``provider_endpoints``); the fallback here is only for a
                    # gateway/local provider whose *flat* address is also empty,
                    # where ``get_api_base`` still has the spec's default to
                    # offer.
                    api_base=ep.api_base or config.get_api_base(model),
                    default_model=model,
                    extra_headers=ep.extra_headers,
                    provider_name=provider_name,
                    wire=wire,
                    model_overlays=model_overlays,
                    extra_body=wire_overrides(provider_name, model) or None,
                    model_overrides=config.agents.defaults.model_overrides,
                )

            provider = EndpointRotorProvider(
                eps,
                make_inner,
                default_model=model,
                strategy=p.endpoint_strategy if p else "sticky",
            )
        elif eps:
            extra_body = wire_overrides(provider_name, model) or None
            provider = LiteLLMProvider(
                api_key=eps[0].api_key,
                # Same fallback as ``make_inner`` above: only reached when the
                # flat address is empty too, for a gateway/local provider's
                # spec default.
                api_base=eps[0].api_base or config.get_api_base(model),
                default_model=model,
                extra_headers=eps[0].extra_headers,
                provider_name=provider_name,
                wire=wire,
                model_overlays=model_overlays,
                extra_body=extra_body,
                model_overrides=config.agents.defaults.model_overrides,
            )
        else:
            extra_body = wire_overrides(provider_name, model) or None
            provider = LiteLLMProvider(
                api_key=p.effective_api_key if p else None,
                api_base=config.get_api_base(model),
                default_model=model,
                extra_headers=p.extra_headers if p else None,
                provider_name=provider_name,
                wire=wire,
                model_overlays=model_overlays,
                extra_body=extra_body,
                model_overrides=config.agents.defaults.model_overrides,
            )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        reasoning_effort=defaults.reasoning_effort,
        timeout=defaults.llm_call_timeout,
        first_token_timeout=defaults.llm_first_token_timeout,
        idle_timeout=defaults.llm_idle_timeout,
    )
    return provider


def make_lazy_provider(config: Config):
    """Provider that defers the real (litellm-importing) build to the first model
    call, so AgentLoop construction stays fast. Credentials are checked now
    (fail-fast preserved) and the real provider is pre-warmed in the background."""
    from opendde_harness.providers.base import GenerationSettings
    from opendde_harness.providers.endpoints import provider_endpoints
    from opendde_harness.providers.lazy import LazyProvider

    check_provider_credentials(config)
    defaults = config.agents.defaults

    p = config.get_provider(defaults.model)
    eps = provider_endpoints(p) if p else []
    initial_endpoint_label = eps[0].label if len(eps) > 1 else None

    provider = LazyProvider(
        factory=lambda: make_provider(config),
        default_model=defaults.model,
        generation=GenerationSettings(
            temperature=defaults.temperature,
            reasoning_effort=defaults.reasoning_effort,
            timeout=defaults.llm_call_timeout,
            first_token_timeout=defaults.llm_first_token_timeout,
            idle_timeout=defaults.llm_idle_timeout,
        ),
        initial_endpoint_label=initial_endpoint_label,
    )
    provider.prewarm()
    return provider


def send_probe(
    *,
    message: str = DEFAULT_PROBE_MESSAGE,
    # A reasoning model can spend half a minute on its first token, and a
    # gateway in front of one adds to that. Fifteen seconds failed setups whose
    # only fault was being slow.
    timeout_s: int = 60,
    max_tokens: int = 200,
) -> tuple[str, int | None, float]:
    """Build provider from current config and exchange one chat message.

    Shared by ``onboard`` Step 3 and ``doctor --probe``. Bypasses the full
    ``AgentLoop`` so the probe only proves the provider answers, not that
    the agent runtime is healthy.

    Returns ``(response_text, tokens_used, elapsed_s)``. Raises ``RuntimeError``
    on provider error, ``asyncio.TimeoutError`` on timeout, or whatever
    ``load_config`` / ``make_provider`` raise on config failure.
    """
    from opendde_harness.config.loader import load_config

    config = load_config()
    provider = make_provider(config)

    start = time.monotonic()
    response = asyncio.run(
        asyncio.wait_for(
            provider.chat_with_retry(
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                temperature=0.3,
            ),
            timeout=timeout_s,
        )
    )
    elapsed = time.monotonic() - start

    if response.finish_reason == "error":
        raise RuntimeError(response.content or "provider returned an error")

    usage = response.usage or {}
    tokens = usage.get("total_tokens") or usage.get("completion_tokens")
    return (response.content or "").strip(), tokens, elapsed


def print_probe_troubleshooting(provider: str | None) -> None:
    """Common-case hints when a probe fails.

    Shared by ``onboard`` Step 3 and ``doctor --probe`` so the diagnostic
    advice stays in one place.
    """
    console.print("\n  [dim]Troubleshooting:[/dim]")
    if provider:
        console.print(
            f"  [dim]·[/dim] [cyan]ddeharness provider test {provider}[/cyan] — re-check credentials without spending tokens"
        )
        console.print(
            f"  [dim]·[/dim] [cyan]ddeharness provider get {provider}[/cyan] — inspect what's actually stored on disk"
        )
    console.print(
        "  [dim]·[/dim] Check the model id in [cyan]~/.opendde_harness/config.json[/cyan] "
        "under [cyan]agents.defaults.model[/cyan] — it should match a model the "
        "provider serves."
    )


def load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from opendde_harness.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        Console(stderr=True).print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


__all__ = [
    "DEFAULT_PROBE_MESSAGE",
    "make_provider",
    "send_probe",
    "print_probe_troubleshooting",
    "load_runtime_config",
]
