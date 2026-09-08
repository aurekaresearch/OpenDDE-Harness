"""Configuration status reported to the TUI (``setup.status``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from opendde_harness.plugin.active import active_registry

if TYPE_CHECKING:
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


def _config_path() -> Path:
    from opendde_harness.config.loader import get_config_path

    return get_config_path()


def _detect_provider_configured(payload: dict) -> bool:
    """Return True iff the loaded config payload indicates a usable provider.

    The onboarding gate's criterion ("required config complete"): at least one
    provider has an ``apiKey`` AND ``agents.defaults.model`` is set. Either
    alone can't drive a turn, so the UI must still park on the setup panel.
    An explicit non-``auto`` ``agents.defaults.provider`` also counts as a
    provider signal (legacy configs that pre-date per-provider sections).
    """
    if not isinstance(payload, dict):
        return False

    agents = payload.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    defaults = defaults if isinstance(defaults, dict) else {}

    model = defaults.get("model")
    if not (isinstance(model, str) and model):
        return False

    # `agents.defaults.provider` used to be waved through on its own, as a
    # provider signal from configs predating per-provider sections. It is now
    # written on every model change, so that branch would let a pinned name
    # stand for credentials nobody has -- the gate would pass with an empty
    # config. The name still says which section to ask about; whether it holds
    # anything is asked below, like every other provider.
    provider = defaults.get("provider")
    if isinstance(provider, str) and provider in {"minimax_global", "minimax_cn"}:
        from opendde_harness.providers.minimax_oauth import load_token

        return load_token("global" if provider == "minimax_global" else "cn") is not None

    providers = payload.get("providers")
    if isinstance(providers, dict):
        # `providers.auth`, like every other gate. Reading `apiKey` off the raw
        # payload made this the seventh rule and it disagreed with the other six
        # in both directions -- on the exact two configurations this module's
        # rewrite was filed to fix. A Gemini section holding only `apiKeyList`
        # parked a working install on the setup panel; Azure with a key and no
        # address was waved through into a chat that then could not run.
        from opendde_harness.config.schema import ProvidersConfig
        from opendde_harness.providers.auth import credential_status

        try:
            sections = ProvidersConfig.model_validate(providers)
        except Exception:
            sections = None
        if sections is not None:
            # Iterate the validated instance's own field names, not the raw
            # payload's keys: `canonical_provider_name` does not decompose
            # camelCase, so a camelCase key like "azureOpenai" -- the shape
            # `ProvidersConfig` serializes to -- fails to resolve back to the
            # `azure_openai` field it validated into, and `sections.get` on it
            # returns None. The declared fields are always snake_case, so
            # asking for those by name always resolves. Extra (unspecced)
            # sections keep their original payload spelling.
            names = set(type(sections).model_fields) | set(sections.model_extra or {})
            for name in names:
                section = sections.get(name)
                if section is not None and credential_status(name, section, include_external=True).ok:
                    return True

    from opendde_harness.providers.registry import split_model_id

    model_prefix, _ = split_model_id(model)
    if model_prefix in {"minimax_global", "minimax_cn"}:
        from opendde_harness.providers.minimax_oauth import load_token

        region = "global" if model_prefix == "minimax_global" else "cn"
        return load_token(region) is not None

    return False


async def setup_status(params: dict) -> dict:
    """Report missing or invalid configuration without claiming readiness."""
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("setup.status: configuration missing at {}", path)
        return {"provider_configured": False, "error": "Configuration is missing. Run ddeharness onboard."}
    except OSError as exc:
        logger.warning("setup.status: read failed for {}: {}", path, exc)
        return {"provider_configured": False, "error": "Configuration cannot be read. Run ddeharness doctor."}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("setup.status: invalid JSON in {}: {}", path, exc)
        return {"provider_configured": False, "error": "Configuration contains invalid JSON. Repair it before onboarding."}

    if not isinstance(payload, dict):
        return {"provider_configured": False, "error": "Configuration must be a JSON object."}
    plugins = payload.get("plugins") or {}
    configs = plugins.get("config") or {} if isinstance(plugins, dict) else {}
    if not isinstance(configs, dict):
        configs = {}
    registry = active_registry()
    compute_configured = True
    for plugin_id, ready in registry.readiness_checks():
        plugin_config = configs.get(plugin_id) or {}
        if not isinstance(plugin_config, dict):
            name = registry.manifest_for(plugin_id).display_name or plugin_id
            return {"provider_configured": False, "error": f"{name} configuration must be a JSON object."}
        compute_configured = compute_configured and ready(plugin_config)
    return {
        "provider_configured": _detect_provider_configured(payload),
        "compute_configured": compute_configured,
    }


def register_setup_methods(dispatcher: "Dispatcher") -> None:
    dispatcher.register("setup.status", setup_status)


__all__ = ["setup_status", "register_setup_methods"]
