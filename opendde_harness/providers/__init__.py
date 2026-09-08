"""LLM provider abstraction module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendde_harness.providers.azure_openai_provider import AzureOpenAIProvider
    from opendde_harness.providers.base import LLMProvider, LLMResponse
    from opendde_harness.providers.litellm_provider import LiteLLMProvider
    from opendde_harness.providers.openai_codex_provider import OpenAICodexProvider

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "AzureOpenAIProvider"]

# Lazy re-exports (PEP 562): importing a provider submodule must not eagerly pull
# ``litellm_provider`` -> litellm, which dominates CLI cold start.
_LAZY_EXPORTS = {
    "LLMProvider": "opendde_harness.providers.base",
    "LLMResponse": "opendde_harness.providers.base",
    "LiteLLMProvider": "opendde_harness.providers.litellm_provider",
    "OpenAICodexProvider": "opendde_harness.providers.openai_codex_provider",
    "AzureOpenAIProvider": "opendde_harness.providers.azure_openai_provider",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
