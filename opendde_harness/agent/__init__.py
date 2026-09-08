"""Agent core module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendde_harness.agent.context import ContextBuilder
    from opendde_harness.agent.loop import AgentLoop
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore"]

# Lazy re-exports (PEP 562): importing a ``opendde_harness.agent`` submodule must not
# eagerly construct ``AgentLoop`` -> litellm, which dominates CLI cold start.
_LAZY_EXPORTS = {
    "ContextBuilder": "opendde_harness.agent.context",
    "AgentLoop": "opendde_harness.agent.loop",
    "MemoryStore": "opendde_harness.memory_engine.consolidate.consolidator",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
