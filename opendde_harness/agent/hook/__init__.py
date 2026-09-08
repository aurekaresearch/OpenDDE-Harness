"""AgentHook abstraction for AgentLoop lifecycle.

Public surface:

- :class:`AgentHook`        — base class (all methods default to no-op).
- :class:`AgentHookContext` — per-turn state carried through the chain.
- :class:`HookDecision`     — what a hook chose: pass-through,
                              short-circuit, or content modification.
- :class:`CompositeHook`    — aggregate multiple hooks into one,
                              with short-circuit + content-chain
                              semantics and exception isolation.
"""

from opendde_harness.agent.hook.base import AgentHook, AgentHookContext, HookDecision
from opendde_harness.agent.hook.composite import CompositeHook

__all__ = [
    "AgentHook",
    "AgentHookContext",
    "HookDecision",
    "CompositeHook",
]
