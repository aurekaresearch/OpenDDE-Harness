"""Memory engine data carriers.

Phase B-3: the :class:`MemoryEngine` ABC + :class:`DefaultMemoryEngine`
facade have been deleted. The L4 indirection turned out to leak too
much surface (subsystem accessors that third-party plugins couldn't
satisfy); the host now talks to :class:`MemoryStore` /
:class:`MemoryConsolidator` / :class:`SkillService` directly and uses
the narrower :class:`MemoryBackend` Protocol
(:mod:`opendde_harness.memory_engine.backend`) as the plugin contract.

What remains in this file is the two data-carrier dataclasses that
:class:`ContextEngine.assemble` returns and consumes:

- :class:`AssembledContext` — the message list + metadata handed to
  AgentLoop for the LLM call.
- :class:`TokenBudget` — per-turn budget breakdown so the engine can
  decide what fits in the prompt.

These live here (rather than next to :class:`ContextEngine` itself)
for historical reasons — the rename ``opendde_harness.context_engine.types``
is a future tidy. Importers cited the old path heavily so we kept
the location stable through the Phase B cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssembledContext:
    """Output of a ``ContextEngine.assemble()`` call.

    The agent's LLM call uses exactly these messages. Nothing else from
    session history reaches the model directly.
    """

    messages: list[dict[str, Any]]
    system_prompt_addition: str | None = None  # Injected summary / working-state
    include_indices: list[int] | None = None  # Which session msg indices survived
    metadata: dict[str, Any] = field(default_factory=dict)  # Debug / telemetry


@dataclass
class TokenBudget:
    """Token budget breakdown for one turn.

    ``context_length`` and ``available_history`` are ``None`` when the model's
    window is unknown -- no table carries it and the user declared none. Every
    consumer checks for that explicitly rather than computing against a stand-in:
    the trimmer does not trim, the curator takes its fast path, and
    :attr:`threshold` reports no trigger. A number in their place used to be
    65,536 for every such model, which trimmed and compacted real windows at a
    fraction of their size.
    """

    context_length: int | None  # Model's context window; None when unknown
    reserved_output: int  # Reserved for completion
    reserved_tools: int  # Tool schemas + results in prompt
    reserved_system: int  # System prompt overhead
    available_history: int | None  # What's left for session history; None when unknown

    @property
    def total_reserved(self) -> int:
        return self.reserved_output + self.reserved_tools + self.reserved_system

    @property
    def known(self) -> bool:
        return self.available_history is not None

    @property
    def threshold(self) -> int | None:
        """Compaction trigger (75% of available_history by default); None when unknown."""
        if self.available_history is None:
            return None
        return int(self.available_history * 0.75)


__all__ = [
    "AssembledContext",
    "TokenBudget",
]
