"""The behavioural seam between spine and the agent: how one turn runs.

spine defines the ``TurnRunner`` protocol and the agent loop implements it;
spine never imports the agent (dependency inversion). ``emit`` is narrowed to
``RunnerEvent`` so a runner cannot emit lifecycle events — the worker owns
those. With no static checker in this repo that narrowing is intent only; the
enforcing guard lives at the scheduler's emit boundary.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from opendde_harness.spine.events import RunnerEvent, Usage
from opendde_harness.spine.turn import TurnRequest

Emit = Callable[[RunnerEvent], Awaitable[None]]
# Read-and-remove this lane's pending injects, to merge at a tool-loop gap. The
# runner may ignore it (the minimal-legal implementation never drains, so every
# inject falls back to an APPEND turn). Synchronous: draining is a deque read.
Drain = Callable[[], list[TurnRequest]]


@dataclass(frozen=True)
class TurnOutcome:
    """What a finished run hands back; the worker fills TurnEnded from it."""

    usage: Usage
    explicit_reply: bool
    # Token/cost/context accounting for the turn, richer than ``usage``.
    usage_detail: dict[str, Any] = field(default_factory=dict)
    text: str | None = None


@runtime_checkable
class TurnRunner(Protocol):
    async def run(self, req: TurnRequest, emit: Emit, drain: Drain) -> TurnOutcome: ...
