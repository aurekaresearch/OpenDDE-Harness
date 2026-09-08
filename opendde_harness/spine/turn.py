"""The intent and ownership of a single request: the one input to submit()."""

from dataclasses import dataclass
from enum import StrEnum

from opendde_harness.spine.message import Media, Source


class Origin(StrEnum):
    """Who the request belongs to — drives pool, control eligibility, accounting."""

    USER = "user"
    SUBAGENT = "subagent"


class BusyPolicy(StrEnum):
    """What to do when the conversation's lane is already busy."""

    APPEND = "append"
    INJECT = "inject"
    INTERRUPT = "interrupt"


@dataclass(frozen=True)
class TurnRequest:
    """One request to process.

    ``message_id`` is the inbound message's own id — the default anchor an
    outbound reply threads back to (the outbound side carries it as the
    reply_to field on Text).
    """

    origin: Origin
    source: Source
    text: str
    media: tuple[Media, ...] = ()
    message_id: str | None = None
    conversation: str | None = None
    busy: BusyPolicy = BusyPolicy.APPEND
    # Verbatim delivery: when set, the turn skips the model entirely and emits
    # this text as-is (a background task pushing a finished result back to its
    # conversation). Runs through the lane so the session write stays serialized;
    # the model never sees it, so it cannot be rewritten. See AgentLoop.run_turn.
    deliver_text: str | None = None
