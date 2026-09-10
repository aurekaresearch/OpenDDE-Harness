"""The single output vocabulary: everything a turn can emit."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from opendde_harness.spine.message import Media, Source


@dataclass(frozen=True)
class Usage:
    """Token accounting for one turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class NoticeKind(StrEnum):
    """Out-of-band signals a turn surfaces to the user."""

    PROGRESS = "progress"
    TOOL_HINT = "tool_hint"
    INJECTED = "injected"
    DELIVERY_FAILED = "delivery_failed"


class ToolPhase(StrEnum):
    """When a tool event fires; outlets render the two phases differently."""

    START = "start"
    COMPLETE = "complete"


# Lifecycle events — emitted by the worker, never by a runner.


@dataclass(frozen=True)
class TurnStarted:
    """Marker that a turn began."""

    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnFailed:
    error: str
    cancelled: bool
    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnEnded:
    usage: Usage
    latency_ms: float
    explicit_reply: bool
    conversation_id: str | None = None


# Deliverable events — emitted by the runner, routed to outlets.


@dataclass(frozen=True)
class ToolEvent:
    phase: ToolPhase
    tool_call_id: str
    name: str = ""
    arguments: dict[str, Any] | None = None
    # Tool-authored call label for the start phase; None -> UI derives one.
    display: str | None = None
    result_preview: str = ""
    truncated: bool = False
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Text:
    content: str
    source: Source | None = None
    reply_to: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class MediaOut:
    media: tuple[Media, ...]
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class StreamDelta:
    delta: str
    stream_id: str | None = None
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Reasoning:
    content: str
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class Notice:
    kind: NoticeKind
    source: Source | None = None
    detail: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class EpisodeStart:
    """Boundary marker: a new model call (episode) begins. ``index`` is the
    0-based step within the turn. Outlets that group a turn into per-call
    episodes use it to start a fresh bucket; others ignore it."""

    index: int
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnRetry:
    """The model call is being run again. ``discard`` says output already
    streamed for this call is void: the re-run replaces it rather than
    appending to it, so an outlet showing live text starts that text over.
    ``attempt`` is the one about to run, out of ``total``."""

    attempt: int
    total: int
    reason: str
    discard: bool
    source: Source | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class TurnUsage:
    """What the model calls of this turn have produced so far, as the vendor
    counted it: cumulative output tokens, and the reasoning share where the
    vendor reports one. Sent after each call; a live display adds its own
    estimate of the call in flight on top."""

    completion_tokens: int
    reasoning_tokens: int
    calls: int
    source: Source | None = None
    conversation_id: str | None = None


RunnerEvent = ToolEvent | Text | MediaOut | StreamDelta | Reasoning | Notice | EpisodeStart | TurnRetry | TurnUsage
# Same union, named for its delivery role: what the hub routes and an Outlet renders.
Deliverable = RunnerEvent
TurnEvent = TurnStarted | TurnFailed | TurnEnded | RunnerEvent
