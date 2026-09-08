"""Structured progress events shared by protein-design surfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

_REDACTED = "[REDACTED]"
_SECRET_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "setcookie",
        "apikey",
        "token",
        "accesstoken",
        "refreshtoken",
        "password",
        "secret",
        "credential",
        "credentials",
    }
)


class ProgressStatus(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressEventType(StrEnum):
    TASK = "task"
    CYCLE = "cycle"
    PHASE = "phase"
    AGENT = "agent"
    SKILL = "skill"
    TOOL = "tool"
    FOLD = "fold"
    GATE = "gate"
    MEMORY = "memory"


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def redact_progress_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return redact_progress_payload(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = _normalized_key(key)
            is_secret = any(
                normalized == secret
                or normalized.endswith(secret)
                or secret in normalized
                for secret in _SECRET_KEYS
            )
            result[key] = _REDACTED if is_secret else redact_progress_payload(item)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_progress_payload(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class DesignProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: ProgressEventType
    status: ProgressStatus
    cycle: int | None = None
    total_cycles: int | None = None
    phase: str = ""
    actor: str = ""
    skill: str | None = None
    tool: str | None = None
    summary: str = ""
    duration_ms: float | None = None
    candidate_count: int | None = None
    input_payload: Any = None
    output_payload: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, **values: Any) -> "DesignProgressEvent":
        for name in ("input_payload", "output_payload", "metadata"):
            if name in values:
                values[name] = redact_progress_payload(values[name])
        return cls.model_validate(values)

    def compact_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude={"input_payload", "output_payload", "metadata"})
        payload["has_details"] = any(
            value not in (None, {}, [], "")
            for value in (self.input_payload, self.output_payload, self.metadata)
        )
        return payload


class ProgressSink(Protocol):
    def __call__(self, event: DesignProgressEvent) -> Any: ...


def emit_progress(
    sink: ProgressSink | Callable[[DesignProgressEvent], Any] | None,
    event: DesignProgressEvent,
) -> None:
    """Persist a redacted event to tracing and fan it out without failing design."""
    try:
        from opendde_harness.tracing import trace

        with trace.span(
            "protein_design.progress",
            {
                "protein_design.task_id": event.task_id,
                "protein_design.cycle_index": event.cycle,
                "protein_design.progress.phase": event.phase,
                "protein_design.progress.status": event.status.value,
                "protein_design.progress.type": event.event_type.value,
            },
            kind="protein_design",
        ) as span:
            span.artifact("protein_design.progress", event.model_dump(mode="json"))
    except Exception:
        pass
    if sink is not None:
        try:
            sink(event)
        except Exception:
            pass


__all__ = [
    "DesignProgressEvent",
    "ProgressEventType",
    "ProgressSink",
    "ProgressStatus",
    "emit_progress",
    "redact_progress_payload",
]
