"""SubscriptionEmitter — per-session subscription registry + 16ms coalesce loop.

Per-session subscription registry with a coalesce loop:

- Per-session subscription map (`session_key → [Subscription]`)
- Each subscription owns a bounded asyncio.Queue (capacity 512)
- 16ms coalesce loop merges consecutive `token.delta` events into one frame
- A full queue briefly backpressures the producer instead of dropping a burst
- A persistently stalled queue emits error(code=-32016) and closes
- Non-token events pass through preserving order

Owned by the RPC server; passed to `register_turn_methods(dispatcher, emitter)`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from loguru import logger

COALESCE_WINDOW_S = 0.016
QUEUE_CAPACITY = 512
QUEUE_PUT_TIMEOUT_S = 1.0


@dataclass
class Subscription:
    sub_id: str
    session_key: str
    queue: asyncio.Queue
    coalesce_task: asyncio.Task | None = None
    closed: bool = False


SendFrame = Callable[[dict[str, Any]], Awaitable[None]]


class SubscriptionEmitter:
    """Routes TurnEvent notifications to per-session subscribers."""

    def __init__(self, send_frame: SendFrame) -> None:
        self._send_frame = send_frame
        self._by_session: dict[str, list[Subscription]] = {}
        self._by_id: dict[str, Subscription] = {}
        self._startup_events: list[dict[str, Any]] = []

    def queue_startup_event(self, event: dict[str, Any]) -> None:
        """Buffer an event produced before any subscription exists.

        Server bring-up (cron start, agent-loop build) precedes the client's
        ``turn.subscribe``, so an ``emit`` at that point is a silent no-op.
        Buffered events are flushed to the first subscription that registers,
        then dropped — they are one-shot startup notices, not a replay log.
        """
        self._startup_events.append(event)

    async def register(self, session_key: str) -> str:
        """Create a subscription, start its coalesce loop, return the sub_id.

        Any queued startup events are flushed to this subscription's session
        if it is the first to register (see ``queue_startup_event``).
        """
        sub = Subscription(
            sub_id=uuid4().hex,
            session_key=session_key,
            queue=asyncio.Queue(maxsize=QUEUE_CAPACITY),
        )
        sub.coalesce_task = asyncio.create_task(self._coalesce_loop(sub))
        self._by_session.setdefault(session_key, []).append(sub)
        self._by_id[sub.sub_id] = sub
        if self._startup_events:
            pending, self._startup_events = self._startup_events, []
            for event in pending:
                await self.emit(session_key, event)
        return sub.sub_id

    async def unregister(self, sub_id: str) -> bool:
        """Close the subscription if it exists and is open. Idempotent."""
        sub = self._by_id.get(sub_id)
        if sub is None or sub.closed:
            self._by_id.pop(sub_id, None)
            return False
        self._mark_closed(sub)
        return True

    async def emit(self, session_key: str, event: dict[str, Any]) -> None:
        """Push event to every open subscriber of session_key.

        A transient burst waits for the 16ms coalescer to drain instead of
        closing the subscription.  Only a queue that stays full for
        ``QUEUE_PUT_TIMEOUT_S`` is treated as a genuine stalled consumer.
        Other subscribers unaffected.
        """
        for sub in list(self._by_session.get(session_key, [])):
            if sub.closed:
                continue
            try:
                await asyncio.wait_for(sub.queue.put(event), QUEUE_PUT_TIMEOUT_S)
            except TimeoutError:
                logger.error(
                    "subscription queue stalled sub_id={} session={} event_type={} queued={}",
                    sub.sub_id,
                    sub.session_key,
                    event.get("type"),
                    sub.queue.qsize(),
                )
                await self._close_overflow(sub)

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Push one event to every currently subscribed TUI session."""
        for session_key in list(self._by_session):
            await self.emit(session_key, event)

    async def close_session(self, session_key: str) -> None:
        """Close all subscriptions belonging to session_key."""
        for sub in list(self._by_session.get(session_key, [])):
            await self.unregister(sub.sub_id)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _mark_closed(self, sub: Subscription) -> None:
        """Mark subscription closed, cancel its loop, drop from indexes."""
        sub.closed = True
        if sub.coalesce_task is not None and not sub.coalesce_task.done():
            sub.coalesce_task.cancel()
        self._by_id.pop(sub.sub_id, None)
        bucket = self._by_session.get(sub.session_key)
        if bucket is not None:
            bucket[:] = [s for s in bucket if s.sub_id != sub.sub_id]
            if not bucket:
                del self._by_session[sub.session_key]

    async def _coalesce_loop(self, sub: Subscription) -> None:
        """Per-subscription 16ms window coalesce loop.

        Each iteration:
          1. Block waiting for the first event.
          2. Sleep 16ms — accumulate any further events into a batch.
          3. Drain non-blockingly.
          4. Merge consecutive token.delta events; pass through others in order.
          5. Write each merged event as a JSON-RPC notification.
        """
        try:
            while not sub.closed:
                first = await sub.queue.get()
                batch: list[dict[str, Any]] = [first]
                await asyncio.sleep(COALESCE_WINDOW_S)
                while not sub.queue.empty():
                    batch.append(sub.queue.get_nowait())
                merged = _merge_consecutive_token_deltas(batch)
                for event in merged:
                    if sub.closed:
                        return
                    await self._send_frame(
                        {
                            "jsonrpc": "2.0",
                            "method": "event",
                            "params": {
                                "subscription_id": sub.sub_id,
                                "event": event,
                            },
                        }
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("subscription coalesce loop crashed sub_id={}", sub.sub_id)
            self._mark_closed(sub)

    async def _close_overflow(self, sub: Subscription) -> None:
        """Emit -32016 overflow notification, then close the subscription.

        Code -32016 from the extension range (-32016..-32049). Originally
        spec'd as -32010 in early drafts — collides with the live
        ConfigFieldReadonlyError.
        """
        if sub.closed:
            return
        try:
            await self._send_frame(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": {
                        "subscription_id": sub.sub_id,
                        "event": {
                            "type": "error",
                            "payload": {
                                "code": -32016,
                                "message": "subscription_capacity_exceeded",
                                "reason": "internal",
                            },
                        },
                    },
                }
            )
        except Exception:
            logger.exception(
                "failed to send overflow notification sub_id={}",
                sub.sub_id,
            )
        finally:
            self._mark_closed(sub)


def _merge_consecutive_token_deltas(
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse runs of `token.delta` events into a single merged frame.

    Non-delta events break the run and pass through unchanged. Preserves
    overall event ordering.
    """
    result: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for event in batch:
        event_type = event.get("type")
        if event_type in {"token.delta", "thinking.delta"}:
            if pending is None or pending.get("type") != event_type:
                if pending is not None:
                    result.append(pending)
                pending = {
                    "type": event_type,
                    "payload": {"text": event["payload"]["text"]},
                }
            else:
                pending["payload"]["text"] += event["payload"]["text"]
        else:
            if pending is not None:
                result.append(pending)
                pending = None
            result.append(event)
    if pending is not None:
        result.append(pending)
    return result


__all__ = [
    "COALESCE_WINDOW_S",
    "QUEUE_CAPACITY",
    "QUEUE_PUT_TIMEOUT_S",
    "SendFrame",
    "Subscription",
    "SubscriptionEmitter",
]
