"""The hub turns a discarding retry into a stream close, so an edit-in-place
outlet's partial message is finished and the re-run opens a new one."""

from opendde_harness.spine.delivery import Capabilities, DeliveryHub
from opendde_harness.spine.events import StreamDelta, TurnRetry
from opendde_harness.spine.message import ChatType, Source

SRC = Source(channel="chat", chat_id="c1", sender_id="u", chat_type=ChatType.DM)


class _Outlet:
    name = "chat"
    capabilities = Capabilities(streaming=True)

    def __init__(self):
        self.log: list[tuple] = []

    async def deliver(self, out):
        self.log.append(("deliver", type(out).__name__))

    async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False):
        self.log.append(("chunk", delta, done))


async def _run(events):
    hub = DeliveryHub()
    outlet = _Outlet()
    hub.register(outlet)
    for event in events:
        await hub.dispatch(event)
    await hub.wait_idle("chat")
    await hub.aclose()
    return outlet.log


async def test_a_discarding_retry_closes_the_stream_before_the_rerun_reopens_it():
    log = await _run(
        [
            StreamDelta(delta="par", source=SRC, conversation_id="conv"),
            TurnRetry(attempt=2, total=4, reason="network", discard=True, source=SRC, conversation_id="conv"),
            StreamDelta(delta="whole", source=SRC, conversation_id="conv"),
        ]
    )

    assert log == [("chunk", "par", False), ("chunk", "", True), ("deliver", "TurnRetry"), ("chunk", "whole", False)]


async def test_a_retry_before_any_output_closes_nothing():
    log = await _run(
        [
            TurnRetry(attempt=2, total=4, reason="network", discard=False, source=SRC, conversation_id="conv"),
            StreamDelta(delta="whole", source=SRC, conversation_id="conv"),
        ]
    )

    assert log == [("deliver", "TurnRetry"), ("chunk", "whole", False)]
