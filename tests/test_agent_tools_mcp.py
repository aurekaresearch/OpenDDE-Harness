"""One MCP server's transport failing must not take the agent turn down with it."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

import anyio
import mcp
import mcp.client.streamable_http
import pytest

from opendde_harness.agent.tools import mcp as mcp_tools
from opendde_harness.agent.tools.registry import ToolRegistry


def _config(url, transport="streamableHttp"):
    return SimpleNamespace(type=transport, url=url, command="", headers=None, tool_timeout=30)


async def test_a_failing_transport_is_this_servers_error_and_the_next_still_connects(monkeypatch):
    attempted = []

    @asynccontextmanager
    async def fake_streamable_http_client(url, http_client):
        attempted.append(url)
        if url == "https://bad.example/mcp":
            # The SDK's transports run anyio task groups; a failure inside one
            # unwound the enclosing task group and cancelled the turn.
            async with anyio.create_task_group() as group:

                async def fail_transport():
                    await anyio.sleep(0)
                    raise RuntimeError("transport failed")

                group.start_soon(fail_transport)
                yield url, object(), None
        else:
            yield url, object(), None

    class FakeSession:
        def __init__(self, read, write):
            self.read = read

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def initialize(self):
            if self.read == "https://bad.example/mcp":
                await asyncio.Event().wait()

        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name="ping", description="", inputSchema={"type": "object", "properties": {}})]
            )

    monkeypatch.setattr(mcp, "ClientSession", FakeSession)
    monkeypatch.setattr(mcp.client.streamable_http, "streamable_http_client", fake_streamable_http_client)

    registry = ToolRegistry()
    async with AsyncExitStack() as stack:
        await mcp_tools.connect_mcp_servers(
            {"bad": _config("https://bad.example/mcp"), "good": _config("https://good.example/mcp")}, registry, stack
        )

    assert attempted == ["https://bad.example/mcp", "https://good.example/mcp"]
    assert any(d["function"]["name"].endswith("ping") for d in registry.get_definitions())
    assert asyncio.current_task().cancelling() == 0


async def test_an_unknown_transport_is_skipped_before_any_connection(monkeypatch):
    attempted = False

    @asynccontextmanager
    async def fake_connection(name, cfg, transport_type, executor):
        nonlocal attempted
        attempted = True
        yield None, SimpleNamespace(tools=[])

    monkeypatch.setattr(mcp_tools, "_mcp_server_connection", fake_connection)

    await mcp_tools.connect_mcp_servers(
        {"svc": _config("https://x", transport="carrier-pigeon")}, ToolRegistry(), AsyncExitStack()
    )

    assert attempted is False


async def test_a_transport_failing_after_the_handshake_never_cancels_the_agent_task(monkeypatch):
    @asynccontextmanager
    async def failing_after_handshake(name, cfg, transport_type, executor):
        async with anyio.create_task_group() as group:

            async def drop_later():
                await anyio.sleep(0.05)
                raise RuntimeError("transport dropped")

            group.start_soon(drop_later)
            yield (
                object(),
                SimpleNamespace(
                    tools=[
                        SimpleNamespace(name="ping", description="", inputSchema={"type": "object", "properties": {}})
                    ]
                ),
            )

    monkeypatch.setattr(mcp_tools, "_mcp_transport", failing_after_handshake)

    registry = ToolRegistry()
    async with AsyncExitStack() as stack:
        await mcp_tools.connect_mcp_servers({"svc": _config("https://x")}, registry, stack)
        await asyncio.sleep(0.2)
        assert asyncio.current_task().cancelling() == 0

    assert any(d["function"]["name"].endswith("ping") for d in registry.get_definitions())
    assert asyncio.current_task().cancelling() == 0


async def test_cancelling_a_stuck_handshake_reaps_its_transport_task(monkeypatch):
    @asynccontextmanager
    async def never_ready(name, cfg, transport_type, executor):
        await asyncio.Event().wait()
        yield None, None

    monkeypatch.setattr(mcp_tools, "_mcp_transport", never_ready)

    async def connect():
        async with mcp_tools._mcp_server_connection("stuck", _config("https://x"), "streamableHttp", None):
            pass

    caller = asyncio.create_task(connect())
    await asyncio.sleep(0.05)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # Reaped before the caller returned, not merely asked to stop.
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mcp:stuck"]


async def test_a_cancellation_during_close_is_honoured_after_the_reap(monkeypatch):
    closed = asyncio.Event()

    @asynccontextmanager
    async def slow_to_close(name, cfg, transport_type, executor):
        try:
            yield object(), SimpleNamespace(tools=[])
        finally:
            await closed.wait()

    monkeypatch.setattr(mcp_tools, "_mcp_transport", slow_to_close)
    entered = asyncio.Event()
    after_close = []

    async def connect():
        async with mcp_tools._mcp_server_connection("slow", _config("https://x"), "streamableHttp", None):
            entered.set()
            await asyncio.Event().wait()
        after_close.append("ran past the close")

    caller = asyncio.create_task(connect())
    await entered.wait()
    caller.cancel()
    await asyncio.sleep(0.05)
    caller.cancel()  # a second cancellation lands while the close is waiting
    closed.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert after_close == []
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "mcp:slow"]
