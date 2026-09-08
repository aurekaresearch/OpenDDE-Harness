"""Task lease heartbeat and endpoint re-resolution for the on-demand local compute container."""

import asyncio

import httpx
import pytest

from opendde_harness.plugin.protein_design.core.contracts import WorkflowConfig
from opendde_harness.plugin.protein_design.core.detached import TaskLeaseHeartbeat
from opendde_harness.plugin.protein_design.servers import compute_pool
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient, ProteinDesignComputeError
from opendde_harness.plugin.protein_design.servers.local_service import ComputeEndpoint


class FakeComputeServer:
    """In-process compute service: records lease calls and refuses connections to retired ports."""

    def __init__(self, live_port: int) -> None:
        self.live_port = live_port
        self.requests: list[tuple[str, str, int, str | None]] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.port != self.live_port:
            raise httpx.ConnectError("connection refused", request=request)
        self.requests.append((request.method, request.url.path, request.url.port, request.headers.get("Authorization")))
        if request.url.path.startswith("/leases/"):
            return httpx.Response(204)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "workers": {"queue": {"running": 0, "queued": 0}}})
        return httpx.Response(200, json={"candidates": [], "port": request.url.port})


@pytest.mark.asyncio
async def test_heartbeat_refreshes_until_exit_and_then_releases():
    server = FakeComputeServer(live_port=18201)
    client = ProteinDesignComputeClient("http://127.0.0.1:18201", transport=server.transport, token="tok")
    async with TaskLeaseHeartbeat(client, "task-1", interval=0.01):
        await asyncio.sleep(0.05)
    refreshes = [item for item in server.requests if item[0] == "PUT"]
    assert len(refreshes) >= 2 and refreshes[0] == ("PUT", "/leases/task-1", 18201, "Bearer tok")
    assert server.requests[-1] == ("DELETE", "/leases/task-1", 18201, "Bearer tok")
    await client.close()


@pytest.mark.asyncio
async def test_heartbeat_failures_never_interrupt_the_task():
    server = FakeComputeServer(live_port=18202)
    client = ProteinDesignComputeClient("http://127.0.0.1:1", transport=server.transport)
    async with TaskLeaseHeartbeat(client, "task-2", interval=0.01):
        await asyncio.sleep(0.03)
    assert server.requests == []
    await client.close()


@pytest.mark.asyncio
async def test_pool_client_re_resolves_once_after_a_refused_connection(monkeypatch):
    server = FakeComputeServer(live_port=18204)
    resolved = []

    def resolver():
        resolved.append(True)
        return ComputeEndpoint("http://127.0.0.1:18204", "new-token")

    client = compute_pool.ReconnectingComputeClient(
        "http://127.0.0.1:18203", resolver=resolver, transport=server.transport, token="old-token",
    )
    payload = await client.top_candidates(1)
    assert payload["port"] == 18204 and resolved == [True]
    assert server.requests[-1][3] == "Bearer new-token"
    await client.population()
    assert resolved == [True]
    await client.close()


@pytest.mark.asyncio
async def test_pool_client_gives_up_when_the_resolved_endpoint_also_refuses():
    server = FakeComputeServer(live_port=18206)
    client = compute_pool.ReconnectingComputeClient(
        "http://127.0.0.1:18205", resolver=lambda: ComputeEndpoint("http://127.0.0.1:18207", None), transport=server.transport,
    )
    with pytest.raises(ProteinDesignComputeError, match="ConnectError"):
        await client.population()
    await client.close()


@pytest.mark.asyncio
async def test_local_placement_pool_uses_reconnecting_client_and_drops_stale_loopback_binding(monkeypatch):
    server = FakeComputeServer(live_port=18208)
    monkeypatch.setattr(
        compute_pool, "ensure_compute_service", lambda config: ComputeEndpoint("http://127.0.0.1:18208", "tok"),
    )
    reconnecting = compute_pool.ReconnectingComputeClient
    monkeypatch.setattr(
        compute_pool, "ReconnectingComputeClient",
        lambda url, **kwargs: reconnecting(url, transport=server.transport, **kwargs),
    )
    pool = compute_pool.ComputePool.from_config({"compute_docker": {"image": "img"}, "compute_token": "tok", "compute_url": "http://127.0.0.1:1"})
    workflow = WorkflowConfig(target="T", target_sequence="ACD", compute_url="http://127.0.0.1:1", compute_worker_id="127.0.0.1")
    selection = await pool.select(workflow)
    assert selection.worker.url == "http://127.0.0.1:18208"
    assert isinstance(selection.client, reconnecting)
    assert server.requests[0][:3] == ("GET", "/health", 18208)
    remote = compute_pool.ComputePool.from_config({"compute_url": "http://compute.example"})
    assert remote._resolver is None
