import asyncio
import os
import threading
import time
from typing import Any

import pytest

from opendde_harness.plugin.protein_design.servers import api as api_module
from opendde_harness.plugin.protein_design.servers.leases import GpuLeaseTable


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _hold(table: GpuLeaseTable, kind: str, granted: list, release: asyncio.Event, **kwargs: Any) -> None:
    async with table.lease(kind, job_id=kwargs.pop("job_id", kind), **kwargs) as devices:
        granted.append(devices)
        await release.wait()


async def test_auto_allocation_prefers_the_least_loaded_gpu() -> None:
    table = GpuLeaseTable([0, 1], free_memory=lambda: {0: 1024, 1: 40960})

    async with table.lease("fold", job_id="first") as granted:
        assert granted == [1]
        async with table.lease("esm", job_id="second") as shared:
            assert shared == [0]


async def test_auto_placement_returns_a_job_kind_to_its_warm_gpu() -> None:
    memory = {0: 40960, 1: 1024}
    table = GpuLeaseTable([0, 1], free_memory=lambda: dict(memory))

    async with table.lease("fold", job_id="first") as granted:
        assert granted == [0]

    # The resident model now holds memory on GPU 0, so free memory alone would
    # push the next fold elsewhere.
    memory.update({0: 1024, 1: 40960})
    async with table.lease("fold", job_id="second") as granted:
        assert granted == [0]


async def test_a_fold_holds_its_gpus_exclusively() -> None:
    table = GpuLeaseTable([0, 1])
    folding = asyncio.Event()
    scoring = asyncio.Event()
    fold_devices: list = []
    esm_devices: list = []

    fold = asyncio.create_task(_hold(table, "fold", fold_devices, folding, count=2))
    await _settle()
    assert fold_devices == [[0, 1]]

    esm = asyncio.create_task(_hold(table, "esm", esm_devices, scoring))
    await _settle()
    assert esm_devices == []

    folding.set()
    await fold
    await _settle()
    assert esm_devices == [[0]]
    scoring.set()
    await esm


async def test_shared_jobs_share_one_gpu_up_to_the_configured_cap() -> None:
    table = GpuLeaseTable([0], shared_jobs_per_gpu=2)
    release = asyncio.Event()
    first: list = []
    second: list = []
    third: list = []

    running = [
        asyncio.create_task(_hold(table, "esm", first, release, job_id="esm-1")),
        asyncio.create_task(_hold(table, "mpnn", second, release, job_id="mpnn-1")),
    ]
    await _settle()
    assert first == [[0]] and second == [[0]]

    queued = asyncio.create_task(_hold(table, "esm", third, release, job_id="esm-2"))
    await _settle()
    assert third == []

    release.set()
    await asyncio.gather(*running, queued)
    assert third == [[0]]


async def test_an_explicit_request_for_a_busy_gpu_waits_instead_of_failing() -> None:
    table = GpuLeaseTable([0, 1])
    folding = asyncio.Event()
    scoring = asyncio.Event()
    fold_devices: list = []
    esm_devices: list = []

    fold = asyncio.create_task(_hold(table, "fold", fold_devices, folding, devices=[0]))
    await _settle()
    esm = asyncio.create_task(_hold(table, "esm", esm_devices, scoring, devices=[0]))
    await _settle()
    assert esm_devices == []

    folding.set()
    await fold
    await _settle()
    assert esm_devices == [[0]]
    scoring.set()
    await esm


async def test_a_placement_outside_the_inventory_is_rejected() -> None:
    table = GpuLeaseTable([0])

    with pytest.raises(ValueError, match="not available"):
        async with table.lease("fold", job_id="job", devices=[3]):
            pass


async def test_a_cpu_worker_still_runs_one_job_at_a_time() -> None:
    table = GpuLeaseTable([])
    release = asyncio.Event()
    first: list = []
    second: list = []

    running = asyncio.create_task(_hold(table, "fold", first, release, count=0))
    await _settle()
    assert first == [[]]

    queued = asyncio.create_task(_hold(table, "fold", second, release, count=0))
    await _settle()
    assert second == []

    release.set()
    await asyncio.gather(running, queued)
    assert second == [[]]


@pytest.mark.parametrize(
    "placement, options, expected",
    [
        (None, {}, (1, None)),
        (None, {"gpus": "none"}, (0, None)),
        (None, {"device": "cpu"}, (0, None)),
        (None, {"gpus": "1,2"}, (2, [1, 2])),
        ({"cp_degree": 2}, {}, (2, None)),
        ({"fold": [2, 3]}, {"gpus": "all"}, (2, [2, 3])),
    ],
)
def test_fold_gpu_requests_follow_the_placement_then_the_fold_options(placement, options, expected) -> None:
    from opendde_harness.plugin.protein_design.core.contracts import Placement

    requested = Placement.model_validate(placement) if placement is not None else None

    assert api_module.fold_gpu_request(requested, options) == expected


def test_leased_devices_reach_the_fold_esm_and_mpnn_payloads() -> None:
    fold = api_module.with_leased_devices("fold", {"options": {}}, [2, 3])
    assert fold["options"]["gpus"] == "2,3"
    assert fold["options"]["esm2_options"]["device"] == "cuda:2"

    pinned = api_module.with_leased_devices("fold", {"options": {"esm2_options": {"device": "cuda:0"}}}, [2])
    assert pinned["options"]["esm2_options"]["device"] == "cuda:0"

    mpnn = api_module.with_leased_devices("generate_soluble_mpnn", {"parameters": {}}, [1])
    assert mpnn["parameters"]["device"] == "cuda:1"

    esm = api_module.with_leased_devices("score_esm", {"options": {}}, [1])
    assert esm["options"]["device"] == "cuda:1"


def test_the_idle_watchdog_counts_only_time_without_work() -> None:
    now = [1000.0]
    busy = [False]
    watchdog = api_module.IdleWatchdog(timeout=600, busy=lambda: busy[0], clock=lambda: now[0])

    now[0] += 599
    assert not watchdog.expired()

    now[0] += 2
    assert watchdog.expired()

    busy[0] = True
    assert watchdog.seconds() == 0.0
    assert not watchdog.expired()

    busy[0] = False
    now[0] += 601
    assert watchdog.expired()


class _StubHarness:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((operation, payload))
        await asyncio.get_running_loop().run_in_executor(None, self.gate.wait)
        return {"candidates": []}

    async def health(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "ok", "models": {}, "workers": {"mode": "stub"}}

    def close(self) -> None:
        self.gate.set()


@pytest.fixture
def service(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("OPENDDE_HARNESS_PROTEIN_DESIGN_TOKEN", raising=False)
    monkeypatch.setattr(api_module, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(api_module, "free_gpu_memory", dict)
    harness = _StubHarness()
    exits: list[str] = []
    # ``create_app`` seeds process-wide model environment defaults.
    environment = dict(os.environ)
    try:
        app = api_module.create_app(harness, exit_process=lambda: exits.append("stopped"))
        with TestClient(app) as client:
            yield client, harness, exits
    finally:
        harness.gate.set()
        os.environ.clear()
        os.environ.update(environment)


def _await(predicate, *, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


FOLD_BODY = {"candidates": [{"candidate_id": "c1", "sequence": "MKV"}], "options": {}}


def test_health_reports_gpu_leases_and_the_idle_countdown(service) -> None:
    client, harness, _exits = service

    submission = client.post("/fold", json=FOLD_BODY).json()
    assert _await(lambda: client.get("/health").json()["workers"]["queue"]["running"] == 1)

    workers = client.get("/health").json()["workers"]
    assert workers["gpu_leases"][0]["jobs"] == [{"job_id": submission["job_id"], "kind": "fold"}]
    assert workers["gpu_leases"][1]["jobs"] == []
    assert workers["idle_seconds"] == 0
    assert workers["idle_timeout_seconds"] == api_module.DEFAULT_IDLE_SECONDS
    assert workers["queue"]["max_concurrent"] == 2

    harness.gate.set()


def test_shutdown_refuses_while_a_job_runs_and_drains_when_forced(service) -> None:
    client, harness, exits = service

    client.post("/fold", json=FOLD_BODY)
    assert _await(lambda: client.get("/health").json()["workers"]["queue"]["running"] == 1)

    busy = client.post("/shutdown", json={"if_idle": True})
    assert busy.status_code == 409
    assert busy.json() == {"detail": "busy", "running": 1, "queued": 0, "tasks": 0}
    assert exits == []

    draining = client.post("/shutdown", json={"if_idle": False})
    assert draining.status_code == 202
    assert client.post("/fold", json=FOLD_BODY).status_code == 503
    assert exits == []

    harness.gate.set()
    assert _await(lambda: exits == ["stopped"])


def test_shutdown_accepts_an_idle_worker(service) -> None:
    client, _harness, exits = service

    accepted = client.post("/shutdown", json={"if_idle": True})

    assert accepted.status_code == 202
    assert _await(lambda: exits == ["stopped"])


def test_a_task_lease_keeps_the_worker_busy_until_it_is_released(service) -> None:
    client, _harness, exits = service

    assert client.put("/leases/task-1").status_code == 204
    workers = client.get("/health").json()["workers"]
    assert workers["task_leases"][0]["task_id"] == "task-1"
    assert 0 < workers["task_leases"][0]["expires_in"] <= api_module.DEFAULT_TASK_LEASE_SECONDS
    assert workers["idle_seconds"] == 0

    busy = client.post("/shutdown", json={"if_idle": True})
    assert busy.status_code == 409
    assert busy.json() == {"detail": "busy", "running": 0, "queued": 0, "tasks": 1}
    assert exits == []

    assert client.delete("/leases/task-1").status_code == 204
    assert client.get("/health").json()["workers"]["task_leases"] == []
    assert client.post("/shutdown", json={"if_idle": True}).status_code == 202


def test_a_task_lease_expires_after_its_ttl_and_the_watchdog_sees_it() -> None:
    now = [0.0]
    leases = api_module.TaskLeases(ttl=180, clock=lambda: now[0])
    watchdog = api_module.IdleWatchdog(timeout=600, busy=lambda: bool(leases.active()), clock=lambda: now[0])

    for _ in range(7):
        leases.refresh("task-1")
        now[0] += 100
        assert not watchdog.expired()

    now[0] += 180
    assert leases.active() == []
    assert not watchdog.expired()
    now[0] += 600
    assert watchdog.expired()


def test_the_placement_of_a_fold_request_reaches_the_harness(service) -> None:
    client, harness, _exits = service

    client.post("/fold", json={**FOLD_BODY, "placement": {"fold": [1]}})
    assert _await(lambda: bool(harness.calls))

    operation, payload = harness.calls[0]
    assert operation == "fold"
    assert payload["options"]["gpus"] == "1"
    assert "placement" not in payload

    harness.gate.set()


def test_the_context_tool_reports_the_gpu_inventory_and_its_leases() -> None:
    import httpx

    from opendde_harness.plugin.protein_design.core.preparation import gpu_inventory

    health = {
        "status": "ok",
        "workers": {"gpu_leases": [{"index": 0, "jobs": [{"job_id": "j1", "kind": "fold"}]}, {"index": 1, "jobs": []}]},
        "gpu": [
            {"index": 0, "name": "NVIDIA H100", "memory_total_mb": 81559, "memory_free_mb": 1024},
            {"index": 1, "name": "NVIDIA H100", "memory_total_mb": 81559, "memory_free_mb": 81000},
        ],
    }
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=health))

    summary = gpu_inventory({"compute_url": "http://compute.invalid:8080"}, transport=transport)

    assert summary["available"] is True
    assert [gpu["index"] for gpu in summary["devices"]] == [0, 1]
    assert summary["devices"][0]["leases"] == [{"job_id": "j1", "kind": "fold"}]
    assert summary["devices"][1]["memory_free_mb"] == 81000


def test_an_unreachable_service_reports_no_gpu_inventory() -> None:
    import httpx

    from opendde_harness.plugin.protein_design.core.preparation import gpu_inventory

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    summary = gpu_inventory({"compute_url": "http://compute.invalid:8080"}, transport=httpx.MockTransport(refuse))

    assert summary["available"] is False
    assert summary["devices"] == []
    assert "did not answer" in summary["reason"]
