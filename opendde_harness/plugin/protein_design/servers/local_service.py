"""On-demand local compute container: runtime state file, endpoint resolver and shutdown requests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_URL

DEFAULT_IDLE_SECONDS = 600


@dataclass(frozen=True)
class ComputeEndpoint:
    url: str
    token: str | None = None


def is_local_placement(config: Mapping[str, Any]) -> bool:
    return bool(config.get("compute_docker")) and not config.get("compute_workers")


def container_name(code_id: str) -> str:
    return f"opendde-compute-{code_id[:12]}"


def state_path() -> Path:
    home = Path(os.environ.get("OPENDDE_HARNESS_HOME", str(Path.home() / ".opendde_harness")))
    return home / "compute" / "local.json"


def read_state() -> dict[str, Any] | None:
    try:
        value = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not value.get("container") or not value.get("url"):
        return None
    return value


def write_state(state: Mapping[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(state), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clear_state() -> None:
    state_path().unlink(missing_ok=True)


def new_state(*, container: str, image: str, code_id: str, port: int) -> dict[str, Any]:
    return {
        "container": container,
        "image": image,
        "code_id": code_id,
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def service_health(url: str, token: str | None, *, timeout: float = 3.0) -> dict[str, Any] | None:
    """The ``/health`` payload of a service that answers, else ``None``."""
    try:
        with httpx.Client(timeout=timeout, trust_env=False, headers=_headers(token)) as client:
            response = client.get(f"{url}/health")
            if response.status_code != 200:
                return None
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def request_shutdown(url: str, token: str | None, *, if_idle: bool, timeout: float = 5.0) -> httpx.Response:
    with httpx.Client(timeout=timeout, trust_env=False, headers=_headers(token)) as client:
        return client.post(f"{url}/shutdown", json={"if_idle": if_idle})


def running_instance(code_id: str | None = None) -> dict[str, Any] | None:
    """The recorded instance while its container runs; only the given code release when ``code_id`` is set."""
    from opendde_harness.cli.onboard_compute import ComputeSetupError, inspect_container

    state = read_state()
    if state is None or (code_id and state.get("code_id") != code_id):
        return None
    try:
        container = inspect_container(str(state["container"]))
    except ComputeSetupError:
        return None
    if not container or not container.get("State", {}).get("Running"):
        return None
    return state


def wait_until_stopped(name: str, *, timeout: float = 60.0) -> bool:
    from opendde_harness.cli.onboard_compute import inspect_container

    deadline = time.monotonic() + timeout
    while True:
        container = inspect_container(name)
        if container is None or not container.get("State", {}).get("Running"):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)


def retire_previous_release(token: str | None, code_id: str) -> None:
    """Ask an earlier release's container to exit once idle, without waiting.

    A new release starts its own container; left alone, the old one holds
    its GPU memory until its idle timeout. Busy, it answers 409 and stays.
    """
    state = running_instance()
    if state is None or state.get("code_id") == code_id:
        return
    try:
        request_shutdown(str(state["url"]), token, if_idle=True, timeout=2.0)
    except httpx.HTTPError:
        pass


def stop_if_idle(token: str | None, *, timeout: float = 60.0) -> bool:
    """Stop the current release's container when idle; ``False`` when it is busy and stays up."""
    from opendde_harness.cli.compute_code import runtime_code_identity

    state = running_instance(runtime_code_identity()["id"])
    if state is None:
        return True
    try:
        response = request_shutdown(str(state["url"]), token, if_idle=True)
    except httpx.HTTPError:
        return True
    if response.status_code == 409:
        return False
    clear_state()
    wait_until_stopped(str(state["container"]), timeout=timeout)
    return True


def release_if_idle(*, timeout: float = 2.0) -> None:
    """Best-effort idle shutdown request for the TUI exit path; never raises."""
    state = read_state()
    if state is None:
        return
    from opendde_harness.cli.onboard_compute import load_protein_design_config

    token = str(load_protein_design_config().get("compute_token") or "") or None
    try:
        request_shutdown(str(state["url"]), token, if_idle=True, timeout=timeout)
    except httpx.HTTPError:
        pass


def instance_status(config: Mapping[str, Any]) -> dict[str, Any]:
    """Runtime state of the local container for ``doctor``: running or not, queue, idle countdown, GPU leases."""
    from opendde_harness.cli.compute_code import runtime_code_identity

    code_id = runtime_code_identity()["id"]
    state = running_instance()
    status: dict[str, Any] = {
        "running": state is not None,
        "container": str(state["container"]) if state else container_name(code_id),
        "port": state.get("port") if state else None,
        "url": state.get("url") if state else None,
        "code_id": str(state.get("code_id") or "") if state else code_id,
        "current_release": state.get("code_id") == code_id if state else True,
        "started_at": state.get("started_at") if state else None,
    }
    if state is None:
        return status
    health = service_health(str(state["url"]), str(config.get("compute_token") or "") or None)
    workers = (health or {}).get("workers") or {}
    queue = workers.get("queue") or {}
    status.update(
        {
            "healthy": health is not None,
            "jobs_running": queue.get("running"),
            "jobs_queued": queue.get("queued"),
            "idle_seconds": workers.get("idle_seconds"),
            "idle_timeout_seconds": workers.get("idle_timeout_seconds"),
            "gpu_leases": workers.get("gpu_leases") or [],
            "task_leases": workers.get("task_leases") or [],
        }
    )
    return status


def ensure_compute_service(config: Mapping[str, Any]) -> ComputeEndpoint:
    """The endpoint tasks talk to; for local placement, the running container or a freshly started one."""
    token = str(config.get("compute_token") or "") or None
    if not is_local_placement(config):
        return ComputeEndpoint(str(config.get("compute_url") or DEFAULT_COMPUTE_URL).rstrip("/"), token)

    import portalocker

    from opendde_harness.cli.compute_code import runtime_code_identity
    from opendde_harness.cli.onboard_compute import ComputeSetupError, DockerSettings, start_service

    if not token:
        raise ComputeSetupError("The local compute service has no saved token; run ddeharness onboard.")
    code_id = runtime_code_identity()["id"]
    lock = state_path().with_name("local.lock")
    lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One starter at a time: concurrent task starts must not race for the container name.
    with portalocker.Lock(str(lock), timeout=900):
        state = running_instance(code_id)
        if state is not None and service_health(str(state["url"]), token) is not None:
            return ComputeEndpoint(str(state["url"]), token)
        retire_previous_release(token, code_id)
        settings = DockerSettings.from_saved(config.get("compute_docker") or {})
        fold = config.get("fold_defaults") or {}
        state = start_service(settings, token, str(fold.get("api_url") or ""), code_id=code_id, quiet=True)
        return ComputeEndpoint(str(state["url"]), token)


__all__ = [
    "DEFAULT_IDLE_SECONDS",
    "ComputeEndpoint",
    "clear_state",
    "container_name",
    "ensure_compute_service",
    "instance_status",
    "is_local_placement",
    "new_state",
    "read_state",
    "release_if_idle",
    "request_shutdown",
    "retire_previous_release",
    "running_instance",
    "service_health",
    "state_path",
    "stop_if_idle",
    "wait_until_stopped",
    "write_state",
]
