"""An earlier on-demand release must leave before a new GPU container starts."""

import httpx
import pytest

from opendde_harness.cli.onboard_compute import ComputeSetupError
from opendde_harness.plugin.protein_design.servers import local_service


def response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "http://127.0.0.1:1/shutdown"))


def test_a_previous_release_stops_before_replacement(monkeypatch):
    asked = []
    waited = []
    cleared = []
    monkeypatch.setattr(
        local_service,
        "running_instance",
        lambda code_id=None: {"code_id": "old", "url": "http://127.0.0.1:1", "container": "c"},
    )
    monkeypatch.setattr(
        local_service,
        "request_shutdown",
        lambda url, token, *, if_idle, timeout=5.0: (asked.append((url, if_idle)), response(202))[1],
    )
    monkeypatch.setattr(local_service, "wait_until_stopped", lambda name, *, timeout=60.0: waited.append(name) or True)
    monkeypatch.setattr(local_service, "clear_state", lambda: cleared.append(True))

    local_service.retire_previous_release("tok", "new")
    local_service.retire_previous_release("tok", "old")

    assert asked == [("http://127.0.0.1:1", True)]
    assert waited == ["c"]
    assert cleared == [True]


def test_a_busy_previous_release_is_preserved(monkeypatch):
    monkeypatch.setattr(
        local_service,
        "running_instance",
        lambda code_id=None: {"code_id": "old", "url": "http://127.0.0.1:1", "container": "c"},
    )
    monkeypatch.setattr(
        local_service,
        "request_shutdown",
        lambda url, token, *, if_idle, timeout=5.0: response(409),
    )
    monkeypatch.setattr(local_service, "wait_until_stopped", lambda *args, **kwargs: pytest.fail("busy worker waited"))
    monkeypatch.setattr(local_service, "clear_state", lambda: pytest.fail("busy worker state cleared"))

    with pytest.raises(ComputeSetupError, match="busy.*preserved"):
        local_service.retire_previous_release("tok", "new")


def test_an_unconfirmed_previous_release_is_preserved(monkeypatch):
    monkeypatch.setattr(
        local_service,
        "running_instance",
        lambda code_id=None: {"code_id": "old", "url": "http://127.0.0.1:1", "container": "c"},
    )
    monkeypatch.setattr(
        local_service,
        "request_shutdown",
        lambda url, token, *, if_idle, timeout=5.0: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )

    with pytest.raises(ComputeSetupError, match="could not confirm"):
        local_service.retire_previous_release("tok", "new")
