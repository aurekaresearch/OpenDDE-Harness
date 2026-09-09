"""The on-demand container of an earlier release is asked to leave when a new one starts."""

from opendde_harness.plugin.protein_design.servers import local_service


def test_a_previous_release_is_asked_to_exit_when_idle(monkeypatch):
    asked = []
    monkeypatch.setattr(
        local_service,
        "running_instance",
        lambda code_id=None: {"code_id": "old", "url": "http://127.0.0.1:1", "container": "c"},
    )
    monkeypatch.setattr(
        local_service, "request_shutdown", lambda url, token, *, if_idle, timeout=5.0: asked.append((url, if_idle))
    )

    local_service.retire_previous_release("tok", "new")
    local_service.retire_previous_release("tok", "old")

    assert asked == [("http://127.0.0.1:1", True)]
