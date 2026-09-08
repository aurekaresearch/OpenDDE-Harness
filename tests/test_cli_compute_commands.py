import pytest
import typer
from typer.testing import CliRunner

from opendde_harness.cli import compute_assets, compute_code, onboard_commands
from opendde_harness.cli.compute_commands import compute_app


@pytest.fixture
def app():
    app = typer.Typer()
    app.add_typer(compute_app, name="compute")
    return app


@pytest.fixture
def prepared(monkeypatch):
    calls = {}
    monkeypatch.setattr(compute_assets, "prepare", lambda *args, **kwargs: calls.setdefault("assets", (args, kwargs)))
    monkeypatch.setattr(compute_assets, "prepare_sources", lambda root: calls.setdefault("sources", root))
    monkeypatch.setattr(compute_code, "prepare_runtime_code", lambda cache, **kwargs: calls.setdefault("code", (cache, kwargs)))
    return calls


def test_prepare_uses_configured_mode_and_weights_root(app, prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {"plugins": {"config": {"protein-design": {
        "fold_defaults": {"execution_mode": "local"},
        "compute_docker": {"weights_dir": str(tmp_path), "opendde_checkpoint": str(tmp_path / "checkpoint/opendde_abag.pt")},
    }}}})
    result = CliRunner().invoke(app, ["compute", "prepare", "--download-workers", "3"])
    assert result.exit_code == 0, result.output
    (root, checkpoint, state_file), kwargs = prepared["assets"]
    assert root == tmp_path.resolve() and checkpoint == "opendde_abag.pt"
    assert state_file == compute_assets.asset_state_path()
    assert kwargs == {"download_workers": 3, "with_opendde": True, "opendde_root": compute_assets.opendde_root({})}
    assert prepared["code"] == (None, {"upstream_dir": None})
    prepared.clear()
    result = CliRunner().invoke(app, ["compute", "prepare", "--opendde-root", str(tmp_path / "data"), "--assets-only"])
    assert result.exit_code == 0, result.output
    assert prepared["assets"][1]["opendde_root"] == tmp_path / "data"


def test_prepare_defaults_to_api_mode_without_configuration(app, prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {})
    result = CliRunner().invoke(app, ["compute", "prepare", "--root", str(tmp_path), "--assets-only"])
    assert result.exit_code == 0, result.output
    (root, _, _), kwargs = prepared["assets"]
    assert root == tmp_path and kwargs["with_opendde"] is False
    assert "code" not in prepared


def test_checkpoint_is_reported_as_ignored_in_api_mode(app, prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {})
    result = CliRunner().invoke(app, ["compute", "prepare", "--root", str(tmp_path), "--checkpoint", "opendde_abag.pt"])
    assert result.exit_code == 0, result.output
    assert "--checkpoint is ignored in api mode" in result.output


def test_unknown_checkpoint_rejected_in_local_mode(app, prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {})
    result = CliRunner().invoke(app, ["compute", "prepare", "--root", str(tmp_path), "--mode", "local", "--checkpoint", "x.pt"])
    assert result.exit_code == 2
    assert "assets" not in prepared


def test_code_only_and_sources_only_skip_downloads(app, prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {})
    result = CliRunner().invoke(app, ["compute", "prepare", "--code-only", "--code-cache", str(tmp_path / "cache"), "--upstream-dir", str(tmp_path / "up")])
    assert result.exit_code == 0, result.output
    assert prepared == {"code": (tmp_path / "cache", {"upstream_dir": tmp_path / "up"})}
    prepared.clear()
    result = CliRunner().invoke(app, ["compute", "prepare", "--sources-only", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert prepared == {"sources": tmp_path}
    assert CliRunner().invoke(app, ["compute", "prepare", "--code-only", "--assets-only"]).exit_code == 2


def test_preparation_errors_exit_one_with_message(app, monkeypatch, tmp_path):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {})
    monkeypatch.setattr(compute_assets, "prepare", lambda *a, **k: (_ for _ in ()).throw(ValueError("curl is required")))
    result = CliRunner().invoke(app, ["compute", "prepare", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "curl is required" in result.output


def test_serve_rejects_unsupported_host(app, monkeypatch):
    from opendde_harness.cli import compute_environment

    monkeypatch.setattr(compute_environment.platform, "system", lambda: "Darwin")
    result = CliRunner().invoke(app, ["compute", "serve"])
    assert result.exit_code == 2
    assert "Linux x86-64" in result.output


def test_prepare_help_lists_no_doctor_flags(app):
    result = CliRunner().invoke(app, ["compute", "prepare", "--help"])
    assert result.exit_code == 0
    assert "--verify-hashes" not in result.output and "--fix" not in result.output
    assert "--sources-only" in result.output


def test_compute_help_lists_prepare_serve_and_stop(app):
    result = CliRunner().invoke(app, ["compute", "--help"])
    assert result.exit_code == 0
    for command in ("prepare", "serve", "stop"):
        assert f" {command} " in result.output or f" {command}\n" in result.output


@pytest.fixture
def local_state(monkeypatch):
    import httpx

    from opendde_harness.plugin.protein_design.servers import local_service

    calls = {"shutdown": [], "cleared": 0, "waited": []}
    state = {"container": "opendde-compute-abc", "url": "http://127.0.0.1:18089", "code_id": "abc", "port": 18089}
    monkeypatch.setattr(local_service, "running_instance", lambda code_id=None: state)
    monkeypatch.setattr(local_service, "wait_until_stopped", lambda name, **k: calls["waited"].append(name) or True)
    monkeypatch.setattr(local_service, "clear_state", lambda: calls.__setitem__("cleared", calls["cleared"] + 1))
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: {"plugins": {"config": {"protein-design": {"compute_token": "tok"}}}})

    def answer(status, payload=None):
        def request_shutdown(url, token, *, if_idle, timeout=5.0):
            calls["shutdown"].append((url, token, if_idle))
            return httpx.Response(status, json=payload or {})

        monkeypatch.setattr(local_service, "request_shutdown", request_shutdown)

    calls["answer"] = answer
    return calls


def test_stop_reports_when_nothing_is_running(app, local_state, monkeypatch):
    from opendde_harness.plugin.protein_design.servers import local_service

    monkeypatch.setattr(local_service, "running_instance", lambda code_id=None: None)
    result = CliRunner().invoke(app, ["compute", "stop"])
    assert result.exit_code == 0 and "not running" in result.output
    assert local_state["shutdown"] == []


def test_stop_requests_idle_shutdown_and_waits(app, local_state):
    local_state["answer"](202)
    result = CliRunner().invoke(app, ["compute", "stop"])
    assert result.exit_code == 0, result.output
    assert local_state["shutdown"] == [("http://127.0.0.1:18089", "tok", True)]
    assert local_state["cleared"] == 1 and local_state["waited"] == ["opendde-compute-abc"]
    assert "opendde-compute-abc stopped" in result.output


def test_stop_refuses_busy_service_unless_forced(app, local_state):
    local_state["answer"](409, {"detail": "busy", "running": 2, "queued": 1})
    result = CliRunner().invoke(app, ["compute", "stop"])
    assert result.exit_code == 1
    assert "2 running, 1 queued" in result.output and "--force" in result.output
    assert local_state["cleared"] == 0 and local_state["waited"] == []
    local_state["answer"](202)
    result = CliRunner().invoke(app, ["compute", "stop", "--force"])
    assert result.exit_code == 0, result.output
    assert local_state["shutdown"][-1] == ("http://127.0.0.1:18089", "tok", False)
    assert local_state["waited"] == ["opendde-compute-abc"]


def test_stop_reports_unreachable_service(app, local_state, monkeypatch):
    import httpx

    from opendde_harness.plugin.protein_design.servers import local_service

    def refuse(url, token, *, if_idle, timeout=5.0):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(local_service, "request_shutdown", refuse)
    result = CliRunner().invoke(app, ["compute", "stop"])
    assert result.exit_code == 1 and "did not answer" in result.output
