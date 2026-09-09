import json
import re

import pytest
import typer
from typer.testing import CliRunner

from opendde_harness.cli import doctor_commands, onboard_commands, onboard_compute
from opendde_harness.cli.doctor_commands import DoctorReport, MemoryInfo, PathsInfo, RoutingInfo


@pytest.fixture
def app(monkeypatch, tmp_path):
    # Never the machine's own config: doctor reads it wherever it is not stubbed.
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    healthy = DoctorReport(
        config_loaded=True,
        paths=PathsInfo(config_path="/test/config.json", config_exists=True, config_valid=True),
        routing=RoutingInfo(model="test-model", provider="test-provider", max_tokens=1, context_window_tokens=None),
    )
    monkeypatch.setattr(doctor_commands, "_gather_static_checks", lambda: healthy)
    monkeypatch.setattr(doctor_commands, "_probe_memory", lambda _: MemoryInfo())
    monkeypatch.setattr(onboard_compute, "docker", lambda *a, **k: pytest.fail("doctor must not call Docker here"))
    app = typer.Typer()
    doctor_commands.register(app)
    return app


def _config(protein_design=None):
    return {"plugins": {"config": {"protein-design": protein_design}}} if protein_design is not None else {}


def test_bare_doctor_without_protein_design_config_exits_zero_and_skips_compute(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    monkeypatch.setattr(
        onboard_compute, "inspect_compute", lambda *a, **k: pytest.fail("compute must not be inspected")
    )
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "Configuration looks healthy" in result.output
    as_json = CliRunner().invoke(app, ["--json"])
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output)["compute"] is None


def test_configured_compute_is_inspected_and_failure_exits_two(app, monkeypatch):
    monkeypatch.setattr(
        onboard_commands, "_load_raw_config", lambda: _config({"compute_url": "http://127.0.0.1:18089"})
    )
    calls = []

    def inspect(config, **kwargs):
        calls.append((config, kwargs))
        return {
            "ready": False,
            "placement": "remote_service",
            "checks": [{"name": "service", "ok": False, "error": "refused"}],
        }

    monkeypatch.setattr(onboard_compute, "inspect_compute", inspect)
    result = CliRunner().invoke(app, ["--verify-hashes"])
    assert result.exit_code == 2, result.output
    assert calls[0][0]["compute_url"] == "http://127.0.0.1:18089"
    assert calls[0][1] == {"verify_hashes": True}
    assert "ddeharness compute prepare" in re.sub(r"\s+", " ", result.output)
    assert "refused" in result.output


def test_compute_only_reports_only_compute(app, monkeypatch):
    monkeypatch.setattr(
        onboard_commands, "_load_raw_config", lambda: _config({"compute_url": "http://127.0.0.1:18089"})
    )
    monkeypatch.setattr(
        onboard_compute, "inspect_compute", lambda *a, **k: {"ready": True, "checks": [{"name": "service", "ok": True}]}
    )
    result = CliRunner().invoke(app, ["--compute-only", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"ready": True, "checks": [{"name": "service", "ok": True}]}


def test_compute_only_without_configuration_exits_one(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    monkeypatch.setattr(
        onboard_compute, "inspect_compute", lambda *a, **k: pytest.fail("compute must not be inspected")
    )
    result = CliRunner().invoke(app, ["--compute-only"])
    assert result.exit_code == 1
    assert "ddeharness onboard" in result.output


def test_doctor_has_no_preparation_flags(app):
    for flag in ("--fix", "--assets-only", "--root", "--mode", "--checkpoint", "--code-only"):
        assert CliRunner().invoke(app, [flag]).exit_code == 2


def test_unknown_context_window_is_said_not_estimated(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    result = CliRunner().invoke(app, [])
    text = re.sub(r"\s+", " ", result.output)
    assert "Context win: unknown" in text
    assert "modelOverlay" in text
    assert "Max tokens: 1 (estimated)" in text


def test_resolved_context_window_names_its_source(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    report = doctor_commands._gather_static_checks()
    report.routing = RoutingInfo(
        model="deepseek/deepseek-v4-flash",
        provider="deepseek",
        max_tokens=384000,
        context_window_tokens=1000000,
        max_tokens_source="models.dev/provider",
        context_window_source="models.dev/provider",
    )
    result = CliRunner().invoke(app, [])
    text = re.sub(r"\s+", " ", result.output)
    assert "Context win: 1000000 (models.dev/provider)" in text
    assert "Max tokens: 384000 (models.dev/provider)" in text
    as_json = json.loads(CliRunner().invoke(app, ["--json"]).output)
    assert as_json["routing"]["context_window_source"] == "models.dev/provider"


def test_worker_pool_renders_each_worker_once_with_one_hint(app, monkeypatch):
    monkeypatch.setattr(
        onboard_commands, "_load_raw_config", lambda: _config({"compute_workers": [{"id": "a"}, {"id": "b"}]})
    )
    worker = {
        "ready": False,
        "placement": "remote_service",
        "checks": [{"name": "service", "ok": False, "error": "down"}],
    }
    monkeypatch.setattr(
        onboard_compute,
        "inspect_compute",
        lambda *a, **k: {
            "ready": False,
            "placement": "worker_pool",
            "checks": [],
            "workers": [{"id": "a", **worker}, {"id": "b", **worker}],
        },
    )
    result = CliRunner().invoke(app, ["--compute-only"])
    assert result.exit_code == 2
    assert result.output.count("Worker: ") == 2
    assert result.output.count("ddeharness onboard") == 1


def test_local_service_renders_queue_idle_countdown_and_gpu_leases(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config({"compute_docker": {"image": "img"}}))
    monkeypatch.setattr(
        onboard_compute,
        "inspect_compute",
        lambda *a, **k: {
            "ready": True,
            "placement": "local_docker",
            "fold_mode": "api",
            "device": "cuda",
            "checks": [
                {"name": "docker", "ok": True},
                {"name": "container", "ok": True},
                {"name": "service", "ok": True},
            ],
            "service": {
                "running": True,
                "healthy": True,
                "container": "opendde-compute-abcdef123456",
                "port": 18089,
                "code_id": "abcdef123456789",
                "current_release": True,
                "jobs_running": 1,
                "jobs_queued": 2,
                "idle_seconds": 30,
                "idle_timeout_seconds": 600,
                "gpu_leases": [{"index": 0, "job_id": "j1"}, "gpu1 free"],
            },
        },
    )
    result = CliRunner().invoke(app, ["--compute-only"])
    assert result.exit_code == 0, result.output
    text = re.sub(r"\s+", " ", result.output)
    assert "Service: running opendde-compute-abcdef123456 port 18089 code abcdef123456" in text
    assert "Jobs: 1 running, 2 queued (idle 30s of 600s)" in text
    assert "GPU lease: index=0, job_id=j1" in text and "GPU lease: gpu1 free" in text
    assert "previous release" not in text


def test_local_service_not_running_is_healthy_and_says_on_demand(app, monkeypatch):
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config({"compute_docker": {"image": "img"}}))
    monkeypatch.setattr(
        onboard_compute,
        "inspect_compute",
        lambda *a, **k: {
            "ready": True,
            "placement": "local_docker",
            "fold_mode": "api",
            "device": "cuda",
            "checks": [{"name": "container", "ok": True}],
            "service": {
                "running": False,
                "container": "opendde-compute-abcdef123456",
                "code_id": "abcdef123456789",
                "current_release": True,
            },
        },
    )
    result = CliRunner().invoke(app, ["--compute-only"])
    assert result.exit_code == 0, result.output
    text = re.sub(r"\s+", " ", result.output)
    assert "Service: not running (starts on demand as opendde-compute-abcdef123456)" in text
    assert "Jobs:" not in text and "ddeharness onboard" not in text


def test_memory_present_but_switched_off_is_reported(app, monkeypatch):
    """A managed memory root with no credentials answers zero hits forever, silently."""
    monkeypatch.setattr(onboard_commands, "_load_raw_config", lambda: _config())
    monkeypatch.setattr(
        doctor_commands,
        "_probe_memory",
        lambda _: MemoryInfo(root="/test/memory", disabled_reason="no credentials for llm"),
    )

    result = CliRunner().invoke(app, [])

    assert "Disabled" in result.output and "no credentials for llm" in result.output
    assert "recall returns nothing" in result.output
    # Not a fault: the user chose not to finish setting it up.
    assert result.exit_code == 0


def test_wire_facts_report_the_models_overlay_over_the_section(tmp_path, monkeypatch):
    from opendde_harness.config import loader
    from opendde_harness.config.schema import Config

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "agents": {"defaults": {"model": "custom/gpt-x", "provider": "custom"}},
                "providers": {
                    "custom": {
                        "apiKey": "k",
                        "apiBase": "http://relay/v1",
                        "wire": "chat",
                        "modelOverlay": {"gpt-x": {"wire": "responses"}},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    config = Config.model_validate(json.loads(path.read_text()))

    assert doctor_commands._wire_facts(config) == ("responses", "model overlay")


def test_a_check_note_is_rendered_beside_its_verdict(app, monkeypatch):
    monkeypatch.setattr(
        onboard_compute,
        "inspect_compute",
        lambda *a, **k: {
            "ready": True,
            "checks": [{"name": "runtime_code", "ok": True, "note": "prepared at first start from cached sources"}],
        },
    )
    monkeypatch.setattr(onboard_compute, "load_protein_design_config", lambda: {"compute_docker": {"x": 1}})

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0, result.output
    assert "OK runtime_code" in result.output and "prepared at first start" in result.output
