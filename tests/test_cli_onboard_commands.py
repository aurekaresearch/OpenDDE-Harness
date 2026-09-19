import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from opendde_harness.cli.onboard_compute import (
    ComputeSetupError,
    DockerSettings,
    create_arguments,
    start_service,
    validate_assets,
)
from opendde_harness.plugin.protein_design.servers import local_service

CODE_ID = "a" * 64


@pytest.fixture
def settings(tmp_path):
    code = tmp_path / "code"
    weights = tmp_path / "weights"
    revision = "a" * 40
    for name in (
        "opendde_harness/plugin/protein_design/servers/api.py",
        "external/__init__.py",
        "external/opendde/runner/inference.py",
        "external/ligandmpnn/model_utils.py",
        "external/ligandmpnn/data_utils.py",
        "external/plip/plip/plipcmd.py",
    ):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    esm = weights / "huggingface/models--facebook--esm2_t33_650M_UR50D"
    (esm / "refs").mkdir(parents=True)
    (esm / "refs/main").write_text(revision)
    for name in (
        "checkpoint/opendde.pt",
        "common/components.cif",
        "common/components.cif.rdkit_mol.pkl",
        "common/obsolete_to_successor.json",
        "common/release_date_cache.json",
        "soluble_mpnn/solublempnn_v_48_020.pt",
        *(
            f"huggingface/{esm.name}/snapshots/{revision}/{file}"
            for file in (
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.txt",
            )
        ),
    ):
        path = weights / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
    return DockerSettings(
        image="example/runtime:test",
        mode="local",
        port=18089,
        gpus="0",
        package_root=str(code),
        state_dir=str(tmp_path / "state"),
        weights_dir=str(weights),
        opendde_data=str(weights),
        opendde_common=str(weights / "common"),
        opendde_checkpoint=str(weights / "checkpoint/opendde.pt"),
    )


@pytest.fixture
def harness_home(tmp_path, monkeypatch):
    home = tmp_path / "harness-home"
    monkeypatch.setenv("OPENDDE_HARNESS_HOME", str(home))
    return home


def test_code_and_weights_use_readonly_mounts(settings):
    validate_assets(settings)
    args, env = create_arguments(settings, "token", "", name="opendde-compute-test", port=18089)
    mounts = [args[i + 1] for i, item in enumerate(args[:-1]) if item == "--mount"]
    assert len(mounts) == 3
    assert f"type=bind,source={settings.package_root},target=/workspace,readonly" in mounts
    assert f"type=bind,source={settings.weights_dir},target=/weights,readonly" in mounts
    assert env["STRUCTPRED_OPENDDE_CHECKPOINT_PATH"] == "/weights/checkpoint/opendde.pt"
    assert env["STRUCTPRED_OPENDDE_CODE_DIR"] == "/workspace/external/opendde"
    assert env["OPENDDE_HARNESS_PROTEIN_SOLUBLE_MPNN_WEIGHTS_PATH"].startswith("/weights/")
    assert all(path.startswith("/workspace") for path in env["PYTHONPATH"].split(":"))
    assert args[args.index("--publish") + 1] == "127.0.0.1:18089:8080"


def test_container_is_ephemeral_with_idle_timeout(settings):
    args, env = create_arguments(settings, "token", "", name="opendde-compute-test", port=18089)
    assert args[:3] == ["run", "--detach", "--rm"]
    assert "--restart" not in args
    assert args[args.index("--name") + 1] == "opendde-compute-test"
    assert args[args.index("--gpus") + 1] == json.dumps("device=0")
    assert env["OPENDDE_HARNESS_COMPUTE_IDLE_SECONDS"] == "600"
    settings.idle_seconds = 90
    assert create_arguments(settings, "token", "", name="c", port=1)[1]["OPENDDE_HARNESS_COMPUTE_IDLE_SECONDS"] == "90"


def test_saved_settings_hold_no_dead_paths(settings):
    saved = settings.saved()
    assert {"hf_cache", "mpnn_weights", "opendde_code", "container_name"}.isdisjoint(saved)
    assert saved["weights_dir"] == settings.weights_dir
    settings.port = 0
    assert "port" not in settings.saved()


def test_saved_settings_ignore_retired_container_name(settings):
    saved = {**settings.saved(), "container_name": "opendde-protein-design-compute", "port": "18089"}
    rebuilt = DockerSettings.from_saved(saved)
    assert rebuilt == settings
    with pytest.raises(ComputeSetupError, match="incomplete or invalid"):
        DockerSettings.from_saved({"image": "x"})


@pytest.mark.parametrize(
    "relative",
    ["external/opendde/runner/inference.py", "external/ligandmpnn/model_utils.py", "external/plip/plip/plipcmd.py"],
)
def test_missing_source_is_rejected(settings, relative):
    (Path(settings.package_root) / relative).unlink()
    with pytest.raises(ComputeSetupError, match="missing, empty or unreadable"):
        validate_assets(settings)


def test_missing_shared_weights_rejected_in_wire(settings):
    settings.mode = "api"
    Path(settings.opendde_checkpoint).unlink()
    validate_assets(settings)
    (Path(settings.weights_dir) / "soluble_mpnn/solublempnn_v_48_020.pt").unlink()
    with pytest.raises(ComputeSetupError, match="missing, empty or unreadable"):
        validate_assets(settings)


def test_checkpoint_outside_opendde_data_rejected(settings, tmp_path):
    settings.opendde_checkpoint = str(tmp_path / "outside.pt")
    with pytest.raises(ComputeSetupError, match="inside the OpenDDE data directory"):
        create_arguments(settings, "token", "", name="c", port=18089)


def test_separate_opendde_data_root_gets_its_own_mount(settings, tmp_path):
    data = tmp_path / "opendde-data"
    settings.opendde_data = str(data)
    settings.opendde_common = str(data / "common")
    settings.opendde_checkpoint = str(data / "checkpoint/opendde_abag.pt")
    args, env = create_arguments(settings, "token", "", name="c", port=18089)
    mounts = [args[i + 1] for i, item in enumerate(args[:-1]) if item == "--mount"]
    assert len(mounts) == 4
    assert f"type=bind,source={data},target=/opendde,readonly" in mounts
    assert env["OPENDDE_ROOT_DIR"] == "/opendde" and env["STRUCTPRED_OPENDDE_ROOT_DIR"] == "/opendde"
    assert env["STRUCTPRED_OPENDDE_COMMON_DIR"] == "/opendde/common"
    assert env["STRUCTPRED_OPENDDE_CHECKPOINT_PATH"] == "/opendde/checkpoint/opendde_abag.pt"
    assert env["OPENDDE_HARNESS_WEIGHTS_DIR"] == "/weights"
    settings.mode = "api"
    args, env = create_arguments(settings, "token", "", name="c", port=18089)
    assert (
        sum(item == "--mount" for item in args) == 3
        and "STRUCTPRED_OPENDDE_COMMON_DIR" not in env
        or env["STRUCTPRED_OPENDDE_COMMON_DIR"].startswith("/weights")
    )


def test_invalid_hf_reference_rejected(settings):
    reference = Path(settings.weights_dir) / "huggingface/models--facebook--esm2_t33_650M_UR50D/refs/main"
    reference.write_text("../../outside")
    with pytest.raises(ComputeSetupError, match="Invalid ESM snapshot"):
        validate_assets(settings)


def test_missing_code_blocks_setup_before_docker(settings, monkeypatch):
    from opendde_harness.cli import onboard_compute

    settings.package_root = ""
    monkeypatch.setattr(onboard_compute, "check_local_docker", lambda: pytest.fail("Docker must not be called"))
    with pytest.raises(ComputeSetupError, match="specific file or directory"):
        start_service(settings, "token", code_id=CODE_ID)


def test_start_service_runs_container_and_records_state(settings, harness_home, monkeypatch):
    from opendde_harness.cli import onboard_compute

    calls = []
    monkeypatch.setattr(
        onboard_compute, "check_local_docker", lambda: calls.append("docker") or {"Runtimes": {"nvidia": {}}}
    )
    monkeypatch.setattr(onboard_compute, "ensure_image", lambda image, **k: calls.append("image"))
    monkeypatch.setattr(onboard_compute, "prepare_assets", lambda _: pytest.fail("start must not download assets"))
    monkeypatch.setattr(onboard_compute, "validate_settings", lambda *a, **k: calls.append("validate"))
    monkeypatch.setattr(onboard_compute, "inspect_container", lambda name: None)
    monkeypatch.setattr(onboard_compute, "docker", lambda *args, **k: calls.append(args[:3]) or "container-id")
    monkeypatch.setattr(onboard_compute, "wait_for_service", lambda *a, **k: calls.append("wait"))
    state = start_service(settings, "token", code_id=CODE_ID)
    assert state["url"] == "http://127.0.0.1:18089" and state["port"] == 18089
    assert state["container"] == "opendde-compute-" + "a" * 12 and state["code_id"] == CODE_ID
    assert calls == ["docker", "image", "validate", ("run", "--detach", "--rm"), "wait"]
    assert json.loads((harness_home / "compute/local.json").read_text()) == state
    assert local_service.read_state() == state


def test_state_file_round_trip(harness_home):
    assert local_service.read_state() is None
    state = local_service.new_state(container="opendde-compute-abc", image="img", code_id="abc", port=18090)
    local_service.write_state(state)
    assert local_service.state_path() == harness_home / "compute/local.json"
    assert local_service.read_state() == state and state["url"] == "http://127.0.0.1:18090"
    assert not list(harness_home.glob("compute/*.tmp"))
    local_service.write_state({"container": "x"})
    assert local_service.read_state() is None
    local_service.clear_state()
    local_service.clear_state()
    assert not local_service.state_path().exists()


def _running(name):
    return {"Id": name, "Name": "/" + name, "Image": "sha256:original", "State": {"Running": True}}


def test_resolver_reuses_running_healthy_container(harness_home, monkeypatch):
    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    monkeypatch.setattr(onboard_compute, "inspect_container", _running)
    monkeypatch.setattr(onboard_compute, "docker", lambda *a, **k: "sha256:original")
    monkeypatch.setattr(
        onboard_compute, "start_service", lambda *a, **k: pytest.fail("a healthy container must be reused")
    )
    monkeypatch.setattr(local_service, "service_health", lambda url, token, **k: {"status": "ok"})
    local_service.write_state(
        local_service.new_state(container="opendde-compute-" + "a" * 12, image="img", code_id=CODE_ID, port=18091)
    )
    endpoint = local_service.ensure_compute_service(
        {"compute_docker": {"image": "img"}, "compute_token": "tok", "compute_url": "http://127.0.0.1:1"}
    )
    assert endpoint == local_service.ComputeEndpoint("http://127.0.0.1:18091", "tok")


def test_resolver_does_not_adopt_a_reused_port_after_container_disappears(harness_home, monkeypatch):
    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    state = local_service.new_state(container="old", image="img", code_id=CODE_ID, port=18091)
    monkeypatch.setattr(local_service, "running_instance", lambda code_id=None: state if code_id else None)
    monkeypatch.setattr(onboard_compute, "inspect_container", lambda name: None)
    monkeypatch.setattr(
        local_service, "service_health", lambda *args, **kwargs: pytest.fail("unverified endpoint must not be reused")
    )
    monkeypatch.setattr(onboard_compute, "start_service", lambda *args, **kwargs: {"url": "http://127.0.0.1:18094"})

    endpoint = local_service.ensure_compute_service(
        {
            "compute_docker": {"image": "img", "mode": "api", "gpus": "all", "package_root": "", "state_dir": "/tmp/s"},
            "compute_token": "tok",
        }
    )

    assert endpoint.url == "http://127.0.0.1:18094"


@pytest.mark.parametrize("busy", [False, True])
def test_resolver_changed_image_replaces_only_an_idle_worker(harness_home, monkeypatch, busy):
    import httpx

    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    monkeypatch.setattr(onboard_compute, "inspect_container", _running)
    # Same image tag and same code release, but the tag now identifies a different environment.
    monkeypatch.setattr(onboard_compute, "docker", lambda *a, **k: "sha256:replacement")
    local_service.write_state(
        local_service.new_state(container="opendde-compute-" + "a" * 12, image="img", code_id=CODE_ID, port=18091)
    )
    stopped = []
    started = []

    def shutdown(url, token, *, if_idle, **kwargs):
        assert if_idle is True
        stopped.append(url)
        return httpx.Response(409 if busy else 202, request=httpx.Request("POST", url + "/shutdown"))

    monkeypatch.setattr(local_service, "request_shutdown", shutdown)
    monkeypatch.setattr(local_service, "wait_until_stopped", lambda name: not busy)
    monkeypatch.setattr(
        onboard_compute, "start_service", lambda *a, **k: started.append(a) or {"url": "http://127.0.0.1:18094"}
    )
    config = {
        "compute_docker": {"image": "img", "mode": "api", "gpus": "all", "package_root": "", "state_dir": "/tmp/s"},
        "compute_token": "tok",
    }
    if busy:
        with pytest.raises(ComputeSetupError, match="busy.*Active tasks were preserved"):
            local_service.ensure_compute_service(config)
        assert started == []
        assert local_service.read_state()["port"] == 18091
    else:
        assert local_service.ensure_compute_service(config).url == "http://127.0.0.1:18094"
        assert len(started) == 1
    assert stopped == ["http://127.0.0.1:18091"]


def test_resolver_stops_old_release_before_starting_new_one(harness_home, monkeypatch):
    import httpx

    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    monkeypatch.setattr(onboard_compute, "inspect_container", _running)
    monkeypatch.setattr(onboard_compute, "docker", lambda *args, **k: pytest.fail(f"unexpected docker {args}"))
    monkeypatch.setattr(
        local_service, "service_health", lambda url, token, **k: pytest.fail("old release must not be probed")
    )
    events = []
    monkeypatch.setattr(
        local_service,
        "request_shutdown",
        lambda url, token, *, if_idle, timeout=5.0: (
            events.append(("shutdown", url, if_idle)),
            httpx.Response(202, request=httpx.Request("POST", url + "/shutdown")),
        )[1],
    )
    monkeypatch.setattr(
        local_service,
        "wait_until_stopped",
        lambda name, *, timeout=60.0: events.append(("stopped", name)) or True,
    )
    old = local_service.new_state(container="opendde-compute-old", image="img", code_id="old-code", port=18092)
    local_service.write_state(old)
    started = []

    def start(settings, token, api_url, *, code_id, quiet):
        events.append(("start", code_id))
        started.append((settings, token, api_url, code_id, quiet))
        state = local_service.new_state(
            container=local_service.container_name(code_id), image=settings.image, code_id=code_id, port=18093
        )
        local_service.write_state(state)
        return state

    monkeypatch.setattr(onboard_compute, "start_service", start)
    config = {
        "compute_docker": {
            "image": "img",
            "mode": "api",
            "gpus": "all",
            "package_root": "",
            "state_dir": "/tmp/s",
            "container_name": "legacy",
        },
        "compute_token": "tok",
        "fold_defaults": {"execution_mode": "api", "api_url": "http://fold.test"},
    }
    endpoint = local_service.ensure_compute_service(config)
    assert endpoint == local_service.ComputeEndpoint("http://127.0.0.1:18093", "tok")
    assert len(started) == 1 and started[0][1:] == ("tok", "http://fold.test", CODE_ID, True)
    assert started[0][0].image == "img" and started[0][0].port == 0
    assert local_service.read_state()["container"] == "opendde-compute-" + "a" * 12
    assert events == [
        ("shutdown", "http://127.0.0.1:18092", True),
        ("stopped", "opendde-compute-old"),
        ("start", CODE_ID),
    ]


def test_resolver_restarts_dead_or_unhealthy_container(harness_home, monkeypatch):
    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    monkeypatch.setattr(onboard_compute, "inspect_container", lambda name: None)
    started = []
    monkeypatch.setattr(
        onboard_compute, "start_service", lambda *a, **k: started.append(a) or {"url": "http://127.0.0.1:18094"}
    )
    local_service.write_state(
        local_service.new_state(container="opendde-compute-" + "a" * 12, image="img", code_id=CODE_ID, port=18091)
    )
    config = {
        "compute_docker": {"image": "img", "mode": "api", "gpus": "all", "package_root": "", "state_dir": "/tmp/s"},
        "compute_token": "tok",
    }
    assert local_service.ensure_compute_service(config).url == "http://127.0.0.1:18094"
    assert len(started) == 1


def test_resolver_passes_remote_configuration_through(monkeypatch):
    from opendde_harness.cli import onboard_compute

    monkeypatch.setattr(
        onboard_compute, "docker", lambda *a, **k: pytest.fail("remote placement must not touch Docker")
    )
    endpoint = local_service.ensure_compute_service({"compute_url": "https://compute.example/", "compute_token": "tok"})
    assert endpoint == local_service.ComputeEndpoint("https://compute.example", "tok")
    assert local_service.ensure_compute_service({}).url == "http://127.0.0.1:8080"
    with pytest.raises(ComputeSetupError, match="no saved token"):
        local_service.ensure_compute_service({"compute_docker": {"image": "img"}})


def test_running_container_with_our_token_is_adopted(settings, harness_home, monkeypatch):
    from opendde_harness.cli import onboard_compute

    monkeypatch.setattr(onboard_compute, "check_local_docker", lambda: {"Runtimes": {"nvidia": {}}})
    monkeypatch.setattr(onboard_compute, "ensure_image", lambda image, **k: None)
    monkeypatch.setattr(onboard_compute, "validate_settings", lambda *a, **k: None)
    monkeypatch.setattr(onboard_compute, "wait_for_service", lambda *a, **k: None)
    container = {
        **_running("opendde-compute-" + "a" * 12),
        "Config": {"Labels": {onboard_compute.MANAGED_LABEL: "true"}, "Env": [f"{onboard_compute.AUTH_ENV}=token"]},
        "HostConfig": {"PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18095"}]}},
    }
    monkeypatch.setattr(onboard_compute, "inspect_container", lambda name: container)
    monkeypatch.setattr(onboard_compute, "docker", lambda *a, **k: "sha256:original")
    assert start_service(settings, "token", code_id=CODE_ID)["port"] == 18095
    container["Image"] = "sha256:another-image"
    with pytest.raises(ComputeSetupError, match="different image.*not adopted or stopped"):
        start_service(settings, "token", code_id=CODE_ID)
    container["Config"]["Env"] = [f"{onboard_compute.AUTH_ENV}=other"]
    with pytest.raises(ComputeSetupError, match="compute stop --force"):
        start_service(settings, "token", code_id=CODE_ID)


def test_cpu_container_does_not_request_gpu(settings, monkeypatch):
    from opendde_harness.cli import onboard_compute as compute

    settings.gpus = "none"
    monkeypatch.setattr(compute.shutil, "which", lambda _: None)
    compute.validate_settings(settings, {"Runtimes": {"runc": {}}}, require_image=False)
    args, env = compute.create_arguments(settings, "token", "", name="c", port=18089)
    assert "--gpus" not in args
    assert env["OPENDDE_HARNESS_COMPUTE_DEVICE"] == "cpu"
    assert env["OPENDDE_HARNESS_PROTEIN_FOLD_EXECUTION_MODE"] == "local"


def test_unsupported_host_fails_before_docker(monkeypatch):
    from opendde_harness.cli import compute_environment, onboard_compute

    monkeypatch.setattr(compute_environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(onboard_compute, "docker", lambda *_: pytest.fail("Docker must not run on unsupported hosts"))
    with pytest.raises(ComputeSetupError, match="Linux x86-64 only"):
        onboard_compute.check_local_docker()


def test_docker_error_retains_daemon_diagnostic_without_token(monkeypatch):
    import subprocess

    from opendde_harness.cli import onboard_compute as compute

    monkeypatch.setattr(compute.shutil, "which", lambda _: "/test/docker")
    monkeypatch.setattr(
        compute.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "Cannot connect to daemon; token=private-token"),
    )
    with pytest.raises(ComputeSetupError) as caught:
        compute.docker("info", "--format", "{{json .}}")
    assert "Cannot connect to daemon" in str(caught.value)
    assert "private-token" not in str(caught.value)


def test_image_pull_reports_layer_bytes_on_a_progress_bar(monkeypatch):

    from opendde_harness.cli import onboard_compute as compute

    output = [
        "v1: Pulling from aurekaresearch/opendde-harness\n",
        "a1b2c3d4e5f6: Downloading [==>    ]  1.2GB/4.5GB\n",
        "0123456789ab: Downloading [=====> ]  500MB/1GB\n",
        "a1b2c3d4e5f6: Pull complete\n",
        "0123456789ab: Extracting [==>    ]  200MB/1GB\n",
        "Status: Downloaded newer image for aurekaresearch/opendde-harness:v1\n",
    ]
    updates = []

    class _Stdout:
        def __init__(self, lines):
            self._lines = iter(lines)

        def __iter__(self):
            return self._lines

        def close(self):
            pass

    class _Process:
        returncode = 0
        stdout = _Stdout(output)

        def wait(self):
            return 0

    class _Bar:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def add_task(self, description, total=None):
            return 1

        def update(self, _task, **kwargs):
            updates.append(kwargs)

    monkeypatch.setattr(compute.shutil, "which", lambda _: "/test/docker")
    monkeypatch.setattr(compute, "docker", lambda *args: "" if args[:2] == ("image", "ls") else "{}")
    monkeypatch.setattr(compute.subprocess, "Popen", lambda *a, **k: _Process())
    monkeypatch.setattr("opendde_harness.cli._download.progress", lambda console: _Bar())

    compute.ensure_image("example/runtime:test")

    # Layer bytes accumulate across layers, and a completed layer counts in full.
    assert updates[-1]["total"] == 5_500_000_000
    assert updates[-1]["completed"] == 5_000_000_000
    assert "extracting 1/2 layers" in updates[-1]["description"]


def test_quiet_image_pull_stays_silent(monkeypatch):
    import subprocess

    from opendde_harness.cli import onboard_compute as compute

    calls = []
    monkeypatch.setattr(compute.shutil, "which", lambda _: "/test/docker")
    monkeypatch.setattr(compute, "docker", lambda *args: "" if args[:2] == ("image", "ls") else "{}")
    monkeypatch.setattr(
        compute.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or subprocess.CompletedProcess(command, 0),
    )

    compute.ensure_image("example/runtime:test", quiet=True)

    assert calls[0][0][-1] == "--quiet" and calls[0][1]["capture_output"] is True


def test_cpu_service_readiness_needs_no_gpu(monkeypatch):
    import httpx

    from opendde_harness.cli import onboard_compute as compute

    client = httpx.Client

    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "gpu": [],
                    "workers": {
                        "backend_ready": True,
                        "device": "cpu",
                        "tools": {"esm2": {"required": True, "ready": True}},
                    },
                },
            )
        return httpx.Response(200, json={"candidates": []})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(compute.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    compute.wait_for_service("http://compute.test", "token", "local", timeout=0)


def test_free_port_skips_ports_in_use():
    import socket

    from opendde_harness.cli.onboard_compute import free_port, port_is_free

    start = free_port(8080)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", start))
        taken.listen()
        assert not port_is_free(start)
        chosen = free_port(start)
    assert chosen > start and port_is_free(chosen)
    assert free_port(start) == start


@pytest.mark.parametrize(
    "value, valid", [("/tmp/weights", True), ("~/weights", True), ("/", False), ("", False), ("/a,b", False)]
)
def test_weights_directory_prompt_validation(value, valid):
    from opendde_harness.cli.onboard_compute import validate_directory

    assert (validate_directory(value) is True) == valid


def test_inspect_compute_uses_saved_weights_dir_and_names_prepare(tmp_path, monkeypatch):
    from opendde_harness.cli import compute_assets, onboard_compute

    roots = []

    def inspect(root, **kwargs):
        roots.append(root)
        return {"ready": False, "files": []}

    monkeypatch.setattr(compute_assets, "inspect_assets", inspect)
    monkeypatch.setattr(onboard_compute, "check_local_docker", lambda: None)
    monkeypatch.setattr(
        local_service, "instance_status", lambda config: {"running": False, "container": "opendde-compute-abc"}
    )
    config = {
        "compute_url": "http://127.0.0.1:18089",
        "compute_docker": {"weights_dir": str(tmp_path), "package_root": "", "gpus": "none"},
    }
    report = onboard_compute.inspect_compute(config)
    assert roots == [tmp_path.resolve()]
    errors = {item["name"]: item.get("error", "") for item in report["checks"]}
    assert "ddeharness compute prepare" in errors["required_assets"]
    assert "container_running" not in errors and errors["container"] == ""
    assert report["service"] == {"running": False, "container": "opendde-compute-abc"}
    assert report["device"] == "cpu" and report["ready"] is False


def test_inspect_compute_probes_the_running_local_container(monkeypatch):
    import httpx

    from opendde_harness.cli import compute_assets, onboard_compute

    monkeypatch.setattr(compute_assets, "inspect_assets", lambda root, **k: {"ready": True, "files": []})
    monkeypatch.setattr(onboard_compute, "check_local_docker", lambda: None)
    monkeypatch.setattr(
        local_service,
        "instance_status",
        lambda config: {"running": True, "url": "http://127.0.0.1:18096", "container": "c"},
    )
    probed = []

    def respond(request):
        probed.append(str(request.url))
        if request.url.path == "/health":
            return httpx.Response(
                200, json={"status": "ok", "workers": {"backend_ready": True, "device": "cuda", "tools": {"esm2": {}}}}
            )
        return httpx.Response(200, json={"candidates": []})

    client = httpx.Client
    monkeypatch.setattr(
        onboard_compute.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs)
    )
    report = onboard_compute.inspect_compute(
        {"compute_url": "http://127.0.0.1:1", "compute_docker": {"weights_dir": "/w", "package_root": "/p"}}
    )
    assert all(url.startswith("http://127.0.0.1:18096/") for url in probed) and probed
    assert report["device"] == "cuda"


class _FakePrompt:
    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


class _FakeQuestionary:
    """Records prompts into the wizard console so info lines and questions share one transcript."""

    class Choice:
        def __init__(self, title, value=None):
            self.title, self.value = title, value

    def __init__(self, console, answers):
        self._console, self._answers = console, answers

    def _prompt(self, message):
        self._console.print(f"? {message}")
        return _FakePrompt(self._answers.pop(0))

    def select(self, message, **_):
        return self._prompt(message)

    def text(self, message, **_):
        return self._prompt(message)

    def password(self, message, **_):
        return self._prompt(message)

    def confirm(self, message, **_):
        return self._prompt(message)


def _fake_row(console, answers, log=None):
    def row(message, options, *, default=None, scheme=None):
        console.print(f"? {message}")
        if log is not None:
            log.append((message, default))
        return answers.pop(0)

    return row


def test_custom_endpoint_models_read_bare_and_manual_entry_comes_first(monkeypatch):
    from opendde_harness.cli import onboard_commands as wizard

    seen = {}

    class _Questionary:
        Choice = _FakeQuestionary.Choice

        def select(self, message, *, choices, **_):
            seen["titles"] = [choice.title for choice in choices]
            return _FakePrompt(choices[1].value)

    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: _Questionary())

    chosen = wizard._select_model_id(["custom/gpt-4o", "custom/qwen3-32b"], provider="custom", manual_first=True)
    # The row reads as the vendor writes it; the id that gets stored keeps the
    # prefix that routes it to the endpoint the user configured.
    assert seen["titles"] == ["Enter a model name", "gpt-4o", "qwen3-32b"]
    assert chosen == "custom/gpt-4o"

    wizard._select_model_id(["deepseek/deepseek-v4-flash", "deepseek/deepseek-r1"])
    assert seen["titles"][:2] == ["deepseek/deepseek-v4-flash", "deepseek/deepseek-r1"]


def _offer(monkeypatch, wizard):
    """Capture what ``_pick_model`` puts in front of the user, and pick the first."""
    offered: dict = {}

    def select(choices, **_kwargs):
        offered["choices"] = list(choices)
        return choices[0]

    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_select_model_id", select)
    return offered


def _pick(wizard, provider):
    # The provider id and nothing else: there is no per-provider spec to hand
    # over any more -- what a provider serves is asked of the model layer.
    return wizard._pick_model(
        provider,
        current_model=None,
        model_ids=None,
        probe_status="skipped",
        user_provided_model=None,
        non_interactive=False,
    )


def test_the_wizards_suggestions_come_from_the_model_service(monkeypatch):
    """One source for the wizard and the picker, so they cannot offer different
    models for the same provider. The rows carry the bare id the endpoint
    serves, which is what the list shows; the id that gets *stored* is qualified
    on the way out, because its prefix is the only thing naming the provider."""
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.providers import common_models

    async def rows(_config):
        return [
            {"id": "deepseek-chat", "name": "DeepSeek Chat", "provider": "deepseek"},
            {"id": "gpt-5.5", "name": "GPT-5.5", "provider": "openai"},
        ]

    monkeypatch.setattr(common_models, "service_rows", rows)
    offered = _offer(monkeypatch, wizard)

    chosen = _pick(wizard, "deepseek")

    # Only this provider's rows, and nothing another provider serves.
    assert offered["choices"] == ["deepseek-chat"]
    assert chosen == "deepseek/deepseek-chat"


def test_the_wizard_falls_back_to_the_shortlist_when_the_service_is_silent(monkeypatch):
    """No Node, no bundle, a configuration the service refused: an empty prompt
    is worse than a curated list, and this is the only path that reaches it."""
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.providers import common_models

    async def refuse(_config):
        raise RuntimeError("no model service here")

    monkeypatch.setattr(common_models, "service_rows", refuse)
    offered = _offer(monkeypatch, wizard)

    _pick(wizard, "deepseek")

    assert "deepseek-v4-flash" in offered["choices"]


@pytest.fixture
def wizard_run(monkeypatch):
    import io

    from rich.console import Console

    from opendde_harness.cli import _choice as choice
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.cli import onboard_compute as compute
    from opendde_harness.cli import onboard_protein_design as step
    from opendde_harness.config import update

    buffer = io.StringIO()
    console = Console(file=buffer, width=400, force_terminal=False)
    answers = ["local", "api", True]
    outcome = {"saved": {}, "ensured": []}
    monkeypatch.setattr(wizard, "console", console)
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: _FakeQuestionary(console, answers))
    # A choice of two or three is a row rather than a list; it answers from the
    # same script and records itself into the same transcript.
    monkeypatch.setattr(choice, "row", _fake_row(console, answers, outcome.setdefault("rows", [])))
    monkeypatch.setattr(compute, "check_local_docker", lambda: {"Runtimes": {"nvidia": {}}})
    monkeypatch.setattr(compute, "default_image", lambda: "example/env:tag")
    monkeypatch.setattr(compute, "gpu_inventory", lambda: ["NVIDIA A800-SXM4-80GB"] * 2)
    monkeypatch.setattr(compute, "validate_settings", lambda *a, **k: None)
    monkeypatch.setattr(compute, "prepare_assets", lambda settings: outcome.setdefault("prepared", settings))
    monkeypatch.setattr(local_service, "stop_if_idle", lambda token, **k: True)
    monkeypatch.setattr(
        local_service,
        "ensure_compute_service",
        lambda config: (
            outcome["ensured"].append(config)
            or local_service.ComputeEndpoint("http://127.0.0.1:8080", config["compute_token"])
        ),
    )
    monkeypatch.setattr(update, "set_plugin_config_fields", lambda name, fields: outcome["saved"].update(fields))
    monkeypatch.delenv("OPENDDE_HARNESS_COMPUTE_IMAGE", raising=False)

    def run(config):
        monkeypatch.setattr(wizard, "_load_raw_config", lambda: config)
        step.configure_protein_design()
        assert answers == []
        return buffer.getvalue(), outcome

    return run


def test_local_docker_step_prints_lifecycle_then_folding_then_weights(wizard_run):
    from opendde_harness.cli import onboard_protein_design as step

    transcript, outcome = wizard_run({})
    expected = [
        "? Where should Protein Design run?",
        "Compute service",
        "Sequence design, scoring, folding, contact analysis and structure alignment, in a Docker container.",
        "Started when a task needs it and removed after 10 idle minutes",
        "Image   example/env:tag",
        "Device  2 × NVIDIA A800-SXM4-80GB (all visible; tasks may pin a device per tool, automatic by default)",
        "Folding",
        "? OpenDDE fold/refold mode:",
        f"Folding API  {step.DEFAULT_OPENDDE_API_URL}",
        "official default service",
        "Weights and code",
        "Harness tool weights (OPENDDE_HARNESS_WEIGHTS_DIR):",
        "  SolubleMPNN: soluble_mpnn/solublempnn_v_48_020.pt",
        "Compute service connected, settings saved.",
    ]
    positions = [transcript.find(line) for line in expected]
    assert all(index >= 0 for index in positions), transcript
    assert positions == sorted(positions), transcript
    # The wizard names what the operator set; the rest of the settings block is
    # this program's own bookkeeping and stays off the screen.
    assert "API port" not in transcript and "opendde-protein-design-compute" not in transcript
    assert "state_dir" not in transcript and "code_mode" not in transcript
    assert "? OpenDDE API URL" not in transcript and "Upstream API token" not in transcript
    assert "OpenDDE data (OPENDDE_ROOT_DIR)" not in transcript and "? Weights" not in transcript
    saved = outcome["saved"]
    assert step.DEFAULT_OPENDDE_API_URL == "https://api.aurekabio.cloud"
    assert saved["fold_defaults"] == {"execution_mode": "api", "api_url": step.DEFAULT_OPENDDE_API_URL}
    assert saved["compute_docker"]["gpus"] == "all" and saved["compute_docker"]["idle_seconds"] == 600
    assert {"port", "container_name"}.isdisjoint(saved["compute_docker"])
    assert saved["compute_url"] == "http://127.0.0.1:8080" and len(saved["compute_token"]) > 20
    assert outcome["prepared"].image == "example/env:tag"
    assert outcome["ensured"][0]["compute_docker"] == saved["compute_docker"]
    assert outcome["ensured"][0]["compute_token"] == saved["compute_token"]


def test_folding_opens_on_local_because_that_is_where_it_runs(wizard_run):
    """The compute container carries the folding weights, so local is the mode
    this runs in: nothing leaves the machine and no service has to answer. The
    row opens there, and a config that already names a mode keeps it."""
    _, outcome = wizard_run({})

    assert dict(outcome["rows"])["OpenDDE fold/refold mode:"] == "local"


def test_folding_keeps_the_mode_a_config_already_names(wizard_run):
    configured = {"plugins": {"config": {"protein-design": {"fold_defaults": {"execution_mode": "api"}}}}}

    _, outcome = wizard_run(configured)

    assert dict(outcome["rows"])["OpenDDE fold/refold mode:"] == "api"


def test_saved_port_override_is_shown_and_kept(wizard_run):
    config = {
        "plugins": {
            "config": {
                "protein-design": {
                    "compute_token": "existing-token",
                    "compute_docker": {"port": 18089, "gpus": "none", "idle_seconds": 120, "container_name": "legacy"},
                }
            }
        }
    }
    transcript, outcome = wizard_run(config)
    assert "API port" in transcript and "18089" in transcript
    assert "removed after 2 idle minutes" in transcript
    # The field list pads its keys to the widest one, so match the pair rather
    # than the spacing between them.
    assert re.search(r"Device +cpu", transcript)
    saved = outcome["saved"]["compute_docker"]
    assert saved["port"] == 18089 and saved["idle_seconds"] == 120 and "container_name" not in saved
    assert outcome["saved"]["compute_token"] == "existing-token"


def test_busy_container_keeps_running_and_settings_are_saved(wizard_run, monkeypatch):
    monkeypatch.setattr(local_service, "stop_if_idle", lambda token, **k: False)
    transcript, outcome = wizard_run({})
    assert "busy and keeps its settings" in transcript
    assert outcome["saved"]["compute_url"] == "http://127.0.0.1:8080"


def test_host_proxy_variables_reach_the_container_through_the_docker_host(settings, monkeypatch):
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:2080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5://user:pw@localhost:1080")

    args, env = create_arguments(settings, "token", "", name="c", port=18089)

    assert env["http_proxy"] == env["HTTP_PROXY"] == "http://host.docker.internal:2080"
    assert env["https_proxy"] == "socks5://user:pw@host.docker.internal:1080"
    assert set(env["no_proxy"].split(",")) == {
        "localhost",
        "127.0.0.1",
        "host.docker.internal",
        "api.aurekabio.cloud",
        "search-protrek.com",
        "protenix-server.com",
    }
    assert env["NO_PROXY"] == env["no_proxy"]
    assert "--add-host=host.docker.internal:host-gateway" in args


def test_public_service_bypass_preserves_existing_proxy_exclusions():
    from opendde_harness.cli.onboard_compute import proxy_environment

    env = proxy_environment({"HTTPS_PROXY": "http://127.0.0.1:17890", "NO_PROXY": "internal.example,::1"})

    assert env["HTTPS_PROXY"] == "http://host.docker.internal:17890"
    assert {"internal.example", "::1", "api.aurekabio.cloud", "search-protrek.com", "protenix-server.com"} <= set(
        env["NO_PROXY"].split(",")
    )
    assert env["NO_PROXY"] == env["no_proxy"]


def test_probe_failure_always_says_something(monkeypatch):
    """A timeout carries no message, so the setup line read "Test failed:" and stopped."""
    from opendde_harness.cli import onboard_commands as wizard

    monkeypatch.setattr(wizard, "_LANG", "en")

    assert "no reply" in wizard._probe_failure(TimeoutError())
    assert wizard._probe_failure(RuntimeError("")) == "RuntimeError"
    assert wizard._probe_failure(RuntimeError("upstream refused the key")) == "upstream refused the key"


@pytest.fixture
def wizard_config(tmp_path, monkeypatch):
    """A config file of this test's own, which every wizard write lands in.

    The wizard's writers ask ``get_config_path()`` rather than taking a path, so
    pinning it is what keeps a flow test off the real file -- and what lets the
    test read back the exact JSON the flow left behind.
    """
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    loader._cache.clear()
    return tmp_path / "config.json"


def _routes(path, model):
    """The config as the loader reads it back, having accepted what was written."""
    from opendde_harness.config.loader import load_config

    config = load_config(path)
    assert config.agents.defaults.model == model
    return config


def test_the_wizards_check_asks_the_model_layer_and_keeps_what_it_serves(tmp_path, monkeypatch):
    """The wizard verifies by asking for a few tokens, not by pinging /v1/models.

    The ping was free and answered a different question: seven of the vendors
    publish no such route -- the wizard had a whole branch for saying so and
    skipping -- Azure serves it at another address, and a 200 from it says
    nothing about whether the model will run. What comes back now is also the
    model list the next step offers as suggestions.
    """
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import loader as _loader
    from opendde_harness.config import update_providers as ops
    from tests._config import declared, write_config

    path = write_config(
        tmp_path / "config.json",
        declared("custom", base_url="https://relay/v1", models=["lab-1"], apiKey="k"),
        model="custom/lab-1",
    )
    monkeypatch.setattr(_loader, "_current_config_path", path)

    calls: list[str] = []

    def probe(name, **kwargs):
        calls.append(name)
        return {
            "ok": True,
            "status": "valid",
            "elapsed_ms": 3,
            "model": "lab-1",
            "models_count": 1,
            "model_ids": ["lab-1"],
            "error": None,
        }

    monkeypatch.setattr(ops, "test_provider", probe)
    ok, status, model_ids = wizard._verify_provider("custom")

    assert (ok, status, model_ids) == (True, "valid", ["lab-1"])
    assert calls == ["custom"]


def test_the_wizard_names_the_field_that_fixes_each_failure(tmp_path, monkeypatch, capsys):
    """The submenu's wording follows the status, and the statuses are the
    service's own words now -- ``auth`` where the old probe said ``invalid_key``."""
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import update_providers as ops

    def probe(name, **kwargs):
        return {
            "ok": False,
            "status": "auth",
            "elapsed_ms": 3,
            "model": "lab-1",
            "models_count": None,
            "model_ids": None,
            "error": "the relay refused the key",
        }

    monkeypatch.setattr(ops, "test_provider", probe)
    ok, status, _ids = wizard._verify_provider("custom")

    assert (ok, status) == (False, "auth")
    # Unwrapped: the console folds the line at the terminal's width.
    printed = " ".join(capsys.readouterr().out.split())
    assert "API key was refused" in printed
    assert "the relay refused the key" in printed


def test_a_near_miss_for_a_pi_id_is_answered_with_the_id_that_reaches_it():
    """The names people actually type, each answered with pi's own id for it.

    A near-miss is a dead end otherwise: it is not a built-in, so it reads as a
    provider this config declares, and is then refused for having no address --
    which is a true sentence about the wrong problem. "azure" and "chatgpt" are
    the two most likely things to type for the Azure resource and the Codex
    sign-in, which makes this the worst possible place to be unhelpful.
    """
    import typer

    from opendde_harness.cli.onboard_commands import _validate_provider_name

    for name, expected in (
        ("azure", "azure-openai-responses"),
        ("chatgpt", "openai-codex"),
        ("bedrock", "amazon-bedrock"),
        ("vertex_ai", "google-vertex"),
        ("claude", "anthropic"),
    ):
        with pytest.raises(typer.BadParameter) as caught:
            _validate_provider_name(name)
        assert expected in str(caught.value), name
        assert "Available providers" not in str(caught.value), name

    # A provider this project removed is answered with the removal, not with a
    # missing address: pi still carries the id, so silence here would accept it.
    with pytest.raises(typer.BadParameter) as caught:
        _validate_provider_name("copilot")
    assert "GitHub Copilot support was removed" in str(caught.value)

    # A name nobody has a table for is a declared provider waiting for an
    # address and a key: there is no catalogue of known vendors left to check a
    # spelling against, and inventing one to reject typos would put back exactly
    # what the model layer was deleted to remove.
    assert _validate_provider_name("nonsense_vendor") == "nonsense_vendor"
    # And an id carries no slash: the slash is what separates it from the model.
    with pytest.raises(typer.BadParameter):
        _validate_provider_name("my-vllm/qwen3-32b")


def test_step_one_is_unfinished_while_the_default_model_belongs_to_nobody_configured(tmp_path, monkeypatch):
    """Signing in outside the wizard must not let the fresh default model through.

    A new config ships `agents.defaults.model` set to a model Anthropic serves.
    Someone who ran `provider login openai-codex` first used to pass a startup
    gate that asked only whether some provider was configured and some model id
    was written -- and was told at the first request to buy an Anthropic key.
    """
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import loader as _loader
    from opendde_harness.config.schema import Config
    from tests._config import keyed, write_config

    # The fresh default model and no provider entry at all: a sign-in run
    # outside the wizard configures a provider, and not this model's.
    path = write_config(tmp_path / "config.json", {}, model=Config().agents.defaults.model)
    monkeypatch.setattr(_loader, "_current_config_path", path)
    monkeypatch.setattr(wizard, "_configured_providers", lambda: ["openai-codex"])

    # What the old gate asked, and why it let this through.
    assert wizard._configured_providers() and wizard._load_current_default_model()
    # What the startup gate asks now.
    assert wizard._is_config_populated() is False

    # And the pair that does work leaves the menu: one provider with a key, and
    # a default model whose prefix names that provider.
    write_config(path, keyed("deepseek", key="sk-deepseek-0000000000000000"), model="deepseek/deepseek-v4-flash")
    monkeypatch.setattr(wizard, "_configured_providers", lambda: ["deepseek"])
    assert wizard._is_config_populated() is True


def test_inspect_compute_treats_unprepared_managed_code_as_ready_when_it_can_start_offline(tmp_path, monkeypatch):
    from opendde_harness.cli import compute_assets, compute_code, onboard_compute

    monkeypatch.setattr(compute_assets, "inspect_assets", lambda root, **kwargs: {"ready": True, "files": []})
    monkeypatch.setattr(onboard_compute, "check_local_docker", lambda: None)
    monkeypatch.setattr(local_service, "instance_status", lambda config: {"running": False, "container": "c"})
    monkeypatch.setattr(
        compute_code, "managed_code_status", lambda: {"identity": {"id": "abc"}, "path": "/cache/x", "prepared": False}
    )
    config = {"compute_docker": {"weights_dir": str(tmp_path), "code_mode": "managed", "gpus": "none"}}

    report = onboard_compute.inspect_compute(config)

    check = next(item for item in report["checks"] if item["name"] == "runtime_code")
    assert check["ok"] is True and "first start" in check["note"]
    assert report["code"] == {"id": "abc"}


def test_a_managed_start_prepares_the_newly_installed_release_itself(settings, harness_home, monkeypatch):
    """After a package upgrade the next task start copies the new code; no manual prepare."""
    from opendde_harness.cli import compute_code, onboard_compute

    prepared = []
    code_root = settings.package_root
    monkeypatch.setattr(compute_code, "prepare_runtime_code", lambda: prepared.append("code") or Path(code_root))
    monkeypatch.setattr(
        onboard_compute, "validate_code_assets", lambda s: prepared.append(("validated", s.package_root))
    )
    monkeypatch.setattr(onboard_compute, "create_arguments", lambda *a, **k: (["run"], {}))
    monkeypatch.setattr(onboard_compute, "check_local_docker", lambda: {"Runtimes": {"nvidia": {}}})
    monkeypatch.setattr(onboard_compute, "ensure_image", lambda image, **k: None)
    monkeypatch.setattr(onboard_compute, "validate_settings", lambda *a, **k: None)
    monkeypatch.setattr(onboard_compute, "inspect_container", lambda name: None)
    monkeypatch.setattr(onboard_compute, "docker", lambda *args, **k: "container-id")
    monkeypatch.setattr(onboard_compute, "wait_for_service", lambda *a, **k: None)
    settings.code_mode = "managed"
    settings.package_root = ""

    start_service(settings, "token", code_id=CODE_ID)

    assert prepared == ["code", ("validated", code_root)]


def test_missing_shared_weights_rejected_in_api_mode(settings):
    settings.mode = "api"
    Path(settings.opendde_checkpoint).unlink()
    validate_assets(settings)
    (Path(settings.weights_dir) / "soluble_mpnn/solublempnn_v_48_020.pt").unlink()
    with pytest.raises(ComputeSetupError, match="missing, empty or unreadable"):
        validate_assets(settings)


class _KeyPrompt:
    """A questionary whose password prompt answers with one prepared value."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.labels: list[str] = []

    def password(self, label, **kwargs):
        self.labels.append(label)
        prompt = self

        class _Ask:
            def ask(self_inner):
                return prompt.answer

        return _Ask()


@pytest.mark.parametrize("answer, expected", [("", None), ("brave-key-123", "brave-key-123")])
def test_the_web_search_step_writes_a_key_or_leaves_duckduckgo(monkeypatch, answer, expected):
    import io

    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import update

    buffer = io.StringIO()
    written: list[str] = []
    monkeypatch.setattr(wizard, "console", wizard._ThemedConsole(file=buffer, width=200, force_terminal=False))
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: _KeyPrompt(answer))
    monkeypatch.setattr(wizard, "_verify_brave_key", lambda key: (True, ""))
    monkeypatch.setattr(update, "set_web_search_key", lambda key: written.append(key))
    warnings: list[str] = []

    result = wizard._step4_web_search(brave_api_key=None, non_interactive=False, warnings=warnings, skip_test=False)

    assert result is None and warnings == []
    assert written == ([expected] if expected else [])
    out = buffer.getvalue()
    assert "DuckDuckGo" in out and "2,000" in out
    assert ("Brave Search configured" in out) == bool(expected)


@pytest.mark.parametrize("non_interactive", [True, False])
def test_the_web_search_step_takes_the_flag_without_prompts(monkeypatch, non_interactive):
    """The flag is the key, with or without a terminal; ``skip_test`` alone decides verification."""
    import io

    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import update

    written: list[str] = []
    verified: list[str] = []
    monkeypatch.setattr(wizard, "console", wizard._ThemedConsole(file=io.StringIO(), width=200, force_terminal=False))
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(update, "set_web_search_key", lambda key: written.append(key))
    monkeypatch.setattr(wizard, "_verify_brave_key", lambda key: (verified.append(key), (True, ""))[1])
    monkeypatch.setattr(wizard, "_require_questionary", lambda: (_ for _ in ()).throw(AssertionError("no prompt")))

    wizard._step4_web_search(
        brave_api_key="brave-key-123", non_interactive=non_interactive, warnings=[], skip_test=True
    )
    wizard._step4_web_search(
        brave_api_key="brave-key-456", non_interactive=non_interactive, warnings=[], skip_test=False
    )
    wizard._step4_web_search(brave_api_key=None, non_interactive=True, warnings=[], skip_test=False)
    assert written == ["brave-key-123", "brave-key-456"] and verified == ["brave-key-456"]


def test_a_key_with_a_line_break_is_neither_sent_nor_saved(monkeypatch):
    import io

    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import update

    buffer = io.StringIO()
    monkeypatch.setattr(wizard, "console", wizard._ThemedConsole(file=buffer, width=200, force_terminal=False))
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: _KeyPrompt("brave\nSECRET"))
    monkeypatch.setattr(wizard, "_verify_brave_key", lambda key: (_ for _ in ()).throw(AssertionError("sent")))
    monkeypatch.setattr(update, "set_web_search_key", lambda key: (_ for _ in ()).throw(AssertionError("saved")))
    warnings: list[str] = []

    wizard._step4_web_search(brave_api_key=None, non_interactive=False, warnings=warnings, skip_test=False)
    assert warnings == ["Web search (Brave)"]
    assert "not saved" in buffer.getvalue() and "SECRET" not in buffer.getvalue()


def test_skipping_the_step_reports_the_key_that_stays(monkeypatch):
    import io

    from opendde_harness.cli import onboard_commands as wizard

    buffer = io.StringIO()
    monkeypatch.setattr(wizard, "console", wizard._ThemedConsole(file=buffer, width=200, force_terminal=False))
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: _KeyPrompt(""))
    monkeypatch.setattr(wizard, "_load_raw_config", lambda: {"tools": {"web": {"brave_api_key": "old"}}})
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    wizard._step4_web_search(brave_api_key=None, non_interactive=False, warnings=[], skip_test=True)
    assert "keeps its Brave key" in buffer.getvalue()


def test_a_rejected_key_is_written_with_a_warning(monkeypatch):
    import io

    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import update

    monkeypatch.setattr(wizard, "console", wizard._ThemedConsole(file=io.StringIO(), width=200, force_terminal=False))
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: _KeyPrompt("brave-key-123"))
    monkeypatch.setattr(wizard, "_verify_brave_key", lambda key: (False, "401 Unauthorized"))
    monkeypatch.setattr(update, "set_web_search_key", lambda key: None)
    warnings: list[str] = []

    wizard._step4_web_search(brave_api_key=None, non_interactive=False, warnings=warnings, skip_test=False)
    assert warnings == ["Web search (Brave)"]


def test_the_web_search_key_setter_writes_and_clears(tmp_path):
    import json

    from opendde_harness.config.update import set_web_search_key

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": {"web": {"jinaApiKey": "j"}}}))
    assert set_web_search_key("brave-key-123", config_path=path) is None
    assert json.loads(path.read_text())["tools"]["web"] == {"jinaApiKey": "j", "braveApiKey": "brave-key-123"}
    assert set_web_search_key("", config_path=path) == "brave-key-123"
    assert json.loads(path.read_text())["tools"]["web"] == {"jinaApiKey": "j"}


def test_the_web_search_key_setter_replaces_the_other_spelling(tmp_path):
    """The schema reads ``brave_api_key`` too; a second spelling beside it is
    a config the loader rejects, and clearing must clear that one as well."""
    import json

    from opendde_harness.config.update import set_web_search_key

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": {"web": {"brave_api_key": "old"}}}))
    assert set_web_search_key("new", config_path=path) == "old"
    assert json.loads(path.read_text())["tools"]["web"] == {"braveApiKey": "new"}
    path.write_text(json.dumps({"tools": {"web": {"brave_api_key": "old"}}}))
    assert set_web_search_key("", config_path=path) == "old"
    assert json.loads(path.read_text())["tools"]["web"] == {}


@pytest.mark.parametrize("hook,expected", [("/usr/bin/nvidia-container-runtime-hook", "all"), (None, "none")])
def test_onboard_gpu_default_with_runc(wizard_run, monkeypatch, hook, expected):
    from opendde_harness.cli import onboard_compute as compute

    monkeypatch.setattr(compute, "check_local_docker", lambda: {"Runtimes": {"runc": {}}})
    monkeypatch.setattr(compute.shutil, "which", lambda name: hook if name == "nvidia-container-runtime-hook" else None)
    _, outcome = wizard_run({})
    assert outcome["saved"]["compute_docker"]["gpus"] == expected


@pytest.mark.parametrize("saved", ["none", "2,3"])
def test_onboard_preserves_explicit_gpu_selection(wizard_run, monkeypatch, saved):
    from opendde_harness.cli import onboard_compute as compute

    monkeypatch.setattr(compute, "gpu_inventory", lambda: ["GPU"] * 4)
    _, outcome = wizard_run({"plugins": {"config": {"protein-design": {"compute_docker": {"gpus": saved}}}}})
    assert outcome["saved"]["compute_docker"]["gpus"] == saved


@pytest.fixture
def memory_step(monkeypatch, tmp_path):
    """Run the memory plugin's wizard step against a throwaway data directory.

    Everything that would touch the network or spawn the memory server is
    stubbed; what is left is the part under test -- what the step decides to
    write into ``config.json`` and into the memory root.
    """
    import io

    from rich.console import Console

    from opendde_harness.cli import onboard_commands as oc
    from opendde_harness.config import loader
    from opendde_harness.plugin.memory.longterm import _library, onboard
    from tests import _config

    config_path = tmp_path / "config.json"
    _config.write_config(config_path, {}, memory={"backend": None}, model="openai/gpt-5")
    monkeypatch.setattr(loader, "_current_config_path", config_path)
    loader._cache.clear()
    # The step exports the memory root; setenv so teardown puts it back.
    monkeypatch.setenv(_library.ROOT_ENV_VAR, str(tmp_path / "memory"))

    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    from opendde_harness.cli._theme import build_rich_theme, detect_scheme

    console.push_theme(build_rich_theme(detect_scheme()))
    monkeypatch.setattr(oc, "console", console)
    monkeypatch.setattr(oc, "_LANG", "en")
    monkeypatch.setattr(onboard, "choose_address", lambda: "http://localhost:18791")
    monkeypatch.setattr(onboard, "ensure_memory_server", _noop_async)

    def run(*, answers: list, skip: bool = False, non_interactive: bool = False):
        from opendde_harness.cli import _choice as choice

        scripted = list(answers)
        monkeypatch.setattr(oc, "_require_questionary", lambda: _FakeQuestionary(console, scripted))
        # The one question is a row of two, answered from the same script.
        monkeypatch.setattr(choice, "row", _fake_row(console, scripted))
        onboard.step(skip=skip, non_interactive=non_interactive, warnings=[])
        return json.loads(config_path.read_text()), console.file.getvalue()

    return run


async def _noop_async(*args, **kwargs):
    return None


def test_memory_step_asks_one_question_and_writes_only_the_service_address(memory_step, tmp_path):
    """Yes turns the backend on, records where the service listens, and writes
    the root's settings with placeholders in the model sections: memory runs on
    the default model, so there is no model, key or endpoint to ask for."""
    from opendde_harness.plugin.memory.longterm import settings

    saved, shown = memory_step(answers=[True])

    assert saved["memory"]["backend"] == "longterm"
    slice_ = saved["plugins"]["config"]["long-term-memory"]
    assert slice_ == {"base_url": "http://localhost:18791", "port": 18791}
    assert settings.memory_root() == tmp_path / "memory"
    assert settings.memory_ready()
    toml = settings.load_memory_config()
    assert toml["llm"]["model"] == settings.PLACEHOLDER_MODEL
    assert toml["multimodal"]["api_key"] == settings.PLACEHOLDER_KEY
    assert "embedding" not in {k for k, v in toml.items() if v.get("api_key")}
    assert shown.count("? ") == 1, "one question"
    assert "running on openai/gpt-5" in shown


def test_memory_step_off_leaves_the_backend_unset(memory_step, tmp_path):
    saved, _ = memory_step(answers=[False])
    assert saved["memory"]["backend"] is None
    assert not (tmp_path / "memory").exists()

    saved, _ = memory_step(answers=[], skip=True)
    assert saved["memory"]["backend"] is None


def test_memory_step_needs_no_input_so_non_interactive_turns_it_on(memory_step):
    saved, shown = memory_step(answers=[], non_interactive=True)
    assert saved["memory"]["backend"] == "longterm"
    assert "? " not in shown


def _quiet_wizard(monkeypatch):
    import io

    from opendde_harness.cli import onboard_commands as wizard

    monkeypatch.setattr(wizard, "console", wizard._ThemedConsole(file=io.StringIO(), width=200, force_terminal=False))
    monkeypatch.setattr(wizard, "_LANG", "en")
    return wizard


def test_onboard_moves_every_configuration_file_aside_and_starts_clean(tmp_path, monkeypatch):
    """A run over old files left behind whatever it did not touch -- a retired
    field, an old provider shape, a checkpoint a pre-release build chose --
    and the next load refused the whole file or a later step kept the value.
    The wizard now starts from nothing: the config, the compute record and
    the memory service's settings move into one backup directory, a memory
    server on the old settings is stopped first, and the workspace stays."""
    import json

    from opendde_harness.config import loader
    from opendde_harness.plugin.memory.longterm import _library, _server

    wizard = _quiet_wizard(monkeypatch)
    data = tmp_path / "home"
    (data / "compute").mkdir(parents=True)
    (data / "memory").mkdir()
    (data / "workspace").mkdir()
    (data / "config.json").write_text(json.dumps({"agents": {"defaults": {"model": "custom/x", "provider": "custom"}}}))
    (data / "compute" / "local.json").write_text('{"container": "opendde-compute-old"}')
    (data / "memory" / _library.CONFIG_FILENAME).write_text("[llm]\n")
    (data / "workspace" / "AGENTS.md").write_text("keep me")
    stopped: list[Path] = []
    monkeypatch.setattr(loader, "_current_config_path", data / "config.json")
    monkeypatch.setattr(_server, "stop_recorded_server", lambda root, **_: stopped.append(Path(root)))

    backup = wizard._retire_existing_setup()

    assert backup is not None and backup.parent == data and backup.name.startswith("backup-")
    assert sorted(p.relative_to(backup).as_posix() for p in backup.rglob("*") if p.is_file()) == [
        "compute/local.json",
        "config.json",
        f"memory/{_library.CONFIG_FILENAME}",
    ]
    assert "provider" in json.loads((backup / "config.json").read_text())["agents"]["defaults"]
    assert not (data / "config.json").exists() and not (data / "compute" / "local.json").exists()
    assert (data / "workspace" / "AGENTS.md").read_text() == "keep me", "the workspace is not configuration"
    assert stopped == [data / "memory"], "the memory server was stopped before its settings moved"
    assert wizard._retire_existing_setup() is None, "nothing to move the second time"


def test_the_probe_ends_the_model_service_inside_its_own_loop(monkeypatch):
    """``asyncio.run`` closes its loop on return; a service child left to the
    garbage collector had its pipe transport finalised after that and printed
    "Event loop is closed" under the wizard's next step."""
    from types import SimpleNamespace

    from opendde_harness.cli import _helpers
    from opendde_harness.providers import pi_service

    ended: list[bool] = []

    async def shutdown_service():
        ended.append(True)

    class _Provider:
        async def chat_with_retry(self, **_):
            return SimpleNamespace(finish_reason="stop", content="Hello there", usage={"total_tokens": 7})

    monkeypatch.setattr(_helpers, "load_config", lambda: object(), raising=False)
    monkeypatch.setattr("opendde_harness.config.loader.load_config", lambda: object())
    monkeypatch.setattr(_helpers, "make_provider", lambda config: _Provider())
    monkeypatch.setattr(pi_service, "shutdown_service", shutdown_service)

    text, tokens, _elapsed = _helpers.send_probe(message="hi", timeout_s=5)

    assert (text, tokens) == ("Hello there", 7)
    assert ended == [True], "the service was ended before the loop closed"


def test_a_container_that_exited_says_so_and_names_the_way_to_its_logs(monkeypatch):
    """A container that exits as it starts removes itself, logs and all, and the
    bare connection error that is left says nothing about why. Whether the
    container is still there is what separates "never came up" from "not
    answering yet", so that is what the message carries."""
    from opendde_harness.cli import onboard_compute as compute

    original = compute.ComputeSetupError("Compute readiness check timed out: ConnectError.")

    monkeypatch.setattr(compute, "inspect_container", lambda name: {"State": {"Running": False}})
    assert compute._readiness_failure(original, "c") is original

    monkeypatch.setattr(compute, "inspect_container", lambda name: None)
    gone = compute._readiness_failure(original, "opendde-compute-1")

    assert "exited as it started" in str(gone)
    assert compute.KEEP_ENV in str(gone) and "docker logs opendde-compute-1" in str(gone)


def test_the_keep_flag_is_what_leaves_a_container_behind(monkeypatch):
    from opendde_harness.cli import onboard_compute as compute

    monkeypatch.delenv(compute.KEEP_ENV, raising=False)
    assert compute.removal_flags() == ["--rm"]

    monkeypatch.setenv(compute.KEEP_ENV, "1")
    assert compute.removal_flags() == []

    monkeypatch.setenv(compute.KEEP_ENV, "  ")
    assert compute.removal_flags() == ["--rm"], "a blank value asks for nothing"


def test_what_the_compute_layers_say_lands_at_the_wizards_gutter(wizard_run, monkeypatch):
    """The compute layers are libraries that also run from a script and inside a
    container, so they write plain lines. During the wizard those lines sit
    beside everything else it has drawn instead of flush against the edge."""
    from opendde_harness.cli import _download
    from opendde_harness.cli import onboard_compute as compute

    monkeypatch.setattr(compute, "prepare_assets", lambda settings: _download.report("Runtime code ready: /tmp/code"))

    transcript, _ = wizard_run({})

    assert "  Runtime code ready: /tmp/code" in transcript
    assert "\nRuntime code ready" not in transcript
    # And the hook is the one it found, once the step is over.
    assert _download.report("") is None


# ---------------------------------------------------------------------------
# Step 1 -- pi's /login at the terminal, run against the gateway's own handlers
# ---------------------------------------------------------------------------


def _row(
    slug, name, *, auth_methods, auth_type, authenticated=False, key_env=None, needs_api_key=True, login_label=None
):
    """One provider row as ``model.options`` answers it, with only what the flow reads."""
    return {
        "slug": slug,
        "name": name,
        "auth_methods": list(auth_methods),
        "auth_type": auth_type,
        "authenticated": authenticated,
        "key_env": key_env,
        "needs_api_key": needs_api_key,
        "needs_base_url": False,
        "login_label": login_label,
        "key_label": None,
        "models": [],
        "warning": "",
    }


CODEX = _row("openai-codex", "OpenAI Codex", auth_methods=["oauth"], auth_type="oauth")
DEEPSEEK = _row("deepseek", "DeepSeek", auth_methods=["key"], auth_type="key")
XAI = _row(
    "xai", "xAI", auth_methods=["oauth", "key"], auth_type="key", login_label="Sign in with SuperGrok or X Premium"
)


class _Prompts:
    """Every prompt of step 1, answered from one script and written to one transcript.

    A list answer names the row to take by a piece of its title; a row answer
    names the option by a piece of its label. So a script reads as what a
    person would do: "Sign in with an API key", "DeepSeek", "sk-…".
    """

    class Choice:
        def __init__(self, title, value=None):
            self.title, self.value = title, value

    class Separator(Choice):
        def __init__(self, line="─"):
            super().__init__(line, value=None)

    def __init__(self, console, answers):
        self.console, self.answers = console, answers
        self.lists: list[list[str]] = []
        self.asked: list[str] = []

    def _next(self, message, instruction=""):
        self.console.print(f"? {message}" + (f"  ({instruction})" if instruction else ""))
        self.asked.append(message)
        return self.answers.pop(0)

    def select(self, message, *, choices, **_):
        self.lists.append([choice.title for choice in choices])
        wanted = self._next(message)
        prompt = self

        class _Ask:
            def ask(self_inner):
                return next(choice.value for choice in choices if wanted in choice.title)

        return _Ask()

    def text(self, message, *, instruction="", **_):
        answer = self._next(message, instruction)

        class _Ask:
            def ask(self_inner):
                return answer

        return _Ask()

    password = text

    def row(self, message, options, *, default=None, back=None, scheme=None):
        wanted = self._next(message)
        self.lists.append([label for label, _ in options])
        if wanted == "Back":
            return back
        return next(value for label, value in options if wanted in label)


@pytest.fixture
def step_one(monkeypatch):
    """The flow with its prompts scripted and its gateway faked, and the transcript."""
    import io

    from opendde_harness.cli import _choice as choice
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.tui_rpc.methods import model as gateway

    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=False)
    answers: list = []
    prompts = _Prompts(console, answers)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(wizard, "console", console)
    monkeypatch.setattr(wizard, "_LANG", "en")
    monkeypatch.setattr(wizard, "_require_questionary", lambda: prompts)
    monkeypatch.setattr(choice, "row", prompts.row)

    real = {name: getattr(gateway, name) for name in ("model_options", "model_save_key", "model_login")}

    def handler(name, answer):
        async def run(params, **_):
            calls.append((name, dict(params)))
            return answer(params) if callable(answer) else answer

        monkeypatch.setattr(gateway, name, run)

    def genuine(*names):
        """The module's own handlers back in place of the fakes."""
        for name in names:
            monkeypatch.setattr(gateway, name, real[name])

    def connected(params):
        return {"provider": {**DEEPSEEK, "slug": params["slug"], "authenticated": True}}

    handler("model_options", {"providers": [CODEX, DEEPSEEK, XAI]})
    handler("model_save_key", connected)

    return SimpleNamespace(
        answers=answers,
        calls=calls,
        genuine=genuine,
        handler=handler,
        prompts=prompts,
        transcript=buffer.getvalue,
        wizard=wizard,
    )


def test_step_one_asks_pis_questions_in_pis_order_then_stores_the_key_through_the_gateway(step_one):
    """The TUI's ``/login``, at a terminal: pi's authentication method first,
    pi's provider list filtered to it, then pi's key form -- and the key goes
    where the TUI's goes, to ``model.save_key``, which answers with the row."""
    step_one.answers[:] = ["Sign in with an API key", "DeepSeek", "sk-deepseek-0123456789"]

    assert step_one.wizard._connect_interactive() == ("deepseek", [])

    assert step_one.prompts.asked == ["Select authentication method:", "Select provider to configure:", "API key"]
    # pi's two options, in pi's order: the subscription first.
    assert step_one.prompts.lists[0] == ["Sign in with an account", "Sign in with an API key"]
    assert step_one.calls == [
        ("model_options", {"include_catalog": False}),
        ("model_save_key", {"slug": "deepseek", "api_key": "sk-deepseek-0123456789"}),
    ]
    transcript = step_one.transcript()
    assert transcript.index("Connect DeepSeek") < transcript.index("? API key")
    assert "the key is stored by the gateway, for this provider" in transcript
    assert "✓ Logged in to DeepSeek" in transcript


def test_the_provider_list_is_the_featured_few_and_the_endpoint_row(step_one):
    """A wizard screen is read at a glance, so it offers the handful almost
    everybody picks, in the featured order, and says where the rest are. Under
    the key method the endpoint is the last option; under the subscription
    method there is no endpoint to declare. An option is its name alone unless
    pi's marker says something: a clean config would otherwise read
    "unconfigured" on every one."""
    supplied = _row(
        "openai", "OpenAI", auth_methods=["key"], auth_type="key", authenticated=True, key_env="OPENAI_API_KEY"
    )
    unfeatured = _row("ant-ling", "Ant Ling", auth_methods=["key"], auth_type="key")
    rows = [unfeatured, DEEPSEEK, XAI, CODEX, supplied]
    step_one.handler("model_options", {"providers": rows})
    step_one.answers[:] = ["Sign in with an account", "Back", "Sign in with an API key", "DeepSeek", "sk-0123456789"]

    step_one.wizard._connect_interactive()

    signs_in, takes_a_key = step_one.prompts.lists[1], step_one.prompts.lists[3]
    assert signs_in == ["OpenAI Codex", "xAI"], "the featured that sign in; nothing to declare"
    assert takes_a_key == ["OpenAI  ·  ✓ env: OPENAI_API_KEY", "DeepSeek", "xAI", "OpenAI Compatible"]
    assert "Every other provider is a /login away" in step_one.transcript()


def test_a_refused_key_is_said_in_the_gateways_words_and_asked_again(step_one):
    """The form stays up with the gateway's own sentence, as the TUI's does; a
    stored key the provider still does not authenticate with is the row's
    warning, and the field is asked again either way."""
    from opendde_harness.tui_rpc.errors import ConfigValidationError

    verdicts = iter(
        [
            ConfigValidationError("'claude' is not a pi provider id"),
            {"provider": {**DEEPSEEK, "authenticated": False, "warning": "the key was refused upstream"}},
            {"provider": {**DEEPSEEK, "authenticated": True}},
        ]
    )

    def judge(_params):
        verdict = next(verdicts)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    step_one.handler("model_save_key", judge)
    step_one.answers[:] = ["Sign in with an API key", "DeepSeek", "one", "two", "three"]

    assert step_one.wizard._connect_interactive() == ("deepseek", [])

    assert step_one.prompts.asked.count("API key") == 3
    transcript = step_one.transcript()
    assert "'claude' is not a pi provider id" in transcript
    assert "the key was refused upstream" in transcript
    assert "sk-" not in transcript


def test_a_blank_key_is_back_where_the_key_is_required_and_an_answer_where_it_is_not(step_one):
    """The TUI's field says which: a variable already supplying the key makes
    blank mean "use it", a provider that can do without one makes blank an
    answer, and only a provider that requires one turns blank into the way out."""
    supplied = _row("openai", "OpenAI", auth_methods=["key"], auth_type="key", key_env="OPENAI_API_KEY")
    # The gate accepts an empty key here (an environment chain, say): the row says so.
    optional = _row("google", "Google", auth_methods=["key"], auth_type="key", needs_api_key=False)
    step_one.handler("model_options", {"providers": [DEEPSEEK, supplied, optional]})
    step_one.answers[:] = ["Sign in with an API key", "OpenAI", ""]
    assert step_one.wizard._connect_interactive() == ("openai", [])
    assert step_one.calls[-1] == ("model_save_key", {"slug": "openai", "api_key": ""})
    assert "OPENAI_API_KEY is already set — leave this blank to use it" in step_one.transcript()

    step_one.answers[:] = ["Sign in with an API key", "Google", ""]
    assert step_one.wizard._connect_interactive() == ("google", [])
    assert step_one.calls[-1] == ("model_save_key", {"slug": "google", "api_key": ""})
    assert "optional for this provider" in step_one.transcript()

    # Blank on a required key is the way back to the list, and the list is asked again.
    step_one.answers[:] = ["Sign in with an API key", "DeepSeek", "", "OpenAI", ""]
    assert step_one.wizard._connect_interactive() == ("openai", [])
    assert step_one.prompts.asked[-4:] == [
        "Select provider to configure:",
        "API key",
        "Select provider to configure:",
        "API key",
    ]


def test_a_failed_sign_in_is_said_the_way_pi_says_it_and_returns_to_the_list(step_one):
    """Entering the provider is what starts a sign-in, so a failed one leaves
    the user on the list with pi's own sentence and the gateway's reason."""
    from opendde_harness.tui_rpc.errors import ConfigValidationError

    def refuse(_params, **_):
        raise ConfigValidationError("the sign-in did not finish (login_failed)")

    step_one.handler("model_login", refuse)
    step_one.answers[:] = [
        "Sign in with an account",
        "OpenAI Codex",
        "Back",
        "Sign in with an API key",
        "DeepSeek",
        "k-0123456789",
    ]

    assert step_one.wizard._connect_interactive() == ("deepseek", [])

    transcript = step_one.transcript()
    assert "Sign in to OpenAI Codex" in transcript
    assert "Failed to login to OpenAI Codex: the sign-in did not finish (login_failed)" in transcript
    assert step_one.prompts.asked[:3] == [
        "Select authentication method:",
        "Select provider to configure:",
        "Select provider to configure:",
    ]


def test_a_named_provider_goes_straight_in_and_one_that_offers_both_is_asked_pis_question(step_one):
    """``--provider`` alone is ``/login <provider>``: one way in goes straight
    to it, both ask pi's question titled with the provider's name and worded
    with its own label where pi wrote one."""
    step_one.handler(
        "model_login", lambda params: {"login_id": params["login_id"], "provider": {**CODEX, "authenticated": True}}
    )

    step_one.answers[:] = []
    assert step_one.wizard._connect_interactive("openai-codex") == ("openai-codex", [])
    assert step_one.prompts.asked == []
    assert "✓ Logged in to OpenAI Codex" in step_one.transcript()

    step_one.answers[:] = ["Sign in with an API key", "sk-xai-0123456789"]
    assert step_one.wizard._connect_interactive("xAI") == ("xai", [])
    assert step_one.prompts.asked == ["Select authentication method for xAI:", "API key"]
    assert step_one.prompts.lists[-1] == ["Sign in with SuperGrok or X Premium", "Sign in with an API key"]


def test_the_flags_connect_without_prompts_through_the_same_handlers(step_one):
    """Headless is the same two handlers fed from flags: a key for one of pi's
    own, an address for an endpoint declared here. What has no headless way in
    is refused with where to go instead."""
    import typer

    step_one.handler(
        "model_declare_provider",
        lambda params: {
            "provider": {"slug": params["provider"], "name": params["provider"], "models": ["a", "b", "published"]}
        },
    )
    headless = step_one.wizard._connect_headless

    assert headless("deepseek", api_key="sk-0123456789", base_url=None, model=None) == ("deepseek", [])
    assert step_one.calls[-1] == ("model_save_key", {"slug": "deepseek", "api_key": "sk-0123456789"})

    assert headless("my-vllm", api_key=None, base_url="http://127.0.0.1:8000/v1", model="a, b") == (
        "my-vllm",
        ["a", "b"],
    )
    assert step_one.calls[-1] == (
        "model_declare_provider",
        {"provider": "my-vllm", "base_url": "http://127.0.0.1:8000/v1", "api_key": "", "model": "a, b"},
    )
    assert headless("my-vllm", api_key=None, base_url="http://127.0.0.1:8000/v1", model=None)[1] == [
        "a",
        "b",
        "published",
    ]

    with pytest.raises(typer.BadParameter, match="ddeharness provider set openai"):
        headless("openai", api_key="sk-0123456789", base_url="https://relay.example/v1", model=None)
    with pytest.raises(typer.BadParameter, match="--base-url is required"):
        headless("my-vllm", api_key=None, base_url=None, model=None)
    with pytest.raises(typer.BadParameter, match="--api-key is required"):
        headless("deepseek", api_key=None, base_url=None, model=None)
    with pytest.raises(typer.Exit):
        headless("openai-codex", api_key=None, base_url=None, model=None)
    assert "ddeharness provider login openai-codex" in step_one.transcript()
    assert step_one.prompts.asked == []


def _offline(monkeypatch):
    """The model service cannot be asked: the gateway answers from the config alone."""
    from opendde_harness.tui_rpc.methods import model as gateway

    async def refuse(*_a, **_k):
        raise RuntimeError("no model service here")

    for name in ("service_rows", "provider_auth", "refresh_models"):
        monkeypatch.setattr(gateway, name, refuse)


def test_the_codex_flow_writes_the_sign_in_and_nothing_else(wizard_config, step_one, monkeypatch):
    """``{"login": "oauth"}`` is the whole entry: the grant itself lives in the
    model service's credential store, which is the process that refreshes it, so
    a copy here would be a second answer that goes stale on the first rotation.

    The real ``model.login`` runs, over a scripted service: pi's login-method
    menu arrives as a step and is answered from the row, the device code is
    shown the way the TUI's view shows it, and the entry is written once the
    grant is stored."""
    from opendde_harness.tui_rpc.methods import model as gateway
    from tests._login import CODE_STEP, MENU_STEP, FakeLoginService

    _offline(monkeypatch)
    step_one.genuine("model_options", "model_save_key", "model_login")
    service = FakeLoginService(
        [MENU_STEP, CODE_STEP], {"provider": "openai-codex", "type": "oauth"}, release_on_answer=True
    )

    async def answer():
        return service

    monkeypatch.setattr(gateway, "_login_service", answer)
    step_one.answers[:] = ["Sign in with an account", "OpenAI Codex", "Device code"]

    assert step_one.wizard._connect_interactive() == ("openai-codex", [])
    step_one.wizard._persist_default_model("openai-codex/gpt-5.5-codex", "openai-codex")

    assert service.answers == [(11, "device_code")]
    transcript = step_one.transcript()
    assert step_one.prompts.lists[-1] == ["Browser login (default)", "Device code login (headless)"]
    assert "https://example.test/device" in transcript
    assert "Enter code: ABCD-1234" in transcript
    assert "Waiting for authentication..." in transcript
    assert "✓ Logged in to OpenAI Codex" in transcript
    assert json.loads(wizard_config.read_text()) == {
        "providers": {"openai-codex": {"login": "oauth"}},
        "agents": {"defaults": {"model": "openai-codex/gpt-5.5-codex"}},
    }
    config = _routes(wizard_config, "openai-codex/gpt-5.5-codex")
    assert config.get_provider_name("openai-codex/gpt-5.5-codex") == "openai-codex"


def test_an_api_key_vendor_writes_the_key_and_nothing_else(wizard_config, step_one, monkeypatch):
    """One of pi's own: pi carries the address, the protocol and the catalogue,
    so the entry adds the credential and no second copy of any of them. The
    real ``model.save_key`` writes it, offline: a gateway that cannot ask pi
    lists the row under its own way in."""
    from opendde_harness.providers import pi_ids

    for name in pi_ids.ENV_KEYS["deepseek"]:
        monkeypatch.delenv(name, raising=False)  # else the form offers the environment's key instead
    _offline(monkeypatch)
    step_one.genuine("model_options", "model_save_key")
    step_one.answers[:] = ["Sign in with an API key", "DeepSeek", "sk-deepseek-0123456789"]

    assert step_one.wizard._connect_interactive() == ("deepseek", [])
    step_one.wizard._persist_default_model("deepseek-v4-flash", "deepseek")

    assert json.loads(wizard_config.read_text()) == {
        "providers": {"deepseek": {"apiKey": "sk-deepseek-0123456789"}},
        "agents": {"defaults": {"model": "deepseek/deepseek-v4-flash"}},
    }
    config = _routes(wizard_config, "deepseek/deepseek-v4-flash")
    # The bare id is qualified on the way in, and the prefix is what routes it.
    assert config.get_api_key("deepseek/deepseek-v4-flash") == "sk-deepseek-0123456789"
    assert config.get_api_base("deepseek/deepseek-v4-flash") is None
    assert "sk-deepseek" not in step_one.transcript()


def test_a_self_hosted_endpoint_writes_its_address_protocol_key_and_catalogue(wizard_config, step_one, monkeypatch):
    """The TUI's endpoint form, field by field: the id, the address, a key only
    if the server wants one, and the ids only for an endpoint that publishes
    none. The real ``model.declare_provider`` writes all four, with Chat
    Completions as the wire, and the ids stored bare under the id that prefixes
    them."""
    _offline(monkeypatch)
    step_one.genuine("model_options", "model_save_key")
    step_one.answers[:] = [
        "Sign in with an API key",
        "OpenAI Compatible",
        "my-vllm",
        "http://127.0.0.1:8000/v1",
        "sk-relay-0123456789",
        "qwen3-32b, qwen3-8b",
    ]

    assert step_one.wizard._connect_interactive() == ("my-vllm", ["qwen3-32b", "qwen3-8b"])
    step_one.wizard._persist_default_model("my-vllm/qwen3-32b", "my-vllm")

    assert step_one.prompts.asked[2:] == ["Provider id", "Base URL", "API key", "Model ids"]
    transcript = step_one.transcript()
    assert "Add an OpenAI-compatible endpoint" in transcript
    assert "✓ Logged in to my-vllm (2 models from http://127.0.0.1:8000/v1)" in transcript
    assert json.loads(wizard_config.read_text()) == {
        "providers": {
            "my-vllm": {
                "apiKey": "sk-relay-0123456789",
                "baseUrl": "http://127.0.0.1:8000/v1",
                "api": "openai-completions",
                "models": ["qwen3-32b", "qwen3-8b"],
            }
        },
        "agents": {"defaults": {"model": "my-vllm/qwen3-32b"}},
    }
    config = _routes(wizard_config, "my-vllm/qwen3-32b")
    entry = config.get_provider("my-vllm/qwen3-32b")
    assert entry is not None and entry.declared is True
    assert entry.model_ids == ["qwen3-32b", "qwen3-8b"]


def test_every_provider_the_flow_offers_can_actually_be_served(tmp_path, monkeypatch):
    """Listing a provider is a promise: every row the gateway offers for a key
    reaches the model layer with that key, and the one built-in a key alone
    cannot reach -- Azure, a tenant's own resource -- is not offered at all."""
    from opendde_harness.config.schema import Config
    from opendde_harness.providers import login_flow, pi_ids
    from opendde_harness.providers.pi_auth import configure_payload
    from opendde_harness.tui_rpc.methods import model as gateway

    _offline(monkeypatch)
    rows = asyncio.run(gateway.model_options({"include_catalog": False}))["providers"]
    offered = [row["slug"] for row in login_flow.offered(rows, "key")]

    # The featured that take a key, in the featured order: the Codex login has
    # no key to paste, and Azure is not a row at all. Offline, pi was not asked
    # which of them also sign in, so only the one with no key to paste does.
    assert offered == [slug for slug in login_flow.FEATURED if slug != "openai-codex"]
    assert [row["slug"] for row in login_flow.offered(rows, "oauth")] == ["openai-codex"]
    assert "azure-openai-responses" not in {row["slug"] for row in rows}

    unreachable = {}
    for name in offered:
        assert pi_ids.is_builtin(name), name
        config = Config.model_validate({"providers": {name: {"apiKey": "sk-not-real"}}})
        payload = configure_payload(config, tmp_path / "pi-credentials.json")
        if payload["apiKeys"].get(name) != "sk-not-real":
            unreachable[name] = "pi's own provider was given no key"
    assert unreachable == {}


def test_the_wizard_paints_with_the_palettes_tokens_only():
    """Every colour is a theme token -- accent, muted, ok, warn and the rest --
    so a light terminal gets the light values. A literal hex or a named colour
    written into a prompt is one the theme cannot swap."""
    from opendde_harness.cli import _choice, onboard_commands
    from opendde_harness.providers import login_flow

    for module in (onboard_commands, _choice, login_flow):
        source = Path(module.__file__).read_text()
        assert not re.search(r"#[0-9a-fA-F]{6}\b", source), f"{module.__name__} carries a literal colour"
        assert not re.search(r"\[(bold )?(green|yellow|red|cyan|blue|magenta|white)\]", source), module.__name__


def test_leaving_the_gateway_ends_only_what_it_started_and_lets_the_service_close(monkeypatch):
    """The loop carries the model service's own tasks -- the reader on the
    child's stdout -- beside the handlers the wizard starts. Ending every task
    on the loop ended the reader, and the shutdown that then waited for it
    ended cancelled too, which the wizard raised as ``CancelledError`` after a
    key had been stored. Only the wizard's own handlers are ended; the service
    closes itself."""
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.providers import pi_service

    stop = asyncio.Event()
    service: dict = {}

    async def reader():
        await stop.wait()

    async def start_service():
        service["reader"] = asyncio.get_running_loop().create_task(reader())

    async def close():
        # The service's own close: end the child, then wait for the reader.
        stop.set()
        await service["reader"]
        service["closed"] = True

    async def forever():
        await asyncio.Event().wait()

    monkeypatch.setattr(pi_service, "shutdown_service", close)

    with wizard._Gateway() as gateway:
        gateway.call(start_service())
        abandoned = gateway.start(forever())

    assert service.get("closed") is True
    assert abandoned.cancelled(), "a handler still running when the wizard leaves is ended"


def test_the_gateway_survives_the_real_model_service(tmp_path, monkeypatch):
    """The case above, against the child itself: a service started by one
    handler is closed cleanly when the wizard leaves."""
    import shutil

    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import loader
    from opendde_harness.tui_rpc.methods.model import model_options
    from tests._config import write_config

    bundle = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "model-service.js"
    if not bundle.exists() or shutil.which("node") is None:
        pytest.skip("ui-tui/dist/model-service.js is not built (cd ui-tui && npm run build) or node is missing")
    monkeypatch.setenv("OPENDDE_MODEL_SERVICE_FAUX", "1")
    path = write_config(
        tmp_path / "config.json", {"deepseek": {"apiKey": "sk-not-real"}}, model="deepseek/deepseek-v4-flash"
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    loader._cache.clear()

    with wizard._Gateway() as gateway:
        rows = gateway.call(model_options({"include_catalog": False}))["providers"]

    assert any(row["slug"] == "deepseek" and row["auth_methods"] for row in rows), "pi was asked, so the service ran"


def _write_raw(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_config_this_build_understands_is_carried_forward(monkeypatch, tmp_path):
    """The rule is the shape, not the release: a build that did not change
    config.json's shape has nothing to regenerate, and re-running the wizard to
    fix one step should not cost the other three."""
    from opendde_harness.cli import onboard_commands as wizard
    from opendde_harness.config import loader
    from opendde_harness.config.schema import CONFIG_SCHEMA_VERSION

    config = tmp_path / "config.json"
    monkeypatch.setattr(loader, "_current_config_path", config)
    loader._cache.clear()

    assert wizard._reusable_config() is False, "nothing on disk is nothing to reuse"

    _write_raw(config, {"schemaVersion": CONFIG_SCHEMA_VERSION, "language": "zh"})
    assert wizard._reusable_config() is True

    # A shape this build does not know is moved aside rather than migrated,
    # and so is one that predates the number.
    _write_raw(config, {"schemaVersion": CONFIG_SCHEMA_VERSION + 1})
    assert wizard._reusable_config() is False
    _write_raw(config, {"language": "en"})
    assert wizard._reusable_config() is False

    config.write_text("{not json", encoding="utf-8")
    assert wizard._reusable_config() is False


def _wizard_start(monkeypatch, *, reusable: bool, fresh: bool) -> list:
    """Run the wizard far enough to see what it did with what was on disk."""
    from opendde_harness.cli import onboard_commands as wizard

    retired: list = []
    monkeypatch.setattr(wizard, "_reusable_config", lambda: reusable)
    monkeypatch.setattr(wizard, "_retire_existing_setup", lambda: retired.append(True))
    monkeypatch.setattr(wizard, "_bootstrap_empty_config", lambda: None)
    monkeypatch.setattr(wizard, "_check_tty_or_die", lambda non_interactive: None)
    monkeypatch.setattr(wizard, "_config_language", lambda: "en")

    try:
        # The run goes on to walk its screens; what is under test is what it did
        # before them, so the walk is allowed to fail.
        wizard.run_wizard(non_interactive=True, fresh=fresh)
    except BaseException:  # noqa: BLE001 - the screens are not what is under test
        pass
    return retired


def test_a_shape_this_build_knows_is_kept_and_a_shape_it_does_not_is_moved_aside(monkeypatch):
    assert _wizard_start(monkeypatch, reusable=True, fresh=False) == [], "a current shape is carried forward"
    assert _wizard_start(monkeypatch, reusable=False, fresh=False) == [True], "an unknown shape starts clean"


def test_fresh_starts_clean_even_when_the_shape_is_current(monkeypatch):
    assert _wizard_start(monkeypatch, reusable=True, fresh=True) == [True]
