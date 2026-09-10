import json
from pathlib import Path

import pytest

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
    return {"Id": name, "Name": "/" + name, "State": {"Running": True}}


def test_resolver_reuses_running_healthy_container(harness_home, monkeypatch):
    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    monkeypatch.setattr(onboard_compute, "inspect_container", _running)
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


def test_resolver_starts_new_release_and_asks_the_old_container_to_exit_when_idle(harness_home, monkeypatch):
    from opendde_harness.cli import compute_code, onboard_compute

    monkeypatch.setattr(compute_code, "runtime_code_identity", lambda: {"id": CODE_ID})
    monkeypatch.setattr(onboard_compute, "inspect_container", _running)
    monkeypatch.setattr(onboard_compute, "docker", lambda *args, **k: pytest.fail(f"unexpected docker {args}"))
    monkeypatch.setattr(
        local_service, "service_health", lambda url, token, **k: pytest.fail("old release must not be probed")
    )
    retired = []
    monkeypatch.setattr(
        local_service, "request_shutdown", lambda url, token, *, if_idle, timeout=5.0: retired.append((url, if_idle))
    )
    old = local_service.new_state(container="opendde-compute-old", image="img", code_id="old-code", port=18092)
    local_service.write_state(old)
    started = []

    def start(settings, token, api_url, *, code_id, quiet):
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
    # Asked, not killed: a busy old release answers 409 and finishes its work.
    assert retired == [("http://127.0.0.1:18092", True)]


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
    monkeypatch.setattr(onboard_compute, "docker", lambda *a, **k: pytest.fail(f"unexpected docker {a}"))
    assert start_service(settings, "token", code_id=CODE_ID)["port"] == 18095
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


@pytest.fixture
def wizard_run(monkeypatch):
    import io

    from rich.console import Console

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
        "Compute service: bioinformatics tools running in a Docker container (SolubleMPNN/ProteinMPNN",
        "Compute container: starts on demand when a task needs it and is removed after 10 minutes idle; ddeharness compute stop",
        "Compute image: example/env:tag",
        "Compute device: 2 × NVIDIA A800-SXM4-80GB (all visible; tasks may pin a device per tool, automatic by default)",
        "? OpenDDE fold/refold mode:",
        f"OpenDDE folding API: {step.DEFAULT_OPENDDE_API_URL} (official default service",
        "Harness tool weights (OPENDDE_HARNESS_WEIGHTS_DIR):",
        "  SolubleMPNN: soluble_mpnn/solublempnn_v_48_020.pt",
        "Compute service connected.",
    ]
    positions = [transcript.find(line) for line in expected]
    assert all(index >= 0 for index in positions), transcript
    assert positions == sorted(positions), transcript
    assert "Compute service API port" not in transcript and "opendde-protein-design-compute" not in transcript
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
    assert "Compute service API port (compute_docker.port): 18089" in transcript
    assert "removed after 2 minutes idle" in transcript and "Compute device: cpu" in transcript
    saved = outcome["saved"]["compute_docker"]
    assert saved["port"] == 18089 and saved["idle_seconds"] == 120 and "container_name" not in saved
    assert outcome["saved"]["compute_token"] == "existing-token"


def test_busy_container_keeps_running_and_settings_are_saved(wizard_run, monkeypatch):
    monkeypatch.setattr(local_service, "stop_if_idle", lambda token, **k: False)
    transcript, outcome = wizard_run({})
    assert "busy and keeps its current settings" in transcript
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


def test_custom_endpoint_credentials_ask_which_wire_and_write_it(monkeypatch):
    from opendde_harness.cli import onboard_commands as wizard

    written = {}
    monkeypatch.setattr(wizard, "_prompt_api_key", lambda provider, **kw: "sk-test")
    monkeypatch.setattr(wizard, "_prompt_base_url", lambda *a, **kw: "https://relay.example/v1")
    monkeypatch.setattr(wizard, "_prompt_wire", lambda *a, **kw: "responses")
    monkeypatch.setattr(wizard, "_write_provider_fields", lambda provider, fields: written.update({provider: fields}))

    wizard._collect_credentials(
        "custom", is_oauth=False, is_custom=True, api_key=None, base_url=None, model=None, non_interactive=False
    )

    assert written["custom"] == {"api_key": "sk-test", "api_base": "https://relay.example/v1", "wire": "responses"}


def test_non_interactive_custom_endpoint_writes_no_wire_unless_given(monkeypatch):
    from opendde_harness.cli import onboard_commands as wizard

    written = {}
    monkeypatch.setattr(wizard, "_write_provider_fields", lambda provider, fields: written.update({provider: fields}))
    args = dict(is_oauth=False, is_custom=True, api_key="sk", base_url="https://relay.example/v1", model="gpt-x")

    wizard._collect_credentials("custom", non_interactive=True, **args)
    assert "wire" not in written["custom"]

    wizard._collect_credentials("custom", non_interactive=True, wire="chat", **args)
    assert written["custom"]["wire"] == "chat"


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
