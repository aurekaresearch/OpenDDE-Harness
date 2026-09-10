"""Local compute containers, prebuilt images and resource checks for onboarding."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import httpx

from opendde_harness.cli.compute_environment import (
    SUBPROCESS_TIMEOUT,
    check_image_environment,
    check_local_platform,
    compute_image,
    load_environment,
)
from opendde_harness.plugin.protein_design.core.asset_paths import DEFAULT_CHECKPOINT, OPENDDE_COMMON_ASSETS
from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_PORT, DEFAULT_OPENDDE_API_URL
from opendde_harness.plugin.protein_design.core.external import DEFAULT_MSA_SERVER_URL, DEFAULT_PROTREK_URL
from opendde_harness.plugin.protein_design.servers import local_service

MANAGED_LABEL = "org.opendde-harness.compute"
AUTH_ENV = "OPENDDE_HARNESS_PROTEIN_DESIGN_TOKEN"
COMMON_ASSETS = OPENDDE_COMMON_ASSETS


class ComputeSetupError(RuntimeError):
    pass


def default_image() -> str:
    return compute_image()


def load_protein_design_config() -> dict[str, Any]:
    """The saved ``plugins.config.protein-design`` slice; ``{}`` when absent or unreadable."""
    from opendde_harness.cli.onboard_commands import _load_raw_config
    from opendde_harness.config.loader import ConfigReadError

    try:
        config = _load_raw_config().get("plugins", {}).get("config", {}).get("protein-design", {})
    except (OSError, ValueError, AttributeError, ConfigReadError):
        return {}
    return config if isinstance(config, dict) else {}


def _diagnostic(value: Any, secrets: tuple[str, ...] = ()) -> str:
    detail = str(value).strip()[-1600:]
    detail = re.sub(r"(https?://)[^/\s]+@", r"\1[redacted]@", detail)
    detail = re.sub(r"(?i)(bearer\s+|(?:token|password|api[_-]?key)[=:]\s*)[^\s,;]+", r"\1[redacted]", detail)
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return detail


def docker(*args: str, env: dict[str, str] | None = None) -> str:
    try:
        executable = shutil.which("docker")
        if not executable:
            raise FileNotFoundError("docker")
        result = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            env={**os.environ, **(env or {})},
            check=False,
        )
    except FileNotFoundError as exc:
        raise ComputeSetupError("Docker CLI is not installed. Install and start Docker before onboarding.") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComputeSetupError("Docker did not respond. Check the daemon and your Docker permissions.") from exc
    if result.returncode:
        secret_values = tuple(
            value
            for key, value in (env or {}).items()
            if any(part in key.upper() for part in ("TOKEN", "PASSWORD", "SECRET", "API_KEY"))
        )
        detail = _diagnostic(result.stderr or result.stdout, secret_values)
        raise ComputeSetupError(
            f"Docker {' '.join(args[:2])} failed (exit {result.returncode}): {detail or 'no diagnostic output'}. "
            "Check Docker daemon status and the selected context; run ddeharness doctor to inspect compute setup."
        )
    return result.stdout.strip()


def check_local_docker() -> dict[str, Any]:
    try:
        check_local_platform()
    except ValueError as exc:
        raise ComputeSetupError(str(exc)) from exc
    endpoint = os.environ.get("DOCKER_HOST", "")
    if os.environ.get("DOCKER_CONTEXT") or not endpoint:
        endpoint = docker("context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
    if not endpoint.startswith(("unix://", "npipe://")):
        raise ComputeSetupError(
            "Local deployment requires a local Docker context. For a remote daemon, deploy there and choose Connect to existing service."
        )
    try:
        info = json.loads(docker("info", "--format", "{{json .}}"))
    except ValueError as exc:
        raise ComputeSetupError(
            "Docker info returned invalid JSON. Check Docker daemon status and the selected context."
        ) from exc
    if not isinstance(info, dict):
        raise ComputeSetupError("Docker info returned an invalid response.")
    if info.get("OSType") != "linux":
        raise ComputeSetupError("The compute image requires Linux containers.")
    if info.get("Architecture") not in {None, "x86_64", "amd64"}:
        raise ComputeSetupError("The compute image requires a Linux x86-64 Docker daemon.")
    return info


def gpu_inventory() -> list[str]:
    """GPU names in device order from nvidia-smi; empty when it is unavailable."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def validate_image(value: str) -> bool | str:
    return (
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*", value))
        or "Enter a Docker image name with an optional tag or digest."
    )


def validate_name(value: str) -> bool | str:
    return (
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)) or "Use letters, digits, dots, underscores or hyphens."
    )


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_port(start: int = DEFAULT_COMPUTE_PORT) -> int:
    """First host port at or above ``start`` that nothing listens on."""
    port = start
    while port <= 65535:
        if port_is_free(port):
            return port
        port += 1
    raise ComputeSetupError(f"No free host port at or above {start} for the compute service.")


def validate_directory(value: str) -> bool | str:
    path = Path(value).expanduser().resolve()
    if not value.strip() or path == Path(path.anchor) or any(c in str(path) for c in (",", "\n", "\r")):
        return "Choose a specific directory without commas or newlines."
    return True


#: One layer status line of ``docker pull`` without a TTY, e.g.
#: ``a1b2c3d4e5f6: Downloading [====>       ]  1.2GB/4.5GB``.
_PULL_STATUS = re.compile(
    r"^(?P<layer>[0-9a-f]{6,}): (?P<status>[A-Za-z][A-Za-z ]+?)"
    r"(?:\s*\[[^\]]*\]\s+(?P<done>[\d.]+\s*[A-Za-z]+)/(?P<total>[\d.]+\s*[A-Za-z]+))?\s*$"
)
_PULL_UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
_PULL_BINARY_UNITS = {"KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
_PULL_DONE = ("Download complete", "Pull complete", "Already exists")
_PULL_TIMEOUT = 7200


def _pull_size(value: str) -> int:
    """Bytes behind a Docker size such as ``1.2GB`` or ``512kB``; 0 when unreadable."""
    match = re.fullmatch(r"([\d.]+)\s*([A-Za-z]+)", value.strip())
    if not match:
        return 0
    unit = match.group(2).upper()
    scale = _PULL_BINARY_UNITS.get(unit) or _PULL_UNITS.get(unit, 0)
    return int(float(match.group(1)) * scale)


def _pull_image(command: list[str], image: str) -> None:
    """Run ``docker pull``, reporting layer download bytes on one progress bar."""
    from rich.console import Console

    from opendde_harness.cli._download import progress

    console = Console()
    layers: dict[str, list[int]] = {}
    extracting: set[str] = set()
    deadline = time.monotonic() + _PULL_TIMEOUT
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        with progress(console) as bar:
            task = bar.add_task(f"Pulling {image}", total=None)
            for raw in process.stdout or ():
                if time.monotonic() > deadline:
                    process.kill()
                    raise ComputeSetupError(f"Compute image pull exceeded {_PULL_TIMEOUT}s and was stopped.")
                line = raw.rstrip()
                match = _PULL_STATUS.match(line)
                if match is None:
                    if line and not console.is_terminal:
                        print(line, flush=True)
                    continue
                layer, status = match.group("layer"), match.group("status").strip()
                if match.group("total"):
                    total = _pull_size(match.group("total"))
                    done = _pull_size(match.group("done"))
                    entry = layers.setdefault(layer, [0, total])
                    entry[1] = total or entry[1]
                    if status == "Downloading":
                        entry[0] = done
                if status in _PULL_DONE and layer in layers:
                    layers[layer][0] = layers[layer][1]
                if status == "Extracting":
                    extracting.add(layer)
                total_bytes = sum(entry[1] for entry in layers.values())
                phase = f", extracting {len(extracting)}/{len(layers)} layers" if extracting else ""
                bar.update(
                    task,
                    completed=sum(entry[0] for entry in layers.values()),
                    total=total_bytes or None,
                    description=f"Pulling {image} ({len(layers)} layers{phase})",
                )
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if process.wait() != 0:
        raise ComputeSetupError(
            f"Compute image pull failed (docker pull exited {process.returncode}). "
            "Check registry access and disk space, then run ddeharness onboard again."
        )


def ensure_image(image: str, *, quiet: bool = False) -> None:
    """Reuse or pull an image without requiring a source checkout."""
    if validate_image(image) is not True:
        raise ComputeSetupError("Invalid compute image name.")
    if docker("image", "ls", "--quiet", "--no-trunc", image):
        docker("image", "inspect", image)
        return
    executable = shutil.which("docker")
    if not executable:
        raise ComputeSetupError("Docker CLI is not installed. Install and start Docker before onboarding.")
    command = [executable, "pull", "--platform", "linux/amd64", image]
    try:
        if quiet:
            subprocess.run([*command, "--quiet"], check=True, timeout=_PULL_TIMEOUT, capture_output=True)
        else:
            _pull_image(command, image)
    except (OSError, subprocess.SubprocessError) as exc:
        # A captured pull reports only its exit status; what went wrong -- rate
        # limited, no route to the registry, no disk -- is on the stderr the
        # capture holds, and without it the message named the image as the
        # suspect when the image was fine.
        detail = getattr(exc, "stderr", None)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "ignore")
        reason = f"{_diagnostic(exc)}: {detail.strip()}" if detail and detail.strip() else _diagnostic(exc)
        raise ComputeSetupError(
            f"Compute image pull failed ({reason}). Run `docker pull {image}` to see the registry's own answer."
        ) from exc
    docker("image", "inspect", image)


def inspect_container(name: str) -> dict[str, Any] | None:
    if validate_name(name) is not True:
        raise ComputeSetupError("Invalid container name.")
    names = docker("container", "ls", "--all", "--format", "{{.Names}}").splitlines()
    if name not in names:
        return None
    return json.loads(docker("container", "inspect", name))[0]


def prepared_path(value: str, *, file: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if not value.strip() or path == Path(path.anchor) or any(c in str(path) for c in (",", "\n", "\r")):
        raise ComputeSetupError("Choose a specific file or directory; mount paths cannot contain commas or newlines.")
    valid = path.is_file() if file else path.is_dir()
    if not valid or not os.access(path, os.R_OK) or (file and path.stat().st_size == 0):
        raise ComputeSetupError(
            f"Required {'file' if file else 'directory'} is missing, empty or unreadable: {path}. "
            "Run ddeharness compute prepare (or ddeharness onboard) first; no download was attempted."
        )
    return path


@dataclass
class DockerSettings:
    image: str
    mode: str
    gpus: str
    package_root: str
    state_dir: str
    output_dir: str = ""
    opendde_data: str = ""
    opendde_common: str = ""
    opendde_checkpoint: str = ""
    python: str = "python"
    shm_size: str = "16g"
    weights_dir: str = ""
    code_mode: str = "checkout"
    port: int = 0
    idle_seconds: int = local_service.DEFAULT_IDLE_SECONDS
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_saved(cls, saved: dict[str, Any]) -> "DockerSettings":
        """Rebuild settings from ``compute_docker``; retired keys such as ``container_name`` are ignored."""
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in saved.items() if key in known}
        values["image"] = str(values.get("image") or default_image())
        try:
            values["port"] = int(values.get("port") or 0)
            values["idle_seconds"] = int(values.get("idle_seconds") or local_service.DEFAULT_IDLE_SECONDS)
            values["env"] = {str(key): str(item) for key, item in (values.get("env") or {}).items()}
            return cls(**values)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ComputeSetupError(
                "Saved compute_docker settings are incomplete or invalid; run ddeharness onboard."
            ) from exc

    def saved(self) -> dict[str, Any]:
        value = asdict(self)
        if not value["port"]:
            del value["port"]
        if not value["env"]:
            del value["env"]
        return value


def validate_settings(
    settings: DockerSettings,
    info: dict[str, Any],
    *,
    require_image: bool = True,
    require_assets: bool = True,
) -> None:
    if validate_image(settings.image) is not True:
        raise ComputeSetupError("Invalid Docker image name.")
    if settings.mode not in {"local", "api"} or not 0 <= settings.port <= 65535:
        raise ComputeSetupError("Invalid folding mode or port.")
    if settings.idle_seconds < 1:
        raise ComputeSetupError("compute_docker.idle_seconds must be a positive number of seconds.")
    if settings.code_mode not in {"managed", "checkout"}:
        raise ComputeSetupError("Code mode must be managed or checkout.")
    if validate_directory(settings.weights_dir) is not True:
        raise ComputeSetupError("Choose a specific weights directory without commas or newlines.")
    if settings.mode == "local" and validate_directory(settings.opendde_data) is not True:
        raise ComputeSetupError("Choose a specific OpenDDE data directory without commas or newlines.")
    if not re.fullmatch(r"all|none|\d+(?:,\d+)*", settings.gpus):
        raise ComputeSetupError("GPU selection must be all, comma-separated device IDs, or none.")
    # Docker can serve --gpus through the NVIDIA hook while its runtime remains runc.
    gpu_runtime = "nvidia" in (info.get("Runtimes") or {}) or shutil.which("nvidia-container-runtime-hook")
    if settings.gpus != "none" and not gpu_runtime:
        raise ComputeSetupError(
            "Docker's NVIDIA runtime is unavailable. Configure NVIDIA Container Toolkit before onboarding."
        )
    if not re.fullmatch(r"[1-9]\d*[mg]", settings.shm_size):
        raise ComputeSetupError("Shared memory must use a positive size such as 16g.")
    if not settings.python or settings.python.startswith("-") or any(c.isspace() for c in settings.python):
        raise ComputeSetupError("Python must be an executable name or path inside the image, not a shell command.")
    external_service_environment(settings.env)
    if require_image:
        try:
            image = json.loads(docker("image", "inspect", settings.image))[0]
            check_image_environment(image, load_environment())
        except ValueError as exc:
            raise ComputeSetupError(str(exc)) from exc
        except ComputeSetupError as exc:
            raise ComputeSetupError(
                "The compute image is not available locally or cannot be inspected. Check the pull output and Docker permissions."
            ) from exc
    if require_assets:
        validate_assets(settings)
    output = Path(settings.state_dir).expanduser().resolve()
    if output == Path(output.anchor) or any(c in str(output) for c in (",", "\n", "\r")):
        raise ComputeSetupError("Choose a specific state directory without commas or newlines.")
    if settings.output_dir and not Path(settings.output_dir).expanduser().resolve().is_relative_to(output):
        raise ComputeSetupError("The task output directory must be inside the mounted state directory.")


def validate_code_assets(settings: DockerSettings) -> None:
    code = prepared_path(settings.package_root)
    if settings.code_mode == "managed":
        from opendde_harness.cli.compute_code import runtime_code_identity, verify_runtime_code

        try:
            verify_runtime_code(code, runtime_code_identity())
        except (OSError, ValueError) as exc:
            raise ComputeSetupError(f"Managed runtime code validation failed: {exc}") from exc
    for name in (
        "opendde_harness/plugin/protein_design/servers/api.py",
        "external/__init__.py",
        "external/opendde/runner/inference.py",
        "external/ligandmpnn/model_utils.py",
        "external/ligandmpnn/data_utils.py",
        "external/plip/plip/plipcmd.py",
    ):
        prepared_path(str(code / name), file=True)


def validate_assets(settings: DockerSettings) -> None:
    validate_code_assets(settings)
    weights = prepared_path(settings.weights_dir)
    prepared_path(str(weights / "soluble_mpnn/solublempnn_v_48_020.pt"), file=True)
    esm = weights / "huggingface/models--facebook--esm2_t33_650M_UR50D"
    reference = prepared_path(str(esm / "refs/main"), file=True).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", reference):
        raise ComputeSetupError("Invalid ESM snapshot reference in the weights directory.")
    for name in ("config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"):
        prepared_path(str(esm / "snapshots" / reference / name), file=True)
    if settings.mode == "local":
        prepared_path(settings.opendde_data)
        common = prepared_path(settings.opendde_common)
        for name in COMMON_ASSETS:
            prepared_path(str(common / name), file=True)
        prepared_path(settings.opendde_checkpoint, file=True)


def prepare_assets(settings: DockerSettings) -> None:
    from opendde_harness.cli.compute_assets import CHECKPOINTS, asset_state_path, prepare

    checkpoint = Path(settings.opendde_checkpoint).name
    if checkpoint not in CHECKPOINTS:
        checkpoint = DEFAULT_CHECKPOINT
    paths = {
        key: value
        for key, value in settings.saved().items()
        if value
        and key
        in (
            "opendde_data",
            "opendde_common",
            "opendde_checkpoint",
        )
    }
    try:
        prepare(
            Path(settings.weights_dir),
            checkpoint,
            asset_state_path(),
            paths=paths,
            opendde_root=Path(settings.opendde_data) if settings.opendde_data else None,
            with_opendde=settings.mode == "local",
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ComputeSetupError(
            f"Compute asset preparation failed: {exc}. No container was started; verified files and partial downloads were retained."
        ) from exc


PROXY_VARIABLES = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
DOCKER_HOST_ALIAS = "host.docker.internal"
LOCAL_PROXY_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
EXTERNAL_SERVICE_VARIABLES = (
    "PROTREK_ENDPOINT",
    "OPENDDE_HARNESS_MSA_SERVER_MODE",
    "MMSEQS_SERVICE_HOST_URL",
    "OPENDDE_HARNESS_MSA_SEARCH_TIMEOUT",
)


def rewrite_proxy_url(value: str) -> str:
    """Point a loopback proxy at the Docker host; the container's own loopback is not it."""
    text = value.strip()
    if not text:
        return text
    scheme, separator, remainder = text.partition("://")
    if not separator:
        scheme, remainder = "", text
    credentials, at, authority = remainder.rpartition("@")
    if authority.startswith("["):
        host, _, port = authority.partition("]")
        host, port = host[1:], port
    else:
        host, colon, number = authority.partition(":")
        port = colon + number
    if host.lower() in LOCAL_PROXY_HOSTS:
        authority = DOCKER_HOST_ALIAS + port
    rebuilt = credentials + at + authority
    return f"{scheme}{separator}{rebuilt}" if separator else rebuilt


def proxy_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Host proxy settings rewritten for the container, in both letter cases."""
    values = dict(os.environ if source is None else source)
    resolved: dict[str, str] = {}
    for name in PROXY_VARIABLES:
        raw = values.get(name) or values.get(name.upper()) or ""
        if not raw.strip():
            continue
        resolved[name] = raw.strip() if name == "no_proxy" else rewrite_proxy_url(raw)
    if not resolved:
        return {}
    bypass = [item.strip() for item in resolved.get("no_proxy", "").split(",") if item.strip()]
    # These public services are directly reachable; a host loopback proxy
    # may not listen on the Docker bridge even after its hostname is rewritten.
    service_hosts = [
        urlsplit(url).hostname for url in (DEFAULT_OPENDDE_API_URL, DEFAULT_PROTREK_URL, DEFAULT_MSA_SERVER_URL)
    ]
    for host in ("localhost", "127.0.0.1", DOCKER_HOST_ALIAS, *service_hosts):
        if host not in bypass:
            bypass.append(host)
    resolved["no_proxy"] = ",".join(bypass)
    return {key: value for name, value in resolved.items() for key in (name, name.upper())}


def external_service_environment(overrides: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Optional-service settings taken from the host, then from ``compute_docker.env``."""
    passthrough = {name: os.environ[name] for name in EXTERNAL_SERVICE_VARIABLES if os.environ.get(name) is not None}
    for key, value in (overrides or {}).items():
        name = str(key).strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ComputeSetupError(f"compute_docker.env names must be environment variable names; got {key!r}.")
        if name == AUTH_ENV or name.startswith("STRUCTPRED_") or name in {"PYTHONPATH", "PROTEIN_DESIGN_OUTPUT_PATH"}:
            raise ComputeSetupError(f"compute_docker.env cannot override {name}; it is set by onboarding.")
        passthrough[name] = str(value)
    return passthrough


def create_arguments(
    settings: DockerSettings, token: str, api_url: str, *, name: str, port: int
) -> tuple[list[str], dict[str, str]]:
    from opendde_harness.cli.compute_assets import model_environment

    state = str(Path(settings.state_dir).expanduser().resolve())
    env = {
        **model_environment(),
        AUTH_ENV: token,
        "PROTEIN_DESIGN_OUTPUT_PATH": str(Path(settings.output_dir).expanduser().resolve())
        if settings.output_dir
        else f"{state}/protein_design",
        "PYTHONPATH": "/workspace:/workspace/external/opendde:/workspace/external/plip",
        "OPENDDE_HARNESS_PROJECT_ROOT": "/workspace",
        "STRUCTPRED_OPENDDE_CODE_DIR": "/workspace/external/opendde",
        "OPENDDE_HARNESS_PROTEIN_FOLD_EXECUTION_MODE": settings.mode,
        "OPENDDE_HARNESS_COMPUTE_DEVICE": "cpu" if settings.gpus == "none" else "cuda",
        "OPENDDE_HARNESS_COMPUTE_IDLE_SECONDS": str(settings.idle_seconds),
        "OPENDDE_HARNESS_WEIGHTS_DIR": "/weights",
        **proxy_environment(),
        **external_service_environment(settings.env),
    }
    mounts = [
        (state, state, False),
        (settings.package_root, "/workspace", True),
        (settings.weights_dir, "/weights", True),
    ]
    if settings.mode == "local":
        weights = Path(settings.weights_dir).expanduser().resolve()
        data = Path(settings.opendde_data).expanduser().resolve()
        if data.is_relative_to(weights):
            base = PurePosixPath("/weights") / data.relative_to(weights).as_posix()
        else:
            base = PurePosixPath("/opendde")
            mounts.append((str(data), str(base), True))
        env["OPENDDE_ROOT_DIR"] = env["STRUCTPRED_OPENDDE_ROOT_DIR"] = str(base)
        for key, value in (
            ("STRUCTPRED_OPENDDE_COMMON_DIR", settings.opendde_common),
            ("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", settings.opendde_checkpoint),
        ):
            try:
                relative = Path(value).expanduser().resolve().relative_to(data)
            except ValueError as exc:
                raise ComputeSetupError(
                    "OpenDDE common data and checkpoint must be inside the OpenDDE data directory."
                ) from exc
            env[key] = str(base / relative.as_posix())
    else:
        env["OPENDDE_HARNESS_OPENDDE_API_URL"] = api_url
    args = [
        "run",
        "--detach",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--label",
        f"{MANAGED_LABEL}=true",
        "--label",
        f"org.opendde-harness.code-mode={settings.code_mode}",
        "--publish",
        f"127.0.0.1:{port}:{DEFAULT_COMPUTE_PORT}",
        "--shm-size",
        settings.shm_size,
        f"--add-host={DOCKER_HOST_ALIAS}:host-gateway",
    ]
    if settings.code_mode == "managed":
        from opendde_harness.cli.compute_code import verify_runtime_code

        identity = verify_runtime_code(Path(settings.package_root))
        args.extend(["--label", f"org.opendde-harness.code-id={identity['id']}"])
        args.extend(["--label", f"org.opendde-harness.code-version={identity['harness_version']}"])
    if settings.gpus != "none":
        args.extend(["--gpus", "all" if settings.gpus == "all" else json.dumps(f"device={settings.gpus}")])
    for source, destination, readonly in mounts:
        source = str(Path(source).expanduser().resolve())
        args.extend(["--mount", f"type=bind,source={source},target={destination}" + (",readonly" if readonly else "")])
    for key in env:
        args.extend(["--env", key])
    args.extend(
        [
            "--entrypoint",
            settings.python,
            settings.image,
            "-m",
            "uvicorn",
            "opendde_harness.plugin.protein_design.servers.api:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            str(DEFAULT_COMPUTE_PORT),
        ]
    )
    return args, env


def wait_for_service(url: str, token: str, mode: str, api_url: str = "", *, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    params = {"backend": "opendde", "execution_mode": mode}
    if api_url:
        params["api_url"] = api_url
    with httpx.Client(
        timeout=5, trust_env=False, headers={"Authorization": f"Bearer {token}"} if token else {}
    ) as client:
        while True:
            try:
                response = client.get(f"{url}/health", params=params)
                if response.status_code in {401, 403}:
                    raise ComputeSetupError(
                        "Compute authentication failed. Check the compute token in ddeharness onboard."
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Unexpected health response")
                workers = payload.get("workers") or {}
                if not isinstance(workers, dict):
                    raise ValueError("Unexpected worker health response")
                evidence = workers.get("backend") or {}
                if not isinstance(evidence, dict):
                    raise ValueError("Unexpected backend health response")
                if mode == "local":
                    missing = [
                        key.removesuffix("_ready")
                        for key in (
                            "code_ready",
                            "root_ready",
                            "common_ready",
                            "checkpoint_ready",
                        )
                        if evidence.get(key) is False
                    ]
                    if missing:
                        raise ComputeSetupError(
                            "Compute service responds, but local resources are unavailable inside the container: "
                            + ", ".join(missing)
                            + ". Check its bind mounts and configured paths; the container exits by itself when idle. No connection settings were saved."
                        )
                device = str(workers.get("device", ""))
                gpu_ready = device == "cpu" or mode != "local" or bool(payload.get("gpu"))
                tool_errors = [
                    name
                    for name, item in (workers.get("tools") or {}).items()
                    if isinstance(item, dict) and item.get("required") and not item.get("ready")
                ]
                if tool_errors:
                    raise ComputeSetupError(
                        "Compute tools are not ready: " + ", ".join(tool_errors) + ". Run ddeharness compute prepare."
                    )
                if payload.get("status") == "ok" and workers.get("backend_ready") is True and gpu_ready:
                    auth = client.get(f"{url}/population/top", params={"top_k": 1})
                    if auth.status_code in {401, 403}:
                        raise ComputeSetupError(
                            "Compute authentication failed. Check the token; the configuration was not saved."
                        )
                    auth.raise_for_status()
                    return
                last_error = str(evidence.get("api_error") or evidence.get("import_error") or payload.get("status"))
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if time.monotonic() >= deadline:
                raise ComputeSetupError(
                    f"Compute readiness check timed out: {last_error}. "
                    "Check the service, prepared resources and folding mode with ddeharness doctor."
                )
            time.sleep(1)


def _reclaim_name(container: dict[str, Any], token: str) -> int | None:
    """Host port of a running container that already holds the name with this token; stopped leftovers are removed."""
    name = str(container.get("Name") or "").lstrip("/")
    config = container.get("Config") or {}
    if (config.get("Labels") or {}).get(MANAGED_LABEL) != "true":
        raise ComputeSetupError(
            f"Container {name} is not managed by OpenDDE Harness; remove or rename it before starting compute."
        )
    if not container.get("State", {}).get("Running"):
        docker("rm", "-f", str(container["Id"]))
        return None
    env = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    bindings = (container.get("HostConfig", {}).get("PortBindings") or {}).get(f"{DEFAULT_COMPUTE_PORT}/tcp") or []
    port = str(bindings[0].get("HostPort", "")) if len(bindings) == 1 else ""
    if env.get(AUTH_ENV) != token or not port.isdigit():
        raise ComputeSetupError(
            f"Container {name} is running with another token or port binding. Run ddeharness compute stop --force, then retry."
        )
    return int(port)


def start_service(
    settings: DockerSettings, token: str, api_url: str = "", *, code_id: str | None = None, quiet: bool = False
) -> dict[str, Any]:
    """Start this release's ephemeral container, wait for readiness and record it in the runtime state file."""
    from opendde_harness.cli.compute_code import prepare_runtime_code, runtime_code_identity

    code_id = code_id or runtime_code_identity()["id"]
    if settings.code_mode == "managed":
        try:
            settings.package_root = str(prepare_runtime_code())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise ComputeSetupError(f"Runtime code preparation failed: {exc}; no container was started.") from exc
    validate_code_assets(settings)
    info = check_local_docker()
    ensure_image(settings.image, quiet=quiet)
    validate_settings(settings, info)
    name = local_service.container_name(code_id)
    existing = inspect_container(name)
    port = _reclaim_name(existing, token) if existing else None
    if port is None:
        Path(settings.state_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        port = settings.port or free_port(DEFAULT_COMPUTE_PORT)
        if settings.port and not port_is_free(port):
            raise ComputeSetupError(
                f"Compute service API port {port} is already in use. Remove compute_docker.port from config.json "
                "so a free port is chosen, or stop the process using it."
            )
        args, env = create_arguments(settings, token, api_url, name=name, port=port)
        docker(*args, env=env)
    state = local_service.new_state(container=name, image=settings.image, code_id=code_id, port=port)
    local_service.write_state(state)
    wait_for_service(state["url"], token, settings.mode, api_url)
    return state


OPTIONAL_EXTERNAL_SERVICES = {
    "protrek": ("ProTrek", "PROTREK_ENDPOINT"),
    "msa": ("Target MSA server", "MMSEQS_SERVICE_HOST_URL"),
}


def external_service_status(tools: Any) -> list[dict[str, Any]]:
    """One entry per optional external service; readiness is reported, never enforced."""
    if not isinstance(tools, dict):
        return []
    services = []
    for name, (label, variable) in OPTIONAL_EXTERNAL_SERVICES.items():
        item = tools.get(name)
        if not isinstance(item, dict):
            continue
        services.append(
            {
                "name": name,
                "label": label,
                "variable": variable,
                "endpoint": item.get("endpoint"),
                "service_check": str(item.get("service_check") or "not_run"),
                "reachable": item.get("reachable"),
                "reason": item.get("reason"),
            }
        )
    return services


def external_service_line(service: Mapping[str, Any]) -> str:
    """The doctor/wizard one-liner for one optional external service."""
    label = str(service.get("label") or service.get("name") or "external service")
    variable = str(service.get("variable") or "")
    endpoint = service.get("endpoint")
    if not endpoint:
        return f"{label}: disabled (optional; set {variable} to enable it)"
    if service.get("reachable"):
        return f"{label}: reachable from the compute container ({endpoint})"
    if service.get("service_check") == "not_run":
        return f"{label}: not probed (optional; endpoint {endpoint})"
    reason = str(service.get("reason") or "unreachable")
    return f"{label}: unreachable from the compute container (optional; {reason}; set {variable} or a proxy)"


def probe_external_services(url: str, token: str, *, timeout: float = 10) -> list[dict[str, Any]]:
    """Optional-service readiness read from a running compute service; never raises."""
    try:
        with httpx.Client(
            timeout=timeout, trust_env=False, headers={"Authorization": f"Bearer {token}"} if token else {}
        ) as client:
            response = client.get(f"{url.rstrip('/')}/health", params={"probe_external": 1})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("workers"), dict):
        return []
    return external_service_status(payload["workers"].get("tools"))


def inspect_compute(config: dict[str, Any], *, verify_hashes: bool = False, timeout: float = 10) -> dict[str, Any]:
    from opendde_harness.cli.compute_assets import inspect_assets, opendde_root, weights_root
    from opendde_harness.cli.compute_code import managed_code_status

    if config.get("compute_workers"):
        from opendde_harness.plugin.protein_design.servers.compute_pool import ComputeWorker

        try:
            workers = [
                ComputeWorker.from_mapping(item, default_token=config.get("compute_token"))
                for item in config["compute_workers"]
            ]
            if len({item.worker_id for item in workers}) != len(workers):
                raise ValueError("Compute worker IDs must be unique")
        except (AttributeError, TypeError, ValueError) as exc:
            return {
                "ready": False,
                "placement": "worker_pool",
                "checks": [{"name": "configuration", "ok": False, "error": _diagnostic(exc)}],
            }
        reports = [
            {
                "id": item.worker_id,
                **inspect_compute(
                    {
                        **config,
                        "compute_workers": [],
                        "compute_docker": {},
                        "compute_url": item.url,
                        "compute_token": item.token,
                    },
                    verify_hashes=verify_hashes,
                    timeout=timeout,
                ),
            }
            for item in workers
        ]
        return {
            "ready": all(item["ready"] for item in reports),
            "placement": "worker_pool",
            "workers": reports,
            "checks": [
                {
                    "name": item["id"],
                    "ok": item["ready"],
                    "error": None if item["ready"] else "Worker is not ready; inspect its checks.",
                }
                for item in reports
            ],
        }
    saved = config.get("compute_docker") or {}
    fold = config.get("fold_defaults") or {}
    mode = fold.get("execution_mode", "api")
    local = bool(saved)
    url = "" if local else str(config.get("compute_url") or "").rstrip("/")
    report: dict[str, Any] = {
        "ready": False,
        "placement": "local_docker" if local else "remote_service",
        "fold_mode": mode,
        "device": None,
        "checks": [],
    }

    def record(name: str, operation):
        try:
            result = operation()
            report["checks"].append({"name": name, "ok": True})
            return result
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            httpx.HTTPError,
            ComputeSetupError,
            subprocess.SubprocessError,
        ) as exc:
            token = str(config.get("compute_token") or "")
            detail = _diagnostic(exc, (token,))
            report["checks"].append({"name": name, "ok": False, "error": detail})
            return None

    if not local and not url:
        report["checks"].append(
            {"name": "configuration", "ok": False, "error": "Run ddeharness onboard to configure Protein Design."}
        )
    if local:
        info = record("docker", check_local_docker)
        if info:
            image = str(saved.get("image") or default_image())
            record(
                "image",
                lambda: check_image_environment(json.loads(docker("image", "inspect", image))[0], load_environment()),
            )
        checkpoint = Path(saved.get("opendde_checkpoint") or DEFAULT_CHECKPOINT).name
        assets = record(
            "assets",
            lambda: inspect_assets(
                weights_root(saved),
                opendde_root=opendde_root(saved),
                checkpoint=checkpoint,
                with_opendde=mode == "local",
                verify_hashes=verify_hashes,
            ),
        )
        report["assets"] = assets
        if assets and not assets["ready"]:
            report["checks"].append(
                {
                    "name": "required_assets",
                    "ok": False,
                    "error": "Required weights are missing or invalid; run ddeharness compute prepare.",
                }
            )
        if saved.get("code_mode") == "managed":
            # Managed code is not saved in the config; its directory follows the
            # release, and the service prepares it at start, so readiness means
            # "startable without a download", not "already copied".
            status = record("runtime_code", managed_code_status)
            if status:
                report["code"] = status["identity"]
                if not status["prepared"]:
                    report["checks"][-1]["note"] = "prepared at first start from cached sources"
        elif not Path(saved.get("package_root") or "", "external/opendde/runner/inference.py").is_file():
            report["checks"].append(
                {
                    "name": "runtime_code",
                    "ok": False,
                    "error": "Configured runtime code is missing; run ddeharness onboard.",
                }
            )
        service = record("container", lambda: local_service.instance_status(config))
        if service:
            report["service"] = service
            if service["running"]:
                url = str(service["url"])
            else:
                report["device"] = "cpu" if saved.get("gpus") == "none" else "cuda"
    if url:

        def probe():
            params = {"backend": "opendde", "execution_mode": mode, "probe_external": 1}
            if fold.get("api_url"):
                params["api_url"] = fold["api_url"]
            token = str(config.get("compute_token") or "")
            with httpx.Client(
                timeout=timeout, trust_env=False, headers={"Authorization": f"Bearer {token}"} if token else {}
            ) as client:
                response = client.get(f"{url}/health", params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("workers"), dict):
                    raise ComputeSetupError("Compute service returned an invalid health response")
                workers = payload.get("workers") or {}
                if payload.get("status") != "ok" or workers.get("backend_ready") is not True:
                    raise ComputeSetupError("Compute service is not ready: " + json.dumps(workers, ensure_ascii=True))
                if not workers.get("tools") or not workers.get("device"):
                    raise ComputeSetupError(
                        "Compute service does not report tool/device readiness. Check the compute token "
                        "(a service with a token hides details from unauthenticated callers), or update its runtime code and rerun onboarding."
                    )
                auth = client.get(f"{url}/population/top", params={"top_k": 1})
                auth.raise_for_status()
                return payload

        payload = record("service", probe)
        if payload:
            report["device"] = payload["workers"].get("device")
            report["tools"] = payload["workers"].get("tools")
            report["external_services"] = external_service_status(payload["workers"].get("tools"))
            report["environment"] = payload.get("environment")
            report["code_version"] = payload.get("code_version")
    report["ready"] = bool(report["checks"]) and all(item["ok"] for item in report["checks"])
    return report
