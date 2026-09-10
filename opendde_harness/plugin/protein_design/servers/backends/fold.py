"""OpenDDE structure prediction backend for protein design.

Usage:
    from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor

    predictor = StructurePredictor(FoldConfig(backend="opendde"))
    results = predictor.predict_batch([sequences_dict])
"""

import atexit
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console

from opendde_harness.plugin.protein_design.core.asset_paths import OPENDDE_COMMON_ASSETS

console = Console()
logger = logging.getLogger(__name__)

DOCKER_EXECUTABLE = shutil.which("docker") or "/usr/bin/docker"
NVIDIA_SMI_EXECUTABLE = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
DOCKER_CONTROL_TIMEOUT_SECONDS = 30

OPENDDE_BACKEND = "opendde"
DEFAULT_OPENDDE_MODEL_NAME = "opendde_v1"
OPENDDE_CHECKPOINT_CONTAINER_DIR = "/workspace/opendde_checkpoint"
LOCAL_FOLD_EXECUTION_MODE = "local"
API_FOLD_EXECUTION_MODE = "api"

_ACTIVE_PREDICTORS: weakref.WeakSet[Any] = weakref.WeakSet()
_PREVIOUS_SIGNAL_HANDLERS: Dict[int, Any] = {}
_SIGNAL_HANDLERS_INSTALLED = False


def normalize_execution_mode(execution_mode: Optional[str]) -> str:
    """Resolve and validate the fold execution mode shared by config and harness."""
    mode = str(execution_mode or os.environ.get("OPENDDE_HARNESS_PROTEIN_FOLD_EXECUTION_MODE", "local")).strip().lower()
    if mode not in {"local", "docker", "api"}:
        raise ValueError("fold execution_mode must be 'local', 'docker', or 'api'")
    return mode


def resolve_fold_seeds(seeds: Optional[List[Union[int, str]]]) -> List[int]:
    """Resolve symbolic and numeric seed values before invoking a backend."""
    resolved = []
    for seed in seeds or [42]:
        if isinstance(seed, str) and seed.strip().lower() == "random":
            resolved.append(random.randint(1, 999999))
            continue
        try:
            resolved.append(int(seed))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid folding seed: {seed!r}") from exc
    return resolved


def _close_active_predictors() -> None:
    """Close every predictor owned by this Python process."""
    for predictor in list(_ACTIVE_PREDICTORS):
        try:
            predictor.close()
        except Exception:
            logger.exception("Failed to close a structure predictor during shutdown")


def _handle_shutdown_signal(signum: int, frame: Any) -> None:
    """Release persistent workers before preserving normal signal semantics."""
    logger.warning("Received signal %s; stopping persistent fold workers", signum)
    _close_active_predictors()
    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum, signal.SIG_DFL)
    if callable(previous):
        previous(signum, frame)
        return
    if previous == signal.SIG_IGN:
        return
    raise SystemExit(128 + signum)


def _install_shutdown_signal_handlers() -> None:
    """Install one process-wide handler without stacking predictor handlers."""
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED or threading.current_thread() is not threading.main_thread():
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous = signal.getsignal(signum)
        if previous == signal.SIG_IGN:
            continue
        _PREVIOUS_SIGNAL_HANDLERS[signum] = previous
        signal.signal(signum, _handle_shutdown_signal)
    _SIGNAL_HANDLERS_INSTALLED = True


atexit.register(_close_active_predictors)


@dataclass
class FoldResult:
    """Result from structure prediction."""

    sequences: Dict[str, str]
    structure_path: List[Path]
    confidence_path: List[Path]
    all_atom_confidence_path: Optional[Path] = None
    iptm: float = 0.0
    ptm: float = 0.0
    plddt: float = 0.0
    ranking_score: float = 0.0
    ipsae: float = 0.0
    loglikelihood: float = 0.0
    backend: str = "unknown"
    run_dir: Optional[Path] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert FoldResult to dictionary."""
        return {
            "sequences": self.sequences,
            "structure_path": self.structure_path,
            "confidence_path": self.confidence_path,
            "all_atom_confidence_path": self.all_atom_confidence_path,
            "iptm": self.iptm,
            "ptm": self.ptm,
            "plddt": self.plddt,
            "ranking_score": self.ranking_score,
            "ipsae": self.ipsae,
            "loglikelihood": self.loglikelihood,
            "backend": self.backend,
            "run_dir": self.run_dir,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class FoldConfig:
    """Configuration for structure prediction."""

    backend: Union[str, List[str]] = "opendde"
    use_msa: bool = True
    enable_msa_search: bool = False  # Search MSA/templates when per-chain MSA is not provided
    need_atom_confidence: bool = True
    seeds: List[int] = field(default_factory=lambda: [42])
    deterministic: bool = True
    target_msa: Optional[Dict[str, Any]] = None
    ipsae_dist_cutoff: float = 10.0
    ipsae_pae_cutoff: float = 10.0

    # Execution settings
    execution_mode: Optional[str] = None
    image: Optional[str] = None
    gpus: str = "all"
    device: Optional[str] = None
    enable_batch_inference: bool = True
    persistent_worker: bool = True
    persistent_worker_timeout_seconds: int = 900
    # Wall-clock timeout for one-shot (non-persistent) Docker batch runs.
    # On expiry the named container is killed so its GPUs are released.
    subprocess_timeout_seconds: int = 7200

    # Remote OpenDDE asynchronous job API
    api_url: Optional[str] = None
    api_poll_interval_seconds: float = 5.0
    api_timeout_seconds: int = 7200
    api_request_timeout_seconds: float = 60.0
    api_stalled_poll_limit: int = 3

    # OpenDDE inference settings
    use_templates: bool = False
    recycling_cycles: int = 10
    diffusion_steps: int = 200
    diffusion_samples: int = 1

    # Paths
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        requested = self.backend
        if isinstance(requested, list):
            requested = requested[0] if len(requested) == 1 else ""
        if str(requested).strip().lower() != "opendde":
            raise ValueError("only the OpenDDE fold and refold backend is currently supported")
        self.backend = "opendde"
        self.seeds = resolve_fold_seeds(self.seeds)
        mode = normalize_execution_mode(self.execution_mode)
        self.execution_mode = mode
        self.device = str(self.device or os.environ.get("OPENDDE_HARNESS_COMPUTE_DEVICE", "auto")).strip().lower()
        if mode != API_FOLD_EXECUTION_MODE and str(self.gpus).strip() == "none":
            if self.device.startswith("cuda"):
                raise ValueError("fold.device=cuda conflicts with fold.gpus=none")
            self.device = "cpu"
        if mode == LOCAL_FOLD_EXECUTION_MODE:
            from opendde_harness.cli.compute_environment import resolve_device

            self.device = resolve_device(self.device)
        elif self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("fold.device must be auto, cpu, or cuda")
        if mode != API_FOLD_EXECUTION_MODE and self.device == "cpu":
            self.gpus = "none"
        if mode == API_FOLD_EXECUTION_MODE:
            from opendde_harness.plugin.protein_design.servers.backends.opendde_api import (
                resolve_opendde_api_url,
            )

            self.api_url = resolve_opendde_api_url(self.api_url)
            if self.api_poll_interval_seconds <= 0:
                raise ValueError("fold.api_poll_interval_seconds must be greater than zero")
            if self.api_timeout_seconds <= 0 or self.api_request_timeout_seconds <= 0:
                raise ValueError("OpenDDE API timeouts must be greater than zero")
            if self.api_stalled_poll_limit < 1:
                raise ValueError("fold.api_stalled_poll_limit must be at least one")


class StructurePredictor:
    """Run OpenDDE structure predictions through the configured execution mode."""

    IMAGE_ENV_VAR = "STRUCTPRED_IMAGE_OPENDDE"
    DEFAULT_IMAGE = "opendde:latest"

    def __init__(self, config: FoldConfig):
        """
        Initialize the structure predictor.

        Args:
            config: FoldConfig instance with prediction settings
        """
        self.config = config

        self.backend = OPENDDE_BACKEND

        self._persistent_workers: Dict[str, Dict[str, Any]] = {}
        self._active_container_names: set[str] = set()
        _ACTIVE_PREDICTORS.add(self)
        _install_shutdown_signal_handlers()
        self._validate_environment()
        if self.config.execution_mode == "docker":
            self._cleanup_orphan_workers()

    def close(self) -> None:
        """Stop all resident work owned by this predictor and release its GPUs."""
        for state in list(getattr(self, "_persistent_workers", {}).values()):
            if state.get("mode") == "local":
                self._stop_local_worker(state)
        names = set(getattr(self, "_active_container_names", set()))
        names.update(
            state.get("container_name")
            for state in getattr(self, "_persistent_workers", {}).values()
            if state.get("container_name")
        )
        for name in names:
            subprocess.run(
                [DOCKER_EXECUTABLE, "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
            )
        if hasattr(self, "_active_container_names"):
            self._active_container_names.clear()
        if hasattr(self, "_persistent_workers"):
            self._persistent_workers.clear()

    def _validate_environment(self) -> None:
        """Validate that required environment variables and tools are available."""
        if self._uses_api_runtime():
            return
        if not self._uses_local_runtime() and shutil.which("docker") is None:
            raise RuntimeError("Docker executable not found in PATH. Please install Docker.")
        logger.debug("OpenDDE backend: using Docker image %s", self._get_image())
        self._validate_opendde_env()

    def _uses_local_runtime(self) -> bool:
        return self.config.execution_mode == LOCAL_FOLD_EXECUTION_MODE

    def _uses_api_runtime(self) -> bool:
        return self.config.execution_mode == API_FOLD_EXECUTION_MODE

    def _validate_opendde_env(self) -> None:
        """Validate OpenDDE-specific environment variables."""
        required_vars = ["STRUCTPRED_OPENDDE_ROOT_DIR", "STRUCTPRED_OPENDDE_CODE_DIR"]
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables for OpenDDE: {missing}\n"
                "Please set:\n"
                "  export STRUCTPRED_OPENDDE_ROOT_DIR=/path/to/opendde_data\n"
                "  export STRUCTPRED_OPENDDE_CODE_DIR=/path/to/OpenDDE-main"
            )
        root_dir = Path(os.environ["STRUCTPRED_OPENDDE_ROOT_DIR"]).expanduser()
        common_dir = Path(os.environ.get("STRUCTPRED_OPENDDE_COMMON_DIR", str(root_dir / "common"))).expanduser()
        missing_assets = [str(common_dir / name) for name in OPENDDE_COMMON_ASSETS if not (common_dir / name).is_file()]
        if missing_assets:
            raise RuntimeError(
                "OpenDDE common assets are missing or are dangling symlinks: "
                + ", ".join(missing_assets)
                + ". Set STRUCTPRED_OPENDDE_COMMON_DIR to a directory containing "
                "the resolved local files; runtime downloads are intentionally disabled."
            )

    def _get_opendde_checkpoint_args(self) -> tuple[Optional[str], List[str]]:
        """Return container checkpoint path and Docker mount args for OpenDDE."""
        checkpoint_path = os.environ.get("STRUCTPRED_OPENDDE_CHECKPOINT_PATH")
        if not checkpoint_path:
            return None, []

        host_path = Path(checkpoint_path).expanduser().resolve()
        if not host_path.is_file():
            raise RuntimeError(
                f"OpenDDE checkpoint does not exist: {host_path}. "
                "Set STRUCTPRED_OPENDDE_CHECKPOINT_PATH to a valid .pt file."
            )

        container_path = f"{OPENDDE_CHECKPOINT_CONTAINER_DIR}/{host_path.name}"
        logger.info(f"Using OpenDDE checkpoint: {host_path}")
        if self._uses_local_runtime():
            return str(host_path), []
        return container_path, ["-v", f"{host_path}:{container_path}:ro"]

    def _get_opendde_model_name(self) -> str:
        """Return OpenDDE model config name, allowing checkpoint-specific overrides."""
        model_name = os.environ.get("STRUCTPRED_OPENDDE_MODEL_NAME", "").strip()
        if model_name:
            return model_name
        return DEFAULT_OPENDDE_MODEL_NAME

    def _get_image(self) -> str:
        """Get Docker image for the current backend."""
        if self.config.image:
            return self.config.image
        return os.environ.get(self.IMAGE_ENV_VAR, self.DEFAULT_IMAGE)

    def predict_batch(
        self,
        sequences_list: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> List[FoldResult]:
        """
        Run batch structure prediction for multiple sequences.

        Args:
            sequences_list: List of sequence dictionaries, each with chain IDs as keys
            output_dir: Output directory for prediction results
            dry_run: If True, prepare inputs but don't execute Docker

        Returns:
            List of FoldResult objects, one per input
        """
        if output_dir is None:
            base_dir = Path(self.config.output_dir) if self.config.output_dir else Path.cwd()
            output_dir = base_dir / "fold"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._predict_batch_opendde(sequences_list, output_dir, dry_run)

    def _uses_persistent_worker(self) -> bool:
        """Return whether this backend can reuse a resident multi-GPU model."""
        return self.config.persistent_worker and not self._uses_api_runtime()

    @staticmethod
    def _local_worker_running(state: Dict[str, Any]) -> bool:
        process = state.get("process")
        return isinstance(process, subprocess.Popen) and process.poll() is None

    @classmethod
    def _worker_is_running(cls, state: Dict[str, Any]) -> bool:
        if state.get("mode") == "local":
            return cls._local_worker_running(state)
        container_name = state.get("container_name")
        return bool(container_name and cls._docker_container_running(container_name))

    @staticmethod
    def _stop_local_worker(state: Dict[str, Any]) -> None:
        process = state.get("process")
        if not isinstance(process, subprocess.Popen) or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Local persistent fold worker %s did not exit", process.pid)

    def _build_local_persistent_worker_command(
        self,
        *,
        worker_dir: Path,
        placeholder_input: Path,
        bootstrap_run_dir: Path,
    ) -> List[str]:
        """Turn the normal local inference command into a resident queue worker."""
        command = self._build_opendde_command(self._get_image(), placeholder_input, bootstrap_run_dir)
        if not command or command[0] != "env":
            raise RuntimeError(f"Local persistent {self.backend} worker requires a local command")
        command.insert(1, f"PERSISTENT_FOLD_QUEUE_DIR={worker_dir}")
        worker_program = str(Path(__file__).with_name("persistent_inference_worker.py"))
        program_index = next(
            (
                index
                for index, value in enumerate(command)
                if value == "runner/inference.py" or value.endswith("/runner/inference.py")
            ),
            None,
        )
        if program_index is not None:
            command[program_index] = worker_program
            return command
        try:
            module_index = command.index("runner.inference")
        except ValueError as exc:
            raise RuntimeError(f"Cannot build local persistent {self.backend} worker command") from exc
        if module_index == 0 or command[module_index - 1] != "-m":
            raise RuntimeError(f"Cannot build local persistent {self.backend} worker command")
        command[module_index - 1 : module_index + 1] = [worker_program]
        return command

    def _persistent_worker_state(self, output_dir: Path, bootstrap_run_dir: Path) -> Dict[str, Any]:
        """Start one backend worker for this predictor/output root if needed."""
        output_root = output_dir.resolve()
        existing = self._persistent_workers.get(self.backend)
        if (
            existing
            and (existing.get("output_root") == output_root or existing.get("mode") == "local")
            and self._worker_is_running(existing)
        ):
            return existing
        if existing and self._worker_is_running(existing):
            if existing.get("mode") == "local":
                self._stop_local_worker(existing)
            else:
                subprocess.run(
                    [DOCKER_EXECUTABLE, "rm", "-f", existing["container_name"]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
                )

        digest = hashlib.sha1(str(output_root).encode("utf-8")).hexdigest()[:10]
        container_name = f"opendde-{self.backend}-worker-{digest}"
        worker_dir = output_root / ".persistent_fold_workers" / f"{self.backend}-{digest}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir = worker_dir / "jobs"
        shutil.rmtree(jobs_dir, ignore_errors=True)
        jobs_dir.mkdir(exist_ok=True)
        bootstrap_run_dir = worker_dir / "bootstrap"
        (bootstrap_run_dir / "input").mkdir(parents=True, exist_ok=True)
        (bootstrap_run_dir / "output").mkdir(exist_ok=True)
        (worker_dir / "ready.json").unlink(missing_ok=True)
        (worker_dir / "shutdown.json").unlink(missing_ok=True)
        # A persistent process only needs a real run-like bootstrap layout; every
        # queued request supplies its own input_json_path and dump_dir afterwards.
        placeholder_input = bootstrap_run_dir / "input" / "placeholder.json"
        placeholder_input.write_text("[]", encoding="utf-8")
        if self._uses_local_runtime():
            command = self._build_local_persistent_worker_command(
                worker_dir=worker_dir,
                placeholder_input=placeholder_input,
                bootstrap_run_dir=bootstrap_run_dir,
            )
            startup_log = worker_dir / "worker.log"
            logger.info("Starting local persistent %s worker", self.backend)
            with startup_log.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            state = {
                "mode": "local",
                "process": process,
                "worker_dir": worker_dir,
                "output_root": output_root,
                "log_path": startup_log,
            }
            self._persistent_workers[self.backend] = state
            try:
                self._wait_for_persistent_worker_ready(state)
            except Exception:
                self._persistent_workers.pop(self.backend, None)
                self._stop_local_worker(state)
                raise
            return state

        subprocess.run(
            [DOCKER_EXECUTABLE, "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
        )

        command = self._build_opendde_command(self._get_image(), placeholder_input, bootstrap_run_dir)

        command.remove("--rm")  # Keep failed worker logs available for diagnosis.
        command[2:2] = [
            "-d",
            "--name",
            container_name,
            "--label",
            "opendde_harness.managed=true",
            "--label",
            f"opendde_harness.owner={self._worker_owner_token()}",
        ]
        image_index = command.index(self._get_image())
        worker_source = Path(__file__).resolve()
        repository_root = worker_source.parents[5]
        repository_mount = "/workspace/opendde"
        command[image_index:image_index] = [
            "-e",
            f"PERSISTENT_FOLD_QUEUE_DIR={worker_dir}",
            "-v",
            f"{repository_root}:{repository_mount}:ro",
            "-v",
            f"{output_root}:{output_root}",
        ]
        worker_program = f"{repository_mount}/{worker_source.relative_to(repository_root).as_posix()}"
        try:
            program_index = command.index("runner/inference.py")
        except ValueError:
            try:
                module_index = command.index("runner.inference")
            except ValueError as exc:
                raise RuntimeError(f"Cannot build persistent {self.backend} worker command") from exc
            if module_index == 0 or command[module_index - 1] != "-m":
                raise RuntimeError(f"Cannot build persistent {self.backend} worker command")
            command[module_index - 1 : module_index + 1] = [worker_program]
        else:
            command[program_index] = worker_program

        logger.info("Starting persistent %s worker: %s", self.backend, container_name)
        launch = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=DOCKER_CONTROL_TIMEOUT_SECONDS
        )
        if launch.returncode != 0:
            raise RuntimeError(f"Failed to start persistent {self.backend} worker: {launch.stderr.strip()}")
        state = {
            "mode": "docker",
            "container_name": container_name,
            "worker_dir": worker_dir,
            "output_root": output_root,
        }
        self._persistent_workers[self.backend] = state
        try:
            self._wait_for_persistent_worker_ready(state)
        except Exception:
            self._persistent_workers.pop(self.backend, None)
            subprocess.run(
                [DOCKER_EXECUTABLE, "rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
            )
            raise
        return state

    @staticmethod
    def _process_start_token(pid: int) -> str:
        """Return the Linux process start tick used to detect PID reuse."""
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields_after_name = stat.rsplit(")", 1)[1].split()
            return fields_after_name[19]
        except (OSError, IndexError):
            return ""

    @classmethod
    def _worker_owner_token(cls) -> str:
        return "|".join((socket.gethostname(), str(os.getpid()), cls._process_start_token(os.getpid())))

    @classmethod
    def _owner_process_is_running(cls, owner: str) -> bool:
        try:
            hostname, pid_text, expected_start = owner.split("|", 2)
            pid = int(pid_text)
        except (TypeError, ValueError):
            return True
        if hostname != socket.gethostname():
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        current_start = cls._process_start_token(pid)
        return not expected_start or not current_start or current_start == expected_start

    @classmethod
    def _cleanup_orphan_workers(cls) -> int:
        """Remove managed workers whose owning process has exited."""
        try:
            result = subprocess.run(
                [
                    DOCKER_EXECUTABLE,
                    "ps",
                    "-a",
                    "--filter",
                    "label=opendde_harness.managed=true",
                    "--format",
                    '{{.ID}}\t{{.Label "opendde_harness.owner"}}',
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                return 0
            removed = 0
            for line in result.stdout.splitlines():
                container_id, separator, owner = line.partition("\t")
                if not separator or not container_id or not owner:
                    continue
                if cls._owner_process_is_running(owner):
                    continue
                subprocess.run(
                    [DOCKER_EXECUTABLE, "rm", "-f", container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
                )
                removed += 1
            if removed:
                logger.info("Removed %d orphaned persistent fold worker(s)", removed)
            return removed
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Could not clean orphaned persistent workers: %s", exc)
            return 0

    @staticmethod
    def _docker_container_running(container_name: str) -> bool:
        inspected = subprocess.run(
            [DOCKER_EXECUTABLE, "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
        )
        return inspected.returncode == 0 and inspected.stdout.strip().lower() == "true"

    def _wait_for_persistent_worker_ready(self, state: Dict[str, Any]) -> None:
        ready_path = state["worker_dir"] / "ready.json"
        deadline = time.monotonic() + self.config.persistent_worker_timeout_seconds
        while time.monotonic() < deadline:
            if ready_path.exists():
                logger.info("Persistent %s worker is ready", self.backend)
                return
            if not self._worker_is_running(state):
                if state.get("mode") == "local":
                    startup_log = state.get("log_path")
                else:
                    logs = subprocess.run(
                        [DOCKER_EXECUTABLE, "logs", state["container_name"]],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
                    )
                    startup_log = state["worker_dir"] / "startup_failure.log"
                    startup_log.write_text(logs.stdout + logs.stderr, encoding="utf-8")
                raise RuntimeError(f"Persistent {self.backend} worker exited during startup; see {startup_log}")
            time.sleep(1.0)
        raise TimeoutError(f"Persistent {self.backend} worker did not become ready")

    def _run_persistent_batch(self, input_path: Path, run_dir: Path, output_dir: Path) -> None:
        """Submit one batch to the resident backend process and wait for completion."""
        state = self._persistent_worker_state(output_dir, run_dir)
        job_id = uuid.uuid4().hex
        job_dir = state["worker_dir"] / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        request = {
            "job_id": job_id,
            "input_json_path": str(input_path.resolve()),
            "dump_dir": str((run_dir / "output").resolve()),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        request_path = job_dir / "request.json"
        temporary = job_dir / "request.json.tmp"
        temporary.write_text(json.dumps(request, indent=2), encoding="utf-8")
        temporary.replace(request_path)

        deadline = time.monotonic() + self.config.persistent_worker_timeout_seconds
        while time.monotonic() < deadline:
            done_path = job_dir / "done.json"
            failed_path = job_dir / "failed.json"
            if done_path.exists():
                return
            if failed_path.exists():
                details = json.loads(failed_path.read_text(encoding="utf-8"))
                raise RuntimeError(f"Persistent {self.backend} worker failed: {details.get('errors')}")
            if not self._worker_is_running(state):
                self._persistent_workers.pop(self.backend, None)
                raise RuntimeError(f"Persistent {self.backend} worker exited while processing {job_id}")
            time.sleep(0.25)
        logger.error(
            "Persistent %s worker timed out for batch %s; removing worker and job",
            self.backend,
            job_id,
        )
        if state.get("mode") == "local":
            self._stop_local_worker(state)
        else:
            subprocess.run(
                [DOCKER_EXECUTABLE, "rm", "-f", state["container_name"]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
            )
        if self._persistent_workers.get(self.backend) is state:
            self._persistent_workers.pop(self.backend, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        raise TimeoutError(f"Persistent {self.backend} worker timed out for batch {job_id}")

    def _predict_batch_opendde(
        self,
        sequences_list: List[Dict[str, Any]],
        output_dir: Path,
        dry_run: bool = False,
    ) -> List[FoldResult]:
        """
        OpenDDE batch prediction using native batch inference.

        Args:
            sequences_list: List of sequence dictionaries
            output_dir: Output directory for prediction results
            dry_run: If True, prepare inputs but don't execute Docker

        Returns:
            List of FoldResult objects, one per input
        """
        logger.info(f"Starting OpenDDE batch prediction for {len(sequences_list)} inputs")

        if self._uses_api_runtime():
            return self._predict_batch_opendde_api(sequences_list, output_dir, dry_run)

        run_dir: Optional[Path] = None
        try:
            run_dir = self._create_run_dir(output_dir, prefix="batch_run")

            batch_input = []
            for idx, seq_dict in enumerate(sequences_list):
                input_item = self._prepare_opendde_batch_item(seq_dict, idx)
                batch_input.append(input_item)

            batch_input_path = run_dir / "input" / "batch_input.json"
            batch_input_path.write_text(json.dumps(batch_input, indent=2), encoding="utf-8")

            logger.info(f"Prepared batch input with {len(batch_input)} items")
            uses_persistent_worker = self._uses_persistent_worker()
            if dry_run or not uses_persistent_worker:
                docker_cmd = self._build_opendde_command(self._get_image(), batch_input_path, run_dir)
            else:
                # Avoid building a one-shot OpenDDE command in persistent-worker
                # mode: command construction allocates/logs a torchrun port even
                # though that command is never executed.
                docker_cmd = [
                    "persistent-worker",
                    self.backend,
                    "command-built-on-worker-start",
                ]
            meta = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "backend": self.backend,
                "num_inputs": len(sequences_list),
                "gpus": self.config.gpus,
                "docker_cmd": docker_cmd,
                "dry_run": dry_run,
            }
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            if dry_run:
                logger.warning("Dry run - Docker command prepared but not executed")
                logger.debug(f"Docker command: {' '.join(docker_cmd)}")
                return self._dry_run_results(sequences_list, run_dir)

            if uses_persistent_worker:
                self._run_persistent_batch(batch_input_path, run_dir, output_dir)
            else:
                # Execute Docker
                log_path = run_dir / "logs" / f"{self.backend}.log"
                logger.info(f"Running OpenDDE batch inference... logs -> {log_path}")

                try:
                    self._run_docker_with_timeout(docker_cmd, log_path, f"{self.backend} batch prediction")
                except Exception as e:
                    logger.exception("Failed to execute OpenDDE Docker command: %s", e)
                    raise

            logger.info("OpenDDE batch prediction completed successfully")

            return self._parse_opendde_batch_results(sequences_list, run_dir)

        except Exception as e:
            logger.exception("OpenDDE batch prediction failed: %s", e)
            if run_dir is not None and (run_dir / "output").is_dir():
                partial_results = self._parse_opendde_batch_results(sequences_list, run_dir)
                successful = sum(result.success for result in partial_results)
                if successful:
                    logger.warning(
                        "Recovered %d/%d partial OpenDDE results after batch failure",
                        successful,
                        len(partial_results),
                    )
                    return partial_results
            return self._failure_results(sequences_list, str(e))

    def _predict_batch_opendde_api(
        self,
        sequences_list: List[Dict[str, Any]],
        output_dir: Path,
        dry_run: bool,
    ) -> List[FoldResult]:
        """Submit one remote OpenDDE job and materialize its artifacts locally."""
        from opendde_harness.plugin.protein_design.servers.backends.opendde_api import (
            OpenDDEJobClient,
        )

        run_dir: Optional[Path] = None
        try:
            run_dir = self._create_run_dir(output_dir, prefix="api_run")
            instances = [
                {
                    "name": f"input_{idx:04d}",
                    "chains": [
                        {"id": chain_id, "sequence": sequence}
                        for chain_id, sequence in self._extract_sequences_only(item).items()
                    ],
                    "model_seeds": self.config.seeds,
                }
                for idx, item in enumerate(sequences_list)
            ]
            request_payload = {
                "instances": instances,
                "parameters": {
                    "n_samples": self.config.diffusion_samples,
                    "n_step": self.config.diffusion_steps,
                    "need_atom_confidence": self.config.need_atom_confidence,
                    "dry_run": dry_run,
                },
            }
            (run_dir / "input" / "api_request.json").write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
            with OpenDDEJobClient(
                self.config.api_url or "",
                request_timeout_seconds=self.config.api_request_timeout_seconds,
            ) as client:
                submitted = client.submit(
                    instances,
                    n_samples=self.config.diffusion_samples,
                    n_step=self.config.diffusion_steps,
                    need_atom_confidence=self.config.need_atom_confidence,
                    dry_run=dry_run,
                )
                (run_dir / "api_submission.json").write_text(json.dumps(submitted, indent=2), encoding="utf-8")
                if dry_run:
                    return self._dry_run_results(sequences_list, run_dir)
                run_id = str(submitted.get("run_id") or "").strip()
                if not run_id:
                    raise RuntimeError("OpenDDE API submission did not return run_id")
                completed = client.wait(
                    run_id,
                    poll_interval_seconds=self.config.api_poll_interval_seconds,
                    timeout_seconds=self.config.api_timeout_seconds,
                    stalled_poll_limit=self.config.api_stalled_poll_limit,
                    status_log_path=run_dir / "logs" / "api_status.jsonl",
                )
                (run_dir / "api_response.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
                archive = client.download(run_id, run_dir / f"fold_{run_id}.zip")
                client.extract_download(archive, run_dir / "output")
            return self._parse_opendde_batch_results(sequences_list, run_dir)
        except Exception as exc:
            logger.exception("Remote OpenDDE batch prediction failed: %s", exc)
            return self._failure_results(sequences_list, str(exc), run_dir)

    def _parse_opendde_batch_results(self, sequences_list: List[Dict[str, Any]], run_dir: Path) -> List[FoldResult]:
        """Parse completed OpenDDE items without discarding partial batches."""
        results = []
        for idx, sequences_dict in enumerate(sequences_list):
            try:
                results.append(self._parse_opendde_batch_result(sequences_dict, run_dir, idx))
            except Exception as exc:
                logger.exception("Failed to parse OpenDDE result for input %d: %s", idx, exc)
                results.append(self._failure_result(sequences_dict, str(exc), run_dir))
        return results

    def _prepare_opendde_batch_item(self, sequences_dict: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Prepare a single OpenDDE batch input item."""
        return {
            "name": f"input_{idx:04d}",
            "modelSeeds": self.config.seeds,
            "sequences": self._build_opendde_protein_chain_entities(sequences_dict),
            "covalent_bonds": [],
        }

    def _build_opendde_protein_chain_entities(self, sequences_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build OpenDDE proteinChain entities, preserving per-chain MSA inputs.

        OpenDDE can consume inline unpairedMsa/pairedMsa strings.
        We inline configured MSA files instead of passing host paths because the
        Docker run only mounts the run directory, not arbitrary YAML path roots.
        """
        sequence_entries = []
        for chain_id, sequence_info in sequences_dict.items():
            sequence, unpaired_msa_path, paired_msa_path = self._extract_sequence_and_msa_paths(
                chain_id,
                sequence_info,
            )

            protein_chain = {
                "id": [chain_id],
                "sequence": sequence,
                "count": 1,
            }

            if self.config.use_msa:
                unpaired_msa = self._read_msa_file(Path(unpaired_msa_path)) if unpaired_msa_path else ""
                paired_msa = self._read_msa_file(Path(paired_msa_path)) if paired_msa_path else ""
                if unpaired_msa or paired_msa or not self._should_search_msa_templates():
                    protein_chain["unpairedMsa"] = unpaired_msa
                    protein_chain["pairedMsa"] = paired_msa

            sequence_entries.append({"proteinChain": protein_chain})
        return sequence_entries

    @staticmethod
    def _find_free_port() -> int:
        """Return a free TCP port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def _cif_has_atom_site_rows(cif_path: Path) -> bool:
        """Return False for mmCIF files that define _atom_site columns but no rows."""
        try:
            lines = Path(cif_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return False

        atom_site_columns_seen = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("_atom_site."):
                atom_site_columns_seen = True
                continue
            if atom_site_columns_seen:
                if line.startswith("_") or line.startswith("loop_") or line.startswith("data_"):
                    return False
                return True
        return not atom_site_columns_seen

    def _parse_opendde_batch_result(self, sequences_dict: Dict[str, Any], run_dir: Path, idx: int) -> FoldResult:
        """Parse OpenDDE batch result for a single input."""
        output_dir = run_dir / "output" / f"input_{idx:04d}"

        if not output_dir.exists():
            raise RuntimeError(f"Output directory not found: {output_dir}")

        cif_files = sorted(output_dir.glob("**/*.cif"))
        valid_cif_files = [path for path in cif_files if self._cif_has_atom_site_rows(path)]
        if cif_files and not valid_cif_files:
            return self._failure_result(
                sequences_dict,
                "OpenDDE produced CIF file(s) with empty _atom_site data: "
                + ", ".join(str(path) for path in cif_files),
                run_dir,
            )

        try:
            cif, confidence, full_data, metrics = self._select_best_diffusion_sample(output_dir, require_atom_site=True)
        except RuntimeError:
            if self._uses_api_runtime():
                raise
            full_data_files = sorted(output_dir.glob("**/*full_data*.json"))
            if not valid_cif_files or not full_data_files:
                raise
            cif = valid_cif_files[0]
            confidence = None
            full_data = full_data_files[0]
            metrics = (0.0, 0.0, 0.0, 0.0)
        iptm, ptm, plddt, ranking_score = metrics

        sequences_only = self._extract_sequences_only(sequences_dict)
        ipsae = 0.0
        if self._can_compute_ipsae(sequences_only, full_data):
            ipsae = self._compute_ipsae_safely(
                confidence_json_path=str(full_data),
                output_cif_path=str(cif),
            )
        elif len(sequences_only) > 1 and getattr(self.config, "need_atom_confidence", True) and not full_data:
            logger.warning(
                "OpenDDE did not produce atom-level PAE confidence data; ipSAE is unavailable for %s",
                output_dir,
            )

        return FoldResult(
            sequences=sequences_only,
            structure_path=[cif],
            confidence_path=[confidence] if confidence else [],
            all_atom_confidence_path=full_data,
            iptm=iptm,
            ptm=ptm,
            plddt=plddt,
            ranking_score=ranking_score,
            ipsae=ipsae,
            backend=self.backend,
            run_dir=run_dir,
            success=True,
        )

    def _extract_sequences_only(self, sequences_dict: Dict[str, Any]) -> Dict[str, str]:
        """Extract sequences only from input dict (handles both formats)."""
        result = {}
        for chain_id, val in sequences_dict.items():
            if isinstance(val, str):
                result[chain_id] = val
            elif isinstance(val, dict):
                result[chain_id] = val.get("sequence", "")
        return result

    def _can_compute_ipsae(
        self,
        sequences_only: Dict[str, str],
        confidence_files: Any,
    ) -> bool:
        """Return whether an interface score has both chains and atom confidence."""
        return (
            len(sequences_only) > 1
            and bool(confidence_files)
            and bool(getattr(self.config, "need_atom_confidence", True))
        )

    def _compute_ipsae_safely(
        self,
        confidence_json_path: str,
        output_cif_path: str,
    ) -> float:
        """Compute ipSAE without turning an optional metric failure into fold failure."""
        from opendde_harness.plugin.protein_design.servers.backends.ipsae import compute_ipsae

        try:
            return compute_ipsae(
                dist_cutoff=self.config.ipsae_dist_cutoff,
                pae_cutoff=self.config.ipsae_pae_cutoff,
                confidence_json_path=str(confidence_json_path),
                output_cif_path=str(output_cif_path),
            )
        except Exception as exc:
            logger.warning(
                "ipSAE calculation failed for %s and %s; continuing with ipSAE=0.0: %s",
                confidence_json_path,
                output_cif_path,
                exc,
            )
            return 0.0

    def _failure_result(self, sequences_dict: dict, error: str, run_dir=None):
        """Return a failed FoldResult."""
        return FoldResult(
            sequences=self._extract_sequences_only(sequences_dict),
            structure_path=[],
            confidence_path=[],
            backend=self.backend,
            run_dir=run_dir,
            success=False,
            error=error,
        )

    def _failure_results(self, sequences_list: list, error: str, run_dir=None):
        """Return a list of failed FoldResults."""
        return [self._failure_result(seq, error, run_dir) for seq in sequences_list]

    def _dry_run_results(self, sequences_list: List[Dict[str, Any]], run_dir: Path) -> List[FoldResult]:
        """Return placeholder results for dry runs."""
        return [
            FoldResult(
                sequences=self._extract_sequences_only(seq_dict),
                structure_path=[],
                confidence_path=[],
                backend=self.backend,
                run_dir=run_dir,
                success=True,
                error="Dry run - no prediction executed",
            )
            for seq_dict in sequences_list
        ]

    def _create_run_dir(self, output_dir: Path, prefix: str = "run") -> Path:
        """Create timestamped run directory with input/output/logs subdirs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"{prefix}_{timestamp}_{self.backend}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "input").mkdir(exist_ok=True)
        (run_dir / "output").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        return run_dir

    def _load_json_object(self, json_path: Path) -> dict:
        """Load the first JSON object from a file.

        OpenDDE distributed inference can leave a valid JSON object followed by
        duplicated trailing bytes when multiple ranks race to write the same
        confidence file.  The first object is complete and is the one we should
        consume.
        """
        text = Path(json_path).read_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            data, end = decoder.raw_decode(text)
            trailing = text[end:].strip()
            if trailing:
                logger.warning(
                    "Ignoring trailing non-JSON bytes in %s after first object: %s",
                    json_path,
                    exc,
                )
            return data

    @staticmethod
    def _normalize_plddt(plddt_raw: Any) -> float:
        """Normalize scalar or per-atom pLDDT to the 0-1 range."""
        if isinstance(plddt_raw, (list, tuple)):
            plddt = sum(plddt_raw) / len(plddt_raw) if plddt_raw else 0.0
        else:
            plddt = float(plddt_raw or 0.0)
        return plddt / 100.0 if plddt > 1.5 else plddt

    def _parse_confidence_json(self, json_path) -> tuple:
        """Parse ipTM, pTM, normalized pLDDT, and ranking score."""
        data = self._load_json_object(Path(json_path))
        iptm = data.get("iptm", 0.0)
        ptm = data.get("ptm", 0.0)
        plddt_raw = data.get(
            "plddt",
            data.get("mean_plddt", data.get("atom_plddts", 0.0)),
        )
        plddt = self._normalize_plddt(plddt_raw)
        ranking_score = data.get("ranking_score", 0.0)
        return iptm, ptm, plddt, ranking_score

    @staticmethod
    def _diffusion_sample_index(path: Path) -> Optional[int]:
        match = re.search(r"_sample_(\d+)", path.name)
        return int(match.group(1)) if match else None

    def _select_best_diffusion_sample(
        self, result_dir: Path, require_atom_site: bool = False
    ) -> tuple[Path, Path, Optional[Path], tuple[float, float, float, float]]:
        """Pair diffusion artifacts and select the highest ranking score."""
        summaries = sorted(result_dir.glob("**/*summary_confidence_sample_*.json"))
        samples = []
        for summary in summaries:
            sample_index = self._diffusion_sample_index(summary)
            if sample_index is None:
                continue
            prefix = summary.name.rsplit("_summary_confidence_sample_", 1)[0]
            cif = summary.with_name(f"{prefix}_sample_{sample_index}.cif")
            if not cif.is_file() or (require_atom_site and not self._cif_has_atom_site_rows(cif)):
                continue
            full_data = summary.with_name(f"{prefix}_full_data_sample_{sample_index}.json")
            if not full_data.is_file():
                full_data = None
            metrics = self._parse_confidence_json(summary)
            samples.append((cif, summary, full_data, metrics, sample_index))

        if not samples:
            if self._uses_api_runtime():
                raise RuntimeError(f"No matched OpenDDE API structure/confidence sample found in {result_dir}")
            cif_files = sorted(result_dir.glob("**/*.cif"))
            if require_atom_site:
                cif_files = [path for path in cif_files if self._cif_has_atom_site_rows(path)]
            confidence_files = sorted(result_dir.glob("**/*confidence*.json"))
            if not cif_files or not confidence_files:
                raise RuntimeError(f"No complete structure/confidence sample found in {result_dir}")
            metrics = self._parse_confidence_json(confidence_files[0])
            full_data = next(
                iter(sorted(result_dir.glob("**/*full_data*.json"))),
                None,
            )
            logger.warning(
                "Sample IDs could not be paired in %s; selected first sorted artifacts: structure=%s confidence=%s",
                result_dir,
                cif_files[0],
                confidence_files[0],
            )
            return cif_files[0], confidence_files[0], full_data, metrics

        cif, summary, full_data, metrics, sample_index = max(
            samples,
            key=lambda item: (
                float(item[3][3]),
                float(item[3][0]),
                -item[4],
            ),
        )
        iptm, ptm, plddt, ranking_score = metrics
        logger.info(
            "Selected OpenDDE sample %d/%d in %s: ranking_score=%.4f ipTM=%.4f pTM=%.4f pLDDT=%.4f structure=%s",
            sample_index,
            len(samples),
            result_dir,
            ranking_score,
            iptm,
            ptm,
            plddt,
            cif,
        )
        return cif, summary, full_data, metrics

    def _should_search_msa_templates(self) -> bool:
        """Return True when missing per-chain MSA/templates should be searched."""
        return bool(self.config.use_msa and self.config.enable_msa_search)

    def _extract_sequence_and_msa_paths(
        self,
        chain_id: str,
        sequence_info: Any,
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Normalize simple/dict sequence inputs to sequence plus optional MSA paths."""
        if isinstance(sequence_info, str):
            return sequence_info, None, None

        if isinstance(sequence_info, dict):
            return (
                sequence_info.get("sequence", ""),
                sequence_info.get("unpairedMsaPath") or None,
                sequence_info.get("pairedMsaPath") or None,
            )

        raise ValueError(f"Invalid sequence format for chain {chain_id}: {type(sequence_info)}")

    def _run_docker_with_timeout(self, docker_cmd: List[str], log_path: Path, description: str) -> None:
        """Run a one-shot Docker batch command with a wall-clock timeout.

        A unique container name is injected so a timeout can kill the container
        explicitly; killing only the docker CLI would otherwise leave the
        container (and its GPUs) running.
        """
        docker_cmd = list(docker_cmd)
        timeout = int(getattr(self.config, "subprocess_timeout_seconds", 7200))
        container_name = f"opendde-oneshot-{self.backend}-{uuid.uuid4().hex[:12]}"
        named = False
        if (
            len(docker_cmd) >= 2
            and docker_cmd[0] == DOCKER_EXECUTABLE
            and docker_cmd[1] == "run"
            and "--name" not in docker_cmd
        ):
            docker_cmd[2:2] = ["--name", container_name]
            named = True

        active_name = container_name if named else None
        if active_name is None and "--name" in docker_cmd:
            name_index = docker_cmd.index("--name") + 1
            if name_index < len(docker_cmd):
                active_name = docker_cmd[name_index]
        if not hasattr(self, "_active_container_names"):
            self._active_container_names = set()
        if active_name:
            self._active_container_names.add(active_name)

        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                proc = subprocess.Popen(
                    docker_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    if active_name:
                        subprocess.run(
                            [DOCKER_EXECUTABLE, "rm", "-f", active_name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
                        )
                    else:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.communicate()
                    cleanup = (
                        f"removed container {active_name}. " if active_name else "terminated the local process group. "
                    )
                    raise RuntimeError(
                        f"{description} timed out after {timeout}s; {cleanup}See log: {log_path}"
                    ) from exc
            finally:
                if active_name:
                    self._active_container_names.discard(active_name)

        if proc.returncode != 0:
            raise RuntimeError(f"{description} failed with exit code {proc.returncode}. See log: {log_path}")

    def _read_msa_file(self, msa_path: Path) -> str:
        """Read MSA file content, return empty string if path is empty or file doesn't exist.

        Args:
            msa_path: Path to the MSA file (.a3m format)

        Returns:
            MSA content as string, or empty string if path is empty/invalid
        """
        if not msa_path or str(msa_path) == "":
            return ""

        if not msa_path.exists():
            logger.warning("Configured MSA file does not exist: %s", msa_path)
            return ""

        try:
            return msa_path.read_text().strip()
        except Exception as exc:
            logger.warning("Failed to read configured MSA file %s: %s", msa_path, exc)
            return ""

    def _normalized_gpu_spec(self) -> str:
        return str(self.config.gpus).strip().strip('"').strip("'")

    def _parse_gpu_ids(self) -> List[str]:
        """Parse GPU IDs from config.gpus."""
        gpu_spec = self._normalized_gpu_spec()
        if gpu_spec == "none" or self.config.device == "cpu":
            return []
        if gpu_spec == "all":
            result = subprocess.run(
                [NVIDIA_SMI_EXECUTABLE, "-L"], capture_output=True, text=True, timeout=DOCKER_CONTROL_TIMEOUT_SECONDS
            )
            if result.returncode == 0:
                return [str(i) for i in range(len(result.stdout.strip().split("\n")))]
            return ["0"]

        if gpu_spec.startswith("device="):
            gpu_spec = gpu_spec.removeprefix("device=")

        return [g.strip() for g in gpu_spec.split(",") if g.strip()]

    def _docker_gpu_request(self) -> str:
        """Return the Docker CLI device request for the configured GPUs."""
        gpu_ids = self._parse_gpu_ids()
        return "all" if self._normalized_gpu_spec() == "all" else f'"device={",".join(map(str, gpu_ids))}"'

    def _local_gpu_environment(self) -> List[str]:
        """Expose only configured GPUs to local torchrun/python commands."""
        gpu_ids = self._parse_gpu_ids()
        return [] if self._normalized_gpu_spec() == "all" else [f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}"]

    def _foldcp_environment(self, gpu_ids: List[str]) -> List[str]:
        """Fold-CP settings for callers that resolve them from the environment."""
        if len(gpu_ids) < 2:
            return []
        return [
            "OPENDDE_FOLDCP_MODE=distributed",
            "OPENDDE_FOLDCP_SIZE_DP=1",
            f"OPENDDE_FOLDCP_SIZE_CP={len(gpu_ids)}",
            f"OPENDDE_FOLDCP_DEVICES={','.join(map(str, gpu_ids))}",
        ]

    def _foldcp_arguments(self, gpu_ids: List[str]) -> List[str]:
        """Run the 1 x P context-parallel mesh OpenDDE's runner expects from torchrun."""
        if len(gpu_ids) < 2:
            return []
        return [
            "--foldcp_mode",
            "distributed",
            "--foldcp_size_dp",
            "1",
            "--foldcp_size_cp",
            str(len(gpu_ids)),
            "--foldcp_devices",
            ",".join(map(str, gpu_ids)),
        ]

    def _deterministic_environment(self) -> List[str]:
        """Torch/CUDA environment controls used by OpenDDE."""
        if not getattr(self.config, "deterministic", True):
            return []
        return ["PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8"]

    def _build_opendde_command(self, image: str, input_path: Path, run_dir: Path) -> List[str]:
        """Build OpenDDE Docker command."""
        opendde_root_dir = os.environ.get("STRUCTPRED_OPENDDE_ROOT_DIR")
        opendde_code_dir = os.environ.get("STRUCTPRED_OPENDDE_CODE_DIR")

        if not opendde_root_dir or not opendde_code_dir:
            raise RuntimeError("Missing OpenDDE environment variables")

        model_name = self._get_opendde_model_name()
        checkpoint_container_path, checkpoint_mount_args = self._get_opendde_checkpoint_args()

        gpu_ids = self._parse_gpu_ids()
        # Enable multi-GPU distributed inference for OpenDDE batch prediction
        # Each GPU will process a subset of the batch in parallel
        nproc = len(gpu_ids)

        seeds_str = ",".join(map(str, self.config.seeds))
        use_template = self.config.use_templates or self._should_search_msa_templates()

        if self._uses_local_runtime():
            python_path = ":".join(part for part in (opendde_code_dir, os.environ.get("PYTHONPATH", "")) if part)
            training_root = os.environ.get("STRUCTPRED_OPENDDE_TRAINING_DATA_ROOT", opendde_root_dir)
            common_dir = Path(os.environ.get("STRUCTPRED_OPENDDE_COMMON_DIR", str(Path(opendde_root_dir) / "common")))
            cmd = [
                "env",
                "NCCL_SOCKET_IFNAME=lo",
                "NCCL_IB_DISABLE=1",
                "NCCL_ASYNC_ERROR_HANDLING=1",
                f"OPENDDE_ROOT_DIR={opendde_root_dir}",
                f"TRAINING_DATA_ROOT={training_root}",
                f"PYTHONPATH={python_path}",
                *self._local_gpu_environment(),
                *self._foldcp_environment(gpu_ids),
                *self._deterministic_environment(),
            ]
            if nproc > 1:
                port = self._find_free_port()
                logger.info(f"Using PyTorch distributed port for OpenDDE: {port}")
                cmd.extend(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        f"--nproc_per_node={nproc}",
                        f"--master_port={port}",
                        str(Path(opendde_code_dir) / "runner" / "inference.py"),
                    ]
                )
            else:
                cmd.extend(
                    [
                        sys.executable,
                        str(Path(opendde_code_dir) / "runner" / "inference.py"),
                    ]
                )
            cmd.extend(
                [
                    "--model_name",
                    model_name,
                    "--device",
                    self.config.device or "auto",
                    "--seeds",
                    seeds_str,
                    "--dump_dir",
                    str((run_dir / "output").resolve()),
                    "--input_json_path",
                    str(input_path.resolve()),
                    "--model.N_cycle",
                    str(self.config.recycling_cycles),
                    "--sample_diffusion.N_sample",
                    str(self.config.diffusion_samples),
                    "--sample_diffusion.N_step",
                    str(self.config.diffusion_steps),
                    "--use_msa",
                    str(self.config.use_msa).lower(),
                    "--use_template",
                    str(use_template).lower(),
                    "--need_atom_confidence",
                    str(self.config.need_atom_confidence).lower(),
                    "--data.ccd_components_file",
                    str(common_dir / "components.cif"),
                    "--data.ccd_components_rdkit_mol_file",
                    str(common_dir / "components.cif.rdkit_mol.pkl"),
                    *self._foldcp_arguments(gpu_ids),
                ]
            )
            if checkpoint_container_path:
                cmd.extend(["--load_checkpoint_path", checkpoint_container_path])
            return cmd

        # Build command as list (not shell string) for security
        cmd = [
            DOCKER_EXECUTABLE,
            "run",
            "--rm",
            *(["--gpus", self._docker_gpu_request()] if self.config.device != "cpu" else []),
            "--network",
            "host",
            "--ipc",
            "host",
            "--shm-size",
            "512g",
            "-e",
            "NCCL_SOCKET_IFNAME=lo",
            "-e",
            "NCCL_IB_DISABLE=1",
            "-e",
            "NCCL_ASYNC_ERROR_HANDLING=1",
            "-e",
            "OPENDDE_ROOT_DIR=/opendde_data",
            "-e",
            "TRAINING_DATA_ROOT=/opendde_data",
            "-e",
            "PYTHONPATH=/workspace/OpenDDE",
            "-v",
            f"{opendde_code_dir}:/workspace/OpenDDE",
            "-v",
            f"{opendde_root_dir}:/opendde_data",
            *checkpoint_mount_args,
            "-v",
            f"{run_dir}:/workspace/work",
            "-w",
            "/workspace/OpenDDE",
            image,
        ]
        for setting in (*self._deterministic_environment(), *self._foldcp_environment(gpu_ids)):
            cmd[-1:-1] = ["-e", setting]

        # Add inference command arguments directly
        if nproc > 1:
            port = self._find_free_port()
            logger.info(f"Using PyTorch distributed port for OpenDDE: {port}")
            cmd.extend(
                [
                    "torchrun",
                    "--nproc_per_node=gpu",
                    f"--master_port={port}",
                    "runner/inference.py",
                ]
            )
        else:
            cmd.extend(["python3", "runner/inference.py"])

        # Add inference arguments
        cmd.extend(
            [
                "--model_name",
                model_name,
                "--device",
                self.config.device or "auto",
                "--seeds",
                seeds_str,
                "--dump_dir",
                "/workspace/work/output",
                "--input_json_path",
                f"/workspace/work/input/{input_path.name}",
                "--model.N_cycle",
                str(self.config.recycling_cycles),
                "--sample_diffusion.N_sample",
                str(self.config.diffusion_samples),
                "--sample_diffusion.N_step",
                str(self.config.diffusion_steps),
                "--use_msa",
                str(self.config.use_msa).lower(),
                "--use_template",
                str(use_template).lower(),
                "--need_atom_confidence",
                str(self.config.need_atom_confidence).lower(),
                *self._foldcp_arguments(gpu_ids),
            ]
        )

        if checkpoint_container_path:
            cmd.extend(["--load_checkpoint_path", checkpoint_container_path])

        return cmd
