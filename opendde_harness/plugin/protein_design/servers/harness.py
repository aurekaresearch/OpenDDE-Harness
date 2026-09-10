"""Compute harness adapters used by the REST service."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from opendde_harness.cli.compute_environment import resolve_device
from opendde_harness.plugin.protein_design.core.asset_paths import OPENDDE_COMMON_ASSETS as _OPENDDE_COMMON_ASSETS
from opendde_harness.plugin.protein_design.core.constants import CANONICAL_AMINO_ACIDS
from opendde_harness.plugin.protein_design.core.contracts import HotspotCoordinate
from opendde_harness.plugin.protein_design.core.external import (
    MSA_SERVICE,
    PROTREK_SERVICE,
    ExternalServiceUnavailableError,
    connection_reason,
    msa_endpoint,
    probe_endpoint,
    protrek_endpoint,
    required_message,
    unavailable_payload,
)
from opendde_harness.plugin.protein_design.core.search_history import normalize_search_candidate

logger = logging.getLogger(__name__)


def _optional_external(
    service: str,
    endpoint: str | None,
    call: Callable[[], Any],
) -> dict[str, Any]:
    """Run one optional external lookup; an outage becomes a result, not an exception."""
    try:
        return {"available": True, "result": call()}
    except ExternalServiceUnavailableError as exc:
        logger.warning("%s is unreachable at %s: %s", service, exc.endpoint or endpoint, exc.reason)
        return unavailable_payload(service, exc.reason, exc.endpoint or endpoint)
    except Exception as exc:
        reason = connection_reason(exc)
        if reason is None:
            raise
        logger.warning("%s is unreachable at %s: %s", service, endpoint, reason)
        return unavailable_payload(service, reason, endpoint)


def _optional_service_endpoints() -> dict[str, str | None]:
    """Configured endpoints of the optional external services, ``None`` when disabled."""
    return {PROTREK_SERVICE: protrek_endpoint(), MSA_SERVICE: msa_endpoint()}


def _probe_gpus() -> list[dict[str, Any]]:
    """Return a small, failure-tolerant GPU readiness snapshot."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    fields = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(row[0]),
                    "name": row[1].strip(),
                    "memory_total_mb": int(row[2]),
                    "memory_used_mb": int(row[3]),
                    "memory_free_mb": int(row[4]),
                    "utilization_percent": int(row[5]),
                }
            )
        except ValueError:
            continue
    return gpus


def _fold_devices(spec: Any) -> frozenset[int] | None:
    """GPU indices one fold configuration occupies; ``None`` means every GPU."""
    text = str(spec if spec is not None else "all").strip().strip('"').strip("'").removeprefix("device=")
    if text == "none":
        return frozenset()
    if not text or text == "all":
        return None
    try:
        return frozenset(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        return None


def _shares_gpu(left: frozenset[int] | None, right: frozenset[int] | None) -> bool:
    if left == frozenset() or right == frozenset():
        return False
    if left is None or right is None:
        return True
    return bool(left & right)


def _probe_local_fold_import(code_dir: Path) -> tuple[bool, str | None]:
    worker = Path(__file__).with_name("backends") / "persistent_inference_worker.py"
    python = sys.executable
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(code_dir), environment.get("PYTHONPATH", "")) if part
    )
    try:
        result = subprocess.run(
            [python, str(worker), "--probe-import"],
            cwd=code_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, None
    detail = (result.stderr or result.stdout).strip()
    return False, detail[-2000:] or f"import probe exited {result.returncode}"


TOOL_ASSET_PREFIXES = {"esm2": "huggingface/", "soluble_mpnn": "soluble_mpnn/"}
ESM_SCORE_CACHE_LIMIT = 50_000


def _assets_by_tool(assets: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    root = Path(assets["root"])
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in TOOL_ASSET_PREFIXES}
    for item in assets["files"]:
        path = Path(item["path"])
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        for name, prefix in TOOL_ASSET_PREFIXES.items():
            if relative.startswith(prefix):
                groups[name].append(item)
    return groups


def _asset_group_ready(group: list[dict[str, Any]]) -> bool:
    """A tool whose assets are absent from the plan is not ready."""
    return bool(group) and all(item["ok"] for item in group)


class PythonProteinDesignHarness:
    def __init__(
        self,
        *,
        output_path: str | None = None,
        max_structure_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self._output_path = (
            Path(output_path).expanduser() if output_path else Path.home() / ".opendde_harness" / "protein_design"
        ).resolve()
        self._max_structure_bytes = max_structure_bytes
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._predictors: dict[str, Any] = {}
        self._predictor_devices: dict[str, frozenset[int] | None] = {}
        self._esm_clients: dict[str, Any] = {}
        self._esm_score_cache: dict[tuple[str, str], float] = {}
        self._soluble_mpnn_clients: dict[str, Any] = {}
        self._populations: dict[Path, list[dict[str, Any]]] = {}
        self._histories: dict[Path, list[dict[str, Any]]] = {}
        self._backend_probe_cache: dict[tuple[str, str], tuple[bool, str | None]] = {}
        self._cache_lock = threading.RLock()
        self._population_locks: dict[Path, threading.RLock] = {}

    def close(self) -> None:
        """Release resident model processes owned by this compute service."""
        with self._cache_lock:
            resident = [
                *self._predictors.values(),
                *self._esm_clients.values(),
                *self._soluble_mpnn_clients.values(),
            ]
            self._predictors.clear()
            self._predictor_devices.clear()
            self._esm_clients.clear()
            self._soluble_mpnn_clients.clear()
            self._esm_score_cache.clear()
        for model in resident:
            close = getattr(model, "close", None)
            if callable(close):
                close()

    async def health(
        self,
        backend: str | None = None,
        execution_mode: str | None = None,
        image: str | None = None,
        api_url: str | None = None,
        *,
        probe_external: bool = False,
    ) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.fold import normalize_execution_mode

        normalized_backend = str(backend or "opendde").strip().lower()
        if normalized_backend != "opendde":
            raise ValueError("only the OpenDDE fold and refold backend is currently supported")
        mode = normalize_execution_mode(execution_mode)
        return await asyncio.to_thread(self._health_snapshot, normalized_backend, mode, image, api_url, probe_external)

    def _health_snapshot(
        self,
        backend: str,
        mode: str,
        image: str | None,
        api_url: str | None,
        probe_external: bool = False,
    ) -> dict[str, Any]:
        from opendde_harness.cli.compute_assets import inspect_assets

        output_ready = os.access(self._output_path, os.W_OK)
        docker_required = mode == "docker"
        docker_socket_ready = Path("/var/run/docker.sock").exists() and os.access(
            "/var/run/docker.sock", os.R_OK | os.W_OK
        )
        docker_cli_ready = shutil.which("docker") is not None

        backend_ready = True
        backend_evidence: dict[str, Any] = {}
        if mode == "local":
            code_dir = Path(os.environ.get("STRUCTPRED_OPENDDE_CODE_DIR", "/opt/opendde")).expanduser()
            root_value = os.environ.get("STRUCTPRED_OPENDDE_ROOT_DIR", "").strip()
            checkpoint_value = os.environ.get("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", "").strip()
            root_dir = Path(root_value).expanduser() if root_value else None
            common_value = os.environ.get("STRUCTPRED_OPENDDE_COMMON_DIR", "").strip()
            common_dir = (
                Path(common_value).expanduser() if common_value else (root_dir / "common" if root_dir else None)
            )
            common_assets = {
                name: bool(common_dir and (common_dir / name).is_file()) for name in _OPENDDE_COMMON_ASSETS
            }
            checkpoint_path = Path(checkpoint_value).expanduser() if checkpoint_value else None
            import_ready, import_error = (
                self._cached_backend_import_probe(
                    backend,
                    code_dir,
                )
                if code_dir.is_dir()
                else (False, "backend code directory is unavailable")
            )
            backend_evidence = {
                "code_dir": str(code_dir),
                "code_ready": code_dir.is_dir(),
                "root_dir": str(root_dir) if root_dir else None,
                "root_ready": bool(root_dir and root_dir.is_dir()),
                "common_dir": str(common_dir) if common_dir else None,
                "common_assets": common_assets,
                "common_ready": bool(common_assets) and all(common_assets.values()),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_ready": bool(checkpoint_path and checkpoint_path.is_file()),
                "import_ready": import_ready,
                "import_error": import_error,
            }
            backend_ready = all(
                backend_evidence[key]
                for key in (
                    "code_ready",
                    "root_ready",
                    "common_ready",
                    "checkpoint_ready",
                    "import_ready",
                )
            )
        elif mode == "api":
            from opendde_harness.plugin.protein_design.servers.backends.opendde_api import (
                OpenDDEJobClient,
                resolve_opendde_api_url,
            )

            resolved_api_url = api_url
            try:
                resolved_api_url = resolve_opendde_api_url(api_url)
                with OpenDDEJobClient(resolved_api_url) as client:
                    client.probe()
                    api_ready = True
                    api_error = None
            except Exception as exc:
                api_ready = False
                api_error = str(exc)
            backend_evidence = {
                "api_url": resolved_api_url or None,
                "api_ready": api_ready,
                "api_error": api_error,
            }
            backend_ready = api_ready
        elif docker_required:
            requested_image = image or self._default_backend_image(backend)
            image_ready = bool(
                requested_image
                and docker_cli_ready
                and docker_socket_ready
                and self._docker_image_exists(requested_image)
            )
            backend_evidence = {
                "image": requested_image,
                "image_ready": image_ready,
            }
            backend_ready = docker_cli_ready and docker_socket_ready and image_ready

        try:
            device = resolve_device()
            device_error = None
        except (ImportError, ValueError, RuntimeError) as exc:
            device = None
            device_error = str(exc)
        assets = inspect_assets(Path(os.environ.get("OPENDDE_HARNESS_WEIGHTS_DIR", "/weights")))
        asset_groups = _assets_by_tool(assets)
        tools = {
            "esm2": {"required": True, "ready": _asset_group_ready(asset_groups["esm2"])},
            "soluble_mpnn": {"required": True, "ready": _asset_group_ready(asset_groups["soluble_mpnn"])},
            "plip": {"required": True, "ready": shutil.which(os.environ.get("PLIP_COMMAND", "plipcmd.py")) is not None},
            "foldmason": {"required": True, "ready": shutil.which("foldmason") is not None},
            "protrek": {
                "required": False,
                "ready": importlib.util.find_spec("gradio_client") is not None,
                "service_check": "not_run",
            },
            "msa": {"required": False, "ready": True, "service_check": "not_run"},
            "developability": {
                "required": False,
                "ready": False,
                "reason": "backend is not included in this distribution",
            },
        }
        for name, endpoint in _optional_service_endpoints().items():
            tools[name]["endpoint"] = endpoint
            if probe_external:
                probe = probe_endpoint(endpoint)
                tools[name]["reachable"] = probe["reachable"]
                tools[name]["reason"] = probe["reason"]
                tools[name]["service_check"] = "reachable" if probe["reachable"] else "unreachable"
        return {
            "status": "ok" if output_ready and backend_ready and device is not None else "degraded",
            "models": {
                "fold": backend_ready,
                "esm2": bool(self._esm_clients),
                "soluble_mpnn": bool(self._soluble_mpnn_clients),
            },
            "workers": {
                "mode": "embedded-python",
                "output_ready": output_ready,
                "execution_mode": mode,
                "device": device,
                "device_error": device_error,
                "tools": tools,
                "assets": assets,
                "docker_cli_ready": docker_cli_ready,
                "docker_socket_ready": docker_socket_ready,
                "docker_required": docker_required,
                "requested_backend": backend,
                "requested_image": image,
                "backend_ready": backend_ready,
                "backend": backend_evidence,
            },
            "gpu": _probe_gpus(),
        }

    def _cached_backend_import_probe(
        self,
        backend: str,
        code_dir: Path,
    ) -> tuple[bool, str | None]:
        key = (backend, str(code_dir.resolve()))
        with self._cache_lock:
            cached = self._backend_probe_cache.get(key)
        if cached is not None:
            return cached
        result = _probe_local_fold_import(code_dir)
        with self._cache_lock:
            self._backend_probe_cache[key] = result
        return result

    @staticmethod
    def _default_backend_image(backend: str) -> str | None:
        if backend == "opendde":
            return "opendde-harness:last"
        return None

    @staticmethod
    def _docker_image_exists(image: str) -> bool:
        docker = shutil.which("docker")
        if docker is None:
            return False
        try:
            result = subprocess.run(
                [docker, "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    async def invoke(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        methods = {
            "fold": self._fold,
            "target_msa_search": self._target_msa_search,
            "score_esm": self._score_esm,
            "generate_soluble_mpnn": self._generate_soluble_mpnn,
            "population_get": self._population_get,
            "population_update": self._population_update,
            "population_top": self._population_top,
            "developability": self._developability,
            "structure_read": self._structure_read,
            "pose_rmsd": self._pose_rmsd,
            "evolution_tree": self._evolution_tree,
            "epitope_analysis": self._epitope_analysis,
            "structure_analysis": self._structure_analysis,
            "protrek_sequence_search": self._protrek_sequence_search,
            "protrek_structure_search": self._protrek_structure_search,
            "generate_esm2_guided": self._generate_esm2_guided,
        }
        method = methods.get(operation)
        if method is None:
            raise NotImplementedError(f"operation {operation!r} requires a configured command")
        return await asyncio.to_thread(method, payload)

    def _target_msa_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.msa import search_target_msa

        target_name = str(payload["target_name"])
        chain_id = str(payload["chain_id"])
        required = bool(payload.get("required", True))
        endpoint = msa_endpoint()
        try:
            return search_target_msa(
                target_name=target_name,
                chain_id=chain_id,
                sequence=str(payload["sequence"]),
                output_root=self._output_path / "msa",
                force=bool(payload.get("force", False)),
            )
        except ExternalServiceUnavailableError as exc:
            logger.warning("%s is unreachable at %s: %s", MSA_SERVICE, exc.endpoint or endpoint, exc.reason)
            if required:
                raise RuntimeError(required_message(MSA_SERVICE, exc.reason, exc.endpoint or endpoint)) from exc
            return {
                "target_name": target_name,
                "chain_id": chain_id,
                "server_url": exc.endpoint or endpoint,
                "server_mode": os.environ.get("OPENDDE_HARNESS_MSA_SERVER_MODE", "protenix").strip().lower()
                or "protenix",
                "available": False,
                "reason": exc.reason,
            }

    def _fold(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.fold import (
            FoldConfig,
            StructurePredictor,
            normalize_execution_mode,
        )

        options = dict(payload.get("options") or {})
        options["execution_mode"] = normalize_execution_mode(options.get("execution_mode"))
        task_output = self._task_output_path(payload)
        options["backend"] = payload.get("backend", "opendde")
        if str(options["backend"]).strip().lower() != "opendde":
            raise ValueError("only the OpenDDE fold and refold backend is currently supported")
        options.setdefault("output_dir", str(task_output))
        allowed = {item.name for item in fields(FoldConfig)}
        config_values = {key: value for key, value in options.items() if key in allowed}
        objective_key = str(options.get("objective_key", "iptm")).lower()
        if (
            options["execution_mode"] == "api"
            and objective_key == "loss"
            and options.get("need_atom_confidence") is False
        ):
            raise ValueError("API loss optimization requires need_atom_confidence: true")
        # Output paths identify jobs, not model configurations.  Keeping them in
        # the cache key loaded one resident model per task and retained every old
        # model on GPU.  A compute service owns one compatible resident predictor;
        # a changed scientific/backend config replaces it explicitly.
        cache_values = {key: value for key, value in config_values.items() if key != "output_dir"}
        cache_key = json.dumps(cache_values, sort_keys=True, default=str)
        devices = _fold_devices(config_values.get("gpus"))
        with self._cache_lock:
            predictor = self._predictors.get(cache_key)
            if predictor is None:
                # Folds leased disjoint GPUs run at the same time, so only a
                # resident model that shares a GPU with this one must go.
                for key in [key for key, held in self._predictor_devices.items() if _shares_gpu(devices, held)]:
                    close = getattr(self._predictors.pop(key, None), "close", None)
                    self._predictor_devices.pop(key, None)
                    if callable(close):
                        close()
                predictor = StructurePredictor(FoldConfig(**config_values))
                self._predictors[cache_key] = predictor
                self._predictor_devices[cache_key] = devices
        raw_candidates = payload.get("candidates") or []
        sequences = [self._candidate_chains(item, options) for item in raw_candidates]
        esm_scores: list[float] = []
        if objective_key == "loss":
            esm_payload = {
                "sequences": [str(item.get("sequence") or "") for item in raw_candidates],
                "options": options.get("esm2_options") or {},
            }
            esm_scores = self._score_esm(esm_payload)["scores"]
        fold_output = task_output / "fold"
        if options.get("post_refold"):
            fold_output = task_output / "post_refold" / uuid.uuid4().hex
        results = predictor.predict_batch(sequences, output_dir=fold_output)
        if len(results) != len(raw_candidates):
            raise RuntimeError(f"OpenDDE returned {len(results)} results for {len(raw_candidates)} candidates")
        candidates = []
        for index, result in enumerate(results):
            source = raw_candidates[index]
            result_data = self._jsonable(result.to_dict())
            source_metadata = dict(source.get("metadata") or {})
            for key in ("parent_id", "skill_id", "status"):
                if source.get(key) is not None:
                    source_metadata.setdefault(key, source[key])
            fold_chains = sequences[index]
            binder_chain_ids = [str(item) for item in options.get("binder_chain_ids") or []]
            binder_chains = {
                chain_id: self._chain_sequence(fold_chains[chain_id])
                for chain_id in binder_chain_ids
                if chain_id in fold_chains
            }
            if not binder_chains:
                raw_binder = source_metadata.get("chains")
                if isinstance(raw_binder, Mapping):
                    binder_chains = {
                        str(chain): self._chain_sequence(sequence)
                        for chain, sequence in raw_binder.items()
                        if str(chain) not in set(options.get("target_chain_ids") or [])
                    }
            source_metadata["chains"] = binder_chains
            metrics = {
                "iptm": float(result.iptm),
                "ptm": float(result.ptm),
                "plddt": float(result.plddt),
                "ipsae": float(result.ipsae),
                "ranking_score": float(result.ranking_score),
            }
            scoring_error = None
            loss_score = None
            structure_path = self._first_path(result_data.get("structure_path"))
            gate_passed, gate_evidence = self._evaluate_gate(
                result_data,
                sequences[index],
                options,
            )
            metrics["gate_passed"] = 1.0 if gate_passed else 0.0
            for key in (
                "cdr_contact_fraction",
                "framework_contact_fraction",
            ):
                value = gate_evidence.get(key)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            metrics["cdr3_gate_passed"] = 1.0 if gate_evidence.get("cdr3_gate_passed") else 0.0
            metrics["cdr_contact_fraction_gate_passed"] = (
                1.0 if gate_evidence.get("cdr_contact_fraction_gate_passed") else 0.0
            )
            source_metadata.update(
                {
                    "gate_passed": gate_passed,
                    "gate_evidence": gate_evidence,
                    "hotspot_gate": gate_evidence,
                }
            )
            if objective_key == "loss" and result.success:
                try:
                    from opendde_harness.plugin.protein_design.servers.backends.loss_confidence_scorer import (
                        score_confidence_loss,
                    )

                    loss_score = score_confidence_loss(
                        backend=str(result_data.get("backend") or payload.get("backend")),
                        confidence_path=result_data["all_atom_confidence_path"],
                        structure_path=structure_path,
                        sequences=dict(result_data.get("sequences") or sequences[index]),
                        binder_chains=list(options.get("binder_chain_ids") or []),
                        fixed_residues=options.get("fixed_residues") or {},
                        iptm=float(result.iptm),
                        esm2_pll=esm_scores[index],
                        loss_weights=options.get("loss_weights") or {},
                        target_hotspots=options.get("target_hotspots") or None,
                    )
                    metrics["loss"] = float(loss_score["loss"])
                    metrics["loglikelihood"] = float(esm_scores[index])
                except Exception as exc:
                    scoring_error = f"loss scoring failed: {exc}"
            fold_success = bool(result.success) and scoring_error is None
            objective = metrics.get(objective_key)
            if objective is not None and not math.isfinite(float(objective)):
                objective = None
                fold_success = False
                scoring_error = scoring_error or "objective is not finite"
            candidates.append(
                {
                    "candidate_id": str(source.get("candidate_id") or source.get("id") or f"candidate_{index}"),
                    "sequence": str(source.get("sequence") or self._binder_sequence(source, options)),
                    "objective": objective,
                    "metrics": metrics,
                    "structure_path": structure_path,
                    "metadata": {
                        **source_metadata,
                        "fold": result_data,
                        "loss": loss_score,
                        "success": fold_success,
                        "error": scoring_error or result.error,
                    },
                }
            )
        if candidates and not any(item["metadata"]["success"] for item in candidates):
            errors = "; ".join(
                f"{item['candidate_id']}: {item['metadata']['error'] or 'fold failed'}" for item in candidates
            )
            raise RuntimeError(f"All OpenDDE candidates failed: {errors}")
        return {"candidates": candidates}

    def _evaluate_gate(
        self,
        result_data: Mapping[str, Any],
        sequences: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Run OpenDDE Harness's hotspot/CDR admission gate on the folded structure."""
        structure_path = self._first_path(result_data.get("structure_path"))
        if not structure_path:
            return False, {"reason": "missing_structure_path"}
        try:
            from opendde_harness.plugin.protein_design.core.gate import evaluate_hotspot_contact_map

            binder_chains = [str(item) for item in options.get("binder_chain_ids") or []]
            if not binder_chains:
                return False, {"reason": "missing_binder_chains"}
            all_cdr_positions: dict[str, set[int]] = {}
            cdr3_positions: dict[str, set[int]] = {}
            configured_cdr3 = options.get("cdr3_positions") or {}
            configured_cdr = options.get("cdr_regions") or {}
            configured_groups = options.get("cdr_region_groups") or {}
            for chain in binder_chains:
                explicit_cdr = configured_cdr.get(chain) if isinstance(configured_cdr, Mapping) else None
                if explicit_cdr:
                    cdr = {int(position) for position in explicit_cdr}
                else:
                    cdr = set()
                all_cdr_positions[chain] = cdr
                explicit = configured_cdr3.get(chain) if isinstance(configured_cdr3, Mapping) else None
                if explicit is not None:
                    cdr3_positions[chain] = {int(position) for position in explicit}
                else:
                    raw_groups = configured_groups.get(chain) if isinstance(configured_groups, Mapping) else None
                    blocks = (
                        [[int(position) for position in group] for group in raw_groups]
                        if raw_groups
                        else self._contiguous_blocks(cdr)
                    )
                    cdr3_positions[chain] = set(blocks[2]) if len(blocks) >= 3 else set()
            raw_hotspots = options.get("target_hotspots") or {}
            hotspots = (
                [
                    HotspotCoordinate(chain=str(chain), position=int(position))
                    for chain, positions in raw_hotspots.items()
                    for position in positions
                ]
                if isinstance(raw_hotspots, Mapping)
                else []
            )
            target_chains = options.get("target_chain_ids")
            if not target_chains:
                target_chains = [str(chain) for chain in sequences if str(chain) not in set(binder_chains)]
            passed, coverage = evaluate_hotspot_contact_map(
                structure_path,
                binder_chains,
                hotspots,
                target_chains=[str(chain) for chain in target_chains],
                distance_cutoff=float(options.get("hotspot_contact_cutoff_a", 5.0)),
                cdr3_positions=cdr3_positions,
                all_cdr_positions=all_cdr_positions,
                cdr_contact_fraction_threshold=float(options.get("cdr_contact_fraction_threshold", 0.5)),
            )
            coverage = self._jsonable(coverage)
            if passed:
                coverage["reason"] = (
                    "cdr_contact_fraction_passed" if coverage.get("epitope_gate_skipped") else "contacted_hotspot"
                )
            elif not coverage.get("cdr_contact_fraction_gate_passed", True):
                coverage["reason"] = "cdr_contact_fraction_below_threshold"
            elif not coverage.get("cdr3_gate_passed", True):
                coverage["reason"] = "cdr3_hotspot_gate_failed"
            else:
                coverage["reason"] = "no_hotspot_contact"
            return bool(passed), coverage
        except Exception as exc:
            return False, {
                "reason": "gate_evaluation_failed",
                "error": str(exc),
                "structure_path": structure_path,
            }

    @staticmethod
    def _contiguous_blocks(positions: set[int]) -> list[list[int]]:
        blocks: list[list[int]] = []
        for position in sorted(positions):
            if not blocks or position != blocks[-1][-1] + 1:
                blocks.append([position])
            else:
                blocks[-1].append(position)
        return blocks

    def _score_esm(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.esm_backend import (
            DEFAULT_CACHE_DIR,
            DEFAULT_MODEL_NAME,
            ESMClient,
        )

        options = dict(payload.get("options") or {})
        options["device"] = resolve_device(options.get("device"))
        model_name = str(options.get("model_name", DEFAULT_MODEL_NAME))
        cache_dir = str(options.get("cache_dir") or DEFAULT_CACHE_DIR)
        client_key = json.dumps(
            {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "offline": bool(options.get("offline", True)),
                "device": options.get("device"),
            },
            sort_keys=True,
        )
        with self._cache_lock:
            client = self._esm_clients.get(client_key)
            if client is None:
                client = ESMClient(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    offline=bool(options.get("offline", True)),
                    device=options.get("device"),
                )
                self._esm_clients[client_key] = client
        sequences = [str(sequence) for sequence in payload["sequences"]]
        with self._cache_lock:
            scores = {
                sequence: self._esm_score_cache[(client_key, sequence)]
                for sequence in sequences
                if (client_key, sequence) in self._esm_score_cache
            }
        missing = [sequence for sequence in dict.fromkeys(sequences) if sequence not in scores]
        if missing:
            calculated = client.calculate_pseudo_log_likelihood_batch(
                missing,
                batch_size=int(options.get("batch_size", 256)),
            )
            scores.update({sequence: float(score) for sequence, score in zip(missing, calculated, strict=True)})
            with self._cache_lock:
                for sequence in missing:
                    self._esm_score_cache[(client_key, sequence)] = scores[sequence]
                while len(self._esm_score_cache) > ESM_SCORE_CACHE_LIMIT:
                    self._esm_score_cache.pop(next(iter(self._esm_score_cache)))
        return {"scores": [scores[sequence] for sequence in sequences]}

    def _generate_soluble_mpnn(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.soluble_mpnn import SolubleMPNNClient

        params = dict(payload.get("parameters") or {})
        constructor_params = {
            "weights_path": params.pop("weights_path", None),
            "backbone_noise": float(params.pop("backbone_noise", 0.0)),
            "device": resolve_device(params.pop("device", None)),
            "seed": params.pop("seed", None),
        }
        client_key = json.dumps(
            constructor_params,
            sort_keys=True,
        )
        with self._cache_lock:
            client = self._soluble_mpnn_clients.get(client_key)
            if client is None:
                client = SolubleMPNNClient(**constructor_params)
                self._soluble_mpnn_clients[client_key] = client
        parent_chains = {
            str(chain): str(sequence).upper() for chain, sequence in (payload.get("parent_chains") or {}).items()
        }
        if not parent_chains:
            raise ValueError("SolubleMPNN requires parent_chains for chain-safe design")
        mutable_by_chain: dict[str, set[int]] = {}
        for item in payload["mutable_positions"]:
            value = str(item)
            if ":" not in value:
                raise ValueError(f"SolubleMPNN mutable position lacks chain ID: {value!r}")
            chain_id, position_text = value.rsplit(":", 1)
            mutable_by_chain.setdefault(chain_id, set()).add(int(position_text))
        unknown_chains = sorted(set(mutable_by_chain) - set(parent_chains))
        if unknown_chains:
            raise ValueError("SolubleMPNN mutable positions reference unknown chains: " + ", ".join(unknown_chains))

        for chain_id, positions in mutable_by_chain.items():
            if any(position < 0 or position >= len(parent_chains[chain_id]) for position in positions):
                raise ValueError(f"SolubleMPNN mutable position out of range on chain {chain_id}")
        raw_explicit = params.pop("design_positions", None)
        if raw_explicit is not None and not isinstance(raw_explicit, Mapping):
            raise ValueError("SolubleMPNN design_positions must be a chain-to-position mapping")
        if raw_explicit is not None and set(raw_explicit) - set(parent_chains):
            raise ValueError("SolubleMPNN design_positions references unknown chains")
        explicit_by_chain = (
            {str(chain): {int(position) for position in positions} for chain, positions in raw_explicit.items()}
            if isinstance(raw_explicit, Mapping)
            else None
        )
        temperature = float(params.pop("temperature", 0.1))
        relax_radius = int(params.pop("relax_radius", 3))
        wt_bias = float(params.pop("wt_bias", 2.0))
        omit_aas = str(params.pop("omit_aas", "") or "")
        bias_aas = params.pop("bias_aas", None)
        if params:
            raise ValueError("unsupported SolubleMPNN parameter(s): " + ", ".join(sorted(params)))
        if not 0 <= relax_radius <= 8:
            raise ValueError("SolubleMPNN relax_radius must be between 0 and 8")
        if not 0.0 <= wt_bias <= 20.0:
            raise ValueError("SolubleMPNN wt_bias must be between 0 and 20")

        anchors_by_chain: dict[str, list[tuple[str, int, str, str]]] = {}
        for raw in payload.get("anchor_mutations") or []:
            chain_id = str(raw.get("chain_id") or raw.get("chain") or "")
            position = int(raw.get("position", raw.get("pos", -1)))
            to_aa = str(raw.get("to_aa") or raw.get("aa") or "").upper()
            if chain_id not in parent_chains or position not in mutable_by_chain.get(chain_id, set()):
                raise ValueError(f"SolubleMPNN anchor targets an immutable or unknown residue: {chain_id}:{position}")
            if to_aa not in CANONICAL_AMINO_ACIDS:
                raise ValueError(f"SolubleMPNN anchor has invalid amino acid: {to_aa!r}")
            anchors_by_chain.setdefault(chain_id, []).append(
                (chain_id, position, parent_chains[chain_id][position], to_aa)
            )

        chain_specs: dict[str, dict[str, Any]] = {}
        for chain_id, parent_sequence in parent_chains.items():
            allowed = mutable_by_chain.get(chain_id, set())
            anchors = anchors_by_chain.get(chain_id, [])
            target_positions = {item[1] for item in anchors}
            if explicit_by_chain is not None:
                designable_zero = set(explicit_by_chain.get(chain_id, set())) | target_positions
            elif anchors:
                designable_zero = set(target_positions)
                for target_position in target_positions:
                    designable_zero.update(
                        position
                        for position in range(target_position - relax_radius, target_position + relax_radius + 1)
                        if position in allowed
                    )
            else:
                designable_zero = set(allowed)
            illegal = sorted(designable_zero - allowed)
            if illegal:
                raise ValueError(f"SolubleMPNN design positions target immutable residues {chain_id}:{illegal}")
            if not designable_zero:
                continue
            forced = {position: to_aa for _chain, position, _old, to_aa in anchors}
            forced_sequence = list(parent_sequence)
            for position, residue in forced.items():
                forced_sequence[position] = residue
            bias_by_res = {
                position + 1: {forced_sequence[position]: wt_bias}
                for position in designable_zero
                if position not in forced
            }
            chain_specs[chain_id] = {
                "parent_sequence": parent_sequence,
                "forced_residues": forced,
                "bias_by_res": bias_by_res,
                "designable_positions": sorted(designable_zero),
            }
        if not chain_specs:
            raise ValueError("SolubleMPNN has no legal CDR positions to design")

        count = int(payload.get("num_sequences", 8))
        per_chain_results = {
            chain_id: client.design(
                pdb_path=str(payload["structure_path"]),
                chain=chain_id,
                designable_positions=spec["designable_positions"],
                forced_residues=spec["forced_residues"],
                omit_aas=omit_aas,
                bias_aas=bias_aas,
                bias_by_res=spec["bias_by_res"],
                temperature=temperature,
                num_sequences=count,
                expected_length=len(spec["parent_sequence"]),
            )
            for chain_id, spec in chain_specs.items()
        }
        candidate_count = min(count, *(len(results) for results in per_chain_results.values()))
        candidates: list[dict[str, Any]] = []
        for index in range(candidate_count):
            chains = dict(parent_chains)
            scores: dict[str, float] = {}
            seqids: dict[str, float] = {}
            for chain_id, results in per_chain_results.items():
                result = results[index]
                sequence = str(result["sequence"]).upper()
                if len(sequence) != len(parent_chains[chain_id]):
                    raise ValueError(
                        f"SolubleMPNN returned wrong length for chain {chain_id}: "
                        f"{len(sequence)} != {len(parent_chains[chain_id])}"
                    )
                spec = chain_specs[chain_id]
                allowed = set(spec["designable_positions"])
                if any(sequence[p] != aa for p, aa in enumerate(parent_chains[chain_id]) if p not in allowed):
                    raise ValueError(f"SolubleMPNN changed a fixed residue on chain {chain_id}")
                if any(sequence[p] != aa for p, aa in spec["forced_residues"].items()):
                    raise ValueError(f"SolubleMPNN changed a forced anchor on chain {chain_id}")
                chains[chain_id] = sequence
                scores[chain_id] = float(result.get("mpnn_score", math.nan))
                seqids[chain_id] = float(result.get("mpnn_seqid", math.nan))
            candidates.append(
                {
                    "chains": chains,
                    "sequence": "".join(chains.values()),
                    "metadata": {
                        "inverse_folding_model": "soluble_mpnn",
                        "soluble_mpnn_weights_sha256": client.weights_sha256,
                        "soluble_mpnn_score_definition": "mean negative log probability over designed residues; lower is better",
                        "soluble_mpnn_scores": scores,
                        "soluble_mpnn_seqids": seqids,
                        "design_positions": {
                            chain: spec["designable_positions"] for chain, spec in chain_specs.items()
                        },
                    },
                }
            )
        return {"candidates": self._jsonable(candidates)}

    def _evolution_tree(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_path = str(payload["candidates_json_path"])
        if requested_path == "in-memory":
            candidates, _ = self._history_state(payload)
            if not candidates:
                candidates, _ = self._population_state(payload)
        else:
            candidates_path = self._resolve_output_file(requested_path, payload=payload)
            value = json.loads(candidates_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                candidates = value.get("history") or value.get("candidates") or []
            else:
                candidates = value
        if not isinstance(candidates, list):
            raise ValueError("evolution analysis candidates must be a JSON list")
        metric = str(payload.get("objective_key") or "loss")
        minimize = bool(payload.get("minimize", True))
        normalized = [
            normalize_search_candidate(
                candidate,
                objective_key=metric,
                minimize=minimize,
            )
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        task_output = self._task_output_path(payload)
        for candidate in normalized:
            raw_path = candidate.get("structure_path")
            if not raw_path:
                continue
            path = Path(str(raw_path)).expanduser().resolve()
            try:
                path.relative_to(task_output)
            except ValueError:
                candidate["structure_path"] = None
                continue
            if not path.is_file() or path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
                candidate["structure_path"] = None
                continue
            candidate["structure_path"] = str(path)

        from opendde_harness.plugin.protein_design.servers.backends.foldmason import (
            run_structural_tree,
        )

        structural = run_structural_tree(
            normalized,
            task_output=task_output,
            cycle=int(payload.get("cycle", 0)),
            binder_chain_ids=[str(value) for value in payload.get("binder_chain_ids") or []],
            cdr_regions={
                str(chain): [int(position) for position in positions]
                for chain, positions in (payload.get("cdr_regions") or {}).items()
            },
            reflection_interval=int(payload.get("reflection_interval", 20)),
            current_parent_id=payload.get("current_parent_id"),
            minimize=minimize,
        )
        tree_paths = {
            str(chain): str(value["tree_path"])
            for chain, value in (structural.get("chains") or {}).items()
            if value.get("tree_path")
        }
        alignment_paths = {
            str(chain): str(value["alignment_path"])
            for chain, value in (structural.get("chains") or {}).items()
            if value.get("alignment_path")
        }
        from .backends import evolution_analysis as evolution_module

        result = evolution_module.analyze_search_history(
            normalized,
            metric="objective",
            minimize=minimize,
            current_parent_id=payload.get("current_parent_id"),
            tree_paths=tree_paths,
            alignment_paths=alignment_paths,
        )
        result["structural_analysis"] = structural
        return {"available": True, "result": result}

    def _epitope_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .backends import epitope_analysis as module

        structure = self._resolve_output_file(payload["structure_path"], {".pdb", ".cif", ".mmcif"}, payload=payload)
        cdr_regions = {
            str(chain): {str(name): {int(position) for position in positions} for name, positions in regions.items()}
            for chain, regions in payload["cdr_regions"].items()
        }
        result = module.analyze_epitope(
            structure_file=str(structure),
            antibody_chains=list(payload["antibody_chains"]),
            antigen_chains=list(payload["antigen_chains"]),
            cdr_regions=cdr_regions,
            cutoff=float(payload.get("cutoff", 4.5)),
            hotspots=[
                (str(item["chain"]), int(item["position"])) if isinstance(item, dict) else tuple(item)
                for item in payload.get("hotspots") or []
            ],
        )
        return {"available": True, "result": self._jsonable(result)}

    def _structure_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.structure_analysis import (
            run_plip_structure_analysis,
        )

        structures = [
            str(self._resolve_output_file(path, {".pdb", ".cif", ".mmcif"}, payload=payload))
            for path in payload["structure_paths"]
        ]
        names = [str(name) for name in payload["candidate_names"]]
        if len(structures) != len(names):
            raise ValueError("structure_paths and candidate_names must have the same length")
        cycle_num = int(payload.get("cycle_num", 0))
        result = run_plip_structure_analysis(
            structure_files=structures,
            cycle_num=cycle_num,
            candidate_names=names,
            binder_chain_ids=list(payload["binder_chain_ids"]),
            target_chain_ids=list(payload["target_chain_ids"]),
            output_dir=str(self._task_output_path(payload) / "agent" / "structure_analysis" / f"cycle_{cycle_num}"),
        )
        return result

    def _protrek_sequence_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.protrek import (
            GRADIO_AVAILABLE,
            search_protrek_sequence,
        )

        if not GRADIO_AVAILABLE:
            return {
                "available": False,
                "result": None,
                "error": "ProTrek client is missing from the compute environment.",
            }
        return _optional_external(
            PROTREK_SERVICE,
            protrek_endpoint(),
            lambda: search_protrek_sequence(
                sequence=str(payload["sequence"]),
                topk=int(payload.get("topk", 5)),
            ),
        )

    def _protrek_structure_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        structure = self._resolve_output_file(payload["structure_path"], {".pdb", ".cif", ".mmcif"}, payload=payload)
        from opendde_harness.plugin.protein_design.servers.backends.protrek import (
            GRADIO_AVAILABLE,
            search_protrek_structure,
        )

        if not GRADIO_AVAILABLE:
            return {
                "available": False,
                "result": None,
                "error": "ProTrek client is missing from the compute environment.",
            }
        return _optional_external(
            PROTREK_SERVICE,
            protrek_endpoint(),
            lambda: search_protrek_structure(
                pdb_file_path=str(structure),
                chain=str(payload.get("chain", "A")),
                topk=int(payload.get("topk", 5)),
            ),
        )

    def _generate_esm2_guided(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.esm_backend import (
            DEFAULT_CACHE_DIR,
            DEFAULT_MODEL_NAME,
            ESMClient,
        )

        options = dict(payload.get("options") or {})
        model_name = str(options.get("model_name", DEFAULT_MODEL_NAME))
        cache_dir = str(options.get("cache_dir") or DEFAULT_CACHE_DIR)
        constructor = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "offline": bool(options.get("offline", True)),
            "device": resolve_device(options.get("device")),
        }
        client_key = json.dumps(constructor, sort_keys=True)
        with self._cache_lock:
            client = self._esm_clients.get(client_key)
            if client is None:
                client = ESMClient(**constructor)
                self._esm_clients[client_key] = client
        parent_chains = {str(chain): str(sequence) for chain, sequence in payload["parent_chains"].items()}
        if any("X" in sequence.upper() for sequence in parent_chains.values()):
            return {"available": False, "result": {"candidates": []}, "error": "parent contains X"}
        mutable = {
            str(chain): {int(position) for position in positions}
            for chain, positions in payload["mutable_positions"].items()
        }
        min_llr = float(payload.get("min_llr", 0.0))
        candidates: list[dict[str, Any]] = []
        for chain_id, sequence in parent_chains.items():
            rows = client.calculate_deep_mutation_scan(
                sequence,
                batch_size=int(options.get("batch_size", 256)),
                scoring_mode="llr",
            )
            for row in rows:
                position_one = int(row.get("Pos", 0))
                position_zero = position_one - 1
                wild_type = str(row.get("WT", "")).upper()
                if position_zero not in mutable.get(chain_id, set()):
                    continue
                if not 0 <= position_zero < len(sequence) or sequence[position_zero].upper() != wild_type:
                    continue
                for amino_acid in sorted(CANONICAL_AMINO_ACIDS):
                    if amino_acid == wild_type or amino_acid not in row:
                        continue
                    llr = float(row[amino_acid])
                    if not math.isfinite(llr) or llr < min_llr:
                        continue
                    chains = dict(parent_chains)
                    residues = list(chains[chain_id])
                    residues[position_zero] = amino_acid
                    chains[chain_id] = "".join(residues)
                    candidate_id = f"esm2_{chain_id}{position_one}{wild_type}{amino_acid}"
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "parent_id": str(payload["parent_id"]),
                            "chains": chains,
                            "sequence": chains[chain_id],
                            "mutations": [[chain_id, position_zero, amino_acid]],
                            "strategy": f"ESM-2 DMS positive LLR at {chain_id}{position_one}: {wild_type}->{amino_acid}",
                            "risk_level": "low",
                            "metadata": {
                                "proposal_source": "esm2_dms",
                                "esm2_llr": round(llr, 6),
                                "chain_id": chain_id,
                                "position_zero_based": position_zero,
                                "position_one_based": position_one,
                                "skill_id": "esm2-guided-mutation",
                            },
                        }
                    )
        candidates.sort(key=lambda item: (-float(item["metadata"]["esm2_llr"]), item["candidate_id"]))
        selected: list[dict[str, Any]] = []
        used_sites: set[tuple[str, int]] = set()
        limit = int(payload.get("num_sequences", 8))
        for require_new_site in (True, False):
            for candidate in candidates:
                site = (
                    str(candidate["metadata"]["chain_id"]),
                    int(candidate["metadata"]["position_zero_based"]),
                )
                if candidate in selected or (require_new_site and site in used_sites):
                    continue
                selected.append(candidate)
                used_sites.add(site)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
        return {"available": True, "result": {"candidates": selected}}

    def _population_get(self, payload: dict[str, Any]) -> dict[str, Any]:
        population, _ = self._population_state(payload)
        return {"candidates": list(population)}

    def _population_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        population, population_path = self._population_state(payload)
        lock = self._population_locks.setdefault(population_path, threading.RLock())
        with lock:
            by_id = {} if payload.get("replace") else {str(item.get("candidate_id")): item for item in population}
            for candidate in payload.get("candidates") or []:
                by_id[str(candidate.get("candidate_id"))] = candidate
            population[:] = by_id.values()
            self._atomic_json_write(population_path, population)
            history, history_path = self._history_state(payload)
            history_by_id = {str(item.get("candidate_id")): item for item in history if item.get("candidate_id")}
            for candidate in payload.get("history_candidates") or []:
                candidate_id = str(candidate.get("candidate_id") or "")
                if candidate_id:
                    history_by_id[candidate_id] = candidate
            history[:] = history_by_id.values()
            self._atomic_json_write(history_path, history)
            if payload.get("final_selection") is not None:
                self._atomic_json_write(
                    population_path.with_name("final_selection.json"),
                    payload["final_selection"],
                )
        return {"candidate_count": len(population)}

    def _population_top(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_k = int(payload.get("top_k", 20))
        population, _ = self._population_state(payload)
        minimize = bool(payload.get("minimize", True))
        scored = sorted(
            population,
            key=lambda item: float("inf") if item.get("objective") is None else float(item["objective"]),
            reverse=not minimize,
        )
        return {"candidates": scored[:top_k]}

    def _developability(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.servers.backends.developability_filter import (
            check_antibody_developability,
        )

        binder_chain_ids = list(payload.get("binder_chain_ids") or [])
        heavy_chain_id = str(binder_chain_ids[0]) if binder_chain_ids else ""
        output_dir = payload.get("output_dir") or str(self._task_output_path(payload) / "developability")
        results: dict[str, Any] = {}
        for candidate in payload.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "candidate")
            try:
                results[candidate_id] = check_antibody_developability(
                    heavy_chain=str(candidate.get("sequence") or ""),
                    name=candidate_id,
                    output_dir=str(output_dir),
                    structure_path=str(candidate.get("structure_path") or ""),
                    heavy_chain_id=heavy_chain_id,
                )
            except Exception as exc:
                results[candidate_id] = {
                    "available": False,
                    "error": str(exc),
                }
        return {"summary": results, "results": results}

    def _structure_read(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_path = payload.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("structure path is required")
        path = Path(raw_path).expanduser().resolve()
        allowed_root = self._task_output_path(payload) if payload.get("task_id") else self._output_path
        try:
            path.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("structure path is outside protein-design output root or requested task") from exc
        if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError(f"unsupported structure format: {path.suffix or '<none>'}")
        if not path.is_file():
            raise ValueError("structure path is not a regular file")
        byte_count = path.stat().st_size
        if byte_count > self._max_structure_bytes:
            raise ValueError(f"structure file exceeds {self._max_structure_bytes} bytes")
        return {
            "path": str(path),
            "filename": path.name,
            "format": path.suffix.lower().removeprefix("."),
            "byte_count": byte_count,
            "text": path.read_text(encoding="utf-8"),
        }

    def _pose_rmsd(self, payload: dict[str, Any]) -> dict[str, Any]:
        from opendde_harness.plugin.protein_design.core.gate import target_aligned_binder_rmsd

        reference = self._validated_structure_path(payload.get("reference_path"), payload=payload)
        mobile = self._validated_structure_path(payload.get("mobile_path"), payload=payload)
        return target_aligned_binder_rmsd(
            reference,
            mobile,
            target_chain_ids=[str(item) for item in payload.get("target_chain_ids") or []],
            binder_chain_ids=[str(item) for item in payload.get("binder_chain_ids") or []],
        )

    def _validated_structure_path(
        self,
        raw_path: Any,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("structure path is required")
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(self._allowed_input_root(payload))
        except ValueError as exc:
            raise ValueError("structure path is outside the requested protein-design task") from exc
        if not path.is_file():
            raise ValueError("structure path is not a regular file")
        return path

    def _resolve_output_file(
        self,
        value: str,
        suffixes: set[str] | None = None,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Path:
        path = Path(value).expanduser().resolve()
        try:
            path.relative_to(self._allowed_input_root(payload))
        except ValueError as exc:
            raise ValueError("input path is outside the requested protein-design task") from exc
        if not path.is_file():
            raise ValueError(f"input path is not a regular file: {path}")
        if suffixes is not None and path.suffix.lower() not in suffixes:
            raise ValueError(f"unsupported input format: {path.suffix or '<none>'}")
        return path

    def _allowed_input_root(self, payload: Mapping[str, Any] | None) -> Path:
        if payload and payload.get("task_id"):
            return self._task_output_path(payload)
        return self._output_path

    def _task_output_path(self, payload: Mapping[str, Any]) -> Path:
        task_id = str(payload.get("task_id") or "").strip()
        if task_id and Path(task_id).name != task_id:
            raise ValueError("invalid protein-design task_id")
        task_output = (self._output_path / task_id).resolve() if task_id else self._output_path
        try:
            task_output.relative_to(self._output_path)
        except ValueError as exc:
            raise ValueError("invalid protein-design task_id") from exc
        task_output.mkdir(parents=True, exist_ok=True)
        return task_output

    def _population_state(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], Path]:
        population_path = self._task_output_path(payload) / "population.json"
        if population_path not in self._populations:
            population: list[dict[str, Any]] = []
            if population_path.is_file():
                value = json.loads(population_path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    population = value
            self._populations[population_path] = population
        return self._populations[population_path], population_path

    def _history_state(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], Path]:
        history_path = self._task_output_path(payload) / "search_history.json"
        if history_path not in self._histories:
            history: list[dict[str, Any]] = []
            if history_path.is_file():
                value = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    history = value
            self._histories[history_path] = history
        return self._histories[history_path], history_path

    @staticmethod
    def _atomic_json_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _candidate_chains(candidate: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(options.get("target_chains") or {})
        chains = candidate.get("chains")
        if not isinstance(chains, dict) or not chains:
            metadata = candidate.get("metadata") or {}
            chains = metadata.get("chains") if isinstance(metadata, Mapping) else None
        if isinstance(chains, dict) and chains:
            binder_options = options.get("binder_chain_options") or {}
            for chain_id, value in chains.items():
                chain_key = str(chain_id)
                template = binder_options.get(chain_key) if isinstance(binder_options, Mapping) else None
                if isinstance(template, Mapping):
                    merged[chain_key] = {
                        **dict(template),
                        "sequence": PythonProteinDesignHarness._chain_sequence(value),
                    }
                else:
                    merged[chain_key] = value
            return merged
        binder_chain_ids = [str(value) for value in options.get("binder_chain_ids") or []]
        if len(binder_chain_ids) > 1:
            candidate_id = str(candidate.get("candidate_id") or "<unknown>")
            raise ValueError(
                f"candidate {candidate_id!r} is missing per-chain binder sequences; "
                "a flat sequence cannot be assigned safely to multiple binder chains"
            )
        binder_chain = binder_chain_ids[0] if binder_chain_ids else str(options.get("binder_chain", "D"))
        binder_options = options.get("binder_chain_options") or {}
        template = binder_options.get(binder_chain) if isinstance(binder_options, Mapping) else None
        if isinstance(template, Mapping):
            merged[binder_chain] = {
                **dict(template),
                "sequence": PythonProteinDesignHarness._chain_sequence(candidate["sequence"]),
            }
        else:
            merged[binder_chain] = candidate["sequence"]
        return merged

    @staticmethod
    def _binder_sequence(candidate: dict[str, Any], options: dict[str, Any]) -> str:
        chains = candidate.get("chains") or {}
        return PythonProteinDesignHarness._chain_sequence(chains.get(str(options.get("binder_chain", "D")), ""))

    @staticmethod
    def _chain_sequence(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("sequence") or "")
        return str(value or "")

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (set, frozenset)):
            return [cls._jsonable(item) for item in sorted(value, key=str)]
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        return value

    @staticmethod
    def _first_path(value: Any) -> str | None:
        if isinstance(value, list):
            return str(value[0]) if value else None
        return str(value) if value else None


def build_harness_from_environment() -> PythonProteinDesignHarness:
    return PythonProteinDesignHarness(output_path=os.environ.get("PROTEIN_DESIGN_OUTPUT_PATH"))
