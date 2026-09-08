"""Persistent multi-GPU inference queue for OpenDDE.

The script runs inside the existing folding Docker image.  It loads the model once,
then rank 0 receives file-backed requests while all DDP ranks execute each batch.
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import runner.inference as inference

LOGGER = logging.getLogger("persistent_inference_worker")
POLL_SECONDS = 0.25


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_configs() -> Any:
    """Build configs through the backend's own current inference API."""
    arg_str = inference.parse_sys_args()

    # Prefer the current inference config factory over legacy module-level trees.
    if hasattr(inference, "build_inference_config"):
        configs = inference.build_inference_config(
            arg_str=arg_str,
            fill_required_with_null=True,
        )
        if hasattr(inference, "update_gpu_compatible_configs"):
            configs = inference.update_gpu_compatible_configs(configs)
        if hasattr(inference, "validate_config_triangle_kernels"):
            inference.validate_config_triangle_kernels(configs)
        return configs

    first_pass = {
        **inference.configs_base,
        **{"data": inference.data_configs},
        **inference.inference_configs,
    }
    preliminary = inference.parse_configs(
        configs=first_pass,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = preliminary.model_name
    base_configs = {
        **inference.configs_base,
        **{"data": inference.data_configs},
        **inference.inference_configs,
    }

    def deep_update(target: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
        for key, value in update.items():
            if isinstance(value, Mapping) and key in target and isinstance(target[key], Mapping):
                deep_update(target[key], value)
            else:
                target[key] = value
        return target

    deep_update(base_configs, inference.model_configs[model_name])
    configs = inference.parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    return inference.update_gpu_compatible_configs(configs)


def _take_next_job(queue_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for request_path in sorted((queue_dir / "jobs").glob("*/request.json")):
        running_path = request_path.with_name("running.json")
        try:
            request_path.replace(running_path)
        except FileNotFoundError:
            continue
        try:
            return json.loads(running_path.read_text(encoding="utf-8")), running_path
        except Exception:
            _atomic_json(
                running_path.with_name("failed.json"),
                {"error": traceback.format_exc(), "finished_at": time.time()},
            )
    return None, None


def _broadcast_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = [job]
    if inference.DIST_WRAPPER.world_size > 1:
        inference.dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _run_job(runner: Any, configs: Any, job: dict[str, Any]) -> str | None:
    try:
        configs.input_json_path = job["input_json_path"]
        configs.dump_dir = job["dump_dir"]
        runner.configs = configs
        runner.dump_dir = configs.dump_dir
        runner.init_basics()
        runner.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )
        inference.infer_predict(runner, configs)
        return None
    except Exception:
        return traceback.format_exc()


def main() -> None:
    if "--probe-import" in sys.argv:
        return
    queue_value = os.environ.get("PERSISTENT_FOLD_QUEUE_DIR")
    if not queue_value:
        raise RuntimeError("PERSISTENT_FOLD_QUEUE_DIR is required")
    queue_dir = Path(queue_value)
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "jobs").mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    configs = _build_configs()
    if not hasattr(inference, "apply_runtime_compatibility"):
        inference.download_inference_cache(configs)
    runner = inference.InferenceRunner(configs)
    configs = runner.configs
    rank = inference.DIST_WRAPPER.rank
    if rank == 0:
        _atomic_json(
            queue_dir / "ready.json",
            {"pid": os.getpid(), "ready_at": time.time(), "world_size": inference.DIST_WRAPPER.world_size},
        )
    if inference.DIST_WRAPPER.world_size > 1:
        inference.dist.barrier()

    while True:
        if rank == 0 and (queue_dir / "shutdown.json").exists():
            next_job = {"shutdown": True}
            running_path = None
        elif rank == 0:
            next_job, running_path = _take_next_job(queue_dir)
        else:
            next_job = None
            running_path = None

        job = _broadcast_job(next_job)
        if job is None:
            time.sleep(POLL_SECONDS)
            continue
        if job.get("shutdown"):
            close = getattr(runner, "close", None)
            if callable(close):
                close()
            break

        error = _run_job(runner, configs, job)
        errors = [None] * inference.DIST_WRAPPER.world_size
        if inference.DIST_WRAPPER.world_size > 1:
            inference.dist.all_gather_object(errors, error)
            inference.dist.barrier()
        else:
            errors[0] = error
        if rank == 0 and running_path is not None:
            status_path = running_path.with_name("failed.json" if any(errors) else "done.json")
            _atomic_json(
                status_path,
                {
                    "job_id": job.get("job_id"),
                    "finished_at": time.time(),
                    "errors": [message for message in errors if message],
                },
            )


if __name__ == "__main__":
    main()
