"""Read-only, credential-free context for preparing a design."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_URL
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import DEFAULT_LOSS_WEIGHTS


def display_url(value: Any) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return None


def compute_device(saved: Mapping[str, Any]) -> str:
    gpus = saved.get("gpus")
    if gpus == "none":
        return "cpu"
    return "cuda" if gpus else "service_selected"


def fold_summary(fold: Mapping[str, Any]) -> dict[str, Any]:
    mode = fold.get("execution_mode") or "service_default"
    return {
        "execution_mode": mode,
        "api_url": display_url(fold.get("api_url")),
        "msa_policy": "service_managed" if mode == "api" else "task_configured",
        "use_msa": fold.get("use_msa"),
        "enable_msa_search": fold.get("enable_msa_search"),
        "need_atom_confidence": fold.get("need_atom_confidence", True) if mode == "api" else None,
    }


def gpu_inventory(config: Mapping[str, Any], *, timeout: float = 2.0, transport: Any = None) -> dict[str, Any]:
    """Read-only GPU snapshot of the default compute endpoint, for placement advice."""
    import httpx

    from opendde_harness.plugin.protein_design.servers.local_service import is_local_placement, read_state

    state = read_state() if is_local_placement(config) else None
    url = str(state["url"] if state else config.get("compute_url") or DEFAULT_COMPUTE_URL).rstrip("/")
    token = str(config.get("compute_token") or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    summary: dict[str, Any] = {"url": display_url(url), "available": False, "devices": []}
    try:
        with httpx.Client(timeout=timeout, trust_env=False, headers=headers, transport=transport) as client:
            payload = client.get(f"{url}/health").json()
    except (httpx.HTTPError, ValueError):
        summary["reason"] = "compute service did not answer /health"
        return summary
    if not isinstance(payload, Mapping) or not payload.get("gpu"):
        summary["reason"] = "compute service reported no GPU inventory"
        return summary
    leases = {
        int(item.get("index", -1)): item.get("jobs") or []
        for item in ((payload.get("workers") or {}).get("gpu_leases") or [])
        if isinstance(item, Mapping)
    }
    summary["available"] = True
    summary["devices"] = [
        {
            "index": int(gpu.get("index", -1)),
            "name": gpu.get("name"),
            "memory_total_mb": gpu.get("memory_total_mb"),
            "memory_free_mb": gpu.get("memory_free_mb"),
            "leases": leases.get(int(gpu.get("index", -1)), []),
        }
        for gpu in payload["gpu"]
        if isinstance(gpu, Mapping)
    ]
    return summary


def compute_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    from opendde_harness.plugin.protein_design.servers.local_service import is_local_placement, read_state

    workers = config.get("compute_workers") or []
    local = is_local_placement(config)
    state = read_state() if local else None
    if local:
        default_url = display_url(state["url"]) if state else None
    else:
        default_url = display_url(config.get("compute_url") or DEFAULT_COMPUTE_URL)
    return {
        "selection": "worker_pool" if workers else ("local_docker" if local else "default_endpoint"),
        "default_url": default_url,
        "lifecycle": "on_demand" if local else "external",
        "explicitly_configured": bool(workers or config.get("compute_url")),
        "workers": [
            {"id": item.get("id") or item.get("worker_id"), "url": display_url(item.get("url"))}
            for item in workers
            if isinstance(item, Mapping)
        ],
        "bound": False,
        "readiness": "not_checked",
    }


def preparation_context(
    config: Mapping[str, Any],
    repository_root: str | None = None,
) -> dict[str, Any]:
    saved = config.get("compute_docker") or {}
    candidates = (
        [repository_root]
        if repository_root
        else [
            os.environ.get("OPENDDE_HARNESS_PROJECT_ROOT"),
            saved.get("package_root"),
            str(Path(__file__).resolve().parents[4]),
            str(Path.cwd()),
        ]
    )
    checked: list[str] = []
    root = None
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if str(path) in checked:
            continue
        checked.append(str(path))
        if (path / "pyproject.toml").is_file() and (path / "docs/examples").is_dir():
            root = path
            break
    # Packaged first: an installed release has no checkout to find, and the
    # examples are the one context an agent reads before writing a new design.
    packaged = Path(__file__).resolve().parents[1] / "examples"
    examples = []
    for target, seed in (("CRLF2", "complete VHH"), ("CACNG1", "masked CDRs")):
        name = f"{target.lower()}_quickstart.yaml"
        path = packaged / name
        if not path.is_file():
            path = root / "docs/examples" / name if root else None
        examples.append(
            {
                "target": target,
                "seed": seed,
                "path": str(path) if path and path.is_file() else None,
                "available": bool(path and path.is_file()),
            }
        )
    return {
        "repository_root": str(root) if root else None,
        "checked_roots": checked,
        "examples": examples,
        "compute": {**compute_summary(config), "gpus": gpu_inventory(config)},
        "fold": fold_summary(config.get("fold_defaults") or {}),
        "execution": {
            "model_tools": "configured_compute_service",
            "shell": "client_host_or_configured_shell_sandbox",
            "compute_platform": f"{platform.system().lower()}/{platform.machine()}",
            "device": compute_device(saved),
            "environment_image": saved.get("image"),
            "code_mode": saved.get("code_mode"),
            "code_path": saved.get("package_root"),
            "weights_path": saved.get("weights_dir"),
            "diagnostic_command": "ddeharness doctor --compute-only",
        },
        "default_loss_weights": dict(DEFAULT_LOSS_WEIGHTS),
        "config_directory": str(Path.home() / ".opendde_harness/protein_design/configs"),
        "notes": [
            "Display URLs omit credentials, query strings and fragments; do not copy them back into configuration.",
            "Omit compute placement and fold execution_mode/api_url to inherit configured defaults unless the user requests overrides.",
            "Local Docker compute starts on demand when a task needs it and stops after an idle timeout; its URL may change between runs.",
            "Explicit YAML fold settings override defaults. API mode requires service-managed MSA and cannot use local A3M paths or use_msa=false.",
            "Only a read-only /health read of the default compute URL was performed; no model loading, worker selection or task launch.",
            "Leave compute.placement unset unless the user names GPUs; set fold to a list of compute.gpus indices, esm and mpnn to one index each.",
            "If an example is unavailable, request its checkout or YAML path; do not repeat global wildcard searches.",
        ],
    }
