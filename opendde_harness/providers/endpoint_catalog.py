"""What an OpenAI-compatible endpoint says it serves: its ``/models`` list.

A relay or a self-hosted server carries no catalogue of its own here, and
the picker used to offer such a section nothing at all. The endpoint knows:
every OpenAI-compatible server answers ``GET /models`` with the ids it
routes, so the picker asks it, once per session per endpoint, and shows
that. Context windows are not in that answer (the OpenAI shape carries
none), so a listed id still sizes as unknown until its overlay says.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

_CACHE_TTL_SECONDS = 300
_FAILURE_TTL_SECONDS = 30
_cache: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = {}
_failure: dict[tuple[str, str], float] = {}


def _key(api_base: str, api_key: str) -> tuple[str, str]:
    return (api_base.rstrip("/"), hashlib.sha256(api_key.encode()).hexdigest()[:16])


def endpoint_models(api_base: str, api_key: str, *, timeout: float = 5.0) -> tuple[str, ...]:
    """The ids the endpoint lists, in its order, or empty when it cannot be asked.

    Cached briefly, failures too: the picker rebuilds on every refresh and an
    endpoint that is down should not cost the full timeout each time.
    """
    if not api_base:
        return ()
    key = _key(api_base, api_key or "")
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    failed = _failure.get(key)
    if failed is not None and now - failed < _FAILURE_TTL_SECONDS:
        return ()
    try:
        ids = _fetch(api_base, api_key, timeout=timeout)
    except Exception:
        _failure[key] = now
        return ()
    _cache[key] = (now, ids)
    _failure.pop(key, None)
    return ids


def reset_cache() -> None:
    _cache.clear()
    _failure.clear()


def _fetch(api_base: str, api_key: str, *, timeout: float) -> tuple[str, ...]:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(f"{api_base.rstrip('/')}/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    return listed_ids(response.json())


def listed_ids(payload: Any) -> tuple[str, ...]:
    """Model ids from an OpenAI-shaped ``{"data": [{"id": ...}]}`` list, or a bare list."""
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return ()
    out: list[str] = []
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else item
        if isinstance(model_id, str) and model_id and model_id not in out:
            out.append(model_id)
    return tuple(out)
