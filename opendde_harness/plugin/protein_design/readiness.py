"""Setup readiness for the protein-design plugin."""

from __future__ import annotations

from typing import Any


def compute_configured(config: dict[str, Any]) -> bool:
    """True once a compute service is configured, by URL or by local workers."""
    return bool(config.get("compute_url") or config.get("compute_workers"))
