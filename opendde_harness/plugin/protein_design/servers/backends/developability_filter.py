"""Developability filter stub — public build.

The objective developability backend is not included in the open-source
distribution. Assessments are delegated to the LLM agent instead.
"""

from __future__ import annotations

from typing import Any, Optional


def check_antibody_developability(
    heavy_chain: str,
    name: str = "candidate",
    output_dir: Optional[str] = None,
    structure_path: str = "",
    heavy_chain_id: str = "",
) -> dict[str, Any]:
    return {"available": False}
