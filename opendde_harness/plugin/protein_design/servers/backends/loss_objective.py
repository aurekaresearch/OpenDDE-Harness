"""Local, gradient-free aggregation of the configurable design loss.

Structural components come from the configured fold model's saved confidence
outputs from OpenDDE. The objective minimizes their weighted sum and
subtracts the ESM-2 pseudo-log-likelihood. This module imports neither an
external design package nor an AlphaFold/JAX design runtime.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "plddt": 1.0,
    "i_plddt": 1.0,
    "pae": 0.1,
    "i_pae": 0.5,
    "i_ptm": 1.0,
    "con": 0.1,
    "i_con": 0.1,
    "rg": 0.1,
    "dgram_cce": 0.01,
    "esm2": 0.1,
}

LOSS_OBJECTIVE_VERSION = "vhh-confidence-contact8-proxy-esm2-v2"


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Loss component {name!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"Loss component {name!r} must be finite")
    return number


def normalize_loss_weights(
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Validate weights without normalizing them.

    These are literal loss coefficients, not a weighted average, so their sum
    is intentionally not normalized to one. A partial YAML mapping overrides
    only the named defaults.
    """
    merged = dict(DEFAULT_LOSS_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(DEFAULT_LOSS_WEIGHTS)
        if unknown:
            raise ValueError(f"Unsupported loss components: {sorted(unknown)}")
        merged.update(weights)
    for name, value in merged.items():
        number = _finite(value, name)
        if number < 0.0:
            raise ValueError("Loss weights must be non-negative")
        merged[name] = number
    if not any(value > 0.0 for value in merged.values()):
        raise ValueError("Loss weights must contain at least one positive value")
    return merged


def calculate_loss_objective(
    structure_components: Mapping[str, Any],
    esm2_pll: Any,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return the minimized fold-confidence loss plus the ESM-2 term.

    ``esm2_pll`` is the mean pseudo-log-likelihood used by this project and is
    normally non-positive.  Subtracting it therefore penalizes less-natural
    sequences, using the established ``loss - scale * lm_ll`` rule.
    Every enabled structural component is required; missing values must never
    be silently replaced by a different backend's ranking score.
    """
    active_weights = normalize_loss_weights(weights)
    lm_weight = active_weights.pop("esm2")

    contributions: dict[str, dict[str, float]] = {}
    structure_loss = 0.0
    for name, weight in active_weights.items():
        if weight == 0.0:
            continue
        if name not in structure_components:
            raise ValueError(f"Missing required loss component: {name}")
        raw = _finite(structure_components[name], name)
        contribution = weight * raw
        contributions[name] = {
            "raw": raw,
            "weight": weight,
            "contribution": contribution,
        }
        structure_loss += contribution

    pll = _finite(esm2_pll, "esm2_pll")
    esm2_contribution = -lm_weight * pll
    total_loss = structure_loss + esm2_contribution
    return {
        "formula_version": LOSS_OBJECTIVE_VERSION,
        "direction": "minimize",
        "structure_loss": structure_loss,
        "esm2_pll": pll,
        "esm2_weight": lm_weight,
        "esm2_contribution": esm2_contribution,
        "loss": total_loss,
        "components": contributions,
    }
