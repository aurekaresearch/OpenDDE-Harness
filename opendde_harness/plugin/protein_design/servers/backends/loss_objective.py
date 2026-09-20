"""Local, gradient-free aggregation of the configurable design loss.

Structural components come from the configured fold model's saved confidence
outputs from OpenDDE. The objective minimizes their weighted sum and
subtracts the ESM-2 pseudo-log-likelihood. This module imports neither an
external design package nor an AlphaFold/JAX design runtime.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from opendde_harness.plugin.protein_design.servers.backends.pyrosetta_analysis import PYROSETTA_METRICS

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
COMPOSITE_LOSS_OBJECTIVE_VERSION = "confidence-contact8-esm2-rosetta-v1"
BOUNDED_LOSS_OBJECTIVE_VERSION = "bounded-fixed-grouped-v1"

# Raw reported confidence metrics, distinct from the normalized structural i_pae.
CONFIDENCE_LOSS_DIRECTIONS = {"min_ipae": "minimize", "ipsae": "maximize"}


class MetricLossTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    weight: float = Field(default=1.0, ge=0)
    direction: Literal["minimize", "maximize"]
    scale: float = Field(default=1.0, gt=0)
    reference: float = 0.0


class LossAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    good: float
    bad: float


class LossGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    budget: float = Field(gt=0, le=1)
    terms: list[str] = Field(min_length=1)


class BoundedLossCombination(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    mode: Literal["bounded_grouped"]
    calibration_id: str = Field(min_length=1)
    groups: dict[str, LossGroup] = Field(min_length=1)
    anchors: dict[str, LossAnchor] = Field(min_length=1)


def normalize_metric_loss_terms(terms: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if terms is None:
        return {}
    if not isinstance(terms, Mapping):
        raise ValueError("metric_loss_terms must be a mapping")
    unknown = set(terms) - (set(PYROSETTA_METRICS) | set(CONFIDENCE_LOSS_DIRECTIONS))
    if unknown:
        raise ValueError(f"Unsupported metric loss terms: {sorted(unknown)}")
    normalized = {name: MetricLossTerm.model_validate(term).model_dump() for name, term in terms.items()}
    for name, direction in CONFIDENCE_LOSS_DIRECTIONS.items():
        if name in normalized and normalized[name]["direction"] != direction:
            raise ValueError(f"Metric loss term {name} requires direction: {direction}")
    return normalized


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Loss component {name!r} must be numeric, not boolean")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"Loss component {name!r} must be finite") from exc
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
        if not isinstance(weights, Mapping):
            raise ValueError("loss_weights must be a mapping")
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


def normalize_loss_combination(
    value: Mapping[str, Any] | None,
    *,
    weights: Mapping[str, float] | None = None,
    metric_terms: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate opt-in fixed anchors without deriving ranges from candidate data."""
    if value is None:
        return None
    combination = BoundedLossCombination.model_validate(value)
    if not combination.calibration_id.strip() or any(not name.strip() for name in combination.groups):
        raise ValueError("Loss calibration and group names must not be blank")
    configured_weights = normalize_loss_weights(weights)
    configured_metrics = normalize_metric_loss_terms(metric_terms)
    active_weights = {name: weight for name, weight in configured_weights.items() if weight > 0}
    active_weights.update({name: term["weight"] for name, term in configured_metrics.items() if term["weight"] > 0})
    if set(combination.anchors) != set(active_weights):
        raise ValueError("loss_combination anchors must cover every enabled component exactly, without extra terms")
    assigned = [term for group in combination.groups.values() for term in group.terms]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(active_weights):
        raise ValueError("loss_combination groups must assign every enabled component exactly once")
    budget_sum = math.fsum(group.budget for group in combination.groups.values())
    if not math.isclose(budget_sum, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("loss_combination group budgets must sum to 1")
    for name, anchor in combination.anchors.items():
        direction = (
            configured_metrics[name]["direction"]
            if name in configured_metrics
            else ("maximize" if name == "esm2" else "minimize")
        )
        if (direction == "minimize" and anchor.good >= anchor.bad) or (
            direction == "maximize" and anchor.good <= anchor.bad
        ):
            raise ValueError(f"loss_combination anchors for {name} must match direction {direction}")
        _finite(anchor.bad - anchor.good, f"{name} anchor span")
    for group in combination.groups.values():
        _finite(sum(active_weights[name] for name in group.terms), "group coefficient sum")
    return combination.model_dump()


def calculate_loss_objective(
    structure_components: Mapping[str, Any],
    esm2_pll: Any,
    *,
    weights: Mapping[str, float] | None = None,
    metric_values: Mapping[str, Any] | None = None,
    metric_terms: Mapping[str, Any] | None = None,
    loss_combination: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the minimized base loss plus optional signed metric contributions.

    ``esm2_pll`` is the mean pseudo-log-likelihood used by this project and is
    normally non-positive.  Subtracting it therefore penalizes less-natural
    sequences, using the established ``loss - scale * lm_ll`` rule.
    Every enabled structural component is required; missing values must never
    be silently replaced by a different backend's ranking score.
    ``base_loss`` retains the confidence/ESM-2 objective before metric terms;
    ``loss`` is the composite consumed by selection. Neither is constrained
    to be positive, and metric references are offsets rather than gates.
    """
    active_weights = normalize_loss_weights(weights)
    combination = normalize_loss_combination(loss_combination, weights=active_weights, metric_terms=metric_terms)
    lm_weight = active_weights.pop("esm2")
    active_terms = normalize_metric_loss_terms(metric_terms)

    contributions: dict[str, dict[str, Any]] = {}
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
    metric_loss = 0.0
    for name, term in active_terms.items():
        if term["weight"] == 0:
            continue
        if name not in (metric_values or {}):
            raise ValueError(f"Missing required loss metric: {name}")
        value = metric_values[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Loss metric {name!r} must be numeric")
        raw = _finite(value, name)
        if name == "min_ipae" and raw < 0:
            raise ValueError("Loss metric 'min_ipae' must be nonnegative")
        if name == "ipsae" and not 0 <= raw <= 1:
            raise ValueError("Loss metric 'ipsae' must be between 0 and 1")
        normalized = (raw - term["reference"]) / term["scale"]
        sign = -1.0 if term["direction"] == "maximize" else 1.0
        contribution = _finite(sign * term["weight"] * normalized, f"legacy {name}" if combination else name)
        metric_loss += contribution
        contributions[name] = {"raw": raw, **term, "normalized": normalized, "contribution": contribution}
    base_loss = _finite(structure_loss + esm2_contribution, "legacy base" if combination else "base")
    total_loss = _finite(base_loss + metric_loss, "legacy total" if combination else "total")
    result = {
        "formula_version": COMPOSITE_LOSS_OBJECTIVE_VERSION if active_terms else LOSS_OBJECTIVE_VERSION,
        "direction": "minimize",
        "structure_loss": structure_loss,
        "esm2_pll": pll,
        "esm2_weight": lm_weight,
        "esm2_contribution": esm2_contribution,
        "base_loss": base_loss,
        "loss": total_loss,
        "components": contributions,
    }
    if active_terms:
        result["metric_loss"] = metric_loss
    if combination is not None:
        return _bounded_loss(result, combination)
    return result


def _bounded_loss(legacy: dict[str, Any], combination: dict[str, Any]) -> dict[str, Any]:
    raw_components = dict(legacy["components"])
    if legacy["esm2_weight"] > 0:
        raw_components["esm2"] = {
            "raw": legacy["esm2_pll"],
            "weight": legacy["esm2_weight"],
            "direction": "maximize",
        }
    components: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for group_name, group in combination["groups"].items():
        coefficient_sum = math.fsum(raw_components[name]["weight"] for name in group["terms"])
        for name in group["terms"]:
            raw = raw_components[name]["raw"]
            weight = raw_components[name]["weight"]
            good, bad = (combination["anchors"][name][key] for key in ("good", "bad"))
            direction = raw_components[name].get("direction", "minimize")
            # Compare before dividing so finite outliers can saturate without overflow.
            if (direction == "minimize" and raw <= good) or (direction == "maximize" and raw >= good):
                penalty = 0.0
            elif (direction == "minimize" and raw >= bad) or (direction == "maximize" and raw <= bad):
                penalty = 1.0
            else:
                penalty = (raw - good) / (bad - good)
            normalized_weight = weight / coefficient_sum
            effective_weight = group["budget"] * normalized_weight
            components[name] = {
                "raw": raw,
                "weight": weight,
                "direction": direction,
                "good": good,
                "bad": bad,
                "penalty": penalty,
                "normalized": penalty,
                "group": group_name,
                "normalized_weight": normalized_weight,
                "budget": group["budget"],
                "effective_weight": effective_weight,
                "contribution": effective_weight * penalty,
            }
        groups[group_name] = {
            "budget": group["budget"],
            "terms": list(group["terms"]),
            "coefficient_sum": coefficient_sum,
            "penalty": math.fsum(
                components[name]["normalized_weight"] * components[name]["penalty"] for name in group["terms"]
            ),
            "contribution": math.fsum(components[name]["contribution"] for name in group["terms"]),
        }
    structure_loss = math.fsum(
        component["contribution"]
        for name, component in components.items()
        if name in DEFAULT_LOSS_WEIGHTS and name != "esm2"
    )
    esm2_component = components.pop("esm2", None)
    esm2_contribution = esm2_component["contribution"] if esm2_component is not None else 0.0
    metric_loss = math.fsum(
        component["contribution"] for name, component in components.items() if name not in DEFAULT_LOSS_WEIGHTS
    )
    base_loss = structure_loss + esm2_contribution
    return {
        "formula_version": BOUNDED_LOSS_OBJECTIVE_VERSION,
        "direction": "minimize",
        "structure_loss": structure_loss,
        "esm2_pll": legacy["esm2_pll"],
        "esm2_weight": legacy["esm2_weight"],
        "esm2_contribution": esm2_contribution,
        "base_loss": base_loss,
        "metric_loss": metric_loss,
        "loss": base_loss + metric_loss,
        "components": components,
        "esm2_component": esm2_component,
        "groups": groups,
        "loss_combination": combination,
        "original_base_loss": legacy["base_loss"],
        "legacy_loss": legacy["loss"],
        "legacy_loss_breakdown": legacy,
        "normalization_policy": "Fixed anchors supersede linear metric reference/scale; legacy fields are diagnostic only.",
    }
