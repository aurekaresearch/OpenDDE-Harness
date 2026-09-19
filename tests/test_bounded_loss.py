"""Fixed-range objective tests; numeric anchors are fixtures, not scientific defaults."""

import copy
import math
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
    BOUNDED_LOSS_OBJECTIVE_VERSION,
    DEFAULT_LOSS_WEIGHTS,
    calculate_loss_objective,
    normalize_loss_combination,
)


def settings():
    weights = {name: 0.0 for name in DEFAULT_LOSS_WEIGHTS}
    weights.update(plddt=2.0, i_ptm=1.0, esm2=1.0)
    terms = {
        "rosetta_interface_dg": {"weight": 2.0, "direction": "minimize", "scale": 10.0, "reference": -10.0},
        "rosetta_interface_sc": {"weight": 1.0, "direction": "maximize", "reference": 0.5},
    }
    combination = {
        "mode": "bounded_grouped",
        "calibration_id": "unit-test-only-not-scientifically-calibrated",
        "groups": {
            "confidence": {"budget": 0.4, "terms": ["plddt", "i_ptm"]},
            "sequence": {"budget": 0.1, "terms": ["esm2"]},
            "interface": {"budget": 0.5, "terms": ["rosetta_interface_dg", "rosetta_interface_sc"]},
        },
        "anchors": {
            "plddt": {"good": 0.0, "bad": 1.0},
            "i_ptm": {"good": 0.0, "bad": 1.0},
            "esm2": {"good": -1.0, "bad": -5.0},
            "rosetta_interface_dg": {"good": -50.0, "bad": 0.0},
            "rosetta_interface_sc": {"good": 0.8, "bad": 0.3},
        },
    }
    return weights, terms, combination


def score(*, raw=None, pll=-3.0, metrics=None, combination=None, weights=None, terms=None):
    default_weights, default_terms, default_combination = settings()
    return calculate_loss_objective(
        {"plddt": 0.25, "i_ptm": 0.75} if raw is None else raw,
        pll,
        weights=default_weights if weights is None else weights,
        metric_values={"rosetta_interface_dg": -25.0, "rosetta_interface_sc": 0.55} if metrics is None else metrics,
        metric_terms=default_terms if terms is None else terms,
        loss_combination=default_combination if combination is None else combination,
    )


def test_bounded_arithmetic_preserves_full_linear_audit():
    weights, terms, combination = settings()
    bounded = score()
    legacy = calculate_loss_objective(
        {"plddt": 0.25, "i_ptm": 0.75},
        -3.0,
        weights=weights,
        metric_terms=terms,
        metric_values={"rosetta_interface_dg": -25.0, "rosetta_interface_sc": 0.55},
    )
    assert bounded["formula_version"] == BOUNDED_LOSS_OBJECTIVE_VERSION
    assert bounded["structure_loss"] == pytest.approx(0.4 * (2 / 3 * 0.25 + 1 / 3 * 0.75))
    assert bounded["esm2_contribution"] == pytest.approx(0.1 * 0.5)
    assert bounded["metric_loss"] == pytest.approx(0.5 * 0.5)
    assert bounded["loss"] == pytest.approx(7 / 15)
    assert bounded["loss"] == bounded["base_loss"] + bounded["metric_loss"]
    assert bounded["original_base_loss"] == legacy["base_loss"]
    assert bounded["legacy_loss"] == legacy["loss"]
    assert bounded["legacy_loss_breakdown"] == legacy
    assert bounded["loss_combination"] == combination
    assert "supersede" in bounded["normalization_policy"]
    all_components = {**bounded["components"], "esm2": bounded["esm2_component"]}
    for name, group in bounded["groups"].items():
        rows = [item for item in all_components.values() if item["group"] == name]
        assert sum(row["normalized_weight"] for row in rows) == pytest.approx(1.0)
        assert sum(row["contribution"] for row in rows) == pytest.approx(group["contribution"])
        assert group["contribution"] == pytest.approx(group["budget"] * group["penalty"])
        assert 0 <= group["contribution"] <= group["budget"]
    for name, component in all_components.items():
        assert component["penalty"] == pytest.approx(
            (component["raw"] - combination["anchors"][name]["good"])
            / (combination["anchors"][name]["bad"] - combination["anchors"][name]["good"])
        )
        assert component["contribution"] == component["effective_weight"] * component["penalty"]


@pytest.mark.parametrize("extreme,expected", [("good", 0.0), ("bad", 1.0)])
def test_bounded_saturates_at_fixed_good_bad_anchors(extreme, expected):
    _, _, config = settings()
    sign = -1 if extreme == "good" else 1
    result = score(
        raw={"plddt": sign * 1e6, "i_ptm": sign * 1e6},
        pll=-sign * 1e6,
        metrics={"rosetta_interface_dg": sign * 1e6, "rosetta_interface_sc": -sign * 1e6},
    )
    assert result["loss"] == pytest.approx(expected)
    assert all(item["penalty"] == expected for item in result["components"].values())
    assert result["esm2_component"]["penalty"] == expected
    assert result["loss_combination"] == config


def test_metric_unit_conversion_does_not_change_bounded_priority():
    _, terms, config = settings()
    original = score()
    config["anchors"]["rosetta_interface_dg"] = {"good": -50000.0, "bad": 0.0}
    changed = score(metrics={"rosetta_interface_dg": -25000.0, "rosetta_interface_sc": 0.55}, combination=config)
    assert changed["loss"] == original["loss"]
    assert changed["legacy_loss"] != original["legacy_loss"]
    terms["rosetta_interface_dg"].update(scale=100000.0, reference=-1234.0)
    diagnostic_only = score(terms=terms)
    assert diagnostic_only["loss"] == original["loss"]
    assert diagnostic_only["legacy_loss"] != original["legacy_loss"]


def test_group_budget_caps_outliers_and_duplicate_correlated_terms_do_not_expand_budget():
    low = score(metrics={"rosetta_interface_dg": -1e6, "rosetta_interface_sc": 0.55})
    high = score(metrics={"rosetta_interface_dg": 1e6, "rosetta_interface_sc": 0.55})
    assert high["loss"] - low["loss"] == pytest.approx(0.5 * 2 / 3)
    weights, terms, config = settings()
    terms["rosetta_interface_unsat_hbonds"] = {"weight": 1.0, "direction": "minimize"}
    config["groups"]["interface"]["terms"].append("rosetta_interface_unsat_hbonds")
    config["anchors"]["rosetta_interface_unsat_hbonds"] = {"good": 0.0, "bad": 20.0}
    added = score(
        metrics={"rosetta_interface_dg": -25.0, "rosetta_interface_sc": 0.55, "rosetta_interface_unsat_hbonds": 10.0},
        weights=weights,
        terms=terms,
        combination=config,
    )
    assert added["loss"] == pytest.approx(score()["loss"])
    assert added["groups"]["interface"]["budget"] == 0.5


def test_positive_coefficient_rescaling_inside_a_group_does_not_change_bounded_score():
    weights, terms, config = settings()
    weights["plddt"] *= 100
    weights["i_ptm"] *= 100
    assert score(weights=weights, terms=terms, combination=config)["loss"] == score()["loss"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing_anchor",
        "extra_anchor",
        "missing_assignment",
        "duplicate_assignment",
        "extra_assignment",
        "empty_group",
        "empty_groups",
        "bad_budget_sum",
        "zero_budget",
        "negative_budget",
        "boolean_budget",
        "missing_calibration",
        "blank_calibration",
        "equal_anchors",
        "wrong_structural_direction",
        "wrong_metric_direction",
        "wrong_esm2_direction",
        "nan_anchor",
        "string_anchor",
        "boolean_anchor",
        "infinite_span",
        "unknown_field",
        "wrong_mode",
    ],
)
def test_bounded_configuration_fails_closed(problem):
    weights, terms, config = settings()
    if problem == "missing_anchor":
        config["anchors"].pop("esm2")
    elif problem == "extra_anchor":
        config["anchors"]["pae"] = {"good": 0.0, "bad": 1.0}
    elif problem == "missing_assignment":
        config["groups"]["confidence"]["terms"].remove("i_ptm")
    elif problem == "duplicate_assignment":
        config["groups"]["confidence"]["terms"].append("esm2")
    elif problem == "extra_assignment":
        config["groups"]["confidence"]["terms"].append("unknown")
    elif problem == "empty_group":
        config["groups"]["confidence"]["terms"] = []
    elif problem == "empty_groups":
        config["groups"] = {}
    elif problem == "bad_budget_sum":
        config["groups"]["confidence"]["budget"] = 0.2
    elif problem in {"zero_budget", "negative_budget", "boolean_budget"}:
        config["groups"]["confidence"]["budget"] = {"zero_budget": 0, "negative_budget": -0.4, "boolean_budget": True}[
            problem
        ]
    elif problem == "missing_calibration":
        config.pop("calibration_id")
    elif problem == "blank_calibration":
        config["calibration_id"] = " "
    elif problem in {"equal_anchors", "wrong_structural_direction"}:
        config["anchors"]["plddt"]["good"] = 1.0 if problem == "equal_anchors" else 2.0
    elif problem == "wrong_metric_direction":
        config["anchors"]["rosetta_interface_sc"] = {"good": 0.3, "bad": 0.8}
    elif problem == "wrong_esm2_direction":
        config["anchors"]["esm2"] = {"good": -5.0, "bad": -1.0}
    elif problem in {"nan_anchor", "string_anchor", "boolean_anchor"}:
        config["anchors"]["plddt"]["good"] = {"nan_anchor": math.nan, "string_anchor": "0", "boolean_anchor": False}[
            problem
        ]
    elif problem == "infinite_span":
        config["anchors"]["plddt"] = {"good": -1e308, "bad": 1e308}
    elif problem == "unknown_field":
        config["recalibrate_each_cycle"] = True
    else:
        config["mode"] = "batch_minmax"
    with pytest.raises(ValueError):
        normalize_loss_combination(config, weights=weights, metric_terms=terms)


@pytest.mark.parametrize("value", [None, math.nan, math.inf, True, "-25"])
def test_bounded_required_metrics_remain_required_and_finite(value):
    metrics = {"rosetta_interface_sc": 0.55}
    if value is not None:
        metrics["rosetta_interface_dg"] = value
    with pytest.raises(ValueError):
        score(metrics=metrics)


def test_bounded_does_not_modify_configuration_or_depend_on_batch_statistics():
    weights, terms, config = settings()
    before = copy.deepcopy((weights, terms, config))
    first = score(weights=weights, terms=terms, combination=config)
    score(metrics={"rosetta_interface_dg": -1e6, "rosetta_interface_sc": -1e6})
    assert score(weights=weights, terms=terms, combination=config) == first
    assert (weights, terms, config) == before


def test_bounded_fails_closed_if_preserved_linear_diagnostic_overflows():
    _, terms, _ = settings()
    terms["rosetta_interface_dg"]["scale"] = 1e-100
    with pytest.raises(ValueError, match="legacy rosetta_interface_dg.*finite"):
        score(metrics={"rosetta_interface_dg": 1e308, "rosetta_interface_sc": 0.55}, terms=terms)


def test_workflow_propagates_explicit_bounded_configuration_and_workload_limits():
    path = Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml"
    data = WorkflowConfigLoader.config_from_path(str(path)).metadata["source_config"]
    weights, terms, config = settings()
    data["design"].update(
        optimization_metric="loss",
        loss_weights=weights,
        metric_loss_terms=terms,
        loss_combination=config,
        post_refold_filter={
            "enabled": True,
            "top_k": 1,
            "max_parents": 1,
            "samples_per_parent": 4,
            "survivors_per_parent": 1,
        },
    )
    data["fold"]["pyrosetta"] = {"enabled": True}
    parsed = WorkflowConfigLoader._normalize_config(data)
    assert parsed.fold_options["loss_combination"] == config
    assert parsed.post_refold_max_parents == 1
    assert parsed.post_refold_samples_per_parent == 4
    assert parsed.post_refold_survivors_per_parent == 1
    assert parsed.metadata["source_config"]["design"]["loss_combination"] == config
    data["design"]["optimization_metric"] = "iptm"
    with pytest.raises(ValueError, match="optimization_metric"):
        WorkflowConfigLoader._normalize_config(data)


def test_bounded_calibration_is_frozen_across_stages_and_runtime_adjustments():
    from opendde_harness.plugin.protein_design.core.contracts import CycleDesignStage, normalize_workflow_adjustments
    from opendde_harness.plugin.protein_design.core.schedule import apply_cycle_schedule, cycle_parameters

    path = Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml"
    data = WorkflowConfigLoader.config_from_path(str(path)).metadata["source_config"]
    weights, terms, config = settings()
    data["design"].update(
        optimization_metric="loss", loss_weights=weights, metric_loss_terms=terms, loss_combination=config
    )
    data["fold"]["pyrosetta"] = {"enabled": True}
    parsed = WorkflowConfigLoader._normalize_config(data)
    parsed.cycle_schedule = [CycleDesignStage(start_cycle=0, end_cycle=0, num_sequences=1)]
    before = copy.deepcopy(parsed.fold_options)
    apply_cycle_schedule(parsed, 0, cycle_parameters(parsed))
    assert parsed.fold_options == before
    with pytest.raises(ValueError):
        CycleDesignStage(start_cycle=0, end_cycle=0, num_sequences=1, loss_combination=config)
    with pytest.raises(ValueError):
        normalize_workflow_adjustments({"loss_combination": config})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_parents", 0),
        ("max_parents", True),
        ("samples_per_parent", "4"),
        ("survivors_per_parent", 0),
        ("survivors_per_parent", 5),
    ],
)
def test_workload_limits_are_strict_and_survivors_cannot_exceed_samples(field, value):
    path = Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml"
    data = WorkflowConfigLoader.config_from_path(str(path)).metadata["source_config"]
    data["design"]["post_refold_filter"] = {
        "enabled": True,
        "max_parents": 1,
        "samples_per_parent": 4,
        "survivors_per_parent": 1,
        field: value,
    }
    with pytest.raises(ValueError):
        WorkflowConfigLoader._normalize_config(data)
