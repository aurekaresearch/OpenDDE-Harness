import math
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
    DEFAULT_LOSS_WEIGHTS,
    LOSS_OBJECTIVE_VERSION,
    calculate_loss_objective,
    normalize_metric_loss_terms,
)


def components():
    return {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS if key != "esm2"}


def test_legacy_loss_and_partial_overrides_are_unchanged():
    result = calculate_loss_objective(components(), -2, weights={"i_ptm": 2.0})
    expected = sum(value * 0.5 for key, value in {**DEFAULT_LOSS_WEIGHTS, "i_ptm": 2}.items() if key != "esm2") + 0.2
    assert result["loss"] == pytest.approx(expected)
    assert result["formula_version"] == LOSS_OBJECTIVE_VERSION
    assert "metric_loss" not in result


def test_composite_loss_directions_scales_and_provenance():
    base = calculate_loss_objective(components(), -2)
    result = calculate_loss_objective(
        components(),
        -2,
        metric_values={"rosetta_interface_dg": -20, "rosetta_interface_sc": 0.7},
        metric_terms={
            "rosetta_interface_dg": {"direction": "minimize", "weight": 0.5, "scale": 10.0},
            "rosetta_interface_sc": {"direction": "maximize", "weight": 2.0, "reference": 0.5},
        },
    )
    assert result["loss"] == pytest.approx(base["loss"] - 1.4)
    assert result["metric_loss"] == pytest.approx(-1.4)
    assert result["structure_loss"] == base["structure_loss"]
    assert result["components"]["rosetta_interface_dg"]["raw"] == -20
    assert result["components"]["rosetta_interface_sc"]["normalized"] == pytest.approx(0.2)


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf, "bad", "1.0", True])
def test_enabled_nonfinite_or_nonnumeric_metric_is_an_error(value):
    with pytest.raises(ValueError):
        calculate_loss_objective(
            components(),
            -2,
            metric_values={"rosetta_interface_dg": value},
            metric_terms={"rosetta_interface_dg": {"direction": "minimize"}},
        )


def test_missing_required_metric_fails_and_zero_weight_does_not_require_it():
    with pytest.raises(ValueError, match="Missing required loss metric"):
        calculate_loss_objective(components(), -2, metric_terms={"rosetta_interface_dg": {"direction": "minimize"}})
    result = calculate_loss_objective(
        components(), -2, metric_terms={"rosetta_interface_dg": {"direction": "minimize", "weight": 0.0}}
    )
    assert result["loss"] == calculate_loss_objective(components(), -2)["loss"]


@pytest.mark.parametrize(
    "term",
    [
        {},
        {"direction": "minimum"},
        {"direction": "minimize", "scale": 0},
        {"direction": "minimize", "weight": -1},
        {"direction": "minimize", "reference": math.inf},
        {"direction": "minimize", "scale": math.nan},
        {"direction": "minimize", "weight": True},
        {"direction": "minimize", "missing": "zero"},
    ],
)
def test_metric_term_configuration_is_strict(term):
    with pytest.raises(ValueError):
        normalize_metric_loss_terms({"rosetta_interface_dg": term})


def test_unknown_metric_names_are_rejected_even_at_zero_weight():
    with pytest.raises(ValueError, match="Unsupported metric"):
        normalize_metric_loss_terms({"rosetta_typo": {"direction": "minimize", "weight": 0}})


def test_loss_overflow_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        calculate_loss_objective(
            components(),
            -2,
            metric_values={"rosetta_interface_dg": 1e308},
            metric_terms={"rosetta_interface_dg": {"direction": "minimize", "weight": 1e308}},
        )
    with pytest.raises(ValueError, match="finite"):
        calculate_loss_objective({key: 1e308 for key in components()}, -2)


def test_workflow_requires_enabled_analysis_for_metric_loss():
    path = Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml"
    config = WorkflowConfigLoader.config_from_path(str(path)).metadata["source_config"]
    config["design"]["optimization_metric"] = "loss"
    config["design"]["metric_loss_terms"] = {"rosetta_interface_dg": {"direction": "minimize", "scale": 10.0}}
    with pytest.raises(ValueError, match="pyrosetta.enabled"):
        WorkflowConfigLoader._normalize_config(config)
    config["fold"]["pyrosetta"] = {"enabled": True}
    parsed = WorkflowConfigLoader._normalize_config(config)
    assert parsed.fold_options["metric_loss_terms"]["rosetta_interface_dg"]["scale"] == 10
    config["design"]["optimization_metric"] = "iptm"
    with pytest.raises(ValueError, match="optimization_metric"):
        WorkflowConfigLoader._normalize_config(config)
