import math
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
    DEFAULT_LOSS_WEIGHTS,
    LOSS_OBJECTIVE_VERSION,
    calculate_loss_objective,
    normalize_loss_weights,
    normalize_metric_loss_terms,
)


def components():
    return {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS if key != "esm2"}


def test_legacy_loss_and_partial_overrides_are_unchanged():
    result = calculate_loss_objective(components(), -2, weights={"i_ptm": 2.0})
    expected = sum(value * 0.5 for key, value in {**DEFAULT_LOSS_WEIGHTS, "i_ptm": 2}.items() if key != "esm2") + 0.2
    assert result["loss"] == pytest.approx(expected)
    assert result["formula_version"] == LOSS_OBJECTIVE_VERSION
    assert result["base_loss"] == result["loss"]
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
    assert result["base_loss"] == base["loss"]
    assert result["loss"] == result["base_loss"] + result["metric_loss"]
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


@pytest.mark.parametrize("value", [True, False])
def test_boolean_base_values_are_not_silent_numeric_scores(value):
    with pytest.raises(ValueError, match="boolean"):
        normalize_loss_weights({"esm2": value})
    with pytest.raises(ValueError, match="boolean"):
        calculate_loss_objective({**components(), "plddt": value}, -2)
    with pytest.raises(ValueError, match="boolean"):
        calculate_loss_objective(components(), value)


@pytest.mark.parametrize("value", [[], [("esm2", 0.1)], True, "esm2"])
def test_loss_weights_require_a_mapping(value):
    with pytest.raises(ValueError, match="loss_weights must be a mapping"):
        normalize_loss_weights(value)


def test_unrepresentable_integer_metric_raises_validation_error():
    with pytest.raises(ValueError, match="must be finite"):
        calculate_loss_objective(
            components(),
            -2,
            metric_values={"rosetta_interface_dg": 10**400},
            metric_terms={"rosetta_interface_dg": {"direction": "minimize"}},
        )


@pytest.mark.parametrize("direction,sign", [("minimize", 1), ("maximize", -1)])
def test_reference_is_a_constant_offset_and_weight_scale_control_sensitivity(direction, sign):
    def score(raw, *, reference=0.0, weight=0.5, scale=10.0):
        return calculate_loss_objective(
            components(),
            -2,
            metric_values={"rosetta_interface_dg": raw},
            metric_terms={
                "rosetta_interface_dg": {
                    "direction": direction,
                    "reference": reference,
                    "weight": weight,
                    "scale": scale,
                }
            },
        )

    first, second = score(-20), score(-10)
    assert second["loss"] - first["loss"] == pytest.approx(sign * 0.5)
    assert score(-20, weight=1.0, scale=20.0)["loss"] == first["loss"]
    for raw in (-20, -10):
        assert score(raw, reference=5.0)["loss"] - score(raw)["loss"] == pytest.approx(-sign * 0.25)
    assert score(-10, reference=5.0)["loss"] - score(-20, reference=5.0)["loss"] == pytest.approx(
        second["loss"] - first["loss"]
    )


def test_recorded_real_scores_preserve_cancellation_and_reverse_base_loss_ranking():
    # Numeric regression fixtures from three independently checked real folds.
    # They exercise aggregation only; they are not a substitute for a real run.
    samples = [
        (
            [
                0.21616413688402547,
                0.30538779619837075,
                0.5657202954014969,
                0.7806467591017581,
                0.7761701345443726,
                0.6299397549016529,
                4.563987961379213,
                -0.2971969536465358,
                0.15262245088545043,
            ],
            -1.4253250414622016,
            (-68.22165166709135, 0.5480666756629944, 22),
            2.378349281636305,
            0.019200022618742985,
        ),
        (
            [
                0.19493240871871376,
                0.2521597298796492,
                0.4748670831518854,
                0.636247988404327,
                0.3621476888656616,
                0.6499388514484674,
                6.647602520052736,
                -0.30031892311445296,
                0.14698864189416008,
            ],
            -1.453310559507372,
            (-49.8483136227145, 0.6157874166965485, 13),
            2.0213737171897304,
            0.06317061935745683,
        ),
        (
            [
                0.195830574868318,
                0.26252659342434037,
                0.46848654895627656,
                0.6280980712785356,
                0.3353450298309326,
                0.7245730977466206,
                7.057648670440072,
                -0.285270390714607,
                0.14438112552579,
            ],
            -1.4347813719068654,
            (-58.967617357373115, 0.6024487018585205, 19),
            2.0492169748516393,
            -0.05161259487553682,
        ),
    ]
    metric_terms = {
        "rosetta_interface_dg": {"direction": "minimize", "weight": 0.5, "scale": 10.0},
        "rosetta_interface_sc": {"direction": "maximize", "weight": 1.0, "reference": 0.5},
        "rosetta_interface_unsat_hbonds": {"direction": "minimize", "weight": 0.25, "scale": 5.0},
    }
    results = []
    for raw_components, pll, raw_metrics, expected_base, expected_composite in samples:
        raw = dict(zip(components(), raw_components, strict=True))
        metrics = dict(zip(metric_terms, raw_metrics, strict=True))
        result = calculate_loss_objective(raw, pll, metric_values=metrics, metric_terms=metric_terms)
        assert result["base_loss"] == pytest.approx(expected_base, abs=1e-12, rel=1e-12)
        assert result["loss"] == pytest.approx(expected_composite, abs=1e-12, rel=1e-12)
        assert result["loss"] == result["base_loss"] + result["metric_loss"]
        results.append(result)
    assert results[1]["base_loss"] < results[2]["base_loss"]
    assert results[2]["loss"] < results[0]["loss"] < results[1]["loss"]
    assert results[2]["loss"] < 0


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
