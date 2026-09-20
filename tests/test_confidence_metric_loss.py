"""Raw confidence terms; anchor values here are test fixtures, not defaults."""

import copy
import json
import math
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends import fold, ipsae
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
    CONFIDENCE_LOSS_DIRECTIONS,
    DEFAULT_LOSS_WEIGHTS,
    calculate_loss_objective,
    normalize_loss_combination,
    normalize_metric_loss_terms,
)
from opendde_harness.plugin.protein_design.servers.backends.pyrosetta_analysis import PYROSETTA_METRICS


def settings():
    weights = {key: 0.0 for key in DEFAULT_LOSS_WEIGHTS}
    weights["plddt"] = 1.0
    terms = {name: {"direction": direction} for name, direction in CONFIDENCE_LOSS_DIRECTIONS.items()}
    combination = {
        "mode": "bounded_grouped",
        "calibration_id": "confidence-test-fixture-only",
        "groups": {"confidence": {"budget": 1.0, "terms": ["plddt", "min_ipae", "ipsae"]}},
        "anchors": {
            "plddt": {"good": 0.0, "bad": 1.0},
            "min_ipae": {"good": 0.0, "bad": 20.0},
            "ipsae": {"good": 1.0, "bad": 0.0},
        },
    }
    return weights, terms, combination


@pytest.mark.parametrize("bounded", [False, True])
def test_confidence_terms_contributions_directions_and_scale(bounded):
    weights, terms, combination = settings()

    def score(min_ipae, ipsae):
        return calculate_loss_objective(
            {"plddt": 0.25},
            0.0,
            weights=weights,
            metric_terms=terms,
            metric_values={"min_ipae": min_ipae, "ipsae": ipsae},
            loss_combination=combination if bounded else None,
        )

    result = score(10.0, 0.75)
    expected_contributions = {"plddt": 0.25, "min_ipae": 10.0, "ipsae": -0.75}
    if bounded:
        expected_contributions = {"plddt": 1 / 12, "min_ipae": 1 / 6, "ipsae": 1 / 12}
        assert 0 <= result["loss"] <= 1
    for name, expected in expected_contributions.items():
        assert result["components"][name]["contribution"] == pytest.approx(expected)
    assert result["loss"] == pytest.approx(sum(expected_contributions.values()))
    assert score(5.0, 0.75)["loss"] < result["loss"]
    assert score(10.0, 0.9)["loss"] < result["loss"]
    assert result["components"]["min_ipae"]["raw"] == 10
    assert result["components"]["ipsae"]["raw"] == 0.75


@pytest.mark.parametrize("name,direction", CONFIDENCE_LOSS_DIRECTIONS.items())
def test_confidence_terms_require_correct_direction_and_explicit_bounded_anchors(name, direction):
    assert name not in PYROSETTA_METRICS
    with pytest.raises(ValueError, match=f"{name} requires direction"):
        normalize_metric_loss_terms({name: {"direction": "maximize" if direction == "minimize" else "minimize"}})
    weights, terms, combination = settings()
    missing = copy.deepcopy(combination)
    missing["anchors"].pop(name)
    with pytest.raises(ValueError, match="anchors must cover"):
        normalize_loss_combination(missing, weights=weights, metric_terms=terms)
    combination["anchors"][name] = {"good": 1.0, "bad": 0.0} if direction == "minimize" else {"good": 0.0, "bad": 1.0}
    with pytest.raises(ValueError, match="must match direction"):
        normalize_loss_combination(combination, weights=weights, metric_terms=terms)


@pytest.mark.parametrize("name,direction", CONFIDENCE_LOSS_DIRECTIONS.items())
@pytest.mark.parametrize("value", [None, True, "0.5", math.nan, math.inf, -1.0])
def test_invalid_confidence_loss_input_fails(name, direction, value):
    with pytest.raises(ValueError):
        calculate_loss_objective(
            {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS},
            -2.0,
            metric_values={name: value},
            metric_terms={name: {"direction": direction}},
        )


@pytest.mark.parametrize("name,direction", CONFIDENCE_LOSS_DIRECTIONS.items())
def test_missing_raw_input_is_not_implicitly_imputed_and_zero_is_valid(name, direction):
    raw = {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS}
    terms = {name: {"direction": direction}}
    # The worker, not the aggregator, explicitly records the ipSAE fallback.
    with pytest.raises(ValueError, match=f"Missing required loss metric: {name}"):
        calculate_loss_objective(raw, -2.0, metric_terms=terms)
    result = calculate_loss_objective(raw, -2.0, metric_terms=terms, metric_values={name: 0.0})
    assert result["components"][name]["contribution"] == 0.0
    terms[name]["weight"] = 0.0
    assert name not in calculate_loss_objective(raw, -2.0, metric_terms=terms)["components"]


def test_ipsae_above_one_is_not_a_valid_loss_input():
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_loss_objective(
            {key: 0.5 for key in DEFAULT_LOSS_WEIGHTS},
            -2.0,
            metric_values={"ipsae": 1.01},
            metric_terms={"ipsae": {"direction": "maximize"}},
        )


@pytest.mark.parametrize("execution_mode", ["api", "local", "docker"])
def test_confidence_only_config_does_not_require_pyrosetta(execution_mode):
    path = Path(__file__).resolve().parents[1] / "docs/examples/crlf2_quickstart.yaml"
    config = WorkflowConfigLoader.config_from_path(str(path)).metadata["source_config"]
    weights, terms, combination = settings()
    config["design"].update(
        optimization_metric="loss", loss_weights=weights, metric_loss_terms=terms, loss_combination=combination
    )
    config["fold"].update(pyrosetta={"enabled": False}, execution_mode=execution_mode, use_msa=True)
    parsed = WorkflowConfigLoader._normalize_config(config)
    assert parsed.fold_options["loss_combination"] == combination
    assert set(parsed.fold_options["metric_loss_terms"]) == {"min_ipae", "ipsae"}
    config["fold"]["need_atom_confidence"] = False
    with pytest.raises(ValueError, match="need_atom_confidence"):
        WorkflowConfigLoader._normalize_config(config)
    config["fold"]["need_atom_confidence"] = True
    config["design"]["optimization_metric"] = "iptm"
    with pytest.raises(ValueError, match="optimization_metric"):
        WorkflowConfigLoader._normalize_config(config)
    config["design"]["optimization_metric"] = "loss"
    terms["rosetta_interface_dg"] = {"direction": "minimize"}
    with pytest.raises(ValueError, match="pyrosetta.enabled"):
        WorkflowConfigLoader._normalize_config(config)


@pytest.mark.parametrize(
    "pae,distance,expected",
    [(2.0, 5.0, 1 / (1 + (2 / (1.24 * 12 ** (1 / 3) - 1.8)) ** 2)), (20.0, 5.0, 0.0), (2.0, 50.0, 0.0)],
)
def test_real_ipsae_calculation_including_measured_zero(tmp_path, pae, distance, expected):
    structure = tmp_path / "complex.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
        f"ATOM      2  CA  ALA B   1    {distance:8.3f}   0.000   0.000  1.00 90.00           C\n"
    )
    confidence = tmp_path / "confidence.json"
    confidence.write_text(json.dumps({"pae": [[0.0, pae], [pae, 0.0]]}))
    assert ipsae.compute_ipsae(10.0, 10.0, str(confidence), str(structure)) == pytest.approx(expected)


@pytest.mark.parametrize("pae", [math.nan, math.inf, -1.0])
def test_invalid_pae_is_uncomputable_not_a_measured_zero(tmp_path, pae):
    structure = tmp_path / "complex.pdb"
    structure.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
        "ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 90.00           C\n"
    )
    confidence = tmp_path / "confidence.json"
    confidence.write_text(json.dumps({"pae": [[0.0, pae], [pae, 0.0]]}))
    with pytest.raises(ValueError, match="finite, nonnegative PAE"):
        ipsae.compute_ipsae(10.0, 10.0, str(confidence), str(structure))


@pytest.mark.parametrize("value", [0.0, 0.5, None, math.nan, math.inf, -0.1, 1.1])
def test_fold_distinguishes_uncomputable_ipsae_from_measured_zero(monkeypatch, value):
    predictor = object.__new__(fold.StructurePredictor)
    predictor.config = fold.FoldConfig(execution_mode="api")

    def compute(**kwargs):
        if value is None:
            raise ValueError("missing confidence")
        return value

    monkeypatch.setattr(ipsae, "compute_ipsae", compute)
    result = predictor._compute_ipsae_safely("confidence.json", "structure.cif")
    if value in (0.0, 0.5):
        assert result == value
    else:
        assert result is None
