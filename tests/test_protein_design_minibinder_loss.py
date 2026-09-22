"""Minibinder objectives do not reinterpret editable regions as antibody regions."""

import math

import numpy as np
import pytest

from opendde_harness.plugin.protein_design.servers.backends import loss_confidence_scorer as scorer


@pytest.fixture
def confidence(monkeypatch):
    count = 14
    data = {
        "token_asym_id": [1] * 2 + [2] * 12,
        "atom_plddt": [0.9] * count,
        "atom_to_token_idx": list(range(count)),
        "token_pair_pae": np.full((count, count), 3.1),
        "contact_probs": np.full((count, count), 0.5),
    }
    monkeypatch.setattr(scorer, "_load_first_json_object", lambda _: data)
    monkeypatch.setattr(
        scorer,
        "_chain_ca_coordinates",
        lambda *_: {
            "A": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "B": np.array([[float(i), 1.0, 0.0] for i in range(12)]),
        },
    )
    return data


def score(**kwargs):
    return scorer.score_confidence_loss(
        backend="opendde",
        confidence_path="unused",
        structure_path="unused",
        sequences={"A": "AC", "B": "ACDEFGHIKLMN"},
        binder_chains=["B"],
        iptm=0.7,
        esm2_pll=-1.2,
        **kwargs,
    )


def test_minibinder_loss_ignores_fixed_masks_and_has_no_antibody_breakdown(confidence):
    results = [
        score(design_type="minibinder", fixed_residues=mask) for mask in ({}, {"B": [0, 1]}, {"B": list(range(12))})
    ]
    assert results[0] == results[1] == results[2]
    result = results[0]
    assert math.isfinite(result["loss"])
    assert result["loss_components"]["i_con"] == pytest.approx(-math.log(0.5))
    assert result["formula_version"].startswith("minibinder-")
    assert "paratope_loss" not in result
    assert "cdr" not in str(result) and "framework" not in str(result)


def test_antibody_default_is_unchanged_and_still_requires_framework(confidence):
    with pytest.raises(ValueError, match="both CDR and framework"):
        score(fixed_residues={})
    legacy = score(fixed_residues={"B": [0, 1]})
    assert legacy == score(design_type="antibody", fixed_residues={"B": [0, 1]})
    assert "paratope_loss_breakdown" in legacy


def test_minibinder_hotspots_and_interface_confidence(confidence):
    confidence["contact_probs"][2:, 0] = 0.9
    confidence["contact_probs"][0, 2:] = 0.9
    result = score(design_type="minibinder", fixed_residues={}, target_hotspots={"A": [1]})
    assert result["target_objective_token_count"] == 1
    assert result["loss_components"]["i_con"] == pytest.approx(-math.log(0.9))
    assert result["loss_components"]["i_plddt"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="one-based"):
        score(design_type="minibinder", fixed_residues={}, target_hotspots={"A": [0]})
    confidence["contact_probs"][:2, 2:] = 0
    confidence["contact_probs"][2:, :2] = 0
    result = score(design_type="minibinder", fixed_residues={})
    assert result["interface_confidence_scope"] == "whole_binder_no_predicted_contacts"
    assert math.isfinite(result["loss"])


def test_invalid_mode_is_not_silently_scored_as_antibody(confidence):
    with pytest.raises(ValueError, match="design_type"):
        score(design_type="unknown", fixed_residues={})
