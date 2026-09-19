import json

import numpy as np
import pytest

from opendde_harness.plugin.protein_design.servers.backends import loss_confidence_scorer as scorer


def test_real_confidence_calculation_combines_candidate_interface_metrics(tmp_path, monkeypatch):
    confidence = tmp_path / "confidence.json"
    confidence.write_text(
        json.dumps(
            {
                "token_asym_id": [0] * 3 + [1] * 12,
                "atom_plddt": [0.8] * 15,
                "atom_to_token_idx": list(range(15)),
                "token_pair_pae": np.full((15, 15), 3.1).tolist(),
                "contact_probs": np.full((15, 15), 0.25).tolist(),
            }
        )
    )
    monkeypatch.setattr(
        scorer,
        "_chain_ca_coordinates",
        lambda *args: {"A": np.arange(9).reshape(3, 3).astype(float), "B": np.arange(36).reshape(12, 3).astype(float)},
    )
    arguments = dict(
        backend="opendde",
        confidence_path=confidence,
        structure_path="fold.pdb",
        sequences={"A": "AAA", "B": "A" * 12},
        binder_chains=["B"],
        fixed_residues={"B": [0, 1, 2]},
        iptm=0.8,
        esm2_pll=-2.0,
    )
    original = scorer.score_confidence_loss(**arguments)
    composite = scorer.score_confidence_loss(
        **arguments,
        metric_values={"rosetta_interface_dg": -15},
        metric_terms={"rosetta_interface_dg": {"direction": "minimize", "scale": 10.0}},
    )
    assert composite["loss"] == pytest.approx(original["loss"] - 1.5)
    assert composite["loss_components"] == original["loss_components"]
    assert composite["loss_objective"]["components"]["rosetta_interface_dg"]["raw"] == -15
    from opendde_harness.plugin.protein_design.servers.backends.loss_objective import DEFAULT_LOSS_WEIGHTS

    names = list(DEFAULT_LOSS_WEIGHTS) + ["rosetta_interface_dg"]
    combination = {
        "mode": "bounded_grouped",
        "calibration_id": "confidence-test-fixture-only",
        "groups": {"all": {"budget": 1.0, "terms": names}},
        "anchors": {name: {"good": 0.0, "bad": 1.0} for name in names},
    }
    combination["anchors"].update(esm2={"good": 0.0, "bad": -5.0}, rosetta_interface_dg={"good": -50.0, "bad": 0.0})
    bounded = scorer.score_confidence_loss(
        **arguments,
        metric_values={"rosetta_interface_dg": -15},
        metric_terms={"rosetta_interface_dg": {"direction": "minimize", "scale": 10.0}},
        loss_combination=combination,
    )
    assert bounded["formula_version"] == "bounded-fixed-grouped-v1"
    assert bounded["loss_objective"]["loss_combination"] == combination
    assert bounded["legacy_loss"] == composite["loss"]
    assert bounded["loss_components"] == original["loss_components"]
    assert 0 <= bounded["loss"] <= 1


def test_mmcif_conversion_preserves_chains_and_source(tmp_path):
    struc = pytest.importorskip("biotite.structure")
    from biotite.structure.io import load_structure, save_structure

    from opendde_harness.plugin.protein_design.servers.backends.pyrosetta_worker import _input_pdb

    atoms = struc.AtomArray(2)
    atoms.chain_id = ["A", "B"]
    atoms.res_id = [1, 1]
    atoms.res_name = ["ALA", "ALA"]
    atoms.atom_name = ["CA", "CA"]
    atoms.element = ["C", "C"]
    atoms.coord = np.asarray([[0, 0, 0], [0, 4, 0]], dtype=float)
    source = tmp_path / "complex.cif"
    save_structure(str(source), atoms)
    before = source.read_bytes()
    converted = _input_pdb(source, tmp_path / "converted.pdb")
    assert list(load_structure(str(converted)).chain_id) == ["A", "B"]
    assert source.read_bytes() == before
    atoms.chain_id = ["AA", "B"]
    save_structure(str(source), atoms)
    with pytest.raises(ValueError, match="multi-character"):
        _input_pdb(source, tmp_path / "ambiguous.pdb")
