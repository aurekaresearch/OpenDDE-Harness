"""Raw Min ipAE must retain units, asymmetry and verified chain identity."""

import json

import pytest

from opendde_harness.plugin.protein_design.servers.backends.interchain_pae import measure_min_interchain_pae


@pytest.fixture
def raw_pae(tmp_path):
    # Structure/input order deliberately differs from both matrix and asym-ID order.
    structure = tmp_path / "complex.cif"
    structure.write_text(
        "data_test\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.label_comp_id\n"
        "_atom_site.label_alt_id\n_atom_site.pdbx_PDB_model_num\n"
        "ATOM 1 CA A 1 ALA . 1\nATOM 2 CA A 2 GLY . 1\n"
        "ATOM 3 CA B 1 SER . 1\nATOM 4 CA B 2 THR . 1\n#\n"
    )
    data = {
        "token_pair_pae": [[0, 6, 0.01, 7], [4, 0, 8, 0.02], [0.01, 5, 0, 9], [3, 0.02, 2, 0]],
        "atom_to_token_idx": [2, 0, 3, 1],
        "token_asym_id": [9, 4, 9, 4],
        "token_has_frame": [1, 1, 1, 1],
    }
    confidence = tmp_path / "confidence.json"

    def measure(**updates):
        confidence.write_text(json.dumps(data))
        return measure_min_interchain_pae(
            **{
                "confidence_path": confidence,
                "structure_path": structure,
                "sequences": {"B": "ST", "A": "AG"},
                "binder_chains": ["B"],
                "target_chains": ["A"],
                **updates,
            }
        )

    return data, structure, measure


def test_raw_minimum_uses_both_directions_and_atom_mapped_chain_identity(raw_pae):
    _, _, measure = raw_pae
    result = measure()
    assert result["status"] == "success"
    assert result["value"] == 2.0
    assert result["units"] == "angstrom"
    assert result["directional_minima"] == {"binder_to_target": 2.0, "target_to_binder": 5.0}
    assert result["chain_mapping"] == {
        "A": {"asym_id": 9, "residue_count": 2, "valid_frame_count": 2},
        "B": {"asym_id": 4, "residue_count": 2, "valid_frame_count": 2},
    }
    assert "not averaged" in result["definition"]
    assert result["matrix_key"] == "token_pair_pae"
    assert result["matrix_shape"] == [4, 4]


def test_multichain_minimum_excludes_within_partner_entries(tmp_path):
    structure = tmp_path / "complex.pdb"
    structure.write_text(
        "".join(
            f"ATOM  {i:5d}  CA  ALA {chain}{1:4d}    {0:8.3f}{0:8.3f}{0:8.3f}  1.00 80.00           C\n"
            for i, chain in enumerate("ABHL", 1)
        )
    )
    confidence = tmp_path / "confidence.json"
    confidence.write_text(
        json.dumps(
            {
                "token_pair_pae": [[0, 0.1, 9, 8], [0.2, 0, 7, 6], [5, 4, 0, 0.01], [3, 2, 0.02, 0]],
                "token_asym_id": [0, 1, 2, 3],
                "atom_to_token_idx": [0, 1, 2, 3],
                "token_has_frame": [True, True, True, True],
            }
        )
    )
    result = measure_min_interchain_pae(
        confidence_path=confidence,
        structure_path=structure,
        sequences={chain: "A" for chain in "ABHL"},
        binder_chains=["H", "L"],
        target_chains=["A", "B"],
    )
    assert result["value"] == 2.0
    assert result["directional_minima"] == {"binder_to_target": 2.0, "target_to_binder": 6.0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, "0.01"])
def test_invalid_pae_values_are_unavailable_not_zero(raw_pae, value):
    data, _, measure = raw_pae
    data["token_pair_pae"][0][0] = value
    result = measure()
    assert result["status"] == "unavailable"
    assert "reason" in result
    assert "value" not in result


@pytest.mark.parametrize(
    "key,value",
    [
        ("token_pair_pae", [[1, 2]]),
        ("token_pair_pae", [[[0] * 4] * 4]),
        ("token_pair_pae", []),
        ("token_pair_pae", [[1, 2], [3]]),
        ("atom_to_token_idx", [0, 1]),
        ("atom_to_token_idx", [2, 0, 3, 4]),
        ("atom_to_token_idx", [2, 0, 3, -1]),
        ("atom_to_token_idx", [2, 0, 3, 0.5]),
        ("atom_to_token_idx", [2, 0, 3, True]),
        ("atom_to_token_idx", [2, 0, 3, 3]),
        ("token_asym_id", [9, 4]),
        ("token_asym_id", [9, 9, 9, 9]),
        ("token_asym_id", [9, 4, 8, 4]),
    ],
)
def test_malformed_or_ambiguous_mapping_is_explicitly_unavailable(raw_pae, key, value):
    data, _, measure = raw_pae
    data[key] = value
    result = measure()
    assert result["status"] == "unavailable"
    assert result["reason"]
    assert "value" not in result


@pytest.mark.parametrize("key", ["token_pair_pae", "token_asym_id", "atom_to_token_idx", "token_has_frame"])
def test_missing_raw_data_is_not_replaced_with_summary_or_loss(raw_pae, key):
    data, _, measure = raw_pae
    del data[key]
    data.update(i_pae=0.001, ipsae=0.99, chain_pair_gpde=[[0, 0.01], [0.02, 0]])
    result = measure()
    assert result["status"] == "unavailable"
    assert "value" not in result


@pytest.mark.parametrize(
    "arguments",
    [
        {"binder_chains": []},
        {"target_chains": []},
        {"binder_chains": ["A"]},
        {"binder_chains": ["B", "B"]},
        {"target_chains": ["X"]},
        {"sequences": {"A": "ST", "B": "AG"}},
        {"confidence_path": None},
        {"structure_path": None},
    ],
)
def test_missing_roles_or_inconsistent_structure_never_guess(raw_pae, arguments):
    _, _, measure = raw_pae
    result = measure(**arguments)
    assert result["status"] == "unavailable"
    assert result["reason"]
    assert "value" not in result


def test_valid_zero_and_pae_spelling_remain_real_measurements(raw_pae):
    data, _, measure = raw_pae
    data["pae"] = data.pop("token_pair_pae")
    data["pae"][3][2] = 0
    result = measure()
    assert result["status"] == "success"
    assert result["value"] == 0.0
    assert result["matrix_key"] == "pae"


def test_multiple_atoms_must_map_to_one_residue_index(raw_pae):
    data, structure, measure = raw_pae
    structure.write_text(structure.read_text().replace("ATOM 2 CA", "ATOM 5 N A 1 ALA . 1\nATOM 2 CA"))
    data["atom_to_token_idx"] = [2, 2, 0, 3, 1]
    assert measure()["value"] == 2.0
    data["atom_to_token_idx"][1] = 0
    assert measure()["status"] == "unavailable"


def test_alternate_locations_and_multiple_models_are_unavailable(raw_pae):
    _, structure, measure = raw_pae
    original = structure.read_text()
    structure.write_text(original.replace("ALA . 1", "ALA A 1"))
    assert measure()["status"] == "unavailable"
    structure.write_text(original.replace("ALA . 1", "ALA . 2"))
    assert measure()["status"] == "unavailable"


@pytest.mark.parametrize("flags", [[1, 1], [1, 1, 1, 2], [1, 1, 1, -1], [1, 1, 1, 0.5], ["1"] * 4, [[1] * 4]])
def test_malformed_frame_flags_are_unavailable(raw_pae, flags):
    data, _, measure = raw_pae
    data["token_has_frame"] = flags
    assert measure()["status"] == "unavailable"


def test_frames_mask_both_axes_and_keep_single_valid_pair_at_index_zero(raw_pae):
    data, _, measure = raw_pae
    data["token_has_frame"] = [1, 1, 0, 0]
    result = measure()
    assert result["value"] == 4.0
    assert result["directional_minima"] == {"binder_to_target": 4.0, "target_to_binder": 6.0}
    assert result["chain_mapping"]["A"]["valid_frame_count"] == 1
    assert result["chain_mapping"]["B"]["valid_frame_count"] == 1
    data["token_has_frame"] = [1, 0, 1, 0]
    result = measure()
    assert result["status"] == "unavailable"
    assert "at least one valid-frame" in result["reason"]
