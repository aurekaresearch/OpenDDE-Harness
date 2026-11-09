"""Compact full_data artifacts must decode to exactly the values OpenDDE wrote."""

import json
from pathlib import Path

import numpy as np
import pytest

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends import loss_confidence_scorer as scorer
from opendde_harness.plugin.protein_design.servers.backends.confidence_artifacts import (
    compact_full_data,
    compact_full_data_dir,
    is_compact_full_data,
    load_full_data,
)
from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor
from opendde_harness.plugin.protein_design.servers.backends.ipsae import compute_ipsae

CHAIN_LENGTHS = {"A": 8, "B": 14}


def dumped(values: np.ndarray) -> list:
    """Mirror OpenDDE's dumper: ``np.round`` on a float32 tensor, then ``tolist``."""
    return np.round(values.astype(np.float32), 2).tolist()


def opendde_full_data(seed: int = 0) -> dict:
    """Build a payload the way OpenDDE's dumper writes it."""
    rng = np.random.default_rng(seed)
    tokens = sum(CHAIN_LENGTHS.values())
    atoms_per_token = 3
    return {
        "atom_plddt": dumped(rng.uniform(20, 99, tokens * atoms_per_token)),
        "token_pair_pde": dumped(rng.uniform(0.3, 25, (tokens, tokens))),
        "contact_probs": dumped(rng.beta(0.3, 3, (tokens, tokens))),
        "token_pair_pae": dumped(rng.uniform(0.5, 31.75, (tokens, tokens))),
        "token_has_frame": [1] * tokens,
        "token_asym_id": [0] * CHAIN_LENGTHS["A"] + [1] * CHAIN_LENGTHS["B"],
        "atom_to_token_idx": np.repeat(np.arange(tokens), atoms_per_token).tolist(),
    }


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def write_cif(path: Path) -> Path:
    """CA-plus-two-atom mmCIF matching ``opendde_full_data`` token/atom layout."""
    lines = [
        "data_test",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    atom_id = 1
    for chain_index, (chain, length) in enumerate(CHAIN_LENGTHS.items()):
        for residue in range(1, length + 1):
            for offset, name in enumerate(("N", "CA", "C")):
                x = 3.8 * residue + 0.5 * offset
                y = 6.0 * chain_index
                lines.append(f"ATOM {atom_id} {name} ALA {chain} {residue} {x:.3f} {y:.3f} 0.000")
                atom_id += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def assert_same_values(original: dict, decoded: dict) -> None:
    assert list(decoded) == list(original)
    for key, value in original.items():
        expected = np.asarray(value)
        got = np.asarray(decoded[key])
        assert got.shape == expected.shape, key
        # Exact equality: no tolerance.
        assert np.array_equal(got, expected), key


def test_compact_npz_decodes_to_exact_json_values_and_removes_json(tmp_path):
    source = write_json(tmp_path / "job_full_data_sample_0.json", opendde_full_data())
    original = json.loads(source.read_text())
    assert original["token_pair_pae"][0][0] != round(original["token_pair_pae"][0][0], 2)  # float32 text

    target = compact_full_data(source)

    assert target == tmp_path / "job_full_data_sample_0.npz"
    assert not source.exists()
    assert is_compact_full_data(target)
    assert_same_values(original, load_full_data(target))
    with np.load(target) as archive:
        assert archive["token_pair_pae"].dtype == np.int16
        assert archive["atom_to_token_idx"].dtype == np.int16


def test_values_outside_the_fixed_point_form_round_trip_losslessly(tmp_path):
    original = {
        "hundredths": [12.34, -0.07, 0.0],
        "fine": [0.123456789, 1e-12],
        "f32": [0.333251953125, 1.0009765625],
        "huge_int": [2**40, -3],
        "flags": [True, False],
        "with_nan": [1.0, float("nan")],
        "ragged": [[1, 2], [3]],
        "label": "opendde",
        "nothing": None,
        "scalar": 7.5,
    }
    source = write_json(tmp_path / "x_full_data_sample_0.json", original)

    decoded = load_full_data(compact_full_data(source))

    with np.load(tmp_path / "x_full_data_sample_0.npz") as archive:
        assert archive["hundredths"].dtype == np.int16
        assert archive["fine"].dtype == np.float64
        assert archive["f32"].dtype == np.float32
    assert decoded["hundredths"].tolist() == [12.34, -0.07, 0.0]
    assert decoded["ragged"] == [[1, 2], [3]]
    assert decoded["label"] == "opendde" and decoded["nothing"] is None
    assert decoded["fine"].tolist() == [0.123456789, 1e-12]
    assert decoded["huge_int"].tolist() == [2**40, -3]
    assert decoded["flags"].tolist() == [True, False]
    assert decoded["with_nan"][0] == 1.0 and np.isnan(decoded["with_nan"][1])
    assert float(decoded["scalar"]) == 7.5


def test_unreadable_json_is_kept_untouched(tmp_path):
    source = tmp_path / "bad_full_data_sample_0.json"
    source.write_text('{"token_pair_pae": [[1.0, ')

    assert compact_full_data(source) == source
    assert source.read_text() == '{"token_pair_pae": [[1.0, '
    assert not list(tmp_path.glob("*.npz"))


def test_terminal_decimal_floats_are_normalized_like_the_loss_reader(tmp_path):
    source = tmp_path / "t_full_data_sample_0.json"
    source.write_text('{"token_pair_pae": [[6., 0.5], [1.25, 0.]]}')

    decoded = load_full_data(compact_full_data(source))

    assert decoded["token_pair_pae"].tolist() == [[6.0, 0.5], [1.25, 0.0]]


def test_ipsae_and_loss_are_identical_for_json_and_npz(tmp_path, monkeypatch):
    data = opendde_full_data(seed=3)
    json_path = write_json(tmp_path / "json/job_full_data_sample_0.json", data)
    npz_path = compact_full_data(write_json(tmp_path / "npz/job_full_data_sample_0.json", data))
    cif = write_cif(tmp_path / "job_sample_0.cif")

    ipsae = [compute_ipsae(10.0, 10.0, str(path), str(cif)) for path in (json_path, npz_path)]
    assert ipsae[0] > 0
    assert ipsae[0] == ipsae[1]

    coords = {
        chain: np.array([[3.8 * i, 6.0 * k, 0.0] for i in range(1, n + 1)])
        for k, (chain, n) in enumerate(CHAIN_LENGTHS.items())
    }
    monkeypatch.setattr(scorer, "_chain_ca_coordinates", lambda *_: coords)
    losses = [
        scorer.score_confidence_loss(
            backend="opendde",
            confidence_path=path,
            structure_path=cif,
            sequences={"A": "A" * CHAIN_LENGTHS["A"], "B": "A" * CHAIN_LENGTHS["B"]},
            binder_chains=["B"],
            iptm=0.7,
            esm2_pll=-1.2,
            design_type="minibinder",
            fixed_residues={},
        )
        for path in (json_path, npz_path)
    ]
    assert losses[0] == losses[1]


def write_fold_output(run_dir: Path, samples: int = 2) -> Path:
    predictions = run_dir / "output/input_0000/input_0000/seed_42/predictions"
    predictions.mkdir(parents=True)
    write_cif(predictions / "input_0000_sample_0.cif")
    for index in range(samples):
        if index:
            (predictions / f"input_0000_sample_{index}.cif").write_text(
                (predictions / "input_0000_sample_0.cif").read_text()
            )
        score = 0.5 + 0.1 * index
        write_json(
            predictions / f"input_0000_summary_confidence_sample_{index}.json",
            {"iptm": score, "ptm": 0.8, "plddt": 80, "ranking_score": score},
        )
        write_json(predictions / f"input_0000_full_data_sample_{index}.json", opendde_full_data(seed=index))
    return predictions


def test_fold_parser_compacts_every_sample_and_reparses_npz(tmp_path):
    run_dir = tmp_path / "run"
    predictions = write_fold_output(run_dir)
    expected = json.loads((predictions / "input_0000_full_data_sample_1.json").read_text())
    predictor = StructurePredictor(FoldConfig(execution_mode="api"))
    sequences = {"A": "A" * CHAIN_LENGTHS["A"], "B": "A" * CHAIN_LENGTHS["B"]}

    first = predictor._parse_opendde_batch_result(sequences, run_dir, 0)

    assert first.all_atom_confidence_path == predictions / "input_0000_full_data_sample_1.npz"
    assert not list(predictions.glob("*full_data*.json"))
    assert len(list(predictions.glob("*full_data*.npz"))) == 2
    assert first.ipsae > 0
    assert_same_values(expected, load_full_data(first.all_atom_confidence_path))

    again = predictor._parse_opendde_batch_result(sequences, run_dir, 0)
    assert again.all_atom_confidence_path == first.all_atom_confidence_path
    assert again.ipsae == first.ipsae


def test_json_format_keeps_original_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    predictions = write_fold_output(run_dir, samples=1)
    predictor = StructurePredictor(FoldConfig(execution_mode="api", full_data_format="json"))

    sequences = {chain: "A" * length for chain, length in CHAIN_LENGTHS.items()}
    result = predictor._parse_opendde_batch_result(sequences, run_dir, 0)

    assert result.all_atom_confidence_path == predictions / "input_0000_full_data_sample_0.json"
    assert not list(predictions.glob("*.npz"))


def test_full_data_format_is_validated_and_accepted_in_yaml():
    assert FoldConfig(execution_mode="api", full_data_format=" NPZ ").full_data_format == "npz"
    with pytest.raises(ValueError, match="full_data_format"):
        FoldConfig(execution_mode="api", full_data_format="parquet")
    assert "full_data_format" in WorkflowConfigLoader._FOLD_FIELDS


def test_compact_dir_reports_where_each_sample_now_lives(tmp_path):
    first = write_json(tmp_path / "a/j_full_data_sample_0.json", opendde_full_data())
    broken = tmp_path / "a/j_full_data_sample_1.json"
    broken.write_text("not json")

    mapping = compact_full_data_dir(tmp_path)

    assert mapping == {first: first.with_suffix(".npz"), broken: broken}
