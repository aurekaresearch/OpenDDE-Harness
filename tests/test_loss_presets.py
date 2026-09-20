"""Keep the shipped opt-in policy, documented weights and package inputs aligned."""

import copy
import tomllib
from pathlib import Path

import pytest
import yaml

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import calculate_loss_objective

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "docs/examples/loss_presets/default_bounded_v1.yaml"


def policy():
    return yaml.safe_load(PRESET.read_text())


def test_preset_merges_into_complete_workflow_without_changing_science_or_workload():
    source = yaml.safe_load((ROOT / "docs/examples/crlf2_quickstart.yaml").read_text())
    original = copy.deepcopy(source)
    preset = policy()
    source["design"].update(preset["design"])
    source["fold"].update({key: value for key, value in preset["fold"].items() if key != "pyrosetta"})
    source["fold"].setdefault("pyrosetta", {"relax_repeats": 7}).update(preset["fold"]["pyrosetta"])
    resolved = WorkflowConfigLoader._normalize_config(source)
    assert source["target"] == original["target"]
    assert source["initial_binders"] == original["initial_binders"]
    for key, value in original["design"].items():
        assert source["design"][key] == value
    assert resolved.objective_key == "loss"
    assert resolved.fold_options["pyrosetta"]["enabled"] is True
    assert resolved.fold_options["pyrosetta"]["on_failure"] == "fail"
    assert resolved.fold_options["pyrosetta"]["relax_repeats"] == 7
    assert resolved.fold_options["need_atom_confidence"] is True
    assert resolved.fold_options["loss_combination"] == preset["design"]["loss_combination"]


@pytest.mark.parametrize("anchor,expected", [("good", 0.0), ("bad", 1.0)])
def test_preset_extremes_and_effective_weights(anchor, expected):
    design = policy()["design"]
    anchors = design["loss_combination"]["anchors"]
    values = {name: bounds[anchor] for name, bounds in anchors.items()}
    result = calculate_loss_objective(
        values,
        values["esm2"],
        weights=design["loss_weights"],
        metric_terms=design["metric_loss_terms"],
        metric_values=values,
        loss_combination=design["loss_combination"],
    )
    expected_weights = {
        "ipsae": 0.2,
        "i_ptm": 0.1,
        "plddt": 0.05,
        "i_plddt": 0.05,
        "rosetta_interface_dg": 0.1,
        "rosetta_interface_sc": 0.1,
        "rosetta_interface_dg_per_sasa": 0.2,
        "esm2": 0.1,
        "i_con": 0.05,
        "rg": 0.025,
        "con": 0.025,
    }
    components = {**result["components"], "esm2": result["esm2_component"]}
    assert set(components) == set(expected_weights)
    for name, weight in expected_weights.items():
        assert components[name]["effective_weight"] == pytest.approx(weight)
        assert components[name]["contribution"] == pytest.approx(expected * weight)
    assert result["loss"] == pytest.approx(expected)
    assert sum(group["budget"] for group in result["groups"].values()) == pytest.approx(1.0)


def test_preset_resources_are_in_both_source_and_wheel_manifests():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    targets = manifest["tool"]["hatch"]["build"]["targets"]
    for filename in ("default_bounded_v1.yaml", "README.md"):
        source = f"docs/examples/loss_presets/{filename}"
        destination = f"opendde_harness/plugin/protein_design/examples/loss_presets/{filename}"
        assert (ROOT / source).is_file()
        assert source in targets["sdist"]["include"]
        assert targets["wheel"]["force-include"][source] == destination
