from pathlib import Path

import pytest
import yaml

from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader

EXAMPLE = Path("docs/examples/crlf2_quickstart.yaml")


@pytest.fixture
def config_path(tmp_path):
    if not EXAMPLE.is_file():
        pytest.skip(f"{EXAMPLE} is not present")

    def write(placement):
        data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        data["compute"] = {"placement": placement} if placement is not None else {}
        path = tmp_path / "task.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return str(path)

    return write


def test_an_omitted_placement_leaves_every_device_automatic(config_path) -> None:
    config = WorkflowConfigLoader.config_from_path(config_path(None))

    assert config.placement.fold is None
    assert config.placement.esm is None
    assert config.placement.mpnn is None
    assert config.placement.cp_degree == 1


def test_auto_is_accepted_as_the_explicit_default(config_path) -> None:
    assert WorkflowConfigLoader.config_from_path(config_path("auto")).placement.cp_degree == 1


def test_an_explicit_placement_sets_the_devices_and_the_cp_degree(config_path) -> None:
    config = WorkflowConfigLoader.config_from_path(config_path({"fold": [1, 2, 3], "esm": 0, "mpnn": 0}))

    assert config.placement.fold == [1, 2, 3]
    assert config.placement.cp_degree == 3
    assert (config.placement.esm, config.placement.mpnn) == (0, 0)


@pytest.mark.parametrize(
    "placement, message",
    [
        ({"fold": [0, 1], "cp_degree": 3}, "exactly cp_degree"),
        ({"fold": [0, 0]}, "repeat"),
        ({"fold": [-1]}, "zero or greater"),
        ({"fold": 2}, "list of GPU indices"),
        ({"esm": -1}, "greater than or equal to 0"),
        ({"device": 0}, "unknown compute.placement config field"),
        ("gpu0", "must be 'auto' or a mapping"),
    ],
)
def test_an_invalid_placement_is_rejected(config_path, placement, message) -> None:
    with pytest.raises(ValueError, match=message):
        WorkflowConfigLoader.config_from_path(config_path(placement))
