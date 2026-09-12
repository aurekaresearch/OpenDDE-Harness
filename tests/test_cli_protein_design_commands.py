"""Protein-design YAML validation summaries expose reviewed cycle policies."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from opendde_harness.cli import protein_design_commands as commands


def test_validate_displays_cycle_schedule(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "docs/examples/crlf2_scheduled.yaml"
    monkeypatch.setattr(commands, "_load_plugin_config", lambda _: {})
    result = CliRunner().invoke(commands.protein_design_app, ["validate", "--config", str(source), "--json"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["router_selection_strategy"] == "weighted"
    assert [stage["num_sequences"] for stage in summary["cycle_schedule"]] == [8, 4, 2]


def test_validate_rejects_overlapping_stages(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "docs/examples/crlf2_scheduled.yaml"
    data = yaml.safe_load(source.read_text())
    data["design"]["cycle_schedule"][1]["start_cycle"] = 3
    path = tmp_path / "overlap.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(commands, "_load_plugin_config", lambda _: {})
    result = CliRunner().invoke(commands.protein_design_app, ["validate", "--config", str(path), "--json"])
    assert result.exit_code == 1
    assert "overlap" in result.output
