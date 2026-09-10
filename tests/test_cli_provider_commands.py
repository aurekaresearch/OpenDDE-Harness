"""``ddeharness provider ...`` writes: the wire per section and per model."""

import json

import pytest
from typer.testing import CliRunner

from opendde_harness.cli.provider_commands import provider_app
from opendde_harness.config import loader
from opendde_harness.config.schema import Config
from opendde_harness.providers.wire import merge_key


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"providers": {"custom": {"apiKey": "sk", "apiBase": "http://relay/v1"}}}))
    monkeypatch.setattr(loader, "_current_config_path", path)
    return path


def test_provider_set_writes_the_wire(config_path):
    result = CliRunner().invoke(provider_app, ["set", "custom", "--wire", "responses"])

    assert result.exit_code == 0, result.output
    assert json.loads(config_path.read_text())["providers"]["custom"]["wire"] == "responses"

    rejected = CliRunner().invoke(provider_app, ["set", "custom", "--wire", "grpc"])
    assert rejected.exit_code == 1
    assert "Validation failed" in rejected.output


def test_provider_model_set_writes_one_models_overlay(config_path):
    result = CliRunner().invoke(
        provider_app,
        ["model", "set", "custom", "gpt-5.6-terra", "--wire", "responses", "--context-window", "262144"],
    )
    assert result.exit_code == 0, result.output

    again = CliRunner().invoke(provider_app, ["model", "set", "custom", "custom/gpt-5.6-terra", "--label", "Terra"])
    assert again.exit_code == 0, again.output

    config = Config.model_validate(json.loads(config_path.read_text()))
    overlays = config.providers.model_overlays()
    entry = overlays[merge_key("custom", "gpt-5.6-terra")]
    # One entry, patched twice, whichever spelling the second call used.
    assert len(overlays) == 1
    assert (entry.wire, entry.context_window_tokens, entry.label) == ("responses", 262144, "Terra")


def test_provider_model_set_needs_a_field(config_path):
    result = CliRunner().invoke(provider_app, ["model", "set", "custom", "gpt-5.6-terra"])

    assert result.exit_code != 0
    assert "at least one of" in result.output


def test_an_overlay_edit_never_rewrites_an_invalid_section_with_defaults(config_path):
    import pytest
    from pydantic import ValidationError

    from opendde_harness.config.update_providers import set_model_overlay

    raw = json.loads(config_path.read_text())
    raw["providers"]["custom"] = {"apiKey": "keep-me", "apiBase": "http://relay/v1", "wire": "telepathy"}
    config_path.write_text(json.dumps(raw))

    with pytest.raises(ValidationError):
        set_model_overlay("custom", "gpt-x", {"context_window_tokens": 1000})

    assert json.loads(config_path.read_text())["providers"]["custom"]["apiKey"] == "keep-me"


def test_provider_model_set_writes_and_validates_the_reasoning_effort(config_path):
    ok = CliRunner().invoke(provider_app, ["model", "set", "custom", "deep-thinker", "--reasoning-effort", "high"])
    assert ok.exit_code == 0, ok.output

    config = Config.model_validate(json.loads(config_path.read_text()))
    assert config.providers.model_overlays()[merge_key("custom", "deep-thinker")].reasoning_effort == "high"

    bad = CliRunner().invoke(provider_app, ["model", "set", "custom", "deep-thinker", "--reasoning-effort", "ultra"])
    assert bad.exit_code != 0
