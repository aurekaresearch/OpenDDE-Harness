"""Manifest and registry handling of the data-only contribution kinds."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from opendde_harness.plugin import PluginManifest, PluginRegistry
from opendde_harness.plugin.discover import DiscoveredPlugin, Source

MANIFEST = """
[plugin]
id = "demo"
version = "0.1"
enabled_by_default = true

[plugin.contributes]
skills_dirs = ["skills"]
readiness = "tests.test_plugin_manifest:demo_ready"

[plugin.contributes.tool_description_notes]
exec = "Demo note."

[[plugin.contributes.prompt_segments]]
name = "who"
slot = "identity"
factory = "tests.test_plugin_manifest:demo_identity"

[[plugin.contributes.prompt_segments]]
name = "rules"
slot = "scope"
factory = "tests.test_plugin_manifest:demo_scope"
"""


def demo_identity() -> str:
    return "You are Demo."


def demo_scope() -> str:
    return "## Demo Scope\n- be brief"


def alpha_scope() -> str:
    return "## Alpha Scope"


def demo_ready(config: dict) -> bool:
    return bool(config.get("url"))


def _activate(manifest_text: str, location: Path | None) -> PluginRegistry:
    registry = PluginRegistry()
    plugin = DiscoveredPlugin(
        manifest=PluginManifest.from_toml_str(manifest_text),
        source=Source.BUNDLED,
        location=location,
    )
    registry.activate([plugin])
    return registry


def test_manifest_parses_data_contributions():
    mf = PluginManifest.from_toml_str(MANIFEST)
    assert mf.contributes.skills_dirs == ["skills"]
    assert mf.contributes.tool_description_notes == {"exec": "Demo note."}
    assert mf.contributes.readiness == "tests.test_plugin_manifest:demo_ready"
    assert [(s.name, s.slot) for s in mf.contributes.prompt_segments] == [("who", "identity"), ("rules", "scope")]


@pytest.mark.parametrize(
    "bad",
    [
        '[[plugin.contributes.prompt_segments]]\nname = "x"\nslot = "footer"\nfactory = "m:f"',
        '[[plugin.contributes.prompt_segments]]\nname = "x"\nslot = "scope"\nfactory = "not a ref"',
        '[plugin.contributes]\nreadiness = "no-colon"',
    ],
)
def test_manifest_rejects_malformed_contributions(bad):
    with pytest.raises(ValidationError):
        PluginManifest.from_toml_str(f'[plugin]\nid = "x"\nversion = "1"\n{bad}')


def test_manifest_rejects_duplicate_prompt_segment_names():
    text = MANIFEST.replace('name = "rules"', 'name = "who"')
    with pytest.raises(ValidationError, match="duplicate prompt_segment"):
        PluginManifest.from_toml_str(text)


def test_registry_exposes_data_contributions(tmp_path):
    registry = _activate(MANIFEST, tmp_path / "opendde-harness-plugin.toml")
    assert registry.prompt_segments("identity") == ["You are Demo."]
    assert registry.prompt_segments("scope") == ["## Demo Scope\n- be brief"]
    assert registry.tool_description_note("exec") == "Demo note."
    assert registry.tool_description_note("read_file") == ""
    assert registry.skills_dirs() == [tmp_path / "skills"]
    checks = registry.readiness_checks()
    assert [plugin_id for plugin_id, _ in checks] == ["demo"]
    assert checks[0][1]({"url": "x"}) is True
    assert checks[0][1]({}) is False


def test_registry_skips_skills_dirs_without_a_manifest_location():
    registry = _activate(MANIFEST, None)
    assert registry.skills_dirs() == []
    assert registry.prompt_segments("identity") == ["You are Demo."]


def test_registry_orders_contributions_by_plugin_id(tmp_path):
    first = MANIFEST.replace('id = "demo"', 'id = "zeta"').replace('name = "rules"', 'name = "zeta-rules"')
    second = MANIFEST.replace('id = "demo"', 'id = "alpha"').replace("demo_scope", "alpha_scope")
    registry = PluginRegistry()
    registry.activate(
        [
            DiscoveredPlugin(PluginManifest.from_toml_str(first), Source.BUNDLED, tmp_path / "z" / "m.toml"),
            DiscoveredPlugin(PluginManifest.from_toml_str(second), Source.BUNDLED, tmp_path / "a" / "m.toml"),
        ]
    )
    assert registry.prompt_segments("scope") == ["## Alpha Scope", "## Demo Scope\n- be brief"]
    assert registry.tool_description_note("exec") == "Demo note. Demo note."
    assert registry.skills_dirs() == [tmp_path / "a" / "skills", tmp_path / "z" / "skills"]
