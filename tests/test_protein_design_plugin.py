"""Protein-design tools integrate with the main assistant's built-in prompt."""

import re
from pathlib import Path
from unittest.mock import Mock

from opendde_harness.agent.tools.shell import ExecTool
from opendde_harness.context_engine.segments import render
from opendde_harness.memory_engine.skill_local.registry import SkillRegistry
from opendde_harness.plugin import active_registry
from opendde_harness.plugin.protein_design.agents.skills import BUILTIN_PROTEIN_DESIGN_SKILLS, ProteinDesignSkillCatalog
from opendde_harness.plugin.protein_design.readiness import compute_configured
from opendde_harness.plugin.protein_design.tools import control

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SKILLS = REPO_ROOT / "opendde_harness" / "plugin" / "protein_design" / "skills"
PLUGIN_NOTE = (
    "Protein Design models run through their configured compute service; this shell does not enter "
    "that container. Use protein_design_context and the Protein Design tools for model execution."
)
CONTROL_TOOLS = (
    "protein_design_context",
    "protein_design_start",
    "protein_design_status",
    "protein_design_candidates",
    "protein_design_adjust",
    "protein_design_stop",
)


def test_plugin_is_active_with_its_data_contributions():
    registry = active_registry()
    assert "protein-design" in registry.activated_ids()
    assert registry.prompt_segments("identity") == []
    assert registry.prompt_segments("scope") == []
    assert registry.skills_dirs() == [PLUGIN_SKILLS]
    assert [plugin_id for plugin_id, _ in registry.readiness_checks()] == ["protein-design"]


def test_identity_prompt_keeps_the_antibody_block_in_place(tmp_path):
    text = render.identity_text(tmp_path, model="openai/gpt-x")
    head = f"# OpenDDE Harness ϒ\n\n{render.PROTEIN_DESIGN_IDENTITY}\n\n## Scope\n"
    assert text.startswith(head)
    assert f"{render.PROTEIN_DESIGN_SCOPE}\n\n## Runtime\n" in text
    assert text.count(render.PROTEIN_DESIGN_SCOPE) == 1
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[:4] == ["## Scope", "## Preparing a design", "## Running and reading a design", "## Runtime"]


def test_scope_names_the_bundled_examples_and_the_context_tool():
    scope = render.PROTEIN_DESIGN_SCOPE
    assert "Load the `protein-design` skill and call `protein_design_context` once" in scope
    for example in ("docs/examples/crlf2_quickstart.yaml", "docs/examples/cacng1_quickstart.yaml"):
        assert example in scope
        assert (REPO_ROOT / example).is_file()


def test_identity_without_design_plugin_preserves_other_contributions(monkeypatch, tmp_path):
    registry = Mock()
    registry.activated_ids.return_value = []
    registry.prompt_segments.side_effect = lambda slot: {
        "identity": ["You are Demo."],
        "scope": ["## Demo Scope\n- be brief"],
    }[slot]
    monkeypatch.setattr(render, "active_registry", lambda: registry)

    text = render.identity_text(tmp_path)

    assert "You are Demo." in text
    assert "## Demo Scope\n- be brief" in text
    assert render.PROTEIN_DESIGN_IDENTITY not in text
    assert render.PROTEIN_DESIGN_SCOPE not in text


def test_scope_only_names_tools_the_plugin_registers():
    source = Path(control.__file__).read_text(encoding="utf-8")
    registered = set(re.findall(r'return "(protein_design_\w+)"', source))
    assert set(CONTROL_TOOLS) <= registered
    assert set(re.findall(r"protein_design_\w+", render.PROTEIN_DESIGN_SCOPE)) <= registered


def test_language_directive_sits_between_identity_and_scope(tmp_path):
    text = render.identity_text(tmp_path, model="openai/gpt-x", language="zh")
    assert f"{render.PROTEIN_DESIGN_IDENTITY}\n\nAlways respond in Simplified Chinese (简体中文)" in text
    assert "another language.\n\n## Scope" in text
    assert "简体中文" not in render.identity_text(tmp_path, model="openai/gpt-x")


def test_workspace_block_does_not_advertise_local_memory_for_external_backends(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage

    skills_line = f"- Custom skills: {get_workspace_storage(tmp_path).skills}/{{skill-name}}/SKILL.md"

    off = render.identity_text(tmp_path, model="openai/gpt-x", long_term_memory=False)
    assert f"## Workspace\nYour workspace is at: {tmp_path}\n{skills_line}\n\n" in off
    assert "user_memory" not in off

    on = render.identity_text(tmp_path, model="openai/gpt-x", long_term_memory=True)
    assert "user.md" not in on
    assert "episodes.md" not in on


def test_the_model_line_is_omitted_when_the_turn_names_no_model(tmp_path):
    """The turn hands in the id the request reaches. Rendering used to read the
    configured id from config, which is not the routed one."""
    assert "running on model" not in render.identity_text(tmp_path)
    assert "You are running on model: openai/gpt-x." in render.identity_text(tmp_path, model="openai/gpt-x")


def test_bash_description_carries_the_plugin_note():
    """The note is keyed on the tool's advertised name, which is now ``bash``."""
    assert ExecTool().description.endswith(PLUGIN_NOTE)


def test_skills_are_discovered_from_the_plugin_directory(tmp_path):
    registry = SkillRegistry(tmp_path / "ws")
    builtin = {m.name: m.path for m in registry.list_all() if m.source == "builtin"}
    assert set(BUILTIN_PROTEIN_DESIGN_SKILLS) | {"protein-design"} <= set(builtin)
    assert "weather" in builtin
    assert all(builtin[name].is_relative_to(PLUGIN_SKILLS) for name in BUILTIN_PROTEIN_DESIGN_SKILLS)
    assert ProteinDesignSkillCatalog.builtin().names() == BUILTIN_PROTEIN_DESIGN_SKILLS


def test_readiness_needs_a_compute_url_or_workers():
    assert compute_configured({}) is False
    assert compute_configured({"compute_url": "http://127.0.0.1:8080"}) is True
    assert compute_configured({"compute_workers": 2}) is True
    assert compute_configured({"compute_url": "", "compute_workers": 0}) is False


def test_quickstart_examples_resolve_without_a_source_checkout(monkeypatch, tmp_path):
    """An installed release has no checkout, so the examples ship in the package."""

    from opendde_harness.plugin.protein_design.core import preparation

    packaged = tmp_path / "examples"
    packaged.mkdir()
    for name in ("crlf2_quickstart.yaml", "cacng1_quickstart.yaml"):
        (packaged / name).write_text("target: {}\n")
    monkeypatch.setattr(preparation, "__file__", str(tmp_path / "core" / "preparation.py"))
    monkeypatch.setenv("OPENDDE_HARNESS_PROJECT_ROOT", str(tmp_path / "no-checkout"))
    monkeypatch.chdir(tmp_path)

    examples = preparation.preparation_context({})["examples"]

    assert [example["available"] for example in examples] == [True, True]
    assert all(str(packaged) in example["path"] for example in examples)
