"""The protein-design plugin supplies what the generic runtime used to hard-code."""

import re
from pathlib import Path

from opendde_harness.agent.tools.shell import ExecTool
from opendde_harness.context_engine.segments import render
from opendde_harness.memory_engine.skill_local.registry import SkillRegistry
from opendde_harness.plugin import active_registry
from opendde_harness.plugin.protein_design.agents.skills import BUILTIN_PROTEIN_DESIGN_SKILLS, ProteinDesignSkillCatalog
from opendde_harness.plugin.protein_design.prompts.identity import antibody_scope, identity_line
from opendde_harness.plugin.protein_design.readiness import compute_configured
from opendde_harness.plugin.protein_design.tools import control

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SKILLS = REPO_ROOT / "opendde_harness" / "plugin" / "protein_design" / "skills"
EXEC_DESCRIPTION = (
    "Execute a shell command on the Harness client host and return its output. "
    "Protein Design models run through their configured compute service; this shell does not enter that container. "
    "Use protein_design_context and the Protein Design tools for model execution."
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
    assert registry.prompt_segments("identity") == [identity_line()]
    assert registry.prompt_segments("scope") == [antibody_scope()]
    assert registry.skills_dirs() == [PLUGIN_SKILLS]
    assert [plugin_id for plugin_id, _ in registry.readiness_checks()] == ["protein-design"]


def test_identity_prompt_keeps_the_antibody_block_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_language_directive", lambda: "")
    text = render.identity_text(tmp_path, model="openai/gpt-x")
    head = "# OpenDDE Harness ϒ\n\nYou are OpenDDE Harness, an antibody design assistant.\n\n## Scope\n"
    assert text.startswith(head)
    assert f"{antibody_scope()}\n\n## Runtime\n" in text
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[:4] == ["## Scope", "## Preparing a design", "## Running and reading a design", "## Runtime"]


def test_scope_names_the_bundled_examples_and_the_context_tool():
    scope = antibody_scope()
    assert "Load the `protein-design` skill and call `protein_design_context` once" in scope
    for example in ("docs/examples/crlf2_quickstart.yaml", "docs/examples/cacng1_quickstart.yaml"):
        assert example in scope
        assert (REPO_ROOT / example).is_file()


def test_scope_only_names_tools_the_plugin_registers():
    source = Path(control.__file__).read_text(encoding="utf-8")
    registered = set(re.findall(r'return "(protein_design_\w+)"', source))
    assert set(CONTROL_TOOLS) <= registered
    assert set(re.findall(r"protein_design_\w+", antibody_scope())) <= registered


def test_language_directive_sits_between_identity_and_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_language_directive", lambda: "\nAlways respond in Simplified Chinese.\n")
    text = render.identity_text(tmp_path, model="openai/gpt-x")
    assert "antibody design assistant.\n\nAlways respond in Simplified Chinese.\n\n## Scope" in text


def test_workspace_block_lists_memory_files_only_when_memory_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_language_directive", lambda: "")
    skills_line = f"- Custom skills: {tmp_path}/skills/{{skill-name}}/SKILL.md"

    monkeypatch.setattr(render, "_long_term_memory_enabled", lambda: False)
    off = render.identity_text(tmp_path, model="openai/gpt-x")
    assert f"## Workspace\nYour workspace is at: {tmp_path}\n{skills_line}\n\n" in off
    assert "user_memory" not in off

    monkeypatch.setattr(render, "_long_term_memory_enabled", lambda: True)
    on = render.identity_text(tmp_path, model="openai/gpt-x")
    assert f"- User profile: {tmp_path}/user_memory/profile/user.md" in on
    assert f"- Episodic log: {tmp_path}/user_memory/episodic/episodes.md" in on
    assert "(grep-searchable; entries start with [YYYY-MM-DD HH:MM])" in on


def test_memory_enablement_follows_the_configured_backend(monkeypatch):
    import opendde_harness.config.loader as loader

    class _Config:
        def __init__(self, backend):
            self.memory = type("M", (), {"backend": backend})()

    monkeypatch.setattr(loader, "load_config", lambda *a, **k: _Config(None))
    assert render._long_term_memory_enabled() is False
    monkeypatch.setattr(loader, "load_config", lambda *a, **k: _Config("longterm"))
    assert render._long_term_memory_enabled() is True

    def _boom(*a, **k):
        raise RuntimeError("no config")

    monkeypatch.setattr(loader, "load_config", _boom)
    assert render._long_term_memory_enabled() is False


def test_exec_description_carries_the_plugin_note():
    assert ExecTool().description == EXEC_DESCRIPTION


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
