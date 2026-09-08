"""Bundled workspace templates and how ``sync_workspace_templates`` places them."""

from importlib.resources import files as pkg_files
from pathlib import Path

from opendde_harness.utils.helpers import sync_workspace_templates

TEMPLATES = Path(str(pkg_files("opendde_harness") / "templates"))
# Template file -> workspace path it is copied to.
PLACEMENT = {
    "SOUL.md": "agent_memory/profile/soul.md",
    "AGENTS.md": "agent_memory/profile/agent.md",
    "USER.md": "user_memory/profile/user.md",
    "TOOLS.md": "TOOLS.md",
}


def test_the_four_templates_ship_with_the_package():
    for name in PLACEMENT:
        assert (TEMPLATES / name).is_file()


def test_a_fresh_workspace_receives_the_bundled_templates(tmp_path):
    added = sync_workspace_templates(tmp_path, silent=True)
    for name, dest in PLACEMENT.items():
        assert dest in added
        assert (tmp_path / dest).read_text(encoding="utf-8") == (TEMPLATES / name).read_text(encoding="utf-8")
    assert (tmp_path / "skills").is_dir()


def test_a_second_sync_leaves_user_edits_untouched(tmp_path):
    sync_workspace_templates(tmp_path, silent=True)
    edited = tmp_path / "agent_memory" / "profile" / "agent.md"
    edited.write_text("# My own notes\n", encoding="utf-8")

    added = sync_workspace_templates(tmp_path, silent=True)
    assert added == []
    assert edited.read_text(encoding="utf-8") == "# My own notes\n"


def test_templates_no_longer_restate_the_product_identity():
    for name in PLACEMENT:
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "You are OpenDDE Harness" not in text
        assert "I am OpenDDE Harness" not in text
