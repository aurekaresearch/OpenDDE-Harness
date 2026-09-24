"""Execution directories must not receive Harness-owned persistence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from opendde_harness.session.manager import SessionManager


def test_session_save_does_not_write_into_workspace(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    manager = SessionManager(workspace)
    session = manager.get_or_create("cli:one")
    session.messages.append({"role": "user", "content": "hello"})
    manager.save(session)
    assert list(workspace.iterdir()) == []
    restored = SessionManager(workspace).get_or_create("cli:one")
    assert restored.messages[0]["content"] == "hello"


def test_templates_do_not_initialize_or_migrate_workspace_memory(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage
    from opendde_harness.utils.helpers import sync_workspace_templates

    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "USER.md").write_text("private legacy memory")
    sync_workspace_templates(workspace, silent=True)
    layout = get_workspace_storage(workspace)
    assert sorted(p.name for p in workspace.iterdir()) == ["USER.md"]
    assert (layout.assistant / "soul.md").exists()
    assert not layout.host_memory.exists()


def test_outbox_state_is_independent_of_local_profile(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
    from opendde_harness.memory_engine.outbox import MemoryOutbox

    workspace = tmp_path / "project"
    workspace.mkdir()
    store = MemoryStore(workspace)
    box = MemoryOutbox(workspace, store)
    box.append(turn_id="t1", session="cli:x", generation=0, messages=[{"role": "user", "content": "one"}])
    first = box.pending()[0]
    assert box.ack(first)
    box.append(turn_id="t2", session="cli:x", generation=0, messages=[{"role": "user", "content": "two"}])
    restarted = MemoryOutbox(workspace, MemoryStore(workspace))
    assert [e.turn_id for e in restarted.pending()] == ["t2"]
    assert not get_workspace_storage(workspace).host_memory.exists()
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("external", [False, True])
async def test_external_and_off_memory_do_not_inject_local_profile(tmp_path, external):
    from opendde_harness.context_engine.base import AssemblyContext
    from opendde_harness.context_engine.segments.memory import MemorySegmentBuilder
    from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore

    class Backend:
        async def recall(self, **kwargs):
            return [SimpleNamespace(text="external fact", id="one", score=1.0)]

    store = MemoryStore(tmp_path)
    store.write_long_term("local duplicate must stay out")
    segment = await MemorySegmentBuilder(store, Backend() if external else None).build(
        AssemblyContext("cli:x", "fact", None, None, None, [])
    )
    assert "local duplicate" not in segment.text
    assert ("external fact" in segment.text) == external


def test_equivalent_workspaces_share_sessions_but_other_workspaces_do_not(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    manager = SessionManager(workspace)
    session = manager.get_or_create("cli:one")
    session.messages.append({"role": "user", "content": "private"})
    manager.save(session)
    assert SessionManager(alias).get_or_create("cli:one").messages[0]["content"] == "private"
    assert SessionManager(tmp_path / "other").get_or_create("cli:one").messages == []


def test_storage_resolution_is_pure_and_uses_supplied_root(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage

    root = tmp_path / "custom-instance"
    workspace = tmp_path / "project"
    layout = get_workspace_storage(workspace, data_dir=root)
    assert layout.assistant == root / "assistant"
    assert layout.skills == root / "skills"
    assert layout.sessions.parent == root / "sessions"
    assert layout.memory_state.parent.parent == root / "state"
    assert not root.exists()
    assert not workspace.exists()


def test_default_export_stays_outside_workspace(tmp_path):
    from opendde_harness.session.export import default_export_path, write_transcript

    workspace = tmp_path / "project"
    workspace.mkdir()
    session = SessionManager(workspace).get_or_create("cli:one")
    session.messages.append({"role": "user", "content": "export me"})
    path = write_transcript(session, default_export_path(workspace, session.key))
    assert "export me" in path.read_text()
    assert path.is_relative_to(Path.home() / ".opendde_harness" / "exports")
    assert list(workspace.iterdir()) == []


async def test_tools_and_child_agents_keep_execution_workspace(tmp_path):
    from opendde_harness.agent.loop import AgentLoop
    from opendde_harness.agent.loop.factory import AgentLoopSettings
    from tests.test_memory_outbox import _Provider, _request

    workspace = tmp_path / "project"
    workspace.mkdir()
    loop = AgentLoop(_Provider(), workspace, AgentLoopSettings(), interactive=False)
    try:
        bash = loop.tools.get("bash")
        write_tool = loop.tools.get("write")
        await write_tool.execute(path="hello.txt", content="hello")
        result = await bash.execute(command="pwd")
        assert str(workspace) in str(result)
        assert (workspace / "hello.txt").read_text() == "hello"
        assert loop.subagents.workspace == workspace
        await loop._process_message(_request())
        assert sorted(p.name for p in workspace.iterdir()) == ["hello.txt"]
    finally:
        await loop.close_mcp()


def test_custom_config_root_controls_sessions(tmp_path, monkeypatch):
    from opendde_harness.config import loader

    root = tmp_path / "custom"
    monkeypatch.setattr(loader, "_current_config_path", root / "config.json")
    workspace = tmp_path / "project"
    manager = SessionManager(workspace)
    manager.save(manager.get_or_create("cli:one"))
    assert len(list((root / "sessions").rglob("one.jsonl"))) == 1
    assert not workspace.exists()
