"""Only explicit, offline migration may move legacy managed data."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def migration(tmp_path, monkeypatch):
    from opendde_harness.config import workspace_migration as m
    from opendde_harness.config.paths import get_workspace_storage

    workspace = tmp_path / "project"
    workspace.mkdir()
    storage = get_workspace_storage(workspace, data_dir=tmp_path / "instance")
    # Unit fixtures have no runtime; process detection has its own tests.
    monkeypatch.setattr(m, "assert_offline", lambda: None)
    return m, workspace, storage


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_preview_is_read_only_and_migration_preserves_cursor_and_user_files(migration):
    m, workspace, storage = migration
    write(workspace / "user_memory/outbox.jsonl", "queue bytes\n")
    write(workspace / "user_memory/state.json", '{"version":1,"extraction_cursor":7}')
    write(workspace / "user_memory/profile/user.md", "unclassified private text")
    write(workspace / "designs/result.txt", "user work")
    write(workspace / "AGENTS.md", "project rules")
    (workspace / "user_memory/state.json").chmod(0o600)
    plan = m.preview_migration(workspace, storage)
    assert not storage.root.exists()
    result = m.apply_migration(plan)
    assert (storage.memory_state / "outbox.jsonl").read_text() == "queue bytes\n"
    assert (storage.memory_state / "state.json").read_text() == '{"version":1,"extraction_cursor":7}'
    assert (storage.memory_state / "state.json").stat().st_mode & 0o777 == 0o600
    assert not (workspace / "user_memory/state.json").exists()
    assert (workspace / "designs/result.txt").read_text() == "user work"
    assert (workspace / "AGENTS.md").read_text() == "project rules"
    assert not storage.host_memory.exists()
    assert result.requires_review
    manifest = json.loads(result.manifest.read_text())
    assert "unclassified private text" not in result.manifest.read_text()
    assert any(Path(row["backup"]).read_text() == "unclassified private text" for row in manifest["items"])
    assert m.apply_migration(m.preview_migration(workspace, storage)).manifest == result.manifest


def test_conflict_and_active_writer_leave_source_untouched(migration, monkeypatch):
    m, workspace, storage = migration
    write(workspace / "TOOLS.md", "old")
    write(storage.assistant / "TOOLS.md", "new")
    with pytest.raises(m.MigrationError, match="conflict"):
        m.apply_migration(m.preview_migration(workspace, storage))
    assert (workspace / "TOOLS.md").read_text() == "old"
    assert (storage.assistant / "TOOLS.md").read_text() == "new"

    def active():
        raise m.MigrationError("active writer PID 123")

    monkeypatch.setattr(m, "assert_offline", active)
    with pytest.raises(m.MigrationError, match="active writer"):
        m.apply_migration(m.preview_migration(workspace, storage))
    assert (workspace / "TOOLS.md").exists()


def test_interrupted_publication_blocks_startup_and_resumes_without_replay(migration, monkeypatch):
    m, workspace, storage = migration
    from opendde_harness.config.paths import assert_storage_ready

    write(workspace / "user_memory/outbox.jsonl", "queue")
    write(workspace / "user_memory/state.json", "cursor")
    real_copy = m.copy_verified
    count = 0

    def interrupted(source, target):
        nonlocal count
        real_copy(source, target)
        if target.is_relative_to(storage.memory_state):
            count += 1
            if count == 1:
                raise OSError("power loss")

    monkeypatch.setattr(m, "copy_verified", interrupted)
    with pytest.raises(OSError, match="power loss"):
        m.apply_migration(m.preview_migration(workspace, storage))
    assert (workspace / "user_memory/outbox.jsonl").exists()
    assert (workspace / "user_memory/state.json").exists()
    with pytest.raises(RuntimeError, match="migration"):
        assert_storage_ready(storage)
    monkeypatch.setattr(m, "copy_verified", real_copy)
    m.apply_migration(m.preview_migration(workspace, storage))
    assert_storage_ready(storage)
    assert (storage.memory_state / "state.json").read_text() == "cursor"


def test_rollback_refuses_new_data_then_restores_unchanged_data(migration):
    m, workspace, storage = migration
    write(workspace / "sessions/cli/one.jsonl", "history")
    result = m.apply_migration(m.preview_migration(workspace, storage))
    extra = storage.sessions / "cli/two.jsonl"
    write(extra, "new history")
    with pytest.raises(m.MigrationError, match="new|changed"):
        m.rollback_migration(result.manifest)
    assert extra.read_text() == "new history"
    extra.unlink()
    m.rollback_migration(result.manifest)
    assert (workspace / "sessions/cli/one.jsonl").read_text() == "history"


def test_symlink_source_is_rejected_without_following_it(migration, tmp_path):
    m, workspace, storage = migration
    outside = tmp_path / "private"
    outside.write_text("private")
    (workspace / "TOOLS.md").symlink_to(outside)
    with pytest.raises(m.MigrationError, match="symlink"):
        m.preview_migration(workspace, storage)
    assert outside.read_text() == "private"


def test_unknown_process_state_is_rejected(monkeypatch):
    from opendde_harness.config import workspace_migration as m

    monkeypatch.setattr(m.sys, "platform", "unsupported")
    with pytest.raises(m.MigrationError, match="verify"):
        m.assert_offline()


def test_manifest_cannot_archive_arbitrary_user_files(migration):
    m, workspace, storage = migration
    write(workspace / "TOOLS.md", "tools")
    result = m.apply_migration(m.preview_migration(workspace, storage))
    write(workspace / "designs/result.txt", "user work")
    manifest = json.loads(result.manifest.read_text())
    row = manifest["items"][0]
    row["source"] = str(workspace / "designs/result.txt")
    row["backup"] = str(result.manifest.parent / "originals/designs/result.txt")
    row["destination"] = None
    row["requires_review"] = True
    result.manifest.write_text(json.dumps(manifest))
    with pytest.raises(m.MigrationError, match="unsafe manifest"):
        m.rollback_migration(result.manifest)
    assert (workspace / "designs/result.txt").read_text() == "user work"


def test_rollback_refuses_new_shared_assistant_data(migration):
    m, workspace, storage = migration
    write(workspace / "TOOLS.md", "tools")
    result = m.apply_migration(m.preview_migration(workspace, storage))
    write(storage.assistant / "agent.md", "new instructions")
    with pytest.raises(m.MigrationError, match="new"):
        m.rollback_migration(result.manifest)
    assert (storage.assistant / "agent.md").read_text() == "new instructions"


def test_cli_preview_creates_no_directories(migration, monkeypatch):
    from typer.testing import CliRunner

    from opendde_harness.cli.commands import app
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", None)
    _, workspace, storage = migration
    write(workspace / "TOOLS.md", "secret instructions")
    result = CliRunner().invoke(
        app, ["workspace", "migrate", "--workspace", str(workspace), "--config", str(storage.root / "config.json")]
    )
    assert result.exit_code == 0, result.output
    assert "Preview only" in result.output
    assert "secret instructions" not in result.output
    assert not storage.root.exists()


def test_real_other_python_process_prevents_migration():
    import subprocess
    import sys

    from opendde_harness.config import workspace_migration as m

    if sys.platform != "linux":
        pytest.skip("Linux process scanner; unsupported platforms fail closed")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(m.MigrationError, match="active writer/runtime"):
            m.assert_offline()
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_interrupted_rollback_cannot_be_unfenced_by_apply(migration, monkeypatch):
    m, workspace, storage = migration
    from opendde_harness.config.paths import assert_storage_ready

    write(workspace / "user_memory/outbox.jsonl", "queue")
    write(workspace / "user_memory/state.json", "cursor")
    result = m.apply_migration(m.preview_migration(workspace, storage))
    original = Path.unlink

    def interrupt(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path == storage.memory_state / "outbox.jsonl":
            raise OSError("power loss during rollback")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt)
        with pytest.raises(OSError, match="power loss"):
            m.rollback_migration(result.manifest)
    with pytest.raises(m.MigrationError, match="rollback"):
        m.apply_migration(m.preview_migration(workspace, storage))
    with pytest.raises(RuntimeError, match="unfinished"):
        assert_storage_ready(storage)
    m.rollback_migration(result.manifest)
    assert_storage_ready(storage)
    assert (workspace / "user_memory/outbox.jsonl").read_text() == "queue"
    assert (workspace / "user_memory/state.json").read_text() == "cursor"


def test_empty_archive_from_interrupted_manifest_creation_is_resumable(migration):
    m, workspace, storage = migration
    write(workspace / "TOOLS.md", "tools")
    (storage.root / "migrations" / storage.scope).mkdir(parents=True)
    result = m.apply_migration(m.preview_migration(workspace, storage))
    assert result.manifest.exists()
    assert (storage.assistant / "TOOLS.md").read_text() == "tools"
