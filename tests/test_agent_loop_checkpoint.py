"""The shadow-git checkpoint: what it snapshots, and one commit at a time.

Two properties of the per-turn snapshot that nothing above it can check. What
the shadow repository tracks decides what the next turn's recovery prompt names,
so the harness's own journal has to be outside it. And the index is one file per
repository, so two turns finishing together have to take it in turns.
"""

from __future__ import annotations

import asyncio

import pytest

from opendde_harness.agent.loop.checkpoint import _DEFAULT_EXCLUDES, CheckpointService


async def _shadow(service: CheckpointService, *args: str) -> str:
    rc, out, _err = await service._git(*args)
    assert rc == 0, f"git {' '.join(args)} failed"
    return out


def _needs_git() -> None:
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")


async def test_shadow_metadata_is_outside_workspace_and_tracks_real_changes(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage

    _needs_git()
    (tmp_path / "answer.py").write_text("x = 1\n")
    service = CheckpointService(tmp_path)
    first, changes = await service.commit_turn("first")
    assert first and changes == ["answer.py"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["answer.py"]
    assert service._git_dir.is_relative_to(get_workspace_storage(tmp_path).checkpoint)
    (tmp_path / "answer.py").write_text("x = 2\n")
    second, changes = await service.commit_turn("second")
    assert second != first and changes == ["answer.py"]


def test_checkpoint_refuses_instance_root_inside_worktree(tmp_path, monkeypatch):
    from opendde_harness.config import loader

    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "internal" / "config.json")
    with pytest.raises(ValueError, match="outside"):
        CheckpointService(tmp_path)


def test_legacy_default_checkpoint_setting_uses_migrated_shadow(tmp_path):
    from opendde_harness.config.paths import get_workspace_storage

    service = CheckpointService(tmp_path, shadow_dir=".opendde_harness/shadow.git")
    assert service._git_dir == get_workspace_storage(tmp_path).checkpoint / "shadow.git"


@pytest.mark.parametrize("destination", ["workspace", "other-scope"])
def test_checkpoint_symlink_cannot_escape_its_scope(tmp_path, destination):
    from opendde_harness.config.paths import get_workspace_storage

    storage = get_workspace_storage(tmp_path)
    target = tmp_path / "cache" if destination == "workspace" else storage.root / "state" / "other" / "checkpoint"
    target.mkdir(parents=True)
    storage.checkpoint.parent.mkdir(parents=True)
    storage.checkpoint.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|scope|outside"):
        CheckpointService(tmp_path)


async def test_the_snapshot_leaves_the_harnesss_own_state_untracked(tmp_path) -> None:
    """The journal, the lock beside it and the extraction queue are ours.

    The agent did not write them and cannot be asked to verify them, so a
    snapshot that tracked them would put them in the next turn's recovery
    prompt as files it had just edited.
    """
    _needs_git()
    (tmp_path / "sessions" / "tui" / ".lock").mkdir(parents=True)
    (tmp_path / "sessions" / "tui" / "chat.jsonl").write_text('{"role":"user"}\n', encoding="utf-8")
    (tmp_path / "sessions" / "tui" / ".lock" / "chat.jsonl.lock").write_text("", encoding="utf-8")
    (tmp_path / "user_memory").mkdir()
    (tmp_path / "user_memory" / "outbox.jsonl").write_text('{"seq":1}\n', encoding="utf-8")
    (tmp_path / "answer.py").write_text("print(42)\n", encoding="utf-8")

    service = CheckpointService(tmp_path)
    cid, changed = await service.commit_turn("turn one")

    assert cid is not None
    assert changed == ["answer.py"], "only the file the agent wrote is this turn's change"


async def test_a_project_directory_named_sessions_is_still_snapshotted(tmp_path) -> None:
    """The exclusion is the workspace's own ``sessions``, not the name anywhere.

    ``/sessions/`` is anchored for exactly this reason: a project that keeps its
    own ``src/sessions`` must not lose it from the safety net because the
    harness happens to use the word.
    """
    _needs_git()
    (tmp_path / "src" / "sessions").mkdir(parents=True)
    (tmp_path / "src" / "sessions" / "store.py").write_text("x = 1\n", encoding="utf-8")

    service = CheckpointService(tmp_path)
    cid, changed = await service.commit_turn("turn one")

    assert cid is not None
    assert changed == ["src/sessions/store.py"]


def test_the_excludes_name_the_journal_and_its_lock() -> None:
    """Stated as a list so the reason survives an edit to the block."""
    assert "/sessions/" in _DEFAULT_EXCLUDES
    assert ".lock/" in _DEFAULT_EXCLUDES
    assert "/user_memory/outbox.jsonl" in _DEFAULT_EXCLUDES


async def test_two_lanes_committing_at_once_take_the_index_in_turns(tmp_path) -> None:
    """Two turns finishing together do not interleave their git invocations.

    One index per repository: ``add -A`` then ``diff --cached`` then ``commit``
    is a sequence another lane's ``add`` cannot be allowed inside, or the second
    caller collides with the first's ``index.lock`` and the snapshot degrades to
    nothing. Two services on one shadow path, because the index belongs to the
    repository rather than to whoever opened it.
    """
    _needs_git()
    (tmp_path / "one.py").write_text("one\n", encoding="utf-8")
    first = CheckpointService(tmp_path)
    second = CheckpointService(tmp_path)
    order: list[str] = []
    failed: list[tuple[str, str, str]] = []

    def tagged(service: CheckpointService, tag: str):
        original = service._git

        async def recording(*args: str):
            order.append(tag)
            rc, out, err = await original(*args)
            if rc != 0:
                failed.append((tag, args[0], err.strip()))
            return rc, out, err

        return recording

    first._git = tagged(first, "first")  # type: ignore[method-assign]
    second._git = tagged(second, "second")  # type: ignore[method-assign]

    async def lane(service: CheckpointService, name: str):
        (tmp_path / name).write_text(name, encoding="utf-8")
        return await service.commit_turn(f"turn {name}")

    results = await asyncio.gather(lane(first, "a.py"), lane(second, "b.py"))

    # Two ways the collision shows, and both are checked because each on its own
    # can be survived by luck: the calls interleave, or one lane's invocation is
    # refused by the other's lock and its snapshot silently degrades to nothing.
    handovers = sum(1 for before, after in zip(order, order[1:]) if before != after)
    assert handovers == 1, f"the two lanes' git calls interleaved: {order}"
    assert failed == [], f"a git invocation was refused: {failed}"
    assert any(cid for cid, _ in results), "at least one lane snapshotted the tree"
    # The snapshot is of the working tree, which both lanes share: whoever takes
    # the index first commits everything on disk, and the second finds nothing of
    # its own left to record.
    tracked = await _shadow(first, "ls-tree", "-r", "--name-only", "HEAD")
    assert {"a.py", "b.py", "one.py"} <= set(tracked.split())
