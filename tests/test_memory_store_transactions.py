"""The memory store's two on-disk promises: atomic writes, preserved damage.

``user.md`` and the store's state file are the host's only durable memory
artifacts now that automatic extraction belongs to the plugin backend. Both go
through one write path, and that path has to survive a crash without leaving a
half-file where the profile was, and has to tell "no state yet" apart from
"state we cannot read" instead of quietly resetting to empty.
"""

import json
import os
from contextlib import contextmanager

import pytest
from loguru import logger

from opendde_harness.memory_engine.consolidate import consolidator as store_module
from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore

_PROFILE = "# Long-term Memory\n\n## Projects\n\n- the assay pipeline\n"


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path)


def _temp_siblings(store: MemoryStore) -> list[str]:
    return sorted(p.name for p in store.memory_dir.iterdir() if p.name.endswith(".tmp"))


@contextmanager
def warnings_logged() -> "list[str]":
    """Collect loguru WARNING lines; ``caplog`` never sees them."""
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    try:
        yield lines
    finally:
        logger.remove(sink)


# ── Atomic writes ──────────────────────────────────────────────────────


def test_a_write_that_never_reached_the_rename_leaves_the_profile_as_it_was(store, monkeypatch):
    """The interrupted write. The rename is the only publishing step, so a
    failure before it must leave the previous profile byte-for-byte intact and
    must say so rather than report a write that did not happen."""
    store.write_long_term(_PROFILE)

    def _die(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(store_module.os, "replace", _die)

    with pytest.raises(OSError):
        store.write_long_term("everything the user knows, lost")

    assert store.read_long_term() == _PROFILE
    assert store.memory_file.read_text(encoding="utf-8") == _PROFILE
    assert _temp_siblings(store) == [], "a write that failed cleans up after itself"


def test_a_temp_file_a_crashed_writer_left_behind_is_not_read_and_not_reused(store):
    """A process killed between the write and the rename leaves its temp file:
    no cleanup can run. Readers must ignore it, and the next write must not
    adopt it -- the failure a single fixed ``.tmp`` sibling allows."""
    store.write_long_term(_PROFILE)
    stray = store.memory_dir / (store.memory_file.name + ".abc123.tmp")
    stray.write_text("half a profile from a killed process", encoding="utf-8")

    assert store.read_long_term() == _PROFILE

    store.write_long_term(_PROFILE + "- and a second project\n")

    assert store.read_long_term().endswith("- and a second project\n")
    assert stray.read_text(encoding="utf-8") == "half a profile from a killed process"


def test_two_writes_never_share_a_temp_name(store, monkeypatch):
    """Unique temp names are the point: with a fixed one, a second writer
    truncates the first writer's half-written file in place."""
    seen: list[str] = []
    real = store_module.tempfile.mkstemp

    def _record(*args, **kwargs):
        fd, name = real(*args, **kwargs)
        seen.append(name)
        return fd, name

    monkeypatch.setattr(store_module.tempfile, "mkstemp", _record)

    store.write_long_term("one")
    store.write_long_term("two")

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert all(os.path.dirname(n) == str(store.memory_dir) for n in seen), "same filesystem as the target"


# ── Versioned state, damage preserved ──────────────────────────────────


def test_no_state_file_is_an_empty_store_and_leaves_nothing_behind(store):
    assert store.read_state() == {}
    assert not store.state_file.exists()
    assert list(store.memory_dir.parent.glob("state.json.damaged-*")) == []


def test_the_extraction_cursor_round_trips_through_the_state_file(store):
    assert store.read_extraction_cursor() == {}

    store.commit_extraction_cursor(seq=7, turn_id="turn-a")

    state = store.read_state()
    assert state["version"] == MemoryStore.STATE_VERSION
    assert store.read_extraction_cursor()["seq"] == 7
    assert store.read_extraction_cursor()["turn_id"] == "turn-a"
    assert store.read_extraction_cursor()["acked_at"]


def test_the_extraction_cursor_never_moves_backwards(store):
    """The outbox drains in order, so the cursor only ever advances. An
    out-of-order acknowledgement would otherwise replay every entry between."""
    store.commit_extraction_cursor(seq=9, turn_id="turn-i")
    store.commit_extraction_cursor(seq=4, turn_id="turn-earlier")

    assert store.read_extraction_cursor() == {
        **store.read_extraction_cursor(),
        "seq": 9,
        "turn_id": "turn-i",
    }


def test_unparseable_state_is_moved_aside_not_overwritten(store):
    """Corrupt JSON is evidence. The store starts from empty and says where it
    put the file it could not read, instead of silently resetting it -- which
    is what made an absent cursor and a damaged one look the same."""
    store.state_file.parent.mkdir(parents=True, exist_ok=True)
    store.state_file.write_text('{"version": 1, "pending_extra', encoding="utf-8")

    with warnings_logged() as lines:
        assert store.read_state() == {}

    aside = list(store.state_file.parent.glob("state.json.damaged-*"))
    assert len(aside) == 1
    assert aside[0].read_text(encoding="utf-8") == '{"version": 1, "pending_extra'
    assert not store.state_file.exists()
    logged = "\n".join(lines)
    assert "unusable" in logged and aside[0].name in logged

    store.commit_extraction_cursor(seq=1, turn_id="turn-a")
    assert store.read_extraction_cursor()["seq"] == 1


def test_state_from_a_version_this_build_does_not_know_is_preserved(store):
    store.state_file.parent.mkdir(parents=True, exist_ok=True)
    store.state_file.write_text(json.dumps({"version": 99, "whatever": True}), encoding="utf-8")

    with warnings_logged() as lines:
        assert store.read_state() == {}

    aside = list(store.state_file.parent.glob("state.json.damaged-*"))
    assert len(aside) == 1
    assert json.loads(aside[0].read_text(encoding="utf-8")) == {"version": 99, "whatever": True}
    assert "version 99" in "\n".join(lines)


def test_state_that_is_not_an_object_is_preserved_too(store):
    store.state_file.parent.mkdir(parents=True, exist_ok=True)
    store.state_file.write_text("[1, 2, 3]", encoding="utf-8")

    assert store.read_state() == {}
    assert len(list(store.state_file.parent.glob("state.json.damaged-*"))) == 1


def test_two_damaged_files_in_the_same_second_do_not_overwrite_each_other(store):
    store.state_file.parent.mkdir(parents=True, exist_ok=True)
    store.state_file.write_text("first", encoding="utf-8")
    store.read_state()
    store.state_file.write_text("second", encoding="utf-8")
    store.read_state()

    aside = sorted(p.read_text(encoding="utf-8") for p in store.state_file.parent.glob("state.json.damaged-*"))
    assert aside == ["first", "second"]


def test_a_cursor_update_holds_the_profile_lock(store, monkeypatch):
    """Cursor updates are read-modify-write across sessions, so they run under
    the same advisory lock every profile writer takes. A lock-free update is
    how two processes draining at once lose one of the two acknowledgements."""
    held: list[bool] = []
    real_locked = MemoryStore.locked

    def _watch(self):
        held.append(True)
        return real_locked(self)

    monkeypatch.setattr(MemoryStore, "locked", _watch)

    store.commit_extraction_cursor(seq=1, turn_id="turn-a")

    assert held, "commit_extraction_cursor did not take the lock"
