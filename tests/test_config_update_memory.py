from pathlib import Path

import pytest

from opendde_harness.config import loader, update_memory
from opendde_harness.plugin.memory.longterm import _library


def test_toml_write_leaves_no_lock_directory_in_memory_root(tmp_path: Path) -> None:
    path = tmp_path / "memory" / _library.CONFIG_FILENAME
    update_memory._write_atomic(path, {"server": {"port": 18791}})
    assert path.read_text().startswith("[server]")
    assert not (path.parent / ".lock").exists()
    assert not path.with_suffix(".toml.tmp").exists()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    return tmp_path


def test_default_root_is_the_memory_dir_on_a_fresh_install(data_dir: Path) -> None:
    assert update_memory.default_memory_root() == data_dir / "memory"
