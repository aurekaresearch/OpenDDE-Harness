"""Finding the built bundles, and staying cheap enough to import on a chat turn.

The model service resolves its bundle on every start, so this lookup must not
drag the whole CLI (and typer with it) into the process.
"""

import os
import subprocess
import sys
from pathlib import Path

from opendde_harness import node_runtime


def _built(directory, name="model-service.js"):
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / name
    bundle.write_text("console.log('bundle')\n")
    return bundle


def test_the_packaged_bundle_wins_over_a_source_build(tmp_path, monkeypatch):
    """An installed wheel's own copy is the one it must run."""
    packaged = _built(tmp_path / "wheel" / "ui-tui" / "dist")
    _built(tmp_path / "checkout" / "ui-tui" / "dist")

    monkeypatch.setattr(node_runtime, "packaged_dist_dir", lambda: packaged.parent)
    monkeypatch.setattr(node_runtime, "ui_tui_dir", lambda: tmp_path / "checkout" / "ui-tui")

    assert node_runtime.resolve_dist("model-service.js") == packaged


def test_a_source_build_answers_when_nothing_is_packaged(tmp_path, monkeypatch):
    """The developer flow: no wheel copy, a bundle built in the checkout."""
    source = _built(tmp_path / "checkout" / "ui-tui" / "dist")

    monkeypatch.setattr(node_runtime, "packaged_dist_dir", lambda: tmp_path / "wheel" / "ui-tui" / "dist")
    monkeypatch.setattr(node_runtime, "ui_tui_dir", lambda: tmp_path / "checkout" / "ui-tui")

    assert node_runtime.resolve_dist("model-service.js") == source
    assert node_runtime.resolve_dist("entry.js") is None


def test_neither_copy_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(node_runtime, "packaged_dist_dir", lambda: tmp_path / "wheel")
    monkeypatch.setattr(node_runtime, "ui_tui_dir", lambda: tmp_path / "checkout")

    assert node_runtime.resolve_dist("model-service.js") is None


def test_the_two_directories_sit_where_the_wheel_and_the_checkout_put_them():
    """One lives inside the installed package, the other beside the checkout."""
    packaged = node_runtime.packaged_dist_dir()
    assert packaged.name == "dist"
    assert packaged.parent.name == "ui-tui"
    assert packaged.parent.parent.name == "opendde_harness"

    source = node_runtime.ui_tui_dir()
    assert source.name == "ui-tui"
    assert source.parent == packaged.parent.parent.parent


def test_importing_it_costs_nothing_from_the_cli():
    """Importing the CLI pulls in typer (~0.2 s); the model service must not pay it."""
    root = Path(node_runtime.__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    probe = (
        "import sys; import opendde_harness.node_runtime; "
        "assert 'typer' not in sys.modules, sorted(m for m in sys.modules if 'typer' in m); "
        "assert not [m for m in sys.modules if m.startswith('opendde_harness.cli')], 'the CLI came too'; "
        "print('clean')"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "clean"


def test_find_node_discovers_nvm_runtime_without_interactive_shell_path(tmp_path, monkeypatch):
    node = tmp_path / "nvm" / "versions" / "node" / "v22.23.2" / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text("#!/bin/sh\necho v22.23.2\n")
    node.chmod(0o755)
    monkeypatch.setenv("NVM_DIR", str(tmp_path / "nvm"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("OPENDDE_HARNESS_NODE", raising=False)
    monkeypatch.setenv("OPENDDE_HARNESS_HOME", str(tmp_path / "no-managed-runtime"))

    path, version = node_runtime.find_node()

    assert path == str(node)
    assert version == (22, 23, 2)
