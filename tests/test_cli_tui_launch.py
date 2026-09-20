"""What `ddeharness tui` spawns, and whether the bundle it needs ships.

There is one terminal UI and one bundle. The launcher's whole job is to find
that bundle -- packaged in the wheel, or built in a source checkout -- and hand
it to Node.
"""

import json
import os
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from opendde_harness import node_runtime
from opendde_harness.cli import onboard_commands, tui_commands
from tests._config import config_dict, keyed

REPO_ROOT = node_runtime.ui_tui_dir().parent
PACKAGED_ENTRY = node_runtime.packaged_dist_dir() / "entry.js"

#: Enough to launch: one provider with a key, and a default model it serves.
#: Written by the one helper that knows the providers section's shape.
MINIMAL_CONFIG = config_dict(keyed("openai", key="sk-test"), model="openai/gpt-4o-mini")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".opendde_harness" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(MINIMAL_CONFIG))
    return tmp_path


@pytest.fixture
def launch(monkeypatch):
    """Record what the launcher would spawn instead of spawning it."""
    calls = []
    monkeypatch.setattr(tui_commands, "find_node", lambda: ("node", (22, 19, 0)))
    monkeypatch.setattr(tui_commands, "resolve_dist", lambda name: node_runtime.packaged_dist_dir() / name)
    monkeypatch.setattr(tui_commands, "run_subprocess", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    monkeypatch.setattr(
        tui_commands, "run_subprocess_with_rpc", lambda *args, **kwargs: calls.append((args, kwargs)) or 0
    )
    monkeypatch.setattr("opendde_harness.cli.update_notice.maybe_refresh_async", lambda: None)
    monkeypatch.setattr(tui_commands, "_stdout_isatty", lambda: False)
    monkeypatch.setattr(onboard_commands, "run_wizard", lambda **kwargs: pytest.fail("wizard must not run"))
    return calls


def test_the_default_launch_runs_the_packaged_bundle(home, launch):
    result = CliRunner().invoke(tui_commands.tui_app, [])
    assert result.exit_code == 0
    (args, _kwargs) = launch[0]
    assert args[1] == [str(PACKAGED_ENTRY)]


def test_dev_runs_tsx_against_the_source(home, launch):
    result = CliRunner().invoke(tui_commands.tui_app, ["--dev"])
    assert result.exit_code == 0
    (args, kwargs) = launch[0]
    assert args[1] == ["tsx", "src/entry.ts"]
    assert kwargs["cwd"] == node_runtime.ui_tui_dir()


def test_check_is_a_stdio_smoke_spawn(home, launch, monkeypatch):
    monkeypatch.delenv("OPENDDE_HARNESS_TUI_CHECK", raising=False)
    result = CliRunner().invoke(tui_commands.tui_app, ["--check"])
    assert result.exit_code == 0
    (args, _kwargs) = launch[0]
    assert args[1] == [str(PACKAGED_ENTRY)]
    assert "run_subprocess_with_rpc" not in str(launch[0])


def test_a_missing_bundle_names_the_directory_to_build_in(home, launch, monkeypatch, capsys):
    monkeypatch.setattr(tui_commands, "resolve_dist", lambda name: None)
    result = CliRunner().invoke(tui_commands.tui_app, [])
    captured = capsys.readouterr()
    assert result.exit_code == 2
    assert "ui-tui" in result.output + captured.out + captured.err
    assert launch == []


def test_the_node_floor_is_what_pi_tui_requires(home, launch, monkeypatch, capsys):
    """pi-tui declares node >= 22.19.0, so 22.0 is not good enough."""
    monkeypatch.setattr(tui_commands, "find_node", lambda: ("node", (22, 0, 0)))
    monkeypatch.setattr(tui_commands.node_runtime, "provisioning_disabled", lambda: True)
    result = CliRunner().invoke(tui_commands.tui_app, [])
    captured = capsys.readouterr()
    assert result.exit_code == 1
    assert "22.19" in result.output + captured.out + captured.err
    assert launch == []


# ── Building the bundle from a checkout ────────────────────────────────


def _hatch_build():
    """The wheel build hook, importable without its build backend.

    Hatchling is a build dependency, not a runtime one, so it is absent from
    the environment the tests run in. Only the hook's base class comes from it
    and nothing here touches that class.
    """
    import importlib.util
    import sys
    import types

    if "hatchling" not in sys.modules:
        for name in ("hatchling", "hatchling.builders", "hatchling.builders.hooks", "hatchling.builders.hooks.plugin"):
            sys.modules.setdefault(name, types.ModuleType(name))
        iface = types.ModuleType("hatchling.builders.hooks.plugin.interface")
        iface.BuildHookInterface = object
        sys.modules["hatchling.builders.hooks.plugin.interface"] = iface

    spec = importlib.util.spec_from_file_location("hatch_build_under_test", REPO_ROOT / "hatch_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


#: Every bundle `npm run build` writes, read from the hook rather than repeated
#: here: a bundle added there must reach the wheel without editing this file.
BUNDLES = _hatch_build().BUNDLES


def _installed_tree(root, *, installed_at, locked_at):
    (root / "node_modules").mkdir(parents=True)
    lock = root / "package-lock.json"
    lock.write_text("{}\n")
    os.utime(lock, (locked_at, locked_at))
    if installed_at is not None:
        marker = root / "node_modules" / ".package-lock.json"
        marker.write_text("{}\n")
        os.utime(marker, (installed_at, installed_at))
    return root


def test_a_dependency_tree_older_than_its_lockfile_counts_as_stale(tmp_path):
    """Upgrading a checkout across the TUI rewrite is exactly this case.

    The sources become the new UI's while node_modules is still the old UI's,
    and npm, asked only to build, fails inside the wheel build with an esbuild
    resolution error that names neither npm nor this directory.
    """
    stale = _hatch_build().node_modules_is_stale

    # Never installed at all.
    assert stale(tmp_path / "absent") is True
    assert stale(_installed_tree(tmp_path / "bare", installed_at=None, locked_at=100)) is True

    # Installed before the lockfile arrived, and installed after it.
    assert stale(_installed_tree(tmp_path / "old", installed_at=100, locked_at=200)) is True
    assert stale(_installed_tree(tmp_path / "fresh", installed_at=200, locked_at=100)) is False


def test_two_builds_of_one_checkout_install_once_and_never_overlap(tmp_path, monkeypatch):
    """`npm ci` empties node_modules before refilling it.

    Two builds that both find the tree stale would both start installing, and
    one install then deletes what the other build is reading. Staleness is a
    fact about the past, not a mutex, so the refresh is serialized per checkout
    and rechecked once the lock is held.
    """
    import threading
    import time
    from types import SimpleNamespace

    hb = _hatch_build()

    ui = tmp_path / "ui-tui"
    (ui / "src").mkdir(parents=True)
    (ui / "src" / "entry.ts").write_text("console.log(1)\n")
    (ui / "package.json").write_text("{}\n")
    (ui / "package-lock.json").write_text("{}\n")
    # Present but never installed for this lockfile: stale, so both callers
    # arrive wanting to install.
    (ui / "node_modules").mkdir()
    dist = ui / "dist"

    assert hb.node_modules_is_stale(ui) is True
    assert hb.bundle_is_stale(ui, dist) is True

    guard = threading.Lock()
    running = []
    ran = []

    def fake_run(cmd, cwd=None, check=False):
        step = cmd[1]
        with guard:
            ran.append((step, list(running)))
            running.append(step)
        # Wide enough that an unserialized second caller would land inside it.
        time.sleep(0.2)
        with guard:
            running.remove(step)
        if step == "run":
            dist.mkdir(parents=True, exist_ok=True)
            for bundle in BUNDLES:
                (dist / bundle).write_text("bundle\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(subprocess, "run", fake_run)

    said = []
    app = SimpleNamespace(display_info=said.append, display_warning=said.append)
    failures = []

    def refresh():
        try:
            hb._refresh_bundle(ui, dist, app)
        except Exception as exc:  # surfaced below rather than lost in a thread
            failures.append(exc)

    threads = [threading.Thread(target=refresh) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert failures == []
    assert all(not thread.is_alive() for thread in threads)

    # One install and one build, and neither started while anything else ran.
    assert [step for step, _ in ran] == ["ci", "run"]
    assert [concurrent for _, concurrent in ran] == [[], []]

    # The second caller waited, rechecked, and found the work already done.
    assert hb.bundle_is_stale(ui, dist) is False


# ── Distribution contents ──────────────────────────────────────────────

# ``uv build`` builds the wheel from the sdist, so a bundle the archive drops is
# a bundle the released wheel cannot ship — and ``dist/`` is gitignored, which
# an ``include`` pattern cannot undo. This builds both with the real Hatchling
# builders against the repo's real pyproject.toml and build hook, standing in a
# byte of placeholder JavaScript for the bundle so the build stays fast.
_BUILD = """
import json, sys, tarfile, tempfile, zipfile
from email.parser import Parser
from pathlib import Path
from hatchling.builders.sdist import SdistBuilder
from hatchling.builders.wheel import WheelBuilder

root, out = Path(sys.argv[1]), Path(sys.argv[2])
sdist = next(SdistBuilder(str(root)).build(directory=str(out)))
with tarfile.open(sdist) as archive:
    sdist_names = archive.getnames()
    unpacked = Path(tempfile.mkdtemp())
    archive.extractall(unpacked, filter="data")
with zipfile.ZipFile(next(WheelBuilder(str(next(unpacked.iterdir()))).build(directory=str(out)))) as wheel:
    metadata = Parser().parsestr(wheel.read(next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))).decode())
    print(json.dumps({"sdist": sdist_names, "wheel": wheel.namelist(),
        "extras": metadata.get_all("Provides-Extra"), "dependencies": metadata.get_all("Requires-Dist")}))
"""


def _stage_release_tree(root, uis=("ui-tui",)):
    """The repo's packaging config over a stub source tree.

    pyproject.toml, the build hook and .gitignore are the real ones — the VCS
    ignore rules are what decide whether ``dist/`` reaches the archive. Nothing
    else here needs to be real, and a full copy of the package would make this
    a slow test for no extra coverage.
    """
    for name in ("pyproject.toml", "hatch_build.py", ".gitignore"):
        shutil.copy2(REPO_ROOT / name, root / name)
    for name in ("README.md", "CHANGELOG.md", "LICENSE"):
        (root / name).write_text("stub\n")
    (root / "LICENSES").mkdir()
    (root / "LICENSES" / "MIT-stub.txt").write_text("stub\n")
    (root / "opendde_harness").mkdir()
    (root / "opendde_harness" / "__init__.py").write_text("")
    (root / "opendde_harness" / "._unwanted.py").write_bytes(b"AppleDouble metadata")
    backends = root / "opendde_harness" / "plugin" / "protein_design" / "servers" / "backends"
    backends.mkdir(parents=True)
    for name in ("pyrosetta_worker.py", "pyrosetta_analysis.py"):
        (backends / name).write_text("# backend\n")
    (root / "docs" / "examples").mkdir(parents=True)
    (root / "docs" / "pyrosetta.md").write_text("# PyRosetta setup\n")
    for name in ("crlf2_quickstart.yaml", "cacng1_quickstart.yaml"):
        (root / "docs" / "examples" / name).write_text("stub: true\n")
    presets = root / "docs" / "examples" / "loss_presets"
    presets.mkdir()
    for name in ("default_bounded_v1.yaml", "README.md"):
        shutil.copy2(REPO_ROOT / "docs" / "examples" / "loss_presets" / name, presets / name)
    (root / "docker").mkdir()
    for name in ("versions.env", "environment.json", "model-checksums.sha256", "ESM-LICENSE.txt"):
        (root / "docker" / name).write_text("stub\n")
    for name in uis:
        (root / name / "dist").mkdir(parents=True)
        for bundle in BUNDLES:
            (root / name / "dist" / bundle).write_text(f'console.log("{name}/{bundle}")\n')
        dependency = root / name / "node_modules" / "unwanted-dependency"
        dependency.mkdir(parents=True)
        for filename in ("README.md", "LICENSE"):
            (dependency / filename).write_text("Unwanted dependency metadata\n")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv supplies the Hatchling build environment")
def test_the_tui_bundle_reaches_the_sdist_and_the_wheel_built_from_it(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    _stage_release_tree(root)

    # Hatchling is the build backend, not a runtime dependency of the venv the
    # tests run in, so the build happens in an ephemeral uv environment.
    ephemeral = [shutil.which("uv"), "run", "--no-project", "--quiet", "--with", "hatchling"]
    built = subprocess.run(
        [*ephemeral, "python", "-", str(root), str(tmp_path / "out")],
        input=_BUILD,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    names = json.loads(built.stdout)

    for bundle in BUNDLES:
        assert any(n.endswith(f"/ui-tui/dist/{bundle}") for n in names["sdist"]), f"{bundle} missing from the sdist"

    # Every bundle, and nothing else of the package: no second copy, no sources.
    shipped = sorted(n for n in names["wheel"] if "ui-tui" in n)
    assert shipped == sorted(f"opendde_harness/ui-tui/dist/{b}" for b in BUNDLES), shipped
    for archive in ("sdist", "wheel"):
        assert not any("node_modules" in n or any(p.startswith("._") for p in n.split("/")) for n in names[archive])
        assert any(n.endswith("/docs/pyrosetta.md") for n in names[archive])
        for resource in ("default_bounded_v1.yaml", "README.md"):
            assert any(n.endswith(f"/examples/loss_presets/{resource}") for n in names[archive])
        for module in ("pyrosetta_worker.py", "pyrosetta_analysis.py"):
            assert any(n.endswith(f"/plugin/protein_design/servers/backends/{module}") for n in names[archive])
    assert "pyrosetta" in names["extras"]
    assert "pyrosetta==2026.29+releasequarterly.80a0635615; extra == 'pyrosetta'" in names["dependencies"]


# The three tests that used to sit here held LiteLLM's loggers at WARNING, so
# its atexit handler could not print below the goodbye line. Nothing imports
# the library any more, so the handler is never registered and the record
# cannot be made; tests/test_providers_no_name_inference.py holds that
# stronger property.
