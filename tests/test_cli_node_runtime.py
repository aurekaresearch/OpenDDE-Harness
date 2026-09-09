import hashlib
import io
import shutil
import tarfile

import pytest
from rich.console import Console

from opendde_harness.cli import node_runtime, tui_commands
from opendde_harness.cli._download import DownloadError

PACKAGE = f"node-v{node_runtime.NODE_VERSION}-linux-x64"
ARCHIVE = PACKAGE + ".tar.gz"
NOTICE = "Node.js >= 22 was not found; installing the Node.js 22.x runtime"


@pytest.fixture
def runtime_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDDE_HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for name in ("VIRTUAL_ENV", "OPENDDE_HARNESS_NODE", node_runtime.DISABLE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(node_runtime, "package_name", lambda *a, **k: PACKAGE)
    return tmp_path / "home" / "runtime"


@pytest.fixture
def tarball(tmp_path):
    source = tmp_path / "build" / PACKAGE
    (source / "bin").mkdir(parents=True)
    node = source / "bin" / "node"
    node.write_text(f"#!/bin/sh\necho v{node_runtime.NODE_VERSION}\n")
    node.chmod(0o755)
    archive = tmp_path / ARCHIVE
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname=PACKAGE)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def _console():
    buffer = io.StringIO()
    return Console(file=buffer, width=400, force_terminal=False), buffer


def _fake_download(archive):
    def download(sources, target, *, sha256=None, **kwargs):
        assert sources == [f"{node_runtime.NODE_DIST}/v{node_runtime.NODE_VERSION}/{ARCHIVE}"]
        if sha256 != hashlib.sha256(archive.read_bytes()).hexdigest():
            raise DownloadError("SHA256/size mismatch; the file was removed")
        shutil.copy(archive, target)
        return target

    return download


def test_install_provisions_runtime_that_find_node_picks(runtime_home, tarball):
    archive, digest = tarball
    console, buffer = _console()
    fetched = []

    def fetch(url):
        fetched.append(url)
        return f"{digest}  {ARCHIVE}\nother  {PACKAGE}.tar.xz\n"

    binary = node_runtime.install_node(console, download=_fake_download(archive), fetch=fetch)
    assert binary == runtime_home / PACKAGE / "bin" / "node" and binary.is_file()
    assert fetched == [f"{node_runtime.NODE_DIST}/v{node_runtime.NODE_VERSION}/SHASUMS256.txt"]
    assert tui_commands.find_node() == (str(binary), (22, 20, 0))
    output = buffer.getvalue()
    assert output.count(NOTICE) == 1
    assert f"Node.js v{node_runtime.NODE_VERSION} installed at {binary}" in output
    assert [path.name for path in runtime_home.iterdir()] == [PACKAGE]


def test_checksum_mismatch_installs_nothing(runtime_home, tarball):
    archive, _ = tarball
    console, buffer = _console()
    with pytest.raises(node_runtime.NodeRuntimeError) as caught:
        node_runtime.install_node(
            console, download=_fake_download(archive), fetch=lambda url: f"{'0' * 64}  {ARCHIVE}\n"
        )
    message = str(caught.value)
    assert f"{node_runtime.NODE_DIST}/v{node_runtime.NODE_VERSION}/{ARCHIVE}" in message
    assert "OPENDDE_HARNESS_NODE" in message and node_runtime.DISABLE_ENV in message
    assert list(runtime_home.iterdir()) == []
    assert tui_commands.find_node() == (None, None)


def test_missing_shasum_entry_and_unusable_binary_are_reported(runtime_home, tarball, monkeypatch):
    archive, digest = tarball
    console, _ = _console()
    with pytest.raises(node_runtime.NodeRuntimeError, match="does not list"):
        node_runtime.install_node(console, download=_fake_download(archive), fetch=lambda url: "")
    monkeypatch.setattr(
        node_runtime.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("Exec format error"))
    )
    with pytest.raises(node_runtime.NodeRuntimeError, match="cannot run on this machine"):
        node_runtime.install_node(console, download=_fake_download(archive), fetch=lambda url: f"{digest}  {ARCHIVE}\n")
    assert list(runtime_home.iterdir()) == []


@pytest.mark.parametrize(
    "system, machine, expected",
    [
        ("Linux", "x86_64", "linux-x64"),
        ("Linux", "aarch64", "linux-arm64"),
        ("Darwin", "arm64", "darwin-arm64"),
        ("Darwin", "x86_64", "darwin-x64"),
        ("Windows", "AMD64", "win-x64"),
    ],
)
def test_package_names_follow_the_official_layout(system, machine, expected):
    package = node_runtime.package_name(system, machine)
    assert package == f"node-v{node_runtime.NODE_VERSION}-{expected}"
    assert node_runtime.archive_name(package).endswith(".zip" if expected.startswith("win") else ".tar.gz")


def test_unsupported_platform_names_manual_options():
    with pytest.raises(node_runtime.NodeRuntimeError, match="OPENDDE_HARNESS_NODE"):
        node_runtime.package_name("FreeBSD", "amd64")


def test_provisioning_is_disabled_by_env_or_explicit_node(monkeypatch):
    for name in ("OPENDDE_HARNESS_NODE", node_runtime.DISABLE_ENV):
        monkeypatch.delenv(name, raising=False)
    assert not node_runtime.provisioning_disabled()
    monkeypatch.setenv(node_runtime.DISABLE_ENV, "1")
    assert node_runtime.provisioning_disabled()
    monkeypatch.delenv(node_runtime.DISABLE_ENV)
    monkeypatch.setenv("OPENDDE_HARNESS_NODE", "/opt/node/bin/node")
    assert node_runtime.provisioning_disabled()
