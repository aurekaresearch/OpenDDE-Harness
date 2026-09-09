"""Provision a private Node.js runtime for the terminal UI when the host has none."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx
from rich.console import Console

from opendde_harness.cli._download import DownloadError, download_file, new_client

# Pinned Node.js release. install.sh and install.ps1 read this line; keep its format.
NODE_VERSION = "22.20.0"
NODE_DIST = "https://nodejs.org/dist"
DISABLE_ENV = "OPENDDE_HARNESS_NO_NODE_INSTALL"
_PLATFORMS = {
    ("linux", "x86_64"): "linux-x64",
    ("linux", "amd64"): "linux-x64",
    ("linux", "aarch64"): "linux-arm64",
    ("linux", "arm64"): "linux-arm64",
    ("darwin", "x86_64"): "darwin-x64",
    ("darwin", "arm64"): "darwin-arm64",
    ("windows", "amd64"): "win-x64",
    ("windows", "x86_64"): "win-x64",
}
_MANUAL_OPTIONS = (
    "Install Node.js >= 22 yourself (https://nodejs.org/, nvm install 22, brew install node@22) and set "
    f"OPENDDE_HARNESS_NODE to its executable, or set {DISABLE_ENV}=1 to skip automatic installation."
)


class NodeRuntimeError(RuntimeError):
    pass


def runtime_root() -> Path:
    return Path(os.environ.get("OPENDDE_HARNESS_HOME", str(Path.home() / ".opendde_harness"))) / "runtime"


def provisioning_disabled() -> bool:
    return bool(os.environ.get(DISABLE_ENV)) or bool(os.environ.get("OPENDDE_HARNESS_NODE"))


def package_name(system: str | None = None, machine: str | None = None) -> str:
    key = ((system or platform.system()).lower(), (machine or platform.machine()).lower())
    try:
        return f"node-v{NODE_VERSION}-{_PLATFORMS[key]}"
    except KeyError:
        raise NodeRuntimeError(f"No official Node.js build exists for {key[0]}/{key[1]}. {_MANUAL_OPTIONS}") from None


def archive_name(package: str) -> str:
    return package + (".zip" if "-win-" in package else ".tar.gz")


def node_binary(directory: Path) -> Path:
    return directory / "node.exe" if "-win-" in directory.name else directory / "bin" / "node"


def expected_sha256(shasums: str, archive: str) -> str:
    for line in shasums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == archive:
            return parts[0].lower()
    raise NodeRuntimeError(f"SHASUMS256.txt does not list {archive}")


def fetch_text(url: str) -> str:
    with new_client() as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _extract(archive: Path, into: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(into, filter="data")


def install_node(
    console: Console | None = None,
    *,
    download: Callable[..., Path] = download_file,
    fetch: Callable[[str], str] = fetch_text,
    translate: Callable[[str, str], str] | None = None,
) -> Path:
    """Download, verify and unpack the pinned release under the runtime root; returns the node executable."""
    t = translate or (lambda en, zh: en)
    console = console or Console()
    package = package_name()
    archive = archive_name(package)
    url = f"{NODE_DIST}/v{NODE_VERSION}/{archive}"
    root = runtime_root()
    destination = root / package
    binary = node_binary(destination)
    console.print(
        t(
            "Node.js >= 22 was not found; installing the Node.js 22.x runtime for the terminal UI into ~/.opendde_harness/runtime.",
            "未找到 Node.js ≥ 22，正在为终端界面安装 Node.js 22.x 运行时到 ~/.opendde_harness/runtime 。",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".node-", dir=root))
    try:
        try:
            digest = expected_sha256(fetch(f"{NODE_DIST}/v{NODE_VERSION}/SHASUMS256.txt"), archive)
            download([url], staging / archive, sha256=digest, description=archive, console=console)
        except (httpx.HTTPError, DownloadError, NodeRuntimeError, OSError) as exc:
            raise NodeRuntimeError(
                f"Node.js download or verification failed for {url}: {exc}. {_MANUAL_OPTIONS}"
            ) from exc
        try:
            _extract(staging / archive, staging)
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            raise NodeRuntimeError(f"Could not unpack {url}: {exc}. {_MANUAL_OPTIONS}") from exc
        if not node_binary(staging / package).is_file():
            raise NodeRuntimeError(f"{url} did not contain {package}; nothing was installed. {_MANUAL_OPTIONS}")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging / package, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    try:
        version = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise NodeRuntimeError(
            f"The downloaded Node.js runtime cannot run on this machine ({exc}); a libc mismatch such as Alpine/musl is the usual cause. "
            + _MANUAL_OPTIONS
        ) from exc
    console.print(t(f"Node.js {version} installed at {binary}", f"Node.js {version} 已安装到 {binary}"))
    return binary


__all__ = [
    "DISABLE_ENV",
    "NODE_VERSION",
    "NodeRuntimeError",
    "expected_sha256",
    "install_node",
    "node_binary",
    "package_name",
    "provisioning_disabled",
    "runtime_root",
]
