"""Where Node is, and where the JavaScript bundles it runs are.

Both the terminal UI (``ddeharness tui``) and the model layer
(:mod:`opendde_harness.providers.model_service`) launch a Node child against a
bundle out of ``ui-tui/dist/``. They need the same two answers, so the answers
live here rather than inside the Typer command that happened to ask first:
importing ``opendde_harness.cli.tui_commands`` pulls in typer for the sake of
two paths, and the model service is on the hot path of an ordinary chat turn.

Stdlib only, deliberately — this module must stay cheap to import.

Not to be confused with :mod:`opendde_harness.cli.node_runtime`, which
*installs* a private Node when the host has none; this one only ever looks.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

#: The floor pi-tui declares in ui-tui/package.json (``engines.node``).
MIN_NODE_VERSION = (22, 19, 0)


def ui_tui_dir() -> Path:
    """The ``ui-tui/`` source tree of a checkout, absent from a wheel.

    opendde_harness/node_runtime.py -> ../ui-tui/. Only ``tui --dev`` (tsx from
    source) needs the tree itself; everything else wants a built bundle.
    """
    return Path(__file__).resolve().parent.parent / "ui-tui"


def packaged_dist_dir() -> Path:
    """Where an installed wheel keeps the prebuilt bundles.

    opendde_harness/node_runtime.py -> ./ui-tui/dist (i.e.
    opendde_harness/ui-tui/dist). pyproject force-includes ui-tui/dist here so a
    `pip`/`uv tool install` ships them without a source checkout.
    """
    return Path(__file__).resolve().parent / "ui-tui" / "dist"


def resolve_dist(name: str) -> Optional[Path]:
    """Locate one prebuilt bundle (``entry.js``, ``model-service.js``).

    Tries, in order:
      1. The packaged copy inside the installed wheel (``opendde_harness/ui-tui/dist``).
      2. The source-tree copy a developer built locally (``ui-tui/dist``).

    The bundles are self-contained (esbuild ``bundle: true``), so no sibling
    ``node_modules`` is needed — only a Node runtime. Returns the first path
    that exists, or ``None`` if neither does.
    """
    for candidate in (packaged_dist_dir() / name, ui_tui_dir() / "dist" / name):
        if candidate.exists():
            return candidate
    return None


def _is_windows() -> bool:
    """Whether the host is Windows. A seam so tests can exercise the Windows
    runtime-layout branch by patching this, instead of patching os.name — the
    latter makes pathlib instantiate an unusable WindowsPath on POSIX hosts."""
    return os.name == "nt"


def find_node() -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    """Find a usable node executable (>= 22.19).

    Returns (path, version_tuple) or (None, None) if not found.
    """
    # Priority 1: OPENDDE_HARNESS_NODE env var — explicit override, NO fallback.
    # When the user sets OPENDDE_HARNESS_NODE they are forcing a specific binary;
    # if it is missing or unusable we must NOT silently fall back to
    # venv/PATH (that would mask misconfiguration).
    candidates: list[str] = []
    if env_node := os.environ.get("OPENDDE_HARNESS_NODE"):
        candidates.append(env_node)
    else:
        # Priority 2: active venv
        if venv := os.environ.get("VIRTUAL_ENV"):
            if _is_windows():
                candidates.append(str(Path(venv) / "Scripts" / "node.exe"))
            else:
                candidates.append(str(Path(venv) / "bin" / "node"))

        # Priority 3: PATH — enumerate EVERY node on PATH, not just the first.
        # shutil.which returns only the first hit, so a stale < 22 node earlier
        # on PATH (e.g. an old /usr/local/bin/node or a version-manager shim)
        # would otherwise shadow a newer one later on PATH (e.g. a Homebrew
        # node 26). The version filter below then picks the first usable one.
        exe = "node.exe" if _is_windows() else "node"
        seen_path: set[str] = set()
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            if not path_dir:
                continue
            cand = os.path.join(path_dir, exe)
            if cand not in seen_path and os.path.isfile(cand):
                seen_path.add(cand)
                candidates.append(cand)

        # Priority 4: nvm installations are not placed on PATH in detached or
        # non-interactive shells, even when the same account's interactive
        # shell uses them. Background workers must still be able to use that
        # already-installed runtime.
        nvm_root = Path(os.environ.get("NVM_DIR", str(Path.home() / ".nvm"))) / "versions" / "node"
        if nvm_root.is_dir():
            candidates.extend(str(path) for path in sorted(nvm_root.glob("v*/bin/node"), reverse=True))

        # Priority 5: OpenDDE Harness-managed private runtime installed by the one-line
        # installer into ~/.opendde_harness/runtime/. This is the zero-config fallback so
        # a user who has no system Node still gets a working `ddeharness tui` after
        # the installer provisioned a private Node here. Glob to tolerate the
        # versioned dir name. The on-disk layout differs by OS: POSIX tarballs
        # nest the binary under bin/ (node-v22.x.y-darwin-arm64/bin/node) while
        # the Windows zip puts node.exe at the top level
        # (node-v22.x.y-win-x64/node.exe) — install.ps1 provisions the latter.
        runtime_root = Path(os.environ.get("OPENDDE_HARNESS_HOME", Path.home() / ".opendde_harness")) / "runtime"
        if runtime_root.is_dir():
            if _is_windows():
                direct = runtime_root / "node" / "node.exe"
                if direct.exists():
                    candidates.append(str(direct))
                candidates.extend(str(p) for p in sorted(runtime_root.glob("node-*/node.exe")))
            else:
                direct = runtime_root / "node" / "bin" / "node"
                if direct.exists():
                    candidates.append(str(direct))
                candidates.extend(str(p) for p in sorted(runtime_root.glob("node-*/bin/node")))

    # Return the first candidate that meets the minimum, in priority order.
    # Track the highest below-minimum candidate seen so that when nothing
    # qualifies the caller can still report the real version ("found 20.20.1,
    # need >= 22.19") instead of a bare "not found".
    best_below_min: Optional[Tuple[str, Tuple[int, int, int]]] = None
    for node_path in candidates:
        if not Path(node_path).exists():
            continue
        try:
            proc = subprocess.run(
                [node_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            match = re.match(r"v(\d+)\.(\d+)\.(\d+)", proc.stdout.strip())
            if not match:
                continue
            version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue
        if version >= MIN_NODE_VERSION:
            return (node_path, version)
        if best_below_min is None or version > best_below_min[1]:
            best_below_min = (node_path, version)

    return best_below_min if best_below_min is not None else (None, None)
