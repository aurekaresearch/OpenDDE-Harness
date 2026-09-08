"""The third-party long-term memory library is named in exactly one module."""

import re
import subprocess
from pathlib import Path

from opendde_harness.plugin.memory.longterm import _library

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = Path("opendde_harness/plugin/memory/longterm/_library.py")
LIBRARY = _library.EXECUTABLE

# Files that may carry the library's name for reasons other than code: the
# dependency pin, the lockfile, the credit in the README, and this test.
ALLOWED_MENTIONS = {
    ADAPTER,
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("README.md"),
    Path(__file__).resolve().relative_to(ROOT),
}

_IMPORT = re.compile(rf"^\s*(?:from\s+{LIBRARY}(?:\.|\s)|import\s+{LIBRARY}(?:\.|\s|$))", re.MULTILINE)


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [Path(line) for line in out.splitlines() if line]


def _read(rel: Path) -> str | None:
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def test_only_the_adapter_module_imports_the_library() -> None:
    importers = []
    for rel in _tracked_text_files():
        if rel.suffix != ".py":
            continue
        text = _read(rel)
        if text is not None and _IMPORT.search(text):
            importers.append(rel)
    assert importers == [ADAPTER]


def test_the_library_is_named_nowhere_else() -> None:
    offenders = []
    for rel in _tracked_text_files():
        if rel in ALLOWED_MENTIONS or rel.parts[:2] == ("ui-tui", "node_modules"):
            continue
        text = _read(rel)
        if text is not None and LIBRARY in text.lower():
            offenders.append(str(rel))
    assert offenders == []


def test_pyproject_names_the_library_only_in_the_dependency_pin() -> None:
    lines = [line for line in _read(Path("pyproject.toml")).splitlines() if LIBRARY in line.lower()]
    assert len(lines) == 1 and lines[0].strip().startswith(f'"{LIBRARY}[')
