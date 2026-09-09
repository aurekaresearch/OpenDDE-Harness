"""Validate release metadata and render GitHub Release notes from CHANGELOG.md."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/aurekaresearch/OpenDDE-Harness"
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?")
HEADING_PATTERN = re.compile(r"^## \[([^]]+)\](?: - ([0-9]{4}-[0-9]{2}-[0-9]{2}))?$")


def release_notes(changelog: str, version: str) -> str:
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"Unsupported release version: {version}")

    lines = changelog.splitlines()
    headings = []
    references = []
    fence = ""
    for index, line in enumerate(lines):
        if fence:
            if re.fullmatch(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
                fence = ""
            continue
        if opening := re.match(r"^ {0,3}(`{3,}|~{3,})", line):
            fence = opening[1]
        elif heading := HEADING_PATTERN.fullmatch(line):
            headings.append((index, heading[1], heading[2]))
        elif re.match(r"^\[[^]]+\]:\s+\S+", line):
            references.append(index)

    versions = [heading[1] for heading in headings]
    if versions[:2] != ["Unreleased", version] or len(versions) != len(set(versions)):
        raise ValueError(f"CHANGELOG must start with [Unreleased] and [{version}], with no duplicate versions")
    start, _, released = headings[1]
    if released is None:
        raise ValueError(f"CHANGELOG release {version} needs a YYYY-MM-DD date")
    date.fromisoformat(released)
    end = headings[2][0] if len(headings) > 2 else len(lines)
    end = min([end, *(index for index in references if start < index < end)])
    body = "\n".join(lines[start + 1 : end]).strip()
    if (
        not body
        or body == "No changes yet."
        or not any(line.strip() and not line.startswith("#") for line in body.splitlines())
    ):
        raise ValueError(f"CHANGELOG release {version} is empty")
    if re.search(r"\b(?:TODO|TBD)\b", body, re.IGNORECASE):
        raise ValueError(f"CHANGELOG release {version} contains a placeholder")
    for label, url in (
        ("Unreleased", f"{REPOSITORY_URL}/compare/v{version}...HEAD"),
        (version, f"{REPOSITORY_URL}/releases/tag/v{version}"),
    ):
        if f"[{label}]: {url}" not in (lines[index] for index in references):
            raise ValueError(f"CHANGELOG link for [{label}] is missing or incorrect")
    return body + "\n"


def render_release_notes(root: Path = ROOT, tag: str | None = None) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match package version {version}")
    return release_notes((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require this tag to match the package version")
    parser.add_argument("--output", type=Path, help="Write release notes here instead of stdout")
    args = parser.parse_args()
    try:
        notes = render_release_notes(tag=args.tag)
        if args.output is None:
            sys.stdout.write(notes)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(notes, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
