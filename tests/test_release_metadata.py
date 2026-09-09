from pathlib import Path

import pytest

from scripts.render_release_notes import REPOSITORY_URL, release_notes, render_release_notes

ROOT = Path(__file__).resolve().parents[1]


def changelog(version="0.0.1", body="### Added\n\n- Initial preview.", released="2026-09-09"):
    return (
        f"# Changelog\n\n## [Unreleased]\n\nNo changes yet.\n\n"
        f"## [{version}] - {released}\n\n{body}\n\n"
        f"[Unreleased]: {REPOSITORY_URL}/compare/v{version}...HEAD\n"
        f"[{version}]: {REPOSITORY_URL}/releases/tag/v{version}\n"
    )


def test_current_release_metadata():
    notes = render_release_notes(ROOT)
    assert "OpenDDE Harness" in notes
    assert "No changes yet." not in notes
    assert "[Unreleased]" not in notes


def test_notes_exclude_other_releases_and_reference_footer():
    text = changelog().replace("[Unreleased]:", "## [0.0.0] - 2026-09-01\n\n- Previous version.\n\n[Unreleased]:", 1)
    assert release_notes(text, "0.0.1") == "### Added\n\n- Initial preview.\n"


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fenced_headings_are_not_release_boundaries(fence):
    body = f"### Added\n\n- Initial preview.\n\n{fence}markdown\n## [9.9.9] - 2026-01-01\n{fence}"
    assert release_notes(changelog(body=body), "0.0.1") == body + "\n"


@pytest.mark.parametrize("version", ["0.0.1", "0.0.1a1", "0.0.1b1", "0.0.1rc1"])
def test_matching_stable_and_prerelease_tags(tmp_path, version):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(changelog(version=version))
    assert render_release_notes(tmp_path, f"v{version}").startswith("### Added")


@pytest.mark.parametrize("tag", ["v0.0.2", "v0.0.1-rc1", "0.0.1", "main"])
def test_wrong_tag_is_rejected(tmp_path, tag):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.0.1"\n')
    with pytest.raises(ValueError, match="does not match"):
        render_release_notes(tmp_path, tag)


@pytest.mark.parametrize("version", ["0.0", "0.0.1+local", "0.0.1-rc1"])
def test_invalid_version_is_rejected(version):
    with pytest.raises(ValueError, match="Unsupported release version"):
        release_notes(changelog(version=version), version)


@pytest.mark.parametrize("body", ["", "No changes yet.", "### Added", "- TODO", "- TBD"])
def test_empty_or_placeholder_notes_are_rejected(body):
    with pytest.raises(ValueError, match="empty|placeholder"):
        release_notes(changelog(body=body), "0.0.1")


@pytest.mark.parametrize("released", ["2026-02-30", "not-a-date"])
def test_invalid_date_is_rejected(released):
    with pytest.raises(ValueError):
        release_notes(changelog(released=released), "0.0.1")


def test_missing_date_is_rejected():
    with pytest.raises(ValueError, match="needs a YYYY-MM-DD date"):
        release_notes(changelog().replace(" - 2026-09-09", ""), "0.0.1")


@pytest.mark.parametrize("heading", ["## [Unreleased]", "## [0.0.1] - 2026-09-09"])
def test_duplicate_sections_are_rejected(heading):
    with pytest.raises(ValueError, match="no duplicate versions"):
        release_notes(changelog() + f"\n{heading}\n\n- Duplicate.", "0.0.1")


def test_missing_or_outdated_release_is_rejected():
    with pytest.raises(ValueError, match="must start with"):
        release_notes(changelog(version="0.0.2"), "0.0.1")


@pytest.mark.parametrize("old", ["compare/v0.0.1...HEAD", "releases/tag/v0.0.1"])
def test_wrong_release_links_are_rejected(old):
    with pytest.raises(ValueError, match="missing or incorrect"):
        release_notes(changelog().replace(old, "wrong-link"), "0.0.1")
