import io
import os
import shutil
import tarfile

import pytest

from opendde_harness.cli import compute_code
from opendde_harness.cli.compute_environment import load_environment

SOURCE_FILES = {
    "opendde": {"runner/inference.py": "print('fold')\n"},
    "ligandmpnn": {"model_utils.py": "tool source\n", "data_utils.py": "tool source\n"},
    "plip": {"plip/plipcmd.py": "tool source\n"},
}


@pytest.fixture
def installed_package(tmp_path, monkeypatch):
    package = tmp_path / "installed/opendde_harness"
    (package / "cli").mkdir(parents=True)
    (package / "__init__.py").write_text("release = 1\n")
    (package / "cli/compute_code.py").write_text("# runtime preparation\n")
    monkeypatch.setattr(compute_code, "__file__", str(package / "cli/compute_code.py"))
    monkeypatch.setattr(compute_code.importlib.metadata, "version", lambda _: "1.0.0")
    return package


@pytest.fixture
def installed_code(installed_package, monkeypatch):
    def prepare(name, _revision, destination, _upstream, **_kwargs):
        for name in SOURCE_FILES[name]:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("tool source\n")

    monkeypatch.setattr(compute_code, "_prepare_upstream", prepare)
    return installed_package


@pytest.fixture
def fake_download(monkeypatch):
    calls = []

    def download(sources, target, **_kwargs):
        name = next(name for name, repo in compute_code.REPOSITORIES.items() if f"/{repo}/" in sources[0])
        calls.append(name)
        target.write_bytes(_source_archive(SOURCE_FILES[name]))
        return target

    monkeypatch.setattr(compute_code, "download_file", download)
    return calls


def _source_archive(files: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, text in files.items():
            member = tarfile.TarInfo(f"repo-abc/{name}")
            member.size = len(text.encode())
            archive.addfile(member, io.BytesIO(text.encode()))
    return stream.getvalue()


def _serve(monkeypatch, handler):
    import httpx

    from opendde_harness.cli import _download

    monkeypatch.setattr(_download, "new_client", lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))


def test_installed_code_prepares_and_reuses_without_checkout(tmp_path, installed_code):
    root = compute_code.prepare_runtime_code(tmp_path / "cache")
    assert root.name.startswith("1.0.0-")
    assert not (root / ".git").exists()
    assert not list(root.glob("*.dist-info"))
    identity = compute_code.verify_runtime_code(root)
    assert identity["environment_id"] == load_environment()["id"]
    assert compute_code.prepare_runtime_code(tmp_path / "cache") == root
    assert (root / "opendde_harness/cli/compute_environment.json").is_file()
    assert (root / "opendde_harness/cli/model-checksums.sha256").is_file()


def test_installed_code_creates_the_external_namespace_without_a_checkout(tmp_path, installed_code):
    assert not (installed_code.parent / "external").exists()
    root = compute_code.prepare_runtime_code(tmp_path / "cache")
    assert (root / "external/__init__.py").stat().st_size > 0
    assert (root / "external/opendde/runner/inference.py").is_file()
    assert compute_code.verify_runtime_code(root)["harness_version"] == "1.0.0"


def test_code_upgrade_keeps_old_snapshot_and_environment(tmp_path, installed_code):
    first = compute_code.prepare_runtime_code(tmp_path / "cache")
    old = (first / "opendde_harness/__init__.py").read_bytes()
    (installed_code / "__init__.py").write_text("release = 2\n")
    second = compute_code.prepare_runtime_code(tmp_path / "cache")
    assert first != second
    assert (first / "opendde_harness/__init__.py").read_bytes() == old
    assert compute_code.verify_runtime_code(first)["environment_sha256"] == compute_code.verify_runtime_code(second)["environment_sha256"]


def test_corrupt_cached_code_is_not_overwritten(tmp_path, installed_code):
    root = compute_code.prepare_runtime_code(tmp_path / "cache")
    source = root / "opendde_harness/__init__.py"
    source.write_text("local edit\n")
    with pytest.raises(ValueError, match="changed or is incomplete"):
        compute_code.prepare_runtime_code(tmp_path / "cache")
    assert source.read_text() == "local edit\n"


def test_failed_source_preparation_is_not_published(tmp_path, installed_code, monkeypatch):
    def fail(*_, **__):
        raise ValueError("download failed")

    monkeypatch.setattr(compute_code, "_prepare_upstream", fail)
    with pytest.raises(ValueError, match="download failed"):
        compute_code.prepare_runtime_code(tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_upstream_download_reports_url_and_extracts(tmp_path, monkeypatch, capsys):
    import httpx

    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=_source_archive({"runner/inference.py": "print('fold')\n"}))

    _serve(monkeypatch, handler)
    compute_code._prepare_upstream("opendde", "a" * 40, tmp_path / "opendde", None, position="1 of 3")
    assert (tmp_path / "opendde/runner/inference.py").read_text() == "print('fold')\n"
    assert seen == ["https://codeload.github.com/aurekaresearch/OpenDDE/tar.gz/" + "a" * 40]
    out = capsys.readouterr().out
    assert "Preparing tool sources (1 of 3): opendde (aaaaaaaaaaaa) from https://codeload.github.com/" in out
    assert not list(tmp_path.glob(".download-*"))


def test_upstream_download_stall_names_url_and_alternatives(tmp_path, monkeypatch):
    import httpx

    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    _serve(monkeypatch, handler)
    with pytest.raises(ValueError, match=r"codeload\.github\.com.*https_proxy.*--upstream-dir"):
        compute_code._prepare_upstream("plip", "b" * 40, tmp_path / "plip", None)
    assert not (tmp_path / "plip").exists()


@pytest.mark.parametrize("name", ["repo/../../outside", "/outside", "repo/C:/outside", "repo/..\\outside"])
def test_source_archive_cannot_escape_destination(tmp_path, name):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = 4
        archive.addfile(member, io.BytesIO(b"data"))
    stream.seek(0)
    with pytest.raises(ValueError, match="Unsafe source archive path"):
        compute_code._extract_source(stream, tmp_path / "source", strip_root=True)
    assert not (tmp_path / "outside").exists()


def test_source_archive_rejects_links(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("repo/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member)
    stream.seek(0)
    with pytest.raises(ValueError, match="regular files only"):
        compute_code._extract_source(stream, tmp_path / "source", strip_root=True)


def test_default_code_cache_follows_the_weights_root(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENDDE_HARNESS_CODE_CACHE", raising=False)
    monkeypatch.delenv("OPENDDE_HARNESS_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert compute_code.default_code_cache() == (tmp_path / ".cache/opendde-harness/runtime-code").resolve()
    monkeypatch.setenv("OPENDDE_HARNESS_WEIGHTS_DIR", str(tmp_path / "weights"))
    assert compute_code.default_code_cache() == (tmp_path / "weights/runtime-code").resolve()
    monkeypatch.setenv("OPENDDE_HARNESS_CODE_CACHE", str(tmp_path / "shared/runtime-code"))
    assert compute_code.default_code_cache() == tmp_path / "shared/runtime-code"


def test_cached_source_archives_are_reused_across_snapshots(tmp_path, installed_package, fake_download, capsys):
    cache = tmp_path / "cache"
    first = compute_code.prepare_runtime_code(cache)
    assert fake_download == ["opendde", "ligandmpnn", "plip"]
    revision = compute_code.source_revisions()["OPENDDE_REV"]
    assert (cache / f"sources/opendde-{revision}.tar.gz").stat().st_size > 0
    assert sorted(path.name for path in (cache / "sources").iterdir()) == sorted(
        f"{name}-{compute_code.source_revisions()[name.upper() + '_REV']}.tar.gz" for name in SOURCE_FILES
    )
    shutil.rmtree(first)
    (installed_package / "__init__.py").write_text("release = 2\n")
    capsys.readouterr()
    second = compute_code.prepare_runtime_code(cache)
    assert second != first and fake_download == ["opendde", "ligandmpnn", "plip"]
    out = capsys.readouterr().out
    assert f"Reusing cached source: opendde ({revision[:12]})" in out
    assert "codeload.github.com" not in out
    assert (second / "external/opendde/runner/inference.py").read_text() == "print('fold')\n"
    compute_code.verify_runtime_code(second)


def test_previous_snapshot_sources_are_linked_without_download(tmp_path, installed_package, fake_download, capsys):
    cache = tmp_path / "cache"
    first = compute_code.prepare_runtime_code(cache)
    shutil.rmtree(cache / "sources")
    (installed_package / "__init__.py").write_text("release = 2\n")
    capsys.readouterr()
    second = compute_code.prepare_runtime_code(cache)
    assert fake_download == ["opendde", "ligandmpnn", "plip"]
    out = capsys.readouterr().out
    assert f"Reusing tool sources from {first.name}: opendde (" in out
    assert not (cache / "sources").exists()
    old = first / "external/ligandmpnn/model_utils.py"
    new = second / "external/ligandmpnn/model_utils.py"
    assert new.stat().st_ino == old.stat().st_ino
    compute_code.verify_runtime_code(first)
    compute_code.verify_runtime_code(second)


def test_corrupt_cached_source_is_downloaded_again_once(tmp_path, installed_package, fake_download, capsys):
    sources = tmp_path / "cache/sources"
    sources.mkdir(parents=True)
    revisions = compute_code.source_revisions()
    corrupt = sources / f"opendde-{revisions['OPENDDE_REV']}.tar.gz"
    corrupt.write_bytes(b"not an archive" * 8)
    empty = sources / f"plip-{revisions['PLIP_REV']}.tar.gz"
    empty.write_bytes(b"")
    root = compute_code.prepare_runtime_code(tmp_path / "cache")
    assert fake_download == ["opendde", "ligandmpnn", "plip"]
    out = capsys.readouterr().out
    assert "Reusing cached source: opendde (" in out
    assert "Reusing cached source: plip (" not in out
    assert tarfile.is_tarfile(corrupt) and tarfile.is_tarfile(empty)
    assert not list(sources.glob(".*"))
    assert (root / "external/opendde/runner/inference.py").is_file()
    compute_code.verify_runtime_code(root)


def test_stale_snapshots_are_pruned_keeping_referenced_and_newest(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    snapshots = {}
    for index, suffix in enumerate("abcdef"):
        snapshot = cache / f"1.0.0-{suffix}"
        (snapshot / "external").mkdir(parents=True)
        (snapshot / compute_code.MANIFEST).write_text("{}")
        os.utime(snapshot, (1_700_000_000 + index, 1_700_000_000 + index))
        snapshots[suffix] = snapshot
    (cache / "sources").mkdir()
    (cache / "sources/opendde-abc.tar.gz").write_bytes(b"x")
    (cache / ".prepare-live/code").mkdir(parents=True)
    commands = []

    def check_output(args, **_kwargs):
        commands.append(args)
        if args[1] == "ps":
            return "c1\nc2\n"
        assert args[1:3] == ["inspect", "--format"] and args[-2:] == ["c1", "c2"]
        return f"{snapshots['a']}\n/shared/harness-weights\n{snapshots['b'] / 'external'}\n\n"

    monkeypatch.setattr(compute_code.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(compute_code.subprocess, "check_output", check_output)
    removed = compute_code.prune_runtime_code(cache, snapshots["f"])
    assert removed == [snapshots["c"]]
    assert commands[0][1:] == ["ps", "-aq", "--filter", "label=org.opendde-harness.code-id"]
    assert sorted(path.name for path in cache.iterdir()) == [".prepare-live", "1.0.0-a", "1.0.0-b", "1.0.0-d", "1.0.0-e", "1.0.0-f", "sources"]
    assert capsys.readouterr().out == f"Removed stale runtime code: {snapshots['c']}\n"
    monkeypatch.setattr(compute_code.shutil, "which", lambda _name: None)
    assert compute_code.prune_runtime_code(cache, snapshots["f"]) == []
    assert (cache / "1.0.0-a").is_dir() and (cache / "1.0.0-d").is_dir()


def test_prepared_code_prunes_stale_snapshots(tmp_path, installed_code, monkeypatch, capsys):
    cache = tmp_path / "cache"
    stale = []
    for index in range(3):
        snapshot = cache / f"0.9.0-{index}"
        (snapshot / "external").mkdir(parents=True)
        (snapshot / compute_code.MANIFEST).write_text("{}")
        os.utime(snapshot, (1_700_000_000 + index, 1_700_000_000 + index))
        stale.append(snapshot)
    monkeypatch.setattr(compute_code, "_referenced_snapshots", lambda: set())
    root = compute_code.prepare_runtime_code(cache)
    assert sorted(path.name for path in cache.iterdir()) == ["0.9.0-1", "0.9.0-2", root.name]
    assert f"Removed stale runtime code: {stale[0]}\n" in capsys.readouterr().out
    assert compute_code.prepare_runtime_code(cache) == root
    assert (cache / "0.9.0-1").is_dir()
