import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from opendde_harness.cli import compute_assets
from opendde_harness.cli.compute_environment import (
    ENVIRONMENT_LABEL,
    ENVIRONMENT_SHA_LABEL,
    check_image_environment,
    environment_digest,
    load_environment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sources_only_prepares_pinned_checkouts_without_models(tmp_path, monkeypatch):
    (tmp_path / "opendde_harness").mkdir()
    (tmp_path / "external").mkdir()
    (tmp_path / "external/__init__.py").write_text("")
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/versions.env").write_text((ROOT / "docker/versions.env").read_text())
    (tmp_path / "docker/environment.json").write_text((ROOT / "docker/environment.json").read_text())
    calls = []
    monkeypatch.setattr(compute_assets, "clone_repository", lambda *args: calls.append(args))
    monkeypatch.setattr(compute_assets, "prepare", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model preparation")))
    compute_assets.prepare_sources(tmp_path)
    assert {path.name for _, _, path in calls} == {"opendde", "ligandmpnn", "plip", "foldmason"}
    revisions = compute_assets.source_revisions(tmp_path / "docker/versions.env")
    for _, revision, path in calls:
        expected = load_environment()["foldmason_revision"] if path.name == "foldmason" else revisions[path.name.upper() + "_REV"]
        assert revision == expected
        assert path.parent == tmp_path / "external"
    assert not (tmp_path / "compute-assets.json").exists()


def test_sources_only_rejects_non_checkout(tmp_path):
    with pytest.raises(ValueError, match="Harness source checkout"):
        compute_assets.prepare_sources(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_git_failures_carry_stderr(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert kwargs.get("timeout")
        raise subprocess.CalledProcessError(128, command, stderr="fatal: repository 'https://example.invalid/x.git' not found\n")

    monkeypatch.setattr(compute_assets.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(compute_assets.subprocess, "run", run)
    with pytest.raises(ValueError, match="git clone failed .*repository 'https://example.invalid/x.git' not found"):
        compute_assets.clone_repository("https://example.invalid/x.git", "a" * 40, tmp_path / "external/x")
    assert not (tmp_path / "external/x").exists()


def test_weights_layout_lists_only_the_tools_the_mode_needs(tmp_path):
    api = compute_assets.weights_layout(tmp_path / "harness", tmp_path / "opendde", with_opendde=False)
    local = compute_assets.weights_layout(
        tmp_path / "harness", tmp_path / "opendde", with_opendde=True, checkpoint="opendde_abag.pt", translate=lambda en, zh: zh
    )
    assert api.splitlines() == [
        f"Harness tool weights (OPENDDE_HARNESS_WEIGHTS_DIR): {tmp_path / 'harness'}",
        "  SolubleMPNN: soluble_mpnn/solublempnn_v_48_020.pt",
        "  ESM2: huggingface/models--facebook--esm2_t33_650M_UR50D",
    ]
    assert local.splitlines()[:3] == [
        f"OpenDDE 数据（OPENDDE_ROOT_DIR）： {tmp_path / 'opendde'}",
        "  checkpoint/opendde_abag.pt, common/",
        f"Harness 工具权重（OPENDDE_HARNESS_WEIGHTS_DIR）： {tmp_path / 'harness'}",
    ]


def test_opendde_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDDE_HARNESS_HOME", str(tmp_path / "harness-home"))
    monkeypatch.delenv("OPENDDE_ROOT_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert compute_assets.opendde_root() == (tmp_path / ".cache/opendde").resolve()
    monkeypatch.setenv("OPENDDE_ROOT_DIR", str(tmp_path / "env"))
    assert compute_assets.opendde_root() == (tmp_path / "env").resolve()
    compute_assets.write_json(compute_assets.asset_state_path(), {"schema_version": 2, "paths": {"opendde_data": str(tmp_path / "prepared")}})
    assert compute_assets.opendde_root() == (tmp_path / "prepared").resolve()
    assert compute_assets.opendde_root({"opendde_data": str(tmp_path / "saved")}) == (tmp_path / "saved").resolve()


def test_local_preparation_uses_both_roots(tmp_path, monkeypatch):
    weights, data = tmp_path / "harness", tmp_path / "opendde"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    def download(destination_root, asset):
        path = destination_root / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return path

    monkeypatch.setattr(compute_assets, "download_asset", download)
    state = compute_assets.prepare(weights, "opendde.pt", tmp_path / "state.json", opendde_root=data, with_opendde=True)
    assert state["root"] == str(weights) and state["opendde_root"] == str(data)
    assert state["paths"]["opendde_checkpoint"] == str(data / "checkpoint/opendde.pt")
    assert state["paths"]["weights_dir"] == str(weights)
    assert (data / "checkpoint/opendde.pt").is_file() and (data / "common/components.cif").is_file()
    assert (weights / "soluble_mpnn/solublempnn_v_48_020.pt").is_file()
    assert not (weights / "checkpoint").exists() and not (data / "soluble_mpnn").exists()
    assert len((weights / "SHA256SUMS").read_text().splitlines()) == 6
    assert len((data / "SHA256SUMS").read_text().splitlines()) == 5
    # Prepared with the general checkpoint, so inspect the same one rather than the default.
    report = compute_assets.inspect_assets(weights, opendde_root=data, with_opendde=True, checkpoint="opendde.pt")
    assert report["opendde_root"] == str(data)
    assert all(item["error"] != "missing or empty" for item in report["files"])
    assert {Path(item["path"]).parents[1] for item in report["files"] if "/checkpoint/" in item["path"] or "/common/" in item["path"]} == {data}


def test_shared_models_are_copied_from_the_opendde_data_root(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    data = b"mpnn"
    mpnn = compute_assets.Asset(compute_assets.SOLUBLE_MPNN_WEIGHTS, "https://example.invalid", hashlib.sha256(data).hexdigest(), len(data))
    monkeypatch.setattr(compute_assets, "shared_asset_plan", lambda: [mpnn])
    previous = tmp_path / ".cache/opendde" / mpnn.relative_path
    previous.parent.mkdir(parents=True)
    previous.write_bytes(data)

    def download(destination_root, asset):
        path = destination_root / asset.relative_path
        assert path.is_file(), "the existing copy must be reused before any download"
        return path

    monkeypatch.setattr(compute_assets, "download_asset", download)
    root = tmp_path / ".cache/opendde-harness"
    compute_assets._prepare_shared_models(root)
    assert (root / mpnn.relative_path).read_bytes() == data and previous.read_bytes() == data


def test_weights_root_precedence(tmp_path, monkeypatch):
    home = tmp_path / "harness-home"
    monkeypatch.setenv("OPENDDE_HARNESS_HOME", str(home))
    monkeypatch.delenv("OPENDDE_HARNESS_WEIGHTS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert compute_assets.weights_root() == (tmp_path / ".cache/opendde-harness").resolve()
    (tmp_path / ".cache/opendde/soluble_mpnn").mkdir(parents=True)
    assert compute_assets.weights_root() == (tmp_path / ".cache/opendde-harness").resolve()
    monkeypatch.setenv("OPENDDE_HARNESS_WEIGHTS_DIR", str(tmp_path / "env"))
    assert compute_assets.weights_root({}) == (tmp_path / "env").resolve()
    compute_assets.write_json(compute_assets.asset_state_path(), {"schema_version": 2, "paths": {"weights_dir": str(tmp_path / "prepared")}})
    assert compute_assets.weights_root({}) == (tmp_path / "prepared").resolve()
    assert compute_assets.weights_root({"weights_dir": str(tmp_path / "saved")}) == (tmp_path / "saved").resolve()
    # A pre-split release recorded the OpenDDE data root here; taking it would
    # download the tool weights a second time inside OpenDDE's own tree.
    inside_opendde = {"weights_dir": str(tmp_path / "data"), "opendde_data": str(tmp_path / "data")}
    assert compute_assets.weights_root(inside_opendde) == (tmp_path / "env").resolve()


def test_build_dry_run_needs_neither_source_assets_nor_weights(tmp_path):
    env = {**os.environ, "TMPDIR": str(tmp_path / "not-created"), "ESM_CACHE": "/missing", "SOLUBLE_MPNN_WEIGHTS": "/missing"}
    result = subprocess.run(
        ["bash", str(ROOT / "docker/build.sh"), "--dry-run", "--push", "example/runtime:first", "example/runtime:second"],
        env=env, text=True, capture_output=True, check=True,
    )
    command = shlex.split(result.stdout)
    assert "--push" in command and "--load" not in command
    assert "--build-context" not in command
    assert [command[i + 1] for i, item in enumerate(command[:-1]) if item == "--tag"] == [
        "example/runtime:first", "example/runtime:second"
    ]
    assert list(tmp_path.iterdir()) == []


def test_model_preparation_preserves_populated_destination(tmp_path):
    existing = tmp_path / "model.pt"
    existing.write_bytes(b"keep")
    result = subprocess.run(["bash", str(ROOT / "docker/prepare-models.sh"), str(tmp_path)], capture_output=True)
    assert result.returncode != 0
    assert existing.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [existing]


def _serve(monkeypatch, handler):
    import httpx

    from opendde_harness.cli import _download

    monkeypatch.setattr(_download, "new_client", lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))


def test_soluble_mpnn_tries_graylab_then_project_hf_repo_with_mirror_then_ipd(tmp_path, monkeypatch):
    import hashlib

    import httpx

    data = b"mpnn"
    graylab, project, ipd = compute_assets.SOLUBLE_MPNN_SOURCES
    assert graylab.startswith("https://ipd.graylab.jhu.edu/") and project.startswith(compute_assets.MODEL_ROOT + "/")
    asset = next(item for item in compute_assets.shared_asset_plan() if item.relative_path == compute_assets.SOLUBLE_MPNN_WEIGHTS)
    assert asset.sources == compute_assets.SOLUBLE_MPNN_SOURCES
    asset = compute_assets.Asset(asset.relative_path, asset.url, hashlib.sha256(data).hexdigest(), len(data), asset.mirrors)
    attempts = []
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    def handler(request):
        url = str(request.url)
        attempts.append(url)
        if url == ipd:
            return httpx.Response(200, content=data)
        if url.startswith("https://huggingface.co/"):
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(404)

    _serve(monkeypatch, handler)
    assert compute_assets.download_asset(tmp_path, asset).read_bytes() == data
    assert attempts == [graylab, project, "https://hf-mirror.com" + project[len("https://huggingface.co"):], ipd]
    _serve(monkeypatch, lambda request: httpx.Response(404))
    with pytest.raises(ValueError, match="Every download source failed"):
        compute_assets.download_asset(tmp_path, compute_assets.Asset("soluble_mpnn/other.pt", asset.url, "b" * 64, 1, asset.mirrors))


def test_environment_contract_rejects_wrong_image():
    spec = load_environment()
    image = {"Os": "linux", "Architecture": "amd64", "Config": {"Labels": {
        ENVIRONMENT_LABEL: spec["id"], ENVIRONMENT_SHA_LABEL: environment_digest(spec),
    }}}
    check_image_environment(image, spec)
    image["Config"]["Labels"][ENVIRONMENT_SHA_LABEL] = "wrong"
    with pytest.raises(ValueError, match="Incompatible compute environment"):
        check_image_environment(image, spec)


def test_software_releases_do_not_publish_environment_images():
    workflow = (ROOT / ".github/workflows/protein-design-compute.yml").read_text()
    assert '"env-*"' in workflow
    assert '"v*"' not in workflow
    assert "refs/tags/v" not in workflow


def test_shared_model_plan_is_650m_and_all_assets_are_pinned():
    assets = compute_assets.shared_asset_plan()
    assert len(assets) == 6
    assert len({asset.relative_path for asset in assets}) == 6
    assert all(len(asset.sha256) == 64 for asset in assets)
    assert all("650M" in asset.url for asset in assets if "huggingface" in asset.relative_path)


def test_shared_model_revision_is_not_overwritten(tmp_path, monkeypatch):
    ref = tmp_path / "huggingface/models--facebook--esm2_t33_650M_UR50D/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("b" * 40)
    monkeypatch.setattr(compute_assets, "download_asset", lambda *_: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="another ESM revision"):
        compute_assets._prepare_shared_models(tmp_path)
    assert ref.read_text() == "b" * 40


def test_pypi_name_and_console_script():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "opendde-harness"
    assert project["scripts"]["ddeharness"] == "opendde_harness.cli.commands:run"
    assert set(project["scripts"]) == {"ddeharness"}


def test_shared_only_preparation_uses_requested_root(tmp_path, monkeypatch):
    root = tmp_path / "weights"
    unrelated = tmp_path / "other-cache"
    monkeypatch.setenv("OPENDDE_ROOT_DIR", str(unrelated))

    def download(destination_root, asset):
        path = destination_root / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"verified model fixture")
        return path

    monkeypatch.setattr(compute_assets, "download_asset", download)
    root.mkdir()
    (root / "SHA256SUMS").write_text("stale  removed/file\n")
    state = compute_assets.prepare(root, "opendde.pt", tmp_path / "state.json", with_opendde=False)
    assert state["paths"] == {"weights_dir": str(root)}
    assert state["checkpoint"] is None
    assert not unrelated.exists()
    assert not (root / "checkpoint").exists()
    checksums = (root / "SHA256SUMS").read_text()
    assert len(checksums.splitlines()) == 6 and "stale" not in checksums
    assert json.loads((tmp_path / "state.json").read_text())["paths"]["weights_dir"] == str(root)
    assert not (root / "compute-assets.json").exists()


def test_asset_inspection_is_mode_aware_and_detects_corruption(tmp_path, monkeypatch):
    import hashlib

    data = b"model"
    shared = compute_assets.Asset("soluble_mpnn/test.pt", "https://example.invalid", hashlib.sha256(data).hexdigest(), len(data))
    local = compute_assets.Asset("checkpoint/opendde.pt", "https://example.invalid", "a" * 64, len(data))
    monkeypatch.setattr(compute_assets, "shared_asset_plan", lambda: [shared])
    monkeypatch.setattr(compute_assets, "asset_plan", lambda _: [local])
    path = tmp_path / shared.relative_path
    path.parent.mkdir()
    path.write_bytes(data)
    ref = tmp_path / "huggingface/models--facebook--esm2_t33_650M_UR50D/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text(compute_assets.source_revisions()["ESM_REV"])
    api = compute_assets.inspect_assets(tmp_path, verify_hashes=True)
    assert api["ready"] and api["opendde"] == "not_required"
    assert not (tmp_path / "checkpoint").exists()
    local_report = compute_assets.inspect_assets(tmp_path, with_opendde=True)
    assert not local_report["ready"]
    path.write_bytes(b"wrong")
    assert compute_assets.inspect_assets(tmp_path)["ready"]
    verified = compute_assets.inspect_assets(tmp_path, verify_hashes=True)
    assert not verified["ready"]
    assert verified["files"][0]["error"] == "SHA256 mismatch"


def test_the_default_checkpoint_is_the_antibody_one():
    """This harness designs antibodies; the general checkpoint stays available by name."""
    from opendde_harness.cli import compute_assets
    from opendde_harness.plugin.protein_design.core.asset_paths import DEFAULT_CHECKPOINT

    assert DEFAULT_CHECKPOINT == "opendde_abag.pt"
    assert set(compute_assets.CHECKPOINTS) == {"opendde.pt", "opendde_abag.pt"}
    assert compute_assets.asset_plan(DEFAULT_CHECKPOINT)[0].relative_path == "checkpoint/opendde_abag.pt"
