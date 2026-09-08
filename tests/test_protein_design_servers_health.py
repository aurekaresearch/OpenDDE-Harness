from pathlib import Path

import pytest

from opendde_harness.cli import compute_assets
from opendde_harness.plugin.protein_design.core.asset_paths import OPENDDE_COMMON_ASSETS
from opendde_harness.plugin.protein_design.servers import harness as harness_module

ESM_ASSET = "huggingface/models--facebook--esm2_t33_650M_UR50D/snapshots/abc/model.safetensors"
MPNN_ASSET = "soluble_mpnn/solublempnn_v_48_020.pt"


def asset_report(root: Path, files: list[tuple[str, bool]]) -> dict:
    return {
        "ready": all(ok for _, ok in files),
        "root": str(root),
        "opendde": "not_required",
        "hashes_verified": False,
        "files": [{"path": str(root / name), "ok": ok, "error": None} for name, ok in files],
    }


@pytest.fixture
def local_backend(tmp_path, monkeypatch):
    code_dir = tmp_path / "code"
    root_dir = tmp_path / "weights"
    code_dir.mkdir()
    (root_dir / "common").mkdir(parents=True)
    for name in OPENDDE_COMMON_ASSETS:
        (root_dir / "common" / name).write_text("asset")
    checkpoint = root_dir / "checkpoint/opendde.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("checkpoint")
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CODE_DIR", str(code_dir))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_ROOT_DIR", str(root_dir))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setattr(
        harness_module.PythonProteinDesignHarness,
        "_cached_backend_import_probe",
        lambda *args: (True, None),
    )
    monkeypatch.setattr(harness_module, "resolve_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(harness_module.shutil, "which", lambda name: None)
    return root_dir


async def test_health_is_ok_when_the_requested_backend_is_ready(tmp_path, monkeypatch, local_backend):
    monkeypatch.setattr(
        compute_assets,
        "inspect_assets",
        lambda *args, **kwargs: asset_report(local_backend, [(ESM_ASSET, False), (MPNN_ASSET, False)]),
    )
    service = harness_module.PythonProteinDesignHarness(output_path=str(tmp_path / "output"))

    payload = await service.health(execution_mode="local")

    assert payload["status"] == "ok"
    assert payload["workers"]["backend_ready"] is True
    assert payload["workers"]["tools"]["foldmason"]["ready"] is False
    assert payload["workers"]["tools"]["esm2"]["ready"] is False


async def test_health_is_degraded_when_the_backend_is_incomplete(tmp_path, monkeypatch, local_backend):
    monkeypatch.delenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH")
    monkeypatch.setattr(
        compute_assets,
        "inspect_assets",
        lambda *args, **kwargs: asset_report(local_backend, [(ESM_ASSET, True), (MPNN_ASSET, True)]),
    )
    service = harness_module.PythonProteinDesignHarness(output_path=str(tmp_path / "output"))

    payload = await service.health(execution_mode="local")

    assert payload["status"] == "degraded"
    assert payload["workers"]["backend"]["checkpoint_ready"] is False
    assert payload["workers"]["tools"]["esm2"]["ready"] is True


async def test_health_rejects_an_unsupported_backend(tmp_path):
    service = harness_module.PythonProteinDesignHarness(output_path=str(tmp_path / "output"))

    with pytest.raises(ValueError):
        await service.health(backend="alphafold")


def test_asset_groups_are_not_ready_when_the_plan_renames_them():
    report = asset_report(Path("/weights"), [("renamed/model.safetensors", True), (MPNN_ASSET, True)])

    groups = harness_module._assets_by_tool(report)

    assert groups["esm2"] == []
    assert harness_module._asset_group_ready(groups["esm2"]) is False
    assert harness_module._asset_group_ready(groups["soluble_mpnn"]) is True


def test_asset_group_is_not_ready_when_one_file_fails():
    report = asset_report(Path("/weights"), [(ESM_ASSET, True), (ESM_ASSET + ".bin", False)])

    groups = harness_module._assets_by_tool(report)

    assert len(groups["esm2"]) == 2
    assert harness_module._asset_group_ready(groups["esm2"]) is False


@pytest.mark.parametrize(
    "spec, expected",
    [("all", None), ("none", frozenset()), ("2,3", frozenset({2, 3})), ('"device=1"', frozenset({1}))],
)
def test_fold_gpu_specs_map_to_the_devices_they_occupy(spec, expected) -> None:
    assert harness_module._fold_devices(spec) == expected


@pytest.mark.parametrize(
    "left, right, shares",
    [
        (frozenset({0}), frozenset({1}), False),
        (frozenset({0, 1}), frozenset({1}), True),
        (None, frozenset({3}), True),
        (frozenset(), None, False),
    ],
)
def test_only_a_resident_model_on_the_same_gpu_must_be_released(left, right, shares) -> None:
    assert harness_module._shares_gpu(left, right) is shares


def test_service_health_payload_validates_against_the_client_contract():
    from fastapi.testclient import TestClient

    from opendde_harness.plugin.protein_design.core.contracts import HealthResponse
    from opendde_harness.plugin.protein_design.servers.api import create_app

    class _Harness:
        async def health(self, backend=None, execution_mode=None, image=None, api_url=None, *, probe_external=False):
            return {"status": "ok", "models": {"fold": True}, "workers": {"device": "cpu", "tools": {}}, "gpu": []}

        async def invoke(self, operation, payload):
            return {}

        def close(self):
            pass

    app = create_app(_Harness())
    payload = TestClient(app).get("/health").json()
    health = HealthResponse.model_validate(payload)
    assert health.status == "ok"
    assert HealthResponse.model_validate({**payload, "future_key": 1}).status == "ok"



def test_default_protrek_endpoint_stays_plain_http(monkeypatch):
    """The upstream service has no TLS; an https default hangs until timeout."""
    from opendde_harness.plugin.protein_design.core import external

    monkeypatch.delenv("PROTREK_ENDPOINT", raising=False)
    assert external.DEFAULT_PROTREK_URL == "http://search-protrek.com/"
    assert external.protrek_endpoint() == "http://search-protrek.com/"
    monkeypatch.setenv("PROTREK_ENDPOINT", "")
    assert external.protrek_endpoint() is None
