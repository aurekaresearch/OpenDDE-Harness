from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.core.asset_paths import checkpoint_status, resolve_checkpoint


def test_checkpoint_resolution_by_mode_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", str(tmp_path / "opendde_abag.pt"))
    assert resolve_checkpoint("antibody") == tmp_path / "opendde_abag.pt"
    assert resolve_checkpoint("minibinder") == tmp_path / "opendde.pt"
    assert resolve_checkpoint("minibinder", "/custom/model.pt") == Path("/custom/model.pt")
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", "/custom/model.pt")
    assert resolve_checkpoint("minibinder") == Path("/custom/model.pt")
    monkeypatch.delenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH")
    monkeypatch.setenv("STRUCTPRED_OPENDDE_ROOT_DIR", str(tmp_path))
    assert resolve_checkpoint("minibinder") == tmp_path / "checkpoint/opendde.pt"


def test_checkpoint_missing_empty_unreadable_and_wrong_mode(tmp_path, monkeypatch):
    path = tmp_path / "custom.pt"
    assert not checkpoint_status(path, "minibinder")["ready"]
    path.touch()
    assert checkpoint_status(path, "minibinder")["error"] == "empty file"
    path.write_bytes(b"test checkpoint")
    assert checkpoint_status(path, "minibinder")["ready"]
    assert not checkpoint_status(tmp_path / "opendde_abag.pt", "minibinder")["ready"]
    monkeypatch.setattr("os.access", lambda *args: False)
    assert not checkpoint_status(path, "minibinder")["ready"]


def test_health_checks_selected_mode_not_optional_checkpoint(tmp_path, monkeypatch):
    from opendde_harness.cli import compute_assets
    from opendde_harness.plugin.protein_design.core.asset_paths import OPENDDE_COMMON_ASSETS
    from opendde_harness.plugin.protein_design.servers import harness as module

    monkeypatch.setenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", str(tmp_path / "opendde_abag.pt"))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CODE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(module, "_probe_gpus", lambda: [])
    monkeypatch.setattr(compute_assets, "inspect_assets", lambda *a, **k: {"root": str(tmp_path), "files": []})
    (tmp_path / "opendde_abag.pt").write_bytes(b"test")
    (tmp_path / "common").mkdir()
    for name in OPENDDE_COMMON_ASSETS:
        (tmp_path / "common" / name).touch()
    engine = module.PythonProteinDesignHarness(output_path=str(tmp_path))
    monkeypatch.setattr(engine, "_cached_backend_import_probe", lambda *a: (True, None))
    antibody = engine._health_snapshot("opendde", "local", None, None, design_type="antibody")
    mini = engine._health_snapshot("opendde", "local", None, None, design_type="minibinder")
    assert antibody["status"] == "ok"
    assert mini["status"] == "degraded"
    assert not antibody["workers"]["backend"]["checkpoints"]["minibinder"]["ready"]
    (tmp_path / "opendde.pt").write_bytes(b"test")
    assert engine._health_snapshot("opendde", "local", None, None, design_type="minibinder")["status"] == "ok"


def test_fold_rejects_missing_checkpoint_before_loading_model(tmp_path, monkeypatch):
    from opendde_harness.plugin.protein_design.servers.harness import PythonProteinDesignHarness

    monkeypatch.setenv("STRUCTPRED_OPENDDE_CHECKPOINT_PATH", str(tmp_path / "opendde_abag.pt"))
    engine = PythonProteinDesignHarness(output_path=str(tmp_path))
    with pytest.raises(ValueError, match="minibinder checkpoint"):
        engine._fold({"options": {"execution_mode": "local", "design_type": "minibinder"}})


@pytest.mark.asyncio
async def test_worker_selection_sends_mode_and_rejects_missing_checkpoint():
    import httpx

    from opendde_harness.plugin.protein_design.core.contracts import WorkflowConfig
    from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient
    from opendde_harness.plugin.protein_design.servers.compute_pool import ComputePool, ComputeWorker

    requests = []

    def handle(request):
        requests.append(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "status": "degraded",
                "workers": {
                    "backend": {
                        "checkpoint_ready": False,
                        "checkpoint_path": "/compute/custom.pt",
                        "checkpoint_error": "missing or unreadable",
                    }
                },
            },
        )

    client = ProteinDesignComputeClient("http://compute", transport=httpx.MockTransport(handle))
    pool = ComputePool([], default_url="http://compute", client_factory=lambda *a, **k: client)
    config = WorkflowConfig(
        target="test",
        fold_options={"design_type": "minibinder", "execution_mode": "local", "checkpoint_path": "/compute/custom.pt"},
    )
    try:
        with pytest.raises(RuntimeError, match="Checkpoint unavailable.*custom.pt"):
            await pool._probe(ComputeWorker(worker_id="test", url="http://compute"), config)
        assert requests[0]["design_type"] == "minibinder"
        assert requests[0]["checkpoint_path"] == "/compute/custom.pt"
    finally:
        await client.close()
