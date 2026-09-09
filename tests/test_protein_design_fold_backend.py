import pytest


def test_local_cpu_fold_command_uses_runtime_python_and_no_gpu(tmp_path, monkeypatch):
    import sys

    from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor

    config = FoldConfig(execution_mode="local", device="cpu", gpus="none", diffusion_samples=5)
    predictor = object.__new__(StructurePredictor)
    predictor.config = config
    monkeypatch.setenv("STRUCTPRED_OPENDDE_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CODE_DIR", str(tmp_path / "code"))
    monkeypatch.setattr(predictor, "_get_opendde_model_name", lambda: "opendde_v1")
    monkeypatch.setattr(predictor, "_get_opendde_checkpoint_args", lambda: ("/weights/checkpoint/opendde.pt", []))
    command = predictor._build_opendde_command("unused", tmp_path / "job.json", tmp_path / "run")
    assert sys.executable in command
    assert "--gpus" not in command
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--sample_diffusion.N_sample") + 1] == "5"
    assert "CUDA_VISIBLE_DEVICES=" in command


def test_two_leased_gpus_launch_fold_cp_context_parallel_inference(tmp_path, monkeypatch):
    from opendde_harness.cli import compute_environment
    from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor

    monkeypatch.setattr(compute_environment, "resolve_device", lambda *args, **kwargs: "cuda")
    config = FoldConfig(execution_mode="local", device="cuda", gpus="2,3")
    predictor = object.__new__(StructurePredictor)
    predictor.config = config
    monkeypatch.setenv("STRUCTPRED_OPENDDE_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CODE_DIR", str(tmp_path / "code"))
    monkeypatch.setattr(predictor, "_get_opendde_model_name", lambda: "opendde_v1")
    monkeypatch.setattr(predictor, "_get_opendde_checkpoint_args", lambda: ("/weights/checkpoint/opendde.pt", []))

    command = predictor._build_opendde_command("unused", tmp_path / "job.json", tmp_path / "run")

    assert "CUDA_VISIBLE_DEVICES=2,3" in command
    assert "OPENDDE_FOLDCP_MODE=distributed" in command
    assert "OPENDDE_FOLDCP_SIZE_CP=2" in command
    assert "OPENDDE_FOLDCP_DEVICES=2,3" in command
    assert "--nproc_per_node=2" in command
    assert command[command.index("--foldcp_mode") + 1] == "distributed"
    assert command[command.index("--foldcp_size_cp") + 1] == "2"
    assert command[command.index("--foldcp_size_dp") + 1] == "1"


def test_one_leased_gpu_keeps_single_process_inference(tmp_path, monkeypatch):
    from opendde_harness.cli import compute_environment
    from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig, StructurePredictor

    monkeypatch.setattr(compute_environment, "resolve_device", lambda *args, **kwargs: "cuda")
    config = FoldConfig(execution_mode="local", device="cuda", gpus="1")
    predictor = object.__new__(StructurePredictor)
    predictor.config = config
    monkeypatch.setenv("STRUCTPRED_OPENDDE_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTPRED_OPENDDE_CODE_DIR", str(tmp_path / "code"))
    monkeypatch.setattr(predictor, "_get_opendde_model_name", lambda: "opendde_v1")
    monkeypatch.setattr(predictor, "_get_opendde_checkpoint_args", lambda: ("/weights/checkpoint/opendde.pt", []))

    command = predictor._build_opendde_command("unused", tmp_path / "job.json", tmp_path / "run")

    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert not [item for item in command if item.startswith("--foldcp") or item.startswith("OPENDDE_FOLDCP")]


def test_api_folding_does_not_require_a_local_device(monkeypatch):
    from opendde_harness.plugin.protein_design.servers.backends.fold import FoldConfig

    monkeypatch.setenv("OPENDDE_HARNESS_COMPUTE_DEVICE", "cuda")
    config = FoldConfig(execution_mode="api", gpus="none", api_url="https://example.invalid")
    assert config.execution_mode == "api"


@pytest.mark.parametrize(
    "payload, valid", [({"openapi": "3.1.0", "paths": {"/jobs": {"post": {}}}}, True), ({"status": "ok"}, False)]
)
def test_api_probe_requires_the_opendde_job_schema(payload, valid):
    import httpx

    from opendde_harness.plugin.protein_design.servers.backends.opendde_api import OpenDDEAPIError, OpenDDEJobClient

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    with OpenDDEJobClient("https://example.invalid", transport=transport) as client:
        if valid:
            client.probe()
        else:
            with pytest.raises(OpenDDEAPIError, match="job API"):
                client.probe()
