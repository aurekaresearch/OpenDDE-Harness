"""Sampling seeds must not create duplicate resident SolubleMPNN models."""

from opendde_harness.plugin.protein_design.servers.harness import PythonProteinDesignHarness


def test_different_sampling_seeds_reuse_one_loaded_model(monkeypatch, tmp_path):
    from opendde_harness.plugin.protein_design.servers.backends import soluble_mpnn

    loaded = []
    sampled = []

    class FakeClient:
        weights_sha256 = "test-weights"

        def __init__(self, **kwargs):
            loaded.append(kwargs)

        def design(self, **kwargs):
            sampled.append(kwargs["seed"])
            return [{"sequence": "GC", "mpnn_score": 1.0, "mpnn_seqid": 0.5}]

    monkeypatch.setattr(soluble_mpnn, "SolubleMPNNClient", FakeClient)
    harness = PythonProteinDesignHarness(output_path=str(tmp_path))
    request = {
        "structure_path": str(tmp_path / "parent.cif"),
        "parent_chains": {"B": "AC"},
        "mutable_positions": ["B:0"],
        "num_sequences": 1,
    }
    try:
        for seed in (41, 42):
            result = harness._generate_soluble_mpnn({**request, "parameters": {"seed": seed, "device": "cpu"}})
            assert result["candidates"][0]["chains"] == {"B": "GC"}
        assert len(loaded) == 1
        assert sampled == [41, 42]
    finally:
        harness.close()
