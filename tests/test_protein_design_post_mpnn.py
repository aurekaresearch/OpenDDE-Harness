"""Terminal workload controls retain sequence constraints and historical defaults."""

import asyncio

import pytest

from opendde_harness.plugin.protein_design.core.contracts import Candidate, WorkflowConfig
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.plugin.protein_design.core.post_mpnn import prepare_post_mpnn


def config(**kwargs):
    return WorkflowConfig(
        target="test",
        binder_chains={"D": "ACDE"},
        mutable_positions={"D": [0, 1]},
        fixed_residues={"D": [0]},
        **kwargs,
    )


def sample(sequence, score):
    return {"chains": {"D": sequence}, "metadata": {"soluble_mpnn_scores": {"D": score}}}


class Compute:
    def __init__(self, samples):
        self.samples = samples
        self.requests = []

    async def generate_soluble_mpnn(self, request):
        self.requests.append(request)
        return {"candidates": self.samples}


def generate(settings, samples):
    compute = Compute(samples)
    parent = Candidate(candidate_id="parent", sequence="ACDE", structure_path="/tmp/parent.cif")
    children = asyncio.run(prepare_post_mpnn(compute, [parent], settings, "task", asyncio.Event()))
    return children, compute.requests


def test_default_terminal_workload_is_unchanged():
    settings = config()
    assert settings.post_refold_max_parents is None
    assert settings.post_refold_samples_per_parent == 40
    assert settings.post_refold_survivors_per_parent == 4
    samples = [sample("A" + residue + "DE", float(index)) for index, residue in enumerate("CDEFG")]
    children, requests = generate(settings, samples * 8)
    assert requests[0].num_sequences == 40
    assert len(children) == 4
    assert all(child.metadata["post_mpnn_selected"] for child in children)


def test_short_terminal_workload_preserves_fixed_residues_and_ranking():
    settings = config(post_refold_samples_per_parent=4, post_refold_survivors_per_parent=1)
    children, requests = generate(
        settings,
        [sample("CCDE", -100.0), sample("AFDE", 2.0), sample("AGDE", 1.0), sample("AXDE", -50.0)],
    )
    assert requests[0].num_sequences == 4
    assert requests[0].mutable_positions == ["D:1"]
    assert len(children) == 1
    assert children[0].sequence == "AGDE"
    assert children[0].metadata["mpnn_rank"] == 1
    assert children[0].metadata["mpnn_sample_index"] == 3


def test_short_terminal_workload_does_not_accept_missing_samples():
    children, _ = generate(
        config(post_refold_samples_per_parent=4, post_refold_survivors_per_parent=1),
        [sample("AFDE", 1.0)],
    )
    assert children[0].metadata["success"] is False
    assert children[0].metadata["post_mpnn_selected"] is False
    assert "Expected 4 SolubleMPNN samples, received 1" in children[0].metadata["post_refold_error"]


def test_terminal_shortfall_reports_configured_survivor_count():
    children, _ = generate(
        config(post_refold_samples_per_parent=2, post_refold_survivors_per_parent=2),
        [sample("AFDE", 1.0)] * 2,
    )
    assert len(children) == 1
    assert "only 1 of 2" in children[0].metadata["post_mpnn_shortfall"]


@pytest.mark.parametrize("minimize", [True, False])
def test_explicit_parent_cap_keeps_objective_order_and_gates(minimize):
    history = [
        {"candidate_id": "good", "objective": 1.0, "gate_passed": True},
        {"candidate_id": "better", "objective": -1.0, "gate_passed": True},
        {"candidate_id": "rejected", "objective": -100.0, "gate_passed": False},
    ]
    settings = config(post_refold_max_parents=1, minimize=minimize)
    pool = DesignOrchestrator._terminal_refold_pool(history, settings)
    assert [item["candidate_id"] for item in pool] == ["better" if minimize else "good"]


@pytest.mark.parametrize(
    "field", ["post_refold_max_parents", "post_refold_samples_per_parent", "post_refold_survivors_per_parent"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_terminal_workload_rejects_nonpositive_or_coerced_counts(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


def test_terminal_survivors_cannot_exceed_sample_count():
    with pytest.raises(ValueError, match="must not exceed"):
        config(post_refold_samples_per_parent=2, post_refold_survivors_per_parent=3)
