"""Inverse folding must honor the workflow's unique-candidate budget."""

import pytest

from opendde_harness.plugin.protein_design.agents.proposals import (
    ProposalContext,
    ProposalExecutor,
    ProposalValidationError,
)
from opendde_harness.plugin.protein_design.core.contracts import DesignAgentOutput


class Compute:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.requests = []

    async def generate_soluble_mpnn(self, request):
        self.requests.append(request)
        return {"candidates": [{"id": "backend-id", "chains": {"B": sequence}} for sequence in next(self.batches)]}


def context(count):
    return ProposalContext("parent", {"B": "ACDE"}, {"B": [0, 1]}, "/parent.cif", count)


def output(*plans):
    return DesignAgentOutput(skill_id="antibody-inverse-folding", candidates=list(plans))


@pytest.mark.parametrize("model_count", [1, 3, 4, 100])
async def test_model_cannot_override_configured_sample_count(model_count):
    compute = Compute([[aa + "CDE" for aa in "GHIKLMNP"]])
    result = await ProposalExecutor(compute)._execute_inverse_folding(
        output({"metadata": {"soluble_mpnn_parameters": {"num_sequences": model_count}}}), context(8)
    )
    assert compute.requests[0].num_sequences == 8
    assert "num_sequences" not in compute.requests[0].parameters
    assert len(result) == len({p.candidate_id for p in result}) == 8


async def test_all_anchor_plans_receive_a_share_of_the_cycle_budget():
    compute = Compute([["GCDE", "GADE"], ["HCDE", "HADE"]])
    result = await ProposalExecutor(compute)._execute_inverse_folding(
        output(
            {"id": "g", "mutations": [["B", 1, "G"]], "strategy": "first hypothesis"},
            {"id": "h", "mutations": [["B", 1, "H"]], "strategy": "second hypothesis"},
        ),
        context(4),
    )
    assert [r.num_sequences for r in compute.requests] == [2, 2]
    assert [r.anchor_mutations[0]["to_aa"] for r in compute.requests] == ["G", "H"]
    assert [p.metadata["anchor_plan_id"] for p in result] == ["g", "g", "h", "h"]
    assert [p.strategy for p in result] == ["first hypothesis"] * 2 + ["second hypothesis"] * 2


async def test_duplicates_and_invalid_sequences_are_replaced_with_valid_unique_samples():
    compute = Compute([["GCDE", "GCDE", "GCDA"], ["HCDE", "ICDE"]])
    result = await ProposalExecutor(compute)._execute_inverse_folding(
        output({"metadata": {"soluble_mpnn_parameters": {"seed": 42}}}), context(3)
    )
    assert [r.num_sequences for r in compute.requests] == [3, 2]
    assert [r.parameters["seed"] for r in compute.requests] == [42, 43]
    assert [p.chains["B"] for p in result] == ["GCDE", "HCDE", "ICDE"]


async def test_exhausted_sampling_reports_shortfall_instead_of_returning_partial_batch():
    compute = Compute([["GCDE", "GCDE"], ["GCDE"], ["GCDE"]])
    with pytest.raises(ProposalValidationError, match="1/2 valid unique candidates after 3 sampling rounds"):
        await ProposalExecutor(compute)._execute_inverse_folding(output({}), context(2))
    assert len(compute.requests) == 3
