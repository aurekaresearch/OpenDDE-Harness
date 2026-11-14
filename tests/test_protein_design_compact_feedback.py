"""Feedback stays concise while preserving failed and unknown gate states."""

import json

from opendde_harness.plugin.protein_design.core.contracts import Candidate, WorkflowConfig
from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator


def test_feedback_is_compact_and_does_not_modify_evidence():
    evidence = {"reason": "failed", "framework_contact_residue_ids": list(range(1000))}
    candidate = Candidate(
        candidate_id="one", sequence="A" * 1000, objective=1, metadata={"gate_evidence": evidence, "gate_passed": False}
    )
    before = candidate.model_dump()
    unknown = Candidate(candidate_id="unknown", sequence="A", objective=2)
    feedback = DesignOrchestrator._search_feedback(
        [candidate, unknown],
        candidate,
        [unknown, candidate],
        WorkflowConfig(target="example"),
        {"candidate_id": "parent", "objective": 3},
        {str(i): i for i in range(20)},
    )
    value = json.loads(feedback)
    assert value["candidate_changes"]["one"]["gate"]["passed"] is False
    assert value["candidate_changes"]["unknown"]["gate"]["passed"] is None
    assert value["candidate_changes"]["one"]["delta"] == -2
    assert value["population_best"]["id"] == "one"
    assert value["working_parent"]["gate"]["framework_contact_count"] == 1000
    assert len(value["recurring_offenders"]) == 5
    assert value["recurring_offenders_omitted"] == 15
    assert "sequence" not in value["working_parent"]
    assert "framework_contact_residue_ids" not in feedback
    assert candidate.model_dump() == before
    assert len(feedback) < 2000
