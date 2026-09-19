"""Quality decisions use current candidate sequences, never stale scaffold annotations."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from test_protein_design_orchestrator import make_config

from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.core.contracts import (
    AnalyzeAgentOutput,
    Candidate,
    QualityBatchOutput,
    QualityCandidateResult,
)
from opendde_harness.providers.base import LLMResponse

SCAFFOLD = "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS"
REFOLDED = "EVQLVESGGGLVQPGGSLRLSCAASGRDFSDHSMAWFRQAPGKGRELVAARSRTGGSTLYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRDEDGRIRTEESEYTFWGQGTQVTVSS"


def quality_case(sequence=REFOLDED):
    groups = [list(range(25, 35)), list(range(49, 59)), list(range(98, 117))]
    cdr = [position for group in groups for position in group]
    config = make_config(
        binder_chains={"D": SCAFFOLD},
        cdr_regions={"D": cdr},
        cdr_region_groups={"D": groups},
        fixed_residues={"D": [position for position in range(len(SCAFFOLD)) if position not in cdr]},
        mutable_positions={"D": cdr},
        metadata={"binder_type": "VHH"},
    )
    candidate = Candidate(
        candidate_id="refolded",
        sequence=sequence,
        objective=0.35,
        metrics={"iptm": 0.79},
        structure_path="current.cif",
        metadata={
            "chains": {"D": sequence},
            "fold": {"sequences": {"D": sequence}},
            "gate_passed": True,
            "gate_evidence": {"reason": "cdr_contact_fraction_passed", "contacts": ["irrelevant"] * 1000},
            "pyrosetta": {
                "status": "success",
                "metrics": {"rosetta_interface_dg": -42.9},
                "contact_residues": ["huge-contact-list"] * 1000,
            },
            "loss": {"duplicated_breakdown": "not-needed-in-quality-prompt"},
        },
    )
    return config, candidate


def test_current_cdr_sequence_and_cysteine_positions_are_authoritative():
    config, candidate = quality_case()
    evidence = ProteinDesignPhases._quality_candidate_evidence(candidate, config)
    chain = evidence["sequence_evidence"]["chains"]["D"]
    assert evidence["sequence_evidence"]["index_base"] == 0
    assert [group["sequence"] for group in chain["cdr_groups"]] == ["GRDFSDHSMA", "ARSRTGGSTL", "AGVRDEDGRIRTEESEYTF"]
    assert chain["cdr_groups"][2]["positions"] == list(range(98, 117))
    assert [row["position"] for row in chain["cysteines"]] == [21, 95]
    assert all(row["fixed"] and not row["mutable"] and not row["in_configured_cdr"] for row in chain["cysteines"])
    assert "Not established" in chain["cysteine_bond_state"]
    assert candidate.metadata["quality_sequence_evidence"] == evidence["sequence_evidence"]
    assert "contact_residues" not in evidence["pyrosetta"]
    assert "contacts" not in evidence["gate_evidence"]


def test_quality_prompt_omits_stale_analysis_and_preserves_high_risk_rejection():
    config, candidate = quality_case()
    output = QualityBatchOutput(
        results={
            "refolded": {
                "liability": "High Risk",
                "overall_risk_level": "High Risk",
                "pass_check": False,
                "reasoning": "Current measured risk requires rejection",
            }
        }
    )
    session = SimpleNamespace(run=AsyncMock(return_value=output))
    compute = SimpleNamespace(developability=AsyncMock(return_value={"results": {"refolded": {"available": False}}}))
    phases = ProteinDesignPhases(session, compute, None, catalog=SimpleNamespace(select=lambda _: []))
    actual = asyncio.run(
        phases.quality_cycle(
            config,
            1,
            [candidate],
            AnalyzeAgentOutput(downstream_header="INCORRECT historical H3 Cys98 CAAAGVRAEDGRVRTLPSE and NPM motif"),
        )
    )
    prompt = session.run.call_args.args[1]
    assert "INCORRECT historical" not in prompt
    assert "CAAAGVRAEDGRVRTLPSE" not in prompt
    assert "NPM motif" not in prompt
    assert "huge-contact-list" not in prompt
    assert "not-needed-in-quality-prompt" not in prompt
    rows = json.loads(prompt.split("<candidates>\n", 1)[1].split("\n</candidates>", 1)[0])
    assert rows[0]["sequence"] == REFOLDED
    assert rows[0]["sequence_evidence"]["chains"]["D"]["cdr_groups"][2]["sequence"] == "AGVRDEDGRIRTEESEYTF"
    assert actual.results["refolded"].pass_check is False
    assert candidate.metadata["quality_check"]["pass_check"] is False
    assert "any High Risk dimension means overall" in prompt
    assert "not automatically CDR liabilities" in AGENT_PROFILES[AgentRole.QUALITY].system_prompt


def test_introduced_cdr_cysteine_is_reported_without_claiming_bond_state():
    sequence = REFOLDED[:100] + "C" + REFOLDED[101:]
    config, candidate = quality_case(sequence)
    chain = ProteinDesignPhases._quality_candidate_evidence(candidate, config)["sequence_evidence"]["chains"]["D"]
    introduced = next(item for item in chain["cysteines"] if item["position"] == 100)
    assert introduced == {"position": 100, "residue": "C", "in_configured_cdr": True, "fixed": False, "mutable": True}


def test_quality_chain_roles_come_from_validated_config_not_chain_letter_or_analysis():
    config, candidate = quality_case()
    config.binder_chains = {"X": SCAFFOLD, "Y": SCAFFOLD}
    for field in ("cdr_regions", "cdr_region_groups", "fixed_residues", "mutable_positions"):
        values = getattr(config, field)["D"]
        setattr(config, field, {"X": values, "Y": values})
    config.metadata["source_config"] = {
        "initial_binders": [{"chains": {"X": {"chain_type": "VH"}, "Y": {"chain_type": "VL"}}}]
    }
    candidate.sequence = REFOLDED * 2
    candidate.metadata["chains"] = {"X": REFOLDED, "Y": REFOLDED}
    candidate.metadata["fold"]["sequences"] = candidate.metadata["chains"]
    chains = ProteinDesignPhases._quality_candidate_evidence(candidate, config)["sequence_evidence"]["chains"]
    assert chains["X"]["configured_chain_type"] == "VH"
    assert chains["Y"]["configured_chain_type"] == "VL"


def test_n_glycosylation_scan_excludes_proline_and_requires_serine_or_threonine():
    sequence = REFOLDED[:25] + "NPSNATNPMW" + REFOLDED[35:]
    config, candidate = quality_case(sequence)
    chain = ProteinDesignPhases._quality_candidate_evidence(candidate, config)["sequence_evidence"]["chains"]["D"]
    sequons = chain["potential_n_glycosylation_sequons"]
    assert [(item["position"], item["motif"]) for item in sequons] == [(28, "NAT")]
    assert sequons[0]["mutable"] is True


@pytest.mark.parametrize("mismatch", ["candidate", "fold", "chain_ids", "length"])
def test_quality_evidence_fails_on_sequence_identity_mismatch(mismatch):
    config, candidate = quality_case()
    if mismatch == "candidate":
        candidate.sequence = SCAFFOLD
    elif mismatch == "fold":
        candidate.metadata["fold"]["sequences"]["D"] = SCAFFOLD
    elif mismatch == "chain_ids":
        candidate.metadata["chains"] = {"E": REFOLDED}
    else:
        candidate.sequence = REFOLDED[:-1]
        candidate.metadata["chains"]["D"] = candidate.sequence
    with pytest.raises(ValueError, match="Quality evidence"):
        ProteinDesignPhases._quality_candidate_evidence(candidate, config)


@pytest.mark.parametrize(
    "field",
    ["expressivity", "immunogenicity", "aggregation", "solubility", "specificity", "liability", "overall_risk_level"],
)
@pytest.mark.parametrize("risk", ["High Risk", "HIGH RISK", " high  risk ", "High-Risk", "HighRisk", "high", "HIGH"])
def test_high_risk_quality_verdict_cannot_claim_pass(field, risk):
    values = {"overall_risk_level": "Medium Risk", "reasoning": "Evidence", "pass_check": True, field: risk}
    with pytest.raises(ValidationError, match=f"High Risk fields: {field}"):
        QualityCandidateResult.model_validate(values)


@pytest.mark.parametrize(
    "risk,passed",
    [
        ("Low Risk", False),
        ("Low Risk", True),
        ("Medium Risk", False),
        ("Medium Risk", True),
        ("Unknown", False),
        ("Unknown", True),
        ("High Risk", False),
    ],
)
def test_quality_risk_guard_preserves_existing_consistent_decisions(risk, passed):
    values = {"overall_risk_level": risk, "liability": risk, "reasoning": "Uncertainty retained", "pass_check": passed}
    result = QualityCandidateResult.model_validate(values)
    assert result.pass_check is passed
    assert result.overall_risk_level == risk
    assert result.liability == risk


class QualityProvider:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    async def chat_with_retry(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return LLMResponse(json.dumps({"results": {"refolded": next(self.decisions)}}))


def run_quality_provider(provider, candidate, config):
    phases = ProteinDesignPhases(
        OpenDDEHarnessStructuredSession(provider, "test"),
        SimpleNamespace(developability=AsyncMock(return_value={"results": {}})),
        None,
        catalog=SimpleNamespace(select=lambda _: []),
    )
    return asyncio.run(phases.quality_cycle(config, 1, [candidate], AnalyzeAgentOutput(downstream_header="unused")))


def test_contradictory_quality_response_uses_actual_structured_repair_path():
    config, candidate = quality_case()
    contradictory = {
        "liability": "HIGH-RISK",
        "overall_risk_level": "Medium Risk",
        "pass_check": True,
        "reasoning": "Risk",
    }
    consistent = {**contradictory, "overall_risk_level": "High Risk", "pass_check": False}
    provider = QualityProvider([contradictory, consistent])
    output = run_quality_provider(provider, candidate, config)
    assert len(provider.calls) == 2
    assert "A High Risk quality assessment cannot pass_check=true" in json.dumps(provider.calls[1]["messages"])
    assert output.results["refolded"].pass_check is False
    assert candidate.metadata["quality_check"]["pass_check"] is False


def test_persistently_contradictory_quality_cannot_reach_admission():
    config, candidate = quality_case()
    contradictory = {"overall_risk_level": "High Risk", "pass_check": True, "reasoning": "Risk"}
    provider = QualityProvider([contradictory] * 3)
    with pytest.raises(ValueError, match="(?s)produced no valid QualityBatchOutput.*High Risk"):
        run_quality_provider(provider, candidate, config)
    assert len(provider.calls) == 3
    assert "quality_check" not in candidate.metadata
