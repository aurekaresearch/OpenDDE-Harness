"""Cache-friendly prompt assembly preserves skill permissions and live inputs."""

import asyncio
import json
from pathlib import Path
from string import Formatter
from unittest.mock import AsyncMock

import pytest

from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
from opendde_harness.plugin.protein_design.agents.prompt_context import bounded_design_memories, context_json
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.agents.skills import SkillDocument
from opendde_harness.plugin.protein_design.prompts.antibody_design import DESIGN_PROMPT, DESIGN_SYSTEM_PROMPT
from opendde_harness.plugin.protein_design.prompts.parent_selection import PARENT_SELECTION_PROMPT
from opendde_harness.plugin.protein_design.prompts.quality_check import QUALITY_CHECK_BATCH_PROMPT
from opendde_harness.plugin.protein_design.prompts.reflect import REFLECT_ANALYSIS_PROMPT
from opendde_harness.providers.base import LLMProvider, LLMResponse


def skill(name, source="builtin"):
    return SkillDocument(name, Path("/unused") / name, f"Instructions for {name}", (), source=source)


def test_builtin_order_is_stable_and_memory_does_not_change_system_prefix():
    session = OpenDDEHarnessStructuredSession(None, "test")
    profile = AGENT_PROFILES[AgentRole.DESIGN]
    a, b = skill("a"), skill("b")
    prefix = session._system_message(profile)
    assert a.content not in prefix and b.content not in prefix
    user = session._user_message("cycle evidence", [b, a])
    assert a.content not in user and b.content not in user
    assert "`local/a`" in user and "`local/b`" in user
    assert "Instructions for advice" not in prefix
    assert "advice" in session._user_message("cycle evidence", [skill("advice", "memory")])
    assert session._system_message(profile) == prefix


def test_catalog_changes_do_not_replace_the_prompt_prefix():
    session = OpenDDEHarnessStructuredSession(None, "test")
    prompt = "Stable analysis\nCurrent evidence"
    for skills in ([skill("a")], [skill("a"), skill("advice", "memory")], []):
        assert session._user_message(prompt, skills).startswith(prompt)


def test_memory_excerpts_are_bounded_deduplicated_and_do_not_mutate_records():
    memories = [" ", "lesson one", " lesson   one ", *[str(i) + "界" * 3000 for i in range(8)]]
    original = list(memories)
    excerpts = bounded_design_memories(memories)
    assert len(excerpts) == 4
    assert excerpts[0] == "lesson one"
    assert all(len(text) <= 600 for text in excerpts)
    assert len("\n".join(excerpts)) <= 2000
    assert all(text.endswith("[excerpt truncated]") for text in excerpts[1:])
    assert memories == original
    assert bounded_design_memories(excerpts) == excerpts
    assert bounded_design_memories([]) == []


def test_memory_total_budget_includes_separators_and_truncation_markers():
    excerpts = bounded_design_memories([str(i) + "x" * 3000 for i in range(8)])
    assert len(excerpts) == 4
    assert len("\n".join(excerpts)) == 2000
    assert len(excerpts[-1]) == 197
    assert bounded_design_memories(excerpts) == excerpts


def test_builtin_requires_use_skill_before_finishing():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat_with_retry.return_value = LLMResponse(content='{"downstream_header":"ok","report":"ok"}')
    session = OpenDDEHarnessStructuredSession(provider, "test", max_attempts=1)
    with pytest.raises(ValueError, match="no available skill was loaded"):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "evidence", skills=[skill("builtin")]))
    assert provider.chat_with_retry.await_count == 1
    messages = provider.chat_with_retry.call_args.kwargs["messages"]
    assert "Instructions for builtin" not in str(messages)


def test_memory_only_skill_still_requires_loading():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat_with_retry.return_value = LLMResponse(content='{"downstream_header":"ok","report":"ok"}')
    session = OpenDDEHarnessStructuredSession(provider, "test", max_attempts=1)
    with pytest.raises(ValueError, match="no available skill was loaded"):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "evidence", skills=[skill("advice", "memory")]))


def test_design_changes_are_after_fixed_constraints_and_output_rules():
    fields = {name: f"value_{name}" for _, name, _, _ in Formatter().parse(DESIGN_PROMPT) if name}
    assert "learned_skill_context" not in fields
    first = DESIGN_PROMPT.format(**fields)
    changed = {
        **fields,
        "parent_binder_sequence": "new_parent",
        "cycle_num": "2",
        "cdr_contact_gate_feedback": "new_gate",
    }
    second = DESIGN_PROMPT.format(**changed)
    prefix = first.split("=== PARENT ===")[0]
    assert second.startswith(prefix)
    assert "=== TARGET ===" not in first
    assert "mutable_positions_formatted" not in fields
    assert "=== OUTPUT RULES ===" not in first
    assert "[chain_id, one_based_position, new_residue]" in DESIGN_SYSTEM_PROMPT
    assert fields["design_skill_route"] in prefix
    assert fields["num_sequences"] in prefix
    assert fields["num_mutations_instruction"] in prefix
    assert "new_parent" in second and "new_gate" in second
    for name in ("feedback_summary", "long_term_memory_context"):
        assert fields[name] in prefix
        updated = DESIGN_PROMPT.format(**{**fields, name: "updated_context"})
        assert "updated_context" in updated and fields[name] not in updated
    sections = [
        "=== LONG-TERM DESIGN MEMORY ===",
        "=== CURRENT REFLECTION ===",
        "=== PARENT ===",
        "=== SEARCH STATE ===",
        "=== CDR CONTACT-FRACTION GATE FEEDBACK (UPDATED EVERY CYCLE) ===",
        "=== QC WARNINGS ===",
        "=== CYCLE REQUIREMENTS ===",
    ]
    positions = [first.index(section) for section in sections]
    assert positions == sorted(positions)
    assert all(first.count(section) == 1 for section in sections)


def test_antibody_rules_are_static_and_not_repeated_in_cycle_template():
    assert not [name for _, name, _, _ in Formatter().parse(DESIGN_SYSTEM_PROMPT) if name]
    assert AGENT_PROFILES[AgentRole.DESIGN].system_prompt == DESIGN_SYSTEM_PROMPT
    for rule in (
        "Never translate chain",
        "Candidates must produce distinct sequences",
        "Learned skills are advisory",
        "soluble_mpnn_parameters",
        "[chain_id, one_based_position, new_residue]",
        "Python-selected parent",
    ):
        assert rule in DESIGN_SYSTEM_PROMPT
        assert rule not in DESIGN_PROMPT
    for policy in ("`configured`", "`cdr_proposal`", "`all_mutable`", "`candidates: []`"):
        assert policy in DESIGN_SYSTEM_PROMPT
    fields = {name for _, name, _, _ in Formatter().parse(DESIGN_PROMPT) if name}
    assert {
        "num_sequences",
        "num_mutations_instruction",
        "design_skill_route",
        "feedback_summary",
        "long_term_memory_context",
        "cycle_num",
    } <= fields


@pytest.mark.parametrize(
    "template,boundary,changes,stable_fields",
    [
        (
            QUALITY_CHECK_BATCH_PROMPT,
            "<objective_developability_evidence>",
            {"objective_tool_results": "new QC", "candidates": "new candidates"},
            ["phase_analyze_summary"],
        ),
        (
            REFLECT_ANALYSIS_PROMPT,
            "<quality_evidence>",
            {"quality_check_summary": "new QC", "parent_name": "new parent", "cycle_num": "8"},
            ["phase_analyze_summary", "objective_key", "binder_chain_ids", "target_chain_ids"],
        ),
        (
            PARENT_SELECTION_PROMPT,
            "Current population:",
            {"candidate_table": "new population", "cycle_num": "8"},
            ["reflect_feedback"],
        ),
    ],
)
def test_other_phase_prompts_keep_rules_before_live_evidence(template, boundary, changes, stable_fields):
    fields = {name: f"value_{name}" for _, name, _, _ in Formatter().parse(template) if name}
    first = template.format(**fields)
    second = template.format(**{**fields, **changes})
    prefix = first.split(boundary)[0]
    assert second.startswith(prefix)
    for name in stable_fields:
        assert fields[name] in prefix
    for value in changes.values():
        assert value in second


def test_canonical_context_preserves_list_order_and_ignores_mapping_insertion_order():
    first = {"B": {"iptm": 0.8, "loss": 1.2}, "A": [3, 1, 2]}
    second = {"A": [3, 1, 2], "B": {"loss": 1.2, "iptm": 0.8}}
    assert context_json(first) == context_json(second)
    assert json.loads(context_json(first)) == first


def test_fresh_phase_messages_reuse_provider_without_growing_history():
    provider = AsyncMock(spec=LLMProvider)
    provider.chat_with_retry.return_value = LLMResponse(content='{"downstream_header":"ok","report":"ok"}')
    session = OpenDDEHarnessStructuredSession(provider, "test", max_attempts=1)

    async def run_twice():
        for prompt in ("first evidence", "next evidence"):
            await session.run(AGENT_PROFILES[AgentRole.ANALYZE], prompt)

    asyncio.run(run_twice())
    first, second = [call.kwargs["messages"] for call in provider.chat_with_retry.call_args_list]
    assert len(first) == len(second) == 2
    assert first[0] == second[0]
    assert first[1] != second[1]


@pytest.mark.parametrize(
    "selected,advice,error",
    [
        ("unknown", [], "not an allowed"),
        ("builtin", ["advice"], "was not loaded"),
        ("builtin", ["unknown"], "unknown advisory"),
        ("advice", [], "not an allowed"),
    ],
)
def test_preloading_does_not_bypass_primary_or_advisory_permissions(selected, advice, error):
    provider = AsyncMock(spec=LLMProvider)
    provider.chat_with_retry.return_value = LLMResponse(
        content=json.dumps({"skill_id": selected, "applied_learned_skill_ids": advice, "candidates": []})
    )
    session = OpenDDEHarnessStructuredSession(provider, "test", max_attempts=1)
    with pytest.raises(ValueError, match=error):
        asyncio.run(
            session.run(
                AGENT_PROFILES[AgentRole.DESIGN], "evidence", skills=[skill("builtin"), skill("advice", "memory")]
            )
        )
