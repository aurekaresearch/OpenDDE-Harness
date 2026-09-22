"""Task-level system context is independent of cycle evidence and skill routing."""

from opendde_harness.plugin.protein_design.agents.phases import ProteinDesignPhases
from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
from opendde_harness.plugin.protein_design.agents.skills import ProteinDesignSkillCatalog
from opendde_harness.plugin.protein_design.core.contracts import WorkflowConfig
from opendde_harness.plugin.protein_design.prompts.antibody_design import DESIGN_PROMPT


def config():
    return WorkflowConfig(
        target="test-target",
        target_sequence="ACDE",
        target_chains={"A": "ACDE"},
        binder_chains={"B": "FGHI"},
        mutable_positions={"B": [1, 2]},
        fixed_residues={"B": [0, 3]},
        hotspots=[2],
    )


def test_system_is_identical_when_cycles_routes_and_memory_change():
    cfg = config()
    session = OpenDDEHarnessStructuredSession(None, "test")
    first = session._system_message(ProteinDesignPhases._design_profile(cfg))
    cfg.metadata.update(cycle=99, reflection="new reflection", gate_feedback="new gate", no_improvement_streak=10)
    cfg.candidates_per_cycle = 12
    second = session._system_message(ProteinDesignPhases._design_profile(cfg))
    assert first == second
    assert "test-target" in first and '"A":"ACDE"' in first
    assert '"B":[1,2]' in first and '"B":[0,3]' in first
    assert "new reflection" not in first and "new gate" not in first
    assert "=== TARGET ===" not in DESIGN_PROMPT
    assert "=== FIXED DESIGN CONSTRAINTS ===" not in DESIGN_PROMPT
    assert "test-target" not in AGENT_PROFILES[AgentRole.DESIGN].system_prompt


def test_task_context_updates_for_explicit_configuration_changes_without_global_mutation():
    cfg = config()
    before = ProteinDesignPhases._design_profile(cfg).system_prompt
    cfg.fold_options["target_hotspots"] = [3]
    assert ProteinDesignPhases._design_profile(cfg).system_prompt != before
    assert ProteinDesignPhases._design_profile(config()).system_prompt == before
    cfg.target = "different-target"
    assert "different-target" in ProteinDesignPhases._design_profile(cfg).system_prompt


def test_design_skills_are_advertised_without_loading_bodies():
    session = OpenDDEHarnessStructuredSession(None, "test")
    skills = ProteinDesignSkillCatalog.builtin().select(["cdr-full-redesign"])
    profile = ProteinDesignPhases._design_profile(config())
    user = session._user_message("cycle evidence", skills)
    assert skills[0].content.rstrip() not in user
    assert "`local/cdr-full-redesign`" in user
    assert skills[0].content.rstrip() not in session._system_message(profile)
    assert user.startswith("cycle evidence")
