"""Typed agent phase adapters for OpenDDE Harness protein design."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
from opendde_harness.plugin.protein_design.agents.proposals import ProposalContext, ProposalExecutor
from opendde_harness.plugin.protein_design.agents.reflection import ReflectOutput
from opendde_harness.plugin.protein_design.agents.router import (
    FULL_REDESIGN_SKILL,
    POINT_MUTATION_SKILL,
    DesignRouteContext,
    route_design_skills,
)
from opendde_harness.plugin.protein_design.agents.session import StructuredSession
from opendde_harness.plugin.protein_design.agents.skills import ProteinDesignSkillCatalog
from opendde_harness.plugin.protein_design.core.contracts import (
    AnalyzeAgentOutput,
    Candidate,
    DesignAgentOutput,
    ParentSelectionOutput,
    PostFilterAgentOutput,
    QualityBatchOutput,
    WorkflowConfig,
)
from opendde_harness.plugin.protein_design.core.design_cases import recurring_offender_ids
from opendde_harness.plugin.protein_design.prompts import (
    ANALYZE_REPORT_PROMPT,
    DESIGN_PROMPT,
    PARENT_SELECTION_PROMPT,
    QUALITY_CHECK_BATCH_PROMPT,
    REFLECT_ANALYSIS_PROMPT,
)
from opendde_harness.plugin.protein_design.tools.agent import ToolContext


def _forced_design_skill(config: WorkflowConfig, cycle: int) -> str | None:
    """Return an explicitly configured basin-reset skill, if one is due."""
    if cycle < config.bootstrap_full_redesign_cycles:
        return FULL_REDESIGN_SKILL
    threshold = config.stagnation_full_redesign_threshold
    streak = int(config.metadata.get("no_improvement_streak", 0))
    last_cycle = int(config.metadata.get("last_stagnation_redesign_cycle", -threshold))
    if threshold and streak >= threshold and cycle - last_cycle >= threshold:
        config.metadata["last_stagnation_redesign_cycle"] = cycle
        return FULL_REDESIGN_SKILL
    return None


@dataclass(frozen=True)
class DesignCycleResult:
    fold_candidates: list[dict[str, Any]]
    selected_skill_id: str
    memories: list[str]
    learned_skill_ids: tuple[str, ...] = ()
    applied_learned_skill_ids: tuple[str, ...] = ()


class ProteinDesignPhases:
    def __init__(
        self,
        session: StructuredSession,
        compute: Any,
        memory: Any,
        *,
        catalog: ProteinDesignSkillCatalog | None = None,
    ) -> None:
        self._session = session
        self._compute = compute
        self._memory = memory
        self._catalog = catalog or ProteinDesignSkillCatalog.builtin()
        self._executor = ProposalExecutor(compute)

    async def analyze_once(self, config: WorkflowConfig) -> AnalyzeAgentOutput:
        prompt = ANALYZE_REPORT_PROMPT.format(
            target_name=config.target,
            target_sequence=config.target_chains,
            target_length=len(config.target_sequence),
            hotspots=config.fold_options.get("target_hotspots", config.hotspots),
            binder_name=self._initial_parent(config).get("candidate_id", "initial_binder"),
            binder_sequence=self._parent_chains(self._initial_parent(config)),
            binder_length=sum(len(value) for value in self._parent_chains(self._initial_parent(config)).values()),
            binder_fixed_residues=config.fixed_residues,
            binder_cdr_regions=config.cdr_regions,
        )
        profile = AGENT_PROFILES[AgentRole.ANALYZE]
        output = await self._session.run(
            profile,
            prompt,
            skills=self._catalog.select(profile.default_skills),
            tool_context=ToolContext(
                compute=self._compute,
                metadata=self._tool_metadata(
                    config,
                    target=config.target,
                    task_id=config.metadata.get("task_id"),
                ),
            ),
        )
        if not isinstance(output, AnalyzeAgentOutput):
            output = AnalyzeAgentOutput.model_validate(output)
        return output

    async def select_parent(
        self,
        config: WorkflowConfig,
        cycle: int,
        candidates: list[Candidate],
    ) -> Candidate:
        ordered = sorted(
            candidates,
            key=lambda candidate: float(candidate.objective)
            if candidate.objective is not None
            else float("inf"),
            reverse=not config.minimize,
        )
        lines = [
            f"Optimization metric: {config.objective_key} "
            f"({'lower' if config.minimize else 'higher'} is better)",
            "| Candidate | Parent | Score | ipTM | pLDDT | ipSAE |",
            "|:--|:--|--:|--:|--:|--:|",
        ]
        for candidate in ordered:
            lines.append(
                f"| {candidate.candidate_id} | "
                f"{candidate.metadata.get('parent_id', 'None')} | "
                f"{candidate.objective} | {self._metric(candidate, 'iptm')} | "
                f"{self._metric(candidate, 'plddt')} | "
                f"{self._metric(candidate, 'ipsae')} |"
            )
        prompt = PARENT_SELECTION_PROMPT.format(
            cycle_num=cycle,
            candidate_table="\n".join(lines),
            reflect_feedback=config.metadata.get(
                "reflection",
                "No reflection feedback is available yet.",
            ),
        )
        profile = AGENT_PROFILES[AgentRole.PARENT_SELECTION]
        try:
            output = await self._session.run(
                profile,
                prompt,
                skills=(),
                tool_context=ToolContext(
                    compute=self._compute,
                    metadata=self._tool_metadata(
                        config,
                        task_id=config.metadata.get("task_id"),
                        cycle=cycle,
                    ),
                ),
            )
            if not isinstance(output, ParentSelectionOutput):
                output = ParentSelectionOutput.model_validate(output)
            selected_name = output.parent_selection.selected_parent_name
            selected = next(
                candidate
                for candidate in ordered
                if candidate.candidate_id == selected_name
            )
            config.metadata["parent_selection"] = output.model_dump(mode="json")
            return selected
        except Exception as exc:
            config.metadata["parent_selection"] = {
                "strategy": "llm",
                "fallback": "population_best",
                "error": str(exc),
            }
            return ordered[0]

    async def design_cycle(
        self,
        config: WorkflowConfig,
        cycle: int,
        analysis: AnalyzeAgentOutput,
        parents: list[dict[str, Any]],
        best: Candidate | None,
    ) -> DesignCycleResult:
        parent = parents[0] if parents else self._initial_parent(config)
        chains = self._parent_chains(parent)
        query = f"cycle={cycle}; best={best.model_dump() if best else None}"
        memories = await self._memory.retrieve(config.target, query)
        learned = await self._memory.retrieve_skills(config.target, query)
        catalog = self._catalog.overlay_learned(learned)
        learned_names = tuple(
            name
            for name in catalog.names()
            if name not in self._catalog.names() and "design" in catalog.require(name).roles
        )
        route = route_design_skills(
            DesignRouteContext(
                parent_sequences=chains,
                mutable_positions=config.mutable_positions,
                population_size=int(config.metadata.get("population_size", len(parents))),
                inverse_folding_available=(
                    bool(parent.get("structure_path") or config.initial_structure_path)
                ),
                esm2_available=config.esm2_available,
                skill_weights=config.skill_weights,
                force_skill_id=_forced_design_skill(config, cycle),
            )
        )
        prompt = DESIGN_PROMPT.format(
            target_name=config.target,
            target_sequence=config.target_chains,
            target_length=len(config.target_sequence),
            hotspots=config.fold_options.get("target_hotspots", config.hotspots),
            binder_chain_ids=list(chains),
            parent_binder_sequence=chains,
            parent_binder_length=sum(len(value) for value in chains.values()),
            parent_structure_path=parent.get("structure_path") or config.initial_structure_path,
            parent_iptm=self._metric(best, "iptm"),
            parent_plddt=self._metric(best, "plddt"),
            parent_ipsae=self._metric(best, "ipsae"),
            parent_ranking_score=best.objective if best else None,
            metric_context=best.metrics if best else {},
            mutable_positions_formatted=dict(config.mutable_positions),
            antibody_population_info=parents,
            parent_selection_mode="python_deterministic",
            parent_selection_guidance="Python selected the parent; do not replace it.",
            phase_analyze_summary=analysis.downstream_header,
            cdr_contact_gate_feedback=config.metadata.get("gate_feedback", "not available"),
            design_skill_route=route.prompt_block(),
            no_improvement_streak=config.metadata.get("no_improvement_streak", 0),
            stagnation_guidance=config.metadata.get("stagnation_guidance", "Use current evidence."),
            long_term_memory_context="\n".join(memories) or "none",
            learned_skill_context=(
                "Retrieved advisory skills: " + ", ".join(learned_names)
                if learned_names
                else "No relevant learned design skills were retrieved."
            ),
            feedback_summary=config.metadata.get("reflection", "none"),
            quality_check_summary=config.metadata.get("quality", "none"),
            num_sequences=config.candidates_per_cycle,
            cycle_num=cycle,
            num_mutations_instruction=config.mutation_count_instruction,
        )
        profile = AGENT_PROFILES[AgentRole.DESIGN]
        skill_names = route.allowed_skill_ids
        output = await self._session.run(
            profile,
            prompt,
            skills=catalog.for_role("design", (*skill_names, *learned_names)),
            tool_context=ToolContext(
                compute=self._compute,
                metadata=self._tool_metadata(
                    config,
                    target=config.target,
                    task_id=config.metadata.get("task_id"),
                    cycle=cycle,
                    parent=parent,
                ),
            ),
        )
        if not isinstance(output, DesignAgentOutput):
            output = DesignAgentOutput.model_validate(output)
        applied_learned = self._validated_learned_skill_ids(
            output.applied_learned_skill_ids, learned_names
        )
        context = ProposalContext(
            parent_id=str(parent.get("candidate_id") or parent.get("id") or "parent"),
            parent_sequences=chains,
            mutable_positions=config.mutable_positions,
            parent_structure_path=parent.get("structure_path") or config.initial_structure_path,
            candidate_count=config.candidates_per_cycle,
            cycle=cycle,
            placement=config.placement,
        )
        selected_skill_id = output.skill_id
        proposals = []
        repair_prompt = prompt
        for attempt in range(3):
            try:
                batch = await self._executor.execute(output, route, context)
            except Exception as exc:
                batch = []
                self._executor.last_errors = [str(exc)]
            existing_ids = {proposal.candidate_id for proposal in proposals}
            existing_sequences = {
                tuple(sorted(proposal.chains.items())) for proposal in proposals
            }
            for proposal in batch:
                fingerprint = tuple(sorted(proposal.chains.items()))
                if (
                    proposal.candidate_id in existing_ids
                    or fingerprint in existing_sequences
                ):
                    continue
                proposals.append(proposal)
                existing_ids.add(proposal.candidate_id)
                existing_sequences.add(fingerprint)
            if len(proposals) >= config.candidates_per_cycle:
                break
            if selected_skill_id not in {POINT_MUTATION_SKILL, FULL_REDESIGN_SKILL}:
                break
            if attempt == 2:
                break
            repair_prompt = (
                f"{prompt}\n\n"
                "Repair only the missing invalid candidates. Preserve the selected "
                f"skill_id {selected_skill_id!r}. Do not repeat valid candidate IDs "
                f"{sorted(existing_ids)}. Validation errors: "
                f"{self._executor.last_errors}. Return corrected JSON only."
            )
            output = await self._session.run(
                profile,
                repair_prompt,
                skills=catalog.for_role("design", (*skill_names, *learned_names)),
                tool_context=ToolContext(
                    compute=self._compute,
                    metadata=self._tool_metadata(
                        config,
                        target=config.target,
                        task_id=config.metadata.get("task_id"),
                        cycle=cycle,
                        parent=parent,
                        repair_attempt=attempt + 1,
                    ),
                ),
            )
            if not isinstance(output, DesignAgentOutput):
                output = DesignAgentOutput.model_validate(output)
            if output.skill_id != selected_skill_id:
                output = output.model_copy(update={"skill_id": selected_skill_id})
            applied_learned = self._validated_learned_skill_ids(
                output.applied_learned_skill_ids, learned_names
            )
        if not proposals:
            raise ValueError(
                "Design Agent produced no valid candidates after format repair: "
                f"{self._executor.last_errors}"
            )
        fold_candidates = []
        for proposal in proposals:
            sequence = "".join(proposal.chains.values())
            candidate_id = f"c{cycle:04d}_{proposal.candidate_id}"
            fold_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "sequence": sequence,
                    "chains": {**config.target_chains, **proposal.chains},
                    "metadata": {
                        **proposal.metadata,
                        "parent_id": proposal.parent_id,
                        "skill_id": selected_skill_id,
                        "applied_learned_skill_ids": list(applied_learned),
                        "mutations": [item.model_dump() for item in proposal.mutations],
                        "strategy": proposal.strategy,
                        "risk_level": proposal.risk_level,
                        "execution_backend": proposal.execution_backend,
                    },
                }
            )
        config.metadata["learned_skills_retrieved"] = list(learned_names)
        config.metadata["learned_skills_applied"] = list(applied_learned)
        return DesignCycleResult(
            fold_candidates,
            selected_skill_id,
            memories,
            learned_names,
            applied_learned,
        )

    @staticmethod
    def _validated_learned_skill_ids(
        requested: list[str], available: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Keep only retrieved advisory skills; never let memory bypass the Router."""
        allowed = set(available)
        return tuple(name for name in dict.fromkeys(requested) if name in allowed)[:3]

    async def quality_cycle(
        self,
        config: WorkflowConfig,
        cycle: int,
        candidates: list[Candidate],
        analysis: AnalyzeAgentOutput,
    ) -> QualityBatchOutput:
        payload = {
            "task_id": config.metadata.get("task_id"),
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "binder_chain_ids": list(config.binder_chains),
            "output_dir": config.fold_options.get("output_dir"),
        }
        try:
            objective = await self._compute.developability(payload)
        except Exception as exc:
            objective = {"available": False, "error": str(exc), "results": []}
        prompt = QUALITY_CHECK_BATCH_PROMPT.format(
            phase_analyze_summary=analysis.downstream_header,
            objective_tool_results=json.dumps(objective, ensure_ascii=False, default=str),
            candidates=json.dumps(payload["candidates"], ensure_ascii=False, default=str),
        )
        profile = AGENT_PROFILES[AgentRole.QUALITY]
        output = await self._session.run(
            profile,
            prompt,
            skills=self._catalog.select(profile.default_skills),
            tool_context=ToolContext(
                compute=self._compute,
                metadata=self._tool_metadata(
                    config,
                    task_id=config.metadata.get("task_id"),
                    cycle=cycle,
                ),
            ),
        )
        if not isinstance(output, QualityBatchOutput):
            output = QualityBatchOutput.model_validate(output)
        objective_results = objective.get("results") if isinstance(objective, Mapping) else None
        if not isinstance(objective_results, Mapping):
            objective_results = {}
        # Candidates are the durable data passed to the population and later
        # post-filter stage. Persist both objective evidence and the Quality
        # Agent decision so post-filter never has to infer or recompute it.
        for candidate in candidates:
            candidate.metadata["developability_evidence"] = dict(
                objective_results.get(candidate.candidate_id) or {}
            )
            decision = output.results.get(candidate.candidate_id)
            if decision is not None:
                candidate.metadata["quality_check"] = decision.model_dump(mode="json")
        return output

    async def reflect_cycle(
        self,
        config: WorkflowConfig,
        cycle: int,
        candidates: list[Candidate],
        best: Candidate | None,
        analysis: AnalyzeAgentOutput,
        quality: QualityBatchOutput | None,
    ) -> ReflectOutput:
        parent = best or (candidates[0] if candidates else None)
        structure_paths = [item.structure_path for item in candidates if item.structure_path]
        prompt = REFLECT_ANALYSIS_PROMPT.format(
            target_name=config.target,
            target_sequence=config.target_sequence,
            target_length=len(config.target_sequence),
            hotspots=config.hotspots,
            quality_check_summary=quality.model_dump() if quality else "unavailable",
            trajectory_summary=config.metadata.get("trajectory_summary", "unavailable"),
            cycle_num=cycle,
            parent_name=parent.candidate_id if parent else "none",
            parent_backend=config.fold_backend,
            parent_sequence=parent.sequence if parent else "",
            parent_iptm=self._metric(parent, "iptm"),
            parent_plddt=self._metric(parent, "plddt"),
            parent_ranking_score=parent.objective if parent else None,
            parent_loglikelihood=self._metric(parent, "loglikelihood"),
            parent_metric_context=parent.metrics if parent else {},
            fold_results_table=json.dumps([item.model_dump() for item in candidates], default=str),
            phase_analyze_summary=analysis.downstream_header,
            candidates_json_path=config.metadata.get("candidates_json_path", "in-memory"),
            objective_key=config.objective_key,
            minimize=config.minimize,
            binder_chain_ids=list(config.binder_chains),
            target_chain_ids=config.target_chain_ids,
            structure_path_catalog=structure_paths,
            epitope_analysis=config.metadata.get("epitope_analysis", "unavailable"),
        )
        profile = AGENT_PROFILES[AgentRole.REFLECTION]
        return await self._session.run(
            profile,
            prompt,
            skills=self._catalog.select(profile.default_skills),
            tool_context=ToolContext(
                compute=self._compute,
                metadata=self._tool_metadata(
                    config,
                    task_id=config.metadata.get("task_id"),
                    cycle=cycle,
                ),
            ),
        )

    async def post_filter_run(
        self,
        config: WorkflowConfig,
        candidates: list[Candidate],
        analysis: AnalyzeAgentOutput,
    ) -> PostFilterAgentOutput:
        recurring_offenders = recurring_offender_ids(config.metadata.get("recurring_offenders"))
        prompt = json.dumps(
            {
                "target": config.target,
                "objective_key": config.objective_key,
                "minimize": config.minimize,
                "top_k": min(config.post_filter_top_k, len(candidates)),
                "analysis": analysis.model_dump(mode="json"),
                "cdr_regions": config.cdr_regions,
                "rank_count": len(candidates),
                "metric_definitions": {
                    "iptm": "Interface confidence, 0-1; higher is generally better, not measured affinity.",
                    "ptm": "Global fold confidence, 0-1; higher is generally better.",
                    "plddt": "Normalized local confidence, 0-1; higher is generally better.",
                    "ranking_score": "Backend composite ranking score; overlaps with other confidence metrics.",
                    "binder_rmsd": "Angstroms, binder CA RMSD after target alignment to the pre-refold pose; lower indicates pose consistency, not affinity.",
                    "ipsae": "Interface confidence derived from aligned error; higher is generally better.",
                    "cdr_contact_fraction": "CDR share of binder contacts; interpret with interface and hotspot evidence.",
                    "framework_contact_fraction": "Framework share of binder contacts; lower is generally preferred.",
                    "loglikelihood": "Sequence-model compatibility; higher is generally better for comparable sequences.",
                    "objective": "Search objective only; direction is given by minimize. Do not let it dictate final order.",
                },
                "candidate_evidence": [
                    self._post_filter_evidence(item, recurring_offenders) for item in candidates
                ],
            },
            ensure_ascii=False,
        )
        profile = AGENT_PROFILES[AgentRole.POST_FILTER]
        return await self._session.run(
            profile,
            prompt,
            skills=self._catalog.select(profile.default_skills),
            tool_context=ToolContext(
                compute=self._compute,
                metadata=self._tool_metadata(
                    config,
                    task_id=config.metadata.get("task_id"),
                    cycle=config.cycles,
                ),
            ),
        )

    @staticmethod
    def _post_filter_evidence(candidate: Candidate, recurring_offenders: list[str]) -> dict[str, Any]:
        metrics = candidate.metrics
        metadata = candidate.metadata
        gate = metadata.get("gate_evidence")
        gate = gate if isinstance(gate, Mapping) else {}
        loss = metadata.get("loss")
        loss = loss if isinstance(loss, Mapping) else {}
        components = loss.get("components") or loss.get("terms") or {}
        if not isinstance(components, Mapping):
            components = {}
        metric_names = (
            "iptm",
            "plddt",
            "ipsae",
            "ranking_score",
            "ptm",
            "binder_rmsd",
            "cdr_contact_fraction",
            "framework_contact_fraction",
            "loglikelihood",
        )
        supplied = {
            name: value if value is not None and math.isfinite(value) else None
            for name in metric_names for value in [metrics.get(name)]
        }
        return {
            "candidate_id": candidate.candidate_id,
            "sequence": candidate.sequence,
            "objective": candidate.objective,
            "objective_components": dict(components),
            "refold_metrics": supplied,
            "binder_rmsd_evidence": metadata.get("binder_rmsd_evidence", {"available": False}),
            "gate_evidence": dict(gate),
            "gate_passed": metadata.get("gate_passed", metrics.get("gate_passed")),
            "chains": metadata.get("chains", {}),
            "interface": {
                "cdr_contact_fraction": metrics.get("cdr_contact_fraction"),
                "framework_contact_fraction": metrics.get("framework_contact_fraction"),
                "contacted_hotspots": gate.get("contacted_hotspots", []),
                "missed_hotspots": gate.get("missed_hotspots", []),
                "cdr3_target_contacts": gate.get("cdr3_total_target_contacts"),
                "recurring_offenders": recurring_offenders,
            },
            "developability": {
                "objective": metadata.get("developability_evidence", {}),
                "quality_agent": metadata.get("quality_check", {}),
            },
            "lineage": {
                "parent_id": metadata.get("parent_id"),
                "skill_id": metadata.get("skill_id"),
            },
            "structure_path": candidate.structure_path,
            "missing_fields": [name for name, value in supplied.items() if value is None],
        }

    @staticmethod
    def _tool_metadata(config: WorkflowConfig, **values: Any) -> dict[str, Any]:
        metadata = {
            "objective_key": config.objective_key,
            "minimize": config.minimize,
            "reflection_interval": config.reflection_interval,
            "binder_chain_ids": list(config.binder_chains),
            "cdr_regions": config.cdr_regions,
            "candidates_json_path": str(
                config.metadata.get("candidates_json_path") or "in-memory"
            ),
            **values,
        }
        if config.llm_model:
            metadata["llm_model"] = config.llm_model
        if config.llm_temperature is not None:
            metadata["llm_temperature"] = config.llm_temperature
        if config.llm_max_tokens is not None:
            metadata["llm_max_tokens"] = config.llm_max_tokens
        return metadata

    @staticmethod
    def _initial_parent(config: WorkflowConfig) -> dict[str, Any]:
        if config.initial_candidates:
            return config.initial_candidates[0]
        return {
            "candidate_id": "initial",
            "chains": config.binder_chains,
            "structure_path": config.initial_structure_path,
        }

    @staticmethod
    def _parent_chains(parent: Mapping[str, Any]) -> dict[str, str]:
        chains = parent.get("chains") or (parent.get("metadata") or {}).get("chains")
        if isinstance(chains, Mapping):
            return {str(key): str(value) for key, value in chains.items()}
        sequence = parent.get("sequence")
        chain_id = str((parent.get("metadata") or {}).get("chain_id", "D"))
        return {chain_id: str(sequence or "")}

    @staticmethod
    def _metric(candidate: Candidate | None, name: str) -> Any:
        return candidate.metrics.get(name) if candidate else None
