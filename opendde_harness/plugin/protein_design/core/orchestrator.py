"""Deterministic protein-design cycle orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from opendde_harness.plugin.protein_design.core.constants import (
    CANONICAL_AMINO_ACIDS,
    is_materialized,
)
from opendde_harness.plugin.protein_design.core.contracts import (
    Candidate,
    FoldRequest,
    PostFilterAgentOutput,
    QualityBatchOutput,
    TaskSnapshot,
    TaskState,
    WorkflowConfig,
)
from opendde_harness.plugin.protein_design.core.design_cases import build_design_case_v2
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.plugin.protein_design.core.population import (
    ConstrainedElitePopulation,
    ParentSampler,
    WorkingParentTracker,
)
from opendde_harness.plugin.protein_design.core.post_mpnn import prepare_post_mpnn
from opendde_harness.plugin.protein_design.core.progress import (
    DesignProgressEvent,
    ProgressEventType,
    ProgressStatus,
    emit_progress,
)
from opendde_harness.plugin.protein_design.core.search_history import (
    build_search_trajectory_summary,
    normalize_search_candidate,
)
from opendde_harness.plugin.protein_design.core.tracing import (
    attach_artifact,
    build_cycle_artifact,
    cycle_span_attributes,
    run_span_attributes,
)
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient
from opendde_harness.tracing import trace

ProgressCallback = Callable[[TaskSnapshot], Awaitable[None] | None]
POST_REFOLD_POOL_MULTIPLIER = 4

logger = logging.getLogger(__name__)


class FoldBatchFailedError(RuntimeError):
    pass


class DesignPhaseRunner(Protocol):
    """The agent phases the orchestrator drives. Returns are Any because the
    concrete output models live in the agents layer, which core must not import."""

    async def analyze_once(self, config: WorkflowConfig) -> Any: ...

    async def select_parent(self, config: WorkflowConfig, cycle: int, candidates: list[Candidate]) -> Candidate: ...

    async def design_cycle(
        self,
        config: WorkflowConfig,
        cycle: int,
        analysis: Any,
        parents: list[dict[str, Any]],
        best: Candidate | None,
    ) -> Any: ...

    async def quality_cycle(
        self,
        config: WorkflowConfig,
        cycle: int,
        candidates: list[Candidate],
        analysis: Any,
    ) -> QualityBatchOutput: ...

    async def reflect_cycle(
        self,
        config: WorkflowConfig,
        cycle: int,
        candidates: list[Candidate],
        best: Candidate | None,
        analysis: Any,
        quality: QualityBatchOutput | None,
    ) -> Any: ...

    async def post_filter_run(
        self,
        config: WorkflowConfig,
        candidates: list[Candidate],
        analysis: Any,
    ) -> PostFilterAgentOutput: ...


@dataclass
class _ProgressState:
    task_id: str
    total_cycles: int
    cycle: int | None = None
    candidate_count: int | None = None
    selected_skill: str | None = None
    pending_phase: tuple[str, float] | None = None


@dataclass
class _DesignSpeculation:
    """One-cycle-ahead Design result prepared while the current batch folds."""

    cycle: int
    parent: dict[str, Any]
    baseline_best_id: str | None
    sampler_state_before: object
    task: asyncio.Task[Any]


@dataclass
class _RunState:
    """Mutable search state carried across cycles of a single run."""

    parents: list[dict[str, Any]]
    snapshot: TaskSnapshot
    cycle: int = 0
    cycle_attempt: int = 0
    best: Candidate | None = None
    pending_speculation: _DesignSpeculation | None = None
    no_improvement_streak: int = 0
    previous_skill_id: str | None = None
    total_scored_candidates: int = 0
    search_history: list[dict[str, Any]] = field(default_factory=list)
    last_candidates: list[Candidate] = field(default_factory=list)
    survivors: list[Candidate] = field(default_factory=list)


class DesignOrchestrator:
    def __init__(
        self,
        compute: ProteinDesignComputeClient,
        memory: DesignMemory,
        phases: DesignPhaseRunner,
        *,
        fold_poll_interval: float = 1.0,
        event_sink: Callable[[DesignProgressEvent], Any] | None = None,
    ) -> None:
        self._compute = compute
        self._memory = memory
        self._fold_poll_interval = fold_poll_interval
        self._phases = phases
        self._event_sink = event_sink
        self._progress_state: _ProgressState | None = None

    async def run(
        self,
        task_id: str,
        config: WorkflowConfig,
        *,
        stop_event: asyncio.Event,
        adjustments: dict[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> TaskSnapshot:
        self._progress_state = _ProgressState(task_id=task_id, total_cycles=config.cycles)
        config.metadata["task_id"] = task_id
        config.metadata["task_id"] = task_id
        run_state = _RunState(
            parents=list(config.initial_candidates),
            snapshot=TaskSnapshot(
                task_id=task_id,
                status=TaskState.RUNNING,
                target=config.target,
                total_cycles=config.cycles,
            ),
        )
        population = ConstrainedElitePopulation(config)
        parent_sampler = ParentSampler(config)
        working_parent = WorkingParentTracker(config)
        recurring_offenders: dict[str, int] = {}
        failed_cycles: list[dict[str, Any]] = []
        known_candidate_ids = {
            str(item.get("candidate_id")) for item in run_state.parents if item.get("candidate_id")
        }
        run_state.snapshot = run_state.snapshot.model_copy(update={"failed_cycles": failed_cycles})
        with trace.span(
            "protein_design.run",
            run_span_attributes(snapshot=run_state.snapshot, config=config),
            kind="protein_design",
        ) as run_span:
            # Publish the run before analysis or folding starts so the tracing
            # dashboard can render the task's cycle-0 empty state immediately.
            run_span.set(
                run_span_attributes(snapshot=run_state.snapshot, config=config, phase="setup")
            ).checkpoint()
            await self._notify(run_state.snapshot, on_progress)
            try:
                self._event("analyze")
                analysis = await self._phases.analyze_once(config)
            except Exception as exc:
                run_state.snapshot = run_state.snapshot.model_copy(
                    update={"status": TaskState.FAILED, "error": str(exc)}
                )
                run_span.set(
                    run_span_attributes(snapshot=run_state.snapshot, config=config)
                ).error(exc).checkpoint()
                await self._notify(run_state.snapshot, on_progress)
                raise
            try:
                await self._bootstrap_initial_fold(
                    run_state,
                    task_id,
                    config,
                    stop_event=stop_event,
                    population=population,
                    working_parent=working_parent,
                )
                while run_state.cycle < config.cycles:
                    running = await self._run_cycle(
                        run_state,
                        task_id,
                        config,
                        analysis=analysis,
                        adjustments=adjustments,
                        population=population,
                        parent_sampler=parent_sampler,
                        working_parent=working_parent,
                        recurring_offenders=recurring_offenders,
                        failed_cycles=failed_cycles,
                        known_candidate_ids=known_candidate_ids,
                        stop_event=stop_event,
                        on_progress=on_progress,
                        run_span=run_span,
                    )
                    if not running:
                        return run_state.snapshot

                terminal, final_selection = await self._terminal_selection(
                    run_state,
                    task_id,
                    config,
                    analysis=analysis,
                    stop_event=stop_event,
                    run_span=run_span,
                )
                self._flush_event()
                run_state.snapshot = run_state.snapshot.model_copy(
                    update={
                        "status": TaskState.COMPLETED,
                        "final_candidates": terminal,
                        "final_selection": final_selection,
                        "selected_skill": None,
                    }
                )
                run_span.set(
                    run_span_attributes(snapshot=run_state.snapshot, config=config)
                ).checkpoint()
                await self._notify(run_state.snapshot, on_progress)
                return run_state.snapshot
            except Exception as exc:
                self._flush_event(failed=True, error=str(exc))
                if stop_event.is_set():
                    run_state.snapshot = run_state.snapshot.model_copy(
                        update={"status": TaskState.STOPPED}
                    )
                    run_span.set(
                        run_span_attributes(snapshot=run_state.snapshot, config=config)
                    ).checkpoint()
                    await self._notify(run_state.snapshot, on_progress)
                    return run_state.snapshot
                run_state.snapshot = run_state.snapshot.model_copy(
                    update={"status": TaskState.FAILED, "error": str(exc)}
                )
                run_span.set(
                    run_span_attributes(snapshot=run_state.snapshot, config=config)
                ).error(exc).checkpoint()
                await self._notify(run_state.snapshot, on_progress)
                raise

    async def _bootstrap_initial_fold(
        self,
        run_state: _RunState,
        task_id: str,
        config: WorkflowConfig,
        *,
        stop_event: asyncio.Event,
        population: ConstrainedElitePopulation,
        working_parent: WorkingParentTracker,
    ) -> None:
        initial_payloads = [
            item for item in config.initial_candidates if is_materialized(item)
        ]
        if initial_payloads:
            self._event("initial_fold")
            initial_scored = await self._fold_cycle(
                task_id,
                config,
                initial_payloads,
                stop_event=stop_event,
            )
            run_state.total_scored_candidates += sum(self._is_scored_candidate(candidate) for candidate in initial_scored)
            materialized = [
                candidate
                for candidate in initial_scored
                if self._is_scored_candidate(candidate) and is_materialized(candidate)
            ]
            for candidate in materialized:
                if not self._passes_gate(candidate):
                    working_parent.consider(candidate)
            initial_admitted = [candidate for candidate in materialized if self._passes_gate(candidate)]
            initial_population = population.update(initial_admitted)
            run_state.best = self._best(initial_population.survivors, config)
            initial_actions = dict(initial_population.actions)
            for candidate in materialized:
                if candidate.candidate_id not in initial_actions:
                    initial_actions[candidate.candidate_id] = "working_parent"
            initial_history = self._history_candidates(
                initial_scored,
                cycle=-1,
                actions=initial_actions,
                admitted_ids={item.candidate_id for item in initial_admitted},
                config=config,
            )
            run_state.search_history.extend(initial_history)
            await self._compute.update_population(
                {
                    "task_id": task_id,
                    "target": config.target,
                    "cycle": -1,
                    "candidates": [item.model_dump() for item in initial_population.survivors],
                    "replace": True,
                    "objective_key": config.objective_key,
                    "minimize": config.minimize,
                    "history_candidates": initial_history,
                }
            )

    async def _run_cycle(
        self,
        run_state: _RunState,
        task_id: str,
        config: WorkflowConfig,
        *,
        analysis: Any,
        adjustments: dict[str, Any],
        population: ConstrainedElitePopulation,
        parent_sampler: ParentSampler,
        working_parent: WorkingParentTracker,
        recurring_offenders: dict[str, int],
        failed_cycles: list[dict[str, Any]],
        known_candidate_ids: set[str],
        stop_event: asyncio.Event,
        on_progress: ProgressCallback | None,
        run_span: Any,
    ) -> bool:
        """Run one search cycle. Returns False when the task was stopped."""

        state = self._require_progress_state()
        state.cycle = run_state.cycle
        state.selected_skill = None
        if stop_event.is_set():
            await self._discard_speculation(
                run_state.pending_speculation,
                parent_sampler,
                restore_sampler=True,
            )
            run_state.pending_speculation = None
            run_state.snapshot = run_state.snapshot.model_copy(
                update={
                    "status": TaskState.STOPPED,
                    "cycle": run_state.cycle,
                    "selected_skill": state.selected_skill,
                }
            )
            run_span.set(run_span_attributes(snapshot=run_state.snapshot, config=config)).checkpoint()
            await self._notify(run_state.snapshot, on_progress)
            return False
        self._apply_adjustments(config, adjustments)
        quality = None
        reflection = None
        cycle_candidates: list[Candidate] = []
        gate_passed: list[Candidate] = []
        admitted: list[Candidate] = []
        cycle_best: Candidate | None = None
        design = None
        global_best_improved = False
        selected_parent_payload: dict[str, Any] | None = None
        previous_best = run_state.best
        no_improvement_streak_before = run_state.no_improvement_streak
        population_size_before = len(population.candidates)
        population_actions: dict[str, str] = {}
        cycle_retry_safe = True
        try:
            speculation = run_state.pending_speculation
            run_state.pending_speculation = None
            if speculation is not None and self._speculation_is_valid(
                speculation,
                cycle=run_state.cycle,
                best=run_state.best,
                population=population.candidates,
                working_parent=working_parent.candidate,
                initial_candidates=config.initial_candidates,
            ):
                run_state.parents = [speculation.parent]
                selected_parent_payload = speculation.parent
                self._event("design_speculation_reuse")
                try:
                    design = await speculation.task
                except Exception:
                    parent_sampler.restore_state(speculation.sampler_state_before)
                    design = None
            elif speculation is not None:
                await self._discard_speculation(
                    speculation,
                    parent_sampler,
                    restore_sampler=True,
                )

            if design is None and population.candidates:
                if config.parent_selection_strategy == "llm":
                    selected = await self._phases.select_parent(
                        config,
                        run_state.cycle,
                        population.candidates,
                    )
                else:
                    selected = parent_sampler.select(
                        population.candidates,
                        run_state.cycle,
                    )
                run_state.parents = [self._candidate_parent_payload(selected, config)]
            elif design is None and working_parent.candidate is not None:
                run_state.parents = [self._candidate_parent_payload(working_parent.candidate, config)]
            elif design is None and not run_state.parents:
                run_state.parents = list(config.initial_candidates)
            if run_state.parents:
                selected_parent_payload = run_state.parents[0]
            config.metadata["population_size"] = len(population.candidates)
            self._event("parent_selection")
            self._flush_event(
                input_payload={
                    "strategy": config.parent_selection_strategy,
                    "population_size": len(population.candidates),
                },
                output_payload={"parents": run_state.parents},
            )
            if design is None:
                self._event("design")
                design = await self._phases.design_cycle(config, run_state.cycle, analysis, run_state.parents, run_state.best)
            state.selected_skill = design.selected_skill_id
            self._event("fold")
            fold_task = asyncio.create_task(
                self._fold_cycle(
                    task_id,
                    config,
                    design.fold_candidates,
                    stop_event=stop_event,
                ),
                name=f"protein-fold-{task_id}-{run_state.cycle}",
            )
            run_state.pending_speculation = self._start_design_speculation(
                config=config,
                cycle=run_state.cycle,
                analysis=analysis,
                best=run_state.best,
                population=population.candidates,
                parent_sampler=parent_sampler,
            )
            try:
                cycle_candidates = await fold_task
                run_state.total_scored_candidates += sum(
                    self._is_scored_candidate(candidate) for candidate in cycle_candidates
                )
                # StructuredSession is intentionally single-owner.
                # If a persistent Fold finishes before the speculative
                # LLM call, join it before Quality/Reflection can use
                # the same session concurrently.
                if run_state.pending_speculation is not None:
                    try:
                        await run_state.pending_speculation.task
                    except Exception as exc:
                        # The next cycle restores sampler state and
                        # falls back to the normal Design path.
                        logger.warning(
                            "design speculation failed for cycle %s; "
                            "falling back to synchronous design: %s",
                            run_state.cycle + 1,
                            exc,
                            exc_info=True,
                        )
            except BaseException:
                await self._discard_speculation(
                    run_state.pending_speculation,
                    parent_sampler,
                    restore_sampler=True,
                )
                run_state.pending_speculation = None
                raise
            state.candidate_count = len(cycle_candidates)
            self._flush_event(
                input_payload={"candidate_count": len(design.fold_candidates)},
                output_payload={
                    "candidate_count": len(cycle_candidates),
                    "candidates": [item.model_dump(mode="json") for item in cycle_candidates],
                },
            )
            self._event("gate")
            materialized = []
            for item in cycle_candidates:
                if not is_materialized(item):
                    self._mark_masked_rejection(item)
                elif not self._is_scored_candidate(item):
                    self._mark_scoring_rejection(item)
                else:
                    materialized.append(item)
            for item in materialized:
                if not self._passes_gate(item):
                    working_parent.consider(item)
            gate_passed = [item for item in materialized if self._passes_gate(item)]
            self._event("quality")
            automatic = [item for item in gate_passed if not self._needs_quality_check(item, config)]
            quality_targets = [item for item in gate_passed if self._needs_quality_check(item, config)]
            if quality_targets:
                quality = await self._phases.quality_cycle(config, run_state.cycle, quality_targets, analysis)
            else:
                quality = QualityBatchOutput(results={})
            admitted = automatic + [
                item
                for item in quality_targets
                if quality.results.get(item.candidate_id) is not None
                and quality.results[item.candidate_id].pass_check
            ]
            # Population mutation is the commit point for a cycle.
            # Retrying after this point could insert the same batch twice.
            cycle_retry_safe = False
            population_result = population.update(admitted)
            population_actions = population_result.actions
            run_state.survivors = population_result.survivors
            cycle_best = self._best(admitted, config)
            run_state.best = self._best(run_state.survivors, config)
            if run_state.best is not None and self._is_better(run_state.best, previous_best, config):
                global_best_improved = True
                run_state.no_improvement_streak = 0
            else:
                global_best_improved = False
                run_state.no_improvement_streak += 1
            self._update_recurring_offenders(
                cycle_candidates,
                recurring_offenders,
            )
            config.metadata["no_improvement_streak"] = run_state.no_improvement_streak
            config.metadata["recurring_offenders"] = dict(recurring_offenders)
            config.metadata["stagnation_guidance"] = (
                "Use the gate evidence and historical skill outcomes to choose "
                "a legal basin-changing proposal."
                if run_state.no_improvement_streak
                else "Continue refining the improved lineage."
            )
            config.metadata["population_actions"] = population_result.actions
            config.metadata["gate_feedback"] = self._search_feedback(
                cycle_candidates,
                working_parent.candidate,
                run_state.survivors,
                config,
                selected_parent_payload,
                recurring_offenders,
            )
            history_batch = self._history_candidates(
                cycle_candidates,
                cycle=run_state.cycle,
                actions=population_result.actions,
                admitted_ids={item.candidate_id for item in admitted},
                config=config,
            )
            history_by_id = {item["candidate_id"]: item for item in run_state.search_history}
            history_by_id.update({item["candidate_id"]: item for item in history_batch})
            run_state.search_history = list(history_by_id.values())
            config.metadata["trajectory_summary"] = json.dumps(
                build_search_trajectory_summary(
                    run_state.search_history,
                    objective_key=config.objective_key,
                    minimize=config.minimize,
                ),
                ensure_ascii=False,
            )
            self._event("population_update")
            await self._compute.update_population(
                {
                    "task_id": task_id,
                    "target": config.target,
                    "cycle": run_state.cycle,
                    "candidates": [item.model_dump() for item in run_state.survivors],
                    "selected_skill_id": design.selected_skill_id,
                    "replace": True,
                    "objective_key": config.objective_key,
                    "minimize": config.minimize,
                    "history_candidates": history_batch,
                }
            )
            run_state.last_candidates = cycle_candidates
        except Exception as exc:
            await self._discard_speculation(
                run_state.pending_speculation,
                parent_sampler,
                restore_sampler=True,
            )
            run_state.pending_speculation = None
            run_span.set(
                {
                    "protein_design.cycle_index": run_state.cycle,
                    "protein_design.cycle.error": str(exc),
                }
            ).checkpoint()
            if cycle_retry_safe and run_state.cycle_attempt < config.cycle_retry_limit:
                run_state.cycle_attempt += 1
                self._event("cycle_retry")
                await self._notify(
                    run_state.snapshot.model_copy(
                        update={
                            "cycle": run_state.cycle,
                            "phase": "cycle_retry",
                            "error": None,
                            "failed_cycles": list(failed_cycles),
                            "selected_skill": state.selected_skill,
                        }
                    ),
                    on_progress,
                )
                return True
            fatal_batch_failure = isinstance(exc, FoldBatchFailedError)
            failure = {
                "cycle": run_state.cycle,
                "attempts": run_state.cycle_attempt + 1,
                "retries": run_state.cycle_attempt,
                "error": str(exc) or type(exc).__name__,
                "population_committed": not cycle_retry_safe,
                "skipped": bool(config.skip_failed_cycles and not fatal_batch_failure),
            }
            failed_cycles.append(failure)
            config.metadata["failed_cycles"] = list(failed_cycles)
            if fatal_batch_failure or not config.skip_failed_cycles:
                raise
            run_span.artifact("protein_design.cycle_failure", failure)
            run_state.snapshot = TaskSnapshot(
                task_id=task_id,
                status=TaskState.RUNNING,
                target=config.target,
                cycle=run_state.cycle + 1,
                total_cycles=config.cycles,
                best_candidate=run_state.best,
                pending_adjustments=dict(adjustments),
                phase="cycle_skipped",
                error=None,
                failed_cycles=list(failed_cycles),
                selected_skill=state.selected_skill,
            )
            await self._notify(run_state.snapshot, on_progress)
            run_state.cycle += 1
            run_state.cycle_attempt = 0
            run_state.last_candidates = []
            return True

        should_reflect = (run_state.cycle + 1) % config.reflection_interval == 0
        if should_reflect:
            self._event("evolution_tree")
            self._event("reflection")
            reflection = None
            reflection_errors: list[str] = []
            for reflection_attempt in range(config.cycle_retry_limit + 1):
                try:
                    reflection = await self._phases.reflect_cycle(
                        config,
                        run_state.cycle,
                        cycle_candidates,
                        run_state.best,
                        analysis,
                        quality,
                    )
                    break
                except Exception as exc:
                    reflection_errors.append(str(exc) or type(exc).__name__)
                    if reflection_attempt < config.cycle_retry_limit:
                        self._event("reflection_retry")
            if reflection_errors and reflection is None:
                config.metadata.setdefault("reflection_failures", []).append(
                    {
                        "cycle": run_state.cycle,
                        "attempts": len(reflection_errors),
                        "retries": max(0, len(reflection_errors) - 1),
                        "error": reflection_errors[-1],
                    }
                )
                run_span.artifact(
                    "protein_design.reflection_failure",
                    config.metadata["reflection_failures"][-1],
                )
            if reflection is not None:
                config.metadata["reflection"] = (
                    reflection.to_design_directives()
                    if hasattr(reflection, "to_design_directives")
                    else str(reflection)
                )
        self._event("memory")
        insights = []
        if reflection is not None:
            insights = [config.metadata.get("reflection", "")]
        selected_skill_id = design.selected_skill_id if design is not None else None
        case_triggers = self._case_triggers(
            should_reflect=should_reflect,
            global_best_improved=global_best_improved,
            previous_skill_id=run_state.previous_skill_id,
            selected_skill_id=selected_skill_id,
            no_improvement_streak=run_state.no_improvement_streak,
            candidates=cycle_candidates,
        )
        if case_triggers:
            await self._memory.record(
                task_id,
                run_state.cycle,
                config.target,
                build_design_case_v2(
                    task_id=task_id,
                    config=config,
                    cycle=run_state.cycle,
                    triggers=case_triggers,
                    candidates=cycle_candidates,
                    population=population.candidates,
                    population_size_before=population_size_before,
                    population_actions=population_actions,
                    admitted_ids={item.candidate_id for item in admitted},
                    selected_parent=selected_parent_payload,
                    previous_global_best=previous_best,
                    best=run_state.best,
                    working_parent=working_parent.candidate,
                    selected_skill_id=selected_skill_id,
                    no_improvement_streak_before=no_improvement_streak_before,
                    no_improvement_streak=run_state.no_improvement_streak,
                ),
                insights,
                finalize=should_reflect,
            )
        run_state.previous_skill_id = selected_skill_id or run_state.previous_skill_id
        self._event("trajectory")
        self._event("visualization")
        with trace.span(
            "protein_design.cycle",
            {
                **cycle_span_attributes(
                    task_id=task_id,
                    cycle=run_state.cycle,
                    candidates=cycle_candidates,
                    cycle_best=cycle_best,
                    global_best=run_state.best,
                ),
                "protein_design.agent.role": "design",
                "protein_design.skill.selected": (design.selected_skill_id if design is not None else None),
                "protein_design.gate.passed": len(gate_passed),
                "protein_design.quality.passed": len(admitted),
                "protein_design.reflection.scheduled": should_reflect,
                "protein_design.memory.retrieved": (len(design.memories) if design is not None else 0),
                "protein_design.memory.skills_retrieved": (
                    len(design.learned_skill_ids) if design is not None else 0
                ),
                "protein_design.memory.skills_applied": (
                    list(design.applied_learned_skill_ids) if design is not None else []
                ),
            },
            kind="protein_design",
        ) as cycle_span:
            structure_artifacts = await self._persist_candidate_structures(
                cycle_span,
                cycle_candidates,
                task_id=task_id,
                target_chain_ids=config.target_chain_ids,
                binder_chain_ids=list(config.binder_chains),
            )
            cycle_span.artifact(
                "protein_design.cycle",
                build_cycle_artifact(
                    task_id=task_id,
                    target=config.target,
                    cycle=run_state.cycle,
                    objective_key=config.objective_key,
                    minimize=config.minimize,
                    candidates=cycle_candidates,
                    cycle_best_id=cycle_best.candidate_id if cycle_best else None,
                    global_best_id=run_state.best.candidate_id if run_state.best else None,
                    known_candidate_ids=known_candidate_ids,
                    structure_artifacts=structure_artifacts,
                    population_candidate_ids={item.candidate_id for item in run_state.survivors},
                    admitted_candidate_ids={item.candidate_id for item in admitted},
                    population_actions=population_actions,
                ),
            )
        known_candidate_ids.update(item.candidate_id for item in cycle_candidates)
        run_state.snapshot = TaskSnapshot(
            task_id=task_id,
            status=TaskState.RUNNING,
            target=config.target,
            cycle=run_state.cycle + 1,
            total_cycles=config.cycles,
            best_candidate=run_state.best,
            pending_adjustments=dict(adjustments),
            failed_cycles=list(failed_cycles),
            selected_skill=state.selected_skill,
        )
        run_span.set(run_span_attributes(snapshot=run_state.snapshot, config=config)).checkpoint()
        await self._notify(run_state.snapshot, on_progress)
        run_state.cycle += 1
        run_state.cycle_attempt = 0
        return True

    async def _terminal_selection(
        self,
        run_state: _RunState,
        task_id: str,
        config: WorkflowConfig,
        *,
        analysis: Any,
        stop_event: asyncio.Event,
        run_span: Any,
    ) -> tuple[list[Candidate], dict[str, Any]]:
        self._event("final_visualization")
        if run_state.total_scored_candidates == 0:
            raise RuntimeError("protein-design search produced no successfully scored candidates")
        terminal = list(run_state.last_candidates)
        post_refold_error: str | None = None
        post_filter_error: str | None = None
        if config.post_filter_enabled:
            self._event("post_refold")
            try:
                raw = self._terminal_refold_pool(run_state.search_history, config)
                if raw:
                    terminal = self._read_candidates({"candidates": raw}, config)
                    terminal = await prepare_post_mpnn(self._compute, terminal, config, task_id, stop_event)
                    foldable = [
                        item.model_dump(mode="json")
                        for item in terminal
                        if item.metadata.get("post_mpnn_selected")
                    ]
                    if not foldable:
                        raise ValueError("SolubleMPNN produced no eligible groups to refold")
                    submission = await self._compute.submit_fold(
                        FoldRequest(
                            task_id=task_id,
                            candidates=foldable,
                            backend="opendde",
                            options={
                                **config.fold_options,
                                "target_chains": (
                                    config.target_chains or config.fold_options.get("target_chains", {})
                                ),
                                "objective_key": config.objective_key,
                                "post_refold": True,
                            },
                            placement=config.placement,
                        )
                    )
                    folded = await self._compute.wait_fold(
                        submission.job_id,
                        poll_interval=self._fold_poll_interval,
                        stop_event=stop_event,
                    )
                    refolded = self._read_candidates(folded.result or {}, config)
                    terminal = self._merge_post_refold_candidates(refolded, terminal)
                    await self._post_refold_pose_evidence(terminal, config, task_id)
                    if not any(self._post_filter_eligible(item) for item in terminal):
                        post_refold_error = folded.error or "post-refold returned no usable candidates"
                else:
                    terminal = []
                    post_refold_error = "trajectory contained no candidates to refold"
            except Exception as exc:
                if stop_event.is_set():
                    raise
                post_refold_error = str(exc) or type(exc).__name__

        if config.post_filter_enabled:
            eligible = [
                candidate
                for candidate in terminal
                if candidate.metadata.get("post_refold_success") and self._post_filter_eligible(candidate)
            ]
        else:
            eligible = [
                candidate
                for candidate in terminal
                if self._is_scored_candidate(candidate) and self._passes_gate(candidate)
            ]
        eligible_ids = {candidate.candidate_id for candidate in eligible}
        rejected = [candidate for candidate in terminal if candidate.candidate_id not in eligible_ids]
        if config.post_filter_enabled and eligible:
            self._event("post_filter")
            try:
                post_filter = await self._phases.post_filter_run(config, eligible, analysis)
                final_selection = self._post_filter_policy_selection(
                    post_filter.model_dump(mode="json"),
                    eligible=eligible,
                    rejected=rejected,
                    terminal=terminal,
                    config=config,
                    post_refold_error=post_refold_error,
                )
            except Exception as exc:
                post_filter_error = str(exc) or type(exc).__name__
                final_selection = self._objective_final_selection(
                    eligible=eligible,
                    rejected=rejected,
                    terminal=terminal,
                    config=config,
                    strategy_summary="PostFilter Agent unavailable; deterministic objective ordering applied.",
                    post_refold_error=post_refold_error,
                    post_filter_error=post_filter_error,
                )
        elif config.post_filter_enabled:
            final_selection = self._failed_post_filter_selection(
                terminal,
                [],
                post_refold_error,
                "No usable post-refold candidates; PostFilter Agent was skipped.",
            )
        else:
            final_selection = self._objective_final_selection(
                eligible=eligible,
                rejected=rejected,
                terminal=terminal,
                config=config,
                strategy_summary="Python objective ordering; post-filter disabled.",
            )
        final_selection.update(
            {
                "post_filter_enabled": bool(config.post_filter_enabled),
                "post_filter_executed": bool(config.post_filter_enabled and eligible),
                "post_filter_top_k": int(config.post_filter_top_k),
                "post_refold_pool_size": len(terminal) if config.post_filter_enabled else 0,
                "candidate_source": "trajectory_mpnn" if config.post_filter_enabled else "terminal",
            }
        )
        if config.post_filter_enabled:
            final_selection["structure_artifacts"] = await self._persist_candidate_structures(
                run_span,
                terminal,
                task_id=task_id,
                target_chain_ids=config.target_chain_ids,
                binder_chain_ids=list(config.binder_chains),
            )
            final_selection["target_chain_ids"] = list(config.target_chain_ids)
            final_selection["binder_chain_ids"] = list(config.binder_chains)
        await self._compute.update_population(
            {
                "task_id": task_id,
                "target": config.target,
                "cycle": config.cycles,
                "candidates": [item.model_dump(mode="json") for item in run_state.survivors],
                "replace": True,
                "objective_key": config.objective_key,
                "minimize": config.minimize,
                "final_selection": final_selection,
            }
        )
        attach_artifact(
            run_span,
            "protein_design.final_selection",
            final_selection,
        )
        return terminal, final_selection

    @staticmethod
    def _terminal_refold_pool(
        search_history: list[dict[str, Any]],
        config: WorkflowConfig,
    ) -> list[dict[str, Any]]:
        """Bound the terminal SolubleMPNN pool: without it one MPNN call runs per historical candidate."""

        unique = {item["candidate_id"]: item for item in search_history}.values()
        retained = [
            record
            for record in unique
            if record.get("gate_passed") is True
            and isinstance(record.get("objective"), float)
            and math.isfinite(record["objective"])
        ]
        retained.sort(key=lambda record: float(record["objective"]), reverse=not config.minimize)
        pool = []
        for record in retained[: POST_REFOLD_POOL_MULTIPLIER * max(int(config.post_filter_top_k), 1)]:
            metadata = {**(record.get("metadata") or {}), "post_refold_success": False}
            for key in ("parent_id", "skill_id", "cycle", "population_action", "chains"):
                if key in record:
                    metadata.setdefault(key, record[key])
            pool.append(
                {
                    key: record[key]
                    for key in ("candidate_id", "sequence", "objective", "metrics", "structure_path")
                    if key in record
                }
                | {"metadata": metadata}
            )
        return pool

    @staticmethod
    def _merge_post_refold_candidates(
        refolded: list[Candidate],
        source: list[Candidate],
    ) -> list[Candidate]:
        source_by_id = {candidate.candidate_id: candidate for candidate in source}
        refolded_by_id: dict[str, Candidate] = {}
        for candidate in refolded:
            if candidate.candidate_id not in source_by_id or candidate.candidate_id in refolded_by_id:
                raise ValueError(f"Unexpected or duplicate refold candidate: {candidate.candidate_id}")
            if candidate.sequence != source_by_id[candidate.candidate_id].sequence:
                raise ValueError(f"Refold changed candidate sequence: {candidate.candidate_id}")
            refolded_by_id[candidate.candidate_id] = candidate
        merged: list[Candidate] = []
        for previous in source:
            candidate = refolded_by_id.get(previous.candidate_id)
            if candidate is None:
                candidate = previous.model_copy(deep=True)
                candidate.metadata.update(
                    success=False,
                    post_refold_success=False,
                    post_refold_error=previous.metadata.get("post_refold_error") or "No refold result returned",
                )
            else:
                fresh_success = candidate.metadata.get("success", candidate.objective is not None)
                candidate.metadata = {
                    **previous.metadata,
                    **candidate.metadata,
                    "success": fresh_success,
                    "post_refold_success": bool(fresh_success),
                    "pre_refold_structure_path": previous.structure_path,
                }
            merged.append(candidate)
        return merged

    async def _post_refold_pose_evidence(
        self,
        candidates: list[Candidate],
        config: WorkflowConfig,
        task_id: str,
    ) -> None:
        for candidate in candidates:
            candidate.metrics.pop("binder_rmsd", None)
            if not candidate.metadata.get("post_refold_success"):
                continue
            try:
                reference = candidate.metadata.get("pre_refold_structure_path")
                if not reference or not candidate.structure_path:
                    raise ValueError("Pre-refold or refold structure unavailable")
                evidence = await self._compute.pose_rmsd(
                    {
                        "task_id": task_id,
                        "reference_path": reference,
                        "mobile_path": candidate.structure_path,
                        "target_chain_ids": config.target_chain_ids,
                        "binder_chain_ids": list(config.binder_chains),
                    }
                )
                rmsd = float(evidence["rmsd"])
                if not math.isfinite(rmsd) or rmsd < 0:
                    raise ValueError("Invalid binder RMSD")
                candidate.metrics["binder_rmsd"] = rmsd
                candidate.metadata["binder_rmsd_evidence"] = {"available": True, **evidence}
            except Exception as exc:
                candidate.metadata["binder_rmsd_evidence"] = {"available": False, "error": str(exc)}

    @staticmethod
    def _post_filter_eligible(candidate: Candidate) -> bool:
        return (
            candidate.metadata.get("success") is not False
            and candidate.objective is not None
            and math.isfinite(float(candidate.objective))
            and bool(candidate.structure_path)
        )

    @classmethod
    def _failed_post_filter_selection(
        cls,
        terminal: list[Candidate],
        eligible: list[Candidate],
        post_refold_error: str | None,
        post_filter_error: str,
    ) -> dict[str, Any]:
        return cls._final_selection_payload(
            strategy_summary="Final selection failed; search candidates are preserved without fallback ranking.",
            decisions=[],
            terminal=terminal,
            eligible=eligible,
            post_refold_error=post_refold_error,
            post_filter_error=post_filter_error,
            mode="failed",
        )

    @staticmethod
    def _hard_rejection_decision(candidate: Candidate, rank: int) -> dict[str, Any]:
        reason = candidate.metadata.get("post_refold_error") or candidate.metadata.get("error")
        return {
            "candidate_id": candidate.candidate_id,
            "rank": rank,
            "pass_filter": False,
            "rationale": f"Hard eligibility rejection: {reason or 'refold failed, objective is non-finite, or structure is missing'}.",
            "strengths": [],
            "risks": [reason or "not_post_filter_eligible"],
            "objective": candidate.objective,
            "hard_eligible": False,
        }

    @classmethod
    def _objective_final_selection(
        cls,
        *,
        eligible: list[Candidate],
        rejected: list[Candidate],
        terminal: list[Candidate],
        config: WorkflowConfig,
        strategy_summary: str,
        post_refold_error: str | None = None,
        post_filter_error: str | None = None,
    ) -> dict[str, Any]:
        ordered = sorted(
            eligible,
            key=lambda item: float(item.objective),
            reverse=not config.minimize,
        )
        decisions = [
            {
                "candidate_id": candidate.candidate_id,
                "rank": rank,
                "pass_filter": rank <= config.post_filter_top_k,
                "rationale": "Deterministic ordering by the configured objective.",
                "strengths": [f"finite {config.objective_key} and hard eligibility passed"],
                "risks": [],
                "objective": candidate.objective,
                "hard_eligible": True,
            }
            for rank, candidate in enumerate(ordered, start=1)
        ]
        decisions.extend(
            cls._hard_rejection_decision(candidate, len(decisions) + index)
            for index, candidate in enumerate(rejected, start=1)
        )
        return cls._final_selection_payload(
            strategy_summary=strategy_summary,
            decisions=decisions,
            terminal=terminal,
            eligible=eligible,
            post_refold_error=post_refold_error,
            post_filter_error=post_filter_error,
            mode="deterministic",
        )

    @classmethod
    def _post_filter_policy_selection(
        cls,
        value: dict[str, Any],
        *,
        eligible: list[Candidate],
        rejected: list[Candidate],
        terminal: list[Candidate],
        config: WorkflowConfig,
        post_refold_error: str | None,
    ) -> dict[str, Any]:
        output = PostFilterAgentOutput.model_validate(value)
        by_id = {candidate.candidate_id: candidate for candidate in eligible}
        decisions = output.decisions
        ids = [item.candidate_id for item in decisions]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError("PostFilter Agent must rank every eligible candidate exactly once")
        if sorted(item.rank for item in decisions) != list(range(1, len(by_id) + 1)):
            raise ValueError("PostFilter Agent ranks must be unique and contiguous from 1")
        ranked = sorted(decisions, key=lambda item: item.rank)
        result = [
            {
                **item.model_dump(),
                "pass_filter": item.rank <= config.post_filter_top_k,
                "objective": by_id[item.candidate_id].objective,
                "hard_eligible": True,
            }
            for item in ranked
        ]
        result.extend(
            cls._hard_rejection_decision(candidate, len(ranked) + index)
            for index, candidate in enumerate(rejected, start=1)
        )
        payload = cls._final_selection_payload(
            strategy_summary=output.strategy_summary,
            decisions=result,
            terminal=terminal,
            eligible=eligible,
            post_refold_error=post_refold_error,
            post_filter_error=None,
            mode="agent",
        )
        payload["risk_notes"] = output.risk_notes
        return payload

    @classmethod
    def _final_selection_payload(
        cls,
        *,
        strategy_summary: str,
        decisions: list[dict[str, Any]],
        terminal: list[Candidate],
        eligible: list[Candidate],
        post_refold_error: str | None,
        post_filter_error: str | None,
        mode: str,
    ) -> dict[str, Any]:
        return {
            "strategy_summary": strategy_summary,
            "decisions": decisions,
            "selected_candidate_ids": [
                str(item["candidate_id"]) for item in decisions if item.get("pass_filter") and item.get("hard_eligible")
            ],
            "mode": mode,
            "post_refold_error": post_refold_error,
            "post_filter_error": post_filter_error,
            "refolded_candidate_count": sum(
                bool(candidate.metadata.get("post_refold_success")) and cls._post_filter_eligible(candidate)
                for candidate in terminal
            ),
            "eligible_candidate_count": len(eligible),
            "candidates": [candidate.model_dump(mode="json") for candidate in terminal],
        }

    def _start_design_speculation(
        self,
        *,
        config: WorkflowConfig,
        cycle: int,
        analysis: Any,
        best: Candidate | None,
        population: list[Candidate],
        parent_sampler: ParentSampler,
    ) -> _DesignSpeculation | None:
        """Prepare cycle N+1 while cycle N folds, without mutating live config."""
        next_cycle = cycle + 1
        if next_cycle >= config.cycles:
            return None
        if (cycle + 1) % config.reflection_interval == 0:
            return None
        if config.parent_selection_strategy == "llm":
            return None
        # Bootstrap and working-parent transitions are intentionally synchronous:
        # their parent can change as soon as the current fold returns.  Speculate
        # only after a scored population/global best makes the lineage stable.
        if best is None or not population:
            return None

        sampler_state = parent_sampler.snapshot_state()
        try:
            selected = parent_sampler.select(population, next_cycle)
            parent = self._candidate_parent_payload(selected, config)

            speculative_config = config.model_copy(deep=True)
            speculative_config.metadata["population_size"] = len(population)
            speculative_best = best.model_copy(deep=True)
            task = asyncio.create_task(
                self._run_speculative_design(
                    speculative_config,
                    next_cycle,
                    analysis,
                    [parent],
                    speculative_best,
                ),
                name=f"protein-design-speculation-{next_cycle}",
            )
        except Exception:
            parent_sampler.restore_state(sampler_state)
            return None
        return _DesignSpeculation(
            cycle=next_cycle,
            parent=parent,
            baseline_best_id=best.candidate_id if best is not None else None,
            sampler_state_before=sampler_state,
            task=task,
        )

    async def _run_speculative_design(
        self,
        config: WorkflowConfig,
        cycle: int,
        analysis: Any,
        parents: list[dict[str, Any]],
        best: Candidate | None,
    ) -> Any:
        """Trace concurrent Design without replacing the foreground Fold phase."""
        started_at = time.perf_counter()
        self._emit_event(
            "design_speculation",
            ProgressStatus.STARTED,
            event_type=ProgressEventType.AGENT,
            summary=f"preparing cycle {cycle} while the current batch folds",
        )
        try:
            result = await self._phases.design_cycle(
                config,
                cycle,
                analysis,
                parents,
                best,
            )
        except BaseException as exc:
            self._emit_event(
                "design_speculation",
                ProgressStatus.FAILED,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                error=str(exc) or type(exc).__name__,
                event_type=ProgressEventType.AGENT,
            )
            raise
        self._emit_event(
            "design_speculation",
            ProgressStatus.COMPLETED,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            event_type=ProgressEventType.AGENT,
            summary=f"cycle {cycle} Design is ready for validation",
        )
        return result

    @classmethod
    def _speculation_is_valid(
        cls,
        speculation: _DesignSpeculation,
        *,
        cycle: int,
        best: Candidate | None,
        population: list[Candidate],
        working_parent: Candidate | None,
        initial_candidates: list[dict[str, Any]],
    ) -> bool:
        if speculation.cycle != cycle:
            return False
        current_best_id = best.candidate_id if best is not None else None
        if current_best_id != speculation.baseline_best_id:
            return False
        expected_parent = cls._parent_identity(speculation.parent)
        available = {
            cls._parent_identity({"candidate_id": item.candidate_id, "sequence": item.sequence}) for item in population
        }
        if working_parent is not None:
            available.add(
                cls._parent_identity(
                    {
                        "candidate_id": working_parent.candidate_id,
                        "sequence": working_parent.sequence,
                    }
                )
            )
        available.update(cls._parent_identity(item) for item in initial_candidates)
        return bool(expected_parent and expected_parent in available)

    @staticmethod
    def _parent_identity(parent: dict[str, Any]) -> str:
        candidate_id = parent.get("candidate_id") or parent.get("id")
        if candidate_id:
            return f"id:{candidate_id}"
        chains = parent.get("chains") or (parent.get("metadata") or {}).get("chains")
        if isinstance(chains, dict) and chains:
            return "chains:" + json.dumps(chains, sort_keys=True, separators=(",", ":"))
        sequence = parent.get("sequence")
        return f"sequence:{sequence}" if sequence else ""

    @staticmethod
    async def _discard_speculation(
        speculation: _DesignSpeculation | None,
        parent_sampler: ParentSampler,
        *,
        restore_sampler: bool,
    ) -> None:
        if speculation is None:
            return
        if restore_sampler:
            parent_sampler.restore_state(speculation.sampler_state_before)
        if speculation.task.done():
            try:
                speculation.task.result()
            except BaseException:
                pass
            return
        speculation.task.cancel()
        try:
            await speculation.task
        except BaseException:
            pass

    def _event(self, name: str) -> None:
        """Finish the previous phase and start ``name`` without blocking the loop."""
        self._flush_event()
        state = self._require_progress_state()
        state.pending_phase = (name, time.perf_counter())
        self._emit_event(name, ProgressStatus.STARTED)

    async def _fold_cycle(
        self,
        task_id: str,
        config: WorkflowConfig,
        candidates: list[dict[str, Any]],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> list[Candidate]:
        submission = await self._compute.submit_fold(
            FoldRequest(
                task_id=task_id,
                candidates=candidates,
                backend="opendde",
                options={
                    **config.fold_options,
                    "target_chains": (config.target_chains or config.fold_options.get("target_chains", {})),
                    "objective_key": config.objective_key,
                },
                placement=config.placement,
            )
        )
        folded = await self._compute.wait_fold(
            submission.job_id,
            poll_interval=self._fold_poll_interval,
            stop_event=stop_event,
        )
        return self._require_scored_fold_batch(
            self._read_candidates(folded.result or {}, config),
            expected_count=len(candidates),
            backend="opendde",
        )

    def _flush_event(
        self,
        *,
        failed: bool = False,
        error: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
    ) -> None:
        state = self._progress_state
        if state is None or state.pending_phase is None:
            return
        name, started_at = state.pending_phase
        state.pending_phase = None
        self._emit_event(
            name,
            ProgressStatus.FAILED if failed else ProgressStatus.COMPLETED,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            error=error,
            input_payload=input_payload,
            output_payload=output_payload,
        )

    def _emit_event(
        self,
        phase: str,
        status: ProgressStatus,
        *,
        duration_ms: float | None = None,
        error: str | None = None,
        event_type: ProgressEventType | None = None,
        summary: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
    ) -> None:
        state = self._require_progress_state()
        resolved_event_type = event_type or {
            "design": ProgressEventType.AGENT,
            "quality": ProgressEventType.AGENT,
            "reflection": ProgressEventType.AGENT,
            "fold": ProgressEventType.FOLD,
            "gate": ProgressEventType.GATE,
            "memory": ProgressEventType.MEMORY,
        }.get(phase, ProgressEventType.PHASE)
        event = DesignProgressEvent.create(
            task_id=state.task_id,
            event_type=resolved_event_type,
            status=status,
            cycle=state.cycle,
            total_cycles=state.total_cycles,
            phase=phase,
            actor="design-agent" if resolved_event_type is ProgressEventType.AGENT else "protein-design",
            skill=state.selected_skill,
            summary=summary or f"{phase} {status.value}",
            duration_ms=duration_ms,
            candidate_count=state.candidate_count,
            error=error,
            input_payload=input_payload,
            output_payload=output_payload,
        )
        emit_progress(self._event_sink, event)

    def _require_progress_state(self) -> _ProgressState:
        state = self._progress_state
        if state is None:
            raise RuntimeError("protein-design progress state is not initialized")
        return state

    @staticmethod
    def _passes_gate(candidate: Candidate) -> bool:
        value = candidate.metadata.get("gate_passed")
        if value is None:
            value = candidate.metrics.get("gate_passed")
        return bool(value) if value is not None else False

    @staticmethod
    def _is_scored_candidate(candidate: Candidate) -> bool:
        success = candidate.metadata.get("success")
        return success is True and candidate.objective is not None and math.isfinite(float(candidate.objective))

    @staticmethod
    def _mark_scoring_rejection(candidate: Candidate) -> None:
        candidate.metadata["gate_passed"] = False
        candidate.metadata["gate_evidence"] = {
            "reason": "fold_or_scoring_failed",
            "error": candidate.metadata.get("error"),
        }
        candidate.metrics["gate_passed"] = 0.0

    @staticmethod
    def _mark_masked_rejection(candidate: Candidate) -> None:
        chains = candidate.metadata.get("chains")
        sequences = chains if isinstance(chains, dict) else {"": candidate.sequence}
        unresolved = {
            str(chain): [
                index
                for index, residue in enumerate(str(sequence).upper())
                if residue not in CANONICAL_AMINO_ACIDS
            ]
            for chain, sequence in sequences.items()
        }
        unresolved = {chain: positions for chain, positions in unresolved.items() if positions}
        candidate.metadata["gate_passed"] = False
        candidate.metadata["gate_evidence"] = {
            "reason": "masked_sequence",
            "unresolved_positions": unresolved,
        }
        candidate.metrics["gate_passed"] = 0.0

    @staticmethod
    def _needs_quality_check(candidate: Candidate, config: WorkflowConfig) -> bool:
        if not config.quality_check_enabled:
            return False
        if candidate.metadata.get("skill_id") == "cdr-full-redesign":
            return False
        iptm = candidate.metrics.get("iptm", candidate.metrics.get("i_ptm", 0.0))
        return float(iptm) >= config.quality_check_threshold

    @staticmethod
    def _candidate_parent_payload(
        candidate: Candidate,
        config: WorkflowConfig,
    ) -> dict[str, Any]:
        payload = candidate.model_dump()
        chains = candidate.metadata.get("chains")
        if not isinstance(chains, dict) or not chains:
            if len(config.binder_chains) <= 1:
                chain_id = (
                    next(iter(config.binder_chains))
                    if config.binder_chains
                    else str(config.fold_options.get("binder_chain", "D"))
                )
                chains = {chain_id: candidate.sequence}
            else:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} is missing per-chain binder sequences; "
                    "refusing to replace a VH/VL child with the initial binder chains"
                )
        payload["chains"] = chains
        return payload

    @staticmethod
    def _search_feedback(
        candidates: list[Candidate],
        working_parent: Candidate | None,
        population: list[Candidate],
        config: WorkflowConfig,
        selected_parent: dict[str, Any] | None,
        recurring_offenders: dict[str, int],
    ) -> str:
        parent_objective = selected_parent.get("objective") if selected_parent is not None else None
        candidate_changes = {}
        for candidate in candidates:
            if candidate.objective is None or parent_objective is None:
                status = "unknown"
                delta = None
            else:
                delta = float(candidate.objective) - float(parent_objective)
                improved = delta < 0 if config.minimize else delta > 0
                status = "improved" if improved else ("unchanged" if delta == 0 else "worsened")
            candidate_changes[candidate.candidate_id] = {
                "objective": candidate.objective,
                "parent_objective": parent_objective,
                "delta": delta,
                "status": status,
            }
        working_evidence = working_parent.metadata.get("gate_evidence", {}) if working_parent is not None else {}
        return json.dumps(
            {
                "objective_key": config.objective_key,
                "minimize": config.minimize,
                "working_parent": (
                    {
                        "candidate": working_parent.model_dump(mode="json"),
                        "violation_count": len(working_evidence.get("framework_contact_residue_ids", [])),
                        "violations": working_evidence.get(
                            "framework_contact_residue_ids",
                            [],
                        ),
                    }
                    if working_parent
                    else None
                ),
                "population_size": len(population),
                "population_best": (population[0].model_dump(mode="json") if population else None),
                "candidate_gate_evidence": {
                    candidate.candidate_id: candidate.metadata.get("gate_evidence", {}) for candidate in candidates
                },
                "candidate_changes": candidate_changes,
                "recurring_offenders": recurring_offenders,
            },
            ensure_ascii=False,
        )

    @classmethod
    def _history_candidates(
        cls,
        candidates: list[Candidate],
        *,
        cycle: int,
        actions: dict[str, str],
        admitted_ids: set[str],
        config: WorkflowConfig,
    ) -> list[dict[str, Any]]:
        result = []
        for candidate in candidates:
            action = actions.get(candidate.candidate_id)
            if action is None:
                if not is_materialized(candidate):
                    action = "masked_sequence_rejected"
                elif not cls._is_scored_candidate(candidate):
                    action = "scoring_failed"
                elif not cls._passes_gate(candidate):
                    reason = (candidate.metadata.get("gate_evidence") or {}).get("reason")
                    action = f"gate_rejected:{reason or 'unknown'}"
                elif candidate.candidate_id not in admitted_ids:
                    action = "quality_rejected"
                else:
                    action = "not_retained"
            result.append(
                normalize_search_candidate(
                    candidate.model_dump(),
                    objective_key=config.objective_key,
                    minimize=config.minimize,
                    cycle=cycle,
                    population_action=action,
                )
                | {"metadata": dict(candidate.metadata)}
            )
        return result

    @staticmethod
    def _update_recurring_offenders(
        candidates: list[Candidate],
        recurring: dict[str, int],
    ) -> None:
        for candidate in candidates:
            evidence = candidate.metadata.get("gate_evidence")
            if not isinstance(evidence, dict):
                continue
            for residue in evidence.get("framework_contact_residue_ids", []):
                if not isinstance(residue, dict):
                    continue
                chain = residue.get("chain")
                position = residue.get("sequence_position")
                if chain is None or position is None:
                    continue
                key = f"{chain}:{position}"
                recurring[key] = recurring.get(key, 0) + 1

    @staticmethod
    def _case_triggers(
        *,
        should_reflect: bool,
        global_best_improved: bool,
        previous_skill_id: str | None,
        selected_skill_id: str | None,
        no_improvement_streak: int,
        candidates: list[Candidate],
    ) -> list[str]:
        triggers = []
        if should_reflect:
            triggers.append("reflection")
        if global_best_improved:
            triggers.append("global_best_improvement")
        if previous_skill_id is not None and selected_skill_id != previous_skill_id:
            triggers.append("workflow_transition")
        if (
            no_improvement_streak
            and selected_skill_id == "cdr-full-redesign"
            and previous_skill_id != "cdr-full-redesign"
        ):
            triggers.append("stagnation_redesign")
        passed = sum(DesignOrchestrator._passes_gate(item) for item in candidates)
        if candidates and 0 < passed < len(candidates):
            triggers.append("mixed_gate_outcome")
        return triggers

    async def _persist_candidate_structures(
        self,
        cycle_span: Any,
        candidates: list[Candidate],
        *,
        task_id: str,
        target_chain_ids: list[str],
        binder_chain_ids: list[str],
    ) -> dict[str, str]:
        structures: dict[str, str] = {}
        for candidate in candidates:
            if not candidate.structure_path:
                continue
            try:
                structure = await self._compute.read_structure(
                    candidate.structure_path,
                    task_id=task_id,
                )
                artifact_path = attach_artifact(
                    cycle_span,
                    "protein_design.structure",
                    {
                        "candidate_id": candidate.candidate_id,
                        "filename": structure.filename,
                        "format": structure.format,
                        "byte_count": structure.byte_count,
                        "target_chain_ids": target_chain_ids,
                        "binder_chain_ids": binder_chain_ids,
                        "text": structure.text,
                    },
                    label=f"candidate-{candidate.candidate_id}",
                )
                if artifact_path:
                    structures[candidate.candidate_id] = artifact_path
            except Exception as exc:
                candidate.metadata["structure_artifact_error"] = str(exc)
        return structures

    @staticmethod
    def _read_candidates(result: dict[str, Any], config: WorkflowConfig) -> list[Candidate]:
        raw = result.get("candidates", [])
        candidates = [Candidate.model_validate(item) for item in raw]
        for candidate in candidates:
            if candidate.objective is None and config.objective_key in candidate.metrics:
                candidate.objective = candidate.metrics[config.objective_key]
            if "success" not in candidate.metadata:
                candidate.metadata["success"] = (
                    candidate.objective is not None
                    and math.isfinite(float(candidate.objective))
                    and not candidate.metadata.get("error")
                )
        return candidates

    @classmethod
    def _require_scored_fold_batch(
        cls,
        candidates: list[Candidate],
        *,
        expected_count: int,
        backend: str,
    ) -> list[Candidate]:
        if any(cls._is_scored_candidate(candidate) for candidate in candidates):
            return candidates
        errors: list[str] = []
        for candidate in candidates:
            error = str(candidate.metadata.get("error") or "").strip()
            if not error:
                continue
            concise = next(
                (line.strip() for line in reversed(error.splitlines()) if line.strip()),
                error,
            )
            if concise not in errors:
                errors.append(concise)
        detail = "; ".join(errors[:3]) or "compute backend returned no scored result"
        raise FoldBatchFailedError(
            f"{backend} fold batch produced 0/{expected_count} successfully scored candidates: {detail}"
        )

    @staticmethod
    def _best(candidates: list[Candidate], config: WorkflowConfig) -> Candidate | None:
        scored = [candidate for candidate in candidates if candidate.objective is not None]
        if not scored:
            return None
        return (
            min(scored, key=lambda item: item.objective)
            if config.minimize
            else max(scored, key=lambda item: item.objective)
        )

    @staticmethod
    def _is_better(candidate: Candidate, current: Candidate | None, config: WorkflowConfig) -> bool:
        if current is None or current.objective is None:
            return True
        if candidate.objective is None:
            return False
        return candidate.objective < current.objective if config.minimize else candidate.objective > current.objective

    @staticmethod
    def _apply_adjustments(config: WorkflowConfig, adjustments: dict[str, Any]) -> None:
        if "num_sequences" in adjustments:
            config.candidates_per_cycle = int(adjustments.pop("num_sequences"))
        if "reflection_interval" in adjustments:
            config.reflection_interval = int(adjustments.pop("reflection_interval"))

    @staticmethod
    async def _notify(snapshot: TaskSnapshot, callback: ProgressCallback | None) -> None:
        if callback is None:
            return
        result = callback(snapshot)
        if asyncio.iscoroutine(result):
            await result
