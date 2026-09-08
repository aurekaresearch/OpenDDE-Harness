"""Unit tests for protein-design orchestration, session budgets and shared predicates."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from opendde_harness.plugin.protein_design.core.constants import (
    CANONICAL_AMINO_ACIDS,
    is_canonical_sequence,
    is_materialized,
)
from opendde_harness.plugin.protein_design.core.contracts import (
    Candidate,
    JobResult,
    JobState,
    JobSubmission,
    PostFilterAgentOutput,
    QualityBatchOutput,
    WorkflowConfig,
)
from opendde_harness.plugin.protein_design.core.memory import DesignMemory
from opendde_harness.plugin.protein_design.core.orchestrator import (
    POST_REFOLD_POOL_MULTIPLIER,
    DesignOrchestrator,
)

BINDER_SEQUENCE = "ACDEFGHIKL"


def make_config(**overrides: Any) -> WorkflowConfig:
    values: dict[str, Any] = {
        "target": "TGT",
        "cycles": 2,
        "candidates_per_cycle": 2,
        "objective_key": "loss",
        "minimize": True,
        "reflection_interval": 100,
        "initial_candidates": [
            {
                "candidate_id": "seed",
                "chains": {"D": BINDER_SEQUENCE},
                "sequence": BINDER_SEQUENCE,
                "metadata": {"chains": {"D": BINDER_SEQUENCE}},
            }
        ],
        "target_sequence": "MKTAYIAKQR",
        "target_chains": {"A": "MKTAYIAKQR"},
        "target_chain_ids": ["A"],
        "binder_chains": {"D": BINDER_SEQUENCE},
        "fixed_residues": {"D": []},
        "cdr_regions": {"D": [0, 1, 2]},
        "mutable_positions": {"D": [0, 1, 2]},
        "population_size": 8,
        "quality_check_enabled": False,
        "post_filter_enabled": True,
        "post_filter_top_k": 1,
    }
    values.update(overrides)
    return WorkflowConfig(**values)


def make_candidate(candidate_id: str, objective: float, *, gate: bool = True) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        sequence=BINDER_SEQUENCE,
        objective=objective,
        metrics={"loss": objective, "gate_passed": 1.0 if gate else 0.0},
        structure_path=f"/tmp/{candidate_id}.cif",
        metadata={
            "chains": {"D": BINDER_SEQUENCE},
            "success": True,
            "gate_passed": gate,
            "post_refold_success": True,
        },
    )


class FakeCompute:
    """Minimal stand-in for ProteinDesignComputeClient."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.population_updates: list[dict[str, Any]] = []
        self.fold_batches: list[list[dict[str, Any]]] = []
        self._counter = 0

    async def submit_fold(self, request: Any) -> JobSubmission:
        self._counter += 1
        job_id = f"job-{self._counter}"
        payload = request.model_dump()
        self.fold_batches.append(payload["candidates"])
        scored = []
        for index, item in enumerate(payload["candidates"]):
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "success": True,
                    "gate_passed": True,
                    "chains": metadata.get("chains") or item.get("chains") or {"D": BINDER_SEQUENCE},
                }
            )
            objective = float(self._counter) + index / 100.0
            scored.append(
                {
                    "candidate_id": item.get("candidate_id") or f"c{self._counter}_{index}",
                    "sequence": item.get("sequence") or BINDER_SEQUENCE,
                    "objective": objective,
                    "metrics": {"loss": objective, "gate_passed": 1.0},
                    "structure_path": f"/tmp/{job_id}_{index}.cif",
                    "metadata": metadata,
                }
            )
        self.jobs[job_id] = {"candidates": scored}
        return JobSubmission(job_id=job_id, status=JobState.QUEUED)

    async def wait_fold(self, job_id: str, **_: Any) -> JobResult:
        return JobResult(job_id=job_id, status=JobState.SUCCEEDED, result=self.jobs[job_id])

    async def update_population(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.population_updates.append(payload)
        return {}

    async def pose_rmsd(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"available": False}

    async def read_structure(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}

    async def generate_soluble_mpnn(self, request: Any) -> dict[str, Any]:
        return {"candidates": []}


class FakePhases:
    def __init__(self) -> None:
        self.post_filter_calls: list[list[Candidate]] = []
        self.post_filter_output: PostFilterAgentOutput | Exception | None = None

    async def analyze_once(self, config: WorkflowConfig) -> dict[str, Any]:
        return {"summary": "analysis"}

    async def select_parent(self, config, cycle, candidates):
        return candidates[0]

    async def design_cycle(self, config, cycle, analysis, parents, best):
        from opendde_harness.plugin.protein_design.agents.phases import DesignCycleResult

        parent = parents[0]
        chains = parent.get("chains") or {"D": BINDER_SEQUENCE}
        fold_candidates = [
            {
                "candidate_id": f"c{cycle}_{index}",
                "sequence": "".join(chains.values()),
                "chains": dict(chains),
                "metadata": {"chains": dict(chains), "parent_id": parent.get("candidate_id")},
            }
            for index in range(config.candidates_per_cycle)
        ]
        return DesignCycleResult(
            fold_candidates=fold_candidates,
            selected_skill_id="cdr-point-mutation",
            memories=[],
        )

    async def quality_cycle(self, config, cycle, candidates, analysis) -> QualityBatchOutput:
        return QualityBatchOutput(results={})

    async def reflect_cycle(self, *args: Any, **kwargs: Any):
        raise AssertionError("reflection is disabled in these tests")

    async def post_filter_run(self, config, candidates, analysis) -> PostFilterAgentOutput:
        self.post_filter_calls.append(list(candidates))
        if isinstance(self.post_filter_output, Exception):
            raise self.post_filter_output
        if self.post_filter_output is not None:
            return self.post_filter_output
        return PostFilterAgentOutput(
            strategy_summary="fake",
            decisions=[
                {"candidate_id": candidate.candidate_id, "rank": rank, "rationale": "ok"}
                for rank, candidate in enumerate(candidates, start=1)
            ],
        )


def make_orchestrator(phases: FakePhases | None = None) -> tuple[DesignOrchestrator, FakeCompute, FakePhases]:
    compute = FakeCompute()
    phases = phases or FakePhases()
    orchestrator = DesignOrchestrator(compute, DesignMemory(None, agent_id="test-agent"), phases, fold_poll_interval=0.0)
    return orchestrator, compute, phases


# --- shared materialization predicate -------------------------------------------------


def test_canonical_alphabet_has_twenty_letters() -> None:
    assert len(CANONICAL_AMINO_ACIDS) == 20
    assert "X" not in CANONICAL_AMINO_ACIDS


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [(BINDER_SEQUENCE, True), (BINDER_SEQUENCE.lower(), True), ("ACDXFG", False), ("", False), (None, False)],
)
def test_is_canonical_sequence(sequence: Any, expected: bool) -> None:
    assert is_canonical_sequence(sequence) is expected


def test_is_materialized_accepts_candidate_and_raw_mapping() -> None:
    candidate = make_candidate("c1", 1.0)
    assert is_materialized(candidate) is True
    assert is_materialized({"chains": {"D": BINDER_SEQUENCE}}) is True
    assert is_materialized({"sequence": BINDER_SEQUENCE}) is True


def test_is_materialized_rejects_masked_sequences() -> None:
    masked = make_candidate("c1", 1.0)
    masked.metadata["chains"] = {"D": "ACDXFGHIKL"}
    assert is_materialized(masked) is False
    assert is_materialized({"chains": {"D": "ACDXFGHIKL"}}) is False
    assert is_materialized({"chains": {}, "sequence": ""}) is False


def test_is_materialized_requires_every_chain() -> None:
    assert is_materialized({"chains": {"D": BINDER_SEQUENCE, "E": "XXXX"}}) is False


# --- terminal refold pool -------------------------------------------------------------


def history_record(candidate_id: str, objective: float | None, gate: bool | None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "parent_id": None,
        "cycle": 0,
        "objective": objective,
        "objective_key": "loss",
        "minimize": True,
        "metrics": {},
        "mutations": [],
        "skill_id": "cdr-point-mutation",
        "gate_passed": gate,
        "gate_reason": None,
        "population_action": "retained",
        "sequence": BINDER_SEQUENCE,
        "chains": {"D": BINDER_SEQUENCE},
        "structure_path": f"/tmp/{candidate_id}.cif",
    }


def test_terminal_refold_pool_drops_unscored_and_gate_failures() -> None:
    config = make_config(post_filter_top_k=10)
    history = [
        history_record("kept", 1.0, True),
        history_record("gate_failed", 0.5, False),
        history_record("gate_unknown", 0.5, None),
        history_record("unscored", None, True),
    ]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    assert [item["candidate_id"] for item in pool] == ["kept"]


def test_terminal_refold_pool_is_capped_and_ordered_best_first() -> None:
    config = make_config(post_filter_top_k=2)
    history = [history_record(f"c{index}", float(100 - index), True) for index in range(50)]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    assert len(pool) == POST_REFOLD_POOL_MULTIPLIER * config.post_filter_top_k
    objectives = [item["objective"] for item in pool]
    assert objectives == sorted(objectives)
    assert objectives[0] == 51.0


def test_terminal_refold_pool_orders_by_objective_when_maximizing() -> None:
    config = make_config(post_filter_top_k=2, minimize=False)
    history = [history_record(f"c{index}", float(index), True) for index in range(50)]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    objectives = [item["objective"] for item in pool]
    assert objectives == sorted(objectives, reverse=True)


def test_terminal_refold_pool_deduplicates_by_candidate_id() -> None:
    config = make_config(post_filter_top_k=10)
    history = [history_record("dup", 1.0, True), history_record("dup", 2.0, True)]
    pool = DesignOrchestrator._terminal_refold_pool(history, config)
    assert len(pool) == 1
    assert pool[0]["objective"] == 2.0


def test_terminal_refold_pool_marks_children_unrefolded() -> None:
    config = make_config(post_filter_top_k=1)
    pool = DesignOrchestrator._terminal_refold_pool([history_record("kept", 1.0, True)], config)
    assert pool[0]["metadata"]["post_refold_success"] is False
    assert pool[0]["metadata"]["chains"] == {"D": BINDER_SEQUENCE}


# --- PostFilter fallback --------------------------------------------------------------


def objective_fallback(eligible: list[Candidate], rejected: list[Candidate]) -> dict[str, Any]:
    return DesignOrchestrator._objective_final_selection(
        eligible=eligible,
        rejected=rejected,
        terminal=eligible + rejected,
        config=make_config(post_filter_top_k=1),
        strategy_summary="fallback",
        post_filter_error="PostFilter Agent must rank every eligible candidate exactly once",
    )


def test_objective_fallback_selects_instead_of_returning_empty() -> None:
    eligible = [make_candidate("worse", 2.0), make_candidate("better", 1.0)]
    payload = objective_fallback(eligible, [])
    assert payload["selected_candidate_ids"] == ["better"]
    assert payload["mode"] == "deterministic"
    assert payload["post_filter_error"]


def test_objective_fallback_ranks_rejected_candidates_last() -> None:
    eligible = [make_candidate("ok", 1.0)]
    rejected = [make_candidate("bad", 9.0, gate=False)]
    payload = objective_fallback(eligible, rejected)
    ranks = {item["candidate_id"]: item["rank"] for item in payload["decisions"]}
    assert ranks["ok"] < ranks["bad"]
    assert payload["selected_candidate_ids"] == ["ok"]


def test_failed_post_filter_selection_still_yields_nothing() -> None:
    """The empty payload stays reachable only when no candidate is eligible at all."""

    terminal = [make_candidate("bad", 9.0, gate=False)]
    payload = DesignOrchestrator._failed_post_filter_selection(terminal, [], None, "skipped")
    assert payload["selected_candidate_ids"] == []
    assert payload["mode"] == "failed"


def test_run_falls_back_to_objective_ordering_when_ranking_is_invalid() -> None:
    phases = FakePhases()
    phases.post_filter_output = PostFilterAgentOutput(
        strategy_summary="broken",
        decisions=[{"candidate_id": "does-not-exist", "rank": 1, "rationale": "wrong"}],
    )
    orchestrator, _compute, _ = make_orchestrator(phases)
    snapshot = asyncio.run(
        orchestrator.run(
            "task-fallback",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert snapshot.final_selection is not None
    assert snapshot.final_selection["selected_candidate_ids"]


# --- end-to-end run -------------------------------------------------------------------


def test_run_completes_and_reports_a_selection() -> None:
    orchestrator, compute, _ = make_orchestrator()
    snapshot = asyncio.run(
        orchestrator.run(
            "task-1",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert snapshot.status.value == "completed"
    assert snapshot.best_candidate is not None
    assert snapshot.final_selection is not None
    assert snapshot.final_selection["selected_candidate_ids"]
    assert compute.population_updates


def test_cycle_snapshots_carry_the_selected_skill() -> None:
    """The TUI renders TaskSnapshot.selected_skill, so the orchestrator must write it."""

    emitted: list[Any] = []
    orchestrator, _compute, _ = make_orchestrator()
    snapshot = asyncio.run(
        orchestrator.run(
            "task-skill",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
            on_progress=lambda item: emitted.append(item),
        )
    )
    running = [item for item in emitted if item.status.value == "running" and item.cycle > 0]
    assert running, "expected at least one in-cycle snapshot"
    assert all(item.selected_skill == "cdr-point-mutation" for item in running)
    assert snapshot.selected_skill is None


def test_run_stops_at_the_next_cycle_boundary() -> None:
    orchestrator, _compute, _ = make_orchestrator()
    stop_event = asyncio.Event()
    stop_event.set()
    snapshot = asyncio.run(
        orchestrator.run(
            "task-stop",
            make_config(post_filter_enabled=False),
            stop_event=stop_event,
            adjustments={},
        )
    )
    assert snapshot.status.value == "stopped"


def test_run_emits_progress_events_for_each_phase() -> None:
    events: list[str] = []
    compute = FakeCompute()
    orchestrator = DesignOrchestrator(
        compute,
        DesignMemory(None, agent_id="test-agent"),
        FakePhases(),
        fold_poll_interval=0.0,
        event_sink=lambda event: events.append(event.phase),
    )
    asyncio.run(
        orchestrator.run(
            "task-events",
            make_config(post_filter_enabled=False),
            stop_event=asyncio.Event(),
            adjustments={},
        )
    )
    assert "analyze" in events
    assert "initial_fold" in events
    assert "final_visualization" in events


# --- session tool-turn budget ---------------------------------------------------------


class ToolCall:
    def __init__(self, name: str, arguments: dict[str, Any], call_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.id = call_id

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {"id": self.id, "function": {"name": self.name, "arguments": "{}"}}


class Response:
    def __init__(self, content: str = "", tool_calls: list[ToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = "stop"


class SkillLoopProvider:
    """A provider that answers every turn with another use_skill call."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat_with_retry(self, **_: Any) -> Response:
        self.calls += 1
        return Response(tool_calls=[ToolCall("use_skill", {"skill_id": f"ghost-{self.calls}"}, f"t{self.calls}")])


def test_structured_session_terminates_on_a_persistent_use_skill_caller() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    provider = SkillLoopProvider()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=3)
    skills = (SkillDocument("only-skill", Path("/tmp/only-skill"), "content", ()),)

    with pytest.raises(RuntimeError, match="use_skill budget"):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert provider.calls == len(skills) + session._max_attempts + 1


def test_structured_session_budget_scales_with_available_skills() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    provider = SkillLoopProvider()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=2)
    skills = tuple(
        SkillDocument(f"skill-{index}", Path(f"/tmp/skill-{index}"), "content", ()) for index in range(3)
    )

    with pytest.raises(RuntimeError, match="use_skill budget"):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert provider.calls == len(skills) + 2 + 1


def test_structured_session_emits_one_input_payload_per_run() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    payloads: list[Any] = []

    class Sink:
        def __call__(self, event: Any) -> None:
            if event.event_type.value == "agent":
                payloads.append(event.input_payload)

    provider = SkillLoopProvider()
    session = OpenDDEHarnessStructuredSession(
        provider, "fake-model", max_attempts=1, progress_sink=Sink()
    )
    renders = 0
    original = session._system_message

    def counting_system_message(*args: Any, **kwargs: Any) -> str:
        nonlocal renders
        renders += 1
        return original(*args, **kwargs)

    session._system_message = counting_system_message
    skills = (SkillDocument("only-skill", Path("/tmp/only-skill"), "content", ()),)
    with pytest.raises(RuntimeError):
        asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert renders == 2


def test_structured_session_widens_the_budget_when_the_model_runs_out_of_tokens() -> None:
    import json as json_module

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession

    class TruncatedThenAnswer:
        def __init__(self) -> None:
            self.budgets: list[int] = []

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.budgets.append(int(kwargs["max_tokens"]))
            if len(self.budgets) == 1:
                response = Response(content="")
                response.finish_reason = "length"
                return response
            return Response(
                content=json_module.dumps({"downstream_header": "epitope", "report": "ok"})
            )

    provider = TruncatedThenAnswer()
    session = OpenDDEHarnessStructuredSession(provider, "fake-model", max_attempts=2)
    result = asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt"))

    assert result.report == "ok"
    assert provider.budgets == [8192, 16384]


# --- config loading -------------------------------------------------------------------


def test_bundled_example_configs_still_normalize() -> None:
    from pathlib import Path

    from opendde_harness.plugin.protein_design.core.runtime import WorkflowConfigLoader

    for name in ("crlf2_quickstart.yaml", "cacng1_quickstart.yaml"):
        path = Path("docs/examples") / name
        if not path.is_file():
            pytest.skip(f"{path} is not present")
        config = WorkflowConfigLoader.config_from_path(str(path))
        assert config.target
        assert config.binder_chains
        assert math.isfinite(float(config.seed))


# --- optional external services ---------------------------------------------------------


class ProtrekOutageCompute(FakeCompute):
    """A worker whose ProTrek call fails the way an egress-less container fails."""

    def __init__(self) -> None:
        super().__init__()
        self.protrek_calls = 0

    async def search_protrek_sequence(self, request: Any) -> Any:
        import httpx

        self.protrek_calls += 1
        raise httpx.ConnectTimeout("timed out")


def _analyze_session(compute: Any) -> tuple[Any, Any]:
    import json as json_module

    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.tools.agent import ProteinDesignToolRegistry

    class ProtrekThenAnswer:
        def __init__(self) -> None:
            self.calls = 0
            self.tool_messages: list[str] = []

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.calls += 1
            self.tool_messages.extend(
                str(message.get("content"))
                for message in kwargs.get("messages", [])
                if message.get("role") == "tool"
            )
            if self.calls == 1:
                return Response(
                    tool_calls=[ToolCall("protrek_sequence_search", {"sequence": BINDER_SEQUENCE}, "t1")]
                )
            return Response(
                content=json_module.dumps({"downstream_header": "epitope", "report": "no homologs"})
            )

    provider = ProtrekThenAnswer()
    session = OpenDDEHarnessStructuredSession(
        provider,
        "fake-model",
        tool_registry=ProteinDesignToolRegistry.for_compute(compute),
        max_attempts=3,
    )
    return provider, session


def test_run_completes_when_the_analysis_agent_hits_a_protrek_outage() -> None:
    from opendde_harness.plugin.protein_design.core.memory import DesignMemory

    compute = ProtrekOutageCompute()

    class ProtrekAnalysisPhases(FakePhases):
        async def analyze_once(self, config: WorkflowConfig) -> dict[str, Any]:
            _provider, session = _analyze_session(compute)
            from opendde_harness.plugin.protein_design.agents.profiles import (
                AGENT_PROFILES,
                AgentRole,
            )

            output = await session.run(AGENT_PROFILES[AgentRole.ANALYZE], "analyze the target")
            return {"summary": output.downstream_header}

    config = make_config(post_filter_enabled=False)
    orchestrator = DesignOrchestrator(
        compute,
        DesignMemory(None, agent_id="test-agent"),
        ProtrekAnalysisPhases(),
        fold_poll_interval=0.0,
    )
    snapshot = asyncio.run(
        orchestrator.run("task-protrek", config, stop_event=asyncio.Event(), adjustments={})
    )

    assert snapshot.status.value == "completed"
    assert not snapshot.failed_cycles
    assert not config.metadata.get("failed_cycles")
    assert compute.protrek_calls >= 1


def test_design_agent_returns_its_reasoning_with_the_tool_turn() -> None:
    """DeepSeek in thinking mode rejects a tool turn that comes back without its reasoning."""
    import json as json_module
    from pathlib import Path

    from opendde_harness.plugin.protein_design.agents.profiles import AGENT_PROFILES, AgentRole
    from opendde_harness.plugin.protein_design.agents.session import OpenDDEHarnessStructuredSession
    from opendde_harness.plugin.protein_design.agents.skills import SkillDocument

    sent: list[list[dict[str, Any]]] = []

    class ThinkingProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_with_retry(self, **kwargs: Any) -> Response:
            self.calls += 1
            sent.append([dict(message) for message in kwargs["messages"]])
            if self.calls == 1:
                first = Response(tool_calls=[ToolCall("use_skill", {"skill_id": "only-skill"}, "t1")])
                first.reasoning_content = "deciding which skill to load"
                return first
            answer = Response(content=json_module.dumps({"downstream_header": "epitope", "report": "ok"}))
            answer.reasoning_content = "writing the report"
            return answer

    session = OpenDDEHarnessStructuredSession(ThinkingProvider(), "fake-model", max_attempts=2)
    skills = (SkillDocument("only-skill", Path("/tmp/only-skill"), "content", ()),)

    result = asyncio.run(session.run(AGENT_PROFILES[AgentRole.ANALYZE], "prompt", skills=skills))

    assert result.report == "ok"
    assistant = [message for message in sent[-1] if message["role"] == "assistant"]
    assert assistant and assistant[0]["reasoning_content"] == "deciding which skill to load"
