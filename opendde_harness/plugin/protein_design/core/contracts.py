"""Stable contracts shared by the protein-design client, service, and tools."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    # Scientific runs must fail on misspelled or stale fields rather than
    # silently running with a default configuration.
    model_config = ConfigDict(extra="forbid")


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Placement(ContractModel):
    """Requested GPU placement for one task; every unset field is chosen automatically."""

    fold: list[int] | None = None
    esm: int | None = Field(default=None, ge=0)
    mpnn: int | None = Field(default=None, ge=0)
    cp_degree: int = Field(default=1, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _default_cp_degree(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and data.get("fold") and data.get("cp_degree") is None:
            return {**data, "cp_degree": len(data["fold"])}
        return data

    @model_validator(mode="after")
    def _check_fold_devices(self) -> "Placement":
        if self.fold is None:
            return self
        if not self.fold:
            raise ValueError("compute.placement.fold must list at least one GPU index")
        if any(index < 0 for index in self.fold):
            raise ValueError("compute.placement.fold GPU indices must be zero or greater")
        if len(set(self.fold)) != len(self.fold):
            raise ValueError("compute.placement.fold must not repeat a GPU index")
        if len(self.fold) != self.cp_degree:
            raise ValueError(
                "compute.placement.fold must list exactly cp_degree GPUs; "
                f"got {len(self.fold)} for cp_degree {self.cp_degree}"
            )
        return self


class HealthResponse(ContractModel):
    # Health is a diagnostic payload from a possibly newer service; unknown
    # keys must not make a healthy worker look unavailable.
    model_config = ConfigDict(extra="ignore")

    status: str = "ok"
    models: dict[str, bool] = Field(default_factory=dict)
    workers: dict[str, Any] = Field(default_factory=dict)
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    environment: dict[str, Any] | None = None
    code_version: str | None = None


class FoldRequest(ContractModel):
    task_id: str | None = None
    candidates: list[dict[str, Any]] = Field(min_length=1)
    backend: Literal["opendde"] = "opendde"
    options: dict[str, Any] = Field(default_factory=dict)
    placement: Placement | None = None


class TargetMsaSearchRequest(ContractModel):
    """One target-chain MSA search executed by the selected compute worker."""

    entity_role: Literal["target"] = "target"
    target_name: str = Field(min_length=1)
    chain_id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    force: bool = False
    required: bool = True


class TargetMsaSearchResponse(ContractModel):
    target_name: str
    chain_id: str
    sequence_sha256: str = ""
    unpaired_msa_path: str = ""
    paired_msa_path: str = ""
    alignment_depth: int = Field(default=0, ge=0)
    cached: bool = False
    server_url: str
    server_mode: str
    available: bool = True
    reason: str | None = None

    @model_validator(mode="after")
    def _available_results_are_complete(self) -> "TargetMsaSearchResponse":
        if self.available and not (
            self.sequence_sha256 and self.unpaired_msa_path and self.paired_msa_path and self.alignment_depth >= 1
        ):
            raise ValueError("an available target MSA result must carry both A3M paths and a positive depth")
        return self


class JobSubmission(ContractModel):
    job_id: str
    status: JobState = JobState.QUEUED


class JobResult(ContractModel):
    job_id: str
    status: JobState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    result: dict[str, Any] | None = None
    error: str | None = None


class ShutdownRequest(ContractModel):
    """Host-side request to retire the compute worker and release its GPUs."""

    if_idle: bool = True


class EsmScoreRequest(ContractModel):
    sequences: list[str] = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)
    placement: Placement | None = None


class EsmScoreResponse(ContractModel):
    scores: list[float]


class SolubleMPNNRequest(ContractModel):
    task_id: str | None = None
    structure_path: str
    mutable_positions: list[str] = Field(default_factory=list)
    parent_chains: dict[str, str] = Field(default_factory=dict)
    anchor_mutations: list[dict[str, Any]] = Field(default_factory=list)
    num_sequences: int = Field(default=8, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    placement: Placement | None = None


class StructureReadResponse(ContractModel):
    path: str
    filename: str
    format: str
    byte_count: int = Field(ge=0)
    text: str


class Candidate(ContractModel):
    candidate_id: str
    sequence: str
    objective: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    structure_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentSelectionDecision(ContractModel):
    selected_parent_name: str
    selection_mode: Literal["exploit", "explore"]
    confidence: Literal["low", "medium", "high"]
    rationale: str


class ParentSelectionOutput(ContractModel):
    parent_selection: ParentSelectionDecision


class CycleDesignStage(ContractModel):
    """Overrides for an inclusive, zero-based interval of design cycles."""

    start_cycle: int = Field(ge=0, strict=True)
    end_cycle: int = Field(ge=0, strict=True)
    num_sequences: int | None = Field(default=None, ge=1, strict=True)
    population_size: int | None = Field(default=None, ge=1, strict=True)
    router_skill_probabilities: dict[str, float] | None = None
    router_selection_strategy: Literal["agent", "weighted"] | None = None
    parent_fitness_temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            if isinstance(value.get("parent_fitness_temperature"), bool):
                raise ValueError("parent_fitness_temperature must be a positive finite number")
            weights = value.get("router_skill_probabilities")
            if isinstance(weights, Mapping) and any(isinstance(item, bool) for item in weights.values()):
                raise ValueError("router_skill_probabilities must contain numeric weights, not booleans")
        return value

    @model_validator(mode="after")
    def validate_stage(self) -> "CycleDesignStage":
        if self.end_cycle < self.start_cycle:
            raise ValueError("end_cycle must be greater than or equal to start_cycle")
        if all(
            getattr(self, key) is None
            for key in (
                "num_sequences",
                "population_size",
                "router_skill_probabilities",
                "parent_fitness_temperature",
                "router_selection_strategy",
            )
        ):
            raise ValueError("a cycle_schedule stage must override at least one design parameter")
        weights = self.router_skill_probabilities
        if weights is not None:
            supported = {"cdr-point-mutation", "cdr-full-redesign", "antibody-inverse-folding", "esm2-guided-mutation"}
            unknown = sorted(set(weights) - supported)
            if unknown:
                raise ValueError("unknown design skill weight(s): " + ", ".join(unknown))
            if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
                raise ValueError(
                    "router_skill_probabilities must contain finite non-negative weights with a positive total"
                )
        return self


class WorkflowConfig(ContractModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    # Compute placement is resolved once, when the task starts.  Keeping the
    # resolved endpoint in the immutable run config makes every later cycle,
    # trace, and population read use the same worker.
    compute_url: str | None = None
    compute_worker_id: str | None = None
    compute_profile: str | None = None
    placement: Placement = Field(default_factory=Placement)
    cycles: int = Field(default=3, ge=1)
    candidates_per_cycle: int = Field(default=8, ge=1)
    fold_backend: Literal["opendde"] = "opendde"
    objective_key: Literal["loss", "iptm"] = "loss"
    minimize: bool = True
    reflection_interval: int = Field(default=20, ge=1)
    cycle_retry_limit: int = Field(default=2, ge=0, le=5)
    skip_failed_cycles: bool = True
    initial_candidates: list[dict[str, Any]] = Field(default_factory=list)
    fold_options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    target_sequence: str = ""
    target_chains: dict[str, Any] = Field(default_factory=dict)
    target_chain_ids: list[str] = Field(default_factory=lambda: ["A"])
    hotspots: list[Any] = Field(default_factory=list)
    binder_chains: dict[str, str] = Field(default_factory=dict)
    fixed_residues: dict[str, list[int]] = Field(default_factory=dict)
    explicit_fixed_residues: dict[str, list[int]] = Field(default_factory=dict)
    designable_residues: dict[str, list[int]] = Field(default_factory=dict)
    cdr_regions: dict[str, list[int]] = Field(default_factory=dict)
    cdr_region_groups: dict[str, list[list[int]]] = Field(default_factory=dict)
    mutable_positions: dict[str, list[int]] = Field(default_factory=dict)
    initial_structure_path: str | None = None
    seed: int = 42
    population_size: int = Field(default=20, ge=1)
    cycle_schedule: list[CycleDesignStage] = Field(default_factory=list)
    parent_fitness_temperature: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    constrained_min_cdr_distance: float = Field(default=0.05, ge=0.0, le=1.0)
    constrained_max_position_reuse_fraction: float = Field(default=0.75, gt=0.0, le=1.0)
    constrained_max_mutation_reuse_fraction: float = Field(default=0.30, gt=0.0, le=1.0)
    parent_selection_strategy: Literal["uniform", "greedy", "fitness", "llm"] = "fitness"
    parent_fitness_temperature_start: float = Field(default=1.0, gt=0.0)
    parent_fitness_temperature_end: float = Field(default=0.2, gt=0.0)
    parent_fitness_uniform_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    quality_check_enabled: bool = True
    quality_check_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = Field(default=None, ge=1)
    bootstrap_full_redesign_cycles: int = Field(default=0, ge=0)
    stagnation_full_redesign_threshold: int = Field(default=0, ge=0)
    skill_weights: dict[str, float] | None = None
    router_selection_strategy: Literal["agent", "weighted"] = "agent"
    esm2_available: bool = True
    mutation_count_instruction: str = "Use the configured bounded CDR mutation count."
    post_filter_enabled: bool = False
    post_filter_top_k: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_cycle_schedule(self) -> "WorkflowConfig":
        previous_end = -1
        for stage in self.cycle_schedule:
            if stage.start_cycle <= previous_end:
                raise ValueError("design.cycle_schedule intervals must be ordered and must not overlap")
            if stage.end_cycle >= self.cycles:
                raise ValueError("design.cycle_schedule end_cycle must be less than design.n_cycles")
            previous_end = stage.end_cycle
        return self


def normalize_workflow_adjustments(params: Mapping[str, Any]) -> dict[str, int]:
    """Validate the small set of safe, cycle-boundary runtime adjustments."""
    allowed = {"num_sequences", "reflection_interval"}
    unknown = sorted(set(map(str, params)) - allowed)
    if unknown:
        raise ValueError("unknown protein-design adjustment field(s): " + ", ".join(unknown))
    if not params:
        raise ValueError("protein-design adjustments must not be empty")
    normalized: dict[str, int] = {}
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"protein-design adjustment {key} must be a positive integer")
        normalized[str(key)] = value
    return normalized


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskSnapshot(ContractModel):
    task_id: str
    status: TaskState
    target: str
    compute_url: str | None = None
    compute_worker_id: str | None = None
    cycle: int = 0
    total_cycles: int
    best_candidate: Candidate | None = None
    error: str | None = None
    pending_adjustments: dict[str, Any] = Field(default_factory=dict)
    phase: str = ""
    selected_skill: str | None = None
    final_candidates: list[Candidate] = Field(default_factory=list)
    final_selection: dict[str, Any] | None = None
    failed_cycles: list[dict[str, Any]] = Field(default_factory=list)


class AnalyzeAgentOutput(ContractModel):
    downstream_header: str
    report: str = ""
    evidence_provenance: list[str] = Field(default_factory=list)


class Mutation(ContractModel):
    """One zero-based CDR edit in OpenDDE Harness's proposal contract."""

    chain_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    from_aa: str | None = None
    to_aa: str = Field(min_length=1)


class CandidateProposal(ContractModel):
    candidate_id: str
    parent_id: str
    chains: dict[str, str]
    mutations: list[Mutation] = Field(default_factory=list)
    strategy: str = ""
    risk_level: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_backend: str = "llm"
    bypass_auxiliary_filters: bool = False


class DesignAgentOutput(ContractModel):
    skill_id: str
    selection_reason: str = ""
    applied_learned_skill_ids: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class QualityCandidateResult(ContractModel):
    expressivity: str = "Unknown"
    immunogenicity: str = "Unknown"
    aggregation: str = "Unknown"
    solubility: str = "Unknown"
    specificity: str = "Unknown"
    liability: str = "Unknown"
    reasoning: str
    overall_risk_level: str
    pass_check: bool


class QualityBatchOutput(ContractModel):
    results: dict[str, QualityCandidateResult]


class PostFilterDecision(ContractModel):
    candidate_id: str = Field(min_length=1)
    rank: int = Field(ge=1, strict=True)
    rationale: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PostFilterAgentOutput(ContractModel):
    strategy_summary: str = Field(min_length=1)
    decisions: list[PostFilterDecision] = Field(min_length=1)
    risk_notes: list[str] = Field(default_factory=list, max_length=8)


class AnalysisResponse(ContractModel):
    available: bool = True
    result: Any = None
    error: str | None = None
    service: str | None = None
    reason: str | None = None
    endpoint: str | None = None


class EvolutionTreeRequest(ContractModel):
    task_id: str | None = None
    candidates_json_path: str
    current_parent_id: str | None = None
    objective_key: Literal["loss", "iptm"] = "loss"
    minimize: bool = True
    cycle: int = Field(default=0, ge=0)
    binder_chain_ids: list[str] = Field(default_factory=list, max_length=8)
    cdr_regions: dict[str, list[int]] = Field(default_factory=dict)
    reflection_interval: int = Field(default=20, ge=1)


class HotspotCoordinate(ContractModel):
    chain: str = Field(min_length=1, max_length=8)
    position: int = Field(ge=0)


class EpitopeAnalysisRequest(ContractModel):
    task_id: str | None = None
    structure_path: str
    antibody_chains: list[str] = Field(min_length=1, max_length=8)
    antigen_chains: list[str] = Field(min_length=1, max_length=8)
    cdr_regions: dict[str, dict[str, list[int]]]
    cutoff: float = Field(default=4.5, gt=0.0, le=20.0)
    hotspots: list[HotspotCoordinate] = Field(default_factory=list, max_length=256)


class StructureAnalysisRequest(ContractModel):
    task_id: str | None = None
    structure_paths: list[str] = Field(min_length=1, max_length=3)
    candidate_names: list[str] = Field(min_length=1, max_length=3)
    binder_chain_ids: list[str] = Field(min_length=1, max_length=8)
    target_chain_ids: list[str] = Field(min_length=1, max_length=8)
    cycle_num: int = Field(default=0, ge=0)


class ProtrekSequenceSearchRequest(ContractModel):
    task_id: str | None = None
    sequence: str = Field(min_length=1, max_length=10000)
    topk: int = Field(default=5, ge=1, le=5)


class ProtrekStructureSearchRequest(ContractModel):
    task_id: str | None = None
    structure_path: str
    chain: str = Field(default="A", min_length=1, max_length=8)
    topk: int = Field(default=5, ge=1, le=5)


class Esm2GuidedProposalRequest(ContractModel):
    task_id: str | None = None
    parent_id: str
    parent_chains: dict[str, str]
    mutable_positions: dict[str, list[int]]
    num_sequences: int = Field(default=8, ge=1, le=256)
    min_llr: float = 0.0
    options: dict[str, Any] = Field(default_factory=dict)
    placement: Placement | None = None
