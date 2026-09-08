"""Materialize and validate Design Agent proposals with Python-owned safety."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from opendde_harness.plugin.protein_design.agents.router import (
    ESM2_GUIDED_MUTATION_SKILL,
    FULL_REDESIGN_SKILL,
    INVERSE_FOLDING_SKILL,
    POINT_MUTATION_SKILL,
    DesignSkillRoute,
)
from opendde_harness.plugin.protein_design.core.constants import CANONICAL_AMINO_ACIDS
from opendde_harness.plugin.protein_design.core.contracts import (
    CandidateProposal,
    DesignAgentOutput,
    Esm2GuidedProposalRequest,
    Mutation,
    Placement,
    SolubleMPNNRequest,
)


class ProposalValidationError(ValueError):
    pass


class ProposalCompute(Protocol):
    async def generate_soluble_mpnn(self, request: SolubleMPNNRequest) -> dict[str, Any]: ...

    async def generate_esm2_guided(self, request: Esm2GuidedProposalRequest) -> Any: ...


@dataclass(frozen=True)
class ProposalContext:
    parent_id: str
    parent_sequences: Mapping[str, str]
    mutable_positions: Mapping[str, tuple[int, ...] | list[int]]
    parent_structure_path: str | None
    candidate_count: int
    cycle: int = 0
    placement: Placement | None = None


class ProposalExecutor:
    def __init__(self, compute: ProposalCompute) -> None:
        self._compute = compute
        self._last_errors: ContextVar[tuple[str, ...]] = ContextVar(
            f"proposal_errors_{id(self)}",
            default=(),
        )

    @property
    def last_errors(self) -> list[str]:
        return list(self._last_errors.get())

    @last_errors.setter
    def last_errors(self, values: list[str]) -> None:
        self._last_errors.set(tuple(str(value) for value in values))

    def _add_error(self, error: str) -> None:
        self._last_errors.set((*self._last_errors.get(), error))

    async def execute(
        self,
        agent_output: DesignAgentOutput,
        route: DesignSkillRoute,
        context: ProposalContext,
    ) -> list[CandidateProposal]:
        self.last_errors = []
        skill_id = route.require_selected(agent_output.skill_id)
        if skill_id in {POINT_MUTATION_SKILL, FULL_REDESIGN_SKILL}:
            proposals = []
            for index, item in enumerate(agent_output.candidates):
                try:
                    proposals.append(self._materialize_llm(item, skill_id, context))
                except ProposalValidationError as exc:
                    self._add_error(f"candidate[{index}]: {exc}")
            if not proposals:
                detail = "; ".join(self.last_errors) or "selected skill produced no candidates"
                raise ProposalValidationError(detail)
        elif skill_id == INVERSE_FOLDING_SKILL:
            proposals = await self._execute_inverse_folding(agent_output, context)
        elif skill_id == ESM2_GUIDED_MUTATION_SKILL:
            proposals = await self._execute_esm2(context)
        else:  # pragma: no cover - guarded by the Router inventory
            raise ProposalValidationError(f"unsupported design skill: {skill_id}")
        return self._validate_batch(proposals, context)

    def _materialize_llm(
        self,
        raw: Mapping[str, Any],
        skill_id: str,
        context: ProposalContext,
    ) -> CandidateProposal:
        mutations = [self._parse_mutation(item) for item in raw.get("mutations", [])]
        if skill_id == POINT_MUTATION_SKILL and not mutations:
            raise ProposalValidationError("point mutation requires at least one mutation")
        if skill_id == POINT_MUTATION_SKILL and not any(
            context.parent_sequences.get(item.chain_id, "")[item.position].upper()
            != item.to_aa.upper()
            for item in mutations
            if item.chain_id in context.parent_sequences
            and 0 <= item.position < len(context.parent_sequences[item.chain_id])
        ):
            raise ProposalValidationError("point mutation must change at least one residue")
        if skill_id == FULL_REDESIGN_SKILL:
            actual = {(item.chain_id, item.position) for item in mutations}
            expected = {
                (chain_id, int(position))
                for chain_id, positions in context.mutable_positions.items()
                for position in positions
            }
            if actual != expected or len(actual) != len(mutations):
                raise ProposalValidationError(
                    "full redesign requires complete mutable CDR coverage exactly once"
                )
        chains = self._apply_mutations(context.parent_sequences, mutations, context)
        return CandidateProposal(
            candidate_id=str(raw.get("candidate_id") or raw.get("id") or "").strip(),
            parent_id=context.parent_id,
            chains=chains,
            mutations=mutations,
            strategy=str(raw.get("strategy", "")),
            risk_level=str(raw.get("risk_level", "unknown")),
            metadata=dict(raw.get("metadata") or {}),
            execution_backend="llm",
            bypass_auxiliary_filters=skill_id == FULL_REDESIGN_SKILL,
        )

    async def _execute_inverse_folding(
        self,
        output: DesignAgentOutput,
        context: ProposalContext,
    ) -> list[CandidateProposal]:
        if not context.parent_structure_path:
            raise ProposalValidationError("inverse folding requires a parent structure")
        first = output.candidates[0] if output.candidates else {}
        metadata = dict(first.get("metadata") or {})
        parameters = dict(metadata.get("soluble_mpnn_parameters") or {})
        num_sequences = int(parameters.pop("num_sequences", context.candidate_count))
        raw_positions = parameters.pop("design_positions", None)
        allowed_positions = {
            str(chain): {int(position) for position in positions}
            for chain, positions in context.mutable_positions.items()
        }
        if raw_positions is not None:
            if not isinstance(raw_positions, Mapping):
                raise ProposalValidationError("SolubleMPNN design_positions must be a chain-to-position mapping")
            selected_positions: dict[str, list[int]] = {}
            for chain, positions in raw_positions.items():
                chain_id = str(chain)
                if chain_id not in allowed_positions:
                    raise ProposalValidationError(f"SolubleMPNN design_positions has unknown chain {chain_id!r}")
                requested = {int(position) for position in positions}
                illegal = sorted(requested - allowed_positions[chain_id])
                if illegal:
                    raise ProposalValidationError(
                        f"SolubleMPNN design_positions targets immutable residues {chain_id}:{illegal}"
                    )
                if requested:
                    selected_positions[chain_id] = sorted(requested)
        else:
            selected_positions = {
                chain: sorted(positions) for chain, positions in allowed_positions.items() if positions
            }
        if raw_positions is not None:
            parameters["design_positions"] = selected_positions
        anchor_mutations = [
            self._parse_mutation(item).model_dump(mode="json")
            for item in first.get("mutations", [])
        ]
        result = await self._compute.generate_soluble_mpnn(
            SolubleMPNNRequest(
                structure_path=context.parent_structure_path,
                mutable_positions=[
                    f"{chain_id}:{int(position)}"
                    for chain_id, positions in context.mutable_positions.items()
                    for position in positions
                ],
                parent_chains=dict(context.parent_sequences),
                anchor_mutations=anchor_mutations,
                num_sequences=min(context.candidate_count, num_sequences),
                parameters=parameters,
                placement=context.placement,
            )
        )
        return self._parse_backend_candidates(
            result,
            skill_id=INVERSE_FOLDING_SKILL,
            backend="inverse_folding",
            context=context,
        )

    async def _execute_esm2(self, context: ProposalContext) -> list[CandidateProposal]:
        response = await self._compute.generate_esm2_guided(
            Esm2GuidedProposalRequest(
                parent_id=context.parent_id,
                parent_chains=dict(context.parent_sequences),
                mutable_positions={
                    chain_id: [int(position) for position in positions]
                    for chain_id, positions in context.mutable_positions.items()
                },
                num_sequences=context.candidate_count,
                placement=context.placement,
            )
        )
        if not getattr(response, "available", False):
            raise ProposalValidationError(
                f"ESM2-guided proposal unavailable: {getattr(response, 'error', None)}"
            )
        return self._parse_backend_candidates(
            getattr(response, "result", None) or {},
            skill_id=ESM2_GUIDED_MUTATION_SKILL,
            backend="esm2",
            context=context,
        )

    def _parse_backend_candidates(
        self,
        payload: Mapping[str, Any],
        *,
        skill_id: str,
        backend: str,
        context: ProposalContext,
    ) -> list[CandidateProposal]:
        raw_candidates = payload.get("candidates") or payload.get("sequences") or []
        proposals: list[CandidateProposal] = []
        for index, raw_value in enumerate(raw_candidates):
            raw = raw_value if isinstance(raw_value, Mapping) else {"sequence": raw_value}
            chains = self._read_backend_chains(raw, context)
            mutations = [self._parse_mutation(item) for item in raw.get("mutations", [])]
            if not mutations:
                mutations = self._derive_substitutions(context.parent_sequences, chains)
            proposals.append(
                CandidateProposal(
                    candidate_id=str(
                        raw.get("candidate_id")
                        or raw.get("id")
                        or f"c{context.cycle:04d}_{context.parent_id}_{backend}_{index + 1:03d}"
                    ),
                    parent_id=context.parent_id,
                    chains=chains,
                    mutations=mutations,
                    strategy=str(raw.get("strategy", f"[{skill_id}] Python backend proposal")),
                    risk_level=str(raw.get("risk_level", "medium")),
                    metadata={**dict(raw.get("metadata") or {}), "skill_id": skill_id},
                    execution_backend=backend,
                )
            )
        return proposals

    @staticmethod
    def _read_backend_chains(
        raw: Mapping[str, Any], context: ProposalContext
    ) -> dict[str, str]:
        if isinstance(raw.get("chains"), Mapping):
            return {str(key): str(value).upper() for key, value in raw["chains"].items()}
        sequence = raw.get("sequence")
        if sequence is not None and len(context.parent_sequences) == 1:
            chain_id = next(iter(context.parent_sequences))
            return {chain_id: str(sequence).upper()}
        raise ProposalValidationError("backend candidate is missing materialized chains")

    @staticmethod
    def _parse_mutation(raw: Any) -> Mutation:
        if isinstance(raw, Mapping):
            payload = {
                "chain_id": raw.get("chain_id", raw.get("chain")),
                "position": raw.get("position", raw.get("pos")),
                "from_aa": raw.get("from_aa"),
                "to_aa": raw.get("to_aa", raw.get("residue", raw.get("aa"))),
            }
        elif isinstance(raw, (list, tuple)) and len(raw) == 3:
            payload = {"chain_id": raw[0], "position": raw[1], "to_aa": raw[2]}
        else:
            raise ProposalValidationError("mutations field must contain three-item arrays")
        try:
            return Mutation.model_validate(payload)
        except Exception as exc:
            raise ProposalValidationError(f"invalid mutation: {raw!r}") from exc

    def _apply_mutations(
        self,
        parent_sequences: Mapping[str, str],
        mutations: list[Mutation],
        context: ProposalContext,
    ) -> dict[str, str]:
        mutable = {
            chain_id: {int(position) for position in positions}
            for chain_id, positions in context.mutable_positions.items()
        }
        result = {chain_id: list(str(sequence).upper()) for chain_id, sequence in parent_sequences.items()}
        seen: set[tuple[str, int]] = set()
        for mutation in mutations:
            key = (mutation.chain_id, mutation.position)
            if key in seen:
                raise ProposalValidationError(f"duplicate mutation target: {key}")
            seen.add(key)
            if mutation.chain_id not in result:
                raise ProposalValidationError(f"unknown chain_id: {mutation.chain_id}")
            if mutation.position not in mutable.get(mutation.chain_id, set()):
                raise ProposalValidationError(
                    f"mutation targets immutable framework residue {mutation.chain_id}:{mutation.position}"
                )
            residue = mutation.to_aa.upper()
            if residue not in CANONICAL_AMINO_ACIDS:
                raise ProposalValidationError(f"non-canonical amino acid: {residue}")
            if mutation.position >= len(result[mutation.chain_id]):
                raise ProposalValidationError(f"mutation position out of range: {key}")
            parent_residue = result[mutation.chain_id][mutation.position]
            if mutation.from_aa not in (None, "", "*") and mutation.from_aa.upper() != parent_residue:
                raise ProposalValidationError(
                    f"mutation source mismatch at {mutation.chain_id}:{mutation.position}: "
                    f"expected {mutation.from_aa.upper()}, parent has {parent_residue}"
                )
            result[mutation.chain_id][mutation.position] = residue
        return {chain_id: "".join(sequence) for chain_id, sequence in result.items()}

    @staticmethod
    def _derive_substitutions(
        parents: Mapping[str, str], chains: Mapping[str, str]
    ) -> list[Mutation]:
        mutations: list[Mutation] = []
        for chain_id, parent in parents.items():
            sequence = chains.get(chain_id)
            if sequence is None or len(sequence) != len(parent):
                continue
            mutations.extend(
                Mutation(chain_id=chain_id, position=index, from_aa=old, to_aa=new)
                for index, (old, new) in enumerate(zip(parent, sequence, strict=True))
                if old != new
            )
        return mutations

    def _validate_batch(
        self,
        proposals: list[CandidateProposal],
        context: ProposalContext,
    ) -> list[CandidateProposal]:
        if not proposals:
            raise ProposalValidationError("selected skill produced no candidates")
        seen_ids: set[str] = set()
        seen_sequences: set[tuple[tuple[str, str], ...]] = set()
        mutable = {
            chain_id: {int(position) for position in positions}
            for chain_id, positions in context.mutable_positions.items()
        }
        for proposal in proposals[: context.candidate_count]:
            if not proposal.candidate_id or proposal.candidate_id in seen_ids:
                raise ProposalValidationError("candidate IDs must be non-empty and unique")
            seen_ids.add(proposal.candidate_id)
            if set(proposal.chains) != set(context.parent_sequences):
                raise ProposalValidationError("candidate chain IDs differ from configured parent")
            for chain_id, sequence in proposal.chains.items():
                if not sequence or set(sequence) - CANONICAL_AMINO_ACIDS:
                    raise ProposalValidationError(f"non-canonical sequence in chain {chain_id}")
                parent = context.parent_sequences[chain_id]
                if len(sequence) != len(parent):
                    raise ProposalValidationError("candidate changed configured chain length")
                for position, (old, new) in enumerate(zip(parent, sequence, strict=True)):
                    if old != new and position not in mutable.get(chain_id, set()):
                        raise ProposalValidationError(
                            f"candidate changed immutable framework residue {chain_id}:{position}"
                        )
            fingerprint = tuple(sorted(proposal.chains.items()))
            if fingerprint in seen_sequences:
                raise ProposalValidationError("duplicate candidate sequence")
            seen_sequences.add(fingerprint)
        return proposals[: context.candidate_count]
