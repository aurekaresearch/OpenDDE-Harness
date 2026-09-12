"""Constrained-elite population and parent sampling."""

from __future__ import annotations

import difflib
import math
import random
from dataclasses import dataclass

from opendde_harness.plugin.protein_design.core.constants import CANONICAL_AMINO_ACIDS
from opendde_harness.plugin.protein_design.core.contracts import Candidate, WorkflowConfig


@dataclass(frozen=True)
class PopulationUpdate:
    survivors: list[Candidate]
    actions: dict[str, str]


class ConstrainedElitePopulation:
    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config
        self._candidates: list[Candidate] = []
        self._reference: dict[str, list[str]] | None = None

    @property
    def candidates(self) -> list[Candidate]:
        return list(self._candidates)

    def update(self, candidates: list[Candidate]) -> PopulationUpdate:
        ordered = sorted(
            [*self._candidates, *candidates],
            key=self._sort_key,
            reverse=not self._config.minimize,
        )
        unique: list[Candidate] = []
        actions: dict[str, str] = {}
        sequence_keys: set[tuple[tuple[str, str], ...]] = set()
        for candidate in ordered:
            if candidate.objective is None or not math.isfinite(float(candidate.objective)):
                actions[candidate.candidate_id] = "invalid_objective_not_retained"
                continue
            if candidate.metadata.get("success") is False:
                actions[candidate.candidate_id] = "failed_candidate_not_retained"
                continue
            key = self._sequence_key(candidate)
            if any(set(sequence.upper()) - CANONICAL_AMINO_ACIDS for _chain, sequence in key):
                actions[candidate.candidate_id] = "masked_sequence_not_retained"
                continue
            if key in sequence_keys:
                actions[candidate.candidate_id] = "duplicate_not_retained"
                continue
            sequence_keys.add(key)
            unique.append(candidate)

        if self._reference is None:
            for candidate in unique:
                signature = self._signature_dict(candidate)
                if any(signature.values()):
                    self._reference = signature
                    break

        selected: list[Candidate] = []
        position_usage: dict[str, int] = {}
        mutation_usage: dict[str, int] = {}
        position_limit = max(
            1,
            math.ceil(self._config.population_size * self._config.constrained_max_position_reuse_fraction),
        )
        mutation_limit = max(
            1,
            math.ceil(self._config.population_size * self._config.constrained_max_mutation_reuse_fraction),
        )

        for candidate in unique:
            if len(selected) >= self._config.population_size:
                actions[candidate.candidate_id] = "capacity_not_retained"
                continue
            nearest = min(
                (self.cdr_distance(candidate, incumbent) for incumbent in selected),
                default=math.inf,
            )
            if nearest < self._config.constrained_min_cdr_distance:
                actions[candidate.candidate_id] = "cdr_distance_not_retained"
                continue
            positions, mutations = self._mutation_features(candidate)
            if any(position_usage.get(token, 0) >= position_limit for token in positions):
                actions[candidate.candidate_id] = "position_reuse_not_retained"
                continue
            if any(mutation_usage.get(token, 0) >= mutation_limit for token in mutations):
                actions[candidate.candidate_id] = "mutation_reuse_not_retained"
                continue
            selected.append(candidate)
            actions[candidate.candidate_id] = "retained_constrained_elite"
            for token in positions:
                position_usage[token] = position_usage.get(token, 0) + 1
            for token in mutations:
                mutation_usage[token] = mutation_usage.get(token, 0) + 1

        self._candidates = selected
        return PopulationUpdate(survivors=list(selected), actions=actions)

    def cdr_distance(self, left: Candidate, right: Candidate) -> float:
        left_regions = self._flat_signature(left)
        right_regions = self._flat_signature(right)
        keys = sorted(set(left_regions) | set(right_regions))
        if not keys:
            return 0.0
        is_single_chain_three_cdrs = (
            len(keys) == 3
            and len({chain for chain, _index in keys}) == 1
            and {index for _chain, index in keys} == {0, 1, 2}
        )
        weights = [0.25, 0.25, 0.50] if is_single_chain_three_cdrs else [1.0 / len(keys)] * len(keys)
        return sum(
            weight
            * self._normalized_edit_distance(
                left_regions.get(key, ""),
                right_regions.get(key, ""),
            )
            for key, weight in zip(keys, weights)
        )

    def _sort_key(self, candidate: Candidate) -> float:
        if candidate.objective is None or not math.isfinite(float(candidate.objective)):
            return math.inf if self._config.minimize else -math.inf
        return float(candidate.objective)

    @staticmethod
    def _sequence_key(candidate: Candidate) -> tuple[tuple[str, str], ...]:
        chains = candidate.metadata.get("chains")
        if isinstance(chains, dict) and chains:
            return tuple(sorted((str(chain), str(sequence)) for chain, sequence in chains.items()))
        return (("", candidate.sequence),)

    def _chains(self, candidate: Candidate) -> dict[str, str]:
        chains = candidate.metadata.get("chains")
        if isinstance(chains, dict) and chains:
            return {str(chain): str(sequence) for chain, sequence in chains.items()}
        if len(self._config.binder_chains) == 1:
            return {next(iter(self._config.binder_chains)): candidate.sequence}
        return {"": candidate.sequence}

    def _groups(self, chain: str, length: int) -> list[list[int]]:
        groups = self._config.cdr_region_groups.get(chain, [])
        if groups:
            return [[position for position in group if 0 <= position < length] for group in groups]
        mutable = self._config.mutable_positions.get(chain, list(range(length)))
        if not mutable:
            return []
        contiguous: list[list[int]] = [[mutable[0]]]
        for position in mutable[1:]:
            if position == contiguous[-1][-1] + 1:
                contiguous[-1].append(position)
            else:
                contiguous.append([position])
        return contiguous

    def _signature_dict(self, candidate: Candidate) -> dict[str, list[str]]:
        return {
            chain: ["".join(sequence[position] for position in group) for group in self._groups(chain, len(sequence))]
            for chain, sequence in sorted(self._chains(candidate).items())
        }

    def _flat_signature(self, candidate: Candidate) -> dict[tuple[str, int], str]:
        return {
            (chain, index): region
            for chain, regions in self._signature_dict(candidate).items()
            for index, region in enumerate(regions)
        }

    def _mutation_features(self, candidate: Candidate) -> tuple[set[str], set[str]]:
        reference = self._reference or {}
        current = self._signature_dict(candidate)
        position_tokens: set[str] = set()
        mutation_tokens: set[str] = set()
        for chain in sorted(set(reference) | set(current)):
            left_regions = reference.get(chain, [])
            right_regions = current.get(chain, [])
            for region_index in range(max(len(left_regions), len(right_regions))):
                left = left_regions[region_index] if region_index < len(left_regions) else ""
                right = right_regions[region_index] if region_index < len(right_regions) else ""
                prefix = f"{chain}:CDR{region_index + 1}"
                matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal":
                        continue
                    paired = min(i2 - i1, j2 - j1)
                    for offset in range(paired):
                        position = f"{prefix}:{i1 + offset + 1}"
                        position_tokens.add(position)
                        mutation_tokens.add(f"{position}:{left[i1 + offset]}>{right[j1 + offset]}")
                    for left_index in range(i1 + paired, i2):
                        position = f"{prefix}:{left_index + 1}"
                        position_tokens.add(position)
                        mutation_tokens.add(f"{position}:{left[left_index]}>-")
                    for insertion_offset, right_index in enumerate(range(j1 + paired, j2), start=1):
                        position = f"{prefix}:ins{i2 + 1}.{insertion_offset}"
                        position_tokens.add(position)
                        mutation_tokens.add(f"{position}:->{right[right_index]}")
        return position_tokens, mutation_tokens

    @staticmethod
    def _normalized_edit_distance(left: str, right: str) -> float:
        if left == right:
            return 0.0
        normalizer = max(len(left), len(right))
        if normalizer == 0:
            return 0.0
        if len(left) == len(right):
            return sum(a != b for a, b in zip(left, right)) / normalizer
        previous = list(range(len(right) + 1))
        for left_index, left_residue in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_residue in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1] + (left_residue != right_residue),
                    )
                )
            previous = current
        return previous[-1] / normalizer


class ParentSampler:
    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)

    def select(self, candidates: list[Candidate], cycle: int) -> Candidate:
        if not candidates:
            raise ValueError("cannot select a parent from an empty population")
        ordered = sorted(
            candidates,
            key=lambda item: math.inf if item.objective is None else float(item.objective),
            reverse=not self._config.minimize,
        )
        strategy = self._config.parent_selection_strategy
        if strategy == "greedy":
            return ordered[0]
        if strategy == "uniform":
            return self._rng.choice(ordered)
        return self._fitness(ordered, cycle, self._rng)

    def snapshot_state(self) -> object:
        """Capture deterministic sampler state for speculative selection."""
        return self._rng.getstate()

    def restore_state(self, state: object) -> None:
        """Roll back a speculative draw that could not be committed."""
        self._rng.setstate(state)

    def temperature(self, cycle: int) -> float:
        if self._config.parent_fitness_temperature is not None:
            return self._config.parent_fitness_temperature
        progress = min(1.0, max(0.0, cycle / max(1, self._config.cycles - 1)))
        start = self._config.parent_fitness_temperature_start
        end = self._config.parent_fitness_temperature_end
        return start * (end / start) ** progress

    def _fitness(
        self,
        candidates: list[Candidate],
        cycle: int,
        rng: random.Random,
    ) -> Candidate:
        temperature = self.temperature(cycle)
        scores = [
            float(candidate.objective) if candidate.objective is not None else math.inf for candidate in candidates
        ]
        utilities = [-value for value in scores] if self._config.minimize else scores
        finite = [value for value in utilities if math.isfinite(value)]
        if not finite or max(finite) == min(finite):
            softmax = [1.0] * len(candidates)
        else:
            lower, upper = min(finite), max(finite)
            normalized = [(value - lower) / (upper - lower) if math.isfinite(value) else 0.0 for value in utilities]
            softmax = [math.exp(max(-700.0, (value - 1.0) / max(temperature, 1e-6))) for value in normalized]
        total = sum(softmax)
        uniform = self._config.parent_fitness_uniform_fraction / len(candidates)
        weights = [(1.0 - self._config.parent_fitness_uniform_fraction) * value / total + uniform for value in softmax]
        return rng.choices(candidates, weights=weights, k=1)[0]


class WorkingParentTracker:
    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config
        self._candidate: Candidate | None = None

    @property
    def candidate(self) -> Candidate | None:
        return self._candidate

    def consider(self, candidate: Candidate) -> bool:
        if self._candidate is None or self._is_better(candidate, self._candidate):
            self._candidate = candidate
            return True
        return False

    def _is_better(self, candidate: Candidate, incumbent: Candidate) -> bool:
        candidate_fraction = self._contact_fraction(candidate)
        incumbent_fraction = self._contact_fraction(incumbent)
        if candidate_fraction != incumbent_fraction:
            return candidate_fraction > incumbent_fraction
        if candidate.objective is None:
            return False
        if incumbent.objective is None:
            return True
        if self._config.minimize:
            return candidate.objective < incumbent.objective
        return candidate.objective > incumbent.objective

    @staticmethod
    def _contact_fraction(candidate: Candidate) -> float:
        evidence = candidate.metadata.get("gate_evidence")
        if isinstance(evidence, dict):
            value = evidence.get("cdr_contact_fraction")
            if value is not None:
                return float(value)
        value = candidate.metadata.get("cdr_contact_fraction")
        if value is None:
            value = candidate.metrics.get("cdr_contact_fraction", 0.0)
        return float(value)
