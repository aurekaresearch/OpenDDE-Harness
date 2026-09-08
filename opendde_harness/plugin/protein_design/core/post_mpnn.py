"""Generate and select sequence variants before terminal refolding."""

from __future__ import annotations

import asyncio
import math

from opendde_harness.plugin.protein_design.core.constants import CANONICAL_AMINO_ACIDS
from opendde_harness.plugin.protein_design.core.contracts import Candidate, SolubleMPNNRequest, WorkflowConfig

MPNN_SAMPLE_COUNT = 40
MPNN_SURVIVORS_PER_PARENT = 4


async def prepare_post_mpnn(
    compute, parents: list[Candidate], config: WorkflowConfig, task_id: str, stop_event: asyncio.Event
) -> list[Candidate]:
    children = []
    for parent in parents:
        if stop_event.is_set():
            raise RuntimeError("Task stopped during post-MPNN")
        try:
            if not parent.structure_path or parent.metadata.get("success") is False:
                raise ValueError("No successful parent structure for SolubleMPNN")
            chains = parent.metadata.get("chains") or {}
            if not chains and len(config.binder_chains) == 1:
                chains = {next(iter(config.binder_chains)): parent.sequence}
            if not chains or set(chains) != set(config.binder_chains):
                raise ValueError("Parent binder chains do not match the configured chains")
            positions = {
                chain: sorted(set(config.mutable_positions.get(chain, [])) - set(config.fixed_residues.get(chain, [])))
                for chain in chains
            }
            count = sum(map(len, positions.values()))
            if not count or any(p < 0 or p >= len(chains[c]) for c, ps in positions.items() for p in ps):
                raise ValueError("No valid mutable positions for SolubleMPNN")
            response = await compute.generate_soluble_mpnn(
                SolubleMPNNRequest(
                    task_id=task_id,
                    structure_path=parent.structure_path,
                    parent_chains=chains,
                    mutable_positions=[f"{c}:{p}" for c, ps in positions.items() for p in ps],
                    num_sequences=MPNN_SAMPLE_COUNT,
                    placement=config.placement,
                )
            )
            if stop_event.is_set():
                raise RuntimeError("Task stopped during post-MPNN")
            samples = response.get("candidates") or []
            if len(samples) != MPNN_SAMPLE_COUNT:
                raise ValueError(f"Expected {MPNN_SAMPLE_COUNT} SolubleMPNN samples, received {len(samples)}")
            ranked = []
            for index, sample in enumerate(samples, start=1):
                proposed = sample.get("chains") or {}
                scores = (sample.get("metadata") or {}).get("soluble_mpnn_scores") or {}
                if set(proposed) != set(chains):
                    continue
                if any(
                    len(proposed[c]) != len(seq)
                    or set(proposed[c]) - CANONICAL_AMINO_ACIDS
                    or any(a != b and p not in positions[c] for p, (a, b) in enumerate(zip(seq, proposed[c])))
                    for c, seq in chains.items()
                ):
                    continue
                try:
                    score = sum(float(scores[c]) * len(ps) for c, ps in positions.items() if ps) / count
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    ranked.append((score, index, proposed, sample.get("metadata") or {}))
            selected, seen = [], set()
            for score, index, proposed, metadata in sorted(ranked, key=lambda item: (item[0], item[1])):
                fingerprint = tuple(sorted(proposed.items()))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                selected.append(
                    Candidate(
                        candidate_id=f"{parent.candidate_id}__mpnn_{index:02d}",
                        sequence="".join(proposed[c] for c in chains),
                        structure_path=parent.structure_path,
                        metadata={
                            "chains": proposed,
                            "parent_id": parent.candidate_id,
                            "skill_id": "antibody-inverse-folding",
                            "post_mpnn_selected": True,
                            "post_refold_success": False,
                            "mpnn_sample_index": index,
                            "mpnn_rank": len(selected) + 1,
                            "mpnn_score": score,
                            "soluble_mpnn_scores": metadata.get("soluble_mpnn_scores"),
                            "soluble_mpnn_weights_sha256": metadata.get("soluble_mpnn_weights_sha256"),
                            "pre_refold_structure_path": parent.structure_path,
                        },
                    )
                )
                if len(selected) == MPNN_SURVIVORS_PER_PARENT:
                    break
            if not selected:
                raise ValueError("No distinct valid SolubleMPNN sequences for this parent")
            if len(selected) < MPNN_SURVIVORS_PER_PARENT:
                shortfall = (
                    f"only {len(selected)} of {MPNN_SURVIVORS_PER_PARENT} distinct valid SolubleMPNN sequences"
                )
                for child in selected:
                    child.metadata["post_mpnn_shortfall"] = shortfall
            children.extend(selected)
        except Exception as exc:
            if stop_event.is_set():
                raise
            failed = parent.model_copy(deep=True)
            failed.metadata.update(
                success=False,
                post_refold_success=False,
                post_mpnn_selected=False,
                post_refold_error=f"SolubleMPNN: {exc}",
            )
            children.append(failed)
    return children
