"""Shared constants and predicates for the protein design plugin."""

from __future__ import annotations

from typing import Any, Mapping

DEFAULT_OPENDDE_API_URL = "https://api.aurekabio.cloud"

DEFAULT_COMPUTE_PORT = 8080
DEFAULT_COMPUTE_URL = f"http://127.0.0.1:{DEFAULT_COMPUTE_PORT}"

CANONICAL_AMINO_ACIDS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


def is_canonical_sequence(sequence: Any) -> bool:
    return bool(sequence) and set(str(sequence).upper()) <= CANONICAL_AMINO_ACIDS


def is_materialized(candidate: Any) -> bool:
    """True when every chain of a Candidate or a raw candidate mapping is fully canonical."""

    if isinstance(candidate, Mapping):
        chains = candidate.get("chains")
        fallback = candidate.get("sequence", "")
    else:
        metadata = getattr(candidate, "metadata", None) or {}
        chains = metadata.get("chains") if isinstance(metadata, Mapping) else None
        fallback = getattr(candidate, "sequence", "")
    sequences = chains.values() if isinstance(chains, Mapping) and chains else [fallback]
    return all(is_canonical_sequence(sequence) for sequence in sequences)
