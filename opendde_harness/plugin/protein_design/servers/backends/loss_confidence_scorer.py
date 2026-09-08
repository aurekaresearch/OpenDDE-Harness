"""Gradient-free design loss from OpenDDE or Protenix confidence outputs."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from opendde_harness.plugin.protein_design.servers.backends.loss_objective import calculate_loss_objective
from opendde_harness.plugin.protein_design.servers.backends.structure_contacts import (
    iter_protein_residues,
    load_structure_model,
)

PROTENIX_LOSS_BACKEND = "protenix-contact8-proxy-gradient-free-v2"
OPENDDE_LOSS_BACKEND = "opendde-contact8-proxy-gradient-free-v1"
CONTACT_PROBABILITY_CUTOFF_ANGSTROM = 8.0
REFERENCE_INTER_CONTACT_CUTOFF_ANGSTROM = 20.0
CONTACT_LOSS_PROXY = "binary_nll_from_confidence_p_distance_lt_8a"


def _resolve_confidence_backend(backend: str) -> tuple[str, str]:
    normalized = str(backend or "").strip().lower()
    if normalized.startswith("opendde"):
        return OPENDDE_LOSS_BACKEND, "OpenDDE"
    if normalized.startswith("protenix"):
        return PROTENIX_LOSS_BACKEND, "Protenix"
    raise ValueError(f"Loss scoring requires a supported fold backend, got {backend!r}; expected OpenDDE or Protenix")


logger = logging.getLogger(__name__)


def _load_first_json_object(path: str | Path) -> Mapping[str, Any]:
    """Load the first complete JSON object and tolerate stale trailing bytes.

    A fold backend can occasionally leave bytes from a previous, longer
    confidence payload after an otherwise complete JSON object. The first
    decoded object
    is authoritative; malformed or incomplete leading JSON still fails.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        # Some backend builds occasionally serialize integral floating-point
        # values as ``6.`` or ``0.``.  Those tokens are valid Python/NumPy
        # numbers but invalid JSON.  Repair only this narrow numeric form and
        # retry; all other malformed or incomplete payloads still fail loudly.
        normalized = re.sub(
            r"(?<![\w.])(-?\d+)\.(?=\s*[,}\]])",
            r"\1.0",
            stripped,
        )
        if normalized == stripped:
            raise
        try:
            value, end = json.JSONDecoder().raw_decode(normalized)
        except json.JSONDecodeError:
            raise exc
        stripped = normalized
        logger.warning(
            "Normalized non-standard terminal-decimal floats in confidence JSON: %s",
            source,
        )
    if not isinstance(value, Mapping):
        raise ValueError(f"Confidence payload root must be an object: {source}")
    trailing = stripped[end:].strip()
    if trailing:
        logger.warning(
            "Ignoring %d stale trailing characters after complete confidence JSON: %s",
            len(trailing),
            source,
        )
    return value


def _ordered_unique(values: np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values.tolist()))


def _mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot aggregate an empty confidence selection")
    return float(finite.mean())


def _token_plddt(full_data: Mapping[str, Any], token_count: int) -> np.ndarray:
    atom_plddt = np.asarray(full_data["atom_plddt"], dtype=float).reshape(-1)
    atom_to_token = np.asarray(full_data["atom_to_token_idx"], dtype=int).reshape(-1)
    if atom_plddt.shape != atom_to_token.shape:
        raise ValueError("atom_plddt and atom_to_token_idx must have equal length")
    counts = np.bincount(atom_to_token, minlength=token_count)
    totals = np.bincount(atom_to_token, weights=atom_plddt, minlength=token_count)
    if np.any(counts[:token_count] == 0):
        raise ValueError("Every protein token must have atom-level pLDDT")
    return totals[:token_count] / counts[:token_count]


def _top_contact_loss(
    contact_probs: np.ndarray,
    rows: Sequence[int],
    columns: Sequence[int],
    *,
    top_k: int,
) -> float:
    if not rows or not columns:
        raise ValueError("Contact loss requires non-empty row and column masks")
    losses = -np.log(np.clip(contact_probs[np.ix_(rows, columns)], 1e-8, 1.0))
    k = min(max(1, int(top_k)), losses.shape[1])
    nearest = np.partition(losses, k - 1, axis=1)[:, :k]
    return _mean(nearest.mean(axis=1))


def _binder_contact_loss(
    contact_probs: np.ndarray,
    chain_token_indices: Mapping[str, Sequence[int]],
) -> float:
    per_residue: list[float] = []
    for indices in chain_token_indices.values():
        indices = list(indices)
        for local_i, token_i in enumerate(indices):
            partners = [token_j for local_j, token_j in enumerate(indices) if abs(local_i - local_j) >= 9]
            if not partners:
                continue
            losses = -np.log(np.clip(contact_probs[token_i, partners], 1e-8, 1.0))
            per_residue.append(float(np.partition(losses, min(1, len(losses) - 1))[:2].mean()))
    return _mean(np.asarray(per_residue, dtype=float))


def _resolve_hotspot_tokens(
    chain_token_indices: Mapping[str, Sequence[int]],
    target_chains: Sequence[str],
    target_hotspots: Mapping[str, Sequence[int]] | None,
) -> tuple[list[int], dict[str, list[int]], str]:
    """Resolve configured one-based target positions to confidence token indices.

    YAML hotspot positions, ``HotspotResidue`` and fold-backend pocket constraints
    all use one-based sequence positions.  Invalid configured hotspots must
    fail loudly: silently falling back to the whole target changes the design
    objective without telling the caller.
    """
    all_target_tokens = [token for chain in target_chains for token in chain_token_indices[chain]]
    if not target_hotspots or not any(target_hotspots.values()):
        return all_target_tokens, {}, "all_target_residues"

    target_chain_set = set(target_chains)
    normalized: dict[str, list[int]] = {}
    hotspot_tokens: list[int] = []
    for raw_chain, raw_positions in target_hotspots.items():
        chain = str(raw_chain)
        if chain not in target_chain_set:
            raise ValueError(
                f"Configured hotspot chain {chain!r} is not a target chain; "
                f"target chains are {sorted(target_chain_set)}"
            )
        positions = sorted(set(int(position) for position in raw_positions))
        if not positions:
            continue
        chain_length = len(chain_token_indices[chain])
        invalid = [position for position in positions if position < 1 or position > chain_length]
        if invalid:
            raise ValueError(
                f"Hotspot positions for chain {chain} must be one-based and within 1..{chain_length}; got {invalid}"
            )
        normalized[chain] = positions
        hotspot_tokens.extend(chain_token_indices[chain][position - 1] for position in positions)

    if not hotspot_tokens:
        raise ValueError("Configured target_hotspots contains no residue positions")
    return hotspot_tokens, normalized, "configured_hotspots"


def _chain_ca_coordinates(
    structure_path: str | Path,
    sequences: Mapping[str, str],
) -> dict[str, np.ndarray]:
    atoms = load_structure_model(structure_path)
    coordinates: dict[str, list[np.ndarray]] = {str(chain): [] for chain in sequences}
    for start, stop in iter_protein_residues(atoms, set(coordinates)):
        chain = str(atoms.chain_id[start])
        ca = np.where(np.asarray(atoms.atom_name[start:stop]) == "CA")[0]
        if ca.size:
            coordinates[chain].append(np.asarray(atoms.coord[start + int(ca[0])], dtype=float))
    result = {chain: np.asarray(values, dtype=float) for chain, values in coordinates.items()}
    for chain, sequence in sequences.items():
        observed = len(result.get(str(chain), ()))
        if observed != len(sequence):
            raise ValueError(f"Structure chain {chain} has {observed} CA residues; expected {len(sequence)}")
    return result


def _geometry_components(
    structure_path: str | Path,
    sequences: Mapping[str, str],
    binder_chains: Sequence[str],
    framework_local: Mapping[str, Sequence[int]],
    contact_probs: np.ndarray,
    chain_token_indices: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    coordinates = _chain_ca_coordinates(structure_path, sequences)
    binder_ca = np.concatenate([coordinates[chain] for chain in binder_chains], axis=0)
    radius = float(np.sqrt(np.square(binder_ca - binder_ca.mean(axis=0)).sum(axis=1).mean() + 1e-8))
    expected_radius = 2.38 * binder_ca.shape[0] ** 0.365
    delta = radius - expected_radius
    rg = delta if delta > 0.0 else math.expm1(delta)

    framework_tokens = [
        chain_token_indices[chain][position] for chain in binder_chains for position in framework_local[chain]
    ]
    dgram_terms: list[float] = []
    flat_coords = {
        chain_token_indices[chain][position]: coordinates[chain][position]
        for chain in binder_chains
        for position in framework_local[chain]
    }
    for offset, token_i in enumerate(framework_tokens):
        for token_j in framework_tokens[offset + 1 :]:
            probability = float(np.clip(contact_probs[token_i, token_j], 1e-8, 1 - 1e-8))
            actual_contact = float(
                np.linalg.norm(flat_coords[token_i] - flat_coords[token_j]) < CONTACT_PROBABILITY_CUTOFF_ANGSTROM
            )
            dgram_terms.append(
                -(actual_contact * math.log(probability) + (1 - actual_contact) * math.log(1 - probability))
            )
    return {
        "rg": float(rg),
        "dgram_cce": _mean(np.asarray(dgram_terms)) if dgram_terms else 0.0,
    }


def score_confidence_loss(
    *,
    backend: str,
    confidence_path: str | Path,
    structure_path: str | Path,
    sequences: Mapping[str, str],
    binder_chains: Sequence[str],
    fixed_residues: Mapping[str, Sequence[int]] | None,
    iptm: float,
    esm2_pll: float,
    loss_weights: Mapping[str, float] | None = None,
    target_hotspots: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Calculate the configured loss from one fold backend's confidence data.

    OpenDDE and Protenix expose the same residue-level confidence contract:
    atom pLDDT, token PAE, and contact probabilities. The backend is explicit
    so metadata cannot claim that an OpenDDE score came from Protenix.
    """
    confidence_backend, backend_label = _resolve_confidence_backend(backend)
    full_data = _load_first_json_object(confidence_path)
    asym_id = np.asarray(full_data["token_asym_id"], dtype=int).reshape(-1)
    token_count = len(asym_id)
    expected_count = sum(len(sequence) for sequence in sequences.values())
    if token_count != expected_count:
        raise ValueError(f"Confidence payload has {token_count} tokens; sequences have {expected_count} residues")
    chain_ids = list(sequences)
    asym_values = _ordered_unique(asym_id)
    if len(asym_values) != len(chain_ids):
        raise ValueError("Confidence asym IDs do not match the configured protein chains")
    chain_token_indices = {
        chain: np.where(asym_id == asym)[0].astype(int).tolist() for chain, asym in zip(chain_ids, asym_values)
    }
    for chain, indices in chain_token_indices.items():
        if len(indices) != len(sequences[chain]):
            raise ValueError(f"Confidence chain {chain} has {len(indices)} tokens; expected {len(sequences[chain])}")

    binder_chains = [str(chain) for chain in binder_chains]
    binder_tokens = [token for chain in binder_chains for token in chain_token_indices[chain]]
    target_chains = [chain for chain in chain_ids if chain not in binder_chains]
    target_tokens = [token for chain in target_chains for token in chain_token_indices[chain]]
    fixed_residues = fixed_residues or {}
    framework_local = {
        chain: sorted(
            position
            for position in set(int(value) for value in fixed_residues.get(chain, ()))
            if 0 <= position < len(sequences[chain])
        )
        for chain in binder_chains
    }
    cdr_local = {
        chain: [position for position in range(len(sequences[chain])) if position not in set(framework_local[chain])]
        for chain in binder_chains
    }
    cdr_tokens = [chain_token_indices[chain][position] for chain in binder_chains for position in cdr_local[chain]]
    framework_tokens = [
        chain_token_indices[chain][position] for chain in binder_chains for position in framework_local[chain]
    ]
    if not cdr_tokens or not framework_tokens:
        raise ValueError("Loss scoring requires both CDR and framework residues")

    objective_target_tokens, normalized_hotspots, target_scope = _resolve_hotspot_tokens(
        chain_token_indices,
        target_chains,
        target_hotspots,
    )

    token_plddt = _token_plddt(full_data, token_count)
    pae = np.asarray(full_data["token_pair_pae"], dtype=float).squeeze()
    contact_probs = np.asarray(full_data["contact_probs"], dtype=float).squeeze()
    if pae.shape != (token_count, token_count) or contact_probs.shape != pae.shape:
        raise ValueError("PAE/contact matrices do not match token count")
    symmetric_pae = (pae + pae.T) / 2.0 / 31.0

    # Match the reference mask orientation. With hotspots, average over
    # hotspot rows and selects the closest CDR columns. Without hotspots, it
    # averages over CDR rows and selects the closest target columns.
    if normalized_hotspots:
        cdr_contact_loss = _top_contact_loss(contact_probs, objective_target_tokens, cdr_tokens, top_k=10)
    else:
        cdr_contact_loss = _top_contact_loss(contact_probs, cdr_tokens, target_tokens, top_k=10)
    # The framework term always covers the complete antigen, even when
    # the positive CDR term is restricted to configured hotspots.
    framework_contact_loss = _top_contact_loss(contact_probs, target_tokens, framework_tokens, top_k=10)
    offset = 1.0
    paratope_loss = cdr_contact_loss * (cdr_contact_loss / max(framework_contact_loss - offset, 1e-8))
    raw = {
        "plddt": 1.0 - _mean(token_plddt[binder_tokens]),
        "i_plddt": 1.0 - _mean(token_plddt[cdr_tokens]),
        "pae": _mean(symmetric_pae[np.ix_(binder_tokens, range(token_count))]),
        "i_pae": _mean(symmetric_pae[np.ix_(cdr_tokens, objective_target_tokens)]),
        "i_ptm": 1.0 - float(iptm),
        "con": _binder_contact_loss(
            contact_probs,
            {chain: chain_token_indices[chain] for chain in binder_chains},
        ),
        "i_con": float(paratope_loss),
        **_geometry_components(
            structure_path,
            sequences,
            binder_chains,
            framework_local,
            contact_probs,
            chain_token_indices,
        ),
    }
    objective = calculate_loss_objective(raw, esm2_pll, weights=loss_weights)
    objective.update(
        {
            "confidence_backend": confidence_backend,
            "confidence_backend_label": backend_label,
            "contact_probability_cutoff_angstrom": (CONTACT_PROBABILITY_CUTOFF_ANGSTROM),
            "contact_probability_semantics": "P(distance < 8 angstrom)",
            "contact_loss_proxy": CONTACT_LOSS_PROXY,
            "reference_inter_contact_cutoff_angstrom": (REFERENCE_INTER_CONTACT_CUTOFF_ANGSTROM),
            "reference_contact_loss_available": False,
            "target_contact_scope": target_scope,
            "target_hotspot_position_semantics": "one_based_sequence_position",
            "target_hotspots_used": normalized_hotspots,
            "target_hotspot_token_count": sum(len(positions) for positions in normalized_hotspots.values()),
            "target_objective_token_count": len(objective_target_tokens),
            "gradient_free": True,
        }
    )
    paratope = {
        "cdr_contact_loss": float(cdr_contact_loss),
        "framework_contact_loss": float(framework_contact_loss),
        "offset": offset,
        "paratope_loss": float(paratope_loss),
        "contact_probability_cutoff_angstrom": (CONTACT_PROBABILITY_CUTOFF_ANGSTROM),
        "contact_probability_semantics": "P(distance < 8 angstrom)",
        "contact_loss_proxy": CONTACT_LOSS_PROXY,
        "reference_inter_contact_cutoff_angstrom": (REFERENCE_INTER_CONTACT_CUTOFF_ANGSTROM),
        "reference_contact_loss_available": False,
        "target_contact_scope": target_scope,
        "target_hotspot_position_semantics": "one_based_sequence_position",
        "target_hotspots_used": normalized_hotspots,
        "target_hotspot_token_count": sum(len(positions) for positions in normalized_hotspots.values()),
        "target_objective_token_count": len(objective_target_tokens),
    }
    return {
        **objective,
        "loss_objective": objective,
        "loss_components": raw,
        "paratope_loss": float(paratope_loss),
        "paratope_loss_breakdown": paratope,
        "loss_confidence_backend": confidence_backend,
        "loss_confidence_backend_label": backend_label,
    }
