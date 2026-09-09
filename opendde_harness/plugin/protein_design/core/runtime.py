"""Workflow config loading and validation for the protein-design plugin."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from opendde_harness.plugin.protein_design.core.contracts import Placement, WorkflowConfig
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import normalize_loss_weights


@dataclass(frozen=True)
class _TargetSection:
    name: str
    chains: dict[str, Any]
    sequence: str
    hotspots: list[int]


@dataclass(frozen=True)
class _BinderSection:
    chains: dict[str, str]
    fixed_residues: dict[str, list[int]]
    explicit_fixed_residues: dict[str, list[int]]
    designable_residues: dict[str, list[int]]
    cdr_regions: dict[str, list[int]]
    cdr_region_groups: dict[str, list[list[int]]]
    chain_options: dict[str, dict[str, Any]]
    chain_type_by_chain: dict[str, str]
    initial_candidates: list[dict[str, Any]]


class WorkflowConfigLoader:
    _DESIGN_FIELDS = frozenset(
        {
            "bootstrap_full_redesign_cycles",
            "cdr_contact_fraction_threshold",
            "constrained_max_mutation_reuse_fraction",
            "constrained_max_position_reuse_fraction",
            "constrained_min_cdr_distance",
            "cycle_retry_limit",
            "enable_quality_check",
            "esm_device",
            "hotspot_contact_cutoff_a",
            "initial_structure_path",
            "loss_weights",
            "n_cycles",
            "num_mutations",
            "num_sequences",
            "optimization_metric",
            "parent_fitness_temperature_end",
            "parent_fitness_temperature_start",
            "parent_fitness_uniform_fraction",
            "parent_selection_strategy",
            "population_size",
            "post_refold_filter",
            "quality_check_threshold",
            "reflection_interval",
            "router_skill_probabilities",
            "skip_failed_cycles",
            "stagnation_full_redesign_threshold",
        }
    )
    _COMPUTE_FIELDS = frozenset({"placement", "profile", "url", "worker_id"})
    _STRUCTURED_FIELDS = frozenset(
        {
            "benchmark_metadata",
            "compute",
            "design",
            "fold",
            "initial_binders",
            "llm",
            "seed",
            "target",
        }
    )
    _LLM_FIELDS = frozenset({"max_tokens", "model_name", "temperature"})
    _FOLD_FIELDS = frozenset(
        {
            "deterministic",
            "diffusion_samples",
            "diffusion_steps",
            "api_poll_interval_seconds",
            "api_request_timeout_seconds",
            "api_stalled_poll_limit",
            "api_timeout_seconds",
            "api_url",
            "enable_batch_inference",
            "enable_msa_search",
            "execution_mode",
            "gpus",
            "image",
            "ipsae_dist_cutoff",
            "ipsae_pae_cutoff",
            "model",
            "need_atom_confidence",
            "persistent_worker",
            "persistent_worker_timeout_seconds",
            "recycling_cycles",
            "seeds",
            "subprocess_timeout_seconds",
            "target_msa",
            "use_msa",
            "use_templates",
        }
    )
    _TARGET_FIELDS = frozenset({"chains", "name"})
    _TARGET_CHAIN_FIELDS = frozenset({"hotspots", "pairedMsaPath", "sequence", "unpairedMsaPath"})
    _BINDER_FIELDS = frozenset({"chains", "name"})
    _BINDER_CHAIN_FIELDS = frozenset(
        {
            "cdr_regions",
            "chain_type",
            "designable_residues",
            "fixed_residues",
            "pairedMsaPath",
            "sequence",
            "unpairedMsaPath",
        }
    )
    _POST_FILTER_FIELDS = frozenset({"enabled", "top_k"})
    _ANTIBODY_CHAIN_TYPE_ALIASES = {
        "VHH": "VHH",
        "VH": "VH",
        "VL": "VL",
        "VK": "VL",
        "VLK": "VL",
        "VL-KAPPA": "VL",
        "VL-LAMBDA": "VL",
        "SCFV": "scFv",
    }

    @classmethod
    def config_from_path(cls, config_path: str, plugin_config: dict[str, Any] | None = None) -> WorkflowConfig:
        path = Path(config_path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("protein-design config must be a YAML object")
        defaults = (plugin_config or {}).get("fold_defaults") or {}
        if defaults:
            fold = cls._mapping_section(data, "fold")
            data = {**data, "fold": {**defaults, **fold}}
        return cls._normalize_config(data)

    @staticmethod
    def _normalize_config(data: dict[str, Any]) -> WorkflowConfig:
        compute = WorkflowConfigLoader._mapping_section(data, "compute")
        WorkflowConfigLoader._reject_unknown_fields("compute", compute, WorkflowConfigLoader._COMPUTE_FIELDS)
        if "post_refold_filter" in data:
            raise ValueError("unknown root config field 'post_refold_filter'; move it under design.post_refold_filter")
        WorkflowConfigLoader._reject_unknown_fields("root", data, WorkflowConfigLoader._STRUCTURED_FIELDS)
        benchmark_metadata = data.get("benchmark_metadata") or {}
        if not isinstance(benchmark_metadata, dict):
            raise ValueError("benchmark_metadata must be a YAML mapping")
        design = WorkflowConfigLoader._mapping_section(data, "design")
        fold = WorkflowConfigLoader._mapping_section(data, "fold")
        llm = WorkflowConfigLoader._mapping_section(data, "llm")
        WorkflowConfigLoader._reject_unknown_fields("design", design, WorkflowConfigLoader._DESIGN_FIELDS)
        WorkflowConfigLoader._reject_unknown_fields("llm", llm, WorkflowConfigLoader._LLM_FIELDS)
        WorkflowConfigLoader._reject_unknown_fields("fold", fold, WorkflowConfigLoader._FOLD_FIELDS)
        target = WorkflowConfigLoader._parse_target(data)
        target_name = target.name
        target_chains = target.chains
        target_sequence = target.sequence
        hotspots = target.hotspots

        binders = WorkflowConfigLoader._parse_binders(data, design)
        binder_chains = binders.chains
        fixed_residues = binders.fixed_residues
        explicit_fixed_residues = binders.explicit_fixed_residues
        designable_residues = binders.designable_residues
        cdr_regions = binders.cdr_regions
        cdr_region_groups = binders.cdr_region_groups
        binder_chain_options = binders.chain_options
        binder_chain_type_by_chain = binders.chain_type_by_chain
        initial_candidates = binders.initial_candidates
        overlapping_chain_ids = sorted(set(target_chains) & set(binder_chains))
        if overlapping_chain_ids:
            raise ValueError("target and binder chain IDs must be distinct: " + ", ".join(overlapping_chain_ids))
        mutable_positions = {
            chain_id: [index for index in range(len(sequence)) if index not in set(fixed_residues.get(chain_id, []))]
            for chain_id, sequence in binder_chains.items()
        }
        skill_weights, post = WorkflowConfigLoader._parse_design_policy(design)
        cycles = design.get("n_cycles", 3)
        fold_backend, fold_options, loss_weights = WorkflowConfigLoader._parse_fold(
            fold,
            design,
            target_chains,
            binder_chain_options,
            binder_chains,
            fixed_residues,
            cdr_regions,
            cdr_region_groups,
        )
        return WorkflowConfig(
            target=str(target_name or "unknown"),
            compute_url=compute.get("url"),
            compute_worker_id=compute.get("worker_id"),
            compute_profile=compute.get("profile"),
            placement=WorkflowConfigLoader._parse_placement(compute.get("placement")),
            cycles=int(cycles),
            candidates_per_cycle=int(design.get("num_sequences", 8)),
            fold_backend=fold_backend,
            objective_key=str(design.get("optimization_metric", "loss")),
            minimize=str(design.get("optimization_metric", "loss")).lower() != "iptm",
            reflection_interval=int(design.get("reflection_interval", 20)),
            cycle_retry_limit=int(design.get("cycle_retry_limit", 2)),
            skip_failed_cycles=bool(design.get("skip_failed_cycles", True)),
            initial_candidates=initial_candidates,
            fold_options=fold_options,
            metadata={
                "source_config": data,
                "loss_weights": loss_weights,
                "binder_type": WorkflowConfigLoader._antibody_format(list(binder_chain_type_by_chain.values())),
                "benchmark_metadata": dict(benchmark_metadata),
            },
            target_sequence=target_sequence,
            target_chains=target_chains,
            target_chain_ids=list(target_chains),
            hotspots=hotspots,
            binder_chains=binder_chains,
            fixed_residues=fixed_residues,
            explicit_fixed_residues=explicit_fixed_residues,
            designable_residues=designable_residues,
            cdr_regions=cdr_regions,
            cdr_region_groups=cdr_region_groups,
            mutable_positions=mutable_positions,
            initial_structure_path=design.get("initial_structure_path"),
            seed=int(data.get("seed", 42)),
            population_size=int(design.get("population_size", 20)),
            constrained_min_cdr_distance=float(design.get("constrained_min_cdr_distance", 0.05)),
            constrained_max_position_reuse_fraction=float(design.get("constrained_max_position_reuse_fraction", 0.75)),
            constrained_max_mutation_reuse_fraction=float(design.get("constrained_max_mutation_reuse_fraction", 0.30)),
            parent_selection_strategy=str(design.get("parent_selection_strategy", "fitness")).lower(),
            parent_fitness_temperature_start=float(design.get("parent_fitness_temperature_start", 1.0)),
            parent_fitness_temperature_end=float(design.get("parent_fitness_temperature_end", 0.2)),
            parent_fitness_uniform_fraction=float(design.get("parent_fitness_uniform_fraction", 0.10)),
            quality_check_enabled=bool(design.get("enable_quality_check", True)),
            quality_check_threshold=float(design.get("quality_check_threshold", 0.7)),
            llm_model=llm.get("model_name"),
            llm_temperature=(float(llm["temperature"]) if llm.get("temperature") is not None else None),
            llm_max_tokens=(int(llm["max_tokens"]) if llm.get("max_tokens") is not None else None),
            bootstrap_full_redesign_cycles=int(design.get("bootstrap_full_redesign_cycles", 0)),
            stagnation_full_redesign_threshold=int(design.get("stagnation_full_redesign_threshold", 0)),
            skill_weights=skill_weights,
            esm2_available=bool((skill_weights or {}).get("esm2-guided-mutation", 0.0) > 0.0),
            mutation_count_instruction=f"Use {design.get('num_mutations', 'the configured number of')} CDR mutations.",
            post_filter_enabled=bool(post.get("enabled", False)),
            post_filter_top_k=int(post.get("top_k", 20)),
        )

    @staticmethod
    def _parse_placement(value: Any) -> Placement:
        if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
            return Placement()
        if not isinstance(value, dict):
            raise ValueError("compute.placement must be 'auto' or a mapping of fold/esm/mpnn/cp_degree")
        WorkflowConfigLoader._reject_unknown_fields(
            "compute.placement", value, frozenset({"cp_degree", "esm", "fold", "mpnn"})
        )
        fold = value.get("fold")
        if fold is not None and not isinstance(fold, list):
            raise ValueError("compute.placement.fold must be a list of GPU indices")
        return Placement.model_validate(value)

    @staticmethod
    def _parse_target(data: dict[str, Any]) -> "_TargetSection":
        target = data.get("target")
        if not isinstance(target, dict):
            raise ValueError("target config must be a YAML mapping")
        WorkflowConfigLoader._reject_unknown_fields("target", target, WorkflowConfigLoader._TARGET_FIELDS)
        target_name = target.get("name")
        if not str(target_name or "").strip():
            raise ValueError("target.name must be a non-empty string")
        raw_target_chains = target.get("chains") or {}
        if not isinstance(raw_target_chains, dict) or not raw_target_chains:
            raise ValueError("target.chains must define at least one target chain")
        target_chains: dict[str, Any] = {}
        for chain_id, raw_chain in raw_target_chains.items():
            chain_key = str(chain_id)
            if not chain_key.strip():
                raise ValueError("target chain IDs must be non-empty")
            if chain_key in target_chains:
                raise ValueError(f"duplicate target chain ID after normalization: {chain_key!r}")
            if isinstance(raw_chain, dict):
                WorkflowConfigLoader._reject_unknown_fields(
                    f"target.chains.{chain_key}",
                    raw_chain,
                    WorkflowConfigLoader._TARGET_CHAIN_FIELDS,
                )
                sequence = raw_chain.get("sequence")
            else:
                sequence = raw_chain
            sequence = WorkflowConfigLoader._normalize_sequence(sequence, f"target.chains.{chain_key}.sequence")
            if isinstance(raw_chain, dict):
                normalized = dict(raw_chain)
                normalized["sequence"] = sequence
                normalized["hotspots"] = WorkflowConfigLoader._parse_positions(raw_chain.get("hotspots"), len(sequence))
                target_chains[chain_key] = normalized
            else:
                target_chains[chain_key] = sequence
        target_sequence = "".join(
            str(value.get("sequence", "") if isinstance(value, dict) else value) for value in target_chains.values()
        )
        hotspots = [
            position
            for value in target_chains.values()
            if isinstance(value, dict)
            for position in value.get("hotspots", [])
        ]
        return _TargetSection(
            name=str(target_name or "unknown"),
            chains=target_chains,
            sequence=target_sequence,
            hotspots=hotspots,
        )

    @staticmethod
    def _parse_binders(data: dict[str, Any], design: dict[str, Any]) -> "_BinderSection":
        binder_chains: dict[str, str] = {}
        fixed_residues: dict[str, list[int]] = {}
        explicit_fixed_residues: dict[str, list[int]] = {}
        designable_residues: dict[str, list[int]] = {}
        cdr_regions: dict[str, list[int]] = {}
        cdr_region_groups: dict[str, list[list[int]]] = {}
        binder_chain_options: dict[str, dict[str, Any]] = {}
        binder_chain_type_by_chain: dict[str, str] = {}
        initial_candidates: list[dict[str, Any]] = []
        initial_candidate_ids: set[str] = set()
        expected_binder_chain_ids: set[str] | None = None
        initial_binders = data.get("initial_binders")
        if not isinstance(initial_binders, list) or not initial_binders:
            raise ValueError("initial_binders must define at least one binder")
        for binder_index, binder in enumerate(initial_binders):
            if not isinstance(binder, dict):
                raise ValueError(f"initial_binders[{binder_index}] must be a YAML mapping")
            WorkflowConfigLoader._reject_unknown_fields(
                f"initial_binders[{binder_index}]",
                binder,
                WorkflowConfigLoader._BINDER_FIELDS,
            )
            raw_chains = binder.get("chains")
            if not isinstance(raw_chains, dict) or not raw_chains:
                raise ValueError(f"initial_binders[{binder_index}].chains must define at least one binder chain")
            chains: dict[str, str] = {}
            candidate_chain_types: list[str] = []
            for chain_id, raw in raw_chains.items():
                item = raw if isinstance(raw, dict) else {"sequence": raw}
                WorkflowConfigLoader._reject_unknown_fields(
                    f"initial_binders[{binder_index}].chains.{chain_id}",
                    item,
                    WorkflowConfigLoader._BINDER_CHAIN_FIELDS,
                )
                chain_key = str(chain_id)
                if not chain_key.strip():
                    raise ValueError("binder chain IDs must be non-empty")
                if chain_key in chains:
                    raise ValueError(f"duplicate binder chain ID after normalization: {chain_key!r}")
                sequence = WorkflowConfigLoader._normalize_sequence(
                    item.get("sequence"),
                    f"initial_binders[{binder_index}].chains.{chain_key}.sequence",
                )
                chain_type = WorkflowConfigLoader._normalize_antibody_chain_type(
                    item.get("chain_type"),
                    f"initial_binders[{binder_index}].chains.{chain_key}.chain_type",
                )
                if item.get("cdr_regions") is None and item.get("fixed_residues") is None:
                    raise ValueError(
                        f"initial_binders[{binder_index}].chains.{chain_key} must define "
                        "cdr_regions or fixed_residues; non-antibody protein design is not supported"
                    )
                if chain_key in binder_chains and binder_chains[chain_key] != sequence:
                    raise ValueError(
                        f"initial binders disagree on sequence for chain {chain_key!r}; "
                        "mixed chain topology requires separate tasks"
                    )
                chains[str(chain_id)] = sequence
                binder_chains.setdefault(str(chain_id), sequence)
                chain_options = {key: str(item[key]) for key in ("unpairedMsaPath", "pairedMsaPath") if item.get(key)}
                previous_options = binder_chain_options.get(chain_key)
                if previous_options is not None and previous_options != chain_options:
                    raise ValueError(f"initial binders disagree on MSA inputs for chain {chain_key!r}")
                binder_chain_options[chain_key] = chain_options
                previous_chain_type = binder_chain_type_by_chain.get(chain_key)
                if previous_chain_type is not None and previous_chain_type != chain_type:
                    raise ValueError(f"initial binders disagree on chain_type for chain {chain_key!r}")
                binder_chain_type_by_chain[chain_key] = chain_type
                candidate_chain_types.append(chain_type)
                explicit = WorkflowConfigLoader._parse_positions(item.get("fixed_residues"), len(sequence))
                requested = WorkflowConfigLoader._parse_positions(item.get("designable_residues"), len(sequence))
                cdr = WorkflowConfigLoader._parse_positions(item.get("cdr_regions"), len(sequence))
                groups = WorkflowConfigLoader._parse_position_groups(item.get("cdr_regions"), len(sequence))
                cdr_was_configured = item.get("cdr_regions") is not None
                designable_was_configured = item.get("designable_residues") is not None
                fixed_was_configured = item.get("fixed_residues") is not None
                if cdr_was_configured and not cdr:
                    raise ValueError(f"cdr_regions for chain {chain_key!r} does not select any residues")
                if designable_was_configured:
                    mutable = set(requested)
                elif cdr_was_configured:
                    mutable = set(cdr)
                else:
                    mutable = set(range(len(sequence)))
                mutable.difference_update(explicit)
                # Legacy antibody configs often define the immutable framework
                # and leave the complementary CDR positions implicit.  Keep the
                # structural CDR annotation aligned with those design permissions
                # so the contact gate does not see an empty CDR.
                if not cdr_was_configured and not designable_was_configured and fixed_was_configured:
                    cdr = sorted(mutable)
                    groups = WorkflowConfigLoader._parse_position_groups(cdr, len(sequence))
                if chain_key in fixed_residues:
                    previous = {
                        "explicit": explicit_fixed_residues[chain_key],
                        "designable": designable_residues[chain_key],
                        "cdr": cdr_regions[chain_key],
                        "cdr_groups": cdr_region_groups[chain_key],
                    }
                    current = {
                        "explicit": explicit,
                        "designable": requested,
                        "cdr": cdr,
                        "cdr_groups": groups,
                    }
                    if previous != current:
                        raise ValueError(f"initial binders disagree on residue permissions for chain {chain_key!r}")
                explicit_fixed_residues[chain_key] = explicit
                designable_residues[chain_key] = requested
                cdr_regions[chain_key] = cdr
                cdr_region_groups[chain_key] = groups
                fixed_residues[chain_key] = sorted(set(range(len(sequence))) - mutable)
            WorkflowConfigLoader._validate_antibody_topology(
                candidate_chain_types,
                f"initial_binders[{binder_index}]",
            )
            current_chain_ids = set(chains)
            if expected_binder_chain_ids is None:
                expected_binder_chain_ids = current_chain_ids
            elif current_chain_ids != expected_binder_chain_ids:
                raise ValueError("all initial binders must define the same binder chain IDs")
            candidate_id = str(binder.get("name") or f"initial_{len(initial_candidates)}")
            if candidate_id in initial_candidate_ids:
                raise ValueError(f"duplicate initial binder name: {candidate_id!r}")
            initial_candidate_ids.add(candidate_id)
            initial_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "chains": chains,
                    "sequence": "".join(chains.values()),
                    "structure_path": design.get("initial_structure_path"),
                    "metadata": {"chains": chains},
                }
            )
        return _BinderSection(
            chains=binder_chains,
            fixed_residues=fixed_residues,
            explicit_fixed_residues=explicit_fixed_residues,
            designable_residues=designable_residues,
            cdr_regions=cdr_regions,
            cdr_region_groups=cdr_region_groups,
            chain_options=binder_chain_options,
            chain_type_by_chain=binder_chain_type_by_chain,
            initial_candidates=initial_candidates,
        )

    @staticmethod
    def _parse_design_policy(
        design: dict[str, Any],
    ) -> tuple[dict[str, float] | None, dict[str, Any]]:
        raw_weights = design.get("router_skill_probabilities")
        supported = {
            "cdr-point-mutation",
            "cdr-full-redesign",
            "antibody-inverse-folding",
            "esm2-guided-mutation",
        }
        if raw_weights is None:
            skill_weights = None
        elif isinstance(raw_weights, dict):
            unknown_skills = sorted(set(map(str, raw_weights)) - supported)
            if unknown_skills:
                raise ValueError("unknown design skill weight(s): " + ", ".join(unknown_skills))
            skill_weights = {str(key): float(value) for key, value in raw_weights.items()}
            invalid_weights = sorted(
                key for key, value in skill_weights.items() if not math.isfinite(value) or value < 0.0
            )
            if invalid_weights:
                raise ValueError("design skill weights must be finite and non-negative: " + ", ".join(invalid_weights))
            if not any(value > 0.0 for value in skill_weights.values()):
                raise ValueError("design.router_skill_probabilities must enable at least one skill")
        else:
            raise ValueError("design.router_skill_probabilities must be a YAML mapping")
        post = design.get("post_refold_filter") or {}
        if not isinstance(post, dict):
            raise ValueError("design.post_refold_filter must be a YAML mapping")
        WorkflowConfigLoader._reject_unknown_fields(
            "design.post_refold_filter",
            post,
            WorkflowConfigLoader._POST_FILTER_FIELDS,
        )
        return skill_weights, post

    @staticmethod
    def _parse_fold(
        fold: dict[str, Any],
        design: dict[str, Any],
        target_chains: dict[str, Any],
        binder_chain_options: dict[str, dict[str, Any]],
        binder_chains: dict[str, str],
        fixed_residues: dict[str, list[int]],
        cdr_regions: dict[str, list[int]],
        cdr_region_groups: dict[str, list[list[int]]],
    ) -> tuple[str, dict[str, Any], dict[str, float]]:
        fold_backend = str(fold.get("model") or "opendde").strip().lower()
        if fold_backend != "opendde":
            raise ValueError("fold.model must be 'opendde'; other fold and refold backends are not supported")
        fold_execution_mode = (
            str(fold.get("execution_mode") or os.environ.get("OPENDDE_HARNESS_PROTEIN_FOLD_EXECUTION_MODE", "local"))
            .strip()
            .lower()
        )
        if fold_execution_mode == "api":
            api_url = str(fold.get("api_url") or os.environ.get("OPENDDE_HARNESS_OPENDDE_API_URL", "")).strip()
            if not api_url:
                raise ValueError("fold.api_url is required for fold.execution_mode: api")
            from opendde_harness.plugin.protein_design.servers.backends.opendde_api import (
                normalize_opendde_api_url,
            )

            normalize_opendde_api_url(api_url)
            if (
                str(design.get("optimization_metric", "loss")).lower() == "loss"
                and fold.get("need_atom_confidence") is False
            ):
                raise ValueError("API loss optimization requires fold.need_atom_confidence: true")
            if fold.get("use_msa") is False:
                raise ValueError(
                    "fold.execution_mode: api always runs the service-managed MSA/template "
                    "pipeline; fold.use_msa cannot be false"
                )
            configured_msa_paths = [
                str(value.get(key))
                for value in target_chains.values()
                if isinstance(value, dict)
                for key in ("unpairedMsaPath", "pairedMsaPath")
                if value.get(key)
            ] + [
                str(value.get(key))
                for value in binder_chain_options.values()
                for key in ("unpairedMsaPath", "pairedMsaPath")
                if value.get(key)
            ]
            if configured_msa_paths:
                raise ValueError(
                    "fold.execution_mode: api accepts raw chains and performs MSA remotely; "
                    "local pairedMsaPath/unpairedMsaPath values cannot be uploaded"
                )
        fold_options = {key: value for key, value in fold.items() if key != "model"}
        loss_weights = normalize_loss_weights(design.get("loss_weights"))
        target_hotspots = {
            str(chain_id): list(value.get("hotspots", []))
            for chain_id, value in target_chains.items()
            if isinstance(value, dict) and value.get("hotspots")
        }
        fold_options.update(
            {
                "binder_chain_ids": list(binder_chains),
                "target_chain_ids": list(target_chains),
                # Keep the complete target-chain payload.  In particular,
                # OpenDDE consumes the configured MSA paths from this
                # mapping.  Collapsing it to ``chain -> sequence`` makes the
                # search/refold paths silently run without the user's MSA.
                "target_chains": {
                    str(chain_id): dict(value) if isinstance(value, dict) else str(value)
                    for chain_id, value in target_chains.items()
                },
                "binder_chain_options": binder_chain_options,
                "fixed_residues": fixed_residues,
                "cdr_regions": cdr_regions,
                "cdr_region_groups": cdr_region_groups,
                "loss_weights": loss_weights,
                "target_hotspots": target_hotspots,
                "cdr_contact_fraction_threshold": float(design.get("cdr_contact_fraction_threshold", 0.5)),
                "hotspot_contact_cutoff_a": float(design.get("hotspot_contact_cutoff_a", 5.0)),
                "esm2_options": {"device": design.get("esm_device")},
            }
        )
        return fold_backend, fold_options, loss_weights

    @staticmethod
    def _mapping_section(data: dict[str, Any], name: str) -> dict[str, Any]:
        value = data.get(name)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{name} config must be a YAML mapping")
        return dict(value)

    @staticmethod
    def _normalize_sequence(value: Any, field: str) -> str:
        sequence = "".join(str(value or "").split()).upper()
        if not sequence:
            raise ValueError(f"{field} must be a non-empty string")
        if not sequence.isascii() or not sequence.isalpha():
            raise ValueError(f"{field} must contain only amino-acid letter codes")
        return sequence

    @staticmethod
    def _normalize_antibody_chain_type(value: Any, field: str) -> str:
        if value is None or not str(value).strip():
            raise ValueError(f"{field} is required; this harness only accepts antibody binders")
        key = str(value).strip().upper().replace("_", "-")
        normalized = WorkflowConfigLoader._ANTIBODY_CHAIN_TYPE_ALIASES.get(key)
        if normalized is None:
            supported = "VHH, scFv, VH, or VL"
            raise ValueError(f"{field}={value!r} is not an antibody chain type; supported types are {supported}")
        return normalized

    @staticmethod
    def _validate_antibody_topology(chain_types: list[str], field: str) -> None:
        if len(chain_types) == 1 and chain_types[0] in {"VHH", "scFv"}:
            return
        if len(chain_types) == 2 and sorted(chain_types) == ["VH", "VL"]:
            return
        raise ValueError(
            f"{field} is not a supported antibody topology; use one VHH/scFv chain or one paired VH and VL chain"
        )

    @staticmethod
    def _antibody_format(chain_types: list[str]) -> str:
        if len(chain_types) == 1:
            return chain_types[0]
        return "VH/VL"

    @staticmethod
    def _reject_unknown_fields(
        section: str,
        value: dict[str, Any],
        allowed: frozenset[str],
    ) -> None:
        unknown = sorted(set(map(str, value)) - allowed)
        if unknown:
            raise ValueError(f"unknown {section} config field(s): {', '.join(unknown)}")

    @staticmethod
    def _parse_positions(value: Any, sequence_length: int) -> list[int]:
        positions: set[int] = set()
        if value is None:
            return []
        tokens = value if isinstance(value, (list, tuple)) else str(value).split(",")
        for raw_token in tokens:
            token = str(raw_token).strip()
            if not token:
                continue
            if ":" in token:
                start, end = (int(part) for part in token.split(":", 1))
                if start > end:
                    raise ValueError(f"invalid residue range {token!r}: start is greater than end")
                if start < 0 or end >= sequence_length:
                    raise ValueError(f"residue range {token!r} is outside sequence length {sequence_length}")
                positions.update(range(start, end + 1))
            else:
                position = int(token)
                if position < 0 or position >= sequence_length:
                    raise ValueError(f"residue position {position} is outside sequence length {sequence_length}")
                positions.add(position)
        return sorted(positions)

    @staticmethod
    def _parse_position_groups(value: Any, sequence_length: int) -> list[list[int]]:
        if value is None:
            return []
        if isinstance(value, str):
            return [
                WorkflowConfigLoader._parse_positions(token, sequence_length)
                for token in value.split(",")
                if token.strip()
            ]
        positions = WorkflowConfigLoader._parse_positions(value, sequence_length)
        if not positions:
            return []
        groups: list[list[int]] = [[positions[0]]]
        for position in positions[1:]:
            if position == groups[-1][-1] + 1:
                groups[-1].append(position)
            else:
                groups.append([position])
        return groups
