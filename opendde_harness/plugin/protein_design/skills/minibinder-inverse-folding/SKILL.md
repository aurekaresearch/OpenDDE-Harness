---
name: minibinder-inverse-folding
description: Select constrained SolubleMPNN sequence redesign for an existing mini binder with an available parent structure.
---

# Mini binder inverse folding

Requires the actual parent structure path supplied in context. Select this skill when backbone-conditioned sequence sampling is appropriate. Never invent a structure path or reconstruct it from a filename.

Return one anchor candidate with `mutations: []` and optional `metadata.soluble_mpnn_parameters`. Supported controls are `temperature` (0.01–1), `num_sequences`, `relax_radius` (0–8), `wt_bias` (0–20), `omit_aas`, `bias_aas`, and `design_positions` (chain to zero-based position list). Omit controls when no evidence supports changing their defaults.

Design positions must be a subset of the supplied mutable mask. The executor owns chain mapping, fixed residues, chain length, deduplication and total candidate count. It generates sequences; do not claim that generation or validation already happened. Re-fold and evaluate candidates before interpreting them as improvements. This changes sequence on an existing backbone, not the backbone topology.
