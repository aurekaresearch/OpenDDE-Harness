---
name: minibinder-point-mutation
description: Propose bounded sequence mutations to an existing non-antibody mini binder using its measured interface and fold evidence.
---

# Mini binder point mutation

Use only the provided chain IDs and zero-based mutable sequence positions. Preserve chain length, every fixed residue, and the configured mutation budget. Output the requested number of distinct candidates using the supplied schema; each mutation uses `chain_id`, `position`, `from_aa`, and `to_aa`.

Choose mutations from current target-interface, hotspot, fold-confidence and prior-cycle evidence. Distinguish interface improvement from disruption of the binder core. Do not infer buried positions or specific contacts without structural evidence. Preserve user-specified motifs; there are no CDRs, antibody frameworks or humanization objectives in this mode.

Describe trade-offs and uncertainty. Predicted confidence is not measured affinity, solubility or experimental success. This skill optimizes an existing sequence; it does not generate new backbones or change sequence length.
