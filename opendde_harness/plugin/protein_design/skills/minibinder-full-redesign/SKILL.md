---
name: minibinder-full-redesign
description: Assign every mutable position of a fixed-length mini binder, including all-X seeds, bootstrap cycles and full-sequence restarts. Not for antibody CDR design or a few point mutations.
---

# Mini Binder Full Redesign

Use the supplied target, hotspots, length, fixed residues and prior evidence to
propose distinct whole-binder sequence hypotheses. Do not invent an existing
fold or measured contacts for a masked seed. There are no CDR/framework rules.

Return the requested Design JSON with `skill_id: minibinder-full-redesign`,
`selection_reason` and the required number of `candidates`. Each candidate has
`candidate_id`, `mutations`, `strategy`, `risk_level` and optional `metadata`.

- Encode assignments as `[chain_id, one_based_position, amino_acid]` triples.
- Assign every mutable position exactly once, including unchanged residues.
  Use only `ACDEFGHIKLMNPQRSTVWY`; resolve every X. Preserve fixed residues,
  chain IDs and length. Do not emit full sequences, indels or target changes.
- Full redesign is not constrained by the point-mutation count budget.
- Diversify actual sequences and structural hypotheses, not just descriptions.
  Favor plausible compact folding and a soluble exterior; ground interface
  choices in supplied target evidence. Do not claim predicted binding is validated.
- Python materializes and validates the assignments before folding. This skill
  generates sequence proposals, not backbone coordinates or folding results.
