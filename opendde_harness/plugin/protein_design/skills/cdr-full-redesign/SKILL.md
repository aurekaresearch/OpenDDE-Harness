---
name: cdr-full-redesign
description: Redesign every system-approved mutable antibody CDR region at native length. Use when the Design Agent route specifies full_redesign, including masked initial CDRs, bootstrap cycles, or stagnation-triggered restarts. Do not use for a few targeted substitutions.
---

# CDR Full Redesign

1. Read the target, antibody header, mutable-position list, prior memory, and
   QC warnings. Treat chain IDs and zero-based positions as immutable input.
2. Form several distinct region-level hypotheses grounded in the supplied
   epitope and structural evidence. Preserve realistic antibody loop chemistry,
   canonical constraints outside H3, VH/VL balance, and framework anchors.
3. Assign one amino acid to every listed mutable CDR position for every
   candidate. Include unchanged parent residues as explicit assignments so the
   validator can verify complete CDR coverage.
4. Diversify mechanisms and risk. Favor H3 reshaping when justified; use H1/H2
   and light-chain CDR changes conservatively unless evidence supports broader
   changes. Avoid unjustified alanine runs and new or removed cysteines.
5. Make the actual CDR assignments diverse, not merely their prose. Every pair
   of candidates must differ at three or more mutable positions (or at every
   mutable position when fewer than three exist). Never copy one complete CDR
   assignment and relabel it with another strategy or score.
6. Do not output full binder sequences or region fills. The pipeline applies
   the validated position assignments to the parent sequence.

## Output contract

Return exactly one JSON object with no prose or code fence before or after it:

```json
{
  "skill_id": "cdr-full-redesign",
  "selection_reason": "A complete CDR basin reset best matches the current parent state.",
  "candidates": [
    {
      "id": "cycle_N_redesign_001",
      "mutations": [["D", 95, "A"], ["D", 96, "R"], ["D", 97, "N"]],
      "strategy": "[H3 / cdr-h3-reshape] concise mechanism",
      "risk_level": "high",
      "metadata": {"hypothesis": "reshape the complete H3 paratope"}
    }
  ]
}
```

- The example chain `D` is illustrative. Copy the configured chain ID from the
  current prompt exactly; never translate a VHH chain to `H`.
- `mutations` must be a JSON array of three-item arrays. Never compress it into
  a string such as `D:25-34=SEQUENCE`, a mutation string such as `D:T99V`, or a
  mapping.
- Include every listed mutable CDR position exactly once. Explicit no-op
  assignments are allowed only here to prove complete coverage.
- Candidate mutation arrays must be pairwise distinct and must yield pairwise
  distinct sequences; metadata and scores are not diversity.
- Return the Router-selected `cdr-full-redesign` once as top-level `skill_id`.
- Context providers are automatic evidence sources; do not declare them in candidates.
- Use only `ACDEFGHIKLMNPQRSTVWY`; do not use insertions or deletions.
- Put skill-specific evidence only in the `metadata` JSON object.
- Prefix each strategy with `[CDR / mechanism_class]`, using one of `shape`,
  `electrostatic`, `rigidification`, `hydrophobic-pack`, `germline-revert`,
  `SHM-mimic`, `interface`, or `cdr-h3-reshape`.
