---
name: cdr-point-mutation
description: Design a bounded number of targeted substitutions or single-residue CDR indels on an existing antibody parent. Use when the Router exposes point-mutation for exploit or explore. Do not use for complete CDR replacement.
---

# CDR Point Mutation

1. Read the antibody header, mutable positions, parent-selection mode, prior
   memory, QC warnings, and exact mutation-count requirement.
2. Tie every change to a named CDR and mechanism. Preserve demonstrated parent
   contacts unless a concrete hypothesis supports changing them.
3. In exploit mode, prefer local conservative changes and the lower allowed
   mutation count. In explore mode, test distinct hypotheses across multiple
   CDRs and prefer the upper allowed count.
4. Treat H3 as the most permissive geometry-tuning region. Respect canonical
   structure in H1/H2 and use L1/L2 conservatively. Avoid mutating native
   cysteine or introducing cysteine without compelling evidence.
5. Use a single-residue insertion or deletion only when loop length is the
   explicit hypothesis. H3 is most tolerant; L2 is most constrained.

## Output contract

Return exactly one JSON object with no prose or code fence before or after it:

```json
{
  "skill_id": "cdr-point-mutation",
  "selection_reason": "Local attributable CDR changes best match the current search state.",
  "candidates": [
    {
      "id": "cycle_N_mut_001",
      "mutations": [["H", 100, "W"]],
      "strategy": "[H3 / interface] concise mechanism",
      "risk_level": "low",
      "metadata": {"hypothesis": "strengthen H3 packing"}
    }
  ]
}
```

- Mutate only listed mutable positions; never mutate framework/fixed residues.
- Return the Router-selected `cdr-point-mutation` once as top-level `skill_id`.
- Context providers are automatic evidence sources; do not declare them in candidates.
- Encode substitutions as `[chain, pos, "A"]`, insertions as
  `[chain, pos, "+A"]`, and deletions as `[chain, pos, "-"]`.
- Use zero-based positions. Do not emit no-ops or duplicate a target position
  within one candidate.
- Match the exact requested candidate and mutation counts.
- Put skill-specific evidence only in the `metadata` JSON object.
- Prefix each strategy with `[CDR / mechanism_class]`. EX : `shape`,
  `electrostatic`, `rigidification`, `hydrophobic-pack`, `germline-revert`,
  `SHM-mimic`, `interface`, `cdr-h3-reshape`... You can propose any new mechanism_class.
