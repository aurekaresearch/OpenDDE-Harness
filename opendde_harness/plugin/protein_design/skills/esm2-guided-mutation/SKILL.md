---
name: esm2-guided-mutation
description: Generate one-step CDR mutation proposals from ESM-2 deep-mutational-scan log-likelihood ratios, then pass those proposals to the normal fold and gate pipeline.
---

# ESM-2 Guided Mutation

Use this as the primary proposal skill only when the parent is a complete, materialized sequence and the Router exposes it.

The runtime performs the ESM-2 DMS and materializes the highest positive LLR
substitutions at mutable CDR positions. The Design Agent must still select this
skill explicitly and provide a valid top-level portfolio. ESM-2 is a sequence
plausibility model, not an affinity predictor.

## Output contract

The Agent only selects this Python-executed skill. Return exactly one JSON
object with no prose or code fence before or after it:

```json
{
  "skill_id": "esm2-guided-mutation",
  "selection_reason": "ESM-2 sequence-plausibility rescue best matches the current bottleneck.",
  "candidates": []
}
```

Do not invent mutations or placeholder candidates; Python performs the DMS and
materializes the proposals.

## Contract

- Python proposes substitutions only at the listed mutable positions using the
  configured semantic chain IDs and zero-based positions.
- The runtime records `esm2_llr`, the parent residue, and the selected rank in
  candidate metadata before folding.

## Guardrails

- Never use this skill for a parent containing `X` or an unmaterialized CDR;
  full redesign must materialize the sequence first.
- A positive LLR does not imply better antigen binding. Keep OpenDDE metrics,
  contact-fraction gates, and developability checks authoritative.
- Treat the generated mutations as proposals; all candidates still undergo
  folding, admission, and population selection.
