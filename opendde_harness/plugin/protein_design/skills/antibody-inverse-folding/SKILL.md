---
name: antibody-inverse-folding
description: Perform structure-conditioned antibody sequence generation with SolubleMPNN as one atomic Design Agent skill while preserving fixed framework residues and optional forced anchors.
---

# Antibody Inverse Folding

Use this as the only primary skill when the Router selects the atomic
`antibody-inverse-folding` skill. Router-approved context providers may also
be loaded, but no other primary design skill may be composed with it.

1. Require a valid parent/backbone structure, binder chain mapping, parent
   sequence, and mutable/fixed positions. Never invent missing chain mappings.
2. Propose explicit, diverse CDR anchor substitutions. For every anchor, choose
   SolubleMPNN sampling controls in `metadata.soluble_mpnn_parameters` based on the current
   bottleneck instead of assuming one fixed setting:
   `temperature`, `num_sequences`, `relax_radius`, `wt_bias`,
   `omit_aas`, and optional global `bias_aas`. To choose an exact CDR subset,
   provide `design_positions` as a chain-to-zero-based-position object; it is
   always intersected with the authoritative mutable CDR list. Python validates
   these values, caps the cycle-wide work, hard-fixes each forced anchor, and
   never permits framework positions to become designable.
3. Treat SolubleMPNN as the sequence generator, not as evidence that a candidate
   binds. Preserve the strategy skill's hypothesis and let downstream folding,
   CDR-RMSD, interface, and developability gates evaluate the sample.
4. Keep chain identities and zero-based sequence-position semantics consistent
   when translating to structure residue numbering.

## Output contract

Return exactly one JSON object with no prose or code fence before or after it.
Use the same candidate JSON shape as every other primary design skill:

```json
{
  "skill_id": "antibody-inverse-folding",
  "selection_reason": "Structure-conditioned local resampling best matches the current bottleneck.",
  "candidates": [
    {
      "id": "cycle_N_inverse_001",
      "mutations": [["D", 103, "F"]],
      "strategy": "[H3 / inverse-folding] structure-conditioned local redesign",
      "risk_level": "medium",
      "metadata": {
        "anchor_rationale": "stabilize the H3 tip",
        "soluble_mpnn_parameters": {
          "temperature": 0.25,
          "num_sequences": 4,
          "relax_radius": 3,
          "wt_bias": 2.0,
          "omit_aas": "",
          "bias_aas": {"Y": 0.5},
          "design_positions": {"D": [101, 102, 103]}
        }
      }
    }
  ]
}
```

## Pipeline rules

- Do not execute shell commands or fabricate SolubleMPNN scores. The Python runner
  owns model execution through `generate_soluble_mpnn` and the
  `opendde_harness.plugin.protein_design.servers.backends.soluble_mpnn` backend.
  This uses LigandMPNN's `soluble_mpnn` model, not its ligand-conditioned model.
  SolubleMPNN scores are mean negative log probabilities over designed residues
  (lower is better); confidence is `exp(-score)` (higher is better).
- Return mutation-style JSON anchors under top-level
  `"skill_id": "antibody-inverse-folding"`. The runner uses them as independent
  structure-conditioned generation hypotheses.
- Encode every anchor in the canonical zero-based three-item array form
  `["D", 103, "F"]` inside `"mutations"`. Do not emit mutation objects with
  `chain`, `position`, `from`, or `to` keys, and do not emit compact strings
  such as `D103F`. The runner accepts those forms only as compatibility
  fallbacks.
- Context providers are automatic evidence sources; do not declare them in candidates.
- Preserve `fixed_residues`, `design_method`, and chain mapping in generated
  candidate metadata.
- Put skill-specific proposal evidence only in the `metadata` JSON object.
- If the required structure or SolubleMPNN client is unavailable, report that the
  inverse-folding route cannot execute; do not silently relabel an LLM sequence
  as inverse-folded.
