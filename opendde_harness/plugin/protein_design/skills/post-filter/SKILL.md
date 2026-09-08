---
name: post-filter
description: |
  Rank all usable post-refold candidates from the complete search trajectory
  through holistic evidence-based judgment and explain the final order.
---

# Post-Refold Selection

## Role

You own the final ranking. Candidates originate from the complete search
trajectory, including candidates no longer in the population. Python excludes
only unusable refold results and validates your output; it does not calculate
weighted scores, preselect by objective, or reorder for diversity.

Weigh interface and fold confidence, target-aligned binder pose RMSD, CDR
engagement, hotspot support, sequence compatibility, measured developability,
and diversity together. Explain tradeoffs. Gate failures are evidence to
interpret, not an automatic veto. Do not use a fixed weighted formula or let
the search objective alone dictate the order. Composite scores overlap with
their underlying measurements; avoid double-counting.

## Evidence rules

- Use only supplied evidence. Missing measurements remain unknown, not zero,
  positive evidence, a defect, or an automatic ranking penalty.
- Binder RMSD measures pose consistency after target alignment, not affinity.
- Raw contacts depend on length; consider normalized contact fractions.
- Hotspot evidence is informative only when hotspots were configured.
- Use concrete developability measurements or identified sequence features.
- Never invent measurements, interactions, residues, or experimental results.

## Output

- `strategy_summary`: explain the evidence and tradeoffs behind the order.
- `decisions`: every supplied candidate exactly once, with `candidate_id`,
  unique contiguous `rank` starting at 1, `rationale`, `strengths`, and `risks`.
- `risk_notes`: evidence-backed batch-wide risks; use an empty list if none.

Python takes the first `top_k` candidates from your order. Your ranking must
cover the full supplied batch even when fewer candidates will be selected.
