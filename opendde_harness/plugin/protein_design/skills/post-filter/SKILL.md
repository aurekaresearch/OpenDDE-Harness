---
name: post-filter
description: |
  Select a concise Top-K from a bounded post-refold shortlist
  through holistic evidence-based judgment.
---

# Post-Refold Selection

## Role

You own the final ranking. Candidates originate from the complete search
trajectory, including candidates no longer in the population. Python excludes
unusable refold results, then builds a bounded shortlist by round-robin metric
leaders and sequence diversity. This is a context budget, not a quality gate
or final ranking. Python validates your selection without rescoring or reordering it.

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

- `strategy_summary`: explain the evidence and tradeoffs within 600 characters.
- `decisions`: exactly `top_k` distinct supplied candidates, with `candidate_id`,
  unique contiguous `rank` from 1 to `top_k`, `rationale` (at most 240 characters),
  and at most two `strengths` and two `risks` (160 characters per item).
- `risk_notes`: at most eight evidence-backed batch-wide risks, 240 characters
  per note; use an empty list if none.

Do not rank or explain unselected candidates. Never choose an ID not supplied
in the shortlist. Excerpts marked truncated are incomplete evidence, not defects.
Python preserves all original candidates and records who was not shortlisted
separately from hard eligibility failures.
