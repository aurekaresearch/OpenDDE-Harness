---
name: post-filter
description: |
  Explain hard-gated post-refold candidates from the search trajectory,
  providing advisory evidence alongside deterministic objective selection.
---

# Post-Refold Selection

## Role

Candidates originate from the search trajectory, including candidates no longer
in the population. Python requires fresh successful scoring, canonical sequences,
structures, the same geometry gates as search, and configured conditional quality
checks. Missing required metrics are failures. Parent verdicts are not reused.
Python owns final ordering by the configured objective and direction (`minimize`),
then selects the first `top_k` eligible candidates. For loss optimization the
objective includes all configured PyRosetta contributions.

Weigh interface and fold confidence, target-aligned binder pose RMSD, CDR
engagement, hotspot support, sequence compatibility, measured developability,
and diversity as advisory evidence. Explain tradeoffs without inventing a new
ranking formula. Geometry gates are hard vetoes; commentary cannot override
eligibility, objective ordering, or top K. Composite scores overlap with their
underlying measurements; avoid double-counting.

## Evidence rules

- Use only supplied evidence. Missing measurements remain unknown, not zero,
  positive evidence, a defect, or an automatic ranking penalty.
- Binder RMSD measures pose consistency after target alignment, not affinity.
- Raw contacts depend on length; consider normalized contact fractions.
- Hotspot evidence is informative only when hotspots were configured.
- Use concrete developability measurements or identified sequence features.
- Never invent measurements, interactions, residues, or experimental results.

## Output

- `strategy_summary`: explain evidence and tradeoffs alongside objective ordering.
- `decisions`: every supplied candidate exactly once, with `candidate_id`,
  unique contiguous `rank` starting at 1, `rationale`, `strengths`, and `risks`.
- `risk_notes`: evidence-backed batch-wide risks; use an empty list if none.

Your ranks should follow the configured objective, but are advisory. Python
independently enforces objective ordering even if your ranks differ. Cover the
full supplied batch even when fewer candidates will be selected. If commentary
is unavailable or invalid, the same deterministic selection policy still applies.
