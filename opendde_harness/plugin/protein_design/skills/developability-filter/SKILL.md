---
name: developability-filter
description: Use when an agent or workflow needs objective antibody pI, TAP, and CDR liability filtering
---

# Developability Filter

## Overview

Use this skill to assess antibody developability — pI, TAP, and CDR liability
checks. When the `check_antibody_developability` tool is available it delegates
to a compute backend; when the backend is not installed the agent falls back to
LLM-based assessment using sequence properties.

## Tool

Call the registered `check_antibody_developability` tool when chain roles are
identifiable. The compute service owns implementation details and output paths;
do not import a backend module or construct an output directory in the agent
prompt.

## Input CSV

Minimum for paired antibodies:

```csv
name,heavy_chain,light_chain,wt_name
demo_wt,HEAVY_SEQ,LIGHT_SEQ,demo_wt
demo_design,HEAVY_SEQ,LIGHT_SEQ,demo_wt
```

For de novo designs, `wt_name` can be blank when SoluProt is skipped.

## Built-In Filters (when backend is available)

- pI: built in.
- Liability: built in, including CDR glycosylation and cysteine flags.
- SoluProt: skipped by the project tool.
- TAP: enabled when a compatible structure and chain mapping are provided.
- BioPhi/OASis: skipped by the project tool.

## Agent Usage

When the workflow already supplies a batch result for every candidate, use
those rows directly and do not repeat the compute call. Call the direct tool
only when objective batch evidence is absent and a new measurement is needed.

Assess these six antibody-specific dimensions independently for every
candidate:

1. **Expressivity** — free or unpaired Cys, long CDR hydrophobic stretches,
   gross VH/VL pI mismatch, and disruption of conserved V-domain residues.
   Do not use codon or GC arguments when only protein sequence is supplied.
2. **Immunogenicity** — framework agreement with the configured scaffold,
   naturalness of CDR substitutions for antibody sequence context, and unusual
   hydrophobic CDR peptides.
3. **Aggregation** — overall hydrophobic content, CDR hydrophobic patches
   (especially H3), amyloid-like motifs, and unusual loop beta propensity.
4. **Solubility** — total and per-CDR charge, VH/VL pI mismatch, large positive
   H3 patches, and extreme CDR charge.
5. **Specificity** — H3 positive or hydrophobic/aromatic clusters, combined-CDR
   positive charge, and obvious cross-reactive motifs.
6. **Liability** — annotate supported hits by CDR and position: deamidation,
   isomerization, N-glycosylation, Met/Trp oxidation context, unpaired Cys, and
   acid-cleavage DP motifs.

Use chain roles and zero-based inclusive CDR ranges from the supplied antibody
identification evidence. If chain identity remains uncertain, make the affected
dimensions Medium Risk and explain the uncertainty.

Decision rule:

- any High Risk dimension -> overall High Risk and fail;
- otherwise any Medium Risk dimension -> overall Medium Risk and pass;
- all Low Risk dimensions -> overall Low Risk and pass.

Return one result for every supplied candidate ID and do not add candidates.

### When the backend is available (`available` is not `False`)

Reconcile the tool result:

- `PI_filter=fail` → raise solubility/formulation concern.
- `liability_filter=fail` → raise liability concern.
- `all_filter_pass=pass` is supportive evidence, not a substitute for full reasoning.

### When the backend is unavailable (result contains `available: False`)

The developability compute service is not installed. Apply LLM-based judgment
using sequence properties:

1. **pI estimation** — count charged residues (Arg, Lys, His vs Asp, Glu) in
   the heavy and light chains. Flag candidates likely outside pH 5–8 as
   formulation risks.
2. **CDR liability scan** — inspect CDR loops (H1, H2, H3, L1, L2, L3) for
   known liability motifs: NG/NS deamidation sites, unpaired cysteines, exposed
   Met/Trp oxidation hotspots, and N-glycosylation sequons (N-X-S/T).
3. **Sequence composition** — very high hydrophobicity patches in CDR-H3 or
   framework regions suggest aggregation risk.
4. Summarize findings as `pi_concern`, `liability_concern`, and
   `overall_developability_risk` (low / medium / high) and include the
   reasoning in the output.

## Common Mistakes

- Do not call this for target chains; it is for antibody binder chains.
- Do not omit the light chain for paired antibodies.
- Do not expect SoluProt or BioPhi/OASis unless those optional checks are enabled.
- Do not skip the assessment when `available: False` — fall back to LLM judgment instead.
- Do not recompute or override objective batch rows supplied by the workflow.
