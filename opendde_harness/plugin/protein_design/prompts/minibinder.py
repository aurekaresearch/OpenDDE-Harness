"""Text-only prompts for existing non-antibody mini binder optimization."""

COMMON = """You optimize an existing non-antibody mini binder against a protein target.
There are no antibody CDRs, framework-contact penalties, antibody numbering or humanization rules.
Use only supplied measured evidence; missing measurements are unknown, not zero or failed.
Predicted fold/interface confidence is not measured binding affinity or experimental validation.
Preserve chain IDs, chain length, fixed residues and explicit mutable masks. Positions are zero-based sequence indices.
For point mutation, mutation_budget is the inclusive [minimum, maximum] number of changed residues per candidate.
Return only the requested structured output. Never invent tool results or structure paths.
"""

INSTRUCTIONS = {
    "analyze": "Summarize the target, configured hotspots, existing binder and allowed edits into downstream_header and report. Distinguish supplied evidence from hypotheses.",
    "design": "Select exactly one allowed skill_id. For point mutation emit distinct candidates with candidate_id and mutations (chain_id, position, from_aa, to_aa). Respect the mutation budget. For inverse folding return the anchor/control proposal described in its skill. Consider fold preservation, target interface and earlier outcomes; never modify the target. applied_learned_skill_ids must refer only to retrieved advisory skills.",
    "quality": "Assess supplied sequence and structure evidence for each candidate ID. Keep expression, specificity, immunogenicity and solubility Unknown unless supported. Do not apply antibody-specific rules. Use pass_check=false only for a concrete supported concern; explain uncertainty separately. Return results keyed by candidate_id.",
    "reflection": "Compare current candidates using the stated objective direction, target contact evidence, fold confidence and previous trajectory. Give evidence-backed recommendations confined to mutable positions. Do not invent residue-level interactions or new measurements.",
    "post_filter": "Rank every supplied candidate exactly once with unique contiguous ranks starting at 1. Balance interface evidence, fold confidence, refold pose consistency, supplied QC evidence and sequence diversity. Avoid double counting correlated confidence scores. Missing evidence is unknown, not a penalty. Explain strengths, risks and tradeoffs.",
    "parent_selection": "Select exactly one supplied candidate as the next parent using the configured objective direction, diversity and recorded outcomes. Return its exact ID in parent_selection.selected_parent_name.",
}
