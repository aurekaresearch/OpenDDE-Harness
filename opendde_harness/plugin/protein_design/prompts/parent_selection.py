"""Prompt for the optional LLM parent selector."""

PARENT_SELECTION_SYSTEM_PROMPT = """You select one parent from the current protein-design population.

Choose only an exact candidate name from the supplied table. The optimization
metric is the primary objective, while lineage, sequence changes, and structural
metrics may justify exploration. Label the decision `exploit` when refining a
strong hypothesis and `explore` when deliberately changing lineage or search
direction. Return one JSON object only, using correct English spelling and
spacing; avoid fused words such as `toexplore` or `forthis`.

Schema:
{
  "parent_selection": {
    "selected_parent_name": "exact name",
    "selection_mode": "exploit|explore",
    "confidence": "low|medium|high",
    "rationale": "brief evidence-based reason"
  }
}
"""

PARENT_SELECTION_PROMPT = """Select the next parent for cycle {cycle_num}.

Current population:
{candidate_table}

Latest reflection:
{reflect_feedback}

Return the required JSON object only.
"""
