"""Prompt templates for OpenDDE Harness protein-design agents."""

from opendde_harness.plugin.protein_design.prompts.analyze import ANALYZE_REPORT_PROMPT, ANALYZE_SYSTEM_PROMPT
from opendde_harness.plugin.protein_design.prompts.design import DESIGN_PROMPT, DESIGN_SYSTEM_PROMPT
from opendde_harness.plugin.protein_design.prompts.parent_selection import (
    PARENT_SELECTION_PROMPT,
    PARENT_SELECTION_SYSTEM_PROMPT,
)
from opendde_harness.plugin.protein_design.prompts.post_filter import POST_FILTER_AGENT_INSTRUCTIONS
from opendde_harness.plugin.protein_design.prompts.quality_check import (
    QUALITY_CHECK_BATCH_PROMPT,
    QUALITY_CHECK_SYSTEM_PROMPT,
)
from opendde_harness.plugin.protein_design.prompts.reflect import REFLECT_ANALYSIS_PROMPT, REFLECT_SYSTEM_PROMPT

__all__ = [
    "ANALYZE_REPORT_PROMPT",
    "ANALYZE_SYSTEM_PROMPT",
    "DESIGN_PROMPT",
    "DESIGN_SYSTEM_PROMPT",
    "POST_FILTER_AGENT_INSTRUCTIONS",
    "PARENT_SELECTION_PROMPT",
    "PARENT_SELECTION_SYSTEM_PROMPT",
    "QUALITY_CHECK_BATCH_PROMPT",
    "QUALITY_CHECK_SYSTEM_PROMPT",
    "REFLECT_ANALYSIS_PROMPT",
    "REFLECT_SYSTEM_PROMPT",
]
