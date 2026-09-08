"""TokenWise — token accounting around LLM calls.

Public API:
    - ``StrategyRegistry``  — chains TokenStrategy hooks around LLM calls.
    - ``UsageTracker``      — strategy: records tokens + cost per call.
    - ``estimate_cost_usd`` — single source of truth for cost estimation.

This package depends on nothing above it; callers assemble the registry.
"""

from opendde_harness.token_wise.pricing import estimate_cost_usd
from opendde_harness.token_wise.registry import StrategyRegistry
from opendde_harness.token_wise.usage_tracker import UsageTracker

__all__ = [
    "StrategyRegistry",
    "UsageTracker",
    "estimate_cost_usd",
]
