"""ContextBuilder — assembles system prompt + history for AgentLoop.

Implementation lives in ``builder.py``.

External callers should keep using:

    from opendde_harness.agent.context import ContextBuilder
"""

from opendde_harness.agent.context.builder import ContextBuilder

__all__ = ["ContextBuilder"]
