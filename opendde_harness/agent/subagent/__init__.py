"""SubagentManager — spawns child AgentLoops for delegated tasks.

Implementation lives in ``manager.py``.

External callers should keep using:

    from opendde_harness.agent.subagent import SubagentManager
"""

from opendde_harness.agent.subagent.manager import SubagentManager

__all__ = ["SubagentManager"]
