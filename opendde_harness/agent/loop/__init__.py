"""AgentLoop — the OpenDDE Harness L2 ReAct executor.

The full ``AgentLoop`` implementation lives in ``main.py``. The package
shape is in place so the file can later be split into ``main.py`` /
``dispatch.py`` / ``runner.py`` without further import churn.

External callers should keep using:

    from opendde_harness.agent.loop import AgentLoop

which re-exports through here.
"""

from opendde_harness.agent.loop.main import AgentLoop, LoopOutcome

__all__ = ["AgentLoop", "LoopOutcome"]
