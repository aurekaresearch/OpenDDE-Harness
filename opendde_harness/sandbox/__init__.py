"""Command execution for the ``exec`` tool.

Public API (import everything from here, not from sub-modules):
    SandboxInitError   — raised when an executor fails to start
    ExecResult         — result of a single exec() call
    SandboxExecutor    — ABC for executor implementations
    DirectExecutor     — host-process executor (no isolation)
"""

from __future__ import annotations

from opendde_harness.sandbox.direct_executor import DirectExecutor
from opendde_harness.sandbox.interfaces import ExecResult, SandboxExecutor, SandboxInitError

__all__ = [
    "ExecResult",
    "SandboxExecutor",
    "SandboxInitError",
    "DirectExecutor",
]
