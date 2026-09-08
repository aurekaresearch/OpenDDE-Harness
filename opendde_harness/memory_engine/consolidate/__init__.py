"""Two-layer long-term memory (MEMORY.md + HISTORY.md).

Migrated from ``opendde_harness.memory_engine.consolidate.consolidator``. The implementation —
``MemoryStore`` (read/write under fcntl lock) + ``MemoryConsolidator``
(boundary-aware token-driven compaction) — is unchanged; only the import
path and physical home moved as part of the EverBrain L4 consolidation.
"""

from opendde_harness.memory_engine.consolidate.consolidator import (
    MemoryConsolidator,
    MemoryStore,
)

__all__ = ["MemoryStore", "MemoryConsolidator"]
