"""Self-contained OpenDDE Harness protein-design plugin."""

from opendde_harness.plugin.protein_design.core.orchestrator import DesignOrchestrator
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient

__all__ = ["DesignOrchestrator", "ProteinDesignComputeClient"]
