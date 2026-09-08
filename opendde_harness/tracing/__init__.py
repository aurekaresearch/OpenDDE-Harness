"""OpenDDE Harness in-tree tracing: audit.span.v1 observability.

Instrumentation is done by annotating opendde's own methods with the
``@trace.instrument(...)`` decorator (see :mod:`opendde_harness.tracing.trace` and the
standard in ``docs/TRACING_STANDARD_API.md``). Nothing is monkeypatched — the
decorators live in opendde's source and are no-op when tracing is disabled, so a
tracing failure can never alter the host's behavior.

Turn off with ``OPENDDE_HARNESS_TRACING=0`` or ``[tracing] enabled = false`` in the ddeharness
config. Spans land at ``~/.opendde_harness/traces/logs/audit-spans.log`` (override with
``OPENDDE_HARNESS_TRACING_DIR``). Open the dashboard with ``ddeharness tracing`` or ``/tracing``.
"""

from __future__ import annotations

from . import config, trace

__all__ = ["enabled", "trace"]


def enabled() -> bool:
    return config.enabled()
