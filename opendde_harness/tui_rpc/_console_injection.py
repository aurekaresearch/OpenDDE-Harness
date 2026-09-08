"""Swap the module-level ``console`` of every CLI command module.

The CLI modules each define ``console = Console()`` at import time and
resolve ``console.print(...)`` by module-global lookup, so ``cli.dispatch``
can capture their output by temporarily replacing that global instead of
threading a console argument through every command signature.

Concurrency: this context manager is NOT internally locked. The cli.dispatch
handler holds a module-level ``asyncio.Lock`` to serialize calls.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from rich.console import Console

import opendde_harness.cli._helpers as ec_helpers
import opendde_harness.cli.commands as ec_commands
import opendde_harness.cli.onboard_commands as ec_onboard
import opendde_harness.cli.provider_commands as ec_provider
import opendde_harness.cli.skill_commands as ec_skill
import opendde_harness.cli.status_commands as ec_status

_CONSOLE_HOSTS: tuple = (
    ec_commands,
    ec_onboard,
    ec_provider,
    ec_skill,
    ec_status,
    ec_helpers,
)


@contextlib.contextmanager
def inject_consoles(out_console: Console) -> Iterator[None]:
    """Temporarily replace module-level ``console`` on all CLI modules.

    ``out_console`` is the Rich ``Console`` the commands write to for the
    duration of the context, typically ``Console(file=StringIO(),
    force_terminal=True, color_system="truecolor", width=<TUI-supplied>)``.
    The original reference is restored on exit regardless of how the
    context body terminated. stderr is captured out-of-band by the
    handler's ``contextlib.redirect_stderr`` wrapping this context.
    """
    originals = {mod: mod.console for mod in _CONSOLE_HOSTS}
    try:
        for mod in _CONSOLE_HOSTS:
            mod.console = out_console
        yield
    finally:
        for mod, orig in originals.items():
            mod.console = orig


__all__ = ["inject_consoles", "_CONSOLE_HOSTS"]
