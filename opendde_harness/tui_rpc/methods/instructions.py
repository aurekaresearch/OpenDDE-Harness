"""``session.instructions`` — what AGENTS.md / ODH.md the next turn will send.

Backs the ``/memory`` command: a list of the files found, what each costs, and
whether it is switched on. The list travels; the contents never do. The TUI has
no use for the bodies, and a session-info payload that carried 128 KiB of
instructions on every refresh would be a strange thing to pay for.

Switching a file off belongs to one conversation and lasts as long as the
gateway process. It is how someone says "not for this conversation", so it is
keyed by session id and deliberately not written to any config file.

Each file also reports whether its content has moved since this conversation
first saw it. The agent can write files, and a file with one of these names
becomes standing instructions wherever it lands, so a change that did not come
from the person at the keyboard is worth a line rather than a silent reload.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opendde_harness.context_engine import project_instructions

if TYPE_CHECKING:
    from opendde_harness.config.opendde_harness import Config
    from opendde_harness.tui_rpc.dispatcher import Dispatcher


def _bootstrap_paths(config: "Config") -> set[Path]:
    """The files segment 2 already renders, which this segment must not repeat."""
    from opendde_harness.config.paths import get_workspace_storage
    from opendde_harness.context_engine.segments.render import BOOTSTRAP_FILES

    workspace = get_workspace_storage(config.workspace_path).assistant

    return {workspace / name for name in BOOTSTRAP_FILES}


def wire_files(files: list[project_instructions.InstructionFile]) -> list[dict[str, Any]]:
    """The list as the UI receives it. Paths and sizes only, never bodies."""
    return [
        {
            "path": file.path,
            "display": file.display,
            "size": file.size,
            "truncated": file.truncated,
            "skipped": file.skipped,
            "enabled": file.enabled,
            "changed": file.changed,
        }
        for file in files
    ]


def current_files(config: "Config", session_key: str = "") -> list[project_instructions.InstructionFile]:
    """What the next turn of ``session_key`` would send, from the launch directory."""
    return project_instructions.current(
        Path(os.getcwd()),
        session_key=session_key,
        already_loaded=_bootstrap_paths(config),
    )


async def session_instructions(params: dict) -> dict:
    """``session.instructions`` — list the files, or switch one on or off.

    ``action`` defaults to listing, so the common call carries no params. A
    toggle that names nothing the search found changes nothing and says so:
    silently doing nothing would be discovered a turn later, when the model
    still followed a file the user thought they had switched off.

    ``session_id`` is whose view this is. Without it the answer describes the
    defaults, and a toggle applies to that unnamed session rather than leaking
    into every conversation this gateway serves.
    """
    from opendde_harness.config.loader import load_config

    config = load_config()
    action = str(params.get("action") or "list").strip().lower()
    # Whose view this is. Absent means the defaults a session would start on,
    # which is what a caller asks for before one is open.
    session_key = str(params.get("session_id") or "")
    files = current_files(config, session_key)
    error: str | None = None
    changed: str | None = None

    if action in ("on", "off"):
        wanted = str(params.get("path") or "").strip()

        if not wanted:
            error = f"/memory {action} needs the path of a file from the list"
        else:
            match = project_instructions.match(wanted, files)

            if match is None:
                same_name = [f.display for f in files if Path(f.path).name == wanted]

                if len(same_name) > 1:
                    # Several directories each have an AGENTS.md, which is the
                    # normal case in a repository that uses them. Guessing one
                    # would switch off a file the user can still see listed.
                    error = f"{wanted} names {len(same_name)} files here: " + ", ".join(same_name)
                else:
                    error = f"no instruction file here is called {wanted}"
            else:
                project_instructions.SESSIONS.set(session_key, match.path, enabled=action == "on")
                changed = match.display
                files = current_files(config, session_key)
    elif action != "list":
        error = f"unknown action {action}"

    return {
        "cwd": os.getcwd(),
        "files": wire_files(files),
        "changed": changed,
        "error": error,
    }


def register_instructions_methods(dispatcher: "Dispatcher") -> None:
    """Register ``session.instructions`` on a dispatcher instance."""
    dispatcher.register("session.instructions", session_instructions)


__all__ = [
    "current_files",
    "register_instructions_methods",
    "session_instructions",
    "wire_files",
]
