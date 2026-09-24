"""Render a stored session into a human-readable Markdown transcript.

Pure rendering (``render_transcript``) is separated from the file write
(``write_transcript``) so every export surface — the ``session.export`` RPC,
the TUI ``/export`` slash command, and the CLI ``session export`` — shares one
rendering and one destination convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opendde_harness.providers import messages as msg
from opendde_harness.session.manager import Session
from opendde_harness.utils.helpers import ensure_dir, safe_filename

_ROLE_HEADINGS = {
    msg.USER: "## 🧑 User",
    msg.ASSISTANT: "## 🤖 Assistant",
    msg.SYSTEM: "## ⚙️ System",
    msg.TOOL_RESULT: "## 🛠 Tool result",
}


def render_transcript(session: Session) -> str:
    """Render ``session`` to a full-fidelity Markdown transcript.

    Includes a header (key, timestamps, message count, title when set) and each
    message in order: user/assistant/system/tool under distinct headings, the
    assistant reasoning block when present, and tool calls/results as fenced
    blocks. Pure — performs no I/O.
    """
    parts: list[str] = [_render_header(session)]
    for message in session.messages:
        parts.append(_render_message(message))
    return "\n\n".join(p for p in parts if p) + "\n"


def default_export_path(workspace: Path, key: str) -> Path:
    """Default destination in the instance's workspace-scoped exports directory.

    The session key's ``:`` is folded to a filesystem-safe name via
    ``safe_filename`` (same encoding the session store uses for its files).
    """
    from opendde_harness.config.paths import get_workspace_storage

    return get_workspace_storage(workspace).exports / f"{safe_filename(key)}.md"


def write_transcript(session: Session, dest: Path) -> Path:
    """Render ``session`` and write it to ``dest``, returning the absolute path.

    Creates the parent directory if absent and overwrites any existing file so
    a re-export reflects the session's current state.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    dest.write_text(render_transcript(session), encoding="utf-8")
    return dest.resolve()


# ── internals ──────────────────────────────────────────────────────────


def _render_header(session: Session) -> str:
    title = (session.metadata or {}).get("title")
    meta = (
        f"_{session.created_at.isoformat(timespec='seconds')}"
        f" → {session.updated_at.isoformat(timespec='seconds')}"
        f" · {len(session.messages)} messages_"
    )
    lines = [f"# Session `{session.key}`", meta]
    if title:
        lines.insert(1, f"**{title}**")
    return "\n".join(lines)


def _render_message(message: dict[str, Any]) -> str:
    role = message.get("role", "")
    heading = _ROLE_HEADINGS.get(role, f"## {role or 'message'}")
    if msg.is_tool_result(message):
        name = message.get("toolName") or message.get("toolCallId") or ""
        suffix = f": `{name}`" if name else ""
        return f"{heading}{suffix}\n\n{_fenced(_content_text(message))}"

    parts: list[str] = [heading]
    reasoning = msg.thinking_of(message)
    if reasoning.strip():
        quoted = "\n".join(f"> {line}" for line in reasoning.splitlines() or [""])
        parts.append(f"> 💭 _thinking_\n{quoted}")
    content = _content_text(message)
    if content:
        parts.append(content)
    for call in msg.tool_calls_of(message):
        parts.append(_render_tool_call(call))
    return "\n\n".join(parts)


def _render_tool_call(call: dict[str, Any]) -> str:
    name = call.get("name") or "tool"
    return f"⏺ **{name}**\n\n{_fenced(json.dumps(call.get('arguments') or {}, ensure_ascii=False, indent=2))}"


def _content_text(message: dict[str, Any]) -> str:
    """A message's content as a transcript reads it: text, and a note per picture.

    ``thinking`` and ``toolCall`` blocks are rendered by the caller, each in its
    own place, so they are not repeated here.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    out: list[str] = []
    for block in msg.blocks_of(message):
        if msg.is_text(block):
            out.append(str(block.get("text") or ""))
        elif msg.is_image(block):
            out.append("[image]")
    return "\n".join(out)


def _fenced(text: str) -> str:
    return f"```\n{text}\n```"


__all__ = ["render_transcript", "default_export_path", "write_transcript"]
