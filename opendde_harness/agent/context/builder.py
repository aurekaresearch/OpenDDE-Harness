"""Holder for the two per-workspace stores the turn path shares, plus the two
message-appending helpers the agent loop uses.

It used to render a second, "representative" system prompt to estimate the
turn's size. That is gone: the context engine renders once and budgets against
what it rendered, so there is no second prompt to drift from the first. The
rendering helpers themselves live in
:mod:`opendde_harness.context_engine.segments.render`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from opendde_harness.config.paths import WorkspaceStorage, get_workspace_storage
from opendde_harness.memory_engine.consolidate.consolidator import MemoryStore
from opendde_harness.memory_engine.skill_forge import LocalSkillCatalog
from opendde_harness.providers import messages as msg
from opendde_harness.security.trust import wrap_untrusted, wrap_untrusted_blocks

if TYPE_CHECKING:
    from opendde_harness.providers.base import LLMProvider


class ContextBuilder:
    """Owns the workspace's :class:`MemoryStore` and :class:`LocalSkillCatalog`."""

    def __init__(
        self,
        workspace: Path,
        skill_forge_config: Any = None,
        llm_provider: "LLMProvider | None" = None,
        *,
        start_watcher: bool = True,
        storage: WorkspaceStorage | None = None,
    ):
        self.workspace = workspace
        self.storage = storage if storage is not None else get_workspace_storage(workspace)
        self.memory = MemoryStore(workspace, memory_dir=self.storage.host_memory, state_dir=self.storage.memory_state)
        self.skills = LocalSkillCatalog(
            workspace,
            config=skill_forge_config,
            llm_provider=llm_provider,
            start_watcher=start_watcher,
            skills_dir=self.storage.skills,
        )

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list.

        Tool output is attacker-influenceable (web pages, file/command
        contents, MCP returns), so it is fenced as untrusted data before it
        reaches the model — every tool result funnels through here. The fence
        goes *inside* the text block, which is where the model reads it.

        ``blocks`` carries multimodal content (an image the tool read) and
        replaces the plain text when present. ``result`` stays the fallback and
        must make sense on its own; the model layer decides where a picture
        goes on the wire.
        """
        content = (
            wrap_untrusted_blocks(blocks, source=tool_name)
            if blocks
            else [msg.text_block(wrap_untrusted(result, source=tool_name))]
        )
        messages.append(msg.tool_result_message(tool_call_id, tool_name, content))
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
        pi_message: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list.

        ``pi_message`` is the model layer's own final message and is appended
        as it came back: it is both what the session records and what the next
        request replays, which is how thinking signatures and native tool-call
        ids survive a turn. The reasoning and the calls beside it are derived
        from the same message and are ignored; ``content`` is honoured only
        where it differs, which is the loop having stripped ``<think>`` debris.

        Without one — a recovery prefill, the reply a terminated action gets, a
        provider that is not pi — the message is built from the fields beside
        it, with the marker that declares it cross-model to pi.
        """
        if pi_message is not None:
            entry = dict(pi_message)
            # The loop strips ``<think>`` debris from a reply before it shows
            # it; the stored message is the replayed one, so the strip has to
            # reach it too or the debris comes back on the next request.
            if isinstance(content, str) and content != msg.text_of(entry):
                entry = msg.with_text(entry, content)
            messages.append(entry)
            return messages
        messages.append(
            msg.assistant_message(
                content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
            )
        )
        return messages
