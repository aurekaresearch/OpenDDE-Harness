"""Build an :class:`AgentLoop` from the one config root.

Every entry point (the TUI RPC server, the protein-design worker) constructs the
loop through :func:`build_agent_loop`, so the config-to-loop mapping exists
once. :class:`AgentLoopSettings` is the flat, config-independent shape the
constructor takes; tests and eval harnesses fill it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from opendde_harness.agent.loop.recovery import RecoveryLimits, limits_from_defaults
from opendde_harness.config.features import ContextConfig, MemoryConfig, RuntimeConfig, SkillForgeConfig
from opendde_harness.config.schema import Config, ExecToolConfig, ToolSearchConfig

if TYPE_CHECKING:
    from opendde_harness.agent.hook import CompositeHook
    from opendde_harness.agent.loop.main import AgentLoop
    from opendde_harness.agent.tools.base import Tool
    from opendde_harness.memory_engine.backend import MemoryBackend
    from opendde_harness.providers.base import LLMProvider
    from opendde_harness.session.manager import SessionManager
    from opendde_harness.token_wise.registry import StrategyRegistry


@dataclass
class AgentLoopSettings:
    """Everything an :class:`AgentLoop` reads from config.

    ``model=None`` means the provider's default. ``context_window_tokens``
    of ``None``/``0`` means "resolve from the model"; a positive value pins
    the window. ``skill_forge=None`` leaves the LLM-backed skill stages
    (query rewriter, gate) unbuilt.
    """

    model: str | None = None
    max_iterations: int = 40
    context_window_tokens: int | None = None
    empty_recovery: RecoveryLimits = field(default_factory=RecoveryLimits)
    max_concurrent_subagents: int = 4
    max_subagent_spawns_per_hour: int = 30
    brave_api_key: str | None = None
    jina_api_key: str | None = None
    web_proxy: str | None = None
    exec_config: ExecToolConfig = field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    disabled_tools: list[str] = field(default_factory=list)
    tool_search: ToolSearchConfig = field(default_factory=ToolSearchConfig)
    skill_forge: SkillForgeConfig | None = None
    context: ContextConfig = field(default_factory=ContextConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_config(cls, config: Config) -> AgentLoopSettings:
        defaults = config.agents.defaults
        tools = config.tools
        return cls(
            model=defaults.model,
            max_iterations=defaults.max_tool_iterations,
            context_window_tokens=defaults.context_window_tokens,
            empty_recovery=limits_from_defaults(defaults),
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            max_subagent_spawns_per_hour=defaults.max_subagent_spawns_per_hour,
            brave_api_key=tools.web.search.api_key or None,
            jina_api_key=tools.web.jina_api_key or None,
            web_proxy=tools.web.proxy or None,
            exec_config=tools.exec,
            restrict_to_workspace=tools.restrict_to_workspace,
            mcp_servers=tools.mcp_servers,
            disabled_tools=tools.disabled_tools,
            tool_search=tools.tool_search,
            skill_forge=config.skill_forge,
            context=config.context,
            runtime=config.runtime,
            memory=config.memory,
        )


def build_agent_loop(
    config: Config,
    *,
    provider: "LLMProvider",
    session_manager: "SessionManager | None" = None,
    backend: "MemoryBackend | None" = None,
    plugin_tools: "list[Tool] | None" = None,
    interactive: bool = True,
    now_fn: Callable[[], datetime] | None = None,
    hooks: "CompositeHook | None" = None,
    strategies: "StrategyRegistry | None" = None,
) -> "AgentLoop":
    """The one place config becomes an :class:`AgentLoop`.

    ``interactive`` is the call site's signal for ``runtime.checkpoint``:
    a one-shot ``-m`` turn has no next turn to recover into, a TUI session
    does.
    """
    from opendde_harness.agent.loop.main import AgentLoop

    return AgentLoop(
        provider,
        config.workspace_path,
        AgentLoopSettings.from_config(config),
        session_manager=session_manager,
        backend=backend,
        plugin_tools=plugin_tools,
        interactive=interactive,
        now_fn=now_fn,
        hooks=hooks,
        strategies=strategies,
    )
