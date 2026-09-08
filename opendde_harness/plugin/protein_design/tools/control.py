"""OpenDDE Harness tools for controlling protein-design tasks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.plugin.context import PluginContext
from opendde_harness.plugin.protein_design.core.contracts import TaskSnapshot, WorkflowConfig
from opendde_harness.plugin.protein_design.core.detached import DetachedDesignTaskController
from opendde_harness.plugin.protein_design.core.preparation import preparation_context

TaskLauncher = Callable[[WorkflowConfig], Awaitable[TaskSnapshot]]


@dataclass
class _PluginServices:
    ctx: PluginContext
    runtime: DetachedDesignTaskController = field(init=False)

    def __post_init__(self) -> None:
        self.runtime = DetachedDesignTaskController(self.ctx.config)


_SERVICES: dict[tuple[str, str], _PluginServices] = {}


def _objects(ctx: PluginContext) -> _PluginServices:
    placement = json.dumps(
        {
            "compute_url": ctx.config.get("compute_url"),
            "compute_workers": ctx.config.get("compute_workers"),
            "request_timeout": ctx.config.get("request_timeout"),
        },
        sort_keys=True,
        default=str,
    )
    key = (str(ctx.services.workspace), placement)
    services = _SERVICES.get(key)
    if services is None:
        services = _PluginServices(ctx)
        _SERVICES[key] = services
    return services


class ContextTool(Tool):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "protein_design_context"

    @property
    def description(self) -> str:
        return (
            "Read this once before preparing a design: return verified example paths, "
            "configured compute placement, folding/MSA policy and default loss weights. "
            "No credentials, network requests or task launch."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repository_root": {
                    "type": "string",
                    "description": "Optional actual checkout path supplied by the user.",
                }
            },
            "additionalProperties": False,
        }

    async def execute(self, repository_root: str | None = None) -> ToolResult:
        payload = await asyncio.to_thread(preparation_context, self._config, repository_root)
        return ToolResult(json.dumps(payload, ensure_ascii=False), "Design preparation context (read-only)")


class StartTool(Tool):
    def __init__(
        self,
        runtime: DetachedDesignTaskController,
        *,
        launcher: TaskLauncher,
    ) -> None:
        self._runtime = runtime
        self._launcher = launcher

    @property
    def name(self) -> str:
        return "protein_design_start"

    @property
    def description(self) -> str:
        return (
            "Start an antigen-antibody design task from a validated YAML path only after "
            "the user explicitly confirms the final target, antibody binder, design, and "
            "compute review. Refuse non-antibody binder-design requests."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "config_path": {"type": "string"},
                "user_confirmed": {
                    "type": "boolean",
                    "const": True,
                    "description": ("True only after the user explicitly confirms the final validated YAML review."),
                },
            },
            "required": ["config_path", "user_confirmed"],
            "additionalProperties": False,
        }

    async def execute(self, config_path: str, user_confirmed: bool = False) -> ToolResult:
        if user_confirmed is not True:
            raise ValueError(
                "protein-design launch requires explicit user confirmation after the final "
                "target, binder, design, and compute review"
            )
        workflow = self._runtime.config_from_path(config_path)
        snapshot = await self._launcher(workflow)
        return ToolResult(
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False),
            f"Started protein design task {snapshot.task_id}",
        )


class StatusTool(Tool):
    def __init__(self, runtime: DetachedDesignTaskController) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "protein_design_status"

    @property
    def description(self) -> str:
        return "Show progress, best candidate, and errors for one or all protein-design tasks."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"task_id": {"type": "string"}}}

    async def execute(self, task_id: str | None = None) -> str:
        value = self._runtime.status(task_id)
        if isinstance(value, list):
            payload = [item.model_dump(mode="json") for item in value]
        else:
            payload = value.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False)


class AdjustTool(Tool):
    def __init__(self, runtime: DetachedDesignTaskController) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "protein_design_adjust"

    @property
    def description(self) -> str:
        return "Queue safe protein-design parameter changes for the next cycle boundary."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "params": {
                    "type": "object",
                    "properties": {
                        "num_sequences": {"type": "integer", "minimum": 1},
                        "reflection_interval": {"type": "integer", "minimum": 1},
                    },
                    "additionalProperties": False,
                    "minProperties": 1,
                },
            },
            "required": ["task_id", "params"],
        }

    async def execute(self, task_id: str, params: dict[str, Any]) -> str:
        return self._runtime.adjust(task_id, params).model_dump_json()


class StopTool(Tool):
    def __init__(self, runtime: DetachedDesignTaskController) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "protein_design_stop"

    @property
    def description(self) -> str:
        return "Stop a protein-design task at the next safe cycle boundary."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        }

    async def execute(self, task_id: str) -> str:
        return self._runtime.stop(task_id).model_dump_json()


class CandidatesTool(Tool):
    def __init__(self, runtime: DetachedDesignTaskController) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "protein_design_candidates"

    @property
    def description(self) -> str:
        return "Return top scored protein-design candidates from the compute population."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Protein-design task whose isolated population should be read.",
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["task_id"],
        }

    async def execute(self, top_k: int = 20, task_id: str | None = None) -> str:
        if not task_id:
            raise ValueError("task_id is required so objective direction is unambiguous")
        payload = await self._runtime.top_candidates(task_id, top_k)
        return json.dumps(payload, ensure_ascii=False)


class TargetMsaSearchTool(Tool):
    """Search an antigen-chain MSA on the selected protein-design worker."""

    def __init__(self, runtime: DetachedDesignTaskController) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "protein_design_search_target_msa"

    @property
    def description(self) -> str:
        return (
            "Search paired and unpaired A3M files for one target/antigen chain through "
            "Protenix's online MSA service. Never use this for binder/antibody chains. "
            "Call only after the user explicitly chooses online MSA search. No compute "
            "argument is required: omit compute_url to use the configured worker pool or "
            "default endpoint. Pass compute_url only when the reviewed design already "
            "specifies one. Bind the final design YAML to the worker returned by this tool."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_role": {
                    "type": "string",
                    "const": "target",
                    "description": "Hard scope: this tool searches antigen/target chains only.",
                },
                "target_name": {"type": "string", "minLength": 1},
                "chain_id": {"type": "string", "minLength": 1},
                "sequence": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Canonical amino-acid sequence of the target chain only.",
                },
                "compute_url": {
                    "type": "string",
                    "description": (
                        "Optional explicit compute endpoint already present in the reviewed "
                        "design. Omit it to use automatic pool/default placement."
                    ),
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Ignore a valid cached MSA and search again.",
                },
                "user_confirmed": {
                    "type": "boolean",
                    "const": True,
                    "description": "True only after the user explicitly selects online MSA search.",
                },
            },
            "required": [
                "entity_role",
                "target_name",
                "chain_id",
                "sequence",
                "user_confirmed",
            ],
            "additionalProperties": False,
        }

    async def execute(
        self,
        target_name: str,
        chain_id: str,
        sequence: str,
        entity_role: str = "target",
        user_confirmed: bool = False,
        compute_url: str | None = None,
        force: bool = False,
    ) -> ToolResult:
        if user_confirmed is not True:
            raise ValueError("online target MSA search requires explicit user confirmation")
        if entity_role.strip() != "target":
            raise ValueError("MSA search is restricted to target/antigen chains")
        compute_url = str(compute_url or "").strip() or None
        payload = await self._runtime.search_target_msa(
            target_name=target_name,
            chain_id=chain_id,
            sequence=sequence,
            compute_url=compute_url,
            force=force,
        )
        return ToolResult(
            json.dumps(payload, ensure_ascii=False),
            f"Target MSA ready for {target_name} chain {chain_id}",
        )


def make_context_tool(ctx: PluginContext) -> ContextTool:
    return ContextTool(dict(ctx.config))


def make_start_tool(ctx: PluginContext) -> StartTool:
    services = _objects(ctx)
    return StartTool(services.runtime, launcher=services.runtime.start)


def make_status_tool(ctx: PluginContext) -> StatusTool:
    return StatusTool(_objects(ctx).runtime)


def make_adjust_tool(ctx: PluginContext) -> AdjustTool:
    return AdjustTool(_objects(ctx).runtime)


def make_stop_tool(ctx: PluginContext) -> StopTool:
    return StopTool(_objects(ctx).runtime)


def make_candidates_tool(ctx: PluginContext) -> CandidatesTool:
    return CandidatesTool(_objects(ctx).runtime)


def make_target_msa_search_tool(ctx: PluginContext) -> TargetMsaSearchTool:
    return TargetMsaSearchTool(_objects(ctx).runtime)
