"""Whitelisted tool definitions for protein-design agent profiles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from opendde_harness.plugin.protein_design.core.contracts import (
    EpitopeAnalysisRequest,
    Esm2GuidedProposalRequest,
    EvolutionTreeRequest,
    ProtrekSequenceSearchRequest,
    ProtrekStructureSearchRequest,
    SolubleMPNNRequest,
    StructureAnalysisRequest,
)
from opendde_harness.plugin.protein_design.core.external import (
    PROTREK_SERVICE,
    ExternalServiceUnavailableError,
    connection_reason,
    short_reason,
    unavailable_message,
)
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeError


class ProteinDesignToolError(RuntimeError):
    pass


def _unavailable_tool_result(service: str, reason: str, endpoint: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "service": service,
        "reason": reason,
        "endpoint": endpoint,
        "result": unavailable_message(service, reason),
    }


@dataclass
class ToolContext:
    compute: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ProteinDesignToolRegistry:
    def __init__(self, definitions: Iterable[ToolDefinition] = ()) -> None:
        self._definitions = {definition.name: definition for definition in definitions}

    @classmethod
    def for_compute(cls, compute: Any) -> "ProteinDesignToolRegistry":
        def scoped_arguments(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
            payload = dict(arguments)
            task_id = context.metadata.get("task_id")
            if task_id:
                # Task scope is trusted runner state. Never let an LLM-provided
                # argument escape into another task's artifacts.
                payload["task_id"] = str(task_id)
            return payload

        async def model_call(
            context: ToolContext,
            arguments: dict[str, Any],
            model: type[BaseModel],
            method: str,
        ) -> dict[str, Any]:
            target = context.compute or compute
            payload = scoped_arguments(context, arguments)
            if model is EvolutionTreeRequest:
                payload["candidates_json_path"] = str(context.metadata.get("candidates_json_path") or "in-memory")
                payload["objective_key"] = (
                    context.metadata.get("objective_key") or payload.get("objective_key") or "loss"
                )
                payload["minimize"] = bool(context.metadata.get("minimize", payload.get("minimize", True)))
                for key in (
                    "cycle",
                    "binder_chain_ids",
                    "cdr_regions",
                    "reflection_interval",
                ):
                    if key in context.metadata:
                        payload[key] = context.metadata[key]
            value = await getattr(target, method)(model.model_validate(payload))
            return cls._as_dict(value)

        async def optional_call(
            context: ToolContext,
            arguments: dict[str, Any],
            model: type[BaseModel],
            method: str,
            service: str,
        ) -> dict[str, Any]:
            """An optional enrichment service never fails the cycle it was called from."""
            try:
                value = await model_call(context, arguments, model, method)
            except ExternalServiceUnavailableError as exc:
                return _unavailable_tool_result(service, exc.reason, exc.endpoint)
            except ProteinDesignComputeError as exc:
                # The service answered, and refused. A rejected argument is the
                # common case: an agent with no structure to search called this
                # with a placeholder path, the task-boundary check returned 400,
                # and the whole design died in its first phase. Hand the refusal
                # back as a result the agent can read and move past, which is
                # what "optional" is supposed to mean.
                return _unavailable_tool_result(service, short_reason(exc), None)
            except Exception as exc:
                reason = connection_reason(exc)
                if reason is None:
                    raise
                return _unavailable_tool_result(service, reason, None)
            if value.get("available") is False:
                reason = str(value.get("reason") or value.get("error") or "the service did not answer")
                return _unavailable_tool_result(service, reason, value.get("endpoint"))
            return value

        async def raw_call(
            context: ToolContext,
            arguments: dict[str, Any],
            method: str,
        ) -> dict[str, Any]:
            target = context.compute or compute
            value = await getattr(target, method)(scoped_arguments(context, arguments))
            return cls._as_dict(value)

        def definition(
            name: str,
            description: str,
            model: type[BaseModel],
            method: str,
            *,
            optional_service: str | None = None,
        ) -> ToolDefinition:
            async def handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
                if optional_service is None:
                    return await model_call(context, arguments, model, method)
                return await optional_call(context, arguments, model, method, optional_service)

            return ToolDefinition(
                name=name,
                description=description,
                parameters=model.model_json_schema(),
                handler=handler,
            )

        async def developability(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
            return await raw_call(context, arguments, "developability")

        definitions = [
            definition(
                "protrek_sequence_search",
                "Search verified protein sequence neighbours.",
                ProtrekSequenceSearchRequest,
                "search_protrek_sequence",
                optional_service=PROTREK_SERVICE,
            ),
            definition(
                "protrek_structure_search",
                "Search verified protein structure neighbours. structure_path must be a "
                "structure file this task produced; skip this tool when the task has none.",
                ProtrekStructureSearchRequest,
                "search_protrek_structure",
                optional_service=PROTREK_SERVICE,
            ),
            definition(
                "generate_soluble_mpnn",
                "Generate CDR sequences conditioned on a parent antibody structure.",
                SolubleMPNNRequest,
                "generate_soluble_mpnn",
            ),
            definition(
                "generate_esm2_guided",
                "Generate ESM2 likelihood-guided CDR substitutions.",
                Esm2GuidedProposalRequest,
                "generate_esm2_guided",
            ),
            definition(
                "analyze_epitope",
                "Analyze antibody CDR contacts with the configured epitope.",
                EpitopeAnalysisRequest,
                "analyze_epitope",
            ),
            definition(
                "analyze_structure",
                "Analyze folded antibody-target structures.",
                StructureAnalysisRequest,
                "analyze_structure",
            ),
            definition(
                "analyze_current_cycle_structure",
                "Analyze structures from the current design cycle.",
                StructureAnalysisRequest,
                "analyze_structure",
            ),
            definition(
                "analyze_evolution_tree",
                "Analyze the current candidate evolutionary tree.",
                EvolutionTreeRequest,
                "analyze_evolution_tree",
            ),
            ToolDefinition(
                name="check_antibody_developability",
                description="Assess antibody developability for a batch of candidates.",
                parameters={
                    "type": "object",
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "items": {"type": "object"},
                        }
                    },
                    "required": ["candidates"],
                },
                handler=developability,
            ),
        ]
        return cls(definitions)

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        raise ProteinDesignToolError(f"compute tool returned {type(value).__name__}, expected object")

    def definitions(self, allowed: Collection[str]) -> list[dict[str, Any]]:
        allowed_names = set(allowed)
        return [definition.as_openai_tool() for name, definition in self._definitions.items() if name in allowed_names]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        *,
        allowed: Collection[str],
    ) -> dict[str, Any]:
        if name not in set(allowed):
            raise ProteinDesignToolError(f"tool {name!r} is not allowed")
        definition = self._definitions.get(name)
        if definition is None:
            raise ProteinDesignToolError(f"tool {name!r} is not registered")
        self._validate_arguments(definition, arguments)
        result = await definition.handler(context, arguments)
        if not isinstance(result, dict):
            raise ProteinDesignToolError(f"tool {name!r} returned {type(result).__name__}, expected object")
        return result

    @staticmethod
    def _validate_arguments(definition: ToolDefinition, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ProteinDesignToolError(f"tool {definition.name!r} arguments must be an object")
        required = definition.parameters.get("required") or []
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ProteinDesignToolError(
                f"tool {definition.name!r} is missing required arguments: {', '.join(missing)}"
            )
