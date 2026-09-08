"""Structured OpenDDE Harness provider sessions for protein-design agents."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Collection
from typing import Any, Protocol, TypeVar, cast

from json_repair import repair_json
from loguru import logger
from pydantic import BaseModel, ValidationError

from opendde_harness.plugin.protein_design.agents.profiles import AgentProfile
from opendde_harness.plugin.protein_design.agents.skills import SkillDocument
from opendde_harness.plugin.protein_design.core.progress import (
    DesignProgressEvent,
    ProgressEventType,
    ProgressStatus,
    emit_progress,
)
from opendde_harness.plugin.protein_design.tools.agent import (
    ProteinDesignToolError,
    ProteinDesignToolRegistry,
    ToolContext,
)
from opendde_harness.providers.base import LLMProvider
from opendde_harness.utils.helpers import build_assistant_message

T = TypeVar("T", bound=BaseModel)

def _assistant_message(response: Any, content: Any, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One assistant turn, carrying whatever the model signed.

    A thinking model wants its own reasoning back: DeepSeek rejects the whole
    request without it ("the reasoning_content in the thinking mode must be
    passed back"), which failed the analyze agent on its first tool call and
    took the design task down before cycle 0.
    """
    return build_assistant_message(
        content,
        tool_calls,
        getattr(response, "reasoning_content", None),
        getattr(response, "thinking_blocks", None),
    )


_MAX_TRUNCATION_RETRIES = 2
_MAX_OUTPUT_TOKENS = 16384

_USE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "use_skill",
        "description": (
            "Load one Router-approved protein-design skill before applying its "
            "instructions. Pass an exact skill id from the available-skill catalog."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "Exact skill id shown in the available-skill catalog.",
                }
            },
            "required": ["skill_id"],
            "additionalProperties": False,
        },
    },
}


class StructuredSession(Protocol):
    async def run(self, profile: AgentProfile[T], prompt: str, **kwargs) -> T: ...


class OpenDDEHarnessStructuredSession:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        tool_registry: ProteinDesignToolRegistry | None = None,
        max_attempts: int = 3,
        temperature: float = 0.2,
        progress_sink=None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._tool_registry = tool_registry or ProteinDesignToolRegistry()
        self._max_attempts = max_attempts
        self._temperature = temperature
        self._progress_sink = progress_sink

    async def close(self) -> None:
        provider = getattr(self._provider, "unwrapped", self._provider)
        if provider is None:
            return
        close = getattr(provider, "aclose", None) or getattr(provider, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
            return
        from opendde_harness.providers.litellm_setup import import_litellm

        close_clients = getattr(import_litellm(), "close_litellm_async_clients", None)
        if callable(close_clients):
            result = close_clients()
            if inspect.isawaitable(result):
                await result

    async def run(
        self,
        profile: AgentProfile[T],
        prompt: str,
        *,
        skills: Collection[SkillDocument] = (),
        tool_context: ToolContext | None = None,
    ) -> T:
        context = tool_context or ToolContext()
        started = time.perf_counter()
        input_payload = {
            "system_prompt": self._system_message(profile, skills),
            "prompt": prompt,
            "skills": [skill.name for skill in skills],
            "tools": self._definitions(profile, bool(skills)),
        }
        self._emit_agent(profile, input_payload, context, ProgressStatus.STARTED)
        try:
            result = await self._run_profile(profile, prompt, skills, context)
        except Exception as exc:
            self._emit_agent(
                profile,
                input_payload,
                context,
                ProgressStatus.FAILED,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )
            raise
        self._emit_agent(
            profile,
            input_payload,
            context,
            ProgressStatus.COMPLETED,
            duration_ms=(time.perf_counter() - started) * 1000,
            output=result,
        )
        return result

    async def _run_profile(
        self,
        profile: AgentProfile[T],
        prompt: str,
        skills: Collection[SkillDocument],
        tool_context: ToolContext,
    ) -> T:
        schema = profile.output_schema
        messages: list[dict] = [
            {"role": "system", "content": self._system_message(profile, skills)},
            {"role": "user", "content": prompt},
        ]
        skill_by_name = {skill.name: skill for skill in skills}
        definitions = self._definitions(profile, bool(skill_by_name))
        seen_calls: set[str] = set()
        loaded_skill_ids: set[str] = set()
        tool_call_counts: dict[str, int] = {}
        tool_call_limits = dict(profile.tool_call_limits)
        tool_turns = 0
        # use_skill is exempt from max_tool_turns and is never withdrawn while skills
        # exist, so it needs its own budget or a skill-looping model never terminates.
        skill_turns = 0
        max_skill_turns = len(skill_by_name) + self._max_attempts
        attempts = 0
        last_error = "empty response"
        max_tokens = int(tool_context.metadata.get("llm_max_tokens", profile.max_tokens))
        truncation_retries = 0
        while attempts < self._max_attempts:
            # Transport failures (HTTP 429/5xx, timeouts, endpoint rotation and
            # model fallback) belong to the provider retry ladder.  Calling
            # ``chat()`` directly here used the structured-output repair budget
            # for transient wire errors and let a short 503 outage kill the
            # complete design task before cycle 0.
            response = await self._provider.chat_with_retry(
                messages=messages,
                tools=definitions or None,
                model=str(tool_context.metadata.get("llm_model") or self._model),
                max_tokens=max_tokens,
                temperature=float(
                    tool_context.metadata.get(
                        "llm_temperature",
                        profile.temperature,
                    )
                ),
            )
            content = response.content or ""
            truncated = response.finish_reason == "length" or getattr(response, "truncated", False)
            if (
                truncated
                and not response.tool_calls
                and truncation_retries < _MAX_TRUNCATION_RETRIES
                and max_tokens < _MAX_OUTPUT_TOKENS
            ):
                # A reasoning model can spend the whole output budget before it
                # emits the JSON, leaving content empty; widen the budget instead
                # of spending a repair attempt on unparsable output.
                truncation_retries += 1
                previous = max_tokens
                max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS)
                logger.warning(
                    "{} output hit the {}-token limit; retrying with max_tokens={}",
                    profile.role,
                    previous,
                    max_tokens,
                )
                continue
            if response.finish_reason == "error":
                # ``chat_with_retry()`` has already exhausted the classified
                # transport retry/fallback policy.  Do not misclassify this as
                # malformed JSON or consume output-repair attempts.
                raise RuntimeError(content or "LLM provider retries exhausted")
            if response.tool_calls:
                if any(call.name != "use_skill" for call in response.tool_calls):
                    tool_turns += 1
                else:
                    skill_turns += 1
                    if skill_turns > max_skill_turns:
                        raise RuntimeError(
                            f"{profile.role} exceeded the use_skill budget of {max_skill_turns} turns "
                            "without returning final JSON"
                        )
                messages.append(
                    _assistant_message(
                        response,
                        response.content,
                        [call.to_openai_tool_call() for call in response.tool_calls],
                    )
                )
                for call in response.tool_calls:
                    key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                    call_limit = tool_call_limits.get(call.name)
                    if call.name == "use_skill":
                        self._emit_tool(
                            profile,
                            call.name,
                            call.arguments,
                            tool_context,
                            ProgressStatus.STARTED,
                        )
                        skill_id = call.arguments.get("skill_id")
                        skill = skill_by_name.get(skill_id) if isinstance(skill_id, str) else None
                        if key in seen_calls:
                            result = {"ok": False, "error": f"skill {skill_id!r} was already loaded"}
                        elif skill is None:
                            result = {
                                "ok": False,
                                "error": (
                                    f"unknown or Router-disallowed skill {skill_id!r}; "
                                    f"choose one of {sorted(skill_by_name)}"
                                ),
                            }
                        else:
                            seen_calls.add(key)
                            loaded_skill_ids.add(skill.name)
                            result = {
                                "ok": True,
                                "skill_id": skill.name,
                                "skill_dir": str(skill.path),
                                "instructions": skill.content,
                            }
                        self._emit_tool(
                            profile,
                            call.name,
                            call.arguments,
                            tool_context,
                            ProgressStatus.COMPLETED if result["ok"] else ProgressStatus.FAILED,
                            output=result if result["ok"] else None,
                            error=None if result["ok"] else result["error"],
                        )
                    elif tool_turns > profile.max_tool_turns:
                        result = {"ok": False, "error": "tool turn limit reached"}
                    elif call_limit is not None and tool_call_counts.get(call.name, 0) >= call_limit:
                        result = {
                            "ok": False,
                            "error": f"tool {call.name!r} call limit reached ({call_limit})",
                        }
                    elif key in seen_calls:
                        result = {"ok": False, "error": f"repeated tool call {call.name!r} was not run"}
                    else:
                        seen_calls.add(key)
                        tool_call_counts[call.name] = tool_call_counts.get(call.name, 0) + 1
                        tool_started = time.perf_counter()
                        self._emit_tool(
                            profile,
                            call.name,
                            call.arguments,
                            tool_context,
                            ProgressStatus.STARTED,
                        )
                        try:
                            value = await self._tool_registry.execute(
                                call.name,
                                call.arguments,
                                tool_context,
                                allowed=profile.allowed_tools,
                            )
                            result = {"ok": True, "result": value}
                            self._emit_tool(
                                profile,
                                call.name,
                                call.arguments,
                                tool_context,
                                ProgressStatus.COMPLETED,
                                duration_ms=(time.perf_counter() - tool_started) * 1000,
                                output=value,
                            )
                        except (ProteinDesignToolError, ValidationError, TypeError, ValueError) as exc:
                            result = {"ok": False, "error": str(exc)}
                            self._emit_tool(
                                profile,
                                call.name,
                                call.arguments,
                                tool_context,
                                ProgressStatus.FAILED,
                                duration_ms=(time.perf_counter() - tool_started) * 1000,
                                error=str(exc),
                            )
                        except Exception as exc:
                            if profile.tool_errors_are_fatal:
                                raise
                            result = {"ok": False, "error": str(exc)}
                            self._emit_tool(
                                profile,
                                call.name,
                                call.arguments,
                                tool_context,
                                ProgressStatus.FAILED,
                                duration_ms=(time.perf_counter() - tool_started) * 1000,
                                error=str(exc),
                            )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        }
                    )
                if tool_turns >= profile.max_tool_turns:
                    definitions = [_USE_SKILL_TOOL] if skill_by_name else []
                elif tool_call_limits:
                    exhausted = {
                        name for name, limit in tool_call_limits.items() if tool_call_counts.get(name, 0) >= limit
                    }
                    definitions = [
                        definition for definition in definitions if definition["function"]["name"] not in exhausted
                    ]
                continue
            try:
                value = json.loads(repair_json(content))
                validated = cast(T, schema.model_validate(value))
                selected_skill = getattr(validated, "skill_id", None)
                missing_skill = (
                    selected_skill
                    if isinstance(selected_skill, str) and selected_skill in skill_by_name and selected_skill not in loaded_skill_ids
                    else None
                )
                if skill_by_name and (not loaded_skill_ids or missing_skill):
                    attempts += 1
                    last_error = (
                        f"selected skill {missing_skill!r} was not loaded"
                        if missing_skill
                        else "no available skill was loaded"
                    )
                    if attempts < self._max_attempts:
                        messages.extend(
                            [
                                _assistant_message(response, content),
                                {
                                    "role": "user",
                                    "content": (
                                        f"{last_error}. Call use_skill with an exact available "
                                        "skill id before returning the final JSON."
                                    ),
                                },
                            ]
                        )
                        continue
                else:
                    return validated
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                attempts += 1
                last_error = str(exc)
                if attempts < self._max_attempts:
                    messages.extend(
                        [
                            _assistant_message(response, content),
                            {
                                "role": "user",
                                "content": f"The output was invalid: {last_error}. Return corrected JSON only.",
                            },
                        ]
                    )
        raise ValueError(f"{profile.role} produced no valid {schema.__name__}: {last_error}")

    def _emit_agent(
        self,
        profile: AgentProfile,
        input_payload: dict[str, Any],
        context: ToolContext,
        status: ProgressStatus,
        *,
        duration_ms: float | None = None,
        output: BaseModel | None = None,
        error: str | None = None,
    ) -> None:
        role = getattr(profile.role, "value", str(profile.role))
        emit_progress(
            self._progress_sink,
            DesignProgressEvent.create(
                task_id=str(context.metadata.get("task_id") or "unknown"),
                event_type=ProgressEventType.AGENT,
                status=status,
                cycle=context.metadata.get("cycle"),
                phase=role,
                actor=role,
                summary=f"{role} agent {status.value}",
                duration_ms=duration_ms,
                input_payload=input_payload,
                output_payload=output,
                error=error,
            ),
        )

    def _emit_tool(
        self,
        profile: AgentProfile,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
        status: ProgressStatus,
        *,
        duration_ms: float | None = None,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        role = getattr(profile.role, "value", str(profile.role))
        emit_progress(
            self._progress_sink,
            DesignProgressEvent.create(
                task_id=str(context.metadata.get("task_id") or "unknown"),
                event_type=ProgressEventType.TOOL,
                status=status,
                cycle=context.metadata.get("cycle"),
                phase=role,
                actor=role,
                tool=name,
                summary=f"{name} {status.value}",
                duration_ms=duration_ms,
                input_payload=arguments,
                output_payload=output,
                error=error,
            ),
        )

    def _system_message(
        self,
        profile: AgentProfile[T],
        skills: Collection[SkillDocument],
    ) -> str:
        skill_block = "\n".join(self._skill_catalog_line(skill) for skill in skills) or "(none)"
        tool_block = json.dumps(
            self._tool_registry.definitions(profile.allowed_tools),
            ensure_ascii=False,
            indent=2,
        )
        schema_block = json.dumps(
            profile.output_schema.model_json_schema(),
            ensure_ascii=False,
        )
        return (
            f"{profile.system_prompt.rstrip()}\n\n"
            "# Available skills\n"
            "Skill bodies are not preloaded. Before applying a skill, call "
            "`use_skill` once with its exact id below and follow the returned "
            "instructions. Do not claim to have used an unloaded skill.\n"
            f"{skill_block}\n\n"
            f"# Available tools\n{tool_block}\n\n"
            "# Output schema\nReturn one JSON object only. "
            f"It must satisfy this JSON schema: {schema_block}"
        )

    def _definitions(self, profile: AgentProfile[T], has_skills: bool) -> list[dict[str, Any]]:
        definitions = self._tool_registry.definitions(profile.allowed_tools)
        if has_skills:
            definitions.append(_USE_SKILL_TOOL)
        return definitions

    @staticmethod
    def _skill_catalog_line(skill: SkillDocument) -> str:
        description = ""
        lines = skill.content.splitlines()
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                stripped = line.strip()
                if stripped == "---":
                    break
                if stripped.startswith("description:"):
                    description = stripped.split(":", 1)[1].strip().strip("'\"")
                    break
        suffix = f" — {description}" if description else ""
        return f"- `{skill.name}`{suffix}"
