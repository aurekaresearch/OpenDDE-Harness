"""Responses API adapter used by OpenDDE Harness's embedded long-term memory service.

The memory library (1.2.x) builds an everalgo Chat Completions client internally.
OpenDDE Harness keeps the public everalgo ``LLMClient`` contract but uses OpenAI's
newer Responses wire by default, so the rest of the library (case extraction,
clustering and skill materialisation) remains unchanged.
"""

from __future__ import annotations

from typing import Any

import openai
from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse, Usage
from pydantic import BaseModel

from opendde_harness.providers.responses_api import responses_content


class ResponsesLLMClient:
    """Adapt ``client.responses`` to everalgo's small chat protocol."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        request: dict[str, Any] = {
            "model": model or self._config.model,
            "input": [
                {
                    **message.model_dump(exclude={"content"}, exclude_none=True),
                    "content": responses_content(message.content),
                }
                for message in messages
            ],
            "temperature": temperature if temperature is not None else self._config.temperature,
            "store": False,
        }
        effective_max = max_tokens if max_tokens is not None else self._config.max_tokens
        if effective_max is not None:
            request["max_output_tokens"] = effective_max
        request.update(self._config.extra)
        request.update(extra)
        # Memory state belongs to the memory service, not the upstream API. Do not let a
        # model override accidentally enable provider-side response storage.
        request["store"] = False

        try:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                response = await self._client.responses.parse(
                    text_format=response_format,
                    **request,
                )
                parsed = response.output_parsed
            else:
                response = await self._client.responses.create(**request)
                parsed = None
        except (openai.OpenAIError, TypeError, ValueError) as exc:
            raise LLMError(str(exc)) from exc

        if response.status == "failed" or response.error is not None:
            raise LLMError(f"Responses API failed: {response.error}")
        finish_reason = "length" if response.status == "incomplete" else "stop"
        usage = None
        if response.usage is not None:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        return ChatResponse(
            content=response.output_text or "",
            model=response.model,
            usage=usage,
            finish_reason=finish_reason,
            parsed=parsed,
            raw=None,
        )
