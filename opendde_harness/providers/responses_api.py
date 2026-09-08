"""OpenAI Responses API translation and hosted web-search handling.

Chat-shaped history, tools and tool choices become Responses input items;
Responses usage and ``web_search_call`` items become the provider's Chat-shaped
usage and built-in tool events.
"""

from __future__ import annotations

import json
from typing import Any


def value_of(obj: Any, name: str, default: Any = None) -> Any:
    """Read one field from a pydantic response object or a plain mapping."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def web_search_action(item: Any) -> tuple[dict[str, Any], str]:
    action = value_of(item, "action", {}) or {}
    action_type = str(value_of(action, "type", "search") or "search")
    arguments: dict[str, Any] = {"action": action_type}
    display = "searching the web"

    if action_type == "search":
        queries = [str(query) for query in (value_of(action, "queries", []) or []) if str(query).strip()]
        legacy_query = str(value_of(action, "query", "") or "").strip()
        if not queries and legacy_query:
            queries = [legacy_query]
        if queries:
            arguments["queries"] = queries
            arguments["query"] = queries[0]
            display = queries[0]
    elif action_type == "open_page":
        url = str(value_of(action, "url", "") or "").strip()
        if url:
            arguments["url"] = url
            display = f"opening {url}"
    elif action_type == "find_in_page":
        url = str(value_of(action, "url", "") or "").strip()
        pattern = str(value_of(action, "pattern", "") or "").strip()
        if url:
            arguments["url"] = url
        if pattern:
            arguments["pattern"] = pattern
        display = f'finding "{pattern}" in {url}'.strip()

    return arguments, display


def url_citations(response: Any) -> list[tuple[str, str]]:
    citations: list[tuple[str, str]] = []
    for item in value_of(response, "output", []) or []:
        if value_of(item, "type", "") != "message":
            continue
        for part in value_of(item, "content", []) or []:
            for annotation in value_of(part, "annotations", []) or []:
                if value_of(annotation, "type", "") != "url_citation":
                    continue
                url = str(value_of(annotation, "url", "") or "").strip()
                title = str(value_of(annotation, "title", "") or "").strip()
                if url:
                    citations.append((url, title))
    return citations


def web_search_preview(item: Any, citations: list[tuple[str, str]]) -> str:
    action = value_of(item, "action", {}) or {}
    source_urls = [
        str(value_of(source, "url", "") or "").strip()
        for source in (value_of(action, "sources", []) or [])
        if str(value_of(source, "url", "") or "").strip()
    ]
    matched_citations = [(url, title) for url, title in citations if not source_urls or url in source_urls]
    unique: dict[str, str] = {}
    for url, title in [*matched_citations, *((url, "") for url in source_urls)]:
        if url not in unique or (title and not unique[url]):
            unique[url] = title

    _, display = web_search_action(item)
    lines = [display]
    if unique:
        lines.append(f"Sources ({len(unique)}):")
        for url, title in unique.items():
            lines.append(f"- {title + ' - ' if title else ''}{url}")
    else:
        lines.append("Search completed")
    return "\n".join(lines)


def is_local_web_search_tool(tool: dict[str, Any]) -> bool:
    if tool.get("type") != "function":
        return False
    function = tool.get("function")
    if isinstance(function, dict):
        return function.get("name") == "web_search"
    return tool.get("name") == "web_search"


def rejects_hosted_web_search(exc: Exception) -> bool:
    message = str(exc).lower()
    names_search = any(
        marker in message
        for marker in (
            "web_search",
            "web search",
            "googlesearch",
            "google_search",
            "server tool",
        )
    )
    rejects_feature = any(
        marker in message
        for marker in (
            "unsupported",
            "not supported",
            "unknown",
            "unrecognized",
            "invalid tool",
            "invalid parameter",
            "extra inputs are not permitted",
        )
    )
    return names_search and rejects_feature


def selects_local_web_search(choice: Any) -> bool:
    if not isinstance(choice, dict) or choice.get("type") != "function":
        return False
    function = choice.get("function")
    if isinstance(function, dict):
        return function.get("name") == "web_search"
    return choice.get("name") == "web_search"


def responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Translate Chat Completions function tools to Responses function tools."""
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            converted.append(dict(tool))
            continue
        function = tool["function"]
        item: dict[str, Any] = {
            "type": "function",
            "name": function["name"],
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        }
        if function.get("description") is not None:
            item["description"] = function["description"]
        if function.get("strict") is not None:
            item["strict"] = function["strict"]
        elif tool.get("strict") is not None:
            item["strict"] = tool["strict"]
        converted.append(item)
    return converted


def responses_tool_choice(choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Translate a named Chat function choice to the Responses shape."""
    if not isinstance(choice, dict):
        return choice
    function = choice.get("function")
    if choice.get("type") == "function" and isinstance(function, dict):
        return {"type": "function", "name": function.get("name", "")}
    return choice


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate persisted Chat history, including tool turns, to input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": responses_tool_output(content),
                }
            )
            continue

        tool_calls = message.get("tool_calls") if role == "assistant" else None
        if content not in (None, "", []):
            items.append({"role": role, "content": responses_content(content)})
        elif not tool_calls:
            # Preserve empty ordinary messages; provider sanitation has already
            # supplied a valid content value where one is required.
            items.append({"role": role, "content": content or ""})

        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    }
                )
    return items


def responses_usage(response: Any) -> dict[str, int]:
    usage = value_of(response, "usage", None)
    if usage is None:
        return {}
    input_tokens = int(value_of(usage, "input_tokens", 0) or 0)
    output_tokens = int(value_of(usage, "output_tokens", 0) or 0)
    result = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": int(value_of(usage, "total_tokens", input_tokens + output_tokens) or 0),
    }
    input_details = value_of(usage, "input_tokens_details", None)
    cached = int(value_of(input_details, "cached_tokens", 0) or 0)
    cache_write = int(value_of(input_details, "cache_write_tokens", 0) or 0)
    if cached:
        result["cache_read_input_tokens"] = cached
    if cache_write:
        result["cache_creation_input_tokens"] = cache_write
    return result


#: Why a Responses run stopped short. Only the token ceiling is truncation; a
#: filtered answer is finished as far as the model is concerned, and calling it
#: truncated invites a continuation that gets filtered again.
_INCOMPLETE_REASONS = {"max_output_tokens": "length", "content_filter": "content_filter"}


def responses_incomplete_reason(incomplete_details: Any) -> str:
    """The finish reason behind ``status == "incomplete"``."""
    reason = str(value_of(incomplete_details, "reason", "") or "")
    return _INCOMPLETE_REASONS.get(reason, "length")


def responses_error_text(payload: Any) -> str:
    """One line for an error event, code first so the retry ladder can read it."""
    error = value_of(payload, "error", None)
    if error is None and isinstance(payload, dict):
        error = payload
    code = str(value_of(error, "code", "") or "")
    message = str(value_of(error, "message", "") or "")
    detail = ": ".join(part for part in (code, message) if part)
    return detail or "unspecified error"


def responses_content(content: Any) -> str | list[dict[str, Any]]:
    """Return Responses-compatible message/function-output content.

    Plain strings stay compact. Chat multimodal blocks are converted one by
    one so images and files remain media instead of being serialized as prose.
    Unknown structured values are preserved as JSON text rather than silently
    dropped.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    blocks = content if isinstance(content, list) else [content]
    converted: list[dict[str, Any]] = []
    for block in blocks:
        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            block = model_dump(exclude_none=True)
        if not isinstance(block, dict):
            converted.append({"type": "input_text", "text": str(block)})
            continue
        block_type = block.get("type")
        if block_type in {"input_text", "input_image", "input_file"}:
            converted.append(dict(block))
        elif block_type == "text":
            converted.append({"type": "input_text", "text": str(block.get("text", ""))})
        elif block_type == "image_url":
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not url:
                converted.append(_json_text(block))
                continue
            item: dict[str, Any] = {"type": "input_image", "image_url": str(url)}
            detail = image.get("detail") if isinstance(image, dict) else block.get("detail")
            if detail is not None:
                item["detail"] = detail
            converted.append(item)
        elif block_type == "file":
            file_value = block.get("file")
            source = file_value if isinstance(file_value, dict) else block
            item = {"type": "input_file"}
            for key in ("file_id", "file_data", "filename"):
                if source.get(key) is not None:
                    item[key] = source[key]
            converted.append(item if len(item) > 1 else _json_text(block))
        else:
            converted.append(_json_text(block))
    return converted


def responses_tool_output(content: Any) -> str | list[dict[str, Any]]:
    """Return a valid ``function_call_output.output`` value."""
    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, list):
        return responses_content(content)
    return json.dumps(content, ensure_ascii=False, default=str)


def _json_text(value: Any) -> dict[str, str]:
    return {
        "type": "input_text",
        "text": json.dumps(value, ensure_ascii=False, default=str),
    }
