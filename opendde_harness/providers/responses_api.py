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


#: The ``thinking_blocks`` entry a Responses-wire provider writes for one
#: response and reads back on the next request. One block per response, not
#: one per reasoning item: the loop folds every non-redacted block of a delta
#: into a single entry, so N blocks would keep N-1 items' worth of nothing.
#: ``provider`` says which backend signed the items, since one backend's
#: ``rs_*`` items are refused by another.
REASONING_BLOCK_TYPE = "reasoning"
PROVIDER_OPENAI = "openai"
PROVIDER_CODEX = "openai_codex"


def reasoning_block(
    items: list[dict[str, Any]], order: list[str], messages: dict[str, str], *, provider: str
) -> dict[str, Any] | None:
    """The block carrying a response's reasoning items, or None when it had none.

    ``order`` is the id of every output item as the response emitted them;
    the next request replays in that order, because the backend pairs each
    ``rs_*`` item with the item that followed it and refuses a broken pair.
    ``messages`` maps each output message's id to its text, since a response
    can carry several messages with reasoning between them and each has to
    be replayed under its own id for the pairing to hold.
    """
    if not items:
        return None
    return {
        "type": REASONING_BLOCK_TYPE,
        "provider": provider,
        "thinking": reasoning_text(items),
        "items": items,
        "order": order,
        "messages": dict(messages),
    }


def message_text(item: Any) -> str:
    """The text of one output message item."""
    return "".join(
        str(value_of(part, "text", "") or "")
        for part in (value_of(item, "content", []) or [])
        if value_of(part, "type", "") == "output_text"
    )


def _message_item(message_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
        "status": "completed",
        "id": message_id,
    }


def reasoning_text(items: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in items:
        for part in (item.get("summary") or []) + (item.get("content") or []):
            text = part.get("text") if isinstance(part, dict) else None
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def reasoning_block_of(message: dict[str, Any], *, provider: str) -> dict[str, Any] | None:
    """This provider's own reasoning block on an assistant message, or None."""
    for block in message.get("thinking_blocks") or []:
        if isinstance(block, dict) and block.get("type") == REASONING_BLOCK_TYPE and block.get("provider") == provider:
            return block
    return None


def join_tool_call_id(call_id: str, item_id: str | None) -> str:
    """The stored id for a Responses call: ``call|item`` when it has an item id."""
    return f"{call_id}|{item_id}" if item_id else call_id


def split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """``call_id`` and the ``fc_*`` item id a Responses call carried, if any.

    A Responses function call has two ids -- the ``call_id`` a result answers
    and the item id the backend pairs reasoning with -- stored as one
    ``call|item`` string so the Chat-shaped history can hold both.
    """
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def responses_assistant_items(message: dict[str, Any], idx: int, *, provider: str) -> list[dict[str, Any]]:
    """One assistant turn as the Responses API wants it replayed.

    Every output item in the order the response emitted it, reasoning items
    included -- the order pi replays, and it is not cosmetic: the backend
    pairs every ``rs_*`` item with the item that followed it and refuses a
    request where the pair is broken, and it is the replayed reasoning that
    lets the model continue its own plan across a tool call rather than start
    over. A turn stored without an order (one produced by another provider or
    on another wire) replays as text then calls, and carries no item ids, so
    the backend has no pairing to validate. A reasoning item nothing followed
    -- a response cut before its item arrived -- is not sent, since alone it
    is refused as unpaired.
    """
    reasoning = reasoning_block_of(message, provider=provider)
    # Item ids are this backend's to validate: a turn another backend produced
    # replays without them, or the pairing check refuses ids it never issued.
    own_turn = reasoning is not None
    reasoning = reasoning or {}
    by_id: dict[str, dict[str, Any]] = {}
    for item in reasoning.get("items") or []:
        if isinstance(item, dict) and item.get("id"):
            by_id[item["id"]] = item

    content = message.get("content")
    text = content if isinstance(content, str) else _text_of(content)
    replayed = reasoning.get("messages") if own_turn else None
    if isinstance(replayed, dict) and replayed:
        for message_id, message_body in replayed.items():
            by_id[str(message_id)] = _message_item(str(message_id), str(message_body))
    elif text:
        by_id[f"msg_{idx}"] = _message_item(f"msg_{idx}", text)
    for n, tool_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function") or {}
        arguments = fn.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id, item_id = split_tool_call_id(tool_call.get("id"))
        item = {
            "type": "function_call",
            "call_id": call_id or f"call_{idx}_{n}",
            "name": fn.get("name") or "",
            "arguments": arguments or "{}",
        }
        if item_id and own_turn:
            item["id"] = item_id
        by_id[item_id or f"fc_{idx}_{n}"] = item

    ordered = [by_id.pop(item_id) for item_id in reasoning.get("order") or [] if item_id in by_id]
    ordered += sorted(by_id.values(), key=lambda item: item["type"] != "message")
    while ordered and ordered[-1]["type"] == "reasoning":
        ordered.pop()
    return ordered


def _text_of(content: Any) -> str:
    if not isinstance(content, list):
        return "" if content is None else str(content)
    return "\n".join(
        str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
    )


def responses_input(messages: list[dict[str, Any]], *, provider: str = PROVIDER_OPENAI) -> list[dict[str, Any]]:
    """Translate persisted Chat history, including tool turns, to input items."""
    items: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            call_id, _ = split_tool_call_id(message.get("tool_call_id"))
            items.append({"type": "function_call_output", "call_id": call_id, "output": responses_tool_output(content)})
        elif role == "assistant":
            replayed = responses_assistant_items(message, idx, provider=provider)
            # An empty assistant turn stays a turn: provider sanitation has
            # already supplied a valid content value where one is required.
            items.extend(replayed or [{"role": role, "content": content or ""}])
        elif content not in (None, "", []):
            items.append({"role": role, "content": responses_content(content)})
        else:
            items.append({"role": role, "content": content or ""})
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
