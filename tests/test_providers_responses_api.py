from opendde_harness.providers.responses_api import (
    responses_content,
    responses_input,
    responses_tool_choice,
    responses_tools,
    responses_usage,
    web_search_action,
    web_search_preview,
)


def test_responses_input_translates_tool_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "exec", "arguments": {"cmd": "ls"}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]
    items = responses_input(messages)

    assert items[0] == {"role": "system", "content": "sys"}
    assert items[1]["content"] == [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "data:x"},
    ]
    assert items[2] == {"type": "function_call", "call_id": "c1", "name": "exec", "arguments": '{"cmd": "ls"}'}
    assert items[3] == {"type": "function_call_output", "call_id": "c1", "output": "out"}


def test_responses_tools_and_choice():
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    assert responses_tools(tools) == [
        {"type": "function", "name": "f", "parameters": {"type": "object"}, "description": "d"}
    ]
    assert responses_tools(None) is None
    assert responses_tool_choice({"type": "function", "function": {"name": "f"}}) == {"type": "function", "name": "f"}
    assert responses_tool_choice("auto") == "auto"


def test_responses_usage_maps_cache_fields():
    usage = responses_usage(
        {"usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 4}}}
    )
    assert usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cache_read_input_tokens": 4}


def test_web_search_preview_lists_sources():
    item = {"action": {"type": "search", "queries": ["cats"], "sources": [{"url": "https://a"}]}}
    arguments, display = web_search_action(item)
    assert arguments == {"action": "search", "queries": ["cats"], "query": "cats"}
    assert display == "cats"
    preview = web_search_preview(item, [("https://a", "A"), ("https://b", "B")])
    assert preview.splitlines() == ["cats", "Sources (1):", "- A - https://a"]


def test_responses_content_keeps_unknown_blocks_as_text():
    assert responses_content("plain") == "plain"
    assert responses_content([{"type": "weird", "x": 1}]) == [
        {"type": "input_text", "text": '{"type": "weird", "x": 1}'}
    ]


def test_failed_response_never_returns_tool_calls():
    """A failed run can still carry a function_call; executing it runs a tool the backend gave up on."""
    from opendde_harness.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(api_key="sk-test", default_model="openai/gpt-5", provider_name="openai")
    failed = {
        "status": "failed",
        "error": {"code": "server_error", "message": "upstream exploded"},
        "output": [{"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"}],
        "output_text": "",
    }

    response = provider._parse_responses_response(failed)

    assert response.finish_reason == "error"
    assert response.tool_calls == []
    assert "server_error" in response.content


def test_incomplete_response_separates_a_filter_from_a_token_limit():
    from opendde_harness.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(api_key="sk-test", default_model="openai/gpt-5", provider_name="openai")
    base = {"status": "incomplete", "output": [], "output_text": "half an answer"}

    length = provider._parse_responses_response({**base, "incomplete_details": {"reason": "max_output_tokens"}})
    filtered = provider._parse_responses_response({**base, "incomplete_details": {"reason": "content_filter"}})

    assert (length.finish_reason, length.truncated) == ("length", True)
    assert (filtered.finish_reason, filtered.truncated) == ("content_filter", False)


def test_a_responses_turn_replays_its_reasoning_in_emission_order():
    from opendde_harness.providers.responses_api import PROVIDER_OPENAI, reasoning_block

    rs = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"}
    block = reasoning_block([rs], ["rs_1", "fc_1"], {}, provider=PROVIDER_OPENAI)
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1|fc_1", "function": {"name": "read", "arguments": "{}"}}],
        "thinking_blocks": [block],
    }

    items = responses_input([assistant, {"role": "tool", "tool_call_id": "call_1|fc_1", "content": "x"}])

    assert [(i.get("type"), i.get("id")) for i in items] == [
        ("reasoning", "rs_1"),
        ("function_call", "fc_1"),
        ("function_call_output", None),
    ]
    assert items[0] == rs
    assert items[1]["call_id"] == "call_1" and items[2]["call_id"] == "call_1"


def test_another_backends_reasoning_is_not_replayed_and_leaves_no_ids_to_validate():
    from opendde_harness.providers.responses_api import PROVIDER_CODEX, reasoning_block

    block = reasoning_block([{"type": "reasoning", "id": "rs_9"}], ["rs_9", "fc_9"], {}, provider=PROVIDER_CODEX)
    assistant = {
        "role": "assistant",
        "content": "hi",
        "tool_calls": [{"id": "call_9|fc_9", "function": {"name": "read", "arguments": "{}"}}],
        "thinking_blocks": [block],
    }

    items = responses_input([assistant])

    assert [i["type"] for i in items] == ["message", "function_call"]
    assert "id" not in items[1]


def test_each_output_message_replays_under_its_own_id_with_its_own_text():
    """A response can carry reasoning between two messages; the pairing only
    holds when each message keeps its id and text."""
    from opendde_harness.providers.responses_api import PROVIDER_OPENAI, reasoning_block

    rs_1 = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e1"}
    rs_2 = {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "e2"}
    block = reasoning_block(
        [rs_1, rs_2],
        ["rs_1", "msg_1", "rs_2", "msg_2"],
        {"msg_1": "first", "msg_2": "second"},
        provider=PROVIDER_OPENAI,
    )
    assistant = {"role": "assistant", "content": "firstsecond", "thinking_blocks": [block]}

    items = responses_input([assistant])

    assert [i.get("id") for i in items] == ["rs_1", "msg_1", "rs_2", "msg_2"]
    assert items[1]["content"][0]["text"] == "first" and items[3]["content"][0]["text"] == "second"
