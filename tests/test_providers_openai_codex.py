import asyncio

import pytest

from opendde_harness.providers.openai_codex_provider import OpenAICodexProvider


def _provider():
    return OpenAICodexProvider(default_model="openai-codex/gpt-5.6-luna")


def test_certificate_failure_never_repeats_the_request_unverified(monkeypatch):
    """The request carries the account's OAuth token; an unverified retry leaks it."""
    calls = []

    async def token():
        return "sk-oauth-token", "account-1"

    async def request(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(
        "opendde_harness.providers.chatgpt_token.access_token_and_account",
        lambda: ("sk-oauth-token", "account-1"),
    )
    monkeypatch.setattr("opendde_harness.providers.openai_codex_provider._request_codex", request)

    response = asyncio.run(_provider().chat(messages=[{"role": "user", "content": "hi"}]))

    assert len(calls) == 1
    assert response.finish_reason == "error"
    assert "certificate" in response.content.lower()


@pytest.mark.parametrize("signature", ["verify=False", "verify = False"])
def test_provider_source_has_no_verification_switch(signature):
    from pathlib import Path

    source = Path("opendde_harness/providers/openai_codex_provider.py").read_text()

    assert signature not in source


# ---------------------------------------------------------------------------
# The stream, and what the next request gets back from it
# ---------------------------------------------------------------------------

from opendde_harness.providers import openai_codex_provider as codex  # noqa: E402

RS_1 = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "look first"}],
    "encrypted_content": "enc-1",
}
RS_2 = {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc-2"}
FC_1 = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": '{"path": "a"}'}
MSG_1 = {"type": "message", "id": "msg_real", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}


def _events(*items, terminal="response.done", status=None):
    out = []
    for item in items:
        out.append({"type": "response.output_item.added", "item": item})
        if item["type"] == "message":
            out.append({"type": "response.output_text.delta", "delta": item["content"][0]["text"]})
        out.append({"type": "response.output_item.done", "item": item})
    payload = {"usage": {"input_tokens": 10, "output_tokens": 5}}
    if status:
        payload["status"] = status
    out.append({"type": terminal, "response": payload})
    return out


def _consume(monkeypatch, events):
    async def fake_iter(response, first_token_timeout, idle_timeout):
        for event in events:
            yield event

    monkeypatch.setattr(codex, "_iter_sse", fake_iter)
    return asyncio.run(codex._consume_sse(None, 1.0, 1.0))


def test_response_done_is_a_clean_finish_not_a_truncation(monkeypatch):
    result = _consume(monkeypatch, _events(MSG_1, terminal="response.done"))

    assert result.finish_reason == "stop"
    assert result.truncated is False
    assert result.usage["prompt_tokens"] == 10


def test_a_stream_with_no_terminal_event_is_still_truncated(monkeypatch):
    events = _events(MSG_1)[:-1]

    result = _consume(monkeypatch, events)

    assert (result.finish_reason, result.truncated) == ("length", True)


def test_reasoning_items_are_kept_with_the_item_each_precedes(monkeypatch):
    result = _consume(monkeypatch, _events(RS_1, FC_1, RS_2, MSG_1))

    assert [item["id"] for item in result.reasoning_items] == ["rs_1", "rs_2"]
    assert result.order == ["rs_1", "fc_1", "rs_2", "msg_real"]
    assert result.messages == {"msg_real": "done"}
    block = result.thinking_blocks()[0]
    assert block["type"] == "reasoning" and block["provider"] == codex.REASONING_BLOCK_PROVIDER
    assert block["thinking"] == "look first"
    assert block["items"][0]["encrypted_content"] == "enc-1"


def test_reasoning_is_replayed_before_the_item_it_produced(monkeypatch):
    result = _consume(monkeypatch, _events(RS_1, FC_1, RS_2, MSG_1))
    assistant = {
        "role": "assistant",
        "content": "done",
        "tool_calls": [{"id": "call_1|fc_1", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}],
        "thinking_blocks": result.thinking_blocks(),
    }

    _, items = codex._convert_messages(
        [{"role": "user", "content": "hi"}, assistant, {"role": "tool", "tool_call_id": "call_1|fc_1", "content": "x"}]
    )

    # Emission order, as pi replays it: the backend pairs each rs_* with the
    # item that followed it, and the model reads the transcript it produced.
    kinds = [(item.get("type", item.get("role")), item.get("id")) for item in items]
    assert kinds == [
        ("user", None),
        ("reasoning", "rs_1"),
        ("function_call", "fc_1"),
        ("reasoning", "rs_2"),
        ("message", "msg_real"),
        ("function_call_output", None),
    ]
    assert items[1] == RS_1 and items[3] == RS_2


def test_a_turn_without_reasoning_replays_as_before():
    assistant = {
        "role": "assistant",
        "content": "hi",
        "tool_calls": [{"id": "call_9", "function": {"name": "f", "arguments": "{}"}}],
    }

    _, items = codex._convert_messages([assistant])

    assert [item["type"] for item in items] == ["message", "function_call"]
    assert items[0]["id"] == "msg_0"


def test_two_consecutive_reasoning_items_are_both_replayed(monkeypatch):
    result = _consume(monkeypatch, _events(RS_1, RS_2, FC_1))
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1|fc_1", "function": {"name": "read_file", "arguments": "{}"}}],
        "thinking_blocks": result.thinking_blocks(),
    }

    _, items = codex._convert_messages([assistant])

    assert [item.get("id") for item in items] == ["rs_1", "rs_2", "fc_1"]


def test_an_unpaired_reasoning_item_is_not_sent_alone(monkeypatch):
    events = _events(RS_1, FC_1, RS_2)
    result = _consume(monkeypatch, events)
    assert result.order == ["rs_1", "fc_1", "rs_2"]

    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1|fc_1", "function": {"name": "read_file", "arguments": "{}"}}],
        "thinking_blocks": result.thinking_blocks(),
    }
    _, items = codex._convert_messages([assistant])

    assert [item.get("id") for item in items] == ["rs_1", "fc_1"]


def _captured_body(monkeypatch, *, reasoning_effort=None, tool_choice=None, tools=None, account="account-1"):
    seen = {}

    async def request(url, headers, body, timeout):
        seen["headers"], seen["body"] = headers, body
        return codex._CodexResult("ok", [], "stop")

    monkeypatch.setattr("opendde_harness.providers.chatgpt_token.access_token_and_account", lambda: ("tok", account))
    monkeypatch.setattr(codex, "_request_codex", request)
    response = asyncio.run(
        _provider().chat(
            [{"role": "user", "content": "hi"}], tools=tools, reasoning_effort=reasoning_effort, tool_choice=tool_choice
        )
    )
    assert response.finish_reason == "stop"
    return seen


def test_reasoning_is_sent_only_with_a_configured_effort_and_then_with_a_summary(monkeypatch):
    # What pi sends, and only when pi sends it.
    assert "reasoning" not in _captured_body(monkeypatch)["body"]
    assert _captured_body(monkeypatch, reasoning_effort="high")["body"]["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }


def test_tool_choice_and_strict_travel_in_responses_shape(monkeypatch):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}, "strict": True}}]
    body = _captured_body(monkeypatch, tools=tools, tool_choice={"type": "function", "function": {"name": "f"}})["body"]

    assert body["tool_choice"] == {"type": "function", "name": "f"}
    assert body["tools"] == [{"type": "function", "name": "f", "parameters": {"type": "object"}, "strict": True}]


def test_missing_account_id_omits_the_header_rather_than_sending_none(monkeypatch):
    headers = _captured_body(monkeypatch, account=None)["headers"]

    assert "chatgpt-account-id" not in headers
    assert headers["Authorization"] == "Bearer tok"


def test_reply_carries_the_reasoning_for_the_loop_to_store(monkeypatch):
    async def request(url, headers, body, timeout):
        return codex._CodexResult(
            "ok", [], "stop", reasoning_items=[RS_1], order=["rs_1", "msg_x"], messages={"msg_x": "ok"}
        )

    monkeypatch.setattr("opendde_harness.providers.chatgpt_token.access_token_and_account", lambda: ("tok", "a"))
    monkeypatch.setattr(codex, "_request_codex", request)

    response = asyncio.run(_provider().chat([{"role": "user", "content": "hi"}]))

    assert response.thinking_blocks[0]["items"] == [RS_1]
    assert response.reasoning_content == "look first"


def test_a_block_list_system_prompt_is_not_erased():
    system_prompt, _ = codex._convert_messages([{"role": "system", "content": [{"type": "text", "text": "be brief"}]}])

    assert system_prompt == "be brief"
