"""Token estimates are additive and cached by content, so a turn costs the new text."""

from opendde_harness.utils import helpers
from opendde_harness.utils.helpers import count_text_tokens, estimate_message_tokens, estimate_prompt_tokens


def test_a_repeated_estimate_never_encodes_twice(monkeypatch):
    encodes = []
    real = helpers.tiktoken.get_encoding

    class _Counting:
        def __init__(self, enc):
            self._enc = enc

        def encode(self, text):
            encodes.append(len(text))
            return self._enc.encode(text)

    monkeypatch.setattr(helpers.tiktoken, "get_encoding", lambda name: _Counting(real(name)))
    helpers._TOKEN_COUNT_CACHE.clear()
    message = {"role": "user", "content": "the same words, counted once " * 50}

    first = estimate_message_tokens(message)
    second = estimate_message_tokens(dict(message))

    assert first == second > 0
    assert len(encodes) == 1


def test_the_prompt_estimate_is_the_sum_of_its_messages_and_tools():
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello there"}]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    assert estimate_prompt_tokens(messages, tools) == sum(
        estimate_message_tokens(m) for m in messages
    ) + count_text_tokens(helpers.json.dumps(tools, ensure_ascii=False))
    assert estimate_prompt_tokens([], None) == 0


def test_an_edited_message_is_a_miss_not_a_stale_hit():
    message = {"role": "tool", "tool_call_id": "c1", "content": "x" * 4000}
    before = estimate_message_tokens(message)
    message["content"] = "[elided]"

    assert estimate_message_tokens(message) < before


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(helpers, "_TOKEN_COUNT_CACHE_MAX", 8)
    helpers._TOKEN_COUNT_CACHE.clear()
    for i in range(20):
        count_text_tokens(f"payload number {i}")

    assert len(helpers._TOKEN_COUNT_CACHE) == 8


def test_counting_from_many_threads_at_once_is_safe():
    import threading

    from opendde_harness.utils import helpers

    helpers._TOKEN_COUNT_CACHE.clear()
    errors: list[BaseException] = []
    texts = [f"text {i} " * 20 for i in range(helpers._TOKEN_COUNT_CACHE_MAX + 200)]

    def work(offset: int):
        try:
            for i in range(len(texts)):
                helpers.count_text_tokens(texts[(i + offset) % len(texts)])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k * 700,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(helpers._TOKEN_COUNT_CACHE) <= helpers._TOKEN_COUNT_CACHE_MAX
