"""Budget trimming must never ship half a tool exchange."""

from opendde_harness.context_engine.history_trimmer import HistoryTrimmer

MESSAGES = [
    {"role": "user", "content": "start"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "read", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    {"role": "assistant", "content": "done"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "ok"},
]


class _Trimmer(HistoryTrimmer):
    """A trimmer whose only budget is the number of messages."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.provider = None
        self.model = "fake/model"
        self.context_window_tokens = 100
        self.get_tool_definitions = list

    def _estimate(self, messages):
        return len(messages) * 10


def _trim(limit_messages: int, ids: list[int], protected: set[int]):
    trimmer = _Trimmer(limit_messages)

    def estimate(provider, model, messages, tools):
        return len(messages) * 10, "count"

    import opendde_harness.context_engine.history_trimmer as module

    original = module.estimate_prompt_tokens_chain
    module.estimate_prompt_tokens_chain = estimate
    try:
        trimmer.context_window_tokens = limit_messages * 10
        return trimmer.trim(
            session_messages=MESSAGES,
            ids=ids,
            protected_ids=protected,
            reserved_output=0,
            build_messages=lambda history: list(history),
        )
    finally:
        module.estimate_prompt_tokens_chain = original


def test_dropping_takes_the_whole_exchange():
    _, outcome = _trim(4, list(range(len(MESSAGES))), set())

    assert HistoryTrimmer.structural_errors(outcome.history) == []
    # The call and its result leave together, rather than the call being
    # stranded with no result.
    assert 1 not in outcome.included_ids and 2 not in outcome.included_ids


def test_history_still_starts_at_a_user_message():
    _, outcome = _trim(3, list(range(len(MESSAGES))), set())

    assert HistoryTrimmer.structural_errors(outcome.history) == []
    assert not outcome.history or outcome.history[0]["role"] == "user"
