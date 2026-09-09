"""Budget trimming must never ship half a tool exchange."""

from opendde_harness.context_engine.history_trimmer import HistoryTrimmer

MESSAGES = [
    {"role": "user", "content": "start"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "function": {"name": "read", "arguments": "{}"}}],
    },
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


def test_unknown_window_ships_everything_and_reports_nothing_enforced():
    trimmer = _Trimmer(0)
    trimmer.context_window_tokens = None

    import opendde_harness.context_engine.history_trimmer as module

    original = module.estimate_prompt_tokens_chain
    module.estimate_prompt_tokens_chain = lambda provider, model, messages, tools: (10**9, "count")
    try:
        messages, outcome = trimmer.trim(
            session_messages=MESSAGES,
            ids=list(range(len(MESSAGES))),
            protected_ids=set(),
            reserved_output=0,
            build_messages=lambda history: list(history),
        )
    finally:
        module.estimate_prompt_tokens_chain = original

    # Nothing to trim against, so nothing is dropped: an invented window used
    # to cut real history here.
    assert outcome.included_ids == list(range(len(MESSAGES)))
    assert outcome.max_prompt_tokens is None
    assert outcome.ok and outcome.over_by == 0 and outcome.warnings == []


def test_protected_messages_survive_a_budget_they_do_not_fit():
    _, outcome = _trim(1, list(range(len(MESSAGES))), {0, 1, 2})

    assert outcome.included_ids == [0, 1, 2]
    assert not outcome.ok and outcome.over_by > 0


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


def test_a_tool_result_answering_a_protected_call_is_protected_with_it():
    """Protecting the assistant call but not its result used to drop both:
    the result was droppable and its exchange took the call along."""
    _, outcome = _trim(1, list(range(len(MESSAGES))), {0, 1})

    assert 1 in outcome.included_ids and 2 in outcome.included_ids
    assert HistoryTrimmer.structural_errors(outcome.history) == []
