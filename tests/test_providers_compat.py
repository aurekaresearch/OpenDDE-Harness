"""One table decides how each family switches thinking on, whoever serves it."""

import pytest

from opendde_harness.providers import compat
from opendde_harness.providers.compat import apply_thinking, can_reason, detect_compat, names_thinking


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("custom/deepseek-v3.2-thinking", "deepseek"),
        ("deepseek/deepseek-v4-flash", "deepseek"),
        ("custom/glm-5.2", "zai"),
        ("zai/glm-4.7", "zai"),
        ("custom/qwen-plus", "qwen"),
        ("dashscope/qwen3-max", "qwen"),
        ("custom/kimi-k2.5", "moonshot"),
        ("custom/grok-4", "xai"),
        ("custom/MiniMax-M2.5", "minimax"),
        ("openrouter/qwen/qwen3-max", "openrouter"),  # the gateway's own switch, whatever vendor the id names
        ("openrouter/deepseek/deepseek-v4-flash", "openrouter"),
        ("openrouter/anthropic/claude-sonnet-5", "openrouter"),
        ("custom/gpt-4.1-mini", "openai"),
        ("custom/gemini-2.5-flash", "openai"),
    ],
)
def test_family_is_read_off_the_id_not_the_section(model, family):
    assert detect_compat(model).family == family


def test_the_endpoint_host_can_name_the_family_too():
    assert detect_compat("custom/my-deployment", api_base="https://open.bigmodel.cn/api/paas/v4").family == "zai"


def test_thinking_variants_are_named_by_their_id():
    assert names_thinking("custom/deepseek-v3.2-thinking")
    assert names_thinking("deepseek/deepseek-reasoner")
    assert names_thinking("custom/deepseek-r1")
    assert not names_thinking("custom/deepseek-v3.2")
    assert not names_thinking("openai/gpt-4.1-mini")


def test_a_named_thinking_variant_is_switched_on_without_an_effort():
    kwargs = {}
    apply_thinking(
        kwargs,
        detect_compat("custom/deepseek-v3.2-thinking"),
        reasoning_effort=None,
        model="custom/deepseek-v3.2-thinking",
    )

    assert kwargs == {"extra_body": {"thinking": {"type": "enabled"}}}


def test_a_hybrid_model_keeps_its_vendors_default_without_an_effort():
    kwargs = {}
    apply_thinking(kwargs, detect_compat("custom/deepseek-v3.2"), reasoning_effort=None, model="custom/deepseek-v3.2")

    assert kwargs == {}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("custom/deepseek-v3.2", {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "high"}),
        # Z.ai takes on/off only; pi sends it no reasoning_effort either.
        ("custom/glm-5.2", {"extra_body": {"thinking": {"type": "enabled", "clear_thinking": False}}}),
        ("custom/qwen-plus", {"extra_body": {"enable_thinking": True, "thinking_budget": 16384}}),
        ("openrouter/deepseek/deepseek-v4-flash", {"extra_body": {"reasoning": {"effort": "high"}}}),
        ("openrouter/qwen/qwen3-14b", {"extra_body": {"reasoning": {"effort": "high"}}}),
        ("openrouter/anthropic/claude-sonnet-5", {"extra_body": {"reasoning": {"effort": "high"}}}),
        ("custom/grok-4", {}),  # xAI rejects reasoning_effort outright
        ("custom/kimi-k2.5", {}),
        ("custom/some-new-model", {"reasoning_effort": "high"}),
    ],
)
def test_an_effort_takes_each_familys_own_shape(model, expected):
    kwargs = {}
    apply_thinking(kwargs, detect_compat(model), reasoning_effort="high", model=model)

    assert kwargs == expected


def test_an_effort_is_withheld_from_a_model_the_catalogue_says_cannot_reason():
    assert can_reason("openai/gpt-4.1-mini") is False
    kwargs = {}
    apply_thinking(kwargs, detect_compat("openai/gpt-4.1-mini"), reasoning_effort="high", model="openai/gpt-4.1-mini")

    assert kwargs == {}


def test_a_users_extra_body_wins_over_the_shipped_switch(monkeypatch):
    kwargs = {"extra_body": {"thinking": {"type": "disabled"}, "top_p": 0.9}}
    apply_thinking(
        kwargs,
        detect_compat("custom/deepseek-v3.2-thinking"),
        reasoning_effort=None,
        model="custom/deepseek-v3.2-thinking",
    )

    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}, "top_p": 0.9}


def test_deepseek_is_the_family_that_needs_the_reasoning_key_back():
    assert detect_compat("custom/deepseek-v3.2").requires_reasoning_content
    assert not detect_compat("custom/glm-5.2").requires_reasoning_content
    assert compat._DEFAULT.supports_reasoning_effort


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("custom/deepseek-v3.2", {"extra_body": {"thinking": {"type": "disabled"}}}),
        ("custom/glm-5.2", {"extra_body": {"thinking": {"type": "disabled"}}}),
        ("custom/qwen-plus", {"extra_body": {"enable_thinking": False}}),
        ("openrouter/anthropic/claude-sonnet-5", {"extra_body": {"reasoning": {"effort": "none"}}}),
        ("custom/some-new-model", {"reasoning_effort": "none"}),
        ("custom/grok-4", {}),
    ],
)
def test_off_switches_thinking_off_explicitly(model, expected):
    kwargs = {}
    apply_thinking(kwargs, detect_compat(model), reasoning_effort="off", model=model)

    assert kwargs == expected


def test_off_overrides_a_thinking_variants_name():
    kwargs = {}
    model = "custom/deepseek-v3.2-thinking"
    apply_thinking(kwargs, detect_compat(model), reasoning_effort="off", model=model)

    assert kwargs == {"extra_body": {"thinking": {"type": "disabled"}}}


@pytest.mark.parametrize(
    ("effort", "max_tokens", "budget"),
    [
        ("minimal", None, 1024),
        ("low", None, 2048),
        ("medium", None, 8192),
        ("high", None, 16384),
        ("xhigh", None, 16384),
        ("high", 4096, 3072),  # kept under the ceiling with room for the answer
        ("high", 1024, None),  # no room at all: no budget field, thinking still on
    ],
)
def test_qwen_takes_its_depth_as_a_token_budget_under_the_ceiling(effort, max_tokens, budget):
    kwargs = {}
    apply_thinking(
        kwargs,
        detect_compat("custom/qwen-plus"),
        reasoning_effort=effort,
        model="custom/qwen-plus",
        max_tokens=max_tokens,
    )

    expected = {"enable_thinking": True}
    if budget:
        expected["thinking_budget"] = budget
    assert kwargs["extra_body"] == expected


def test_the_vocabulary_folds_onto_deepseeks_three_steps():
    kwargs = {}
    apply_thinking(kwargs, detect_compat("custom/deepseek-v3.2"), reasoning_effort="max", model="custom/deepseek-v3.2")
    assert kwargs["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("model", "level", "sent", "codex_sent"),
    [
        ("openai/gpt-6-astra", "max", "max", "max"),
        ("openai/gpt-5.6-luna", "max", "max", "max"),
        ("openai/gpt-5.5", "max", "xhigh", "xhigh"),
        ("openai/gpt-5.1", "xhigh", "high", "high"),
        ("openai/gpt-5.5", "minimal", "minimal", "low"),
        ("openai/gpt-5.1", "minimal", "minimal", "minimal"),
        ("openai/gpt-5.5", "off", "none", "none"),
        ("custom/some-new-model", "xhigh", "high", "high"),
    ],
)
def test_openai_levels_clamp_to_what_the_generation_takes_as_pi_maps_them(model, level, sent, codex_sent):
    from opendde_harness.providers.compat import openai_effort

    assert openai_effort(model, level) == sent
    assert openai_effort(model, level, codex=True) == codex_sent


def test_a_custom_section_pointed_at_openrouter_is_openrouter_too():
    compat = detect_compat("custom/qwen/qwen3-max", api_base="https://openrouter.ai/api/v1")

    assert compat.family == "openrouter"


def test_openrouter_takes_the_whole_vocabulary_unfolded():
    kwargs = {}
    apply_thinking(
        kwargs, detect_compat("openrouter/openai/gpt-5.5"), reasoning_effort="max", model="openrouter/openai/gpt-5.5"
    )

    assert kwargs == {"extra_body": {"reasoning": {"effort": "max"}}}
