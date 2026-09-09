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
        ("openrouter/qwen/qwen3-max", "qwen"),  # the vendor named in the id wins over the gateway
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
        (
            "custom/glm-5.2",
            {"extra_body": {"thinking": {"type": "enabled", "clear_thinking": False}}, "reasoning_effort": "high"},
        ),
        ("custom/qwen-plus", {"extra_body": {"enable_thinking": True}}),
        (
            "openrouter/deepseek/deepseek-v4-flash",
            {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "high"},
        ),
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
