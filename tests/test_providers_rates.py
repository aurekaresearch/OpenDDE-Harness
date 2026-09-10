"""The window and output-ceiling ladders: declared, tabled, bundled, or unknown."""

import httpx
import pytest

from opendde_harness.config.schema import ModelOverlay
from opendde_harness.providers import rates
from opendde_harness.providers.rates import (
    SOURCE_BUILTIN,
    SOURCE_ESTIMATED,
    SOURCE_LITELLM,
    SOURCE_NATIVE,
    SOURCE_OVERLAY,
    SOURCE_SERVED,
    SOURCE_UNKNOWN,
    resolve_context_window,
    resolve_max_output_tokens,
)

# Captured before the autouse fixture below replaces them, for the one
# test that exercises the real tier.
_REAL_WINDOW_TIER = rates._try_litellm_context_window
_REAL_OUTPUT_TIER = rates._try_litellm_max_output


@pytest.fixture(autouse=True)
def no_litellm(monkeypatch):
    """Answer the LiteLLM tier with nothing unless a test says otherwise.

    Keeps the suite independent of the pinned table's contents, and keeps
    the import (seconds) out of every test that is not about that tier.
    """
    monkeypatch.setattr(rates, "_try_litellm_context_window", lambda model, *, allow_import=True: None)
    monkeypatch.setattr(rates, "_try_litellm_max_output", lambda model, *, allow_import=True: None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the window path must not reach the network")

    monkeypatch.setattr(httpx, "Client", refuse)
    monkeypatch.setattr("urllib.request.urlopen", refuse)


def test_shipped_default_model_resolves_from_the_bundled_snapshot():
    resolved = resolve_context_window("deepseek/deepseek-v4-flash")

    assert resolved.tokens == 1_000_000
    assert resolved.source == SOURCE_SERVED
    assert resolved.known


def test_provider_row_is_asked_before_the_vendor_row():
    # OpenRouter serves this model with a window that differs from DeepSeek's
    # own figure; the id names OpenRouter, so OpenRouter's row answers.
    via_gateway = resolve_context_window("openrouter/deepseek/deepseek-v4-flash")
    direct = resolve_context_window("deepseek/deepseek-v4-flash")

    assert via_gateway.source == SOURCE_SERVED
    assert direct.source == SOURCE_SERVED
    assert via_gateway.tokens != direct.tokens


def test_vendor_row_answers_when_the_provider_has_none(monkeypatch):
    from opendde_harness.providers import catalog

    monkeypatch.setattr(catalog, "served_limit", lambda model: None)

    resolved = resolve_context_window("deepseek/deepseek-v4-flash")

    assert resolved.source == SOURCE_NATIVE
    assert resolved.tokens == 1_000_000


@pytest.mark.parametrize(
    ("model", "tokens"),
    [
        ("custom/gpt-5.6-terra", 1_050_000),  # the relay's prefix is where it is reached; the name is what it is
        ("custom/deepseek-v4-flash", 1_000_000),
        ("hosted_vllm/qwen3-32b", 131_072),
        ("ollama/qwen3-32b", 131_072),
        ("minimax-global/claude-opus-5", 1_000_000),
    ],
)
def test_a_prefixed_id_is_recognised_by_its_model_name(model, tokens):
    resolved = resolve_context_window(model)

    assert (resolved.tokens, resolved.source) == (tokens, SOURCE_NATIVE)


def test_a_relay_cap_is_declared_in_the_overlay_and_wins():
    resolved = resolve_context_window("custom/gpt-5.6-terra", overlay=ModelOverlay(contextWindowTokens=262_144))

    assert (resolved.tokens, resolved.source) == (262_144, SOURCE_OVERLAY)


@pytest.mark.parametrize(
    "model",
    [
        "aihubmix/some-model-it-does-not-list",
        "custom/Qwen3-32B",  # a name is matched exactly: no case folding
        "qwen3-32b",  # bare: names no provider
        "",
    ],
)
def test_unlisted_models_are_unknown_not_estimated(model):
    resolved = resolve_context_window(model)

    assert resolved.tokens is None
    assert resolved.source == SOURCE_UNKNOWN
    assert not resolved.known


def test_overlay_beats_every_table():
    declared = resolve_context_window("deepseek/deepseek-v4-flash", overlay=ModelOverlay(contextWindowTokens=32_768))

    assert (declared.tokens, declared.source) == (32_768, SOURCE_OVERLAY)


def test_overlay_sizes_a_deployment_no_table_lists():
    resolved = resolve_context_window("hosted_vllm/qwen3-32b", overlay=ModelOverlay(contextWindowTokens=32_768))

    assert (resolved.tokens, resolved.source) == (32_768, SOURCE_OVERLAY)


def test_the_provider_row_beats_litellm_and_litellm_beats_the_vendor_row(monkeypatch):
    from opendde_harness.providers import catalog

    monkeypatch.setattr(rates, "_try_litellm_context_window", lambda model, *, allow_import=True: 128_000)

    # LiteLLM's table is wrong in the refusing direction on a tenth of the
    # models both carry (MiniMax M2.5 at 1,000,000 against 204,800), so the
    # provider's own row answers first.
    served = resolve_context_window("deepseek/deepseek-v4-flash")
    assert (served.tokens, served.source) == (1_000_000, SOURCE_SERVED)

    with pytest.MonkeyPatch.context() as without_provider_row:
        without_provider_row.setattr(catalog, "served_limit", lambda model: None)
        tabled = resolve_context_window("deepseek/deepseek-v4-flash")
    assert (tabled.tokens, tabled.source) == (128_000, SOURCE_LITELLM)


def test_allow_import_is_forwarded_to_the_litellm_tier(monkeypatch):
    seen = []
    monkeypatch.setattr(
        rates, "_try_litellm_context_window", lambda model, *, allow_import=True: seen.append(allow_import) or None
    )

    resolve_context_window("hosted_vllm/qwen3-32b", allow_import=False)

    assert seen == [False]


def test_output_ceiling_ladder():
    from opendde_harness.providers import catalog

    served = resolve_max_output_tokens("deepseek/deepseek-v4-flash")
    with pytest.MonkeyPatch.context() as without_provider_row:
        without_provider_row.setattr(catalog, "served_limit", lambda model: None)
        native = resolve_max_output_tokens("deepseek/deepseek-v4-flash")
    declared = resolve_max_output_tokens("deepseek/deepseek-v4-flash", overlay=ModelOverlay(maxOutputTokens=4_096))
    unknown = resolve_max_output_tokens("aihubmix/some-model-it-does-not-list")
    empty = resolve_max_output_tokens(None)

    assert (served.tokens, served.source) == (384_000, SOURCE_SERVED)
    assert native.source == SOURCE_NATIVE and native.tokens > 0
    assert (declared.tokens, declared.source) == (4_096, SOURCE_OVERLAY)
    assert (unknown.tokens, unknown.source) == (rates.DEFAULT_MAX_OUTPUT_TOKENS, SOURCE_ESTIMATED)
    assert (empty.tokens, empty.source) == (rates.DEFAULT_MAX_OUTPUT_TOKENS, SOURCE_ESTIMATED)
    assert not unknown.known


def test_no_window_constant_survives():
    # The 65,536 stand-in is what this ladder replaced; nothing may reintroduce it.
    assert not hasattr(rates, "DEFAULT_CONTEXT_WINDOW_TOKENS")
    assert not hasattr(rates, "effective_context_window")


def test_the_litellm_tier_reads_the_static_table_only(monkeypatch):
    """``get_model_info`` reaches the network for some drivers (ollama asks its
    server); the window ladder runs on the event loop and must not."""
    import sys

    fake = type("L", (), {})()
    fake.model_cost = {"ollama/qwen3-32b": {"max_input_tokens": 40960, "max_output_tokens": 8192}}

    def never(*a, **k):
        raise AssertionError("get_model_info must not be called")

    fake.get_model_info = never
    monkeypatch.setattr(rates, "_litellm_price_table", lambda: fake.model_cost)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(rates, "_try_litellm_context_window", _REAL_WINDOW_TIER)
    monkeypatch.setattr(rates, "_try_litellm_max_output", _REAL_OUTPUT_TIER)

    assert rates._try_litellm_context_window("ollama/qwen3-32b") == 40960
    assert rates._try_litellm_max_output("ollama/qwen3-32b") == 8192
    assert rates._try_litellm_context_window("ollama/unknown-model") is None


def test_codex_models_resolve_from_the_built_in_row_not_the_api_row():
    """The ChatGPT backend serves OpenAI's models behind its own window; the
    API row for the same id (1,050,000) would refuse every request past 272k."""
    window = resolve_context_window("openai-codex/gpt-5.6-luna")
    output = resolve_max_output_tokens("openai-codex/gpt-5.6-luna")

    assert (window.tokens, window.source) == (272_000, SOURCE_BUILTIN)
    assert (output.tokens, output.source) == (128_000, SOURCE_BUILTIN)
    assert resolve_context_window("openai-codex/gpt-5.3-codex-spark").tokens == 128_000
    assert resolve_context_window("openai-codex/not-a-slug-the-backend-offers").tokens is None
    # An overlay still wins over the built-in row.
    assert (
        resolve_context_window("openai-codex/gpt-5.6-luna", overlay=ModelOverlay(contextWindowTokens=1000)).tokens
        == 1000
    )
