"""Which provider answers a model id, when several are configured."""

from opendde_harness.config.schema import Config

CUSTOM = {"custom": {"apiKey": "sk-corp", "apiBase": "https://llm.corp.internal/v1"}}
LOCAL = {"ollamaChat": {"apiBase": "http://127.0.0.1:11434"}}
GATEWAY = {"openrouter": {"apiKey": "sk-or"}}


def _route(providers, model):
    config = Config.model_validate({"providers": providers, "agents": {"defaults": {"model": model}}})
    return config._match_provider(model)[1]


def test_a_named_endpoint_answers_only_when_it_is_named():
    # The prompt and this key went to an endpoint the user never chose for the
    # model, while the picker still named the vendor.
    assert _route(CUSTOM, "anthropic/claude-sonnet-5") is None
    assert _route(CUSTOM, "gpt-5.5") is None
    assert _route(CUSTOM, "custom/gpt-5.5") == "custom"


def test_a_local_deployment_takes_unprefixed_ids_only():
    assert _route(LOCAL, "anthropic/claude-sonnet-5") is None
    # A bare id names nobody in particular, and a local box is likely serving it.
    assert _route(LOCAL, "qwen3-32b") == "ollama_chat"
    assert _route(LOCAL, "llama3.2") == "ollama_chat"


def test_a_keyword_gateway_still_routes_other_vendors():
    assert _route({**CUSTOM, **GATEWAY}, "anthropic/claude-sonnet-5") == "openrouter"


def test_the_vendor_itself_wins_when_it_has_credentials():
    providers = {**CUSTOM, **GATEWAY, "anthropic": {"apiKey": "sk-ant"}}
    assert _route(providers, "anthropic/claude-sonnet-5") == "anthropic"
