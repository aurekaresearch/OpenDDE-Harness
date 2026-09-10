"""``model.overlay`` writes one field of the current model and puts it in front of the live loop;
the picker lists what an endpoint-addressed provider serves."""

import json
from types import SimpleNamespace

import pytest

from opendde_harness.config import loader
from opendde_harness.config.schema import Config, ModelOverlay
from opendde_harness.providers.wire import merge_key
from opendde_harness.tui_rpc.errors import ConfigValidationError
from opendde_harness.tui_rpc.methods import model as model_methods
from opendde_harness.tui_rpc.methods.model import model_overlay


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "agents": {"defaults": {"model": "custom/deep-thinker", "provider": "custom"}},
                "providers": {"custom": {"apiKey": "k", "apiBase": "http://relay/v1"}},
            }
        )
    )
    monkeypatch.setattr(loader, "_current_config_path", path)
    return path


def _loop():
    inner = SimpleNamespace(model_overlays={})
    lazy = SimpleNamespace(_provider=SimpleNamespace(_inners=[inner]))
    loop = SimpleNamespace(settings=SimpleNamespace(model_overlays={}), provider=lazy, context_window_tokens=None)

    def refresh():
        from opendde_harness.providers.catalog import overlay_for
        from opendde_harness.providers.rates import resolve_context_window

        overlay = overlay_for(loop.settings.model_overlays, "custom/deep-thinker")
        loop.context_window_tokens = resolve_context_window("custom/deep-thinker", overlay=overlay).tokens

    loop.refresh_context_window = refresh
    return loop, inner


async def test_the_level_is_persisted_and_applied_to_the_running_providers(config_path):
    key = merge_key("custom", "deep-thinker")
    loop, inner = _loop()

    result = await model_overlay({"field": "reasoning_effort", "value": "high"}, lambda: loop)

    assert result == {"model": "custom/deep-thinker", "field": "reasoning_effort", "value": "high"}
    stored = Config.model_validate(json.loads(config_path.read_text())).providers.model_overlays()
    assert stored[key].reasoning_effort == "high"
    assert loop.settings.model_overlays[key] == ModelOverlay(reasoningEffort="high")
    assert inner.model_overlays[key].reasoning_effort == "high"

    assert (await model_overlay({"field": "reasoning_effort", "value": "default"}, lambda: loop))["value"] is None
    assert inner.model_overlays[key].reasoning_effort is None


async def test_a_context_window_declared_in_session_sizes_the_loop_at_once(config_path):
    loop, _ = _loop()

    result = await model_overlay({"field": "context_window_tokens", "value": "128k"}, lambda: loop)

    assert result["value"] == 128_000 and result["context_window_tokens"] == 128_000
    assert loop.context_window_tokens == 128_000

    cleared = await model_overlay({"field": "context_window_tokens", "value": "default"}, lambda: loop)
    assert cleared["context_window_tokens"] is None


@pytest.mark.parametrize(
    "params",
    [
        {"field": "reasoning_effort", "value": "ultra"},
        {"field": "context_window_tokens", "value": "lots"},
        {"field": "context_window_tokens", "value": "0"},
        {"field": "colour", "value": "blue"},
    ],
)
async def test_a_bad_field_or_value_is_refused_before_anything_is_written(config_path, params):
    before = config_path.read_text()

    with pytest.raises(ConfigValidationError):
        await model_overlay(params, None)

    assert config_path.read_text() == before


def test_an_endpoint_provider_lists_what_its_endpoint_serves_and_the_model_in_use(monkeypatch):
    asked = []

    def fake_endpoint_models(api_base, api_key, *, timeout=5.0):
        asked.append((api_base, api_key))
        return ("gpt-4.1", "deepseek-v4-flash")

    monkeypatch.setattr("opendde_harness.providers.endpoint_catalog.endpoint_models", fake_endpoint_models)
    section = SimpleNamespace(models=[], api_base="https://relay/v1", effective_api_key="sk", endpoints=[])

    entry = model_methods._build_provider_entry(
        "custom",
        current_provider="custom",
        current_model="custom/qwen-plus",
        providers={"custom": {"configured": True}},
        section=section,
    )

    assert asked == [("https://relay/v1", "sk")]
    assert entry["models"][0] == "custom/qwen-plus"
    assert "gpt-4.1" in entry["models"] and "deepseek-v4-flash" in entry["models"]


def test_a_vendor_with_a_catalogue_is_not_asked_for_its_model_list(monkeypatch):
    monkeypatch.setattr(
        "opendde_harness.providers.endpoint_catalog.endpoint_models",
        lambda *a, **k: pytest.fail("a catalogued vendor must not be asked"),
    )
    section = SimpleNamespace(models=[], api_base="https://api.deepseek.com", effective_api_key="sk", endpoints=[])

    entry = model_methods._build_provider_entry(
        "deepseek", current_provider=None, providers={"deepseek": {"configured": True}}, section=section
    )

    assert entry["models"]
