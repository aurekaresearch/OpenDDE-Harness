"""The bundled models.dev snapshot and the readers over it."""

import json

import pytest

from opendde_harness.config.schema import ModelOverlay
from opendde_harness.providers import catalog
from opendde_harness.providers.catalog import (
    SNAPSHOT,
    SOURCE_SNAPSHOT,
    describe,
    model_cost,
    native_limit,
    overlay_for,
    served_limit,
)
from opendde_harness.providers.wire import merge_key

#: Providers a fresh install must label offline. A total would pass while a
#: whole provider silently dropped out of the refresh (see the script's
#: ``reachable_providers``), so the list is asserted by name.
LABELLED_PROVIDERS = (
    "aihubmix",
    "anthropic",
    "dashscope",
    "deepseek",
    "gemini",
    "minimax",
    "moonshot",
    "openai",
    "openrouter",
    "zai",
)


@pytest.fixture
def fake_snapshot(monkeypatch):
    """Swap the bundled file for a hand-written table, and restore the cache."""

    def install(table):
        monkeypatch.setattr(catalog, "_snapshot", lambda: table)
        return table

    yield install


def test_snapshot_is_pinned_and_under_the_size_gate():
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert set(data["_source"]) == {"repo", "ref", "sha"}
    assert len(data["_source"]["sha"]) == 40
    assert SNAPSHOT.stat().st_size < 1024 * 1024


def test_every_labelled_provider_has_named_rows():
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    for provider in LABELLED_PROVIDERS:
        rows = data[provider]["models"]
        assert rows, provider
        assert all(row.get("name") for row in rows.values()), provider


def test_canonical_rows_carry_a_positive_context():
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    native = data["_models"]
    assert len(native) > 300
    assert all(row["limit"]["context"] > 0 for row in native.values())
    assert native["deepseek/deepseek-v4-flash"]["limit"]["context"] == 1_000_000


def test_inheriting_rows_keep_both_limits():
    # Provider rows that restate one limit field inherit the other from the
    # vendor's definition; a shallow merge in the generator lost it on 298
    # rows and answered OpenRouter's deepseek-v4-flash with a 16384 ceiling.
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert data["openrouter"]["models"]["deepseek/deepseek-v4-flash"]["limit"] == {
        "context": 1_048_576,
        "output": 384_000,
    }
    assert data["anthropic"]["models"]["claude-sonnet-4-5"]["limit"] == {"context": 1_000_000, "output": 64_000}
    assert served_limit("openrouter/deepseek/deepseek-v4-flash")["output"] == 384_000


def test_every_section_is_readable_by_a_stored_id():
    # A section under a spelling `split_model_id` never produces is dead
    # weight: the whole nano-gpt section once was.
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    sections = [key for key in data if not key.startswith("_")]
    assert all("-" not in key and key == key.lower() for key in sections)
    assert data["nano_gpt"]["models"]
    assert served_limit("nano-gpt/Doctor-Shotgun/MS3.2-24B-Magnum-Diamond")["context"] > 0


def test_describe_and_cost_read_the_provider_row():
    row = describe("deepseek", "deepseek-v4-flash")
    cost = model_cost("deepseek/deepseek-v4-flash")

    assert row.source == SOURCE_SNAPSHOT
    assert row.label == "DeepSeek V4 Flash"
    assert cost["input"] > 0


def test_served_limit_reads_only_the_provider_the_id_names():
    assert served_limit("deepseek/deepseek-v4-flash")["context"] == 1_000_000
    assert served_limit("custom/gpt-5.6-terra") is None  # no row for a custom endpoint
    assert served_limit("gpt-5.6-terra") is None  # a bare id names no provider


def test_native_limit_recognises_a_prefixed_id_by_its_model_name():
    assert "openai/gpt-5.6-terra" in catalog._snapshot()["_models"]
    assert native_limit("openai/gpt-5.6-terra")["context"] == 1_050_000
    # The same model behind a relay, a reseller, or a self-hosted server.
    assert native_limit("custom/gpt-5.6-terra")["context"] == 1_050_000
    assert native_limit("siliconflow/gpt-5.6-terra")["context"] == 1_050_000
    assert native_limit("minimax-global/claude-opus-5")["context"] == 1_000_000
    assert native_limit("gpt-5.6-terra") is None  # a bare id names no provider
    assert native_limit("custom/GPT-5.6-Terra") is None  # exact name, no case folding


def test_every_model_name_in_the_bundled_table_belongs_to_one_vendor():
    """What makes matching by name safe: the table never files one name under two vendors."""
    names = [key.partition("/")[2] for key in catalog._snapshot()["_models"]]

    assert len(names) == len(set(names))
    assert len(catalog._bare_index()) == len(names)


def test_a_relay_model_is_labelled_by_the_vendors_name():
    row = catalog.describe("custom", "deepseek-v4-flash")

    assert row.label == "DeepSeek-V4-Flash"


def test_native_limit_is_an_exact_key_match(fake_snapshot):
    fake_snapshot(
        {
            "_models": {
                "qwen/qwen3-32b": {"limit": {"context": 40_960}},
                "minimax/MiniMax-M3": {"limit": {"context": 1_000_000}},
            }
        }
    )

    # A self-hosted deployment is the vendor's model under another prefix.
    assert native_limit("hosted_vllm/qwen3-32b") == {"context": 40_960}
    assert native_limit("hostedVllm/qwen3-32b") == {"context": 40_960}
    assert native_limit("qwen3-32b") is None  # bare: names no provider
    assert native_limit("openrouter/qwen/qwen3-32b") is None  # not the table's key, and no name match
    assert native_limit("minimax/minimax-m3") is None  # no case folding
    assert native_limit("minimax/MiniMax-M3") == {"context": 1_000_000}


def test_a_damaged_snapshot_costs_answers_not_startup(fake_snapshot):
    fake_snapshot({})

    assert describe("deepseek", "deepseek-v4-flash").described is False
    assert model_cost("deepseek/deepseek-v4-flash") is None
    assert served_limit("deepseek/deepseek-v4-flash") is None
    assert native_limit("deepseek/deepseek-v4-flash") is None


def test_overlay_for_matches_by_identity_not_spelling():
    overlays = {merge_key("custom", "GPT-x"): ModelOverlay(contextWindowTokens=8_000)}

    assert overlay_for(overlays, "custom/gpt-x").context_window_tokens == 8_000
    assert overlay_for(overlays, "openai/gpt-x") is None  # another provider's model of the same name
    assert overlay_for(overlays, "gpt-x") is None
    assert overlay_for({}, "custom/gpt-x") is None


def test_the_built_in_row_answers_only_for_the_provider_it_is_written_for():
    from opendde_harness.providers.catalog import builtin_limit, model_reasoning

    assert builtin_limit("openai-codex/gpt-5.6-luna") == {"context": 272_000, "output": 128_000}
    assert builtin_limit("openai-codex/gpt-5.3-codex-spark")["context"] == 128_000
    assert builtin_limit("openai/gpt-5.6-luna") is None
    assert builtin_limit("openai-codex/gpt-9-unlisted") is None
    assert builtin_limit("gpt-5.6-luna") is None
    assert model_reasoning("openai-codex/gpt-5.6-luna") is True
