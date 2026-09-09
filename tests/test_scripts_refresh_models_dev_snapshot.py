"""The snapshot generator's inheritance, which the vendored data depends on."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refresh_models_dev_snapshot.py"


@pytest.fixture(scope="module")
def refresh():
    spec = importlib.util.spec_from_file_location("refresh_models_dev_snapshot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANONICAL = {
    "deepseek/deepseek-v4-flash": {
        "name": "DeepSeek V4 Flash",
        "description": "Fast lane",
        "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.0028},
        "limit": {"context": 1_000_000, "input": 900_000, "output": 384_000},
    }
}


def test_a_restated_table_merges_key_by_key(refresh):
    # A reseller restating only its window keeps the inherited output ceiling;
    # a top-level replace dropped it, and the ceiling resolved to the estimate.
    providers = {
        "openrouter": {
            "models": {
                "deepseek/deepseek-v4-flash": {
                    "base_model": "deepseek/deepseek-v4-flash",
                    "cost": {"input": 0.09, "output": 0.18},
                    "limit": {"context": 1_048_576},
                }
            }
        }
    }

    refresh.resolve_inheritance(providers, CANONICAL)
    row = providers["openrouter"]["models"]["deepseek/deepseek-v4-flash"]

    assert row["limit"] == {"context": 1_048_576, "input": 900_000, "output": 384_000}
    assert row["cost"] == {"input": 0.09, "output": 0.18, "cache_read": 0.0028}
    assert row["name"] == "DeepSeek V4 Flash"
    assert "base_model" not in row


def test_base_model_omit_removes_inherited_paths_after_the_merge(refresh):
    providers = {
        "pioneer": {
            "models": {
                "flash": {
                    "base_model": "deepseek/deepseek-v4-flash",
                    "base_model_omit": ["limit.input", "cost.cache_read", "nope.deeper"],
                    "limit": {"context": 256_000},
                }
            }
        }
    }

    refresh.resolve_inheritance(providers, CANONICAL)
    row = providers["pioneer"]["models"]["flash"]

    assert row["limit"] == {"context": 256_000, "output": 384_000}
    assert row["cost"] == {"input": 0.14, "output": 0.28}
    # The canonical row is not what was edited.
    assert CANONICAL["deepseek/deepseek-v4-flash"]["limit"]["input"] == 900_000
    assert CANONICAL["deepseek/deepseek-v4-flash"]["cost"]["cache_read"] == 0.0028


def test_sections_are_keyed_by_the_normalized_provider_name(refresh):
    providers = {"nano-gpt": {"models": {"tiny": {"name": "Tiny", "limit": {"context": 32_768}}}}}

    snapshot = refresh.build(providers, wanted={"nano-gpt"})

    assert set(snapshot) == {"nano_gpt"}


def test_the_tarball_is_fetched_by_the_recorded_sha_not_the_branch_tip(refresh):
    assert refresh.tarball_url("abc123").endswith("/tar.gz/abc123")
    assert "refs/heads" not in refresh.tarball_url("abc123")
