import json

import pytest

from opendde_harness.config import loader
from opendde_harness.config._fields import write_json_atomic
from opendde_harness.config.loader import ConfigSchemaError, load_config, save_config
from opendde_harness.config.opendde_harness import OpenDDEHarnessConfig, load_opendde_harness_config
from opendde_harness.config.schema import Config
from opendde_harness.config.update import init_extension_block_defaults, set_skill_blocked
from opendde_harness.plugin.memory.longterm import _library


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(loader, "_current_config_path", path)
    monkeypatch.delenv("OPENDDE_HARNESS_LANGUAGE", raising=False)
    return path


def _write(path, data):
    write_json_atomic(path, data)


def test_loads_base_and_feature_blocks_in_either_spelling(config_path):
    _write(
        config_path,
        {
            "agents": {"defaults": {"model": "openai/gpt-4o", "max_tool_iterations": 9}},
            "skillForge": {"enabled": False, "router": {"topK": 2}},
            "memory": {"userId": "alice"},
            "plugins": {"disabled": ["long-term-memory"]},
        },
    )
    config = load_config()

    assert config.agents.defaults.model == "openai/gpt-4o"
    assert config.agents.defaults.max_tool_iterations == 9
    assert config.skill_forge.enabled is False
    assert config.skill_forge.router.top_k == 2
    assert config.memory.user_id == "alice"
    assert config.plugins.disabled == ["long-term-memory"]
    assert config.context.protect_first_n == 3


def test_missing_file_gives_defaults(config_path):
    config = load_config()
    assert config == Config()


def test_compat_names_are_the_single_root(config_path):
    assert OpenDDEHarnessConfig is Config
    assert load_opendde_harness_config is load_config


def test_unknown_key_is_rejected(config_path):
    _write(config_path, {"skillForge": {"noSuchKnob": 1}})
    with pytest.raises(ValueError, match="fails schema validation"):
        load_config()


@pytest.mark.parametrize(
    ("data", "key"),
    [
        ({"skillForge": {_library.EXECUTABLE: {"enabled": True}}}, f"skillForge.{_library.EXECUTABLE}"),
        ({"agents": {"defaults": {"model": "m", "contextWindowTokens": 65536}}, "channels": {}}, "channels"),
    ],
)
def test_keys_from_retired_releases_are_rejected_by_name(config_path, data, key):
    """Nothing migrates an old config: the retired key is named and the remedy is onboard."""
    _write(config_path, data)
    with pytest.raises(ConfigSchemaError) as info:
        load_config()

    message = str(info.value)
    assert key in message
    assert "ddeharness onboard" in message
    assert json.loads(config_path.read_text()) == data


def test_env_override_prefix(config_path, monkeypatch):
    _write(config_path, {"agents": {"defaults": {"model": "from-file"}}})
    monkeypatch.setenv("OPENDDE_HARNESS_LANGUAGE", "zh")
    monkeypatch.setenv("OPENDDE_HARNESS_AGENTS__DEFAULTS__MODEL", "from-env")
    config = load_config()

    assert config.language == "zh"
    # A value the file sets wins over the environment.
    assert config.agents.defaults.model == "from-file"


def test_cache_returns_isolated_copies_and_follows_the_file(config_path):
    _write(config_path, {"agents": {"defaults": {"model": "one"}}})
    first = load_config()
    first.agents.defaults.workspace = "/edited"
    second = load_config()

    assert second.agents.defaults.workspace != "/edited"
    assert second == load_config()

    _write(config_path, {"agents": {"defaults": {"model": "two"}}})
    assert load_config().agents.defaults.model == "two"


def test_save_writes_base_blocks_only_and_round_trips(config_path):
    config = Config()
    config.agents.defaults.model = "saved/model"
    save_config(config)
    on_disk = json.loads(config_path.read_text())

    assert set(on_disk) == {"agents", "cli", "providers", "tools", "language"}
    assert on_disk["agents"]["defaults"]["maxToolIterations"] == 40
    assert load_config() == config


def test_feature_blocks_round_trip_as_camel_case():
    config = Config.model_validate(
        {"skillForge": {"llmGateMaxSelect": 4}, "runtime": {"checkpoint": {"policy": "never"}}}
    )
    dumped = config.model_dump(by_alias=True)

    assert dumped["skillForge"]["llmGateMaxSelect"] == 4
    assert dumped["memory"]["memoryTopK"] == 5
    assert dumped["runtime"]["checkpoint"]["policy"] == "never"
    assert Config.model_validate(dumped) == config


def test_seeded_feature_defaults_load_and_patch_in_place(config_path):
    save_config(Config())
    init_extension_block_defaults()
    on_disk = json.loads(config_path.read_text())

    assert on_disk["memory"] == {"backend": "longterm", "userId": "default", "agentId": "default", "memoryTopK": 5}
    assert on_disk["skillForge"]["memory"] == {"enabled": True}
    assert on_disk["plugins"]["config"]["long-term-memory"] == {"base_url": "http://localhost:18791"}
    assert load_config().skill_forge.memory.enabled is True

    assert set_skill_blocked("Weather", True) == ["Weather"]
    assert load_config().skill_forge.blocklist == ["Weather"]
    assert json.loads(config_path.read_text())["skillForge"]["memory"] == {"enabled": True}
