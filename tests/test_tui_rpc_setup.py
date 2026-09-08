import pytest

from opendde_harness.plugin.registry import PluginRegistry
from opendde_harness.tui_rpc.dispatcher import Dispatcher
from opendde_harness.tui_rpc.methods import setup


@pytest.mark.asyncio
async def test_setup_status_never_treats_missing_or_invalid_config_as_ready(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(setup, "_config_path", lambda: path)
    assert (await setup.setup_status({}))["provider_configured"] is False
    path.write_text("{broken")
    assert (await setup.setup_status({}))["provider_configured"] is False
    assert path.read_text() == "{broken"


@pytest.mark.asyncio
async def test_setup_status_reports_compute_and_provider_separately(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(setup, "_config_path", lambda: path)
    path.write_text('{"plugins": {"config": {"protein-design": {"compute_url": "http://127.0.0.1:8080"}}}}')
    status = await setup.setup_status({})
    assert status == {"provider_configured": False, "compute_configured": True}


@pytest.mark.asyncio
async def test_setup_status_without_readiness_plugins_reports_compute_configured(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(setup, "_config_path", lambda: path)
    monkeypatch.setattr(setup, "active_registry", PluginRegistry)
    path.write_text('{"plugins": {"config": {"protein-design": "not an object"}}}')
    status = await setup.setup_status({})
    assert status == {"provider_configured": False, "compute_configured": True}


@pytest.mark.asyncio
async def test_setup_registers_only_status(monkeypatch):
    dispatcher = Dispatcher()
    setup.register_setup_methods(dispatcher)
    monkeypatch.setattr(setup, "_config_path", lambda: None)
    response = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "setup.run", "params": {}})
    assert response["error"]["code"] == -32601
