import sys
from types import SimpleNamespace

import pytest

from opendde_harness.cli import compute_environment
from opendde_harness.cli.compute_environment import check_local_platform, load_environment, resolve_device


@pytest.mark.parametrize(
    "available, requested, expected",
    [(False, "auto", "cpu"), (True, "auto", "cuda"), (True, "cpu", "cpu"), (True, "cuda:1", "cuda:1")],
)
def test_compute_device_resolution(monkeypatch, available, requested, expected):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: 2)),
    )
    assert resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["mps", "cuda:2", "gpu", "cuda:-1"])
def test_invalid_or_missing_compute_device_is_rejected(monkeypatch, requested):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
    )
    with pytest.raises(ValueError):
        resolve_device(requested)


def test_device_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("OPENDDE_HARNESS_COMPUTE_DEVICE", "cpu")
    assert resolve_device() == "cpu"


@pytest.mark.parametrize(
    "system, machine, ok", [("Linux", "x86_64", True), ("Linux", "aarch64", False), ("Darwin", "x86_64", False)]
)
def test_local_platform_check(monkeypatch, system, machine, ok):
    monkeypatch.setattr(compute_environment.platform, "system", lambda: system)
    monkeypatch.setattr(compute_environment.platform, "machine", lambda: machine)
    if ok:
        check_local_platform()
    else:
        with pytest.raises(ValueError, match="Linux x86-64 only"):
            check_local_platform()


def test_environment_contract_is_read_once():
    assert load_environment() is load_environment()
    assert load_environment()["platform"] == "linux/amd64"
