from opendde_harness.tui_rpc.methods.cli_dispatch import _is_dispatch_compatible


def test_compute_serve_is_never_dispatched_in_chat():
    assert _is_dispatch_compatible(["compute", "serve"]) is False
    assert _is_dispatch_compatible(["compute", "serve", "--device", "cpu"]) is False


def test_one_shot_commands_are_dispatchable():
    assert _is_dispatch_compatible(["doctor"]) is True
    assert _is_dispatch_compatible(["compute", "prepare", "--help"]) is True


def test_interactive_entry_points_are_blocked():
    assert _is_dispatch_compatible(["tui"]) is False
    assert _is_dispatch_compatible(["onboard"]) is False
    assert _is_dispatch_compatible(["agent"]) is False
