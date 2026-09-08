"""Find the long-term memory roots on this machine and say what state each is in.

opendde used to assume there was exactly one root at one address and start a server
whenever that address did not answer. Both assumptions were wrong in ways that
cost users their memory: a root can already be served on another port, and a root
can belong to the user rather than to opendde_harness.

Discovery answers four questions per candidate root, and deliberately keeps them
apart because they fail independently:

``configured``
    Does ``[llm]`` carry a model and a key? Without it no server can finish
    starting, so an unconfigured root is a half-built one, not a broken one.
``declared_url``
    What address does the root's config toml say a server for this root listens
    on? The root is self-describing; this is the authority.
``alive``
    Does that address answer ``/health``?
``lock_held``
    Is the OME jobstore lock taken? That lock is per data directory, not per
    port, so it is the only reliable answer to "is something already serving
    this data". A root can be locked while its declared address is silent --
    that is what a server started on a different port looks like.

Nothing here writes, signals, or starts anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opendde_harness.plugin.memory.longterm._library import CONFIG_FILENAME
from opendde_harness.plugin.memory.longterm._server import (
    DEFAULT_MEMORY_BASE_URL,
    _probe_health,
    ome_lock_held,
)


@dataclass(frozen=True)
class RootState:
    """One candidate memory root and what could be observed about it."""

    root: Path
    owned: bool
    configured: bool
    declared_url: str | None
    alive: bool
    lock_held: bool

    @property
    def exists(self) -> bool:
        return (self.root / CONFIG_FILENAME).is_file()

    @property
    def serving(self) -> bool:
        """A server for this root is up and reachable where it says it is."""
        return self.alive

    @property
    def busy_elsewhere(self) -> bool:
        """Something serves this data, but not at the address it declares.

        A server started with a ``--port`` override, an ``[api]`` port override
        in its environment, or a non-server holder of the lock (the library's
        demo command, an embedded engine) all land here. It is the one state that cannot be fixed
        by starting something: one data directory admits one engine.
        """
        return self.lock_held and not self.alive


def _describe(root: Path, *, owned: bool) -> RootState:
    from opendde_harness.config.update_memory import role_configured_in

    data = _read_toml(root)
    api = data.get("api") or {}
    host, port = api.get("host"), api.get("port")
    declared = f"http://{host}:{port}" if host and port else None

    return RootState(
        root=root,
        owned=owned,
        # Through the ops layer rather than re-reading the fields here: one
        # definition of "configured", shared with the wizard and doctor.
        configured=role_configured_in(data, "llm"),
        declared_url=declared,
        alive=_probe_health(declared) if declared else False,
        lock_held=ome_lock_held(root),
    )


def _read_toml(root: Path) -> dict:
    """Parse the root's config toml, or ``{}`` when absent or unreadable.

    Read straight off the path instead of through ``load_memory_config``: that
    helper follows the *active* root, and discovery is precisely the code that
    cannot assume which root is active yet.
    """
    import tomllib

    path = root / CONFIG_FILENAME
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def discover() -> list[RootState]:
    """Candidate roots, best first.

    Order is the preference order, not a ranking of health: a recorded root wins
    even when it is in a worse state than another candidate, because switching
    roots behind the user's back would silently change which memories opendde has.

    Only roots opendde creates for itself are scanned. A memory service the user runs is
    never discovered: finding one means offering it, offering it means asking
    for a decision the user did not come to make, and the only answer opendde can
    honour -- read-only reuse -- is one it cannot infer from a path anyway.
    Pointing opendde at such a server is an explicit turn in the wizard where the
    person who knows the address types it.
    """
    from opendde_harness.config.update_memory import (
        _recorded_slice,
        default_memory_root,
        root_is_opendde_harness_owned,
    )

    states: list[RootState] = []
    seen: set[Path] = set()

    def add(root: Path, *, owned: bool) -> RootState:
        state = _describe(root, owned=owned)
        states.append(state)
        seen.add(root)
        return state

    slice_ = _recorded_slice()
    recorded = slice_.get("root")
    if recorded:
        root = Path(str(recorded)).expanduser()
        owned = bool(slice_["owned"]) if "owned" in slice_ else root_is_opendde_harness_owned(root)
        add(root, owned=owned)

    default = default_memory_root()
    if default not in seen:
        add(default, owned=True)

    return states


def pick(states: list[RootState]) -> RootState | None:
    """The root opendde should use, or ``None`` when a new one has to be created.

    Prefers a root that is already configured -- a half-built root has nothing
    to reuse -- and among those keeps discovery order, which puts the recorded
    root first. ``configured`` alone is the test: it can only be true of a root
    whose toml was read, so re-checking that the file exists adds a stat and no
    information.
    """
    for state in states:
        if state.configured:
            return state
    return None


def default_new_root_url() -> str:
    """The address a freshly created root should declare.

    Not the ``[api]`` default the library ships (8000): that port is one of the most
    commonly occupied on a developer machine, and it only ever appeared in
    opendde's roots because opendde overrode it on the command line and never wrote
    the file.
    """
    return DEFAULT_MEMORY_BASE_URL


__all__ = ["RootState", "default_new_root_url", "discover", "pick"]
