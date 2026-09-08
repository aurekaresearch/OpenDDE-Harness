"""Atomic operations for the long-term memory settings (``<root>/<config toml>``).

This module is the ONLY write path for the memory-model sections
(llm / embedding / rerank / multimodal) and for the ``[api]`` address. The
onboard wizard's memory step writes here; the memory library reads it back
through its own settings loader (user-level toml, its env prefix). It lives
apart from opendde's ``config.json`` because the library owns this channel.

Only those sections are writable; the rest the library ships (memory / sqlite /
lancedb) are preserved untouched on every write.

**Which root.** The library resolves its root from its root env variable
(default: a bare directory under the home). opendde does not read that variable
as an input — it *writes* it from the root recorded in ``plugins.config["long-term-memory"]["root"]``, so the
choice of root is an explicit, recorded decision rather than something inherited
from an ambient environment. A root picked up from the environment and never
written down was silent data loss waiting to happen: run opendde without the
variable and the memories are still on disk while opendde reports none.

**Which owner.** ``plugins.config["long-term-memory"]["owned"]`` says whether opendde
may write to that root at all. A root opendde created is opendde's to configure and
serve; a root the user manages is read-only — opendde records its address and
nothing else.

Boot sequence (called by ``make_backend`` / ``make_understand_media_tool``):

1. :func:`configure_memory_env` — the library's root env variable → the
   recorded root
2. :func:`ensure_memory_home` — create the config toml + ``ome.toml`` from
   shipped templates (skip if exists). Owned roots only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from opendde_harness.plugin.memory.longterm._library import (
    CONFIG_FILENAME,
    OME_CONFIG_FILENAME,
    ROOT_ENV_VAR,
    config_templates,
)

logger = logging.getLogger(__name__)

_DATA_SUBDIR = "memory"

WRITABLE_SECTIONS = ("llm", "embedding", "rerank", "multimodal", "api")

# Sections holding a model + credentials, as opposed to the address.
MODEL_SECTIONS = ("llm", "embedding", "rerank", "multimodal")


def default_memory_root() -> Path:
    """Where opendde puts its own memory home.

    Under opendde's data directory, so it follows ``--config`` and reads as
    opendde's property rather than a squatter in the library's default root.
    """
    return _data_dir() / _DATA_SUBDIR


def _data_dir() -> Path:
    from opendde_harness.config.paths import get_data_dir

    return get_data_dir()


def root_is_opendde_harness_owned(root: Path | str) -> bool:
    """True when ``root`` is the one opendde creates for itself.

    Used only to infer ``owned`` for a config that records a root without the
    field. Once recorded, the field is the answer -- a root the user points
    opendde at cannot be classified by its path.
    """
    return Path(root).expanduser() == default_memory_root()


def _recorded_slice() -> dict[str, Any]:
    """opendde's ``plugins.config["long-term-memory"]``, read as raw JSON.

    Raw rather than through the validated config so that ``ddeharness doctor`` and
    the runtime can ask "which root" without paying for schema validation, and
    so an unrelated validation error elsewhere cannot make the memory path
    unreadable. An absent or unparseable file reads as "nothing recorded".
    """
    from opendde_harness.config.loader import get_config_path

    try:
        with get_config_path().open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    plugins = data.get("plugins") or {}
    slice_ = (plugins.get("config") or {}).get("long-term-memory") if isinstance(plugins, dict) else None
    return slice_ if isinstance(slice_, dict) else {}


def memory_root() -> Path:
    """The active memory root: the recorded one, else the default."""
    recorded = _recorded_slice().get("root")
    if recorded:
        return Path(str(recorded)).expanduser()
    return default_memory_root()


def owned_memory_root() -> Path:
    """The root opendde may create in and write to.

    Deliberately not the same question as :func:`memory_root`. The active root
    can be one the user manages, and "opendde needs a root of its own" must never
    resolve to that one: a user who declines to share theirs would otherwise have
    it adopted, seeded with templates and overwritten with opendde's models.
    """
    if memory_owned():
        return memory_root()
    return default_memory_root()


def memory_owned() -> bool:
    """Whether opendde may write to and start the active root.

    ``False`` means read-only reuse: record the address, never touch the config,
    never start or stop the process. When the config records no ``owned``, the
    answer is inferred from the path, and anything opendde did not create is
    treated as not ours -- the conservative direction.
    """
    slice_ = _recorded_slice()
    if "owned" in slice_:
        return bool(slice_["owned"])
    return root_is_opendde_harness_owned(memory_root())


class MemoryRootNotOwnedError(RuntimeError):
    """A write was attempted against a root the user manages.

    Not a ``PermissionError``: that is a filesystem condition and callers catch
    it as one (``except OSError``), which would swallow exactly the signal this
    is meant to raise.
    """


def _require_owned(action: str) -> None:
    """Refuse a write unless opendde owns the active root.

    The read-only promise used to live only at the call sites that happened to
    remember it -- the same shape as the drift this whole change is about, where
    one rule was enforced in several places and one of them was wrong. Enforcing
    it at the write primitives means a new caller cannot quietly opt out.
    """
    if memory_owned():
        return
    raise MemoryRootNotOwnedError(f"refusing to {action}: {memory_root()} is managed by the user, not by opendde")


def get_memory_config_path() -> Path:
    """Path of the user-level memory config toml."""
    return memory_root() / CONFIG_FILENAME


def configure_memory_env(root: Path | str | None = None) -> None:
    """Point the memory library at ``root`` (default: the recorded root).

    Sets the library's root env variable so it resolves both its config file
    and its data directories (sqlite / lancedb / .index / ome.toml) under it.

    Assigns rather than ``setdefault``: an ambient value is not an
    input to opendde's choice of root. Following it silently pointed opendde at a
    root nothing had recorded, so the next run without the variable reported no
    memories while they sat on disk. Operators who want a different root record
    it in the config instead.

    Must run BEFORE the library's ``load_settings()`` -- which is ``@cache``-d
    -- first executes, or in-process library imports keep the earlier root.
    """
    resolved = Path(root).expanduser() if root is not None else memory_root()
    os.environ[ROOT_ENV_VAR] = str(resolved)


def ensure_memory_home(root: Path | str | None = None) -> None:
    """Ensure the memory home directory has the required config files.

    Two steps, both idempotent:

    1. **Create** the config toml from the shipped template if absent.
       Users who already ran ``ddeharness onboard`` have this file; new
       installs get the template with empty API keys (onboard fills
       them later).
    2. **Create** ``ome.toml`` from the shipped template if absent.
       Without this file the OME engine's ``ConfigReloader`` raises
       ``FileNotFoundError`` and the memory backend silently degrades.

    Callers must gate this on :func:`memory_owned`: dropping template files into
    a root the user manages is an unrequested write, and "the files are usually
    there already" is not a basis for a read-only promise.
    """
    _require_owned("create config templates in")
    base = Path(root).expanduser() if root is not None else memory_root()
    base.mkdir(parents=True, exist_ok=True)

    config_toml = base / CONFIG_FILENAME
    ome_toml = base / OME_CONFIG_FILENAME

    templates = config_templates()
    if templates is None:
        return

    for target, template in [
        (config_toml, templates[0]),
        (ome_toml, templates[1]),
    ]:
        if target.exists():
            continue
        shutil.copy2(template, target)
        logger.info("created %s from template", target)


def load_memory_config() -> dict[str, Any]:
    """Return the parsed user-level toml, or ``{}`` when absent."""
    path = get_memory_config_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as TOML via temp-file + rename, so a Ctrl+C mid-write
    never leaves a half-written toml.

    No sidecar lock: the memory server owns ``<root>/.lock`` as a lock file,
    and the ``.lock/`` directory that ``atomic_replace`` creates would shadow it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        tomli_w.dump(data, f)
    os.replace(tmp, path)


def memory_section(section: str) -> dict[str, Any]:
    """Current values of a memory config section, or ``{}``."""
    return load_memory_config().get(section, {}) or {}


def role_configured_in(data: dict[str, Any], section: str) -> bool:
    """Whether ``section`` of an already-parsed config toml counts as configured.

    Same criterion as :func:`memory_role_configured`, applied to a toml the caller
    read itself. Discovery needs this: it inspects candidate roots before any of
    them is the active one, and a second hand-rolled "does it have a key" check
    is exactly what made two callers disagree once before.
    """
    sec = data.get(section) or {}
    return bool(sec.get("model") and sec.get("api_key"))


def memory_role_configured(section: str) -> bool:
    """True iff the user really configured this memory role.

    Sole criterion for "configured", shared by every caller: model AND api_key.
    The shipped config template seeds each section's model name with an
    empty api_key, so a model alone also holds on a fresh install -- two callers
    disagreeing on this made the wizard's Back loop on itself forever.

    Lives beside the writers rather than in the wizard so a reader does not have
    to import it: the wizard module costs ~290ms to load, which `ddeharness doctor`
    (a millisecond command) would otherwise pay just to answer this.
    """
    return role_configured_in(load_memory_config(), section)


def set_memory_section(section: str, fields: dict[str, Any]) -> None:
    """Merge ``fields`` into ``[section]`` of the user-level toml.

    ``None`` values are dropped (treated as "leave unset"); existing keys in
    the section and every other section are preserved.
    """
    if section not in WRITABLE_SECTIONS:
        raise KeyError(f"unknown memory section {section!r}; writable: {WRITABLE_SECTIONS}")
    _require_owned(f"write [{section}]")
    data = load_memory_config()
    clean = {k: v for k, v in fields.items() if v is not None}
    data[section] = {**data.get(section, {}), **clean}
    _write_atomic(get_memory_config_path(), data)


def clear_memory_section(section: str) -> None:
    """Drop ``[section]`` from the user-level toml (no-op if absent)."""
    if section not in WRITABLE_SECTIONS:
        raise KeyError(f"unknown memory section {section!r}; writable: {WRITABLE_SECTIONS}")
    _require_owned(f"clear [{section}]")
    data = load_memory_config()
    if section not in data:
        return
    del data[section]
    _write_atomic(get_memory_config_path(), data)


def memory_declared_address() -> str | None:
    """The address the root's config toml declares, or ``None`` when unset.

    This is the authority on where a server for that root listens: the server reads
    ``[api]`` at startup, and opendde no longer overrides it on the command line.
    Everything else -- opendde's ``base_url``, a doctor probe -- is a copy of it.
    """
    api = memory_section("api")
    host = api.get("host")
    port = api.get("port")
    if not host or not port:
        return None
    return f"http://{host}:{port}"


def set_memory_api(*, host: str, port: int) -> None:
    """Record the address a server for this root must listen on.

    opendde used to pass ``--port`` on the command line, which overrode the toml
    and left the file describing an address nobody was using. Writing it instead
    makes the root self-describing: anything that can read the directory knows
    where its server lives, with no second place to drift out of sync.
    """
    set_memory_section("api", {"host": host, "port": int(port)})
