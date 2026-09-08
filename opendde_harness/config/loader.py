"""Configuration loading utilities."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from opendde_harness.config.schema import FEATURE_FIELDS, Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None

# Parsed configs keyed by path, with the file's (mtime_ns, size, inode) at
# parse time. The TUI RPC server loads the config on every request;
# re-validating an unchanged file each time is wasted work, and a changed file
# is caught by the stat check -- the inode because every writer replaces the
# file atomically, and the kernel's timestamp granularity would otherwise let a
# same-size rewrite within one tick read as unchanged. Entries are handed out
# as deep copies so a caller that edits its copy (``load_runtime_config``
# overrides the workspace) cannot leak the edit.
_cache: dict[str, tuple[tuple[int, int, int], Config]] = {}

# Paths already warned about as malformed in this process; repeated
# load_config calls (status/doctor load more than once) warn only once.
_warned_paths: set[str] = set()


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".opendde_harness" / "config.json"


class ConfigReadError(Exception):
    """An existing config file could not be parsed. Callers doing a
    read-modify-write MUST NOT proceed: overwriting would replace the user's
    whole config with just their section (data loss). Only a genuinely-absent
    file is safe to create fresh.

    Deliberately NOT a RuntimeError: the CLI write commands wrap their ops in a
    broad ``except RuntimeError`` (for provider OAuth-refusal etc.), and we want
    a parse error to bypass those and reach the single ``run()`` handler (or a
    caller's explicit ``except ConfigReadError``), not be swept up implicitly."""


class ConfigSchemaError(ValueError):
    """A config file parsed as JSON but does not match the schema.

    Every model is ``extra='forbid'``, so a key this release does not know --
    a typo, or one a retired release wrote -- lands here rather than being
    silently ignored. Nothing rewrites an old config in place: the remedy is
    ``ddeharness onboard``, which writes a fresh one.
    """


def _unknown_keys(exc: ValidationError) -> list[str]:
    return [".".join(str(part) for part in err["loc"]) for err in exc.errors() if err["type"] == "extra_forbidden"]


def _schema_error(path: Path, exc: ValidationError) -> ConfigSchemaError:
    unknown = _unknown_keys(exc)
    lines = [f"Config at {path} fails schema validation:"]
    if unknown:
        lines.append("unknown key(s): " + ", ".join(unknown))
        lines.append(
            "Remove them, or run `ddeharness onboard` to write a fresh config. "
            "Keys from earlier releases are not migrated."
        )
    lines.append(str(exc))
    return ConfigSchemaError("\n".join(lines))


def read_raw_or_raise(path: Path) -> dict[str, Any]:
    """Read a config file as raw JSON for a read-modify-write cycle.

    Returns ``{}`` ONLY when the file is absent. A present-but-unreadable file
    raises :class:`ConfigReadError` rather than returning ``{}`` -- returning
    ``{}`` and then writing was the bug that wiped a real config over a lone
    JSON syntax error (e.g. a // comment). The single read path for every
    ``update_*`` write module.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}  # empty file: no data to lose, safe to create fresh (like absent)
        data = json.loads(text)
        # A valid-JSON non-object (null / list / scalar) is not a usable config;
        # return {} so callers get a mapping (not None) without an AttributeError.
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ConfigReadError(
            f"{path} is not valid JSON ({exc}). Fix it first (JSON allows no comments or "
            "trailing commas); your config was left unchanged."
        ) from exc


def _file_stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    cached = _cache.get(str(path))
    if cached is not None and cached[0] == _file_stamp(path):
        return cached[1].model_copy(deep=True)

    config: Config | None = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            # Boot on defaults for a malformed file (a transient mid-write race
            # shouldn't brick callers) but warn LOUDLY -- a persistent syntax
            # error would else revert every setting with no visible cause.
            # Raising instead needs atomic save_config first (separate change).
            msg = (
                f"config at {path} is not valid JSON ({e}) -- IGNORING it and running on "
                "DEFAULTS. Fix the file (JSON allows no comments or trailing commas) and restart."
            )
            # Single user-visible channel: the stderr print (visible under any
            # loguru sink config). The log-file trace uses stdlib logging, NOT
            # loguru — loguru's default sink echoes DEBUG to stderr, which
            # would re-duplicate the warning on plain CLI runs; the stdlib
            # record reaches the file sink via the CLI's logging intercept.
            if str(path) not in _warned_paths:
                _warned_paths.add(str(path))
                print(f"WARNING: {msg}", file=sys.stderr)
            logging.getLogger(__name__).debug(msg)
        else:
            # A clean parse re-arms the warning: the dedup exists to silence
            # repeated loads of the same broken state within one command, not
            # to spend the one warning a long-lived process (the TUI RPC
            # server reloads every turn) gets for a later re-breakage.
            _warned_paths.discard(str(path))
            try:
                config = Config.model_validate(data)
            except ValidationError as e:
                # Schema mismatch is a user/programmer error — surface
                # loudly rather than masking with defaults. Silently
                # using defaults makes "feature X did nothing" debug
                # take 24h instead of 24s.
                raise _schema_error(path, e) from e
            stamp = _file_stamp(path)
            if stamp is not None:
                _cache[str(path)] = (stamp, config.model_copy(deep=True))

    if config is None:
        config = Config()

    return config


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save the base blocks of ``config`` to file.

    The feature blocks (``FEATURE_FIELDS``) are not written: the onboarding
    wizard seeds the user-facing subset of them with
    ``config.update.init_extension_block_defaults`` and every later change
    is patched in place by the ``update_*`` modules.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True, exclude=set(FEATURE_FIELDS))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
