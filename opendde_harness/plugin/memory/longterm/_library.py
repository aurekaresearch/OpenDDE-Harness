"""The one place OpenDDE Harness touches the long-term memory library by name.

Every ``import`` of the library and every literal it dictates at runtime lives
here, so the rest of the tree speaks only of "long-term memory". The literals
below are not naming choices: the library resolves its root from the
``EVEROS_ROOT`` variable, reads every setting through the ``EVEROS_`` env
prefix, refuses to start without ``<root>/everos.toml``, ships its CLI as the
``everos`` executable. Renaming any of them here would break the library, so they are exported as constants and the
callers never spell them.

The library imports themselves are deferred into functions: the plugin package
must stay import-cheap (discovery touches it), and the multimodal extra is
optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT_ENV_VAR = "EVEROS_ROOT"
ENV_PREFIX = "EVEROS_"
API_ENV_PREFIX = "EVEROS_API__"
MULTIMODAL_ENV_PREFIX = "EVEROS_MULTIMODAL__"
CONFIG_FILENAME = "everos.toml"
OME_CONFIG_FILENAME = "ome.toml"
EXECUTABLE = "everos"
SERVER_CMDLINE = "everos server start"
MULTIMODAL_EXTRA = "everos[multimodal]"


def config_templates() -> tuple[Path, Path] | None:
    """The shipped ``(config toml, ome toml)`` templates, or ``None`` when the
    library is not importable."""
    try:
        from everos.entrypoints.cli.commands.init_cmd import _EVEROS_TEMPLATE, _OME_TEMPLATE
    except ImportError:
        return None
    return Path(_EVEROS_TEMPLATE), Path(_OME_TEMPLATE)


def multimodal_parser_installed() -> None:
    """Raise when the optional multimodal parser extra is absent or unusable."""
    from everos.memory.extract.parser import require_multimodal

    require_multimodal()


def multimodal_api() -> dict[str, Any]:
    """The multimodal parser surface, keyed by role.

    ``ImportError`` propagates so the caller can report the missing extra.
    """
    from everos.component.llm.client import LLMNotConfiguredError, get_multimodal_llm_client
    from everos.core.errors import MultimodalNotEnabledError, UnsupportedModalityError
    from everos.memory.extract.parser import enrich_content_items, require_multimodal

    return {
        "require_multimodal": require_multimodal,
        "get_client": get_multimodal_llm_client,
        "enrich_content_items": enrich_content_items,
        "not_configured_errors": (MultimodalNotEnabledError, LLMNotConfiguredError),
        "unsupported_error": UnsupportedModalityError,
    }


def patch_llm_client(factory: Any) -> None:
    """Replace the library's LLM client construction seam with ``factory``.

    The library's client module imported its builder into its own namespace, so
    the patch has to land on that exact attribute before the server app is built.
    """
    from everos.component.llm import client as client_module

    client_module.build_client = factory


def start_server(root: Path) -> None:
    """Run the library's HTTP server in this process, serving ``root``.

    ``start`` is declared as a Typer command. Calling it like an ordinary
    function without every argument leaves Typer ``OptionInfo`` objects in the
    omitted slots (not their declared defaults), which later fails at
    ``log_level.upper()``. Supply the concrete CLI defaults explicitly.
    """
    from everos.entrypoints.cli.commands.server import start

    start(host=None, port=None, root=str(root), reload=False, log_level=None)


__all__ = [
    "API_ENV_PREFIX",
    "CONFIG_FILENAME",
    "ENV_PREFIX",
    "EXECUTABLE",
    "MULTIMODAL_ENV_PREFIX",
    "MULTIMODAL_EXTRA",
    "OME_CONFIG_FILENAME",
    "ROOT_ENV_VAR",
    "SERVER_CMDLINE",
    "config_templates",
    "multimodal_api",
    "multimodal_parser_installed",
    "patch_llm_client",
    "start_server",
]
