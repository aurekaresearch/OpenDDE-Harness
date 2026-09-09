"""Import litellm with its terminal noise silenced, at import and afterwards.

litellm prints a "Provider List" banner (gated by ``suppress_debug_info``) and,
because it installs its own stderr ``StreamHandler`` on its ``LiteLLM*`` loggers,
emits DEBUG to the terminal *while importing*. Raise those loggers' levels across
the import so that DEBUG never reaches the terminal, then restore them.

Then detach that handler for good. opendde's ``_strip_tty_stream_handlers`` cannot
do it: it runs while the CLI sets up logging, and every litellm import in opendde is
deferred, so the handler is installed *after* the strip has already run and
nothing removes it. What follows is a session where every litellm record --
including DEBUG, because opendde's stdlib intercept sets the root level to 0 -- is
written straight to the terminal, over the Ink screen. Detaching leaves the
records propagating to root, so they still reach the log file sink.
"""

import logging
import os
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

# litellm attaches its stderr handler to all three (litellm/_logging.py).
_LITELLM_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
_OPENDDE_HARNESS_RESPONSES_USAGE_PATCH = "_opendde_harness_preserves_responses_api_usage"


def _preserve_responses_api_usage(
    original: Callable[..., Any],
    response_api_usage_type: type[Any],
) -> Callable[..., Any]:
    """Undo LiteLLM's invalid Responses streaming usage mutation.

    ``Logging._get_assembled_streaming_response`` (still, on 1.100.0) converts
    a typed ``ResponseAPIUsage`` to Chat-Completions usage and assigns the
    dumped dictionary back to ``ResponsesAPIResponse.usage``. The field is
    declared as ``ResponseAPIUsage``; background logging later calls
    ``model_dump()`` and Pydantic reports the incompatible value on every
    streamed Responses call.

    Keep the original Responses usage object on the assembled response.
    LiteLLM's cost and standard-logging helpers accept ``ResponseAPIUsage``
    directly, so token accounting remains intact. Re-check on each bump by
    running a streamed Responses call with this patch off; delete it once
    the warning is gone.
    """

    @wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = kwargs.get("result", args[0] if args else None)
        source_response = getattr(result, "response", None)
        source_usage = getattr(source_response, "usage", None)

        assembled = original(self, *args, **kwargs)
        if (
            isinstance(source_usage, response_api_usage_type)
            and assembled is source_response
            and not isinstance(
                getattr(assembled, "usage", None),
                response_api_usage_type,
            )
        ):
            assembled.usage = source_usage
        return assembled

    setattr(wrapped, _OPENDDE_HARNESS_RESPONSES_USAGE_PATCH, True)
    return wrapped


def _patch_litellm_responses_usage_mutation() -> None:
    """Install the Responses usage fix once."""
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.llms.openai import ResponseAPIUsage

    current = Logging._get_assembled_streaming_response
    if getattr(current, _OPENDDE_HARNESS_RESPONSES_USAGE_PATCH, False):
        return
    Logging._get_assembled_streaming_response = _preserve_responses_api_usage(
        current,
        ResponseAPIUsage,
    )


def _detach_tty_handlers(loggers: list[logging.Logger]) -> None:
    """Remove the terminal ``StreamHandler``s litellm put on its own loggers."""
    tty_streams = (sys.stderr, sys.stdout)
    for lg in loggers:
        for handler in list(lg.handlers):
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in tty_streams:
                lg.removeHandler(handler)


def _point_oauth_tokens_at_opendde_harness() -> None:
    """Send the credentials LiteLLM's drivers own to opendde's OAuth directory.

    Both authenticators read their variable in ``__init__`` and create the
    directory, so these have to be set before litellm is imported at all. An
    explicit setting by the user wins.
    """
    from opendde_harness.config.paths import get_oauth_dir

    oauth_dir = get_oauth_dir()
    os.environ.setdefault("GITHUB_COPILOT_TOKEN_DIR", str(oauth_dir / "github_copilot"))
    os.environ.setdefault("CHATGPT_TOKEN_DIR", str(oauth_dir / "chatgpt"))


#: What the first request imports lazily on top of ``import litellm``: the
#: OpenAI SDK's resource and type trees and a handful of LiteLLM internals,
#: 261 modules measured on a cold process, about three seconds of file reads
#: -- on the event loop, at the moment the user sends their first message.
#: Recorded from a real first call rather than guessed; a module that no
#: longer exists is skipped, so a LiteLLM bump cannot break the warm-up.
_FIRST_REQUEST_MODULES = (
    "openai.lib.streaming",
    "openai.resources",
    "openai.resources.audio",
    "openai.resources.batches",
    "openai.resources.beta",
    "openai.resources.chat",
    "openai.resources.completions",
    "openai.resources.containers",
    "openai.resources.embeddings",
    "openai.resources.evals",
    "openai.resources.files",
    "openai.resources.fine_tuning",
    "openai.resources.images",
    "openai.resources.models",
    "openai.resources.moderations",
    "openai.resources.skills",
    "openai.resources.uploads",
    "openai.resources.vector_stores",
    "openai.resources.videos",
    "openai.types.beta",
    "openai.types.chat",
    "openai.types.containers",
    "openai.types.evals",
    "openai.types.fine_tuning",
    "openai.types.skills",
    "openai.types.uploads",
    "openai.types.vector_stores",
    "litellm._service_logger",
    "litellm.integrations.prometheus_services",
    "litellm.litellm_core_utils.get_supported_openai_params",
    "litellm.litellm_core_utils.llm_request_utils",
    "litellm.llms.litellm_proxy",
    "litellm.llms.litellm_proxy.chat",
    "litellm.llms.litellm_proxy.chat.transformation",
    "litellm.llms.openai_like.dynamic_config",
    "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic",
    "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.output_params_utils",
    "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation",
    "litellm.proxy._experimental",
    "litellm.proxy._experimental.mcp_server",
    "litellm.proxy._experimental.mcp_server.utils",
    "litellm.proxy.openai_files_endpoints",
    "litellm.proxy.openai_files_endpoints.common_utils",
    "litellm.responses.mcp",
    "litellm.responses.mcp.chat_completions_handler",
    "litellm.responses.mcp.litellm_proxy_mcp_handler",
    "litellm.types.integrations.prometheus",
    "litellm.types.litellm_core_utils",
    "litellm.types.litellm_core_utils.streaming_chunk_builder_utils",
)


def warm_first_request_imports() -> None:
    """Import what the first request would, so it does not happen on the loop.

    Called from the lazy provider's prewarm thread after the provider is
    built. Idempotent: an already-imported module costs a dict lookup.
    """
    import importlib

    for name in _FIRST_REQUEST_MODULES:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            continue


def import_litellm():
    """Import litellm with its banner disabled and its terminal handler detached."""
    _point_oauth_tokens_at_opendde_harness()
    # LiteLLM otherwise performs a blocking GitHub Raw request while importing
    # its model-price catalogue.  OpenDDE Harness pins LiteLLM in ``uv.lock``,
    # so its bundled backup is the matching, deterministic default.  Users who
    # explicitly want the live catalogue can still set this variable to False.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    loggers = [logging.getLogger(name) for name in _LITELLM_LOGGERS]
    prev_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        import litellm

        litellm.suppress_debug_info = True
    finally:
        for lg, prev in zip(loggers, prev_levels):
            lg.setLevel(prev)

    _detach_tty_handlers(loggers)
    _patch_litellm_responses_usage_mutation()

    return litellm
