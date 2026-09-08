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

    LiteLLM 1.85.0's ``Logging._get_assembled_streaming_response`` converts a
    typed ``ResponseAPIUsage`` to Chat-Completions usage and assigns the dumped
    dictionary back to ``ResponsesAPIResponse.usage``. The field is declared
    as ``ResponseAPIUsage``; background logging later calls ``model_dump()``
    and Pydantic correctly reports the incompatible value.

    Keep the original Responses usage object on the assembled response. The
    current LiteLLM cost and standard-logging helpers already accept
    ``ResponseAPIUsage`` directly, so token accounting remains intact.
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
    """Install the narrow LiteLLM 1.x Responses compatibility fix once."""
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
