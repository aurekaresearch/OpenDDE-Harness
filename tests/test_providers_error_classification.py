"""How a failed call is bucketed, since the bucket decides retry and fallback."""

import asyncio
import ssl

import httpx
import pytest

from opendde_harness.providers.base import LLMProvider


@pytest.mark.parametrize(
    ("message", "category", "retryable"),
    [
        # A status code the message actually reports.
        ("Error code: 503 - insufficient_system_resource", "server", True),
        ("HTTP 502 Bad Gateway", "server", True),
        # Digits that merely contain one. Both used to buy four billed attempts
        # and then every fallback model for a request that cannot succeed.
        ("Invalid value for 'max_tokens': 5000", "unknown", False),
        ("request id: 20260908213316503720727", "unknown", False),
        # DeepSeek's capacity wording, which the billing bucket claimed through
        # its "insufficient" match and made non-retryable.
        ("DeepseekException - insufficient_system_resource", "server", True),
        ("Insufficient Balance", "billing", False),
        # Measured against a new-api relay: an unknown model answers this, not a 404.
        (
            "APIError: OpenAIException - No available channel for group default model x (distributor).",
            "model_unavailable",
            False,
        ),
    ],
)
def test_classification(message, category, retryable):
    verdict = LLMProvider.classify_error(content=message)

    assert (verdict.category, verdict.retryable) == (category, retryable)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadError(""),
        httpx.WriteError(""),
        httpx.ConnectTimeout(""),
        httpx.ReadTimeout(""),
        httpx.PoolTimeout(""),
        httpx.ProxyError(""),
        httpx.ConnectError("[Errno 111] Connection refused"),
        ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
        ssl.SSLError(1, "EOF occurred in violation of protocol"),
        OSError("Temporary failure in name resolution"),
        ConnectionResetError(104, "Connection reset by peer"),
        asyncio.TimeoutError(),
    ],
)
def test_real_transport_failures_are_retryable_network_errors(exc):
    # Measured against these exception objects: nine of the fourteen used to
    # classify as unknown, which is fatal -- no retry, no fallback -- on the
    # httpx providers that LiteLLM's exception mapping does not cover.
    verdict = LLMProvider.classify_error(exc)

    assert verdict.category == "network"
    assert verdict.retryable and verdict.should_fallback


def test_a_status_error_is_not_mistaken_for_a_transport_failure():
    request = httpx.Request("POST", "https://relay/v1/chat/completions")
    exc = httpx.HTTPStatusError("Client error", request=request, response=httpx.Response(400, request=request))

    assert LLMProvider.classify_error(exc).category != "network"
