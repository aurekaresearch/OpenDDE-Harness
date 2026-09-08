"""How a failed call is bucketed, since the bucket decides retry and fallback."""

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
    ],
)
def test_classification(message, category, retryable):
    verdict = LLMProvider.classify_error(content=message)

    assert (verdict.category, verdict.retryable) == (category, retryable)
