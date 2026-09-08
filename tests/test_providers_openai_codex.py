import asyncio

import pytest

from opendde_harness.providers.openai_codex_provider import OpenAICodexProvider


def _provider():
    return OpenAICodexProvider(default_model="openai-codex/gpt-5.6-luna")


def test_certificate_failure_never_repeats_the_request_unverified(monkeypatch):
    """The request carries the account's OAuth token; an unverified retry leaks it."""
    calls = []

    async def token():
        return "sk-oauth-token", "account-1"

    async def request(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(
        "opendde_harness.providers.chatgpt_token.access_token_and_account",
        lambda: ("sk-oauth-token", "account-1"),
    )
    monkeypatch.setattr("opendde_harness.providers.openai_codex_provider._request_codex", request)

    response = asyncio.run(_provider().chat(messages=[{"role": "user", "content": "hi"}]))

    assert len(calls) == 1
    assert response.finish_reason == "error"
    assert "certificate" in response.content.lower()


@pytest.mark.parametrize("signature", ["verify=False", "verify = False"])
def test_provider_source_has_no_verification_switch(signature):
    from pathlib import Path

    source = Path("opendde_harness/providers/openai_codex_provider.py").read_text()

    assert signature not in source
