"""A drop after the first delta is the agent loop's retry; the rotor's part is
to remember which endpoint dropped so the re-run lands on another."""

import pytest

from opendde_harness.providers.base import ErrorClassification, LLMProvider, LLMResponse, StreamDelta
from opendde_harness.providers.endpoint_rotor import EndpointRotorProvider
from opendde_harness.providers.endpoints import ResolvedEndpoint


class _Inner(LLMProvider):
    def __init__(self, label: str, fail_after_first: bool):
        super().__init__()
        self.label = label
        self.fail_after_first = fail_after_first

    async def chat(self, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.label, finish_reason="stop")

    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        yield StreamDelta(content=self.label)
        if self.fail_after_first:
            raise RuntimeError("connection reset by peer")
        yield StreamDelta(content=None, finish_reason="stop")

    def classify_error(self, exc=None, content=None):
        return ErrorClassification("network", retryable=True, should_fallback=True)

    def get_default_model(self):
        return "fake/model"


def _rotor(*inners):
    endpoints = [ResolvedEndpoint(label=i.label, api_key="k", api_base=None, extra_headers=None) for i in inners]
    lookup = {i.label: i for i in inners}
    return EndpointRotorProvider(endpoints, lambda ep: lookup[ep.label], default_model="fake/model")


async def _drain(rotor):
    out = []
    async for delta in rotor.chat_stream([{"role": "user", "content": "hi"}]):
        out.append(delta.content)
    return out


async def test_a_mid_stream_drop_is_raised_and_the_endpoint_cooled():
    rotor = _rotor(_Inner("a", fail_after_first=True), _Inner("b", fail_after_first=False))

    with pytest.raises(RuntimeError):
        await _drain(rotor)

    assert rotor._state.failure_count[0] == 1
    assert await _drain(rotor) == ["b", None]


async def test_a_clean_stream_leaves_the_endpoint_healthy():
    rotor = _rotor(_Inner("a", fail_after_first=False))

    assert await _drain(rotor) == ["a", None]
    assert rotor._state.failure_count.get(0, 0) == 0


class _ErrorDeltaInner(_Inner):
    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        yield StreamDelta(content=self.label)
        yield StreamDelta(
            content=None,
            finish_reason="error",
            error_classification=ErrorClassification("network", retryable=True, should_fallback=True),
        )


async def test_a_terminal_error_delta_after_output_cools_the_endpoint_too():
    rotor = _rotor(_ErrorDeltaInner("a", fail_after_first=False), _Inner("b", fail_after_first=False))

    assert await _drain(rotor) == ["a", None]
    assert rotor._state.failure_count[0] == 1
    assert await _drain(rotor) == ["b", None]
