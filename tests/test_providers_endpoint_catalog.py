"""An OpenAI-compatible endpoint's own model list, cached per endpoint."""

import httpx

from opendde_harness.providers import endpoint_catalog


def test_ids_are_read_from_the_openai_shape_and_a_bare_list():
    assert endpoint_catalog.listed_ids({"data": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}) == ("a", "b")
    assert endpoint_catalog.listed_ids(["x", {"id": "y"}, 3]) == ("x", "y")
    assert endpoint_catalog.listed_ids({"object": "list"}) == ()


def test_the_list_is_fetched_once_and_a_failure_is_not_retried_at_once(monkeypatch):
    endpoint_catalog.reset_cache()
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if "down" in url:
            raise httpx.ConnectError("no route")
        return httpx.Response(200, json={"data": [{"id": "m1"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    assert endpoint_catalog.endpoint_models("https://relay/v1/", "k") == ("m1",)
    assert endpoint_catalog.endpoint_models("https://relay/v1", "k") == ("m1",)
    assert calls == ["https://relay/v1/models"]

    assert endpoint_catalog.endpoint_models("https://down/v1", "k") == ()
    assert endpoint_catalog.endpoint_models("https://down/v1", "k") == ()
    assert calls.count("https://down/v1/models") == 1
