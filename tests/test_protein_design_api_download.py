"""Completed API predictions survive transient result-transfer failures."""

import io
import zipfile

import httpx
import pytest

from opendde_harness.plugin.protein_design.servers.backends.opendde_api import OpenDDEAPIError, OpenDDEJobClient


def archive_bytes():
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as bundle:
        bundle.writestr("result.cif", "data_structure\n")
    return content.getvalue()


class InterruptedStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"incomplete archive"
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


@pytest.mark.parametrize("failure", ["disconnect", "timeout", "corrupt", 408, 429, 500, 502, 503, 504])
def test_download_retries_same_job_and_publishes_only_valid_zip(tmp_path, monkeypatch, failure):
    requests = []
    delays = []
    payload = archive_bytes()
    destination = tmp_path / "result.zip"
    destination.write_bytes(b"previous result")
    monkeypatch.setattr("opendde_harness.plugin.protein_design.servers.backends.opendde_api.time.sleep", delays.append)

    def handle(request):
        requests.append((request.method, request.url.path))
        assert destination.read_bytes() == b"previous result"
        if len(requests) == 1:
            if failure == "disconnect":
                return httpx.Response(200, stream=InterruptedStream())
            if failure == "timeout":
                raise httpx.ReadTimeout("slow stream", request=request)
            if failure == "corrupt":
                return httpx.Response(200, content=b"invalid zip")
            return httpx.Response(failure, content=b"temporary error")
        assert not destination.with_suffix(".zip.part").exists()
        return httpx.Response(200, content=payload)

    with OpenDDEJobClient("https://example.invalid", transport=httpx.MockTransport(handle)) as client:
        assert client.download("completed-job", destination) == destination
    assert requests == [("GET", "/api/v1/folding/opendde/jobs/completed-job/download")] * 2
    assert delays == [3.0]
    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".zip.part").exists()


@pytest.mark.parametrize("status, attempts", [(401, 1), (403, 1), (404, 1), (503, 6)])
def test_download_failure_is_bounded_and_preserves_existing_file(tmp_path, monkeypatch, status, attempts):
    calls = []
    monkeypatch.setattr("opendde_harness.plugin.protein_design.servers.backends.opendde_api.time.sleep", lambda _: None)
    destination = tmp_path / "result.zip"
    destination.write_bytes(b"previous result")

    def handle(request):
        calls.append(request)
        return httpx.Response(status, content=b"failure")

    with OpenDDEJobClient("https://example.invalid", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenDDEAPIError, match=f"completed-job result download failed after {attempts} attempts"):
            client.download("completed-job", destination)
    assert len(calls) == attempts
    assert destination.read_bytes() == b"previous result"
    assert not destination.with_suffix(".zip.part").exists()
