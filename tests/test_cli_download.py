import hashlib

import httpx
import pytest
from rich.console import Console

from opendde_harness.cli import _download
from opendde_harness.cli._download import DownloadError, download_file, with_mirrors

DATA = b"0123456789" * 1000
SHA = hashlib.sha256(DATA).hexdigest()


def _serve(monkeypatch, handler):
    monkeypatch.setattr(_download, "new_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def test_with_mirrors_adds_hf_mirror_after_each_origin_url(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    urls = ["https://huggingface.co/a/b/resolve/x/f.pt", "https://example.invalid/f.pt"]
    assert with_mirrors(urls) == [
        "https://huggingface.co/a/b/resolve/x/f.pt",
        "https://hf-mirror.com/a/b/resolve/x/f.pt",
        "https://example.invalid/f.pt",
    ]
    monkeypatch.setenv("HF_ENDPOINT", "https://hf.internal/")
    assert with_mirrors(urls)[1] == "https://hf.internal/a/b/resolve/x/f.pt"


def test_download_resumes_a_partial_file_with_a_range_request(tmp_path, monkeypatch, capsys):
    target = tmp_path / "f.part"
    target.write_bytes(DATA[:4000])
    seen = []

    def handler(request):
        seen.append(request.headers.get("Range"))
        return httpx.Response(206, content=DATA[4000:], headers={"Content-Length": str(len(DATA) - 4000)})

    _serve(monkeypatch, handler)
    with (tmp_path / "log").open("w") as log:
        console = Console(file=log, force_terminal=False)
        assert (
            download_file(["https://example.invalid/f"], target, sha256=SHA, size=len(DATA), console=console) == target
        )
    assert target.read_bytes() == DATA and seen == ["bytes=4000-"]
    assert capsys.readouterr().out.strip() == "Source: https://example.invalid/f"
    assert (tmp_path / "log").read_text() == ""


def test_download_restarts_when_the_server_ignores_the_range(tmp_path, monkeypatch):
    target = tmp_path / "f.part"
    target.write_bytes(b"stale-prefix")
    _serve(monkeypatch, lambda request: httpx.Response(200, content=DATA))
    download_file(["https://example.invalid/f"], target, sha256=SHA)
    assert target.read_bytes() == DATA


def test_download_skips_the_network_for_a_complete_verified_partial(tmp_path, monkeypatch):
    target = tmp_path / "f.part"
    target.write_bytes(DATA)
    _serve(monkeypatch, lambda request: pytest.fail("no request expected"))
    download_file(["https://example.invalid/f"], target, sha256=SHA, size=len(DATA))


def test_download_falls_back_to_the_mirror_on_connection_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    attempts = []

    def handler(request):
        attempts.append(str(request.url))
        if request.url.host == "huggingface.co":
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(200, content=DATA)

    _serve(monkeypatch, handler)
    sources = with_mirrors(["https://huggingface.co/org/repo/resolve/rev/f.pt"])
    download_file(sources, tmp_path / "f.part", sha256=SHA)
    assert attempts == sources
    assert (
        "Source failed (ConnectError): https://huggingface.co/org/repo/resolve/rev/f.pt; trying the next source"
        in capsys.readouterr().out
    )


def test_download_rejects_a_digest_mismatch_and_removes_the_file(tmp_path, monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, content=b"wrong"))
    target = tmp_path / "f.part"
    with pytest.raises(DownloadError, match="Every download source failed"):
        download_file(["https://example.invalid/f"], target, sha256=SHA)
    assert not target.exists()


def test_download_reports_http_errors_and_size_limits(tmp_path, monkeypatch, capsys):
    _serve(monkeypatch, lambda request: httpx.Response(404))
    with pytest.raises(DownloadError):
        download_file(["https://example.invalid/missing"], tmp_path / "m.part")
    assert "Source failed (HTTP 404): https://example.invalid/missing" in capsys.readouterr().out.splitlines()
    _serve(monkeypatch, lambda request: httpx.Response(200, content=DATA))
    with pytest.raises(DownloadError, match="size or time limit"):
        download_file(["https://example.invalid/big"], tmp_path / "b.part", max_bytes=100)
