"""Streaming HTTP downloads with resume, incremental hashing and one shared progress display."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

HF_ORIGIN = "https://huggingface.co"
DEFAULT_HF_MIRROR = "https://hf-mirror.com"
CHUNK = 1024 * 1024
TIMEOUT = httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=30.0)


class DownloadError(ValueError):
    pass


def progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        DownloadColumn(),
        TransferSpeedColumn(),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not console.is_terminal,
        transient=True,
    )


def new_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": "opendde-harness"})


def with_mirrors(sources: Sequence[str]) -> list[str]:
    """Add the Hugging Face mirror (HF_ENDPOINT, else hf-mirror.com) after every huggingface.co URL."""
    mirror = (os.environ.get("HF_ENDPOINT") or DEFAULT_HF_MIRROR).rstrip("/")
    result: list[str] = []
    for url in sources:
        candidates = [url, mirror + url[len(HF_ORIGIN) :]] if url.startswith(HF_ORIGIN + "/") else [url]
        result.extend(candidate for candidate in candidates if candidate not in result)
    return result


def _file_digest(path: Path) -> tuple[hashlib._Hash, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest, size


def download_file(
    sources: Sequence[str],
    target: Path,
    *,
    sha256: str | None = None,
    size: int | None = None,
    description: str | None = None,
    console: Console | None = None,
    max_bytes: int | None = None,
    deadline_s: float = 7200,
) -> Path:
    """Stream the first working source into ``target``, resuming a partial file and verifying it.

    ``target`` may already hold a partial transfer; it is continued with a Range
    request (or restarted when the server ignores the range). The digest is
    computed while streaming, so a verified file is never read back. A source
    that fails (connection, HTTP status, limit, or SHA256/size mismatch, which
    also removes the file) is reported and the next one is tried.
    """
    console = console or Console()
    name = description or target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.is_file() else 0
    if existing and size is not None and existing == size:
        digest, _ = _file_digest(target)
        if sha256 is None or digest.hexdigest() == sha256:
            return target
        existing = 0
    failures = []
    with new_client() as client, progress(console) as bar:
        for index, url in enumerate(sources):
            print(f"Source: {url}", flush=True)
            try:
                _fetch(client, bar, url, target, name, existing, sha256, size, max_bytes, deadline_s)
                return target
            except (httpx.HTTPError, httpx.InvalidURL, DownloadError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    reason = f"HTTP {exc.response.status_code}"
                elif isinstance(exc, DownloadError):
                    reason = str(exc)
                else:
                    reason = type(exc).__name__
                failures.append(f"{url}: {reason}")
                remaining = len(sources) - index - 1
                print(f"Source failed ({reason}): {url}" + ("; trying the next source" if remaining else ""), flush=True)
                existing = target.stat().st_size if target.is_file() else 0
    raise DownloadError(f"Every download source failed for {name}: " + "; ".join(failures))


def _fetch(
    client: httpx.Client,
    bar: Progress,
    url: str,
    target: Path,
    name: str,
    existing: int,
    sha256: str | None,
    size: int | None,
    max_bytes: int | None,
    deadline_s: float,
) -> None:
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 416 and existing:
            # The partial file is already complete; only the digest is left to check.
            resumed, mode = True, None
        else:
            response.raise_for_status()
            resumed = response.status_code == 206 and existing > 0
            mode = "ab" if resumed else "wb"
        digest = hashlib.sha256()
        received = 0
        if resumed:
            digest, received = _file_digest(target)
        length = response.headers.get("Content-Length")
        total = (received + int(length)) if length and length.isdigit() else size
        task = bar.add_task(name, total=total, completed=received)
        deadline = time.monotonic() + deadline_s
        if mode is not None:
            with target.open(mode) as handle:
                for chunk in response.iter_bytes(CHUNK):
                    received += len(chunk)
                    if (max_bytes is not None and received > max_bytes) or time.monotonic() > deadline:
                        raise DownloadError("download exceeded its size or time limit")
                    handle.write(chunk)
                    digest.update(chunk)
                    bar.update(task, advance=len(chunk))
        bar.remove_task(task)
    if (size is not None and received != size) or (sha256 is not None and digest.hexdigest() != sha256):
        target.unlink(missing_ok=True)
        raise DownloadError("SHA256/size mismatch; the file was removed")
