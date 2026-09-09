"""Client for the asynchronous OpenDDE folding job API."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import httpx


class OpenDDEAPIError(RuntimeError):
    """Raised when the remote OpenDDE service rejects or fails a job."""


def normalize_opendde_api_url(value: str) -> str:
    """Return an origin URL suitable for the documented OpenDDE API paths."""
    normalized = str(value).strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"OpenDDE API URL must be an HTTP(S) origin without an API path, query, or fragment: {value!r}"
        )
    return normalized


class OpenDDEJobClient:
    JOBS_PATH = "/api/v1/folding/opendde/jobs"

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float = 60.0,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized = normalize_opendde_api_url(base_url)
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._client = httpx.Client(
            base_url=normalized,
            timeout=request_timeout_seconds,
            headers=headers,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenDDEJobClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def submit(
        self,
        instances: Sequence[Mapping[str, Any]],
        *,
        n_samples: int,
        n_step: int,
        need_atom_confidence: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            self.JOBS_PATH,
            json={
                "instances": [dict(item) for item in instances],
                "parameters": {
                    "n_samples": int(n_samples),
                    "n_step": int(n_step),
                    "need_atom_confidence": bool(need_atom_confidence),
                    "dry_run": bool(dry_run),
                },
            },
            expected_statuses={200, 202},
        )

    def probe(self) -> None:
        """Verify that the configured service exposes the OpenDDE API."""
        try:
            response = self._client.get("/api/v1/folding/opendde/openapi.json", timeout=5.0)
        except httpx.HTTPError as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise OpenDDEAPIError(f"OpenDDE API probe failed: {detail}") from exc
        self._raise_for_status(response, {200})

        try:
            schema = response.json()
        except ValueError as exc:
            raise OpenDDEAPIError("OpenDDE API probe returned invalid JSON") from exc
        paths = schema.get("paths") if isinstance(schema, Mapping) else None
        if (
            not isinstance(paths, Mapping)
            or not schema.get("openapi")
            or not any(
                isinstance(paths.get(path), Mapping) and "post" in paths[path] for path in ("/jobs", self.JOBS_PATH)
            )
        ):
            raise OpenDDEAPIError("The endpoint does not expose the OpenDDE job API")

    def wait(
        self,
        run_id: str,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float,
        stalled_poll_limit: int = 3,
        status_log_path: Path | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        stalled_polls = 0
        while True:
            payload = self._request_json(
                "GET",
                f"{self.JOBS_PATH}/{run_id}",
                expected_statuses={200},
            )
            if status_log_path is not None:
                status_log_path.parent.mkdir(parents=True, exist_ok=True)
                with status_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
            if payload.get("done"):
                error = payload.get("error")
                if error:
                    message = error.get("message") if isinstance(error, Mapping) else error
                    raise OpenDDEAPIError(f"OpenDDE job {run_id} failed: {message}")
                if not isinstance(payload.get("response"), Mapping):
                    raise OpenDDEAPIError(f"OpenDDE job {run_id} completed without a response")
                return payload
            stalled_polls = stalled_polls + 1 if payload.get("possibly_stalled") else 0
            if stalled_poll_limit > 0 and stalled_polls >= stalled_poll_limit:
                raise OpenDDEAPIError(
                    f"OpenDDE job {run_id} reported possibly_stalled for {stalled_polls} consecutive polls"
                )
            if time.monotonic() >= deadline:
                stalled = " (service reports possibly_stalled)" if payload.get("possibly_stalled") else ""
                raise TimeoutError(f"OpenDDE job {run_id} did not finish within {timeout_seconds:g}s{stalled}")
            time.sleep(max(0.05, poll_interval_seconds))

    def download(self, run_id: str, destination: Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with self._client.stream("GET", f"{self.JOBS_PATH}/{run_id}/download") as response:
                self._raise_for_status(response, {200})
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    @staticmethod
    def extract_download(archive: Path, destination: Path) -> None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise OpenDDEAPIError(f"unsafe path in OpenDDE result archive: {member.filename!r}")
            bundle.extractall(destination)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        expected_statuses: set[int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise OpenDDEAPIError(f"{method} {path} failed: {detail}") from exc
        self._raise_for_status(response, expected_statuses)
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenDDEAPIError(f"{method} {path} returned non-JSON content") from exc
        if not isinstance(payload, dict):
            raise OpenDDEAPIError(f"{method} {path} returned a non-object response")
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response, expected: set[int]) -> None:
        if response.status_code in expected:
            return
        body = response.text.strip()
        if len(body) > 1000:
            body = body[:1000] + "..."
        raise OpenDDEAPIError(
            f"{response.request.method} {response.request.url.path} failed: "
            f"HTTP {response.status_code}; response={body or '<empty>'}"
        )
