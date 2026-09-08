"""Optional external services (ProTrek, the online MSA server) and their outages.

A compute container without egress must degrade to an unavailable result rather
than failing a design run, so connection-level failures are classified here and
translated into one structured shape shared by the service, the agent tools and
the readiness checks.
"""

from __future__ import annotations

import errno
import os
import socket
from typing import Any

PROTREK_SERVICE = "protrek"
MSA_SERVICE = "msa"

SERVICE_LABELS = {
    PROTREK_SERVICE: "ProTrek",
    MSA_SERVICE: "MSA server",
}

PROXY_HINT = (
    "give the compute container egress: set http_proxy/https_proxy on the host before "
    "ddeharness onboard, or add compute_docker.env in config.json"
)

DEFAULT_PROBE_TIMEOUT_SECONDS = 3.0
# The upstream ProTrek service serves plain HTTP only; https hangs until timeout.
DEFAULT_PROTREK_URL = "http://search-protrek.com/"
DEFAULT_MSA_SERVER_URL = "https://protenix-server.com/api/msa"

_NETWORK_ERRNOS = frozenset(
    {
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    }
)

_MESSAGE_MARKERS = (
    "connecttimeout",
    "connecterror",
    "connectionerror",
    "connection refused",
    "connection reset",
    "could not fetch config",
    "could not resolve",
    "name or service not known",
    "network is unreachable",
    "nodename nor servname",
    "no route to host",
    "proxyerror",
    "readtimeout",
    "temporary failure in name resolution",
    "timed out",
)


class ExternalServiceUnavailableError(RuntimeError):
    """One optional external service could not be reached."""

    def __init__(self, service: str, reason: str, endpoint: str | None = None) -> None:
        super().__init__(f"{SERVICE_LABELS.get(service, service)} is unavailable: {reason}")
        self.service = service
        self.reason = reason
        self.endpoint = endpoint

    def payload(self) -> dict[str, Any]:
        return unavailable_payload(self.service, self.reason, self.endpoint)


def short_reason(exc: BaseException) -> str:
    message = " ".join(str(exc).split())[:160]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def connection_reason(exc: BaseException) -> str | None:
    """Short reason when ``exc`` or anything it wraps is a connection-level failure."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ExternalServiceUnavailableError):
            return current.reason
        if _is_connection_error(current):
            return short_reason(exc if current is exc else current)
        current = current.__cause__ or current.__context__
    return None


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (socket.gaierror, socket.herror, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError) and exc.errno in _NETWORK_ERRNOS:
        return True
    try:
        import httpx
    except ImportError:
        httpx = None
    if httpx is not None and isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ProxyError,
            httpx.NetworkError,
        ),
    ):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _MESSAGE_MARKERS)


def unavailable_payload(service: str, reason: str, endpoint: str | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "service": service,
        "reason": reason,
        "endpoint": endpoint,
        "result": None,
        "error": None,
    }


def unavailable_message(service: str, reason: str) -> str:
    label = SERVICE_LABELS.get(service, service)
    return f"{label} search is unavailable ({reason}); continue without it."


def required_message(service: str, reason: str, endpoint: str | None) -> str:
    label = SERVICE_LABELS.get(service, service)
    target = endpoint or "the configured endpoint"
    return f"{label} at {target} is unreachable ({reason}); {PROXY_HINT}"


def protrek_endpoint() -> str | None:
    """Remote ProTrek endpoint, or None when PROTREK_ENDPOINT is set to an empty value."""
    endpoint = os.environ.get("PROTREK_ENDPOINT")
    if endpoint is None:
        return DEFAULT_PROTREK_URL
    return endpoint.strip() or None


def msa_endpoint() -> str:
    return os.environ.get("MMSEQS_SERVICE_HOST_URL", DEFAULT_MSA_SERVER_URL).strip() or DEFAULT_MSA_SERVER_URL


def probe_endpoint(
    endpoint: str | None,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Reachability of one external endpoint from this process, never raising."""

    if not endpoint:
        return {"reachable": False, "endpoint": None, "reason": "disabled"}
    try:
        import httpx
    except ImportError as exc:
        return {"reachable": False, "endpoint": endpoint, "reason": short_reason(exc)}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            client.head(endpoint)
    except Exception as exc:
        reason = connection_reason(exc)
        if reason is not None:
            return {"reachable": False, "endpoint": endpoint, "reason": reason}
        # A protocol-level rejection still proves the endpoint answered.
        return {"reachable": True, "endpoint": endpoint, "reason": None}
    return {"reachable": True, "endpoint": endpoint, "reason": None}
