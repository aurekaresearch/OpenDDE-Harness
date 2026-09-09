"""Static compute-worker registry and task-level worker selection."""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import httpx

from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_URL
from opendde_harness.plugin.protein_design.core.contracts import HealthResponse, WorkflowConfig
from opendde_harness.plugin.protein_design.servers.client import ProteinDesignComputeClient, ProteinDesignComputeError
from opendde_harness.plugin.protein_design.servers.local_service import (
    ComputeEndpoint,
    ensure_compute_service,
    is_local_placement,
)


def _is_loopback(url: str) -> bool:
    return urlsplit(url).hostname in {"127.0.0.1", "localhost", "::1"}


class ReconnectingComputeClient(ProteinDesignComputeClient):
    """Client for the on-demand local container: one connection failure re-resolves the endpoint and retries.

    A refused connection means the request never reached a server, so the
    retry is safe for every method. Re-resolving starts the container again
    when it was stopped (idle timeout or ``compute stop --force``) and follows
    it to a new port.
    """

    def __init__(
        self,
        base_url: str,
        *,
        resolver: Callable[[], ComputeEndpoint],
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(base_url, timeout=timeout, transport=transport, token=token)
        self._resolver = resolver
        self._timeout = timeout
        self._transport = transport

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await super()._send(method, path, **kwargs)
        except ProteinDesignComputeError as exc:
            if not isinstance(exc.__cause__, httpx.ConnectError):
                raise
        endpoint = await asyncio.to_thread(self._resolver)
        await self._client.aclose()
        self._client = httpx.AsyncClient(
            base_url=endpoint.url,
            timeout=self._timeout,
            transport=self._transport,
            headers={"Authorization": f"Bearer {endpoint.token}"} if endpoint.token else None,
        )
        return await super()._send(method, path, **kwargs)


def _string_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(item) for item in value)


def normalize_compute_url(value: str) -> str:
    url = str(value).strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid protein-design compute URL: {value!r}")
    return url


@dataclass(frozen=True)
class ComputeWorker:
    worker_id: str
    url: str
    token: str | None = None
    profiles: frozenset[str] = frozenset()
    backends: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_token: str | None = None,
    ) -> "ComputeWorker":
        url = normalize_compute_url(str(value.get("url") or ""))
        worker_id = str(value.get("id") or value.get("worker_id") or urlsplit(url).hostname or url)
        return cls(
            worker_id=worker_id,
            url=url,
            token=str(value.get("token") or default_token or "") or None,
            profiles=_string_set(value.get("profiles") or value.get("profile")),
            backends=_string_set(value.get("backends")),
        )


@dataclass(frozen=True)
class ComputeSelection:
    worker: ComputeWorker
    client: ProteinDesignComputeClient
    health: HealthResponse


class ComputePool:
    def __init__(
        self,
        workers: list[ComputeWorker],
        *,
        default_url: str = DEFAULT_COMPUTE_URL,
        default_token: str | None = None,
        timeout: float = 60.0,
        client_factory: Callable[..., ProteinDesignComputeClient] = ProteinDesignComputeClient,
        endpoint_resolver: Callable[[], ComputeEndpoint] | None = None,
    ) -> None:
        self._workers = list(workers)
        self._default_url = normalize_compute_url(default_url)
        self._default_token = default_token
        self._timeout = timeout
        self._client_factory = client_factory
        self._resolver = endpoint_resolver
        self._clients: dict[tuple[str, str | None], ProteinDesignComputeClient] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ComputePool":
        default_token = str(config.get("compute_token") or "") or None
        raw_workers = config.get("compute_workers") or []
        if not isinstance(raw_workers, list):
            raise ValueError("protein-design compute_workers must be a list")
        workers = [
            ComputeWorker.from_mapping(item, default_token=default_token)
            for item in raw_workers
            if isinstance(item, Mapping)
        ]
        if len(workers) != len(raw_workers):
            raise ValueError("each protein-design compute worker must be an object")
        worker_ids = [worker.worker_id for worker in workers]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("protein-design compute worker IDs must be unique")
        return cls(
            workers,
            default_url=str(config.get("compute_url") or DEFAULT_COMPUTE_URL),
            default_token=default_token,
            timeout=float(config.get("request_timeout", 60.0)),
            endpoint_resolver=functools.partial(ensure_compute_service, dict(config))
            if is_local_placement(config)
            else None,
        )

    def client(self, worker: ComputeWorker) -> ProteinDesignComputeClient:
        key = (worker.url, worker.token)
        client = self._clients.get(key)
        if client is None:
            if self._resolver is not None:
                client = ReconnectingComputeClient(
                    worker.url,
                    resolver=self._resolver,
                    timeout=self._timeout,
                    token=worker.token,
                )
            else:
                client = self._client_factory(
                    worker.url,
                    timeout=self._timeout,
                    token=worker.token,
                )
            self._clients[key] = client
        return client

    async def select(self, config: WorkflowConfig) -> ComputeSelection:
        if self._resolver is not None and not self._workers:
            endpoint = await asyncio.to_thread(self._resolver)
            self._default_url = normalize_compute_url(endpoint.url)
            self._default_token = endpoint.token
            # A loopback URL bound by an earlier run names this same on-demand container, whose port may have changed.
            if config.compute_url and _is_loopback(config.compute_url):
                config = config.model_copy(update={"compute_url": None, "compute_worker_id": None})
        candidates, explicit = self._candidates(config)
        if not candidates:
            requested = config.compute_worker_id or config.compute_profile or config.compute_url
            raise RuntimeError(f"no protein-design compute worker matches {requested or config.fold_backend!r}")

        results = await asyncio.gather(
            *(self._probe(worker, config) for worker in candidates),
            return_exceptions=True,
        )
        healthy: list[ComputeSelection] = []
        errors: list[str] = []
        for worker, result in zip(candidates, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{worker.worker_id}: {type(result).__name__}: {result}")
            elif result.health.status == "ok":
                healthy.append(result)
            else:
                errors.append(f"{worker.worker_id}: health={result.health.status}")
        if not healthy:
            scope = "requested worker" if explicit else "worker pool"
            raise RuntimeError(f"protein-design {scope} is unavailable: " + "; ".join(errors))
        return min(healthy, key=self._load_key)

    def _candidates(self, config: WorkflowConfig) -> tuple[list[ComputeWorker], bool]:
        if config.compute_url:
            url = normalize_compute_url(config.compute_url)
            matched = [worker for worker in self._workers if worker.url == url]
            if matched:
                if config.compute_worker_id:
                    matched = [worker for worker in matched if worker.worker_id == config.compute_worker_id]
                return self._filter_backend(matched, config.fold_backend), True
            worker_id = config.compute_worker_id or urlsplit(url).hostname or url
            return self._filter_backend(
                [ComputeWorker(worker_id=worker_id, url=url, token=self._default_token)],
                config.fold_backend,
            ), True
        if config.compute_worker_id:
            return self._filter_backend(
                [worker for worker in self._workers if worker.worker_id == config.compute_worker_id],
                config.fold_backend,
            ), True

        candidates = self._workers or [
            ComputeWorker(
                worker_id=urlsplit(self._default_url).hostname or "default",
                url=self._default_url,
                token=self._default_token,
            )
        ]
        if config.compute_profile:
            candidates = [worker for worker in candidates if config.compute_profile in worker.profiles]
        return self._filter_backend(candidates, config.fold_backend), bool(config.compute_profile)

    @staticmethod
    def _filter_backend(workers: list[ComputeWorker], backend: str) -> list[ComputeWorker]:
        return [worker for worker in workers if not worker.backends or backend in worker.backends]

    async def _probe(
        self,
        worker: ComputeWorker,
        config: WorkflowConfig,
    ) -> ComputeSelection:
        client = self.client(worker)
        health = await client.health(
            backend=config.fold_backend,
            execution_mode=str(config.fold_options.get("execution_mode") or "") or None,
            image=str(config.fold_options.get("image") or "") or None,
            api_url=str(config.fold_options.get("api_url") or "") or None,
        )
        return ComputeSelection(worker=worker, client=client, health=health)

    @staticmethod
    def _load_key(selection: ComputeSelection) -> tuple[float, int, str]:
        queue = selection.health.workers.get("queue") or {}
        capacity = max(1, int(queue.get("max_concurrent") or 1))
        running = int(queue.get("running") or 0)
        queued = int(queue.get("queued") or 0)
        return ((running + queued) / capacity, queued, selection.worker.worker_id)
