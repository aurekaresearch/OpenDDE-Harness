"""GPU leases for the compute service's in-process job scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

FOLD_KIND = "fold"
DEFAULT_SHARED_JOBS_PER_GPU = 2


@dataclass
class _Request:
    kind: str
    job_id: str
    count: int
    devices: tuple[int, ...] | None
    granted: tuple[int, ...] = field(default=())


class GpuLeaseTable:
    """Hand out the worker's GPUs to concurrent jobs.

    A fold holds its GPUs exclusively; ESM and SolubleMPNN jobs share one GPU
    up to ``shared_jobs_per_gpu`` but never join a running fold.  Waiters are
    served first-in-first-out, so a fold waiting for a busy GPU set cannot be
    starved by a stream of shared jobs.
    """

    def __init__(
        self,
        devices: Iterable[int],
        *,
        shared_jobs_per_gpu: int = DEFAULT_SHARED_JOBS_PER_GPU,
        free_memory: Callable[[], dict[int, int]] | None = None,
    ) -> None:
        self._devices = sorted({int(index) for index in devices})
        self._shared_jobs_per_gpu = max(1, int(shared_jobs_per_gpu))
        self._free_memory = free_memory or (lambda: {})
        self._condition = asyncio.Condition()
        self._waiting: list[_Request] = []
        self._held: list[_Request] = []
        self._recent: dict[str, tuple[int, ...]] = {}

    @property
    def devices(self) -> list[int]:
        return list(self._devices)

    @property
    def capacity(self) -> int:
        return max(1, len(self._devices))

    @property
    def busy(self) -> bool:
        return bool(self._held or self._waiting)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "jobs": [
                    {"job_id": request.job_id, "kind": request.kind}
                    for request in self._held
                    if index in request.granted
                ],
            }
            for index in self._devices
        ]

    @asynccontextmanager
    async def lease(
        self,
        kind: str,
        *,
        job_id: str,
        count: int = 1,
        devices: Sequence[int] | None = None,
    ):
        requested = tuple(int(index) for index in devices) if devices is not None else None
        if requested is not None:
            unknown = sorted(set(requested) - set(self._devices))
            if unknown:
                raise ValueError(
                    "requested GPU placement is not available on this compute worker: "
                    + ", ".join(map(str, unknown))
                )
        request = _Request(kind=kind, job_id=job_id, count=max(0, int(count)), devices=requested)
        free = self._free_memory() if requested is None and self._devices else {}
        async with self._condition:
            self._waiting.append(request)
            try:
                while self._waiting[0] is not request or not self._grant(request, free):
                    await self._condition.wait()
            finally:
                if request in self._waiting:
                    self._waiting.remove(request)
                self._condition.notify_all()
        try:
            yield list(request.granted)
        finally:
            async with self._condition:
                if request in self._held:
                    self._held.remove(request)
                self._condition.notify_all()

    def _grant(self, request: _Request, free: dict[int, int]) -> bool:
        selected = self._select(request, free)
        if selected is None:
            return False
        request.granted = tuple(selected)
        # Returning a job kind to the GPUs it used last keeps the resident
        # model of that job warm instead of reloading it elsewhere.
        self._recent[request.kind] = request.granted
        self._held.append(request)
        return True

    def _select(self, request: _Request, free: dict[int, int]) -> list[int] | None:
        if not self._devices:
            return [] if not self._held else None
        if request.devices is None and request.count == 0:
            return []
        holders = {
            index: [held for held in self._held if index in held.granted] for index in self._devices
        }

        def usable(index: int) -> bool:
            running = holders[index]
            if request.kind == FOLD_KIND:
                return not running
            if any(held.kind == FOLD_KIND for held in running):
                return False
            return len(running) < self._shared_jobs_per_gpu

        if request.devices is not None:
            return list(request.devices) if all(map(usable, request.devices)) else None
        count = request.count if request.kind == FOLD_KIND else 1
        recent = set(self._recent.get(request.kind, ()))
        candidates = sorted(
            (index for index in self._devices if usable(index)),
            key=lambda index: (index not in recent, len(holders[index]), -free.get(index, 0), index),
        )
        return candidates[:count] if len(candidates) >= count else None
