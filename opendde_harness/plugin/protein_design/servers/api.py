"""FastAPI application exposing the protein-design compute harness."""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opendde_harness.plugin.protein_design.core.constants import DEFAULT_COMPUTE_PORT
from opendde_harness.plugin.protein_design.core.contracts import (
    AnalysisResponse,
    EpitopeAnalysisRequest,
    Esm2GuidedProposalRequest,
    EsmScoreRequest,
    EsmScoreResponse,
    EvolutionTreeRequest,
    FoldRequest,
    JobResult,
    JobState,
    JobSubmission,
    Placement,
    ProtrekSequenceSearchRequest,
    ProtrekStructureSearchRequest,
    ShutdownRequest,
    SolubleMPNNRequest,
    StructureAnalysisRequest,
    StructureReadResponse,
    TargetMsaSearchRequest,
    TargetMsaSearchResponse,
)
from opendde_harness.plugin.protein_design.core.external import (
    ExternalServiceUnavailableError,
    required_message,
)
from opendde_harness.plugin.protein_design.servers.harness import (
    PythonProteinDesignHarness,
    _probe_gpus,
    build_harness_from_environment,
)
from opendde_harness.plugin.protein_design.servers.leases import (
    DEFAULT_SHARED_JOBS_PER_GPU,
    FOLD_KIND,
    GpuLeaseTable,
)

DEFAULT_IDLE_SECONDS = 600.0
DEFAULT_TASK_LEASE_SECONDS = 180.0
IDLE_POLL_SECONDS = 30.0


@dataclass
class _Job:
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None
    created_at: float = 0.0
    cancel_requested: bool = False


def visible_cuda_devices() -> list[int]:
    try:
        import torch
    except ImportError:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except (AssertionError, RuntimeError):
        return []


def free_gpu_memory() -> dict[int, int]:
    return {int(gpu["index"]): int(gpu["memory_free_mb"]) for gpu in _probe_gpus()}


def with_leased_devices(operation: str, payload: dict[str, Any], devices: Sequence[int]) -> dict[str, Any]:
    """Point one harness call at the GPUs its lease granted."""

    if not devices:
        return payload
    payload = dict(payload)
    primary = f"cuda:{devices[0]}"
    if operation == "fold":
        options = dict(payload.get("options") or {})
        options["gpus"] = ",".join(str(index) for index in devices)
        esm2_options = dict(options.get("esm2_options") or {})
        if not esm2_options.get("device"):
            esm2_options["device"] = primary
        options["esm2_options"] = esm2_options
        payload["options"] = options
    elif operation == "generate_soluble_mpnn":
        parameters = dict(payload.get("parameters") or {})
        parameters["device"] = primary
        payload["parameters"] = parameters
    else:
        options = dict(payload.get("options") or {})
        options["device"] = primary
        payload["options"] = options
    return payload


def fold_gpu_request(placement: Placement | None, options: dict[str, Any]) -> tuple[int, list[int] | None]:
    """Resolve one fold's GPU demand into a (count, explicit devices) lease request."""

    spec = str(options.get("gpus", "")).strip().strip('"').strip("'").removeprefix("device=")
    if spec == "none" or str(options.get("device", "")).strip().lower() == "cpu":
        return 0, None
    if placement is not None and placement.fold:
        return len(placement.fold), list(placement.fold)
    if spec and spec != "all":
        devices = [int(part) for part in spec.split(",") if part.strip()]
        if devices:
            return len(devices), devices
    return (placement.cp_degree if placement is not None else 1), None


class TaskLeases:
    """Heartbeat leases that keep the worker alive while a design task runs.

    A detached design task spends long LLM-only phases between compute calls, so
    job activity alone cannot tell the watchdog that the container is needed.
    """

    def __init__(self, *, ttl: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl = max(1.0, float(ttl))
        self._clock = clock
        self._expiry: dict[str, float] = {}

    def refresh(self, task_id: str) -> None:
        self._expiry[task_id] = self._clock() + self.ttl

    def release(self, task_id: str) -> None:
        self._expiry.pop(task_id, None)

    def active(self) -> list[dict[str, Any]]:
        now = self._clock()
        for task_id in [task_id for task_id, expires in self._expiry.items() if expires <= now]:
            del self._expiry[task_id]
        return [
            {"task_id": task_id, "expires_in": round(expires - now, 3)}
            for task_id, expires in self._expiry.items()
        ]


class IdleWatchdog:
    """Retire an unused compute worker so its GPUs return to the host."""

    def __init__(
        self,
        *,
        timeout: float,
        busy: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout = max(0.0, float(timeout))
        self._busy = busy
        self._clock = clock
        self._last = clock()

    def touch(self) -> None:
        self._last = self._clock()

    def seconds(self) -> float:
        if self._busy():
            self.touch()
            return 0.0
        return max(0.0, self._clock() - self._last)

    def expired(self) -> bool:
        return bool(self.timeout) and self.seconds() >= self.timeout

    async def run(self, on_expire: Callable[[], None]) -> None:
        if not self.timeout:
            return
        while not self.expired():
            await asyncio.sleep(min(IDLE_POLL_SECONDS, self.timeout))
        on_expire()


class JobStore:
    def __init__(self, harness: PythonProteinDesignHarness, *, leases: GpuLeaseTable) -> None:
        self._harness = harness
        self._jobs: dict[str, _Job] = {}
        self._leases = leases
        self._max_retained = 500
        self._accepting = True

    def submit(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        kind: str = FOLD_KIND,
        count: int = 1,
        devices: Sequence[int] | None = None,
    ) -> JobSubmission:
        if not self._accepting:
            raise RuntimeError("compute worker is shutting down and accepts no new jobs")
        job_id = uuid.uuid4().hex
        self._prune()
        job = _Job(created_at=time.monotonic())
        self._jobs[job_id] = job
        job.task = asyncio.create_task(
            self._run(job, job_id, operation, payload, kind=kind, count=count, devices=devices),
            name=f"protein-design-job-{job_id}",
        )
        return JobSubmission(job_id=job_id)

    def get(self, job_id: str) -> JobResult:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return JobResult(
            job_id=job_id,
            status=job.state,
            progress=job.progress,
            result=job.result,
            error=job.error,
        )

    def queue_stats(self) -> dict[str, int]:
        """Small scheduler contract consumed by the multi-worker dispatcher."""

        return {
            "max_concurrent": self._leases.capacity,
            "running": sum(job.state == JobState.RUNNING for job in self._jobs.values()),
            "queued": sum(job.state == JobState.QUEUED for job in self._jobs.values()),
        }

    @property
    def busy(self) -> bool:
        stats = self.queue_stats()
        return bool(stats["running"] or stats["queued"] or self._leases.busy)

    @property
    def accepting(self) -> bool:
        return self._accepting

    def stop_accepting(self) -> None:
        self._accepting = False

    async def drain(self) -> None:
        active = [job.task for job in self._jobs.values() if job.task and not job.task.done()]
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def cancel(self, job_id: str) -> JobResult:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.cancel_requested = True
        # A running Python/CUDA call executes in a worker thread and cannot be
        # force-cancelled safely. Keep it behind its GPU lease until it exits,
        # discard its result, and report cancellation immediately.
        # A queued job has not entered the harness and can be cancelled now.
        if job.state == JobState.QUEUED and job.task is not None and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        job.state = JobState.CANCELLED
        return self.get(job_id)

    def _prune(self) -> None:
        if len(self._jobs) < self._max_retained:
            return
        completed = sorted(
            (
                (job_id, job)
                for job_id, job in self._jobs.items()
                if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
            ),
            key=lambda item: item[1].created_at,
        )
        for job_id, _job in completed[: max(1, len(self._jobs) - self._max_retained + 1)]:
            self._jobs.pop(job_id, None)

    async def close(self) -> None:
        active = [job.task for job in self._jobs.values() if job.task and not job.task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def _run(
        self,
        job: _Job,
        job_id: str,
        operation: str,
        payload: dict[str, Any],
        *,
        kind: str,
        count: int,
        devices: Sequence[int] | None,
    ) -> None:
        try:
            async with self._leases.lease(kind, job_id=job_id, count=count, devices=devices) as granted:
                job.state = JobState.RUNNING
                result = await self._harness.invoke(operation, with_leased_devices(operation, payload, granted))
                if not job.cancel_requested:
                    job.result = result
                    job.progress = 1.0
                    job.state = JobState.SUCCEEDED
        except asyncio.CancelledError:
            job.state = JobState.CANCELLED
            raise
        except Exception as exc:
            if not job.cancel_requested:
                job.error = str(exc)
                job.state = JobState.FAILED


def create_app(
    harness: PythonProteinDesignHarness | None = None,
    *,
    exit_process: Callable[[], None] | None = None,
):
    try:
        from fastapi import FastAPI, Header, HTTPException, Query, Request
        from fastapi.responses import JSONResponse, Response
    except ImportError as exc:
        raise RuntimeError("Install OpenDDE Harness with the 'protein-design' extra to run the compute service") from exc

    from opendde_harness.cli.compute_assets import model_environment

    for key, value in model_environment(os.environ.get("OPENDDE_HARNESS_WEIGHTS_DIR", "/weights")).items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("OPENDDE_HARNESS_PROJECT_ROOT", "/workspace")
    os.environ.setdefault("STRUCTPRED_OPENDDE_CODE_DIR", "/workspace/external/opendde")
    os.environ.setdefault("PROTEIN_DESIGN_OUTPUT_PATH", "/output")
    environment = None
    code_version = None
    if Path("/opt/compute-environment.json").is_file():
        from opendde_harness import __version__
        from opendde_harness.cli.compute_code import MANIFEST, verify_runtime_code
        from opendde_harness.cli.compute_environment import check_runtime_environment

        environment = check_runtime_environment()
        code_root = Path(os.environ.get("OPENDDE_HARNESS_PROJECT_ROOT", "/workspace"))
        code_version = verify_runtime_code(code_root)["harness_version"] if (code_root / MANIFEST).is_file() else __version__
    active_harness = harness or build_harness_from_environment()
    leases = GpuLeaseTable(
        visible_cuda_devices(),
        shared_jobs_per_gpu=int(
            os.environ.get("OPENDDE_HARNESS_SHARED_JOBS_PER_GPU", str(DEFAULT_SHARED_JOBS_PER_GPU))
        ),
        free_memory=free_gpu_memory,
    )
    jobs = JobStore(active_harness, leases=leases)
    task_leases = TaskLeases(
        ttl=float(os.environ.get("OPENDDE_HARNESS_TASK_LEASE_SECONDS", str(DEFAULT_TASK_LEASE_SECONDS)))
    )
    idle = IdleWatchdog(
        timeout=float(os.environ.get("OPENDDE_HARNESS_COMPUTE_IDLE_SECONDS", str(DEFAULT_IDLE_SECONDS))),
        busy=lambda: jobs.busy or bool(task_leases.active()),
    )
    stop_process = exit_process or (lambda: signal.raise_signal(signal.SIGTERM))

    async def retire(*, drain: bool) -> None:
        if drain:
            await jobs.drain()
        # Let the accepted response reach the host before uvicorn tears the
        # process down.
        await asyncio.sleep(0.2)
        stop_process()

    @asynccontextmanager
    async def lifespan(_app):
        watchdog = asyncio.create_task(idle.run(stop_process), name="protein-design-idle-watchdog")
        try:
            yield
        finally:
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
            await jobs.close()
            close = getattr(active_harness, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

    app = FastAPI(title="OpenDDE Harness Protein Design Compute", version="1", lifespan=lifespan)
    app.state.jobs = jobs
    app.state.leases = leases
    app.state.idle = idle
    app.state.task_leases = task_leases

    @app.middleware("http")
    async def record_activity(request: Request, call_next):
        if request.url.path != "/health":
            idle.touch()
        return await call_next(request)

    configured_token = os.environ.get("OPENDDE_HARNESS_PROTEIN_DESIGN_TOKEN", "").strip()
    if configured_token:

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            if request.url.path != "/health":
                supplied = request.headers.get("authorization", "")
                if supplied != f"Bearer {configured_token}":
                    return JSONResponse(status_code=401, content={"detail": "unauthorized"})
            return await call_next(request)

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    async def invalid_request(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    async def invoke_analysis(operation: str, payload: dict[str, Any]) -> AnalysisResponse:
        try:
            return AnalysisResponse.model_validate(await active_harness.invoke(operation, payload))
        except ExternalServiceUnavailableError as exc:
            return AnalysisResponse.model_validate(exc.payload())

    async def invoke_leased(operation: str, request: Any, kind: str) -> dict[str, Any]:
        """Run one short GPU tool under a shared lease taken for its duration."""

        if not jobs.accepting:
            raise HTTPException(status_code=503, detail="compute worker is shutting down")
        payload = request.model_dump()
        payload.pop("placement", None)
        requested = getattr(request.placement, kind, None) if request.placement else None
        async with leases.lease(
            kind,
            job_id=uuid.uuid4().hex,
            devices=None if requested is None else [requested],
        ) as granted:
            return await active_harness.invoke(operation, with_leased_devices(operation, payload, granted))

    @app.get("/health")
    async def health(
        backend: str | None = None,
        execution_mode: str | None = None,
        image: str | None = None,
        api_url: str | None = None,
        probe_external: bool = False,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = dict(
            await active_harness.health(
                backend, execution_mode, image, api_url, probe_external=probe_external
            )
        )
        queue = jobs.queue_stats()
        if configured_token and authorization != f"Bearer {configured_token}":
            return {"status": payload["status"], "workers": {"queue": queue}}
        workers = dict(payload.get("workers") or {})
        workers["queue"] = queue
        workers["gpu_leases"] = leases.snapshot()
        workers["idle_seconds"] = round(idle.seconds(), 3)
        workers["idle_timeout_seconds"] = idle.timeout
        workers["task_leases"] = task_leases.active()
        payload["workers"] = workers
        if environment is not None:
            payload["environment"] = environment
            payload["code_version"] = code_version
        return payload

    @app.post("/fold", response_model=JobSubmission)
    async def fold(request: FoldRequest) -> JobSubmission:
        payload = request.model_dump()
        placement = request.placement
        count, devices = fold_gpu_request(placement, payload.get("options") or {})
        payload.pop("placement", None)
        try:
            return jobs.submit("fold", payload, kind=FOLD_KIND, count=count, devices=devices)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/shutdown")
    async def shutdown(request: ShutdownRequest) -> JSONResponse:
        queue = jobs.queue_stats()
        tasks = len(task_leases.active())
        if request.if_idle and (jobs.busy or tasks):
            return JSONResponse(
                status_code=409,
                content={"detail": "busy", "running": queue["running"], "queued": queue["queued"], "tasks": tasks},
            )
        jobs.stop_accepting()
        app.state.shutdown = asyncio.create_task(
            retire(drain=not request.if_idle), name="protein-design-shutdown"
        )
        return JSONResponse(
            status_code=202,
            content={"detail": "shutting down", "running": queue["running"], "queued": queue["queued"]},
        )

    @app.put("/leases/{task_id}", response_class=Response)
    async def refresh_task_lease(task_id: str):
        task_leases.refresh(task_id)
        return Response(status_code=204)

    @app.delete("/leases/{task_id}", response_class=Response)
    async def release_task_lease(task_id: str):
        task_leases.release(task_id)
        return Response(status_code=204)

    @app.post("/search/msa/target", response_model=TargetMsaSearchResponse)
    async def search_target_msa(request: TargetMsaSearchRequest) -> TargetMsaSearchResponse:
        try:
            result = await active_harness.invoke("target_msa_search", request.model_dump())
        except ExternalServiceUnavailableError as exc:
            if request.required:
                raise HTTPException(
                    status_code=502,
                    detail=required_message(exc.service, exc.reason, exc.endpoint),
                ) from exc
            result = {
                "target_name": request.target_name,
                "chain_id": request.chain_id,
                "server_url": exc.endpoint or "",
                "server_mode": "",
                "available": False,
                "reason": exc.reason,
            }
        return TargetMsaSearchResponse.model_validate(result)

    @app.get("/fold/{job_id}", response_model=JobResult)
    async def fold_status(job_id: str) -> JobResult:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown fold job") from exc

    @app.delete("/fold/{job_id}", response_model=JobResult)
    async def cancel_fold(job_id: str) -> JobResult:
        try:
            return await jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown fold job") from exc

    @app.post("/score/esm", response_model=EsmScoreResponse)
    async def score_esm(request: EsmScoreRequest) -> EsmScoreResponse:
        return EsmScoreResponse.model_validate(await invoke_leased("score_esm", request, "esm"))

    @app.post("/generate/soluble_mpnn")
    async def generate_soluble_mpnn(request: SolubleMPNNRequest) -> dict[str, Any]:
        return await invoke_leased("generate_soluble_mpnn", request, "mpnn")

    @app.get("/population")
    async def population(task_id: str | None = None) -> dict[str, Any]:
        return await active_harness.invoke("population_get", {"task_id": task_id})

    @app.post("/population")
    async def update_population(payload: dict[str, Any]) -> dict[str, Any]:
        return await active_harness.invoke("population_update", payload)

    @app.get("/population/top")
    async def top_candidates(
        top_k: int = Query(default=20, ge=1),
        task_id: str | None = None,
        minimize: bool = True,
    ) -> dict[str, Any]:
        return await active_harness.invoke(
            "population_top",
            {"top_k": top_k, "task_id": task_id, "minimize": minimize},
        )

    @app.post("/qc/developability")
    async def developability(payload: dict[str, Any]) -> dict[str, Any]:
        return await active_harness.invoke("developability", payload)

    @app.post("/analysis/evolution-tree", response_model=AnalysisResponse)
    async def analyze_evolution_tree(request: EvolutionTreeRequest) -> AnalysisResponse:
        return await invoke_analysis("evolution_tree", request.model_dump())

    @app.post("/analysis/epitope", response_model=AnalysisResponse)
    async def analyze_epitope(request: EpitopeAnalysisRequest) -> AnalysisResponse:
        return await invoke_analysis("epitope_analysis", request.model_dump())

    @app.post("/analysis/structure", response_model=AnalysisResponse)
    async def analyze_structure(request: StructureAnalysisRequest) -> AnalysisResponse:
        return await invoke_analysis("structure_analysis", request.model_dump())

    @app.post("/analysis/pose-rmsd")
    async def pose_rmsd(payload: dict[str, Any]) -> dict[str, Any]:
        return await active_harness.invoke("pose_rmsd", payload)

    @app.post("/search/protrek/sequence", response_model=AnalysisResponse)
    async def search_protrek_sequence(request: ProtrekSequenceSearchRequest) -> AnalysisResponse:
        return await invoke_analysis("protrek_sequence_search", request.model_dump())

    @app.post("/search/protrek/structure", response_model=AnalysisResponse)
    async def search_protrek_structure(request: ProtrekStructureSearchRequest) -> AnalysisResponse:
        return await invoke_analysis("protrek_structure_search", request.model_dump())

    @app.post("/generate/esm2-guided", response_model=AnalysisResponse)
    async def generate_esm2_guided(request: Esm2GuidedProposalRequest) -> AnalysisResponse:
        return AnalysisResponse.model_validate(await invoke_leased("generate_esm2_guided", request, "esm"))

    @app.get("/structure", response_model=StructureReadResponse)
    async def structure(
        path: str = Query(min_length=1),
        task_id: str | None = None,
    ) -> StructureReadResponse:
        result = await active_harness.invoke("structure_read", {"path": path, "task_id": task_id})
        return StructureReadResponse.model_validate(result)

    return app


def run() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install OpenDDE Harness with the 'protein-design' extra to run the compute service") from exc
    uvicorn.run(
        "opendde_harness.plugin.protein_design.servers.api:create_app",
        factory=True,
        host=os.environ.get("OPENDDE_HARNESS_PROTEIN_DESIGN_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPENDDE_HARNESS_PROTEIN_DESIGN_PORT", str(DEFAULT_COMPUTE_PORT))),
    )
