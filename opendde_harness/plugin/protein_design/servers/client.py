"""Async client for the protein-design compute REST contract."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

from opendde_harness.plugin.protein_design.core.contracts import (
    AnalysisResponse,
    EpitopeAnalysisRequest,
    Esm2GuidedProposalRequest,
    EsmScoreRequest,
    EsmScoreResponse,
    EvolutionTreeRequest,
    FoldRequest,
    HealthResponse,
    JobResult,
    JobState,
    JobSubmission,
    ProtrekSequenceSearchRequest,
    ProtrekStructureSearchRequest,
    SolubleMPNNRequest,
    StructureAnalysisRequest,
    StructureReadResponse,
    TargetMsaSearchRequest,
    TargetMsaSearchResponse,
)


class ProteinDesignComputeError(RuntimeError):
    pass


EVOLUTION_TREE_REQUEST_TIMEOUT_SECONDS = 150.0


class ProteinDesignComputeClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        token: str | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else None
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(
        self,
        *,
        backend: str | None = None,
        execution_mode: str | None = None,
        image: str | None = None,
        api_url: str | None = None,
    ) -> HealthResponse:
        params = {
            key: value
            for key, value in {
                "backend": backend,
                "execution_mode": execution_mode,
                "image": image,
                "api_url": api_url,
            }.items()
            if value
        } or None
        return HealthResponse.model_validate(
            await self._request("GET", "/health", params=params)
        )

    async def submit_fold(self, request: FoldRequest) -> JobSubmission:
        data = await self._request("POST", "/fold", json=request.model_dump())
        return JobSubmission.model_validate(data)

    async def search_target_msa(
        self, request: TargetMsaSearchRequest
    ) -> TargetMsaSearchResponse:
        data = await self._request(
            "POST",
            "/search/msa/target",
            json=request.model_dump(),
            timeout=3600.0,
        )
        return TargetMsaSearchResponse.model_validate(data)

    async def fold_status(self, job_id: str) -> JobResult:
        return JobResult.model_validate(await self._request("GET", f"/fold/{job_id}"))

    async def cancel_fold(self, job_id: str) -> JobResult:
        return JobResult.model_validate(await self._request("DELETE", f"/fold/{job_id}"))

    async def wait_fold(
        self,
        job_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 3600.0,
        stop_event: asyncio.Event | None = None,
    ) -> JobResult:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                await self.cancel_fold(job_id)
                raise ProteinDesignComputeError(f"fold job {job_id} cancelled by task stop")
            result = await self.fold_status(job_id)
            if result.status == JobState.SUCCEEDED:
                return result
            if result.status in {JobState.FAILED, JobState.CANCELLED}:
                raise ProteinDesignComputeError(result.error or f"fold job {job_id} {result.status}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"fold job {job_id} exceeded {timeout:.0f}s")
            await asyncio.sleep(poll_interval)

    async def refresh_task_lease(self, task_id: str) -> None:
        """Heartbeat that keeps an on-demand worker alive between compute calls."""
        await self._send("PUT", f"/leases/{task_id}")

    async def release_task_lease(self, task_id: str) -> None:
        await self._send("DELETE", f"/leases/{task_id}")

    async def score_esm(self, request: EsmScoreRequest) -> EsmScoreResponse:
        data = await self._request("POST", "/score/esm", json=request.model_dump())
        return EsmScoreResponse.model_validate(data)

    async def generate_soluble_mpnn(self, request: SolubleMPNNRequest) -> dict[str, Any]:
        return await self._request("POST", "/generate/soluble_mpnn", json=request.model_dump())

    async def population(self, task_id: str | None = None) -> dict[str, Any]:
        params = {"task_id": task_id} if task_id else None
        return await self._request("GET", "/population", params=params)

    async def update_population(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/population", json=dict(payload))

    async def top_candidates(
        self,
        top_k: int = 20,
        *,
        task_id: str | None = None,
        minimize: bool = True,
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/population/top",
            params={"top_k": top_k, "task_id": task_id, "minimize": minimize},
        )

    async def developability(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/qc/developability", json=dict(payload))

    async def analyze_evolution_tree(self, request: EvolutionTreeRequest) -> AnalysisResponse:
        data = await self._request(
            "POST",
            "/analysis/evolution-tree",
            json=request.model_dump(),
            timeout=EVOLUTION_TREE_REQUEST_TIMEOUT_SECONDS,
        )
        return AnalysisResponse.model_validate(data)

    async def analyze_epitope(self, request: EpitopeAnalysisRequest) -> AnalysisResponse:
        data = await self._request("POST", "/analysis/epitope", json=request.model_dump())
        return AnalysisResponse.model_validate(data)

    async def analyze_structure(self, request: StructureAnalysisRequest) -> AnalysisResponse:
        data = await self._request("POST", "/analysis/structure", json=request.model_dump())
        return AnalysisResponse.model_validate(data)

    async def search_protrek_sequence(self, request: ProtrekSequenceSearchRequest) -> AnalysisResponse:
        data = await self._request("POST", "/search/protrek/sequence", json=request.model_dump())
        return AnalysisResponse.model_validate(data)

    async def search_protrek_structure(self, request: ProtrekStructureSearchRequest) -> AnalysisResponse:
        data = await self._request("POST", "/search/protrek/structure", json=request.model_dump())
        return AnalysisResponse.model_validate(data)

    async def generate_esm2_guided(self, request: Esm2GuidedProposalRequest) -> AnalysisResponse:
        data = await self._request("POST", "/generate/esm2-guided", json=request.model_dump())
        return AnalysisResponse.model_validate(data)

    async def read_structure(
        self,
        path: str,
        *,
        task_id: str | None = None,
    ) -> StructureReadResponse:
        data = await self._request("GET", "/structure", params={"path": path, "task_id": task_id})
        return StructureReadResponse.model_validate(data)

    async def pose_rmsd(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/analysis/pose-rmsd",
            json=dict(payload),
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._send(method, path, **kwargs)
        data = response.json()
        if not isinstance(data, dict):
            raise ProteinDesignComputeError(f"{method} {path} returned {type(data).__name__}, expected object")
        return data

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text.strip()
            if len(body) > 1_000:
                body = f"{body[:1_000]}..."
            detail = f"; response={body}" if body else ""
            raise ProteinDesignComputeError(
                f"{method} {path} failed: HTTP {exc.response.status_code}{detail}"
            ) from exc
        except httpx.RequestError as exc:
            message = str(exc).strip()
            detail = f": {message}" if message else ""
            raise ProteinDesignComputeError(
                f"{method} {path} failed: {type(exc).__name__}{detail}"
            ) from exc
        return response
