# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from ..bundle import CodebookBundle
from ..codebook_codec import CodebookEnvelope, CodebookCodec
from ..guarantee import ContourViolation, GuaranteeMode, GuaranteeViolation, ContextGuaranteeProfile
from ..context_codec import ContextEnvelope, ContextCodecConfig, ContextCodec
from ..resident_tier import ResidentTierConfig, ResidentPlanner
from ..telemetry import HyperQuantMetrics
from ..vector_codec import VectorCodec, RotatedScalarEnvelope, native_fwht_status
from ..utils import ndarray_from_b64, ndarray_to_b64
from ..defaults import VECTOR_PREFER_NATIVE_FWHT_DEFAULT
from .models import (
    CodebookCompressRequest,
    CodebookCompressResponse,
    CodebookCompressionStatsModel,
    DecompressRequest,
    DecompressResponse,
    HealthResponse,
    VectorCompressRequest,
    VectorCompressResponse,
    VectorCompressionStatsModel,
    ContextCompressRequest,
    ResidentPlanRequest,
    ResidentPlanResponse,
    ContextCompressResponse,
    ContextCompressionStatsModel,
)


DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_HTTP_BODY_OVERHEAD_BYTES = 1024 * 1024


class _BodySizeLimitMiddleware:
    def __init__(self, app, *, max_http_body_bytes: int, metrics: HyperQuantMetrics) -> None:
        self.app = app
        self.max_http_body_bytes = int(max_http_body_bytes)
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body_too_large_detail = f"request body exceeds max_http_body_bytes={self.max_http_body_bytes}"
        for key, value in scope.get("headers", []):
            if key.lower() != b"content-length":
                continue
            try:
                declared = int(value.decode("latin1"))
            except ValueError:
                declared = None
            if declared is not None and declared > self.max_http_body_bytes:
                self.metrics.observe_error("http", "request_too_large")
                await JSONResponse(status_code=413, content={"detail": body_too_large_detail})(scope, receive, send)
                return

        seen = 0
        too_large = False
        sent_too_large_response = False
        request_stream_ended = False

        async def drain_remaining_body() -> None:
            nonlocal request_stream_ended
            while True:
                message = await receive()
                if message.get("type") != "http.request":
                    return
                request_stream_ended = not message.get("more_body", False)
                if not message.get("more_body", False):
                    return

        async def guarded_receive():
            nonlocal seen, too_large, request_stream_ended
            message = await receive()
            if too_large:
                return {"type": "http.request", "body": b"", "more_body": False}
            if message.get("type") == "http.request":
                request_stream_ended = not message.get("more_body", False)
                seen += len(message.get("body", b""))
                if seen > self.max_http_body_bytes:
                    too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message):
            nonlocal sent_too_large_response
            if too_large:
                if not sent_too_large_response:
                    sent_too_large_response = True
                    if not request_stream_ended:
                        await drain_remaining_body()
                    self.metrics.observe_error("http", "request_too_large")
                    await JSONResponse(status_code=413, content={"detail": body_too_large_detail})(scope, receive, send)
                return
            await send(message)

        await self.app(scope, guarded_receive, guarded_send)
        if too_large and not sent_too_large_response:
            if not request_stream_ended:
                await drain_remaining_body()
            self.metrics.observe_error("http", "request_too_large")
            await JSONResponse(status_code=413, content={"detail": body_too_large_detail})(scope, receive, send)


def _pydantic_model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError(f"unsupported pydantic model type: {type(model)!r}")


def create_app(
    bundle_path: str | Path,
    *,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    max_concurrency: int | None = None,
) -> FastAPI:
    bundle = CodebookBundle.load(bundle_path)
    compressor = CodebookCodec(bundle=bundle)
    metrics = HyperQuantMetrics()

    resolved_max_concurrency = int(max_concurrency or max(2, min(32, os.cpu_count() or 2)))
    max_request_bytes = int(max_request_bytes)
    server_prefer_native_fwht = bool(VECTOR_PREFER_NATIVE_FWHT_DEFAULT)
    max_http_body_bytes = int(max_request_bytes * 2 + DEFAULT_MAX_HTTP_BODY_OVERHEAD_BYTES)
    semaphore = asyncio.Semaphore(resolved_max_concurrency)

    @lru_cache(maxsize=64)
    def get_context_compressor(config: ContextCodecConfig) -> ContextCodec:
        return ContextCodec(config)

    @lru_cache(maxsize=64)
    def get_vector_compressor(bits: int, group_size: int, rotation_seed: int, residual_topk: int, prefer_native_fwht: bool) -> VectorCodec:
        return VectorCodec(
            bits=bits,
            group_size=group_size,
            rotation_seed=rotation_seed,
            residual_topk=residual_topk,
            prefer_native_fwht=prefer_native_fwht,
        )

    @lru_cache(maxsize=64)
    def get_resident_planner(
        page_size: int,
        rank: int,
        bits: int,
        group_size: int,
        hot_pages: int,
        rotation_seed: int,
        residual_topk: int,
        prefix_keep_vectors: int,
        suffix_keep_vectors: int,
        low_rank_error_threshold: float,
        ref_round_decimals: int,
        enable_page_ref: bool,
        page_ref_rel_rms_threshold: float,
        enable_int8_fallback: bool,
        try_int8_for_protected: bool,
        int8_rel_rms_threshold: float,
        int8_max_abs_threshold: float,
        prefer_native_fwht: bool,
        allow_vector_for_protected: bool,
    ) -> ResidentPlanner:
        return ResidentPlanner(
            ResidentTierConfig(
                page_size=page_size,
                rank=rank,
                bits=bits,
                group_size=group_size,
                hot_pages=hot_pages,
                rotation_seed=rotation_seed,
                residual_topk=residual_topk,
                prefix_keep_vectors=prefix_keep_vectors,
                suffix_keep_vectors=suffix_keep_vectors,
                low_rank_error_threshold=low_rank_error_threshold,
                ref_round_decimals=ref_round_decimals,
                enable_page_ref=enable_page_ref,
                page_ref_rel_rms_threshold=page_ref_rel_rms_threshold,
                enable_int8_fallback=enable_int8_fallback,
                try_int8_for_protected=try_int8_for_protected,
                int8_rel_rms_threshold=int8_rel_rms_threshold,
                int8_max_abs_threshold=int8_max_abs_threshold,
                prefer_native_fwht=prefer_native_fwht,
                allow_vector_for_protected=allow_vector_for_protected,
            )
        )

    async def run_bound(fn):
        async with semaphore:
            return await run_in_threadpool(fn)

    app = FastAPI(title="HyperQuant", version=bundle.version)
    app.state.bundle = bundle
    app.state.compressor = compressor
    app.state.metrics = metrics
    app.state.max_request_bytes = max_request_bytes
    app.state.max_http_body_bytes = max_http_body_bytes
    app.state.max_concurrency = resolved_max_concurrency
    app.add_middleware(_BodySizeLimitMiddleware, max_http_body_bytes=max_http_body_bytes, metrics=metrics)

    def internal_server_error(_exc: Exception) -> HTTPException:
        return HTTPException(status_code=500, detail="internal server error")

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        native = native_fwht_status(auto_build=False)
        return HealthResponse(
            status="ok",
            version=bundle.version,
            chunk_size=bundle.chunk_size,
            codebook_size=bundle.codebook_size,
            normalize=bundle.normalize,
            max_request_bytes=max_request_bytes,
            max_http_body_bytes=max_http_body_bytes,
            max_concurrency=resolved_max_concurrency,
            native_fwht_available=bool(native["available"]),
            native_fwht_path=native["path"],
            native_fwht_error=native["error"],
            routes=["codebook", "vector", "context", "resident"],
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(metrics.metrics_payload().decode("utf-8"))

    @app.post("/v1/codebook/compress", response_model=CodebookCompressResponse)
    async def codebook_compress_endpoint(request: CodebookCompressRequest) -> CodebookCompressResponse:
        started = time.perf_counter()

        def do_compress():
            array = ndarray_from_b64(request.array_b64, max_bytes=max_request_bytes)
            return compressor.compress(
                array,
                protected_vector_indices=request.protected_vector_indices,
            )

        try:
            envelope, stats = await run_bound(do_compress)
        except ValueError as exc:
            metrics.observe_error("compress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - FastAPI behavior tested through endpoint
            metrics.observe_error("compress", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_compress(stats, latency_seconds=time.perf_counter() - started)
        return CodebookCompressResponse(
            envelope_b64=envelope.to_base64(),
            stats=CodebookCompressionStatsModel(**stats.to_dict()),
        )

    @app.post("/v1/codebook/decompress", response_model=DecompressResponse)
    async def decompress_endpoint(request: DecompressRequest) -> DecompressResponse:
        started = time.perf_counter()

        def do_decompress():
            envelope = CodebookEnvelope.from_base64(request.envelope_b64, max_bytes=max_request_bytes)
            return compressor.decompress(envelope)

        try:
            restored = await run_bound(do_decompress)
        except ValueError as exc:
            metrics.observe_error("decompress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("decompress", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_decompress(latency_seconds=time.perf_counter() - started)
        return DecompressResponse(array_b64=ndarray_to_b64(restored))

    @app.post("/v1/vector/compress", response_model=VectorCompressResponse)
    async def vector_compress_endpoint(request: VectorCompressRequest) -> VectorCompressResponse:
        started = time.perf_counter()

        def do_vector_compress():
            array = ndarray_from_b64(request.array_b64, max_bytes=max_request_bytes)
            vector = get_vector_compressor(request.bits, request.group_size, request.rotation_seed, request.residual_topk, request.prefer_native_fwht)
            return vector.compress(array)

        try:
            envelope, stats = await run_bound(do_vector_compress)
        except ValueError as exc:
            metrics.observe_error("vector_compress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("vector_compress", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_vector_compress(stats, latency_seconds=time.perf_counter() - started)
        return VectorCompressResponse(
            envelope_b64=envelope.to_base64(),
            stats=VectorCompressionStatsModel(**stats.to_dict()),
        )

    @app.post("/v1/vector/decompress", response_model=DecompressResponse)
    async def vector_decompress_endpoint(request: DecompressRequest) -> DecompressResponse:
        started = time.perf_counter()

        def do_vector_decompress():
            envelope = RotatedScalarEnvelope.from_base64(request.envelope_b64, max_bytes=max_request_bytes)
            prefer_native_fwht = server_prefer_native_fwht and envelope.rotation_kind == "structured_fwht"
            vector = get_vector_compressor(
                envelope.bits,
                envelope.group_size,
                envelope.rotation_seed,
                envelope.residual_topk,
                prefer_native_fwht,
            )
            return vector.decompress(envelope)

        try:
            restored = await run_bound(do_vector_decompress)
        except ValueError as exc:
            metrics.observe_error("vector_decompress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("vector_decompress", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_vector_decompress(latency_seconds=time.perf_counter() - started)
        return DecompressResponse(array_b64=ndarray_to_b64(restored))

    @app.post("/v1/resident/plan", response_model=ResidentPlanResponse)
    async def resident_plan_endpoint(request: ResidentPlanRequest) -> ResidentPlanResponse:
        started = time.perf_counter()

        def do_resident_plan():
            array = ndarray_from_b64(request.array_b64, max_bytes=max_request_bytes)
            planner = get_resident_planner(
                request.page_size,
                request.rank,
                request.bits,
                request.group_size,
                request.hot_pages,
                request.rotation_seed,
                request.residual_topk,
                request.prefix_keep_vectors,
                request.suffix_keep_vectors,
                request.low_rank_error_threshold,
                request.ref_round_decimals,
                request.enable_page_ref,
                request.page_ref_rel_rms_threshold,
                request.enable_int8_fallback,
                request.try_int8_for_protected,
                request.int8_rel_rms_threshold,
                request.int8_max_abs_threshold,
                request.prefer_native_fwht,
                request.allow_vector_for_protected,
            )
            return planner.plan(
                array,
                concurrent_sessions=request.concurrent_sessions,
                active_window_tokens=request.active_window_tokens,
                runtime_value_bytes=request.runtime_value_bytes,
                budget_bytes=request.budget_bytes,
                protected_vector_indices=request.protected_vector_indices,
            )

        try:
            plan = await run_bound(do_resident_plan)
        except ValueError as exc:
            metrics.observe_error("resident_plan", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("resident_plan", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_resident_plan(plan, latency_seconds=time.perf_counter() - started)
        return ResidentPlanResponse(plan=plan.to_dict())

    @app.post("/v1/context/compress", response_model=ContextCompressResponse)
    async def context_compress_endpoint(request: ContextCompressRequest) -> ContextCompressResponse:
        started = time.perf_counter()
        context_config = ContextCodecConfig(
            page_size=request.page_size,
            rank=request.rank,
            prefix_keep_vectors=request.prefix_keep_vectors,
            suffix_keep_vectors=request.suffix_keep_vectors,
            low_rank_error_threshold=request.low_rank_error_threshold,
            ref_round_decimals=request.ref_round_decimals,
            enable_page_ref=request.enable_page_ref,
            page_ref_rel_rms_threshold=request.page_ref_rel_rms_threshold,
            enable_int8_fallback=request.enable_int8_fallback,
            try_int8_for_protected=request.try_int8_for_protected,
            int8_rel_rms_threshold=request.int8_rel_rms_threshold,
            int8_max_abs_threshold=request.int8_max_abs_threshold,
        )
        context = get_context_compressor(context_config)

        def do_context_compress():
            array = ndarray_from_b64(request.array_b64, max_bytes=max_request_bytes)
            guarantee_profile = None
            if request.guarantee is not None:
                guarantee_profile = ContextGuaranteeProfile(**_pydantic_model_to_dict(request.guarantee))
            return context.compress(
                array,
                protected_vector_indices=request.protected_vector_indices,
                guarantee_profile=guarantee_profile,
                guarantee_mode=GuaranteeMode.FAIL_CLOSED if request.fail_closed else GuaranteeMode.ALLOW_BEST_EFFORT,
            )

        try:
            envelope, stats = await run_bound(do_context_compress)
        except ContourViolation as exc:
            metrics.observe_error("context_compress", "contour_violation")
            raise HTTPException(
                status_code=422,
                detail={"type": "contour_violation", "failures": list(exc.failures)},
            ) from exc
        except GuaranteeViolation as exc:
            metrics.observe_error("context_compress", "guarantee_violation")
            raise HTTPException(
                status_code=422,
                detail={"type": "guarantee_violation", "failures": list(exc.failures)},
            ) from exc
        except ValueError as exc:
            metrics.observe_error("context_compress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("context_compress", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_context_compress(stats, latency_seconds=time.perf_counter() - started)
        return ContextCompressResponse(
            envelope_b64=envelope.to_base64(),
            stats=ContextCompressionStatsModel(**stats.to_dict()),
        )

    @app.post("/v1/context/decompress", response_model=DecompressResponse)
    async def context_decompress_endpoint(request: DecompressRequest) -> DecompressResponse:
        started = time.perf_counter()

        def do_context_decompress():
            envelope = ContextEnvelope.from_base64(request.envelope_b64, max_bytes=max_request_bytes)
            context = get_context_compressor(
                ContextCodecConfig(page_size=envelope.page_size, rank=envelope.rank)
            )
            return context.decompress(envelope)

        try:
            restored = await run_bound(do_context_decompress)
        except ValueError as exc:
            metrics.observe_error("context_decompress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("context_decompress", "internal_error")
            raise internal_server_error(exc) from exc
        metrics.observe_context_decompress(latency_seconds=time.perf_counter() - started)
        return DecompressResponse(array_b64=ndarray_to_b64(restored))

    return app
