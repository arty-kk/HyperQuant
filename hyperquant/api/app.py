# Copyright 2026 Сацук Артём Венедиктович (Satsuk Artem)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    @app.middleware("http")
    async def enforce_content_length(request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is not None and size > max_http_body_bytes:
                metrics.observe_error("http", "request_too_large")
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"request body exceeds max_http_body_bytes={max_http_body_bytes}"},
                )
        return await call_next(request)

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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            vector = get_vector_compressor(envelope.bits, envelope.group_size, envelope.rotation_seed, envelope.residual_topk, True)
            return vector.decompress(envelope)

        try:
            restored = await run_bound(do_vector_decompress)
        except ValueError as exc:
            metrics.observe_error("vector_decompress", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("vector_decompress", "internal_error")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            )

        try:
            plan = await run_bound(do_resident_plan)
        except ValueError as exc:
            metrics.observe_error("resident_plan", "bad_request")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            metrics.observe_error("resident_plan", "internal_error")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metrics.observe_context_decompress(latency_seconds=time.perf_counter() - started)
        return DecompressResponse(array_b64=ndarray_to_b64(restored))

    return app
