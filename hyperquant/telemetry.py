# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .codebook_codec import CodebookStats
from .context_codec import ContextCompressionStats
from .resident_tier import ResidentPlan
from .vector_codec import RotatedScalarStats


class HyperQuantMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests_total = Counter(
            "hyperquant_requests_total",
            "Total number of compress/decompress API requests.",
            ["endpoint"],
            registry=self.registry,
        )
        self.errors_total = Counter(
            "hyperquant_errors_total",
            "Total number of rejected or failed API requests.",
            ["endpoint", "reason"],
            registry=self.registry,
        )
        self.original_bytes = Counter(
            "hyperquant_original_bytes_total",
            "Total original bytes processed by compression requests.",
            registry=self.registry,
        )
        self.stored_bytes = Counter(
            "hyperquant_stored_bytes_total",
            "Total in-memory bytes produced by compression requests.",
            registry=self.registry,
        )
        self.serialized_bytes = Counter(
            "hyperquant_serialized_bytes_total",
            "Total serialized payload bytes produced by compression requests.",
            registry=self.registry,
        )
        self.mode_chunks = Counter(
            "hyperquant_mode_chunks_total",
            "How many chunks were emitted in each compression mode.",
            ["mode"],
            registry=self.registry,
        )
        self.context_pages = Counter(
            "hyperquant_context_pages_total",
            "How many pages were emitted in each Context mode.",
            ["mode"],
            registry=self.registry,
        )
        self.guarantee_total = Counter(
            "hyperquant_guarantee_total",
            "Number of guarantee evaluations by outcome.",
            ["outcome"],
            registry=self.registry,
        )
        self.contour_total = Counter(
            "hyperquant_contour_total",
            "Routing contour outcomes for Context requests.",
            ["contour", "route"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "hyperquant_request_latency_seconds",
            "End-to-end request latency by endpoint.",
            ["endpoint"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry,
        )
        self.last_ratio = Gauge(
            "hyperquant_last_compression_ratio",
            "Effective serialized compression ratio of the latest compression request.",
            registry=self.registry,
        )
        self.last_storage_ratio = Gauge(
            "hyperquant_last_storage_compression_ratio",
            "In-memory storage compression ratio of the latest compression request.",
            registry=self.registry,
        )
        self.last_rms_error = Gauge(
            "hyperquant_last_rms_error",
            "RMS reconstruction error of the latest compression request.",
            registry=self.registry,
        )
        self.last_projected_resident_ratio = Gauge(
            "hyperquant_last_projected_resident_ratio",
            "Projected resident-memory ratio of the latest memory planning request versus baseline.",
            registry=self.registry,
        )

    def _observe_common_compress(self, *, endpoint: str, original_bytes: int, stored_bytes: int, serialized_bytes: int, compression_ratio: float, storage_compression_ratio: float, rms_error: float, latency_seconds: float | None = None) -> None:
        self.requests_total.labels(endpoint=endpoint).inc()
        if latency_seconds is not None:
            self.request_latency.labels(endpoint=endpoint).observe(latency_seconds)
        self.original_bytes.inc(original_bytes)
        self.stored_bytes.inc(stored_bytes)
        self.serialized_bytes.inc(serialized_bytes)
        self.last_ratio.set(compression_ratio)
        self.last_storage_ratio.set(storage_compression_ratio)
        self.last_rms_error.set(rms_error)

    def observe_compress(self, stats: CodebookStats, *, latency_seconds: float | None = None) -> None:
        self._observe_common_compress(
            endpoint="compress",
            original_bytes=stats.original_bytes,
            stored_bytes=stats.stored_bytes,
            serialized_bytes=stats.serialized_bytes,
            compression_ratio=stats.compression_ratio,
            storage_compression_ratio=stats.storage_compression_ratio,
            rms_error=stats.rms_error,
            latency_seconds=latency_seconds,
        )
        for mode, count in stats.mode_counts.items():
            self.mode_chunks.labels(mode=mode).inc(count)

    def observe_vector_compress(self, stats: RotatedScalarStats, *, latency_seconds: float | None = None) -> None:
        self._observe_common_compress(
            endpoint="vector_compress",
            original_bytes=stats.original_bytes,
            stored_bytes=stats.stored_bytes,
            serialized_bytes=stats.serialized_bytes,
            compression_ratio=stats.compression_ratio,
            storage_compression_ratio=stats.storage_compression_ratio,
            rms_error=stats.rms_error,
            latency_seconds=latency_seconds,
        )

    def observe_context_compress(self, stats: ContextCompressionStats, *, latency_seconds: float | None = None) -> None:
        self._observe_common_compress(
            endpoint="context_compress",
            original_bytes=stats.original_bytes,
            stored_bytes=stats.stored_bytes,
            serialized_bytes=stats.serialized_bytes,
            compression_ratio=stats.compression_ratio,
            storage_compression_ratio=stats.storage_compression_ratio,
            rms_error=stats.rms_error,
            latency_seconds=latency_seconds,
        )
        for mode, count in stats.page_mode_counts.items():
            self.context_pages.labels(mode=mode).inc(count)
        self.contour_total.labels(contour=stats.contour, route=stats.route_recommendation).inc()
        if stats.guarantee_passed is True:
            self.guarantee_total.labels(outcome="passed").inc()
        elif stats.guarantee_passed is False:
            self.guarantee_total.labels(outcome="failed").inc()

    def observe_decompress(self, *, latency_seconds: float | None = None) -> None:
        self.requests_total.labels(endpoint="decompress").inc()
        if latency_seconds is not None:
            self.request_latency.labels(endpoint="decompress").observe(latency_seconds)

    def observe_vector_decompress(self, *, latency_seconds: float | None = None) -> None:
        self.requests_total.labels(endpoint="vector_decompress").inc()
        if latency_seconds is not None:
            self.request_latency.labels(endpoint="vector_decompress").observe(latency_seconds)

    def observe_context_decompress(self, *, latency_seconds: float | None = None) -> None:
        self.requests_total.labels(endpoint="context_decompress").inc()
        if latency_seconds is not None:
            self.request_latency.labels(endpoint="context_decompress").observe(latency_seconds)

    def observe_resident_plan(self, plan: ResidentPlan, *, latency_seconds: float | None = None) -> None:
        self.requests_total.labels(endpoint="resident_plan").inc()
        if latency_seconds is not None:
            self.request_latency.labels(endpoint="resident_plan").observe(latency_seconds)
        self.last_projected_resident_ratio.set(
            float(plan.projected_resident_bytes_per_session / max(plan.baseline_resident_bytes_per_session, 1))
        )

    def observe_error(self, endpoint: str, reason: str) -> None:
        self.errors_total.labels(endpoint=endpoint, reason=reason).inc()

    def metrics_payload(self) -> bytes:
        return generate_latest(self.registry)
