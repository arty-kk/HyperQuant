# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np

from .codebook_codec import CodebookStats, CodebookCodec
from .guarantee import GuaranteeMode, ContextGuaranteeProfile
from .context_codec import ContextCompressionStats, ContextCodec


T = TypeVar("T")


@dataclass(frozen=True)
class TimingStats:
    iterations: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "stdev_ms": self.stdev_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    timing: TimingStats
    stats: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "timing": self.timing.to_dict(),
            "stats": self.stats,
        }


def benchmark_array(compressor: CodebookCodec, array: np.ndarray) -> CodebookStats:
    _, stats = compressor.compress(array)
    return stats


def benchmark_context_array(
    compressor: ContextCodec,
    array: np.ndarray,
    *,
    guarantee_profile: ContextGuaranteeProfile | None = None,
    guarantee_mode: GuaranteeMode = GuaranteeMode.ALLOW_BEST_EFFORT,
) -> ContextCompressionStats:
    _, stats = compressor.compress(
        array,
        guarantee_profile=guarantee_profile,
        guarantee_mode=guarantee_mode,
    )
    return stats


def time_callable(fn: Callable[[], T], *, iterations: int = 10, warmup: int = 1) -> tuple[T, TimingStats]:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    result: T | None = None
    for _ in range(warmup):
        result = fn()

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = fn()
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    assert result is not None
    ordered = sorted(samples_ms)
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    timing = TimingStats(
        iterations=iterations,
        warmup=warmup,
        mean_ms=float(statistics.mean(samples_ms)),
        median_ms=float(statistics.median(samples_ms)),
        p95_ms=float(ordered[p95_index]),
        stdev_ms=float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
        min_ms=float(min(samples_ms)),
        max_ms=float(max(samples_ms)),
    )
    return result, timing


def stats_to_pretty_json(stats: CodebookStats | ContextCompressionStats) -> str:
    return json.dumps(stats.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)


def report_to_pretty_json(report: BenchmarkReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
