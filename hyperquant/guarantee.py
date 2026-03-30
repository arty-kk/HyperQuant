# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class GuaranteeMode(str, Enum):
    FAIL_CLOSED = "fail_closed"
    ALLOW_BEST_EFFORT = "allow_best_effort"


@dataclass(frozen=True)
class ContextGuaranteeProfile:
    min_compression_ratio: float = 30.0
    min_cosine_similarity: float = 0.999
    max_rms_error: float = 0.010
    max_max_abs_error: float | None = 0.050

    def validate(self) -> None:
        if self.min_compression_ratio <= 0:
            raise ValueError("min_compression_ratio must be > 0")
        if not (0.0 <= self.min_cosine_similarity <= 1.0):
            raise ValueError("min_cosine_similarity must be in [0, 1]")
        if self.max_rms_error < 0:
            raise ValueError("max_rms_error must be >= 0")
        if self.max_max_abs_error is not None and self.max_max_abs_error < 0:
            raise ValueError("max_max_abs_error must be >= 0 when provided")

    def to_dict(self) -> dict[str, float | None]:
        return {
            "min_compression_ratio": self.min_compression_ratio,
            "min_cosine_similarity": self.min_cosine_similarity,
            "max_rms_error": self.max_rms_error,
            "max_max_abs_error": self.max_max_abs_error,
        }


@dataclass(frozen=True)
class GuaranteeOutcome:
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if self.passed:
            return "guarantee passed"
        return "; ".join(self.failures) if self.failures else "guarantee failed"


class GuaranteeViolation(RuntimeError):
    def __init__(self, failures: Iterable[str]) -> None:
        self.failures = tuple(str(item) for item in failures)
        message = "; ".join(self.failures) if self.failures else "guarantee failed"
        super().__init__(message)


class ContourViolation(GuaranteeViolation):
    pass


def evaluate_context_stats(
    *,
    compression_ratio: float,
    cosine_similarity: float,
    rms_error: float,
    max_abs_error: float,
    profile: ContextGuaranteeProfile,
) -> GuaranteeOutcome:
    profile.validate()
    failures: list[str] = []

    if compression_ratio < profile.min_compression_ratio:
        failures.append(
            f"compression_ratio={compression_ratio:.6f} < min_compression_ratio={profile.min_compression_ratio:.6f}"
        )
    if cosine_similarity < profile.min_cosine_similarity:
        failures.append(
            f"cosine_similarity={cosine_similarity:.6f} < min_cosine_similarity={profile.min_cosine_similarity:.6f}"
        )
    if rms_error > profile.max_rms_error:
        failures.append(
            f"rms_error={rms_error:.6f} > max_rms_error={profile.max_rms_error:.6f}"
        )
    if profile.max_max_abs_error is not None and max_abs_error > profile.max_max_abs_error:
        failures.append(
            f"max_abs_error={max_abs_error:.6f} > max_max_abs_error={profile.max_max_abs_error:.6f}"
        )

    return GuaranteeOutcome(passed=not failures, failures=tuple(failures))
