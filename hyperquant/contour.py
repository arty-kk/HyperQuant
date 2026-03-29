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

from dataclasses import dataclass, field
from enum import Enum


class ProductContour(str, Enum):
    CONTEXT_STRUCTURED = "context_structured"
    GENERIC_CONSERVATIVE = "generic_conservative"
    REJECT = "reject"


@dataclass(frozen=True)
class ContourThresholds:
    min_pages: int = 16
    max_protected_fraction: float = 0.20
    min_structural_fraction: float = 0.70
    min_low_rank_or_ref_pages: int = 8

    def validate(self) -> None:
        if self.min_pages <= 0:
            raise ValueError("min_pages must be > 0")
        if not (0.0 <= self.max_protected_fraction <= 1.0):
            raise ValueError("max_protected_fraction must be in [0, 1]")
        if not (0.0 <= self.min_structural_fraction <= 1.0):
            raise ValueError("min_structural_fraction must be in [0, 1]")
        if self.min_low_rank_or_ref_pages < 0:
            raise ValueError("min_low_rank_or_ref_pages must be >= 0")


@dataclass(frozen=True)
class ContourAnalysis:
    contour: ProductContour
    supported: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    total_pages: int = 0
    protected_fraction: float = 0.0
    structural_fraction: float = 0.0
    low_rank_fraction: float = 0.0
    page_ref_fraction: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "contour": self.contour.value,
            "supported": self.supported,
            "reasons": list(self.reasons),
            "total_pages": self.total_pages,
            "protected_fraction": self.protected_fraction,
            "structural_fraction": self.structural_fraction,
            "low_rank_fraction": self.low_rank_fraction,
            "page_ref_fraction": self.page_ref_fraction,
        }


def analyze_context_contour(
    *,
    total_pages: int,
    protected_pages: int,
    low_rank_pages: int,
    page_ref_pages: int,
    thresholds: ContourThresholds,
) -> ContourAnalysis:
    thresholds.validate()
    if total_pages <= 0:
        return ContourAnalysis(
            contour=ProductContour.REJECT,
            supported=False,
            reasons=("total_pages must be > 0",),
            total_pages=total_pages,
        )

    protected_fraction = protected_pages / total_pages
    low_rank_fraction = low_rank_pages / total_pages
    page_ref_fraction = page_ref_pages / total_pages
    structural_fraction = (low_rank_pages + page_ref_pages) / total_pages

    reasons: list[str] = []
    contour = ProductContour.CONTEXT_STRUCTURED

    if total_pages < thresholds.min_pages:
        contour = ProductContour.GENERIC_CONSERVATIVE
        reasons.append(f"total_pages={total_pages} < min_pages={thresholds.min_pages}")
    if protected_fraction > thresholds.max_protected_fraction:
        contour = ProductContour.GENERIC_CONSERVATIVE
        reasons.append(
            f"protected_fraction={protected_fraction:.6f} > max_protected_fraction={thresholds.max_protected_fraction:.6f}"
        )
    if structural_fraction < thresholds.min_structural_fraction:
        contour = ProductContour.GENERIC_CONSERVATIVE
        reasons.append(
            f"structural_fraction={structural_fraction:.6f} < min_structural_fraction={thresholds.min_structural_fraction:.6f}"
        )
    if (low_rank_pages + page_ref_pages) < thresholds.min_low_rank_or_ref_pages:
        contour = ProductContour.GENERIC_CONSERVATIVE
        reasons.append(
            f"low_rank_or_ref_pages={low_rank_pages + page_ref_pages} < min_low_rank_or_ref_pages={thresholds.min_low_rank_or_ref_pages}"
        )

    return ContourAnalysis(
        contour=contour,
        supported=contour == ProductContour.CONTEXT_STRUCTURED,
        reasons=tuple(reasons),
        total_pages=total_pages,
        protected_fraction=protected_fraction,
        structural_fraction=structural_fraction,
        low_rank_fraction=low_rank_fraction,
        page_ref_fraction=page_ref_fraction,
    )
