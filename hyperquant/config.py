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

from dataclasses import dataclass
from enum import IntEnum


class CompressionMode(IntEnum):
    CODEBOOK_ONLY = 0
    SIGN_RESIDUAL = 1
    INT8_RESIDUAL = 2
    FP16_FALLBACK = 3


@dataclass(frozen=True)
class CodebookConfig:
    chunk_size: int = 32
    codebook_size: int = 256
    rotation_seed: int = 17
    normalize: bool = True
    sample_size: int = 20_000
    training_iterations: int = 15

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be > 1")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be > 0")
        if self.training_iterations <= 0:
            raise ValueError("training_iterations must be > 0")


@dataclass(frozen=True)
class CompressionConfig:
    codebook_error_threshold: float = 0.030
    sign_error_threshold: float = 0.080
    int8_error_threshold: float = 0.200
    protected_mode: CompressionMode = CompressionMode.FP16_FALLBACK

    def validate(self) -> None:
        thresholds = (
            self.codebook_error_threshold,
            self.sign_error_threshold,
            self.int8_error_threshold,
        )
        if any(value < 0 for value in thresholds):
            raise ValueError("error thresholds must be >= 0")
        if not (
            self.codebook_error_threshold
            <= self.sign_error_threshold
            <= self.int8_error_threshold
        ):
            raise ValueError(
                "thresholds must satisfy codebook <= sign <= int8"
            )
