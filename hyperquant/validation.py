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
from typing import Iterable

import numpy as np

SAFE_FLOAT_DTYPES = {
    "float16",
    "float32",
    "float64",
}


@dataclass(frozen=True)
class ShapeLimits:
    max_ndim: int = 8
    max_elements: int = 128 * 1024 * 1024
    max_last_dim: int = 32768


def validate_float_dtype(dtype: str | np.dtype) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved.name not in SAFE_FLOAT_DTYPES:
        allowed = ", ".join(sorted(SAFE_FLOAT_DTYPES))
        raise ValueError(f"unsupported dtype {resolved.name!r}; supported dtypes: {allowed}")
    return resolved


def validate_shape(shape: Iterable[int], *, min_ndim: int = 2, limits: ShapeLimits | None = None) -> tuple[int, ...]:
    checked = tuple(int(v) for v in shape)
    bounds = limits or ShapeLimits()
    if len(checked) < min_ndim:
        raise ValueError(f"expected at least {min_ndim} dimensions")
    if len(checked) > bounds.max_ndim:
        raise ValueError(f"shape exceeds max_ndim={bounds.max_ndim}")
    if any(v <= 0 for v in checked):
        raise ValueError("shape dimensions must be > 0")
    if checked[-1] > bounds.max_last_dim:
        raise ValueError(f"last dimension exceeds max_last_dim={bounds.max_last_dim}")
    elements = int(np.prod(checked))
    if elements > bounds.max_elements:
        raise ValueError(f"array exceeds max_elements={bounds.max_elements}")
    return checked


def validate_numeric_finite_array(
    array: np.ndarray,
    *,
    limits: ShapeLimits | None = None,
    min_ndim: int = 2,
) -> np.ndarray:
    values = np.asarray(array)
    validate_float_dtype(values.dtype)
    validate_shape(values.shape, min_ndim=min_ndim, limits=limits)
    if not np.isfinite(values).all():
        raise ValueError("array contains NaN or Inf")
    return values
