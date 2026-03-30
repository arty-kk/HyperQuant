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

import numpy as np


def build_context_like_array(n_tokens: int = 4096, dim: int = 128, page_size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(123)
    n_pages = n_tokens // page_size
    pages: list[np.ndarray] = []
    templates: list[np.ndarray] = []

    for _ in range(4):
        topic = rng.standard_normal((1, dim)).astype(np.float32)
        coeff = rng.standard_normal((page_size, 1)).astype(np.float32) * 0.7
        basis = rng.standard_normal((1, dim)).astype(np.float32)
        page = topic + coeff @ basis + 0.002 * rng.standard_normal((page_size, dim)).astype(np.float32)
        templates.append(page.astype(np.float32))

    for page_idx in range(n_pages):
        if page_idx < 4:
            pages.append(templates[page_idx])
        elif page_idx < 16:
            pages.append(templates[page_idx % 4].copy())
        elif page_idx >= n_pages - 1:
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 3)).astype(np.float32)
            basis = rng.standard_normal((3, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.03 * rng.standard_normal((page_size, dim)).astype(np.float32)
            pages.append(page.astype(np.float32))
        else:
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 1)).astype(np.float32)
            basis = rng.standard_normal((1, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.008 * rng.standard_normal((page_size, dim)).astype(np.float32)
            pages.append(page.astype(np.float32))

    return np.concatenate(pages, axis=0).astype(np.float32)


def build_random_array(n_tokens: int = 4096, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(999)
    return rng.standard_normal((n_tokens, dim)).astype(np.float32)
