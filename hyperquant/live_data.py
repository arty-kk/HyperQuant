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

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiveDataProfile:
    seed: int = 20260329
    page_size: int = 64
    dim: int = 128



def generate_online_vector_stream(
    n_vectors: int = 16384,
    dim: int = 128,
    *,
    seed: int = 20260329,
    heavy_tail_fraction: float = 0.08,
) -> np.ndarray:
    """
    Live-like streaming vectors for online service benchmarking.

    The stream mixes Gaussian traffic with bursty heavy-tail events to mimic
    embeddings, KV rows, and retrieval vectors under real serving load.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    low_rank_basis = rng.standard_normal((16, dim)).astype(np.float32)
    coeffs = rng.standard_normal((n_vectors, 16)).astype(np.float32)
    stream = 0.65 * base + 0.35 * (coeffs @ low_rank_basis) / math.sqrt(16)

    burst_count = max(1, int(dim * heavy_tail_fraction))
    burst_channels = rng.choice(dim, size=burst_count, replace=False)
    burst_mask = rng.random((n_vectors, 1), dtype=np.float32) < 0.18
    burst_noise = rng.standard_t(df=3, size=(n_vectors, burst_count)).astype(np.float32) * 1.35
    stream[:, burst_channels] += burst_mask.astype(np.float32) * burst_noise
    return stream.astype(np.float32)



def generate_structured_long_context(
    n_tokens: int = 4096,
    dim: int = 128,
    page_size: int = 64,
    *,
    seed: int = 20260329,
    policy_pages: int = 6,
    replay_pages: int = 14,
    recent_tokens: int = 96,
) -> np.ndarray:
    """
    Structured long-context stream with:
    - repeated policy / retrieval pages,
    - strongly low-rank topical middle,
    - a small noisy recent tail.
    """
    rng = np.random.default_rng(seed)
    n_pages = math.ceil(n_tokens / page_size)
    pages: list[np.ndarray] = []
    templates: list[np.ndarray] = []

    for _ in range(policy_pages):
        topic = rng.standard_normal((1, dim)).astype(np.float32)
        coeff = rng.standard_normal((page_size, 1)).astype(np.float32) * 0.65
        basis = rng.standard_normal((1, dim)).astype(np.float32)
        page = topic + coeff @ basis + 0.0025 * rng.standard_normal((page_size, dim)).astype(np.float32)
        templates.append(page.astype(np.float32))

    recent_pages = math.ceil(recent_tokens / page_size)
    for page_idx in range(n_pages):
        if page_idx < policy_pages:
            pages.append(templates[page_idx])
            continue
        if page_idx < policy_pages + replay_pages:
            pages.append(templates[page_idx % policy_pages].copy())
            continue
        if page_idx >= n_pages - recent_pages:
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 3)).astype(np.float32)
            basis = rng.standard_normal((3, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.028 * rng.standard_normal((page_size, dim)).astype(np.float32)
            pages.append(page.astype(np.float32))
            continue
        topic = rng.standard_normal((1, dim)).astype(np.float32)
        coeff = rng.standard_normal((page_size, 1)).astype(np.float32)
        basis = rng.standard_normal((1, dim)).astype(np.float32)
        page = topic + coeff @ basis + 0.007 * rng.standard_normal((page_size, dim)).astype(np.float32)
        pages.append(page.astype(np.float32))

    return np.concatenate(pages, axis=0)[:n_tokens].astype(np.float32)



def generate_mixed_long_context(
    n_tokens: int = 8192,
    dim: int = 128,
    page_size: int = 64,
    *,
    seed: int = 20260329,
) -> np.ndarray:
    """
    Harder live-like long-context stream with a blend of repeated pages,
    low-rank topical segments, bursty outliers, and moderately complex tails.
    Useful for stress testing route selection when the structured context contour is not met.
    """
    rng = np.random.default_rng(seed)
    n_pages = math.ceil(n_tokens / page_size)
    templates: list[np.ndarray] = []
    for _ in range(6):
        topic = rng.standard_normal((1, dim)).astype(np.float32)
        coeff = rng.standard_normal((page_size, 2)).astype(np.float32)
        basis = rng.standard_normal((2, dim)).astype(np.float32)
        page = topic + coeff @ basis + 0.01 * rng.standard_normal((page_size, dim)).astype(np.float32)
        templates.append(page.astype(np.float32))

    pages: list[np.ndarray] = []
    for page_idx in range(n_pages):
        if page_idx < 4:
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 4)).astype(np.float32)
            basis = rng.standard_normal((4, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.03 * rng.standard_normal((page_size, dim)).astype(np.float32)
        elif page_idx % 9 == 0:
            page = templates[page_idx % len(templates)].copy()
        elif page_idx % 7 in (1, 2, 3):
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 1)).astype(np.float32)
            basis = rng.standard_normal((1, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.006 * rng.standard_normal((page_size, dim)).astype(np.float32)
        else:
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 3)).astype(np.float32)
            basis = rng.standard_normal((3, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.02 * rng.standard_normal((page_size, dim)).astype(np.float32)
        if page_idx % 5 == 0:
            outlier_idx = rng.choice(dim, size=8, replace=False)
            page[:, outlier_idx] += rng.standard_t(df=3, size=(page_size, 8)).astype(np.float32) * 2.3
        pages.append(page.astype(np.float32))

    return np.concatenate(pages, axis=0)[:n_tokens].astype(np.float32)


__all__ = [
    "LiveDataProfile",
    "generate_online_vector_stream",
    "generate_structured_long_context",
    "generate_mixed_long_context",
]
