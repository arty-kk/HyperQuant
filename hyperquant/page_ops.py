# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .utils import EPS


def hash_page(page: np.ndarray, valid_length: int, decimals: int) -> bytes:
    rounded = np.round(page[:valid_length], decimals=decimals).astype(np.float16, copy=False)
    digest = hashlib.blake2b(rounded.tobytes(), digest_size=16)
    digest.update(np.asarray(valid_length, dtype=np.int32).tobytes())
    return digest.digest()


def relative_rms(reference: np.ndarray, candidate: np.ndarray) -> float:
    diff = reference - candidate
    return float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(reference * reference)) + EPS))


def max_abs_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(reference - candidate)))


def top_rank_factors(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    if rank <= 0:
        raise ValueError("rank must be > 0")
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float32), np.empty((0, matrix.shape[1]), dtype=np.float32)

    if rank == 1:
        v = matrix[0].astype(np.float32, copy=True)
        norm = float(np.linalg.norm(v))
        if norm <= EPS:
            v = np.ones((matrix.shape[1],), dtype=np.float32)
            norm = float(np.linalg.norm(v))
        v /= max(norm, EPS)
        for _ in range(4):
            u = matrix @ v
            u_norm = float(np.linalg.norm(u))
            if u_norm <= EPS:
                break
            u /= u_norm
            v = matrix.T @ u
            v_norm = float(np.linalg.norm(v))
            if v_norm <= EPS:
                break
            v /= v_norm
        u = matrix @ v
        sigma = float(np.linalg.norm(u))
        if sigma <= EPS:
            us = np.zeros((matrix.shape[0], 1), dtype=np.float32)
            vt = np.zeros((1, matrix.shape[1]), dtype=np.float32)
            return us, vt
        us = (u / sigma)[:, None] * sigma
        vt = v[None, :]
        return us.astype(np.float32), vt.astype(np.float32)

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    take = min(rank, s.shape[0])
    us = u[:, :take].astype(np.float32, copy=False) * s[:take].astype(np.float32, copy=False)[None, :]
    return us.astype(np.float32), vt[:take, :].astype(np.float32, copy=False)


def quantize_page_int8(
    page: np.ndarray,
    valid_length: int,
    page_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    core = page[:valid_length]
    mins = core.min(axis=0)
    maxs = core.max(axis=0)
    scales = (maxs - mins) / 255.0
    scales = np.where(scales < EPS, 1.0, scales)

    mins_fp16 = mins.astype(np.float16)
    scales_fp16 = scales.astype(np.float16)
    mins_f32 = mins_fp16.astype(np.float32)[None, :]
    scales_f32 = scales_fp16.astype(np.float32)[None, :]
    q_core = np.round((core - mins_f32) / scales_f32).clip(0, 255).astype(np.uint8)
    recon_core = q_core.astype(np.float32) * scales_f32 + mins_f32

    q_page = np.zeros((page_size, page.shape[1]), dtype=np.uint8)
    q_page[:valid_length] = q_core
    recon_page = np.zeros_like(page, dtype=np.float32)
    recon_page[:valid_length] = recon_core
    rel_rms = relative_rms(core, recon_core)
    max_abs = max_abs_error(core, recon_core)
    return mins_fp16, scales_fp16, q_page, recon_page, rel_rms, max_abs


def protected_mask(
    n_vectors: int,
    protected_vector_indices: Sequence[int] | None,
    *,
    prefix_keep_vectors: int,
    suffix_keep_vectors: int,
) -> np.ndarray:
    mask = np.zeros(n_vectors, dtype=bool)
    if prefix_keep_vectors > 0:
        mask[: min(prefix_keep_vectors, n_vectors)] = True
    if suffix_keep_vectors > 0:
        mask[max(0, n_vectors - suffix_keep_vectors) :] = True
    for idx in protected_vector_indices or ():
        if idx < 0 or idx >= n_vectors:
            raise ValueError(f"protected vector index {idx} outside valid range [0, {n_vectors})")
        mask[idx] = True
    return mask
