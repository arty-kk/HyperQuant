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

import hashlib
import io
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np

from .contour import ContourAnalysis, ContourThresholds, ProductContour, analyze_context_contour
from .defaults import (
    CONTEXT_ENABLE_INT8_FALLBACK_DEFAULT,
    CONTEXT_ENABLE_PAGE_REF_DEFAULT,
    CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT,
    CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT,
    CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT,
    CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT,
    CONTEXT_PAGE_SIZE_DEFAULT,
    CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT,
    CONTEXT_RANK_DEFAULT,
    CONTEXT_REF_ROUND_DECIMALS_DEFAULT,
    CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT,
    CONTEXT_TRY_INT8_FOR_PROTECTED_DEFAULT,
)
from .guarantee import (
    ContourViolation,
    GuaranteeMode,
    GuaranteeViolation,
    ContextGuaranteeProfile,
    evaluate_context_stats,
)
from .utils import EPS, bytes_from_b64, bytes_to_b64, sha256_hex
from .validation import ShapeLimits, validate_float_dtype, validate_numeric_finite_array, validate_shape


class ContextPageMode(IntEnum):
    PAGE_REF = 0
    LOW_RANK = 1
    FP16 = 2
    INT8 = 3


@dataclass(frozen=True)
class ContextCodecConfig:
    page_size: int = CONTEXT_PAGE_SIZE_DEFAULT
    rank: int = CONTEXT_RANK_DEFAULT
    prefix_keep_vectors: int = CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT
    suffix_keep_vectors: int = CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT
    low_rank_error_threshold: float = CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT
    ref_round_decimals: int = CONTEXT_REF_ROUND_DECIMALS_DEFAULT
    enable_page_ref: bool = CONTEXT_ENABLE_PAGE_REF_DEFAULT
    page_ref_rel_rms_threshold: float = CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT
    enable_int8_fallback: bool = CONTEXT_ENABLE_INT8_FALLBACK_DEFAULT
    try_int8_for_protected: bool = CONTEXT_TRY_INT8_FOR_PROTECTED_DEFAULT
    int8_rel_rms_threshold: float = CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT
    int8_max_abs_threshold: float = CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT

    def validate(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be > 0")
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.prefix_keep_vectors < 0 or self.suffix_keep_vectors < 0:
            raise ValueError("prefix_keep_vectors and suffix_keep_vectors must be >= 0")
        if self.low_rank_error_threshold < 0:
            raise ValueError("low_rank_error_threshold must be >= 0")
        if self.ref_round_decimals < 0:
            raise ValueError("ref_round_decimals must be >= 0")
        if self.page_ref_rel_rms_threshold < 0:
            raise ValueError("page_ref_rel_rms_threshold must be >= 0")
        if self.int8_rel_rms_threshold < 0:
            raise ValueError("int8_rel_rms_threshold must be >= 0")
        if self.int8_max_abs_threshold < 0:
            raise ValueError("int8_max_abs_threshold must be >= 0")


@dataclass
class ContextCompressionStats:
    original_bytes: int
    stored_bytes: int
    serialized_bytes: int
    compression_ratio: float
    storage_compression_ratio: float
    rms_error: float
    max_abs_error: float
    cosine_similarity: float
    page_mode_counts: dict[str, int]
    protected_vectors: int
    protected_pages: int
    payload_sha256: str = ""
    guarantee_passed: bool | None = None
    guarantee_failures: list[str] = field(default_factory=list)
    guarantee_profile: dict[str, object] | None = None
    contour: str = ProductContour.REJECT.value
    contour_supported: bool | None = None
    contour_reasons: list[str] = field(default_factory=list)
    contour_details: dict[str, object] | None = None
    route_recommendation: str = "reject"

    def to_dict(self) -> dict[str, object]:
        return {
            "original_bytes": self.original_bytes,
            "stored_bytes": self.stored_bytes,
            "serialized_bytes": self.serialized_bytes,
            "compression_ratio": self.compression_ratio,
            "storage_compression_ratio": self.storage_compression_ratio,
            "rms_error": self.rms_error,
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.cosine_similarity,
            "page_mode_counts": self.page_mode_counts,
            "protected_vectors": self.protected_vectors,
            "protected_pages": self.protected_pages,
            "payload_sha256": self.payload_sha256,
            "guarantee_passed": self.guarantee_passed,
            "guarantee_failures": list(self.guarantee_failures),
            "guarantee_profile": self.guarantee_profile,
            "contour": self.contour,
            "contour_supported": self.contour_supported,
            "contour_reasons": list(self.contour_reasons),
            "contour_details": self.contour_details,
            "route_recommendation": self.route_recommendation,
        }


@dataclass
class ContextEnvelope:
    original_shape: tuple[int, ...]
    original_dtype: str
    page_size: int
    rank: int
    page_modes: np.ndarray
    page_lengths: np.ndarray
    page_ref_indices: np.ndarray
    low_rank_means: np.ndarray
    low_rank_us: np.ndarray
    low_rank_vt: np.ndarray
    int8_mins: np.ndarray
    int8_scales: np.ndarray
    int8_pages: np.ndarray
    fp16_pages: np.ndarray
    schema_version: str = "context-envelope.v4"
    _bytes_cache: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _base64_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def validate(self) -> None:
        validate_shape(self.original_shape)
        validate_float_dtype(self.original_dtype)
        if self.page_size <= 0:
            raise ValueError("page_size must be > 0")
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if self.page_modes.ndim != 1 or self.page_lengths.ndim != 1 or self.page_ref_indices.ndim != 1:
            raise ValueError("page metadata arrays must be 1D")
        n_pages = int(self.page_modes.shape[0])
        if self.page_lengths.shape[0] != n_pages or self.page_ref_indices.shape[0] != n_pages:
            raise ValueError("page metadata array lengths must match")
        if n_pages == 0:
            raise ValueError("payload must contain at least one page")
        if np.any(self.page_lengths <= 0) or np.any(self.page_lengths > self.page_size):
            raise ValueError("page_lengths must be in [1, page_size]")
        allowed = {int(mode) for mode in ContextPageMode}
        if any(int(v) not in allowed for v in self.page_modes.tolist()):
            raise ValueError("page_modes contains unsupported values")

        low_rank_count = int(np.sum(self.page_modes == int(ContextPageMode.LOW_RANK)))
        int8_count = int(np.sum(self.page_modes == int(ContextPageMode.INT8)))
        fp16_count = int(np.sum(self.page_modes == int(ContextPageMode.FP16)))
        page_ref_count = int(np.sum(self.page_modes == int(ContextPageMode.PAGE_REF)))
        dim = int(self.original_shape[-1])

        if self.low_rank_means.shape != (low_rank_count, dim):
            raise ValueError("low_rank_means shape mismatch")
        if self.low_rank_us.shape != (low_rank_count, self.page_size, self.rank):
            raise ValueError("low_rank_us shape mismatch")
        if self.low_rank_vt.shape != (low_rank_count, self.rank, dim):
            raise ValueError("low_rank_vt shape mismatch")
        if self.int8_mins.shape != (int8_count, dim):
            raise ValueError("int8_mins shape mismatch")
        if self.int8_scales.shape != (int8_count, dim):
            raise ValueError("int8_scales shape mismatch")
        if self.int8_pages.shape != (int8_count, self.page_size, dim):
            raise ValueError("int8_pages shape mismatch")
        if self.fp16_pages.shape != (fp16_count, self.page_size, dim):
            raise ValueError("fp16_pages shape mismatch")

        for page_idx, mode_value in enumerate(self.page_modes.tolist()):
            ref_idx = int(self.page_ref_indices[page_idx])
            if int(mode_value) == int(ContextPageMode.PAGE_REF):
                if ref_idx < 0 or ref_idx >= page_idx:
                    raise ValueError("page_ref_indices must point to an earlier page")
            else:
                if ref_idx != -1:
                    raise ValueError("non-reference pages must use page_ref_indices=-1")

        expected_vectors = int(np.prod(self.original_shape[:-1]))
        if int(np.sum(self.page_lengths)) != expected_vectors:
            raise ValueError("page_lengths do not sum to the flattened vector count")
        if page_ref_count == n_pages:
            raise ValueError("payload cannot contain only reference pages")

    def storage_bytes(self) -> int:
        arrays = (
            self.page_modes,
            self.page_lengths,
            self.page_ref_indices,
            self.low_rank_means,
            self.low_rank_us,
            self.low_rank_vt,
            self.int8_mins,
            self.int8_scales,
            self.int8_pages,
            self.fp16_pages,
        )
        metadata_bytes = (
            8 * len(self.original_shape)
            + len(self.original_dtype.encode("utf-8"))
            + len(self.schema_version.encode("utf-8"))
            + 16
        )
        return metadata_bytes + int(sum(array.nbytes for array in arrays))

    def to_bytes(self) -> bytes:
        cached = self._bytes_cache
        if cached is not None:
            return cached
        self.validate()
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            original_shape=np.asarray(self.original_shape, dtype=np.int64),
            original_dtype=np.asarray(self.original_dtype),
            page_size=np.asarray(self.page_size, dtype=np.int64),
            rank=np.asarray(self.rank, dtype=np.int64),
            schema_version=np.asarray(self.schema_version),
            page_modes=self.page_modes,
            page_lengths=self.page_lengths,
            page_ref_indices=self.page_ref_indices,
            low_rank_means=self.low_rank_means,
            low_rank_us=self.low_rank_us,
            low_rank_vt=self.low_rank_vt,
            int8_mins=self.int8_mins,
            int8_scales=self.int8_scales,
            int8_pages=self.int8_pages,
            fp16_pages=self.fp16_pages,
        )
        payload = buffer.getvalue()
        self._bytes_cache = payload
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ContextEnvelope":
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            envelope = cls(
                original_shape=tuple(int(v) for v in data["original_shape"].tolist()),
                original_dtype=str(data["original_dtype"].item()),
                page_size=int(data["page_size"].item()),
                rank=int(data["rank"].item()),
                page_modes=data["page_modes"].astype(np.uint8),
                page_lengths=data["page_lengths"].astype(np.int32),
                page_ref_indices=data["page_ref_indices"].astype(np.int32),
                low_rank_means=data["low_rank_means"].astype(np.float16),
                low_rank_us=data["low_rank_us"].astype(np.float16),
                low_rank_vt=data["low_rank_vt"].astype(np.float16),
                int8_mins=data["int8_mins"].astype(np.float16) if "int8_mins" in data.files else np.empty((0, int(data["original_shape"].tolist()[-1])), dtype=np.float16),
                int8_scales=data["int8_scales"].astype(np.float16) if "int8_scales" in data.files else np.empty((0, int(data["original_shape"].tolist()[-1])), dtype=np.float16),
                int8_pages=data["int8_pages"].astype(np.uint8) if "int8_pages" in data.files else np.empty((0, int(data["page_size"].item()), int(data["original_shape"].tolist()[-1])), dtype=np.uint8),
                fp16_pages=data["fp16_pages"].astype(np.float16),
                schema_version=str(data["schema_version"].item()) if "schema_version" in data.files else "context-envelope.v3",
            )
        envelope._bytes_cache = bytes(payload)
        envelope.validate()
        return envelope

    def to_base64(self) -> str:
        cached = self._base64_cache
        if cached is not None:
            return cached
        encoded = bytes_to_b64(self.to_bytes())
        self._base64_cache = encoded
        return encoded

    @classmethod
    def from_base64(cls, value: str, *, max_bytes: int | None = None) -> "ContextEnvelope":
        return cls.from_bytes(bytes_from_b64(value, max_bytes=max_bytes))


class ContextCodec:
    """
    Context compressor targeting effective long-context reduction.

    Keep a tiny hot band, deduplicate repeated pages, compress the cold middle into
    low-rank pages, and fall back to per-channel affine int8 before fp16.

    Context-route contract:
    - validate that the input stays inside the declared contour,
    - compress the input,
    - reconstruct locally,
    - measure exact stats on the real payload,
    - optionally enforce a fail-closed guarantee profile.
    """

    def __init__(self, config: ContextCodecConfig | None = None) -> None:
        self.config = config or ContextCodecConfig()
        self.config.validate()
        self._shape_limits = ShapeLimits()
        self._contour_thresholds = ContourThresholds()

    def _flatten_vectors(self, array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], str]:
        values = validate_numeric_finite_array(np.asarray(array), limits=self._shape_limits)
        shape = tuple(values.shape)
        return values.reshape(-1, values.shape[-1]).astype(np.float32, copy=False), shape, np.dtype(values.dtype).name

    @staticmethod
    def _hash_page(page: np.ndarray, valid_length: int, decimals: int) -> bytes:
        rounded = np.round(page[:valid_length], decimals=decimals).astype(np.float16, copy=False)
        digest = hashlib.blake2b(rounded.tobytes(), digest_size=16)
        digest.update(np.asarray(valid_length, dtype=np.int32).tobytes())
        return digest.digest()

    @staticmethod
    def _relative_rms(reference: np.ndarray, candidate: np.ndarray) -> float:
        diff = reference - candidate
        return float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(reference * reference)) + EPS))

    @staticmethod
    def _max_abs_error(reference: np.ndarray, candidate: np.ndarray) -> float:
        return float(np.max(np.abs(reference - candidate)))

    @staticmethod
    def _top_rank_factors(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _quantize_page_int8(
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
        rel_rms = ContextCodec._relative_rms(core, recon_core)
        max_abs = ContextCodec._max_abs_error(core, recon_core)
        return mins_fp16, scales_fp16, q_page, recon_page, rel_rms, max_abs

    @staticmethod
    def _merge_pages(
        pages: list[np.ndarray],
        page_lengths: list[int] | np.ndarray,
        original_shape: tuple[int, ...],
        original_dtype: str,
    ) -> np.ndarray:
        merged = np.concatenate(
            [page[: int(length)] for page, length in zip(pages, page_lengths)],
            axis=0,
        )
        restored = merged.reshape(original_shape)
        return restored.astype(np.dtype(original_dtype), copy=False)

    def _protected_mask(
        self,
        n_vectors: int,
        protected_vector_indices: Sequence[int] | None,
    ) -> np.ndarray:
        mask = np.zeros(n_vectors, dtype=bool)
        if self.config.prefix_keep_vectors > 0:
            mask[: min(self.config.prefix_keep_vectors, n_vectors)] = True
        if self.config.suffix_keep_vectors > 0:
            mask[max(0, n_vectors - self.config.suffix_keep_vectors) :] = True
        for idx in protected_vector_indices or ():
            if idx < 0 or idx >= n_vectors:
                raise ValueError(f"protected vector index {idx} outside valid range [0, {n_vectors})")
            mask[idx] = True
        return mask

    def compress(
        self,
        array: np.ndarray,
        protected_vector_indices: Sequence[int] | None = None,
        guarantee_profile: ContextGuaranteeProfile | None = None,
        guarantee_mode: GuaranteeMode = GuaranteeMode.ALLOW_BEST_EFFORT,
    ) -> tuple[ContextEnvelope, ContextCompressionStats]:
        vectors, original_shape, original_dtype = self._flatten_vectors(array)
        n_vectors, dim = vectors.shape
        page_size = self.config.page_size
        n_pages = math.ceil(n_vectors / page_size)
        protected_mask = self._protected_mask(n_vectors, protected_vector_indices)

        page_modes: list[int] = []
        page_lengths: list[int] = []
        page_refs: list[int] = []
        low_rank_means: list[np.ndarray] = []
        low_rank_us: list[np.ndarray] = []
        low_rank_vt: list[np.ndarray] = []
        int8_mins: list[np.ndarray] = []
        int8_scales: list[np.ndarray] = []
        int8_pages: list[np.ndarray] = []
        fp16_pages: list[np.ndarray] = []

        approx_pages: list[np.ndarray] = []
        page_cache: dict[bytes, list[int]] = {}
        protected_pages = 0

        for page_idx in range(n_pages):
            start = page_idx * page_size
            end = min(start + page_size, n_vectors)
            valid_length = end - start
            page = np.zeros((page_size, dim), dtype=np.float32)
            page[:valid_length] = vectors[start:end]

            is_protected = bool(np.any(protected_mask[start:end]))
            if is_protected:
                protected_pages += 1
            page_lengths.append(valid_length)

            digest = b""
            if not is_protected and self.config.enable_page_ref:
                digest = self._hash_page(page, valid_length, self.config.ref_round_decimals)
                matched_ref: int | None = None
                for candidate_idx in page_cache.get(digest, []):
                    candidate_page = approx_pages[candidate_idx][:valid_length]
                    ref_rel_rms = self._relative_rms(page[:valid_length], candidate_page)
                    if ref_rel_rms <= self.config.page_ref_rel_rms_threshold:
                        matched_ref = int(candidate_idx)
                        break
                if matched_ref is not None:
                    page_modes.append(int(ContextPageMode.PAGE_REF))
                    page_refs.append(matched_ref)
                    approx_pages.append(approx_pages[matched_ref])
                    continue

            if not is_protected and valid_length > 1:
                core = page[:valid_length]
                mean = core.mean(axis=0, keepdims=True)
                centered = core - mean
                target_rank = min(self.config.rank, valid_length, dim)
                us, vt = self._top_rank_factors(centered, target_rank)

                mean_fp16 = mean.reshape(-1).astype(np.float16)
                us_fp16 = np.zeros((page_size, self.config.rank), dtype=np.float16)
                vt_fp16 = np.zeros((self.config.rank, dim), dtype=np.float16)
                if target_rank > 0:
                    us_fp16[:valid_length, :target_rank] = us.astype(np.float16)
                    vt_fp16[:target_rank, :] = vt.astype(np.float16)
                recon_core = (
                    mean_fp16.astype(np.float32)[None, :]
                    + us_fp16[:valid_length].astype(np.float32) @ vt_fp16.astype(np.float32)
                )
                rel_rms = self._relative_rms(core, recon_core)

                low_rank_bytes = dim * 2 + page_size * self.config.rank * 2 + self.config.rank * dim * 2
                fp16_bytes = page_size * dim * 2
                if rel_rms <= self.config.low_rank_error_threshold and low_rank_bytes < fp16_bytes:
                    approx_page = np.zeros_like(page)
                    approx_page[:valid_length] = recon_core.astype(np.float32)
                    page_modes.append(int(ContextPageMode.LOW_RANK))
                    page_refs.append(-1)
                    low_rank_means.append(mean_fp16)
                    low_rank_us.append(us_fp16)
                    low_rank_vt.append(vt_fp16)
                    approx_pages.append(approx_page)
                    if digest:
                        page_cache.setdefault(digest, []).append(page_idx)
                    continue

            if self.config.enable_int8_fallback and (not is_protected or self.config.try_int8_for_protected):
                mins_fp16, scales_fp16, q_page, recon_page, rel_rms, max_abs = self._quantize_page_int8(page, valid_length, page_size)
                if rel_rms <= self.config.int8_rel_rms_threshold and max_abs <= self.config.int8_max_abs_threshold:
                    page_modes.append(int(ContextPageMode.INT8))
                    page_refs.append(-1)
                    int8_mins.append(mins_fp16)
                    int8_scales.append(scales_fp16)
                    int8_pages.append(q_page)
                    approx_pages.append(recon_page)
                    if digest:
                        page_cache.setdefault(digest, []).append(page_idx)
                    continue

            page_modes.append(int(ContextPageMode.FP16))
            page_refs.append(-1)
            fp16_payload = page.astype(np.float16)
            fp16_pages.append(fp16_payload)
            approx_pages.append(fp16_payload.astype(np.float32))
            if digest:
                page_cache.setdefault(digest, []).append(page_idx)

        low_rank_means_arr = np.asarray(low_rank_means, dtype=np.float16)
        if low_rank_means_arr.size == 0:
            low_rank_means_arr = low_rank_means_arr.reshape(0, dim)
        low_rank_us_arr = np.asarray(low_rank_us, dtype=np.float16)
        if low_rank_us_arr.size == 0:
            low_rank_us_arr = low_rank_us_arr.reshape(0, page_size, self.config.rank)
        low_rank_vt_arr = np.asarray(low_rank_vt, dtype=np.float16)
        if low_rank_vt_arr.size == 0:
            low_rank_vt_arr = low_rank_vt_arr.reshape(0, self.config.rank, dim)
        int8_mins_arr = np.asarray(int8_mins, dtype=np.float16)
        if int8_mins_arr.size == 0:
            int8_mins_arr = int8_mins_arr.reshape(0, dim)
        int8_scales_arr = np.asarray(int8_scales, dtype=np.float16)
        if int8_scales_arr.size == 0:
            int8_scales_arr = int8_scales_arr.reshape(0, dim)
        int8_pages_arr = np.asarray(int8_pages, dtype=np.uint8)
        if int8_pages_arr.size == 0:
            int8_pages_arr = int8_pages_arr.reshape(0, page_size, dim)
        fp16_pages_arr = np.asarray(fp16_pages, dtype=np.float16)
        if fp16_pages_arr.size == 0:
            fp16_pages_arr = fp16_pages_arr.reshape(0, page_size, dim)

        envelope = ContextEnvelope(
            original_shape=original_shape,
            original_dtype=original_dtype,
            page_size=page_size,
            rank=self.config.rank,
            page_modes=np.asarray(page_modes, dtype=np.uint8),
            page_lengths=np.asarray(page_lengths, dtype=np.int32),
            page_ref_indices=np.asarray(page_refs, dtype=np.int32),
            low_rank_means=low_rank_means_arr,
            low_rank_us=low_rank_us_arr,
            low_rank_vt=low_rank_vt_arr,
            int8_mins=int8_mins_arr,
            int8_scales=int8_scales_arr,
            int8_pages=int8_pages_arr,
            fp16_pages=fp16_pages_arr,
        )
        payload_bytes = envelope.to_bytes()
        reconstructed = self._merge_pages(approx_pages, page_lengths, original_shape, original_dtype).astype(np.float32, copy=False)
        contour = analyze_context_contour(
            total_pages=n_pages,
            protected_pages=protected_pages,
            low_rank_pages=int(np.sum(envelope.page_modes == int(ContextPageMode.LOW_RANK))),
            page_ref_pages=int(np.sum(envelope.page_modes == int(ContextPageMode.PAGE_REF))),
            thresholds=self._contour_thresholds,
        )
        stats = self._stats(
            original=np.asarray(array, dtype=np.float32),
            reconstructed=reconstructed,
            envelope=envelope,
            protected_vectors=int(protected_mask.sum()),
            protected_pages=protected_pages,
            contour=contour,
            payload_sha256=sha256_hex(payload_bytes),
            serialized_bytes=len(payload_bytes),
        )

        if not contour.supported and guarantee_mode == GuaranteeMode.FAIL_CLOSED:
            failures = [f"unsupported contour: {contour.contour.value}"]
            failures.extend(contour.reasons)
            stats.guarantee_passed = False
            stats.guarantee_failures = list(failures)
            raise ContourViolation(failures)

        if guarantee_profile is not None:
            outcome = evaluate_context_stats(
                compression_ratio=stats.compression_ratio,
                cosine_similarity=stats.cosine_similarity,
                rms_error=stats.rms_error,
                max_abs_error=stats.max_abs_error,
                profile=guarantee_profile,
            )
            stats.guarantee_passed = outcome.passed
            stats.guarantee_failures = list(outcome.failures)
            stats.guarantee_profile = guarantee_profile.to_dict()
            if guarantee_mode == GuaranteeMode.FAIL_CLOSED and not outcome.passed:
                raise GuaranteeViolation(outcome.failures)

        return envelope, stats

    def decompress(self, envelope: ContextEnvelope) -> np.ndarray:
        envelope.validate()
        dim = int(envelope.original_shape[-1])
        pages: list[np.ndarray] = []
        low_rank_cursor = 0
        int8_cursor = 0
        fp16_cursor = 0

        for page_idx, mode_value in enumerate(envelope.page_modes.tolist()):
            length = int(envelope.page_lengths[page_idx])
            mode = ContextPageMode(int(mode_value))

            if mode == ContextPageMode.PAGE_REF:
                ref_idx = int(envelope.page_ref_indices[page_idx])
                if ref_idx < 0 or ref_idx >= len(pages):
                    raise ValueError("invalid page reference in payload")
                page = pages[ref_idx].copy()
            elif mode == ContextPageMode.LOW_RANK:
                mean = envelope.low_rank_means[low_rank_cursor].astype(np.float32)
                us = envelope.low_rank_us[low_rank_cursor].astype(np.float32)
                vt = envelope.low_rank_vt[low_rank_cursor].astype(np.float32)
                page = np.zeros((envelope.page_size, dim), dtype=np.float32)
                page[:length] = mean[None, :] + us[:length] @ vt
                low_rank_cursor += 1
            elif mode == ContextPageMode.INT8:
                mins = envelope.int8_mins[int8_cursor].astype(np.float32)
                scales = envelope.int8_scales[int8_cursor].astype(np.float32)
                q_page = envelope.int8_pages[int8_cursor].astype(np.float32)
                page = q_page * scales[None, :] + mins[None, :]
                int8_cursor += 1
            elif mode == ContextPageMode.FP16:
                page = envelope.fp16_pages[fp16_cursor].astype(np.float32)
                fp16_cursor += 1
            else:  # pragma: no cover
                raise ValueError(f"unsupported page mode: {mode_value}")
            pages.append(page)

        if (
            low_rank_cursor != envelope.low_rank_means.shape[0]
            or int8_cursor != envelope.int8_mins.shape[0]
            or fp16_cursor != envelope.fp16_pages.shape[0]
        ):
            raise ValueError("payload page cursors do not match encoded arrays")

        return self._merge_pages(pages, envelope.page_lengths, envelope.original_shape, envelope.original_dtype)

    def _stats(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        envelope: ContextEnvelope,
        protected_vectors: int,
        protected_pages: int,
        contour: ContourAnalysis,
        payload_sha256: str,
        serialized_bytes: int,
    ) -> ContextCompressionStats:
        delta = reconstructed - original
        original_bytes = int(original.nbytes)
        stored_bytes = int(envelope.storage_bytes())
        rms_error = float(np.sqrt(np.mean(delta * delta)))
        max_abs_error = float(np.max(np.abs(delta)))

        original_flat = original.reshape(-1)
        recon_flat = reconstructed.reshape(-1)
        denom = float(np.linalg.norm(original_flat) * np.linalg.norm(recon_flat))
        cosine_similarity = float(original_flat @ recon_flat / denom) if denom > EPS else 1.0

        page_mode_counts = {
            "page_ref": int(np.sum(envelope.page_modes == int(ContextPageMode.PAGE_REF))),
            "low_rank": int(np.sum(envelope.page_modes == int(ContextPageMode.LOW_RANK))),
            "int8": int(np.sum(envelope.page_modes == int(ContextPageMode.INT8))),
            "fp16": int(np.sum(envelope.page_modes == int(ContextPageMode.FP16))),
        }
        compression_ratio = float(original_bytes / max(serialized_bytes, 1))
        storage_compression_ratio = float(original_bytes / max(stored_bytes, 1))
        route = "context_codec" if contour.supported else "conservative_codebook"
        return ContextCompressionStats(
            original_bytes=original_bytes,
            stored_bytes=stored_bytes,
            serialized_bytes=int(serialized_bytes),
            compression_ratio=compression_ratio,
            storage_compression_ratio=storage_compression_ratio,
            rms_error=rms_error,
            max_abs_error=max_abs_error,
            cosine_similarity=cosine_similarity,
            page_mode_counts=page_mode_counts,
            protected_vectors=protected_vectors,
            protected_pages=protected_pages,
            payload_sha256=payload_sha256,
            contour=contour.contour.value,
            contour_supported=contour.supported,
            contour_reasons=list(contour.reasons),
            contour_details=contour.to_dict(),
            route_recommendation=route,
        )
