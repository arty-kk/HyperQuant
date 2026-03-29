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

import io
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .bundle import CodebookBundle
from .codebook import merge_from_chunks, split_into_chunks
from .config import CompressionConfig, CompressionMode
from .utils import EPS, bytes_from_b64, bytes_to_b64
from .validation import ShapeLimits, validate_float_dtype, validate_numeric_finite_array, validate_shape


@dataclass
class CodebookStats:
    original_bytes: int
    stored_bytes: int
    serialized_bytes: int
    compression_ratio: float
    storage_compression_ratio: float
    rms_error: float
    max_abs_error: float
    cosine_similarity: float
    mode_counts: dict[str, int]

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
            "mode_counts": self.mode_counts,
        }


@dataclass
class CodebookEnvelope:
    original_shape: tuple[int, ...]
    original_dtype: str
    chunk_size: int
    normalize: bool
    modes: np.ndarray
    indices: np.ndarray
    chunk_scales: np.ndarray
    sign_scales: np.ndarray
    sign_bits: np.ndarray
    int8_scales: np.ndarray
    int8_residuals: np.ndarray
    fp16_chunks: np.ndarray
    schema_version: str = "generic-envelope.v2"
    _bytes_cache: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _base64_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def validate(self) -> None:
        validate_shape(self.original_shape)
        validate_float_dtype(self.original_dtype)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if self.modes.ndim != 1 or self.indices.ndim != 1 or self.chunk_scales.ndim != 1:
            raise ValueError("modes, indices, and chunk_scales must be 1D")
        n_chunks = int(self.modes.shape[0])
        if self.indices.shape[0] != n_chunks or self.chunk_scales.shape[0] != n_chunks:
            raise ValueError("core chunk arrays must have the same length")
        if n_chunks == 0:
            raise ValueError("payload must contain at least one chunk")

        dim = int(self.original_shape[-1])
        if dim % self.chunk_size != 0:
            raise ValueError("original_shape last dimension must be divisible by chunk_size")
        expected_chunks = int(np.prod(self.original_shape[:-1])) * (dim // self.chunk_size)
        if expected_chunks != n_chunks:
            raise ValueError("chunk metadata length does not match original_shape")

        allowed = {int(mode) for mode in CompressionMode}
        if any(int(v) not in allowed for v in self.modes.tolist()):
            raise ValueError("modes contains unsupported values")

        sign_count = int(np.sum(self.modes == int(CompressionMode.SIGN_RESIDUAL)))
        int8_count = int(np.sum(self.modes == int(CompressionMode.INT8_RESIDUAL)))
        fp16_count = int(np.sum(self.modes == int(CompressionMode.FP16_FALLBACK)))

        if self.sign_scales.shape != (sign_count,):
            raise ValueError("sign_scales shape mismatch")
        if self.int8_scales.shape != (int8_count,):
            raise ValueError("int8_scales shape mismatch")
        if self.sign_bits.shape != (sign_count, (self.chunk_size + 7) // 8):
            raise ValueError("sign_bits shape mismatch")
        if self.int8_residuals.shape != (int8_count, self.chunk_size):
            raise ValueError("int8_residuals shape mismatch")
        if self.fp16_chunks.shape != (fp16_count, self.chunk_size):
            raise ValueError("fp16_chunks shape mismatch")

    def storage_bytes(self) -> int:
        arrays = (
            self.modes,
            self.indices,
            self.chunk_scales,
            self.sign_scales,
            self.sign_bits,
            self.int8_scales,
            self.int8_residuals,
            self.fp16_chunks,
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
            chunk_size=np.asarray(self.chunk_size, dtype=np.int64),
            normalize=np.asarray(int(self.normalize), dtype=np.int64),
            schema_version=np.asarray(self.schema_version),
            modes=self.modes,
            indices=self.indices,
            chunk_scales=self.chunk_scales,
            sign_scales=self.sign_scales,
            sign_bits=self.sign_bits,
            int8_scales=self.int8_scales,
            int8_residuals=self.int8_residuals,
            fp16_chunks=self.fp16_chunks,
        )
        payload = buffer.getvalue()
        self._bytes_cache = payload
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CodebookEnvelope":
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            envelope = cls(
                original_shape=tuple(int(v) for v in data["original_shape"].tolist()),
                original_dtype=str(data["original_dtype"].item()),
                chunk_size=int(data["chunk_size"].item()),
                normalize=bool(int(data["normalize"].item())),
                modes=data["modes"].astype(np.uint8),
                indices=data["indices"],
                chunk_scales=data["chunk_scales"].astype(np.float16),
                sign_scales=data["sign_scales"].astype(np.float16),
                sign_bits=data["sign_bits"].astype(np.uint8),
                int8_scales=data["int8_scales"].astype(np.float16),
                int8_residuals=data["int8_residuals"].astype(np.int8),
                fp16_chunks=data["fp16_chunks"].astype(np.float16),
                schema_version=str(data["schema_version"].item()) if "schema_version" in data.files else "generic-envelope.v1",
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
    def from_base64(cls, value: str, *, max_bytes: int | None = None) -> "CodebookEnvelope":
        return cls.from_bytes(bytes_from_b64(value, max_bytes=max_bytes))


class CodebookCodec:
    def __init__(
        self,
        bundle: CodebookBundle,
        config: CompressionConfig | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config or CompressionConfig()
        self.config.validate()
        self.codebook = bundle.codebook.astype(np.float32)
        self.rotation = bundle.rotation.astype(np.float32)
        self.rotation_t = self.rotation.T.astype(np.float32)
        self.index_dtype = np.uint8 if bundle.codebook_size <= 256 else np.uint16
        self._shape_limits = ShapeLimits()

    def _vector_and_chunk_counts(self, original_shape: tuple[int, ...]) -> tuple[int, int]:
        dim = int(original_shape[-1])
        chunks_per_vector = dim // self.bundle.chunk_size
        n_vectors = int(np.prod(original_shape[:-1]))
        return n_vectors, chunks_per_vector

    def _prepare(self, array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int, str]:
        validated = validate_numeric_finite_array(np.asarray(array), limits=self._shape_limits)
        chunks, original_shape, chunks_per_vector = split_into_chunks(
            validated, self.bundle.chunk_size
        )
        rotated = chunks @ self.rotation
        return rotated.astype(np.float32), original_shape, chunks_per_vector, np.dtype(validated.dtype).name

    def _protected_chunk_mask(
        self,
        original_shape: tuple[int, ...],
        protected_vector_indices: Sequence[int] | None,
    ) -> np.ndarray:
        n_vectors, chunks_per_vector = self._vector_and_chunk_counts(original_shape)
        mask = np.zeros(n_vectors * chunks_per_vector, dtype=bool)
        if not protected_vector_indices:
            return mask
        for vector_idx in protected_vector_indices:
            if vector_idx < 0 or vector_idx >= n_vectors:
                raise ValueError(
                    f"protected vector index {vector_idx} outside valid range [0, {n_vectors})"
                )
            start = vector_idx * chunks_per_vector
            end = start + chunks_per_vector
            mask[start:end] = True
        return mask

    @staticmethod
    def _pairwise_sq_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        data_norm = np.sum(data * data, axis=1, keepdims=True)
        cent_norm = np.sum(centroids * centroids, axis=1, keepdims=True).T
        distances = data_norm + cent_norm - 2.0 * (data @ centroids.T)
        return np.maximum(distances, 0.0)

    def _restore_rotated_chunks(
        self,
        *,
        modes: np.ndarray,
        indices: np.ndarray,
        chunk_scales: np.ndarray,
        sign_scales: np.ndarray,
        sign_bits: np.ndarray,
        int8_scales: np.ndarray,
        int8_residuals: np.ndarray,
        fp16_chunks: np.ndarray,
        normalize: bool,
    ) -> np.ndarray:
        base = self.codebook[indices.astype(np.int64)].astype(np.float32)
        restored = base.copy()

        sign_mask = modes == CompressionMode.SIGN_RESIDUAL
        if np.any(sign_mask):
            unpacked = np.unpackbits(
                sign_bits,
                axis=1,
                count=self.bundle.chunk_size,
                bitorder="little",
            ).astype(np.float32)
            signs = unpacked * 2.0 - 1.0
            restored[sign_mask] = base[sign_mask] + sign_scales.astype(np.float32)[:, None] * signs

        int8_mask = modes == CompressionMode.INT8_RESIDUAL
        if np.any(int8_mask):
            restored[int8_mask] = (
                base[int8_mask]
                + int8_scales.astype(np.float32)[:, None] * int8_residuals.astype(np.float32)
            )

        fp16_mask = modes == CompressionMode.FP16_FALLBACK
        if np.any(fp16_mask):
            restored[fp16_mask] = fp16_chunks.astype(np.float32)

        if normalize:
            restored *= chunk_scales.astype(np.float32)[:, None]
        return restored

    def compress(
        self,
        array: np.ndarray,
        protected_vector_indices: Sequence[int] | None = None,
    ) -> tuple[CodebookEnvelope, CodebookStats]:
        original = np.asarray(array)
        original_fp32 = validate_numeric_finite_array(original, limits=self._shape_limits).astype(np.float32, copy=False)
        rotated, original_shape, _, original_dtype = self._prepare(original_fp32)
        protected_chunks = self._protected_chunk_mask(original_shape, protected_vector_indices)

        if self.bundle.normalize:
            norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            scales = np.maximum(norms, EPS)
            normalized = rotated / scales
        else:
            scales = np.ones((rotated.shape[0], 1), dtype=np.float32)
            normalized = rotated

        distances = self._pairwise_sq_distance(normalized, self.codebook)
        indices = distances.argmin(axis=1).astype(self.index_dtype)
        base = self.codebook[indices.astype(np.int64)]
        residual = normalized - base
        error = np.sqrt(np.mean(residual * residual, axis=1))

        modes = np.full(len(normalized), CompressionMode.FP16_FALLBACK, dtype=np.uint8)
        modes[error <= self.config.int8_error_threshold] = CompressionMode.INT8_RESIDUAL
        modes[error <= self.config.sign_error_threshold] = CompressionMode.SIGN_RESIDUAL
        modes[error <= self.config.codebook_error_threshold] = CompressionMode.CODEBOOK_ONLY
        modes[protected_chunks] = np.uint8(self.config.protected_mode)

        sign_mask = modes == CompressionMode.SIGN_RESIDUAL
        int8_mask = modes == CompressionMode.INT8_RESIDUAL
        fp16_mask = modes == CompressionMode.FP16_FALLBACK

        sign_residual = residual[sign_mask]
        sign_scales = np.mean(np.abs(sign_residual), axis=1, keepdims=True)
        sign_scales = np.where(sign_scales < EPS, 1.0, sign_scales)
        sign_bits = (
            np.packbits((sign_residual >= 0).astype(np.uint8), axis=1, bitorder="little")
            if sign_residual.size
            else np.empty((0, (self.bundle.chunk_size + 7) // 8), dtype=np.uint8)
        )

        int8_residual = residual[int8_mask]
        int8_scales = np.max(np.abs(int8_residual), axis=1, keepdims=True) / 127.0
        int8_scales = np.where(int8_scales < EPS, 1.0, int8_scales)
        int8_payload = (
            np.round(int8_residual / int8_scales).clip(-127, 127).astype(np.int8)
            if int8_residual.size
            else np.empty((0, self.bundle.chunk_size), dtype=np.int8)
        )

        fp16_payload = normalized[fp16_mask].astype(np.float16)

        envelope = CodebookEnvelope(
            original_shape=original_shape,
            original_dtype=original_dtype,
            chunk_size=self.bundle.chunk_size,
            normalize=self.bundle.normalize,
            modes=modes.astype(np.uint8),
            indices=indices.astype(self.index_dtype),
            chunk_scales=scales.squeeze(1).astype(np.float16),
            sign_scales=sign_scales.squeeze(1).astype(np.float16),
            sign_bits=sign_bits.astype(np.uint8),
            int8_scales=int8_scales.squeeze(1).astype(np.float16),
            int8_residuals=int8_payload.astype(np.int8),
            fp16_chunks=fp16_payload.astype(np.float16),
        )

        restored_rotated = self._restore_rotated_chunks(
            modes=envelope.modes,
            indices=envelope.indices,
            chunk_scales=envelope.chunk_scales,
            sign_scales=envelope.sign_scales,
            sign_bits=envelope.sign_bits,
            int8_scales=envelope.int8_scales,
            int8_residuals=envelope.int8_residuals,
            fp16_chunks=envelope.fp16_chunks,
            normalize=envelope.normalize,
        )
        cartesian = restored_rotated @ self.rotation_t
        reconstructed = merge_from_chunks(cartesian, original_shape, self.bundle.chunk_size).astype(np.float32, copy=False)
        payload_bytes = envelope.to_bytes()
        stats = self._stats(
            original=original_fp32,
            reconstructed=reconstructed,
            envelope=envelope,
            serialized_bytes=len(payload_bytes),
        )
        return envelope, stats

    def decompress(self, envelope: CodebookEnvelope) -> np.ndarray:
        envelope.validate()
        if envelope.chunk_size != self.bundle.chunk_size:
            raise ValueError("payload chunk_size does not match bundle")
        restored = self._restore_rotated_chunks(
            modes=envelope.modes,
            indices=envelope.indices,
            chunk_scales=envelope.chunk_scales,
            sign_scales=envelope.sign_scales,
            sign_bits=envelope.sign_bits,
            int8_scales=envelope.int8_scales,
            int8_residuals=envelope.int8_residuals,
            fp16_chunks=envelope.fp16_chunks,
            normalize=envelope.normalize,
        )
        cartesian = restored @ self.rotation_t
        reconstructed = merge_from_chunks(cartesian, envelope.original_shape, self.bundle.chunk_size)
        return reconstructed.astype(np.dtype(envelope.original_dtype), copy=False)

    def _stats(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        envelope: CodebookEnvelope,
        serialized_bytes: int,
    ) -> CodebookStats:
        delta = reconstructed - original
        original_bytes = int(original.nbytes)
        stored_bytes = int(envelope.storage_bytes())
        rms_error = float(np.sqrt(np.mean(delta * delta)))
        max_abs_error = float(np.max(np.abs(delta)))

        original_flat = original.reshape(-1)
        recon_flat = reconstructed.reshape(-1)
        denom = float(np.linalg.norm(original_flat) * np.linalg.norm(recon_flat))
        cosine_similarity = float(original_flat @ recon_flat / denom) if denom > EPS else 1.0

        mode_counts = {
            "codebook_only": int(np.sum(envelope.modes == CompressionMode.CODEBOOK_ONLY)),
            "sign_residual": int(np.sum(envelope.modes == CompressionMode.SIGN_RESIDUAL)),
            "int8_residual": int(np.sum(envelope.modes == CompressionMode.INT8_RESIDUAL)),
            "fp16_fallback": int(np.sum(envelope.modes == CompressionMode.FP16_FALLBACK)),
        }
        compression_ratio = float(original_bytes / max(serialized_bytes, 1))
        storage_compression_ratio = float(original_bytes / max(stored_bytes, 1))
        return CodebookStats(
            original_bytes=original_bytes,
            stored_bytes=stored_bytes,
            serialized_bytes=int(serialized_bytes),
            compression_ratio=compression_ratio,
            storage_compression_ratio=storage_compression_ratio,
            rms_error=rms_error,
            max_abs_error=max_abs_error,
            cosine_similarity=cosine_similarity,
            mode_counts=mode_counts,
        )
