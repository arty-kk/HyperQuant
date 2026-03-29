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
import math
from dataclasses import dataclass, field

import numpy as np

from .compat import StrEnum
from .native_core import fwht_rows, native_fwht_status
from .utils import EPS, bytes_from_b64, bytes_to_b64, sha256_hex
from .validation import ShapeLimits, validate_float_dtype, validate_numeric_finite_array, validate_shape


class RotationKind(StrEnum):
    DENSE_QR = "dense_qr"
    STRUCTURED_FWHT = "structured_fwht"


_NORMAL_LLOYD_MAX: dict[int, dict[str, np.ndarray]] = {
    2: {
        "centroids": np.array([-1.510446, -0.452787, 0.452787, 1.510446], dtype=np.float32),
        "thresholds": np.array([-0.981617, 0.0, 0.981617], dtype=np.float32),
    },
    3: {
        "centroids": np.array([-2.151868, -1.343796, -0.755906, -0.245054, 0.245054, 0.755906, 1.343796, 2.151868], dtype=np.float32),
        "thresholds": np.array([-1.747832, -1.049851, -0.50048, 0.0, 0.50048, 1.049851, 1.747832], dtype=np.float32),
    },
    4: {
        "centroids": np.array([-2.73228, -2.068618, -1.617623, -1.255803, -0.941925, -0.656392, -0.387772, -0.128287, 0.128287, 0.387772, 0.656392, 0.941925, 1.255803, 1.617623, 2.068618, 2.73228], dtype=np.float32),
        "thresholds": np.array([-2.400449, -1.84312, -1.436713, -1.098864, -0.799158, -0.522082, -0.25803, 0.0, 0.25803, 0.522082, 0.799158, 1.098864, 1.436713, 1.84312, 2.400449], dtype=np.float32),
    },
}


@dataclass(frozen=True)
class RotatedScalarConfig:
    bits: int = 3
    group_size: int = 128
    rotation_kind: RotationKind = RotationKind.STRUCTURED_FWHT
    rotation_seed: int = 17
    normalize: bool = True
    prefer_native_fwht: bool = True
    profile_name: str = "vector_codec"
    residual_topk: int = 1

    def validate(self) -> None:
        if self.bits not in _NORMAL_LLOYD_MAX:
            raise ValueError(f"bits must be one of {sorted(_NORMAL_LLOYD_MAX)}")
        if self.group_size <= 0:
            raise ValueError("group_size must be > 0")
        if self.group_size & (self.group_size - 1):
            raise ValueError("group_size must be a power of two")
        if not self.normalize:
            raise ValueError("normalize=False is not supported for rotated-scalar scalar quantization")
        if self.residual_topk < 0:
            raise ValueError("residual_topk must be >= 0")
        if self.residual_topk >= self.group_size:
            raise ValueError("residual_topk must be smaller than group_size")


@dataclass
class RotatedScalarStats:
    algorithm: str
    transform: str
    bits: int
    group_size: int
    residual_topk: int
    native_fwht_used: bool
    original_bytes: int
    stored_bytes: int
    serialized_bytes: int
    compression_ratio: float
    storage_compression_ratio: float
    effective_bits_per_value: float
    residual_bits_per_value: float
    rms_error: float
    max_abs_error: float
    cosine_similarity: float
    payload_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "transform": self.transform,
            "bits": self.bits,
            "group_size": self.group_size,
            "residual_topk": self.residual_topk,
            "native_fwht_used": self.native_fwht_used,
            "original_bytes": self.original_bytes,
            "stored_bytes": self.stored_bytes,
            "serialized_bytes": self.serialized_bytes,
            "compression_ratio": self.compression_ratio,
            "storage_compression_ratio": self.storage_compression_ratio,
            "effective_bits_per_value": self.effective_bits_per_value,
            "residual_bits_per_value": self.residual_bits_per_value,
            "rms_error": self.rms_error,
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.cosine_similarity,
            "payload_sha256": self.payload_sha256,
        }


@dataclass
class RotatedScalarEnvelope:
    original_shape: tuple[int, ...]
    original_dtype: str
    group_size: int
    bits: int
    normalize: bool
    padded_dim: int
    rotation_kind: str
    rotation_seed: int
    residual_topk: int
    norms: np.ndarray
    packed_indices: np.ndarray
    residual_positions: np.ndarray
    residual_values: np.ndarray
    schema_version: str = "rotated-scalar-envelope.v2"
    _bytes_cache: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _base64_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def validate(self) -> None:
        validate_shape(self.original_shape)
        validate_float_dtype(self.original_dtype)
        if self.group_size <= 0 or self.group_size & (self.group_size - 1):
            raise ValueError("group_size must be a positive power of two")
        if self.bits not in _NORMAL_LLOYD_MAX:
            raise ValueError(f"bits must be one of {sorted(_NORMAL_LLOYD_MAX)}")
        if self.padded_dim < int(self.original_shape[-1]):
            raise ValueError("padded_dim must be >= original last dimension")
        if self.padded_dim % self.group_size != 0:
            raise ValueError("padded_dim must be divisible by group_size")
        if self.rotation_kind not in {kind.value for kind in RotationKind}:
            raise ValueError("unsupported rotation_kind")
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")
        if self.residual_topk < 0 or self.residual_topk >= self.group_size:
            raise ValueError("residual_topk must be in [0, group_size)")
        if self.norms.ndim != 1:
            raise ValueError("norms must be 1D")
        if self.packed_indices.ndim != 1:
            raise ValueError("packed_indices must be 1D")
        if self.residual_positions.ndim != 2 or self.residual_values.ndim != 2:
            raise ValueError("residual side-channel arrays must be 2D")
        expected_groups = int(np.prod(self.original_shape[:-1])) * (self.padded_dim // self.group_size)
        if self.norms.shape != (expected_groups,):
            raise ValueError("norms shape mismatch")
        expected_bytes = (expected_groups * self.group_size * self.bits + 7) // 8
        if self.packed_indices.shape != (expected_bytes,):
            raise ValueError("packed_indices size mismatch")
        if self.residual_positions.shape != (expected_groups, self.residual_topk):
            raise ValueError("residual_positions shape mismatch")
        if self.residual_values.shape != (expected_groups, self.residual_topk):
            raise ValueError("residual_values shape mismatch")
        if self.residual_topk > 0:
            if np.any(self.residual_positions < 0) or np.any(self.residual_positions >= self.group_size):
                raise ValueError("residual_positions contain out-of-range indices")

    @property
    def value_count(self) -> int:
        return int(np.prod(self.original_shape))

    def storage_bytes(self) -> int:
        metadata_bytes = (
            8 * len(self.original_shape)
            + len(self.original_dtype.encode("utf-8"))
            + len(self.rotation_kind.encode("utf-8"))
            + len(self.schema_version.encode("utf-8"))
            + 32
        )
        return metadata_bytes + int(
            self.norms.nbytes + self.packed_indices.nbytes + self.residual_positions.nbytes + self.residual_values.nbytes
        )

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
            group_size=np.asarray(self.group_size, dtype=np.int64),
            bits=np.asarray(self.bits, dtype=np.int64),
            normalize=np.asarray(int(self.normalize), dtype=np.int64),
            padded_dim=np.asarray(self.padded_dim, dtype=np.int64),
            rotation_kind=np.asarray(self.rotation_kind),
            rotation_seed=np.asarray(self.rotation_seed, dtype=np.int64),
            residual_topk=np.asarray(self.residual_topk, dtype=np.int64),
            schema_version=np.asarray(self.schema_version),
            norms=self.norms.astype(np.float16),
            packed_indices=self.packed_indices.astype(np.uint8),
            residual_positions=self.residual_positions.astype(np.uint16),
            residual_values=self.residual_values.astype(np.float16),
        )
        payload = buffer.getvalue()
        self._bytes_cache = payload
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> "RotatedScalarEnvelope":
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            group_count = int(np.asarray(data["norms"]).shape[0])
            residual_topk = int(data["residual_topk"].item()) if "residual_topk" in data.files else 0
            if residual_topk > 0:
                residual_positions = data["residual_positions"].astype(np.uint16)
                residual_values = data["residual_values"].astype(np.float16)
            else:
                residual_positions = np.empty((group_count, 0), dtype=np.uint16)
                residual_values = np.empty((group_count, 0), dtype=np.float16)
            envelope = cls(
                original_shape=tuple(int(v) for v in data["original_shape"].tolist()),
                original_dtype=str(data["original_dtype"].item()),
                group_size=int(data["group_size"].item()),
                bits=int(data["bits"].item()),
                normalize=bool(int(data["normalize"].item())),
                padded_dim=int(data["padded_dim"].item()),
                rotation_kind=str(data["rotation_kind"].item()),
                rotation_seed=int(data["rotation_seed"].item()),
                residual_topk=residual_topk,
                norms=data["norms"].astype(np.float16),
                packed_indices=data["packed_indices"].astype(np.uint8),
                residual_positions=residual_positions,
                residual_values=residual_values,
                schema_version=str(data["schema_version"].item()) if "schema_version" in data.files else "rotated-scalar-envelope.v1",
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
    def from_base64(cls, value: str, *, max_bytes: int | None = None) -> "RotatedScalarEnvelope":
        return cls.from_bytes(bytes_from_b64(value, max_bytes=max_bytes))


class RotatedScalarCodec:
    def __init__(self, config: RotatedScalarConfig | None = None) -> None:
        self.config = config or RotatedScalarConfig()
        self.config.validate()
        self._shape_limits = ShapeLimits()
        self._levels = 1 << self.config.bits
        reference = _NORMAL_LLOYD_MAX[self.config.bits]
        scale = 1.0 / math.sqrt(self.config.group_size)
        self._centroids = reference["centroids"] * scale
        self._thresholds = reference["thresholds"] * scale
        self._dense_rotation: np.ndarray | None = None
        self._dense_rotation_t: np.ndarray | None = None
        self._signs: np.ndarray | None = None
        self._perm: np.ndarray | None = None
        self._inverse_perm: np.ndarray | None = None
        self._native_fwht_used = False

        if self.config.rotation_kind == RotationKind.DENSE_QR:
            self._dense_rotation = self._orthogonal_matrix(self.config.group_size, self.config.rotation_seed)
            self._dense_rotation_t = self._dense_rotation.T.astype(np.float32)
        else:
            rng = np.random.default_rng(self.config.rotation_seed)
            self._signs = np.where(rng.standard_normal(self.config.group_size) >= 0.0, 1.0, -1.0).astype(np.float32)
            self._perm = rng.permutation(self.config.group_size).astype(np.int64)
            self._inverse_perm = np.argsort(self._perm).astype(np.int64)

    @staticmethod
    def _orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((dim, dim), dtype=np.float32)
        q, r = np.linalg.qr(matrix)
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        q = q * signs[np.newaxis, :]
        return q.astype(np.float32)

    def _flatten_groups(self, array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], str, int]:
        validated = validate_numeric_finite_array(np.asarray(array), limits=self._shape_limits)
        original_shape = tuple(validated.shape)
        original_dtype = np.dtype(validated.dtype).name
        vectors = validated.reshape(-1, original_shape[-1]).astype(np.float32, copy=False)
        dim = int(vectors.shape[1])
        groups_per_vector = math.ceil(dim / self.config.group_size)
        padded_dim = groups_per_vector * self.config.group_size
        if padded_dim != dim:
            padded = np.zeros((vectors.shape[0], padded_dim), dtype=np.float32)
            padded[:, :dim] = vectors
            vectors = padded
        groups = vectors.reshape(-1, self.config.group_size)
        return groups, original_shape, original_dtype, padded_dim

    @staticmethod
    def _pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
        flat = np.asarray(indices, dtype=np.uint8).reshape(-1, 1)
        shifts = np.arange(bits, dtype=np.uint8)
        bitplanes = ((flat >> shifts) & 1).astype(np.uint8)
        packed = np.packbits(bitplanes.reshape(-1), bitorder="little")
        return packed.astype(np.uint8, copy=False)

    @staticmethod
    def _unpack_indices(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
        flat_bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little", count=count * bits)
        bitplanes = flat_bits.reshape(count, bits)
        shifts = np.arange(bits, dtype=np.uint8)
        values = np.sum(bitplanes << shifts, axis=1, dtype=np.uint16)
        return values.astype(np.uint8, copy=False)

    def _forward_rotate(self, normalized: np.ndarray) -> np.ndarray:
        if self.config.rotation_kind == RotationKind.DENSE_QR:
            assert self._dense_rotation is not None
            return (normalized @ self._dense_rotation).astype(np.float32, copy=False)
        assert self._signs is not None and self._perm is not None
        prepared = normalized[:, self._perm] * self._signs[None, :]
        transformed, native_used = fwht_rows(prepared, prefer_native=self.config.prefer_native_fwht)
        self._native_fwht_used = bool(native_used)
        return transformed

    def _inverse_rotate(self, rotated: np.ndarray) -> np.ndarray:
        if self.config.rotation_kind == RotationKind.DENSE_QR:
            assert self._dense_rotation_t is not None
            return (rotated @ self._dense_rotation_t).astype(np.float32, copy=False)
        assert self._signs is not None and self._inverse_perm is not None
        restored, native_used = fwht_rows(rotated, prefer_native=self.config.prefer_native_fwht)
        self._native_fwht_used = self._native_fwht_used or bool(native_used)
        restored = restored * self._signs[None, :]
        return restored[:, self._inverse_perm]

    def _select_residuals(self, rotated: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        topk = self.config.residual_topk
        group_count = rotated.shape[0]
        if topk <= 0:
            return np.empty((group_count, 0), dtype=np.uint16), np.empty((group_count, 0), dtype=np.float16)
        approx = self._centroids[indices.astype(np.int64)].astype(np.float32)
        residual_error = np.abs(rotated - approx)
        if topk == 1:
            positions = residual_error.argmax(axis=1, keepdims=True).astype(np.uint16)
        else:
            positions = np.argpartition(residual_error, -topk, axis=1)[:, -topk:].astype(np.uint16, copy=False)
            order = np.argsort(positions, axis=1)
            positions = np.take_along_axis(positions, order, axis=1)
        rows = np.arange(group_count, dtype=np.int64)[:, None]
        values = rotated[rows, positions.astype(np.int64)].astype(np.float16)
        return positions, values

    def compress(self, array: np.ndarray) -> tuple[RotatedScalarEnvelope, RotatedScalarStats]:
        self._native_fwht_used = False
        groups, original_shape, original_dtype, padded_dim = self._flatten_groups(array)
        norms = np.linalg.norm(groups, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, EPS)
        normalized = groups / safe_norms

        rotated = self._forward_rotate(normalized)
        indices = np.digitize(rotated, self._thresholds, right=False).astype(np.uint8)
        residual_positions, residual_values = self._select_residuals(rotated, indices)
        packed_indices = self._pack_indices(indices, self.config.bits)

        envelope = RotatedScalarEnvelope(
            original_shape=original_shape,
            original_dtype=original_dtype,
            group_size=self.config.group_size,
            bits=self.config.bits,
            normalize=self.config.normalize,
            padded_dim=padded_dim,
            rotation_kind=self.config.rotation_kind.value,
            rotation_seed=self.config.rotation_seed,
            residual_topk=self.config.residual_topk,
            norms=safe_norms.squeeze(1).astype(np.float16),
            packed_indices=packed_indices,
            residual_positions=residual_positions,
            residual_values=residual_values,
        )
        payload = envelope.to_bytes()
        reconstructed = self.decompress(envelope).astype(np.float32, copy=False)
        stats = self._stats(
            original=np.asarray(array, dtype=np.float32),
            reconstructed=reconstructed,
            envelope=envelope,
            serialized_bytes=len(payload),
            payload_sha256=sha256_hex(payload),
        )
        return envelope, stats

    def decompress(self, envelope: RotatedScalarEnvelope) -> np.ndarray:
        envelope.validate()
        if envelope.group_size != self.config.group_size:
            raise ValueError("payload group_size does not match compressor config")
        if envelope.bits != self.config.bits:
            raise ValueError("payload bits do not match compressor config")
        if envelope.rotation_kind != self.config.rotation_kind.value:
            raise ValueError("payload rotation_kind does not match compressor config")
        if envelope.rotation_seed != self.config.rotation_seed:
            raise ValueError("payload rotation_seed does not match compressor config")
        if envelope.residual_topk != self.config.residual_topk:
            raise ValueError("payload residual_topk does not match compressor config")

        group_count = envelope.norms.shape[0]
        indices = self._unpack_indices(envelope.packed_indices, group_count * self.config.group_size, self.config.bits)
        indices = indices.reshape(group_count, self.config.group_size)
        rotated = self._centroids[indices.astype(np.int64)].astype(np.float32)
        if envelope.residual_topk > 0:
            rows = np.arange(group_count, dtype=np.int64)[:, None]
            rotated[rows, envelope.residual_positions.astype(np.int64)] = envelope.residual_values.astype(np.float32)
        restored = self._inverse_rotate(rotated)
        restored *= envelope.norms.astype(np.float32)[:, None]

        n_vectors = int(np.prod(envelope.original_shape[:-1]))
        merged = restored.reshape(n_vectors, envelope.padded_dim)
        merged = merged[:, : int(envelope.original_shape[-1])]
        reconstructed = merged.reshape(envelope.original_shape)
        return reconstructed.astype(np.dtype(envelope.original_dtype), copy=False)

    def _stats(
        self,
        *,
        original: np.ndarray,
        reconstructed: np.ndarray,
        envelope: RotatedScalarEnvelope,
        serialized_bytes: int,
        payload_sha256: str,
    ) -> RotatedScalarStats:
        delta = reconstructed - original
        original_bytes = int(original.nbytes)
        stored_bytes = int(envelope.storage_bytes())
        value_count = max(envelope.value_count, 1)
        quantized_bits = envelope.packed_indices.nbytes * 8
        norm_bits = envelope.norms.nbytes * 8
        residual_bits = (envelope.residual_positions.nbytes + envelope.residual_values.nbytes) * 8
        effective_bits_per_value = float((quantized_bits + norm_bits + residual_bits) / value_count)
        residual_bits_per_value = float(residual_bits / value_count)
        original_flat = original.reshape(-1)
        recon_flat = reconstructed.reshape(-1)
        denom = float(np.linalg.norm(original_flat) * np.linalg.norm(recon_flat))
        cosine_similarity = float(original_flat @ recon_flat / denom) if denom > EPS else 1.0
        return RotatedScalarStats(
            algorithm=self.config.profile_name,
            transform=self.config.rotation_kind.value,
            bits=self.config.bits,
            group_size=self.config.group_size,
            residual_topk=self.config.residual_topk,
            native_fwht_used=bool(self._native_fwht_used) if self.config.rotation_kind == RotationKind.STRUCTURED_FWHT else False,
            original_bytes=original_bytes,
            stored_bytes=stored_bytes,
            serialized_bytes=int(serialized_bytes),
            compression_ratio=float(original_bytes / max(serialized_bytes, 1)),
            storage_compression_ratio=float(original_bytes / max(stored_bytes, 1)),
            effective_bits_per_value=effective_bits_per_value,
            residual_bits_per_value=residual_bits_per_value,
            rms_error=float(np.sqrt(np.mean(delta * delta))),
            max_abs_error=float(np.max(np.abs(delta))),
            cosine_similarity=cosine_similarity,
            payload_sha256=payload_sha256,
        )


class DenseRotationBaseline(RotatedScalarCodec):
    def __init__(self, bits: int = 3, group_size: int = 128, rotation_seed: int = 17, *, residual_topk: int = 1) -> None:
        super().__init__(
            RotatedScalarConfig(
                bits=bits,
                group_size=group_size,
                rotation_kind=RotationKind.DENSE_QR,
                rotation_seed=rotation_seed,
                normalize=True,
                prefer_native_fwht=False,
                profile_name="dense_rotation_baseline",
                residual_topk=residual_topk,
            )
        )


class VectorCodec(RotatedScalarCodec):
    def __init__(
        self,
        bits: int = 3,
        group_size: int = 128,
        rotation_seed: int = 17,
        *,
        prefer_native_fwht: bool = True,
        residual_topk: int = 1,
    ) -> None:
        super().__init__(
            RotatedScalarConfig(
                bits=bits,
                group_size=group_size,
                rotation_kind=RotationKind.STRUCTURED_FWHT,
                rotation_seed=rotation_seed,
                normalize=True,
                prefer_native_fwht=prefer_native_fwht,
                profile_name="vector_codec",
                residual_topk=residual_topk,
            )
        )


__all__ = [
    "RotationKind",
    "RotatedScalarConfig",
    "RotatedScalarEnvelope",
    "RotatedScalarStats",
    "RotatedScalarCodec",
    "DenseRotationBaseline",
    "VectorCodec",
    "native_fwht_status",
]
