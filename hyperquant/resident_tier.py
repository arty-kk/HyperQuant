# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from .compat import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np

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
    RESIDENT_ALLOW_VECTOR_FOR_PROTECTED_DEFAULT,
    RESIDENT_HOT_PAGES_DEFAULT,
    VECTOR_BITS_DEFAULT,
    VECTOR_GROUP_SIZE_DEFAULT,
    VECTOR_PREFER_NATIVE_FWHT_DEFAULT,
    VECTOR_RESIDUAL_TOPK_DEFAULT,
    VECTOR_ROTATION_SEED_DEFAULT,
)
from .guarantee import ContourViolation, GuaranteeMode, GuaranteeViolation
from .context_codec import ContextCodecConfig, ContextCodec
from .page_ops import hash_page, max_abs_error, protected_mask, quantize_page_int8, relative_rms, top_rank_factors
from .vector_codec import VectorCodec
from .utils import sha256_hex
from .validation import ShapeLimits, validate_float_dtype, validate_numeric_finite_array, validate_shape


class ResidentPageMode(StrEnum):
    PAGE_REF = "page_ref"
    LOW_RANK = "low_rank"
    INT8 = "int8"
    VECTOR = "vector"
    FP16 = "fp16"


@dataclass(frozen=True)
class ResidentTierConfig:
    page_size: int = CONTEXT_PAGE_SIZE_DEFAULT
    rank: int = CONTEXT_RANK_DEFAULT
    bits: int = VECTOR_BITS_DEFAULT
    group_size: int = VECTOR_GROUP_SIZE_DEFAULT
    rotation_seed: int = VECTOR_ROTATION_SEED_DEFAULT
    residual_topk: int = VECTOR_RESIDUAL_TOPK_DEFAULT
    hot_pages: int = RESIDENT_HOT_PAGES_DEFAULT
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
    prefer_native_fwht: bool = VECTOR_PREFER_NATIVE_FWHT_DEFAULT
    allow_vector_for_protected: bool = RESIDENT_ALLOW_VECTOR_FOR_PROTECTED_DEFAULT

    def validate(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be > 0")
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.bits not in {2, 3, 4}:
            raise ValueError("bits must be one of {2, 3, 4}")
        if self.group_size <= 0 or self.group_size & (self.group_size - 1):
            raise ValueError("group_size must be a positive power of two")
        if self.residual_topk < 0:
            raise ValueError("residual_topk must be >= 0")
        if self.residual_topk >= self.group_size:
            raise ValueError("residual_topk must be smaller than group_size")
        if self.hot_pages <= 0:
            raise ValueError("hot_pages must be > 0")
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

    def to_dict(self) -> dict[str, object]:
        return {
            "page_size": self.page_size,
            "rank": self.rank,
            "bits": self.bits,
            "group_size": self.group_size,
            "rotation_seed": self.rotation_seed,
            "residual_topk": self.residual_topk,
            "hot_pages": self.hot_pages,
            "prefix_keep_vectors": self.prefix_keep_vectors,
            "suffix_keep_vectors": self.suffix_keep_vectors,
            "low_rank_error_threshold": self.low_rank_error_threshold,
            "ref_round_decimals": self.ref_round_decimals,
            "enable_page_ref": self.enable_page_ref,
            "page_ref_rel_rms_threshold": self.page_ref_rel_rms_threshold,
            "enable_int8_fallback": self.enable_int8_fallback,
            "try_int8_for_protected": self.try_int8_for_protected,
            "int8_rel_rms_threshold": self.int8_rel_rms_threshold,
            "int8_max_abs_threshold": self.int8_max_abs_threshold,
            "prefer_native_fwht": self.prefer_native_fwht,
            "allow_vector_for_protected": self.allow_vector_for_protected,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ResidentTierConfig":
        return cls(**payload)


@dataclass(frozen=True)
class ResidentPageDescriptor:
    page_index: int
    mode: str
    length: int
    ref_page_index: int = -1
    file_name: str | None = None
    compressed_bytes: int = 0
    rel_rms_error: float = 0.0
    max_abs_error: float = 0.0
    payload_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "mode": self.mode,
            "length": self.length,
            "ref_page_index": self.ref_page_index,
            "file_name": self.file_name,
            "compressed_bytes": self.compressed_bytes,
            "rel_rms_error": self.rel_rms_error,
            "max_abs_error": self.max_abs_error,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ResidentPageDescriptor":
        return cls(
            page_index=int(payload["page_index"]),
            mode=str(payload["mode"]),
            length=int(payload["length"]),
            ref_page_index=int(payload.get("ref_page_index", -1)),
            file_name=str(payload["file_name"]) if payload.get("file_name") else None,
            compressed_bytes=int(payload.get("compressed_bytes", 0)),
            rel_rms_error=float(payload.get("rel_rms_error", 0.0)),
            max_abs_error=float(payload.get("max_abs_error", 0.0)),
            payload_sha256=str(payload["payload_sha256"]) if payload.get("payload_sha256") else None,
        )


@dataclass(frozen=True)
class ResidentTierStats:
    route: str
    original_bytes: int
    artifact_bytes: int
    manifest_bytes: int
    compression_ratio: float
    rms_error: float
    max_abs_error: float
    cosine_similarity: float
    page_mode_counts: dict[str, int]
    unique_materialized_pages: int
    reference_pages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "original_bytes": self.original_bytes,
            "artifact_bytes": self.artifact_bytes,
            "manifest_bytes": self.manifest_bytes,
            "compression_ratio": self.compression_ratio,
            "rms_error": self.rms_error,
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.cosine_similarity,
            "page_mode_counts": dict(self.page_mode_counts),
            "unique_materialized_pages": self.unique_materialized_pages,
            "reference_pages": self.reference_pages,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ResidentTierStats":
        return cls(
            route=str(payload["route"]),
            original_bytes=int(payload["original_bytes"]),
            artifact_bytes=int(payload["artifact_bytes"]),
            manifest_bytes=int(payload["manifest_bytes"]),
            compression_ratio=float(payload["compression_ratio"]),
            rms_error=float(payload["rms_error"]),
            max_abs_error=float(payload["max_abs_error"]),
            cosine_similarity=float(payload["cosine_similarity"]),
            page_mode_counts={str(k): int(v) for k, v in dict(payload["page_mode_counts"]).items()},
            unique_materialized_pages=int(payload["unique_materialized_pages"]),
            reference_pages=int(payload["reference_pages"]),
        )


@dataclass(frozen=True)
class ResidentTierManifest:
    original_shape: tuple[int, ...]
    original_dtype: str
    config: ResidentTierConfig
    pages: list[ResidentPageDescriptor]
    stats: ResidentTierStats
    schema_version: str = "resident-tier.v1"

    def validate(self) -> None:
        validate_shape(self.original_shape)
        validate_float_dtype(self.original_dtype)
        self.config.validate()
        if not self.pages:
            raise ValueError("pages must not be empty")
        page_count = len(self.pages)
        total_vectors = 0
        for expected_idx, page in enumerate(self.pages):
            if page.page_index != expected_idx:
                raise ValueError("page indices must be contiguous and ordered")
            if page.length <= 0 or page.length > self.config.page_size:
                raise ValueError("page length must be in [1, page_size]")
            if page.mode not in {mode.value for mode in ResidentPageMode}:
                raise ValueError("unsupported page mode")
            if page.mode == ResidentPageMode.PAGE_REF.value:
                if page.ref_page_index < 0 or page.ref_page_index >= page.page_index:
                    raise ValueError("page_ref must point to an earlier page")
                if page.file_name is not None:
                    raise ValueError("page_ref pages must not have file_name")
            else:
                if page.file_name is None:
                    raise ValueError("materialized pages must have file_name")
                if not page.payload_sha256:
                    raise ValueError("materialized pages must include payload_sha256")
                if page.ref_page_index != -1:
                    raise ValueError("non-reference pages must use ref_page_index=-1")
            total_vectors += page.length
        expected_vectors = int(np.prod(self.original_shape[:-1]))
        if total_vectors != expected_vectors:
            raise ValueError("page lengths do not sum to total vector count")
        if page_count != math.ceil(expected_vectors / self.config.page_size):
            raise ValueError("page count does not match shape and page_size")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "original_shape": list(self.original_shape),
            "original_dtype": self.original_dtype,
            "config": self.config.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "stats": self.stats.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ResidentTierManifest":
        manifest = cls(
            original_shape=tuple(int(v) for v in payload["original_shape"]),
            original_dtype=str(payload["original_dtype"]),
            config=ResidentTierConfig.from_dict(dict(payload["config"])),
            pages=[ResidentPageDescriptor.from_dict(item) for item in list(payload["pages"])],
            stats=ResidentTierStats.from_dict(dict(payload["stats"])),
            schema_version=str(payload.get("schema_version", "resident-tier.v1")),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_path(cls, path: str | Path) -> "ResidentTierManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ResidentPlan:
    chosen_route: str
    concurrent_sessions: int
    active_window_tokens: int
    runtime_value_bytes: int
    budget_bytes: int | None
    baseline_resident_bytes_per_session: int
    projected_resident_bytes_per_session: int
    artifact_bytes_per_session: int
    resident_savings_bytes_per_session: int
    resident_savings_ratio: float
    fleet_baseline_resident_bytes: int
    fleet_projected_resident_bytes: int
    fits_budget: bool | None
    max_sessions_within_budget_baseline: int | None = None
    max_sessions_within_budget_projected: int | None = None
    capacity_gain_vs_baseline: float | None = None
    candidates: dict[str, dict[str, object]] = field(default_factory=dict)
    page_mode_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "chosen_route": self.chosen_route,
            "concurrent_sessions": self.concurrent_sessions,
            "active_window_tokens": self.active_window_tokens,
            "runtime_value_bytes": self.runtime_value_bytes,
            "budget_bytes": self.budget_bytes,
            "baseline_resident_bytes_per_session": self.baseline_resident_bytes_per_session,
            "projected_resident_bytes_per_session": self.projected_resident_bytes_per_session,
            "artifact_bytes_per_session": self.artifact_bytes_per_session,
            "resident_savings_bytes_per_session": self.resident_savings_bytes_per_session,
            "resident_savings_ratio": self.resident_savings_ratio,
            "fleet_baseline_resident_bytes": self.fleet_baseline_resident_bytes,
            "fleet_projected_resident_bytes": self.fleet_projected_resident_bytes,
            "fits_budget": self.fits_budget,
            "max_sessions_within_budget_baseline": self.max_sessions_within_budget_baseline,
            "max_sessions_within_budget_projected": self.max_sessions_within_budget_projected,
            "capacity_gain_vs_baseline": self.capacity_gain_vs_baseline,
            "candidates": self.candidates,
            "page_mode_counts": dict(self.page_mode_counts),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ResidentAccessReport:
    manifest_bytes: int
    resident_cache_bytes: int
    resident_total_bytes: int
    cached_pages: int
    max_hot_pages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_bytes": self.manifest_bytes,
            "resident_cache_bytes": self.resident_cache_bytes,
            "resident_total_bytes": self.resident_total_bytes,
            "cached_pages": self.cached_pages,
            "max_hot_pages": self.max_hot_pages,
        }


@dataclass
class _TieredBuildArtifacts:
    manifest: ResidentTierManifest
    payload_bytes_total: int


@dataclass(frozen=True)
class ResidentBenchmarkArtifacts:
    report: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(self.report, indent=2, ensure_ascii=False, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# Resident benchmark", ""]
        meta = self.report.get("meta", {})
        if meta:
            lines.extend(["## Meta", ""])
            for key, value in meta.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        for workload_name, workload in self.report.get("workloads", {}).items():
            lines.extend([f"## {workload_name}", ""])
            lines.append("| candidate | resident/session bytes | artifact/session bytes | ratio vs baseline |")
            lines.append("|---|---:|---:|---:|")
            for candidate_name, candidate in workload.get("plan", {}).get("candidates", {}).items():
                lines.append(
                    f"| {candidate_name} | {int(candidate.get('resident_bytes_per_session', 0))} | {int(candidate.get('artifact_bytes_per_session', 0))} | {float(candidate.get('resident_ratio_vs_baseline', 0.0)):.4f} |"
                )
            lines.append("")
            stats = workload.get("tiered_store", {})
            if stats:
                lines.append(
                    f"> Tiered route `{workload.get('plan', {}).get('chosen_route', 'resident_tier')}`: resident/session = {int(workload.get('plan', {}).get('projected_resident_bytes_per_session', 0))} bytes, artifact/session = {int(workload.get('plan', {}).get('artifact_bytes_per_session', 0))} bytes, slice read = {float(stats.get('slice_read_mean_ms', 0.0)):.3f} ms."
                )
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class _TieredPageEncoder:
    def __init__(self, config: ResidentTierConfig) -> None:
        self.config = config
        self.config.validate()
        self._shape_limits = ShapeLimits()
        self._context = ContextCodec(
            ContextCodecConfig(
                page_size=config.page_size,
                rank=config.rank,
                prefix_keep_vectors=config.prefix_keep_vectors,
                suffix_keep_vectors=config.suffix_keep_vectors,
                low_rank_error_threshold=config.low_rank_error_threshold,
                ref_round_decimals=config.ref_round_decimals,
                enable_page_ref=config.enable_page_ref,
                page_ref_rel_rms_threshold=config.page_ref_rel_rms_threshold,
                enable_int8_fallback=config.enable_int8_fallback,
                try_int8_for_protected=config.try_int8_for_protected,
                int8_rel_rms_threshold=config.int8_rel_rms_threshold,
                int8_max_abs_threshold=config.int8_max_abs_threshold,
            )
        )
        self._vector = VectorCodec(
            bits=config.bits,
            group_size=config.group_size,
            rotation_seed=config.rotation_seed,
            prefer_native_fwht=config.prefer_native_fwht,
            residual_topk=config.residual_topk,
        )

    @staticmethod
    def _serialize_npz(**arrays: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    def _flatten_vectors(self, array: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], str]:
        validated = validate_numeric_finite_array(np.asarray(array), limits=self._shape_limits)
        original_shape = tuple(validated.shape)
        original_dtype = np.dtype(validated.dtype).name
        vectors = validated.reshape(-1, original_shape[-1]).astype(np.float32, copy=False)
        return vectors, original_shape, original_dtype

    def _encode_pages(
        self,
        array: np.ndarray,
        *,
        output_dir: Path | None = None,
        protected_vector_indices: Sequence[int] | None = None,
    ) -> _TieredBuildArtifacts:
        vectors, original_shape, original_dtype = self._flatten_vectors(array)
        n_vectors, dim = vectors.shape
        protected_vector_mask = protected_mask(
            n_vectors,
            protected_vector_indices,
            prefix_keep_vectors=self.config.prefix_keep_vectors,
            suffix_keep_vectors=self.config.suffix_keep_vectors,
        )
        page_size = self.config.page_size
        n_pages = math.ceil(n_vectors / page_size)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            pages_dir = output_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
        else:
            pages_dir = None

        page_descriptors: list[ResidentPageDescriptor] = []
        mode_counts = {mode.value: 0 for mode in ResidentPageMode}
        payload_bytes_total = 0
        unique_materialized_pages = 0
        reference_pages = 0

        digest_cache: dict[bytes, list[tuple[int, np.ndarray]]] = {}
        sum_sq_diff = 0.0
        orig_sq = 0.0
        recon_sq = 0.0
        dot = 0.0
        max_abs_error_total = 0.0

        for page_idx in range(n_pages):
            start = page_idx * page_size
            end = min(start + page_size, n_vectors)
            valid_length = int(end - start)
            core = vectors[start:end]
            page = np.zeros((page_size, dim), dtype=np.float32)
            page[:valid_length] = core
            is_protected = bool(np.any(protected_vector_mask[start:end]))
            digest: bytes | None = None
            descriptor: ResidentPageDescriptor | None = None
            payload_bytes = b""
            reconstructed_page = np.zeros_like(page)

            if not is_protected and self.config.enable_page_ref:
                digest = hash_page(page, valid_length, self.config.ref_round_decimals)
                for candidate_idx, candidate_page in digest_cache.get(digest, []):
                    ref_rel_rms = relative_rms(page[:valid_length], candidate_page[:valid_length])
                    if ref_rel_rms <= self.config.page_ref_rel_rms_threshold:
                        descriptor = ResidentPageDescriptor(
                            page_index=page_idx,
                            mode=ResidentPageMode.PAGE_REF.value,
                            length=valid_length,
                            ref_page_index=candidate_idx,
                            file_name=None,
                            compressed_bytes=0,
                            rel_rms_error=ref_rel_rms,
                            max_abs_error=max_abs_error(page[:valid_length], candidate_page[:valid_length]),
                            payload_sha256=None,
                        )
                        reconstructed_page[:valid_length] = candidate_page[:valid_length]
                        reference_pages += 1
                        break

            if descriptor is None and not is_protected:
                mean = page[:valid_length].mean(axis=0)
                centered = page[:valid_length] - mean[None, :]
                us, vt = top_rank_factors(centered, self.config.rank)
                approx_core = mean[None, :] + us @ vt
                rel_rms = relative_rms(page[:valid_length], approx_core)
                if rel_rms <= self.config.low_rank_error_threshold:
                    mean_fp16 = mean.astype(np.float16)
                    us_fp16 = us.astype(np.float16)
                    vt_fp16 = vt.astype(np.float16)
                    payload_bytes = self._serialize_npz(mean=mean_fp16, us=us_fp16, vt=vt_fp16)
                    file_name = f"pages/page_{page_idx:06d}.npz"
                    if pages_dir is not None:
                        (output_dir / file_name).write_bytes(payload_bytes)
                    reconstructed_page[:valid_length] = approx_core
                    descriptor = ResidentPageDescriptor(
                        page_index=page_idx,
                        mode=ResidentPageMode.LOW_RANK.value,
                        length=valid_length,
                        file_name=file_name,
                        compressed_bytes=len(payload_bytes),
                        rel_rms_error=rel_rms,
                        max_abs_error=max_abs_error(page[:valid_length], approx_core),
                        payload_sha256=sha256_hex(payload_bytes),
                    )

            if descriptor is None and self.config.enable_int8_fallback and (not is_protected or self.config.try_int8_for_protected):
                mins_fp16, scales_fp16, q_page, recon_page, rel_rms, max_abs = quantize_page_int8(page, valid_length, page_size)
                if rel_rms <= self.config.int8_rel_rms_threshold and max_abs <= self.config.int8_max_abs_threshold:
                    payload_bytes = self._serialize_npz(
                        mins=mins_fp16,
                        scales=scales_fp16,
                        q=q_page[:valid_length],
                    )
                    file_name = f"pages/page_{page_idx:06d}.npz"
                    if pages_dir is not None:
                        (output_dir / file_name).write_bytes(payload_bytes)
                    reconstructed_page = recon_page
                    descriptor = ResidentPageDescriptor(
                        page_index=page_idx,
                        mode=ResidentPageMode.INT8.value,
                        length=valid_length,
                        file_name=file_name,
                        compressed_bytes=len(payload_bytes),
                        rel_rms_error=rel_rms,
                        max_abs_error=max_abs,
                        payload_sha256=sha256_hex(payload_bytes),
                    )

            if descriptor is None and (not is_protected or self.config.allow_vector_for_protected):
                vector_env, _ = self._vector.compress(page[:valid_length])
                payload_bytes = vector_env.to_bytes()
                file_name = f"pages/page_{page_idx:06d}.bin"
                if pages_dir is not None:
                    (output_dir / file_name).write_bytes(payload_bytes)
                restored_core = self._vector.decompress(vector_env).astype(np.float32, copy=False)
                reconstructed_page[:valid_length] = restored_core
                descriptor = ResidentPageDescriptor(
                    page_index=page_idx,
                    mode=ResidentPageMode.VECTOR.value,
                    length=valid_length,
                    file_name=file_name,
                    compressed_bytes=len(payload_bytes),
                    rel_rms_error=relative_rms(page[:valid_length], restored_core),
                    max_abs_error=float(np.max(np.abs(page[:valid_length] - restored_core))),
                    payload_sha256=sha256_hex(payload_bytes),
                )

            if descriptor is None:
                fp16_page = page[:valid_length].astype(np.float16)
                payload_bytes = self._serialize_npz(page=fp16_page)
                file_name = f"pages/page_{page_idx:06d}.npz"
                if pages_dir is not None:
                    (output_dir / file_name).write_bytes(payload_bytes)
                reconstructed_page[:valid_length] = fp16_page.astype(np.float32)
                descriptor = ResidentPageDescriptor(
                    page_index=page_idx,
                    mode=ResidentPageMode.FP16.value,
                    length=valid_length,
                    file_name=file_name,
                    compressed_bytes=len(payload_bytes),
                    rel_rms_error=relative_rms(page[:valid_length], reconstructed_page[:valid_length]),
                    max_abs_error=float(np.max(np.abs(page[:valid_length] - reconstructed_page[:valid_length]))),
                    payload_sha256=sha256_hex(payload_bytes),
                )

            assert descriptor is not None
            page_descriptors.append(descriptor)
            mode_counts[descriptor.mode] += 1
            payload_bytes_total += descriptor.compressed_bytes
            if descriptor.mode != ResidentPageMode.PAGE_REF.value:
                unique_materialized_pages += 1

            reference = page[:valid_length]
            reconstructed = reconstructed_page[:valid_length]
            diff = reconstructed - reference
            sum_sq_diff += float(np.sum(diff * diff))
            orig_sq += float(np.sum(reference * reference))
            recon_sq += float(np.sum(reconstructed * reconstructed))
            dot += float(np.sum(reference * reconstructed))
            max_abs_error_total = max(max_abs_error_total, float(np.max(np.abs(diff))))

            if digest is not None and descriptor.mode != ResidentPageMode.PAGE_REF.value:
                digest_cache.setdefault(digest, []).append((page_idx, reconstructed_page.copy()))

        original_bytes = int(vectors.shape[0] * vectors.shape[1] * np.dtype(original_dtype).itemsize)
        manifest_placeholder = ResidentTierManifest(
            original_shape=original_shape,
            original_dtype=original_dtype,
            config=self.config,
            pages=page_descriptors,
            stats=ResidentTierStats(
                route="resident_tier",
                original_bytes=original_bytes,
                artifact_bytes=payload_bytes_total,
                manifest_bytes=0,
                compression_ratio=float(original_bytes / max(payload_bytes_total, 1)),
                rms_error=float(math.sqrt(sum_sq_diff / max(vectors.size, 1))),
                max_abs_error=max_abs_error_total,
                cosine_similarity=float(dot / max(math.sqrt(orig_sq * recon_sq), 1e-12)),
                page_mode_counts=mode_counts,
                unique_materialized_pages=unique_materialized_pages,
                reference_pages=reference_pages,
            ),
        )
        manifest_json = manifest_placeholder.to_json()
        manifest_bytes = len(manifest_json.encode("utf-8"))
        final_artifact_bytes = payload_bytes_total + manifest_bytes
        final_stats = ResidentTierStats(
            route="resident_tier",
            original_bytes=original_bytes,
            artifact_bytes=final_artifact_bytes,
            manifest_bytes=manifest_bytes,
            compression_ratio=float(original_bytes / max(final_artifact_bytes, 1)),
            rms_error=float(math.sqrt(sum_sq_diff / max(vectors.size, 1))),
            max_abs_error=max_abs_error_total,
            cosine_similarity=float(dot / max(math.sqrt(orig_sq * recon_sq), 1e-12)),
            page_mode_counts=mode_counts,
            unique_materialized_pages=unique_materialized_pages,
            reference_pages=reference_pages,
        )
        manifest = ResidentTierManifest(
            original_shape=original_shape,
            original_dtype=original_dtype,
            config=self.config,
            pages=page_descriptors,
            stats=final_stats,
        )
        if output_dir is not None:
            (output_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
            payload_total = sum(page.compressed_bytes for page in page_descriptors)
            written_manifest_bytes = (output_dir / "manifest.json").stat().st_size
            final_stats = ResidentTierStats(
                route="resident_tier",
                original_bytes=original_bytes,
                artifact_bytes=payload_total + written_manifest_bytes,
                manifest_bytes=written_manifest_bytes,
                compression_ratio=float(original_bytes / max(payload_total + written_manifest_bytes, 1)),
                rms_error=final_stats.rms_error,
                max_abs_error=final_stats.max_abs_error,
                cosine_similarity=final_stats.cosine_similarity,
                page_mode_counts=mode_counts,
                unique_materialized_pages=unique_materialized_pages,
                reference_pages=reference_pages,
            )
            manifest = ResidentTierManifest(
                original_shape=original_shape,
                original_dtype=original_dtype,
                config=self.config,
                pages=page_descriptors,
                stats=final_stats,
            )
            (output_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        return _TieredBuildArtifacts(manifest=manifest, payload_bytes_total=payload_bytes_total)


class ResidentTierStore:
    def __init__(self, base_dir: str | Path, manifest: ResidentTierManifest) -> None:
        self.base_dir = Path(base_dir)
        self.manifest = manifest
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._vector = VectorCodec(
            bits=self.manifest.config.bits,
            group_size=self.manifest.config.group_size,
            rotation_seed=self.manifest.config.rotation_seed,
            prefer_native_fwht=self.manifest.config.prefer_native_fwht,
            residual_topk=self.manifest.config.residual_topk,
        )

    @classmethod
    def build(
        cls,
        array: np.ndarray,
        output_dir: str | Path,
        *,
        config: ResidentTierConfig | None = None,
        protected_vector_indices: Sequence[int] | None = None,
    ) -> "ResidentTierStore":
        cfg = config or ResidentTierConfig()
        encoder = _TieredPageEncoder(cfg)
        artifacts = encoder._encode_pages(array, output_dir=Path(output_dir), protected_vector_indices=protected_vector_indices)
        return cls(output_dir, artifacts.manifest)

    @classmethod
    def open(cls, output_dir: str | Path) -> "ResidentTierStore":
        base_dir = Path(output_dir)
        manifest = ResidentTierManifest.from_path(base_dir / "manifest.json")
        return cls(base_dir, manifest)

    def access_report(self) -> ResidentAccessReport:
        manifest_bytes = self.manifest.stats.manifest_bytes
        resident_cache_bytes = int(sum(page.nbytes for page in self._cache.values()))
        return ResidentAccessReport(
            manifest_bytes=manifest_bytes,
            resident_cache_bytes=resident_cache_bytes,
            resident_total_bytes=manifest_bytes + resident_cache_bytes,
            cached_pages=len(self._cache),
            max_hot_pages=self.manifest.config.hot_pages,
        )

    def _remember(self, page_idx: int, page: np.ndarray) -> np.ndarray:
        self._cache[page_idx] = page
        self._cache.move_to_end(page_idx)
        while len(self._cache) > self.manifest.config.hot_pages:
            self._cache.popitem(last=False)
        return page

    def evict_all(self) -> None:
        self._cache.clear()

    def preload_pages(self, page_indices: Sequence[int]) -> None:
        for page_idx in page_indices:
            self.get_page(page_idx)

    def _read_verified_payload(self, descriptor: ResidentPageDescriptor) -> bytes:
        if descriptor.file_name is None:
            raise ValueError("descriptor does not reference a payload file")
        if not descriptor.payload_sha256:
            raise ValueError(f"missing payload_sha256 for page {descriptor.page_index}")
        payload = (self.base_dir / descriptor.file_name).read_bytes()
        actual = sha256_hex(payload)
        if actual != descriptor.payload_sha256:
            raise ValueError(
                f"sha256 mismatch for page {descriptor.page_index}: expected {descriptor.payload_sha256}, got {actual}"
            )
        return payload

    def _decode_page(self, descriptor: ResidentPageDescriptor) -> np.ndarray:
        dim = int(self.manifest.original_shape[-1])
        page = np.zeros((self.manifest.config.page_size, dim), dtype=np.float32)
        length = descriptor.length
        if descriptor.mode == ResidentPageMode.PAGE_REF.value:
            ref_page = self.get_page(descriptor.ref_page_index)
            page[:length] = ref_page[:length]
            return page

        payload = self._read_verified_payload(descriptor)
        if descriptor.mode == ResidentPageMode.LOW_RANK.value:
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                mean = data["mean"].astype(np.float32)
                us = data["us"].astype(np.float32)
                vt = data["vt"].astype(np.float32)
            page[:length] = mean[None, :] + us[:length] @ vt
        elif descriptor.mode == ResidentPageMode.INT8.value:
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                mins = data["mins"].astype(np.float32)
                scales = data["scales"].astype(np.float32)
                q = data["q"].astype(np.float32)
            page[:length] = q * scales[None, :] + mins[None, :]
        elif descriptor.mode == ResidentPageMode.VECTOR.value:
            from .vector_codec import RotatedScalarEnvelope

            envelope = RotatedScalarEnvelope.from_bytes(payload)
            restored = self._vector.decompress(envelope).astype(np.float32, copy=False)
            page[:length] = restored[:length]
        elif descriptor.mode == ResidentPageMode.FP16.value:
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                restored = data["page"].astype(np.float32)
            page[:length] = restored[:length]
        else:
            raise ValueError(f"unsupported page mode: {descriptor.mode}")
        return page

    def verify_integrity(self) -> dict[str, object]:
        self.evict_all()
        checked_pages = 0
        checked_payloads = 0
        for descriptor in self.manifest.pages:
            if descriptor.mode != ResidentPageMode.PAGE_REF.value:
                checked_payloads += 1
            self.get_page(descriptor.page_index)
            checked_pages += 1
        self.evict_all()
        return {
            "checked_pages": checked_pages,
            "checked_payloads": checked_payloads,
            "page_mode_counts": dict(self.manifest.stats.page_mode_counts),
        }

    def get_page(self, page_idx: int) -> np.ndarray:
        if page_idx < 0 or page_idx >= len(self.manifest.pages):
            raise IndexError("page index out of range")
        cached = self._cache.get(page_idx)
        if cached is not None:
            self._cache.move_to_end(page_idx)
            return cached.copy()

        descriptor = self.manifest.pages[page_idx]
        page = self._decode_page(descriptor)
        return self._remember(page_idx, page).copy()

    def get_slice(self, start_vector: int, end_vector: int) -> np.ndarray:
        n_vectors = int(np.prod(self.manifest.original_shape[:-1]))
        if start_vector < 0 or end_vector < start_vector or end_vector > n_vectors:
            raise ValueError("invalid slice bounds")
        if start_vector == end_vector:
            return np.empty((0, int(self.manifest.original_shape[-1])), dtype=np.dtype(self.manifest.original_dtype))
        page_size = self.manifest.config.page_size
        first_page = start_vector // page_size
        last_page = (end_vector - 1) // page_size
        parts: list[np.ndarray] = []
        for page_idx in range(first_page, last_page + 1):
            page = self.get_page(page_idx)
            page_start = page_idx * page_size
            local_start = max(0, start_vector - page_start)
            local_end = min(page_size, end_vector - page_start, self.manifest.pages[page_idx].length)
            parts.append(page[local_start:local_end])
        merged = np.concatenate(parts, axis=0)
        return merged.astype(np.dtype(self.manifest.original_dtype), copy=False)


class ResidentPlanner:
    def __init__(self, config: ResidentTierConfig | None = None) -> None:
        self.config = config or ResidentTierConfig()
        self.config.validate()
        self._tier_encoder = _TieredPageEncoder(self.config)
        self._vector = VectorCodec(
            bits=self.config.bits,
            group_size=self.config.group_size,
            rotation_seed=self.config.rotation_seed,
            prefer_native_fwht=self.config.prefer_native_fwht,
            residual_topk=self.config.residual_topk,
        )
        self._context = ContextCodec(
            ContextCodecConfig(
                page_size=self.config.page_size,
                rank=self.config.rank,
                prefix_keep_vectors=self.config.prefix_keep_vectors,
                suffix_keep_vectors=self.config.suffix_keep_vectors,
                low_rank_error_threshold=self.config.low_rank_error_threshold,
                ref_round_decimals=self.config.ref_round_decimals,
                enable_page_ref=self.config.enable_page_ref,
                page_ref_rel_rms_threshold=self.config.page_ref_rel_rms_threshold,
                enable_int8_fallback=self.config.enable_int8_fallback,
                try_int8_for_protected=self.config.try_int8_for_protected,
                int8_rel_rms_threshold=self.config.int8_rel_rms_threshold,
                int8_max_abs_threshold=self.config.int8_max_abs_threshold,
            )
        )

    def plan(
        self,
        array: np.ndarray,
        *,
        concurrent_sessions: int = 1,
        active_window_tokens: int | None = None,
        runtime_value_bytes: int = 2,
        budget_bytes: int | None = None,
        protected_vector_indices: Sequence[int] | None = None,
    ) -> ResidentPlan:
        validated = validate_numeric_finite_array(np.asarray(array), limits=ShapeLimits())
        original_shape = tuple(validated.shape)
        n_vectors = int(np.prod(original_shape[:-1]))
        dim = int(original_shape[-1])
        if concurrent_sessions <= 0:
            raise ValueError("concurrent_sessions must be > 0")
        if runtime_value_bytes <= 0:
            raise ValueError("runtime_value_bytes must be > 0")
        if active_window_tokens is None:
            active_window_tokens = min(n_vectors, self.config.hot_pages * self.config.page_size)
        if active_window_tokens <= 0:
            raise ValueError("active_window_tokens must be > 0")
        active_window_tokens = min(active_window_tokens, n_vectors)
        active_pages = math.ceil(active_window_tokens / self.config.page_size)
        baseline_bytes = n_vectors * dim * runtime_value_bytes

        tier_artifacts = self._tier_encoder._encode_pages(validated, output_dir=None, protected_vector_indices=protected_vector_indices)
        tier_manifest = tier_artifacts.manifest
        tier_resident_bytes = tier_manifest.stats.manifest_bytes + active_pages * self.config.page_size * dim * runtime_value_bytes

        _, vector_stats = self._vector.compress(validated)
        context_stats = None
        context_error = None
        try:
            _, context_stats = self._context.compress(validated, guarantee_mode=GuaranteeMode.ALLOW_BEST_EFFORT)
        except (ValueError, ContourViolation, GuaranteeViolation) as exc:
            context_error = str(exc)

        candidates: dict[str, dict[str, object]] = {
            "baseline_full_resident": {
                "resident_bytes_per_session": baseline_bytes,
                "artifact_bytes_per_session": baseline_bytes,
                "resident_ratio_vs_baseline": 1.0,
                "notes": ["No compression, everything resident in RAM."],
            },
            "vector_codec_full_envelope": {
                "resident_bytes_per_session": int(vector_stats.stored_bytes),
                "artifact_bytes_per_session": int(vector_stats.serialized_bytes),
                "resident_ratio_vs_baseline": float(vector_stats.stored_bytes / max(baseline_bytes, 1)),
                "compression_ratio": float(vector_stats.compression_ratio),
                "rms_error": float(vector_stats.rms_error),
                "cosine_similarity": float(vector_stats.cosine_similarity),
                "notes": ["Whole-array vector-codec envelope kept resident."],
            },
            "resident_tier": {
                "resident_bytes_per_session": int(tier_resident_bytes),
                "artifact_bytes_per_session": int(tier_manifest.stats.artifact_bytes),
                "resident_ratio_vs_baseline": float(tier_resident_bytes / max(baseline_bytes, 1)),
                "compression_ratio": float(tier_manifest.stats.compression_ratio),
                "rms_error": float(tier_manifest.stats.rms_error),
                "cosine_similarity": float(tier_manifest.stats.cosine_similarity),
                "page_mode_counts": dict(tier_manifest.stats.page_mode_counts),
                "notes": [
                    "Only manifest plus hot pages are resident.",
                    f"Projected hot window = {active_pages} page(s).",
                ],
            },
        }
        if context_stats is not None:
            candidates["context_codec_full_envelope"] = {
                "resident_bytes_per_session": int(context_stats.stored_bytes),
                "artifact_bytes_per_session": int(context_stats.serialized_bytes),
                "resident_ratio_vs_baseline": float(context_stats.stored_bytes / max(baseline_bytes, 1)),
                "compression_ratio": float(context_stats.compression_ratio),
                "rms_error": float(context_stats.rms_error),
                "cosine_similarity": float(context_stats.cosine_similarity),
                "contour": context_stats.contour,
                "contour_supported": context_stats.contour_supported,
                "notes": ["Whole-array Context envelope kept resident."],
            }
        elif context_error is not None:
            candidates["context_codec_full_envelope"] = {
                "resident_bytes_per_session": None,
                "artifact_bytes_per_session": None,
                "resident_ratio_vs_baseline": None,
                "error": context_error,
                "notes": ["Input did not qualify for the Context whole-array route."],
            }

        candidate_order = [
            (name, payload)
            for name, payload in candidates.items()
            if isinstance(payload.get("resident_bytes_per_session"), int)
        ]
        chosen_route, chosen_payload = min(candidate_order, key=lambda item: int(item[1]["resident_bytes_per_session"]))

        projected_resident_bytes_per_session = int(chosen_payload["resident_bytes_per_session"])
        fleet_baseline = baseline_bytes * concurrent_sessions
        fleet_projected = projected_resident_bytes_per_session * concurrent_sessions
        fits_budget = None if budget_bytes is None else fleet_projected <= budget_bytes
        max_sessions_baseline = None if budget_bytes is None else int(budget_bytes // max(baseline_bytes, 1))
        max_sessions_projected = None if budget_bytes is None else int(budget_bytes // max(projected_resident_bytes_per_session, 1))
        capacity_gain = None
        if max_sessions_baseline is not None:
            capacity_gain = float(max_sessions_projected / max(max_sessions_baseline, 1))
        notes = [f"Baseline assumes {runtime_value_bytes} byte(s) per runtime value."]
        if chosen_route == "resident_tier":
            notes.append("Tiered route is selected because it minimizes resident RAM under the requested hot window.")
        if budget_bytes is not None and fits_budget is False:
            notes.append("Projected fleet resident bytes exceed the requested budget.")
        if capacity_gain is not None:
            notes.append(
                f"Under the requested budget, projected session capacity changes from {max_sessions_baseline} to {max_sessions_projected}."
            )

        return ResidentPlan(
            chosen_route=chosen_route,
            concurrent_sessions=concurrent_sessions,
            active_window_tokens=active_window_tokens,
            runtime_value_bytes=runtime_value_bytes,
            budget_bytes=budget_bytes,
            baseline_resident_bytes_per_session=baseline_bytes,
            projected_resident_bytes_per_session=projected_resident_bytes_per_session,
            artifact_bytes_per_session=int(chosen_payload["artifact_bytes_per_session"]),
            resident_savings_bytes_per_session=int(max(baseline_bytes - projected_resident_bytes_per_session, 0)),
            resident_savings_ratio=float(1.0 - projected_resident_bytes_per_session / max(baseline_bytes, 1)),
            fleet_baseline_resident_bytes=fleet_baseline,
            fleet_projected_resident_bytes=fleet_projected,
            fits_budget=fits_budget,
            max_sessions_within_budget_baseline=max_sessions_baseline,
            max_sessions_within_budget_projected=max_sessions_projected,
            capacity_gain_vs_baseline=capacity_gain,
            candidates=candidates,
            page_mode_counts=dict(tier_manifest.stats.page_mode_counts),
            notes=notes,
        )


def build_resident_store(
    array: np.ndarray,
    output_dir: str | Path,
    *,
    config: ResidentTierConfig | None = None,
    protected_vector_indices: Sequence[int] | None = None,
) -> ResidentTierManifest:
    return ResidentTierStore.build(array, output_dir, config=config, protected_vector_indices=protected_vector_indices).manifest


def run_resident_benchmark(
    workloads: dict[str, np.ndarray],
    *,
    config: ResidentTierConfig | None = None,
    concurrent_sessions: int = 8,
    active_window_tokens: int = 256,
    runtime_value_bytes: int = 2,
    slice_iterations: int = 5,
) -> ResidentBenchmarkArtifacts:
    cfg = config or ResidentTierConfig()
    planner = ResidentPlanner(cfg)
    report: dict[str, object] = {
        "meta": {
            "page_size": cfg.page_size,
            "rank": cfg.rank,
            "bits": cfg.bits,
            "group_size": cfg.group_size,
            "hot_pages": cfg.hot_pages,
            "concurrent_sessions": concurrent_sessions,
            "active_window_tokens": active_window_tokens,
            "runtime_value_bytes": runtime_value_bytes,
            "slice_iterations": slice_iterations,
        },
        "workloads": {},
    }

    for workload_name, array in workloads.items():
        plan = planner.plan(
            array,
            concurrent_sessions=concurrent_sessions,
            active_window_tokens=active_window_tokens,
            runtime_value_bytes=runtime_value_bytes,
        )
        with TemporaryDirectory(prefix="hyperquant_resident_store_") as tmpdir:
            store = ResidentTierStore.build(array, tmpdir, config=cfg)
            n_vectors = int(np.prod(array.shape[:-1]))
            window = min(active_window_tokens, n_vectors)
            timings_ms: list[float] = []
            for iteration in range(slice_iterations):
                start = min(iteration * max(window // 2, 1), max(n_vectors - window, 0))
                end = start + window
                import time

                t0 = time.perf_counter()
                restored = store.get_slice(start, end)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                timings_ms.append(elapsed_ms)
                assert restored.shape[0] == end - start
            report["workloads"][workload_name] = {
                "shape": list(array.shape),
                "plan": plan.to_dict(),
                "tiered_store": {
                    "artifact_bytes": store.manifest.stats.artifact_bytes,
                    "manifest_bytes": store.manifest.stats.manifest_bytes,
                    "access_report": store.access_report().to_dict(),
                    "slice_read_mean_ms": float(np.mean(timings_ms)),
                    "slice_read_p95_ms": float(np.percentile(timings_ms, 95)),
                    "page_mode_counts": dict(store.manifest.stats.page_mode_counts),
                    "compression_ratio": float(store.manifest.stats.compression_ratio),
                    "rms_error": float(store.manifest.stats.rms_error),
                    "cosine_similarity": float(store.manifest.stats.cosine_similarity),
                },
            }
    return ResidentBenchmarkArtifacts(report=report)


__all__ = [
    "ResidentPageMode",
    "ResidentTierConfig",
    "ResidentPageDescriptor",
    "ResidentTierStats",
    "ResidentTierManifest",
    "ResidentPlan",
    "ResidentAccessReport",
    "ResidentBenchmarkArtifacts",
    "ResidentTierStore",
    "ResidentPlanner",
    "build_resident_store",
    "run_resident_benchmark",
]
