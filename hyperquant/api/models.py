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

from typing import List, Literal

from pydantic import BaseModel, Field

from ..defaults import (
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
    RESIDENT_ACTIVE_WINDOW_TOKENS_DEFAULT,
    RESIDENT_ALLOW_VECTOR_FOR_PROTECTED_DEFAULT,
    RESIDENT_CONCURRENT_SESSIONS_DEFAULT,
    RESIDENT_HOT_PAGES_DEFAULT,
    RESIDENT_RUNTIME_VALUE_BYTES_DEFAULT,
    VECTOR_BITS_DEFAULT,
    VECTOR_GROUP_SIZE_DEFAULT,
    VECTOR_PREFER_NATIVE_FWHT_DEFAULT,
    VECTOR_RESIDUAL_TOPK_DEFAULT,
    VECTOR_ROTATION_SEED_DEFAULT,
)


class CodebookCompressRequest(BaseModel):
    array_b64: str = Field(..., description="Base64-encoded .npy payload.")
    protected_vector_indices: List[int] = Field(default_factory=list)


class DecompressRequest(BaseModel):
    envelope_b64: str


class VectorCompressRequest(BaseModel):
    array_b64: str = Field(..., description="Base64-encoded .npy payload.")
    bits: int = Field(default=VECTOR_BITS_DEFAULT, ge=2, le=4)
    group_size: int = Field(default=VECTOR_GROUP_SIZE_DEFAULT, gt=0)
    rotation_seed: int = Field(default=VECTOR_ROTATION_SEED_DEFAULT)
    residual_topk: int = Field(default=VECTOR_RESIDUAL_TOPK_DEFAULT, ge=0)
    prefer_native_fwht: bool = VECTOR_PREFER_NATIVE_FWHT_DEFAULT


class ContextGuaranteeModel(BaseModel):
    min_compression_ratio: float = Field(default=30.0, gt=0)
    min_cosine_similarity: float = Field(default=0.999, ge=0.0, le=1.0)
    max_rms_error: float = Field(default=0.010, ge=0.0)
    max_max_abs_error: float | None = Field(default=0.050, ge=0.0)


class ContextCompressRequest(BaseModel):
    array_b64: str = Field(..., description="Base64-encoded .npy payload.")
    protected_vector_indices: List[int] = Field(default_factory=list)
    page_size: int = Field(default=CONTEXT_PAGE_SIZE_DEFAULT, gt=0)
    rank: int = Field(default=CONTEXT_RANK_DEFAULT, gt=0)
    prefix_keep_vectors: int = Field(default=CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT, ge=0)
    suffix_keep_vectors: int = Field(default=CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT, ge=0)
    low_rank_error_threshold: float = Field(default=CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT, ge=0.0)
    ref_round_decimals: int = Field(default=CONTEXT_REF_ROUND_DECIMALS_DEFAULT, ge=0)
    enable_page_ref: bool = CONTEXT_ENABLE_PAGE_REF_DEFAULT
    page_ref_rel_rms_threshold: float = Field(default=CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT, ge=0.0)
    enable_int8_fallback: bool = CONTEXT_ENABLE_INT8_FALLBACK_DEFAULT
    try_int8_for_protected: bool = CONTEXT_TRY_INT8_FOR_PROTECTED_DEFAULT
    int8_rel_rms_threshold: float = Field(default=CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT, ge=0.0)
    int8_max_abs_threshold: float = Field(default=CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT, ge=0.0)
    fail_closed: bool = True
    guarantee: ContextGuaranteeModel | None = None


class CodebookCompressionStatsModel(BaseModel):
    original_bytes: int
    stored_bytes: int
    serialized_bytes: int
    compression_ratio: float
    storage_compression_ratio: float
    rms_error: float
    max_abs_error: float
    cosine_similarity: float
    mode_counts: dict[str, int]


class VectorCompressionStatsModel(BaseModel):
    algorithm: str
    transform: Literal["dense_qr", "structured_fwht"]
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
    payload_sha256: str


class ContextCompressionStatsModel(BaseModel):
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
    payload_sha256: str
    guarantee_passed: bool | None = None
    guarantee_failures: list[str] = Field(default_factory=list)
    guarantee_profile: dict[str, object] | None = None
    contour: Literal["context_structured", "generic_conservative", "reject"]
    contour_supported: bool | None = None
    contour_reasons: list[str] = Field(default_factory=list)
    contour_details: dict[str, object] | None = None
    route_recommendation: Literal["context_codec", "conservative_codebook", "reject"]


class ResidentPlanRequest(BaseModel):
    array_b64: str = Field(..., description="Base64-encoded .npy payload.")
    concurrent_sessions: int = Field(default=RESIDENT_CONCURRENT_SESSIONS_DEFAULT, gt=0)
    active_window_tokens: int = Field(default=RESIDENT_ACTIVE_WINDOW_TOKENS_DEFAULT, gt=0)
    runtime_value_bytes: int = Field(default=RESIDENT_RUNTIME_VALUE_BYTES_DEFAULT, gt=0)
    budget_bytes: int | None = Field(default=None, gt=0)
    page_size: int = Field(default=CONTEXT_PAGE_SIZE_DEFAULT, gt=0)
    rank: int = Field(default=CONTEXT_RANK_DEFAULT, gt=0)
    bits: int = Field(default=VECTOR_BITS_DEFAULT, ge=2, le=4)
    group_size: int = Field(default=VECTOR_GROUP_SIZE_DEFAULT, gt=0)
    hot_pages: int = Field(default=RESIDENT_HOT_PAGES_DEFAULT, gt=0)
    rotation_seed: int = Field(default=VECTOR_ROTATION_SEED_DEFAULT)
    residual_topk: int = Field(default=VECTOR_RESIDUAL_TOPK_DEFAULT, ge=0)
    prefix_keep_vectors: int = Field(default=CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT, ge=0)
    suffix_keep_vectors: int = Field(default=CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT, ge=0)
    low_rank_error_threshold: float = Field(default=CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT, ge=0.0)
    ref_round_decimals: int = Field(default=CONTEXT_REF_ROUND_DECIMALS_DEFAULT, ge=0)
    enable_page_ref: bool = CONTEXT_ENABLE_PAGE_REF_DEFAULT
    page_ref_rel_rms_threshold: float = Field(default=CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT, ge=0.0)
    enable_int8_fallback: bool = CONTEXT_ENABLE_INT8_FALLBACK_DEFAULT
    try_int8_for_protected: bool = CONTEXT_TRY_INT8_FOR_PROTECTED_DEFAULT
    int8_rel_rms_threshold: float = Field(default=CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT, ge=0.0)
    int8_max_abs_threshold: float = Field(default=CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT, ge=0.0)
    prefer_native_fwht: bool = VECTOR_PREFER_NATIVE_FWHT_DEFAULT
    allow_vector_for_protected: bool = RESIDENT_ALLOW_VECTOR_FOR_PROTECTED_DEFAULT


class ResidentPlanResponse(BaseModel):
    plan: dict[str, object]


class CodebookCompressResponse(BaseModel):
    envelope_b64: str
    stats: CodebookCompressionStatsModel


class VectorCompressResponse(BaseModel):
    envelope_b64: str
    stats: VectorCompressionStatsModel


class ContextCompressResponse(BaseModel):
    envelope_b64: str
    stats: ContextCompressionStatsModel


class DecompressResponse(BaseModel):
    array_b64: str


class HealthResponse(BaseModel):
    status: str
    version: str
    chunk_size: int
    codebook_size: int
    normalize: bool
    max_request_bytes: int
    max_http_body_bytes: int
    max_concurrency: int
    native_fwht_available: bool
    native_fwht_path: str | None = None
    native_fwht_error: str | None = None
    routes: list[str] = Field(default_factory=list)
