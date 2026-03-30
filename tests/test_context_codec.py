# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import numpy as np
import pytest

from hyperquant.guarantee import ContourViolation, GuaranteeMode, GuaranteeViolation, ContextGuaranteeProfile
from hyperquant.context_codec import (
    ContextEnvelope,
    ContextCodecConfig,
    ContextCodec,
)
from hyperquant.utils import bytes_to_b64
from tests.array_builders import build_context_like_array, build_random_array


def test_context_roundtrip_and_ratio() -> None:
    array = build_context_like_array()
    compressor = ContextCodec(
        ContextCodecConfig(
            page_size=64,
            rank=1,
            prefix_keep_vectors=32,
            suffix_keep_vectors=64,
            low_rank_error_threshold=0.03,
            ref_round_decimals=3,
            enable_page_ref=True,
        )
    )

    envelope, stats = compressor.compress(
        array,
        guarantee_profile=ContextGuaranteeProfile(),
        guarantee_mode=GuaranteeMode.FAIL_CLOSED,
    )
    restored = compressor.decompress(envelope)

    assert restored.shape == array.shape
    assert stats.compression_ratio >= 30.0
    assert stats.storage_compression_ratio >= 30.0
    assert stats.cosine_similarity > 0.999
    assert stats.guarantee_passed is True
    assert stats.page_mode_counts["low_rank"] > 0
    assert stats.page_mode_counts["page_ref"] > 0
    assert "int8" in stats.page_mode_counts
    assert stats.contour == "context_structured"
    assert stats.contour_supported is True
    assert stats.route_recommendation == "context_codec"
    assert len(stats.payload_sha256) == 64

    payload = envelope.to_bytes()
    restored_envelope = ContextEnvelope.from_bytes(payload)
    restored_again = compressor.decompress(restored_envelope)
    np.testing.assert_allclose(restored, restored_again, rtol=0, atol=0)


def test_context_fail_closed_rejects_non_context_like_input() -> None:
    array = build_random_array()
    compressor = ContextCodec(ContextCodecConfig())

    with pytest.raises(ContourViolation):
        compressor.compress(
            array,
            guarantee_profile=ContextGuaranteeProfile(),
            guarantee_mode=GuaranteeMode.FAIL_CLOSED,
        )


def test_context_rejects_non_finite_input() -> None:
    array = build_context_like_array()
    array[0, 0] = np.nan
    compressor = ContextCodec(ContextCodecConfig())
    with pytest.raises(ValueError, match="NaN or Inf"):
        compressor.compress(array)


def test_context_envelope_validation_rejects_invalid_reference() -> None:
    array = build_context_like_array()
    compressor = ContextCodec(ContextCodecConfig())
    envelope, _ = compressor.compress(array, guarantee_mode=GuaranteeMode.ALLOW_BEST_EFFORT)
    raw = envelope.to_bytes()
    loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    page_ref_indices = loaded["page_ref_indices"].astype(np.int32)
    page_ref_indices[0] = 123
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        original_shape=loaded["original_shape"],
        original_dtype=loaded["original_dtype"],
        page_size=loaded["page_size"],
        rank=loaded["rank"],
        schema_version=loaded["schema_version"],
        page_modes=loaded["page_modes"],
        page_lengths=loaded["page_lengths"],
        page_ref_indices=page_ref_indices,
        low_rank_means=loaded["low_rank_means"],
        low_rank_us=loaded["low_rank_us"],
        low_rank_vt=loaded["low_rank_vt"],
        int8_mins=loaded["int8_mins"],
        int8_scales=loaded["int8_scales"],
        int8_pages=loaded["int8_pages"],
        fp16_pages=loaded["fp16_pages"],
    )
    with pytest.raises(ValueError, match="page_ref_indices"):
        ContextEnvelope.from_base64(bytes_to_b64(buffer.getvalue()))
