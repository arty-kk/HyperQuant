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

from hyperquant.vector_codec import (
    DenseRotationBaseline,
    VectorCodec,
    RotatedScalarEnvelope,
    native_fwht_status,
)


def build_vector_stream(rows: int = 1024, dim: int = 128, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((rows, dim)).astype(np.float32)
    coeff = rng.standard_normal((rows, 8)).astype(np.float32)
    basis = rng.standard_normal((8, dim)).astype(np.float32)
    return 0.65 * base + 0.35 * (coeff @ basis) / np.sqrt(np.float32(8.0))


def build_low_rankish_stream(rows: int = 1024, dim: int = 128, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    topic = rng.standard_normal((1, dim)).astype(np.float32)
    coeff = rng.standard_normal((rows, 4)).astype(np.float32)
    basis = rng.standard_normal((4, dim)).astype(np.float32)
    noise = 0.01 * rng.standard_normal((rows, dim)).astype(np.float32)
    return (topic + coeff @ basis + noise).astype(np.float32)


def test_vector_codec_roundtrip_baseline_and_vector() -> None:
    array = build_vector_stream()
    dense = DenseRotationBaseline(bits=3, group_size=128, residual_topk=1)
    vector = VectorCodec(bits=3, group_size=128, prefer_native_fwht=True, residual_topk=1)

    dense_env, dense_stats = dense.compress(array)
    vector_env, vector_stats = vector.compress(array)

    dense_restored = dense.decompress(dense_env)
    vector_restored = vector.decompress(vector_env)

    assert dense_restored.shape == array.shape
    assert vector_restored.shape == array.shape
    assert dense_stats.compression_ratio > 5.0
    assert vector_stats.compression_ratio > 5.0
    assert dense_stats.cosine_similarity > 0.97
    assert vector_stats.cosine_similarity > 0.97
    assert abs(vector_stats.rms_error - dense_stats.rms_error) < 0.03
    assert vector_stats.transform == "structured_fwht"
    assert dense_stats.transform == "dense_qr"
    assert vector_stats.residual_topk == 1
    assert vector_stats.effective_bits_per_value <= 3.5


def test_vector_codec_serialization_and_padding() -> None:
    array = build_vector_stream(rows=257, dim=96)
    vector = VectorCodec(bits=4, group_size=64, prefer_native_fwht=False, residual_topk=1)

    envelope, stats = vector.compress(array)
    payload = envelope.to_bytes()
    restored_envelope = RotatedScalarEnvelope.from_bytes(payload)
    restored = vector.decompress(restored_envelope)

    assert restored.shape == array.shape
    assert restored_envelope.padded_dim == 128
    assert restored_envelope.residual_topk == 1
    assert stats.compression_ratio > 4.0
    np.testing.assert_allclose(restored, vector.decompress(envelope), rtol=0, atol=0)


def test_residual_rescue_improves_quality_on_structured_vectors() -> None:
    array = build_low_rankish_stream()
    plain = VectorCodec(bits=3, group_size=128, prefer_native_fwht=False, residual_topk=0)
    rescued = VectorCodec(bits=3, group_size=128, prefer_native_fwht=False, residual_topk=1)

    _, plain_stats = plain.compress(array)
    rescued_env, rescued_stats = rescued.compress(array)

    assert rescued_stats.rms_error < plain_stats.rms_error
    assert rescued_stats.cosine_similarity >= plain_stats.cosine_similarity
    assert rescued_stats.effective_bits_per_value > plain_stats.effective_bits_per_value
    assert rescued.decompress(rescued_env).shape == array.shape


def test_native_fwht_status_shape() -> None:
    status = native_fwht_status(auto_build=False)
    assert set(status) == {"available", "path", "error"}
    assert isinstance(status["available"], bool)
