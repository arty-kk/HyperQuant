# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from hyperquant.codebook import MiniBatchKMeansTrainer
from hyperquant.codebook_codec import CodebookEnvelope, CodebookCodec
from hyperquant.config import CodebookConfig


def build_demo_array() -> np.ndarray:
    rng = np.random.default_rng(123)
    base = rng.standard_normal((16, 128)).astype(np.float32)
    weights = rng.standard_normal((256, 16)).astype(np.float32)
    return weights @ base + 0.01 * rng.standard_normal((256, 128)).astype(np.float32)


def test_roundtrip_and_serialization() -> None:
    array = build_demo_array()
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=64, sample_size=2048, training_iterations=6))
    bundle = trainer.train(array)
    compressor = CodebookCodec(bundle)

    envelope, stats = compressor.compress(array)
    restored = compressor.decompress(envelope)

    assert restored.shape == array.shape
    assert stats.compression_ratio > 1.0
    assert stats.storage_compression_ratio > 1.0
    assert stats.cosine_similarity > 0.98

    payload = envelope.to_bytes()
    restored_envelope = CodebookEnvelope.from_bytes(payload)
    restored_again = compressor.decompress(restored_envelope)
    np.testing.assert_allclose(restored, restored_again, rtol=0, atol=0)


def test_rejects_non_float_input() -> None:
    array = np.arange(256 * 128, dtype=np.int32).reshape(256, 128)
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=64, sample_size=2048, training_iterations=6))
    bundle = trainer.train(build_demo_array())
    compressor = CodebookCodec(bundle)
    with pytest.raises(ValueError, match="unsupported dtype"):
        compressor.compress(array)
