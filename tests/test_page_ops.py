# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from hyperquant.page_ops import protected_mask, quantize_page_int8, relative_rms, top_rank_factors


def test_protected_mask_marks_prefix_suffix_and_explicit_indices() -> None:
    mask = protected_mask(8, [3], prefix_keep_vectors=1, suffix_keep_vectors=2)
    assert mask.tolist() == [True, False, False, True, False, False, True, True]


def test_protected_mask_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="outside valid range"):
        protected_mask(4, [4], prefix_keep_vectors=0, suffix_keep_vectors=0)


def test_top_rank_factors_reconstructs_rank_one_matrix() -> None:
    matrix = np.outer(np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([2.0, -1.0], dtype=np.float32))
    us, vt = top_rank_factors(matrix, rank=1)
    reconstructed = us @ vt
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-5, rtol=0)


def test_quantize_page_int8_returns_reasonable_error() -> None:
    page = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(4, 4)
    mins, scales, q_page, recon_page, rel_rms, max_abs = quantize_page_int8(page, valid_length=4, page_size=4)
    assert mins.shape == (4,)
    assert scales.shape == (4,)
    assert q_page.shape == (4, 4)
    assert recon_page.shape == (4, 4)
    assert rel_rms == pytest.approx(relative_rms(page, recon_page), abs=1e-6)
    assert max_abs < 0.02
