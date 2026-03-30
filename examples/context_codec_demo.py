# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperquant.guarantee import GuaranteeMode, ContextGuaranteeProfile
from hyperquant.context_codec import ContextCodecConfig, ContextCodec


def build_context_like_array(
    n_tokens: int = 4096,
    dim: int = 128,
    page_size: int = 64,
    prefix_pages: int = 4,
    repeat_pages: int = 12,
    recent_tokens: int = 64,
    seed: int = 123,
) -> np.ndarray:
    """
    Synthetic workload designed to resemble long-context production prompts:
    - repeated policy/instruction pages (page_ref wins hard),
    - large topical middle blocks (rank-1 page structure),
    - small noisy recent tail (kept in fp16).
    """
    rng = np.random.default_rng(seed)
    n_pages = math.ceil(n_tokens / page_size)
    pages: list[np.ndarray] = []
    templates: list[np.ndarray] = []

    for _ in range(prefix_pages):
        topic = rng.standard_normal((1, dim)).astype(np.float32)
        coeff = rng.standard_normal((page_size, 1)).astype(np.float32) * 0.7
        basis = rng.standard_normal((1, dim)).astype(np.float32)
        page = topic + coeff @ basis + 0.002 * rng.standard_normal((page_size, dim)).astype(np.float32)
        templates.append(page.astype(np.float32))

    for page_idx in range(n_pages):
        if page_idx < prefix_pages:
            pages.append(templates[page_idx])
        elif page_idx < prefix_pages + repeat_pages:
            pages.append(templates[page_idx % prefix_pages].copy())
        elif page_idx >= n_pages - math.ceil(recent_tokens / page_size):
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 3)).astype(np.float32)
            basis = rng.standard_normal((3, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.03 * rng.standard_normal((page_size, dim)).astype(np.float32)
            pages.append(page.astype(np.float32))
        else:
            topic = rng.standard_normal((1, dim)).astype(np.float32)
            coeff = rng.standard_normal((page_size, 1)).astype(np.float32)
            basis = rng.standard_normal((1, dim)).astype(np.float32)
            page = topic + coeff @ basis + 0.008 * rng.standard_normal((page_size, dim)).astype(np.float32)
            pages.append(page.astype(np.float32))

    return np.concatenate(pages, axis=0)[:n_tokens].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-demo-data", help="Optional path to write the synthetic .npy workload.")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()

    array = build_context_like_array(
        n_tokens=args.tokens,
        dim=args.dim,
        page_size=args.page_size,
    )

    if args.write_demo_data:
        path = Path(args.write_demo_data)
        np.save(path, array, allow_pickle=False)
        print(f"wrote synthetic Context-like workload to {path}")

    compressor = ContextCodec(
        ContextCodecConfig(
            page_size=args.page_size,
            rank=1,
            prefix_keep_vectors=32,
            suffix_keep_vectors=64,
            low_rank_error_threshold=0.03,
            ref_round_decimals=3,
            enable_page_ref=True,
        )
    )
    _, stats = compressor.compress(
        array,
        guarantee_profile=ContextGuaranteeProfile(),
        guarantee_mode=GuaranteeMode.FAIL_CLOSED,
    )
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
