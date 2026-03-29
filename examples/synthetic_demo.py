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

import argparse
from pathlib import Path

import numpy as np


def synthetic_kv_like_data(
    n_vectors: int = 2048,
    dim: int = 128,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((32, dim)).astype(np.float32)
    coeffs = rng.standard_normal((n_vectors, 32)).astype(np.float32)
    signal = coeffs @ basis
    noise = 0.03 * rng.standard_normal((n_vectors, dim)).astype(np.float32)
    return signal + noise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-demo-data", default="")
    args = parser.parse_args()

    data = synthetic_kv_like_data()
    if args.write_demo_data:
        target = Path(args.write_demo_data)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, data, allow_pickle=False)
        print(f"wrote demo data to {target}")
    else:
        print(data.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
