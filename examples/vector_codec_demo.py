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
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperquant.live_data import generate_online_vector_stream
from hyperquant.vector_codec import VectorCodec, native_fwht_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo for the Vector codec.")
    parser.add_argument("--output", type=Path, help="Optional path to save the generated vector stream as .npy")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260329)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vectors = generate_online_vector_stream(n_vectors=args.rows, dim=args.dim, seed=args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, vectors, allow_pickle=False)

    compressor = VectorCodec(bits=args.bits, group_size=args.group_size)
    envelope, stats = compressor.compress(vectors)
    restored = compressor.decompress(envelope)
    report = {
        "shape": list(vectors.shape),
        "native_fwht": native_fwht_status(auto_build=False),
        "stats": stats.to_dict(),
        "max_restore_delta": float(np.max(np.abs(restored.astype(np.float32) - vectors.astype(np.float32)))),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
