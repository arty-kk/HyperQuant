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

import base64
import hashlib
import io
import json
from typing import Any

import numpy as np


EPS = 1e-8


def ndarray_to_b64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ndarray_from_b64(value: str, *, max_bytes: int | None = None) -> np.ndarray:
    raw = bytes_from_b64(value, max_bytes=max_bytes)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def bytes_to_b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def bytes_from_b64(value: str, *, max_bytes: int | None = None) -> bytes:
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:  # pragma: no cover - validation detail depends on stdlib
        raise ValueError("invalid base64 payload") from exc
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError(f"decoded payload exceeds max_bytes={max_bytes}")
    return raw


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
