# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CodebookBundle:
    codebook: np.ndarray
    rotation: np.ndarray
    chunk_size: int
    normalize: bool = True
    version: str = "0.6.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.codebook = np.asarray(self.codebook, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        if self.codebook.ndim != 2:
            raise ValueError("codebook must be a 2D array")
        if self.rotation.ndim != 2:
            raise ValueError("rotation must be a 2D array")
        if self.rotation.shape != (self.chunk_size, self.chunk_size):
            raise ValueError(
                "rotation shape must match (chunk_size, chunk_size)"
            )
        if self.codebook.shape[1] != self.chunk_size:
            raise ValueError("codebook second dimension must equal chunk_size")

    @property
    def codebook_size(self) -> int:
        return int(self.codebook.shape[0])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            codebook=self.codebook.astype(np.float32),
            rotation=self.rotation.astype(np.float32),
            chunk_size=np.asarray(self.chunk_size, dtype=np.int64),
            normalize=np.asarray(int(self.normalize), dtype=np.int64),
            version=np.asarray(self.version),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CodebookBundle":
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            metadata_raw = str(payload["metadata"].item())
            metadata = json.loads(metadata_raw) if metadata_raw else {}
            return cls(
                codebook=payload["codebook"].astype(np.float32),
                rotation=payload["rotation"].astype(np.float32),
                chunk_size=int(payload["chunk_size"].item()),
                normalize=bool(int(payload["normalize"].item())),
                version=str(payload["version"].item()),
                metadata=metadata,
            )
