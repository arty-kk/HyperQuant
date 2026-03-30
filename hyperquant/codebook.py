# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import numpy as np

from .bundle import CodebookBundle
from .config import CodebookConfig
from .utils import EPS


def orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs[np.newaxis, :]
    return q.astype(np.float32)


def split_into_chunks(vectors: np.ndarray, chunk_size: int) -> Tuple[np.ndarray, tuple[int, ...], int]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("expected at least a 2D array shaped (..., dim)")
    dim = int(array.shape[-1])
    if dim % chunk_size != 0:
        raise ValueError(
            f"last dimension {dim} must be divisible by chunk_size {chunk_size}"
        )
    n_vectors = int(np.prod(array.shape[:-1]))
    vectors_2d = array.reshape(n_vectors, dim)
    chunks_per_vector = dim // chunk_size
    chunks = vectors_2d.reshape(n_vectors, chunks_per_vector, chunk_size)
    return chunks.reshape(-1, chunk_size), tuple(array.shape), chunks_per_vector


def merge_from_chunks(
    chunks: np.ndarray,
    original_shape: tuple[int, ...],
    chunk_size: int,
) -> np.ndarray:
    dim = int(original_shape[-1])
    chunks_per_vector = dim // chunk_size
    n_vectors = int(np.prod(original_shape[:-1]))
    merged = np.asarray(chunks, dtype=np.float32).reshape(
        n_vectors, chunks_per_vector, chunk_size
    )
    return merged.reshape(original_shape)


class MiniBatchKMeansTrainer:
    def __init__(self, config: CodebookConfig) -> None:
        config.validate()
        self.config = config

    def _prepare_chunks(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunks, _, _ = split_into_chunks(vectors, self.config.chunk_size)
        rotation = orthogonal_matrix(self.config.chunk_size, self.config.rotation_seed)
        rotated = chunks @ rotation
        if self.config.normalize:
            norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            rotated = rotated / np.maximum(norms, EPS)
        return rotated.astype(np.float32), rotation

    def _sample(self, chunks: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.config.rotation_seed)
        if len(chunks) <= self.config.sample_size:
            return chunks
        indices = rng.choice(len(chunks), size=self.config.sample_size, replace=False)
        return chunks[indices]

    @staticmethod
    def _pairwise_sq_distance(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        data_norm = np.sum(data * data, axis=1, keepdims=True)
        cent_norm = np.sum(centroids * centroids, axis=1, keepdims=True).T
        distances = data_norm + cent_norm - 2.0 * (data @ centroids.T)
        return np.maximum(distances, 0.0)

    def _init_kmeans_pp(self, data: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.config.rotation_seed)
        k = self.config.codebook_size
        if len(data) < k:
            raise ValueError(
                "training sample smaller than codebook_size; increase data or reduce codebook_size"
            )
        centroids = np.empty((k, data.shape[1]), dtype=np.float32)
        first = int(rng.integers(0, len(data)))
        centroids[0] = data[first]
        distances = np.sum((data - centroids[0]) ** 2, axis=1)
        for i in range(1, k):
            total = float(distances.sum())
            if total <= 0:
                centroids[i:] = data[rng.choice(len(data), size=k - i, replace=False)]
                break
            probs = distances / total
            next_idx = int(rng.choice(len(data), p=probs))
            centroids[i] = data[next_idx]
            candidate_dist = np.sum((data - centroids[i]) ** 2, axis=1)
            distances = np.minimum(distances, candidate_dist)
        return centroids

    def train(self, vectors: np.ndarray) -> CodebookBundle:
        prepared, rotation = self._prepare_chunks(vectors)
        data = self._sample(prepared)
        centroids = self._init_kmeans_pp(data)
        rng = np.random.default_rng(self.config.rotation_seed + 1)
        batch_size = min(4096, len(data))

        for _ in range(self.config.training_iterations):
            if len(data) > batch_size:
                batch_idx = rng.choice(len(data), size=batch_size, replace=False)
                batch = data[batch_idx]
            else:
                batch = data
            distances = self._pairwise_sq_distance(batch, centroids)
            assignment = distances.argmin(axis=1)

            for centroid_idx in range(self.config.codebook_size):
                mask = assignment == centroid_idx
                if not np.any(mask):
                    continue
                centroids[centroid_idx] = batch[mask].mean(axis=0)

            empty = []
            distances_full = self._pairwise_sq_distance(data, centroids)
            assignment_full = distances_full.argmin(axis=1)
            for centroid_idx in range(self.config.codebook_size):
                if not np.any(assignment_full == centroid_idx):
                    empty.append(centroid_idx)
            if empty:
                replacement = rng.choice(len(data), size=len(empty), replace=False)
                centroids[np.array(empty, dtype=np.int64)] = data[replacement]

        return CodebookBundle(
            codebook=centroids.astype(np.float32),
            rotation=rotation.astype(np.float32),
            chunk_size=self.config.chunk_size,
            normalize=self.config.normalize,
            metadata={
                "trainer": self.__class__.__name__,
                **asdict(self.config),
            },
        )
