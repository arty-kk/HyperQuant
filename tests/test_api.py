# SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io

import httpx
import numpy as np
import pytest

from hyperquant.api import app as api_app
from hyperquant.api.app import create_app
from hyperquant.bundle import CodebookBundle
from hyperquant.codebook import MiniBatchKMeansTrainer
from hyperquant.config import CodebookConfig
from hyperquant.context_codec import ContextEnvelope
from hyperquant.utils import bytes_to_b64, ndarray_from_b64, ndarray_to_b64
from tests.array_builders import build_context_like_array, build_random_array


def build_demo_array() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((64, 128)).astype(np.float32)


async def _scenario_client(app, fn):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await fn(client)


def test_api_roundtrip_and_context_contract(tmp_path) -> None:
    array = build_demo_array()
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=32, sample_size=1024, training_iterations=4))
    bundle = trainer.train(array)
    bundle_path = tmp_path / "bundle.npz"
    bundle.save(bundle_path)
    app = create_app(bundle_path, max_request_bytes=4 * 1024 * 1024)

    async def scenario(client: httpx.AsyncClient) -> None:
        health = await client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["max_request_bytes"] == 4 * 1024 * 1024
        assert health.json()["max_http_body_bytes"] >= health.json()["max_request_bytes"]
        assert health.json()["max_concurrency"] >= 1

        compress_response = await client.post(
            "/v1/codebook/compress",
            json={"array_b64": ndarray_to_b64(array), "protected_vector_indices": [0]},
        )
        assert compress_response.status_code == 200
        compress_payload = compress_response.json()
        assert compress_payload["stats"]["compression_ratio"] > 1.0
        assert compress_payload["stats"]["storage_compression_ratio"] > 1.0

        decompress_response = await client.post(
            "/v1/codebook/decompress",
            json={"envelope_b64": compress_payload["envelope_b64"]},
        )
        assert decompress_response.status_code == 200
        restored = ndarray_from_b64(decompress_response.json()["array_b64"])
        assert restored.shape == array.shape

        context_good = build_context_like_array()
        context_response = await client.post(
            "/v1/context/compress",
            json={
                "array_b64": ndarray_to_b64(context_good),
                "fail_closed": True,
                "guarantee": {
                    "min_compression_ratio": 30.0,
                    "min_cosine_similarity": 0.999,
                    "max_rms_error": 0.01,
                    "max_max_abs_error": 0.05,
                },
            },
        )
        assert context_response.status_code == 200
        context_payload = context_response.json()
        assert context_payload["stats"]["guarantee_passed"] is True
        assert context_payload["stats"]["compression_ratio"] >= 30.0
        assert context_payload["stats"]["contour"] == "context_structured"
        assert context_payload["stats"]["route_recommendation"] == "context_codec"
        assert "int8" in context_payload["stats"]["page_mode_counts"]

        context_decompress = await client.post(
            "/v1/context/decompress",
            json={"envelope_b64": context_payload["envelope_b64"]},
        )
        assert context_decompress.status_code == 200
        restored_context = ndarray_from_b64(context_decompress.json()["array_b64"])
        assert restored_context.shape == context_good.shape

        context_bad = build_random_array()
        context_bad_response = await client.post(
            "/v1/context/compress",
            json={
                "array_b64": ndarray_to_b64(context_bad),
                "fail_closed": True,
                "guarantee": {
                    "min_compression_ratio": 30.0,
                    "min_cosine_similarity": 0.999,
                    "max_rms_error": 0.01,
                    "max_max_abs_error": 0.05,
                },
            },
        )
        assert context_bad_response.status_code == 422
        assert context_bad_response.json()["detail"]["type"] == "contour_violation"

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert "hyperquant_requests_total" in metrics.text
        assert "hyperquant_guarantee_total" in metrics.text
        assert "hyperquant_contour_total" in metrics.text
        assert "hyperquant_request_latency_seconds" in metrics.text
        assert "hyperquant_errors_total" in metrics.text

    asyncio.run(_scenario_client(app, scenario))


def test_api_rejects_malformed_context_payload(tmp_path) -> None:
    array = build_demo_array()
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=32, sample_size=1024, training_iterations=4))
    bundle = trainer.train(array)
    bundle_path = tmp_path / "bundle.npz"
    bundle.save(bundle_path)
    app = create_app(bundle_path, max_request_bytes=4 * 1024 * 1024)

    async def scenario(client: httpx.AsyncClient) -> None:
        context_good = build_context_like_array()
        context_response = await client.post(
            "/v1/context/compress",
            json={"array_b64": ndarray_to_b64(context_good), "fail_closed": False},
        )
        payload = context_response.json()["envelope_b64"]
        envelope = ContextEnvelope.from_base64(payload, max_bytes=4 * 1024 * 1024)
        raw = envelope.to_bytes()
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
        page_ref_indices = loaded["page_ref_indices"].astype(np.int32)
        page_ref_indices[0] = 99
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
        resp = await client.post("/v1/context/decompress", json={"envelope_b64": bytes_to_b64(buffer.getvalue())})
        assert resp.status_code == 400

    asyncio.run(_scenario_client(app, scenario))


def test_api_vector_roundtrip_and_health_routes(tmp_path) -> None:
    array = build_demo_array()
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=32, sample_size=1024, training_iterations=4))
    bundle = trainer.train(array)
    bundle_path = tmp_path / "bundle.npz"
    bundle.save(bundle_path)
    app = create_app(bundle_path, max_request_bytes=4 * 1024 * 1024)

    async def scenario(client: httpx.AsyncClient) -> None:
        health = await client.get("/healthz")
        assert health.status_code == 200
        payload = health.json()
        assert "vector" in payload["routes"]
        assert "native_fwht_available" in payload

        vector_response = await client.post(
            "/v1/vector/compress",
            json={
                "array_b64": ndarray_to_b64(array),
                "bits": 3,
                "group_size": 128,
                "rotation_seed": 17,
                "residual_topk": 1,
                "prefer_native_fwht": True,
            },
        )
        assert vector_response.status_code == 200
        vector_payload = vector_response.json()
        assert vector_payload["stats"]["compression_ratio"] > 1.0
        assert vector_payload["stats"]["effective_bits_per_value"] <= 3.5
        assert vector_payload["stats"]["transform"] == "structured_fwht"
        assert vector_payload["stats"]["residual_topk"] == 1

        vector_decompress = await client.post(
            "/v1/vector/decompress",
            json={"envelope_b64": vector_payload["envelope_b64"]},
        )
        assert vector_decompress.status_code == 200
        restored = ndarray_from_b64(vector_decompress.json()["array_b64"])
        assert restored.shape == array.shape

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert 'endpoint="vector_compress"' in metrics.text
        assert 'endpoint="vector_decompress"' in metrics.text

    asyncio.run(_scenario_client(app, scenario))


def test_api_resident_plan_endpoint(tmp_path) -> None:
    array = build_random_array(n_tokens=4096, dim=64)
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=32, sample_size=1024, training_iterations=4))
    bundle = trainer.train(build_demo_array())
    bundle_path = tmp_path / "bundle.npz"
    bundle.save(bundle_path)
    app = create_app(bundle_path, max_request_bytes=8 * 1024 * 1024)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/resident/plan",
            json={
                "array_b64": ndarray_to_b64(array),
                "concurrent_sessions": 4,
                "active_window_tokens": 64,
                "runtime_value_bytes": 2,
                "page_size": 32,
                "group_size": 64,
                "hot_pages": 4,
                "residual_topk": 1,
            },
        )
        assert response.status_code == 200
        payload = response.json()["plan"]
        assert payload["chosen_route"] == "resident_tier"
        assert payload["resident_savings_ratio"] > 0.7
        assert "resident_tier" in payload["candidates"]

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert 'endpoint="resident_plan"' in metrics.text
        assert "hyperquant_last_projected_resident_ratio" in metrics.text

    asyncio.run(_scenario_client(app, scenario))


def test_api_returns_500_for_internal_worker_error(tmp_path, monkeypatch) -> None:
    array = build_demo_array()
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=32, sample_size=1024, training_iterations=4))
    bundle = trainer.train(array)
    bundle_path = tmp_path / "bundle.npz"
    bundle.save(bundle_path)

    def raising_ndarray_from_b64(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_app, "ndarray_from_b64", raising_ndarray_from_b64)
    app = create_app(bundle_path, max_request_bytes=4 * 1024 * 1024)

    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/codebook/compress",
            json={"array_b64": ndarray_to_b64(array)},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "internal server error"
        metrics = await client.get("/metrics")
        assert 'endpoint="compress",reason="internal_error"' in metrics.text

    asyncio.run(_scenario_client(app, scenario))


@pytest.mark.parametrize("headers", [{}, {"content-length": "invalid"}, {"content-length": "10"}])
def test_api_rejects_oversized_body_without_reliable_content_length(tmp_path, headers) -> None:
    array = build_demo_array()
    trainer = MiniBatchKMeansTrainer(CodebookConfig(chunk_size=32, codebook_size=32, sample_size=1024, training_iterations=4))
    bundle = trainer.train(array)
    bundle_path = tmp_path / "bundle.npz"
    bundle.save(bundle_path)
    app = create_app(bundle_path, max_request_bytes=1024)

    oversized_payload = b'{"array_b64":"' + (b"A" * (2 * 1024 * 1024)) + b'"}'

    async def scenario(client: httpx.AsyncClient) -> None:
        async def stream():
            midpoint = len(oversized_payload) // 2
            yield oversized_payload[:midpoint]
            yield oversized_payload[midpoint:]

        request_headers = {"content-type": "application/json", **headers}
        response = await client.post("/v1/codebook/compress", content=stream(), headers=request_headers)
        assert response.status_code == 413
        assert "max_http_body_bytes" in response.json()["detail"]

    asyncio.run(_scenario_client(app, scenario))
