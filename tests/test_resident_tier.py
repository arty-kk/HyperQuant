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

import pytest
import numpy as np

from hyperquant.live_data import generate_mixed_long_context, generate_online_vector_stream
from hyperquant.resident_tier import ResidentTierConfig, ResidentPlanner, ResidentPageMode, ResidentTierStore


def test_tiered_store_build_open_and_slice(tmp_path) -> None:
    array = generate_mixed_long_context(n_tokens=1024, dim=64, page_size=32, seed=20260329)
    config = ResidentTierConfig(page_size=32, group_size=64, hot_pages=2, residual_topk=1)

    store = ResidentTierStore.build(array, tmp_path / "store", config=config)
    manifest = store.manifest
    assert manifest.stats.compression_ratio > 2.0
    assert manifest.stats.page_mode_counts["vector"] > 0

    restored_slice = store.get_slice(50, 150)
    assert restored_slice.shape == (100, 64)
    diff = restored_slice.astype(np.float32) - array[50:150].astype(np.float32)
    assert float(np.sqrt(np.mean(diff * diff))) < 0.5

    store.preload_pages([0, 1, 2])
    access = store.access_report()
    assert access.cached_pages <= config.hot_pages
    assert access.resident_total_bytes >= access.manifest_bytes

    reopened = ResidentTierStore.open(tmp_path / "store")
    restored_again = reopened.get_slice(50, 150)
    np.testing.assert_allclose(restored_slice, restored_again, rtol=0, atol=0)
    integrity = reopened.verify_integrity()
    assert integrity["checked_pages"] == len(manifest.pages)
    assert integrity["checked_payloads"] == sum(1 for page in manifest.pages if page.mode != ResidentPageMode.PAGE_REF.value)


def test_resident_planner_prefers_tier_for_large_mixed_workload() -> None:
    array = generate_mixed_long_context(n_tokens=4096, dim=64, page_size=32, seed=20260329)
    planner = ResidentPlanner(ResidentTierConfig(page_size=32, group_size=64, hot_pages=4, residual_topk=1))
    plan = planner.plan(
        array,
        concurrent_sessions=8,
        active_window_tokens=64,
        runtime_value_bytes=2,
        budget_bytes=600_000,
    )
    payload = plan.to_dict()
    assert payload["chosen_route"] == "resident_tier"
    assert payload["resident_savings_ratio"] > 0.80
    assert payload["fits_budget"] is True
    assert payload["max_sessions_within_budget_baseline"] is not None
    assert payload["max_sessions_within_budget_projected"] is not None
    assert payload["max_sessions_within_budget_projected"] > payload["max_sessions_within_budget_baseline"]
    assert payload["capacity_gain_vs_baseline"] > 1.0
    assert payload["candidates"]["resident_tier"]["resident_bytes_per_session"] < payload["candidates"]["vector_codec_full_envelope"]["resident_bytes_per_session"]


def test_resident_planner_handles_online_vector_stream() -> None:
    array = generate_online_vector_stream(n_vectors=4096, dim=64, seed=20260329)
    planner = ResidentPlanner(ResidentTierConfig(page_size=32, group_size=64, hot_pages=4, residual_topk=1))
    plan = planner.plan(array, concurrent_sessions=4, active_window_tokens=64, runtime_value_bytes=2)
    assert plan.chosen_route == "resident_tier"
    assert plan.projected_resident_bytes_per_session < plan.baseline_resident_bytes_per_session
    assert plan.candidates["resident_tier"]["page_mode_counts"]["vector"] > 0


def test_tiered_store_detects_payload_tampering(tmp_path) -> None:
    array = generate_mixed_long_context(n_tokens=1024, dim=64, page_size=32, seed=20260329)
    config = ResidentTierConfig(page_size=32, group_size=64, hot_pages=2, residual_topk=1)
    store = ResidentTierStore.build(array, tmp_path / "store", config=config)
    page = next(page for page in store.manifest.pages if page.mode != ResidentPageMode.PAGE_REF.value and page.file_name)
    payload_path = tmp_path / "store" / page.file_name
    tampered = bytearray(payload_path.read_bytes())
    tampered[-1] ^= 0x01
    payload_path.write_bytes(bytes(tampered))

    reopened = ResidentTierStore.open(tmp_path / "store")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        reopened.get_page(page.page_index)
