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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperquant.route_benchmark import run_route_benchmark
from hyperquant.live_data import (
    generate_mixed_long_context,
    generate_online_vector_stream,
    generate_structured_long_context,
)
from hyperquant.resident_tier import ResidentTierConfig, ResidentPlanner, run_resident_benchmark

EVIDENCE_DIR = ROOT / "docs" / "evidence"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _render_capacity_markdown(payload: dict[str, object]) -> str:
    lines = ["# Capacity example", ""]
    lines.append(f"Generated: `{payload['meta']['date']}`")
    lines.append("")
    lines.append("This document shows illustrative node-capacity planning on live-like synthetic workloads.")
    lines.append(
        "For each workload, the RAM budget equals the amount needed to keep the baseline fully resident "
        f"for {payload['meta']['budget_sessions']} sessions."
    )
    lines.append("")
    for workload_name, workload in payload["workloads"].items():
        plan = workload["plan"]
        lines.extend([f"## {workload_name}", ""])
        lines.append(f"- shape: `{tuple(workload['shape'])}`")
        lines.append(f"- budget: `{int(workload['budget_bytes']):,} B`")
        lines.append(f"- chosen route: `{plan['chosen_route']}`")
        lines.append(f"- baseline resident/session: `{int(plan['baseline_resident_bytes_per_session']):,} B`")
        lines.append(f"- projected resident/session: `{int(plan['projected_resident_bytes_per_session']):,} B`")
        lines.append(f"- max sessions within budget (baseline): `{int(plan['max_sessions_within_budget_baseline'])}`")
        lines.append(f"- max sessions within budget (projected): `{int(plan['max_sessions_within_budget_projected'])}`")
        lines.append(f"- capacity gain vs baseline: `{float(plan['capacity_gain_vs_baseline']):.2f}x`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _update_hash_manifest(evidence_dir: Path) -> Path:
    hash_path = evidence_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for path in sorted(evidence_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "SHA256SUMS.txt":
            continue
        rel = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    _write_text(hash_path, "\n".join(lines) + ("\n" if lines else ""))
    return hash_path


def _build_capacity_example(
    *,
    date_tag: str,
    config: ResidentTierConfig,
    runtime_value_bytes: int,
    active_window_tokens: int,
    budget_sessions: int,
    concurrent_sessions: int,
) -> tuple[Path, Path]:
    workloads = {
        "online_vector_stream": generate_online_vector_stream(n_vectors=16384, dim=128),
        "structured_long_context": generate_structured_long_context(n_tokens=16384, dim=128, page_size=config.page_size),
        "mixed_long_context": generate_mixed_long_context(n_tokens=32768, dim=128, page_size=config.page_size),
    }
    planner = ResidentPlanner(config)
    report: dict[str, object] = {
        "meta": {
            "date": date_tag,
            "budget_sessions": budget_sessions,
            "concurrent_sessions": concurrent_sessions,
            "active_window_tokens": active_window_tokens,
            "runtime_value_bytes": runtime_value_bytes,
            "page_size": config.page_size,
            "hot_pages": config.hot_pages,
            "group_size": config.group_size,
            "bits": config.bits,
        },
        "workloads": {},
    }
    for workload_name, array in workloads.items():
        baseline_bytes = int(array.size) * runtime_value_bytes
        budget_bytes = baseline_bytes * budget_sessions
        plan = planner.plan(
            array,
            concurrent_sessions=concurrent_sessions,
            active_window_tokens=active_window_tokens,
            runtime_value_bytes=runtime_value_bytes,
            budget_bytes=budget_bytes,
        )
        report["workloads"][workload_name] = {
            "shape": list(array.shape),
            "budget_bytes": budget_bytes,
            "plan": plan.to_dict(),
        }
    json_path = EVIDENCE_DIR / "capacity-example.json"
    md_path = EVIDENCE_DIR / "capacity-example.md"
    _write_json(json_path, report)
    _write_text(md_path, _render_capacity_markdown(report))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the benchmark and evidence pack for HyperQuant.")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date tag recorded inside generated files.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest before generating the proof pack.")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark timing iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Benchmark warmup iterations.")
    parser.add_argument("--slice-iterations", type=int, default=5, help="Memory-tier slice-read timing iterations.")
    parser.add_argument("--budget-sessions", type=int, default=8, help="Capacity example budget in baseline-session units.")
    parser.add_argument("--concurrent-sessions", type=int, default=8, help="Concurrent session count used in memory planning.")
    parser.add_argument("--active-window-tokens", type=int, default=256, help="Hot-window size used in memory planning.")
    parser.add_argument("--runtime-value-bytes", type=int, default=2, help="Runtime bytes per value for baseline memory modeling.")
    parser.add_argument(
        "--prefer-native-fwht",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prefer the native FWHT implementation when available.",
    )
    args = parser.parse_args(argv)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)

    route_benchmark = run_route_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        prefer_native_fwht=args.prefer_native_fwht,
    )
    route_json = EVIDENCE_DIR / "route-benchmark.json"
    route_md = EVIDENCE_DIR / "route-benchmark.md"
    _write_text(route_json, route_benchmark.to_json() + "\n")
    _write_text(route_md, route_benchmark.to_markdown())

    memory_config = ResidentTierConfig()
    memory = run_resident_benchmark(
        {
            "online_vector_stream": generate_online_vector_stream(n_vectors=16384, dim=128),
            "structured_long_context": generate_structured_long_context(n_tokens=16384, dim=128, page_size=memory_config.page_size),
            "mixed_long_context": generate_mixed_long_context(n_tokens=32768, dim=128, page_size=memory_config.page_size),
        },
        config=memory_config,
        concurrent_sessions=args.concurrent_sessions,
        active_window_tokens=args.active_window_tokens,
        runtime_value_bytes=args.runtime_value_bytes,
        slice_iterations=args.slice_iterations,
    )
    memory_json = EVIDENCE_DIR / "resident-benchmark.json"
    memory_md = EVIDENCE_DIR / "resident-benchmark.md"
    _write_text(memory_json, memory.to_json() + "\n")
    _write_text(memory_md, memory.to_markdown())

    capacity_json, capacity_md = _build_capacity_example(
        date_tag=args.date,
        config=memory_config,
        runtime_value_bytes=args.runtime_value_bytes,
        active_window_tokens=args.active_window_tokens,
        budget_sessions=args.budget_sessions,
        concurrent_sessions=args.concurrent_sessions,
    )

    hash_manifest = _update_hash_manifest(EVIDENCE_DIR)

    summary = {
        "route_json": route_json.relative_to(ROOT).as_posix(),
        "route_md": route_md.relative_to(ROOT).as_posix(),
        "memory_json": memory_json.relative_to(ROOT).as_posix(),
        "memory_md": memory_md.relative_to(ROOT).as_posix(),
        "capacity_json": capacity_json.relative_to(ROOT).as_posix(),
        "capacity_md": capacity_md.relative_to(ROOT).as_posix(),
        "hash_manifest": hash_manifest.relative_to(ROOT).as_posix(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
