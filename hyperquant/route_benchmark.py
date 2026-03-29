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

import json
from dataclasses import dataclass

import numpy as np

from .benchmark import TimingStats, time_callable
from .guarantee import GuaranteeMode, ContextGuaranteeProfile
from .live_data import (
    generate_mixed_long_context,
    generate_online_vector_stream,
    generate_structured_long_context,
)
from .context_codec import ContextCodecConfig, ContextCodec
from .vector_codec import DenseRotationBaseline, VectorCodec, RotatedScalarStats, native_fwht_status


@dataclass(frozen=True)
class CodecBenchmarkResult:
    stats: dict[str, object]
    encode_timing: TimingStats
    decode_timing: TimingStats

    def to_dict(self) -> dict[str, object]:
        return {
            "stats": self.stats,
            "encode_timing": self.encode_timing.to_dict(),
            "decode_timing": self.decode_timing.to_dict(),
        }


@dataclass(frozen=True)
class RouteBenchmarkArtifacts:
    report: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(self.report, indent=2, ensure_ascii=False, sort_keys=True)

    def to_markdown(self) -> str:
        lines: list[str] = ["# Route benchmark", ""]
        meta = self.report.get("meta", {})
        if meta:
            lines.append("## Meta")
            lines.append("")
            for key, value in meta.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        for workload_name, workload in self.report.get("workloads", {}).items():
            lines.append(f"## {workload_name}")
            lines.append("")
            lines.append("| codec | ratio | rms | cosine | encode ms | decode ms |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for codec_name, codec_result in workload.get("codecs", {}).items():
                stats = codec_result["stats"]
                encode = codec_result["encode_timing"]
                decode = codec_result["decode_timing"]
                lines.append(
                    f"| {codec_name} | {stats.get('compression_ratio', 0):.3f} | {stats.get('rms_error', 0):.6f} | {stats.get('cosine_similarity', 0):.6f} | {encode.get('mean_ms', 0):.3f} | {decode.get('mean_ms', 0):.3f} |"
                )
            insight = workload.get("insight")
            if insight:
                lines.append("")
                lines.append(f"> {insight}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"



def _benchmark_vector_codec(compressor, array: np.ndarray, *, iterations: int, warmup: int) -> CodecBenchmarkResult:
    envelope, stats = compressor.compress(array)
    _, encode_timing = time_callable(lambda: compressor.compress(array), iterations=iterations, warmup=warmup)
    _, decode_timing = time_callable(lambda: compressor.decompress(envelope), iterations=iterations, warmup=warmup)
    return CodecBenchmarkResult(stats=stats.to_dict(), encode_timing=encode_timing, decode_timing=decode_timing)



def _benchmark_context_codec(
    compressor: ContextCodec,
    array: np.ndarray,
    *,
    iterations: int,
    warmup: int,
    with_guarantee: bool,
) -> CodecBenchmarkResult:
    guarantee_profile = ContextGuaranteeProfile() if with_guarantee else None
    envelope, stats = compressor.compress(
        array,
        guarantee_profile=guarantee_profile,
        guarantee_mode=GuaranteeMode.FAIL_CLOSED if with_guarantee else GuaranteeMode.ALLOW_BEST_EFFORT,
    )
    _, encode_timing = time_callable(
        lambda: compressor.compress(
            array,
            guarantee_profile=guarantee_profile,
            guarantee_mode=GuaranteeMode.FAIL_CLOSED if with_guarantee else GuaranteeMode.ALLOW_BEST_EFFORT,
        ),
        iterations=iterations,
        warmup=warmup,
    )
    _, decode_timing = time_callable(lambda: compressor.decompress(envelope), iterations=iterations, warmup=warmup)
    return CodecBenchmarkResult(stats=stats.to_dict(), encode_timing=encode_timing, decode_timing=decode_timing)



def _workload_summary(codecs: dict[str, CodecBenchmarkResult]) -> str:
    if not codecs:
        return ""
    best_ratio_name = max(codecs, key=lambda name: float(codecs[name].stats.get("compression_ratio", 0.0)))
    fastest_encode_name = min(codecs, key=lambda name: float(codecs[name].encode_timing.mean_ms))
    return (
        f"Highest compression ratio: `{best_ratio_name}`. Fastest encode path: `{fastest_encode_name}`. "
        "All codecs were measured on the same synthetic workload with the same harness."
    )



def run_route_benchmark(
    *,
    bits: int = 3,
    group_size: int = 128,
    vector_count: int = 16384,
    vector_dim: int = 128,
    structured_tokens: int = 4096,
    mixed_tokens: int = 8192,
    page_size: int = 64,
    iterations: int = 5,
    warmup: int = 1,
    prefer_native_fwht: bool = True,
    with_context_guarantee: bool = True,
    residual_topk: int = 1,
) -> RouteBenchmarkArtifacts:
    online_vectors = generate_online_vector_stream(n_vectors=vector_count, dim=vector_dim)
    structured_context = generate_structured_long_context(n_tokens=structured_tokens, dim=vector_dim, page_size=page_size)
    mixed_context = generate_mixed_long_context(n_tokens=mixed_tokens, dim=vector_dim, page_size=page_size)

    baseline_codec = DenseRotationBaseline(bits=bits, group_size=group_size, residual_topk=residual_topk)
    vector_codec = VectorCodec(bits=bits, group_size=group_size, prefer_native_fwht=prefer_native_fwht, residual_topk=residual_topk)
    context_codec = ContextCodec(
        ContextCodecConfig(
            page_size=page_size,
            rank=1,
            prefix_keep_vectors=32,
            suffix_keep_vectors=64,
            low_rank_error_threshold=0.03,
            ref_round_decimals=3,
            enable_page_ref=True,
            page_ref_rel_rms_threshold=0.005,
            enable_int8_fallback=True,
            try_int8_for_protected=True,
            int8_rel_rms_threshold=0.01,
            int8_max_abs_threshold=0.05,
        )
    )

    vector_codecs = {
        "dense_rotation_baseline": _benchmark_vector_codec(baseline_codec, online_vectors, iterations=iterations, warmup=warmup),
        "vector_codec": _benchmark_vector_codec(vector_codec, online_vectors, iterations=iterations, warmup=warmup),
    }
    structured_codecs = {
        "dense_rotation_baseline": _benchmark_vector_codec(baseline_codec, structured_context, iterations=iterations, warmup=warmup),
        "vector_codec": _benchmark_vector_codec(vector_codec, structured_context, iterations=iterations, warmup=warmup),
        "context_codec": _benchmark_context_codec(context_codec, structured_context, iterations=iterations, warmup=warmup, with_guarantee=with_context_guarantee),
    }
    mixed_codecs = {
        "dense_rotation_baseline": _benchmark_vector_codec(baseline_codec, mixed_context, iterations=iterations, warmup=warmup),
        "vector_codec": _benchmark_vector_codec(vector_codec, mixed_context, iterations=iterations, warmup=warmup),
        "context_codec": _benchmark_context_codec(context_codec, mixed_context, iterations=iterations, warmup=warmup, with_guarantee=False),
    }

    report = {
        "meta": {
            "bits": bits,
            "group_size": group_size,
            "iterations": iterations,
            "warmup": warmup,
            "page_size": page_size,
            "vector_count": vector_count,
            "vector_dim": vector_dim,
            "structured_tokens": structured_tokens,
            "mixed_tokens": mixed_tokens,
            "prefer_native_fwht": prefer_native_fwht,
            "residual_topk": residual_topk,
            "native_fwht": native_fwht_status(auto_build=False),
        },
        "workloads": {
            "online_vector_stream": {
                "shape": list(online_vectors.shape),
                "codecs": {name: result.to_dict() for name, result in vector_codecs.items()},
                "insight": _workload_summary(vector_codecs),
            },
            "structured_long_context": {
                "shape": list(structured_context.shape),
                "codecs": {name: result.to_dict() for name, result in structured_codecs.items()},
                "insight": _workload_summary(structured_codecs),
            },
            "mixed_long_context": {
                "shape": list(mixed_context.shape),
                "codecs": {name: result.to_dict() for name, result in mixed_codecs.items()},
                "insight": _workload_summary(mixed_codecs),
            },
        },
    }
    return RouteBenchmarkArtifacts(report=report)


__all__ = [
    "CodecBenchmarkResult",
    "RouteBenchmarkArtifacts",
    "run_route_benchmark",
]
