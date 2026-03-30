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
import uvicorn

from .api.app import create_app
from .audit import audit_context_input
from .benchmark import (
    BenchmarkReport,
    benchmark_array,
    benchmark_context_array,
    report_to_pretty_json,
    stats_to_pretty_json,
    time_callable,
)
from .bundle import CodebookBundle
from .codebook import MiniBatchKMeansTrainer
from .codebook_codec import CodebookEnvelope, CodebookCodec
from .config import CodebookConfig
from .route_benchmark import run_route_benchmark
from .guarantee import ContourViolation, GuaranteeMode, GuaranteeViolation, ContextGuaranteeProfile
from .live_data import (
    generate_mixed_long_context,
    generate_online_vector_stream,
    generate_structured_long_context,
)
from .context_codec import (
    ContextEnvelope,
    ContextCodecConfig,
    ContextCodec,
)
from .defaults import (
    CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT,
    CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT,
    CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT,
    CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT,
    CONTEXT_PAGE_SIZE_DEFAULT,
    CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT,
    CONTEXT_RANK_DEFAULT,
    CONTEXT_REF_ROUND_DECIMALS_DEFAULT,
    CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT,
    RESIDENT_ACTIVE_WINDOW_TOKENS_DEFAULT,
    RESIDENT_CONCURRENT_SESSIONS_DEFAULT,
    RESIDENT_HOT_PAGES_DEFAULT,
    RESIDENT_RUNTIME_VALUE_BYTES_DEFAULT,
    VECTOR_BITS_DEFAULT,
    VECTOR_GROUP_SIZE_DEFAULT,
    VECTOR_RESIDUAL_TOPK_DEFAULT,
    VECTOR_ROTATION_SEED_DEFAULT,
)
from .resident_tier import (
    ResidentTierConfig,
    ResidentPlanner,
    ResidentTierStore,
    build_resident_store,
    run_resident_benchmark,
)
from .vector_codec import DenseRotationBaseline, VectorCodec, RotatedScalarEnvelope



def _load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)



def _build_context_config(args: argparse.Namespace) -> ContextCodecConfig:
    return ContextCodecConfig(
        page_size=args.page_size,
        rank=args.rank,
        prefix_keep_vectors=args.prefix_keep_vectors,
        suffix_keep_vectors=args.suffix_keep_vectors,
        low_rank_error_threshold=args.low_rank_error_threshold,
        ref_round_decimals=args.ref_round_decimals,
        enable_page_ref=not args.disable_page_ref,
        page_ref_rel_rms_threshold=args.page_ref_rel_rms_threshold,
        enable_int8_fallback=not args.disable_int8_fallback,
        try_int8_for_protected=not args.disable_int8_for_protected,
        int8_rel_rms_threshold=args.int8_rel_rms_threshold,
        int8_max_abs_threshold=args.int8_max_abs_threshold,
    )



def _build_guarantee_profile(args: argparse.Namespace) -> ContextGuaranteeProfile | None:
    if not getattr(args, "with_guarantee", False):
        return None
    return ContextGuaranteeProfile(
        min_compression_ratio=args.min_compression_ratio,
        min_cosine_similarity=args.min_cosine_similarity,
        max_rms_error=args.max_rms_error,
        max_max_abs_error=args.max_max_abs_error,
    )



def _build_vector_compressor(args: argparse.Namespace) -> VectorCodec:
    return VectorCodec(
        bits=args.bits,
        group_size=args.group_size,
        rotation_seed=args.rotation_seed,
        residual_topk=args.residual_topk,
        prefer_native_fwht=not args.disable_native_fwht,
    )



def _build_dense_baseline(args: argparse.Namespace) -> DenseRotationBaseline:
    return DenseRotationBaseline(
        bits=args.bits,
        group_size=args.group_size,
        rotation_seed=args.rotation_seed,
        residual_topk=args.residual_topk,
    )



def _build_resident_tier_config(args: argparse.Namespace) -> ResidentTierConfig:
    return ResidentTierConfig(
        page_size=args.page_size,
        rank=args.rank,
        bits=args.bits,
        group_size=args.group_size,
        rotation_seed=args.rotation_seed,
        hot_pages=args.hot_pages,
        residual_topk=args.residual_topk,
        prefix_keep_vectors=args.prefix_keep_vectors,
        suffix_keep_vectors=args.suffix_keep_vectors,
        low_rank_error_threshold=args.low_rank_error_threshold,
        ref_round_decimals=args.ref_round_decimals,
        enable_page_ref=not args.disable_page_ref,
        page_ref_rel_rms_threshold=args.page_ref_rel_rms_threshold,
        enable_int8_fallback=not args.disable_int8_fallback,
        try_int8_for_protected=not args.disable_int8_for_protected,
        int8_rel_rms_threshold=args.int8_rel_rms_threshold,
        int8_max_abs_threshold=args.int8_max_abs_threshold,
        prefer_native_fwht=not args.disable_native_fwht,
        allow_vector_for_protected=args.allow_vector_for_protected,
    )



def cmd_train_codebook(args: argparse.Namespace) -> int:
    vectors = _load_array(args.input)
    config = CodebookConfig(
        chunk_size=args.chunk_size,
        codebook_size=args.codebook_size,
        rotation_seed=args.rotation_seed,
        normalize=not args.disable_normalize,
        sample_size=args.sample_size,
        training_iterations=args.training_iterations,
    )
    trainer = MiniBatchKMeansTrainer(config)
    bundle = trainer.train(vectors)
    bundle.save(args.output)
    print(f"saved bundle to {args.output}")
    return 0



def cmd_compress_file(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    bundle = CodebookBundle.load(args.bundle)
    compressor = CodebookCodec(bundle)
    envelope, stats = compressor.compress(array)
    Path(args.output).write_bytes(envelope.to_bytes())
    print(stats_to_pretty_json(stats))
    return 0



def cmd_decompress_file(args: argparse.Namespace) -> int:
    bundle = CodebookBundle.load(args.bundle)
    compressor = CodebookCodec(bundle)
    envelope = CodebookEnvelope.from_bytes(Path(args.input).read_bytes())
    restored = compressor.decompress(envelope)
    np.save(args.output, restored, allow_pickle=False)
    print(f"saved reconstructed array to {args.output}")
    return 0



def cmd_codebook_benchmark(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    bundle = CodebookBundle.load(args.bundle)
    compressor = CodebookCodec(bundle)
    stats = benchmark_array(compressor, array)
    print(stats_to_pretty_json(stats))
    return 0



def cmd_context_compress_file(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    compressor = ContextCodec(_build_context_config(args))
    envelope, stats = compressor.compress(
        array,
        guarantee_profile=_build_guarantee_profile(args),
        guarantee_mode=GuaranteeMode.FAIL_CLOSED if args.fail_closed else GuaranteeMode.ALLOW_BEST_EFFORT,
    )
    Path(args.output).write_bytes(envelope.to_bytes())
    print(stats_to_pretty_json(stats))
    return 0



def cmd_context_decompress_file(args: argparse.Namespace) -> int:
    envelope = ContextEnvelope.from_bytes(Path(args.input).read_bytes())
    compressor = ContextCodec(ContextCodecConfig(page_size=envelope.page_size, rank=envelope.rank))
    restored = compressor.decompress(envelope)
    np.save(args.output, restored, allow_pickle=False)
    print(f"saved reconstructed array to {args.output}")
    return 0



def cmd_context_benchmark(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    compressor = ContextCodec(_build_context_config(args))
    _, timing = time_callable(
        lambda: compressor.compress(
            array,
            guarantee_profile=_build_guarantee_profile(args),
            guarantee_mode=GuaranteeMode.FAIL_CLOSED if args.fail_closed else GuaranteeMode.ALLOW_BEST_EFFORT,
        )[1],
        iterations=args.iterations,
        warmup=args.warmup,
    )
    stats = benchmark_context_array(
        compressor,
        array,
        guarantee_profile=_build_guarantee_profile(args),
        guarantee_mode=GuaranteeMode.FAIL_CLOSED if args.fail_closed else GuaranteeMode.ALLOW_BEST_EFFORT,
    )
    report = BenchmarkReport(timing=timing, stats=stats.to_dict())
    print(report_to_pretty_json(report))
    return 0



def cmd_vector_compress_file(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    compressor = _build_vector_compressor(args)
    envelope, stats = compressor.compress(array)
    Path(args.output).write_bytes(envelope.to_bytes())
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0



def cmd_vector_decompress_file(args: argparse.Namespace) -> int:
    compressor = _build_vector_compressor(args)
    envelope = RotatedScalarEnvelope.from_bytes(Path(args.input).read_bytes())
    restored = compressor.decompress(envelope)
    np.save(args.output, restored, allow_pickle=False)
    print(f"saved reconstructed array to {args.output}")
    return 0



def cmd_vector_benchmark(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    compressor = _build_vector_compressor(args)
    envelope, stats = compressor.compress(array)
    _, encode_timing = time_callable(lambda: compressor.compress(array), iterations=args.iterations, warmup=args.warmup)
    _, decode_timing = time_callable(lambda: compressor.decompress(envelope), iterations=args.iterations, warmup=args.warmup)
    payload = {
        "stats": stats.to_dict(),
        "encode_timing": encode_timing.to_dict(),
        "decode_timing": decode_timing.to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0



def cmd_dense_baseline_benchmark(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    compressor = _build_dense_baseline(args)
    envelope, stats = compressor.compress(array)
    _, encode_timing = time_callable(lambda: compressor.compress(array), iterations=args.iterations, warmup=args.warmup)
    _, decode_timing = time_callable(lambda: compressor.decompress(envelope), iterations=args.iterations, warmup=args.warmup)
    payload = {
        "stats": stats.to_dict(),
        "encode_timing": encode_timing.to_dict(),
        "decode_timing": decode_timing.to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0



def cmd_generate_live_data(args: argparse.Namespace) -> int:
    scenario = args.scenario
    if scenario == "online":
        array = generate_online_vector_stream(n_vectors=args.n_vectors, dim=args.dim, seed=args.seed)
    elif scenario == "structured":
        array = generate_structured_long_context(n_tokens=args.tokens, dim=args.dim, page_size=args.page_size, seed=args.seed)
    elif scenario == "mixed":
        array = generate_mixed_long_context(n_tokens=args.tokens, dim=args.dim, page_size=args.page_size, seed=args.seed)
    else:  # pragma: no cover
        raise ValueError(f"unsupported scenario: {scenario}")
    np.save(args.output, array, allow_pickle=False)
    print(f"saved {scenario} live-like dataset to {args.output}")
    print(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype), "scenario": scenario}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0



def cmd_route_benchmark(args: argparse.Namespace) -> int:
    artifacts = run_route_benchmark(
        bits=args.bits,
        group_size=args.group_size,
        vector_count=args.n_vectors,
        vector_dim=args.dim,
        structured_tokens=args.structured_tokens,
        mixed_tokens=args.mixed_tokens,
        page_size=args.page_size,
        iterations=args.iterations,
        warmup=args.warmup,
        prefer_native_fwht=not args.disable_native_fwht,
        with_context_guarantee=not args.disable_context_guarantee,
        residual_topk=args.residual_topk,
    )
    if args.json_output:
        Path(args.json_output).write_text(artifacts.to_json(), encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(artifacts.to_markdown(), encoding="utf-8")
    print(artifacts.to_json())
    return 0



def cmd_benchmark_suite(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    output: dict[str, object] = {}

    if args.mode in {"codebook", "both"}:
        if not args.bundle:
            raise ValueError("--bundle is required for codebook or both benchmark modes")
        bundle = CodebookBundle.load(args.bundle)
        compressor = CodebookCodec(bundle)
        stats, timing = time_callable(
            lambda: benchmark_array(compressor, array),
            iterations=args.iterations,
            warmup=args.warmup,
        )
        output["codebook"] = BenchmarkReport(timing=timing, stats=stats.to_dict()).to_dict()

    if args.mode in {"context", "both"}:
        compressor = ContextCodec(_build_context_config(args))
        stats, timing = time_callable(
            lambda: benchmark_context_array(
                compressor,
                array,
                guarantee_profile=_build_guarantee_profile(args),
                guarantee_mode=GuaranteeMode.FAIL_CLOSED if args.fail_closed else GuaranteeMode.ALLOW_BEST_EFFORT,
            ),
            iterations=args.iterations,
            warmup=args.warmup,
        )
        output["context"] = BenchmarkReport(timing=timing, stats=stats.to_dict()).to_dict()

    payload = json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"saved benchmark report to {args.output}")
    print(payload)
    return 0



def cmd_context_audit_input(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    artifacts = audit_context_input(
        array,
        context_config=_build_context_config(args),
        guarantee_profile=_build_guarantee_profile(args),
        bundle_path=args.bundle,
    )
    if args.json_output:
        Path(args.json_output).write_text(artifacts.to_json(), encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(artifacts.markdown, encoding="utf-8")
    print(artifacts.markdown)
    return 0



def cmd_resident_plan(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    planner = ResidentPlanner(_build_resident_tier_config(args))
    plan = planner.plan(
        array,
        concurrent_sessions=args.concurrent_sessions,
        active_window_tokens=args.active_window_tokens,
        runtime_value_bytes=args.runtime_value_bytes,
        budget_bytes=args.budget_bytes,
    )
    payload = json.dumps(plan.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"saved memory plan to {args.output}")
    print(payload)
    return 0



def cmd_build_resident_store(args: argparse.Namespace) -> int:
    array = _load_array(args.input)
    manifest = build_resident_store(array, args.output, config=_build_resident_tier_config(args))
    print(manifest.to_json())
    return 0



def cmd_read_resident_slice(args: argparse.Namespace) -> int:
    store = ResidentTierStore.open(args.store)
    restored = store.get_slice(args.start, args.end)
    np.save(args.output, restored, allow_pickle=False)
    payload = {
        "output": str(args.output),
        "shape": list(restored.shape),
        "access_report": store.access_report().to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0



def cmd_verify_resident_store(args: argparse.Namespace) -> int:
    store = ResidentTierStore.open(args.store)
    report = store.verify_integrity()
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0



def cmd_resident_benchmark(args: argparse.Namespace) -> int:
    workloads = {
        "online_vector_stream": generate_online_vector_stream(n_vectors=args.n_vectors, dim=args.dim, seed=args.seed),
        "structured_long_context": generate_structured_long_context(n_tokens=args.structured_tokens, dim=args.dim, page_size=args.page_size, seed=args.seed + 1),
        "mixed_long_context": generate_mixed_long_context(n_tokens=args.mixed_tokens, dim=args.dim, page_size=args.page_size, seed=args.seed + 2),
    }
    artifacts = run_resident_benchmark(
        workloads,
        config=_build_resident_tier_config(args),
        concurrent_sessions=args.concurrent_sessions,
        active_window_tokens=args.active_window_tokens,
        runtime_value_bytes=args.runtime_value_bytes,
        slice_iterations=args.slice_iterations,
    )
    if args.json_output:
        Path(args.json_output).write_text(artifacts.to_json(), encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(artifacts.to_markdown(), encoding="utf-8")
    print(artifacts.to_json())
    return 0



def cmd_serve(args: argparse.Namespace) -> int:
    app = create_app(
        args.bundle,
        max_request_bytes=args.max_request_bytes,
        max_concurrency=args.max_concurrency,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0



def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--page-size", type=int, default=CONTEXT_PAGE_SIZE_DEFAULT)
    parser.add_argument("--rank", type=int, default=CONTEXT_RANK_DEFAULT)
    parser.add_argument("--prefix-keep-vectors", type=int, default=CONTEXT_PREFIX_KEEP_VECTORS_DEFAULT)
    parser.add_argument("--suffix-keep-vectors", type=int, default=CONTEXT_SUFFIX_KEEP_VECTORS_DEFAULT)
    parser.add_argument("--low-rank-error-threshold", type=float, default=CONTEXT_LOW_RANK_ERROR_THRESHOLD_DEFAULT)
    parser.add_argument("--ref-round-decimals", type=int, default=CONTEXT_REF_ROUND_DECIMALS_DEFAULT)
    parser.add_argument("--page-ref-rel-rms-threshold", type=float, default=CONTEXT_PAGE_REF_REL_RMS_THRESHOLD_DEFAULT)
    parser.add_argument("--disable-page-ref", action="store_true")
    parser.add_argument("--disable-int8-fallback", action="store_true")
    parser.add_argument("--disable-int8-for-protected", action="store_true")
    parser.add_argument("--int8-rel-rms-threshold", type=float, default=CONTEXT_INT8_REL_RMS_THRESHOLD_DEFAULT)
    parser.add_argument("--int8-max-abs-threshold", type=float, default=CONTEXT_INT8_MAX_ABS_THRESHOLD_DEFAULT)



def _add_guarantee_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--with-guarantee", action="store_true", help="Evaluate the context-route guarantee contract.")
    parser.add_argument("--fail-closed", action="store_true", help="Reject the result if the guarantee does not pass.")
    parser.add_argument("--min-compression-ratio", type=float, default=30.0)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--max-rms-error", type=float, default=0.010)
    parser.add_argument("--max-max-abs-error", type=float, default=0.050)



def _add_timing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)



def _add_vector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bits", type=int, default=VECTOR_BITS_DEFAULT)
    parser.add_argument("--group-size", type=int, default=VECTOR_GROUP_SIZE_DEFAULT)
    parser.add_argument("--rotation-seed", type=int, default=VECTOR_ROTATION_SEED_DEFAULT)
    parser.add_argument("--residual-topk", type=int, default=VECTOR_RESIDUAL_TOPK_DEFAULT, help="Number of rotated coefficients per group stored exactly as a residual rescue side-channel.")
    parser.add_argument("--disable-native-fwht", action="store_true")



def _add_resident_tier_args(parser: argparse.ArgumentParser) -> None:
    _add_context_args(parser)
    _add_vector_args(parser)
    parser.add_argument("--hot-pages", type=int, default=RESIDENT_HOT_PAGES_DEFAULT)
    parser.add_argument("--allow-vector-for-protected", action="store_true")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperquant")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train-codebook")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--chunk-size", type=int, default=32)
    train.add_argument("--codebook-size", type=int, default=256)
    train.add_argument("--rotation-seed", type=int, default=17)
    train.add_argument("--sample-size", type=int, default=20000)
    train.add_argument("--training-iterations", type=int, default=15)
    train.add_argument("--disable-normalize", action="store_true")
    train.set_defaults(func=cmd_train_codebook)

    compress = sub.add_parser("codebook-compress-file")
    compress.add_argument("--bundle", required=True)
    compress.add_argument("--input", required=True)
    compress.add_argument("--output", required=True)
    compress.set_defaults(func=cmd_compress_file)

    decompress = sub.add_parser("codebook-decompress-file")
    decompress.add_argument("--bundle", required=True)
    decompress.add_argument("--input", required=True)
    decompress.add_argument("--output", required=True)
    decompress.set_defaults(func=cmd_decompress_file)

    benchmark = sub.add_parser("codebook-benchmark")
    benchmark.add_argument("--bundle", required=True)
    benchmark.add_argument("--input", required=True)
    benchmark.set_defaults(func=cmd_codebook_benchmark)

    vector_compress = sub.add_parser("vector-compress-file")
    vector_compress.add_argument("--input", required=True)
    vector_compress.add_argument("--output", required=True)
    _add_vector_args(vector_compress)
    vector_compress.set_defaults(func=cmd_vector_compress_file)

    vector_decompress = sub.add_parser("vector-decompress-file")
    vector_decompress.add_argument("--input", required=True)
    vector_decompress.add_argument("--output", required=True)
    _add_vector_args(vector_decompress)
    vector_decompress.set_defaults(func=cmd_vector_decompress_file)

    vector_benchmark = sub.add_parser("vector-benchmark")
    vector_benchmark.add_argument("--input", required=True)
    _add_vector_args(vector_benchmark)
    _add_timing_args(vector_benchmark)
    vector_benchmark.set_defaults(func=cmd_vector_benchmark)

    dense_baseline = sub.add_parser("dense-baseline-benchmark")
    dense_baseline.add_argument("--input", required=True)
    _add_vector_args(dense_baseline)
    _add_timing_args(dense_baseline)
    dense_baseline.set_defaults(func=cmd_dense_baseline_benchmark)

    live_data = sub.add_parser("generate-live-data")
    live_data.add_argument("--scenario", choices=["online", "structured", "mixed"], default="structured")
    live_data.add_argument("--output", required=True)
    live_data.add_argument("--n-vectors", type=int, default=16384)
    live_data.add_argument("--tokens", type=int, default=4096)
    live_data.add_argument("--dim", type=int, default=128)
    live_data.add_argument("--page-size", type=int, default=64)
    live_data.add_argument("--seed", type=int, default=20260329)
    live_data.set_defaults(func=cmd_generate_live_data)

    route_benchmark = sub.add_parser("route-benchmark")
    _add_vector_args(route_benchmark)
    _add_timing_args(route_benchmark)
    route_benchmark.add_argument("--n-vectors", type=int, default=16384)
    route_benchmark.add_argument("--structured-tokens", type=int, default=4096)
    route_benchmark.add_argument("--mixed-tokens", type=int, default=8192)
    route_benchmark.add_argument("--dim", type=int, default=128)
    route_benchmark.add_argument("--page-size", type=int, default=64)
    route_benchmark.add_argument("--disable-context-guarantee", action="store_true")
    route_benchmark.add_argument("--json-output")
    route_benchmark.add_argument("--markdown-output")
    route_benchmark.set_defaults(func=cmd_route_benchmark)

    context_compress = sub.add_parser("context-compress-file")
    context_compress.add_argument("--input", required=True)
    context_compress.add_argument("--output", required=True)
    _add_context_args(context_compress)
    _add_guarantee_args(context_compress)
    context_compress.set_defaults(func=cmd_context_compress_file)

    context_decompress = sub.add_parser("context-decompress-file")
    context_decompress.add_argument("--input", required=True)
    context_decompress.add_argument("--output", required=True)
    context_decompress.set_defaults(func=cmd_context_decompress_file)

    context_benchmark = sub.add_parser("context-benchmark")
    context_benchmark.add_argument("--input", required=True)
    _add_context_args(context_benchmark)
    _add_guarantee_args(context_benchmark)
    _add_timing_args(context_benchmark)
    context_benchmark.set_defaults(func=cmd_context_benchmark)

    benchmark_suite = sub.add_parser("benchmark-suite")
    benchmark_suite.add_argument("--input", required=True)
    benchmark_suite.add_argument("--bundle")
    benchmark_suite.add_argument("--mode", choices=["codebook", "context", "both"], default="both")
    benchmark_suite.add_argument("--output")
    _add_context_args(benchmark_suite)
    _add_guarantee_args(benchmark_suite)
    _add_timing_args(benchmark_suite)
    benchmark_suite.set_defaults(func=cmd_benchmark_suite)

    context_audit = sub.add_parser("context-audit-input")
    context_audit.add_argument("--input", required=True)
    context_audit.add_argument("--bundle")
    context_audit.add_argument("--json-output")
    context_audit.add_argument("--markdown-output")
    _add_context_args(context_audit)
    _add_guarantee_args(context_audit)
    context_audit.set_defaults(func=cmd_context_audit_input)

    resident_plan = sub.add_parser("resident-plan")
    resident_plan.add_argument("--input", required=True)
    resident_plan.add_argument("--output")
    resident_plan.add_argument("--concurrent-sessions", type=int, default=RESIDENT_CONCURRENT_SESSIONS_DEFAULT)
    resident_plan.add_argument("--active-window-tokens", type=int, default=RESIDENT_ACTIVE_WINDOW_TOKENS_DEFAULT)
    resident_plan.add_argument("--runtime-value-bytes", type=int, default=RESIDENT_RUNTIME_VALUE_BYTES_DEFAULT)
    resident_plan.add_argument("--budget-bytes", type=int)
    _add_resident_tier_args(resident_plan)
    resident_plan.set_defaults(func=cmd_resident_plan)

    build_resident_store = sub.add_parser("build-resident-store")
    build_resident_store.add_argument("--input", required=True)
    build_resident_store.add_argument("--output", required=True)
    _add_resident_tier_args(build_resident_store)
    build_resident_store.set_defaults(func=cmd_build_resident_store)

    read_resident_slice = sub.add_parser("read-resident-slice")
    read_resident_slice.add_argument("--store", required=True)
    read_resident_slice.add_argument("--start", type=int, required=True)
    read_resident_slice.add_argument("--end", type=int, required=True)
    read_resident_slice.add_argument("--output", required=True)
    read_resident_slice.set_defaults(func=cmd_read_resident_slice)

    verify_resident_store = sub.add_parser("verify-resident-store")
    verify_resident_store.add_argument("--store", required=True)
    verify_resident_store.add_argument("--output")
    verify_resident_store.set_defaults(func=cmd_verify_resident_store)

    resident_benchmark = sub.add_parser("resident-benchmark")
    resident_benchmark.add_argument("--n-vectors", type=int, default=16384)
    resident_benchmark.add_argument("--structured-tokens", type=int, default=16384)
    resident_benchmark.add_argument("--mixed-tokens", type=int, default=32768)
    resident_benchmark.add_argument("--dim", type=int, default=128)
    resident_benchmark.add_argument("--concurrent-sessions", type=int, default=RESIDENT_CONCURRENT_SESSIONS_DEFAULT)
    resident_benchmark.add_argument("--active-window-tokens", type=int, default=RESIDENT_ACTIVE_WINDOW_TOKENS_DEFAULT)
    resident_benchmark.add_argument("--runtime-value-bytes", type=int, default=RESIDENT_RUNTIME_VALUE_BYTES_DEFAULT)
    resident_benchmark.add_argument("--slice-iterations", type=int, default=5)
    resident_benchmark.add_argument("--seed", type=int, default=20260329)
    resident_benchmark.add_argument("--json-output")
    resident_benchmark.add_argument("--markdown-output")
    _add_resident_tier_args(resident_benchmark)
    resident_benchmark.set_defaults(func=cmd_resident_benchmark)

    serve = sub.add_parser("serve")
    serve.add_argument("--bundle", required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    serve.add_argument("--max-concurrency", type=int)
    serve.set_defaults(func=cmd_serve)

    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ContourViolation as exc:
        print(f"contour violation: {exc}", file=sys.stderr)
        return 3
    except GuaranteeViolation as exc:
        print(f"guarantee violation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
