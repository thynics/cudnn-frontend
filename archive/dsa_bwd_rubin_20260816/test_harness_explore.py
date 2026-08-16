#!/usr/bin/env python3
"""Balanced Rubin benchmark for the frozen DSA baseline and candidate."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import torch


RUBIN_PORTABLE_SMEM_LIMIT = 232_448
RUBIN_OVERSIZED_SMEM_LIMIT = 334_848
RUBIN_OVERSIZED_MIN_LIVE_BYTES = 300 * 1024


def verify_rubin_native_kernel(compiled, implementation: str) -> list[str]:
    """Verify that the internal SM107a compiler produced one main kernel.

    Rubin oversized mode is selected by the internal DSL launch lowering from
    the requested dynamic-SMEM size. The benchmark separately validates that
    each oversized candidate's traced live request is at least 300 KiB.
    """
    symbols = list(getattr(compiled, "kernel_info", {}))
    main_prefix = (
        "kernel_cutlass_bwd_"
        if implementation == "baseline"
        else "kernel_cutlass_kernel_"
    )
    main_symbols = [symbol for symbol in symbols if symbol.startswith(main_prefix)]
    if len(main_symbols) != 1:
        raise RuntimeError(
            f"{implementation}: expected one native SM107a main kernel, got "
            f"{main_symbols} from {symbols}"
        )
    print(
        f"RUBIN_OVERSIZED_NATIVE impl={implementation} "
        f"sym={main_symbols[0]} backend=internal_dsl_tvm_ffi",
        flush=True,
    )
    return main_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate", default="vpagealias_b")
    parser.add_argument(
        "--reference-candidate",
        default="",
        help="Optional two-CTA module used as the paired control.",
    )
    parser.add_argument(
        "--candidate-compile-uumn",
        action="store_true",
        help="Compile only the candidate with ptxas -uumn.",
    )
    parser.add_argument("--topks", default="128,256,512,1024,2048")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--nheads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup-pairs", type=int, default=8)
    parser.add_argument("--paired-samples", type=int, default=32)
    parser.add_argument(
        "--candidate-smem-mode",
        choices=("portable", "oversized", "native"),
        default="oversized",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    return {
        "median_ms": median,
        "p10_ms": percentile(values, 0.10),
        "p90_ms": percentile(values, 0.90),
        "mad_ms": statistics.median(abs(value - median) for value in values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def bootstrap_median_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    bootstraps = []
    for _ in range(10000):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        bootstraps.append(statistics.median(sample))
    return [percentile(bootstraps, 0.025), percentile(bootstraps, 0.975)]


def run_order(index: int, baseline: str, candidate: str) -> tuple[str, str]:
    if index % 2 == 0:
        return baseline, candidate
    return candidate, baseline


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    topks = [int(value) for value in args.topks.split(",")]
    expected_topks = [64, 128, 256, 512, 1024, 2048]
    if not topks or topks != sorted(set(topks)) or any(
        value not in expected_topks for value in topks
    ):
        raise ValueError(f"unexpected top-k protocol: {topks}")

    os.environ["DSA_BL_QDO_STAGE"] = "1"
    os.environ["DSA_BL_K_STAGE"] = "1"
    os.environ["DSA_BL_HALFK"] = "0"
    os.environ["DSA_BL_KSTAGE2"] = "0"
    # Both staged sources use the same launch-only 280,576-byte request;
    # their frozen live SMEM layouts and offsets are unchanged.
    os.environ["DSA_BL_OVPAD"] = "0"
    os.environ.pop("DSA_DEV_IKET", None)
    os.environ.pop("DKG_IKET_INSTRUMENTATION_METHOD", None)
    sys.path[:0] = [
        str(repo / "python"),
        str(repo / "test/python"),
        str(repo),
        str(repo / "benchmark/dsa"),
    ]

    import cutlass
    import cutlass.cute as cute
    cutlass_dsl_version = importlib.metadata.version(
        "nvidia-cutlass-dsl-internal"
    )
    print(
        f"RUBIN_DSL_BACKEND version={cutlass_dsl_version} file={cutlass.__file__}",
        flush=True,
    )
    from benchmark.dsa import benchmark_dsa_sparse_attention_backward as bench
    from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name()
    capability = list(torch.cuda.get_device_capability())
    if capability != [10, 7]:
        raise RuntimeError(f"expected Rubin sm_107, got {gpu} {capability}")

    class_name = "FlashAttentionDSABackwardSm100TwoCTAV2"
    package = "cudnn.deepseek_sparse_attention.sparse_attention_backward"
    from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_baseline import (
        FlashAttentionDSABackwardSm100,
    )

    def load_two_cta(module_suffix: str):
        return getattr(
            importlib.import_module(
                f"{package}.dsa_bwd_sm100_2cta_{module_suffix}"
            ),
            class_name,
        )

    baseline_name = args.reference_candidate or "baseline"
    implementations = {
        baseline_name: (
            load_two_cta(args.reference_candidate)
            if args.reference_candidate
            else FlashAttentionDSABackwardSm100,
            bool(args.reference_candidate),
            not bool(args.reference_candidate),
        ),
        args.candidate: (load_two_cta(args.candidate), True, False),
    }

    print(
        "DSA_COMPARE_START "
        f"gpu={gpu} topks={topks} baseline={baseline_name} "
        f"candidate={args.candidate} "
        f"warmup_pairs={args.warmup_pairs} paired_samples={args.paired_samples}",
        flush=True,
    )
    rows = []
    started = time.monotonic()
    stream = resolve_stream(None)

    for topk in topks:
        case_started = time.monotonic()
        print(f"DSA_COMPARE_CASE_BEGIN topk={topk}", flush=True)
        torch.manual_seed(args.seed + topk)
        q, kv, sink, dout, indices, lengths = bench.make_inputs(
            args.seqlen,
            topk,
            args.seqlen,
            args.nheads,
            args.head_dim,
            args.head_dim,
            torch.bfloat16,
            use_attn_sink=True,
            use_topk_length=True,
        )
        scale = 1.0 / math.sqrt(args.head_dim)
        out, lse = bench.reference_forward(
            q, kv, sink, indices, scale, args.head_dim
        )
        problem_shape = (
            args.seqlen,
            args.seqlen,
            args.head_dim,
            (args.nheads, 1),
        )
        accumulator = cutlass.Float32

        def workspace_shapes(implementation_class):
            return (
                tuple(
                    int(value)
                    for value in implementation_class._get_workspace_size_LSE_OdO(
                        args.seqlen,
                        args.head_dim,
                        args.nheads,
                        1,
                        accumulator,
                    )
                ),
                tuple(
                    int(value)
                    for value in implementation_class._get_workspace_size_dKV(
                        args.seqlen, args.head_dim, 1, accumulator
                    )
                ),
            )

        shapes = {
            name: workspace_shapes(implementation_class)
            for name, (implementation_class, _, _) in implementations.items()
        }
        if len(set(shapes.values())) != 1:
            raise RuntimeError(f"workspace mismatch at topk={topk}: {shapes}")
        common_shapes = shapes[baseline_name]
        buffers = {
            "dq": torch.empty_like(q),
            "dkv": torch.zeros_like(kv),
            "d_sink": torch.zeros_like(sink),
            "workspace_lse_odo": torch.zeros(
                *common_shapes[0], dtype=torch.uint8, device="cuda"
            ),
            "workspace_dkv": torch.zeros(
                *common_shapes[1], dtype=torch.uint8, device="cuda"
            ),
        }

        def build_runner(
            implementation_class,
            has_trace_args: bool,
            is_one_cta_baseline: bool,
            name: str,
        ):
            kernel = implementation_class(
                head_dim=args.head_dim,
                head_dim_v=args.head_dim,
                block_tile=64,
                max_topk=topk,
            )
            if is_one_cta_baseline:
                kernel._setup_attributes()
                observed = {
                    "load_mma_QdO_stage": kernel.load_mma_QdO_stage,
                    "load_mma_K_stage": kernel.load_mma_K_stage,
                }
                expected = {
                    "load_mma_QdO_stage": 1,
                    "load_mma_K_stage": 1,
                }
                if observed != expected:
                    raise RuntimeError(
                        f"baseline environment isolation failed: {observed}"
                    )
            prototypes = [
                to_cute_tensor(q, divisibility=args.head_dim),
                to_cute_tensor(kv, divisibility=args.head_dim),
                to_cute_tensor(out, divisibility=args.head_dim),
                to_cute_tensor(dout, divisibility=args.head_dim),
                to_cute_tensor(lse, assumed_align=4),
                to_cute_tensor(sink),
                to_cute_tensor(indices),
                to_cute_tensor(lengths),
                to_cute_tensor(buffers["dq"], divisibility=args.head_dim),
                to_cute_tensor(buffers["dkv"], divisibility=args.head_dim),
                to_cute_tensor(buffers["d_sink"]),
                to_cute_tensor(buffers["workspace_lse_odo"]),
                to_cute_tensor(buffers["workspace_dkv"]),
            ]
            runtime = [
                q,
                kv,
                out,
                dout,
                lse,
                sink,
                indices,
                lengths,
                buffers["dq"],
                buffers["dkv"],
                buffers["d_sink"],
                buffers["workspace_lse_odo"],
                buffers["workspace_dkv"],
            ]
            if has_trace_args:
                prototypes.extend([None, 0, 0])
                runtime.extend([None, 0, 0])
            prototypes.extend([scale, stream])
            runtime.extend([scale, stream])
            compile_options = "--enable-tvm-ffi --gpu-arch sm_107a"
            if name == args.candidate and args.candidate_compile_uumn:
                compile_options += " --ptxas-options '--uumn'"
            compiled = cute.compile(
                kernel,
                problem_shape,
                *prototypes,
                # Rubin is exposed as CC 10.7.  The internal DSL and CUDA
                # 13.4 toolchain use its native full-feature sm_107a target.
                # Match the historically validated Rubin execution backend.
                options=compile_options,
            )
            configured_kernels = verify_rubin_native_kernel(compiled, name)
            torch.cuda.synchronize()

            if is_one_cta_baseline:
                live_smem_bytes = int(kernel.shared_storage.size_in_bytes())
            else:
                live_smem_bytes = int(kernel.shared_storage_bytes)
            # The generated launch consumes the traced shared-storage size.
            # Gate that actual value instead of the obsolete 243,712-byte
            # marker, which was never wired into launch lowering.
            requested_smem_bytes = live_smem_bytes
            if (
                is_one_cta_baseline
                or args.candidate_smem_mode == "portable"
                or (
                    args.candidate_smem_mode == "native"
                    and requested_smem_bytes <= RUBIN_PORTABLE_SMEM_LIMIT
                )
            ):
                if requested_smem_bytes > RUBIN_PORTABLE_SMEM_LIMIT:
                    raise RuntimeError(
                        f"portable baseline SMEM gate failed: {requested_smem_bytes}"
                    )
                mode = "portable"
                runtime_selection = "internal_dsl_native_launch_lowering"
            else:
                if not (
                    RUBIN_OVERSIZED_MIN_LIVE_BYTES <= requested_smem_bytes
                    <= RUBIN_OVERSIZED_SMEM_LIMIT
                ):
                    raise RuntimeError(
                        f"oversized-SMEM hard gate failed for {name}: "
                        f"live={live_smem_bytes} request={requested_smem_bytes}"
                    )
                mode = "328KB_oversized"
                runtime_selection = "internal_dsl_native_launch_lowering"
            smem_modes[name] = {
                "mode": mode,
                "live_bytes": live_smem_bytes,
                "launch_request_bytes": requested_smem_bytes,
                "portable_limit_bytes": RUBIN_PORTABLE_SMEM_LIMIT,
                "physical_limit_bytes": RUBIN_OVERSIZED_SMEM_LIMIT,
                "runtime_selection": runtime_selection,
                "configured_kernels": configured_kernels,
            }
            print(
                f"DSA_COMPARE_OVERSIZED topk={topk} impl={name} "
                f"live_bytes={live_smem_bytes} "
                f"launch_request_bytes={requested_smem_bytes}",
                flush=True,
            )

            def launch():
                compiled(problem_shape, *runtime)

            return launch

        def reset_accumulators():
            buffers["dkv"].zero_()
            buffers["workspace_dkv"].zero_()
            buffers["d_sink"].zero_()

        runners = {}
        smem_modes = {}
        for name, (
            implementation_class,
            has_trace_args,
            is_one_cta_baseline,
        ) in implementations.items():
            compile_started = time.monotonic()
            runners[name] = build_runner(
                implementation_class,
                has_trace_args,
                is_one_cta_baseline,
                name,
            )
            print(
                f"DSA_COMPARE_COMPILED topk={topk} impl={name} "
                f"duration_s={time.monotonic() - compile_started:.3f}",
                flush=True,
            )
            reset_accumulators()
            print(
                f"DSA_COMPARE_FRESH_SMOKE_BEGIN topk={topk} impl={name}",
                flush=True,
            )
            fresh_smoke_started = time.monotonic()
            runners[name]()
            torch.cuda.synchronize()
            print(
                f"DSA_COMPARE_FRESH_SMOKE_DONE topk={topk} impl={name} "
                f"wall_s={time.monotonic() - fresh_smoke_started:.6f}",
                flush=True,
            )

        # Isolate the first native launch before batched warmups. This is also
        # the runtime hard gate for the candidate's oversized-SMEM request.
        for name in (baseline_name, args.candidate):
            reset_accumulators()
            print(
                f"DSA_COMPARE_SMOKE_BEGIN topk={topk} impl={name}", flush=True
            )
            smoke_started = time.monotonic()
            runners[name]()
            torch.cuda.synchronize()
            print(
                f"DSA_COMPARE_SMOKE_DONE topk={topk} impl={name} "
                f"wall_s={time.monotonic() - smoke_started:.6f}",
                flush=True,
            )

        for index in range(args.warmup_pairs):
            for name in run_order(index, baseline_name, args.candidate):
                reset_accumulators()
                runners[name]()
        torch.cuda.synchronize()

        def time_program_only(launch) -> float:
            reset_accumulators()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            launch()
            end_event.record()
            end_event.synchronize()
            return float(start_event.elapsed_time(end_event))

        raw_ms = {name: [] for name in implementations}
        for index in range(args.paired_samples):
            for name in run_order(index, baseline_name, args.candidate):
                raw_ms[name].append(time_program_only(runners[name]))

        summary = {name: summarize(values) for name, values in raw_ms.items()}
        ratio_values = [
            candidate_ms / baseline_ms
            for candidate_ms, baseline_ms in zip(
                raw_ms[args.candidate], raw_ms[baseline_name]
            )
        ]
        ratios = {
            "candidate_over_baseline": {
                "raw": ratio_values,
                "median": statistics.median(ratio_values),
                "ci95": bootstrap_median_ci(
                    ratio_values, args.seed + 100000 + topk
                ),
            }
        }
        winner = min(summary, key=lambda name: summary[name]["median_ms"])
        row = {
            "topk": topk,
            "summary": summary,
            "raw_ms": raw_ms,
            "ratios": ratios,
            "shared_memory_mode": smem_modes,
            "winner": winner,
            "case_duration_s": time.monotonic() - case_started,
        }
        rows.append(row)
        print(
            "DSA_COMPARE_RESULT "
            f"topk={topk} baseline_ms={summary[baseline_name]['median_ms']:.6f} "
            f"candidate_ms={summary[args.candidate]['median_ms']:.6f} "
            f"candidate_over_baseline={ratios['candidate_over_baseline']['median']:.6f} "
            f"winner={winner}",
            flush=True,
        )
        del runners, buffers, out, lse, q, kv, sink, dout, indices, lengths
        torch.cuda.empty_cache()

    payload = {
        "status": "pass",
        "benchmark": "rubin_balanced_2way_direct_program_only_topk_sweep_20260816",
        "correctness": {"status": "skipped", "reason": "perf-only user request"},
        "gpu": gpu,
        "compute_capability": capability,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "codegen_target": "sm_107a_native",
        "cutlass_dsl_version": cutlass_dsl_version,
        "rubin_shared_memory_mode": (
            "baseline_and_candidate_portable"
            if args.candidate_smem_mode == "portable"
            else "baseline_portable_candidate_native"
            if args.candidate_smem_mode == "native"
            else "baseline_portable_candidate_328KB_oversized"
        ),
        "rubin_oversized_min_live_bytes": RUBIN_OVERSIZED_MIN_LIVE_BYTES,
        "repo": str(repo),
        "baseline": baseline_name,
        "candidate": args.candidate,
        "seqlen_q": args.seqlen,
        "seqlen_kv": args.seqlen,
        "nheads": args.nheads,
        "head_dim_qk": args.head_dim,
        "head_dim_v": args.head_dim,
        "dtype": "bfloat16",
        "topks": topks,
        "warmup_pairs": args.warmup_pairs,
        "paired_samples": args.paired_samples,
        "timing_scope": "kernel program only; dkv/workspace/d_sink reset excluded",
        "ordering": "AB/BA alternating paired order",
        "rows": rows,
        "duration_s": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"DSA_COMPARE_PASS output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
