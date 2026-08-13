#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fair direct-kernel A/B benchmark for SM100 DSA backward two-CTA.

The established ``FlashAttentionDSABackwardSm100`` implementation is the
baseline and ``FlashAttentionDSABackwardSm100TwoCTA`` is the candidate.  Both
are compiled directly with ``cute.compile`` against the same tensor objects,
output buffers, and workspaces.  Compilation, allocation, and output/workspace
reset are outside CUDA-event timing.

Before timing, the script runs both implementations at the requested benchmark
shape and requires all three public outputs (dQ, dKV, and dSink) to agree.  The
timed samples alternate A/B and B/A pairs, producing an ABBA launch sequence for
every two pairs.

Example:
    python benchmark/dsa/benchmark_dsa_sparse_attention_backward_sm100_2_cta.py \
        --seqlen 4096 --topk 2048 --warmup-pairs 4 --paired-samples 20 \
        --json /tmp/dsa_bwd_sm100_2_cta.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2_cta import FlashAttentionDSABackwardSm100TwoCTA
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

BASELINE = "sm100"
CANDIDATE = "sm100_2_cta"
HEADS = 128
HEAD_DIM = 512
HEAD_DIM_V = 512
BLOCK_TILE = 64
CORRECTNESS_ATOL = 5e-2
CORRECTNESS_RTOL = 5e-2


@dataclass
class BenchmarkCase:
    problem_shape: tuple[int, int, int, tuple[int, int]]
    q: torch.Tensor
    kv: torch.Tensor
    out: torch.Tensor
    dout: torch.Tensor
    lse: torch.Tensor
    attn_sink: torch.Tensor
    topk_idxs: torch.Tensor
    topk_length: torch.Tensor
    dq: torch.Tensor
    dkv: torch.Tensor
    d_sink: torch.Tensor
    workspace_lse_odo: torch.Tensor
    workspace_dkv: torch.Tensor
    softmax_scale: float

    def runtime_args(self, stream: Any) -> tuple[Any, ...]:
        return (
            self.problem_shape,
            self.q,
            self.kv,
            self.out,
            self.dout,
            self.lse,
            self.attn_sink,
            self.topk_idxs,
            self.topk_length,
            self.dq,
            self.dkv,
            self.d_sink,
            self.workspace_lse_odo,
            self.workspace_dkv,
            self.softmax_scale,
            stream,
        )

    def compile_args(self, stream: Any) -> tuple[Any, ...]:
        return (
            self.problem_shape,
            to_cute_tensor(self.q, divisibility=HEAD_DIM),
            to_cute_tensor(self.kv, divisibility=HEAD_DIM),
            to_cute_tensor(self.out, divisibility=HEAD_DIM_V),
            to_cute_tensor(self.dout, divisibility=HEAD_DIM_V),
            to_cute_tensor(self.lse, assumed_align=4),
            to_cute_tensor(self.attn_sink),
            to_cute_tensor(self.topk_idxs),
            to_cute_tensor(self.topk_length),
            to_cute_tensor(self.dq, divisibility=HEAD_DIM),
            to_cute_tensor(self.dkv, divisibility=HEAD_DIM),
            to_cute_tensor(self.d_sink),
            to_cute_tensor(self.workspace_lse_odo),
            to_cute_tensor(self.workspace_dkv),
            self.softmax_scale,
            stream,
        )


@torch.no_grad()
def reference_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create production-valid out and KV-only LSE without using either BWD."""
    seqlen, num_heads, _ = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((seqlen, num_heads), dtype=torch.float32, device=q.device)
    for begin in range(0, seqlen, chunk):
        end = min(begin + chunk, seqlen)
        indices = topk_idxs[begin:end].to(torch.int64)
        selected_kv = kv[indices].to(torch.float32)
        scores = torch.einsum("qhd,qkd->qhk", q[begin:end].to(torch.float32), selected_kv) * softmax_scale
        lse_kv = torch.logsumexp(scores, dim=-1)
        lse_with_sink = torch.logaddexp(lse_kv, attn_sink.view(1, num_heads))
        probabilities = torch.exp(scores - lse_with_sink.unsqueeze(-1))
        out[begin:end] = torch.einsum("qhk,qkd->qhd", probabilities, selected_kv).to(torch.bfloat16)
        lse[begin:end] = lse_kv
    return out, lse


@torch.no_grad()
def make_case(seqlen: int, topk: int) -> BenchmarkCase:
    device = torch.device("cuda")
    torch.manual_seed(20260813)

    q = (torch.randn((seqlen, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    kv = (torch.randn((seqlen, HEAD_DIM), device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    dout = (torch.randn((seqlen, HEADS, HEAD_DIM_V), device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    attn_sink = torch.linspace(-2.0, 2.0, HEADS, device=device, dtype=torch.float32)

    rows = torch.arange(seqlen, device=device, dtype=torch.int32).unsqueeze(1)
    columns = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0)
    topk_idxs = torch.remainder(rows * 131 + columns, seqlen).contiguous()
    topk_length = torch.full((seqlen,), topk, device=device, dtype=torch.int32)
    softmax_scale = HEAD_DIM**-0.5
    out, lse = reference_forward(q, kv, attn_sink, topk_idxs, softmax_scale)

    workspace_shapes = []
    for kernel_cls in (FlashAttentionDSABackwardSm100, FlashAttentionDSABackwardSm100TwoCTA):
        workspace_shapes.append(
            (
                kernel_cls._get_workspace_size_LSE_OdO(seqlen, HEAD_DIM, HEADS, 1, cutlass.Float32),
                kernel_cls._get_workspace_size_dKV(seqlen, HEAD_DIM, 1, cutlass.Float32),
            )
        )
    if workspace_shapes[0] != workspace_shapes[1]:
        raise RuntimeError(f"A/B workspace contracts differ: baseline={workspace_shapes[0]}, candidate={workspace_shapes[1]}")

    workspace_lse_odo_shape, workspace_dkv_shape = workspace_shapes[0]
    return BenchmarkCase(
        problem_shape=(seqlen, seqlen, HEAD_DIM, (HEADS, 1)),
        q=q,
        kv=kv,
        out=out,
        dout=dout,
        lse=lse,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        topk_length=topk_length,
        dq=torch.empty_like(q),
        dkv=torch.zeros_like(kv),
        d_sink=torch.zeros_like(attn_sink),
        workspace_lse_odo=torch.zeros(workspace_lse_odo_shape, device=device, dtype=torch.uint8),
        workspace_dkv=torch.zeros(workspace_dkv_shape, device=device, dtype=torch.uint8),
        softmax_scale=softmax_scale,
    )


def compile_kernel(kernel_cls: type, case: BenchmarkCase, compile_args: tuple[Any, ...]) -> tuple[Any, float]:
    kernel = kernel_cls(
        element_dtype=cutlass.BFloat16,
        head_dim=HEAD_DIM,
        head_dim_v=HEAD_DIM_V,
        block_tile=BLOCK_TILE,
        max_topk=case.topk_idxs.shape[1],
    )
    started = time.perf_counter()
    compiled = cute.compile(kernel, *compile_args, options=compile_options())
    return compiled, time.perf_counter() - started


@torch.no_grad()
def reset_outputs(case: BenchmarkCase) -> None:
    case.dq.zero_()
    case.dkv.zero_()
    case.d_sink.zero_()
    case.workspace_lse_odo.zero_()
    case.workspace_dkv.zero_()


@torch.no_grad()
def launch(compiled: Any, runtime_args: tuple[Any, ...], case: BenchmarkCase) -> None:
    reset_outputs(case)
    compiled(*runtime_args)


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    actual_fp32 = actual.to(torch.float32)
    expected_fp32 = expected.to(torch.float32)
    difference = torch.abs(actual_fp32 - expected_fp32)
    denominator = torch.clamp(torch.abs(expected_fp32), min=1e-7)
    return {
        "max_abs": float(difference.max().item()),
        "max_rel": float((difference / denominator).max().item()),
        "nonfinite": int((~torch.isfinite(actual_fp32)).sum().item()),
    }


@torch.no_grad()
def correctness_gate(
    compiled_baseline: Any,
    compiled_candidate: Any,
    runtime_args: tuple[Any, ...],
    case: BenchmarkCase,
) -> dict[str, Any]:
    launch(compiled_baseline, runtime_args, case)
    torch.cuda.synchronize()
    expected = {
        "dq": case.dq.clone(),
        "dkv": case.dkv.clone(),
        "d_sink": case.d_sink.clone(),
    }

    launch(compiled_candidate, runtime_args, case)
    torch.cuda.synchronize()
    actual = {"dq": case.dq, "dkv": case.dkv, "d_sink": case.d_sink}

    details: dict[str, Any] = {}
    for name in ("dq", "dkv", "d_sink"):
        details[name] = tensor_error(actual[name], expected[name])
        try:
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=CORRECTNESS_ATOL,
                rtol=CORRECTNESS_RTOL,
            )
        except AssertionError as error:
            raise RuntimeError(f"cross-implementation correctness gate failed for {name}: {error}") from error

    return {
        "passed": True,
        "shape": {
            "seqlen_q": case.q.shape[0],
            "seqlen_kv": case.kv.shape[0],
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "topk": case.topk_idxs.shape[1],
        },
        "atol": CORRECTNESS_ATOL,
        "rtol": CORRECTNESS_RTOL,
        "outputs": details,
    }


class EventTimer:
    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # Materialize both events before collecting a sample.
        self.start.record()
        self.end.record()
        self.end.synchronize()

    @torch.no_grad()
    def measure(self, compiled: Any, runtime_args: tuple[Any, ...], case: BenchmarkCase) -> float:
        reset_outputs(case)
        self.start.record()
        compiled(*runtime_args)
        self.end.record()
        self.end.synchronize()
        return float(self.start.elapsed_time(self.end))


def pair_order(pair_index: int) -> tuple[str, str]:
    # Adjacent pairs are AB then BA, i.e. one balanced ABBA block.
    if pair_index % 2 == 0:
        return BASELINE, CANDIDATE
    return CANDIDATE, BASELINE


@torch.no_grad()
def warm_up(
    compiled: dict[str, Any],
    runtime_args: tuple[Any, ...],
    case: BenchmarkCase,
    warmup_pairs: int,
) -> None:
    for pair_index in range(warmup_pairs):
        for name in pair_order(pair_index):
            launch(compiled[name], runtime_args, case)
    torch.cuda.synchronize()


def collect_samples(
    compiled: dict[str, Any],
    runtime_args: tuple[Any, ...],
    case: BenchmarkCase,
    paired_samples: int,
) -> list[dict[str, Any]]:
    timer = EventTimer()
    samples = []
    for pair_index in range(paired_samples):
        order = pair_order(pair_index)
        measurements = {}
        for name in order:
            measurements[name] = timer.measure(compiled[name], runtime_args, case)
        samples.append(
            {
                "pair": pair_index,
                "order": "AB" if order[0] == BASELINE else "BA",
                "baseline_ms": measurements[BASELINE],
                "candidate_ms": measurements[CANDIDATE],
                "candidate_over_baseline": measurements[CANDIDATE] / measurements[BASELINE],
            }
        )
    return samples


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "pstdev_ms": statistics.pstdev(values),
    }


def backward_flops(seqlen: int, topk: int) -> int:
    return 2 * seqlen * HEADS * topk * (3 * HEAD_DIM + 2 * HEAD_DIM_V)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seqlen", type=int, default=4096, help="query and KV sequence length (default: 4096)")
    parser.add_argument("--topk", type=int, default=2048, help="sparse keys per query; positive and 64-aligned (default: 2048)")
    parser.add_argument(
        "--warmup-pairs",
        type=int,
        default=4,
        help="number of untimed alternating A/B pairs; must be nonnegative and even (default: 4)",
    )
    parser.add_argument(
        "--paired-samples",
        type=int,
        default=20,
        help="number of timed alternating A/B pairs; must be positive and even (default: 20)",
    )
    parser.add_argument("--json", type=Path, default=None, metavar="PATH", help="write full gate, raw samples, and summary to PATH")
    args = parser.parse_args()

    if args.seqlen <= 0:
        parser.error("--seqlen must be positive")
    if args.topk <= 0 or args.topk > args.seqlen or args.topk % 64 != 0:
        parser.error("--topk must be positive, no larger than --seqlen, and divisible by 64")
    if args.warmup_pairs < 0 or args.warmup_pairs % 2 != 0:
        parser.error("--warmup-pairs must be nonnegative and even so warm-up order is ABBA-balanced")
    if args.paired_samples <= 0 or args.paired_samples % 2 != 0:
        parser.error("--paired-samples must be positive and even so timed order is ABBA-balanced")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise RuntimeError(f"this fixed SM100 benchmark requires compute capability 10.0, found {capability[0]}.{capability[1]}")

    device_name = torch.cuda.get_device_name()
    print(f"DSA BWD direct A/B on {device_name}: BF16 H={HEADS} D={HEAD_DIM} S={args.seqlen} topk={args.topk}")
    print("baseline=sm100, candidate=sm100_2_cta; shared inputs/outputs/workspaces; timing order=ABBA")

    stream = resolve_stream()
    case = make_case(args.seqlen, args.topk)
    compile_args = case.compile_args(stream)
    runtime_args = case.runtime_args(stream)

    compiled_baseline, baseline_compile_s = compile_kernel(FlashAttentionDSABackwardSm100, case, compile_args)
    compiled_candidate, candidate_compile_s = compile_kernel(FlashAttentionDSABackwardSm100TwoCTA, case, compile_args)
    compiled = {BASELINE: compiled_baseline, CANDIDATE: compiled_candidate}
    print(f"compile (excluded): baseline={baseline_compile_s:.2f}s candidate={candidate_compile_s:.2f}s")

    gate = correctness_gate(compiled_baseline, compiled_candidate, runtime_args, case)
    print("correctness gate: PASS " + ", ".join(f"{name} max_abs={metrics['max_abs']:.6g}" for name, metrics in gate["outputs"].items()))

    torch.cuda.empty_cache()
    warm_up(compiled, runtime_args, case, args.warmup_pairs)
    samples = collect_samples(compiled, runtime_args, case, args.paired_samples)

    baseline_values = [sample["baseline_ms"] for sample in samples]
    candidate_values = [sample["candidate_ms"] for sample in samples]
    baseline_summary = summarize(baseline_values)
    candidate_summary = summarize(candidate_values)
    paired_ratios = [sample["candidate_over_baseline"] for sample in samples]
    flops = backward_flops(args.seqlen, args.topk)
    baseline_summary["mean_tflops"] = flops / (baseline_summary["mean_ms"] * 1e-3) / 1e12
    candidate_summary["mean_tflops"] = flops / (candidate_summary["mean_ms"] * 1e-3) / 1e12

    comparison = {
        "candidate_speedup_over_baseline": baseline_summary["mean_ms"] / candidate_summary["mean_ms"],
        "candidate_latency_delta_percent": (candidate_summary["mean_ms"] / baseline_summary["mean_ms"] - 1.0) * 100.0,
        "paired_ratio_mean": statistics.fmean(paired_ratios),
        "paired_ratio_median": statistics.median(paired_ratios),
    }
    print(
        f"baseline : {baseline_summary['mean_ms']:.4f} ms mean, {baseline_summary['median_ms']:.4f} ms median, "
        f"{baseline_summary['mean_tflops']:.2f} TFLOP/s"
    )
    print(
        f"candidate: {candidate_summary['mean_ms']:.4f} ms mean, {candidate_summary['median_ms']:.4f} ms median, "
        f"{candidate_summary['mean_tflops']:.2f} TFLOP/s"
    )
    print(
        f"candidate latency delta={comparison['candidate_latency_delta_percent']:+.2f}%, "
        f"candidate speedup over baseline={comparison['candidate_speedup_over_baseline']:.4f}x"
    )

    result = {
        "schema_version": 1,
        "device": {"name": device_name, "compute_capability": [capability[0], capability[1]]},
        "configuration": {
            "dtype": "bfloat16",
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "head_dim_v": HEAD_DIM_V,
            "seqlen_q": args.seqlen,
            "seqlen_kv": args.seqlen,
            "topk": args.topk,
            "warmup_pairs": args.warmup_pairs,
            "paired_samples": args.paired_samples,
            "timing_order": "ABBA",
            "reset_excluded": True,
            "compile_excluded": True,
        },
        "implementations": {
            "baseline": FlashAttentionDSABackwardSm100.__name__,
            "candidate": FlashAttentionDSABackwardSm100TwoCTA.__name__,
        },
        "compile_seconds": {"baseline": baseline_compile_s, "candidate": candidate_compile_s},
        "correctness": gate,
        "samples": samples,
        "summary": {"baseline": baseline_summary, "candidate": candidate_summary, "comparison": comparison},
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
