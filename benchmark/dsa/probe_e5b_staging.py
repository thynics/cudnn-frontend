#!/usr/bin/env python3
"""E5b-0 probe: staged score-K TMA consumer vs the e4ca gather control.

Milestone E5b of E5_STAGING_DESIGN.md, perf-only judgment leg:

  1. Build backward inputs at a reduced perf shape (tokens=512,
     S_kv=4096, topk=2048, H=128, D=512, bf16; staging = 1 GB).
  2. Pre-produce the contiguous K staging tensor with torch
     (index_select; -1/hole rows zero-filled) and report its cost --
     E5b consumes it offline, the wave-pipelined producer is E5c.
  3. cute.compile + run BOTH the e4ca class (control) and the e5b
     class directly on identical inputs/workspaces, fair-boundary
     style (direct compiled calls, ABBA paired samples, accumulator
     resets outside the timed interval, medians + paired ratio).
  4. Crosscheck e5b outputs against e4ca (gate: dq exactly 0,
     dkv <= 0.002).
  5. Print one JSON report.

Run via:  ./benchmark/dsa/run_remote_probe.sh benchmark/dsa/probe_e5b_staging.py
"""

import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path[:0] = ["python", "test/python", "."]

# E5B_TOKENS sweeps the wave count: 512 (default, ~7 waves, 1 GB cold
# staging stream), 74 (exactly one full wave), 8 (16 MB staging, L2-hot
# across sample loops -- discriminates memory-system tax vs in-kernel
# protocol cost).
TOKENS = int(os.environ.get("E5B_TOKENS", "512"))
S_KV = 4096
TOPK = 2048
HEADS = 128
D = 512
TOPK_PAD = (TOPK + 63) // 64 * 64

WARMUP_MIN_S = 1.0
WARMUP_BLOCK_PAIRS = 8
PAIRED_SAMPLES = 48
SAMPLE_BLOCK_PAIRS = 12
PRODUCE_REPEAT = 10

VARIANT_DIR = Path(
    "python/cudnn/deepseek_sparse_attention/sparse_attention_backward"
)
VARIANT_PKG = "cudnn.deepseek_sparse_attention.sparse_attention_backward"


def load_variant_class(filename, alias):
    """Import a self-contained variant module directly by file path.

    The module name is planted inside the real package so its relative
    `from .dsa_bwd_sm100 import ...` resolves; the active-source
    dsa_bwd_sm100_2cta_v2.py slot is never touched.
    """

    importlib.import_module(VARIANT_PKG)
    spec = importlib.util.spec_from_file_location(
        f"{VARIANT_PKG}.{alias}",
        VARIANT_DIR / filename,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FlashAttentionDSABackwardSm100TwoCTAV2


def produce_staging(kv, indices, lengths, staging, chunk_tokens=64):
    """Gather sparse KV rows into the contiguous staging tensor.

    staging[token, slot] = kv[indices[token, slot]] for valid slots
    (slot < lengths[token] and index >= 0); every other row -- holes,
    beyond-length slots, and the topk->topk_padded tail -- is exactly
    zero, preserving the kernel's base zero-fill semantics bitwise.
    """

    tokens, topk = indices.shape
    device = indices.device
    slot_ids = torch.arange(topk, device=device)
    for begin in range(0, tokens, chunk_tokens):
        end = min(begin + chunk_tokens, tokens)
        idx = indices[begin:end].long()
        valid = idx >= 0
        if lengths is not None:
            valid &= slot_ids[None, :] < lengths[begin:end, None].long()
        safe = torch.where(valid, idx, torch.zeros_like(idx))
        rows = kv.index_select(0, safe.reshape(-1)).reshape(
            end - begin, topk, kv.shape[1]
        )
        rows = torch.where(valid[..., None], rows, torch.zeros_like(rows))
        staging[begin:end, :topk] = rows
    if staging.shape[1] > topk:
        staging[:, topk:] = 0


def bench_events(fn, repeat, sync_each=False):
    values = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
        if sync_each:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    values = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return values


def relative_iqr(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (q3 - q1) / statistics.median(values)


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(20260811)

    import cutlass
    import cutlass.cute as cute
    from benchmark.dsa import benchmark_dsa_sparse_attention_backward as bench
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
    from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
        to_cute_tensor,
    )

    e4ca_cls = load_variant_class(
        "dsa_bwd_sm100_2cta_e4ca.py", "dsa_bwd_e4ca_probe"
    )
    e5b_cls = load_variant_class(
        "dsa_bwd_sm100_2cta_e5b.py", "dsa_bwd_e5b_probe"
    )

    q, kv, sink, dout, indices, lengths = bench.make_inputs(
        TOKENS,
        TOPK,
        S_KV,
        HEADS,
        D,
        D,
        torch.bfloat16,
        use_attn_sink=True,
        use_topk_length=True,
    )
    scale = 1.0 / math.sqrt(D)
    out, lse = bench.reference_forward(q, kv, sink, indices, scale, D)

    # ------------------------------------------------------------------
    # Staging producer (torch, offline for E5b).
    # ------------------------------------------------------------------
    staging = torch.zeros(
        TOKENS, TOPK_PAD, D, device="cuda", dtype=torch.bfloat16
    )
    produce = lambda: produce_staging(kv, indices, lengths, staging)
    produce()
    torch.cuda.synchronize()
    produce_ms = statistics.median(
        bench_events(produce, PRODUCE_REPEAT, sync_each=True)
    )

    # ------------------------------------------------------------------
    # Direct compiled runners on shared tensors/workspaces.
    # ------------------------------------------------------------------
    def workspace_shapes(impl_cls):
        accumulator = cutlass.Float32
        return (
            tuple(
                int(v)
                for v in impl_cls._get_workspace_size_LSE_OdO(
                    TOKENS, D, HEADS, 1, accumulator
                )
            ),
            tuple(
                int(v)
                for v in impl_cls._get_workspace_size_dKV(
                    S_KV, D, 1, accumulator
                )
            ),
        )

    control_shapes = workspace_shapes(e4ca_cls)
    candidate_shapes = workspace_shapes(e5b_cls)
    if control_shapes != candidate_shapes:
        raise RuntimeError(
            f"workspace mismatch: {control_shapes} vs {candidate_shapes}"
        )

    buffers = {
        "dq": torch.empty_like(q),
        "dkv": torch.zeros_like(kv),
        "d_sink": torch.zeros_like(sink),
        "workspace_lse_odo": torch.zeros(
            *control_shapes[0], dtype=torch.uint8, device="cuda"
        ),
        "workspace_dkv": torch.zeros(
            *control_shapes[1], dtype=torch.uint8, device="cuda"
        ),
    }
    problem_shape = (TOKENS, S_KV, D, (HEADS, 1))
    stream = resolve_stream(None)

    def build_direct_runner(impl_cls, with_staging):
        kernel = impl_cls(
            head_dim=D,
            head_dim_v=D,
            block_tile=64,
            max_topk=TOPK,
        )
        prototypes = [
            to_cute_tensor(q, divisibility=D),
            to_cute_tensor(kv, divisibility=D),
            to_cute_tensor(out, divisibility=D),
            to_cute_tensor(dout, divisibility=D),
            to_cute_tensor(lse, assumed_align=4),
            to_cute_tensor(sink),
            to_cute_tensor(indices),
            to_cute_tensor(lengths),
        ]
        runtime = [q, kv, out, dout, lse, sink, indices, lengths]
        if with_staging:
            # e5b's __call__ takes mKStage right after mTopkLength.
            prototypes.append(to_cute_tensor(staging, divisibility=D))
            runtime.append(staging)
        prototypes.extend(
            [
                to_cute_tensor(buffers["dq"], divisibility=D),
                to_cute_tensor(buffers["dkv"], divisibility=D),
                to_cute_tensor(buffers["d_sink"]),
                to_cute_tensor(buffers["workspace_lse_odo"]),
                to_cute_tensor(buffers["workspace_dkv"]),
                None,
                0,
                0,
                scale,
                stream,
            ]
        )
        runtime.extend(
            [
                buffers["dq"],
                buffers["dkv"],
                buffers["d_sink"],
                buffers["workspace_lse_odo"],
                buffers["workspace_dkv"],
                None,
                0,
                0,
                scale,
                stream,
            ]
        )
        compiled = cute.compile(
            kernel,
            problem_shape,
            *prototypes,
            options=compile_options(),
        )
        torch.cuda.synchronize()

        def launch():
            compiled(problem_shape, *runtime)

        return launch

    def reset_accumulators():
        buffers["dkv"].zero_()
        buffers["workspace_dkv"].zero_()
        buffers["d_sink"].zero_()

    runners = {
        "e4ca": build_direct_runner(e4ca_cls, False),
        "e5b": build_direct_runner(e5b_cls, True),
    }

    def pair_order(index):
        return (
            ("e4ca", "e5b"),
            ("e5b", "e4ca"),
            ("e5b", "e4ca"),
            ("e4ca", "e5b"),
        )[index % 4]

    # Warmup (ABBA, resets included, off the record).
    import time

    warmup_pairs = 0
    warmup_started = time.monotonic()
    while True:
        for index in range(warmup_pairs, warmup_pairs + WARMUP_BLOCK_PAIRS):
            for name in pair_order(index):
                reset_accumulators()
                runners[name]()
        torch.cuda.synchronize()
        warmup_pairs += WARMUP_BLOCK_PAIRS
        if time.monotonic() - warmup_started >= WARMUP_MIN_S:
            break

    # ------------------------------------------------------------------
    # Crosscheck at the exact timed shape.
    # ------------------------------------------------------------------
    reset_accumulators()
    runners["e4ca"]()
    torch.cuda.synchronize()
    control_outputs = {
        name: buffers[name].clone() for name in ("dq", "dkv", "d_sink")
    }
    reset_accumulators()
    runners["e5b"]()
    torch.cuda.synchronize()
    crosscheck = {
        f"max_abs_diff_{name}": float(
            (control_outputs[name].float() - buffers[name].float())
            .abs()
            .max()
        )
        for name in control_outputs
    }
    crosscheck["all_outputs_finite"] = all(
        bool(torch.isfinite(t.float()).all())
        for t in (
            *control_outputs.values(),
            buffers["dq"],
            buffers["dkv"],
            buffers["d_sink"],
        )
    )
    crosscheck["gate"] = (
        "PASS"
        if crosscheck["all_outputs_finite"]
        and crosscheck["max_abs_diff_dq"] == 0.0
        and crosscheck["max_abs_diff_dkv"] <= 0.002
        else "FAIL"
    )

    # ------------------------------------------------------------------
    # Timed ABBA blocks (resets ordered before the start event).
    # ------------------------------------------------------------------
    samples = {"e4ca": [], "e5b": []}

    def time_block(first_pair, pair_count):
        records = []
        for index in range(first_pair, first_pair + pair_count):
            for name in pair_order(index):
                records.append(
                    (
                        name,
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                )
        for name, start, end in records:
            reset_accumulators()
            start.record()
            runners[name]()
            end.record()
        torch.cuda.synchronize()
        return [(n, s.elapsed_time(e)) for n, s, e in records]

    while len(samples["e4ca"]) < PAIRED_SAMPLES:
        first = len(samples["e4ca"])
        count = min(SAMPLE_BLOCK_PAIRS, PAIRED_SAMPLES - first)
        for name, elapsed_ms in time_block(first, count):
            samples[name].append(elapsed_ms)

    e4ca_median = statistics.median(samples["e4ca"])
    e5b_median = statistics.median(samples["e5b"])
    paired_ratios = [
        b / a for a, b in zip(samples["e4ca"], samples["e5b"])
    ]
    tiles_per_token = TOPK // 64
    delta_us_per_token = (e5b_median - e4ca_median) * 1e3 / TOKENS

    report = {
        "shape": {
            "tokens": TOKENS,
            "s_kv": S_KV,
            "topk": TOPK,
            "topk_padded": TOPK_PAD,
            "heads": HEADS,
            "head_dim": D,
            "staging_gb": round(
                TOKENS * TOPK_PAD * D * 2 / 2**30, 3
            ),
        },
        "staging_produce_ms": round(produce_ms, 4),
        "staging_produce_us_per_token": round(
            produce_ms * 1e3 / TOKENS, 3
        ),
        "e4ca_median_ms": round(e4ca_median, 5),
        "e5b_median_ms": round(e5b_median, 5),
        "paired_ratio_e5b_over_e4ca": round(
            statistics.median(paired_ratios), 5
        ),
        "paired_ratio_relative_iqr": round(
            relative_iqr(paired_ratios), 5
        ),
        "e4ca_relative_iqr": round(relative_iqr(samples["e4ca"]), 5),
        "e5b_relative_iqr": round(relative_iqr(samples["e5b"]), 5),
        "e4ca_us_per_token": round(e4ca_median * 1e3 / TOKENS, 3),
        "e5b_us_per_token": round(e5b_median * 1e3 / TOKENS, 3),
        "delta_us_per_token": round(delta_us_per_token, 3),
        "delta_us_per_tile": round(
            delta_us_per_token / tiles_per_token, 4
        ),
        "gate_delta_le_minus_0p5us_per_tile": (
            delta_us_per_token / tiles_per_token <= -0.5
        ),
        "paired_samples": len(samples["e4ca"]),
        "crosscheck": crosscheck,
    }
    print(json.dumps(report, indent=2))
    if crosscheck["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
