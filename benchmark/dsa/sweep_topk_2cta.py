#!/usr/bin/env python3
"""Sweep top-k for the production baseline vs a registered 2-CTA candidate.

For each top-k, benchmarks (a) the public
``cudnn.DSA.sparse_attention_backward_wrapper`` (production SM100 baseline)
and (b) a registered ``dsa_bwd_sm100_2cta_<impl>.py`` implementation on the
same inputs, reporting latency, TFLOPS, and a baseline-vs-candidate output
cross-check (max abs diff on dq/dkv).

Usage (on a B200 node with the repo's python/ importable):

    python3 benchmark/dsa/sweep_topk_2cta.py --impl final \
        --topks 128,256,512,1024,2048 --json sweep_results.json

Timing convention matches benchmark_dsa_sparse_attention_backward.py
(CUDA events around ``repeat`` calls, including per-call dkv zeroing and
workspace handling, mirroring the public-wrapper cost model).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import benchmark_dsa_sparse_attention_backward as bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impl", default="final")
    p.add_argument("--class-name",
                   default="FlashAttentionDSABackwardSm100TwoCTAV2")
    p.add_argument("--topks", type=bench.comma_separated_ints,
                   default=[128, 256, 512, 1024, 2048])
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--nheads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def build_case(seqlen: int, topk: int, nheads: int, head_dim: int):
    q, kv, attn_sink, dout, topk_idxs, topk_length = bench.make_inputs(
        seqlen, topk, seqlen, nheads, head_dim, head_dim,
        torch.bfloat16, use_attn_sink=True, use_topk_length=True)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    out, lse = bench.reference_forward(
        q, kv, attn_sink, topk_idxs, softmax_scale, head_dim)
    return dict(q=q, kv=kv, out=out, dout=dout, lse=lse,
                attn_sink=attn_sink, topk_idxs=topk_idxs,
                topk_length=topk_length, softmax_scale=softmax_scale)


def time_run(run, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def baseline_leg(case, args):
    from cudnn import DSA
    q = case["q"]
    dq = torch.empty_like(q)
    dkv = torch.zeros_like(case["kv"])

    def run():
        DSA.sparse_attention_backward_wrapper(
            q, case["kv"], case["out"], case["dout"], case["lse"],
            case["attn_sink"], case["topk_idxs"],
            softmax_scale=case["softmax_scale"],
            topk_length=case["topk_length"], dq=dq, dkv=dkv)

    ms = time_run(run, args.warmup, args.repeat)
    return ms, dq, dkv


def candidate_leg(case, topk: int, args):
    import cutlass
    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.utils.compiler import (
        compile_options,
    )
    from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
        to_cute_tensor,
    )

    impl_mod = importlib.import_module(
        "cudnn.deepseek_sparse_attention.sparse_attention_backward."
        f"dsa_bwd_sm100_2cta_{args.impl}")
    impl_cls = getattr(impl_mod, args.class_name)

    q, kv = case["q"], case["kv"]
    S_q, H, D = q.shape
    S_kv = kv.shape[0]
    device = q.device

    dq = torch.empty_like(q)
    dkv = torch.zeros_like(kv)
    d_sink = torch.zeros_like(case["attn_sink"])
    acc = cutlass.Float32
    ws_lse = torch.zeros(
        *impl_cls._get_workspace_size_LSE_OdO(S_q, D, H, 1, acc),
        dtype=torch.uint8, device=device)
    ws_dkv = torch.zeros(
        *impl_cls._get_workspace_size_dKV(S_kv, D, 1, acc),
        dtype=torch.uint8, device=device)

    kernel_obj = impl_cls(head_dim=D, head_dim_v=D, block_tile=64,
                          max_topk=topk)
    problem_shape = (S_q, S_kv, D, (H, 1))
    stream = resolve_stream(None)

    compiled = cute.compile(
        kernel_obj,
        problem_shape,
        to_cute_tensor(q, divisibility=D),
        to_cute_tensor(kv, divisibility=D),
        to_cute_tensor(case["out"], divisibility=D),
        to_cute_tensor(case["dout"], divisibility=D),
        to_cute_tensor(case["lse"], assumed_align=4),
        to_cute_tensor(case["attn_sink"]),
        to_cute_tensor(case["topk_idxs"]),
        to_cute_tensor(case["topk_length"]),
        to_cute_tensor(dq, divisibility=D),
        to_cute_tensor(dkv, divisibility=D),
        to_cute_tensor(d_sink),
        to_cute_tensor(ws_lse),
        to_cute_tensor(ws_dkv),
        None, 0, 0,  # trace_buffer / trace_token_idx / trace_batch_idx
        case["softmax_scale"],
        stream,
        options=compile_options(),
    )

    def run():
        dkv.fill_(0)
        ws_dkv.fill_(0)
        d_sink.fill_(0)
        compiled(problem_shape, q, kv, case["out"], case["dout"],
                 case["lse"], case["attn_sink"], case["topk_idxs"],
                 case["topk_length"], dq, dkv, d_sink, ws_lse, ws_dkv,
                 None, 0, 0, case["softmax_scale"], stream)

    ms = time_run(run, args.warmup, args.repeat)
    run()
    torch.cuda.synchronize()
    return ms, dq, dkv


def main() -> int:
    args = parse_args()
    seqlen, H, D = args.seqlen, args.nheads, args.head_dim
    print(f"sweep impl={args.impl} class={args.class_name} "
          f"seqlen={seqlen} nheads={H} head_dim={D} "
          f"warmup={args.warmup} repeat={args.repeat} "
          f"device={torch.cuda.get_device_name()}")
    header = (f"{'topk':<6} {'base ms':<9} {'base TF':<8} "
              f"{'2cta ms':<9} {'2cta TF':<8} {'ratio':<7} "
              f"{'max|d_dq|':<10} {'max|d_dkv|':<10}")
    print(header)
    rows = []
    for topk in args.topks:
        row = {"topk": topk, "seqlen": seqlen, "nheads": H, "head_dim": D}
        try:
            if topk % 64:
                raise ValueError("topk must be a multiple of 64")
            case = build_case(seqlen, topk, H, D)
            flops = bench.flops_bwd(seqlen, topk, H, D, D)
            base_ms, base_dq, base_dkv = baseline_leg(case, args)
            cand_ms, cand_dq, cand_dkv = candidate_leg(case, topk, args)
            row.update(
                baseline_ms=round(base_ms, 4),
                baseline_tflops=round(flops / (base_ms * 1e-3) / 1e12, 2),
                candidate_ms=round(cand_ms, 4),
                candidate_tflops=round(flops / (cand_ms * 1e-3) / 1e12, 2),
                ratio=round(cand_ms / base_ms, 4),
                max_abs_diff_dq=float((base_dq - cand_dq).abs().max()),
                max_abs_diff_dkv=float((base_dkv - cand_dkv).abs().max()),
            )
            print(f"{topk:<6} {row['baseline_ms']:<9.3f} "
                  f"{row['baseline_tflops']:<8.1f} "
                  f"{row['candidate_ms']:<9.3f} "
                  f"{row['candidate_tflops']:<8.1f} "
                  f"{row['ratio']:<7.3f} "
                  f"{row['max_abs_diff_dq']:<10.4f} "
                  f"{row['max_abs_diff_dkv']:<10.4f}")
            del case
        except Exception as e:  # keep sweeping the remaining points
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"{topk:<6} ERROR: {row['error']}")
        rows.append(row)
        torch.cuda.empty_cache()
    print("SWEEP_JSON " + json.dumps(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"SWEEP_RESULT {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
