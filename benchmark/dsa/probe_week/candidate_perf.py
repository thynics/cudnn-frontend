#!/usr/bin/env python3
"""Time a registered 2-CTA DSA backward implementation without baseline JIT."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


OWN_REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REPO_ROOT = Path(
    os.environ.get("DSA_SOURCE_REPO", str(OWN_REPO_ROOT))
).resolve()
DSA_BENCH_DIR = SOURCE_REPO_ROOT / "benchmark" / "dsa"
sys.path.insert(0, str(SOURCE_REPO_ROOT / "python"))
sys.path.insert(0, str(DSA_BENCH_DIR))

import torch

import sweep_topk_2cta as sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", required=True)
    parser.add_argument(
        "--class-name", default="FlashAttentionDSABackwardSm100TwoCTAV2"
    )
    parser.add_argument("--topks", type=sweep.bench.comma_separated_ints,
                        default=[2048])
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--nheads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--reduce-dephase-ns", type=int, default=None)
    parser.add_argument("--reduce-pace-ns", type=int, default=None)
    parser.add_argument("--gather-dephase-ns", type=int, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--trace-token", type=int, default=8)
    parser.add_argument("--trace-batch", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reduce_dephase_ns is not None and args.reduce_dephase_ns < 0:
        raise ValueError("reduce dephase must be non-negative")
    if args.reduce_pace_ns is not None and args.reduce_pace_ns < 0:
        raise ValueError("reduce pace must be non-negative")
    if args.gather_dephase_ns is not None and args.gather_dephase_ns < 0:
        raise ValueError("gather dephase must be non-negative")
    rows = []
    print(
        f"candidate-only impl={args.impl} class={args.class_name} "
        f"seqlen={args.seqlen} nheads={args.nheads} "
        f"head_dim={args.head_dim} warmup={args.warmup} "
        f"repeat={args.repeat} device={torch.cuda.get_device_name()}"
    )
    for topk in args.topks:
        if topk % 64:
            raise ValueError("topk must be a multiple of 64")
        case = sweep.build_case(
            args.seqlen, topk, args.nheads, args.head_dim
        )
        flops = sweep.bench.flops_bwd(
            args.seqlen, topk, args.nheads, args.head_dim, args.head_dim
        )
        candidate_ms, dq, dkv = sweep.candidate_leg(case, topk, args)
        row = {
            "impl": args.impl,
            "topk": topk,
            "candidate_ms": round(candidate_ms, 6),
            "candidate_tflops": round(
                flops / (candidate_ms * 1e-3) / 1e12, 3
            ),
            "dq_abs_max": float(dq.abs().max()),
            "dkv_abs_max": float(dkv.abs().max()),
        }
        rows.append(row)
        print("CANDIDATE_RESULT " + json.dumps(row, sort_keys=True))
        del case, dq, dkv
        torch.cuda.empty_cache()
    print("CANDIDATE_JSON " + json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
