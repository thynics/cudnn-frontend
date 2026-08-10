#!/usr/bin/env python3
"""E5a probe: K-staging producer economics on B200.

Answers the two gate questions of E5_STAGING_DESIGN.md before any main
kernel surgery:
  Q1  bare rate: how fast can one wave (74 tokens x topk 2048 x D512
      bf16 rows) be gathered from the KV tensor into a contiguous GMEM
      staging buffer?  (needs << 222us wave compute time)
  Q2  interference: run the same gather on a side stream concurrently
      with the production 2-CTA backward -- how much does each side
      lose?  (gate: main kernel slowdown < 3%)

Pure-torch producer stands in for the future staging kernel: a gather
of this shape is bandwidth-bound, so torch.index_select measures the
achievable floor within a few percent.
"""
import json
import math
import statistics
import sys
import time

import torch

sys.path[:0] = ["python", "test/python", "."]

S_KV, TOPK, D, HEADS = 4096, 2048, 512, 128
WAVE_TOKENS = 74
REPEAT = 50


def bench(fn, repeat=REPEAT, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(start, end))


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    torch.manual_seed(0)
    kv = torch.randn(S_KV, D, device=dev, dtype=torch.bfloat16)
    idx = torch.rand(WAVE_TOKENS, S_KV, device=dev).argsort(-1)[:, :TOPK].to(torch.int64)
    staging = torch.empty(WAVE_TOKENS, TOPK, D, device=dev, dtype=torch.bfloat16)

    def produce():
        torch.index_select(kv, 0, idx.reshape(-1), out=staging.view(-1, D))

    bare_ms = bench(produce)
    bytes_moved = 2 * WAVE_TOKENS * TOPK * D * 2  # read + write, bf16
    report = {
        "wave_tokens": WAVE_TOKENS,
        "staging_mb": round(WAVE_TOKENS * TOPK * D * 2 / 2**20, 1),
        "bare_ms_per_wave": round(bare_ms, 4),
        "bare_us_per_token": round(bare_ms * 1e3 / WAVE_TOKENS, 2),
        "effective_gbps": round(bytes_moved / (bare_ms * 1e-3) / 1e9, 1),
    }

    # Q2: concurrency with the production backward.
    from cudnn import DSA
    from fe_api.dsa.dsa_reference import ref_sparse_attention_forward
    q = torch.randn(64, HEADS, D, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(512, D, device=dev, dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    sink = torch.linspace(-2.0, 2.0, HEADS, device=dev)
    ind = torch.rand(64, 512, device=dev).argsort(-1)[:, :256].to(torch.int32)
    lengths = torch.full((64,), 256, device=dev, dtype=torch.int32)
    scale = 1.0 / math.sqrt(D)
    out, lse = ref_sparse_attention_forward(q, kv_c, sink, ind, topk_length=lengths, softmax_scale=scale)

    def bwd():
        DSA.sparse_attention_backward_wrapper(
            q, kv_c, out, dout, lse, sink, ind, softmax_scale=scale, topk_length=lengths
        )

    solo_bwd_ms = bench(bwd, repeat=30)
    side = torch.cuda.Stream()

    def bwd_with_producer():
        with torch.cuda.stream(side):
            produce()
        bwd()

    both_bwd_ms = bench(bwd_with_producer, repeat=30)
    report["solo_bwd_ms"] = round(solo_bwd_ms, 4)
    report["bwd_with_concurrent_producer_ms"] = round(both_bwd_ms, 4)
    report["interference_percent"] = round((both_bwd_ms / solo_bwd_ms - 1) * 100, 2)
    report["gate_interference_lt_3pct"] = report["interference_percent"] < 3.0
    report["gate_rate_fits_wave"] = bare_ms < 0.15  # 150us << 222us wave window

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
