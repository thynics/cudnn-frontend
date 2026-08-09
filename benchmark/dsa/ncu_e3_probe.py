#!/usr/bin/env python3
"""Candidate-only single-topk run for NCU profiling (E3-NCU attribution).

r2 lesson: an ncu --launch-skip window counted from process start lands
inside build_case's reference machinery (dozens of torch/cuBLAS
launches) and never reaches the DSA kernels.  This probe therefore
wraps ONLY the compiled DSA call in an NVTX range named ``dsa_bwd`` so
the runner can gate profiling with ``--nvtx --nvtx-include "dsa_bwd/"``:
the gated launch stream then contains exactly the DSA kernel sequence
(the torch fills stay outside the range).

The candidate leg mirrors sweep_topk_2cta.candidate_leg verbatim except
for the NVTX range.  The active profile (compat vs e3pad) is selected
by the DSA_RUBIN1_* env, exactly as in run_e3pair_gr100.sh.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "sweep_topk_2cta", HERE / "sweep_topk_2cta.py"
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

import torch  # noqa: E402  (after sweep import sets sys.path)


def candidate_leg_nvtx(case, topk: int, warmup: int, repeat: int):
    """sweep_topk_2cta.candidate_leg with an NVTX range around the DSA call."""
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
        "dsa_bwd_sm100_2cta_rubin_1")
    impl_cls = impl_mod.FlashAttentionDSABackwardSm100TwoCTAV2

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
        # Fills stay OUTSIDE the NVTX range: the gated launch stream is
        # exactly the DSA kernel sequence.
        dkv.fill_(0)
        ws_dkv.fill_(0)
        d_sink.fill_(0)
        torch.cuda.nvtx.range_push("dsa_bwd")
        compiled(problem_shape, q, kv, case["out"], case["dout"],
                 case["lse"], case["attn_sink"], case["topk_idxs"],
                 case["topk_length"], dq, dkv, d_sink, ws_lse, ws_dkv,
                 None, 0, 0, case["softmax_scale"], stream)
        torch.cuda.nvtx.range_pop()

    ms = sweep.time_run(run, warmup, repeat)
    run()
    torch.cuda.synchronize()
    return ms, dq, dkv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=8)
    args = p.parse_args()

    case = sweep.build_case(4096, args.topk, 128, 512)
    ms, dq, dkv = candidate_leg_nvtx(
        case, args.topk, args.warmup, args.repeat
    )
    assert torch.isfinite(dq).all(), "non-finite dq"
    assert torch.isfinite(dkv).all(), "non-finite dkv"
    print(f"E3NCU_PROBE topk={args.topk} candidate_ms={ms:.4f}")
    print("E3NCU_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
