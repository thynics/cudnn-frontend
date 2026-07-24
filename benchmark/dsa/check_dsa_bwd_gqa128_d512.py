#!/usr/bin/env python3
"""Single-case correctness smoke test for SM100 DSA backward.

The fixed shape targets the optimization configuration directly:
GQA=128 (one shared KV head), d_qk=d_v=512, BF16 inputs, and sparse TopK.
The kernel gradients are compared with the repository's FP32 PyTorch
autograd reference for dQ, dKV, and dSink.
"""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

import cudnn
from cudnn import DSA
from fe_api.dsa.dsa_reference import (
    check_ref_dsa_sparse_attention_backward,
    ref_sparse_attention_forward,
)


SEED = 0
SEQLEN_Q = 64
SEQLEN_KV = 512
TOPK = 128
NUM_HEADS = 128
HEAD_DIM = 512
ATOL = 5e-2
RTOL = 5e-2


def make_inputs():
    torch.manual_seed(SEED)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(SEQLEN_Q, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    kv = torch.randn(SEQLEN_KV, HEAD_DIM, device=device, dtype=dtype)
    dout = torch.randn_like(q)
    attn_sink = torch.linspace(-2.0, 2.0, NUM_HEADS, device=device, dtype=torch.float32)

    # Unique sparse indices per query row, matching the performance benchmark.
    topk_idxs = torch.rand(SEQLEN_Q, SEQLEN_KV, device=device).argsort(dim=-1)[:, :TOPK].to(torch.int32)
    topk_length = torch.full((SEQLEN_Q,), TOPK, device=device, dtype=torch.int32)
    return q, kv, dout, attn_sink, topk_idxs, topk_length


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"This smoke test targets SM100, found SM{major}{minor}")

    print(f"Device: {torch.cuda.get_device_name()} (SM{major}{minor})")
    print(f"cuDNN Frontend: {Path(cudnn.__file__).resolve()}")
    print(
        f"Config: seqlen_q={SEQLEN_Q}, seqlen_kv={SEQLEN_KV}, topk={TOPK}, "
        f"gqa={NUM_HEADS}, d_qk=d_v={HEAD_DIM}, dtype=bfloat16, seed={SEED}"
    )

    q, kv, dout, attn_sink, topk_idxs, topk_length = make_inputs()
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    check_ref_dsa_sparse_attention_backward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        dout,
        lse,
        result["dq"],
        result["dkv"],
        result["d_sink"],
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        atol=ATOL,
        rtol=RTOL,
    )
    print("PASS: SM100 DSA backward GQA=128, D=512")


if __name__ == "__main__":
    main()
