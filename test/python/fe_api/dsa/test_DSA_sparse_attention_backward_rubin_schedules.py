# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical coverage of Rubin DSA backward's long-sequence schedules."""

import pytest
import torch

from fe_api.dsa.dsa_reference import ref_sparse_attention_forward_chunked
from fe_api.dsa.test_DSA_sparse_attention_backward_rubin import _require_rubin

pytestmark = pytest.mark.L1


def _check_gradients(q, kv, sink, indices, out, dout, lse, dq, dkv, d_sink, lengths, scale):
    """Check gradients against the saved-output backward equations.

    Backward consumes BF16 O and KV-only LSE. Recomputing FP32 O changes delta
    and hence all three gradient targets; that error accumulates for long SQ.
    The short-sequence tests retain the repository's autograd comparison.
    """
    dq_ref = torch.zeros_like(q, dtype=torch.float32)
    dkv_ref = torch.zeros_like(kv, dtype=torch.float32)
    d_sink_ref = torch.zeros_like(sink, dtype=torch.float64)
    positions = torch.arange(indices.shape[1], device=q.device)[None, :]
    for begin in range(0, q.shape[0], 32):
        end = min(begin + 32, q.shape[0])
        idx = indices[begin:end].long()
        valid = (idx >= 0) & (idx < kv.shape[0])
        if lengths is not None:
            valid &= positions < lengths[begin:end, None].clamp(0, indices.shape[1])
        safe = idx.clamp(0, kv.shape[0] - 1)
        keys = kv[safe].float()
        keys = torch.where(valid[..., None], keys, torch.zeros_like(keys))
        qf, do = q[begin:end].float(), dout[begin:end].float()
        scores = torch.einsum("chd,ckd->chk", qf, keys) * scale
        norm = torch.logaddexp(lse[begin:end], sink[None, :])
        scores.masked_fill_(~valid[:, None, :], float("-inf"))
        p = torch.exp(scores - norm[..., None]).masked_fill_(~valid[:, None, :], 0.0)
        delta = (out[begin:end].float() * do).sum(-1)
        dp = torch.einsum("chv,ckv->chk", do, keys[..., :512])
        ds = p * (dp - delta[..., None]) * scale
        dq_ref[begin:end] = torch.einsum("chk,ckd->chd", ds, keys)
        dk = torch.einsum("chk,chd->ckd", ds, qf)
        dk[..., :512] += torch.einsum("chk,chv->ckv", p, do)
        dk = torch.where(valid[..., None], dk, torch.zeros_like(dk))
        dkv_ref.index_add_(0, safe.reshape(-1), dk.reshape(-1, q.shape[2]))
        delta64 = (out[begin:end].double() * dout[begin:end].double()).sum(-1)
        norm64 = torch.logaddexp(lse[begin:end].double(), sink.double()[None, :])
        sink_probability = torch.exp(sink.double()[None, :] - norm64)
        d_sink_ref -= (sink_probability * delta64).sum(0)
    torch.testing.assert_close(dq, dq_ref.to(q.dtype), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dkv, dkv_ref.to(kv.dtype), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(d_sink, d_sink_ref.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "head_dim,sq,skv,topk,has_lengths",
    [
        (512, 8192, 8192, 128, False),  # compact_long
        (512, 4096, 8193, 512, True),  # TMEM with the larger dKV zero schedule
        (576, 4096, 4096, 512, True),  # full two-CTA, prefetch and row interleave
        (576, 8192, 8192, 1024, True),  # long-sequence KV prefetch
        (576, 4096, 4096, 1152, True),  # full-grid row interleave and PDL
        (576, 4096, 4096, 2048, False),  # wide finalize
        (576, 8192, 8192, 128, True),  # zero16
    ],
)
def test_rubin_backward_long_sequence_numerics(head_dim, sq, skv, topk, has_lengths):
    _require_rubin()
    from cudnn import DSA

    torch.manual_seed(107)
    q = torch.randn(sq, 128, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(skv, head_dim, dtype=torch.bfloat16, device="cuda")
    sink = torch.randn(128, dtype=torch.float32, device="cuda")
    indices = torch.randint(0, skv, (sq, topk), dtype=torch.int32, device="cuda")
    indices[:, 1::16] = -1
    indices[:, 2::16] = skv
    lengths = torch.randint(0, topk + 1, (sq,), dtype=torch.int32, device="cuda") if has_lengths else None
    scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)
    dout = torch.randn_like(out)
    plan = DSA.SparseAttentionBackward(q, kv, out, dout, lse, sink, indices, sample_topk_length=lengths, softmax_scale=scale)
    assert plan.check_support()
    assert plan._backend == f"sm107_h128_d{head_dim}"
    plan.compile()
    workspace_bytes = plan.scratch_workspace_bytes()
    storage = torch.full((workspace_bytes + 128,), 0xA5, dtype=torch.uint8, device="cuda")
    dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sink)
    plan.execute(q, kv, out, dout, lse, sink, indices, topk_length=lengths, dq=dq, dkv=dkv, d_sink=d_sink, workspace=storage[:workspace_bytes])
    torch.cuda.synchronize()
    assert (storage[workspace_bytes:] == 0xA5).all()
    _check_gradients(q, kv, sink, indices, out, dout, lse, dq, dkv, d_sink, lengths, scale)
