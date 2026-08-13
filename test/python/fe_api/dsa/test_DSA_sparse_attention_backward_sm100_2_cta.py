# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness coverage for the explicit SM100 two-CTA DSA backward path."""

import math

import pytest
import torch

_NUM_HEADS = 128
_HEAD_DIM = 512
_DEFAULT_TOPK = 128
_DEFAULT_SEQLEN_KV = 192


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("SM100 GPU required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 0):
        pytest.skip(f"SM100 GPU required, found SM{major}{minor}")


def _case_shape(case: str) -> tuple[int, int, int]:
    if case == "length_boundaries":
        return 9, _DEFAULT_SEQLEN_KV, _DEFAULT_TOPK
    if case in ("sentinel_holes", "all_empty"):
        return 4, _DEFAULT_SEQLEN_KV, _DEFAULT_TOPK
    if case == "production_topk_2048":
        return 1, 2048, 2048
    return 3, _DEFAULT_SEQLEN_KV, _DEFAULT_TOPK


def _make_topk_inputs(case: str, seqlen_q: int, seqlen_kv: int, topk: int, generator: torch.Generator):
    device = torch.device("cuda")
    topk_idxs = torch.stack([torch.randperm(seqlen_kv, device=device, generator=generator)[:topk] for _ in range(seqlen_q)]).to(torch.int32)
    topk_length = None

    if case == "length_boundaries":
        topk_length = torch.tensor(
            [-7, 0, 1, 63, 64, 65, 127, 128, topk + 17],
            dtype=torch.int32,
            device=device,
        )
    elif case == "sentinel_holes":
        positions = torch.arange(topk, device=device)
        topk_idxs[0, positions % 3 == 0] = -1
        topk_idxs[1, :17] = -1
        topk_idxs[2, 61:68] = -1
        topk_idxs[3, -19:] = -1
    elif case == "all_empty":
        topk_idxs.fill_(torch.iinfo(torch.int32).max)
        topk_length = torch.tensor(
            [0, -1, -64, 0],
            dtype=torch.int32,
            device=device,
        )
    elif case == "production_topk_2048":
        topk_length = torch.full((seqlen_q,), topk, dtype=torch.int32, device=device)

    return topk_idxs, topk_length


def _reference_forward_fp32(q, kv, attn_sink, topk_idxs, topk_length, softmax_scale):
    """Sparse K=V attention with KV-only LSE, expressed only in PyTorch."""
    seqlen_q, _, _ = q.shape
    topk = topk_idxs.shape[1]
    positions = torch.arange(topk, device=q.device).unsqueeze(0)

    if topk_length is None:
        within_length = torch.ones_like(topk_idxs, dtype=torch.bool)
    else:
        lengths = topk_length.to(torch.int64).clamp(min=0, max=topk)
        within_length = positions < lengths.unsqueeze(1)
    active_positive_oob = within_length & (topk_idxs >= kv.shape[0])
    if bool(active_positive_oob.any()):
        raise ValueError("active top-k indices must be smaller than seqlen_kv")
    valid = within_length & (topk_idxs >= 0)

    safe_idxs = topk_idxs.clamp(min=0, max=kv.shape[0] - 1).to(torch.int64)
    selected_kv = kv[safe_idxs]
    scores = torch.einsum("qhd,qkd->qhk", q, selected_kv) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    # logsumexp(all -inf) has an undefined autograd derivative. Empty rows are
    # evaluated through a finite, inactive branch and are restored to -inf.
    nonempty = valid.any(dim=1)
    reduction_scores = torch.where(
        nonempty[:, None, None],
        scores,
        torch.zeros_like(scores),
    )
    safe_lse = torch.logsumexp(reduction_scores, dim=-1)
    kv_lse = torch.where(
        nonempty[:, None],
        safe_lse,
        torch.full_like(safe_lse, float("-inf")),
    )
    lse_with_sink = torch.logaddexp(kv_lse, attn_sink.unsqueeze(0))
    probabilities = torch.exp(scores - lse_with_sink.unsqueeze(-1))
    probabilities = torch.where(valid.unsqueeze(1), probabilities, 0.0)
    out = torch.einsum("qhk,qkd->qhd", probabilities, selected_kv)

    assert out.shape == (seqlen_q, _NUM_HEADS, _HEAD_DIM)
    return out, kv_lse


@pytest.mark.L0
@pytest.mark.parametrize(
    "case",
    ["dense", "length_boundaries", "sentinel_holes", "all_empty", "production_topk_2048"],
)
def test_DSA_sparse_attention_backward_sm100_2_cta(case):
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100
        from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2_cta import FlashAttentionDSABackwardSm100TwoCTA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm100()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1200 + len(case))
    seqlen_q, seqlen_kv, topk = _case_shape(case)
    softmax_scale = 1.0 / math.sqrt(_HEAD_DIM)

    q = torch.randn(
        seqlen_q,
        _NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv = torch.randn(
        seqlen_kv,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    attn_sink = torch.linspace(-1.5, 1.5, _NUM_HEADS, dtype=torch.float32, device=device)
    topk_idxs, topk_length = _make_topk_inputs(case, seqlen_q, seqlen_kv, topk, generator)

    selected_implementation = _interface_sm100._resolve_backward_implementation("sm100_2_cta", q, _HEAD_DIM, topk)
    assert selected_implementation is FlashAttentionDSABackwardSm100TwoCTA

    q_ref = q.float().detach().requires_grad_(True)
    kv_ref = kv.float().detach().requires_grad_(True)
    sink_ref = attn_sink.detach().requires_grad_(True)
    out_ref, lse = _reference_forward_fp32(
        q_ref,
        kv_ref,
        sink_ref,
        topk_idxs,
        topk_length,
        softmax_scale,
    )
    out = out_ref.detach().to(torch.bfloat16)
    dout = torch.randn(
        out.shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    dq_ref, dkv_ref, d_sink_ref = torch.autograd.grad(
        out_ref,
        (q_ref, kv_ref, sink_ref),
        grad_outputs=dout.float(),
    )
    if case != "all_empty":
        assert d_sink_ref.abs().max() > 1e-4

    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse.detach(),
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        implementation="sm100_2_cta",
    )
    torch.cuda.synchronize()

    assert any(
        key[0] == "sm100_2_cta" and key[6] == topk and key[7] == (topk_length is not None) for key in _interface_sm100.flash_attn_bwd_sm100.compile_cache
    ), "the public wrapper did not compile the requested sm100_2_cta implementation"

    dq = result["dq"].float()
    dkv = result["dkv"].float()
    d_sink = result["d_sink"].float()
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dkv).all()
    assert torch.isfinite(d_sink).all()

    torch.testing.assert_close(dq, dq_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dkv, dkv_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(d_sink, d_sink_ref, atol=5e-2, rtol=5e-2)

    if topk_length is not None:
        empty_rows = topk_length <= 0
        assert torch.equal(dq[empty_rows], torch.zeros_like(dq[empty_rows]))
    if case == "all_empty":
        assert torch.equal(dkv, torch.zeros_like(dkv))
        assert torch.equal(d_sink, torch.zeros_like(d_sink))
