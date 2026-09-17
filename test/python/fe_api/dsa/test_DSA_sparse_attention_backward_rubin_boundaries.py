# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empty, saturating and padded-row limits of the Rubin backward schedules."""

import math

import pytest
import torch

from fe_api.dsa.test_DSA_sparse_attention_backward_rubin import _modules, _require_rubin

_SCHEDULES = [
    pytest.param(512, 1, 128, "d512_compact", marks=pytest.mark.L0),
    pytest.param(512, 9, 512, "d512_tmem", marks=pytest.mark.L0),
    pytest.param(576, 1, 128, "d576_small", marks=pytest.mark.L0),
    pytest.param(576, 9, 512, "d576_small", marks=pytest.mark.L0),
    pytest.param(512, 8192, 128, "d512_compact_long", marks=pytest.mark.L1),
    pytest.param(576, 4096, 128, "d576", marks=pytest.mark.L1),
    pytest.param(576, 4096, 512, "d576", marks=pytest.mark.L1),
]


def _run_rubin(monkeypatch, variant, q, kv, out, dout, lse, sink, indices, lengths, scale):
    api, iface = _modules()
    calls = []
    execute = iface._execute_sm107

    def launch(*args, **kwargs):
        calls.append(args[0])
        return execute(*args, **kwargs)

    api._cache_of_SparseAttentionBackwardObjects.clear()
    monkeypatch.setattr(iface, "_execute_sm107", launch)
    result = api.sparse_attention_backward_wrapper(q, kv, out, dout, lse, sink, indices, topk_length=lengths, softmax_scale=scale)
    torch.cuda.synchronize()
    plan = next(iter(api._cache_of_SparseAttentionBackwardObjects.values()))
    assert plan._backend == f"sm107_h128_d{q.shape[2]}"
    assert plan._rubin_config.variant == variant
    assert calls == [plan._compiled_kernel]
    assert tuple(result.keys()) == ("dq", "dkv", "d_sink")
    return result


@pytest.mark.parametrize("head_dim,sq,topk,variant", _SCHEDULES)
@pytest.mark.parametrize("has_lengths", [False, True])
@pytest.mark.parametrize(
    "sink_value,empty",
    [(2e38, False), (2.4e38, False), (math.inf, False), (-math.inf, True), pytest.param(3e38, False, id="rescale-overflow")],
)
def test_rubin_backward_sink_limits(monkeypatch, head_dim, sq, topk, variant, has_lengths, sink_value, empty):
    _require_rubin()
    torch.manual_seed(786)
    q = torch.randn(sq, 128, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(1, head_dim, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(sq, 128, 512, dtype=torch.bfloat16, device="cuda")
    dout = torch.randn_like(out)
    sink = torch.full((128,), sink_value, dtype=torch.float32, device="cuda")
    indices = torch.full((sq, topk), -1, dtype=torch.int32, device="cuda")
    scale = 192**-0.5
    if sink_value == 3e38:
        # Both statistics are finite, but both overflow in log2 units.
        q.zero_()
        kv.zero_()
        q[:, :, 0] = 1e19
        kv[:, 0] = 2.5e19
        scale = 1.0
    if empty:
        lse = torch.full((sq, 128), -math.inf, dtype=torch.float32, device="cuda")
    else:
        indices[:, 0] = 0
        lse = ((q.float() * kv[0].float()).sum(-1) * scale).contiguous()
    lengths = torch.full((sq,), 0 if empty else 1, dtype=torch.int32, device="cuda") if has_lengths else None
    result = _run_rubin(monkeypatch, variant, q, kv, out, dout, lse, sink, indices, lengths, scale)
    for name, value in result.items():
        assert torch.equal(value, torch.zeros_like(value)), f"{name} must be zero for sink={sink_value}, empty={empty}"


@pytest.mark.parametrize("head_dim,sq,topk,variant", _SCHEDULES)
@pytest.mark.parametrize("has_lengths", [False, True])
@pytest.mark.parametrize("dout_value", [1.0, float(2**70)])
def test_rubin_backward_padded_slots_with_negative_lse(monkeypatch, head_dim, sq, topk, variant, has_lengths, dout_value):
    _require_rubin()
    torch.manual_seed(676)
    q = torch.full((sq, 128, head_dim), 10.0, dtype=torch.bfloat16, device="cuda")
    kv = -torch.ones(2, head_dim, dtype=torch.bfloat16, device="cuda")
    out = kv[0, :512].view(1, 1, 512).expand(sq, 128, 512).contiguous()
    # Exact O*dO cancellation makes the single valid key's dQ/dK zero.
    dout = torch.full_like(out, dout_value)
    sink = torch.full((128,), -math.inf, dtype=torch.float32, device="cuda")
    indices = torch.full((sq, topk), -1, dtype=torch.int32, device="cuda")
    indices[:, 0] = 0
    lengths = torch.ones(sq, dtype=torch.int32, device="cuda") if has_lengths else None
    scale = 1.0
    lse = torch.full((sq, 128), -10.0 * head_dim, dtype=torch.float32, device="cuda")
    result = _run_rubin(monkeypatch, variant, q, kv, out, dout, lse, sink, indices, lengths, scale)
    assert torch.equal(result["dq"], torch.zeros_like(q))
    assert torch.equal(result["d_sink"], torch.zeros_like(sink))
    expected_dkv = torch.zeros_like(kv)
    expected_dkv[0, :512] = sq * 128 * dout_value
    torch.testing.assert_close(result["dkv"], expected_dkv, atol=0, rtol=0)


@pytest.mark.parametrize("head_dim,sq,topk,variant", _SCHEDULES)
@pytest.mark.parametrize("has_lengths", [False, True])
@pytest.mark.parametrize("logit,sink_value,probability", [(3e38, 2.5e38, 1.0), (-3e38, -math.inf, 1.0), (3e38, None, 0.5)])
def test_rubin_backward_finite_logit_limits(monkeypatch, head_dim, sq, topk, variant, has_lengths, logit, sink_value, probability):
    _require_rubin()
    q = torch.zeros(sq, 128, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(1, head_dim, dtype=torch.bfloat16, device="cuda")
    q[:, :, 0] = logit
    kv[0, 0] = 1.0
    lse = q[:, :, 0].float().contiguous()
    sink = lse[0].clone() if sink_value is None else torch.full((128,), sink_value, dtype=torch.float32, device="cuda")
    out = torch.zeros(sq, 128, 512, dtype=torch.bfloat16, device="cuda")
    out[:, :, 0] = probability
    dout = torch.zeros_like(out)
    # Orthogonal dO keeps dS and dSink zero while measuring the KV probability.
    dout[:, :, 1] = 1.0
    indices = torch.full((sq, topk), -1, dtype=torch.int32, device="cuda")
    indices[:, 0] = 0
    lengths = torch.ones(sq, dtype=torch.int32, device="cuda") if has_lengths else None
    result = _run_rubin(monkeypatch, variant, q, kv, out, dout, lse, sink, indices, lengths, 1.0)
    assert torch.equal(result["dq"], torch.zeros_like(q))
    assert torch.equal(result["d_sink"], torch.zeros_like(sink))
    expected_dkv = torch.zeros_like(kv)
    expected_dkv[0, 1] = sq * 128 * probability
    torch.testing.assert_close(result["dkv"], expected_dkv, atol=0, rtol=0)
