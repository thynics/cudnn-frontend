# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin DSA backward routing, plan lifecycle, and numerical contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0


def _modules():
    pytest.importorskip("cutlass")
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import api, _interface_sm107

    return api, _interface_sm107


def _metadata_plan(monkeypatch, head_dim=512, sq=9, skv=529, topk=128, heads=128, dtype=torch.bfloat16, capability=(10, 7), lengths=True, **kwargs):
    """Build an actual API plan without allocating CUDA storage."""
    api, _ = _modules()
    from cudnn.api_base import TensorDesc

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(multi_processor_count=144))
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())

    def desc(shape, dtype):
        return TensorDesc(dtype, shape, TensorDesc._compute_contiguous_stride(shape), tuple(reversed(range(len(shape)))), torch.device("cuda:0"))

    return api.SparseAttentionBackward(
        desc((sq, heads, head_dim), dtype),
        desc((skv, head_dim), dtype),
        desc((sq, heads, 512), dtype),
        desc((sq, heads, 512), dtype),
        desc((sq, heads), torch.float32),
        desc((heads,), torch.float32),
        desc((sq, topk), torch.int32),
        sample_topk_length=desc((sq,), torch.int32) if lengths else None,
        **kwargs,
    )


@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize("topk", [128, 512, 1024, 1152, 2048])
def test_rubin_plan_routes_supported_shapes(monkeypatch, head_dim, topk):
    _, iface = _modules()
    monkeypatch.setattr(iface, "cutedsl_requirement_error", lambda _: None)
    plan = _metadata_plan(monkeypatch, head_dim=head_dim, topk=topk)
    assert plan.check_support()
    assert plan._backend == f"sm107_h128_d{head_dim}"
    assert plan.scratch_workspace_bytes() == 16 * 128 * 8 + 536 * head_dim * 4


@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"capability": (10, 0)}, "blackwell"),
        ({"capability": (10, 3)}, "blackwell"),
        ({"capability": (9, 0)}, None),
        ({"dtype": torch.float16}, "generic_m64"),
        ({"heads": 64}, "generic_m64"),
        ({"topk": 129}, "generic_m64"),
        ({"deterministic": True}, "generic_m64"),
    ],
)
def test_rubin_plan_preserves_other_routes(monkeypatch, head_dim, overrides, expected):
    plan = _metadata_plan(monkeypatch, head_dim=head_dim, **overrides)
    assert plan.check_support()
    if expected == "blackwell":
        expected = "h128_2cta_m64" if head_dim == 512 else "h128_d576_2cta_m64"
    assert plan._backend == expected


@pytest.mark.parametrize("head_dim", [512, 576])
def test_rubin_plan_does_not_route_strided_inputs(monkeypatch, head_dim):
    from dataclasses import replace

    plan = _metadata_plan(monkeypatch, head_dim=head_dim)
    plan.q_desc = replace(plan.q_desc, stride=(256 * head_dim, head_dim, 1))
    assert plan.check_support()
    assert plan._backend == "generic_m64"


def test_rubin_version_gate_precedes_kernel_import(monkeypatch):
    _, iface = _modules()
    error = "Rubin DSA backward requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"
    monkeypatch.setattr(iface, "cutedsl_requirement_error", lambda _: error)
    plan = _metadata_plan(monkeypatch)
    with pytest.raises(RuntimeError, match="found 4.6.2"):
        plan.check_support()
    with pytest.raises(RuntimeError, match="found 4.6.2"):
        iface._get_kernel_class("d512_tmem")


@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize("invalid", ["dq", "dkv", "d_sink", "q_stride", "length_dtype", "workspace_size", "workspace_alignment", "scale"])
def test_rubin_execute_rejects_invalid_storage(head_dim, invalid):
    """Storage validation must reject bad arguments before reaching a launch.

    CPU storage suffices here because every tested call fails before CUDA.
    """
    _, iface = _modules()
    q = torch.empty((1, 128, head_dim), dtype=torch.bfloat16)
    kv = torch.empty((1, head_dim), dtype=torch.bfloat16)
    out = torch.empty((1, 128, 512), dtype=torch.bfloat16)
    sink = torch.empty(128, dtype=torch.float32)
    required = 8 * 128 * 8 + 8 * head_dim * 4
    args = dict(
        q=q,
        kv=kv,
        out=out,
        dout=out,
        lse=torch.empty((1, 128), dtype=torch.float32),
        attn_sink=sink,
        topk_idxs=torch.empty((1, 128), dtype=torch.int32),
        topk_length=torch.empty(1, dtype=torch.int32),
        dq=torch.empty_like(q),
        dkv=torch.empty_like(kv),
        d_sink=torch.empty_like(sink),
        workspace=torch.empty(required, dtype=torch.uint8),
        softmax_scale=0.1,
        split_count=1,
        current_stream=None,
    )
    if invalid in ("dq", "d_sink"):
        args[invalid] = None
    elif invalid == "dkv":
        args["dkv"] = torch.empty_like(kv, dtype=torch.float32)
    elif invalid == "q_stride":
        args["q"] = torch.empty((1, 256, head_dim), dtype=torch.bfloat16)[:, ::2]
    elif invalid == "length_dtype":
        args["topk_length"] = torch.empty(1, dtype=torch.int64)
    elif invalid == "workspace_size":
        args["workspace"] = args["workspace"][:-1]
    elif invalid == "workspace_alignment":
        args["workspace"] = torch.empty(required + 1, dtype=torch.uint8)[1:]
    else:
        args["softmax_scale"] = torch.tensor(0.1)

    def unexpected_launch(*args):
        pytest.fail("invalid storage reached the kernel")

    error = TypeError if invalid == "scale" else ValueError
    match = {
        "q_stride": "q must be contiguous",
        "length_dtype": "topk_length",
        "workspace_size": "workspace",
        "workspace_alignment": "aligned",
        "scale": "Python scalar",
    }.get(invalid, invalid)
    with pytest.raises(error, match=match):
        iface._execute_sm107(unexpected_launch, **args)


@pytest.mark.parametrize(
    "head_dim,sq,topk,expected_variant,expected_flags",
    [
        (512, 4096, 128, "d512_compact", {"short_sq": True, "wide_zero": True}),
        (512, 8192, 128, "d512_compact_long", {"short_sq": False, "wide_zero": True}),
        (512, 4096, 512, "d512_tmem", {"short_sq": True}),
        (576, 9, 128, "d576_small", {"short_sq": True, "enable_pdl": True, "split_regime": True}),
        (576, 4096, 512, "d576", {"short_sq": False, "enable_pdl": True, "enable_kv_prefetch": True, "enable_score_row_interleave": True}),
        (576, 8192, 2048, "d576", {"enable_wide_k2048_finalize": True}),
    ],
)
def test_rubin_plan_compiles_selected_schedule_and_reuses_artifact(monkeypatch, head_dim, sq, topk, expected_variant, expected_flags):
    """Exact extents and softmax scale must not cause another compilation."""
    _, iface = _modules()
    monkeypatch.setattr(iface, "cutedsl_requirement_error", lambda _: None)
    calls = []

    def kernel_class(variant):
        assert variant == expected_variant
        return lambda **kwargs: kwargs

    def compile_kernel(kernel, *args, **kwargs):
        assert kwargs["options"] == "--enable-tvm-ffi --gpu-arch sm_107a"
        for flag, value in expected_flags.items():
            if flag == "split_regime":
                assert args[-1] == value
            else:
                assert kernel[flag] == value
        calls.append(args)
        return object()

    monkeypatch.setattr(iface, "_get_kernel_class", kernel_class)
    monkeypatch.setattr(iface.cute, "compile", compile_kernel)
    iface._compile_sm107.cache_clear()
    try:
        plans = [_metadata_plan(monkeypatch, head_dim=head_dim, sq=sq + delta, skv=8192 - delta, topk=topk, softmax_scale=0.1 + delta) for delta in (0, 1)]
        for plan in plans:
            assert plan.check_support()
            plan.compile()
        assert len(calls) == 1
        assert plans[0]._compiled_kernel is plans[1]._compiled_kernel
    finally:
        iface._compile_sm107.cache_clear()


def _require_rubin():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin SM107 GPU required")
    _, iface = _modules()
    error = iface.cutedsl_requirement_error("Rubin DSA backward")
    if error:
        pytest.skip(error)


@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize("sq,topk", [(1, 128), (9, 512), (9, 1024), (9, 1152), (9, 2048)])
@pytest.mark.parametrize("has_lengths", [False, True])
def test_rubin_backward_numerics_and_workspace_reuse(monkeypatch, head_dim, sq, topk, has_lengths):
    """Exercise output ownership, masking, explicit streams, and graph replay."""
    _require_rubin()
    from cuda.bindings import driver as cuda
    from fe_api.dsa.dsa_reference import check_ref_dsa_sparse_attention_backward, ref_sparse_attention_forward_chunked

    api, iface = _modules()
    launches = []
    execute = iface._execute_sm107

    def record_launch(compiled_kernel, *args, **kwargs):
        launches.append(compiled_kernel)
        return execute(compiled_kernel, *args, **kwargs)

    monkeypatch.setattr(iface, "_execute_sm107", record_launch)
    torch.manual_seed(107)
    skv = topk + 17
    q = torch.randn(sq, 128, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(skv, head_dim, dtype=torch.bfloat16, device="cuda")
    sink = torch.full((128,), 3.0, dtype=torch.float32, device="cuda")
    indices = torch.randint(0, skv - 1, (sq, topk), dtype=torch.int32, device="cuda")
    indices[:, 1::8] = -1
    indices[:, 2::8] = skv + 3
    length_values = [65] if sq == 1 else [-3, 0, 1, 63, 64, 65, topk - 1, topk, topk + 1]
    lengths = torch.tensor(length_values, dtype=torch.int32, device="cuda") if has_lengths else None
    if lengths is not None:
        indices[torch.arange(topk, device="cuda")[None, :] >= lengths[:, None]] = skv - 1
    scale = head_dim**-0.5
    out, lse = ref_sparse_attention_forward_chunked(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)
    dout = torch.randn_like(out)
    poisoned_kv = kv.clone()
    poisoned_kv[-1] = float("nan")
    plan = api.SparseAttentionBackward(q, poisoned_kv, out, dout, lse, sink, indices, sample_topk_length=lengths, softmax_scale=scale)
    assert plan.check_support()
    assert plan._backend == f"sm107_h128_d{head_dim}"
    plan.compile()
    workspace_bytes = plan.scratch_workspace_bytes()
    storage = torch.empty(workspace_bytes + 128, dtype=torch.uint8, device="cuda")
    workspace = storage[:workspace_bytes]
    dq, dkv, d_sink = torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sink)
    side_stream = torch.cuda.Stream()

    def launch():
        return plan.execute(
            q,
            poisoned_kv,
            out,
            dout,
            lse,
            sink,
            indices,
            topk_length=lengths,
            dq=dq,
            dkv=dkv,
            d_sink=d_sink,
            workspace=workspace,
            current_stream=cuda.CUstream(side_stream.cuda_stream),
        )

    def forbidden(*args, **kwargs):
        pytest.fail("execute() must not compile, allocate, or copy inputs")

    for _ in range(2):
        for tensor in (dq, dkv, d_sink):
            tensor.fill_(float("nan"))
        storage.fill_(0xA5)
        side_stream.wait_stream(torch.cuda.current_stream())
        with monkeypatch.context() as patch:
            patch.setattr(iface.cute, "compile", forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like"):
                patch.setattr(torch, name, forbidden)
            torch.cuda.set_sync_debug_mode("error")
            try:
                actual = launch()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        side_stream.synchronize()
        assert all(a is b for a, b in zip(actual, (dq, dkv, d_sink), strict=True))
        assert launches[-1] is plan._compiled_kernel
        assert (storage[workspace_bytes:] == 0xA5).all()
        check_ref_dsa_sparse_attention_backward(
            q, kv, sink, indices, out, dout, lse, dq, dkv, d_sink, topk_length=lengths, softmax_scale=scale, atol=5e-2, rtol=5e-2
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    check_ref_dsa_sparse_attention_backward(
        q, kv, sink, indices, out, dout, lse, dq, dkv, d_sink, topk_length=lengths, softmax_scale=scale, atol=5e-2, rtol=5e-2
    )

    launch_count = len(launches)
    result = api.sparse_attention_backward_wrapper(q, poisoned_kv, out, dout, lse, sink, indices, topk_length=lengths, softmax_scale=scale, workspace=workspace)
    assert len(launches) == launch_count + 1
    assert launches[-1] is plan._compiled_kernel
    check_ref_dsa_sparse_attention_backward(
        q, kv, sink, indices, out, dout, lse, result["dq"], result["dkv"], result["d_sink"], topk_length=lengths, softmax_scale=scale, atol=5e-2, rtol=5e-2
    )
