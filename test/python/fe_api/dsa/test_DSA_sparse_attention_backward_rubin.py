import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_reference import (
    check_ref_dsa_sparse_attention_backward,
    ref_sparse_attention_forward,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=107)
def test_DSA_sparse_attention_backward_sm107_ragged_path():
    try:
        from cuda.bindings import driver as cuda
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
            _interface_sm107,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin SM107 is required")

    device = torch.device("cuda")
    seqlen_q, seqlen_kv = 4, 4096
    num_heads, head_dim, topk = 128, 512, 512
    q = torch.randn(
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(
        seqlen_kv,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(seqlen_kv, device=device)[:topk] for _ in range(seqlen_q)]).to(torch.int32)
    topk_length = torch.tensor(
        [0, 63, 128, 512],
        dtype=torch.int32,
        device=device,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    _interface_sm107.flash_attn_bwd_sm107.compile_cache.clear()
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
        stream=stream,
    )
    torch.cuda.synchronize()

    assert _interface_sm107.flash_attn_bwd_sm107.compile_cache
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
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=108)
def test_DSA_sparse_attention_backward_sm107_short_topk_falls_back():
    try:
        from cuda.bindings import driver as cuda
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
            _interface_sm100,
            _interface_sm107,
            api as _api,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin SM107 is required")

    device = torch.device("cuda")
    seqlen_q, seqlen_kv = 2, 4096
    num_heads, head_dim, topk = 128, 512, 128
    q = torch.randn(
        seqlen_q,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(
        seqlen_kv,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    topk_idxs = torch.stack([torch.randperm(seqlen_kv, device=device)[:topk] for _ in range(seqlen_q)]).to(torch.int32)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    out, lse = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
    )
    dout = torch.randn_like(out)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    _api._cache_of_SparseAttentionBackwardObjects.clear()
    _interface_sm100.flash_attn_bwd_sm100.compile_cache.clear()
    _interface_sm107.flash_attn_bwd_sm107.compile_cache.clear()
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    assert _interface_sm100.flash_attn_bwd_sm100.compile_cache
    assert not _interface_sm107.flash_attn_bwd_sm107.compile_cache
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
        atol=5e-2,
        rtol=5e-2,
    )
