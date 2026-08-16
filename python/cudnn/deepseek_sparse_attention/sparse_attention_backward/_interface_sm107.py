"""Internal Rubin interface for the fixed-shape DSA backward kernel."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor


def _get_rubin_kernel():
    # Keep Rubin-only DSL features outside every SM90/SM100 import path.
    from .dsa_bwd_sm107 import FlashAttentionDSABackwardSm107

    return FlashAttentionDSABackwardSm107


def flash_attn_bwd_sm107(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    current_stream=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the shape-gated two-CTA Rubin DSA backward kernel."""
    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    head_dim_v = out.shape[2]
    device = q.device

    assert q.dtype == torch.bfloat16
    assert q.dtype == kv.dtype == out.dtype == dout.dtype
    assert q.shape == out.shape == dout.shape == (total_S_q, 128, 512)
    assert kv.shape == (4096, 512)
    assert lse.shape == (total_S_q, 128)
    assert attn_sink.shape == (128,)
    assert topk_idxs.shape[0] == total_S_q
    assert topk_idxs.shape[1] in (512, 1024, 2048)
    assert lse.dtype == torch.float32
    assert attn_sink.dtype == torch.float32
    assert topk_idxs.dtype == torch.int32
    tensors_to_check = [q, kv, out, dout, lse, attn_sink, topk_idxs]
    if topk_length is not None:
        assert topk_length.shape == (total_S_q,)
        assert topk_length.dtype == torch.int32
        tensors_to_check.append(topk_length)
    assert all(tensor.is_cuda and tensor.device == device for tensor in tensors_to_check)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    block_tile = 64
    num_head_blocks = (num_head + block_tile - 1) // block_tile
    batch_size = 1

    q, kv, out, dout = [tensor.contiguous() for tensor in (q, kv, out, dout)]
    lse = lse.contiguous()

    if dq is None:
        dq = torch.empty_like(q)
    else:
        assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
        assert dq.dtype == q.dtype, f"dq dtype mismatch: expected {q.dtype}, got {dq.dtype}"
        assert dq.device == device, f"dq device mismatch: expected {device}, got {dq.device}"

    if dkv is None:
        dkv = torch.zeros(total_S_kv, head_dim, dtype=kv.dtype, device=device)
    else:
        expected_dkv_shape = (total_S_kv, head_dim)
        assert dkv.shape == expected_dkv_shape, f"dkv shape mismatch: expected {expected_dkv_shape}, got {dkv.shape}"
        assert dkv.dtype == kv.dtype, f"dkv dtype mismatch: expected {kv.dtype}, got {dkv.dtype}"
        assert dkv.device == device, f"dkv device mismatch: expected {device}, got {dkv.device}"
        dkv.fill_(0)
    d_sink = torch.zeros_like(attn_sink)

    kernel_class = _get_rubin_kernel()
    acc_dtype = cutlass.Float32
    workspace_LSE_OdO = torch.zeros(
        *kernel_class._get_workspace_size_LSE_OdO(
            total_S_q,
            head_dim,
            num_head,
            batch_size,
            acc_dtype,
        ),
        dtype=torch.uint8,
        device=device,
    )
    workspace_dKV = torch.zeros(
        *kernel_class._get_workspace_size_dKV(
            total_S_kv,
            head_dim,
            batch_size,
            acc_dtype,
        ),
        dtype=torch.uint8,
        device=device,
    )

    problem_shape = (total_S_q, total_S_kv, head_dim, (num_head, batch_size))
    current_stream = resolve_stream(current_stream)
    has_topk_length = topk_length is not None
    max_topk = topk_idxs.shape[1]
    capability = tuple(torch.cuda.get_device_capability(device))
    compile_key = (
        device.index,
        capability,
        cutlass.BFloat16,
        head_dim,
        head_dim_v,
        num_head,
        block_tile,
        max_topk,
        has_topk_length,
    )

    if compile_key not in flash_attn_bwd_sm107.compile_cache:
        q_tensor = to_cute_tensor(q, divisibility=head_dim)
        kv_tensor = to_cute_tensor(kv, divisibility=head_dim)
        out_tensor = to_cute_tensor(out, divisibility=head_dim_v)
        dout_tensor = to_cute_tensor(dout, divisibility=head_dim_v)
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        attn_sink_tensor = to_cute_tensor(attn_sink)
        topk_idxs_tensor = to_cute_tensor(topk_idxs)
        topk_length_tensor = to_cute_tensor(topk_length) if has_topk_length else None
        dq_tensor = to_cute_tensor(dq, divisibility=head_dim)
        dkv_tensor = to_cute_tensor(dkv, divisibility=head_dim)
        d_sink_tensor = to_cute_tensor(d_sink)
        workspace_LSE_OdO_tensor = to_cute_tensor(workspace_LSE_OdO)
        workspace_dKV_tensor = to_cute_tensor(workspace_dKV)

        kernel_obj = kernel_class(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            block_tile=block_tile,
            max_topk=max_topk,
        )

        with torch.cuda.nvtx.range("flash_attn_bwd_sm107_compile"):
            flash_attn_bwd_sm107.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                problem_shape,
                q_tensor,
                kv_tensor,
                out_tensor,
                dout_tensor,
                lse_tensor,
                attn_sink_tensor,
                topk_idxs_tensor,
                topk_length_tensor,
                dq_tensor,
                dkv_tensor,
                d_sink_tensor,
                workspace_LSE_OdO_tensor,
                workspace_dKV_tensor,
                softmax_scale,
                current_stream,
                options=compile_options(device=device, capability=capability),
            )

    with torch.cuda.nvtx.range("flash_attn_bwd_sm107_kernel"):
        flash_attn_bwd_sm107.compile_cache[compile_key](
            problem_shape,
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            topk_length,
            dq,
            dkv,
            d_sink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            current_stream,
        )

    return dq, dkv, d_sink


flash_attn_bwd_sm107.compile_cache = {}
