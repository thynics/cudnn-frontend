# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time compilation and caller-owned storage for Rubin DSA backward."""

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.frost.buffers import cutedsl_requirement_error

from ._interface_sm100 import _carve_workspace_sm100, _workspace_shapes_sm100
from ._interface_sm100_d576 import _split_count

BACKENDS = ("sm107_h128_d512", "sm107_h128_d576")
_TOPK_WIDTHS = (128, 512, 1024, 1152, 2048)


def _select_sm107_backend(num_heads, head_dim, *, head_dim_v, dtype, max_topk, device_capability, deterministic=False, is_contiguous=True):
    """Select only the contiguous BF16 H128 envelope supported by the kernels."""
    if (
        device_capability == (10, 7)
        and dtype == torch.bfloat16
        and num_heads == 128
        and head_dim in (512, 576)
        and head_dim_v == 512
        and max_topk in _TOPK_WIDTHS
        and is_contiguous
        and not deterministic
    ):
        return f"sm107_h128_d{head_dim}"
    return None


@dataclass(frozen=True)
class _KernelConfig:
    """Finite schedule choices; exact sequence extents remain runtime inputs."""

    variant: str
    wide_zero: bool = False
    short_sq: bool = False
    single_query: bool = False
    split_regime: bool = False
    enable_pdl: bool = False
    enable_kv_prefetch: bool = False
    enable_wide_k2048_finalize: bool = False
    enable_zero16_k128: bool = False
    enable_score_row_interleave: bool = False


def _select_config(head_dim, sq, skv, max_topk, has_topk_length, sm_count):
    """Choose a schedule and split count from declared tensor metadata."""
    if head_dim == 512:
        variant = "d512_tmem" if max_topk >= 512 else "d512_compact_long" if sq >= 8192 else "d512_compact"
        return _KernelConfig(variant, wide_zero=skv <= 8192, short_sq=sq < 8192), 1

    # Only short queries with explicit lengths use split clusters. Dense rows
    # use a single cluster before BF16 dKV conversion.
    split = _split_count(sq, max_topk, sm_count) if has_topk_length and sq < 4096 else 1
    return (
        _KernelConfig(
            "d576" if sq >= 4096 else "d576_small",
            short_sq=sq < 4096,
            single_query=sq == 1,
            split_regime=split > 1,
            enable_pdl=max_topk == 128 or (max_topk in (512, 1024, 2048) and sq >= 4096),
            enable_kv_prefetch=max_topk == 512 or (max_topk == 1024 and sq >= 8192),
            enable_wide_k2048_finalize=max_topk == 2048 and sq >= 4096,
            enable_zero16_k128=max_topk == 128 and sq >= 8192 and skv >= 8192,
            enable_score_row_interleave=max_topk in (512, 1024, 2048) and sq >= 4096 and skv >= 4096,
        ),
        split,
    )


def _get_kernel_class(variant):
    """Import the shared implementation after checking the DSL version."""
    error = cutedsl_requirement_error("Rubin DSA backward")
    if error:
        raise RuntimeError(error)
    if variant in ("d512_compact", "d512_compact_long", "d512_tmem"):
        from .dsa_bwd_sm107_h128_d512_2cta import FlashAttentionDSABackwardSm107TwoIssuerDq4 as Kernel
    elif variant in ("d576", "d576_small"):
        from .dsa_bwd_sm107_h128_d576_2cta import FlashAttentionDSABackwardSm107H128D576TwoCTA as Kernel
    else:
        raise ValueError(f"Unknown Rubin DSA backward variant: {variant}")
    return Kernel


@lru_cache(maxsize=None)
def _compile_sm107(head_dim, max_topk, has_topk_length, config):
    """Compile the complete launch sequence from descriptors, before execute()."""
    kernel_class = _get_kernel_class(config.variant)
    sq, skv = cute.sym_int(), cute.sym_int()

    def tensor(dtype, shape, alignment=16):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    q = tensor(cutlass.BFloat16, (sq, 128, head_dim))
    kv = tensor(cutlass.BFloat16, (skv, head_dim))
    out = tensor(cutlass.BFloat16, (sq, 128, 512))
    lse = tensor(cutlass.Float32, (sq, 128), 4 if head_dim == 512 else 8)
    sink = tensor(cutlass.Float32, (128,), 4 if head_dim == 512 else 16)
    indices = tensor(cutlass.Int32, (sq, max_topk), 4 if head_dim == 512 else 16)
    lengths = tensor(cutlass.Int32, (sq,), 4) if has_topk_length else None
    stats_workspace = tensor(cutlass.Uint8, (cute.sym_int(),))
    dkv_workspace = tensor(cutlass.Uint8, (cute.sym_int(),))
    kwargs = dict(element_dtype=cutlass.BFloat16, head_dim=head_dim, head_dim_v=512, block_tile=64, max_topk=max_topk)
    if head_dim == 512:
        kwargs.update(wide_zero=config.wide_zero, short_sq=config.short_sq)
    else:
        kwargs.update(
            short_sq=config.short_sq,
            single_query=config.single_query,
            enable_pdl=config.enable_pdl,
            enable_kv_prefetch=config.enable_kv_prefetch,
            enable_wide_k2048_finalize=config.enable_wide_k2048_finalize,
            enable_zero16_k128=config.enable_zero16_k128,
            enable_score_row_interleave=config.enable_score_row_interleave,
        )
    args = [
        (cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(head_dim), (cutlass.Int32(128), cutlass.Int32(1))),
        q,
        kv,
        out,
        out,
        lse,
        sink,
        indices,
        lengths,
        q,
        kv,
        sink,
        stats_workspace,
        dkv_workspace,
        cutlass.Float32(1.0),
    ]
    if head_dim == 576:
        args.append(cutlass.Int32(1))
    args.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False))
    if head_dim == 576:
        args.append(config.split_regime)
    # Use the plan's architecture even if a Blackwell plan has already populated
    # the ambient-device compiler cache in this process.
    return cute.compile(kernel_class(**kwargs), *args, options=compile_options(device_capability=(10, 7)))


def _execute_sm107(
    compiled_kernel,
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
    workspace,
    softmax_scale,
    split_count,
    current_stream,
):
    """Validate preallocated storage and launch without compilation or copies."""
    sq, skv, head_dim = q.shape[0], kv.shape[0], q.shape[2]
    specifications = (
        ("q", q, (sq, 128, head_dim), torch.bfloat16, 16),
        ("kv", kv, (skv, head_dim), torch.bfloat16, 16),
        ("out", out, (sq, 128, 512), torch.bfloat16, 16),
        ("dout", dout, (sq, 128, 512), torch.bfloat16, 16),
        ("lse", lse, (sq, 128), torch.float32, 4 if head_dim == 512 else 8),
        ("attn_sink", attn_sink, (128,), torch.float32, 4 if head_dim == 512 else 16),
        ("topk_idxs", topk_idxs, (sq, topk_idxs.shape[1]), torch.int32, 4 if head_dim == 512 else 16),
        ("dq", dq, (sq, 128, head_dim), torch.bfloat16, 16),
        ("dkv", dkv, (skv, head_dim), torch.bfloat16, 16),
        ("d_sink", d_sink, (128,), torch.float32, 4 if head_dim == 512 else 16),
    )
    if topk_length is not None:
        specifications += (("topk_length", topk_length, (sq,), torch.int32, 4),)
    for name, value, shape, dtype, alignment in specifications:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
        if tuple(value.shape) != shape or value.dtype != dtype or value.device != q.device:
            raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {q.device}")
        if not value.is_contiguous() or value.data_ptr() % alignment:
            raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")

    stats_shape, dkv_shape = _workspace_shapes_sm100(sq, skv, head_dim, 128, False)
    stats_workspace, dkv_workspace = _carve_workspace_sm100(workspace, q.device, stats_shape, dkv_shape)
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    if not isinstance(scale, (float, int)):
        raise TypeError("softmax_scale must be a Python scalar")
    args = [
        (sq, skv, head_dim, (128, 1)),
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
        stats_workspace.view(-1),
        dkv_workspace.view(-1),
        scale,
    ]
    if head_dim == 576:
        args.append(split_count)
    with torch.cuda.device(q.device):
        args.append(resolve_stream(current_stream))
        with torch.cuda.nvtx.range(f"flash_attn_bwd_sm107_kernel[h128_d{head_dim}]"):
            compiled_kernel(*args)
    return dq, dkv, d_sink
