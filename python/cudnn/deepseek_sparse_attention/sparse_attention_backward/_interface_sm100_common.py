# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared workspace contracts for the SM100 DSA backward interfaces."""

import math
from typing import Optional, Tuple

import cutlass

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100
from .dsa_bwd_sm100_deterministic import DETERMINISTIC_DKV_MODES, FlashAttentionDSABackwardSm100Deterministic, deterministic_kernel_class

_BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))
_WORKSPACE_ALIGNMENT = 128
_INT32_ADDRESSABLE_ELEMENTS = 1 << 31


# Backends on which a ``deterministic_dkv_mode="chain"`` plan can run (``None``: backend not yet selected):
# the generic M64 kernel, and the H64 + H32 composition that deterministic H96 D576 keeps where neither
# conflict-rank layout applies to the shape.
_CHAIN_MODE_BACKENDS = ("generic_m64", "h96_h64_h32")


def deterministic_dkv_mode_error(dkv_mode, deterministic: bool, backend: Optional[str] = None) -> Optional[str]:
    """Return why a plan cannot honor ``deterministic_dkv_mode``, or ``None``.

    The option selects the dKV storage of deterministic SM100 generic M64 plans. Unknown values are
    rejected, every value but the default needs ``deterministic=True``, and ``'chain'`` is rejected on a
    backend that never runs the generic M64 kernel (the SM90 and two-CTA backends) rather than ignored.
    """
    if not isinstance(dkv_mode, str) or dkv_mode not in DETERMINISTIC_DKV_MODES:
        return f"deterministic_dkv_mode must be one of {DETERMINISTIC_DKV_MODES}, got {dkv_mode!r}"
    if dkv_mode != "records" and not deterministic:
        return f"deterministic_dkv_mode={dkv_mode!r} requires deterministic=True"
    if dkv_mode == "chain" and backend is not None and backend not in _CHAIN_MODE_BACKENDS:
        return "deterministic_dkv_mode='chain' is supported only by the deterministic SM100 generic_m64 backend"
    return None


def _align_workspace_bytes(num_bytes: int) -> int:
    """Round a workspace segment size up to the shared alignment boundary."""
    return -(-int(num_bytes) // _WORKSPACE_ALIGNMENT) * _WORKSPACE_ALIGNMENT


def _workspace_shapes_sm100(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool,
    backend: Optional[str] = None,
    deterministic_relay_cluster_size: Optional[int] = None,
    max_topk: Optional[int] = None,
    dkv_mode: str = "records",
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return byte-shaped LSE/OdO and dKV scratch layouts for one launch.

    Generic deterministic M64 launches use the adaptive shard specialization,
    including the row-aware cluster relay for large KV extents. H96/H128 D576
    two-CTA kernels use their compact 96-plane workspace. An unspecified
    backend retains the conservative 128-plane capacity. ``dkv_mode`` is the
    plan's ``deterministic_dkv_mode`` (see ``deterministic_dkv_mode_error``).
    """
    mode_error = deterministic_dkv_mode_error(dkv_mode, deterministic, backend)
    if mode_error is not None:
        raise ValueError(mode_error)
    if deterministic_relay_cluster_size is not None:
        if not deterministic:
            raise ValueError("deterministic_relay_cluster_size requires deterministic=True")
        if backend != "generic_m64":
            raise ValueError("deterministic_relay_cluster_size is supported only by the deterministic generic_m64 backend")
    acc_dtype = cutlass.Float32
    workspace_lse_odo_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
        total_s_q,
        head_dim,
        num_heads,
        1,
        acc_dtype,
    )
    if deterministic:
        if backend == "generic_m64":
            dkv_kernel_cls = deterministic_kernel_class(
                total_s_q,
                total_s_kv,
                head_dim,
                num_heads,
                deterministic_relay_cluster_size,
                max_topk=max_topk,
                dkv_mode=dkv_mode,
            )
        elif backend in {"h96_d576_2cta_m64", "h128_d576_2cta_m64"}:
            from .dsa_bwd_sm100_h128_d576_2cta_deterministic import TwoCTADeterministicReduction

            dkv_kernel_cls = TwoCTADeterministicReduction
        else:
            dkv_kernel_cls = FlashAttentionDSABackwardSm100Deterministic
    else:
        dkv_kernel_cls = FlashAttentionDSABackwardSm100
    if hasattr(dkv_kernel_cls, "_get_workspace_size_dKV_for_rank"):
        # Conflict-rank plans (generic M64 only) size their waves from the plan (``rank_m64_wave``).
        workspace_dkv_shape = dkv_kernel_cls._get_workspace_size_dKV_for_rank(total_s_kv, head_dim, max_topk, 1, acc_dtype, num_heads, total_s_q)
    elif hasattr(dkv_kernel_cls, "_get_workspace_size_dKV_for_q"):
        workspace_dkv_shape = dkv_kernel_cls._get_workspace_size_dKV_for_q(
            total_s_q,
            total_s_kv,
            head_dim,
            num_heads,
            1,
            acc_dtype,
        )
    else:
        workspace_dkv_shape = dkv_kernel_cls._get_workspace_size_dKV(
            total_s_kv,
            head_dim,
            1,
            acc_dtype,
        )
    return workspace_lse_odo_shape, workspace_dkv_shape


def _requires_wide_data_addressing(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    head_dim_v: int,
    num_heads: int,
    max_topk: int,
    deterministic: bool,
    backend: Optional[str],
    *,
    include_workspace: bool,
    deterministic_relay_cluster_size: Optional[int] = None,
    dkv_mode: str = "records",
) -> bool:
    """Return whether a compact plan tensor can exceed signed Int32 offsets.

    All public SM100 DSA backward tensors are contiguous. Their largest
    element offset is therefore ``numel - 1``. The generic compiler also
    describes byte-shaped workspaces with compact layouts, while the D576
    compiler already gives those workspace extents an Int64 ABI independently.
    """
    spans = [
        total_s_q * num_heads * head_dim,
        total_s_q * num_heads * head_dim_v,
        total_s_q * num_heads,
        total_s_q * max_topk,
        total_s_kv * head_dim,
    ]
    if include_workspace:
        spans.extend(
            math.prod(shape)
            for shape in _workspace_shapes_sm100(
                total_s_q,
                total_s_kv,
                head_dim,
                num_heads,
                deterministic,
                backend,
                deterministic_relay_cluster_size,
                max_topk=max_topk,
                dkv_mode=dkv_mode,
            )
        )
    return any(span > _INT32_ADDRESSABLE_ELEMENTS for span in spans)


def flash_attn_bwd_sm100_workspace_size(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool = False,
    *,
    backend: Optional[str] = None,
    deterministic_relay_cluster_size: Optional[int] = None,
    max_topk: Optional[int] = None,
    dkv_mode: str = "records",
) -> int:
    """Return the reusable caller scratch required by one SM100 launch.

    The generic deterministic M64 backend selects 32, 64, or 128 dKV planes
    for ordinary extents. If that policy would exceed 16 GiB, it selects a
    8-, 16-, or 32-plane cluster relay and adds packed per-wave KV-row rank
    masks. It switches to the eight-plane cluster-16 layout when the preferred
    relay would exceed its memory-first scratch budget. A non-``None``
    ``deterministic_relay_cluster_size`` requests the exact 4-, 8-, 16-, 32-,
    or 64-CTA logical relay and reports that specialization's complete
    workspace. The 32-/64-CTA options use two/four ordered 16-CTA physical
    launches.
    H96/H128 D576 two-CTA backends use a compact 96-plane layout; unspecified
    backends use the conservative fixed 128-plane layout. The result includes
    alignment padding and is the minimum byte count for a contiguous
    ``torch.uint8`` workspace.

    Deterministic generic M64 plans store dKV in conflict-rank storage where
    ``rank_dkv_route`` selects it, which needs the plan's top-k width: pass
    ``max_topk`` and ``dkv_mode`` (the plan's ``deterministic_dkv_mode``) to
    get the plan's exact size. Without ``max_topk`` the result is the shard or
    relay layout's size, an upper bound of every conflict-rank plan of the
    shape except chain-mode H96 D576 plans (bounded by the 128-plane layout).
    Pass the backend the plan selects; for deterministic H96 D576 it depends on
    the extents, ``max_topk`` and ``dkv_mode`` (``_select_sm100_backend``).
    """
    workspace_lse_odo_shape, workspace_dkv_shape = _workspace_shapes_sm100(
        total_s_q,
        total_s_kv,
        head_dim,
        num_heads,
        deterministic,
        backend,
        deterministic_relay_cluster_size,
        max_topk=max_topk,
        dkv_mode=dkv_mode,
    )
    return _align_workspace_bytes(math.prod(workspace_lse_odo_shape)) + _align_workspace_bytes(math.prod(workspace_dkv_shape))
