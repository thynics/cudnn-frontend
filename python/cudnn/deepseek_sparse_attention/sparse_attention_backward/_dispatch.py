"""Internal architecture and shape dispatch for DSA backward."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

SM90_IMPLEMENTATION = "sm90"
SM100_IMPLEMENTATION = "sm100"
SM107_IMPLEMENTATION = "sm107"

_RUBIN_CAPABILITY = (10, 7)
_RUBIN_HEADS = 128
_RUBIN_HEAD_DIM = 512
_RUBIN_SEQLEN_KV = 4096
_RUBIN_BLOCK_TILE = 64
_RUBIN_TOPK = frozenset((512, 1024, 2048))


def select_sparse_attention_backward_implementation(
    *,
    capability: Tuple[int, int],
    device_type: str,
    all_inputs_same_device: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    dout_dtype: torch.dtype,
    q_shape: Sequence[int],
    kv_shape: Sequence[int],
    out_shape: Sequence[int],
    dout_shape: Sequence[int],
    lse_shape: Sequence[int],
    attn_sink_shape: Sequence[int],
    topk_idxs_shape: Sequence[int],
    block_tile: int,
) -> str:
    """Return the internal implementation tag using metadata only.

    Rubin currently has a performance-specialized fixed-shape kernel.  Every
    shape outside its validated envelope continues to use the SM100 kernel,
    including when it is executed natively on an SM107 device.
    """
    capability = tuple(capability)
    if len(capability) == 2 and capability[0] == 9:
        return SM90_IMPLEMENTATION

    q_shape = tuple(q_shape)
    kv_shape = tuple(kv_shape)
    out_shape = tuple(out_shape)
    dout_shape = tuple(dout_shape)
    lse_shape = tuple(lse_shape)
    attn_sink_shape = tuple(attn_sink_shape)
    topk_idxs_shape = tuple(topk_idxs_shape)

    if len(q_shape) != 3 or len(kv_shape) != 2 or len(topk_idxs_shape) != 2:
        return SM100_IMPLEMENTATION

    seqlen_q = q_shape[0]
    expected_q_shape = (seqlen_q, _RUBIN_HEADS, _RUBIN_HEAD_DIM)
    is_rubin_shape = (
        q_shape == expected_q_shape
        and out_shape == expected_q_shape
        and dout_shape == expected_q_shape
        and kv_shape == (_RUBIN_SEQLEN_KV, _RUBIN_HEAD_DIM)
        and lse_shape == (seqlen_q, _RUBIN_HEADS)
        and attn_sink_shape == (_RUBIN_HEADS,)
        and topk_idxs_shape[0] == seqlen_q
        and topk_idxs_shape[1] in _RUBIN_TOPK
    )
    is_rubin_dtype = q_dtype == kv_dtype == out_dtype == dout_dtype == torch.bfloat16

    if (
        capability == _RUBIN_CAPABILITY
        and device_type == "cuda"
        and all_inputs_same_device
        and is_rubin_dtype
        and is_rubin_shape
        and int(block_tile) == _RUBIN_BLOCK_TILE
    ):
        return SM107_IMPLEMENTATION

    return SM100_IMPLEMENTATION
