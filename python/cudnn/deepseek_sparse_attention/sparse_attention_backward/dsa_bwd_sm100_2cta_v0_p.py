"""Safe CG1 baseline-core reset for the experimental SM100 DSA backward v0_p.

The H128 work unit is launched as two independent H64 CTAs by the canonical
backward kernel.  All MMA, TMA, shared-memory, and TMEM ownership stays
CTA-local; dKV keeps the canonical FP32 atomic accumulation contract.

The private core retains the post-6c577ac TMEM WAR and
read-before-deallocation ordering fixes that were accidentally rolled back
from the shared development source, while preserving the current FP32
sum-OdO math.  Release and IKET builds select this exact same class; neither
build substitutes a different kernel.
"""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .dsa_bwd_sm100_v0_p_core import FlashAttentionDSABackwardSm100


V0_P_CG1_BASELINE_CORE = True
V0_P_SAFE_CG1_CORE = True


class FlashAttentionDSABackwardSm100TwoCTAV0(
    FlashAttentionDSABackwardSm100
):
    """Canonical CG1 core adapted to the isolated two-CTA harness contract."""

    # The harness uses N_TILE to select its validated H128/D512/topk path.
    N_TILE = 64

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        assert head_dim == 512
        assert head_dim_v == 512
        assert block_tile == 64
        super().__init__(head_dim, head_dim_v, block_tile, max_topk)

    @cute.kernel
    def _copy_clamp_topk_lengths_v0_p(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        topk_indices: cute.Tensor,
        count: Int32,
    ):
        """Clamp empty rows and hide their synthetic first sparse entry."""

        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        index = block_idx * 256 + thread_idx
        if index < count:
            value = source[index]
            if value == Int32(0):
                value = Int32(1)
                topk_indices[index, 0] = -topk_indices[index, 0] - Int32(2)
            destination[index] = value

    @cute.kernel
    def _restore_empty_topk_indices_v0_p(
        self,
        lengths: cute.Tensor,
        topk_indices: cute.Tensor,
        count: Int32,
    ):
        """Undo the stream-ordered empty-row sentinel encoding."""

        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        index = block_idx * 256 + thread_idx
        if index < count:
            if lengths[index] == Int32(0):
                topk_indices[index, 0] = -topk_indices[index, 0] - Int32(2)

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        """Adapt the harness signature to the canonical CG1 launcher."""

        del trace_buffer, trace_token_idx, trace_batch_idx
        if cutlass.const_expr(mTopkLength is not None):
            raw_topk_lengths = mTopkLength
            raw_topk_indices = mTopkIdxs
            length_count = mTopkLength.shape[0]
            length_scratch = cute.make_tensor(
                cute.recast_ptr(mdKV.iterator, dtype=Int32),
                cute.make_layout((length_count,), stride=(1,)),
            )
            self._copy_clamp_topk_lengths_v0_p(
                mTopkLength,
                length_scratch,
                mTopkIdxs,
                length_count,
            ).launch(
                grid=[cute.ceil_div(length_count, 256), 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )
            mTopkLength = length_scratch

        super().__call__(
            problem_shape,
            mQ,
            mKV,
            mOut,
            mdO,
            mLSE,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            mdQ,
            mdKV,
            mdSink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            stream,
        )

        if cutlass.const_expr(mTopkLength is not None):
            self._restore_empty_topk_indices_v0_p(
                raw_topk_lengths,
                raw_topk_indices,
                length_count,
            ).launch(
                grid=[cute.ceil_div(length_count, 256), 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )

    @cute.jit
    def _load_kv_rows(
        self,
        mKV: cute.Tensor,
        sK_slice: cute.Tensor,
        rTopkIdx: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        mTopkLength: Optional[cute.Tensor],
        is_first: bool,
        local_tidx: Int32,
        local_warp_idx: Int32,
        async_copy_atom: cute.CopyAtom,
        async_thr_copy: cute.TiledCopy,
    ):
        """Load valid sparse KV rows and zero padding or index holes."""

        token_idx, _, batch_idx = cute.arch.block_idx()
        rows_per_warp = self.block_tile // self.num_load_KV_warps
        for i in range(rows_per_warp):
            row = i * self.num_load_KV_warps + local_warp_idx
            idx = tile_index * self.block_tile + row
            tile_sK = sK_slice[row, (None, None)]
            topk_idx = rTopkIdx[i]

            if cutlass.const_expr(mTopkLength is not None):
                if cutlass.const_expr(is_first):
                    if idx < topk:
                        if topk_idx >= 0:
                            self._copy_kv_row(
                                mKV,
                                topk_idx,
                                batch_idx,
                                tile_sK,
                                local_tidx,
                                async_copy_atom,
                                async_thr_copy,
                            )
                        else:
                            self._zero_kv_row(tile_sK, local_tidx)
                    else:
                        self._zero_kv_row(tile_sK, local_tidx)
                else:
                    if topk_idx >= 0:
                        self._copy_kv_row(
                            mKV,
                            topk_idx,
                            batch_idx,
                            tile_sK,
                            local_tidx,
                            async_copy_atom,
                            async_thr_copy,
                        )
                    else:
                        self._zero_kv_row(tile_sK, local_tidx)
            else:
                if idx < topk:
                    if topk_idx >= 0:
                        self._copy_kv_row(
                            mKV,
                            topk_idx,
                            batch_idx,
                            tile_sK,
                            local_tidx,
                            async_copy_atom,
                            async_thr_copy,
                        )
                    else:
                        self._zero_kv_row(tile_sK, local_tidx)
                else:
                    self._zero_kv_row(tile_sK, local_tidx)
