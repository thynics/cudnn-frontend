# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek Sparse Attention backward, SM100 two-CTA variant B.

The vkq6w final production schedule uses a four-stage true-K32 round ring
and reducer pacing while preserving the serial S -> dP -> dV -> dQ -> dK
chain."""

import math
from typing import Optional, Tuple
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05, warp
from cutlass.cute.typing import Float32, Int32
from .dsa_bwd_sm100_2_cta_A import _FlashAttentionDSABackwardSm100TwoCTABase, _cpasync_bulk_s2cluster, _mbarrier_wait_acquire_cluster


@dsl_user_op
def _nanosleep_u32(ns: Int32, *, loc=None, ip=None) -> None:
    """Pace reducer atomic bursts with a warp nanosleep hint."""
    llvm.inline_asm(
        None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


class FlashAttentionDSABackwardSm100TwoCTAVariantB(_FlashAttentionDSABackwardSm100TwoCTABase):
    """Two-CTA production kernel: 20 warps per CTA, five CG2 GEMMs per KV tile.

    Per-CTA roles: warps 0-3 gather the sparse K tile and the dQ-A
    images; warps 4-7 run the softmax backward and publish P/dS; warps
    8-15 drain the fused dV+dK partial sums to the f32 workspace; warp
    16 issues every GEMM and manages pipeline credits; warp 17 feeds
    the round ring from the stationary panels; warp 18 relays P/dS
    across the cluster; warp 19 relays ring TMA completions.

    Steady-state schedule per KV tile (SERIAL, no rotation): the whole
    chain of tile t issues in order S(t) -> dP(t) -> dV r0/r1 -> dQ ->
    dK r0/r1.  The four-slot true-K32 ring carries 16 generations/tile:
    eight dO followed by eight Q generations.  K_dQ occupies one score_kv
    loan generation between score K(t) and score K(t+1), and W16 releases
    that loan only after both dQ rounds and the TMEM-store fence.  P is
    relayed first on warp 18 so its exchange lands under the dP shadow;
    the same lane relays dS afterward.  dQ stays TMEM-resident across all
    tiles and stores through a two-round staged TMA epilogue.
    """

    THREADS_PER_CTA = 640
    GATHER_WARPS = 4
    MATH_WARP_BEGIN = 4
    MATH_WARPS = 4
    REDUCE_WARP_BEGIN = 8
    REDUCE_WARPS = 8
    MMA_WARP = 16
    LOAD_WARP = 17
    RELAY_WARP = 18
    COMMIT_WARP = 19
    GATHER_THREADS = GATHER_WARPS * 32
    MATH_THREAD_BEGIN = MATH_WARP_BEGIN * 32
    MATH_THREADS = MATH_WARPS * 32
    REDUCE_THREAD_BEGIN = REDUCE_WARP_BEGIN * 32
    REDUCE_THREADS = REDUCE_WARPS * 32
    DKV_MMA_TILER = (256, 64, 64)
    ROUND_TILER = (256, 64, 32)
    ROUND_STAGE_ELEMENTS = 4096
    ROUND_STAGE_BYTES = 8192
    PDS_BLOCK_ELEMENTS = 2048
    PDS_BLOCK_BYTES = 4096
    TMEM_S_OFFSET = 0
    TMEM_S1_OFFSET = 32
    TMEM_DP_OFFSET = 64
    TMEM_DP1_OFFSET = 96
    TMEM_DQ0_OFFSET = 128
    TMEM_DQ1_OFFSET = 256
    TMEM_DKV0_OFFSET = 384
    TMEM_DKV1_OFFSET = 448
    SCORE_DONE_STAGES = 2
    ROUND_PANELS_PER_TILE = 8
    ROUND_GENS_PER_TILE = 16
    ROUND_STAGES = 4
    MMA_DONE_STAGES = 2
    REDUCE_PACE_NS = 100
    REDUCE_DEPHASE_NS = 90

    def __init__(self, element_dtype, head_dim: int, head_dim_v: int, block_tile: int, max_topk: int = 0):
        super().__init__(element_dtype, head_dim, head_dim_v, block_tile, max_topk)
        self.math_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=self.MATH_THREADS)
        self.cta_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=self.THREADS_PER_CTA)
        self.gather_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=self.GATHER_THREADS)

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
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        """Compile preprocessing, the CG2 main kernel, and postprocessing."""
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout((mQ.shape[1], mQ.shape[2], (mQ.shape[0], 1)), stride=(mQ.stride[1], mQ.stride[2], (mQ.stride[0], 0)))
        )
        mKV = cute.make_tensor(mKV.iterator, cute.make_layout((mKV.shape[0], mKV.shape[1], (1, 1)), stride=(mKV.stride[0], mKV.stride[1], (0, 0))))
        mOut = cute.make_tensor(
            mOut.iterator, cute.make_layout((mOut.shape[1], mOut.shape[2], (mOut.shape[0], 1)), stride=(mOut.stride[1], mOut.stride[2], (mOut.stride[0], 0)))
        )
        mdO = cute.make_tensor(
            mdO.iterator, cute.make_layout((mdO.shape[1], mdO.shape[2], (mdO.shape[0], 1)), stride=(mdO.stride[1], mdO.stride[2], (mdO.stride[0], 0)))
        )
        mdQ = cute.make_tensor(
            mdQ.iterator, cute.make_layout((mdQ.shape[2], mdQ.shape[1], (mdQ.shape[0], 1)), stride=(mdQ.stride[2], mdQ.stride[1], (mdQ.stride[0], 0)))
        )
        mdQ_epi = cute.make_tensor(
            mdQ.iterator, cute.make_layout((self.H_TILE_CLUSTER, self.D_HEAD, mdQ.shape[2]), stride=(mdQ.stride[1], mdQ.stride[0], mdQ.stride[2]))
        )
        mdKV = cute.make_tensor(mdKV.iterator, cute.make_layout((mdKV.shape[1], mdKV.shape[0], (1, 1)), stride=(mdKV.stride[1], mdKV.stride[0], (0, 0))))
        mLSE = cute.make_tensor(mLSE.iterator, cute.make_layout((mLSE.shape[1], (mLSE.shape[0], 1)), stride=(mLSE.stride[1], (mLSE.stride[0], 0))))
        mdSink = cute.make_tensor(mdSink.iterator, cute.make_layout((mdSink.shape[0], (1, 1)), stride=(1, (0, 0))))
        mAttnSink = cute.make_tensor(mAttnSink.iterator, mdSink.layout)
        mTopkIdxs = cute.make_tensor(
            mTopkIdxs.iterator, cute.make_layout((mTopkIdxs.shape[1], (mTopkIdxs.shape[0], 1)), stride=(mTopkIdxs.stride[1], (mTopkIdxs.stride[0], 0)))
        )
        if cutlass.const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(mTopkLength.iterator, cute.make_layout((mTopkLength.shape[0], (1, 1)), stride=(mTopkLength.stride[0], (0, 0))))
        mQT = cute.make_tensor(
            mQ.iterator, cute.make_layout((self.D_HEAD, self.H_TILE_CLUSTER, mQ.shape[2]), stride=(mQ.stride[1], mQ.stride[0], mQ.stride[2]))
        )
        mdOT = cute.make_tensor(
            mdO.iterator, cute.make_layout((self.D_HEAD, self.H_TILE_CLUSTER, mdO.shape[2]), stride=(mdO.stride[1], mdO.stride[0], mdO.stride[2]))
        )
        cg1 = tcgen05.CtaGroup.ONE
        cg2 = tcgen05.CtaGroup.TWO
        stationary_tiler = (self.H_TILE_CTA, self.N_TILE, self.D_HEAD)
        stationary_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cg1, stationary_tiler[:2]
        )
        score_tiler = (self.H_TILE_CLUSTER, self.N_TILE, self.K_CHUNK)
        dkv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cg2, self.DKV_MMA_TILER[:2]
        )
        dq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.acc_dtype, cg2, self.DQ_MMA_TILER[:2]
        )
        score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cg2, score_tiler[:2]
        )
        dp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cg2, score_tiler[:2]
        )
        atom_thr_size = cute.size(dkv_tiled_mma.thr_id.shape)
        assert atom_thr_size == self.CLUSTER_SHAPE_MNK[0]
        assert cute.size(dq_tiled_mma.thr_id.shape) == atom_thr_size
        assert cute.size(score_tiled_mma.thr_id.shape) == atom_thr_size
        assert cute.size(dp_tiled_mma.thr_id.shape) == atom_thr_size
        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(self.CLUSTER_SHAPE_MNK), (dkv_tiled_mma.thr_id.shape,))
        score_a_layout_staged = sm100_utils.make_smem_layout_a(score_tiled_mma, score_tiler, self.element_dtype, self.K_CHUNKS)
        stationary_a_layout_staged = sm100_utils.make_smem_layout_a(stationary_tiled_mma, stationary_tiler, self.element_dtype, 1)
        score_b_layout_staged = sm100_utils.make_smem_layout_b(score_tiled_mma, score_tiler, self.element_dtype, self.K_CHUNKS)
        dkv_a_layout_staged = sm100_utils.make_smem_layout_a(dkv_tiled_mma, self.DKV_MMA_TILER, self.element_dtype, 1)
        round_a_layout_staged = sm100_utils.make_smem_layout_a(dkv_tiled_mma, self.ROUND_TILER, self.element_dtype, 1)
        dkv_b_layout_staged = sm100_utils.make_smem_layout_b(dkv_tiled_mma, self.DKV_MMA_TILER, self.element_dtype, 1)
        dq_a_layout_staged = sm100_utils.make_smem_layout_a(dq_tiled_mma, self.DQ_MMA_TILER, self.element_dtype, 1)
        dq_b_layout_staged = sm100_utils.make_smem_layout_b(dq_tiled_mma, self.DQ_MMA_TILER, self.element_dtype, 1)
        dq_epi_tile = (self.H_TILE_CLUSTER, self.D_TILE_CTA)
        dq_epi_layout_staged = sm100_utils.make_smem_layout_epi(self.element_dtype, utils.LayoutEnum.from_tensor(mdQ_epi), dq_epi_tile, 1)
        dq_epi_layout = cute.select(dq_epi_layout_staged, mode=[0, 1])
        dq_epi_bytes = cute.size_in_bytes(self.element_dtype, dq_epi_layout_staged)
        assert dq_epi_bytes <= 32 * 1024
        tma_atom_dq_epi, tma_tensor_dq_epi = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mdQ_epi, dq_epi_layout, dq_epi_tile)
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(stationary_a_layout_staged) == cute.cosize(score_a_layout_staged)
        assert stationary_a_layout_staged.inner == score_a_layout_staged.inner
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 16384
        round_stage_elements = cute.cosize(round_a_layout_staged)
        assert round_stage_elements == self.ROUND_STAGE_ELEMENTS
        assert cute.cosize(dkv_a_layout_staged) == 2 * round_stage_elements
        assert round_a_layout_staged.inner == dkv_a_layout_staged.inner
        assert cute.cosize(dkv_b_layout_staged) <= 4096
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096
        assert cute.cosize(score_a_layout_staged) >= self.H_TILE_CTA * self.N_TILE
        assert cute.cosize(score_b_layout_staged) >= self.QUADRANT_ELEMENTS
        stationary_a_layout = cute.select(stationary_a_layout_staged, mode=[0, 1, 2])
        score_a_layout = cute.select(score_a_layout_staged, mode=[0, 1, 2])
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mQ, stationary_a_layout, stationary_tiler, stationary_tiled_mma)
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mdO, stationary_a_layout, stationary_tiler, stationary_tiled_mma)
        score_a_stage_bytes = cute.size_in_bytes(self.element_dtype, score_a_layout)
        round_a_layout = cute.select(round_a_layout_staged, mode=[0, 1, 2])
        round_tma_atom_qt, round_tma_tensor_qt = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mQT, round_a_layout, self.ROUND_TILER, dkv_tiled_mma, cluster_layout_vmnk.shape
        )
        round_tma_atom_dot, round_tma_tensor_dot = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mdOT, round_a_layout, self.ROUND_TILER, dkv_tiled_mma, cluster_layout_vmnk.shape
        )
        round_stage_bytes = cute.size_in_bytes(self.element_dtype, round_a_layout)
        assert round_stage_bytes == self.ROUND_STAGE_BYTES
        local_bulk_stage_offset = cute.cosize(round_a_layout_staged)
        assert local_bulk_stage_offset == self.ROUND_STAGE_ELEMENTS
        assert cute.cosize(score_a_layout_staged) == 8 * local_bulk_stage_offset
        assert score_a_layout_staged.inner == round_a_layout_staged.inner
        SharedStorage = self._make_shared_storage(
            score_a_layout_staged, score_b_layout_staged, dkv_a_layout_staged, dkv_b_layout_staged, dq_a_layout_staged, dq_b_layout_staged
        )
        self.shared_storage = SharedStorage
        self.shared_storage_bytes = SharedStorage.size_in_bytes()
        assert self.shared_storage_bytes <= self.MAX_SMEM_BYTES
        score_tmem_load = self._make_score_tmem_load()
        dq_cta_shape = (self.D_TILE_CTA, self.H_TILE_CLUSTER, self.N_TILE)
        dq_epi_tile = sm100_utils.compute_epilogue_tile_shape(dq_cta_shape, True, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype)
        dq_tmem_load = sm100_utils.get_tmem_load_op(dq_cta_shape, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, dq_epi_tile, True)
        sum_OdO, scaled_LSE, mdKV_acc = self.get_workspace_tensor(problem_shape, workspace_LSE_OdO, workspace_dKV, mQ.shape[2][0], mKV.shape[0], self.acc_dtype)
        mdKV_acc = cute.make_tensor(mdKV_acc.iterator, mdKV.layout)
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))
        self.sum_OdO(mOut, mdO, sum_OdO, mLSE, mAttnSink, scaled_LSE, sum_OdO_scale, LSE_scale, problem_shape).launch(
            grid=self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q),
            block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )
        self.kernel(
            tma_atom_q,
            tma_tensor_q,
            tma_atom_do,
            tma_tensor_do,
            round_tma_atom_qt,
            round_tma_tensor_qt,
            round_tma_atom_dot,
            round_tma_tensor_dot,
            mKV,
            mdQ,
            mdKV_acc,
            mTopkIdxs,
            mTopkLength,
            scaled_LSE,
            sum_OdO,
            Float32(softmax_scale),
            score_tiled_mma,
            dp_tiled_mma,
            dkv_tiled_mma,
            dq_tiled_mma,
            score_a_layout_staged,
            score_b_layout_staged,
            round_a_layout_staged,
            dkv_b_layout_staged,
            dq_a_layout_staged,
            dq_b_layout_staged,
            cluster_layout_vmnk,
            score_tmem_load,
            dq_tmem_load,
            tma_atom_dq_epi,
            tma_tensor_dq_epi,
            dq_epi_layout_staged,
            score_a_stage_bytes,
            round_stage_bytes,
            stationary_tiled_mma,
            stationary_a_layout_staged,
        ).launch(
            grid=(2 * problem_shape[0], 1, problem_shape[3][1]),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=self.CLUSTER_SHAPE_MNK,
            smem=self.shared_storage_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )
        self.block_seq = 4 if self.max_topk == 2048 else 32
        self.num_threads_D_convert = 32
        self.num_threads_seq = 4 if self.max_topk == 2048 else self.block_seq
        convert_grid_x = (mKV.shape[0] + self.block_seq - 1) // self.block_seq
        self.convert_canonical(mdKV_acc, mdKV, mKV.shape[0]).launch(
            grid=[convert_grid_x, 1, 1], block=[self.num_threads_D_convert, self.num_threads_seq, 1], stream=stream
        )
        self.sum_dSink(sum_OdO, scaled_LSE, mAttnSink, mdSink, problem_shape).launch(
            grid=(cute.ceil_div(problem_shape[0], self.dSink_block_q), problem_shape[3][0], problem_shape[3][1]),
            block=[self.dSink_num_threads, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _make_score_tmem_load(self):
        """Use the 16-DP/256-bit score accumulator load required by the publish store layout."""
        return cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype)

    def _make_shared_storage(
        self, score_a_layout_staged, score_b_layout_staged, dkv_a_layout_staged, dkv_b_layout_staged, dq_a_layout_staged, dq_b_layout_staged
    ):
        element_dtype = self.element_dtype
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 8192
        assert cute.cosize(score_b_layout_staged) == 2 * cute.cosize(dkv_a_layout_staged)
        assert cute.cosize(dkv_a_layout_staged) == 8192
        assert cute.cosize(dkv_b_layout_staged) <= 2048
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096

        @cute.struct
        class SharedStorage:
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            kscore_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 8]
            pds_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dkv_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dq_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            landing_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            relay_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_tma_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            loan_epi_safe_mbar: cutlass.Int64
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            p_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            ds_local_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            stationary_q: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            stationary_do: cute.struct.Align[cute.struct.MemRange[element_dtype, 32768], 1024]
            score_kv: cute.struct.Align[cute.struct.MemRange[element_dtype, 16384], 1024]
            round_buf_a0: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            round_buf_a1: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            round_buf_b0: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            round_buf_b1: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            p_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            p_xchg: cute.struct.Align[cute.struct.MemRange[element_dtype, 2048], 1024]
            ds_image: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            ds_blocks: cute.struct.Align[cute.struct.MemRange[element_dtype, 4096], 1024]
            ds_xchg: cute.struct.Align[cute.struct.MemRange[element_dtype, 2048], 1024]
            stats: cute.struct.Align[cute.struct.MemRange[Float32, 128], 1024]

        assert SharedStorage.size_in_bytes() <= self.MAX_SMEM_BYTES
        return SharedStorage

    @cute.jit
    def _kd_round_rows(self, tensor: cute.Tensor) -> cute.Tensor:
        """Return an [N64, D128] row-major view of one dQ-A round buffer."""
        return cute.composition(tensor[None, None, None, 0], cute.make_layout((self.N_TILE, self.D_TILE_CTA), stride=(self.D_TILE_CTA, 1)))

    @cute.jit
    def _fill_kdq_pair(
        self,
        mKV: cute.Tensor,
        kd_rows_0: cute.Tensor,
        kd_rows_1: cute.Tensor,
        batch_idx: Int32,
        rank: Int32,
        role_tidx: Int32,
        thread_count: cutlass.Constexpr[int],
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        kv_index_0: Int32,
        kv_index_1: Int32,
        kv_index_2: Int32,
        kv_index_3: Int32,
    ) -> None:
        """Fill both sparse K-dQ panels from preloaded KV indices."""
        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = Int32(self.D_TILE_CLUSTER) + rank * Int32(self.D_TILE_CTA)
        assert self.N_TILE % groups_total == 0
        rows_per_group = self.N_TILE // groups_total
        assert rows_per_group == 4
        kdq_local_n = [row_iteration * groups_total + group_index for row_iteration in range(rows_per_group)]
        kdq_kv_index = [kv_index_0, kv_index_1, kv_index_2, kv_index_3]
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = kdq_local_n[row_iteration]
            kv_index = kdq_kv_index[row_iteration]
            if kv_index >= Int32(0):
                self._copy_sparse_k_d128_row(mKV, kd_rows_0, Int32(local_n), kv_index, batch_idx, d_offset_0, index_in_group, copy_atom, thread_copy)
                self._copy_sparse_k_d128_row(mKV, kd_rows_1, Int32(local_n), kv_index, batch_idx, d_offset_1, index_in_group, copy_atom, thread_copy)
            else:
                self._zero_sparse_k_d128_row(kd_rows_0, Int32(local_n), index_in_group)
                self._zero_sparse_k_d128_row(kd_rows_1, Int32(local_n), index_in_group)

    @cute.jit
    def _gather_kdq(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        kd_rows_0: cute.Tensor,
        kd_rows_1: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        role_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ) -> None:
        """Rendezvous-free kdq fill into the score_kv loan halves (kq).

        The K_dQ images live in score_kv under a kscore generation the
        CALLER has already acquired -- no load-warp barrier, no
        kdq_ready close.  Completion is the caller's cp.async drain +
        fence + kscore producer commit, the same protocol as
        _load_score_kv.
        """
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = self.GATHER_THREADS // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE // groups_total
        assert rows_per_group == 4
        kdq_local_n = [row_iteration * groups_total + group_index for row_iteration in range(rows_per_group)]
        kdq_kv_index = []
        for local_n in kdq_local_n:
            global_n = tile_index * Int32(self.N_TILE) + Int32(local_n)
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = mTopkIdxs[global_n, (token_idx, batch_idx)]
            kdq_kv_index.append(kv_index)
        self._fill_kdq_pair(
            mKV,
            kd_rows_0,
            kd_rows_1,
            batch_idx,
            rank,
            role_tidx,
            self.GATHER_THREADS,
            copy_atom,
            thread_copy,
            kdq_kv_index[0],
            kdq_kv_index[1],
            kdq_kv_index[2],
            kdq_kv_index[3],
        )

    @cute.jit
    def _issue_dq_rounds(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        kd_fragment_a: cute.Tensor,
        kd_fragment_b: cute.Tensor,
        ds_fragment: cute.Tensor,
        accumulate: cutlass.Boolean,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Issue both dQ rounds from one score_kv loan generation.

        Both 16 KiB K_dQ panels live simultaneously in the two score_kv
        halves.  One wait covers both rounds; one release returns score_kv
        to the gather warps for the next tile's score-K generation.
        """
        kscore_pipeline.consumer_wait(kscore_consumer_state)
        assert cute.size(kd_fragment_a, mode=[2]) == 4
        assert cute.size(kd_fragment_b, mode=[2]) == 4
        for round_index in cutlass.range_constexpr(self.D_ROUNDS):
            mma = dq_tiled_mma.with_()
            mma.set(tcgen05.Field.ACCUMULATE, accumulate)
            if cutlass.const_expr(round_index == 0):
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_a, mode=[2])):
                    cute.gemm(mma, t_dq_0, kd_fragment_a[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_0)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_b, mode=[2])):
                    cute.gemm(mma, t_dq_1, kd_fragment_b[None, None, k_block, 0], ds_fragment[None, None, k_block, 0], t_dq_1)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        kscore_pipeline.consumer_release(kscore_consumer_state)
        kscore_consumer_state.advance()
        return kscore_consumer_state

    @cute.jit
    def _issue_dkv_pass(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        b_k_half: cutlass.Constexpr[int],
        accumulate: cutlass.Constexpr[bool],
    ) -> None:
        """Issue one self-contained K32 stage against its original B half."""
        k_blocks = cute.size(a_fragment, mode=[2])
        assert k_blocks == 2
        assert cute.size(b_fragment, mode=[2]) == 2 * k_blocks
        b_k_block_offset = b_k_half * k_blocks
        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        for k_block in cutlass.range_constexpr(k_blocks):
            cute.gemm(mma, t_dkv, a_fragment[None, None, k_block, 0], b_fragment[None, None, b_k_block_offset + k_block, 0], t_dkv)
            mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _zero_dq(
        self, rank_coordinates: cute.Tensor, mdQ: cute.Tensor, round_index: cutlass.Constexpr[int], token_idx: Int32, batch_idx: Int32, tidx: Int32
    ) -> None:
        """Write the required all-zero dQ result when no tile is issued."""
        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            linear_index = tidx
            while linear_index < cute.size(rank_coordinates):
                coordinate = cute.idx2crd(linear_index, rank_coordinates.shape)
                logical_coordinate = rank_coordinates[coordinate]
                d_in_round = Int32(cute.get(logical_coordinate, mode=[0]))
                head = Int32(cute.get(logical_coordinate, mode=[1]))
                mdQ[Int32(round_index * self.D_TILE_CLUSTER) + d_in_round, head, (token_idx, batch_idx)] = self.element_dtype(0.0)
                linear_index += Int32(self.MATH_THREADS_PER_CTA)

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        round_tma_atom_qt: cute.CopyAtom,
        round_tma_tensor_qt: cute.Tensor,
        round_tma_atom_dot: cute.CopyAtom,
        round_tma_tensor_dot: cute.Tensor,
        mKV: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        scaled_lse: cute.Tensor,
        sum_odo: cute.Tensor,
        scale_softmax: Float32,
        score_tiled_mma: cute.TiledMma,
        dp_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        dq_tiled_mma: cute.TiledMma,
        score_a_layout_staged: cute.ComposedLayout,
        score_b_layout_staged: cute.ComposedLayout,
        round_a_layout_staged: cute.ComposedLayout,
        dkv_b_layout_staged: cute.ComposedLayout,
        dq_a_layout_staged: cute.ComposedLayout,
        dq_b_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        score_tmem_load: cute.CopyAtom,
        dq_tmem_load: cute.CopyAtom,
        tma_atom_dq_epi: cute.CopyAtom,
        tma_tensor_dq_epi: cute.Tensor,
        dq_epi_layout_staged: cute.ComposedLayout,
        score_a_stage_bytes: cutlass.Constexpr[int],
        round_stage_bytes: cutlass.Constexpr[int],
        stationary_tiled_mma: cute.TiledMma,
        stationary_a_layout_staged: cute.ComposedLayout,
    ):
        """Execute the vkq6w serial five-GEMM two-CTA schedule."""
        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)
        peer_rank = Int32(1) - rank
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        is_leader_cta = rank == Int32(0)
        if warp_idx == Int32(self.LOAD_WARP):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(round_tma_atom_qt)
            cpasync.prefetch_descriptor(round_tma_atom_dot)
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_holding_buf_ptr = storage.tmem_holding_buf.ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr
        stationary_tma_mbars = storage.stationary_tma_mbars.data_ptr()
        stationary_ready_mbar = storage.stationary_ready_mbar.data_ptr()
        landing_mbars = storage.landing_mbars.data_ptr()
        relay_mbars = storage.relay_mbars.data_ptr()
        pds_ready_mbars = storage.pds_ready_mbars.data_ptr()
        p_ready_mbars = storage.p_ready_mbars.data_ptr()
        ds_local_ready_mbar = storage.ds_local_ready_mbar.data_ptr()
        round_tma_mbars = storage.round_tma_mbars.data_ptr()
        loan_epi_safe_mbar = storage.loan_epi_safe_mbar.ptr
        stationary_q_raw = storage.stationary_q.data_ptr()
        stationary_do_raw = storage.stationary_do.data_ptr()
        round_slot_raw = (storage.round_buf_a0.data_ptr(), storage.round_buf_a1.data_ptr(), storage.round_buf_b0.data_ptr(), storage.round_buf_b1.data_ptr())
        score_kv_raw = storage.score_kv.data_ptr()
        stationary_q = storage.stationary_q.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_do = storage.stationary_do.get_tensor(score_a_layout_staged.outer, swizzle=score_a_layout_staged.inner)
        stationary_q_tma = storage.stationary_q.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        stationary_do_tma = storage.stationary_do.get_tensor(stationary_a_layout_staged.outer, swizzle=stationary_a_layout_staged.inner)
        k_n = storage.score_kv.get_tensor(score_b_layout_staged.outer, swizzle=score_b_layout_staged.inner)
        s_dq_epi = cute.make_tensor(cute.recast_ptr(storage.score_kv.data_ptr(), dq_epi_layout_staged.inner, self.element_dtype), dq_epi_layout_staged.outer)[
            None, None, 0
        ]
        kdq_loan_ptr_0 = cute.make_ptr(self.element_dtype, score_kv_raw.toint(), score_kv_raw.memspace, assumed_align=1024)
        kdq_loan_ptr_1 = cute.make_ptr(self.element_dtype, score_kv_raw.toint() + Int32(16384), score_kv_raw.memspace, assumed_align=1024)
        kdq_loan = (
            cute.make_tensor(cute.recast_ptr(kdq_loan_ptr_0, dq_a_layout_staged.inner, dtype=self.element_dtype), dq_a_layout_staged.outer),
            cute.make_tensor(cute.recast_ptr(kdq_loan_ptr_1, dq_a_layout_staged.inner, dtype=self.element_dtype), dq_a_layout_staged.outer),
        )
        round_slots = (
            storage.round_buf_a0.get_tensor(round_a_layout_staged.outer, swizzle=round_a_layout_staged.inner),
            storage.round_buf_a1.get_tensor(round_a_layout_staged.outer, swizzle=round_a_layout_staged.inner),
            storage.round_buf_b0.get_tensor(round_a_layout_staged.outer, swizzle=round_a_layout_staged.inner),
            storage.round_buf_b1.get_tensor(round_a_layout_staged.outer, swizzle=round_a_layout_staged.inner),
        )
        p_blocks_raw = storage.p_blocks.data_ptr()
        ds_blocks_raw = storage.ds_blocks.data_ptr()
        ds_image_raw = storage.ds_image.data_ptr()
        p_blocks = (
            cute.make_tensor(cute.recast_ptr(p_blocks_raw, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer),
            cute.make_tensor(
                cute.recast_ptr(p_blocks_raw + self.PDS_BLOCK_ELEMENTS, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer
            ),
        )
        ds_blocks = (
            cute.make_tensor(cute.recast_ptr(ds_blocks_raw, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer),
            cute.make_tensor(
                cute.recast_ptr(ds_blocks_raw + self.PDS_BLOCK_ELEMENTS, dkv_b_layout_staged.inner, dtype=self.element_dtype), dkv_b_layout_staged.outer
            ),
        )
        ds_image = storage.ds_image.get_tensor(dq_b_layout_staged.outer, swizzle=dq_b_layout_staged.inner)
        score_store_layout = sm100_utils.make_smem_layout_epi(self.element_dtype, utils.LayoutEnum.COL_MAJOR, (self.H_TILE_CTA, self.N_TILE), 1)
        assert cute.cosize(score_store_layout) == cute.cosize(dq_b_layout_staged)
        assert score_store_layout.inner == dq_b_layout_staged.inner
        assert score_store_layout.inner == dkv_b_layout_staged.inner
        score_store_domain = cute.make_layout((score_store_layout.outer.shape, 1, 1, 1), stride=(score_store_layout.outer.stride, 0, 0, 0))
        assert cute.cosize(score_store_domain) == cute.cosize(dq_b_layout_staged)
        ds_image_store = storage.ds_image.get_tensor(score_store_domain, swizzle=score_store_layout.inner)
        p_block_stage = p_blocks[0][None, None, None, 0]
        assert cute.size(p_block_stage, mode=[0, 0]) == self.N_TILE_CTA
        assert cute.size(p_block_stage, mode=[0, 1]) == 16
        assert cute.size(p_block_stage, mode=[1]) == 1
        assert cute.size(p_block_stage, mode=[2]) == 4
        assert cute.size(p_block_stage) == self.PDS_BLOCK_ELEMENTS
        p_block_raw_ptrs = (p_blocks_raw, p_blocks_raw + self.PDS_BLOCK_ELEMENTS)
        ds_block_raw_ptrs = (ds_blocks_raw, ds_blocks_raw + self.PDS_BLOCK_ELEMENTS)
        flat_pds_block_layout = cute.make_layout((self.PDS_BLOCK_ELEMENTS,), stride=(1,))
        p_xchg_raw = storage.p_xchg.get_tensor(flat_pds_block_layout)
        ds_xchg_raw = storage.ds_xchg.get_tensor(flat_pds_block_layout)
        softmax_stats = storage.stats.get_tensor(cute.make_layout((self.H_TILE_CTA, 2), stride=(1, self.H_TILE_CTA)))
        stats_copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS), self.acc_dtype, num_bits_per_copy=64)
        stats_tiled_copy = cute.make_tiled_copy_tv(stats_copy_atom, cute.make_layout((32,), stride=(1,)), cute.make_layout((2,), stride=(1,)))
        stats_thread_copy = stats_tiled_copy.get_slice(tidx % Int32(32))
        g_scaled_lse = cute.flat_divide(scaled_lse, (self.H_TILE_CTA,))
        g_sum_odo = cute.flat_divide(sum_odo, (self.H_TILE_CTA,))
        t_g_scaled_lse = stats_thread_copy.partition_S(g_scaled_lse[None, rank, (token_idx, batch_idx)])
        t_s_scaled_lse = stats_thread_copy.partition_D(softmax_stats[None, 0])
        t_g_sum_odo = stats_thread_copy.partition_S(g_sum_odo[None, rank, (token_idx, batch_idx)])
        t_s_sum_odo = stats_thread_copy.partition_D(softmax_stats[None, 1])
        g_q = cute.local_tile(tma_tensor_q, cute.select((self.H_TILE_CTA, self.N_TILE, self.D_HEAD), mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        g_do = cute.local_tile(tma_tensor_do, cute.select((self.H_TILE_CTA, self.N_TILE, self.D_HEAD), mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        stationary_thr_mma = stationary_tiled_mma.get_slice(0)
        rank_g_q = stationary_thr_mma.partition_A(g_q)
        rank_g_do = stationary_thr_mma.partition_A(g_do)
        t_q_smem, t_q_gmem = cpasync.tma_partition(
            tma_atom_q, 0, cute.make_layout(1), cute.group_modes(stationary_q_tma, 0, 3), cute.group_modes(rank_g_q, 0, 3)
        )
        t_do_smem, t_do_gmem = cpasync.tma_partition(
            tma_atom_do, 0, cute.make_layout(1), cute.group_modes(stationary_do_tma, 0, 3), cute.group_modes(rank_g_do, 0, 3)
        )
        rank_score_mma = score_tiled_mma.get_slice(rank)
        rank_dkv_mma = dkv_tiled_mma.get_slice(rank)
        rank_dq_mma = dq_tiled_mma.get_slice(rank)
        rank_score_coordinates = rank_score_mma.partition_C(cute.make_identity_tensor((self.H_TILE_CLUSTER, self.N_TILE)))
        rank_dq_coordinates = rank_dq_mma.partition_C(cute.make_identity_tensor(self.DQ_MMA_TILER[:2]))
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        g_qt_round = cute.local_tile(round_tma_tensor_qt, cute.select(self.ROUND_TILER, mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        g_dot_round = cute.local_tile(round_tma_tensor_dot, cute.select(self.ROUND_TILER, mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        rank_g_qt_round = rank_dkv_mma.partition_A(g_qt_round)
        rank_g_dot_round = rank_dkv_mma.partition_A(g_dot_round)
        t_qt_round_smem_0, t_qt_round_gmem = cpasync.tma_partition(
            round_tma_atom_qt, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[0], 0, 3), cute.group_modes(rank_g_qt_round, 0, 3)
        )
        t_qt_round_smem_1, _ = cpasync.tma_partition(
            round_tma_atom_qt, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[1], 0, 3), cute.group_modes(rank_g_qt_round, 0, 3)
        )
        t_qt_round_smem_2, _ = cpasync.tma_partition(
            round_tma_atom_qt, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[2], 0, 3), cute.group_modes(rank_g_qt_round, 0, 3)
        )
        t_qt_round_smem_3, _ = cpasync.tma_partition(
            round_tma_atom_qt, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[3], 0, 3), cute.group_modes(rank_g_qt_round, 0, 3)
        )
        t_dot_round_smem_0, t_dot_round_gmem = cpasync.tma_partition(
            round_tma_atom_dot, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[0], 0, 3), cute.group_modes(rank_g_dot_round, 0, 3)
        )
        t_dot_round_smem_1, _ = cpasync.tma_partition(
            round_tma_atom_dot, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[1], 0, 3), cute.group_modes(rank_g_dot_round, 0, 3)
        )
        t_dot_round_smem_2, _ = cpasync.tma_partition(
            round_tma_atom_dot, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[2], 0, 3), cute.group_modes(rank_g_dot_round, 0, 3)
        )
        t_dot_round_smem_3, _ = cpasync.tma_partition(
            round_tma_atom_dot, block_coord_vmnk[2], a_cta_layout, cute.group_modes(round_slots[3], 0, 3), cute.group_modes(rank_g_dot_round, 0, 3)
        )
        t_qt_round_smem = (t_qt_round_smem_0, t_qt_round_smem_1, t_qt_round_smem_2, t_qt_round_smem_3)
        t_dot_round_smem = (t_dot_round_smem_0, t_dot_round_smem_1, t_dot_round_smem_2, t_dot_round_smem_3)
        score_q_fragment = score_tiled_mma.make_fragment_A(stationary_q)
        score_do_fragment = dp_tiled_mma.make_fragment_A(stationary_do)
        score_k_fragment = score_tiled_mma.make_fragment_B(k_n)
        dp_k_fragment = dp_tiled_mma.make_fragment_B(k_n)
        dq_kd_fragment_a = dq_tiled_mma.make_fragment_A(kdq_loan[0])
        dq_kd_fragment_b = dq_tiled_mma.make_fragment_A(kdq_loan[1])
        dq_ds_fragment = dq_tiled_mma.make_fragment_B(ds_image)
        assert cute.cosize(round_a_layout_staged) == self.ROUND_STAGE_ELEMENTS
        round_fragments = (
            dkv_tiled_mma.make_fragment_A(round_slots[0]),
            dkv_tiled_mma.make_fragment_A(round_slots[1]),
            dkv_tiled_mma.make_fragment_A(round_slots[2]),
            dkv_tiled_mma.make_fragment_A(round_slots[3]),
        )
        for round_slot in cutlass.range_constexpr(self.ROUND_STAGES):
            round_slot_tensor = round_slots[round_slot]
            round_fragment = round_fragments[round_slot]
            assert cute.cosize(round_slot_tensor.layout) == self.ROUND_STAGE_ELEMENTS
            assert cute.size(round_slot_tensor, mode=[2]) == 2
            assert cute.size(round_fragment, mode=[2]) == 2
            for k_block in cutlass.range_constexpr(2):
                k_block_slice = round_slot_tensor[None, None, k_block, 0]
                k_block_offset = round_slot_tensor.layout((0, 0, k_block, 0))
                k_block_cosize = cute.cosize(k_block_slice.layout)
                assert k_block_offset >= 0
                assert k_block_offset + k_block_cosize <= self.ROUND_STAGE_ELEMENTS
        p_fragments = (dkv_tiled_mma.make_fragment_B(p_blocks[0]), dkv_tiled_mma.make_fragment_B(p_blocks[1]))
        ds_fragments = (dkv_tiled_mma.make_fragment_B(ds_blocks[0]), dkv_tiled_mma.make_fragment_B(ds_blocks[1]))
        kv_copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), self.element_dtype, num_bits_per_copy=128)
        kv_thread_copy = cute.make_tiled_copy_tv(kv_copy_atom, cute.make_layout((1,)), cute.make_layout((8,))).get_slice(0)
        atom_thr_size = cute.size(score_tiled_mma.thr_id.shape)
        leader_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        math_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.MATH_THREADS)
        gather_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.GATHER_THREADS)
        reduce_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size * self.REDUCE_THREADS)
        load_elect_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size)
        pipe_s_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.SCORE_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.s_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_s_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_s_done.sync_object_full,
            sync_object_empty=pipe_s_done.sync_object_empty,
            num_stages=pipe_s_done.num_stages,
            producer_mask=pipe_s_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_s_done.cta_group,
        )
        pipe_dp_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.SCORE_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dp_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dp_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_dp_done.sync_object_full,
            sync_object_empty=pipe_dp_done.sync_object_empty,
            num_stages=pipe_dp_done.num_stages,
            producer_mask=pipe_dp_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_dp_done.cta_group,
        )
        pipe_kscore = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=gather_group,
            consumer_group=leader_group,
            barrier_storage=storage.kscore_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_kscore = pipeline.PipelineAsyncUmma(
            sync_object_full=pipe_kscore.sync_object_full,
            sync_object_empty=pipe_kscore.sync_object_empty,
            num_stages=pipe_kscore.num_stages,
            producer_mask=Int32(0),
            consumer_mask=pipe_kscore.consumer_mask,
            cta_group=pipe_kscore.cta_group,
        )
        pipe_round = pipeline.PipelineAsyncUmma.create(
            num_stages=self.ROUND_STAGES,
            producer_group=load_elect_group,
            consumer_group=leader_group,
            barrier_storage=storage.round_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_round = pipeline.PipelineAsyncUmma(
            sync_object_full=pipe_round.sync_object_full,
            sync_object_empty=pipe_round.sync_object_empty,
            num_stages=pipe_round.num_stages,
            producer_mask=Int32(0),
            consumer_mask=pipe_round.consumer_mask,
            cta_group=pipe_round.cta_group,
        )
        pds_commit_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, atom_thr_size)
        pipe_pds = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pds_commit_group,
            consumer_group=leader_group,
            barrier_storage=storage.pds_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_pds = pipeline.PipelineAsyncUmma(
            sync_object_full=pipe_pds.sync_object_full,
            sync_object_empty=pipe_pds.sync_object_empty,
            num_stages=pipe_pds.num_stages,
            producer_mask=Int32(0),
            consumer_mask=pipe_pds.consumer_mask,
            cta_group=pipe_pds.cta_group,
        )
        pipe_dkv_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.MMA_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=reduce_group,
            barrier_storage=storage.dkv_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dkv_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_dkv_done.sync_object_full,
            sync_object_empty=pipe_dkv_done.sync_object_empty,
            num_stages=pipe_dkv_done.num_stages,
            producer_mask=pipe_dkv_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_dkv_done.cta_group,
        )
        pipe_dq_done = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dq_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dq_done = pipeline.PipelineUmmaAsync(
            sync_object_full=pipe_dq_done.sync_object_full,
            sync_object_empty=pipe_dq_done.sync_object_empty,
            num_stages=pipe_dq_done.num_stages,
            producer_mask=pipe_dq_done.producer_mask,
            consumer_mask=Int32(0),
            cta_group=pipe_dq_done.cta_group,
        )
        if tidx == Int32(0):
            cute.arch.mbarrier_init(stationary_tma_mbars, 1)
            cute.arch.mbarrier_init(stationary_tma_mbars + 1, 1)
            cute.arch.mbarrier_init(stationary_ready_mbar, 2)
            cute.arch.mbarrier_init(stationary_ready_mbar + 1, 2)
            cute.arch.mbarrier_init(landing_mbars, 1)
            cute.arch.mbarrier_init(landing_mbars + 1, 1)
            cute.arch.mbarrier_init(relay_mbars, 2)
            cute.arch.mbarrier_init(relay_mbars + 1, 2)
            cute.arch.mbarrier_init(pds_ready_mbars, self.MATH_WARPS)
            cute.arch.mbarrier_init(p_ready_mbars, self.MATH_WARPS)
            cute.arch.mbarrier_init(ds_local_ready_mbar, 2)
            for round_slot in cutlass.range_constexpr(self.ROUND_STAGES):
                cute.arch.mbarrier_init(round_tma_mbars + round_slot, 1)
            cute.arch.mbarrier_init(loan_epi_safe_mbar, 1)
        cute.arch.fence_view_async_shared()
        self.cta_barrier.arrive_and_wait()
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(
            tmem_holding_buf_ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.MATH_WARP_BEGIN,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        score_c_layout = score_tiled_mma.make_fragment_C(score_tiled_mma.partition_shape_C((self.H_TILE_CLUSTER, self.N_TILE))).layout
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(dkv_tiled_mma.partition_shape_C(self.DKV_MMA_TILER[:2])).layout
        dq_c_layout = dq_tiled_mma.make_fragment_C(dq_tiled_mma.partition_shape_C(self.DQ_MMA_TILER[:2])).layout
        t_score = cute.make_tensor(tmem_ptr + self.TMEM_S_OFFSET, score_c_layout)
        t_score_pp = cute.make_tensor(tmem_ptr + self.TMEM_S1_OFFSET, score_c_layout)
        t_dp = cute.make_tensor(tmem_ptr + self.TMEM_DP_OFFSET, score_c_layout)
        t_dp_pp = cute.make_tensor(tmem_ptr + self.TMEM_DP1_OFFSET, score_c_layout)
        t_dq = (cute.make_tensor(tmem_ptr + self.TMEM_DQ0_OFFSET, dq_c_layout), cute.make_tensor(tmem_ptr + self.TMEM_DQ1_OFFSET, dq_c_layout))
        t_dkv = (cute.make_tensor(tmem_ptr + self.TMEM_DKV0_OFFSET, dkv_c_layout), cute.make_tensor(tmem_ptr + self.TMEM_DKV1_OFFSET, dkv_c_layout))
        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = Int32(mTopkIdxs.shape[0])
        if topk > Int32(mTopkIdxs.shape[0]):
            topk = Int32(mTopkIdxs.shape[0])
        if topk < Int32(0):
            topk = Int32(0)
        tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(self.N_TILE)
        tile_count = cute.arch.make_warp_uniform(tile_count)
        if warp_idx < Int32(self.MATH_WARP_BEGIN):
            cute.arch.setmaxregister_decrease(48)
        elif warp_idx >= Int32(self.MMA_WARP):
            cute.arch.setmaxregister_decrease(64)
        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            cute.arch.setmaxregister_increase(136)
        else:
            cute.arch.setmaxregister_increase(112)
        if warp_idx < Int32(self.GATHER_WARPS):
            gather_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            gather_kd_rows_0 = self._kd_round_rows(kdq_loan[0])
            gather_kd_rows_1 = self._kd_round_rows(kdq_loan[1])
            if tile_count > Int32(0):
                pipe_kscore.producer_acquire(gather_state)
                self._load_score_kv(mKV, mTopkIdxs, k_n, token_idx, batch_idx, tile_count - Int32(1), topk, rank, tidx, kv_copy_atom, kv_thread_copy)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                pipe_kscore.producer_commit(gather_state)
                gather_state.advance()
                for score_iter in cutlass.range(Int32(0), tile_count):
                    pipe_kscore.producer_acquire(gather_state)
                    self._gather_kdq(
                        mKV,
                        mTopkIdxs,
                        gather_kd_rows_0,
                        gather_kd_rows_1,
                        token_idx,
                        batch_idx,
                        tile_count - Int32(1) - score_iter,
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                    if score_iter != tile_count - Int32(1):
                        next_iter = score_iter + Int32(1)
                        pipe_kscore.producer_acquire(gather_state)
                        self._load_score_kv(
                            mKV, mTopkIdxs, k_n, token_idx, batch_idx, tile_count - Int32(1) - next_iter, topk, rank, tidx, kv_copy_atom, kv_thread_copy
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        pipe_kscore.producer_commit(gather_state)
                        gather_state.advance()
                pipe_kscore.producer_tail(gather_state)
                self.gather_barrier.arrive_and_wait()
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(loan_epi_safe_mbar)
        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            mtx = tidx - Int32(self.MATH_THREAD_BEGIN)
            if warp_idx == Int32(self.MATH_WARP_BEGIN):
                if tile_count > Int32(0):
                    cute.copy(stats_copy_atom, t_g_scaled_lse[None, 0], t_s_scaled_lse[None, 0])
                    cute.copy(stats_copy_atom, t_g_sum_odo[None, 0], t_s_sum_odo[None, 0])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
            self.math_barrier.arrive_and_wait()
            s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            dp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.SCORE_DONE_STAGES)
            pds_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dq_done_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            score_copy = tcgen05.make_tmem_copy(score_tmem_load, t_score)
            score_thread = score_copy.get_slice(mtx)
            score_source = score_thread.partition_S(t_score)
            score_coordinates = score_thread.partition_D(rank_score_coordinates)
            dp_copy = tcgen05.make_tmem_copy(score_tmem_load, t_dp)
            dp_thread = dp_copy.get_slice(mtx)
            dp_source = dp_thread.partition_S(t_dp)
            score_copy_pp = tcgen05.make_tmem_copy(score_tmem_load, t_score_pp)
            score_source_pp = score_copy_pp.get_slice(mtx).partition_S(t_score_pp)
            dp_copy_pp = tcgen05.make_tmem_copy(score_tmem_load, t_dp_pp)
            dp_source_pp = dp_copy_pp.get_slice(mtx).partition_S(t_dp_pp)
            smem_store_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.COL_MAJOR, self.element_dtype, self.acc_dtype, score_copy)
            assert isinstance(smem_store_atom.op, warp.StMatrix8x8x16bOp)
            assert smem_store_atom.op.num_matrices == 4
            tiled_copy_r2s = cute.make_tiled_copy_D(smem_store_atom, score_copy)
            thread_copy_r2s = tiled_copy_r2s.get_slice(mtx)
            t_rs_ds = thread_copy_r2s.partition_D(ds_image_store)
            assert cute.size(t_rs_ds, mode=[4]) == 1
            t_rs_ds_tile = t_rs_ds[None, None, None, None, 0]
            aligned_p_blocks_ptr = cute.make_ptr(self.element_dtype, p_blocks[0].iterator.toint(), p_blocks[0].memspace, assumed_align=16)
            aligned_ds_blocks_ptr = cute.make_ptr(self.element_dtype, ds_blocks[0].iterator.toint(), ds_blocks[0].memspace, assumed_align=16)
            p_local_store = cute.make_tensor(cute.recast_ptr(aligned_p_blocks_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            ds_local_store = cute.make_tensor(cute.recast_ptr(aligned_ds_blocks_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            aligned_p_xchg_ptr = cute.make_ptr(
                self.element_dtype,
                p_xchg_raw.iterator.toint() - mtx // Int32(self.H_TILE_CTA) * Int32(self.PDS_BLOCK_BYTES),
                p_xchg_raw.memspace,
                assumed_align=16,
            )
            aligned_ds_xchg_ptr = cute.make_ptr(
                self.element_dtype,
                ds_xchg_raw.iterator.toint() - mtx // Int32(self.H_TILE_CTA) * Int32(self.PDS_BLOCK_BYTES),
                ds_xchg_raw.memspace,
                assumed_align=16,
            )
            p_xchg_store = cute.make_tensor(cute.recast_ptr(aligned_p_xchg_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            ds_xchg_store = cute.make_tensor(cute.recast_ptr(aligned_ds_xchg_ptr, score_store_layout.inner, dtype=self.element_dtype), score_store_domain)
            t_rs_p_local = thread_copy_r2s.partition_D(p_local_store)
            t_rs_ds_local = thread_copy_r2s.partition_D(ds_local_store)
            t_rs_p_xchg = thread_copy_r2s.partition_D(p_xchg_store)
            t_rs_ds_xchg = thread_copy_r2s.partition_D(ds_xchg_store)
            assert cute.size(t_rs_p_local, mode=[4]) == 1
            assert cute.size(t_rs_ds_local, mode=[4]) == 1
            assert cute.size(t_rs_p_xchg, mode=[4]) == 1
            assert cute.size(t_rs_ds_xchg, mode=[4]) == 1
            t_rs_p_local_tile = t_rs_p_local[None, None, None, None, 0]
            t_rs_ds_local_tile = t_rs_ds_local[None, None, None, None, 0]
            t_rs_p_xchg_tile = t_rs_p_xchg[None, None, None, None, 0]
            t_rs_ds_xchg_tile = t_rs_ds_xchg[None, None, None, None, 0]
            r_score = cute.make_rmem_tensor(score_coordinates.shape, self.acc_dtype)
            r_dp = cute.make_rmem_tensor(score_coordinates.shape, self.acc_dtype)
            r_p = cute.make_rmem_tensor(score_coordinates.shape, self.element_dtype)
            r_ds = cute.make_rmem_tensor(score_coordinates.shape, self.element_dtype)
            softmax_scale_log2_e = scale_softmax * Float32(math.log2(math.e))
            hoist_group_bases = [2 * (h_group % 2) + 16 * (h_group // 2) for h_group in range(4)]
            hoist_group_local_h = [Int32(cute.get(score_coordinates[group_base], mode=[0])) % Int32(self.H_TILE_CTA) for group_base in hoist_group_bases]
            hoist_band_indices = [[group_base + j % 2 + 4 * (j // 2) for j in range(8)] for group_base in hoist_group_bases]
            hoist_lse = [softmax_stats[hoist_group_local_h[h_group], 0] for h_group in range(4)]
            hoist_delta = [softmax_stats[hoist_group_local_h[h_group], 1] for h_group in range(4)]
            for loop_iter in cutlass.range(tile_count):
                pipe_s_done.consumer_wait(s_state)
                if s_state.index == Int32(0):
                    cute.copy(score_copy, score_source, r_score)
                else:
                    cute.copy(score_copy_pp, score_source_pp, r_score)
                cute.arch.fence_view_async_tmem_load()
                pipe_s_done.consumer_release(s_state)
                s_state.advance()
                assert cute.size(r_score) == self.N_TILE_CTA
                for h_group in cutlass.range_constexpr(4):
                    lse = hoist_lse[h_group]
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        v0, v1 = cute.arch.fma_packed_f32x2((r_score[i0], r_score[i1]), (softmax_scale_log2_e, softmax_scale_log2_e), (lse, lse))
                        v0 = cute.math.exp2(v0, fastmath=True)
                        v1 = cute.math.exp2(v1, fastmath=True)
                        r_score[i0] = v0
                        r_score[i1] = v1
                        r_p[i0] = self.element_dtype(v0)
                        r_p[i1] = self.element_dtype(v1)
                pipe_pds.producer_acquire(pds_state)
                r_p_store = thread_copy_r2s.retile(r_p)
                assert t_rs_p_local_tile.shape == r_p_store.shape
                assert t_rs_p_xchg_tile.shape == r_p_store.shape
                if cute.arch.make_warp_uniform(mtx // Int32(self.H_TILE_CTA)) == cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()):
                    cute.copy(tiled_copy_r2s, r_p_store, t_rs_p_local_tile)
                else:
                    cute.copy(tiled_copy_r2s, r_p_store, t_rs_p_xchg_tile)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(p_ready_mbars)
                pipe_dp_done.consumer_wait(dp_state)
                if dp_state.index == Int32(0):
                    cute.copy(dp_copy, dp_source, r_dp)
                else:
                    cute.copy(dp_copy_pp, dp_source_pp, r_dp)
                cute.arch.fence_view_async_tmem_load()
                pipe_dp_done.consumer_release(dp_state)
                dp_state.advance()
                for h_group in cutlass.range_constexpr(4):
                    delta = hoist_delta[h_group]
                    for pair in cutlass.range_constexpr(4):
                        i0 = hoist_band_indices[h_group][2 * pair]
                        i1 = hoist_band_indices[h_group][2 * pair + 1]
                        d0, d1 = cute.arch.add_packed_f32x2((r_dp[i0], r_dp[i1]), (delta, delta))
                        d0, d1 = cute.arch.mul_packed_f32x2((d0, d1), (r_score[i0], r_score[i1]))
                        d0, d1 = cute.arch.mul_packed_f32x2((d0, d1), (scale_softmax, scale_softmax))
                        r_ds[i0] = self.element_dtype(d0)
                        r_ds[i1] = self.element_dtype(d1)
                r_ds_store = thread_copy_r2s.retile(r_ds)
                assert t_rs_ds_local_tile.shape == r_ds_store.shape
                assert t_rs_ds_xchg_tile.shape == r_ds_store.shape
                if cute.arch.make_warp_uniform(mtx // Int32(self.H_TILE_CTA)) == cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()):
                    cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_local_tile)
                assert t_rs_ds_tile.shape == r_ds_store.shape
                cute.copy(tiled_copy_r2s, r_ds_store, t_rs_ds_tile)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(pds_ready_mbars)
                pds_state.advance()
            if tile_count > Int32(0):
                pipe_dq_done.consumer_wait(dq_done_state)
                _mbarrier_wait_acquire_cluster(loan_epi_safe_mbar, Int32(0))
                self._store_dq_epi_tma(
                    t_dq[0], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, tma_tensor_dq_epi, 0, token_idx, batch_idx, rank, mtx
                )
                self._store_dq_epi_tma(
                    t_dq[1], dq_tmem_load, rank_dq_coordinates, s_dq_epi, tma_atom_dq_epi, tma_tensor_dq_epi, 1, token_idx, batch_idx, rank, mtx
                )
                pipe_dq_done.consumer_release(dq_done_state)
                dq_done_state.advance()
            else:
                self._zero_dq(rank_dq_coordinates, mdQ, 0, token_idx, batch_idx, mtx)
                self._zero_dq(rank_dq_coordinates, mdQ, 1, token_idx, batch_idx, mtx)
        elif warp_idx < Int32(self.MMA_WARP):
            rtx = tidx - Int32(self.REDUCE_THREAD_BEGIN)
            dkv_wait = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            dkv_rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.MMA_DONE_STAGES)
            for loop_iter in cutlass.range(tile_count):
                tile_index = tile_count - Int32(1) - loop_iter
                dkv_wait, dkv_rel = self._drain_dkv(
                    t_dkv[0], t_dkv[1], mdKV_acc, mTopkIdxs, tile_index, topk, token_idx, batch_idx, rtx, rank, pipe_dkv_done, dkv_wait, dkv_rel
                )
        elif warp_idx == Int32(self.MMA_WARP):
            if is_leader_cta:
                s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                dp_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.SCORE_DONE_STAGES)
                kscore_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                round_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ROUND_STAGES)
                pds_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                dkv_acq = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                dkv_com = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.MMA_DONE_STAGES)
                dq_done_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                if tile_count > Int32(0):
                    _mbarrier_wait_acquire_cluster(stationary_ready_mbar, Int32(0))
                pipe_dq_done.producer_acquire(dq_done_prod)
                relay_phase = Int32(0)
                for loop_iter in cutlass.range(tile_count):
                    pipe_kscore.consumer_wait(kscore_cons)
                    s_prod = self._issue_score(score_tiled_mma, t_score, t_score_pp, score_q_fragment, score_k_fragment, pipe_s_done, s_prod)
                    if loop_iter == Int32(0):
                        _mbarrier_wait_acquire_cluster(stationary_ready_mbar + 1, Int32(0))
                    dp_prod = self._issue_score(dp_tiled_mma, t_dp, t_dp_pp, score_do_fragment, dp_k_fragment, pipe_dp_done, dp_prod)
                    pipe_kscore.consumer_release(kscore_cons)
                    kscore_cons.advance()
                    dq_acc = loop_iter != Int32(0)
                    round_cons, kscore_cons, dkv_acq, dkv_com, pds_cons = self._issue_grads(
                        dq_tiled_mma,
                        dkv_tiled_mma,
                        t_dq[0],
                        t_dq[1],
                        t_dkv[0],
                        t_dkv[1],
                        dq_kd_fragment_a,
                        dq_kd_fragment_b,
                        dq_ds_fragment,
                        round_fragments[0],
                        round_fragments[1],
                        round_fragments[2],
                        round_fragments[3],
                        p_fragments[0],
                        p_fragments[1],
                        ds_fragments[0],
                        ds_fragments[1],
                        dq_acc,
                        relay_phase,
                        relay_mbars,
                        ds_local_ready_mbar,
                        pipe_round,
                        round_cons,
                        pipe_kscore,
                        kscore_cons,
                        pipe_pds,
                        pds_cons,
                        pipe_dkv_done,
                        dkv_acq,
                        dkv_com,
                    )
                    pipe_pds.consumer_release(pds_cons)
                    pds_cons.advance()
                    relay_phase = Int32(1) - relay_phase
                if tile_count > Int32(0):
                    pipe_dq_done.producer_commit(dq_done_prod)
                    dq_done_prod.advance()
                    pipe_s_done.producer_tail(s_prod)
                    pipe_dp_done.producer_tail(dp_prod)
                    pipe_dkv_done.producer_tail(dkv_com)
                    pipe_dq_done.producer_tail(dq_done_prod)
        elif warp_idx == Int32(self.LOAD_WARP):
            if tile_count > Int32(0):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars, score_a_stage_bytes * self.K_CHUNKS)
                    cute.arch.mbarrier_arrive_and_expect_tx(stationary_tma_mbars + 1, score_a_stage_bytes * self.K_CHUNKS)
                cute.copy(tma_atom_q, t_q_gmem[None, rank, 0], t_q_smem[None, 0], tma_bar_ptr=stationary_tma_mbars)
                cute.copy(tma_atom_do, t_do_gmem[None, rank, 0], t_do_smem[None, 0], tma_bar_ptr=stationary_tma_mbars + 1)
                cute.arch.mbarrier_wait(stationary_tma_mbars, Int32(0))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(stationary_ready_mbar, Int32(0))
                cute.arch.mbarrier_wait(stationary_tma_mbars + 1, Int32(0))
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(stationary_ready_mbar + 1, Int32(0))
                for loop_iter in cutlass.range(tile_count):
                    tile_index = tile_count - Int32(1) - loop_iter
                    for flat_gen in cutlass.range_constexpr(self.ROUND_PANELS_PER_TILE):
                        if cutlass.const_expr(flat_gen < 2):
                            grad_round = 0
                            tensor_kind = 0
                        elif cutlass.const_expr(flat_gen < 4):
                            grad_round = 1
                            tensor_kind = 0
                        elif cutlass.const_expr(flat_gen < 6):
                            grad_round = 0
                            tensor_kind = 1
                        else:
                            grad_round = 1
                            tensor_kind = 1
                        h_half = flat_gen % 2
                        for k_half in cutlass.range_constexpr(2):
                            micro_gen = 2 * flat_gen + k_half
                            round_slot = micro_gen % self.ROUND_STAGES
                            source_h32 = 2 * h_half + k_half
                            round_acq = pipeline.PipelineState(
                                self.ROUND_STAGES,
                                loop_iter * Int32(self.ROUND_GENS_PER_TILE) + Int32(micro_gen),
                                Int32(round_slot),
                                Int32(1 ^ micro_gen // self.ROUND_STAGES & 1),
                            )
                            pipe_round.producer_acquire(round_acq)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(round_tma_mbars + round_slot, round_stage_bytes)
                            round_dst_raw = round_slot_raw[round_slot]
                            local_src_offset = 2 * self.ROUND_STAGE_ELEMENTS * (2 * grad_round + h_half) + self.ROUND_STAGE_ELEMENTS // 2 * k_half
                            if cutlass.const_expr(tensor_kind == 0):
                                if rank == Int32(h_half):
                                    with cute.arch.elect_one():
                                        _cpasync_bulk_s2cluster(
                                            stationary_do_raw + local_src_offset, round_dst_raw, round_tma_mbars + round_slot, round_stage_bytes // 2, rank
                                        )
                                        _cpasync_bulk_s2cluster(
                                            stationary_do_raw + local_src_offset + self.ROUND_STAGE_ELEMENTS,
                                            round_dst_raw + self.ROUND_STAGE_ELEMENTS // 2,
                                            round_tma_mbars + round_slot,
                                            round_stage_bytes // 2,
                                            rank,
                                        )
                                else:
                                    cute.copy(
                                        round_tma_atom_dot,
                                        t_dot_round_gmem[None, grad_round, source_h32],
                                        t_dot_round_smem[round_slot][None, 0],
                                        tma_bar_ptr=round_tma_mbars + round_slot,
                                    )
                            elif rank == Int32(h_half):
                                with cute.arch.elect_one():
                                    _cpasync_bulk_s2cluster(
                                        stationary_q_raw + local_src_offset, round_dst_raw, round_tma_mbars + round_slot, round_stage_bytes // 2, rank
                                    )
                                    _cpasync_bulk_s2cluster(
                                        stationary_q_raw + local_src_offset + self.ROUND_STAGE_ELEMENTS,
                                        round_dst_raw + self.ROUND_STAGE_ELEMENTS // 2,
                                        round_tma_mbars + round_slot,
                                        round_stage_bytes // 2,
                                        rank,
                                    )
                            else:
                                cute.copy(
                                    round_tma_atom_qt,
                                    t_qt_round_gmem[None, grad_round, source_h32],
                                    t_qt_round_smem[round_slot][None, 0],
                                    tma_bar_ptr=round_tma_mbars + round_slot,
                                )
                round_tail = pipeline.PipelineState(self.ROUND_STAGES, tile_count * Int32(self.ROUND_GENS_PER_TILE), Int32(0), Int32(1))
                pipe_round.producer_tail(round_tail)
        elif warp_idx == Int32(self.RELAY_WARP):
            relay_lane = tidx % Int32(32)
            if relay_lane == Int32(0):
                for loop_iter in cutlass.range(tile_count):
                    cute.arch.mbarrier_wait(p_ready_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[0], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                    else:
                        _cpasync_bulk_s2cluster(p_xchg_raw.iterator, p_block_raw_ptrs[1], landing_mbars, self.PDS_BLOCK_BYTES, peer_rank)
                    _mbarrier_wait_acquire_cluster(landing_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(relay_mbars, Int32(0))
                    cute.arch.mbarrier_wait(pds_ready_mbars, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(ds_local_ready_mbar, Int32(0))
                    cute.arch.mbarrier_arrive_and_expect_tx(landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_cta_rank_in_cluster=peer_rank)
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(ds_image_raw + Int32(2048), ds_block_raw_ptrs[0], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                    else:
                        _cpasync_bulk_s2cluster(ds_image_raw, ds_block_raw_ptrs[1], landing_mbars + 1, self.PDS_BLOCK_BYTES, peer_rank)
                    pds_com = pipeline.PipelineState(1, loop_iter, Int32(0), Int32(1) ^ loop_iter & Int32(1))
                    pipe_pds.producer_commit(pds_com)
                    _mbarrier_wait_acquire_cluster(landing_mbars + 1, loop_iter & Int32(1))
                    cute.arch.mbarrier_arrive(relay_mbars + 1, Int32(0))
                if tile_count > Int32(0):
                    pds_tail = pipeline.PipelineState(1, tile_count, Int32(0), Int32(1) ^ tile_count & Int32(1))
                    pipe_pds.producer_tail(pds_tail)
        elif warp_idx == Int32(self.COMMIT_WARP):
            commit_com = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ROUND_STAGES)
            w19_phase = [Int32(0), Int32(0), Int32(0), Int32(0)]
            for loop_iter in cutlass.range(tile_count):
                for micro_gen in cutlass.range_constexpr(self.ROUND_GENS_PER_TILE):
                    round_slot = micro_gen % self.ROUND_STAGES
                    cute.arch.mbarrier_wait(round_tma_mbars + round_slot, w19_phase[round_slot])
                    w19_phase[round_slot] = Int32(1) - w19_phase[round_slot]
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(commit_com)
                    commit_com.advance()
        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp_idx == Int32(self.MATH_WARP_BEGIN):
            cute.arch.dealloc_tmem(tmem_ptr, self.TMEM_COLUMNS, is_two_cta=True)

    @cute.jit
    def _issue_score(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Issue one score-side CG2 GEMM over four resident D128 chunks."""
        done_pipeline.producer_acquire(producer_state)
        if producer_state.index == Int32(0):
            self._issue_score_chunks(tiled_mma, accumulator_0, a_fragment, b_fragment)
        else:
            self._issue_score_chunks(tiled_mma, accumulator_1, a_fragment, b_fragment)
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_chunks(self, tiled_mma: cute.TiledMma, accumulator: cute.Tensor, a_fragment: cute.Tensor, b_fragment: cute.Tensor):
        """One full-K score GEMM into a single ping-pong accumulator."""
        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            for k_block in cutlass.range(0, k_blocks_per_chunk, unroll=4):
                cute.gemm(mma, accumulator, a_fragment[None, None, k_block, chunk], b_fragment[None, None, k_block, chunk], accumulator)
                mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_grads(
        self,
        dq_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        dq_kd_fragment_a: cute.Tensor,
        dq_kd_fragment_b: cute.Tensor,
        dq_ds_fragment: cute.Tensor,
        round_fragment_0: cute.Tensor,
        round_fragment_1: cute.Tensor,
        round_fragment_2: cute.Tensor,
        round_fragment_3: cute.Tensor,
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        dq_accumulate: cutlass.Boolean,
        relay_phase: Int32,
        relay_mbars: cute.Pointer,
        ds_local_ready_mbar: cute.Pointer,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dkv_commit_state: pipeline.PipelineState,
    ):
        """Issue the serial dV, dQ, then dK gradient chain for one tile."""
        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_0, p_fragment_0, 0, False)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_1, p_fragment_0, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_2, p_fragment_1, 0, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_3, p_fragment_1, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        dkv_done_pipeline.producer_acquire(dkv_acquire_state)
        dkv_acquire_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_0, p_fragment_0, 0, False)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_1, p_fragment_0, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_2, p_fragment_1, 0, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_3, p_fragment_1, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        pds_pipeline.consumer_wait(pds_consumer_state)
        _mbarrier_wait_acquire_cluster(ds_local_ready_mbar, relay_phase)
        kscore_consumer_state = self._issue_dq_rounds(
            dq_tiled_mma, t_dq_0, t_dq_1, dq_kd_fragment_a, dq_kd_fragment_b, dq_ds_fragment, dq_accumulate, kscore_pipeline, kscore_consumer_state
        )
        _mbarrier_wait_acquire_cluster(relay_mbars + 1, relay_phase)
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_0, ds_fragment_0, 0, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_1, ds_fragment_0, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_2, ds_fragment_1, 0, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_0, round_fragment_3, ds_fragment_1, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_0, ds_fragment_0, 0, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_1, ds_fragment_0, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_2, ds_fragment_1, 0, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass(dkv_tiled_mma, t_dkv_1, round_fragment_3, ds_fragment_1, 1, True)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_commit_state)
        dkv_commit_state.advance()
        return (round_consumer_state, kscore_consumer_state, dkv_acquire_state, dkv_commit_state, pds_consumer_state)

    @cute.kernel
    def convert_canonical(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen: Int32):
        """Decode the baseline reducer's within-panel column scramble.

        The reducer stores each thread's register-gathered FP32x4 quad at
        group index dp_idx//4 of its 128-column panel (the production
        store_dKV addressing), so the workspace column order inside every
        panel is the baseline permutation; this override replaces the
        canonical copy with the baseline convert's dim_idx decode.  The
        scramble is panel-base invariant, so the same formula covers our
        2*round+rank panel bases.
        """
        assert self.same_hdim_kv
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        if seq_id < seqlen:
            acc_row = mdKV_acc[None, seq_id, (0, batch_idx)]
            out_row = mdKV[None, seq_id, (0, batch_idx)]
            tile_acc_row = cute.flat_divide(acc_row, (64,))
            tile_acc_row = cute.flat_divide(tile_acc_row, (32,))
            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    scrambled = tile_acc_row[tidx, j, i]
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    out_row[dim_idx] = self.element_dtype(scrambled)

    @cute.jit
    def _drain_dkv(
        self,
        t_dkv_0: cute.Tensor,
        t_dkv_1: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        rtx: Int32,
        rank: Int32,
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):
        """Drain both rank-owned dKV slots and pace global atomics."""
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        t_dkv_core_0 = t_dkv_0[(None, None), 0, 0]
        t_dkv_core_1 = t_dkv_1[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype)
        tiled_t2r_0 = tcgen05.make_tmem_copy(tmem_load_atom, t_dkv_core_0)
        thread_t2r_0 = tiled_t2r_0.get_slice(dp_idx)
        tiled_t2r_1 = tcgen05.make_tmem_copy(tmem_load_atom, t_dkv_core_1)
        thread_t2r_1 = tiled_t2r_1.get_slice(dp_idx)
        c_dkv = cute.make_identity_tensor((self.D_TILE_CTA, self.N_TILE))
        thread_coordinates = self.split_wg(thread_t2r_0.partition_D(c_dkv), 2, wg_idx)
        thread_source_0 = self.split_wg(thread_t2r_0.partition_S(t_dkv_core_0), 2, wg_idx)
        thread_source_1 = self.split_wg(thread_t2r_1.partition_S(t_dkv_core_1), 2, wg_idx)
        thread_values_0 = cute.make_rmem_tensor(thread_coordinates.shape, self.acc_dtype)
        thread_values_1 = cute.make_rmem_tensor(thread_coordinates.shape, self.acc_dtype)
        tile_base = tile_index * Int32(self.N_TILE)
        r_topk = cute.make_rmem_tensor((8,), cutlass.Int32)
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            local_row = Int32(cute.get(thread_coordinates[coord_base], mode=[1]))
            global_row = tile_base + local_row
            if global_row < topk:
                r_topk[i] = mTopkIdxs[global_row, (token_idx, batch_idx)]
            else:
                r_topk[i] = Int32(-1)
        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        assert cute.size(thread_values_0) == self.N_TILE // 2
        reduce_cohort = rank * Int32(2) + wg_idx
        _nanosleep_u32(reduce_cohort * Int32(self.REDUCE_DEPHASE_NS))
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            rdkv_frg_0 = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdkv_frg_0[0] = thread_values_0[coord_base]
            rdkv_frg_0[1] = thread_values_0[coord_base + 2]
            rdkv_frg_0[2] = thread_values_0[coord_base + 16]
            rdkv_frg_0[3] = thread_values_0[coord_base + 18]
            kv_index = r_topk[i]
            if kv_index >= Int32(0):
                dkv_row = mdKV_acc[None, kv_index, (0, batch_idx)]
                tile_row = cute.flat_divide(dkv_row, (128,))
                tile_row_0 = tile_row[None, sub_tile_idx_0]
                tile_row_0 = cute.flat_divide(tile_row_0, (4,))
                target_frg_0 = tile_row_0[None, dp_idx // 4]
                cute.arch.atomic_add(target_frg_0.iterator.llvm_ptr, rdkv_frg_0.load())
            _nanosleep_u32(Int32(self.REDUCE_PACE_NS))
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        _nanosleep_u32(reduce_cohort * Int32(self.REDUCE_DEPHASE_NS))
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            rdkv_frg_1 = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdkv_frg_1[0] = thread_values_1[coord_base]
            rdkv_frg_1[1] = thread_values_1[coord_base + 2]
            rdkv_frg_1[2] = thread_values_1[coord_base + 16]
            rdkv_frg_1[3] = thread_values_1[coord_base + 18]
            kv_index = r_topk[i]
            if kv_index >= Int32(0):
                dkv_row = mdKV_acc[None, kv_index, (0, batch_idx)]
                tile_row = cute.flat_divide(dkv_row, (128,))
                tile_row_1 = tile_row[None, sub_tile_idx_1]
                tile_row_1 = cute.flat_divide(tile_row_1, (4,))
                target_frg_1 = tile_row_1[None, dp_idx // 4]
                cute.arch.atomic_add(target_frg_1.iterator.llvm_ptr, rdkv_frg_1.load())
            _nanosleep_u32(Int32(self.REDUCE_PACE_NS))
        return (wait_state, release_state)
