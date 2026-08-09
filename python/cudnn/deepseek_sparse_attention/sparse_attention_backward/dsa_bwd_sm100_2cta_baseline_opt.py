"""baseline_opt: the production 1-CTA baseline plus three per-launch knives.

Fork of ``FlashAttentionDSABackwardSm100`` (dsa_bwd_sm100.py) that back-ports
the three fixed-cost optimizations discovered in the 2-CTA campaign.  Numerics
are bit-identical to baseline; only issue order and SMEM staging change.

Knives (each env-gated, read at import time, default ON):

  DSA_BOPT_EPI (knife 1, dQ epilogue batching)
      Baseline stores the four D128 dQ panels serially through a 16 KiB
      1-stage staging slot carved from the dead sK allocation: each panel
      pays T2R -> bf16 -> SMEM -> TMA and the next panel's acquire waits the
      previous TMA's completion.  sK is 64 KiB in the D512 shape -- exactly
      4 x 16 KiB -- so this knife stages ALL four panels first and then
      issues the four TMA stores back-to-back under one commit group.

  DSA_BOPT_DQ_EARLY (knife 2, early dQ generation commit)
      Baseline commits the mma_compute_dQ pipeline after the whole KV loop
      and after the final t2r_dKV4 rendezvous, so the compute warps' dQ
      epilogue is fully serialized behind the tail dKV part-2 GEMMs and the
      reduce drain.  The dQ TMEM generation is actually complete when the
      last tile's dQ GEMMs are issued (the tcgen05 commit tracks their
      completion, which also covers their sK operand reads, making the
      sdQ-over-sK staging write safe).  This knife issues the commit right
      after the last tile's dQ (and dQ4) GEMMs, overlapping the epilogue
      with the tail dKV GEMMs and the reduce atomics.

  DSA_BOPT_SPLIT_QDO (knife 3, split stationary readiness)
      Baseline loads Q and dO through a single 128 KiB-tx barrier, so the
      first S GEMM waits both.  This knife gives Q its own barrier: S waits
      only Q (64 KiB); the first dP waits the dO barrier.  Expected ~null on
      this kernel (the first K gather is the slower prologue leg) -- kept
      for the controlled experiment.

Harness: benchmark/dsa/sweep_topk_2cta.py loads this module via
  --impl baseline_opt --class-name FlashAttentionDSABackwardSm100BaselineOpt
(the candidate-leg trace arguments are accepted and ignored).  Env flags are
baked at import: use a fresh process per flag combination.
"""

import math
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.typing import Float32, Int32

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

_BOPT_EPI = os.environ.get("DSA_BOPT_EPI", "1") == "1"
_BOPT_DQ_EARLY = os.environ.get("DSA_BOPT_DQ_EARLY", "1") == "1"
_BOPT_SPLIT_QDO = os.environ.get("DSA_BOPT_SPLIT_QDO", "1") == "1"


class FlashAttentionDSABackwardSm100BaselineOpt(FlashAttentionDSABackwardSm100):
    """Baseline with the three per-launch fixed-cost knives applied."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        super().__init__(head_dim, head_dim_v, block_tile, max_topk)
        self.bopt_epi = _BOPT_EPI
        self.bopt_dq_early = _BOPT_DQ_EARLY
        self.bopt_split_qdo = _BOPT_SPLIT_QDO
        # Knife 1 stages all four dQ panels in the dead sK allocation.
        self.dq_epi_stages = 4 if _BOPT_EPI else 1

    def _setup_attributes(self):
        super()._setup_attributes()
        self.load_mma_Q_stage = 1
        self.mma_compute_dealloc_stage = 1

    # ------------------------------------------------------------------
    # Pipeline factories: knife 3 splits the Q/dO transaction bytes.
    # ------------------------------------------------------------------
    def make_and_init_load_mma_QdO_pipeline(self, load_mma_QdO_mbar_ptr):
        load_mma_QdO_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_QdO_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        tx_count = self.tma_copy_dO_bytes if self.bopt_split_qdo else self.tma_copy_QdO_bytes
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_QdO_mbar_ptr,
            num_stages=self.load_mma_QdO_stage,
            producer_group=load_mma_QdO_producer_group,
            consumer_group=load_mma_QdO_consumer_group,
            tx_count=tx_count,
            defer_sync=True,
        )

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dealloc_pipeline(self, mma_compute_dealloc_mbar_ptr):
        """Knife 2 dealloc gate (review-panel fix).

        The early dQ commit alone would leave compute's dealloc_tmem
        unordered against the tail part-2 dKV UMMA writes and the reduce
        dKV3 T2R.  This 1-stage pipeline is committed by the MMA warp at
        the ORIGINAL (baseline) commit position -- after the loop and the
        same-hdim t2r_dKV4 rendezvous -- and waited by the compute warps
        immediately before dealloc_tmem, restoring baseline's provable
        ordering while keeping the epilogue overlap.
        """
        mma_compute_dealloc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.mma_warp_id]))
        mma_compute_dealloc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dealloc_mbar_ptr,
            num_stages=self.mma_compute_dealloc_stage,
            producer_group=mma_compute_dealloc_producer_group,
            consumer_group=mma_compute_dealloc_consumer_group,
            defer_sync=True,
        )

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
        """Baseline __call__ with sweep-compatible (ignored) trace arguments."""

        _ = trace_buffer
        _ = trace_token_idx
        _ = trace_batch_idx

        # [M, H, D] -> [H, D, (M, 1)]
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout((mQ.shape[1], mQ.shape[2], (mQ.shape[0], 1)), stride=(mQ.stride[1], mQ.stride[2], (mQ.stride[0], 0)))
        )

        # [N, D] -> [N, D, (1, 1)]
        mKV = cute.make_tensor(mKV.iterator, cute.make_layout((mKV.shape[0], mKV.shape[1], (1, 1)), stride=(mKV.stride[0], mKV.stride[1], (0, 0))))

        # [M, H, Dv] -> [H, Dv, (M, 1)]
        mOut = cute.make_tensor(
            mOut.iterator, cute.make_layout((mOut.shape[1], mOut.shape[2], (mOut.shape[0], 1)), stride=(mOut.stride[1], mOut.stride[2], (mOut.stride[0], 0)))
        )

        # [M, H, Dv] -> [H, Dv, (M, 1)]
        mdO = cute.make_tensor(
            mdO.iterator, cute.make_layout((mdO.shape[1], mdO.shape[2], (mdO.shape[0], 1)), stride=(mdO.stride[1], mdO.stride[2], (mdO.stride[0], 0)))
        )
        # [M, H, D] -> [D, H, (M, 1)]
        mdQ = cute.make_tensor(
            mdQ.iterator, cute.make_layout((mdQ.shape[2], mdQ.shape[1], (mdQ.shape[0], 1)), stride=(mdQ.stride[2], mdQ.stride[1], (mdQ.stride[0], 0)))
        )
        # [N, D] -> [D, N, (1, 1)]
        mdKV = cute.make_tensor(mdKV.iterator, cute.make_layout((mdKV.shape[1], mdKV.shape[0], (1, 1)), stride=(mdKV.stride[1], mdKV.stride[0], (0, 0))))

        # [M, H] -> [H, (M, 1)]
        mLSE = cute.make_tensor(mLSE.iterator, cute.make_layout((mLSE.shape[1], (mLSE.shape[0], 1)), stride=(mLSE.stride[1], (mLSE.stride[0], 0))))

        # [H] -> [H, (1, 1)]
        mdSink = cute.make_tensor(mdSink.iterator, cute.make_layout((mdSink.shape[0], (1, 1)), stride=(1, (0, 0))))
        mAttnSink = cute.make_tensor(mAttnSink.iterator, mdSink.layout)

        # [M, TopK] -> [TopK, (M, 1)]
        mTopkIdxs = cute.make_tensor(
            mTopkIdxs.iterator, cute.make_layout((mTopkIdxs.shape[1], (mTopkIdxs.shape[0], 1)), stride=(mTopkIdxs.stride[1], (mTopkIdxs.stride[0], 0)))
        )
        # [M] -> [M, (1, 1)] when provided; None means non-compact (use full topk, -1 entries in topk_idxs)
        if cutlass.const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(mTopkLength.iterator, cute.make_layout((mTopkLength.shape[0], (1, 1)), stride=(mTopkLength.stride[0], (0, 0))))

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE

        # S = Q @ KV
        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cta_group, self.QK_mma_tiler[:2]
        )

        # dP = dO @ KV
        dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, cta_group, self.dOV_mma_tiler[:2]
        )

        # dKV = dO^T @ P
        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cta_group, self.dOP_mma_tiler[:2]
        )
        # dKV = Q^T @ dS
        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cta_group, self.QdS_mma_tiler[:2]
        )
        # dQ = KV @ dS^T
        KdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.acc_dtype, cta_group, self.KdS_mma_tiler[:2]
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            # dKV4: Q^T[512:575] @ dS -> (64, 64) output
            dKV4_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.acc_dtype, cta_group, self.dKV4_mma_tiler[:2]
            )
            # dQ4: K[512:575] @ dS^T -> (64, 64) output
            dQ4_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype, self.element_dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.acc_dtype, cta_group, self.dQ4_mma_tiler[:2]
            )
        else:
            dKV4_tiled_mma = None
            dQ4_tiled_mma = None

        self.cluster_layout_vmnk = cute.make_layout(((1), (1, 1, 1)), stride=((0), (0, 0, 0)))

        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_QdO_stage)
        K_smem_layout_staged = sm100_utils.make_smem_layout_b(QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_K_stage)
        dO_smem_layout_staged = sm100_utils.make_smem_layout_a(dOV_tiled_mma, self.dOV_mma_tiler, self.element_dtype, self.load_mma_QdO_stage)
        if cutlass.const_expr(not self.same_hdim_kv):
            V_smem_layout_staged = sm100_utils.make_smem_layout_b(dOV_tiled_mma, self.dOV_mma_tiler, self.element_dtype, self.load_mma_K_stage)
        else:
            V_smem_layout_staged = K_smem_layout_staged

        dOT_smem_layout_staged = sm100_utils.make_smem_layout_a(dOP_tiled_mma, self.dOP_cta_tiler, self.element_dtype, self.load_mma_QdO_stage)
        P_smem_layout_staged = sm100_utils.make_smem_layout_b(dOP_tiled_mma, self.dOP_mma_tiler, self.element_dtype, self.compute_mma_P_stage)
        P_smem_layout_store_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype, utils.LayoutEnum.COL_MAJOR, self.QK_mma_tiler[:2], self.load_mma_K_stage
        )
        K_smem_layout_staged_2 = sm100_utils.make_smem_layout_a(KdS_tiled_mma, self.KdS_cta_tiler, self.element_dtype, self.load_mma_K_stage)
        if cutlass.const_expr(not self.same_hdim_kv):
            # Tail view: partition sK with 64-wide blocks, giving head_dim/64 sub-tiles
            K_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
                dQ4_tiled_mma, (self.head_dim, self.block_tile, self.block_tile), self.element_dtype, self.load_mma_K_stage
            )
        else:
            K_tail_smem_layout_staged = None
        dST_smem_layout_staged = sm100_utils.make_smem_layout_b(KdS_tiled_mma, self.KdS_mma_tiler, self.element_dtype, self.compute_mma_dS_stage)
        QT_smem_layout_staged = sm100_utils.make_smem_layout_a(QdS_tiled_mma, self.QdS_cta_tiler, self.element_dtype, self.load_mma_QdO_stage)
        if cutlass.const_expr(not self.same_hdim_kv):
            # Tail view: partition sQ with 64-wide blocks
            QT_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
                dKV4_tiled_mma, (self.head_dim, self.block_tile, self.block_tile), self.element_dtype, self.load_mma_QdO_stage
            )
        else:
            QT_tail_smem_layout_staged = None
        dS_smem_layout_staged = sm100_utils.make_smem_layout_b(QdS_tiled_mma, self.QdS_mma_tiler, self.element_dtype, self.compute_mma_dS_stage)
        dS_smem_layout_store_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype, utils.LayoutEnum.COL_MAJOR, self.dOV_mma_tiler[:2], self.load_mma_K_stage
        )

        # Knife 1: stage all four dQ panels in the dead sK allocation.
        dQ_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype, utils.LayoutEnum.from_tensor(mdQ), (self.KdS_mma_tiler[0], self.KdS_mma_tiler[1]), self.dq_epi_stages
        )
        # The staging aliases sK, so all stages must fit inside it.
        assert cute.cosize(dQ_smem_layout_staged) <= cute.cosize(K_smem_layout_staged)

        dKV_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype, utils.LayoutEnum.from_tensor(mdKV), (self.dOP_mma_tiler[0], self.dOP_mma_tiler[1] // 2), self.mma_reduce_dKV_stage
        )

        LSE_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_sum_OdO_stage))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mQ, Q_smem_layout, self.QK_mma_tiler, QK_tiled_mma, self.cluster_layout_vmnk.shape
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mdO, dO_smem_layout, self.dOV_mma_tiler, dOV_tiled_mma, self.cluster_layout_vmnk.shape
        )

        dQ_smem_layout = cute.select(dQ_smem_layout_staged, mode=[0, 1])
        tma_atom_dQ, tma_tensor_dQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ,
            dQ_smem_layout,
            (self.KdS_mma_tiler[0], self.KdS_mma_tiler[1]),
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            dQ4_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.element_dtype, utils.LayoutEnum.from_tensor(mdQ), (self.dQ4_mma_tiler[0], self.dQ4_mma_tiler[1]), self.mma_compute_dQ_stage
            )
            dQ4_smem_layout = cute.select(dQ4_smem_layout_staged, mode=[0, 1])
            tma_atom_dQ_64, tma_tensor_dQ_64 = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mdQ,
                dQ4_smem_layout,
                (self.dQ4_mma_tiler[0], self.dQ4_mma_tiler[1]),
            )
        else:
            dQ4_smem_layout_staged = None
            tma_atom_dQ_64 = None
            tma_tensor_dQ_64 = None

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(self.element_dtype, dO_smem_layout)
        self.tma_copy_QdO_bytes = self.tma_copy_Q_bytes + self.tma_copy_dO_bytes

        _max_smem_bytes = 227 * 1024

        @cute.struct
        class SharedStorage:
            load_mma_QdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_QdO_stage * 2]
            # Knife 3: dedicated Q-readiness barrier (always allocated, 16B).
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_Q_stage * 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]
            load_compute_LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_LSE_stage * 2]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_sum_OdO_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            mma_compute_dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dQ_stage * 2]
            # Knife 2: dealloc gate (always allocated, 16B).
            mma_compute_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dealloc_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_P_stage * 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_reduce_dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_reduce_dKV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(K_smem_layout_staged)], self.buffer_align_bytes]
            sdO: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(dO_smem_layout_staged)], self.buffer_align_bytes]
            sP: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(P_smem_layout_staged)], self.non_tma_align_bytes]
            sdS: cute.struct.Align[cute.struct.MemRange[self.element_dtype, cute.cosize(dS_smem_layout_staged)], self.non_tma_align_bytes]
            sLSE: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)], self.non_tma_align_bytes]
            sSum_OdO: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)], self.non_tma_align_bytes]

        assert (
            SharedStorage.size_in_bytes() <= _max_smem_bytes
        ), f"SharedStorage ({SharedStorage.size_in_bytes()} bytes) exceeds {_max_smem_bytes} bytes (227KB)"
        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, mdKV_acc = self.get_workspace_tensor(
            problem_shape,
            workspace_LSE_OdO,
            workspace_dKV,
            mQ.shape[2][0],
            mKV.shape[0],
            self.acc_dtype,
        )
        mdKV_acc = cute.make_tensor(mdKV_acc.iterator, mdKV.layout)

        # ============ Sum OdO ============
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)

        self.sum_OdO(
            mOut,
            mdO,
            sum_OdO,
            mLSE,
            mAttnSink,
            scaled_LSE,
            sum_OdO_scale,
            LSE_scale,
            problem_shape,
        ).launch(
            grid=sum_OdO_grid,
            block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        num_head_blocks = cute.ceil_div(problem_shape[3][0], self.block_tile)
        bwd_grid = (problem_shape[0], num_head_blocks, problem_shape[3][1])
        self.bwd(
            problem_shape,
            QK_tiled_mma,
            dOV_tiled_mma,
            dOP_tiled_mma,
            QdS_tiled_mma,
            KdS_tiled_mma,
            dKV4_tiled_mma,
            dQ4_tiled_mma,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dQ,
            tma_tensor_dQ,
            tma_atom_dQ_64,
            tma_tensor_dQ_64,
            mKV,
            mdQ,
            mdKV_acc,
            mdSink,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            scaled_LSE,
            sum_OdO,
            softmax_scale,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            dO_smem_layout_staged,
            V_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
            P_smem_layout_store_staged,
            K_smem_layout_staged_2,
            K_tail_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            QT_tail_smem_layout_staged,
            dS_smem_layout_staged,
            dS_smem_layout_store_staged,
            dKV_smem_layout_staged,
            dQ_smem_layout_staged,
            dQ4_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.block_seq = 4 if self.max_topk == 2048 else 32
        self.num_threads_D_convert = 32
        self.num_threads_seq = 4 if self.max_topk == 2048 else self.block_seq
        self.convert_elem_per_load = 4

        convert_grid_x = (mKV.shape[0] + self.block_seq - 1) // self.block_seq
        convert_grid = [
            convert_grid_x,
            1,
            1,
        ]
        convert_block = [self.num_threads_D_convert, self.num_threads_seq, 1]
        self.convert(
            mdKV_acc,
            mdKV,
            mKV.shape[0],
        ).launch(
            grid=convert_grid,
            block=convert_block,
            stream=stream,
        )

        if cutlass.const_expr(self.same_hdim_kv):
            dSink_grid = (
                cute.ceil_div(problem_shape[0], self.dSink_block_q),
                problem_shape[3][0],
                problem_shape[3][1],
            )
            self.sum_dSink(
                sum_OdO,
                scaled_LSE,
                mAttnSink,
                mdSink,
                problem_shape,
            ).launch(
                grid=dSink_grid,
                block=[self.dSink_num_threads, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def bwd(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: Optional[cute.CopyAtom],
        tma_tensor_dQ_64: Optional[cute.Tensor],
        mKV: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mdSink: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        scale_softmax: Float32 | float,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_store_staged: cute.ComposedLayout,
        K_smem_layout_staged_2: cute.ComposedLayout,
        K_tail_smem_layout_staged: Optional[cute.ComposedLayout],
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        QT_tail_smem_layout_staged: Optional[cute.ComposedLayout],
        dS_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_store_staged: cute.ComposedLayout,
        dKV_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        dQ4_smem_layout_staged: Optional[cute.ComposedLayout],
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
    ):
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        tidx, _, batch_idx = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        max_seqlen_q, max_seqlen_kv, head_dim, (num_heads, batch_size) = problem_shape

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dQ)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_QdO_pipeline = self.make_and_init_load_mma_QdO_pipeline(
            storage.load_mma_QdO_mbar_ptr.data_ptr(),
        )
        load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(
            storage.load_mma_Q_mbar_ptr.data_ptr(),
        )
        load_mma_K_pipeline = self.make_and_init_load_mma_K_pipeline(
            storage.load_mma_K_mbar_ptr.data_ptr(),
        )
        load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(
            storage.load_compute_LSE_mbar_ptr.data_ptr(),
        )
        load_compute_sum_OdO_pipeline = self.make_and_init_load_compute_sum_OdO_pipeline(
            storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr(),
        )
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr.data_ptr(),
        )
        mma_compute_dQ_pipeline = self.make_and_init_mma_compute_dQ_pipeline(
            storage.mma_compute_dQ_mbar_ptr.data_ptr(),
        )
        mma_compute_dealloc_pipeline = self.make_and_init_mma_compute_dealloc_pipeline(
            storage.mma_compute_dealloc_mbar_ptr.data_ptr(),
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr(),
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr(),
        )
        mma_reduce_dKV_pipeline = self.make_and_init_mma_reduce_dKV_pipeline(
            storage.mma_reduce_dKV_mbar_ptr.data_ptr(),
        )
        compute_tmastore_dQ_pipeline = self.make_and_init_compute_tmastore_dQ_pipeline()

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        pipeline.pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sK.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)
        sP_store = storage.sP.get_tensor(P_smem_layout_store_staged.outer, swizzle=P_smem_layout_store_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        sdS = storage.sdS.get_tensor(dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner)
        sdS_store = storage.sdS.get_tensor(dS_smem_layout_store_staged.outer, swizzle=dS_smem_layout_store_staged.inner)
        # reuse sK
        sdQ_ptr = cute.recast_ptr(sK.iterator, dQ_smem_layout_staged.inner)
        sdQ = cute.make_tensor(sdQ_ptr, dQ_smem_layout_staged.outer)

        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)

        sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
        sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)

        sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
        sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)

        sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
        sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

        sK_2_ptr = cute.recast_ptr(sK.iterator, K_smem_layout_staged_2.inner)
        sK_2 = cute.make_tensor(sK_2_ptr, K_smem_layout_staged_2.outer)

        if cutlass.const_expr(not self.same_hdim_kv):
            # sK_tail: view sK storage with 64-wide partitioning, access block 8 (cols 512:575)
            # K_tail_smem_layout_staged partitions head_dim=576 into 64-wide blocks → 9 blocks
            sK_tail_ptr = cute.recast_ptr(sK.iterator, K_tail_smem_layout_staged.inner)
            sK_tail_full = cute.make_tensor(sK_tail_ptr, K_tail_smem_layout_staged.outer)
            sK_tail = sK_tail_full[None, 8, None, None]  # block 8 = cols 512:575

            # sQT_tail: view sQ storage with 64-wide partitioning, access block 8
            sQT_tail_ptr = cute.recast_ptr(sQ.iterator, QT_tail_smem_layout_staged.inner)
            sQT_tail_full = cute.make_tensor(sQT_tail_ptr, QT_tail_smem_layout_staged.outer)
            sQT_tail = sQT_tail_full[None, 8, None, None]  # block 8 = rows 512:575

            # sdQ4: reuse sK for the 64×64 dQ4 epilogue
            sdQ4_ptr = cute.recast_ptr(sK.iterator, dQ4_smem_layout_staged.inner)
            sdQ4 = cute.make_tensor(sdQ4_ptr, dQ4_smem_layout_staged.outer)

        pipeline.pipeline_init_wait()

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = mTopkIdxs.shape[0]

        tile_count = cute.ceil_div(topk, self.block_tile)

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                QK_tiled_mma,
                dOV_tiled_mma,
                tma_atom_Q,
                tma_tensor_Q,
                tma_atom_dO,
                tma_tensor_dO,
                mLSE,
                mSum_OdO,
                sQ,
                sdO,
                sLSE,
                sSum_OdO,
                (load_mma_QdO_pipeline, load_mma_Q_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline),
            )

        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4 = self.get_tmem_tensor(
                QK_tiled_mma, dOV_tiled_mma, QdS_tiled_mma, KdS_tiled_mma, dKV4_tiled_mma, dQ4_tiled_mma, tmem_ptr_base
            )

            # (MMA, MMA_M, MMA_K, STAGE)
            tSrQ = QK_tiled_mma.make_fragment_A(sQ)
            # (MMA, MMA_N, MMA_K, STAGE)
            tSrK = QK_tiled_mma.make_fragment_B(sK)

            tdKVrQT = QdS_tiled_mma.make_fragment_A(sQT)
            tdKVrdS = QdS_tiled_mma.make_fragment_B(sdS)
            # Inelegant, but I don't know how to handle the correct modes for cute.gemm
            tdKVrQT_shape = (tdKVrQT.shape[0], 1, tdKVrQT.shape[1], tdKVrQT.shape[2], tdKVrQT.shape[3])
            tdKVrQT_stride = (tdKVrQT.stride[0], 0, tdKVrQT.stride[1], tdKVrQT.stride[2], tdKVrQT.stride[3])
            tdKVrQT = cute.make_tensor(tdKVrQT.iterator, cute.make_layout(tdKVrQT_shape, stride=tdKVrQT_stride))

            tdPrdO = dOV_tiled_mma.make_fragment_A(sdO)
            tdPrV = dOV_tiled_mma.make_fragment_B(sV)

            tdQrK = KdS_tiled_mma.make_fragment_A(sK_2)
            tdQrdST = KdS_tiled_mma.make_fragment_B(sdST)

            tdQrK_shape = (tdQrK.shape[0], 1, tdQrK.shape[1], tdQrK.shape[2], tdQrK.shape[3])
            tdQrK_stride = (tdQrK.stride[0], 0, tdQrK.stride[1], tdQrK.stride[2], tdQrK.stride[3])
            tdQrK = cute.make_tensor(tdQrK.iterator, cute.make_layout(tdQrK_shape, stride=tdQrK_stride))

            tdKVrdOT = dOP_tiled_mma.make_fragment_A(sdOT)
            tdKVrP = dOP_tiled_mma.make_fragment_B(sP)
            tdKVrdOT_shape = (tdKVrdOT.shape[0], 1, tdKVrdOT.shape[1], tdKVrdOT.shape[2], tdKVrdOT.shape[3])
            tdKVrdOT_stride = (tdKVrdOT.stride[0], 0, tdKVrdOT.stride[1], tdKVrdOT.stride[2], tdKVrdOT.stride[3])
            tdKVrdOT = cute.make_tensor(tdKVrdOT.iterator, cute.make_layout(tdKVrdOT_shape, stride=tdKVrdOT_stride))

            if cutlass.const_expr(not self.same_hdim_kv):
                # dQ4 fragment: sK_tail (64-wide, single M-block) @ dS^T
                # sK_tail has 3 modes after slicing: (tile, K_blocks, stage)
                # make_fragment_A returns 3 modes: (MMA, MMA_K, STAGE)
                # Reshape to 5 modes: (MMA, 1_dummy, 1_M_block, MMA_K, STAGE)
                tdQrK_tail = dQ4_tiled_mma.make_fragment_A(sK_tail)
                tdQrK_tail_shape = (tdQrK_tail.shape[0], 1, 1, tdQrK_tail.shape[1], tdQrK_tail.shape[2])
                tdQrK_tail_stride = (tdQrK_tail.stride[0], 0, 0, tdQrK_tail.stride[1], tdQrK_tail.stride[2])
                tdQrK_tail = cute.make_tensor(tdQrK_tail.iterator, cute.make_layout(tdQrK_tail_shape, stride=tdQrK_tail_stride))

                # dKV4 fragment: sQT_tail (64-wide, single M-block) @ dS
                tdKVrQT_tail = dKV4_tiled_mma.make_fragment_A(sQT_tail)
                tdKVrQT_tail_shape = (tdKVrQT_tail.shape[0], 1, 1, tdKVrQT_tail.shape[1], tdKVrQT_tail.shape[2])
                tdKVrQT_tail_stride = (tdKVrQT_tail.stride[0], 0, 0, tdKVrQT_tail.stride[1], tdKVrQT_tail.stride[2])
                tdKVrQT_tail = cute.make_tensor(tdKVrQT_tail.iterator, cute.make_layout(tdKVrQT_tail_shape, stride=tdKVrQT_tail_stride))

                tdKVrdS_4 = dKV4_tiled_mma.make_fragment_B(sdS)
            else:
                tdQrK_tail = None
                tdKVrQT_tail = None
                tdKVrdS_4 = None

            self.mma(
                QK_tiled_mma,
                dOV_tiled_mma,
                dOP_tiled_mma,
                QdS_tiled_mma,
                KdS_tiled_mma,
                dKV4_tiled_mma,
                dQ4_tiled_mma,
                tSrQ,
                tSrK,
                tdPrdO,
                tdPrV,
                tdKVrdOT,
                tdKVrP,
                tdQrK,
                tdQrdST,
                tdKVrQT,
                tdKVrdS,
                tdQrK_tail,
                tdKVrQT_tail,
                tdKVrdS_4,
                tStS,
                tdPtdP,
                (tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4),
                (tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4),
                tile_count,
                sdS,
                (
                    load_mma_QdO_pipeline,
                    load_mma_Q_pipeline,
                    load_mma_K_pipeline,
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    mma_compute_dQ_pipeline,
                    mma_compute_dealloc_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_reduce_dKV_pipeline,
                ),
            )

        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4 = self.get_tmem_tensor(
                QK_tiled_mma, dOV_tiled_mma, QdS_tiled_mma, KdS_tiled_mma, dKV4_tiled_mma, dQ4_tiled_mma, tmem_ptr_base
            )

            self.compute(
                tma_atom_dQ,
                tma_tensor_dQ,
                tma_atom_dQ_64,
                tma_tensor_dQ_64,
                dQ4_tiled_mma,
                tStS,
                tdPtdP,
                (tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4),
                sLSE,
                sSum_OdO,
                sP_store,
                sdS,
                sdS_store,
                sdQ,
                sdQ4 if not self.same_hdim_kv else None,
                scale_softmax,
                tile_count,
                (
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    load_compute_LSE_pipeline,
                    load_compute_sum_OdO_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_compute_dQ_pipeline,
                    compute_tmastore_dQ_pipeline,
                ),
            )

            if cutlass.const_expr(self.bopt_dq_early):
                # Knife 2 dealloc gate: dealloc_tmem must not race the tail
                # part-2 dKV UMMA writes / reduce dKV3 T2R.
                mma_compute_dealloc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_compute_dealloc_stage
                )
                mma_compute_dealloc_pipeline.consumer_wait(mma_compute_dealloc_consumer_state)
                mma_compute_dealloc_pipeline.consumer_release(mma_compute_dealloc_consumer_state)
                mma_compute_dealloc_consumer_state.advance()

            if warp_idx == self.compute_warp_id[0]:
                cute.arch.dealloc_tmem(tmem_ptr_base, self.num_tmem_alloc_cols)

        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            tStS, tdPtdP, tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdKVtdKV4, tdQtdQ4 = self.get_tmem_tensor(
                QK_tiled_mma, dOV_tiled_mma, QdS_tiled_mma, KdS_tiled_mma, dKV4_tiled_mma, dQ4_tiled_mma, tmem_ptr_base
            )

            self.reduce_dKV(
                (tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4),
                mdKV_acc,
                mTopkIdxs,
                max_seqlen_kv,
                tile_count,
                topk,
                mma_reduce_dKV_pipeline,
            )

        elif warp_idx in self.load_KV_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load_KV)
            self.load_KV(
                mKV,
                mTopkIdxs,
                sK,
                tile_count,
                topk,
                load_mma_K_pipeline,
                mTopkLength,
            )

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

    @cute.jit
    def load(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        pipelines,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        local_tidx = tidx % self.threads_per_warp

        load_mma_QdO_pipeline, load_mma_Q_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline = pipelines

        load_mma_QdO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_QdO_stage)
        load_mma_Q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_Q_stage)
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage)

        # (bM, bK, RestM, RestK, (M, B))
        gQ = cute.local_tile(tma_tensor_Q, cute.select(self.QK_mma_tiler, mode=[0, 2]), (None, None, (token_idx, batch_idx)))
        gdO = cute.local_tile(tma_tensor_dO, cute.select(self.dOV_mma_tiler, mode=[0, 2]), (None, None, (token_idx, batch_idx)))

        QK_thr_mma = QK_tiled_mma.get_slice(0)
        tSgQ = QK_thr_mma.partition_A(gQ)
        tQsQ, tQgQ_mkl = cpasync.tma_partition(tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))

        dOV_thr_mma = dOV_tiled_mma.get_slice(0)
        tdPgdO = dOV_thr_mma.partition_A(gdO)
        tdPsdO, tdPgdO_mkl = cpasync.tma_partition(tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tdPgdO, 0, 3))

        if cutlass.const_expr(self.bopt_split_qdo):
            # Knife 3: Q on its own barrier (S waits 64 KiB, not 128 KiB).
            load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
            q_tma_barrier = load_mma_Q_pipeline.producer_get_barrier(load_mma_Q_producer_state)
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[None, head_block_idx, 0],
                tQsQ[None, load_mma_Q_producer_state.index],
                tma_bar_ptr=q_tma_barrier,
            )
            load_mma_Q_producer_state.advance()

            # dO keeps the original pipeline (tx_count = dO bytes only).
            load_mma_QdO_pipeline.producer_acquire(load_mma_QdO_producer_state)
            tma_barrier = load_mma_QdO_pipeline.producer_get_barrier(load_mma_QdO_producer_state)
            cute.copy(
                tma_atom_dO,
                tdPgdO_mkl[None, head_block_idx, 0],
                tdPsdO[None, load_mma_QdO_producer_state.index],
                tma_bar_ptr=tma_barrier,
            )
            load_mma_QdO_producer_state.advance()
        else:
            # Load Q and dO
            load_mma_QdO_pipeline.producer_acquire(load_mma_QdO_producer_state)
            tma_barrier = load_mma_QdO_pipeline.producer_get_barrier(load_mma_QdO_producer_state)
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[None, head_block_idx, 0],
                tQsQ[None, load_mma_QdO_producer_state.index],
                tma_bar_ptr=tma_barrier,
            )

            # Load dO
            cute.copy(
                tma_atom_dO,
                tdPgdO_mkl[None, head_block_idx, 0],
                tdPsdO[None, load_mma_QdO_producer_state.index],
                tma_bar_ptr=tma_barrier,
            )
            load_mma_QdO_producer_state.advance()

        async_copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS), self.acc_dtype, num_bits_per_copy=64)
        thr_layout = cute.make_layout((32), stride=(1))
        val_layout = cute.make_layout((2), stride=(1))
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)
        thr_async_copy = async_tiled_copy.get_slice(local_tidx)

        # (64, 1, M, B)
        gLSE = cute.flat_divide(mLSE, (self.block_tile,))
        gSum_OdO = cute.flat_divide(mSum_OdO, (self.block_tile,))

        # Load LSE
        load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

        gLSE_for_copy = thr_async_copy.partition_S(gLSE[None, head_block_idx, (token_idx, batch_idx)])
        sLSE_for_copy = thr_async_copy.partition_D(sLSE)

        cute.copy(
            async_copy_atom,
            gLSE_for_copy[None, 0],
            sLSE_for_copy[None, 0, load_compute_LSE_producer_state.index],
        )
        load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
        load_compute_LSE_producer_state.advance()

        # Load Sum_OdO
        load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)

        gSum_OdO_for_copy = thr_async_copy.partition_S(gSum_OdO[None, head_block_idx, (token_idx, batch_idx)])
        sSum_OdO_for_copy = thr_async_copy.partition_D(sSum_OdO)

        cute.copy(
            async_copy_atom,
            gSum_OdO_for_copy[None, 0],
            sSum_OdO_for_copy[None, 0, load_compute_sum_OdO_producer_state.index],
        )

        load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
        load_compute_sum_OdO_producer_state.advance()

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdPrV: cute.Tensor,
        tdKVrdOT: cute.Tensor,
        tdKVrP: cute.Tensor,
        tdQrK: cute.Tensor,
        tdQrdST: cute.Tensor,
        tdKVrQT: cute.Tensor,
        tdKVrdS: cute.Tensor,
        tdQrK_tail: Optional[cute.Tensor],
        tdKVrQT_tail: Optional[cute.Tensor],
        tdKVrdS_4: Optional[cute.Tensor],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdKVtdKV: Tuple,
        tdQtdQ: Tuple,
        tile_count: Int32,
        sdS: cute.Tensor,
        pipelines,
    ):
        (
            load_mma_QdO_pipeline,
            load_mma_Q_pipeline,
            load_mma_K_pipeline,
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            mma_compute_dQ_pipeline,
            mma_compute_dealloc_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_reduce_dKV_pipeline,
        ) = pipelines
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV
        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % self.threads_per_warp
        token_idx, _, batch_idx = cute.arch.block_idx()

        load_mma_QdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_QdO_stage)
        load_mma_Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage)
        load_mma_K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_K_stage)
        mma_compute_S_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        mma_compute_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage)
        mma_compute_dealloc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dealloc_stage)
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage)
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage)
        mma_reduce_dKV_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dKV_stage)

        if cutlass.const_expr(self.bopt_split_qdo):
            # Knife 3: the first S gates on Q only.
            load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
        else:
            load_mma_QdO_pipeline.consumer_wait(load_mma_QdO_consumer_state)
        mma_compute_dQ_pipeline.producer_acquire(mma_compute_dQ_producer_state)
        if cutlass.const_expr(self.bopt_dq_early):
            mma_compute_dealloc_pipeline.producer_acquire(mma_compute_dealloc_producer_state)

        tile_index = tile_count - 1
        is_first_mma = True
        while tile_index >= 0:

            load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
            # Gemm S = Q @ K
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                cute.gemm(
                    QK_tiled_mma,
                    tStS,
                    tSrQ[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tSrK[None, None, k_block, load_mma_K_consumer_state.index],
                    tStS,
                )
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            # Knife 3: dP additionally gates on the dO barrier, once.
            if cutlass.const_expr(self.bopt_split_qdo):
                if is_first_mma:
                    load_mma_QdO_pipeline.consumer_wait(load_mma_QdO_consumer_state)

            # Gemm dP = dO @ V
            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll=4):
                cute.gemm(
                    dOV_tiled_mma,
                    tdPtdP,
                    tdPrdO[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tdPrV[None, None, k_block, load_mma_K_consumer_state.index],
                    tdPtdP,
                )
                dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            # Gemm dKV = dO @ P part1
            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
            mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)

            # dKV0
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV0,
                    tdKVrdOT[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV0,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dKV1
            if cutlass.const_expr(self.same_hdim_kv):
                if not is_first_mma:
                    self.t2r_dKV4_done_barrier.arrive_and_wait()
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV1,
                    tdKVrdOT[None, None, 1, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV1,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            # Gemm dKV = Q @ dS part1
            # dKV0
            QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV0,
                    tdKVrQT[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV0,
                )
            # dKV1
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV1,
                    tdKVrQT[None, None, 1, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV1,
                )

            # Notify to reduce the first part of dKV (dKV0, dKV1)
            mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
            mma_reduce_dKV_producer_state.advance()

            # Gemm dKV4 = Q^T[512:575] @ dS (round 1.5, only GEMM5, no GEMM3)
            # dKV4 skips producer_acquire — barrier1 guarantees TMEM safety.
            # Only producer_commit is needed to notify consumer.
            if cutlass.const_expr(not self.same_hdim_kv):
                # barrier1: wait for reduce warps to finish T2R of dKV0/dKV1
                self.t2r_dKV01_done_barrier.arrive_and_wait()

                dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrdS_4, mode=[2]), unroll=2):
                    cute.gemm(
                        dKV4_tiled_mma,
                        tdKVtdKV4,
                        tdKVrQT_tail[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS_4[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV4,
                    )
                    dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Commit dKV4 on pipeline (no acquire needed) to notify consumer
                mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                mma_reduce_dKV_producer_state.advance()

            # Gemm dQ = K @ dS

            # dQ0
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ0,
                    tdQrK[None, None, 0, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ0,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ1
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ1,
                    tdQrK[None, None, 1, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ1,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ2
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ2,
                    tdQrK[None, None, 2, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ2,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ3
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ3,
                    tdQrK[None, None, 3, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ3,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ4 (tail 64 cols: K[512:575] @ dS^T)
            if cutlass.const_expr(not self.same_hdim_kv):
                dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
                for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                    cute.gemm(
                        dQ4_tiled_mma,
                        tdQtdQ4,
                        tdQrK_tail[None, None, 0, k_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ4,
                    )
                    dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Knife 2: the dQ TMEM generation is complete once the last
            # tile's dQ (and dQ4) GEMMs are issued -- commit it here so the
            # compute epilogue overlaps the tail dKV part-2 GEMMs and the
            # reduce drain.  The tcgen05 commit tracks completion of every
            # UMMA issued so far, which also covers the dQ GEMMs' sK reads,
            # so the sdQ-over-sK staging write is safe.  Pure side effect
            # under a dynamic branch; the pipeline state advances after the
            # loop on both paths.
            if cutlass.const_expr(self.bopt_dq_early):
                if tile_index == 0:
                    mma_compute_dQ_pipeline.producer_commit(mma_compute_dQ_producer_state)

            # KV is used
            load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
            load_mma_K_consumer_state.advance()

            # Gemm dKV = dO @ P part2
            # Wait for reduce warps to finish T2R of dKV4 from TMEM,
            # since dKV2 shares the same TMEM offset as dKV4/dKV0.
            if cutlass.const_expr(not self.same_hdim_kv):
                self.t2r_dKV4_done_barrier.arrive_and_wait()
            mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
            # dKV2
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV2,
                    tdKVrdOT[None, None, 2, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV2,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dKV3
            if cutlass.const_expr(self.same_hdim_kv):
                self.t2r_dKV01_done_barrier.arrive_and_wait()
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV3,
                    tdKVrdOT[None, None, 3, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV3,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # P is used
            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            # Gemm dKV = Q @ dS
            # dKV2
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV2,
                    tdKVrQT[None, None, 2, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV2,
                )
            # dKV3
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV3,
                    tdKVrQT[None, None, 3, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV3,
                )

            mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
            mma_reduce_dKV_producer_state.advance()

            # dS is used
            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

            is_first_mma = False
            tile_index -= 1

        if cutlass.const_expr(self.same_hdim_kv):
            self.t2r_dKV4_done_barrier.arrive_and_wait()

        if cutlass.const_expr(self.bopt_dq_early):
            # Dealloc gate: completion-tracking commit at the ORIGINAL
            # baseline commit position (post-loop, post same-hdim t2r
            # rendezvous).  The producer_tail additionally drains the
            # mma<->reduce pipeline: the reduce warps' final
            # consumer_release happens after their last store_dKV T2R
            # (both hdim paths), so the gate transitively covers the
            # final drain T2Rs as well -- the dealloc ordering is then
            # provable for every TMEM access, not timing-assumed
            # (review-panel MAJOR + verify-round MAJOR).
            mma_reduce_dKV_pipeline.producer_tail(mma_reduce_dKV_producer_state)
            mma_compute_dealloc_pipeline.producer_commit(mma_compute_dealloc_producer_state)
            mma_compute_dealloc_producer_state.advance()
            # Degenerate tile_count <= 0: the in-loop commit never fired;
            # commit here so the compute consumer is never stranded.
            # (Live only on the 576 path: the same-hdim degenerate case
            # hangs at the 288-thread t2r rendezvous above, exactly like
            # baseline.)
            if tile_count <= 0:
                mma_compute_dQ_pipeline.producer_commit(mma_compute_dQ_producer_state)
        else:
            mma_compute_dQ_pipeline.producer_commit(mma_compute_dQ_producer_state)
        mma_compute_dQ_producer_state.advance()

        # Q and dO is used
        load_mma_QdO_pipeline.consumer_release(load_mma_QdO_consumer_state)
        load_mma_QdO_consumer_state.advance()
        if cutlass.const_expr(self.bopt_split_qdo):
            load_mma_Q_pipeline.consumer_release(load_mma_Q_consumer_state)
            load_mma_Q_consumer_state.advance()

    @cute.jit
    def _stage_dQ_panel(
        self,
        sdQ_stage: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
    ):
        """store_dQ minus barriers/TMA: T2R -> bf16 -> one SMEM stage."""
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )

        cdQ = cute.make_identity_tensor(cute.select(self.KdS_mma_tiler, mode=[0, 1]))

        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)

        tTR_cdQ = thr_t2r_dQ.partition_D(cdQ)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)

        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)

        tRS_rdQ = self.quantize(tTR_rdQ, 4)

        cute.arch.fence_view_async_tmem_load()

        # ((64,2),(8,8),(1,1))
        thread_layout = cute.make_ordered_layout((128, 64), (0, 1))
        sdQ_slice_tmp = cute.composition(sdQ_stage, thread_layout)
        sdQ_slice = cute.composition(sdQ_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdQ.shape))
        cute.autovec_copy(tRS_rdQ, sdQ_slice)

    @cute.jit
    def compute(
        self,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: Optional[cute.CopyAtom],
        tma_tensor_dQ_64: Optional[cute.Tensor],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: Tuple,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        sP_store: cute.Tensor,
        sdS: cute.Tensor,
        sdS_store: cute.Tensor,
        sdQ: cute.Tensor,
        sdQ4: Optional[cute.Tensor],
        scale_softmax: Float32,
        tile_count: Int32,
        pipelines,
    ):
        (
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dQ_pipeline,
            compute_tmastore_dQ_pipeline,
        ) = pipelines

        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.compute_warp_id[0] * self.threads_per_warp
        tidx_in_warp = tidx % self.threads_per_warp

        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        mma_compute_dQ_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage)
        compute_mma_P_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_P_stage)
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage)
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage)
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage)
        compute_tmastore_dQ_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_tmastore_dQ_stage)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )

        # (((16,4),64), 1, 1):(((65536,2097152),1),0,0)
        tStS = tStS[(None, None), 0, 0]
        tdPtdP = tdPtdP[(None, None), 0, 0]

        dp_idx = tidx_in_wg % 128
        cS = cute.make_identity_tensor(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        cS = cute.composition(cS, sP_store[None, None, compute_mma_P_producer_state.index].layout)
        cdP = cute.make_identity_tensor(cute.select(self.dOV_mma_tiler, mode=[0, 1]))
        cdP = cute.composition(cdP, sdS_store[None, None, compute_mma_dS_producer_state.index].layout)

        tiled_t2r_S = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        tiled_t2r_dP = tcgen05.make_tmem_copy(tmem_load_atom, tdPtdP)
        thr_t2r_S = tiled_t2r_S.get_slice(tidx % 128)
        thr_t2r_dP = tiled_t2r_dP.get_slice(tidx % 128)

        tTR_cS = thr_t2r_S.partition_D(cS)
        tTR_sS = thr_t2r_S.partition_D(sP_store[None, None, compute_mma_P_producer_state.index])
        tTR_rS = cute.make_rmem_tensor(tTR_sS.shape, self.acc_dtype)

        tTR_tS = thr_t2r_S.partition_S(tStS)

        tTR_cdP = thr_t2r_dP.partition_D(cdP)
        tTR_sdP = thr_t2r_dP.partition_D(sdS_store[None, None, compute_mma_dS_producer_state.index])
        tTR_rdP = cute.make_rmem_tensor(tTR_sdP.shape, self.acc_dtype)

        tTR_tdP = thr_t2r_dP.partition_S(tdPtdP)

        load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_wait(load_compute_sum_OdO_consumer_state)

        log2_e = Float32(math.log2(math.e))
        softmax_scale_log2_e = scale_softmax * log2_e

        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.element_dtype,
        )
        smem_store_p = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r_S)
        thr_smem_store_p = smem_store_p.get_slice(tidx % 128)
        sP_store_slice = sP_store[None, None, compute_mma_P_producer_state.index]
        tRS_sP = thr_smem_store_p.partition_D(sP_store_slice)
        tRS_rP = cute.make_rmem_tensor(tRS_sP.shape, self.element_dtype)

        smem_store_ds = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r_dP)
        thr_smem_store_ds = smem_store_ds.get_slice(tidx % 128)
        sdS_store_slice = sdS_store[None, None, compute_mma_dS_producer_state.index]
        tRS_sdS = thr_smem_store_ds.partition_D(sdS_store_slice)
        tRS_rdS = cute.make_rmem_tensor(tRS_sdS.shape, self.element_dtype)

        tile_index = tile_count - 1
        while tile_index >= 0:
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)

            cute.copy(tiled_t2r_S, tTR_tS, tTR_rS)

            for i in cutlass.range(0, cute.size(tTR_rS), 2, unroll_full=True):

                lse = (
                    sLSE[cute.get(tTR_cS[i], mode=[0]), load_compute_LSE_consumer_state.index],
                    sLSE[cute.get(tTR_cS[i + 1], mode=[0]), load_compute_LSE_consumer_state.index],
                )

                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rS[i], tTR_rS[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)

            tTR_rS_f16 = self.quantize(tTR_rS, 4)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            # ======= stsm ============
            tRS_rP.store(smem_store_p.retile(tTR_rS_f16).load())
            cute.copy(smem_store_p, tRS_rP, tRS_sP)

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # Notify for P
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()

            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

            cute.copy(tiled_t2r_dP, tTR_tdP, tTR_rdP)

            for i in cutlass.range(0, cute.size(tTR_rdP), 2, unroll_full=True):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[
                            cute.get(tTR_cdP[i], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                        sSum_OdO[
                            cute.get(tTR_cdP[i + 1], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                    ),
                )

                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdP[i], tTR_rdP[i + 1]), (tTR_rS[i], tTR_rS[i + 1]))

            tTR_rdP_f16 = self.quantize(tTR_rdP, 4, scale_softmax)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()

            tRS_rdS.store(smem_store_ds.retile(tTR_rdP_f16).load())
            cute.copy(smem_store_ds, tRS_rdS, tRS_sdS)

            # self.compute_sync_barrier.arrive_and_wait()

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            tile_index -= 1

        load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)

        # Store dQ
        tdQtdQ0 = tdQtdQ0[(None, None), 0, 0]
        tdQtdQ1 = tdQtdQ1[(None, None), 0, 0]
        tdQtdQ2 = tdQtdQ2[(None, None), 0, 0]
        tdQtdQ3 = tdQtdQ3[(None, None), 0, 0]

        # (512, 64)
        gdQ = cute.local_tile(tma_tensor_dQ, cute.select(self.KdS_mma_tiler, mode=[0, 1]), (None, None, (token_idx, batch_idx)))
        # (128, 64)
        gdQ0 = gdQ[None, None, 0, head_block_idx]
        gdQ1 = gdQ[None, None, 1, head_block_idx]
        gdQ2 = gdQ[None, None, 2, head_block_idx]
        gdQ3 = gdQ[None, None, 3, head_block_idx]

        # Knife 1: four distinct staging stages inside the dead sK
        # allocation; without the knife all four alias stage 0 (baseline).
        if cutlass.const_expr(self.bopt_epi):
            sdQ_slice0 = sdQ[None, None, 0]
            sdQ_slice1 = sdQ[None, None, 1]
            sdQ_slice2 = sdQ[None, None, 2]
            sdQ_slice3 = sdQ[None, None, 3]
        else:
            sdQ_slice0 = sdQ[None, None, mma_compute_dQ_consumer_state.index]
            sdQ_slice1 = sdQ_slice0
            sdQ_slice2 = sdQ_slice0
            sdQ_slice3 = sdQ_slice0

        # ((64,2),(8,8),(1,1))
        tdQsdQ0, tdQgdQ0_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice0, 0, 2),
            cute.group_modes(gdQ0, 0, 2),
        )
        tdQsdQ1, tdQgdQ1_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice1, 0, 2),
            cute.group_modes(gdQ1, 0, 2),
        )
        tdQsdQ2, tdQgdQ2_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice2, 0, 2),
            cute.group_modes(gdQ2, 0, 2),
        )
        tdQsdQ3, tdQgdQ3_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice3, 0, 2),
            cute.group_modes(gdQ3, 0, 2),
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            tdQtdQ4 = tdQtdQ4[(None, None), 0, 0]
            gdQ4 = cute.local_tile(tma_tensor_dQ_64, cute.select(self.dQ4_mma_tiler, mode=[0, 1]), (None, None, (token_idx, batch_idx)))
            gdQ4 = gdQ4[None, None, 8, head_block_idx]

            sdQ4_slice = sdQ4[None, None, mma_compute_dQ_consumer_state.index]

            tdQsdQ4, tdQgdQ4_mkl = cpasync.tma_partition(
                tma_atom_dQ_64,
                0,
                cute.make_layout(1),
                cute.group_modes(sdQ4_slice, 0, 2),
                cute.group_modes(gdQ4, 0, 2),
            )

        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        mma_compute_dQ_pipeline.consumer_wait(mma_compute_dQ_consumer_state)

        if cutlass.const_expr(self.bopt_epi):
            # Knife 1: stage all four panels, then issue the four TMA
            # stores back-to-back under one commit group.
            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            # Wait in all threads for the acquire to complete
            self.compute_sync_barrier.arrive_and_wait()

            self._stage_dQ_panel(sdQ_slice0, tdQtdQ0, dp_idx)
            self._stage_dQ_panel(sdQ_slice1, tdQtdQ1, dp_idx)
            self._stage_dQ_panel(sdQ_slice2, tdQtdQ2, dp_idx)
            self._stage_dQ_panel(sdQ_slice3, tdQtdQ3, dp_idx)

            self.compute_sync_barrier.arrive_and_wait()

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            self.compute_sync_barrier.arrive_and_wait()

            if warp_idx == self.compute_warp_id[0]:
                cute.copy(tma_atom_dQ, tdQsdQ0, tdQgdQ0_mkl)
                cute.copy(tma_atom_dQ, tdQsdQ1, tdQgdQ1_mkl)
                cute.copy(tma_atom_dQ, tdQsdQ2, tdQgdQ2_mkl)
                cute.copy(tma_atom_dQ, tdQsdQ3, tdQgdQ3_mkl)
                compute_tmastore_dQ_pipeline.producer_commit()

            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()
        else:
            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            # Wait in all threads for the acquire to complete
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ(
                tma_atom_dQ,
                sdQ_slice0,
                tdQsdQ0,
                tdQgdQ0_mkl,
                tdQtdQ0,
                dp_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()

            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ(
                tma_atom_dQ,
                sdQ_slice1,
                tdQsdQ1,
                tdQgdQ1_mkl,
                tdQtdQ1,
                dp_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()

            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ(
                tma_atom_dQ,
                sdQ_slice2,
                tdQsdQ2,
                tdQgdQ2_mkl,
                tdQtdQ2,
                dp_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()

            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ(
                tma_atom_dQ,
                sdQ_slice3,
                tdQsdQ3,
                tdQgdQ3_mkl,
                tdQtdQ3,
                dp_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()
            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

        # Store dQ4 (tail 64 cols)
        if cutlass.const_expr(not self.same_hdim_kv):
            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ_64(
                tma_atom_dQ_64,
                sdQ4_slice,
                tdQsdQ4,
                tdQgdQ4_mkl,
                tdQtdQ4,
                dp_idx,
                wg_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()
            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

        mma_compute_dQ_pipeline.consumer_release(mma_compute_dQ_consumer_state)
        mma_compute_dQ_consumer_state.advance()

        compute_tmastore_dQ_pipeline.producer_tail()
