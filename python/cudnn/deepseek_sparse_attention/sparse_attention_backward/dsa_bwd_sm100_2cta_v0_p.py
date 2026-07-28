"""Pipelined SM100 two-CTA DSA backward, under staged integration.

This module materializes the v0 execution contract for BF16 GQA128/D512:

* one static three-stage, 48-KiB-per-stage operand FIFO;
* persistent lifecycle roles and all planned full/empty pipelines;
* two-stage P/dS storage retained through both D256 gradient rounds; and
* true CG2 completion-driven operand and TMEM recycling.

The kernel now uses the production F/BV/BQ operand-loading contract together
with final-generation S/dP T2R, FP32 P/dS math, and directed DSM exchange.
It also performs rank-owned dKV T2R/FP32 atomics and a staged dQ epilogue
through one fully drained operand main region.  It is intentionally not wired
into the public interface until the remaining runtime control plane closes.

This self-contained module includes the common two-CTA host/layout base and
the complete verified v0 implementation used by the canonical harness.
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.typing import BFloat16, Float32, Int32

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100


@dsl_user_op
def _atomic_add_fp32x4_v0_p(
    value_0: Float32,
    value_1: Float32,
    value_2: Float32,
    value_3: Float32,
    destination: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one aligned, result-discarding FP32x4 global reduction."""

    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .v4 .f32 values;\n\t"
            "mov.f32 values.x, $1;\n\t"
            "mov.f32 values.y, $2;\n\t"
            "mov.f32 values.z, $3;\n\t"
            "mov.f32 values.w, $4;\n\t"
            "red.global.add.v4.f32 [$0], values;\n\t"
            "}\n"
        ),
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _map_smem_to_cluster_rank(
    smem_ptr: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map a CTA-local shared-memory pointer to another cluster rank."""

    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cpasync_bulk_s2cluster(
    source: cute.Pointer,
    destination: cute.Pointer,
    completion_barrier: cute.Pointer,
    copy_bytes: int | Int32,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one shared-to-cluster bulk copy to ``peer_rank``."""

    source_i32 = source.toint(loc=loc, ip=ip).ir_value()
    destination_i32 = _map_smem_to_cluster_rank(
        destination,
        peer_rank,
        loc=loc,
        ip=ip,
    ).ir_value()
    barrier_i32 = _map_smem_to_cluster_rank(
        completion_barrier,
        peer_rank,
        loc=loc,
        ip=ip,
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i32,
            source_i32,
            barrier_i32,
            Int32(copy_bytes).ir_value(loc=loc, ip=ip),
        ],
        (
            "cp.async.bulk.shared::cluster.shared::cta."
            "mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];"
        ),
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionDSABackwardSm100TwoCTA(FlashAttentionDSABackwardSm100):
    """Fixed GQA128/D512 two-CTA DSA backward implementation."""

    arch = 100

    H_TILE_CLUSTER = 128
    H_TILE_CTA = 64
    N_TILE = 64
    N_TILE_CTA = 32
    D_HEAD = 512
    D_TILE_CLUSTER = 256
    D_TILE_CTA = 128
    D_ROUNDS = D_HEAD // D_TILE_CLUSTER
    K_CHUNK = 128
    K_CHUNKS = D_HEAD // K_CHUNK

    DKV_MMA_TILER = (D_TILE_CLUSTER, N_TILE, H_TILE_CLUSTER)
    DQ_MMA_TILER = (D_TILE_CLUSTER, H_TILE_CLUSTER, N_TILE)

    CLUSTER_SHAPE_MNK = (2, 1, 1)
    MATH_THREADS_PER_CTA = 128
    MATH_WARPS = MATH_THREADS_PER_CTA // 32
    THREADS_PER_CTA = 256
    KV_LOAD_THREADS = 128
    KV_LOAD_THREAD_BEGIN = MATH_THREADS_PER_CTA
    KV_GROUP_SIZE = 8
    KV_NUM_GROUPS = KV_LOAD_THREADS // KV_GROUP_SIZE
    TMEM_COLUMNS = 512
    MAX_SMEM_BYTES = 232_448
    QUADRANT_ELEMENTS = H_TILE_CTA * N_TILE_CTA
    QUADRANT_BYTES = QUADRANT_ELEMENTS * (BFloat16.width // 8)

    TMEM_S_OFFSET = 0
    TMEM_DP_OFFSET = 64
    TMEM_DKV0_OFFSET = 128
    TMEM_DKV1_OFFSET = 192
    TMEM_DQ0_OFFSET = 256
    TMEM_DQ1_OFFSET = 384

    # One group-wide completion generation is consumed synchronously after
    # each issued operation.  The stage exists to obtain the tcgen05 commit
    # completion contract, not to encode task identity.
    MMA_DONE_STAGES = 2
    SCORE_SOURCE_BARRIERS = 2

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
        self.element_dtype = BFloat16
        self.acc_dtype = Float32
        self.threads_per_cta = self.THREADS_PER_CTA
        self.shared_storage = None
        self.shared_storage_bytes = 0
        self.layout_report = {}
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.THREADS_PER_CTA,
        )

    def _specialize_shared_storage(
        self,
        default_storage,
        score_a_layout_staged,
        score_b_layout_staged,
        dkv_a_layout_staged,
        dkv_b_layout_staged,
        dq_a_layout_staged,
        dq_b_layout_staged,
    ):
        """Allow a derived execution schedule to replace only main-kernel SMEM.

        The sequential checkpoint owns the common tensor normalization, MMA
        construction, preprocessing, and postprocessing.  Pipelined variants
        need those exact objects but have a different shared-memory lifetime
        graph.  Keeping the hook at the storage boundary avoids copying the
        large host-side ``__call__`` while leaving the checkpoint's default
        layout unchanged.
        """

        return default_storage

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
        """Compile preprocessing, the CG2 main kernel, and postprocessing."""

        # External tensors use the same logical views as the established
        # one-CTA path.  The main kernel derives both score [H,D] and
        # gradient [D,H] coordinates from these canonical layouts.
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[1], mQ.shape[2], (mQ.shape[0], 1)),
                stride=(mQ.stride[1], mQ.stride[2], (mQ.stride[0], 0)),
            ),
        )
        mKV = cute.make_tensor(
            mKV.iterator,
            cute.make_layout(
                (mKV.shape[0], mKV.shape[1], (1, 1)),
                stride=(mKV.stride[0], mKV.stride[1], (0, 0)),
            ),
        )
        mOut = cute.make_tensor(
            mOut.iterator,
            cute.make_layout(
                (mOut.shape[1], mOut.shape[2], (mOut.shape[0], 1)),
                stride=(mOut.stride[1], mOut.stride[2], (mOut.stride[0], 0)),
            ),
        )
        mdO = cute.make_tensor(
            mdO.iterator,
            cute.make_layout(
                (mdO.shape[1], mdO.shape[2], (mdO.shape[0], 1)),
                stride=(mdO.stride[1], mdO.stride[2], (mdO.stride[0], 0)),
            ),
        )
        mdQ = cute.make_tensor(
            mdQ.iterator,
            cute.make_layout(
                (mdQ.shape[2], mdQ.shape[1], (mdQ.shape[0], 1)),
                stride=(mdQ.stride[2], mdQ.stride[1], (mdQ.stride[0], 0)),
            ),
        )
        # Keep an external-order [H,D,(token,batch)] view for the v0 staged
        # dQ epilogue.  The established mdQ view above remains [D,H,...] for
        # the sequential direct-store checkpoint.
        mdQ_epi = cute.make_tensor(
            mdQ.iterator,
            cute.make_layout(
                (
                    self.H_TILE_CLUSTER,
                    self.D_HEAD,
                    mdQ.shape[2],
                ),
                stride=(
                    mdQ.stride[1],
                    mdQ.stride[0],
                    mdQ.stride[2],
                ),
            ),
        )
        mdKV = cute.make_tensor(
            mdKV.iterator,
            cute.make_layout(
                (mdKV.shape[1], mdKV.shape[0], (1, 1)),
                stride=(mdKV.stride[1], mdKV.stride[0], (0, 0)),
            ),
        )
        mLSE = cute.make_tensor(
            mLSE.iterator,
            cute.make_layout(
                (mLSE.shape[1], (mLSE.shape[0], 1)),
                stride=(mLSE.stride[1], (mLSE.stride[0], 0)),
            ),
        )
        mdSink = cute.make_tensor(
            mdSink.iterator,
            cute.make_layout((mdSink.shape[0], (1, 1)), stride=(1, (0, 0))),
        )
        mAttnSink = cute.make_tensor(mAttnSink.iterator, mdSink.layout)
        mTopkIdxs = cute.make_tensor(
            mTopkIdxs.iterator,
            cute.make_layout(
                (mTopkIdxs.shape[1], (mTopkIdxs.shape[0], 1)),
                stride=(mTopkIdxs.stride[1], (mTopkIdxs.stride[0], 0)),
            ),
        )
        if cutlass.const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(
                mTopkLength.iterator,
                cute.make_layout(
                    (mTopkLength.shape[0], (1, 1)),
                    stride=(mTopkLength.stride[0], (0, 0)),
                ),
            )
        mQT = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (self.D_HEAD, self.H_TILE_CLUSTER, mQ.shape[2]),
                stride=(mQ.stride[1], mQ.stride[0], mQ.stride[2]),
            ),
        )
        mdOT = cute.make_tensor(
            mdO.iterator,
            cute.make_layout(
                (self.D_HEAD, self.H_TILE_CLUSTER, mdO.shape[2]),
                stride=(mdO.stride[1], mdO.stride[0], mdO.stride[2]),
            ),
        )

        cg2 = tcgen05.CtaGroup.TWO
        score_tiler = (self.H_TILE_CLUSTER, self.N_TILE, self.K_CHUNK)
        dkv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.K,
            self.acc_dtype,
            cg2,
            self.DKV_MMA_TILER[:2],
        )
        dq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            cg2,
            self.DQ_MMA_TILER[:2],
        )
        score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cg2,
            score_tiler[:2],
        )
        dp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cg2,
            score_tiler[:2],
        )
        atom_thr_size = cute.size(dkv_tiled_mma.thr_id.shape)
        assert atom_thr_size == self.CLUSTER_SHAPE_MNK[0]
        assert cute.size(dq_tiled_mma.thr_id.shape) == atom_thr_size
        assert cute.size(score_tiled_mma.thr_id.shape) == atom_thr_size
        assert cute.size(dp_tiled_mma.thr_id.shape) == atom_thr_size

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.CLUSTER_SHAPE_MNK),
            (dkv_tiled_mma.thr_id.shape,),
        )

        score_a_layout_staged = sm100_utils.make_smem_layout_a(
            score_tiled_mma,
            score_tiler,
            self.element_dtype,
            self.K_CHUNKS,
        )
        score_b_layout_staged = sm100_utils.make_smem_layout_b(
            score_tiled_mma,
            score_tiler,
            self.element_dtype,
            self.K_CHUNKS,
        )
        dkv_a_layout_staged = sm100_utils.make_smem_layout_a(
            dkv_tiled_mma,
            self.DKV_MMA_TILER,
            self.element_dtype,
            1,
        )
        dkv_b_layout_staged = sm100_utils.make_smem_layout_b(
            dkv_tiled_mma,
            self.DKV_MMA_TILER,
            self.element_dtype,
            1,
        )
        dq_a_layout_staged = sm100_utils.make_smem_layout_a(
            dq_tiled_mma,
            self.DQ_MMA_TILER,
            self.element_dtype,
            1,
        )
        dq_b_layout_staged = sm100_utils.make_smem_layout_b(
            dq_tiled_mma,
            self.DQ_MMA_TILER,
            self.element_dtype,
            1,
        )
        dq_epi_tile = (
            self.H_TILE_CLUSTER,
            self.D_TILE_CTA,
        )
        dq_epi_layout_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.from_tensor(mdQ_epi),
            dq_epi_tile,
            1,
        )
        dq_epi_layout = cute.select(
            dq_epi_layout_staged,
            mode=[0, 1],
        )
        dq_epi_bytes = cute.size_in_bytes(
            self.element_dtype,
            dq_epi_layout_staged,
        )
        assert dq_epi_bytes <= 32 * 1024
        tma_atom_dq_epi, tma_tensor_dq_epi = (
            cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdQ_epi,
                dq_epi_layout,
                dq_epi_tile,
            )
        )
        self.layout_report = {
            "score_a": str(score_a_layout_staged),
            "score_b": str(score_b_layout_staged),
            "dkv_a_staged": str(dkv_a_layout_staged),
            "dkv_b_staged": str(dkv_b_layout_staged),
            "dq_a_staged": str(dq_a_layout_staged),
            "dq_b_staged": str(dq_b_layout_staged),
            "dq_epi_staged": str(dq_epi_layout_staged),
            "dq_epi_bytes": dq_epi_bytes,
        }
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 16384
        assert cute.cosize(dkv_b_layout_staged) <= 4096
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096
        assert cute.cosize(score_a_layout_staged) >= (
            self.H_TILE_CTA * self.N_TILE
        )
        assert cute.cosize(score_b_layout_staged) >= (
            self.QUADRANT_ELEMENTS
        )

        # Q and dO are regular score-A tensors.  Completion is CTA-local
        # while the subsequent MMA remains a genuine CG2 instruction.
        score_a_layout = cute.select(
            score_a_layout_staged,
            mode=[0, 1, 2],
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(
            tcgen05.CtaGroup.ONE
        )
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            score_a_layout,
            score_tiler,
            score_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdO,
            score_a_layout,
            score_tiler,
            dp_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        score_a_stage_bytes = cute.size_in_bytes(
            self.element_dtype,
            score_a_layout,
        )
        grad_a_layout = cute.select(
            dkv_a_layout_staged,
            mode=[0, 1, 2],
        )
        tma_atom_qt, tma_tensor_qt = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQT,
            grad_a_layout,
            self.DKV_MMA_TILER,
            dkv_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_dot, tma_tensor_dot = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdOT,
            grad_a_layout,
            self.DKV_MMA_TILER,
            dkv_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        grad_a_stage_bytes = cute.size_in_bytes(
            self.element_dtype,
            grad_a_layout,
        )

        @cute.struct
        class SharedStorage:
            # Reused for score/gradient TMA completion generations.
            source_done_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.SCORE_SOURCE_BARRIERS,
            ]
            exchange_mbars: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            mma_full_empty_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.MMA_DONE_STAGES * 2,
            ]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64

            score_q: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(score_a_layout_staged),
                ],
                1024,
            ]
            score_do: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(score_a_layout_staged),
                ],
                1024,
            ]
            score_kv: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(score_b_layout_staged),
                ],
                1024,
            ]
            p_t: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(dkv_b_layout_staged),
                ],
                1024,
            ]
            ds_dk_t: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(dkv_b_layout_staged),
                ],
                1024,
            ]
            kv_t: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(dq_a_layout_staged),
                ],
                1024,
            ]
            ds_dq: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    cute.cosize(dq_b_layout_staged),
                ],
                1024,
            ]

        SharedStorage = self._specialize_shared_storage(
            SharedStorage,
            score_a_layout_staged,
            score_b_layout_staged,
            dkv_a_layout_staged,
            dkv_b_layout_staged,
            dq_a_layout_staged,
            dq_b_layout_staged,
        )
        self.shared_storage = SharedStorage
        self.shared_storage_bytes = SharedStorage.size_in_bytes()
        assert self.shared_storage_bytes <= self.MAX_SMEM_BYTES

        score_cta_shape = (
            self.H_TILE_CTA,
            self.N_TILE,
            self.K_CHUNK,
        )
        score_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            score_cta_shape,
            True,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
        )
        score_tmem_load = sm100_utils.get_tmem_load_op(
            score_cta_shape,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            score_epi_tile,
            True,
        )
        dkv_cta_shape = (
            self.D_TILE_CTA,
            self.N_TILE,
            self.H_TILE_CLUSTER,
        )
        dkv_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            dkv_cta_shape,
            True,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
        )
        dkv_tmem_load = sm100_utils.get_tmem_load_op(
            dkv_cta_shape,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            dkv_epi_tile,
            True,
        )
        dq_cta_shape = (
            self.D_TILE_CTA,
            self.H_TILE_CLUSTER,
            self.N_TILE,
        )
        dq_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            dq_cta_shape,
            True,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
        )
        dq_tmem_load = sm100_utils.get_tmem_load_op(
            dq_cta_shape,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            dq_epi_tile,
            True,
        )

        sum_OdO, scaled_LSE, mdKV_acc = self.get_workspace_tensor(
            problem_shape,
            workspace_LSE_OdO,
            workspace_dKV,
            mQ.shape[2][0],
            mKV.shape[0],
            self.acc_dtype,
        )
        mdKV_acc = cute.make_tensor(mdKV_acc.iterator, mdKV.layout)

        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))
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
            grid=self._compute_sum_OdO_grid(
                problem_shape,
                self.sum_OdO_block_q,
            ),
            block=[
                self.sum_OdO_num_threads_d,
                self.sum_OdO_num_threads_q,
                1,
            ],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.kernel(
            problem_shape,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_qt,
            tma_tensor_qt,
            tma_atom_dot,
            tma_tensor_dot,
            mQ,
            mKV,
            mdO,
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
            dkv_a_layout_staged,
            dkv_b_layout_staged,
            dq_a_layout_staged,
            dq_b_layout_staged,
            cluster_layout_vmnk,
            score_tmem_load,
            dkv_tmem_load,
            dq_tmem_load,
            tma_atom_dq_epi,
            tma_tensor_dq_epi,
            dq_epi_layout_staged,
            score_a_stage_bytes,
            grad_a_stage_bytes,
            trace_buffer,
            trace_token_idx,
            trace_batch_idx,
        ).launch(
            grid=(
                2 * problem_shape[0],
                1,
                problem_shape[3][1],
            ),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=self.CLUSTER_SHAPE_MNK,
            smem=self.shared_storage_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.block_seq = 4 if self.max_topk == 2048 else 32
        self.num_threads_D_convert = 32
        self.num_threads_seq = 4 if self.max_topk == 2048 else self.block_seq
        convert_grid_x = (
            mKV.shape[0] + self.block_seq - 1
        ) // self.block_seq
        self.convert_canonical(
            mdKV_acc,
            mdKV,
            mKV.shape[0],
        ).launch(
            grid=[convert_grid_x, 1, 1],
            block=[
                self.num_threads_D_convert,
                self.num_threads_seq,
                1,
            ],
            stream=stream,
        )

        self.sum_dSink(
            sum_OdO,
            scaled_LSE,
            mAttnSink,
            mdSink,
            problem_shape,
        ).launch(
            grid=(
                cute.ceil_div(problem_shape[0], self.dSink_block_q),
                problem_shape[3][0],
                problem_shape[3][1],
            ),
            block=[self.dSink_num_threads, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def convert_canonical(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        if seq_id < seqlen:
            for d_block in cutlass.range_constexpr(
                self.D_HEAD // self.num_threads_D_convert
            ):
                d = d_block * self.num_threads_D_convert + tidx
                mdKV[d, seq_id, (0, batch_idx)] = self.element_dtype(
                    mdKV_acc[d, seq_id, (0, batch_idx)]
                )

    @cute.jit
    def _issue_four_chunks(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state,
    ):
        """Issue one score-side CG2 GEMM over four resident D128 chunks."""

        done_pipeline.producer_acquire(producer_state)
        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        for flat_k_block in cutlass.range_constexpr(
            self.K_CHUNKS * k_blocks_per_chunk
        ):
            chunk = flat_k_block // k_blocks_per_chunk
            k_block = flat_k_block % k_blocks_per_chunk
            cute.gemm(
                mma,
                accumulator,
                a_fragment[None, None, k_block, chunk],
                b_fragment[None, None, k_block, chunk],
                accumulator,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _copy_sparse_k_d128_row(
        self,
        mKV: cute.Tensor,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        kv_index: Int32,
        batch_idx: Int32,
        d_offset: Int32,
        index_in_group: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Copy one D128 slice of a sparse KV row with 128-bit cp.async."""

        source_row_full = mKV[kv_index, None, (0, batch_idx)]
        source_row_offset = source_row_full.iterator + d_offset
        source_row = cute.make_tensor(
            cute.make_ptr(
                self.element_dtype,
                source_row_offset.llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((self.K_CHUNK,)),
        )
        source_chunks = cute.flat_divide(source_row, (8,))
        destination_row_tensor = destination_rows[
            destination_row,
            None,
        ]
        destination_chunks = cute.flat_divide(
            destination_row_tensor,
            (8,),
        )
        for tile in cutlass.range_constexpr(self.K_CHUNK // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            thread_source = thread_copy.partition_S(
                source_chunks[None, chunk_index]
            )
            thread_destination = thread_copy.partition_D(
                destination_chunks[None, chunk_index]
            )
            cute.copy(copy_atom, thread_source, thread_destination)

    @cute.jit
    def _zero_sparse_k_d128_row(
        self,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        index_in_group: Int32,
    ):
        """Cooperatively zero one D128 sparse-row destination."""

        destination_row_tensor = destination_rows[
            destination_row,
            None,
        ]
        destination_chunks = cute.flat_divide(
            destination_row_tensor,
            (8,),
        )
        for tile in cutlass.range_constexpr(self.K_CHUNK // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            destination_chunks[None, chunk_index].fill(0.0)

    @cute.jit
    def _load_score_kv(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        destination: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Gather the rank-owned N32 x D512 score B with 128-bit copies."""

        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_iteration * self.KV_NUM_GROUPS + group_index
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = mTopkIdxs[
                    topk_slot,
                    (token_idx, batch_idx),
                ]

            for chunk in cutlass.range_constexpr(self.K_CHUNKS):
                destination_rows = cute.composition(
                    destination[None, None, None, chunk],
                    cute.make_layout(
                        (self.N_TILE_CTA, self.K_CHUNK)
                    ),
                )
                if kv_index >= 0:
                    self._copy_sparse_k_d128_row(
                        mKV,
                        destination_rows,
                        local_n,
                        kv_index,
                        batch_idx,
                        Int32(chunk * self.K_CHUNK),
                        index_in_group,
                        copy_atom,
                        thread_copy,
                    )
                else:
                    self._zero_sparse_k_d128_row(
                        destination_rows,
                        local_n,
                        index_in_group,
                    )

    @cute.jit
    def _load_grad_a(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        coordinate_partition: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tidx: Int32,
    ):
        """Load one rank-owned D128 x H128 dO.T or Q.T partition."""

        linear_index = tidx
        while linear_index < cute.size(destination):
            source_coordinate = cute.idx2crd(
                linear_index,
                coordinate_partition.shape,
            )
            destination_coordinate = cute.idx2crd(
                linear_index,
                destination.shape,
            )
            logical_coordinate = coordinate_partition[
                source_coordinate
            ]
            d_index = Int32(cute.get(logical_coordinate, mode=[0]))
            head = Int32(cute.get(logical_coordinate, mode=[1]))
            destination[destination_coordinate] = source[
                head,
                d_index,
                (token_idx, batch_idx),
            ]
            linear_index += self.THREADS_PER_CTA

    @cute.jit
    def _load_grad_k(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        destination: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        round_index: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Gather one rank-owned D128 x N64 gradient K.T operand."""

        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE // self.KV_NUM_GROUPS
        destination_rows = cute.composition(
            destination[None, None, None, 0],
            cute.make_layout(
                (self.N_TILE, self.D_TILE_CTA),
                stride=(self.D_TILE_CTA, 1),
            ),
        )
        d_offset = (
            round_index * self.D_TILE_CLUSTER
            + rank * self.D_TILE_CTA
        )
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            logical_n = row_iteration * self.KV_NUM_GROUPS + group_index
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = mTopkIdxs[
                    topk_slot,
                    (token_idx, batch_idx),
                ]
            if kv_index >= 0:
                self._copy_sparse_k_d128_row(
                    mKV,
                    destination_rows,
                    logical_n,
                    kv_index,
                    batch_idx,
                    Int32(d_offset),
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
            else:
                self._zero_sparse_k_d128_row(
                    destination_rows,
                    logical_n,
                    index_in_group,
                )

    @cute.jit
    def _compute_pd_from_tmem(
        self,
        t_score: cute.Tensor,
        t_dp: cute.Tensor,
        score_tmem_load: cute.CopyAtom,
        rank_score_coordinates: cute.Tensor,
        scaled_lse: cute.Tensor,
        sum_odo: cute.Tensor,
        p_scratch: cute.Tensor,
        ds_scratch: cute.Tensor,
        scale_softmax: Float32,
        token_idx: Int32,
        batch_idx: Int32,
        tidx: Int32,
        done_pipeline,
        consumer_state,
    ):
        """T2R score/dP, run FP32 softmax math, and stage local P/dS."""

        math_state = consumer_state.clone()
        if tidx < self.MATH_THREADS_PER_CTA:
            tiled_score_t2r = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_score,
            )
            score_thread = tiled_score_t2r.get_slice(tidx)
            score_source = score_thread.partition_S(t_score)
            score_coordinates = score_thread.partition_D(
                rank_score_coordinates
            )
            r_score = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.acc_dtype,
            )

            tiled_dp_t2r = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_dp,
            )
            dp_thread = tiled_dp_t2r.get_slice(tidx)
            dp_source = dp_thread.partition_S(t_dp)
            r_dp = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.acc_dtype,
            )

            done_pipeline.consumer_wait(math_state)
            cute.copy(tiled_score_t2r, score_source, r_score)
            cute.arch.fence_view_async_tmem_load()
            done_pipeline.consumer_release(math_state)
            math_state.advance()

            done_pipeline.consumer_wait(math_state)
            cute.copy(tiled_dp_t2r, dp_source, r_dp)
            cute.arch.fence_view_async_tmem_load()
            done_pipeline.consumer_release(math_state)

            softmax_scale_log2_e = scale_softmax * Float32(
                math.log2(math.e)
            )
            for value_index in cutlass.range_constexpr(
                cute.size(r_score)
            ):
                head = Int32(
                    cute.get(score_coordinates[value_index], mode=[0])
                )
                n_index = Int32(
                    cute.get(score_coordinates[value_index], mode=[1])
                )
                p_value = cute.math.exp2(
                    r_score[value_index] * softmax_scale_log2_e
                    + scaled_lse[head, (token_idx, batch_idx)],
                    fastmath=True,
                )
                ds_value = (
                    (
                        r_dp[value_index]
                        + sum_odo[head, (token_idx, batch_idx)]
                    )
                    * p_value
                    * scale_softmax
                )
                local_h = head % self.H_TILE_CTA
                scratch_offset = n_index * self.H_TILE_CTA + local_h
                p_scratch[scratch_offset] = self.element_dtype(p_value)
                ds_scratch[scratch_offset] = self.element_dtype(
                    ds_value
                )
            cute.arch.fence_view_async_shared()

        consumer_state.advance()
        consumer_state.advance()
        cute.arch.barrier()
        return consumer_state

    @cute.jit
    def _issue_dv(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        dout_fragment: cute.Tensor,
        p_fragment: cute.Tensor,
        done_pipeline,
        producer_state,
    ):
        done_pipeline.producer_acquire(producer_state)
        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range_constexpr(
            cute.size(dout_fragment, mode=[2])
        ):
            cute.gemm(
                mma,
                t_dkv,
                dout_fragment[None, None, k_block, 0],
                p_fragment[None, None, k_block, 0],
                t_dkv,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_dk(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        q_fragment: cute.Tensor,
        ds_dk_fragment: cute.Tensor,
        done_pipeline,
        producer_state,
    ):
        done_pipeline.producer_acquire(producer_state)
        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, True)
        for k_block in cutlass.range_constexpr(
            cute.size(q_fragment, mode=[2])
        ):
            cute.gemm(
                mma,
                t_dkv,
                q_fragment[None, None, k_block, 0],
                ds_dk_fragment[None, None, k_block, 0],
                t_dkv,
            )
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_dq(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq: cute.Tensor,
        kv_fragment: cute.Tensor,
        ds_dq_fragment: cute.Tensor,
        accumulate: bool,
        done_pipeline,
        producer_state,
    ):
        """Issue one persistent dQ.T contribution."""

        done_pipeline.producer_acquire(producer_state)
        mma = dq_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        for k_block in cutlass.range_constexpr(
            cute.size(kv_fragment, mode=[2])
        ):
            cute.gemm(
                mma,
                t_dq,
                kv_fragment[None, None, k_block, 0],
                ds_dq_fragment[None, None, k_block, 0],
                t_dq,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)

        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _wait_mma(
        self,
        done_pipeline,
        consumer_state,
        tidx: Int32,
    ):
        math_state = consumer_state.clone()
        if tidx < self.MATH_THREADS_PER_CTA:
            done_pipeline.consumer_wait(math_state)
            done_pipeline.consumer_release(math_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _atomic_dkv_from_tmem(
        self,
        t_dkv: cute.Tensor,
        dkv_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        round_index: int,
        tile_index: Int32,
        topk: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        tidx: Int32,
        done_pipeline,
        consumer_state,
    ):
        """T2R the rank-owned D128 x N64 dKV and atomically accumulate."""

        math_state = consumer_state.clone()
        if tidx < self.MATH_THREADS_PER_CTA:
            done_pipeline.consumer_wait(math_state)
            tiled_t2r = tcgen05.make_tmem_copy(
                dkv_tmem_load,
                t_dkv,
            )
            thread_t2r = tiled_t2r.get_slice(tidx)
            thread_source = thread_t2r.partition_S(t_dkv)
            thread_coordinates = thread_t2r.partition_D(
                rank_coordinates
            )
            thread_values = cute.make_rmem_tensor(
                thread_coordinates.shape,
                self.acc_dtype,
            )
            cute.copy(tiled_t2r, thread_source, thread_values)
            cute.arch.fence_view_async_tmem_load()
            done_pipeline.consumer_release(math_state)

            for value_index in cutlass.range_constexpr(
                cute.size(thread_values)
            ):
                d_in_round = Int32(
                    cute.get(thread_coordinates[value_index], mode=[0])
                )
                n_index = Int32(
                    cute.get(thread_coordinates[value_index], mode=[1])
                )
                topk_slot = tile_index * self.N_TILE + n_index
                if topk_slot < topk:
                    kv_index = mTopkIdxs[
                        topk_slot,
                        (token_idx, batch_idx),
                    ]
                    if kv_index >= 0:
                        d_index = (
                            round_index * self.D_TILE_CLUSTER
                            + d_in_round
                        )
                        destination_ptr = (
                            mdKV_acc.iterator
                            + d_index * mdKV_acc.stride[0]
                            + kv_index * mdKV_acc.stride[1]
                        )
                        cute.arch.atomic_add(
                            destination_ptr.llvm_ptr,
                            thread_values[value_index],
                        )
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _store_dq_from_tmem(
        self,
        t_dq: cute.Tensor,
        dq_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        mdQ: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        token_idx: Int32,
        batch_idx: Int32,
        tidx: Int32,
    ):
        """Store one rank-owned D128 x H128 dQ.T slice as BF16."""

        if tidx < self.MATH_THREADS_PER_CTA:
            tiled_t2r = tcgen05.make_tmem_copy(dq_tmem_load, t_dq)
            thread_t2r = tiled_t2r.get_slice(tidx)
            thread_source = thread_t2r.partition_S(t_dq)
            thread_coordinates = thread_t2r.partition_D(
                rank_coordinates
            )
            thread_values = cute.make_rmem_tensor(
                thread_coordinates.shape,
                self.acc_dtype,
            )
            cute.copy(tiled_t2r, thread_source, thread_values)
            cute.arch.fence_view_async_tmem_load()
            for value_index in cutlass.range_constexpr(
                cute.size(thread_values)
            ):
                d_in_round = Int32(
                    cute.get(thread_coordinates[value_index], mode=[0])
                )
                head = Int32(
                    cute.get(thread_coordinates[value_index], mode=[1])
                )
                d_index = (
                    round_index * self.D_TILE_CLUSTER + d_in_round
                )
                mdQ[
                    d_index,
                    head,
                    (token_idx, batch_idx),
                ] = self.element_dtype(thread_values[value_index])

    @cute.jit
    def _stage_local_pd(
        self,
        p: cute.Tensor,
        ds: cute.Tensor,
        p_scratch: cute.Tensor,
        ds_scratch: cute.Tensor,
        rank: Int32,
        tidx: Int32,
    ):
        """Write each rank-local H64 x N64 input to simple N-major scratch."""

        linear_index = tidx
        while linear_index < self.H_TILE_CTA * self.N_TILE:
            local_h = linear_index // self.N_TILE
            n_index = linear_index % self.N_TILE
            scratch_offset = n_index * self.H_TILE_CTA + local_h
            p_scratch[scratch_offset] = p[rank, local_h, n_index]
            ds_scratch[scratch_offset] = ds[rank, local_h, n_index]
            linear_index += self.THREADS_PER_CTA

    @cute.jit
    def _exchange_peer_n32(
        self,
        source_scratch: cute.Tensor,
        inbox: cute.Tensor,
        remote_full: cute.Pointer,
        source_done: cute.Pointer,
        peer_rank: Int32,
        phase: Int32,
        tidx: Int32,
    ):
        """Send the peer-owned N32 half as one real 4096-byte S2CLUSTER."""

        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                remote_full,
                self.QUADRANT_BYTES,
                peer_cta_rank_in_cluster=peer_rank,
            )
            _cpasync_bulk_s2cluster(
                source_scratch.iterator
                + peer_rank * self.QUADRANT_ELEMENTS,
                inbox.iterator,
                remote_full,
                self.QUADRANT_BYTES,
                peer_rank,
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.mbarrier_arrive(source_done)

        cute.arch.mbarrier_wait(source_done, phase)
        if tidx == 0:
            cute.arch.mbarrier_wait(remote_full, phase)
        cute.arch.barrier()

    @cute.jit
    def _materialize_dkv_b(
        self,
        source_partition: cute.Tensor,
        coordinate_partition: cute.Tensor,
        destination: cute.Tensor,
        local_scratch: cute.Tensor,
        remote_inbox: cute.Tensor,
        rank: Int32,
        tidx: Int32,
    ):
        """Materialize one N32 x H128 nested B operand without raw offsets."""

        if tidx < self.MATH_THREADS_PER_CTA:
            for slot in cutlass.range_constexpr(32):
                linear_index = (
                    tidx + slot * self.MATH_THREADS_PER_CTA
                )
                source_coordinate = cute.idx2crd(
                    linear_index,
                    source_partition.shape,
                )
                logical_coordinate = coordinate_partition[
                    source_coordinate
                ]
                n_index = Int32(
                    cute.get(logical_coordinate, mode=[0])
                )
                head = Int32(cute.get(logical_coordinate, mode=[1]))
                local_h = head % self.H_TILE_CTA
                value = self.element_dtype(0.0)
                if head // self.H_TILE_CTA == rank:
                    value = local_scratch[
                        n_index * self.H_TILE_CTA + local_h
                    ]
                else:
                    value = remote_inbox[
                        (
                            n_index - rank * self.N_TILE_CTA
                        )
                        * self.H_TILE_CTA
                        + local_h
                    ]
                destination_coordinate = cute.idx2crd(
                    linear_index,
                    destination.shape,
                )
                destination[destination_coordinate] = value

    @cute.jit
    def _materialize_ds_both(
        self,
        dkv_source_partition: cute.Tensor,
        dkv_coordinate_partition: cute.Tensor,
        dkv_destination: cute.Tensor,
        dq_source_partition: cute.Tensor,
        dq_coordinate_partition: cute.Tensor,
        dq_destination: cute.Tensor,
        local_scratch: cute.Tensor,
        remote_inbox: cute.Tensor,
        rank: Int32,
        tidx: Int32,
    ):
        """Write exact dK and dQ nested B operands from disjoint scratch."""

        if tidx < self.MATH_THREADS_PER_CTA:
            for slot in cutlass.range_constexpr(32):
                linear_index = (
                    tidx + slot * self.MATH_THREADS_PER_CTA
                )

                dkv_source_coordinate = cute.idx2crd(
                    linear_index,
                    dkv_source_partition.shape,
                )
                dkv_logical_coordinate = dkv_coordinate_partition[
                    dkv_source_coordinate
                ]
                n_index = Int32(
                    cute.get(dkv_logical_coordinate, mode=[0])
                )
                head = Int32(
                    cute.get(dkv_logical_coordinate, mode=[1])
                )
                local_h = head % self.H_TILE_CTA
                dkv_value = self.element_dtype(0.0)
                if head // self.H_TILE_CTA == rank:
                    dkv_value = local_scratch[
                        n_index * self.H_TILE_CTA + local_h
                    ]
                else:
                    dkv_value = remote_inbox[
                        (
                            n_index - rank * self.N_TILE_CTA
                        )
                        * self.H_TILE_CTA
                        + local_h
                    ]
                dkv_destination_coordinate = cute.idx2crd(
                    linear_index,
                    dkv_destination.shape,
                )
                dkv_destination[dkv_destination_coordinate] = dkv_value

                dq_source_coordinate = cute.idx2crd(
                    linear_index,
                    dq_source_partition.shape,
                )
                dq_logical_coordinate = dq_coordinate_partition[
                    dq_source_coordinate
                ]
                head = Int32(
                    cute.get(dq_logical_coordinate, mode=[0])
                )
                n_index = Int32(
                    cute.get(dq_logical_coordinate, mode=[1])
                )
                dq_value = local_scratch[
                    n_index * self.H_TILE_CTA
                    + head % self.H_TILE_CTA
                ]
                dq_destination_coordinate = cute.idx2crd(
                    linear_index,
                    dq_destination.shape,
                )
                dq_destination[dq_destination_coordinate] = dq_value

    @cute.kernel
    def kernel(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_qt: cute.CopyAtom,
        tma_tensor_qt: cute.Tensor,
        tma_atom_dot: cute.CopyAtom,
        tma_tensor_dot: cute.Tensor,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
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
        dkv_a_layout_staged: cute.ComposedLayout,
        dkv_b_layout_staged: cute.ComposedLayout,
        dq_a_layout_staged: cute.ComposedLayout,
        dq_b_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        score_tmem_load: cute.CopyAtom,
        dkv_tmem_load: cute.CopyAtom,
        dq_tmem_load: cute.CopyAtom,
        tma_atom_dq_epi: cute.CopyAtom,
        tma_tensor_dq_epi: cute.Tensor,
        dq_epi_layout_staged: cute.ComposedLayout,
        score_a_stage_bytes: cutlass.Constexpr[int],
        grad_a_stage_bytes: cutlass.Constexpr[int],
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Run one complete Top-K traversal in a two-CTA cluster."""

        # The sequential checkpoint keeps its direct dQ stores.  These
        # launcher-built values are consumed by the v0 kernel override.
        _ = tma_atom_dq_epi
        _ = tma_tensor_dq_epi
        _ = dq_epi_layout_staged
        _ = trace_buffer
        _ = trace_token_idx
        _ = trace_batch_idx

        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        peer_rank = Int32(1) - rank
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        is_leader_cta = rank == 0
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_qt)
            cpasync.prefetch_descriptor(tma_atom_dot)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        source_done_mbars = storage.source_done_mbars.data_ptr()
        exchange_mbars = storage.exchange_mbars.data_ptr()
        done_mbars = storage.mma_full_empty_mbars.data_ptr()

        atom_thr_size = cute.size(score_tiled_mma.thr_id.shape)
        done_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.MMA_DONE_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                1,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                atom_thr_size * self.MATH_THREADS_PER_CTA,
            ),
            barrier_storage=done_mbars,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        done_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.MMA_DONE_STAGES,
        )
        done_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.MMA_DONE_STAGES,
        )
        s_score_q = storage.score_q.get_tensor(
            score_a_layout_staged.outer,
            swizzle=score_a_layout_staged.inner,
        )
        s_score_do = storage.score_do.get_tensor(
            score_a_layout_staged.outer,
            swizzle=score_a_layout_staged.inner,
        )
        s_score_kv = storage.score_kv.get_tensor(
            score_b_layout_staged.outer,
            swizzle=score_b_layout_staged.inner,
        )

        s_grad_a = cute.make_tensor(
            cute.recast_ptr(
                s_score_q.iterator,
                dkv_a_layout_staged.inner,
            ),
            dkv_a_layout_staged.outer,
        )
        s_grad_k = storage.kv_t.get_tensor(
            dq_a_layout_staged.outer,
            swizzle=dq_a_layout_staged.inner,
        )

        s_p = storage.p_t.get_tensor(
            dkv_b_layout_staged.outer,
            swizzle=dkv_b_layout_staged.inner,
        )
        s_ds_dk = storage.ds_dk_t.get_tensor(
            dkv_b_layout_staged.outer,
            swizzle=dkv_b_layout_staged.inner,
        )
        s_ds_dq = storage.ds_dq.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )

        scratch_layout = cute.make_layout(
            (self.H_TILE_CTA * self.N_TILE,),
            stride=(1,),
        )
        inbox_layout = cute.make_layout(
            (self.QUADRANT_ELEMENTS,),
            stride=(1,),
        )
        # Score operands are dead once S/dP have reached TMEM.  Reuse their
        # backing storage for the simple-layout P/dS exchange sources and
        # inbox, keeping those tensors disjoint from the nested MMA operand
        # destinations.  This also avoids thread-local cache arrays (and
        # their local-memory spills) during layout materialization.
        p_scratch = storage.score_q.get_tensor(scratch_layout)
        ds_scratch = storage.score_do.get_tensor(scratch_layout)
        bridge_inbox = storage.score_kv.get_tensor(inbox_layout)

        rank_score_mma = score_tiled_mma.get_slice(rank)
        rank_dp_mma = dp_tiled_mma.get_slice(rank)
        rank_dkv_mma = dkv_tiled_mma.get_slice(rank)
        rank_dq_mma = dq_tiled_mma.get_slice(rank)

        rank_score_coordinates = rank_score_mma.partition_C(
            cute.make_identity_tensor(
                (self.H_TILE_CLUSTER, self.N_TILE)
            )
        )
        rank_dkv_coordinates = rank_dkv_mma.partition_C(
            cute.make_identity_tensor(self.DKV_MMA_TILER[:2])
        )
        rank_dq_coordinates = rank_dq_mma.partition_C(
            cute.make_identity_tensor(self.DQ_MMA_TILER[:2])
        )

        dkv_b_identity = cute.local_tile(
            cute.make_identity_tensor(
                (self.N_TILE, self.H_TILE_CLUSTER)
            ),
            cute.select(self.DKV_MMA_TILER, mode=[1, 2]),
            (None, None),
        )
        dq_b_identity = cute.local_tile(
            cute.make_identity_tensor(
                (self.H_TILE_CLUSTER, self.N_TILE)
            ),
            cute.select(self.DQ_MMA_TILER, mode=[1, 2]),
            (None, None),
        )
        rank_dkv_b_coordinates = rank_dkv_mma.partition_B(
            dkv_b_identity
        )
        rank_dq_b_coordinates = rank_dq_mma.partition_B(
            dq_b_identity
        )
        dkv_b_coordinates = rank_dkv_b_coordinates[
            None,
            None,
            None,
            0,
            0,
        ]
        dq_b_coordinates = rank_dq_b_coordinates[
            None,
            None,
            None,
            0,
            0,
        ]

        g_q = cute.local_tile(
            tma_tensor_q,
            cute.select(
                (
                    self.H_TILE_CLUSTER,
                    self.N_TILE,
                    self.K_CHUNK,
                ),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        g_do = cute.local_tile(
            tma_tensor_do,
            cute.select(
                (
                    self.H_TILE_CLUSTER,
                    self.N_TILE,
                    self.K_CHUNK,
                ),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        rank_g_q = rank_score_mma.partition_A(g_q)
        rank_g_do = rank_dp_mma.partition_A(g_do)
        a_cta_layout = cute.make_layout(
            cute.slice_(
                cluster_layout_vmnk,
                (0, 0, None, 0),
            ).shape
        )
        t_q_smem, t_q_gmem = cpasync.tma_partition(
            tma_atom_q,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(s_score_q, 0, 3),
            cute.group_modes(rank_g_q, 0, 3),
        )
        t_do_smem, t_do_gmem = cpasync.tma_partition(
            tma_atom_do,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(s_score_do, 0, 3),
            cute.group_modes(rank_g_do, 0, 3),
        )
        # RestM is the unique H128 pack; preserve RestK for four D chunks.
        t_q_gmem = t_q_gmem[None, 0, None]
        t_do_gmem = t_do_gmem[None, 0, None]

        g_qt = cute.local_tile(
            tma_tensor_qt,
            cute.select(self.DKV_MMA_TILER, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        g_dot = cute.local_tile(
            tma_tensor_dot,
            cute.select(self.DKV_MMA_TILER, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        rank_g_qt = rank_dkv_mma.partition_A(g_qt)
        rank_g_dot = rank_dkv_mma.partition_A(g_dot)
        t_qt_smem, t_qt_gmem = cpasync.tma_partition(
            tma_atom_qt,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(s_grad_a, 0, 3),
            cute.group_modes(rank_g_qt, 0, 3),
        )
        t_dot_smem, t_dot_gmem = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(s_grad_a, 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )
        t_qt_gmem = t_qt_gmem[None, None, 0]
        t_dot_gmem = t_dot_gmem[None, None, 0]

        score_q_fragment = score_tiled_mma.make_fragment_A(s_score_q)
        score_kv_fragment = score_tiled_mma.make_fragment_B(s_score_kv)
        dp_do_fragment = dp_tiled_mma.make_fragment_A(s_score_do)
        dp_kv_fragment = dp_tiled_mma.make_fragment_B(s_score_kv)
        grad_a_fragment = dkv_tiled_mma.make_fragment_A(s_grad_a)
        p_fragment = dkv_tiled_mma.make_fragment_B(s_p)
        ds_dk_fragment = dkv_tiled_mma.make_fragment_B(s_ds_dk)
        grad_k_fragment = dq_tiled_mma.make_fragment_A(s_grad_k)
        ds_dq_fragment = dq_tiled_mma.make_fragment_B(s_ds_dq)

        kv_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(
                cache_mode=cpasync.LoadCacheMode.GLOBAL,
            ),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        kv_thread_copy = cute.make_tiled_copy_tv(
            kv_copy_atom,
            cute.make_layout((1,)),
            cute.make_layout((8,)),
        ).get_slice(0)

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk,
            is_relaxed=False,
        )
        pipeline.pipeline_init_wait(
            cluster_shape_mn=cluster_layout_vmnk
        )

        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        score_c_shape = score_tiled_mma.partition_shape_C(
            (self.H_TILE_CLUSTER, self.N_TILE)
        )
        score_c_layout = score_tiled_mma.make_fragment_C(
            score_c_shape
        ).layout
        dp_c_shape = dp_tiled_mma.partition_shape_C(
            (self.H_TILE_CLUSTER, self.N_TILE)
        )
        dp_c_layout = dp_tiled_mma.make_fragment_C(dp_c_shape).layout
        dkv_c_shape = dkv_tiled_mma.partition_shape_C(
            self.DKV_MMA_TILER[:2]
        )
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(
            dkv_c_shape
        ).layout
        dq_c_shape = dq_tiled_mma.partition_shape_C(
            self.DQ_MMA_TILER[:2]
        )
        dq_c_layout = dq_tiled_mma.make_fragment_C(dq_c_shape).layout

        t_score = cute.make_tensor(
            tmem_ptr + self.TMEM_S_OFFSET,
            score_c_layout,
        )
        t_dp = cute.make_tensor(
            tmem_ptr + self.TMEM_DP_OFFSET,
            dp_c_layout,
        )
        t_dkv = (
            cute.make_tensor(
                tmem_ptr + self.TMEM_DKV0_OFFSET,
                dkv_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr + self.TMEM_DKV1_OFFSET,
                dkv_c_layout,
            ),
        )
        t_dq = (
            cute.make_tensor(
                tmem_ptr + self.TMEM_DQ0_OFFSET,
                dq_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr + self.TMEM_DQ1_OFFSET,
                dq_c_layout,
            ),
        )

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = mTopkIdxs.shape[0]
        assert self.max_topk % self.N_TILE == 0
        tile_count = self.max_topk // self.N_TILE

        if warp_idx >= self.MATH_WARPS:
            cute.arch.setmaxregister_decrease(48)
        cute.arch.barrier()
        if warp_idx < self.MATH_WARPS:
            cute.arch.setmaxregister_increase(256)
        cute.arch.barrier()

        if tidx == 0:
            cute.arch.mbarrier_init(exchange_mbars, 1)
            cute.arch.mbarrier_init(exchange_mbars + 1, 1)
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # The benchmark specialization compiles one exact max_topk variant,
        # and its topk_length is the same full extent.  Keeping the traversal
        # constexpr also keeps tcgen05 TiledMma SSA updates in one dominating
        # region; sparse-short variants remain on the established fallback.
        for tile_ordinal in cutlass.range_constexpr(tile_count):
            tile_index = Int32(tile_count - 1 - tile_ordinal)
            first_tile = tile_ordinal == 0
            if tidx == 0:
                for barrier_index in range(
                    self.SCORE_SOURCE_BARRIERS
                ):
                    cute.arch.mbarrier_init(
                        source_done_mbars + barrier_index,
                        1,
                    )
            cute.arch.barrier()

            if warp_idx == 0:
                q_barrier = source_done_mbars
                do_barrier = source_done_mbars + 1
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        q_barrier,
                        score_a_stage_bytes * self.K_CHUNKS,
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        do_barrier,
                        score_a_stage_bytes * self.K_CHUNKS,
                    )
                for chunk in cutlass.range_constexpr(self.K_CHUNKS):
                    cute.copy(
                        tma_atom_q,
                        t_q_gmem[None, chunk],
                        t_q_smem[None, chunk],
                        tma_bar_ptr=q_barrier,
                    )
                    cute.copy(
                        tma_atom_do,
                        t_do_gmem[None, chunk],
                        t_do_smem[None, chunk],
                        tma_bar_ptr=do_barrier,
                    )

            if (
                tidx >= self.KV_LOAD_THREAD_BEGIN
                and tidx
                < self.KV_LOAD_THREAD_BEGIN + self.KV_LOAD_THREADS
            ):
                loader_tidx = tidx - self.KV_LOAD_THREAD_BEGIN
                self._load_score_kv(
                    mKV,
                    mTopkIdxs,
                    s_score_kv,
                    token_idx,
                    batch_idx,
                    tile_index,
                    topk,
                    rank,
                    loader_tidx,
                    kv_copy_atom,
                    kv_thread_copy,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
            for barrier_index in range(self.SCORE_SOURCE_BARRIERS):
                cute.arch.mbarrier_wait(
                    source_done_mbars + barrier_index,
                    Int32(0),
                )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            if is_leader_cta and warp_idx == 0:
                done_producer_state = self._issue_four_chunks(
                    score_tiled_mma,
                    t_score,
                    score_q_fragment,
                    score_kv_fragment,
                    done_pipeline,
                    done_producer_state,
                )
                done_producer_state = self._issue_four_chunks(
                    dp_tiled_mma,
                    t_dp,
                    dp_do_fragment,
                    dp_kv_fragment,
                    done_pipeline,
                    done_producer_state,
                )

            done_consumer_state = self._compute_pd_from_tmem(
                t_score,
                t_dp,
                score_tmem_load,
                rank_score_coordinates,
                scaled_lse,
                sum_odo,
                p_scratch,
                ds_scratch,
                scale_softmax,
                token_idx,
                batch_idx,
                tidx,
                done_pipeline,
                done_consumer_state,
            )

            self._exchange_peer_n32(
                p_scratch,
                bridge_inbox,
                exchange_mbars,
                exchange_mbars + 1,
                peer_rank,
                Int32(0),
                tidx,
            )
            self._materialize_dkv_b(
                dkv_b_coordinates,
                dkv_b_coordinates,
                s_p[None, None, None, 0],
                p_scratch,
                bridge_inbox,
                rank,
                tidx,
            )
            # P and dS share the remote inbox.  Join all local readers, then
            # join the CTA pair before either peer overwrites the inbox with
            # the dS exchange generation.
            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            self._exchange_peer_n32(
                ds_scratch,
                bridge_inbox,
                exchange_mbars,
                exchange_mbars + 1,
                peer_rank,
                Int32(1),
                tidx,
            )
            self._materialize_ds_both(
                dkv_b_coordinates,
                dkv_b_coordinates,
                s_ds_dk[None, None, None, 0],
                dq_b_coordinates,
                dq_b_coordinates,
                s_ds_dq[None, None, None, 0],
                ds_scratch,
                bridge_inbox,
                rank,
                tidx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            for round_index in cutlass.range_constexpr(self.D_ROUNDS):
                if tidx == 0:
                    cute.arch.mbarrier_init(source_done_mbars, 1)
                cute.arch.barrier()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            source_done_mbars,
                            grad_a_stage_bytes,
                        )
                    cute.copy(
                        tma_atom_dot,
                        t_dot_gmem[None, round_index],
                        t_dot_smem[None, 0],
                        tma_bar_ptr=source_done_mbars,
                    )
                if (
                    tidx >= self.KV_LOAD_THREAD_BEGIN
                    and tidx
                    < self.KV_LOAD_THREAD_BEGIN + self.KV_LOAD_THREADS
                ):
                    loader_tidx = tidx - self.KV_LOAD_THREAD_BEGIN
                    self._load_grad_k(
                        mKV,
                        mTopkIdxs,
                        s_grad_k,
                        token_idx,
                        batch_idx,
                        tile_index,
                        topk,
                        Int32(round_index),
                        rank,
                        loader_tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_wait(
                    source_done_mbars,
                    Int32(0),
                )
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                if is_leader_cta and warp_idx == 0:
                    done_producer_state = self._issue_dv(
                        dkv_tiled_mma,
                        t_dkv[round_index],
                        grad_a_fragment,
                        p_fragment,
                        done_pipeline,
                        done_producer_state,
                    )
                done_consumer_state = self._wait_mma(
                    done_pipeline,
                    done_consumer_state,
                    tidx,
                )

                if tidx == 0:
                    cute.arch.mbarrier_init(source_done_mbars, 1)
                cute.arch.barrier()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            source_done_mbars,
                            grad_a_stage_bytes,
                        )
                    cute.copy(
                        tma_atom_qt,
                        t_qt_gmem[None, round_index],
                        t_qt_smem[None, 0],
                        tma_bar_ptr=source_done_mbars,
                    )
                cute.arch.mbarrier_wait(
                    source_done_mbars,
                    Int32(0),
                )
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                if is_leader_cta and warp_idx == 0:
                    done_producer_state = self._issue_dk(
                        dkv_tiled_mma,
                        t_dkv[round_index],
                        grad_a_fragment,
                        ds_dk_fragment,
                        done_pipeline,
                        done_producer_state,
                    )
                done_consumer_state = self._atomic_dkv_from_tmem(
                    t_dkv[round_index],
                    dkv_tmem_load,
                    rank_dkv_coordinates,
                    mdKV_acc,
                    mTopkIdxs,
                    round_index,
                    tile_index,
                    topk,
                    token_idx,
                    batch_idx,
                    tidx,
                    done_pipeline,
                    done_consumer_state,
                )

                if is_leader_cta and warp_idx == 0:
                    done_producer_state = self._issue_dq(
                        dq_tiled_mma,
                        t_dq[round_index],
                        grad_k_fragment,
                        ds_dq_fragment,
                        not first_tile,
                        done_pipeline,
                        done_producer_state,
                    )
                done_consumer_state = self._wait_mma(
                    done_pipeline,
                    done_consumer_state,
                    tidx,
                )
                cute.arch.barrier()

        tmem.relinquish_alloc_permit()
        self._store_dq_from_tmem(
            t_dq[0],
            dq_tmem_load,
            rank_dq_coordinates,
            mdQ,
            0,
            token_idx,
            batch_idx,
            tidx,
        )
        self._store_dq_from_tmem(
            t_dq[1],
            dq_tmem_load,
            rank_dq_coordinates,
            mdQ,
            1,
            token_idx,
            batch_idx,
            tidx,
        )

        cute.arch.barrier()
        if is_leader_cta and warp_idx == 0:
            done_pipeline.producer_tail(done_producer_state)
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)


# The host decoder mirrors this fixed, collision-free address layout.
# A role leader owns every address in its lane, so tracing needs neither a
# global append counter nor an atomic operation.
TRACE_HEADER_WORDS = 16
TRACE_ROLE_COUNT = 10
TRACE_ISSUE_SLOTS = 33
TRACE_EVENT_SLOTS_PER_ISSUE = 256
TRACE_VERSION = 1

TRACE_ROLE_CONTROL = 0
TRACE_ROLE_GATHER = 1
TRACE_ROLE_LOAD = 2
TRACE_ROLE_DESC_BQ = 3
TRACE_ROLE_MMA = 4
TRACE_ROLE_MATH = 5
TRACE_ROLE_XCHG = 6
TRACE_ROLE_REDUCE_R0 = 7
TRACE_ROLE_REDUCE_R1 = 8
TRACE_ROLE_DQ_EPI = 9

TRACE_F_LOAD_BEGIN = 0
TRACE_F_LOAD_END = 1
TRACE_BV_LOAD_BEGIN = 2
TRACE_BV_LOAD_END = 3
TRACE_BQ_WAIT_BEGIN = 4
TRACE_BQ_WAIT_END = 5
TRACE_BQ_LOAD_BEGIN = 6
TRACE_BQ_LOAD_END = 7
TRACE_DESC_BEGIN = 8
TRACE_DESC_END = 9
TRACE_SDP_BEGIN = 10
TRACE_SDP_END = 11
TRACE_GRAD_BEGIN = 12
TRACE_GRAD_END = 13
TRACE_S_WAIT_BEGIN = 14
TRACE_S_WAIT_END = 15
TRACE_S_T2R_BEGIN = 16
TRACE_S_T2R_END = 17
TRACE_DP_WAIT_BEGIN = 18
TRACE_DP_WAIT_END = 19
TRACE_DP_T2R_BEGIN = 20
TRACE_DP_T2R_END = 21
TRACE_PD_ACQUIRE_BEGIN = 22
TRACE_PD_ACQUIRE_END = 23
TRACE_MATH_BEGIN = 24
TRACE_MATH_END = 25
TRACE_REMOTE_WAIT_BEGIN = 26
TRACE_REMOTE_WAIT_END = 27
TRACE_PD_PUBLISH = 28
TRACE_XCHG_WAIT_BEGIN = 29
TRACE_XCHG_WAIT_END = 30
TRACE_XCHG_BEGIN = 31
TRACE_XCHG_ISSUED = 32
TRACE_XCHG_SOURCE_DONE = 33
TRACE_REDUCE_WAIT_BEGIN = 34
TRACE_REDUCE_WAIT_END = 35
TRACE_REDUCE_T2R_BEGIN = 36
TRACE_REDUCE_T2R_END = 37
TRACE_REDUCE_ATOMIC_BEGIN = 38
TRACE_REDUCE_ATOMIC_END = 39
TRACE_DQ_WAIT_BEGIN = 40
TRACE_DQ_WAIT_END = 41
TRACE_DQ_T2R_BEGIN = 42
TRACE_DQ_T2R_END = 43
TRACE_DQ_STORE_BEGIN = 44
TRACE_DQ_STORE_END = 45
TRACE_PRE_EPI_JOIN_BEGIN = 46
TRACE_PRE_EPI_JOIN_END = 47
TRACE_FINAL_JOIN_BEGIN = 48
TRACE_FINAL_JOIN_END = 49
TRACE_STREAM_DONE = 50
TRACE_CTX_COMMIT = 51


@dsl_user_op
def _read_global_timer(*, loc=None, ip=None) -> cutlass.Int64:
    """Read the GPU-wide nanosecond timer used to align both cluster ranks."""

    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _read_smid(*, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _trace_stamp(
    trace_buffer: Optional[cute.Tensor],
    token_idx: Int32,
    batch_idx: Int32,
    trace_token_idx: Int32,
    trace_batch_idx: Int32,
    rank: Int32,
    role: cutlass.Constexpr[int],
    issue_seq: Int32,
    tag: cutlass.Constexpr[int],
    sub_index: cutlass.Constexpr[int] = 0,
) -> None:
    """Write one statically addressed timestamp from an elected role thread."""

    if cutlass.const_expr(trace_buffer is not None):
        if token_idx == trace_token_idx:
            if batch_idx == trace_batch_idx:
                if issue_seq >= Int32(0):
                    if issue_seq < Int32(TRACE_ISSUE_SLOTS):
                        record_index = (
                            Int32(TRACE_HEADER_WORDS)
                            + (
                                (
                                    rank * Int32(TRACE_ROLE_COUNT)
                                    + Int32(role)
                                )
                                * Int32(TRACE_ISSUE_SLOTS)
                                + issue_seq
                            )
                            * Int32(TRACE_EVENT_SLOTS_PER_ISSUE)
                            + Int32(tag * 4 + sub_index)
                        )
                        trace_buffer[record_index] = (
                            _read_global_timer()
                        )


@cute.jit
def _trace_header_begin(
    trace_buffer: Optional[cute.Tensor],
    token_idx: Int32,
    batch_idx: Int32,
    trace_token_idx: Int32,
    trace_batch_idx: Int32,
    rank: Int32,
) -> None:
    if cutlass.const_expr(trace_buffer is not None):
        if token_idx == trace_token_idx:
            if batch_idx == trace_batch_idx:
                base = rank * Int32(4)
                trace_buffer[base] = cutlass.Int64(TRACE_VERSION)
                trace_buffer[base + Int32(1)] = cutlass.Int64(
                    _read_smid()
                )
                trace_buffer[base + Int32(2)] = _read_global_timer()


@cute.jit
def _trace_header_end(
    trace_buffer: Optional[cute.Tensor],
    token_idx: Int32,
    batch_idx: Int32,
    trace_token_idx: Int32,
    trace_batch_idx: Int32,
    rank: Int32,
) -> None:
    if cutlass.const_expr(trace_buffer is not None):
        if token_idx == trace_token_idx:
            if batch_idx == trace_batch_idx:
                trace_buffer[
                    rank * Int32(4) + Int32(3)
                ] = _read_global_timer()


@dsl_user_op
def _atomic_and_shared_i32(
    pointer: cute.Pointer,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Atomically clear one reducer-context pending bit in CTA SMEM."""

    pointer_i32 = pointer.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        T.i32(),
        [pointer_i32, value.ir_value(loc=loc, ip=ip)],
        "atom.shared::cta.and.b32 $0, [$1], $2;",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(result)


@dsl_user_op
def _mbarrier_try_wait(
    barrier: cute.Pointer,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Boolean:
    """Poll one CTA-shared mbarrier generation without blocking a role."""

    barrier_i32 = barrier.toint(loc=loc, ip=ip).ir_value()
    ready = llvm.inline_asm(
        T.i32(),
        [barrier_i32, phase.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "p, [$1], $2, 1;\n\t"
        "selp.u32 $0, 1, 0, p;\n\t"
        "}",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(ready) != Int32(0)


@cute.jit
def _dkv_partition_coord(local_n: Int32, global_h: Int32):
    return (
        (local_n, global_h % Int32(16)),
        Int32(0),
        global_h // Int32(16),
    )


@cute.jit
def _dq_partition_coord(local_h: Int32, n_index: Int32):
    return (
        (local_h, n_index % Int32(16)),
        Int32(0),
        n_index // Int32(16),
    )


@cute.jit
def _compute_and_store_pd(
    owner: cutlass.Constexpr[object],
    r_score: cute.Tensor,
    r_dp: cute.Tensor,
    softmax_stats: cute.Tensor,
    valid_lo: Int32,
    valid_hi: Int32,
    math_tidx: Int32,
    softmax_scale_log2_e: Float32,
    scale_softmax: Float32,
    r_p: cute.Tensor,
    r_dsq: cute.Tensor,
    apply_mask: cutlass.Constexpr[bool],
) -> None:
    """Compute one score-distributed BF16 P/dS fragment in registers."""

    # For the CG2 H128xN64 accumulator copied by W8-W11, each thread owns
    # one rank-local H row and one contiguous N32 quadrant:
    #
    #   local_h = math_tidx % 64
    #   n_owner = math_tidx // 64
    #   local_n = value_index
    #
    # This mapping has been exhaustively validated over the H64xN64 rank
    # tile.  Keep the x4 FP32/BF16 fragments and the three scalar destinations,
    # but remove per-value identity-coordinate decoding and address rebuilds.
    local_h = math_tidx % Int32(owner.H_TILE_CTA)
    n_owner = math_tidx // Int32(owner.H_TILE_CTA)

    lse = softmax_stats[local_h, 0]
    sum_odo = softmax_stats[local_h, 1]
    if cutlass.const_expr(apply_mask):
        valid_bits = valid_lo
        if n_owner != Int32(0):
            valid_bits = valid_hi

    p_fp32 = cute.make_rmem_tensor((4,), owner.acc_dtype)
    ds_fp32 = cute.make_rmem_tensor((4,), owner.acc_dtype)
    p_bf16 = cute.make_rmem_tensor((4,), owner.element_dtype)
    ds_bf16 = cute.make_rmem_tensor((4,), owner.element_dtype)

    for fragment_index in cutlass.range_constexpr(
        cute.size(r_score) // 4
    ):
        for pair_index in cutlass.range_constexpr(2):
            value_index_0 = fragment_index * 4 + pair_index * 2
            value_index_1 = value_index_0 + 1
            pair_offset = pair_index * 2
            lse_0 = lse
            lse_1 = lse
            if cutlass.const_expr(apply_mask):
                is_valid_0 = (
                    (
                        valid_bits >> Int32(value_index_0)
                    )
                    & Int32(1)
                ) != Int32(0)
                is_valid_1 = (
                    (
                        valid_bits >> Int32(value_index_1)
                    )
                    & Int32(1)
                ) != Int32(0)
                if not is_valid_0:
                    lse_0 = Float32(float("-inf"))
                if not is_valid_1:
                    lse_1 = Float32(float("-inf"))

            p_0, p_1 = cute.arch.fma_packed_f32x2(
                (
                    r_score[value_index_0],
                    r_score[value_index_1],
                ),
                (
                    softmax_scale_log2_e,
                    softmax_scale_log2_e,
                ),
                (lse_0, lse_1),
            )
            p_0 = cute.math.exp2(p_0, fastmath=True)
            p_1 = cute.math.exp2(p_1, fastmath=True)

            ds_0, ds_1 = cute.arch.add_packed_f32x2(
                (
                    r_dp[value_index_0],
                    r_dp[value_index_1],
                ),
                (
                    sum_odo,
                    sum_odo,
                ),
            )
            ds_0, ds_1 = cute.arch.mul_packed_f32x2(
                (ds_0, ds_1),
                (p_0, p_1),
            )
            ds_0, ds_1 = cute.arch.mul_packed_f32x2(
                (ds_0, ds_1),
                (scale_softmax, scale_softmax),
            )
            if cutlass.const_expr(apply_mask):
                if not is_valid_0:
                    p_0 = Float32(0.0)
                    ds_0 = Float32(0.0)
                if not is_valid_1:
                    p_1 = Float32(0.0)
                    ds_1 = Float32(0.0)
            p_fp32[pair_offset] = p_0
            p_fp32[pair_offset + 1] = p_1
            ds_fp32[pair_offset] = ds_0
            ds_fp32[pair_offset + 1] = ds_1

        p_bf16.store(p_fp32.load().to(owner.element_dtype))
        ds_bf16.store(ds_fp32.load().to(owner.element_dtype))
        for fragment_value in cutlass.range_constexpr(4):
            value_index = fragment_index * 4 + fragment_value
            r_p[value_index] = p_bf16[fragment_value]
            r_dsq[value_index] = ds_bf16[fragment_value]


@cute.jit
def _issue_exchange(
    owner: cutlass.Constexpr[object],
    source: cute.Tensor,
    destination_stage: cute.Pointer,
    remote_full: cute.Pointer,
    destination_quadrant: Int32,
    peer_rank: Int32,
) -> None:
    """Copy one pre-swizzled 4-KiB image into the peer final quadrant."""

    destination = (
        destination_stage
        + destination_quadrant * owner.XCHG_ELEMENTS
    )
    cute.arch.mbarrier_arrive_and_expect_tx(
        remote_full,
        owner.QUADRANT_BYTES,
        peer_cta_rank_in_cluster=peer_rank,
    )
    _cpasync_bulk_s2cluster(
        source.iterator,
        destination,
        remote_full,
        owner.QUADRANT_BYTES,
        peer_rank,
    )


@cute.jit
def _run_exchange_role(
    self,
    rank: Int32,
    issued_stream_state: cute.Tensor,
    issued_stream_done_mbars: cute.Pointer,
    raw_p_dv: cute.Tensor,
    raw_ds_dk: cute.Tensor,
    raw_p_xchg: cute.Tensor,
    raw_ds_xchg: cute.Tensor,
    p_local_ready: cute.Pointer,
    ds_local_ready: cute.Pointer,
    p_remote_full: cute.Pointer,
    ds_remote_full: cute.Pointer,
    p_source_done: cute.Pointer,
    ds_source_done: cute.Pointer,
    token_idx: Int32,
    batch_idx: Int32,
    trace_buffer: Optional[cute.Tensor],
    trace_token_idx: Int32,
    trace_batch_idx: Int32,
) -> None:
    """Have W6 send the two fixed 4-KiB peer quadrants for each tile."""

    peer = Int32(1) - rank
    issue_seq = Int32(0)
    self._record_trace(
        trace_buffer,
        token_idx,
        batch_idx,
        trace_token_idx,
        trace_batch_idx,
        rank,
        TRACE_ROLE_XCHG,
        issue_seq,
        TRACE_XCHG_WAIT_BEGIN,
    )
    active = self._resolve_pd_tile_or_done(
        issue_seq,
        p_local_ready,
        issued_stream_state,
        issued_stream_done_mbars,
    )
    while active:
        stage = issue_seq % Int32(self.PD_STAGES)
        phase = (
            issue_seq // Int32(self.PD_STAGES)
        ) & Int32(1)
        cute.arch.mbarrier_wait(
            ds_local_ready + stage,
            Int32(phase),
        )
        self._record_trace(
            trace_buffer,
            token_idx,
            batch_idx,
            trace_token_idx,
            trace_batch_idx,
            rank,
            TRACE_ROLE_XCHG,
            issue_seq,
            TRACE_XCHG_WAIT_END,
        )
        self._record_trace(
            trace_buffer,
            token_idx,
            batch_idx,
            trace_token_idx,
            trace_batch_idx,
            rank,
            TRACE_ROLE_XCHG,
            issue_seq,
            TRACE_XCHG_BEGIN,
        )
        _issue_exchange(
            self,
            raw_p_xchg,
            raw_p_dv.iterator
            + stage * self.PD_NESTED_ELEMENTS_PER_STAGE,
            p_remote_full + stage,
            rank,
            peer,
        )
        _issue_exchange(
            self,
            raw_ds_xchg,
            raw_ds_dk.iterator
            + stage * self.PD_NESTED_ELEMENTS_PER_STAGE,
            ds_remote_full + stage,
            rank,
            peer,
        )
        cute.arch.cp_async_bulk_commit_group()
        self._record_trace(
            trace_buffer,
            token_idx,
            batch_idx,
            trace_token_idx,
            trace_batch_idx,
            rank,
            TRACE_ROLE_XCHG,
            issue_seq,
            TRACE_XCHG_ISSUED,
        )
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.mbarrier_arrive(p_source_done)
        cute.arch.mbarrier_arrive(ds_source_done)
        self._record_trace(
            trace_buffer,
            token_idx,
            batch_idx,
            trace_token_idx,
            trace_batch_idx,
            rank,
            TRACE_ROLE_XCHG,
            issue_seq,
            TRACE_XCHG_SOURCE_DONE,
        )
        issue_seq += Int32(1)
        self._record_trace(
            trace_buffer,
            token_idx,
            batch_idx,
            trace_token_idx,
            trace_batch_idx,
            rank,
            TRACE_ROLE_XCHG,
            issue_seq,
            TRACE_XCHG_WAIT_BEGIN,
        )
        active = self._resolve_pd_tile_or_done(
            issue_seq,
            p_local_ready,
            issued_stream_state,
            issued_stream_done_mbars,
        )


@cute.jit
def _run_math_role(
    self,
    math_barrier,
    tidx: Int32,
    rank: Int32,
    token_idx: Int32,
    batch_idx: Int32,
    issued_ctx: cute.Tensor,
    issued_stream_state: cute.Tensor,
    issued_stream_done_mbars: cute.Pointer,
    reducer_ctx: cute.Tensor,
    t_score: cute.Tensor,
    t_dp: cute.Tensor,
    score_tmem_load: cute.CopyAtom,
    rank_score_coordinates: cute.Tensor,
    scaled_lse: cute.Tensor,
    sum_odo: cute.Tensor,
    softmax_stats: cute.Tensor,
    scale_softmax: Float32,
    s_pipeline,
    dp_pipeline,
    p_dv_pipeline,
    ds_dk_pipeline,
    ds_dq_pipeline,
    raw_p_dv: cute.Tensor,
    raw_ds_dk: cute.Tensor,
    raw_ds_dq: cute.Tensor,
    raw_p_xchg: cute.Tensor,
    raw_ds_xchg: cute.Tensor,
    dkv_b_layout: cute.ComposedLayout,
    dq_b_layout: cute.ComposedLayout,
    score_store_layout: cute.ComposedLayout,
    score_store_domain: cute.Layout,
    p_local_ready: cute.Pointer,
    ds_local_ready: cute.Pointer,
    p_remote_full: cute.Pointer,
    ds_remote_full: cute.Pointer,
    p_source_done: cute.Pointer,
    ds_source_done: cute.Pointer,
    issued_ctx_mbars: cute.Pointer,
    reducer_ctx_mbars: cute.Pointer,
    ctx_reader_done_mbars: cute.Pointer,
    trace_buffer: Optional[cute.Tensor],
    trace_token_idx: Int32,
    trace_batch_idx: Int32,
) -> None:
    """T2R one final S/dP generation, compute FP32 P/dS, and publish it."""

    math_tidx = tidx - self.MATH_WARPS[0] * 32
    is_math_leader = math_tidx == 0
    peer = Int32(1) - rank

    score_copy = tcgen05.make_tmem_copy(
        score_tmem_load,
        t_score,
    )
    score_thread = score_copy.get_slice(math_tidx)
    score_source = score_thread.partition_S(t_score)
    score_coordinates = score_thread.partition_D(
        rank_score_coordinates
    )
    r_score = cute.make_rmem_tensor(
        score_coordinates.shape,
        self.acc_dtype,
    )
    r_p = cute.make_rmem_tensor(
        score_coordinates.shape,
        self.element_dtype,
    )
    r_dsq = cute.make_rmem_tensor(
        score_coordinates.shape,
        self.element_dtype,
    )
    smem_store_atom = sm100_utils.get_smem_store_op(
        utils.LayoutEnum.COL_MAJOR,
        self.element_dtype,
        self.acc_dtype,
        score_copy,
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(
        smem_store_atom,
        score_copy,
    )
    thread_copy_r2s = tiled_copy_r2s.get_slice(math_tidx)
    r_p_store = thread_copy_r2s.retile(r_p)
    r_dsq_store = thread_copy_r2s.retile(r_dsq)
    dp_copy = tcgen05.make_tmem_copy(
        score_tmem_load,
        t_dp,
    )
    dp_thread = dp_copy.get_slice(math_tidx)
    dp_source = dp_thread.partition_S(t_dp)
    r_dp = cute.make_rmem_tensor(
        score_coordinates.shape,
        self.acc_dtype,
    )

    s_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        1,
    )
    dp_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        1,
    )
    p_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        self.PD_STAGES,
    )
    dsk_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        self.PD_STAGES,
    )
    dsq_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        self.PD_STAGES,
    )

    softmax_scale_log2_e = (
        scale_softmax * Float32(math.log2(math.e))
    )

    # Stats depend only on (rank-local head, token, batch), not on the
    # sparse tile. W8 moves both H64 vectors once with 64-bit cp.async;
    # the remaining math warps join only at the 128-thread named barrier.
    stats_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(
            cache_mode=cpasync.LoadCacheMode.ALWAYS
        ),
        self.acc_dtype,
        num_bits_per_copy=64,
    )
    stats_copy = cute.make_tiled_copy_tv(
        stats_atom,
        cute.make_layout((32,), stride=(1,)),
        cute.make_layout((2,), stride=(1,)),
    )
    stats_thread = stats_copy.get_slice(
        math_tidx % Int32(32)
    )
    g_lse = cute.flat_divide(
        scaled_lse,
        (self.H_TILE_CTA,),
    )
    g_sum_odo = cute.flat_divide(
        sum_odo,
        (self.H_TILE_CTA,),
    )
    t_g_lse = stats_thread.partition_S(
        g_lse[None, rank, (token_idx, batch_idx)]
    )
    t_g_sum_odo = stats_thread.partition_S(
        g_sum_odo[None, rank, (token_idx, batch_idx)]
    )
    t_s_lse = stats_thread.partition_D(
        softmax_stats[None, 0]
    )
    t_s_sum_odo = stats_thread.partition_D(
        softmax_stats[None, 1]
    )
    if math_tidx < Int32(32):
        cute.copy(
            stats_atom,
            t_g_lse[None, 0],
            t_s_lse[None, 0],
        )
        cute.copy(
            stats_atom,
            t_g_sum_odo[None, 0],
            t_s_sum_odo[None, 0],
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
    math_barrier.arrive_and_wait()
    cute.arch.fence_view_async_shared()

    issue_seq = Int32(0)
    active = self._resolve_issued_context_or_done(
        issue_seq,
        issued_ctx_mbars,
        issued_stream_state,
        issued_stream_done_mbars,
    )
    while active:
        context_slot = issue_seq % Int32(self.CONTEXT_STAGES)

        # Exactly one final S/dP T2R and release per logical SDP tile.
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_S_WAIT_BEGIN,
            )
            s_pipeline.consumer_wait(s_state)
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_S_WAIT_END,
            )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_S_T2R_BEGIN,
            )
        math_barrier.arrive_and_wait()
        cute.copy(score_copy, score_source, r_score)
        cute.arch.fence_view_async_tmem_load()
        math_barrier.arrive_and_wait()
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_S_T2R_END,
            )
            s_pipeline.consumer_release(s_state)
        s_state.advance()

        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_DP_WAIT_BEGIN,
            )
            dp_pipeline.consumer_wait(dp_state)
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_DP_WAIT_END,
            )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_DP_T2R_BEGIN,
            )
        math_barrier.arrive_and_wait()
        cute.copy(dp_copy, dp_source, r_dp)
        cute.arch.fence_view_async_tmem_load()
        math_barrier.arrive_and_wait()
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_DP_T2R_END,
            )
            dp_pipeline.consumer_release(dp_state)
        dp_state.advance()

        stage = issue_seq % Int32(self.PD_STAGES)
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_PD_ACQUIRE_BEGIN,
            )
            p_dv_pipeline.producer_acquire(p_state)
            ds_dk_pipeline.producer_acquire(dsk_state)
            ds_dq_pipeline.producer_acquire(dsq_state)
            if issue_seq > Int32(0):
                previous_phase = (
                    issue_seq - Int32(1)
                ) & Int32(1)
                cute.arch.mbarrier_wait(
                    p_source_done,
                    Int32(previous_phase),
                )
                cute.arch.mbarrier_wait(
                    ds_source_done,
                    Int32(previous_phase),
                )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_PD_ACQUIRE_END,
            )
        math_barrier.arrive_and_wait()
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_MATH_BEGIN,
            )

        p_stage = self._make_pd_stage_view(
            raw_p_dv,
            stage,
            self.PD_NESTED_ELEMENTS_PER_STAGE,
            dkv_b_layout,
        )
        dsk_stage = self._make_pd_stage_view(
            raw_ds_dk,
            stage,
            self.PD_NESTED_ELEMENTS_PER_STAGE,
            dkv_b_layout,
        )
        dsq_stage = self._make_pd_stage_view(
            raw_ds_dq,
            stage,
            self.PD_LOCAL_ELEMENTS_PER_STAGE,
            dq_b_layout,
        )
        valid_lo = Int32(
            issued_ctx[
                self.CTX_VALID_LO_WORD,
                context_slot,
            ]
        )
        valid_hi = Int32(
            issued_ctx[
                self.CTX_VALID_HI_WORD,
                context_slot,
            ]
        )
        all_valid = (
            (valid_lo & valid_hi) == Int32(-1)
        )
        if all_valid:
            _compute_and_store_pd(
                self,
                r_score,
                r_dp,
                softmax_stats,
                valid_lo,
                valid_hi,
                math_tidx,
                softmax_scale_log2_e,
                scale_softmax,
                r_p,
                r_dsq,
                False,
            )
        else:
            _compute_and_store_pd(
                self,
                r_score,
                r_dp,
                softmax_stats,
                valid_lo,
                valid_hi,
                math_tidx,
                softmax_scale_log2_e,
                scale_softmax,
                r_p,
                r_dsq,
                True,
            )

        # The score fragment is H64xN64 with byte image h + n*64 under the
        # same S<3,4,3> swizzle as one dKV B-operand quadrant.  W8-W9 own the
        # low N32 half and W10-W11 own the high N32 half, so each warp can
        # uniformly choose its final-stage or standalone exchange destination.
        n_owner = cute.arch.make_warp_uniform(
            math_tidx // Int32(self.H_TILE_CTA)
        )
        owns_n = n_owner == rank
        aligned_p_ptr = cute.make_ptr(
            self.element_dtype,
            p_stage.iterator.toint(),
            p_stage.memspace,
            assumed_align=16,
        )
        aligned_dsk_ptr = cute.make_ptr(
            self.element_dtype,
            dsk_stage.iterator.toint(),
            dsk_stage.memspace,
            assumed_align=16,
        )
        p_local_store = cute.make_tensor(
            cute.recast_ptr(
                aligned_p_ptr,
                score_store_layout.inner,
                dtype=self.element_dtype,
            ),
            score_store_domain,
        )
        dsk_local_store = cute.make_tensor(
            cute.recast_ptr(
                aligned_dsk_ptr,
                score_store_layout.inner,
                dtype=self.element_dtype,
            ),
            score_store_domain,
        )
        p_xchg_store = cute.make_tensor(
            cute.recast_ptr(
                raw_p_xchg.iterator
                - n_owner * self.XCHG_ELEMENTS,
                score_store_layout.inner,
                dtype=self.element_dtype,
            ),
            score_store_domain,
        )
        ds_xchg_store = cute.make_tensor(
            cute.recast_ptr(
                raw_ds_xchg.iterator
                - n_owner * self.XCHG_ELEMENTS,
                score_store_layout.inner,
                dtype=self.element_dtype,
            ),
            score_store_domain,
        )
        t_rs_p_local = thread_copy_r2s.partition_D(
            p_local_store
        )
        t_rs_dsk_local = thread_copy_r2s.partition_D(
            dsk_local_store
        )
        t_rs_p_xchg = thread_copy_r2s.partition_D(
            p_xchg_store
        )
        t_rs_ds_xchg = thread_copy_r2s.partition_D(
            ds_xchg_store
        )
        assert cute.size(t_rs_p_local, mode=[4]) == 1
        assert cute.size(t_rs_dsk_local, mode=[4]) == 1
        assert cute.size(t_rs_p_xchg, mode=[4]) == 1
        assert cute.size(t_rs_ds_xchg, mode=[4]) == 1
        t_rs_p_local_tile = t_rs_p_local[
            None, None, None, None, 0
        ]
        t_rs_dsk_local_tile = t_rs_dsk_local[
            None, None, None, None, 0
        ]
        t_rs_p_xchg_tile = t_rs_p_xchg[
            None, None, None, None, 0
        ]
        t_rs_ds_xchg_tile = t_rs_ds_xchg[
            None, None, None, None, 0
        ]
        assert t_rs_p_local_tile.shape == r_p_store.shape
        assert t_rs_dsk_local_tile.shape == r_dsq_store.shape
        assert t_rs_p_xchg_tile.shape == r_p_store.shape
        assert t_rs_ds_xchg_tile.shape == r_dsq_store.shape
        if owns_n:
            cute.copy(
                tiled_copy_r2s,
                r_p_store,
                t_rs_p_local_tile,
            )
            cute.copy(
                tiled_copy_r2s,
                r_dsq_store,
                t_rs_dsk_local_tile,
            )
        else:
            cute.copy(
                tiled_copy_r2s,
                r_p_store,
                t_rs_p_xchg_tile,
            )
            cute.copy(
                tiled_copy_r2s,
                r_dsq_store,
                t_rs_ds_xchg_tile,
            )

        aligned_dsq_ptr = cute.make_ptr(
            self.element_dtype,
            dsq_stage.iterator.toint(),
            dsq_stage.memspace,
            assumed_align=16,
        )
        dsq_store_stage = cute.make_tensor(
            cute.recast_ptr(
                aligned_dsq_ptr,
                score_store_layout.inner,
                dtype=self.element_dtype,
            ),
            score_store_domain,
        )
        t_rs_dsq = thread_copy_r2s.partition_D(
            dsq_store_stage
        )
        assert cute.size(t_rs_dsq, mode=[4]) == 1
        t_rs_dsq_tile = t_rs_dsq[
            None, None, None, None, 0
        ]
        assert t_rs_dsq_tile.shape == r_dsq_store.shape
        cute.copy(
            tiled_copy_r2s,
            r_dsq_store,
            t_rs_dsq_tile,
        )

        cute.arch.fence_view_async_shared()
        math_barrier.arrive_and_wait()
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_MATH_END,
            )
            # dQ consumes only the two CTA-local H64xN64 dS partitions.
            # Publish this pair-ready generation before either directed
            # exchange; dK still waits for the remote dS quadrants below.
            ds_dq_pipeline.producer_commit(dsq_state)
            cute.arch.mbarrier_arrive(
                ctx_reader_done_mbars + context_slot
            )
            # The exchange depends only on the completed P/dS stores, not on
            # the load coordinator releasing IssuedCtx. Start it immediately
            # so the remote quadrants and their consumers can progress while
            # context detach waits for the final metadata reader.
            cute.arch.mbarrier_arrive(p_local_ready + stage)
            cute.arch.mbarrier_arrive(ds_local_ready + stage)
            self._detach_issued_context(
                issue_seq,
                issued_ctx,
                reducer_ctx,
                issued_ctx_mbars,
                reducer_ctx_mbars,
                ctx_reader_done_mbars,
                peer,
            )

        phase = (
            issue_seq // Int32(self.PD_STAGES)
        ) & Int32(1)
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_REMOTE_WAIT_BEGIN,
            )
        cute.arch.mbarrier_wait(
            p_remote_full + stage,
            Int32(phase),
        )
        cute.arch.mbarrier_wait(
            ds_remote_full + stage,
            Int32(phase),
        )
        if is_math_leader:
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_REMOTE_WAIT_END,
            )

        cute.arch.fence_view_async_shared()
        math_barrier.arrive_and_wait()
        if is_math_leader:
            p_dv_pipeline.producer_commit(p_state)
            ds_dk_pipeline.producer_commit(dsk_state)
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MATH,
                issue_seq,
                TRACE_PD_PUBLISH,
            )
        # Keep all four math warps on the same issue until the leader has
        # published the P/dS consumer generations.
        math_barrier.arrive_and_wait()
        p_state.advance()
        dsk_state.advance()
        dsq_state.advance()
        issue_seq += Int32(1)
        active = self._resolve_issued_context_or_done(
            issue_seq,
            issued_ctx_mbars,
            issued_stream_state,
            issued_stream_done_mbars,
        )

    if is_math_leader and issue_seq > Int32(0):
        tail_index = issue_seq % Int32(self.PD_STAGES)
        tail_phase = (
            Int32(1)
            ^ (
                (
                    issue_seq // Int32(self.PD_STAGES)
                )
                & Int32(1)
            )
        )
        p_tail_state = pipeline.PipelineState(
            self.PD_STAGES,
            issue_seq,
            tail_index,
            tail_phase,
        )
        dsk_tail_state = pipeline.PipelineState(
            self.PD_STAGES,
            issue_seq,
            tail_index,
            tail_phase,
        )
        dsq_tail_state = pipeline.PipelineState(
            self.PD_STAGES,
            issue_seq,
            tail_index,
            tail_phase,
        )
        p_dv_pipeline.producer_tail(p_tail_state)
        ds_dk_pipeline.producer_tail(dsk_tail_state)
        ds_dq_pipeline.producer_tail(dsq_tail_state)
class FlashAttentionDSABackwardSm100TwoCTAV0(
    FlashAttentionDSABackwardSm100TwoCTA
):
    """Three-stage two-CTA lifecycle for GQA128/D512."""

    # Bind the math and exchange roles as class methods. Dynamic CuTe loops
    # may capture the kernel's implicit constexpr ``self`` but cannot flatten
    # the same Python owner object when passed to a free function.
    _math_role = _run_math_role
    _exchange_role = _run_exchange_role

    @cute.jit
    def _record_trace(
        self,
        trace_buffer: Optional[cute.Tensor],
        token_idx: Int32,
        batch_idx: Int32,
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
        rank: Int32,
        role: cutlass.Constexpr[int],
        issue_seq: Int32,
        tag: cutlass.Constexpr[int],
        sub_index: cutlass.Constexpr[int] = 0,
    ) -> None:
        _trace_stamp(
            trace_buffer,
            token_idx,
            batch_idx,
            trace_token_idx,
            trace_batch_idx,
            rank,
            role,
            issue_seq,
            tag,
            sub_index,
        )

    THREADS_PER_CTA = 640

    GATHER_WARPS = (0, 1, 2, 3)
    LOAD_COORDINATOR_WARP = 4
    MMA_WARP = 5
    EXCHANGE_WARP = 6
    DESCRIPTOR_WARP = 7
    MATH_WARPS = (8, 9, 10, 11)
    REDUCE_ROUND0_WARPS = (12, 13, 14, 15)
    REDUCE_ROUND1_WARPS = (16, 17, 18, 19)
    REDUCE_THREADS_PER_ROUND = 128

    LOAD_START_BARRIER_ID = 2
    LOAD_DONE_BARRIER_ID = 3
    MATH_BARRIER_ID = 4
    DQ_EPILOGUE_R2S_BARRIER_ID = 5
    REDUCE_ROUND0_METADATA_BARRIER_ID = 6
    REDUCE_ROUND1_METADATA_BARRIER_ID = 7
    LOAD_PARTICIPANTS = 5 * 32

    OP_STAGES = 3
    PD_STAGES = 2
    CONTEXT_STAGES = 2
    REDUCER_STAGES = 2
    ROUND_STAGES = 2

    OP_MAIN_ELEMENTS_PER_STAGE = 32 * 1024 * 8 // 16
    OP_SIDE_ELEMENTS_PER_STAGE = 16 * 1024 * 8 // 16
    OP_ELEMENTS_PER_STAGE = OP_MAIN_ELEMENTS_PER_STAGE + OP_SIDE_ELEMENTS_PER_STAGE
    OP_BYTES_PER_STAGE = 48 * 1024
    OP_PAYLOAD_BYTES = OP_STAGES * OP_BYTES_PER_STAGE
    OP_MAIN_OFFSET_BYTES = 0
    OP_F_DO_OFFSET_BYTES = 16 * 1024
    OP_SIDE_OFFSET_BYTES = 32 * 1024

    PD_NESTED_ELEMENTS_PER_STAGE = 32 * 128
    PD_LOCAL_ELEMENTS_PER_STAGE = 64 * 64
    XCHG_ELEMENTS = 64 * 32
    PD_PAYLOAD_BYTES = 56 * 1024
    MAIN_PAYLOAD_BYTES = OP_PAYLOAD_BYTES + PD_PAYLOAD_BYTES

    ISSUED_TILE_CONTEXT_BYTES = 272
    REDUCER_CONTEXT_BYTES = 288
    ISSUED_TILE_CONTEXT_WORDS = ISSUED_TILE_CONTEXT_BYTES // 4
    REDUCER_CONTEXT_WORDS = REDUCER_CONTEXT_BYTES // 4
    CTX_ISSUE_SEQ_WORD = 0
    CTX_LOGICAL_TILE_WORD = 1
    CTX_KV_BASE_WORD = 2
    CTX_VALID_LO_WORD = 66
    CTX_VALID_HI_WORD = 67
    REDUCER_PENDING_MASK_WORD = 68
    STREAM_WORK_EPOCH_WORD = 0
    STREAM_ISSUED_COUNT_WORD = 1
    STREAM_DONE_WORD = 2
    STREAM_PAD_WORD = 3

    ISSUED_FULL_MBAR_BASE = 0
    ISSUED_EMPTY_MBAR_BASE = CONTEXT_STAGES
    REDUCER_FULL_MBAR_BASE = 0
    REDUCER_EMPTY_MBAR_BASE = REDUCER_STAGES
    STREAM_DONE_FULL_MBAR = 0
    STREAM_DONE_ACK_MBAR = 1
    TRAVERSAL_DESCRIPTOR_BYTES = 288
    MAX_TRAVERSAL_TILES = 32
    TRAVERSAL_DESCRIPTOR_WORDS = TRAVERSAL_DESCRIPTOR_BYTES // 4
    DESCRIPTOR_EXECUTE_WORD = ISSUED_TILE_CONTEXT_WORDS
    ISSUED_CTX_RING_BYTES = CONTEXT_STAGES * ISSUED_TILE_CONTEXT_BYTES
    REDUCER_CTX_RING_BYTES = REDUCER_STAGES * REDUCER_CONTEXT_BYTES
    LOAD_CTX_RING_BYTES = (
        CONTEXT_STAGES
        * FlashAttentionDSABackwardSm100TwoCTA.N_TILE
        * 4
    )
    ISSUED_STREAM_STATE_BYTES = 16
    FIXED_METADATA_BYTES = (
        ISSUED_CTX_RING_BYTES
        + REDUCER_CTX_RING_BYTES
        + LOAD_CTX_RING_BYTES
        + ISSUED_STREAM_STATE_BYTES
    )

    OP_PIPELINE_MBAR_COUNT = 2 * OP_STAGES
    DV_TO_BQ_PIPELINE_MBAR_COUNT = 2 * ROUND_STAGES
    BQ_PIPELINE_MBAR_COUNT = 2 * ROUND_STAGES
    S_PIPELINE_MBAR_COUNT = 2
    DP_PIPELINE_MBAR_COUNT = 2
    P_DV_PIPELINE_MBAR_COUNT = 2 * PD_STAGES
    DS_DK_PIPELINE_MBAR_COUNT = 2 * PD_STAGES
    DS_DQ_PIPELINE_MBAR_COUNT = 2 * PD_STAGES
    DKV_PIPELINE_MBAR_COUNT = 2 * ROUND_STAGES
    DQ_PIPELINE_MBAR_COUNT = 2 * ROUND_STAGES
    ISSUED_CTX_PIPELINE_MBAR_COUNT = 2 * CONTEXT_STAGES
    REDUCER_CTX_PIPELINE_MBAR_COUNT = 2 * REDUCER_STAGES

    SOURCE_KIND_COUNT = 6
    SOURCE_MBAR_COUNT = SOURCE_KIND_COUNT * OP_STAGES
    CONTROL_STATS_WORDS = 32
    SOFTMAX_STATS_HEADS = (
        FlashAttentionDSABackwardSm100TwoCTA.H_TILE_CTA
    )
    SOFTMAX_LSE_STATS_WORD = CONTROL_STATS_WORDS
    SOFTMAX_SUM_ODO_STATS_WORD = (
        SOFTMAX_LSE_STATS_WORD + SOFTMAX_STATS_HEADS
    )
    STATS_WORDS = (
        SOFTMAX_SUM_ODO_STATS_WORD + SOFTMAX_STATS_HEADS
    )
    # The 512-byte loader ring pushes the 1024-byte-aligned struct from
    # 203 KiB to the next whole-KiB boundary.
    EXPECTED_SHARED_STORAGE_BYTES = 208_896

    # DEVELOPMENT-ONLY diagnostics. These are deliberately not
    # exposed by the public interface and must be removed before integration.
    # A probe caller may override them before cute.compile to isolate the
    # already-proven whole AsyncUmma FIFO from auxiliary pipeline groups.
    DIAGNOSTIC_OPERAND_ONLY = False
    # 1=S/dP, 2=+P/dS retention, 3=+dO/Q refill, 4=+dKV,
    # 5=+dQ-final (the default full lifecycle).
    DIAGNOSTIC_AUX_STAGE = 5

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        super().__init__(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            block_tile=block_tile,
            max_topk=max_topk,
        )
        self.threads_per_cta = self.THREADS_PER_CTA
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.THREADS_PER_CTA,
        )
        self.load_start_barrier = pipeline.NamedBarrier(
            barrier_id=self.LOAD_START_BARRIER_ID,
            num_threads=self.LOAD_PARTICIPANTS,
        )
        self.load_done_barrier = pipeline.NamedBarrier(
            barrier_id=self.LOAD_DONE_BARRIER_ID,
            num_threads=self.LOAD_PARTICIPANTS,
        )
        self.math_barrier = pipeline.NamedBarrier(
            barrier_id=self.MATH_BARRIER_ID,
            num_threads=len(self.MATH_WARPS) * 32,
        )
        self.dq_epilogue_r2s_barrier = pipeline.NamedBarrier(
            barrier_id=self.DQ_EPILOGUE_R2S_BARRIER_ID,
            num_threads=self.REDUCE_THREADS_PER_ROUND,
        )
        self.reduce_round0_metadata_barrier = pipeline.NamedBarrier(
            barrier_id=self.REDUCE_ROUND0_METADATA_BARRIER_ID,
            num_threads=self.REDUCE_THREADS_PER_ROUND,
        )
        self.reduce_round1_metadata_barrier = pipeline.NamedBarrier(
            barrier_id=self.REDUCE_ROUND1_METADATA_BARRIER_ID,
            num_threads=self.REDUCE_THREADS_PER_ROUND,
        )

        assert self.OP_MAIN_ELEMENTS_PER_STAGE == 16_384
        assert self.OP_SIDE_ELEMENTS_PER_STAGE == 8_192
        assert self.OP_PAYLOAD_BYTES == 144 * 1024
        assert self.PD_PAYLOAD_BYTES == 56 * 1024
        assert self.MAIN_PAYLOAD_BYTES == 200 * 1024
        assert self.ISSUED_CTX_RING_BYTES == 544
        assert self.REDUCER_CTX_RING_BYTES == 576
        assert self.FIXED_METADATA_BYTES == 1_648
        assert self.CONTEXT_STAGES == 2
        assert self.REDUCER_STAGES == 2
        assert self.D_ROUNDS == 2
        assert (
            self.DESCRIPTOR_EXECUTE_WORD
            < self.TRAVERSAL_DESCRIPTOR_WORDS
        )
        assert self.max_topk <= (
            self.MAX_TRAVERSAL_TILES * self.N_TILE
        )
        assert (
            len(self.REDUCE_ROUND0_WARPS) * 32
            == self.REDUCE_THREADS_PER_ROUND
        )
        assert (
            len(self.REDUCE_ROUND1_WARPS) * 32
            == self.REDUCE_THREADS_PER_ROUND
        )

    @cute.jit
    def _make_operand_slot_view(
        self,
        raw_slots: cute.Tensor,
        slot: Int32,
        offset_bytes: cutlass.Constexpr[int],
        layout: cute.ComposedLayout,
    ):
        """Attach one typed swizzled operand view to a raw 48-KiB slot."""

        stage_bytes = raw_slots[None, slot]
        return cute.make_tensor(
            cute.recast_ptr(
                stage_bytes.iterator + offset_bytes,
                layout.inner,
                dtype=self.element_dtype,
            ),
            layout.outer,
        )

    @cute.jit
    def _snapshot_load_context(
        self,
        issue_seq: Int32,
        issued_ctx: cute.Tensor,
        load_ctx: cute.Tensor,
        ctx_reader_done_mbars: cute.Pointer,
        tidx: Int32,
    ) -> None:
        """Snapshot loader-owned KV indices and release its IssuedCtx credit."""

        slot = issue_seq % Int32(self.CONTEXT_STAGES)
        if tidx < Int32(self.N_TILE):
            load_ctx[tidx, slot] = issued_ctx[
                self.CTX_KV_BASE_WORD + tidx,
                slot,
            ]
        cute.arch.fence_view_async_shared()
        self.load_start_barrier.arrive_and_wait()
        cute.arch.fence_view_async_shared()
        if tidx == Int32(self.LOAD_COORDINATOR_WARP * 32):
            cute.arch.mbarrier_arrive(
                ctx_reader_done_mbars + slot
            )

    @cute.jit
    def _gather_score_kv_chunk(
        self,
        mKV: cute.Tensor,
        load_ctx: cute.Tensor,
        destination: cute.Tensor,
        batch_idx: Int32,
        issue_seq: Int32,
        chunk: cutlass.Constexpr[int],
        rank: Int32,
        loader_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Gather the rank-owned N32 x D128 F side operand."""

        index_in_group = loader_tidx % self.KV_GROUP_SIZE
        group_index = loader_tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        destination_rows = cute.composition(
            destination,
            cute.make_layout((self.N_TILE_CTA, self.K_CHUNK)),
        )
        context_slot = issue_seq % Int32(self.CONTEXT_STAGES)
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_iteration * self.KV_NUM_GROUPS + group_index
            logical_n = rank * self.N_TILE_CTA + local_n
            kv_index = load_ctx[logical_n, context_slot]
            if kv_index >= 0:
                self._copy_sparse_k_d128_row(
                    mKV,
                    destination_rows,
                    local_n,
                    kv_index,
                    batch_idx,
                    Int32(chunk * self.K_CHUNK),
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
            else:
                self._zero_sparse_k_d128_row(
                    destination_rows,
                    local_n,
                    index_in_group,
                )

    @cute.jit
    def _gather_grad_k_round(
        self,
        mKV: cute.Tensor,
        load_ctx: cute.Tensor,
        destination: cute.Tensor,
        batch_idx: Int32,
        issue_seq: Int32,
        round_index: cutlass.Constexpr[int],
        rank: Int32,
        loader_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Gather full N64 for the rank-owned gradient D128 slice."""

        index_in_group = loader_tidx % self.KV_GROUP_SIZE
        group_index = loader_tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE // self.KV_NUM_GROUPS
        destination_rows = cute.composition(
            destination,
            cute.make_layout(
                (self.N_TILE, self.D_TILE_CTA),
                stride=(self.D_TILE_CTA, 1),
            ),
        )
        d_offset = (
            round_index * self.D_TILE_CLUSTER
            + rank * self.D_TILE_CTA
        )
        context_slot = issue_seq % Int32(self.CONTEXT_STAGES)
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            logical_n = row_iteration * self.KV_NUM_GROUPS + group_index
            kv_index = load_ctx[logical_n, context_slot]
            if kv_index >= 0:
                self._copy_sparse_k_d128_row(
                    mKV,
                    destination_rows,
                    logical_n,
                    kv_index,
                    batch_idx,
                    Int32(d_offset),
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
            else:
                self._zero_sparse_k_d128_row(
                    destination_rows,
                    logical_n,
                    index_in_group,
                )

    @cute.jit
    def _make_pd_stage_view(
        self,
        raw_tensor: cute.Tensor,
        stage: Int32,
        elements_per_stage: cutlass.Constexpr[int],
        layout: cute.ComposedLayout,
    ):
        """Attach one retained P/dS stage to its operation-specific layout."""

        return cute.make_tensor(
            cute.recast_ptr(
                raw_tensor.iterator + stage * elements_per_stage,
                layout.inner,
                dtype=self.element_dtype,
            ),
            layout.outer,
        )

    @cute.jit
    def _init_mbar_range(
        self,
        base: cute.Pointer,
        count: cutlass.Constexpr[int],
    ) -> None:
        for stage in cutlass.range_constexpr(count):
            cute.arch.mbarrier_init(base + stage, 1)

    @cute.jit
    def _init_pair_mbar_range(
        self,
        base: cute.Pointer,
        count: cutlass.Constexpr[int],
    ) -> None:
        """Initialize a symmetric local+peer event with two arrivals."""

        for stage in cutlass.range_constexpr(count):
            cute.arch.mbarrier_init(base + stage, 2)

    @cute.jit
    def _pair_arrive(
        self,
        barrier: cute.Pointer,
        peer_rank: Int32,
    ) -> None:
        """Publish one logical event symmetrically to both cluster ranks."""

        cute.arch.mbarrier_arrive(barrier)
        cute.arch.mbarrier_arrive(
            barrier,
            peer_cta_rank_in_cluster=peer_rank,
        )

    @cute.jit
    def _wait_pair(
        self,
        barrier: cute.Pointer,
        phase: Int32,
    ) -> None:
        cute.arch.mbarrier_wait(barrier, phase)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _publish_issued_context(
        self,
        issue_seq: Int32,
        descriptor: cute.Tensor,
        issued_ctx: cute.Tensor,
        issued_ctx_mbars: cute.Pointer,
        peer_rank: Int32,
    ) -> None:
        """Commit one staged descriptor after its IssuedCtx slot is acquired."""

        slot = issue_seq % Int32(self.CONTEXT_STAGES)
        lane = cute.arch.lane_idx()
        cute.arch.fence_view_async_shared()
        if lane == Int32(0):
            descriptor[self.CTX_ISSUE_SEQ_WORD] = issue_seq
        cute.arch.sync_warp()

        issued_ctx[lane, slot] = descriptor[lane]
        issued_ctx[lane + Int32(32), slot] = descriptor[
            lane + Int32(32)
        ]
        if lane < Int32(4):
            issued_ctx[lane + Int32(64), slot] = descriptor[
                lane + Int32(64)
            ]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        if lane == Int32(0):
            self._pair_arrive(
                issued_ctx_mbars
                + self.ISSUED_FULL_MBAR_BASE
                + slot,
                peer_rank,
            )
        cute.arch.sync_warp()

    @cute.jit
    def _publish_issued_stream_done(
        self,
        work_epoch: Int32,
        issued_tile_count: Int32,
        issued_stream_state: cute.Tensor,
        issued_stream_done_mbars: cute.Pointer,
        peer_rank: Int32,
    ) -> None:
        """Publish a sticky end state without consuming metadata-ring credit."""

        issued_stream_state[
            self.STREAM_WORK_EPOCH_WORD
        ] = cutlass.Uint32(work_epoch)
        issued_stream_state[
            self.STREAM_ISSUED_COUNT_WORD
        ] = cutlass.Uint32(issued_tile_count)
        issued_stream_state[
            self.STREAM_DONE_WORD
        ] = cutlass.Uint32(1)
        issued_stream_state[
            self.STREAM_PAD_WORD
        ] = cutlass.Uint32(0)
        cute.arch.fence_view_async_shared()
        self._pair_arrive(
            issued_stream_done_mbars
            + self.STREAM_DONE_FULL_MBAR,
            peer_rank,
        )

    @cute.jit
    def _detach_issued_context(
        self,
        issue_seq: Int32,
        issued_ctx: cute.Tensor,
        reducer_ctx: cute.Tensor,
        issued_ctx_mbars: cute.Pointer,
        reducer_ctx_mbars: cute.Pointer,
        ctx_reader_done_mbars: cute.Pointer,
        peer_rank: Int32,
    ) -> None:
        """Copy reducer metadata by value, then release the IssuedCtx slot."""

        slot = issue_seq % Int32(self.CONTEXT_STAGES)
        epoch = (
            issue_seq // Int32(self.CONTEXT_STAGES)
        ) & Int32(1)
        producer_phase = epoch ^ Int32(1)
        cute.arch.mbarrier_wait(
            ctx_reader_done_mbars + slot,
            epoch,
        )
        cute.arch.fence_view_async_shared()
        self._wait_pair(
            reducer_ctx_mbars
            + self.REDUCER_EMPTY_MBAR_BASE
            + slot,
            producer_phase,
        )
        for word in cutlass.range_constexpr(
            self.ISSUED_TILE_CONTEXT_WORDS
        ):
            reducer_ctx[word, slot] = issued_ctx[word, slot]
        reducer_ctx[
            self.REDUCER_PENDING_MASK_WORD,
            slot,
        ] = Int32(0b11)
        cute.arch.fence_view_async_shared()
        self._pair_arrive(
            reducer_ctx_mbars
            + self.REDUCER_FULL_MBAR_BASE
            + slot,
            peer_rank,
        )
        self._pair_arrive(
            issued_ctx_mbars
            + self.ISSUED_EMPTY_MBAR_BASE
            + slot,
            peer_rank,
        )

    @cute.jit
    def _decode_traversal_descriptor(
        self,
        mTopkIdxs: cute.Tensor,
        descriptor: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        topk: Int32,
        logical_tile: Int32,
    ) -> None:
        """Stage one reverse-order N64 descriptor and its execute bit."""

        lane = cute.arch.lane_idx()
        topk_slot_lo = (
            logical_tile * Int32(self.N_TILE) + lane
        )
        kv_index_lo = Int32(-1)
        if topk_slot_lo < topk:
            kv_index_lo = mTopkIdxs[
                topk_slot_lo,
                (token_idx, batch_idx),
            ]
        descriptor[
            self.CTX_KV_BASE_WORD + lane
        ] = kv_index_lo
        valid_lo = cute.arch.vote_ballot_sync(
            kv_index_lo >= Int32(0)
        )

        n_index_hi = lane + Int32(32)
        topk_slot_hi = (
            logical_tile * Int32(self.N_TILE) + n_index_hi
        )
        kv_index_hi = Int32(-1)
        if topk_slot_hi < topk:
            kv_index_hi = mTopkIdxs[
                topk_slot_hi,
                (token_idx, batch_idx),
            ]
        descriptor[
            self.CTX_KV_BASE_WORD + n_index_hi
        ] = kv_index_hi
        valid_hi = cute.arch.vote_ballot_sync(
            kv_index_hi >= Int32(0)
        )

        if lane == Int32(0):
            descriptor[self.CTX_ISSUE_SEQ_WORD] = Int32(-1)
            descriptor[self.CTX_LOGICAL_TILE_WORD] = logical_tile
            descriptor[self.CTX_VALID_LO_WORD] = valid_lo
            descriptor[self.CTX_VALID_HI_WORD] = valid_hi
            descriptor[self.DESCRIPTOR_EXECUTE_WORD] = Int32(
                (valid_lo | valid_hi) != Int32(0)
            )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _resolve_issued_context_or_done(
        self,
        issue_seq: Int32,
        issued_ctx_mbars: cute.Pointer,
        issued_stream_state: cute.Tensor,
        issued_stream_done_mbars: cute.Pointer,
    ) -> cutlass.Boolean:
        """Resolve exactly ``IssuedCtx(issue_seq)`` or the sticky stream end."""

        slot = issue_seq % Int32(self.CONTEXT_STAGES)
        phase = (
            issue_seq // Int32(self.CONTEXT_STAGES)
        ) & Int32(1)
        resolved = cutlass.Boolean(False)
        has_context = cutlass.Boolean(False)
        while not resolved:
            context_ready = _mbarrier_try_wait(
                issued_ctx_mbars
                + self.ISSUED_FULL_MBAR_BASE
                + slot,
                phase,
            )
            if context_ready:
                cute.arch.fence_view_async_shared()
                has_context = cutlass.Boolean(True)
                resolved = cutlass.Boolean(True)
            else:
                done_ready = _mbarrier_try_wait(
                    issued_stream_done_mbars
                    + self.STREAM_DONE_FULL_MBAR,
                    Int32(0),
                )
                if done_ready:
                    cute.arch.fence_view_async_shared()
                    final_count = Int32(
                        issued_stream_state[
                            self.STREAM_ISSUED_COUNT_WORD
                        ]
                    )
                    if final_count > issue_seq:
                        self._wait_pair(
                            issued_ctx_mbars
                            + self.ISSUED_FULL_MBAR_BASE
                            + slot,
                            phase,
                        )
                        has_context = cutlass.Boolean(True)
                    else:
                        has_context = cutlass.Boolean(False)
                    resolved = cutlass.Boolean(True)
        return has_context

    @cute.jit
    def _resolve_reducer_context_or_done(
        self,
        issue_seq: Int32,
        reducer_ctx_mbars: cute.Pointer,
        issued_stream_state: cute.Tensor,
        issued_stream_done_mbars: cute.Pointer,
    ) -> cutlass.Boolean:
        """Resolve one detached reducer record or the final issued count."""

        slot = issue_seq % Int32(self.REDUCER_STAGES)
        phase = (
            issue_seq // Int32(self.REDUCER_STAGES)
        ) & Int32(1)
        resolved = cutlass.Boolean(False)
        has_context = cutlass.Boolean(False)
        while not resolved:
            context_ready = _mbarrier_try_wait(
                reducer_ctx_mbars
                + self.REDUCER_FULL_MBAR_BASE
                + slot,
                phase,
            )
            if context_ready:
                cute.arch.fence_view_async_shared()
                has_context = cutlass.Boolean(True)
                resolved = cutlass.Boolean(True)
            else:
                done_ready = _mbarrier_try_wait(
                    issued_stream_done_mbars
                    + self.STREAM_DONE_FULL_MBAR,
                    Int32(0),
                )
                if done_ready:
                    cute.arch.fence_view_async_shared()
                    final_count = Int32(
                        issued_stream_state[
                            self.STREAM_ISSUED_COUNT_WORD
                        ]
                    )
                    if final_count > issue_seq:
                        self._wait_pair(
                            reducer_ctx_mbars
                            + self.REDUCER_FULL_MBAR_BASE
                            + slot,
                            phase,
                        )
                        has_context = cutlass.Boolean(True)
                    else:
                        has_context = cutlass.Boolean(False)
                    resolved = cutlass.Boolean(True)
        return has_context

    @cute.jit
    def _resolve_pd_tile_or_done(
        self,
        issue_seq: Int32,
        local_ready_mbars: cute.Pointer,
        issued_stream_state: cute.Tensor,
        issued_stream_done_mbars: cute.Pointer,
    ) -> cutlass.Boolean:
        """Resolve one math/XCHG generation without waiting for final count."""

        stage = issue_seq % Int32(self.PD_STAGES)
        phase = (
            issue_seq // Int32(self.PD_STAGES)
        ) & Int32(1)
        resolved = cutlass.Boolean(False)
        has_tile = cutlass.Boolean(False)
        while not resolved:
            tile_ready = _mbarrier_try_wait(
                local_ready_mbars + stage,
                phase,
            )
            if tile_ready:
                cute.arch.fence_view_async_shared()
                has_tile = cutlass.Boolean(True)
                resolved = cutlass.Boolean(True)
            else:
                done_ready = _mbarrier_try_wait(
                    issued_stream_done_mbars
                    + self.STREAM_DONE_FULL_MBAR,
                    Int32(0),
                )
                if done_ready:
                    cute.arch.fence_view_async_shared()
                    final_count = Int32(
                        issued_stream_state[
                            self.STREAM_ISSUED_COUNT_WORD
                        ]
                    )
                    if final_count > issue_seq:
                        cute.arch.mbarrier_wait(
                            local_ready_mbars + stage,
                            phase,
                        )
                        cute.arch.fence_view_async_shared()
                        has_tile = cutlass.Boolean(True)
                    else:
                        has_tile = cutlass.Boolean(False)
                    resolved = cutlass.Boolean(True)
        return has_tile

    @cute.jit
    def _load_f_task(
        self,
        raw_slots: cute.Tensor,
        score_a_layout: cute.ComposedLayout,
        score_b_layout: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        tma_atom_do: cute.CopyAtom,
        rank_g_q: cute.Tensor,
        rank_g_do: cute.Tensor,
        block_coord_vmnk,
        a_cta_layout: cute.Layout,
        mKV: cute.Tensor,
        load_ctx: cute.Tensor,
        batch_idx: Int32,
        issue_seq: Int32,
        chunk: cutlass.Constexpr[int],
        rank: Int32,
        tidx: Int32,
        warp_idx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        score_q_source_mbars: cute.Pointer,
        score_do_source_mbars: cute.Pointer,
        op_pipeline,
        producer_state: pipeline.PipelineState,
        score_a_stage_bytes: cutlass.Constexpr[int],
        token_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ) -> pipeline.PipelineState:
        """Produce one F task and advance one persistent whole-task state."""

        if tidx == Int32(self.GATHER_WARPS[0] * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_GATHER,
                issue_seq,
                TRACE_F_LOAD_BEGIN,
                chunk,
            )
        if tidx == Int32(self.LOAD_COORDINATOR_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_LOAD,
                issue_seq,
                TRACE_F_LOAD_BEGIN,
                chunk,
            )

        slot = producer_state.index
        if warp_idx == self.LOAD_COORDINATOR_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(
                    score_q_source_mbars + slot,
                    1,
                )
                cute.arch.mbarrier_init(
                    score_do_source_mbars + slot,
                    1,
                )
                op_pipeline.producer_acquire(producer_state)

        self.load_start_barrier.arrive_and_wait()
        f_q = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_MAIN_OFFSET_BYTES,
            score_a_layout,
        )
        f_do = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_F_DO_OFFSET_BYTES,
            score_a_layout,
        )
        f_kv = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_SIDE_OFFSET_BYTES,
            score_b_layout,
        )

        if warp_idx == self.LOAD_COORDINATOR_WARP:
            t_q_smem, t_q_gmem = cpasync.tma_partition(
                tma_atom_q,
                block_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(f_q, 0, 3),
                cute.group_modes(rank_g_q, 0, 3),
            )
            t_do_smem, t_do_gmem = cpasync.tma_partition(
                tma_atom_do,
                block_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(f_do, 0, 3),
                cute.group_modes(rank_g_do, 0, 3),
            )
            t_q_gmem = t_q_gmem[None, 0, None]
            t_do_gmem = t_do_gmem[None, 0, None]
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    score_q_source_mbars + slot,
                    score_a_stage_bytes,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    score_do_source_mbars + slot,
                    score_a_stage_bytes,
                )
            cute.copy(
                tma_atom_q,
                t_q_gmem[None, chunk],
                t_q_smem[None],
                tma_bar_ptr=score_q_source_mbars + slot,
            )
            cute.copy(
                tma_atom_do,
                t_do_gmem[None, chunk],
                t_do_smem[None],
                tma_bar_ptr=score_do_source_mbars + slot,
            )

        if warp_idx <= self.GATHER_WARPS[-1]:
            self._gather_score_kv_chunk(
                mKV,
                load_ctx,
                f_kv,
                batch_idx,
                issue_seq,
                chunk,
                rank,
                tidx,
                copy_atom,
                thread_copy,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()

        if warp_idx == self.LOAD_COORDINATOR_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_wait(
                    score_q_source_mbars + slot,
                    Int32(0),
                )
                cute.arch.mbarrier_wait(
                    score_do_source_mbars + slot,
                    Int32(0),
                )

        self.load_done_barrier.arrive_and_wait()
        cute.arch.fence_view_async_shared()
        if warp_idx == self.LOAD_COORDINATOR_WARP:
            with cute.arch.elect_one():
                op_pipeline.producer_commit(producer_state)
        if tidx == Int32(self.GATHER_WARPS[0] * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_GATHER,
                issue_seq,
                TRACE_F_LOAD_END,
                chunk,
            )
        if tidx == Int32(self.LOAD_COORDINATOR_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_LOAD,
                issue_seq,
                TRACE_F_LOAD_END,
                chunk,
            )
        producer_state.advance()
        return producer_state

    @cute.jit
    def _load_bv_task(
        self,
        raw_slots: cute.Tensor,
        dkv_a_layout: cute.ComposedLayout,
        dq_a_layout: cute.ComposedLayout,
        tma_atom_dot: cute.CopyAtom,
        rank_g_dot: cute.Tensor,
        block_coord_vmnk,
        a_cta_layout: cute.Layout,
        mKV: cute.Tensor,
        load_ctx: cute.Tensor,
        batch_idx: Int32,
        issue_seq: Int32,
        round_index: cutlass.Constexpr[int],
        rank: Int32,
        tidx: Int32,
        warp_idx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
        grad_do_source_mbars: cute.Pointer,
        grad_k_source_mbars: cute.Pointer,
        op_pipeline,
        producer_state: pipeline.PipelineState,
        grad_a_stage_bytes: cutlass.Constexpr[int],
        token_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ) -> pipeline.PipelineState:
        """Produce one BV task and advance one persistent whole-task state."""

        if tidx == Int32(self.GATHER_WARPS[0] * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_GATHER,
                issue_seq,
                TRACE_BV_LOAD_BEGIN,
                round_index,
            )
        if tidx == Int32(self.LOAD_COORDINATOR_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_LOAD,
                issue_seq,
                TRACE_BV_LOAD_BEGIN,
                round_index,
            )

        slot = producer_state.index
        if warp_idx == self.LOAD_COORDINATOR_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(
                    grad_do_source_mbars + slot,
                    1,
                )
                cute.arch.mbarrier_init(
                    grad_k_source_mbars + slot,
                    1,
                )
                op_pipeline.producer_acquire(producer_state)

        self.load_start_barrier.arrive_and_wait()
        bv_do = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_MAIN_OFFSET_BYTES,
            dkv_a_layout,
        )
        bv_k = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_SIDE_OFFSET_BYTES,
            dq_a_layout,
        )

        if warp_idx == self.LOAD_COORDINATOR_WARP:
            t_dot_smem, t_dot_gmem = cpasync.tma_partition(
                tma_atom_dot,
                block_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(bv_do, 0, 3),
                cute.group_modes(rank_g_dot, 0, 3),
            )
            t_dot_gmem = t_dot_gmem[None, None, 0]
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    grad_do_source_mbars + slot,
                    grad_a_stage_bytes,
                )
                cute.arch.mbarrier_arrive(
                    grad_k_source_mbars + slot
                )
            cute.copy(
                tma_atom_dot,
                t_dot_gmem[None, round_index],
                t_dot_smem[None],
                tma_bar_ptr=grad_do_source_mbars + slot,
            )

        if warp_idx <= self.GATHER_WARPS[-1]:
            self._gather_grad_k_round(
                mKV,
                load_ctx,
                bv_k,
                batch_idx,
                issue_seq,
                round_index,
                rank,
                tidx,
                copy_atom,
                thread_copy,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()

        if warp_idx == self.LOAD_COORDINATOR_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_wait(
                    grad_do_source_mbars + slot,
                    Int32(0),
                )
                cute.arch.mbarrier_wait(
                    grad_k_source_mbars + slot,
                    Int32(0),
                )

        self.load_done_barrier.arrive_and_wait()
        cute.arch.fence_view_async_shared()
        if warp_idx == self.LOAD_COORDINATOR_WARP:
            with cute.arch.elect_one():
                op_pipeline.producer_commit(producer_state)
        if tidx == Int32(self.GATHER_WARPS[0] * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_GATHER,
                issue_seq,
                TRACE_BV_LOAD_END,
                round_index,
            )
        if tidx == Int32(self.LOAD_COORDINATOR_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_LOAD,
                issue_seq,
                TRACE_BV_LOAD_END,
                round_index,
            )
        producer_state.advance()
        return producer_state

    @cute.jit
    def _begin_bq_refill(
        self,
        round_index: cutlass.Constexpr[int],
        bv_slot: Int32,
        refill_count: Int32,
        raw_slots: cute.Tensor,
        dkv_a_layout: cute.ComposedLayout,
        tma_atom_qt: cute.CopyAtom,
        rank_g_qt: cute.Tensor,
        block_coord_vmnk,
        a_cta_layout: cute.Layout,
        grad_q_source_mbars: cute.Pointer,
        do_empty_pipeline,
        q_full_pipeline,
        grad_a_stage_bytes: cutlass.Constexpr[int],
        issue_seq: Int32,
        rank: Int32,
        tidx: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ) -> None:
        """Issue one admitted BQ TMA without waiting for its completion."""

        do_phase = (
            refill_count // Int32(self.ROUND_STAGES)
        ) & Int32(1)
        do_state = pipeline.PipelineState(
            self.ROUND_STAGES,
            refill_count,
            refill_count % Int32(self.ROUND_STAGES),
            do_phase,
        )
        q_state = pipeline.PipelineState(
            self.ROUND_STAGES,
            refill_count,
            refill_count % Int32(self.ROUND_STAGES),
            Int32(1) ^ do_phase,
        )

        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DESC_BQ,
                issue_seq,
                TRACE_BQ_WAIT_BEGIN,
                round_index,
            )
        do_ready = do_empty_pipeline.consumer_try_wait(do_state)
        with cute.arch.elect_one():
            do_empty_pipeline.consumer_wait(do_state, do_ready)
            q_full_pipeline.producer_acquire(q_state)
        cute.arch.sync_warp()
        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DESC_BQ,
                issue_seq,
                TRACE_BQ_WAIT_END,
                round_index,
            )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DESC_BQ,
                issue_seq,
                TRACE_BQ_LOAD_BEGIN,
                round_index,
            )

        bq_q = self._make_operand_slot_view(
            raw_slots,
            bv_slot,
            self.OP_MAIN_OFFSET_BYTES,
            dkv_a_layout,
        )
        t_qt_smem, t_qt_gmem = cpasync.tma_partition(
            tma_atom_qt,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(bq_q, 0, 3),
            cute.group_modes(rank_g_qt, 0, 3),
        )
        t_qt_gmem = t_qt_gmem[None, None, 0]
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(
                grad_q_source_mbars + bv_slot,
                1,
            )
            cute.arch.mbarrier_arrive_and_expect_tx(
                grad_q_source_mbars + bv_slot,
                grad_a_stage_bytes,
            )
        cute.copy(
            tma_atom_qt,
            t_qt_gmem[None, round_index],
            t_qt_smem[None],
            tma_bar_ptr=grad_q_source_mbars + bv_slot,
        )

    @cute.jit
    def _finish_bq_refill(
        self,
        round_index: cutlass.Constexpr[int],
        bv_slot: Int32,
        refill_count: Int32,
        grad_q_source_mbars: cute.Pointer,
        do_empty_pipeline,
        q_full_pipeline,
        issue_seq: Int32,
        rank: Int32,
        tidx: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ) -> None:
        """Retire one previously issued Q TMA and publish its full generation."""

        do_phase = (
            refill_count // Int32(self.ROUND_STAGES)
        ) & Int32(1)
        do_state = pipeline.PipelineState(
            self.ROUND_STAGES,
            refill_count,
            refill_count % Int32(self.ROUND_STAGES),
            do_phase,
        )
        q_state = pipeline.PipelineState(
            self.ROUND_STAGES,
            refill_count,
            refill_count % Int32(self.ROUND_STAGES),
            Int32(1) ^ do_phase,
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_wait(
                grad_q_source_mbars + bv_slot,
                Int32(0),
            )
        cute.arch.sync_warp()
        cute.arch.fence_view_async_shared()
        with cute.arch.elect_one():
            q_full_pipeline.producer_commit(q_state)
            do_empty_pipeline.consumer_release(do_state)
        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DESC_BQ,
                issue_seq,
                TRACE_BQ_LOAD_END,
                round_index,
            )

    @cute.jit
    def _mma_sdp_tile(
        self,
        raw_slots: cute.Tensor,
        score_a_layout: cute.ComposedLayout,
        score_b_layout: cute.ComposedLayout,
        score_tiled_mma: cute.TiledMma,
        dp_tiled_mma: cute.TiledMma,
        t_score: cute.Tensor,
        t_dp: cute.Tensor,
        op_pipeline,
        op_state: pipeline.PipelineState,
        s_pipeline,
        s_state: pipeline.PipelineState,
        dp_pipeline,
        dp_state: pipeline.PipelineState,
        issue_seq: Int32,
        rank: Int32,
        tidx: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Consume the fixed four-entry F group for one issued tile."""

        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MMA,
                issue_seq,
                TRACE_SDP_BEGIN,
            )
        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 1
        ):
            s_pipeline.producer_acquire(s_state)
            dp_pipeline.producer_acquire(dp_state)

        for _chunk in cutlass.range_constexpr(self.K_CHUNKS):
            op_pipeline.consumer_wait(op_state)
            slot = op_state.index
            f_q = self._make_operand_slot_view(
                raw_slots,
                slot,
                self.OP_MAIN_OFFSET_BYTES,
                score_a_layout,
            )
            f_do = self._make_operand_slot_view(
                raw_slots,
                slot,
                self.OP_F_DO_OFFSET_BYTES,
                score_a_layout,
            )
            f_kv = self._make_operand_slot_view(
                raw_slots,
                slot,
                self.OP_SIDE_OFFSET_BYTES,
                score_b_layout,
            )
            score_q_fragment = score_tiled_mma.make_fragment_A(
                f_q
            )
            score_kv_fragment = score_tiled_mma.make_fragment_B(
                f_kv
            )
            dp_do_fragment = dp_tiled_mma.make_fragment_A(f_do)
            dp_kv_fragment = dp_tiled_mma.make_fragment_B(f_kv)

            score_mma = score_tiled_mma.with_()
            score_mma.set(
                tcgen05.Field.ACCUMULATE,
                _chunk != 0,
            )
            for k_block in cutlass.range_constexpr(
                cute.size(score_q_fragment, mode=[2])
            ):
                cute.gemm(
                    score_mma,
                    t_score,
                    score_q_fragment[None, None, k_block],
                    score_kv_fragment[None, None, k_block],
                    t_score,
                )
                score_mma.set(
                    tcgen05.Field.ACCUMULATE,
                    True,
                )

            dp_mma = dp_tiled_mma.with_()
            dp_mma.set(
                tcgen05.Field.ACCUMULATE,
                _chunk != 0,
            )
            for k_block in cutlass.range_constexpr(
                cute.size(dp_do_fragment, mode=[2])
            ):
                cute.gemm(
                    dp_mma,
                    t_dp,
                    dp_do_fragment[None, None, k_block],
                    dp_kv_fragment[None, None, k_block],
                    t_dp,
                )
                dp_mma.set(
                    tcgen05.Field.ACCUMULATE,
                    True,
                )
            op_pipeline.consumer_release(op_state)
            op_state.advance()

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 1
        ):
            cute.arch.fence_view_async_tmem_store()
            s_pipeline.producer_commit(s_state)
            dp_pipeline.producer_commit(dp_state)
            s_state.advance()
            dp_state.advance()
        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MMA,
                issue_seq,
                TRACE_SDP_END,
            )
        return op_state, s_state, dp_state

    @cute.jit
    def _mma_grad_pre_q(
        self,
        round_index: cutlass.Constexpr[int],
        accumulate_dq: cutlass.Constexpr[bool],
        is_final: cutlass.Boolean,
        issue_seq: Int32,
        raw_slots: cute.Tensor,
        raw_p_dv: cute.Tensor,
        raw_ds_dq: cute.Tensor,
        dkv_a_layout: cute.ComposedLayout,
        dq_a_layout: cute.ComposedLayout,
        dkv_b_layout: cute.ComposedLayout,
        dq_b_layout: cute.ComposedLayout,
        dkv_tiled_mma: cute.TiledMma,
        dq_tiled_mma: cute.TiledMma,
        t_dkv_round: cute.Tensor,
        t_dq_round: cute.Tensor,
        op_pipeline,
        op_wait_state: pipeline.PipelineState,
        p_dv_pipeline,
        p_wait_state: pipeline.PipelineState,
        ds_dq_pipeline,
        dsq_wait_state: pipeline.PipelineState,
        do_empty_pipeline,
        do_state: pipeline.PipelineState,
        q_full_pipeline,
        tma_atom_qt: cute.CopyAtom,
        rank_g_qt: cute.Tensor,
        block_coord_vmnk,
        a_cta_layout: cute.Layout,
        grad_q_source_mbars: cute.Pointer,
        grad_a_stage_bytes: cutlass.Constexpr[int],
        dkv_pipeline,
        dkv_acquire_state: pipeline.PipelineState,
        dq_final_pipeline,
        dq_state: pipeline.PipelineState,
        rank: Int32,
        tidx: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Consume one BV/BQ round with a lexical dQ accumulate mode."""

        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MMA,
                issue_seq,
                TRACE_GRAD_BEGIN,
                round_index,
            )
        op_pipeline.consumer_wait(op_wait_state)
        slot = op_wait_state.index
        op_wait_state.advance()

        pd_stage = issue_seq % Int32(self.PD_STAGES)
        bv_do = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_MAIN_OFFSET_BYTES,
            dkv_a_layout,
        )
        bq_k = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_SIDE_OFFSET_BYTES,
            dq_a_layout,
        )
        p_operand = self._make_pd_stage_view(
            raw_p_dv,
            pd_stage,
            self.PD_NESTED_ELEMENTS_PER_STAGE,
            dkv_b_layout,
        )
        dsq_operand = self._make_pd_stage_view(
            raw_ds_dq,
            pd_stage,
            self.PD_LOCAL_ELEMENTS_PER_STAGE,
            dq_b_layout,
        )
        dv_a_fragment = dkv_tiled_mma.make_fragment_A(bv_do)
        dq_a_fragment = dq_tiled_mma.make_fragment_A(bq_k)
        p_fragment = dkv_tiled_mma.make_fragment_B(p_operand)
        dsq_fragment = dq_tiled_mma.make_fragment_B(
            dsq_operand
        )

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 4
        ):
            dkv_pipeline.producer_acquire(dkv_acquire_state)
            dkv_acquire_state.advance()
        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 3
        ):
            do_empty_pipeline.producer_acquire(do_state)

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 2
            and round_index == 0
        ):
            p_dv_pipeline.consumer_wait(p_wait_state)
            p_wait_state.advance()

        dv_mma = dkv_tiled_mma.with_()
        dv_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range_constexpr(
            cute.size(dv_a_fragment, mode=[2])
        ):
            cute.gemm(
                dv_mma,
                t_dkv_round,
                dv_a_fragment[None, None, k_block],
                p_fragment[None, None, k_block],
                t_dkv_round,
            )
            dv_mma.set(tcgen05.Field.ACCUMULATE, True)

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 3
        ):
            do_empty_pipeline.producer_commit(do_state)
            do_state.advance()

        # Acquire the final-full generation before issuing the last dQ GEMM.
        # Otherwise a consumer can observe a generation that did not order
        # the UMMA producing the final accumulator value.
        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 5
        ):
            if is_final:
                dq_final_pipeline.producer_acquire(dq_state)

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 2
            and round_index == 0
        ):
            ds_dq_pipeline.consumer_wait(dsq_wait_state)
            dsq_wait_state.advance()

        dq_mma = dq_tiled_mma.with_()
        dq_mma.set(
            tcgen05.Field.ACCUMULATE,
            accumulate_dq,
        )
        for k_block in cutlass.range_constexpr(
            cute.size(dq_a_fragment, mode=[2])
        ):
            cute.gemm(
                dq_mma,
                t_dq_round,
                dq_a_fragment[None, None, k_block],
                dsq_fragment[None, None, k_block],
                t_dq_round,
            )
            dq_mma.set(tcgen05.Field.ACCUMULATE, True)

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 5
        ):
            if is_final:
                cute.arch.fence_view_async_tmem_store()
                dq_final_pipeline.producer_commit(dq_state)
                dq_state.advance()

        # The local MMA warp owns its CTA's Q replacement.  Waiting until
        # after dQ issue preserves the useful dV/dQ overlap, while returning
        # immediately after the TMA issue lets round 0 overlap round 1.
        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 3
        ):
            refill_count = (
                issue_seq * Int32(self.D_ROUNDS)
                + Int32(round_index)
            )
            self._begin_bq_refill(
                round_index,
                slot,
                refill_count,
                raw_slots,
                dkv_a_layout,
                tma_atom_qt,
                rank_g_qt,
                block_coord_vmnk,
                a_cta_layout,
                grad_q_source_mbars,
                do_empty_pipeline,
                q_full_pipeline,
                grad_a_stage_bytes,
                issue_seq,
                rank,
                tidx,
                token_idx,
                batch_idx,
                trace_buffer,
                trace_token_idx,
                trace_batch_idx,
            )

        return (
            slot,
            op_wait_state,
            p_wait_state,
            dsq_wait_state,
            do_state,
            dkv_acquire_state,
            dq_state,
        )

    @cute.jit
    def _mma_grad_post_q(
        self,
        round_index: cutlass.Constexpr[int],
        issue_seq: Int32,
        slot: Int32,
        raw_slots: cute.Tensor,
        raw_ds_dk: cute.Tensor,
        dkv_a_layout: cute.ComposedLayout,
        dkv_b_layout: cute.ComposedLayout,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv_round: cute.Tensor,
        op_pipeline,
        op_release_state: pipeline.PipelineState,
        p_dv_pipeline,
        p_release_state: pipeline.PipelineState,
        ds_dk_pipeline,
        dsk_wait_state: pipeline.PipelineState,
        dsk_release_state: pipeline.PipelineState,
        ds_dq_pipeline,
        dsq_release_state: pipeline.PipelineState,
        q_full_pipeline,
        q_state: pipeline.PipelineState,
        dkv_pipeline,
        dkv_commit_state: pipeline.PipelineState,
        rank: Int32,
        tidx: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Finish one gradient round after W5 has replaced dO with Q."""

        pd_stage = issue_seq % Int32(self.PD_STAGES)
        bq_q = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_MAIN_OFFSET_BYTES,
            dkv_a_layout,
        )
        dsk_operand = self._make_pd_stage_view(
            raw_ds_dk,
            pd_stage,
            self.PD_NESTED_ELEMENTS_PER_STAGE,
            dkv_b_layout,
        )
        dk_a_fragment = dkv_tiled_mma.make_fragment_A(bq_q)
        dsk_fragment = dkv_tiled_mma.make_fragment_B(
            dsk_operand
        )

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 3
        ):
            q_full_pipeline.consumer_wait(q_state)

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 2
            and round_index == 0
        ):
            ds_dk_pipeline.consumer_wait(dsk_wait_state)
            dsk_wait_state.advance()

        dk_mma = dkv_tiled_mma.with_()
        dk_mma.set(tcgen05.Field.ACCUMULATE, True)
        for k_block in cutlass.range_constexpr(
            cute.size(dk_a_fragment, mode=[2])
        ):
            cute.gemm(
                dk_mma,
                t_dkv_round,
                dk_a_fragment[None, None, k_block],
                dsk_fragment[None, None, k_block],
                t_dkv_round,
            )

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 3
        ):
            q_full_pipeline.consumer_release(q_state)
            q_state.advance()
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 4
        ):
            dkv_pipeline.producer_commit(dkv_commit_state)
            dkv_commit_state.advance()

        op_pipeline.consumer_release(op_release_state)
        op_release_state.advance()
        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 2
            and round_index == self.D_ROUNDS - 1
        ):
            p_dv_pipeline.consumer_release(p_release_state)
            ds_dk_pipeline.consumer_release(dsk_release_state)
            ds_dq_pipeline.consumer_release(dsq_release_state)
            p_release_state.advance()
            dsk_release_state.advance()
            dsq_release_state.advance()

        if tidx == Int32(self.MMA_WARP * 32):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_MMA,
                issue_seq,
                TRACE_GRAD_END,
                round_index,
            )
        return (
            op_release_state,
            p_release_state,
            dsk_wait_state,
            dsk_release_state,
            dsq_release_state,
            q_state,
            dkv_commit_state,
        )

    @cute.jit
    def _mma_grad_pair(
        self,
        accumulate_dq: cutlass.Constexpr[bool],
        is_final: cutlass.Boolean,
        issue_seq: Int32,
        raw_slots: cute.Tensor,
        raw_p_dv: cute.Tensor,
        raw_ds_dk: cute.Tensor,
        raw_ds_dq: cute.Tensor,
        dkv_a_layout: cute.ComposedLayout,
        dq_a_layout: cute.ComposedLayout,
        dkv_b_layout: cute.ComposedLayout,
        dq_b_layout: cute.ComposedLayout,
        dkv_tiled_mma: cute.TiledMma,
        dq_tiled_mma: cute.TiledMma,
        t_dkv_round0: cute.Tensor,
        t_dkv_round1: cute.Tensor,
        t_dq_round0: cute.Tensor,
        t_dq_round1: cute.Tensor,
        op_pipeline,
        op_state: pipeline.PipelineState,
        p_dv_pipeline,
        p_wait_state: pipeline.PipelineState,
        p_release_state: pipeline.PipelineState,
        ds_dk_pipeline,
        dsk_wait_state: pipeline.PipelineState,
        dsk_release_state: pipeline.PipelineState,
        ds_dq_pipeline,
        dsq_wait_state: pipeline.PipelineState,
        dsq_release_state: pipeline.PipelineState,
        do_empty_pipeline,
        do_state: pipeline.PipelineState,
        q_full_pipeline,
        q_state: pipeline.PipelineState,
        tma_atom_qt: cute.CopyAtom,
        rank_g_qt: cute.Tensor,
        block_coord_vmnk,
        a_cta_layout: cute.Layout,
        grad_q_source_mbars: cute.Pointer,
        grad_a_stage_bytes: cutlass.Constexpr[int],
        dkv_pipeline,
        dkv_state: pipeline.PipelineState,
        dq_final_pipeline,
        dq_state: pipeline.PipelineState,
        rank: Int32,
        tidx: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Issue both dV+dQ halves before either Q-dependent dK half."""

        op_wait_state = op_state.clone()
        op_release_state = op_state.clone()
        dkv_acquire_state = dkv_state.clone()
        dkv_commit_state = dkv_state.clone()

        (
            slot0,
            op_wait_state,
            p_wait_state,
            dsq_wait_state,
            do_state,
            dkv_acquire_state,
            dq_state,
        ) = self._mma_grad_pre_q(
            0,
            accumulate_dq,
            is_final,
            issue_seq,
            raw_slots,
            raw_p_dv,
            raw_ds_dq,
            dkv_a_layout,
            dq_a_layout,
            dkv_b_layout,
            dq_b_layout,
            dkv_tiled_mma,
            dq_tiled_mma,
            t_dkv_round0,
            t_dq_round0,
            op_pipeline,
            op_wait_state,
            p_dv_pipeline,
            p_wait_state,
            ds_dq_pipeline,
            dsq_wait_state,
            do_empty_pipeline,
            do_state,
            q_full_pipeline,
            tma_atom_qt,
            rank_g_qt,
            block_coord_vmnk,
            a_cta_layout,
            grad_q_source_mbars,
            grad_a_stage_bytes,
            dkv_pipeline,
            dkv_acquire_state,
            dq_final_pipeline,
            dq_state,
            rank,
            tidx,
            token_idx,
            batch_idx,
            trace_buffer,
            trace_token_idx,
            trace_batch_idx,
        )
        (
            slot1,
            op_wait_state,
            p_wait_state,
            dsq_wait_state,
            do_state,
            dkv_acquire_state,
            dq_state,
        ) = self._mma_grad_pre_q(
            1,
            accumulate_dq,
            is_final,
            issue_seq,
            raw_slots,
            raw_p_dv,
            raw_ds_dq,
            dkv_a_layout,
            dq_a_layout,
            dkv_b_layout,
            dq_b_layout,
            dkv_tiled_mma,
            dq_tiled_mma,
            t_dkv_round1,
            t_dq_round1,
            op_pipeline,
            op_wait_state,
            p_dv_pipeline,
            p_wait_state,
            ds_dq_pipeline,
            dsq_wait_state,
            do_empty_pipeline,
            do_state,
            q_full_pipeline,
            tma_atom_qt,
            rank_g_qt,
            block_coord_vmnk,
            a_cta_layout,
            grad_q_source_mbars,
            grad_a_stage_bytes,
            dkv_pipeline,
            dkv_acquire_state,
            dq_final_pipeline,
            dq_state,
            rank,
            tidx,
            token_idx,
            batch_idx,
            trace_buffer,
            trace_token_idx,
            trace_batch_idx,
        )

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 3
        ):
            first_refill_count = (
                issue_seq * Int32(self.D_ROUNDS)
            )
            self._finish_bq_refill(
                0,
                slot0,
                first_refill_count,
                grad_q_source_mbars,
                do_empty_pipeline,
                q_full_pipeline,
                issue_seq,
                rank,
                tidx,
                token_idx,
                batch_idx,
                trace_buffer,
                trace_token_idx,
                trace_batch_idx,
            )
            self._finish_bq_refill(
                1,
                slot1,
                first_refill_count + Int32(1),
                grad_q_source_mbars,
                do_empty_pipeline,
                q_full_pipeline,
                issue_seq,
                rank,
                tidx,
                token_idx,
                batch_idx,
                trace_buffer,
                trace_token_idx,
                trace_batch_idx,
            )

        (
            op_release_state,
            p_release_state,
            dsk_wait_state,
            dsk_release_state,
            dsq_release_state,
            q_state,
            dkv_commit_state,
        ) = self._mma_grad_post_q(
            0,
            issue_seq,
            slot0,
            raw_slots,
            raw_ds_dk,
            dkv_a_layout,
            dkv_b_layout,
            dkv_tiled_mma,
            t_dkv_round0,
            op_pipeline,
            op_release_state,
            p_dv_pipeline,
            p_release_state,
            ds_dk_pipeline,
            dsk_wait_state,
            dsk_release_state,
            ds_dq_pipeline,
            dsq_release_state,
            q_full_pipeline,
            q_state,
            dkv_pipeline,
            dkv_commit_state,
            rank,
            tidx,
            token_idx,
            batch_idx,
            trace_buffer,
            trace_token_idx,
            trace_batch_idx,
        )
        (
            op_release_state,
            p_release_state,
            dsk_wait_state,
            dsk_release_state,
            dsq_release_state,
            q_state,
            dkv_commit_state,
        ) = self._mma_grad_post_q(
            1,
            issue_seq,
            slot1,
            raw_slots,
            raw_ds_dk,
            dkv_a_layout,
            dkv_b_layout,
            dkv_tiled_mma,
            t_dkv_round1,
            op_pipeline,
            op_release_state,
            p_dv_pipeline,
            p_release_state,
            ds_dk_pipeline,
            dsk_wait_state,
            dsk_release_state,
            ds_dq_pipeline,
            dsq_release_state,
            q_full_pipeline,
            q_state,
            dkv_pipeline,
            dkv_commit_state,
            rank,
            tidx,
            token_idx,
            batch_idx,
            trace_buffer,
            trace_token_idx,
            trace_batch_idx,
        )

        return (
            op_release_state,
            p_wait_state,
            p_release_state,
            dsk_wait_state,
            dsk_release_state,
            dsq_wait_state,
            dsq_release_state,
            do_state,
            q_state,
            dkv_commit_state,
            dq_state,
        )

    @cute.jit
    def _reduce_dkv_round_v0(
        self,
        t_dkv: cute.Tensor,
        dkv_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        mdKV_acc: cute.Tensor,
        reducer_ctx: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        issue_seq: Int32,
        rank: Int32,
        local_tidx: Int32,
        reducer_ctx_mbars: cute.Pointer,
        dkv_pipeline,
        consumer_state,
        token_idx: Int32,
        batch_idx: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Release pair-owned TMEM before consuming FP32 atomic arguments."""

        reducer_slot = issue_seq % Int32(self.REDUCER_STAGES)
        if local_tidx == Int32(0):
            if cutlass.const_expr(round_index == 0):
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R0,
                    issue_seq,
                    TRACE_REDUCE_WAIT_BEGIN,
                    round_index,
                )
            else:
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R1,
                    issue_seq,
                    TRACE_REDUCE_WAIT_BEGIN,
                    round_index,
                )
        dkv_pipeline.consumer_wait(consumer_state)
        if local_tidx == Int32(0):
            if cutlass.const_expr(round_index == 0):
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R0,
                    issue_seq,
                    TRACE_REDUCE_WAIT_END,
                    round_index,
                )
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R0,
                    issue_seq,
                    TRACE_REDUCE_T2R_BEGIN,
                    round_index,
                )
            else:
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R1,
                    issue_seq,
                    TRACE_REDUCE_WAIT_END,
                    round_index,
                )
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R1,
                    issue_seq,
                    TRACE_REDUCE_T2R_BEGIN,
                    round_index,
                )
        tiled_t2r = tcgen05.make_tmem_copy(
            dkv_tmem_load,
            t_dkv,
        )
        thread_t2r = tiled_t2r.get_slice(local_tidx)
        thread_source = thread_t2r.partition_S(t_dkv)
        thread_coordinates = thread_t2r.partition_D(
            rank_coordinates
        )
        thread_values = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )
        cute.copy(tiled_t2r, thread_source, thread_values)
        cute.arch.fence_view_async_tmem_load()
        if local_tidx == Int32(0):
            if cutlass.const_expr(round_index == 0):
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R0,
                    issue_seq,
                    TRACE_REDUCE_T2R_END,
                    round_index,
                )
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R0,
                    issue_seq,
                    TRACE_REDUCE_ATOMIC_BEGIN,
                    round_index,
                )
            else:
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R1,
                    issue_seq,
                    TRACE_REDUCE_T2R_END,
                    round_index,
                )
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R1,
                    issue_seq,
                    TRACE_REDUCE_ATOMIC_BEGIN,
                    round_index,
                )

        # Both ranks contribute 128 reducer threads.  TMEM becomes reusable
        # after these 256 releases, independently of the global atomics.
        dkv_pipeline.consumer_release(consumer_state)

        # The ROW_MAJOR T2R gives each reducer thread one D row and all
        # N=0..63 values. Transpose each 4-lane x 4-N register block so every
        # lane owns one N and four contiguous, 16-byte-aligned D values. One
        # result-discarding red.v4 then replaces four scalar atomics without
        # changing the contribution count or the TMEM/reducer lifetimes.
        assert cute.size(thread_values) == self.N_TILE
        lane_in_quad = local_tidx % Int32(4)
        for vector_index in cutlass.range_constexpr(
            self.N_TILE // 4
        ):
            value_base = vector_index * 4
            value_0 = thread_values[value_base]
            value_1 = thread_values[value_base + 1]
            value_2 = thread_values[value_base + 2]
            value_3 = thread_values[value_base + 3]

            # Swap lane bit 0 with register-index bit 0. Select the two
            # required sources before shuffling so this stage uses two
            # shuffles instead of four.
            swap_0 = value_0
            swap_1 = value_1
            if (lane_in_quad & Int32(1)) == Int32(0):
                swap_0 = value_1
                swap_1 = value_3
            else:
                swap_0 = value_0
                swap_1 = value_2
            peer_0 = cute.arch.shuffle_sync_bfly(swap_0, offset=1)
            peer_1 = cute.arch.shuffle_sync_bfly(swap_1, offset=1)
            stage_0 = value_0
            stage_1 = value_1
            stage_2 = value_2
            stage_3 = value_3
            if (lane_in_quad & Int32(1)) == Int32(0):
                stage_0 = value_0
                stage_1 = peer_0
                stage_2 = value_2
                stage_3 = peer_1
            else:
                stage_0 = peer_0
                stage_1 = value_1
                stage_2 = peer_1
                stage_3 = value_3

            # Swap lane bit 1 with register-index bit 1, again selecting the
            # required pair before issuing the two shuffles.
            swap_0 = stage_0
            swap_1 = stage_1
            if (lane_in_quad & Int32(2)) == Int32(0):
                swap_0 = stage_2
                swap_1 = stage_3
            else:
                swap_0 = stage_0
                swap_1 = stage_1
            peer_0 = cute.arch.shuffle_sync_bfly(swap_0, offset=2)
            peer_1 = cute.arch.shuffle_sync_bfly(swap_1, offset=2)
            vector_0 = stage_0
            vector_1 = stage_1
            vector_2 = stage_2
            vector_3 = stage_3
            if (lane_in_quad & Int32(2)) == Int32(0):
                vector_0 = stage_0
                vector_1 = stage_1
                vector_2 = peer_0
                vector_3 = peer_1
            else:
                vector_0 = peer_0
                vector_1 = peer_1
                vector_2 = stage_2
                vector_3 = stage_3

            logical_coordinate = thread_coordinates[value_base]
            d_in_round = Int32(
                cute.get(logical_coordinate, mode=[0])
            )
            n_index = (
                Int32(
                    cute.get(logical_coordinate, mode=[1])
                )
                + lane_in_quad
            )
            kv_index = reducer_ctx[
                self.CTX_KV_BASE_WORD + n_index,
                reducer_slot,
            ]
            if kv_index >= Int32(0):
                d_index = (
                    Int32(round_index * self.D_TILE_CLUSTER)
                    + d_in_round
                    - lane_in_quad
                )
                destination_ptr = (
                    mdKV_acc.iterator
                    + d_index * mdKV_acc.stride[0]
                    + kv_index * mdKV_acc.stride[1]
                )
                _atomic_add_fp32x4_v0_p(
                    vector_0,
                    vector_1,
                    vector_2,
                    vector_3,
                    destination_ptr,
                )

        if cutlass.const_expr(round_index == 0):
            self.reduce_round0_metadata_barrier.arrive_and_wait()
        else:
            self.reduce_round1_metadata_barrier.arrive_and_wait()
        if local_tidx == Int32(0):
            if cutlass.const_expr(round_index == 0):
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R0,
                    issue_seq,
                    TRACE_REDUCE_ATOMIC_END,
                    round_index,
                )
            else:
                self._record_trace(
                    trace_buffer,
                    token_idx,
                    batch_idx,
                    trace_token_idx,
                    trace_batch_idx,
                    rank,
                    TRACE_ROLE_REDUCE_R1,
                    issue_seq,
                    TRACE_REDUCE_ATOMIC_END,
                    round_index,
                )
        if local_tidx == Int32(0):
            pending_ptr = (
                reducer_ctx.iterator
                + self.REDUCER_PENDING_MASK_WORD
                + reducer_slot * self.REDUCER_CONTEXT_WORDS
            )
            if cutlass.const_expr(round_index == 0):
                clear_mask = Int32(-2)
            else:
                clear_mask = Int32(-3)
            old_mask = _atomic_and_shared_i32(
                pending_ptr,
                clear_mask,
            )
            if (old_mask & clear_mask) == Int32(0):
                self._pair_arrive(
                    reducer_ctx_mbars
                    + self.REDUCER_EMPTY_MBAR_BASE
                    + reducer_slot,
                    rank ^ Int32(1),
                )

        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _store_dq_round_v0(
        self,
        t_dq: cute.Tensor,
        dq_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        s_dq_epi: cute.Tensor,
        tma_atom_dq_epi: cute.CopyAtom,
        tma_tensor_dq_epi: cute.Tensor,
        dq_source_done_mbar,
        round_index: cutlass.Constexpr[int],
        rank: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        local_tidx: Int32,
        dq_pipeline,
        consumer_state,
        issue_seq: Int32,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Stage and TMA-store one rank-owned final dQ D128 slice."""

        if local_tidx == Int32(0):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DQ_EPI,
                issue_seq,
                TRACE_DQ_WAIT_BEGIN,
                round_index,
            )
        dq_pipeline.consumer_wait(consumer_state)
        if local_tidx == Int32(0):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DQ_EPI,
                issue_seq,
                TRACE_DQ_WAIT_END,
                round_index,
            )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DQ_EPI,
                issue_seq,
                TRACE_DQ_T2R_BEGIN,
                round_index,
            )
        tiled_t2r = tcgen05.make_tmem_copy(
            dq_tmem_load,
            t_dq,
        )
        thread_t2r = tiled_t2r.get_slice(local_tidx)
        thread_source = thread_t2r.partition_S(t_dq)
        thread_coordinates = thread_t2r.partition_D(
            rank_coordinates
        )
        thread_values = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )
        cute.copy(tiled_t2r, thread_source, thread_values)
        cute.arch.fence_view_async_tmem_load()
        if local_tidx == Int32(0):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DQ_EPI,
                issue_seq,
                TRACE_DQ_T2R_END,
                round_index,
            )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DQ_EPI,
                issue_seq,
                TRACE_DQ_STORE_BEGIN,
                round_index,
            )
        # TMEM may be recycled as soon as every rank-owned value is in
        # registers; the global store does not extend the final-full lifetime.
        dq_pipeline.consumer_release(consumer_state)

        for value_index in cutlass.range_constexpr(
            cute.size(thread_values)
        ):
            d_in_cluster = Int32(
                cute.get(
                    thread_coordinates[value_index],
                    mode=[0],
                )
            )
            head = Int32(
                cute.get(
                    thread_coordinates[value_index],
                    mode=[1],
                )
            )
            local_d = (
                d_in_cluster
                - rank * Int32(self.D_TILE_CTA)
            )
            s_dq_epi[
                head,
                local_d,
            ] = self.element_dtype(thread_values[value_index])

        # Every R2S writer publishes its values and joins before the elected
        # warp hands the complete tile to the async-shared TMA proxy.
        cute.arch.fence_view_async_shared()
        self.dq_epilogue_r2s_barrier.arrive_and_wait()

        g_dq_tiles = cute.local_tile(
            tma_tensor_dq_epi,
            (
                self.H_TILE_CLUSTER,
                self.D_TILE_CTA,
            ),
            (None, None, (token_idx, batch_idx)),
        )
        global_d_tile = Int32(round_index * 2) + rank
        g_dq_tile = g_dq_tiles[
            None,
            None,
            0,
            global_d_tile,
        ]
        t_smem, t_gmem = cpasync.tma_partition(
            tma_atom_dq_epi,
            0,
            cute.make_layout(1),
            cute.group_modes(s_dq_epi, 0, 2),
            cute.group_modes(g_dq_tile, 0, 2),
        )

        if local_tidx < Int32(32):
            cute.arch.fence_view_async_shared()
            cute.copy(tma_atom_dq_epi, t_smem, t_gmem)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(
                    dq_source_done_mbar
                )

        # Source-side completion, not TMA destination visibility, authorizes
        # the next round to overwrite the aliased operand main region.
        cute.arch.mbarrier_wait(
            dq_source_done_mbar,
            Int32(round_index % 2),
        )
        if local_tidx == Int32(0):
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DQ_EPI,
                issue_seq,
                TRACE_DQ_STORE_END,
                round_index,
            )
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _zero_dq_round_v0(
        self,
        mdQ: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        rank: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        local_tidx: Int32,
    ) -> None:
        """Zero one rank/round output when no dQ generation was initialized."""

        linear = local_tidx
        while linear < self.D_TILE_CTA * self.H_TILE_CLUSTER:
            local_d = linear // self.H_TILE_CLUSTER
            head = linear % self.H_TILE_CLUSTER
            d_index = (
                round_index * self.D_TILE_CLUSTER
                + rank * self.D_TILE_CTA
                + local_d
            )
            mdQ[
                d_index,
                head,
                (token_idx, batch_idx),
            ] = self.element_dtype(0.0)
            linear += self.REDUCE_THREADS_PER_ROUND

    def _specialize_shared_storage(
        self,
        default_storage,
        score_a_layout_staged,
        score_b_layout_staged,
        dkv_a_layout_staged,
        dkv_b_layout_staged,
        dq_a_layout_staged,
        dq_b_layout_staged,
    ):
        """Build the v0 200-KiB payload and its typed pipeline barriers."""

        del default_storage
        assert cute.cosize(dkv_b_layout_staged) == self.PD_NESTED_ELEMENTS_PER_STAGE
        assert cute.cosize(dq_b_layout_staged) == self.PD_LOCAL_ELEMENTS_PER_STAGE

        @cute.struct
        class SharedStorage:
            # Fixed 200-KiB data plane.  The operand region remains raw so a
            # constexpr F/BV branch can attach exactly one legal typed view.
            operand_slots: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    self.OP_PAYLOAD_BYTES,
                ],
                1024,
            ]
            p_dv: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    self.PD_STAGES * self.PD_NESTED_ELEMENTS_PER_STAGE,
                ],
                1024,
            ]
            ds_dk: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    self.PD_STAGES * self.PD_NESTED_ELEMENTS_PER_STAGE,
                ],
                1024,
            ]
            ds_dq: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    self.PD_STAGES * self.PD_LOCAL_ELEMENTS_PER_STAGE,
                ],
                1024,
            ]
            p_xchg: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    self.XCHG_ELEMENTS,
                ],
                1024,
            ]
            ds_xchg: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    self.XCHG_ELEMENTS,
                ],
                1024,
            ]

            # Worst-case indexed/sparse metadata plane.  Dense execution does
            # not manufacture ring events, but it uses the identical static
            # storage shape so there is no hidden residency dispatch.
            traversal_descriptor: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.TRAVERSAL_DESCRIPTOR_WORDS,
                ],
                16,
            ]
            issued_ctx_ring: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    (
                        self.CONTEXT_STAGES
                        * self.ISSUED_TILE_CONTEXT_WORDS
                    ),
                ],
                16,
            ]
            # The load roles snapshot only the 64 sparse KV indices they
            # consume.  IssuedCtx can then detach as soon as math has read its
            # masks, while F/BV continue from this loader-owned ring.
            load_ctx_ring: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.CONTEXT_STAGES * self.N_TILE,
                ],
                16,
            ]
            reducer_ctx_ring: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    (
                        self.REDUCER_STAGES
                        * self.REDUCER_CONTEXT_WORDS
                    ),
                ],
                16,
            ]
            issued_stream_state: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint32,
                    self.ISSUED_STREAM_STATE_BYTES // 4,
                ],
                16,
            ]

            # Standard full/empty pipelines.
            operand_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_PIPELINE_MBAR_COUNT,
            ]
            dv_to_bq_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.DV_TO_BQ_PIPELINE_MBAR_COUNT,
            ]
            bq_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.BQ_PIPELINE_MBAR_COUNT,
            ]
            s_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.S_PIPELINE_MBAR_COUNT,
            ]
            dp_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.DP_PIPELINE_MBAR_COUNT,
            ]
            p_dv_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.P_DV_PIPELINE_MBAR_COUNT,
            ]
            ds_dk_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.DS_DK_PIPELINE_MBAR_COUNT,
            ]
            ds_dq_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.DS_DQ_PIPELINE_MBAR_COUNT,
            ]
            dkv_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.DKV_PIPELINE_MBAR_COUNT,
            ]
            dq_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.DQ_PIPELINE_MBAR_COUNT,
            ]
            issued_ctx_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.ISSUED_CTX_PIPELINE_MBAR_COUNT,
            ]
            reducer_ctx_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.REDUCER_CTX_PIPELINE_MBAR_COUNT,
            ]

            # Source-specific operand completion.
            score_q_source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_STAGES,
            ]
            score_do_source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_STAGES,
            ]
            score_kv_source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_STAGES,
            ]
            grad_do_source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_STAGES,
            ]
            grad_k_source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_STAGES,
            ]
            grad_q_source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.OP_STAGES,
            ]

            # Directed DSM exchange protocol.
            p_local_store_ready_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.PD_STAGES,
            ]
            p_remote_full_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.PD_STAGES,
            ]
            p_pair_ready_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.PD_STAGES,
            ]
            ds_local_store_ready_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.PD_STAGES,
            ]
            ds_remote_full_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.PD_STAGES,
            ]
            ds_pair_ready_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.PD_STAGES,
            ]
            p_xchg_source_done_mbar: cutlass.Int64
            ds_xchg_source_done_mbar: cutlass.Int64

            # Metadata detach, stream tail, and epilogue control.
            ctx_reader_done_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.CONTEXT_STAGES,
            ]
            # Reuse the formerly redundant reducer-copy-complete storage as
            # a two-stage pair-wide traversal-descriptor consensus ring.
            descriptor_consensus_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.REDUCER_STAGES,
            ]
            issued_stream_done_ack_mbars: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            operand_consumer_done_mbar: cutlass.Int64
            dq_epilogue_source_done_mbar: cutlass.Int64
            outer_role_drain_mbar: cutlass.Int64

            stats: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint32,
                    self.STATS_WORDS,
                ],
                128,
            ]

            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64

        assert (
            SharedStorage.size_in_bytes()
            == self.EXPECTED_SHARED_STORAGE_BYTES
        )
        return SharedStorage

    @cute.kernel
    def kernel(
        self,
        problem_shape: Tuple[
            Int32,
            Int32,
            Int32,
            Tuple[Int32, Int32],
        ],
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_qt: cute.CopyAtom,
        tma_tensor_qt: cute.Tensor,
        tma_atom_dot: cute.CopyAtom,
        tma_tensor_dot: cute.Tensor,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
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
        dkv_a_layout_staged: cute.ComposedLayout,
        dkv_b_layout_staged: cute.ComposedLayout,
        dq_a_layout_staged: cute.ComposedLayout,
        dq_b_layout_staged: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout,
        score_tmem_load: cute.CopyAtom,
        dkv_tmem_load: cute.CopyAtom,
        dq_tmem_load: cute.CopyAtom,
        tma_atom_dq_epi: cute.CopyAtom,
        tma_tensor_dq_epi: cute.Tensor,
        dq_epi_layout_staged: cute.ComposedLayout,
        score_a_stage_bytes: cutlass.Constexpr[int],
        grad_a_stage_bytes: cutlass.Constexpr[int],
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
    ):
        """Run the pipelined role graph with production operand loads.

        F and BV are filled by four 128-bit gather warps plus a CTA-local
        TMA coordinator and are published only after their 160-thread source
        join.  BQ refills the same BV main region after the group-aware dV
        completion generation. P/dS math/exchange and dKV reduction are
        numerical; final dQ uses the drained operand-main TMA epilogue.
        """

        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        peer_rank = rank ^ Int32(1)
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        is_leader_cta = rank == 0
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)

        if tidx == Int32(0):
            _trace_header_begin(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
            )

        _ = problem_shape
        _ = mQ

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Extract every storage pointer and tensor before any dynamic role
        # branch.  A field access captured from such a branch makes the DSL
        # attempt to flatten the entire SharedStorage object.
        operand_mbars_ptr = storage.operand_mbars.data_ptr()
        dv_to_bq_mbars_ptr = storage.dv_to_bq_mbars.data_ptr()
        bq_mbars_ptr = storage.bq_mbars.data_ptr()
        s_mbars_ptr = storage.s_mbars.data_ptr()
        dp_mbars_ptr = storage.dp_mbars.data_ptr()
        p_dv_mbars_ptr = storage.p_dv_mbars.data_ptr()
        ds_dk_mbars_ptr = storage.ds_dk_mbars.data_ptr()
        ds_dq_mbars_ptr = storage.ds_dq_mbars.data_ptr()
        dkv_mbars_ptr = storage.dkv_mbars.data_ptr()
        dq_mbars_ptr = storage.dq_mbars.data_ptr()
        issued_ctx_mbars_ptr = storage.issued_ctx_mbars.data_ptr()
        reducer_ctx_mbars_ptr = storage.reducer_ctx_mbars.data_ptr()

        score_q_source_mbars_ptr = (
            storage.score_q_source_mbars.data_ptr()
        )
        score_do_source_mbars_ptr = (
            storage.score_do_source_mbars.data_ptr()
        )
        score_kv_source_mbars_ptr = (
            storage.score_kv_source_mbars.data_ptr()
        )
        grad_do_source_mbars_ptr = (
            storage.grad_do_source_mbars.data_ptr()
        )
        grad_k_source_mbars_ptr = (
            storage.grad_k_source_mbars.data_ptr()
        )
        grad_q_source_mbars_ptr = (
            storage.grad_q_source_mbars.data_ptr()
        )

        p_local_store_ready_mbars_ptr = (
            storage.p_local_store_ready_mbars.data_ptr()
        )
        p_remote_full_mbars_ptr = (
            storage.p_remote_full_mbars.data_ptr()
        )
        p_pair_ready_mbars_ptr = (
            storage.p_pair_ready_mbars.data_ptr()
        )
        ds_local_store_ready_mbars_ptr = (
            storage.ds_local_store_ready_mbars.data_ptr()
        )
        ds_remote_full_mbars_ptr = (
            storage.ds_remote_full_mbars.data_ptr()
        )
        ds_pair_ready_mbars_ptr = (
            storage.ds_pair_ready_mbars.data_ptr()
        )
        p_xchg_source_done_mbar_ptr = (
            storage.p_xchg_source_done_mbar.ptr
        )
        ds_xchg_source_done_mbar_ptr = (
            storage.ds_xchg_source_done_mbar.ptr
        )

        ctx_reader_done_mbars_ptr = (
            storage.ctx_reader_done_mbars.data_ptr()
        )
        descriptor_consensus_mbars_ptr = (
            storage.descriptor_consensus_mbars.data_ptr()
        )
        issued_stream_done_ack_mbars_ptr = (
            storage.issued_stream_done_ack_mbars.data_ptr()
        )
        operand_consumer_done_mbar_ptr = (
            storage.operand_consumer_done_mbar.ptr
        )
        dq_epilogue_source_done_mbar_ptr = (
            storage.dq_epilogue_source_done_mbar.ptr
        )
        outer_role_drain_mbar_ptr = (
            storage.outer_role_drain_mbar.ptr
        )
        tmem_holding_buf_ptr = storage.tmem_holding_buf.ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr

        raw_operand = storage.operand_slots.get_tensor(
            cute.make_layout((self.OP_PAYLOAD_BYTES,))
        )
        raw_slots = storage.operand_slots.get_tensor(
            cute.make_layout(
                (self.OP_BYTES_PER_STAGE, self.OP_STAGES),
                stride=(1, self.OP_BYTES_PER_STAGE),
            )
        )
        dq_epi_bytes = cute.size_in_bytes(
            self.element_dtype,
            dq_epi_layout_staged,
        )
        assert dq_epi_bytes <= (
            self.OP_MAIN_ELEMENTS_PER_STAGE
            * self.element_dtype.width
            // 8
        )
        # This is an alias, not an additional SharedStorage member.  It is
        # consumed only after the full operand/P-dS/xchg drain and the
        # pre-epilogue cluster rendezvous below.
        s_dq_epi = cute.make_tensor(
            cute.recast_ptr(
                raw_operand.iterator + self.OP_MAIN_OFFSET_BYTES,
                dq_epi_layout_staged.inner,
                self.element_dtype,
            ),
            dq_epi_layout_staged.outer,
        )[None, None, 0]
        raw_p_dv = storage.p_dv.get_tensor(
            cute.make_layout(
                (
                    self.PD_STAGES
                    * self.PD_NESTED_ELEMENTS_PER_STAGE,
                )
            )
        )
        raw_ds_dk = storage.ds_dk.get_tensor(
            cute.make_layout(
                (
                    self.PD_STAGES
                    * self.PD_NESTED_ELEMENTS_PER_STAGE,
                )
            )
        )
        raw_ds_dq = storage.ds_dq.get_tensor(
            cute.make_layout(
                (
                    self.PD_STAGES
                    * self.PD_LOCAL_ELEMENTS_PER_STAGE,
                )
            )
        )
        raw_p_xchg = storage.p_xchg.get_tensor(
            cute.make_layout((self.XCHG_ELEMENTS,))
        )
        raw_ds_xchg = storage.ds_xchg.get_tensor(
            cute.make_layout((self.XCHG_ELEMENTS,))
        )
        traversal_descriptor = (
            storage.traversal_descriptor.get_tensor(
                cute.make_layout((self.TRAVERSAL_DESCRIPTOR_WORDS,))
            )
        )
        issued_ctx_ring = storage.issued_ctx_ring.get_tensor(
            cute.make_layout(
                (
                    self.ISSUED_TILE_CONTEXT_WORDS,
                    self.CONTEXT_STAGES,
                ),
                stride=(1, self.ISSUED_TILE_CONTEXT_WORDS),
            )
        )
        load_ctx_ring = storage.load_ctx_ring.get_tensor(
            cute.make_layout(
                (self.N_TILE, self.CONTEXT_STAGES),
                stride=(1, self.N_TILE),
            )
        )
        reducer_ctx_ring = storage.reducer_ctx_ring.get_tensor(
            cute.make_layout(
                (
                    self.REDUCER_CONTEXT_WORDS,
                    self.REDUCER_STAGES,
                ),
                stride=(1, self.REDUCER_CONTEXT_WORDS),
            )
        )
        issued_stream_state = (
            storage.issued_stream_state.get_tensor(
                cute.make_layout(
                    (self.ISSUED_STREAM_STATE_BYTES // 4,)
                )
            )
        )
        stats = storage.stats.get_tensor(
            cute.make_layout((self.STATS_WORDS,))
        )
        softmax_stats = cute.make_tensor(
            cute.recast_ptr(
                stats.iterator + self.SOFTMAX_LSE_STATS_WORD,
                dtype=self.acc_dtype,
            ),
            cute.make_layout(
                (self.H_TILE_CTA, 2),
                stride=(1, self.H_TILE_CTA),
            ),
        )

        # Every async-to-UMMA path is relayed by one elected coordinator in
        # each CTA.  The real CG2 consumer is the rank-zero MMA warp; all
        # lanes take the warp-uniform wait/gemm/release path while the
        # pipeline elects its one counted commit lane.
        async_pair = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            2,
        )
        umma_one = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            1,
        )
        reduce_pair = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (
                cute.size(dkv_tiled_mma.thr_id.shape)
                * self.REDUCE_THREADS_PER_ROUND
            ),
        )

        op_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.OP_STAGES,
            producer_group=async_pair,
            consumer_group=umma_one,
            barrier_storage=operand_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        op_producer, op_consumer = op_pipeline.make_participants()
        do_empty_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.ROUND_STAGES,
            producer_group=umma_one,
            consumer_group=async_pair,
            barrier_storage=dv_to_bq_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        do_empty_producer, do_empty_consumer = (
            do_empty_pipeline.make_participants()
        )
        q_full_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.ROUND_STAGES,
            producer_group=async_pair,
            consumer_group=umma_one,
            barrier_storage=bq_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        q_full_producer, q_full_consumer = (
            q_full_pipeline.make_participants()
        )
        s_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=umma_one,
            consumer_group=async_pair,
            barrier_storage=s_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        s_producer, s_consumer = s_pipeline.make_participants()
        dp_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=umma_one,
            consumer_group=async_pair,
            barrier_storage=dp_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        dp_producer, dp_consumer = dp_pipeline.make_participants()
        p_dv_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.PD_STAGES,
            producer_group=async_pair,
            consumer_group=umma_one,
            barrier_storage=p_dv_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        p_dv_producer, p_dv_consumer = p_dv_pipeline.make_participants()
        ds_dk_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.PD_STAGES,
            producer_group=async_pair,
            consumer_group=umma_one,
            barrier_storage=ds_dk_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ds_dk_producer, ds_dk_consumer = (
            ds_dk_pipeline.make_participants()
        )
        ds_dq_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.PD_STAGES,
            producer_group=async_pair,
            consumer_group=umma_one,
            barrier_storage=ds_dq_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ds_dq_producer, ds_dq_consumer = (
            ds_dq_pipeline.make_participants()
        )
        dkv_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.ROUND_STAGES,
            producer_group=umma_one,
            consumer_group=reduce_pair,
            barrier_storage=dkv_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        dq_final_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.ROUND_STAGES,
            producer_group=umma_one,
            consumer_group=reduce_pair,
            barrier_storage=dq_mbars_ptr,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Custom metadata/source/DSM barriers are not standard UMMA
        # pipelines, but they are initialized once in the same generation.
        if tidx == 0:
            self._init_pair_mbar_range(
                issued_ctx_mbars_ptr,
                self.ISSUED_CTX_PIPELINE_MBAR_COUNT,
            )
            self._init_pair_mbar_range(
                reducer_ctx_mbars_ptr,
                self.REDUCER_CTX_PIPELINE_MBAR_COUNT,
            )
            self._init_mbar_range(
                score_q_source_mbars_ptr,
                self.OP_STAGES,
            )
            self._init_mbar_range(
                score_do_source_mbars_ptr,
                self.OP_STAGES,
            )
            self._init_mbar_range(
                score_kv_source_mbars_ptr,
                self.OP_STAGES,
            )
            self._init_mbar_range(
                grad_do_source_mbars_ptr,
                self.OP_STAGES,
            )
            self._init_mbar_range(
                grad_k_source_mbars_ptr,
                self.OP_STAGES,
            )
            self._init_mbar_range(
                grad_q_source_mbars_ptr,
                self.OP_STAGES,
            )
            self._init_mbar_range(
                p_local_store_ready_mbars_ptr,
                self.PD_STAGES,
            )
            self._init_mbar_range(
                p_remote_full_mbars_ptr,
                self.PD_STAGES,
            )
            self._init_mbar_range(
                p_pair_ready_mbars_ptr,
                self.PD_STAGES,
            )
            self._init_mbar_range(
                ds_local_store_ready_mbars_ptr,
                self.PD_STAGES,
            )
            self._init_mbar_range(
                ds_remote_full_mbars_ptr,
                self.PD_STAGES,
            )
            self._init_mbar_range(
                ds_pair_ready_mbars_ptr,
                self.PD_STAGES,
            )
            for stage in cutlass.range_constexpr(
                self.CONTEXT_STAGES
            ):
                if cutlass.const_expr(
                    self.DIAGNOSTIC_AUX_STAGE >= 3
                ):
                    cute.arch.mbarrier_init(
                        ctx_reader_done_mbars_ptr + stage,
                        3,
                    )
                else:
                    cute.arch.mbarrier_init(
                        ctx_reader_done_mbars_ptr + stage,
                        2,
                    )
            self._init_pair_mbar_range(
                descriptor_consensus_mbars_ptr,
                self.REDUCER_STAGES,
            )
            self._init_pair_mbar_range(
                issued_stream_done_ack_mbars_ptr,
                2,
            )
            cute.arch.mbarrier_init(
                p_xchg_source_done_mbar_ptr,
                1,
            )
            cute.arch.mbarrier_init(
                ds_xchg_source_done_mbar_ptr,
                1,
            )
            cute.arch.mbarrier_init(
                operand_consumer_done_mbar_ptr,
                1,
            )
            cute.arch.mbarrier_init(
                dq_epilogue_source_done_mbar_ptr,
                1,
            )
            cute.arch.mbarrier_init(
                outer_role_drain_mbar_ptr,
                1,
            )

        # Only the isolated lifecycle diagnostic needs a synthetic operand
        # payload.  The default path fully overwrites every live F/BV/BQ
        # region and avoids a redundant 144-KiB CTA-wide clear.
        if cutlass.const_expr(self.DIAGNOSTIC_OPERAND_ONLY):
            for element in cutlass.range(
                tidx,
                cute.size(raw_operand),
                self.THREADS_PER_CTA,
            ):
                raw_operand[element] = cutlass.Uint8(0)
        for element in cutlass.range(
            tidx,
            cute.size(raw_p_dv),
            self.THREADS_PER_CTA,
        ):
            raw_p_dv[element] = self.element_dtype(0.0)
            raw_ds_dk[element] = self.element_dtype(0.0)
            raw_ds_dq[element] = self.element_dtype(0.0)
        for element in cutlass.range(
            tidx,
            self.XCHG_ELEMENTS,
            self.THREADS_PER_CTA,
        ):
            raw_p_xchg[element] = self.element_dtype(0.0)
            raw_ds_xchg[element] = self.element_dtype(0.0)

        if tidx < Int32(
            self.CONTEXT_STAGES
            * self.ISSUED_TILE_CONTEXT_WORDS
        ):
            context_slot = (
                tidx // Int32(self.ISSUED_TILE_CONTEXT_WORDS)
            )
            context_word = (
                tidx % Int32(self.ISSUED_TILE_CONTEXT_WORDS)
            )
            issued_ctx_ring[
                context_word,
                context_slot,
            ] = Int32(-1)
        if tidx < Int32(
            self.REDUCER_STAGES
            * self.REDUCER_CONTEXT_WORDS
        ):
            reducer_slot = (
                tidx // Int32(self.REDUCER_CONTEXT_WORDS)
            )
            reducer_word = (
                tidx % Int32(self.REDUCER_CONTEXT_WORDS)
            )
            reducer_ctx_ring[
                reducer_word,
                reducer_slot,
            ] = Int32(0)
        if tidx < Int32(self.CONTEXT_STAGES * self.N_TILE):
            load_context_slot = tidx // Int32(self.N_TILE)
            load_context_word = tidx % Int32(self.N_TILE)
            load_ctx_ring[
                load_context_word,
                load_context_slot,
            ] = Int32(-1)
        if tidx < Int32(
            self.ISSUED_STREAM_STATE_BYTES // 4
        ):
            issued_stream_state[tidx] = cutlass.Uint32(0)
        if tidx == 0:
            traversal_descriptor[0] = Int32(0)
            stats[0] = cutlass.Uint32(0)

        cute.arch.fence_view_async_shared()
        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk,
            is_relaxed=False,
        )
        pipeline.pipeline_init_wait(
            cluster_shape_mn=cluster_layout_vmnk,
        )

        tmem = utils.TmemAllocator(
            tmem_holding_buf_ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.MMA_WARP,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        score_c_shape = score_tiled_mma.partition_shape_C(
            (self.H_TILE_CLUSTER, self.N_TILE)
        )
        score_c_layout = score_tiled_mma.make_fragment_C(
            score_c_shape
        ).layout
        dp_c_shape = dp_tiled_mma.partition_shape_C(
            (self.H_TILE_CLUSTER, self.N_TILE)
        )
        dp_c_layout = dp_tiled_mma.make_fragment_C(
            dp_c_shape
        ).layout
        dkv_c_shape = dkv_tiled_mma.partition_shape_C(
            self.DKV_MMA_TILER[:2]
        )
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(
            dkv_c_shape
        ).layout
        dq_c_shape = dq_tiled_mma.partition_shape_C(
            self.DQ_MMA_TILER[:2]
        )
        dq_c_layout = dq_tiled_mma.make_fragment_C(
            dq_c_shape
        ).layout

        t_score = cute.make_tensor(
            tmem_ptr + self.TMEM_S_OFFSET,
            score_c_layout,
        )
        t_dp = cute.make_tensor(
            tmem_ptr + self.TMEM_DP_OFFSET,
            dp_c_layout,
        )
        t_dkv = (
            cute.make_tensor(
                tmem_ptr + self.TMEM_DKV0_OFFSET,
                dkv_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr + self.TMEM_DKV1_OFFSET,
                dkv_c_layout,
            ),
        )
        t_dq = (
            cute.make_tensor(
                tmem_ptr + self.TMEM_DQ0_OFFSET,
                dq_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr + self.TMEM_DQ1_OFFSET,
                dq_c_layout,
            ),
        )
        tmem.relinquish_alloc_permit()

        score_a_layout = cute.select(
            score_a_layout_staged,
            mode=[0, 1, 2],
        )
        score_b_layout = cute.select(
            score_b_layout_staged,
            mode=[0, 1, 2],
        )
        dkv_a_layout = cute.select(
            dkv_a_layout_staged,
            mode=[0, 1, 2],
        )
        dkv_b_layout = cute.select(
            dkv_b_layout_staged,
            mode=[0, 1, 2],
        )
        dq_a_layout = cute.select(
            dq_a_layout_staged,
            mode=[0, 1, 2],
        )
        dq_b_layout = cute.select(
            dq_b_layout_staged,
            mode=[0, 1, 2],
        )
        # P/dS use the score T2R distribution.  Its COL_MAJOR H64xN64 byte
        # image is also the exact pair of K-major dKV H64 quadrants, so one
        # epilogue alias serves both dKV operands, dS_dQ, and both xchg images.
        score_store_layout = (
            sm100_utils.make_smem_layout_epi(
                self.element_dtype,
                utils.LayoutEnum.COL_MAJOR,
                (self.H_TILE_CTA, self.N_TILE),
                1,
            )
        )
        assert (
            cute.cosize(score_store_layout)
            == self.PD_LOCAL_ELEMENTS_PER_STAGE
        )
        score_store_domain = cute.make_layout(
            (
                score_store_layout.outer.shape,
                1,
                1,
                1,
            ),
            stride=(
                score_store_layout.outer.stride,
                0,
                0,
                0,
            ),
        )
        assert (
            cute.cosize(score_store_domain)
            == self.PD_LOCAL_ELEMENTS_PER_STAGE
        )

        rank_score_mma = score_tiled_mma.get_slice(rank)
        rank_dp_mma = dp_tiled_mma.get_slice(rank)
        rank_dkv_mma = dkv_tiled_mma.get_slice(rank)
        rank_dq_mma = dq_tiled_mma.get_slice(rank)
        rank_score_coordinates = rank_score_mma.partition_C(
            cute.make_identity_tensor(
                (self.H_TILE_CLUSTER, self.N_TILE)
            )
        )
        rank_dkv_coordinates = rank_dkv_mma.partition_C(
            cute.make_identity_tensor(self.DKV_MMA_TILER[:2])
        )
        rank_dq_coordinates = rank_dq_mma.partition_C(
            cute.make_identity_tensor(self.DQ_MMA_TILER[:2])
        )
        a_cta_layout = cute.make_layout(
            cute.slice_(
                cluster_layout_vmnk,
                (0, 0, None, 0),
            ).shape
        )

        g_q = cute.local_tile(
            tma_tensor_q,
            cute.select(
                (
                    self.H_TILE_CLUSTER,
                    self.N_TILE,
                    self.K_CHUNK,
                ),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        g_do = cute.local_tile(
            tma_tensor_do,
            cute.select(
                (
                    self.H_TILE_CLUSTER,
                    self.N_TILE,
                    self.K_CHUNK,
                ),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        rank_g_q = rank_score_mma.partition_A(g_q)
        rank_g_do = rank_dp_mma.partition_A(g_do)

        g_qt = cute.local_tile(
            tma_tensor_qt,
            cute.select(self.DKV_MMA_TILER, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        g_dot = cute.local_tile(
            tma_tensor_dot,
            cute.select(self.DKV_MMA_TILER, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        rank_g_qt = rank_dkv_mma.partition_A(g_qt)
        rank_g_dot = rank_dkv_mma.partition_A(g_dot)

        kv_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(
                cache_mode=cpasync.LoadCacheMode.GLOBAL,
            ),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        kv_tiled_copy = cute.make_tiled_copy_tv(
            kv_copy_atom,
            cute.make_layout((1,)),
            cute.make_layout((8,)),
        )
        kv_thread_copy = kv_tiled_copy.get_slice(0)

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = mTopkIdxs.shape[0]
        assert self.max_topk % self.N_TILE == 0
        traversal_tile_count = (
            topk + Int32(self.N_TILE - 1)
        ) // Int32(self.N_TILE)
        traversal_capacity = self.max_topk // self.N_TILE
        if traversal_tile_count > Int32(traversal_capacity):
            traversal_tile_count = Int32(traversal_capacity)
        if traversal_tile_count < Int32(0):
            traversal_tile_count = Int32(0)

        if warp_idx == self.LOAD_COORDINATOR_WARP:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_dot)
        elif warp_idx == self.DESCRIPTOR_WARP:
            cpasync.prefetch_descriptor(tma_atom_qt)
            cpasync.prefetch_descriptor(tma_atom_dq_epi)

        if cutlass.const_expr(self.DIAGNOSTIC_OPERAND_ONLY):
            # Exact production block/SMEM/TMEM geometry, but only the
            # PipelineAsyncUmma whole FIFO from the standalone PASS probe.
            # This path is never selected by the production class default.
            if warp_idx == self.LOAD_COORDINATOR_WARP:
                with cute.arch.elect_one():
                    diagnostic_producer = op_producer.clone()
                    ordinal = Int32(0)
                    diagnostic_task_count = (
                        Int32(self.K_CHUNKS + self.D_ROUNDS)
                        * traversal_tile_count
                    )
                    while ordinal < diagnostic_task_count:
                        diagnostic_slot = diagnostic_producer.index
                        producer_handle = (
                            diagnostic_producer.acquire_and_advance()
                        )
                        raw_slots[
                            self.OP_MAIN_OFFSET_BYTES,
                            diagnostic_slot,
                        ] = cutlass.Uint8(0)
                        raw_slots[
                            self.OP_SIDE_OFFSET_BYTES,
                            diagnostic_slot,
                        ] = cutlass.Uint8(0)
                        cute.arch.fence_view_async_shared()
                        producer_handle.commit()
                        ordinal += Int32(1)
                    if diagnostic_task_count > Int32(0):
                        diagnostic_producer.tail()

            if is_leader_cta and warp_idx == self.MMA_WARP:
                diagnostic_consumer = op_consumer.clone()
                ordinal = Int32(0)
                diagnostic_task_count = (
                    Int32(self.K_CHUNKS + self.D_ROUNDS)
                    * traversal_tile_count
                )
                while ordinal < diagnostic_task_count:
                    diagnostic_slot = diagnostic_consumer.index
                    consumer_handle = (
                        diagnostic_consumer.wait_and_advance()
                    )
                    diagnostic_a = self._make_operand_slot_view(
                        raw_slots,
                        diagnostic_slot,
                        self.OP_MAIN_OFFSET_BYTES,
                        score_a_layout,
                    )
                    diagnostic_b = self._make_operand_slot_view(
                        raw_slots,
                        diagnostic_slot,
                        self.OP_SIDE_OFFSET_BYTES,
                        score_b_layout,
                    )
                    diagnostic_a_fragment = (
                        score_tiled_mma.make_fragment_A(
                            diagnostic_a
                        )
                    )
                    diagnostic_b_fragment = (
                        score_tiled_mma.make_fragment_B(
                            diagnostic_b
                        )
                    )
                    diagnostic_mma = score_tiled_mma.with_()
                    diagnostic_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        False,
                    )
                    for k_block in cutlass.range_constexpr(
                        cute.size(
                            diagnostic_a_fragment,
                            mode=[2],
                        )
                    ):
                        cute.gemm(
                            diagnostic_mma,
                            t_score,
                            diagnostic_a_fragment[
                                None,
                                None,
                                k_block,
                            ],
                            diagnostic_b_fragment[
                                None,
                                None,
                                k_block,
                            ],
                            t_score,
                        )
                        diagnostic_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            True,
                        )
                    consumer_handle.release()
                    ordinal += Int32(1)

            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tmem.free(tmem_ptr)
            return

        # The loader derives the fixed superstep from committed contexts.
        # It never waits for the final count: each successor is resolved by
        # exactly one IssuedCtx-full generation or the independent sticky
        # done state.
        if (
            warp_idx <= self.GATHER_WARPS[-1]
            or warp_idx == self.LOAD_COORDINATOR_WARP
        ):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.OP_STAGES,
            )
            first_valid = self._resolve_issued_context_or_done(
                Int32(0),
                issued_ctx_mbars_ptr,
                issued_stream_state,
                issued_stream_done_ack_mbars_ptr,
            )
            issue_seq = Int32(0)
            if first_valid:
                self._snapshot_load_context(
                    Int32(0),
                    issued_ctx_ring,
                    load_ctx_ring,
                    ctx_reader_done_mbars_ptr,
                    tidx,
                )
                for chunk in cutlass.range_constexpr(self.K_CHUNKS):
                    producer_state = self._load_f_task(
                        raw_slots,
                        score_a_layout,
                        score_b_layout,
                        tma_atom_q,
                        tma_atom_do,
                        rank_g_q,
                        rank_g_do,
                        block_coord_vmnk,
                        a_cta_layout,
                        mKV,
                        load_ctx_ring,
                        batch_idx,
                        Int32(0),
                        chunk,
                        rank,
                        tidx,
                        warp_idx,
                        kv_copy_atom,
                        kv_thread_copy,
                        score_q_source_mbars_ptr,
                        score_do_source_mbars_ptr,
                        op_pipeline,
                        producer_state,
                        score_a_stage_bytes,
                        token_idx,
                        trace_buffer,
                        trace_token_idx,
                        trace_batch_idx,
                    )

                active = cutlass.Boolean(True)
                while active:
                    next_seq = issue_seq + Int32(1)
                    has_next = self._resolve_issued_context_or_done(
                        next_seq,
                        issued_ctx_mbars_ptr,
                        issued_stream_state,
                        issued_stream_done_ack_mbars_ptr,
                    )
                    if has_next:
                        self._snapshot_load_context(
                            next_seq,
                            issued_ctx_ring,
                            load_ctx_ring,
                            ctx_reader_done_mbars_ptr,
                            tidx,
                        )
                    for local_task in cutlass.range_constexpr(
                        self.K_CHUNKS + self.D_ROUNDS
                    ):
                        if cutlass.const_expr(
                            local_task < self.K_CHUNKS
                        ):
                            if has_next:
                                producer_state = self._load_f_task(
                                    raw_slots,
                                    score_a_layout,
                                    score_b_layout,
                                    tma_atom_q,
                                    tma_atom_do,
                                    rank_g_q,
                                    rank_g_do,
                                    block_coord_vmnk,
                                    a_cta_layout,
                                    mKV,
                                    load_ctx_ring,
                                    batch_idx,
                                    next_seq,
                                    local_task,
                                    rank,
                                    tidx,
                                    warp_idx,
                                    kv_copy_atom,
                                    kv_thread_copy,
                                    score_q_source_mbars_ptr,
                                    score_do_source_mbars_ptr,
                                    op_pipeline,
                                    producer_state,
                                    score_a_stage_bytes,
                                    token_idx,
                                    trace_buffer,
                                    trace_token_idx,
                                    trace_batch_idx,
                                )
                        else:
                            round_index = (
                                local_task - self.K_CHUNKS
                            )
                            producer_state = self._load_bv_task(
                                raw_slots,
                                dkv_a_layout,
                                dq_a_layout,
                                tma_atom_dot,
                                rank_g_dot,
                                block_coord_vmnk,
                                a_cta_layout,
                                mKV,
                                load_ctx_ring,
                                batch_idx,
                                issue_seq,
                                round_index,
                                rank,
                                tidx,
                                warp_idx,
                                kv_copy_atom,
                                kv_thread_copy,
                                grad_do_source_mbars_ptr,
                                grad_k_source_mbars_ptr,
                                op_pipeline,
                                producer_state,
                                grad_a_stage_bytes,
                                token_idx,
                                trace_buffer,
                                trace_token_idx,
                                trace_batch_idx,
                            )
                    issue_seq += Int32(1)
                    active = has_next

            if first_valid and warp_idx == self.LOAD_COORDINATOR_WARP:
                # Reconstruct the final producer state algebraically.  This
                # keeps producer_tail independent of values defined in the
                # runtime superstep's child region.
                with cute.arch.elect_one():
                    task_count = (
                        Int32(self.K_CHUNKS + self.D_ROUNDS)
                        * issue_seq
                    )
                    tail_state = pipeline.PipelineState(
                        self.OP_STAGES,
                        task_count,
                        task_count % Int32(self.OP_STAGES),
                        Int32(1)
                        ^ (
                            (
                                task_count
                                // Int32(self.OP_STAGES)
                            )
                            & Int32(1)
                        ),
                    )
                    stats[0] = cutlass.Uint32(task_count)
                    op_pipeline.producer_tail(tail_state)

            if warp_idx == self.LOAD_COORDINATOR_WARP:
                with cute.arch.elect_one():
                    self._wait_pair(
                        issued_stream_done_ack_mbars_ptr
                        + self.STREAM_DONE_FULL_MBAR,
                        Int32(0),
                    )
                    self._pair_arrive(
                        issued_stream_done_ack_mbars_ptr
                        + self.STREAM_DONE_ACK_MBAR,
                        peer_rank,
                    )

        # W7 owns descriptor production only.  It can wait for IssuedCtx
        # ring credit without delaying Q replacement, which is serviced by
        # W5 in both CTAs below.
        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 3
            )
            and warp_idx == self.DESCRIPTOR_WARP
        ):
            traversal_seq = Int32(0)
            committed_count = Int32(0)
            pending_descriptor = cutlass.Boolean(False)
            while traversal_seq < traversal_tile_count:
                logical_tile = (
                    traversal_tile_count
                    - Int32(1)
                    - traversal_seq
                )
                descriptor_slot = (
                    traversal_seq
                    % Int32(self.CONTEXT_STAGES)
                )
                descriptor_phase = (
                    traversal_seq
                    // Int32(self.CONTEXT_STAGES)
                ) & Int32(1)
                if tidx == Int32(self.DESCRIPTOR_WARP * 32):
                    self._record_trace(
                        trace_buffer,
                        token_idx,
                        batch_idx,
                        trace_token_idx,
                        trace_batch_idx,
                        rank,
                        TRACE_ROLE_DESC_BQ,
                        traversal_seq,
                        TRACE_DESC_BEGIN,
                    )
                self._decode_traversal_descriptor(
                    mTopkIdxs,
                    traversal_descriptor,
                    token_idx,
                    batch_idx,
                    topk,
                    logical_tile,
                )
                if tidx == Int32(self.DESCRIPTOR_WARP * 32):
                    self._record_trace(
                        trace_buffer,
                        token_idx,
                        batch_idx,
                        trace_token_idx,
                        trace_batch_idx,
                        rank,
                        TRACE_ROLE_DESC_BQ,
                        traversal_seq,
                        TRACE_DESC_END,
                    )
                with cute.arch.elect_one():
                    self._pair_arrive(
                        descriptor_consensus_mbars_ptr
                        + descriptor_slot,
                        peer_rank,
                    )
                    self._wait_pair(
                        descriptor_consensus_mbars_ptr
                        + descriptor_slot,
                        descriptor_phase,
                    )
                cute.arch.sync_warp()
                cute.arch.fence_view_async_shared()

                if (
                    traversal_descriptor[
                        self.DESCRIPTOR_EXECUTE_WORD
                    ]
                    != Int32(0)
                ):
                    slot = (
                        committed_count
                        % Int32(self.CONTEXT_STAGES)
                    )
                    epoch = (
                        committed_count
                        // Int32(self.CONTEXT_STAGES)
                    ) & Int32(1)
                    with cute.arch.elect_one():
                        self._wait_pair(
                            issued_ctx_mbars_ptr
                            + self.ISSUED_EMPTY_MBAR_BASE
                            + slot,
                            epoch ^ Int32(1),
                        )
                    cute.arch.sync_warp()
                    self._publish_issued_context(
                        committed_count,
                        traversal_descriptor,
                        issued_ctx_ring,
                        issued_ctx_mbars_ptr,
                        peer_rank,
                    )
                    if tidx == Int32(
                        self.DESCRIPTOR_WARP * 32
                    ):
                        self._record_trace(
                            trace_buffer,
                            token_idx,
                            batch_idx,
                            trace_token_idx,
                            trace_batch_idx,
                            rank,
                            TRACE_ROLE_DESC_BQ,
                            committed_count,
                            TRACE_CTX_COMMIT,
                        )
                    committed_count += Int32(1)
                traversal_seq += Int32(1)

            with cute.arch.elect_one():
                self._publish_issued_stream_done(
                    token_idx,
                    committed_count,
                    issued_stream_state,
                    issued_stream_done_ack_mbars_ptr,
                    peer_rank,
                )
            self._record_trace(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_DESC_BQ,
                Int32(TRACE_ISSUE_SLOTS - 1),
                TRACE_STREAM_DONE,
            )
            cute.arch.sync_warp()

            with cute.arch.elect_one():
                self._wait_pair(
                    issued_stream_done_ack_mbars_ptr
                    + self.STREAM_DONE_ACK_MBAR,
                    Int32(0),
                )

        # The nonleader W5 mirrors the leader's issued-tile walk and produces
        # the peer CTA's half of every Q-full generation.  Both Q rounds are
        # issued before either completion is waited, matching the leader W5
        # sequence while leaving W7 completely independent.
        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 3
            )
            and not is_leader_cta
            and warp_idx == self.MMA_WARP
        ):
            issue_seq = Int32(0)
            whole_ordinal = Int32(self.K_CHUNKS)
            active = self._resolve_issued_context_or_done(
                issue_seq,
                issued_ctx_mbars_ptr,
                issued_stream_state,
                issued_stream_done_ack_mbars_ptr,
            )
            if active:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        ctx_reader_done_mbars_ptr
                    )
            while active:
                has_next = self._resolve_issued_context_or_done(
                    issue_seq + Int32(1),
                    issued_ctx_mbars_ptr,
                    issued_stream_state,
                    issued_stream_done_ack_mbars_ptr,
                )
                if has_next:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(
                            ctx_reader_done_mbars_ptr
                            + (
                                (issue_seq + Int32(1))
                                % Int32(self.CONTEXT_STAGES)
                            )
                        )
                if has_next:
                    whole_ordinal += Int32(self.K_CHUNKS)
                first_refill_count = (
                    issue_seq * Int32(self.D_ROUNDS)
                )
                first_slot = (
                    whole_ordinal % Int32(self.OP_STAGES)
                )
                second_slot = (
                    (whole_ordinal + Int32(1))
                    % Int32(self.OP_STAGES)
                )
                self._begin_bq_refill(
                    0,
                    first_slot,
                    first_refill_count,
                    raw_slots,
                    dkv_a_layout,
                    tma_atom_qt,
                    rank_g_qt,
                    block_coord_vmnk,
                    a_cta_layout,
                    grad_q_source_mbars_ptr,
                    do_empty_pipeline,
                    q_full_pipeline,
                    grad_a_stage_bytes,
                    issue_seq,
                    rank,
                    tidx,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )
                self._begin_bq_refill(
                    1,
                    second_slot,
                    first_refill_count + Int32(1),
                    raw_slots,
                    dkv_a_layout,
                    tma_atom_qt,
                    rank_g_qt,
                    block_coord_vmnk,
                    a_cta_layout,
                    grad_q_source_mbars_ptr,
                    do_empty_pipeline,
                    q_full_pipeline,
                    grad_a_stage_bytes,
                    issue_seq,
                    rank,
                    tidx,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )
                self._finish_bq_refill(
                    0,
                    first_slot,
                    first_refill_count,
                    grad_q_source_mbars_ptr,
                    do_empty_pipeline,
                    q_full_pipeline,
                    issue_seq,
                    rank,
                    tidx,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )
                self._finish_bq_refill(
                    1,
                    second_slot,
                    first_refill_count + Int32(1),
                    grad_q_source_mbars_ptr,
                    do_empty_pipeline,
                    q_full_pipeline,
                    issue_seq,
                    rank,
                    tidx,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )
                whole_ordinal += Int32(self.D_ROUNDS)
                issue_seq += Int32(1)
                active = has_next

            if issue_seq > Int32(0):
                with cute.arch.elect_one():
                    q_count = (
                        Int32(self.D_ROUNDS) * issue_seq
                    )
                    q_tail_state = pipeline.PipelineState(
                        self.ROUND_STAGES,
                        q_count,
                        q_count % Int32(self.ROUND_STAGES),
                        Int32(1)
                        ^ (
                            (
                                q_count
                                // Int32(self.ROUND_STAGES)
                            )
                            & Int32(1)
                        ),
                    )
                    q_full_pipeline.producer_tail(q_tail_state)

        # The leader CTA's MMA warp is the only CG2 issue role.  Each F
        # ordinal performs real QK+dOV before releasing the whole slot.  Each
        # BV ordinal performs real dV, waits the Q refill, then dK+dQ.
        if is_leader_cta and warp_idx == self.MMA_WARP:
            op_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.OP_STAGES,
            )
            s_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            dp_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            p_wait_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.PD_STAGES,
            )
            p_release_state = p_wait_state.clone()
            dsk_wait_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.PD_STAGES,
            )
            dsk_release_state = dsk_wait_state.clone()
            dsq_wait_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.PD_STAGES,
            )
            dsq_release_state = dsq_wait_state.clone()
            do_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            q_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.ROUND_STAGES,
            )
            dkv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            dq_final_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )

            first_valid = self._resolve_issued_context_or_done(
                Int32(0),
                issued_ctx_mbars_ptr,
                issued_stream_state,
                issued_stream_done_ack_mbars_ptr,
            )
            if first_valid:
                if cutlass.const_expr(
                    self.DIAGNOSTIC_AUX_STAGE >= 3
                ):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(
                            ctx_reader_done_mbars_ptr
                        )
                op_state, s_state, dp_state = self._mma_sdp_tile(
                    raw_slots,
                    score_a_layout,
                    score_b_layout,
                    score_tiled_mma,
                    dp_tiled_mma,
                    t_score,
                    t_dp,
                    op_pipeline,
                    op_state,
                    s_pipeline,
                    s_state,
                    dp_pipeline,
                    dp_state,
                    Int32(0),
                    rank,
                    tidx,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )

                # The first actual issued tile is a fixed lexical region.
                # Both dQ round accumulators are therefore initialized with
                # ACCUMULATE=False without a runtime MMA control operand.
                has_next = self._resolve_issued_context_or_done(
                    Int32(1),
                    issued_ctx_mbars_ptr,
                    issued_stream_state,
                    issued_stream_done_ack_mbars_ptr,
                )
                if has_next:
                    if cutlass.const_expr(
                        self.DIAGNOSTIC_AUX_STAGE >= 3
                    ):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                ctx_reader_done_mbars_ptr
                                + Int32(1)
                            )
                    op_state, s_state, dp_state = (
                        self._mma_sdp_tile(
                            raw_slots,
                            score_a_layout,
                            score_b_layout,
                            score_tiled_mma,
                            dp_tiled_mma,
                            t_score,
                            t_dp,
                            op_pipeline,
                            op_state,
                            s_pipeline,
                            s_state,
                            dp_pipeline,
                            dp_state,
                            Int32(1),
                            rank,
                            tidx,
                            token_idx,
                            batch_idx,
                            trace_buffer,
                            trace_token_idx,
                            trace_batch_idx,
                        )
                    )
                first_is_final = not has_next
                (
                    op_state,
                    p_wait_state,
                    p_release_state,
                    dsk_wait_state,
                    dsk_release_state,
                    dsq_wait_state,
                    dsq_release_state,
                    do_state,
                    q_state,
                    dkv_producer_state,
                    dq_final_producer_state,
                ) = self._mma_grad_pair(
                    False,
                    first_is_final,
                    Int32(0),
                    raw_slots,
                    raw_p_dv,
                    raw_ds_dk,
                    raw_ds_dq,
                    dkv_a_layout,
                    dq_a_layout,
                    dkv_b_layout,
                    dq_b_layout,
                    dkv_tiled_mma,
                    dq_tiled_mma,
                    t_dkv[0],
                    t_dkv[1],
                    t_dq[0],
                    t_dq[1],
                    op_pipeline,
                    op_state,
                    p_dv_pipeline,
                    p_wait_state,
                    p_release_state,
                    ds_dk_pipeline,
                    dsk_wait_state,
                    dsk_release_state,
                    ds_dq_pipeline,
                    dsq_wait_state,
                    dsq_release_state,
                    do_empty_pipeline,
                    do_state,
                    q_full_pipeline,
                    q_state,
                    tma_atom_qt,
                    rank_g_qt,
                    block_coord_vmnk,
                    a_cta_layout,
                    grad_q_source_mbars_ptr,
                    grad_a_stage_bytes,
                    dkv_pipeline,
                    dkv_producer_state,
                    dq_final_pipeline,
                    dq_final_producer_state,
                    rank,
                    tidx,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )

                issue_seq = Int32(1)
                active = has_next
                while active:
                    has_next = self._resolve_issued_context_or_done(
                        issue_seq + Int32(1),
                        issued_ctx_mbars_ptr,
                        issued_stream_state,
                        issued_stream_done_ack_mbars_ptr,
                    )
                    if has_next:
                        if cutlass.const_expr(
                            self.DIAGNOSTIC_AUX_STAGE >= 3
                        ):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(
                                    ctx_reader_done_mbars_ptr
                                    + (
                                        (
                                            issue_seq
                                            + Int32(1)
                                        )
                                        % Int32(
                                            self.CONTEXT_STAGES
                                        )
                                    )
                                )
                        op_state, s_state, dp_state = (
                            self._mma_sdp_tile(
                                raw_slots,
                                score_a_layout,
                                score_b_layout,
                                score_tiled_mma,
                                dp_tiled_mma,
                                t_score,
                                t_dp,
                                op_pipeline,
                                op_state,
                                s_pipeline,
                                s_state,
                                dp_pipeline,
                                dp_state,
                                issue_seq + Int32(1),
                                rank,
                                tidx,
                                token_idx,
                                batch_idx,
                                trace_buffer,
                                trace_token_idx,
                                trace_batch_idx,
                            )
                        )
                    is_final = not has_next
                    (
                        op_state,
                        p_wait_state,
                        p_release_state,
                        dsk_wait_state,
                        dsk_release_state,
                        dsq_wait_state,
                        dsq_release_state,
                        do_state,
                        q_state,
                        dkv_producer_state,
                        dq_final_producer_state,
                    ) = self._mma_grad_pair(
                        True,
                        is_final,
                        issue_seq,
                        raw_slots,
                        raw_p_dv,
                        raw_ds_dk,
                        raw_ds_dq,
                        dkv_a_layout,
                        dq_a_layout,
                        dkv_b_layout,
                        dq_b_layout,
                        dkv_tiled_mma,
                        dq_tiled_mma,
                        t_dkv[0],
                        t_dkv[1],
                        t_dq[0],
                        t_dq[1],
                        op_pipeline,
                        op_state,
                        p_dv_pipeline,
                        p_wait_state,
                        p_release_state,
                        ds_dk_pipeline,
                        dsk_wait_state,
                        dsk_release_state,
                        ds_dq_pipeline,
                        dsq_wait_state,
                        dsq_release_state,
                        do_empty_pipeline,
                        do_state,
                        q_full_pipeline,
                        q_state,
                        tma_atom_qt,
                        rank_g_qt,
                        block_coord_vmnk,
                        a_cta_layout,
                        grad_q_source_mbars_ptr,
                        grad_a_stage_bytes,
                        dkv_pipeline,
                        dkv_producer_state,
                        dq_final_pipeline,
                        dq_final_producer_state,
                        rank,
                        tidx,
                        token_idx,
                        batch_idx,
                        trace_buffer,
                        trace_token_idx,
                        trace_batch_idx,
                    )
                    issue_seq += Int32(1)
                    active = has_next

                stats[1] = cutlass.Uint32(
                    Int32(self.K_CHUNKS + self.D_ROUNDS)
                    * issue_seq
                )
                if cutlass.const_expr(
                    self.DIAGNOSTIC_AUX_STAGE >= 1
                ):
                    sdp_phase = (
                        Int32(1)
                        ^ (issue_seq & Int32(1))
                    )
                    s_tail_state = pipeline.PipelineState(
                        1,
                        issue_seq,
                        Int32(0),
                        sdp_phase,
                    )
                    dp_tail_state = pipeline.PipelineState(
                        1,
                        issue_seq,
                        Int32(0),
                        sdp_phase,
                    )
                    s_pipeline.producer_tail(s_tail_state)
                    dp_pipeline.producer_tail(dp_tail_state)
                if cutlass.const_expr(
                    self.DIAGNOSTIC_AUX_STAGE >= 3
                ):
                    grad_count = (
                        Int32(self.D_ROUNDS)
                        * issue_seq
                    )
                    grad_phase = (
                        Int32(1)
                        ^ (
                            (
                                grad_count
                                // Int32(self.ROUND_STAGES)
                            )
                            & Int32(1)
                        )
                    )
                    do_tail_state = pipeline.PipelineState(
                        self.ROUND_STAGES,
                        grad_count,
                        grad_count
                        % Int32(self.ROUND_STAGES),
                        grad_phase,
                    )
                    do_empty_pipeline.producer_tail(
                        do_tail_state
                    )
                    q_tail_state = pipeline.PipelineState(
                        self.ROUND_STAGES,
                        grad_count,
                        grad_count
                        % Int32(self.ROUND_STAGES),
                        grad_phase,
                    )
                    with cute.arch.elect_one():
                        q_full_pipeline.producer_tail(
                            q_tail_state
                        )
                if cutlass.const_expr(
                    self.DIAGNOSTIC_AUX_STAGE >= 4
                ):
                    dkv_tail_state = pipeline.PipelineState(
                        self.ROUND_STAGES,
                        grad_count,
                        grad_count
                        % Int32(self.ROUND_STAGES),
                        grad_phase,
                    )
                    dkv_pipeline.producer_tail(
                        dkv_tail_state
                    )

        # P/dS math consumes one final S/dP generation after all four F
        # chunks have accumulated.  W8-W11 keep CUDA-core math in FP32 and
        # publish retained P/dS only after W6's directed DSM exchanges.
        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 2
            )
            and warp_idx >= self.MATH_WARPS[0]
            and warp_idx <= self.MATH_WARPS[-1]
        ):
            self._math_role(
                self.math_barrier,
                tidx,
                rank,
                token_idx,
                batch_idx,
                issued_ctx_ring,
                issued_stream_state,
                issued_stream_done_ack_mbars_ptr,
                reducer_ctx_ring,
                t_score,
                t_dp,
                score_tmem_load,
                rank_score_coordinates,
                scaled_lse,
                sum_odo,
                softmax_stats,
                scale_softmax,
                s_pipeline,
                dp_pipeline,
                p_dv_pipeline,
                ds_dk_pipeline,
                ds_dq_pipeline,
                raw_p_dv,
                raw_ds_dk,
                raw_ds_dq,
                raw_p_xchg,
                raw_ds_xchg,
                dkv_b_layout,
                dq_b_layout,
                score_store_layout,
                score_store_domain,
                p_local_store_ready_mbars_ptr,
                ds_local_store_ready_mbars_ptr,
                p_remote_full_mbars_ptr,
                ds_remote_full_mbars_ptr,
                p_xchg_source_done_mbar_ptr,
                ds_xchg_source_done_mbar_ptr,
                issued_ctx_mbars_ptr,
                reducer_ctx_mbars_ptr,
                ctx_reader_done_mbars_ptr,
                trace_buffer,
                trace_token_idx,
                trace_batch_idx,
            )

        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 2
            )
            and warp_idx == self.EXCHANGE_WARP
        ):
            with cute.arch.elect_one():
                self._exchange_role(
                    rank,
                    issued_stream_state,
                    issued_stream_done_ack_mbars_ptr,
                    raw_p_dv,
                    raw_ds_dk,
                    raw_p_xchg,
                    raw_ds_xchg,
                    p_local_store_ready_mbars_ptr,
                    ds_local_store_ready_mbars_ptr,
                    p_remote_full_mbars_ptr,
                    ds_remote_full_mbars_ptr,
                    p_xchg_source_done_mbar_ptr,
                    ds_xchg_source_done_mbar_ptr,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )

        # dKV generations are interleaved [issue0/r0, issue0/r1, ...].
        # Each round-owned 128-thread role consumes its generation, then
        # advances once more to skip the other round.
        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 4
            )
            and warp_idx >= self.REDUCE_ROUND0_WARPS[0]
            and warp_idx <= self.REDUCE_ROUND0_WARPS[-1]
        ):
            local_tidx = (
                tidx - self.REDUCE_ROUND0_WARPS[0] * 32
            )
            round0_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.ROUND_STAGES,
            )
            issue_seq = Int32(0)
            active = self._resolve_reducer_context_or_done(
                issue_seq,
                reducer_ctx_mbars_ptr,
                issued_stream_state,
                issued_stream_done_ack_mbars_ptr,
            )
            while active:
                round0_state = self._reduce_dkv_round_v0(
                    t_dkv[0],
                    dkv_tmem_load,
                    rank_dkv_coordinates,
                    mdKV_acc,
                    reducer_ctx_ring,
                    0,
                    issue_seq,
                    rank,
                    local_tidx,
                    reducer_ctx_mbars_ptr,
                    dkv_pipeline,
                    round0_state,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )
                round0_state.advance()
                issue_seq += Int32(1)
                active = self._resolve_reducer_context_or_done(
                    issue_seq,
                    reducer_ctx_mbars_ptr,
                    issued_stream_state,
                    issued_stream_done_ack_mbars_ptr,
                )

        elif (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 4
            )
            and warp_idx >= self.REDUCE_ROUND1_WARPS[0]
            and warp_idx <= self.REDUCE_ROUND1_WARPS[-1]
        ):
            local_tidx = (
                tidx - self.REDUCE_ROUND1_WARPS[0] * 32
            )
            round1_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.ROUND_STAGES,
            )
            round1_state.advance()
            issue_seq = Int32(0)
            active = self._resolve_reducer_context_or_done(
                issue_seq,
                reducer_ctx_mbars_ptr,
                issued_stream_state,
                issued_stream_done_ack_mbars_ptr,
            )
            while active:
                round1_state = self._reduce_dkv_round_v0(
                    t_dkv[1],
                    dkv_tmem_load,
                    rank_dkv_coordinates,
                    mdKV_acc,
                    reducer_ctx_ring,
                    1,
                    issue_seq,
                    rank,
                    local_tidx,
                    reducer_ctx_mbars_ptr,
                    dkv_pipeline,
                    round1_state,
                    token_idx,
                    batch_idx,
                    trace_buffer,
                    trace_token_idx,
                    trace_batch_idx,
                )
                round1_state.advance()
                issue_seq += Int32(1)
                active = self._resolve_reducer_context_or_done(
                    issue_seq,
                    reducer_ctx_mbars_ptr,
                    issued_stream_state,
                    issued_stream_done_ack_mbars_ptr,
                )

        # All load/MMA/math/exchange/reducer work is drained before final dQ
        # begins to read the persistent TMEM accumulator.
        if tidx == Int32(0):
            _trace_stamp(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_CONTROL,
                Int32(TRACE_ISSUE_SLOTS - 1),
                TRACE_PRE_EPI_JOIN_BEGIN,
            )
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        cute.arch.fence_view_async_shared()
        if tidx == Int32(0):
            _trace_stamp(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_CONTROL,
                Int32(TRACE_ISSUE_SLOTS - 1),
                TRACE_PRE_EPI_JOIN_END,
            )
        final_issued_tile_count = Int32(
            issued_stream_state[self.STREAM_ISSUED_COUNT_WORD]
        )

        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 5
            )
            and warp_idx >= self.REDUCE_ROUND0_WARPS[0]
            and warp_idx <= self.REDUCE_ROUND0_WARPS[-1]
        ):
            local_tidx = (
                tidx - self.REDUCE_ROUND0_WARPS[0] * 32
            )
            dq_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.ROUND_STAGES,
            )
            if final_issued_tile_count > Int32(0):
                for round_index in cutlass.range_constexpr(
                    self.D_ROUNDS
                ):
                    dq_consumer_state = self._store_dq_round_v0(
                        t_dq[round_index],
                        dq_tmem_load,
                        rank_dq_coordinates,
                        s_dq_epi,
                        tma_atom_dq_epi,
                        tma_tensor_dq_epi,
                        dq_epilogue_source_done_mbar_ptr,
                        round_index,
                        rank,
                        token_idx,
                        batch_idx,
                        local_tidx,
                        dq_final_pipeline,
                        dq_consumer_state,
                        final_issued_tile_count - Int32(1),
                        trace_buffer,
                        trace_token_idx,
                        trace_batch_idx,
                    )
            else:
                for round_index in cutlass.range_constexpr(
                    self.D_ROUNDS
                ):
                    self._zero_dq_round_v0(
                        mdQ,
                        round_index,
                        rank,
                        token_idx,
                        batch_idx,
                        local_tidx,
                    )

        # producer_tail waits for dQ consumers, so it must execute after the
        # epilogue role rather than in W5 before the pre-epilogue join.
        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 5
            )
            and is_leader_cta
            and warp_idx == self.MMA_WARP
        ):
            dq_tail_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            if final_issued_tile_count > Int32(0):
                for _ in cutlass.range_constexpr(self.D_ROUNDS):
                    dq_tail_state.advance()
                dq_final_pipeline.producer_tail(dq_tail_state)

        # No role may free the shared 512-column allocation while a reducer,
        # epilogue consumer, or staged store remains live.
        if tidx == Int32(0):
            _trace_stamp(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_CONTROL,
                Int32(TRACE_ISSUE_SLOTS - 1),
                TRACE_FINAL_JOIN_BEGIN,
            )
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tidx == Int32(0):
            _trace_stamp(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
                TRACE_ROLE_CONTROL,
                Int32(TRACE_ISSUE_SLOTS - 1),
                TRACE_FINAL_JOIN_END,
            )
            _trace_header_end(
                trace_buffer,
                token_idx,
                batch_idx,
                trace_token_idx,
                trace_batch_idx,
                rank,
            )
        tmem.free(tmem_ptr)
