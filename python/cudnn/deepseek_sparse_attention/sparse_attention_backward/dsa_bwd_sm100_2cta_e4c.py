"""DeepSeek Sparse Attention backward, SM100 two-CTA implementation.

v_final_exp E1: chunk-streamed score-K.  The K gather publishes each
D128 chunk on its own mbarrier (cp.async commit-group ladder, no drain
of younger chunks), and the S GEMM consumes chunk c on that signal
while chunks c+1.. are still filling.  dP, the kscore lease pipeline,
the loan, and the ring are untouched.

Fixed GQA128 / D512 shape.  One 2-CTA cluster per token; all five GEMMs
run cta_group::2.  Stationary Q/dO panels are token-resident in SMEM;
the score K tile is gathered per KV tile; gradient operands stream
through a two-slot round ring.  dQ accumulates in TMEM across all KV
tiles and stores through a staged TMA epilogue; dKV drains per tile via
red.global.add into an f32 workspace.
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
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05, warp
from cutlass.cute.typing import BFloat16, Float32, Int32

from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100


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


@dsl_user_op
def _cp_async_mbarrier_arrive(
    mbar: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """cp.async.mbarrier.arrive WITHOUT .noinc (paired-protocol half).

    The DSL's cute.arch helper only wraps the .noinc form; the paired
    completion protocol needs the incrementing form (+1 pending
    immediately, -1 asynchronously when all the thread's prior cp.async
    ops complete).  Static asm string -- trivially free of .noinc.
    """

    mbar_i32 = mbar.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [mbar_i32],
        "cp.async.mbarrier.arrive.shared::cta.b64 [$0];",
        "r",
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

    def _make_score_tmem_load(self, score_cta_shape, score_epi_tile):
        """Select the S/dP T2R atom (hook: v9.3's V2 class overrides it).

        The base behavior is byte-equivalent to the original inline
        get_tmem_load_op call, so the dormant bring-up classes are
        unaffected by the hook refactor.
        """

        return sm100_utils.get_tmem_load_op(
            score_cta_shape,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            score_epi_tile,
            True,
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

        cg1 = tcgen05.CtaGroup.ONE
        cg2 = tcgen05.CtaGroup.TWO
        stationary_tiler = (
            self.H_TILE_CTA,
            self.N_TILE,
            self.D_HEAD,
        )
        stationary_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cg1,
            stationary_tiler[:2],
        )
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
        stationary_a_layout_staged = sm100_utils.make_smem_layout_a(
            stationary_tiled_mma,
            stationary_tiler,
            self.element_dtype,
            1,
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
            "stationary_a": str(stationary_a_layout_staged),
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
        assert cute.cosize(stationary_a_layout_staged) == cute.cosize(
            score_a_layout_staged
        )
        assert stationary_a_layout_staged.inner == score_a_layout_staged.inner
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
        stationary_a_layout = cute.select(
            stationary_a_layout_staged,
            mode=[0, 1, 2],
        )
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
            stationary_a_layout,
            stationary_tiler,
            stationary_tiled_mma,
        )
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdO,
            stationary_a_layout,
            stationary_tiler,
            stationary_tiled_mma,
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
        score_tmem_load = self._make_score_tmem_load(
            score_cta_shape,
            score_epi_tile,
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
            stationary_tiled_mma,
            stationary_a_layout_staged,
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
        row_local_n = [
            row_iteration * self.KV_NUM_GROUPS + group_index
            for row_iteration in range(rows_per_group)
        ]
        row_kv_index = []
        for local_n in row_local_n:
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = mTopkIdxs[
                    topk_slot,
                    (token_idx, batch_idx),
                ]
            row_kv_index.append(kv_index)

        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_local_n[row_iteration]
            kv_index = row_kv_index[row_iteration]

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
    def _store_dq_epi_tma_v12(
        self,
        t_dq: cute.Tensor,
        dq_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        s_dq_epi: cute.Tensor,
        tma_atom_dq_epi: cute.CopyAtom,
        tma_tensor_dq_epi: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        token_idx: Int32,
        batch_idx: Int32,
        rank: Int32,
        mtx: Int32,
    ):
        """Store one rank-owned dQ round via SMEM staging + one bulk TMA.

        v12 (P4, ported from the V0 epilogue / the b244255 precedent):
        T2R exactly as the scalar path, but the values land in the dead
        score-K allocation and leave as a single 32 KiB
        CopyBulkTensorTileS2G instead of 16,384 scattered 2-byte STGs per
        CTA per round.  Source-side completion (wait_group.read) plus the
        math barrier authorize round 1 to overwrite the staging.
        """

        if mtx < self.MATH_THREADS_PER_CTA:
            tiled_t2r = tcgen05.make_tmem_copy(dq_tmem_load, t_dq)
            thread_t2r = tiled_t2r.get_slice(mtx)
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
                local_d = d_in_round - rank * Int32(
                    self.D_TILE_CTA
                )
                s_dq_epi[
                    head,
                    local_d,
                ] = self.element_dtype(thread_values[value_index])
        cute.arch.fence_view_async_shared()
        self.math_barrier.arrive_and_wait()

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
        if mtx < Int32(32):
            cute.arch.fence_view_async_shared()
            cute.copy(tma_atom_dq_epi, t_smem, t_gmem)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        # Source-side completion broadcast: round 1 may overwrite the
        # staging only after the engine has READ round 0's bytes.
        self.math_barrier.arrive_and_wait()

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
def _atomic_add_fp32x4_v1(
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


@dsl_user_op
def _mbarrier_wait_acquire_cluster(
    barrier: cute.Pointer,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Block on one local mbarrier phase with a cluster-scope acquire."""

    barrier_i32 = barrier.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [barrier_i32, phase.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "CLUSTER_WAIT_LOOP:\n\t"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
            "p, [$0], $1, 10000000;\n\t"
            "@!p bra CLUSTER_WAIT_LOOP;\n\t"
            "}"
        ),
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cvt_bf16x2_f32(
    lo: Float32,
    hi: Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Pack two FP32 values as one BF16x2 register."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_u32(
    value: cutlass.Uint32,
    destination: cute.Pointer,
    completion_barrier: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Push one packed BF16x2 register to peer DSM and complete four bytes.

    This is the exact CP0-dependent primitive.  The CP0 line must confirm
    the mnemonic, constraints and transaction accounting before A0 is run.
    Keeping the helper scalar makes every logical destination pointer
    explicit; a later code-clean pass may vectorize four adjacent registers
    without changing the route contract.
    """

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
            barrier_i32,
            cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        (
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
            "[$0], $2, [$1];"
        ),
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_shared_remote_u32x4(
    destination: cute.Pointer,
    completion_barrier: cute.Pointer,
    peer_rank: Int32,
    word0: cutlass.Uint32,
    word1: cutlass.Uint32,
    word2: cutlass.Uint32,
    word3: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store eight packed BF16 values and complete 16 destination bytes.

    This is mechanically aligned with
    ``agents/workspaces/v1_cp0/packed_bf16_cluster_store_probe.py``.
    """

    destination_i32 = destination.toint(loc=loc, ip=ip).ir_value()
    barrier_i32 = completion_barrier.toint(
        loc=loc,
        ip=ip,
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i32,
            barrier_i32,
            peer_rank.ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word3).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .u32 remote_destination;\n\t"
            ".reg .u32 remote_mbar;\n\t"
            ".reg .v4 .u32 packed;\n\t"
            "mapa.shared::cluster.u32 remote_destination, $0, $2;\n\t"
            "mapa.shared::cluster.u32 remote_mbar, $1, $2;\n\t"
            "mov.u32 packed.x, $3;\n\t"
            "mov.u32 packed.y, $4;\n\t"
            "mov.u32 packed.z, $5;\n\t"
            "mov.u32 packed.w, $6;\n\t"
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.u32 "
            "[remote_destination], packed, [remote_mbar];\n\t"
            "}"
        ),
        "r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _load_shared_u32_at(
    tensor: cute.Tensor,
    coordinate,
) -> cutlass.Uint32:
    pointer = cute.recast_ptr(
        tensor.iterator + tensor.layout(coordinate),
        dtype=cutlass.Uint32,
    )
    packed = cute.make_tensor(pointer, cute.make_layout((1,)))
    return packed[0]


@cute.jit
def _store_shared_u32_at(
    tensor: cute.Tensor,
    coordinate,
    value: cutlass.Uint32,
) -> None:
    pointer = cute.recast_ptr(
        tensor.iterator + tensor.layout(coordinate),
        dtype=cutlass.Uint32,
    )
    packed = cute.make_tensor(pointer, cute.make_layout((1,)))
    packed[0] = value


@cute.jit
def _store_remote_u32_at(
    tensor: cute.Tensor,
    coordinate,
    value: cutlass.Uint32,
    completion_barrier: cute.Pointer,
    peer_rank: Int32,
) -> None:
    _store_shared_remote_u32(
        value,
        tensor.iterator + tensor.layout(coordinate),
        completion_barrier,
        peer_rank,
    )


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
            self._detach_issued_context(
                issue_seq,
                issued_ctx,
                reducer_ctx,
                issued_ctx_mbars,
                reducer_ctx_mbars,
                ctx_reader_done_mbars,
                peer,
            )
        math_barrier.arrive_and_wait()
        if is_math_leader:
            cute.arch.mbarrier_arrive(p_local_ready + stage)
            cute.arch.mbarrier_arrive(ds_local_ready + stage)

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


@dsl_user_op
def _store_shared_seq_v4(
    counter: cute.Pointer,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Release-store a monotonic tile sequence number to CTA shared."""

    counter_i32 = counter.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [counter_i32, Int32(value).ir_value(loc=loc, ip=ip)],
        "st.release.cta.shared.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _wait_shared_seq_v4(
    counter: cute.Pointer,
    target: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Spin until the shared sequence number reaches ``target``.

    A monotonic counter has no phase parity, so the producer may run any
    number of generations ahead of the waiter (a count-1 mbarrier here
    deadlocks once the producer leads by two: the parity aliases back).
    """

    counter_i32 = counter.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [counter_i32, Int32(target).ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            ".reg .u32 v;\n\t"
            "SEQ_WAIT_LOOP:\n\t"
            "ld.acquire.cta.shared.u32 v, [$0];\n\t"
            "setp.ge.u32 p, v, $1;\n\t"
            "@!p bra SEQ_WAIT_LOOP;\n\t"
            "}"
        ),
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _store_shared_bf16_at_v2(
    tensor: cute.Tensor,
    coordinate,
    value: BFloat16,
) -> None:
    """Scalar BF16 store through a (possibly swizzled) SMEM tensor layout."""

    pointer = cute.recast_ptr(
        tensor.iterator + tensor.layout(coordinate),
        dtype=BFloat16,
    )
    packed = cute.make_tensor(pointer, cute.make_layout((1,)))
    packed[0] = value


class FlashAttentionDSABackwardSm100TwoCTAV2(
    FlashAttentionDSABackwardSm100TwoCTA
):
    """Two-CTA production kernel: 20 warps per CTA, five CG2 GEMMs per KV tile.

    Per-CTA roles: warps 0-3 gather the sparse K tile and the dQ-A
    images; warps 4-7 run the softmax backward and publish P/dS; warps
    8-15 drain the fused dV+dK partial sums to the f32 workspace; warp
    16 issues every GEMM and manages pipeline credits; warp 17 feeds
    the round ring from the stationary panels; warp 18 relays P/dS
    across the cluster; warp 19 relays ring TMA completions.

    Steady-state schedule per KV tile (rotated): S(t) and dP(t) issue
    first, then tile t-1's gradients (two dQ rounds, eight dV/dK
    passes).  The kdq rendezvous hands the gather-filled dQ-A images to
    the ring through a count-128 hardware arrival; the score K tile is
    released after dP and its space time-shares with the round-0 dO
    quadrants.  dQ stays TMEM-resident across all tiles and stores
    through a two-round staged TMA epilogue.
    """

    THREADS_PER_CTA = 640

    # Warp roles (5 warps per named group where applicable).
    GATHER_WARPS = 4
    MATH_WARP_BEGIN = 4
    MATH_WARPS = 4
    REDUCE_WARP_BEGIN = 8
    REDUCE_WARPS = 8
    MMA_WARP = 16
    LOAD_WARP = 17
    RELAY_WARP = 18
    COMMIT_WARP = 19
    IDLE_WARP = 19

    GATHER_THREADS = GATHER_WARPS * 32
    MATH_THREAD_BEGIN = MATH_WARP_BEGIN * 32
    MATH_THREADS = MATH_WARPS * 32
    REDUCE_THREAD_BEGIN = REDUCE_WARP_BEGIN * 32
    REDUCE_THREADS = REDUCE_WARPS * 32

    # Two-pass H reduction: the dkv MMA reduces one H64 half per issue.
    # This single override retargets every base-provided dkv layout and the
    # QT/dOT TMA atoms to [D128, H64] quadrant slabs.
    DKV_MMA_TILER = (256, 64, 64)
    H_PASSES = 2
    PDS_BLOCK_ELEMENTS = 2_048
    PDS_BLOCK_BYTES = 4_096

    TMEM_S_OFFSET = 0
    TMEM_S1_OFFSET = 32
    TMEM_DP_OFFSET = 64
    TMEM_DP1_OFFSET = 96
    TMEM_DQ0_OFFSET = 128
    TMEM_DQ1_OFFSET = 256
    TMEM_DKV0_OFFSET = 384
    TMEM_DKV1_OFFSET = 448

    SCORE_DONE_STAGES = 2

    ROUND_GENS_PER_TILE = 8
    ROUND_STAGES = 2

    MMA_DONE_STAGES = 2

    SOFTMAX_GROUPED_STATS = True

    OWN_HALF_BULK = True

    IKET_V2_NATIVE_PROVENANCE = "V2_NATIVE_PROVENANCE"

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        block_tile: int,
        max_topk: int = 0,
    ):
        super().__init__(head_dim, head_dim_v, block_tile, max_topk)
        self.math_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.MATH_THREADS,
        )
        self.cta_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.THREADS_PER_CTA,
        )
        self.gather_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.GATHER_THREADS,
        )
        self.kdq_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=(self.GATHER_WARPS + 1) * 32,
        )

    def _make_score_tmem_load(self, score_cta_shape, score_epi_tile):
        """v9.3: force the 16-DP/256-bit T2R atom for S/dP.

        get_smem_store_op keys the publish store atom off the T2R atom's
        thread-value ownership; the default (non-16-DP) choice made it
        fall back to CopyUniversalOp -- the v8 SASS showed 96 scalar
        STS.U16 + 96 PRMT per warp per tile (~4.6 us real) and ZERO
        stmatrix.  Ld16x256b(Rep 4) has (num_dp=16, num_bits=256, rep=4,
        pack=NONE), which fires use_stmatrix_m8n8_4x's f32->bf16 clause:
        the publish lowers to stmatrix.m8n8.x4.trans (4 STSM per tensor
        per warp-tile).  Ownership shifts to 4 h-rows x 8 n-cols per
        thread (host-probe verified against the CuTe traits); the
        n-half stays warp-uniform, so the P/dS local/xchg machinery
        survives; only the softmax stats indexing changes (grouped, see
        the math role).
        """

        return cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(
                tcgen05.copy.Repetition(4)
            ),
            self.acc_dtype,
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
        element_dtype = self.element_dtype

        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        # Re-tiled DKV (K = H64): quadrant slab and 4 KiB B blocks.
        assert cute.cosize(dkv_a_layout_staged) <= 8192
        assert (
            cute.cosize(score_b_layout_staged)
            == 2 * cute.cosize(dkv_a_layout_staged)
        )
        assert cute.cosize(dkv_a_layout_staged) == 8192
        assert cute.cosize(dkv_b_layout_staged) <= 2048
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 4096

        @cute.struct
        class SharedStorageV2:
            # Pipeline barrier arrays (full+empty per stage).
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            kscore_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            # E1: per-chunk score-K readiness (leader-CTA barriers).
            kchunk_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            # E3: per-chunk score-K retirement (dP's tcgen05 release
            # train, multicast to both CTAs; count 1).
            krel_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            pds_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dkv_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dq_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            # Raw single-phase-per-tile barriers.
            stationary_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            landing_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            relay_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            loan_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            loan_epi_safe_mbar: cutlass.Int64
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            khot_seq: cute.struct.MemRange[cutlass.Int64, 1]
            kdq_ready_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

            stationary_q: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 32768],
                1024,
            ]
            stationary_do: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 32768],
                1024,
            ]
            score_kv: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 16384],
                1024,
            ]
            round_buf_a: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 8192],
                1024,
            ]
            round_buf_b: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 8192],
                1024,
            ]
            p_blocks: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 4096],
                1024,
            ]
            p_xchg: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 2048],
                1024,
            ]
            ds_image: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 4096],
                1024,
            ]
            ds_blocks: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 4096],
                1024,
            ]
            ds_xchg: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 2048],
                1024,
            ]
            stats: cute.struct.Align[
                cute.struct.MemRange[Float32, 128],
                1024,
            ]

        assert SharedStorageV2.size_in_bytes() <= self.MAX_SMEM_BYTES
        return SharedStorageV2

    @cute.jit
    def _load_score_kv_streamed(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        destination: cute.Tensor,
        kchunk_mbars: cute.Pointer,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Chunk-streamed score-K gather (E1).

        Same per-row copy protocol as _load_score_kv, restructured
        chunk-major: one cp.async commit group per D128 chunk, then a
        wait-group ladder -- wait_group(K_CHUNKS-1-c) retires chunk c's
        group while the younger chunks' copies stay in flight, so the
        crew never drains its own pipeline.  Chunk c's landing (both
        halves: every gather thread of both CTAs arrives once) is
        published on the leader CTA's kchunk_mbars[c] for the S GEMM.
        """

        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        row_local_n = [
            row_iteration * self.KV_NUM_GROUPS + group_index
            for row_iteration in range(rows_per_group)
        ]
        row_kv_index = []
        for local_n in row_local_n:
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = mTopkIdxs[
                    topk_slot,
                    (token_idx, batch_idx),
                ]
            row_kv_index.append(kv_index)

        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            destination_rows = cute.composition(
                destination[None, None, None, chunk],
                cute.make_layout(
                    (self.N_TILE_CTA, self.K_CHUNK)
                ),
            )
            for row_iteration in cutlass.range_constexpr(
                rows_per_group
            ):
                local_n = row_local_n[row_iteration]
                kv_index = row_kv_index[row_iteration]
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
            cute.arch.cp_async_commit_group()

        # E4a: two-session publish.  Session A = chunks 0-1 (early S
        # food), session B = chunks 2-3 (the future lending window's
        # tail).  Rung arithmetic: 4 commit groups issued above; group
        # FIFO retirement makes wait_group(2) == chunks 0-1 complete.
        cute.arch.cp_async_wait_group(2)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            kchunk_mbars,
            Int32(0),
        )
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            kchunk_mbars + 1,
            Int32(0),
        )

    @cute.jit
    def _load_score_kv_streamed_gated(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        destination: cute.Tensor,
        kchunk_mbars: cute.Pointer,
        borrow_free_mbar: cute.Pointer,
        borrow_phase: Int32,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Steady-state K fill with the E4b lending gate.

        Session A (chunks 0-1) fills at the lease exactly as before.
        Session B (chunks 2-3) waits for the borrowed dO_r1 h0 tenant
        to be consumed (the leader's borrow_free commit fires after
        dV_r1 pass a) before overwriting the parcel it lives in.
        """

        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        row_local_n = [
            row_iteration * self.KV_NUM_GROUPS + group_index
            for row_iteration in range(rows_per_group)
        ]
        row_kv_index = []
        for local_n in row_local_n:
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = mTopkIdxs[
                    topk_slot,
                    (token_idx, batch_idx),
                ]
            row_kv_index.append(kv_index)

        # Session A: chunks 0-1.
        for chunk in cutlass.range_constexpr(2):
            destination_rows = cute.composition(
                destination[None, None, None, chunk],
                cute.make_layout(
                    (self.N_TILE_CTA, self.K_CHUNK)
                ),
            )
            for row_iteration in cutlass.range_constexpr(
                rows_per_group
            ):
                local_n = row_local_n[row_iteration]
                kv_index = row_kv_index[row_iteration]
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
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            kchunk_mbars,
            Int32(0),
        )

        # E4b lending gate: the borrowed tenant must be consumed
        # before session B's copies may launch.
        cute.arch.mbarrier_wait(
            borrow_free_mbar,
            borrow_phase,
        )

        # Session B: chunks 2-3.
        for chunk in cutlass.range_constexpr(2):
            destination_rows = cute.composition(
                destination[None, None, None, chunk + 2],
                cute.make_layout(
                    (self.N_TILE_CTA, self.K_CHUNK)
                ),
            )
            for row_iteration in cutlass.range_constexpr(
                rows_per_group
            ):
                local_n = row_local_n[row_iteration]
                kv_index = row_kv_index[row_iteration]
                if kv_index >= 0:
                    self._copy_sparse_k_d128_row(
                        mKV,
                        destination_rows,
                        local_n,
                        kv_index,
                        batch_idx,
                        Int32((chunk + 2) * self.K_CHUNK),
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
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            kchunk_mbars + 1,
            Int32(0),
        )

    @cute.jit
    def _load_score_kv_streamed_early(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        destination: cute.Tensor,
        kchunk_mbars: cute.Pointer,
        krel_mbars: cute.Pointer,
        kscore_pipeline,
        kscore_producer_state: pipeline.PipelineState,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Prologue-K(1)-only early fill (E3).

        dP(0)'s per-chunk tcgen05 release train retires K(0)'s chunks
        in order; chunks 0..2 of K(1) start filling on those signals
        (phase 0 -- the train's first ever firing), while chunk 3
        keeps the whole-tile pipe lease so the kscore state machine is
        untouched.  ONLY legal where K directly succeeds K: in the
        steady loop the loan lives in score_kv between two K
        generations, so the release train must never gate fills there.
        """

        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        row_local_n = [
            row_iteration * self.KV_NUM_GROUPS + group_index
            for row_iteration in range(rows_per_group)
        ]
        row_kv_index = []
        for local_n in row_local_n:
            logical_n = rank * self.N_TILE_CTA + local_n
            topk_slot = tile_index * self.N_TILE + logical_n
            kv_index = Int32(-1)
            if topk_slot < topk:
                kv_index = mTopkIdxs[
                    topk_slot,
                    (token_idx, batch_idx),
                ]
            row_kv_index.append(kv_index)

        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            if cutlass.const_expr(chunk < self.K_CHUNKS - 1):
                cute.arch.mbarrier_wait(
                    krel_mbars + chunk,
                    Int32(0),
                )
            else:
                kscore_pipeline.producer_acquire(
                    kscore_producer_state
                )
            destination_rows = cute.composition(
                destination[None, None, None, chunk],
                cute.make_layout(
                    (self.N_TILE_CTA, self.K_CHUNK)
                ),
            )
            for row_iteration in cutlass.range_constexpr(
                rows_per_group
            ):
                local_n = row_local_n[row_iteration]
                kv_index = row_kv_index[row_iteration]
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
            cute.arch.cp_async_commit_group()

        # E4a: same two-session publish as the steady fill.
        cute.arch.cp_async_wait_group(2)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            kchunk_mbars,
            Int32(0),
        )
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(
            kchunk_mbars + 1,
            Int32(0),
        )

    @cute.jit
    def _kd_round_rows_v2(
        self,
        tensor: cute.Tensor,
    ) -> cute.Tensor:
        """[N64 rows, D128 contiguous] view of one dQ-A round buffer.

        Row n is the 128-element contiguous D-quarter slice the sparse-row
        copy helpers expect (destination_rows[n, None]); flat index
        d + 128*n maps to logical (m=d, k=n) of the MN-major dq_a layout.
        Mirrors the verified base _load_grad_k composition convention.
        """

        return cute.composition(
            tensor[None, None, None, 0],
            cute.make_layout(
                (self.N_TILE, self.D_TILE_CTA),
                stride=(self.D_TILE_CTA, 1),
            ),
        )

    @cute.jit
    def _split_wg_t1d_v2(
        self,
        tensor: cute.Tensor,
        wg_idx: Int32,
        num_wg: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        """Split the first nontrivial value mode across warp groups."""

        reduced_shape = cute.product_each(tensor.shape)
        rank = len(reduced_shape)
        if cutlass.const_expr(reduced_shape[1] > 1):
            assert rank >= 2
            tensor = cute.logical_divide(
                tensor,
                (
                    reduced_shape[0],
                    reduced_shape[1] // num_wg,
                ),
            )
            coordinate = (
                None,
                (None, wg_idx),
            ) + (None,) * (rank - 2)
        else:
            assert rank >= 3
            if cutlass.const_expr(rank == 3):
                tensor = cute.logical_divide(
                    tensor,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2] // num_wg,
                    ),
                )
                coordinate = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                tensor = cute.logical_divide(
                    tensor,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coordinate = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return tensor[coordinate]

    @cute.jit
    def _fill_kdq_pair_v2(
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
        lane_idx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ) -> None:
        """Gather BOTH [N64, Dq(r,c)] dQ-A rounds in one fused pass.

        One warp (32 threads = 4 groups of 8) covers all 64 KV rows with a
        single index read per row; each row contributes one 256-byte
        D-quarter slice per round at column offset 256*r + 128*rank.
        Invalid rows are zero-filled so the dQ GEMM sees exact zeros.
        """

        index_in_group = lane_idx % self.KV_GROUP_SIZE
        group_index = lane_idx // self.KV_GROUP_SIZE
        groups_per_warp = 32 // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = (
            Int32(self.D_TILE_CLUSTER)
            + rank * Int32(self.D_TILE_CTA)
        )
        rows_per_group = self.N_TILE // groups_per_warp
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_iteration * groups_per_warp + group_index
            global_n = tile_index * Int32(self.N_TILE) + Int32(local_n)
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = mTopkIdxs[global_n, (token_idx, batch_idx)]
            if kv_index >= Int32(0):
                self._copy_sparse_k_d128_row(
                    mKV,
                    kd_rows_0,
                    Int32(local_n),
                    kv_index,
                    batch_idx,
                    d_offset_0,
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
                self._copy_sparse_k_d128_row(
                    mKV,
                    kd_rows_1,
                    Int32(local_n),
                    kv_index,
                    batch_idx,
                    d_offset_1,
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
            else:
                self._zero_sparse_k_d128_row(
                    kd_rows_0,
                    Int32(local_n),
                    index_in_group,
                )
                self._zero_sparse_k_d128_row(
                    kd_rows_1,
                    Int32(local_n),
                    index_in_group,
                )

    @cute.jit
    def _fill_kdq_pair_v8(
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
        thread_count: cutlass.Constexpr[int],
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ) -> None:
        """Gather BOTH [N64, Dq(r,c)] dQ-A rounds across the gather warps.

        v8 offload: identical per-row protocol to _fill_kdq_pair_v2 (one
        index read per row, one 256-byte D-quarter slice per round at
        column offset 256*r + 128*rank, zero-fill for invalid rows), but
        partitioned over `thread_count` threads: 128 gather threads form
        16 groups of KV_GROUP_SIZE=8, so each group covers 4 rows instead
        of the load warp's 16 -- a 4x wider fill.
        """

        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = (
            Int32(self.D_TILE_CLUSTER)
            + rank * Int32(self.D_TILE_CTA)
        )
        assert self.N_TILE % groups_total == 0
        rows_per_group = self.N_TILE // groups_total
        kdq_local_n = [
            row_iteration * groups_total + group_index
            for row_iteration in range(rows_per_group)
        ]
        kdq_kv_index = []
        for local_n in kdq_local_n:
            global_n = tile_index * Int32(self.N_TILE) + Int32(
                local_n
            )
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = mTopkIdxs[global_n, (token_idx, batch_idx)]
            kdq_kv_index.append(kv_index)

        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = kdq_local_n[row_iteration]
            kv_index = kdq_kv_index[row_iteration]
            if kv_index >= Int32(0):
                self._copy_sparse_k_d128_row(
                    mKV,
                    kd_rows_0,
                    Int32(local_n),
                    kv_index,
                    batch_idx,
                    d_offset_0,
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
                self._copy_sparse_k_d128_row(
                    mKV,
                    kd_rows_1,
                    Int32(local_n),
                    kv_index,
                    batch_idx,
                    d_offset_1,
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
            else:
                self._zero_sparse_k_d128_row(
                    kd_rows_0,
                    Int32(local_n),
                    index_in_group,
                )
                self._zero_sparse_k_d128_row(
                    kd_rows_1,
                    Int32(local_n),
                    index_in_group,
                )

    @cute.jit
    def _fill_kdq_pair_vk7(
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
        """_fill_kdq_pair_v8's copy half, fed pre-A indices (W2).

        Identical per-row protocol; the four per-thread topk indices
        arrive as arguments, loaded before barrier A so their GMEM
        latency hides under the rendezvous wait instead of serializing
        ahead of the first cp.async.
        """

        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = (
            Int32(self.D_TILE_CLUSTER)
            + rank * Int32(self.D_TILE_CTA)
        )
        assert self.N_TILE % groups_total == 0
        rows_per_group = self.N_TILE // groups_total
        assert rows_per_group == 4
        kdq_local_n = [
            row_iteration * groups_total + group_index
            for row_iteration in range(rows_per_group)
        ]
        kdq_kv_index = [
            kv_index_0,
            kv_index_1,
            kv_index_2,
            kv_index_3,
        ]

        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = kdq_local_n[row_iteration]
            kv_index = kdq_kv_index[row_iteration]
            if kv_index >= Int32(0):
                self._copy_sparse_k_d128_row(
                    mKV,
                    kd_rows_0,
                    Int32(local_n),
                    kv_index,
                    batch_idx,
                    d_offset_0,
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
                self._copy_sparse_k_d128_row(
                    mKV,
                    kd_rows_1,
                    Int32(local_n),
                    kv_index,
                    batch_idx,
                    d_offset_1,
                    index_in_group,
                    copy_atom,
                    thread_copy,
                )
            else:
                self._zero_sparse_k_d128_row(
                    kd_rows_0,
                    Int32(local_n),
                    index_in_group,
                )
                self._zero_sparse_k_d128_row(
                    kd_rows_1,
                    Int32(local_n),
                    index_in_group,
                )

    @cute.jit
    def _gather_kdq_v8(
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
        kdq_ready_mbar: cute.Pointer,
    ) -> None:
        """Gather-side half of the v8 kdq handshake.

        Barrier A: the load warp holds both g0/g1 round-stage credits, so
        the round buffers are safe to overwrite.  The gather warps then
        write both K_dQ images, drain their own cp.async groups, fence,
        and barrier B hands the generations back for the load warp's two
        commits.
        """

        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = self.GATHER_THREADS // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE // groups_total
        assert rows_per_group == 4
        kdq_local_n = [
            row_iteration * groups_total + group_index
            for row_iteration in range(rows_per_group)
        ]
        kdq_kv_index = []
        for local_n in kdq_local_n:
            global_n = tile_index * Int32(self.N_TILE) + Int32(
                local_n
            )
            kv_index = Int32(-1)
            if global_n < topk:
                kv_index = mTopkIdxs[global_n, (token_idx, batch_idx)]
            kdq_kv_index.append(kv_index)

        self.kdq_barrier.arrive_and_wait()
        self._fill_kdq_pair_vk7(
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
        _cp_async_mbarrier_arrive(kdq_ready_mbar)
        cute.arch.mbarrier_arrive(kdq_ready_mbar, rank)

    @cute.jit
    def _fill_score_loan_do_r0_vc2(
        self,
        rank: Int32,
        warp_idx: Int32,
        stationary_do_raw: cute.Pointer,
        score_kv_raw: cute.Pointer,
        loan_tma_mbars: cute.Pointer,
        loan_phase: Int32,
        tma_atom_dot: cute.CopyAtom,
        t_dot_gmem: cute.Tensor,
        t_dot_loan_smem_a: cute.Tensor,
        t_dot_loan_smem_b: cute.Tensor,
        grad_a_stage_bytes: cutlass.Constexpr[int],
    ) -> Int32:
        """Fill round-0 dO into the two halves of ``score_kv``.

        All four gather warps call this helper.  Warp 0 launches one local
        bulk copy and one peer-panel TMA per CTA; the named gather barriers
        publish launch and completion to the remaining gather threads before
        their collective kscore producer commit.
        """

        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    loan_tma_mbars,
                    grad_a_stage_bytes,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    loan_tma_mbars + 1,
                    grad_a_stage_bytes,
                )
            if rank == Int32(0):
                with cute.arch.elect_one():
                    _cpasync_bulk_s2cluster(
                        stationary_do_raw,
                        score_kv_raw,
                        loan_tma_mbars,
                        grad_a_stage_bytes,
                        rank,
                    )
                cute.copy(
                    tma_atom_dot,
                    t_dot_gmem[None, 0, 1],
                    t_dot_loan_smem_b[None, 0],
                    tma_bar_ptr=loan_tma_mbars + 1,
                )
            else:
                cute.copy(
                    tma_atom_dot,
                    t_dot_gmem[None, 0, 0],
                    t_dot_loan_smem_a[None, 0],
                    tma_bar_ptr=loan_tma_mbars,
                )
                with cute.arch.elect_one():
                    _cpasync_bulk_s2cluster(
                        stationary_do_raw + Int32(8192),
                        score_kv_raw + Int32(8192),
                        loan_tma_mbars + 1,
                        grad_a_stage_bytes,
                        rank,
                    )

        # Launch visibility, then completion visibility, across all four
        # gather warps.  Only warp 0 polls the raw completion barriers.
        self.gather_barrier.arrive_and_wait()
        if warp_idx == Int32(0):
            cute.arch.mbarrier_wait(
                loan_tma_mbars,
                loan_phase,
            )
            cute.arch.mbarrier_wait(
                loan_tma_mbars + 1,
                loan_phase,
            )
        self.gather_barrier.arrive_and_wait()
        cute.arch.fence_view_async_shared()
        return Int32(1) - loan_phase

    @cute.jit
    def _issue_dkv_pass_v2(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        accumulate: cutlass.Constexpr[bool],
    ) -> None:
        """Issue one K=H64 dV/dK pass into the round's dKV accumulator."""

        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        for k_block in cutlass.range_constexpr(
            cute.size(a_fragment, mode=[2])
        ):
            cute.gemm(
                mma,
                t_dkv,
                a_fragment[None, None, k_block, 0],
                b_fragment[None, None, k_block, 0],
                t_dkv,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_dq_rounds_v2(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        kd_fragment_a: cute.Tensor,
        kd_fragment_b: cute.Tensor,
        ds_fragment: cute.Tensor,
        accumulate: cutlass.Boolean,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        issue_seq: Int32,
    ) -> pipeline.PipelineState:
        """Issue both persistent dQ rounds back-to-back (v1_deep_p lesson).

        Round r consumes the round-region generation holding K_dQ(r); each
        generation is released immediately after its issue so the load warp
        can begin the dO quadrant refills.
        """

        for round_index in cutlass.range_constexpr(self.D_ROUNDS):
            packed_issue = (
                issue_seq * Int32(self.D_ROUNDS)
                + Int32(round_index)
            )
            round_pipeline.consumer_wait(round_consumer_state)
            mma = dq_tiled_mma.with_()
            mma.set(tcgen05.Field.ACCUMULATE, accumulate)
            if cutlass.const_expr(round_index == 0):
                for k_block in cutlass.range_constexpr(
                    cute.size(kd_fragment_a, mode=[2])
                ):
                    cute.gemm(
                        mma,
                        t_dq_0,
                        kd_fragment_a[None, None, k_block, 0],
                        ds_fragment[None, None, k_block, 0],
                        t_dq_0,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for k_block in cutlass.range_constexpr(
                    cute.size(kd_fragment_b, mode=[2])
                ):
                    cute.gemm(
                        mma,
                        t_dq_1,
                        kd_fragment_b[None, None, k_block, 0],
                        ds_fragment[None, None, k_block, 0],
                        t_dq_1,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.fence_view_async_tmem_store()
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
        return round_consumer_state

    @cute.jit
    def _zero_dq_v2(
        self,
        rank_coordinates: cute.Tensor,
        mdQ: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        token_idx: Int32,
        batch_idx: Int32,
        tidx: Int32,
    ) -> None:
        """Write the required all-zero dQ result when no tile is issued."""

        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            linear_index = tidx
            while linear_index < cute.size(rank_coordinates):
                coordinate = cute.idx2crd(
                    linear_index,
                    rank_coordinates.shape,
                )
                logical_coordinate = rank_coordinates[coordinate]
                d_in_round = Int32(
                    cute.get(logical_coordinate, mode=[0])
                )
                head = Int32(
                    cute.get(logical_coordinate, mode=[1])
                )
                mdQ[
                    Int32(round_index * self.D_TILE_CLUSTER)
                    + d_in_round,
                    head,
                    (token_idx, batch_idx),
                ] = self.element_dtype(0.0)
                linear_index += Int32(
                    self.MATH_THREADS_PER_CTA
                )

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
        stationary_tiled_mma: cute.TiledMma,
        stationary_a_layout_staged: cute.ComposedLayout,
    ):
        """v2 rotated-schedule two-CTA backward (design: 优化设计文档_v2.md)."""

        _ = problem_shape
        _ = mQ
        _ = mdO
        _ = trace_buffer
        _ = trace_token_idx
        _ = trace_batch_idx

        physical_x, _, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(
            cute.arch.warp_idx()
        )
        rank = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(rank)
        peer_rank = Int32(1) - rank
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        is_leader_cta = rank == Int32(0)

        if warp_idx == Int32(self.LOAD_WARP):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_qt)
            cpasync.prefetch_descriptor(tma_atom_dot)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_holding_buf_ptr = storage.tmem_holding_buf.ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr
        stationary_tma_mbars = storage.stationary_tma_mbars.data_ptr()
        stationary_ready_mbar = storage.stationary_ready_mbar.data_ptr()
        landing_mbars = storage.landing_mbars.data_ptr()
        relay_mbars = storage.relay_mbars.data_ptr()
        pds_ready_mbars = storage.pds_ready_mbars.data_ptr()
        round_tma_mbars = storage.round_tma_mbars.data_ptr()
        loan_tma_mbars = storage.loan_tma_mbars.data_ptr()
        loan_epi_safe_mbar = storage.loan_epi_safe_mbar.ptr
        kdq_ready_mbar = storage.kdq_ready_mbar.data_ptr()
        kchunk_mbars = storage.kchunk_mbars.data_ptr()
        krel_mbars = storage.krel_mbars.data_ptr()
        khot_seq = cute.recast_ptr(
            storage.khot_seq.data_ptr(),
            dtype=cutlass.Int32,
        )
        # Raw pointers used inside role branches must be extracted here:
        # the struct instance itself cannot cross a dynamic-if region.
        stationary_q_raw = storage.stationary_q.data_ptr()
        stationary_do_raw = storage.stationary_do.data_ptr()
        round_buf_a_raw = storage.round_buf_a.data_ptr()
        round_buf_b_raw = storage.round_buf_b.data_ptr()
        score_kv_raw = storage.score_kv.data_ptr()

        # ------------------------------------------------------------------
        # SMEM tensor views.
        # ------------------------------------------------------------------
        stationary_q = storage.stationary_q.get_tensor(
            score_a_layout_staged.outer,
            swizzle=score_a_layout_staged.inner,
        )
        stationary_do = storage.stationary_do.get_tensor(
            score_a_layout_staged.outer,
            swizzle=score_a_layout_staged.inner,
        )
        stationary_q_tma = storage.stationary_q.get_tensor(
            stationary_a_layout_staged.outer,
            swizzle=stationary_a_layout_staged.inner,
        )
        stationary_do_tma = storage.stationary_do.get_tensor(
            stationary_a_layout_staged.outer,
            swizzle=stationary_a_layout_staged.inner,
        )
        k_n = storage.score_kv.get_tensor(
            score_b_layout_staged.outer,
            swizzle=score_b_layout_staged.inner,
        )
        loan_quad = (
            cute.make_tensor(
                cute.recast_ptr(
                    score_kv_raw,
                    dkv_a_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                dkv_a_layout_staged.outer,
            ),
            cute.make_tensor(
                cute.recast_ptr(
                    score_kv_raw + Int32(8192),
                    dkv_a_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                dkv_a_layout_staged.outer,
            ),
        )
        s_dq_epi = cute.make_tensor(
            cute.recast_ptr(
                storage.score_kv.data_ptr(),
                dq_epi_layout_staged.inner,
                self.element_dtype,
            ),
            dq_epi_layout_staged.outer,
        )[None, None, 0]
        round_kd = (
            storage.round_buf_a.get_tensor(
                dq_a_layout_staged.outer,
                swizzle=dq_a_layout_staged.inner,
            ),
            storage.round_buf_b.get_tensor(
                dq_a_layout_staged.outer,
                swizzle=dq_a_layout_staged.inner,
            ),
        )
        round_quad = (
            storage.round_buf_a.get_tensor(
                dkv_a_layout_staged.outer,
                swizzle=dkv_a_layout_staged.inner,
            ),
            storage.round_buf_b.get_tensor(
                dkv_a_layout_staged.outer,
                swizzle=dkv_a_layout_staged.inner,
            ),
        )
        p_blocks_raw = storage.p_blocks.data_ptr()
        ds_blocks_raw = storage.ds_blocks.data_ptr()
        ds_image_raw = storage.ds_image.data_ptr()
        p_blocks = (
            cute.make_tensor(
                cute.recast_ptr(
                    p_blocks_raw,
                    dkv_b_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                dkv_b_layout_staged.outer,
            ),
            cute.make_tensor(
                cute.recast_ptr(
                    p_blocks_raw + self.PDS_BLOCK_ELEMENTS,
                    dkv_b_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                dkv_b_layout_staged.outer,
            ),
        )
        ds_blocks = (
            cute.make_tensor(
                cute.recast_ptr(
                    ds_blocks_raw,
                    dkv_b_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                dkv_b_layout_staged.outer,
            ),
            cute.make_tensor(
                cute.recast_ptr(
                    ds_blocks_raw + self.PDS_BLOCK_ELEMENTS,
                    dkv_b_layout_staged.inner,
                    dtype=self.element_dtype,
                ),
                dkv_b_layout_staged.outer,
            ),
        )
        ds_image = storage.ds_image.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )
        # Whole-image dS store view (the production-verified byte identity
        # between the COL_MAJOR epi store image and the dq-B operand).
        score_store_layout = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.H_TILE_CTA, self.N_TILE),
            1,
        )
        assert (
            cute.cosize(score_store_layout)
            == cute.cosize(dq_b_layout_staged)
        )
        assert (
            score_store_layout.inner
            == dq_b_layout_staged.inner
        )
        assert (
            score_store_layout.inner
            == dkv_b_layout_staged.inner
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
            == cute.cosize(dq_b_layout_staged)
        )
        ds_image_store = storage.ds_image.get_tensor(
            score_store_domain,
            swizzle=score_store_layout.inner,
        )
        # Preserve the exact nested/swizzled K64 partition-B byte image.
        # A raw 4 KiB DSM copy is then layout-preserving because every block
        # has the same type and alignment at source and destination.
        p_block_stage = p_blocks[0][None, None, None, 0]
        assert (
            cute.size(p_block_stage, mode=[0, 0])
            == self.N_TILE_CTA
        )
        assert cute.size(p_block_stage, mode=[0, 1]) == 16
        assert cute.size(p_block_stage, mode=[1]) == 1
        assert cute.size(p_block_stage, mode=[2]) == 4
        assert cute.size(p_block_stage) == self.PDS_BLOCK_ELEMENTS
        p_block_raw_ptrs = (
            p_blocks_raw,
            p_blocks_raw + self.PDS_BLOCK_ELEMENTS,
        )
        ds_block_raw_ptrs = (
            ds_blocks_raw,
            ds_blocks_raw + self.PDS_BLOCK_ELEMENTS,
        )
        flat_pds_block_layout = cute.make_layout(
            (self.PDS_BLOCK_ELEMENTS,),
            stride=(1,),
        )
        p_xchg_raw = storage.p_xchg.get_tensor(
            flat_pds_block_layout
        )
        ds_xchg_raw = storage.ds_xchg.get_tensor(
            flat_pds_block_layout
        )
        softmax_stats = storage.stats.get_tensor(
            cute.make_layout(
                (self.H_TILE_CTA, 2),
                stride=(1, self.H_TILE_CTA),
            )
        )

        # ------------------------------------------------------------------
        # GMEM partitions.
        # ------------------------------------------------------------------
        stats_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(
                cache_mode=cpasync.LoadCacheMode.ALWAYS
            ),
            self.acc_dtype,
            num_bits_per_copy=64,
        )
        stats_tiled_copy = cute.make_tiled_copy_tv(
            stats_copy_atom,
            cute.make_layout((32,), stride=(1,)),
            cute.make_layout((2,), stride=(1,)),
        )
        stats_thread_copy = stats_tiled_copy.get_slice(
            tidx % Int32(32)
        )
        g_scaled_lse = cute.flat_divide(
            scaled_lse,
            (self.H_TILE_CTA,),
        )
        g_sum_odo = cute.flat_divide(
            sum_odo,
            (self.H_TILE_CTA,),
        )
        t_g_scaled_lse = stats_thread_copy.partition_S(
            g_scaled_lse[None, rank, (token_idx, batch_idx)]
        )
        t_s_scaled_lse = stats_thread_copy.partition_D(
            softmax_stats[None, 0]
        )
        t_g_sum_odo = stats_thread_copy.partition_S(
            g_sum_odo[None, rank, (token_idx, batch_idx)]
        )
        t_s_sum_odo = stats_thread_copy.partition_D(
            softmax_stats[None, 1]
        )

        g_q = cute.local_tile(
            tma_tensor_q,
            cute.select(
                (self.H_TILE_CTA, self.N_TILE, self.D_HEAD),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        g_do = cute.local_tile(
            tma_tensor_do,
            cute.select(
                (self.H_TILE_CTA, self.N_TILE, self.D_HEAD),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        stationary_thr_mma = stationary_tiled_mma.get_slice(0)
        rank_g_q = stationary_thr_mma.partition_A(g_q)
        rank_g_do = stationary_thr_mma.partition_A(g_do)
        t_q_smem, t_q_gmem = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(stationary_q_tma, 0, 3),
            cute.group_modes(rank_g_q, 0, 3),
        )
        t_do_smem, t_do_gmem = cpasync.tma_partition(
            tma_atom_do,
            0,
            cute.make_layout(1),
            cute.group_modes(stationary_do_tma, 0, 3),
            cute.group_modes(rank_g_do, 0, 3),
        )

        rank_score_mma = score_tiled_mma.get_slice(rank)
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

        # Per-CTA quadrant TMA partitions (mQT/mdOT tiled (D256, H64)).
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
        a_cta_layout = cute.make_layout(
            cute.slice_(
                cluster_layout_vmnk,
                (0, 0, None, 0),
            ).shape
        )
        t_qt_smem_a, t_qt_gmem = cpasync.tma_partition(
            tma_atom_qt,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(round_quad[0], 0, 3),
            cute.group_modes(rank_g_qt, 0, 3),
        )
        t_qt_smem_b, _ = cpasync.tma_partition(
            tma_atom_qt,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(round_quad[1], 0, 3),
            cute.group_modes(rank_g_qt, 0, 3),
        )
        t_dot_smem_a, t_dot_gmem = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(round_quad[0], 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )
        t_dot_smem_b, _ = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(round_quad[1], 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )
        t_dot_loan_smem_a, _ = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(loan_quad[0], 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )
        t_dot_loan_smem_b, _ = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(loan_quad[1], 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )

        # ------------------------------------------------------------------
        # MMA fragments.
        # ------------------------------------------------------------------
        score_q_fragment = score_tiled_mma.make_fragment_A(
            stationary_q
        )
        score_do_fragment = dp_tiled_mma.make_fragment_A(
            stationary_do
        )
        score_k_fragment = score_tiled_mma.make_fragment_B(k_n)
        dp_k_fragment = dp_tiled_mma.make_fragment_B(k_n)
        dq_kd_fragment_a = dq_tiled_mma.make_fragment_A(
            round_kd[0]
        )
        dq_kd_fragment_b = dq_tiled_mma.make_fragment_A(
            round_kd[1]
        )
        dq_ds_fragment = dq_tiled_mma.make_fragment_B(ds_image)
        quad_fragment_a = dkv_tiled_mma.make_fragment_A(
            round_quad[0]
        )
        quad_fragment_b = dkv_tiled_mma.make_fragment_A(
            round_quad[1]
        )
        loan_quad_fragment_a = dkv_tiled_mma.make_fragment_A(
            loan_quad[0]
        )
        loan_quad_fragment_b = dkv_tiled_mma.make_fragment_A(
            loan_quad[1]
        )
        p_fragments = (
            dkv_tiled_mma.make_fragment_B(p_blocks[0]),
            dkv_tiled_mma.make_fragment_B(p_blocks[1]),
        )
        ds_fragments = (
            dkv_tiled_mma.make_fragment_B(ds_blocks[0]),
            dkv_tiled_mma.make_fragment_B(ds_blocks[1]),
        )

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

        # ------------------------------------------------------------------
        # Pipelines.
        # ------------------------------------------------------------------
        atom_thr_size = cute.size(score_tiled_mma.thr_id.shape)
        leader_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            1,
        )
        math_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            atom_thr_size * self.MATH_THREADS,
        )
        gather_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            atom_thr_size * self.GATHER_THREADS,
        )
        reduce_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            atom_thr_size * self.REDUCE_THREADS,
        )
        load_elect_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            atom_thr_size,
        )

        pipe_s_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.SCORE_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.s_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dp_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.SCORE_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dp_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_kscore = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=gather_group,
            consumer_group=leader_group,
            barrier_storage=storage.kscore_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_round = pipeline.PipelineAsyncUmma.create(
            num_stages=self.ROUND_STAGES,
            producer_group=load_elect_group,
            consumer_group=leader_group,
            barrier_storage=storage.round_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pds_commit_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            atom_thr_size,
        )
        pipe_pds = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pds_commit_group,
            consumer_group=leader_group,
            barrier_storage=storage.pds_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dkv_done = pipeline.PipelineUmmaAsync.create(
            num_stages=self.MMA_DONE_STAGES,
            producer_group=leader_group,
            consumer_group=reduce_group,
            barrier_storage=storage.dkv_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipe_dq_done = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dq_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
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
            cute.arch.mbarrier_init(
                pds_ready_mbars,
                self.MATH_THREADS,
            )
            cute.arch.mbarrier_init(round_tma_mbars, 1)
            cute.arch.mbarrier_init(round_tma_mbars + 1, 1)
            cute.arch.mbarrier_init(loan_tma_mbars, 1)
            cute.arch.mbarrier_init(loan_tma_mbars + 1, 1)
            cute.arch.mbarrier_init(
                kdq_ready_mbar,
                self.GATHER_THREADS,
            )
            # E1: chunk c ready when every gather thread of BOTH CTAs
            # has arrived once (remote arrives land on the leader CTA).
            cute.arch.mbarrier_init(
                kchunk_mbars,
                2 * self.GATHER_THREADS,
            )
            cute.arch.mbarrier_init(
                kchunk_mbars + 1,
                2 * self.GATHER_THREADS,
            )
            # E4b: [2] = borrow TMA landing (expect_tx, one arm per
            # tile); [3] = borrow_free (one tcgen05 commit arrival).
            cute.arch.mbarrier_init(kchunk_mbars + 2, 1)
            cute.arch.mbarrier_init(kchunk_mbars + 3, 1)
            # E3: one tcgen05 commit arrival per chunk per tile.
            cute.arch.mbarrier_init(krel_mbars, 1)
            cute.arch.mbarrier_init(krel_mbars + 1, 1)
            cute.arch.mbarrier_init(krel_mbars + 2, 1)
            cute.arch.mbarrier_init(krel_mbars + 3, 1)
            cute.arch.mbarrier_init(loan_epi_safe_mbar, 1)
            _store_shared_seq_v4(khot_seq, Int32(0))
        cute.arch.fence_view_async_shared()
        self.cta_barrier.arrive_and_wait()

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
            allocator_warp_id=self.MATH_WARP_BEGIN,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        score_c_layout = score_tiled_mma.make_fragment_C(
            score_tiled_mma.partition_shape_C(
                (self.H_TILE_CLUSTER, self.N_TILE)
            )
        ).layout
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(
            dkv_tiled_mma.partition_shape_C(
                self.DKV_MMA_TILER[:2]
            )
        ).layout
        dq_c_layout = dq_tiled_mma.make_fragment_C(
            dq_tiled_mma.partition_shape_C(
                self.DQ_MMA_TILER[:2]
            )
        ).layout
        t_score = cute.make_tensor(
            tmem_ptr + self.TMEM_S_OFFSET,
            score_c_layout,
        )
        t_score_pp = cute.make_tensor(
            tmem_ptr + self.TMEM_S1_OFFSET,
            score_c_layout,
        )
        t_dp = cute.make_tensor(
            tmem_ptr + self.TMEM_DP_OFFSET,
            score_c_layout,
        )
        t_dp_pp = cute.make_tensor(
            tmem_ptr + self.TMEM_DP1_OFFSET,
            score_c_layout,
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

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = Int32(mTopkIdxs.shape[0])
        if topk > Int32(mTopkIdxs.shape[0]):
            topk = Int32(mTopkIdxs.shape[0])
        if topk < Int32(0):
            topk = Int32(0)
        tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(
            self.N_TILE
        )

        if (
            warp_idx < Int32(self.MATH_WARP_BEGIN)
            or warp_idx >= Int32(self.MMA_WARP)
        ):
            cute.arch.setmaxregister_decrease(48)
        else:
            if warp_idx < Int32(self.REDUCE_WARP_BEGIN):
                cute.arch.setmaxregister_increase(128)
            else:
                cute.arch.setmaxregister_increase(128)

        # ==================================================================
        # Role bodies.
        # ==================================================================
        if warp_idx < Int32(self.GATHER_WARPS):
            gather_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            loan_phase = Int32(0)
            gather_kd_rows_0 = self._kd_round_rows_v2(round_kd[0])
            gather_kd_rows_1 = self._kd_round_rows_v2(round_kd[1])
            if tile_count > Int32(0):
                # Prologue: K(0), chunk-streamed (E1).
                pipe_kscore.producer_acquire(gather_state)
                self._load_score_kv_streamed(
                    mKV,
                    mTopkIdxs,
                    k_n,
                    kchunk_mbars,
                    token_idx,
                    batch_idx,
                    tile_count - Int32(1),
                    topk,
                    rank,
                    tidx,
                    kv_copy_atom,
                    kv_thread_copy,
                )
                pipe_kscore.producer_commit(gather_state)
                gather_state.advance()

                if tile_count == Int32(1):
                    # The only score tile is followed directly by its tail
                    # gradients: publish kdq early, then borrow score_kv for
                    # the round-0 dO pair once score has released K(0).
                    self._gather_kdq_v8(
                        mKV,
                        mTopkIdxs,
                        gather_kd_rows_0,
                        gather_kd_rows_1,
                        token_idx,
                        batch_idx,
                        Int32(0),
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                        kdq_ready_mbar,
                    )
                    pipe_kscore.producer_acquire(gather_state)
                    loan_phase = self._fill_score_loan_do_r0_vc2(
                        rank,
                        warp_idx,
                        stationary_do_raw,
                        score_kv_raw,
                        loan_tma_mbars,
                        loan_phase,
                        tma_atom_dot,
                        t_dot_gmem,
                        t_dot_loan_smem_a,
                        t_dot_loan_smem_b,
                        grad_a_stage_bytes,
                    )
                    pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                else:
                    # No preceding gradient exists after score(0), so the
                    # first recycled score stage still carries K(1).
                    next_iter = Int32(1)
                    # E3: chunks 0..2 of K(1) enter as dP(0) retires
                    # K(0)'s chunks; the pipe lease gates only chunk 3.
                    self._load_score_kv_streamed_early(
                        mKV,
                        mTopkIdxs,
                        k_n,
                        kchunk_mbars,
                        krel_mbars,
                        pipe_kscore,
                        gather_state,
                        token_idx,
                        batch_idx,
                        tile_count - Int32(1) - next_iter,
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    pipe_kscore.producer_commit(gather_state)
                    gather_state.advance()
                    self._gather_kdq_v8(
                        mKV,
                        mTopkIdxs,
                        gather_kd_rows_0,
                        gather_kd_rows_1,
                        token_idx,
                        batch_idx,
                        tile_count - Int32(1),
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                        kdq_ready_mbar,
                    )

                    # score_iter i consumes K(i), then grads(i-1) consumes
                    # the borrowed dO generation.  The second generation is
                    # K(i+1), except in the final iteration where it is the
                    # tail tile's borrowed dO generation.
                    gather_borrow_phase = Int32(0)
                    for score_iter in cutlass.range(
                        Int32(1),
                        tile_count,
                    ):
                        pipe_kscore.producer_acquire(gather_state)
                        loan_phase = self._fill_score_loan_do_r0_vc2(
                            rank,
                            warp_idx,
                            stationary_do_raw,
                            score_kv_raw,
                            loan_tma_mbars,
                            loan_phase,
                            tma_atom_dot,
                            t_dot_gmem,
                            t_dot_loan_smem_a,
                            t_dot_loan_smem_b,
                            grad_a_stage_bytes,
                        )
                        pipe_kscore.producer_commit(gather_state)
                        gather_state.advance()

                        if score_iter == tile_count - Int32(1):
                            # E4b FIX: the base fence-recommit is dead --
                            # the borrowed dO_r1 h0 fill of grads(tc-2)
                            # dirties the loan h1 bytes (16-32K) between
                            # the two loan consumptions, so the tail
                            # tile's loan must be REFILLED.  Ordering:
                            # wait borrow_free(tc-2) (the leader's
                            # commit after dV_r1 pass a) -- at that
                            # arrival the old loan (pipe acquire), the
                            # borrowed tenant (this signal) and the K
                            # bytes (dP(tc-1) < dVr1a in-order) are all
                            # provably dead, so the full refill races
                            # nobody.  The phase continues the steady
                            # session-B counter (fills consumed
                            # 0..tc-3; this waits tc-2).
                            pipe_kscore.producer_acquire(gather_state)
                            cute.arch.mbarrier_wait(
                                kchunk_mbars + 3,
                                gather_borrow_phase,
                            )
                            gather_borrow_phase = (
                                Int32(1) - gather_borrow_phase
                            )
                            loan_phase = self._fill_score_loan_do_r0_vc2(
                                rank,
                                warp_idx,
                                stationary_do_raw,
                                score_kv_raw,
                                loan_tma_mbars,
                                loan_phase,
                                tma_atom_dot,
                                t_dot_gmem,
                                t_dot_loan_smem_a,
                                t_dot_loan_smem_b,
                                grad_a_stage_bytes,
                            )
                            pipe_kscore.producer_commit(gather_state)
                            gather_state.advance()
                            self._gather_kdq_v8(
                                mKV,
                                mTopkIdxs,
                                gather_kd_rows_0,
                                gather_kd_rows_1,
                                token_idx,
                                batch_idx,
                                Int32(0),
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                                kdq_ready_mbar,
                            )
                        else:
                            next_iter = score_iter + Int32(1)
                            pipe_kscore.producer_acquire(gather_state)
                            self._load_score_kv_streamed_gated(
                                mKV,
                                mTopkIdxs,
                                k_n,
                                kchunk_mbars,
                                kchunk_mbars + 3,
                                gather_borrow_phase,
                                token_idx,
                                batch_idx,
                                tile_count - Int32(1) - next_iter,
                                topk,
                                rank,
                                tidx,
                                kv_copy_atom,
                                kv_thread_copy,
                            )
                            gather_borrow_phase = (
                                Int32(1) - gather_borrow_phase
                            )
                            pipe_kscore.producer_commit(gather_state)
                            gather_state.advance()
                            self._gather_kdq_v8(
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
                                kdq_ready_mbar,
                            )
                pipe_kscore.producer_tail(gather_state)
                # producer_tail observes the final borrowed dO generation's
                # true UMMA source-read completion.  Publish that fact to the
                # math epilogue before it reuses score_kv as dQ staging.
                self.gather_barrier.arrive_and_wait()
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        # Signal this CTA's local epilogue consumers.  The
                        # rank operand maps the local pointer back to self;
                        # rank 0 would strand CTA 1 on its private mbarrier.
                        cute.arch.mbarrier_arrive(
                            loan_epi_safe_mbar,
                            rank,
                        )

        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            # --- math: stats, per-tile softmax + publication, dQ epilogue.
            mtx = tidx - Int32(self.MATH_THREAD_BEGIN)
            if warp_idx == Int32(self.MATH_WARP_BEGIN):
                if tile_count > Int32(0):
                    cute.copy(
                        stats_copy_atom,
                        t_g_scaled_lse[None, 0],
                        t_s_scaled_lse[None, 0],
                    )
                    cute.copy(
                        stats_copy_atom,
                        t_g_sum_odo[None, 0],
                        t_s_sum_odo[None, 0],
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
            self.math_barrier.arrive_and_wait()

            s_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.SCORE_DONE_STAGES,
            )
            dp_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.SCORE_DONE_STAGES,
            )
            pds_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            dq_done_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                1,
            )

            score_copy = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_score,
            )
            score_thread = score_copy.get_slice(mtx)
            score_source = score_thread.partition_S(t_score)
            score_coordinates = score_thread.partition_D(
                rank_score_coordinates
            )
            dp_copy = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_dp,
            )
            dp_thread = dp_copy.get_slice(mtx)
            dp_source = dp_thread.partition_S(t_dp)
            score_copy_pp = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_score_pp,
            )
            score_source_pp = score_copy_pp.get_slice(
                mtx
            ).partition_S(t_score_pp)
            dp_copy_pp = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_dp_pp,
            )
            dp_source_pp = dp_copy_pp.get_slice(
                mtx
            ).partition_S(t_dp_pp)
            smem_store_atom = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                self.element_dtype,
                self.acc_dtype,
                score_copy,
            )
            assert isinstance(
                smem_store_atom.op,
                warp.StMatrix8x8x16bOp,
            )
            assert smem_store_atom.op.num_matrices == 4
            tiled_copy_r2s = cute.make_tiled_copy_D(
                smem_store_atom,
                score_copy,
            )
            thread_copy_r2s = tiled_copy_r2s.get_slice(mtx)
            t_rs_ds = thread_copy_r2s.partition_D(
                ds_image_store
            )
            assert cute.size(t_rs_ds, mode=[4]) == 1
            t_rs_ds_tile = t_rs_ds[None, None, None, None, 0]
            n_owner = cute.arch.make_warp_uniform(
                Int32(
                    cute.get(
                        score_coordinates[0],
                        mode=[1],
                    )
                )
                // Int32(self.N_TILE_CTA)
            )
            owns_n = n_owner == rank
            aligned_p_blocks_ptr = cute.make_ptr(
                self.element_dtype,
                p_blocks[0].iterator.toint(),
                p_blocks[0].memspace,
                assumed_align=16,
            )
            aligned_ds_blocks_ptr = cute.make_ptr(
                self.element_dtype,
                ds_blocks[0].iterator.toint(),
                ds_blocks[0].memspace,
                assumed_align=16,
            )
            p_local_store = cute.make_tensor(
                cute.recast_ptr(
                    aligned_p_blocks_ptr,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            )
            ds_local_store = cute.make_tensor(
                cute.recast_ptr(
                    aligned_ds_blocks_ptr,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            )
            aligned_p_xchg_ptr = cute.make_ptr(
                self.element_dtype,
                p_xchg_raw.iterator.toint()
                - n_owner * Int32(self.PDS_BLOCK_BYTES),
                p_xchg_raw.memspace,
                assumed_align=16,
            )
            aligned_ds_xchg_ptr = cute.make_ptr(
                self.element_dtype,
                ds_xchg_raw.iterator.toint()
                - n_owner * Int32(self.PDS_BLOCK_BYTES),
                ds_xchg_raw.memspace,
                assumed_align=16,
            )
            p_xchg_store = cute.make_tensor(
                cute.recast_ptr(
                    aligned_p_xchg_ptr,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            )
            ds_xchg_store = cute.make_tensor(
                cute.recast_ptr(
                    aligned_ds_xchg_ptr,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            )
            t_rs_p_local = thread_copy_r2s.partition_D(
                p_local_store
            )
            t_rs_ds_local = thread_copy_r2s.partition_D(
                ds_local_store
            )
            t_rs_p_xchg = thread_copy_r2s.partition_D(
                p_xchg_store
            )
            t_rs_ds_xchg = thread_copy_r2s.partition_D(
                ds_xchg_store
            )
            assert cute.size(t_rs_p_local, mode=[4]) == 1
            assert cute.size(t_rs_ds_local, mode=[4]) == 1
            assert cute.size(t_rs_p_xchg, mode=[4]) == 1
            assert cute.size(t_rs_ds_xchg, mode=[4]) == 1
            t_rs_p_local_tile = t_rs_p_local[
                None, None, None, None, 0
            ]
            t_rs_ds_local_tile = t_rs_ds_local[
                None, None, None, None, 0
            ]
            t_rs_p_xchg_tile = t_rs_p_xchg[
                None, None, None, None, 0
            ]
            t_rs_ds_xchg_tile = t_rs_ds_xchg[
                None, None, None, None, 0
            ]
            r_score = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.acc_dtype,
            )
            r_dp = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.acc_dtype,
            )
            r_p = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.element_dtype,
            )
            r_ds = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.element_dtype,
            )

            for loop_iter in cutlass.range(tile_count):
                pipe_s_done.consumer_wait(s_state)
                if s_state.index == Int32(0):
                    cute.copy(score_copy, score_source, r_score)
                else:
                    cute.copy(
                        score_copy_pp,
                        score_source_pp,
                        r_score,
                    )
                cute.arch.fence_view_async_tmem_load()
                pipe_s_done.consumer_release(s_state)
                s_state.advance()

                pipe_dp_done.consumer_wait(dp_state)
                if dp_state.index == Int32(0):
                    cute.copy(dp_copy, dp_source, r_dp)
                else:
                    cute.copy(dp_copy_pp, dp_source_pp, r_dp)
                cute.arch.fence_view_async_tmem_load()
                pipe_dp_done.consumer_release(dp_state)
                dp_state.advance()

                softmax_scale_log2_e = scale_softmax * Float32(
                    math.log2(math.e)
                )
                assert cute.size(r_score) == self.N_TILE_CTA
                if cutlass.const_expr(self.SOFTMAX_GROUPED_STATS):
                    group_bases = [
                        2 * (h_group % 2) + 16 * (h_group // 2)
                        for h_group in range(4)
                    ]
                    group_local_h = [
                        Int32(
                            cute.get(
                                score_coordinates[group_base],
                                mode=[0],
                            )
                        )
                        % Int32(self.H_TILE_CTA)
                        for group_base in group_bases
                    ]
                    band_indices = [
                        [
                            group_base + (j % 2) + 4 * (j // 2)
                            for j in range(8)
                        ]
                        for group_base in group_bases
                    ]
                    for h_group in cutlass.range_constexpr(4):
                        lse = softmax_stats[group_local_h[h_group], 0]
                        for value_index in band_indices[h_group]:
                            r_score[value_index] = cute.math.exp2(
                                (
                                    r_score[value_index]
                                    * softmax_scale_log2_e
                                    + lse
                                ),
                                fastmath=True,
                            )
                    for h_group in cutlass.range_constexpr(4):
                        delta = softmax_stats[
                            group_local_h[h_group], 1
                        ]
                        for value_index in band_indices[h_group]:
                            p_value = r_score[value_index]
                            ds_value = (
                                (r_dp[value_index] + delta)
                                * p_value
                                * scale_softmax
                            )
                            r_p[value_index] = self.element_dtype(
                                p_value
                            )
                            r_ds[value_index] = self.element_dtype(
                                ds_value
                            )
                else:
                    # Assumption-free fallback: per-value coordinate
                    # lookup (32 stats loads/thread).  Flip the class
                    # flag if the grouped arithmetic is ever suspect.
                    for local_n in cutlass.range_constexpr(
                        self.N_TILE_CTA
                    ):
                        local_h = Int32(
                            cute.get(
                                score_coordinates[local_n],
                                mode=[0],
                            )
                        ) % Int32(self.H_TILE_CTA)
                        lse = softmax_stats[local_h, 0]
                        delta = softmax_stats[local_h, 1]
                        p_value = cute.math.exp2(
                            (
                                r_score[local_n]
                                * softmax_scale_log2_e
                                + lse
                            ),
                            fastmath=True,
                        )
                        ds_value = (
                            (r_dp[local_n] + delta)
                            * p_value
                            * scale_softmax
                        )
                        r_p[local_n] = self.element_dtype(p_value)
                        r_ds[local_n] = self.element_dtype(ds_value)

                pipe_pds.producer_acquire(pds_state)

                # Publish P/dS with stmatrix. Each pair of warps owns one
                # N32 half, so this branch is warp-uniform.
                r_p_store = thread_copy_r2s.retile(r_p)
                r_ds_store = thread_copy_r2s.retile(r_ds)
                assert t_rs_p_local_tile.shape == r_p_store.shape
                assert t_rs_ds_local_tile.shape == r_ds_store.shape
                assert t_rs_p_xchg_tile.shape == r_p_store.shape
                assert t_rs_ds_xchg_tile.shape == r_ds_store.shape
                if owns_n:
                    cute.copy(
                        tiled_copy_r2s,
                        r_p_store,
                        t_rs_p_local_tile,
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        r_ds_store,
                        t_rs_ds_local_tile,
                    )
                else:
                    cute.copy(
                        tiled_copy_r2s,
                        r_p_store,
                        t_rs_p_xchg_tile,
                    )

                # Whole-image dS store for the dQ B operand.
                assert t_rs_ds_tile.shape == r_ds_store.shape
                cute.copy(
                    tiled_copy_r2s,
                    r_ds_store,
                    t_rs_ds_tile,
                )

                # No validity mask (baseline-identical invariant pair):
                # invalid columns see S=dP=0 from zero-filled K rows, so
                # P/dS stay finite; dQ is protected by zero-filled K_dQ
                # rows and dKV garbage columns are dropped by the drain
                # predicates (global_n < topk, kv_index >= 0).

                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(
                    pds_ready_mbars,
                    rank,
                )
                pds_state.advance()

            # dQ epilogue: wait for the last dQ generation, then store both
            # rank-owned [D128, H128] slices (disjoint across CTAs/rounds).
            if tile_count > Int32(0):
                pipe_dq_done.consumer_wait(dq_done_state)
                _mbarrier_wait_acquire_cluster(
                    loan_epi_safe_mbar,
                    Int32(0),
                )
                # E4b: the borrowed dO_r1 h0 of the LAST tile is read by
                # dV_r1 pass a strictly later than loan_epi_safe's
                # anchor; the epilogue staging overwrites its parcel, so
                # wait the final borrow_free phase (one commit per tile,
                # last phase = (tile_count-1) mod 2).
                _mbarrier_wait_acquire_cluster(
                    kchunk_mbars + 3,
                    (tile_count - Int32(1)) % Int32(2),
                )
                self._store_dq_epi_tma_v12(
                    t_dq[0],
                    dq_tmem_load,
                    rank_dq_coordinates,
                    s_dq_epi,
                    tma_atom_dq_epi,
                    tma_tensor_dq_epi,
                    0,
                    token_idx,
                    batch_idx,
                    rank,
                    mtx,
                )
                self._store_dq_epi_tma_v12(
                    t_dq[1],
                    dq_tmem_load,
                    rank_dq_coordinates,
                    s_dq_epi,
                    tma_atom_dq_epi,
                    tma_tensor_dq_epi,
                    1,
                    token_idx,
                    batch_idx,
                    rank,
                    mtx,
                )
                pipe_dq_done.consumer_release(dq_done_state)
                dq_done_state.advance()
            else:
                self._zero_dq_v2(
                    rank_dq_coordinates,
                    mdQ,
                    0,
                    token_idx,
                    batch_idx,
                    mtx,
                )
                self._zero_dq_v2(
                    rank_dq_coordinates,
                    mdQ,
                    1,
                    token_idx,
                    batch_idx,
                    mtx,
                )

        elif warp_idx < Int32(self.MMA_WARP):
            # --- reduce: one fused drain call per tile; slot 0 is T2R'd
            # and released off the head commit, slot 1 off the tail commit,
            # then both atomic bursts run back-to-back.  Split wait/release
            # states let each release trail its own fence.
            rtx = tidx - Int32(self.REDUCE_THREAD_BEGIN)
            dkv_wait = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.MMA_DONE_STAGES,
            )
            dkv_rel = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.MMA_DONE_STAGES,
            )
            for loop_iter in cutlass.range(tile_count):
                tile_index = tile_count - Int32(1) - loop_iter
                dkv_wait, dkv_rel = self._drain_dkv_v8(
                    t_dkv[0],
                    t_dkv[1],
                    mdKV_acc,
                    mTopkIdxs,
                    tile_index,
                    topk,
                    token_idx,
                    batch_idx,
                    rtx,
                    rank,
                    loop_iter,
                    pipe_dkv_done,
                    dkv_wait,
                    dkv_rel,
                )

        elif warp_idx == Int32(self.MMA_WARP):
            # --- leader MMA: rotated schedule.  The follower CTA's MMA warp
            # executes no pipeline operation at all (FA4 rule).
            if is_leader_cta:
                s_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.SCORE_DONE_STAGES,
                )
                dp_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.SCORE_DONE_STAGES,
                )
                kscore_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    1,
                )
                round_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.ROUND_STAGES,
                )
                pds_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    1,
                )
                dkv_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.MMA_DONE_STAGES,
                )
                dq_done_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    1,
                )
                if tile_count > Int32(0):
                    _mbarrier_wait_acquire_cluster(
                        stationary_ready_mbar,
                        Int32(0),
                    )
                pipe_dq_done.producer_acquire(dq_done_prod)

                relay_phase = Int32(0)
                # E3: multicast mask for the dP release train (self |
                # peer), same recipe as the kscore pipe's UMMA release.
                krel_peer_mask = (
                    pipeline.PipelineAsyncUmma._compute_peer_cta_mask(
                        cluster_layout_vmnk
                    )
                )
                # E4a: S gates on the two SESSION barriers (chunks 0-1,
                # then 2-3) instead of the whole-tile lease; the lease
                # wait moves down to dP (second reader, needs the full
                # tile anyway -- E1's proven position).  This is the
                # split-fill tax probe: two in-loop gates, both
                # trivially satisfied while session B fills run
                # back-to-back with session A (no lending yet).
                kchunk_phase = Int32(0)
                for loop_iter in cutlass.range(tile_count):
                    has_prev = loop_iter > Int32(0)

                    s_prod = self._issue_score_v2_sess(
                        score_tiled_mma,
                        t_score,
                        t_score_pp,
                        score_q_fragment,
                        score_k_fragment,
                        pipe_s_done,
                        s_prod,
                        kchunk_mbars,
                        kchunk_phase,
                    )
                    kchunk_phase = Int32(1) - kchunk_phase

                    # dP(t) then early K recycle.  The first dP also
                    # gates on the dO half of the split stationary load.
                    if loop_iter == Int32(0):
                        _mbarrier_wait_acquire_cluster(
                            stationary_ready_mbar + 1,
                            Int32(0),
                        )
                    pipe_kscore.consumer_wait(kscore_cons)
                    dp_prod = self._issue_dp_v2_release(
                        dp_tiled_mma,
                        t_dp,
                        t_dp_pp,
                        score_do_fragment,
                        dp_k_fragment,
                        pipe_dp_done,
                        dp_prod,
                        krel_mbars,
                        krel_peer_mask,
                        loop_iter == Int32(0),
                    )
                    pipe_kscore.consumer_release(kscore_cons)
                    kscore_cons.advance()

                    if has_prev:
                        dq_acc = loop_iter != Int32(1)
                        (
                            round_cons,
                            kscore_cons,
                            dkv_prod,
                            pds_cons,
                            dq_done_prod,
                        ) = (
                            self._issue_prev_grads_head_v2(
                                dq_tiled_mma,
                                dkv_tiled_mma,
                                t_dq[0],
                                t_dq[1],
                                t_dkv[0],
                                dq_kd_fragment_a,
                                dq_kd_fragment_b,
                                dq_ds_fragment,
                                quad_fragment_a,
                                quad_fragment_b,
                                loan_quad_fragment_a,
                                loan_quad_fragment_b,
                                p_fragments[0],
                                p_fragments[1],
                                ds_fragments[0],
                                ds_fragments[1],
                                dq_acc,
                                relay_phase,
                                relay_mbars,
                                pipe_round,
                                round_cons,
                                pipe_kscore,
                                kscore_cons,
                                pipe_pds,
                                pds_cons,
                                pipe_dkv_done,
                                dkv_prod,
                                pipe_dq_done,
                                dq_done_prod,
                                False,
                                loop_iter - Int32(1),
                            )
                        )
                        round_cons, dkv_prod = (
                            self._issue_prev_grads_tail_v2(
                                dkv_tiled_mma,
                                t_dkv[1],
                                quad_fragment_a,
                                quad_fragment_b,
                                loan_quad_fragment_b,
                                p_fragments[0],
                                p_fragments[1],
                                ds_fragments[0],
                                ds_fragments[1],
                                pipe_round,
                                round_cons,
                                pipe_dkv_done,
                                dkv_prod,
                                kchunk_mbars,
                                krel_peer_mask,
                                loop_iter - Int32(1),
                            )
                        )
                        pipe_pds.consumer_release(pds_cons)
                        pds_cons.advance()
                        relay_phase = Int32(1) - relay_phase

                # Drain the final tile's gradients.
                if tile_count > Int32(0):
                    dq_acc = tile_count != Int32(1)
                    (
                        round_cons,
                        kscore_cons,
                        dkv_prod,
                        pds_cons,
                        dq_done_prod,
                    ) = (
                        self._issue_prev_grads_head_v2(
                            dq_tiled_mma,
                            dkv_tiled_mma,
                            t_dq[0],
                            t_dq[1],
                            t_dkv[0],
                            dq_kd_fragment_a,
                            dq_kd_fragment_b,
                            dq_ds_fragment,
                            quad_fragment_a,
                            quad_fragment_b,
                            loan_quad_fragment_a,
                            loan_quad_fragment_b,
                            p_fragments[0],
                            p_fragments[1],
                            ds_fragments[0],
                            ds_fragments[1],
                            dq_acc,
                            relay_phase,
                            relay_mbars,
                            pipe_round,
                            round_cons,
                            pipe_kscore,
                            kscore_cons,
                            pipe_pds,
                            pds_cons,
                            pipe_dkv_done,
                            dkv_prod,
                            pipe_dq_done,
                            dq_done_prod,
                            True,
                            tile_count - Int32(1),
                        )
                    )
                    round_cons, dkv_prod = (
                        self._issue_prev_grads_tail_v2(
                            dkv_tiled_mma,
                            t_dkv[1],
                            quad_fragment_a,
                            quad_fragment_b,
                            loan_quad_fragment_b,
                            p_fragments[0],
                            p_fragments[1],
                            ds_fragments[0],
                            ds_fragments[1],
                            pipe_round,
                            round_cons,
                            pipe_dkv_done,
                            dkv_prod,
                            kchunk_mbars,
                            krel_peer_mask,
                            tile_count - Int32(1),
                        )
                    )
                    pipe_pds.consumer_release(pds_cons)
                    pds_cons.advance()

                    pipe_s_done.producer_tail(s_prod)
                    pipe_dp_done.producer_tail(dp_prod)
                    pipe_dkv_done.producer_tail(dkv_prod)
                    pipe_dq_done.producer_tail(dq_done_prod)

        elif warp_idx == Int32(self.LOAD_WARP):
            lane_idx = tidx % Int32(32)
            round_acq = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            round_com = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            kdq_ready_phase = Int32(0)
            if tile_count > Int32(0):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        stationary_tma_mbars,
                        score_a_stage_bytes * self.K_CHUNKS,
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        stationary_tma_mbars + 1,
                        score_a_stage_bytes * self.K_CHUNKS,
                    )
                cute.copy(
                    tma_atom_q,
                    t_q_gmem[None, rank, 0],
                    t_q_smem[None, 0],
                    tma_bar_ptr=stationary_tma_mbars,
                )
                cute.copy(
                    tma_atom_do,
                    t_do_gmem[None, rank, 0],
                    t_do_smem[None, 0],
                    tma_bar_ptr=stationary_tma_mbars + 1,
                )
                # Split readiness: S needs only Q, dP needs only dO, so the
                # leader can issue the first S one TMA earlier.
                cute.arch.mbarrier_wait(
                    stationary_tma_mbars,
                    Int32(0),
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        stationary_ready_mbar,
                        Int32(0),
                    )
                cute.arch.mbarrier_wait(
                    stationary_tma_mbars + 1,
                    Int32(0),
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        stationary_ready_mbar + 1,
                        Int32(0),
                    )

                for loop_iter in cutlass.range(tile_count):
                    tile_index = (
                        tile_count - Int32(1) - loop_iter
                    )
                    # g0/g1: both K_dQ rounds in one fused gather pass
                    # (one index read per row; both stage credits held).
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    self.kdq_barrier.arrive_and_wait()
                    cute.arch.mbarrier_wait(
                        kdq_ready_mbar,
                        kdq_ready_phase,
                    )
                    kdq_ready_phase = Int32(1) - kdq_ready_phase
                    cute.arch.fence_view_async_shared()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()

                    # g2..g7: Q_r0, dO_r1, Q_r1 panel fills, pipelined
                    # depth 2.  dO_r0 is supplied through score_kv by the
                    # gather role.
                    # commit for gen q-1 happens after gen q's TMA is in
                    # flight; barrier q%2 was last waited at iteration q-1,
                    # so it is never re-armed while pending.
                    for flat_gen in cutlass.range_constexpr(6):
                        if cutlass.const_expr(flat_gen < 2):
                            grad_round = 0
                            tensor_kind = 1  # Q_r0
                        elif cutlass.const_expr(flat_gen < 4):
                            grad_round = 1
                            tensor_kind = 0  # dO_r1
                        else:
                            grad_round = 1
                            tensor_kind = 1  # Q_r1
                        h_half = flat_gen % 2
                        pipe_round.producer_acquire(round_acq)
                        round_acq.advance()
                        # E4b: g4 (dO_r1 h0) is the borrowed generation:
                        # it lands in the score_kv loan_quad[1] parcel
                        # and completes on the borrow mbar (W19 bridges
                        # it into the normal pipe commit).
                        if cutlass.const_expr(flat_gen == 2):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    kchunk_mbars + 2,
                                    grad_a_stage_bytes,
                                )
                        else:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    round_tma_mbars + h_half,
                                    grad_a_stage_bytes,
                                )
                        # The h==rank panel is a contiguous byte-identical
                        # 16 KiB slice of this CTA's own stationary tensor
                        # (K-major [H64,D512] M-atom pair 4r+2c): one local
                        # bulk copy replaces the fragmented 64-row TMA box.
                        # The h==peer panel keeps the GMEM TMA (L2-hot).
                        if cutlass.const_expr(tensor_kind == 0):
                            if cutlass.const_expr(h_half == 0):
                                if cutlass.const_expr(self.OWN_HALF_BULK):
                                    if rank == Int32(0):
                                        with cute.arch.elect_one():
                                            _cpasync_bulk_s2cluster(
                                                stationary_do_raw
                                                + 4096 * (4 * grad_round),
                                                score_kv_raw + Int32(8192),
                                                kchunk_mbars + 2,
                                                grad_a_stage_bytes,
                                                rank,
                                            )
                                    else:
                                        cute.copy(
                                            tma_atom_dot,
                                            t_dot_gmem[
                                                None,
                                                grad_round,
                                                0,
                                            ],
                                            t_dot_loan_smem_b[None, 0],
                                            tma_bar_ptr=kchunk_mbars + 2,
                                        )
                                else:
                                    cute.copy(
                                        tma_atom_dot,
                                        t_dot_gmem[
                                            None,
                                            grad_round,
                                            0,
                                        ],
                                        t_dot_loan_smem_b[None, 0],
                                        tma_bar_ptr=kchunk_mbars + 2,
                                    )
                            else:
                                if cutlass.const_expr(self.OWN_HALF_BULK):
                                    if rank == Int32(1):
                                        with cute.arch.elect_one():
                                            _cpasync_bulk_s2cluster(
                                                stationary_do_raw
                                                + 4096 * (4 * grad_round + 2),
                                                round_buf_b_raw,
                                                round_tma_mbars + 1,
                                                grad_a_stage_bytes,
                                                rank,
                                            )
                                    else:
                                        cute.copy(
                                            tma_atom_dot,
                                            t_dot_gmem[
                                                None,
                                                grad_round,
                                                1,
                                            ],
                                            t_dot_smem_b[None, 0],
                                            tma_bar_ptr=round_tma_mbars + 1,
                                        )
                                else:
                                    cute.copy(
                                        tma_atom_dot,
                                        t_dot_gmem[
                                            None,
                                            grad_round,
                                            1,
                                        ],
                                        t_dot_smem_b[None, 0],
                                        tma_bar_ptr=round_tma_mbars + 1,
                                    )
                        else:
                            if cutlass.const_expr(h_half == 0):
                                if cutlass.const_expr(self.OWN_HALF_BULK):
                                    if rank == Int32(0):
                                        with cute.arch.elect_one():
                                            _cpasync_bulk_s2cluster(
                                                stationary_q_raw
                                                + 4096 * (4 * grad_round),
                                                round_buf_a_raw,
                                                round_tma_mbars,
                                                grad_a_stage_bytes,
                                                rank,
                                            )
                                    else:
                                        cute.copy(
                                            tma_atom_qt,
                                            t_qt_gmem[
                                                None,
                                                grad_round,
                                                0,
                                            ],
                                            t_qt_smem_a[None, 0],
                                            tma_bar_ptr=round_tma_mbars,
                                        )
                                else:
                                    cute.copy(
                                        tma_atom_qt,
                                        t_qt_gmem[
                                            None,
                                            grad_round,
                                            0,
                                        ],
                                        t_qt_smem_a[None, 0],
                                        tma_bar_ptr=round_tma_mbars,
                                    )
                            else:
                                if cutlass.const_expr(self.OWN_HALF_BULK):
                                    if rank == Int32(1):
                                        with cute.arch.elect_one():
                                            _cpasync_bulk_s2cluster(
                                                stationary_q_raw
                                                + 4096 * (4 * grad_round + 2),
                                                round_buf_b_raw,
                                                round_tma_mbars + 1,
                                                grad_a_stage_bytes,
                                                rank,
                                            )
                                    else:
                                        cute.copy(
                                            tma_atom_qt,
                                            t_qt_gmem[
                                                None,
                                                grad_round,
                                                1,
                                            ],
                                            t_qt_smem_b[None, 0],
                                            tma_bar_ptr=round_tma_mbars + 1,
                                        )
                                else:
                                    cute.copy(
                                        tma_atom_qt,
                                        t_qt_gmem[
                                            None,
                                            grad_round,
                                            1,
                                        ],
                                        t_qt_smem_b[None, 0],
                                        tma_bar_ptr=round_tma_mbars + 1,
                                    )
                    for _ in cutlass.range_constexpr(6):
                        round_com.advance()
                pipe_round.producer_tail(round_acq)

        elif warp_idx == Int32(self.RELAY_WARP):
            if tidx % Int32(32) == Int32(0):
                landing_phase = Int32(0)
                ready_phase = Int32(0)
                pds_com = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    1,
                )
                for loop_iter in cutlass.range(tile_count):
                    cute.arch.mbarrier_wait(
                        pds_ready_mbars,
                        ready_phase,
                    )
                    ready_phase = Int32(1) - ready_phase
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        landing_mbars,
                        self.PDS_BLOCK_BYTES,
                        peer_cta_rank_in_cluster=peer_rank,
                    )
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(
                            p_xchg_raw.iterator,
                            p_block_raw_ptrs[0],
                            landing_mbars,
                            self.PDS_BLOCK_BYTES,
                            peer_rank,
                        )
                    else:
                        _cpasync_bulk_s2cluster(
                            p_xchg_raw.iterator,
                            p_block_raw_ptrs[1],
                            landing_mbars,
                            self.PDS_BLOCK_BYTES,
                            peer_rank,
                        )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        landing_mbars + 1,
                        self.PDS_BLOCK_BYTES,
                        peer_cta_rank_in_cluster=peer_rank,
                    )
                    if rank == Int32(0):
                        _cpasync_bulk_s2cluster(
                            ds_image_raw + Int32(2048),
                            ds_block_raw_ptrs[0],
                            landing_mbars + 1,
                            self.PDS_BLOCK_BYTES,
                            peer_rank,
                        )
                    else:
                        _cpasync_bulk_s2cluster(
                            ds_image_raw,
                            ds_block_raw_ptrs[1],
                            landing_mbars + 1,
                            self.PDS_BLOCK_BYTES,
                            peer_rank,
                        )
                    pipe_pds.producer_commit(pds_com)
                    pds_com.advance()
                    _mbarrier_wait_acquire_cluster(
                        landing_mbars,
                        landing_phase,
                    )
                    cute.arch.mbarrier_arrive(
                        relay_mbars,
                        Int32(0),
                    )
                    _mbarrier_wait_acquire_cluster(
                        landing_mbars + 1,
                        landing_phase,
                    )
                    cute.arch.mbarrier_arrive(
                        relay_mbars + 1,
                        Int32(0),
                    )
                    landing_phase = Int32(1) - landing_phase
                if tile_count > Int32(0):
                    pipe_pds.producer_tail(pds_com)

        elif warp_idx == Int32(self.COMMIT_WARP):
            commit_com = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            w19_phase_0 = Int32(0)
            w19_phase_1 = Int32(0)
            w19_phase_borrow = Int32(0)
            for loop_iter in cutlass.range(tile_count):
                # g0/g1: the kdq pair, committed by the load warp's
                # rendezvous -- skip their positions.
                commit_com.advance()
                commit_com.advance()
                for flat_gen in cutlass.range_constexpr(6):
                    if cutlass.const_expr(flat_gen == 2):
                        # E4b: the borrowed generation completes on its
                        # own mbar; the commit below still rides the
                        # normal pipe (count 2 = both CTAs' bridges).
                        cute.arch.mbarrier_wait(
                            kchunk_mbars + 2,
                            w19_phase_borrow,
                        )
                        w19_phase_borrow = Int32(1) - w19_phase_borrow
                    elif cutlass.const_expr(flat_gen % 2 == 0):
                        cute.arch.mbarrier_wait(
                            round_tma_mbars,
                            w19_phase_0,
                        )
                        w19_phase_0 = Int32(1) - w19_phase_0
                    else:
                        cute.arch.mbarrier_wait(
                            round_tma_mbars + 1,
                            w19_phase_1,
                        )
                        w19_phase_1 = Int32(1) - w19_phase_1
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(commit_com)
                    commit_com.advance()

        # ==================================================================
        # Common tail: full-cluster rendezvous, then TMEM release.
        # ==================================================================
        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)

    @cute.jit
    def _issue_score_v2_sess(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
        kchunk_mbars: cute.Pointer,
        kchunk_phase: Int32,
    ) -> pipeline.PipelineState:
        """S issue gated on the two fill sessions (E4a).

        Session A (chunks 0-1) gates the first half, session B (chunks
        2-3) the second.  Two in-loop parity waits per tile -- half of
        E1's four; this variant prices exactly that pair while both
        sessions still land back-to-back (gates near-always satisfied).
        """

        done_pipeline.producer_acquire(producer_state)
        if producer_state.index == Int32(0):
            self._issue_score_chunks_sess(
                tiled_mma,
                accumulator_0,
                a_fragment,
                b_fragment,
                kchunk_mbars,
                kchunk_phase,
            )
        else:
            self._issue_score_chunks_sess(
                tiled_mma,
                accumulator_1,
                a_fragment,
                b_fragment,
                kchunk_mbars,
                kchunk_phase,
            )
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_chunks_sess(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        kchunk_mbars: cute.Pointer,
        kchunk_phase: Int32,
    ):
        """chunks_v7 with a session gate before chunk 0 and chunk 2."""

        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        # e4c-alpha: plain-parity waits -- the base pipe consumer_wait
        # was plain parity too (the publish side's fence+release arrive
        # carries visibility); the acquire.cluster spin measured +0.22us
        # inside S_ISSUE.
        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            if cutlass.const_expr(chunk == 0):
                cute.arch.mbarrier_wait(
                    kchunk_mbars,
                    kchunk_phase,
                )
            if cutlass.const_expr(chunk == 2):
                cute.arch.mbarrier_wait(
                    kchunk_mbars + 1,
                    kchunk_phase,
                )
            for k_block in cutlass.range(
                0,
                k_blocks_per_chunk,
                unroll=4,
            ):
                cute.gemm(
                    mma,
                    accumulator,
                    a_fragment[None, None, k_block, chunk],
                    b_fragment[None, None, k_block, chunk],
                    accumulator,
                )
                mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_dp_v2_release(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
        krel_mbars: cute.Pointer,
        krel_peer_mask: Int32,
        fire_train: cutlass.Boolean,
    ) -> pipeline.PipelineState:
        """dP issue with the E3 per-chunk release train.

        E34: the train's only consumer is the prologue K(1) early fill,
        so the commits fire on tile 0 only (fire_train); the other 31
        tiles keep the pure chunks_v7 issue rhythm.  The gate is a
        side-effect-only dynamic branch -- no pipeline state crosses it
        (v7 rule).

        After chunk c's k_blocks one tcgen05.commit (elect_one, CG2,
        multicast to both CTAs) arms krel_mbars[c]; the hardware
        arrival fires when every prior MMA completes, i.e. no earlier
        than S's and dP's chunk-c reads, so chunk c of score_kv is
        provably dead from that arrival.  Commits are fire-and-forget:
        zero waits added to the leader (the E1 lesson -- probes on the
        pacer cost, commits do not).
        """

        done_pipeline.producer_acquire(producer_state)
        # e4c-beta: ONE fire_train branch per issue (was 4 in-loop
        # predicates, +0.22us in dP_ISSUE); the steady path runs the
        # pristine chunks_v7 body.
        if producer_state.index == Int32(0):
            if fire_train:
                self._issue_score_chunks_v7_release(
                    tiled_mma,
                    accumulator_0,
                    a_fragment,
                    b_fragment,
                    krel_mbars,
                    krel_peer_mask,
                )
            else:
                self._issue_score_chunks_v7(
                    tiled_mma,
                    accumulator_0,
                    a_fragment,
                    b_fragment,
                )
        else:
            if fire_train:
                self._issue_score_chunks_v7_release(
                    tiled_mma,
                    accumulator_1,
                    a_fragment,
                    b_fragment,
                    krel_mbars,
                    krel_peer_mask,
                )
            else:
                self._issue_score_chunks_v7(
                    tiled_mma,
                    accumulator_1,
                    a_fragment,
                    b_fragment,
                )
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_chunks_v7_release(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        krel_mbars: cute.Pointer,
        krel_peer_mask: Int32,
    ):
        """chunks_v7 plus one retirement commit per chunk (e4c: the
        caller branches on fire_train once; this body always commits)."""

        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            for k_block in cutlass.range(
                0,
                k_blocks_per_chunk,
                unroll=4,
            ):
                cute.gemm(
                    mma,
                    accumulator,
                    a_fragment[None, None, k_block, chunk],
                    b_fragment[None, None, k_block, chunk],
                    accumulator,
                )
                mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(
                    krel_mbars + chunk,
                    krel_peer_mask,
                    tcgen05.CtaGroup.TWO,
                )

    @cute.jit
    def _issue_score_v2(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
        issue_seq: Int32,
        is_dp: cutlass.Constexpr[bool],
    ) -> pipeline.PipelineState:
        """Issue one score-side CG2 GEMM over four resident D128 chunks.

        v7: the accumulator ping-pongs between two TMEM bases selected by
        the producer stage index; the pipeline state itself never crosses
        the dynamic stage branch (only the pure-side-effect GEMM loop
        does).
        """

        done_pipeline.producer_acquire(producer_state)
        if producer_state.index == Int32(0):
            self._issue_score_chunks_v7(
                tiled_mma,
                accumulator_0,
                a_fragment,
                b_fragment,
            )
        else:
            self._issue_score_chunks_v7(
                tiled_mma,
                accumulator_1,
                a_fragment,
                b_fragment,
            )
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_chunks_v7(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
    ):
        """One full-K score GEMM into a single ping-pong accumulator."""

        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        # Baseline-shaped runtime issue loops (unroll=4, flat k-mode runtime
        # index only): the fully unrolled 32-atom body bloated the leader
        # warp past its register budget and tripled the per-atom issue cost.
        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            for k_block in cutlass.range(
                0,
                k_blocks_per_chunk,
                unroll=4,
            ):
                cute.gemm(
                    mma,
                    accumulator,
                    a_fragment[None, None, k_block, chunk],
                    b_fragment[None, None, k_block, chunk],
                    accumulator,
                )
                mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_score_v2_streamed(
        self,
        tiled_mma: cute.TiledMma,
        accumulator_0: cute.Tensor,
        accumulator_1: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
        kchunk_mbars: cute.Pointer,
        kchunk_phase: Int32,
    ) -> pipeline.PipelineState:
        """S with chunk-gated B (E1): consume chunk c on kchunk_mbars[c].

        Structure mirrors _issue_score_v2; the data gate moves from one
        whole-tile kscore consumer_wait (now taken later, before dP) to
        four per-chunk waits inside the issue loop, so S's chunk c
        overlaps the gather of chunks c+1 and onward.
        """

        done_pipeline.producer_acquire(producer_state)
        if producer_state.index == Int32(0):
            self._issue_score_chunks_streamed(
                tiled_mma,
                accumulator_0,
                a_fragment,
                b_fragment,
                kchunk_mbars,
                kchunk_phase,
            )
        else:
            self._issue_score_chunks_streamed(
                tiled_mma,
                accumulator_1,
                a_fragment,
                b_fragment,
                kchunk_mbars,
                kchunk_phase,
            )
        cute.arch.fence_view_async_tmem_store()
        done_pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def _issue_score_chunks_streamed(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        kchunk_mbars: cute.Pointer,
        kchunk_phase: Int32,
    ):
        """One full-K S GEMM, each chunk gated on its own landing."""

        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        for chunk in cutlass.range_constexpr(self.K_CHUNKS):
            _mbarrier_wait_acquire_cluster(
                kchunk_mbars + chunk,
                kchunk_phase,
            )
            for k_block in cutlass.range(
                0,
                k_blocks_per_chunk,
                unroll=4,
            ):
                cute.gemm(
                    mma,
                    accumulator,
                    a_fragment[None, None, k_block, chunk],
                    b_fragment[None, None, k_block, chunk],
                    accumulator,
                )
                mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_prev_grads_head_v2(
        self,
        dq_tiled_mma: cute.TiledMma,
        dkv_tiled_mma: cute.TiledMma,
        t_dq_0: cute.Tensor,
        t_dq_1: cute.Tensor,
        t_dkv_0: cute.Tensor,
        dq_kd_fragment_a: cute.Tensor,
        dq_kd_fragment_b: cute.Tensor,
        dq_ds_fragment: cute.Tensor,
        quad_fragment_a: cute.Tensor,
        quad_fragment_b: cute.Tensor,
        loan_quad_fragment_a: cute.Tensor,
        loan_quad_fragment_b: cute.Tensor,
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        dq_accumulate: cutlass.Boolean,
        relay_phase: Int32,
        relay_mbars: cute.Pointer,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        kscore_pipeline,
        kscore_consumer_state: pipeline.PipelineState,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_producer_state: pipeline.PipelineState,
        dq_done_pipeline,
        dq_done_state: pipeline.PipelineState,
        commit_dq: cutlass.Constexpr[bool],
        issue_seq: Int32,
    ):
        """Gradient block, first half: dQ rounds + round-0 dV/dK passes.

        The tail keeps v12's early dQ completion commit.  A separate
        loan_epi_safe barrier, published after the final kscore producer
        tail, protects score_kv from the dQ epilogue until both borrowed
        dV source reads have truly completed.
        """

        # P/dS images, blocks, and xchg staging of the previous tile are
        # published (and its DSM sends issued): invariant I1's only gate.
        pds_pipeline.consumer_wait(pds_consumer_state)

        # dQ both rounds back-to-back; releases free the round buffers for
        # the quadrant refills.
        round_consumer_state = self._issue_dq_rounds_v2(
            dq_tiled_mma,
            t_dq_0,
            t_dq_1,
            dq_kd_fragment_a,
            dq_kd_fragment_b,
            dq_ds_fragment,
            dq_accumulate,
            round_pipeline,
            round_consumer_state,
            issue_seq,
        )

        if cutlass.const_expr(commit_dq):
            dq_done_pipeline.producer_commit(dq_done_state)
            dq_done_state.advance()

        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)

        packed_issue = issue_seq * Int32(self.D_ROUNDS * 4)
        dkv_done_pipeline.producer_acquire(dkv_producer_state)
        kscore_pipeline.consumer_wait(kscore_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_0,
            loan_quad_fragment_a,
            p_fragment_0,
            False,
        )
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_0,
            loan_quad_fragment_b,
            p_fragment_1,
            True,
        )
        kscore_pipeline.consumer_release(kscore_consumer_state)
        kscore_consumer_state.advance()

        _mbarrier_wait_acquire_cluster(
            relay_mbars + 1,
            relay_phase,
        )
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_0,
            quad_fragment_a,
            ds_fragment_0,
            True,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_0,
            quad_fragment_b,
            ds_fragment_1,
            True,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_producer_state)
        dkv_producer_state.advance()

        return (
            round_consumer_state,
            kscore_consumer_state,
            dkv_producer_state,
            pds_consumer_state,
            dq_done_state,
        )

    @cute.jit
    def _issue_prev_grads_tail_v2(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv_1: cute.Tensor,
        quad_fragment_a: cute.Tensor,
        quad_fragment_b: cute.Tensor,
        borrow_fragment: cute.Tensor,
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_producer_state: pipeline.PipelineState,
        kchunk_mbars: cute.Pointer,
        krel_peer_mask: Int32,
        issue_seq: Int32,
    ):
        """Gradient block, second half: round-1 dV/dK passes.

        v8: per-slot generations -- this half acquires slot 1's own
        generation and commits it at the end.
        """

        packed_issue = (
            issue_seq * Int32(self.D_ROUNDS * 4) + Int32(4)
        )
        dkv_done_pipeline.producer_acquire(dkv_producer_state)
        # E4b: g4's pipe wait doubles as the both-CTA landing signal of
        # the borrowed dO_r1 h0 (W19 bridged its mbar into the commit).
        # The release hoists AHEAD of the pass: the ring slot's bytes
        # are untouched by the borrowed read, so W17 may refill Q_r1 h0
        # one consumer earlier -- the point of the loan.
        round_pipeline.consumer_wait(round_consumer_state)
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_1,
            borrow_fragment,
            p_fragment_0,
            False,
        )
        # Retire the borrowed parcel: gather's session B (K(t+1)
        # chunks 2-3) waits this commit before overwriting 16-32K.
        with cute.arch.elect_one():
            tcgen05.commit(
                kchunk_mbars + 3,
                krel_peer_mask,
                tcgen05.CtaGroup.TWO,
            )
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_1,
            quad_fragment_b,
            p_fragment_1,
            True,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_1,
            quad_fragment_a,
            ds_fragment_0,
            True,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_1,
            quad_fragment_b,
            ds_fragment_1,
            True,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        cute.arch.fence_view_async_tmem_store()
        dkv_done_pipeline.producer_commit(dkv_producer_state)
        dkv_producer_state.advance()

        return round_consumer_state, dkv_producer_state

    @cute.kernel
    def convert_canonical(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
    ):
        """Decode the baseline reducer's within-panel column scramble.

        The v6 drain stores each thread's register-gathered FP32x4 quad at
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
                    dim_idx = (
                        tidx // 4
                        + tidx % 4 * 8
                        + j * 32
                        + i * 64
                    )
                    out_row[dim_idx] = self.element_dtype(scrambled)

    @cute.jit
    def _drain_dkv_v8(
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
        issue_seq: Int32,
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):
        """Drain both rank-owned dKV slots in one fused register pass.

        Per-slot mechanics are the v6 baseline-verbatim reducer (Ld16x256b
        Rep-4 T2R split across two warp groups, register-gathered FP32x4
        quads, preloaded KV indices, thread-group-addressed 16B red.global;
        column scramble decoded by convert_canonical).  v8 keeps the v7
        fused savings -- ONE shared KV-index preload, a fused back-to-back
        atomic section, no reduce_sync_barrier (each thread releases after
        its own fenced loads; the producer's acquire counts all 256
        arrivals) -- but returns to per-slot generations: slot 0 is waited,
        T2R'd, fenced, and released as soon as the grads HEAD commits, a
        full grads-tail before slot 1, restoring the v6 head start and the
        leader's acquire slack.  Split wait/release pipeline states allow
        both releases to trail their own fences (round_acq/round_com
        pattern).
        """

        packed_issue = issue_seq * Int32(self.D_ROUNDS)

        # --- slot 0: head-committed generation.
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()

        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        # Baseline slices the fragment-C TMEM tensor down to its atom core
        # before building the tmem copy (dsa_bwd_sm100.py L1903-1907); the
        # full-rank tensor makes the tiler rank exceed the 2-D identity.
        t_dkv_core_0 = t_dkv_0[(None, None), 0, 0]
        t_dkv_core_1 = t_dkv_1[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r_0 = tcgen05.make_tmem_copy(
            tmem_load_atom,
            t_dkv_core_0,
        )
        thread_t2r_0 = tiled_t2r_0.get_slice(dp_idx)
        tiled_t2r_1 = tcgen05.make_tmem_copy(
            tmem_load_atom,
            t_dkv_core_1,
        )
        thread_t2r_1 = tiled_t2r_1.get_slice(dp_idx)
        c_dkv = cute.make_identity_tensor(
            (self.D_TILE_CTA, self.N_TILE)
        )
        thread_coordinates = self.split_wg(
            thread_t2r_0.partition_D(c_dkv),
            2,
            wg_idx,
        )
        thread_source_0 = self.split_wg(
            thread_t2r_0.partition_S(t_dkv_core_0),
            2,
            wg_idx,
        )
        thread_source_1 = self.split_wg(
            thread_t2r_1.partition_S(t_dkv_core_1),
            2,
            wg_idx,
        )
        thread_values_0 = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )
        thread_values_1 = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )

        # Preload the per-thread KV indices as independent loads so they
        # overlap the T2R (baseline reducer pattern); shared by both slots.
        tile_base = tile_index * Int32(self.N_TILE)
        r_topk = cute.make_rmem_tensor((8,), cutlass.Int32)
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            local_row = Int32(
                cute.get(
                    thread_coordinates[coord_base],
                    mode=[1],
                )
            )
            global_row = tile_base + local_row
            if global_row < topk:
                r_topk[i] = mTopkIdxs[
                    global_row,
                    (token_idx, batch_idx),
                ]
            else:
                r_topk[i] = Int32(-1)

        cute.copy(tiled_t2r_0, thread_source_0, thread_values_0)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()

        assert cute.size(thread_values_0) == self.N_TILE // 2
        sub_tile_idx_0 = rank
        sub_tile_idx_1 = Int32(2) + rank
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            rdkv_frg_0 = cute.make_rmem_tensor(
                (4,),
                self.acc_dtype,
            )
            rdkv_frg_0[0] = thread_values_0[coord_base]
            rdkv_frg_0[1] = thread_values_0[coord_base + 2]
            rdkv_frg_0[2] = thread_values_0[coord_base + 16]
            rdkv_frg_0[3] = thread_values_0[coord_base + 18]

            kv_index = r_topk[i]
            if kv_index >= Int32(0):
                dkv_row = mdKV_acc[
                    None,
                    kv_index,
                    (0, batch_idx),
                ]
                tile_row = cute.flat_divide(dkv_row, (128,))
                tile_row_0 = tile_row[None, sub_tile_idx_0]
                tile_row_0 = cute.flat_divide(tile_row_0, (4,))
                target_frg_0 = tile_row_0[None, dp_idx // 4]
                cute.arch.atomic_add(
                    target_frg_0.iterator.llvm_ptr,
                    rdkv_frg_0.load(),
                )

        # --- slot 1: tail-committed generation.
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            rdkv_frg_1 = cute.make_rmem_tensor(
                (4,),
                self.acc_dtype,
            )
            rdkv_frg_1[0] = thread_values_1[coord_base]
            rdkv_frg_1[1] = thread_values_1[coord_base + 2]
            rdkv_frg_1[2] = thread_values_1[coord_base + 16]
            rdkv_frg_1[3] = thread_values_1[coord_base + 18]

            kv_index = r_topk[i]
            if kv_index >= Int32(0):
                dkv_row = mdKV_acc[
                    None,
                    kv_index,
                    (0, batch_idx),
                ]
                tile_row = cute.flat_divide(dkv_row, (128,))
                tile_row_1 = tile_row[None, sub_tile_idx_1]
                tile_row_1 = cute.flat_divide(tile_row_1, (4,))
                target_frg_1 = tile_row_1[None, dp_idx // 4]
                cute.arch.atomic_add(
                    target_frg_1.iterator.llvm_ptr,
                    rdkv_frg_1.load(),
                )
        return wait_state, release_state


FlashAttentionDSABackwardSm100TwoCTAV1 = (
    FlashAttentionDSABackwardSm100TwoCTAV2
)
FlashAttentionDSABackwardSm100TwoCTAV0 = (
    FlashAttentionDSABackwardSm100TwoCTAV2
)
