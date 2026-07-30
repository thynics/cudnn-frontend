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
        ("cp.async.bulk.shared::cluster.shared::cta." "mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];"),
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionDSABackwardSm100TwoCTA(FlashAttentionDSABackwardSm100):

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
    THREADS_PER_CTA = 256
    KV_LOAD_THREADS = 128
    KV_GROUP_SIZE = 8
    KV_NUM_GROUPS = KV_LOAD_THREADS // KV_GROUP_SIZE
    TMEM_COLUMNS = 512
    MAX_SMEM_BYTES = 232_448
    QUADRANT_ELEMENTS = H_TILE_CTA * N_TILE_CTA

    TMEM_S_OFFSET = 0
    TMEM_DP_OFFSET = 64
    TMEM_DKV0_OFFSET = 128
    TMEM_DKV1_OFFSET = 192
    TMEM_DQ0_OFFSET = 256
    TMEM_DQ1_OFFSET = 384

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
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.THREADS_PER_CTA,
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
        tma_atom_dq_epi, tma_tensor_dq_epi = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdQ_epi,
            dq_epi_layout,
            dq_epi_tile,
        )

        stationary_a_layout = cute.select(
            stationary_a_layout_staged,
            mode=[0, 1, 2],
        )
        score_a_layout = cute.select(
            score_a_layout_staged,
            mode=[0, 1, 2],
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
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
            tma_atom_q,
            tma_tensor_q,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_qt,
            tma_tensor_qt,
            tma_atom_dot,
            tma_tensor_dot,
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
            dkv_a_layout_staged,
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
            grad_a_stage_bytes,
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
        convert_grid_x = (mKV.shape[0] + self.block_seq - 1) // self.block_seq
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
            thread_source = thread_copy.partition_S(source_chunks[None, chunk_index])
            thread_destination = thread_copy.partition_D(destination_chunks[None, chunk_index])
            cute.copy(copy_atom, thread_source, thread_destination)

    @cute.jit
    def _zero_sparse_k_d128_row(
        self,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        index_in_group: Int32,
    ):

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
                    cute.make_layout((self.N_TILE_CTA, self.K_CHUNK)),
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

        if mtx < self.MATH_THREADS_PER_CTA:
            tiled_t2r = tcgen05.make_tmem_copy(dq_tmem_load, t_dq)
            thread_t2r = tiled_t2r.get_slice(mtx)
            thread_source = thread_t2r.partition_S(t_dq)
            thread_coordinates = thread_t2r.partition_D(rank_coordinates)
            thread_values = cute.make_rmem_tensor(
                thread_coordinates.shape,
                self.acc_dtype,
            )
            cute.copy(tiled_t2r, thread_source, thread_values)
            cute.arch.fence_view_async_tmem_load()
            for value_index in cutlass.range_constexpr(cute.size(thread_values)):
                d_in_round = Int32(cute.get(thread_coordinates[value_index], mode=[0]))
                head = Int32(cute.get(thread_coordinates[value_index], mode=[1]))
                local_d = d_in_round - rank * Int32(self.D_TILE_CTA)
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
        self.math_barrier.arrive_and_wait()


@dsl_user_op
def _mbarrier_wait_acquire_cluster(
    barrier: cute.Pointer,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> None:

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
def _store_shared_seq_v4(
    counter: cute.Pointer,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:

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


class FlashAttentionDSABackwardSm100TwoCTAV2(FlashAttentionDSABackwardSm100TwoCTA):

    THREADS_PER_CTA = 640

    GATHER_WARPS = 4
    MATH_WARP_BEGIN = 4
    MATH_WARPS = 4
    REDUCE_WARP_BEGIN = 8
    REDUCE_WARPS = 8
    MMA_WARP = 16
    LOAD_WARP = 17
    RELAY_WARP = 18

    GATHER_THREADS = GATHER_WARPS * 32
    MATH_THREAD_BEGIN = MATH_WARP_BEGIN * 32
    MATH_THREADS = MATH_WARPS * 32
    REDUCE_THREAD_BEGIN = REDUCE_WARP_BEGIN * 32
    REDUCE_THREADS = REDUCE_WARPS * 32

    DKV_MMA_TILER = (256, 64, 64)
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

    ROUND_STAGES = 2

    MMA_DONE_STAGES = 2

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

        return cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
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

        @cute.struct
        class SharedStorageV2:
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            kscore_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            pds_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dkv_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dq_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            landing_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            relay_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            khot_seq: cute.struct.MemRange[cutlass.Int64, 1]
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

        return SharedStorageV2

    @cute.jit
    def _kd_round_rows_v2(
        self,
        tensor: cute.Tensor,
    ) -> cute.Tensor:

        return cute.composition(
            tensor[None, None, None, 0],
            cute.make_layout(
                (self.N_TILE, self.D_TILE_CTA),
                stride=(self.D_TILE_CTA, 1),
            ),
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

        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset_0 = rank * Int32(self.D_TILE_CTA)
        d_offset_1 = Int32(self.D_TILE_CLUSTER) + rank * Int32(self.D_TILE_CTA)
        rows_per_group = self.N_TILE // groups_total
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_iteration * groups_total + group_index
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
    ) -> None:

        self.kdq_barrier.arrive_and_wait()
        self._fill_kdq_pair_v8(
            mKV,
            mTopkIdxs,
            kd_rows_0,
            kd_rows_1,
            token_idx,
            batch_idx,
            tile_index,
            topk,
            rank,
            role_tidx,
            self.GATHER_THREADS,
            copy_atom,
            thread_copy,
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        self.kdq_barrier.arrive_and_wait()

    @cute.jit
    def _issue_dkv_pass_v2(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        accumulate: cutlass.Constexpr[bool],
    ) -> None:

        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        for k_block in cutlass.range_constexpr(cute.size(a_fragment, mode=[2])):
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
    ) -> pipeline.PipelineState:

        for round_index in cutlass.range_constexpr(self.D_ROUNDS):
            round_pipeline.consumer_wait(round_consumer_state)
            mma = dq_tiled_mma.with_()
            mma.set(tcgen05.Field.ACCUMULATE, accumulate)
            if cutlass.const_expr(round_index == 0):
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_a, mode=[2])):
                    cute.gemm(
                        mma,
                        t_dq_0,
                        kd_fragment_a[None, None, k_block, 0],
                        ds_fragment[None, None, k_block, 0],
                        t_dq_0,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for k_block in cutlass.range_constexpr(cute.size(kd_fragment_b, mode=[2])):
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

        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            linear_index = tidx
            while linear_index < cute.size(rank_coordinates):
                coordinate = cute.idx2crd(
                    linear_index,
                    rank_coordinates.shape,
                )
                logical_coordinate = rank_coordinates[coordinate]
                d_in_round = Int32(cute.get(logical_coordinate, mode=[0]))
                head = Int32(cute.get(logical_coordinate, mode=[1]))
                mdQ[
                    Int32(round_index * self.D_TILE_CLUSTER) + d_in_round,
                    head,
                    (token_idx, batch_idx),
                ] = self.element_dtype(0.0)
                linear_index += Int32(self.MATH_THREADS_PER_CTA)

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_qt: cute.CopyAtom,
        tma_tensor_qt: cute.Tensor,
        tma_atom_dot: cute.CopyAtom,
        tma_tensor_dot: cute.Tensor,
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
        dkv_a_layout_staged: cute.ComposedLayout,
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
        grad_a_stage_bytes: cutlass.Constexpr[int],
        stationary_tiled_mma: cute.TiledMma,
        stationary_a_layout_staged: cute.ComposedLayout,
    ):

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
        khot_seq = cute.recast_ptr(
            storage.khot_seq.data_ptr(),
            dtype=cutlass.Int32,
        )
        stationary_q_raw = storage.stationary_q.data_ptr()
        stationary_do_raw = storage.stationary_do.data_ptr()
        round_buf_a_raw = storage.round_buf_a.data_ptr()
        round_buf_b_raw = storage.round_buf_b.data_ptr()

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
        score_store_layout = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.H_TILE_CTA, self.N_TILE),
            1,
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
        ds_image_store = storage.ds_image.get_tensor(
            score_store_domain,
            swizzle=score_store_layout.inner,
        )
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
        p_xchg_raw = storage.p_xchg.get_tensor(flat_pds_block_layout)
        softmax_stats = storage.stats.get_tensor(
            cute.make_layout(
                (self.H_TILE_CTA, 2),
                stride=(1, self.H_TILE_CTA),
            )
        )

        stats_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=64,
        )
        stats_tiled_copy = cute.make_tiled_copy_tv(
            stats_copy_atom,
            cute.make_layout((32,), stride=(1,)),
            cute.make_layout((2,), stride=(1,)),
        )
        stats_thread_copy = stats_tiled_copy.get_slice(tidx % Int32(32))
        g_scaled_lse = cute.flat_divide(
            scaled_lse,
            (self.H_TILE_CTA,),
        )
        g_sum_odo = cute.flat_divide(
            sum_odo,
            (self.H_TILE_CTA,),
        )
        t_g_scaled_lse = stats_thread_copy.partition_S(g_scaled_lse[None, rank, (token_idx, batch_idx)])
        t_s_scaled_lse = stats_thread_copy.partition_D(softmax_stats[None, 0])
        t_g_sum_odo = stats_thread_copy.partition_S(g_sum_odo[None, rank, (token_idx, batch_idx)])
        t_s_sum_odo = stats_thread_copy.partition_D(softmax_stats[None, 1])

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
        rank_score_coordinates = rank_score_mma.partition_C(cute.make_identity_tensor((self.H_TILE_CLUSTER, self.N_TILE)))
        rank_dq_coordinates = rank_dq_mma.partition_C(cute.make_identity_tensor(self.DQ_MMA_TILER[:2]))

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

        score_q_fragment = score_tiled_mma.make_fragment_A(stationary_q)
        score_do_fragment = dp_tiled_mma.make_fragment_A(stationary_do)
        score_k_fragment = score_tiled_mma.make_fragment_B(k_n)
        dp_k_fragment = dp_tiled_mma.make_fragment_B(k_n)
        dq_kd_fragment_a = dq_tiled_mma.make_fragment_A(round_kd[0])
        dq_kd_fragment_b = dq_tiled_mma.make_fragment_A(round_kd[1])
        dq_ds_fragment = dq_tiled_mma.make_fragment_B(ds_image)
        quad_fragment_a = dkv_tiled_mma.make_fragment_A(round_quad[0])
        quad_fragment_b = dkv_tiled_mma.make_fragment_A(round_quad[1])
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

        score_c_layout = score_tiled_mma.make_fragment_C(score_tiled_mma.partition_shape_C((self.H_TILE_CLUSTER, self.N_TILE))).layout
        dkv_c_layout = dkv_tiled_mma.make_fragment_C(dkv_tiled_mma.partition_shape_C(self.DKV_MMA_TILER[:2])).layout
        dq_c_layout = dq_tiled_mma.make_fragment_C(dq_tiled_mma.partition_shape_C(self.DQ_MMA_TILER[:2])).layout
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
        tile_count = (topk + Int32(self.N_TILE - 1)) // Int32(self.N_TILE)

        if warp_idx < Int32(self.MATH_WARP_BEGIN) or warp_idx >= Int32(self.MMA_WARP):
            cute.arch.setmaxregister_decrease(48)
        else:
            if warp_idx < Int32(self.REDUCE_WARP_BEGIN):
                cute.arch.setmaxregister_increase(128)
            else:
                cute.arch.setmaxregister_increase(128)

        if warp_idx < Int32(self.GATHER_WARPS):
            gather_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            gather_kd_rows_0 = self._kd_round_rows_v2(round_kd[0])
            gather_kd_rows_1 = self._kd_round_rows_v2(round_kd[1])
            if tile_count > Int32(0):
                pipe_kscore.producer_acquire(gather_state)
                self._load_score_kv(
                    mKV,
                    mTopkIdxs,
                    k_n,
                    token_idx,
                    batch_idx,
                    tile_count - Int32(1),
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
                for loop_iter in cutlass.range(tile_count - Int32(1)):
                    next_iter = loop_iter + Int32(1)
                    pipe_kscore.producer_acquire(gather_state)
                    self._load_score_kv(
                        mKV,
                        mTopkIdxs,
                        k_n,
                        token_idx,
                        batch_idx,
                        tile_count - Int32(1) - next_iter,
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
                    self._gather_kdq_v8(
                        mKV,
                        mTopkIdxs,
                        gather_kd_rows_0,
                        gather_kd_rows_1,
                        token_idx,
                        batch_idx,
                        tile_count - Int32(1) - loop_iter,
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
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
                )
                pipe_kscore.producer_tail(gather_state)

        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
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
            score_coordinates = score_thread.partition_D(rank_score_coordinates)
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
            score_source_pp = score_copy_pp.get_slice(mtx).partition_S(t_score_pp)
            dp_copy_pp = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_dp_pp,
            )
            dp_source_pp = dp_copy_pp.get_slice(mtx).partition_S(t_dp_pp)
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
            thread_copy_r2s = tiled_copy_r2s.get_slice(mtx)
            t_rs_ds = thread_copy_r2s.partition_D(ds_image_store)
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
                p_xchg_raw.iterator.toint() - n_owner * Int32(self.PDS_BLOCK_BYTES),
                p_xchg_raw.memspace,
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
            t_rs_p_local = thread_copy_r2s.partition_D(p_local_store)
            t_rs_ds_local = thread_copy_r2s.partition_D(ds_local_store)
            t_rs_p_xchg = thread_copy_r2s.partition_D(p_xchg_store)
            t_rs_p_local_tile = t_rs_p_local[None, None, None, None, 0]
            t_rs_ds_local_tile = t_rs_ds_local[None, None, None, None, 0]
            t_rs_p_xchg_tile = t_rs_p_xchg[None, None, None, None, 0]
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

                softmax_scale_log2_e = scale_softmax * Float32(math.log2(math.e))
                for h_group in cutlass.range_constexpr(4):
                    group_base = 2 * (h_group % 2) + 16 * (h_group // 2)
                    local_h = Int32(
                        cute.get(
                            score_coordinates[group_base],
                            mode=[0],
                        )
                    ) % Int32(self.H_TILE_CTA)
                    lse = softmax_stats[local_h, 0]
                    delta = softmax_stats[local_h, 1]
                    for j in cutlass.range_constexpr(8):
                        value_index = group_base + (j % 2) + 4 * (j // 2)
                        p_value = cute.math.exp2(
                            (r_score[value_index] * softmax_scale_log2_e + lse),
                            fastmath=True,
                        )
                        ds_value = (r_dp[value_index] + delta) * p_value * scale_softmax
                        r_p[value_index] = self.element_dtype(p_value)
                        r_ds[value_index] = self.element_dtype(ds_value)

                pipe_pds.producer_acquire(pds_state)

                r_p_store = thread_copy_r2s.retile(r_p)
                r_ds_store = thread_copy_r2s.retile(r_ds)
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

                cute.copy(
                    tiled_copy_r2s,
                    r_ds_store,
                    t_rs_ds_tile,
                )

                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(
                    pds_ready_mbars,
                    rank,
                )
                pds_state.advance()

            if tile_count > Int32(0):
                pipe_dq_done.consumer_wait(dq_done_state)
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
                    pipe_dkv_done,
                    dkv_wait,
                    dkv_rel,
                )

        elif warp_idx == Int32(self.MMA_WARP):
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
                for loop_iter in cutlass.range(tile_count):
                    has_prev = loop_iter > Int32(0)

                    pipe_kscore.consumer_wait(kscore_cons)
                    s_prod = self._issue_score_v2(
                        score_tiled_mma,
                        t_score,
                        t_score_pp,
                        score_q_fragment,
                        score_k_fragment,
                        pipe_s_done,
                        s_prod,
                    )

                    if loop_iter == Int32(0):
                        _mbarrier_wait_acquire_cluster(
                            stationary_ready_mbar + 1,
                            Int32(0),
                        )
                    dp_prod = self._issue_score_v2(
                        dp_tiled_mma,
                        t_dp,
                        t_dp_pp,
                        score_do_fragment,
                        dp_k_fragment,
                        pipe_dp_done,
                        dp_prod,
                    )
                    pipe_kscore.consumer_release(kscore_cons)
                    kscore_cons.advance()

                    if has_prev:
                        dq_acc = loop_iter != Int32(1)
                        round_cons, dkv_prod, pds_cons, dq_done_prod = self._issue_prev_grads_head_v2(
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
                            p_fragments[0],
                            p_fragments[1],
                            ds_fragments[0],
                            ds_fragments[1],
                            dq_acc,
                            relay_phase,
                            relay_mbars,
                            pipe_round,
                            round_cons,
                            pipe_pds,
                            pds_cons,
                            pipe_dkv_done,
                            dkv_prod,
                            pipe_dq_done,
                            dq_done_prod,
                            False,
                        )
                        round_cons, dkv_prod = self._issue_prev_grads_tail_v2(
                            dkv_tiled_mma,
                            t_dkv[1],
                            quad_fragment_a,
                            quad_fragment_b,
                            p_fragments[0],
                            p_fragments[1],
                            ds_fragments[0],
                            ds_fragments[1],
                            pipe_round,
                            round_cons,
                            pipe_dkv_done,
                            dkv_prod,
                        )
                        pipe_pds.consumer_release(pds_cons)
                        pds_cons.advance()
                        relay_phase = Int32(1) - relay_phase

                if tile_count > Int32(0):
                    dq_acc = tile_count != Int32(1)
                    round_cons, dkv_prod, pds_cons, dq_done_prod = self._issue_prev_grads_head_v2(
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
                        p_fragments[0],
                        p_fragments[1],
                        ds_fragments[0],
                        ds_fragments[1],
                        dq_acc,
                        relay_phase,
                        relay_mbars,
                        pipe_round,
                        round_cons,
                        pipe_pds,
                        pds_cons,
                        pipe_dkv_done,
                        dkv_prod,
                        pipe_dq_done,
                        dq_done_prod,
                        True,
                    )
                    round_cons, dkv_prod = self._issue_prev_grads_tail_v2(
                        dkv_tiled_mma,
                        t_dkv[1],
                        quad_fragment_a,
                        quad_fragment_b,
                        p_fragments[0],
                        p_fragments[1],
                        ds_fragments[0],
                        ds_fragments[1],
                        pipe_round,
                        round_cons,
                        pipe_dkv_done,
                        dkv_prod,
                    )
                    pipe_pds.consumer_release(pds_cons)
                    pds_cons.advance()

                    pipe_s_done.producer_tail(s_prod)
                    pipe_dp_done.producer_tail(dp_prod)
                    pipe_dkv_done.producer_tail(dkv_prod)
                    pipe_dq_done.producer_tail(dq_done_prod)

        elif warp_idx == Int32(self.LOAD_WARP):
            round_acq = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            round_com = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            tma_phase_0 = Int32(0)
            tma_phase_1 = Int32(0)
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
                    tile_index = tile_count - Int32(1) - loop_iter
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    self.kdq_barrier.arrive_and_wait()
                    self.kdq_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()

                    for flat_gen in cutlass.range_constexpr(8):
                        grad_round = flat_gen // 4
                        tensor_kind = (flat_gen // 2) % 2
                        h_half = flat_gen % 2
                        pipe_round.producer_acquire(round_acq)
                        round_acq.advance()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                round_tma_mbars + h_half,
                                grad_a_stage_bytes,
                            )
                        if cutlass.const_expr(tensor_kind == 0):
                            if cutlass.const_expr(h_half == 0):
                                if rank == Int32(0):
                                    with cute.arch.elect_one():
                                        _cpasync_bulk_s2cluster(
                                            stationary_do_raw + 4096 * (4 * grad_round),
                                            round_buf_a_raw,
                                            round_tma_mbars,
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
                                        t_dot_smem_a[None, 0],
                                        tma_bar_ptr=round_tma_mbars,
                                    )
                            else:
                                if rank == Int32(1):
                                    with cute.arch.elect_one():
                                        _cpasync_bulk_s2cluster(
                                            stationary_do_raw + 4096 * (4 * grad_round + 2),
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
                            if cutlass.const_expr(h_half == 0):
                                if rank == Int32(0):
                                    with cute.arch.elect_one():
                                        _cpasync_bulk_s2cluster(
                                            stationary_q_raw + 4096 * (4 * grad_round),
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
                                if rank == Int32(1):
                                    with cute.arch.elect_one():
                                        _cpasync_bulk_s2cluster(
                                            stationary_q_raw + 4096 * (4 * grad_round + 2),
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
                        if cutlass.const_expr(flat_gen > 0):
                            if cutlass.const_expr((flat_gen - 1) % 2 == 0):
                                cute.arch.mbarrier_wait(
                                    round_tma_mbars,
                                    tma_phase_0,
                                )
                                tma_phase_0 = Int32(1) - tma_phase_0
                            else:
                                cute.arch.mbarrier_wait(
                                    round_tma_mbars + 1,
                                    tma_phase_1,
                                )
                                tma_phase_1 = Int32(1) - tma_phase_1
                            with cute.arch.elect_one():
                                pipe_round.producer_commit(round_com)
                            round_com.advance()

                    cute.arch.mbarrier_wait(
                        round_tma_mbars + 1,
                        tma_phase_1,
                    )
                    tma_phase_1 = Int32(1) - tma_phase_1
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
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

        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)

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
    ) -> pipeline.PipelineState:

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
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        dq_accumulate: cutlass.Boolean,
        relay_phase: Int32,
        relay_mbars: cute.Pointer,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_producer_state: pipeline.PipelineState,
        dq_done_pipeline,
        dq_done_state: pipeline.PipelineState,
        commit_dq: cutlass.Constexpr[bool],
    ):

        pds_pipeline.consumer_wait(pds_consumer_state)

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
        )

        if cutlass.const_expr(commit_dq):
            dq_done_pipeline.producer_commit(dq_done_state)
            dq_done_state.advance()

        _mbarrier_wait_acquire_cluster(relay_mbars, relay_phase)

        dkv_done_pipeline.producer_acquire(dkv_producer_state)
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_0,
            quad_fragment_a,
            p_fragment_0,
            False,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_0,
            quad_fragment_b,
            p_fragment_1,
            True,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
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
        p_fragment_0: cute.Tensor,
        p_fragment_1: cute.Tensor,
        ds_fragment_0: cute.Tensor,
        ds_fragment_1: cute.Tensor,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_producer_state: pipeline.PipelineState,
    ):

        dkv_done_pipeline.producer_acquire(dkv_producer_state)
        round_pipeline.consumer_wait(round_consumer_state)
        self._issue_dkv_pass_v2(
            dkv_tiled_mma,
            t_dkv_1,
            quad_fragment_a,
            p_fragment_0,
            False,
        )
        round_pipeline.consumer_release(round_consumer_state)
        round_consumer_state.advance()
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
        done_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):

        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()

        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
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
        c_dkv = cute.make_identity_tensor((self.D_TILE_CTA, self.N_TILE))
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
