"""Pipelined SM100 two-CTA DSA backward, v3.2 "T3-64" score-transposed form.

This module materializes the v3.2 (T3-64) execution contract for BF16
GQA128/D512 per the T3 design doc (main form) + T32_BUILD_ADDENDUM
(dq_b rank-symmetric dual sub-image, errata) + the transposed-orientation
final ruling (five-GEMM table).  Per kv-bundle of 128 gathered tokens
(16 bundles), all GEMMs CG2, all-f32 accumulators, bf16 operands:

* G1 St_c = K . (Q_c)^T and G2 dPt_c = V . (dO_c)^T, (M,N,K)=(128,64,512),
  two h-chunks ping-pong; A = K gather own-kv64 x D512 chased in 2x8KB
  D-pieces; B = resident Q/dO panels (N-split views, 32 heads/CTA/chunk).
* math: Pt_c = exp(s*St_c - lse[h]); dSt_c = Pt_c * (dPt_c - Delta[h]) * s,
  constants indexed by the COLUMN (head) axis, warp-uniform loads; publish
  bf16 via stmatrix into the P slab and dS slab [own-kv64 x H128] (16,384B
  each), plus the dq_b own-half second image and the peer-half push
  (8KB bulk DSM fallback; st.async direct write is a V32-TODO upgrade).
* G3 dV += Pt . dO-slab / G4 dK += dSt . Q-slab: (kv128, D_c=128,
  h64-chunk accumulate); B slabs are GMEM/L2 TMA natural [H x D] layout,
  2-deep supply ring, 12 gens/bundle (4 dO + 4 Q + 4 kdq, 16KB each);
  dKV TMEM double slot (dV 64 cols + dK 64 cols), drained per completed
  [own-kv64 x 128] block by the reducer warps (T2R + f32 GMEM atomics).
* G5 dQt += Kt . dSt: (256,128,kv128) as two K=kv64 waves, B = dq_b base
  + h*8192 window; A = kdq stream; dQt persistent in 256 TMEM columns,
  drained once per query tile through the staged dQ epilogue.

SMEM account (addendum section 2): K chase 16,384 + Q panel 65,536 +
dO panel 65,536 + P slab 16,384 + dS slab 16,384 + supply ring 32,768 +
dq_b 16,384 + stats/mbar 2,048 = 231,424 of 232,448.  TMEM 512 columns:
dQ [0,256) + S pp [256,320) + dP pp [320,384) + dV [384,448) + dK [448,512).

Leader is dr-major; chase pieces for bundle t+1 are pinned at the
G3(c1,0)(t) wait point; G5 waves gate on mb_dqb[h] cluster gates; per
errata #1, mb_dqb_free[h] hangs on the LAST consuming wave (both D-rounds).

It is intentionally not wired into the public interface until the remaining
runtime control plane closes.  This self-contained module includes the
common two-CTA host/layout base and the v3.2 implementation.
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


class _IketProxy:
    """Forward to real IKET when loaded; otherwise make annotations no-ops."""

    @staticmethod
    def _api():
        experimental = getattr(cute, "experimental", None)
        return getattr(experimental, "iket", None)

    @classmethod
    def mark(cls, *args):
        api = cls._api()
        return None if api is None else api.mark(*args)

    @classmethod
    def range_start(cls, *args):
        api = cls._api()
        return None if api is None else api.range_start(*args)

    @classmethod
    def range_end(cls, *args):
        api = cls._api()
        return None if api is None else api.range_end(*args)


_iket = _IketProxy()


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

    # Score/gradient operand contract (v0/v17a values; the v3.2
    # score-transposed class overrides every constant in this block):
    #   score A = stationary panel view (K_CHUNKS chunk windows), score B
    #   = the gathered-K ring; dkv A = streamed [D,H] quadrants (MN-major)
    #   and dkv B = local P/dS blocks (K-major); the dq epilogue stores
    #   the natural [H,D] tile.
    SCORE_A_IS_STATIONARY = True
    SCORE_A_STAGES = K_CHUNKS
    SCORE_B_STAGES = K_CHUNKS
    SCORE_A_MAX_ELEMENTS = 32768
    SCORE_B_MAX_ELEMENTS = 16384
    # v5 tiling4 hook: the score-plane MMA N (host score_tiler mode 1).
    # Every pre-v5 schedule used N_TILE (64) here; the v5 head-outer
    # sub-tile class overrides it to the h32 sub-tile width.
    SCORE_MMA_N = N_TILE
    # v5 tiling4 hook: the tiler used to derive the streamed dkv-B gen
    # SMEM layout (and, on the GRAD_STREAM_IS_B path, its TMA atoms).
    # Identical to DKV_MMA_TILER for every pre-v5 schedule; the v5
    # class narrows the K mode to one h16 box.
    DKV_B_TILER = DKV_MMA_TILER
    # v5.2 hook: the dQ-eviction MMA tiler (None = no eviction; the
    # v5.2 class sets (128, 32, 64) and evicts per-(t, r) blocks).
    DQ_EVICT_TILER = None
    STATIONARY_TILE_H = H_TILE_CTA
    STATIONARY_STAGES = 1
    DKV_A_MAJOR = OperandMajorMode.MN
    DKV_B_MAJOR = OperandMajorMode.K
    DKV_A_STAGES = 1
    DKV_B_STAGES = 1
    DKV_A_MAX_ELEMENTS = 16384
    DKV_B_MAX_ELEMENTS = 4096
    DQ_B_STAGES = 1
    DQ_B_MAX_ELEMENTS = 4096
    # False: gradient Q/dO stream loads as dkv-A quadrants from the
    # transposed mQT/mdOT views.  True (v3.2): the stream is the dkv-B
    # operand, TMA'd from the GMEM-natural [H,D] tensors.
    GRAD_STREAM_IS_B = False
    # True (v3.2): the persistent accumulator is dQ^T, the epilogue tile
    # is [D,H] over a D-innermost view of mdQ.
    DQ_EPI_TRANSPOSED = False

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

    def _carve_dq_acc(
        self,
        workspace_LSE_OdO,
        problem_shape,
        total_seqlen_Q,
    ):
        """v5.2 hook: the f32 dQ eviction partial-sum tensor.

        None for every schedule without dQ eviction; the v5.2 class
        carves it from the (extended) LSE/OdO workspace tail.
        """

        return None

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
        # v3.2 (DQ_EPI_TRANSPOSED): the persistent accumulator is dQ^T,
        # so the epilogue view is [D,H,(token,batch)] with the GMEM-
        # contiguous D axis innermost (first box dim).
        if cutlass.const_expr(self.DQ_EPI_TRANSPOSED):
            mdQ_epi = cute.make_tensor(
                mdQ.iterator,
                cute.make_layout(
                    (
                        self.D_HEAD,
                        self.H_TILE_CLUSTER,
                        mdQ.shape[2],
                    ),
                    stride=(
                        mdQ.stride[0],
                        mdQ.stride[1],
                        mdQ.stride[2],
                    ),
                ),
            )
        else:
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
        # v3.2: the stationary panel is TMA'd as STATIONARY_STAGES boxes
        # of STATIONARY_TILE_H rows (2 x 32 for the transposed class: the
        # CTA's N-half heads of h-chunk c, H[c*64+rank*32 : +32), land in
        # panel stage c), keeping every round-gen chunk window a single
        # contiguous box.  v0 keeps the one 64-row box.
        if cutlass.const_expr(self.SCORE_A_IS_STATIONARY):
            # v0/v17a: the panel is the score-A operand; the helper MMA
            # carries the box on its M-mode (64, legal).
            stationary_tiler = (
                self.STATIONARY_TILE_H,
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
        else:
            # v3.2: the panel is the (zero-copy) score-B operand and its
            # box is 32 rows.  MmaF16BF16Op forbids M=32, so the helper
            # MMA carries the box on its N-mode instead ((M=64, N=32) is
            # legal; the M-mode is a dummy -- only the B fraction of
            # this MMA is ever used, for the panel SMEM layout, the Q/dO
            # TMA atoms, and the kernel-side partition_B).
            stationary_tiler = (
                self.H_TILE_CTA,
                self.STATIONARY_TILE_H,
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
        # v5: the score-plane N is a class hook (N_TILE for the pre-v5
        # schedules, the h32 sub-tile width for the head-outer class).
        score_tiler = (self.H_TILE_CLUSTER, self.SCORE_MMA_N, self.K_CHUNK)
        dkv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            self.DKV_A_MAJOR,
            self.DKV_B_MAJOR,
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
        # v5.2: the dQ-eviction MMA ((128, 32) CG2) and its A layout.
        # The 2-stage (128,32,64) A layout is byte-identical to the
        # FROZEN kdq gen bytes ([own-D128 x kv64] under the legacy
        # (256,128,64) single-stage layout): make_smem_layout_a's
        # MN-major order is (2,1,3) -- rest_k fastest -- so the legacy
        # m-half stride (4096 elements) IS the new layout's stage
        # stride; stage == d_half.  Asserted below (cosize + swizzle).
        if cutlass.const_expr(self.DQ_EVICT_TILER is not None):
            dq_evict_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype,
                self.element_dtype,
                OperandMajorMode.MN,
                OperandMajorMode.MN,
                self.acc_dtype,
                cg2,
                self.DQ_EVICT_TILER[:2],
            )
            dq_a_evict_layout_staged = sm100_utils.make_smem_layout_a(
                dq_evict_tiled_mma,
                self.DQ_EVICT_TILER,
                self.element_dtype,
                2,
            )
        else:
            dq_evict_tiled_mma = dq_tiled_mma
            dq_a_evict_layout_staged = sm100_utils.make_smem_layout_a(
                dq_tiled_mma,
                self.DQ_MMA_TILER,
                self.element_dtype,
                1,
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

        # v3.2 staging semantics: score A stages = chase ring slots (2),
        # score B stages = D64 piece windows over the stationary panel
        # (8, zero-copy); dkv A stages = h-chunk sub-images of the P/dS
        # slab (2); dq B stages = kv-wave sub-images of dq_b (2).
        score_a_layout_staged = sm100_utils.make_smem_layout_a(
            score_tiled_mma,
            score_tiler,
            self.element_dtype,
            self.SCORE_A_STAGES,
        )
        if cutlass.const_expr(self.SCORE_A_IS_STATIONARY):
            stationary_a_layout_staged = sm100_utils.make_smem_layout_a(
                stationary_tiled_mma,
                stationary_tiler,
                self.element_dtype,
                self.STATIONARY_STAGES,
            )
        else:
            # v3.2: B-operand derivation of the same [32 x D512] K-major
            # SW128B box (the panel IS score-B; see the helper-MMA note).
            stationary_a_layout_staged = sm100_utils.make_smem_layout_b(
                stationary_tiled_mma,
                stationary_tiler,
                self.element_dtype,
                self.STATIONARY_STAGES,
            )
        score_b_layout_staged = sm100_utils.make_smem_layout_b(
            score_tiled_mma,
            score_tiler,
            self.element_dtype,
            self.SCORE_B_STAGES,
        )
        dkv_a_layout_staged = sm100_utils.make_smem_layout_a(
            dkv_tiled_mma,
            self.DKV_MMA_TILER,
            self.element_dtype,
            self.DKV_A_STAGES,
        )
        # v5: the dkv-B gen layout derives from the DKV_B_TILER hook
        # (== DKV_MMA_TILER pre-v5; K narrowed to one h16 box for the
        # head-outer class, stage = box).
        dkv_b_layout_staged = sm100_utils.make_smem_layout_b(
            dkv_tiled_mma,
            self.DKV_B_TILER,
            self.element_dtype,
            self.DKV_B_STAGES,
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
            self.DQ_B_STAGES,
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
        assert cute.cosize(score_a_layout_staged) <= self.SCORE_A_MAX_ELEMENTS
        if cutlass.const_expr(self.SCORE_A_IS_STATIONARY):
            # v0: score A is a zero-copy chunk view of the stationary
            # panel, so the panel layout must be byte- and swizzle-
            # compatible with the score-A staged layout.
            assert cute.cosize(stationary_a_layout_staged) == cute.cosize(
                score_a_layout_staged
            )
            assert (
                stationary_a_layout_staged.inner
                == score_a_layout_staged.inner
            )
        else:
            # v3.2: score B is the zero-copy panel view instead; the
            # swizzle-atom compatibility obligation moves to score B.
            # The atom identity is checked here (compile-time, static
            # layouts); the intra-stage block ORDER identity (panel
            # [32 x 512] == 8 contiguous [32 x 64] score-B stages) is
            # what the correctness gate adjudicates on hardware.
            assert (
                stationary_a_layout_staged.inner
                == score_b_layout_staged.inner
            ), (
                str(stationary_a_layout_staged.inner),
                str(score_b_layout_staged.inner),
            )
            assert cute.cosize(stationary_a_layout_staged) >= cute.cosize(
                score_b_layout_staged
            )
        assert cute.cosize(score_b_layout_staged) <= self.SCORE_B_MAX_ELEMENTS
        assert cute.cosize(dkv_a_layout_staged) <= self.DKV_A_MAX_ELEMENTS
        assert cute.cosize(dkv_b_layout_staged) <= self.DKV_B_MAX_ELEMENTS
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= self.DQ_B_MAX_ELEMENTS
        if cutlass.const_expr(self.DQ_EVICT_TILER is not None):
            # v5.2 byte-identity gates for the eviction A view.
            assert cute.cosize(dq_a_evict_layout_staged) == cute.cosize(
                dq_a_layout_staged
            ), (
                cute.cosize(dq_a_evict_layout_staged),
                cute.cosize(dq_a_layout_staged),
            )
            assert (
                dq_a_evict_layout_staged.inner
                == dq_a_layout_staged.inner
            ), (
                str(dq_a_evict_layout_staged.inner),
                str(dq_a_layout_staged.inner),
            )
            # v5.2 TMEM budget (change order Z1), echo on failure.
            assert self.TMEM_BUDGET <= self.TMEM_COLUMNS, (
                self.TMEM_BUDGET
            )
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
        if cutlass.const_expr(self.SCORE_A_IS_STATIONARY):
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
        else:
            # v3.2: the B gmem convention is (N, K) = (H rows, D), which
            # is exactly the natural mQ/mdO view -- no re-view needed.
            tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mQ,
                stationary_a_layout,
                stationary_tiler,
                stationary_tiled_mma,
            )
            tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_B(
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
        if cutlass.const_expr(self.GRAD_STREAM_IS_B):
            # v3.2: the streamed gradient operand is the dkv-B slab
            # [K=h64-chunk x N=own-D64], TMA'd from the transposed
            # [D,H] views (the B gmem convention is (N,K), and N is the
            # D axis here, GMEM-contiguous).  Stage index = h-chunk, so
            # one gen carries both chunk windows of one D-round.  The
            # atom names keep their v17a identities (qt/dot) so the
            # supply-loop plumbing stays recognizable.
            grad_a_layout = cute.select(
                dkv_b_layout_staged,
                mode=[0, 1, 2],
            )
            tma_atom_qt, tma_tensor_qt = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mQT,
                grad_a_layout,
                self.DKV_B_TILER,
                dkv_tiled_mma,
                cluster_layout_vmnk.shape,
            )
            tma_atom_dot, tma_tensor_dot = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mdOT,
                grad_a_layout,
                self.DKV_B_TILER,
                dkv_tiled_mma,
                cluster_layout_vmnk.shape,
            )
        else:
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
        # v5.2: the f32 dQ eviction partial-sum tensor (None for every
        # schedule without eviction).
        mdQ_acc = self._carve_dq_acc(
            workspace_LSE_OdO,
            problem_shape,
            mQ.shape[2][0],
        )

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
            dq_evict_tiled_mma,
            dq_a_evict_layout_staged,
            mdQ_acc,
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
    def _copy_sparse_k_row_v32(
        self,
        mKV: cute.Tensor,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        kv_index: Int32,
        batch_idx: Int32,
        d_offset: Int32,
        index_in_group: Int32,
        copy_elems: cutlass.Constexpr[int],
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """Width-explicit clone of _copy_sparse_k_d128_row.

        v3.2 overrides K_CHUNK to 64 (the 128 B chase segment), so the
        kdq fill -- which still moves 256 B D_TILE_CTA rows -- names its
        width explicitly instead of inheriting K_CHUNK.
        """

        source_row_full = mKV[kv_index, None, (0, batch_idx)]
        source_row_offset = source_row_full.iterator + d_offset
        source_row = cute.make_tensor(
            cute.make_ptr(
                self.element_dtype,
                source_row_offset.llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((copy_elems,)),
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
        for tile in cutlass.range_constexpr(copy_elems // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            thread_source = thread_copy.partition_S(
                source_chunks[None, chunk_index]
            )
            thread_destination = thread_copy.partition_D(
                destination_chunks[None, chunk_index]
            )
            cute.copy(copy_atom, thread_source, thread_destination)

    @cute.jit
    def _zero_sparse_k_row_v32(
        self,
        destination_rows: cute.Tensor,
        destination_row: Int32,
        index_in_group: Int32,
        copy_elems: cutlass.Constexpr[int],
    ):
        """Width-explicit clone of _zero_sparse_k_d128_row."""

        destination_row_tensor = destination_rows[
            destination_row,
            None,
        ]
        destination_chunks = cute.flat_divide(
            destination_row_tensor,
            (8,),
        )
        for tile in cutlass.range_constexpr(copy_elems // 64):
            chunk_index = tile * self.KV_GROUP_SIZE + index_in_group
            destination_chunks[None, chunk_index].fill(0.0)

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
    def _load_chase_piece_v32(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        destination_rows: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        tile_index: Int32,
        piece_index: Int32,
        topk: Int32,
        rank: Int32,
        tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ):
        """v3.2 chase: gather one [own-kv64 x D64] K/V piece (8,192 B).

        The score plane is transposed, so the gathered K rows are the
        score-A M-half: CTA `rank` owns bundle rows
        kv[rank*64:(rank+1)*64] of the 128-token bundle `tile_index`.
        One call fills ONE D64 piece (gather row segments of 128 B,
        ruling-B geometry); v5 streams 32 pieces per bundle (4 head
        passes x 8 D-slices, piece_index = piece_global % 8 -- the
        same kv rows re-gather every pass, L2-hot for t >= 1) through
        the 2-slot chase ring.  `destination_rows` is the [64, K_CHUNK]
        row view of the ring slot.  V == K: this single fetch feeds
        the pass's two score-plane consumers (G1(t), G2(t)).
        """

        index_in_group = tidx % self.KV_GROUP_SIZE
        group_index = tidx // self.KV_GROUP_SIZE
        chase_rows = 2 * self.N_TILE_CTA
        rows_per_group = chase_rows // self.KV_NUM_GROUPS
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = row_iteration * self.KV_NUM_GROUPS + group_index
            topk_slot = (
                tile_index * (2 * self.N_TILE)
                + rank * chase_rows
                + local_n
            )
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
                    local_n,
                    kv_index,
                    batch_idx,
                    piece_index * self.K_CHUNK,
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

        # The sequential checkpoint keeps its direct dQ stores.  These
        # launcher-built values are consumed by the v0 kernel override.
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
    ISSUED_STREAM_STATE_BYTES = 16
    FIXED_METADATA_BYTES = (
        ISSUED_CTX_RING_BYTES
        + REDUCER_CTX_RING_BYTES
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
    EXPECTED_SHARED_STORAGE_BYTES = 207_872

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
        assert self.FIXED_METADATA_BYTES == 1_136
        assert self.CONTEXT_STAGES == 2
        assert self.REDUCER_STAGES == 2
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
    def _gather_score_kv_chunk(
        self,
        mKV: cute.Tensor,
        issued_ctx: cute.Tensor,
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
            kv_index = issued_ctx[
                self.CTX_KV_BASE_WORD + logical_n,
                context_slot,
            ]
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
        issued_ctx: cute.Tensor,
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
            kv_index = issued_ctx[
                self.CTX_KV_BASE_WORD + logical_n,
                context_slot,
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
        issued_ctx: cute.Tensor,
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
                issued_ctx,
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
        issued_ctx: cute.Tensor,
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
                issued_ctx,
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
    def _refill_bq_task(
        self,
        round_index: cutlass.Constexpr[int],
        bv_slot: Int32,
        raw_slots: cute.Tensor,
        dkv_a_layout: cute.ComposedLayout,
        tma_atom_qt: cute.CopyAtom,
        rank_g_qt: cute.Tensor,
        block_coord_vmnk,
        a_cta_layout: cute.Layout,
        grad_q_source_mbars: cute.Pointer,
        do_empty_pipeline,
        do_state: pipeline.PipelineState,
        do_ready: cutlass.Boolean,
        q_full_pipeline,
        q_state: pipeline.PipelineState,
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
        """Service one already-admitted BV refill without taking a whole slot."""

        if tidx == Int32(self.DESCRIPTOR_WARP * 32):
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
        with cute.arch.elect_one():
            do_empty_pipeline.consumer_wait(do_state, do_ready)
            q_full_pipeline.producer_acquire(q_state)
        cute.arch.sync_warp()
        if tidx == Int32(self.DESCRIPTOR_WARP * 32):
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
        if tidx == Int32(self.DESCRIPTOR_WARP * 32):
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
    def _mma_grad_round(
        self,
        round_index: cutlass.Constexpr[int],
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
        t_dkv_round: cute.Tensor,
        t_dq_round: cute.Tensor,
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
        op_pipeline.consumer_wait(op_state)
        slot = op_state.index

        pd_stage = issue_seq % Int32(self.PD_STAGES)
        bv_do = self._make_operand_slot_view(
            raw_slots,
            slot,
            self.OP_MAIN_OFFSET_BYTES,
            dkv_a_layout,
        )
        bq_q = self._make_operand_slot_view(
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
        dsk_operand = self._make_pd_stage_view(
            raw_ds_dk,
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
        dk_a_fragment = dkv_tiled_mma.make_fragment_A(bq_q)
        dq_a_fragment = dq_tiled_mma.make_fragment_A(bq_k)
        p_fragment = dkv_tiled_mma.make_fragment_B(p_operand)
        dsk_fragment = dkv_tiled_mma.make_fragment_B(
            dsk_operand
        )
        dsq_fragment = dq_tiled_mma.make_fragment_B(
            dsq_operand
        )

        if cutlass.const_expr(
            self.DIAGNOSTIC_AUX_STAGE >= 4
        ):
            dkv_pipeline.producer_acquire(dkv_state)
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
            dkv_pipeline.producer_commit(dkv_state)
            dkv_state.advance()

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

        op_pipeline.consumer_release(op_state)
        op_state.advance()
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
            op_state,
            p_wait_state,
            p_release_state,
            dsk_wait_state,
            dsk_release_state,
            dsq_wait_state,
            dsq_release_state,
            do_state,
            q_state,
            dkv_state,
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

        for value_index in cutlass.range_constexpr(
            cute.size(thread_values)
        ):
            d_in_round = Int32(
                cute.get(
                    thread_coordinates[value_index],
                    mode=[0],
                )
            )
            n_index = Int32(
                cute.get(
                    thread_coordinates[value_index],
                    mode=[1],
                )
            )
            kv_index = reducer_ctx[
                self.CTX_KV_BASE_WORD + n_index,
                reducer_slot,
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
                        issued_ctx_ring,
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
                                    issued_ctx_ring,
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
                                issued_ctx_ring,
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
                    if warp_idx == self.LOAD_COORDINATOR_WARP:
                        with cute.arch.elect_one():
                            context_slot = (
                                issue_seq
                                % Int32(self.CONTEXT_STAGES)
                            )
                            cute.arch.mbarrier_arrive(
                                ctx_reader_done_mbars_ptr
                                + context_slot
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

        # W7 multiplexes descriptor production and the older BV's Q refill.
        # A pending executable descriptor never blocks on IssuedCtx credit:
        # while its target slot is live, dO-empty is polled and serviced
        # first.  Traversal completion publishes sticky done independently
        # of both metadata-ring and BQ credit.
        if (
            cutlass.const_expr(
                self.DIAGNOSTIC_AUX_STAGE >= 3
            )
            and warp_idx == self.DESCRIPTOR_WARP
        ):
            traversal_seq = Int32(0)
            committed_count = Int32(0)
            pending_descriptor = cutlass.Boolean(False)
            done_published = cutlass.Boolean(False)
            refill_count = Int32(0)
            whole_ordinal = Int32(self.K_CHUNKS)
            engine_active = cutlass.Boolean(True)
            while engine_active:
                progressed = cutlass.Boolean(False)
                refill_issue_seq = (
                    refill_count // Int32(self.D_ROUNDS)
                )
                refill_round = (
                    refill_count % Int32(self.D_ROUNDS)
                )
                refill_admitted = cutlass.Boolean(False)
                refill_has_next = cutlass.Boolean(False)
                if refill_issue_seq < committed_count:
                    if (
                        refill_issue_seq + Int32(1)
                        < committed_count
                    ):
                        refill_admitted = cutlass.Boolean(True)
                        refill_has_next = cutlass.Boolean(True)
                    else:
                        if done_published:
                            refill_admitted = cutlass.Boolean(True)

                if refill_admitted:
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
                    do_ready = (
                        do_empty_pipeline.consumer_try_wait(
                            do_state
                        )
                    )
                    if do_ready:
                        if refill_round == Int32(0):
                            if refill_has_next:
                                whole_ordinal += Int32(
                                    self.K_CHUNKS
                                )
                            self._refill_bq_task(
                                0,
                                whole_ordinal
                                % Int32(self.OP_STAGES),
                                raw_slots,
                                dkv_a_layout,
                                tma_atom_qt,
                                rank_g_qt,
                                block_coord_vmnk,
                                a_cta_layout,
                                grad_q_source_mbars_ptr,
                                do_empty_pipeline,
                                do_state,
                                do_ready,
                                q_full_pipeline,
                                q_state,
                                grad_a_stage_bytes,
                                refill_issue_seq,
                                rank,
                                tidx,
                                token_idx,
                                batch_idx,
                                trace_buffer,
                                trace_token_idx,
                                trace_batch_idx,
                            )
                        else:
                            self._refill_bq_task(
                                1,
                                whole_ordinal
                                % Int32(self.OP_STAGES),
                                raw_slots,
                                dkv_a_layout,
                                tma_atom_qt,
                                rank_g_qt,
                                block_coord_vmnk,
                                a_cta_layout,
                                grad_q_source_mbars_ptr,
                                do_empty_pipeline,
                                do_state,
                                do_ready,
                                q_full_pipeline,
                                q_state,
                                grad_a_stage_bytes,
                                refill_issue_seq,
                                rank,
                                tidx,
                                token_idx,
                                batch_idx,
                                trace_buffer,
                                trace_token_idx,
                                trace_batch_idx,
                            )
                        whole_ordinal += Int32(1)
                        refill_count += Int32(1)
                        progressed = cutlass.Boolean(True)

                if not progressed:
                    if pending_descriptor:
                        slot = (
                            committed_count
                            % Int32(self.CONTEXT_STAGES)
                        )
                        epoch = (
                            committed_count
                            // Int32(self.CONTEXT_STAGES)
                        ) & Int32(1)
                        empty_ready = _mbarrier_try_wait(
                            issued_ctx_mbars_ptr
                            + self.ISSUED_EMPTY_MBAR_BASE
                            + slot,
                            epoch ^ Int32(1),
                        )
                        if empty_ready:
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
                            pending_descriptor = (
                                cutlass.Boolean(False)
                            )
                            progressed = cutlass.Boolean(True)
                    else:
                        if traversal_seq < traversal_tile_count:
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
                            pending_descriptor = (
                                traversal_descriptor[
                                    self.DESCRIPTOR_EXECUTE_WORD
                                ]
                                != Int32(0)
                            )
                            traversal_seq += Int32(1)
                            progressed = cutlass.Boolean(True)
                        else:
                            if not done_published:
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
                                done_published = cutlass.Boolean(True)
                                progressed = cutlass.Boolean(True)

                engine_active = cutlass.Boolean(True)
                if done_published:
                    if (
                        refill_count
                        == Int32(self.D_ROUNDS) * committed_count
                    ):
                        engine_active = cutlass.Boolean(False)

            if committed_count > Int32(0):
                with cute.arch.elect_one():
                    q_count = Int32(self.D_ROUNDS) * committed_count
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

            with cute.arch.elect_one():
                self._wait_pair(
                    issued_stream_done_ack_mbars_ptr
                    + self.STREAM_DONE_ACK_MBAR,
                    Int32(0),
                )

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
                for round_index in cutlass.range_constexpr(
                    self.D_ROUNDS
                ):
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
                    ) = self._mma_grad_round(
                        round_index,
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
                        t_dkv[round_index],
                        t_dq[round_index],
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
                    for round_index in cutlass.range_constexpr(
                        self.D_ROUNDS
                    ):
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
                        ) = self._mma_grad_round(
                            round_index,
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
                            t_dkv[round_index],
                            t_dq[round_index],
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


class FlashAttentionDSABackwardSm100TwoCTAV1A0(
    FlashAttentionDSABackwardSm100TwoCTA
):
    """Sequential-macro bring-up for the v1 data plane.

    A0 deliberately starts from the already validated common CG2 host/layout
    construction.  It owns a different SharedStorage and kernel; no v0
    operand FIFO or reducer-context storage is reachable from this class.
    Once CP0 closes the packed-BF16 DSM primitives, this class is the
    correctness bring-up point before the same atomic spans are split across
    persistent load/MMA/math/route/reduce roles.
    """

    THREADS_PER_CTA = 256
    MAIN_BARRIER_ID = 2
    ROUTE_THREADS = 256
    MATH_WARP_COUNT = 4
    MATH_THREADS_PER_CTA = 128
    assert MATH_THREADS_PER_CTA == MATH_WARP_COUNT * 32
    KV_LOAD_THREAD_BEGIN = 128
    KV_LOAD_THREADS = 128

    OP_STAGES = 2
    PD_STAGES = 2
    CONTEXT_STAGES = 4
    OP_BYTES_PER_STAGE = 32 * 1024
    PD_BYTES_PER_STAGE = 16 * 1024
    OP_PAYLOAD_BYTES = OP_STAGES * OP_BYTES_PER_STAGE
    PD_PAYLOAD_BYTES = PD_STAGES * PD_BYTES_PER_STAGE
    STATIONARY_BYTES_PER_TENSOR = 64 * 1024
    DATA_PAYLOAD_BYTES = (
        2 * STATIONARY_BYTES_PER_TENSOR
        + OP_PAYLOAD_BYTES
        + PD_PAYLOAD_BYTES
    )

    TILE_CTX_WORDS = 68
    TILE_CTX_BYTES = TILE_CTX_WORDS * 4
    TILE_CTX_RING_BYTES = CONTEXT_STAGES * TILE_CTX_BYTES
    TRAVERSAL_DESCRIPTOR_WORDS = 72
    TRAVERSAL_DESCRIPTOR_BYTES = TRAVERSAL_DESCRIPTOR_WORDS * 4
    STREAM_STATE_WORDS = 4

    CTX_ISSUE_SEQ_WORD = 0
    CTX_LOGICAL_TILE_WORD = 1
    CTX_KV_BASE_WORD = 2
    CTX_VALID_LO_WORD = 66
    CTX_VALID_HI_WORD = 67
    DESCRIPTOR_EXECUTE_WORD = 68

    # Two completion barriers are reserved per typed route for each macro
    # lane.  A0 uses only the subset needed by its serialized schedule; the
    # fixed array lets the role-specialized successor retain the exact
    # 224-KiB data plane without changing its storage ABI.
    ROUTE_KIND_COUNT = 8
    ROUTE_MBAR_COUNT = ROUTE_KIND_COUNT * PD_STAGES
    MMA_DONE_STAGES = (
        FlashAttentionDSABackwardSm100TwoCTA.MMA_DONE_STAGES
    )
    MMA_MBAR_COUNT = 2 * MMA_DONE_STAGES
    SOURCE_MBAR_COUNT = 2
    MAX_MBAR_COUNT = 96

    STATS_WORDS = (
        FlashAttentionDSABackwardSm100TwoCTA.H_TILE_CTA * 2
    )
    CONTROL_WORDS = 32
    STATS_CONTROL_BYTES = (
        STATS_WORDS * (Float32.width // 8)
        + CONTROL_WORDS * 4
    )

    assert DATA_PAYLOAD_BYTES == 224 * 1024
    assert TILE_CTX_BYTES == 272
    assert TILE_CTX_RING_BYTES == 1_088
    assert TRAVERSAL_DESCRIPTOR_BYTES == 288
    assert STATS_CONTROL_BYTES == 640

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
        self.main_barrier = pipeline.NamedBarrier(
            barrier_id=self.MAIN_BARRIER_ID,
            num_threads=self.THREADS_PER_CTA,
        )

    @cute.jit
    def _score_chunk_rows_v1(
        self,
        tensor: cute.Tensor,
        chunk: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        return cute.composition(
            tensor[None, None, None, chunk],
            cute.make_layout(
                (self.H_TILE_CTA, self.K_CHUNK),
            ),
        )

    @cute.jit
    def _k_chunk_rows_v1(
        self,
        tensor: cute.Tensor,
        chunk: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        return cute.composition(
            tensor[None, None, None, chunk],
            cute.make_layout(
                (self.N_TILE_CTA, self.K_CHUNK),
            ),
        )

    @cute.jit
    def _kd_round_rows_v1(
        self,
        tensor: cute.Tensor,
        round_index: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        return cute.composition(
            tensor[None, None, None, round_index],
            cute.make_layout(
                (self.D_TILE_CTA, self.N_TILE),
            ),
        )

    @cute.jit
    def _grad_a_rows_v1(self, tensor: cute.Tensor) -> cute.Tensor:
        return cute.composition(
            tensor[None, None, None, 0],
            cute.make_layout(
                (self.D_TILE_CTA, self.H_TILE_CLUSTER),
            ),
        )

    @cute.jit
    def _p_rows_v1(self, tensor: cute.Tensor) -> cute.Tensor:
        return cute.composition(
            tensor[None, None, None, 0],
            cute.make_layout(
                (self.N_TILE_CTA, self.H_TILE_CLUSTER),
            ),
        )

    @cute.jit
    def _dsh_rows_v1(self, tensor: cute.Tensor) -> cute.Tensor:
        return cute.composition(
            tensor[None, None, None, 0],
            cute.make_layout(
                (self.H_TILE_CTA, self.N_TILE),
            ),
        )

    @cute.jit
    def _issue_dq_stage_v1(
        self,
        dq_tiled_mma: cute.TiledMma,
        t_dq: cute.Tensor,
        k_d_fragment: cute.Tensor,
        ds_h_fragment: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        accumulate: cutlass.Constexpr[bool],
        done_pipeline,
        producer_state,
    ):
        """Publish both disjoint dQ D256 rounds before either round waits.

        The existing caller invokes this helper once per round and then
        consumes one completion generation.  The round-0 invocation issues
        both generations back-to-back into the two TMEM dQ regions; the
        round-1 invocation intentionally issues no additional work.  This
        preserves the caller's two balanced consumer waits while increasing
        the maximum outstanding dQ depth from one generation to two.
        """

        assert self.D_ROUNDS <= self.MMA_DONE_STAGES
        if cutlass.const_expr(round_index == 0):
            dq_round_stride = (
                self.TMEM_DQ1_OFFSET - self.TMEM_DQ0_OFFSET
            )
            for deep_round_index in cutlass.range_constexpr(
                self.D_ROUNDS
            ):
                round_t_dq = cute.make_tensor(
                    t_dq.iterator
                    + deep_round_index * dq_round_stride,
                    t_dq.layout,
                )
                done_pipeline.producer_acquire(producer_state)
                mma = dq_tiled_mma.with_()
                mma.set(tcgen05.Field.ACCUMULATE, accumulate)
                for k_block in cutlass.range_constexpr(
                    cute.size(k_d_fragment, mode=[2])
                ):
                    cute.gemm(
                        mma,
                        round_t_dq,
                        k_d_fragment[
                            None,
                            None,
                            k_block,
                            deep_round_index,
                        ],
                        ds_h_fragment[None, None, k_block, 0],
                        round_t_dq,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.arch.fence_view_async_tmem_store()
                done_pipeline.producer_commit(producer_state)
                producer_state.advance()
        return producer_state

    @cute.jit
    def _decode_ctx_v1(
        self,
        mTopkIdxs: cute.Tensor,
        context: cute.Tensor,
        descriptor: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        topk: Int32,
        logical_tile: Int32,
        issue_seq: Int32,
        tidx: Int32,
    ) -> cutlass.Boolean:
        """Decode one traversal tile into an immutable 272-byte context."""

        self.main_barrier.arrive_and_wait()
        if tidx == Int32(0):
            valid_lo = Int32(0)
            valid_hi = Int32(0)
            execute = Int32(0)
            context[self.CTX_ISSUE_SEQ_WORD] = issue_seq
            context[self.CTX_LOGICAL_TILE_WORD] = logical_tile
            for logical_n in cutlass.range_constexpr(self.N_TILE):
                topk_slot = (
                    logical_tile * Int32(self.N_TILE)
                    + Int32(logical_n)
                )
                kv_index = Int32(-1)
                if topk_slot < topk:
                    kv_index = mTopkIdxs[
                        topk_slot,
                        (token_idx, batch_idx),
                    ]
                context[
                    self.CTX_KV_BASE_WORD + logical_n
                ] = kv_index
                if kv_index >= Int32(0):
                    execute = Int32(1)
                    if cutlass.const_expr(logical_n < 32):
                        valid_lo = (
                            valid_lo
                            | (
                                Int32(1)
                                << Int32(logical_n)
                            )
                        )
                    else:
                        valid_hi = (
                            valid_hi
                            | (
                                Int32(1)
                                << Int32(logical_n - 32)
                            )
                        )
            context[self.CTX_VALID_LO_WORD] = valid_lo
            context[self.CTX_VALID_HI_WORD] = valid_hi
            descriptor[self.DESCRIPTOR_EXECUTE_WORD] = execute
        self.main_barrier.arrive_and_wait()
        return (
            descriptor[self.DESCRIPTOR_EXECUTE_WORD]
            != Int32(0)
        )

    @cute.jit
    def _load_k_from_ctx_v1(
        self,
        mKV: cute.Tensor,
        context: cute.Tensor,
        destination: cute.Tensor,
        batch_idx: Int32,
        rank: Int32,
        loader_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ) -> None:
        """One and only one G2S producer for the rank-owned N32 K rows."""

        index_in_group = loader_tidx % self.KV_GROUP_SIZE
        group_index = loader_tidx // self.KV_GROUP_SIZE
        rows_per_group = self.N_TILE_CTA // self.KV_NUM_GROUPS
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = (
                row_iteration * self.KV_NUM_GROUPS
                + group_index
            )
            logical_n = rank * self.N_TILE_CTA + local_n
            kv_index = context[
                self.CTX_KV_BASE_WORD + logical_n
            ]
            for chunk in cutlass.range_constexpr(self.K_CHUNKS):
                destination_rows = self._k_chunk_rows_v1(
                    destination,
                    chunk,
                )
                if kv_index >= Int32(0):
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
    def _capture_k_round_v1(
        self,
        source: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        captured: cute.Tensor,
        tidx: Int32,
    ) -> None:
        """Capture one complete N32 x D256 K_N round before any rewrite."""

        for vector_iteration in cutlass.range_constexpr(4):
            flat_vector = (
                tidx
                + Int32(vector_iteration * self.ROUTE_THREADS)
            )
            local_n = flat_vector // Int32(32)
            d_vector_in_round = flat_vector % Int32(32)
            chunk_in_round = d_vector_in_round // Int32(16)
            d_in_chunk = (
                d_vector_in_round % Int32(16)
            ) * Int32(8)
            source_chunk = (
                Int32(round_index * 2) + chunk_in_round
            )
            for word in cutlass.range_constexpr(4):
                coordinate = (
                    local_n,
                    d_in_chunk + Int32(2 * word),
                )
                if source_chunk == Int32(0):
                    rows = self._k_chunk_rows_v1(source, 0)
                    captured[
                        vector_iteration * 4 + word
                    ] = _load_shared_u32_at(rows, coordinate)
                elif source_chunk == Int32(1):
                    rows = self._k_chunk_rows_v1(source, 1)
                    captured[
                        vector_iteration * 4 + word
                    ] = _load_shared_u32_at(rows, coordinate)
                elif source_chunk == Int32(2):
                    rows = self._k_chunk_rows_v1(source, 2)
                    captured[
                        vector_iteration * 4 + word
                    ] = _load_shared_u32_at(rows, coordinate)
                else:
                    rows = self._k_chunk_rows_v1(source, 3)
                    captured[
                        vector_iteration * 4 + word
                    ] = _load_shared_u32_at(rows, coordinate)

    @cute.jit
    def _route_k_full_rewrite_v1(
        self,
        source: cute.Tensor,
        destination: cute.Tensor,
        destination_full: cute.Pointer,
        rank: Int32,
        peer_rank: Int32,
        tidx: Int32,
    ) -> None:
        """Transpose K_N into K_D with one overwrite-safe D256 round at a time.

        Each round is a 2x2 CTA/N-owner-to-D-owner transpose.  Both ranks
        capture all 16 KiB belonging to that round before either rank writes
        through the native K_D layout.  Every diagonal value is explicitly
        rewritten locally and every off-diagonal value is explicitly written
        to the peer.  No correctness step assumes byte identity between the
        score-B and dQ-A composed layouts, or quadrant disjointness inside a
        round.
        """

        # A 16-KiB round contains four BF16x8 vectors per thread: sixteen
        # packed u32 registers, versus 32 for a whole-field capture.
        captured = cute.make_rmem_tensor((16,), cutlass.Uint32)
        rows_r0 = self._kd_round_rows_v1(destination, 0)
        rows_r1 = self._kd_round_rows_v1(destination, 1)
        for round_index in cutlass.range_constexpr(
            self.D_ROUNDS
        ):
            if tidx == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    destination_full,
                    8 * 1024,
                )
            cute.arch.fence_view_async_shared()
            self.main_barrier.arrive_and_wait()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            self._capture_k_round_v1(
                source,
                round_index,
                captured,
                tidx,
            )
            self.main_barrier.arrive_and_wait()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            for vector_iteration in cutlass.range_constexpr(4):
                flat_vector = (
                    tidx
                    + Int32(
                        vector_iteration * self.ROUTE_THREADS
                    )
                )
                local_n = flat_vector // Int32(32)
                d_vector_in_round = flat_vector % Int32(32)
                owner_d = (
                    d_vector_in_round // Int32(16)
                )
                local_d = (
                    d_vector_in_round % Int32(16)
                ) * Int32(8)
                global_n = (
                    rank * Int32(self.N_TILE_CTA) + local_n
                )
                words = vector_iteration * 4
                if cutlass.const_expr(round_index == 0):
                    destination_rows = rows_r0
                else:
                    destination_rows = rows_r1
                if owner_d == rank:
                    for word in cutlass.range_constexpr(4):
                        _store_shared_u32_at(
                            destination_rows,
                            (
                                local_d + Int32(2 * word),
                                global_n,
                            ),
                            captured[words + word],
                        )
                else:
                    _store_shared_remote_u32x4(
                        destination_rows.iterator
                        + destination_rows.layout(
                            (local_d, global_n)
                        ),
                        destination_full,
                        peer_rank,
                        captured[words],
                        captured[words + 1],
                        captured[words + 2],
                        captured[words + 3],
                    )

            _mbarrier_wait_acquire_cluster(
                destination_full,
                Int32(round_index),
            )
            cute.arch.fence_view_async_shared()
            self.main_barrier.arrive_and_wait()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def _materialize_qdo_round_v1(
        self,
        stationary_q: cute.Tensor,
        stationary_do: cute.Tensor,
        destination_q: cute.Tensor,
        destination_do: cute.Tensor,
        q_destination_full: cute.Pointer,
        do_destination_full: cute.Pointer,
        round_index: cutlass.Constexpr[int],
        rank: Int32,
        peer_rank: Int32,
        tidx: Int32,
    ) -> None:
        """Stream the two H-owner stationary tensors into D-owner views."""

        if tidx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                q_destination_full,
                16 * 1024,
            )
            cute.arch.mbarrier_arrive_and_expect_tx(
                do_destination_full,
                16 * 1024,
            )
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        q_destination_rows = self._grad_a_rows_v1(destination_q)
        do_destination_rows = self._grad_a_rows_v1(
            destination_do
        )
        q_source_rows_0 = self._score_chunk_rows_v1(
            stationary_q,
            0,
        )
        q_source_rows_1 = self._score_chunk_rows_v1(
            stationary_q,
            1,
        )
        q_source_rows_2 = self._score_chunk_rows_v1(
            stationary_q,
            2,
        )
        q_source_rows_3 = self._score_chunk_rows_v1(
            stationary_q,
            3,
        )
        do_source_rows_0 = self._score_chunk_rows_v1(
            stationary_do,
            0,
        )
        do_source_rows_1 = self._score_chunk_rows_v1(
            stationary_do,
            1,
        )
        do_source_rows_2 = self._score_chunk_rows_v1(
            stationary_do,
            2,
        )
        do_source_rows_3 = self._score_chunk_rows_v1(
            stationary_do,
            3,
        )
        route_tidx = tidx % Int32(128)
        is_q_route = tidx >= Int32(128)
        local_h_lane = route_tidx // Int32(
            self.D_TILE_CLUSTER // 8
        )
        d_vector = route_tidx % Int32(
            self.D_TILE_CLUSTER // 8
        )
        d_in_cluster = d_vector * Int32(8)
        global_d = (
            Int32(round_index * self.D_TILE_CLUSTER)
            + d_in_cluster
        )
        chunk = global_d // Int32(self.K_CHUNK)
        d_in_chunk = global_d % Int32(self.K_CHUNK)
        owner_d = d_in_cluster // Int32(self.D_TILE_CTA)
        local_d = d_in_cluster % Int32(self.D_TILE_CTA)

        # Both score-A and dKV-A are contiguous along D.  Each 128-thread team
        # therefore fixes one H row and moves one native BF16x8 D vector.
        # Source and destination do not alias, so only that vector is live.
        for vector_iteration in cutlass.range(16, unroll=1):
            local_h = (
                local_h_lane
                + vector_iteration
                * Int32(
                    (self.THREADS_PER_CTA // 2)
                    // (self.D_TILE_CLUSTER // 8)
                )
            )
            global_h = rank * Int32(self.H_TILE_CTA) + local_h

            packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for word in cutlass.range_constexpr(4):
                d0 = d_in_chunk + Int32(2 * word)
                if is_q_route:
                    if chunk == Int32(0):
                        packed[word] = _load_shared_u32_at(
                            q_source_rows_0,
                            (local_h, d0),
                        )
                    elif chunk == Int32(1):
                        packed[word] = _load_shared_u32_at(
                            q_source_rows_1,
                            (local_h, d0),
                        )
                    elif chunk == Int32(2):
                        packed[word] = _load_shared_u32_at(
                            q_source_rows_2,
                            (local_h, d0),
                        )
                    else:
                        packed[word] = _load_shared_u32_at(
                            q_source_rows_3,
                            (local_h, d0),
                        )
                else:
                    if chunk == Int32(0):
                        packed[word] = _load_shared_u32_at(
                            do_source_rows_0,
                            (local_h, d0),
                        )
                    elif chunk == Int32(1):
                        packed[word] = _load_shared_u32_at(
                            do_source_rows_1,
                            (local_h, d0),
                        )
                    elif chunk == Int32(2):
                        packed[word] = _load_shared_u32_at(
                            do_source_rows_2,
                            (local_h, d0),
                        )
                    else:
                        packed[word] = _load_shared_u32_at(
                            do_source_rows_3,
                            (local_h, d0),
                        )

            if is_q_route:
                # Issue the peer stripe first so the disjoint local stripe
                # can cover the final DSM transaction drain.
                if owner_d != rank:
                    _store_shared_remote_u32x4(
                        q_destination_rows.iterator
                        + q_destination_rows.layout(
                            (local_d, global_h)
                        ),
                        q_destination_full,
                        peer_rank,
                        packed[0],
                        packed[1],
                        packed[2],
                        packed[3],
                    )
                else:
                    for word in cutlass.range_constexpr(4):
                        _store_shared_u32_at(
                            q_destination_rows,
                            (
                                local_d + Int32(2 * word),
                                global_h,
                            ),
                            packed[word],
                        )
            else:
                if owner_d != rank:
                    _store_shared_remote_u32x4(
                        do_destination_rows.iterator
                        + do_destination_rows.layout(
                            (local_d, global_h)
                        ),
                        do_destination_full,
                        peer_rank,
                        packed[0],
                        packed[1],
                        packed[2],
                        packed[3],
                    )
                else:
                    for word in cutlass.range_constexpr(4):
                        _store_shared_u32_at(
                            do_destination_rows,
                            (
                                local_d + Int32(2 * word),
                                global_h,
                            ),
                            packed[word],
                        )

        _mbarrier_wait_acquire_cluster(
            q_destination_full,
            Int32(round_index),
        )
        _mbarrier_wait_acquire_cluster(
            do_destination_full,
            Int32(round_index),
        )
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _load_native_qt_round_v1(
        self,
        tma_atom_qt: cute.CopyAtom,
        t_qt_smem: cute.Tensor,
        t_qt_gmem: cute.Tensor,
        destination_full: cute.Pointer,
        grad_a_stage_bytes: cutlass.Constexpr[int],
        round_index: cutlass.Constexpr[int],
        warp_idx: Int32,
        tidx: Int32,
    ) -> None:
        """Load one launcher-native QT/dKV-A operand for dK."""

        if tidx == Int32(0):
            cute.arch.mbarrier_init(destination_full, 1)
        self.main_barrier.arrive_and_wait()
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    destination_full,
                    grad_a_stage_bytes,
                )
            cute.copy(
                tma_atom_qt,
                t_qt_gmem[None, round_index],
                t_qt_smem[None, 0],
                tma_bar_ptr=destination_full,
            )
        cute.arch.mbarrier_wait(destination_full, Int32(0))
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _load_native_dot_round_v1(
        self,
        tma_atom_dot: cute.CopyAtom,
        t_dot_smem: cute.Tensor,
        t_dot_gmem: cute.Tensor,
        destination_full: cute.Pointer,
        grad_a_stage_bytes: cutlass.Constexpr[int],
        round_index: cutlass.Constexpr[int],
        warp_idx: Int32,
        tidx: Int32,
    ) -> None:
        """Load one launcher-native dOT/dKV-A operand for the dV oracle."""

        if tidx == Int32(0):
            cute.arch.mbarrier_init(destination_full, 1)
        self.main_barrier.arrive_and_wait()
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    destination_full,
                    grad_a_stage_bytes,
                )
            cute.copy(
                tma_atom_dot,
                t_dot_gmem[None, round_index],
                t_dot_smem[None, 0],
                tma_bar_ptr=destination_full,
            )
        cute.arch.mbarrier_wait(destination_full, Int32(0))
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _route_ds_full_rewrite_v1(
        self,
        source_h: cute.Tensor,
        destination_n: cute.Tensor,
        destination_full: cute.Pointer,
        route_phase: Int32,
        rank: Int32,
        peer_rank: Int32,
        tidx: Int32,
    ) -> None:
        """Capture dS_H, then destructively rewrite the same 8 KiB as dS_N."""

        if tidx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                destination_full,
                4 * 1024,
            )
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        source_rows = self._dsh_rows_v1(source_h)
        captured = cute.make_rmem_tensor((8,), cutlass.Uint32)
        for vector_iteration in cutlass.range_constexpr(2):
            flat_vector = (
                tidx
                + Int32(vector_iteration * self.ROUTE_THREADS)
            )
            n_index = flat_vector // Int32(8)
            h_vector = flat_vector % Int32(8)
            local_h = h_vector * Int32(8)
            for word in cutlass.range_constexpr(4):
                captured[vector_iteration * 4 + word] = (
                    _load_shared_u32_at(
                        source_rows,
                        (
                            local_h + Int32(2 * word),
                            n_index,
                        ),
                    )
                )

        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        destination_rows = self._p_rows_v1(destination_n)
        for vector_iteration in cutlass.range_constexpr(2):
            flat_vector = (
                tidx
                + Int32(vector_iteration * self.ROUTE_THREADS)
            )
            n_index = flat_vector // Int32(8)
            h_vector = flat_vector % Int32(8)
            local_h = h_vector * Int32(8)
            owner_n = n_index // Int32(self.N_TILE_CTA)
            local_n = n_index % Int32(self.N_TILE_CTA)
            global_h = rank * Int32(self.H_TILE_CTA) + local_h
            words = vector_iteration * 4
            if owner_n == rank:
                for word in cutlass.range_constexpr(4):
                    _store_shared_u32_at(
                        destination_rows,
                        (
                            local_n,
                            global_h + Int32(2 * word),
                        ),
                        captured[words + word],
                    )
            else:
                _store_shared_remote_u32x4(
                    destination_rows.iterator
                    + destination_rows.layout((local_n, global_h)),
                    destination_full,
                    peer_rank,
                    captured[words],
                    captured[words + 1],
                    captured[words + 2],
                    captured[words + 3],
                )

        _mbarrier_wait_acquire_cluster(
            destination_full,
            route_phase,
        )
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def _compute_publish_pd_v1(
        self,
        t_score: cute.Tensor,
        t_dp: cute.Tensor,
        score_tmem_load: cute.CopyAtom,
        rank_score_coordinates: cute.Tensor,
        softmax_stats: cute.Tensor,
        scale_softmax: Float32,
        context: cute.Tensor,
        p_h_store_destination: cute.Tensor,
        ds_h_store_destination: cute.Tensor,
        done_pipeline,
        consumer_state: pipeline.PipelineState,
        tidx: Int32,
    ) -> pipeline.PipelineState:
        """T2R S/dP and publish score-distributed P_H/dS_H images."""

        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        math_state = consumer_state.clone()
        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            score_copy = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_score,
            )
            score_thread = score_copy.get_slice(tidx)
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
            r_ds = cute.make_rmem_tensor(
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
            thread_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            r_p_store = thread_copy_r2s.retile(r_p)
            r_ds_store = thread_copy_r2s.retile(r_ds)
            dp_copy = tcgen05.make_tmem_copy(
                score_tmem_load,
                t_dp,
            )
            dp_thread = dp_copy.get_slice(tidx)
            dp_source = dp_thread.partition_S(t_dp)
            r_dp = cute.make_rmem_tensor(
                score_coordinates.shape,
                self.acc_dtype,
            )

            done_pipeline.consumer_wait(math_state)
            cute.copy(score_copy, score_source, r_score)
            cute.arch.fence_view_async_tmem_load()
            done_pipeline.consumer_release(math_state)
            math_state.advance()

            done_pipeline.consumer_wait(math_state)
            cute.copy(dp_copy, dp_source, r_dp)
            cute.arch.fence_view_async_tmem_load()
            done_pipeline.consumer_release(math_state)

            math_tidx = tidx
            local_h = math_tidx % Int32(self.H_TILE_CTA)
            n_owner = math_tidx // Int32(self.H_TILE_CTA)
            valid_bits = context[self.CTX_VALID_LO_WORD]
            if n_owner != Int32(0):
                valid_bits = context[self.CTX_VALID_HI_WORD]
            softmax_scale_log2_e = (
                scale_softmax * Float32(math.log2(math.e))
            )
            lse = softmax_stats[local_h, 0]
            delta = softmax_stats[local_h, 1]

            assert cute.size(r_score) == self.N_TILE_CTA
            assert cute.size(r_p) == self.N_TILE_CTA
            assert cute.size(r_ds) == self.N_TILE_CTA
            for local_n in cutlass.range_constexpr(
                self.N_TILE_CTA
            ):
                is_valid = (
                    (
                        valid_bits >> Int32(local_n)
                    )
                    & Int32(1)
                ) != Int32(0)
                p_value = cute.math.exp2(
                    (
                        r_score[local_n]
                        * softmax_scale_log2_e
                        + lse
                    ),
                    fastmath=True,
                )
                ds_value = (
                    (
                        r_dp[local_n]
                        + delta
                    )
                    * p_value
                    * scale_softmax
                )
                if not is_valid:
                    p_value = Float32(0.0)
                    ds_value = Float32(0.0)
                r_p[local_n] = self.element_dtype(p_value)
                r_ds[local_n] = self.element_dtype(ds_value)

            # P and dS first use the same production-verified score T2R/R2S
            # byte image. P_H is destructively routed to P_N only after all
            # threads and both CTAs have completed this whole-field store.
            t_rs_p = thread_copy_r2s.partition_D(
                p_h_store_destination
            )
            t_rs_ds = thread_copy_r2s.partition_D(
                ds_h_store_destination
            )
            assert cute.size(t_rs_p, mode=[4]) == 1
            assert cute.size(t_rs_ds, mode=[4]) == 1
            t_rs_p_tile = t_rs_p[
                None,
                None,
                None,
                None,
                0,
            ]
            t_rs_ds_tile = t_rs_ds[
                None,
                None,
                None,
                None,
                0,
            ]
            assert t_rs_p_tile.shape == r_p_store.shape
            assert t_rs_ds_tile.shape == r_ds_store.shape
            cute.copy(
                tiled_copy_r2s,
                r_p_store,
                t_rs_p_tile,
            )
            cute.copy(
                tiled_copy_r2s,
                r_ds_store,
                t_rs_ds_tile,
            )

        consumer_state.advance()
        consumer_state.advance()
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        return consumer_state

    @cute.jit
    def _issue_dv_dk_final_v1(
        self,
        dkv_tiled_mma: cute.TiledMma,
        t_dkv: cute.Tensor,
        dout_fragment: cute.Tensor,
        p_fragment: cute.Tensor,
        q_fragment: cute.Tensor,
        ds_dk_fragment: cute.Tensor,
        done_pipeline,
        producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Publish one final dV-overwrite plus dK-accumulate generation."""

        done_pipeline.producer_acquire(producer_state)

        dv_mma = dkv_tiled_mma.with_()
        dv_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range_constexpr(
            cute.size(dout_fragment, mode=[2])
        ):
            cute.gemm(
                dv_mma,
                t_dkv,
                dout_fragment[None, None, k_block, 0],
                p_fragment[None, None, k_block, 0],
                t_dkv,
            )
            dv_mma.set(tcgen05.Field.ACCUMULATE, True)

        dk_mma = dkv_tiled_mma.with_()
        dk_mma.set(tcgen05.Field.ACCUMULATE, True)
        for k_block in cutlass.range_constexpr(
            cute.size(q_fragment, mode=[2])
        ):
            cute.gemm(
                dk_mma,
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
    def _split_wg_t1d(
        self,
        tensor: cute.Tensor,
        wg_idx: Int32,
        num_wg: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        """Split the first nontrivial value mode across warp groups."""

        # This is the FA4 SM100 backward split rule.  product_each removes
        # nested shape structure before selecting the value-bearing mode,
        # unlike the baseline helper that blindly splits the final mode.
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
    def _atomic_dkv_from_context_v1(
        self,
        t_dkv: cute.Tensor,
        dkv_tmem_load: cute.CopyAtom,
        rank_coordinates: cute.Tensor,
        mdKV_acc: cute.Tensor,
        context: cute.Tensor,
        round_index: cutlass.Constexpr[int],
        batch_idx: Int32,
        tidx: Int32,
        done_pipeline,
        consumer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Reduce one tile with two warp groups sharing ROW_MAJOR rows."""

        # T1d eight-warp ROW_MAJOR N-half reducer.
        # Keep the registered 128-thread done-pipeline group, then lend the
        # other four resident warps to the reducer through full-CTA barriers.
        math_state = consumer_state.clone()
        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            done_pipeline.consumer_wait(math_state)
        self.main_barrier.arrive_and_wait()

        tiled_t2r = tcgen05.make_tmem_copy(
            dkv_tmem_load,
            t_dkv,
        )
        dp_idx = tidx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = tidx // Int32(self.MATH_THREADS_PER_CTA)
        thread_t2r = tiled_t2r.get_slice(dp_idx)
        thread_source = self._split_wg_t1d(
            thread_t2r.partition_S(t_dkv),
            wg_idx,
            2,
        )
        thread_coordinates = self._split_wg_t1d(
            thread_t2r.partition_D(rank_coordinates),
            wg_idx,
            2,
        )
        thread_values = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )

        # V1_SPAN_REDUCE_T2R_BEGIN
        cute.copy(tiled_t2r, thread_source, thread_values)
        cute.arch.fence_view_async_tmem_load()
        # V1_SPAN_REDUCE_T2R_END
        self.main_barrier.arrive_and_wait()
        if tidx < Int32(self.MATH_THREADS_PER_CTA):
            done_pipeline.consumer_release(math_state)

        # V1_SPAN_REDUCE_ATOMIC_BEGIN
        # Each duplicate dp_idx now owns one N32 half of its proven D-row.
        # The same T1b four-shuffle network transposes eight 4x4 blocks per
        # thread, so both warp groups jointly reproduce T1b's exact N64
        # vectors and addresses with half the per-thread issue chain.
        assert cute.size(thread_values) == self.N_TILE // 2
        lane_in_quad = tidx % Int32(4)
        for vector_index in cutlass.range_constexpr(
            self.N_TILE // 8
        ):
            value_base = vector_index * 4
            value_0 = thread_values[value_base]
            value_1 = thread_values[value_base + 1]
            value_2 = thread_values[value_base + 2]
            value_3 = thread_values[value_base + 3]

            swap_0 = value_0
            swap_1 = value_1
            if (lane_in_quad & Int32(1)) == Int32(0):
                swap_0 = value_1
                swap_1 = value_3
            else:
                swap_0 = value_0
                swap_1 = value_2
            peer_0 = cute.arch.shuffle_sync_bfly(
                swap_0,
                offset=1,
            )
            peer_1 = cute.arch.shuffle_sync_bfly(
                swap_1,
                offset=1,
            )
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

            swap_0 = stage_0
            swap_1 = stage_1
            if (lane_in_quad & Int32(2)) == Int32(0):
                swap_0 = stage_2
                swap_1 = stage_3
            else:
                swap_0 = stage_0
                swap_1 = stage_1
            peer_0 = cute.arch.shuffle_sync_bfly(
                swap_0,
                offset=2,
            )
            peer_1 = cute.arch.shuffle_sync_bfly(
                swap_1,
                offset=2,
            )
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
            kv_index = context[
                self.CTX_KV_BASE_WORD + n_index
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
                _atomic_add_fp32x4_v1(
                    vector_0,
                    vector_1,
                    vector_2,
                    vector_3,
                    destination_ptr,
                )
        # V1_SPAN_REDUCE_ATOMIC_END
        self.main_barrier.arrive_and_wait()
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _zero_dq_v1(
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
        dq_evict_tiled_mma: cute.TiledMma,
        dq_a_evict_layout_staged: cute.ComposedLayout,
        mdQ_acc: Optional[cute.Tensor],
    ):
        """Serialized two-tile v1 macro used only for correctness bring-up.

        This override is deliberately conservative: all cross-CTA routes
        complete before the next destructive alias transition.  It already
        enforces the v1 traffic contract (one stationary Q/dO load and one K
        load per logical tile), while later candidates may split the stable
        span anchors below across persistent roles without changing data
        ownership.
        """

        _ = problem_shape
        _ = tma_atom_qt
        _ = tma_tensor_qt
        _ = tma_atom_dot
        _ = tma_tensor_dot
        _ = dq_evict_tiled_mma
        _ = dq_a_evict_layout_staged
        _ = mdQ_acc
        _ = mQ
        _ = mdO
        _ = grad_a_stage_bytes
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
        peer_rank = Int32(1) - rank
        token_idx = physical_x // self.CLUSTER_SHAPE_MNK[0]
        is_leader_cta = rank == Int32(0)
        block_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            rank
        )

        if warp_idx == Int32(0):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_do)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        done_mbars = storage.mma_mbars.data_ptr()
        source_mbars = storage.source_mbars.data_ptr()
        route_mbars = storage.route_mbars.data_ptr()
        tmem_holding_buf_ptr = storage.tmem_holding_buf.ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar.ptr

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
        contexts = storage.tile_ctx.get_tensor(
            cute.make_layout(
                (
                    self.TILE_CTX_WORDS,
                    self.CONTEXT_STAGES,
                ),
                stride=(1, self.TILE_CTX_WORDS),
            )
        )
        descriptor = (
            storage.traversal_descriptor.get_tensor(
                cute.make_layout(
                    (self.TRAVERSAL_DESCRIPTOR_WORDS,)
                )
            )
        )
        softmax_stats = storage.stats.get_tensor(
            cute.make_layout(
                (self.H_TILE_CTA, 2),
                stride=(1, self.H_TILE_CTA),
            )
        )
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
        stats_thread_copy = stats_tiled_copy.get_slice(tidx % Int32(32))
        g_scaled_lse = cute.flat_divide(
            scaled_lse,
            (self.H_TILE_CTA,),
        )
        g_sum_odo = cute.flat_divide(
            sum_odo,
            (self.H_TILE_CTA,),
        )
        t_g_scaled_lse = stats_thread_copy.partition_S(
            g_scaled_lse[
                None,
                rank,
                (token_idx, batch_idx),
            ]
        )
        t_s_scaled_lse = stats_thread_copy.partition_D(
            softmax_stats[None, 0]
        )
        t_g_sum_odo = stats_thread_copy.partition_S(
            g_sum_odo[
                None,
                rank,
                (token_idx, batch_idx),
            ]
        )
        t_s_sum_odo = stats_thread_copy.partition_D(
            softmax_stats[None, 1]
        )

        # Each typed 32-KiB operand lane is interpreted first as score K_N,
        # then as the native two-round dQ-A tensor, and finally as dKV-A.
        dq_a_two_round_layout = sm100_utils.make_smem_layout_a(
            dq_tiled_mma,
            self.DQ_MMA_TILER,
            self.element_dtype,
            self.D_ROUNDS,
        )
        assert (
            cute.cosize(dq_a_two_round_layout)
            == cute.cosize(score_b_layout_staged)
        )
        assert (
            dq_a_two_round_layout.inner
            == dq_a_layout_staged.inner
        )
        assert (
            score_b_layout_staged.inner
            == dq_a_two_round_layout.inner
        )

        k_n_0 = storage.op_lane_a.get_tensor(
            score_b_layout_staged.outer,
            swizzle=score_b_layout_staged.inner,
        )
        k_n_1 = storage.op_lane_b.get_tensor(
            score_b_layout_staged.outer,
            swizzle=score_b_layout_staged.inner,
        )
        k_d_0 = storage.op_lane_a.get_tensor(
            dq_a_two_round_layout.outer,
            swizzle=dq_a_two_round_layout.inner,
        )
        k_d_1 = storage.op_lane_b.get_tensor(
            dq_a_two_round_layout.outer,
            swizzle=dq_a_two_round_layout.inner,
        )
        grad_do = storage.op_lane_a.get_tensor(
            dkv_a_layout_staged.outer,
            swizzle=dkv_a_layout_staged.inner,
        )
        grad_q = storage.op_lane_b.get_tensor(
            dkv_a_layout_staged.outer,
            swizzle=dkv_a_layout_staged.inner,
        )

        # P and dS are distinct typed 8-KiB fields.  dS publication uses the
        # verified score-output store alias; dQ and dK consume whole-field
        # dQ-B and dKV-B aliases on opposite sides of the destructive route.
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

        p_h_store_0 = storage.pd_lane_a_p.get_tensor(
            score_store_domain,
            swizzle=score_store_layout.inner,
        )
        p_h_store_1 = storage.pd_lane_b_p.get_tensor(
            score_store_domain,
            swizzle=score_store_layout.inner,
        )
        p_h_0 = storage.pd_lane_a_p.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )
        p_h_1 = storage.pd_lane_b_p.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )
        p_0 = storage.pd_lane_a_p.get_tensor(
            dkv_b_layout_staged.outer,
            swizzle=dkv_b_layout_staged.inner,
        )
        p_1 = storage.pd_lane_b_p.get_tensor(
            dkv_b_layout_staged.outer,
            swizzle=dkv_b_layout_staged.inner,
        )
        ds_h_store_0 = storage.pd_lane_a_ds.get_tensor(
            score_store_domain,
            swizzle=score_store_layout.inner,
        )
        ds_h_store_1 = storage.pd_lane_b_ds.get_tensor(
            score_store_domain,
            swizzle=score_store_layout.inner,
        )
        ds_h_0 = storage.pd_lane_a_ds.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )
        ds_h_1 = storage.pd_lane_b_ds.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )
        ds_n_0 = storage.pd_lane_a_ds.get_tensor(
            dkv_b_layout_staged.outer,
            swizzle=dkv_b_layout_staged.inner,
        )
        ds_n_1 = storage.pd_lane_b_ds.get_tensor(
            dkv_b_layout_staged.outer,
            swizzle=dkv_b_layout_staged.inner,
        )

        atom_thr_size = cute.size(
            score_tiled_mma.thr_id.shape
        )
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
            cute.make_identity_tensor(
                self.DKV_MMA_TILER[:2]
            )
        )
        rank_dq_coordinates = rank_dq_mma.partition_C(
            cute.make_identity_tensor(
                self.DQ_MMA_TILER[:2]
            )
        )

        g_q = cute.local_tile(
            tma_tensor_q,
            cute.select(
                (
                    self.H_TILE_CTA,
                    self.N_TILE,
                    self.D_HEAD,
                ),
                mode=[0, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        g_do = cute.local_tile(
            tma_tensor_do,
            cute.select(
                (
                    self.H_TILE_CTA,
                    self.N_TILE,
                    self.D_HEAD,
                ),
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

        score_q_fragment = score_tiled_mma.make_fragment_A(
            stationary_q
        )
        score_do_fragment = dp_tiled_mma.make_fragment_A(
            stationary_do
        )
        score_k_0_fragment = score_tiled_mma.make_fragment_B(
            k_n_0
        )
        score_k_1_fragment = score_tiled_mma.make_fragment_B(
            k_n_1
        )
        dp_k_0_fragment = dp_tiled_mma.make_fragment_B(k_n_0)
        dp_k_1_fragment = dp_tiled_mma.make_fragment_B(k_n_1)

        dq_k_0_fragment = dq_tiled_mma.make_fragment_A(k_d_0)
        dq_k_1_fragment = dq_tiled_mma.make_fragment_A(k_d_1)
        # dQ consumes the original H-owner dS_H image.  The destructive
        # H->N route is legal only after both dQ rounds are operand-safe.
        dq_ds_0_fragment = dq_tiled_mma.make_fragment_B(ds_h_0)
        dq_ds_1_fragment = dq_tiled_mma.make_fragment_B(ds_h_1)

        grad_do_fragment = dkv_tiled_mma.make_fragment_A(
            grad_do
        )
        grad_q_fragment = dkv_tiled_mma.make_fragment_A(grad_q)
        p_0_fragment = dkv_tiled_mma.make_fragment_B(p_0)
        p_1_fragment = dkv_tiled_mma.make_fragment_B(p_1)
        dk_ds_0_fragment = dkv_tiled_mma.make_fragment_B(ds_n_0)
        dk_ds_1_fragment = dkv_tiled_mma.make_fragment_B(ds_n_1)

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
            tmem_holding_buf_ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )

        if tidx == Int32(0):
            # Typed peer-copy completion barriers remain valid for the whole
            # kernel and advance only by parity.  pipeline_init_arrive emits
            # the mbarrier-init fence that publishes these initializations to
            # the peer CTA before any remote complete_tx can target them.
            for route_mbar_offset in cutlass.range_constexpr(6):
                cute.arch.mbarrier_init(
                    route_mbars + route_mbar_offset,
                    1,
                )
            cute.arch.mbarrier_init(route_mbars + 8, 1)
            cute.arch.mbarrier_init(route_mbars + 9, 1)
        cute.arch.fence_view_async_shared()
        self.main_barrier.arrive_and_wait()

        pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk,
            is_relaxed=False,
        )
        pipeline.pipeline_init_wait(
            cluster_shape_mn=cluster_layout_vmnk,
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

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = mTopkIdxs.shape[0]
        assert self.max_topk % self.N_TILE == 0
        tile_count = self.max_topk // self.N_TILE
        macro_count = (tile_count + 1) // 2

        if warp_idx >= Int32(self.MATH_WARP_COUNT):
            cute.arch.setmaxregister_decrease(48)
        self.main_barrier.arrive_and_wait()
        if warp_idx < Int32(self.MATH_WARP_COUNT):
            cute.arch.setmaxregister_increase(256)
        self.main_barrier.arrive_and_wait()

        # Stationary Q/dO source barriers are initialized here but no dense
        # G2S request is issued until the first nonempty macro is decoded.
        if tidx == Int32(0):
            cute.arch.mbarrier_init(source_mbars, 1)
            cute.arch.mbarrier_init(source_mbars + 1, 1)
        self.main_barrier.arrive_and_wait()

        issued_count = Int32(0)
        lane_a_count = Int32(0)
        lane_b_count = Int32(0)
        stationary_loaded = Int32(0)
        for macro_index in cutlass.range(
            0,
            macro_count,
            1,
            unroll=1,
        ):
            ordinal_0 = macro_index * Int32(2)
            logical_tile_0 = Int32(tile_count - 1) - ordinal_0
            context_0 = contexts[None, Int32(0)]
            context_1 = contexts[None, Int32(1)]
            active_0 = self._decode_ctx_v1(
                mTopkIdxs,
                context_0,
                descriptor,
                token_idx,
                batch_idx,
                topk,
                logical_tile_0,
                ordinal_0,
                tidx,
            )
            active_1 = cutlass.Boolean(False)
            has_lane_1 = (
                ordinal_0 + Int32(1) < Int32(tile_count)
            )
            if has_lane_1:
                logical_tile_1 = (
                    Int32(tile_count - 2) - ordinal_0
                )
                active_1 = self._decode_ctx_v1(
                    mTopkIdxs,
                    context_1,
                    descriptor,
                    token_idx,
                    batch_idx,
                    topk,
                    logical_tile_1,
                    ordinal_0 + Int32(1),
                    tidx,
                )

            macro_active = active_0 | active_1
            if active_0:
                lane_a_count += Int32(1)
            if active_1:
                lane_b_count += Int32(1)

            # Compact issue sequence is independent of raw sparse ordinal.
            # Slots 0/1 are safe to reuse in A0 because the entire macro is
            # retired before the next decode; the role-specialized successor
            # will rotate the same immutable records over TileCtx4.
            if tidx == Int32(0):
                compact_issue = issued_count
                if active_0:
                    context_0[
                        self.CTX_ISSUE_SEQ_WORD
                    ] = compact_issue
                    compact_issue += Int32(1)
                if active_1:
                    context_1[
                        self.CTX_ISSUE_SEQ_WORD
                    ] = compact_issue
            cute.arch.fence_view_async_shared()
            self.main_barrier.arrive_and_wait()

            # Lazy first-use staging is a cluster-uniform equivalent of a
            # has-any-issued prepass: T=0 performs no Q/dO or stats GMEM
            # access, while T>0 performs exactly one generation per CTA.
            if (
                macro_active
                and stationary_loaded == Int32(0)
            ):
                # V1_SPAN_LOAD_QDO_BEGIN
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            source_mbars,
                            score_a_stage_bytes * self.K_CHUNKS,
                        )
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            source_mbars + 1,
                            score_a_stage_bytes
                            * self.K_CHUNKS,
                        )
                    cute.copy(
                        tma_atom_q,
                        t_q_gmem[None, rank, 0],
                        t_q_smem[None, 0],
                        tma_bar_ptr=source_mbars,
                    )
                    cute.copy(
                        tma_atom_do,
                        t_do_gmem[None, rank, 0],
                        t_do_smem[None, 0],
                        tma_bar_ptr=source_mbars + 1,
                    )
                    cute.arch.mbarrier_wait(
                        source_mbars,
                        Int32(0),
                    )
                    cute.arch.mbarrier_wait(
                        source_mbars + 1,
                        Int32(0),
                    )
                cute.arch.fence_view_async_shared()
                self.main_barrier.arrive_and_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                # V1_SPAN_LOAD_QDO_END

                # V1_SPAN_LOAD_STATS_BEGIN
                if tidx < Int32(self.H_TILE_CTA):
                    global_h = (
                        rank * Int32(self.H_TILE_CTA) + tidx
                    )
                    if warp_idx == Int32(0):
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
                    _ = softmax_stats[
                        global_h,
                        1,
                    ]
                cute.arch.fence_view_async_shared()
                self.main_barrier.arrive_and_wait()
                # V1_SPAN_LOAD_STATS_END
                stationary_loaded = Int32(1)

            # V1_SPAN_LOAD_K_BEGIN
            if active_0:
                if (
                    tidx >= Int32(self.KV_LOAD_THREAD_BEGIN)
                    and tidx
                    < Int32(
                        self.KV_LOAD_THREAD_BEGIN
                        + self.KV_LOAD_THREADS
                    )
                ):
                    self._load_k_from_ctx_v1(
                        mKV,
                        context_0,
                        k_n_0,
                        batch_idx,
                        rank,
                        tidx
                        - Int32(self.KV_LOAD_THREAD_BEGIN),
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                self.main_barrier.arrive_and_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            if active_1:
                if (
                    tidx >= Int32(self.KV_LOAD_THREAD_BEGIN)
                    and tidx
                    < Int32(
                        self.KV_LOAD_THREAD_BEGIN
                        + self.KV_LOAD_THREADS
                    )
                ):
                    self._load_k_from_ctx_v1(
                        mKV,
                        context_1,
                        k_n_1,
                        batch_idx,
                        rank,
                        tidx
                        - Int32(
                            self.KV_LOAD_THREAD_BEGIN
                        ),
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                self.main_barrier.arrive_and_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            # V1_SPAN_LOAD_K_END

            if active_0:
                # V1_SPAN_SDP_ISSUE_BEGIN
                if is_leader_cta and warp_idx == Int32(0):
                    done_producer_state = (
                        self._issue_four_chunks(
                            score_tiled_mma,
                            t_score,
                            score_q_fragment,
                            score_k_0_fragment,
                            done_pipeline,
                            done_producer_state,
                        )
                    )
                    done_producer_state = (
                        self._issue_four_chunks(
                            dp_tiled_mma,
                            t_dp,
                            score_do_fragment,
                            dp_k_0_fragment,
                            done_pipeline,
                            done_producer_state,
                        )
                    )
                # V1_SPAN_SDP_ISSUE_END
                # V1_SPAN_MATH_P_BEGIN
                # V1_SPAN_MATH_DS_BEGIN
                done_consumer_state = (
                    self._compute_publish_pd_v1(
                        t_score,
                        t_dp,
                        score_tmem_load,
                        rank_score_coordinates,
                        softmax_stats,
                        scale_softmax,
                        context_0,
                        p_h_store_0,
                        ds_h_store_0,
                        done_pipeline,
                        done_consumer_state,
                        tidx,
                    )
                )
                # V1_SPAN_MATH_P_END
                # V1_SPAN_MATH_DS_END
            if active_1:
                # V1_SPAN_SDP_ISSUE_BEGIN
                if is_leader_cta and warp_idx == Int32(0):
                    done_producer_state = (
                        self._issue_four_chunks(
                            score_tiled_mma,
                            t_score,
                            score_q_fragment,
                            score_k_1_fragment,
                            done_pipeline,
                            done_producer_state,
                        )
                    )
                    done_producer_state = (
                        self._issue_four_chunks(
                            dp_tiled_mma,
                            t_dp,
                            score_do_fragment,
                            dp_k_1_fragment,
                            done_pipeline,
                            done_producer_state,
                        )
                    )
                # V1_SPAN_SDP_ISSUE_END
                # V1_SPAN_MATH_P_BEGIN
                # V1_SPAN_MATH_DS_BEGIN
                done_consumer_state = (
                    self._compute_publish_pd_v1(
                        t_score,
                        t_dp,
                        score_tmem_load,
                        rank_score_coordinates,
                        softmax_stats,
                        scale_softmax,
                        context_1,
                        p_h_store_1,
                        ds_h_store_1,
                        done_pipeline,
                        done_consumer_state,
                        tidx,
                    )
                )
                # V1_SPAN_MATH_P_END
                # V1_SPAN_MATH_DS_END

            # Replace the former direct register-shuffle P_N publication with
            # a whole-field score R2S P_H image plus the already-proven
            # full-capture-before-rewrite H->N route.
            if active_0:
                self._route_ds_full_rewrite_v1(
                    p_h_0,
                    p_0,
                    route_mbars + 2,
                    (
                        lane_a_count - Int32(1)
                    ) & Int32(1),
                    rank,
                    peer_rank,
                    tidx,
                )
            if active_1:
                self._route_ds_full_rewrite_v1(
                    p_h_1,
                    p_1,
                    route_mbars + 3,
                    (
                        lane_b_count - Int32(1)
                    ) & Int32(1),
                    rank,
                    peer_rank,
                    tidx,
                )

            # V1_SPAN_ROUTE_K_BEGIN
            if active_0:
                self._route_k_full_rewrite_v1(
                    k_n_0,
                    k_d_0,
                    route_mbars,
                    rank,
                    peer_rank,
                    tidx,
                )
            if active_1:
                self._route_k_full_rewrite_v1(
                    k_n_1,
                    k_d_1,
                    route_mbars + 1,
                    rank,
                    peer_rank,
                    tidx,
                )
            # V1_SPAN_ROUTE_K_END

            # K_D and the original dS_H now have their final non-aliasing dQ
            # lifetime.  No destructive dS rewrite is legal in this region.
            if active_0:
                for round_index in cutlass.range_constexpr(
                    self.D_ROUNDS
                ):
                    # V1_SPAN_DQ_ISSUE_BEGIN
                    if is_leader_cta and warp_idx == Int32(0):
                        if issued_count == Int32(0):
                            if cutlass.const_expr(round_index == 0):
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[0],
                                        dq_k_0_fragment,
                                        dq_ds_0_fragment,
                                        0,
                                        False,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                            else:
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[1],
                                        dq_k_0_fragment,
                                        dq_ds_0_fragment,
                                        1,
                                        False,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                        else:
                            if cutlass.const_expr(round_index == 0):
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[0],
                                        dq_k_0_fragment,
                                        dq_ds_0_fragment,
                                        0,
                                        True,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                            else:
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[1],
                                        dq_k_0_fragment,
                                        dq_ds_0_fragment,
                                        1,
                                        True,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                    # V1_SPAN_DQ_ISSUE_END
                    done_consumer_state = self._wait_mma(
                        done_pipeline,
                        done_consumer_state,
                        tidx,
                    )
                    # V1_SPAN_DQ_SAFE
                issued_count += Int32(1)
            if active_1:
                for round_index in cutlass.range_constexpr(
                    self.D_ROUNDS
                ):
                    # V1_SPAN_DQ_ISSUE_BEGIN
                    if (
                        is_leader_cta
                        and warp_idx == Int32(0)
                    ):
                        if issued_count == Int32(0):
                            if cutlass.const_expr(
                                round_index == 0
                            ):
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[0],
                                        dq_k_1_fragment,
                                        dq_ds_1_fragment,
                                        0,
                                        False,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                            else:
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[1],
                                        dq_k_1_fragment,
                                        dq_ds_1_fragment,
                                        1,
                                        False,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                        else:
                            if cutlass.const_expr(
                                round_index == 0
                            ):
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[0],
                                        dq_k_1_fragment,
                                        dq_ds_1_fragment,
                                        0,
                                        True,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                            else:
                                done_producer_state = (
                                    self._issue_dq_stage_v1(
                                        dq_tiled_mma,
                                        t_dq[1],
                                        dq_k_1_fragment,
                                        dq_ds_1_fragment,
                                        1,
                                        True,
                                        done_pipeline,
                                        done_producer_state,
                                    )
                                )
                    # V1_SPAN_DQ_ISSUE_END
                    done_consumer_state = self._wait_mma(
                        done_pipeline,
                        done_consumer_state,
                        tidx,
                    )
                    # V1_SPAN_DQ_SAFE
                issued_count += Int32(1)

            # Both dQ rounds are operand-safe for every live macro lane.
            # Only now may dS_H be destructively rewritten as dS_N for dK.
            # V1_SPAN_ROUTE_DS_BEGIN
            if active_0:
                self._route_ds_full_rewrite_v1(
                    ds_h_0,
                    ds_n_0,
                    route_mbars + 4,
                    (
                        lane_a_count - Int32(1)
                    ) & Int32(1),
                    rank,
                    peer_rank,
                    tidx,
                )
            if active_1:
                self._route_ds_full_rewrite_v1(
                    ds_h_1,
                    ds_n_1,
                    route_mbars + 5,
                    (
                        lane_b_count - Int32(1)
                    ) & Int32(1),
                    rank,
                    peer_rank,
                    tidx,
                )
            # V1_SPAN_ROUTE_DS_END

            # K lanes are now dead.  Stream stationary tensors once per D256
            # round, then retain each tile's P/dS through dV and dK.
            for round_index in cutlass.range_constexpr(
                self.D_ROUNDS
            ):
                if macro_active:
                    # V1_SPAN_MAT_DO_BEGIN
                    # V1_SPAN_MAT_Q_BEGIN
                    self._materialize_qdo_round_v1(
                        stationary_q,
                        stationary_do,
                        grad_q,
                        grad_do,
                        route_mbars + 8,
                        route_mbars + 9,
                        round_index,
                        rank,
                        peer_rank,
                        tidx,
                    )
                    # V1_SPAN_MAT_DO_END
                    # V1_SPAN_MAT_Q_END

                if active_0:
                    # V1_SPAN_DVDK_FINAL_ISSUE_BEGIN
                    if is_leader_cta and warp_idx == Int32(0):
                        done_producer_state = (
                            self._issue_dv_dk_final_v1(
                                dkv_tiled_mma,
                                t_dkv[0],
                                grad_do_fragment,
                                p_0_fragment,
                                grad_q_fragment,
                                dk_ds_0_fragment,
                                done_pipeline,
                                done_producer_state,
                            )
                        )
                    # V1_SPAN_DVDK_FINAL_ISSUE_END
                if active_0:
                    done_consumer_state = (
                        self._atomic_dkv_from_context_v1(
                            t_dkv[0],
                            dkv_tmem_load,
                            rank_dkv_coordinates,
                            mdKV_acc,
                            context_0,
                            round_index,
                            batch_idx,
                            tidx,
                            done_pipeline,
                            done_consumer_state,
                        )
                    )
                    # V1_SPAN_DK_SAFE
                if active_1:
                    # V1_SPAN_DVDK_FINAL_ISSUE_BEGIN
                    if (
                        is_leader_cta
                        and warp_idx == Int32(0)
                    ):
                        done_producer_state = (
                            self._issue_dv_dk_final_v1(
                                dkv_tiled_mma,
                                t_dkv[1],
                                grad_do_fragment,
                                p_1_fragment,
                                grad_q_fragment,
                                dk_ds_1_fragment,
                                done_pipeline,
                                done_producer_state,
                            )
                        )
                    # V1_SPAN_DVDK_FINAL_ISSUE_END
                    done_consumer_state = (
                        self._atomic_dkv_from_context_v1(
                            t_dkv[1],
                            dkv_tmem_load,
                            rank_dkv_coordinates,
                            mdKV_acc,
                            context_1,
                            round_index,
                            batch_idx,
                            tidx,
                            done_pipeline,
                            done_consumer_state,
                        )
                    )
                    # V1_SPAN_DK_SAFE

        tmem.relinquish_alloc_permit()
        if issued_count > Int32(0):
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
        else:
            self._zero_dq_v1(
                rank_dq_coordinates,
                mdQ,
                0,
                token_idx,
                batch_idx,
                tidx,
            )
            self._zero_dq_v1(
                rank_dq_coordinates,
                mdQ,
                1,
                token_idx,
                batch_idx,
                tidx,
            )

        # V1_SPAN_TAIL_BEGIN
        self.main_barrier.arrive_and_wait()
        if is_leader_cta and warp_idx == Int32(0):
            done_pipeline.producer_tail(done_producer_state)
        self.main_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)
        # V1_SPAN_TAIL_END

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
        """Materialize the exact v1 224-KiB payload envelope."""

        del default_storage
        assert score_b_layout_staged.inner == dq_a_layout_staged.inner
        assert score_a_layout_staged.inner == dkv_a_layout_staged.inner
        assert dkv_b_layout_staged.inner == dq_b_layout_staged.inner

        stationary_elements = cute.cosize(score_a_layout_staged)
        op_lane_elements = cute.cosize(score_b_layout_staged)
        p_lane_elements = cute.cosize(dkv_b_layout_staged)
        ds_lane_elements = cute.cosize(dq_b_layout_staged)
        assert (
            stationary_elements
            * self.element_dtype.width
            // 8
            == self.STATIONARY_BYTES_PER_TENSOR
        )
        assert (
            op_lane_elements
            * self.element_dtype.width
            // 8
            == self.OP_BYTES_PER_STAGE
        )
        assert (
            op_lane_elements
            == 2 * cute.cosize(dq_a_layout_staged)
        )
        assert op_lane_elements == cute.cosize(
            dkv_a_layout_staged
        )
        assert p_lane_elements == ds_lane_elements

        @cute.struct
        class SharedStorage:
            stationary_q: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    stationary_elements,
                ],
                1024,
            ]
            stationary_do: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    stationary_elements,
                ],
                1024,
            ]
            op_lane_a: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    op_lane_elements,
                ],
                1024,
            ]
            op_lane_b: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    op_lane_elements,
                ],
                1024,
            ]
            pd_lane_a_p: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    p_lane_elements,
                ],
                1024,
            ]
            pd_lane_a_ds: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    ds_lane_elements,
                ],
                1024,
            ]
            pd_lane_b_p: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    p_lane_elements,
                ],
                1024,
            ]
            pd_lane_b_ds: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype,
                    ds_lane_elements,
                ],
                1024,
            ]

            tile_ctx: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.CONTEXT_STAGES * self.TILE_CTX_WORDS,
                ],
                16,
            ]
            traversal_descriptor: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    self.TRAVERSAL_DESCRIPTOR_WORDS,
                ],
                16,
            ]
            stream_state: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint32,
                    self.STREAM_STATE_WORDS,
                ],
                16,
            ]

            mma_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.MMA_MBAR_COUNT,
            ]
            source_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.SOURCE_MBAR_COUNT,
            ]
            route_mbars: cute.struct.MemRange[
                cutlass.Int64,
                self.ROUTE_MBAR_COUNT,
            ]
            stats: cute.struct.MemRange[
                self.acc_dtype,
                self.STATS_WORDS,
            ]
            control: cute.struct.MemRange[
                cutlass.Uint32,
                self.CONTROL_WORDS,
            ]

            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64

        assert (
            self.MMA_MBAR_COUNT
            + self.SOURCE_MBAR_COUNT
            + self.ROUTE_MBAR_COUNT
            <= self.MAX_MBAR_COUNT
        )
        assert SharedStorage.size_in_bytes() == 232_448
        assert SharedStorage.size_in_bytes() <= self.MAX_SMEM_BYTES
        self.v1_storage_report = {
            "stationary_elements": stationary_elements,
            "op_lane_elements": op_lane_elements,
            "p_lane_elements": p_lane_elements,
            "ds_lane_elements": ds_lane_elements,
            "data_payload_bytes": self.DATA_PAYLOAD_BYTES,
            "tile_ctx_ring_bytes": self.TILE_CTX_RING_BYTES,
            "traversal_descriptor_bytes": (
                self.TRAVERSAL_DESCRIPTOR_BYTES
            ),
            "stats_control_bytes": self.STATS_CONTROL_BYTES,
            "mbar_count": (
                self.MMA_MBAR_COUNT
                + self.SOURCE_MBAR_COUNT
                + self.ROUTE_MBAR_COUNT
            ),
            "shared_storage_bytes": SharedStorage.size_in_bytes(),
        }
        return SharedStorage


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
    """v17a (E2 stop-loss gate): baseline-form math loop on the v12 base.

    Fork of v12 changing ONLY the math warpgroup's per-tile loop:
    (1) packed f32x2 pair math mirroring the baseline
    (dsa_bwd_sm100.py:1825-1874); (2) a phased P-then-dS structure
    (compute + publish P, then compute + publish dS).  The pds single
    gate, the count-128 pds_ready arrive, and every pipeline/mbar/
    credit structure are byte-identical to v12.  Acceptance
    (root-cause report 2026-07-31): per-tile sum-MATH_SOFTMAX
    1.87 -> <=1.1; dS-phase MATH_STORE <=0.85; period UNCHANGED
    (ring-model self-check; one pre-excused benign delta mechanism is
    documented at the acceptance-signature comment in the loop).
    MATH_SOFTMAX / MATH_STORE emit two instances per tile, payload 2i
    (P) / 2i+1 (dS); other math spans keep payload = i.

    IKET name SET unchanged from v12 (31 names, chair-frozen).  NOTE
    (readout plan): the current toolchain (CuTe DSL 4.6.1) caps IKET
    name registration at 29 -- the 30th registration aborts trace
    compile (vm5probe first-run lesson); v12's 31 names passed only on
    the old toolchain, so a straight trace build of this file is
    expected to STOP at trace compile.  Run release smoke + the math
    SASS gates first (neither needs a trace); collect the acceptance
    span window from a vm6probe-style variant that retires >=2
    non-signature spans (MATH_PDS_ACQ/MATH_BAR1 is the established
    swap) per the report's "vm6probe window i=14-17" channel.

    Inherited v12 notes:

    v12: per-launch tail + pds-ring surgery on the v11 base.

    v9's two terminal levers (credit-gated peer push; CONFIG B bulk-
    reduce drain) were both condemned by hardware economics (24.63 /
    18.36 ms; correctness passed twice, so the protocols were sound --
    the end-to-end DSM/engine costs were not).  v9.3 returns to the v8
    base (11.945 ms) and lands the one remaining SASS-diagnosed lever:
    the P/dS publish lowered to 96 scalar STS.U16 + 96 PRMT per warp
    per tile (ZERO stmatrix; ~4.6 us real) because the default S/dP T2R
    atom is not 16-DP, so get_smem_store_op fell back to
    CopyUniversalOp.  The fix forces Ld16x256b(Rep 4) (host-probe
    verified ownership: 4 h-rows x 8 n-cols per thread, n-half still
    warp-uniform, quad structure identical to the drain's), which fires
    the stmatrix m8n8.x4.trans branch; a build-time assert makes the
    premise a trace failure instead of a silent regression.  Softmax
    stats indexing becomes group-hoisted and coordinate-derived; the
    four zero-information S/dP ACQUIRE/PUBLISH spans are retired so the
    IKET name count is exactly 31.

    Inherited v8 notes:

    v8: spill fix + kdq offload + per-slot drain cadence.

    The v7 tile-1 critical path (trace + SASS forensics) put the excess in
    exactly three cells: (1) the math publish ran 6.1 us against a ~1.5 us
    architectural tax -- the SASS shows REG=96 launch, STACK=1384 B/thread,
    and 39 STL + 43 LDL between the softmax fragment materialization and
    the first math barrier: a confirmed register spill at the 128-reg
    warpgroup budget; (2) the W17 supply chain serialized ~10 us/tile
    (ROUTE_K ~4.9 real single-warp gather + 8 credit-coupled quadrant
    fills); (3) v7's fused single dkv generation pushed slot-0's T2R to
    the tail commit.  v8 attacks all three:

    * Register rebalance: math warpgroup increase(176), reduce warpgroups
      increase(104), gather/leader decrease(48).  The setmaxnreg pool is
      the CTA's launch allocation (640*96 = 61,440), so the dec supply
      (12,288) must cover the inc demand exactly -- 176/104 balances to
      the register, the same invariant v7's 48/128 satisfied.  Reduce
      warps may spill mildly at 104; they carry ~6 us of slack.  MATH_PD
      gains SOFTMAX/PDS_ACQ/STORE/BAR1 sub-spans so the fix is directly
      readable from the span tables.
    * kdq offload: the four gather warps (idle ~80%) write both K_dQ
      images through a two-phase named-barrier handshake while W17 keeps
      sole ownership of the round pipeline ops; the gather loop is
      software-pipelined one tile ahead so the credit-gated kdq handshake
      never delays the next tile's score gather.  The khot advisory
      machinery is retired (the kdq readers now ARE the warps that just
      loaded the rows).
    * Per-slot dkv_done generations return (head/tail commits, 2 stages),
      restoring the slot-0 T2R head start and the leader's acquire slack,
      while the reducer keeps the v7 fused savings that mattered: shared
      KV-index preload, fused atomic section, no reduce_sync_barrier.
      REDUCE_T2R/REDUCE_ATOMIC/WAIT_dK payloads return to two per tile,
      which also restores the trace-table contract.

    Deferred to v9 (design risk too high for one shot): peer-half quadrant
    fills via cluster DSM push (needs a receiver-armed expect_tx protocol),
    and the CONFIG B bulk-reduce drain endgame.

    Inherited v7 notes:

    v7: aggressive cross-tile pipeline reordering (de-convoy).

    v6 measured 15.27 ms with drain code parity (5.02 vs baseline 4.80 us)
    and proved the drain is FIXED-COST bound: halving the atomic volume
    changed nothing, and the 8.6 us period carried ~3.6 us/tile of convoy
    cost above the 5.0 us drain floor -- three single-stage handoffs
    (kscore, S/dP TMEM credit, dkv per-round generations) each serialized
    a full latency into the ring while the MMA track sat 16.5% busy.  v7
    attacks all three with cross-tile overlap and NO new SMEM:

    * Leader reorder: S(t) and dP(t) both issue BEFORE the previous tile's
      gradient block, so kscore releases immediately and gather(t+1) runs
      concurrently with grads(t-1)/math(t) -- the gather leaves the
      critical ring entirely.
    * S/dP TMEM ping-pong: CG2 per-SM M_MMA=64 accumulators fold into 128
      datapaths x 32 columns (CUTLASS mma_traits_sm100_frag.hpp UMMA_2SM
      M64 Interleaved atom), so stage-1 buffers fit the existing holes at
      columns 32/96.  With 2-stage s_done/dp_done pipelines, S(t+1) no
      longer waits for math's T2R of S(t).
    * Fused single-generation dKV drain: one dkv_done generation per tile
      (acquire in the grads head, commit after the round-1 passes); the
      reducer does one wait, back-to-back slot T2Rs, ONE fence, ONE
      release, both atomic bursts, and drops the inherited
      reduce_sync_barrier.  Halves the drain's fixed per-call costs --
      the quantity v6 proved dominant.  dkv_done depth drops to 1 for
      TMEM-alias safety (one generation now covers both slots).

    Inherited v6 notes:

    v5 measured 15.16 ms: the preload alone recovered 0.77 us/tile but the
    drain stayed at 6.34 us (1.29x baseline for HALF the scalars) and the
    REDUCE role remained the sole pacer (99% busy) with every other role's
    waits chained to it.  v6 removes the remaining per-scalar deficit by
    running the production store_dKV path unchanged on our [D128, N64]
    slots (identical shape): Ld16x256b(Rep 4) T2R, register-gathered quads
    without shuffles, thread-group scrambled 16B red.global, panel index
    2*round + rank, plus the matching baseline convert decode.  If the
    drain lands near half of baseline's 4.9 us, the convoy unwinds and the
    period is set by the supply/math legs; if it does not, cross-token
    same-address contention caps every 2-CTA rearrangement and the
    architecture question is answered negatively.

    Inherited v5 notes:

    v4 measured 15.05 ms (1.88x): the leader issue cost collapsed to 0.26-
    0.57x of baseline, which exposed the dKV drain as the sole pacer
    (REDUCE 99.4% busy, 7.11 us/tile vs baseline 4.53 for HALF the atomic
    scalars).  The 3.5x per-scalar inefficiency traces to the ported
    reducer reading each KV index inside the per-vector loop, serialized
    behind the shuffle/branch chain; the baseline preloads all eight
    indices as independent loads up front.  v5 mirrors the baseline
    preload.  If the drain then reaches baseline per-scalar efficiency,
    the halved traffic puts the architecture's ceiling at ~5.5-6.5 ms;
    if not, cross-token same-address contention caps every 2-CTA design.

    Inherited v4 notes:

    v3 measured 14.28 ms (1.77x): the remaining pacers were the leader's
    UMMA issue cost (73 ns/atom vs the baseline's 22 at 48 registers), the
    cold-line K_dQ gather racing ahead of the score gather, and fragmented
    64-row TMA boxes for the gradient panels.  v4 raises the issue
    warpgroup to 80 registers, switches the S/dP issue loop to the
    baseline-shaped runtime unroll-4 form, orders K_dQ behind a rows-hot
    signal from the score gather, and fills h==rank panels with one local
    16 KiB bulk copy of the byte-identical stationary slice.

    Inherited v3 notes:

    Forked from the native v2 lineage at 414217e (my 08001ad design + the
    B200 correctness fixes + native IKET + vectorized sender stores).  The
    trusted preopt_1dbab0b trace showed the round-region supply chain was
    the wall (MAT_QDO 15.6us + ROUTE_K 5.6us of serial fill round-trips per
    tile); v3 software-pipelines the panel TMA fills over two rotating
    barriers, fuses the two K_dQ gathers into one indexed pass, and splits
    stationary readiness so S gates only on Q.

    Design contract: T3细粒度全f32转置设计_20260803.md (main form) +
    T32_BUILD_ADDENDUM.md (overrides) + 转置取向终裁_20260803.md 3.1.
    Relative to the v17a data plane the score plane is TRANSPOSED (heads
    fall on the N axis) and every tensor stays in the CTA that produced it:

    * S^T/dP^T are CG2 M=kv128 with the chased K gather piece as natural
      M-split A (V==K, single fetch, K-outer piece order) and the resident
      Q/dO panels as K-major zero-copy N-split B (32 heads/CTA/chunk).
    * softmax constants (lse, delta) index the COLUMN axis, so the math
      warps take warp-uniform stat loads; P/dS publish as bf16 stmatrix
      sub-images [own-kv64 x h64] stacked chunk-major, plus the dq_b
      own-half second image and the 8,192B peer-half push (bulk DSM
      fallback; st.async direct write is a registered V32-TODO upgrade).
    * dV/dK reduce H128 as TWO CG2 passes of K=h64 accumulating into the
      per-tensor TMEM slot; each completed [own-kv64 x D128] block is
      drained rank-owned by the reducer warps (T2R + f32 GMEM atomics),
      8 drain trips per bundle.
    * All gradient B/A stream operands ([h128 x own-D64] Q/dO rounds and
      [own-D128 x kv64] kdq tiles) come through a rank-symmetric 2x16 KiB
      round region with ONE producer (the load warp) and ONE consumer (the
      leader MMA warp): a single 2-stage pipeline carries all twelve
      generations per bundle, so no barrier ever skips a phase.
    * dQ^T accumulates in place across the query tile (256 persistent
      TMEM columns); G5 kv-waves gate on the mb_dqb[h] cluster gates and
      mb_dqb_free[h] hangs on the LAST consuming wave (both D-rounds,
      addendum errata #1).
    * The leader is dr-major; the chase pieces for bundle t+1 are pinned
      at the G3(c1,0)(t) wait point, keeping the tensor cores gap-free.
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
    IDLE_WARP = 19

    GATHER_THREADS = GATHER_WARPS * 32
    MATH_THREAD_BEGIN = MATH_WARP_BEGIN * 32
    MATH_THREADS = MATH_WARPS * 32
    REDUCE_THREAD_BEGIN = REDUCE_WARP_BEGIN * 32
    REDUCE_THREADS = REDUCE_WARPS * 32

    # v5 tiling4 score plane (V5_TILING4_DEMO_SPEC Z1/Z3).
    #
    # G1/G2 score plane (S^T = K.Q^T / dP^T = V.dO^T), CG2:
    #   (M,N,K) = (kv128, h32-sub-tile, D64-piece), head-outer: pass t
    #   re-streams the SAME eight chased K/V pieces (L2-hot for t >= 1)
    #   and each piece feeds G1(t), G2(t) only (V==K single fetch).  B
    #   is the resident Q/dO panel through a K-major zero-copy h16
    #   window per CTA (the N-half of the N32 MMA).
    #
    # Sub-tile head map (forced by the FROZEN panel residency -- CTA r
    # holds H[c*64+r*32 : +32) for c in {0,1} -- and the hardware CG2
    # B N-half split): sub-tile t = (c, j) with c = t//2, j = t%2;
    # fragment column n in [0,32) is head
    #   head(t, n) = c*64 + (n//16)*32 + j*16 + (n%16),
    # i.e. columns [0,16) come from CTA0's panel rows [j*16,+16) of
    # panel stage c and columns [16,32) from CTA1's.  Every sub-tile is
    # two h16 boxes, 32 heads apart, inside the NATURAL h64 chunk image
    # -- which is what keeps the P/dS slab chunk images, the dq_b
    # image, and the 8,192 B relay payload byte-identical to v32 (the
    # frozen relay/G5/epilogue paths' precondition).
    SCORE_MMA_TILER = (128, 32, 64)
    SCORE_MMA_N = 32
    SCORE_D_PIECES = 8
    # SCORE_H_CHUNKS survives as the CONTAINER count (2 natural h64
    # chunk images: panel stages, P/dS slab images); the SCHEDULING
    # unit is now the h32 sub-tile (SUB_TILES).
    SCORE_H_CHUNKS = 2
    SUB_TILES = 4
    SUB_TILE_H = 32
    SUB_TILE_BOX = 16
    # Per-thread math fragment: (kv64 x h32) f32 folded to 128 DP x 16
    # TMEM columns = 2,048 values / 128 math threads.
    SUB_TILE_VALS = 16

    # v3.2 host-builder contract overrides (see the base-class block):
    # the chase is score A (2-slot ring of [own-kv64 x D64] pieces), the
    # stationary panel becomes the zero-copy score-B view (8 D64 piece
    # windows), the gradient stream moves to the dkv-B operand (GMEM-
    # natural [H,D] TMA), and the dq epilogue stores the transposed tile.
    K_CHUNK = 64
    K_CHUNKS = 8
    SCORE_A_IS_STATIONARY = False
    # v5.1b (item 5): K RESIDENCY.  The 2-slot chase ring retires; the
    # score-A buffer becomes one resident [own-kv64 x D512] image per
    # bundle (65,536 B), staged as 8 D64-piece windows (stage == piece,
    # 8,192 B contiguous blocks).  The gather fills all 8 pieces ONCE
    # per bundle (the v5 per-pass re-gather retires with the ring:
    # -75% gather traffic) under a single 1-stage completion gate
    # (pipe_kres); every score pass window-reads the same residency.
    SCORE_A_STAGES = 8
    # v5.1b (item 6): the score-B container is now the 2-stage Q/dO
    # STRIP buffer (the one-shot dual panels retire).  One strip =
    # [h16 x D512] (16,384 B) = the pass's own CTA window, double-
    # buffered (stage = t % 2).  Byte identity, same algebra as the v5
    # panel windows: a strip stage's k64 blocks are 1,024-element
    # units, so the 16-stage [n16 x k64] score-B layout binds the
    # WHOLE 2-strip buffer at its base and the fragment stage index is
    #   (t % 2) * 8 + piece
    # (strip stage picks the 8,192-element half, piece the k64 block).
    SCORE_B_STAGES = 16
    SCORE_A_MAX_ELEMENTS = 32768
    SCORE_B_MAX_ELEMENTS = 32768
    DKV_A_MAJOR = OperandMajorMode.K
    DKV_B_MAJOR = OperandMajorMode.MN
    DKV_A_STAGES = 2
    # v5.1 (pair batching): the streamed grad gen returns to FULL h64
    # width -- pair P = {t=2P, t=2P+1} unions to EXACTLY chunk P's
    # natural h64 head interval (the two sub-tiles' h16 boxes tile it),
    # so one gen = [h64 x own-D64] (8,192 B, ONE TMA box, the v32
    # single-stage box form) serves BOTH accumulate chains of one
    # (pair, r, tensor) block.  One stage per gen; gen k16-block kb
    # pairs with slab chunk-P column block kb (identical head sets).
    DKV_B_STAGES = 1
    DKV_A_MAX_ELEMENTS = 8192
    DKV_B_MAX_ELEMENTS = 8192
    DKV_B_TILER = (128, 128, 64)
    DQ_B_STAGES = 2
    DQ_B_MAX_ELEMENTS = 8192
    GRAD_STREAM_IS_B = True
    # v17a's dQ accumulator was ALREADY [D x H] (its dq tiler M is the
    # D axis) and _store_dq_epi_tma_v12's SMEM scatter performs the
    # [D,H] -> [H,D] flip before the natural-view TMA, so the v3.2
    # score-plane transposition changes NOTHING here: keep the v17a
    # epilogue orientation.
    DQ_EPI_TRANSPOSED = False
    # v5.1b: the stationary box is one [h16 x D512] STRIP (the pass's
    # CTA window); STATIONARY_STAGES = 2 now means the strip DOUBLE
    # BUFFER (stage = t % 2), streamed per bundle by W17 through
    # pipe_strip -- not the old one-shot panel chunk boxes.
    STATIONARY_TILE_H = 16
    STATIONARY_STAGES = 2

    # Head-axis chunking (fixed, static, NATURAL): h-chunk c covers
    # heads H[c*64:(c+1)*64).  The G1/G2 B N-halves are hardware-
    # assigned per CTA rank; the strip for pass t = (c, j) holds the
    # CTA's h16 window H[c*64 + rank*32 + j*16 : +16) (gmem h16-tile
    # 4c + 2*rank + j).  P/dS slab K-axes, the streamed dO/Q gen rows,
    # the softmax stat indices, dq_b columns (own-H64 =
    # H[rank*64:(rank+1)*64)) and the dQ epilogue all use natural head
    # order -- no permutation anywhere.
    HEAD_CHUNK_GROUP = 32

    # G3/G4 gradient plane (dV += P^T.dO / dK += dS^T.Q), CG2:
    #   (M,N,K) = (kv128, D_c=128, h64-chunk); both h-chunks accumulate
    #   into ONE [own-kv64 x D128] TMEM block per (tensor, D-round); four
    #   D-rounds cover D512.  A = local P/dS slab sub-image (K-major),
    #   B = streamed [h128 x own-D64 N-half] gen (GMEM-natural [H x D]).
    DKV_MMA_TILER = (128, 128, 64)
    DKV_D_ROUNDS = 4
    # H_PASSES is RETIRED in v5 (the h64-chunk grads loop died with
    # _issue_dkv_round_v32; kept for trailer/audit cross-reference).
    H_PASSES = 2

    # v5.2 G5 dQ plane (dQ^T += K^T.dS^T), CG2 -- dQ EVICTION form:
    #   (M,N,K) = (D128-round, h32-block, kv64-wave); SIXTEEN blocks
    #   per bundle (4 D-rounds x 4 h32 windows), each a FRESH rotating
    #   accumulator (2 x 16-column TMEM slots) chained over the two
    #   kv64 waves and immediately offloaded to the f32 dQ workspace
    #   by the reduce warps (plain LDG+FADD+STG; the cluster owns its
    #   token's dQ rows exclusively -- deterministic same-cluster f32
    #   order, NO atomics).  Head map of block (t, r), column n:
    #     head(t, n) = (n//16)*64 + t*16 + (n%16)
    #   (each CTA supplies its dq_b's columns [t*16, +16) -- the only
    #   N-half form compatible with the FROZEN dq_b byte image).
    #   A = kdq stream gen d-half window (stage == d_half; the frozen
    #   [own-D128 x kv64] gen's m-half stride 4096 == the (128,32,64)
    #   2-stage auto layout's stage stride -- order-(2,1,3) algebra);
    #   B = hand-derived dq_b window view, stage tiled (t, wave) with
    #   strides (16, 4096) elements (32 B mid-atom window starts, the
    #   k_block precedent).  Issue order (r_old, t, d_half): the round
    #   ring holds ONE kdq gen pair at a time, so r_old groups are
    #   forced; DQ_EPI payload keeps the ordered b*16 + t*4 + r
    #   encoding (non-monotonic within a bundle, see V5_BUILD_LOG).
    # DQ_MMA_TILER keeps the LEGACY (256,128,64) value: it feeds only
    # the _zero_dq_v2 coordinate decode (tile_count == 0 path) and the
    # dormant dq-epi machinery.
    DQ_MMA_TILER = (256, 128, 64)
    DQ_EVICT_TILER = (128, 32, 64)
    DQ_D_ROUNDS = 2
    DQ_KV_WAVES = 2
    DQ_EVICT_ROUNDS = 4
    DQ_EVICT_WINDOWS = 4
    DQ_EVICT_SLOTS = 2

    # P/dS publish geometry: per h-chunk one [own-kv64 x h64] bf16
    # sub-image (H-contiguous 128B rows, SW128B), stacked chunk-major so
    # G3/G4 h-chunk descriptor windows land on 8,192B boundaries and the
    # dS peer half is one contiguous bulk-DSM payload.
    PDS_BLOCK_ELEMENTS = 4_096
    PDS_BLOCK_BYTES = 8_192

    # v5.2 TMEM 512-column map (all f32, per CTA), allocated UNGUARDED
    # ([fix-r7] structural: 32 + 64 + 256 = 352 <= 512, [352,512)
    # free -- 160 columns of unallocated headroom, booked):
    #   dQ rot [0,32):   2 x 16-column ROTATING eviction slots (block
    #                    g -> slot g%2; 16 blocks/bundle, even => the
    #                    slot pattern restarts per bundle, static);
    #   S pp [32,64) / dP pp [64,96): 2 stages x 16 cols each (M128
    #                    CG2 fold => 128 DP x 16 columns per h32);
    #   dV/dK army [96,352): FOUR 64-column slots.  [fix-r7]: 16
    #                    blocks/bundle mod 4 == 0, so the slot phase
    #                    resets every bundle and block k = pair*8 +
    #                    2r + p gives slot = k % 4 = (2r + p) % 4 --
    #                    a COMPILE-TIME constant (pair*8 mod 4 == 0):
    #                    dV slots {0,2,0,2}, dK slots {1,3,1,3} over
    #                    the rounds.  Static offsets fold alignment
    #                    automatically (the r7 DSL ruling forbids
    #                    .align on TMEM pointers; runtime addressing
    #                    is retired).  Four slots = the double-
    #                    warpgroup pipeline's necessary floor (2 in
    #                    drain + 2 in flight); the drain throughput
    #                    is the pacer, so the 6-slot window's extra
    #                    runway (~0.7 us/pair) is forfeit by ruling.
    TMEM_DQ_SLOT0 = 0
    TMEM_DQ_SLOT1 = 16
    TMEM_S_OFFSET = 32
    TMEM_S1_OFFSET = 48
    TMEM_DP_OFFSET = 64
    TMEM_DP1_OFFSET = 80
    TMEM_DKV_BASE = 96
    TMEM_DKV_SLOT_COLS = 64
    TMEM_DKV_SLOTS = 4
    # TMEM budget echo-assert (host side, __call__).
    TMEM_BUDGET = 32 + 64 + 4 * 64

    # v7: S/dP TMEM double-buffer depth (cross-chunk: S(c+1) no longer
    # waits for math's T2R of S(c)).
    SCORE_DONE_STAGES = 2

    # v5.1 round-region generations per bundle (fixed order, one
    # producer/one consumer, 2-stage ring; 20 mod 2 == 0 keeps the
    # phase law).  The G5 waves stay at the END of the leader's bundle
    # (v5 spec Z3), so FIFO consistency of the single ring keeps the
    # kdq generations behind the grad gens on the producer side:
    # g0..g15: dO(P,r)(A) / Q(P,r)(B) pairs, pair-major then r then
    #          tensor [grads(pair P) D-round r; one FULL-WIDE
    #          [h64 x own-D64] gen per (P, r, tensor) block; dO on
    #          even gens = buf A, Q odd = B]
    # g16 kdq_r0w0(A)  g17 kdq_r0w1(B)   [G5 D-round 0, kv waves 0/1]
    # g18 kdq_r1w0(A)  g19 kdq_r1w1(B)   [G5 D-round 1, kv waves 0/1]
    # The gather<->W17 kdq named-barrier RENDEZVOUS sequence is
    # unchanged (r0(b), r1(b), r0(b+1), ... globally), so the frozen
    # kdq fill machine is untouched.
    ROUND_GENS_PER_TILE = 20
    ROUND_STAGES = 2

    # v5.2 slot army, [fix-r7] FOUR-deep: one generation per
    # [own-kv64 x D128] block, slot == pipeline stage == (2r + p) % 4
    # (static; 16 blocks/bundle mod 4 == 0 keeps stage and slot in
    # permanent lockstep).  The reducer consumes generations in
    # (dV, dK) pairs (fused drain); 4 slots = 2 draining + 2 in
    # flight, the double-warpgroup floor.
    MMA_DONE_STAGES = 4

    # v9.3 hoisting is DISABLED for v3.2: the transposed plane's
    # constants live on the COLUMN axis, and the Ld16x256b(Rep4)
    # fragment's (a1, L) groups are constant in the ROW coordinate only
    # -- heads vary within each group, so per-group hoisting is invalid.
    # The assumption-free per-value column lookup (packed pairs with
    # DISTINCT pair constants) is the v3.2 form.
    # V32-TODO(perf): derive the column-axis group structure of the
    # Rep4 fragment and re-enable a hoisted variant if it exists.
    SOFTMAX_GROUPED_STATS = False

    # v4's OWN_HALF_BULK panel optimization is structurally dead in
    # v3.2 (a round gen's own-head half is not a contiguous panel slice
    # under the [h128 x own-D64] B geometry); it was removed together
    # with the dormant v17a issue helpers.

    # Source-native IKET names that distinguish the active V2 kernel from
    # dormant bring-up classes patched by the external trace harness.
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
        # v8 kdq offload handshake: the four gather warps plus the load
        # warp rendezvous twice per tile (A: W17 holds both g0/g1 round
        # credits; B: the gather warps' K_dQ fills are written and fenced).
        self.kdq_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=(self.GATHER_WARPS + 1) * 32,
        )

    def _make_score_tmem_load(self, score_cta_shape, score_epi_tile):
        """v9.3 atom family, v5 repetition: 16-DP/256-bit T2R for S/dP.

        get_smem_store_op keys the publish store atom off the T2R atom's
        thread-value ownership; the default (non-16-DP) choice made it
        fall back to CopyUniversalOp -- the v8 SASS showed 96 scalar
        STS.U16 + 96 PRMT per warp per tile (~4.6 us real) and ZERO
        stmatrix.  The v5 sub-tile fragment is 128 DP x 16 TMEM columns
        (h32 fold), so the SAME Ld16x256b family carries Repetition(2):
        one op = 16 DP x (256b = 8 f32) x 2 reps = the full 16-column
        window (lesson #15: derive within the family, never hand-write
        the T2R geometry).  DSL source verified (blackwell_helpers
        get_smem_store_op): num_rep in (2,4,8,16,32) still fires
        use_stmatrix_m8n8_4x's f32->bf16 clause -> StMatrix8x8x16bOp
        with num_matrices == 4, so the v9.3 build gate below holds
        unchanged.
        """

        return cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(
                tcgen05.copy.Repetition(2)
            ),
            self.acc_dtype,
        )

    @staticmethod
    def _get_workspace_size_LSE_OdO(
        q: int,
        d: int,
        h: int,
        b: int,
        acc_dtype,
    ):
        """v5.2: extend each (h, q) workspace entry by one f32 D-row.

        Layout: [sum_OdO vectors][scaled_LSE vectors][dQ f32 partial
        sums, D per (h, q)].  [fix-r9] The claim that the harness
        allocates via impl_cls was WRONG: sha forensics show staging
        patches _interface_sm100.py with unknown class-wiring, and the
        r8 IMA proves the base 8 B/entry sizing reached the allocator
        (the decode bug alone cannot leave the region: d_g < D always).
        Allocation sufficiency is now enforced at the interface's
        torch.zeros site itself (entry = max(impl, 8 + D*4)); this
        override remains correct-but-not-load-bearing.  Cross-run
        safety does NOT rely on the zero fill: the first bundle's
        eviction stores directly (no LDG).
        """

        d_r = (d + 7) // 8 * 8
        q_r = (q + 7) // 8 * 8
        acc_bytes = acc_dtype.width // 8
        workspace_bytes = 2 * acc_bytes + d_r * acc_bytes
        return (b, h, q_r, workspace_bytes)

    def _carve_dq_acc(
        self,
        workspace_LSE_OdO,
        problem_shape,
        total_seqlen_Q,
    ):
        """v5.2: carve the f32 dQ partial-sum view from the extended
        LSE/OdO workspace tail (indexed [head, d, (token, batch)])."""

        H = cute.size(problem_shape[3][0])
        D = cute.round_up(problem_shape[2], 8)
        q_r = cute.round_up(total_seqlen_Q, 8)
        acc_bytes = self.acc_dtype.width // 8
        # [fix-r9] No host-side allocation gate is possible here: the
        # interface's compile_key excludes total_S_q, so the workspace
        # layout is dynamic at trace time and cute.cosize(...) yields a
        # dynamic Boolean that cannot enter host control flow (the r9
        # compile abort; lesson #17).  Sufficiency is instead guaranteed
        # at the allocation site: _interface_sm100.py sizes the entry as
        # max(impl entry, 8 + D*4) unconditionally, so the dQ tail below
        # exists no matter which impl class the harness staged.
        base_bytes = cute.assume(
            2 * H * q_r * acc_bytes,
            divby=64,
        )
        dq_iter = cute.recast_ptr(
            workspace_LSE_OdO.iterator + base_bytes,
            dtype=self.acc_dtype,
        )
        return cute.make_tensor(
            dq_iter,
            cute.make_layout(
                (H, D, (q_r, 1)),
                stride=(
                    D,
                    1,
                    (cute.assume(H * D, divby=64), 0),
                ),
            ),
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

        # v5.1b byte-exact plan (per CTA), cap 232,448 (item 7):
        #   Q strip double buffer (2 x [h16 x D512])   32,768
        #   dO strip double buffer                     32,768
        #   K residency [own-kv64 x D512] (8 pieces)   65,536
        #   round region (2 x 16,384)                  32,768
        #   P slab (2 x [kv64 x h64] chunk-major)      16,384
        #   dS slab (2 x [kv64 x h64] chunk-major)     16,384
        #   dq_b dual sub-image (2 x [kv64 x H64])     16,384
        #   softmax stats (lse+delta, h128, f32)        1,024
        #   mbarriers / holding buf                  <= 1,024
        #   total                                     215,040
        # Upper bounds echo the staged-layout cosizes in ELEMENTS (bf16):
        #   score_a: K residency, 8 D64-piece stages       = 32,768
        #   score_b: strip-buffer zero-copy view           = 16,384
        #   dkv_a:   P/dS slab, 2 chunk sub-images         = 8,192
        #   dkv_b:   full-wide round gen [h64 x own-D64]   = 4,096
        #   dq_a:    kdq gen [own-D128 M-half x kv64]      = 8,192
        #   dq_b:    dual sub-image 2 x [kv64 x own-H64]   = 8,192
        assert cute.cosize(score_a_layout_staged) <= 32768
        assert cute.cosize(score_b_layout_staged) <= 16384
        assert cute.cosize(dkv_a_layout_staged) <= 8192
        assert cute.cosize(dkv_b_layout_staged) <= 8192
        assert cute.cosize(dq_a_layout_staged) <= 8192
        assert cute.cosize(dq_b_layout_staged) <= 8192

        @cute.struct
        class SharedStorageV2:
            # Pipeline barrier arrays (full+empty per stage).
            s_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            dp_done_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            # v5.1b: the K residency completion gate (1 stage,
            # full+empty) -- the 2-slot chase ring is retired.
            kres_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            # v5.1b: the Q/dO strip stream (2 stages, full+empty per
            # stage; one generation = the pass's Q+dO strip pair).
            strip_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            round_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            # v5 (spec Z4): pds is a 2-stage pipeline now (full+empty
            # per stage).
            pds_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            # v5.2 slot army ([fix-r7] 4-stage dkv_done ring,
            # full+empty per stage).
            dkv_done_mbars: cute.struct.MemRange[cutlass.Int64, 8]
            # v5.2: the dQ eviction handoff (2 stages, full+empty per
            # stage; leader UMMA producer -> reduce consumers).
            dq_evict_mbars: cute.struct.MemRange[cutlass.Int64, 4]
            # v5.2: DORMANT (the dQ TMA epilogue and its dq_done gate
            # retired with the eviction; kept for byte-account
            # stability).
            dq_done_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            # Raw single-phase-per-tile barriers.
            stationary_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            stationary_ready_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            landing_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            relay_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            round_tma_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            # v12 (P2i): math -> relay-warp publish handoff (count-128).
            pds_ready_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            # v3.2 dq_b cluster gates (addendum section 1): mb_dqb[h] =
            # half h ready (local stmatrix commit AND peer landing, both
            # relay-arrived per errata #2); mb_dqb_free[h] = released by
            # the LAST consuming G5 wave_h (BOTH D-rounds, errata #1).
            dqb_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            dqb_free_mbars: cute.struct.MemRange[cutlass.Int64, 2]
            khot_seq: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

            # v5.1b: Q/dO strip double buffers (2 x [h16 x D512]
            # per tensor, 32,768 B each -- the one-shot panels are
            # retired).
            stationary_q: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 16384],
                1024,
            ]
            stationary_do: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 16384],
                1024,
            ]
            # v5.1b: K residency [own-kv64 x D512] (65,536 B, 8
            # D64-piece stages; single buffer under pipe_kres).
            score_kv: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 32768],
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
            # P publish slab: chunk-major stacked [own-kv64 x h64] bf16
            # sub-images (G3-A h-chunk windows land on 8,192 B bounds).
            p_slab: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 8192],
                1024,
            ]
            # dS publish slab: same stacking; sub[1-rank] doubles as the
            # contiguous 8,192 B bulk-DSM source for the dq_b peer push.
            ds_slab: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 8192],
                1024,
            ]
            # dq_b dual sub-image (addendum section 1): sub_img[w] =
            # kv[w*64:(w+1)*64] rows x own-H64, MN-major (H-contiguous
            # 128 B rows, SW128B), strictly rank-symmetric offsets;
            # sub[rank] = own local image, sub[1-rank] = peer landing.
            dq_b: cute.struct.Align[
                cute.struct.MemRange[element_dtype, 8192],
                1024,
            ]
            # Column-axis softmax stats: lse[h128] then delta[h128].
            stats: cute.struct.Align[
                cute.struct.MemRange[Float32, 256],
                1024,
            ]

        assert SharedStorageV2.size_in_bytes() <= self.MAX_SMEM_BYTES
        # v5.1b SELF-CHECK (item 7): the plan above must fit the new
        # total (upper-bound form per the audit rule -- exact-size
        # asserts are brittle against alignment padding; the actual
        # value is echoed on failure).
        assert SharedStorageV2.size_in_bytes() <= 215_040, (
            SharedStorageV2.size_in_bytes()
        )
        return SharedStorageV2

    # ------------------------------------------------------------------
    # Small helpers cloned from the v1 bring-up (sibling class; the v2
    # schedule reuses only these verified leaf routines).
    # ------------------------------------------------------------------

    @cute.jit
    def _chase_slot_rows_v32(
        self,
        tensor: cute.Tensor,
        slot: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        """[kv64 rows, D64] gather view of one chase ring slot.

        Mirrors the v17a _load_score_kv destination composition: the
        flat (row, d) coordinate indexes the staged score-A slot's
        canonical (M, K) space, so the slot's own swizzle keeps the
        128 B row segments physically contiguous for the 16 B cp.async
        chunks.
        """

        return cute.composition(
            tensor[None, None, None, slot],
            cute.make_layout(
                (2 * self.N_TILE_CTA, self.K_CHUNK)
            ),
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
        dq_round: Int32,
        topk: Int32,
        rank: Int32,
        role_tidx: Int32,
        thread_count: cutlass.Constexpr[int],
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ) -> None:
        """v3.2: gather BOTH kv-wave kdq images of ONE dQ D-round.

        G5's A operand is the kdq stream [own-D128 M-half x kv64]:
        kd_rows_0 receives wave 0 (bundle rows kv[0:64)), kd_rows_1
        wave 1 (kv[64:128)), both at column offset 256*dq_round +
        128*rank (the CTA's M-half of the round).  Same per-row protocol
        as the v8 fill (one index read per row per wave, 256-byte slice,
        zero-fill for invalid rows), partitioned over `thread_count`
        threads in KV_GROUP_SIZE=8 groups.
        """

        index_in_group = role_tidx % self.KV_GROUP_SIZE
        group_index = role_tidx // self.KV_GROUP_SIZE
        groups_total = thread_count // self.KV_GROUP_SIZE
        d_offset = (
            dq_round * Int32(self.D_TILE_CLUSTER)
            + rank * Int32(self.D_TILE_CTA)
        )
        bundle_rows = 2 * self.N_TILE
        wave_rows = self.N_TILE
        assert wave_rows % groups_total == 0
        rows_per_group = wave_rows // groups_total
        for row_iteration in cutlass.range_constexpr(rows_per_group):
            local_n = (
                row_iteration * groups_total + group_index
            )
            for wave in cutlass.range_constexpr(self.DQ_KV_WAVES):
                global_n = (
                    tile_index * Int32(bundle_rows)
                    + Int32(wave * wave_rows)
                    + Int32(local_n)
                )
                kv_index = Int32(-1)
                if global_n < topk:
                    kv_index = mTopkIdxs[
                        global_n,
                        (token_idx, batch_idx),
                    ]
                if cutlass.const_expr(wave == 0):
                    if kv_index >= Int32(0):
                        self._copy_sparse_k_row_v32(
                            mKV,
                            kd_rows_0,
                            Int32(local_n),
                            kv_index,
                            batch_idx,
                            d_offset,
                            index_in_group,
                            self.D_TILE_CTA,
                            copy_atom,
                            thread_copy,
                        )
                    else:
                        self._zero_sparse_k_row_v32(
                            kd_rows_0,
                            Int32(local_n),
                            index_in_group,
                            self.D_TILE_CTA,
                        )
                else:
                    if kv_index >= Int32(0):
                        self._copy_sparse_k_row_v32(
                            mKV,
                            kd_rows_1,
                            Int32(local_n),
                            kv_index,
                            batch_idx,
                            d_offset,
                            index_in_group,
                            self.D_TILE_CTA,
                            copy_atom,
                            thread_copy,
                        )
                    else:
                        self._zero_sparse_k_row_v32(
                            kd_rows_1,
                            Int32(local_n),
                            index_in_group,
                            self.D_TILE_CTA,
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
        dq_round: Int32,
        topk: Int32,
        rank: Int32,
        role_tidx: Int32,
        copy_atom: cute.CopyAtom,
        thread_copy: cute.TiledCopy,
    ) -> None:
        """Gather-side half of ONE v3.2 kdq handshake (one D-round).

        v3.2 runs TWO handshakes per bundle: round 0 (gens g0/g1, the
        two kv-wave images) right after the bundle's own chase, round 1
        (gens g10/g11) after the NEXT bundle's chase, so the late round-1
        credits (freed near the end of grads(t)) never delay the chase.
        Barrier A: the load warp holds both round-stage credits, so the
        round buffers are safe to overwrite.  The gather warps then write
        both kv-wave K_dQ images, drain their own cp.async groups, fence,
        and barrier B hands the generations back for the load warp's two
        commits.
        """

        self.kdq_barrier.arrive_and_wait()
        self._fill_kdq_pair_v8(
            mKV,
            mTopkIdxs,
            kd_rows_0,
            kd_rows_1,
            token_idx,
            batch_idx,
            tile_index,
            dq_round,
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
        dq_evict_tiled_mma: cute.TiledMma,
        dq_a_evict_layout_staged: cute.ComposedLayout,
        mdQ_acc: Optional[cute.Tensor],
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
        # v5.1b: stationary_ready_mbar is DORMANT (the one-shot panel
        # readiness gates retired with the panels; kept initialized so
        # the mbar-init block stays byte-stable).
        stationary_ready_mbar = storage.stationary_ready_mbar.data_ptr()
        landing_mbars = storage.landing_mbars.data_ptr()
        relay_mbars = storage.relay_mbars.data_ptr()
        pds_ready_mbars = storage.pds_ready_mbars.data_ptr()
        round_tma_mbars = storage.round_tma_mbars.data_ptr()
        # v3.2 dq_b cluster gates.
        dqb_mbars = storage.dqb_mbars.data_ptr()
        dqb_free_mbars = storage.dqb_free_mbars.data_ptr()
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

        # ------------------------------------------------------------------
        # SMEM tensor views.
        # ------------------------------------------------------------------
        stationary_q_tma = storage.stationary_q.get_tensor(
            stationary_a_layout_staged.outer,
            swizzle=stationary_a_layout_staged.inner,
        )
        stationary_do_tma = storage.stationary_do.get_tensor(
            stationary_a_layout_staged.outer,
            swizzle=stationary_a_layout_staged.inner,
        )
        # v5.1b zero-copy score-B STRIP views (item 6).  One strip
        # stage ([h16 x D512] K-major box, 16,384 B) is byte-identical
        # to eight contiguous [n16 x k64] score-B stages (1,024-element
        # blocks -- same algebra as the v5 panel windows), so the
        # 16-stage score-B layout binds the WHOLE 2-strip buffer at its
        # base: fragment stage index = (t % 2) * 8 + piece.  One view
        # per tensor; the old per-chunk +16,384 second view retires
        # with the panels.
        # V32-TODO(audit): host-probe the byte identity (same SW128B
        # K-major atom on both sides; cosize equality is asserted in
        # __call__, the atom identity is not).
        q_strip_b = cute.make_tensor(
            cute.recast_ptr(
                stationary_q_raw,
                score_b_layout_staged.inner,
                dtype=self.element_dtype,
            ),
            score_b_layout_staged.outer,
        )
        do_strip_b = cute.make_tensor(
            cute.recast_ptr(
                stationary_do_raw,
                score_b_layout_staged.inner,
                dtype=self.element_dtype,
            ),
            score_b_layout_staged.outer,
        )
        # v5.1b K residency (item 5): score A, one resident
        # [own-kv64 x D512] image, 8 D64-piece stages (stage == piece).
        k_chase = storage.score_kv.get_tensor(
            score_a_layout_staged.outer,
            swizzle=score_a_layout_staged.inner,
        )
        # v5.2: DORMANT (the dQ TMA epilogue retired with the
        # eviction); the staging view is kept so the dormant epi
        # helpers stay compilable.
        s_dq_epi = cute.make_tensor(
            cute.recast_ptr(
                storage.round_buf_a.data_ptr(),
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
        # v5.2 eviction-A views of the SAME kdq gen bytes: the 2-stage
        # (128,32,64) layout's stage stride (4,096 elements) equals the
        # legacy layout's m-half stride (order-(2,1,3) algebra, byte
        # identity asserted host-side), so stage == d_half.  The
        # legacy round_kd views above stay for the FROZEN gather-side
        # kdq fill machine.
        round_kd_evict = (
            storage.round_buf_a.get_tensor(
                dq_a_evict_layout_staged.outer,
                swizzle=dq_a_evict_layout_staged.inner,
            ),
            storage.round_buf_b.get_tensor(
                dq_a_evict_layout_staged.outer,
                swizzle=dq_a_evict_layout_staged.inner,
            ),
        )
        # v5.1 round-gen B views: one gen = one FULL-WIDE
        # [h64 x own-D64] box (8,192 B, single stage) covering the
        # pair's whole chunk interval.  The gen sits at the head of
        # its (unchanged) 16,384 B round buffer; buffer assignment by
        # gen parity (dO even -> A, Q odd -> B) is preserved by the
        # 20-gen order.
        round_grad = (
            storage.round_buf_a.get_tensor(
                dkv_b_layout_staged.outer,
                swizzle=dkv_b_layout_staged.inner,
            ),
            storage.round_buf_b.get_tensor(
                dkv_b_layout_staged.outer,
                swizzle=dkv_b_layout_staged.inner,
            ),
        )
        p_slab_raw = storage.p_slab.data_ptr()
        ds_slab_raw = storage.ds_slab.data_ptr()
        dq_b_raw = storage.dq_b.data_ptr()
        # G3/G4 A operands: the P/dS slabs bound directly with the
        # 2-stage (h-chunk sub-image) dkv-A layout.  The staged stage
        # stride (4,096 elements = 8,192 B) matches the chunk-major
        # sub-image stacking by construction, so stage index == h-chunk.
        p_slab_t = storage.p_slab.get_tensor(
            dkv_a_layout_staged.outer,
            swizzle=dkv_a_layout_staged.inner,
        )
        ds_slab_t = storage.ds_slab.get_tensor(
            dkv_a_layout_staged.outer,
            swizzle=dkv_a_layout_staged.inner,
        )
        # G5 B operand: dq_b bound with the 2-stage (kv-wave sub-image)
        # dq-B layout; the 8,192 B stage jump is the addendum's single-
        # base-descriptor window hop (a SW128B 1,024 B atom multiple, so
        # the swizzle phase is preserved -- rank-symmetric on both CTAs).
        dq_b_t = storage.dq_b.get_tensor(
            dq_b_layout_staged.outer,
            swizzle=dq_b_layout_staged.inner,
        )
        # Publish store image: one [own-kv64 x h64] bf16 sub-image,
        # H-contiguous 128 B rows.  The byte identity between this store
        # image and one dkv-A slab stage is what makes the stmatrix
        # publish directly produce the G3/G4 A operand (and, for dq_b,
        # the G5 B operand).
        score_store_layout = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (2 * self.N_TILE_CTA, self.N_TILE),
            1,
        )
        assert (
            cute.cosize(score_store_layout)
            == self.PDS_BLOCK_ELEMENTS
        ), cute.cosize(score_store_layout)
        # V32-TODO(audit): host-probe that the ROW_MAJOR epi image's
        # swizzle atom equals the K-major dkv-A operand atom (SW128B on
        # both sides expected; the v8 STS.U16 precedent makes this a
        # mandatory SASS gate before any GPU run).
        assert (
            score_store_layout.inner
            == dkv_a_layout_staged.inner
        )
        assert (
            score_store_layout.inner
            == dq_b_layout_staged.inner
        )
        # v5 sub-tile publish domain (spec Z4).  The publish target
        # stays the NATURAL [own-kv64 x h64] chunk image (the frozen
        # relay payload / dq_b / G3-A byte forms all require natural
        # head order), but one math pass covers only the sub-tile's
        # TWO h16 column boxes, 32 heads apart.  The column mode is
        # regrouped (16, 2):(1, 32) -- box-local head, box-hi -- and
        # the window mode J = (2):(16) sits at TOP-LEVEL domain mode 1
        # ([fix-r1]; see the placement note at the construction):
        # slicing the partitioned tensor at J = t%2 selects the
        # sub-tile's boxes.  The offset rides the LAYOUT coordinates
        # (never the pointer), so the SW128B swizzle stays anchored at
        # the 1,024 B-aligned image base (a raw +32 B pointer offset
        # would NOT commute with the swizzle).
        # Strides derive from the epi layout family; the column
        # premise is asserted SEMANTICALLY (fix-r0): the hardware r0
        # gate showed the epi outer as
        #   ((8,8),(64,1),(1,1)):((64,512),(1,0),(0,0)),
        # i.e. tile_to_shape's coalesce keeps DEGENERATE size-1/
        # stride-0 sub-modes inside the column and stage modes, so a
        # naked shape[1]/stride[1] LEAF comparison is form-brittle
        # ((64,1) != 64) even though the semantics -- 64 contiguous
        # unit-stride columns -- are exactly the premise the J-mode
        # domain needs.  Coalescing the column mode and pinning
        # rank == 1, size == 64, cosize == 64 is mathematically
        # equivalent to `layout == (64):(1)` (for a coalesced rank-1
        # layout cosize = (size-1)*stride + 1), so this is the SAME
        # contract in degenerate-robust form, not a loosening.
        score_store_cols = cute.coalesce(
            cute.select(score_store_layout.outer, mode=[1])
        )
        assert cute.rank(score_store_cols) == 1, str(
            score_store_layout.outer
        )
        assert cute.size(score_store_cols) == self.N_TILE, str(
            score_store_layout.outer
        )
        assert cute.cosize(score_store_cols) == self.N_TILE, str(
            score_store_layout.outer
        )
        assert cute.size(score_store_layout.outer, mode=[2]) == 1, str(
            score_store_layout.outer
        )
        # [fix-r1] J placement: the hardware r1 gate showed that
        # partition_D consumes domain mode 0 ENTIRELY -- with J inside
        # mode 0 the 4,096-element bundle divided against the copy's
        # 2,048-element D-tile and the tile-iteration factor (== J)
        # FOLDED into the rest mode ((8,1),(2,2),1,1,1: mode[1] became
        # (in-tile 2, J 2), sub-mode order machinery-internal), while
        # the top-level padding modes 1..3 passed through verbatim to
        # output modes 2..4 (both the v32 slice-at-mode-4 precedent
        # and the r1 echo confirm this passthrough law).  So J lives
        # at TOP-LEVEL domain mode 1 (shape 2, stride 16): mode 0 is
        # then EXACTLY the copy tile (2,048 = 64 rows x 32 cols,
        # v32-congruent where mode 0 == tile == 4,096), no folding,
        # and J passes through deterministically to output mode [2].
        # The J offset still rides the LAYOUT coordinates (never the
        # pointer), so the SW128B swizzle stays anchored at the
        # 1,024 B-aligned image base -- the Z3b contract is intact.
        score_store_domain = cute.make_layout(
            (
                (
                    score_store_layout.outer.shape[0],
                    (self.SUB_TILE_BOX, 2),
                ),
                2,
                1,
                1,
            ),
            stride=(
                (
                    score_store_layout.outer.stride[0],
                    (1, self.SUB_TILE_H),
                ),
                self.SUB_TILE_BOX,
                0,
                0,
            ),
        )
        assert (
            cute.cosize(score_store_domain)
            == self.PDS_BLOCK_ELEMENTS
        )
        # Publish-domain width contract (spec Z4, asserted explicitly):
        # each J slice of the domain exposes EXACTLY the h32 sub-tile
        # -- SUB_TILE_H columns as (16, 2) boxes -- and the top-level
        # J window mode carries the two slices.
        assert (
            cute.size(score_store_domain, mode=[0, 1])
            == self.SUB_TILE_H
        ), str(score_store_domain)
        assert cute.size(score_store_domain, mode=[1]) == 2, str(
            score_store_domain
        )
        # Per h-chunk stmatrix targets (static sub-image bases).
        p_store = (
            cute.make_tensor(
                cute.recast_ptr(
                    p_slab_raw,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            ),
            cute.make_tensor(
                cute.recast_ptr(
                    p_slab_raw + self.PDS_BLOCK_ELEMENTS,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            ),
        )
        ds_store = (
            cute.make_tensor(
                cute.recast_ptr(
                    ds_slab_raw,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            ),
            cute.make_tensor(
                cute.recast_ptr(
                    ds_slab_raw + self.PDS_BLOCK_ELEMENTS,
                    score_store_layout.inner,
                    dtype=self.element_dtype,
                ),
                score_store_domain,
            ),
        )
        # dq_b own-half second image (addendum section 1): the local
        # stmatrix target is sub_img[rank] (the CTA's own kv-half rows).
        # rank is runtime, so the base is a dynamic pointer.
        dqb_own_ptr = cute.make_ptr(
            self.element_dtype,
            dq_b_raw.toint()
            + rank * Int32(self.PDS_BLOCK_BYTES),
            cute.AddressSpace.smem,
            assumed_align=1024,
        )
        dqb_own_store = cute.make_tensor(
            cute.recast_ptr(
                dqb_own_ptr,
                score_store_layout.inner,
                dtype=self.element_dtype,
            ),
            score_store_domain,
        )
        # Bulk-DSM peer push endpoints (fallback path; the st.async
        # register push is a registered V32-TODO upgrade): source is the
        # LOCAL dS slab peer-half sub-image, destination is the PEER
        # CTA's dq_b sub_img[rank] (same offset on both CTAs).
        dsm_src_ptr_int = (
            ds_slab_raw.toint()
            + (Int32(1) - rank) * Int32(self.PDS_BLOCK_BYTES)
        )
        dsm_dst_offset_int = (
            dq_b_raw.toint()
            + rank * Int32(self.PDS_BLOCK_BYTES)
        )
        # Column-axis softmax stats: lse[h128] then delta[h128] (both
        # CTAs hold the FULL head vector -- the transposed plane indexes
        # constants by the column axis).
        softmax_stats = storage.stats.get_tensor(
            cute.make_layout(
                (self.H_TILE_CLUSTER, 2),
                stride=(1, self.H_TILE_CLUSTER),
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
        # v3.2: constants index the COLUMN (head) axis, and each CTA's
        # score fragment spans ALL 128 head columns (C = M-half x full
        # N), so BOTH CTAs load the FULL lse/delta vectors (no rank
        # half-selection).
        g_scaled_lse = cute.flat_divide(
            scaled_lse,
            (self.H_TILE_CLUSTER,),
        )
        g_sum_odo = cute.flat_divide(
            sum_odo,
            (self.H_TILE_CLUSTER,),
        )
        t_g_scaled_lse = stats_thread_copy.partition_S(
            g_scaled_lse[None, 0, (token_idx, batch_idx)]
        )
        t_s_scaled_lse = stats_thread_copy.partition_D(
            softmax_stats[None, 0]
        )
        t_g_sum_odo = stats_thread_copy.partition_S(
            g_sum_odo[None, 0, (token_idx, batch_idx)]
        )
        t_s_sum_odo = stats_thread_copy.partition_D(
            softmax_stats[None, 1]
        )

        # v3.2: the panel boxes are STATIONARY_TILE_H(32)-row B-operand
        # tiles (the TMA coordinates 2c + rank walk FOUR 32-row gmem
        # H-tiles), partitioned through the helper MMA's B fraction.
        g_q = cute.local_tile(
            tma_tensor_q,
            cute.select(
                (
                    self.H_TILE_CTA,
                    self.STATIONARY_TILE_H,
                    self.D_HEAD,
                ),
                mode=[1, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        g_do = cute.local_tile(
            tma_tensor_do,
            cute.select(
                (
                    self.H_TILE_CTA,
                    self.STATIONARY_TILE_H,
                    self.D_HEAD,
                ),
                mode=[1, 2],
            ),
            (None, None, (token_idx, batch_idx)),
        )
        stationary_thr_mma = stationary_tiled_mma.get_slice(0)
        rank_g_q = stationary_thr_mma.partition_B(g_q)
        rank_g_do = stationary_thr_mma.partition_B(g_do)
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
        # v5: the score C tile is one h32 sub-tile (kv128 x 32); the
        # coordinate mode [1] value n is the FRAGMENT column -- the
        # physical head is head(t, n) (see the class head-map note).
        rank_score_coordinates = rank_score_mma.partition_C(
            cute.make_identity_tensor(
                (self.H_TILE_CLUSTER, self.SUB_TILE_H)
            )
        )
        rank_dkv_coordinates = rank_dkv_mma.partition_C(
            cute.make_identity_tensor(self.DKV_MMA_TILER[:2])
        )
        rank_dq_coordinates = rank_dq_mma.partition_C(
            cute.make_identity_tensor(self.DQ_MMA_TILER[:2])
        )

        # v5.1 per-CTA round-gen B TMA partitions: mQT/mdOT ([D,H]
        # views) tiled (N=D128-round, K=h64-box); the D-tile
        # coordinate walks the four D-rounds, the K-tile coordinate
        # the TWO h64 tiles (tile P == chunk P == pair P's full head
        # interval) -- gen (P, r) is ONE box copy -- and the CTA takes
        # its own N-half columns.
        g_qt = cute.local_tile(
            tma_tensor_qt,
            cute.select(self.DKV_B_TILER, mode=[1, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        g_dot = cute.local_tile(
            tma_tensor_dot,
            cute.select(self.DKV_B_TILER, mode=[1, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        rank_g_qt = rank_dkv_mma.partition_B(g_qt)
        rank_g_dot = rank_dkv_mma.partition_B(g_dot)
        b_cta_layout = cute.make_layout(
            cute.slice_(
                cluster_layout_vmnk,
                (0, None, 0, 0),
            ).shape
        )
        t_qt_smem_a, t_qt_gmem = cpasync.tma_partition(
            tma_atom_qt,
            block_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(round_grad[0], 0, 3),
            cute.group_modes(rank_g_qt, 0, 3),
        )
        t_qt_smem_b, _ = cpasync.tma_partition(
            tma_atom_qt,
            block_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(round_grad[1], 0, 3),
            cute.group_modes(rank_g_qt, 0, 3),
        )
        t_dot_smem_a, t_dot_gmem = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(round_grad[0], 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )
        t_dot_smem_b, _ = cpasync.tma_partition(
            tma_atom_dot,
            block_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(round_grad[1], 0, 3),
            cute.group_modes(rank_g_dot, 0, 3),
        )

        # ------------------------------------------------------------------
        # MMA fragments.
        # ------------------------------------------------------------------
        # v3.2 operand fragments: chase is score A, panels are score B
        # (per h-chunk views), P/dS slabs are dkv A (stage = h-chunk),
        # round gens are dkv B, dq_b is the dq B (stage = kv-wave).
        score_q_fragment = score_tiled_mma.make_fragment_B(
            q_strip_b
        )
        score_do_fragment = dp_tiled_mma.make_fragment_B(
            do_strip_b
        )
        score_k_fragment = score_tiled_mma.make_fragment_A(k_chase)
        dp_k_fragment = dp_tiled_mma.make_fragment_A(k_chase)
        # v5.2 eviction G5 fragments.  A: the kdq gen d-half windows
        # (stage == d_half, byte identity asserted host-side).
        dq_kd_fragment_a = dq_evict_tiled_mma.make_fragment_A(
            round_kd_evict[0]
        )
        dq_kd_fragment_b = dq_evict_tiled_mma.make_fragment_A(
            round_kd_evict[1]
        )
        # B: a HAND-DERIVED window view of the FROZEN dq_b bytes (the
        # auto (128,32,64) MN-major staged-B would select an SW32 atom
        # and cannot bind the SW128 image).  Legacy stage form (from
        # the (256,128,64) staged-B, order (2,1,3), hardware-
        # validated):
        #   offset(h, kv) = (kv//8)*512 + sw((kv%8)*64 + h),
        #   wave stage stride 4096.
        # The eviction view narrows N to a 16-head window and tiles
        # the stage mode as (t-window 4, wave 2) -> flat stage index
        # t + 4*w at offset t*16 + w*4096 elements (32 B mid-swizzle-
        # atom window starts -- the k_block descriptor precedent).
        # Profile ((atom_n, atom_k), rest_n, rest_k, stage) with
        # atom_k split (8,2):(64,512) and rest_k 4 x 1024 (kv 0..63).
        dq_b_evict_layout = cute.make_layout(
            (
                (16, (8, 2)),
                1,
                4,
                (4, 2),
            ),
            stride=(
                (1, (64, 512)),
                0,
                1024,
                (16, 4096),
            ),
        )
        # Tight cosize pin: the view must cover the 8,192-element
        # dq_b image bijectively-in-range (echo on failure).
        assert cute.cosize(dq_b_evict_layout) == 8192, str(
            dq_b_evict_layout
        )
        assert cute.cosize(dq_b_layout_staged) == 8192, str(
            dq_b_layout_staged
        )
        dq_b_evict_t = cute.make_tensor(
            cute.recast_ptr(
                dq_b_raw,
                dq_b_layout_staged.inner,
                dtype=self.element_dtype,
            ),
            dq_b_evict_layout,
        )
        dq_ds_fragment = dq_evict_tiled_mma.make_fragment_B(
            dq_b_evict_t
        )
        grad_fragment_a = dkv_tiled_mma.make_fragment_B(
            round_grad[0]
        )
        grad_fragment_b = dkv_tiled_mma.make_fragment_B(
            round_grad[1]
        )
        p_fragment = dkv_tiled_mma.make_fragment_A(p_slab_t)
        ds_fragment = dkv_tiled_mma.make_fragment_A(ds_slab_t)

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
        # v5.1b (item 5): the K residency completion gate replaces the
        # 2-slot chase ring -- ONE generation per bundle:
        #   producer_acquire (gather) <- empty: the leader's release,
        #     tcgen05-tracked, fires when score(3)(i-1)'s reads
        #     complete (the "K(i+1) gather gate = score(3)(i)
        #     completion edge" contract);
        #   producer_commit (gather, all 256 threads) -> full: the
        #     leader's bundle-head consumer_wait.
        pipe_kres = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=gather_group,
            consumer_group=leader_group,
            barrier_storage=storage.kres_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # v5.1b (item 6): the Q/dO strip stream -- one generation =
        # the pass's Q+dO strip pair, 4 generations per bundle,
        # stage = t % 2:
        #   producer_acquire (W17 elect) <- empty: the leader's
        #     release, tcgen05-tracked, fires when score(t-2)'s reads
        #     complete;
        #   producer_commit (W17 elect, after its own TMA-completion
        #     mbar wait) -> full: the score pass's strip wait.
        pipe_strip = pipeline.PipelineAsyncUmma.create(
            num_stages=2,
            producer_group=load_elect_group,
            consumer_group=leader_group,
            barrier_storage=storage.strip_mbars.data_ptr(),
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
        # v5 (spec Z4): the pds handoff becomes SUB-TILE granular --
        # 2 stages, 4 generations/bundle, and MATH is the producer
        # (v12's relay-commit form encoded the bundle-level handoff;
        # the per-sub-tile cadence must commit right after each
        # sub-tile publish, which only math can order).  Both CTAs'
        # math threads arrive the leading CTA's full mbar, so the
        # producer count carries the atom_thr_size factor exactly like
        # the s_done/dp_done consumer groups (lesson #10's cousin on
        # the producer side; a per-CTA count would flip the phase
        # twice per generation and desynchronize the leader).
        # The relay's DSM-source WAR (math(b+1) overwriting the dS
        # peer half while the push(b) reads it) is NOT carried by this
        # pipeline: it is covered transitively by mb_dqb(b) -> G5(b)
        # -> score(0)(b+1) -> math(b+1) (see the relay block note).
        pipe_pds = pipeline.PipelineAsyncUmma.create(
            num_stages=2,
            producer_group=math_group,
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
        # v5.2: the dQ eviction handoff replaces dq_done (the TMA
        # epilogue retired).  Leader UMMA producer commits one
        # generation per (t, r) block; the reduce warps of both CTAs
        # consume (T2R + plain RMW offload) -- consumer count carries
        # the atom_thr_size factor via reduce_group (lesson #10).
        pipe_dq_evict = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=leader_group,
            consumer_group=reduce_group,
            barrier_storage=storage.dq_evict_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # v3.2 dq_b free gate (mb_dqb_free): 1-stage Umma pipeline.  The
        # leader's group commit after the LAST G5 wave of the bundle
        # frees the whole dq_b region -- the tcgen05 group commit tracks
        # every previously issued MMA, so it covers BOTH D-rounds and
        # BOTH waves (errata #1's "last consuming wave" obligation).
        # Consumers are the math warpgroups of BOTH CTAs (count carries
        # the atom_thr_size factor); the relay's peer push is
        # transitively gated: math arrives pds_ready only after its own
        # free wait, and the relay pushes only after pds_ready.
        # The leader pre-arms one initial-free commit before the bundle
        # loop so math's wait is unconditional (no cross-branch pipeline
        # state mutation).
        pipe_dqb_free = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=leader_group,
            consumer_group=math_group,
            barrier_storage=storage.dqb_free_mbars.data_ptr(),
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
            # v3.2 mb_dqb[w] ready gates (relay_mbars precedent: count
            # 2, both ranks arrive at the LEADING CTA, the leader waits
            # with the cluster-acquire wait).  One arrival per CTA's
            # relay = "my sub_img[w] is ready" (own stmatrix commit for
            # w == rank, observed peer landing for w == 1-rank --
            # errata #2's relay-arrive shape).  mb_dqb_free is a
            # pipeline (see pipe_dqb_free), not a raw gate.
            cute.arch.mbarrier_init(dqb_mbars, 2)
            cute.arch.mbarrier_init(dqb_mbars + 1, 2)
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

        score_c_layout = score_tiled_mma.make_fragment_C(
            score_tiled_mma.partition_shape_C(
                (self.H_TILE_CLUSTER, self.SUB_TILE_H)
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
        # v5.2: the eviction C fragment ((128,32) CG2 fold -> 128 DP x
        # 16 columns per CTA) and its coordinate identity (offloader
        # decode).
        dq_evict_c_layout = dq_evict_tiled_mma.make_fragment_C(
            dq_evict_tiled_mma.partition_shape_C(
                self.DQ_EVICT_TILER[:2]
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
        # v5.2 rotating dQ eviction slots (block g -> slot g%2,
        # static: 16 blocks/bundle is even).
        t_dq_rot = (
            cute.make_tensor(
                tmem_ptr + self.TMEM_DQ_SLOT0,
                dq_evict_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr + self.TMEM_DQ_SLOT1,
                dq_evict_c_layout,
            ),
        )
        # v5.2 dkv slot army, [fix-r7] FOUR static slots: 16 blocks/
        # bundle mod 4 == 0 makes slot == stage == (2r + p) % 4 a
        # COMPILE-TIME constant (pair*8 mod 4 == 0), so the slot
        # tensors are plain static-offset views -- alignment folds
        # automatically (the r7 DSL ruling forbids .align on TMEM
        # pointers; the runtime-addressed 6-slot form is retired).
        t_dkv_army = (
            cute.make_tensor(
                tmem_ptr + self.TMEM_DKV_BASE,
                dkv_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr
                + (self.TMEM_DKV_BASE + self.TMEM_DKV_SLOT_COLS),
                dkv_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr
                + (
                    self.TMEM_DKV_BASE
                    + 2 * self.TMEM_DKV_SLOT_COLS
                ),
                dkv_c_layout,
            ),
            cute.make_tensor(
                tmem_ptr
                + (
                    self.TMEM_DKV_BASE
                    + 3 * self.TMEM_DKV_SLOT_COLS
                ),
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
        # v3.2: one "tile" is a 128-token kv bundle (2 x N_TILE): the
        # cluster gathers 128 topk rows per bundle (own-kv64 per CTA).
        tile_count = (topk + Int32(2 * self.N_TILE - 1)) // Int32(
            2 * self.N_TILE
        )

        if (
            warp_idx < Int32(self.MATH_WARP_BEGIN)
            or warp_idx >= Int32(self.MMA_WARP)
        ):
            cute.arch.setmaxregister_decrease(48)
        else:
            # v11: rebalance AGAIN, back toward the reducer.  The v9_3
            # drain SASS showed the true pacer's mechanism: at 104 regs
            # the 64 T2R value registers spill to local memory (65 LDL +
            # 28 STL inside the REDG loop) and all 16 REDGs serialize on
            # one register set -- 306ns/op vs the baseline reducer's
            # 138ns.  Post-stmatrix the math publish needs far fewer
            # registers (STACK 1040 -> 600), so shift 32 regs from math
            # to reduce.  Pool stays the launch allocation (640*96 =
            # 61,440), exact balance:
            #   dec supply: 8 warps * (96-48) * 32            = 12,288
            #   inc demand: 4*(128-96)*32 + 8*(128-96)*32     = 12,288
            #   totals: 256*48 + 128*128 + 256*128 = 61,440 = 640*96.
            # (v11 probe 3: setmaxnreg values must be multiples of 8, so
            # there is NO step between reduce=120 and reduce=128; 144/120
            # left 23 residual drain spills incl. six 64-bit address
            # temporaries.  math at 128 is the watch item -- pre-stmatrix
            # it spilled badly there, post-stmatrix 144 held 63/14 with
            # margin; the compile gate decides.)
            if warp_idx < Int32(self.REDUCE_WARP_BEGIN):
                cute.arch.setmaxregister_increase(128)
            else:
                cute.arch.setmaxregister_increase(128)

        # ==================================================================
        # Role bodies.
        # ==================================================================
        if warp_idx < Int32(self.GATHER_WARPS):
            # --- gather, v5.1b: the K RESIDENCY fill (item 5) --
            # eight rank-owned [own-kv64 x D64] pieces per bundle,
            # gathered ONCE into the resident image (the v5 4x
            # per-pass re-gather retires with the chase ring: -75%
            # gather traffic) -- PLUS the kdq image fills offloaded
            # from the load warp (128 threads vs 32).  Per iteration
            # the order is: [kres acquire; fill 8 pieces; commit] ->
            # r1(b-1) rendezvous -> r0(b) rendezvous.  The global kdq
            # named-barrier sequence r0(b), r1(b), r0(b+1), ... is
            # UNCHANGED (frozen kdq machine; r1(b-1) merely moves from
            # the old piece-2 boundary to after the fill, which only
            # tightens the K prefetch -- the fill no longer waits the
            # late r1 credits).
            _iket.mark("ROLE_KV_LOAD", rank)
            gather_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                1,
            )
            gather_kd_rows_0 = self._kd_round_rows_v2(round_kd[0])
            gather_kd_rows_1 = self._kd_round_rows_v2(round_kd[1])
            # v5.1b resident-K piece row views ([kv64 rows, D64
            # contiguous], stage == D-piece).
            kres_rows = (
                self._chase_slot_rows_v32(k_chase, 0),
                self._chase_slot_rows_v32(k_chase, 1),
                self._chase_slot_rows_v32(k_chase, 2),
                self._chase_slot_rows_v32(k_chase, 3),
                self._chase_slot_rows_v32(k_chase, 4),
                self._chase_slot_rows_v32(k_chase, 5),
                self._chase_slot_rows_v32(k_chase, 6),
                self._chase_slot_rows_v32(k_chase, 7),
            )
            if tile_count > Int32(0):
                for loop_iter in cutlass.range(tile_count):
                    bundle_idx = tile_count - Int32(1) - loop_iter
                    # edge: the leader's kres release -- tcgen05-
                    # tracked score(3)(b-1) READ completion (the
                    # "K(i+1) gather gate = score(3)(i) completion
                    # edge" contract); bundle 0 eats the init credit.
                    # The wait sits OUTSIDE the LOAD_K span so the
                    # span reads as pure fill time.
                    pipe_kres.producer_acquire(gather_state)
                    load_k_token = _iket.range_start(
                        "LOAD_K(i)",
                        loop_iter,
                    )
                    for piece in cutlass.range_constexpr(
                        self.SCORE_D_PIECES
                    ):
                        self._load_chase_piece_v32(
                            mKV,
                            mTopkIdxs,
                            kres_rows[piece],
                            token_idx,
                            batch_idx,
                            bundle_idx,
                            Int32(piece),
                            topk,
                            rank,
                            tidx,
                            kv_copy_atom,
                            kv_thread_copy,
                        )
                        cute.arch.cp_async_commit_group()
                    # One drain + fence covers all eight piece groups,
                    # then the single residency commit (all 256 gather
                    # threads arrive) -> the leader's bundle-head wait.
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    pipe_kres.producer_commit(gather_state)
                    gather_state.advance()
                    _iket.range_end(
                        load_k_token,
                        loop_iter,
                    )
                    # r1(b-1) rendezvous AFTER the fill/commit (K
                    # prefetch never waits the late r1 credits) and
                    # BEFORE r0(b) (global kdq order preserved).
                    # edge (W17 side): G5 r0(b-1) consumption freed
                    # the g18/g19 ring credits.
                    if loop_iter > Int32(0):
                        self._gather_kdq_v8(
                            mKV,
                            mTopkIdxs,
                            gather_kd_rows_0,
                            gather_kd_rows_1,
                            token_idx,
                            batch_idx,
                            bundle_idx + Int32(1),
                            Int32(1),
                            topk,
                            rank,
                            tidx,
                            kv_copy_atom,
                            kv_thread_copy,
                        )
                    # Round-0 kdq of THIS bundle (g16/g17, feeds the
                    # G5 r0 waves at the bundle tail).
                    self._gather_kdq_v8(
                        mKV,
                        mTopkIdxs,
                        gather_kd_rows_0,
                        gather_kd_rows_1,
                        token_idx,
                        batch_idx,
                        bundle_idx,
                        Int32(0),
                        topk,
                        rank,
                        tidx,
                        kv_copy_atom,
                        kv_thread_copy,
                    )
                # Epilogue: the last bundle's round-1 kdq (no next fill).
                self._gather_kdq_v8(
                    mKV,
                    mTopkIdxs,
                    gather_kd_rows_0,
                    gather_kd_rows_1,
                    token_idx,
                    batch_idx,
                    Int32(0),
                    Int32(1),
                    topk,
                    rank,
                    tidx,
                    kv_copy_atom,
                    kv_thread_copy,
                )
                pipe_kres.producer_tail(gather_state)

        elif warp_idx < Int32(self.REDUCE_WARP_BEGIN):
            # --- math: stats, per-tile softmax + publication, dQ epilogue.
            _iket.mark("ROLE_MATH", rank)
            mtx = tidx - Int32(self.MATH_THREAD_BEGIN)
            if warp_idx == Int32(self.MATH_WARP_BEGIN):
                load_stats_token = _iket.range_start(
                    "LOAD_STATS",
                    Int32(0),
                )
                if tile_count > Int32(0):
                    # The 32-thread x 2-value machine moves 64 heads
                    # per tile; the v3.2 stats vectors are the FULL 128
                    # heads, so BOTH rest tiles must be copied (audit
                    # F4: tile 1 missing left head[64:128) stats as
                    # uninitialized SMEM garbage).
                    cute.copy(
                        stats_copy_atom,
                        t_g_scaled_lse[None, 0],
                        t_s_scaled_lse[None, 0],
                    )
                    cute.copy(
                        stats_copy_atom,
                        t_g_scaled_lse[None, 1],
                        t_s_scaled_lse[None, 1],
                    )
                    cute.copy(
                        stats_copy_atom,
                        t_g_sum_odo[None, 0],
                        t_s_sum_odo[None, 0],
                    )
                    cute.copy(
                        stats_copy_atom,
                        t_g_sum_odo[None, 1],
                        t_s_sum_odo[None, 1],
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                _iket.range_end(
                    load_stats_token,
                    Int32(0),
                )
            self.math_barrier.arrive_and_wait()

            s_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.SCORE_DONE_STAGES,
            )
            dp_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.SCORE_DONE_STAGES,
            )
            # v5: pds is 2-stage, math-produced (stage == t % 2).
            pds_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                2,
            )
            dqb_free_state = pipeline.make_pipeline_state(
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
            # v7 ping-pong: identical static layouts at the stage-1 bases.
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
                utils.LayoutEnum.ROW_MAJOR,
                self.element_dtype,
                self.acc_dtype,
                score_copy,
            )
            # v9.3 premise as a BUILD gate: the publish must lower to
            # stmatrix; a silent CopyUniversalOp fallback (the v8 scalar
            # STS.U16 + PRMT path) fails the trace instead of the run.
            # V32-TODO(audit): the transposed [kv64 x h64] ROW_MAJOR
            # image may legitimately select the TRANSPOSED stmatrix
            # variant; if trace-prepare rejects this isinstance, widen
            # it to the trans op class after the SASS gate confirms
            # stmatrix lowering (rev0 step 3 of the design doc).
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
            # v5 publish targets (spec Z4): sub-tile t = (c, j) publishes
            # its [own-kv64 x h32] fragment as the TWO h16 column boxes
            # of the NATURAL chunk-c image selected by the J-mode slice
            # (j = t % 2; see the score_store_domain note).  The CG2
            # accumulator is M-half x FULL N, so every math warp owns
            # local rows only; the only cross-CTA payload is still the
            # relay's bundle-level dq_b peer push, byte-identical to
            # v32 because the chunk images assemble in natural order
            # from the j = 0 / j = 1 publishes.
            t_rs_p_0 = thread_copy_r2s.partition_D(p_store[0])
            t_rs_p_1 = thread_copy_r2s.partition_D(p_store[1])
            t_rs_ds_0 = thread_copy_r2s.partition_D(ds_store[0])
            t_rs_ds_1 = thread_copy_r2s.partition_D(ds_store[1])
            t_rs_dqb = thread_copy_r2s.partition_D(dqb_own_store)
            # [fix-r1] The J window is a TOP-LEVEL domain mode now
            # (mode 1), and top-level modes 1..3 pass through
            # partition_D verbatim to output modes 2..4 (empirical law
            # from the v32 slice-at-mode-4 precedent plus the r1 echo,
            # where the in-mode-0 J folded into the rest mode
            # instead).  J must therefore surface at output mode [2];
            # echo the shape if the machinery moved it.
            assert cute.size(t_rs_p_0, mode=[2]) == 2, str(
                t_rs_p_0.shape
            )
            assert cute.size(t_rs_ds_0, mode=[2]) == 2, str(
                t_rs_ds_0.shape
            )
            assert cute.size(t_rs_dqb, mode=[2]) == 2, str(
                t_rs_dqb.shape
            )
            # Sub-tile store tiles, indexed by t: (chunk c = t//2 picks
            # the tensor, window j = t%2 indexes the J mode at [2];
            # the trailing size-1 passthrough modes stay, keeping the
            # sliced arity at four modes exactly like v32's tiles).
            t_rs_p_tiles = (
                t_rs_p_0[None, None, 0, None, None],
                t_rs_p_0[None, None, 1, None, None],
                t_rs_p_1[None, None, 0, None, None],
                t_rs_p_1[None, None, 1, None, None],
            )
            t_rs_ds_tiles = (
                t_rs_ds_0[None, None, 0, None, None],
                t_rs_ds_0[None, None, 1, None, None],
                t_rs_ds_1[None, None, 0, None, None],
                t_rs_ds_1[None, None, 1, None, None],
            )
            # dq_b own-image windows: only the two sub-tiles with
            # c == rank write here; their J windows are j = 0 / j = 1.
            t_rs_dqb_tiles = (
                t_rs_dqb[None, None, 0, None, None],
                t_rs_dqb[None, None, 1, None, None],
            )
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

            # v5 (spec Z4): one 128-token bundle per iteration with FOUR
            # h32 sub-tile phases inside (done-pipeline stage == t % 2,
            # so the T2R source select stays STATIC).  pds becomes a
            # 2-stage pipeline with MATH as the producer: one acquire +
            # one commit PER SUB-TILE (4/bundle), handing each sub-tile
            # to the leader's grads(t) individually -- the t/t+1
            # in-flight window that realizes the M(t) || G(t-1) overlap.
            # The dq_b free wait stays bundle-level (t == 0 head): the
            # leader pre-arms an initial-free commit, so no first-bundle
            # branch.
            for loop_iter in cutlass.range(tile_count):
                for sub_tile in cutlass.range_constexpr(
                    self.SUB_TILES
                ):
                    chunk_payload = loop_iter * Int32(
                        2 * self.SUB_TILES
                    ) + Int32(2 * sub_tile)
                    wait_s_token = _iket.range_start(
                        "WAIT_S(i)",
                        chunk_payload,
                    )
                    # edge: leader S_ISSUE(t) commit (tcgen05-tracked).
                    pipe_s_done.consumer_wait(s_state)
                    _iket.range_end(
                        wait_s_token,
                        chunk_payload,
                    )
                    t2r_s_token = _iket.range_start(
                        "T2R_S(i)",
                        chunk_payload,
                    )
                    if cutlass.const_expr(sub_tile % 2 == 0):
                        cute.copy(score_copy, score_source, r_score)
                    else:
                        cute.copy(
                            score_copy_pp,
                            score_source_pp,
                            r_score,
                        )
                    cute.arch.fence_view_async_tmem_load()
                    # edge -> leader's s acquire for pass t+2 (TMEM
                    # ping-pong stage WAR).
                    pipe_s_done.consumer_release(s_state)
                    s_state.advance()
                    _iket.range_end(
                        t2r_s_token,
                        chunk_payload,
                    )

                    wait_dp_token = _iket.range_start(
                        "WAIT_dP(i)",
                        chunk_payload,
                    )
                    # edge: leader dP_ISSUE(t) commit.
                    pipe_dp_done.consumer_wait(dp_state)
                    _iket.range_end(
                        wait_dp_token,
                        chunk_payload,
                    )
                    t2r_dp_token = _iket.range_start(
                        "T2R_dP(i)",
                        chunk_payload,
                    )
                    if cutlass.const_expr(sub_tile % 2 == 0):
                        cute.copy(dp_copy, dp_source, r_dp)
                    else:
                        cute.copy(dp_copy_pp, dp_source_pp, r_dp)
                    cute.arch.fence_view_async_tmem_load()
                    # edge -> leader's dp acquire for pass t+2.
                    pipe_dp_done.consumer_release(dp_state)
                    dp_state.advance()
                    _iket.range_end(
                        t2r_dp_token,
                        chunk_payload,
                    )

                    math_pd_token = _iket.range_start(
                        "MATH_PD(i)",
                        chunk_payload,
                    )
                    # v5 math: same packed f32x2 pair schedule as v17a/
                    # E2, but the constants index the PHYSICAL head of
                    # fragment column n (class head-map note):
                    #   head(t, n) = (t//2)*64 + (n//16)*32
                    #              + (t%2)*16 + (n%16).
                    # Packed pairs remain per-value lookups with
                    # DISTINCT pair constants (the assumption-free
                    # shape), so the h16-box seam inside a pair needs
                    # no special case.
                    phase_p_payload = chunk_payload
                    phase_ds_payload = chunk_payload + Int32(1)
                    head_base = Int32(
                        (sub_tile // 2) * self.H_TILE_CTA
                        + (sub_tile % 2) * self.SUB_TILE_BOX
                    )

                    # --- P phase: packed softmax, then publish P. ---
                    math_softmax_token = _iket.range_start(
                        "MATH_SOFTMAX(i)",
                        phase_p_payload,
                    )
                    softmax_scale_log2_e = scale_softmax * Float32(
                        math.log2(math.e)
                    )
                    assert cute.size(r_score) == self.SUB_TILE_VALS
                    for pair in cutlass.range_constexpr(
                        self.SUB_TILE_VALS // 2
                    ):
                        v0 = 2 * pair
                        v1 = v0 + 1
                        n_0 = Int32(
                            cute.get(
                                score_coordinates[v0],
                                mode=[1],
                            )
                        )
                        n_1 = Int32(
                            cute.get(
                                score_coordinates[v1],
                                mode=[1],
                            )
                        )
                        head_0 = (
                            head_base
                            + n_0 % Int32(self.SUB_TILE_BOX)
                            + (n_0 // Int32(self.SUB_TILE_BOX))
                            * Int32(self.SUB_TILE_H)
                        )
                        head_1 = (
                            head_base
                            + n_1 % Int32(self.SUB_TILE_BOX)
                            + (n_1 // Int32(self.SUB_TILE_BOX))
                            * Int32(self.SUB_TILE_H)
                        )
                        s_0, s_1 = cute.arch.fma_packed_f32x2(
                            (r_score[v0], r_score[v1]),
                            (
                                softmax_scale_log2_e,
                                softmax_scale_log2_e,
                            ),
                            (
                                softmax_stats[head_0, 0],
                                softmax_stats[head_1, 0],
                            ),
                        )
                        p_0 = cute.math.exp2(s_0, fastmath=True)
                        p_1 = cute.math.exp2(s_1, fastmath=True)
                        # Keep FP32 P live for the dS phase.
                        r_score[v0] = p_0
                        r_score[v1] = p_1
                        r_p[v0] = self.element_dtype(p_0)
                        r_p[v1] = self.element_dtype(p_1)
                    _iket.range_end(
                        math_softmax_token,
                        phase_p_payload,
                    )

                    # ONE pds acquire per SUB-TILE (spec Z4, 2-stage):
                    # edge: leader grads(t-2) consumer_release (UMMA-
                    # tracked -- the previous user of pds stage t%2;
                    # its G3/G4 reads of the slab bytes this publish
                    # may overwrite are a subset, see the stage note
                    # in the leader).  MATH_PDS_ACQ payload widens to
                    # per-sub-tile (bundle*4 + t, spec Z8).
                    math_pds_acq_token = _iket.range_start(
                        "MATH_PDS_ACQ(i)",
                        loop_iter * Int32(self.SUB_TILES)
                        + Int32(sub_tile),
                    )
                    pipe_pds.producer_acquire(pds_state)
                    if cutlass.const_expr(sub_tile == 0):
                        # dq_b free gate, once per bundle and BEFORE
                        # any dq_b write on this CTA (the own stmatrix
                        # below; the relay peer push is transitively
                        # gated by pds_ready).  edge: leader's group
                        # commit after the previous bundle's LAST G5
                        # wave (covers both D-rounds and both waves,
                        # errata #1).
                        pipe_dqb_free.consumer_wait(dqb_free_state)
                        pipe_dqb_free.consumer_release(dqb_free_state)
                        dqb_free_state.advance()
                    _iket.range_end(
                        math_pds_acq_token,
                        loop_iter * Int32(self.SUB_TILES)
                        + Int32(sub_tile),
                    )

                    math_store_token = _iket.range_start(
                        "MATH_STORE(i)",
                        phase_p_payload,
                    )
                    # Publish the sub-tile's two h16 P boxes into the
                    # NATURAL chunk image (J-mode pre-sliced target).
                    r_p_store = thread_copy_r2s.retile(r_p)
                    assert (
                        t_rs_p_tiles[sub_tile].shape
                        == r_p_store.shape
                    ), (
                        str(t_rs_p_tiles[sub_tile].shape),
                        str(r_p_store.shape),
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        r_p_store,
                        t_rs_p_tiles[sub_tile],
                    )
                    _iket.range_end(
                        math_store_token,
                        phase_p_payload,
                    )

                    # --- dS phase: packed dS math, then publish dS. ---
                    math_softmax_token = _iket.range_start(
                        "MATH_SOFTMAX(i)",
                        phase_ds_payload,
                    )
                    # Column-axis delta, per-value lookup (distinct
                    # pair constants, same rationale as the P phase).
                    for pair in cutlass.range_constexpr(
                        self.SUB_TILE_VALS // 2
                    ):
                        v0 = 2 * pair
                        v1 = v0 + 1
                        n_0 = Int32(
                            cute.get(
                                score_coordinates[v0],
                                mode=[1],
                            )
                        )
                        n_1 = Int32(
                            cute.get(
                                score_coordinates[v1],
                                mode=[1],
                            )
                        )
                        head_0 = (
                            head_base
                            + n_0 % Int32(self.SUB_TILE_BOX)
                            + (n_0 // Int32(self.SUB_TILE_BOX))
                            * Int32(self.SUB_TILE_H)
                        )
                        head_1 = (
                            head_base
                            + n_1 % Int32(self.SUB_TILE_BOX)
                            + (n_1 // Int32(self.SUB_TILE_BOX))
                            * Int32(self.SUB_TILE_H)
                        )
                        ds_0, ds_1 = cute.arch.add_packed_f32x2(
                            (r_dp[v0], r_dp[v1]),
                            (
                                softmax_stats[head_0, 1],
                                softmax_stats[head_1, 1],
                            ),
                        )
                        ds_0, ds_1 = cute.arch.mul_packed_f32x2(
                            (ds_0, ds_1),
                            (r_score[v0], r_score[v1]),
                        )
                        ds_0, ds_1 = cute.arch.mul_packed_f32x2(
                            (ds_0, ds_1),
                            (scale_softmax, scale_softmax),
                        )
                        r_ds[v0] = self.element_dtype(ds_0)
                        r_ds[v1] = self.element_dtype(ds_1)
                    _iket.range_end(
                        math_softmax_token,
                        phase_ds_payload,
                    )

                    math_store_token = _iket.range_start(
                        "MATH_STORE(i)",
                        phase_ds_payload,
                    )
                    r_ds_store = thread_copy_r2s.retile(r_ds)
                    # Sub-tile dS publish into the natural chunk image
                    # (chunk (t//2)'s J window; chunk (1-rank) doubles
                    # as half of the relay's 8,192 B DSM payload once
                    # both its windows have landed).
                    assert (
                        t_rs_ds_tiles[sub_tile].shape
                        == r_ds_store.shape
                    ), (
                        str(t_rs_ds_tiles[sub_tile].shape),
                        str(r_ds_store.shape),
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        r_ds_store,
                        t_rs_ds_tiles[sub_tile],
                    )
                    # dq_b own-half image (addendum section 1): the
                    # sub-tiles with c == rank carry ONLY own-H64
                    # columns, so the same register fragment re-stores
                    # into dq_b sub_img[rank]'s J window -- zero TMEM
                    # re-read, natural head order preserved ("4x h16
                    # box" granularity, spec Z4).  The free gate was
                    # taken at the t == 0 phase.
                    if Int32(sub_tile // 2) == rank:
                        cute.copy(
                            tiled_copy_r2s,
                            r_ds_store,
                            t_rs_dqb_tiles[sub_tile % 2],
                        )

                    # No validity mask (baseline-identical invariant
                    # pair): invalid columns see S=dP=0 from zero-
                    # filled K rows, so P/dS stay finite; dQ is
                    # protected by zero-filled K_dQ rows and dKV
                    # garbage columns are dropped by the drain
                    # predicates (global_n < topk, kv_index >= 0).

                    cute.arch.fence_view_async_shared()
                    # Sub-tile pds commit (spec Z4: math is the pds
                    # producer now; 4 commits/bundle).  edge -> leader
                    # grads(t) consumer_wait; the fence above orders
                    # the P/dS/dq_b stmatrix stores before the
                    # leader's descriptor reads.
                    pipe_pds.producer_commit(pds_state)
                    pds_state.advance()
                    _iket.range_end(
                        math_store_token,
                        phase_ds_payload,
                    )
                    _iket.range_end(
                        math_pd_token,
                        chunk_payload,
                    )
                # Once per BUNDLE: the DSM push and dqb gate arming
                # belong to the relay warp (W18); the pds commit moved
                # to the per-sub-tile cadence above (spec Z4).  Each
                # math thread hands off with one release-semantics
                # mbarrier arrive after its own fenced stores (covers
                # all four sub-tile publishes AND the dq_b own image).
                # edge -> relay's pds_ready wait (bundle-level).
                math_bar1_token = _iket.range_start(
                    "MATH_BAR1(i)",
                    loop_iter,
                )
                cute.arch.mbarrier_arrive(
                    pds_ready_mbars,
                    rank,
                )
                _iket.range_end(
                    math_bar1_token,
                    loop_iter,
                )
            # v5.2: the dQ TMA epilogue is RETIRED (change order item
            # 2) -- the reducer's last-bundle RMW already wrote the
            # terminal mdQ values, so the per-token seam disappears.
            # Only the F3 dqb-free credit drain survives (pre-arm 1 +
            # 1/bundle = tile_count+1 commits vs tile_count in-loop
            # consumes; without this the leader's producer_tail waits
            # a release that never comes).
            if tile_count > Int32(0):
                pipe_dqb_free.consumer_wait(dqb_free_state)
                pipe_dqb_free.consumer_release(dqb_free_state)
                dqb_free_state.advance()
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
            # v5: math owns the pds producer role (spec Z4); the
            # producer_tail is only a drain of outstanding producer
            # credit (the v12 S1 epilogue-ordering rationale is moot
            # now that the epilogue is retired).
            if tile_count > Int32(0):
                pipe_pds.producer_tail(pds_state)

        elif warp_idx < Int32(self.MMA_WARP):
            # --- reduce, v5.2: FUSED dV+dK drains (8/bundle; V == K
            # makes d(KV) = dV + dK share one destination, so the pair
            # is summed in registers and issued as a SINGLE red.global
            # stream -- the T3-HO4 P0 gate measurement) + the dQ
            # eviction offload (16 blocks/bundle; plain LDG/FADD/STG,
            # the cluster owns its token's dQ rows exclusively).
            # This warpgroup pair is the only idle-enough role that
            # can offload dQ: tcgen05.ld gives each warp only its own
            # 32-DP slice, so the [128 DP x 16 col] slot needs a full
            # warpgroup (W18/W19 = 2 warps = 64 DP, physically short),
            # and the drain fusion frees ~half the reduce throughput.
            _iket.mark("ROLE_REDUCE", rank)
            rtx = tidx - Int32(self.REDUCE_THREAD_BEGIN)
            dkv_wait = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.MMA_DONE_STAGES,
            )
            dkv_rel = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.MMA_DONE_STAGES,
            )
            evict_wait = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                2,
            )
            evict_rel = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                2,
            )
            for loop_iter in cutlass.range(tile_count):
                tile_index = tile_count - Int32(1) - loop_iter
                # 8 fused (dV, dK) drains, leader commit order (pair-
                # major then D-round).  Fused-block payload =
                # b*8 + pair*4 + r (change order item 3).
                for g_pair in cutlass.range_constexpr(2):
                    for d_round in cutlass.range_constexpr(
                        self.DKV_D_ROUNDS
                    ):
                        # edge: leader dkv commits for (P, r, dV) and
                        # (P, r, dK) -- consumed as a pair from the
                        # 6-deep ring.
                        # [fix-r7] static slot selection at the call
                        # site: slot = (2r + p) % 4, pair-independent.
                        dkv_wait, dkv_rel = self._drain_dkv_fused_v52(
                            t_dkv_army[(2 * d_round) % 4],
                            t_dkv_army[(2 * d_round + 1) % 4],
                            mdKV_acc,
                            mTopkIdxs,
                            tile_index,
                            Int32(d_round),
                            topk,
                            token_idx,
                            batch_idx,
                            rtx,
                            rank,
                            loop_iter * Int32(8)
                            + Int32(g_pair * 4 + d_round),
                            pipe_dkv_done,
                            dkv_wait,
                            dkv_rel,
                        )
                # 16 dQ eviction offloads, leader issue order
                # (r_old, t, d_half); DQ_EPI payload keeps the ordered
                # b*16 + t*4 + r encoding (r = 2*r_old + d_half).
                is_first = loop_iter == Int32(0)
                is_last = loop_iter == tile_count - Int32(1)
                for r_old in cutlass.range_constexpr(2):
                    for t_window in cutlass.range_constexpr(4):
                        for d_half in cutlass.range_constexpr(2):
                            # edge: leader dq_evict commit for block
                            # (t, 2*r_old + d_half).
                            evict_wait, evict_rel = (
                                self._offload_dq_block_v52(
                                    t_window,
                                    2 * r_old + d_half,
                                    t_dq_rot[d_half],
                                    mdQ_acc,
                                    mdQ,
                                    token_idx,
                                    batch_idx,
                                    rtx,
                                    rank,
                                    is_first,
                                    is_last,
                                    loop_iter * Int32(16)
                                    + Int32(
                                        t_window * 4
                                        + 2 * r_old
                                        + d_half
                                    ),
                                    pipe_dq_evict,
                                    evict_wait,
                                    evict_rel,
                                )
                            )

        elif warp_idx == Int32(self.MMA_WARP):
            # --- leader MMA: rotated schedule.  The follower CTA's MMA warp
            # executes no pipeline operation at all (FA4 rule).
            _iket.mark(
                self.IKET_V2_NATIVE_PROVENANCE,
                rank,
            )
            if is_leader_cta:
                _iket.mark("ROLE_MMA", rank)
                s_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.SCORE_DONE_STAGES,
                )
                dp_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.SCORE_DONE_STAGES,
                )
                # v5.1b: strip stream consumer (stage = t % 2) and
                # the 1-stage K residency gate.
                strip_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    2,
                )
                kres_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    1,
                )
                round_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.ROUND_STAGES,
                )
                # v5: pds is 2-stage, consumed per grads(t) sub-tile.
                pds_cons = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    2,
                )
                dkv_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.MMA_DONE_STAGES,
                )
                # v5.2: the dQ eviction producer (2 rotating slots).
                dq_evict_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    2,
                )
                dqb_free_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    1,
                )
                if tile_count > Int32(0):
                    # v5.1b: the one-shot panel readiness gates are
                    # retired -- score operand readiness now rides
                    # pipe_strip (per pass) and pipe_kres (per bundle).
                    # Pre-armed initial dq_b free commit (empty group):
                    # bundle 0's math consumes it unconditionally.
                    pipe_dqb_free.producer_acquire(dqb_free_prod)
                    pipe_dqb_free.producer_commit(dqb_free_prod)
                    dqb_free_prod.advance()

                dq_kd_frags = (
                    dq_kd_fragment_a,
                    dq_kd_fragment_b,
                )
                grad_frags = (
                    grad_fragment_a,
                    grad_fragment_b,
                )
                # v5.1 pair-batched bundle schedule (change order
                # v5.1 item 1, total order):
                #   score(0); score(1); score(2); grads(pair0={t0,t1});
                #   score(3); grads(pair1={t2,t3}); G5 r0; G5 r1;
                #   dq_b free group-commit.
                # The score/math sub-tile pipeline is UNTOUCHED (math
                # commits pds per sub-tile, 4/bundle); only the grads
                # consumption batches to pair granularity -- protocol
                # tax redemption: 8 blocks/pair with ONE dkv+round
                # handshake set per block, drains 32 -> 16/bundle.
                # grads(pair0) waits BOTH pds stages (math(0)+math(1))
                # up front; math(2)/math(3) publish over grads(pair0)
                # execution and gate grads(pair1).  v5.2: the bundle
                # tail is the 16-block dQ EVICTION plane (rotating
                # 2x16-column slots, offloaded by the reducers); the
                # single-FIFO round ring keeps the kdq generations
                # behind the 16 grad gens (see ROUND_GENS_PER_TILE).
                # v5.1b: K is RESIDENT -- the
                # bundle-head kres wait admits the gather's single
                # 8-piece fill, and the release after score(3) is the
                # tcgen05-tracked completion edge that gates the
                # gather's NEXT-bundle fill.
                for loop_iter in cutlass.range(tile_count):
                    dqb_parity = loop_iter & Int32(1)

                    # edge: gather's K(i) fill commit (8 pieces + one
                    # cp.async drain + fence).
                    pipe_kres.consumer_wait(kres_cons)

                    # The @cute.jit boundary re-materializes state
                    # arguments: helper advances must come back through
                    # the returns (rev3 dominance failure, lesson #14).
                    # sub_tile is a Python int (range_constexpr), so
                    # the operand/accumulator selection below is trace-
                    # time argument routing, not a staged branch.
                    for sub_tile in cutlass.range_constexpr(
                        self.SUB_TILES
                    ):
                        strip_cons, s_prod, dp_prod = (
                            self._issue_score_pass_v5(
                                sub_tile,
                                score_tiled_mma,
                                dp_tiled_mma,
                                t_score
                                if sub_tile % 2 == 0
                                else t_score_pp,
                                t_dp
                                if sub_tile % 2 == 0
                                else t_dp_pp,
                                score_k_fragment,
                                dp_k_fragment,
                                score_q_fragment,
                                score_do_fragment,
                                pipe_strip,
                                strip_cons,
                                pipe_s_done,
                                s_prod,
                                pipe_dp_done,
                                dp_prod,
                                loop_iter,
                            )
                        )
                        if cutlass.const_expr(sub_tile == 2):
                            # v5.1 rotation point: grads(pair0) issues
                            # here; score(3) follows immediately and
                            # its execution overlaps pair0's MMA
                            # window (math(2)/math(3) publish over it).
                            pds_cons, dkv_prod, round_cons = (
                                self._issue_grads_pair_v51(
                                    0,
                                    dkv_tiled_mma,
                                    t_dkv_army[0],
                                    t_dkv_army[1],
                                    t_dkv_army[2],
                                    t_dkv_army[3],
                                    p_fragment,
                                    ds_fragment,
                                    grad_frags[0],
                                    grad_frags[1],
                                    pipe_pds,
                                    pds_cons,
                                    pipe_dkv_done,
                                    dkv_prod,
                                    pipe_round,
                                    round_cons,
                                    loop_iter,
                                )
                            )
                    # v5.1b: K residency release AFTER score(3) --
                    # tcgen05-tracked, so the empty edge fires when
                    # score(3)(i)'s reads complete: exactly the
                    # "K(i+1) gather gate = score(3)(i) completion
                    # edge" contract.  Issued before grads(pair1) so
                    # the gather's next-bundle fill overlaps the whole
                    # grads/G5 tail.
                    pipe_kres.consumer_release(kres_cons)
                    kres_cons.advance()

                    # grads(pair1): the bundle's last pair has no
                    # score to hide under (score(0) of bundle b+1
                    # cannot start before G5(b) frees the TMEM/ring
                    # order -- bundle boundary shape unchanged).
                    pds_cons, dkv_prod, round_cons = (
                        self._issue_grads_pair_v51(
                            1,
                            dkv_tiled_mma,
                            t_dkv_army[0],
                            t_dkv_army[1],
                            t_dkv_army[2],
                            t_dkv_army[3],
                            p_fragment,
                            ds_fragment,
                            grad_frags[0],
                            grad_frags[1],
                            pipe_pds,
                            pds_cons,
                            pipe_dkv_done,
                            dkv_prod,
                            pipe_round,
                            round_cons,
                            loop_iter,
                        )
                    )

                    # v5.2 G5 EVICTION plane (change order item 2):
                    # SIXTEEN (t, r) blocks -- 4 D128-rounds x 4 h32
                    # windows -- each a fresh rotating accumulator,
                    # offloaded by the reduce warps through
                    # pipe_dq_evict.  Issue order (r_old, t, d_half):
                    # the round ring holds ONE kdq gen pair at a time,
                    # which forces the r_old grouping (DQ_EPI payload
                    # keeps the ordered b*16 + t*4 + r encoding; the
                    # in-bundle sequence is non-monotonic across the
                    # two r_old groups, see V5_BUILD_LOG).  Rotating
                    # slot index == d_half (block ordinal parity --
                    # r_old*8 + t*2 + d_half mod 2 == d_half, static).
                    # Both dq_b wave gates are HOISTED: every block's
                    # K chain reads both kv waves.
                    # edge: relay arms -- own image stored (math) AND
                    # peer push landed (errata #2 pair).
                    _mbarrier_wait_acquire_cluster(
                        dqb_mbars,
                        dqb_parity,
                    )
                    _mbarrier_wait_acquire_cluster(
                        dqb_mbars + 1,
                        dqb_parity,
                    )
                    for r_old in cutlass.range_constexpr(2):
                        # kdq gen pair (r_old) held for the whole
                        # 8-block group (clone+advance double wait),
                        # released together after it.
                        # edge: W17 kdq handshake commits (gens
                        # 16+2*r_old / 17+2*r_old).
                        pipe_round.consumer_wait(round_cons)
                        round_cons_w1 = round_cons.clone()
                        round_cons_w1.advance()
                        pipe_round.consumer_wait(round_cons_w1)
                        for t_window in cutlass.range_constexpr(4):
                            for d_half in cutlass.range_constexpr(2):
                                # edge: the reducer offloaded the
                                # block two generations back (2
                                # rotating dQ slots).
                                pipe_dq_evict.producer_acquire(
                                    dq_evict_prod
                                )
                                self._issue_dq_block_v52(
                                    t_window,
                                    d_half,
                                    dq_evict_tiled_mma,
                                    t_dq_rot[d_half],
                                    dq_kd_frags[0],
                                    dq_kd_frags[1],
                                    dq_ds_fragment,
                                )
                                # edge -> the reducer's DQ_EPI wait
                                # for this block (tcgen05-tracked).
                                pipe_dq_evict.producer_commit(
                                    dq_evict_prod
                                )
                                dq_evict_prod.advance()
                        pipe_round.consumer_release(round_cons)
                        round_cons.advance()
                        pipe_round.consumer_release(round_cons)
                        round_cons.advance()
                    # edge -> math(b+1)'s bundle-head dqb-free wait
                    # (the group commit tracks every issued MMA, so it
                    # covers all sixteen blocks -- errata #1 shape).
                    pipe_dqb_free.producer_acquire(dqb_free_prod)
                    pipe_dqb_free.producer_commit(dqb_free_prod)
                    dqb_free_prod.advance()

                # v5.2 tail: the dQ TMA epilogue retired with the
                # eviction (the last bundle's RMW wrote the terminal
                # mdQ values), so dq_done is gone -- the tail is now a
                # pure producer-credit drain.
                if tile_count > Int32(0):
                    tail_token = _iket.range_start(
                        "TAIL",
                        tile_count - Int32(1),
                    )
                    pipe_s_done.producer_tail(s_prod)
                    pipe_dp_done.producer_tail(dp_prod)
                    pipe_dkv_done.producer_tail(dkv_prod)
                    pipe_dq_evict.producer_tail(dq_evict_prod)
                    pipe_dqb_free.producer_tail(dqb_free_prod)
                    _iket.range_end(
                        tail_token,
                        tile_count - Int32(1),
                    )

        elif warp_idx == Int32(self.LOAD_WARP):
            # --- load, v5.1b: the Q/dO STRIP stream (item 6; the
            # one-shot panels are retired) + 20 round gens per bundle.
            # Per bundle: 4 strip-pair gens (stage = t % 2, one Q+dO
            # [h16 x D512] pair per score pass, synchronous TMA on the
            # reused stationary mbar), then g0..g15 the 16 full-wide
            # (pair, D-round) x (dO, Q) gradient gens (software-
            # pipelined over two rotating barriers), then g16/g17 kdq
            # D-round 0 and g18/g19 kdq D-round 1 handshakes.
            _iket.mark("ROLE_KV_LOAD", rank)
            lane_idx = tidx % Int32(32)
            round_acq = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            round_com = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ROUND_STAGES,
            )
            strip_acq = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                2,
            )
            strip_com = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                2,
            )
            tma_phase_0 = Int32(0)
            tma_phase_1 = Int32(0)
            strip_tma_phase = Int32(0)
            # One strip = [h16 x D512] bf16 (16,384 B); a strip-pair
            # gen carries Q + dO on one completion mbar.
            strip_stage_bytes = (
                self.STATIONARY_TILE_H
                * self.D_HEAD
                * (self.element_dtype.width // 8)
            )
            if tile_count > Int32(0):
                for loop_iter in cutlass.range(tile_count):
                    tile_index = (
                        tile_count - Int32(1) - loop_iter
                    )
                    # v5.1b (item 6): the strip-pair stream -- four
                    # generations per bundle, stage = t % 2, filled
                    # ahead of the round gens (the score passes at the
                    # bundle head are the earliest consumers).  Strip
                    # t holds the CTA's window rows
                    # H[(t//2)*64 + rank*32 + (t%2)*16 : +16)
                    # = gmem h16-tile 4*(t//2) + (t%2) + 2*rank.
                    # Synchronous per gen: expect_tx covers the Q+dO
                    # pair on the (reused) stationary mbar, W17 waits
                    # its own TMA, then commits the strip generation.
                    # LOAD_QDO span name survives with per-gen payload
                    # bundle*4 + t.
                    for strip_t in cutlass.range_constexpr(
                        self.SUB_TILES
                    ):
                        load_qdo_token = _iket.range_start(
                            "LOAD_QDO",
                            loop_iter * Int32(self.SUB_TILES)
                            + Int32(strip_t),
                        )
                        # edge: leader's strip release -- tcgen05-
                        # tracked score(t-2) B-read completion (init
                        # credits cover t = 0, 1 of bundle 0).
                        # v5.2 MAT_ACQ probe (gen ordinals: strips
                        # 0-3, wide 4-19, kdq 20-23).
                        mat_acq_token = _iket.range_start(
                            "MAT_ACQ(m,g)",
                            loop_iter * Int32(32)
                            + Int32(strip_t),
                        )
                        pipe_strip.producer_acquire(strip_acq)
                        strip_acq.advance()
                        _iket.range_end(
                            mat_acq_token,
                            loop_iter * Int32(32)
                            + Int32(strip_t),
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                stationary_tma_mbars,
                                2 * strip_stage_bytes,
                            )
                        cute.copy(
                            tma_atom_q,
                            t_q_gmem[
                                None,
                                Int32(
                                    4 * (strip_t // 2)
                                    + strip_t % 2
                                )
                                + Int32(2) * rank,
                                0,
                            ],
                            t_q_smem[None, strip_t % 2],
                            tma_bar_ptr=stationary_tma_mbars,
                        )
                        cute.copy(
                            tma_atom_do,
                            t_do_gmem[
                                None,
                                Int32(
                                    4 * (strip_t // 2)
                                    + strip_t % 2
                                )
                                + Int32(2) * rank,
                                0,
                            ],
                            t_do_smem[None, strip_t % 2],
                            tma_bar_ptr=stationary_tma_mbars,
                        )
                        cute.arch.mbarrier_wait(
                            stationary_tma_mbars,
                            strip_tma_phase,
                        )
                        strip_tma_phase = Int32(1) - strip_tma_phase
                        with cute.arch.elect_one():
                            pipe_strip.producer_commit(strip_com)
                        strip_com.advance()
                        _iket.range_end(
                            load_qdo_token,
                            loop_iter * Int32(self.SUB_TILES)
                            + Int32(strip_t),
                        )
                    # g0..g15: FULL-WIDE round-gen B TMA fills, one
                    # gen per (pair, D-round, tensor), pair-major
                    # (v5.1 item 3): dO -> buf A (barrier 0), Q -> buf
                    # B (barrier 1).  Each gen is ONE [h64 x own-D64]
                    # box (expect_tx = 8,192 B, GMEM-natural [H,D]
                    # rows through the transposed [D,H] view at
                    # h64-tile P == chunk P -- the pair's two h32
                    # sub-tile head sets tile it exactly).  Software-
                    # pipelined exactly like v17a/v32: issue gen q,
                    # then wait/commit gen q-1 on the other barrier,
                    # so no barrier is re-armed while pending.
                    # MAT_QDO narrows to per-pair spans (payload =
                    # bundle*2 + pair); by the pipelining, span
                    # pair-1's tail wait rides inside span pair's
                    # window (documented straddle, V5_BUILD_LOG).
                    mat_qdo_token = _iket.range_start(
                        "MAT_QDO(m,r)",
                        loop_iter * Int32(2),
                    )
                    for flat_gen in cutlass.range_constexpr(16):
                        gen_pair = flat_gen // 8
                        grad_round = (flat_gen % 8) // 2
                        tensor_kind = flat_gen % 2
                        if cutlass.const_expr(
                            flat_gen % 8 == 0 and flat_gen > 0
                        ):
                            _iket.range_end(
                                mat_qdo_token,
                                loop_iter * Int32(2)
                                + Int32(gen_pair - 1),
                            )
                            mat_qdo_token = _iket.range_start(
                                "MAT_QDO(m,r)",
                                loop_iter * Int32(2)
                                + Int32(gen_pair),
                            )
                        # edge: leader released gen (flat_gen - 2)
                        # inside grads(pair (flat_gen-2)//8) -- 2-stage
                        # ring.
                        mat_acq_token = _iket.range_start(
                            "MAT_ACQ(m,g)",
                            loop_iter * Int32(32)
                            + Int32(4 + flat_gen),
                        )
                        pipe_round.producer_acquire(round_acq)
                        round_acq.advance()
                        _iket.range_end(
                            mat_acq_token,
                            loop_iter * Int32(32)
                            + Int32(4 + flat_gen),
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                round_tma_mbars + tensor_kind,
                                grad_a_stage_bytes,
                            )
                        if cutlass.const_expr(tensor_kind == 0):
                            cute.copy(
                                tma_atom_dot,
                                t_dot_gmem[
                                    None,
                                    grad_round,
                                    gen_pair,
                                ],
                                t_dot_smem_a[None, 0],
                                tma_bar_ptr=round_tma_mbars,
                            )
                        else:
                            cute.copy(
                                tma_atom_qt,
                                t_qt_gmem[
                                    None,
                                    grad_round,
                                    gen_pair,
                                ],
                                t_qt_smem_b[None, 0],
                                tma_bar_ptr=round_tma_mbars + 1,
                            )
                        if cutlass.const_expr(flat_gen > 0):
                            if cutlass.const_expr(
                                (flat_gen - 1) % 2 == 0
                            ):
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

                    # Drain the last in-flight fill (gen 15, barrier 1).
                    cute.arch.mbarrier_wait(
                        round_tma_mbars + 1,
                        tma_phase_1,
                    )
                    tma_phase_1 = Int32(1) - tma_phase_1
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    _iket.range_end(
                        mat_qdo_token,
                        loop_iter * Int32(2) + Int32(1),
                    )

                    # g16/g17: kdq D-round 0, both kv-wave images in
                    # one gather pass (both stage credits held).  Kept
                    # behind the grad gens (bundle-tail G5 + single-
                    # ring FIFO); the named-barrier rendezvous ORDER
                    # with the gather warps (r0(b), r1(b), ...) is
                    # unchanged, so the frozen kdq machine is
                    # untouched.  edge: ring credits g14/g15 freed by
                    # the leader's grads(pair1) reads.
                    route_k_token = _iket.range_start(
                        "ROUTE_K(i)",
                        Int32(2) * loop_iter,
                    )
                    mat_acq_token = _iket.range_start(
                        "MAT_ACQ(m,g)",
                        loop_iter * Int32(32) + Int32(20),
                    )
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    _iket.range_end(
                        mat_acq_token,
                        loop_iter * Int32(32) + Int32(20),
                    )
                    mat_acq_token = _iket.range_start(
                        "MAT_ACQ(m,g)",
                        loop_iter * Int32(32) + Int32(21),
                    )
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    _iket.range_end(
                        mat_acq_token,
                        loop_iter * Int32(32) + Int32(21),
                    )
                    # v8: the gather warps write the K_dQ images (128
                    # threads vs 32).  Handshake A publishes "both stage
                    # credits held"; handshake B returns "both images
                    # written, cp.async-drained, and fenced by each filling
                    # thread", after which the two commits are safe.
                    self.kdq_barrier.arrive_and_wait()
                    self.kdq_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    _iket.range_end(
                        route_k_token,
                        Int32(2) * loop_iter,
                    )

                    # g18/g19: kdq D-round 1 handshake (both kv-wave
                    # images).  edge: its stage credits free when the
                    # leader consumes g16/g17 in G5 r0, which is why
                    # the gather side runs this rendezvous AFTER the
                    # next bundle's chase begins (piece-2 boundary).
                    route_k1_token = _iket.range_start(
                        "ROUTE_K(i)",
                        Int32(2) * loop_iter + Int32(1),
                    )
                    mat_acq_token = _iket.range_start(
                        "MAT_ACQ(m,g)",
                        loop_iter * Int32(32) + Int32(22),
                    )
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    _iket.range_end(
                        mat_acq_token,
                        loop_iter * Int32(32) + Int32(22),
                    )
                    mat_acq_token = _iket.range_start(
                        "MAT_ACQ(m,g)",
                        loop_iter * Int32(32) + Int32(23),
                    )
                    pipe_round.producer_acquire(round_acq)
                    round_acq.advance()
                    _iket.range_end(
                        mat_acq_token,
                        loop_iter * Int32(32) + Int32(23),
                    )
                    self.kdq_barrier.arrive_and_wait()
                    self.kdq_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    with cute.arch.elect_one():
                        pipe_round.producer_commit(round_com)
                    round_com.advance()
                    _iket.range_end(
                        route_k1_token,
                        Int32(2) * loop_iter + Int32(1),
                    )
                pipe_round.producer_tail(round_acq)
                pipe_strip.producer_tail(strip_acq)

        elif warp_idx == Int32(self.RELAY_WARP):
            # --- relay, v3.2: the dq_b peer-push engine.  Per bundle:
            # (1) wait the math handoff (count-128 pds_ready; it covers
            #     the P/dS slab publishes AND the dq_b own image, and
            #     math arrives it only after its dqb-free wait, so this
            #     push is transitively free-gated);
            # (2) ONE 8,192 B bulk DSM: the LOCAL dS slab sub[1-rank]
            #     (= dS^T[own-kv64 x peer-H64], published contiguous by
            #     the chunk-major stacking) lands in the PEER's dq_b
            #     sub_img[rank] -- strictly rank-symmetric offsets.
            #     (st.async register push is the registered V32-TODO
            #     upgrade; this is the addendum's fallback form.)
            # (3) [v5, spec Z4] the pds commit MOVED to math's per-
            #     sub-tile cadence -- this warp no longer touches
            #     pipe_pds.  Its dS-slab source read is WAR-covered
            #     transitively: mb_dqb(b) full => both pushes landed
            #     => G5(b) may issue => leader reaches score(0)(b+1)
            #     => math(b+1) can first overwrite the slab.
            # (4) errata #2 relay-arrive: arrive mb_dqb[rank] at the
            #     leading CTA for the OWN image, then wait the LOCAL
            #     landing mbar (the peer's push INTO me -- the
            #     completion-tx mbar lives at the destination, so only
            #     I can observe it) and arrive mb_dqb[1-rank].
            # Deadlock shape: my send depends only on my math; my
            # landing wait depends on the peer's send; the same cross
            # pattern as v11/v17a's relay.
            if tidx % Int32(32) == Int32(0):
                landing_phase = Int32(0)
                ready_phase = Int32(0)
                dqb_dst_ptr = dq_b_raw + rank * Int32(
                    self.PDS_BLOCK_ELEMENTS
                )
                dqb_src_ptr = ds_slab_raw + (
                    Int32(1) - rank
                ) * Int32(self.PDS_BLOCK_ELEMENTS)
                for loop_iter in cutlass.range(tile_count):
                    cute.arch.mbarrier_wait(
                        pds_ready_mbars,
                        ready_phase,
                    )
                    ready_phase = Int32(1) - ready_phase
                    route_ds_token = _iket.range_start(
                        "ROUTE_dS(i)",
                        loop_iter,
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        landing_mbars,
                        self.PDS_BLOCK_BYTES,
                        peer_cta_rank_in_cluster=peer_rank,
                    )
                    # Helper signature is (source_local, dest_at_peer)
                    # -- the V31_SURGERY_SPEC section 0 convention and
                    # both base-class call sites agree.
                    _cpasync_bulk_s2cluster(
                        dqb_src_ptr,
                        dqb_dst_ptr,
                        landing_mbars,
                        self.PDS_BLOCK_BYTES,
                        peer_rank,
                    )
                    _iket.range_end(
                        route_ds_token,
                        loop_iter,
                    )
                    # Own sub_img[rank] is ready (math's stmatrix is
                    # fenced under pds_ready): arrive the leading CTA's
                    # ready gate for wave `rank`.
                    if rank == Int32(0):
                        cute.arch.mbarrier_arrive(
                            dqb_mbars,
                            Int32(0),
                        )
                    else:
                        cute.arch.mbarrier_arrive(
                            dqb_mbars + 1,
                            Int32(0),
                        )
                    # Peer landing observed locally, then relay-arrive
                    # the gate for wave `1-rank` (errata #2).
                    _mbarrier_wait_acquire_cluster(
                        landing_mbars,
                        landing_phase,
                    )
                    if rank == Int32(0):
                        cute.arch.mbarrier_arrive(
                            dqb_mbars + 1,
                            Int32(0),
                        )
                    else:
                        cute.arch.mbarrier_arrive(
                            dqb_mbars,
                            Int32(0),
                        )
                    landing_phase = Int32(1) - landing_phase

        # ==================================================================
        # Common tail: full-cluster rendezvous, then TMEM release.
        # ==================================================================
        tmem.relinquish_alloc_permit()
        self.cta_barrier.arrive_and_wait()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)

    @cute.jit
    def _issue_score_pass_v5(
        self,
        sub_tile: cutlass.Constexpr[int],
        score_tiled_mma: cute.TiledMma,
        dp_tiled_mma: cute.TiledMma,
        t_s: cute.Tensor,
        t_dp: cute.Tensor,
        k_fragment: cute.Tensor,
        dp_k_fragment: cute.Tensor,
        q_fragment: cute.Tensor,
        do_fragment: cute.Tensor,
        strip_pipeline,
        strip_consumer_state: pipeline.PipelineState,
        s_done_pipeline,
        s_producer_state: pipeline.PipelineState,
        dp_done_pipeline,
        dp_producer_state: pipeline.PipelineState,
        issue_seq: Int32,
    ):
        """v5.1b head-outer score pass t: 8 window reads x (S_t, dP_t).

        A operand (item 5): the RESIDENT K image -- fragment stage ==
        D-piece (8 stages); no per-piece gate (the bundle-level
        pipe_kres wait/release lives in the leader, K(i+1)'s gather
        gate = this pass sequence's last completion edge).
        B operand (item 6): the Q/dO STRIP double buffer -- fragment
        stage index = (t%2)*8 + piece (strip stage t%2 holds THIS
        pass's h16 window, filled by W17 under pipe_strip).
          strip consumer_wait  <- edge: W17's strip-pair(t) commit
                                  (TMA landed + fenced);
          strip consumer_release-> edge: W17's strip acquire for pass
                                  t+2 (tcgen05-tracked: fires when
                                  this pass's B reads complete).
        Done-pipeline cadence unchanged from v5: one (s, dp) stage
        pair (stage == t%2) acquired up front, committed after the
        pass, 4 commits/bundle; S(t+1) issuable while math T2Rs S(t).
        ACCUMULATE=False on the pass's first k-block.  RETURNS the
        advanced (strip, s, dp) states (lesson #14 return-and-
        reassign across the @cute.jit boundary).
        """

        s_done_pipeline.producer_acquire(s_producer_state)
        dp_done_pipeline.producer_acquire(dp_producer_state)
        # edge: W17 strip-pair(t) TMA complete (stage t % 2).
        strip_pipeline.consumer_wait(strip_consumer_state)
        # Canonical-name contract (harness trace-prepare requires
        # S_ISSUE/dP_ISSUE): payload = bundle * 4 + t (spec Z8).  The
        # window-read pass interleaves both planes, so S_ISSUE covers
        # pass start .. the last G1 atom and dP_ISSUE the residual dP
        # tail; the two tile the pass window exactly.
        pass_payload = issue_seq * Int32(self.SUB_TILES) + Int32(
            sub_tile
        )
        score_issue_token = _iket.range_start(
            "S_ISSUE(i)",
            pass_payload,
        )
        s_mma = score_tiled_mma.with_()
        dp_mma = dp_tiled_mma.with_()
        k_blocks = cute.size(k_fragment, mode=[2])
        for piece in cutlass.range_constexpr(self.SCORE_D_PIECES):
            if cutlass.const_expr(piece == 0):
                s_mma.set(tcgen05.Field.ACCUMULATE, False)
                dp_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range_constexpr(k_blocks):
                # G1(t): S^T sub-tile accumulator (A = resident K
                # piece window; B = strip window).
                cute.gemm(
                    s_mma,
                    t_s,
                    k_fragment[None, None, k_block, piece],
                    q_fragment[
                        None,
                        None,
                        k_block,
                        (sub_tile % 2) * 8 + piece,
                    ],
                    t_s,
                )
                if cutlass.const_expr(
                    piece == self.SCORE_D_PIECES - 1
                    and k_block == k_blocks - 1
                ):
                    _iket.range_end(
                        score_issue_token,
                        pass_payload,
                    )
                    score_issue_token = _iket.range_start(
                        "dP_ISSUE(i)",
                        pass_payload,
                    )
                # G2(t): dP^T sub-tile accumulator (V == K: the SAME
                # resident piece window is the A operand).
                cute.gemm(
                    dp_mma,
                    t_dp,
                    dp_k_fragment[None, None, k_block, piece],
                    do_fragment[
                        None,
                        None,
                        k_block,
                        (sub_tile % 2) * 8 + piece,
                    ],
                    t_dp,
                )
                s_mma.set(tcgen05.Field.ACCUMULATE, True)
                dp_mma.set(tcgen05.Field.ACCUMULATE, True)
        _iket.range_end(
            score_issue_token,
            pass_payload,
        )
        cute.arch.fence_view_async_tmem_store()
        # Strip stage release: W17 may refill stage t%2 for pass t+2
        # once this pass's B reads complete (UMMA-tracked).
        strip_pipeline.consumer_release(strip_consumer_state)
        strip_consumer_state.advance()
        # Sub-tile stage commits (stage == t % 2, one pair per pass).
        s_done_pipeline.producer_commit(s_producer_state)
        s_producer_state.advance()
        dp_done_pipeline.producer_commit(dp_producer_state)
        dp_producer_state.advance()
        return (
            strip_consumer_state,
            s_producer_state,
            dp_producer_state,
        )

    @cute.jit
    def _issue_dkv_block_pair_v51(
        self,
        pair: cutlass.Constexpr[int],
        dkv_tiled_mma: cute.TiledMma,
        t_acc: cute.Tensor,
        slab_fragment: cute.Tensor,
        gen_fragment: cute.Tensor,
    ):
        """v5.1 G3/G4: one (pair, tensor, D-round) block, K = h64.

        Pair P = {t=2P, t=2P+1}: the two sub-tiles' h16 boxes tile
        chunk P's FULL h64 interval, so A = the P or dS slab chunk
        image P with ALL FOUR K16 column blocks consumed, and B = one
        full-wide [h64 x own-D64] gen (single stage; gen k16-block kb
        carries exactly the heads of slab column block kb).  The two
        accumulate chains issue back-to-back into the SAME TMEM slot
        (spec v5.1 item 2):
          chain t-even (j=0): kb {0, 2}, ACCUMULATE=False on the
                              first atom (fresh block);
          chain t-odd  (j=1): kb {1, 3}, ACCUMULATE=True (chained).
        f32 accumulation-order change is covered by the standing
        "reordering is legal" ruling of the v5 demo spec.  Every
        (pair, tensor, round) block is drained before its TMEM slot
        is reused.
        """

        assert cute.size(gen_fragment, mode=[2]) == 4, str(
            gen_fragment.shape
        )
        assert cute.size(gen_fragment, mode=[3]) == 1, str(
            gen_fragment.shape
        )
        mma = dkv_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in (0, 2, 1, 3):
            cute.gemm(
                mma,
                t_acc,
                slab_fragment[None, None, kb, pair],
                gen_fragment[None, None, kb, 0],
                t_acc,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def _issue_grads_pair_v51(
        self,
        pair: cutlass.Constexpr[int],
        dkv_tiled_mma: cute.TiledMma,
        t_dkv_s0: cute.Tensor,
        t_dkv_s1: cute.Tensor,
        t_dkv_s2: cute.Tensor,
        t_dkv_s3: cute.Tensor,
        p_fragment: cute.Tensor,
        ds_fragment: cute.Tensor,
        gen_fragment_a: cute.Tensor,
        gen_fragment_b: cute.Tensor,
        pds_pipeline,
        pds_consumer_state: pipeline.PipelineState,
        dkv_done_pipeline,
        dkv_producer_state: pipeline.PipelineState,
        round_pipeline,
        round_consumer_state: pipeline.PipelineState,
        loop_iter: Int32,
    ):
        """v5.1 grads(pair): 4 D-rounds x (dV, dK), pair K = h64.

        pds handoff (v5.1 item 1, minimal change on the 2-stage math-
        produced pipeline): BOTH sub-tile stages of the pair are
        waited UP FRONT (every block reads both sub-tiles' slab
        columns) and released together after the pair's 8 blocks --
        the v32 score-helper clone+advance double-stage precedent:
          consumer_wait(stage 0 gen) <- edge: math(2P) pds commit;
          consumer_wait(stage 1 gen) <- edge: math(2P+1) pds commit;
          2x consumer_release        -> edge: math(2P+2)/math(2P+3)
                                       pds acquires (UMMA-tracked:
                                       fire when the pair's G3/G4
                                       descriptor reads complete).
        Per D-round r, dV then dK; each block:
          dkv acquire  <- edge: reducer FUSED drain of the block FOUR
                          generations back ([fix-r7] slot army: 4-deep
                          dkv_done ring; slot == stage == (2r + p) % 4
                          is a COMPILE-TIME constant since 16 blocks/
                          bundle mod 4 == 0 -- static tuple selection,
                          no runtime TMEM arithmetic);
          round wait   <- edge: W17 full-wide gen (pair*4 + r)*2 + p
                          landed (FIFO index);
          round release-> edge: W17's producer_acquire two gens ahead
                          (UMMA-tracked);
          dkv commit   -> edge: reducer WAIT_dK for this block (the
                          reducer consumes generations in (dV, dK)
                          pairs -- fused drain).
        dVdK_ISSUE payload = b*16 + pair*8 + r*2 + p (v5.1 item 4).
        RETURNS the advanced (pds, dkv, round) states (lesson #14).
        """

        pds_pipeline.consumer_wait(pds_consumer_state)
        pds_state_odd = pds_consumer_state.clone()
        pds_state_odd.advance()
        pds_pipeline.consumer_wait(pds_state_odd)
        for d_round in cutlass.range_constexpr(
            self.DKV_D_ROUNDS
        ):
            block_payload = loop_iter * Int32(
                2 * self.DKV_D_ROUNDS * 2
            ) + Int32(
                (pair * self.DKV_D_ROUNDS + d_round) * 2
            )
            # [fix-r7] static slot selection: (2r + p) % 4, a
            # Python int under the range_constexpr round loop.
            slot_tensors = (
                t_dkv_s0,
                t_dkv_s1,
                t_dkv_s2,
                t_dkv_s3,
            )
            t_slot_dv = slot_tensors[(2 * d_round) % 4]
            dkv_done_pipeline.producer_acquire(
                dkv_producer_state
            )
            round_pipeline.consumer_wait(round_consumer_state)
            dkv_issue_token = _iket.range_start(
                "dVdK_ISSUE(i,r,p)",
                block_payload,
            )
            self._issue_dkv_block_pair_v51(
                pair,
                dkv_tiled_mma,
                t_slot_dv,
                p_fragment,
                gen_fragment_a,
            )
            _iket.range_end(
                dkv_issue_token,
                block_payload,
            )
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
            dkv_done_pipeline.producer_commit(dkv_producer_state)
            dkv_producer_state.advance()
            t_slot_dk = slot_tensors[(2 * d_round + 1) % 4]
            dkv_done_pipeline.producer_acquire(
                dkv_producer_state
            )
            round_pipeline.consumer_wait(round_consumer_state)
            dkv_issue_token = _iket.range_start(
                "dVdK_ISSUE(i,r,p)",
                block_payload + Int32(1),
            )
            self._issue_dkv_block_pair_v51(
                pair,
                dkv_tiled_mma,
                t_slot_dk,
                ds_fragment,
                gen_fragment_b,
            )
            _iket.range_end(
                dkv_issue_token,
                block_payload + Int32(1),
            )
            round_pipeline.consumer_release(round_consumer_state)
            round_consumer_state.advance()
            dkv_done_pipeline.producer_commit(dkv_producer_state)
            dkv_producer_state.advance()
        # Pair slab-stage releases: math's publishes of sub-tiles
        # 2P+2 / 2P+3 may proceed once the pair's last G4 completes
        # (both releases are tcgen05 group commits tracking all
        # previously issued MMAs, so each covers the whole pair).
        pds_pipeline.consumer_release(pds_consumer_state)
        pds_consumer_state.advance()
        pds_pipeline.consumer_release(pds_consumer_state)
        pds_consumer_state.advance()
        return (
            pds_consumer_state,
            dkv_producer_state,
            round_consumer_state,
        )

    @cute.jit
    def _issue_dq_block_v52(
        self,
        t_window: cutlass.Constexpr[int],
        d_half: cutlass.Constexpr[int],
        dq_tiled_mma: cute.TiledMma,
        t_dq_slot: cute.Tensor,
        kd_fragment_w0: cute.Tensor,
        kd_fragment_w1: cute.Tensor,
        ds_b_fragment: cute.Tensor,
    ):
        """v5.2 G5: one (t, r) dQ eviction block, K chained over kv128.

        (M,N,K) = (D128, h32, kv64) x two kv waves.  A = the kdq gen
        d-half window (stage == d_half; wave w's gen lives in round
        buf w).  B = the hand-derived dq_b window view (flat stage
        index t + 4*w).  The rotating TMEM slot starts FRESH
        (ACCUMULATE=False on the first atom): partial sums accumulate
        in the f32 dQ workspace via the reducer's RMW offload, never
        across bundles in TMEM.  Output column n of this block is
        physical head (n//16)*64 + t*16 + (n%16); the D rows are the
        A operand's strips D[r_old*256 + rank*128 + d_half*64, +64)
        ([fix-r8]: the legacy gen M-half interposes rank between
        r_old and d_half -- see the offloader's decode contract).
        """

        mma = dq_tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks = cute.size(kd_fragment_w0, mode=[2])
        for k_block in cutlass.range_constexpr(k_blocks):
            cute.gemm(
                mma,
                t_dq_slot,
                kd_fragment_w0[None, None, k_block, d_half],
                ds_b_fragment[None, None, k_block, t_window],
                t_dq_slot,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)
        for k_block in cutlass.range_constexpr(k_blocks):
            cute.gemm(
                mma,
                t_dq_slot,
                kd_fragment_w1[None, None, k_block, d_half],
                ds_b_fragment[None, None, k_block, t_window + 4],
                t_dq_slot,
            )
            mma.set(tcgen05.Field.ACCUMULATE, True)

    # convert_canonical: NOT overridden in v3.2.  The v6/v8 scramble
    # decode belonged to the thread-indexed drain addressing; the v3.2
    # drain stores through coordinate-exact identity decoding (natural
    # order), which pairs with the BASE class's canonical per-element
    # copy (audit F6: keeping the v6 decode here permuted every
    # 32-column group of the output).

    @cute.jit
    def _drain_dkv_fused_v52(
        self,
        t_slot_dv: cute.Tensor,
        t_slot_dk: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        tile_index: Int32,
        d_round: Int32,
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
        """v5.2: FUSED drain of one (dV, dK) pair -- ONE red.global
        stream (the T3-HO4 P0 gate measurement).

        Physical basis: V == K, so dV and dK of the same (pair, r)
        target the SAME dKV accumulator rows; summing the pair in
        registers before the atomics halves the red.global traffic
        (expected ~1.2-1.4 us/pair vs the old 2 x ~2 us).  Both
        generations of the pair are waited up front (dV then dK,
        consecutive stages of the 4-deep ring; [fix-r7] the slot
        tensors arrive as STATIC call-site selections -- slot ==
        (2r + p) % 4, pair-independent), T2R'd
        back-to-back under one fence, released together, then the
        summed fragment runs the v32 decode verbatim (per-pair kv row
        + f32x2 atomics).  WAIT_dK/REDUCE_T2R/REDUCE_ATOMIC payloads
        = the fused-pair issue_seq (b*8 + pair*4 + r).
        """

        wait_dk_token = _iket.range_start(
            "WAIT_dK(i,r)",
            issue_seq,
        )
        # dV generation, then the pair's second consecutive stage
        # (dK).  [fix-r7]: the slot tensors are static call-site
        # selections; only the pipeline generations are waited here.
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        _iket.range_end(
            wait_dk_token,
            issue_seq,
        )

        reduce_t2r_token = _iket.range_start(
            "REDUCE_T2R(i,r)",
            issue_seq,
        )
        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        t_core_dv = t_slot_dv[(None, None), 0, 0]
        t_core_dk = t_slot_dk[(None, None), 0, 0]
        # The M64-interleaved UMMA_2SM fragment core is
        # (m64,(n64,h2)) with TMEM-ENCODED strides (lane stride 2^16):
        # lane = m + 64*h, column = n.  make_tmem_copy needs the
        # physical (DP, col) congruence, so regroup to ((m64,h2),n64)
        # -- a pure mode permutation; the encoded strides ride along
        # verbatim (v32 drain precedent).
        assert t_core_dv.shape == (
            self.N_TILE,
            (self.N_TILE, 2),
        ), str(t_core_dv.layout)
        core_stride = t_core_dv.layout.stride
        phys_layout = cute.make_layout(
            ((self.N_TILE, 2), self.N_TILE),
            stride=(
                (core_stride[0], core_stride[1][1]),
                core_stride[1][0],
            ),
        )
        t_phys_dv = cute.make_tensor(
            t_core_dv.iterator,
            phys_layout,
        )
        t_phys_dk = cute.make_tensor(
            t_core_dk.iterator,
            phys_layout,
        )
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(
            tmem_load_atom,
            t_phys_dv,
        )
        thread_t2r = tiled_t2r.get_slice(dp_idx)
        c_dkv = cute.make_identity_tensor(
            ((self.N_TILE, 2), self.N_TILE)
        )
        thread_coordinates = self.split_wg(
            thread_t2r.partition_D(c_dkv),
            2,
            wg_idx,
        )
        thread_source_dv = self.split_wg(
            thread_t2r.partition_S(t_phys_dv),
            2,
            wg_idx,
        )
        thread_source_dk = self.split_wg(
            thread_t2r.partition_S(t_phys_dk),
            2,
            wg_idx,
        )
        thread_values_dv = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )
        thread_values_dk = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )

        cute.copy(tiled_t2r, thread_source_dv, thread_values_dv)
        cute.copy(tiled_t2r, thread_source_dk, thread_values_dk)
        cute.arch.fence_view_async_tmem_load()
        # Both slots free together (two consecutive releases).
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        _iket.range_end(
            reduce_t2r_token,
            issue_seq,
        )

        assert cute.size(thread_values_dv) == self.N_TILE // 2
        # Register fusion: d(KV) = dV + dK (same destination rows).
        for i in cutlass.range_constexpr(self.N_TILE // 2):
            thread_values_dv[i] = (
                thread_values_dv[i] + thread_values_dk[i]
            )

        reduce_atomic_token = _iket.range_start(
            "REDUCE_ATOMIC(i,r)",
            issue_seq,
        )
        # Per-PAIR gather row: the Rep-4 fragment spans multiple fold
        # rows per thread (v8 precedent: four dp rows), so the kv row
        # and its topk index are decoded from THIS pair's coordinate
        # (audit F5: a single hoisted row misroutes 3/4 of the mass).
        # A one-entry cache elides the redundant index reloads within
        # a row run.
        cached_kv_row = Int32(-1)
        kv_index = Int32(-1)
        for i in cutlass.range_constexpr(self.N_TILE // 4):
            v0 = 2 * i
            v1 = v0 + 1
            rdkv_frg = cute.make_rmem_tensor(
                (2,),
                self.acc_dtype,
            )
            rdkv_frg[0] = thread_values_dv[v0]
            rdkv_frg[1] = thread_values_dv[v1]
            pair_kv = Int32(
                cute.get(
                    thread_coordinates[v0],
                    mode=[0, 0],
                )
            )
            if pair_kv != cached_kv_row:
                cached_kv_row = pair_kv
                pair_global_row = (
                    tile_index * Int32(2 * self.N_TILE)
                    + rank * Int32(2 * self.N_TILE_CTA)
                    + pair_kv
                )
                kv_index = Int32(-1)
                if pair_global_row < topk:
                    kv_index = mTopkIdxs[
                        pair_global_row,
                        (token_idx, batch_idx),
                    ]
            if kv_index >= Int32(0):
                # D within the block = 64 * h(fold half, mode [0,1])
                # + n (mode [1]); the block's GMEM quadrant is d_round.
                d_local = Int32(
                    cute.get(
                        thread_coordinates[v0],
                        mode=[1],
                    )
                ) + Int32(
                    cute.get(
                        thread_coordinates[v0],
                        mode=[0, 1],
                    )
                ) * Int32(2 * self.N_TILE_CTA)
                dkv_row = mdKV_acc[
                    None,
                    kv_index,
                    (0, batch_idx),
                ]
                tile_row = cute.flat_divide(dkv_row, (128,))
                quad_row = tile_row[None, d_round]
                pair_row = cute.flat_divide(quad_row, (2,))
                target_frg = pair_row[None, d_local // Int32(2)]
                cute.arch.atomic_add(
                    target_frg.iterator.llvm_ptr,
                    rdkv_frg.load(),
                )
        _iket.range_end(
            reduce_atomic_token,
            issue_seq,
        )
        return wait_state, release_state

    @cute.jit
    def _offload_dq_block_v52(
        self,
        t_window: cutlass.Constexpr[int],
        r_index: cutlass.Constexpr[int],
        t_dq_slot: cute.Tensor,
        mdQ_acc: cute.Tensor,
        mdQ: cute.Tensor,
        token_idx: Int32,
        batch_idx: Int32,
        rtx: Int32,
        rank: Int32,
        is_first: cutlass.Boolean,
        is_last: cutlass.Boolean,
        issue_seq: Int32,
        evict_pipeline,
        wait_state: pipeline.PipelineState,
        release_state: pipeline.PipelineState,
    ):
        """v5.2: offload ONE (t, r) dQ eviction block.

        The cluster owns its token's dQ rows exclusively, and each
        workspace address is touched by exactly ONE thread per bundle
        (static lane ownership, bundle-invariant decode), so ordering
        is same-thread per-location program order throughout:
          first bundle: STG only (initializes the buffer; also makes
            the kernel stateless across runs);
          last bundle:  LDG + FADD + single bf16 cast into mdQ (the
            terminal value -- f32 accumulation + ONE final rounding,
            the same numeric class as the retired TMEM epilogue);
          middle:       red.global.add.f32 ([v5.3-L1] one-way traffic:
            the r10 ledger convicted the retired LDG+FADD+STG chain's
            load round-trip of overloading the reduce lanes -- drain
            service 3.97us/pair vs 1.9us/pair demand -- which paced
            the in-tile grads gaps AND the 36us end-of-bundle K-
            turnover stall.  Same-thread same-address coherence keeps
            the f32 accumulation order bit-identical, and the last
            bundle's LDG observes every prior red by the same
            per-location guarantee -- precision iron rule intact).
        Decode contract ([fix-r8] corrected): the block's D coverage
        is dictated by the A operand -- the kdq gen holds the LEGACY
        M-half slice D[r_old*256 + rank*128, +128), so its d_half
        window is D[r_old*256 + rank*128 + d_half*64, +64): two
        64-strips 128 apart per cluster block, NOT a contiguous
        r*128 slab.  Value at fold coordinate ((m, n_hi), n_low) is
          d = (r//2)*256 + rank*128 + (r%2)*64 + m,
          head = n_hi*64 + t*16 + n_low
        (r = 2*r_old + d_half; the eight (r_old, rank, d_half)
        strips tile D512 bijectively).
        DQ_EPI(r) is the offload span now (payload b*16 + t*4 + r --
        semantic change logged in V5_BUILD_LOG).
        """

        dq_epi_token = _iket.range_start(
            "DQ_EPI(r)",
            issue_seq,
        )
        # edge: the leader's dq_evict commit (tcgen05-tracked block
        # completion).
        evict_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        dp_idx = rtx % Int32(self.MATH_THREADS_PER_CTA)
        wg_idx = rtx // Int32(self.MATH_THREADS_PER_CTA)
        t_core = t_dq_slot[(None, None), 0, 0]
        # (128,32) CG2 fold core: (m64,(n16,2)) TMEM-encoded; regroup
        # to ((m64,2),n16) exactly like the dkv drain.
        assert t_core.shape == (
            self.N_TILE,
            (self.SUB_TILE_BOX, 2),
        ), str(t_core.layout)
        core_stride = t_core.layout.stride
        t_phys = cute.make_tensor(
            t_core.iterator,
            cute.make_layout(
                ((self.N_TILE, 2), self.SUB_TILE_BOX),
                stride=(
                    (core_stride[0], core_stride[1][1]),
                    core_stride[1][0],
                ),
            ),
        )
        # [fix-r5] Repetition(1), NOT (2): split_wg carves its input
        # along mode [2] -- the COLUMN-iteration mode of the
        # partitioned tensor -- by the warpgroup count (v8-era drain
        # shape contract, hardware r5 echo).  A Rep(2) atom covers all
        # 16 columns in one op (rest_col = 1, 1 // 2 == 0 -> the r5
        # ValueError); Rep(1) covers 8 columns per op (rest_col = 2),
        # so the two warpgroups split the block 8/8 column-wise
        # exactly like the fused drain's Rep(4)-on-64 shape, and each
        # warp still reads only its own 32-DP physical window.
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(1)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(
            tmem_load_atom,
            t_phys,
        )
        thread_t2r = tiled_t2r.get_slice(dp_idx)
        c_dq = cute.make_identity_tensor(
            ((self.N_TILE, 2), self.SUB_TILE_BOX)
        )
        thread_coordinates = self.split_wg(
            thread_t2r.partition_D(c_dq),
            2,
            wg_idx,
        )
        thread_source = self.split_wg(
            thread_t2r.partition_S(t_phys),
            2,
            wg_idx,
        )
        thread_values = cute.make_rmem_tensor(
            thread_coordinates.shape,
            self.acc_dtype,
        )
        cute.copy(tiled_t2r, thread_source, thread_values)
        cute.arch.fence_view_async_tmem_load()
        # edge -> the leader's dq_evict acquire two blocks ahead.
        evict_pipeline.consumer_release(release_state)
        release_state.advance()

        # 2,048 values / 256 threads = 8 per thread (echo on drift).
        assert cute.size(thread_values) == 8, str(
            thread_coordinates.shape
        )
        # Four RMW variants hoisted around the unrolled value loop
        # (dynamic branches around static loops -- gather precedent).
        if is_first:
            if is_last:
                for i in cutlass.range_constexpr(8):
                    d_g = Int32(
                        (r_index // 2) * 256
                        + (r_index % 2) * 64
                    ) + rank * Int32(128) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 0],
                        )
                    )
                    head = Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 1],
                        )
                    ) * Int32(64) + Int32(
                        t_window * 16
                    ) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[1],
                        )
                    )
                    mdQ[
                        d_g,
                        head,
                        (token_idx, batch_idx),
                    ] = self.element_dtype(thread_values[i])
            else:
                for i in cutlass.range_constexpr(8):
                    d_g = Int32(
                        (r_index // 2) * 256
                        + (r_index % 2) * 64
                    ) + rank * Int32(128) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 0],
                        )
                    )
                    head = Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 1],
                        )
                    ) * Int32(64) + Int32(
                        t_window * 16
                    ) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[1],
                        )
                    )
                    mdQ_acc[
                        head,
                        d_g,
                        (token_idx, batch_idx),
                    ] = thread_values[i]
        else:
            if is_last:
                for i in cutlass.range_constexpr(8):
                    d_g = Int32(
                        (r_index // 2) * 256
                        + (r_index % 2) * 64
                    ) + rank * Int32(128) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 0],
                        )
                    )
                    head = Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 1],
                        )
                    ) * Int32(64) + Int32(
                        t_window * 16
                    ) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[1],
                        )
                    )
                    mdQ[
                        d_g,
                        head,
                        (token_idx, batch_idx),
                    ] = self.element_dtype(
                        mdQ_acc[
                            head,
                            d_g,
                            (token_idx, batch_idx),
                        ]
                        + thread_values[i]
                    )
            else:
                for i in cutlass.range_constexpr(8):
                    d_g = Int32(
                        (r_index // 2) * 256
                        + (r_index % 2) * 64
                    ) + rank * Int32(128) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 0],
                        )
                    )
                    head = Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[0, 1],
                        )
                    ) * Int32(64) + Int32(
                        t_window * 16
                    ) + Int32(
                        cute.get(
                            thread_coordinates[i],
                            mode=[1],
                        )
                    )
                    # [v5.3-L1] fire-and-forget red.global.add.f32
                    # (idiom: the fused dkv drain's atomic_add call).
                    # batch stride is 0 in the carve -- omitted.
                    destination_ptr = (
                        mdQ_acc.iterator
                        + head * mdQ_acc.stride[0]
                        + d_g * mdQ_acc.stride[1]
                        + token_idx * mdQ_acc.stride[2][0]
                    )
                    cute.arch.atomic_add(
                        destination_ptr.llvm_ptr,
                        thread_values[i],
                    )
        _iket.range_end(
            dq_epi_token,
            issue_seq,
        )
        return wait_state, release_state

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
        wait_dk_token = _iket.range_start(
            "WAIT_dK(i,r)",
            packed_issue,
        )
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        _iket.range_end(
            wait_dk_token,
            packed_issue,
        )

        reduce_t2r_token = _iket.range_start(
            "REDUCE_T2R(i,r)",
            packed_issue,
        )
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
        _iket.range_end(
            reduce_t2r_token,
            packed_issue,
        )

        # v11: slot 0's atomics run HERE, before the tail-commit wait --
        # restoring the v6 per-slot sequencing.  This halves the T2R
        # value liveness (32 registers instead of 64 held across the
        # slot-1 wait: the v9_3 SASS showed exactly that 64-register
        # window spilling to local memory inside the REDG loop), and
        # overlaps slot 0's atomic burst with the leader's grads tail.
        assert cute.size(thread_values_0) == self.N_TILE // 2
        reduce_atomic_token = _iket.range_start(
            "REDUCE_ATOMIC(i,r)",
            packed_issue,
        )
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
        _iket.range_end(
            reduce_atomic_token,
            packed_issue,
        )

        # --- slot 1: tail-committed generation.
        wait_dk_token_1 = _iket.range_start(
            "WAIT_dK(i,r)",
            packed_issue + Int32(1),
        )
        done_pipeline.consumer_wait(wait_state)
        wait_state.advance()
        _iket.range_end(
            wait_dk_token_1,
            packed_issue + Int32(1),
        )
        reduce_t2r_token_1 = _iket.range_start(
            "REDUCE_T2R(i,r)",
            packed_issue + Int32(1),
        )
        cute.copy(tiled_t2r_1, thread_source_1, thread_values_1)
        cute.arch.fence_view_async_tmem_load()
        done_pipeline.consumer_release(release_state)
        release_state.advance()
        _iket.range_end(
            reduce_t2r_token_1,
            packed_issue + Int32(1),
        )

        reduce_atomic_token_1 = _iket.range_start(
            "REDUCE_ATOMIC(i,r)",
            packed_issue + Int32(1),
        )
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
        _iket.range_end(
            reduce_atomic_token_1,
            packed_issue + Int32(1),
        )
        return wait_state, release_state


# ======================================================================
# V32 SELF-AUDIT (build trailer, v3.2 "T3-64" score-transposed form)
# ======================================================================
#
# SMEM account as implemented (per CTA, cap 232,448; the storage struct
# asserts == 231,424 at build time):
#   stationary Q panel (2 x [h32 x D512] stages)        65,536
#   stationary dO panel                                 65,536
#   score K/V chase ring (2 x [own-kv64 x D64])         16,384
#   round region (2 x 16,384, 12 gens/bundle)           32,768
#   P slab (2 x [own-kv64 x h64] chunk-major)           16,384
#   dS slab (same; sub[1-rank] = DSM payload)           16,384
#   dq_b dual sub-image (2 x [kv64 x own-H64])          16,384
#   softmax stats (lse[128] + delta[128], f32)           1,024
#   mbarriers / holding buf (padded)                  <= 1,024
#   total                                              231,424  (slack 1,024)
#
# TMEM 512-column map (all f32, per CTA), tmem.allocate UNGUARDED:
#   dQ^T  [0,256)   persistent, 128 cols per D-round x 2 (M256 CG2:
#                   128 lanes x full N=128)
#   S pp  [256,320) 2 stages x 32 cols (M128 CG2 fold 128DP x N/2)
#   dP pp [320,384) 2 stages x 32 cols
#   dV    [384,448) one [kv128 x D128] block (128DP x 64 cols)
#   dK    [448,512) same
#
# Generation law: ROUND_GENS_PER_TILE = 12 (mod 2 == 0, phase law holds)
#   g0/g1 kdq r0 waves | g2..g9 four (dO,Q) rounds | g10/g11 kdq r1.
# Chase: 8 pieces/bundle through a 2-slot ring; slot released only after
#   its FOUR score consumers issue (kill-list 3 release edge in-graph);
#   bundle t+1's pieces stream while the leader runs grads(t) (the
#   G3(c1,0)(t) pin point is realized by the ring credits freed in the
#   score phase).
#
# Addendum errata coverage:
#   #1 (free edge covers BOTH D-round consumers): mb_dqb_free is the
#      1-stage pipe_dqb_free; the leader's producer_commit is issued
#      AFTER G5(r1,w1) and the tcgen05 group commit tracks every
#      previously issued MMA, so the free edge covers r0 AND r1 of both
#      waves.  Math waits it once per bundle BEFORE any dq_b write; the
#      relay's peer push is transitively gated via pds_ready.
#   #2 (asymmetric arrival): each CTA's relay locally waits its own
#      landing mbar (completion-tx lives at the destination) and then
#      remote-arrives the leading CTA's mb_dqb[1-rank]; own-image
#      readiness arrives mb_dqb[rank] after the count-128 pds_ready.
#      Both gates are count-2 (one arrival per CTA), leader waits with
#      _mbarrier_wait_acquire_cluster at parity loop_iter & 1.
#   #3 (forward-only waits): math waits {S/dP done, pds empty, dqb
#      free(t-1 edge)}; leader waits {chase full, round full, mb_dqb(t),
#      pds full(t)}; relay waits {pds_ready(t), landing(t)}; gather/W17
#      pace on ring credits -- all edges point forward.
#   #4 (one-sided skew absorption): wave order is static kv0 -> kv1;
#      priced, not symmetric.
#
# V32-TODO registry (grep "V32-TODO"):
#   * st.async register push upgrade for the dq_b peer half (bulk-DSM
#     fallback implemented; addendum's primary form pending SASS gate).
#   * audit: stationary panel byte-identity with the staged score-B
#     layout (SW128B K-major atom equality; asserted only by cosize).
#   * audit: ROW_MAJOR publish image swizzle atom == dkv-A operand atom;
#     get_smem_store_op may select the TRANSPOSED stmatrix variant --
#     the isinstance build gate must be widened if trace-prepare
#     rejects it (v8 STS.U16 precedent makes this a mandatory gate).
#   * audit: drain fragment decode (kv = dp % 64, D = 64*(dp//64) + n)
#     and the {n, n+1} pair adjacency for the f32x2 atomics.
#   * perf: column-axis grouped-stat hoisting (v9.3 analog) disabled;
#     derive the Rep4 column-group structure if one exists.
#   * perf: math waits the whole-bundle dq_b free gate (per-wave
#     granularity was traded for the pipeline encoding of errata #1).
#
# Assumption ledger (load-bearing, derived not measured):
#   * CG2 operand split: A M-half per CTA, B N-half per CTA (hardware
#     exchanges B), C = M-half x FULL N per CTA -- this closes the TMEM
#     map (256+64+64+64+64 = 512) and the 12-gen byte account exactly.
#   * V == K single fetch: the chase piece feeds G1 and G2; dV and dK
#     merge into the same GMEM accumulator row via f32 atomics.
#   * Natural head chunking (chunk c = H[c*64:(c+1)*64)): panel = two
#     32-row boxes; round-gen chunk windows single-box; no permutation.
#   * v17a dQ accumulator was already [D x H]; the epilogue SMEM
#     scatter performs the transpose, so DQ_EPI_TRANSPOSED stays False.
# ======================================================================
# ======================================================================
# V5 TILING4 SELF-AUDIT ADDENDUM (supersedes the v32 trailer's TMEM map,
# generation law, chase count and pds paragraphs; everything it does not
# mention is inherited verbatim -- SMEM byte account unchanged except
# pds_mbars 2 -> 4 Int64, absorbed by the 1,024 B header pad).
# ======================================================================
#
# Sub-tile geometry (V5_TILING4_DEMO_SPEC Z1): SUB_TILES = 4 head-outer
# h32 sub-tiles, t = (c, j), head(t, n) = c*64 + (n//16)*32 + j*16 +
# (n%16) -- forced by the frozen panel residency + CG2 B N-half split.
# Every byte container (panel, P/dS slab chunk images, dq_b, relay
# payload) keeps NATURAL head order; sub-tile publishes/reads address
# two h16 boxes inside them (J-mode store slices / K16 descriptor
# blocks {j, j+2} / score-B stage index 2p+j).
#
# TMEM map: dQ^T [0,256) | S pp [256,288) 2x16 | dP pp [288,320) 2x16
# | dV [320,384) | dK [384,448) | [448,512) FREE.  448/512 used.
#
# Generation law: ROUND_GENS_PER_TILE = 36 (mod 2 == 0 phase law):
# g0..g31 half-wide (t, r) x (dO, Q) grad gens (4 KB: two h16 boxes),
# g32/g33 kdq r0, g34/g35 kdq r1 -- kdq moved BEHIND the grad gens to
# keep the single round ring FIFO-consistent with the bundle-tail G5.
# Chase: 32 pieces/bundle (4 passes x 8 D-slices, same kv rows per
# pass, L2-hot re-gather); ring depth, credits, kdq rendezvous
# positions unchanged.
#
# Sub-tile pipeline cadences (per bundle): s_done/dp_done 4 commits
# (stage = t%2); pds 2-stage MATH-produced, 4 acquire/commit vs 4
# leader wait/release (grads(t)); dkv_done 32 generations; drains 32.
# Leader total order: score(0); {score(t); grads(t-1)} t=1..3;
# grads(3); G5 r0; G5 r1; dqb-free commit; TAIL after the bundle loop.
#
# Steady-state wait graph (one bundle b, edges point at the waiter):
#   score(t)   <- kscore ring credits (gather piece t*8+p)
#              <- s/dp empty (math T2R of sub-tile t-2)
#   math(t)    <- s/dp full (score(t) UMMA commit)
#              <- pds empty (leader grads(t-2) UMMA release)
#              <- dqb_free (t==0 only; leader G5(b-1) group commit)
#   grads(t)   <- pds full (math(t) commit)
#              <- round full (W17 gen (t*4+r)*2+p)
#              <- dkv empty (reducer drain of the tensor's prev block)
#   G5 r0/r1   <- round full (kdq g32..g35)
#              <- mb_dqb[w] (relay: own stored + peer landed)
#   relay push <- pds_ready (math, bundle-level, after t=3)
#   W17 gen q  <- round empty (leader consumed gen q-2)
#   gather p   <- kscore empty (leader released piece p-2)
# DSM-source WAR (math(b+1) overwriting the dS peer chunk while
# push(b) reads it) closes transitively: mb_dqb(b) full => both pushes
# landed => G5(b) issues => leader reaches score(0)(b+1) => math(b+1)
# T2R gate.  No backward edge; the four-role kdq cycle of the v32
# audit is broken at the same point it was in rev5 (r1(prev) rendezvous
# before the piece-2 ring block).
#
# Trace-readout contract (spec Z8): S_ISSUE/dP_ISSUE payload = b*4+t;
# WAIT_S/T2R_S/WAIT_dP/T2R_dP/MATH_PD payload = 8b+2t (P/dS phases
# +0/+1); MATH_PDS_ACQ = b*4+t; MAT_QDO = b*4+t (span t carries the
# pipelined tail wait of span t-1's last gen); dVdK_ISSUE and the
# reducer spans = ((b*4+t)*4+r)*2+p; ROUTE_K/LOAD_K/MATH_BAR1/ROUTE_dS
# payloads unchanged.  Static names: 27 spans + provenance = 28 (cap
# 28, net +0 vs the v32 rev5 base).
#
# Overlap verdict hooks (spec judgment contract): steady state should
# show S_ISSUE(4b+t+1) overlapping MATH_PD(8b+2t), dVdK_ISSUE of
# sub-tile t overlapping MATH_PD(8b+2(t+1)), and the REDUCE lanes
# continuous across t -- against the v32 rev5 serial three-phase
# baseline.
#
# V5 residual risk register (audit before first hardware run):
#   * partition_D J-mode position: asserted (mode[2] == 2, shape echo).
#     [fix-r1] the r1 gate showed partition_D consumes domain mode 0
#     entirely (an in-mode-0 J folded into the rest mode with
#     machinery-internal sub-mode order); J now sits at TOP-LEVEL
#     domain mode 1 -- mode 0 == the copy tile exactly (v32-congruent)
#     -- and passes through to output mode [2] per the passthrough law
#     confirmed by both hardware echoes.  Remaining exposure = the
#     sliced-tile vs retile shape congruence (asserted with echo).
#     [fix-r0] the upstream premise asserts were re-formed after the
#     first hardware gate: the epi outer carries degenerate size-1
#     sub-modes ((64,1):(1,0) columns), so the flat-leaf comparisons
#     became coalesce+rank/size/cosize (semantically identical pin);
#     the publish-domain width contract (32 cols/J-slice, 2 slices)
#     is now asserted explicitly.  Remaining exposure = the partition
#     placement itself.
#   * N32 CG2 MMA fold: the (64,(16,2)):(1,(128,64)) interleaved atom
#     is assumed legal per spec ("MMA N32 legal tier"); make_fragment_C
#     shape asserts (SUB_TILE_VALS) and the layout report catch a
#     mismatch at trace-prepare.
#   * score-B 16-stage byte identity: algebra in the SCORE_B_STAGES
#     note; the cosize/atom asserts hold host-side, the block-order
#     identity is adjudicated by the correctness gate exactly as v32's
#     8-stage form was.
#   * half-wide gen TMA: 2 KB boxes over h16-tile grid {4c+j, 4c+j+2};
#     expect_tx derives from grad_a_stage_bytes (2,048) -- byte count
#     asserted nowhere host-side, watch the hang signature if the TMA
#     tiler disagrees (v31 rev1 precedent).
#   * MAT_QDO straddle: span t includes the wait for span t-1's last
#     in-flight TMA (software pipelining); do not read MAT_QDO edges
#     as pure per-sub-tile supply latency.
# ======================================================================
# ======================================================================
# V5.1 PAIR-BATCHING ADDENDUM (change order 2026-08-04; supersedes the
# V5 addendum's generation law, grads cadences and dVdK/MAT_QDO payload
# paragraphs; score/math/gather/chase are UNTOUCHED from v5).
# ======================================================================
#
# Pair algebra: pair P = {t=2P, t=2P+1}; the two sub-tiles' h16 boxes
# tile chunk P's full h64 interval, so the pair's grads window reads
# ALL FOUR K16 column blocks of slab chunk image P against one
# FULL-WIDE [h64 x own-D64] gen (single TMA box; gen k16-block kb ==
# slab column block kb head-for-head).  "Two accumulate chains" =
# k-block issue order {0,2} (t-even, ACCUMULATE=False first) then
# {1,3} (t-odd, chained True) into the same TMEM slot; f32 reorder is
# covered by the standing v5 ruling.
#
# Generation law: ROUND_GENS_PER_TILE = 20 (mod 2 == 0): g0..g15
# full-wide (P, r) x (dO, Q) grad gens (8 KB each), g16/g17 kdq r0,
# g18/g19 kdq r1.  DKV_B_TILER K = 64, DKV_B_STAGES = 1.
#
# Leader total order: score(0); score(1); score(2); grads(pair0);
# score(3); grads(pair1); G5 r0; G5 r1; dqb-free commit.  grads(pair)
# waits BOTH pds stages up front (clone+advance double-wait, v32
# score-helper precedent) and releases both after its 8 blocks; math's
# per-sub-tile pds cadence (4 commits/bundle, 2-stage) is unchanged --
# math(2P+2)/math(2P+3) publish over grads(pairP) execution.
# dkv_done 16 generations/bundle; drains 16.
#
# Wait-graph deltas vs the V5 addendum (all other edges unchanged):
#   grads(pairP) <- pds full x2 (math(2P), math(2P+1))
#                <- round full (W17 gen (P*4+r)*2+p)
#                <- dkv empty (reducer, same-tensor previous block)
#   math(2P+2)   <- pds empty (grads(pairP) first release)
#   W17 gen q    <- round empty (leader consumed gen q-2; kdq r0
#                   credits now free at grads(pair1) r3)
# Liveness closes exactly as v5 (kdq rendezvous order unchanged;
# cross-bundle chain via G5 -> score(0)(b+1) intact).
#
# Trace-readout deltas: dVdK_ISSUE and the reducer spans =
# b*16 + pair*8 + r*2 + p; MAT_QDO = b*2 + pair (2 spans/bundle,
# straddle note applies).  All other payloads (S_ISSUE b*4+t, math
# spans 8b+2t, ...) unchanged.  Static names: net +0 (28/28 held).
#
# Experiment verdict hooks: grads windows 4 -> 2 per bundle (expect
# ~2 x 16 us vs 4 x 13 us), drains 32 -> 16, protocol tax per pass
# 0.5 us -> ~0.15 us target, slot-pressure cadence halved; the first
# two score passes' inter-pass gaps should hold at the 0.4 us class.
# ======================================================================
# ======================================================================
# V5.1B RESIDENCY ADDENDUM (supplementary order items 5-7; supersedes
# the earlier addenda's chase/panel/SMEM paragraphs; math/relay/reduce/
# G5/kdq/drain/epilogue are UNTOUCHED from v5.1a).
# ======================================================================
#
# K residency (item 5): score-A = one resident [own-kv64 x D512] image
# (65,536 B, 8 D64-piece stages, stage == piece).  The gather fills
# all 8 pieces ONCE per bundle (v5's 4x per-pass re-gather retired
# with the 2-slot chase ring: -75% gather traffic) under pipe_kres
# (1-stage AsyncUmma, gather_group -> leader):
#   gather producer_acquire <- leader release, tcgen05-tracked
#     score(3)(i-1) READ completion ("K(i+1) gather gate = score(3)(i)
#     completion edge"); release sits after score(3), before
#     grads(pair1), so the next fill overlaps the whole grads/G5 tail;
#   gather producer_commit (8 pieces + one cp.async drain + fence)
#     -> leader bundle-head consumer_wait.
# Score passes window-read: A fragment stage index == piece.
# Gather kdq rendezvous: r1(b-1) moves from the old piece-2 boundary
# to after the fill/commit (global r0(b), r1(b), ... order unchanged;
# the fill no longer waits late r1 credits -- strictly tighter).
#
# Q/dO strips (item 6): the one-shot dual panels retire.  Per tensor,
# a 2-stage [h16 x D512] strip double buffer (2 x 16,384 B); strip t
# holds the CTA window H[(t//2)*64 + rank*32 + (t%2)*16 : +16) (gmem
# h16-tile 4*(t//2) + (t%2) + 2*rank).  pipe_strip (2-stage AsyncUmma,
# load_elect -> leader), one generation = the pass's Q+dO pair:
#   W17 acquire <- leader release = score(t-2) B-read completion;
#   W17 commit (synchronous TMA on the reused stationary mbar,
#     expect_tx = 32,768 B) -> the pass's strip wait (in the score
#     helper, before the first gemm).
# Byte identity: one strip stage == eight contiguous [n16 x k64]
# score-B stages, so the 16-stage score-B layout binds the 2-strip
# buffer at its base; B fragment stage index = (t%2)*8 + piece.
# stationary_ready_mbar is dormant (kept initialized).
#
# SMEM account (item 7, asserted <= 215,040 upper-bound with echo):
#   header mbars pad 1,024 | Q strips 32,768 | dO strips 32,768 |
#   K residency 65,536 | round 32,768 | P slab 16,384 | dS slab
#   16,384 | dq_b 16,384 | stats 1,024 = 215,040 (-16,384 vs v5.1a).
#
# Wait-graph deltas (all other edges unchanged from V5.1):
#   score(t)  <- strip full(t%2) [W17] + s/dp empty [math T2R(t-2)]
#             (+ the bundle-head kres full wait in the leader)
#   gather(i) <- kres empty [score(3)(i-1) completion]
#   W17 strip(t) <- strip empty [score(t-2) completion]
#   W17 round/kdq edges and the kdq rendezvous law: unchanged.
# Liveness: strip(2)(b) gate = score(0)(b) completion << math(0/1)
# publish, so W17 never delays grads(pair0); cross-bundle closes via
# G5 -> score(0)(b+1) exactly as before.
#
# Trace-readout deltas: LOAD_QDO becomes the strip-supply span
# (4/bundle, payload = b*4 + t); LOAD_K keeps payload = b but now
# reads as pure 8-piece fill time (the kres gate wait sits OUTSIDE
# the span).  All other payloads unchanged.  Static names net +0
# (28/28 held; kscore had no span of its own).
# ======================================================================
# ======================================================================
# V5.2 EVICTION + SLOT ARMY + FUSED DRAIN ADDENDUM (change order
# 2026-08-04-b; supersedes the earlier addenda's TMEM map, G5/dQ, drain
# and dq_done/TAIL paragraphs; score/math-loop/gather/W17-supply-form/
# relay/kdq are UNTOUCHED from v5.1b except the MAT_ACQ probes).
# ======================================================================
#
# TMEM map ([fix-r7]): dQ rot 2x16 [0,32) | S pp [32,64) | dP pp
# [64,96) | dV/dK army 4x64 [96,352) | [352,512) FREE (160 columns
# of booked headroom; 32+64+256 = 352).
#
# G5 eviction plane: (M,N,K) = (D128, h32, kv64) CG2, 16 blocks/bundle
# = 4 D-rounds x 4 h32 windows, K chained over both kv waves; issue
# order (r_old, t, d_half) -- the round ring holds one kdq gen pair at
# a time (double-hold, double-release); rotating slot == d_half
# (static block-ordinal parity).  Block (t, r) column n is head
# (n//16)*64 + t*16 + (n%16), D rows = the A strips
# D[r_old*256 + rank*128 + d_half*64, +64) ([fix-r8]: the legacy gen
# M-half interposes rank; the eight strips tile D512).  A = kdq gen
# d-half windows (2-stage auto layout, stage stride 4096 == legacy
# m-half stride, order-(2,1,3) algebra); B = hand dq_b window view
# ((16,(8,2)),1,4,(4,2)) : ((1,(64,512)),0,1024,(16,4096)), flat stage
# t + 4*w (32 B mid-atom starts, k_block precedent).
#
# dQ eviction: pipe_dq_evict (UmmaAsync, 2 stages, leader -> reduce).
# The reduce warpgroups offload each block with plain LDG/FADD/STG on
# the f32 workspace (carved from the EXTENDED LSE/OdO workspace --
# V2 overrides _get_workspace_size_LSE_OdO: entry bytes 8 -> 8 + D*4;
# the harness allocates via impl_cls, the public wrapper path is
# unaffected).  First bundle stores directly (stateless across runs);
# last bundle casts the terminal sum straight into mdQ (bf16, ONE
# final rounding -- the retired TMA epilogue's numeric class).  The
# dQ TMA epilogue, dq_done pipeline and the TAIL dq_done commit are
# retired; _zero_dq_v2 keeps the tile_count == 0 contract.
# Offloader role choice: tcgen05.ld exposes only the executing warp's
# 32-DP slice, so [128 DP x 16 col] needs a full warpgroup -- W18/W19
# (2 warps) are physically short; the reduce pair has post-fusion
# slack and the T2R machinery in place.
#
# Fused drain: the reducer consumes dkv_done generations in (dV, dK)
# pairs from the FOUR-slot ring ([fix-r7]: slot == stage ==
# (2r + p) % 4, a COMPILE-TIME constant because 16 blocks/bundle mod
# 4 == 0 resets the phase every bundle -- static tuple selection,
# alignment folds automatically), sums the pair in registers (V == K
# => same destination), and issues ONE red.global f32x2 stream -- 8
# fused drains/bundle (the T3-HO4 P0 gate: expect ~1.2-1.4 us/pair vs
# the old 2 x ~2 us).  4 slots = 2 draining + 2 in flight (the
# double-warpgroup floor); drain throughput is the pacer, so the
# 6-slot runway (~0.7 us/pair) is forfeit by main-session ruling.
#
# Wait-graph deltas (all other edges unchanged from v5.1b):
#   grads block  <- dkv empty [fused drain of gen n-4, fix-r7]
#   G5 block     <- dq_evict empty [offload of block n-2]
#                <- round full [kdq gen pair r_old, double-hold]
#                <- mb_dqb[0] AND mb_dqb[1] (hoisted: every block
#                   reads both waves)
#   fused drain  <- dkv full x2 [leader (P,r,dV) + (P,r,dK) commits]
#   offload      <- dq_evict full [leader block commit]
#   math dqb_free edge, kres/strip edges: unchanged.
# Liveness: leader G5 stalls at block 2 on evict credits until the
# reducer finishes its 8 fused drains and starts offloading -- a
# forward chain (drains depend only on already-issued grads commits),
# no cycle; bundle-0 init credits: dkv 4 ([fix-r7]), evict 2,
# ring/kres/strip as before.
#
# [fix-r6 -> fix-r7] The r6 alignment attempt (.align(min_align) on
# the runtime slot pointers) was struck down by the DSL itself
# ("aligning a TMEM pointer is not supported", r7 gate).  Structural
# resolution: the runtime addressing's root cause was 16 mod 6 != 0
# (cross-bundle slot drift); with FOUR slots 16 mod 4 == 0, the slot
# index is compile-time static and the whole alignment question
# dissolves (static offsets fold, the v32 form).  The dQ rotating
# slots and G5 accumulators were always static and unaffected.
#
# [fix-r5] The offload T2R atom is Ld16x256b Repetition(1) (16 DP x
# 8 cols): split_wg splits the partitioned tensor's LAST (column-
# iteration) mode by the warpgroup count, so the 16-column block
# needs rest_col = 2 (Rep(2) collapses it to 1 -> the r5 gate).  The
# two warpgroups split the block 8/8 column-wise, drain-congruent.
#
# Trace-readout deltas: DQ_EPI(r) is now the OFFLOAD span (payload
# b*16 + t*4 + r; includes the evict wait -- rhythm visibility);
# WAIT_dK/REDUCE_T2R/REDUCE_ATOMIC = one set per FUSED pair (payload
# b*8 + pair*4 + r); MAT_ACQ(m,g) reinstated on every W17 supply
# acquire (payload b*32 + ordinal: strips 0-3, wide 4-19, kdq 20-23)
# -- the S_ISSUE-slope cross-examination witness.  dVdK_ISSUE payload
# UNCHANGED (b*16 + pair*8 + r*2 + p); DQ_EPI in-bundle sequence is
# non-monotonic across the two r_old groups.  Names: 28 spans +
# provenance = 29/29 (MAT_ACQ reinstated into the ledger).
#
# Verdict hooks (pre-registered): period ~35 -> ~20-22 us (e2e
# ~14-15 ms); MERGE lanes halve and decouple from MMA issue; grads
# windows collapse toward back-to-back bursts; a new DQ_EPI offload
# cadence appears at the bundle tails.  R7/R8 (residency window /
# strip descriptor layouts) remain the prime suspects for the r4
# S_ISSUE slope -- MAT_ACQ is the cross-examination probe.
# ======================================================================
