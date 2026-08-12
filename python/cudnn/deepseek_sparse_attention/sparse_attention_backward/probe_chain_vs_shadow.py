#!/usr/bin/env python3
"""probe_chain_vs_shadow: P-chain vs dP-shadow, in vitro (campaign §10.5).

Question (必然暴露定理, hardware form): with the kernel's EXACT shapes,
issue counts, and instruction sequences -- but nothing else running --
is [wake + T2R(S) + packed exp2 + publish(store+fence+arrive) + DSM
4KB exchange] longer than the dP MMA shadow [32 CG2 atoms,
M128N64K512, issue->commit-landed]?

Three arms, every iteration, timestamps via %globaltimer (no IKET, no
span boundaries, no instrumentation registers on the measured paths):

  A  shadow isolated : leader alone issues the dP replica, waits its
                       umma-done pipeline.        t: go, issue_end, done
  B  chain isolated  : math warps + exchange warp run the full P chain,
                       leader idle.               t: wake, t2r, math,
                                                  publish, xchg_seen, landed
  C  race (in-kernel form): both released by the same cluster barrier.
                       exposure := chain_landed - shadow_done

Replication ledger (все aligned with dsa_bwd_sm100_2cta_final_ser_kq4c):
  - dP replica: score_tiler (128,64,128) CG2, K_CHUNKS=4, K-major/K-major,
    32 atoms issued exactly as _issue_four_chunks (:942).
  - T2R: Ld16x256bOp(Rep(4)) on the CG2 score fragment (:4151).
  - math: 16x fma_packed_f32x2 + 32x exp2 + bf16 downcast per thread
    (kq6a form: stats are register constants).
  - publish: StMatrix8x8x16bOp x4 via get_smem_store_op tiled copy into
    the S<3,4,3> COL_MAJOR (64,64) image; owns_n split p_local/p_xchg;
    close = fence_view_async_shared + sync_warp + elect_one arrive
    (count-4 p_ready), i.e. the kq2 close.
  - exchange: cp.async.bulk.shared::cluster 4096B to the peer inbox,
    landing mbar with expect_tx, both directions.

Run on B200 (release venv, solo GPU), ~seconds:

  CUTE_DSL_ARCH=sm_100a \
  /home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python \
      probe_chain_vs_shadow.py

Output: medians of every segment + the race exposure, with the
pre-registered verdict table from ledger §10.5.
"""

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import statistics

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warp
from cutlass.cute.typing import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except ImportError:
    from cutlass.utils import sm100_utils
import cutlass.utils as utils

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

# --- kq4c constants, verbatim ---
ELEM = BFloat16
ACC = Float32
H_TILE_CLUSTER = 128
H_TILE_CTA = 64
N_TILE = 64
N_TILE_CTA = 32
K_CHUNK = 128
K_CHUNKS = 4  # D512 / K_CHUNK
PDS_BLOCK_BYTES = 4096
CLUSTER_SHAPE_MNK = (2, 1, 1)

ITERS = 34
STEADY = (8, 30)
SLOTS = 16
MATH_WARPS = 4
LEADER_WARP = 4
XCHG_WARP = 5
THREADS_PER_CTA = 256


@dsl_user_op
def _read_global_timer(*, loc=None, ip=None) -> cutlass.Int64:
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
        (
            "cp.async.bulk.shared::cluster.shared::cta."
            "mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];"
        ),
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class ChainShadowProbe:
    def __init__(self):
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=THREADS_PER_CTA,
        )

    @cute.jit
    def __call__(
        self,
        mResults: cute.Tensor,
        stream: cuda.CUstream,
    ):
        cg2 = tcgen05.CtaGroup.TWO
        score_tiler = (H_TILE_CLUSTER, N_TILE, K_CHUNK)
        dp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ELEM,
            ELEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
            ACC,
            cg2,
            score_tiler[:2],
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(CLUSTER_SHAPE_MNK),
            (dp_tiled_mma.thr_id.shape,),
        )
        a_layout_staged = sm100_utils.make_smem_layout_a(
            dp_tiled_mma,
            score_tiler,
            ELEM,
            K_CHUNKS,
        )
        b_layout_staged = sm100_utils.make_smem_layout_b(
            dp_tiled_mma,
            score_tiler,
            ELEM,
            K_CHUNKS,
        )
        score_cta_shape = (H_TILE_CTA, N_TILE, K_CHUNK)
        score_epi_tile = sm100_utils.compute_epilogue_tile_shape(
            score_cta_shape,
            True,
            utils.LayoutEnum.ROW_MAJOR,
            ACC,
        )
        score_tmem_load = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            ACC,
        )
        score_store_domain = (N_TILE, N_TILE)
        score_store_layout = sm100_utils.make_smem_layout_epi(
            ELEM,
            utils.LayoutEnum.COL_MAJOR,
            score_store_domain,
            1,
        )

        @cute.struct
        class SharedStorage:
            score_a: cute.struct.Align[
                cute.struct.MemRange[ELEM, 32768], 1024
            ]
            score_b: cute.struct.Align[
                cute.struct.MemRange[ELEM, 16384], 1024
            ]
            p_block: cute.struct.Align[
                cute.struct.MemRange[ELEM, 4096], 1024
            ]
            p_xchg: cute.struct.Align[
                cute.struct.MemRange[ELEM, 2048], 1024
            ]
            inbox: cute.struct.Align[
                cute.struct.MemRange[ELEM, 2048], 1024
            ]
            dp_done_mbars: cute.struct.MemRange[Int64, 2]
            p_ready_mbar: cute.struct.MemRange[Int64, 1]
            landing_mbar: cute.struct.MemRange[Int64, 1]
            tmem_holding_buf: Int32
            tmem_dealloc_mbar: Int64

        self.shared_storage = SharedStorage

        self.kernel(
            dp_tiled_mma,
            cluster_layout_vmnk,
            a_layout_staged,
            b_layout_staged,
            score_tmem_load,
            score_store_domain,
            score_store_layout,
            mResults,
        ).launch(
            grid=(2, 1, 1),
            block=[THREADS_PER_CTA, 1, 1],
            cluster=CLUSTER_SHAPE_MNK,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        dp_tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        a_layout_staged: cute.ComposedLayout,
        b_layout_staged: cute.ComposedLayout,
        score_tmem_load: cute.CopyAtom,
        score_store_domain: cute.Shape,
        score_store_layout: cute.ComposedLayout,
        mResults: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(
            cute.arch.warp_idx()
        )
        rank = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        peer_rank = Int32(1) - rank

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        s_a = storage.score_a.get_tensor(a_layout_staged)
        s_b = storage.score_b.get_tensor(b_layout_staged)

        # Extract every storage-derived value BEFORE any dynamic control
        # flow: the DSL tree-flattens dynamic-if scopes and cannot
        # flatten the raw @cute.struct instance (first-run lesson).
        p_ready_bar = p_ready_bar
        landing_bar = landing_bar
        p_block_int = p_block_int
        p_xchg_int = p_xchg_int
        inbox_int = inbox_int

        # mbar init
        if tidx == Int32(0):
            cute.arch.mbarrier_init(p_ready_bar, MATH_WARPS)
            cute.arch.mbarrier_init(landing_bar, 1)

        # pipelines (umma-done, consumer = the leader warp itself)
        leader_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            1,
        )
        atom_thr_size = cute.size(dp_tiled_mma.thr_id.shape)
        leader_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            atom_thr_size,
        )
        pipe_dp_done = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=leader_group,
            consumer_group=leader_consumer_group,
            barrier_storage=storage.dp_done_mbars.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

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
        tmem.allocate(128)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC)

        c_shape = dp_tiled_mma.partition_shape_C(
            (H_TILE_CLUSTER, N_TILE)
        )
        c_layout = dp_tiled_mma.make_fragment_C(c_shape).layout
        t_score = cute.make_tensor(tmem_ptr + 0, c_layout)
        t_dp = cute.make_tensor(tmem_ptr + 64, c_layout)

        rank_mma = dp_tiled_mma.get_slice(rank)
        rank_coords = rank_mma.partition_C(
            cute.make_identity_tensor((H_TILE_CLUSTER, N_TILE))
        )
        a_fragment = dp_tiled_mma.make_fragment_A(s_a)
        b_fragment = dp_tiled_mma.make_fragment_B(s_b)

        dp_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2
        )
        dp_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )

        results = mResults  # [ITERS, 2, SLOTS] int64

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # ---- role bodies -------------------------------------------------
        if warp_idx == Int32(LEADER_WARP):
            # leader: arms A and C issue the dP replica
            for i in cutlass.range(ITERS):
                # ARM A -----------------------------------------------
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                t0 = _read_global_timer()
                if rank == Int32(0):
                    dp_producer = self._issue_dp_replica(
                        dp_tiled_mma,
                        t_dp,
                        a_fragment,
                        b_fragment,
                        pipe_dp_done,
                        dp_producer,
                    )
                t1 = _read_global_timer()
                pipe_dp_done.consumer_wait(dp_consumer)
                pipe_dp_done.consumer_release(dp_consumer)
                dp_consumer.advance()
                t2 = _read_global_timer()
                # ARM B (idle) ----------------------------------------
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                # ARM C -----------------------------------------------
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                t9 = _read_global_timer()
                if rank == Int32(0):
                    dp_producer = self._issue_dp_replica(
                        dp_tiled_mma,
                        t_dp,
                        a_fragment,
                        b_fragment,
                        pipe_dp_done,
                        dp_producer,
                    )
                t10 = _read_global_timer()
                pipe_dp_done.consumer_wait(dp_consumer)
                pipe_dp_done.consumer_release(dp_consumer)
                dp_consumer.advance()
                t11 = _read_global_timer()
                with cute.arch.elect_one():
                    results[i, rank, 0] = t0
                    results[i, rank, 1] = t1
                    results[i, rank, 2] = t2
                    results[i, rank, 9] = t9
                    results[i, rank, 10] = t10
                    results[i, rank, 11] = t11
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

        elif warp_idx < Int32(MATH_WARPS):
            mtx = tidx
            score_copy = tcgen05.make_tmem_copy(
                score_tmem_load, t_score
            )
            score_thread = score_copy.get_slice(mtx)
            score_source = score_thread.partition_S(t_score)
            score_coordinates = score_thread.partition_D(rank_coords)
            r_score = cute.make_rmem_tensor(
                score_coordinates.shape, ACC
            )
            r_p = cute.make_rmem_tensor(
                score_coordinates.shape, ELEM
            )
            smem_store_atom = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                ELEM,
                ACC,
                score_copy,
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(
                smem_store_atom,
                score_copy,
            )
            thread_copy_r2s = tiled_copy_r2s.get_slice(mtx)
            n_owner = cute.arch.make_warp_uniform(
                Int32(
                    cute.get(score_coordinates[0], mode=[1])
                )
                // Int32(N_TILE_CTA)
            )
            owns_n = n_owner == rank
            p_local_store = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        ELEM,
                        p_block_int,
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    score_store_layout.inner,
                    dtype=ELEM,
                ),
                score_store_domain,
            )
            p_xchg_store = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        ELEM,
                        p_xchg_int
                        - n_owner * Int32(PDS_BLOCK_BYTES),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    score_store_layout.inner,
                    dtype=ELEM,
                ),
                score_store_domain,
            )
            t_rs_local = thread_copy_r2s.partition_D(p_local_store)
            t_rs_xchg = thread_copy_r2s.partition_D(p_xchg_store)
            assert cute.size(t_rs_local, mode=[4]) == 1
            t_rs_local_tile = t_rs_local[None, None, None, None, 0]
            t_rs_xchg_tile = t_rs_xchg[None, None, None, None, 0]
            scale_c = Float32(0.1250)
            lse_c = Float32(1.0)

            for i in cutlass.range(ITERS):
                # ARM A (idle)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                # ARM B: chain isolated -------------------------------
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                t3 = _read_global_timer()
                cute.copy(score_copy, score_source, r_score)
                cute.arch.fence_view_async_tmem_load()
                t4 = _read_global_timer()
                self._p_math(r_score, r_p, scale_c, lse_c)
                t5 = _read_global_timer()
                r_p_store = thread_copy_r2s.retile(r_p)
                if owns_n:
                    cute.copy(
                        tiled_copy_r2s, r_p_store, t_rs_local_tile
                    )
                else:
                    cute.copy(
                        tiled_copy_r2s, r_p_store, t_rs_xchg_tile
                    )
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        p_ready_bar
                    )
                t6 = _read_global_timer()
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        results[i, rank, 3] = t3
                        results[i, rank, 4] = t4
                        results[i, rank, 5] = t5
                        results[i, rank, 6] = t6
                # ARM C: same chain, racing the leader ---------------
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                t12 = _read_global_timer()
                cute.copy(score_copy, score_source, r_score)
                cute.arch.fence_view_async_tmem_load()
                self._p_math(r_score, r_p, scale_c, lse_c)
                r_p_store2 = thread_copy_r2s.retile(r_p)
                if owns_n:
                    cute.copy(
                        tiled_copy_r2s, r_p_store2, t_rs_local_tile
                    )
                else:
                    cute.copy(
                        tiled_copy_r2s, r_p_store2, t_rs_xchg_tile
                    )
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        p_ready_bar
                    )
                t13 = _read_global_timer()
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        results[i, rank, 12] = t12
                        results[i, rank, 13] = t13
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

        elif warp_idx == Int32(XCHG_WARP):
            p_phase = Int32(0)
            l_phase = Int32(0)
            for i in cutlass.range(ITERS):
                # ARM A (idle)
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                # ARM B ----------------------------------------------
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        landing_bar,
                        PDS_BLOCK_BYTES,
                    )
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.mbarrier_wait(
                    p_ready_bar, p_phase
                )
                p_phase = p_phase ^ Int32(1)
                t7 = _read_global_timer()
                with cute.arch.elect_one():
                    _cpasync_bulk_s2cluster(
                        cute.make_ptr(
                            ELEM,
                            p_xchg_int,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        ),
                        cute.make_ptr(
                            ELEM,
                            inbox_int,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        ),
                        landing_bar,
                        PDS_BLOCK_BYTES,
                        peer_rank,
                    )
                cute.arch.mbarrier_wait(
                    landing_bar, l_phase
                )
                l_phase = l_phase ^ Int32(1)
                t8 = _read_global_timer()
                with cute.arch.elect_one():
                    results[i, rank, 7] = t7
                    results[i, rank, 8] = t8
                # ARM C ----------------------------------------------
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        landing_bar,
                        PDS_BLOCK_BYTES,
                    )
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.mbarrier_wait(
                    p_ready_bar, p_phase
                )
                p_phase = p_phase ^ Int32(1)
                with cute.arch.elect_one():
                    _cpasync_bulk_s2cluster(
                        cute.make_ptr(
                            ELEM,
                            p_xchg_int,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        ),
                        cute.make_ptr(
                            ELEM,
                            inbox_int,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        ),
                        landing_bar,
                        PDS_BLOCK_BYTES,
                        peer_rank,
                    )
                cute.arch.mbarrier_wait(
                    landing_bar, l_phase
                )
                l_phase = l_phase ^ Int32(1)
                t14 = _read_global_timer()
                with cute.arch.elect_one():
                    results[i, rank, 14] = t14
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

        else:
            for i in cutlass.range(ITERS):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()

    @cute.jit
    def _issue_dp_replica(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        a_fragment: cute.Tensor,
        b_fragment: cute.Tensor,
        done_pipeline,
        producer_state,
    ):
        """Verbatim _issue_four_chunks: 32 CG2 atoms over 4 D128 chunks."""
        done_pipeline.producer_acquire(producer_state)
        mma = tiled_mma.with_()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        k_blocks_per_chunk = cute.size(a_fragment, mode=[2])
        for flat_k_block in cutlass.range_constexpr(
            K_CHUNKS * 8
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
    def _p_math(self, r_score, r_p, scale_c, lse_c):
        """kq6a P math: 16 packed FMAs + 32 exp2 + bf16 downcast."""
        assert cute.size(r_score) == N_TILE_CTA
        for pair in cutlass.range_constexpr(N_TILE_CTA // 2):
            i0 = 2 * pair
            i1 = 2 * pair + 1
            v0, v1 = cute.arch.fma_packed_f32x2(
                (r_score[i0], r_score[i1]),
                (scale_c, scale_c),
                (lse_c, lse_c),
            )
            v0 = cute.math.exp2(v0, fastmath=True)
            v1 = cute.math.exp2(v1, fastmath=True)
            r_score[i0] = v0
            r_score[i1] = v1
            r_p[i0] = ELEM(v0)
            r_p[i1] = ELEM(v1)


def main():
    torch.cuda.init()
    results = torch.zeros(ITERS, 2, SLOTS, dtype=torch.int64, device="cuda")
    probe = ChainShadowProbe()
    stream = resolve_stream(None)
    compiled = cute.compile(
        probe,
        to_cute_tensor(results, assumed_align=8),
        stream,
        options=compile_options(),
    )
    compiled(to_cute_tensor(results, assumed_align=8), stream)
    torch.cuda.synchronize()
    r = results.cpu().numpy()

    def med(expr):
        vals = [expr(i) for i in range(*STEADY)]
        vals = [v for v in vals if v > 0]
        return statistics.median(vals) if vals else float("nan")

    print("=== probe_chain_vs_shadow (ns, medians over steady iters) ===")
    print("[A] shadow isolated (leader, rank0):")
    print(f"    issue span        : {med(lambda i: r[i,0,1]-r[i,0,0]):8.0f}")
    print(f"    issue->done       : {med(lambda i: r[i,0,2]-r[i,0,0]):8.0f}")
    print("[B] chain isolated (math warp0 + xchg, per rank0):")
    print(f"    T2R               : {med(lambda i: r[i,0,4]-r[i,0,3]):8.0f}")
    print(f"    math              : {med(lambda i: r[i,0,5]-r[i,0,4]):8.0f}")
    print(f"    publish(close)    : {med(lambda i: r[i,0,6]-r[i,0,5]):8.0f}")
    print(f"    xchg wait+copy    : {med(lambda i: r[i,0,8]-r[i,0,6]):8.0f}")
    print(f"    CHAIN total       : {med(lambda i: r[i,0,8]-r[i,0,3]):8.0f}")
    print("[C] race (both released together):")
    print(f"    shadow done       : {med(lambda i: r[i,0,11]-r[i,0,9]):8.0f}")
    print(f"    chain landed      : {med(lambda i: r[i,0,14]-r[i,0,9]):8.0f}")
    print(f"    EXPOSURE          : {med(lambda i: r[i,0,14]-r[i,0,11]):8.0f}")
    print()
    print("verdict table (ledger §10.5): chain>shadow isolated AND race")
    print("exposure ~ in-kernel 1.4-1.8us -> theorem holds in hardware;")
    print("chain<=shadow isolated but race exposed -> interference-type;")
    print("race exposure << in-kernel -> kernel relay protocol is the gap.")


if __name__ == "__main__":
    main()
