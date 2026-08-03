#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blackwell sm_100 簇内 bulk-DSM push 的 complete_tx 落点探针 (CuTe DSL 4.5.2)。

================================================================================
RUNBOOK
================================================================================
目标（R1 生死门）：验证一条
    cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
把本 CTA SMEM 的 4,096B 推到 peer CTA SMEM 时，complete_tx 的 mbarrier 操作数
能否直接指向 **peer CTA 的 mbar**（mapa 簇地址），且 expect_tx 由 peer 自己
在本地 arm —— 使 peer 仅在自己本地 mbar 上 wait 即可感知数据到达，发送端
无需再发任何 arrive。

三个变体（一个 kernel 类按编译期 self.variant 特化，host 逐个编译+发射）:
  main     : R1 主案。peer 本地 arrive_and_expect_tx(4096) 于自己的 dqb_ready
             mbar；发送端 bulk push 的 dst 与 mbar 都经 mapa 指向 peer。
  fallback : R1 回退。发送端本地 arm 自己的 tx_done mbar（expect_tx 4096），
             push 的 mbar 操作数留在本地；发送端等到 tx_done 相位翻转后，
             elected lane 对 peer 的 dqb_ready 发一次 remote arrive
             （cluster scope）；peer 等本地 dqb_ready。
  prod     : 校准点（非门）。v15/v17a 生产在役形态：发送端 *远程*
             arrive_and_expect_tx(peer mbar) + push(mbar 也指 peer)。
             若 prod 都 fail，说明探针/环境坏了，勿采信 main/fallback 结论。

拓扑：cluster (2,1,1)，每 CTA 1 warp（32 线程），单 cluster，grid=(2,1,1)。
流程：两 CTA 各写 rank 特征 pattern 到 src（word i = (0x1000+rank)<<16 | i），
dst 预填哨兵 0x0BAD0000+i → fence_view_async_shared → cluster 同步 →
按变体发 4,096B push → 各自在本地 dqb_ready 上做**有界** try_wait（不会挂死，
超时自报）→ 逐字校验 dst == 对端 pattern → 结果写 gmem flags。

环境（haifa B200 venv，与 mma_atom_price.py 同）:
    需要: nvidia-cutlass-dsl==4.5.2, torch(+cu12x), cuda-python (cuda.bindings)
    source <venv>/bin/activate
    timeout 180 python3 dsm_cluster_probe.py                 # 默认三变体全跑
    python3 dsm_cluster_probe.py --variants main,fallback    # 只跑两个门
    python3 dsm_cluster_probe.py --json out.json

预期输出（stdout）:
    R1_MAIN ok            （或 fail + 下一行缩进的两 CTA 细节）
    R1_FALLBACK ok
    R1_CALIB_PRODFORM ok
    末尾机读一行: "DSM_CLUSTER_PROBE_JSON [...]"
判读:
    - main ok                        → R1 主案成立，设计用主案。
    - main fail: wait=timeout 且 dst0 == exp0（数据到了但 mbar 没翻）
                                     → complete_tx 没落到 peer mbar，转回退。
    - main fail: cute.compile 抛错（ptxas 拒编）→ 转回退。
    - main fail: bad>0 且 dst0 是哨兵 0x0BAD00xx → push 本身没发出去，先查环境
                 （对照 prod 变体）。
    - fallback 必须 ok（回退路径本身的可行性验证）。
    - prod fail                      → 探针或环境损坏（该形态已在 v15/v17a
                                       生产验证过），全部结论作废重查。

flags 布局（每变体 16 x u32，索引 = 槽位k*?; 偶数=cta0 奇数=cta1）:
    [0/1]  dqb_ready wait 状态: 0=kernel没写(没跑到), 1=ok, 2=超时
    [2/3]  dst 错误 word 计数（red.add，全 warp 汇总）
    [4/5]  首个错误 word 下标（red.min；host 预置 0xFFFFFFFF）
    [6/7]  wait 后实测 dst[0]      [8/9]  实测 dst[1023]
    [10/11] fallback 专用: 发送端本地 tx_done wait 状态（其余变体恒 0）
    [12/13] 期望 dst[0]（=对端 pattern word0，便于肉眼比对）
    [14/15] dqb wait 的 try 次数（诊断"接近超时"）

================================================================================
DSL 硬约束自查（历史教训，逐条声明）
================================================================================
1. setmaxnreg: 全文件不使用。
2. cute.Tensor 不作 kernel 运行时实参：kernel 唯一运行时实参是 flags_ptr
   (cutlass.Int64 标量，torch data_ptr)；SMEM 张量全部 kernel 内经
   SmemAllocator/@cute.struct 构造；gmem 写走 st.global/red.global inline PTX。
3. staged 禁 raise：kernel/@cute.jit 内零 raise 零 assert-on-dynamic；host
   侧 try/except 按变体兜底。
4. staged 编译期分支全包 cutlass.const_expr（self.variant 三态、fallback 的
   tx_status 落盘）；所有动态分支内被重新绑定的名字（tx_status/status/spins/
   bad/first_bad/keep*）均在分支/循环之前顶层预初始化。
5. 编译期 int 判定：不使用 isinstance/type(x) is int 相关技巧，无该场景。
6. tmem: 本探针不触碰（无 TmemAllocator）。
7. mbarrier 相位：每个 mbar 单次使用、全程只等 parity 0；无相位翻转复用。

================================================================================
两条路径的 PTX 形态（wrapper 会实际发射的指令，供 review）
================================================================================
主案 main（发送端视角；peer 已本地做过
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [dqb_ready], 4096; ）:
    mapa.shared::cluster.u32  %r_dst,  %local_dst_smem, %peer;
    mapa.shared::cluster.u32  %r_mbar, %local_dqb_smem, %peer;
    cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
        [%r_dst], [%local_src], 4096, [%r_mbar];
    // peer: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 有界自旋

回退 fallback（发送端视角）:
    mbarrier.arrive.expect_tx.shared::cta.b64 _, [tx_done], 4096;   // 本地 arm
    mapa.shared::cluster.u32  %r_dst, %local_dst_smem, %peer;
    cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
        [%r_dst], [%local_src], 4096, [tx_done];                    // mbar 本地
    // 等 tx_done parity0（acquire.cluster）后:
    //   cute.arch.mbarrier_arrive(dqb_ready, peer_cta_rank_in_cluster=peer)
    //   （v17a _pair_arrive 在役同款 => remote arrive, cluster scope）

================================================================================
已知风险点（未经编译验证，需远端 runner 迭代确认的 API 点）
================================================================================
RK1. 主案语义本身（本探针的目的）：PTX 语法上 mbar 操作数是 shared::cluster
     空间地址、v15/v17a 已在生产用 *远程 arm + 远程 complete_tx* 形态跑通；
     本探针新增验证的是 *本地 arm + 远程 complete_tx* 的组合。若 fail，
     flags 的 (wait, dst0) 组合可区分"tx 丢了"vs"数据到了 mbar 没翻"。
RK2. cute.arch.mbarrier_arrive_and_expect_tx(bar, bytes) 本地形态与
     (bar, bytes, peer_cta_rank_in_cluster=peer) 远程形态：均有 v15/v17a
     在役先例（v15 L2301 / L1563），低风险；若 kwarg 名不符以 DSL 4.5.2
     签名为准调整。
RK3. cute.arch.mbarrier_arrive(bar, peer_cta_rank_in_cluster=peer) 远程
     arrive：v17a _pair_arrive (L4288) 在役先例，低风险。
RK4. _mbarrier_try_wait_parity_acq_cluster 的
     "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64" 单发形态：
     v15 有同 opcode 的 bra 自旋版(L2890)与无 acquire 的单发版(L2860)，
     本文件取两者拼接（单发 + acquire.cluster + 字面量 suspend hint
     100000ns）。若 ptxas 拒：把 hint 改字面量 1 或去掉 .acquire.cluster
     （去掉后语义上需在 wait 后补 fence，见 RK7）。
RK5. fallback 中 push 的 mbar 用**裸本地 smem 地址**充当 shared::cluster
     操作数：v15 _cpasync_bulk_g2s (L161-177) 对 dst/mbar 均传裸本地地址
     且生产在役 => 自身 CTA 的 shared::cta 地址可作簇空间别名，低风险。
RK6. st.global.u32 / red.global.add.u32 / red.global.min.u32 inline PTX
     （"l,r" 约束）：v15 _cpasync_bulk_s2g 用过 "l,r,r"，低风险。
RK7. 数据可见性链：wait 采用 acquire.cluster；生产 W18 在 plain
     mbarrier_wait 后直接读 inbox 也正确，故本探针不额外补
     fence.proxy.async。若 main 出现 "wait=ok 但 bad>0 且重跑值漂移"，
     先怀疑可见性而非落点，在 wait 后加 cute.arch.fence_view_async_shared()
     重试。
RK8. cute.compile(probe, cutlass.Int64(ptr), stream, options="--gpu-arch
     sm_100a")：纯 Int64 标量实参编译路径与 mma_atom_price.py R5 同族。
     若报参数/选项错：先去掉 options 再试；仍不行加 "--enable-tvm-ffi"。
RK9. 有界自旋参数 TRY_LIMIT=20000 x hint 100000ns ≈ 单 mbar 最长 ~2s。
     若机器负载导致假超时（spins 打满且 prod 也超时），调大 TRY_LIMIT。
RK10. 超时路径下 tidx0 读 dst[0]/dst[1023] 采样本质是 racy（可能与迟到的
     DSM 写并发），仅作诊断字段，不参与 ok 判定。
RK11. fallback 分支里"动态 if tidx==0 内嵌 while 自旋"结构（scf.if 套
     scf.while）在参考 kernel 中无同款先例（v15 的 if tidx==0 都是直线
     代码）。若 tracer/IR 拒绝：改为全 warp 对 tx_done 做同谱自旋
     （把 tx 自旋段从 if tidx==0 提出到顶层，与 dqb 等待同款；tx_done
     只被 tidx0 arm 但任何线程都可 try_wait），随后仍仅 tidx0 发
     remote arrive。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.typing import Int32


# ======================================================================
# inline-PTX 原语（形态照抄 v15/v17a 在役 wrapper）
# ======================================================================


@dsl_user_op
def _map_smem_to_cluster_rank(
    smem_ptr: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """mapa：把本 CTA 的 shared::cta 地址映射成 peer rank 的簇地址。"""

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
def _cpasync_bulk_s2cluster_peer_bar(
    source: cute.Pointer,
    destination: cute.Pointer,
    completion_barrier: cute.Pointer,
    copy_bytes: int | Int32,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """主案/prod push：dst 与 mbar **都** mapa 到 peer（v15 在役同款）。"""

    source_i32 = source.toint(loc=loc, ip=ip).ir_value()
    destination_i32 = _map_smem_to_cluster_rank(
        destination, peer_rank, loc=loc, ip=ip
    ).ir_value()
    barrier_i32 = _map_smem_to_cluster_rank(
        completion_barrier, peer_rank, loc=loc, ip=ip
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
def _cpasync_bulk_s2cluster_local_bar(
    source: cute.Pointer,
    destination: cute.Pointer,
    completion_barrier: cute.Pointer,
    copy_bytes: int | Int32,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """回退 push：dst mapa 到 peer，mbar 留发送端本地（RK5）。"""

    source_i32 = source.toint(loc=loc, ip=ip).ir_value()
    destination_i32 = _map_smem_to_cluster_rank(
        destination, peer_rank, loc=loc, ip=ip
    ).ir_value()
    barrier_i32 = completion_barrier.toint(loc=loc, ip=ip).ir_value()
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
def _mbarrier_try_wait_parity_acq_cluster(
    barrier: cute.Pointer,
    parity: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """单发非阻塞 try_wait（acquire.cluster），返回 1=相位已翻/0=未翻（RK4）。

    suspend hint 100000ns 为字面量；调用方自旋计数保证有界不挂死。
    """

    barrier_i32 = barrier.toint(loc=loc, ip=ip).ir_value()
    ready = llvm.inline_asm(
        T.i32(),
        [barrier_i32, parity.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
            "p, [$1], $2, 100000;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t"
            "}"
        ),
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(ready)


@dsl_user_op
def _st_global_u32(
    address: cutlass.Int64,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _red_global_add_u32(
    address: cutlass.Int64,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _red_global_min_u32(
    address: cutlass.Int64,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.global.min.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# ======================================================================
# 探针 kernel
# ======================================================================


class DsmClusterProbe:
    """单变体 DSM push 探针；variant 是编译期属性，cute.compile 按实例特化。"""

    THREADS_PER_CTA = 32
    CLUSTER_SHAPE_MNK = (2, 1, 1)
    WORDS = 1024                     # 4,096B / 4
    BYTES = 4096
    TRY_LIMIT = 20000                # x 100us hint ≈ 最长 ~2s/mbar（RK9）
    FLAG_WORDS = 16

    VARIANTS = ("main", "fallback", "prod")

    def __init__(self, variant: str):
        assert variant in self.VARIANTS, f"bad variant {variant}"
        self.variant = variant
        self.shared_storage = None

    # ------------------------------------------------------------------
    # host 入口（唯一运行时实参: flags gmem 基址 Int64 标量 —— 约束#2）
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, flags_ptr: cutlass.Int64, stream: cuda.CUstream):
        @cute.struct
        class SharedStorage:
            # mbars[0] = dqb_ready（数据到达门）; mbars[1] = tx_done（回退用）
            mbars: cute.struct.MemRange[cutlass.Int64, 2]
            src: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, DsmClusterProbe.WORDS],
                128,
            ]
            dst: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, DsmClusterProbe.WORDS],
                128,
            ]

        self.shared_storage = SharedStorage

        self.kernel(flags_ptr).launch(
            grid=(self.CLUSTER_SHAPE_MNK[0], 1, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=self.CLUSTER_SHAPE_MNK,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # device kernel（单 warp/CTA；单相位 parity 0 —— 约束#7）
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(self, flags_ptr: cutlass.Int64):
        tidx, _, _ = cute.arch.thread_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        peer = Int32(1) - rank

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbars = storage.mbars.data_ptr()
        src_t = storage.src.get_tensor(cute.make_layout((self.WORDS,)))
        dst_t = storage.dst.get_tensor(cute.make_layout((self.WORDS,)))

        # 本 CTA 的 flags 槽基址: 偶数=cta0 奇数=cta1（槽 k 的地址 = base+8k）
        flags_base = flags_ptr + cutlass.Int64(rank) * cutlass.Int64(4)

        # 动态分支/循环里会重新绑定的名字，全部顶层预初始化（约束#4）
        tx_status = Int32(0)
        status = Int32(2)
        spins = Int32(0)
        bad = Int32(0)
        first_bad = Int32(0x7FFFFFFF)

        # ---- mbar init（每个 count=1）----
        if tidx == 0:
            cute.arch.mbarrier_init(mbars, 1)
            cute.arch.mbarrier_init(mbars + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- 主案的接收方本地 arm：必须先于 cluster 同步（先于对端 push）----
        if cutlass.const_expr(self.variant == "main"):
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(mbars, self.BYTES)

        # ---- pattern 写入: src=rank 特征, dst=哨兵 ----
        word = tidx
        while word < Int32(self.WORDS):
            src_t[word] = (Int32(0x1000) + rank) * Int32(0x10000) + word
            dst_t[word] = Int32(0x0BAD0000) + word
            word += Int32(self.THREADS_PER_CTA)
        cute.arch.fence_view_async_shared()   # 泛型写 -> async proxy 可见
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # ---- 按变体发 4,096B push（elected lane = tidx 0，v15 W18 同款）----
        if cutlass.const_expr(self.variant == "main"):
            # R1 主案: mbar 操作数 = peer 的 dqb_ready（mapa），发送端零 arrive
            if tidx == 0:
                _cpasync_bulk_s2cluster_peer_bar(
                    src_t.iterator,
                    dst_t.iterator,
                    mbars,
                    self.BYTES,
                    peer,
                )

        if cutlass.const_expr(self.variant == "prod"):
            # 校准: 生产在役形态 = 发送端远程 arm + 远程 complete_tx
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbars,
                    self.BYTES,
                    peer_cta_rank_in_cluster=peer,
                )
                _cpasync_bulk_s2cluster_peer_bar(
                    src_t.iterator,
                    dst_t.iterator,
                    mbars,
                    self.BYTES,
                    peer,
                )

        if cutlass.const_expr(self.variant == "fallback"):
            # 回退: tx 落本地 tx_done；完成后对 peer 发 remote arrive
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(mbars + 1, self.BYTES)
                _cpasync_bulk_s2cluster_local_bar(
                    src_t.iterator,
                    dst_t.iterator,
                    mbars + 1,
                    self.BYTES,
                    peer,
                )
                tx_status = Int32(2)
                tx_spins = Int32(0)
                keep_tx = Int32(1)
                while keep_tx == 1:
                    tx_ready = _mbarrier_try_wait_parity_acq_cluster(
                        mbars + 1, Int32(0)
                    )
                    tx_spins += 1
                    if tx_ready != 0:
                        tx_status = Int32(1)
                        keep_tx = Int32(0)
                    if tx_spins >= Int32(self.TRY_LIMIT):
                        keep_tx = Int32(0)
                if tx_status == 1:
                    # v17a _pair_arrive 在役同款: remote arrive, cluster scope
                    cute.arch.mbarrier_arrive(
                        mbars,
                        peer_cta_rank_in_cluster=peer,
                    )

        # ---- 有界等待本地 dqb_ready parity 0（全 warp 同谱自旋，不挂死）----
        keep = Int32(1)
        while keep == 1:
            got_ready = _mbarrier_try_wait_parity_acq_cluster(mbars, Int32(0))
            spins += 1
            if got_ready != 0:
                status = Int32(1)
                keep = Int32(0)
            if spins >= Int32(self.TRY_LIMIT):
                keep = Int32(0)
        cute.arch.barrier()

        # ---- 逐字校验 dst == 对端 pattern（超时也照跑，结果作诊断）----
        word = tidx
        while word < Int32(self.WORDS):
            got = dst_t[word]
            exp = (Int32(0x1000) + peer) * Int32(0x10000) + word
            if got != exp:
                bad += 1
                if first_bad == Int32(0x7FFFFFFF):
                    first_bad = word
            word += Int32(self.THREADS_PER_CTA)

        _red_global_add_u32(flags_base + cutlass.Int64(4 * 2), bad)
        if bad > 0:
            _red_global_min_u32(flags_base + cutlass.Int64(4 * 4), first_bad)
        if tidx == 0:
            _st_global_u32(flags_base + cutlass.Int64(4 * 0), status)
            _st_global_u32(flags_base + cutlass.Int64(4 * 6), dst_t[0])
            _st_global_u32(
                flags_base + cutlass.Int64(4 * 8), dst_t[self.WORDS - 1]
            )
            _st_global_u32(flags_base + cutlass.Int64(4 * 10), tx_status)
            _st_global_u32(
                flags_base + cutlass.Int64(4 * 12),
                (Int32(0x1000) + peer) * Int32(0x10000),
            )
            _st_global_u32(flags_base + cutlass.Int64(4 * 14), spins)

        # ---- 收尾: 全簇 drain，任何在途 DSM 流量落地前谁都不许退 ----
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


# ======================================================================
# host 侧: 编译 / 发射 / 判读
# ======================================================================

GATE_NAMES = {
    "main": "R1_MAIN",
    "fallback": "R1_FALLBACK",
    "prod": "R1_CALIB_PRODFORM",
}


def gpu_arch_flag() -> str:
    cap = torch.cuda.get_device_capability()
    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_100f"}.get(cap)
    if arch is None:
        raise RuntimeError(f"unsupported compute capability {cap}: need sm_100 家族")
    return arch


def resolve_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def decode_flags(flags: torch.Tensor, variant: str) -> dict:
    vals = [_u32(v) for v in flags.cpu().to(torch.int64).tolist()]
    ctas = []
    for r in (0, 1):
        ctas.append(
            {
                "wait": vals[0 + r],          # 0 没写 / 1 ok / 2 超时
                "bad_words": vals[2 + r],
                "first_bad": vals[4 + r],     # 0xFFFFFFFF = 无
                "dst0": f"0x{vals[6 + r]:08x}",
                "dst1023": f"0x{vals[8 + r]:08x}",
                "tx_wait": vals[10 + r],      # 仅 fallback 有效
                "exp0": f"0x{vals[12 + r]:08x}",
                "spins": vals[14 + r],
            }
        )
    ok = all(c["wait"] == 1 and c["bad_words"] == 0 for c in ctas)
    if variant == "fallback":
        ok = ok and all(c["tx_wait"] == 1 for c in ctas)
    return {"variant": variant, "ok": ok, "cta": ctas, "raw": vals}


def run_variant(variant: str, arch: str, stream) -> dict:
    probe = DsmClusterProbe(variant)
    flags = torch.zeros(DsmClusterProbe.FLAG_WORDS, dtype=torch.int32, device="cuda")
    flags[4] = -1   # first_bad 槽预置 0xFFFFFFFF（red.min 初值）
    flags[5] = -1
    torch.cuda.synchronize()
    compiled = cute.compile(
        probe,
        cutlass.Int64(flags.data_ptr()),
        stream,
        options=f"--gpu-arch {arch}",
    )
    compiled(cutlass.Int64(flags.data_ptr()), stream)
    torch.cuda.synchronize()
    return decode_flags(flags, variant)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="sm_100 簇内 bulk-DSM push complete_tx 落点探针 (R1)"
    )
    p.add_argument(
        "--variants",
        default="main,fallback,prod",
        help="逗号分隔，取值 main/fallback/prod，按序执行",
    )
    p.add_argument("--json", default=None, help="可选: 结果 JSON 输出路径")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in DsmClusterProbe.VARIANTS:
            print(f"unknown variant: {v}", file=sys.stderr)
            return 2

    torch.zeros(1, device="cuda")   # 初始化 CUDA 上下文
    stream = resolve_stream()
    arch = gpu_arch_flag()
    print(
        f"device={torch.cuda.get_device_name()} arch={arch} "
        f"variants={','.join(variants)} bytes={DsmClusterProbe.BYTES} "
        f"try_limit={DsmClusterProbe.TRY_LIMIT}"
    )

    results = []
    exit_code = 0
    for variant in variants:
        gate = GATE_NAMES[variant]
        try:
            result = run_variant(variant, arch, stream)
        except Exception as exc:  # 编译/发射失败也算该变体 fail，继续跑其余
            result = {
                "variant": variant,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        if result["ok"]:
            print(f"{gate} ok")
        else:
            print(f"{gate} fail")
            if variant != "prod":
                exit_code = 1
        if "error" in result:
            print(f"    error: {result['error']}")
        else:
            for r, c in enumerate(result["cta"]):
                wait_s = {0: "unwritten", 1: "ok", 2: "timeout"}.get(
                    c["wait"], str(c["wait"])
                )
                tx_s = {0: "-", 1: "ok", 2: "timeout"}.get(
                    c["tx_wait"], str(c["tx_wait"])
                )
                print(
                    f"    cta{r}: wait={wait_s} bad_words={c['bad_words']} "
                    f"first_bad={c['first_bad'] if c['first_bad'] != 0xFFFFFFFF else '-'} "
                    f"dst0={c['dst0']} exp0={c['exp0']} dst1023={c['dst1023']} "
                    f"tx_wait={tx_s} spins={c['spins']}"
                )

    print("DSM_CLUSTER_PROBE_JSON " + json.dumps(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"DSM_CLUSTER_PROBE_RESULT {args.json}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
