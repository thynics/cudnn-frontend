#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blackwell sm_100 tcgen05 dense bf16 CG2 MMA atom price microbench (CuTe DSL).

================================================================================
RUNBOOK
================================================================================
目标：测 M128 x N{16,32,64,128,256} x K16 CG2 (cta_group::2) dense bf16 MMA
原子的均摊单价（ns/atom），A/B 均 SMEM 源，f32 累加。单 leader warp 背靠背
发射，分批 tcgen05.commit 提交；由 N 扫描判断 exec 是否随 FLOPs(∝N) 线性
缩放，还是存在平坦的每原子发射地板。

环境（computelab / haifa B200 venv）:
    需要: nvidia-cutlass-dsl==4.5.2, torch(+cu12x), cuda-python (cuda.bindings)
    source <venv>/bin/activate
    # 强烈建议先锁频，否则 boost 漂移会污染 ns/atom:
    #   sudo nvidia-smi -lgc 1965,1965   (按机器实际 SM clock 调整)
    python3 mma_atom_price.py                        # 默认: 两种 chunk 模式都跑
    python3 mma_atom_price.py --mode fixed           # 纯发射地板(同一 SMEM chunk 反复发射)
    python3 mma_atom_price.py --mode rotate          # 8 chunk 轮转(模拟真实描述符步进)
    python3 mma_atom_price.py --dual-acc 1           # 双累加器交替(排除同累加器链式依赖)
    python3 mma_atom_price.py --n-list 64,128 --atoms 131072 --json out.json
    python3 mma_atom_price.py --load-mode all       # none/smem/t2r 三档背景负载对照

预期输出（stdout）:
    每行一个测点:
    mode  N  atoms  batches  total_ms  idle_ms  ns/atom  ns/atom(net)  ns/atom/N16(net)  TFLOPS
    末尾一行机读 JSON: "MMA_ATOM_PRICE_JSON [...]"
    判读: 若 ns/atom(net) ∝ N (即 ns/atom/N16 列近似常数) => exec 管线峰值速率;
          若 ns/atom(net) 对 N 平坦 => 每原子发射/退休地板。

结构:
    - MmaAtomPriceBench: 每个 (N, mode, idle) 组合一个实例, cute.compile 特化。
      kernel 内: 单 cluster(2,1,1) x 32 线程/CTA。零填 SMEM -> cluster 同步 ->
      leader CTA warp0 发射 num_batches x BATCH_ATOMS 个 gemm 原子, 每批一个
      tcgen05.commit(multicast mask=3) -> 两 CTA 各自在本地 mbarrier 上等
      (arrival count = num_batches, 单相位 parity 0) -> cluster 同步 -> tmem.free。
    - idle 对照 kernel: 同结构同循环, 仅去掉 gemm(commit 保留), 用于扣除
      launch/tmem alloc/零填/同步/循环骨架开销。
    - num_batches 是运行时 Int32: host 侧自动放大直到 kernel 时间 >= --target-ms,
      无需重编译。

================================================================================
DSL 硬约束自查（对照任务清单，声明已过）
================================================================================
1. setmaxnreg: 本文件完全不使用。
2. 动态指针/张量实参: kernel 唯一运行时实参是 Int32 num_batches；不传 cute.Tensor
   也不传裸指针。SMEM 张量全部 kernel 内经 SmemAllocator/@cute.struct 构造。
3. staged 代码无 raise: kernel/@cute.jit 内只有 assert（且均为 trace-time 编译期
   int 断言）；所有运行时校验在 host main() 里做。
4. staged 编译期分支全部包 cutlass.const_expr（isinstance(ComposedLayout)、
   not self.idle、self.dual_acc）；运行时分支只有 `if is_leader_cta and warp_idx == 0`
   （DSA v17a 同款先例），分支内首绑定名字(mma 等)不在 join 后使用。
5. 编译期 int 鉴别: 本文件不做 isinstance(int) 判定；chunk/acc 序列在 __init__
   里用纯 python list 预生成，range_constexpr 内直接下标。
6. tmem.allocate 无动态守卫: 与 v17a 完全同构（TmemAllocator + allocator_warp_id=0,
   is_two_cta=True + dealloc mbar），两 CTA 均无条件调用。
7. instruction shape: CG2 M=128 固定; N ∈ {16..256} 且 N%16==0（__init__ 断言）;
   经 make_trivial_tiled_mma -> MmaF16BF16Op(instruction_shape=(128,N,16)) 逐 N 特化。

================================================================================
已知风险点（未经编译验证，需要远端 runner 迭代确认的 API 调用点）
================================================================================
R1. tcgen05.commit(done_mbar, 3, CtaGroup.TWO) 多播 + mbarrier_init(ptr, num_batches)
    大动态到达计数 + 单次 parity-0 等待：常规管线用 count=1 的分级 barrier，
    此"聚合计数"用法非常规。若挂死/相位错：回退方案 A) 每批 commit 后原地
    mbarrier_wait 并翻转 parity（牺牲批间流水）；B) 换 pipeline.PipelineUmmaAsync。
    (mbarrier 硬件到达计数上限 2^20-1, host 已限 num_batches <= 1_000_000。)
R2. idle 对照 kernel 的 commit 无任何未提交 MMA —— 假定按 PTX 语义立即到达。
    若 idle 模式挂死: 把 kernel() 中 IDLE-commit 分支改为
    `cute.arch.mbarrier_arrive(done_mbar)`（elect_one 下），语义等价于计数递增。
R3. make_smem_layout_a/b 以 mma_tiler=(128,N,16) (K 主模 16 元素 = 32B) 走
    的 layout atom 启发式路径未验证；kernel 内 assert MMA_K==1 / STAGE==k_chunks
    会在 trace 时把形状问题暴露为编译错。
R4. tiled_mma / smem layout 在 host __call__ 与 kernel 内各自重建（不作 kernel
    实参传递，规避 TiledMma/ComposedLayout 注解匹配风险）——两处输入相同、
    结构均静态，理论一致；若 struct 尺寸与 kernel 内 cosize 不一致会在
    SmemAllocator.allocate 处炸。
R5. cute.compile(bench, Int32, CUstream, options="--gpu-arch sm_100a")：纯标量
    实参 + 无 --enable-tvm-ffi 的编译路径未验证。若报参数/选项错: 先去掉
    options 再试；仍不行加 "--enable-tvm-ffi"。
R6. `cutlass.range(tidx, cosize, 32)` 动态起点零填循环 与
    `cutlass.range(num_batches)` 动态批循环内嵌 64 个展开 gemm + elect_one
    commit：v17a 有同款先例，但 batch_atoms 过大可能编译慢/代码膨胀，
    可 --batch-atoms 32 降档。
8.  同一 TMEM 累加器被所有原子 ACCUMULATE=True 链式写：按标准 GEMM 主循环
    语义应满速流水；若怀疑同累加器限速, 用 --dual-acc 1 交替两个累加器
    （第二累加器列偏移 = N, 沿用 v17a 的 spacing=N_tile 约定; 2N<=512 已断言）。
9.  计时含 prologue（tmem alloc + 零填 + cluster 同步, ~几 µs 常数），已用
    idle kernel 差分扣除; ns/atom(net) 为扣除后值。commit 本身每批 1 次,
    摊到每原子 1/batch_atoms, 视为协议成本的一部分, 不另外扣。
10. N=16 的 CG2 SS 形状合法性以任务书为准（N%16 全合法）；若个别 N 编译被
    IR 校验拒绝, host 按 N 粒度 try/except, 继续扫其余点。

--------------------------------------------------------------------------------
t2r 背景负载档追加（none/smem 已 B200 验证通过后新增, 未经编译验证）
--------------------------------------------------------------------------------
设计要点: t2r 档 threads_per_cta=256; warp0=发射, warps4-7=T2R warpgroup
    (dp_idx = tidx-128 ∈ [0,128)), warps1-3 全程 idle 直落收尾等待。目标
    TMEM 列区 [2N, 2N+64)（恒避开双累加器区 [0,2N); __init__ 断言
    2N+64<=512）, (128DP x 64col) f32 视图。T2R 机器逐件抄 v5 现役形态:
    Ld16x256bOp(Repetition(2)) + make_tmem_copy + get_slice(dp_idx) +
    partition_S/partition_D + make_rmem_tensor + copy +
    fence_view_async_tmem_load（v5 教训#15: 几何只在原子族内派生;
    TMEM 指针禁 align()）。退出协议与 smem 档相同（try_wait 轮询 done mbar,
    每轮 >= 一次完整 128DPx64col 读）。
R11. TMEM 视图 layout 为手写 (128,64):(2^16,1)：lane 步长 2^16 取自 v5
     zone3 phys_layout 注释（"TMEM-ENCODED strides (lane stride 2^16)"),
     列步长 1 与现存 dual-acc `tmem_ptr + N` 列偏移约定一致, DSL
     find_tmem_tensor_col_offset 的 0xFFFF 列掩码亦印证。若 make_tmem_copy
     拒收平铺 (128,64) 形状或分区错位: 回退为 v5 同款嵌套 ((64,2),64)
     regroup, 或改从 make_fragment_C 布局同族派生。
R12. T2R 读值即弃, 存在 tcgen05.ld 被编译器/ptxas 消除的理论风险。已加
     防 DCE 锚: `if num_batches == 0`（host 断言 nb>=1, 运行时永不触发,
     谓词不可证伪）内把全部 64 个 rmem 元素存回 SMEM => 所有目的寄存器
     保活。若 SASS 复核仍见 ld 被删: 改为真实累加 sink + 循环外单次落存。
R13. warp-DP 象限假设: warps4-7 构成对齐 warpgroup, 组内 warp w 只可访问
     DP lanes [32w, 32w+32)（tcgen05.ld 硬件限制）, 与 get_slice(tidx-128)
     的映射一致。若 make_tmem_copy 线程枚举与该假设冲突会在编译/运行期报
     非法 TMEM 访问: 届时改用 4 组 Ld32x32bOp 每 warp 自管 32DP 的降档。
R14. t2r 档 N=256 越界(2*256+64>512)属预期: __init__ assert 在 host 抛出,
     主循环 try/except 记 ERROR 行继续扫其余 N; 若必须测 N=256, 需缩列宽
     或单累加器模式下改列偏移, 本档不做。
R15. 目标列区从未被写, T2R 读到未初始化 TMEM（可能含 NaN 位型）: 值不参与
     任何计算/比较, tcgen05.ld 本身不触发 FP 异常, 无害。寄存器开销 +64
     f32/线程(仅 warps4-7), 无 setmaxnreg（约束#1）, 单 SM 单 CTA 常驻,
     预计不构成占用瓶颈; 若 ptxas 报寄存器超限: Repetition 降为 x1 并把
     列宽减半(32col)分两轮读。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

MBARRIER_MAX_ARRIVALS = 1_000_000  # 硬件 2^20-1, 留余量


@dsl_user_op
def _mbarrier_try_wait(
    barrier: cute.Pointer,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Boolean:
    """Poll one CTA-shared mbarrier generation without blocking a role.
    (verbatim copy of the hardware-proven helper in dsa_bwd_sm100_2cta_v5)"""

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


def gpu_arch_flag() -> str:
    """--gpu-arch 值（tcgen05 需要 arch-specific 后缀, 参照 DSA compiler.py）。"""
    cap = torch.cuda.get_device_capability()
    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_100f"}.get(cap)
    if arch is None:
        raise RuntimeError(f"unsupported compute capability {cap}: need sm_100 家族")
    return arch


class MmaAtomPriceBench:
    """单 (N, chunk 模式, idle) 组合的 CG2 MMA 原子单价 kernel。

    编译期参数全部走 self 属性（cute.compile 按实例特化）；运行时仅
    num_batches: Int32 与 stream。
    """

    THREADS_PER_CTA = 32          # 遗留字面(勿用); 实际线程数见 self.threads_per_cta
    TMEM_COLUMNS = 512            # 一次性顶格分配, 省去逐 N 的合法性分档
    T2R_DP = 128                  # t2r 档: T2R 视图 datapath 数(整 warpgroup)
    T2R_COLS = 64                 # t2r 档: T2R 视图列数(f32: 1 元素 = 1 列)
    T2R_LANE_STRIDE = 1 << 16     # TMEM 编码 DP-lane 步长(v5 phys_layout 同款)
    CLUSTER_SHAPE_MNK = (2, 1, 1)
    CLUSTER_MASK = 3              # tcgen05.commit 多播到 cluster 内两个 CTA
    INSTR_M = 128                 # CG2 只测 M128
    INSTR_K = 16                  # f16/bf16 kind 固定 K16

    def __init__(
        self,
        instr_n: int,
        batch_atoms: int = 64,
        k_chunks: int = 8,
        rotate: bool = True,
        idle: bool = False,
        dual_acc: bool = False,
        num_clusters: int = 1,
        load_mode: str = "none",
        t2r_target: str = "far",
        issuers: int = 1,
    ):
        assert instr_n % 16 == 0 and 16 <= instr_n <= 256, f"bad N={instr_n}"
        assert batch_atoms >= 1 and k_chunks >= 1 and num_clusters >= 1
        # 背景负载: smem = 双 CTA 的 warp1-3 对操作数区打 STS 风暴 (值恒 0,
        # 与 MMA 读同 bank 竞争), 模拟内核里 TMA/pds/relay 对 CG2 跨 CTA
        # 取数的压迫; t2r = 双 CTA 的 warps4-7 整 warpgroup 对 TMEM 空闲
        # 列区持续 tcgen05.ld (T2R), 复现内核里 math/reduce 读流量与 TC
        # 累加器写全程并发的 TMEM 端口争抢; none = 原始安静基线。
        assert load_mode in ("none", "smem", "t2r", "fma"), f"bad load_mode={load_mode}"
        assert t2r_target in ("far", "near", "acc"), f"bad t2r_target={t2r_target}"
        # F0 探针: 双发射 warp——warp0/warp1 各自向不相交累加器并发入队,
        # 判 TC 前端跨 warp 入队是串行还是并行(V10 双发射的归零风险)。
        assert issuers in (1, 2), f"bad issuers={issuers}"
        assert load_mode == "none" or issuers == 1, "issuers=2 仅 none 档"
        if issuers == 2:
            assert 2 * instr_n <= self.TMEM_COLUMNS, "双发射需双累加器入 512 列"
        self.issuers = issuers
        self.t2r_target = t2r_target
        self.load_mode = load_mode
        self.threads_per_cta = {"none": 32 * issuers, "smem": 128, "t2r": 256, "fma": 256}[load_mode]
        if load_mode == "t2r":
            # 目标列区 [2N, 2N+64): 恒避开双累加器区 [0, 2N), 不越 512 列。
            assert (2 * instr_n if t2r_target == "far" else instr_n if t2r_target == "near" else 0) + self.T2R_COLS <= self.TMEM_COLUMNS, (
                f"t2r 列区 [{2 * instr_n}, {2 * instr_n + self.T2R_COLS}) "
                f"越界 TMEM {self.TMEM_COLUMNS} 列 (N={instr_n} 过大)"
            )
        if dual_acc:
            assert 2 * instr_n <= self.TMEM_COLUMNS, "dual-acc 放不进 512 列 TMEM"
        self.instr_n = instr_n
        self.batch_atoms = batch_atoms
        self.k_chunks = k_chunks
        self.rotate = rotate
        self.idle = idle
        self.dual_acc = dual_acc
        self.num_clusters = num_clusters
        self.element_dtype = BFloat16
        self.acc_dtype = Float32
        # 纯 python 预生成每个展开位的 chunk / 累加器编号, 避免 staged 三目分支
        self.chunk_ids = [
            (i % k_chunks) if rotate else 0 for i in range(batch_atoms)
        ]
        self.acc_ids = [(i % 2) if dual_acc else 0 for i in range(batch_atoms)]
        self.shared_storage = None
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

    # ------------------------------------------------------------------
    # 静态构造（host trace 与 kernel trace 各调一次, 输入相同 => 结构一致, 见 R4）
    # ------------------------------------------------------------------
    def _make_tiled_mma(self):
        # make_trivial_tiled_mma 内部即 MmaF16BF16Op(
        #     ab_dtype, acc_dtype, (*mma_tiler_mn, 16)=instruction_shape,
        #     CtaGroup.TWO, OperandSource.SMEM(默认), K-major, K-major)
        return sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.TWO,
            (self.INSTR_M, self.instr_n),
        )

    def _operand_layouts(self, tiled_mma):
        mma_tiler = (self.INSTR_M, self.instr_n, self.INSTR_K)
        a_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, self.element_dtype, self.k_chunks
        )
        b_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, self.element_dtype, self.k_chunks
        )
        return a_layout_staged, b_layout_staged

    # ------------------------------------------------------------------
    # host 入口
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, num_batches: Int32, stream: cuda.CUstream):
        tiled_mma = self._make_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        assert atom_thr_size == self.CLUSTER_SHAPE_MNK[0]  # CG2 => 2

        a_layout_staged, b_layout_staged = self._operand_layouts(tiled_mma)
        cosize_a = cute.cosize(a_layout_staged)  # per-CTA: M/2 x K16 x k_chunks
        cosize_b = cute.cosize(b_layout_staged)  # per-CTA: N/2 x K16 x k_chunks

        @cute.struct
        class SharedStorage:
            done_mbars: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64
            sA: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cosize_a],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cosize_b],
                1024,
            ]

        self.shared_storage = SharedStorage

        self.kernel(num_batches).launch(
            grid=(
                self.CLUSTER_SHAPE_MNK[0] * self.num_clusters,
                1,
                1,
            ),
            block=[self.threads_per_cta, 1, 1],
            cluster=self.CLUSTER_SHAPE_MNK,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(self, num_batches: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = rank == 0

        # ---- 静态重建（与 host 侧一致, R4） ----
        tiled_mma = self._make_tiled_mma()
        a_layout_staged, b_layout_staged = self._operand_layouts(tiled_mma)
        cosize_a = cute.cosize(a_layout_staged)
        cosize_b = cute.cosize(b_layout_staged)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        done_mbar = storage.done_mbars.data_ptr()

        # ---- MMA 操作数 SMEM 张量（含 swizzle） ----
        if cutlass.const_expr(isinstance(a_layout_staged, cute.ComposedLayout)):
            s_a = storage.sA.get_tensor(
                a_layout_staged.outer, swizzle=a_layout_staged.inner
            )
        else:
            s_a = storage.sA.get_tensor(a_layout_staged)
        if cutlass.const_expr(isinstance(b_layout_staged, cute.ComposedLayout)):
            s_b = storage.sB.get_tensor(
                b_layout_staged.outer, swizzle=b_layout_staged.inner
            )
        else:
            s_b = storage.sB.get_tensor(b_layout_staged)

        # ---- TMEM（v17a 同构; ctor 内自带 dealloc mbar 的 init+fence） ----
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # ---- done mbarrier: 聚合到达计数 = num_batches（R1） ----
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(done_mbar, num_batches * Int32(self.issuers))
        cute.arch.mbarrier_init_fence()

        tmem.allocate(self.TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tmem.relinquish_alloc_permit()

        # ---- 零填两侧操作数（吞吐 bench, 内容只求非 NaN/Inf 干扰之外的确定性） ----
        a_flat = storage.sA.get_tensor(cute.make_layout((cosize_a,)))
        for element in cutlass.range(tidx, cosize_a, self.threads_per_cta):
            a_flat[element] = self.element_dtype(0.0)
        b_flat = storage.sB.get_tensor(cute.make_layout((cosize_b,)))
        for element in cutlass.range(tidx, cosize_b, self.threads_per_cta):
            b_flat[element] = self.element_dtype(0.0)
        cute.arch.fence_view_async_shared()  # 泛型写 -> async proxy 可见
        cute.arch.barrier()
        cute.arch.cluster_arrive()           # 两 CTA 的 SMEM/mbar 就绪后才允许发射
        cute.arch.cluster_wait()

        # ---- MMA 操作数 fragment 与 TMEM 累加器 ----
        a_fragment = tiled_mma.make_fragment_A(s_a)   # ((MMA), MMA_M, MMA_K, STAGE)
        b_fragment = tiled_mma.make_fragment_B(s_b)
        assert cute.size(a_fragment, mode=[2]) == 1          # tiler K == instr K
        assert cute.size(a_fragment, mode=[3]) == self.k_chunks
        assert cute.size(b_fragment, mode=[3]) == self.k_chunks

        acc_shape = tiled_mma.partition_shape_C((self.INSTR_M, self.instr_n))
        acc_layout = tiled_mma.make_fragment_C(acc_shape).layout
        # 双累加器: 第二个的列偏移沿用 v17a spacing = N_tile 约定
        t_acc = (
            cute.make_tensor(tmem_ptr, acc_layout),
            cute.make_tensor(tmem_ptr + self.instr_n, acc_layout),
        )

        # ---- 发射循环: 仅 leader CTA warp0（cute.gemm 内部自带 elect_one） ----
        if is_leader_cta and warp_idx < self.issuers:
            mma = tiled_mma.with_()
            if cutlass.const_expr(not self.idle):
                # priming: ACCUMULATE=False 覆写 TMEM 垃圾, 之后恒 True。
                # 多出的 1~2 个原子计入首批 commit, 量级 1/65536, 忽略。
                mma.set(tcgen05.Field.ACCUMULATE, False)
                acc_prime = t_acc[0]
                if cutlass.const_expr(self.issuers == 2):
                    if warp_idx == 1:
                        acc_prime = t_acc[1]
                cute.gemm(
                    mma,
                    acc_prime,
                    a_fragment[None, None, 0, 0],
                    b_fragment[None, None, 0, 0],
                    acc_prime,
                )
                if cutlass.const_expr(self.dual_acc):
                    cute.gemm(
                        mma,
                        t_acc[1],
                        a_fragment[None, None, 0, 0],
                        b_fragment[None, None, 0, 0],
                        t_acc[1],
                    )
                mma.set(tcgen05.Field.ACCUMULATE, True)
            for _batch in cutlass.range(num_batches):
                if cutlass.const_expr(not self.idle):
                    for i in cutlass.range_constexpr(self.batch_atoms):
                        acc = t_acc[self.acc_ids[i]]
                        if cutlass.const_expr(self.issuers == 2):
                            acc = t_acc[0]
                            if warp_idx == 1:
                                acc = t_acc[1]
                        cute.gemm(
                            mma,
                            acc,
                            a_fragment[None, None, 0, self.chunk_ids[i]],
                            b_fragment[None, None, 0, self.chunk_ids[i]],
                            acc,
                        )
                with cute.arch.elect_one():
                    tcgen05.commit(
                        done_mbar,
                        self.CLUSTER_MASK,
                        tcgen05.CtaGroup.TWO,
                    )

        # ---- 背景负载: 双 CTA 的 warp1-3 对操作数区打 STS 风暴 ----
        # 值恒 0.0 (操作数本就全零) => 不改 MMA 结果; 目标是与 UMMA 的
        # SMEM 操作数取数 (含 CG2 跨 CTA B 读) 竞争 bank/端口。
        # 退出: 轮询与发射端同一 done mbar (parity 0), 完成后落回主收尾。
        if cutlass.const_expr(self.load_mode == "smem"):
            if warp_idx > 0:
                lane = tidx - Int32(32)
                offset = Int32(0)
                storm_done = cutlass.Boolean(False)
                while not storm_done:
                    for i in cutlass.range_constexpr(32):
                        a_flat[
                            (lane * Int32(37) + offset + Int32(i * 96))
                            % Int32(cosize_a)
                        ] = self.element_dtype(0.0)
                        b_flat[
                            (lane * Int32(37) + offset + Int32(i * 96))
                            % Int32(cosize_b)
                        ] = self.element_dtype(0.0)
                    offset = offset + Int32(1)
                    storm_done = _mbarrier_try_wait(done_mbar, Int32(0))

        # ---- 背景负载: warps1-7 稠密 FMA 自旋（发射位稀释探针） ----
        # 复现内核 pass-1 现场: math/reduce 共驻 warp 满负荷执行 ALU 流,
        # 与 leader 的 gemm 入队指令争抢 SM warp 调度器发射位。纯寄存器
        # FMA 链（每轮 128 条, 8 路独立累加链保 ILP), 不触 SMEM/TMEM/L2,
        # 与 smem/t2r 档正交。退出协议同 smem 档。
        if cutlass.const_expr(self.load_mode == "fma"):
            if warp_idx > 0:
                acc0 = Float32(1.0) + Float32(tidx) * Float32(1e-6)
                acc1 = Float32(1.1); acc2 = Float32(1.2); acc3 = Float32(1.3)
                acc4 = Float32(1.4); acc5 = Float32(1.5); acc6 = Float32(1.6)
                acc7 = Float32(1.7)
                storm_done = cutlass.Boolean(False)
                while not storm_done:
                    for _spin in cutlass.range_constexpr(16):
                        acc0 = acc0 * Float32(1.0000001) + Float32(1e-7)
                        acc1 = acc1 * Float32(1.0000002) + Float32(1e-7)
                        acc2 = acc2 * Float32(1.0000003) + Float32(1e-7)
                        acc3 = acc3 * Float32(1.0000004) + Float32(1e-7)
                        acc4 = acc4 * Float32(1.0000005) + Float32(1e-7)
                        acc5 = acc5 * Float32(1.0000006) + Float32(1e-7)
                        acc6 = acc6 * Float32(1.0000007) + Float32(1e-7)
                        acc7 = acc7 * Float32(1.0000008) + Float32(1e-7)
                    storm_done = _mbarrier_try_wait(done_mbar, Int32(0))
                # 防 DCE 锚: 不可证伪谓词内落存(运行时永不触发)。
                if num_batches == Int32(0):
                    a_flat[Int32(0)] = self.element_dtype(
                        acc0 + acc1 + acc2 + acc3
                        + acc4 + acc5 + acc6 + acc7
                    )

        # ---- 背景负载: 双 CTA 的 warps4-7 整 warpgroup 持续 T2R 读 TMEM ----
        # 复现内核结构: math/reduce 的 tcgen05.ld 读流量与 TC 累加器写全程
        # 并发（TMEM 端口争抢主嫌探针）。目标列区 [2N, 2N+64) 恒避开双累加
        # 器区; (128DP x 64col) f32 视图, DP-lane 编码步长 2^16（v5 zone3
        # phys_layout 同款; DSL find_tmem_tensor_col_offset 的 0xFFFF 列掩
        # 码印证列域在低 16 位, f32 一元素 = 一列）。T2R 机器逐件抄 v5 现役
        # 形态（教训#15: 几何只在 Ld16x256b 原子族内派生; TMEM 指针不 align）。
        # warp1-3 此档全程 idle, 直接落到收尾 mbarrier_wait。
        # 退出协议与 smem 档相同: 每轮一次完整拷贝 + fence 后轮询 done mbar。
        if cutlass.const_expr(self.load_mode == "t2r"):
            if warp_idx >= 4:
                dp_idx = tidx - Int32(128)  # warpgroup 内相对 DP 索引 0..127
                t_t2r = cute.make_tensor(
                    tmem_ptr + {"far": 2 * self.instr_n, "near": self.instr_n, "acc": 0}[
                        self.t2r_target
                    ],
                    cute.make_layout(
                        (self.T2R_DP, self.T2R_COLS),
                        stride=(self.T2R_LANE_STRIDE, 1),
                    ),
                )
                t2r_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(2)),
                    self.acc_dtype,
                )
                tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, t_t2r)
                thread_t2r = tiled_t2r.get_slice(dp_idx)
                t2r_source = thread_t2r.partition_S(t_t2r)
                t2r_coordinates = thread_t2r.partition_D(
                    cute.make_identity_tensor((self.T2R_DP, self.T2R_COLS))
                )
                r_t2r = cute.make_rmem_tensor(
                    t2r_coordinates.shape, self.acc_dtype
                )
                # 128DP x 64col / 128 线程 = 每线程 64 f32（漂移即编译期报）
                assert cute.size(r_t2r) == (
                    self.T2R_DP * self.T2R_COLS // 128
                ), str(t2r_coordinates.shape)
                storm_done = cutlass.Boolean(False)
                while not storm_done:
                    cute.copy(tiled_t2r, t2r_source, r_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    # 防 DCE 锚: host 已断言 num_batches >= 1, 此分支运行时
                    # 永不触发; 但谓词对编译器不可证伪 => 全部 T2R 目的寄存
                    # 器保活, tcgen05.ld 不可被消除。读到的值即弃（不参与
                    # 任何计算/计时路径）。
                    if num_batches == Int32(0):
                        for i in cutlass.range_constexpr(cute.size(r_t2r)):
                            a_flat[Int32(i)] = self.element_dtype(r_t2r[i])
                    storm_done = _mbarrier_try_wait(done_mbar, Int32(0))

        # ---- 收尾: 两 CTA 都等自己本地 mbar（多播到达）, 再 cluster 同步 ----
        cute.arch.mbarrier_wait(done_mbar, Int32(0))
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.free(tmem_ptr)


# ======================================================================
# host 侧: 编译 / 校准 / 计时 / 报表
# ======================================================================


def resolve_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def time_once(compiled, num_batches: int, stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    compiled(num_batches, stream)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def time_median(compiled, num_batches: int, stream, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        compiled(num_batches, stream)
    torch.cuda.synchronize()
    return statistics.median(
        time_once(compiled, num_batches, stream) for _ in range(reps)
    )


def calibrate_batches(compiled, init_batches: int, stream, target_ms: float) -> int:
    """放大 num_batches 直到单 kernel >= target_ms（运行时实参, 无需重编译）。"""
    nb = init_batches
    compiled(nb, stream)  # 冷启动
    torch.cuda.synchronize()
    t = time_once(compiled, nb, stream)
    while t < target_ms and nb < MBARRIER_MAX_ARRIVALS:
        nb = min(int(nb * max(target_ms / max(t, 1e-3), 1.0) * 1.25) + 1,
                 MBARRIER_MAX_ARRIVALS)
        t = time_once(compiled, nb, stream)
    return nb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tcgen05 CG2 bf16 MMA atom price bench")
    p.add_argument("--n-list", default="16,32,64,128,256",
                   help="instruction N 扫描列表")
    p.add_argument("--mode", choices=["fixed", "rotate", "both"], default="both",
                   help="fixed=同一 chunk 反复发射(纯发射地板); "
                        "rotate=8 chunk 轮转(描述符步进); both=两者都测")
    p.add_argument("--atoms", type=int, default=65536,
                   help="初始目标原子数(会按 --target-ms 自动放大)")
    p.add_argument("--target-ms", type=float, default=1.0,
                   help="单 kernel 最短时长(校准下限)")
    p.add_argument("--batch-atoms", type=int, default=64,
                   help="每次 tcgen05.commit 前展开发射的原子数")
    p.add_argument("--k-chunks", type=int, default=8,
                   help="SMEM K 深度 chunk 数(rotate 模式轮转其间)")
    p.add_argument("--dual-acc", type=int, choices=[0, 1], default=0,
                   help="1=两个 TMEM 累加器交替(排除同累加器链)")
    p.add_argument("--load-mode",
                   choices=["none", "smem", "t2r", "fma", "both", "all"],
                   default="none",
                   )
    p.add_argument("--issuers", type=int, choices=[1, 2], default=1,
                   help="F0 探针: 并发发射 warp 数(2=warp0/1 各自向不相交"
                        "累加器入队, 判 TC 前端跨 warp 串行性)")
    p.add_argument("--t2r-target", choices=["far", "near", "acc"],
                   default="far",
                   help="t2r 档目标列区: far=[2N,2N+64) 远列(原状); "
                        "near=[N,N+64) 邻列; acc=[0,64) 直读活跃累加器区")
    p.add_argument("--clusters", type=int, default=1,
                   help="并发 cluster 数(>1 仅作时间不变性 sanity check; "
                        "计时/原子数仍按单 cluster 报)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--json", default=None, help="可选: 结果 JSON 输出路径")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_list = [int(x) for x in args.n_list.split(",") if x]
    modes = ["fixed", "rotate"] if args.mode == "both" else [args.mode]

    torch.zeros(1, device="cuda")  # 初始化 CUDA 上下文
    stream = resolve_stream()
    arch = gpu_arch_flag()
    print(f"device={torch.cuda.get_device_name()} arch={arch} "
          f"batch_atoms={args.batch_atoms} k_chunks={args.k_chunks} "
          f"dual_acc={args.dual_acc} clusters={args.clusters} "
          f"target_ms={args.target_ms} reps={args.reps}")
    print("提示: 未锁频则 ns/atom 会随 boost 漂移 (nvidia-smi -lgc)。")

    # ---- idle 对照: 与 N 无关, 只编一份(N=128 占位) ----
    idle_bench = MmaAtomPriceBench(
        instr_n=128, batch_atoms=args.batch_atoms, k_chunks=args.k_chunks,
        rotate=False, idle=True, dual_acc=False, num_clusters=args.clusters,
    )
    idle_compiled = cute.compile(
        idle_bench, Int32(1), stream, options=f"--gpu-arch {arch}"
    )

    header = (f"{'mode':<7} {'N':<4} {'atoms':<10} {'batches':<8} "
              f"{'total_ms':<9} {'idle_ms':<8} {'ns/atom':<9} "
              f"{'ns/atom.net':<12} {'ns/N16.net':<10} {'TFLOPS':<8}")
    print(header)
    print("-" * len(header))

    if args.load_mode == "both":
        load_modes = ["none", "smem"]
    elif args.load_mode == "all":
        load_modes = ["none", "smem", "t2r", "fma"]
    else:
        load_modes = [args.load_mode]
    rows = []
    for load_mode in load_modes:
      for mode in modes:
        for n in n_list:
            row = {"mode": mode, "N": n, "batch_atoms": args.batch_atoms,
                   "k_chunks": args.k_chunks, "dual_acc": args.dual_acc,
                   "load_mode": load_mode, "t2r_target": args.t2r_target, "issuers": args.issuers}
            try:
                bench = MmaAtomPriceBench(
                    instr_n=n,
                    batch_atoms=args.batch_atoms,
                    k_chunks=args.k_chunks,
                    rotate=(mode == "rotate"),
                    idle=False,
                    dual_acc=bool(args.dual_acc),
                    num_clusters=args.clusters,
                    load_mode=load_mode,
                    t2r_target=args.t2r_target,
                    issuers=args.issuers,
                )
                compiled = cute.compile(
                    bench, Int32(1), stream, options=f"--gpu-arch {arch}"
                )
                init_nb = max(1, math.ceil(args.atoms / args.batch_atoms))
                nb = calibrate_batches(compiled, init_nb, stream, args.target_ms)
                assert 1 <= nb <= MBARRIER_MAX_ARRIVALS
                atoms = nb * args.batch_atoms * args.issuers

                total_ms = time_median(
                    compiled, nb, stream, args.warmup, args.reps
                )
                idle_ms = time_median(
                    idle_compiled, nb, stream, args.warmup, args.reps
                )

                ns_atom = total_ms * 1e6 / atoms
                ns_atom_net = max(total_ms - idle_ms, 0.0) * 1e6 / atoms
                ns_n16_net = ns_atom_net / (n / 16.0)
                flops = atoms * 2 * 128 * n * 16  # 单 cluster (SM 对)
                tflops = flops / (total_ms * 1e-3) / 1e12

                row.update(
                    atoms=atoms, batches=nb,
                    total_ms=round(total_ms, 4), idle_ms=round(idle_ms, 4),
                    ns_per_atom=round(ns_atom, 4),
                    ns_per_atom_net=round(ns_atom_net, 4),
                    ns_per_n16_net=round(ns_n16_net, 4),
                    tflops=round(tflops, 2),
                )
                print(f"{mode:<7} {n:<4} {atoms:<10} {nb:<8} "
                      f"{total_ms:<9.4f} {idle_ms:<8.4f} {ns_atom:<9.4f} "
                      f"{ns_atom_net:<12.4f} {ns_n16_net:<10.4f} {tflops:<8.1f}")
            except Exception as e:  # 单点失败继续扫
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"{mode:<7} {n:<4} ERROR: {row['error']}")
            rows.append(row)

    print("MMA_ATOM_PRICE_JSON " + json.dumps(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"MMA_ATOM_PRICE_RESULT {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
