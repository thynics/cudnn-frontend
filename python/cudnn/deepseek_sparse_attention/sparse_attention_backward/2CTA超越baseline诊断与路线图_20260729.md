# 2-CTA 超越 baseline 可行性诊断与路线图（2026-07-29）

> 产生方式：16-agent 并行取证（v9.3 内核 / baseline 内核 / 全部 trace / commit 决策史 /
> PTX-DSL 约束核实）→ 定量诊断 → 五视角独立提案 → 逐案对抗验证。
> 取证基线：v8=11.945ms 实测 + v9.3=12.091ms 实测（/Users/longcheng/v10/）+
> v11（b3c7464..14657ec，待测）。诊断章节写于 v9.3 实测披露之前，其
> "三轨并列 pacer" 模型已被 v9.3 实测修正为 "drain 单独 pacer"——以附录各
> 判决章节的修正为准。结论摘要见文末"总路线图"。

# 定量诊断：v8/v9.2/v9.3 pacer 模型与理论下限

数据口径：148 SM、1 CTA/SM、4096 cluster÷74 并发 = 55.35 波；wall ≈ 55.35 × cluster 驻留。基准自洽性验证：baseline 8.090 ms ÷ 55.35 = 146.2 µs 驻留 → 4.57 µs/tile；TC 核对 5 GEMM × 2·64·64·512 = 21 MFLOP/CTA-tile ÷ 4.57 µs = 4.59 TFLOPS/SM × 148 = 679 TFLOPS，与 `/Users/longcheng/v9/baseline_performance.json` 的 679.52 精确吻合 → 换算模型可信。v8 = 11.945 ms（`/Users/longcheng/v8/validation_summary.json`, ratio 1.4817）→ 215.8 µs/cluster → **6.74 µs/tile 毛节拍，扣固定成本 40.1 µs 后净节拍 5.49 µs/tile**。

---

## 1. v8 的 6.74 µs/tile 分解：三轨拥挤 + 一个真环

**先立一个方法论事实**：单 cluster trace（IKET on）的绝对数系统性偏大。trace 节拍 11.14–11.74 µs vs 无 trace 毛节拍 6.74 µs（×1.7）；对照 baseline trace 节拍 4.61 vs 无 trace 4.57（×1.01，17 个 span 名 vs v8 的 26 个）。故 trace 只能用于**结构**（谁等谁、缝隙在哪），稳态绝对量必须用 wall 反推。这正是证据包里"稳态净节拍 5.5 < trace REDUCE 纯功 6.9"张力的解释：instrumentation 膨胀 + 全网格暖 L2。

**trace 结构（i=2 稳态 tile，SVG 逐 rect）**：
- 显式串行链是 W17：ROUTE_K 17.12–18.88 → MAT_QDO r0 18.82–26.05 (7.23) → r1 25.89–28.67 (2.78)，缝隙 <0.15 µs，占满整个节拍。**但 r0/r1 字节相同耗时差 2.6×** → r0 是延迟吸收体（credit 等待 + TMA 往返），纯发射+publish ≈ 2×2.7 + ROUTE_K 纯份 ≈ **5.5 µs/tile**。
- REDUCE：WAIT_dK 5.12 µs 闲 → T2R 1.4 → ATOMIC r0 4.00 (1.9 ns/条) → r1 1.60 (**0.72–0.78 ns/条 ≈ baseline 效率**)，两 burst 重叠仅 0.6–0.8 µs。40% 闲——**在 trace 里 REDUCE 是被供给链 chain 住的，不是吞吐饱和**。
- MATH：MATH_PD 包络 10.5 µs 代际近串行；构成（span 表 `/Users/longcheng/v8/span_stats_with_waits.md:24-36`）：T2R 1.87 + SOFTMAX 3.29 + PDS_ACQ 2.22（显式等待）+ STORE 4.97（SASS 实锤 160 STS.U16 + PRMT 风暴，`summary.txt`）+ BAR1 0.95 + ROUTE_P/dS 1.29。
- MMA union 16.5%、issue 合计 ~1.6 µs（S_ISSUE 0.469 + dP 0.496 + dQ/dVdK 0.66）——从不是 pacer。gather LOAD_K 1.69——有大量 slack。

**稳态模型（净节拍 5.49）**：约束 32 × math_wall ≤ 175.7 µs（净驻留）⇒ math 真实 wall ≤ 5.49，同理 REDUCE、W17。三条轨在无 trace 稳态下各自压缩到 ~5–5.5 µs，**挤在同一节拍水位上**：
- REDUCE：span 均值口径 6.66 µs/tile（`two_trace_tables.md:21`），union 98.1%——稳态最接近饱和的角色；
- MATH：STORE ~5 µs（真功）+ softmax + T2R，包络贴节拍；
- W17：纯发射 ~5.5 µs 贴节拍。

**真环（把三轨锁成 convoy 的机制）**：pds 1 级闭环 `math STORE(t) → BAR1 → 单线程 DSM 4KiB×2 → W18 relay → leader relay wait (L13863) → leader grads(t) 头+尾 8 pass → pds release (L13311) → math(t+1) ACQ 解锁`；叠加 round 10 代链把 kdq(t) 拴在 leader 消费 g8/g9(t-1)（grads 尾第 3/4 pass）之后。环长 ≈ math 纯功 + DSM 往返 + relay + leader 梯度发射 ≈ 5–6 µs——**pds 1 级使节拍 = 环延迟而非 max(各轨)**。这解释了 v9 的两次单杠杆失败（改任何一轨,环不缩短）和 REDUCE 的 5.12 µs WAIT 缺口（burst 冷启动的来源）。

结论：**v8 稳态是 REDUCE(6.66 span)/MATH(≥5 真功)/W17(5.5 纯发射) 三轨并列 pacer，被 pds 单级环同步成 convoy；slack 全在 MMA/gather/W19。**

## 2. 逐墙修复后残余

**(a) math（stmatrix 修复 = v9.3 增量 a+b 之后）**
- 硬下限锚点：baseline 在 128 线程上做同量 T2R（2×32 f32/thr）+ 32 exp2 + FMA + quantize + 2 tensor stmatrix，全包 **0.977 µs**（span 表 :50,:62）。
- v9.3 相对 baseline 的结构性加项：第 3 份镜像（24 KiB vs 16 KiB STS，+~0.15）；grouped stats 8 次 SMEM load/thr（v8 fallback 64 次是 SOFTMAX 3.29 的主体，应压至 ~1–1.5）；2 次 128-thr math_barrier（v8 BAR1 0.95）；单线程 DSM arm+issue（ROUTE_P/dS 1.29）。
- 残余估计：**硬下限 ~2.0 µs；现实 3.5–4.5 µs**（若 T2R 1.87 不随 stage/fence 精简而压缩、softmax 只到 1.5，则 1.9+1.5+0.4+1.0+1.3 ≈ 6 的悲观上界仍存在）。验收门：SASS STSM>0、STORE span → <0.5 µs。

**(b) 供给链（理想解耦）**
- 带宽核对：G2S 每 tile 每 CTA = LOAD_K 32 KiB + kdq 32 KiB + QDO TMA 64 KiB = 128 KiB / 4.1 µs = 31 GB/s/SM（全部 L2 热：K cache 4 MB、Q/dO 128 KiB×148 常驻 L2），×148 ≈ 4.6 TB/s L2 读——离硬件顶远；**墙是软件发射/延迟，不是字节**（trace 的 r0/r1 2.6× 差已证明）。
- W17 纯协调 = ~30 个 elect_one pipeline 操作 ≈ 1–2 µs；r1 外推的每 round 纯发射+publish 2.7 µs。若两 round 真正重叠（更深 round 缓冲 / 解开与 leader g8/g9 的耦合）：**残余 ≈ 2.8–4 µs/tile，硬下限 ~1.5**。
- gather 侧 LOAD_K 1.69 + kdq fill（gather warps 有 slack）不构成约束。

**(c) reduce（最关键）**
- 发射吞吐锚点：baseline 2.203 µs/part ÷ 4096 red.v4 = **0.538 ns/条 ≈ 1 条/SM-clk（1.86 GHz）= LSU 发射界**；且 baseline 两 CTA 写**相同地址**都没破这个数 → L2 同址争用不是限制。GPU 级核对：148 SM × 16B/0.538ns ≈ 4.4 TB/s L2 RMW，workspace 8 MB 全 L2 常驻——可持续。
- 2-CTA 量：4096 条/tile/CTA（减半 + pair 内 D-quarter 不相交，比 baseline 还少一半 GPU 级原子流量）→ **发射下限 4096 × 0.538 = 2.20 µs/tile**。T2R 2×32 KiB TMEM：baseline 证明可与原子发射融合流水（store_dKV 模式），不加 wall。
- **v6 "~5 µs 固定成本" 判决：协议造成，非硬件必然**。三重证据：① v8 trace 内 r1 burst 已达 0.72–0.78 ns/条（=baseline 效率）——同一 kernel、同一硬件路径；② r0 的 1.9 ns/条超额 ≈ 2048×1.15ns ≈ 2.35 µs，恰是 5.12 µs WAIT_dK 空窗后管线排空的重灌成本；③ baseline reducer WAIT 仅 0.334 µs、流从不干涸，故从不付重灌费。v6 的 8.6 µs 周期 = 5.0 drain + 3.6 convoy，正是"每 burst 都冷启动"的形态。
- 残余：**2.2（发射）+ 一次/tile 重灌摊销 0.3–0.6 + T2R 未重叠残余 0.2–0.4 ≈ 2.5–3.2 µs/tile**。达成条件 = burst 连续（v9.4 交错折叠成单 4096 条 burst + WAIT_dK→0，即 leader slot commit 早于 reducer 排空）。

## 3. 当前数据面的每 tile 硬下限

| 分量 | 下限 | 现实修复后 | 依据 |
|---|---|---|---|
| TC busy | ~1.5–1.8 µs | 同 | baseline ~2 µs（M64 半速 S/dP），CG2 全宽减半 S/dP 份 |
| leader 发射 | ~2.6 µs | 2.6–3.5 | 104 atom × 14.6 ns（S_ISSUE 实测折算）+ ~30 pipeline 操作 |
| reduce | 2.2 | 2.5–3.2 | §2c |
| math | 2.0 | 3.5–4.5 | §2a |
| W17 供给 | 1.5 | 2.8–4.0 | §2b |
| 队列流量（L2/SMEM/async proxy） | 均 <25% 占用 | — | §2b 带宽核对 |

**若完全解耦（节拍 = max）：诚实区间 3.2–4.8 µs/tile**；乐观 3.4–3.5（reduce pacer），中性 4.2–4.5（math/W17 残余 pacer）。对照目标：追平线 = (146.2 − F)/32；F=15 µs 时 **4.10 µs**，F=40（现状）时 3.32——**固定成本不修，节拍即使打到硬下限也追不平**（这正是证据包 3.31 µs 数字的含义）。
**若 pds 环不解耦（节拍 = 环长）：环 = math 3.5 + DSM/relay ~0.8 + leader grads ~1.5 ≈ 5.5–6 µs > 4.1 → 必输。**

## 4. per-token 固定成本

现状（span 表）：v8 = TAIL 24.48 + DQ_EPI 14.30 + LOAD_QDO 0.77 + LOAD_STATS 0.54 ≈ **40.1 µs** vs baseline ≈ 8.2 µs → 摊销差 **+1.0 µs/tile**（占追平预算 4.57 的 22%）；v9 更差（53.3，v9_final 表 :22,:37）。可压缩路径：
- **dQ TMA epilogue**：v8 是 2×7.15 µs 全串行 256 条标量 STG/thr（tma_atom_dq_epi 在 L11859 被显式丢弃）。改 T2R→R2S→TMA（baseline 4 panel 串行做到 7.23；v2 dqepitma 实测 wall −2.2%）→ 2 round 流水 ≈ 5–6 µs，且可提前与末 2 个 tile 重叠 → 可见 ~3–4 µs。
- **TAIL 24.5 → ~8–12**：TAIL 的主体是末 tile grads 块 + drain 在无下一 tile 掩护下裸奔；EPI 前移重叠 + producer_tail 合并可砍半。
- LOAD 1.3 基本不可压。
- 结果：**现实 F ≈ 12–18 µs（摊销 0.4–0.56/tile，vs baseline 0.26）**；persistent cluster（多 token/cluster，藏 TMEM alloc/QDO 于前一 token 尾部）可再压至 ~8–10，但收益仅 ~0.15 µs/tile，且新增一个调度环——按教训 #6（环外工作最便宜）风险大于收益，列为可选。

## 5. 结论

**可达区间**（wall = 55.35 × (32c + F)）：
- **乐观 6.9 ms（0.85×）**：c=3.5（reduce 2.8 为 pacer，全解耦）、F=12。全部 5 个条件同时落地。
- **中性 8.5–8.7 ms（1.05–1.08×）**：c=4.4（math 或 W17 残余顶节拍，pds 半解耦）、F=16。**略输 baseline**。
- **悲观 10.4–10.9 ms（1.3×）**：仅 v9.3 的 store+drain 修复落地、pds 环与 F 不动（c≈4.7、F≈40）——与 07e0838 commit 预期 8.5–10.5 的上端吻合。

**要 <8.09 ms 必须同时满足（按必要性排序）**：
1. **pds 环解耦，使节拍 = max 而非环长**——唯一在数学上无替代的条件（环长下限 ~5.5 > 4.1）。手段：P/dS 出口双缓冲。**硬障碍：SMEM slack 仅 1.5 KiB**（SharedStorageV2 合计 230,912/232,448），双份 blocks+xchg 需 +24 KiB；资金来源只能是镜像合并（ds_image 8 KiB 与 ds_blocks/xchg 统一布局）或 round buffer 重组。注意 A-in-TMEM 逃逸口对 dV/dK 无效（需 P^T/dS^T 作 A，TMEM-A 恒 K-major 不可转置；产出 S^T 需 M=64 的 CG2 TS atom，不存在）——SMEM publish 在当前数据面不可删，T2 定理的 16 KiB DSM 已在下界。
2. **reduce 连续 burst ≤3.2 µs/tile**：v9.4 折叠 + WAIT_dK→0（leader slot commit 提前于 reducer 排空）。可达性证据在 kernel 内部（r1 = 0.72 ns/条）。
3. **math ≤4.5 µs**：v9.3 stmatrix（SASS 门：STSM>0、PRMT 清零）+ grouped softmax 兑现，且 T2R/barrier 不反弹。
4. **F ≤15 µs**：dQ TMA epilogue + TAIL/EPI 重叠。不做则追平线降到 3.3 µs，低于现实下限。
5. **W17 双 round 重叠 ≤4 µs**：保持 v8 kdq 卸载，加深 round 缓冲或解开 g8/g9 反向耦合（条件 1 的副产品）。

**诚实总评**：修完全部已知墙后 2-CTA 的理论最好水位 ≈ 0.85× baseline（~6.9 ms），上限薄的根因是账目对称性：2-CTA 从 baseline 手里拿走的只有 reduce 减半（+2.2 µs/tile 理论额度）与 gather 减半（baseline 处 K load 本就不是墙），却新付出 DSM 交换、3 份镜像、relay、round 供给、双倍固定成本。中性情形落在 1.05×——**胜负完全取决于条件 1（pds 双缓冲/去环）能否在 1.5 KiB SMEM slack 下实现**；条件 2/3 有 kernel 内实测效率背书，属工程兑现问题。下一步最高信息量动作：先实测 v9.3（同时检验条件 2、3 与"三轨并列"模型——若落在 9.5–10.5 区间则模型成立，pds 环即为最后主墙）。

关键引用：`/Users/longcheng/v8/span_stats_with_waits.md:21-44,50-64`；`/Users/longcheng/v8/two_trace_tables.md:19,21,35-37`；`/Users/longcheng/v8/validation_summary.json`（11.94521/8.061792）；`/Users/longcheng/v8/summary.txt`（math_sts=160, STSM=0）；`/Users/longcheng/v9_final/two_trace_tables.md:21,35,37`；`/Users/longcheng/v9/candidate_performance.json`（24.630246）；内核 `KDIR/dsa_bwd_sm100_2cta_v9_3.py` L12308(pds 1 级)、L13311(release 位点)、L13863(relay wait)、L14258-14366(v9.4 drain)、L11313-11379(SMEM 预算)、L11859(dQ TMA 弃用)。

---

# 附录：reduce/atomic 视角 提案对抗验证判决

# 对抗性验证报告：reduce 视角三提案 + 否决清单

## 先决事实（核查中发现，推翻部分前提）

**F1. 提案审查的对象已过时：HEAD 不是 v9_3，是 v11（commit b3c7464）。** git log 显示提案未见的两个后续 commit：`e82fb19`（v9.4 drain fold 回退，证实提案勘误）和 `b3c7464`（"v11: reducer de-spill (the drain paradox, located)"，新文件 `KDIR/dsa_bwd_sm100_2cta_v11.py`，14355 行）。

**F2. v9.3 已实测，结果在 /Users/longcheng/v10/**（提案执行序第①步已完成）：`validation_summary.json` = candidate **12.091 ms** vs baseline 7.995 ms（**1.512x，比 v8 的 11.945 还差**）。且 stmatrix 修复确认落地：`stage0_user_gate_summary.md` publish 窗 STSM=20、PRMT=32、STS.U16=0；trace 中 MATH_STORE 4.97→~1.1-1.5µs、MATH_PD 10.14→3.49µs（`stage2_report.md` i=1）。**math 侧砍掉 ~6µs 包络后 wall 纹丝不动 → 诊断的"三轨并列 pacer"模型被证伪**（诊断自己的判据"落在 9.5-10.5 则模型成立"未满足）。v9.3 trace 里 drain 独占节拍：REDUCE_ATOMIC r0 4.0-4.2 + r1 2.75，drain 忙链 ≈ 8.1µs ≈ trace 周期 8.6µs（`stage2_report.md` i=1..3 表）。

**F3. r0/r1 2.6x 差的真机制已被 SASS 定位为寄存器溢出，不是"管线重灌"。** `/Users/longcheng/v10/drain_t2r_through_last_atomic.sass`：16 条 REDG.E.ADD.F32x4 窗内混有 **65 LDL + 28 STL**，且全部 REDG 使用同一寄存器组（数据 R4-R7、地址 R8.64）——串行 build→issue→rebuild、零流水。成因：v8 split-state 形状把两 slot 共 64 个 T2R 值寄存器跨 tail-commit wait 保活，在 104 寄存器上限（v9_3 L12450）下溢出到 LMEM。r0 慢 = 读回溢出的 thread_values_0；r1 快 = thread_values_1 刚在 wait s1 后写入未及溢出。提案假设裁决"(iii) 空窗重灌为主因"**被证伪**——空窗长度不在方程里，溢出才在。

---

## P1（burst-into-wait 重排）：判决 = **采纳，但它已经被 HEAD 实现（v11），且机制论证需换轴**

- **查重**：v11 就是 P1 的整块移动——slot0 原子循环上移到 release s0 与 wait s1 之间（diff v9_3↔v11 逐行核实），外加 P1 没有的必要配套：**寄存器再平衡 math 176→144 / reduce 104→120**（v11 L12445-12452，池恰平衡 256×48+128×144+256×120=61,440）。
- **约束合规**：✓。SMEM/TMEM/pipeline 协议零变化；wait/release 顺序不变（wait0→rel0→wait1→rel1），无死锁/parity/ABA 问题；burst 只读寄存器不读已释放的 TMEM。
- **证据修正**：P1 攻击的确实是（现在已单独确认的）pacer，但收益机制不是"burst 覆盖 wait1 + 暖流承接"（提案估 0.7-1.2µs），而是**液性减半 64→32 regs 消灭 LMEM 往返**（v11 预期 drain busy 5.9→3.3-3.7µs，period→max(ring~4.3, drain~3.5)，release ~8.0-8.5ms）。提案低估了自己。
- **一个 P1 原文的隐患**：P1 声称"寄存器峰值只降不升（104 内）"即可——存疑。v9_3 在 104 regs 下 32 值+fragments+8 索引+地址已经溢出（63 LDL 也出现在 math publish 窗），仅移块不调 104 上限可能 de-spill 不彻底；v11 的 +16 regs 再平衡很可能是承重件。若做 v9_3→v11 bisect 需意识到这是两个杠杆。
- **风险核查**：P1 自报的 CSE/地址链回归风险合法（e82fb19 证明编译器已跨循环 CSE，分开两 burst 有重物化风险）；v11 的验收门"drain 窗 LDL+STL≈0"比提案的"地址 op≤80"更对症。**残余不确定性**：v6（ce8bfb4）本就是 per-slot 顺序仍测得 ~5.0µs drain——v6 缺索引预载且有 reduce_sync_barrier，不完全可比，但提示 de-spill 后未必直达 3.3。
- **修正收益**：drain busy −1.9~−2.6µs/tile（v11 口径）；wall 取决于 ring 是否如 v11 假设缩到 ~4.3。
- **依赖**：无冲突；下一步唯一动作 = 实测 v11。

## P2a（dQ 移到 grads tail 之后 + kdq 改 g8/g9）：判决 = **击毙（现规格下）**

三条独立杀因：

1. **收益量化错误**。dQ 段实测（v9.3 `stage2_report.md` i=1 leader 时间线）：WAIT_dQ 0.064+0.064、dQ_ISSUE 0.064+0.256，leader 占位 8.416→9.120 ≈ **0.70µs**（v8 span 表 L41/L43 均值更小：0.050/0.079）——提案的 0.5-1.0µs 取的是上沿。且 head 4 个 pass 的起点是 **W17 供给节奏**（p0..p3 = 9.664/10.144/10.752/11.232，间距 ~0.5µs，与 MAT_QDO r0 包络 6.4→10.72 咬合），不是被 dQ 顶着——移走 dQ 后 head commit 实际只能前移 ~0.3-0.5µs。
2. **漏算了一条新增串行边**。kdq 改 g8/g9 后，在 ROUND_STAGES=2 的单 FIFO（L11209）里，gather 的 kdq(t) 填充信用 = g6/g7(t) 释放 = leader 的 tail dK pass——填充（同规格 gather 实测 LOAD_K 1.69µs 量级）被**完整暴露在 tail commit 与 dQ 之间**；而 dQ 消费 ds_image（L12222）受单级 pds 管线（pds_mbars=2 即 1 stage，L11320/L12308）保护，pds release（L13329）必须等 dQ → **ring 加长 ~1-1.5µs**。"gather slack 巨大"不救场：约束是 FIFO 信用到达时刻，不是 gather 占用率；预填不可能，加深 stage 无 SMEM（slack 1.5KiB）。提案对这条只字未提。
3. **理论基础已塌**。其收益故事（缩短 5.12µs 空窗避免 burst 冷启动重灌）建立在 F3 已证伪的机制上。baseline 先例（BASE:1539 part1 commit 先于 dQ）不可迁移——baseline 没有把 K_dQ 耦合进共享 FIFO 和 pds 环。

修正收益：现 regime **净负 0.5-1.5µs/tile**；即使条件 1（pds 双缓冲）落地后，暴露的 kdq 等待仍加长 leader 串行链（leader 发射预算 2.6-3.5 + 1-1.5 ≈ 逼近节拍），除非另建带独立缓冲的 kdq 管线——SMEM 不允许。条件性复活门槛：v11 实测后 drain 仍是 pacer + 找到 kdq 缓冲资金，两者同时成立才值得重开。

## P2b（relay wait 拆分）：判决 = **修改后采纳（收益≈0，降级为搭车项）**

- **机制核实全部通过**：head/tail 各自 pass 0-1 只用 p_fragments、pass 2-3 才用 ds_fragments（L13899-13959 / L14008-14068）；relay_mbars=P、relay_mbars+1=dS（DSM 发送 L13030-13081 先 P 后 dS；relay warp L13713-13729 按序转换）；拆分不动 mbar/phase，无 parity 风险；dS wait 放 head pass2 前即覆盖 tail。零字节成本。✓
- **但收益在现序下≈0**：leader 到达合并 wait 的时刻 ≈ 9.12（dQ 之后），而 ROUTE_dS 8.352 结束 + 4KiB DSM 落地 ≈ 8.7-8.9——到点时两个 relay 通常都已 landed；p0 前的 0.54µs 缝隙是 round 供给等待不是 relay。0.1-0.3µs 的收益只有在 dQ 被移走（=已击毙的 2a）后才出现。提案自己也把它标为 2a 的配套。
- 历史查重：splitpds（9110e23）拆的是 P/dS 的 release 生命期（v2 时代，噪声级 +0.19%），与本条拆 wait 不同构，不构成前科。
- 结论：不值得为它单独跑验证轮；仅当某次实测改动顺路时捎带。

## P3（WG-per-slot 独立流）：判决 = **击毙**

- **寄存器账被硬证据证伪**：v9.3 SASS 已实锤"64 f32 跨 wait 保活在 104 regs 下溢出"（F3）。P3 按构造每 thread 保活 **64 f32 payload**（Rep8 单 slot）+ 16 地址 + 索引——即使在 v11 的 120 regs 新上限下也是重造 v11 刚治好的病。提案原文"64 payload + 地址，104 内紧但可行"直接与 v10 SASS 矛盾。
- **收益已被 v11 用一半的寄存器代价拿走**：v11 的 per-slot 顺序（slot0 burst 先于 wait1）已实现"T2R(s1)/burst(s0) 解串行 + 跨 tile WG 提前"的主体。P3 相对 v11 的增量只剩跨 slot 的 wait 并行，而它的自设前提（ring 解耦 + WAIT→小）成立时 reduce 已不是 pacer。
- design16 前科（per-thread 链形状敏感）+ dkv_done consumer 按 WG 分立的 mbarrier count 改动风险，收益/风险比不支持保留 phase-2 席位。删。

## 否决清单（c/d/e）核查：三杀全部维持

- **(c) CONFIG B**：账目核实（64KiB/tile/CTA÷512B=128 条；v9 实测 24.63ms + v9.2 bisect 归因，e2f3756/32fdec5）。维持，高置信。
- **(d) 发射者/宽度**：red.v4.f32 确为 f32 归约向量上限；16GiB staging 账（4096×32×128KiB）无误。维持。
- **(e) N-split**：tcgen05.ld 仅本 CTA——TMEM 跨 CTA 不可达，成立。**但注意其论证有一条腿已失效**："r1 证明地址模式无关"现在只证明"未溢出时快"；地址无罪的有效证据只剩 baseline 自身 scramble 地址达 0.538ns/条（baseline L1528-1548 unscramble 注释 + REDUCE_dKV 2.203µs）。杀判靠余下的腿仍然成立。维持。

## 排序（修正收益/风险）与执行序修正

1. **P1（=v11，已落地）**：唯一攻击已确认 pacer 的项。动作 = 实测 v11；SASS 门：drain 窗 LDL+STL≈0、REG≤144/120、publish 门不回退（STSM=20/PRMT=32/STS.U16=0 已在 v9.3 验证）。预期 drain busy 3.3-3.7µs、release 8.0-8.5ms（v11 commit 自估，含 ring~4.3 的假设，v6 先例提示留下行风险）。
2. **P2b**：收益≈0、风险≈0，搭车项，不单独排期。
3. **P2a**：击毙——净负（kdq 暴露上环 > 0.3-0.5µs 的 commit 前移），且理论基础（重灌说）已被 SASS 证伪。
4. **P3**：击毙——寄存器账与 v10 SASS 直接矛盾，收益被 v11 半价拿走。

提案"汇总"里 P1+P2 → 2.8-3.2µs 的 reduce 侧账相应改写为：**P1/v11 单独 → 3.3-3.7µs（待实测）**；到 ≤3.2 的最后一段缺口不再经由 P2a，而是记在单寄存器组串行发射的残余与 pds 环（诊断条件 1）名下。

关键路径引用：`KDIR/dsa_bwd_sm100_2cta_v11.py` L12445-12452（144/120 再平衡）、drain 块移动区（≈L14150-14290）；`KDIR/dsa_bwd_sm100_2cta_v9_3.py` L14244/14256/14282-14351、L11205-11209、L11693-11764、L12222、L12308、L13329、L13862-13885、L13030-13081、L13713-13729、L11621-11665；实测 `/Users/longcheng/v10/validation_summary.json`（12.091059/7.995213）、`stage0_user_gate_summary.md`、`drain_t2r_through_last_atomic.sass`（65 LDL/28 STL/16 REDG 同组 R4/R8）、`stage2_report.md`；git b3c7464、e82fb19、07e0838、916b7cb、ce8bfb4。

---

# 附录：math/publish 视角 提案对抗验证判决

# 对抗性验证结论（math 视角 4 提案）

## 先决事实修正：提案集写在过时的证据状态上

本轮核实发现三件仓库里已发生、提案与诊断均未纳入的事：

1. **v9.3 已实测：12.09 ms**（memory `dsa-bwd-2cta-v2-design.md`；工件在 `/Users/longcheng/v10/`，run `20260729T093807Z_v9_3_v9_3`；topk128 锚点 3.036883 ms，`/Users/longcheng/v10/topk128_candidate_performance.json`）。stmatrix 修复完整兑现（`/Users/longcheng/v10/stage0_user_gate_summary.md`：STSM=20、PRMT=32、**STS.U16=0**；`stage2_report.md`：MATH_STORE raw per-warp 4.58→**1.082 µs**，MATH_PD 包络 10.14→~6.5-6.9）**但 wall 不动（12.09 vs v8 11.945，甚至 +0.145）**。"math 是三轨并列 pacer" 被这次实测证伪：drain raw busy 5.9 vs 周期 6.9 是唯一贴满的轨。
2. **v9.4 drain 折叠已回滚**（e82fb19）：SASS 证明地址链早被编译器 CSE（80→96 反升 1.2×），诊断条件 2 的 "v9.4 折叠" 机制已死；取代它的真凶已定位——**reducer 寄存器溢出**（v11 commit b3c7464：104 reg 下 64 个 T2R value 寄存器溢出 LMEM，REDG 循环内 65 LDL+28 STL，16 条 REDG 单寄存器组全串行，306 ns/op vs baseline 138）。
3. **v11 已写入待测**（`KDIR/dsa_bwd_sm100_2cta_v11.py`，= v9_3 + slot 时序恢复 + 寄存器再平衡 **math 144**/reduce 120），预期 8.0-8.5 ms，届时下一道墙的预估是 **ring ~4.3 µs**。

这直接改写各提案的攻击对象判定：当前（12.09 ms）唯一 pacer 是 drain；math 侧任何节拍杠杆在 v11 落地前 wall 兑现 ≈ 0。

---

## 提案 1：stmatrix 收益核定 + 删 ds_xchg —— **拆分判决**

**(a) stmatrix 部分：无需采纳——已落地且已实测，wall 收益 = 0。** 提案把它当 "待核定的 pacer 攻击"，实测答案：span 全兑现（store raw −76%，但停在 1.08 µs，未达提案预测的 0.4-0.7），wall 一动不动。"是真 pacer" 的论断被 12.09 ms 证伪。验收门中 "PRMT 清零" 也已被现实修订为 PRMT=32（gate_summary，PASS）。

**(b) ds_xchg 删除：修改后采纳（定性为资源腾挪，非性能杠杆）。** 我独立验证了构造性字节恒等：非拥有 warp（n_owner=1−rank）写 xchg 的地址 = `ds_xchg_raw − n_owner*4096 + 全像偏移[n_owner*4096, +4096)` = `ds_xchg_raw + [0,4096)`（V93:12726-12748），内容逐字节 = ds_image 半块 (1−rank)（全像由 4 warp 无条件重写，V93:12995-12999）；DSM 源改 `ds_image + (1−rank)*4096` 合法（Align-1024 +4096 → 16B 对齐；`_cpasync_bulk_s2cluster` 签名 src/dst 核对无误，V93:90-130, 13067-13081；dst `ds_block_raw_ptrs[rank]` 不变）。时序保护闭合：DSM 源读完成先于 landing→relay→leader passes→pds release→math(t+1) acquire，与今日 xchg 同一条链；ds_image 被 dQ UMMA 与 DSM 并发只读，无害；13007 的 fence + BAR1 覆盖 async-proxy 可见性。SMEM 账正确（struct 实测 230,912，删后 226,816、slack 5,632）。P 侧不可同法（p_blocks 两 slot = 本地写 + landing，无第三像，V93:11301 注释与 13036-13050 证实）。
**修正收益：wall ≈ 0 µs/tile（math 非 pacer；非拥有 warp 12→8 STSM ~0.1 µs 是 slack）；真实价值 = 4,096 B SMEM + 简化。** 风险低；保留提案自带的两步验证（先改源不删字段）。

## 提案 2：pds 环手术 —— **(i)+(ii) 修改后采纳（条件：v11 之后），(iii) 击毙**

**(iii) P 先行——击毙，且能说出与 earlyp 的"本质区别"不存在的机制**：leader 进入 pds 环的第一个消费者是 **dQ rounds，读的是 ds_image 全像**（V93:13862 consumer_wait → 13866 dQ rounds，dq_ds_fragment=fragment_B(ds_image)，V93:12222）——需要的是 math **最后**产出的 dS，不是 P。P 提前到达只前移 pass1/2（P 消费者），而 pass1/2 在 leader 单指令流里排在 dQ rounds 之后，且 round buffer 供给序（dQ 先释放、quad 后填，V93:13864-13865 注释）禁止把 P passes 提到 dQ 前。所以 "环解锁点前移" 不成立；这与 earlyp（f960683，18.027 ms，wall 零变化，报告在 KDIR）不只是同形——失败机制相同。"语境已换" 的辩护对该机制无效。

**(i) DSM/commit 迁 W18 + (ii) relay 拆双相——机制核实通过**：pds 单级、producer=math 128 线程（V93:12308-12315）属实；consumer_release 位于 grads tail 后（13329/13385）、软件侧无更早释放点属实；W18 每 tile 仅 2 次 cluster wait + 2 次 remote arrive（实际行号 13709-13729，提案引 13691-13712 略偏）；relay_mbars 本就按 P/dS 分立（12340-13341 init count=2），(ii) 只是把 13882-13885 的 dS wait 挪到 pass3（13930-13935）前——pass1/2 只读 p_fragments、pass3/4 才读 ds_fragments（13899-13959 核实），且每 pass 的 B desc 在某一 CTA 必读 landing 块，故 relay_p 必须留在 pass1 前——(ii) 的位置选择正确。(i) 的 count-128 mbar + W18 单线程 commit 无死锁环（两侧 W18 对称，landing 交叉，phase 单调）；acquire/commit 分属不同 warp 与 reducer 的 dkv_wait/dkv_rel 拆分状态（13167-13173）同构，DSL 可表达。mbar +16-24B 塞 748B pad，账正确。
**修正收益（降）**：BAR1 0.954 只是 straggler 探测器——warp skew 不因删 barrier 而消失，只是转移成 W18 的 mbar wait；环长净缩 = barrier 会合开销 + math 流内单线程发射窗 ≈ **0.5-1.2 µs**（提案 0.8-1.5 偏乐观）；math 包络 −1.5-2 µs。**wall 兑现前提 = v11 drain de-spill 先落地**（当前 drain 5.9 贴 6.9 周期，环不是墙）；届时它攻击的 ring ~4.3 是预估的下一 pacer——这是本提案集里唯一在 post-v11 世界攻击 pacer 的节拍杠杆。按提案的最小实验切法（先只做 (i)）执行。

## 提案 3：T2R 双发单 fence + scale 预折 —— **击毙（作为独立杠杆）；可作搭车项**

机制本身核实无误：r_score/r_dp 本就同时活跃，(i) 不加寄存器活性；延迟 release_S ~1µs 被 s_done 2 级（SCORE_DONE_STAGES=2，V93:11203）吸收；(ii) 代数正确（(dp+delta)·p·scale = (dp·scale+delta·scale)·p），stats 无其他消费者（12573-12586 后仅 softmax 读）。但两点击毙其独立价值：① v9_3 实测后 math 离 pacer 更远（MATH_PD 包络已 6.5-6.9 trace 口径，T2R_S/T2R_dP raw 0.45-1.44），回收 0.5-0.8 µs 打的是纯 slack，自我降级声明属实且应执行到底；② "96 reg / 176 预算，无溢出风险" 的账已过时——v11 把 math 预算压到 **144**（v11 diff，setmaxregister_increase(176)→(144)），且 math 窗口现存 LDL=63/STL=14、STACK=600 的残余溢出（gate_summary），任何 math 寄存器扰动需重过 SASS 门。**修正收益：wall 0；仅当因 P2 已在改 math 热循环时顺手带上。**

## 提案 4：dQ epilogue TMA 化 —— **采纳，四案中唯一无条件的 wall 正收益**

历史查重结果是正面且干净的：b244255 与本提案**机制逐项相同**（staging alias 死掉的 32 KiB score_kv + TMA S2G），wall −2.2% ≈ −0.39 ms 且未改源复测一致（17.603/17.601 ms，`KDIR/dsa_bwd_sm100_2cta_v2_dqepitma_b244255_report.md`），从未因故回滚——design-16 重写（64aca16）静默丢弃了它，无失败记录；host 管线仍在被弃（V93:11843-11861）。deadness/顺序链我逐环验证：leader 末次 kscore release（13279）→ grads tail → dq_done tcgen05.commit（13387，UMMA 退休后到达，故 dP 对 score_kv 的最后 async-proxy 读已退休）→ math consumer_wait（13103）→ EPI；round buffers 在 dq_done full 可见时也已死（tail passes 的读随同一 commit 退休）。补一次 proxy fence 即可，提案已列。

**三处修正**：① **F=40.1 是双计**——leader 的 `pipe_dq_done.producer_tail`（13393，在 TAIL span 内）等 math 的 `consumer_release`（13140，在两轮 EPI **之后**），故 TAIL 24.48 已包含 DQ_EPI 14.3；真串行尾 ≈ 26 µs。但边际收益幸存：尾部等待 = max(EPI 14.3, reduce 尾 ~6.6-7)，EPI→~7 后省 ≈ **7 µs/cluster ≈ −0.35~0.45 ms wall**，与 b244255 实测 −0.39 ms 精确互证；**再往下压 EPI 无收益**（reduce 尾成为新 max）。② dqepitma 时代 TMA epi 的 **span 反而是 17.76 µs/launch**（>标量）而 wall 变好——验收门必须是 release wall，提案的 "DQ_EPI→6-8µs span" 预测无先例支撑，勿当门槛。③ round1 staging "alias round_buf_a（16K）" 字节账错：每轮需 32 KiB（32K el/round/CTA），应用 round_buf_a+b 连续 32K 或 score_kv 串行两轮。
**修正收益：−0.35~0.45 ms wall（12 ms 位面 ~3%；对任何 pacer 状态可加）。风险低。**

---

## 视角内排序（修正收益/风险）与依赖

1. **P4**（无条件 −0.35~0.45 ms，正面先例 + 全链核实通过；独立，可先行）
2. **P1(b) xchg 删除**（wall≈0 但零风险 + 4KB SMEM；为 P2 的 mbar 与任何未来 staging 供资；P1(a) 已随 v9.3 落地、收益已核定为 span 兑现/wall 归零）
3. **P2(i)+(ii)**（post-v11 攻击预估的下一 pacer ring ~4.3；环缩 0.5-1.2 µs；**排程依赖：先实测 v11**——若 v11 落在 8.0-8.5 且 ring 确为新墙，此提案升为第一优先；若 drain 仍 >4.3 则继续搁置）
4. **P3**（搭车项，单独不做）
5. **P2(iii)**：击毙（earlyp 同机制，leader 环入口需 dS 全像，提前 P 无解锁效果）。

合成落点校准：提案自评 "math 侧单独无法给出 <8.09 ms" 成立且应加强——v9.3 实测证明 math 侧已交付的最大单杠杆买到 0 wall；math 视角剩余的真实筹码 = P4 的 −0.4 ms（固定成本）+ P2 在 v11 之后对 ring 的 0.5-1.2 µs/tile（×55.35 波×32 tile ≈ 至多 −0.9~2.1 ms，取决于 ring 是否顶节拍）。两者全兑现 + v11 达预期 8.0-8.5，才把 <8.09 拉进误差带内。

---

# 附录：供给侧视角 提案对抗验证判决

All key facts are now verified against code, traces, and git history. Composing the adversarial verdict.

# 对抗性验证结果（supply 视角提案）

## 先核实提案的"结构性事实 0"——部分成立，但推论有一个致命漏洞

**成立的部分**（逐行核实）：
- gen 解码/映射：`flat_gen//4=round, (fg//2)%2=tensor, fg%2=h_half`（v9_3.py L13510-13513），h_half 选 buffer/mbar（L13528, L13544/13578），own/peer 按 `rank==h` 分 bulk/TMA（L13538-13549）——**rank0 的 4 条 peer TMA 确实全在 buf B/mbar1** ✓。
- pds 环包含整个 grads 块：`pds consumer_wait` 在 head 顶（L13862），`consumer_release` 在 **tail 之后**（L13329/L13385），而 8 个 dV/dK pass 每个都 `round_pipeline.consumer_wait` 在 dVdK_ISSUE span 之外（L13894/13912/13930/13948 及 tail 对应位）——**panel 供给暴露确实加长 pds 环** ✓。
- 单 producer/单 consumer、2-stage、10 gen/tile：pipe_round 2 stage（L3872, L12300-12307），round region 2×16 KiB（L11345-11352），且 V2 docstring 明文契约"a single 2-stage pipeline carries all ten generations per tile, so no barrier ever skips a phase"（L11147-11151）。

**致命漏洞（击毙 P1/P2 的核心）——cluster 级不变性**：pipe_round 是 cluster 管线（`cta_layout_vmnk=cluster`，L12305；CG2 pass 的 B 操作数读双 CTA SMEM 半），leader 的 pass(s) 必须等**两个 CTA** 的 gen s 都 commit。而 panel (·,·,h) 对 rank h 是本地 bulk、对 rank 1-h 必是 TMA，且同一 gen 两 CTA 必须装**同一逻辑 panel**（各自 M 半）——**任何排列下每个 gen 在 cluster 级恰好含 1 条 TMA**。配上 2-stage 信用链 `issue(s) ≥ release(s-2) = pass(s-2) ≥ complete(s-2)`，每 tile 的供给串行链恒为 **4 个 TMA 往返（4L）**，与 gen 顺序无关。旧序（TMA 在 rank1 偶 gen/rank0 奇 gen 链上各 4 条串行）与新序（rank1 g2,g3,g8,g9 / rank0 g4-g7）给出**逐 pass 完全相同的节拍：每 2 gen 一个 L**。P1 的"每 CTA 单 mbar 串行"是真的，但它从来不是约束——cluster 级本来就有 2 条 TMA 并飞（每 CTA 一条）。

---

## 提案 1（gen/pass 重排）：**击毙**（作为性能提案）

理由：
1. **收益推导错在把 per-CTA 包络当 cluster 临界路径**。如上，4L 链对重排不变。修正收益 ≈ 0-0.3 µs/tile（仅消掉单 warp 程序序的 zipper 耦合，且该耦合速率 max(sw, (L+sw)/2) 中 sw≈0.1-0.3 µs 本就不 binding）。
2. **r0=7.23/r1=2.78 的 2.6× 不对称主要不是"单通道周转"**：MAT_QDO r0 span 从 ROUTE_K(t) 结束（grads(t-1) tail 期间）开始，第一个 `producer_acquire(g2)` 要等 g0(t) 释放 = grads(t) 的 dQ round 0——中间隔一整个 math 窗口。r0 吸收的是这个窗口。因此**提案自己的验收门（r0≈r1）在实现正确时也不会达成**，实验设计失效。
3. **历史查重不利**：tmapair（aed6812，"gave the two h0/h1 quadrant refills independent raw TMA barriers and issued both copies before waiting"）目标与 P1 同构——让 panel TMA 并飞——实测 18.441 vs 前代 17.994 ms（−2.49%），回滚（654004a）。"不加新对象"的区别不成立，因为并发度从来不是墙。报告原文还预警了 P1 用的证据口径："MAT_QDO … cannot be used as proof that hardware overlap occurred"（tmapair_report.md）。
4. 合法性方面无硬伤（4 pass 同打 t_dkv_0/t_dkv_1、首 pass ACC=F 已核实 L13899-13965/L13999-14019；重排只改 FP 累加序，容差级），但这只说明"可以做"，不说明"值得做"。

**残值**：若作为 1 次廉价证伪实验（验证 cluster 不变性模型），预期 wall Δ≈0；信息量存在但低于提案宣称。

## 提案 2（kdq 移交 + W19 双 producer）：**击毙**

**(a) kdq 移交 gather**——零收益：
- `WAIT_dQ(i,r)` 实测 mean **0.050 µs**（n=64，span_stats_with_waits.md:41）：leader 从不等 kdq。kdq 供给链有一整个 math 窗口的余量（ROUTE_K(t) 在 grads(t-1) tail 期间完成，消费在 grads(t)）。
- 移交后 W17 的下一个阻塞点 `acquire(g2(t))` 等 g0(t) 释放（grads(t) dQ r0）——比 ROUTE_K 等的 g8/g9(t-1)（grads(t-1) tail）**更晚**。"W17 直接从 g2 起跑"是幻觉：它只是换个地方睡觉。fill 本身已在 gather（v8 起，L11646-11665），移交只挪握手记账，kdq commit 时刻不变。
- 历史：v8 commit 明文"**W17 keeps sole ownership of round pipeline ops**"且"kdq never delays the next score gather"是 v8 的刻意设计；V2 docstring 的单 producer 契约（L11149-11151）是相位安全根基。为零收益破除它，纯负 EV。

**(b) W19 分通道**——收益 ≤0.3 µs 且不稳：单 producer 的 zipper（commit(s) 被 body s+1 的 acquire 卡）只在 sw > (L+sw')/2 时 binding，生产态不成立；trace 里看到的 per-gen 软件开销是 ×1.7 仪器膨胀的产物。风险侧是实打实的：若与 (a) 叠加，同一 cluster 管线 stage0/1 的 empty-mbar 相位由 **3 个 producer**（gather/W19/W17）分段推进，每 stage 每 tile 翻相 5 次——教训 #9 的原型场景。修正收益/风险比不支持实施。

## 提案 3（K_dQ 复用 score K）：**同意搁置，并加重为击毙**

- 收益侧核实无误：打在 slack 上（WAIT_dQ 0.05 µs；gather union 29.8%）。
- 成本侧比提案自评更差：新增 16 KiB/tile DSM 走 async proxy——v9 protocol A（同引擎、同类跨 CTA 握手）实测被判死：v9.2 二分 **18.36 ms = push 自身 +6.4 ms**（07e0838 commit 原文"condemned by hardware economics at 24.63/18.36ms"）✓；且 score_kv 复用需在 **1-stage** kscore 环（L12292-12299）上加跨 CTA 完成门，gather(t+1) 的 K 回收现在紧跟 S/dP issue，新门直接串行化 gather 轨。
- 提案引用的布局断言 **L10761 在休眠的 V1 类**里（`_specialize_shared_storage` "v1 224-KiB envelope"，L10748）；活动 V2 类的版本（L11285-11311）只有 cosize 断言——字节恒等前提比宣称的弱一档。
- 重启触发条件（leader 等 kdq）在 2-stage round 管线结构下几乎不可能出现。**关闭此路线。**

## 提案 4（b/d/e 判决 + khot 清理）：**采纳，其中 (d) 的一条论据必须更正**

- **(b) peer-half 保 TMA**：✓ 实测背书充分（+6.4 ms 二分），字节账正确（64 KiB/CTA/tile）。采纳。
- **(d) round 深度 3-4 否决**：SMEM 账核实无误（SharedStorageV2 数据段 229,888 B + mbar 块 ≈ 230,912 / 232,448，slack 1,536 B；struct L11313-11378）。**但"P1 用映射达成了深度 3 想买的东西"是错的**（cluster 不变性）：深度 ≥3 是唯一能把 4L 链压到 2L 的杠杆。否决理由仅剩"SMEM 拍卖输给条件 1"——这成立，但意味着若 ds_image 合并筹到 >24 KiB 或条件 1 失败，深度 3 应立即重上桌。
- **(e) 常驻否决**：✓ 算术核实（~304 vs 227 KiB）；descriptor 逃逸口确实不存在（S/dP 的 K_CHUNKS 循环跨全部 4 个 D128 chunk，L13815；stationary 不可被覆写）。采纳。
- khot_seq 死代码：✓（L11329 声明、L12344 唯一写入、零读者）。可删。

## 合成修正与排序

提案包宣称的 **1.1-2.4 µs/tile 修正为 ~0-0.5 µs/tile**。"结构性事实 0"的诊断价值保留（供给暴露确在 pds 环内，≈4L），但两个零 SMEM 提案都动不了 4L——它对 gen 排序**不变**。真正的杠杆全在 SMEM 拍卖桌上：pds 双缓冲（条件 1，+24 KiB）、round 深度 3（+16 KiB）、或 splitpds 系（9110e23，拆 ds_image 门让 dQ 提前）——即供给问题应并入条件 1 的 SMEM/协议讨论，而非独立的重排轨道。另注意：条件 1 落地后 leader 轨 ≈ S/dP issue + grads 块（含 4L）≈ 3-4.5 µs，贴着 4.10 µs 追平线——4L 届时可能成为新墙，且 P1/P2 依然无解。

**视角内排序（修正收益/风险）**：
1. **P4**：采纳（负结果固化 + 死代码清理；零风险，防止重烧 B200 槽位）。
2. **P1**:击毙；如需 1 次廉价模型证伪实验可跑，预期 Δ≈0，验收门需改（r0≈r1 无效）。
3. **P2**：击毙（(a)=0；(b)≤0.3 µs 对 3-producer 相位风险）。
4. **P3**：击毙（负 EV；v9 同引擎失败先例 +6.4 ms）。

关键证据：v9_3.py L13510-13513/13528-13604（映射）、L13674-13701（depth-2 发射/等待序）、L13862+L13329/13385（pds 环界）、L13894-13965（release-after-issue 突发）、L12300-12315（cluster 管线）、L11147-11151（单 producer 契约）、L11170（IDLE_WARP=19）、L11313-11378（SMEM 账，slack 1,536 B）；span_stats_with_waits.md:28,31,39,41（MAT_QDO 4.83/max 7.23、ROUTE_K 1.789、WAIT_dK、WAIT_dQ 0.050）；tmapair 报告（18.441 ms，−2.49%，"cannot be used as proof that hardware overlap occurred"）；git 07e0838（v9.2 bisect 18.36 ms）、916b7cb（v8 kdq offload 设计决定）、654004a/15964f7（回滚记录）。

---

# 附录：调度/摊销视角 提案对抗验证判决

# 对抗性验证结果（schedule 视角提案 S1–S4）

## 横切发现：提案与诊断共享的一个账目错误（先立此桩，S1/S2 收益都要按它修正）

**DQ_EPI 不是串行在 TAIL 之后，而是嵌套在 TAIL 尾部之内。** 证据链：
- TAIL span 的终点在四个 producer_tail 之后（v9_3.py L13390-13394），其中 `pipe_dq_done.producer_tail`（L13393）按 `PipelineUmmaAsync.producer_tail` 语义（CuTeDSL pipeline/sm100.py L766-791：leader 逐 stage 等 empty barrier）阻塞到 math 的 `consumer_release(dq_done_state)`——而该 release 在 **两个 DQ_EPI 完成之后**（L13140，位于 L13104-13139 两次 `_store_dq_from_tmem` 之后）。
- 数字自洽：TAIL 24.480 − DQ_EPI 14.304 = 10.18 µs ≈ 末 tile 的 W17 供给链 2×MAT_QDO = 9.66 µs（`/Users/longcheng/v8/two_trace_tables.md:35,37`）。即 TAIL = leader 发射段 ~10.2 + 等 math epi ~14.3。
- 推论 1：诊断的 F = TAIL + DQ_EPI + LOAD ≈ 40.1 µs **双计了 14.3 µs**，真实 per-token 尾部 ≈ 26–28 µs（净稳态节拍相应为 ~5.9 而非 5.49 µs/tile）。
- 推论 2：S1 的前提句"DQ_EPI 当前严格串行在 TAIL 24.48 之后"错误，其收益额度不是 14.3 µs 而是"把 epi 起点从 t≈10.2 提前到 t≈2–4"的差额，且新 TAIL 终点被 reducer 末段 drain（slot0/1 T2R+atomic，自 tail-commit ~t10 起 ≈7 µs → 结束 ~15–17）托底。

另注：git 上已有比 v9.3 更新的 b3c7464 (v11, reducer de-spill)；v11 的四个锚点结构不变（v11.py L13100 producer_tail(pds)、L13105 consumer_wait(dq_done)、L13389 commit、L11859 弃 tma_atom_dq_epi），S1/S2 对 v11 同样适用。

---

## S1（dq_done 提前 commit + pds producer_tail 后移）：**修改后采纳**（收益减半）

**机制核实（全部通过）**：
- 代码事实与提案一致：TAIL 内 commit 在 8 个 dKV pass 与 pds release 之后（L13385-13388）；head 内顺序 = pds wait（L13862）→ `_issue_dq_rounds_v2`（L13866-13877）→ relay wait（L13881-13885）→ 4 pass；math 侧 `producer_tail(pds_state)`（L13097-13098）确实在 `consumer_wait(dq_done)`（L13103）之前，是必须一起动的闸门——提案定位正确。
- TMEM 并发合法性**实证**：V2 布局 DQ0=128, DQ1=256, DKV0=384, DKV1=448（L11196-11199），math T2R 读 [128,384)、leader 写 [384,512)，不相交 ✓。
- 协议：pipe_dq_done 1 stage（L12324-12331），每 token 一次 commit，无 parity/ABA 面；commit 是被动指令（tcgen05.commit，与 L13795 每 tile 的 s_done commit 同型），提前不构成环；死锁链检查：leader 的 relay wait 依赖 math tile-31 的 DSM（TAIL 前已发），math 的 dq_done wait 依赖 leader 已发出的 commit——无环 ✓。tile_count==1/0 边界均安全（L13097/13102 的 guard 保留即可）。
- SMEM/TMEM/mbar 账：全零 ✓。

**收益修正（唯一击点）**：epi 起点 10.2→2–4（TAIL 入口的 pds(31) wait 可能吃 1–2 µs：math(31) 的 publish 未必先于 leader 进 TAIL），epi 终点 ≈16.3–18.3；drain 路径终点 ≈15–17；新 TAIL ≈17 vs 旧 24.5。**修正收益 ≈ 6–8 µs/token = 0.19–0.25 µs/tile ≈ 0.33–0.44 ms wall（v8 基），不是提案的 12–14 µs / 0.66–0.79 ms**。附带收益句"reduce 末 burst 与 DQ_EPI 从此并行"无效——它们今天已经并行（都在 t_r 后）。风险评估维持：纯删串行点，无新环；LSU 争用（epi 32,768 条 STG/CTA vs 末段 8,192 原子）可能再蚀 ~1 µs。

**依赖**：无。验收门改为：TAIL span 从 24.5 → ~17（而非 wall −0.7 ms）。

## S2（dQ epilogue TMA 化，staging=score_kv）：**修改后采纳**（当期收益与 S1 近乎替代而非叠加；终局件价值不变）

**机制核实（全部通过）**：
- 字节账精确：score_kv = 16384 el × 2 B = 32,768 B（L11341-11344）；dq_epi staging 每 round [H128,D128] BF16 = 32,768 B，host 侧 assert `dq_epi_bytes <= 32*1024`（L494）；SMEM +0 ✓。工件齐全且现被弃用（host L476-502、传参 L793-795、kernel 弃用 L11859-11861）；V0 可移植模式完整（L5825-5879，含 fence→tma_partition→copy→bulk_commit→wait_group(read)）。
- alias 安全实证：gather 恰好做 tile_count 次 kscore fill，无越尾 prefetch（prologue L12473-12497 + 循环 `range(tile_count-1)` L12502），末次 kdq gather 写 round buf 不碰 score_kv（L12548-12561）；score_kv 最后读者 = S/dP(31) MMA，被 dq_done 的 UMMA-tracked commit 覆盖（S1 移位后 commit 仍在 S/dP(31) 之后发射，跟踪关系不变）✓。同步不需新 mbar：elected warp wait_group 后过 math_barrier 即可广播（V0 用 mbar 是因为它没有等价 barrier）。
- 历史查重结论：**非回滚同构**。b244255 实测 17.603/17.601 ms vs 前代 9110e23 的 17.994 ms = **−2.2% wall、双跑复现、dense correctness 通过**（report.md 明载"staging aliases the dead 32 KiB score-K allocation"——与本案同一手法）；其消失是 design-16/v4 重构线（64aca16→16a3c62）换底稿时丢弃，git 上**无 revert commit**，不属"已否决方案复活"。

**收益修正**：因横切发现，"独立看省 ~8.3 µs/token"高估——S2 单独落地时 epi 起点仍是 t≈10.2，新 TAIL ≈ max(drain ~15–17, 10.2+6) ≈ 16–17，省 ~6–7 µs/token，**与 S1 单独落地几乎等值；S1+S2 合计 ≈ 7.5–9.5 µs/token ≈ 0.42–0.53 ms，不是 0.7–0.8 ms**（此后 TAIL 由 reducer 末段 drain 托底，schedule 侧再无零字节手段）。终局价值维持：F≤15 时 14.3 µs 标量 epi 必然重新露头 + 消除与原子共栈的 65,536 条 2B 散射 STG。
**新增风险（提案未列）**：R2S 若走标量 STS 会重演 v8 MATH_STORE 的 STS.U16 降级（summary.txt: 160 STS + PRMT ≈ 5 µs）——必须复用 v9.3 的 16-DP T2R atom 路径；SASS 门加一条：epi 窗口 STS 向量化/STSM，否则 5–6 µs 的估计不成立。

**依赖**：与 S1 同批（互为替代下限、合并上限）；与 S3 组合需新增 kscore-acquire←epi-wait_group 跨 role 闸门（提案已自认，+1 mbar，环外）。

## S3（persistent 2-token cluster）：**有条件采纳**（维持启动条件；补一处实现约束；收益按修正 F 重算）

- 波次不变性证明**核实成立**：4096 = 74×55+26，2048 = 74×27+50，两者浪费均 (74−r)/(waves×74) = 1.158% ✓——S3 正确杀掉了候选 d。
- grid 事实 ✓（L804-808）；跨界管线连续性机制上可行（dkv_done/s_done/dp_done/round/kscore 皆 generation 化，phase 自然翻转；dq_epi_done 闸门在 n+1 tile-1 处，S1 后实际不阻塞）。
- **发现一处提案遗漏的读者**：stationary_q 除 S/dP MMA 外还有 OWN_HALF_BULK 路径（L11227 `OWN_HALF_BULK=True`；L13641-13651：rank1 从 `stationary_q_raw + 4096*(4*grad_round+2)` 经 bulk DSM 填 round_buf_b，**每 tile 都读，含 TAIL 期的末 tile 填充**）。"leader 末 dP 后 commit 即换装"不充分；救济是免费的但必须写死：**新 QDO TMA 只能由 W17 在其末 tile fill drain（L13693-13701）之后按程序序发射**，不能挂在收到 commit 信号的任意时点。
- 收益修正：S1/S2 落地后 F ≈ 17(TAIL) + 1.3(LOAD) + ~3(头部 alloc/init) ≈ 21 µs；跨界残余 ≈ max(末 tile W17 链与下一 token 供给的真串行段) ≈ 8–10 µs → **省 ~10–12 µs/token ≈ 0.55–0.66 ms**（提案的 10–15 上端偏乐观）。
- 判决维持提案自设的启动条件：稳态节拍未破 ~4.3 µs 前不动它；风险为四案最高（五角色+host、两个新闸门、trace 工具假设单 token）。两步 bring-up（T=1 iso-perf 骨架先行）合理。注意 v11 若改变 drain 形态，S3 的 TAIL 遮蔽账需重算。

## S4（否决/降级清单）：**采纳（四项数字全部核实）**

1. LOAD_STATS：0.544−0.256 = 0.288 µs/token × 55.35 = **15.9 µs wall** ✓（span 表 :15-16）——否决正确。
2. 波次量化 1.158% 不变性 ✓（见 S3）；tail-wave 绝对量 11.945×1.158% ≈ 0.138 ms ✓。
3. SMEM 账精确：SharedStorageV2 = 1024(头) + 229,888(数据+stats) = **230,912 / 232,448，slack 1,536 B**（逐项复算 L11313-11378）✓；pds 出口双份 blocks+xchg = 8+4+8+4 = 24 KiB ✓ 装不下；"leader 时长由 W17 fill 链决定"与 span 证据一致（MAT_QDO 9.66 vs grads issue 合计 0.66，two_trace_tables :20,:35）✓。relay wait 在 head 中位于 dQ rounds 之后（L13881）且其等待对象一 tile 前已落地——"被遮蔽"成立。
4. g2..g9 tile 不变量观察属实（stationary 128 KiB/tile 重复过窗，含 OWN_HALF_BULK 的 SMEM 源）——移交供给链视角正确。

---

## 视角内排序（修正收益 / 风险）

1. **S1**：0.33–0.44 ms，~10 行 diff，协议零风险——首发。验收门：TAIL 24.5→~17。
2. **S2**：与 S1 合并共 0.42–0.53 ms；有 −2.2% wall 实测先例（b244255）且非因罪回滚；终局（F≤15）必需件。SASS 门需加"STS 向量化"。
3. **S4**：零成本止损，防止 16 µs 级的伪优化——照单执行。
4. **S3**：0.55–0.66 ms 潜力最大但风险最高、且只在稳态节拍先破 ~4.3 µs 后才有决定性意义；实现必须遵守 OWN_HALF_BULK 程序序约束。

**总账修正**：S1+S2 合计 ≈ 0.42–0.53 ms（非提案的 0.7–0.8），把 v8 从 11.95 推到 ~11.4–11.5，v9.3 若落 9.5–10.5 则推到 ~9.0–10.1；同时横切发现把诊断的 F 从 40.1 修正为 ~26–28、净稳态节拍从 5.49 修正为 ~5.9 µs/tile——追平线预算相应从 3.32 放宽到 ~3.7 µs/tile，但稳态墙比诊断认定的还高 0.4 µs/tile，条件 1（pds 去环）的必要性只增不减。

---

# 附录：结构替代视角 提案对抗验证判决

All verification complete. Compiling the adversarial review.

# 对抗性验证结论（structural 提案组）

## 前置：两个改变评估基准的新事实（git 史核实，提案与诊断均未反映）

1. **v9.4 drain 折叠已被回滚** (`e82fb19` "Revert the v9.4 drain fold in v9_3: its premise was falsified at SASS")：预检实测地址操作 80→96 (+1.20x) 而非减半——v8 的分 slot 循环是直线代码，编译器早已跨循环 CSE，"per-slot 重计算开销"不存在；手工融合反而加剧 104 寄存器帽下的 rematerialization。诊断条件 #2 与提案对 (e) 的否决理由中所有 "v9.4 折叠" 引用作废。
2. **v11 已定位 drain 悖论的真机制** (`b3c7464`, `2b905f6`)：v9_3 drain SASS 显示 REDG 循环内 65 LDL + 28 STL（104 寄存器下 64 个 T2R 值寄存器 spill 到 LMEM）+ 全部 16 条 REDG.E.ADD.F32x4 复用同一寄存器组 R4-R9（零原子间流水）→ 306ns/op vs baseline 138ns/op 的 3x 赤字。v6 以来"~5µs 固定成本=协议冷启动"的解释被部分取代：是**寄存器 spill + 单寄存器组串行化**。v11 修法=恢复 v6 形状的分 slot 时序 + 寄存器再平衡 144/120→136/124，预期 drain 5.9→3.3-3.7µs，**period → max(ring ~4.3, drain ~3.5)**。这独立佐证：stmatrix 修复后 pds 环 ≈ **4.3µs**（不是诊断引用的 v8 时代 5.5-6µs）——S1 打的墙仍是主墙，但高度矮了 1-1.5µs。

---

## S1（pds 四路 commit-gated 裂解）——判决：**修改后采纳**

**约束合规（全部通过）**：
- SMEM 账核实：struct 实测（`dsa_bwd_sm100_2cta_v9_3.py:11313-11378`）mbar 头 33×Int64+Int64+Int32=276B，对齐垫 748B；4 管线×2 mbar − 现有 pds_mbars 2 = 净 +6 Int64 = 48B < 748B ✓，数据区 229,888B 不动，总 230,912B 不变。
- 机制同构性核实：现有 `pipe_pds.consumer_release`（PipelineAsyncUmma，L12308 单级）即 umma-commit 式释放；grads head 中途 `producer_commit(dkv)`（L13967，fence 后）证明 mid-sequence commit 释放是已验证模式 ✓。producer_group=math_group（L12310）→ 四路 commit 是逐线程 arrive，无需额外 barrier ✓。
- 消费点声明逐条核实：ds_image 唯一消费者 = dQ 两 round（L12222 `make_fragment_B(ds_image)`，L11737-11753），且 dQ 在 relay wait **之前**（L13862 pds wait → L13866 dQ → L13881 relay wait → L13899 dKV pass1）✓；P 末次消费 = tail pass2、dS = tail pass4（head/tail 各 4 pass 顺序 P,P,dS,dS，L13899-13959 同构 tail）✓。store 重排合法：P/dS 在同一 softmax 循环内先于任何 store 算完（L12928-12942）✓。无死锁：leader 序列内所有 release 点不等待 math(t+1) 的任何产出，环单向 ✓。

**必须修正的两点**：
- **历史查重不完全成立**：splitpds 报告（`..._splitpds_9110e23_report.md`）原文 "P is released after the final dV pass; dS is released after the final dK pass"——即 **S1 的 p_blk/ds_blk 中途释放 splitpds 已做过**（P 释放点=tail pass2 正是 S1 的方案），结果 17.994ms 噪声级。S1 的真增量只有两项：① image 提前到 dQ-round1 commit、xchg 提前到 relay 观察点；② 把 pass4 后的残余串行段从"全部 24KiB store"缩到"仅 ds_blk 4KiB store"。提案说 splitpds "release 仍在末 pass" 是半错的。
- **收益高估**：−1.0~−2.0µs 的上端把 v8 标量 store 时代的环长（5.5-6）当起点；但 stmatrix 修复是 S1 前置条件，修复后环 ≈4.3（v11 commit 独立估计）。残余环核算：pass4(t-1) 完成 → ds_blk store ~0.3 + BAR1 ~0.15 + DSM+landing+relay ~0.6 + dKV 发射 ~1.0 + pass4 完成 ~0.7 ≈ **2.8-3.3µs**（提案的 2.6-3.3 吻合）。节拍 → max(drain 3.5 [v11 后], math 3.5-4.5, W17 2.8-4, 残余环 3.3)。

**修正收益：−0.3~−1.2µs/tile**（中位 ~0.7），F=16 下对应 wall ~8.3-9.3ms 区间的下移；仍是唯一作用于 post-v11 pacer 的杠杆。风险：IKET 名额已精确=31（`e2f3756` 明言 31 是编码上限）——新 ACQ span 必须以退役换新增；release 点错置只会退化成 splitpds（噪声），不会错算。依赖：v9.3 stmatrix + v11 drain 先落地并实测确认环为 pacer；与 S2 有一处交互（见下）。

## S2（ds_xchg 消除，DSM 源改 image slab）——判决：**采纳（本组最扎实）**

**字节恒等被代码证明得比提案还强**：xchg store 视图本身就是 image 域视图——L12719-12748 用 `ds_xchg_raw − n_owner×PDS_BLOCK_BYTES` 作基址、`score_store_domain` 全域 + 同一 `score_store_layout.inner` swizzle 构造，即 **现行代码已保证 xchg 内容 ≡ image 的 4KiB slab 逐字节**（swizzle 周期 1024B | 4096B offset，L11998-12009 三布局 inner 相等断言）。把 DSM 源指针（操作数，非 descriptor，无 CG2 同偏移约束）换成 `ds_image + 4096×(1−rank)` 是恒等变换。对齐：image 1024 对齐，slab 满足 PTX 16B 要求（`09_instruction_set.md:9797`）✓。ds_xchg 在 V2 活动类中无其他消费者（仅 L11369 struct/L12059 raw/L12726 store 视图/L13067 送源）✓。SMEM −4096B（230,912→226,816，slack 1536→5632）✓。顺带删 L12987-12991 的 else 分支 dS 拷贝与 L12726-12748 半套视图，还有 v11 register-rebalance 语境下 math 侧活跃范围的轻微缓解。

**提案漏写的一个新约束**：ds_image 变成 DSM 源后，其生命期门 = max(dQ 消费, **对端读完成**)。单级 pds 下无影响（释放本就在 relay+grads 之后）；但与 S1 合成时 image 的释放点必须从 "dQ round1 commit" 后移到 "relay wait 之后"（leader 序列上仅差 ~0.2-0.5µs，稳态≈0，但必须写进 S1 的释放表，否则 math(t+1) 覆写 image 时对端可能未读完——正确性 bug）。历史无同构失败：CONFIG B 复用 ds_image 作 staging 池的草案 fatal（越界赛跑 ROUTE_dS landing，`32fdec5` 自述）是 32 行池视图越界问题，与本方案只读 slab 无关。

**修正收益：math 轨 −0.1~−0.25µs/tile**（v11 门 STSM 20/warp，删 1/5 目的地 ≈ −4 STSM；仅当 math 是残余 pacer 时兑现为 wall），外加 4KiB SMEM 与协议简化。风险接近零（build-time assert + dense 一票验证）。

## S3（relay 消除 / DSM mbar 重定位到 leader）——判决：**降级为"S1 后可选清理"，微测门维持**

- PTX 引文逐条核实无误：`09_instruction_set.md:9807`（完成机制表）、:9814（mbar 走 generic-proxy）、:9834（complete-tx = .release@cluster）、:9887（S2S 语法例）、:10412/:10437/:10439（tensor 变体 mbar 置放规则含 cta_group::2 "dst 或其 peer"）。平面 S2S 语法（:9752-9767）确无 `.cta_group` 修饰符；现行 helper `_cpasync_bulk_s2cluster`（L90-131）把 dst **和 mbar 都** `_map_smem_to_cluster_rank(..., peer_rank)`——即今天 mbar 恒与 dst 同 CTA。S3 中 CTA1→CTA0 的两条 send（mbar 在 dst CTA）是文档内行为；只有 CTA0→CTA1 的两条需要未背书的 "mbar 在源 CTA" 形态 → 微基准门设置正确，退化方案（leader 直等 CTA0 本地 landing_mbars，CTA1 保留一跳 relay）经核实可行（CTA0 landing 本就 CTA0 本地，L13713-13721）。
- **提案未写明的新竞态**：arm 从 sender（L13030 远端 arrive_and_expect_tx，天然先于本线程 send）移到 leader 后，arm(t+1) 必须 happens-before 全部 4 条 send(t+1)。可满足：leader 在 landing-wait(t) 之后立即 arm，而解锁 send(t+1) 的 blk release 在其后的 commit 才发出——程序序+因果闭合。必须进实现清单。
- **收益下修**：若 S1 落地，relay 跳（W18 唤醒+远端 arrive ≈0.1-0.3µs）位于残余环（~3.3µs）上，而残余环低于 drain/math 轨 → 稳态 wall 收益 ≈ **0~0.15µs/tile**（非 0.2-0.4）。真实价值=释放 2 个 W18 warp、TAIL 内 relay 排空消失（作用于 F）、协议减一层。

## S4（两 kernel 拆分否决重估）——判决：**维持否决（确认）**

算术复核：55.35×[(32×2.9+15)+(32×2.45+15)]µs = 55.35×201.2 = **11.14ms** ✓；spill 32×16KiB×8192 CTA = 4GiB ✓（k2 双 CTA 全量回读 → 12GiB 往返，~5TB/s 下 ≥2.4ms 纯 HBM 底噪）。结构论证成立：拆分把 fused 的 max(2.9, 2.45) 变 sum(5.35µs) > baseline 4.57，PDL 级联上界也只回到 max。一处反向修正：v11 证明 drain 赤字=寄存器 spill，则 k2 专职 reducer kernel（无 math/leader 抢寄存器）反而**更容易**达到 2.2µs 发射下限——但这只让被否决方案的分项更可信，不改变 sum>max 的死刑。否决置信度：高。

## 否决清单抽查

- (e) 细粒度 slot 的否决理由引用 "v9.4 折叠已覆盖" **已过时**（v9.4 回滚），但 v11 的分 slot 时序覆盖同一问题（slot0 burst 与 grads tail 重叠），否决结论不变、依据需换成 v11。
- (a)/A-in-TMEM/QDO 零拷贝三条未重新独立验证（引用的是约束包既有结论），与本组提案无交互，维持原判但标注"未复核"。

## 组内排序（修正收益/风险）

1. **S2**：≈零风险纯减法，−0.1~0.25µs + 4KiB slack + 寄存器缓解；先做（一天）。
2. **S1**：唯一作用于 post-v11 pacer（环 ~4.3）的杠杆，修正收益 −0.3~−1.2µs/tile；中等工程风险（释放点表含 S2 交互项、IKET 31 名额、避免退化成 splitpds）；前置=v11 实测确认环为 pacer。
3. **S3**：30 分钟微测后仅在 S1 落地且残余环成为 pacer 时并入；否则只取退化半份（免 CTA0 一跳）+ W18 释放。
4. **S4**：保持否决。

合成预期下修：三者合计对节拍 −0.4~−1.5µs/tile（提案原称 −1.2~−2.3 高估，主因 stmatrix/v11 已先吃掉环的 1-1.5µs）；<8.09ms 仍必须叠加条件 4（F≤16µs，非本视角）且 v11 的 drain 3.3-3.7 兑现。

关键引用：`KDIR/dsa_bwd_sm100_2cta_v9_3.py` L90-131（DSM helper：mbar 与 dst 同 rank 映射）、L11313-11378（SMEM 精确账）、L11693-11764/L12222（ds_image 唯一消费）、L11998-12009+L12719-12748（S2 字节恒等的双重代码证据）、L12308/L12952/L13091（pds 单级/ACQ/commit）、L13329（tail 后 release）、L13862-13959（grads 序：pds wait→dQ→relay→4 pass）、L13709-13729（W18 relay）；git `e82fb19`/`b3c7464`/`2b905f6`（v9.4 回滚与 v11 de-spill）、`32fdec5`/`e2f3756`（CONFIG B 教训与 IKET=31）；`..._splitpds_9110e23_report.md`（P 释放点=final dV pass，17.994ms）；PTX `09_instruction_set.md` :9752-9767/:9797/:9807/:9814/:9834/:9887/:10412/:10437/:10439。

---

# 总路线图（五视角判决合成）

**回答"2-CTA 框架内能否超越 baseline (7.995-8.09ms)"：可以，但余量 ≈ 0.5-1 µs/tile，且有两个必须同时成立的胜负手。**

账目对称性：2-CTA 从 baseline 手里拿走的结构红利只有 dKV 原子减半（发射下限 4.4→2.2 µs/tile）
与 K gather 减半（baseline 处本就不是墙）；新付出 DSM 交换、P/dS 三份镜像、relay、
round 供给、双倍固定成本。追平线 ≈ 4.1-4.3 µs/tile（F≈15-16µs 时）。

## 执行序（按信息量/收益/风险排序）

| # | 动作 | 预期 | 前置/门 |
|---|---|---|---|
| 0 | **实测 v11**（寄存器再平衡已写好, HEAD 14657ec 128/128 探针） | drain busy 5.9→3.3-3.7µs, release ~8.0-8.5ms | SASS 门: drain 窗 LDL+STL≈0 |
| 1a | dq_done 提前 commit + pds producer_tail 后移（~10 行） | −0.33~0.44 ms | 无; 验收 TAIL 24.5→~17µs |
| 1b | dQ epilogue TMA 化（staging=死掉的 score_kv 32KiB） | 与 1a 合计 −0.42~0.53 ms | b244255 正面先例; epi 必须走 16-DP T2R+向量 store（SASS 门） |
| 1c | 删 ds_xchg（DSM 源改 ds_image slab, 字节恒等已双重代码证明） | wall≈0, +4KiB SMEM slack | build-time assert + dense 一票 |
| 2 | **pds 环手术**（post-v11 主墙 ~4.3µs）: 四路 commit-gated 裂解 + DSM/commit 迁 W18 + relay 拆双相 | 环 4.3→~3.3, 节拍 −0.3~1.2µs/tile | v11 实测确认环为 pacer; 与 1c 交互: ds_image 释放点必须后移到 relay wait 之后 |
| 3 | (可选) persistent 2-token cluster | −0.55~0.66 ms | 仅当节拍已破 ~4.3µs; OWN_HALF_BULK 程序序约束 |

落点估计：0+1 全兑现 ≈ 7.5-8.0ms（贴线）；+2 兑现 ≈ 6.9-7.5ms（0.87-0.94×）。
若 v11 实测 drain 仍 >4.3µs（v6 先例提示的下行风险），框架接近死刑——应触发诚实的架构终审。

## 击毙清单（对抗验证后不要再走）

- K_dQ 复用 score-K via DSM：v9.2 二分实锤 peer push 自身 +6.4ms，同引擎同协议先例
- gen/pass 重排 & panel TMA 并飞：cluster 级 4-TMA 链对排列不变（tmapair −2.49% 回滚先例）
- kdq 移交 gather / W19 双 producer：WAIT_dQ=0.05µs 纯 slack + 3-producer 相位风险
- CONFIG B 复活（任何形态）：128×50ns 引擎串行 + 进环成本, v9 实测 +12.7ms
- WG-per-slot 独立 drain 流：重造 v11 刚治好的寄存器溢出病
- 两 kernel 拆分：sum(2.9+2.45)>max, 复核 11.14ms, 维持否决
- panel 常驻 SMEM：304 KiB > 227 KiB, 无 descriptor 逃逸口
- N-split drain 重划：tcgen05.ld 仅本 CTA, TMEM 跨 CTA 不可达
- earlyp 同机制的"P 先行"：leader 环入口需要 dS 全像, 提前 P 无解锁效果
- LOAD_STATS 优化 / 波次调优：全程 ≤16µs wall, 伪优化
