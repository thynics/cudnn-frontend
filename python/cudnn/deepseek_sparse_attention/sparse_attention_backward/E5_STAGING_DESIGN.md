# E5 设计：K-staging 前置化双内核架构（超越 baseline 的候选路径）

日期：2026-08-11。基座：e4ca（9.563 / ratio 1.168，出借 + plain-parity 门）。
目标缺口：ratio 1.168 → <1.0，需 −0.85µs/tile 以上。E 战役已证明微手术
（门控/出借/重排）单刀量级 0.1-0.5 且受池守恒制约；本设计动供给架构本身。

## 核心判断（trace 与台账支撑）

这台机器所有残余等待的共同祖先是**稀疏 gather 的速率（3.2µs/tile）与
其租约串行链**：
- score_kv 三租户分时（K/loan/借住）的全部复杂度源于 K 装填慢且占地久；
- E1 流式死于"生产者比消费者慢 5 倍"（0.8 vs 0.15 µs/chunk）；
- dV_r0p0 饥饿（+0.58）源于 gather 班子的串行档期；
- 冷启动 K bubble（~3.9µs/token）源于装填不可提前跨 token；
- 台账三钥匙之一"加槽资金"死于 SMEM 无源——但若 K 供给变成 TMA，
  租约链熔断，槽的需求本身消失。

**把 gather 从主内核的每 tile 关键路径上整体移除，上述五项同时解锁。**

## 架构

```
[前置生产者] topk 索引 → 稀疏 KV 行 gather → GMEM 连续暂存
             staging[wave][token][topk, D512] bf16（tile 内致密化，洞已消除）
[主内核消费] K(t) 装填 = 一条 TMA box（~0.2µs，硬件引擎，零 gather 班子参与）
```

- **分波流水**：staging 按 wave（~74 cluster 并发额度）分块生产，
  生产 wave w+1 与消费 wave w 重叠（多 stream 或 persistent-producer）。
  staging 驻留 = 每波 74 × topk×512×2B ≈ 150MB（batch 全量 8GB 不可行，
  分波必须）。
- **主内核简化**：kscore 租约 → 平凡双缓冲 TMA（8KB chunk 级都行，
  TMA 速率 ≫ 消费，E1 流式的前提第一次成立）；gather 4 warp 转岗或减员
  （寄存器池 +4096 → leader 48-reg 悬崖解除）；loan/借住机制可保留
  也可被更深的 ring 取代（SMEM 预算因 score_kv 简化而松动）。
- **holes/lengths 免费优化**：致密化在生产者侧完成，主内核不再见 -1。

## 账（理论池，逐项对应实测靶点）

| 项 | 来源 | 估值 µs/tile |
|---|---|---|
| dV_r0p0 饥饿（crew 档期消失） | e4b/e4c trace 实测 | 0.4-0.58 |
| 租约/货架裁缝空间（K 供给自由化） | 台账 shelf/lease 分析 | 0.2-0.4 |
| 冷启动 K bubbles | r4 冷窗实测 3.9µs/token | 0.12 |
| E1 流式复活（S 与 TMA 真重叠） | E1 判决的翻案条件 | 0.1-0.3 |
| gather 减员 → M2 复活可能 | M2 吸收判决前提变化 | 0-0.3 |
| **合计** | | **0.8-1.7** |

ratio 1.168 − (0.8~1.7)/6.09 → **0.97~1.06**。中值压线 <1.0。

## 生产者成本核算（成败关键）

纯带宽 gather：每 token 读+写 2×2MB，4096 token ≈ 16GB ≈ 2ms@8TB/s。
**不重叠则全盘皆输**（+2ms >> 全部收益）；重叠后主内核是延迟/发射束缚
（HBM 利用率低），净成本预计 <0.2ms。风险度量：先做 5 行的独立
microbench（gather 内核裸跑 + 与主内核并发跑各测一次）再动主内核。

## 里程碑（每步独立可测）

1. **E5a probe**：独立 staging 生产者内核 + microbench（裸速率、与
   final 并发时的干扰税）。判读门：并发干扰 <3%。
2. **E5b**：主内核 K 路径改双缓冲 TMA（staging 由 E5a 离线预生产，
   perf-only 判读，correctness 用预生产 staging 保证逐位）。
   判读门：−0.5µs/tile 起步。
3. **E5c**：波间流水（stream 分块或 persistent producer）→ 端到端。
4. **E5d**：在松动的 SMEM/寄存器预算上复检 E1 流式、M2、环深。

## 已知风险与止损

- staging 内存与波调度复杂度（batch 形状多样性）；
- 生产者内核自身的 launch/同步开销（每波一次）；
- 若 E5a 干扰税 >3% 或 E5b <0.3µs/tile：归档，结论回到
  "SM100 超越 baseline 不可达"的台账判定，本设计作为 Rubin 资产。

## 终审（2026-08-11，E5b-0 硅上判决：架构死刑）

E5b-0（kscore→PipelineTmaUmma，staging 离线预生产）三点 token 扫掠，
correctness 三点全过（dq 逐位 0）：

| tokens | staging 尺寸 | e5b/e4ca paired ratio |
|--------|-------------|----------------------|
| 8      | 16 MB（L2 热） | **1.016** |
| 74     | 148 MB（≈L2 边界） | 1.071 |
| 512    | 1 GB（全冷流）    | 1.092 |

**判决与归因**（单调梯度即 L2 签名）：

1. **供给协议本身 ≈ 免费但也 ≈ 零收益**：L2 热点（t8）残差 +1.6%，
   其中 ~1.1% 是 α（会话门）被 revert 的已知账（e4b/e4ca=1.011），
   TMA 换型净值 ≈0。即：**crew gather 从来不在 pacer 关键路径上**
   ——六定律①的又一次应验（E5b 成功移除了 crew 串行，什么也没买到）。
2. **staging 的内存经济学是根本性反转**：kv 本体只有 4–8 MB，
   **永远 L2 常驻**——e4ca 的"慢 gather"读的是 L2 热行，从未付过
   HBM 代价；而 staging 把稀疏选择展开成稠密流（tokens×topk×D），
   摧毁了跨 token 的行复用，每字节强制付 HBM 写+读一个来回。
   冷流税 = +5.5~7.5%，且 E5c 波间流水无法治愈（148 MB/波 > 126 MB
   L2，写后读窗口内必被逐出）。
3. **E5c/E5d/kdq-staging 连坐死刑**：一切"把 K/kdq 供给搬出 crew"
   的方案，上界收益 = t8 残差所示的 ≈0，而 staging 变体还要倒贴
   内存税。E5a 干扰探针的盲区：它测了生产者对 backward 的干扰，
   没测消费者读冷 staging 的税——本次补齐。

**战略后果**：供给侧重构全线关闭。ratio<1.0 只能从 pacer 串行环
本身（S→dP→grads 依赖链、tcgen05 commit/wait 跳数、reduce/原子路径）
下刀——与 vre_1 的 v12 pacer 判决同构：工作量手术全出局，只能动环。
