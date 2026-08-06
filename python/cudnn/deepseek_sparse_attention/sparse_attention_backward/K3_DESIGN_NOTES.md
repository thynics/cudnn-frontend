# K3（round 环深 2→3）设计审计（2026-08-06，基于 vk_2 实码 + vk_2_trace 定价）

结论先行：K3 不是"只治 0.45/gen 平台"的小刀——**深度 3 使被顺序不变性定理
杀死的两种重叠全部合法化**，W17 链 6.92 → ~4.3-4.6，period 直接落到
math-bound（~5.9-6.0）。它是 K 系列的主钥匙。唯一障碍是 16KB 资金，
实账确认内部为零，来源只有 panel 池（vre_3 方向）或 Rubin。

## 一、资金实账（SharedStorageV2 逐项求和，不再靠记忆）

| 字段 | 字节 |
|---|---|
| stationary_q / stationary_do | 65,536 × 2 |
| score_kv | 32,768 |
| round_buf_a / round_buf_b | 16,384 × 2 |
| p_blocks 8K + p_xchg 4K + ds_image 8K + ds_blocks 8K + ds_xchg 4K | 32,768 |
| stats 512 + mbars ~300 + 对齐 | ~2,000 |
| **合计** | **~230,200** |

预算 232,448 ⇒ **slack ≈ 2.2KB**。第三槽需 16KB（gen 尺寸 = [H64×D128] bf16
= 16KB，由 DKV_MMA_TILER 固定，不可半槽）。**内部无资金，确认。**

## 二、K3 解锁什么（与旧认识的关键区别）

顺序不变性定理（K2 桌面判决）在深度 2 下成立的根源：任何时刻只有一个
生产者能写环。深度 3 打破它，一次解锁**三个**重叠：

1. **kdq fill 与火车重叠（K2'' 合法化）**：gen n 的 acquire 从等 n−2 改为
   等 n−3 ⇒ kdq(下一对) 的信用在本窗火车**中段**（leader pass 5/6，
   ≈T+3.7-4.2）就绪。W17 在火车 gen3/gen4 之间插入 [acquire×2 + barrier A]
   （等待 ~0-0.5），gather 的 1.4µs 填充与火车尾并行，下窗头 barrier B
   残余 ≈ 0。**ROUTE_K 2.04 → ~0.3-0.5**。
2. **火车 lag-1 → lag-2**：TMA 完成等待（MAT_WAIT 实测 1.368/tile）在
   双飞行下压到 ~0.5。**火车 4.88 → ~4.0**。
3. **leader 侧 0.45/gen 平台 → ~0.25**（ve_1 已实证批处理下 0.42→0.20 的
   同类机制；这项在 W17 去饱和后属于二阶收益）。

合成：W17 链 ≈ 4.3-4.6 ⇒ **period 7.089 → max(math 5.84, W17, drain)
≈ 5.9-6.0**，按 vg_4 实测汇率 ≈ **−0.85~1.0ms e2e**。
叠加 M2（SOFTMAX 2.047→~0.95，math 链 5.84→4.7）：period → ~5.2-5.5，
**合计 −1.3~1.5ms ⇒ 9.90 → 8.4-8.6 = baseline 带**（E1：baseline 被钉在
8.4-8.6）。超越再看 D1/K5。

## 三、机械清单（vh_1 资产回收）

vh_1 的环深 3 机械全部可直接回收（当时死于 score_kv 段环资金，不是环机械）：
round_mbars 4→6、round_tma_mbars 2→3（per-slot tma_phase 列表）、
round_buf_c 字段、静态槽位映射。差异一处：vk_2 每 tile 8 gens，
8 mod 3 = 2 ⇒ 槽位映射非 tile 不变（PHASE_DYNAMIC_INDEX 死刑先例）
⇒ **9-gen 周期（8 真 + 1 pad）**，pad 由 leader 每 tile 退役一次
（acquire→commit→consume→release，信用账 9=9）。vh_1 用的是 12=12 的
同款审计。

## 四、风险面（预登记）

1. **到货边一票否决**：若资金来自 panel 时分/流式，被逐出段的 S/dP A-read
   到货边必须先过纸面（A-chunk TMA 完成 < S_ISSUE(t+1)）；
2. pad gen 信用平衡（9=9 机检）；
3. W17 火车中段插入 acquire/A 的三路径（prologue 首对无前窗——同步会合一次；
   尾 tile 不发 A；tile_count==1 退化路径）；
4. 深度 3 下 kscore/loan 相位不变（它们不在 round 环上）✓。

## 五、资金选项分析

| 选项 | 可行性 | 备注 |
|---|---|---|
| 内部 slack | **无**（实账 2.2KB） | 已关闭 |
| M3 重布局省 8KB + slack | 10.2KB < 16KB | 不够且重构级 |
| **panel 时分/流式（vre_3）** | 待其线程验证 | 见下两问 |
| Rubin | 确定 | 架构答案，K3 机械直接平移 |

**panel 残留分析**（本审计新增）：两块 panel 的每一字节每 tile 都被
S/dP MMA 读（A 操作数全量消费），**静态逐出不存在**；时分逐出则与环槽的
近全窗占用直接冲突。因此 vre_3 若可行，其形态几乎必然是
**A 操作数按 D128 chunk 每 tile 从 GMEM 重取**（A-chunk = 16KB 恰好
= 槽尺寸；W17 的 peer 半区象限 + loan dO_r0 已经在做同类 GMEM 每 tile
重取 ~96KB/tile/CTA，panel 流式再加 +64~128KB —— L2 带宽税是真实的）。
**给 vre_3 线程的两个关键问题**：
1. 它逐出的到底是什么（MMA A-staging 流式化，还是别的形态）？
2. 它的 A-chunk 到货边在新 period（~5.9）下是否闭合？

## 六、行动序

1. **问 vre_3 状态与设计形态**（当前最高价值未知量）；
2. vre_3 可行 ⇒ K3 rev（机械照 §三，资金按其形态接入，预登记门：
   correctness 4/4；candidate < 9.55 达标；< 9.75 部分；≥ 9.90 证伪回退）；
3. vre_3 死/停滞 ⇒ 降级打 **K2-swap**（−0.3~0.5ms，K2_DESK_NOTES 末章），
   K3 机械封存等 Rubin；
4. M2 终裁独立并行——K3 落地后它的转化率 ~1:1，两刀合计通向 baseline 带。
