# Offloading 审计报告（2026-07-31）

**问题**：v12 的 8×dVdK 供应接力 gap（3.03µs/tile，period 的 63%）能否通过 offloading 解决——
把操作数提前搬进 SMEM，或卸载到其他存储层级/计算阶段？

**方法**：11-agent 两阶段对抗工作流（3 视角生成 → 每候选 xhigh 独立证伪），全部判词落到
源码行级证据；另有 2-agent 字节账本审计。产物：本报告。

---

## 一、字节账本（源码级定稿，v12 SharedStorageV2 @ 11395-11460）

| 字段 | 字节 | 用途 |
|---|---:|---|
| mbarrier/标量头 | 284 (+740 pad) | 35×Int64 + Int32，首个 1024 对齐字段强制补齐 |
| stationary_q | 65,536 | 驻留 Q [64×512] bf16（S 的 B / own-half bulk 源） |
| stationary_do | 65,536 | 驻留 dO [64×512] bf16 |
| score_kv | 32,768 | score K stage（兼 dQ epilogue 别名） |
| round_buf_a/b | 2×16,384 | **接力环 2 槽**（10 gen/tile 流过 163,840B） |
| p_blocks + p_xchg | 12,288 | P 本地块+交换+落地 |
| ds_image + ds_blocks | 16,384 | dS 全像 + dK 的 B 块 |
| ds_xchg | 4,096 | **死字段**（P1b 已退役，留作两步计划） |
| stats | 512 (+512 尾 pad) | softmax LSE/delta |
| **合计** | **231,424 / 232,448** | **slack 恰 1,024B** |

**全常驻需求**：Q^T 65,536 + dO^T 65,536（CG2 M-split 在 D、h 是 K 维 → 每 CTA 必持有
全部 8 个象限，own-half 零拷贝已证死）+ K_dQ 32,768 = 163,840B；净增 **+131,072B**。
K_dQ 按 topk 索引逐 tile gather，**原理上不可预搬**（与容量无关）。
结论：bf16 下"提前全搬"死于算术，缺口 ~127KB vs 可回收上限 5,632B。

---

## 二、死刑清单（本轮新增行级证据）

| 候选 | 死因 |
|---|---|
| TMEM A-staging / A-from-TMEM | ① TMEM 512/512 零洞（S 0/dP 64/dQ0 128/dQ1 256/dKV 384-448，v7 连 32/96 列洞都被乒乓占用）；② 驱逐 dQ 腾列=重复记账——dQ MMA 每 tile 必须有 TMEM 落点，真实需求 768（全形态）/640（dV-only）> 512，即 D-own 死因复现；③ **新硬约束：SM100 TMEM-A 禁转置**（cutlass `mma_sm100_umma.hpp:639` static_assert a_major==K；idesc 的 a-major 位只作用于 SMEM 描述符），而 dkv A 是 MN-major 转置视图——attention bwd 的所有 A 候选（dO^T/Q^T/P^T/dS^T）都是转置视图，结构性死 |
| L2 carve-out (accessPolicyWindow) | 热集 ~18MB 散布 73 cluster 且随 token 滑动，单静态窗口不可表达；只攻 D 段（0.14）中的争用份额，2/3 gap 是信令；上限 ≤0.1µs |
| SMEM 时分复用（借 stationary/score_kv 死窗） | 无死窗：stationary 活到 flat_gen=7（S/dP A 源 + bulk 源）；score_kv 在 grads 窗口正被 gather(t+1) 重填；CG2 同偏移规则使落地砸对端活数据 |
| 跨 tile 配对 dVdK（一 fill 两耗，圈数减半） | 机制成立、非伪装死案，但强制双 tile P/dS 同驻 = **+12,288B 内禀**（TMEM-as-B 非法、peer-SMEM 不可达）> 5,632B 上限；dV-only 变体 +8,192B 也超且圈数只省 25% |
| persistent cluster 跨 token 重叠 | 实测冷 tile-1 仅 +0.5-0.6µs → 摊薄 0.015-0.08µs/tile，低于行动阈值 4-20×，换最大结构改动 |

---

## 三、幸存者：FP8 家族（"压缩即 offloading"——唯一活门）

### 1. FP8_DVDK_DEPTH4 —— **ALIVE**（唯一）
A panel（Q^T/dO^T）+ B（P/dS blocks）转 e4m3；象限 16→8KB，**32KB round 区原地重切
4 槽（零新字节）**；gen 数不变 8（v13 的 null 反成为 s≈0.21 串行地板的校准数据）；
kdq 变 pair-gen 使 12 slot-gen ≡ 0 mod 4（解掉杀死 depth-3 的相位墙）。
- 收益：raw −1.6~−2.1µs（校准排队模型 s=max(b,exec,L/k)，k 2→4，L~0.9 信令主导不随字节变）；
  realized 被 math 墙 5.5±0.3 封顶 → **−1.0~−1.25，period → 5.3-5.6**。
- SMEM **净 −12.3KB**——反哺 P/dS 双缓冲（v18a 全形态 12,288B 首次有资金）。
- 须修正两处（refuter 抓获）：① DSL 断言 a_dtype==b_dtype（`mma.py:786`）——"A e4m3 + B e5m2"
  不成立；改全 e4m3 + per-tile delayed dS scale（DeepSeek-V3 先例，恰好折进 per-tile dKV
  reducer drain）或 2 行 DSL patch；② fp8 转置 GMEM 副本不存在（mQT/mdOT 是 stride 视图），
  需新预转换 pass（几十 µs 级）。③ P 需固定 2^k scale（P~1/topk 会 flush below 2^-9）。
- **硬门（顺序）**：(a) fp8 publish stmatrix SASS 门（v9.3 病理区，失败=净负 → KILL）；
  (b) CG2 f8f6f4 MN-major A 描述符 host 编译探针；(c) 数值容差 smoke；(d) 形状 one-off
  微基准（fill 间距应保持 ~0.45，若涨向 0.6+ 则收益塌缩 → KILL）。

### 2. FP8_FULL_RESIDENCY —— NEEDS_MICROBENCH（对"全搬 SMEM"的最终答案）
全操作数 fp8 后 **panel 常驻在字节上变得可行**：新字节计划 ~171-185KB vs 232,448 cap
（~60KB 富余，可顺带双缓冲 K stage 和 P/dS）。整个 W17 供应环、round 管线、leader
的逐 pass acquire/wait/release **全部删除**——直接退出接力经济学（删环 edges 6-7 =
4.35µs），而非调参。Standalone −0.85~−1.35（period 5.2-5.7）。
- 判 ALIVE 前必须过：① **dK 的 scale 代数**——dK=Σ_h dS·Q 在 h 上收缩，per-row dS scale
  在求和号内不可折（候选自称可折是错的）；唯一修复=per-token 标量 dS scale，其 amax 是
  未来信息（保守界 10-100× 高估 vs e4m3 下溢地板）——先跑零 GPU 的离线量化仿真，不过
  容差直接 KILL 不烧 rev；② fp8 publish lowering SASS 门；③ 数值契约变更需 owner 批准
  （最大 scope）。

### 3. FP8_HFUSE_K128 —— NEEDS_MICROBENCH（退路形态）
h-half 融合 K=128：8 圈→4 圈，fp8 融合 gen 恰=16KB=一个现有槽。−0.8~−1.4 realized。
per-pass exec 不变（f8 K32/atom×4=bf16 K64×4 同 atom 数），收益全在圈数减半。
注意其 "D 段随 fp8 缩小" 子论断被 L 源不变性（0.72≈0.73）证伪——收益只来自圈数。

---

## 四、战略排序

1. FP8 家族三形态共享同一次**数值契约变更**（只能花一次）与同一批零 GPU 前置门
   （SASS/编译/离线仿真）——先跑门，再定形态。
2. **勿 standalone 花掉**：DEPTH4 standalone 只是复刻 v18a+E2 的 5.3-5.6 落点；真值在
   与 E2（v17a math 改写）组合后**穿越 5.2-5.3 供应地板**——rearchitect-P1 的 CG1 路
   被 PTX 同 cta_group 禁令判死后，这是唯一 CG2 合法的穿地板路径。
3. 若离线仿真过容差且 owner 授权最大 scope，FULL_RESIDENCY 是终局形态（删环而非调环）；
   否则 DEPTH4 是低 scope 的第一 rev（P/dS fp8 + 4 槽，Q/K/V/dO 主路径不动）。
