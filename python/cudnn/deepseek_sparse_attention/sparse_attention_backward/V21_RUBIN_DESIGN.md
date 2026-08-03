# v2.1（Rubin 容量优化版）设计规格

基底：dsa_bwd_sm100_2cta_v12.py（锚点血统）。目标硬件：Rubin（假定每 SM SMEM
328KB，单 CTA 可用按 327KB = 334,848B 计，落地时按实际工具链修正）。

## 两把钥匙（对应定速环两堵墙）

### 钥匙一：补给环深度 2 → 5（供应墙）
- 相位法则：10 gen/tile mod 5 = 0 ✓（合法静态深度集 {2,5,10} 的成员）；
- 环 5 × 16,384B = 81,920B（+49,152 vs v12）；
- gen→slot = flat 序号 mod 5，tile 不变：kdq 对占 slot 0/1，八块面板占
  slot 2,3,4,0,1（同 tile 内先后复用 0/1，信用管理天然覆盖）；
- 提前量 4 gen × 0.43µs ≈ 1.7µs ≥ 信用环圈时 0.48µs → 供应等待消失，
  dVdK 回到 exec 节拍（~0.1µs/pass）；
- round_tma_mbars 2 → 5（按 slot 索引），每 mbar 每 tile armed 2 次，
  相位翻转节奏与 v12 的 mod-2 版同构；
- W17 生产循环推广到 mod-5（滞后提交结构保留——深环下它不再关键）；
- fragment 视图 round_kd[0..4] / round_quad[0..4] 编译期静态绑定。

### 钥匙二：P/dS 双缓冲（math 墙的锚点边）
- pipe_pds num_stages 1 → 2；P 区（blocks+xchg）与 dS 区（image+blocks）
  按 tile 奇偶双份：+24,576B（ds_xchg 死字段就地退役，不复制）；
- 布局静态性：tile 奇偶 = 周期 2，**leader/math/W18 三个 tile 循环做 2-tile
  宏展开**（偶体用 buf0 视图、奇体用 buf1 视图，全部编译期绑定；尾 tile 单独
  处理）——规避动态槽索引（PHASE_DYNAMIC_INDEX 判死先例）；
- 语义：math(t+1) 在 grads(t) 排空前即可向另一套缓冲发布 → 锚点边消灭
  （E3 已证单侧手术只会搬家，必须 P/dS 全双份——本版有字节，直接双份，
  不需要 v19 的腾挪三件套）。

## 字节账（每 CTA）

```
v12 基线                         230,912
− 旧环 (2×16,384)                −32,768
+ 新环 (5×16,384)                +81,920
+ P/dS 第二套 (12,288+20,480
  −ds_xchg 4,096 退役×2)          +24,576
+ 新增 mbar（round×3、pds×2 对）    ~+200（pad 内）
──────────────────────────────────────
≈ 304,700B = 297.6KB  ≤ 327KB（余 ~30KB）
```

SMEM 断言：上界式 + 实际值回显（E3 教训），双档：
`RUBIN: ≤334,848`；`B200_COMPAT: ≤232,448`。

## 预期落点（沿用已标定环模型）

period = max(math 墙 4.4-4.8[若叠 E2 形态则取下沿], 排空 4.35, 供应-exec ~2.5)
+ 接缝 ~0.6 → **~5.0±0.3**（Blackwell 口径数字；Rubin 的 CUDA core/张量核
提速另算）。供应墙与锚点边双消后，定速权移交 math——后续叠 E2/Rep8 即主战场。

## 配置开关（import 时读 env）

| 开关 | 默认 | 含义 |
|---|---|---|
| `DSA_V21_B200_COMPAT` | 0 | 1 = 退化 (环2, pds1, cap 232,448)，语义≈v12，用于在 B200 上回归验证推广后的机器正确性 |

默认即 Rubin 构型；在 B200 误跑会被 SMEM 断言以明确信息拦下。

## 验证计划（当下能做 vs 待 Rubin）

1. **现在（B200）**：COMPAT=1 过 smoke（正确性 4/4 + release ≈ v12 的 11.65）
   ——证明"推广到 N 槽/2-tile 宏展开"的机器在 N=2/单缓冲档没有破坏语义；
2. **现在（零 GPU）**：Rubin 构型 host 断言全绿 + py_compile + DSL 8 条整块审计；
3. **待 Rubin 工具链**：改 cap 常数与 arch target，跑 smoke + trace（验收签名：
   dVdK per-pass 间距 0.43→~0.15；MATH_STORE 与 G9 解耦）。

## 已知不适用项（防复议）

FP8 面板（ISA 互斥）、混 cta_group（PTX 禁令）、own-half 零拷贝（描述符
同地址）——容量解决不了这三条，v2.1 不碰。
