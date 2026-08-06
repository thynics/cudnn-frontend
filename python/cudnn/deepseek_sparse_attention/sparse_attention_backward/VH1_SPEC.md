# vh_1 spec（2026-08-06）——K 段环捐赠 + round 环深 3（gap 正面攻击第一期）

基座 vg_5（9.771 @ 8.390，ratio 1.1646）。目标：grads 到货 gap 平台（~0.4µs/gen ×8）
塌向执行节拍。资金：PIN-1 已核死的 K 段环捐赠 16KB；拍卖对手（锚点边 +16KB 需求）已被
vg_1 零字节拿走。判决库背书："深度 ≥3 是唯一能把 4L 压到 2L 的原理性杠杆，
被拍卖否决，非原理否决"。

## 四件套

1. **K 段环**：score_kv 32KB 单驻 → **2×8KB D128 段环**（4 段/tile 流经 2 槽）。
   gather 逐段填充+逐段 commit（cp_async 组按段收口）；净捐 16KB。
2. **S/dP 逐 chunk 交错发射**：段环下每段必须被 S 和 dP 都读完才可覆写，
   分离发射破流。改为 `for c in 0..3: wait seg → S-gemm(c) → dP-gemm(c) → release`，
   两累加器互不依赖（v7.py 交错先例）；ACCUMULATE=False 仅 c==0。
   tile0 的"split readiness（S 只等 Q）"微优化让位（chunk0 前须两 panel 齐）。
3. **round 环深 2→3**：新增 round_buf_c 16KB（= 捐赠字节）。
   **vc_2 的 loan 阵亡**（32KB 死区消失），dO_r0 回环：quadrant gens 8/tile。
4. **12 gen/tile 定相（2 个 pad gen）**：10 真 gen mod 3 ≠ 0 会引发逐 tile 槽位漂移
   （PHASE_DYNAMIC_INDEX 死刑先例）；补 2 个空 gen（acquire→commit 即回，
   leader 对称 wait→release），12 mod 3 = 0 ⇒ 槽位映射 tile 不变，**零宏展开**。

## 静态槽位映射（tile 不变）

| pos | gen | slot/buf | 消费者 |
|---|---|---|---|
| 0,1 | kdq r0, r1 | 0,1 | dQ（kd 视图不变） |
| 2,3 | dO_r0 h0,h1 | 2,0 | pass[0][1]（原 loan 位） |
| 4,5 | Q_r0 h0,h1 | 1,2 | pass[2][3] |
| 6,7 | dO_r1 h0,h1 | 0,1 | pass[4][5] |
| 8,9 | Q_r1 h0,h1 | 2,0 | pass[6][7] |
| 10,11 | **pad** | 1,2 | leader 即时 wait+release |

fragment：round_kd a/b 不变（kdq 恒槽 0/1）；round_quad ×3 视图（a/b/c）；
每 pass 的 fragment 按上表静态绑定。W17：TMA mbar ×3（槽各一），lag-1 commit。

## 连带手术（两处必须同刀）

- **dQ epilogue staging 重定位**：`s_dq_epi` 现骑 score_kv（需 32KB 整，断言在案）。
  改骑 **round_buf_a**（layout 跨 a+b，两字段 16384B/1024 对齐无缝隙，host 断言
  `ptr_b == ptr_a + 8192` 落死）；安全边：W17 在 `pipe_round.producer_tail` 后
  arrive epi-safe mbar（复用 loan_epi_safe 字段），math 的 epi 首 store 前 wait。
- **kscore 管线 1→2 stage**（段环）：kscore_mbars 2→4；gen 流连续跨 tile
  （4/tile mod 2 = 0，无相位问题；prologue 填段 0/1，段 2 等 leader 释放段 0，
  无死锁——启动即流水）。

## 删除清单

loan 全家：loan_tma_mbars、`_fill_score_loan_do_r0_vc2` 调用（稳态/尾/单 tile 三处）、
loan_quad fragments、leader 的 loan kscore 消费、gather 的 loan 填充段。
（vg_5 里这些是 vc_2 血统；ve_1 的 alias 机器不在本基座。）

## 预期与止损

- 账：失 loan（[0][1] 0.03/0.41 → ~0.4×2）≈ +0.4µs/tile；
  得 8 quadrant 到货被深度覆盖（L≈0.65 ≤ 2×消费间隔）≈ −1.9µs/tile（trace）；
  K 段流附带红利：S 可在段 0 就绪即发射，K-margin 链蒸发。
  净 −0.7~−1.1µs/tile（release 折扣后）→ **9.771 → 8.6~9.1ms**。
- 止损门：release ≥ vg_5（9.771 @ 8.390 漂移归一）即回退；
  correctness 红线照旧（lengths/holes 覆盖奇尾）。
- 预登记 trace 信号（工装恢复后）：dVdK gap[0..7] 均值 ≤0.15；
  WAIT_dQ 不回归；S_ISSUE 起点提前（段 0 红利）。

## 风险

pad gen 的信用节拍（+4 协议操作/tile，~0.2µs 摊销，账已计）；
三槽 fragment 静态绑定的核对表（上表即合同）；
交错发射后 s_done/dp_done 的 commit 位置（两 GEMM 全部发完后各 commit 一次，
tracked set 覆盖交错链——与 v7.py 的 pass 尾双 commit 同构）。
