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

---

## 实现进度（2026-08-06，vh_1 建造中）

**已落地**（`dsa_bwd_sm100_2cta_vh_1.py`，py_compile 过，DSL trace 未过）：

- host：score_b layout 由 K_CHUNKS 段改 `K_SEG_STAGES=2` 分段；断言 ≤8192；
- V2 常量：`ROUND_STAGES=3`、`ROUND_GENS_PER_TILE=12`、`K_SEG_STAGES=2`；
- 存储：score_kv 16384→8192 elem（捐 16KB）、新增 `round_buf_c`、
  kscore_mbars 2→4、round_mbars 4→6、round_tma_mbars 2→3、loan_tma_mbars 删除；
- 视图/片段：loan_quad 全删、`round_quad` 三元组、`quad_fragment_c`、
  dot/qt 的 buf-c TMA partition、`s_dq_epi` 改骑 round_buf_a；
- gather 分支**整段重写**为段流（4 段/tile 逐段 acquire/fill/commit +
  每 tile 一次 kdq 会合），`_load_score_kv_segment` 新增（沿用 vg_4 索引预取）；
- grads head 的 dV(r0) 两 pass 改读 slot 2/0（`quad_fragment_c` / `quad_fragment_a`）。

**未完成（下一步接手清单，按依赖序）**：

1. **leader 的 S/dP 逐 chunk 交错**：`_issue_score_v2` / `_issue_score_chunks_v7`
   目前一次性跑完 4 chunk 且假设 K 全驻。需改为
   `for c in 0..3: kscore.consumer_wait → S-gemm(c) → dP-gemm(c) → kscore.consumer_release`，
   ACCUMULATE=False 仅 c==0；两个 done pipe 在 4 段全发完后各 commit 一次。
   注意：leader 现在持 **两个** kscore consumer state（S/dP 共用同一段，单 state 即可）。
2. **W17 round 循环改 12 gen**：dO_r0 对回环（原 loan 位）、pad×2、
   TMA mbar 按 slot mod 3、lag-1 commit 保持；`t_*_smem_c` 已就绪。
3. **grads tail 的 4 个 pass** 按 slot map 换 fragment（当前仍是 a/b 交替）。
4. **epi-safe 门迁移**：原由 gather 在 `pipe_kscore.producer_tail` 后 arrive，
   现 staging 在 round_buf_a/b ⇒ 改由 **W17 在 `pipe_round.producer_tail` 后** arrive
   （math 侧 wait 点不变，`loan_epi_safe_mbar` 字段名保留）。
5. 尾/单 tile 路径复核（gather 已统一，无分叉；leader/W17 的 tile_count==1 分支需重看）。

**风险提示**：kscore 现为 2-stage、每 tile 4 gen（4 mod 2 = 0，相位安全）；
leader 必须**每段 release 一次**，否则 gather 在第 3 段 acquire 上死锁。

---

## Trace 采集 review（2026-08-06，上机前）

**查了四项，两项发现问题并已修**：

| 项 | 结论 |
|---|---|
| IKET 名额 | 31 个，与 vc_2/vg_5 完全相同，**零新增**（工具链上限内，vc_2 trace 已验证 31 可用） |
| span 基数契约 | MAT_QDO **2/tile**（vd_1 正是死在这里：只发 1 → aggregator 报 observed 32 vs expected 64）；ROUTE_K 1、LOAD_K 1、S_ISSUE/dP_ISSUE 各 1、dVdK_ISSUE 8、dQ_ISSUE 2 —— 全部与 vc_2 一致 |
| **等待落在 issue span 内**（已修） | 见下 |
| **ping-pong 分支在最内层**（已修） | 见下 |

### 修复 1：kscore 等待与 issue span 的分离

段环让 leader 每 tile 等 4 次 K（vc_2 只等 1 次且在 span 外）。初版把 4 次全包进
S_ISSUE/dP_ISSUE —— **供给饥饿会伪装成"发射变慢"**，正是 harness 契约
（"pure issue ranges must remain separate from intervening waits"）点名禁止的形态。

改为 v7 的 D-half 约定：段 0 的等待**提到首个 span 之前**（tile 头供给缺口 = 前置 gap），
段 2 的等待**夹在两个 span 之间**（= inter-span gap），段 1/3 各留一个在自己 span 内
（对称，流式消费不可避免）。**两个名字的语义随之改变**：在 vh_1 里
`S_ISSUE`/`dP_ISSUE` 表示交错 pass 的**两个 D 半区**（每个含两个平面各一半的 atom），
不再是"S 平面 / dP 平面"——读 trace 与跨版本比较时必须按此口径。

### 修复 2：累加器 ping-pong 分支上提

初版在**每个 atom** 上做 `producer_state.index == 0` 的运行时测试
（4 chunk × k_blocks × 2 = 数十次/tile）——v3/v4 的血账正是"完全展开的发射体
把 leader 撑爆寄存器、每 atom 发射成本三倍"。已上提到 **segment 层**：
两个平面共用同一条 2-stage 管线、同拍 acquire/commit，故 stage index 恒相等，
**一次测试同时选中两个累加器**（8 次/tile）。

### 未加 span 的已知盲区（供读 trace 时注意）

pad 代的 wait/release 不在任何 span 内。W17 对 pad 是 acquire→立即 commit（零工作），
正常不阻塞；若环满导致阻塞，会表现为 grads 尾之后一段**无归属的 gap**。
