# rubin_3 实现笔记（2026-08-11）

交付文件：`dsa_bwd_sm100_2cta_rubin_3.py`（由 rubin_2 复制后手术；类名合同
`FlashAttentionDSABackwardSm100TwoCTAV2` 保持，V1/V0 别名保留 7423-7427）。
设计出处：`RUBIN2_IMPL_NOTES_20260811.md` 附录分支 A（半 pantry 变体）。
`python3 -m py_compile` 通过。**所有行号均指 rubin_3 新文件。**

---

## 0. 核心内容：rubin_3 = rubin_2 + Q_r1 半 pantry

Q 象限与 dO 象限同为 token 不变量。round-1 的两个 Q 象限（dK tail 两
pass 的 A 操作数）钉进 `stationary_q` 尾部的 2 个 pantry 槽，视图构造与
rubin_2 的 dO pantry **逐字节同构**（`recast_ptr(字段基址+偏移,
dkv_a_layout_staged.inner) + outer`，同布局对象、同 swizzle、同 TMA
partition 构造），因此同址锁（CG2 共享描述符 rank-same-address）满足方式
与硅上已证的环槽/dO pantry 完全一致。round-0 的 Q 象限（dK head）仍走环。

全量 Q pantry 超预算（rubin_2 附录已算死：357,888 > 334,848）；半 pantry
+ 环深 3→2（kdq×2 + Q_r0×2 = 4 代，4 % 2 == 0 相位合法）装得下：
**325,120 ≤ 334,848**（余 9,728 B）。

## 1. 配方逐条落点（行号 = rubin_3 新文件）

| 配方条目 | 落点 |
|---|---|
| 1. 复制 + 类名合同不变 + 文件头 docstring 重写 | 文件头 1-47；类定义 3999；类 docstring 4002-4023；V1/V0 别名 7423-7427 |
| 2. stationary_q 32768→49152 元素（64→96KB） | struct 字段 4310-4316；Q_PANTRY_BASE_ELEMENTS=32_768 / Q_PANTRY_SLOT_ELEMENTS=8_192 / Q_PANTRY_SLOTS=2（4127-4129，几何注释 4118-4126）；`q_pantry_quads` 视图（5218-5230，与 dO 的 `pantry_quads` 5197-5208 逐字节同构）；S GEMM 的 A（`score_q_fragment`，5556-5558）与 `stationary_q`/`stationary_q_tma` 视图（5141-5144/5149-5152）**零改动**（读字段前 32768 元素 = rubin_2 原样） |
| 3. Q pantry 装载 | TMA partition `t_qt_pantry_slots`（5530-5539，gmem 侧复用既有 `t_qt_gmem`）；两次象限 TMA `t_qt_gmem[None,1,0]/[None,1,1]` → 槽 0/1，挂 `stationary_tma_mbars`（Q 腿）（6683-6699）；Q 腿 expect_tx +2×16,384 B（6626-6634）；readiness 注释（6700-6708）——S 首发多等 32KB，属预期 |
| 4. 环 6 代→4 代、深 3→2 | ROUND_GENS_PER_TILE=4 / ROUND_STAGES=2（4076-4077，4%2==0 断言 4078）；PANEL_SLOTS 字面量 (0, 1)（4088-4091，槽复用注释 4081-4085）；round_mbars 6→4（4294）、round_tma_mbars 3→2（4303）；round_buf_2 删除（struct 4327-4334 只余 0/1；kernel 侧 tuple 5130-5137）；W17 面板循环 4→2、grad_round 恒 0（6765-6767）、尾随 advance 4→2（6861-6862）；W19 相位变量 2 个（6907-6908）、循环 4→2（6914）、slot 2 臂删除（6915-6927） |
| 5. leader 手术 | dK tail 两 pass 改用 Q pantry 静态 fragment：`dk_q_fragments`（5573-5576）；`_issue_prev_grads_tail_v2` 摘除 round_pipeline/round_consumer_state 形参与 4 处 wait/release，只返回 dkv_producer_state（7139-7199）；两处调用点改为 `dkv_prod = (...)` 并传 `dk_q_fragments[0]/[1]`（稳态 6522-6538、末 tile 6589-6605）；dK head 两 pass（Q_r0 环代 g2/g3）与 dQ 环序保持（head fn 7014 起，调用点 6486-6521/6553-6588 传 `quad_fragments[PANEL_SLOTS[0]]/[PANEL_SLOTS[1]]` = 槽 0/1） |
| 6. 存储断言 | 上界式 + 值回显（4356-4367），注释预期 ≈325,120 B；host assert：Q pantry 槽 cosize 与 dkv_a 恒等等五条，模式逐抄 dO pantry（4245-4280） |
| 7. M2 knob 原样保留 | `_RUBIN2_M2`/`DSA_RUBIN2_M2`/`_exp2_ffma_deg6` 一字未动（70/75/4102/6211/6258） |

## 2. expect_tx 账（W17 stationary 装载段）

| mbar | rubin_2 | rubin_3 | 组成 |
|---|---|---|---|
| `stationary_tma_mbars+0`（Q 腿，S 首发所等） | 65,536 B | **98,304 B**（6630-6634） | `score_a_stage_bytes*K_CHUNKS`(=4×16,384=65,536，Q 半区) + `Q_PANTRY_SLOTS*grad_a_stage_bytes`(=2×16,384=32,768，Q pantry) |
| `stationary_tma_mbars+1`（dO 腿，dP 首发所等） | 131,072 B | 131,072 B（6638-6641，未动） | 65,536(dO 半区) + 4×16,384(dO pantry) |

腿上挂的 copy 数：Q 腿 = 1（半区）+ 2（pantry）= 3 发；dO 腿 = 1 + 4 = 5
发。就绪协议无新 mbar：`stationary_ready_mbar+0`（count=2，双 CTA 到达）
现在同时背书 Q 半区 + Q pantry；首个 dK tail pass（Q pantry 消费者）严格
晚于首个 S（首个 grads 块在 loop_iter 1），协议面充分。

## 3. 环配平表（每 tile，4 = 4 = 4）

| 侧 | 操作 | 计数 | 落点 |
|---|---|---|---|
| producer_acquire（W17） | kdq g0/g1 | 2 | 6734/6736 |
| | panel g2/g3（Q_r0 h0/h1） | 2 | 6769（循环 6765 ×2） |
| producer_commit | W17 kdq（rendezvous 后 elect_one） | 2 | 6746/6749 |
| | W19 panel（等 round_tma_mbars 后） | 2 | 6929（循环 6914 ×2） |
| leader consumer_wait/release | dQ 两 round（g0/g1） | 2 | `_issue_dq_rounds_v2` 4838/4856（D_ROUNDS=2 循环） |
| | dK head 两 pass（g2/g3） | 2 | 7106/7114、7116/7124 |
| | dK tail | **0**（Q pantry 静态 fragment） | 7179-7194 无任何环操作 |

状态推进：round_acq 4/tile、round_com 4/tile（kdq 2 + skip 2）、
commit_com 4/tile（skip 2 + panel 2）、round_cons 4/tile。grep 全文
`round_pipeline.consumer_wait` = `consumer_release` = 3 处（dQ 循环 1 +
head 2），tail 0 处；`PANEL_SLOTS[2]/[3]`、`round_buf_2`、
`round_tma_mbars + 2`、`w19_phase_2` 代码 0 命中（Q_r1 环残留清零）。

## 4. 字节账（手算，断言只取上界并回显实际值）

1,024(mbar 区，36×Int64+Int32 → pad) + 98,304(stationary_q 96KB)
+ 131,072(stationary_do 128KB) + 32,768(score_kv) + 32,768(环 2×16KB)
+ 28,672(PDS 单面: 8K p_blocks + 4K p_xchg + 8K ds_image + 8K ds_blocks)
+ 512(stats) = **325,120 B ≤ 334,848**（= rubin_2 的 308,736 + 32,768
− 16,384 − 24B mbar 缩减被对齐吃掉）。

填充↔消费映射：Q 槽 0 = (r1,h0) → tail dK pass1、槽 1 = (r1,h1) → tail
dK pass2（W17 6688-6699 ↔ leader 7179-7194）；dO pantry 映射不变。

## 5. 审计聚焦：最可能出错的三个位置

1. **Q_r0 供货窗收窄（深 2 的 acquire 门位移，性能面首查点）**：
   深 3 时 g2(t) 的 acquire 等的是 g5(t-1) 的 release——整整提前一个
   tile；深 2 时 g2(t) 等的是 g0(t)——**消费它的同一个 grads 块内**的
   dQ round-0 release（producer_acquire(G) 门 = release(G-2)）。Q_r0 的
   16KB 填装（TMA 或 own-half DSM bulk）只能藏在 relay+1 等待和两个 dV
   pass 底下。FIFO+credit 无环、无死锁（每个等待都由 leader 程序序中严格
   更早的事件满足），但若 B200 实测 dK head 的 consumer_wait 出现新
   stall 平台（RK_ACQ 变 pacer），根因就是这里——这是深 2 的结构代价，
   不是 bug。kdq 侧同理：g0(t) 的 acquire 门从 g3(t-1)（rubin_2）进一步
   移到 g2(t-1)（同块内 dK head h0 之后）。
2. **Q pantry TMA 落点 = 字段内偏移指针（5530-5539 / 6688-6699）**：
   与 rubin_2 dO pantry 完全同款的悬置风险——`tma_atom_qt` 的 SMEM 侧
   落点是 `recast_ptr(stationary_q_raw + 32768/40960 元素)` 的字段内
   视图，而非独立 struct 字段。布局、swizzle、1024 对齐 host assert
   全部逐同（4245-4280），但「TMA 目的地允许字段内偏移指针」只有编译腿
   能终审；dO pantry 在 rubin_2 尚未过编译腿，rubin_3 把同一假设扩到了
   第二个宿主字段。若 descriptor 校验拒绝，改法与 rubin_2 笔记同：把
   2 个 Q 槽拆成独立 struct 字段（字节账不变，视图/fragment 代码不动）。
3. **首 S 冷启动推迟量可能大于名义 32KB（W17 copy 发射序，6643-6699）**：
   W17 的发射序是 Q 半区 → dO 半区 → 4 发 dO pantry → 2 发 Q pantry，
   而首 S 只等 Q 腿（mbar+0）。若 TMA 引擎按发射序服务，Q pantry 两发的
   完成被 dO 腿 5 发 copy（160KB）押后，Q 腿的 mbar 到齐时刻远晚于
   "+32KB" 的名义账。属一次性冷段、不进稳态；但若配对腿 trace 显示冷
   启动段异常拉长，第一刀是把两发 Q pantry copy 提到 dO 半区 copy 之前
   （纯排序手术，协议/字节账全不变）。

## 6. 自查表（全部通过）

| 项 | 结果 |
|---|---|
| py_compile | PASS（`python3 -m py_compile` 无输出） |
| 环配平 | 4 = 4 = 4（§3） |
| 相位律 | 4 % 2 == 0（类体断言 4078） |
| Q_r1 环残留 | grep `PANEL_SLOTS[2]|PANEL_SLOTS[3]|round_buf_2|round_tma_mbars + 2|w19_phase_2|g4/g5` → 代码 0 命中（仅 docstring/注释文字提及） |
| dK tail 环操作数 | 0（两 pass 全静态 fragment 直发） |
| mbar 初始化 ↔ struct | round 管线 2×2=4 ↔ round_mbars[4]；round_tma init ×2（ROUND_STAGES 自适应循环）↔ [2]；其余原样 |
| 存储字节 | 手算 325,120 B ≤ 334,848（运行时断言回显实际值） |
| M2 knob | `DSA_RUBIN2_M2` env 名与实现原样（70/4102/6211/6258） |
| 既有文件 | 未修改任何既有文件；未 git commit；新文件仅 2 个（kernel + 本笔记） |
