# vre_3 手术规格（S2 环深 + dOᵀ 流式 + B-lite）— 2026-08-05

## r2 修订（r1 correctness 尸检 + 架构修正）——本节为准

**r1 判负根因（dense 70% 污染）**：r1 把 dOscore 放进主环并让 leader 以
[S, dP, grads, dq] 序消费——但 FIFO 消费指针是单一序列，leader 迭代 k 的
dP(k) 在 grads(k−1) 之前执行时，指针停在上一组的 qdo 处 ⇒ **dP 读到象限
数据**。"消费滞后一拍自洽"论证是错的（r1 头注里的推理作废）；且任何把
dP 排到迭代尾部的修法都会让 [dP→math→pub→grads] 串进约束环（period ~7，
更差）。**单环 + dP 流式不可两全，dP 必须离开主环。**

**r2 架构**：
- **dos 专用管道**：dP 的 score-A 走独立的 2 槽 dos 缓冲（2×16KB，
  `dos_buf` + `pipe_dos` 2-stage + 专用 TMA mbar），chunk 粒度 TMA
  （新 host atom：`make_tiled_tma_atom_A(op, mdO, score_a_layout,
  score_tiler, score_tiled_mma)`，16KB box）。dP fragment 用 2-stage
  dos 布局，chunk c 静态取 stage c%2（槽复用，描述符=槽地址，天然正确）。
  leader 程序序回到 v12 形态 [S, dP(dos), grads, dq尾, pads]——dP 保持
  头部，math 边不入环。
- **主环 12 gen**：[qdo×8 | kdq×2 融合 | pad×2]，12 mod 4 = 0；槽位映射
  与 r1 相同（dO→a/b、Q→c/d、kdq→a/b、pad→c/d）。消费严格滞后一组
  （iter 0 零消费，iter k 消费 group k−1，TAIL 收 group T−1），kdq 零移位。
- **SMEM 精确闭合**：225.5 = v12 225.5 − dO panel 64 + 环 c/d 32 + dos 32。
- W17 每迭代：[dos×4 chunk fill（头部，接受首 tile dP ~1µs 一次性延迟）→
  qdo×8 → kdq 融合 → pad×2]。
- 信用算术（r2 复核）：f0(组k) 等 kdq(组k−1) 的 dq 读（iter k ~+3.8）⇒
  grads(k) 于 iter k+1 +1.4 全预填；kdq(组k) 等自组 f4 释放（+2.6）⇒
  dq 尾部停顿 ~0.5（预算内）。预期 period 4.5-5.0 不变。

以下 r1 原文中与本节冲突处（dOscore 入主环、16 gen、整板 TMA、消费滞后
论证）一律以 r2 为准。

基线：`dsa_bwd_sm100_2cta_v12.py`（9bc35b3）→ 新文件 `dsa_bwd_sm100_2cta_vre3.py`。
判据来源：vre_1 分解（period 6.948；LOAD 纯工作 2.80/等待 4.00）+ S1 证伪的
FIFO 信用算术（饱和 2 深单 FIFO 上消费序置换一阶不变；环长 = pds 边 + N×节拍 + 尾部填充）。

## 目标与预期

深度 2→4 直接打 4µs 等待池（节拍 0.45→~0.3）；dOᵀ panel 流式化腾出 64KB
（32KB 给两个新环槽，~32KB 余量）；B-lite 把 pds 边缩 ~0.2-0.3。
**预期 period 6.95 → 5.3-5.6**（kdq 保持融合握手的上沿；解冻 kdq 留 vre_3b 再拿 ~0.4）。
判读阈值按 5.3-5.6 设，避免假阴性。

## 每 tile 16 个 gen 的 FIFO 序（唯一活序，勿改）

```
位置  0-3   dOscore c0..c3   16KB TMA     消费者: dP(t+1)（唯一不等 pds 的消费者，必须排头）
位置  4-11  qdo f0..f7       16KB TMA/bulk 消费者: grads(t) 8 pass
位置 12-13  kdq r0/r1        16KB gather   消费者: dQ(t)（尾置）
位置 14-15  pad ×2           空 gen        W17 acquire+立即 commit；leader wait+立即 release
```

16 mod 4 = 0 ⇒ 槽位映射全部编译期静态：slot = 位置 mod 4。
dOscore c→槽 c；qdo f→槽 f mod 4；kdq r0/r1→槽 0/1（dq_kd fragment 绑定不变！）；pad→槽 2/3。
反例算术（已证死，勿回退）：dOscore 不排头 ⇒ 全链重新 convoy；kdq 排头 ⇒ 回到 6.9 原地。

## SMEM 账（预算 227KB）

删 stationary_do（−64KB）；round 区 2×16KB→4×16KB（+32KB）。合计 ~193.5KB ✓。
struct 变更：删 `stationary_do`；`round_buf_c/d` 新增（各 8192 elem，Align 1024，
紧跟 a/b 声明保证 a..d 物理连续——dOscore 的 4-stage 视图依赖此连续性）；
`round_mbars` 4→8；`round_tma_mbars` 2 不变（在飞 ≤2 的 1-behind 纪律不变，
mbar 按位置奇偶轮转；每 tile 用 mbar 的 gen 数 = 12，mod 2 = 0 相位稳定）。

## 刀口清单

1. **常量**：`ROUND_STAGES = 4`。
2. **storage struct**：如上。
3. **视图构造**（kernel 头部）：
   - 删 stationary_do / stationary_do_tma 两个 get_tensor；改为在 round_buf_a
     基址上用 `make_tensor(recast_ptr(round_buf_a_raw, inner, dtype), outer)`
     重建同布局视图（score_a_layout_staged 4-stage ≡ 4 环槽；
     stationary_a_layout_staged 同理，供 TMA partition）。cosize 32768 = 4×8192
     恰好铺满 a..d ✓。
   - `round_kd` 不变（绑 a/b = 槽 0/1）。
   - `round_quad` 2 元组→4 元组（c/d 用 recast_ptr 模式复制 a/b 的构造）。
   - `quad_fragment` a/b→a/b/c/d 四个。
   - `t_qt_smem_*` / `t_dot_smem_*` partition ×2→×4（c/d 复制）。
   - raw ptr：round_buf_c_raw/d_raw 新增。
4. **prologue（W17）**：只载 Q panel（一次 TMA + ready arrive）；dO 侧
   TMA/wait/arrive 全删；`stationary_ready_mbar+1` 路径废弃（leader 首个 dP 的
   gate 变成环 gen wait）。
5. **W17 主循环**（每 tile，按 FIFO 序）：
   a. dOscore c0..c3：acquire → mbar(pos%2) expect_tx 16384 → `cute.copy(tma_atom_do,
      t_do_gmem[None, rank, c], t_dos_ring[None, c], ...)` → 1-behind wait+commit。
      span 复用 LOAD_QDO，payload = loop_iter+1。
   b. qdo f0..f7：现有代码，仅两处改动——目标槽/裸指针/分区按 f mod 4 选
      a/b/c/d；**dO 的 OWN_HALF_BULK 分支删除**（源 stationary_do 已亡，两个
      h-half 都走 GMEM TMA）；Q 侧 bulk 保留（源 stationary_q 仍在）。
   c. kdq：现有融合握手**原样整体后移**至 f7 之后（两 acquire → 双 barrier
      会合 → 两 commit；本轮不解冻）。
   d. pad ×2：acquire → elect_one commit，无填充无 mbar。
6. **leader**（对照 S1 死稿的正确部分复用）：
   - 每迭代顺序：S(t+1) → **dP(t+1) 流式**（新 helper `_issue_score_streamed_vre3`：
     4 chunk 循环 {ring consumer_wait → 该 chunk 的 gemm 链 → UMMA-tracked release}，
     dp_done commit 在第 4 chunk 后；fragment = ring 基址 stationary 布局的
     make_fragment_A，chunk c 静态取 stage c）→ grads head/tail（不含 dq，
     head 中 `_issue_dq_rounds_v2` 调用与 commit_dq 块删除）→
     `_issue_dq_rounds_v2`（消费位置 12/13）→ pad wait+release ×2 →
     **pds release（必须在 dq 之后**，dq 读 dS_H，此 release 是对 math(t+1)
     发布的 WAR 门）→ TAIL 加 dq_done commit（早提交形态废，纪录一次性
     ~2.5µs/token 代价）。
   - grads head/tail 内 4 个 pass 的 quad fragment 轮转 a,b,c,d（原 a,b,a,b）。
   - 首 tile：dP(0) 无 stationary gate，直接 ring wait（环冷启动信用天然满足）。
7. **relay B-lite**：pds commit 从两次 DSM send 之后**移到 pds_ready wait 之后、
   send 之前**（commit 语义 = 双 CTA math 已发布；send 落地另有 landing mbar
   把关，互不依赖）。省 relay 发射段 ~0.2-0.3µs 的 pds 边。
8. **span/payload**：无新 IKET name。dVdK/WAIT_dQ/dQ_ISSUE/MAT_QDO/ROUTE_K
   payload 律不变；trace 若因 31-name 上限编不过，本轮判据 = correctness + perf。

## 正确性红线（复核清单）

- pds release 在 dq 之后（S1 教训 #1）。
- 每 tile producer 16 acquire/commit = consumer 16 wait/release；producer_tail 不变。
- dO 双半区 TMA 后，expect_tx 仍 = grad_a_stage_bytes（16384）× 每 gen 一次。
- kdq_barrier 会合次数/相位不变（gather 侧零改动）。
- quad c/d 视图与 a/b 字节布局完全同构（同 layout 不同基址）。
- dq_kd/quad 槽 0/1 复用：kdq gen（位置 12/13→槽 0/1）与 qdo f4/f5（槽 0/1）
  之间隔 f6/f7/pad——深度 4 下位置 12 acquire 等位置 8 release ✓ 无别名风险。

## 判读（proxy 三件套）

correctness 4/4 硬门；perf 同节点 v12 对照：**candidate < v12×0.87（≈period ≤6.0）
⇒ 方向成立保留；< v12×0.80（≤5.6）⇒ 达标**；≥ v12 ⇒ 证伪，trace 尸检先查
dP 流式段（S/dP 头部新增 ring 等待）与 kdq 尾。
