# v3.1 (v19-MAXFUSE) 手术规格 —— 实现基准文件

基座：v18a（双 ready W18 / pds_ds 管线 / dQ 前置 leader，correctness 4/4 硬件背书）。
文件：`dsa_bwd_sm100_2cta_v31.py`。旗标：`DSA_V19_CHASE` / `DSA_V19_DBUF`（默认全开；全关 = v18a 逐位同构）。
批准案：CG2终局设计_20260731.md。本文是其 8 件必做的落地语义，含从源码逐字节重推的数据流向。

## 0. 数据流向基准（从 v17a/v18a 源码重推，实现时不得偏离）

- P/dS 像 [own-H64 × kv64] COL_MAJOR（h 最快）；字节序 = kv 半幅分块：bytes[0:4096)=kv半0、[4096:8192)=kv半1。
- owns_n 语义：`n_owner = 片段 kv 半幅索引`；`owns_n = (n_owner == rank)`。owns 警组自然落位于 blocks 的 `block[rank]`；
  非 owns 警组经 `−n_owner*4096B` 偏置落位于 xchg 基址。
- **DSM 方向（关键，两处调用参数序 = (src_local, dst_at_peer)）**：
  - P：src = 本地 `p_xchg`（非 owns 警组写入 = 本像 kv半(peer) 内容），dst = **peer 的 `p_blocks[sender_rank]`**。
  - dS（v12 P1b 现状）：src = 本地 `ds_image ± 半幅`（rank0 取 +4096B 即 kv半1；rank1 取 +0），dst = peer 的 `ds_blocks[sender_rank]`。
- 消费端：`p_fragments/ds_fragments = make_fragment_B(blocks[0]/[1])`；pass+0/+2 用 block0（收缩 H半0），pass+1/+3 用 block1。
  两 CTA 同名 block 同偏移（CG2 同地址规则天然满足）；内容互补：block[own-rank]=本地写，block[peer-rank]=对端落地。
- `dq_ds_fragment = make_fragment_B(ds_image)`（整像，dQ 的 B）。

## 1. DBUF 手术件（DSA_V19_DBUF）

### 1a. struct
- `p_blocks`: 4096→8192 元素（**奇偶双份**，每份 4096 元素含 2 块）；`ds_image`: 4096→8192 元素（奇偶双份）。
- `ds_blocks` **删除**：块视图改为 `ds_image[parity] + b*4096B` 别名（字节恒等由现存 :12120-12131 断言背书）。
- `ds_xchg` **复活**（2048 元素 ×1）：出站推送暂存（P1b 反演）。`p_xchg` ×1 不变。
- 新 mbar：`xchg_read_mbars[1]`（读释放）；`pds_mbars` 2→4、`pds_ds_mbars` 2→4（num_stages 1→2）。
- `stats` 迁至 struct 首字段（offset 0，天然 1024 对齐，免 Align 槽）；mbar 群随后；头部合计 ≤1,024。
- 字节账（DBUF+CHASE）：131,072 + 24,576 + 32,768 + 16,384 + 4,096 + 16,384 + 4,096 + 头 1,024 = **230,400**。
  断言：上界式 + 实际值回显（禁精确断言，E3 教训）。

### 1b. math 发布（每 tile，parity = t%2）
- 目的地指针全部按 parity 选择：`p_blocks_raw + parity*4096元素`、`ds_image_raw + parity*4096元素`。
  实现模式 = **动态 if 双份视图**（`_issue_score_v2` 乒乓先例 / v2_1 "整套双份"先例）：预建 parity0/1 两套
  store tile 视图，`if parity == 0: copy(...p0视图) else: copy(...p1视图)`。
- P 分支不变（owns→local[parity]，非 owns→p_xchg）。
- dS 分支重写：owns 警组的 `ds_local` 块写 **删除**（own 半经整像别名免费获得）；非 owns 警组 **新增**
  `ds_xchg` 写（复活 v17a 死代码 `t_rs_ds_xchg_tile` 机器，v18a 已删须重建）；整像写保留（→ ds_image[parity]）。
- math 写 `p_xchg`/`ds_xchg` 前：等 `xchg_read_mbars` phase(t-1)（首 tile 免等，phase 初值处理）。
- 双 ready 机器保持 v18a：早腿 arrive `pds_ready[0]`（整像写+fence 后）；晚腿 arrive `pds_ready[1]`（P + xchg 写+fence 后）。
- `pds`/`pds_ds` producer_acquire 换 2-stage state（松弛 = dkv(t-2)-done）。

### 1c. W18 relay（v18a 程序 + 三处新件）
顺序（每 tile）：
1. 早腿：等 `pds_ready[0]` → `pipe_pds_ds.producer_commit`（feeds dQ(t)，**必须在 dq_done 等待之前**——FATAL-1 反环）。
2. 等 `pds_ready[1]`（晚腿）。
3. **dq_done(t) 落地门（新）**：观察式等待 dq_done full mbar 相位（不消费，vm5probe 观察式先例）——
   保证两 CTA 的 dQ(t) MMA 硬件完成（AsyncUmma commit 完成沿、CG2 簇可见），对端整像可被覆写。
4. P 推送：src `p_xchg`，dst peer `p_blocks[parity(t)]` 块[rank]（expect_tx 远端武装不变）。
5. dS 推送（dst 改向）：src **`ds_xchg`**（不再取活像半幅），dst **peer `ds_image[parity(t)] + rank*4096B`**（= 别名块位）。
6. `pipe_pds.producer_commit`。
7. **读释放（新）**：`cp.async.bulk.wait_group.read 0`（PTX wrapper，:1568-1578 休眠先例，rev0 SASS 验证）→
   arrive `xchg_read_mbars`。
8. 落地转换（relay_mbars，dS 先）不变。

### 1d. leader
- dQ 前置保持 v18a（`_issue_prev_dq_v18a`，门 = pds_ds）。
- **pass 重排（FATAL-2 必做）**：dVdK 8 pass 从 (p,p,ds,ds)×2 轮改为 **dV 四连发先行 + dK 四连发殿后**：
  head(r0): p0,p1 → tail(r1): p0,p1 → head2(r0): ds0,ds1 → tail2(r1): ds0,ds1。
  dS 首消费从全局第 3 位挪到第 5 位（1-based），给 dq_done→落地→消费链让出窗口；
  dkv 槽/accumulate 标志随重排重接（r0/r1 槽序保持，仅 B 种类重组）。
  预注册窗口断言：dS 落地完成沿 − dS 首消费沿 ≥ 0（trace 探针位）。
- `pds`/`pds_ds` consumer state 换 2-stage。
- 消费端 fragment 按 parity 动态 if 选择（两套预建）。

### 1e. W17 gen 重排（配合 pass 重排）
gen 顺序从 `kdq,kdq,dO(r0h0),dO(r0h1),Q(r0h0),Q(r0h1),dO(r1h0),dO(r1h1),Q(r1h0),Q(r1h1)`
改为 **`kdq,kdq,dO×4（r0h0,r0h1,r1h0,r1h1）,Q×4`**（dO 象限先行喂 dV 四连发）。
静态序改动仅在 flat_gen→(kind,round,half) 的映射表；延迟提交格与相位律（10 mod 2=0）不变。

## 2. CHASE 手术件（DSA_V19_CHASE，独立旗标，rev1 隔离验证）
- `score_kv` 16384→12288 元素（3 chunk 驻留）；chunk3 静态映射至 slot0（追逐）。
- 消费端 `_issue_score_chunks_v7`：chunk 0-2 走 3-chunk 片段，chunk3 走 slot0 别名片段（const_expr 分支）。
- 生产端 `_load_score_kv`：拆两趟——首趟 chunk0-2，尾趟 chunk3→slot0，尾趟须等 S 与 dP 的 chunk0 消费完成
  （门 = s_done/dp_done 既有完成沿的观察式等待；rev0 窗算已过，chase 2/4 同族）。
- `dq_epi` staging（别名 score_kv 死区）从 [H128,D128] 单波改 **[H64,D128]×2 波**（24,576B 容量适配）。
- S/dP 最小交错序（接缝缓解）：S0,dP0,S1,S2,S3,dP1,dP2,dP3 —— 在 `_issue_score_chunks_v7` 调用序上实现。

## 3. 守恒与审计门
- IKET 静态名 ≤29 清点（新增探针位不得越界）。
- staged 块全改动过 DSL 8 条约束整块审计（流程律）。
- 全关旗标 = v18a 逐位同构；DBUF 单开、CHASE 单开可独立 bisect。
- 正确性红线：dq_done 观察门失效 ⇒ 对端 dQ 未完成即覆写整像 = 静默数值损坏（holes/lengths 用例重点）。

## 4. 已知欠账（审计面板必查）
- read-release PTX 方向（`wait_group.read`）无 DSM 生产先例——rev0 SASS 必验（批准案原文）。
- 观察式等待 dq_done 的相位推导（首 tile / 尾 tile 边界）。
- parity 双份的动态 if 在 math 8 警组的寄存器压力（v9.3 后 math 144 regs 地板，STACK 预算）。
- chunk3 追逐的尾块（tile_count 边界）行为。
