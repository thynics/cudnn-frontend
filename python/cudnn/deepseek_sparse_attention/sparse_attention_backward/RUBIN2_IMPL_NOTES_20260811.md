# rubin_2 实现笔记（2026-08-11）

交付文件：`dsa_bwd_sm100_2cta_rubin_2.py`（由 rubin_1 复制后手术；类名合同
`FlashAttentionDSABackwardSm100TwoCTAV2` 保持，harness 以字面名 getattr —
见 `benchmark/dsa/sweep_topk_2cta.py:41,107`）。
设计规格：`RUBIN2_dO全量面板设计_20260810.md`。
`python3 -m py_compile` 通过。**所有行号均指 rubin_2 新文件。**

---

## 0. 核心裁定：主路径（同字节全量面板 + 静态 recast 视图）被桌面代数否证，
##    改用等价的「pantry」实现（非 §5 降级退路；6 代环等全部结构收益保留）

### 0.1 否证推导

CG2 共享描述符语义（同址锁）：tcgen05 2-CTA MMA 的 A 操作数按 M 均分到两
CTA，指令只带**一个** SMEM 描述符地址，每个 CTA 在**自己**的 SMEM 里、按
**同一本地地址**取自己的 M-half。证据链（全部来自文件实际内容，非记忆）：

1. rubin_1 环协议硅上已证：`round_buf[slot]` 两 CTA 同地址、各自装各自
   M-half（GMEM 侧 `rank_g_dot = rank_dkv_mma.partition_A(g_dot)` 是
   rank-切片分区），`make_fragment_A(round_quad[slot])` 的 fragment 布局
   cosize=8192（仅 per-CTA 半份），描述符遍历只覆盖本地 16KB。
2. kdq gather 的 `d_offset = 256*r + 128*rank`（`_fill_kdq_pair_*`）直接
   写死了 dq-MMA A 操作数的 M-split：CTA c 持 D[256r+128c, +128)。
3. `V_S1_SPEC.md` 定理 1 的框架同一口径：dkv-A 的 own-chunk 集合为
   rank0={c0,c2}、rank1={c1,c3}（c_k = K-major 面板的第 k 个 D128 chunk，
   chunk 号 = 2r+c），同址锁要求二者在**同一地址**供货。

由此，若两 CTA 持**相同字节**的全量面板（规格 §1 的表述），对 dV pass
(r,h) 的任意静态偏移 X：CTA0 在 X 读到的是 chunk 2r（正确），CTA1 在同一
X 读到**同样的** chunk 2r 字节，但硬件把它当作 M 行 [128,256)（应为
chunk 2r+1）——dV 的 D 高半行会用低半行的 dO 值计算，**静默错数**。
同理，规格/配方中「dP 视图偏移 = rank * 半面板字节数」在共享描述符下也
不成立（描述符只取 leader 的偏移 0，CTA1 会重复供 h0 半区）。

结论：「identical bytes + 单一静态视图」无法从布局代数确证——而且可被
上述推导直接否证。配方第 5 条的触发条件（"若无法从布局代数确证偏移"）
成立。

### 0.2 采用的实现：全量面板的 rank-有序两半区（"pantry"）——优于 §5 退路

不必退到 §5（dO 回 2 代环）；存在一个**可证明**的全量面板实现，兑现主
路径的全部结构收益（6 代环、dV-A 永生零拷贝、dO 每 tile 运输为零）：

`stationary_do` 字段 32768 → **65536 元素（128KB）**，内部两半区：

- `[0, 32768)`：本 rank H64 半区（dP 的 A 视图，偏移 0，两 CTA 同地址、
  各持己半——与 rubin_1 的 stationary_do **逐字节同语义**，视图构造代码
  一行未改）；
- `[32768, 65536)`：**dV-A pantry**：4 个 8192 元素槽（q = 2r + h），
  槽 q 以 `dkv_a_layout_staged` 形态存放**本 CTA** 的 dO.T (r,h) 象限
  M-half——即 rubin_1 里 ring TMA `t_dot_gmem[None, r, h]` 每 tile 送进
  round_buf 的**同一来源、同一布局**的字节，现在每 token 只送一次。

每个 pantry 槽 = 一个"钉死地址的 ring 槽克隆"（同布局对象、同 swizzle、
同 TMA partition 构造），因此其 fragment 对同址锁的满足方式与硅上已证的
环槽完全一致：两 CTA 同地址、各持己半。dV 四个 pass 直接消费永生
fragment，无任何管线等待/释放。

与规格文本的偏差（审计必读）：

| 规格/配方原文 | 本实现 | 原因 |
|---|---|---|
| 两 CTA 持相同字节 | 两 CTA 持**相同并集**（全 H128 的信息量），字节排布 rank-有序（own-half 语义 + own-M-half pantry） | §0.1 否证 |
| dP 视图偏移 = rank*半面板 | 偏移 = 0（两 CTA 同址各持己半，= rubin_1 原样） | 同址锁 |
| dO stationary TMA 两次 copy | 1 次半区 copy + 4 次象限 TMA（合计 128KB/CTA，"tx_count 翻倍"应验：65,536 → 131,072 B） | 象限粒度才能装 pantry |
| （若退路）dO 保留 2 代环 | 未采用：pantry 全额兑现主路径，无需退路 | §0.2 论证 |

字节账与规格 §2 逐项相符：1,024(mbar) + 65,536(q) + **131,072(dO)** +
32,768(kv) + **49,152(环 3×16KB)** + 28,672(PDS 单面) + 512(stats)
= **308,736 B ≤ 334,848**（存储断言带实际值回显，行 4244-4254）。

---

## 1. 配方逐条落点（行号 = rubin_2 新文件）

| 配方条目 | 落点 |
|---|---|
| 1. 复制 + 类名保持 + 文件头 docstring 重写 | 文件头 1-46；类定义 3942；V1/V0 别名保留 7254-7259 |
| 2a. env 开关删除（COMPAT/E3PAD/REG_K1A/SPIN_K2 及读取、`import os`） | 模块头整段删除（grep 全文仅 docstring 第 5-6 行以文字提及）；`_nanosleep` 删除；`_k2_consumer_wait/_k2_producer_acquire/_k2_mbar_wait` 三 helper 删除，调用点还原为直呼（5679-5683, 5728, 6556, 6737/6743/6749, 7110, 7218） |
| 2b. 常量固定 | PRODUCER_REGS=64 / REDUCE_REGS=120（3972-3973；kernel setmaxregister 用之：5661, 5666）；MAX_SMEM_BYTES=334_848（4037）；PDS_FACES=1（4035） |
| 2c. struct 只留单面变体、无 e3pad | 单一 `SharedStorageV2`（4176-4242）：round_mbars[6]、round_tma_mbars[3]、landing[2]、relay[2]、pds_mbars[2]、pds_ready[1]、stationary_do 65536 元素、round_buf_0/1/2 |
| 2d. 双面代码臂删除 | `_issue_dkv_pass_faced` 整体删除（改直呼 `_issue_dkv_pass_v2`）；`_issue_dq_rounds_v2` 去面化（4699-4746）；math 发布单面（6113-6127）+ pds_ready 单 mbar（6136-6139）；relay 单面（6691-6712）；f1 fragment 别名块删除（5424-5433 只留单面名） |
| 3. 环 10→6 代、深 3 | ROUND_GENS_PER_TILE=6 / ROUND_STAGES=3（4017-4018）；PANEL_SLOTS 字面量 (2,0,1,2)（4026-4031，避开类作用域 genexp NameError 先例）；round_mbars=6、round_tma_mbars=3（struct 4181/4190；init 循环 5534/5561 按常量自适应）；W19 三个逐槽相位（6726-6728）；W17 生产循环 8→4、tensor_kind 恒 Q、grad_round/h_half 由 flat_gen 推导（6581-6586），Q 的 OWN_HALF_BULK 臂原样保留（6600-6679） |
| 4. stationary_do 全量化 | 字段 65536 元素（4202-4205）；dO 腿 tx 翻倍 = `score_a_stage_bytes*K_CHUNKS + 4*grad_a_stage_bytes` = 131,072 B（6473-6480）；半区 TMA 原样（6486-6491）+ 4 次 pantry 象限 TMA（6493-6520）；dP 的 A 片段字节语义不变：`stationary_do` 视图/`score_do_fragment` 构造行**零改动**（视图读字段前 32768 元素 = rubin_1 原样） |
| 5. dV-A 零拷贝 | `pantry_quads`：`cute.make_tensor(cute.recast_ptr(stationary_do_raw + OFFSET(r,h), dkv_a_layout_staged.inner, dtype), dkv_a_layout_staged.outer)`（5086-5104）；`dv_a_fragments = make_fragment_A(view)`（5418-5423）；pantry TMA partition（5372-5391）；host assert 见 §2 |
| 6. leader 梯度块 | head：dV r0 两 pass 无任何等待/释放（6909-6924），dQ（kdq 代）与 dK（Q 代）管线序保持，consumer 推进 8→4（head 4 + tail 2 + dQ 2 = 6/tile）；head 6840-6961、tail 6964-7030；调用点 6308-6371（稳态）/6383-6445（末 tile）；kscore 单租户未动 |
| 7. 存储断言 | 上界式 + 值回显（4244-4254），预期 ≈308,736 B（~309KB） |
| 8. IKET provenance 保留 / loan 残留清零 | `IKET_V2_NATIVE_PROVENANCE`（4057）；grep "loan" 全文 0 命中 |

---

## 2. OFFSET 推导与 host assert 清单

### 推导

- rubin_1 own-half bulk 注释与代码（rubin_1:6970-7027）给出：K-major
  [H64,D512] 半面板中，(r, c) 的 dkv-A M-half = 连续 16KB chunk，元素偏移
  `4096*(4r+2c)` = `8192*(2r+c)`，即 chunk 号 2r+c——rank c 的 own-half
  bulk 只在 h==c 臂出现，h≠c 走象限 TMA；二者硅上字节恒等（生产路径）。
- 全量面板下 4 个 (r,h) 槽位偏移（本实现，pantry 形态）：
  **OFFSET(r,h) = 32768 + 8192*(2r + h)** 元素（bf16，×2 = 字节）。
  槽内容 = 本 CTA 的 (r,h) 象限 M-half（chunk 2r+rank of 半区 h），由
  `t_dot_gmem[None, r, h]` TMA 直接以 dkv_a 布局写入——**无需任何 recast
  恒等假设**（与环槽构造逐同）。
- 填充↔消费映射：q0=(r0,h0)→head dV pass1、q1=(r0,h1)→head dV pass2、
  q2=(r1,h0)→tail dV pass1、q3=(r1,h1)→tail dV pass2（W17 6493-6520 ↔
  leader 6325-6326/6357-6358）。

### host assert（`_specialize_shared_storage`，4133-4171）

1. `PANTRY_SLOT_ELEMENTS == cute.cosize(dkv_a_layout_staged)`（槽=一个
   dkv-A 象限镜像）；
2. 槽字节数 == `size_in_bytes(select(dkv_a_layout_staged,[0,1,2]))`
   （= grad_a_stage_bytes = 16,384 B，与 TMA expect_tx 同源）；
3. `PANTRY_BASE_ELEMENTS == cute.cosize(score_a_layout_staged)`（dP 半区
   恰占前半，字节语义 = rubin_1 stationary_do）;
4. `32768 + 4*8192 == 65536`（四槽恰满后半，无重叠无尾巴）；
5. 每槽基址 1024B 对齐；
6. 继承自 rubin_1 的既有布局断言全部保留（cosize 上界、
   `score_b == 2*dkv_a` 等，4113-4126）。

### 就绪协议（无新 mbar）

pantry 4 次 TMA 与 dO 半区共用 `stationary_tma_mbars+1`（tx 翻倍），
W17 等完后向 leader 的 `stationary_ready_mbar+1`（count=2，双 CTA 到达）
到达——首个 dP 之前 pantry 在**两个 CTA** 都已就绪，而首个 dV pass 严格
晚于首个 dP（rotated schedule），无新协议面。pantry 此后只读（dP 半区与
pantry 无写者重入；Q own-half bulk 只读 stationary_q）。

---

## 3. 采用退路/偏差的点

- **未采用 §5 字面退路**（dO 保留 2 代环 + 本地全量 bulk）：pantry 实现
  以同等 SMEM（131,072 B dO 面板）兑现主路径全部结构收益，且每个部件都
  落在硅上已证机制上（象限 TMA、环槽 fragment 构造、count-2 ready mbar）。
  若审计坚持 §5 字面形态，改法：恢复 dO 四代入环（gens=10，深度需回 2 ——
  深 5 超 SMEM 上界），dO 臂两 rank 均走本地 bulk
  `stationary_do_raw + 32768*h_swap + 4096*(4r+2*rank)`；代价是失去 6 代
  环与 W17/W19 收缩。
- 规格 §1 的「两 CTA 相同字节」「dP 视图偏移=rank*半面板」两句按 §0.1
  否证修订；已在文件头 docstring 显式标注（NOTE 段，行 25-30）。

---

## 4. 自查表（全部通过）

| 项 | 计数/结果 |
|---|---|
| py_compile | PASS（`python3 -m py_compile` 无输出） |
| 环代配平（每 tile） | W17 producer_acquire = 2(kdq)+4(panel) = **6**；producer_commit = W17 kdq 2 + W19 panel 4 = **6**；leader consumer_wait/release = dQ 2 + dK(head) 2 + dK(tail) 2 = **6**；round_acq/round_com/commit_com advance 各 6 |
| dV pass 环操作数 | **0**（4 pass 全部静态 fragment 直发；dkv_done acquire/commit 与 relay 门原样） |
| mbar 初始化 ↔ struct | round 管线 2×3=6 ↔ round_mbars[6]；pds 2×1=2 ↔ pds_mbars[2]；round_tma init ×3 ↔ [3]；landing ×2 ↔ [2]；relay ×2 ↔ [2]；pds_ready ×1 ↔ [1]；stationary_tma[2]/ready[2]/kdq[1]/epi_safe[1] 原样 |
| 被删 env 引用 | grep `_RUBIN1_|SPIN_K2|E3PAD|REG_K1A|B200_COMPAT|_k2_|_nanosleep|_issue_dkv_pass_faced|use_face1|grads_face|tensor_kind|round_buf_3|p_blocks_1` → 代码 0 命中（仅文件头 docstring 文字提及一次） |
| loan 残留 | grep "loan" → 0 命中 |
| PANEL_SLOTS/相位 | (2,0,1,2)：槽 2 每 tile 被踩 2 次、槽 0/1 各 1 次 → W19 三相位变量逐槽翻转（6726-6753）；6%3==0 相位律断言在类体 |
| 存储字节 | 手算 308,736 B ≤ 334,848（运行时断言回显实际值） |

---

## 5. 审计聚焦：最可能出错的三个位置

1. **pantry TMA partition 的合法性（5372-5391）**：`tma_atom_dot` 是按
   `grad_a_layout`（= select(dkv_a_layout_staged)）建的原子，rubin_1 里它
   的 SMEM 侧落点是 1024 对齐的独立 struct 字段；rubin_2 落点是
   `recast_ptr(stationary_do_raw + 偏移)` 的字段内视图。布局/对齐逐同
   （host assert 4,5 条），但「TMA 目的地允许字段内偏移指针」这一点只有
   编译腿能终审——若 descriptor 校验拒绝，把 pantry 拆成 4 个独立 struct
   字段即可（字节账不变，视图代码不动）。
2. **首 dP 就绪语义变宽（6473-6525）**：`stationary_ready_mbar+1` 现在
   同时背书 dO 半区 + pantry（tx 131,072）。协议上只推迟首 dP（一次性、
   ~64KB L2 热），但若首 tile 时序对 trace 基线敏感，需在 k5 配对腿里
   确认冷启动段无新 pacer。
3. **kdq 代 g0/g1 的 acquire 门位移**：深 3 下 g0(t) 的 acquire 等的是
   g3(t-1)（dK r0 h1）的 release；rubin_1 深 5 时等的是 g5(t-1)（dO_r1
   h0）。门更早、无环（FIFO 保持），但 gather→W17 rendezvous 的实时位置
   随之前移；若实测 kdq_barrier 变 pacer，属性能面而非正确性面。
