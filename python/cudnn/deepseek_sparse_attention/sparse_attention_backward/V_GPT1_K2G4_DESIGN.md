# v_gpt_1（K2-G4）设计：kdq 稀疏 gather 换 TMA tile::gather4

**版本线**：v_gpt（独立于 vk_*，来源=外部推理 agent 提案）｜ **基座**：vk_2
**日期**：2026-08-06 ｜ **状态**：桌面推导完成，待容器侧两项核查后开写

## 0. 一句话

把 kdq 镜像（[N64×D128]×2 round，dQ 的 A 操作数）的逐行 `cp.async` + 160 线程
kdq_barrier 双会合，替换为 W17 单 warp 发射的 TMA `tile::gather4`；
ring 协议、leader 调度、MMA 描述符、字节内容全部不动。

## 1. 桌面推导结果（本地已闭合的门）

### G1a 指令通道 ✓
指令存在且形态确认（CUTLASS `SM100_TMA_LOAD_2D_GATHER4`）：
```
cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4
    .mbarrier::complete_tx::bytes[.L2::cache_hint]
    [smem], [desc, {col, row0, row1, row2, row3}], [mbar](, hint);
```
坐标序 = **列坐标在前，4 个行坐标在后**（NVIDIA 论坛staff 确认 + CUTLASS 一致）。
本仓库已有 `llvm.inline_asm + @dsl_user_op` 裸 PTX 先例（`_cpasync_bulk_s2cluster`），
指令发射无障碍。

### G3 源几何 ✓（比预期更简单）
host 把 mKV 重建为 `(S_kv, D512, (1,1))`、批模 stride=(0,0)（vk_2:341-346）——
**无 batch 维**。tensormap = 纯 2D `[S_kv 行 × D512 列]` bf16，行 stride =
mKV.stride[0]（运行时值，测试负载 512）。列坐标单位 = 元素。

### G2 目的布局：SW128 坐实，朴素形态否决，SW128 形态推导闭合 ✓

- `dq_a_layout_staged` = make_smem_layout_a(MN-major, bf16, [M=D128, K=N64])
  → cutlass 4.5.0 选择 **MN_SW128**（2048 bits % 1024 == 0，源码逻辑核实）。
- 驱动约束：swizzle≠NONE 时 box 内维 ≤ swizzle 尺寸（128B）
  ⇒ **提案原文的 16×[4行×256B] 非法**，必须切半宽。
- **实测布局（R4 容器 compile 探针，2026-08-06）**：
  `S<3,4,3> ∘ 0 ∘ (((64,2),16),1,4,1):(((1,4096),64),0,1024,0)`，cosize 8192 元素。
  化简（注意 (16,4):(64,1024) 因 16·64=1024 合并为 n·64）：
  ```
  addr(d, n) = base + Sw₃,₄,₃( 2B·(d mod 64) + 128B·n + 8192B·(d÷64) )
  ```
  即**半平面主序**：d 半块 0 的 64 行（128B/行，连续 8KB）在前，半块 1 在后。
  （我最初推导的 1KB/2KB 原子交错式是错的，以实测为准——实测形态对 gather4
  更有利：行距恒 128B、行组永不跨结构边界。）
- TMA 的 SW128 XOR 按**绝对 SMEM 地址位**作用（与 CuTe Swizzle 同构），
  box 内行距 = box 内维字节数 = 128B，恰等于半平面内行距 ✓。

**最终几何**：每 round **32 笔 gather4**（16 个行组 × 2 个 d 半块），每笔
[4 行 × 64 elem(128B)] = 512B：
```
col(round r, half h)   = 256·r + 128·rank + 64·h     （元素，tensormap dim0 坐标）
smem_base(n0, h)       = slot_base + 8192·h + 128·n0 （字节）
rows                   = topk_idx[n0 .. n0+3]（无效行钳位，见 §2.3）
```
每 tile 2 round × 32 = 64 笔，expect_tx = 16384B/round。发射成本 ~64×几 ns，可忽略。

### 约束核对清单（驱动 API 文档）
- boxDim[0]=64 ≤ 256 ✓；64×2B=128B 是 16B 倍数 ✓ 且 ≤ SW128 ✓；
- 目的地址均为 128B 对齐（slot 1024 对齐 + 偏移皆 128B 倍数）✓；
- oobFill=zero（TMA tile 模式 OOB 读零填）——gather4 继承 tile 语义（见 R1）。

## 2. 实现形态（单杠杆）

### 2.1 host
- 增加一个 gather4 专用 tensormap：优先路径 = 用现有 `cpasync.make_tiled_tma_atom`
  按 [64 elem × 4 row] box + SW128 对 mKV 2D 视图正常编码（gather4 复用 tiled
  编码的 rank-2 map，无独立 encode API），kernel 内取其描述符指针发裸 PTX。
- 其余 host 不动。

### 2.2 kernel
- **W17**：在现 kdq 会合位置改为：16 lane 各持 4 个 topk 索引（vg_4 手法批量
  预取）→ 每 lane 发 2 round × 2 half = 4 笔 gather4（lane 内串行、warp 内 32
  笔并飞）→ 按现有 ring 协议 expect_tx/等待/提交 g0/g1。
- **gather 警组**：4 个 `_gather_kdq_v8` 调用点全部退役；kdq_barrier 退役。
  LOAD_K 路径一行不动（单杠杆纪律）。
- **leader / relay / math / reducer**：零改动。

### 2.3 无效行处理
holes（索引 -1）与尾 tile（global_n ≥ topk）统一钳位：
`row = valid ? idx : S_kv`（S_kv 必 OOB → 零填）。每索引一条 select，
不依赖"负坐标零填"这个文档未明示的行为（R1 的保险）。

## 3. 残余风险核查结果（容器侧，2026-08-06，任务 dsa-vgpt1-g4-desk-r1）

| # | 结果 |
|---|---|
| R1 | **PASS**：OOB_ZERO tensormap 下越界行零填（signed s32 坐标 + OOB 通则 + gather4 继承 tiled 规则的组合推论，非逐字明示）。§2.3 的 S_kv 钳位保留作保险。 |
| R2 | **PASS**：ISA 明示 boxDim[1] 必须 = 1（不是 4）。tensormap = rank-2、interleave 不支持、swizzle 未被禁止、dim0 box = 行长。 |
| R3 | **PASS（无原生封装，显式描述符路径可行）**：cutlass **4.5.1**（容器实际版本）Python 包只有内部 IR 枚举 `TmaLoadMode.gather4`，无公开 atom。可行路径：`cpasync.copy_tensormap(tma_atom, tensormap_ptr)`（公开导出，helpers.py:396-413）把 atom 描述符复制进显式 128B workspace 指针 → 指针作 kernel 参数 → inline asm（"l" 约束，CCCL `cp_async_bulk_tensor_tile_gather4` 签名佐证）。禁读私有 `_trait.value`。 |
| R4 | **实测与推导不符，已按实测重推**（见 §1）：实际为半平面主序，smem_base = slot + 8192·h + 128·n0。 |

**新增实现风险（R5）**：host 侧 `make_tiled_tma_atom` 是否接受 [64×1]-box + SW128
的 smem 布局来产出 boxDim=[64,1] 的 tensormap——需 compile 探针；不行则退
interface 层 `cuTensorMapEncodeTiled`（cuda-python bindings）手工编码 128B 描述符。

## 4. 预登记判据（沿用外部提案 + 本线补充）

- correctness 4/4（lengths/holes 是 OOB 路径的主考场）；
- 机制门：ROUTE_K（更名 KDQ_G4）≥30% 下降；WAIT_dQ 不再贴 kdq commit；
- 性能门：candidate_ms 与同日 ratio 同向改善 ≥0.1ms（vk_2 = 9.902@8.609 对照）；
- ROUTE_K 大降但 release null ⇒ 停，转 pacer 判读（MAT_QDO / PDS）；
- 死锁面：W17 新等待 = gather4 mbar；若挂，首查 expect_tx 字节数与实际笔数乘积。

## 5.5 双视图桌面探针终判（2026-08-06，probe_vgpt_dualview r1/r2）

- **S2（score_kv→dq-A own-half alias，K2b）：真 FAIL，永久关闭。**
  score_kv 按 [own-n32×d64] 4KB 块装（d-半块 stride 2048 元素），dq-A 规范形态
  要求 4096；仅 (d<64,n<32) 象限字节重合。K2 可建形态的 own 半永久落在 S2S 拷贝腿。
- **S1（panel→dkv-A 零拷贝视图）：布局锁 PASS，描述符锁 FAIL。**
  字节层恒等成立（4 窗 × 8192 点零失配，偏移 = 4096·(d0/64) 元素；修正 V2
  tiler K=H64 后 dkv-A 与 dq-A 同构）。**但消费不可行**：CG2 单描述符地址在
  两 CTA 各自 SMEM 同偏移解释，每个 h-pass 两 CTA 都需要该 h 的字节——owner
  侧在面板、peer 侧在环槽，地址不对称即违反描述符锁。面板改 d-split 可解 dkv
  侧却破坏 score-A（CG2 的 K 维须每 CTA 全量持有 D512）。**own-h×full-D（score
  要求）与 own-d×full-h（dkv 要求）在 128KB/CTA 下不可调和；全量面板需 256KB
  = Rubin 容量。** 面板代的"视图化删除"在 SM100 上关闭；SM100 剩余面板杠杆
  = owner-push（方向 B，改运输不删代）。
- 方法论备注：本探针只测了四锁链的第 1 锁（布局）就冠名"主钥匙"，第 2 锁
  （描述符 rank 对称）纸面即可先判——**四锁链应按锁序过，先纸面后探针**。
  布局恒等这个事实本身归档备用（d-split 重分区/方向 C/Rubin 形态直接受益）。

## 5.6 矿脉 G/B 桌面判定（2026-08-06，probe r3）

**矿脉 G（粒度环）——紧凑形态死于布局锁，视图形态存疑待 S5**：
- S3/S4 FAIL：make_smem_layout 的 K=32 紧凑 staging 与 K=64 staging 结构不同
  （A 侧 d-半块 stride 2048 vs 4096；B 侧 swizzle 退化 SW128→SW64）。
  "同字节自然承载两代描述符"不成立。
- 幸存形态：**视图描述符**——K=32 的 MMA pass 直接走 K=64 布局的 h 子窗
  （strides 不变、shape 截短、start_addr 平移 h0·64 元素；LBO/SBO 与 K=64
  描述符相同）。canonical 合法性待 S5（mma_desc 对视图布局的接受性）。
- 但成本清单在增长：own 半填充退化为 2×4KB 碎片（K=64 布局里 h 半区不连续）
  ⇒ 每 tile 引擎操作 6→24 笔；v_gpt_1 验尸刚证明引擎小事务排队是真实毒性
  （4KB 好于 512B，但计数 ×4）。混合粒度信用（kdq 沿 n、面板沿 h）设计复杂度高。
- **处置：挂起**。S5 探针便宜可发，但即使 PASS，工程复杂度/风险比已不优于矿脉 B。
  待 B 落地后按剩余等待再评。

**矿脉 B（owner-push）——全门已过，晋升为下一个 kernel rev（v_gpt_2）**：
- 布局锁：S1 恒等 = owner 面板窗与 ring 槽内容布局全同 ⇒ **DSM 推送是平坦
  16KB bulk 拷贝，零转换**；
- 描述符锁：ring 槽 rank 对称不变（只换填充来源）✓；
- 字节锁：零新增（mbar 沿用 round_tma/round mbar 结构，DSM 落地 arm 用
  arrive_and_expect_tx 现成手法）✓；
- 协议锁：跨 CTA 信用可见性由 UMMA-tracked release 多播保证（v2 文档已核）；
  死锁面 = 对称推送互等，三路径等待图（稳态/首 tile/tile_count==1）建造前必画；
- 证据基础：v_gpt_1 验尸的 L2 压力（REDUCE_ATOMIC 1.8→3.49）+ 每 CTA 每 tile
  省 48-64KB GMEM 往返；填充延迟 DSM ~0.7µs/16KB vs GMEM TMA 1.5-2µs。
- 预期：MAT_WAIT/供给 L 下降 + L2 让位原子；预登记门照单杠杆纪律另立。

## 5. 提案对照记录（外部 agent 方案的三处修正已另行回复）

R3-Late 的 16KB union 地址不连续（ds_blocks 隔断）；PANEL-MCAST 几何不成立
（A 是 M-split，两 CTA 需要不同字节）；DQ-Ready 预期按 vg_1 A/B 证据减半。
本文件只承载 K2-G4 主刀。
