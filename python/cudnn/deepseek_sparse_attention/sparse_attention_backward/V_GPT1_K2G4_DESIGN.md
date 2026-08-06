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

## 5. 提案对照记录（外部 agent 方案的三处修正已另行回复）

R3-Late 的 16KB union 地址不连续（ds_blocks 隔断）；PANEL-MCAST 几何不成立
（A 是 M-split，两 CTA 需要不同字节）；DQ-Ready 预期按 vg_1 A/B 证据减半。
本文件只承载 K2-G4 主刀。
