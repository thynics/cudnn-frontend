# vd_2 手术规格（方向 D 双 tile 批处理 + C4 pds 深 2 + kscore 流式捐赠）— 2026-08-05

基线：`dsa_bwd_sm100_2cta_v12.py`（9bc35b3 血统）→ 新文件 `dsa_bwd_sm100_2cta_vd2.py`。
立项依据：四层证伪链（vre_3 供给侧 / S1 重排 / 混合所有权 ISA / H TBC-reduce）
收敛后，CG2 铁律内仅存的未测主杠杆。方案出处：战略评审 §3（D）+ §4（C4），
P0 佐证（baseline drain-bound 85% ⇒ reducer 有 slack 吃 D 的 drain 重排）。

## 核心机制

1. **D（摊薄"每 tile"量词）**：quadrant 字节 token-stationary，每对 kv tile
   只物化一次——8 个 quadrant gen 服务两个 tile（原 16），每 gen 连发两个
   N=64 dkv pass（tile 2j 与 2j+1 的 P/dS block 各一），MAT_ACQ 信用压力减半，
   零新带宽。
2. **C4（随 D 免费）**：P/dS 发布面双份 ⇒ pds 管道深 1→2，math(t+1) 发布
   slot (t+1)%2 不再等 grads 读完 slot t%2——定速环锚点边解锁。
3. **kscore 捐赠**：score_kv 32KB 单驻 → 2×16KB 双缓冲按 D256 分半流式
   （gather 填充分两半提交，score/dP 发射按半消费），净腾 16KB 给发布面。

## SMEM 账（预算 227KB，v12 现用 225.5）

| 项 | Δ |
|---|---|
| p_blocks / ds_blocks 双份 | +16KB |
| ds_image 双份（dq 的 B，随 pds 槽轮转） | +8KB |
| p_xchg 退休（P1b 式：send 直接源自 block 半区） | −4KB |
| ds_xchg（P1b 已死字段）删除 | −4KB |
| kscore 32→2×16 流式 | −16KB（净 0：物理仍 32KB，但语义变双缓冲，无需新增） |

注意：kscore 的"捐赠"实为把 32KB 从单驻改双缓冲后**不新增**字节——真正的净增
= +16+8−8 = **+16KB** ⇒ 总量 ~241.5 超预算!!⇒ **必须真拿走 kscore 16KB**：
score_kv 降为 2×8KB（D256 半区 × kv64 own = 8KB/半?）——核对：score B =
[kv64-own × D512] bf16 = 64×512×2 = 65,536B?? 与 struct score_kv 16384 elem
= 32KB 矛盾 ⇒ **实现前必须先核 score_kv 的真实几何**（PIN-1）。若 score B
半区 = 16KB，则双缓冲 2×16 = 32KB 无净省 ⇒ 改用**单缓冲 2×8KB D128 四段流**
（4 段 × 8KB，2 深）净省 16KB，score 发射按 K_CHUNK 段消费（发射循环已按
chunk 分段，天然适配）。

## TMEM 账（512 列，v12 精确闭合）

dQ [256,512) 不动；S/dP pp [0,128) 不动；dKV [128,256) 128 列改**成对分时**：
- 每 (pair, round)：dV-pair 双宽块 [t0|t1] 占满 128 列 → commit → reducer
  T2R+原子（体积 = 2 tile 份）→ release → dK-pair 复用同 128 列 → commit → drain。
- dkv_done gen 数 8/2tile → 4/pair（2 round × 2 tensor），drain 体积不变、
  调用数减半（P3 证实 fixed-cost 主导，调用减半是纯赢项）。
- 风险：dV→dK 之间新增一条 drain-wait 边；P0/vre_1 显示 reducer 空闲 ~4µs/tile，
  预算内。若 trace 显示该边绑定 ⇒ 预登记回退：dV/dK 各 64 列单 tile 宽、
  pair 内按 tile 串行（退回一半 D 收益，保 C4）。

## 调度（leader 旋转节奏改双拍）

- 偶迭代 k=2j+2：S(k)、dP(k)、**grads_pair(2j, 2j+1)**（8 gen × 2 pass）、
  dq(2j)、dq(2j+1)、pds release ×2（在两个 dq 之后，S1 红线）。
- 奇迭代 k=2j+1：S(k)、dP(k)（无 grads——供应窗口加倍即 D 的准入判据）。
- math：逐 tile 不变，发布 slot t%2（寻址 = 基址 + (t%2)·PAIR_STRIDE）；
  MATH_PDS_ACQ 等 slot t%2 的 grads 释放（深 2 ⇒ 等的是 pair 两拍前）。
- relay：逐 tile 不变，源/落地按 slot t%2。
- kdq/ROUTE_K/dQ 波/dQ epilogue：逐 tile 完全不动。
- **奇数 tile_count 尾**：最后一个孤 tile 走单 tile 兼容路径（per-gen 单 pass
  + 单宽 drain），lengths/holes 模式必经，正确性红线。

## 信用算术（实现前逐边预检，vre_3 纪律）

- round 环（10 gen/tile → 偶拍 10+空拍?）：quadrant gen 数减半但 kdq 仍逐
  tile ⇒ 每两 tile gen 序 = [kdq(2j)×2, quad×8, kdq(2j+1)×2] = 12，
  12 mod 2 = 0 相位律保持（环深仍 2）；奇拍 leader 不消费 quad ⇒ 消费滞后
  与 W17 生产序需重排为同一双拍结构——**W17 循环改双拍**（PIN-2：逐边推演
  W17 双拍与 leader 双拍的 FIFO 对齐，含 tile_count 奇偶两种收尾）。
- pds 深 2：math 生产 / leader 消费 各 2 深 ✓；relay commit（B-lite 不带入，
  保持 v12 原位——单变量纪律）。
- dkv_done：4 gen/pair，reducer 消费按双宽块（T2R 视图 PIN-3：128 列双宽的
  drain 坐标折叠需重推 _drain 视图）。

## 预期与判读

- 供应等待 ~×0.7（kdq/ROUTE_K 不变），pds 边解锁：period 6.95 → **~5.0-5.4**。
- proxy 三件套判决：correctness 4/4 硬门（含 lengths/holes 奇 tile 数）；
  vd2 < v12×0.80 达标；0.80-0.92 部分成立；≥0.92 证伪且本方向线终结
  （届时 CG2 范式内弹药耗尽，回到战略裁决）。

## PIN 清单（写代码前必须核死）

1. score_kv 真实几何与 score-B 分段消费的现有结构（决定捐赠形态）。
2. W17/leader 双拍 FIFO 对齐全推演（含奇尾）。
3. 双宽 dkv 块的 TMEM 折叠/drain 坐标（_drain 系视图重推）。
4. math 发布寻址的 slot 参数化点（stmatrix 目标 + relay 源 + landing 目标）。
