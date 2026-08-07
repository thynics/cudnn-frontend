# v_s1 终稿 SPEC：dO 切片流 + loan 退役（基座 = final @ cc107c4/73c657f）

**日期**：2026-08-07 ｜ **状态**：设计定稿，进入实现 ｜ 桌面草案 = V_S1_DESIGN.md（本文修正其三处）

## 0. 对 V_S1_DESIGN.md 的三处修正（桌面推演证明）

1. **"own-h 代可从流 staging S2S/别名" 不成立**。三重锁死：
   - CG2 描述符同址锁 ⇒ dP 的 chunk c 在两 CTA 必须同槽同序（公共流序）；
   - 流结束时 2 槽驻留 = 两 CTA 各自 panel 的**同两个** chunk 索引；但 own-quadrant
     对 rank0={chunk0,2}、rank1={chunk1,3}，并集为全部四 chunk ⇒ 任何 <64KB 的
     staging 必然驱逐某 rank 的 own 源（3 槽同样不解）；
   - 窗口论证：chunk c 存活窗 [land, land(c+2)) 终止于 dP(t) 中段，而任何 ring
     credit（gen g 需 gen g-2 的 UMMA release）最早在 grads(t) 的 dQ r0 之后才开，
     源先于 credit 过期。**结论：dO 的 4 个 quadrant 代全部走 GMEM TMA（mdOT，
     L2 热）= 今天 OWN_HALF_BULK=False 分支的现成代码。**
   - 推论：任何"半驻留 parking"方案（32KB stream + 32KB park）SMEM 净省 = 0，
     违背钱庄目的，全部否决。
2. **byte 账修正**。原稿"已付 48KB/净增 16KB"把 loan own-half 的 SMEM bulk 误记
   为 GMEM。实账：今天 dO GMEM = 32KB/tile/CTA（loan peer 16 + dO_r1 peer 16）；
   v_s1 = stream 64 + quadrant 4×16 = 128KB/tile/CTA，**净增 +96KB/tile/CTA
   （≈+14GB/s/CTA，全 L2 命中，dO 面板 128KB/token 常驻 L2）**。这是本 rev 的
   头号实测风险（v_gpt REDUCE_ATOMIC 先例），预登记监控。
3. **流生产者 = gather warp1**（原稿开放题）。否 W17：vre_3 铁律（禁给饱和生产者
   迭代头加活）+ 它在环上的 acquire 链（g8 acquire ← dK r1 释放，slot 尾）与流
   第二半拍位（dP c0/c1 释放，slot 头）无法同时 poised。否 W19：final 里它并不
   闲（中继 6→8 个环 TMA 完成，串行 waits 会假耦合流 commit）。warp1 在 loan
   退役后空闲，其职责序单调可证（见 §3）。

## 1. 机器形态

- `stream_do` = 2×16KB（score-A 家族 2-stage 布局 `make_smem_layout_a(score_mma,
  score_tiler, bf16, 2)`；final:506 断言家族 swizzle-inner 同 ⇒ 零新布局代数）。
  chunk c → slot c%2，公共序 0,1,2,3。SMEM 净省 32KB（不再投资，v_s2 专款）。
- `pipe_dostream` = PipelineAsyncUmma(stages=2, producer=elect×2CTA,
  consumer=leader)。TMA 完成经每 CTA 本地 raw `dostream_tma_mbars[2]` 桥接 →
  warp1 wait 后 producer_commit（复用 W19↔round_tma_mbars 既有模式；PTX TMA
  完成只能落本地 mbar，CG2 消费需 pipeline 的双 CTA arrive 语义）。
- **dP 消费**：`_issue_dp_streamed_v_s1`——per chunk：consumer_wait → 8×k_block
  GEMM（A=stream_do[..,c%2]，B=score_kv[..,c] 不变）→ consumer_release（umma
  arrive，源读完成即放槽）。accumulate 旗序与 v7 逐位同（chunk0/kblock0=False）。
  dostream 状态在 accumulator ping-pong 动态分支**外**推进（v7 教训）。
- **ring 8→10 代/tile**：kdq×2 + dO_r0 h0/h1(纯TMA) + Q_r0 h0/h1(bulk/TMA 不变)
  + dO_r1 h0/h1(纯TMA) + Q_r1 h0/h1(bulk/TMA 不变)。W19 中继 8 个。
  head 消费序：dQ×2 → dV r0 h0/h1(环) → dK r0 h0/h1(环)；tail 不变。
- **kscore 回归单代/tile**：LOAD_K(t+1) 门 = dP(t) release（RK_ACQ#2 0.494 →
  #1 0.094 应收）。gather 循环全一致化：tile_count==1 特例、尾 tile
  "recommit unchanged" 分支、loan 三处调用点、gather_barrier 双会合全部删除。
- **首 tile 冷启动改善**：dP(0) chunk0 只等 16KB（原等 64KB 面板）；
  stationary_ready[1]/stationary_tma_mbars[1] 机器删除（Q 侧保留）。
- **epi 门保留**：loan_epi_safe 语义改为"kscore 尾（K(last) 被 dP(last) 读完）"，
  gather producer_tail 后 arrive 不变——零稳态成本，保住显式 happens-before
  （不赌 UMMA 跨 barrier 的 arrive 序）。
- **数值**：逐位同 final（所有 MMA 形状/次序/累加序不变，纯操作数运输手术）。

## 2. warp1 职责序（关键正确性论证）

每迭代 i（slot i = 处理 tile idx tc-1-i 的时间槽）：
```
[kscore acquire → K(i) 装载(集体) → commit(集体 arrive)]   门=dP(i-1) release，slot i-1 中段
[stream c0,c1(slot i)：acquire×2+arm+TMA]                  credit=dP(i-1) c2/c3 读毕 ≤ kscore 门 ⇒ 零阻塞
[wait dostream_tma_mbars → commit ×2]                      TMA 飞行 ~0.3µs
[stream c2,c3(slot i)：acquire 阻塞]                        credit=dP(i) c0/c1 读毕，slot i 头 —— poised
[wait+commit ×2]
[kdq(block i)：kdq_barrier + fill + arrive]                会合时刻 slot i 尾
```
四个阻塞点释放时刻单调递增，且每个集体职责（K commit、kdq_barrier）之前的阻塞
点都严格早于该职责的依赖方需要时刻 ⇒ 无环。死锁核心检查：dP(i) 依赖 K(i) 的
集体 commit，warp1 在任何 stream 阻塞**之前**已完成该 arrive ✓。
冷启动（i=0）：K(0) 无 acquire（处女代）→ c0,c1 处女 credit 零阻塞 → c2,c3
poised 等 dP(0)；warps0/2/3 先行到 kdq_barrier 等待，无害。
尾（i=tc-1）：无 slot tc 的流；循环后 pipe_dostream.producer_tail。
all_empty：tile_count==0 守卫，全机器不转。

## 3. 暴露风险与预登记（判读门见 prompt.txt）

1. **dP 头增长**：2 槽 4 chunk ⇒ c2/c3 的 TMA 只能在 c0/c1 被读后起飞；估
   暴露 +0.1~0.3µs/tile（16KB L2 飞行 vs 相邻 chunk 读间隔），若被 leader 下游
   PDS_WAIT 0.506 池吸收则为 0（棘轮律：dP 不在 pacer 自等链上）。vd_1 警告
   （2 固定槽=每奇偶深度 1）在此显形即暴露版本。
2. **+96KB/tile/CTA L2 流量**：REDUCE_ATOMIC 与周期成对监控（后续 mode all 腿）。
3. **dV r0 环代 just-in-time**：原 loan 预装优势消失，g2/g3 填充窗 = dQ r1 +
   relay 等待（~0.5µs）对 16KB TMA（~0.3µs）——够但不再宽裕。
4. 对冲收益：kscore 链删环（vk_6 实测该环 +1.0~1.5µs/窗）+ gather warp0 卸载
   （16KB bulk+16KB TMA+双会合/tile）+ 首 tile 冷启动。K 侧现有 +1.20µs slack
   意味着删环收益可能先变 slack 不变周期 ⇒ **中性即胜，SMEM 32KB 是本金**。

## 4. 实现清单（v_s1.py = final.py + 以下手术；已按对抗核查修订）

- `__call__`：+stream_a_layout_staged(2-stage) + 断言族（cosize 16384、
  swizzle inner 同、per-stage 字节 = score_a_stage_bytes、**per-stage 布局
  str 全等**）；+tma_atom_do_chunk = **v0 硅上已证构造**（score_a_layout
  单 stage box + score_tiler(H128,N64,K128) + dp CG2 mma +
  cluster_layout_vmnk.shape，v0.py:478-485）——实现期弃用了最初的
  stationary-cg1 自创构造；launch +3 参数，仅 V2 kernel 签名 +3
  （base TwoCTA.kernel 本就是签名失配死代码，final 时代已 +2 失配，不动）。
- kernel 侧分区（v0 模式）：g_do 按 (H128, K128) local_tile →
  dp_tiled_mma.get_slice(rank).partition_A →
  tma_partition(block_coord, a_cta_layout) → `[None,0,None]` 切唯一
  RestM → warp1 以 `[None, chunk]`/`[None, slot]` 发射（rank 半区烙在
  分区里，helper 无 rank 参数）。
- SharedStorageV2：stationary_do(32768el)→stream_do(16384el)；+dostream_mbars[4]
  +dostream_tma_mbars[2]；−loan_tma_mbars[2]；stationary_tma_mbars/ready 2→1。
- 删 `_fill_score_loan_do_r0_vc2` 与 loan_quad 视图/fragment/分区。
- gather 角色：一致化循环 + warp1 流职责（§2），两分支自包含
  （pipeline 状态不跨动态 warp 分支）。
- leader：dP 改 `_issue_dp_streamed_v_s1`；删 stationary_ready[1] 等待；head
  签名去 loan/kscore 参数，dV r0 走环。
- W17：prologue 只装 Q；6-gen→8-gen（dO 代 = 纯 GMEM TMA 双半区）。
- W19：6→8。
- host 探针 probe_v_s1_layouts.py：布局代数断言。本机无 cutlass wheel，
  **改为 B200 容器内先跑**（委托 prompt 要求 validation 腿前执行）。

## 6. 硅上判决（2026-08-07，r3 = dsa-vs1-r3-1786083915，rev e281049）——负停

- **数值全胜**：探针 PASS；候选/参照两腿 validation PASS、correctness 4/4，
  四 case max_abs 指纹程序化比对**逐位同**（dense/lengths/holes/all_empty 全部
  与 final 指纹一致）。手术的协议面（10 代环、dostream 管线、warp1 职责、
  loan 退役、kscore 单代）全部正确。
- **性能负停**：候选 13.467873ms vs 参照 v_w3_2 9.456525ms（同 service/job/
  node/container），Δ=+4.011ms ≈ **+2.27µs/tile**；two_cta_over_baseline
  1.6216 vs 1.1441 同向。远超 0.05ms 负停门。
- **量级判读**（未 trace，假说排序）：+2.27µs/tile 落在约束清单"暴露态"
  预测区间（vd_1：2 固定槽=每奇偶深度 1，+1.4~2.5µs/tile）顶部——
  首要嫌疑 = dP 头暴露（chunk2/3 TMA 起飞被 chunk0/1 的 MMA 源读 +
  umma→warp1→TMA→中继 的整圈往返闸住，且 dP 拉长 1:1 顺延 pds→math→
  grads 脊柱）+ dV r0 环代 just-in-time 三角接力（leader 放 g0→W17→W19→
  leader）取代 loan 预装。次要嫌疑 = +96KB/tile/CTA L2。**需 trace 腿裁决**。
- r1（探针 MLIR context）、r2（provenance mark）两轮 prepare 教训已各自
  修复入库（62cf799 / e281049）。
- 候选修补方向（未开工，待裁决）：
  (a) **S/dP chunk 交错发射**——leader 把 S(t) 的 4 个 chunk 与 dP(t) 的
      chunk 等待交错（S 在常驻 Q 上零等待），让 dP c0/c1 的源读提早完成、
      c2/c3 的 TMA 飞行藏进 S 剩余 chunk 的执行影子里；不加 SMEM、
      每 accumulator 累加序不变 ⇒ 数值逐位同保持。
  (b) 流深 2→3（+16KB，SMEM 净省降为 16KB，违单杠杆纪律需重裁）。
  (c) 若 trace 判 L2 为主因，v_s1 家族按 byte 账接近死刑。

## 5. 对抗核查回执（2026-08-07，6 视角 workflow，零 blocker/major）

- 等待图/死锁：warp1 职责序、kdq_barrier 各方、10 代环链、冷启动/单
  tile/all_empty 全过；确认 dV r0 环代 just-in-time 为纯延迟风险（§3.3）。
- 账面：10 产=10 耗；kscore/dostream/ring 三管线计数与相位全平；
  SMEM 净 -32KB。
- 数值面：dP accumulate 旗序逐位同 v7；dV r0 环代 GMEM box 与原 loan
  同源（t_dot_gmem[None,0,h]）；S/dQ/tail/drain/epilogue 零 diff。
- 修复项（已落）：僵尸 rank 参数删除；W17 过期 own-half 注释改写；
  per-stage outer 等式断言补入；SPEC §4 与实现对齐（本节）。
