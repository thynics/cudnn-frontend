# v_s1 设计草案：dO 切片流 + loan 退役（基座 = final）

**日期**：2026-08-07 ｜ **状态**：⚠️已被 V_S1_SPEC.md 取代（终稿修正本稿三处：
①own-h S2S/别名被 CG2 同址锁+驱逐论证判死，dO quadrant 代全走 GMEM TMA；
②byte 账实为净增 +96KB/tile/CTA 而非 +16KB——本稿把 loan own-half 的 SMEM bulk
误记为 GMEM；③流生产者 = gather warp1，非 W17/W19）｜ **用户已确认方向与基座**

## 提案（用户原案 + 账面精化）

dV/dK 中 P/dS 常驻不动；把 dO 面板从 token 常驻改为**每 tile 2-stage × 4-slice
切片流**（slice = [D128×H64] = 16KB，恰为 quadrant/chunk 消费粒度）；Q 面板
第一步保持常驻（"先加载完一个，切另一个"）；loan 机器（dO_r0 借宿 score_kv）
整体退役。

## 字节账（对 v_s1 有利的核心事实）

- dO 重取今天已付 75%：loan dO_r0 32KB + dO_r1 peer 半 16KB = 48KB/tile 已走
  GMEM；整面板流式 64KB/tile，**净增仅 16KB/tile**（L2 命中，~+3GB/s/CTA）。
- 净释放 ~32KB（64KB 面板 − 2stage×16KB staging）——**本 rev 不投资**（单杠杆），
  v_s2 投向 kscore 深度 2（canonical-K 解锁 → kdq gather 退役）。
- loan 退役 = kscore 串行链删一环（S/dP读→loan填→dQ读→K(t+1)填，vk_6 实测
  该链 +1.0-1.5µs/窗）——**删除非搬运，守恒律不适用**。
- score_kv 生命周期解耦（dQ epilogue staging 的 loan_epi_safe 门消失）。

## 消费侧兼容性（已核）

- dP 按 4 chunk 发射（与 S 同构，_issue_score_chunks 家族）→ chunk c 等 slice c
  即发，score_kv 的 4-chunk staged 消费模式直接复制；
- dV 的 quadrant [D128×H64] = 恰好一个 slice；own-h 代可从流 staging S2S（或
  直接别名，视生命周期）；peer-h 代照旧 GMEM TMA 进环（不动运输拓扑——
  v_gpt_2 教训）。

## 已读代码事实（final 行号）

- loan：_fill_score_loan_do_r0_vc2 @4575（warp0 发 1 本地 bulk + 1 peer TMA，
  loan_tma_mbars[2] @4127，epi_safe @4128；t_dot_loan_smem_a/b 目标 = score_kv
  两半区 +0/+8192 元素）；
- kscore 双代结构：K 代（dP 后 release#1）→ loan 代（grads 中 release#2，
  LOAD_K(t+1) acquire ← round-0 两次 dV MMA 完成）——v_s1 退役第二代后
  kscore 回归单代/tile，LOAD_K(t+1) 的 acquire 门变为 release#1（dP 完成）
  ——**这本身就是一条链缩短**。

## 待做（步骤 1 余量）

1. ✅ **已核（2026-08-07）：布局面零新发明**。final:4885-4888——dP 消费
   stationary_do 用的就是 `score_a_layout_staged`（4-chunk staged），且
   final:506 断言 stationary 与 score_a 的 swizzle inner 相等。流式化 =
   staging 物理缩为该家族的 2-stage 形态（chunk c 落 slot c%2），dP 的
   chunk-issue 描述符 stage 索引 c→c%2——vh_1 的 12-gen 定相代码可回收，
   且 dO 是 TMA 快生产者（vh_1 铁律例外侧，与其 K-gather 死因本质不同）。
   TMA 侧：stationary_do_tma 视图（final:4893）换 2-stage 布局 + 逐 chunk
   发射（4 笔 16KB/tile，box 天然存在）。
2. 新流水线信用环设计：2-stage FIFO（producer = W17 或 gather？dO slice TMA
   发射点、mbar 账、与 S/dP chunk 发射的相位配对）；三路径等待图
   （稳态/首 tile/tile_count==1）；
3. dV 消费点（loan_quad_fragment_a/b → 流 staging fragment）与 dO_r1 own 半
   的来源改写；
4. 预登记门：四 case 逐位同 final；≥0.05 达标 / null 归档 / 负停；
   REDUCE_ATOMIC 监控（L2 副作用，v_gpt 先例）。

## 风险面

dP(t) 对流的新依赖边（slice 必须提前一拍在飞；TMA 是最快生产者，vh_1 例外侧，
但需等待图证明）；+16KB/tile L2 流量副作用；首 tile 冷启动相位。

## 战略定位

v_s1 中性即胜（钱庄 rev）；v_s2（kscore 深 2 + canonical-K）才是对 1.14ms
缺口的收益刀。若 v_s1+v_s2 兑现，"超越 baseline Rubin-locked"判定需重审——
资金侧被釜底抽薪。
