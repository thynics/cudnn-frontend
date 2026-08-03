# v3.2 (T3-64) 建造附录 —— dS 双像修复规格 + 第三审勘误 + 窗算判决

主设计文档：T3细粒度全f32转置设计_20260803.md。本附录记录其 §4 必修缺陷的修复终稿
（8-agent 工作流产出，第三审双轴通过 refuted=false ×2），实现必须以本文为准。

## 1. dq_b 修复终稿（取代主文档 §3 的 dq_b 行）

- **独立双子像单区，rank 对称**：总区 16,384B/CTA/bundle，单缓冲，两 CTA 严格同 SMEM 偏移。
  逻辑 [kv128 × own-H64]，MN-major（H 连续 128B 行），SW128B swizzle。
  物理 = 两个堆叠子像：`sub_img[0] = kv[0:64] 行`（基址+0，8,192B）、`sub_img[1] = kv[64:128] 行`（基址+8,192）。
  静态约定 CTA_h 拥有 kv-half h ⇒ CTA0 内 sub[0]=own 本地像 / sub[1]=peer 落地；CTA1 镜像。
  字节角色互补但**地址角色完全对称** → CG2 同地址规则合法，单一 B 基描述符 + h×8,192 窗口跳段
  （8,192 为 SW128B 1,024B swizzle 原子整数倍，相位保形）。dS slab 别名死刑不复现。
- **own 半写入**：math 在 dS 发布 epilogue 内**第二次 stmatrix**（寄存器仍驻留，零 TMEM 重读），
  [own-kv64 × own-指派-H64] → 本地 sub_img[own]。写前等 `mb_dqb_free[own](t-1)`。
- **peer 半写入**（8,192B）：同 epilogue 第三段，st.async.shared::cluster 寄存器直写对方镜像偏移；
  **备选预案**（若 SASS 门毙掉直写）：dS slab 改堆叠 [kv64×H64]×2 子像（G4-A 两段 K64 描述符窗口读），
  peer 半退化为 8,192B 连续 bulk DSM。
- **门**：`mb_dqb[h]`（cluster-scope 聚合，L4 机器）：half h 就绪 = CTA_h 本地 stmatrix 提交 ∧ st.async 落地。
  释放 `mb_dqb_free[h]` = G5 wave_h 的 MMA-arrive 簇广播。
- **G5 发射**：保留 K 维分半：wave_h = (M256, N128, K64)，B 窗口 = dq_b 基址 + h×8,192，
  accumulate=1 链入 dQᵀ 256 列持久累加器；wave 序静态 kv0→kv1。

## 2. SMEM 终账（231,424 / 232,448，余 1,024）

K chase 2×8,192=16,384 ｜ Q 面板 65,536 ｜ dO 面板 65,536 ｜ P slab 16,384 ｜ dS slab 16,384
｜ 供应环 2×16,384=32,768 ｜ dq_b 16,384（own 8,192 + peer 落地 8,192）｜ stats/mbar 2,048。
TMEM：dQ 256 + S 乒乓 64 + dP 乒乓 64 + dV 槽 64 + dK 槽 64 = 512 恰满（全 f32）。

## 3. 第三审勘误（实现期义务，正确性级优先）

1. **【正确性级】`mb_dqb_free[h]` 必须挂在该 half 的最后一个消费原子上**（wave_h × 2 D-round 共 2 次消费）：
   count 覆盖两个 D-round 的 arrive，或挂 D-round-1 的完成沿。只挂 r0 ⇒ math(t+1) 覆写时 r1 仍在读 = 数据竞争。
2. **E3 到达门不对称**：st.async 的 completion-tx mbar 位于目的 CTA，leader 无法直接观测对端落地——
   需 relay arrive（peer warp 本地等落地 mbar 后远端 arrive 到 leader 门）或镜像双门。实现选型入审计。
3. 无环不变量措辞：E2/E3 等对方 φ≤3(t)，E5 等 φ5(t)，均前向。
4. 分半 wave 只单向吸收偏斜（CTA1 晚可吸收；CTA0 晚则 wave1 队头阻塞）——接缝定价已计入，勿按对称宣传。

## 4. D_c=256 单槽相位拆分：裁决不升格

E[净收益] ≈ −0.1µs（暴露下界 0.12µs 不可归零），中心回 ~5.5 无穿透。登记为条件复议期权
（扳机：裁决 A 原子单价 ≤0.035 ∧ rev1 实测 drain−math 墙差 ≥ +0.3）。被 dV/dK 异粒度 +
G3/G4 逐半启动（零暴露期权）支配。**v3.2 主形态维持：h64 / D128 / dKV 双槽 / 全 f32。**

## 5. math 税修正

own 半二次 stmatrix +0.05–0.1µs/bundle ⇒ math 墙 4.90–5.90，中心 ~5.55，总带 5.0–6.0 维持。
