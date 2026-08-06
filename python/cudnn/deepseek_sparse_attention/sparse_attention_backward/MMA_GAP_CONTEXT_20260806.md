# DSA 反向 2-CTA kernel：稳态 MMA gap 问题 —— 完整上下文

**日期**：2026-08-06 ｜ **现役版本**：vk_2 ｜ **读者**：无代码访问权限的推理 agent

## 0. 阅读说明

- 本文档只陈述**事实与测量**。既往团队做出的解释性结论全部集中在 §9，并明确标注为
  "假设"——它们可能是错的，请独立推理。
- 所有时间若无特别说明，单位为 µs；e2e 单位为 ms。
- 存在两种测量口径（release / trace），定义与换算陷阱见 §5。**两口径数字禁止直接互算。**
- 每个测量都标注了来源 run。术语表在 §10。

---

## 1. 问题陈述

### 1.1 目标

现役 kernel（vk_2）端到端 9.902 ms，同日 baseline（另一实现，1-CTA 架构）8.609 ms，
比值 1.1502。目标：追平或超越 baseline。

### 1.2 "MMA gap" 指什么（现象定义）

稳态期每个 KV tile 周期 6.80（trace 口径，S_ISSUE 相邻起点间距的 32-tile 中位数）。
在这 6.80 里，MMA 发射角色（leader warp）的发射忙时只有 1.86，其余 4.94 为等待。
具体时间线（vk_1_trace run，2026-08-06，注入口径，每 tile）：

```
S+dP 发射 1.02
  → [0.41 等待，无 span 标记]
  → WAIT_dQ 0.15 + dQ round0 发射
  → [0.39 间隙]
  → dQ round1 发射
  → [0.54 复合等待，无 span 标记]
  → dVdK 发射 ×8，每次发射前间隙 0.35–0.58（间隙合计 3.36）
  → [0.50 间隙]
  → 下一 tile 的 S 发射
```

MMA 单元本身是异步的；上述 gap 是**发射流**的间隙。周期 6.80 与各角色忙时的差值
构成了每 token 的主要时间超额（对 baseline 的分解见 §7.6）。

### 1.3 剩余缺口的构成（测量事实）

- vk_2 与 baseline 的 e2e 差 = 1.29 ms ≈ 每 token 23µs（release 口径，55.35 波）。
- 尾段（dQ epilogue + 收尾）经三次独立手术（§8 vg_6/vk_3）全部测得 ≈0 收益后，
  修正估计其真实超额 ≈ 0.10–0.15 ms。
- 其余（~85–90%）在稳态期的 32 个 tile 循环里。

---

## 2. 计算任务（数学事实）

### 2.1 形状与数据类型

- DeepSeek Sparse Attention 反向。H=128 头，D=D_v=512，topk=2048，S_kv=4096，
  测试负载 S_q=1（每 token 一次 kernel 实例；一个 grid 内所有 token 并行）。
- 输入/输出 bf16，累加 f32。softmax 统计量（LSE、Σ(O·dO)）由前置小 kernel 算好，
  f32 存于 workspace。

### 2.2 稀疏结构

每 token 有 topk=2048 个 KV 行索引（`mTopkIdxs`，可变长度由 `mTopkLength` 截断）。
这 2048 行切成 **32 个 N64 tile**（每 tile 64 行）。KV 行在 GMEM 中不连续（gather
按索引逐行读取）。tile 遍历顺序为**索引从高到低**。

### 2.3 五个 GEMM（转置平面公式）

每 tile 依次需要（记号：n=tile 内 KV 行 0..63，h=头 0..127，d=特征 0..511）：

| GEMM | 公式 | C 形状（cluster） | 说明 |
|---|---|---|---|
| G1 | Sᵀ[h,n] = Σ_d K[n,d]·Qᵀ[d,h] | H128×N64 | K 分 4 个 D128 chunk 累加 |
| G2 | dPᵀ[h,n] = Σ_d V[n,d]·dOᵀ[d,h] | H128×N64 | V≡K（同一张量） |
| G3 | dVᵀ[d,n] += Σ_h Pᵀ·dO | D256×N64 ×2 round ×2 H-pass | 每 tile 累加进 tmem |
| G4 | dKᵀ[d,n] += Σ_h dSᵀ·Q | 同上 | 同上 |
| G5 | dQᵀ[d,h] += Σ_n Kᵀ·dSᵀ | D256×H128 ×2 round | **跨全部 32 tile 累加** |

中间的逐元素计算（math）：P = exp2(S·scale·log2e + LSE)，dS = P·(dP − Δ)·scale，
并将 P、dS 转 bf16 写入 SMEM 供 G3/G4/G5 消费。

### 2.4 梯度落盘方式

- **dQ**：常驻 tmem，32 个 tile 累加完后统一 epilogue（§4.6）。
- **dV/dK**：每 tile 算完即由 reducer warps 从 tmem 取出，以 `red.global.add.v4.f32`
  原子加进 GMEM 的 f32 workspace（按 topk 行索引 scatter，谓词 kv_index≥0）；
  kernel 族结束后另有 canonical 转换步骤产出 bf16 终值。

### 2.5 正确性契约

四个用例（dense / lengths / holes / all_empty）全部 PASS 才算正确。历史 PASS 值
参考量级：dense dq≈0.004、dkv≈0.0097、dsink≈0.0011；lengths dq≈0.012、dkv≈0.073。

---

## 3. 硬件与平台事实（B200 / SM100）

- SMEM 每 CTA 可用上限 **232,448 B**；当前占用 ~229,888 B（slack ~2.5KB）。
- 每 SM tmem 512 列（本 kernel 列分配见 §4.3）；tcgen05 MMA。
- **2-CTA cluster**，本 kernel 全部 GEMM 用 cta_group::2（CG2）：M 维跨两 CTA 各半，
  B 操作数按 N 维分半。工程事实：CUTLASS 的 CG2 描述符要求操作数在两个 CTA 的
  SMEM **同地址**（rank 对称）；按 rank 加不同偏移的布置会破坏描述符构造
  （§8 M3 行有一次被此规则挡死的桌面判决）。
- DSM：cluster 内跨 CTA SMEM 写（`cp.async.bulk` s2cluster + mbarrier expect_tx）。
  实测本 kernel 中继 warp 路由 4KB 一次 ≈ 0.17（trace 口径）。
- TMA bulk tensor copy（GMEM↔SMEM）；`cp.async` 逐行拷贝（gather 用，256B 粒度）。
- named barrier（CTA 内指定线程数会合）、mbarrier（含 phase/parity 与 tx-count）、
  `setmaxnreg` 动态寄存器重分配。
- B200：148 SM = 74 cluster。本负载 **55.35 波**（每 SM 槽位串行处理 ~55 个 token）。
  SMEM 占用决定每 SM 同时只驻 1 个 CTA（无 co-residency）。
- kernel 启动契约：每 token 一个 cluster，grid 一次启动全部 token。
  **此契约是规格，不可更改**（persistent-cluster 类方案已被否决，§6）。

---

## 4. 现役实现架构（vk_2 = 血统 v12→vc_2→vg_1..5→去 split-publish）

### 4.1 线程组织（每 CTA 640 线程 = 20 warps）

| warp | 角色 | 寄存器 | 职责 |
|---|---|---|---|
| w0–3 | **gather**（128 线程） | 48 | K tile 稀疏 gather（score 用）+ kdq 镜像 gather（dQ-A 用） |
| w4–7 | **math**（128 线程） | 128 | S/dP 的 T2R、softmax 反向、P/dS 写 SMEM、dQ epilogue |
| w8–15 | **reduce**（256 线程） | 128 | dV/dK 从 tmem 取出 + GMEM 原子加 |
| w16 | **leader**（MMA 发射） | 48 | 全部 GEMM 的发射与管线信用管理 |
| w17 | **load**（W17） | 48 | Q/dO 面板装载 + ring 槽位供给 + kdq 会合对端 |
| w18 | **relay**（W18） | 48 | P/dS 跨 CTA DSM 路由 |
| w19 | 空闲 | 48 | 无职责（vd_1 曾尝试启用，null） |

寄存器池恒等式：256×48 + 384×128 = 640×96。

### 4.2 SMEM 地图（每 CTA，bf16=2B）

| 区域 | 大小 | 内容 | 生命期 |
|---|---|---|---|
| stationary_q | 64 KB | Qᵀ 面板 [D512×H64 own-h-half]（token 常驻） | 每 token 装一次（TMA） |
| stationary_do | 64 KB | dOᵀ 面板（同上） | 同上 |
| score_kv | 32 KB | 当前 K tile 的 own-n32×D512（S/dP 的 B 半份） | 每 tile 重填（gather） |
| round_buf_a | 16 KB | ring 槽 0（grads A 操作数流转） | 每 tile 4 代 |
| round_buf_b | 16 KB | ring 槽 1 | 每 tile 4 代 |
| p_blocks | 8 KB | P 的 bf16 块（G3 的 B） | 每 tile |
| p_xchg | 4 KB | P 跨 CTA 交换到货区 | 每 tile |
| ds_image | 8 KB | dS 完整镜像（G5 的 B） | 每 tile |
| ds_blocks | 8 KB | dS 的 bf16 块（G4 的 B） | 每 tile |
| ds_xchg | 4 KB | dS 跨 CTA 交换到货区 | 每 tile |
| stats | 0.5 KB | LSE/Δ f32 | 每 token |
| mbars 等 | ~0.3 KB | 约 30 个 Int64 mbarrier | — |

**dQ epilogue 的 32KB staging 别名在 score_kv 上**（score_kv 死后复用，见 §4.6）。
**dO_r0 的两个 quadrant（各 16KB）经"时间借用"（loan）住在 score 区域**（§4.5）。

### 4.3 tmem 列地图（512 列，f32）

| 列 | 内容 |
|---|---|
| 0–63 | S（ping/pong 两份，S0/S1） |
| 64–127 | dP（ping/pong） |
| 128–191 | dKV 槽 0（当前 tile 的 dV 或 dK 部分和） |
| 192–255 | dKV 槽 1 |
| 256–383 | dQ round0（跨 tile 累加，常驻） |
| 384–511 | dQ round1（同上） |

### 4.4 管线与信用（深度 / 生产者 / 消费者 / 释放点）

| 管线 | 深度 | 生产者 | 消费者 | 释放点 |
|---|---|---|---|---|
| kscore | 1 | gather | leader（S、dP 的 B） | leader 在 dP(t) 发射后立即释放 |
| round ring | **2 槽** | W17（+gather 代填 kdq） | leader（grads 的 A） | 每代由对应 MMA 的 UMMA-tracked 完成释放 |
| pds | 1 | math 写、**relay 提交** | leader（G3/4/5 的 B） | leader 消费完 grads 后释放 |
| s_done/dp_done | 2 | leader MMA | math | math T2R 后释放 |
| dkv_done | 2 | leader MMA | reducer | reducer T2R 后释放 |
| dq_done | 1 | leader | math（epilogue 门） | epilogue 完成 |
| loan | 2 mbar | gather 触发 TMA | leader（dO_r0 A） | grads 读完；另有 epi-safe mbar 防 epilogue 与 loan 冲突 |
| stationary | 一次性 | W17 TMA | 各角色 | token 生命期 |
| landing/relay mbars | — | 对端 DSM | relay→leader | 每 tile |

named barriers：kdq_barrier（gather 128 线程 + W17，双次会合）、math_barrier（math
128 线程）、cta_barrier（640）。

### 4.5 稳态每 tile 事件序（按角色；括号内为 trace span 名）

**leader（w16）**，轮换调度——tile t 的 S/dP 之后发射 tile t−1 的梯度：
1. S(t) 发射（S_ISSUE）：4 个 D128-chunk 的 CG2 MMA，B=score_kv。
2. dP(t) 发射（dP_ISSUE）：同构，V≡K 复用同一 score_kv。**发射完立即释放 kscore 信用**。
3. WAIT_dQ + dQ(t−1) round0 发射（dQ_ISSUE）：A=kdq 镜像（ring 槽），B=ds_image。
4. dQ(t−1) round1 发射。
5. dVdK(t−1) ×8 次发射（dVdK_ISSUE）：2 梯度（dV,dK）×2 D-round ×2 H-pass；
   A=Q/dO quadrant（ring 槽或 loan 区），B=p_blocks/ds_blocks；每次发射前等 ring
   到货 + tmem 槽位；槽 0 的 4 次与槽 1 的 4 次交替占用 dKV tmem 两槽。
6. 尾 tile 之后：TAIL（管线排空 + tmem 释放会合）。

**gather（w0–3）**，软件流水提前一拍：
1. K(t+1) 的 score gather（LOAD_K）：等 kscore 信用（leader 在 dP(t) 后释放）→
   64 行 × D512 逐行 `cp.async`（每行 4×256B），按索引乱序、无效行清零 → 提交。
2. tile t 的 **kdq 会合**：与 W17 在 kdq_barrier 会合 → 128 线程分 16 组×8 线程
   gather kdq 镜像 = [N64×D128]×2 round（同一批 topk 行、列窗 256·r+128·rank，
   每行每 round 一个 256B 切片）直接写进 ring 槽 → `cp.async` 排空 → 再会合。
   （kdq 行与 LOAD_K(t) 是**同一批 KV 行**的不同列窗/不同目的布局。）

**W17（load warp）**：
1. ROUTE_K span：kdq 会合区（两次 kdq_barrier 之间持有 2 个 ring 信用，等 gather
   填完，然后代表 ring 提交这 2 代）。
2. MAT_QDO ×2（m0/m1）：6 个面板代（Q_r0×2、dO_r1×2、Q_r1×2）的供给火车——每代
   = own-h-half 从 stationary 面板 S2S bulk 拷贝 8KB + peer-h-half GMEM TMA 8KB，
   进 ring 槽；TMA 完成用 2 个 round_tma_mbars 轮转、**滞后一代提交**（lag-1）。
   每代必须先 acquire ring 信用（等 leader 消费掉两代前的那代）。
3. dO_r0×2 走 loan：TMA 到 score 区域（gather 在 kscore 临界区内触发）。

**math（w4–7）**：WAIT_S → T2R_S（S 从 tmem 到寄存器）→ WAIT_dP → T2R_dP →
SOFTMAX（32 路 exp2 全宽条带 + dS 链）→ STORE（bf16 转换 + P/dS 写 SMEM，
stmatrix）→ PDS_ACQ（等 pds 信用）→ 发布 arrive（count-128 mbar）。

**relay（w18）**：等 math 的 ready mbar → DSM 送 dS 4KB 到对端 → DSM 送 P 4KB →
提交 pds → 等本侧到货（对端发来的）→ 转成 leader 可见的 arrive。每 tile 忙时
0.32/6.8 ≈ 5% duty。

**reducer（w8–15）**：每 tile 2 轮（dKV tmem 槽 0/1）：WAIT_dK（等 MMA 完成信号）
→ REDUCE_T2R（Ld16x256b×Rep4 从 tmem 取 [D128×N64] 的本 rank 份额）→
REDUCE_ATOMIC（按预取好的 topk 索引 8 行×v4.f32 原子加进 workspace）。
rank 分域：每 CTA 只管自己的 D 半区 → **单 CTA reducer 体积是 baseline 的一半**。

### 4.6 序幕与尾段

**序幕**（每 token 一次）：W17 TMA 装 stationary 面板（LOAD_QDO ~0.45–0.6）；
LSE/Δ 装载；K(0) gather → S/dP(0)。实测 tile0 周期 3.9、tile1 7.1（慢启动 ~4）；
冷启动时 peer K(0) 路由 5.4、首个 WAIT_dK 11.4。head 合计 ~3.3。

**尾段**：dq_done 等待 → DQ_EPI ×2 round：math 128 线程 T2R dQ round [D128×H128]
f32 → bf16 → 写 score_kv 里的 32KB staging（现为 stmatrix 向量店，vk_3 起）→
math_barrier → 单次 32KB bulk TMA 到 GMEM → 等 TMA 读完 → math_barrier。
之后 leader TAIL（管线 tail 链 + cta 会合 + tmem 释放）。
trace 口径 DQ_EPI 2×11.9 + TAIL 27.7（重叠后尾窗 ~28）；
**注意：此尾段的 trace 数字被证实严重高估 release 行为**（§5.3、§8 vk_3 行）。

---

## 5. 测量体系（口径、税、判读规则）

### 5.1 两种口径

- **release 口径**：无插桩运行的 e2e 毫秒（candidate_ms）与同 run 的 baseline_ms。
  性能判决**只认这个口径**。
- **trace 口径**：IKET 运行时注入（trampoline 插桩）下采集的 span 时间线。用于看
  结构（相对位置、序、比例），**不用于总量**。

### 5.2 已测事实

- 同日 baseline_ms 漂移 8.268–8.61（±2%），且相邻两 run 可差 1.2%。
- 判读规则（既定纪律）：candidate_ms 与同日 ratio **并列判读**；两指标方向矛盾
  ⇒ 判为漂移内（null）。收益门槛惯例 ≥0.1 ms。
- vk_1 的 trace 口径每 token 246.1，release 口径每 token 177.8（9.844ms/55.35 波）
  ⇒ **平均注入税 ~38%**。baseline 的 trace 几乎无税（span 稀疏，154.6 ≈ release）。
- **税分布不均匀的证据**：trace 显示 DQ_EPI 2×11.9（占比大），但对其三个成分的
  三次独立手术（§8）release 收益全部 ≈0；标量循环区段（每线程 128 store + 128
  转换 + 地址算）指令密度远超均值。推论（假设，见 §9）：该区段局部税率远超 38%。
- **跨版本的 trace 周期禁止直接比较**（插桩集合不同 → 税不同）。
- span 语义：IKET 软件注解包络。显式 WAIT_* span 单列；其他 span 内嵌的等待
  **包含在包络里**（除非另有说明）。span 名额上限 31（超过会破坏采集）。

### 5.3 已知观测债

- leader 的 0.41（pds_dS 等待）与 0.54（复合等待）两段**无 span**，其构成是推断。
- gather 的 kdq fill 段无独立 span（藏在会合窗口里）。
- vk_1_trace 已含供给等待细分名额（RK_ACQ / MAT_ACQ / MAT_WAIT），
  但该 run 的逐 span 细读因后处理工具的名字映射缺失而只完成部分。

---

## 6. 不可违反的约束（规格/用户钦定）

1. **精度铁律**：数值结果不得改变。bit-exact 的重排/换指令形态可以；
   "近似替换"类（如 FFMA 多项式换 MUFU exp2）需逐案终裁（M2 尚未裁）；
   混合精度（f16 累加、FP8）**永久禁止**。
2. **CG2 铁律**：全部 GEMM 保持 cta_group::2。退守 CG1 无意义。
3. **启动契约不可改**：每 token 一 cluster 的 grid 语义是规格。persistent
   cluster / 多 token 复用 CTA 生命期的一切变体已被否决。
4. **kv128 tile、token pairing、重算 forward** 三个方向 2026-08-03 已被否决，禁再议。
5. feature 无开关：要么不要，要么默认开（不留 env flag）。
6. 实验纪律：单杠杆单 rev，预登记止损门。

---

## 7. 稳态测量数据汇编

### 7.1 vk_1_trace（2026-08-06，注入口径，稳态中段均值，每 tile）

| 角色 | 分量 | 值 |
|---|---|---|
| 周期 | S_ISSUE 间距中位数 | **6.80** |
| leader | 发射忙时合计 | 1.86 |
| leader | S+dP 发射 | 1.02 |
| leader | dQ×2 + dVdK×8 发射 | ~0.84 |
| leader | dVdK 发射前间隙 ×8 | 0.35–0.58，Σ3.36 |
| W17 | ROUTE_K（kdq 会合区） | 2.27 |
| W17 | MAT_QDO ×2（面板火车） | 4.38 |
| W17 | **串行合计** | **6.65 ≈ 周期（零 slack）** |
| math | 忙时（T2R_S 0.98 + T2R_dP 0.55 + SOFTMAX 2.22 + STORE 0.64） | 4.39 |
| math | PDS_ACQ 等待 | 1.15 |
| math | MATH_STORE(t) 结束时刻 − S_ISSUE(t+1) | **+0.86（侵入下一拍）** |
| reducer | 忙时（T2R 0.69 + ATOM 1.81） | 2.50（duty 37%） |
| reducer | WAIT_dK（每 round，提前到位等待） | 1.08 |
| gather | K 供给 slack：S_ISSUE(t) − LOAD_K(t).end | +1.20（不绑定） |
| relay | 忙时 | 0.32（duty 5%） |

**时刻对齐事实**（稳态某 tile 的绝对时间戳，µs）：WAIT_dQ 的结束 = kdq ring 提交
（84.06 / 84.09 两侧贴合）；dQ 发射释放 ring 槽 0/1 后 MAT_QDO 火车立即启动
（84.10 与 84.13 同时刻）。

### 7.2 vc_2 时代对照（vc_2 trace，注入口径）

- dVdK 逐 pass 发射前间隙：[0.99, 0.36, 0.45, 0.35, 0.44, 0.35, 0.44, 0.51]。
- MATH_SOFTMAX 2.156（vg_2/vg_5 手术前）、MATH_PDS_ACQ 0.823（split publish 前）。

### 7.3 C vs S1 A/B（two_trace_tables 包络语义，每 tile，us；C=无 split publish）

| 项 | C | S1(=vg_5) |
|---|---|---|
| K/KV load | 2.778 | 2.635 |
| MATH_PDS_ACQ | 1.739 | 2.008 |
| MATH_SOFTMAX | 3.333 | 3.519 |
| MATH_STORE | 1.213 | 1.272 |
| MAT_QDO | 5.283 | 5.056 |
| P+dS T2R/math | 8.054 | 8.946 |
| ROUTE_K | 2.010 | 2.224 |
| ROUTE_P / ROUTE_dS | 0.304 / 0.355 | 0.611 / 0.526 |
| S+dP 发射 | 1.026 | 1.028 |
| dKV T2R+atomic | 5.867 | 5.631 |
| dQ epilogue | 23.712 | 23.424 |
| TAIL | 28.0 | 27.328 |

（注意：此表与 §7.1 的同名项数字不同——聚合语义不同（包络含嵌套等待、按因子
加权），两表各自内部可比，互相不可直接比。）

### 7.4 每 token 固定成本（trace 口径，vk_1）

head ~3.3 + 慢启动 ~4 + DQ_EPI 2×11.9 + TAIL 27.7 ⇒ F_trace ≈ 31。
**修正**：基于 §8 三次尾段手术全零，release 侧真实 ΔF（对 baseline）≈ 0.10–0.15 ms
e2e（原按平均税折算的 0.35–0.43 ms 作废）。

### 7.5 供给带宽事实

- 每 tile 经 ring 流转的梯度 A 操作数：8 代 × 16KB = 128KB；另有 dO_r0 2×16KB
  经 loan。合计 160KB/tile 的 SMEM 供给流量，全部通过 **2 个 16KB ring 槽 + loan 区**。
- 每代供给成本：own-half 8KB S2S bulk + peer-half 8KB GMEM TMA（面板代）；
  kdq 代则是 GMEM 稀疏 gather（64 行 × 256B × 2 round）。
- 跨 CTA 交换总量：每 tile 仅 P 4KB + dS 4KB（DSM）。

### 7.6 baseline 内部分解（同 run baseline trace，2026-08-06；近零税）

| 项 | 值 |
|---|---|
| 周期 | 4.648 |
| reducer 忙时 | 3.98（1.985×2 part，**duty 85.7%**） |
| leader | 4.10（S_dP 1.21 + dVdKdQ 2.82，duty 88%，含内部未标记等待） |
| gather | 3.13（duty 67%） |
| math | 忙时 ~0.9；WAIT_S 3.71（大量闲置） |
| F | head 2.46 + tail ~4.3（dQ_epilogue 6.95 与末 tile 重叠）≈ 6.8/token |
| 每 token | 154.6（trace ≈ release） |

**双边对照事实**：baseline reducer 忙时 3.98/tile vs 本 kernel 2.50/tile
（rank 分域使单 CTA reducer 体积减半）。

---

## 8. 实验台账（变更 → 测量结果；全部 correctness 4/4 除非另注）

release 口径：candidate_ms @ 同日 baseline_ms（ratio）。

| 代号 | 单杠杆 | 结果 | 判决与关键细节 |
|---|---|---|---|
| v12 | （族基座）上述 20-warp 架构 | — | 现役血统起点 |
| vc_2 | score-K/dO 时间借用（dO_r0 借宿 score 区） | 10.572 @ 8.268 (1.2786) | 采纳为基座 |
| vc_3 | score-K 二次借用（loanQ） | 11.836 @ 8.430 (1.4040) | 负。r1 曾**软死锁**：尾 tile 三方环（gather 卡 loanQ acquire ← grads dV 释放 ← dQ ← 末次 kdq 会合被排在 loans 之后）；r2 重排后正确但负 |
| vd_1 | 供给环拆双生产者（W17+W19 双 lane） | 10.833 @ 8.310 (1.3036) | null 偏负。**排除了两个假说成分**：生产者串行排队 ≈0、跨奇偶 head-of-line ≈0（2 槽环按奇偶本就是两条 depth-1 链，lag-1 提交已让两奇偶 TMA 并飞） |
| ve_1 | 双 tile 宏批处理 + P/dS 第二发布面按别名寄宿 score-K | 12.917 @ 8.401 (1.5376) | 证伪。**机制侧全兑现**（trace 实测：dVdK 节拍 0.42→0.20/pass；MATH_PDS_ACQ 0.823→0.056；math 墙 4.58→2.80；drain 2.91→2.54），**但**：K(下一对) gather 须等别名读完 → +3.07/pair 串行尾；rotated schedule 被迫放弃 → +2.4/pair。净 +5.5/pair 调度税 |
| vg_1 | split publish（dS 先发/P 后发，双 pds 管线） | 10.503 @ 8.336 (1.2599) | 曾采纳；后经 C/S1 A/B（§7.3）判其净贡献≈0 且携带注入卡死触发结构，已整体退役 |
| vg_2 | exp 条带流水（8 EX2 在飞） | 10.450 @ 8.394 (1.2450) | 采纳（bit-exact 重排） |
| vg_3 | T2R 双发单 fence | 10.454 @ 8.313 | null（不在关键路径） |
| vg_4 | topk 索引批量预取（两处稀疏 gather 的行索引载入提升到行循环前） | **9.936** @ 8.376 (1.1862) | 采纳，单刀 −0.51ms |
| vg_5 | exp 全宽条带（32 EX2 在飞，结果写回原寄存器） | **9.771** @ 8.390 (1.1646) | 采纳（bit-exact） |
| vg_6 r1 | dQ epilogue 双缓冲（staging→ring 槽） | correctness FAIL 5.4% | dq_done 是早提交（只跟踪 dQ MMA），epi 开始时 ring 仍被尾部 dVdK 读（WAR） |
| vg_6 r2 | 双缓冲改借 stationary dO 面板 | 9.781 @ 8.345 (1.1720) | null：**TMA 飞行/读等待不是尾段大头** |
| vh_1 | score_kv 32KB 常驻 K → 2×8KB D128 段环，省出 16KB 给 ring 深 3 | **13.631** @ 8.552 (1.5938) | 证伪。实测 +2.2/tile：S/dP 逐段等 K（gather 是最慢供给腿 ~1.5/tile 纯工作、稀疏散射）+ loan 退役回吐 ~0.4 + pad 代协议开销 |
| （注入案） | vg_1..vg_5 家族在 IKET 注入下 GPU P0/100% 自旋卡死；排他变体：C（去 split publish）过、S1（拆共享 producer state）过、S2（并 ready mbar、留共享 state）**挂**、a_advance（留共享 state、挪 advance）**挂**、s1a 过 | — | 触发要件 = relay 两管线共享一个 pipeline state 的代码形态；CUTLASS 4.5.0 语义层无病（容器内核对：depth-1 的 producer_tail 不改 state） |
| vk_1 | vg_5 + S1 修复（拆 state） | 9.844 @ 8.605 (1.1440) | 可采 trace 的 vg_5 等价物 |
| vk_2 | **现役**：C 形态转正（split publish 整体退役） | 9.902 @ 8.609 (**1.1502**) | 与 vk_1 漂移内同水位；协议面更简 |
| vk_3 | dQ epilogue 标量店→stmatrix（T2R atom 同时换 Ld16x256b） | 9.857 @ 8.519 (1.1570) | **硬 null**（ms −0.045 但 ratio 反劣于 vk_2；两指标矛盾）。r1 曾命中预登记 build-gate（partition_D rank 断言），r2 布局代数正确、correctness 4/4 |
| M3（桌面） | dS 单份双主序（省一次 dS 存储 + 8KB） | 未上机 | 被 CG2 描述符 rank 对称锁挡死：可别名块是 ds_blocks[rank]（rank 相关偏移）。字节同构性本身成立（ds_image 两半与 dkv-B 块字节同构，中继直接 DSM ds_image+2048 为证） |
| v9（旧案） | K_dQ 复用 score-K（credit-gated peer push 机器） | +6.4ms | 死于该推送机器的经济性（供参考：与"复用 score_kv 字节"不是同一机器） |

**汇总模式（事实）**：本轮 4 项采纳（vg_1 后撤销为 3 项：vg_2/4/5）全部是零字节的
发射序/延迟手术；4 项证伪/负（vc_3、vd_1、ve_1、vh_1）全部涉及 SMEM 重新布管。

**另有平行探索**：vre_3（面板流式化换 ring 深度）在另一会话中途调试未有结论；
K2（kdq 退役，用 score_kv 字节 + DSM 替代第二次稀疏 gather）桌面工作已开（其
几何事实已并入 §4.5/§7.5），未上机。M2（FFMA deg6 exp 替换 MUFU，SOFTMAX
2.22→~1.0 的候选）数值半门已过（误差与 MUFU 同量级 1.9–3.9e-6），等"近似替换"
档终裁，未上机。D1（warp 重划 640→512）未上机。

---

## 9. 既往解释性结论（**全部是假设**，可能有错，请独立验证）

以下是团队基于 §7/§8 数据做出的推断，按提出时间排列：

1. （vc_2 era）定速环分解：周期 Σ6.848 = dVdK 供给+排空 63% / math 24% / relay 12%，
   k=1 零松弛闭环。
2. （vre_1 判决）各角色真工作 ≤3.2/周期 6.95 ⇒ 单角色减负手术无效，只能打断依赖环。
3. （vd_1 null 之后）0.4/代的发射间隙平台的剩余解释 = 每奇偶 depth-1 的
   "信用-飞行"延迟暴露；买家只剩加槽（+32KB）或删操作数。
4. （vh_1 验尸）"常驻且已预取"的操作数，其 SMEM 占用是物化的前瞻时间；改环 =
   1:1 拿前瞻换字节，仅当生产者远快于消费节拍才划算。
5. （vk_1 trace 复盘，"判决 1–5"）：pacer = W17 供给链自身饱和（6.65≈周期）；
   ROUTE_K 的真身是 kdq 会合、压在关键路径（依据 §7.1 的时刻对齐）；2 槽环的
   代 n+2 acquire 等 leader 消费代 n ⇒ 火车与消费互锁；math STORE 侵入下一拍
   与 PDS_ACQ 1.15 相关；drain 与 K 供给都不绑定。
6. （E1 落章的解读）baseline 贴着自身 reducer drain 地板跑（3.98 busy/4.648 period），
   下探空间 ≤0.6 ⇒ 追平线是物理量而非幻影；本 kernel reducer 体积减半（2.50）
   是 CG2 对 baseline 的结构性优势来源。
7. （vk_3 硬零之后）trace 的 DQ_EPI 11.9/round 被注入税放大数倍；release 侧
   真实尾段超额仅 ~0.10–0.15ms ⇒ 剩余缺口 85–90% 在稳态侧。
8. （手术阶梯，当时的 EV 排序）K2 kdq 退役（own 半区来自 score_kv 同字节、peer
   半区 8KB/round 走 DSM，预期 −0.8~1.5/tile）→ K3 环深 3（K2 之后才有意义，
   资金候选 = dO 面板段流或 staging 迁址）→ K4 M2 终裁 → K5 ve_1 复活（需真
   24KB）。K1（尾段）已死于 vk_3 硬零。
9. （注入案机制）共享 pipeline state 的 lowering 后代码形态与 trampoline 相互作用
   （S2/a_advance 挂、S1/s1a 过为判据）；对 IKET 工装可报 bug。

---

## 10. 术语表

- **tile**：一个 N64 KV 块（64 个 topk 行）；每 token 32 个，从高索引往低处理。
- **round**：D512 特征维的一半（D256 cluster / D128 每 CTA）；dQ、dVdK、kdq 均按
  2 round 处理。
- **pass**：dVdK 的 H64 半份归约（H_PASSES=2）。
- **代（gen）**：ring 槽的一次填充-消费周期；稳态每 tile 8 代（g0/g1=kdq、
  g2/g3=Q_r0、g4/g5=dO_r1、g6/g7=Q_r1）。
- **quadrant**：梯度 A 操作数的 [D128×H64] 切片（16KB bf16），own-h-half 来自
  stationary 面板的 S2S 拷贝，peer-h-half 来自 GMEM TMA。
- **panel（面板）**：token 常驻的 Qᵀ/dOᵀ [D512×H64]（own-h 半份），64KB×2。
- **kdq**：dQ GEMM 的 A 操作数镜像 [N64×D128]×2 round——与 score_kv 同一批 KV 行
  的不同列窗（D-major 布局），现由 gather 警组从 GMEM 第二次稀疏 gather 得到。
- **loan（时间借用）**：dO_r0 的 2 个 quadrant 暂住 score 区域的机制（vc_2 引入）。
- **plane（转置平面）**：全部 GEMM 按转置形式组织（Sᵀ/dPᵀ/dVᵀ/dKᵀ/dQᵀ）。
- **rank / own-half / peer-half**：cluster 内 CTA 编号（0/1）；CG2 下操作数与结果
  按 rank 分半（B 按 n 分半、A/C 按 M 分半、面板按 h 分半）。
- **rotated schedule（轮换调度）**：tile t 的 S/dP 先发，随后发 tile t−1 的梯度。
- **drain（排空）**：reducer 把 dKV 部分和从 tmem 取出并原子加进 GMEM workspace。
- **baseline**：同仓库的 1-CTA 参照实现（CG1，无跨 CTA 交换；其内部分解见 §7.6）。
- **IKET / 注入 / span**：在运行中给 kernel 打 trampoline 采集软件注解时间线的
  工具；span 是命名时间段，名额上限 31。
- **波（wave）**：grid 中 token 数 / 74 个 cluster 槽位 = 55.35。
- **candidate/baseline_ms**：release 口径的被测/参照 e2e。
- **kdq_barrier / math_barrier**：named barrier（§4.4）。
- **mbarrier phase/parity**：硬件到达计数屏障的相位语义。
- **UMMA-tracked release**：管线信用由 MMA 完成事件（tcgen05 commit）而非线程
  显式 arrive 释放。

## 11. 开放事实清单（截至 2026-08-06）

1. leader 的 0.41 / 0.54 两段等待的内部构成未直接测量（观测债，§5.3）。
2. vk_1_trace 的 RK_ACQ/MAT_ACQ/MAT_WAIT 细分读数已采集成功但未完成解析。
3. K2 的两个前置桌面项未闭合：score_kv swizzle 在 MN-major A 视图下的 CG2 描述符
   合法性推导（K2b 的门）；不闭合则退化为 8KB S2S 拷贝方案（K2a）。
4. M2 等用户对"近似替换"档的终裁。
5. ROUND_STAGES=2→3 无已验证的 16KB 资金来源（vh_1 的来源已证伪；vre_3 未有结论）。
6. W19（32 线程）与 math 警组 duty ~65%、reducer duty 37% 的富余未被利用。
7. IKET 局部税率未标定（只有全 kernel 平均 38% 与"指令密集区更高"的间接证据）。
