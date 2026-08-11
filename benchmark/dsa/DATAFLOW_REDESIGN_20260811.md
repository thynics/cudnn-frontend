# 算法级数据流重设计：GR100 DSA bwd <3.12 µs/tile 战役（2026-08-11）

**前置台账**：`python/cudnn/deepseek_sparse_attention/sparse_attention_backward/E3_GR100悬崖税与native异常台账_20260809.md`
（k4–k21 十八轮上机，双翼墙定界完成）。本文是其结论（"<3.12 需要算法级新数据流"）的设计接续。

**纪律**：本文所有量级论证锚定台账实测数；无实测支撑的机制一律标注**待上机裁决**并给出裁决实验。
上机执行走 proxy 委托唯一通道，合同 checkout-at-runtime + 引号 heredoc + 预登记判读带（k 系列铁律）。

---

## 0. 目标、口径与事实锚点

### 0.1 目标与口径
- **目标**：steady < 3.12 µs/tile（baseline 单 CTA 翼的 per-tile 边际率）。
- **steady 口径 = topk 斜率**：wall 对 topk 做扫描后取每 tile 边际时间；固定成本（init/尾巴/launch）**按定义不计入**。
  自检（k19 bl 臂实测 wall 0.9037/1.6351/4.4544 ms @topk 128/512/2048）：
  (4.4544−1.6351)ms ÷ 24 tiles ÷ 37.93 波 = **3.097 µs/tile** ≈ 3.12 ✓（口径闭合）。
- **推论（本文的选刀准绳）**：候选数据流必须改变 **per-tile 边际成本**；只削固定成本的方案在本口径下收益恒为 0。

### 0.2 双翼墙（台账终审，全部实测）
| 翼 | 地板 | 墙 |
|---|---:|---|
| CG2 双 CTA（m51 冠军） | 4.0877 µs/tile | 协议税（barrier stall 8× vs baseline：11.48 vs 1.41）+ spill（local_ld 34.2M vs 1.54M = 22×）；物理三重顶格：TMEM 512/512、SMEM 334,848B、相位铁律。十轴旋钮穷尽（k4–k19）。 |
| baseline 单 CTA | 3.12 µs/tile | gather cp.async 的**本征内存延迟**（long_scoreboard 比 11.0，k18/k21）；加深被双杀：k19 字节（stage=2 需 346,112B > 物理 334,848B）、k21 协议（D256 半 tile 随 topk 恶化 −4.4%/−19.9%/−38.2%）。 |

### 0.3 问题形状与代码级事实（本文推理的底座）
- 负载：seqlen 4096、GQA128（nheads=128，**单 latent KV 头**）、D=Dv=512 bf16、block_tile=64、topk∈{128..2048}、batch=1。
- **KV cache 总量 = 4096 行 × 512 × 2B = 4 MB**（`benchmark_dsa_sparse_attention_backward.py:47`）。
- **topk 索引每 token 内无重复**：harness 用 `argsort` 取前 topk（同文件 :55），且 top-k 语义本身即无放回选择——**token 内跨 tile 不存在重复行**，这是构造性事实，不依赖数据分布。
- baseline grid = (4096 token, 2 head-block, 1) = **8192 CTA / 216 SM = 37.93 波**（与 k18 NCU 波数 37.9 闭合）；
  每 CTA：20 warps（4 gather + 4 compute + 8 reduce + 4 infra）、96 regs、1 CTA/SM。
- gather 现状（`dsa_bwd_sm100_baseline.py`）：`load_KV`（:1270）逐 tile `producer_acquire → 64 行 × _copy_kv_row(cp.async 128b×8 lane, LoadCacheMode.GLOBAL=.cg 旁路 L1) → cp_async_wait_group(0) → barrier → producer_commit`；
  `load_mma_K_stage = 1`（:137）；K 缓冲直到 math 侧发完 **dQ MMA** 才 `consumer_release`（:1628）。
  ⇒ **stage=1 下 gather(t+1) 与 tile t 的整条 math 链完全串行**——这就是 ls 11.0 暴露在斜率里的结构位置。
- 每 tile gather 字节 = 64 行 × 1024B = **64 KB**；@topk2048 全局 gather 流量 = 8192 CTA × 32 tiles × 64KB = **17.2 GB / 4.4544 ms ≈ 3.86 TB/s 的 L2→SMEM 读**（KV 仅 4MB，全部 L2 命中；DRAM 侧≈0）。

---

## 一、候选 A：topk 批组化（gather 重构）

### 1.1 机制审计：批组化家族逐项定价（先杀死不该做的）

| 子方向 | 判决 | 依据（实测/构造性） |
|---|---|---|
| (a) token 内跨 tile 排序/去重 | **死，0 收益** | 构造性：top-k 无放回，token 内 topk 索引互异（§0.3）。没有可去的重。 |
| (b) 行聚簇装载（排序后合并相邻行为大事务） | **本 harness 死；生产长上下文再议** | 每行 1024B = 8×128B sector **全消费**——排序不减少 L2 sector 事务数；KV 4MB 全 L2 驻留，无 DRAM 行局部性可赚。仅当 KV > L2（生产 128K 上下文：128K 行 × 1KB = 134MB > 132MB L2）时聚簇才改变 DRAM 行为。→ 生产注记，本 harness 不可测。 |
| (c) L2 驻留分块 | **无事可做（本形状）** | 4MB ≪ 132MB，KV 已 100% 驻留。生产注记：驻留上限 ≈ 132MB/1KB ≈ 13.5 万行；同时 Q/dO 流式流量 512MB/launch 会冲刷 L2，长上下文需配 evict 优先级提示（cp.async .cg + TMA evict_first）。 |
| (d) 跨 head-block 去重 | **真实 2× 冗余，但有前科定价** | 同 token 的两个 CTA（head-block 0/1）gather **同一份 64KB**——确定性成立，与数据无关。全局 gather 流量可 2×→1×。但 CG2 翼正是"每 token 一 cluster、gather 摊给 128 头"的设计（该牌已打），结局是协议税 8× 把省下的全吃回（k18）。任何 cluster 化去重必须先证明自己不复刻这笔税。 |
| (e) 跨 token 去重/批组 | **上限太薄，机制昂贵** | harness 索引随机 ⇒ 相邻 token 期望重叠 = topk²/4096 行：topk128→4 行（3.1%）、512→64 行（12.5%）、2048→1024 行（50%）。可赚上限 = 重叠率 × gather 份额；且利用它需要跨 CTA 协调（token=CTA 边界）或 SMEM 里同时驻两个 token 的状态（+131KB Q/dO，字节死）。真实模型的索引局部性可能远高于随机——**待实测**：离线直方图（dump 生产 topk_idxs，算相邻 token Jaccard），0 行 GPU 代码。 |
| (f) 换运输机器（TMA tile::gather4） | **死（B200 判例，机制同构）** | v_gpt_1 r2b：correctness 4/4 全对、性能 +1.8ms——TMA 引擎是**串行队列**，512B 级小事务在它上面排队且拖累面板大事务（VG_BUILD_LOG 验尸）；k21 补刀：cp.async 直达形态就是该物化的地板。GR100 复测优先级极低。 |

**审计结论**：任务书列举的"批组化/排序/去重"在本形状上大多被构造性事实或实测判例杀死。
gather 侧真正剩下的两个杠杆是：**① 把 gather 延迟从斜率里摘除（重叠而非减量）；② 把 2× head-block 冗余流量摘除（仅当 L2 带宽是墙时付钱）**。

### 1.2 主杠杆：stage=2 复活——k19 死刑的字节账复核

k19 的死刑判决基于 blk2 实测 SharedStorage = 346,112B。**复核这本账**：

- HEAD 的 SharedStorage 逐项（`dsa_bwd_sm100_baseline.py:424-446`，stage 全 1）：
  sQ 65,536 + sK 65,536 + sdO 65,536 + sP 8,192 + sdS 8,192 + mbars/sLSE/sSum_OdO/对齐 ≈ 2,048
  = **215,040B**——与 k18 NCU 动态 SMEM 215KB、以及 346,112 − 131,072 = 215,040 **双向闭合**。
- blk2 的 +131,072 = **sK 与 sK2 各 +65,536（物理双份）**（k19 归因原文）。
- **但 HEAD 代码里 sK2 是零字节别名**：`sK_2_ptr = cute.recast_ptr(sK.iterator, K_smem_layout_staged_2.inner)`（:844-845）——同一块物理 K 字节的 A-视图 recast，不占 SMEM。
  ⇒ **若 stage=2 下别名可保持，字节账 = 215,040 + 65,536（sK 第二 stage） = 280,576B < 334,848B，余 54,272B**。
  即使 P/dS 的 store 视图（:345-347、:366-368，均 key 在 `load_mma_K_stage` 上）连带把 sP/sdS 翻倍（+16,384），也只到 **296,960B，仍余 37,888B**。
- **k19 的死刑只对"别名破裂的实现形态"成立**。别名在 stage=2 的合法性（staged recast 的 swizzle 组合是否仍字节等价）是纯编译期问题，**一次本地编译即可裁决**，不烧 GPU 窗口。
- stage=3 = 280,576 + 65,536 = 346,112 > 334,848：**stage=2 是这条轴的物理终点**，没有后续加深幻想。

**作用机制（这刀为什么动斜率）**：stage=1 时每 tile 边际成本 ≈ gather 延迟 + math 链（§0.3 串行结构）；
stage=2 后 gather(t+1) 在 math(t) 开始时即可启动，斜率 → **max(gather, math) + ε**。
这不是减少工作，是把 k21 判定的"本征内存延迟"从关键路径上**重叠掉**——正对 baseline 翼的墙。

**预期量级（预估带，机制份额未分解，待上机裁决）**：
baseline warp 时间中 long_scoreboard ≈ 11.0 × 22.6% ÷ 5 warps/SMSP ≈ **50%**（k18 数据，按 k9 归一法）。
若其主体是 gather 等待且被完全重叠：斜率理论下限 ≈ math 链 ≈ 1.6 µs/tile；保守计入 mbar 双缓冲协议 ε 与部分不可重叠份额，
**预估带 −0.4 ~ −1.1 µs/tile（落点 2.0–2.7）**。任一落点均 <3.12 且 <m51 4.0877。

**代价核查（悬崖税）**：280,576B > compat 上限 232,448B ⇒ 必须 oversized 启动模式（8KB L1）。
CG2 的悬崖税 +0.431 µs/tile 是在家族 local 流量 ~12.8GB/launch、miss 48→86% 下付的（E3 台账 C3）；
baseline local_ld 仅 1.54M 条/launch（m51 的 1/22），且 gather 走 .cg 不经 L1
⇒ 线性外推税 ~0.02 µs/tile 量级。**此数为外推非实测——k22 设隔离臂直接定价（见 §3.3）**。

**最小可测切片 S1（baseline 底盘）**：
| 改动 | 位置 | 规模 |
|---|---|---|
| stage 旋钮 1→2（env 门控 `DSA_BL_KSTAGE2=1`） | `dsa_bwd_sm100_baseline.py:137` | ~5 行 |
| SMEM 断言 227KB → 334,848（仅旋钮开时） | 同文件 :422、:448-450 | ~5 行 |
| oversized 启动属性移植（rubin_1 e3pad/k4 臂同款 plumbing） | baseline `__call__` launch 处 + `_interface_sm100.py` | ~30-60 行 |
| 别名合法性编译期 gate：断言 stage=2 下 SharedStorage ∈ {280,576, 296,960} | 本地编译，不上机 | ~5 行 |
| （备用）若别名破裂：手工 per-stage recast（对每 stage 以显式字节偏移重建 A-视图） | :844-845 邻域 | ~10-20 行 |

**风险点**：① 别名破裂且手工 recast 也不字节等价 → 回到 k19 死刑（本地编译期即知，零窗口成本）；
② P/dS store 视图随 `load_mma_K_stage` 翻倍引发的索引错位（:345-347/:366-368 需与 sP/sdS 分配对齐，correctness 门兜底）；
③ oversized 悬崖税超外推（隔离臂定价）；④ mbar 双缓冲给 gather warp 加发射税（k15 教训的镜像——但这里是编译期静态 stage，无动态切片算术，属 K1a 史训的安全侧）。

### 1.3 次级杠杆：cluster-2 跨 head-block gather 去重（S3，条件启动）

- **机制**：把同 token 的两个 head-block CTA 编成 cluster-2；gather 行分工 32/32，各自 cp.async 入本地 SMEM 后以 **LSU st.async 跨写对端**（不是 TMA push——v_gpt_2 的 owner-push 串行化死刑是 TMA 引擎判例）。全局 gather L2 读流量 2×→1×（3.86→1.93 TB/s @topk2048），每 CTA gather 发射数减半。
- **付钱条件**：仅当 L2 带宽/队列是墙。若 S1 落地后斜率已由 math 链主导，流量减半不再动斜率。
  ⇒ **先测后做**：k22 附带 ncu 臂读 `lts__t_sectors` 带宽利用率（0 行 kernel 代码）定界。
- **风险定价（两张前科）**：CG2 协议税 8×（k18）警告 cluster 耦合的 designed-wait 成本；v_gpt_2（+2.09ms）警告跨 CTA 推送把并行运输串行化。本切片的协议面必须限定为：每 tile 一次 cluster barrier（复用现有 `load_KV_sync_barrier` 升级为 cluster 级），**不引入环/信用协议**。
- **切片规模**：`load_KV`/`_load_kv_rows`/`_copy_kv_row`（:1164-1385）+ launch cluster 参数，~80-150 行；正确性门同 k1 标准。
- **预期量级**：若 L2 BW-bound 成立，上限 = gather 份额 × 流量减半收益；在 S1 之后叠加时预估 ≤ −0.3 µs/tile。**待上机裁决**。

---

## 二、候选 B：persistent-CTA 多 token 软流水

### 2.1 机制分解：它动什么、不动什么

设想：216 常驻 CTA（baseline 翼）或 108 常驻 cluster（CG2 翼）经原子工作队列迭代多个 (token, head-block) 工作项，跨 item 保持 TMEM 分配与管线状态，消除 per-block init/teardown。

**逐项对账**：

1. **固定成本削除（真实且可观，但按 steady 口径 = 0）**
   - 固定成本实测：baseline ~18 µs/cluster、compat 30.9、native 39.4（E3 台账 §1）；
     m51 @topk512 的 item 生命周期 64.3 µs 中，steady 部分仅 8×4.09=32.7 µs，
     尾巴 14.4 µs + 头部 ~17 µs ⇒ **固定成本 ≈ 49% wall @topk512**（k16 role 时间线）。
   - k16 已证 gather/producer warp 在尾巴 14.4 µs 期间完全空闲（45.0-48.8 收工 vs drain 64.3）
     ⇒ 软流水把 item(i+1) 的 gather/Q/dO 装载藏进 item(i) 尾巴，**wall 上限 −20~−27% @topk512**（回收 min(init, tail)）。
   - **但 steady = topk 斜率，固定成本被口径整体扣除**：wall(topk) 曲线整体下移，斜率不动。
     **明说：纯 head/tail 重叠形态的 persistent-CTA 对本战役目标（<3.12 steady）收益恒为 0。**
   - SM100 前科同判：persistent 2-token cluster 评估收益 −0.55~0.66ms wall（封存·未死）；
     Offloading 审计实测冷 tile 仅 +0.5-0.6 µs → 摊 0.015-0.08 µs/tile，低于行动阈值。

2. **steady-moving 变体 = 跨 item 的 tile 级交错**（math(itemA, tile t) 的环等待被 gather(itemB) 填充）
   —— 这才动边际率，但**资源审计判死**：
   - m51：TMEM 512/512 满格（S/S1+DP/DP1+DQ 2×128+DKV 2×64，k10 盘点）、SMEM 334,848 顶格
     （slim 后余 58,368B）、regs 96×640 顶死动态池——**第二个 item 的累加器与环状态无处安放**。
   - baseline：第二 item 需 Q+dO+K ≥ 196,608B；oversized 全开也只余 119,808B（S1 落地后余 54,272B）。
   - 共享缓冲交错（item B 复用 item A 的 stationary Q/dO 槽）不成立：Q/dO 在 item 内每 tile 都被 MMA
     消费，直到 :1704 才 release——交错窗口只存在于 item 边界 = 又退化为固定成本重叠。
   - **结论：在两个现有底盘的物理约束内，persistent-CTA 不存在动 steady 的形态。**
     它是"把 issue_active 12.6%/22.6% 的空转用第二租户填满"的正确直觉，但该直觉的资源前提
     （放得下第二租户）恰是 k9/k10/k19 反复实测顶格的三重物理墙。

3. **次级红利核查**：跨 item 的 L2 暖驻留——KV 仅 4MB 全驻留，无增益；launch 开销——单 launch 8192 CTA
   本就一次 launch，无 per-item launch 可省。均为 0。

### 2.2 最小可测切片 S4（仅当需要交付 wall 数字时启动）

- **底盘**：m51（固定成本占比最大、k16 时间线已给出重叠窗口的直接证据）。
- **改动**：grid 4096 clusters → 108 persistent clusters + GMEM 原子 ticket 取 (token) 工作项；
  跨 item 保持 TMEM alloc（省 alloc/dealloc 与 UVIRTCOUNT.DEALLOC rendezvous）；
  item 边界处 gather/producer warp 提前进入下一 item（复用 k16 的 per-warp 时间戳验证重叠达成）。
  规模 ~150-300 行（主循环外包一层 while + 三处 pipeline state 复位）。
- **风险点**：tmem_holding_buf 跨 item 协议、mbar/pipeline phase 复位纪律（相位铁律的 item 边界版）、
  role 时间线 trace 语义随 item 迭代失效（trace 工具需加 item 维度）。
- **预登记判读带**：steady 斜率 ±0.05 带内 = **steady-neutral 证实**（本文 §2.1 机制判断闭合）；
  wall @topk512 ≤ −15% = wall 交付价值成立；斜率若意外 ≤ −0.10 ⇒ §2.1 资源审计有漏，回炉验尸（这将是高价值意外）。

---

## 三、对比裁决

### 3.1 排序表

| 候选 | 预期 steady 收益 | 实现风险 | 最小切片工时 | 裁决 |
|---|---|---|---|---|
| **A-S1：stage=2 复活（别名审计 + oversized 上舱）** | **−0.4~−1.1 µs/tile（预估带，唯一有机制直接动斜率且字节账成立的刀）** | 中低：编译期 gate 先行零窗口排雷；协议不变（无新环/新 designed-wait）；悬崖税有隔离臂定价 | 旋钮+断言 ~10 行，oversized 移植 ~30-60 行；k19 合同模板直接复用 | **先做** |
| A-S2：L2 带宽定界探针 | 0（纯测量） | ≈0：ncu 一臂，0 行 kernel 代码 | 搭 k22 窗口顺风车 | 与 S1 同窗 |
| A-S3：cluster-2 gather 去重 | ≤−0.3 µs/tile，**仅当 S2 判 BW-bound** | 中高：CG2 协议税 + v_gpt_2 串行化两张前科 | ~80-150 行 | 条件启动 |
| B-S4：persistent-CTA 软流水 | **0（steady 口径，§2.1 机制判死）**；wall −15~−27% @topk512 | 高：TMEM/mbar 跨 item 协议 + trace 语义重建 | ~150-300 行 | 降级为 wall 专项，第三顺位 |

### 3.2 推荐

**先做候选 A 的 S1**。理由链：
1. 它是两条候选中**唯一**在机制上改变 per-tile 边际成本、且资源账（280,576B < 334,848B）经复核成立的刀；
2. 它的最大不确定性（别名合法性）在**本地编译期**即可裁决，不消耗排队窗口——这在 0328 单节点排队成为第一瓶颈的现状下是决定性优势；
3. 它直接攻击 k21 定界的 baseline 翼之墙（gather 本征延迟），而不是绕开它；
4. persistent-CTA（候选 B）对战役指标收益为 0（§2.1 明证），只应作为 S1 落地后的 wall 交付附加项。

### 3.3 k22 验收判决带（预登记，S1 切片）

**臂设计**（一个 0328 窗口，串行三臂 + 一 ncu 腿；合同 checkout-at-runtime，全臂 correctness 门 = k1 标准 dq=0 / dkv≤0.00195）：
| 臂 | 构型 | 用途 |
|---|---|---|
| bl | HEAD baseline（compat, stage=1） | 对照锚（预期复现 3.10±0.03 斜率） |
| blo1 | baseline + oversized 启动，stage=1 | **悬崖税隔离剂量**（唯一变量=启动模式） |
| blo2 | baseline + oversized + stage=2 | 主判决臂 |
| ncu@bl | lts 带宽 + stall 分解（k9 归一法） | S2 定界：L2 BW 利用率 ⇒ S3 去留 |

**门与带**：
- **G0（本地，先于入队）**：stage=2 编译期 SharedStorage ∈ {280,576, 296,960}B 断言过 = 别名保持；
  破裂则走手工 recast 备用路径；两路皆死 ⇒ k19 死刑维持原判，S1 关闭、S3/S4 顺位递补，**不烧窗口**。
- **G1**：三臂 correctness 全过；blo1/blo2 的 NCU launch carveout 实测 = 335,872B（oversized 生效铁证，E3 C2 同法）。
- **G2 判决带（steady 斜率，topk 128/512/2048 三点）**：
  - blo1 − bl（税剂量）：≤ +0.05 预期内；**> +0.15 ⇒ oversized 路线红灯**，S1 终止（compat 内无字节路径），转 S3/S4；
  - blo2 − bl（主判决）：
    - **≤ −0.40**（落点 ≤2.7）：major_win——<3.12 达成方向确认；下一步斜率分解 trace（gather 是否已完全遮蔽）+ 清扫；注意 stage=3 物理不可行（§1.2），后续刀在 math 链侧；
    - **−0.40 ~ −0.10**：effective——部分重叠兑现；读 per-warp 时间戳定位残余暴露段再迭代;
    - **±0.10 带内**：机制未兑现——读 ncu 腿分家：L2 BW ≥~70% 峰值 ⇒ 判 BW-bound，S3 升主力；BW 低 ⇒ gather 等待不在斜率主项（k18 的 ls 11.0 需重新归因），回台账重开定位；
    - **> +0.10**：回退——验尸顺序：悬崖税超模（对照 blo1 剂量）→ mbar 双缓冲发射税（SASS 对比 gather warp 热点）。
- **判决记账**：无论落点，blo1−bl 的税剂量与 blo2 的斜率都写回 E3 台账续章（k22 条目），双指标（斜率+wall）同录。

### 3.4 后续队列（k22 之后的分支树）

- blo2 达标 ⇒ S2 结果决定是否叠 S3（去重）再压；wall 交付需求出现时启动 S4（persistent）；
- blo2 带内/回退 ⇒ 按 G2 分家走 S3 或回定位；S4 始终不因 steady 目标启动；
- 生产化注记（不占用当前窗口）：长上下文（KV > 132MB L2）时 §1.1(b)(c) 的聚簇/驻留分块从"判死"转为"待评估"，
  且跨 token 重叠率需用真实模型索引直方图重测（§1.1(e)，离线，0 GPU 代码）。

---

*本文档由 GR100 战役台账（k4–k21）派生；所有引用数字可在台账对应章节按 job id 溯源。*
