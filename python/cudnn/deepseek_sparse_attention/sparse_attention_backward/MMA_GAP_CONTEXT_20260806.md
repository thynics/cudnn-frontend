# DSA 反向 2-CTA kernel：稳态 MMA gap 问题 —— 完整上下文

**日期**：2026-08-06 ｜ **版本**：v2（经 7 路独立代码校验修订）｜ **现役 kernel**：vk_2
**读者**：无代码访问权限的推理 agent

## 0. 阅读说明

- 本文档只陈述**事实与测量**。既往团队做出的解释性结论全部集中在 §9，并明确标注为
  "假设"——它们可能是错的，请独立推理。
- v2 修订说明：v1 的全部结构断言已由 7 个独立校验进程对照 vk_2 源码逐条取证
  （128+ 条断言，每条有行号证据）；v1 中 8 处错误与 40+ 处遗漏已修正/补全。
  trace 测量数字另经内部算术一致性校验。
- 所有时间若无特别说明，单位为 µs；e2e 单位为 ms。
- 存在两种测量口径（release / trace），定义与换算陷阱见 §5。**两口径数字禁止直接互算。**
- 每个测量都标注了来源。术语表在 §10。

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

（两段无 span 等待的**代码侧候选门序列**是已知的，见 §4.5 leader 段；各门的时长
归属未测量。）MMA 单元本身是异步的；上述 gap 是**发射流**的间隙。周期 6.80 与各
角色忙时的差值构成了每 token 的主要时间超额（对 baseline 的分解见 §7.6）。

### 1.3 剩余缺口的构成（测量事实）

- vk_2 与 baseline 的 e2e 差 = 1.29 ms ≈ 每 token 23µs（release 口径，55.35 波）。
- 尾段（dQ epilogue + 收尾）经三次独立手术（§8 vg_6/vk_3）全部测得 ≈0 收益后，
  修正估计其真实超额 ≈ 0.10–0.15 ms。
- 其余（~85–90%）在稳态期的 32 个 tile 循环里。

---

## 2. 计算任务（数学事实）

### 2.1 形状与数据类型

- DeepSeek Sparse Attention 反向。H=128 头，D=D_v=512。
- 测试负载配置（非源码常量；源码按 max_topk 参数化）：topk=2048，S_kv=4096，
  S_q=1。每 token 一个 2-CTA cluster，一次 grid 启动覆盖全部 token。
- 输入/输出 bf16，累加 f32。softmax 统计量（LSE、Σ(O·dO)）由前置小 kernel 算好，
  f32 存于 workspace。

### 2.2 稀疏结构

每 token 有 topk 个 KV 行索引（`mTopkIdxs`，运行时由 `mTopkLength` 截断并夹取到
[0, 索引表长度]）。topk=2048 时切成 **32 个 N64 tile**（每 tile 64 行）。KV 行在
GMEM 中不连续（gather 按索引逐行读取）。tile 遍历顺序为**索引从高到低**。
越界/无效行（global_n ≥ topk 或索引 <0）在 SMEM 中**零填充**，使 GEMM 见到精确
零——这是 lengths/holes 用例正确性的机制。tile_count==0（all_empty）时 math 角色
经旁路直接写全零 bf16 dQ，不走 tmem epilogue。

### 2.3 五个 GEMM（转置平面公式）

每 tile 依次需要（记号：n=tile 内 KV 行 0..63，h=头 0..127，d=特征 0..511）：

| GEMM | 公式 | C 形状（cluster） | 说明 |
|---|---|---|---|
| G1 | Sᵀ[h,n] = Σ_d Q[h,d]·K[n,d] | H128×N64 | K 分 4 个 D128 chunk 累加 |
| G2 | dPᵀ[h,n] = Σ_d dO[h,d]·V[n,d] | H128×N64 | V≡K（同一张量） |
| G3 | dVᵀ[d,n] += Σ_h P·dO | D256×N64，2 D-round × 2 H-pass | 与 G4 共槽累加（见下） |
| G4 | dKᵀ[d,n] += Σ_h dS·Q | 同上 | 同上 |
| G5 | dQᵀ[d,h] += Σ_n K[n,d]·dS[n,h] | D256×H128，2 D-round | **跨全部 32 tile 累加** |

**G3/G4 的融合累加（重要）**：dV 与 dK 并非独立累加——每个 D-round 共用**同一个**
dKV tmem 槽，4 个 pass 依次为 dV-h0（开槽，acc=False）→ dV-h1 → dK-h0 → dK-h1
（全 acc=True）。落盘的是**合并梯度 dKV = dV + dK**（V≡K 的直接推论），workspace
也只有一个 mdKV_acc。

中间的逐元素计算（math）：P = exp2(S·scale·log2e + LSE)，dS = P·(dP − Δ)·scale，
并将 P、dS 转 bf16 写入 SMEM 供 G3/G4/G5 消费。

### 2.4 梯度落盘方式

- **dQ**：常驻 tmem，32 个 tile 累加完后统一 epilogue（§4.6）。
- **dKV（=dV+dK 合并）**：每 tile 算完即由 reducer warps 从 tmem 取出，以
  `red.global.add.v4.f32` 原子加进 GMEM 的 f32 workspace（按 topk 行索引 scatter，
  谓词 kv_index≥0）；kernel 族结束后另有 canonical 转换步骤产出 bf16 终值。

### 2.5 正确性契约

四个用例（dense / lengths / holes / all_empty，定义在 harness 侧）全部 PASS 才算
正确。历史 PASS 值参考量级：dense dq≈0.004、dkv≈0.0097、dsink≈0.0011；
lengths dq≈0.012、dkv≈0.073。

---

## 3. 硬件与平台事实（B200 / SM100）

- SMEM 每 CTA 可用上限 **232,448 B**。当前 V2 共享内存 struct 实际分配
  **231,424 B**（= mbar 头部 308 B + 对齐 padding 716 B + 数据 payload 229,888 B +
  尾部对齐 512 B），**真实 slack = 1,024 B**。（注意：229,888 只是 11 个数据字段
  之和；任何 ≥1KB 的新增字段都会超限。）
- 每 SM tmem 512 列；tcgen05 MMA。
- **2-CTA cluster**，全部 GEMM 用 cta_group::2（CG2）：C 沿 M 维跨两 CTA 各半；
  B 沿该 GEMM 的 N 维按 rank 分半。工程事实：CUTLASS 的 CG2 描述符要求操作数在
  两个 CTA 的 SMEM **同地址**（rank 对称）；按 rank 加不同偏移的布置会破坏描述符
  构造（§8 M3 行有一次被此规则挡死的桌面判决）。
- DSM：cluster 内跨 CTA SMEM 写（`cp.async.bulk` s2cluster + mbarrier expect_tx）。
  同一指令也可映射回本 CTA 做**本地 SMEM→SMEM bulk 拷贝**（供给侧大量使用）。
  实测中继路由 4KB 一次 ≈ 0.17（trace 口径）。
- TMA bulk tensor copy（GMEM↔SMEM）；`cp.async` 逐行拷贝（gather 用，256B 粒度）。
- named barrier、mbarrier（含 phase/parity 与 tx-count）、`setmaxnreg`。
- B200：148 SM = 74 cluster。本负载 **55.35 波**。SMEM 占用决定每 SM 同时只驻
  1 个 CTA（无 co-residency）。
- kernel 启动契约：每 token 一个 cluster，grid 一次启动全部 token。
  **此契约是规格，不可更改**（persistent-cluster 类方案已被否决，§6）。

---

## 4. 现役实现架构（vk_2 = 血统 v12→vc_2→vg_2/4/5，split publish 已退役）

### 4.1 线程组织（每 CTA 640 线程 = 20 warps）

| warp | 角色 | 寄存器 | 职责 |
|---|---|---|---|
| w0–3 | **gather**（128 线程） | 48 | K tile 稀疏 gather + kdq 镜像 gather + loan 填充（warp0） |
| w4–7 | **math**（128 线程） | 128 | S/dP 的 T2R、softmax 反向、P/dS 发布、dQ epilogue |
| w8–15 | **reduce**（256 线程） | 128 | dKV 从 tmem 取出 + GMEM 原子加 |
| w16 | **leader**（MMA 发射） | 48 | 全部 GEMM 发射与管线信用管理 |
| w17 | **load**（W17） | 48 | 面板装载 + ring 槽供给 + kdq 会合对端 |
| w18 | **relay**（W18） | 48 | **仅 lane 0 单线程工作**：P/dS 跨 CTA DSM 路由 + pds 管线的提交/收尾 |
| w19 | 空闲 | 48 | 无任何角色分支（仅走公共尾段） |

寄存器池恒等式：256×48（gather + w16–19）+ 384×128（math + reduce）= 640×96。

### 4.2 SMEM 地图（每 CTA，bf16=2B）

| 区域 | 大小 | 内容 | 生命期 |
|---|---|---|---|
| stationary_q | 64 KB | Qᵀ 面板 [D512×H64 own-h-half]（token 常驻） | 每 token 装一次（TMA） |
| stationary_do | 64 KB | dOᵀ 面板（同上） | 同上 |
| score_kv | 32 KB | 当前 K tile 的 own-n32×D512（S/dP 的 B 半份） | 每 tile 重填（gather） |
| round_buf_a | 16 KB | ring 槽 0（h0 类的代都进此槽） | 每 tile 4 代 |
| round_buf_b | 16 KB | ring 槽 1（h1 类的代） | 每 tile 4 代 |
| p_blocks | 8 KB | P 的 dKV-B 消费形态：2 个 [own-n32×h64] 块（own-h 块本地写 + peer-h 块 DSM 到货） | 每 tile |
| p_xchg | 4 KB | **出站暂存**：math 的非 own-n 警对写入 [peer-n32×own-h64] 的 P 出站块，relay 从此发出 | 每 tile |
| ds_image | 8 KB | dS 的 **[own-h64 × 全 n64]** 镜像 = **dQ 的 B 半份（纯本地）**，兼 DSM 出站源；其 ±2048 元素两半 = n-半块，各与 dkv-B 块字节同构（swizzle inner 相等有断言） | 每 tile |
| ds_blocks | 8 KB | dS 的 dKV-B 消费形态：2 个 [own-n32×h64] 块（own-h 块本地写 + peer-h 块 DSM 到货） | 每 tile |
| ds_xchg | 4 KB | **死字段**（v12 P1b 起无读无写，保留分配） | — |
| stats | 0.5 KB | 列 0=scaled LSE、列 1=Δ（f32，[H64×2]） | 每 token |
| mbar 头部 | 308 B | **38 个 Int64**（37 个 mbarrier + khot_seq 计数器）+ tmem_holding_buf Int32 | — |

**别名/借用**：dQ epilogue 的 32KB staging 别名在 score_kv 上（score_kv 死后复用）；
dO_r0 的两个 16KB quadrant 经"时间借用"（loan）住在 score_kv 的两半区（+0 与
+8192 元素处）。

### 4.3 tmem 列地图（512 列，f32）—— v1 此表整体有误，以下为核实值

| 列 | 内容 |
|---|---|
| 0–31 / 32–63 | S（ping / pong） |
| 64–95 / 96–127 | dP（ping / pong） |
| **128–255** | **dQ round0**（跨 tile 累加，常驻） |
| **256–383** | **dQ round1**（同上） |
| **384–447** | **dKV 槽 0**（当前 tile round0 的 dV+dK 融合部分和） |
| **448–511** | **dKV 槽 1**（round1 的融合部分和） |

**设计事实**：V2 相对休眠基类**故意对调**了 dQ/dKV 两区（基类为 dKV@128/192、
dQ@256/384），使 math 的 dQ epilogue（读 [128,384)）与 leader 尾段的 dKV 写入
（[384,512)）**列不相交**——v12 S1 优化（pds producer_tail 移到 epilogue 之后）
以此为前提。

### 4.4 管线与信用

**统一语义（重要）**：下表三条 PipelineAsyncUmma 管线（kscore/round/pds）的
`consumer_release` 与四条 PipelineUmmaAsync 管线（s_done/dp_done/dkv_done/dq_done）
的 `producer_commit` **全部编译为 `tcgen05.commit`（UMMA-tracked）**：调用指令的
位置只决定"何时挂上跟踪"，**信号实际到达在对应 MMA 真正完成读/写时**（并多播到
两个 CTA）。下表"释放点"均指调用位置。

| 管线 | 类型/深度 | 生产者 | 消费者 | 说明 |
|---|---|---|---|---|
| kscore | AsyncUmma / 1 | gather（256 线程组） | leader | **每 tile 两代**：K 代（S/dP 的 B；leader 在 dP 发射后 release#1）→ 借贷 dO_r0 代（grads 里第二次 wait/release#2）。**depth-1 相位配对**：loan 填充的 acquire ← release#1（即 dP 的 MMA 完成，WAR 保护）；**LOAD_K(下一 tile) 的 acquire ← release#2（即梯度块 round-0 两次 dV MMA 完成）** |
| round ring | AsyncUmma / 2 槽 | W17（elect lane；kdq 两代由 gather 代填、W17 提交） | leader（grads 的 A） | 每 tile 8 代；每代 release 紧跟对应 MMA 发射后（UMMA-tracked） |
| pds | AsyncUmma / 1 | **acquire=math，写=math，commit=relay lane0**（commit group=2：两 CTA 的 relay 各 arrive 一次到 leading CTA） | leader（G3/4/5 的 B） | leader 在 grads 块头 consumer_wait（该块第一道门）；grads 发射完后 release |
| s_done / dp_done | UmmaAsync / 2 | leader MMA | math | ping-pong tmem 信用；math T2R+fence 后 release |
| dkv_done | UmmaAsync / 2 | leader MMA | reducer | 槽 0 head 内 acquire/commit，槽 1 tail 内；reducer T2R+fence 后 release（原子加在 release 之后） |
| dq_done | UmmaAsync / 1 | leader | math（epilogue 门） | **每 token 一次**：acquire 在 tile 循环前，commit 提前到末 tile grads head 内（信号随最后一次 dQ MMA 完成） |
| loan | 2 裸 mbar | gather **warp0**：一次 16KB 本地 S2S（own-h 半，源=stationary_do）+ 一次 16KB GMEM TMA（peer-h 半），各挂一个 mbar | leader（round-0 两次 dV 的 A） | **信用本体 = kscore 的第二代**（释放即 leader 的第二次 kscore release）；尾 tile 有一个"空重提交"代（不拷贝，仅重新发布未变字节） |
| stationary | 裸 mbar ×2+2 | W17 TMA | 各角色 | **拆分就绪**：Q/dO 各自完成后分别 arrive；leader 首个 S 只等 Q-ready、首个 dP 只等 dO-ready |
| landing / relay | 裸 mbar 各 2 | 发送方先对**对端** landing mbar `arrive_and_expect_tx(4KB)` 再发 bulk | relay 等落地→arrive relay_mbars（两 CTA 各计 1，目标 CTA0）| leader 侧双门**分裂**：P-landing 门在 grads 头，dS-landing 门下移到 pass 3 前（v12 P2ii） |
| pds_ready | count-128 裸 mbar | math 每线程 STORE 后 arrive | relay lane0 | math→relay 交接边 |
| loan_epi_safe | 裸 mbar ×1 | gather warp0 在 kscore producer_tail 后（经 gather_barrier 汇聚）对**本 CTA** arrive | math | 与 dq_done 构成 epilogue 的**双门** |

named barriers（V2 活跃的全部 5 个）：kdq_barrier（id=4 区间，**160 线程** = gather
128 + W17 32，每 tile 双次会合）、math_barrier（id=3，128）、cta_barrier（id=2，
640）、**gather_barrier（id=5，128）**——loan 机制部件（填充的"发射可见/完成可见"
双会合 + epi-safe 发布前汇聚）、tmem_alloc_barrier（id=1，640）。

### 4.5 稳态每 tile 事件序（按角色；括号内为 trace span 名；行号已核）

**leader（w16）**，轮换调度——tile t 的 S/dP 之后发射 tile t−1 的梯度：
1. s_done/dp_done ping-pong producer_acquire → S(t) 发射（S_ISSUE，4 个 D128-chunk，
   B=score_kv）→ dP(t) 发射（dP_ISSUE，V≡K 复用 score_kv）→ **kscore release #1**。
2. 梯度块 grads(t−1)，其**第一道门 = pds consumer_wait**（无 span，即 §1.2 的
   0.41 段位置）。
3. WAIT_dQ（等 kdq ring 代）→ dQ r0 发射（A=kdq 镜像 g0；**B=ds_image 整体，
   纯本地**——`make_fragment_B(ds_image)`，不含任何 DSM 到货数据）→ ring
   release → dQ r1 发射（g1，B 同）。
4. **P-landing 等待** → **kscore consumer_wait #2**（loan dO_r0，gather 已把同一
   32KB stage 重新发布为两个 dO 面板 quadrant）→ dVdK head 4 pass（round0，
   dKV 槽 0：dV-h0[A=loan-a]、dV-h1[A=loan-b]，**dS-landing 等待** + ring wait →
   dK-h0[A=ring g2 Q_r0-h0]、dK-h1[g3]）→ 槽 0 commit → kscore release #2。
5. dVdK tail 4 pass（round1，dKV 槽 1：dV-h0[g4 dO_r1-h0]、dV-h1[g5]、
   dK-h0[g6 Q_r1-h0]、dK-h1[g7]，每 pass 前 ring wait）→ 槽 1 commit →
   pds release。
6. 末 tile：dq_done 早提交在其 grads head 内。TAIL span = **末 tile 的完整梯度
   发射 + pds release + 四条管线 producer_tail**（span 从末 tile 梯度发射前开始，
   不是"排空尾巴"）；tmem 释放与 cta/cluster 会合在 span 之外的公共尾。

**gather（w0–3）**，软件流水提前一拍，每稳态迭代产出**两个 kscore 代**：
1. loan 代：warp0 在 kscore 临界区内发 dO_r0 的两笔 16KB（own-h 半 = stationary_do
   本地 S2S bulk；peer-h 半 = GMEM TMA），warps1-3 经 gather_barrier 双会合跟随。
2. K(t+1) 代（LOAD_K）：**每 CTA 装 own-n32 行 × D512**（64 行是双 CTA 合计），
   每行 4×256B `cp.async`，无效行零填充 → 提交。
3. tile t 的 kdq 会合：与 W17 在 kdq_barrier（160 线程）会合 → 128 线程分 16 组
   ×8 线程（每组 4 行）gather kdq 镜像 = [N64×D128]×2 round（同一批 topk 行，
   列窗 256·r+128·rank，每行每 round 一个 256B 切片）直接写 ring 槽 → 排空 →
   再会合。尾 tile 特殊：loan"空重提交"代 + 收尾时 warp0 在 kscore producer_tail
   后 arrive loan_epi_safe。

**W17（load warp）**：
1. ROUTE_K span：kdq 会合区（双次 kdq_barrier 之间持有 2 个 ring 信用，等 gather
   填完后 elect_one 提交 g0/g1 两代）。
2. MAT_QDO ×2（**不对称**：m0 = 2 代 Q_r0；m1 = 4 代 dO_r1+Q_r1 及尾排空）：
   **每代 = 单笔 16KB 整槽填充**——h_half==本 rank 的代整体走 stationary 面板
   S2S bulk，另一半的代整体走 GMEM TMA（对单 CTA：每 round 一代 S2S + 一代 TMA）；
   h0 代固定进 round_buf_a、h1 代进 round_buf_b；TMA/S2S 完成都以
   `complete_tx` 打在 2 个 round_tma_mbars 上轮转，**滞后一代提交**（lag-1）。
   每代须先 acquire ring 信用（等 leader 消费掉两代前的那代）。
3. 每 token 序幕：stationary Q/dO 两笔 TMA + 拆分就绪 arrive。

**math（w4–7）**。坐标系（重要）：S/dP 的 C 沿 M（=H）跨 CTA 分半，**每 CTA 的
math 看到的是 [own-h64 × 全 n64]**（线程映射 local_h = tidx%64，n_owner =
tidx//64——128 线程 = 64 个 h × 2 个 n-半区警对）。事件序：WAIT_S → T2R_S
（+s_done release）→ WAIT_dP → T2R_dP（+dp_done release）→ SOFTMAX（32 路 exp2
全宽条带 + dS 链 + **bf16 转换**）→ **PDS_ACQ（pds producer_acquire，先于写入）**
→ STORE（纯 stmatrix 发布，**owner 分裂按 n-半区**：own-n 警对写 P/dS 终块
[own-n32×own-h64]，非 own 警对写 p_xchg 出站块 [peer-n32×own-h64]；全部 128 线程
另写整幅 ds_image [own-h64×n64]——own-n 警对的 dS 实际写两处）→ 每线程 arrive
count-128 pds_ready。

**relay（w18，仅 lane 0）**：等 pds_ready → **ROUTE_P 先**（对对端 landing mbar
arrive_and_expect_tx(4KB) → bulk 发 p_xchg → 对端 p_blocks 的本 rank 块）→
**ROUTE_dS 后**（发 ds_image 的 (1−rank) n-半份 → 对端 ds_blocks；rank0 取
+2048 元素处，rank1 取起始处）→ pds producer_commit → 等本侧两个 landing →
分别 arrive relay_mbars。

**reducer（w8–15）**：每 tile **一次融合调用**（分离的 wait/release 状态对）：
槽 0（=round0，D 四分区 {rank}）：WAIT_dK → REDUCE_T2R（Ld16x256b×Rep4）→
fence → release → REDUCE_ATOMIC（**在槽 1 的 wait 之前跑**，与 leader 的 grads
tail 重叠）；槽 1（=round1，四分区 {2+rank}）同构。**rank 分域是交错四分区**：
rank0 管 D[0,128)∪D[256,384)，rank1 管 D[128,256)∪D[384,512)。每线程 8 行
×v4.f32 原子加，行索引已预取；workspace 列序有 v4 组置乱（按 dp_idx//4 寻址）。

### 4.6 序幕与尾段

**序幕**（每 token 一次）：W17 TMA 装 stationary 面板 + 拆分就绪；LSE/Δ 装载；
K(0)、K(1) gather → S/dP(0)。实测 tile0 周期 3.9、tile1 7.1；冷启动时 peer K(0)
路由 5.4、首个 WAIT_dK 11.4。head ~3.3。

**尾段**：math 等 dq_done **与** loan_epi_safe 双门 → DQ_EPI ×2 round，每 round：
T2R dQ [D128×H128] f32 →（**标量店**：每 math 线程 128 次逐元素 bf16 写进
score_kv 上的 32KB staging，坐标变换 (d,h)→[head, local_d] 由逐元素索引吸收）→
fence → math_barrier → 单次 32KB bulk TMA 到 GMEM → 等引擎读完 → math_barrier。
之后 dq_done release。leader 侧 TAIL 见 §4.5-6。
**注意**：标量店是 vk_2 现役形态；stmatrix 向量店只存在于已归档为 null 的 vk_3
（§8）。此尾段的 trace 数字被证实严重高估 release 行为（§5.2、§8 vk_3 行）。

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

- leader 的 0.41 与 0.54 两段等待**无 span**。代码侧候选门已核清（§4.5：
  pds consumer_wait；P-landing、kscore#2、dS-landing），但**各门时长归属未测**。
- gather 的 kdq fill 段无独立 span（藏在会合窗口里）。
- vk_1_trace 已含供给等待细分名额（RK_ACQ / MAT_ACQ / MAT_WAIT），采集成功但
  逐 span 细读因后处理工具的名字映射缺失只完成部分。

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

### 7.1b R0 细分解析（同一 vk_1_trace 解码 JSON，稳态 tile 8–24，cta0；2026-08-06）

| 分量 | 值 | 含义 |
|---|---|---|
| MAT_ACQ 每 tile 合计（8 次） | **2.386**（单次 p50 0.288，>0.2µs 者 5.7 次/tile） | W17 等 ring 信用（leader 消费侧） |
| MAT_WAIT 每 tile 合计（6 次） | **0.990**（单次 p50 0.064，>0.2µs 者 1.6 次/tile） | W17 等 TMA/S2S 完成（lag-1 暴露） |
| ⇒ MAT_QDO 4.40 包络中的真实发射工作 | **≈1.0** | 4.40 − 2.39 − 0.99 |
| RK_ACQ #1（loan acquire，等 dP MMA 完成） | 0.094 | 几乎免费 |
| RK_ACQ #2（K acquire，等 dV-r0 MMA 完成） | 0.494 | §4.4 α 配对的直接测量 |
| dP.end → WAIT_dQ.start（pds 门段） | 0.414 | 与复盘 0.41 一致 |
| dQ r0→r1 间隙 | **0.632** | 供给依赖≈0（kdq 双代同时提交）——发射路径开销样本 |
| dQ r1→dVdK1（复合门段） | 0.552 | P-landing/dkv_done/kscore#2 未分解 |
| dVdK 相邻间隙（7 个） | [0.03, 0.34, 0.52, 0.46, 0.47, 0.40, 0.54] | **p1→p2 无管线操作 = 0.03**；有 ring wait 的 0.34–0.54 |
| MATH_PDS_ACQ | 1.178（p90 1.568） | softmax 完成后干等 pds 信用 |
| 末 dVdK → 下一 S | 0.502 | |

交叉核对：period 6.816 / ROUTE_K 2.254 / MAT_QDO 4.396，与 §7.1 外部复盘一致。

### 7.2 vc_2 时代对照（vc_2 trace，注入口径）

- dVdK 逐 pass 发射前间隙：[0.99, 0.36, 0.45, 0.35, 0.44, 0.35, 0.44, 0.51]。
- MATH_SOFTMAX 2.156（vg_2/vg_5 手术前）、MATH_PDS_ACQ 0.823（vc_2 形态）。

### 7.3 C vs S1 A/B（two_trace_tables 包络语义；C=无 split publish；完整 18 行）

| 项 | C | S1(=vg_5) | 单位 |
|---|---|---|---|
| K/KV load | 2.778 | 2.635 | per H128 launch |
| Q/dO startup（LOAD_QDO） | 0.448 | 0.608 | per H128 launch |
| LSE/Σ(O·dO) startup（LOAD_STATS） | 0.512 | 0.544 | per H128 launch |
| MATH_BAR1 | 0.600 | 1.045 | 每 tile |
| MATH_PDS_ACQ | 1.739 | 2.008 | 每 tile |
| MATH_SOFTMAX | 3.333 | 3.519 | 每 tile |
| MATH_STORE | 1.213 | 1.272 | 每 tile |
| MAT_QDO | 5.283 | 5.056 | 每 tile |
| P+dS T2R/math | 8.054 | 8.946 | 每 tile |
| ROUTE_K | 2.010 | 2.224 | 每 tile |
| ROUTE_P / ROUTE_dS | 0.304 / 0.355 | 0.611 / 0.526 | 每 tile |
| Routes + MAT_QDO 小计 | 7.952 | 8.417 | 每 tile |
| S+dP 发射 | 1.026 | 1.028 | 每 tile |
| dQ+dVdK 发射（leader 纯忙时） | 0.791 | 0.875 | 每 tile |
| dKV T2R+atomic | 5.867 | 5.631 | 每 tile |
| dQ epilogue | 23.712 | 23.424 | **per launch** |
| TAIL | 28.0 | 27.328 | **per launch** |

（注意：此表与 §7.1 同名项数字不同——聚合语义不同（包络含嵌套等待、按因子加权），
两表各自内部可比，互相不可直接比。）

### 7.4 每 token 固定成本（trace 口径，vk_1）

head ~3.3；慢启动净贡献 ≈0（tile0 周期 3.9 快于稳态、tile1 7.1）；尾窗 ~28
（DQ_EPI 2×11.9 与 TAIL 27.7 相互重叠）⇒ **F_trace ≈ 31**（≈ head + 尾窗）。
**修正**：基于 §8 三次尾段手术全零，release 侧真实 ΔF（对 baseline）≈
0.10–0.15 ms e2e（v1 按平均税折算的 0.35–0.43 ms 作废）。

### 7.5 供给带宽事实

- 每 tile 经 ring 流转的梯度 A 操作数：8 代 × 16KB = 128KB；另有 dO_r0 2×16KB
  经 loan。合计 160KB/tile 的 SMEM 供给流量，通过 **2 个 16KB ring 槽 + loan 区**。
- **面板代（6 代）**：每代单笔 16KB——own-h 类走面板 S2S bulk，peer-h 类走 GMEM
  TMA（每 CTA 每 round 各一代）；lag-1 提交使两类传输并飞。
- **kdq 代（2 代）**：GMEM 稀疏 gather（64 行 × 256B × 2 round，gather 警组代填）。
- **loan（dO_r0）**：gather warp0 一次调用同发 16KB S2S（own-h）+ 16KB TMA
  （peer-h），与面板代同一"own=S2S / peer=TMA"规则，但两半同时填。
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
| vc_2 | score-K/dO 时间借用（dO_r0 借宿 score 区，骑 kscore 第二代） | 10.572 @ 8.268 (1.2786) | 采纳为基座 |
| vc_3 | score-K 二次借用（loanQ） | 11.836 @ 8.430 (1.4040) | 负。r1 曾**软死锁**：尾 tile 三方环（gather 卡 loanQ acquire ← grads dV 释放 ← dQ ← 末次 kdq 会合被排在 loans 之后）；r2 重排后正确但负 |
| vd_1 | 供给环拆双生产者（W17+W19 双 lane） | 10.833 @ 8.310 (1.3036) | null 偏负。**排除了两个假说成分**：生产者串行排队 ≈0、跨奇偶 head-of-line ≈0（2 槽环按 h 奇偶本就是两条 depth-1 链，lag-1 提交已让两类传输并飞） |
| ve_1 | 双 tile 宏批处理 + P/dS 第二发布面按别名寄宿 score-K | 12.917 @ 8.401 (1.5376) | 证伪。**机制侧全兑现**（trace 实测：dVdK 节拍 0.42→0.20/pass；MATH_PDS_ACQ 0.823→0.056；math 墙 4.58→2.80；drain 2.91→2.54），**但**：K(下一对) gather 须等别名读完 → +3.07/pair 串行尾（loan 骑 kscore 的直接后果）；rotated schedule 被迫放弃 → +2.4/pair。净 +5.5/pair 调度税 |
| vg_1 | split publish（dS 先发/P 后发，双 pds 管线） | 10.503 @ 8.336 (1.2599) | 曾采纳；后经 C/S1 A/B 判净贡献 ≈0（判据：±0.1ms 界内且当日漂移 2.6% 远超其原始收益 −0.069ms）且携带注入卡死触发结构，整体退役 |
| vg_2 | exp 条带流水（8 EX2 在飞） | 10.450 @ 8.394 (1.2450) | 采纳（bit-exact 重排） |
| vg_3 | T2R 双发单 fence | 10.454 @ 8.313 | null（不在关键路径） |
| vg_4 | topk 索引批量预取（两处稀疏 gather 的行索引载入提升到行循环前） | **9.936** @ 8.376 (1.1862) | 采纳，单刀 −0.51ms |
| vg_5 | exp 全宽条带（32 EX2 在飞，结果写回原寄存器） | **9.771** @ 8.390 (1.1646) | 采纳（bit-exact） |
| vg_6 r1 | dQ epilogue 双缓冲（staging→ring 槽） | correctness FAIL 5.4% | dq_done 是早提交（只跟踪 dQ MMA），epi 开始时 ring 仍被尾部 dVdK 读（WAR）。**可复用结论：epi 时刻真正死透的缓冲只有 stationary panels** |
| vg_6 r2 | 双缓冲改借 stationary dO 面板 | 9.781 @ 8.345 (1.1720) | null：**TMA 飞行/读等待不是尾段大头** |
| vh_1 | score_kv 32KB 常驻 K → 2×8KB D128 段环，省出 16KB 给 ring 深 3 | **13.631** @ 8.552 (1.5938) | 证伪。实测 +2.2/tile：S/dP 逐段等 K（gather 是最慢供给腿 ~1.5/tile 纯工作、稀疏散射）+ loan 退役回吐 ~0.4 + pad 代协议开销 |
| （注入案） | 直接实测卡死 = vg_1/vg_2/vg_4（IKET 注入下 GPU P0/100%/2.0GiB 自旋 80-100s，已进 kernel）；vg_5 未单独 bisect，由变体外推。排他变体：**C**（去 split publish）过 9.896 @ 8.607 (1.1497)、**S1**（拆共享 state）过 9.854 @ 8.508 (1.1582)、S2（并 ready mbar、留共享 state）**挂**、a_advance（留共享 state、挪 advance）**挂**、s1a 过 | — | 触发要件 = relay 两管线共享一个 pipeline state 的代码形态；CUTLASS 4.5.0 语义层无病（容器内核对：depth-1 的 producer_tail 不改 state）。**方法论记录**：此案曾被误诊为 infra 故障（报错串在成功 run 同样出现=无害噪声）→ 规则：报错必须先在成功 run 里做对照 |
| vk_1 | vg_5 + S1 修复（拆 state） | 9.844 @ 8.605 (1.1440) | 可采 trace 的 vg_5 等价物。vg_5_trace（细化 span 版）继承共享 state 不可用，已 rebase 为 vk_1_trace |
| vk_2 | **现役**：C 形态转正（split publish 整体退役） | 9.902 @ 8.609 (**1.1502**) | 与 vk_1 漂移内同水位；协议面更简 |
| vk_3 | dQ epilogue 标量店→stmatrix（T2R atom 同换 Ld16x256b） | 9.857 @ 8.519 (1.1570) | **硬 null**（ms −0.045 但 ratio 反劣于 vk_2；两指标矛盾）。r1 曾命中预登记 build-gate（partition_D rank 断言），r2 布局代数正确、correctness 4/4。**验尸**：trace 的 DQ_EPI 11.9/round 系注入税放大（标量循环指令密度远超均值）；release 真实 ~3-4/round，store 份额 ~0.4 |
| M3（桌面） | dS 单份双主序（省一次 dS 存储 + 8KB） | 未上机 | 被 CG2 描述符 rank 对称锁挡死：可别名块是 ds_blocks[rank]（rank 相关偏移）。字节同构性本身成立（ds_image 两半与 dkv-B 块字节同构，中继直接 DSM ds_image+2048 为证） |
| v9（旧案，VG 台账外的更早记录） | K_dQ 复用 score-K（credit-gated peer push 机器） | +6.4ms | 死于该推送机器的经济性（供参考：与"复用 score_kv 字节"不是同一机器） |

**汇总模式（事实）**：采纳的 vg_2/4/5 全部是零字节的发射序/延迟手术；4 项证伪/负
（vc_3、vd_1、ve_1、vh_1）全部涉及 SMEM 重新布管。

**vre_3 r4 判决落章（2026-08-06 补）**：环深 2→4 + stationary_do 退役 + dO 流式
（v12 基）= **12.675 @ 8.311 (1.5252)，证伪**。correctness 4/4（r4 已治好 r1 的
数值污染与 r3 的死锁），纯性能负：比同日 v12（11.955）慢 6.0%。**含义：ring 加深
的第二个（也是最后一个）已知资金源在 SM100 上不成立**；SMEM 重布管类战绩变为
5 连败（vc_3/vd_1/ve_1/vh_1/vre_3）。

K2（kdq 退役，用 score_kv 字节 + DSM 替代第二次稀疏 gather）桌面工作已开（其
几何事实已并入 §4.5/§7.5），未上机。M2（FFMA deg6 exp 替换 MUFU；SOFTMAX
vc_2 时代 2.156→台账预期 ≤1.1，vk_1_trace 新测 2.22）数值半门已过（deg6 与
MUFU 同误差量级 1.9–3.9e-6），等"近似替换"档终裁，未上机。D1（warp 重划
640→512）未上机。

### 8.1 vk_2 K2 定价 trace（2026-08-06，注入口径，稳态窗口 issue_seq 8..23）

诊断 run（无性能判决；baseline 8.481 / candidate 9.858 仅留档）。本次 31 名额
置换：退役近零的 WAIT_S/WAIT_dP，换入 **PDS_WAIT(i)**、**GRAD_SUP_WAIT(i)**——
§1.2 两段无标记等待自此有直接测量（vk_2 基座上）。

- **period 7.089/tile**（窗口内 15 个间距，range 6.688–7.520）。与 vk_1_trace 的
  6.80 **不可直接比**（插桩集合不同，§5 跨版本禁比规则适用）。
- **W17 串链复核（vk_2 上仍饱和）**：ROUTE_K 2.041 + MAT_QDO×2 4.878 = 6.919 =
  period 的 **97.6%**（slack 0.170）。
- **leader 两段等待的直接测量**：PDS_WAIT 0.506/tile；GRAD_SUP_WAIT 0.260/tile。
- **供给行分解（H128 logical 口径）**：**MAT_ACQ 3.708**/tile（8 次环信用 acquire）
  ｜MAT_WAIT 1.368/tile（6 次 TMA 完成等待）｜RK_ACQ 0.858/tile。
  **信用等待主导供给行**。其中 ROUTE_K 内 kdq 两代的 MAT_ACQ 合计 0.604
  （占 ROUTE_K 包络 27.4%）。
- **dVdK 逐 pass gap_before（vk_2）**：[0.326, 0.028, 0.436, 0.544, 0.464, 0.492,
  0.414, 0.494]，Σ3.198（vc_2 时代 Σ3.890）。
- **math 行（单 warp 均值，M2 定价口径）**：SOFTMAX 2.047｜PDS_ACQ 1.264｜
  STORE 1.005｜T2R_S 1.030｜T2R_dP 0.493。
- **WAIT_dQ 0.142/tile（两次合计）；时刻对齐**：round0 的 WAIT_dQ 结束贴合
  PDS_WAIT/串行前驱（+0.028/+0.112），距 kdq commit（ROUTE_K.end 代理）**晚
  +0.554**——**kdq 在 vk_2 上先于需求就绪，不再是 dQ 发射的绑定门**。
  （§9-5"判决 2"是 vk_1（含 split publish）上的测量，在 vk_2 上不成立；
  绑定门随基座变化。）

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
10. （§1.2 未标记等待的归属）0.41 段 = pds consumer_wait、0.54 段 = P-landing +
    kscore#2 复合——这是把 §4.5 的代码门序与 §7.1 的 trace 位置对齐后的推断，
    时长归属未直接测量。

---

## 10. 术语表

- **tile**：一个 N64 KV 块（64 个 topk 行）；topk=2048 时每 token 32 个，从高索引
  往低处理。
- **round**：D512 特征维的一半（D256 cluster / D128 每 CTA）；dQ、dKV、kdq 均按
  2 round 处理。
- **pass**：dVdK 的 H64 半份归约（每 round 4 pass：dV×2 + dK×2，融合累加进同一
  tmem 槽）。
- **代（gen）**：ring 槽的一次填充-消费周期；稳态每 tile 8 代（g0/g1=kdq、
  g2/g3=Q_r0、g4/g5=dO_r1、g6/g7=Q_r1）。g0/g1 由 gather 警组 128 线程代填、
  W17 只持有并提交信用；其余 6 代由 W17 亲发。h0 类代固定进 round_buf_a、
  h1 类进 round_buf_b。
- **quadrant**：梯度 A 操作数的 [D128×H64] 切片（16KB bf16）。**每个 quadrant 代
  是单源的**：own-h 的代整体来自 stationary 面板 S2S 拷贝，peer-h 的代整体来自
  GMEM TMA。
- **panel（面板）**：token 常驻的 Qᵀ/dOᵀ [D512×H64 own-h-half]，64KB×2。
- **kdq**：dQ GEMM 的 A 操作数镜像 [N64×D128]×2 round——与 score_kv 同一批 KV 行
  的不同列窗（d 连续布局），现由 gather 警组从 GMEM 第二次稀疏 gather 得到。
- **loan（时间借用）**：dO_r0 的 2 个 quadrant 暂住 score_kv 的机制（vc_2 引入）；
  其信用本体是 kscore 管线的第二代（K 代与 loan 代在同一条 depth-1 管线上交替）。
- **plane（转置平面）**：全部 GEMM 按转置形式组织（Sᵀ/dPᵀ/dVᵀ/dKᵀ/dQᵀ）。
- **rank / own-half / peer-half**：cluster 内 CTA 编号（0/1）。CG2 分半规则：
  C 沿 M 维分半；**B 沿该 GEMM 自己的 N 维分半**——对 S/dP/dVdK 是 KV 行 n，
  对 dQ 是头 h；面板按 h 分半；reducer 落盘按交错 D 四分区 {rank, 2+rank}。
- **rotated schedule（轮换调度）**：tile t 的 S/dP 先发，随后发 tile t−1 的梯度。
- **drain（排空）**：reducer 把 dKV（=dV+dK 融合）部分和从 tmem 取出并原子加进
  GMEM workspace。
- **baseline**：同仓库的 1-CTA 参照实现（CG1，无跨 CTA 交换；内部分解见 §7.6）。
- **IKET / 注入 / span**：运行时给 kernel 打 trampoline 采集软件注解时间线的
  工具；span 是命名时间段，名额上限 31。
- **波（wave）**：负载/硬件事实（非源码常量）：token 数 / 74 cluster 槽位 = 55.35。
- **UMMA-tracked**：管线信用信号编译为 `tcgen05.commit`——调用位置只决定挂上
  跟踪的时机，信号在对应 MMA 真正完成时到达并多播两 CTA。
- **kdq_barrier / gather_barrier / math_barrier / cta_barrier**：见 §4.4。
- 备注：源码中 `H_PASSES`、`ROUND_GENS_PER_TILE` 等常量仅为注释性锚点（定义后
  零引用），2-pass/8-gen 结构由手写展开保证。

## 11. 开放事实清单（截至 2026-08-06）

1. ~~leader 两段无 span 等待的时长归属~~ **已闭合**（§8.1：PDS_WAIT 0.506、
   GRAD_SUP_WAIT 0.260，vk_2 直接测量）。
2. ~~细分读数未解析~~ **已闭合**（§8.1 供给行分解，vk_2 基座）。
3. K2 的前置桌面项：K2b（swizzle 零拷贝别名）的 CG2 描述符合法性推导未闭合；
   **K2a（拷贝方案）新增一项待核**：score_b_layout（swizzled B）与 dq_a_layout
   （swizzled A）之间 256B 行片的连续可拷性——最坏退化为每 tile 每 CTA
   128 个 256B 分片拷贝（warp0 的 32 线程分担，发射成本仍远低于 GMEM gather）。
4. M2 等用户对"近似替换"档的终裁。
5. ROUND_STAGES=2→3/4 在 SM100 上**已无任何已验证资金源**（vh_1、vre_3 r4 双双
   证伪）——若 MAT_ACQ 主导的信用饥饿（§8.1 已实测）在 K2 之后仍是绑定项，
   该平台即 SM100 结构地板，转 Rubin 项。
6. W19（32 线程）与 math 警组、reducer（duty 37%）的富余未被利用。
7. IKET 局部税率未标定（只有全 kernel 平均 38% 与"指令密集区更高"的间接证据）。
8. SMEM 真实 slack 仅 1,024 B——任何新增 SMEM 字段的方案必须先解决资金来源。
9. score_kv 三租户轮转（K(t) → loan 的 dO_r0 → K(t+1)，§4.4 kscore 行的相位配对）
   已由独立读码二次确认（2026-08-06）：K2a 的拷贝必须插在
   [dQ(t−1) 环信用释放, loan(t−1) 填充] 窗口内，把拷贝排在 warp0 的
   loan 填充之前即天然满足（该窗口与现行 kdq 会合窗不同——现行 kdq 从 GMEM
   取数，不依赖 score_kv 存活；K2a 拷贝依赖）。
