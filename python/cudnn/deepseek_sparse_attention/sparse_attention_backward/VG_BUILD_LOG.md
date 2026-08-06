# VG 系列构建日志（2026-08-05/06）

底盘 vc_2（10.572 @ 8.268，ratio 1.2786）。纪律：单杠杆单 rev、预登记止损门、
跨版本比 **candidate_ms 与同日 ratio 并列**（同日 baseline 实测漂移 8.268-8.400，
±1%，单看 ratio 会误判）。

## 结果台账

| 版本 | 杠杆 | candidate | baseline | ratio | 判决 |
|---|---|---|---|---|---|
| vc_2 | （基座）score-K/dO 时间借用 | 10.572 | 8.268 | 1.2786 | 现役 |
| vd_1 | 供给环拆双 lane（W17/W19） | 10.833 | 8.310 | 1.3036 | **null 偏负，归档** |
| ve_1 | 双 tile 宏批处理 + 第二发布面别名 | 12.917 | 8.401 | 1.5376 | **证伪，归档**（见下） |
| vc_3 | score-K 二次借用（loanQ） | 11.836 | 8.430 | 1.4040 | **负，归档** |
| **vg_1** | split publish（dS-early/P-late） | 10.503 | 8.336 | 1.2599 | **采纳** |
| **vg_2** | M1 exp 条带流水（8 EX2 在飞） | **10.450** | 8.394 | **1.2450** | **采纳，当前最优** |
| vg_3 | T2R 双发单 fence | 10.454 | 8.313 | 1.2575 | **null**（candidate 与 vg_2 同，ratio 差异=基线漂移） |
| **vg_4** | topk 索引批量预取（两个 sparse gather） | **9.936** | 8.376 | **1.1862** | **采纳，本轮最大单刀 −0.51ms** |
| **vg_5** | exp 全宽条带（32 EX2 在飞，结果写回 r_score） | **9.771** | 8.390 | **1.1646** | **采纳，当前最优** |
| vg_6 r1 | epi 双缓冲（round1 → ring） | — | — | — | **correctness FAIL 5.4%**：dq_done 是早提交（只跟踪 dQ MMA），ring 在 epi 开始时仍被尾部 dVdK 读 → WAR |
| vg_6 r2 | epi 双缓冲（round1 → stationary dO panel） | 9.781 | 8.345 | 1.1720 | **null**（修正后正确，但 epi 大头是标量 scatter 不是 TMA 飞行）→ 归档，基座保持 vg_5 |
| vh_1 | K 段环捐赠（32→2×8KB）→ round 环深 3 | 13.631 | 8.552 | 1.5938 | **证伪，归档**（correctness 4/4 PASS，纯性能崩塌；验尸见结构性发现 #4） |
| **vk_1** | vg_5 + S1 双 producer state（修注入卡死） | 9.844 | 8.605 | 1.1440 | **采纳**（语义=vg_5，解锁 trace 通道） |
| **vk_2** | split publish 退役（排他实验 C 转正） | 9.902 | — | — | **现役基座**（与 vg_5/vk_1 漂移内持平，协议面减半） |
| vk_3 r2 | dQ epilogue stmatrix 向量化（P1/K1） | 9.857 | 8.519 | 1.1570 | **null**（仅 −0.045 vs vk_2，< 0.1 门；correctness 4/4、布局代数正确；SASS 验尸中——头号嫌疑 math 警组寄存器溢出 f32×128+bf16×64） |

全部版本 correctness 4/4 PASS。累计 10.572 → **9.771（−7.6%）**，ratio 1.2786 → **1.1646**。

## 本轮方法论收获：延迟暴露是当前最肥的矿

vg_4（−0.51ms）与 vg_2/vg_5（合计 −0.22ms）打的是同一件事：**长延迟操作与其消费者
在程序序上贴身**，导致每次都吃满延迟。三处已修：

1. sparse gather 的 topk 索引（GMEM，每行一次，逐行串行）→ 批量预取；
2. softmax 的 MUFU.EX2（结果下一句就被 dS 用掉）→ 全宽条带 + 结果写回 r_score
   （零寄存器增长）。

排除项：T2R 双发（vg_3 null——不在关键路径）；reducer 索引（v5 已预取，无重复）。

**下次找矿的判据**：任何"load/长延迟指令 → 下一句消费"的结构，先看它在不在
每 tile 重复的路径上。

### 已定位但未动的下一个大靶子：dQ epilogue

ve_1 trace 实测 `DQ_EPI(r)` **11.32 µs**（n=16）。按当前刻度换算：每 token 驻留
≈ 176 µs（9.771ms ÷ 55.35 波），32 tile × ~5.5 µs——**epilogue 独占 ~6.4%**。
它搬的字节只有 128 KB/CTA（带宽视角 ~0.1 µs），说明 11 µs 几乎全是
T2R + SMEM staging + TMA 的串行结构成本，与 vg_4/vg_5 修掉的是同一类病。
路线图记的 F（每 launch 固定成本）≈26-28 µs vs baseline ≈8.2 µs，
折合 ~1.0 ms 差距——**约等于当前剩余缺口 1.37 ms 的大头**。
优先级建议提到 D1 之前：先拍一张 epilogue 的分段 trace（或直接读
`_store_dq_epi_tma_v12` 找串行点），代价远低于 warp 重划。

**vg_6 已裁决其中两个成分**：TMA 飞行/读等待（双缓冲重叠 = null，非大头）；
剩余唯一嫌疑 = **标量 scatter（128 STS.U16/线程/round）+ T2R**。对应手术 =
S2 预登记的"16-DP T2R + stmatrix 向量 store"（v9.3 在 P/dS publish 上的同配方），
难点是目的地是 [H,D] 而累加器坐标是 (d,h)——需要 transpose 布局代数 +
get_smem_store_op 在组合布局上的推导，属中风险手术，留给下一轮。
**vg_6 顺带留下一条可复用结论：epi 时刻真正死透的缓冲只有 stationary panels**
（dq_done 早提交使 ring 和 P/dS face 都可能仍被尾部 dVdK 读）。

## 结构性发现（本轮最重要的产出）

### 1. ve_1 验尸：别名换零字节 = 相位分离，代价 > 量词减半的收益

ve_1 的**机制全部兑现**（trace 实测）：dVdK 节拍 0.42→**0.20 µs/pass**（12 gen/pair
生效）；`MATH_PDS_ACQ` 0.823→**0.056**（C4 锚点边解锁生效，= VF1 doc R0 信号 #1 PASS）；
math 墙 4.58→2.80、drain 2.91→2.54。

**但调度回归吃掉了全部收益**，根因单一——P/dS 像#2 借宿 score-K：

1. K(下一对) 的 gather 必须等 grads 读完别名 → **+3.07 µs/pair 串行尾**
   （= VF1 doc R0 预登记信号 #2 FIRED）；
2. 更致命：**v7 rotated schedule 被迫放弃**（score(j) 的 K 与像#2 抢同一块 score_kv），
   leader 空等 tile1 的 math → **+2.4 µs/pair**。

合计 5.5 µs/pair 调度税 vs 1.7 µs/tile 机制收益。

**结论入账**：双 tile 批处理机制本身是对的，**解锁前提是 24 KB 真 SMEM**
（PDS1 最小 = p_blocks 8 + ds_image 8 + ds_blocks 8）。唯一可能的来源是
panel 流式化（vre_3 方向）。在拿到这 24 KB 之前，ve_1 不应重跑。

### 2. 供给侧协议手术全族测尽

时间借用（vc_2 ✓ 已用满）｜二次借用（vc_3 ✗）｜gen 重排（cluster 不变量 ✗）｜
双生产者（vd_1 ✗）｜fill 提速（165 ns 纯工作，无肉）。

vd_1 的 null 排除了两个成分：**生产者串行排队 ≈ 0、跨奇偶 head-of-line ≈ 0**
（旧 2-stage 环按奇偶本就是两条 depth-1 链，W17 的 lag-commit 早已让两个奇偶
的 TMA 并飞——vd_1 只是拓扑等价重建）。剩余 0.4 µs/gen 平台的唯一解释是
**每奇偶 depth-1 的信用-飞行延迟暴露**，买家只剩：加槽（+32 KB）或删操作数。

### 3. M3（dS 单份双主序）的前提成立但被 CG2 描述符锁挡住

代码注释自证 `ds_image` 两半与 dkv B blocks **字节同构**（relay 直接 DSM 送
`ds_image+2048` 即为证），所以 math 确实把 dS 存了两遍。但可别名的那一块是
`ds_blocks[rank]`——**rank 相关偏移**，与 CG2 描述符两 CTA 同地址规则冲突。
要拿这 0.35-0.45 µs 必须先做 exact-P/dS 重布局（[h128 × own-n32] 形态 +
rank 置换的 n 序），属重构级，不是 rider。

### 4. vh_1 验尸：常驻已预取操作数的 SMEM = 物化的前瞻时间

- 机制：score_kv 32KB 常驻 K 的"大小"就是 K 的前瞻量——整块常驻使 gather 领先
  leader 一整个 tile；2×8KB 段环把前瞻压到半 tile。K gather 是全 kernel 最慢的
  供给腿（~1.5µs 纯工作 + 稀疏散射），S/dP 从"从不等 K"变"每段等 K"。叠加
  vc_2 loan 退役回吐（~0.4µs/tile）与 pad 代协议开销，+2.2µs/tile 与实测吻合。
- **铁律入账**：把"常驻+已预取"改环 = 按 1:1 拿前瞻时间换字节；只有当该操作数的
  生产者远快于消费节拍时才是净收益。K gather 恰是最慢生产者 → 负 EV。
- 环深 3 未被证伪；被证伪的是这笔 16KB 资金。剩余买家：panel 流式化
  （query-stationary，付 L2 带宽不付前瞻）或 Rubin 容量。
- 战绩模式定型：本轮 4 胜全是零字节延迟手术（vg_1/2/4/5），4 负全是 SMEM 重布管
  （vc_3/vd_1/ve_1/vh_1）。227KB 每个字节都在承重。

### 5. trace 停摆重定性：不是 infra，是 split publish 的注入期卡死（2026-08-06）

- **误诊更正**：08-05 21:48 起的 exit 124 曾归档为 infra（VsmTopologyMapper 报错）；
  直登复核证明该报错在**成功的** baseline/vc_2 采集里同样出现 = 无害噪声，且停摆
  起点与首次采集 vg_1+ 内核重合。**规则：报错串必须先在成功 run 里做对照，才允许
  用于归因。**
- **bisect @9c11a29**（隔离 clone，SHA 全程核对）：vg_1/vg_2/vg_4 全部在 candidate
  capture 卡死（GPU P0/100%/2.0GiB 自旋 80-100s，已进 kernel 非未启动）→ 首触发 =
  vg_1 split publish；vg_2/4 仅是继承。cuda-gdb attach 引发次生错误，不可用。
- **排他变体判决**（同一 clone，--candidate-file 注入，互斥不叠加）：
  - **C**（vg_5 反向移除 vc_2→vg_1 全部 23 hunk）：全绿含 candidate trace；
    9.896 @ 8.607，ratio 1.1497。
  - **S1**（vg_5 仅拆 relay 共享 pds_com 为双 producer state）：全绿含 candidate
    trace；9.854 @ 8.508，ratio 1.1582。
  - S2（单 ready mbar、保留共享 state）：跑动中，作机制分类（挂 ⇒ 指认共享 state
    的代码形态；过 ⇒ 注入层对 patch-site 形态敏感）。
  - 静态嫌疑核查记录：pds_mbars 已扩 [4]（无越界）；dS/P 两个 arrive 各有 fence；
    容器内核对 CUTLASS 4.5.0 的 producer_tail 在 num_stages=1 **不** mutate state
    （一度按新版语义误判，已撤回）。**API 语义层无病，病在共享 state 的
    lowering/代码形态与注入的交互**——纸面分析原则上摸不到这一层。
- **性能判读**：当日 baseline 漂移 8.39→8.61（2.6%，远超台账登记的 ±1%）；
  C/S1/vg_5 的 candidate_ms 与 ratio 两个指标方向互相矛盾 → **split publish 在
  vg_5 合成里的净贡献 ≈ 0（±0.1ms），与漂移不可分**。vg_1 单刀原始收益
  （−0.069ms）本就在漂移尺度内。
- **裁决路径（零 GPU）**：C 与 S1 的解码 trace 均已产出。S1 与 vg_5 语义全同 ⇒
  **S1 的 trace 就是 vg_5 的 gap 台账**；C 是 split publish 的严格 A/B。读两份
  trace：若 S1 侧 dQ 头部空洞/`MATH_PDS_ACQ` 显示 split publish 仍买到真实重叠
  → 采 S1（最小修复）；否则 → 采 C（协议更简：少一条管线、一个 mbar、一次交接）。
- 附带：vg_5_trace（3bcdde3，MAT_ACQ/MAT_WAIT/RK_ACQ 细化名额）继承共享 pds_com，
  采 C/S1 之前不可用，采定后需 rebase。

## 下一步（按 可兑现额 / 阻塞项）

| 优先 | 杠杆 | 预期 | 阻塞 |
|---|---|---|---|
| 0 | **读 C/S1 解码 trace**（gap 台账 + split publish A/B → 定 C vs S1 新基座；重排矿点 1/2） | 定当前 pacer | 无（零 GPU，文件已在 scratch） |
| 1 | **D1 warp 重划 640→512** | −0.5-0.8 µs/tile | 需重新安置 4 个 warp 的角色（gather 128 线程是 ROUTE_K 主力，不能简单砍）；D4 microbench 先定上限 |
| 2 | **M2 FFMA deg6 exp** | SOFTMAX 2.16→≤1.1 | **等用户对"近似替换"档终裁**（数值半门已 PASS，deg6 与 MUFU 同误差量级） |
| 3 | M1 条带加宽 8→16/32 | 小 | 寄存器溢出风险（math@128） |
| 4 | exact-P/dS 重布局 | 解锁 M3 + 省 8 KB | 重构级 |
| 5 | ve_1 复活 | 大 | 需 24 KB（panel 流式化） |

**E1 已落章（2026-08-06，vk_1_trace 同 run 双采）**：baseline period 4.648，
reducer busy 3.98（duty 85.7%）——**drain-bound 实锤**，自身下探空间 ≤0.6µs，
被钉在 ~8.4-8.6；vk_1 的 reducer busy 仅 2.50/tile（rank 分域体积减半）——
**CG2 的 1.5µs/tile 结构性 drain 优势首次双边实测确认，超越的物理来源坐实**。
F 账：vk_1 F≈31µs/token vs baseline 6.8，ΔF≈0.35-0.43ms（缺口 ~30%）。

**pacer 判决（vk_1 trace，推翻 vre_1 时代的"环松弛"框架）**：W17 供给链自身
饱和——ROUTE_K 2.27（真身 = kdq 会合区）+ MAT_QDO×2 4.38 = 6.65 ≈ period 6.80，
零 slack。这重释了两桩旧案：vd_1 null（拓扑等价、槽没变多）与 vh_1 里环深 3
即使资金无害也兑现不了（供给链饱和时加槽无用）。手术梯子 K1-K5 见
外部复盘（~/proxy/dsa-vk1-traceable-r1）；K3（环深）必须排在 K2（kdq 退役）
之后。CG1 混流 ISA 裁定仍未落章。

## 工装备忘

- trace 采集在 21:48 后连续超时（exit 124，capture_2cta 阶段）——**已重定性：非
  infra，系 vg_1 split publish 注入期卡死**（见结构性发现 #5）；VsmTopologyMapper
  报错是无害噪声（成功 run 同样打印）。baseline/vc_2 通道一直正常。
- `.git/index.lock` 出现过陈旧残留（0 字节、无 git 进程），清理后正常。
