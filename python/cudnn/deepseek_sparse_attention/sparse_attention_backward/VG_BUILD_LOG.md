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
| **vk_2** | split publish 退役（排他实验 C 转正） | 9.902 | 8.609 | 1.1502 | **现役基座**（与 vg_5/vk_1 漂移内持平，协议面减半；记账 run = vk_3 委托单里的对照腿） |
| vk_3 r2 | dQ epilogue stmatrix 向量化（P1/K1） | 9.857 | 8.519 | 1.1570 | **硬 null，归档**（ms −0.045 但同日 ratio 反而劣于 vk_2 的 1.1502——两指标方向矛盾 = 漂移内为零；correctness 4/4、布局代数正确；验尸见下） |
| v_f1_1 | relay commit-first（R2：pds 提交前移到 pds_ready 一清） | 9.869 | 8.449 | 1.1681 | **null 不晋升**（双指标矛盾：ms −0.033 但 ratio 劣于 vk_2 的 1.1502，当日 baseline 8.449 为历史最低）。**成对判读**：机制兑现 PDS_WAIT 0.506→0.140；WAIT_dQ 持平 0.146（kdq 未吸收）；**GRAD_SUP_WAIT 0.260→0.438 吸收 +0.178**；period 7.089→7.100 持平。**结构结论：环绕过单段移除——绑定锚在 landing/loan 复合门与 dVdK 信用火车，不在 pds 门** |

**v_f1_1_trace2 复合门四分解（2026-08-06，诊断 rev 8e892ef）**：四分量全小——
LAND_P 0.108 / DKV_ACQ 0.054 / KS2 0.098 / LAND_DS 0.052（cta0 稳态均值）。
**等待池守恒发现**：同一 v_f1_1 代码两次 trace，PDS_WAIT+GRAD_SUP 之和守恒
（r1: 0.142+0.442=0.58；trace2: 0.390+0.260=0.65）但分布在门间漂移——
**梯度块头的 ~0.6µs 等待池由 4-5 个近同时到位的并行锚共同托底，压任何单门
只是重新分配**。预登记判读命中"四分量全小"分支 ⇒ **下一杠杆 = 火车侧**
（dQr1→dVdK1 0.408 + dVdK gaps Σ2.91 是最大块）。
顺带 release 数据点：trace2 帐面 9.856 @ 8.568 = ratio 1.150236，与 vk_2
的 1.1502 精确重合——R2 性能中性三度确认。
火车侧两把刀的证据基础：R4a（dQ 双发合并：0.632 间隙 vs 0.03 无管线操作
对照样本，kdq 双代同时提交 ⇒ 第二次 wait 免费）；R3-proper（面板代完成 tx
直接武装 ring full mbar，消灭 W17 的 lag-1 提交延迟——MAT_WAIT ~1.0/tile +
提交被 MAT_ACQ ~3.0 阻塞的循环依赖）。
| vre_3 r4 | 环深 2→4 + stationary_do 退役 + dos 流式（v12 基） | 12.675 | 8.311 | 1.5252 | **证伪，归档**（correctness 4/4——r4 治好了 r1 数值污染与 r3 死锁；纯性能负：同日 v12 11.955，慢 6.0%。K3 主钥匙在 SM100 上不成立） |
| vk_4 | M2：softmax exp2 → deg-6 FFMA（近似替换档，已终裁放行） | 9.818 | 8.442 | 1.1630 | **null（预期内分支），代码保留**——同日 vk_2 9.858，仅 −0.041 < 0.05 门。数值面完美：**四 case max_abs 与 vk_2 逐位相同**（deg6 偏移仅现于 mean/cosine 末位），获准类首次硅上端到端确认。性能面 = pacer 判决第三次独立确认（math 松弛 ~1.2µs 全额吸收 SOFTMAX 缩短）。**M2 应收账款条件：W17 < 5.84**——K2-swap（→~6.4）不够，实质 Rubin 锁定 |
| **vk_5** | W1'：W17 卸 TMA 等待，W19 提交中继 | **9.757** | 8.555 | **1.1405** | **部分成立（贴上界 0.3µs），建议采纳**——同日 vk_2 9.857，−0.100ms 且**双指标同向**（ratio 1.1405 vs 1.1666），vk 系列首个真实正收益；correctness 四 case max_abs 与 vk_2 逐位相同（纯重排类 ✓）。验尸：仅兑现期望 1/3——**2 槽信用界吃掉了链压缩**（W17 的 TMA 等待转成 acquire 空等；环的供给循环 = release→ready 逐 gen 延迟链，与谁在等无关；省下的只有 W17 串行化的跳数）。顺序不变性定理的信用界推论第二次显形 |
| v_gpt_1 r2b | kdq 换 TMA tile::gather4（外部 agent 提案 K2-G4，v_gpt 独立谱系） | 11.707 | 8.368 | 1.3989 | **证伪，归档**（correctness 4/4 PASS——tensormap 构造/半平面地址代数/SW128 相位/OOB 零填全对；纯性能回归 +1.8ms；验尸见下） |

**v_gpt_1 验尸（2026-08-06）：TMA 异步引擎是共享串行资源，512B 级小事务在它上面排队**
- 机制全对（4/4 PASS 证明 gather4 全链路成立，含 holes/-1 与尾 tile 的 OOB 零填）。
- trace（同口径自比）显示**每条腿都变慢**：period 6.80→9.78；ROUTE_K 2.27→3.17
  （64 个索引载入挪上 W17 串行路径 + TMA 飞行裸暴露）；MAT_QDO 2.19→2.88/span
  （**面板也慢了**——64 笔 512B gather4 与面板 bulk 拷贝共用每 SM 异步引擎，
  小事务排队拖累大事务）；REDUCE_ATOMIC 1.8→3.49/span（L2 通路受离散小读干扰）；
  dVdK gap 出现单个 2.9µs 大洞。
- **教训入账**：cp.async（LSU 路径，128 线程并行）→ TMA 引擎（串行队列）对
  512B 级稀疏小事务是负 EV；"transport 优化 kdq"一族到此测尽（cp.async 的
  2.27 已是该物化的地板）。
- **对方向 A 的含义**：kdq 成本无法靠换搬运机器消除，只能**删除物化本身**
  （canonical-K：own 半区 8KB S2S + peer 半区 8KB DSM，皆引擎友好大块事务）。
  No-kdq 的架构论证被这次证伪**加强**。
- 工程遗产：gather4 全套管道（tensormap 经 make_tiled_tma_atom 以 (1,64)-box +
  SW128 组合布局构造 + copy_tensormap 128B 全局落脚（注意指针对齐标注 ≥64）+
  inline asm 发射）已硅上验证，归档备用。
- 另：r2 曾 33 分钟 compile 无输出后 SSH 255（r2b 正常，判环境瞬态，留观）。

**尾段章节关闭（vk_3 验尸，2026-08-06）**：三刀独立手术全部 ≈ 零——TMA 飞行
（vg_6 双缓冲 null）、标量 scatter（vk_3 stmatrix null）、T2R atom（vk_3 同时把
epilogue T2R 换成 Ld16x256b(Rep4)，也在同一个零里）。唯一自洽解释：
**trace 的 DQ_EPI 11.9µs/round 被注入税严重放大**——标量循环是指令最稠密的
区段（128×~6 ops/线程），其局部税率远超全 kernel 平均的 38%，release 侧
真实 DQ_EPI 估计只有 ~3-4µs/round，store 份额 ~0.4µs（与实测 −0.8µs/token 吻合）。
**F 账修正**：ΔF 真实 ≈ 0.10-0.15ms（原估 0.35-0.43 作废），剩余缺口
~85-90% 在稳态侧（W17 饱和链）。教训入账：**跨口径换算不能用全 kernel
平均税率折算指令密度异常的区段**；span 稀疏 ≠ 局部税低。
尾段剩余嫌疑（T2R fence/双 barrier/TMA wait 结构）合计 ~0.1ms，不值再开 rev。
SASS 计数降级为可选存档项，不再花机时。

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
  - S2（单 ready mbar、保留共享 state）：**挂**（run_outcome fail，exit 143 @
    capture_2cta；perf 段 9.809 后 trace 卡死）——共享 state 的代码形态被指认。
  - a_advance（保留共享 state、仅挪 advance 位置）：**挂**（同 exit 143）——
    挪相位算术救不了共享结构本身。
  - s1a（S1 的变体）：过（run_outcome pass, exit 0）。
  - 机制终判：**触发要件 = relay 两条 pds 管线共享同一 producer state 的代码形态**；
    拆开（S1/s1a）或删除第二条管线（C）均消除卡死。
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
F 账：vk_1 F≈31µs/token（trace 口径）vs baseline 6.8。
~~ΔF≈0.35-0.43ms（缺口 ~30%）~~ **已作废**——vk_3 验尸（见上）证明该折算用了
全 kernel 平均税率，而 DQ_EPI 区段局部税远超均值；修正 ΔF≈0.10-0.15ms。

**pacer 判决（vk_1 trace，推翻 vre_1 时代的"环松弛"框架）**：W17 供给链自身
饱和——ROUTE_K 2.27（真身 = kdq 会合区）+ MAT_QDO×2 4.38 = 6.65 ≈ period 6.80，
零 slack。这重释了两桩旧案：vd_1 null（拓扑等价、槽没变多）与 vh_1 里环深 3
即使资金无害也兑现不了（供给链饱和时加槽无用）。手术梯子 K1-K5 见
外部复盘（~/proxy/dsa-vk1-traceable-r1）；K3（环深）必须排在 K2（kdq 退役）
之后。CG1 混流 ISA 裁定仍未落章。

## K2 定价采集（vk_2_trace @ c091522，2026-08-06，稳态 tile 9..24）

- **W17 饱和在现役基座复核成立且更紧**：ROUTE_K 2.041 + MAT_QDO×2 4.878 =
  6.919 = **97.6%** of period 7.089（vk_1 上为 6.65/6.80，97.8%）。K2 前提 ✓。
- **判决 2 被推翻（本采集最重要的产出）**：WAIT_dQ 仅 0.071µs/次，其结束贴
  PDS_WAIT end（+0.028/+0.112），而 kdq commit **早 ready 0.554µs**——
  **dQ 的门是 pds 发布，不是 kdq 会合**。外部复盘"grads 块头被 kdq 顶住
  1.3-1.6µs"在现役基座不成立。
- 新 span 首采：PDS_WAIT 0.506、GRAD_SUP_WAIT 0.260；ROUTE_K 内信用等待
  （前两个 MAT_ACQ）0.604。可搬上界 Σ1.370µs/tile（三段有重叠，非直接兑现额）。
- 供给行合计：MAT_ACQ 3.708 / MAT_WAIT 1.368 / RK_ACQ 0.858。
- dVdK gap 台账：[0.326,0.028,0.436,0.544,0.464,0.492,0.414,0.494]，
  Σ3.198 vs vc_2 的 3.890（vg 系列净减 0.69；头部 0.99→0.33 已治愈）。
- math 行（M2 单 warp 口径，现役基座基准）：SOFTMAX **2.047**、PDS_ACQ 1.264、
  STORE 1.005、T2R×2 1.523——math 串链 ≈5.84，将是 K2 之后的下一个 pacer。

**K2 形态修订：de-rotation 重设计降级为备选，新首选 = K2'（kdq 后置/会合隐藏）**
判决 2 推翻后 kdq 无需退役：把 W17 的 kdq 会合从链头挪到链尾（gen 序改为
quads 先、kdq 末两位；2+6 偶数分组 ⇒ 槽位奇偶映射不变，fragment 视图零改动），
gather 的 ~1.4µs 填充隐藏在 4.9µs panel 火车之下；leader 侧 dQ 从 grads 块头
移到块尾（dQ 本就是 pds 门控，挪动无代价；dq_done 仅末 tile 相关）。
零字节、零新管线、零 DSM、无 loan/kscore 改动。~~预期 −0.5~0.8ms~~
**等待图判决（同日深夜）：K2' 与其流水化改进 K2'' 双双纸面证伪**——
2 槽 FIFO 下环周期 ≈ 全部 8 gen 供给时间之和，对排列顺序一阶不变；
kdq 的 2.0µs（信用 0.6 + 填充 1.4）挪到哪都串行计入，链头反而是最优位
（信用在上一窗已释放）。零机时排掉两个必 null 的 rev。
幸存形态（K2_DESK_NOTES 末章）：**K2-swap**（loan 改载 kdq，2×16KB 恰好
= 借用窗；−0.3~0.5ms，风险中）与 **de-rotation K-split**（−0.5~0.8ms，
风险中高）。**战略结论：K3（+1 槽，16KB，资金=vre_3 panel 流式化）一把
钥匙开两把锁**（kdq 税 + 0.45/gen 信用平台）——vre_3 状态 = 当前最高
价值未知量。

## 工装备忘

- trace 采集在 21:48 后连续超时（exit 124，capture_2cta 阶段）——**已重定性：非
  infra，系 vg_1 split publish 注入期卡死**（见结构性发现 #5）；VsmTopologyMapper
  报错是无害噪声（成功 run 同样打印）。baseline/vc_2 通道一直正常。
- `.git/index.lock` 出现过陈旧残留（0 字节、无 git 进程），清理后正常。

## K2 桌面判决（2026-08-06 晚）：K2a/K2b 纸面否决，改打 v_f2_1（K2c）

**K2a/K2b（kdq 从 score_kv 拿字节）到货边一票否决**：kdq(t) 的 ring 槽 g0/g1(t)
空闲 ← g6/g7(t−1) 释放 = train(t−1) 末段 dK-r1 MMA 完成（窗口 t 晚段）；而 K(t)
死亡 = loan(t−1) 填充 ← dP(t) MMA 完成（窗口 t 早段）；loan 又喂同一 train 的头
两个 pass。强令 copy-before-loan ⇒ copy←槽←train 末段←loan←copy，**闭环死锁**；
暂存需 16KB（无源）；gen 重排救不了（kdq 需要 ~8 代前瞻，2 槽环只给 2 代）。
**推论入账（vh_1 教训的推广）：共享字节 = 共享生命期 = 耦合调度。第二次 GMEM
gather 不是浪费——它就是 kdq 填充时机与 score_kv 生命期的解耦器（且 L2 命中，
真实代价只有发射/地址工作）。**

**v_f2_1（独立前缀 v_f2）= kdq 供给段自治**：gather 警组自持 g0/g1 信用、自填、
按 round 拆分提交（fill r0→commit g0→fill r1→commit g1）；W17 的 160 线程
kdq_barrier 双会合退役（barrier id 7 释放），W17 仅 advance 越过这两代，直入
panel 火车。双生产者相位记账 = vd_1 已证结构。零字节。预期：W17 串链 6.919→
~4.9（甩掉 ROUTE_K 段）；dQ r0 提前 ~半个 fill 解锁；release −0.2~0.5ms。
止损门：correctness 4/4；candidate ≤9.80 且 ratio 同向改善 = 采纳；≥10.0 或
双指标矛盾 = 证伪归档；超时按 SOFT DEADLOCK 处置流程。

## v_f2_1（K2c）四连败证伪：kdq 供给段整族关闭（2026-08-06 晚）

| rev | 单变量 | 结果 |
|---|---|---|
| r1 (d0fe890) | 初版 | 编译错：`cute.jit` 装饰器叠层（脚本回滚残留）；无 GPU 信息 |
| r2 (77e2d65) | gather 自治 + 拆分提交 | correctness FAIL：dense dQ 4.8%（max 1.029 @ (7,101,57)） |
| r3 (eda65fb) | + 还原 v8 线程形状（warp0 统一 acquire、barrier 后 fence、elect_one commit） | FAIL：7.0%（max @ (40,66,181)）——线程形状假说证伪 |
| r4 (b0f3ee0) | + 撤拆分提交（数据路径逐字节 = v8） | FAIL：7.7%（**max 又回到 (7,101,57)**）——与 v8 唯一差异 = 生产者身份 |

**判决**：ring 管线的 kdq 两代生产权**不可从 W17 转移**（至少在当前管线原语的
使用方式下）。K2a 纸面否决 + K2c 三次实测证伪 ⇒ **kdq 供给段手术整族在 SM100
上关闭**。v_f2_1 归档，现役基座回 vk_2。

**验尸线索（供后续原语级追因）**：
1. 损坏有可复现空间签名（r2/r4 同一最差坐标），非纯时序混沌；量级 4.8→7.0→7.7%。
2. vd_1 先例被我误读：它按**奇偶**拆双生产者 = 每个物理 mbar 只有一个生产者线程；
   v_f2_1 是**首次让两个生产者在同一 full mbar 上按 tile 内顺序交错到达**
   （gather: gen0 → W17: gens 2,4,6 同一 slot-0 mbar）。这是唯一未被任何先例
   覆盖的结构差异，现为头号机制假说。
3. cluster 管线 full mbar 到达计数 = 2/代（每 CTA 一个 elect lane，双方打到
   leading CTA；pds 创建处注释自证）。计数错配会相位翻倍失步——但 v_f2_1 的
   commit 形状每 CTA 恰 1 次，账面成立；矛盾待管线源码裁决。
4. **待办artifact**：容器 venv 的 CUTLASS pipeline 源码 dump
   （PipelineAsyncUmma 的 producer_acquire/commit、cluster arrive 语义）——
   固定格式自动化脚本忽略了该请求，需 runner 会话人工执行。
5. 教训：'先例覆盖'的判定必须精确到结构同构（谁在哪个 mbar 上到达），
   不能停留在'双生产者'这个词面。

**棋盘现状**：供给段（K2a/K2c）关闭；K3 资金已被 vre_3 r4 证伪；尾段已关闭
（vk_3）。剩余杠杆：**M2 FFMA deg6 exp（等近似替换档终裁，唯一大额零风险杠杆）**、
D1/角色再平衡（~0.4µs 级）、SM100 追平线接受判定（E1 已落章：baseline 被自身
drain 地板钉在 ~8.4-8.6）。
