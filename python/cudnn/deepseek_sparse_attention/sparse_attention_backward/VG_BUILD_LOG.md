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

全部版本 correctness 4/4 PASS。累计 10.572 → 10.450（−1.2%）。

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

## 下一步（按 可兑现额 / 阻塞项）

| 优先 | 杠杆 | 预期 | 阻塞 |
|---|---|---|---|
| 1 | **D1 warp 重划 640→512** | −0.5-0.8 µs/tile | 需重新安置 4 个 warp 的角色（gather 128 线程是 ROUTE_K 主力，不能简单砍）；D4 microbench 先定上限 |
| 2 | **M2 FFMA deg6 exp** | SOFTMAX 2.16→≤1.1 | **等用户对"近似替换"档终裁**（数值半门已 PASS，deg6 与 MUFU 同误差量级） |
| 3 | M1 条带加宽 8→16/32 | 小 | 寄存器溢出风险（math@128） |
| 4 | exact-P/dS 重布局 | 解锁 M3 + 省 8 KB | 重构级 |
| 5 | ve_1 复活 | 大 | 需 24 KB（panel 流式化） |

**E1（baseline 内部分解）与 CG1 混流 ISA 裁定仍未落章**——前者决定"超越"是否物理可能
（若 baseline 本身 drain-bound ~4.6，追平线即地板），后者决定 P1 梯度归位这条
4.7-5.2 µs 备用路是否存在。两项都是零 GPU 成本。

## 工装备忘

- trace 采集在 21:48 后连续超时（exit 124，capture_2cta 阶段，IKET VsmTopologyMapper
  报 RmCtrlGetVsmMappings 失败）——`--mode validation` 可绕过，正确性+性能不受影响。
- `.git/index.lock` 出现过陈旧残留（0 字节、无 git 进程），清理后正常。
