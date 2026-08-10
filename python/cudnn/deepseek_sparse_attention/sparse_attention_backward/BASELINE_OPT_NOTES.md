# baseline_opt 运行与判读笔记（2026-08-09）

## ★ r5 实测定局（B200 umb-b200-239，22c036d，warmup20/repeat200，venv=dsa_iket_h128_venv_2606/DSL 4.6.1）

三门全 PASS（dq 位级恒等 / dkv 底噪 / 节点稳定 0.75%）+ ENV_ANCHOR PASS（baseline@2048=8.814）。

| topk | baseline(wrapper腿) | all_off(直调) | all_on | **harness 不对称** | **三刀真值** | Δ①epi | Δ②dq_early | Δ③split |
|---|---|---|---|---|---|---|---|---|
| 128 | 1.378 | 1.106 | 1.079 | 0.272 | +0.028 (2.5%) | −0.001 | +0.036 | −0.001 |
| 256 | 1.834 | 1.564 | 1.530 | 0.270 | +0.034 (2.2%) | +0.004 | +0.043 | −0.001 |
| 512 | 2.780 | 2.553 | 2.535 | 0.227 | +0.018 (0.7%) | +0.000 | +0.029 | −0.002 |
| 1024 | 4.764 | 4.573 | 4.569 | 0.191 | +0.004 (0.1%) | −0.014 | +0.018 | −0.008 |
| 2048 | 8.814 | 8.647 | 8.641 | 0.166 | +0.006 (0.07%) | −0.010 | +0.033 | +0.000 |

## ★★ r6b 健康环境复现定局（umbriel-b200-091，.venv/DSL 4.5.0 release，22c036d，2026-08-10）

三门 + 双锚定全 PASS（dq 位级恒等再次成立；baseline@2048=8.796、final@2048=9.194 均在锚内）。
**全列同机同轮复现**——final 曲线不再依赖台账/推断：

| topk | base(wrapper) | base(直调=all_off) | baseline_opt | final | **final/直调base** | 不对称 | 刀真值 |
|---|---|---|---|---|---|---|---|
| 128 | 1.330 | 1.053 | 1.036（0.984） | 1.281 | **1.217** | 0.277 | +17µs |
| 256 | 1.785 | 1.514 | 1.501（0.992） | 1.798 | **1.188** | 0.271 | +12µs |
| 512 | 2.743 | 2.523 | 2.520（0.999） | 2.849 | **1.129** | 0.220 | +3µs |
| 1024 | 4.736 | 4.541 | 4.543（1.000） | 4.956 | **1.091** | 0.195 | −2µs |
| 2048 | 8.796 | 8.624 | 8.633（1.001） | 9.194 | **1.066** | 0.172 | −9µs |

- **F1c 判定**："final 在 topk=128 快 4%"（0.964 wrapper 口径，本轮原样复现）被完全分解：
  final 对称口径在 128 是 **1.217（慢 21.7%，全曲线最差点）**，被 0.277ms 的 wrapper
  不对称遮成"快 4%"。final 对直调 baseline 的差 = 固定项 ~0.205ms + ~0.011ms/tile-step。
  **final 无优势区间；小 topk 甜点 = 100% harness 伪影。**
- **F2_FLAG**：对称口径 final@2048 = 1.066，低于台账 1.137（预登记窗 [1.08,1.22] 出界）。
  符号一致（final 恒慢），幅度随环境/工具链漂移（本轮 DSL 4.5.0、umbriel-091；
  台账 = 战役工具链、异节点）。台账定格数字维持为战役口径锚点。
- 不对称 0.17-0.28ms 在健康 env 与 iket venv 基本同值——是 wrapper 路径本身的
  per-call 成本（对比 7/30 旧面板 baseline@128=1.051→今 1.33，期间已增长），非 venv 特有。
- 刀真值本 env 收窄：②仍唯一为正（Δ² +21~36µs），三刀合计 +1.6%@128 → 噪声@2048。

## r5 面板（iket venv，final 列失真弃用）

判决：**②是唯一真刀（+20-40µs/call，随 topk 近常量=per-launch 固定项，符合设计）；
①判 null（大 topk 侧 −10µs 噪声级偏害——baseline 串行 epilogue 本就大半藏在 reduce
尾部 drain 之下）；③判 null（0，序幕 K-gather-bound 预判命中，分支 E）。**
sweep 的 wrapper腿-vs-直调腿不对称实测 0.17-0.27ms/call（本 venv），**比三刀本身大
一个量级**——一切历史 sweep ratio（含旧 final 面板与"final 小 topk 甜点"）都含此
美化。final 腿本轮 13.19ms@2048 vs 台账 9.44 = **本 venv（IKET 补丁 DSL 4.6.1）对
2-CTA final 严重失真**（1-CTA 家族在锚点内），final 绝对值以台账/标准管线为准；
本 venv 禁用于 2-CTA 判读。产物：~/proxy/dsa-bopt-sweep-r5-1786288559/。

`dsa_bwd_sm100_2cta_baseline_opt.py` = 生产 baseline（dsa_bwd_sm100.py）的子类 fork，
回移植 2-CTA 战役发现的三把每-launch 固定成本刀。数值位级等同 baseline，只改发射
顺序与 SMEM staging。动机：新 final 的 topk 扫描显示其 ~4%@topk128 的领先全部来自
每-token 固定项（dQ epilogue TMA 化 + 提前 commit + 序幕拆分），而非 CG2 架构红利；
本 fork 用于 knife-vs-knife 的公平对照。

## 三把刀与开关（import 时读取，默认全开；换挡必须 fresh process——compile cache 不含 env）

| 刀 | env | 内容 | 预期 |
|---|---|---|---|
| ① | `DSA_BOPT_EPI` | dQ epilogue：sdQ staging 从 1-stage 16KB 扩到 4-stage 64KB（恰 = 死 sK 全域），四个 D128 panel 先全部 T2R→bf16→SMEM，再在单个 commit group 内连发 4 条 TMA store；砍掉 3 次串行 TMA 完成等待 + 3 组 barrier 对 | −2~3µs/token |
| ② | `DSA_BOPT_DQ_EARLY` | mma_compute_dQ 的 producer_commit 从"循环外+t2r_dKV4 屏障后"提前到最后一个 tile 的 dQ（含 576 的 dQ4）发射之后；compute 的 dQ epilogue 与尾 tile dKV part-2 GEMM + reduce 原子并行。附带 **dealloc 门**（评审面板 MAJOR 修复）：新增 1-stage UmmaAsync 管线 mma_compute_dealloc，mma 在旧 commit 位置（循环外、same_hdim t2r_dKV4 屏障后）做完成追踪 commit，compute 仅在 dealloc_tmem 前 consumer_wait——恢复 dealloc 与尾 part-2 UMMA 写/reduce dKV3 T2R 的可证明排序，epilogue 重叠收益不受影响 | −2~3µs/token |
| ③ | `DSA_BOPT_SPLIT_QDO` | Q 独立 barrier（新 load_mma_Q pipeline，tx=64KB；原 QdO pipeline 降为 dO-only 64KB）；首个 S 只等 Q，首个 dP 才等 dO | 预判 null（序幕瓶颈腿是首 tile K gather ~3.2µs） |

## 安全论证要点（已过三路评审面板：协议 refuter xhigh FIX_THEN_PUSH→已修、转写对账 CLEAN、DSL 七条 CLEAN）

1. **②的 sdQ-over-sK 覆写安全**：PipelineUmmaAsync 的 producer_commit 是 tcgen05
   commit，完成语义覆盖此前发射的全部 UMMA 的操作数读；commit 放在末 tile dQ/dQ4
   发射后 ⇒ compute consumer_wait 放行时 S/dP/dQ 对 sK 的读全部完成；part2 的
   dKV2/3 读 sP/sdOT/**sQT**/sdS（面板更正：补 sQT），不碰 sK。TMEM 侧 dQ0-3 列区
   [192,448) 与 part2 写的列不相交：same_hdim 下 dKV2 别名列 448（=576 模式的 dQ4
   列，same_hdim 下 compute 不读列 448）、dKV3 别名 128；576 模式 dQ4@448 与
   dKV2@64/dKV3@128 不相交。
2. **②的 dealloc 门**（面板 MAJOR + 复审轮 MAJOR 修复）：提前 commit 本身不排序
   compute 的 dealloc_tmem 与尾 part-2 dKV UMMA 写/reduce 末段 T2R。修复 =
   mma 在旧 commit 位置先 `mma_reduce_dKV_pipeline.producer_tail`（reduce 的最终
   consumer_release 在其全部 store_dKV T2R 之后，两 hdim 路径皆然，传递性覆盖
   末段 T2R——含 baseline 自己都只靠时序的 same_hdim dKV2 T2R 窗口），再 commit
   mma_compute_dealloc 1-stage 门；compute 仅在 dealloc 前 wait。dealloc 排序
   对全部 TMEM 访问**可证明**，强于 baseline（baseline 的 dKV2 窗口纯靠 epilogue
   时长遮蔽）。epilogue 重叠收益不受影响（等待只发生在 dealloc 这一条指令前）。
3. **①的 4-stage staging 合法**：D512 下 cosize(sdQ 4-stage)=32768 元素=64KB=
   cosize(sK)（编译期 assert 钉死；两侧同为 element_dtype，量纲可比。assert 在
   python -O 下会被剥除——按 DSL 教训 #3 staged 禁 raise，护栏保持 assert 形态，
   -O 运行属禁止事项）；576 下 sK=72KB 更宽。576 的 sdQ4 仍别名 sK 基址（= stage0），
   其 producer_acquire（1-stage TmaStore = wait 全部 outstanding）保证四条主 panel
   TMA 读完后才覆写。
4. **退化档（tile_count<=0，benchmark 不触及）**：②的循环外补充 commit 条件为
   `tile_count <= 0`（面板更正：覆盖负 topk 脏数据）。注意 same_hdim 下该护栏是
   死代码——mma 先挂在循环外 288 线程 t2r_dKV4 屏障（与 baseline 完全等价的挂法）；
   护栏仅在 576 路径真正生效。③开启时该退化档 dO 的 consumer_wait 不触发、仅
   release：1-stage 单次使用，无悬挂无竞争。
5. **转写白名单补记**：__call__ docstring 与 Knife 1 邻近注释随手术同步改写
   （面板核销）；非手术区已恢复与 baseline 逐字节一致（含 3 处 Unicode 注释字符）。

## 运行协议（proxy 委托）

```bash
# 正确性 + 全量 topk 面板（对照公共 wrapper baseline）
python benchmark/dsa/sweep_topk_2cta.py \
  --impl baseline_opt \
  --class-name FlashAttentionDSABackwardSm100BaselineOpt \
  --topks 128,256,512,1024,2048
# 消融：DSA_BOPT_EPI/DSA_BOPT_DQ_EARLY/DSA_BOPT_SPLIT_QDO 逐一置 0，fresh process
```

门：max|d_dq|、max|d_dkv| 与 baseline 腿逐位对账（数值应位级相同——同一 GEMM 序、
同一量化路径；容差按 sweep 现有口径）；全开 vs 全关（=纯 baseline 行为）先各跑一遍。

## 判读

- 全开预期：topk=128 总时长 −0.10~0.20ms（比值从 1.00 基准往下），topk=2048
  −0.1~0.3ms（~1-3%）；final 对 baseline_opt 在 128 的 4% 领先应被抹平或反转。
- 消融预期：①②各占大头，③≈0（若③显著非零，说明序幕瓶颈判断错误，值得单独 trace）。
- 若全关档与公共 wrapper baseline 不一致（>run 方差），说明 fork 本身引入了扰动，
  先查 SMEM struct 布局差（多了 16B Q mbar 字段）。

## 已知口径事项

- sweep 两腿 harness 不对称仍在（baseline 腿计时内 torch.zeros 分配 workspace）；
  baseline_opt 走 candidate 腿（预分配 + fill_(0)），对比 final 时口径一致，
  对比公共 wrapper baseline 时留意 ~几十 µs 的不对称。
- 模块名带 2cta 前缀纯粹是 sweep 的 `--impl` 命名约定（importlib 拼
  `dsa_bwd_sm100_2cta_{impl}`），kernel 本身是 1-CTA/CG1。
