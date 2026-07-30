# v16 runner 注记（随提交发布）

v16 = final 定版血统 + Rev-1 两杠杆（评审报告《两把刀评审_20260730》《头尾供应
解剖_20260730》的唯一幸存项），trace-free（与 final 同，无 IKET 名额占用）。
final.py 定版不动，仅作 release 对照。

## 杠杆与 bisect 开关（import 时读 env，同 v15 先例）

| 开关 | 默认 | 含义 |
|---|---|---|
| `DSA_V16_EAGER` | 1 | W19 急切提交引擎：round 环 8 个 panel commit 从 W17 滞后位（等下一次 fill 发射才提交，k_eff=1）移到空闲 warp19 单 lane 循环（TMA 落地即提交，k_eff→2）。W17 保留 acquire + kdq commit（拆分生产者，先例 = pds 管线的 math-acquire/W18-commit） |
| `DSA_V16_PDS_HOIST` | 1 | W18 的 pds producer_commit 从双 DSM push 之后上提到 pds_ready 之后，把慢 CTA 的 push 发射窗移出 leader 的 dQ 门。推送半区仍由 landing→relay_mbars 门保护 |

双 0 = 逐语义复现 final。注意三点：① env 在 **import 时**读取——bisect 必须
每格新进程且在进程启动前 export；② 只有字面 `"0"` 关闭（空串/"false"/"off"
均视为开）；③ 默认 1/1 是 Rev-1 合并任务书钦定，与《两把刀评审》§4 的
"全默认 0、基线 v15 rev11"计划相反（本实现基线为 final）——分析侧请以
manifest/日志中打印的 `V16_EAGER`/`V16_PDS_HOIST` 类属性实际值为准。

## 预期（B200 release，H128/D512/S4096/topk2048）

- EAGER 单开：period 7.12 → ~6.6-6.7（−0.4~0.5µs/tile）
- EAGER+HOIST：→ ~6.33-6.53（HOIST 增量 −0.19~0.29）
- 判读扳机：HOIST 腿净降 <0.15µs → 单独回滚 HOIST（回退证实"另一腿补位"）；
  EAGER 单开回退 >2% → 使能项 NO-GO 全案复议。

## stage-0 门（B200 run 之前）

- G0: SMEM 不变（零新增字节，期望 DYNAMIC_SMEM 231,424 同 final/v15）。
- G1: USETMAXREG 不变（W16-19 组 decrease 48；池精确 61,440 =
  (8×48 + 4×128 + 8×128)×32——注意 final 实测几何为 48/128/128，
  非 v15 注记的 64/120）。
- G2: W16/W17/W18 分支零新增 STL/LDL（硬门）；新增条款：W19 循环体零 STL/LDL。
- G3: 不变（stmatrix 8 / S2G 2 / W18 G2S 不适用——v16 无 L2X）。

## 运行序列

1. smoke：`./benchmark/dsa/run_b200_pipeline.sh --impl v16 --mode validation`
   （正确性 4 型 + release 对时 vs baseline；候选默认 1/1）。
2. bisect（fresh process per cell）：全关 / EAGER 单开 / EAGER+HOIST，
   三点 release 对时（全关应复现 final ≈11.6ms）。
3. 通过后 vm6probe 复采（独立探针孪生文件，另行 fork；顺带采三个 PARK
   复活触发器：KDQ_SPLIT / DS_FIRST / 腿拆分，预期全不触发）。

## 失败处置

- hang/超时 = STOP：优先怀疑环拆分生产者的相位或 skip-advance 对齐，
  回 W19 循环推演表全审；watchdog 写 .FAILED。
- 正确性挂 = 立即用 bisect 定位到单杠杆；HOIST 引发的挂优先重审
  dq_ds_fragment(ds_image) 的双半就绪链。
