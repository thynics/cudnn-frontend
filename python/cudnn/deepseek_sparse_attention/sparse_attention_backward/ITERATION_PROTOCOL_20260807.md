# DSA bwd 迭代协议 v2（2026-08-07 起用，取代 r1-r3 的成对协议）

**动机**：r1-r3 每轮 = 探针 + 候选腿 + 参照腿，参照腿把 v_w3_2 完整重编译
重测一遍，轮次成本 ×2。编译是每腿大头；bench 本身（warmup5+repeat5）
只有几十毫秒 GPU 时间。

## 腿型（三种，按需取用）

1. **迭代腿（默认）**：单腿 `--mode validation`。**正确性+性能一次拿到**
   （四 case 指纹 + baseline/candidate bench 同在 validation_summary.json）。
   不跑参照腿，不跑探针。
2. **锚定腿**：`--impl v_w3_2 --mode validation`，仅在 service/job/node/image
   任一变化时、或连续 ~10 轮后跑一次，刷新冻结水线。
3. **trace 腿**：`--mode all`（IKET 捕获，需 trace 孪生文件），仅在需要
   span 级归因时。探针 probe_v_s1_layouts.py 降级为条件触发
   （image/DSL 版本变化或诊断布局类编译失败时）。

## 判读（双冻结基准，抗漂移）

- 正确性硬门：四 case max_abs 指纹逐位同 final（指纹表见 V_S1_SPEC.md /
  历轮 prompt）。
- 性能双指标（**全部对冻结值，不再要求同 run 参照**）：
  - 绝对水线：candidate_ms vs **9.4565ms**（v_w3_2 @ r3，service
    20260807T055917Z-1873583 / umb-b200-248）；
  - 比值水线：two_cta_over_baseline vs **1.1441**（in-leg 1-CTA baseline
    免费自带，比值口径天然抵消日间漂移 ±2%）。
  - 两指标同向才下结论；矛盾 = null 并触发一次锚定腿。
- 门不变：≥0.05ms 达标 / null 归档 / 负停（诊断腿可豁免，见下）。
- **终审例外**：任何要进"交付/超越 baseline 判定"的结论，仍须一次
  同 service 成对复测确认（成对协议只从迭代循环退役，不从终审退役）。

## 变体打包（迭代加速的主力）

一个任务目录、一个 service 窗口内**串跑多个候选变体**：
```
run_dsa_b200_pipeline.sh --task-dir <dir> --impl v_s1_d1 --mode validation --note <slug>_d1
run_dsa_b200_pipeline.sh --task-dir <dir> --impl v_s1_a  --mode validation --note <slug>_a
...
```
每变体一份 validation_summary，全部对冻结水线判读；service
建立/同步成本一次摊销。诊断变体（如流深 4 的 D1）标注"诊断腿，
负停门豁免，只取量化归因"。

## prompt.txt 模板（迭代腿）

```
# 任务：<impl> rN —— <一句话意图>（迭代腿，冻结水线判读）
## 精确 revision
- repo/branch/revision/文件/SHA256（逐一核验，不符即停）
## 执行
- run_dsa_b200_pipeline.sh --task-dir <本目录> --impl <impl> --mode validation --note <slug>
-（可选）追加变体腿若干，同 service 串跑
## 预登记判读门
- 四 case 指纹逐位同 final（指纹表）
- candidate_ms vs 9.4565 且 ratio vs 1.1441，双指标同向；≥0.05 达标/null 归档/负停
-（诊断腿声明豁免负停，只记录量化结论）
## 产物
- lightweight/ + result.txt（指纹对照、双指标、结论）+ done.txt
## 失败协议
- 编译/验证失败=可迭代非终局；存全量 traceback + result.txt 即可 done.txt
```

## 水线台账

| 日期 | service/node | v_w3_2 ms | ratio | 备注 |
|---|---|---|---|---|
| 2026-08-07 | 20260807T055917Z-1873583 / umb-b200-248 | 9.456525 | 1.144074 | r3 参照腿 |
| 2026-08-09 | 20260809T094229Z-4074329 / umbriel-b200-068 | 10.045306 | 1.067867 | vfinalexgh r3 锚定腿；该 node 绝对值慢 ~6%、baseline 慢 ~14%，冻结值判读在此窗口失效，必须同窗配对 |
