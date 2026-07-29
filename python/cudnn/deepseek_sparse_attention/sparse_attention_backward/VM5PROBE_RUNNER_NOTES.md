# vm5probe runner 注记（随提交发布，优先级高于此前 prompt 中冲突的条目）

vm5probe = M5 插桩探针变体，**永远不是性能候选**。不要把它的耗时当性能结果、
不要用它更新任何 performance JSON。

## 覆盖此前 prompt 的条目

1. **第 1 步预检作废**：此前 prompt 要求
   `python -c "from cutlass.cute.arch import mbarrier_try_wait"`，ImportError 即 STOP。
   **该检查已过时——代码最终版使用文件内自带的 `_mbarrier_try_wait` PTX wrapper
   （硬件验证过），不依赖 cutlass 的该 API。ImportError 不是失败，直接继续。**

## 必须跳过的 harness 校验（按设计会失败）

- two_trace_tables 比较器、span_family_validity、per-tile issue-count 检查、
  role-partition / provenance 校验（iket_span_semantic_map_dsa_v2_native.json）。
  原因：6 个 span 名退役（WAIT_S/T2R_S/WAIT_dP/T2R_dP/MATH_PDS_ACQ/MATH_BAR1），
  5 个新增（RK_ACQ/MAT_ACQ/MAT_WAIT/PROBE_GEN/PROBE_DONE），且 W19 成为新的
  发射 warp 但没有 ROLE_* mark。只做 raw 提取：
  `trace/traces/2cta/iket/pid_*/iket.decoded_results.json` → CSV，原样交付。

## 额外检查项（写进 probe_integrity.md）

- **W19 归因**：确认 decoder 对无 mark 的 W19（unmarked warp with events）如何归因，
  报告处理方式。
- **缓冲溢出**：W19 新增 ~480 ranges/CTA、W17 新增 ~544 ranges/CTA——检查 decoded
  结果中的 overflow/drop 指示。
- **PROBE 流完整性**：每个 grads block 应有 10 个 PROBE_GEN + 3 个 PROBE_DONE，
  每 tile 另有 2 个 PROBE_DONE（S/dP）。若在某 tile 截断：给出最后成功的
  (name, payload)。截断后的近零时长 PROBE range 是 disarm 后的幽灵条目，
  分析侧会过滤，不必处理。
- **超时预期**：W19 探针在所有 build（perf/correctness/trace）都运行。健康 run
  的 wall 应与 v12 同量级（候选 ≈ 11.65ms ±2%）；若 parity/顺序模型出错，
  每 launch 最多慢 ~5-10 秒（每 CTA 一次 2^22 自旋 disarm，~55 waves）——这是
  有界 fail-soft，**不是 hang**。确认 harness 各步超时阈值 ≥ 60 秒/launch，
  必要时调高后重跑该步。
- **SASS/寄存器门**：对照 v12 build，报告 W17/W19（同在 decrease(48) 组）的
  新增 spill；有 spill 记录数值即可，不作为 STOP 条件（插桩变体）。

## 产物清单不变

按此前 prompt 第 4 步的 10 项文件名写入 `/home/longcheng/artifact/vm5probe_run1.partial/`，
`sha256sum * > MANIFEST.sha256` 后 `mv` 原子发布为 `/home/longcheng/artifact/vm5probe_run1/`。
STOP 级失败写 `/home/longcheng/artifact/vm5probe_run1.FAILED`（含原因与日志文件名）。
ncu 部分照旧：跑在未插桩的 v12 与 baseline 上，metric 三组
（sm__pipe_tensor* 占空、lts__t_sector*hit*、dram__bytes*），可证伪预测：
v12 tensor 占空 21-40%、baseline ~40-45%。
