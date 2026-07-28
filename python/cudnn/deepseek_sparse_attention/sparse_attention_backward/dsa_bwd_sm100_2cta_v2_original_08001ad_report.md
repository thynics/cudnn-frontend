# 原始 V2（08001ad）测试报告

## 结论

原始 V2 **没有通过 correctness，也没有产生合法的正式性能或 IKET
trace 数据**。它在第一个 `dense` correctness case 中失败，标准流水线随即
终止，未进入 release benchmark、IKET capture、双 trace 表或聚合时间线阶段。

因此，任何声称属于原始 `08001ad` V2 的 latency、加速比、trace 对比或
pipeline 时间线，若来自后续源码、旧实现路由、baseline 复用，或来自文件内
未执行的旧 V1 类，均属于错误归因，必须作废。

## 原始版本身份

| 字段 | 值 |
|---|---|
| Commit | `08001addf1f0fbc3c0e035ccae314a7e9dbf8340` |
| Subject | `Add DSA v2 rotated-schedule 2-CTA experiment and design doc` |
| Author / committer | `longcheng <longcheng@nvidia.com>` |
| Author date | `2026-07-28T22:55:45+08:00` |
| Commit date | `2026-07-28T22:57:23+08:00` |
| V2 source SHA256 | `1425bbca516f24bccbdaa9fa573f17defd002f005db5ccc7152b5df055678a3e` |
| GPU | NVIDIA B200 |

## 原始 correctness 数据

下表中的 `dense 执行阶段耗时` 只是失败 correctness case 的外层阶段计时，
**不是正式 kernel latency，禁止当作性能数据使用**。

| 项目 | 20260728T145847Z 原始运行 | 20260728T210935Z 精确字节复测 |
|---|---:|---:|
| Candidate SHA256 | `1425bb…678a3e` | `1425bb…678a3e` |
| 编译时间 | 15.849394 s | 15.510627 s |
| dense 执行阶段耗时 | 1.360 ms | 1.350 ms |
| dKV 错误元素 | 240466 / 262144 | 240466 / 262144 |
| dKV 错误比例 | 91.7% | 91.7% |
| 最大绝对误差 | 3.758673906326294 | 3.758673906326294 |
| 最大绝对误差位置 | `(129, 135)` | `(129, 135)` |
| 最大相对误差 | 117896.4453125 | 117896.4453125 |
| 最大相对误差位置 | `(14, 380)` | `(14, 380)` |
| correctness | **FAIL** | **FAIL** |

两次运行得到完全一致的错误计数和最大误差，复现了原始实现的错误。

## 正式性能数据

| 项目 | Baseline | 原始 V2 |
|---|---:|---:|
| release latency | 未执行 | 未执行 |
| TFLOP/s | 未执行 | 未执行 |
| same-run ratio | 不存在 | 不存在 |

原因：correctness 是性能测试的硬前置条件；原始 V2 未通过该条件。这里不能用
失败 case 的 1.350/1.360 ms 阶段计时替代正式 benchmark，也不能引用任何
后续修复版、baseline-derived 版本或旧实现的 latency。

## IKET / trace 对比

| 项目 | Baseline | 原始 V2 |
|---|---:|---:|
| IKET capture | 未执行 | 未执行 |
| decoded trace | 未生成 | 未生成 |
| 双 trace span 表 | 未生成 | 未生成 |
| role 聚合时间线 | 未生成 | 未生成 |

原始 V2 活跃类内没有 source-native IKET annotation。旧 harness 的自动补丁
命中了同文件内未执行的 `V1A0/T1d` 类；将那条旧类 trace 标为 V2 trace 会
构成错误归因。

## 对历史虚假数据手段的结论

历史上通过以下手段让测试或 benchmark 表面通过：

- 将 V2 路由到旧实现或 baseline-derived 实现；
- release 与 trace 使用不同实现；
- 把旧 V1 类的 trace 标成 V2；
- 隐去原始 V2 correctness 失败，再展示其他源码 SHA 的性能。

我们严肃谴责上述行为。它不是普通优化失误，而是通过路由和 hack 构造测试、
benchmark 与 trace 表面通过的假象，进而制造虚假性能数据的严重工程诚信
违规。相关“V2 超过 baseline”结论全部无效，必须撤回；不能继续引用，也
不能用任何后续真实优化收益或委婉措辞淡化、掩盖这段历史。

后续 V2 结果必须同时证明：显式实例化活动 V2 类、release/trace 源码 SHA
一致、无 candidate-side trace patch、全 CG2、每 tile 两轮共八个纯 K64
dV/dK pass、correctness 先通过，以及性能来自 IKET-off 的 same-run A/B。
