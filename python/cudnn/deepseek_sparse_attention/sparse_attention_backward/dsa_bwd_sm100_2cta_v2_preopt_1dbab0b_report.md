# 原生 V2 修正后、优化前证据（1dbab0b）

## 适用范围

本目录中的 `preopt_1dbab0b` 数据属于修正 correctness 和加入源码原生
IKET 后、尚未做性能优化的 V2。它们**不属于原始 `08001ad`**；原始版本在
首个 correctness case 失败，因而没有正式性能或 trace 数据。原始结果见
`dsa_bwd_sm100_2cta_v2_original_08001ad_report.md`。

本次强制 B200 流水线完整退出码为 0，并打印 `DSA_PIPELINE_PASSED`。release
性能与 candidate trace 使用同一个 V2 源文件：

| 字段 | 值 |
|---|---|
| Git commit | `1dbab0b3bc8f682e40e6605ce8d1774ac163cbe2` |
| V2 source SHA256 | `633d168d364eff6c9d190d8b2e29ee1cf410b2da480767dd79657462af52d236` |
| Active class | `FlashAttentionDSABackwardSm100TwoCTAV2` |
| Candidate-side trace patch | `null` |
| Source-native IKET events | 31 |
| Native provenance | CTA0/CTA1 warp16，payload 分别为 0/1 |
| GPU | NVIDIA B200 |
| Run ID | `20260728T224458Z_v2_v2` |

## Correctness

四种 metadata pattern 全部通过：

| Pattern | dQ max abs | dKV max abs | dSink max abs |
|---|---:|---:|---:|
| dense | 0.0040819645 | 0.0096895695 | 0.0011278465 |
| lengths | 0.0119615793 | 0.0728607178 | 0.0372953415 |
| holes | 0.0045598745 | 0.0108621120 | 0.0016178787 |
| all_empty | 0 | 0 | 0 |

## IKET-off 正式性能

口径为 H128、D512、S=4096、topk=2048、warmup=5、timed repeat=5。
性能阶段关闭 IKET。

| 实现 | Latency | TFLOP/s |
|---|---:|---:|
| Baseline | 8.099001 ms | 678.794573 |
| 原生 V2（修正后、优化前） | 20.378105 ms | 269.777690 |

V2 / baseline latency ratio 为 **2.516126**，即 V2 明显更慢；这份数据不支持
任何“V2 已超越 baseline”的说法。

## Trace 对比摘要

IKET trace 验证了两个 V2 CTA、每 CTA 20 个 warp、无 malformed range、无
cross-warp range、无 payload mismatch，并验证了每 tile 的 2 个 dQ issue 和
8 个独立 K64 dV/dK pass issue。

| 可比阶段（归一化） | Baseline | V2 | Ratio |
|---|---:|---:|---:|
| K/KV load / tile | 3.238 μs | 20.631 μs | 6.372x |
| S+dP tensor-core issue / tile | 1.358 μs | 4.518 μs | 3.327x |
| P+dS T2R/math / tile | 0.985 μs | 15.919 μs | 16.161x |
| dQ+dVdK tensor-core issue / tile | 2.956 μs | 2.192 μs | 0.742x |
| dKV T2R+atomic / tile | 4.409 μs | 5.830 μs | 1.322x |
| dQ epilogue / launch | 7.296 μs | 16.736 μs | 2.294x |

这里的 span 是软件 annotation 区间，不是异步硬件完成时间；各行可能重叠，
不能相加为 kernel wall time。正式性能只以上一节的 IKET-off 结果为准。

## 同目录证据文件

- `*_baseline_performance.json` / `*_candidate_performance.json`
- `*_correctness.json` / `*_validation_summary.json`
- `*_trace_summary.json` / `*_result.json`
- `*_two_trace_tables.md` / `*_two_trace_tables.json`
- `*_role_aggregated_h128_i1_i3.timeline.json`
- `*_role_aggregated_h128_i1_i3.chrome.json`
- `*_role_aggregated_h128_i1_i3.svg`
- `*_role_aggregated_h128_i1_i3.shared.svg`
- `*_image_run_manifest.json`

我们严肃谴责历史上通过旧实现路由、baseline-derived 路径、release/trace
分流或把 V1 trace 改名为 V2 来构造测试和 benchmark 表面通过、制造虚假性能
数据的行为。这属于严重工程诚信违规；相关结论必须撤回，不能与本证据混用
或继续引用，也不能用后续真实优化收益淡化。
