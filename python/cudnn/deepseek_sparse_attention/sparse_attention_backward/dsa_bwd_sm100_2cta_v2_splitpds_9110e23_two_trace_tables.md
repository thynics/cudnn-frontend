# IKET 双 trace span 耗时

- Baseline: `/tmp/dsa-image-run.FRflE1xx/repo/agents/dev_pipeline/trace_archive/20260728T235149Z_v2_v2/trace/traces/baseline`
- Candidate: `/tmp/dsa-image-run.FRflE1xx/repo/agents/dev_pipeline/trace_archive/20260728T235149Z_v2_v2/trace/traces/2cta`
- 口径: `mean_us`，role profile `dsa_v2_native`，semantic profile `dsa_v2_native`，显式 `WAIT_*` 已排除；单位均为 μs。
- 注意: 非 WAIT span 内部的隐式等待无法扣除；各行忽略 overlap，不能相加为 kernel wall time。
- V2 provenance: runtime marker 已在 CTA0/CTA1 的 warp16 验证；marker payload 状态为 `matched_cta_rank`。
- V2 issue 粒度: 每 tile 为 2 个 dQ issue 和 8 个 payload 独立的 dV/dK pass issue；中间 wait 不计入 issue span。
- V2 P/dS math 粒度: `split_phase`，每 tile 2 个逻辑 phase / 16 条原始 range；拆分 phase 使用绝对时长求和，不把中间 WAIT_dP/T2R 间隙吞入 math。phase 0 包含 P-stage producer acquire，phase 1 包含 dS-stage producer acquire，因此该项是 software scope，不是纯计算或异步完成时间。

## 表 1：可比阶段耗时

| 阶段 | 归一化单位 | Baseline 公式 | Baseline | Candidate 公式 | Candidate | Δ | Ratio |
|---|---|---|---|---|---|---|---|
| Q/dO startup load issue | per H128 launch | LOAD_QDO | 0.864 | LOAD_QDO | 0.608 | -0.256 | 0.704x |
| LSE/sum(O*dO) startup load | per H128 launch | LOAD_STATS | 0.144 | LOAD_STATS | 0.928 | 0.784 | 6.444x |
| K/KV load | per tile | LOAD_K(i) | 3.264 | LOAD_K(i) | 12.660 | 9.396 | 3.879x |
| S+dP tensor-core issue | per tile | S_dP(tile) | 1.380 | S_ISSUE(i) + dP_ISSUE(i) | 4.523 | 3.143 | 3.278x |
| P+dS T2R/math | per tile | P(tile) + dS(tile) | 1.067 | T2R_S(i) + T2R_dP(i) + 2×MATH_PD(i,phase) | 13.202 | 12.135 | 12.373x |
| dQ+dVdK tensor-core issue | per tile | dV_dK_dQ(tile) | 2.980 | 2×dQ_ISSUE(i,r) + 8×dVdK_ISSUE(i,r,p) | 1.625 | -1.355 | 0.545x |
| dKV T2R+atomic | per tile | 2×REDUCE_dKV(tile,part) | 4.498 | 2×REDUCE_T2R(i,r) + 2×REDUCE_ATOMIC(i,r) | 5.661 | 1.163 | 1.259x |
| dQ epilogue | per launch | dQ_epilogue | 7.456 | 2×DQ_EPI(r) | 15.104 | 7.648 | 2.026x |

## 表 2：Candidate 额外耗时

| 额外阶段 | 归一化单位 | 公式 | 耗时 | 说明 |
|---|---|---|---|---|
| S+dP acquire/publish | per tile | S_ACQUIRE(i) + S_PUBLISH(i) + dP_ACQUIRE(i) + dP_PUBLISH(i) | 0.751 | Pipeline bookkeeping outside the pure MMA issue ranges. |
| ROUTE_P | per tile | ROUTE_P(i) | 0.614 | No standalone baseline span; software route-issue scope. |
| ROUTE_K | per tile | ROUTE_K(i) | 5.335 | No standalone baseline span; inclusive load-path scope. |
| ROUTE_dS | per tile | ROUTE_dS(i) | 0.635 | No standalone baseline span; software route-issue scope. |
| MAT_QDO | per tile | 2×MAT_QDO(m,r) | 8.010 | Two payload-decoded Q/dO materialization rounds per tile; inclusive. |
| Routes + MAT_QDO subtotal | per tile | ROUTE_P(i) + ROUTE_K(i) + ROUTE_dS(i) + 2×MAT_QDO(m,r) | 14.594 | Factor-weighted absolute sum; components can overlap and this is not kernel wall time. |
| TAIL | per launch | TAIL | 24.480 | Final gradient drain and pipeline-tail inclusive scope. |
