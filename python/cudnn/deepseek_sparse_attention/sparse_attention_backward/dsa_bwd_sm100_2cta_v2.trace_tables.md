# IKET 双 trace span 耗时

- Baseline: `/tmp/dsa-image-run.T0Lik7bc/repo/agents/dev_pipeline/trace_archive/20260728T192143Z_v2_v2/trace/traces/baseline`
- Candidate: `/tmp/dsa-image-run.T0Lik7bc/repo/agents/dev_pipeline/trace_archive/20260728T192143Z_v2_v2/trace/traces/2cta`
- 口径: `mean_us`，role profile `dsa`，semantic profile `dsa_v1`，显式 `WAIT_*` 已排除；单位均为 μs。
- 注意: 非 WAIT span 内部的隐式等待无法扣除；各行忽略 overlap，不能相加为 kernel wall time。

## 表 1：可比阶段耗时

| 阶段 | 归一化单位 | Baseline 公式 | Baseline | Candidate 公式 | Candidate | Δ | Ratio |
|---|---|---|---|---|---|---|---|
| Q/dO startup load issue | per launch | LOAD_QDO | 0.864 | LOAD_QDO | 2.816 | 1.952 | 3.259x |
| LSE/sum(O*dO) startup load issue | per launch | LOAD_STATS | 0.144 | LOAD_STATS | 0.512 | 0.368 | 3.556x |
| K/KV load | per tile | LOAD_K(i) | 3.311 | LOAD_K(i) | 1.340 | -1.971 | 0.405x |
| S+dP tensor-core issue | per tile | S_dP(tile) | 1.478 | S_ISSUE(i) + dP_ISSUE(i) | 5.015 | 3.537 | 3.393x |
| P+dS T2R/math | per tile | P(tile) + dS(tile) | 1.259 | T2R_S(i) + T2R_dP(i) + MATH_PD(i) | 3.643 | 2.384 | 2.894x |
| dQ+dVdK tensor-core issue | per tile | dV_dK_dQ(tile) | 3.099 | 2×dQ_ISSUE(i,r) + 2×dVdK_ISSUE(i,r) | 3.597 | 0.498 | 1.161x |
| dKV T2R+atomic | per tile | 2×REDUCE_dKV(tile,part) | 4.706 | 2×REDUCE_T2R(i,r) + 2×REDUCE_ATOMIC(i,r) | 2.774 | -1.932 | 0.589x |
| dQ epilogue | per launch | dQ_epilogue | 7.872 | 2×DQ_EPI(r) | 11.584 | 3.712 | 1.472x |

## 表 2：Candidate 额外耗时

| 额外阶段 | 归一化单位 | 公式 | 耗时 | 说明 |
|---|---|---|---|---|
| ROUTE_P | per tile | ROUTE_P(i) | 2.132 | No standalone baseline span; inclusive. |
| ROUTE_K | per tile | ROUTE_K(i) | 5.657 | No standalone baseline span; inclusive. |
| ROUTE_dS | per tile | ROUTE_dS(i) | 2.285 | Active MATH-role scope; inclusive. |
| MAT_QDO | per tile equivalent | MAT_QDO(m,r) | 7.898 | Two macro-rounds serve two tiles; inclusive. |
| Routes + MAT_QDO subtotal | per tile | ROUTE_P(i) + ROUTE_K(i) + ROUTE_dS(i) + MAT_QDO(m,r) | 17.972 | Absolute sum; not kernel wall time. |
| TAIL | per launch | TAIL | 12.032 | Drain/free inclusive scope. |
