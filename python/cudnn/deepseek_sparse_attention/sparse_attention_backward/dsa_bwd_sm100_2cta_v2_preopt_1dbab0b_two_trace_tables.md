# IKET 双 trace span 耗时

- Baseline: `/tmp/dsa-image-run.HGDiiLYY/repo/agents/dev_pipeline/trace_archive/20260728T224458Z_v2_v2/trace/traces/baseline`
- Candidate: `/tmp/dsa-image-run.HGDiiLYY/repo/agents/dev_pipeline/trace_archive/20260728T224458Z_v2_v2/trace/traces/2cta`
- 口径: `mean_us`，role profile `dsa_v2_native`，semantic profile `dsa_v2_native`，显式 `WAIT_*` 已排除；单位均为 μs。
- 注意: 非 WAIT span 内部的隐式等待无法扣除；各行忽略 overlap，不能相加为 kernel wall time。
- V2 provenance: runtime marker 已在 CTA0/CTA1 的 warp16 验证；marker payload 状态为 `matched_cta_rank`。
- V2 issue 粒度: 每 tile 为 2 个 dQ issue 和 8 个 payload 独立的 dV/dK pass issue；中间 wait 不计入 issue span。

## 表 1：可比阶段耗时

| 阶段 | 归一化单位 | Baseline 公式 | Baseline | Candidate 公式 | Candidate | Δ | Ratio |
|---|---|---|---|---|---|---|---|
| Q/dO startup load issue | per H128 launch | LOAD_QDO | 0.672 | LOAD_QDO | 1.248 | 0.576 | 1.857x |
| LSE/sum(O*dO) startup load | per H128 launch | LOAD_STATS | 0.144 | LOAD_STATS | 0.864 | 0.720 | 6.000x |
| K/KV load | per tile | LOAD_K(i) | 3.238 | LOAD_K(i) | 20.631 | 17.393 | 6.372x |
| S+dP tensor-core issue | per tile | S_dP(tile) | 1.358 | S_ISSUE(i) + dP_ISSUE(i) | 4.518 | 3.160 | 3.327x |
| P+dS T2R/math | per tile | P(tile) + dS(tile) | 0.985 | T2R_S(i) + T2R_dP(i) + MATH_PD(i) | 15.919 | 14.934 | 16.161x |
| dQ+dVdK tensor-core issue | per tile | dV_dK_dQ(tile) | 2.956 | 2×dQ_ISSUE(i,r) + 8×dVdK_ISSUE(i,r,p) | 2.192 | -0.764 | 0.742x |
| dKV T2R+atomic | per tile | 2×REDUCE_dKV(tile,part) | 4.409 | 2×REDUCE_T2R(i,r) + 2×REDUCE_ATOMIC(i,r) | 5.830 | 1.421 | 1.322x |
| dQ epilogue | per launch | dQ_epilogue | 7.296 | 2×DQ_EPI(r) | 16.736 | 9.440 | 2.294x |

## 表 2：Candidate 额外耗时

| 额外阶段 | 归一化单位 | 公式 | 耗时 | 说明 |
|---|---|---|---|---|
| S+dP acquire/publish | per tile | S_ACQUIRE(i) + S_PUBLISH(i) + dP_ACQUIRE(i) + dP_PUBLISH(i) | 0.825 | Pipeline bookkeeping outside the pure MMA issue ranges. |
| ROUTE_P | per tile | ROUTE_P(i) | 0.440 | No standalone baseline span; software route-issue scope. |
| ROUTE_K | per tile | ROUTE_K(i) | 5.613 | No standalone baseline span; inclusive load-path scope. |
| ROUTE_dS | per tile | ROUTE_dS(i) | 0.460 | No standalone baseline span; software route-issue scope. |
| MAT_QDO | per tile | 2×MAT_QDO(m,r) | 15.620 | Two payload-decoded Q/dO materialization rounds per tile; inclusive. |
| Routes + MAT_QDO subtotal | per tile | ROUTE_P(i) + ROUTE_K(i) + ROUTE_dS(i) + 2×MAT_QDO(m,r) | 22.133 | Factor-weighted absolute sum; components can overlap and this is not kernel wall time. |
| TAIL | per launch | TAIL | 34.976 | Final gradient drain and pipeline-tail inclusive scope. |
