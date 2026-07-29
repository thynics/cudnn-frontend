# V2 design-16-warp evidence (`64aca16`)

This record covers the source-native V2 change that brings the CTA role
partition back to the design document:

```text
W0-W3 gather, W4-W7 math, W8-W11 dKV reduce,
W12 MMA leader, W13 load, W14 relay, W15 idle
```

The dKV drain now uses 128 threads with 64 FP32 values per thread instead of
256 threads with 32 values per thread. Total FP32 coverage and total
`red.global.add.v4.f32` calls are unchanged. The five GEMMs remain CG2, and
no V0/V1 implementation, baseline implementation, routing, or candidate-side
trace patch is used.

The first mandatory run completed correctness, release performance, and raw
trace capture, but failed in the trace-summary stage because a fourth harness
check still required exactly 40 warp lifetimes. That infrastructure failure
is preserved as local run
`.dsa_b200_results/20260729T005749Z_design16_v2`; it was not treated as a
kernel pass.

The harness was then changed fail-closed:

- `design16` requires exactly warps 0-15 in each CTA, marker warp 12, rank
  payloads, and the exact W0-W15 role partition.
- `legacy20` retains the former exact 20-warp contract.
- mixed layouts, missing/extra warps, bad markers, bad payloads, bad roles,
  and V1 fallback all fail.
- the summary layer accepts 32/40 only for `v2native`, then requires its
  strict provenance metadata and exact `2 * warps_per_cta` agreement.

Eighteen CPU positive/negative schema tests passed. The failed run's real
aggregate also passed a design16 summary smoke test before deployment.

The unchanged-source mandatory rerun:

```text
./benchmark/dsa/run_b200_pipeline.sh --impl v2 --note design16-recheck
```

completed with exit code 0 and printed `DSA_PIPELINE_PASSED`. Its remote run
ID is `20260729T010329Z_design16-recheck_v2`; its local lightweight directory
is `.dsa_b200_results/20260729T010320Z_design16-recheck_v2`.

Release and trace used the same V2 source:

```text
SHA256 2dfdfa4f58ca4c30c1a4ad2280549ce5894d4cf139dff35936bd8e768436b9f1
```

The candidate trace used no patch. Runtime provenance proved `design16` in
both CTAs, with marker warp 12 and payloads matching CTA rank. Malformed,
cross-warp, and payload-mismatch range counts were zero.

## Correctness

| Pattern | dQ max abs | dKV max abs | dSink max abs |
|---|---:|---:|---:|
| dense | 0.0040819645 | 0.0096895695 | 0.0011278465 |
| lengths | 0.0119615793 | 0.0728607178 | 0.0372953415 |
| holes | 0.0045598745 | 0.0108621120 | 0.0016178787 |
| all_empty | 0 | 0 | 0 |

## Release performance

The mandatory same-run contract was H128/D512/S4096/topk2048, IKET disabled,
five warmups, and five timed repeats.

| Run | baseline | V2 | V2/baseline |
|---|---:|---:|---:|
| pre-summary-fix run | 8.147449 ms | 17.007341 ms | 2.087444x |
| mandatory passing rerun | 8.113094 ms | 17.008269 ms | 2.096397x |

The two V2 measurements differ by 0.000928 ms and average 17.007805 ms.
That is 3.38% faster than the replicated 20-warp dQ-TMA mean of
17.602090 ms, and 16.54% faster than the source-native pre-optimization
measurement of 20.378105 ms. V2 still takes about 2.10x baseline latency and
does **not** beat baseline.

## Selected trace means

| Stage | baseline | V2 |
|---|---:|---:|
| K/KV load | 3.236 us/tile | 15.522 us/tile |
| S+dP issue | 1.316 us/tile | 4.195 us/tile |
| P+dS T2R/math | 0.938 us/tile | 15.842 us/tile |
| dQ+dVdK issue | 2.944 us/tile | 2.414 us/tile |
| dKV T2R+atomic | 4.378 us/tile | 54.562 us/tile |
| dQ epilogue | 7.232 us/launch | 19.264 us/launch |

Candidate-only inclusive scopes are ROUTE_K 5.327 us/tile, two MAT_QDO rounds
10.525 us/tile, routes plus MAT_QDO 16.963 us/tile, and TAIL
33.376 us/launch. The IKET CTA lifetime envelope is 551.840 us.

Relative to the replicated 20-warp dQ-TMA trace, the dKV T2R+atomic software
scope is 8.29x larger, TAIL is 21.70% larger, and the CTA lifetime envelope
is 21.97% larger. This is consistent with each reducer thread carrying twice
the T2R/atomic chain. It is a serious single-cluster latency signal even
though the IKET-off S4096 grid throughput improves. These software spans
ignore overlap and do not represent atomic retire or kernel wall time; the
release result and trace regression are both retained without hiding either.

The mandatory lightweight output does not expose PTXAS spill/local-memory
statistics, so this report does not claim a measured zero-spill result.
