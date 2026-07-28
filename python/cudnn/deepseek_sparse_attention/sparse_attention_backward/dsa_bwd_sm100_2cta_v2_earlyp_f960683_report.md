# V2 early-P evidence (`f960683`)

This record covers the source-native V2 change that publishes P before waiting
for dP. The mandatory command was:

```text
./benchmark/dsa/run_b200_pipeline.sh --impl v2
```

The run completed with exit code 0 and printed `DSA_PIPELINE_PASSED`.
Its run ID is `20260728T233922Z_v2_v2`; the local lightweight evidence
directory is `.dsa_b200_results/20260728T233912Z_v2_v2`.

The pipeline checkout was branch tip `b80dd5b`, which only added a separate V3
file after `f960683`. The V2 source was byte-identical to `f960683`:

```text
SHA256 9c019766fd8ebee2be589a06a42bced9289babdd7108e9d86a42fa38cb183b8e
```

## Correctness

All mandatory patterns passed.

| Pattern | dQ max abs | dKV max abs | dSink max abs |
|---|---:|---:|---:|
| dense | 0.0040819645 | 0.0096895695 | 0.0011278465 |
| lengths | 0.0119615793 | 0.0728607178 | 0.0372953415 |
| holes | 0.0045598745 | 0.0108621120 | 0.0016178787 |
| all_empty | 0 | 0 | 0 |

## Release performance

The benchmark used the mandatory same-run H128/D512/S4096/topk2048 contract
with IKET disabled, five warmups, and five timed repeats.

| Implementation | Latency | TFLOP/s |
|---|---:|---:|
| baseline | 8.017568 ms | 685.688977 |
| V2 | 18.027232 ms | 304.958522 |

V2/baseline latency ratio: **2.248466x**. This version does **not** beat the
baseline.

## Trace integrity and interpretation

The candidate trace came directly from
`dsa_bwd_sm100_2cta_v2.py`; no candidate-side instrumentation patch was used.
The runtime `V2_NATIVE_PROVENANCE` markers matched CTA rank, and malformed,
cross-warp, and payload-mismatch range counts were all zero.

The split P/dS math schema was validated as two payload-distinct logical phases
per tile and sixteen raw ranges per tile. The reported formula is
`T2R_S(i) + T2R_dP(i) + 2×MATH_PD(i,phase)`. Phase 0 includes the P-stage
producer acquire, so it is an inclusive software scope, not pure math or an
asynchronous-completion measurement.

Selected trace means:

| Stage | baseline | V2 |
|---|---:|---:|
| K/KV load | 3.319 μs/tile | 13.560 μs/tile |
| S+dP issue | 1.445 μs/tile | 4.677 μs/tile |
| P+dS T2R/math | 1.144 μs/tile | 15.232 μs/tile |
| dQ+dVdK issue | 3.042 μs/tile | 2.232 μs/tile |
| dKV T2R+atomic | 4.611 μs/tile | 5.663 μs/tile |
| dQ epilogue | 7.456 μs/launch | 15.520 μs/launch |

The early-P edit therefore preserved correctness but did not materially improve
wall time. The inclusive P phase exposes blocking on the still-combined P/dS
stage; this is evidence for separating their lifetimes, not evidence of a
performance win.
