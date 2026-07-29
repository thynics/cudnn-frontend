# V2 paired-quadrant-TMA failure evidence (`aed6812`)

This record covers a source-native V2 experiment that gave the two h0/h1
quadrant refills independent raw TMA barriers and issued both copies before
waiting. It did not route to V0/V1/baseline or another implementation, and it
did not alter the all-CG2 math/data ownership design.

The mandatory command:

```text
./benchmark/dsa/run_b200_pipeline.sh --impl v2
```

completed with exit code 0 and printed `DSA_PIPELINE_PASSED`. The remote run ID
is `20260729T000456Z_v2_v2`; the local lightweight directory is
`.dsa_b200_results/20260729T000438Z_v2_v2`.

The release and trace stages used the same V2 source:

```text
commit aed6812085c35249f146039603bae40293b863cf
SHA256 c96db7bb592f2f4f7122303e7d0a9aad900987289d846502742c79a416ad54ef
```

No candidate-side trace patch was used. Runtime V2 provenance matched both CTA
ranks. Malformed, cross-warp, and payload-mismatch range counts were all zero.

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

| Implementation | Latency | TFLOP/s |
|---|---:|---:|
| baseline | 8.157114 ms | 673.958753 |
| V2 paired-TMA | 18.441197 ms | 298.112865 |

V2/baseline latency ratio: **2.260750x**. V2 does **not** beat the baseline.
The preceding retained V2 (`9110e23`) measured 17.993697 ms, so this experiment
is about **2.49% slower**. This is a regression, not an optimization result.

## Selected trace means

| Stage | baseline | V2 paired-TMA |
|---|---:|---:|
| K/KV load | 3.189 us/tile | 13.373 us/tile |
| S+dP issue | 1.270 us/tile | 4.578 us/tile |
| P+dS T2R/math | 0.842 us/tile | 13.814 us/tile |
| dQ+dVdK issue | 2.877 us/tile | 1.647 us/tile |
| dKV T2R+atomic | 4.278 us/tile | 5.694 us/tile |
| dQ epilogue | 7.456 us/launch | 15.136 us/launch |

Candidate-only inclusive scopes include ROUTE_K at 5.356 us/tile and two
MAT_QDO rounds at 8.600 us/tile. `MAT_QDO` includes acquire, TMA issue, raw
completion wait, and typed publish; it is not a pure TMA latency and cannot be
used as proof that hardware overlap occurred.

The candidate CTA lifetime envelope increased from 444.640 us in `9110e23` to
468.896 us. Annotated tile-completion cadence also worsened:

| Tile | `9110e23` | `aed6812` |
|---|---:|---:|
| i=2 | 11.936 us | 12.064 us |
| i=3 | 12.448 us | 13.536 us |

## Decision

The experiment is correctness-valid and source-native, but the release result
and trace both reject its performance hypothesis. Keeping it as the active V2
would be unjustified. It is therefore recorded as a failed experiment and
reverted explicitly; no winner claim is made from these data.
