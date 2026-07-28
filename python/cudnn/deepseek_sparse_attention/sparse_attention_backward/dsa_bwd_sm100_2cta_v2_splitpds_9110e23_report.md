# V2 split-P/dS evidence (`9110e23`)

This record covers the source-native V2 change that gives P and dS independent
single-stage full/empty lifetimes. P is released after the final dV pass; dS
is released after the final dK pass. The all-CG2 data plane, two rounds with
eight K64 dV/dK issues, sender-side stmatrix stores, and one 4 KiB P plus one
4 KiB dS DSM send per CTA/tile are unchanged.

The mandatory command:

```text
./benchmark/dsa/run_b200_pipeline.sh --impl v2
```

completed with exit code 0 and printed `DSA_PIPELINE_PASSED`. The remote run ID
is `20260728T235149Z_v2_v2`; the local lightweight directory is
`.dsa_b200_results/20260728T235136Z_v2_v2`.

The release and trace stages used the same V2 source:

```text
SHA256 d5c9a9490308ca3d4079fd09472fb951983c137d899ad0fc1805eced8b086621
```

No candidate-side trace patch was used. Runtime V2 provenance matched both CTA
ranks, and malformed, cross-warp, and payload-mismatch range counts were zero.

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
| baseline | 8.127142 ms | 676.444181 |
| V2 | 17.993697 ms | 305.526889 |

V2/baseline latency ratio: **2.214025x**. V2 does **not** beat the baseline.
Compared with the preceding early-P V2 measurement (18.027232 ms), the change
is only about 0.19%; that is noise-scale evidence, not a performance win.

## Selected trace means

The P/dS row is
`T2R_S(i) + T2R_dP(i) + 2×MATH_PD(i,phase)`. Both phases are inclusive
software scopes: phase 0 includes the P pipeline acquire/commit and phase 1
includes the dS pipeline acquire/commit. Neither is pure math or asynchronous
completion time.

| Stage | baseline | V2 |
|---|---:|---:|
| K/KV load | 3.264 μs/tile | 12.660 μs/tile |
| S+dP issue | 1.380 μs/tile | 4.523 μs/tile |
| P+dS T2R/math | 1.067 μs/tile | 13.202 μs/tile |
| dQ+dVdK issue | 2.980 μs/tile | 1.625 μs/tile |
| dKV T2R+atomic | 4.498 μs/tile | 5.661 μs/tile |
| dQ epilogue | 7.456 μs/launch | 15.104 μs/launch |

Candidate-only inclusive scopes include ROUTE_K at 5.335 μs/tile, two
MAT_QDO rounds at 8.010 μs/tile, and TAIL at 24.480 μs/launch. These identify
the serialized operand-supply path as the next bottleneck; the component sums
can overlap and are not kernel wall time.
