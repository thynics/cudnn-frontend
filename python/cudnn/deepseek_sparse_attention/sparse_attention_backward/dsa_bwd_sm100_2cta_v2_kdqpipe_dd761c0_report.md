# V2 two-group K_dQ pipeline evidence (`dd761c0`)

This source-native V2 experiment issued the two K_dQ `cp.async` groups before
draining them. It used `wait_group(1)` to publish generation A early and
`wait_group(0)` to publish generation B. The all-CG2 math, ten-generation
FIFO, layouts, ownership, TMA path, and P/dS exchange were unchanged. It did
not route to or call baseline, V0, V1, or V3.

The mandatory command:

```text
./benchmark/dsa/run_b200_pipeline.sh --impl v2
```

completed with exit code 0 and printed `DSA_PIPELINE_PASSED`. The remote run ID
is `20260729T001904Z_v2_v2`; the local lightweight directory is
`.dsa_b200_results/20260729T001848Z_v2_v2`.

The release and trace stages used the same V2 source:

```text
commit dd761c017c124b16f6b120b6d7c36dd7ad2f84e9
SHA256 34d6dae8b36cc36f65790b985e4aa7c77f58045d2370e4ab65dcc7e849d868a0
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
| baseline | 8.031686 ms | 684.483664 |
| V2 K_dQ pipeline | 17.892128 ms | 307.261279 |

V2/baseline latency ratio: **2.227693x**. V2 does **not** beat the baseline.
The immediately preceding byte-identical restored V2 run measured baseline
8.036806 ms and candidate 17.999750 ms (ratio 2.239665x). The apparent
candidate change is about -0.60%, which is too small and internally
inconsistent with the trace to claim as a reliable optimization.

## Trace comparison with the restored V2

| Metric | restored V2 | `dd761c0` |
|---|---:|---:|
| ROUTE_K | 5.350 us/tile | 4.994 us/tile |
| MAT_QDO | 7.925 us/tile | 8.349 us/tile |
| Routes + MAT_QDO subtotal | 14.513 us/tile | 14.523 us/tile |
| CTA lifetime envelope | 442.880 us | 446.208 us |
| tile i=2 completion cadence | 12.000 us | 11.744 us |
| tile i=3 completion cadence | 12.160 us | 12.960 us |

`ROUTE_K` is an inclusive software scope, not a direct asynchronous completion
measurement. The reduced `ROUTE_K` span was offset by a longer downstream
`MAT_QDO` span; the combined supply envelope did not improve, and the capture
lifetime increased. These data do not support retaining the change as a
meaningful optimization.

## Decision

The experiment is correctness-valid, source-native, and compliant with the V2
data plane, but its wall-time hypothesis is not supported. It is recorded as a
failed/noise-scale experiment and reverted explicitly. No winner claim is made
from the single 17.892128 ms sample.
