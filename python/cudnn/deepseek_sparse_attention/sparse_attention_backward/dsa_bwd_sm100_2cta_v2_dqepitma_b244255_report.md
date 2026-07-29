# V2 native dQ-TMA evidence (`b244255`)

This record covers the source-native V2 change that replaces the final scalar
dQ stores with an H128 x D128 BF16 shared-memory staging tile and a TMA S2G
store. The staging tile aliases the dead 32 KiB score-K allocation. The five
GEMMs remain CG2, and the score, dP, dQ, dV, and dK algorithms and ownership
rules are unchanged. No V0/V1 implementation, baseline implementation,
candidate routing, or candidate-side trace patch is used.

The mandatory command:

```text
./benchmark/dsa/run_b200_pipeline.sh --impl v2
```

completed with exit code 0 and printed `DSA_PIPELINE_PASSED`. The remote run ID
is `20260729T003827Z_v2_v2`; the local lightweight directory is
`.dsa_b200_results/20260729T003637Z_v2_v2`.

An unchanged-source replication also completed with exit code 0 and printed
`DSA_PIPELINE_PASSED`. Its remote run ID is
`20260729T004503Z_dqepitma-recheck_v2`; its local lightweight directory is
`.dsa_b200_results/20260729T004446Z_dqepitma-recheck_v2`.

The release and trace stages used the same V2 source:

```text
SHA256 c83e43aa8de6864c226516041c820bd4ecdb5b6e6b3c2224da79004872991eb3
```

Runtime V2 provenance matched both CTA ranks. Malformed, cross-warp, and
payload-mismatch range counts were zero.

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
| initial | 8.027200 ms | 17.603322 ms | 2.192959x |
| unchanged-source replication | 8.082649 ms | 17.600858 ms | 2.177610x |

V2 does **not** beat the baseline. The two V2 measurements differ by only
0.002464 ms and average 17.602090 ms. Against split-P/dS `9110e23`
(17.993697 ms), the initial run is 2.17% faster; against the latest exact
restored-source run (18.002324 ms), it is 2.22% faster. The unchanged-source
replication confirms that the release-throughput improvement is reproducible,
but neither run is a baseline winner.

## Selected trace means

| Stage | baseline | V2 |
|---|---:|---:|
| K/KV load | 3.270 us/tile | 12.837 us/tile |
| S+dP issue | 1.316 us/tile | 4.584 us/tile |
| P+dS T2R/math | 0.943 us/tile | 13.117 us/tile |
| dQ+dVdK issue | 2.924 us/tile | 2.183 us/tile |
| dKV T2R+atomic | 4.378 us/tile | 5.873 us/tile |
| dQ epilogue | 7.104 us/launch | 17.760 us/launch |

Candidate-only inclusive scopes include ROUTE_K at 5.246 us/tile, two
MAT_QDO rounds at 7.607 us/tile, routes plus MAT_QDO at 13.893 us/tile, and
TAIL at 27.264 us/launch.

The unchanged-source trace reproduced dQ epilogue at 17.760 us/launch. Its
ROUTE_K, MAT_QDO, routes-plus-MAT_QDO, TAIL, and CTA lifetime envelope were
5.230 us/tile, 7.530 us/tile, 13.789 us/tile, 27.424 us/launch, and
452.448 us, respectively. The first trace's CTA lifetime envelope was
453.056 us.

The dQ annotation now includes shared publication, TMA issue, and the
source-read completion wait. It is a software scope, not destination
visibility or pure store time. Its 17.760 us duration is 17.58% longer than
the split-P/dS trace's 15.104 us scalar-store scope; TAIL is 11.37% longer and
the IKET CTA lifetime envelope is 1.89% longer. Therefore the trace does not
support the original local-span-shortening hypothesis even though the
IKET-off release run improved. Both facts are retained; neither is hidden or
reinterpreted as proof that V2 beat baseline.
