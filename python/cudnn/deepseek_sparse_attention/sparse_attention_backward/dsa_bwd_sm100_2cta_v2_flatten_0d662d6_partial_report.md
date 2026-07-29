# V2 `0d662d6` partial B200 evidence

This is a deliberately incomplete evidence record. It must not be cited as a
full `DSA_PIPELINE_PASSED` run.

## Provenance

- Revision: `0d662d6e8da16c8f5070a57a7fba88c778e478f3`
- V2 source SHA256:
  `328897ea1f8c273daf43bf925951e61d87851c78fa271c4d9ab44084a1e25654`
- Active class: `FlashAttentionDSABackwardSm100TwoCTAV2`
- Native provenance event: `V2_NATIVE_PROVENANCE`
- Candidate instrumentation patch: none
- Production source modified during tracing: no
- GPU: NVIDIA B200

## Release correctness

The CuTe 4.5 release path compiled, and the repository validation accepted all
four required patterns.

| Pattern | dQ max abs | dKV max abs | dSink max abs |
|---|---:|---:|---:|
| dense | 0.0040819645 | 0.0096895695 | 0.0011278465 |
| lengths | 0.0119615793 | 0.0728607178 | 0.0372953415 |
| holes | 0.0045598745 | 0.0108621120 | 0.0016178787 |
| all-empty | 0 | 0 | 0 |

## Release performance

Matched IKET-off shape: `S_q=4096`, `S_kv=4096`, `H=128`, `Dqk=Dv=512`,
`topk=topk_length=2048`, BF16, warmup 5, repeat 5.

| Implementation | Latency (ms) | TFLOP/s |
|---|---:|---:|
| baseline | 8.122662 | 676.817267 |
| native V2 | 14.142822 | 388.717191 |

V2/baseline is `1.741156x`: V2 is about 74.1% slower. This does **not**
satisfy the performance contract.

## Trace status

- Trace toolchain: CuTe 4.6.1 with IKET 0.7.12.
- The last observed output was the baseline Perfetto dump.
- The user requested that the stalled run be stopped; the client exited 130.
- The run produced no lightweight result and no persisted trace-stage status.
- It is therefore not possible to prove from this run whether candidate
  capture had begun.
- There is no usable candidate trace, two-trace table, or aggregate timeline
  for this revision.
- The full pipeline did not pass.

The B200 pipeline lock was released. Allocation job `3327070` was then
explicitly cancelled, disappeared from `squeue`, and was recorded by `sacct`
as cancelled.

## Engineering-integrity statement

Historical attempts that routed V2 to an old implementation, delegated to
baseline, or special-cased tests and benchmarks to manufacture "V2
performance" are severe engineering-integrity violations. Those results are
invalid. No such result is used in this report.
