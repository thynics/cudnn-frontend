# SM100 H128 Two-CTA DSA Backward PR Preparation

This is a working evidence checklist for the H128/D512 two-CTA specialization.
It is not an upstream PR description and should be updated only with results
produced from the final integrated source.

## 2026-08-29 hierarchical dSink repair outcome

**Status: precision and performance gates pass; ready for final maintainer
review in our fork.** No NVIDIA PR has been created. The main computation
remains genuine two-CTA/CG2. dSink now uses the upstream-equivalent 256-query
FP32 reduction tree instead of per-query atomics.

Evidence source:

- Product/evidence commit: `b6d003bb7fb41c0b7ed694743e48fa4f61c82536`.
- Repair branch: `perf/dsa-bwd-sm100-2cta-hier-dsink`.
- Candidate kernel SHA256: `19fbbd6b32a9b645c98f019be62a5696be1983938a4bd074cf56b648183d7d58`.
- Candidate interface SHA256: `8473f9b021ba50a9c80be9bb80c6ff21bcb90fbf7ea2ca78e1f992e4b22981e9`.
- Baseline: current upstream `develop` at
  `606e16f9786ea7a13e0462c8a63edf0d7f72ae85`.
- Raw artifacts are retained outside the checkout under
  `/home/longcheng/cudnn-frontend-dsa-bwd-hier-results-20260829/`.
- SHA256: precision `63297000...ff07`, Graph50/full
  `05df7b77...e659`, eager-hot200 `3bcea921...79f`, final focused
  gate `1d3b32c6...c39c`.

Environment:

- NVIDIA B200, 148 SMs, UUID
  `GPU-bede695e-f6aa-70af-f9e6-ae482b31a8b4`, MIG disabled.
- Driver `610.57.04`; CUDA/NVCC `13.3`; PyTorch
  `2.13.0a0+9186a08b2c.nv26.07`; cuDNN backend `9.24.0`.
- CuTe DSL exactly `4.5.2`; active `libs-cu13` IR/runtime hashes were checked
  against the pinned wheel (`73b760...f6d8` and `deb32d...0a1`).
- The selected GPU UUID had no foreign compute process before or after each
  measured case.

The original 4.5.2 smoke exposed a flattening-only integration bug:
`from __future__ import annotations` stringified the local `@cute.struct`
`MemRange` fields. CuTe DSL 4.5.2 intentionally consumes concrete annotation
objects. Removing that future import restored the original AVO behavior; it
does not alter arithmetic or the two-CTA topology. This was not a backend bug.
The repaired run compiled and executed on the same pinned 4.5.2 backend.

### Performance versus current upstream

The same-process A/B uses canonical independent-row random indices, four
Williams-balanced treatment labels, raw samples, independent CUDA Graph
pools, 100,000-sample hierarchical bootstrap intervals, and a duplicate-arm
drift gate of +/-0.5%.

| Mode | Equal-weight geomean speedup | 95% CI | Duplicate-drift validity |
| --- | ---: | ---: | --- |
| Graph50, 20 windows | `1.15158x` | `[1.15146, 1.15170]` | valid, 16/16 cases |
| Eager hot, 200-window confirmation | `1.18146x` | `[1.18135, 1.18157]` | valid, 16/16 cases |
| Eager cold-L2, 20 windows | `1.17742x` | `[1.17712, 1.17771]` | invalid: 2/16 duplicate CIs too wide |

Every per-shape point estimate and 95% speedup interval is above `1.0x`.
Valid-mode per-shape speedups range from `1.03261x` to `1.43897x` for eager
hot and from `1.03307x` to `1.30955x` for Graph50. The gain remains much
larger without `topk_length` (eager geomean `1.34164x`) than with a
full-length tensor (`1.04042x`). Cold-L2 remains exploratory because its
duplicate-arm validity gate did not pass.

Relative to the rejected per-query-atomic candidate, the repair retains
`99.79%` of eager speedup and `99.08%` of Graph50 speedup. The tested launch
order swap changed the two critical-case geomean by less than `0.01%`; the
historical `main -> dKV convert -> dSink reduce` order was retained.

### Strict precision result

The audit covers three seeds, all four dispatched top-k widths, both
topk-length paths, and 50 unsynchronized repeats of each implementation. TF32
was disabled at process start and in PyTorch. The supplied-out/LSE dSink
contract is evaluated in FP64. All 24 cases pass the hard precision gate and
the existing `atol=rtol=5e-2` sanity gate.

| Quantity | Candidate versus baseline |
| --- | --- |
| dQ | bitwise stable; worst rel-L2 `2.2271e-3` vs baseline `2.2311e-3` |
| dKV | same FP32-atomic/BF16-output path; worst rel-L2 `2.402776e-3` vs baseline `2.402836e-3` |
| dSink, supplied-out/LSE FP64 contract | better in 24/24; worst rel-L2 `5.50e-7` vs baseline `2.153e-3` |
| dSink repeatability at Q=257 | bitwise stable in 24/24 cases; zero hard jitter regressions |
| Hard precision verdict | `complete`, `precision_claim_eligible=true` |

At Q=4096/8192, 50 interleaved repeats show the repaired candidate inside the
baseline dSink jitter envelope. For top-k128, candidate spans are
`4.47e-8/5.96e-8` versus baseline `4.47e-8/7.45e-8`; for top-k2048 both are
about `1e-9`. Candidate FP64 max error is `3.8e-8--5.8e-8` for top-k128,
versus baseline `1.6e-4--4.5e-4`.

The repair keeps candidate FP32 O*dO, publishes FP32 `sum_OdO` and
`scaled_LSE`, and restores the upstream 256-query/32-thread warp tree. It
issues 16/32 dSink atomics per head at Q=4096/8192, adds one launch, and uses
`8*Q*H` workspace bytes (4/8 MiB). The core two-CTA computation is unchanged.

### Correctness and compatibility

- Exact H128 gate: `13 passed`, zero skipped/deselected.
- Hierarchical dSink FP64/tail/repeatability regression: `1 passed`.
- Whole DSA backward L0/L1 file: `40 passed`, one legitimate SM90-only skip.
- A real H128/top-k513 call routed to and executed `generic_m64` successfully.
- Two independent CUDA streams, each with 20 back-to-back candidate calls,
  passed; dQ matched bitwise, dSink matched exactly, and dKV stayed within
  `1e-4` rel-L2 of the serial controls.
- Direct poisoned outputs, 50-call bursts, and CUDA Graph checks before and
  after timing passed in all 16 performance cases.

## Source and target

- Target repository base: `NVIDIA/cudnn-frontend` `develop` at `606e16f9`.
- Preparation branch: `perf/dsa-bwd-sm100-2cta-hier-dsink`.
- Source repository snapshot: `avo-sparse-attention` at `409b900a`.
- Measured AVO source snapshot: `90cf2666`.
- Candidate lineage recorded by the source repository: `cf63f86d`.
- G56 register setting: `GATHER_SETMAXREG = 56`; this was an uncommitted
  one-line source-tree change and is preserved explicitly in the flattened
  specialization.

The historical AVO measurements compare against an older two-CTA Variant B,
not the current cuDNN Frontend kernel. They are provenance only and must not be
used as the upstream performance claim.

## Dispatch and compatibility contract

The specialization is selected only when all of the following are true:

- device capability is exactly `(10, 0)` (B200);
- inputs are BF16;
- `H = 128` and `D_qk = D_v = 512`;
- top-k width is one of `128`, `512`, `1024`, or `2048`.

FP16, D576, other head counts, other top-k widths, and SM103+ retain the
existing generic or H16 implementation. The public API and returned
`dq`/`dkv`/`d_sink` contract do not change.

## Numerical precision contract

The final review must show that both the baseline and candidate use:

| Quantity | Required candidate path |
| --- | --- |
| score, dP, dQ, dV, dK MMA accumulators | FP32 |
| softmax and dS arithmetic/reductions | FP32 before the established BF16 MMA-operand conversion |
| P and dS tensor-core operand storage | BF16, at the same boundary as the source and baseline |
| dKV workspace and global atomics | FP32 |
| dSink arithmetic/reduction/atomics | FP32 |
| public dQ/dKV storage | one final BF16 rounding |

No TF32, FP16 accumulation/atomics, earlier downcast, dropped term, truncated
reduction, or additional approximate math relative to the reviewed source is
permitted. Existing fast `exp2` and P/dS operand conversion sites must be
disclosed rather than described as full-FP32 end-to-end. Active positive
out-of-range top-k indices must be treated as inactive in the score gather,
K-dQ gather, and dKV scatter paths.

## Required correctness evidence

- Focused file and full DSA L0/L1 pytest results.
- Candidate widths: `128`, `512`, `1024`, `2048`, with and without
  `topk_length`.
- Length boundaries: negative, zero, `1`, `63`, `64`, `65`, `127`, `128`,
  `129`, `topk - 1`, `topk`, and values above the width.
- Dense indices, sentinel holes, active and ignored positive OOB indices,
  all-empty rows, strong/disabled sink, and non-default softmax scale.
- Poisoned caller outputs and exact-zero empty rows.
- Non-default stream, two-stream concurrency, 50-100 unsynchronized repeated
  calls, and CUDA Graph capture/replay.
- `compute-sanitizer` memcheck, racecheck, initcheck, and synccheck on a shape
  with at least two pipeline iterations.
- Per-output max-absolute error and RMS error against the eager FP32 reference,
  plus candidate-vs-baseline and baseline-vs-baseline FP32-atomic jitter.

## Required performance evidence

Baseline is the current canonical kernel at `606e16f9`, measured on the same
B200 and software environment as the candidate.

Primary matrix:

- `S_q = S_kv` in `{4096, 8192}`;
- BF16, `H = 128`, `D_qk = D_v = 512`;
- top-k in `{128, 512, 1024, 2048}`;
- `topk_length` present and absent.

Use identical inputs and independent outputs/workspaces. Run interleaved A/B
windows with reversed order and a duplicate-baseline drift control. Preserve
all raw samples and report median, MAD, p5/p95, paired-bootstrap 95% CI, every
shape, and the equal-weight geometric mean. Report both the complete public
backward call and the main-kernel decomposition.

Provisional acceptance gate:

- geometric-mean speedup at least `1.01x`;
- paired 95% CI lower bound above `1.00x`;
- no key shape regresses by more than 2%;
- duplicate-baseline drift stays within 0.5%.

## Resource and topology evidence

Record old/new grid, block, cluster, registers/thread, stack/local memory,
dynamic SMEM, TMEM columns, workspace bytes, occupancy, and launch count. The
candidate is expected to show:

- `cluster = (2, 1, 1)` and `grid.x = 2 * S_q`;
- 640 threads per CTA;
- five main tensor-core products using `CtaGroup.TWO`;
- 512 TMEM columns and at most 232,448 bytes SMEM per CTA;
- main kernel plus FP32-dKV conversion and hierarchical FP32-dSink launches;
- dSink stats workspace is 4 MiB at Q=4096 and 8 MiB at Q=8192.

## Environment record

Archive GPU model/UUID/SM count, MIG state, driver, locked clock, power and
temperature, CUDA/NVCC, Python, PyTorch, cuda-python, cuDNN, and CuTe DSL.
The primary evidence environment is CuTe DSL 4.5.2. Also run compile and
correctness compatibility checks with the current supported CuTe DSL because
the package dependency is `nvidia-cutlass-dsl>=4.5.0`.

## Upstream description outline

1. Summary and affected area.
2. Why a B200 H128 specialization is needed.
3. Two-CTA topology, ownership, pipelines, and buffer lifetime.
4. Precision and numerical-equivalence proof.
5. Shape-gated dispatch and fallback/API compatibility.
6. Per-shape performance table and measurement protocol.
7. Resource/topology comparison and causal evidence.
8. Correctness, sanitizer, stream, CUDA Graph, and regression-test results.
9. Source provenance, license, and exact reproduction commands.

The closest historical templates are:

- <https://github.com/NVIDIA/cudnn-frontend/pull/684>
- <https://github.com/NVIDIA/cudnn-frontend/pull/318>
- <https://github.com/NVIDIA/cudnn-frontend/pull/664>
- <https://github.com/NVIDIA/cudnn-frontend/pull/395>
- <https://github.com/NVIDIA/cudnn-frontend/pull/396>
