# SM100 H128 Two-CTA DSA Backward PR Preparation

This is a working evidence checklist for the H128/D512 two-CTA specialization.
It is not an upstream PR description and should be updated only with results
produced from the final integrated source.

## 2026-08-29 strict B200 review outcome

**Status: not ready for an upstream PR under the no-precision-regression
policy.** The specialization has a clear, reproducible performance win, and
all final regression tests pass, but its per-query dSink atomic reduction has
measurable repeat jitter that the upstream hierarchical reduction does not.

Evidence source:

- Product/evidence commit: `cc7808b2d81d306d948c49c006ca9e4d6d73c1a0`.
- Final test-only commit: `85e08e97081e846da69810436b069bc9dfff68d8`.
- The test-only commit scales the active-OOB test inputs by `1/10`; it does
  not change the candidate kernel, interface, tolerance, or performance path.
- Candidate kernel SHA256: `3aaf47fc340036208ad1c986dc1c56e5a4454d1579a16791dc3a7914002974e2`.
- Baseline: current upstream `develop` at
  `606e16f9786ea7a13e0462c8a63edf0d7f72ae85`.
- Raw artifacts are retained outside the checkout under
  `/home/longcheng/cudnn-frontend-dsa-bwd-results-20260829/`.
- SHA256: precision `d9c79301...a079f`, Graph50/full
  `8eaebcd0...7be7`, eager60 `7e6e37a5...60ef`, cold200
  `a8334195...3cb8`.

Environment:

- NVIDIA B200, 148 SMs, UUID
  `GPU-4cd39877-8085-3aa4-ae91-ec13810fa7a7`, MIG disabled.
- Driver `595.58.03`; CUDA/NVCC `13.3`; PyTorch
  `2.13.0a0+9186a08b2c.nv26.07`; cuDNN backend `9.24.0`.
- CuTe DSL exactly `4.5.2`; active `libs-cu13` IR/runtime hashes were checked
  against the pinned wheel (`73b760...f6d8` and `deb32d...0a1`).
- The selected GPU UUID had no foreign compute process before or after each
  measured case.

The first 4.5.2 smoke exposed a flattening-only integration bug:
`from __future__ import annotations` stringified the local `@cute.struct`
`MemRange` fields. CuTe DSL 4.5.2 intentionally consumes concrete annotation
objects. Removing that future import restored the original AVO behavior; it
does not alter arithmetic or the two-CTA topology. This was not a backend bug.

### Performance versus current upstream

The same-process A/B uses canonical independent-row random indices, four
Williams-balanced treatment labels, raw samples, independent CUDA Graph
pools, 100,000-sample hierarchical bootstrap intervals, and a duplicate-arm
drift gate of +/-0.5%.

| Mode | Equal-weight geomean speedup | 95% CI | Duplicate-drift validity |
| --- | ---: | ---: | --- |
| Graph50, 20 windows | `1.16222x` | `[1.16208, 1.16235]` | valid, 16/16 cases |
| Eager hot, 60-window confirmation | `1.18395x` | `[1.18381, 1.18408]` | valid, 16/16 cases |
| Eager cold-L2, 200-window confirmation | `1.18204x` | `[1.18179, 1.18242]` | invalid: 1/16 duplicate CI too wide |

Every per-shape point estimate and 95% speedup interval is above `1.0x`.
Valid-mode per-shape speedups range from `1.03328x` to `1.43251x` for eager
hot and from `1.03667x` to `1.33329x` for Graph50. The gain is much larger
without `topk_length` (eager geomean `1.34370x`) than with a full-length
tensor (`1.04318x`). The cold-L2 point estimate is retained as exploratory
only: its one invalid case had a duplicate point ratio of `1.00438`, inside
the bound, but a noise-broadened CI of `[0.99958, 1.01337]`.

### Strict precision result

The audit covers three seeds, all four dispatched top-k widths, both
topk-length paths, and 50 unsynchronized repeats of each implementation. TF32
was disabled at process start and in PyTorch. All 24 cases pass the existing
`atol=rtol=5e-2` upstream correctness gate, and dQ is bitwise repeatable.

| Quantity | Candidate versus baseline |
| --- | --- |
| dQ RMS / rel-L2 | better in 24/24 cases; max-abs is worse in 4/24 cases |
| dKV | identical max-abs; RMS/rel-L2 split 12 better / 12 worse; worst repeat jitter is higher (`0.001953125` vs `0.0009765625` max-abs) |
| dSink, supplied-out/LSE contract oracle | better in 24/24; worst rel-L2 `6.80e-7` vs baseline `2.15e-3` |
| dSink, mathematical FP32 oracle | RMS/rel-L2 better in 24/24; max-abs worse in 4/24 |
| dSink repeat jitter | nonzero in 24/24 candidate cases, zero in 24/24 baseline cases; candidate worst max-abs `2.24e-8` |

At the real performance sequence lengths, candidate dSink is also much closer
to the supplied BF16-out/FP32-LSE contract oracle: the worst single-call
max-abs error is `4.02e-7` versus baseline `3.32e-4`. This improvement comes
from promoting O and dO before the FP32 FMA. It does not remove the strict
repeatability regression caused by changing the cross-query reduction tree.

The upstream kernel first reduces each 256-query window in a fixed FP32 warp
tree and issues 16/32 atomics per head for Q=4096/8192. The candidate issues
4096/8192 per-query FP32 atomics per head. The dtype is unchanged, but the
addition depth and ordering are not; therefore the current candidate cannot
be described as having no precision regression.

Recommended repair: keep the main CG2 kernel and its exact `(2,1,1)` cluster,
write FP32 `sum_OdO` and `scaled_LSE` partials, then reuse the upstream
256-query `sum_dSink` reduction. This adds one launch and `8*Q*H` bytes of
workspace (4 MiB at Q=4096, 8 MiB at Q=8192) while restoring the established
dSink reduction topology. The complete public-call performance comparison
must then be rerun.

### Correctness and compatibility

- Exact H128 gate: `13 passed`, zero skipped/deselected.
- Whole DSA backward L0/L1 file: `39 passed`, one legitimate SM90-only skip.
- A real H128/top-k513 call routed to and executed `generic_m64` successfully.
- Two independent CUDA streams, each with 20 back-to-back candidate calls,
  passed; dQ matched bitwise and dKV/dSink stayed within `1e-4` rel-L2 of the
  serial controls.
- Direct poisoned outputs, 50-call bursts, and CUDA Graph checks before and
  after timing passed in all 16 performance cases.

## Source and target

- Target repository base: `NVIDIA/cudnn-frontend` `develop` at `606e16f9`.
- Preparation branch: `perf/dsa-bwd-sm100-2cta-g56`.
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
- main kernel plus FP32-dKV conversion launch.

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
