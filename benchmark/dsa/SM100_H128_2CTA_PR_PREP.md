# SM100 H128 Two-CTA DSA Backward PR Preparation

This is a working evidence checklist for the H128/D512 two-CTA specialization.
It is not an upstream PR description and should be updated only with results
produced from the final integrated source.

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
