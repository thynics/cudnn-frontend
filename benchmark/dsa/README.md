# DSA Sparse Attention Backward Benchmark

Microbenchmark for the DeepSeek Sparse Attention (DSA) backward kernel in the
cuDNN Frontend CuTe DSL package, driven through the public
`cudnn.DSA.sparse_attention_backward_wrapper` API. The wrapper dispatches to
the Hopper (SM90) or Blackwell (SM100) implementation based on the active
CUDA device.

## What is measured

Inputs are flat FlashMLA-shaped tensors: `q (S_q, H, d_qk)`, a shared
`kv (S_kv, d_qk)` buffer (K = V), and per-query global top-k indices
`topk_idxs (S_q, topk)` with unique random indices per query row. The forward
`out`/`lse` consumed by the backward kernel come from a chunked PyTorch
reference, since the production forward (FlashMLA) is out of scope for this
repository.

Each timed iteration is one full wrapper call — gradient-buffer zeroing,
workspace allocation, and the preprocess/backward/convert kernels — i.e. the
cost a training step pays per backward invocation. Timing uses a single
CUDA-event window around `--repeat` iterations and reports the average.

Reported TFLOPS use the 5-matmul model of the backward pass (recompute S, dV,
dP, dQ, dK):

```
FLOPs = 2 * S_q * H * topk * (3 * d_qk + 2 * d_v)
```

## Requirements

- Hopper (SM90) or Blackwell (SM100) GPU
- PyTorch with CUDA support
- `pip install nvidia-cudnn-frontend[cutedsl]` (or a development install of
  this repository's `python/` package with the `cutedsl` extra)

## How to run

Default sweep (`seqlens 4096,8192 x topks 128,512,1024,2048`, bf16,
`d_qk = d_v = 512`, 64 heads):

```bash
python benchmark_dsa_sparse_attention_backward.py
```

Custom shapes and CSV output:

```bash
python benchmark_dsa_sparse_attention_backward.py --seqlens 4096,8192,16384 --topks 512,2048 --csv results.csv
python benchmark_dsa_sparse_attention_backward.py --head-dim 576   # 512 value dims + 64 RoPE dims
python benchmark_dsa_sparse_attention_backward.py --nheads 16 --head-dim 576  # SM100 H16/D576 M128 backend
python benchmark_dsa_sparse_attention_backward.py --nheads 128 --head-dim 512 # B200 H128/D512 two-CTA backend
```

Options:

- `--seqlens` — comma-separated total query lengths; `seqlen_kv = seqlen_q`
  for every config. Configs with `topk > seqlen_kv` are skipped.
- `--topks` — comma-separated top-k values.
- `--nheads` — number of query heads (default 64).
- `--head-dim` — QK head dim, `512` or `576`; `head_dim_v` is derived (512).
- `--dtype` — `bfloat16` (default) or `float16`.
- `--no-attn-sink` — disable the attention sink (passes `-inf` sink logits).
- `--no-topk-length` — omit the `topk_length` tensor. Kernels with and
  without `topk_length` are different compiled variants; the default
  benchmarks the `topk_length` variant with every row at the full top-k
  count.
- `--warmup` / `--repeat` — iterations per config (defaults 10 / 50; the
  first warmup iteration also triggers kernel compilation).
- `--csv` — write results to a CSV file.

## Results

### B200

Generated on an NVIDIA B200 with the default
sweep settings (`nheads=64`, `d_qk = d_v = 512`, bf16, attention sink and
`topk_length` enabled, `warmup=10`, `repeat=50`), using `torch 2.12.1`,
`nvidia-cutlass-dsl 4.5.2`, and `nvidia-cudnn-frontend` built from this
repository.

| seqlen_q | seqlen_kv | topk | BWD ms | BWD TFLOPS |
|---------:|----------:|-----:|-------:|-----------:|
|     4096 |      4096 |  128 |  0.563 |     305.09 |
|     4096 |      4096 |  512 |  1.243 |     552.87 |
|     4096 |      4096 | 1024 |  2.198 |     625.35 |
|     4096 |      4096 | 2048 |  4.168 |     659.46 |
|     8192 |      8192 |  128 |  1.094 |     313.97 |
|     8192 |      8192 |  512 |  2.489 |     552.16 |
|     8192 |      8192 | 1024 |  4.538 |     605.76 |
|     8192 |      8192 | 2048 |  8.562 |     642.06 |

## Profiling

`profile` mode runs a single warmed-up backward call (using the first value
of `--seqlens` and the last value of `--topks`) wrapped in
`cudaProfilerStart/Stop` and an NVTX range, so nsys/ncu capture only the
kernels of interest:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o dsa_bwd \
  python benchmark_dsa_sparse_attention_backward.py profile --seqlens 8192 --topks 2048

ncu --profile-from-start off -o dsa_bwd \
  python benchmark_dsa_sparse_attention_backward.py profile --seqlens 8192 --topks 2048
```

## H128 two-CTA candidate A/B

`benchmark_dsa_sparse_attention_backward_ab.py` compares the current H128
two-CTA specialization with the canonical generic implementation loaded from
the pinned upstream `606e16f9` interface blob. Both run through the public
wrapper in one process with identical inputs. Four independent treatment arms
use Williams-order balancing, private CUDA Graph pools, duplicate drift
controls, raw samples, and paired bootstrap confidence intervals.

The runner is fail-closed on a clean 148-SM B200, CuTe DSL 4.5.2, MIG disabled,
and no foreign process on the selected GPU UUID. Inputs use the benchmark's
canonical seeded per-row random ordering. Write each result to a new path
outside the checkout so the clean-tree and immutable-evidence audits remain
valid:

```bash
python benchmark_dsa_sparse_attention_backward_ab.py \
  --output /path/to/results/dsa_bwd_h128_2cta_ab.json
```

Use reduced settings only as a smoke test; they are not publication evidence:

```bash
python benchmark_dsa_sparse_attention_backward_ab.py \
  --seqlens 4096 --topks 128 --length-modes full \
  --modes eager_hot --graph-calls 2 \
  --warmup-windows 0 --measured-windows 1 \
  --correctness-calls 2 --bootstrap-samples 100 \
  --output /path/to/results/smoke.json
```

Run the separate strict precision audit before making a performance claim. It
uses analytical FP32 dQ/dKV references and a supplied-out/LSE FP64 dSink
reference with TF32 disabled, three fixed seeds, both top-k-length paths, all
four specialized widths, and 50 unsynchronized repeats per implementation. It
also reports baseline/baseline and candidate/candidate jitter rather than
treating the upstream `5e-2` tolerance as proof of equal precision:

```bash
NVIDIA_TF32_OVERRIDE=0 python check_dsa_sparse_attention_backward_precision_ab.py \
  --output /path/to/results/dsa_bwd_h128_2cta_precision.json
```

### Hierarchical dSink repair

The H128 candidate now publishes two FP32 statistics planes from its genuine
two-CTA main kernel and reduces dSink in a separate 256-query FP32 warp tree.
This replaces the earlier per-query atomic path while retaining the optimized
FP32 O-dot-dO calculation. On B200 with CuTe DSL 4.5.2, the strict 24-case
precision audit reports `precision_claim_eligible=true`. The valid full-matrix
results versus upstream `606e16f9` are `1.15158x` Graph50 (95% CI
`[1.15146, 1.15170]`) and `1.18146x` eager-hot (95% CI
`[1.18135, 1.18157]`). See `SM100_H128_2CTA_PR_PREP.md` for the complete
protocol and artifact hashes.

#### Final focused test_harness run

```text
===== cudnn-frontend conftest.py ====
cuDNN Frontend Version: 1.28.0
cuDNN Frontend Path: /home/scratch.longcheng_gpu/cudnn-frontend-dsa-bwd-2cta-g56/repo/python/cudnn/__init__.py
cuDNN Backend Version: 92400
PyTorch Version: 2.13.0a0+9186a08b2c.nv26.07
PyTorch Path: /usr/local/lib/python3.12/dist-packages/torch/__init__.py
PyTorch GPU Name: NVIDIA B200
PyTorch SM Arch Version: (10, 0)
PyTorch CUDA Version: 13.3
PyTorch cuDNN Version: 92400
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/scratch.longcheng_gpu/cudnn-frontend-dsa-bwd-2cta-g56/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/scratch.longcheng_gpu/cudnn-frontend-dsa-bwd-2cta-g56/repo/test/python
configfile: pytest.ini
plugins: xdist-3.8.0, typeguard-4.5.2, anyio-4.14.2
collecting ... collected 1 item

test/python/fe_api/dsa/test_DSA_sparse_attention_backward.py::test_DSA_sparse_attention_backward_sm100_h128_hierarchical_dsink_repeatability PASSED [100%]

=============================== warnings summary ===============================
../../../../usr/local/lib/python3.12/dist-packages/torch/jit/_script.py:1488
../../../../usr/local/lib/python3.12/dist-packages/torch/jit/_script.py:1488
  /usr/local/lib/python3.12/dist-packages/torch/jit/_script.py:1488: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
==================================== PASSES ====================================
PASSED test/python/fe_api/dsa/test_DSA_sparse_attention_backward.py::test_DSA_sparse_attention_backward_sm100_h128_hierarchical_dsink_repeatability
======================== 1 passed, 2 warnings in 16.50s ========================
```
