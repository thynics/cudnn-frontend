# DSA Backward SM100 2-CTA Variants

Run the following commands from the repository root. Both checks require an
SM100 GPU and the cuDNN Frontend CuTe DSL dependencies.

The default `implementation="sm100"` remains the baseline. The two final
implementations are explicit opt-ins:

- `implementation="sm100_2_cta_A"` uses `dsa_bwd_sm100_2_cta_A.py`.
- `implementation="sm100_2_cta_B"` uses `dsa_bwd_sm100_2_cta_B.py`.

There is intentionally no `sm100_2_cta` alias.

## Correctness

```bash
(
  cd test/python
  PYTHONPATH=../../python:. python -m pytest -q \
    fe_api/dsa/test_DSA_sparse_attention_backward_sm100_2_cta.py
)
```

This runs the same five repository cases through both explicit two-CTA
selectors and checks their gradients against an independent FP32 PyTorch
autograd reference. The suite also verifies resolver identity and that the
public wrapper's compile-cache key contains the selected implementation.

## Performance

```bash
PYTHONPATH=python python \
  benchmark/dsa/benchmark_dsa_sparse_attention_backward_sm100_2_cta.py \
  --candidate sm100_2_cta_A \
  --seqlen 4096 \
  --topk 2048 \
  --warmup-pairs 4 \
  --paired-samples 20 \
  --json /tmp/dsa_bwd_sm100_2_cta_A.json
```

```bash
PYTHONPATH=python python \
  benchmark/dsa/benchmark_dsa_sparse_attention_backward_sm100_2_cta.py \
  --candidate sm100_2_cta_B \
  --seqlen 4096 \
  --topk 2048 \
  --warmup-pairs 4 \
  --paired-samples 20 \
  --json /tmp/dsa_bwd_sm100_2_cta_B.json
```

The benchmark compares `dsa_bwd_sm100.py` with the selected A or B source using
the same inputs, outputs, and workspaces. It runs an exact-shape
cross-implementation correctness gate before collecting ABBA-balanced
CUDA-event samples. Compilation, allocation, and buffer resets are excluded
from timing.
