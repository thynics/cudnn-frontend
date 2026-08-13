# DSA Backward SM100 2-CTA

Run the following commands from the repository root. Both checks require an
SM100 GPU and the cuDNN Frontend CuTe DSL dependencies.

## Correctness

```bash
(
  cd test/python
  PYTHONPATH=../../python:. python -m pytest -q \
    fe_api/dsa/test_DSA_sparse_attention_backward_sm100_2_cta.py
)
```

This runs the public DSA wrapper with `implementation="sm100_2_cta"` and
checks its gradients against an independent FP32 PyTorch autograd reference.
The suite also verifies that the public wrapper compiled the requested 2-CTA
implementation.

## Performance

```bash
PYTHONPATH=python python \
  benchmark/dsa/benchmark_dsa_sparse_attention_backward_sm100_2_cta.py \
  --seqlen 4096 \
  --topk 2048 \
  --warmup-pairs 4 \
  --paired-samples 20 \
  --json /tmp/dsa_bwd_sm100_2_cta.json
```

The benchmark compares `dsa_bwd_sm100.py` with
`dsa_bwd_sm100_2_cta.py` using the same inputs, outputs, and workspaces. It
runs an exact-shape cross-implementation correctness gate before collecting
ABBA-balanced CUDA-event samples. Compilation, allocation, and buffer resets
are excluded from timing.
