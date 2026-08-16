# DSA sparse-attention backward benchmark

This benchmark drives the public
`cudnn.DSA.sparse_attention_backward_wrapper` API. Dispatch is automatic:

- SM90 uses the Hopper implementation.
- SM100/SM103 use the Blackwell implementation.
- SM107 uses the Rubin implementation only for its validated performance
  envelope; every other shape falls back to the SM100 implementation.

The Rubin envelope is BF16, `H=128`, `Dqk=Dv=512`, `S_kv=4096`, block tile
64, and `topk` 512, 1024, or 2048. Runtime `topk_length` remains fully
supported. `topk` 128 and 256 deliberately use the fallback because the
two-CTA kernel is slower at those sizes.

## Run

```bash
python benchmark/dsa/benchmark_dsa_sparse_attention_backward.py
```

For the Rubin path, use:

```bash
python benchmark/dsa/benchmark_dsa_sparse_attention_backward.py \
  --seqlens 4096 \
  --topks 512,1024,2048 \
  --nheads 128 \
  --head-dim 512
```

Useful options include `--dtype`, `--no-topk-length`, `--warmup`, `--repeat`,
and `--csv`. The first warmup iteration compiles the selected CuTe DSL
specialization.

Inputs use the flat FlashMLA contract: `q (S_q,H,D)`, shared `kv (S_kv,D)`,
and `topk_idxs (S_q,topk)`. Reference forward produces the `out` and KV-only
`lse` consumed by backward. Reported FLOP/s uses the five-matmul model:

```text
FLOPs = 2 * S_q * H * topk * (3 * Dqk + 2 * Dv)
```

Historical kernels, trace tooling, and experiment-specific runners are kept
on `archive/dsa_bwd_rubin_experiments_20260816`; they are intentionally not
part of the production feature branch.
