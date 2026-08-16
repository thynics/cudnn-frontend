# Rubin `vpagealias_b` optimization

Status: validated Rubin candidate; not yet promoted into the develop source tree.

## Exact mapping

- Baseline (one CTA):
  `/home/longcheng/cudnn-frontend-dev-longcheng-pagealias-aug6/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_baseline.py`
  - SHA256: `a86b353a2349962bd4404818a906ea2d4df4ca5cb88b22d15ea234fc4e8ff3d7`
- B200-best source used as the Rubin starting point:
  `/home/longcheng/cudnn-frontend-dev-longcheng-pagealias-aug6/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_vpagealias_b.py`
  - SHA256: `43dc5aab3c6cd1e112b8aa9564044d20febcdc74cf2920004a78fb37f9629d84`
- Frozen Rubin control (the previous validated candidate):
  `/home/longcheng/dsa-trace-workspaces/rubin-opt/vpagealias-b/kernel_rubin_fixed_tiles_qr1_smem.py`
  - SHA256: `651c8a572b325cf3e55170d472914dda6785b3e0b8efb10ebdf90f712abbcf98`
- Current validated Rubin candidate:
  `/home/longcheng/dsa-trace-workspaces/rubin-opt/vpagealias-b/kernel_exp_dq_role_move.py`
  - SHA256: `190cc27f7ddde3a712c319a2aaa7717301c29403bf85ec3332470659fe40d13c`

## Candidate contract

- Native target: `sm_107a`, internal CUTLASS DSL
  `0.3.0+20260803235612.d88cc85`.
- Live SMEM: 329,728 bytes (`AllowOversized`, 300+ KiB mode).
- TMEM allocation: 576 columns; columns 512–575 are currently unused.
- The loop span is the compile-time `max_topk / N_TILE`, matching the exact
  baseline. Runtime `mTopkLength`, `None`, and `-1` validity predicates are all
  preserved. This is a general dynamic-length path, not the rejected
  full-length-only specialization.
- Q-r1 uses the original per-tile SMEM round path. The faster Q-r1 TMEM-A path
  was rejected because it corrupted only dKV columns 256–511; its S2T source
  lifetime was safe, so the remaining fault is in its layout/operand mapping.
- The dQ epilogue keeps the original two 64-value TMEM drains, but runs on
  gather warps W0–W3 after their final K/KdQ production. Math warps W4–W7 hand
  their 128-register allocation to W0–W3 after the final P/dS publish. This
  removes dQ drain work from the spill-heavy math role without delaying dKV.

## Correctness

`final_dq_role_dynamic_correctness.json` passes candidate vs exact baseline vs Torch for
all six cases at `seqlen_kv=4096`, `max_topk=2048`:

- `None` with all indices valid
- dynamic `[2048] * 6`
- `None` with tile-sized and scattered `-1` holes
- ragged lengths including zero
- positive lengths spanning 64/128 tile boundaries
- all-zero lengths

Positive lengths run the exact baseline's real length specialization. Cases
containing zero are represented equivalently as `None` plus a `-1` tail only
for the baseline, avoiding its unsafe zero-length `tile_index=-1` path.

## Rubin paired sweep

Sixteen AB/BA paired samples after four warmup pairs, milliseconds:

| topK | baseline | candidate | candidate / baseline | winner |
|---:|---:|---:|---:|---|
| 128 | 0.660256 | 0.724624 | 1.098088 | baseline |
| 256 | 0.911296 | 0.925328 | 1.016464 | baseline |
| 512 | 1.407248 | 1.320496 | 0.938836 | candidate |
| 1024 | 2.407248 | 2.130128 | 0.884871 | candidate |
| 2048 | 4.412112 | 3.695008 | 0.837695 | candidate |

The two-CTA candidate wins from topK 512 onward and is 16.2% faster at 2048.
For a production dispatcher, retain the exact one-CTA baseline for 128/256.

Local evidence:

- `final_dq_role_dynamic_correctness.json`
- `final_dq_role_rubin_sweep.json`
- `experiment_manifest.md`
- `fast_fixedtiles_dkv_diag.json` (rejected TMEM-A localization)

Remote source artifacts:

- `/home/scratch.longcheng_gpu/.dsa-rubin/rubin-feature-explore/runs/dq-role-move-correctness-20260816T131431Z`
- `/home/scratch.longcheng_gpu/.dsa-rubin/rubin-feature-explore/runs/dq-role-final190-baseline-sweep-20260816T133143Z`

Regenerable SMART raw SQLite/PFM and REST caches were removed; semantic traces,
frozen source, manifests, and the compact evidence above were retained.
