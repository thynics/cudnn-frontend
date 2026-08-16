# Rubin feature exploration

Frozen control:

- `kernel_rubin_fixed_tiles_qr1_smem.py`
- SHA256 `651c8a572b325cf3e55170d472914dda6785b3e0b8efb10ebdf90f712abbcf98`

Every experiment starts from that source and changes one mechanism at a time.
Correctness must retain runtime `mTopkLength`, `None`, `-1`, ragged, tile-boundary,
and zero-length behavior.

| Experiment | File | Status | Purpose |
|---|---|---|---|
| self A/B | control source | pass | 3.743136 vs 3.745872 ms at topK 2048; +0.058% noise |
| compile-uumn | control source | rejected | Repeated launches caused a contained peer-memory/hardware fault; plain self A/B was stable |
| uniform-laneid | `kernel_exp_uniform_laneid.py` | neutral | 3.744192 vs 3.745584 ms at topK 2048; +0.039% |
| uniform TMEM base | `kernel_exp_uniform_tmem_base.py` | neutral | Full correctness passes; one converged broadcast for the CTA-wide TMEM root is -0.095% paired at 128 and -0.012% at 2048, both noise |
| xu64-exp2 | `kernel_exp_xu64.py` | neutral | 3.742768 vs 3.743184 ms at topK 2048; paired ratio 0.999996 |
| packed-exp2 | `kernel_exp_packed_exp2.py` | rejected | Full correctness passes; -0.14% at 128 but +0.43% at 512 and +0.35% at 2048 |
| dq-stream | `kernel_exp_dq_stream.py` | rejected | Full correctness passes, but 4×32 is +1.23% at 128 and +2.08% at 2048 |
| dq-role-move | `kernel_exp_dq_role_move.py` | winner | Full correctness passes; move unchanged 2×64 dQ drain from math W4–7 to idle gather W0–3: -2.43% at 128, -2.10% at 512, and -1.32% at 2048 |
| dq-role final vs one-CTA | `final_dq_role_rubin_sweep.json` | pass | Baseline retains 128/256; candidate wins by 6.12%, 11.51%, and 16.23% at 512/1024/2048 with 329,728 B live SMEM |
| dq-role 144 regs | transient edit | rejected | Full correctness passes; only -2.14% at 128 and -1.40% at 2048, so extra gather registers trade away the stronger short-length win for noise-level long gain |
| 16-warp CTA | `kernel_exp_w16.py` | rejected | Compiles on SM107a but traps with illegal instruction on first nonzero execution |
| qr1-tmem-v2 | `kernel_exp_qr1_tmem_v2.py` | rejected | Compiles/launches; dV half correct, dK half reads invalid TMEM data |
| qr1-tmem-v3 | `kernel_exp_qr1_tmem_v3.py` | rejected | Canonical 64-column physical layout removes NaN/Inf, but nonzero dK still mismatches in columns 256–511 |
| qr1-tmem-v4 | `kernel_exp_qr1_tmem_v4.py` | rejected | Official full-staged indexing produces the same finite 0.8% dK cols 256–511 mismatch as v3; TMEM-A physical fragment mapping remains unresolved |
| tma-gather4-v3 | `kernel_exp_tma_gather4_v3.py` | rejected | Full correctness passes, but per-32-row TMA mbarrier overhead is +27% at 128 and +182% at 2048 |
| direct Q/dO multicast v1 | `kernel_exp_direct_qdo_multicast.py` | rejected | A W19-only `cluster_arrive/wait` rendezvous deadlocks the first nonzero case; v2 replaced it with a dedicated peer-mbarrier handshake so the data mapping could be tested |
| direct Q/dO multicast v2 | `kernel_exp_direct_qdo_multicast_v2.py` | rejected | The handshake runs, but dKV fails (30,535/2,097,152 values, 1.5% in the full case). `partition_A(rank)` maps the identical-looking index to different D owners: Q uses D0:128/D128:256 and dO uses D256:384/D384:512, so no source tile is actually duplicated across the two CTAs |
| epilogue features | code/SASS audit | no experiment | dQ already uses UTMASTG and dKV uses max-width REDG.E.ADD.F32x4; ordinary 256-bit stores/cache hints cannot preserve these semantics |
| SMEM stage depth | code/trace audit | no experiment | 329,728/334,848 B leaves 5,120 B; reclaiming dead 4 KiB `ds_xchg` still cannot fit the smallest useful 12 KiB stage, and all producers already lead consumers by 0.28–1.41 us |
| L2 ownership scheduler | grid/data-reuse audit | no experiment | One `(2,1,1)` cluster already owns one token's H128 and full TopK; there is no cross-cluster Q-row reuse axis, while sparse KV rows are token-random and logical token swizzles cannot select a GR100 die side |
