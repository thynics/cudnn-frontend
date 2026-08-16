# Rubin vpagealias_b 300+ KiB design

## Frozen base

- Source: `dsa_bwd_sm100_2cta_vpagealias_b.py`
- Revision: `d06469b028de7e459edd7f5b791779148c06d7ff`
- SHA256: `43dc5aab3c6cd1e112b8aa9564044d20febcdc74cf2920004a78fb37f9629d84`
- This is the current mapped candidate. `vpagealias_a` is diagnostic evidence only.

## Target

Run on Rubin SM107 using native `sm_107a`, TVM-FFI and hardware `AllowOversized` mode. The effective main-kernel
SMEM allocation must exceed 300 KiB and remain at or below 334,848 bytes. Optimize until the candidate is faster
than the exact one-CTA baseline at the same workload and timing protocol.

## First structural revision

Keep B's rotated scheduling and computation, but remove its cyclic Q/dO page alias ownership:

1. Keep `stationary_q` and `stationary_do` read-only for the complete kernel.
2. Add independent 32 KiB `direct_q`, 32 KiB `direct_do`, and 32 KiB `loan_do` storage.
3. Fill `direct_q`/`direct_do` once. Publish one readiness generation; never restore or overwrite them.
4. Delete `_transition_alias_pages_vb` and all per-tile alias acquire/release operations.
5. Initially preserve the existing score-loan generation schedule but point it at `loan_do`, not `score_kv`.

Expected live SMEM before removing obsolete barriers: 231,424 + 3 * 32,768 = 329,728 bytes. This is 322 KiB,
above 300 KiB and 5,120 bytes below Rubin's 334,848-byte limit.

## Final validation protocol

- Every GPU command runs under a hard timeout.
- Compare against the frozen exact one-CTA baseline with alternating AB/BA paired timing.
- Validate the complete dynamic-length contract against both the exact baseline and
  the Torch autograd reference before accepting a performance result.
- Exercise a 4096-row KV with 2048 selected rows, `None`, full runtime length,
  holes, ragged and tile-boundary lengths, and all-zero lengths.
- Sweep topK 128–2048 only after the full correctness gate passes.

The accepted implementation and measurements are recorded in `README.md`.
