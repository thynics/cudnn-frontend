# final_ser optimization ledger (2026-08-15)

## Frozen anchor

- Source: `dsa_bwd_sm100_2cta_final_ser_kq6q.py`
- Historical release commit: `6109041a5600fa85173b5859e17620e3b709a6e9`
- Historical source SHA256: `47f5d66492a864ce3a5efa4efbfde1e1aab3d4c2af3687babe4d8d1457b07e51`
- Historical paired result: candidate 9.786480 ms, baseline 8.318800 ms,
  ratio 1.180053 (B200, BF16, Sq=Skv=4096, H=128, D=512, topk=2048).
- Current experiment branch starts at `1729cd83bf13dbb58cb2f5ee1113218f12fc65a2`.
- Fixed perf protocol: direct baseline/candidate compile, identical inputs and
  workspaces, 8 warm-up pairs, 24 ABBA measured pairs, reset work excluded.
- Acceptance: at least 2% paired improvement, reproduced twice. Smaller changes
  remain provisional unless SASS/SMART evidence proves a critical-path removal.

## Pipeline map

The leader order is `S -> dP -> dV(r0,r1) -> dQ -> dK(r0,r1)`.  P/dS math
consumes S/dP in parallel; strict P-first relay gates dV, local dS gates dQ,
and relayed dS gates dK.  `score_kv` is time-shared between score-K and K_dQ.
The four-stage K32 round ring streams stationary dO/Q panels to dV/dK.

## Experiment table

| ID | Single variable | Prediction and evidence gate | Status | Result |
|---|---|---|---|---|
| M0 | Fresh exact-source kq6q replay | Re-establish paired B200 wall time and exact identity before changes. | pass | 9.501616 ms vs 8.074912 ms baseline; ratio 1.174088; 24 paired samples. |
| H1 | Spill containment only: specialize `pipe_s_done`/`pipe_dp_done` `consumer_mask=Int32(0)` and rematerialize the publication rank test next to P and dS stores. | Remove the known `[R1]` and `[R1+4]` stack live ranges without changing pipeline order, barriers, register budgets, or SMEM. Gate on line-info SASS totals and every remaining LDL/STL site, then paired B200. | accepted as structural cleanup | 9.519888 ms vs 8.134352 ms baseline; ratio 1.164844. Candidate absolute time is +0.19% vs M0, but exact line-info SASS drops 39 to 37 local instructions (LDL 23 to 21; STL remains 16) and removes the two hot loop reloads. |
| H2 | Candidate-derived N=4 named-barrier pacing, zero sleep. | Preserve atomic volume and rejoin all eight reducer warps after four atomics. | rejected | 9.756656 ms vs 8.065568 ms baseline; ratio 1.206043; +2.49% candidate wall time vs H1. The collective tail is harmful in the serial pipeline. |
| H2b | Sleep-only grouping: 150 ns after each group of four atomics, no cross-warp barrier. | Keep two quiet cuts per 8-op burst while reducing pacing overhead from 16 sleeps/tile to 4 and avoiding collective tail latency. | rejected | 9.652112 ms vs 8.029952 ms baseline; ratio 1.197703; +1.39% candidate wall time vs H1. Original per-atomic 150 ns pacing remains best. |
| H3 | Stream the two P publication groups into the tail of P math while retaining one final fence/sync/arrive. | Shorten the P compute-to-publish tail without adding hot-region local traffic. Gate on exact line-info SASS, exact-shape crosscheck, and a same-process H1/H3 ABBA comparison. | rejected | H3 has 36 local instructions versus H1's 37 and no new P hot-region spill, but the decision-grade B200 comparison is H1 9.259136 ms versus H3 9.260848 ms. H3/H1 is 1.000211; first half 1.000225, second half 1.000197, block-bootstrap 95% CI [1.000048, 1.000368]. The exact-shape crosscheck passes. Reverted to H1. |
| H4 | Raise the complete late warpgroup W16-W19 from `setmaxregister_decrease(64)` to 72, consuming the exact 1,024-register launch-allocation slack. | Give the leader/loader/relay/committer more dynamic registers without changing pipeline order. Require a no-cache line-info SASS audit, TopK=128 deadlock canary, and same-process H1/H4 ABBA. | rejected | Canary and exact-shape crosscheck pass. Local instructions fall only 37 to 36; the loader's three local accesses and committer spill remain. At TopK=2048, H1 is 9.258656 ms and H4 is 9.272320 ms: H4/H1 1.001417, first half 1.001509, second half 1.001244, block-bootstrap 95% CI [1.001237, 1.001521]. Reverted to H1 without SMART capture. |
| H5 | Rematerialize CTA rank at every W17 loader use and substitute the proven-equal constexpr `h_half` as the bulk-copy destination CTA. | Eliminate the rank live range across the fully unrolled 16-generation loader without changing slot, phase, barrier, or copy ownership. Require all four W17 rank-stack accesses to disappear in no-cache line-info SASS before timing. | rejected at codegen gate | Semantics review passes, but the compiler CSEs the rank reads. Exact SASS remains 37 local instructions and retains the same `LDL.LU + STL + STL + LDL` W17 stack sequence. No wall-time or SMART run was performed; reverted to H1. |
| H6 | Replace W17's hoisted rank with side-effecting `%cluster_ctarank` reads at its stationary-load and owner-copy uses. | Force short-lived special-register values after H5's ordinary rematerialization was CSE'd. Require the W17 stack sequence to disappear without increasing total local traffic before timing. | rejected at codegen gate | The no-cache build rises from H1's 37 local instructions to 57 (`STL=16`, `LDL=41`). The two unrolled owner tests each gain eight attributed LDLs, for a net increase of 20 local loads. No performance run was performed; reverted to H1. |
| H7 | Split P and dS storage ownership, release P after dV-r1 source-read completion, and reuse its aligned 8 KiB allocation for late round generation G12. Preserve all 16 normal round generations with a data-free G12 phase token and protect the alias with a one-stage late pipeline. | Remove G12 from the contended four-slot round ring without increasing SMEM or changing dV/dQ/dK order. Require release/trace parity, a barrier happens-before proof, no-cache line-info SASS, a TopK=128 canary, and same-process H1/H7 ABBA. | rejected | Static barrier/phase audit and normalized release/trace parity pass. H7 remains REG=96, STACK=8 and 37 local instructions (`STL=16`, `LDL=21`, unattributed=0), exactly H1's total; the 19th source location is only the existing W16 STL being re-attributed from the generic JIT entry to fragment setup, with no hot-role increase. TopK=128 crosscheck/deadlock canary passes and gives H7/H1 0.998487. At TopK=2048, however, H1 is 9.260080 ms and H7 is 9.316224 ms: H7/H1 1.005928, first half 1.005839, second half 1.005968, block-bootstrap 95% CI [1.005796, 1.006147]. The extra alias-side coordination costs more than relieving one ring generation; reverted to H1 without SMART capture. |
| H8 | Reshape reducer issue from the H1 `150 ns / no cohort skew` pattern to four CTA/WG cohorts with `100 ns` inter-chunk pacing and `90 ns` cohort spacing; then split-screen pure pacing and pure dephase variants. | The exact H1 SMART trace shows the preceding tile's R1 atomic envelope covers essentially the complete next-tile P/relay wait, while the two reducer warpgroups start within a few ns. Test whether lowering instantaneous REDG concentration shortens P publication and peer landing without reducer lag. | rejected | The combination wins the TopK=128 canary (H8/H1 0.988834) but regresses at TopK=2048: 1.005992, CI [1.005535, 1.006064]. Exact-H1 class-override screens explain the reversal: pace 125/150 is 1.031037 (strong regression), pace 175/150 is 0.999903 with CI crossing 1, and pace150 plus 40 ns cohort spacing is 0.999912 with CI crossing 1. Thus H1's 150 ns pacing is already on the long-shape plateau; neither more pacing nor cohort dephase improves the critical path. Release and trace sources were reverted byte-for-byte to H1. |
| H9 | Split P readiness without a new barrier: rank 1 opens relay 0 as soon as CTA0's block 0 lands, W16 issues both `p_fragment_0` passes, then waits for rank 1's block 1 landing before the remaining dV passes. | Hide the measured rank-1 P/publication tail and about 0.267 us of peer landing behind the first two dV passes. Require a barrier/phase proof, release/trace parity, no-cache spill audit, TopK=128 canary, and same-process TopK=2048 ABBA. | rejected | Static barrier/phase audit and release/trace parity pass; exact-shape crosschecks pass. No-cache codegen stays REG=96/STACK=8 but local traffic rises from H1's 37 to 38 instructions (`STL=16`, `LDL=22`), with rank-dependent W18 relay control replacing the old straight-line spill sites. TopK=128 is neutral (H9/H1 0.999663, CI crosses 1). At TopK=2048 H1 is 9.274960 ms and H9 is 9.352368 ms: H9/H1 1.008226, first half 1.008248, second half 1.008221, CI [1.008119, 1.008558]. The repeated split-gate/control cost is larger than the overlap it exposes. Revert to H1; do not spend a SMART capture on the losing build. |

## Historical suffix audit

- The lowest archived absolute time is `final_ser_kq6m` at 9.529648 ms, but
  it was measured against a 7.994672 ms baseline in a different run.  Its
  source SHA256 is
  `18884aa404dbd7a84420b3cc2ea0e65257731e8cb78635cee32f4d5a2addb892`.
- A same-process B200 comparison removes that run-to-run ambiguity: H1/kq6q
  is 9.270592 ms and kq6m is 9.522384 ms.  The kq6m/H1 ratio is 1.027166;
  first half 1.027203, second half 1.027083, block-bootstrap 95% CI
  [1.026986, 1.027260].  The exact-shape crosscheck passes.
- Therefore the historical 9.529648 ms record does not replace H1.  The
  current optimization anchor remains kq6q/H1.

## Evidence boundary

- Old kq6e SMART and old kq6q IKET are templates only. Their timestamps, PCs,
  register locations, and span durations are not evidence for a changed build.
- Every accepted source gets a fresh source hash, line-info SASS/spill report,
  paired B200 result, and (when trace is used) source/SASS/PFM identity manifest.

## Current decision anchor

- Retained source is H1, SHA256
  `90610120873cebb9177e892e9cecf34df1adab52919961b5d65cdf7b1d8d708e`.
- H1/H3 adjudication used one B200 process, shared tensors/output/workspaces,
  32 warm-up pairs, and 48 ABBA/BAAB measured pairs. Accumulator resets were
  outside the CUDA event interval.
- Result artifact:
  `outputs/final_ser_kq6q_h1_h3_sameprocess_20260815/final_ser_h1_h3_abba.json`,
  SHA256 `f47fd2673261474d8ce695507d1278a7861038ba58fb55300b93c3fe38bfd98b`.
- Historical suffix adjudication artifact:
  `outputs/final_ser_kq6q_h6_volatile_rank_20260815/final_ser_h1_kq6m_abba.json`,
  SHA256 `3c82a99e6f627032795962f4757a9ac15086b93e1c5dd7e4915974c4a3934d4d`.
- H7 no-cache spill product:
  `outputs/final_ser_kq6q_h7_palias_20260815/spill/logs/codegen/compile/*.spill_product.json`,
  SHA256 `dfdf0971d5e0b0bc75aaf60990bea34eb44da6cfe2b1f6e2ef7d96dcf515cdfa`.
- H7 TopK=128 canary:
  `outputs/final_ser_kq6q_h7_palias_20260815/topk128_canary/topk128_canary.json`,
  SHA256 `9b504e66c13edd029ba8e9004f0ef9115db639b8e0d9db0cfba14ed9967fe9c7`.
- H7 TopK=2048 adjudication:
  `outputs/final_ser_kq6q_h7_palias_20260815/topk2048_abba/final_ser_h1_h7_abba_topk2048.json`,
  SHA256 `0c4b836da6ea08f3f4c4c03259d3fa7e8f4137f39077bf7db588752a8e2fbf31`.
- H8 reduce-phase screen:
  `outputs/final_ser_kq6q_h8_reduce_phase_20260815/`; the TopK=2048
  combination artifact SHA256 is
  `c796983a39e7cc0ec4a137450a9f4099e2dcb52bf37b530b9b8cfb40d545c0b6`,
  and the pure-dephase artifact SHA256 is
  `4cdbf3b0d3d8bdaca32ba1396a821fd4313c8e9ee733013330d10b5201725dcf`.
- H9 split-P gate artifacts:
  `outputs/final_ser_kq6q_h9_split_p_gate_20260815/`.  The no-cache spill
  product SHA256 is
  `4a13f6af0080a61eb4fd9ab91b785e93e7ec1a5160c117b9e0a3b6bd37c5dbe2`,
  the TopK=128 canary SHA256 is
  `7d8e647895af23c1a3fcf952c4902902e3908348c5e02f65f3beaa684f2c61e8`,
  and the TopK=2048 adjudication SHA256 is
  `08b5f4710dc7876b0ee9b7641c824f1bb8da4bbea765a3b5f073685a49fa67f9`.
- The existing H1 SMART capture remains exact after the revert: release SHA256
  `90610120873cebb9177e892e9cecf34df1adab52919961b5d65cdf7b1d8d708e`,
  trace-twin SHA256
  `adcbf184f9d558a7f84d6ffbeb80b42d1e11ea01b33d2d6bb2bcf6398fe815b4`.

## Exact H1 P-relay decomposition

- Steady tiles 1..30: W16 `WAIT_P_RELAY` median is 0.8498 us. About
  0.5409 us elapses before the latest cluster P publication completes.
- After that P end, relay remains exposed for about 0.2960 us. The dominant
  component is rank 0/W18 waiting for the peer P landing: 0.2670 us median;
  relay-open bookkeeping is 0.0260 us and W16 wake-up is only 0.0051 us.
- CTA1 is the systematic P straggler: its P-publish end trails CTA0 by
  0.2864 us median. This is a real exact-native cross-CTA skew, not a role
  aggregation artifact.
- Therefore the next structural target is the CTA-asymmetric P producer and
  peer-landing path. H8 proves that globally retuning atomic pacing does not
  remove that asymmetry at TopK=2048.
