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
- The existing H1 SMART capture remains exact after the revert: release SHA256
  `90610120873cebb9177e892e9cecf34df1adab52919961b5d65cdf7b1d8d708e`,
  trace-twin SHA256
  `adcbf184f9d558a7f84d6ffbeb80b42d1e11ea01b33d2d6bb2bcf6398fe815b4`.
