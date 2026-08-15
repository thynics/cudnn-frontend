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
- The existing H1 SMART capture remains exact after the revert: release SHA256
  `90610120873cebb9177e892e9cecf34df1adab52919961b5d65cdf7b1d8d708e`,
  trace-twin SHA256
  `adcbf184f9d558a7f84d6ffbeb80b42d1e11ea01b33d2d6bb2bcf6398fe815b4`.
