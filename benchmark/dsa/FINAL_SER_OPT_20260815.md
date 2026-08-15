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
| H1 | Spill containment only: specialize `pipe_s_done`/`pipe_dp_done` `consumer_mask=Int32(0)` and rematerialize the publication rank test next to P and dS stores. | Remove the known `[R1]` and `[R1+4]` stack live ranges without changing pipeline order, barriers, register budgets, or SMEM. Gate on line-info SASS totals and every remaining LDL/STL site, then paired B200. | provisional; SASS running | 9.519888 ms vs 8.134352 ms baseline; ratio 1.164844. Candidate absolute time is +0.19% vs M0, so no wall-time win. Keep only if exact SASS removes the intended spill sites. |
| H2 | Candidate-derived N=4 named-barrier pacing, zero sleep. | Preserve atomic volume and rejoin all eight reducer warps after four atomics. | rejected | 9.756656 ms vs 8.065568 ms baseline; ratio 1.206043; +2.49% candidate wall time vs H1. The collective tail is harmful in the serial pipeline. |
| H2b | Sleep-only grouping: 150 ns after each group of four atomics, no cross-warp barrier. | Keep two quiet cuts per 8-op burst while reducing pacing overhead from 16 sleeps/tile to 4 and avoiding collective tail latency. | running | pending |
| H3 | SMEM lifetime/time-sharing change, one loan at a time. | Use SMART to identify an exposed leader wait first. Change only the owning generation/release boundary; reject ring additions that serialize TMA or increase exposed round waits. | pending | pending |

## Evidence boundary

- Old kq6e SMART and old kq6q IKET are templates only. Their timestamps, PCs,
  register locations, and span durations are not evidence for a changed build.
- Every accepted source gets a fresh source hash, line-info SASS/spill report,
  paired B200 result, and (when trace is used) source/SASS/PFM identity manifest.
