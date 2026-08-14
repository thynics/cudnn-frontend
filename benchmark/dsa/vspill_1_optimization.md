# Optimization Ledger — DSA backward vspill_1 (B200/CuTe DSL)

## Baseline (frozen — session 1, 2026-08-14)

- kernel: 9.202976 ms · 597.367448 TFLOPS · 1-CTA reference: 8.207888 ms · candidate/reference: 1.119219
- primary metric: paired candidate/reference ratio, then candidate median ms · protocol: 8 warmup ABBA pairs + 24 timed ABBA pairs, program-only CUDA events · arbiter: `run_b200_pipeline.sh --impl vspill_1 --mode perf`
- correctness: PASS (`dense`, `lengths`, `holes`, `all_empty`) in `outputs/20260814_vspill1_m0_fresh/`
- workload: B200 sm_100a, BF16, Sq=4096, Skv=4096, H=128, Dqk=Dv=512, topk=2048
- config: 2-CTA cluster, N64 KV tile, H64/CTA, D128/CTA, 32 KV tiles, 20 warps/CTA, 2 TMEM dKV stages, 2 round stages, reducer pacing N=4/sleep=0
- source: commit `4baffdbd571fcb7c6754e2d89c0738f587ca6d17`, SHA256 `ac840c3bafded39fb12f14eb50704b55acf68965e1f5c6e8187a58b67cabdb2a`
- workspace backup: immutable byte-identical `dsa_bwd_sm100_2cta_vfinal_aug_6.py` plus git commit `4baffdbd`

## Current best

- attempt: M0 · 9.202976 ms · 597.367448 TFLOPS · 1.000000x vs frozen candidate (paired ratio 1.119219)
- config delta vs baseline: none

## Kernel structure map (session 1 — full source read)

- roles/CTA: W0-W3 sparse gather; W4-W7 P/dS math and dQ epilogue; W8-W15 dKV T2R+REDG; W16 sole cluster MMA issuer on rank 0; W17 round loader; W18 P/dS DSM relay; W19 TMA commit relay.
- score path: S(t), dP(t) ping-pong in four resident D128 chunks; K score image is a single-stage borrowed buffer.
- gradient path: dQ two rounds, then dV/dK head into dKV slot 0 and tail into slot 1; round ring depth 2.
- P/dS path: four math warps T2R S/dP, packed math, STSM publication; P and dS have separate one-stage credits; W18 performs two 4-KiB DSM sends.
- reducer: 8 warps per CTA (16 per cluster), each slot loads 32 FP32 values/thread and emits eight FP32x4 REDG groups; current CTA-wide barrier after groups 4 and 8.
- tail: dQ TMA epilogue, full CTA+cluster rendezvous, TMEM free.
- exposed knobs: reducer warp count, burst phase/pacing, register split, dKV stage depth, score/gradient issue order, round stage depth, P/dS publication order, own-half bulk policy.

## Bound analysis (living)

- measured state: exact-current NCU classifies M0 as L1TEX-latency/resource-contention bound, not HBM-bandwidth bound. Full NCU reports 8.74 ms, DRAM 2.44%, L1/TEX 63.91%, L2 31.81%, compute 43.91%, and only 0.24 eligible warps/scheduler.
- issue gap: long scoreboard is 18.5 of 28.1 cycles between issued instructions (65.6%); exact SourceCounters records 322,759 not-issued long-scoreboard samples out of 460,497. The largest sites are pipeline-poll branches in the reducer and math roles, so the number is an end-to-end dependency symptom rather than isolated load latency.
- spill: exact-current NCU executes 4,845,568 local-memory spilling requests and 1,220,608 shared-memory spilling requests. Local traffic is 5.71% of L1TEX sectors and NCU's isolated local-memory rule gives a 20.02% upper bound; 100% of local loads hit L1, so spill is real but not yet proven to explain the full 0.910 ms REDG-off delta.
- component isolation: exact aug6 REDG-off measures 8.004798 ms versus same-device M0
  8.914872 ms, a 0.910074 ms / 10.21% release-timing upper bound. Exact NCU measures
  7.823488 versus 8.742880 ms (-10.52%), but executed instructions also fall 34.90%.
  The runtime-false branch therefore isolates the joint REDG operand/address/live-range and
  atomic side-effect path, not atomic memory alone.
- reducer spill localization: exact M0's reducer-relative SASS region
  `[0x10800,0x11a00)` contains one `LDL`, one `STL`, and one `LDL`, each
  executing 2,097,152 times. Their 6,291,456 instructions are 37.24% of M0 spill
  instructions and form a near-one-to-one signature of REDG-off's net 6,225,920
  spill-instruction drop. Cross-binary ptxas reallocation prevents a strict one-site causal
  claim; source-local liveness relief is the smallest safe discriminating test.
- baseline comparison: 1-CTA uses eight reducer warps total; candidate uses eight per CTA, sixteen per cluster, so symmetric pipeline release can create a 2x-wide cluster REDG burst for the same logical work.
- occupancy: 20 warps/CTA, 96 allocated registers/thread, and 231.42 KiB dynamic SMEM force one block/SM; theoretical and achieved occupancy both equal about 31.2%. Register tuning cannot add a second resident CTA while SMEM remains full, but role-neutral register-pool redistribution is still available.
- P/dS: aug5 split P/dS release improved paired ratio by about 0.22%; retain it. Current compact no-atomic trace is aug3, not exact aug_6.
- estimated headroom: no-write historical upper bound is enough to cross the 1-CTA reference; legal scheduling must recover at least ~11% candidate time without removing required accumulation.
- exact profile: `profile/vspill1-aug6-current-20260814/` (`source_aug6_m0.ncu-rep` SHA256 `07e9bdca...`, `full_aug6_m0.ncu-rep` SHA256 `bdc35e2d...`).
- exact A/B report:
  `profile/vspill1-aug6-m0-vs-redg-off-20260814/REPORT.md`
- last refreshed: exact M0 vs REDG-off full+source NCU / 2026-08-14

## Catalog adjudication (session 1)

- config: queued reducer-warp-count and dKV-stage experiments; 2-CTA is fixed.
- staging: queued coarse double-buffer/off-loop write drain; per-chunk synchronous TMA is rejected by aug7.
- warp-topology: A2 four reducers/CTA rejected; producer/atomic-writer split remains queued.
- math-path: grouped stats/STSM path already optimized; PTX audit queued before changing softmax math.
- data-movement: queued vector-width/coalescing audit and write-combining; direct STG historical probe was worse.
- scheduling: queued CTA/WG phase staggering, dV/dK issue reordering, and score-vs-gradient cadence changes.
- design: queued global-workspace privatization/finalize only if collision geometry shows enough reuse.

## Attempts

| id | date | hypothesis (WHY it should help) | change (WHAT, minimal) | result (ms/TFLOPS) | vs candidate-only M0 | verdict | evidence |
|----|------|--------------------------------|------------------------|--------------------|---------|---------|----------|
| P1 | 2026-08-14 | If exact aug6 speeds up when only REDG is suppressed, dKV writes remain a dominant lever. | Runtime-false the two atomics; retain T2R, top-k loads, addresses, fences, releases, pacing, and control flow. | 8.004798 ms / 686.783 TFLOPS vs M0 8.914872 ms | +10.21% | INCONCLUSIVE | Revisions `3c82bab` vs `4baffdb`; same B200, 8 warmups + 24 repeats. The 0.910074 ms bound needs NCU cause-splitting. |
| A1 | 2026-08-14 | Rank-1 delay may break the simultaneous 16-warp REDG burst and lower peak queue pressure. | Inserted 256 ns rank-1 sleep before each slot burst; retained N=4 pacing and all handoffs. | 9.015439 ms / 609.794 TFLOPS | -1.13% | REJECTED | Revision `8199655` vs M0 8.914872 ms; fixed delay adds tail/backpressure. |
| A2 | 2026-08-14 | Four reducers/CTA match baseline's eight cluster-wide writers; doubled sequential work may lower peak pressure. | W8-W11 active, W12-W15 idle; one N64 T2R fragment, 16 REDG groups/thread, 160 reducer registers. | 9.092323 ms / 604.637 TFLOPS | -1.99% | REJECTED | Revision `5ae43d4` vs M0 8.914872 ms: +0.177451 ms / 1.019905x runtime. Source restored byte-identically to M0. |
| A3 | 2026-08-14 | Exact M0 executes three reducer-local spill sites 6.291M times; raising reducer warps by eight registers appeared able to remove this path without changing occupancy. | Not run: reducer `setmaxregister_increase(120→128)` would consume 63,488 registers/CTA versus the 61,440-register launch pool; each SMSP would be 16 warp-register units short. | N/A | N/A | REJECTED | Static `setmaxnreg` pool audit: physical 65,536-register capacity does not enlarge the launch-time `640 threads × 96 registers` CTA pool; an unfunded increase may block permanently. |
| A4 | 2026-08-14 | Reducer `LDL [R1+0x4] → R2UR` at `0x10800` and `0x11940` reloads the uniform loop bound 4.194M times and carries 11,336 long-scoreboard samples; explicitly retaining `tile_count` in the uniform domain should remove this stack round trip without changing the CTA pool. | Applied `cute.arch.make_warp_uniform(tile_count)` immediately after the existing ceil-div; preserved role split `48/128/120/64`, atomics, schedule, and topology. | Compile-only: reducer locals unchanged apart from `-0x30` PC shift (`LDL 0x107d0`, `STL 0x11090`, `LDL 0x11910`); whole main kernel 48→49 LDL, 21→22 STL, STACK 16→24 bytes. | N/A | REJECTED | Revision `108006c`; exact B200 cubin `/home/scratch.longcheng_gpu/.dsa-allinone/a4-108006c/capture/`. The uniform hint did not shorten the live range and worsened static spill shape, so perf was skipped by the predeclared SASS gate. |
| A5 | 2026-08-14 | Reducer-local scalar rematerialization should remove three hot local sites without changing the register pool. | Recomputed topk/tile count/rank after role dispatch. | M0 9.073792 ms; A5 9.095104 ms / 604.452 TFLOPS | -0.088% | REJECTED | B200 48-pair AB. Static main kernel 48→45 LDL, 21 STL/STACK16; reducer has no local ops. Artifact `.dsa-allinone/a5-1120280/perf-m0-vs-a5-r1/perf.json`. Restored M0. |
| A6 | 2026-08-14 | Four static rank/warp REDG orders could spread concurrent row-index waves without delays. | Not run: 0/2/4/6 mechanism clones 16→64 static REDG sites. | Historical final-gate ratio 1.009475 | N/A | REJECTED | `76f5df1` includes dual-warp-loan+rotation vs f987, so rotation-only timing is unavailable; final gate was 0.948% slower/STACK24. Existing structural cost rejected a duplicate current port. |
| A7 | 2026-08-14 | Slot-0's terminal 256-thread barrier re-locksteps reducers into a sharp slot-1 burst; removing only that gate should let wait/T2R latency dephase them naturally. | Retained the N=4 mid-slot gates and final slot-1 gate; removed only slot-0 terminal pace gate. | M0 9.087824 ms; A7 9.086576 ms / 605.020 TFLOPS | +0.029% | REJECTED | Same-B200 ABBA-balanced 48-pair run, commit `def3acf`; ratio 0.999709. Artifact `.dsa-allinone/a7-def3acf/perf-m0-vs-a7-r1-result/perf.json`. Below the predeclared 2% acceptance threshold; no second run or trace. |
| A8 | 2026-08-14 | The two CTA/two-WG reducers issue the same top-k row wave together; complementary 3/5 orders may lower same-address REDG queue collisions without adding dynamic work or barriers. | `cohort = wg_idx ^ rank`; cohort 0 issued groups 0:3 then 3:8, cohort 1 issued 3:8 then 0:3, rendezvousing at the existing mid and terminal barriers. Compile-time ranges preserved static register indexing. | M0 8.972000 ms; A8 9.014064 ms / 609.887 TFLOPS | -0.434% | REJECTED | Same-B200 ABBA-balanced 48-pair run, commit `5b2c835`; ratio 1.004336. REG96/STACK16 unchanged, static vector-atomic sites 16→32. Artifact `.dsa-allinone/a8-5b2c835/perf-m0-vs-a8-r1/perf.json`. |
| A9 | 2026-08-14 | If cross-query clusters serialize on the same `(KV row, D quad)`, hashing tokens over two independent FP32 workspace planes should relieve that serialization without changing REDG count, writer width, or upstream work. | Allocated two 8-MiB dKV planes, routed `token_idx % 2`, and summed both planes in the canonical BF16 finalize kernel. | M0 8.922320 ms; A9 8.925792 ms / 615.918 TFLOPS | -0.051% | REJECTED | Same-B200 ABBA-balanced 24-pair diagnostic; ratio 1.000508. Exact candidate workspace was `[1,2,4096,2048]` bytes versus one reference plane. Correct end-to-end P2 is neutral, so cross-query same-address serialization is not a useful production lever. Artifact `.dsa-allinone/a9-3e625dd/run-r0/perf-result/perf.json`; source restored to M0. |
| A10 | 2026-08-14 | The remaining atomic anomaly may be the synchronized 16-warp injection width rather than same-address collision; two WG waves keep each epoch at 64 warp REDG while halving simultaneous writers. | All eight reducer warps retained their original T2R fragment. `cohort = wg_idx ^ rank`; one 4-warp WG/CTA executed all eight vector atomics per phase, then all reducers rendezvoused on the existing barrier before the complementary phase. | M0 8.916064 ms; A10 9.136528 ms / 601.712 TFLOPS | -2.465% | REJECTED | Same-B200 ABBA-balanced 24-pair diagnostic; ratio 1.024654. PTX retained exactly 16 vector-atomic sites, so work was not cloned. Halving writer width serializes useful issue throughput; no second run, correctness, NCU, or trace. Artifact `.dsa-allinone/a10-029ccf6/run-r0/perf-result/perf.json`; source restored to M0. |
| A11 | 2026-08-14 | R0 already overlaps dK usefully, while R1 overlaps the following S/dP window; moving only R1 behind the next gradient-head completion may preserve atomic throughput while shifting half the interference. | After slot1 T2R/fence/release, non-final tiles observe the next `pipe_dkv_done` slot0 generation without advancing it, then issue R1; the next drain consumes the same ready generation normally. R0 and both TMEM releases remain unchanged. | PENDING | PENDING | RUNNING | Predeclared static gate: REG96/STACK16 and reducer local sites must not worsen. Historical both-slot gates were -3.03%/-6.06%, but this single-wave phase change is untested. Accept only at >=2%, then correctness and repeat. |

## Next steps

- [x] P1 exact-current REDG-off isolation — 8.004798 vs 8.914872 ms, a measured 0.910074 ms / 10.21% write-path upper bound — class data-movement
- [x] Localize exact-current reducer LDL/STL from exact SourceCounters and dynamic role
  boundaries — three sites / 6.291M instructions; DSL 4.5 line-marked `.sass` dump unavailable
  — class staging
- [ ] Four-wave CTA/WG phase stagger without extra total barriers — halve instantaneous cluster write issue width — expected 3-10% — cost code — class scheduling
- [x] Four reducer warps per CTA with doubled per-thread drain — rejected at 9.092323 ms, 1.99% slower than candidate-only M0 — class warp-topology
- [ ] Separate P and dS REDG slots in time (dV slot before/under dK compute) — avoid two back-to-back bursts — expected 2-6% — cost code — class scheduling
- [ ] Move dKV atomic issue behind the next score's TMA/score MMA rather than P/dS math — exploit different pipe pressure — expected 2-6% — cost design-revision — class scheduling
- [ ] Add a coarse two-tile write queue with no per-chunk waits — preserve aug7's off-critical-path goal without its 32 barriers/tile — expected 5-12% — cost design-revision — class staging
- [ ] Audit FP32x4 REDG address order and L2 sector locality; reorder i/subtile traversal — improve merging and reduce queue residency — expected 1-4% — cost code/NCU — class data-movement
- [x] A3 unfunded reducer allocation 120→128 — statically rejected before launch: 63,488
  required versus 61,440 in the CTA launch pool; possible permanent `setmaxnreg.inc` wait —
  class config
- [x] Reducer-role local rematerialization removed all three exact reducer local sites but was
  0.088% slower in a 48-pair AB; rejected and restored — class staging
- [ ] Sweep dKV done stages 2→3 subject to TMEM budget — add producer slack around bursty drain — expected 1-5% — cost config — class staging
- [ ] Reorder dQ rounds versus dV/dK head after P/dS publish — cover relay/write long poles with useful MMA — expected 2-8% — cost design-revision — class scheduling
- [ ] Privatize dKV partials by query block and finalize in a bandwidth-efficient kernel if collision statistics permit — remove in-pipeline atomics — expected uncertain, potentially >10% — cost design-revision — class design

## Dead ends (do NOT retry)

- N=2 CTA-wide pacing — paired ratio 1.116771, worse than N=4 lineage — `outputs/20260814T131534Z_vfinal_aug_6_578418/perf.json` — regime current reducer topology
- Per-chunk compact TMA drain with barrier/wait/reuse on every chunk — 21.238784 ms, ratio 2.667651 — `outputs/20260814T151311Z_vfinal_aug_7_600215/perf.json` — regime aug7
- Uniformly raising all low-role register budgets — older exact SASS stack sites increased rather than decreased — `profile/vspill1-spill-forensics-e213fd4-20260814/REPORT.md` — regime old source, direction only
- Four reducers/CTA with one 64-value fragment and 16 REDG groups/thread — 9.092323 ms, 1.99% slower than exact candidate-only M0 — revision `5ae43d4` — regime current math/pacing, source restored to M0
- Unfunded reducer 120→128 register increase — requires 63,488 registers/CTA versus the
  61,440-register launch pool and can wait forever at `setmaxnreg.inc`; rejected statically,
  never launched — regime current 640-thread role split
- Global `make_warp_uniform(tile_count)` hint — exact B200 compile leaves all three reducer
  local sites intact and worsens the main kernel from 48/21 LDL/STL with STACK16 to 49/22 with
  STACK24; revision `108006c`, no perf run — regime current `48/128/120/64`
- Reducer-local rematerialization — removed the target reducer LDL/STL sites and reduced static
  main-kernel LDL 48→45 with unchanged 21 STL/STACK16, but measured 0.088% slower than same-device
  M0 over 48 AB pairs; revision `1120280`, source restored — regime current reducer schedule
- Four-way static REDG rotation — `76f5df1` cloned 16→64 REDG sites and its combined
  dual-warp-loan+rotation candidate missed the f987 final gate by 0.948%; there was no direct c6
  arm, so rotation-only timing is unavailable. Current duplicate abandoned due the known
  8,248-instruction/STACK24 structural cost — regime dual parent; smaller schedules remain live.
- Slot-0 terminal pacing-barrier removal — safe but neutral: M0/A7 measured
  9.087824/9.086576 ms over 48 ABBA-balanced pairs (+0.029%, ratio 0.999709), far below
  the 2% acceptance threshold. The slot-1 wait/T2R does not create useful natural dephasing by
  itself — revision `def3acf`, source restored — regime current N=4 reducer schedule.
- Complementary 3/5 reducer group ordering — retained all 16 dynamic REDG and four barriers per
  tile with REG96/STACK16, but cloned static vector-atomic sites 16→32 and measured
  8.972000/9.014064 ms over 48 ABBA-balanced pairs (-0.434%, ratio 1.004336). Address-wave
  permutation without reducing active writers is not a useful lever — revision `5b2c835`, source
  restored — regime current N=4 reducer schedule.
- Two-plane query-hashed dKV workspace — preserved 2-CTA, REDG count, writer width, and upstream
  work, then correctly summed both FP32 planes in conversion, but measured 8.922320/8.925792 ms
  over 24 ABBA-balanced pairs (-0.051%, ratio 1.000508). Cross-query same-address serialization
  does not explain the candidate gap at an end-to-end production boundary — revision `3e625dd`,
  artifact `.dsa-allinone/a9-3e625dd/run-r0/perf-result/perf.json` — regime current M0.
- Two-wave 16→8 reducer-writer gate — retained all eight reducer warps and exactly 16 static
  vector-atomic sites, but measured 8.916064/9.136528 ms over 24 ABBA-balanced pairs (-2.465%,
  ratio 1.024654). The current 16-warp injection width is useful throughput rather than the gap;
  serializing complementary WGs is rejected — revision `029ccf6`, artifact
  `.dsa-allinone/a10-029ccf6/run-r0/perf-result/perf.json` — regime current M0.

## Session log

- 2026-08-14 session 1 continuation: exact-current REDG-off measured a 10.21% diagnostic upper bound; A1 rank stagger and A2 four-reducer topology both rejected; A2 source restored byte-identically to M0. Exact M0 source+full NCU collected before any new SMART trace.
- 2026-08-14 exact NCU closeout: REDG-off is a joint atomic plus reducer-live-range
  bound, not atomic-only. Three exact reducer-local sites account for 6.291M dynamic spill
  instructions as a strong cross-binary signature, not strict one-site causality. A3 direct
  120→128 was rejected before editing because it exceeds the CTA launch pool; no new SMART
  trace warranted before source-local liveness and compile-only SASS tests.
- 2026-08-14 A4 static gate: global uniform `tile_count` retained none of the intended values
  and added one static LDL/STL pair plus 8 stack bytes. Rejected without perf; A5 moves the
  rematerialization after reducer role dispatch and keeps the GPU lease for a fast compile gate.
- 2026-08-14 A5 closeout: role-local rematerialization removed the three reducer-region local
  sites, but same-device M0/A5 measured 9.073792/9.095104 ms over 48 AB pairs (-0.088%). The
  spill signature is not on the current end-to-end critical path; A5 was rejected and restored.
- 2026-08-14 A7 closeout: removing only the slot-0 terminal pacing barrier was correctness-safe
  but neutral at 9.087824/9.086576 ms over 48 ABBA-balanced pairs (+0.029%). Rejected without a
  second run or SMART trace; source restored to the byte-identical M0 reducer schedule.
- 2026-08-14 A8 closeout: two complementary 3/5 REDG group orders kept dynamic work and barrier
  count unchanged and compiled at REG96/STACK16, but were 0.434% slower (8.972000/9.014064 ms,
  48 ABBA-balanced pairs). Pure address permutation is rejected; no second run or trace.
- 2026-08-14 A9 closeout: a correct two-plane query hash changed only the global accumulation
  collision domain plus final sum and measured -0.051% end-to-end. Rejected below the 2% gate;
  no second run, correctness sweep, NCU, or SMART trace was warranted. The next discriminator is
  real 16→8 reducer-writer gating with unchanged dynamic REDG and fragment topology.
- 2026-08-14 A10 closeout: real two-wave writer gating preserved 16 static vector-atomic sites
  but was 2.465% slower. Both address-plane and issue-width hypotheses are now falsified at the
  production boundary; next test should change R1 overlap phase without delaying R0 or TMEM credit.
