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
| A4 | 2026-08-14 | Reducer `LDL [R1+0x4] → R2UR` at `0x10800` and `0x11940` reloads the uniform loop bound 4.194M times and carries 11,336 long-scoreboard samples; explicitly retaining `tile_count` in the uniform domain should remove this stack round trip without changing the CTA pool. | Pending: apply `cute.arch.make_warp_uniform(tile_count)` immediately after the existing ceil-div; preserve role split `48/128/120/64`, atomics, schedule, and topology. | pending | pending | — | Pre-change hypothesis from exact M0 SourceCounters and SASS dataflow: `[R1+0x4]` gates loop entry, feeds per-tile state, and terminates the reducer loop. |

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
- [ ] Reducer-role local rematerialization of loop-bound/CTA-rank values — eliminate the three
  exact local sites without changing any role budget — expected 1-5% — cost code — class staging
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

## Session log

- 2026-08-14 session 1 continuation: exact-current REDG-off measured a 10.21% diagnostic upper bound; A1 rank stagger and A2 four-reducer topology both rejected; A2 source restored byte-identically to M0. Exact M0 source+full NCU collected before any new SMART trace.
- 2026-08-14 exact NCU closeout: REDG-off is a joint atomic plus reducer-live-range
  bound, not atomic-only. Three exact reducer-local sites account for 6.291M dynamic spill
  instructions as a strong cross-binary signature, not strict one-site causality. A3 direct
  120→128 was rejected before editing because it exceeds the CTA launch pool; no new SMART
  trace warranted before source-local liveness and compile-only SASS tests.
