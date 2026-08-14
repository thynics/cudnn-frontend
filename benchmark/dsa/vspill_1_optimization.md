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

- measured state: mixed latency/resource-contention bound; exact-current full attribution pending.
- component isolation: old 2-CTA lineage retained T2R/index/sync but disabled REDG and measured 7.808864 ms; ordinary STG measured 9.396032 ms. This proves a large write-path/interference lever historically, but exact aug_6 must be remeasured before using the magnitude.
- baseline comparison: 1-CTA uses eight reducer warps total; candidate uses eight per CTA, sixteen per cluster, so symmetric pipeline release can create a 2x-wide cluster REDG burst for the same logical work.
- spill: prior post-RA stack evidence is real but tied to an older source SHA; exact-current PCs and live ranges are not yet established. Uniform register expansion previously worsened stack traffic.
- P/dS: aug5 split P/dS release improved paired ratio by about 0.22%; retain it. Current compact no-atomic trace is aug3, not exact aug_6.
- estimated headroom: no-write historical upper bound is enough to cross the 1-CTA reference; legal scheduling must recover at least ~11% candidate time without removing required accumulation.
- last refreshed: M0 / 2026-08-14

## Catalog adjudication (session 1)

- config: queued reducer-warp-count and dKV-stage experiments; 2-CTA is fixed.
- staging: queued coarse double-buffer/off-loop write drain; per-chunk synchronous TMA is rejected by aug7.
- warp-topology: queued 4 reducers/CTA and producer/atomic-writer split.
- math-path: grouped stats/STSM path already optimized; PTX audit queued before changing softmax math.
- data-movement: queued vector-width/coalescing audit and write-combining; direct STG historical probe was worse.
- scheduling: queued CTA/WG phase staggering, dV/dK issue reordering, and score-vs-gradient cadence changes.
- design: queued global-workspace privatization/finalize only if collision geometry shows enough reuse.

## Attempts

| id | date | hypothesis (WHY it should help) | change (WHAT, minimal) | result (ms/TFLOPS) | vs best | verdict | evidence |
|----|------|--------------------------------|------------------------|--------------------|---------|---------|----------|
| P1 | 2026-08-14 | If exact aug_6 still approaches the historical no-write bound when only REDG is suppressed, required dKV writes—not current P/dS or round changes—remain the dominant interference lever. | Retained T2R, top-k preload, addresses, fences, releases, pacing, and loop control; runtime-false only the two atomic calls. | 8.004798 ms / 686.783 TFLOPS (candidate-only; exact full M0 8.914872 ms) | +10.21% diagnostic upper bound | INCONCLUSIVE | Exact revisions `3c82bab` vs `4baffdb`, same B200 allocation/protocol, 8 warmups + 24 repeats. Confirms a 0.910074 ms write-path lever; counters are still needed to split LSU/MIO/L2/issue causes. |
| A1 | 2026-08-14 | Both CTAs release the same dKV generation symmetrically, so 16 reducer warps issue REDG together. Delaying rank 1 by 256 ns after T2R release should lower peak write-queue pressure while hiding the delay under independent pipeline work. | Pending: restore REDG and insert one rank-1 nanosleep before each slot's eight-group atomic burst; preserve N=4 pacing and all producer/consumer edges. | pending | pending | INCONCLUSIVE | First use candidate-only screen; run full correctness+ABBA only if at least 2% faster. |

## Next steps

- [ ] P1 exact-current REDG-off isolation — determines whether write scheduling remains the dominant lever — expected diagnostic delta up to 15% — cost code/probe — class data-movement
- [ ] Rebind exact-current ptxas stack/LDL/STL with line info — prevents optimizing stale PCs — expected attribution only — cost compile/NCU-source — class staging
- [ ] Four-wave CTA/WG phase stagger without extra total barriers — halve instantaneous cluster write issue width — expected 3-10% — cost code — class scheduling
- [ ] Four reducer warps per CTA with doubled per-thread drain — match baseline's eight reducer warps per cluster and free register pool — expected 5-12% — cost design-revision — class warp-topology
- [ ] Separate P and dS REDG slots in time (dV slot before/under dK compute) — avoid two back-to-back bursts — expected 2-6% — cost code — class scheduling
- [ ] Move dKV atomic issue behind the next score's TMA/score MMA rather than P/dS math — exploit different pipe pressure — expected 2-6% — cost design-revision — class scheduling
- [ ] Add a coarse two-tile write queue with no per-chunk waits — preserve aug7's off-critical-path goal without its 32 barriers/tile — expected 5-12% — cost design-revision — class staging
- [ ] Audit FP32x4 REDG address order and L2 sector locality; reorder i/subtile traversal — improve merging and reduce queue residency — expected 1-4% — cost code/NCU — class data-movement
- [ ] Sweep reducer register allocation 112/120/128 while keeping total pool neutral — eliminate exact proven spills only — expected 0-3% — cost config — class config
- [ ] Sweep dKV done stages 2→3 subject to TMEM budget — add producer slack around bursty drain — expected 1-5% — cost config — class staging
- [ ] Reorder dQ rounds versus dV/dK head after P/dS publish — cover relay/write long poles with useful MMA — expected 2-8% — cost design-revision — class scheduling
- [ ] Privatize dKV partials by query block and finalize in a bandwidth-efficient kernel if collision statistics permit — remove in-pipeline atomics — expected uncertain, potentially >10% — cost design-revision — class design

## Dead ends (do NOT retry)

- N=2 CTA-wide pacing — paired ratio 1.116771, worse than N=4 lineage — `outputs/20260814T131534Z_vfinal_aug_6_578418/perf.json` — regime current reducer topology
- Per-chunk compact TMA drain with barrier/wait/reuse on every chunk — 21.238784 ms, ratio 2.667651 — `outputs/20260814T151311Z_vfinal_aug_7_600215/perf.json` — regime aug7
- Uniformly raising all low-role register budgets — older exact SASS stack sites increased rather than decreased — `profile/vspill1-spill-forensics-e213fd4-20260814/REPORT.md` — regime old source, direction only

## Session log

- 2026-08-14 session 1: M0 correctness+paired perf frozen; full current source read; old profile identities audited; exact-current REDG-off isolation queued before expensive trace. No kernel change yet.
