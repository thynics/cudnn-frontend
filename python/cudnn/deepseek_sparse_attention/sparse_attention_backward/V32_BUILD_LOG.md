[0] build start
[1] docstrings + class constants (tilers/TMEM map/gens) done, py_compile OK, ~180 lines
[2] _specialize_shared_storage (231,424 account, dqb gates, chase ring) done, py_compile OK, ~120 lines
[3] host layouts + TMA atoms (contract constants, gated builders, transposed dq-epi view) done, py_compile OK, ~230 lines
[4] gather/chase (piece loader, 2-slot ring driver, kdq wave-pair fills, width-explicit leaf copies) done, py_compile OK, ~340 lines
[5a] prologue views/fragments (panel-B, chase-A, slab-A, gen-B, dq_b-B, transposed partitions) done, py_compile OK, ~330 lines
[5] prologue complete (pipelines: chase ring depth 2, dqb gate init count-2 per relay precedent, bundle tile_count, TMEM map auto via constants) py_compile OK, ~80 lines
[6] MMA issue helpers (_issue_score_pieces_v32 K-outer, _issue_dkv_round_v32, _issue_dq_wave_v32) done, py_compile OK, ~200 lines
[7] math role (chunked loop, column-axis stats, whole-image publishes, dq_b own image + free gate) done, py_compile OK, ~420 lines
[11] leader schedule (dr-major: score K-outer, G5 r0, grads r0..r3, G5 r1 + free commit; dq_done group commit tail) done, py_compile OK, ~310 lines
[8] W17 supply (2-box panels, 12-gen ring: kdq r0 / 4x(dO,Q) TMA / kdq r1) done, py_compile OK, ~260 lines
[9] relay/cluster gates (single 8KB DSM peer push, mb_dqb errata-#2 arming, st.async marked V32-TODO) done, py_compile OK, ~120 lines
[10] drains + dQ epilogue wiring (_drain_dkv_block_v32 x8/bundle, epilogue kept v17a-orientation, staging moved to round bufs) done, py_compile OK, ~240 lines
[12] V32 SELF-AUDIT trailer + dormant-helper deletion (-462 lines) + IKET budget (26 static names <= 29) done, py_compile OK
[13] final sweep: stale-comment fixes, build complete at 14992 lines, py_compile OK
[14] rotation surgery: prologue score(0) + guarded score(t+1) per iter (probe-A 67% backpressure verdict); gather co-rotation: r1(prev) kdq moved out of the piece-2 boundary to after the chase loop (three-axis audit caught the four-role cycle); trace-reading note: S_ISSUE/dP_ISSUE payload=scored bundle b, wall-clock now lands in leader iteration b-1 (b=0 in prologue) -- pair ad-hoc probes with payload-1. py_compile OK
