# Agent context

The B200-best mapped candidate is `vpagealias_b` at commit `d06469b`. Its original
dual Q/dO alias restore/ownership protocol hung on Rubin SM107. The Rubin branch
keeps B's pipeline, uses 300+ KiB oversized SMEM to remove that cyclic lifetime,
and has now been made both runnable and faster than the exact one-CTA baseline
for topK 512–2048. The current candidate is `kernel_exp_dq_role_move.py`
(SHA256 `190cc27f7ddde3a712c319a2aaa7717301c29403bf85ec3332470659fe40d13c`):
it moves the unchanged dQ TMEM epilogue from spill-heavy math W4–W7 to idle
gather W0–W3 and hands over the 128-register allocation. See `README.md` for
the accepted source and measurements.

Rubin execution uses internal DSL `0.3.0+20260803235612.d88cc85`, CUDA 13.4, TVM-FFI, native `sm_107a`; the DSL
automatically selects `cudaFuncAttributeSharedMemoryMode=AllowOversized`. Maximum visible SMEM is 334,848 bytes.
The exact baseline SHA is `a86b353a2349962bd4404818a906ea2d4df4ca5cb88b22d15ea234fc4e8ff3d7`.

No NCU profiling unless the user explicitly asks. The final candidate must pass
the dynamic-length correctness gate even though intermediate performance trials
may skip correctness.
