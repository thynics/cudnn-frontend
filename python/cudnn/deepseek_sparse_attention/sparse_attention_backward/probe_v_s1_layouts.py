"""v_s1 desk probe: layout algebra behind the dO chunk stream.

r2 (post-mortem of dsa-vs1-r1-1786082768): every cute layout/MMA
construction needs an active MLIR trace context -- bare host calls die
with "Expected an MLIR object (got None)" inside make_trivial_tiled_mma
(_pack_x -> MakeShapeOp with no insertion point; verified against the
nvidia-cutlass-dsl-libs-base 4.5.0 sources).  The probe body therefore
lives in a @cute.jit function driven by cute.compile: the asserts are
plain Python on static values, so they fire at trace time; no kernel is
launched and no GPU is required (arch falls back to CUTE_DSL_ARCH or
the sm_100 default).

Asserted identities (the ones the v_s1 kernel relies on):

1. family identity: the 2-stage stream member's per-stage layout equals
   the 4-stage score family's per-stage layout (outer AND swizzle) --
   the chunk TMA atom is built from the latter, the staging view from
   the former, so their byte orders must coincide;
2. size ledger: 2-stage cosize 16384 elements (32 KiB), per-stage
   16 KiB == score_a_stage_bytes == the per-chunk expect_tx.

The chunk TMA atom itself is the v0-silicon-proven construction
(score_a_layout box + score_tiler + dp CG2 mma + cluster shape,
dsa_bwd_sm100_2cta_v0.py:478-485); its gmem-side partition shape is
exercised by the compile leg, not here (atom construction needs the
gmem tensor).

Run inside the container from the repo root:

    python3 python/cudnn/deepseek_sparse_attention/sparse_attention_backward/\
probe_v_s1_layouts.py

Expected output ends with "ALL PROBES PASS".
"""

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05  # noqa: F401

try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except ImportError:
    from cutlass.utils import sm100_utils

# --- v_s1 constants, verbatim ---
H_TILE_CLUSTER = 128
N_TILE = 64
K_CHUNK = 128
K_CHUNKS = 4  # D_HEAD // K_CHUNK
ELEM = cutlass.BFloat16
ACC = cutlass.Float32


@cute.jit
def probe():
    cg2 = tcgen05.CtaGroup.TWO
    score_tiler = (H_TILE_CLUSTER, N_TILE, K_CHUNK)
    score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM,
        ELEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
        ACC,
        cg2,
        score_tiler[:2],
    )
    score_a_layout_staged = sm100_utils.make_smem_layout_a(
        score_tiled_mma,
        score_tiler,
        ELEM,
        K_CHUNKS,
    )
    stream_a_layout_staged = sm100_utils.make_smem_layout_a(
        score_tiled_mma,
        score_tiler,
        ELEM,
        2,
    )
    score_a_layout = cute.select(score_a_layout_staged, mode=[0, 1, 2])
    stream_chunk = cute.select(stream_a_layout_staged, mode=[0, 1, 2])
    score_a_stage_bytes = cute.size_in_bytes(ELEM, score_a_layout)

    # Trace-time record (plain prints run once, during tracing).
    print("score_a_staged :", score_a_layout_staged)
    print("stream_a_staged:", stream_a_layout_staged)
    print("per-stage score:", score_a_layout)
    print("per-stage strm :", stream_chunk)
    print("stage bytes    :", score_a_stage_bytes)

    assert cute.cosize(stream_a_layout_staged) == 16384, (
        "stream staging must be 2 x 8192 elements (32 KiB)"
    )
    assert (
        stream_a_layout_staged.inner == score_a_layout_staged.inner
    ), "stream/score swizzle families diverged"
    assert cute.size_in_bytes(ELEM, stream_chunk) == score_a_stage_bytes, (
        "per-stage byte counts diverged"
    )
    assert str(stream_chunk) == str(score_a_layout), (
        "per-stage layouts diverged: the chunk TMA box (built from the"
        " score family) would no longer match the stream staging view"
    )


def main():
    cute.compile(probe)
    print("ALL PROBES PASS")


if __name__ == "__main__":
    main()
