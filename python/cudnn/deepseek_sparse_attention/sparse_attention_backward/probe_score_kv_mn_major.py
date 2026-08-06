"""K2 desk probe: can score_kv's bytes serve as an MN-major dQ-A descriptor?

CPU-only (no GPU, no pipeline lock): replicates the host-side layout
construction of vk_2 verbatim, then asks the ported UMMA descriptor
builder (utils/sm100/mma_desc.py) whether each D128 chunk of the score-B
staging admits a legal MN(D)-major smem descriptor -- i.e., whether K2's
own-half alias is buildable, or whether it must fall back to the
gather-warp S2S copy (+2 ring gens).

Run inside the container from the repo root:

    python3 python/cudnn/deepseek_sparse_attention/sparse_attention_backward/\
probe_score_kv_mn_major.py

Interpretation:
  - "Major.K  (natural view): OK"   -> tool calibration (must pass);
  - "Major.MN chunk c: OK"          -> alias legal for that chunk;
  - "Major.MN chunk c: ValueError"  -> alias illegal -> S2S fallback.
The dq_a reference section prints what a known-legal MN-major A staging
looks like (the kdq images' own layout) for side-by-side comparison.
"""

import traceback

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05  # noqa: F401
import cutlass.utils as utils  # noqa: F401

try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except ImportError:
    from cutlass.utils import sm100_utils

import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "utils", "sm100"),
)
import mma_desc  # noqa: E402

# --- vk_2 constants, verbatim ---
H_TILE_CLUSTER = 128
N_TILE = 64
K_CHUNK = 128
K_CHUNKS = 4  # D_HEAD // K_CHUNK
D_TILE_CLUSTER = 256
D_TILE_CTA = 128
DQ_MMA_TILER = (D_TILE_CLUSTER, H_TILE_CLUSTER, N_TILE)
ELEM = cutlass.BFloat16
ACC = cutlass.Float32


@cute.jit
def probe():
    cg2 = tcgen05.CtaGroup.TWO
    score_tiler = (H_TILE_CLUSTER, N_TILE, K_CHUNK)
    score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM, ELEM,
        OperandMajorMode.MN,   # A (Qt panel), as in vk_2 L470+
        OperandMajorMode.K,    # B (K rows, D contiguous)
        ACC, cg2, score_tiler[:2],
    )
    dq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM, ELEM,
        OperandMajorMode.MN,   # A = Kt, D(M)-major
        OperandMajorMode.MN,   # B = dSt
        ACC, cg2, DQ_MMA_TILER[:2],
    )
    score_b_layout_staged = sm100_utils.make_smem_layout_b(
        score_tiled_mma, score_tiler, ELEM, K_CHUNKS,
    )
    dq_a_layout_staged = sm100_utils.make_smem_layout_a(
        dq_tiled_mma, DQ_MMA_TILER, ELEM, 1,
    )
    cute.printf("score_b_staged outer: {}", score_b_layout_staged.outer)
    cute.printf("score_b_staged inner: {}", score_b_layout_staged.inner)
    cute.printf("dq_a_staged     outer: {}", dq_a_layout_staged.outer)
    cute.printf("dq_a_staged     inner: {}", dq_a_layout_staged.inner)


def main():
    # Host-side (non-JIT) reconstruction for the descriptor math: the
    # mma_desc helpers operate on plain cute layouts.
    cg2 = tcgen05.CtaGroup.TWO
    score_tiler = (H_TILE_CLUSTER, N_TILE, K_CHUNK)
    score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM, ELEM, OperandMajorMode.MN, OperandMajorMode.K,
        ACC, cg2, score_tiler[:2],
    )
    dq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM, ELEM, OperandMajorMode.MN, OperandMajorMode.MN,
        ACC, cg2, DQ_MMA_TILER[:2],
    )
    score_b = sm100_utils.make_smem_layout_b(
        score_tiled_mma, score_tiler, ELEM, K_CHUNKS,
    )
    dq_a = sm100_utils.make_smem_layout_a(
        dq_tiled_mma, DQ_MMA_TILER, ELEM, 1,
    )
    print("score_b_layout_staged:", score_b)
    print("dq_a_layout_staged  :", dq_a)

    swz_b = score_b.inner
    outer_b = score_b.outer
    print("score_b swizzle:", swz_b)

    # Calibration: the natural K-major reading must be legal.
    try:
        # slice one chunk: modes assumed ((n...),(d...),chunk) -- print
        # first, adapt indexing from the printed shape if this throws.
        chunk0 = cute.slice_(outer_b, (None, None, 0))
        base = mma_desc.make_smem_desc_base(
            chunk0, swz_b, mma_desc.Major.K
        )
        print("Major.K  (natural view): OK  base=0x%x" % base)
    except Exception:
        print("Major.K  (natural view): FAILED (calibration!)")
        traceback.print_exc()

    # The question: each D128 chunk read MN(D)-major as dQ-A.
    for chunk in range(K_CHUNKS):
        try:
            sl = cute.slice_(outer_b, (None, None, chunk))
            # transpose the (n, d) modes to the A view's (d, n)
            transposed = cute.select(sl, mode=[1, 0])
            base = mma_desc.make_smem_desc_base(
                transposed, swz_b, mma_desc.Major.MN
            )
            print(
                "Major.MN chunk %d: OK  base=0x%x  -> alias LEGAL"
                % (chunk, base)
            )
        except ValueError as err:
            print(
                "Major.MN chunk %d: ValueError (%s) -> alias ILLEGAL,"
                " S2S fallback" % (chunk, err)
            )
        except Exception:
            print("Major.MN chunk %d: unexpected failure:" % chunk)
            traceback.print_exc()

    # Reference: what a known-legal MN-major A looks like.
    try:
        dq_sl = cute.slice_(dq_a.outer, (None, None, 0))
        base = mma_desc.make_smem_desc_base(
            dq_sl, dq_a.inner, mma_desc.Major.MN
        )
        print("dq_a reference (Major.MN): OK  base=0x%x" % base)
    except Exception:
        print("dq_a reference: FAILED")
        traceback.print_exc()


if __name__ == "__main__":
    main()
