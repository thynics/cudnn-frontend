#!/usr/bin/env python3
"""v_gpt desk probe: two zero-copy dual-view legality questions, CPU-only.

No GPU, no pipeline lock.  Reuses gate_0b1_host_assert's calibrated
indexer/sweep machinery and the ported UMMA descriptor rules
(utils/sm100/mma_desc.py).  Run inside the container from the repo root:

    python3 python/cudnn/deepseek_sparse_attention/sparse_attention_backward/\
probe_vgpt_dualview.py

S1 (keystone -- panel -> dkv-A zero-copy view):
    Can the CG2 dV/dK MMA's A operand ([D128 x H64] MN-major, the ring
    quadrant layout dkv_a_layout_staged) be a ZERO-COPY view into the
    token-resident stationary panel (stationary_a_layout_staged,
    [H64 x D512] K-major SW128, 1 stage)?  If yes for all
    (round r, rank rk) windows at a constant element offset, the three
    own-h panel generations per tile can become descriptor views: no
    S2S copy, no ring slot, no credit -- generation deletion at zero
    bytes.  (Precedent: the baseline ships the sK/sK_2 K-major-B /
    MN-major-A dual view of one buffer; vk_2 itself asserts
    stationary.inner == score_a.inner.)

    Identity: exists const o(r, rk) s.t. for all h in [0,64), dl in [0,128):
        idx_panel(h, 256*r + 128*rk + dl) == idx_dkvA(dl, h) + o(r, rk)

S2 (K2 own-half alias, still-open question from K2_DESK_NOTES):
    Can each D128 chunk of the score-B staging (score_kv bytes,
    [N64-own-half? x D128] K-major x4 chunks) serve as the MN(D)-major
    dQ-A operand window?  Identity per chunk c:
        idx_scoreB(n, d128, c) == idx_dqA(128*?; see sweep) + o(c)
    plus mma_desc Major.MN canonical-form acceptance of the window.

Interpretation:
  - S1 PASS  -> own-h panel gens deletable as views (funds the 8->4
    milestone and, via the freed ring slot, kscore depth-2).
  - S1 FAIL  -> own-h gens keep the S2S copy; deletion needs owner-push
    restructuring instead.
  - S2 PASS  -> K2 buildable-form's own-half alias leg is legal.
  - S2 FAIL  -> K2 falls back to S2S copy (already-booked tax).
Controls: P (self-identity, must pass), N (shifted coord, must fail).
"""

import sys
import traceback

import cutlass  # noqa: F401
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import BFloat16, Float32

try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except ImportError:
    from cutlass.utils import sm100_utils

import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "utils", "sm100"),
)

from gate_0b1_host_assert import (  # noqa: E402
    make_a_indexer,
    make_b_indexer,
    sweep,
)

# --- vk_2 constants, verbatim ---
H_TILE_CLUSTER = 128
H_TILE_CTA = 64
N_TILE = 64
K_CHUNK = 128
K_CHUNKS = 4
D_HEAD = 512
D_TILE_CLUSTER = 256
D_TILE_CTA = 128
# V2 override (vk_2): two-pass H reduction -- K = H64 per issue, NOT the
# base class's H128.  r1 of this probe used the base value by mistake;
# the 50%-mismatch-at-clean-offsets signature was the wrong M-pair
# stride (8192 vs 4096), not a real verdict.
DKV_MMA_TILER = (D_TILE_CLUSTER, N_TILE, 64)
DQ_MMA_TILER = (D_TILE_CLUSTER, H_TILE_CLUSTER, N_TILE)
SCORE_TILER = (H_TILE_CLUSTER, N_TILE, K_CHUNK)
STATIONARY_TILER = (H_TILE_CTA, N_TILE, D_HEAD)


def build_layouts():
    cg1 = tcgen05.CtaGroup.ONE
    cg2 = tcgen05.CtaGroup.TWO

    stationary_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16,
        OperandMajorMode.K, OperandMajorMode.K,
        Float32, cg1, STATIONARY_TILER[:2],
    )
    dkv_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16,
        OperandMajorMode.MN, OperandMajorMode.K,
        Float32, cg2, DKV_MMA_TILER[:2],
    )
    dq_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16,
        OperandMajorMode.MN, OperandMajorMode.MN,
        Float32, cg2, DQ_MMA_TILER[:2],
    )
    score_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16,
        OperandMajorMode.K, OperandMajorMode.K,
        Float32, cg2, SCORE_TILER[:2],
    )

    L_panel = sm100_utils.make_smem_layout_a(
        stationary_mma, STATIONARY_TILER, BFloat16, 1,
    )
    L_dkva = sm100_utils.make_smem_layout_a(
        dkv_mma, DKV_MMA_TILER, BFloat16, 1,
    )
    L_dqa = sm100_utils.make_smem_layout_a(
        dq_mma, DQ_MMA_TILER, BFloat16, 1,
    )
    L_scoreb = sm100_utils.make_smem_layout_b(
        score_mma, SCORE_TILER, BFloat16, K_CHUNKS,
    )
    return L_panel, L_dkva, L_dqa, L_scoreb


def body():
    L_panel, L_dkva, L_dqa, L_scoreb = build_layouts()
    print("L_panel  (stationary_a):", L_panel)
    print("L_dkva   (dkv_a)       :", L_dkva)
    print("L_dqa    (dq_a)        :", L_dqa)
    print("L_scoreb (score_b)     :", L_scoreb)
    assert "((64,2),16),1,4,1" in str(L_dqa), "dq_a calibration vs R4"
    assert ",8192)" not in str(L_dkva), "dkv_a still on base K=128 tiler"

    idx_panel = make_a_indexer(L_panel)   # (h, d, stage)
    idx_dkva = make_a_indexer(L_dkva)     # (d_local, h, stage)
    idx_dqa = make_a_indexer(L_dqa)       # (d_local, n, stage)
    idx_scoreb = make_b_indexer(L_scoreb)  # (n, d128, chunk-as-stage)

    ok = True

    # ---- controls --------------------------------------------------
    pts_p = [((h, d, 0), (h, d, 0)) for h in range(0, 64, 7) for d in range(0, 512, 31)]
    ok &= sweep("S0/P panel self", idx_panel, idx_panel, pts_p) == 0
    pts_n = [((h, d, 0), ((h + 1) % 64, d, 0)) for h in range(0, 64, 7) for d in range(0, 512, 31)]
    ok &= sweep("S0/N panel shifted (must mismatch)", idx_panel, idx_panel, pts_n) > 0

    # ---- S1: panel window == dkv-A quadrant, per (round, rank) -----
    s1_all = True
    for r in range(2):
        for rk in range(2):
            d0 = 256 * r + 128 * rk
            base = idx_panel(0, d0, 0) - idx_dkva(0, 0, 0)
            pts = [
                ((h, d0 + dl, 0), (dl, h, 0))
                for h in range(64)
                for dl in range(128)
            ]
            mism = sweep(
                f"S1 r={r} rk={rk} (offset {base})",
                idx_panel,
                lambda dl, h, s, _b=base: idx_dkva(dl, h, s) + _b,
                pts,
            )
            s1_all &= mism == 0
    print("PROBE_S1_PANEL_DKVA_VIEW:", "PASS" if s1_all else "FAIL")
    ok &= s1_all

    # ---- S2: score-B chunk window == dq-A D128 window --------------
    # dq-A per-CTA M covers D256 cluster / rank half; its (d_local, n)
    # window for chunk c should equal score-B chunk c bytes.
    s2_all = True
    for c in range(K_CHUNKS):
        base = idx_scoreb(0, 0, c) - idx_dqa(0, 0, 0)
        pts = [
            ((n, d, c), (d, n, 0))
            for n in range(N_TILE)
            for d in range(K_CHUNK)
        ]
        mism = sweep(
            f"S2 chunk={c} (offset {base})",
            idx_scoreb,
            lambda d, n, s, _b=base: idx_dqa(d, n, s) + _b,
            pts,
        )
        s2_all &= mism == 0
    print("PROBE_S2_SCOREB_DQA_VIEW:", "PASS" if s2_all else "FAIL")
    ok &= s2_all

    print("PROBE_VGPT_DUALVIEW_RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    try:
        body()
    except Exception as e:  # noqa: BLE001
        print("DIRECT_MODE_FAILED:", repr(e))
        traceback.print_exc()
        print("retrying under @cute.jit trace context ...")
        try:
            cute.jit(body)()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("PROBE_VGPT_DUALVIEW_RESULT: ERROR")
            sys.exit(2)
