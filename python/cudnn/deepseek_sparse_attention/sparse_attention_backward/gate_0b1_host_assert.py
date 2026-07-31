#!/usr/bin/env python3
"""Gate [0b-1]: CG1 transpose-view descriptor legality -- zero-GPU host assert.

Question: can the P1 CG1 dV/dK MMA's A operand be a ZERO-COPY transposed view
([D=512 x H=64], MN-major) of the v17a-resident stationary Q/dO buffer
([H=64 x D=512], K-major, SW128, 1 stage)?

Byte-map identity to prove (same physical smem bytes, two descriptors):
    idx_stat(h, d) == idx_qt(d, h)   for all h in [0,64), d in [0,512)

where
  L_stat = make_smem_layout_a(CG1 mma(A=K-major,(64,64)),  tiler (64,64,512))
           -- exactly v17a stationary_a_layout_staged (dsa_bwd_sm100_2cta_v17a.py:377-445)
  L_qt   = make_smem_layout_a(CG1 mma(A=MN-major,(128,64)), tiler (512,64,64))
           -- exactly baseline QT/dOT view layout (dsa_bwd_sm100.py:308-314,343,357)

Controls:
  P1 (positive, known-true): v17a score chunk view: 4-stage [64x128] K-major over the
     same buffer (v17a:434-445,514-518 + numerically shipped) ->
     idx_stat(h,d) == idx_score(h, d%128, stage=d//128)
  P2 (baseline replica): L_stat576 ([64x576] K-major = baseline sQ) vs L_qt:
     idx_stat576(h,d) == idx_qt(d,h) for d<512 (the numerically shipped baseline pair,
     dsa_bwd_sm100.py:838-842)
  N1 (negative): idx_stat(h,d) vs idx_qt(d,(h+1)%64) must produce many mismatches
     (comparator sanity).
"""

import sys
import traceback

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import BFloat16, Float32

H = 64
D = 512
D576 = 576
CHUNK = 128


def isz(shape):
    """Generic size of an int-or-nested-tuple shape."""
    if isinstance(shape, (int,)):
        return shape
    try:
        n = 1
        for s in shape:
            n *= isz(s)
        return n
    except TypeError:
        return int(shape)


def hier_coord(x, shape):
    """Colexicographic hierarchical coordinate of linear x within shape."""
    if isinstance(shape, int):
        return x
    try:
        it = list(shape)
    except TypeError:
        return x
    coords = []
    for s in it:
        n = isz(s)
        coords.append(hier_coord(x % n, s))
        x //= n
    return tuple(coords)


def layout_index(L, coord):
    """Physical offset (elements) of hierarchical coord under (possibly composed) layout L."""
    errs = []
    for fn in (
        lambda: L(coord),
        lambda: cute.crd2idx(coord, L),
    ):
        try:
            return int(fn())
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
    # fallback: outer layout then swizzle
    try:
        off = int(cute.crd2idx(coord, L.outer))
        inner = L.inner
        try:
            return int(inner(off))
        except Exception:  # noqa: BLE001
            return int(cute.crd2idx(off, inner))
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
        raise RuntimeError("layout_index failed: " + " | ".join(errs))


def make_a_indexer(L):
    """(m, k, stage) -> physical element offset, for a staged smem A layout.

    Expected outer profile ((bM, bK), restM, restK, STAGE) per partition_shape_A.
    Handles nested atom modes generically.
    """
    shp = L.outer.shape if hasattr(L, "outer") else L.shape
    atom = shp[0]
    a_m, a_k = atom[0], atom[1]
    bM, bK = isz(a_m), isz(a_k)
    rM, rK = shp[1], shp[2]

    def idx(m, k, stage=0):
        cm = hier_coord(m % bM, a_m)
        ck = hier_coord(k % bK, a_k)
        coord = (
            (cm, ck),
            hier_coord(m // bM, rM),
            hier_coord(k // bK, rK),
            stage,
        )
        return layout_index(L, coord)

    return idx


def sweep(name, fa, fb, coords, max_report=5):
    mism = 0
    samples = []
    for (ma, ka, sa), (mb, kb, sb) in coords:
        ia = fa(ma, ka, sa)
        ib = fb(mb, kb, sb)
        if ia != ib:
            mism += 1
            if len(samples) < max_report:
                samples.append(((ma, ka, sa), ia, (mb, kb, sb), ib))
    print(f"[{name}] points={len(coords)} mismatches={mism}")
    for s in samples:
        print(f"    sample mismatch: A{s[0]} -> {s[1]}  vs  B{s[2]} -> {s[3]}")
    return mism


def make_b_indexer(L):
    """(n, k, stage) -> physical element offset, for a staged smem B layout.

    Expected outer profile ((bN, bK), restN, restK, STAGE) per
    partition_shape_B.  Same generic machinery as make_a_indexer.
    """
    shp = L.outer.shape if hasattr(L, "outer") else L.shape
    atom = shp[0]
    a_n, a_k = atom[0], atom[1]
    bN, bK = isz(a_n), isz(a_k)
    rN, rK = shp[1], shp[2]

    def idx(n, k, stage=0):
        cn = hier_coord(n % bN, a_n)
        ck = hier_coord(k % bK, a_k)
        coord = (
            (cn, ck),
            hier_coord(n // bN, rN),
            hier_coord(k // bK, rK),
            stage,
        )
        return layout_index(L, coord)

    return idx


def body_0c1():
    """Gate [0c-1] (A1 host assert A): score-B panel / dQ-A dual view.

    The A1 kernel gathers the full [N64, D512] K panel once as the
    K-major score B (make_smem_layout_b(QK CG1 (64,64), (64,64,512)))
    and re-views the SAME bytes as the MN-major dQ A operand
    (make_smem_layout_a(KdS CG1 (128,64) MNxMN, (512,64,64))) -- the
    baseline sK/sK_2 pair (dsa_bwd_sm100.py:844-845).  Identity:
        idx_b(n, d) == idx_a(d, n)  for all n in [0,64), d in [0,512).
    Failure downgrades A1 to the kdq streaming fallback (chair spec).
    """

    cg1 = tcgen05.CtaGroup.ONE
    score_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, cg1, (H, 64),
    )
    L_kv_b = sm100_utils.make_smem_layout_b(
        score_mma, (H, 64, D), BFloat16, 1
    )
    kds_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.MN, OperandMajorMode.MN,
        Float32, cg1, (128, 64),
    )
    L_k2 = sm100_utils.make_smem_layout_a(kds_mma, (D, 64, H), BFloat16, 1)

    print("L_kv_b:", L_kv_b)
    print("L_k2  :", L_k2)
    inner_eq = str(L_kv_b.inner) == str(L_k2.inner)
    print("inner(kv_b)==inner(k2):", inner_eq,
          "|", L_kv_b.inner, "vs", L_k2.inner)
    print("cosize kv_b:", cute.cosize(L_kv_b), " k2:", cute.cosize(L_k2))

    idx_b = make_b_indexer(L_kv_b)
    idx_a = make_a_indexer(L_k2)

    main = sweep(
        "MAIN kv_b(n,d)==k2(d,n) [gate 0c-1]",
        idx_b,
        idx_a,
        [((n, d, 0), (d, n, 0)) for n in range(64) for d in range(D)],
    )
    neg = sweep(
        "N1 kv_b(n,d)==k2(d,(n+1)%64) (must FAIL)",
        idx_b,
        idx_a,
        [((n, d, 0), (d, (n + 1) % 64, 0))
         for n in range(64) for d in range(0, D, 7)],
    )
    ok = (main == 0) and (neg > 0)
    print()
    print("GATE_0C1_MAIN:", "OK" if main == 0 else "FAILED")
    print("GATE_0C1_N1_NEGATIVE:",
          "OK" if neg > 0 else "FAILED(comparator-blind)")
    print("GATE_0C1_RESULT:", "PASS" if ok else "FAIL")
    return ok


def body():
    print("cutlass module:", cutlass.__file__)
    cg1 = tcgen05.CtaGroup.ONE

    # --- v17a stationary (K-major CG1), exact reproduction of v17a:372-385,440-445
    stat_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, cg1, (H, 64),
    )
    L_stat = sm100_utils.make_smem_layout_a(stat_mma, (H, 64, D), BFloat16, 1)

    # --- baseline sQ shape replica ([64 x 576] K-major, dsa_bwd_sm100.py:298-300,335)
    L_stat576 = sm100_utils.make_smem_layout_a(stat_mma, (H, 64, D576), BFloat16, 1)

    # --- proposed CG1 transpose view (MN-major), exact reproduction of
    #     baseline QdS/dOP path (dsa_bwd_sm100.py:308-314,343,357)
    qt_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.MN, OperandMajorMode.K,
        Float32, cg1, (128, 64),
    )
    L_qt = sm100_utils.make_smem_layout_a(qt_mma, (D, 64, H), BFloat16, 1)

    # --- v17a score chunk view (CG2 K-major staged x4), v17a:405-413,434-439
    cg2 = tcgen05.CtaGroup.TWO
    score_mma = sm100_utils.make_trivial_tiled_mma(
        BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K,
        Float32, cg2, (2 * H, 64),
    )
    L_score = sm100_utils.make_smem_layout_a(
        score_mma, (2 * H, 64, CHUNK), BFloat16, D // CHUNK
    )

    print("L_stat   :", L_stat)
    print("L_stat576:", L_stat576)
    print("L_qt     :", L_qt)
    print("L_score  :", L_score)

    inner_eq_stat_qt = str(L_stat.inner) == str(L_qt.inner)
    inner_eq_stat_score = str(L_score.inner) == str(L_stat.inner)
    print("inner(stat)==inner(qt):   ", inner_eq_stat_qt,
          "|", L_stat.inner, "vs", L_qt.inner)
    print("inner(stat)==inner(score):", inner_eq_stat_score)

    idx_stat = make_a_indexer(L_stat)
    idx_stat576 = make_a_indexer(L_stat576)
    idx_qt = make_a_indexer(L_qt)
    idx_score = make_a_indexer(L_score)

    # cosize sanity
    print("cosize stat:", cute.cosize(L_stat), " qt:", cute.cosize(L_qt),
          " score:", cute.cosize(L_score), " stat576:", cute.cosize(L_stat576))

    all_hd = [((h, d, 0), (h, d, 0)) for h in range(H) for d in range(D)]

    # P1 positive control: stat vs score chunk view (known-true from shipped v17a)
    p1 = sweep(
        "P1 stat==score-chunk (known-true control)",
        idx_stat,
        idx_score,
        [((h, d, 0), (h, d % CHUNK, d // CHUNK)) for h in range(H) for d in range(D)],
    )

    # MAIN: stat (h,d) == qt (d,h)
    main = sweep(
        "MAIN stat(h,d)==qt(d,h) [gate 0b-1]",
        idx_stat,
        idx_qt,
        [((h, d, 0), (d, h, 0)) for h in range(H) for d in range(D)],
    )

    # P2 baseline replica: stat576 (h,d) == qt (d,h) for d<512
    p2 = sweep(
        "P2 stat576(h,d)==qt(d,h), d<512 (baseline pair replica)",
        idx_stat576,
        idx_qt,
        [((h, d, 0), (d, h, 0)) for h in range(H) for d in range(D)],
    )

    # N1 negative control: must mismatch
    n1 = sweep(
        "N1 stat(h,d)==qt(d,(h+1)%64) (must FAIL)",
        idx_stat,
        idx_qt,
        [((h, d, 0), (d, (h + 1) % H, 0)) for h in range(H) for d in range(0, D, 7)],
    )

    ok = (p1 == 0) and (main == 0) and (p2 == 0) and (n1 > 0)
    print()
    print("GATE_0B1_P1_CONTROL:", "OK" if p1 == 0 else "FAILED")
    print("GATE_0B1_MAIN:", "OK" if main == 0 else "FAILED")
    print("GATE_0B1_P2_BASELINE_REPLICA:", "OK" if p2 == 0 else "FAILED")
    print("GATE_0B1_N1_NEGATIVE:", "OK" if n1 > 0 else "FAILED(comparator-blind)")
    print("GATE_0B1_RESULT:", "PASS" if ok else "FAIL")


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
            print("GATE_0B1_RESULT: ERROR")
            sys.exit(2)
    try:
        body_0c1()
    except Exception as e:  # noqa: BLE001
        print("DIRECT_MODE_FAILED (0c1):", repr(e))
        traceback.print_exc()
        print("retrying under @cute.jit trace context ...")
        try:
            cute.jit(body_0c1)()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print("GATE_0C1_RESULT: ERROR")
            sys.exit(2)
