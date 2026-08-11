#!/usr/bin/env python3
"""kq2 desk probe: per-warp byte footprint of the P publish into p_xchg.

Question (kq4d strong form): the two NON-OWNER math warps of each CTA
publish P into the 4 KiB p_xchg via stmatrix through

    thread_copy_r2s = make_tiled_copy_D(
        StMatrix8x8x16bOp(transpose, num_matrices=4),
        make_tmem_copy(Ld16x256bOp(Rep(4)), t_score),
    ).get_slice(mtx)
    t_rs_p_xchg = thread_copy_r2s.partition_D(p_xchg_store)

where p_xchg_store views the buffer through score_store_layout.inner
(S<3,4,3> swizzle) over score_store_domain, with the base pointer
shifted by -n_owner*4096 B.  Is each writing warp's byte footprint a
CONTIGUOUS range (so a per-warp 2 KiB cp.async.bulk could send it
incrementally), and if not, what is the run structure (per warp, per
half-warp)?

Method (host-only, no GPU): reconstruct the exact kq2 objects inside a
@cute.jit trace (dsa_bwd_sm100_2cta_final_ser_kq2.py:4140-4162,
5217-5256, 5639-5657, 5857-5999), partition an identity tensor shaped
like score_store_domain per thread, evaluate the composed
swizzle-over-domain layout per coordinate at trace time (all static),
then analyse the byte sets in plain python.  Run:

    CUTE_DSL_ARCH=sm_100a python3 probe_kq2_pxchg_warp_footprint.py
"""

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warp

try:
    import cutlass.utils.blackwell_helpers as sm100_utils
except ImportError:
    from cutlass.utils import sm100_utils
import cutlass.utils as utils

# --- kq2 constants, verbatim (class + V2 overrides) ---
ELEM = cutlass.BFloat16
ACC = cutlass.Float32
H_TILE_CLUSTER = 128
H_TILE_CTA = 64
N_TILE = 64
N_TILE_CTA = 32
D_TILE_CLUSTER = 256
DQ_MMA_TILER = (D_TILE_CLUSTER, H_TILE_CLUSTER, N_TILE)
DKV_MMA_TILER = (256, 64, 64)  # V2 override
PDS_BLOCK_BYTES = 4096
ELEM_BYTES = 2
N_THREADS = 128  # mtx range of the 4 math warps

# trace-time capture (filled by the jit body, analysed after)
CAPTURE = {
    "domain_coords": {},   # t -> list of 32 domain coords (h,n)
    "pre_offsets": {},     # t -> list of 32 pre-swizzle element offsets
    "phys_offsets": {},    # t -> list of 32 post-swizzle element offsets
    "mma_first": {},       # (rank, t) -> (h, n) of score_coordinates[0]
    "swizzle_str": None,
    "store_layout_str": None,
}


def coord_to_py(c):
    if isinstance(c, tuple):
        return tuple(coord_to_py(x) for x in c)
    return int(c)


def py_swizzle_343(off):
    """Reference CuTe Swizzle<3,4,3> on an element offset (cross-check)."""
    yyy = (off >> (4 + 3)) & 0b111
    return off ^ (yyy << 4)


def capture_store_side(tiled_copy_r2s, ident, score_store_domain, comp):
    """Plain-python (trace-time) capture of every math thread's r2s D
    partition over the store domain.  Lives outside the @cute.jit body
    so the loops run natively with static python ints.  `comp` is the
    swizzle-composed domain (inner o 0 o score_store_domain): its
    evaluation is exactly the swizzled-pointer dereference offset of
    p_xchg_store (kq2:5959-5966)."""
    for t in range(N_THREADS):
        thr = tiled_copy_r2s.get_slice(t)
        tile = thr.partition_D(ident)[
            (None, None, None, None, 0)
        ]
        n_vals = cute.size(tile.shape)
        assert n_vals == N_TILE_CTA  # 32 elements per thread
        coords, pres, phys = [], [], []
        for i in range(n_vals):
            c = coord_to_py(tile[i])
            pre = int(cute.crd2idx(c, score_store_domain))
            ph = int(comp(c))
            # (h, n) from the domain coord: mode0 = ((h,1),(n0,n1),(1,1))
            (hm, nm, _o) = c[0]
            h = hm[0]
            n = nm[0] + 8 * nm[1]
            coords.append((h, n))
            pres.append(pre)
            phys.append(ph)
        CAPTURE["domain_coords"][t] = coords
        CAPTURE["pre_offsets"][t] = pres
        CAPTURE["phys_offsets"][t] = phys


def capture_mma_side(score_tiled_mma, score_copy):
    """Trace-time capture of score_coordinates[0] per (rank, thread)."""
    for rank in (0, 1):
        rank_score_mma = score_tiled_mma.get_slice(rank)
        rank_score_coordinates = rank_score_mma.partition_C(
            cute.make_identity_tensor(
                (H_TILE_CLUSTER, N_TILE)
            )
        )
        for t in range(N_THREADS):
            sc = score_copy.get_slice(t).partition_D(
                rank_score_coordinates
            )
            CAPTURE["mma_first"][(rank, t)] = coord_to_py(sc[0])


@cute.jit
def probe():
    cg2 = tcgen05.CtaGroup.TWO

    # ---- kq2 __call__ lines 462-470: the CG2 score MMA --------------
    score_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM,
        ELEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
        ACC,
        cg2,
        (H_TILE_CLUSTER, N_TILE),
    )
    # ---- kernel lines 5639-5657: fragment C -> t_score --------------
    score_c_layout = score_tiled_mma.make_fragment_C(
        score_tiled_mma.partition_shape_C(
            (H_TILE_CLUSTER, N_TILE)
        )
    ).layout
    tmem_ptr = cute.make_ptr(
        ACC, 0, cute.AddressSpace.tmem, assumed_align=1024
    )
    t_score = cute.make_tensor(tmem_ptr, score_c_layout)

    # ---- _make_score_tmem_load (kq2:4140-4162) -----------------------
    score_tmem_load = cute.make_copy_atom(
        tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
        ACC,
    )
    # ---- kernel 5857-5900: tmem copy -> store atom -> r2s tiled copy -
    score_copy = tcgen05.make_tmem_copy(score_tmem_load, t_score)
    smem_store_atom = sm100_utils.get_smem_store_op(
        utils.LayoutEnum.COL_MAJOR, ELEM, ACC, score_copy
    )
    assert isinstance(smem_store_atom.op, warp.StMatrix8x8x16bOp)
    assert smem_store_atom.op.num_matrices == 4
    print("store atom transpose:", smem_store_atom.op.transpose)
    tiled_copy_r2s = cute.make_tiled_copy_D(
        smem_store_atom, score_copy
    )

    # ---- kernel 5217-5256: score_store_layout / domain ----------------
    score_store_layout = sm100_utils.make_smem_layout_epi(
        ELEM,
        utils.LayoutEnum.COL_MAJOR,
        (H_TILE_CTA, N_TILE),
        1,
    )
    score_store_domain = cute.make_layout(
        (score_store_layout.outer.shape, 1, 1, 1),
        stride=(score_store_layout.outer.stride, 0, 0, 0),
    )
    CAPTURE["store_layout_str"] = str(score_store_layout)
    CAPTURE["swizzle_str"] = str(score_store_layout.inner)
    print("score_store_layout:", score_store_layout)
    print("score_store_domain:", score_store_domain)

    # kernel host asserts (5223-5252) reproduced
    dq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM,
        ELEM,
        OperandMajorMode.MN,
        OperandMajorMode.MN,
        ACC,
        cg2,
        DQ_MMA_TILER[:2],
    )
    dq_b_layout_staged = sm100_utils.make_smem_layout_b(
        dq_tiled_mma, DQ_MMA_TILER, ELEM, 1
    )
    dkv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        ELEM,
        ELEM,
        OperandMajorMode.MN,
        OperandMajorMode.K,
        ACC,
        cg2,
        DKV_MMA_TILER[:2],
    )
    dkv_b_layout_staged = sm100_utils.make_smem_layout_b(
        dkv_tiled_mma, DKV_MMA_TILER, ELEM, 1
    )
    assert cute.cosize(score_store_layout) == cute.cosize(
        dq_b_layout_staged
    )
    assert str(score_store_layout.inner) == str(
        dq_b_layout_staged.inner
    )
    assert str(score_store_layout.inner) == str(
        dkv_b_layout_staged.inner
    )
    assert cute.cosize(score_store_domain) == cute.cosize(
        dq_b_layout_staged
    )
    print("kernel host asserts reproduced OK")

    comp = cute.make_composed_layout(
        score_store_layout.inner, 0, score_store_domain
    )

    # ---- per-thread D partition of the store image --------------------
    ident = cute.make_identity_tensor(score_store_domain.shape)
    capture_store_side(
        tiled_copy_r2s, ident, score_store_domain, comp
    )

    # ---- mma coordinates (n_owner cross-check, kernel 5367-5374,5907) -
    capture_mma_side(score_tiled_mma, score_copy)


def runs_of(byte_set):
    """Sorted contiguous runs [(start, size_bytes), ...] of a byte set
    where each element covers ELEM_BYTES bytes."""
    offs = sorted(byte_set)
    runs = []
    start = prev = offs[0]
    for o in offs[1:]:
        if o == prev + ELEM_BYTES:
            prev = o
        else:
            runs.append((start, prev + ELEM_BYTES - start))
            start = prev = o
    runs.append((start, prev + ELEM_BYTES - start))
    return runs


def describe_runs(tag, runs):
    sizes = sorted({r[1] for r in runs})
    starts = [r[0] for r in runs]
    strides = sorted(
        {b - a for a, b in zip(starts, starts[1:])}
    )
    align16 = all(
        s % 16 == 0 and z % 16 == 0 for s, z in runs
    )
    print(
        f"  {tag}: runs={len(runs)} sizes={sizes} "
        f"start_strides={strides} first={starts[0]} last={starts[-1]} "
        f"16B_ok={align16}"
    )
    return runs


def analyse():
    print()
    print("=" * 72)
    print("ANALYSIS (pure python on captured static values)")
    print("=" * 72)
    swz = CAPTURE["swizzle_str"]
    print("swizzle:", swz)

    # cross-check the python S<3,4,3> reference against cute's own eval
    assert "S<3,4,3>" in CAPTURE["store_layout_str"], (
        "swizzle family changed; update py_swizzle_343"
    )
    mism = 0
    for t in range(N_THREADS):
        for pre, ph in zip(
            CAPTURE["pre_offsets"][t], CAPTURE["phys_offsets"][t]
        ):
            if py_swizzle_343(pre) != ph:
                mism += 1
    print("py S<3,4,3> reference vs cute eval mismatches:", mism)
    assert mism == 0

    # coverage sanity: 128 threads x 32 elems == full 4096-element image
    all_phys = set()
    for t in range(N_THREADS):
        all_phys.update(CAPTURE["phys_offsets"][t])
    assert all_phys == set(range(64 * 64)), "image not fully covered"
    print("full-image coverage: OK (4096 distinct elements)")

    # mma-coordinate n must equal domain n per thread-set half (warp
    # uniformity of n_owner), and rank1 h = rank0 h + 64.
    for w in range(4):
        n_halves = set()
        for t in range(32 * w, 32 * w + 32):
            for (h, n) in CAPTURE["domain_coords"][t]:
                n_halves.add(n // N_TILE_CTA)
        assert len(n_halves) == 1, f"warp {w} spans both n halves!"
    print("store-side n-half is warp-uniform: OK")

    print()
    print("per-warp coverage and n_owner:")
    warp_info = {}
    for w in range(4):
        hs, ns = set(), set()
        byte_set = set()
        for t in range(32 * w, 32 * w + 32):
            for (h, n) in CAPTURE["domain_coords"][t]:
                hs.add(h)
                ns.add(n)
            for ph in CAPTURE["phys_offsets"][t]:
                byte_set.add(ELEM_BYTES * ph)
        n_owner = min(ns) // N_TILE_CTA
        mma0 = CAPTURE["mma_first"][(0, 32 * w)]
        mma1 = CAPTURE["mma_first"][(1, 32 * w)]
        assert mma0[1] // N_TILE_CTA == n_owner
        assert mma1[1] // N_TILE_CTA == n_owner
        assert mma1[0] == mma0[0] + H_TILE_CTA
        warp_info[w] = (hs, ns, n_owner, byte_set)
        print(
            f"  mtx warp {w} (phys warp {w + 4}): "
            f"h=[{min(hs)},{max(hs)}] ({len(hs)} rows), "
            f"n=[{min(ns)},{max(ns)}] ({len(ns)} cols), "
            f"n_owner={n_owner}, "
            f"mma[0] rank0={mma0} rank1={mma1}, "
            f"bytes={len(byte_set) * ELEM_BYTES}"
        )

    print()
    print("=" * 72)
    print("(1)/(2) per-warp footprint inside the 4 KiB p_xchg")
    print("   (non-owner warps only; xchg base shift = -n_owner*4096 B)")
    print("=" * 72)
    verdicts = {}
    for rank in [0, 1]:
        non_owners = [
            w
            for w in range(4)
            if warp_info[w][2] != rank
        ]
        print(
            f"rank {rank}: non-owner (xchg-writing) mtx warps ="
            f" {non_owners} (phys {[w + 4 for w in non_owners]})"
        )
        for w in non_owners:
            hs, ns, n_owner, byte_set = warp_info[w]
            shifted = {
                b - n_owner * PDS_BLOCK_BYTES for b in byte_set
            }
            assert min(shifted) >= 0 and max(shifted) < PDS_BLOCK_BYTES
            runs = runs_of(shifted)
            describe_runs(f"warp {w} (h {min(hs)}-{max(hs)})", runs)
            contiguous = len(runs) == 1
            verdicts[(rank, w)] = (contiguous, runs)
            if contiguous:
                s, z = runs[0]
                print(
                    f"    -> CONTIGUOUS [{s}, {s + z}) "
                    f"({z} B)"
                )
            else:
                print(
                    f"    -> NOT contiguous: {len(runs)} runs"
                )

            # (3) half-warp granularity
            for half in range(2):
                hb = set()
                for t in range(
                    32 * w + 16 * half, 32 * w + 16 * half + 16
                ):
                    for ph in CAPTURE["phys_offsets"][t]:
                        hb.add(
                            ELEM_BYTES * ph
                            - n_owner * PDS_BLOCK_BYTES
                        )
                describe_runs(
                    f"  half-warp {w}.{half} (threads "
                    f"{16 * half}-{16 * half + 15})",
                    runs_of(hb),
                )

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    all_contig = all(v[0] for v in verdicts.values())
    for (rank, w), (contig, runs) in sorted(verdicts.items()):
        s, z = runs[0]
        tag = (
            f"[{s},{s + z}) {z}B"
            if contig
            else f"{len(runs)} runs of {sorted({r[1] for r in runs})}B"
        )
        print(
            f"  rank {rank} mtx-warp {w}: "
            f"{'CONTIGUOUS ' + tag if contig else 'FRAGMENTED ' + tag}"
        )
    print(
        "STRONG-FORM kq4d (per-warp 2KiB cp.async.bulk):",
        "LEGAL" if all_contig else "ILLEGAL",
    )

    # warp-pair sanity (both writers together == the whole 4 KiB)
    for rank in [0, 1]:
        pair = set()
        for w in range(4):
            if warp_info[w][2] != rank:
                pair.update(
                    b - warp_info[w][2] * PDS_BLOCK_BYTES
                    for b in warp_info[w][3]
                )
        assert pair == set(range(0, PDS_BLOCK_BYTES, ELEM_BYTES))
    print(
        "warp-PAIR granularity: exact contiguous [0, 4096) "
        "(both ranks) -- verified"
    )

    # full run lists for the two h-half shapes (rank 0 writers)
    print()
    print("full run lists (rank 0 writers), [start, size]B:")
    for w in (2, 3):
        hs, ns, n_owner, byte_set = warp_info[w]
        shifted = {
            b - n_owner * PDS_BLOCK_BYTES for b in byte_set
        }
        print(
            f"  warp {w} (h {min(hs)}-{max(hs)}):",
            runs_of(shifted),
        )


def main():
    cute.compile(probe)
    analyse()
    print("PROBE DONE")


if __name__ == "__main__":
    main()
