# final_ser_kq6q H1 spill report

## Identity and protocol

- M0 source SHA256: `47f5d66492a864ce3a5efa4efbfde1e1aab3d4c2af3687babe4d8d1457b07e51`
- H1 commit: `a9c19261f0c3332747c3a9da15834968e6402b15`
- H1 source SHA256: `90610120873cebb9177e892e9cecf34df1adab52919961b5d65cdf7b1d8d708e`
- Shape: BF16, Sq=Skv=4096, H=128, D=512, topk=2048, sm_100a.
- Both builds were compiled with `CUTE_DSL_NO_CACHE=True`,
  `CUTE_DSL_KEEP=ptx,cubin`, and `CUTE_DSL_LINEINFO=True`, then disassembled
  with `nvdisasm -g -c` and parsed with `sass_spill_to_py_locs.py`.
- Exact artifacts are under `outputs/final_ser_kq6q_{m0,h1}_spill/`.

## Totals and verdict

| Build | STL | LDL (including LDL.LU) | Total local instructions | Locations |
|---|---:|---:|---:|---:|
| M0 | 16 | 23 | 39 | 21 |
| H1 | 16 | 21 | 37 | 18 |

H1 removes two loop-body reloads present in M0: `LDL [R1]` at M0 PC
`0x4b00` and `LDL [R1]` at M0 PC `0x5340`. These are the two dynamic
reloads associated with the runtime UMMA consumer mask and the long-lived
publication predicate. Static stores remain, and several other roles retain
their own local state. H1 therefore contains the intended spill antennas but
does not make the whole kernel spill-free.

The paired B200 result is wall-time neutral: H1 9.519888 ms versus M0
9.501616 ms across separate 24-pair ABBA runs. The source change is retained
as structural cleanup, not claimed as a standalone performance win.

## Every remaining H1 source location

The parser attributes all 37 local instructions; `unattributed=0`.

| Source line | Count | Mnemonics | Representative SASS |
|---:|---:|---|---|
| 407 | 16 | LDL 8, LDL.LU 2, STL 6 | `LDL R3,[R1+0x4]`; `LDL R116,[R1+0x4]` |
| 1191 | 3 | LDL 2, STL 1 | `LDL R5,[R1]`; `STL [R1+0x4],R4` |
| 6901 | 3 | LDL 1, LDL.LU 1, STL 1 | `LDL.LU R4,[R1+0x4]`; `STL [R1+0x4],R0` |
| 5855 | 1 | STL 1 | `STL [R1],R2` |
| 5958 | 1 | STL 1 | `STL [R1+0x4],R8` |
| 4880 | 1 | LDL.LU 1 | `LDL.LU R42,[R1+0x4]` |
| 1142 | 1 | STL 1 | `STL [R1+0x4],R33` |
| 6232 | 1 | STL 1 | `STL [R1+0x4],R13` |
| 6402 | 1 | LDL.LU 1 | `LDL.LU R40,[R1+0x4]` |
| 6398 | 1 | STL 1 | `STL [R1+0x4],R20` |
| 1719 | 1 | LDL 1 | `LDL R131,[R1+0x4]` |
| 1752 | 1 | LDL 1 | `LDL R0,[R1]` |
| 6904 | 1 | STL 1 | `@P0 STL [R1+0x4],R4` |
| 7132 | 1 | LDL 1 | `LDL R0,[R1+0x4]` |
| 7145 | 1 | STL 1 | `STL [R1+0x4],R0` |
| 7090 | 1 | LDL 1 | `LDL R0,[R1+0x4]` |
| 7054 | 1 | LDL 1 | `LDL R0,[R1+0x4]` |
| 7159 | 1 | STL 1 | `STL [R1+0x4],R3` |

Line attribution can land on a nearby DSL boundary rather than the exact
high-level value. The JSON artifacts retain full source context and every
instruction PC; they are authoritative for subsequent PC-level analysis.
