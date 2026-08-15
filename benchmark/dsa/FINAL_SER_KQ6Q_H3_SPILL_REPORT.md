# final_ser_kq6q H3 P-stream SASS and spill report

## Identity and protocol

- H3 commit: `865eaf2eec3fe665a0d74b0612e0416687b528af`.
- Release SHA256: `29afe306b22c5a70f8c6cf1a2cb1d524d588b68abb52320f519f1985eab3e0ee`.
- Trace-twin SHA256: `bc97a4def7739b1e33f201889ac06456ee013e05442ac0ebdb799db99f110751`.
- Shape: BF16, Sq=Skv=4096, H=128, D=512, topk=2048, sm_100a.
- The detached B200 worktree was compiled with `CUTE_DSL_NO_CACHE=True`,
  `CUTE_DSL_KEEP=ptx,cubin`, and `CUTE_DSL_LINEINFO=True`; the exact CUBIN was
  disassembled with `nvdisasm -g -c` and parsed with
  `sass_spill_to_py_locs.py`.
- Exact artifacts are under `outputs/final_ser_kq6q_h3_pstream_spill/`.

The exact source identity gates the capture because stale CUBIN PCs cannot
prove the changed P publication schedule. Next: reject any artifact whose
source manifest differs from the commit and SHA above.

## Code-generation verdict

H3 preserves one logical four-STSM publication per ownership path because the
local and exchange destinations are mutually exclusive. Next: count dynamic
paths rather than treating the eight static ownership-arm stores as eight
executed stores.

- Head math is PCs `0x4a80..0x4d50`: 8 `FFMA2`, 16 `MUFU.EX2`, and 8
  `F2FP.BF16.F32.PACK_AB`.
- The first two dynamic stores are either local PCs `0x4d70,0x4d80` or exchange
  PCs `0x4da0,0x4db0`.
- Tail math follows the first store pair: another 8 `FFMA2`, 16 `MUFU.EX2`,
  and 8 `F2FP.BF16.F32.PACK_AB` occur before or around the tail pair.
- The compiler interleaves the first tail store at PC `0x4f40` with the final
  six `MUFU.EX2` and four pack instructions; the remaining tail stores are PCs
  `0x5000..0x5020`. This proves that source-level streaming survived codegen.
- One close post-dominates both ownership paths: `MEMBAR.ALL.CTA` at `0x5070`,
  `FENCE.VIEW.ASYNC.S` at `0x5080`, and elected `SYNCS.ARRIVE` at `0x50a0`.

The P hot region contains no new `STL` or `LDL` because the only P-loop local
instructions remain the pre-existing state store at source line 6398 and
parity-copy reload at line 6402. Next: decide H3 by same-process H1/H3 wall
time and fresh SMART evidence, not by source order alone.

## Authoritative spill totals

| Build | STL | LDL (including LDL.LU) | Combined | Locations | Unattributed |
|---|---:|---:|---:|---:|---:|
| H1 | 16 | 21 | 37 | 18 | 0 |
| H3 | 16 | 20 | 36 | 18 | 0 |

H3 removes one coarse-boundary `LDL` and adds no location because all other
semantic sites remain count-equivalent after downstream line shifts. Next:
retain H3 only if its direct H1 comparison is non-regressing.

The main kernel remains `REG:96 STACK:8 SHARED:1024 LOCAL:0`; `LOCAL:0` does
not override the instruction-level parser evidence. Next: continue reporting
both resource usage and exact local instructions.

## Every H3 source location

All 36 instructions are attributed; no parser location is omitted.

| Source line | Count | Mnemonics | Representative SASS |
|---:|---:|---|---|
| 407 | 15 | LDL 7, LDL.LU 2, STL 6 | `LDL R5,[R1+0x4]` |
| 1191 | 3 | LDL 2, STL 1 | `LDL R5,[R1]` |
| 6949 | 3 | LDL 1, LDL.LU 1, STL 1 | `LDL.LU R4,[R1+0x4]` |
| 5855 | 1 | STL 1 | `STL [R1],R2` |
| 5958 | 1 | STL 1 | `STL [R1+0x4],R8` |
| 4881 | 1 | LDL.LU 1 | `LDL.LU R43,[R1+0x4]` |
| 1142 | 1 | STL 1 | `STL [R1+0x4],R8` |
| 6232 | 1 | STL 1 | `STL [R1+0x4],R13` |
| 6402 | 1 | LDL.LU 1 | `LDL.LU R37,[R1+0x4]` |
| 6398 | 1 | STL 1 | `STL [R1+0x4],R37` |
| 1719 | 1 | LDL 1 | `LDL R131,[R1+0x4]` |
| 1752 | 1 | LDL 1 | `LDL R0,[R1]` |
| 6952 | 1 | STL 1 | `@P0 STL [R1+0x4],R4` |
| 7180 | 1 | LDL 1 | `LDL R0,[R1+0x4]` |
| 7193 | 1 | STL 1 | `STL [R1+0x4],R0` |
| 7138 | 1 | LDL 1 | `LDL R0,[R1+0x4]` |
| 7102 | 1 | LDL 1 | `LDL R0,[R1+0x4]` |
| 7207 | 1 | STL 1 | `STL [R1+0x4],R3` |

The remaining actionable clusters are the runtime chunk index at line 1191
and the four-stage round producer/committer state at lines 6949/6952/7193
because they retain paired or loop-carried local accesses. Next: change only
one cluster after the H3 wall-time decision and re-run the same lineinfo gate.
