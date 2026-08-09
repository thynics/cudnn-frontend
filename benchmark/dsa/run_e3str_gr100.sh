#!/usr/bin/env bash
# E3-STR: three-arm NCU attribution of the native structure residual
# (one-click, GR100).
#
# Ledger context (E3_GR100悬崖税与native异常台账_20260809.md): the native
# anomaly decomposes as 9.78 = compat 4.60 + cliff tax 1.28 (l1_class,
# closed by E3-NCU r3) + structure residual 3.91 us/tile (open).  This
# leg profiles THREE arms of the same candidate in one job:
#   compat  DSA_RUBIN1_B200_COMPAT=1  ring2/1face, normal launch
#   e3pad   DSA_RUBIN1_E3PAD=1       ring2/1face + dead pad, oversized
#   native  (both unset)              ring5/2face, oversized
# native minus e3pad isolates the pure structure effect under identical
# launch mode / L1, and classifies it:
#   spill_amplification  S1: ring5/2face inflates local-memory traffic
#                        (register pressure), amplified by the 8 KB L1
#   serialization        S2: the never-perf-validated mod-5/dual-face
#                        arms over-serialize (stall-side signature)
#   smem_path            S3: shared-memory path degrades (>256 KiB
#                        addressing / banking)
#   mixed_structure / unexplained_structure otherwise (null -> E7/E5)
#
# Runner contract (the ONLY thing the runner does):
#   cd <repo> && bash benchmark/dsa/run_e3str_gr100.sh
#   -> success prints  E3STR_OK / E3STR_ARTIFACTS <dir>
#   -> failure prints  E3STR_FAILED stage=<name> exit=<rc>
#   Either way, dump the artifacts dir (default benchmark/dsa/e3str_out,
#   override with E3STR_OUT=...).
#
# Prerequisites and NCU handling identical to run_e3ncu_gr100.sh
# (NVTX-gated probe, import-based readback, reference-kernel validity
# gate, decimal-Kbyte units).
#
# Knobs: E3STR_TOPK (512), E3STR_SKIP (8), E3STR_COUNT (24),
# E3STR_NCU (ncu binary path), E3STR_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${E3STR_OUT:-${REPO}/benchmark/dsa/e3str_out}"
TOPK="${E3STR_TOPK:-512}"
SKIP="${E3STR_SKIP:-8}"
COUNT="${E3STR_COUNT:-24}"
PROBE="${REPO}/benchmark/dsa/ncu_e3_probe.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "E3STR_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "E3STR_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "E3STR_CONFIG topk=${TOPK} skip=${SKIP} count=${COUNT}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    uname -a
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
E3STR_ALLOW_ANY_CC="${E3STR_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("E3STR_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}, sort_keys=True))
if os.environ.get("E3STR_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set E3STR_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"E3STR_GPU_ARCH_FLAG {flag}")
if os.environ.get("E3STR_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# ncu discovery with per-candidate chip validation (e3str r1 lesson:
# the NGC image's PATH ncu 2026.2.1 does not know GR100 -- query-metrics
# prints "Skipping unsupported chip Unknown" and an empty list).  A
# candidate is accepted only if its metric list contains
# gpu__time_duration for the present chip.
STAGE=ncu_discovery
ncu_knows_chip() {
    "$1" --query-metrics 2>/dev/null | awk '{print $1}' \
        | grep -qx "gpu__time_duration"
}
if [[ -n "${E3STR_NCU:-}" ]]; then
    # Explicit contract: if the caller pins a binary, it must work.
    if [[ ! -x "${E3STR_NCU}" ]] || ! ncu_knows_chip "${E3STR_NCU}"; then
        echo "ERROR pinned E3STR_NCU=${E3STR_NCU} is not executable or does not support this chip" >&2
        false
    fi
    NCU="${E3STR_NCU}"
else
    CANDIDATES=()
    if [[ -n "${CUDA_HOME:-}" ]]; then
        CANDIDATES+=("${CUDA_HOME}/ncu" "${CUDA_HOME}/bin/ncu")
    fi
    if command -v ncu >/dev/null 2>&1; then
        CANDIDATES+=("$(command -v ncu)")
    fi
    while IFS= read -r g; do
        CANDIDATES+=("${g}")
    done < <(ls -1 /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V)
    NCU=""
    for cand in ${CANDIDATES[@]+"${CANDIDATES[@]}"}; do
        if [[ ! -x "${cand}" ]]; then
            echo "E3STR_NCU_REJECT ${cand} (not executable)"
            continue
        fi
        if ncu_knows_chip "${cand}"; then
            NCU="${cand}"
            break
        fi
        echo "E3STR_NCU_REJECT ${cand} ($({ "${cand}" --query-metrics 2>&1 || true; } | head -1))"
    done
    if [[ -z "${NCU}" ]]; then
        echo "ERROR no ncu candidate supports this chip (set E3STR_NCU to a chip-aware binary, e.g. the internal CUDA 13.4 toolkit's ncu)" >&2
        false
    fi
fi
echo "E3STR_NCU_BIN ${NCU}"
"${NCU}" --version | tee "${OUT}/ncu_version.log"

STAGE=metrics
"${NCU}" --query-metrics > "${OUT}/ncu_available_metrics.txt" 2>&1 || true
METRICS="$(AVAIL="${OUT}/ncu_available_metrics.txt" python3 - <<'PY'
import os

REQUESTED = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "launch__shared_mem_config_size",
    "launch__shared_mem_per_block_dynamic",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
]
REQUIRED_BASES = {
    "gpu__time_duration",
    "smsp__sass_inst_executed_op_local_ld",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active",
}

avail = set()
with open(os.environ["AVAIL"]) as f:
    for line in f:
        token = line.split()[0] if line.split() else ""
        if token:
            avail.add(token)

kept = [m for m in REQUESTED if m.split(".")[0] in avail]
kept_bases = {m.split(".")[0] for m in kept}
missing = REQUIRED_BASES - kept_bases
if missing:
    raise SystemExit(
        f"required NCU metrics unavailable on this chip/ncu: {sorted(missing)}"
    )
print(",".join(kept))
PY
)"
echo "E3STR_METRICS ${METRICS}" | tee "${OUT}/ncu_metrics_used.log"

# --------------------------------------------------------------- profiling
# Per-arm profile asserts, one python process per arm (env read at import).
profile_check() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        E3STR_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)

arm = os.environ["E3STR_ARM"]
print("E3STR_PROFILE " + json.dumps({
    "arm": arm,
    "round_stages": K.ROUND_STAGES,
    "pds_faces": K.PDS_FACES,
    "max_smem_bytes": K.MAX_SMEM_BYTES,
    "e3pad": K.E3PAD,
}, sort_keys=True))
if arm == "compat":
    assert (K.ROUND_STAGES, K.PDS_FACES) == (2, 1)
    assert not K.E3PAD and K.MAX_SMEM_BYTES == 232_448
elif arm == "e3pad":
    assert (K.ROUND_STAGES, K.PDS_FACES) == (2, 1)
    assert K.E3PAD and K.MAX_SMEM_BYTES == 334_848
else:
    assert arm == "native"
    assert (K.ROUND_STAGES, K.PDS_FACES) == (5, 2)
    assert not K.E3PAD and K.MAX_SMEM_BYTES == 334_848
PY
}
STAGE=profile_compat
profile_check compat DSA_RUBIN1_B200_COMPAT=1 | tee "${OUT}/profiles.log"
STAGE=profile_e3pad
profile_check e3pad DSA_RUBIN1_E3PAD=1 | tee -a "${OUT}/profiles.log"
STAGE=profile_native
profile_check native | tee -a "${OUT}/profiles.log"

profile_arm() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        "${NCU}" --target-processes all -f \
        --export "${OUT}/ncu_${arm}" \
        --metrics "${METRICS}" \
        --nvtx --nvtx-include "dsa_bwd/" \
        --launch-skip "${SKIP}" \
        --launch-count "${COUNT}" \
        python3 "${PROBE}" --topk "${TOPK}" --warmup 2 --repeat 8 \
        2>&1 | tee "${OUT}/ncu_${arm}.log"
    if grep -q "ERR_NVGPUCTRPERM" "${OUT}/ncu_${arm}.log"; then
        echo "ERROR profiling permission denied (ERR_NVGPUCTRPERM)" >&2
        false
    fi
    grep -q "E3NCU_PROBE_OK" "${OUT}/ncu_${arm}.log" || {
        echo "ERROR probe did not complete for arm=${arm}" >&2
        false
    }
    "${NCU}" --import "${OUT}/ncu_${arm}.ncu-rep" --csv --page raw \
        > "${OUT}/ncu_${arm}_import.csv"
    [[ -s "${OUT}/ncu_${arm}_import.csv" ]] || {
        echo "ERROR empty import CSV for arm=${arm}" >&2
        false
    }
}
STAGE=ncu_compat
profile_arm compat DSA_RUBIN1_B200_COMPAT=1
STAGE=ncu_e3pad
profile_arm e3pad DSA_RUBIN1_E3PAD=1
STAGE=ncu_native
profile_arm native

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import csv
import json
import os
import statistics
from pathlib import Path

out = Path(os.environ["OUT_DIR"])

def parse_wide(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 3, f"no launch rows in {csv_path.name}"
    hdr, units = rows[0], rows[1]
    ik = hdr.index("Kernel Name")
    launches = []
    for r in rows[2:]:
        if len(r) != len(hdr):
            continue
        m = {}
        for j, name in enumerate(hdr):
            try:
                m[name] = float(r[j].replace(",", ""))
            except ValueError:
                pass
        launches.append({"kernel": r[ik], "m": m})
    assert launches, f"no parsable launch rows in {csv_path.name}"
    return launches, dict(zip(hdr, units))

def is_reference_kernel(name):
    return name.startswith("void at::") or "nvjet" in name

def main_kernel_medians(launches):
    totals = {}
    for entry in launches:
        d = entry["m"].get("gpu__time_duration.sum", 0.0)
        totals[entry["kernel"]] = totals.get(entry["kernel"], 0.0) + d
    main = max(totals, key=totals.get)
    if is_reference_kernel(main):
        raise SystemExit(
            "HARD GATE: dominant profiled kernel is reference machinery "
            f"({main[:80]}); NVTX gating did not reach the DSA kernels"
        )
    rows = [e["m"] for e in launches if e["kernel"] == main]
    assert len(rows) >= 2, f"only {len(rows)} instances of main kernel"
    metrics = set().union(*rows)
    med = {m: statistics.median([r[m] for r in rows if m in r])
           for m in metrics}
    return main, len(rows), med

def kbyte_to_bytes(value, unit):
    if value is None:
        return None
    return value * 1000.0 if unit.startswith("Kbyte") else value

arms = {}
for arm in ("compat", "e3pad", "native"):
    launches, units = parse_wide(out / f"ncu_{arm}_import.csv")
    name, n, med = main_kernel_medians(launches)
    arms[arm] = {"kernel": name, "instances": n, "metrics": med,
                 "units": units}

# compat and e3pad share the exact struct signature; native's mangled
# name may differ (different staged layouts) -- require only that all
# three passed the reference-kernel gate.
assert arms["compat"]["kernel"] == arms["e3pad"]["kernel"], (
    "compat/e3pad main kernels differ: "
    f"{arms['compat']['kernel'][:60]} vs {arms['e3pad']['kernel'][:60]}"
)

def g(arm, metric):
    return arms[arm]["metrics"].get(metric)

def pair_ratio(num_arm, den_arm, metric):
    a, b = g(num_arm, metric), g(den_arm, metric)
    if a is None or b is None:
        return None
    return a / max(b, 1e-9)

def miss_pct(arm):
    hit = g(arm, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum")
    miss = g(arm, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum")
    if hit is None or miss is None or (hit + miss) == 0:
        return None
    return 100.0 * miss / (hit + miss)

def local_total(arm):
    ld = g(arm, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum") or 0.0
    st = g(arm, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum") or 0.0
    return ld + st

def bank_ratio(num_arm, den_arm):
    vals = []
    for op in ("ld", "st"):
        r = pair_ratio(
            num_arm, den_arm,
            f"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum",
        )
        if r is not None:
            vals.append(r)
    return max(vals) if vals else None

BAR = "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"

per_arm = {
    arm: {
        "kernel": arms[arm]["kernel"],
        "instances": arms[arm]["instances"],
        "duration": g(arm, "gpu__time_duration.sum"),
        "regs_per_thread": g(arm, "launch__registers_per_thread"),
        "dynamic_smem_bytes": kbyte_to_bytes(
            g(arm, "launch__shared_mem_per_block_dynamic"),
            arms[arm]["units"].get(
                "launch__shared_mem_per_block_dynamic", ""),
        ),
        "carveout_bytes": kbyte_to_bytes(
            g(arm, "launch__shared_mem_config_size"),
            arms[arm]["units"].get("launch__shared_mem_config_size", ""),
        ),
        "local_ld_miss_pct": miss_pct(arm),
        "local_sectors_total": local_total(arm),
        "barrier_stall": g(arm, BAR),
    }
    for arm in ("compat", "e3pad", "native")
}

# Cliff re-check (e3pad vs compat) -- expected to reproduce r3.
cliff = {
    "R_duration": pair_ratio("e3pad", "compat", "gpu__time_duration.sum"),
    "R_local_l2": pair_ratio(
        "e3pad", "compat",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum"),
    "miss_pct_delta": (
        None if per_arm["e3pad"]["local_ld_miss_pct"] is None
        or per_arm["compat"]["local_ld_miss_pct"] is None
        else per_arm["e3pad"]["local_ld_miss_pct"]
        - per_arm["compat"]["local_ld_miss_pct"]),
}

# Structure residual (native vs e3pad) -- the leg's actual question.
r_local = (
    per_arm["native"]["local_sectors_total"]
    / max(per_arm["e3pad"]["local_sectors_total"], 1e-9))
r_barrier = pair_ratio("native", "e3pad", BAR)
r_bank = bank_ratio("native", "e3pad")
r_wave = max(
    [r for r in (
        pair_ratio("native", "e3pad",
                   "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
        pair_ratio("native", "e3pad",
                   "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
    ) if r is not None] or [None]
) if True else None
structure = {
    "R_duration": pair_ratio("native", "e3pad", "gpu__time_duration.sum"),
    "R_local_sectors": r_local,
    "R_barrier_stall": r_barrier,
    "R_bank_conflicts": r_bank,
    "R_shared_wavefronts": r_wave,
    "R_local_l2": pair_ratio(
        "native", "e3pad",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum"),
    "regs_delta": (
        None if per_arm["native"]["regs_per_thread"] is None
        or per_arm["e3pad"]["regs_per_thread"] is None
        else per_arm["native"]["regs_per_thread"]
        - per_arm["e3pad"]["regs_per_thread"]),
}

# Pre-registered classification of the structure residual:
s1 = r_local is not None and r_local >= 1.30
s2 = (
    r_barrier is not None and r_barrier >= 1.30
    and (r_local is None or r_local < 1.15)
)
s3 = (
    (r_bank is not None and r_bank >= 1.30)
    or (r_wave is not None and r_wave >= 1.30)
)
fired = [name for name, flag in
         (("spill_amplification", s1),
          ("serialization", s2),
          ("smem_path", s3)) if flag]
if len(fired) == 1:
    verdict = fired[0]
elif len(fired) > 1:
    verdict = "mixed_structure"
else:
    verdict = "unexplained_structure"

summary = {
    "per_arm": per_arm,
    "cliff_recheck_e3pad_vs_compat": cliff,
    "structure_native_vs_e3pad": structure,
    "fired": fired,
    "verdict": verdict,
}
(out / "summary_e3str.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("E3STR_SUMMARY " + json.dumps(structure, sort_keys=True))
print("E3STR_VERDICT " + verdict)
PY

echo "E3STR_OK"
echo "E3STR_ARTIFACTS ${OUT}"
