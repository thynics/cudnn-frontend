#!/usr/bin/env bash
# K1-NCU: post-despill repacing measurement (one-click, GR100).
#
# K1a (producer warpgroup 48->64 regs, reduce 128->120) reclaimed
# compat 4.598->4.145 and native 9.778->4.470 us/tile (k1 r5 verdict:
# k1a_effective on both arms; structure residual collapsed +3.91->+0.33).
# This leg re-profiles all four arms with the proven E3STR metric set to
# answer, pre-registered:
#   Q-A  did K1a kill the producer spin-reload byte bucket?
#        (ck local sectors / cb: <=0.60 confirmed, <=0.85 partial,
#         else not_confirmed)
#   Q-B  is the remaining nk-ck = +0.33 us/tile the residual cliff tax
#        on the shrunken local traffic?  (nk vs ck: L2-local delta and
#        miss% under 8KB vs 108KB L1, alongside main-kernel duration
#        delta)
#   Q-C  what paces ck now?  (top stall classes per arm -- descriptive,
#        feeds the K2-target decision: math publish path vs reduce REDG
#        neighbourhood vs protocol ring)
#
# Arms (same candidate, one job, same node):
#   cb  DSA_RUBIN1_B200_COMPAT=1                          ring2/1face 48/128
#   ck  DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1     ring2/1face 64/120
#   nb  (none)                                            ring5/2face 48/128
#   nk  DSA_RUBIN1_REG_K1A=1                              ring5/2face 64/120
#
# Runner contract:
#   cd <repo> && K1NCU_NCU=<chip-aware ncu> bash benchmark/dsa/run_k1ncu_gr100.sh
#   -> success prints  K1NCU_OK / K1NCU_ARTIFACTS <dir>
#   -> failure prints  K1NCU_FAILED stage=<name> exit=<rc>
#   Artifacts dir: benchmark/dsa/k1ncu_out (override K1NCU_OUT).
#
# Knobs: K1NCU_TOPK (512), K1NCU_SKIP (8), K1NCU_COUNT (24), K1NCU_NCU,
# K1NCU_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K1NCU_OUT:-${REPO}/benchmark/dsa/k1ncu_out}"
TOPK="${K1NCU_TOPK:-512}"
SKIP="${K1NCU_SKIP:-8}"
COUNT="${K1NCU_COUNT:-24}"
PROBE="${REPO}/benchmark/dsa/ncu_e3_probe.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K1NCU_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K1NCU_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K1NCU_CONFIG topk=${TOPK} skip=${SKIP} count=${COUNT}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
K1NCU_ALLOW_ANY_CC="${K1NCU_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K1NCU_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
}, sort_keys=True))
if os.environ.get("K1NCU_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K1NCU_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K1NCU_GPU_ARCH_FLAG {flag}")
if os.environ.get("K1NCU_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# Pinned-only ncu contract (e3str campaign law).
STAGE=ncu_discovery
ncu_knows_chip() {
    local probe_log="${OUT}/ncu_query_probe_$(basename "$1").log"
    local attempt
    for attempt in 1 2; do
        "$1" --query-metrics > "${probe_log}" 2>&1 || true
        # Pipe-free on purpose (pipefail + awk|grep -q = SIGPIPE 141).
        if awk '$1=="gpu__time_duration"{found=1} END{exit !found}' "${probe_log}"; then
            return 0
        fi
        echo "K1NCU_NCU_QUERY_ATTEMPT ${attempt} failed for $1; head:"
        head -5 "${probe_log}" | sed 's/^/    | /'
        [[ "${attempt}" == "1" ]] && sleep 10
    done
    return 1
}
if [[ -z "${K1NCU_NCU:-}" ]]; then
    echo "ERROR this leg requires a pinned chip-aware binary: set K1NCU_NCU=/path/to/ncu" >&2
    false
fi
if [[ ! -e "${K1NCU_NCU}" ]]; then
    echo "ERROR pinned K1NCU_NCU=${K1NCU_NCU} does not exist inside this container (host mount missing?)" >&2
    false
elif [[ ! -x "${K1NCU_NCU}" ]]; then
    echo "ERROR pinned K1NCU_NCU=${K1NCU_NCU} exists but is not executable" >&2
    false
elif ! ncu_knows_chip "${K1NCU_NCU}"; then
    echo "ERROR pinned K1NCU_NCU=${K1NCU_NCU} runs but its metric query lacks gpu__time_duration after 2 attempts" >&2
    false
fi
NCU="${K1NCU_NCU}"
echo "K1NCU_NCU_BIN ${NCU}"
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
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss",
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
echo "K1NCU_METRICS ${METRICS}" | tee "${OUT}/ncu_metrics_used.log"

# --------------------------------------------------------------- profiling
profile_check() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        K1NCU_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)

arm = os.environ["K1NCU_ARM"]
print("K1NCU_PROFILE " + json.dumps({
    "arm": arm,
    "round_stages": K.ROUND_STAGES,
    "pds_faces": K.PDS_FACES,
    "max_smem_bytes": K.MAX_SMEM_BYTES,
    "reg_k1a": K.REG_K1A,
    "producer_regs": K.PRODUCER_REGS,
    "reduce_regs": K.REDUCE_REGS,
}, sort_keys=True))
compat = arm.startswith("c")
k1a = arm.endswith("k")
assert (K.ROUND_STAGES, K.PDS_FACES) == ((2, 1) if compat else (5, 2))
assert K.MAX_SMEM_BYTES == (232_448 if compat else 334_848)
assert K.REG_K1A is k1a
assert (K.PRODUCER_REGS, K.REDUCE_REGS) == ((64, 120) if k1a else (48, 128))
assert not K.E3PAD
PY
}
STAGE=profile_cb
profile_check cb DSA_RUBIN1_B200_COMPAT=1 | tee "${OUT}/profiles.log"
STAGE=profile_ck
profile_check ck DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1 | tee -a "${OUT}/profiles.log"
STAGE=profile_nb
profile_check nb | tee -a "${OUT}/profiles.log"
STAGE=profile_nk
profile_check nk DSA_RUBIN1_REG_K1A=1 | tee -a "${OUT}/profiles.log"

profile_arm() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A "$@" \
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
STAGE=ncu_cb
profile_arm cb DSA_RUBIN1_B200_COMPAT=1
STAGE=ncu_ck
profile_arm ck DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1
STAGE=ncu_nb
profile_arm nb
STAGE=ncu_nk
profile_arm nk DSA_RUBIN1_REG_K1A=1

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import csv
import json
import os
import statistics
from pathlib import Path

out = Path(os.environ["OUT_DIR"])

def load(arm):
    rows = list(csv.reader(open(out / f"ncu_{arm}_import.csv",
                                newline="", errors="replace")))
    hdr = rows[0]
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
        launches.append((r[ik], m))
    totals = {}
    for k, m in launches:
        totals[k] = totals.get(k, 0) + m.get("gpu__time_duration.sum", 0)
    main = max(totals, key=totals.get)
    assert not (main.startswith("void at::") or "nvjet" in main), (
        f"arm={arm}: dominant kernel is reference machinery ({main[:60]})"
    )
    inst = [m for k, m in launches if k == main]
    assert len(inst) >= 2, f"arm={arm}: only {len(inst)} main instances"
    med = {mm: statistics.median([r[mm] for r in inst if mm in r])
           for mm in set().union(*inst)}
    return main, len(inst), med

M = {}
for arm in ("cb", "ck", "nb", "nk"):
    name, n, med = load(arm)
    M[arm] = med
    M[arm]["_kernel"] = name
    M[arm]["_instances"] = n

def g(arm, metric):
    return M[arm].get(metric)

LD = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"
ST = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
HIT = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum"
MISS = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum"
DUR = "gpu__time_duration.sum"
STALLS = {
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "mio": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "lg": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "short_sb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
}

def local_total(arm):
    return (g(arm, LD) or 0.0) + (g(arm, ST) or 0.0)

def miss_pct(arm):
    h, m = g(arm, HIT), g(arm, MISS)
    if h is None or m is None or (h + m) == 0:
        return None
    return 100.0 * m / (h + m)

per_arm = {
    arm: {
        "kernel": M[arm]["_kernel"][:80],
        "instances": M[arm]["_instances"],
        "duration_ms": g(arm, DUR),
        "local_ld_sectors": g(arm, LD),
        "local_st_sectors": g(arm, ST),
        "local_ld_miss_pct": miss_pct(arm),
        "l2_local_ld_MB": (g(arm, MISS) or 0.0) * 32 / 1e6,
        "stalls": {k: g(arm, v) for k, v in STALLS.items()},
        "ldl_inst": g(arm, "smsp__sass_inst_executed_op_local_ld.sum"),
        "stl_inst": g(arm, "smsp__sass_inst_executed_op_local_st.sum"),
    }
    for arm in ("cb", "ck", "nb", "nk")
}

# Q-A: did K1a kill the producer spin-reload bucket?
r_kill = local_total("ck") / max(local_total("cb"), 1e-9)
if r_kill <= 0.60:
    qa = "k1a_spill_kill_confirmed"
elif r_kill <= 0.85:
    qa = "k1a_spill_kill_partial"
else:
    qa = "k1a_spill_kill_not_confirmed"

# Q-B: residual cliff account (nk vs ck).
qb = {
    "duration_delta_ms": (g("nk", DUR) or 0) - (g("ck", DUR) or 0),
    "l2_local_ld_delta_MB": per_arm["nk"]["l2_local_ld_MB"]
    - per_arm["ck"]["l2_local_ld_MB"],
    "nk_miss_pct": per_arm["nk"]["local_ld_miss_pct"],
    "ck_miss_pct": per_arm["ck"]["local_ld_miss_pct"],
    "r_local_nk_over_ck": local_total("nk") / max(local_total("ck"), 1e-9),
}

# Q-C: new pacer (descriptive).
def top_stalls(arm, n=2):
    s = per_arm[arm]["stalls"]
    return sorted(
        ((k, v) for k, v in s.items() if v is not None),
        key=lambda kv: -kv[1],
    )[:n]

summary = {
    "per_arm": per_arm,
    "qa_spill_kill": {"ratio_ck_over_cb": r_kill, "verdict": qa},
    "qb_residual_cliff": qb,
    "qc_top_stalls": {arm: top_stalls(arm) for arm in ("cb", "ck", "nb", "nk")},
}
(out / "summary_k1ncu.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K1NCU_SUMMARY " + json.dumps({
    "qa": qa,
    "ratio_ck_over_cb_local": round(r_kill, 4),
    "nk_minus_ck_duration_ms": round(qb["duration_delta_ms"], 4),
    "nk_miss_pct": qb["nk_miss_pct"],
    "ck_top_stall": top_stalls("ck", 1),
}, sort_keys=True, default=str))
print("K1NCU_VERDICT " + qa)
PY

echo "K1NCU_OK"
echo "K1NCU_ARTIFACTS ${OUT}"
