#!/usr/bin/env bash
# E3-NCU paired attribution (one-click, GR100).
#
# Follow-up to run_e3pair_gr100.sh, which measured the oversized-mode
# cliff tax at +1.28 us/tile (cliff_tax_partial).  This script profiles
# the SAME two arms (compat = normal launch vs e3pad = oversized launch,
# byte-identical live layout) under Nsight Compute at one topk point and
# classifies the tax:
#   l1_class            local/stack traffic misses the 8 KB L1 -> despill
#                       surgery can redeem the tax
#   banking_class       shared-memory bank conflicts inflate under the
#                       oversized array configuration -> infrastructure
#   mixed               both signatures fire
#   unexplained_infra   neither fires -> tax lives outside these
#                       counters; escalate to trace
#
# Runner contract (the ONLY thing the runner does):
#   cd <repo> && bash benchmark/dsa/run_e3ncu_gr100.sh
#   -> success prints  E3NCU_OK / E3NCU_ARTIFACTS <dir>
#   -> failure prints  E3NCU_FAILED stage=<name> exit=<rc>
#   Either way, dump the artifacts dir (default benchmark/dsa/e3ncu_out,
#   override with E3NCU_OUT=...).
#
# Prerequisites: same stack as run_e3pair_gr100.sh (GR100 CC 10.7,
# CUDA 13.4, torch + cutlass DSL importable, sm_107a arch map) plus an
# ncu binary that supports CC 10.7 and profiling permission
# (ERR_NVGPUCTRPERM is detected and reported as a named failure).
#
# Knobs: E3NCU_TOPK (512), E3NCU_SKIP (35), E3NCU_COUNT (28),
# E3NCU_NCU (ncu binary path), E3NCU_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${E3NCU_OUT:-${REPO}/benchmark/dsa/e3ncu_out}"
TOPK="${E3NCU_TOPK:-512}"
SKIP="${E3NCU_SKIP:-35}"
COUNT="${E3NCU_COUNT:-28}"
PROBE="${REPO}/benchmark/dsa/ncu_e3_probe.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "E3NCU_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "E3NCU_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "E3NCU_CONFIG topk=${TOPK} skip=${SKIP} count=${COUNT}"

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
E3NCU_ALLOW_ANY_CC="${E3NCU_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("E3NCU_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}, sort_keys=True))
if os.environ.get("E3NCU_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set E3NCU_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"E3NCU_GPU_ARCH_FLAG {flag}")
if os.environ.get("E3NCU_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# ncu discovery: env override, PATH, CUDA_HOME, common install globs.
STAGE=ncu_discovery
NCU="${E3NCU_NCU:-}"
if [[ -z "${NCU}" ]]; then
    if command -v ncu >/dev/null 2>&1; then
        NCU="$(command -v ncu)"
    elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/ncu" ]]; then
        NCU="${CUDA_HOME}/bin/ncu"
    else
        NCU="$(ls -1 /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1 || true)"
    fi
fi
if [[ -z "${NCU}" || ! -x "${NCU}" ]]; then
    echo "ERROR ncu binary not found (set E3NCU_NCU=/path/to/ncu)" >&2
    false
fi
echo "E3NCU_NCU_BIN ${NCU}"
"${NCU}" --version | tee "${OUT}/ncu_version.log"

# Metric availability filter: keep only requested metrics whose base name
# the installed ncu knows on this chip; hard-require the discriminant set.
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
echo "E3NCU_METRICS ${METRICS}" | tee "${OUT}/ncu_metrics_used.log"

# --------------------------------------------------------------- profiling
profile_arm() {
    local arm="$1" var="$2"
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "${var}=1" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        "${NCU}" --target-processes all -f \
        --export "${OUT}/ncu_${arm}" \
        --metrics "${METRICS}" \
        --launch-skip "${SKIP}" \
        --launch-count "${COUNT}" \
        --csv \
        python3 "${PROBE}" --topk "${TOPK}" --warmup 2 --repeat 8 \
        2>&1 | tee "${OUT}/ncu_${arm}.log"
    if grep -q "ERR_NVGPUCTRPERM" "${OUT}/ncu_${arm}.log"; then
        echo "ERROR profiling permission denied (ERR_NVGPUCTRPERM); run in a container/driver config with counters enabled" >&2
        false
    fi
    grep -q "E3NCU_PROBE_OK" "${OUT}/ncu_${arm}.log" || {
        echo "ERROR probe did not complete for arm=${arm}" >&2
        false
    }
}
STAGE=ncu_compat
profile_arm compat DSA_RUBIN1_B200_COMPAT
STAGE=ncu_e3pad
profile_arm e3pad DSA_RUBIN1_E3PAD

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import csv
import json
import os
import statistics
from pathlib import Path

out = Path(os.environ["OUT_DIR"])

def parse(log_path):
    """Parse ncu --csv long-format rows embedded in the run log."""
    launches = {}
    header = None
    idx = {}
    with open(log_path, newline="") as f:
        for row in csv.reader(f):
            if header is None:
                if "Kernel Name" in row and "Metric Name" in row:
                    header = row
                    idx = {name: header.index(name) for name in
                           ("ID", "Kernel Name", "Metric Name",
                            "Metric Value")}
                continue
            if len(row) != len(header):
                continue
            lid = row[idx["ID"]]
            kname = row[idx["Kernel Name"]]
            metric = row[idx["Metric Name"]]
            raw = row[idx["Metric Value"]].replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            launches.setdefault(lid, {"kernel": kname, "m": {}})
            launches[lid]["m"][metric] = value
    assert launches, f"no CSV metric rows parsed from {log_path.name}"
    return launches

def main_kernel_medians(launches):
    totals = {}
    for entry in launches.values():
        d = entry["m"].get("gpu__time_duration.sum", 0.0)
        totals[entry["kernel"]] = totals.get(entry["kernel"], 0.0) + d
    main = max(totals, key=totals.get)
    rows = [e["m"] for e in launches.values() if e["kernel"] == main]
    assert len(rows) >= 3, f"only {len(rows)} instances of main kernel"
    metrics = set().union(*rows)
    med = {m: statistics.median([r[m] for r in rows if m in r])
           for m in metrics}
    return main, len(rows), med

arms = {}
for arm in ("compat", "e3pad"):
    name, n, med = main_kernel_medians(parse(out / f"ncu_{arm}.log"))
    arms[arm] = {"kernel": name, "instances": n, "metrics": med}

assert arms["compat"]["kernel"] == arms["e3pad"]["kernel"], (
    "main kernels differ between arms: "
    f"{arms['compat']['kernel']} vs {arms['e3pad']['kernel']}"
)

def g(arm, metric, default=None):
    return arms[arm]["metrics"].get(metric, default)

def ratio(metric):
    a = g("e3pad", metric)
    b = g("compat", metric)
    if a is None or b is None:
        return None
    return a / max(b, 1e-9)

def miss_pct(arm):
    hit = g(arm, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum")
    miss = g(arm, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum")
    if hit is None or miss is None or (hit + miss) == 0:
        return None
    return 100.0 * miss / (hit + miss)

bank = None
r_ld = ratio("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
r_st = ratio("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
if r_ld is not None and r_st is not None:
    bank = max(r_ld, r_st)
elif r_ld is not None or r_st is not None:
    bank = r_ld if r_ld is not None else r_st

disc = {
    "carveout_bytes": {
        "compat": g("compat", "launch__shared_mem_config_size"),
        "e3pad": g("e3pad", "launch__shared_mem_config_size"),
    },
    "local_ld_miss_pct": {
        "compat": miss_pct("compat"),
        "e3pad": miss_pct("e3pad"),
    },
    "R_local_l2": ratio(
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum"
    ),
    "R_bank_conflicts": bank,
    "R_duration": ratio("gpu__time_duration.sum"),
    "R_long_scoreboard": ratio(
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
}

# Pre-registered classification (bands fixed before the run):
mp_c = disc["local_ld_miss_pct"]["compat"]
mp_e = disc["local_ld_miss_pct"]["e3pad"]
l1_fires = (
    mp_c is not None and mp_e is not None
    and (mp_e - mp_c) >= 20.0
    and disc["R_local_l2"] is not None
    and disc["R_local_l2"] >= 2.0
)
bank_fires = (
    disc["R_bank_conflicts"] is not None
    and disc["R_bank_conflicts"] >= 1.5
)
if l1_fires and bank_fires:
    verdict = "mixed"
elif l1_fires:
    verdict = "l1_class"
elif bank_fires:
    verdict = "banking_class"
else:
    verdict = "unexplained_infra"

summary = {"arms": arms, "discriminants": disc, "verdict": verdict}
(out / "summary_e3ncu.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("E3NCU_SUMMARY " + json.dumps(disc, sort_keys=True))
print("E3NCU_VERDICT " + verdict)
PY

echo "E3NCU_OK"
echo "E3NCU_ARTIFACTS ${OUT}"
