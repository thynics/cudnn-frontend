#!/usr/bin/env bash
# K9: self-selecting stall-map probe (one-click, GR100).
#
# Runs LAST in the five-job window (k4..k8).  Reads the summary JSONs the
# earlier jobs left in benchmark/dsa/{k5,k6,k7,k8}_out/, picks the candidate
# arm with the LOWEST steady period among {r2, r2m2, r3, r3m2}, and profiles
# that arm under Nsight Compute with a broad stall-reason metric set.  The
# stall table of the window winner (or of the least-bad regressor -- a
# regression's stall map is exactly the diagnosis input) is the targeting
# data for the next optimization knife.
#
# Runner contract (the ONLY thing the runner does):
#   cd <repo> && bash benchmark/dsa/run_k9_gr100.sh
#   -> success prints  K9_OK / K9_ARTIFACTS <dir>
#   -> failure prints  K9_FAILED stage=<name> exit=<rc>
#
# Prerequisites: same stack as run_e3ncu_gr100.sh (GR100 CC 10.7, chip-aware
# ncu, profiling permission) plus at least one of k5/k6/k7/k8 summaries on
# disk (hard-gated at arm_selection).
#
# Knobs: K9_ARM (override auto-selection: r2|r2m2|r3|r3m2), K9_TOPK (512),
# K9_SKIP (8), K9_COUNT (24), K9_NCU, K9_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K9_OUT:-${REPO}/benchmark/dsa/k9_out}"
TOPK="${K9_TOPK:-512}"
SKIP="${K9_SKIP:-8}"
COUNT="${K9_COUNT:-24}"
PROBE="${REPO}/benchmark/dsa/ncu_e3_probe.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K9_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K9_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K9_CONFIG topk=${TOPK} skip=${SKIP} count=${COUNT} arm_override=${K9_ARM:-auto}"

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
K9_ALLOW_ANY_CC="${K9_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K9_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}, sort_keys=True))
if os.environ.get("K9_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K9_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K9_GPU_ARCH_FLAG {flag}")
if os.environ.get("K9_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# ------------------------------------------------------------ arm_selection
# Every k-kit writes its JSON as "summary_k4.json" inside its own out dir
# (shared lineage); per-kit verdict markers differ but the period map key is
# uniform.  ck (compat+K1a champion) is a first-class candidate; e3k1 is not.
STAGE=arm_selection
DSA_DIR="${REPO}/benchmark/dsa" OUT_DIR="${OUT}" K9_ARM="${K9_ARM:-}" \
python3 - <<'PY' | tee "${OUT}/arm_selection.log"
import json
import os
from pathlib import Path

dsa = Path(os.environ["DSA_DIR"])
out = Path(os.environ["OUT_DIR"])
ARM_TO_IMPL = {
    "ck":   ("rubin_1", "DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1"),
    "r2":   ("rubin_2", ""),
    "r2m2": ("rubin_2", "DSA_RUBIN2_M2=1"),
    "r3":   ("rubin_3", ""),
    "r3m2": ("rubin_3", "DSA_RUBIN2_M2=1"),
}
candidates = {}
for kdir in ("k5_out", "k6_out", "k7_out", "k8_out"):
    p = dsa / kdir / "summary_k4.json"
    if not p.exists():
        print(f"K9_ARM_SOURCE {kdir} absent")
        continue
    periods = json.loads(p.read_text()).get("steady_period_us_per_tile", {})
    print(f"K9_ARM_SOURCE {kdir} periods={json.dumps(periods, sort_keys=True)}")
    for arm, period in periods.items():
        if arm in ARM_TO_IMPL:
            # keep the best (lowest) reading per arm across windows
            if arm not in candidates or period < candidates[arm]:
                candidates[arm] = period

override = os.environ.get("K9_ARM", "")
if override:
    assert override in ARM_TO_IMPL, f"K9_ARM={override} not in {sorted(ARM_TO_IMPL)}"
    arm = override
    print(f"K9_ARM_OVERRIDE {arm}")
else:
    assert candidates, (
        "no candidate arm found in k5/k6/k7/k8 summaries; "
        "set K9_ARM to force one"
    )
    arm = min(candidates, key=candidates.get)

impl, extra = ARM_TO_IMPL[arm]
print("K9_ARM_SELECTED " + json.dumps({
    "arm": arm,
    "impl": impl,
    "extra_env": extra,
    "candidates": candidates,
}, sort_keys=True))
(out / "arm_selected.env").write_text(
    f"ARM={arm}\nIMPL={impl}\nEXTRA=\"{extra}\"\n"
)
PY
# shellcheck disable=SC1091
source "${OUT}/arm_selected.env"
echo "K9_ARM_ENV ARM=${ARM} IMPL=${IMPL} EXTRA=${EXTRA:-}"

# ncu discovery: env override, PATH, CUDA_HOME, common install globs.
STAGE=ncu_discovery
NCU="${K9_NCU:-}"
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
    echo "ERROR ncu binary not found (set K9_NCU=/path/to/ncu)" >&2
    false
fi
echo "K9_NCU_BIN ${NCU}"
"${NCU}" --version | tee "${OUT}/ncu_version.log"

# Metric availability filter: broad stall-reason set; keep whatever this
# chip/ncu knows, hard-require the core discriminants.
STAGE=metrics
"${NCU}" --query-metrics > "${OUT}/ncu_available_metrics.txt" 2>&1 || true
METRICS="$(AVAIL="${OUT}/ncu_available_metrics.txt" python3 - <<'PY'
import os

STALL_REASONS = [
    "long_scoreboard", "short_scoreboard", "barrier", "membar", "wait",
    "drain", "dispatch_stall", "imc_miss", "lg_throttle",
    "math_pipe_throttle", "mio_throttle", "misc", "no_instruction",
    "not_selected", "selected", "sleeping", "tex_throttle",
    "branch_resolving",
]
REQUESTED = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
] + [
    f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio"
    for r in STALL_REASONS
]
REQUIRED_BASES = {
    "gpu__time_duration",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active",
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
echo "K9_METRICS ${METRICS}" | tee "${OUT}/ncu_metrics_used.log"

# --------------------------------------------------------------- profiling
STAGE=ncu_profile
# shellcheck disable=SC2086  # EXTRA holds space-separated KEY=VAL pairs
env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD \
    -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN2_M2 \
    ${EXTRA:-} \
    PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    "${NCU}" --target-processes all -f \
    --export "${OUT}/ncu_${ARM}" \
    --metrics "${METRICS}" \
    --nvtx --nvtx-include "dsa_bwd/" \
    --launch-skip "${SKIP}" \
    --launch-count "${COUNT}" \
    python3 "${PROBE}" --impl "${IMPL}" --topk "${TOPK}" --warmup 2 --repeat 8 \
    2>&1 | tee "${OUT}/ncu_${ARM}.log"
if grep -q "ERR_NVGPUCTRPERM" "${OUT}/ncu_${ARM}.log"; then
    echo "ERROR profiling permission denied (ERR_NVGPUCTRPERM)" >&2
    false
fi
grep -q "E3NCU_PROBE_OK" "${OUT}/ncu_${ARM}.log" || {
    echo "ERROR probe did not complete for arm=${ARM}" >&2
    false
}
"${NCU}" --import "${OUT}/ncu_${ARM}.ncu-rep" --csv --page raw \
    > "${OUT}/ncu_${ARM}_import.csv"
[[ -s "${OUT}/ncu_${ARM}_import.csv" ]] || {
    echo "ERROR empty import CSV for arm=${ARM}" >&2
    false
}

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" K9_ARM_NAME="${ARM}" python3 - <<'PY'
import csv
import json
import os
import statistics
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
arm = os.environ["K9_ARM_NAME"]

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
    return launches

def is_reference_kernel(name):
    return name.startswith("void at::") or "nvjet" in name

launches = parse_wide(out / f"ncu_{arm}_import.csv")
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
med = {m: statistics.median([r[m] for r in rows if m in r]) for m in metrics}

PREFIX = "smsp__average_warps_issue_stalled_"
stalls = {
    k[len(PREFIX):].replace("_per_issue_active.ratio", ""): v
    for k, v in med.items() if k.startswith(PREFIX)
}
ranked = sorted(stalls.items(), key=lambda kv: kv[1], reverse=True)
top = ranked[0][0] if ranked else "none"

summary = {
    "arm": arm,
    "kernel": main,
    "instances": len(rows),
    "duration_us_median": med.get("gpu__time_duration.sum"),
    "issue_active_pct": med.get(
        "smsp__issue_active.avg.pct_of_peak_sustained_active"),
    "stall_ranking": ranked,
    "other_metrics": {k: v for k, v in med.items()
                      if not k.startswith(PREFIX)},
}
(out / "summary_k9.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K9_SUMMARY " + json.dumps({
    "arm": arm,
    "top5_stalls": ranked[:5],
    "issue_active_pct": summary["issue_active_pct"],
}, sort_keys=True))
print(f"K9_VERDICT top_stall={top}")
PY

echo "K9_OK"
echo "K9_ARTIFACTS ${OUT}"
