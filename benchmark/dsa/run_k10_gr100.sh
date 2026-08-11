#!/usr/bin/env bash
# K10: SASS-level stall attribution probe via PC sampling (one-click, GR100).
#
# Follow-up to k9 (ck stall map: long_scoreboard 55%/barrier 27%).  Reads the
# earlier jobs left in benchmark/dsa/{k5,k6,k7,k8}_out/, picks the candidate
# arm with the LOWEST steady period among {r2, r2m2, r3, r3m2}, and profiles
# that arm under Nsight Compute with a broad stall-reason metric set.  The
# stall table of the window winner (or of the least-bad regressor -- a
# regression's stall map is exactly the diagnosis input) is the targeting
# data for the next optimization knife.
#
# Runner contract (the ONLY thing the runner does):
#   cd <repo> && bash benchmark/dsa/run_k9_gr100.sh
#   -> success prints  K10_OK / K10_ARTIFACTS <dir>
#   -> failure prints  K10_FAILED stage=<name> exit=<rc>
#
# Prerequisites: same stack as run_e3ncu_gr100.sh (GR100 CC 10.7, chip-aware
# ncu, profiling permission) plus at least one of k5/k6/k7/k8 summaries on
# disk (hard-gated at arm_selection).
#
# Knobs: K10_ARM (override auto-selection: r2|r2m2|r3|r3m2), K10_TOPK (512),
# K10_SKIP (8), K10_COUNT (24), K10_NCU, K10_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K10_OUT:-${REPO}/benchmark/dsa/k10_out}"
TOPK="${K10_TOPK:-512}"
SKIP="${K10_SKIP:-8}"
COUNT="${K10_COUNT:-24}"
PROBE="${REPO}/benchmark/dsa/ncu_e3_probe.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K10_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K10_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K10_CONFIG topk=${TOPK} skip=${SKIP} count=${COUNT} arm_override=${K10_ARM:-auto}"

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
K10_ALLOW_ANY_CC="${K10_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K10_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}, sort_keys=True))
if os.environ.get("K10_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K10_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K10_GPU_ARCH_FLAG {flag}")
if os.environ.get("K10_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# ------------------------------------------------------------ arm_selection
# Every k-kit writes its JSON as "summary_k4.json" inside its own out dir
# (shared lineage); per-kit verdict markers differ but the period map key is
# uniform.  ck (compat+K1a champion) is a first-class candidate; e3k1 is not.
STAGE=arm_selection
DSA_DIR="${REPO}/benchmark/dsa" OUT_DIR="${OUT}" K10_ARM="${K10_ARM:-}" \
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
        print(f"K10_ARM_SOURCE {kdir} absent")
        continue
    periods = json.loads(p.read_text()).get("steady_period_us_per_tile", {})
    print(f"K10_ARM_SOURCE {kdir} periods={json.dumps(periods, sort_keys=True)}")
    for arm, period in periods.items():
        if arm in ARM_TO_IMPL:
            # keep the best (lowest) reading per arm across windows
            if arm not in candidates or period < candidates[arm]:
                candidates[arm] = period

override = os.environ.get("K10_ARM", "")
if override:
    assert override in ARM_TO_IMPL, f"K10_ARM={override} not in {sorted(ARM_TO_IMPL)}"
    arm = override
    print(f"K10_ARM_OVERRIDE {arm}")
else:
    assert candidates, (
        "no candidate arm found in k5/k6/k7/k8 summaries; "
        "set K10_ARM to force one"
    )
    arm = min(candidates, key=candidates.get)

impl, extra = ARM_TO_IMPL[arm]
print("K10_ARM_SELECTED " + json.dumps({
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
echo "K10_ARM_ENV ARM=${ARM} IMPL=${IMPL} EXTRA=${EXTRA:-}"

# ncu discovery: env override, PATH, CUDA_HOME, common install globs.
STAGE=ncu_discovery
NCU="${K10_NCU:-}"
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
    echo "ERROR ncu binary not found (set K10_NCU=/path/to/ncu)" >&2
    false
fi
echo "K10_NCU_BIN ${NCU}"
"${NCU}" --version | tee "${OUT}/ncu_version.log"

# PC-sampling sections: WarpStateStats carries smsp__pcsamp_* warp-stall
# sampling; SourceCounters maps samples to SASS addresses; SpeedOfLight for
# context.  No metric availability filter needed -- section availability is
# validated by ncu itself at profile time.
# --------------------------------------------------------------- profiling
STAGE=ncu_profile
# shellcheck disable=SC2086  # EXTRA holds space-separated KEY=VAL pairs
env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD \
    -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN2_M2 \
    ${EXTRA:-} \
    PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    "${NCU}" --target-processes all -f \
    --export "${OUT}/ncu_${ARM}" \
    --section WarpStateStats --section SourceCounters \
    --section SpeedOfLight --import-source no \
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
"${NCU}" --import "${OUT}/ncu_${ARM}.ncu-rep" --csv --page source \
    > "${OUT}/ncu_${ARM}_source.csv"
[[ -s "${OUT}/ncu_${ARM}_source.csv" ]] || {
    echo "ERROR empty source-page CSV for arm=${ARM}" >&2
    false
}

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" K10_ARM_NAME="${ARM}" python3 - <<'PY'
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
arm = os.environ["K10_ARM_NAME"]

# ncu --page source --csv: one row per SASS instruction; stall-sampling
# columns vary by version, so detect them by name.
with open(out / f"ncu_{arm}_source.csv", newline="") as f:
    rows = list(csv.reader(f))
assert len(rows) >= 2, "empty source page"
hdr = rows[0]

def cols_matching(*subs):
    return [j for j, h in enumerate(hdr)
            if all(sub.lower() in h.lower() for sub in subs)]

src_col = next((j for j, h in enumerate(hdr) if h.lower() == "source"), None)
assert src_col is not None, f"no Source column in {hdr[:12]}"
samp_all = cols_matching("sampling data (all)")
samp_ls  = cols_matching("stalled", "long")   # long scoreboard sampling col
if not samp_ls:
    samp_ls = cols_matching("long_scoreboard")
samp_bar = cols_matching("stalled", "barrier") or cols_matching("barrier")
print(f"K10_COLUMNS all={samp_all} long={samp_ls} barrier={samp_bar}")

def val(row, js):
    tot = 0
    for j in js:
        if j < len(row):
            try:
                tot += int(float(row[j].replace(",", "")))
            except ValueError:
                pass
    return tot

def opcode(sass):
    sass = sass.strip()
    m = re.match(r"(?:@!?P\d+\s+)?([A-Z0-9._]+)", sass)
    return m.group(1) if m else sass[:16]

by_op_all, by_op_ls, by_op_bar = (defaultdict(int) for _ in range(3))
per_inst = []
for r in rows[1:]:
    if len(r) <= src_col or not r[src_col].strip():
        continue
    op = opcode(r[src_col])
    a, l, b = val(r, samp_all), val(r, samp_ls), val(r, samp_bar)
    by_op_all[op] += a
    by_op_ls[op] += l
    by_op_bar[op] += b
    if a:
        per_inst.append((a, l, b, r[src_col].strip()[:90]))

tot_all = sum(by_op_all.values()) or 1
top_ops = sorted(by_op_all.items(), key=lambda kv: kv[1], reverse=True)[:15]
per_inst.sort(reverse=True)

summary = {
    "arm": arm,
    "total_samples": tot_all,
    "top_opcodes_all_samples": [
        {"op": op, "samples": n, "pct": round(100 * n / tot_all, 2),
         "long_scoreboard": by_op_ls[op], "barrier": by_op_bar[op]}
        for op, n in top_ops
    ],
    "top_instructions": [
        {"samples": a, "long_scoreboard": l, "barrier": b, "sass": t}
        for a, l, b, t in per_inst[:30]
    ],
}
(out / "summary_k10.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K10_SUMMARY " + json.dumps(
    {"arm": arm, "top6_opcodes": summary["top_opcodes_all_samples"][:6]},
    sort_keys=True))

ls_by_class = defaultdict(int)
for op, n in by_op_ls.items():
    if op.startswith("LDL"):
        ls_by_class["LDL_spill"] += n
    elif "TCGEN05" in op or op.startswith("TMEM"):
        ls_by_class["TCGEN05_tmem"] += n
    elif op.startswith("LDG"):
        ls_by_class["LDG_global"] += n
    elif op.startswith("LDS"):
        ls_by_class["LDS_shared"] += n
    else:
        ls_by_class["other"] += n
print("K10_LS_CLASSES " + json.dumps(dict(ls_by_class), sort_keys=True))
winner = max(ls_by_class, key=ls_by_class.get) if ls_by_class else "none"
print(f"K10_VERDICT long_scoreboard_source={winner}")
PY

echo "K10_OK"
echo "K10_ARTIFACTS ${OUT}"
