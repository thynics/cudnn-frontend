#!/usr/bin/env bash
# E3-SASS: spill-site localization for the fix line (one-click, GR100).
#
# Ledger context (E3_GR100悬崖税与native异常台账_20260809.md §0): the native
# anomaly is local-memory traffic x 8KB-L1, with the structure dose proven
# by STL instruction counts (compat/e3pad bit-identical 35,692,544 vs
# native 52,502,528, all at 96 regs/thread).  This leg maps WHERE those
# LDL/STL live: NCU SourceCounters gives per-SASS-instruction executed
# counts; role segmentation happens offline from the returned CSVs
# (SETMAXNREG / named-barrier ids / tcgen05 / RED markers).
#
# Two arms of the same candidate, one job:
#   compat  DSA_RUBIN1_B200_COMPAT=1  ring2/1face  (base spill map)
#   native  (both unset)              ring5/2face  (amplified map)
# (e3pad is skipped: bit-identical code to compat.)
#
# Runner contract:
#   cd <repo> && E3SASS_NCU=<chip-aware ncu> bash benchmark/dsa/run_e3sass_gr100.sh
#   -> success prints  E3SASS_OK / E3SASS_ARTIFACTS <dir>
#   -> failure prints  E3SASS_FAILED stage=<name> exit=<rc>
#   Artifacts dir: benchmark/dsa/e3sass_out (override E3SASS_OUT).
#   NOTE: the profiling artifacts (.ncu-rep + import CSVs) are written
#   BEFORE the summary stage; a summary-stage failure still leaves a
#   complete offline-analysable delivery.
#
# Knobs: E3SASS_TOPK (512), E3SASS_SKIP (8), E3SASS_COUNT (8),
# E3SASS_NCU, E3SASS_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${E3SASS_OUT:-${REPO}/benchmark/dsa/e3sass_out}"
TOPK="${E3SASS_TOPK:-512}"
SKIP="${E3SASS_SKIP:-8}"
COUNT="${E3SASS_COUNT:-8}"
PROBE="${REPO}/benchmark/dsa/ncu_e3_probe.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "E3SASS_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "E3SASS_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "E3SASS_CONFIG topk=${TOPK} skip=${SKIP} count=${COUNT}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    uname -a
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
E3SASS_ALLOW_ANY_CC="${E3SASS_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("E3SASS_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
}, sort_keys=True))
if os.environ.get("E3SASS_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set E3SASS_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"E3SASS_GPU_ARCH_FLAG {flag}")
if os.environ.get("E3SASS_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# ncu discovery: pinned-only contract for this leg (the e3str campaign
# proved exactly one chip-aware binary on these nodes; auto-discovery
# adds nothing but risk here).
STAGE=ncu_discovery
ncu_knows_chip() {
    local probe_log="${OUT}/ncu_query_probe_$(basename "$1").log"
    local attempt
    for attempt in 1 2; do
        "$1" --query-metrics > "${probe_log}" 2>&1 || true
        # Pipe-free on purpose: under pipefail, `awk | grep -q` dies of
        # SIGPIPE (141) when the match sits high in a long output.
        if awk '$1=="gpu__time_duration"{found=1} END{exit !found}' "${probe_log}"; then
            return 0
        fi
        echo "E3SASS_NCU_QUERY_ATTEMPT ${attempt} failed for $1; head:"
        head -5 "${probe_log}" | sed 's/^/    | /'
        [[ "${attempt}" == "1" ]] && sleep 10
    done
    return 1
}
if [[ -z "${E3SASS_NCU:-}" ]]; then
    echo "ERROR this leg requires a pinned chip-aware binary: set E3SASS_NCU=/path/to/ncu" >&2
    false
fi
if [[ ! -e "${E3SASS_NCU}" ]]; then
    echo "ERROR pinned E3SASS_NCU=${E3SASS_NCU} does not exist inside this container (host mount missing?)" >&2
    false
elif [[ ! -x "${E3SASS_NCU}" ]]; then
    echo "ERROR pinned E3SASS_NCU=${E3SASS_NCU} exists but is not executable" >&2
    false
elif ! ncu_knows_chip "${E3SASS_NCU}"; then
    echo "ERROR pinned E3SASS_NCU=${E3SASS_NCU} runs but its metric query lacks gpu__time_duration after 2 attempts" >&2
    false
fi
NCU="${E3SASS_NCU}"
echo "E3SASS_NCU_BIN ${NCU}"
"${NCU}" --version | tee "${OUT}/ncu_version.log"

# --------------------------------------------------------------- profiling
profile_check() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        E3SASS_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)

arm = os.environ["E3SASS_ARM"]
print("E3SASS_PROFILE " + json.dumps({
    "arm": arm,
    "round_stages": K.ROUND_STAGES,
    "pds_faces": K.PDS_FACES,
    "max_smem_bytes": K.MAX_SMEM_BYTES,
    "e3pad": K.E3PAD,
}, sort_keys=True))
if arm == "compat":
    assert (K.ROUND_STAGES, K.PDS_FACES) == (2, 1)
    assert not K.E3PAD and K.MAX_SMEM_BYTES == 232_448
else:
    assert arm == "native"
    assert (K.ROUND_STAGES, K.PDS_FACES) == (5, 2)
    assert not K.E3PAD and K.MAX_SMEM_BYTES == 334_848
PY
}
STAGE=profile_compat
profile_check compat DSA_RUBIN1_B200_COMPAT=1 | tee "${OUT}/profiles.log"
STAGE=profile_native
profile_check native | tee -a "${OUT}/profiles.log"

profile_arm() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        "${NCU}" --target-processes all -f \
        --export "${OUT}/ncu_${arm}" \
        --section SourceCounters \
        --section LaunchStats \
        --metrics gpu__time_duration.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum \
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
        > "${OUT}/ncu_${arm}_raw.csv"
    [[ -s "${OUT}/ncu_${arm}_raw.csv" ]] || {
        echo "ERROR empty raw import CSV for arm=${arm}" >&2
        false
    }
    "${NCU}" --import "${OUT}/ncu_${arm}.ncu-rep" --csv --page source \
        > "${OUT}/ncu_${arm}_source.csv" \
        2> "${OUT}/ncu_${arm}_source_import.stderr.log" || true
    if [[ ! -s "${OUT}/ncu_${arm}_source.csv" ]]; then
        echo "E3SASS_SOURCE_PAGE_EMPTY arm=${arm} (stderr head follows)"
        head -5 "${OUT}/ncu_${arm}_source_import.stderr.log" | sed 's/^/    | /'
    fi
}
STAGE=ncu_compat
profile_arm compat DSA_RUBIN1_B200_COMPAT=1
STAGE=ncu_native
profile_arm native

# ----------------------------------------------------------------- summary
# All profiling artifacts are on disk at this point; a failure below is
# an analysable delivery, not a lost run.
STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

out = Path(os.environ["OUT_DIR"])


def pick(header, *needles, exact=None):
    if exact is not None:
        for j, name in enumerate(header):
            if name == exact:
                return j
    for j, name in enumerate(header):
        low = name.lower()
        if all(n.lower() in low for n in needles):
            return j
    return None


def parse_source(arm):
    path = out / f"ncu_{arm}_source.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"source CSV missing/empty for arm={arm}")
    with open(path, newline="", errors="replace") as f:
        rows = list(csv.reader(f))
    header = None
    start = 0
    for i, r in enumerate(rows[:10]):
        cells = [c.lower() for c in r]
        if any("address" in c for c in cells) and any(
            "source" in c or "instruction" in c for c in cells
        ):
            header, start = r, i + 1
            break
    assert header is not None, (
        f"no source-page header found in {path.name}; "
        f"first row: {rows[0][:6] if rows else 'EMPTY'}"
    )
    a = pick(header, "address")
    s = pick(header, exact="Source") or pick(header, "source")
    e = (pick(header, exact="Instructions Executed")
         or pick(header, "instructions", "executed")
         or pick(header, "inst_executed"))
    assert s is not None and e is not None, (
        f"missing Source/Instructions-Executed columns in {header[:12]}"
    )
    sites = []
    for r in rows[start:]:
        if len(r) <= max(s, e):
            continue
        text = r[s].strip()
        if not text:
            continue
        try:
            executed = float(r[e].replace(",", ""))
        except ValueError:
            continue
        tokens = text.split()
        if not tokens:
            continue
        op = tokens[1] if tokens[0].startswith("@") and len(tokens) > 1 \
            else tokens[0]
        op_class = re.split(r"[.\s]", op)[0]
        addr = r[a].strip() if a is not None and a < len(r) else ""
        sites.append((addr, op_class, op, executed, text))
    assert len(sites) >= 5000, (
        f"only {len(sites)} SASS rows parsed for arm={arm} "
        "(source page likely lacks SASS view)"
    )
    return sites


def summarize(sites):
    hist = {}
    for _, op_class, _, executed, _ in sites:
        agg = hist.setdefault(op_class, [0, 0.0])
        agg[0] += 1
        agg[1] += executed
    def top(op_class, n=40):
        rows = [x for x in sites if x[1] == op_class]
        rows.sort(key=lambda x: -x[3])
        return [
            {"address": r[0], "op": r[2], "executed": r[3]}
            for r in rows[:n]
        ]
    return hist, top


summary = {}
for arm in ("compat", "native"):
    sites = parse_source(arm)
    hist, top = summarize(sites)
    ldl = hist.get("LDL", [0, 0.0])
    stl = hist.get("STL", [0, 0.0])
    assert stl[1] > 0 and ldl[1] > 0, (
        f"arm={arm}: no LDL/STL executed counts in source page"
    )
    summary[arm] = {
        "sass_rows": len(sites),
        "ldl_sites": ldl[0],
        "ldl_executed": ldl[1],
        "stl_sites": stl[0],
        "stl_executed": stl[1],
        "opcode_class_executed_top": dict(
            sorted(
                ((k, v[1]) for k, v in hist.items()),
                key=lambda kv: -kv[1],
            )[:40]
        ),
        "top_ldl": top("LDL"),
        "top_stl": top("STL"),
    }

r_stl = (summary["native"]["stl_executed"]
         / max(summary["compat"]["stl_executed"], 1e-9))
r_ldl = (summary["native"]["ldl_executed"]
         / max(summary["compat"]["ldl_executed"], 1e-9))
summary["cross_arm"] = {
    "R_stl_executed": r_stl,
    "R_ldl_executed": r_ldl,
    # r8 counter anchors: STL x1.471, LDL x1.544 (per-launch smsp counts).
    "consistent_with_r8": bool(1.25 <= r_stl <= 1.75),
}
(out / "summary_e3sass.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("E3SASS_SUMMARY " + json.dumps({
    "compat_stl_executed": summary["compat"]["stl_executed"],
    "native_stl_executed": summary["native"]["stl_executed"],
    "R_stl_executed": round(r_stl, 4),
    "R_ldl_executed": round(r_ldl, 4),
    "consistent_with_r8": summary["cross_arm"]["consistent_with_r8"],
}, sort_keys=True))
print("E3SASS_VERDICT sites_mapped")
PY

echo "E3SASS_OK"
echo "E3SASS_ARTIFACTS ${OUT}"
