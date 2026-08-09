#!/usr/bin/env bash
# E3-240 paired probe (one-click, GR100).
#
# Purpose: isolate the R-394 oversized-launch-mode tax.  Two arms of the
# SAME rubin_1 candidate run back-to-back on the same node:
#   compat  DSA_RUBIN1_B200_COMPAT=1  ring2/1face, 226,816 B  -> normal mode
#   e3pad   DSA_RUBIN1_E3PAD=1       identical protocol + 16 KiB dead tail
#                                     pad, 243,712 B          -> oversized
# Live bytes are offset-identical in both arms; the only difference is the
# launch mode (328 KB carveout, 8 KB L1).  e3pad minus compat = cliff tax.
#
# Runner contract (the ONLY thing the runner does):
#   cd <repo> && bash benchmark/dsa/run_e3pair_gr100.sh
#   -> on success prints  E3PAIR_OK / E3PAIR_ARTIFACTS <dir>
#   -> on failure prints  E3PAIR_FAILED stage=<name> exit=<rc>
#   Either way, dump the artifacts dir (default benchmark/dsa/e3pair_out,
#   override with E3PAIR_OUT=...).  Everything needed for analysis and
#   every error message is inside it.
#
# Prerequisites on the node (same stack as the rubin-native topk run):
#   GR100 (CC 10.7), CUDA 13.4 toolchain, torch + nvidia-cutlass-dsl
#   importable, repo python/ working (`from cudnn import DSA` must succeed),
#   gpu_arch_flag() resolving to sm_107a.
#
# Knobs: E3PAIR_WARMUP (20), E3PAIR_REPEAT (300), E3PAIR_TOPKS
# (128,256,512,1024,2048), E3PAIR_ALLOW_ANY_CC=1 to skip the CC gate.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${E3PAIR_OUT:-${REPO}/benchmark/dsa/e3pair_out}"
WARMUP="${E3PAIR_WARMUP:-20}"
REPEAT="${E3PAIR_REPEAT:-300}"
TOPKS="${E3PAIR_TOPKS:-128,256,512,1024,2048}"
SWEEP="${REPO}/benchmark/dsa/sweep_topk_2cta.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
# Unbuffered python so a signal-killed sweep still leaves its progress
# rows in the logs (one-shot silicon: forensics beat throughput).
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "E3PAIR_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
# Flush the tee process substitution before the shell exits so the final
# marker line cannot race a snapshot-style log collector.
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "E3PAIR_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "E3PAIR_CONFIG warmup=${WARMUP} repeat=${REPEAT} topks=${TOPKS}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    uname -a
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,power.draw,memory.total,compute_cap --format=csv,noheader,nounits || true
    command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "nvcc: not on PATH (ok, DSL uses its own)"
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
E3PAIR_ALLOW_ANY_CC="${E3PAIR_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("E3PAIR_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}, sort_keys=True))
if os.environ.get("E3PAIR_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set E3PAIR_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401  (public baseline wrapper importable)
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"E3PAIR_GPU_ARCH_FLAG {flag}")
if os.environ.get("E3PAIR_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# Per-arm profile asserts.  One python process per arm: the profile env
# is read once at import time.
profile_arm() {
    local arm="$1" var="$2"
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "${var}=1" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        E3PAIR_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)

arm = os.environ["E3PAIR_ARM"]
print("E3PAIR_PROFILE " + json.dumps({
    "arm": arm,
    "round_stages": K.ROUND_STAGES,
    "pds_faces": K.PDS_FACES,
    "max_smem_bytes": K.MAX_SMEM_BYTES,
    "e3pad": K.E3PAD,
    "e3pad_elements": K.E3PAD_ELEMENTS,
}, sort_keys=True))
assert K.ROUND_STAGES == 2 and K.PDS_FACES == 1
if arm == "compat":
    assert not K.E3PAD and K.MAX_SMEM_BYTES == 232_448
else:
    assert K.E3PAD and K.E3PAD_ELEMENTS == 8_192
    assert K.MAX_SMEM_BYTES == 334_848
PY
}
STAGE=profile_compat
profile_arm compat DSA_RUBIN1_B200_COMPAT | tee "${OUT}/profiles.log"
STAGE=profile_e3pad
profile_arm e3pad DSA_RUBIN1_E3PAD | tee -a "${OUT}/profiles.log"

# ------------------------------------------------------------------ sweeps
# The in-candidate storage asserts (232,448 < e3pad bytes < 262,144) run at
# first compile: a completed e3pad sweep is itself proof the probe crossed
# the legacy line without touching the 256 KiB window line.
run_arm() {
    local arm="$1" var="$2"
    STAGE="sweep_${arm}"
    nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw --format=csv,noheader,nounits \
        > "${OUT}/clocks_${arm}_before.log" 2>&1 || true
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD "${var}=1" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 "${SWEEP}" \
        --impl rubin_1 \
        --class-name FlashAttentionDSABackwardSm100TwoCTAV2 \
        --topks "${TOPKS}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --json "${OUT}/${arm}_sweep.json" \
        2>&1 | tee "${OUT}/sweep_${arm}.log"
    nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw --format=csv,noheader,nounits \
        > "${OUT}/clocks_${arm}_after.log" 2>&1 || true
}
run_arm compat DSA_RUBIN1_B200_COMPAT
run_arm e3pad DSA_RUBIN1_E3PAD

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
DIFF_GATE = 0.05  # same atol gate as the rubin-native topk run

def load(path):
    """Skip (but record) error rows; hard-gate only on numerics."""
    rows = json.loads(path.read_text())
    table, errors = {}, []
    for r in rows:
        if "error" in r:
            errors.append({"topk": r["topk"], "error": r["error"]})
            continue
        for key in ("max_abs_diff_dq", "max_abs_diff_dkv"):
            v = r[key]
            if not (v == v and v <= DIFF_GATE):  # NaN-safe finite gate
                raise SystemExit(
                    f"HARD GATE: {path.name} topk={r['topk']} "
                    f"{key}={v} exceeds {DIFF_GATE}"
                )
        table[r["topk"]] = r
    return table, errors

rc, errors_c = load(out / "compat_sweep.json")
re_, errors_e = load(out / "e3pad_sweep.json")
row_errors = (
    [dict(arm="compat", **e) for e in errors_c]
    + [dict(arm="e3pad", **e) for e in errors_e]
)
topks = sorted(set(rc) & set(re_))
assert len(topks) >= 2, (
    f"need >=2 common valid topk points, got {topks}; "
    f"row errors: {row_errors}"
)

# sm_count: prefer live torch; fall back to the preflight record so this
# block can be re-run off-node against the dumped artifacts.
try:
    import torch
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    sm_count = None
    for line in (out / "preflight.log").read_text().splitlines():
        if line.startswith("E3PAIR_DEVICE "):
            sm_count = json.loads(line.split(" ", 1)[1])["sm_count"]
            break
    assert sm_count, "sm_count unavailable (no torch, no preflight.log)"
tokens = rc[topks[0]]["seqlen"]
waves = float(tokens) / (sm_count / 2.0)

def slope_us_per_tile(table):
    xs = [t / 64.0 for t in topks]
    ys = [table[t]["candidate_ms"] for t in topks]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope_ms = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return slope_ms * 1000.0 / waves

period_compat = slope_us_per_tile(rc)
period_e3pad = slope_us_per_tile(re_)
tax = period_e3pad - period_compat
if tax >= 2.0:
    verdict = "cliff_tax_dominant"
elif tax > 0.3:
    verdict = "cliff_tax_partial"
else:
    verdict = "cliff_tax_minor"

summary = {
    "sm_count": sm_count,
    "tokens": tokens,
    "waves": waves,
    "diff_gate": DIFF_GATE,
    "row_errors": row_errors,
    "partial": bool(row_errors),
    "per_topk": {
        str(t): {
            "compat_ms": rc[t]["candidate_ms"],
            "e3pad_ms": re_[t]["candidate_ms"],
            "e3pad_over_compat": round(
                re_[t]["candidate_ms"] / rc[t]["candidate_ms"], 4
            ),
            "compat_ratio_vs_baseline": rc[t]["ratio"],
            "e3pad_ratio_vs_baseline": re_[t]["ratio"],
            "normalized_tax": round(re_[t]["ratio"] / rc[t]["ratio"], 4),
        }
        for t in topks
    },
    "steady_period_us_per_tile": {
        "compat": round(period_compat, 3),
        "e3pad": round(period_e3pad, 3),
        "cliff_tax": round(tax, 3),
    },
    "verdict": verdict,
}
(out / "summary_e3pair.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("E3PAIR_SUMMARY " + json.dumps(
    summary["steady_period_us_per_tile"], sort_keys=True))
print("E3PAIR_VERDICT " + verdict)
PY

echo "E3PAIR_OK"
echo "E3PAIR_ARTIFACTS ${OUT}"
