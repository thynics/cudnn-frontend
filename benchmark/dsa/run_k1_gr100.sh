#!/usr/bin/env bash
# K1a despill probe, paired measurement (one-click, GR100).
#
# e3sass r1 located the dominant spill bucket in the producer warps'
# wait loops (48-reg warpgroup 16-19; a single spilled slot reloads
# 60M+ times/launch).  K1a = producer 48->64, reduce 128->120 (same
# 61,440-register pool), env-gated by DSA_RUBIN1_REG_K1A so one
# revision carries both arms.  Four arms, one job, same node:
#   cb  compat base    DSA_RUBIN1_B200_COMPAT=1
#   ck  compat + K1a   DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1
#   nb  native base    (none)
#   nk  native + K1a   DSA_RUBIN1_REG_K1A=1
# Each arm runs the repo sweep (public-baseline leg included), so
# correctness cross-check and node drift normalization come free.
#
# Runner contract:
#   cd <repo> && bash benchmark/dsa/run_k1_gr100.sh
#   -> success prints  K1PAIR_OK / K1PAIR_ARTIFACTS <dir>
#   -> failure prints  K1PAIR_FAILED stage=<name> exit=<rc>
#   Artifacts dir: benchmark/dsa/k1_out (override K1PAIR_OUT).
#
# Knobs: K1PAIR_TOPKS (128,512,2048), K1PAIR_WARMUP (20),
# K1PAIR_REPEAT (300), K1PAIR_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K1PAIR_OUT:-${REPO}/benchmark/dsa/k1_out}"
WARMUP="${K1PAIR_WARMUP:-20}"
REPEAT="${K1PAIR_REPEAT:-300}"
TOPKS="${K1PAIR_TOPKS:-128,512,2048}"
SWEEP="${REPO}/benchmark/dsa/sweep_topk_2cta.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K1PAIR_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K1PAIR_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K1PAIR_CONFIG warmup=${WARMUP} repeat=${REPEAT} topks=${TOPKS}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
K1PAIR_ALLOW_ANY_CC="${K1PAIR_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K1PAIR_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
}, sort_keys=True))
if os.environ.get("K1PAIR_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K1PAIR_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K1PAIR_GPU_ARCH_FLAG {flag}")
if os.environ.get("K1PAIR_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# Per-arm profile asserts (env read once at import).
profile_check() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        K1PAIR_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)

arm = os.environ["K1PAIR_ARM"]
print("K1PAIR_PROFILE " + json.dumps({
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
if compat:
    assert (K.ROUND_STAGES, K.PDS_FACES) == (2, 1) and not K.E3PAD
    assert K.MAX_SMEM_BYTES == 232_448
else:
    assert (K.ROUND_STAGES, K.PDS_FACES) == (5, 2) and not K.E3PAD
    assert K.MAX_SMEM_BYTES == 334_848
if k1a:
    assert K.REG_K1A and (K.PRODUCER_REGS, K.REDUCE_REGS) == (64, 120)
else:
    assert not K.REG_K1A and (K.PRODUCER_REGS, K.REDUCE_REGS) == (48, 128)
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

# ------------------------------------------------------------------ sweeps
run_arm() {
    local arm="$1"; shift
    STAGE="sweep_${arm}"
    nvidia-smi --query-gpu=pstate,clocks.current.sm,power.draw --format=csv,noheader,nounits \
        > "${OUT}/clocks_${arm}_before.log" 2>&1 || true
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 "${SWEEP}" \
        --impl rubin_1 \
        --class-name FlashAttentionDSABackwardSm100TwoCTAV2 \
        --topks "${TOPKS}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --json "${OUT}/${arm}_sweep.json" \
        2>&1 | tee "${OUT}/sweep_${arm}.log"
}
run_arm cb DSA_RUBIN1_B200_COMPAT=1
run_arm ck DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1
run_arm nb
run_arm nk DSA_RUBIN1_REG_K1A=1

# ----------------------------------------------------------------- summary
STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
DIFF_GATE = 0.05

def load(path):
    rows = json.loads(path.read_text())
    table, errors = {}, []
    for r in rows:
        if "error" in r:
            errors.append({"topk": r["topk"], "error": r["error"]})
            continue
        for key in ("max_abs_diff_dq", "max_abs_diff_dkv"):
            v = r[key]
            if not (v == v and v <= DIFF_GATE):
                raise SystemExit(
                    f"HARD GATE: {path.name} topk={r['topk']} "
                    f"{key}={v} exceeds {DIFF_GATE}"
                )
        table[r["topk"]] = r
    return table, errors

arms, errors = {}, {}
for arm in ("cb", "ck", "nb", "nk"):
    arms[arm], errors[arm] = load(out / f"{arm}_sweep.json")
row_errors = [
    dict(arm=a, **e) for a in errors for e in errors[a]
]
topks = sorted(set.intersection(*[set(t) for t in arms.values()]))
assert len(topks) >= 2, (
    f"need >=2 common valid topk points, got {topks}; errors: {row_errors}"
)

try:
    import torch
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    sm_count = None
    for line in (out / "preflight.log").read_text().splitlines():
        if line.startswith("K1PAIR_DEVICE "):
            sm_count = json.loads(line.split(" ", 1)[1])["sm_count"]
            break
    assert sm_count, "sm_count unavailable"
tokens = arms["cb"][topks[0]]["seqlen"]
waves = float(tokens) / (sm_count / 2.0)

def slope_us_per_tile(table):
    xs = [t / 64.0 for t in topks]
    ys = [table[t]["candidate_ms"] for t in topks]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return m * 1000.0 / waves

period = {arm: slope_us_per_tile(t) for arm, t in arms.items()}
d_native = period["nk"] - period["nb"]
d_compat = period["ck"] - period["cb"]

def band(delta):
    if delta <= -0.30:
        return "k1a_effective"
    if delta < 0.10:
        return "k1a_neutral"
    return "k1a_harmful"

summary = {
    "sm_count": sm_count,
    "tokens": tokens,
    "waves": waves,
    "row_errors": row_errors,
    "per_topk": {
        str(t): {
            arm: arms[arm][t]["candidate_ms"] for arm in arms
        } | {
            "baseline_cb": arms["cb"][t]["baseline_ms"],
            "baseline_drift_max_pct": round(100 * (
                max(arms[a][t]["baseline_ms"] for a in arms)
                / min(arms[a][t]["baseline_ms"] for a in arms) - 1
            ), 3),
        }
        for t in topks
    },
    "steady_period_us_per_tile": {a: round(p, 4) for a, p in period.items()},
    "delta_native_k1a": round(d_native, 4),
    "delta_compat_k1a": round(d_compat, 4),
    "verdict_native": band(d_native),
    "verdict_compat": band(d_compat),
}
(out / "summary_k1.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K1PAIR_SUMMARY " + json.dumps({
    "periods": summary["steady_period_us_per_tile"],
    "delta_native_k1a": summary["delta_native_k1a"],
    "delta_compat_k1a": summary["delta_compat_k1a"],
}, sort_keys=True))
print("K1PAIR_VERDICT native=" + summary["verdict_native"]
      + " compat=" + summary["verdict_compat"])
PY

echo "K1PAIR_OK"
echo "K1PAIR_ARTIFACTS ${OUT}"
