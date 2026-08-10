#!/usr/bin/env bash
# K2 spin-despill probe, paired measurement (one-click, GR100).
#
# e3sassk1 r1: after K1a, every hot mbarrier wait loop still reloads its
# spilled mbar-base slot ([R1+0x108]-class) once per poll -- the reduce
# dkv_done wait alone is 45.5M LDL/launch (~48% of remaining compat
# LDL); register funding is ruled out (same loop spun 48.5M times under
# the original 128-reg reduce budget).  K2 = uniform try_wait pre-spin
# + token-gated pipeline call (slow path statically skipped), env-gated
# by DSA_RUBIN1_SPIN_K2 on top of K1a.  Four arms, one job, same node:
#   ck   compat + K1a        DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1
#   ck2  compat + K1a + K2   ... + DSA_RUBIN1_SPIN_K2=1
#   nk   native + K1a        DSA_RUBIN1_REG_K1A=1
#   nk2  native + K1a + K2   ... + DSA_RUBIN1_SPIN_K2=1
# Each arm runs the repo sweep (public-baseline leg included), so
# correctness cross-check and node drift normalization come free.
# K2 touches synchronization codegen, so the correctness gate is the
# headline check, not a formality.
#
# Runner contract:
#   cd <repo> && bash benchmark/dsa/run_k2_gr100.sh
#   -> success prints  K2PAIR_OK / K2PAIR_ARTIFACTS <dir>
#   -> failure prints  K2PAIR_FAILED stage=<name> exit=<rc>
#   Artifacts dir: benchmark/dsa/k2_out (override K2PAIR_OUT).
#
# Knobs: K2PAIR_TOPKS (128,512,2048), K2PAIR_WARMUP (20),
# K2PAIR_REPEAT (300), K2PAIR_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K2PAIR_OUT:-${REPO}/benchmark/dsa/k2_out}"
WARMUP="${K2PAIR_WARMUP:-20}"
REPEAT="${K2PAIR_REPEAT:-300}"
TOPKS="${K2PAIR_TOPKS:-128,512,2048}"
SWEEP="${REPO}/benchmark/dsa/sweep_topk_2cta.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K2PAIR_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K2PAIR_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K2PAIR_CONFIG warmup=${WARMUP} repeat=${REPEAT} topks=${TOPKS}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
K2PAIR_ALLOW_ANY_CC="${K2PAIR_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K2PAIR_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
}, sort_keys=True))
if os.environ.get("K2PAIR_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K2PAIR_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K2PAIR_GPU_ARCH_FLAG {flag}")
if os.environ.get("K2PAIR_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# Per-arm profile asserts (env read once at import).
profile_check() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        K2PAIR_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)

arm = os.environ["K2PAIR_ARM"]
print("K2PAIR_PROFILE " + json.dumps({
    "arm": arm,
    "round_stages": K.ROUND_STAGES,
    "pds_faces": K.PDS_FACES,
    "max_smem_bytes": K.MAX_SMEM_BYTES,
    "reg_k1a": K.REG_K1A,
    "producer_regs": K.PRODUCER_REGS,
    "reduce_regs": K.REDUCE_REGS,
    "spin_k2": K.SPIN_K2,
}, sort_keys=True))
compat = arm.startswith("c")
spin = arm.endswith("2")
if compat:
    assert (K.ROUND_STAGES, K.PDS_FACES) == (2, 1) and not K.E3PAD
    assert K.MAX_SMEM_BYTES == 232_448
else:
    assert (K.ROUND_STAGES, K.PDS_FACES) == (5, 2) and not K.E3PAD
    assert K.MAX_SMEM_BYTES == 334_848
assert K.REG_K1A and (K.PRODUCER_REGS, K.REDUCE_REGS) == (64, 120)
assert K.SPIN_K2 is spin
PY
}
STAGE=profile_ck
profile_check ck DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1 | tee "${OUT}/profiles.log"
STAGE=profile_ck2
profile_check ck2 DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1 DSA_RUBIN1_SPIN_K2=1 | tee -a "${OUT}/profiles.log"
STAGE=profile_nk
profile_check nk DSA_RUBIN1_REG_K1A=1 | tee -a "${OUT}/profiles.log"
STAGE=profile_nk2
profile_check nk2 DSA_RUBIN1_REG_K1A=1 DSA_RUBIN1_SPIN_K2=1 | tee -a "${OUT}/profiles.log"

# ------------------------------------------------------------------ sweeps
run_arm() {
    local arm="$1"; shift
    STAGE="sweep_${arm}"
    nvidia-smi --query-gpu=pstate,clocks.current.sm,power.draw --format=csv,noheader,nounits \
        > "${OUT}/clocks_${arm}_before.log" 2>&1 || true
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 "$@" \
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
run_arm ck DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1
run_arm ck2 DSA_RUBIN1_B200_COMPAT=1 DSA_RUBIN1_REG_K1A=1 DSA_RUBIN1_SPIN_K2=1
run_arm nk DSA_RUBIN1_REG_K1A=1
run_arm nk2 DSA_RUBIN1_REG_K1A=1 DSA_RUBIN1_SPIN_K2=1

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
for arm in ("ck", "ck2", "nk", "nk2"):
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
        if line.startswith("K2PAIR_DEVICE "):
            sm_count = json.loads(line.split(" ", 1)[1])["sm_count"]
            break
    assert sm_count, "sm_count unavailable"
tokens = arms["ck"][topks[0]]["seqlen"]
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
d_native = period["nk2"] - period["nk"]
d_compat = period["ck2"] - period["ck"]

def band(delta):
    if delta <= -0.05:
        return "k2_effective"
    if delta < 0.05:
        return "k2_neutral"
    return "k2_regression"

summary = {
    "sm_count": sm_count,
    "tokens": tokens,
    "waves": waves,
    "row_errors": row_errors,
    "per_topk": {
        str(t): {
            arm: arms[arm][t]["candidate_ms"] for arm in arms
        } | {
            "baseline_ck": arms["ck"][t]["baseline_ms"],
            "baseline_drift_max_pct": round(100 * (
                max(arms[a][t]["baseline_ms"] for a in arms)
                / min(arms[a][t]["baseline_ms"] for a in arms) - 1
            ), 3),
        }
        for t in topks
    },
    "steady_period_us_per_tile": {a: round(p, 4) for a, p in period.items()},
    "delta_native_k2": round(d_native, 4),
    "delta_compat_k2": round(d_compat, 4),
    "verdict_native": band(d_native),
    "verdict_compat": band(d_compat),
}
(out / "summary_k2.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K2PAIR_SUMMARY " + json.dumps({
    "periods": summary["steady_period_us_per_tile"],
    "delta_native_k2": summary["delta_native_k2"],
    "delta_compat_k2": summary["delta_compat_k2"],
}, sort_keys=True))
print("K2PAIR_VERDICT native=" + summary["verdict_native"]
      + " compat=" + summary["verdict_compat"])
PY

echo "K2PAIR_OK"
echo "K2PAIR_ARTIFACTS ${OUT}"
