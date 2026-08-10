#!/usr/bin/env bash
# K8: post-K1a cliff-tax repricing (one-click, GR100).
#
# Goal context (超越 baseline campaign): the Rubin-affinity capacity
# plays (dO full-panel replication + zero-copy dV-A views, deeper
# supply rings) all require crossing the 228KB oversized cliff
# (328KB carveout, 8KB L1).  The cliff entry fee was priced pre-K1a at
# +1.28 us/tile (E3-240); K1a removed ~46% of the local traffic the
# cliff punishes, so the fee must be repriced before any capacity
# design is funded.  Two arms of the same candidate, one job:
#   ck    compat + K1a              DSA_RUBIN1_B200_COMPAT=1 + REG_K1A=1
#   e3k1  compat + dead-pad + K1a   DSA_RUBIN1_E3PAD=1 + REG_K1A=1
# Identical protocol and live bytes; the only difference is the launch
# mode (oversized carveout).  e3k1 - ck = the post-K1a cliff fee.
#
# Runner contract:
#   cd <repo> && bash benchmark/dsa/run_k8_gr100.sh
#   -> success prints  K8_OK / K8_ARTIFACTS <dir>
#   -> failure prints  K8_FAILED stage=<name> exit=<rc>
#   Artifacts dir: benchmark/dsa/k8_out (override K8_OUT).
#
# Knobs: K8_TOPKS (128,512,2048), K8_WARMUP (20), K8_REPEAT (300),
# K8_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K8_OUT:-${REPO}/benchmark/dsa/k8_out}"
WARMUP="${K8_WARMUP:-20}"
REPEAT="${K8_REPEAT:-300}"
TOPKS="${K8_TOPKS:-128,512,2048}"
SWEEP="${REPO}/benchmark/dsa/sweep_topk_2cta.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K8_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K8_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K8_CONFIG warmup=${WARMUP} repeat=${REPEAT} topks=${TOPKS}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
K8_ALLOW_ANY_CC="${K8_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K8_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
}, sort_keys=True))
if os.environ.get("K8_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K8_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K8_GPU_ARCH_FLAG {flag}")
if os.environ.get("K8_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# Per-arm profile asserts (env read once at import).
profile_check() {
    local arm="$1"; shift
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN2_M2 "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        K8_ARM="${arm}" \
        python3 - <<'PY'
import json
import os

arm = os.environ["K8_ARM"]
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_3 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)
print("K8_PROFILE " + json.dumps({
    "arm": arm, "gens": K.ROUND_GENS_PER_TILE, "stages": K.ROUND_STAGES,
    "max_smem_bytes": K.MAX_SMEM_BYTES, "m2_exp": K.M2_EXP,
}, sort_keys=True))
assert (K.ROUND_GENS_PER_TILE, K.ROUND_STAGES) == (4, 2)
assert K.MAX_SMEM_BYTES == 334_848
assert K.M2_EXP is (arm == "r3m2")
PY
}
STAGE=profile_r3
profile_check r3 | tee "${OUT}/profiles.log"
STAGE=profile_r3m2
profile_check r3m2 DSA_RUBIN2_M2=1 | tee -a "${OUT}/profiles.log"

# ------------------------------------------------------------------ sweeps
run_arm() {
    local arm="$1"; local impl="$2"; shift 2
    STAGE="sweep_${arm}"
    nvidia-smi --query-gpu=pstate,clocks.current.sm,power.draw --format=csv,noheader,nounits \
        > "${OUT}/clocks_${arm}_before.log" 2>&1 || true
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN2_M2 "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 "${SWEEP}" \
        --impl "${impl}" \
        --class-name FlashAttentionDSABackwardSm100TwoCTAV2 \
        --topks "${TOPKS}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --json "${OUT}/${arm}_sweep.json" \
        2>&1 | tee "${OUT}/sweep_${arm}.log"
}
run_arm r3 rubin_3
run_arm r3m2 rubin_3 DSA_RUBIN2_M2=1

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
for arm in ("r3", "r3m2"):
    arms[arm], errors[arm] = load(out / f"{arm}_sweep.json")
row_errors = [dict(arm=a, **e) for a in errors for e in errors[a]]
topks = sorted(set(arms["r3"]) & set(arms["r3m2"]))
assert len(topks) >= 2, (
    f"need >=2 common valid topk points, got {topks}; errors: {row_errors}"
)

try:
    import torch
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    sm_count = None
    for line in (out / "preflight.log").read_text().splitlines():
        if line.startswith("K8_DEVICE "):
            sm_count = json.loads(line.split(" ", 1)[1])["sm_count"]
            break
    assert sm_count, "sm_count unavailable"
tokens = arms["r3"][topks[0]]["seqlen"]
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
# Pre-registered bands (rubin_2 pantry vs ck anchor):
#   <= -0.05  r3m2_effective      -> pantry pays; iterate (Q-pantry next)
#   <  +0.05  r3m2_neutral        -> structure wins eaten by cliff fee
#   >= +0.05  r3m2_regression     -> analyze (pantry TMA? first-dP? kdq gate?)
delta = period["r3m2"] - period["r3"]
if delta <= -0.05:
    verdict = "r3m2_effective"
elif delta < 0.05:
    verdict = "r3m2_neutral"
else:
    verdict = "r3m2_regression"

summary = {
    "sm_count": sm_count,
    "tokens": tokens,
    "waves": waves,
    "row_errors": row_errors,
    "per_topk": {
        str(t): {
            arm: arms[arm][t]["candidate_ms"] for arm in arms
        } | {
            "baseline_anchor": arms["r3"][t]["baseline_ms"],
            "baseline_drift_max_pct": round(100 * (
                max(arms[a][t]["baseline_ms"] for a in arms)
                / min(arms[a][t]["baseline_ms"] for a in arms) - 1
            ), 3),
        }
        for t in topks
    },
    "steady_period_us_per_tile": {a: round(p, 4) for a, p in period.items()},
    "delta_m2_on_r3": round(delta, 4),
    "nk_anchor_us_per_tile": 4.4679,
    "verdict": verdict,
}
(out / "summary_k4.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K8_SUMMARY " + json.dumps({
    "periods": summary["steady_period_us_per_tile"],
    "delta_m2_on_r3": summary["delta_m2_on_r3"],
}, sort_keys=True))
print("K8_VERDICT " + verdict)
PY

echo "K8_OK"
echo "K8_ARTIFACTS ${OUT}"
