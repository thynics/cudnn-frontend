#!/usr/bin/env bash
# K15: post-K1a cliff-tax repricing (one-click, GR100).
#
# Goal context (超越 baseline campaign): the Rubin-affinity capacity
# plays (dO full-panel replication + zero-copy dV-A views, deeper
# supply rings) all require crossing the 228KB oversized cliff
# (328KB carveout, 8KB L1).  The cliff entry fee was priced pre-K1a at
# +1.28 us/tile (E3-240); K1a removed ~46% of the local traffic the
# cliff punishes, so the fee must be repriced before any capacity
# design is funded.  Two arms of the same candidate, one job:
#   m51 anchor (MIX51+K1a fat struct) vs m51s (SLIM51 slim struct,
#   28,672 B dead face-1 removed).  Expected neutral: same oversized
#   carveout, behavior-identical layout -> validates the foundation
#   for +32KB double-buffer knives.
#
# Runner contract:
#   cd <repo> && bash benchmark/dsa/run_k15_gr100.sh
#   -> success prints  K15_OK / K15_ARTIFACTS <dir>
#   -> failure prints  K15_FAILED stage=<name> exit=<rc>
#   Artifacts dir: benchmark/dsa/k15_out (override K15_OUT).
#
# Knobs: K15_TOPKS (128,512,2048), K15_WARMUP (20), K15_REPEAT (300),
# K15_ALLOW_ANY_CC=1.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K15_OUT:-${REPO}/benchmark/dsa/k15_out}"
WARMUP="${K15_WARMUP:-20}"
REPEAT="${K15_REPEAT:-300}"
TOPKS="${K15_TOPKS:-128,512,2048}"
SWEEP="${REPO}/benchmark/dsa/sweep_topk_2cta.py"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "K15_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT

echo "K15_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=${OUT}"
echo "K15_CONFIG warmup=${WARMUP} repeat=${REPEAT} topks=${TOPKS}"

# ---------------------------------------------------------------- preflight
STAGE=preflight
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,memory.total,compute_cap --format=csv,noheader,nounits || true
} > "${OUT}/environment.log" 2>&1
cat "${OUT}/environment.log"

PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
K15_ALLOW_ANY_CC="${K15_ALLOW_ANY_CC:-0}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import json
import os

import torch

cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print("K15_DEVICE " + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "compute_capability": list(cc),
    "sm_count": props.multi_processor_count,
    "torch": torch.__version__,
}, sort_keys=True))
if os.environ.get("K15_ALLOW_ANY_CC") != "1":
    assert list(cc) == [10, 7], (
        f"expected GR100 (CC 10.7), got {cc}; "
        "set K15_ALLOW_ANY_CC=1 to override"
    )

from cudnn import DSA  # noqa: F401
from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag
flag = gpu_arch_flag()
print(f"K15_GPU_ARCH_FLAG {flag}")
if os.environ.get("K15_ALLOW_ANY_CC") != "1":
    assert flag == "sm_107a", f"gpu_arch_flag()={flag}, expected sm_107a"
PY

# Per-arm profile asserts (env read once at import).
profile_check() {
    local arm="$1"; local exp_p="$2"; local exp_r="$3"; shift 3
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN1_MIX51 -u DSA_RUBIN1_PRODUCER_REGS -u DSA_RUBIN1_REDUCE_REGS -u DSA_RUBIN1_SLIM51 -u DSA_RUBIN1_KV2 "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        K15_ARM="${arm}" K15_EXP_P="${exp_p}" K15_EXP_R="${exp_r}" \
        python3 - <<'PY'
import json
import os

arm = os.environ["K15_ARM"]
exp_p = int(os.environ["K15_EXP_P"])
exp_r = int(os.environ["K15_EXP_R"])
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_rubin_1 import (
    FlashAttentionDSABackwardSm100TwoCTAV2 as K,
)
print("K15_PROFILE " + json.dumps({
    "arm": arm, "round_stages": K.ROUND_STAGES, "pds_faces": K.PDS_FACES,
    "max_smem_bytes": K.MAX_SMEM_BYTES,
    "producer_regs": K.PRODUCER_REGS, "reduce_regs": K.REDUCE_REGS,
}, sort_keys=True))
assert (K.ROUND_STAGES, K.PDS_FACES) == (5, 1) and not K.E3PAD
assert K.MAX_SMEM_BYTES == 334_848
assert (K.PRODUCER_REGS, K.REDUCE_REGS) == (exp_p, exp_r)
assert K.SLIM51
assert K.KV2 == (arm == "m51skv2")
assert K.KSCORE_STAGES == (2 if arm == "m51skv2" else 1)
PY
}
STAGE=profile_m51s
profile_check m51s 64 120 DSA_RUBIN1_MIX51=1 DSA_RUBIN1_SLIM51=1 DSA_RUBIN1_REG_K1A=1 | tee "${OUT}/profiles.log"
STAGE=profile_m51skv2
profile_check m51skv2 64 120 DSA_RUBIN1_MIX51=1 DSA_RUBIN1_SLIM51=1 DSA_RUBIN1_KV2=1 DSA_RUBIN1_REG_K1A=1 | tee -a "${OUT}/profiles.log"

# ------------------------------------------------------------------ sweeps
run_arm() {
    local arm="$1"; local impl="$2"; shift 2
    STAGE="sweep_${arm}"
    nvidia-smi --query-gpu=pstate,clocks.current.sm,power.draw --format=csv,noheader,nounits \
        > "${OUT}/clocks_${arm}_before.log" 2>&1 || true
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN1_MIX51 -u DSA_RUBIN1_PRODUCER_REGS -u DSA_RUBIN1_REDUCE_REGS -u DSA_RUBIN1_SLIM51 -u DSA_RUBIN1_KV2 "$@" \
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
run_arm m51s rubin_1 DSA_RUBIN1_MIX51=1 DSA_RUBIN1_SLIM51=1 DSA_RUBIN1_REG_K1A=1
run_arm m51skv2 rubin_1 DSA_RUBIN1_MIX51=1 DSA_RUBIN1_SLIM51=1 DSA_RUBIN1_KV2=1 DSA_RUBIN1_REG_K1A=1

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
for arm in ("m51s", "m51skv2"):
    arms[arm], errors[arm] = load(out / f"{arm}_sweep.json")
row_errors = [dict(arm=a, **e) for a in errors for e in errors[a]]
topks = sorted(set.intersection(*[set(arms[a]) for a in arms]))
assert len(topks) >= 2, (
    f"need >=2 common valid topk points, got {topks}; errors: {row_errors}"
)

try:
    import torch
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    sm_count = None
    for line in (out / "preflight.log").read_text().splitlines():
        if line.startswith("K15_DEVICE "):
            sm_count = json.loads(line.split(" ", 1)[1])["sm_count"]
            break
    assert sm_count, "sm_count unavailable"
tokens = arms["m51s"][topks[0]]["seqlen"]
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
# Pre-registered bands (register re-tune arms vs m51 anchor):
# Pre-registered: best re-tune arm vs m51 anchor, band +/-0.03
deltas = {"m51skv2": period["m51skv2"] - period["m51s"]}
d = deltas["m51skv2"]
if d <= -0.05:
    verdict = "kv2_effective"
elif d < 0.05:
    verdict = "kv2_neutral"
else:
    verdict = "kv2_regression"

summary = {
    "sm_count": sm_count,
    "tokens": tokens,
    "waves": waves,
    "row_errors": row_errors,
    "per_topk": {
        str(t): {
            arm: arms[arm][t]["candidate_ms"] for arm in arms
        } | {
            "baseline_ck": arms["m51s"][t]["baseline_ms"],
            "baseline_drift_max_pct": round(100 * (
                max(arms[a][t]["baseline_ms"] for a in arms)
                / min(arms[a][t]["baseline_ms"] for a in arms) - 1
            ), 3),
        }
        for t in topks
    },
    "steady_period_us_per_tile": {a: round(p, 4) for a, p in period.items()},
    "deltas_vs_m51s": {a: round(d, 4) for a, d in deltas.items()},
    "nk_anchor_us_per_tile": 4.4679,
    "verdict": verdict,
}
(out / "summary_k4.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("K15_SUMMARY " + json.dumps({
    "periods": summary["steady_period_us_per_tile"],
    "deltas_vs_m51s": summary["deltas_vs_m51s"],
}, sort_keys=True))
print("K15_VERDICT " + verdict)
PY

echo "K15_OK"
echo "K15_ARTIFACTS ${OUT}"
