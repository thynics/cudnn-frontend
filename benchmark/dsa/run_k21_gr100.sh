#!/usr/bin/env bash
# K21: D256 half-tile K staging (DSA_BL_HALFK) on GR100.
# Arms: bl (shipped, HALFK=0) / blhk (DSA_BL_HALFK=1).
# HALFK stages K/V as 2x (64, 256) half-D tiles per K stage (same bytes,
# 2x gather pipeline depth); S/dP accumulate across the halves and half 0
# is recycled to the gather warps after dQ1.
# Metric: baseline_ms per topk (the env knob acts on the BASELINE leg;
# the candidate leg is a constant m51s correctness partner).
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K21_OUT:-${REPO}/benchmark/dsa/k21_out}"
TOPKS="${K21_TOPKS:-128,512,2048}"
mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1
STAGE=init
trap 'rc=$?; echo "K21_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT
echo "K21_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

run_arm() {
    local arm="$1"; shift
    STAGE="sweep_${arm}"
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_SPIN_K2 \
        -u DSA_RUBIN1_KV2 -u DSA_RUBIN1_TAIL \
        -u DSA_RUBIN1_PRODUCER_REGS -u DSA_RUBIN1_REDUCE_REGS \
        -u DSA_BL_K_STAGE -u DSA_BL_QDO_STAGE -u DSA_BL_HALFK \
        DSA_RUBIN1_MIX51=1 DSA_RUBIN1_SLIM51=1 DSA_RUBIN1_REG_K1A=1 "$@" \
        PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
        timeout --signal=KILL 600 \
        python3 "${REPO}/benchmark/dsa/sweep_topk_2cta.py" \
        --impl rubin_1 \
        --class-name FlashAttentionDSABackwardSm100TwoCTAV2 \
        --topks "${TOPKS}" --warmup 20 --repeat 300 \
        --json "${OUT}/${arm}_sweep.json" \
        2>&1 | tee "${OUT}/sweep_${arm}.log"
    grep -q "SWEEP_JSON" "${OUT}/sweep_${arm}.log"
}
run_arm bl
run_arm blhk DSA_BL_HALFK=1

STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
arms = {}
for arm in ("bl", "blhk"):
    arms[arm] = {e["topk"]: e for e in
                 json.loads((out / f"{arm}_sweep.json").read_text())}
topks = sorted(set(arms["bl"]) & set(arms["blhk"]))
gain = {t: round(1.0 - arms["blhk"][t]["baseline_ms"]
                 / arms["bl"][t]["baseline_ms"], 4)
        for t in topks}
base_ms = {a: {t: arms[a][t]["baseline_ms"] for t in topks} for a in arms}
corr = {a: max(max(arms[a][t]["max_abs_diff_dkv"],
                   arms[a][t]["max_abs_diff_dq"]) for t in topks)
        for a in arms}
max_corr = max(corr.values())
best = max(gain.values())
summary = {"baseline_ms": base_ms, "gain_vs_bl": gain,
           "max_corr_diff": corr}
(out / "summary_k21.json").write_text(json.dumps(summary, indent=2) + "\n")
print("K21_SUMMARY " + json.dumps(summary, sort_keys=True))
assert max_corr <= 0.05, (
    f"K21 correctness gate failed: max_abs_diff {max_corr} > 0.05")
if best >= 0.05:
    verdict = "halfk_effective"
elif best > -0.02:
    verdict = "halfk_neutral"
else:
    verdict = "halfk_regression"
print("K21_VERDICT " + verdict)
PY
echo "K21_OK"
echo "K21_ARTIFACTS ${OUT}"
