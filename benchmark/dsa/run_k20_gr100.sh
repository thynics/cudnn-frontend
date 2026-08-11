#!/usr/bin/env bash
# K20: deepen the baseline gather-load pipelines on GR100.
# Arms: bl / blk2q2 (K=2,QdO=2) / blk3q2 (K=3,QdO=2) -- QdO stacking.
# Metric: baseline_ms per topk (the env knobs act on the BASELINE leg;
# the candidate leg is a constant m51s correctness partner).
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K20_OUT:-${REPO}/benchmark/dsa/k20_out}"
TOPKS="${K20_TOPKS:-128,512,2048}"
mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1
STAGE=init
trap 'rc=$?; echo "K20_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT
echo "K20_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

run_arm() {
    local arm="$1"; shift
    STAGE="sweep_${arm}"
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_SPIN_K2 \
        -u DSA_RUBIN1_KV2 -u DSA_RUBIN1_TAIL \
        -u DSA_RUBIN1_PRODUCER_REGS -u DSA_RUBIN1_REDUCE_REGS \
        -u DSA_BL_K_STAGE -u DSA_BL_QDO_STAGE \
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
run_arm blk2q2 DSA_BL_K_STAGE=2 DSA_BL_QDO_STAGE=2
run_arm blk3q2 DSA_BL_K_STAGE=3 DSA_BL_QDO_STAGE=2

STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
arms = {}
for arm in ("bl", "blk2q2", "blk3q2"):
    arms[arm] = {e["topk"]: e for e in
                 json.loads((out / f"{arm}_sweep.json").read_text())}
topks = sorted(set.intersection(*[set(v) for v in arms.values()]))
gain = {
    a: {t: round(1.0 - arms[a][t]["baseline_ms"] / arms["bl"][t]["baseline_ms"], 4)
        for t in topks}
    for a in ("blk2q2", "blk3q2")
}
base_ms = {a: {t: arms[a][t]["baseline_ms"] for t in topks} for a in arms}
corr = {a: max(arms[a][t]["max_abs_diff_dkv"] for t in topks) for a in arms}
best_arm = max(gain, key=lambda a: max(gain[a].values()))
best = max(gain[best_arm].values())
summary = {"baseline_ms": base_ms, "gain_vs_bl": gain,
           "max_corr_diff": corr}
(out / "summary_k19.json").write_text(json.dumps(summary, indent=2) + "\n")
print("K20_SUMMARY " + json.dumps(summary, sort_keys=True))
if best >= 0.05:
    verdict = f"bl_deepen_effective_{best_arm}"
elif best > -0.02:
    verdict = "bl_deepen_neutral"
else:
    verdict = "bl_deepen_regression"
print("K20_VERDICT " + verdict)
PY
echo "K20_OK"
echo "K20_ARTIFACTS ${OUT}"
