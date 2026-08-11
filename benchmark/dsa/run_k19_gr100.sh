#!/usr/bin/env bash
# K19: deepen the baseline gather-load pipelines on GR100.
# Arms: bl (shipped, all stages 1) / blk2 (K_STAGE=2) / blk3 (K_STAGE=3).
# Metric: baseline_ms per topk (the env knobs act on the BASELINE leg;
# the candidate leg is a constant m51s correctness partner).
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K19_OUT:-${REPO}/benchmark/dsa/k19_out}"
TOPKS="${K19_TOPKS:-128,512,2048}"
mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1
STAGE=init
trap 'rc=$?; echo "K19_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT
echo "K19_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

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
run_arm blk2 DSA_BL_K_STAGE=2
run_arm blk3 DSA_BL_K_STAGE=3

STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
arms = {}
for arm in ("bl", "blk2", "blk3"):
    arms[arm] = {e["topk"]: e for e in
                 json.loads((out / f"{arm}_sweep.json").read_text())}
topks = sorted(set.intersection(*[set(v) for v in arms.values()]))
gain = {
    a: {t: round(1.0 - arms[a][t]["baseline_ms"] / arms["bl"][t]["baseline_ms"], 4)
        for t in topks}
    for a in ("blk2", "blk3")
}
base_ms = {a: {t: arms[a][t]["baseline_ms"] for t in topks} for a in arms}
corr = {a: max(arms[a][t]["max_abs_diff_dkv"] for t in topks) for a in arms}
best_arm = max(gain, key=lambda a: max(gain[a].values()))
best = max(gain[best_arm].values())
summary = {"baseline_ms": base_ms, "gain_vs_bl": gain,
           "max_corr_diff": corr}
(out / "summary_k19.json").write_text(json.dumps(summary, indent=2) + "\n")
print("K19_SUMMARY " + json.dumps(summary, sort_keys=True))
if best >= 0.05:
    verdict = f"bl_deepen_effective_{best_arm}"
elif best > -0.02:
    verdict = "bl_deepen_neutral"
else:
    verdict = "bl_deepen_regression"
print("K19_VERDICT " + verdict)
PY
echo "K19_OK"
echo "K19_ARTIFACTS ${OUT}"
