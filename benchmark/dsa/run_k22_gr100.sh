#!/usr/bin/env bash
# K22: A-S1 stage=2 revival (DSA_BL_KSTAGE2) on GR100, with the oversized
# launch tax priced in isolation (DATAFLOW_REDESIGN_20260811 §3.3).
# Arms: bl   (shipped baseline, compat launch, K stage=1)
#       blo1 (DSA_BL_OVPAD=1: oversized launch only, stage=1 -- tax dose)
#       blo2 (DSA_BL_KSTAGE2=1: oversized + K stage=2 -- main verdict arm)
# Metric: baseline_ms per topk (the env knobs act on the BASELINE leg;
# the candidate leg is a constant m51s correctness partner).
# Verdict: steady-slope delta via the (ms2048 - ms512) differential,
# converted to us/tile with 24 tiles x 37.93 waves (8192 CTA / 216 SM).
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K22_OUT:-${REPO}/benchmark/dsa/k22_out}"
TOPKS="${K22_TOPKS:-128,512,2048}"
mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1
STAGE=init
trap 'rc=$?; echo "K22_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT
echo "K22_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

run_arm() {
    local arm="$1"; shift
    STAGE="sweep_${arm}"
    env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_SPIN_K2 \
        -u DSA_RUBIN1_KV2 -u DSA_RUBIN1_TAIL \
        -u DSA_RUBIN1_PRODUCER_REGS -u DSA_RUBIN1_REDUCE_REGS \
        -u DSA_BL_K_STAGE -u DSA_BL_QDO_STAGE -u DSA_BL_HALFK \
        -u DSA_BL_KSTAGE2 -u DSA_BL_OVPAD \
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
run_arm blo1 DSA_BL_OVPAD=1
run_arm blo2 DSA_BL_KSTAGE2=1

STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
arms = {}
for arm in ("bl", "blo1", "blo2"):
    arms[arm] = {e["topk"]: e for e in
                 json.loads((out / f"{arm}_sweep.json").read_text())}
topks = sorted(set.intersection(*[set(v) for v in arms.values()]))
assert 512 in topks and 2048 in topks, (
    f"slope needs topk 512 and 2048; got {topks}")

# Steady slope via the (ms2048 - ms512) differential (design doc G2):
# 24 tiles between the two points, 37.93 waves (8192 CTA / 216 SM).
TILES_DIFF = (2048 - 512) // 64
WAVES = 8192 / 216.0


def slope_us_per_tile(arm):
    d_ms = arms[arm][2048]["baseline_ms"] - arms[arm][512]["baseline_ms"]
    return d_ms * 1000.0 / TILES_DIFF / WAVES


slopes = {a: round(slope_us_per_tile(a), 4) for a in arms}
tax = round(slopes["blo1"] - slopes["bl"], 4)     # oversized dose
main = round(slopes["blo2"] - slopes["bl"], 4)    # main verdict
base_ms = {a: {t: arms[a][t]["baseline_ms"] for t in topks} for a in arms}
corr = {a: max(max(arms[a][t]["max_abs_diff_dkv"],
                   arms[a][t]["max_abs_diff_dq"]) for t in topks)
        for a in arms}
max_corr = max(corr.values())

if tax <= 0.05:
    tax_verdict = "tax_ok"
elif tax <= 0.15:
    tax_verdict = "tax_elevated"
else:
    tax_verdict = "tax_red_light"  # G2: oversized route red light

if main <= -0.40:
    verdict = "kstage2_major_win"
elif main <= -0.10:
    verdict = "kstage2_effective"
elif main <= 0.10:
    verdict = "kstage2_neutral_bw_bound"
else:
    verdict = "kstage2_regression"

summary = {"baseline_ms": base_ms, "slope_us_per_tile": slopes,
           "tax_blo1_minus_bl": tax, "delta_blo2_minus_bl": main,
           "max_corr_diff": corr, "tax_verdict": tax_verdict}
(out / "summary_k22.json").write_text(json.dumps(summary, indent=2) + "\n")
print("K22_SUMMARY " + json.dumps(summary, sort_keys=True))
assert max_corr <= 0.05, (
    f"K22 correctness gate failed: max_abs_diff {max_corr} > 0.05")
print("K22_TAX_VERDICT " + tax_verdict)
print("K22_VERDICT " + verdict)
PY
echo "K22_OK"
echo "K22_ARTIFACTS ${OUT}"
