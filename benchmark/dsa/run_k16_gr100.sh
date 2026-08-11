#!/usr/bin/env bash
# K16: per-warp role-timeline trace of the m51 champion (one arm, one
# topk point).  Correctness gate intact; performance numbers ignored.
# Output: k16_out/trace.json + K16_SUMMARY per-warp durations.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${K16_OUT:-${REPO}/benchmark/dsa/k16_out}"
mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
TEE_PID=$!
export PYTHONUNBUFFERED=1
STAGE=init
trap 'rc=$?; echo "K16_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR
trap 'exec 1>&- 2>&-; wait "${TEE_PID}" 2>/dev/null || true' EXIT
echo "K16_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

STAGE=trace_run
env -u DSA_RUBIN1_B200_COMPAT -u DSA_RUBIN1_E3PAD -u DSA_RUBIN1_REG_K1A \
    -u DSA_RUBIN1_SPIN_K2 -u DSA_RUBIN1_MIX51 -u DSA_RUBIN1_SLIM51 \
    -u DSA_RUBIN1_KV2 -u DSA_RUBIN1_PRODUCER_REGS -u DSA_RUBIN1_REDUCE_REGS \
    DSA_RUBIN1_MIX51=1 DSA_RUBIN1_REG_K1A=1 \
    PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    timeout --signal=KILL 600 \
    python3 "${REPO}/benchmark/dsa/sweep_topk_2cta.py" \
    --impl rubin_1 \
    --class-name FlashAttentionDSABackwardSm100TwoCTAV2 \
    --topks 512 --warmup 5 --repeat 20 \
    --trace-out "${OUT}/trace.json" \
    2>&1 | tee "${OUT}/sweep.log"
grep -q "TRACE_OUT" "${OUT}/sweep.log"

STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
t = json.loads((out / "trace.json").read_text())
assert t["version"] == 2, f"bad trace version {t['version']}"
rows = []
for key, s in t["start_ns"].items():
    f = t["finish_ns"][key]
    if s and f:
        rows.append((key, s, f, f - s))
assert rows, "no stamped warps (trace tile never ran?)"
t0 = min(s for _, s, _, _ in rows)
rows.sort(key=lambda r: r[2])
table = [
    {"warp": k, "start_us": round((s - t0) / 1e3, 2),
     "finish_us": round((f - t0) / 1e3, 2),
     "dur_us": round(d / 1e3, 2)}
    for k, s, f, d in rows
]
straggler = table[-1]
spread = round(table[-1]["finish_us"] - table[0]["finish_us"], 2)
print("K16_SUMMARY " + json.dumps({
    "warps_stamped": len(rows),
    "straggler": straggler,
    "finish_spread_us": spread,
    "first_finisher": table[0],
}, sort_keys=True))
(out / "summary_k16.json").write_text(
    json.dumps({"table": table, "straggler": straggler,
                "finish_spread_us": spread}, indent=2) + "\n")
print("K16_VERDICT straggler=" + straggler["warp"])
PY
echo "K16_OK"
echo "K16_ARTIFACTS ${OUT}"
