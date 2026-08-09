#!/usr/bin/env bash
# baseline_opt knife panel (one-click, B200 / SM100).
#
# Purpose: quantify the three back-ported per-launch knives on the production
# baseline and settle the topk-scaling comparison.  Six sweep legs run
# back-to-back on the same node, each a fresh python process (the knife env
# flags are baked at module import):
#   final         impl=final   FlashAttentionDSABackwardSm100TwoCTAV2
#   bopt_all_off  impl=baseline_opt, EPI=0 DQ_EARLY=0 SPLIT_QDO=0  (fork
#                 sanity + dkv nondeterminism noise floor)
#   bopt_all_on   impl=baseline_opt, all knives on
#   bopt_no_epi   knife 1 off, 2/3 on
#   bopt_no_dqe   knife 2 off, 1/3 on
#   bopt_no_split knife 3 off, 1/2 on
# Every leg re-measures the production-wrapper baseline, giving six
# baseline_ms samples per topk = a free node-stability estimate.
#
# Runner contract (the ONLY thing the runner does):
#   cd <repo> && bash benchmark/dsa/run_bopt_sweep_b200.sh
#   -> on success prints  BOPT_OK / BOPT_ARTIFACTS <dir>
#   -> on failure prints  BOPT_FAILED stage=<name> exit=<rc>
#   Either way, collect the artifacts dir (default
#   benchmark/dsa/bopt_sweep_out, override with BOPT_OUT=...).  Everything
#   needed for analysis and every error message is inside it.
#
# Prerequisites on the node (same stack as the standard B200 pipeline):
#   B200 (CC 10.0), torch + nvidia-cutlass-dsl importable, repo python/
#   working (`from cudnn import DSA` must succeed).
#
# Knobs: BOPT_WARMUP (20), BOPT_REPEAT (200), BOPT_TOPKS
# (128,256,512,1024,2048), BOPT_ALLOW_ANY_CC=1 to skip the CC gate.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${BOPT_OUT:-${REPO}/benchmark/dsa/bopt_sweep_out}"
WARMUP="${BOPT_WARMUP:-20}"
REPEAT="${BOPT_REPEAT:-200}"
TOPKS="${BOPT_TOPKS:-128,256,512,1024,2048}"
SWEEP="${REPO}/benchmark/dsa/sweep_topk_2cta.py"
BOPT_CLASS="FlashAttentionDSABackwardSm100BaselineOpt"
FINAL_CLASS="FlashAttentionDSABackwardSm100TwoCTAV2"

mkdir -p -- "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
export PYTHONUNBUFFERED=1

STAGE=init
trap 'rc=$?; echo "BOPT_FAILED stage=${STAGE} exit=${rc} (artifacts: ${OUT})"' ERR

STAGE=preflight
echo "=== preflight ==="
sha256sum \
  "${REPO}/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_baseline_opt.py" \
  "${REPO}/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py" \
  "${REPO}/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_final.py" \
  "${REPO}/benchmark/dsa/sweep_topk_2cta.py" \
  "${REPO}/benchmark/dsa/run_bopt_sweep_b200.sh" \
  | tee "${OUT}/sha256.txt"
PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
REPO_DIR="${REPO}" \
python3 - <<'PY' | tee "${OUT}/preflight.log"
import importlib
import os

import torch

name = torch.cuda.get_device_name()
cc = torch.cuda.get_device_capability()
print(f"device={name} cc={cc} torch={torch.__version__}")
if os.environ.get("BOPT_ALLOW_ANY_CC") != "1":
    assert cc == (10, 0), f"expected B200 (CC 10.0), got {cc}"
from cudnn import DSA  # noqa: F401  (wrapper import gate)
pkg = "cudnn.deepseek_sparse_attention.sparse_attention_backward"
m_opt = importlib.import_module(f"{pkg}.dsa_bwd_sm100_2cta_baseline_opt")
m_fin = importlib.import_module(f"{pkg}.dsa_bwd_sm100_2cta_final")
assert hasattr(m_opt, "FlashAttentionDSABackwardSm100BaselineOpt")
assert hasattr(m_fin, "FlashAttentionDSABackwardSm100TwoCTAV2")
print("preflight OK")
PY

run_leg() {
  # run_leg <name> <impl> <class> <epi> <dq_early> <split_qdo>
  local name="$1" impl="$2" cls="$3" epi="$4" dqe="$5" spl="$6"
  STAGE="leg_${name}"
  echo "=== leg ${name}: impl=${impl} EPI=${epi} DQ_EARLY=${dqe} SPLIT_QDO=${spl} ==="
  DSA_BOPT_EPI="${epi}" \
  DSA_BOPT_DQ_EARLY="${dqe}" \
  DSA_BOPT_SPLIT_QDO="${spl}" \
  PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${SWEEP}" \
    --impl "${impl}" \
    --class-name "${cls}" \
    --topks "${TOPKS}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --json "${OUT}/${name}.json" \
    | tee "${OUT}/${name}.log"
}

run_leg final         final        "${FINAL_CLASS}" 1 1 1
run_leg bopt_all_off  baseline_opt "${BOPT_CLASS}"  0 0 0
run_leg bopt_all_on   baseline_opt "${BOPT_CLASS}"  1 1 1
run_leg bopt_no_epi   baseline_opt "${BOPT_CLASS}"  0 1 1
run_leg bopt_no_dqe   baseline_opt "${BOPT_CLASS}"  1 0 1
run_leg bopt_no_split baseline_opt "${BOPT_CLASS}"  1 1 0

STAGE=summary
OUT_DIR="${OUT}" python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
legs = ["final", "bopt_all_off", "bopt_all_on",
        "bopt_no_epi", "bopt_no_dqe", "bopt_no_split"]
data = {}
for leg in legs:
    data[leg] = {row["topk"]: row for row in json.loads((out / f"{leg}.json").read_text())}
topks = sorted(data["bopt_all_on"].keys())

gates = {"gate_dq_bitexact": True, "gate_dkv_noise": True, "gate_node_stable": True}
lines = []
lines.append("# baseline_opt B200 topk panel\n")
lines.append("| topk | base ms (6腿 min..max, spread%) | final ms (ratio) | "
             "bopt_all_on ms (ratio) | Δ①epi | Δ②dq_early | Δ③split | "
             "bopt dq diff max | bopt dkv diff (floor→max) |")
lines.append("|---|---|---|---|---|---|---|---|---|")
summary = []
for t in topks:
    rows = {leg: data[leg].get(t, {}) for leg in legs}
    if any("error" in r or "baseline_ms" not in r for r in rows.values()):
        errs = {leg: rows[leg].get("error", "missing") for leg in legs
                if "error" in rows[leg] or "baseline_ms" not in rows[leg]}
        lines.append(f"| {t} | LEG ERROR {errs} | | | | | | | |")
        summary.append({"topk": t, "error": str(errs)})
        for g in gates:
            gates[g] = False
        continue
    bases = [rows[leg]["baseline_ms"] for leg in legs]
    spread = (max(bases) - min(bases)) / min(bases)
    if spread > 0.02:
        gates["gate_node_stable"] = False
    base_ref = min(bases)
    fin = rows["final"]["candidate_ms"]
    allon = rows["bopt_all_on"]["candidate_ms"]
    alloff = rows["bopt_all_off"]["candidate_ms"]
    d_epi = rows["bopt_no_epi"]["candidate_ms"] - allon
    d_dqe = rows["bopt_no_dqe"]["candidate_ms"] - allon
    d_spl = rows["bopt_no_split"]["candidate_ms"] - allon
    bopt_legs = [l for l in legs if l.startswith("bopt_")]
    dq_max = max(rows[l]["max_abs_diff_dq"] for l in bopt_legs)
    if dq_max != 0.0:
        gates["gate_dq_bitexact"] = False
    dkv_floor = rows["bopt_all_off"]["max_abs_diff_dkv"]
    dkv_max = max(rows[l]["max_abs_diff_dkv"] for l in bopt_legs)
    if dkv_max > max(4.0 * dkv_floor, 0.05):
        gates["gate_dkv_noise"] = False
    lines.append(
        f"| {t} | {min(bases):.3f}..{max(bases):.3f} ({spread * 100:.1f}%) "
        f"| {fin:.3f} ({fin / base_ref:.3f}) "
        f"| {allon:.3f} ({allon / base_ref:.3f}) "
        f"| {d_epi:+.3f} | {d_dqe:+.3f} | {d_spl:+.3f} "
        f"| {dq_max:.4g} | {dkv_floor:.4g}->{dkv_max:.4g} |")
    summary.append({
        "topk": t, "baseline_ms_min": min(bases), "baseline_ms_max": max(bases),
        "baseline_spread": round(spread, 4),
        "final_ms": fin, "final_ratio": round(fin / base_ref, 4),
        "bopt_all_on_ms": allon, "bopt_ratio": round(allon / base_ref, 4),
        "bopt_all_off_ms": alloff,
        "knife_epi_ms": round(d_epi, 4), "knife_dq_early_ms": round(d_dqe, 4),
        "knife_split_ms": round(d_spl, 4),
        "dq_diff_max": dq_max, "dkv_floor": dkv_floor, "dkv_max": dkv_max,
    })
lines.append("")
for g, ok in gates.items():
    lines.append(f"- {g}: {'PASS' if ok else 'FAIL'}")
(out / "summary.json").write_text(json.dumps(
    {"gates": gates, "rows": summary}, indent=2))
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "BOPT_OK"
echo "BOPT_ARTIFACTS ${OUT}"
