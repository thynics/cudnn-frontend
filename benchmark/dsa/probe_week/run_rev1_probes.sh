#!/usr/bin/env bash
# V4 rev1 probes runner (worker-container side).
#   PART 1: mixed_residency_probe.cu  (R14 additivity + R6 dual-squad exp + R5 REDG-under-mix)
#   PART 2: dsm_cluster_probe.py      (R1 cluster-DSM mbar semantics)  -- runs only if present
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
echo "=== PART1: mixed residency ==="
out=/tmp/mixed_residency_probe
built=""
for arch in sm_100a sm_100 sm_90a; do
  if nvcc -O3 -arch=${arch} -o "${out}" "${here}/mixed_residency_probe.cu" 2>/tmp/nvcc_err.log; then
    echo "PROBE_BUILD arch=${arch}"; built=1; break
  fi
done
if [[ -n "${built}" ]]; then "${out}"; else echo "PART1_BUILD_FAILED"; cat /tmp/nvcc_err.log; fi

echo "=== PART1b: drain channel (kernel-grade REDG shape) ==="
out2=/tmp/drain_channel_probe
b2=""
for arch in sm_100a sm_100 sm_90a; do
  if nvcc -O3 -arch=${arch} -o "${out2}" "${here}/drain_channel_probe.cu" 2>/tmp/nvcc_err2.log; then
    echo "PROBE_BUILD arch=${arch}"; b2=1; break
  fi
done
if [[ -n "${b2}" ]]; then "${out2}"; else echo "PART1B_BUILD_FAILED"; cat /tmp/nvcc_err2.log; fi

echo "=== PART2: cluster-DSM mbar semantics ==="
if [[ -f "${here}/dsm_cluster_probe.py" ]]; then
  /home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python "${here}/dsm_cluster_probe.py" || echo "PART2_FAILED rc=$?"
else
  echo "PART2_SKIPPED (dsm_cluster_probe.py not yet committed)"
fi
