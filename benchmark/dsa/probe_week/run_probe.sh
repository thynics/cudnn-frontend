#!/usr/bin/env bash
# Probe week M1/M2/D4/D2: compile + run math_drain_probe.cu inside the B200 worker.
# Invoked as a single-token command via:
#   b200_manager.sh with-lock --label probe-week --sync-repo <thynics> -- \
#     bash /home/scratch.longcheng_gpu/cudnn-frontend-thynics/benchmark/dsa/probe_week/run_probe.sh
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
out=/tmp/math_drain_probe
for arch in sm_100a sm_100 sm_90a; do
  if nvcc -O3 -arch=${arch} -o "${out}" "${here}/math_drain_probe.cu" 2>/tmp/nvcc_err.log; then
    echo "PROBE_BUILD arch=${arch}"
    break
  fi
  out_built=""
done
[[ -x "${out}" ]] || { echo "PROBE_BUILD_FAILED"; cat /tmp/nvcc_err.log; exit 1; }
"${out}"
