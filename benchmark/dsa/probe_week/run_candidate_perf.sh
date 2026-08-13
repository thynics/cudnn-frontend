#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DSA_REPO:-/home/scratch.longcheng_gpu/cudnn-frontend-thynics}"
impl="${1:?usage: run_candidate_perf.sh IMPL [TOPKS [WARMUP [REPEAT]]]}"
topks="${2:-2048}"
warmup="${3:-20}"
repeat="${4:-100}"
python_bin="/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python"

cd -- "${repo}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo}/python:${repo}/test/python:${repo}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export DSA_BL_QDO_STAGE=1
export DSA_BL_K_STAGE=1
export DSA_BL_HALFK=0
export DSA_BL_KSTAGE2=0
export DSA_BL_OVPAD=0
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

exec "${python_bin}" benchmark/dsa/sweep_topk_2cta.py \
  --impl "${impl}" \
  --topks "${topks}" \
  --warmup "${warmup}" \
  --repeat "${repeat}"
