#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DSA_REPO:-/home/scratch.longcheng_gpu/cudnn-frontend-thynics}"
out="${DSA_SPILL_OUT:-/home/scratch.longcheng_gpu/dsa-vkq6v-spill/baseline45}"
python_bin="/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python"

mkdir -p -- "${out}/dump" "${out}/cache"
cd -- "${repo}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo}/python:${repo}/test/python:${repo}"
export CUTE_DSL_KEEP=cubin
export CUTE_DSL_LINEINFO=True
export CUTE_DSL_DUMP_DIR="${out}/dump"
export CUTE_DSL_CACHE_DIR="${out}/cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# The managed worker is long-lived; never inherit another baseline experiment.
export DSA_BL_QDO_STAGE=1
export DSA_BL_K_STAGE=1
export DSA_BL_HALFK=0
export DSA_BL_KSTAGE2=0
export DSA_BL_OVPAD=0
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

exec "${python_bin}" benchmark/dsa/sweep_topk_2cta.py \
  --impl vkq6v \
  --topks 2048 \
  --seqlen 4096 \
  --nheads 128 \
  --head-dim 512 \
  --warmup 3 \
  --repeat 10 \
  --json "${out}/smoke.json"
