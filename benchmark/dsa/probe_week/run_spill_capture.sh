#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DSA_REPO:-/home/scratch.longcheng_gpu/cudnn-frontend-thynics}"
out="${DSA_SPILL_OUT:-/home/scratch.longcheng_gpu/dsa-vkq6v-spill/baseline45_clean}"
python_bin="/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python"

mkdir -p -- "${out}/dump" "${out}/cache"
cd -- "${repo}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo}/python:${repo}/test/python:${repo}"
export CUTE_DSL_KEEP=cubin
export CUTE_DSL_LINEINFO=True
export CUTE_DSL_DUMP_DIR="${out}/dump"
export CUTE_DSL_CACHE_DIR="${out}/cache"
export CUTE_DSL_NO_CACHE=True
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# The managed worker is long-lived; never inherit another baseline experiment.
export DSA_BL_QDO_STAGE=1
export DSA_BL_K_STAGE=1
export DSA_BL_HALFK=0
export DSA_BL_KSTAGE2=0
export DSA_BL_OVPAD=0
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

printf 'SPILL_CAPTURE_ENV DSA_BL_KSTAGE2=%s CUTE_DSL_NO_CACHE=%s\n' \
  "${DSA_BL_KSTAGE2}" "${CUTE_DSL_NO_CACHE}"

"${python_bin}" - <<'PY'
import os
from cudnn.deepseek_sparse_attention.sparse_attention_backward import dsa_bwd_sm100

kernel = dsa_bwd_sm100.FlashAttentionDSABackwardSm100(512, 512, 64, 2048)
kernel._setup_attributes()
print(
    "SPILL_CAPTURE_IMPORT",
    dsa_bwd_sm100.__file__,
    "env=", os.environ.get("DSA_BL_KSTAGE2"),
    "kstage2=", kernel.load_mma_K_kstage2,
    "k_stage=", kernel.load_mma_K_stage,
    flush=True,
)
PY

exec "${python_bin}" benchmark/dsa/sweep_topk_2cta.py \
  --impl vkq6v \
  --topks 2048 \
  --seqlen 4096 \
  --nheads 128 \
  --head-dim 512 \
  --warmup 3 \
  --repeat 10 \
  --json "${out}/smoke.json"
