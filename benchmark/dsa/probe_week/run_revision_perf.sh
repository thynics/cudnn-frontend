#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DSA_REPO:-/home/scratch.longcheng_gpu/cudnn-frontend-thynics}"
revision="${1:?usage: run_revision_perf.sh REVISION IMPL [TOPKS [WARMUP [REPEAT]]]}"
impl="${2:?usage: run_revision_perf.sh REVISION IMPL [TOPKS [WARMUP [REPEAT]]]}"
topks="${3:-2048}"
warmup="${4:-20}"
repeat="${5:-100}"
python_bin="/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python"
driver="${repo}/benchmark/dsa/probe_week/candidate_perf.py"

if [[ ! "${revision}" =~ ^[0-9A-Za-z._/-]+$ ]]; then
  echo "ERROR: unsafe revision '${revision}'" >&2
  exit 2
fi
revision_id="$(git -C "${repo}" rev-parse --verify "${revision}^{commit}")"
revision_tag="${revision_id:0:12}"
worktree_root="/home/scratch.longcheng_gpu/dsa-revision-perf"
source_repo="${worktree_root}/${revision_tag}"
mkdir -p -- "${worktree_root}"
if [[ -e "${source_repo}/.git" ]]; then
  actual_id="$(git -C "${source_repo}" rev-parse HEAD)"
  if [[ "${actual_id}" != "${revision_id}" ]]; then
    echo "ERROR: ${source_repo} is ${actual_id}, expected ${revision_id}" >&2
    exit 2
  fi
else
  git -C "${repo}" worktree add --detach "${source_repo}" "${revision_id}"
fi

export DSA_SOURCE_REPO="${source_repo}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${source_repo}/python:${source_repo}/test/python:${source_repo}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export DSA_BL_QDO_STAGE=1
export DSA_BL_K_STAGE=1
export DSA_BL_HALFK=0
export DSA_BL_KSTAGE2=0
export DSA_BL_OVPAD=0
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

echo "REVISION_PERF source=${source_repo} revision=${revision_id} impl=${impl}"
exec "${python_bin}" "${driver}" \
  --impl "${impl}" \
  --topks "${topks}" \
  --warmup "${warmup}" \
  --repeat "${repeat}"
