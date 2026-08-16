#!/usr/bin/env bash
set -Eeuo pipefail

revision="${1:?revision}"
result_dir="${2:?result_dir}"
driver="${3:?driver}"
candidate_source="${4:?candidate source}"
expected_candidate_sha="${5:?candidate sha256}"
case_name="${6:-all}"
diagnose_dkv_columns="${7:-0}"

repo="/home/scratch.longcheng_gpu/cudnn-frontend-thynics"
worktree="${result_dir}/worktree"
python_bin="/home/scratch.longcheng_gpu/.dsa-rubin/vpagealias-b-300k-20260816/.venv-rubin/bin/python3"
interface_template="/home/scratch.longcheng_gpu/dsa-b200-harness-image/interface_sm100.py"
package_rel="python/cudnn/deepseek_sparse_attention/sparse_attention_backward"
stage="bootstrap"

mkdir -p -- "${result_dir}"
exec > >(tee -a "${result_dir}/remote.log") \
  2> >(tee -a "${result_dir}/remote.stderr.log" >&2)

finish() {
  local rc=$?
  set +e
  if [[ -e "${worktree}/.git" ]]; then
    git -C "${repo}" worktree remove --force "${worktree}" >/dev/null 2>&1 || true
  fi
  printf '{"exit_code":%d,"stage":"%s"}\n' "${rc}" "${stage}" \
    >"${result_dir}/status.json"
  trap - EXIT
  exit "${rc}"
}
trap finish EXIT

[[ "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${expected_candidate_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${result_dir}" == /home/scratch.longcheng_gpu/* ]]
[[ -s "${driver}" && -s "${candidate_source}" && -x "${python_bin}" ]]
[[ "$(sha256sum "${candidate_source}" | awk '{print $1}')" == "${expected_candidate_sha}" ]]

stage="create_worktree"
git -C "${repo}" cat-file -e "${revision}^{commit}"
git -C "${repo}" worktree add --detach "${worktree}" "${revision}"

stage="prepare_runtime"
cp -- "${interface_template}" "${worktree}/${package_rel}/_interface_sm100.py"
cp -- \
  /home/scratch.longcheng_gpu/cudnn-frontend/python/cudnn/_compiled_module.cpython-312-x86_64-linux-gnu.so \
  "${worktree}/python/cudnn/"

cache_root="/home/scratch.longcheng_gpu/.dsa-rubin-cache-internal-nightly-20260803-cuda134"
mkdir -p -- "${cache_root}/xdg" "${cache_root}/cuda" "${cache_root}/cute"
export XDG_CACHE_HOME="${cache_root}/xdg"
export CUDA_CACHE_PATH="${cache_root}/cuda"
export CUTE_DSL_CACHE_DIR="${cache_root}/cute"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${worktree}/python:${worktree}/test/python:${worktree}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cuda134="/home/scratch.longcheng_gpu/.cuda-toolkit/cuda-38248128-min"
export CUDA_HOME="${cuda134}"
export CUDA_PATH="${cuda134}"
export CUDA_TOOLKIT_PATH="${cuda134}"
export PATH="${cuda134}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda134}/nvvm/lib:${cuda134}/nvvm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUTE_DSL_ARCH=sm_107a
export CUTE_DSL_ENABLE_TVM_FFI=1
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

stage="correctness"
diagnostic_args=()
if [[ "${diagnose_dkv_columns}" == "1" ]]; then
  diagnostic_args+=(--diagnose-dkv-columns)
fi
timeout --signal=TERM --kill-after=30s 1200 \
  "${python_bin}" "${driver}" \
  --repo "${worktree}" \
  --candidate-source "${candidate_source}" \
  --case "${case_name}" \
  "${diagnostic_args[@]}" \
  --output "${result_dir}/dynamic_correctness.json"
stage="complete"
