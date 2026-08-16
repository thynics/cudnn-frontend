#!/usr/bin/env bash
set -Eeuo pipefail

revision="${1:?revision}"
result_dir="${2:?result_dir}"
driver="${3:?driver}"
expected_baseline_sha="${4:?baseline sha256}"
candidate_source="${5:?candidate source}"
expected_candidate_sha="${6:?candidate sha256}"

repo="/home/scratch.longcheng_gpu/cudnn-frontend-thynics"
worktree="${result_dir}/worktree"
package_rel="python/cudnn/deepseek_sparse_attention/sparse_attention_backward"
python_bin="/home/scratch.longcheng_gpu/.dsa-rubin/vpagealias-b-300k-20260816/.venv-rubin/bin/python3"
interface_template="/home/scratch.longcheng_gpu/dsa-b200-harness-image/interface_sm100.py"
candidate_suffix="${DSA_RUN_CANDIDATE:-variant}"
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
[[ "${expected_baseline_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${expected_candidate_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${result_dir}" == /home/scratch.longcheng_gpu/* ]]
[[ "${candidate_suffix}" =~ ^[a-z0-9_]+$ ]]
[[ -s "${driver}" && -s "${candidate_source}" ]]
[[ "$(sha256sum "${candidate_source}" | awk '{print $1}')" == "${expected_candidate_sha}" ]]

stage="create_worktree"
git -C "${repo}" cat-file -e "${revision}^{commit}"
git -C "${repo}" worktree add --detach "${worktree}" "${revision}"

package="${worktree}/${package_rel}"
baseline_source="${package}/dsa_bwd_sm100_baseline.py"
stage="install_variant"
[[ "$(sha256sum "${baseline_source}" | awk '{print $1}')" == "${expected_baseline_sha}" ]]
cp -- "${candidate_source}" "${package}/dsa_bwd_sm100_2cta_${candidate_suffix}.py"
cp -- "${interface_template}" "${package}/_interface_sm100.py"
cp -- \
  /home/scratch.longcheng_gpu/cudnn-frontend/python/cudnn/_compiled_module.cpython-312-x86_64-linux-gnu.so \
  "${worktree}/python/cudnn/"

cache_root="/home/scratch.longcheng_gpu/.dsa-rubin-cache-internal-nightly-20260803-cuda134"
mkdir -p -- "${cache_root}/home" "${cache_root}/xdg" "${cache_root}/cuda" \
  "${cache_root}/cute"
export HOME="${cache_root}/home"
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

run_topks="${DSA_RUN_TOPKS:-128,256,512,1024,2048}"
run_warmup_pairs="${DSA_RUN_WARMUP_PAIRS:-4}"
run_paired_samples="${DSA_RUN_PAIRED_SAMPLES:-16}"
stage="benchmark"
timeout --signal=TERM --kill-after=30s 1200 \
  "${python_bin}" "${driver}" \
  --repo "${worktree}" \
  --candidate "${candidate_suffix}" \
  --topks "${run_topks}" \
  --seqlen 4096 \
  --nheads 128 \
  --head-dim 512 \
  --warmup-pairs "${run_warmup_pairs}" \
  --paired-samples "${run_paired_samples}" \
  --candidate-smem-mode oversized \
  --output "${result_dir}/compare.json"
stage="complete"
