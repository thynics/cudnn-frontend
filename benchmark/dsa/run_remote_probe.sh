#!/usr/bin/env bash
# Self-service remote probe runner: run a repo-relative python script on
# the managed B200 worker via the same manager/with-lock mechanics as
# run_v_w3_2_allinone.sh, without the harness legs.
#
#   ./benchmark/dsa/run_remote_probe.sh benchmark/dsa/probe_e5a_staging.py [label]
#
# The remote repo is synced to origin/dev/longcheng HEAD by the manager
# (--sync-repo); the worktree gets the allinone workspace fixes
# (interface template, compiled module, final->v2 active source) so
# `from cudnn import DSA` works.  Artifacts land in outputs/<run_id>/.
set -euo pipefail

script_rel="${1:?usage: run_remote_probe.sh <repo-relative-script> [label]}"
label="${2:-dsa-probe}"
frontend="${COMPUTELAB_B200_FRONTEND:-computelab-sc-01}"
remote_manager="/home/scratch.longcheng_gpu/dsa-b200-harness-image/b200_manager.sh"
remote_repo="/home/scratch.longcheng_gpu/cudnn-frontend-thynics"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_probe_$$"
remote_root="/home/scratch.longcheng_gpu/.dsa-probe/${run_id}"
out_dir="outputs/${run_id}"
mkdir -p "${out_dir}"

payload_local="$(mktemp)"
cat >"${payload_local}" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
script_rel="$1"; repo="$2"; result="$3"; worktree="$4"
mkdir -p "${result}"
exec > >(tee -a "${result}/remote.log") 2>&1
finish() {
  rc=$?
  git -C "${repo}" worktree remove --force "${worktree}" >/dev/null 2>&1 || true
  exit "${rc}"
}
trap finish EXIT

git -C "${repo}" worktree add --detach "${worktree}" HEAD
package="${worktree}/python/cudnn/deepseek_sparse_attention/sparse_attention_backward"
cp /home/scratch.longcheng_gpu/dsa-b200-harness-image/interface_sm100.py \
  "${package}/_interface_sm100.py"
cp "${package}/dsa_bwd_sm100_2cta_final.py" \
  "${package}/dsa_bwd_sm100_2cta_v2.py"
compiled_module="$(find "${repo}/python/cudnn" -maxdepth 1 -type f \
  -name '_compiled_module*.so' -print -quit)"
if [[ -z "${compiled_module}" ]]; then
  compiled_module="$(find /home/scratch.longcheng_gpu/cudnn-frontend/python/cudnn \
    -maxdepth 1 -type f -name '_compiled_module*.so' -print -quit)"
fi
cp "${compiled_module}" "${worktree}/python/cudnn/"

cache_root="/home/scratch.longcheng_gpu/.dsa-allinone-cache"
mkdir -p "${cache_root}/home" "${cache_root}/xdg" "${cache_root}/cuda"
export HOME="${cache_root}/home"
export XDG_CACHE_HOME="${cache_root}/xdg"
export CUDA_CACHE_PATH="${cache_root}/cuda"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${worktree}/python:${worktree}/test/python:${worktree}"
export DSA_DEV_CANDIDATE_VARIANT="v2native"
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

cd "${worktree}"
/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python "${script_rel}" \
  | tee "${result}/report.json"
REMOTE

ssh -o BatchMode=yes "${frontend}" "mkdir -p '${remote_root}'"
scp -q "${payload_local}" "${frontend}:${remote_root}/probe.sh"
ssh -o BatchMode=yes "${frontend}" "chmod 700 '${remote_root}/probe.sh'"
remote_command="$(printf "%q " \
  "${remote_manager}" with-lock --label "${label}" --sync-repo "${remote_repo}" -- \
  "${remote_root}/probe.sh" "${script_rel}" "${remote_repo}" \
  "${remote_root}/result" "${remote_root}/worktree")"
set +e
ssh -o BatchMode=yes "${frontend}" "${remote_command}"
run_rc=$?
set -e
scp -q "${frontend}:${remote_root}/result/*" "${out_dir}/" 2>/dev/null || true
echo "PROBE_RC=${run_rc} artifacts=${out_dir}"
[[ -s "${out_dir}/report.json" ]] && cat "${out_dir}/report.json"
exit "${run_rc}"
