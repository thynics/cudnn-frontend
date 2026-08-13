#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DSA_REPO:-/home/scratch.longcheng_gpu/cudnn-frontend-thynics}"
revision="${1:?usage: run_ncu_revision.sh REVISION IMPL TAG MODE [RUN_DIR]}"
impl="${2:?usage: run_ncu_revision.sh REVISION IMPL TAG MODE [RUN_DIR]}"
tag="${3:?usage: run_ncu_revision.sh REVISION IMPL TAG MODE [RUN_DIR]}"
mode="${4:?usage: run_ncu_revision.sh REVISION IMPL TAG MODE [RUN_DIR]}"
profile_run_dir="${5:-${repo}/profile/vkq6w-r1-vs-zero-spill-20260813}"

if [[ ! "${revision}" =~ ^[0-9A-Za-z._/-]+$ ]]; then
  echo "ERROR: unsafe revision '${revision}'" >&2
  exit 2
fi
if [[ ! "${impl}" =~ ^[0-9A-Za-z_]+$ ]]; then
  echo "ERROR: unsafe implementation '${impl}'" >&2
  exit 2
fi
if [[ ! "${tag}" =~ ^[0-9A-Za-z_-]+$ ]]; then
  echo "ERROR: unsafe tag '${tag}'" >&2
  exit 2
fi
if [[ "${mode}" != "full" && "${mode}" != "source" ]]; then
  echo "ERROR: mode must be full or source" >&2
  exit 2
fi

mkdir -p -- \
  "${profile_run_dir}/harness" \
  "${profile_run_dir}/reports" \
  "${profile_run_dir}/analysis"

ncu_bin="$(command -v ncu)"
kernel_regex="regex:kernel_cutlass_kernel_.*dsa_bwd_sm100_2cta_${impl}.*FlashAttentionDSABackwardSm100TwoCTAV2"
runner="${repo}/benchmark/dsa/probe_week/run_revision_perf.sh"
report_base="${profile_run_dir}/reports/${mode}_${tag}"

export CUTE_DSL_LINEINFO=True
export CUTE_DSL_NO_CACHE=True
export CUTE_DSL_KEEP=ptx,cubin

printf 'NCU_REVISION_ENV ncu=%s revision=%s impl=%s tag=%s mode=%s\n' \
  "${ncu_bin}" "${revision}" "${impl}" "${tag}" "${mode}"
"${ncu_bin}" --version
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

ncu_args=(
  --target-processes all
  --kernel-name "${kernel_regex}"
  --launch-count 1
  --export "${report_base}"
  --force-overwrite
)
if [[ "${mode}" == "full" ]]; then
  ncu_args+=(
    --set full
    --section PmSampling
    --section PmSampling_WarpStates
  )
else
  ncu_args+=(
    --set source
    --section SourceCounters
  )
fi

"${ncu_bin}" "${ncu_args[@]}" \
  "${runner}" "${revision}" "${impl}" 2048 0 1

test -s "${report_base}.ncu-rep"
echo "NCU_REVISION_OK ${report_base}.ncu-rep"
