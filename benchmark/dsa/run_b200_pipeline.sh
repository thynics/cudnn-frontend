#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./benchmark/dsa/run_b200_pipeline.sh --impl v0|v1
      [--note SHORT_SLUG] [--output-dir PATH]

One-click DSA backward validation on B200:
  1. acquire the single global B200 pipeline lock;
  2. git pull --ff-only the fixed Computelab scratch worktree;
  3. reuse the global B200 service, or block while creating and deploying it;
  4. run correctness, uninstrumented baseline/candidate performance, and
     baseline/candidate IKET capture;
  5. generate the two Markdown/JSON span tables;
  6. download only lightweight summaries and print the Markdown tables.

Raw IKET traces remain under /home/scratch.longcheng_gpu on Computelab.
All remote output is streamed live. Failures preserve the non-zero exit code
and download stage status plus compact log tails when available.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/../.." && pwd -P)"
frontend="${DSA_COMPUTELAB_FRONTEND:-longcheng@computelab-sc-01}"
remote_runner="/home/scratch.longcheng_gpu/dsa-b200-harness-image/run_from_frontend.sh"
implementation=""
note=""
output_dir=""

while (($#)); do
  case "$1" in
    --impl)
      [[ $# -ge 2 ]] || {
        echo "ERROR: --impl requires v0 or v1" >&2
        exit 2
      }
      implementation="${2,,}"
      shift 2
      ;;
    --note)
      [[ $# -ge 2 ]] || {
        echo "ERROR: --note requires a short slug" >&2
        exit 2
      }
      note="${2,,}"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || {
        echo "ERROR: --output-dir requires a path" >&2
        exit 2
      }
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${implementation}" in
  v0|v1) ;;
  *)
    echo "ERROR: --impl must be v0 or v1" >&2
    usage >&2
    exit 2
    ;;
esac
if [[ -z "${note}" ]]; then
  note="${implementation}"
fi
if [[ ! "${note}" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]]; then
  echo "ERROR: --note must match [a-z0-9][a-z0-9_-]{0,63}" >&2
  exit 2
fi
if [[ ! "${frontend}" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: invalid DSA_COMPUTELAB_FRONTEND token" >&2
  exit 2
fi

for executable in ssh scp mktemp awk tee; do
  command -v -- "${executable}" >/dev/null 2>&1 || {
    echo "ERROR: required executable is missing: ${executable}" >&2
    exit 2
  }
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)_${note}_${implementation}"

if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/.dsa_b200_results/${run_id}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
if [[ -e "${output_dir}" ]]; then
  echo "ERROR: refusing to overwrite local output: ${output_dir}" >&2
  exit 2
fi

local_log="$(mktemp /tmp/dsa-b200-client.XXXXXXXX.log)"
cleanup() {
  rm -f -- "${local_log}"
}
trap cleanup EXIT

echo "DSA_INPUT impl=${implementation}"
echo "DSA_SOURCE_SYNC remote=git-pull-under-global-lock"

set +e
ssh -tt -o BatchMode=yes "${frontend}" \
  "${remote_runner}" \
  --impl "${implementation}" \
  --mode all \
  --note "${note}" \
  2>&1 | tee "${local_log}"
remote_rc=${PIPESTATUS[0]}
set -e

lightweight_remote="$(
  tr -d '\r' <"${local_log}" |
    awk '$1 == "LIGHTWEIGHT_RESULT" {value=$2} END {print value}'
)"
if [[ -n "${lightweight_remote}" ]]; then
  mkdir -p -- "${output_dir}"
  scp -q -r -o BatchMode=yes \
    "${frontend}:${lightweight_remote}/." \
    "${output_dir}/"
  install -m 644 -- "${local_log}" "${output_dir}/client.log"
  echo "DSA_LIGHTWEIGHT_RESULT ${output_dir}"
fi

if ((remote_rc != 0)); then
  echo "DSA_PIPELINE_FAILED exit_code=${remote_rc}" >&2
  if [[ -d "${output_dir}" ]]; then
    for status_file in \
      run_outcome.json \
      status.json \
      validation_status.json \
      trace_status.json \
      two_trace_tables_status.json; do
      if [[ -s "${output_dir}/${status_file}" ]]; then
        echo "===== ${status_file} =====" >&2
        sed -n '1,200p' "${output_dir}/${status_file}" >&2
      fi
    done
    if [[ -s "${output_dir}/failure_tail.log" ]]; then
      echo "===== failure_tail.log =====" >&2
      sed -n '1,600p' "${output_dir}/failure_tail.log" >&2
    fi
  else
    echo "No remote lightweight result was produced; inspect the streamed log above." >&2
  fi
  exit "${remote_rc}"
fi

tables="${output_dir}/two_trace_tables.md"
if [[ ! -s "${tables}" ]]; then
  echo "ERROR: pipeline passed but two_trace_tables.md is missing" >&2
  exit 1
fi

echo
cat "${tables}"
echo
echo "DSA_PIPELINE_PASSED"
echo "DSA_TABLES ${tables}"
echo "DSA_TABLES_JSON ${output_dir}/two_trace_tables.json"
echo "DSA_VALIDATION ${output_dir}/validation_summary.json"
