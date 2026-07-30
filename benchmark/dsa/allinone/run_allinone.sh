#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./benchmark/dsa/allinone/run_allinone.sh --impl vNAME
      [--artifact-dir DIR]        default /home/longcheng/artifact/<impl>_run<N>
      [--windows 1-3,14-17]       trace tile windows for the readout
      [--skip stage0|pipeline|ncu]  repeatable
      [--gates PATH]              stage-0 expectations JSON
      [--stage0-capture DIR]      pre-existing compile capture (skips compiling)
      [--reference-capture DIR]   matched reference capture for stage-0

One command per validation round.  Sequences the four things every round
repeats, with the same STOP/publish protocol every time:

  stage0    compile-only SASS gates   (DSA_STAGE0_CMD or --stage0-capture)
  pipeline  the repository's one-click B200 validation (correctness,
            release timing, IKET captures, span tables)
  ncu       optional companion profile (DSA_NCU_CMD)
  readout   round_readout.py over the decoded trace + perf JSONs
  publish   <artifact-dir>.partial -> MANIFEST.sha256 -> atomic rename;
            any STOP writes <artifact-dir>.FAILED instead

Command hooks (environment; each optional, stage SKIPPED when unset):
  DSA_STAGE0_CMD   produces a compile capture directory; invoked as:
                     $DSA_STAGE0_CMD <impl> <capture-out-dir>
                   (typically the private compile helper with
                    CUTE_DSL_KEEP=ptx,cubin,sass)
  DSA_PIPELINE_CMD default: ./benchmark/dsa/run_b200_pipeline.sh
  DSA_NCU_CMD      invoked as: $DSA_NCU_CMD <impl> <out-dir>
  DSA_DECODED_GLOB glob locating the candidate decoded results after the
                   pipeline stage, default:
                   ${DSA_RESULTS_ROOT:-.dsa_b200_results}/*/trace/traces/2cta/iket/pid_*/iket.decoded_results.json
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/../../.." && pwd -P)"

impl=""; artifact_dir=""; windows="1-3,14-17"; gates=""
passed_with_skips=0
stage0_capture=""; reference_capture=""
skip_stage0=""; skip_pipeline=""; skip_ncu=""

while (($#)); do
  arg="$1"
  case "${arg}" in
    --impl|--artifact-dir|--windows|--skip|--gates|--stage0-capture|--reference-capture)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${arg} requires a value" >&2; exit 2; }
      ;;
  esac
  case "${arg}" in
    --impl) impl="$(printf '%s' "$2" | tr 'A-Z' 'a-z')"; shift 2 ;;
    --artifact-dir) artifact_dir="$2"; shift 2 ;;
    --windows) windows="$2"; shift 2 ;;
    --skip)
      case "$2" in
        stage0) skip_stage0=1 ;;
        pipeline) skip_pipeline=1 ;;
        ncu) skip_ncu=1 ;;
        *) echo "ERROR: --skip must be stage0|pipeline|ncu" >&2; exit 2 ;;
      esac
      shift 2 ;;
    --gates) gates="$2"; shift 2 ;;
    --stage0-capture) stage0_capture="$2"; shift 2 ;;
    --reference-capture) reference_capture="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: ${arg}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${impl}" =~ ^v[a-z0-9_]{1,31}$ ]] || {
  echo "ERROR: --impl must match v[a-z0-9_]{1,31}" >&2; exit 2; }

if [[ -z "${artifact_dir}" ]]; then
  base="/home/longcheng/artifact"
  n=1
  while [[ -e "${base}/${impl}_run${n}" \
        || -e "${base}/${impl}_run${n}.FAILED" \
        || -e "${base}/${impl}_run${n}.partial" ]]; do n=$((n+1)); done
  artifact_dir="${base}/${impl}_run${n}"
fi
partial="${artifact_dir}.partial"
[[ ! -e "${partial}" ]] || {
  echo "ERROR: stale partial exists: ${partial} (inspect/remove first)" >&2
  exit 2
}
[[ ! -e "${artifact_dir}" ]] || {
  echo "ERROR: artifact dir already exists: ${artifact_dir}" >&2
  exit 2
}
mkdir -p "${partial}"

fail_stop() {
  # $1 = stage, $2 = reason; preserves partial content for forensics.
  {
    echo "stage: $1"
    echo "reason: $2"
    echo "impl: ${impl}"
    echo "revision: $(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "partial: ${partial}"
    date -u +%Y-%m-%dT%H:%M:%SZ
  } > "${artifact_dir}.FAILED"
  echo "DSA_ALLINONE_STOP stage=$1 reason=$2" >&2
  exit 1
}

log() { echo "DSA_ALLINONE $*"; }

trap 'fail_stop internal "unexpected failure at line ${LINENO}"' ERR

# ---------------------------------------------------------------- preflight
candidate_rel="python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_${impl}.py"
[[ -f "${repo_root}/${candidate_rel}" ]] \
  || fail_stop preflight "implementation not registered: ${candidate_rel}"
git -C "${repo_root}" diff --quiet HEAD -- "${candidate_rel}" \
  || fail_stop preflight "implementation has uncommitted changes"
python3 -m py_compile "${repo_root}/${candidate_rel}" \
  || fail_stop preflight "py_compile failed"
env | grep '^DSA_V' | sort | tee "${partial}/bisect_flags.txt" || true
log "preflight OK impl=${impl} artifact=${artifact_dir}"

# ------------------------------------------------------------------ stage0
if [[ -z "${skip_stage0}" ]]; then
  if [[ -z "${stage0_capture}" && -n "${DSA_STAGE0_CMD:-}" ]]; then
    stage0_capture="${partial}/stage0_capture"
    log "stage0: compiling via DSA_STAGE0_CMD"
    ${DSA_STAGE0_CMD} "${impl}" "${stage0_capture}" \
      || fail_stop stage0 "compile capture failed"
  fi
  if [[ -n "${stage0_capture}" ]]; then
    [[ -n "${reference_capture}" ]] \
      || fail_stop stage0 "--reference-capture required with a capture"
    gate_args=(--capture-root "${stage0_capture}"
               --reference-root "${reference_capture}"
               --output-dir "${partial}/stage0_analysis")
    if [[ -n "${gates}" ]]; then
      # Auto-fill the per-run provenance pins from the checkout so the
      # template's UPDATE_PER_RUN placeholders never reach the gates.
      python3 - "${gates}" "${partial}/gates_effective.json" \
          "${repo_root}" "${candidate_rel}" <<'PYFILL'
import hashlib, json, subprocess, sys
gates, out, root, rel = sys.argv[1:5]
d = json.loads(open(gates).read())
d["TARGET_REVISION"] = subprocess.run(
    ["git", "-C", root, "rev-parse", "HEAD"],
    capture_output=True, text=True).stdout.strip()
h = hashlib.sha256()
h.update(open(f"{root}/{rel}", "rb").read())
d["TARGET_SOURCE_SHA256"] = h.hexdigest()
open(out, "w").write(json.dumps(d, indent=1))
PYFILL
      gate_args+=(--expectations "${partial}/gates_effective.json")
    fi
    if ! python3 "${script_dir}/stage0_analyzer.py" "${gate_args[@]}" \
        > "${partial}/stage0_analyzer.log" 2>&1; then
      [[ "${stage0_capture}" == "${partial}"* ]] \
        || cp -r "${stage0_capture}" "${partial}/" 2>/dev/null || true
      fail_stop stage0 "gate STOP -- see stage0_analysis/ + analyzer log"
    fi
    log "stage0 gates PASS"
  else
    log "stage0 SKIPPED (no DSA_STAGE0_CMD / --stage0-capture)"
    echo "stage0: SKIPPED" > "${partial}/stage0_gate_report.md"
  fi
else
  log "stage0 skipped by flag"
fi

# ---------------------------------------------------------------- pipeline
if [[ -z "${skip_pipeline}" ]]; then
  pipeline_cmd="${DSA_PIPELINE_CMD:-${repo_root}/benchmark/dsa/run_b200_pipeline.sh}"
  log "pipeline: ${pipeline_cmd} --impl ${impl}"
  if ! ${pipeline_cmd} --impl "${impl}" \
      2>&1 | tee "${partial}/pipeline.log"; then
    fail_stop pipeline "one-click pipeline failed (see pipeline.log)"
  fi
  grep -q "DSA_PIPELINE_PASSED" "${partial}/pipeline.log" \
    || fail_stop pipeline "DSA_PIPELINE_PASSED marker missing"
  # harvest the standard pipeline outputs referenced in the log
  for k in DSA_TABLES DSA_TABLES_JSON DSA_VALIDATION; do
    p="$(awk -v k="$k" '$1 == k || $1 == k":" {print $2}' "${partial}/pipeline.log" | tail -1)"
    [[ -n "$p" && -e "$p" ]] && cp -r "$p" "${partial}/" || true
  done
else
  log "pipeline skipped by flag"
fi

# --------------------------------------------------------------------- ncu
if [[ -z "${skip_ncu}" && -n "${DSA_NCU_CMD:-}" ]]; then
  log "ncu: DSA_NCU_CMD"
  ${DSA_NCU_CMD} "${impl}" "${partial}" \
    || fail_stop ncu "ncu companion failed"
else
  log "ncu SKIPPED"
fi

# ----------------------------------------------------------------- readout
decoded_glob="${DSA_DECODED_GLOB:-${DSA_RESULTS_ROOT:-${repo_root}/.dsa_b200_results}/*/trace/traces/2cta/iket/pid_*/iket.decoded_results.json}"
decoded="$(ls -t ${decoded_glob} 2>/dev/null | head -1 || true)"
if [[ -n "${decoded}" ]]; then
  cp "${decoded}" "${partial}/candidate_iket.decoded_results.json"
  perf_args=()
  for tag in baseline candidate; do
    p="$(ls -t ${DSA_RESULTS_ROOT:-${repo_root}/.dsa_b200_results}/*/*${tag}*performance*.json 2>/dev/null | head -1 || true)"
    [[ -n "$p" ]] && perf_args+=("--perf-${tag}" "$p")
  done
  python3 "${script_dir}/round_readout.py" \
    --decoded "${partial}/candidate_iket.decoded_results.json" \
    --out-dir "${partial}" --windows "${windows}" \
    ${perf_args[@]+"${perf_args[@]}"} \
    || fail_stop readout "round_readout.py failed"
  log "readout OK"
else
  log "readout SKIPPED (no decoded results matched: ${decoded_glob})"
  {
    echo "readout: SKIPPED"
    echo "glob: ${decoded_glob}"
    echo "hint: set DSA_RESULTS_ROOT (or DSA_DECODED_GLOB) to the harness"
    echo "results tree that holds iket.decoded_results.json"
  } > "${partial}/readout_SKIPPED.txt"
  passed_with_skips=1
fi

# ----------------------------------------------------------------- publish
if command -v sha256sum >/dev/null 2>&1; then
  hasher=(sha256sum)
else
  hasher=(shasum -a 256)
fi
( cd "${partial}" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | xargs -0 "${hasher[@]}" ) > "${partial}/MANIFEST.sha256" \
  || fail_stop publish "manifest generation failed"
mv "${partial}" "${artifact_dir}"
log "PUBLISHED ${artifact_dir}"
if [[ "${passed_with_skips}" == "1" ]]; then
  echo "DSA_ALLINONE_PASSED_WITH_SKIPS ${artifact_dir}"
else
  echo "DSA_ALLINONE_PASSED ${artifact_dir}"
fi
