#!/usr/bin/env bash
set -Eeuo pipefail

# Managed-B200 correctness gate, fair direct benchmark, and IKET trace capture.
# Successful local output contains only perf.json and trace.json. Failures keep
# session.log plus every compact remote diagnostic/result produced before the
# failing stage. The global service allocation and Docker worker are reused;
# only its per-request pipeline lock is held for the duration of this run.

usage() {
  cat <<'EOF'
Usage:
  ./benchmark/dsa/run_v_w3_2_allinone.sh [options]

Options:
  --impl TOKEN          Candidate token (default: final)
  --remote-repo PATH    ComputeLab checkout (default:
                        /home/scratch.longcheng_gpu/cudnn-frontend-thynics)
  --output-dir PATH     New local result directory
  -h, --help            Show this help

Success: OUTPUT_DIR/{perf.json,trace.json}
Failure: OUTPUT_DIR/session.log plus remote logs/status/partial JSON artifacts

The existing managed B200 allocation/Docker worker is reused when healthy.
This command never stops that service; it releases only pipeline.lock.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/../.." && pwd -P)"
frontend="${COMPUTELAB_B200_FRONTEND:-computelab-sc-01}"
implementation="final"
remote_repo="/home/scratch.longcheng_gpu/cudnn-frontend-thynics"
remote_manager="/home/scratch.longcheng_gpu/dsa-b200-harness-image/b200_manager.sh"
output_dir=""

while (($#)); do
  case "$1" in
    --impl)
      [[ $# -ge 2 ]] || { echo "ERROR: --impl needs a value" >&2; exit 2; }
      implementation="$2"
      shift 2
      ;;
    --remote-repo)
      [[ $# -ge 2 ]] || { echo "ERROR: --remote-repo needs a value" >&2; exit 2; }
      remote_repo="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --output-dir needs a value" >&2; exit 2; }
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${implementation}" =~ ^[a-z][a-zA-Z0-9_]{0,31}$ ]] || {
  echo "ERROR: --impl must match [a-z][a-zA-Z0-9_]{0,31}" >&2
  exit 2
}
[[ "${remote_repo}" == /home/scratch.longcheng_gpu/* ]] || {
  echo "ERROR: --remote-repo must be under /home/scratch.longcheng_gpu" >&2
  exit 2
}

for executable in bash git python3 scp ssh tar; do
  command -v "${executable}" >/dev/null 2>&1 || {
    echo "ERROR: missing executable: ${executable}" >&2
    exit 2
  }
done
expanded_hostname="$(
  ssh -G "${frontend}" 2>/dev/null |
    awk 'tolower($1) == "hostname" { print $2; exit }'
)"
[[ "${expanded_hostname}" == "computelab-sc-01" ]] || {
  echo "ERROR: ${frontend} expands to unsupported host ${expanded_hostname:-<empty>}" >&2
  exit 2
}

run_id="$(date -u +%Y%m%dT%H%M%SZ)_${implementation}_$$"
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/outputs/${run_id}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
[[ ! -e "${output_dir}" ]] || {
  echo "ERROR: refusing to overwrite local output: ${output_dir}" >&2
  exit 2
}
mkdir -p "${output_dir}"

session_log="${output_dir}/session.log"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/dsa-v-w3-2-allinone.XXXXXXXX")"
remote_payload_local="${temp_dir}/remote.sh"
remote_run_root="/home/scratch.longcheng_gpu/.dsa-allinone/${run_id}"
remote_payload="${remote_run_root}/remote.sh"
remote_result="${remote_run_root}/result"
remote_worktree="${remote_run_root}/worktree"
completed=0

exec > >(tee -a "${session_log}") 2>&1

cleanup() {
  local rc=$?
  set +e
  rm -rf "${temp_dir}"
  if ((completed)); then
    rm -f "${session_log}"
  elif ((rc != 0)); then
    echo "FAILED rc=${rc}; diagnostics: ${output_dir}" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cat >"${remote_payload_local}" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

implementation="$1"
repo="$2"
result="$3"
worktree="$4"
run_root="$(dirname "${result}")"
package_rel="python/cudnn/deepseek_sparse_attention/sparse_attention_backward"
interface_template="/home/scratch.longcheng_gpu/dsa-b200-harness-image/interface_sm100.py"
release_python="${DSA_VALIDATE_PYTHON:-/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python}"
trace_python="${DSA_TRACE_PYTHON:-/home/scratch.longcheng_gpu/dsa_iket_h128_venv_2606/bin/python}"
stage="bootstrap"
error_line=""
error_command=""

mkdir -p "${result}"
exec > >(tee -a "${result}/remote.log") \
  2> >(tee -a "${result}/remote.stderr.log" >&2)

record_error() {
  error_line="$1"
  error_command="$2"
}
trap 'record_error "${LINENO}" "${BASH_COMMAND}"' ERR

finish() {
  local rc=$?
  set +e
  if [[ -n "${worktree}" && -d "${worktree}" ]]; then
    git -C "${repo}" worktree remove --force "${worktree}" >/dev/null 2>&1 || true
  fi
  STATUS_RC="${rc}" \
  STATUS_STAGE="${stage}" \
  STATUS_LINE="${error_line}" \
  STATUS_COMMAND="${error_command}" \
  STATUS_IMPL="${implementation}" \
  STATUS_REPO="${repo}" \
  python3 - "${result}/status.json" <<'PY_STATUS'
import json
import os
import sys

rc = int(os.environ["STATUS_RC"])
payload = {
    "status": "pass" if rc == 0 else "fail",
    "exit_code": rc,
    "stage": os.environ["STATUS_STAGE"],
    "error_line": os.environ["STATUS_LINE"] or None,
    "error_command": os.environ["STATUS_COMMAND"] or None,
    "implementation": os.environ["STATUS_IMPL"],
    "repo": os.environ["STATUS_REPO"],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY_STATUS
  trap - EXIT
  exit "${rc}"
}
trap finish EXIT

[[ "${repo}" == /home/scratch.longcheng_gpu/* ]]
[[ "${result}" == /home/scratch.longcheng_gpu/* ]]
[[ "${worktree}" == /home/scratch.longcheng_gpu/* ]]
[[ -d "${repo}/.git" || -f "${repo}/.git" ]]

stage="create_worktree"
git -C "${repo}" worktree add --detach "${worktree}" HEAD

package="${worktree}/${package_rel}"
release_source="${package}/dsa_bwd_sm100_2cta_${implementation}.py"
trace_source="${package}/dsa_bwd_sm100_2cta_${implementation}_trace.py"
active_source="${package}/dsa_bwd_sm100_2cta_v2.py"
stage="locate_runtime"
[[ -s "${release_source}" ]]
[[ -s "${trace_source}" ]]
[[ -s "${interface_template}" ]]
[[ -x "${release_python}" ]]
[[ -x "${trace_python}" ]]

compiled_module="$(find "${repo}/python/cudnn" -maxdepth 1 -type f \
  -name '_compiled_module*.so' -print -quit)"
if [[ -z "${compiled_module}" ]]; then
  compiled_module="$(find /home/scratch.longcheng_gpu/cudnn-frontend/python/cudnn \
    -maxdepth 1 -type f -name '_compiled_module*.so' -print -quit)"
fi
[[ -n "${compiled_module}" && -s "${compiled_module}" ]]

stage="prepare_release"
cp "${interface_template}" "${package}/_interface_sm100.py"
cp "${release_source}" "${active_source}"
cp "${compiled_module}" "${worktree}/python/cudnn/"

# Persistent serialized caches make repeat runs fast while staying under the
# mandated ComputeLab scratch root.
cache_root="/home/scratch.longcheng_gpu/.dsa-allinone-cache"
mkdir -p "${cache_root}/home" "${cache_root}/xdg" "${cache_root}/cuda"
export HOME="${cache_root}/home"
export XDG_CACHE_HOME="${cache_root}/xdg"
export CUDA_CACHE_PATH="${cache_root}/cuda"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${worktree}/python:${worktree}/test/python:${worktree}"
export DSA_DEV_CANDIDATE_VARIANT="v2native"
export DSA_DEV_IMPLEMENTATION="${implementation}"
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

stage="release_correctness_and_perf"
"${release_python}" - "${worktree}" "${result}/correctness.json" "${result}/perf.json" <<'PY_RELEASE'
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

repo = Path(sys.argv[1]).resolve()
correctness_output = Path(sys.argv[2]).resolve()
perf_output = Path(sys.argv[3]).resolve()

os.environ["DSA_DEV_CANDIDATE_VARIANT"] = "v2native"
os.environ.pop("DSA_DEV_IKET", None)
os.environ.pop("DKG_IKET_INSTRUMENTATION_METHOD", None)
sys.path[:0] = [str(repo / "python"), str(repo / "test/python"), str(repo)]

from cudnn import DSA
from fe_api.dsa.dsa_reference import ref_sparse_attention_forward
from benchmark.dsa import benchmark_dsa_sparse_attention_backward as bench

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
gpu = torch.cuda.get_device_name()
capability = list(torch.cuda.get_device_capability())
if "B200" not in gpu or capability != [10, 0]:
    raise RuntimeError(f"expected NVIDIA B200 sm_100, got {gpu} capability={capability}")


def phase(name: str):
    class Phase:
        def __enter__(self):
            self.started = time.monotonic()
            print(f"DSA_ALLINONE_PHASE_BEGIN {name}", flush=True)

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                elapsed = time.monotonic() - self.started
                print(f"DSA_ALLINONE_PHASE_END {name} duration_s={elapsed:.6f}", flush=True)
            return False

    return Phase()


# Correctness: H128/D512/K256, four metadata patterns. The first dense call
# also performs the only K256 compilation; there is no redundant compile run.
c_s_q, c_s_kv, c_heads, c_dim, c_topk = 64, 512, 128, 512, 256
c_seed = 20260724
patterns = ("dense", "lengths", "holes", "all_empty")


def metadata(pattern: str):
    indices = torch.rand(c_s_q, c_s_kv, device="cuda").argsort(dim=-1)[:, :c_topk].to(torch.int32)
    lengths = torch.full((c_s_q,), c_topk, device="cuda", dtype=torch.int32)
    if pattern == "lengths":
        candidates = torch.tensor(
            [0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256],
            device="cuda",
            dtype=torch.int32,
        )
        lengths = candidates[torch.arange(c_s_q, device="cuda") % candidates.numel()].contiguous()
    elif pattern == "holes":
        indices[:, 64:128] = -1
        indices[:, 3::17] = -1
    elif pattern == "all_empty":
        indices.fill_(-1)
    elif pattern != "dense":
        raise ValueError(pattern)
    return indices.contiguous(), lengths


def error_metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    denominator = actual.norm() * expected.norm()
    cosine = float((actual.flatten() @ expected.flatten()) / denominator) if float(denominator) else float(actual.norm() == expected.norm())
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()), "cosine": cosine}


correctness_records = []
for pattern in patterns:
    torch.manual_seed(c_seed)
    q = torch.randn(c_s_q, c_heads, c_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(c_s_kv, c_dim, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    sink = torch.linspace(-2.0, 2.0, c_heads, device="cuda")
    indices, lengths = metadata(pattern)
    scale = 1.0 / math.sqrt(c_dim)
    out, lse = ref_sparse_attention_forward(q, kv, sink, indices, topk_length=lengths, softmax_scale=scale)

    with phase(f"correctness:{pattern}"):
        result = DSA.sparse_attention_backward_wrapper(
            q, kv, out, dout, lse, sink, indices,
            softmax_scale=scale, topk_length=lengths,
        )
        torch.cuda.synchronize()

    q_ref = q.float().detach().requires_grad_(True)
    kv_ref = kv.float().detach().requires_grad_(True)
    sink_ref = sink.float().detach().requires_grad_(True)
    out_ref, _ = ref_sparse_attention_forward(
        q_ref, kv_ref, sink_ref, indices,
        topk_length=lengths, softmax_scale=scale,
    )
    out_ref.backward(dout.float())
    torch.testing.assert_close(result["dq"].float(), q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(result["dkv"].float(), kv_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(result["d_sink"].float(), sink_ref.grad, atol=5e-2, rtol=5e-2)

    correctness_records.append({
        "pattern": pattern,
        "dq": error_metrics(result["dq"], q_ref.grad),
        "dkv": error_metrics(result["dkv"], kv_ref.grad),
        "d_sink": error_metrics(result["d_sink"], sink_ref.grad),
    })
    print(f"PASS correctness:{pattern}", flush=True)
    del q, kv, dout, sink, indices, lengths, out, lse, result, q_ref, kv_ref, sink_ref, out_ref
    torch.cuda.empty_cache()

correctness_payload = {
    "status": "pass",
    "trace_enabled": False,
    "gpu": gpu,
    "compute_capability": capability,
    "seqlen_q": c_s_q,
    "seqlen_kv": c_s_kv,
    "nheads": c_heads,
    "head_dim": c_dim,
    "topk": c_topk,
    "patterns": list(patterns),
    "records": correctness_records,
}
correctness_output.write_text(json.dumps(correctness_payload, indent=2, sort_keys=True) + "\n")

# Release performance: today's corrected fair boundary.  Baseline and
# candidate are both compiled and called directly, share all tensors and
# workspaces, and keep reset work outside the CUDA event interval.  This avoids
# the old wrapper-vs-direct bias while retaining the validation anchor shape.
import cutlass
import cutlass.cute as cute
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100 import (
    FlashAttentionDSABackwardSm100,
)
from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_2cta_v2 import (
    FlashAttentionDSABackwardSm100TwoCTAV2,
)
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

p_s_q = p_s_kv = 4096
p_heads, p_dim, p_topk = 128, 512, 2048
warmup_pairs, paired_samples = 8, 24
torch.manual_seed(20260810)
q, kv, sink, dout, indices, lengths = bench.make_inputs(
    p_s_q,
    p_topk,
    p_s_kv,
    p_heads,
    p_dim,
    p_dim,
    torch.bfloat16,
    use_attn_sink=True,
    use_topk_length=True,
)
scale = 1.0 / math.sqrt(p_dim)
out, lse = bench.reference_forward(q, kv, sink, indices, scale, p_dim)


def workspace_shapes(impl_cls):
    accumulator = cutlass.Float32
    return (
        tuple(
            int(value)
            for value in impl_cls._get_workspace_size_LSE_OdO(
                p_s_q, p_dim, p_heads, 1, accumulator
            )
        ),
        tuple(
            int(value)
            for value in impl_cls._get_workspace_size_dKV(
                p_s_kv, p_dim, 1, accumulator
            )
        ),
    )


baseline_shapes = workspace_shapes(FlashAttentionDSABackwardSm100)
candidate_shapes = workspace_shapes(FlashAttentionDSABackwardSm100TwoCTAV2)
if baseline_shapes != candidate_shapes:
    raise RuntimeError(
        f"workspace mismatch: baseline={baseline_shapes} candidate={candidate_shapes}"
    )

buffers = {
    "dq": torch.empty_like(q),
    "dkv": torch.zeros_like(kv),
    "d_sink": torch.zeros_like(sink),
    "workspace_lse_odo": torch.zeros(
        *baseline_shapes[0], dtype=torch.uint8, device="cuda"
    ),
    "workspace_dkv": torch.zeros(
        *baseline_shapes[1], dtype=torch.uint8, device="cuda"
    ),
}
problem_shape = (p_s_q, p_s_kv, p_dim, (p_heads, 1))
stream = resolve_stream(None)


def build_direct_runner(impl_cls, has_trace_args):
    kernel = impl_cls(
        head_dim=p_dim,
        head_dim_v=p_dim,
        block_tile=64,
        max_topk=p_topk,
    )
    prototypes = [
        to_cute_tensor(q, divisibility=p_dim),
        to_cute_tensor(kv, divisibility=p_dim),
        to_cute_tensor(out, divisibility=p_dim),
        to_cute_tensor(dout, divisibility=p_dim),
        to_cute_tensor(lse, assumed_align=4),
        to_cute_tensor(sink),
        to_cute_tensor(indices),
        to_cute_tensor(lengths),
        to_cute_tensor(buffers["dq"], divisibility=p_dim),
        to_cute_tensor(buffers["dkv"], divisibility=p_dim),
        to_cute_tensor(buffers["d_sink"]),
        to_cute_tensor(buffers["workspace_lse_odo"]),
        to_cute_tensor(buffers["workspace_dkv"]),
    ]
    runtime = [
        q,
        kv,
        out,
        dout,
        lse,
        sink,
        indices,
        lengths,
        buffers["dq"],
        buffers["dkv"],
        buffers["d_sink"],
        buffers["workspace_lse_odo"],
        buffers["workspace_dkv"],
    ]
    if has_trace_args:
        prototypes.extend([None, 0, 0])
        runtime.extend([None, 0, 0])
    prototypes.extend([scale, stream])
    runtime.extend([scale, stream])
    compiled = cute.compile(
        kernel,
        problem_shape,
        *prototypes,
        options=compile_options(),
    )
    torch.cuda.synchronize()

    def launch():
        compiled(problem_shape, *runtime)

    return launch


def reset_accumulators():
    buffers["dkv"].zero_()
    buffers["workspace_dkv"].zero_()
    buffers["d_sink"].zero_()


with phase("perf:compile:baseline"):
    baseline_launch = build_direct_runner(FlashAttentionDSABackwardSm100, False)
with phase("perf:compile:candidate"):
    candidate_launch = build_direct_runner(
        FlashAttentionDSABackwardSm100TwoCTAV2, True
    )

runners = {"baseline": baseline_launch, "candidate": candidate_launch}


def pair_order(index):
    return (
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
    )[index % 4]


with phase("perf:warmup"):
    for index in range(warmup_pairs):
        for name in pair_order(index):
            reset_accumulators()
            runners[name]()
    torch.cuda.synchronize()

# Cross-implementation correctness at the exact timed shape.
reset_accumulators()
baseline_launch()
torch.cuda.synchronize()
baseline_outputs = {
    name: buffers[name].clone() for name in ("dq", "dkv", "d_sink")
}
reset_accumulators()
candidate_launch()
torch.cuda.synchronize()
crosscheck = {
    f"max_abs_diff_{name}": float(
        (baseline_outputs[name] - buffers[name]).abs().max()
    )
    for name in baseline_outputs
}
crosscheck["all_outputs_finite"] = all(
    bool(torch.isfinite(tensor).all())
    for tensor in (*baseline_outputs.values(), buffers["dq"], buffers["dkv"], buffers["d_sink"])
)
crosscheck["gate"] = (
    "PASS"
    if crosscheck["all_outputs_finite"]
    and crosscheck["max_abs_diff_dq"] == 0.0
    and crosscheck["max_abs_diff_dkv"] <= 0.002
    and crosscheck["max_abs_diff_d_sink"] <= 0.05
    else "FAIL"
)
if crosscheck["gate"] != "PASS":
    raise RuntimeError(f"fair perf correctness gate failed: {crosscheck}")


def time_program_only(launch):
    # The reset is ordered before the start event on the same stream.
    reset_accumulators()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


samples = {"baseline": [], "candidate": []}
with phase("perf:timed:fair_direct_program_only"):
    for index in range(paired_samples):
        for name in pair_order(index):
            samples[name].append(time_program_only(runners[name]))

baseline_latency_ms = statistics.median(samples["baseline"])
candidate_latency_ms = statistics.median(samples["candidate"])
paired_ratios = [
    candidate_ms / baseline_ms
    for baseline_ms, candidate_ms in zip(samples["baseline"], samples["candidate"])
]
paired_ratio = statistics.median(paired_ratios)
flops = bench.flops_bwd(p_s_q, p_topk, p_heads, p_dim, p_dim)
perf = {
    "status": "pass",
    "benchmark": "fair_direct_program_only_20260810",
    "primary_metric": "program_only",
    "trace_enabled": False,
    "fairness_contract": {
        "baseline_path": "direct cute.compile(FlashAttentionDSABackwardSm100)",
        "candidate_path": "direct cute.compile(FlashAttentionDSABackwardSm100TwoCTAV2)",
        "same_inputs": True,
        "same_output_and_workspace_addresses": True,
        "same_reset_set": ["dkv", "workspace_dkv", "d_sink"],
        "resets_excluded_from_cuda_event": True,
        "compile_and_allocation_outside_timing": True,
        "order": "ABBA-balanced paired samples",
    },
    "gpu": gpu,
    "compute_capability": capability,
    "seqlen_q": p_s_q,
    "seqlen_kv": p_s_kv,
    "topk": p_topk,
    "nheads": p_heads,
    "head_dim_qk": p_dim,
    "head_dim_v": p_dim,
    "dtype": "bfloat16",
    "warmup_pairs": warmup_pairs,
    "paired_samples": paired_samples,
    "baseline_latency_ms": round(baseline_latency_ms, 6),
    "candidate_latency_ms": round(candidate_latency_ms, 6),
    "candidate_over_baseline": round(paired_ratio, 6),
    "candidate_speedup_percent": round((1.0 - paired_ratio) * 100.0, 6),
    "latency_ms": round(candidate_latency_ms, 6),
    "tflops": round(flops / (candidate_latency_ms * 1e-3) / 1e12, 6),
    "correctness_crosscheck": crosscheck,
    "raw_ms": samples,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
}
perf_output.write_text(json.dumps(perf, indent=2, sort_keys=True) + "\n")
print(json.dumps(perf, indent=2, sort_keys=True), flush=True)
PY_RELEASE

stage="prepare_trace"
cp "${trace_source}" "${active_source}"
[[ -x "${trace_python}" ]]
standalone_site="$(${trace_python} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[[ -s "${standalone_site}/iket/cli/main.py" ]]
[[ -s "${standalone_site}/iket/profiler/libiket_injection.so" ]]

trace_workload="${run_root}/trace_workload.py"
cat >"${trace_workload}" <<'PY_TRACE'
from types import SimpleNamespace
import os
import torch

os.environ["DSA_DEV_CANDIDATE_VARIANT"] = "v2native"
from benchmark.dsa.benchmark_dsa_sparse_attention_backward import setup_case

torch.manual_seed(0)
workload = SimpleNamespace(
    dtype="bfloat16",
    head_dim=512,
    head_dim_v=512,
    nheads=128,
    no_attn_sink=False,
    no_topk_length=False,
)
run = setup_case(seqlen_q=1, seqlen_kv=4096, topk=2048, args=workload)
torch.cuda.synchronize()
run()
torch.cuda.synchronize()
print("IKET workload complete: S_q=1 S_kv=4096 H=128 D=512 topk=2048", flush=True)
PY_TRACE

export DKG_IKET_INSTRUMENTATION_METHOD="NativeDump"
export DSA_DEV_IKET=1
export IKET_STANDALONE_SITE_PACKAGES="${standalone_site}"
capture="${result}/iket_capture"
stage="iket_capture"
"${trace_python}" -c \
  'from iket.cli.main import entrypoint; raise SystemExit(entrypoint())' \
  --output-dir "${capture}" \
  --log-level info \
  profile \
  --postprocess perfetto \
  --keep \
  -- \
  "${trace_python}" "${trace_workload}"

stage="verify_trace"
shopt -s nullglob
decoded=("${capture}"/iket/pid_*/iket.decoded_results.json)
(( ${#decoded[@]} == 1 ))
[[ -s "${decoded[0]}" ]]
cp "${decoded[0]}" "${result}/trace.json"

stage="complete"
echo "DSA_ALLINONE_PASS perf=${result}/perf.json trace=${result}/trace.json"
REMOTE_SCRIPT

echo "Staging all-in-one payload: ${frontend}:${remote_payload}"
ssh -o BatchMode=yes "${frontend}" "mkdir -p '${remote_run_root}'"
scp -q "${remote_payload_local}" "${frontend}:${remote_payload}"
ssh -o BatchMode=yes "${frontend}" \
  "chmod 700 '${remote_payload}' && test -x '${remote_payload}'"

manager_label="dsa-allinone"
remote_command="$(printf "%q " \
  "${remote_manager}" \
  with-lock \
  --label "${manager_label}" \
  --sync-repo "${remote_repo}" \
  -- \
  "${remote_payload}" \
  "${implementation}" \
  "${remote_repo}" \
  "${remote_result}" \
  "${remote_worktree}")"

echo "Reusing the managed B200 worker and acquiring pipeline.lock..."
set +e
ssh -o BatchMode=yes "${frontend}" "${remote_command}"
run_rc=$?
set -e

if ((run_rc != 0)); then
  echo "Remote run failed (rc=${run_rc}); downloading diagnostics..." >&2
  scp -q -r "${frontend}:${remote_result}/." "${output_dir}/" || \
    echo "WARNING: remote diagnostics were unavailable; session.log is complete" >&2
  exit "${run_rc}"
fi

managed_status="$(
  ssh -o BatchMode=yes "${frontend}" \
    "$(printf '%q' "${remote_manager}") status"
)"
printf '%s\n' "${managed_status}"
if ! grep -Fq 'heartbeat=fresh state=ready' <<<"${managed_status}"; then
  echo "ERROR: managed B200 worker was not retained after pipeline.lock release" >&2
  exit 70
fi

echo "Downloading compact success artifacts..."
ssh -o BatchMode=yes "${frontend}" \
  "tar -C '${remote_result}' -cf - perf.json trace.json" |
  tar -C "${output_dir}" -xf -

python3 - "${output_dir}/perf.json" "${output_dir}/trace.json" <<'PY_VERIFY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    perf = json.load(handle)
if perf.get("status") != "pass" or perf.get("trace_enabled") is not False:
    raise SystemExit(f"invalid perf JSON: {perf}")
if not os.path.getsize(sys.argv[2]):
    raise SystemExit("trace JSON is empty")
with open(sys.argv[2], encoding="utf-8") as handle:
    json.load(handle)
print(
    f"PASS latency_ms={perf['latency_ms']} tflops={perf['tflops']} "
    f"perf={sys.argv[1]} trace={sys.argv[2]}"
)
PY_VERIFY

completed=1
