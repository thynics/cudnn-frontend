#!/usr/bin/env python3
"""Standalone Rubin correctness gate for a DSA backward candidate source.

The candidate is imported directly from ``--candidate-source`` under the real
DSA package name.  No repository source is copied or overwritten.  The gate
compiles the frozen one-CTA baseline and the candidate for both optional-
length specializations, then checks their gradients against the repository's
PyTorch autograd reference.

Run the GPU entry under an external hard timeout; ``--describe`` is CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path


# These must be selected before importing torch/CuTe DSL.
os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# Freeze the exact one-CTA baseline configuration used by test_harness.py.
os.environ["DSA_BL_QDO_STAGE"] = "1"
os.environ["DSA_BL_K_STAGE"] = "1"
os.environ["DSA_BL_HALFK"] = "0"
os.environ["DSA_BL_KSTAGE2"] = "0"
os.environ["DSA_BL_OVPAD"] = "0"
os.environ.pop("DSA_DEV_IKET", None)
os.environ.pop("DKG_IKET_INSTRUMENTATION_METHOD", None)


EXPECTED_BASELINE_SHA256 = (
    "a86b353a2349962bd4404818a906ea2d4df4ca5cb88b22d15ea234fc4e8ff3d7"
)
PACKAGE = "cudnn.deepseek_sparse_attention.sparse_attention_backward"
CLASS_NAME = "FlashAttentionDSABackwardSm100TwoCTAV2"

SEED = 20260816
SEQLEN_Q = 6
SEQLEN_KV = 4096
NHEADS = 128
HEAD_DIM = 512
MAX_TOPK = 2048
ATOL = 5.0e-2
RTOL = 5.0e-2

CASE_SPECS = {
    "full_valid_none": {
        "topk_length": None,
        "description": (
            "mTopkLength=None with all 2048 sparse indices valid in a 4096-row KV"
        ),
    },
    "full_length_dynamic": {
        "topk_length": [2048, 2048, 2048, 2048, 2048, 2048],
        "description": (
            "the benchmark's full-length runtime mTopkLength specialization"
        ),
    },
    "none_holes": {
        "topk_length": None,
        "description": (
            "mTopkLength=None with one full 64-slot hole and scattered -1 holes"
        ),
    },
    "ragged_lengths": {
        "topk_length": [0, 1, 63, 65, 127, 2047],
        "description": (
            "one distinct runtime length per token; valid indices remain after each "
            "length so the length gate, rather than -1 padding, must exclude them"
        ),
    },
    "tile_boundary_lengths": {
        "topk_length": [1, 63, 64, 128, 2047, 2048],
        "description": (
            "positive runtime lengths spanning the 64/128 tile boundaries and "
            "the full 2048 extent"
        ),
    },
    "all_zero": {
        "topk_length": [0, 0, 0, 0, 0, 0],
        "description": "zero-length liveness and zero-gradient contract",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--class-name", default=CLASS_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case",
        choices=("all", *CASE_SPECS),
        default="all",
        help="run one isolated case or the full compile-reusing suite",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print the static test contract without importing torch or using a GPU",
    )
    parser.add_argument(
        "--diagnose-dkv-columns",
        action="store_true",
        help=(
            "for a single full_valid_none case, record per-column dKV errors and "
            "test common column permutations"
        ),
    )
    args = parser.parse_args()
    if not args.describe:
        missing = [
            name
            for name in ("repo", "candidate_source", "output")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("required unless --describe: " + ", ".join(missing))
        if args.diagnose_dkv_columns and args.case != "full_valid_none":
            parser.error(
                "--diagnose-dkv-columns requires --case full_valid_none so the "
                "reported tensors come from exactly one launch of each implementation"
            )
    return args


def describe_payload() -> dict:
    return {
        "target": "Rubin SM107a",
        "shape": {
            "seqlen_q": SEQLEN_Q,
            "seqlen_kv": SEQLEN_KV,
            "nheads": NHEADS,
            "head_dim": HEAD_DIM,
            "max_topk": MAX_TOPK,
            "dtype": "bfloat16",
        },
        "cases": CASE_SPECS,
        "comparisons": [
            "exact_one_cta_baseline_vs_torch",
            "candidate_vs_torch",
            "candidate_vs_exact_one_cta_baseline",
        ],
        "baseline_metadata": (
            "positive runtime lengths use the frozen baseline's real length-tensor "
            "specialization; only cases containing a zero-length token are "
            "canonicalized to the equivalent None + -1-tail representation to "
            "avoid its unsafe tile_index=-1 path"
        ),
        "outputs": ["dq", "dkv", "d_sink"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "zero_length_contract": (
            "allowed by the current SM100 implementation and exercised by the "
            "repository all_empty correctness gate; the public wrapper does not "
            "reject zero"
        ),
        "external_timeout_required": True,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(candidate_source: Path, class_name: str):
    importlib.import_module(PACKAGE)
    module_name = f"{PACKAGE}._rubin_correctness_{sha256(candidate_source)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, candidate_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {candidate_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def make_metadata(torch, case_name: str):
    indices = torch.stack(
        [
            torch.randperm(SEQLEN_KV, device="cuda")[:MAX_TOPK]
            for _ in range(SEQLEN_Q)
        ]
    ).to(torch.int32)
    lengths_spec = CASE_SPECS[case_name]["topk_length"]
    lengths = None
    if lengths_spec is not None:
        lengths = torch.tensor(lengths_spec, device="cuda", dtype=torch.int32)
    if case_name == "none_holes":
        # Exercise a completely absent tile plus holes at non-tile positions.
        indices[:, 64:128] = -1
        indices[:, 3::17] = -1
        indices[::2, 0] = -1
        indices[1::2, 2047] = -1
    return indices.contiguous(), lengths


def build_case(torch, ref_forward, case_name: str) -> dict:
    torch.manual_seed(SEED)
    q = torch.randn(
        SEQLEN_Q, NHEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(SEQLEN_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    sink = torch.linspace(-2.0, 2.0, NHEADS, device="cuda")
    indices, lengths = make_metadata(torch, case_name)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    out, lse = ref_forward(
        q,
        kv,
        sink,
        indices,
        topk_length=lengths,
        softmax_scale=scale,
    )

    q_ref = q.float().detach().requires_grad_(True)
    kv_ref = kv.float().detach().requires_grad_(True)
    sink_ref = sink.float().detach().requires_grad_(True)
    out_ref, _ = ref_forward(
        q_ref,
        kv_ref,
        sink_ref,
        indices,
        topk_length=lengths,
        softmax_scale=scale,
    )
    out_ref.backward(dout.float())
    reference = {
        "dq": q_ref.grad.detach(),
        "dkv": kv_ref.grad.detach(),
        "d_sink": sink_ref.grad.detach(),
    }
    case = {
        "name": case_name,
        "q": q,
        "kv": kv,
        "out": out.detach(),
        "dout": dout,
        "lse": lse.detach(),
        "sink": sink,
        "indices": indices,
        "lengths": lengths,
        "scale": scale,
        "reference": reference,
    }
    baseline_indices = indices
    baseline_lengths = lengths
    baseline_representation = "none" if lengths is None else "direct_lengths"
    if lengths is not None and bool((lengths == 0).any()):
        positions = torch.arange(MAX_TOPK, device="cuda").unsqueeze(0)
        baseline_indices = indices.masked_fill(positions >= lengths[:, None], -1)
        baseline_lengths = None
        baseline_representation = "canonicalized_none_negative_one_tail"
    case["baseline_case"] = {
        **{key: value for key, value in case.items() if key != "reference"},
        "indices": baseline_indices.contiguous(),
        "lengths": baseline_lengths,
    }
    case["baseline_representation"] = baseline_representation
    return case


def verify_native_kernel(compiled, implementation: str) -> list[str]:
    symbols = list(getattr(compiled, "kernel_info", {}))
    prefix = (
        "kernel_cutlass_bwd_"
        if implementation == "baseline"
        else "kernel_cutlass_kernel_"
    )
    main_symbols = [symbol for symbol in symbols if symbol.startswith(prefix)]
    if len(main_symbols) != 1:
        raise RuntimeError(
            f"{implementation}: expected one native main kernel, got "
            f"{main_symbols} from {symbols}"
        )
    return main_symbols


class DirectRunner:
    def __init__(
        self,
        *,
        torch,
        cutlass,
        cute,
        to_cute_tensor,
        resolve_stream,
        implementation_class,
        implementation_name: str,
        has_trace_args: bool,
        sample: dict,
    ):
        self.torch = torch
        self.name = implementation_name
        self.has_trace_args = has_trace_args
        self.problem_shape = (
            SEQLEN_Q,
            SEQLEN_KV,
            HEAD_DIM,
            (NHEADS, 1),
        )
        self.stream = resolve_stream(None)
        self.buffers = {
            "dq": torch.empty_like(sample["q"]),
            "dkv": torch.zeros_like(sample["kv"]),
            "d_sink": torch.zeros_like(sample["sink"]),
            "workspace_lse_odo": torch.zeros(
                *implementation_class._get_workspace_size_LSE_OdO(
                    SEQLEN_Q, HEAD_DIM, NHEADS, 1, cutlass.Float32
                ),
                dtype=torch.uint8,
                device="cuda",
            ),
            "workspace_dkv": torch.zeros(
                *implementation_class._get_workspace_size_dKV(
                    SEQLEN_KV, HEAD_DIM, 1, cutlass.Float32
                ),
                dtype=torch.uint8,
                device="cuda",
            ),
        }
        kernel = implementation_class(
            head_dim=HEAD_DIM,
            head_dim_v=HEAD_DIM,
            block_tile=64,
            max_topk=MAX_TOPK,
        )
        if implementation_name == "baseline":
            kernel._setup_attributes()
            observed = (
                int(kernel.load_mma_QdO_stage),
                int(kernel.load_mma_K_stage),
            )
            if observed != (1, 1):
                raise RuntimeError(
                    f"exact baseline stage isolation failed: observed={observed}"
                )

        prototypes = [
            to_cute_tensor(sample["q"], divisibility=HEAD_DIM),
            to_cute_tensor(sample["kv"], divisibility=HEAD_DIM),
            to_cute_tensor(sample["out"], divisibility=HEAD_DIM),
            to_cute_tensor(sample["dout"], divisibility=HEAD_DIM),
            to_cute_tensor(sample["lse"], assumed_align=4),
            to_cute_tensor(sample["sink"]),
            to_cute_tensor(sample["indices"]),
            (
                to_cute_tensor(sample["lengths"])
                if sample["lengths"] is not None
                else None
            ),
            to_cute_tensor(self.buffers["dq"], divisibility=HEAD_DIM),
            to_cute_tensor(self.buffers["dkv"], divisibility=HEAD_DIM),
            to_cute_tensor(self.buffers["d_sink"]),
            to_cute_tensor(self.buffers["workspace_lse_odo"]),
            to_cute_tensor(self.buffers["workspace_dkv"]),
        ]
        if has_trace_args:
            prototypes.extend([None, 0, 0])
        prototypes.extend([sample["scale"], self.stream])
        self.compiled = cute.compile(
            kernel,
            self.problem_shape,
            *prototypes,
            options="--enable-tvm-ffi --gpu-arch sm_107a",
        )
        self.main_symbols = verify_native_kernel(self.compiled, implementation_name)
        torch.cuda.synchronize()

    def run(self, case: dict) -> dict:
        self.buffers["dq"].fill_(float("nan"))
        self.buffers["dkv"].zero_()
        self.buffers["d_sink"].zero_()
        self.buffers["workspace_lse_odo"].zero_()
        self.buffers["workspace_dkv"].zero_()
        runtime = [
            case["q"],
            case["kv"],
            case["out"],
            case["dout"],
            case["lse"],
            case["sink"],
            case["indices"],
            case["lengths"],
            self.buffers["dq"],
            self.buffers["dkv"],
            self.buffers["d_sink"],
            self.buffers["workspace_lse_odo"],
            self.buffers["workspace_dkv"],
        ]
        if self.has_trace_args:
            runtime.extend([None, 0, 0])
        runtime.extend([case["scale"], self.stream])
        self.compiled(self.problem_shape, *runtime)
        self.torch.cuda.synchronize()
        return {
            name: self.buffers[name].clone()
            for name in ("dq", "dkv", "d_sink")
        }


def tensor_metrics(torch, actual, expected) -> dict:
    actual = actual.float()
    expected = expected.float()
    finite_actual = torch.isfinite(actual)
    finite_expected = torch.isfinite(expected)
    finite_both = finite_actual & finite_expected
    metrics = {
        "actual_nan": int(torch.isnan(actual).sum()),
        "actual_inf": int(torch.isinf(actual).sum()),
        "expected_nan": int(torch.isnan(expected).sum()),
        "expected_inf": int(torch.isinf(expected).sum()),
    }
    if not bool(finite_both.any()):
        metrics.update(
            max_abs=float("inf"),
            max_rel=float("inf"),
            rel_p50=float("inf"),
            rel_p95=float("inf"),
            rel_p99=float("inf"),
            rmse=float("inf"),
            max_abs_index=None,
        )
        return metrics
    difference = (actual - expected).abs()
    finite_diff = difference[finite_both]
    relative = finite_diff / expected[finite_both].abs().clamp_min(1.0e-8)
    flat_difference = torch.where(
        finite_both,
        difference,
        torch.full_like(difference, float("inf")),
    ).flatten()
    max_flat_index = int(flat_difference.argmax())
    max_index = tuple(
        int(value)
        for value in torch.unravel_index(
            torch.tensor(max_flat_index, device=actual.device), actual.shape
        )
    )
    quantiles = torch.quantile(
        relative,
        torch.tensor([0.50, 0.95, 0.99], device=relative.device),
    )
    metrics.update(
        max_abs=float(finite_diff.max()),
        max_rel=float(relative.max()),
        rel_p50=float(quantiles[0]),
        rel_p95=float(quantiles[1]),
        rel_p99=float(quantiles[2]),
        rmse=float(torch.sqrt(torch.mean(finite_diff.square()))),
        max_abs_index=max_index,
    )
    return metrics


def compare_outputs(torch, actual: dict, expected: dict) -> dict:
    result = {"within_tolerance": True, "outputs": {}}
    failures = []
    for name in ("dq", "dkv", "d_sink"):
        result["outputs"][name] = tensor_metrics(
            torch, actual[name], expected[name]
        )
        try:
            torch.testing.assert_close(
                actual[name].float(),
                expected[name].float(),
                atol=ATOL,
                rtol=RTOL,
            )
        except AssertionError as error:
            result["within_tolerance"] = False
            failures.append(f"{name}: {str(error)[:1200]}")
    if failures:
        result["failures"] = failures
    return result


def _dkv_error_summary(torch, actual, expected) -> dict:
    """Return tolerance-aware scalar error metrics for one dKV view."""

    actual = actual.float()
    expected = expected.float()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual - expected).abs()
    tolerance = ATOL + RTOL * expected.abs()
    mismatch = (~finite) | (difference > tolerance)
    finite_difference = difference[finite]
    if bool(finite_difference.numel()):
        rmse = float(torch.sqrt(torch.mean(finite_difference.square())))
        max_abs = float(finite_difference.max())
    else:
        rmse = float("inf")
        max_abs = float("inf")
    mismatch_count = int(mismatch.sum())
    element_count = mismatch.numel()
    return {
        "mismatch_count": mismatch_count,
        "element_count": element_count,
        "mismatch_fraction": mismatch_count / element_count,
        "rmse": rmse,
        "max_abs": max_abs,
    }


def _candidate_column_permutations(torch, column_count: int, device) -> list[dict]:
    """Build likely D/rank/subtile mappings, coalescing equivalent tests."""

    identity = torch.arange(column_count, device=device, dtype=torch.long)
    permutations = []
    by_indices = {}

    def register(name: str, indices, *, aliases=()):
        indices = indices.to(device=device, dtype=torch.long).reshape(-1)
        if indices.numel() != column_count:
            raise ValueError(f"{name}: expected {column_count} columns")
        if not bool(torch.equal(torch.sort(indices).values, identity)):
            raise ValueError(f"{name}: indices are not a permutation")
        key = tuple(indices.cpu().tolist())
        if key in by_indices:
            by_indices[key]["aliases"].extend((name, *aliases))
            return
        entry = {
            "name": name,
            "aliases": list(aliases),
            "indices": indices,
        }
        by_indices[key] = entry
        permutations.append(entry)

    register("identity", identity)

    # Swap the two halves of every 16/32/64-column block.  The chunk-pair
    # aliases make explicit the other common naming convention (two adjacent
    # 8/16/32-column subtiles).
    for block_width in (16, 32, 64):
        offset = identity % block_width
        swapped = identity - offset + (offset + block_width // 2) % block_width
        register(
            f"block{block_width}_half_swap",
            swapped,
            aliases=(f"chunk{block_width // 2}_pair_swap",),
        )

        # Also isolate the same mapping to one block.  A global permutation
        # would hide a real one-panel bug by corrupting every already-correct
        # column; this matters here because the observed 3% mismatch is close
        # to exactly 16 of 512 columns.
        for start in range(0, column_count, block_width):
            local = identity.clone()
            local[start : start + block_width] = swapped[
                start : start + block_width
            ]
            register(f"block{block_width}_half_swap_at_{start:03d}", local)

    # In the alternative convention, "16/32/64 half swap" means exchanging
    # a pair of adjacent 16/32/64-column chunks.  Cover each possible local
    # pair explicitly in addition to the all-block mappings above.
    for chunk_width in (16, 32, 64):
        block_width = 2 * chunk_width
        for start in range(0, column_count, block_width):
            local = identity.clone()
            local[start : start + block_width] = torch.cat(
                (
                    identity[start + chunk_width : start + block_width],
                    identity[start : start + chunk_width],
                )
            )
            register(f"chunk{chunk_width}_pair_swap_at_{start:03d}", local)

    # XOR masks cover lane/vector bit swaps as well as larger subtile, rank,
    # and round swaps.  Duplicate maps above are retained as aliases.
    for mask in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        aliases = ()
        if mask == 64:
            aliases = ("chunk64_pair_swap",)
        elif mask == 128:
            aliases = ("rank128_swap_within_each_round",)
        elif mask == 256:
            aliases = ("round256_half_swap",)
        register(f"xor_{mask}", identity ^ mask, aliases=aliases)

    if column_count == 512:
        # Candidate dKV ownership is naturally [round=2, rank=2, D=128].
        # Test a round/rank axis transpose and rank/subtile interleavings at
        # each likely vector granularity.  Test the inverse as well because a
        # gather permutation has a direction when the axis sizes differ.
        rank_round = (
            identity.reshape(2, 2, 128).permute(1, 0, 2).contiguous().reshape(-1)
        )
        register("rank_round_axis_swap_128", rank_round)

        for round_index in range(2):
            start = round_index * 256
            local_rank_swap = identity.clone()
            local_rank_swap[start : start + 256] = torch.cat(
                (identity[start + 128 : start + 256], identity[start : start + 128])
            )
            register(f"rank128_swap_round{round_index}_only", local_rank_swap)
        for rank_index in range(2):
            first = rank_index * 128
            second = 256 + rank_index * 128
            local_round_swap = identity.clone()
            local_round_swap[first : first + 128] = identity[second : second + 128]
            local_round_swap[second : second + 128] = identity[first : first + 128]
            register(f"round256_swap_rank{rank_index}_only", local_round_swap)

        for subtile_width in (16, 32, 64):
            rank_subtile = (
                identity.reshape(2, 2, 128 // subtile_width, subtile_width)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1)
            )
            name = f"rank_subtile_axis_swap_{subtile_width}"
            register(name, rank_subtile)
            inverse = torch.empty_like(rank_subtile)
            inverse[rank_subtile] = identity
            register(f"{name}_inverse", inverse)
            for round_index in range(2):
                start = round_index * 256
                local = identity.clone()
                local_mapping = (
                    identity[start : start + 256]
                    .reshape(2, 128 // subtile_width, subtile_width)
                    .permute(1, 0, 2)
                    .contiguous()
                    .reshape(-1)
                )
                local[start : start + 256] = local_mapping
                local_name = f"{name}_round{round_index}_only"
                register(local_name, local)
                local_inverse = torch.empty_like(local)
                local_inverse[local] = identity
                register(f"{local_name}_inverse", local_inverse)

    return permutations


def diagnose_dkv_columns(torch, candidate, baseline, *, top_count: int = 24) -> dict:
    """Diagnose whether candidate dKV errors are a fixed column permutation."""

    if candidate.shape != baseline.shape or candidate.ndim != 2:
        raise ValueError(
            "dKV column diagnostics require equal rank-2 tensors, got "
            f"candidate={tuple(candidate.shape)} baseline={tuple(baseline.shape)}"
        )
    candidate = candidate.float()
    baseline = baseline.float()
    difference = (candidate - baseline).abs()
    mismatch = difference > (ATOL + RTOL * baseline.abs())
    finite = torch.isfinite(candidate) & torch.isfinite(baseline)
    mismatch |= ~finite

    mismatch_counts = mismatch.sum(dim=0)
    finite_difference = torch.where(finite, difference, torch.zeros_like(difference))
    finite_counts = finite.sum(dim=0).clamp_min(1)
    column_rmse = torch.sqrt(finite_difference.square().sum(dim=0) / finite_counts)
    column_max = torch.where(
        finite,
        difference,
        torch.full_like(difference, float("inf")),
    ).max(dim=0).values

    mismatch_counts_host = mismatch_counts.cpu().tolist()
    column_rmse_host = column_rmse.cpu().tolist()
    column_max_host = column_max.cpu().tolist()
    row_count, column_count = candidate.shape
    per_column = [
        {
            "column": column,
            "mismatch_count": int(mismatch_counts_host[column]),
            "element_count": row_count,
            "mismatch_fraction": mismatch_counts_host[column] / row_count,
            "rmse": float(column_rmse_host[column]),
            "max_abs": float(column_max_host[column]),
        }
        for column in range(column_count)
    ]

    grouped = {}
    for width in (16, 32, 64):
        groups = []
        for start in range(0, column_count, width):
            end = min(start + width, column_count)
            summary = _dkv_error_summary(
                torch, candidate[:, start:end], baseline[:, start:end]
            )
            groups.append(
                {
                    "group": start // width,
                    "column_start": start,
                    "column_end_exclusive": end,
                    **summary,
                }
            )
        grouped[str(width)] = groups

    def top_by(metric: str) -> list[dict]:
        return sorted(
            per_column,
            key=lambda item: (
                item[metric],
                item["mismatch_count"],
                item["rmse"],
                item["max_abs"],
            ),
            reverse=True,
        )[:top_count]

    identity_summary = _dkv_error_summary(torch, candidate, baseline)
    permutation_tests = []
    for permutation in _candidate_column_permutations(
        torch, column_count, candidate.device
    ):
        permuted = candidate.index_select(1, permutation["indices"])
        summary = _dkv_error_summary(torch, permuted, baseline)
        identity_mismatches = identity_summary["mismatch_count"]
        identity_rmse = identity_summary["rmse"]
        mismatch_reduction = (
            0.0
            if identity_mismatches == 0
            else 1.0 - summary["mismatch_count"] / identity_mismatches
        )
        rmse_reduction = (
            0.0
            if identity_rmse == 0.0
            else 1.0 - summary["rmse"] / identity_rmse
        )
        permutation_tests.append(
            {
                "name": permutation["name"],
                "aliases": permutation["aliases"],
                **summary,
                "mismatch_reduction_vs_identity": mismatch_reduction,
                "rmse_reduction_vs_identity": rmse_reduction,
                "within_tolerance": summary["mismatch_count"] == 0,
                "significant_improvement": (
                    mismatch_reduction >= 0.5 and rmse_reduction >= 0.5
                ),
                "near_eliminates_error": (
                    mismatch_reduction >= 0.9 and rmse_reduction >= 0.9
                ),
            }
        )
    permutation_tests.sort(
        key=lambda item: (item["mismatch_count"], item["rmse"], item["max_abs"])
    )
    best = permutation_tests[0]
    mapping_supported = best["name"] != "identity" and best[
        "near_eliminates_error"
    ]

    return {
        "comparison": "candidate_vs_exact_one_cta_baseline",
        "shape": list(candidate.shape),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "identity": identity_summary,
        "per_column": per_column,
        "groups_by_column_width": grouped,
        "top_error_columns": {
            "by_mismatch_count": top_by("mismatch_count"),
            "by_rmse": top_by("rmse"),
            "by_max_abs": top_by("max_abs"),
        },
        "permutation_tests_best_first": permutation_tests,
        "best_permutation": best,
        "column_mapping_hypothesis_supported": mapping_supported,
        "conclusion": (
            "a tested fixed column mapping nearly eliminates both tolerance "
            "mismatches and RMSE"
            if mapping_supported
            else "none of the tested fixed column mappings nearly eliminates "
            "both tolerance mismatches and RMSE"
        ),
    }


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.describe:
        print(json.dumps(describe_payload(), indent=2, sort_keys=True))
        return 0

    repo = args.repo.resolve()
    candidate_source = args.candidate_source.resolve()
    output = args.output.resolve()
    baseline_source = (
        repo
        / "python/cudnn/deepseek_sparse_attention/sparse_attention_backward"
        / "dsa_bwd_sm100_baseline.py"
    )
    if not baseline_source.is_file():
        raise FileNotFoundError(baseline_source)
    if not candidate_source.is_file():
        raise FileNotFoundError(candidate_source)
    baseline_sha = sha256(baseline_source)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            "exact baseline SHA mismatch: "
            f"expected={EXPECTED_BASELINE_SHA256} actual={baseline_sha}"
        )

    sys.path[:0] = [
        str(repo / "python"),
        str(repo / "test/python"),
        str(repo),
        str(repo / "benchmark/dsa"),
    ]

    import torch
    import cutlass
    import cutlass.cute as cute
    from fe_api.dsa.dsa_reference import ref_sparse_attention_forward
    from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_baseline import (
        FlashAttentionDSABackwardSm100,
    )
    from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name()
    capability = list(torch.cuda.get_device_capability())
    if capability != [10, 7]:
        raise RuntimeError(f"expected Rubin SM107, got {gpu} capability={capability}")

    candidate_class = load_candidate(candidate_source, args.class_name)
    selected_names = list(CASE_SPECS) if args.case == "all" else [args.case]
    payload = {
        "status": "running",
        "contract": describe_payload(),
        "gpu": gpu,
        "compute_capability": capability,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cutlass_dsl_version": importlib.metadata.version(
            "nvidia-cutlass-dsl-internal"
        ),
        "repo": str(repo),
        "baseline_source": str(baseline_source),
        "baseline_sha256": baseline_sha,
        "candidate_source": str(candidate_source),
        "candidate_sha256": sha256(candidate_source),
        "candidate_class": args.class_name,
        "selected_cases": selected_names,
        "records": [],
    }
    write_payload(output, payload)

    started = time.monotonic()
    cases = {
        name: build_case(torch, ref_sparse_attention_forward, name)
        for name in selected_names
    }
    runners = {}
    for has_lengths in sorted(
        {
            case["baseline_case"]["lengths"] is not None
            for case in cases.values()
        }
    ):
        sample = next(
            case["baseline_case"]
            for case in cases.values()
            if (case["baseline_case"]["lengths"] is not None) == has_lengths
        )
        specialization = "lengths" if has_lengths else "none"
        print(
            f"COMPILE_BEGIN impl=baseline specialization={specialization}",
            flush=True,
        )
        runners[("baseline", has_lengths)] = DirectRunner(
            torch=torch,
            cutlass=cutlass,
            cute=cute,
            to_cute_tensor=to_cute_tensor,
            resolve_stream=resolve_stream,
            implementation_class=FlashAttentionDSABackwardSm100,
            implementation_name="baseline",
            has_trace_args=False,
            sample=sample,
        )
        print(
            f"COMPILE_DONE impl=baseline specialization={specialization}",
            flush=True,
        )
    for has_lengths in sorted(
        {cases[name]["lengths"] is not None for name in selected_names}
    ):
        sample = next(
            case
            for case in cases.values()
            if (case["lengths"] is not None) == has_lengths
        )
        specialization = "lengths" if has_lengths else "none"
        print(f"COMPILE_BEGIN impl=candidate specialization={specialization}", flush=True)
        runners[("candidate", has_lengths)] = DirectRunner(
            torch=torch,
            cutlass=cutlass,
            cute=cute,
            to_cute_tensor=to_cute_tensor,
            resolve_stream=resolve_stream,
            implementation_class=candidate_class,
            implementation_name="candidate",
            has_trace_args=True,
            sample=sample,
        )
        print(f"COMPILE_DONE impl=candidate specialization={specialization}", flush=True)

    all_pass = True
    for name in selected_names:
        case = cases[name]
        has_lengths = case["lengths"] is not None
        baseline_has_lengths = case["baseline_case"]["lengths"] is not None
        print(f"CASE_BEGIN name={name}", flush=True)
        case_started = time.monotonic()
        baseline = runners[("baseline", baseline_has_lengths)].run(
            case["baseline_case"]
        )
        candidate = runners[("candidate", has_lengths)].run(case)
        baseline_vs_torch = compare_outputs(torch, baseline, case["reference"])
        candidate_vs_torch = compare_outputs(torch, candidate, case["reference"])
        candidate_vs_baseline = compare_outputs(torch, candidate, baseline)
        dkv_column_diagnostics = None
        if args.diagnose_dkv_columns:
            dkv_column_diagnostics = diagnose_dkv_columns(
                torch, candidate["dkv"], baseline["dkv"]
            )
        case_pass = all(
            comparison["within_tolerance"]
            for comparison in (
                baseline_vs_torch,
                candidate_vs_torch,
                candidate_vs_baseline,
            )
        )
        all_pass &= case_pass
        record = {
            "case": name,
            "description": CASE_SPECS[name]["description"],
            "topk_length": CASE_SPECS[name]["topk_length"],
            "baseline_representation": case["baseline_representation"],
            "status": "pass" if case_pass else "fail",
            "baseline_vs_torch": baseline_vs_torch,
            "candidate_vs_torch": candidate_vs_torch,
            "candidate_vs_baseline": candidate_vs_baseline,
            "duration_s": time.monotonic() - case_started,
        }
        if dkv_column_diagnostics is not None:
            record["dkv_column_diagnostics"] = dkv_column_diagnostics
            diagnostic_best = dkv_column_diagnostics["best_permutation"]
            print(
                "DKV_COLUMN_DIAGNOSTICS "
                f"identity_mismatches="
                f"{dkv_column_diagnostics['identity']['mismatch_count']} "
                f"best={diagnostic_best['name']} "
                f"best_mismatches={diagnostic_best['mismatch_count']} "
                f"best_rmse={diagnostic_best['rmse']:.9g} "
                f"mapping_supported="
                f"{dkv_column_diagnostics['column_mapping_hypothesis_supported']}",
                flush=True,
            )
        payload["records"].append(record)
        payload["status"] = "running"
        write_payload(output, payload)
        print(f"CASE_{'PASS' if case_pass else 'FAIL'} name={name}", flush=True)

    payload["status"] = "pass" if all_pass else "fail"
    payload["duration_s"] = time.monotonic() - started
    payload["native_main_symbols"] = {
        f"{implementation}_{'lengths' if has_lengths else 'none'}": runner.main_symbols
        for (implementation, has_lengths), runner in runners.items()
    }
    write_payload(output, payload)
    print(
        f"DYNAMIC_CORRECTNESS_{'PASS' if all_pass else 'FAIL'} output={output}",
        flush=True,
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
