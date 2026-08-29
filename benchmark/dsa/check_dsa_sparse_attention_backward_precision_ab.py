#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict FP32-oracle and repeatability audit for the H128 DSA backward A/B.

The test deliberately uses Sq=257 and Skv=max(256, 2*topk): the upstream dSink
reduction crosses its 256-query block boundary, while the candidate performs
one FP32 global contribution per query.  Both implementations receive the
same BF16 inputs and run through the same public wrapper.  No timings are
collected.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

import benchmark_dsa_sparse_attention_backward_ab as ab

ERROR_METRICS = ("max_abs_error", "rms_error", "relative_l2_error")
OUTPUTS = ab.OUTPUT_NAMES
ORACLES = ("mathematical_fp32", "kernel_contract_fp32")
UPSTREAM_ATOL = 5.0e-2
UPSTREAM_RTOL = 5.0e-2


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_ints, default=[20260829, 20260830, 20260831])
    parser.add_argument("--topks", type=parse_ints, default=[128, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def fp32_oracles(inputs: dict[str, torch.Tensor | None]) -> dict[str, tuple[torch.Tensor, ...]]:
    required = {name: inputs[name] for name in ("q", "kv", "dout", "sink", "indices", "out", "lse")}
    if any(value is None for value in required.values()):
        raise AssertionError("precision oracle inputs are incomplete")
    q = required["q"].float()
    kv = required["kv"].float()
    dout = required["dout"].float()
    sink = required["sink"].float()
    indices = required["indices"].long()
    supplied_out = required["out"].float()
    supplied_lse = required["lse"].float()
    lengths = inputs["lengths"]
    seqlen_q, _, head_dim = q.shape
    seqlen_kv = kv.shape[0]
    scale = head_dim**-0.5

    valid = (indices >= 0) & (indices < seqlen_kv)
    if lengths is not None:
        positions = torch.arange(indices.shape[1], device=indices.device).unsqueeze(0)
        valid &= positions < lengths.unsqueeze(1)
    mask = torch.zeros((seqlen_q, seqlen_kv), dtype=torch.bool, device=q.device)
    rows = torch.arange(seqlen_q, device=q.device).unsqueeze(1).expand_as(indices)
    mask[rows[valid], indices[valid]] = True

    scores = torch.einsum("qhd,kd->qhk", q, kv) * scale
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    exact_lse = torch.logsumexp(scores, dim=-1)
    dp = torch.einsum("qhd,kd->qhk", dout, kv)

    def gradients(denominator: torch.Tensor, delta_out: torch.Tensor) -> tuple[torch.Tensor, ...]:
        probabilities = torch.exp(scores - denominator.unsqueeze(-1))
        delta = (dout * delta_out).sum(dim=-1)
        ds = probabilities * (dp - delta.unsqueeze(-1))
        dq = torch.einsum("qhk,kd->qhd", ds, kv) * scale
        dv = torch.einsum("qhk,qhd->kd", probabilities, dout)
        dk = torch.einsum("qhk,qhd->kd", ds, q) * scale
        p_sink = torch.exp(sink.unsqueeze(0) - denominator)
        dsink = (-p_sink * delta).sum(dim=0)
        return dq, dk + dv, dsink

    exact_denominator = torch.logaddexp(exact_lse, sink.unsqueeze(0))
    exact_out = torch.einsum("qhk,kd->qhd", torch.exp(scores - exact_denominator.unsqueeze(-1)), kv)
    contract_denominator = torch.logaddexp(supplied_lse, sink.unsqueeze(0))
    return {
        "mathematical_fp32": gradients(exact_denominator, exact_out),
        "kernel_contract_fp32": gradients(contract_denominator, supplied_out),
    }


@torch.no_grad()
def enqueue_error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, torch.Tensor]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    return {
        "max_abs_error": difference.abs().max(),
        "rms_error": difference.square().mean().sqrt(),
        "relative_l2_error": torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected_f32).clamp_min(1.0e-30),
        "outside_upstream_tolerance": torch.count_nonzero(~torch.isclose(actual_f32, expected_f32, atol=UPSTREAM_ATOL, rtol=UPSTREAM_RTOL)),
        "nonfinite": torch.count_nonzero(~torch.isfinite(actual_f32)),
    }


@torch.no_grad()
def enqueue_jitter_metrics(actual: torch.Tensor, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics = enqueue_error_metrics(actual, anchor)
    metrics["byte_mismatch"] = torch.count_nonzero(actual.contiguous().view(torch.uint8) != anchor.contiguous().view(torch.uint8))
    return metrics


def materialize_metrics(pending: dict[str, torch.Tensor], context: str) -> dict[str, float | int]:
    nonfinite = int(pending["nonfinite"].item())
    if nonfinite:
        raise RuntimeError(f"nonfinite output in {context}: {nonfinite} values")
    result: dict[str, float | int] = {}
    for name, value in pending.items():
        scalar = value.item()
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise RuntimeError(f"nonfinite metric {name} in {context}: {scalar}")
        result[name] = scalar
    return result


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(max(values)),
    }


def summarize_samples(samples: list[dict]) -> dict:
    oracle_summary = {}
    for oracle in ORACLES:
        oracle_summary[oracle] = {}
        for output in OUTPUTS:
            rows = [sample["oracles"][oracle][output] for sample in samples]
            oracle_summary[oracle][output] = {metric: distribution([float(row[metric]) for row in rows]) for metric in ERROR_METRICS}
            oracle_summary[oracle][output]["upstream_tolerance_passed"] = all(
                int(row["outside_upstream_tolerance"]) == 0 and int(row["nonfinite"]) == 0 for row in rows
            )
    jitter_summary = {}
    for output in OUTPUTS:
        rows = [sample["jitter_vs_first"][output] for sample in samples]
        jitter_summary[output] = {metric: distribution([float(row[metric]) for row in rows]) for metric in ERROR_METRICS}
        jitter_summary[output]["byte_changing_runs"] = sum(int(row["byte_mismatch"]) != 0 for row in rows)
        jitter_summary[output]["max_byte_mismatch"] = max(int(row["byte_mismatch"]) for row in rows)
    return {"oracles": oracle_summary, "jitter_vs_first": jitter_summary}


def run_variant(suite: ab.CaseSuite, label: str, oracles: dict[str, tuple[torch.Tensor, ...]], repeats: int) -> dict:
    pending_samples = []
    anchor = None
    for iteration in range(repeats):
        ab.poison_public_outputs(suite, label)
        actual = suite.calls[label](None)
        if anchor is None:
            anchor = tuple(value.detach().clone() for value in actual)
        pending_samples.append(
            {
                "iteration": iteration,
                "oracles": {
                    oracle: {output: enqueue_error_metrics(value, reference) for output, value, reference in zip(OUTPUTS, actual, oracle_values, strict=True)}
                    for oracle, oracle_values in oracles.items()
                },
                "jitter_vs_first": {output: enqueue_jitter_metrics(value, first) for output, value, first in zip(OUTPUTS, actual, anchor, strict=True)},
            }
        )
    torch.cuda.synchronize()
    samples = []
    for pending in pending_samples:
        iteration = pending["iteration"]
        samples.append(
            {
                "iteration": iteration,
                "oracles": {
                    oracle: {output: materialize_metrics(metrics, f"{label} iteration {iteration} {oracle} {output}") for output, metrics in outputs.items()}
                    for oracle, outputs in pending["oracles"].items()
                },
                "jitter_vs_first": {
                    output: materialize_metrics(metrics, f"{label} iteration {iteration} jitter {output}")
                    for output, metrics in pending["jitter_vs_first"].items()
                },
            }
        )
    return {"samples": samples, "summary": summarize_samples(samples)}


def compare_summaries(baseline: dict, candidate: dict) -> dict:
    regressions = []
    jitter_regressions = []
    comparisons = {}
    for oracle in ORACLES:
        comparisons[oracle] = {}
        for output in OUTPUTS:
            comparisons[oracle][output] = {}
            for metric in ERROR_METRICS:
                baseline_rows = baseline["oracles"][oracle][output][metric]
                candidate_rows = candidate["oracles"][oracle][output][metric]
                jitter_allowance = baseline["jitter_vs_first"][output][metric]["max"]
                row = {
                    "baseline_p95": baseline_rows["p95"],
                    "candidate_p95": candidate_rows["p95"],
                    "baseline_max": baseline_rows["max"],
                    "candidate_max": candidate_rows["max"],
                    "observed_jitter_allowance": jitter_allowance,
                }
                row["resolved_regression"] = (
                    candidate_rows["p95"] > baseline_rows["p95"] + jitter_allowance or candidate_rows["max"] > baseline_rows["max"] + jitter_allowance
                )
                if row["resolved_regression"]:
                    regressions.append({"oracle": oracle, "output": output, "metric": metric, **row})
                comparisons[oracle][output][metric] = row
    for output in OUTPUTS:
        for metric in ERROR_METRICS:
            baseline_jitter = baseline["jitter_vs_first"][output][metric]["max"]
            candidate_jitter = candidate["jitter_vs_first"][output][metric]["max"]
            if candidate_jitter > baseline_jitter:
                jitter_regressions.append(
                    {
                        "output": output,
                        "metric": metric,
                        "baseline_max": baseline_jitter,
                        "candidate_max": candidate_jitter,
                    }
                )
    coarse_passed = all(
        impl["oracles"][oracle][output]["upstream_tolerance_passed"] for impl in (baseline, candidate) for oracle in ORACLES for output in OUTPUTS
    )
    dq_stable = all(impl["jitter_vs_first"]["dq"]["byte_changing_runs"] == 0 for impl in (baseline, candidate))
    return {
        "coarse_upstream_tolerance_passed": coarse_passed,
        "dq_bitwise_stable": dq_stable,
        "resolved_regressions": regressions,
        "candidate_jitter_regressions": jitter_regressions,
        "no_resolved_regression": coarse_passed and dq_stable and not regressions and not jitter_regressions,
        "comparisons": comparisons,
        "interpretation": "Candidate oracle error may be excused only by baseline self-jitter, never by candidate self-jitter; any measured candidate jitter above baseline is a regression. No resolved regression is still not bitwise equivalence.",
    }


def main() -> None:
    args = parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must be a nonempty unique list")
    if not args.topks or len(args.topks) != len(set(args.topks)) or not set(args.topks) <= {128, 512, 1024, 2048}:
        raise ValueError("--topks must be a nonempty unique subset of 128,512,1024,2048")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("strict FP32 oracle requires NVIDIA_TF32_OVERRIDE=0 in the process environment")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 invalidates the strict FP32 oracle")
    repo_root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    if args.output == repo_root or repo_root in args.output.parents:
        raise ValueError("precision output must be outside the git checkout")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite precision evidence: {args.output}")

    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("failed to enforce the strict FP32 oracle math mode")
    try:
        shared_closure = ab.audit_shared_closure(repo_root)
        candidate = importlib.import_module(ab.CANDIDATE_INTERFACE_MODULE)
        import_origins = ab.audit_import_origins(repo_root, candidate)
        baseline = ab.load_baseline_interface(repo_root)
        if baseline.flash_attn_bwd_sm100.compile_cache is candidate.flash_attn_bwd_sm100.compile_cache:
            raise RuntimeError("baseline and candidate unexpectedly share one compile cache")
        if baseline._select_sm100_backend(128, 512) != ("generic_m64", 64):
            raise RuntimeError("baseline interface does not route H128/D512 to generic_m64")
        for topk in args.topks:
            if candidate._select_sm100_backend(
                128,
                512,
                head_dim_v=512,
                dtype=torch.bfloat16,
                max_topk=topk,
                device_capability=(10, 0),
            ) != ("h128_2cta_m64", 64):
                raise RuntimeError(f"candidate interface does not route topk={topk} to h128_2cta_m64")
        environment = ab.collect_environment(repo_root, shared_closure, import_origins)
        ab.validate_environment(environment)
        environment["software"]["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        environment["software"]["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        environment["software"]["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        environment["software"]["tf32_environment"] = {
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        }
        cases = []
        for seed in args.seeds:
            for topk in args.topks:
                for length_mode in ("none", "full"):
                    suite = ab.make_case(
                        seqlen=257,
                        seqlen_kv=max(256, 2 * topk),
                        topk=topk,
                        has_topk_length=length_mode == "full",
                        seed=seed,
                        baseline=baseline,
                        candidate=candidate,
                    )
                    oracles = fp32_oracles(suite.inputs)
                    baseline_result = run_variant(suite, "baseline_a", oracles, args.repeats)
                    candidate_result = run_variant(suite, "candidate_a", oracles, args.repeats)
                    has_lengths = length_mode == "full"
                    candidate_cache_keys = list(candidate.flash_attn_bwd_sm100.compile_cache)
                    baseline_cache_keys = list(baseline.flash_attn_bwd_sm100.compile_cache)
                    if not any(len(key) == 8 and key[0] == "h128_2cta_m64" and key[-2:] == (topk, has_lengths) for key in candidate_cache_keys):
                        raise RuntimeError(f"candidate precision run missed the two-CTA cache key for topk={topk}, mode={length_mode}")
                    if not any(len(key) == 7 and key[-2:] == (topk, has_lengths) for key in baseline_cache_keys):
                        raise RuntimeError(f"baseline precision run missed the generic cache key for topk={topk}, mode={length_mode}")
                    assessment = compare_summaries(baseline_result["summary"], candidate_result["summary"])
                    cases.append(
                        {
                            "case": suite.metadata,
                            "baseline": baseline_result,
                            "candidate": candidate_result,
                            "assessment": assessment,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "case": suite.metadata,
                                "no_resolved_regression": assessment["no_resolved_regression"],
                                "resolved_regression_count": len(assessment["resolved_regressions"]),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        selected_uuid = environment["gpu"]["identity"]["uuid"]
        state_after = ab.gpu_state(selected_uuid)
        ab.validate_gpu_state(state_after, selected_uuid, environment["gpu"]["self_pid_aliases"])
        environment["gpu"]["state_after"] = state_after
        status_after = ab.command_output(["git", "-C", str(repo_root), "status", "--porcelain"], required=True)
        environment["repository"]["status_after"] = status_after
        if status_after:
            raise RuntimeError("precision checkout became dirty during the run")
        passed = all(case["assessment"]["no_resolved_regression"] for case in cases)
        result = {
            "schema_version": 1,
            "status": "complete" if passed else "numerical_regression",
            "precision_claim_eligible": (
                passed and args.seeds == [20260829, 20260830, 20260831] and args.topks == [128, 512, 1024, 2048] and args.repeats == 50
            ),
            "protocol": {
                "baseline": f"upstream interface at {ab.BASELINE_COMMIT}",
                "candidate": environment["repository"]["head"],
                "shape": {"seqlen_q": 257, "seqlen_kv": "max(256, 2*topk)", "heads": 128, "head_dim": 512},
                "topks": args.topks,
                "dtype": "torch.bfloat16",
                "accumulation_reference": "analytical dense FP32 gradients with TF32 disabled and highest FP32 matmul precision",
                "seeds": args.seeds,
                "length_modes": ["none", "full"],
                "repeats": args.repeats,
                "upstream_tolerance": {"atol": UPSTREAM_ATOL, "rtol": UPSTREAM_RTOL},
                "argv": sys.argv,
            },
            "environment": environment,
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        if temporary_output.exists():
            raise FileExistsError(f"refusing to overwrite stale temporary evidence: {temporary_output}")
        temporary_output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(temporary_output, args.output)
        print(json.dumps({"output": str(args.output), "status": result["status"]}, indent=2), flush=True)
        if not passed:
            raise SystemExit("candidate has a resolved numerical regression")
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


if __name__ == "__main__":
    main()
