#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fair same-process B200 A/B for the H128 DSA backward specialization.

The baseline interface is loaded directly from the pinned upstream git blob.
The candidate and baseline then run through the same public cuDNN Frontend
wrapper in one Python/CUDA context. Four independent treatment arms use a
carryover-balanced Williams design; no product API or selector test hook is
needed.
"""

from __future__ import annotations

import argparse
import base64
import csv
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import types
from typing import Callable

from cuda.bindings import driver as cuda
import numpy as np
import torch

import cudnn as cudnn_module
from cudnn import DSA
from cudnn.deepseek_sparse_attention.sparse_attention_backward import api as backward_api

BASELINE_COMMIT = "606e16f9786ea7a13e0462c8a63edf0d7f72ae85"
INTERFACE_PATH = "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py"
CANDIDATE_INTERFACE_MODULE = "cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100"
BASELINE_INTERFACE_MODULE = "cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100_baseline_606e"

SHARED_CLOSURE_PATHS = (
    "python/cudnn/__init__.py",
    "python/cudnn/api_base.py",
    "python/cudnn/deepseek_sparse_attention/__init__.py",
    "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/__init__.py",
    "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/api.py",
    "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py",
    "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_h16.py",
    "python/cudnn/deepseek_sparse_attention/utils/compiler.py",
    "python/cudnn/deepseek_sparse_attention/utils/runtime.py",
    "python/cudnn/deepseek_sparse_attention/utils/tensor_conversion.py",
)

LABELS = ("baseline_a", "candidate_a", "baseline_b", "candidate_b")
OUTPUT_NAMES = ("dq", "dkv", "d_sink")
EXPECTED_CUTE_DSL = "4.5.2"
EXPECTED_CUTLASS_IR_SHA256 = "73b760621e35910305e7bdf8f4c2c0d928c10527a243f8f11a76046edba4f6d8"
EXPECTED_CUTE_RUNTIME_SHA256 = "deb32d6b6857ba753f421dd004d418d68825ca023dd2b51a95c8f106a17fb0a1"
ELEMENTWISE_ATOL = 0.004
ELEMENTWISE_RTOL = 0.01
RMS_ATOL = 1.0e-6
RMS_RTOL = 0.005


def command_output(command: list[str], *, required: bool = False) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        if required:
            raise RuntimeError(f"command failed: {command!r}: {error}") from error
        return f"unavailable: {error}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def cutlass_backend_audit() -> dict:
    distribution = importlib.metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    libraries = []
    for package_path in distribution.files or ():
        component = None
        if package_path.name.startswith("_cutlass_ir") and package_path.name.endswith(".so"):
            component = "cutlass_ir"
        elif package_path.name == "libcute_dsl_runtime.so":
            component = "cute_dsl_runtime"
        if component is None:
            continue
        active_path = Path(distribution.locate_file(package_path)).resolve()
        expected_record = package_path.hash.value if package_path.hash else None
        actual_record = record_digest(active_path) if active_path.is_file() else None
        libraries.append(
            {
                "component": component,
                "path": str(active_path),
                "sha256": sha256_file(active_path) if active_path.is_file() else None,
                "record_digest": expected_record,
                "active_record_digest": actual_record,
                "matches_record": bool(expected_record and actual_record == expected_record),
            }
        )
    expected_hashes = {
        "cutlass_ir": EXPECTED_CUTLASS_IR_SHA256,
        "cute_dsl_runtime": EXPECTED_CUTE_RUNTIME_SHA256,
    }
    by_component = {row["component"]: row for row in libraries}
    verified = (
        distribution.version == EXPECTED_CUTE_DSL
        and set(by_component) == set(expected_hashes)
        and all(row["matches_record"] for row in libraries)
        and all(by_component[name]["sha256"] == digest for name, digest in expected_hashes.items())
    )
    return {
        "distribution": "nvidia-cutlass-dsl-libs-cu13",
        "version": distribution.version,
        "libraries": libraries,
        "expected_sha256": expected_hashes,
        "verified": verified,
    }


def git_blob(repo_root: Path, revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo_root), "show", f"{revision}:{path}"])


def audit_shared_closure(repo_root: Path) -> dict[str, dict[str, str | bool]]:
    rows = {}
    for relative in SHARED_CLOSURE_PATHS:
        current = (repo_root / relative).read_bytes()
        baseline = git_blob(repo_root, BASELINE_COMMIT, relative)
        rows[relative] = {
            "current_sha256": sha256_bytes(current),
            "baseline_sha256": sha256_bytes(baseline),
            "byte_identical": current == baseline,
        }
    mismatches = [path for path, row in rows.items() if not row["byte_identical"]]
    if mismatches:
        raise RuntimeError(f"shared baseline closure differs from {BASELINE_COMMIT}: {mismatches}")
    return rows


def require_module_origin(module: types.ModuleType, expected: Path) -> None:
    actual = Path(module.__file__ or "").resolve()
    expected = expected.resolve()
    if actual != expected:
        raise RuntimeError(f"module {module.__name__} came from {actual}, expected {expected}")


def audit_import_origins(repo_root: Path, candidate: types.ModuleType) -> dict[str, str]:
    dsa_module = importlib.import_module("cudnn.deepseek_sparse_attention")
    if DSA.sparse_attention_backward_wrapper is not backward_api.sparse_attention_backward_wrapper:
        raise RuntimeError("public DSA wrapper identity does not match the audited backward_api module")
    expected_modules = {
        cudnn_module: repo_root / "python/cudnn/__init__.py",
        dsa_module: repo_root / "python/cudnn/deepseek_sparse_attention/__init__.py",
        backward_api: repo_root / "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/api.py",
        candidate: repo_root / INTERFACE_PATH,
        importlib.import_module("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_2cta"): repo_root
        / "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_h128_2cta.py",
        importlib.import_module("cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100"): repo_root
        / "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py",
        importlib.import_module("cudnn.deepseek_sparse_attention.utils.compiler"): repo_root / "python/cudnn/deepseek_sparse_attention/utils/compiler.py",
        importlib.import_module("cudnn.deepseek_sparse_attention.utils.runtime"): repo_root / "python/cudnn/deepseek_sparse_attention/utils/runtime.py",
        importlib.import_module("cudnn.deepseek_sparse_attention.utils.tensor_conversion"): repo_root
        / "python/cudnn/deepseek_sparse_attention/utils/tensor_conversion.py",
    }
    for module, expected in expected_modules.items():
        require_module_origin(module, expected)
    return {module.__name__: str(Path(module.__file__ or "").resolve()) for module in expected_modules}


def load_baseline_interface(repo_root: Path) -> types.ModuleType:
    source = git_blob(repo_root, BASELINE_COMMIT, INTERFACE_PATH)
    module = types.ModuleType(BASELINE_INTERFACE_MODULE)
    module.__file__ = f"{repo_root / INTERFACE_PATH}@{BASELINE_COMMIT}"
    module.__package__ = "cudnn.deepseek_sparse_attention.sparse_attention_backward"
    sys.modules[BASELINE_INTERFACE_MODULE] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@contextmanager
def routed_interface(module: types.ModuleType):
    previous = backward_api._iface_sm100
    backward_api._iface_sm100 = module
    try:
        yield
    finally:
        backward_api._iface_sm100 = previous


def module_for_label(label: str, baseline: types.ModuleType, candidate: types.ModuleType) -> types.ModuleType:
    return baseline if label.startswith("baseline") else candidate


def williams_orders(window: int) -> tuple[tuple[str, ...], ...]:
    """Eight position- and first-order-carryover-balanced treatment orders."""
    base_row = (0, 1, 3, 2)
    design = tuple(tuple(LABELS[(value + offset) % len(LABELS)] for value in base_row) for offset in range(len(LABELS)))
    boundary_spread = (0, 1, 2, 3, 0, 3, 2, 1)
    start = window % len(boundary_spread)
    return tuple(design[boundary_spread[(start + offset) % len(boundary_spread)]] for offset in range(len(boundary_spread)))


def make_indices(seqlen_q: int, seqlen_kv: int, topk: int, seed: int, generator: torch.Generator) -> tuple[torch.Tensor, dict]:
    """Match the repository benchmark's independent random ordering per row."""
    if not 0 < topk <= seqlen_kv:
        raise ValueError(f"invalid topk={topk} for seqlen_kv={seqlen_kv}")
    random_keys = torch.rand((seqlen_q, seqlen_kv), dtype=torch.float32, device="cuda", generator=generator)
    indices = random_keys.argsort(dim=-1)[:, :topk].to(torch.int32).contiguous()
    del random_keys
    host = indices.cpu()
    sorted_rows = torch.sort(indices, dim=1).values
    duplicate_count = int((sorted_rows[:, 1:] == sorted_rows[:, :-1]).sum().item()) if topk > 1 else 0
    if duplicate_count:
        raise AssertionError(f"index generator produced {duplicate_count} duplicates")
    return indices, {
        "construction": "seeded torch.rand(Sq,Skv).argsort(dim=-1)[:,:topk]",
        "source": "benchmark/dsa/benchmark_dsa_sparse_attention_backward.py:make_inputs",
        "seed": seed,
        "duplicate_count": duplicate_count,
        "range": [int(indices.min().item()), int(indices.max().item())],
        "sha256": hashlib.sha256(host.numpy().tobytes()).hexdigest(),
    }


@torch.no_grad()
def forward_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    *,
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    for begin in range(0, q.shape[0], chunk):
        end = min(begin + chunk, q.shape[0])
        selected = kv[indices[begin:end].long()].float()
        scores = torch.einsum("qhd,qkd->qhk", q[begin:end].float(), selected) * scale
        lse_kv = torch.logsumexp(scores, dim=-1)
        denominator = torch.logaddexp(lse_kv, sink.unsqueeze(0))
        probabilities = torch.exp(scores - denominator.unsqueeze(-1))
        output[begin:end] = torch.einsum("qhk,qkd->qhd", probabilities, selected).to(q.dtype)
        lse[begin:end] = lse_kv
    return output, lse


@dataclass
class CaseSuite:
    calls: dict[str, Callable[[torch.cuda.Stream | None], tuple[torch.Tensor, ...]]]
    raw_calls: dict[str, Callable[[cuda.CUstream | None], tuple[torch.Tensor, ...]]]
    interfaces: dict[str, types.ModuleType]
    public_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]]
    inputs: dict[str, torch.Tensor | None]
    keepalive: tuple[object, ...]
    metadata: dict


def make_case(
    *,
    seqlen: int,
    topk: int,
    has_topk_length: bool,
    seed: int,
    baseline: types.ModuleType,
    candidate: types.ModuleType,
    seqlen_kv: int | None = None,
) -> CaseSuite:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    resolved_seqlen_kv = seqlen if seqlen_kv is None else seqlen_kv
    shape = (seqlen, 128, 512)
    q = (torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator) * 0.1).contiguous()
    kv = (torch.randn((resolved_seqlen_kv, 512), dtype=torch.bfloat16, device="cuda", generator=generator) * 0.1).contiguous()
    dout = (torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator) * 0.1).contiguous()
    sink = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32, device="cuda")
    indices, index_metadata = make_indices(seqlen, resolved_seqlen_kv, topk, seed, generator)
    lengths = torch.full((seqlen,), topk, dtype=torch.int32, device="cuda") if has_topk_length else None
    scale = 512**-0.5
    out, lse = forward_reference(q, kv, sink, indices, scale)
    public_outputs = {label: (torch.full_like(q, float("nan")), torch.full_like(kv, float("nan"))) for label in LABELS}

    def invoke_raw(label: str, cu_stream: cuda.CUstream | None = None) -> tuple[torch.Tensor, ...]:
        dq, dkv = public_outputs[label]
        result = DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            sink,
            indices,
            softmax_scale=scale,
            topk_length=lengths,
            dq=dq,
            dkv=dkv,
            stream=cu_stream,
        )
        values = (result["dq"], result["dkv"], result["d_sink"])
        if values[0] is not dq or values[1] is not dkv:
            raise RuntimeError(f"{label} did not honor caller-owned dq/dkv")
        return values

    def invoke(label: str, stream: torch.cuda.Stream | None = None) -> tuple[torch.Tensor, ...]:
        cu_stream = cuda.CUstream(stream.cuda_stream) if stream is not None else None
        with routed_interface(module_for_label(label, baseline, candidate)):
            return invoke_raw(label, cu_stream)

    calls = {label: (lambda stream=None, label=label: invoke(label, stream)) for label in LABELS}
    raw_calls = {label: (lambda cu_stream=None, label=label: invoke_raw(label, cu_stream)) for label in LABELS}
    interfaces = {label: module_for_label(label, baseline, candidate) for label in LABELS}
    return CaseSuite(
        calls=calls,
        raw_calls=raw_calls,
        interfaces=interfaces,
        public_outputs=public_outputs,
        inputs={"q": q, "kv": kv, "dout": dout, "sink": sink, "indices": indices, "lengths": lengths, "out": out, "lse": lse},
        keepalive=(q, kv, dout, sink, indices, lengths, out, lse, public_outputs),
        metadata={
            "seqlen_q": seqlen,
            "seqlen_kv": resolved_seqlen_kv,
            "topk": topk,
            "topk_length": "full" if has_topk_length else "none",
            "dtype": "torch.bfloat16",
            "num_heads": 128,
            "head_dim": 512,
            "head_dim_v": 512,
            "indices": index_metadata,
            "seed": seed,
        },
    )


def output_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    rms_error = float(difference.square().mean().sqrt().item())
    reference_rms = float(expected_f32.square().mean().sqrt().item())
    rms_tolerance = RMS_ATOL + RMS_RTOL * reference_rms
    nonfinite = int((~torch.isfinite(actual_f32)).sum().item())
    outside = int((~torch.isclose(actual_f32, expected_f32, atol=ELEMENTWISE_ATOL, rtol=ELEMENTWISE_RTOL)).sum().item())
    return {
        "passed": nonfinite == 0 and outside == 0 and math.isfinite(rms_error) and rms_error <= rms_tolerance,
        "max_abs_error": float(difference.abs().max().item()),
        "rms_error": rms_error,
        "reference_rms": reference_rms,
        "rms_relative_error": rms_error / max(reference_rms, 1.0e-12),
        "rms_tolerance": rms_tolerance,
        "outside_elementwise_tolerance": outside,
        "nonfinite": nonfinite,
    }


def compare_outputs(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> dict:
    outputs = {name: output_metrics(value, reference) for name, value, reference in zip(OUTPUT_NAMES, actual, expected, strict=True)}
    return {"passed": all(row["passed"] for row in outputs.values()), "outputs": outputs}


@torch.no_grad()
def contract_dsink_reference(inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
    out = inputs["out"]
    dout = inputs["dout"]
    lse = inputs["lse"]
    sink = inputs["sink"]
    if out is None or dout is None or lse is None or sink is None:
        raise AssertionError("missing tensors for the dSink contract reference")
    delta = (out.float() * dout.float()).sum(dim=-1)
    denominator = torch.logaddexp(lse.float(), sink.float().unsqueeze(0))
    return (-torch.exp(sink.float().unsqueeze(0) - denominator) * delta).sum(dim=0)


def poison_public_outputs(suite: CaseSuite, label: str, stream: torch.cuda.Stream | None = None) -> None:
    context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        for output in suite.public_outputs[label]:
            output.fill_(float("nan"))


def direct_correctness_gate(suite: CaseSuite, consecutive: int) -> dict:
    poison_public_outputs(suite, "baseline_a")
    reference_values = suite.calls["baseline_a"](None)
    torch.cuda.synchronize()
    reference = tuple(value.detach().clone() for value in reference_values)
    dsink_reference = contract_dsink_reference(suite.inputs)
    torch.cuda.synchronize()
    variants = {}
    for label in LABELS:
        poison_public_outputs(suite, label)
        single = suite.calls[label](None)
        torch.cuda.synchronize()
        single_check = compare_outputs(single, reference)
        poison_public_outputs(suite, label)
        burst = None
        for _ in range(consecutive):
            burst = suite.calls[label](None)
        if burst is None:
            raise AssertionError("consecutive correctness burst was empty")
        torch.cuda.synchronize()
        burst_check = compare_outputs(burst, reference)
        single_dsink_check = output_metrics(single[2], dsink_reference)
        burst_dsink_check = output_metrics(burst[2], dsink_reference)
        variants[label] = {
            "passed": single_check["passed"] and burst_check["passed"] and single_dsink_check["passed"] and burst_dsink_check["passed"],
            "single_poisoned": single_check,
            "burst": burst_check,
            "single_dsink_vs_fp32_contract": single_dsink_check,
            "burst_dsink_vs_fp32_contract": burst_dsink_check,
        }
    return {
        "passed": all(row["passed"] for row in variants.values()),
        "consecutive_calls": consecutive,
        "reference": "fresh baseline_a output cloned before a poisoned single call and an uninterrupted unsynchronized burst",
        "variants": variants,
    }


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    capture_stream: torch.cuda.Stream
    outputs: tuple[torch.Tensor, ...]
    calls: int
    pool_token: object


def capture_graph(call: Callable[[torch.cuda.Stream | None], tuple[torch.Tensor, ...]], calls: int) -> CapturedGraph:
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        call(capture_stream)
        call(capture_stream)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    captured_outputs = None
    with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="global"):
        for _ in range(calls):
            captured_outputs = call(capture_stream)
    capture_stream.synchronize()
    if captured_outputs is None:
        raise AssertionError("CUDA Graph capture produced no output")
    return CapturedGraph(graph, capture_stream, tuple(captured_outputs), calls, graph.pool())


class EventTimer:
    def __init__(self, suite: CaseSuite, captured: dict[str, CapturedGraph], cold_l2_bytes: int):
        self.suite = suite
        self.captured = captured
        self.stream = torch.cuda.Stream()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        generator = torch.Generator(device="cuda").manual_seed(0xC01D12)
        self.flush = torch.randint(-128, 128, (cold_l2_bytes,), dtype=torch.int8, device="cuda", generator=generator)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.cu_stream = cuda.CUstream(self.stream.cuda_stream)

    def measure(self, label: str, mode: str) -> tuple[float, float]:
        if mode == "eager_cold_l2":
            with torch.cuda.stream(self.stream):
                self.flush.sum()
            self.stream.synchronize()
        previous = backward_api._iface_sm100
        backward_api._iface_sm100 = self.suite.interfaces[label]
        try:
            host_start = time.perf_counter()
            with torch.cuda.stream(self.stream):
                self.start.record(self.stream)
                if mode == "graph50":
                    self.captured[label].graph.replay()
                else:
                    self.suite.raw_calls[label](self.cu_stream)
                self.end.record(self.stream)
            self.end.synchronize()
        finally:
            backward_api._iface_sm100 = previous
        divisor = self.captured[label].calls if mode == "graph50" else 1
        latency_ms = float(self.start.elapsed_time(self.end) / divisor)
        host_wall_ms = (time.perf_counter() - host_start) * 1000.0 / divisor
        return latency_ms, host_wall_ms


def bootstrap_exp_mean(log_values: list[float], samples: int, seed: int) -> list[float]:
    logs = np.asarray(log_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = logs[rng.integers(0, len(logs), size=(samples, len(logs)))].mean(axis=1)
    return [float(value) for value in np.quantile(np.exp(draws), [0.025, 0.975])]


def summarize_windows(windows: list[dict], bootstrap_samples: int, seed: int) -> dict:
    def label_summary(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "geomean_ms": math.exp(statistics.fmean(math.log(value) for value in values)),
            "median_ms": statistics.median(values),
            "mad_ms": statistics.median(abs(value - statistics.median(values)) for value in values),
            "p5_ms": float(np.quantile(ordered, 0.05)),
            "p95_ms": float(np.quantile(ordered, 0.95)),
        }

    def paired_summary(field: str, summary_seed: int) -> dict:
        means = {label: [window[field][label] for window in windows] for label in LABELS}
        log_speedup = [
            0.5
            * (
                math.log(window[field]["baseline_a"])
                + math.log(window[field]["baseline_b"])
                - math.log(window[field]["candidate_a"])
                - math.log(window[field]["candidate_b"])
            )
            for window in windows
        ]
        baseline_drift = [math.log(a / b) for a, b in zip(means["baseline_a"], means["baseline_b"], strict=True)]
        candidate_drift = [math.log(a / b) for a, b in zip(means["candidate_a"], means["candidate_b"], strict=True)]
        labels = {label: label_summary(values) for label, values in means.items()}
        baseline_ms = math.sqrt(labels["baseline_a"]["geomean_ms"] * labels["baseline_b"]["geomean_ms"])
        candidate_ms = math.sqrt(labels["candidate_a"]["geomean_ms"] * labels["candidate_b"]["geomean_ms"])
        return {
            "speedup": math.exp(statistics.fmean(log_speedup)),
            "speedup_ci95": bootstrap_exp_mean(log_speedup, bootstrap_samples, summary_seed),
            "log_speedup_by_window": log_speedup,
            "centered_baseline_geomean_ms": baseline_ms,
            "centered_candidate_geomean_ms": candidate_ms,
            "labels": labels,
            "baseline_duplicate_ratio": math.exp(statistics.fmean(baseline_drift)),
            "baseline_duplicate_ratio_ci95": bootstrap_exp_mean(baseline_drift, bootstrap_samples, summary_seed + 1),
            "candidate_duplicate_ratio": math.exp(statistics.fmean(candidate_drift)),
            "candidate_duplicate_ratio_ci95": bootstrap_exp_mean(candidate_drift, bootstrap_samples, summary_seed + 2),
        }

    device = paired_summary("means_ms", seed)
    host = paired_summary("host_wall_means_ms", seed + 10)
    drift_bounds = device["baseline_duplicate_ratio_ci95"] + device["candidate_duplicate_ratio_ci95"]
    return {**device, "host_wall": host, "duplicate_drift_within_0p5pct": all(0.995 <= value <= 1.005 for value in drift_bounds)}


def aggregate_speedup(log_matrix: np.ndarray, bootstrap_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bootstrap_chunks = []
    for begin in range(0, bootstrap_samples, 5000):
        count = min(5000, bootstrap_samples - begin)
        indices = rng.integers(0, log_matrix.shape[1], size=(count, log_matrix.shape[0], log_matrix.shape[1]))
        source = np.broadcast_to(log_matrix, (count, *log_matrix.shape))
        bootstrap_chunks.append(np.exp(np.take_along_axis(source, indices, axis=2).mean(axis=(1, 2))))
    samples = np.concatenate(bootstrap_chunks)
    return {
        "equal_weight_geomean_speedup": float(np.exp(log_matrix.mean())),
        "speedup_ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
    }


def measure_mode(
    suite: CaseSuite,
    captured: dict[str, CapturedGraph],
    *,
    mode: str,
    warmup_windows: int,
    measured_windows: int,
    cold_l2_bytes: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    timer = EventTimer(suite, captured, cold_l2_bytes)
    for window in range(warmup_windows):
        for order in williams_orders(window):
            for label in order:
                timer.measure(label, mode)
    windows = []
    for window in range(measured_windows):
        samples = []
        values = {label: [] for label in LABELS}
        host_values = {label: [] for label in LABELS}
        for order_index, order in enumerate(williams_orders(window)):
            for position, label in enumerate(order):
                latency_ms, host_wall_ms = timer.measure(label, mode)
                values[label].append(latency_ms)
                host_values[label].append(host_wall_ms)
                samples.append(
                    {
                        "order_index": order_index,
                        "position": position,
                        "label": label,
                        "latency_ms": latency_ms,
                        "host_wall_ms": host_wall_ms,
                    }
                )
        windows.append(
            {
                "window": window,
                "means_ms": {label: statistics.fmean(rows) for label, rows in values.items()},
                "host_wall_means_ms": {label: statistics.fmean(rows) for label, rows in host_values.items()},
                "ordered_samples": samples,
            }
        )
    return {
        "mode": mode,
        "warmup_windows": warmup_windows,
        "measured_windows": measured_windows,
        "windows": windows,
        "summary": summarize_windows(windows, bootstrap_samples, seed),
    }


def graph_correctness_gate(suite: CaseSuite, captured: dict[str, CapturedGraph], replays: int = 2) -> dict:
    poison_public_outputs(suite, "baseline_a")
    reference_values = suite.calls["baseline_a"](None)
    torch.cuda.synchronize()
    reference = tuple(value.detach().clone() for value in reference_values)
    dsink_reference = contract_dsink_reference(suite.inputs)
    variants = {}
    for label in LABELS:
        checks = []
        for _ in range(replays):
            for output in captured[label].outputs:
                output.fill_(float("nan"))
            captured[label].graph.replay()
            torch.cuda.synchronize()
            comparison = compare_outputs(captured[label].outputs, reference)
            dsink_check = output_metrics(captured[label].outputs[2], dsink_reference)
            comparison["dsink_vs_fp32_contract"] = dsink_check
            comparison["passed"] = comparison["passed"] and dsink_check["passed"]
            checks.append(comparison)
        variants[label] = {"passed": all(check["passed"] for check in checks), "replays": checks}
    return {"passed": all(row["passed"] for row in variants.values()), "replay_count": replays, "variants": variants}


def parse_csv_rows(output: str, expected_columns: int, description: str) -> list[list[str]]:
    rows = [[field.strip() for field in row] for row in csv.reader(output.splitlines()) if row]
    malformed = [row for row in rows if len(row) != expected_columns]
    if malformed:
        raise RuntimeError(f"malformed {description} from nvidia-smi: {malformed}")
    return rows


def gpu_identity() -> dict:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"fair benchmark requires exactly one CUDA-visible GPU, found {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    raw_uuid = properties.uuid
    selected_uuid = raw_uuid.decode("ascii") if isinstance(raw_uuid, bytes) else str(raw_uuid)
    selected_uuid = selected_uuid.strip()
    if not selected_uuid.startswith("GPU-"):
        selected_uuid = f"GPU-{selected_uuid}"
    rows = parse_csv_rows(
        command_output(
            [
                "nvidia-smi",
                f"--id={selected_uuid}",
                "--query-gpu=index,name,uuid,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            required=True,
        ),
        4,
        "GPU identity",
    )
    if len(rows) != 1:
        raise RuntimeError(f"selected CUDA GPU UUID resolved to {len(rows)} nvidia-smi rows: {rows}")
    index, name, uuid, mig_mode = rows[0]
    if uuid != selected_uuid:
        raise RuntimeError(f"torch selected UUID {selected_uuid}, but nvidia-smi returned {uuid}")
    if "B200" not in name:
        raise RuntimeError(f"fair benchmark requires B200, found {name}")
    if mig_mode.lower() != "disabled":
        raise RuntimeError(f"fair benchmark requires MIG disabled, found {mig_mode}")
    return {"index": int(index), "name": name, "uuid": uuid, "mig_mode": mig_mode}


def gpu_state(uuid: str) -> dict:
    state_rows = parse_csv_rows(
        command_output(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            required=True,
        ),
        10,
        "GPU state",
    )
    if len(state_rows) != 1:
        raise RuntimeError(f"UUID-scoped GPU query returned {len(state_rows)} rows for {uuid}: {state_rows}")
    fields = (
        "timestamp",
        "name",
        "uuid",
        "driver_version",
        "pstate",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
        "temperature_c",
        "mig_mode",
    )
    state = dict(zip(fields, state_rows[0], strict=True))
    process_rows = parse_csv_rows(
        command_output(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            required=True,
        ),
        4,
        "compute process list",
    )
    processes = [{"gpu_uuid": row[0], "pid": int(row[1]), "process_name": row[2], "used_memory_mib": row[3]} for row in process_rows]
    if state["uuid"] != uuid or any(process["gpu_uuid"] != uuid for process in processes):
        raise RuntimeError(f"nvidia-smi UUID scope mismatch for {uuid}: state={state}, processes={processes}")
    return {"state": state, "compute_processes": processes}


def self_pid_aliases() -> list[int]:
    aliases = {os.getpid()}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            aliases.update(int(value) for value in line.split()[1:])
            break
    return sorted(aliases)


def validate_gpu_state(snapshot: dict, uuid: str, allowed_pids: list[int]) -> None:
    errors = []
    state = snapshot["state"]
    if state["uuid"] != uuid:
        errors.append(f"selected UUID changed from {uuid} to {state['uuid']}")
    if "B200" not in state["name"]:
        errors.append(f"expected B200, found {state['name']}")
    if state["mig_mode"].lower() != "disabled":
        errors.append(f"MIG must be disabled, found {state['mig_mode']}")
    foreign = [process for process in snapshot["compute_processes"] if process["pid"] not in allowed_pids]
    if foreign:
        errors.append(f"foreign compute processes found on selected GPU: {foreign}")
    if errors:
        raise RuntimeError("invalid exclusive-GPU state:\n- " + "\n- ".join(errors))


def collect_environment(repo_root: Path, shared_closure: dict, import_origins: dict[str, str]) -> dict:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    cutlass_backend = cutlass_backend_audit()
    identity = gpu_identity()
    pid_aliases = self_pid_aliases()
    state_before = gpu_state(identity["uuid"])
    validate_gpu_state(state_before, identity["uuid"], pid_aliases)
    candidate_path = repo_root / INTERFACE_PATH
    candidate_kernel = repo_root / "python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_h128_2cta.py"
    return {
        "hostname": socket.gethostname(),
        "gpu": {
            "identity": identity,
            "self_pid_aliases": pid_aliases,
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "l2_cache_bytes": int(getattr(properties, "L2_cache_size", 0)),
            "state_before": state_before,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "cutlass_backend": cutlass_backend,
            "cuda_python": importlib.metadata.version("cuda-python"),
            "nvcc": command_output(["nvcc", "--version"], required=True),
        },
        "repository": {
            "root": str(repo_root),
            "head": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], required=True),
            "baseline": BASELINE_COMMIT,
            "status": command_output(["git", "-C", str(repo_root), "status", "--porcelain"], required=True),
            "candidate_interface_sha256": sha256_bytes(candidate_path.read_bytes()),
            "candidate_kernel_sha256": sha256_bytes(candidate_kernel.read_bytes()),
            "baseline_interface_sha256": sha256_bytes(git_blob(repo_root, BASELINE_COMMIT, INTERFACE_PATH)),
            "shared_closure": shared_closure,
            "import_origins": import_origins,
        },
    }


def validate_environment(environment: dict) -> None:
    errors = []
    gpu = environment["gpu"]
    software = environment["software"]
    repository = environment["repository"]
    if tuple(gpu["compute_capability"]) != (10, 0) or "B200" not in gpu["name"]:
        errors.append(f"expected B200/SM100, found {gpu['name']} {gpu['compute_capability']}")
    if gpu["multiprocessor_count"] != 148:
        errors.append(f"expected full 148-SM B200, found {gpu['multiprocessor_count']} SMs")
    if gpu["l2_cache_bytes"] <= 0:
        errors.append("CUDA runtime did not report a positive L2 cache size")
    if gpu["identity"]["mig_mode"].lower() != "disabled":
        errors.append(f"expected MIG disabled, found {gpu['identity']['mig_mode']}")
    if software["cutlass_dsl"] != EXPECTED_CUTE_DSL:
        errors.append(f"expected CuTe DSL {EXPECTED_CUTE_DSL}, found {software['cutlass_dsl']}")
    if not software["cutlass_backend"]["verified"]:
        errors.append("active CuTe DSL IR/runtime are not the pinned libs-cu13 4.5.2 binaries")
    if repository["status"]:
        errors.append("benchmark checkout must be clean")
    if errors:
        raise RuntimeError("invalid fair-benchmark environment:\n- " + "\n- ".join(errors))


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=parse_ints, default=[4096, 8192])
    parser.add_argument("--topks", type=parse_ints, default=[128, 512, 1024, 2048])
    parser.add_argument("--length-modes", type=parse_strings, default=["none", "full"], choices=None)
    parser.add_argument("--modes", type=parse_strings, default=["eager_hot", "eager_cold_l2", "graph50"])
    parser.add_argument("--graph-calls", type=int, default=50)
    parser.add_argument("--warmup-windows", type=int, default=3)
    parser.add_argument("--measured-windows", type=int, default=20)
    parser.add_argument("--correctness-calls", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--cold-l2-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=94401)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_modes = {"eager_hot", "eager_cold_l2", "graph50"}
    for name in ("seqlens", "topks", "length_modes", "modes"):
        values = getattr(args, name)
        if not values:
            raise ValueError(f"--{name.replace('_', '-')} must not be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"--{name.replace('_', '-')} must not contain duplicates: {values}")
    if not set(args.modes) <= valid_modes:
        raise ValueError(f"unsupported modes: {set(args.modes) - valid_modes}")
    if not set(args.length_modes) <= {"none", "full"}:
        raise ValueError(f"unsupported length modes: {args.length_modes}")
    if min(args.graph_calls, args.measured_windows, args.correctness_calls, args.bootstrap_samples) <= 0:
        raise ValueError("graph calls, windows, correctness calls, and bootstrap samples must be positive")
    if args.warmup_windows < 0:
        raise ValueError("warmup windows must be nonnegative")
    if min(args.seqlens + args.topks) <= 0 or max(args.topks) > min(args.seqlens):
        raise ValueError(f"all sequence lengths and top-k values must be positive with topk <= seqlen: {args.seqlens}, {args.topks}")
    if "graph50" in args.modes and args.graph_calls != 50:
        raise ValueError("graph50 mode requires exactly --graph-calls 50")

    repo_root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    if args.output == repo_root or repo_root in args.output.parents:
        raise ValueError("benchmark output must be outside the git checkout")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {args.output}")
    shared_closure = audit_shared_closure(repo_root)
    candidate = importlib.import_module(CANDIDATE_INTERFACE_MODULE)
    import_origins = audit_import_origins(repo_root, candidate)
    baseline = load_baseline_interface(repo_root)
    if baseline.flash_attn_bwd_sm100.compile_cache is candidate.flash_attn_bwd_sm100.compile_cache:
        raise RuntimeError("baseline and candidate unexpectedly share one compile cache")
    if baseline._select_sm100_backend(128, 512) != ("generic_m64", 64):
        raise RuntimeError("baseline interface does not route H128/D512 to generic_m64")
    candidate_route = candidate._select_sm100_backend(
        128,
        512,
        head_dim_v=512,
        dtype=torch.bfloat16,
        max_topk=512,
        device_capability=(10, 0),
    )
    if candidate_route != ("h128_2cta_m64", 64):
        raise RuntimeError(f"candidate route mismatch: {candidate_route}")

    environment = collect_environment(repo_root, shared_closure, import_origins)
    validate_environment(environment)
    if "eager_cold_l2" in args.modes and args.cold_l2_bytes < 2 * environment["gpu"]["l2_cache_bytes"]:
        raise ValueError(f"cold-L2 buffer must be at least 2x L2: {args.cold_l2_bytes} < {2 * environment['gpu']['l2_cache_bytes']}")
    cases = []
    case_index = 0
    for pair_index, (seqlen, topk) in enumerate((s, k) for s in args.seqlens for k in args.topks):
        input_seed = args.seed + pair_index
        for length_mode in args.length_modes:
            route = candidate._select_sm100_backend(
                128,
                512,
                head_dim_v=512,
                dtype=torch.bfloat16,
                max_topk=topk,
                device_capability=(10, 0),
            )
            if route != ("h128_2cta_m64", 64):
                raise RuntimeError(f"candidate unexpectedly falls back for topk={topk}: {route}")
            suite = make_case(
                seqlen=seqlen,
                topk=topk,
                has_topk_length=length_mode == "full",
                seed=input_seed,
                baseline=baseline,
                candidate=candidate,
            )
            for label in LABELS:
                suite.calls[label](None)
            torch.cuda.synchronize()
            candidate_cache_keys = list(candidate.flash_attn_bwd_sm100.compile_cache)
            baseline_cache_keys = list(baseline.flash_attn_bwd_sm100.compile_cache)
            has_lengths = length_mode == "full"
            if not any(len(key) == 8 and key[0] == "h128_2cta_m64" and key[-2:] == (topk, has_lengths) for key in candidate_cache_keys):
                raise RuntimeError(f"candidate cache did not record the two-CTA route for topk={topk}, mode={length_mode}")
            if not any(len(key) == 7 and key[-2:] == (topk, has_lengths) for key in baseline_cache_keys):
                raise RuntimeError(f"baseline cache did not record generic_m64 for topk={topk}, mode={length_mode}")
            direct = direct_correctness_gate(suite, args.correctness_calls)
            if not direct["passed"]:
                raise RuntimeError(f"direct correctness failed for {suite.metadata}")

            captured = {label: capture_graph(suite.calls[label], args.graph_calls) for label in LABELS}
            if len({repr(item.pool_token) for item in captured.values()}) != len(LABELS):
                raise RuntimeError("treatment CUDA graphs unexpectedly share a graph pool")
            graph_before = graph_correctness_gate(suite, captured)
            if not graph_before["passed"]:
                raise RuntimeError(f"pre-timing graph correctness failed for {suite.metadata}")

            selected_uuid = environment["gpu"]["identity"]["uuid"]
            allowed_pids = environment["gpu"]["self_pid_aliases"]
            state_before_timing = gpu_state(selected_uuid)
            validate_gpu_state(state_before_timing, selected_uuid, allowed_pids)
            modes = {}
            for mode_index, mode in enumerate(args.modes):
                modes[mode] = measure_mode(
                    suite,
                    captured,
                    mode=mode,
                    warmup_windows=args.warmup_windows,
                    measured_windows=args.measured_windows,
                    cold_l2_bytes=args.cold_l2_bytes,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + case_index * 17 + mode_index,
                )
            graph_after = graph_correctness_gate(suite, captured)
            if not graph_after["passed"]:
                raise RuntimeError(f"post-timing graph correctness failed for {suite.metadata}")
            state_after_timing = gpu_state(selected_uuid)
            validate_gpu_state(state_after_timing, selected_uuid, allowed_pids)
            cases.append(
                {
                    "case": suite.metadata,
                    "correctness": {"direct": direct, "graph_before_timing": graph_before, "graph_after_timing": graph_after},
                    "gpu_state_before_timing": state_before_timing,
                    "gpu_state_after_timing": state_after_timing,
                    "modes": modes,
                }
            )
            print(
                json.dumps(
                    {
                        "case": suite.metadata,
                        "speedup": {mode: row["summary"]["speedup"] for mode, row in modes.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            case_index += 1

    aggregate = {}
    for mode_index, mode in enumerate(args.modes):
        matrix = np.asarray([case["modes"][mode]["summary"]["log_speedup_by_window"] for case in cases], dtype=np.float64)
        host_matrix = np.asarray([case["modes"][mode]["summary"]["host_wall"]["log_speedup_by_window"] for case in cases], dtype=np.float64)
        aggregate[mode] = {
            **aggregate_speedup(matrix, args.bootstrap_samples, args.seed + 1000 + mode_index),
            "host_wall": aggregate_speedup(host_matrix, args.bootstrap_samples, args.seed + 2000 + mode_index),
            "case_count": len(cases),
            "duplicate_drift_within_0p5pct": all(case["modes"][mode]["summary"]["duplicate_drift_within_0p5pct"] for case in cases),
        }

    drift_ok = all(row["duplicate_drift_within_0p5pct"] for row in aggregate.values())
    state_after = gpu_state(environment["gpu"]["identity"]["uuid"])
    validate_gpu_state(state_after, environment["gpu"]["identity"]["uuid"], environment["gpu"]["self_pid_aliases"])
    environment["gpu"]["state_after"] = state_after
    status_after = command_output(["git", "-C", str(repo_root), "status", "--porcelain"], required=True)
    environment["repository"]["status_after"] = status_after
    if status_after:
        raise RuntimeError("benchmark checkout became dirty during the run")
    canonical_protocol = (
        args.seqlens == [4096, 8192]
        and args.topks == [128, 512, 1024, 2048]
        and args.length_modes == ["none", "full"]
        and args.modes == ["eager_hot", "eager_cold_l2", "graph50"]
        and args.graph_calls == 50
        and args.warmup_windows == 3
        and args.measured_windows == 20
        and args.correctness_calls == 50
        and args.bootstrap_samples == 100_000
    )
    result = {
        "schema_version": 2,
        "status": "complete" if drift_ok else "invalid_duplicate_drift",
        "canonical_protocol": canonical_protocol,
        "performance_claim_eligible": canonical_protocol and drift_ok,
        "protocol": {
            "labels": LABELS,
            "baseline": f"upstream interface at {BASELINE_COMMIT}",
            "candidate": environment["repository"]["head"],
            "topology": "candidate core grid.x=2*S_q, cluster=(2,1,1), CG2; public call also includes FP32-dKV-to-BF16 helper",
            "orders": "four-treatment Williams rows plus exact reverses",
            "seed": args.seed,
            "seqlens": args.seqlens,
            "topks": args.topks,
            "length_modes": args.length_modes,
            "input_generation": "canonical seeded torch.rand(Sq,Skv).argsort per repository benchmark; same tensors for none/full length pair",
            "graph_calls": args.graph_calls,
            "warmup_windows": args.warmup_windows,
            "measured_windows": args.measured_windows,
            "correctness_calls": args.correctness_calls,
            "bootstrap_samples": args.bootstrap_samples,
            "modes": args.modes,
            "cold_l2_bytes": args.cold_l2_bytes,
            "device_l2_bytes": environment["gpu"]["l2_cache_bytes"],
            "correctness_tolerances": {
                "elementwise_atol": ELEMENTWISE_ATOL,
                "elementwise_rtol": ELEMENTWISE_RTOL,
                "rms_atol": RMS_ATOL,
                "rms_rtol": RMS_RTOL,
            },
            "timed_scope": {
                "eager": "public DSA wrapper launch through completion; route selection and CUstream construction excluded",
                "graph50": "one replay of a graph containing exactly 50 public calls, divided by 50",
                "cold_l2": "random noncompressible buffer sum and synchronization excluded from the measured interval",
            },
            "argv": sys.argv,
        },
        "environment": environment,
        "cases": cases,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    if temporary_output.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary evidence: {temporary_output}")
    temporary_output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary_output, args.output)
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, indent=2, allow_nan=False), flush=True)
    if not drift_ok:
        raise SystemExit("duplicate-treatment drift exceeded the predeclared +/-0.5% validity bound")


if __name__ == "__main__":
    main()
