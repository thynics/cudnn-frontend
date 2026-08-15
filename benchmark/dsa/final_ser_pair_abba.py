#!/usr/bin/env python3
"""Decision-grade same-process ABBA comparison for two final_ser sources.

Both source files are loaded under distinct package module names, compiled
directly, and launched against the same tensors, outputs, and workspaces.  The
only timed region is the compiled program; accumulator resets are ordered
before the start event on the same CUDA stream.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO_ROOT / "python"),
    str(REPO_ROOT / "test" / "python"),
    str(REPO_ROOT),
]

import torch


BASELINE_ENVIRONMENT = {
    "DSA_BL_QDO_STAGE": "1",
    "DSA_BL_K_STAGE": "1",
    "DSA_BL_HALFK": "0",
    "DSA_BL_KSTAGE2": "0",
    "DSA_BL_OVPAD": "0",
}
PACKAGE = "cudnn.deepseek_sparse_attention.sparse_attention_backward"
CLASS_NAME = "FlashAttentionDSABackwardSm100TwoCTAV2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-source", required=True, type=Path)
    parser.add_argument("--right-source", required=True, type=Path)
    parser.add_argument("--left-label", default="h1")
    parser.add_argument("--right-label", default="h3")
    parser.add_argument("--left-expected-sha256", default=None)
    parser.add_argument("--right-expected-sha256", default=None)
    parser.add_argument("--left-reduce-pace-ns", type=int, default=None)
    parser.add_argument("--right-reduce-pace-ns", type=int, default=None)
    parser.add_argument("--left-reduce-dephase-ns", type=int, default=None)
    parser.add_argument("--right-reduce-dephase-ns", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-pairs", type=int, default=32)
    parser.add_argument("--paired-samples", type=int, default=48)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--nheads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=512)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_label(label: str) -> str:
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,31}", label):
        raise ValueError(f"invalid label: {label!r}")
    return label.lower()


def load_source(
    path: Path,
    label: str,
    expected_sha256: str | None,
) -> tuple[ModuleType, type[Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    source_sha256 = sha256(path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} source SHA mismatch: expected={expected_sha256} "
            f"actual={source_sha256} path={path}"
        )
    module_name = f"{PACKAGE}._pair_{label}_{source_sha256[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    impl_cls = getattr(module, CLASS_NAME)
    return module, impl_cls, source_sha256


def phase(name: str):
    class Phase:
        def __enter__(self):
            self.started = time.monotonic()
            print(f"FINAL_SER_PAIR_PHASE_BEGIN {name}", flush=True)

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                elapsed = time.monotonic() - self.started
                print(
                    f"FINAL_SER_PAIR_PHASE_END {name} duration_s={elapsed:.6f}",
                    flush=True,
                )
            return False

    return Phase()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def block_bootstrap_median_ci(
    paired_ratios: list[float],
    block_size: int,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if len(paired_ratios) % block_size:
        raise ValueError("paired sample count must be divisible by block size")
    blocks = [
        paired_ratios[index : index + block_size]
        for index in range(0, len(paired_ratios), block_size)
    ]
    generator = random.Random(seed)
    medians: list[float] = []
    for _ in range(resamples):
        sample = [
            value
            for _ in blocks
            for value in blocks[generator.randrange(len(blocks))]
        ]
        medians.append(statistics.median(sample))
    return {
        "block_size": block_size,
        "blocks": len(blocks),
        "resamples": resamples,
        "seed": seed,
        "lower_95": percentile(medians, 0.025),
        "upper_95": percentile(medians, 0.975),
    }


def main() -> int:
    args = parse_args()
    left_label = check_label(args.left_label)
    right_label = check_label(args.right_label)
    if left_label == right_label:
        raise ValueError("left and right labels must differ")
    if args.warmup_pairs < 1 or args.paired_samples < 8:
        raise ValueError("insufficient warmup or paired samples")
    if args.paired_samples % 8:
        raise ValueError("paired samples must be divisible by 8")
    if args.bootstrap_resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    if args.topk % 64:
        raise ValueError("topk must be a multiple of 64")

    os.environ["DSA_DEV_CANDIDATE_VARIANT"] = "v2native"
    os.environ.update(BASELINE_ENVIRONMENT)
    os.environ.pop("DSA_DEV_IKET", None)
    os.environ.pop("DKG_IKET_INSTRUMENTATION_METHOD", None)

    from benchmark.dsa import benchmark_dsa_sparse_attention_backward as bench
    import cutlass
    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
    from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
        to_cute_tensor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu = torch.cuda.get_device_name()
    capability = list(torch.cuda.get_device_capability())
    if "B200" not in gpu or capability != [10, 0]:
        raise RuntimeError(
            f"expected NVIDIA B200 sm_100, got {gpu} capability={capability}"
        )

    _, left_cls, left_sha256 = load_source(
        args.left_source, left_label, args.left_expected_sha256
    )
    _, right_cls, right_sha256 = load_source(
        args.right_source, right_label, args.right_expected_sha256
    )

    class_overrides = {
        left_label: {
            "REDUCE_PACE_NS": args.left_reduce_pace_ns,
            "REDUCE_DEPHASE_NS": args.left_reduce_dephase_ns,
        },
        right_label: {
            "REDUCE_PACE_NS": args.right_reduce_pace_ns,
            "REDUCE_DEPHASE_NS": args.right_reduce_dephase_ns,
        },
    }
    for label, impl_cls in ((left_label, left_cls), (right_label, right_cls)):
        for name, value in class_overrides[label].items():
            if value is None:
                continue
            if value < 0:
                raise ValueError(f"{label} {name} must be non-negative")
            setattr(impl_cls, name, value)

    seqlen = args.seqlen
    nheads = args.nheads
    head_dim = args.head_dim
    topk = args.topk
    torch.manual_seed(20260810)
    q, kv, sink, dout, indices, lengths = bench.make_inputs(
        seqlen,
        topk,
        seqlen,
        nheads,
        head_dim,
        head_dim,
        torch.bfloat16,
        use_attn_sink=True,
        use_topk_length=True,
    )
    scale = 1.0 / math.sqrt(head_dim)
    out, lse = bench.reference_forward(q, kv, sink, indices, scale, head_dim)

    def workspace_shapes(impl_cls: type[Any]):
        accumulator = cutlass.Float32
        return (
            tuple(
                int(value)
                for value in impl_cls._get_workspace_size_LSE_OdO(
                    seqlen, head_dim, nheads, 1, accumulator
                )
            ),
            tuple(
                int(value)
                for value in impl_cls._get_workspace_size_dKV(
                    seqlen, head_dim, 1, accumulator
                )
            ),
        )

    left_shapes = workspace_shapes(left_cls)
    right_shapes = workspace_shapes(right_cls)
    if left_shapes != right_shapes:
        raise RuntimeError(
            f"workspace mismatch: {left_label}={left_shapes} "
            f"{right_label}={right_shapes}"
        )

    buffers = {
        "dq": torch.empty_like(q),
        "dkv": torch.zeros_like(kv),
        "d_sink": torch.zeros_like(sink),
        "workspace_lse_odo": torch.zeros(
            *left_shapes[0], dtype=torch.uint8, device="cuda"
        ),
        "workspace_dkv": torch.zeros(
            *left_shapes[1], dtype=torch.uint8, device="cuda"
        ),
    }
    problem_shape = (seqlen, seqlen, head_dim, (nheads, 1))
    stream = resolve_stream(None)

    def build_direct_runner(impl_cls: type[Any]):
        kernel = impl_cls(
            head_dim=head_dim,
            head_dim_v=head_dim,
            block_tile=64,
            max_topk=topk,
        )
        prototypes = [
            to_cute_tensor(q, divisibility=head_dim),
            to_cute_tensor(kv, divisibility=head_dim),
            to_cute_tensor(out, divisibility=head_dim),
            to_cute_tensor(dout, divisibility=head_dim),
            to_cute_tensor(lse, assumed_align=4),
            to_cute_tensor(sink),
            to_cute_tensor(indices),
            to_cute_tensor(lengths),
            to_cute_tensor(buffers["dq"], divisibility=head_dim),
            to_cute_tensor(buffers["dkv"], divisibility=head_dim),
            to_cute_tensor(buffers["d_sink"]),
            to_cute_tensor(buffers["workspace_lse_odo"]),
            to_cute_tensor(buffers["workspace_dkv"]),
            None,
            0,
            0,
            scale,
            stream,
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
            None,
            0,
            0,
            scale,
            stream,
        ]
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

    with phase(f"compile:{left_label}"):
        left_launch = build_direct_runner(left_cls)
    with phase(f"compile:{right_label}"):
        right_launch = build_direct_runner(right_cls)
    runners = {left_label: left_launch, right_label: right_launch}

    def reset_accumulators():
        buffers["dkv"].zero_()
        buffers["workspace_dkv"].zero_()
        buffers["d_sink"].zero_()

    reset_accumulators()
    left_launch()
    torch.cuda.synchronize()
    left_outputs = {
        name: buffers[name].clone() for name in ("dq", "dkv", "d_sink")
    }
    reset_accumulators()
    right_launch()
    torch.cuda.synchronize()
    crosscheck = {
        f"max_abs_diff_{name}": float(
            (left_outputs[name] - buffers[name]).abs().max()
        )
        for name in left_outputs
    }
    crosscheck["all_outputs_finite"] = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            *left_outputs.values(),
            buffers["dq"],
            buffers["dkv"],
            buffers["d_sink"],
        )
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
        raise RuntimeError(f"exact-shape crosscheck failed: {crosscheck}")

    def pair_order(index: int) -> tuple[str, str]:
        return (
            (left_label, right_label),
            (right_label, left_label),
            (right_label, left_label),
            (left_label, right_label),
        )[index % 4]

    with phase("warmup"):
        for index in range(args.warmup_pairs):
            for label in pair_order(index):
                reset_accumulators()
                runners[label]()
        torch.cuda.synchronize()

    def time_program_only(launch) -> float:
        reset_accumulators()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    samples: dict[str, list[float]] = {left_label: [], right_label: []}
    with phase("timed:fair_direct_program_only"):
        for index in range(args.paired_samples):
            for label in pair_order(index):
                samples[label].append(time_program_only(runners[label]))

    paired_ratios = [
        right_ms / left_ms
        for left_ms, right_ms in zip(
            samples[left_label], samples[right_label], strict=True
        )
    ]
    half = args.paired_samples // 2
    right_over_left = statistics.median(paired_ratios)
    first_half_ratio = statistics.median(paired_ratios[:half])
    second_half_ratio = statistics.median(paired_ratios[half:])
    bootstrap = block_bootstrap_median_ci(
        paired_ratios,
        block_size=4,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    accept_right = (
        first_half_ratio <= 0.995
        and second_half_ratio <= 0.995
        and float(bootstrap["upper_95"]) < 1.0
    )

    payload = {
        "status": "pass",
        "benchmark": "final_ser_same_process_direct_abba_v1",
        "primary_metric": f"{right_label}_over_{left_label}",
        "decision": "accept_right" if accept_right else "retain_left",
        "acceptance_contract": {
            "first_half_right_over_left_max": 0.995,
            "second_half_right_over_left_max": 0.995,
            "block_bootstrap_upper_95_max_exclusive": 1.0,
        },
        "sources": {
            left_label: {
                "path": str(args.left_source.resolve()),
                "sha256": left_sha256,
            },
            right_label: {
                "path": str(args.right_source.resolve()),
                "sha256": right_sha256,
            },
        },
        "class_overrides": class_overrides,
        "fairness_contract": {
            "same_process": True,
            "same_inputs": True,
            "same_output_and_workspace_addresses": True,
            "same_reset_set": ["dkv", "workspace_dkv", "d_sink"],
            "resets_excluded_from_cuda_event": True,
            "compile_and_allocation_outside_timing": True,
            "order": "ABBA-balanced paired samples",
            "baseline_environment": BASELINE_ENVIRONMENT,
        },
        "gpu": gpu,
        "compute_capability": capability,
        "shape": {
            "seqlen_q": seqlen,
            "seqlen_kv": seqlen,
            "topk": topk,
            "nheads": nheads,
            "head_dim_qk": head_dim,
            "head_dim_v": head_dim,
            "dtype": "bfloat16",
        },
        "warmup_pairs": args.warmup_pairs,
        "paired_samples": args.paired_samples,
        "latency_ms": {
            left_label: statistics.median(samples[left_label]),
            right_label: statistics.median(samples[right_label]),
        },
        "right_over_left": right_over_left,
        "first_half_right_over_left": first_half_ratio,
        "second_half_right_over_left": second_half_ratio,
        "paired_ratios": paired_ratios,
        "block_bootstrap_median_ci": bootstrap,
        "raw_ms": samples,
        "crosscheck": crosscheck,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(
        f"FINAL_SER_PAIR_RESULT decision={payload['decision']} "
        f"ratio={right_over_left:.6f} "
        f"ci95=[{bootstrap['lower_95']:.6f},{bootstrap['upper_95']:.6f}] "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
