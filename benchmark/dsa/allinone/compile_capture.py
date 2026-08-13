#!/usr/bin/env python3
"""Compile-only Stage-0 capture for a registered DSA 2-CTA implementation.

In-repo default for run_allinone.sh's DSA_STAGE0_CMD hook:

    python3 benchmark/dsa/allinone/compile_capture.py --impl v15 --out DIR

Builds dummy problem tensors (H128/D512/S4096/topk2048 defaults), drives
cute.compile on the implementation's __call__ with CUTE_DSL_KEEP=
ptx,cubin,sass, and arranges the artifacts in the layout
stage0_analyzer.py expects:

    <out>/logs/codegen/compile/<kernel>.{ptx,sm_100a.cubin,sass}
    <out>/logs/codegen/compile/artifact_manifest.json
    <out>/resource_usage.txt          (cuobjdump --dump-resource-usage)
    <out>/source_manifest.json        (impl path, sha256, git revision)

Compile-only: nothing is launched; a visible CUDA device is still
required for the DSL target/driver queries and torch allocations.
Needs cuobjdump on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_REL = "python/cudnn/deepseek_sparse_attention/sparse_attention_backward"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impl", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--class-name",
                   default="FlashAttentionDSABackwardSm100TwoCTAV2")
    p.add_argument("--seqlen-q", type=int, default=4096)
    p.add_argument("--seqlen-kv", type=int, default=4096)
    p.add_argument("--nheads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--head-dim-v", type=int, default=512)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--no-topk-length", action="store_true",
                   help="compile the mTopkLength=None specialization")
    p.add_argument(
        "--allow-cubin-only",
        action="store_true",
        help=(
            "accept a lineinfo CUBIN without an automatically emitted SASS "
            "file (needed by CUTLASS DSL 4.5; disassemble it separately)"
        ),
    )
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    dump_dir = out / "logs" / "codegen" / "compile"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # KEEP env must be in place before cutlass import.
    os.environ.setdefault("CUTE_DSL_KEEP", "ptx,cubin,sass")
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)

    sys.path.insert(0, str(REPO_ROOT / "python"))
    import importlib

    import torch
    import cutlass  # noqa: F401  (env consumed at import)
    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.utils.compiler import (
        compile_options,
    )
    from cudnn.deepseek_sparse_attention.utils.runtime import (
        resolve_stream,
    )
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
        to_cute_tensor,
    )

    impl_mod = importlib.import_module(
        "cudnn.deepseek_sparse_attention.sparse_attention_backward."
        f"dsa_bwd_sm100_2cta_{args.impl}"
    )
    impl_cls = getattr(impl_mod, args.class_name)

    device = "cuda"
    dtype = torch.bfloat16
    S_q, S_kv = args.seqlen_q, args.seqlen_kv
    H, D, Dv, topk = args.nheads, args.head_dim, args.head_dim_v, args.topk

    q = torch.randn(S_q, H, D, device=device, dtype=dtype) / 10
    kv = torch.randn(S_kv, D, device=device, dtype=dtype) / 10
    out_t = torch.randn(S_q, H, Dv, device=device, dtype=dtype) / 10
    dout = torch.randn(S_q, H, Dv, device=device, dtype=dtype) / 10
    lse = torch.randn(S_q, H, device=device, dtype=torch.float32)
    attn_sink = torch.linspace(-2.0, 2.0, H, device=device,
                               dtype=torch.float32)
    topk_idxs = (torch.rand(S_q, S_kv, device=device)
                 .argsort(dim=-1)[:, :topk].to(torch.int32).contiguous())
    topk_length = (None if args.no_topk_length else
                   torch.full((S_q,), topk, dtype=torch.int32,
                              device=device))
    dq = torch.empty_like(q)
    dkv = torch.zeros(S_kv, D, dtype=dtype, device=device)
    d_sink = torch.zeros_like(attn_sink)

    base_cls = impl_cls.__mro__[-2]  # FlashAttentionDSABackwardSm100
    acc = cutlass.Float32
    ws_lse_shape = impl_cls._get_workspace_size_LSE_OdO(
        S_q, D, H, 1, acc)
    ws_dkv_shape = impl_cls._get_workspace_size_dKV(S_kv, D, 1, acc)
    ws_lse = torch.zeros(*ws_lse_shape, dtype=torch.uint8, device=device)
    ws_dkv = torch.zeros(*ws_dkv_shape, dtype=torch.uint8, device=device)

    kernel_obj = impl_cls(
        head_dim=D, head_dim_v=Dv, block_tile=64, max_topk=topk)

    problem_shape = (S_q, S_kv, D, (H, 1))
    stream = resolve_stream(None)

    call_args = [
        problem_shape,
        to_cute_tensor(q, divisibility=D),
        to_cute_tensor(kv, divisibility=D),
        to_cute_tensor(out_t, divisibility=Dv),
        to_cute_tensor(dout, divisibility=Dv),
        to_cute_tensor(lse, assumed_align=4),
        to_cute_tensor(attn_sink),
        to_cute_tensor(topk_idxs),
        (None if topk_length is None else to_cute_tensor(topk_length)),
        to_cute_tensor(dq, divisibility=D),
        to_cute_tensor(dkv, divisibility=D),
        to_cute_tensor(d_sink),
        to_cute_tensor(ws_lse),
        to_cute_tensor(ws_dkv),
        None,  # trace_buffer
        0,     # trace_token_idx
        0,     # trace_batch_idx
        1.0 / (D ** 0.5),
        stream,
    ]
    if getattr(kernel_obj, "V15_L2X", False):
        ws_pds_shape = impl_cls._get_workspace_size_pds(S_q)
        ws_pds = torch.zeros(*ws_pds_shape, dtype=torch.uint8,
                             device=device)
        call_args.append(to_cute_tensor(ws_pds))

    print(f"COMPILE_CAPTURE compiling impl={args.impl} "
          f"class={args.class_name} keep={os.environ['CUTE_DSL_KEEP']}")
    cute.compile(kernel_obj, *call_args, options=compile_options())

    artifacts = []
    cubins = sorted(dump_dir.glob("*.cubin"))
    for f in sorted(dump_dir.iterdir()):
        if f.suffix in (".ptx", ".cubin", ".sass"):
            artifacts.append({
                "path": f.name,
                "bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            })
    if not args.allow_cubin_only and not any(a["path"].endswith(".sass") for a in artifacts):
        raise SystemExit("no .sass artifact was kept -- check "
                         "CUTE_DSL_KEEP handling in this DSL version")

    manifest = {
        "artifacts": artifacts,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cutlass_path": cutlass.__file__,
        "cutlass_version": getattr(cutlass, "__version__", "unknown"),
        "keep": os.environ["CUTE_DSL_KEEP"],
        "dump_dir": str(dump_dir),
    }
    (dump_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    # resource_usage.txt in cuobjdump format (the analyzer's parser
    # keys on the "Function kernel_cutlass_kernel_..." block).
    if not cubins:
        raise SystemExit("no cubin kept; cannot produce resource usage")
    res = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(cubins[0])],
        capture_output=True, text=True, check=True)
    (out / "resource_usage.txt").write_text(res.stdout)

    impl_path = REPO_ROOT / PKG_REL / f"dsa_bwd_sm100_2cta_{args.impl}.py"
    rev = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    (out / "source_manifest.json").write_text(json.dumps({
        "impl": args.impl,
        "class": args.class_name,
        "path": str(impl_path),
        "sha256": sha256_file(impl_path),
        "revision": rev,
        "base_class": base_cls.__name__,
        "flags": {k: v for k, v in os.environ.items()
                  if k.startswith("DSA_V")},
    }, indent=2))

    print(f"COMPILE_CAPTURE_OK {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
