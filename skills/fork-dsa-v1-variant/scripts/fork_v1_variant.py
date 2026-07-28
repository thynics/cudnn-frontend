#!/usr/bin/env python3
"""Create a byte-identical, user-named copy of the current DSA v1 module."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys


KERNEL_DIR = Path(
    "python/cudnn/deepseek_sparse_attention/"
    "sparse_attention_backward"
)
SOURCE_NAME = "dsa_bwd_sm100_2cta_v1.py"
VARIANT_RE = re.compile(r"^v[a-z0-9_]{1,31}$")
RESERVED_VARIANTS = {"v0", "v1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the current DSA v1 module to a new, isolated variant module."
        )
    )
    parser.add_argument(
        "variant",
        help="variant token, for example vt2a or vroute_k",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    return Path(__file__).resolve().parents[3]


def main() -> int:
    args = parse_args()
    requested_variant = args.variant
    variant = requested_variant.lower()
    if not VARIANT_RE.fullmatch(variant):
        print(
            "ERROR: variant must start with v and match "
            "[a-z0-9_]{1,31} after lowercase normalization",
            file=sys.stderr,
        )
        return 2
    if variant in RESERVED_VARIANTS:
        print(
            f"ERROR: {variant} is canonical and cannot be a variant name",
            file=sys.stderr,
        )
        return 2

    repo_root = repository_root(args.repo_root)
    source = repo_root / KERNEL_DIR / SOURCE_NAME
    target = (
        repo_root
        / KERNEL_DIR
        / f"dsa_bwd_sm100_2cta_{variant}.py"
    )
    if not source.is_file():
        print(f"ERROR: current v1 source is missing: {source}", file=sys.stderr)
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                target_stream.write(chunk)
        os.chmod(target, source.stat().st_mode)
    except FileExistsError:
        print(f"ERROR: refusing to overwrite existing variant: {target}", file=sys.stderr)
        return 2
    except BaseException:
        if target.exists():
            target.unlink()
        raise

    source_sha = sha256_file(source)
    target_sha = sha256_file(target)
    if source_sha != target_sha:
        target.unlink()
        print("ERROR: copied variant did not preserve the v1 bytes", file=sys.stderr)
        return 1

    print(f"DSA_VARIANT_CREATED variant={variant}")
    if requested_variant != variant:
        print(
            "DSA_VARIANT_NORMALIZED "
            f"requested={requested_variant} variant={variant}"
        )
    print(f"DSA_VARIANT_SOURCE {source}")
    print(f"DSA_VARIANT_TARGET {target}")
    print(f"DSA_VARIANT_SHA256 {target_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
