#!/usr/bin/env python3
"""Candidate-only single-topk run for NCU profiling (E3-NCU attribution).

Runs ONLY the rubin_1 candidate leg (no baseline leg) so an ncu
--launch-skip/--launch-count window lands on a clean, repeating launch
sequence: per iteration ~3 torch fills + the DSA kernel sequence, of
which the main bwd kernel dominates by duration.  The active profile
(compat vs e3pad) is selected by the DSA_RUBIN1_* env, exactly as in
run_e3pair_gr100.sh.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "sweep_topk_2cta", HERE / "sweep_topk_2cta.py"
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

import torch  # noqa: E402  (after sweep import sets sys.path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=8)
    args0 = p.parse_args()

    args = SimpleNamespace(
        impl="rubin_1",
        class_name="FlashAttentionDSABackwardSm100TwoCTAV2",
        warmup=args0.warmup,
        repeat=args0.repeat,
    )
    case = sweep.build_case(4096, args0.topk, 128, 512)
    ms, dq, dkv = sweep.candidate_leg(case, args0.topk, args)
    assert torch.isfinite(dq).all(), "non-finite dq"
    assert torch.isfinite(dkv).all(), "non-finite dkv"
    print(f"E3NCU_PROBE topk={args0.topk} candidate_ms={ms:.4f}")
    print("E3NCU_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
