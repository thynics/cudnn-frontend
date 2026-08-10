#!/usr/bin/env python3
"""E6a gate probe: is TMA tile::gather4 available in the remote DSL?

Checks the active cutlass wheel for CopyBulkTensor2DGather4G2SOp and the
gather4-aware make_tiled_tma_atom / tma_partition plumbing, plus the
cache_policy (L2 eviction hint) kwarg on TMA copy traits.  Pure import
introspection -- no GPU work.  Prints one JSON report.
"""

import inspect
import json
import sys

sys.path[:0] = ["python", "test/python", "."]

report = {}

import cutlass  # noqa: E402

report["cutlass_version"] = getattr(cutlass, "__version__", "unknown")

import cutlass.cute as cute  # noqa: E402
from cutlass.cute.nvgpu import cpasync  # noqa: E402

report["gather4_op"] = hasattr(cpasync, "CopyBulkTensor2DGather4G2SOp")

try:
    sig = inspect.signature(cpasync.make_tiled_tma_atom)
    report["make_tiled_tma_atom_params"] = sorted(sig.parameters)
    report["gather4_atom_maker"] = "gmem_coord_tensor" in sig.parameters
except (AttributeError, ValueError, TypeError) as exc:
    report["gather4_atom_maker"] = f"introspection failed: {exc}"

try:
    doc = inspect.getdoc(cpasync.tma_partition) or ""
    report["tma_partition_gather4_doc"] = "gather4" in doc
except AttributeError as exc:
    report["tma_partition_gather4_doc"] = f"missing: {exc}"

report["cache_eviction_enum"] = hasattr(cute, "CacheEvictionPriority")

try:
    src = inspect.getsource(cpasync)
    report["cache_policy_kwarg"] = "cache_policy" in src
except (OSError, TypeError):
    import cutlass.cute.nvgpu.cpasync.copy as _cp

    report["cache_policy_kwarg"] = "cache_policy" in inspect.getsource(_cp)

report["gate_gather4"] = bool(
    report["gather4_op"] and report.get("gather4_atom_maker") is True
)

print(json.dumps(report, indent=2))
