"""Shared cute.compile option helpers.

The cute DSL compiler accepts ``--gpu-arch <sm_XXX>`` to lock SASS to a
specific architecture. Without it, the compiler falls back to the device
arch reported by ``torch.cuda.get_device_capability()`` via the cute DSL's
internal map (see ``cutlass/base_dsl/runtime/cuda.py``). That map currently
hardcodes ``(10, 0) → "sm_100a"`` (B200) but treats unknown caps as
``"sm_<major><minor>"`` *without* the architecture-specific ``a`` suffix —
which silently drops sm_X-a-only features (TMA bulk, tcgen05, etc.) on
B300 and beyond.

So we always pass an explicit ``--gpu-arch`` chosen at runtime from the
device capability. ``compile_options(extra)`` is the single entry point;
DSA ``cute.compile`` call sites should route through it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch

# (compute_capability) → cute DSL --gpu-arch flag value.
# H100, B200/B300, and Rubin require architecture-specific variants because
# the kernels use TMA / tcgen05 instructions that are only guaranteed to lower
# correctly under the matching SASS gencode.
_ARCH_MAP = {
    (9, 0): "sm_90a",  # Hopper H100
    (10, 0): "sm_100a",  # Blackwell B200
    (10, 3): "sm_103a",  # Blackwell Ultra B300
    (10, 7): "sm_107a",  # Rubin GR100 native architecture target
}


@lru_cache(maxsize=None)
def _gpu_arch_flag_for_capability(capability: Tuple[int, int]) -> str:
    arch = _ARCH_MAP.get(capability)
    if arch is None:
        raise RuntimeError(
            f"Unsupported GPU compute capability {capability} for DSA CuTe kernels. " "Add it to deepseek_sparse_attention/utils/compiler.py::_ARCH_MAP."
        )
    return arch


def gpu_arch_flag(device: Optional[object] = None, capability: Optional[Tuple[int, int]] = None) -> str:
    """Return the architecture flag for ``device`` or an explicit capability.

    Capability-to-flag conversion is cached, while device resolution happens
    on every call.  This avoids reusing the first active device's architecture
    in a process that switches devices or contains mixed GPU generations.
    """
    if capability is None:
        if not torch.cuda.is_available():
            raise RuntimeError("cute.compile requires CUDA; no GPU available")
        capability = torch.cuda.get_device_capability(device)
    normalized_capability = tuple(int(value) for value in capability)
    if len(normalized_capability) != 2:
        raise ValueError(f"Invalid GPU compute capability: {capability}")
    return _gpu_arch_flag_for_capability(normalized_capability)


def compile_options(
    extra: str = "",
    *,
    device: Optional[object] = None,
    capability: Optional[Tuple[int, int]] = None,
) -> str:
    """Build the ``options=`` string for ``cute.compile``.

    Always emits ``--enable-tvm-ffi`` and a runtime-chosen ``--gpu-arch``;
    pass any kernel-specific knobs (``--opt-level 3`` etc.) via ``extra``.

    Example:
        cute.compile(..., options=compile_options("--opt-level 3"))
    """
    parts = ["--enable-tvm-ffi", f"--gpu-arch {gpu_arch_flag(device=device, capability=capability)}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)
