"""baseref: the canonical 1-CTA baseline dressed in the candidate interface.

Purpose (campaign ledger §10.3): let the one-click hold pipeline capture a
BASELINE IKET trace through its normal candidate slot.  The pipeline's
correctness gate (candidate vs baseline) passes trivially, the perf leg
reads ~1.00x, and the trace leg produces the lean baseline twin capture --
all without the payload/manager.lock dispatch path.

This release-side shim wraps dsa_bwd_sm100_baseline and swallows the three
candidate-only trace arguments.
"""
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .dsa_bwd_sm100_baseline import (
    FlashAttentionDSABackwardSm100 as _CanonicalBaseline,
)


class FlashAttentionDSABackwardSm100TwoCTAV2(_CanonicalBaseline):
    """Candidate-interface adapter over the canonical baseline."""

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        trace_buffer: Optional[cute.Tensor],
        trace_token_idx: Int32,
        trace_batch_idx: Int32,
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        _CanonicalBaseline.__call__(
            self,
            problem_shape,
            mQ,
            mKV,
            mOut,
            mdO,
            mLSE,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            mdQ,
            mdKV,
            mdSink,
            workspace_LSE_OdO,
            workspace_dKV,
            softmax_scale,
            stream,
        )
