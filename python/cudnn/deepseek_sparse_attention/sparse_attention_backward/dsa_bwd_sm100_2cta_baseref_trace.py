"""baseref trace twin: the lean IKET baseline twin in the candidate interface.

Same adapter as dsa_bwd_sm100_2cta_baseref.py but over baseline_trace (the
boundary-fixed, lean-filtered IKET twin of the canonical baseline).  The hold
pipeline's trace leg runs this at Sq=1 and yields the decision-grade baseline
segment table (campaign ledger §10.3).
"""
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .baseline_trace import (
    FlashAttentionDSABackwardSm100 as _BaselineTwin,
)


class FlashAttentionDSABackwardSm100TwoCTAV2(_BaselineTwin):
    """Candidate-interface adapter over the lean baseline IKET twin."""

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
        _BaselineTwin.__call__(
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
