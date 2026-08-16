import pytest
import torch

from cudnn.deepseek_sparse_attention.sparse_attention_backward._dispatch import (
    SM100_IMPLEMENTATION,
    SM107_IMPLEMENTATION,
    SM90_IMPLEMENTATION,
    select_sparse_attention_backward_implementation,
)
from cudnn.deepseek_sparse_attention.utils import compiler

pytestmark = pytest.mark.L0


def _select(
    *,
    capability=(10, 7),
    dtype=torch.bfloat16,
    heads=128,
    head_dim=512,
    seqlen_kv=4096,
    topk=512,
    block_tile=64,
    device_type="cuda",
    all_inputs_same_device=True,
):
    seqlen_q = 7
    q_shape = (seqlen_q, heads, head_dim)
    return select_sparse_attention_backward_implementation(
        capability=capability,
        device_type=device_type,
        all_inputs_same_device=all_inputs_same_device,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        dout_dtype=dtype,
        q_shape=q_shape,
        kv_shape=(seqlen_kv, head_dim),
        out_shape=q_shape,
        dout_shape=q_shape,
        lse_shape=(seqlen_q, heads),
        attn_sink_shape=(heads,),
        topk_idxs_shape=(seqlen_q, topk),
        block_tile=block_tile,
    )


@pytest.mark.parametrize("topk", (512, 1024, 2048))
def test_sm107_dispatches_validated_rubin_envelope(topk):
    assert _select(topk=topk) == SM107_IMPLEMENTATION


@pytest.mark.parametrize(
    "overrides",
    (
        {"topk": 128},
        {"topk": 256},
        {"dtype": torch.float16},
        {"heads": 64},
        {"head_dim": 576},
        {"seqlen_kv": 8192},
        {"block_tile": 128},
        {"capability": (10, 0)},
        {"capability": (10, 3)},
        {"device_type": "cpu"},
        {"all_inputs_same_device": False},
    ),
)
def test_nonvalidated_shapes_fall_back_to_sm100(overrides):
    assert _select(**overrides) == SM100_IMPLEMENTATION


def test_sm90_dispatch_is_unchanged():
    assert _select(capability=(9, 0)) == SM90_IMPLEMENTATION


def test_architecture_flag_resolution_is_device_scoped(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    capabilities = {0: (10, 0), 1: (10, 7)}
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: capabilities[int(device)],
    )

    assert compiler.gpu_arch_flag(device=0) == "sm_100a"
    assert compiler.gpu_arch_flag(device=1) == "sm_107a"
    assert compiler.compile_options(device=1) == "--enable-tvm-ffi --gpu-arch sm_107a"


def test_explicit_capability_does_not_query_active_device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_: pytest.fail("active device must not be queried"),
    )
    assert compiler.gpu_arch_flag(capability=(10, 3)) == "sm_103a"
