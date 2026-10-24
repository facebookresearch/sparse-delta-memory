# Copyright (c) Meta Platforms, Inc. and affiliates.
"""GPU smoke test for SparseDeltaMemory across its configuration flags.

Instantiates the layer, runs a training forward + backward (the Triton and CUDA kernels
JIT-compile on first use) and checks shapes, finiteness and that every parameter that
should receive a gradient does. Runs under pytest or standalone (no pytest needed):

    PYTHONPATH=. python tests/test_smoke_sdm.py
"""
import torch

from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

try:
    import pytest

    _needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
    _parametrize = pytest.mark.parametrize
except ImportError:  # standalone run through main()
    def _needs_cuda(fn):
        return fn

    def _parametrize(*args, **kwargs):
        return lambda fn: fn

DIM, B, T = 256, 2, 128

# (label, overrides)
CONFIGS = [
    ("h1_minimal", dict(num_heads=1, output_gate=False, normalize_readings=False,
                        backprop_on_memory=False)),
    ("h1_gate_norm", dict(num_heads=1, output_gate=True, normalize_readings=True)),
    ("h4_gate_norm", dict(num_heads=4, output_gate=True, normalize_readings=True)),
    ("h1_query_batchnorm", dict(num_heads=1, query_batchnorm=True)),
    ("h1_key_weighted_decay", dict(num_heads=1, key_weighted_decay=True)),
    ("h2_block128", dict(num_heads=2, chunk_size=128)),
    ("h2_block48", dict(num_heads=2, chunk_size=48, num_writes=48, num_reads=24)),
    ("h2_snapshot_fp8", dict(num_heads=2, snapshot_quant="fp8")),
    ("h2_snapshot_int4", dict(num_heads=2, snapshot_quant="int4")),
    ("h2_hd128_k64", dict(num_heads=2, num_writes=64, num_reads=64)),
    ("h2_l2norm_acts", dict(num_heads=2, read_act="l2_normalize", write_act="l2_normalize")),
]


def smoke(overrides, dtype, seq_len=T):
    cfg = dict(dim=DIM, slots_per_head=1024, num_reads=8, num_writes=8, chunk_size=64,
               read_act="Softmax", write_act="Softmax")
    cfg.update(overrides)
    torch.manual_seed(0)
    layer = SparseDeltaMemory(SparseDeltaMemoryArgs(**cfg), 0).cuda()
    layer.init_weights(init_std=0.02)
    layer = layer.to(dtype)
    x = torch.randn(B, seq_len, DIM, device="cuda", dtype=dtype, requires_grad=True)

    out, state = layer(x)
    assert out.shape == (B, seq_len, DIM) and out.dtype == dtype
    assert torch.isfinite(out).all(), "non-finite output"
    assert state.cache_len == seq_len

    out.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "bad input gradient"
    missing = [n for n, p in layer.named_parameters() if p.grad is None]
    # Wo.bias and the readings-norm bias get gradients too; nothing may be disconnected.
    assert not missing, f"parameters without gradient: {missing}"
    bad = [n for n, p in layer.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad}"
    assert x.grad.abs().max() > 0


@_needs_cuda
@_parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@_parametrize("label,overrides", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_forward_backward_smoke(label, overrides, dtype):
    smoke(overrides, dtype)


@_needs_cuda
@_parametrize("seq_len", [100, 37])
def test_training_sequence_not_a_multiple_of_the_block(seq_len):
    smoke(dict(num_heads=2), torch.bfloat16, seq_len=seq_len)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    print("GPU:", torch.cuda.get_device_name(0))
    for dtype in (torch.float32, torch.bfloat16):
        for label, overrides in CONFIGS:
            smoke(overrides, dtype)
            print(f"[PASS] {label:24s} {str(dtype).split('.')[-1]}")
    for seq_len in (100, 37):
        smoke(dict(num_heads=2), torch.bfloat16, seq_len=seq_len)
        print(f"[PASS] {'T=' + str(seq_len):24s} bfloat16")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
