# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Standalone GPU smoke test for SparseDeltaMemory.

Self-contained: imports only `lingua.sparse_delta_memory` + torch — no internal framework, no
distributed/NCCL. Instantiates the layer across the supported flag configs, runs a
forward + backward on CUDA (the Triton + CUDA kernels JIT-compile on first use), and
asserts outputs and gradients are finite. Run in the release env:

    PYTHONPATH=. python tests/smoke_sdm.py
"""
import torch

from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

DIM, B, T = 256, 2, 128


def smoke(nh, og, nr, qb, kwd, bpm, dtype, tag):
    args = SparseDeltaMemoryArgs(
        dim=DIM, num_heads=nh, slots_per_head=1024, num_reads=8, num_writes=8,
        memory_block_size=64, output_gate=og, normalize_readings=nr,
        query_batchnorm=qb, key_weighted_decay=kwd, backprop_on_memory=bpm,
        read_act="Softmax", write_act="Softmax",
    )
    layer = SparseDeltaMemory(args, 0).cuda()
    layer.init_weights(init_std=0.02)
    layer = layer.to(dtype)
    x = torch.randn(B, T, DIM, device="cuda", dtype=dtype, requires_grad=True)

    out, _ = layer(x)
    assert out.shape == (B, T, DIM), f"bad shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite output"

    out.float().pow(2).mean().backward()
    bad = [n for n, p in layer.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad}"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "non-finite input grad"
    dt = str(dtype).split(".")[-1]
    print(f"[PASS] {tag:34s} dtype={dt:8s} out{tuple(out.shape)} + grads finite")


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    print("GPU:", torch.cuda.get_device_name(0))
    cfgs = [
        (1, False, False, False, False, False, "H1 minimal"),
        (1, True,  True,  False, False, False, "H1 +output_gate +readings_norm"),
        (4, True,  True,  False, False, False, "H4 +og +nr"),
        (1, True,  True,  True,  False, False, "H1 +query_batchnorm"),
        (1, True,  True,  False, True,  False, "H1 +key_weighted_decay"),
        (1, True,  True,  False, False, True,  "H1 +backprop_on_memory"),
    ]
    for dtype in (torch.float32, torch.bfloat16):
        for *flags, tag in cfgs:
            smoke(*flags, dtype, tag)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
