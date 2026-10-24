# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Decode host cost (per token, the route branch and one memo hit): the layer's decode branch
launches decode._decode_launch, the public sdm_decode_step's kernel, arguments and bits, with no package check per token
while the bank keeps the layout the public step checked; a swapped bank gets the public step's checks again."""
import pytest
import torch

from sdm_testlib import requires_cuda

pytestmark = requires_cuda


def test_the_layer_decode_path_is_the_public_step_without_checks(monkeypatch):
    from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs, decode as dec, layer as layer_mod
    from sparse_delta_memory.cuda import _device, _devmemo, _image
    args = SparseDeltaMemoryArgs(dim=128, num_heads=2, num_writes=4, num_reads=4, slots_per_head=1024, chunk_size=64)
    layer = SparseDeltaMemory(args, layer_id=0).cuda().to(torch.bfloat16)
    layer.init_weights()
    x = torch.randn(2, 12, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _, cache = layer(x[:, :8])
        _, cache = layer(x[:, 8:9], cache=cache)
    seen, counts = [], dict.fromkeys(("image", "backend", "device_backend", "tbind", "public"), 0)
    launch = layer_mod._decode_launch

    def spy(memory, *rest):
        seen.append((memory.clone(), [t.clone() for t in rest]))
        out = launch(memory, *rest)
        seen[-1] += (out.clone(), memory.clone())
        return out

    def count(name, fn):
        def wrapped(*a, **k):
            counts[name] += 1
            return fn(*a, **k)
        return wrapped
    monkeypatch.setattr(layer_mod, "_decode_launch", spy)
    monkeypatch.setattr(layer_mod, "sdm_decode_step", count("public", layer_mod.sdm_decode_step))
    monkeypatch.setattr(_device, "tbind", count("tbind", _device.tbind))
    monkeypatch.setattr(_device, "backend", count("device_backend", _device.backend))
    monkeypatch.setattr(_image, "require_supported_image", count("image", _image.require_supported_image))
    monkeypatch.setattr(_devmemo, "backend", count("backend", _devmemo.backend))
    with torch.no_grad():
        for t in range(9, 12):
            _, cache = layer(x[:, t:t + 1], cache=cache)
    assert len(seen) == 3 and counts == dict.fromkeys(counts, 0), counts   # zero package checks per decode token
    for bank0, rest, out, bank1 in seen:                                   # the public step's bits on the same inputs
        bank = bank0.clone()
        assert torch.equal(dec.sdm_decode_step(bank, *rest), out) and torch.equal(bank, bank1)


def test_a_swapped_bank_is_checked_again():
    """A cache bank swapped for one with more columns or too few slot rows, or a cache
    reused for a larger batch, is refused (no launch)."""
    from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs
    args = SparseDeltaMemoryArgs(dim=128, num_heads=2, num_writes=4, num_reads=4, slots_per_head=1024, chunk_size=64)
    layer = SparseDeltaMemory(args, layer_id=0).cuda().to(torch.bfloat16)
    layer.init_weights()
    x = torch.randn(2, 10, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _, cache = layer(x[:, :8])
        _, cache = layer(x[:, 8:9], cache=cache)
        x4 = torch.randn(4, 1, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="needs 8192 rows"):
            layer(x4, cache=cache)                                  # the batch-2 bank for a batch-4 step
        rows = cache.memory.shape[0]
        cache.memory = torch.zeros(rows, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="v has shape"):
            layer(x[:, 9:10], cache=cache)
        cache.memory = torch.zeros(1, 64, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="needs 4096 rows"):
            layer(x[:, 9:10], cache=cache)
