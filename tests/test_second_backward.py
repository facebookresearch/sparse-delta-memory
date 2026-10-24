# Copyright (c) Meta Platforms, Inc. and affiliates.
"""A second backward through one training-op call (``retain_graph=True``, then another backward)
raises a RuntimeError: the forward advances the bank in place and the first backward restores it,
so a second backward would start from the restored bank and return a wrong gradient. The first
backward of a retained graph is bitwise equal to the backward of a fresh call (a consistency
check; the gradients themselves are checked against references elsewhere in the suite)."""
import pytest
import torch

from sdm_testlib import bits_equal, make_op_inputs, requires_cuda

pytestmark = requires_cuda

SPH, DIM, W, R, P, T = 256, 64, 4, 4, 2, 64
GRAD_INPUTS = ("memory", "k_val", "v", "beta", "g", "q_val")


def _call(d):
    from sparse_delta_memory.memory_ops import sdm_write_read

    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in GRAD_INPUTS}
    out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"],
                         leaves["beta"], leaves["g"], d["q_idx"], leaves["q_val"],
                         chunk_size=64, num_memory_slots=SPH, num_parallel=P)
    return out, leaves


def test_second_backward_raises_and_first_is_unchanged():
    d = make_op_inputs(P, T, W, R, DIM, SPH, torch.float32, seed=7)
    out, leaves = _call(d)
    gy = torch.randn(out.shape, generator=torch.Generator().manual_seed(1)).to(out)
    first = torch.autograd.grad(out, [leaves[n] for n in GRAD_INPUTS], gy, retain_graph=True)
    with pytest.raises(RuntimeError, match="second backward through one call"):
        torch.autograd.grad(out, [leaves[n] for n in GRAD_INPUTS], gy)
    out2, leaves2 = _call(d)
    fresh = torch.autograd.grad(out2, [leaves2[n] for n in GRAD_INPUTS], gy)
    for name, a, b in zip(GRAD_INPUTS, first, fresh):
        assert bits_equal(a, b), name


def test_cuda_graph_training_backward_every_step():
    """The cuda_graph option's runner takes several backwards on its own stand-in ctx (warm-up,
    capture, replays after re-running the forward): the guard must never fire there. Five
    training steps, one backward each: one forward and one backward capture, every other call
    replayed, no eager fallback. A retained double backward of a graphed call then replays its
    forward instead of raising."""
    from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

    torch.manual_seed(0)
    layer = SparseDeltaMemory(SparseDeltaMemoryArgs(
        dim=256, cuda_graph=True, num_heads=2, slots_per_head=1024, num_writes=16, num_reads=16,
        chunk_size=64), 0)
    layer.init_weights()
    layer = layer.to("cuda", torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(3)

    def batch():
        x = torch.randn(2, 256, 256, device="cuda", generator=gen).to(torch.bfloat16)
        gy = torch.randn(2, 256, 256, device="cuda", generator=gen).to(torch.bfloat16)
        return x.requires_grad_(True), gy

    for _ in range(5):
        x, gy = batch()
        out, _ = layer(x)
        out.backward(gy)
    stats = dict(layer.graph_runner.stats)
    assert stats == dict(forward_capture=1, forward_replay=5, backward_capture=1,
                         backward_replay=4), stats
    x, gy = batch()
    out, _ = layer(x)
    out.backward(gy, retain_graph=True)
    out.backward(gy)  # the option's exception: replays the forward first, no RuntimeError
