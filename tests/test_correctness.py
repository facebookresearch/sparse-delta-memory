# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Layer-level correctness of SparseDeltaMemory.

The kernels are deterministic, so repeated runs of one configuration are bitwise equal, and
so are two calls that take the same path (train and eval mode under no_grad). Two chunk
sizes round differently: fp32 (TF32 off) is gated tightly, bf16 with fixed tolerances
(``assert_close_across``). Op-level checks against a float64 reference live in
test_f64_oracle.py and test_autograd_reference.py; eval-path checks, including plain and
reentrant activation checkpointing, in test_eval_paths.py; determinism in
test_determinism.py.
"""
import contextlib

import pytest
import torch

from sdm_testlib import (
    ProjectionReplay,
    OP_GRADS,
    OP_INPUTS,
    assert_close_across,
    assert_runs_bitwise,
    layer_fwd_bwd,
    layer_inputs,
    make_cotangent,
    make_layer,
    make_op_inputs,
    rel_max,
    rel_rms,
    requires_cuda,
)

BASE = dict(dim=128, slots_per_head=1024, num_writes=16, num_reads=16, chunk_size=64,
            normalize_readings=True)


# bf16 layer, chunk 64 vs 128: relative RMS per tensor, relative max for the per-head
# parameters (at most TINY_NUMEL elements, sums with heavy cancellation). Both chunk sizes
# select the same slots; the difference is the bf16 bank rounded at other chunk boundaries:
# up to 6.1e-3 (grad.Wq_read.bias) under the 1e-2 relative RMS bound, and up to 1.2e-2 on the
# per-head parameters, under 4 bf16 eps.
CHUNK_SIZE_BF16 = dict(rms=1e-2, tiny_max=4 * torch.finfo(torch.bfloat16).eps)


def _pair(seed=42, dtype=torch.bfloat16, **kw):
    """Two layers with identical weights that differ only in ``kw``-selected options."""
    a = make_layer(seed=seed, dtype=dtype, **dict(BASE, **kw.get("a", {})))
    b = make_layer(seed=seed + 1, dtype=dtype, **dict(BASE, **kw.get("b", {})))
    b.load_state_dict(a.state_dict())
    return a, b


@requires_cuda
@pytest.mark.parametrize("H", [1, 2])
def test_layer_output_and_gradients_are_chunk_size_invariant(H, ieee_fp32):
    x, gy = layer_inputs(2, 256, BASE["dim"], dtype=torch.float32)
    a, b = _pair(dtype=torch.float32, a=dict(num_heads=H, chunk_size=64),
                 b=dict(num_heads=H, chunk_size=128))
    ra, rb = layer_fwd_bwd(a, x, gy), layer_fwd_bwd(b, x, gy)
    for k in ra:
        assert rel_rms(ra[k], rb[k]) < 1e-5, f"H={H} fp32 {k}: {rel_rms(ra[k], rb[k]):.3e}"

    x, gy = x.bfloat16(), gy.bfloat16()
    a, b = _pair(a=dict(num_heads=H, chunk_size=64), b=dict(num_heads=H,
                                                                  chunk_size=128))
    runs_a = [layer_fwd_bwd(a, x, gy) for _ in range(3)]
    runs_b = [layer_fwd_bwd(b, x, gy) for _ in range(3)]
    # The bf16 bank is rounded at different chunk boundaries: fixed bounds, see
    # CHUNK_SIZE_BF16.
    assert_close_across(runs_a, runs_b, **CHUNK_SIZE_BF16, label=f"H={H} bf16 cs 64 vs 128")


@requires_cuda
@pytest.mark.parametrize("T", [100, 37])
def test_training_on_a_sequence_that_is_not_a_block_multiple(T, ieee_fp32):
    """A training step on T tokens equals the first T positions of a step on more tokens
    (causality), including every gradient."""
    layer = make_layer(seed=0, dtype=torch.float32, **dict(BASE, num_heads=2))
    x, gy = layer_inputs(2, 128, BASE["dim"], dtype=torch.float32, seed=2)
    gy_short = gy.clone()
    gy_short[:, T:] = 0
    full = layer_fwd_bwd(layer, x, gy_short)
    short = layer_fwd_bwd(layer, x[:, :T].contiguous(), gy[:, :T].contiguous())
    assert rel_max(full["out"][:, :T], short["out"]) < 1e-5
    assert rel_max(full["grad_x"][:, :T], short["grad_x"]) < 1e-5
    for k in full:
        if k.startswith("grad."):
            assert rel_max(full[k], short[k]) < 1e-5, k


@requires_cuda
@pytest.mark.parametrize("kw", [dict(num_heads=1), dict(num_heads=2, key_weighted_decay=True),
                                dict(num_heads=2, query_batchnorm=True)],
                         ids=["h1", "h2_key_weighted_decay", "h2_query_batchnorm"])
def test_train_forward_matches_eval_forward(kw):
    """Under no_grad both modes take the op's no-grad path with the same slabs, stream
    groups and gather geometry; only BatchNorm (and memory statistics) depend on the mode.
    So the outputs are bitwise equal."""
    layer = make_layer(seed=42, **dict(BASE, **kw))
    x, _ = layer_inputs(2, 256, BASE["dim"], seed=7, scale=0.1)
    runs = {}
    for train in (True, False):
        layer.train(train)
        with torch.no_grad():
            runs[train] = [{"out": layer(x)[0]} for _ in range(3)]
    if kw.get("query_batchnorm"):
        # BatchNorm uses batch statistics in training and running statistics in eval.
        assert rel_rms(runs[True][0]["out"], runs[False][0]["out"]) > 1e-2
        return
    assert_runs_bitwise(runs[True] + runs[False], label=str(kw))


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("kw", [dict(num_heads=1), dict(num_heads=2),
                                dict(num_heads=2, key_weighted_decay=True),
                                dict(num_heads=1, dim=2048)],
                         ids=["h1", "h2", "h2_key_weighted_decay", "h1_d2048"])
def test_token_by_token_decode_matches_the_full_forward(kw, dtype, ieee_fp32):
    """The decode consumes the full forward's projection rows wherever their inputs are bit-identical
    (ProjectionReplay): some GPUs' cuBLAS gives one row other bits than the same row among T rows (A100: other
    kernels for M = 1), which flips near-ties of the product-key selection and moves the whole output (7.8e-2 at
    h1_d2048 bf16 on A100); where cuBLAS is row-count invariant (GB300) nothing is replayed. The gate then measures the
    layer's own decode-vs-full difference (the bank rounded per token vs per chunk) on every GPU."""
    layer = make_layer(seed=42, dtype=dtype, **dict(BASE, **kw))
    layer.eval()
    x, _ = layer_inputs(1, 96, layer.dim, dtype=dtype, seed=99, scale=0.1)
    rep = ProjectionReplay(layer)
    with torch.no_grad():
        with rep.record():
            full, state = layer(x)
        assert state.cache_len == 96
        cache, outs = None, []
        with rep.decode():
            for t in range(x.shape[1]):
                o, cache = layer(x[:, t:t + 1], cache=cache)
                outs.append(o)
    rep.close()
    assert cache.cache_len == 96
    err = rel_rms(full, torch.cat(outs, dim=1))
    assert err < (1e-5 if dtype is torch.float32 else 1.5e-2), \
        f"{kw} {dtype}: {err:.3e} ({rep.replaced} projection rows replayed)"


@requires_cuda
def test_selective_activation_checkpointing_gives_the_same_gradients(ieee_fp32):
    """Selective checkpointing that saves matmul outputs (as lingua's policy does) around the
    layer: the op hides its internals from the policy and is recomputed as a whole."""
    from functools import partial

    from torch.utils.checkpoint import (
        CheckpointPolicy,
        checkpoint,
        create_selective_checkpoint_contexts,
    )

    def policy(ctx, op, *args, **kwargs):
        saved = (torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
                 torch.ops.aten.baddbmm.default, torch.ops.aten.addmm.default)
        return CheckpointPolicy.MUST_SAVE if op in saved else CheckpointPolicy.PREFER_RECOMPUTE

    layer = make_layer(seed=3, dtype=torch.float32, **dict(BASE, num_heads=2))
    x, gy = layer_inputs(2, 128, BASE["dim"], dtype=torch.float32, seed=8)
    plain = layer_fwd_bwd(layer, x, gy)

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, t):
            return checkpoint(lambda u: self.inner(u)[0], t, use_reentrant=False,
                              context_fn=partial(create_selective_checkpoint_contexts,
                                                 policy)), None

    sac = layer_fwd_bwd(Wrapped(layer), x, gy)
    for k in plain:
        name = k.replace("grad.", "grad.inner.") if k.startswith("grad.") else k
        assert rel_max(plain[k], sac[name]) < 1e-5, k


def _op_fwd_bwd(d, G, autocast=False):
    from sparse_delta_memory.memory_ops import sdm_write_read

    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"],
                             leaves["beta"], leaves["g"], d["q_idx"], leaves["q_val"],
                             chunk_size=32, num_memory_slots=256, num_parallel=2)
        grads = torch.autograd.grad(out, [leaves[n] for n in OP_INPUTS], grad_outputs=G)
    return dict(zip(OP_GRADS, grads), out=out.detach())


@requires_cuda
def test_the_op_runs_outside_torch_dispatch_modes(monkeypatch):
    """The op saves its own backward state, so its internal ops are hidden from dispatch
    modes (selective checkpointing would otherwise intercept every one of them). This pins
    the private ``torch.utils._python_dispatch._disable_current_modes`` that it relies on."""
    from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

    from sparse_delta_memory import memory_ops

    assert memory_ops._disable_current_modes is _disable_current_modes

    class Recorder(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.add(str(func.overloadpacket))
            return func(*args, **(kwargs or {}))

    d = make_op_inputs(2, 64, 8, 8, 32, 256, torch.float32, seed=1)
    G = make_cotangent(d["v"].reshape(128, 32))
    internal = {"aten.empty", "aten.bmm", "aten.baddbmm"}
    for hidden in (True, False):
        if not hidden:  # control: the recorder sees the internals once they are exposed
            monkeypatch.setattr(memory_ops, "_disable_current_modes", contextlib.nullcontext)
        with Recorder() as rec:
            _op_fwd_bwd(d, G)
        seen = rec.ops & internal
        assert (not seen) if hidden else seen == internal, sorted(rec.ops)


@requires_cuda
def test_the_backward_ignores_autocast(ieee_fp32):
    """Under autocast the backward matmuls still run on their float32 operands."""
    d = make_op_inputs(2, 64, 8, 8, 32, 256, torch.float32, seed=2)
    G = make_cotangent(d["v"].reshape(128, 32))
    plain, cast = _op_fwd_bwd(d, G), _op_fwd_bwd(d, G, autocast=True)
    for k in plain:
        assert cast[k].dtype == plain[k].dtype
        assert rel_max(plain[k], cast[k]) < 1e-5, k


def test_same_seed_gives_identical_weights():
    a = make_layer(seed=123, dtype=torch.float32, device="cpu", **BASE)
    b = make_layer(seed=123, dtype=torch.float32, device="cpu", **BASE)
    c = make_layer(seed=124, dtype=torch.float32, device="cpu", **BASE)
    pa, pb, pc = (dict(m.named_parameters()) for m in (a, b, c))
    assert all(torch.equal(pa[n], pb[n]) for n in pa)
    assert not all(torch.equal(pa[n], pc[n]) for n in pa)
