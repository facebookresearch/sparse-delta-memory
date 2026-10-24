# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Layer I/O kernels (head split/merge, per-batch bank expand), the repeated-write-slot
counter, the op's argument checks and its shared-bank opt-in."""
import numpy as np
import pytest
import torch
from einops import rearrange

from sdm_testlib import (
    GATE_FP32,
    bits_equal,
    layer_fwd_bwd,
    layer_inputs,
    make_layer,
    make_op_inputs,
    rel_max,
    require_free_memory,
    requires_cuda,
    streams_own_their_slots,
)

pytestmark = requires_cuda


# ========================================================= head split / head merge
HEAD_CASES = [
    # (B, T, H, CD, dtype): packed bf16 rows, row-per-program copies and the views
    (2, 64, 4, 128, torch.bfloat16),
    (2, 37, 8, 64, torch.bfloat16),
    (3, 16, 2, 96, torch.bfloat16),
    (2, 64, 4, 128, torch.float32),
    (2, 5, 3, 256, torch.float32),
    (2, 1, 4, 128, torch.bfloat16),     # T == 1 (decode)
    (2, 64, 1, 128, torch.bfloat16),    # H == 1
]


@pytest.mark.parametrize("B,T,H,CD,dtype", HEAD_CASES,
                         ids=[f"B{c[0]}T{c[1]}H{c[2]}CD{c[3]}_{str(c[4])[6:]}" for c in HEAD_CASES])
def test_head_split_and_merge_are_bitwise_rearranges(B, T, H, CD, dtype):
    from sparse_delta_memory.memory_ops import head_merge, head_split

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, T, H * CD, device="cuda", generator=g).to(dtype).requires_grad_(True)
    y = head_split(x, H)
    assert bits_equal(y, rearrange(x.detach(), "b t (h k) -> (b h) t k", h=H))
    gy = torch.randn(y.shape, device="cuda", generator=g).to(dtype)
    y.backward(gy)
    assert bits_equal(x.grad, rearrange(gy, "(b h) t k -> b t (h k)", h=H))

    z_in = torch.randn(B * H, T, CD, device="cuda", generator=g).to(dtype).requires_grad_(True)
    z = head_merge(z_in, H)
    assert bits_equal(z, rearrange(z_in.detach(), "(b h) t d -> b t (h d)", h=H))
    gz = torch.randn(z.shape, device="cuda", generator=g).to(dtype)
    z.backward(gz)
    assert bits_equal(z_in.grad, rearrange(gz, "b t (h d) -> (b h) t d", h=H))
    assert bits_equal(head_merge(head_split(x.detach(), H), H), x.detach())


def test_head_split_handles_a_strided_input():
    from sparse_delta_memory.memory_ops import head_split

    x = torch.randn(2, 64, 2 * 4 * 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert not x.is_contiguous()
    assert bits_equal(head_split(x, 4), rearrange(x, "b t (h k) -> (b h) t k", h=4))


# ======================================================== per-batch memory expand
S, D, B = 4096, 320, 3


def _param(requires_grad=False, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(S, D, device="cuda", generator=g).to(torch.bfloat16).requires_grad_(
        requires_grad)


def test_expand_matches_torch_and_its_backward_sums_the_copies():
    from sparse_delta_memory.memory_ops import expand_memory_batch

    g = torch.Generator(device="cuda").manual_seed(1)
    cot = torch.randn(B * S, D, device="cuda", generator=g).to(torch.bfloat16)
    grads = []
    for _ in range(2):
        p = _param(requires_grad=True)
        out = expand_memory_batch(p, B)
        assert bits_equal(out, p.detach().unsqueeze(0).expand(B, S, D).reshape(B * S, D))
        out.backward(cot)
        grads.append(p.grad)
    assert bits_equal(grads[0], grads[1]), "backward not reproducible"
    ref = cot.double().view(B, S, D).sum(0)
    assert ((grads[0].double() - ref).abs() <= ref.abs() * 2.0 ** -8 + 1e-30).all()


def test_expand_returns_private_mutable_copies():
    from sparse_delta_memory.memory_ops import expand_memory_batch

    p = _param()
    snapshot = p.clone()
    out = expand_memory_batch(p, B)
    assert out.is_contiguous() and 0 not in out.stride()
    assert out.untyped_storage().data_ptr() != p.untyped_storage().data_ptr()
    view = out.view(B, S, D)
    before = view.clone()
    view[1].add_(1.0)
    view[1, 7].zero_()
    assert bits_equal(view[0], before[0]) and bits_equal(view[2], before[2])
    assert bits_equal(p, snapshot), "writing the expanded bank changed the parameter"


@pytest.mark.slow
def test_batch_reduction_indexes_past_int32():
    """The expand backward sums [B, S, D] over B; B * S * D > 2**31 needs int64 offsets."""
    from sparse_delta_memory.memory_ops import triton_sum_dim0

    require_free_memory(16)
    Bb, Dd = 2, 1024
    Ss = (1 << 30) // Dd + 64                     # Bb * Ss * Dd = 2**31 + 131072
    rows = (torch.arange(Ss, device="cuda") % 61 * 2).view(1, Ss, 1)
    x = (rows + torch.arange(Bb, device="cuda").view(Bb, 1, 1)).expand(Bb, Ss, Dd)
    x = x.to(torch.bfloat16).contiguous()
    out = triton_sum_dim0(x)
    del x
    want = (rows[0] * 2 + 1).to(torch.bfloat16).expand(Ss, Dd)
    assert torch.equal(out, want)


# ============================================================ repeated write slots
def _op_args(dtype=torch.float32, **kw):
    return make_op_inputs(2, 64, 4, 4, 32, 64, dtype, seed=0, **kw)


def _call(d, **kw):
    from sparse_delta_memory.memory_ops import sdm_write_read

    args = dict(chunk_size=32, num_memory_slots=64, num_parallel=2)
    args.update(kw)
    return sdm_write_read(d["memory"], d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"],
                          d["q_idx"], d["q_val"], **args)


def test_repeated_write_slots_are_counted():
    from sparse_delta_memory.memory_ops import repeated_write_slots

    # The counter is per process and every op call of every earlier test adds to it.
    repeated_write_slots(reset=True)
    d = _op_args()
    with torch.no_grad():
        _call(d)
        assert repeated_write_slots(reset=True) == 0
        # Token 5 of stream 0 writes its first slot twice, token 9 of stream 1 three times.
        d["k_idx"][0, 5, 1] = d["k_idx"][0, 5, 0]
        d["k_idx"][1, 9, 1:3] = d["k_idx"][1, 9, 0]
        _call(d)
        assert repeated_write_slots(reset=False) == 3
        assert repeated_write_slots(reset=True) == 3
        assert repeated_write_slots() == 0


# ======================================================================= op arguments
@pytest.mark.parametrize("change,exc,match", [
    (dict(chunk_size=True), TypeError, "chunk_size"),
    (dict(chunk_size=15), ValueError, "chunk_size"),
    (dict(chunk_size=257), ValueError, "chunk_size"),
    (dict(num_parallel=2.0), TypeError, "num_parallel"),
    (dict(num_parallel=3), ValueError, "num_parallel"),
    (dict(snapshot_quant="fp4"), ValueError, "snapshot_quant"),
    (dict(key_weighted_decay=True), ValueError, "key_weighted_decay"),
    # Malformed counts are type errors whatever the bank layout, opt-in or not (the
    # shared-bank check only sees well-formed ones; 0 and 128 make _op_args' bank shared).
    *[(dict(num_parallel=n, num_memory_slots=s, **o), TypeError, "num_parallel")
      for n in (2.0, np.int64(2), True) for s in (0, 128) for o in ({}, dict(shared_bank=True))],
    *[(dict(num_memory_slots=s), TypeError, "num_memory_slots") for s in (128.0, False)],
])
def test_op_rejects_malformed_arguments(change, exc, match):
    with pytest.raises(exc, match=match):
        _call(_op_args(), **change)


def test_op_rejects_unsupported_tensors():
    d = _op_args()
    for name, bad, exc in (("k_idx", d["k_idx"].float(), TypeError),
                           ("memory", d["memory"].half(), TypeError),
                           ("memory", d["memory"].double(), TypeError),
                           ("memory", d["memory"].t().contiguous().t(), ValueError),
                           ("v", d["v"][..., :31], ValueError)):
        with pytest.raises(exc, match=name):
            _call(dict(d, **{name: bad}))
    # Four slot ranges of 32 but only two parallel streams.
    with pytest.raises(ValueError, match="num_parallel"):
        _call(d, num_memory_slots=32)
    # A bank of the wrong rank gets its own error whatever the bank layout, opt-in or not
    # (the shared-bank check runs on validated arguments only).
    for bad in (d["memory"].view(-1), d["memory"].view(2, 64, 32), d["memory"][0, 0]):
        for change in ({}, dict(num_memory_slots=0), dict(num_memory_slots=128),
                       dict(num_memory_slots=0, shared_bank=True)):
            with pytest.raises(ValueError, match=r"\[num_slots, head_dim\]"):
                _call(dict(d, memory=bad), **change)


@pytest.mark.parametrize("W,R", [(257, 4), (4, 257)])
def test_op_rejects_more_than_256_slots_per_token(W, R):
    d = make_op_inputs(1, 8, W, R, 16, 512, torch.float32, seed=0)
    with pytest.raises(ValueError, match="256"):
        _call(d, chunk_size=16, num_memory_slots=512, num_parallel=1)


# ======================================================================== shared banks
def _run(d, *, train=True, flat=False, **kw):
    """Readings and final bank (with ``train`` also every input gradient) of one call on
    private copies of ``d``; by default two streams of 64 slots each. ``flat`` passes the
    [P * T, *] layout of the same streams."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    names = ("memory", "k_val", "v", "beta", "g", "q_val")
    leaves = {n: d[n].detach().clone().requires_grad_(train) for n in names}
    args = dict(chunk_size=32, num_memory_slots=64, num_parallel=2)
    args.update(kw)

    def lay(t):
        return t.reshape(-1, t.shape[-1]) if flat else t

    with torch.enable_grad() if train else torch.no_grad():
        out = sdm_write_read(leaves["memory"], *(lay(t) for t in (
            d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"], d["q_idx"],
            leaves["q_val"])), **args)
    res = {"readings": out.detach().clone(), "final_bank": leaves["memory"].detach().clone()}
    if train:
        grads = torch.autograd.grad(out, [leaves[n] for n in names], torch.ones_like(out))
        res.update({"grad_" + n: g for n, g in zip(names, grads)})
    return res


@pytest.mark.parametrize("flat", [False, True], ids=["3d", "2d"])
@pytest.mark.parametrize("train", [True, False], ids=["train", "no_grad"])
@pytest.mark.parametrize("slots", [0, 128], ids=["slots_0", "slots_of_the_bank"])
def test_streams_over_one_shared_bank_need_the_opt_in(slots, train, flat, ieee_fp32):
    """Several streams over one shared bank (num_memory_slots 0, or the bank's 128 rows; [P,
    T, *] or [P * T, *] inputs) are refused unless shared_bank=True asserts that the streams
    select disjoint slots, which the op cannot check. With it, streams on disjoint slot
    ranges compute what the same streams compute over slot ranges of their own."""
    d = _op_args()  # two streams, each on its own 64 of the bank's 128 rows
    assert d["memory"].shape[0] == 128
    assert streams_own_their_slots(d["k_idx"], 64) and streams_own_their_slots(d["q_idx"], 64)
    with pytest.raises(ValueError, match="shared_bank=True"):
        _run(d, train=train, flat=flat, num_memory_slots=slots)
    shared = _run(d, train=train, flat=flat, num_memory_slots=slots, shared_bank=True)
    own = _run(d, train=train, flat=flat)
    bad = {k: rel_max(ref, shared[k]) for k, ref in own.items()
           if not rel_max(ref, shared[k]) < GATE_FP32}
    assert not bad, bad


@pytest.mark.parametrize("slots", [0, 64], ids=["slots_0", "slots_of_the_bank"])
def test_one_stream_and_own_slot_ranges_need_no_opt_in(slots):
    """One stream over its whole bank (num_memory_slots 0 or the bank's 64 rows), in both
    paths, and streams on slot ranges of their own run without shared_bank."""
    one = make_op_inputs(1, 64, 4, 4, 32, 64, torch.float32, seed=0)
    for train in (True, False):
        _run(one, train=train, num_memory_slots=slots, num_parallel=1)
        _run(_op_args(), train=train)


@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "cuda_graph"])
def test_the_layer_needs_no_opt_in(cuda_graph):
    """The layer gives every stream (batch row x head) its own slot range: training steps,
    eager or replayed from the layer's CUDA graphs, a no-grad prefill and a decode step run
    as before."""
    from sparse_delta_memory.cuda_graph import reset_cuda_graphs

    layer = make_layer(seed=0, cuda_graph=cuda_graph)
    x, gy = layer_inputs(2, 64, 256)
    try:
        for _ in range(3):
            layer_fwd_bwd(layer, x, gy)
        if cuda_graph:
            assert layer.graph_runner.stats.get("backward_replay", 0) >= 1, \
                dict(layer.graph_runner.stats)
        layer.eval()
        with torch.no_grad():
            layer(x)
            cache = layer.create_kv_cache(bsz=2, seq_len=0, dtype=x.dtype, device="cuda")
            layer(x[:, :1], cache=cache)
    finally:
        reset_cuda_graphs()


# ======================================================================= index limits
@pytest.mark.parametrize("P,T,C,W,hd,gib", [
    (70000, 16, 16, 1, 16, 1),
    pytest.param(8200, 16, 256, 1, 1024, 112, marks=pytest.mark.slow),
    pytest.param(524289, 16, 16, 256, 1, 200, marks=pytest.mark.slow),
], ids=["streams_past_65535", "tokens_x_head_dim", "write_entries"])
def test_training_call_at_index_limits(P, T, C, W, hd, gib):
    """A differentiable call with more streams than a grid's y or z axis holds (65535),
    with P * C * head_dim (the dual matmul's token rows) above 2**31, or with P * C * W (the
    movers' entries) above 2**31: the readings, final bank and gradients of the first and
    last streams match the same streams run alone."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    require_free_memory(gib)
    sph = 2 * W
    gen = torch.Generator(device="cuda").manual_seed(0)
    # Token t writes slots 2w + t % 2 (distinct, colliding every other token) and reads
    # slot 7t mod sph.
    tok = torch.arange(T, device="cuda", dtype=torch.int32).view(1, T, 1)
    off = (torch.arange(P, device="cuda", dtype=torch.int32) * sph).view(P, 1, 1)
    d = dict(memory=torch.randn(P * sph, hd, device="cuda", generator=gen) * 0.1,
             k_idx=torch.arange(W, device="cuda", dtype=torch.int32) * 2 + tok % 2 + off,
             k_val=torch.rand(P, T, W, device="cuda", generator=gen) * (2 / W),
             v=torch.randn(P, T, hd, device="cuda", generator=gen) * 0.1,
             beta=torch.rand(P, T, 1, device="cuda", generator=gen) * 0.3,
             g=-torch.rand(P, T, 1, device="cuda", generator=gen) * 0.5,
             q_idx=tok * 7 % sph + off,
             q_val=torch.randn(P, T, 1, device="cuda", generator=gen) * 0.3)
    grad_out = torch.randn(P, T, hd, device="cuda", generator=gen)
    names = ("memory", "k_val", "v", "beta", "g", "q_val")

    def run(x, lo, hi):
        """Readings, final bank and input gradients of streams lo..hi-1 given as ``x``."""
        leaves = {n: x[n].requires_grad_(True) for n in names}
        out = sdm_write_read(leaves["memory"], x["k_idx"], leaves["k_val"], leaves["v"],
                             leaves["beta"], leaves["g"], x["q_idx"], leaves["q_val"],
                             chunk_size=C, num_memory_slots=sph, num_parallel=hi - lo)
        final = leaves["memory"].detach().clone()
        out.backward(grad_out[lo:hi].reshape(out.shape))
        res = {"readings": out.detach().view(hi - lo, T, hd), "final": final}
        res.update({"grad_" + n: leaves[n].grad for n in names})
        return res

    big = run(d, 0, P)  # the backward restores the bank for the runs below
    for lo, hi in ((0, 4), (P - 4, P)):
        x = {n: t.detach()[lo * sph:hi * sph] if n == "memory" else t.detach()[lo:hi]
             for n, t in d.items()}
        x = {n: t - lo * sph if n.endswith("idx") else t.clone() for n, t in x.items()}
        for name, ref in run(x, lo, hi).items():
            rows = slice(lo * sph, hi * sph) if name in ("final", "grad_memory") else slice(lo, hi)
            assert rel_max(ref, big[name][rows]) <= 1e-5, (lo, name)
