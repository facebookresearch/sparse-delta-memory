# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Token gather (forward K+Q gather and backward gather-weighted-reduce).

References: a float64 gather-reduce for every head_dim / width / dtype, and an
order-exact float32 replay of the reduction order at head_dim 128, W = R = 64, which pins
the output bits.
"""

import pytest
import torch

from sdm_testlib import bits_equal, rel_max, requires_cuda

pytestmark = requires_cuda

HEAD_DIMS = (1, 16, 64, 65, 96, 128, 129, 256, 512, 960, 961, 1024, 1025, 1920, 2048, 4096)
WIDTHS = ((1, 1), (3, 5), (16, 64), (64, 64), (1, 256), (256, 1), (256, 256))


def _gather():
    from sparse_delta_memory.cuda import gather_cuda
    return gather_cuda


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _slots(rows, width, n, gen):
    """``rows`` rows of ``width`` distinct ids in [0, n), in random order."""
    return torch.rand(rows, n, device="cuda", generator=gen).argsort(dim=-1)[:, :width].contiguous()


def _inputs(hd, W, R, dtype, *, rows, slots, seed, idx_dtype=torch.int64, w_dtype=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w_dtype = w_dtype or dtype
    return dict(
        bank=torch.randn(slots, hd, device="cuda", generator=gen).to(dtype),
        idx=_slots(rows, W, slots, gen).to(idx_dtype),
        w=torch.randn(rows, W, device="cuda", generator=gen).to(w_dtype),
        q_idx=_slots(rows, R, slots, gen).to(idx_dtype),
        q_w=torch.randn(rows, R, device="cuda", generator=gen).to(w_dtype),
    )


def _reference(bank, idx, w, row0, tokens, chunk, valid):
    """float64 (out [tokens, hd], raw rows [tokens, W, hd]) with padding zeroed."""
    idx, w = idx[row0:row0 + tokens], w[row0:row0 + tokens].double()
    rows = bank[idx]
    out = torch.einsum("tn,tnd->td", w, rows.double())
    pad = torch.arange(tokens, device=bank.device) % chunk >= valid
    out[pad] = 0
    rows[pad] = 0
    return out, rows, pad


def _run(d, *, has_q, tokens, chunk, valid, row0=0, stash_row0=0, out_row0=0):
    g = _gather()
    bank, W, hd = d["bank"], d["idx"].shape[-1], d["bank"].shape[1]
    width = max(W, d["q_idx"].shape[-1]) if has_q else W
    geo = g.gather_geometry(hd, bank.dtype, width, tokens, num_sms=_num_sms())
    stash = torch.full((stash_row0 + tokens, W, hd), 3.0, device="cuda", dtype=bank.dtype)
    out = torch.full((out_row0 + tokens, hd), 3.0, device="cuda")
    q_out = torch.full_like(out, 3.0)
    q = dict(q_idx=d["q_idx"], q_weights=d["q_w"], q_out=q_out) if has_q else {}
    g.token_gather(bank, d["idx"], d["w"], stash, out, geo, tokens=tokens, chunk=chunk,
                   valid=valid, row0=row0, stash_row0=stash_row0, out_row0=out_row0, **q)
    return (stash[stash_row0:], out[out_row0:], q_out[out_row0:] if has_q else None), geo


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("W,R", WIDTHS, ids=[f"W{w}R{r}" for w, r in WIDTHS])
@pytest.mark.parametrize("hd", HEAD_DIMS)
def test_matches_float64_reference(hd, W, R, dtype):
    """rel_max <= 1e-6 in both modes; exact stash rows; padding gives exact zeros."""
    chunk = 16
    tokens = 2 * chunk if hd * max(W, R) > 2 ** 17 else 6 * chunk
    row0 = chunk
    d = _inputs(hd, W, R, dtype, rows=row0 + tokens + chunk, slots=max(2 * W, 2 * R, 600),
                seed=hd * 7 + W * 3 + R)
    for has_q in (True, False):
        (stash, out, q_out), geo = _run(d, has_q=has_q, tokens=tokens, chunk=chunk, valid=11,
                                        row0=row0, stash_row0=2, out_row0=3)
        ref, rows, pad = _reference(d["bank"], d["idx"], d["w"], row0, tokens, chunk, 11)
        assert rel_max(ref, out) <= 1e-6, (geo, rel_max(ref, out))
        assert bits_equal(stash, rows), geo
        assert bits_equal(out[pad], torch.zeros_like(out[pad])), geo
        if has_q:
            ref_q, _, _ = _reference(d["bank"], d["q_idx"], d["q_w"], row0, tokens, chunk, 11)
            assert rel_max(ref_q, q_out) <= 1e-6, (geo, rel_max(ref_q, q_out))
            assert bits_equal(q_out[pad], torch.zeros_like(q_out[pad])), geo


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32], ids=["wbf16", "wf32"])
def test_index_and_weight_dtypes(idx_dtype, w_dtype):
    d = _inputs(256, 48, 17, torch.bfloat16, rows=256, slots=4096, seed=5,
                idx_dtype=idx_dtype, w_dtype=w_dtype)
    (stash, out, q_out), _ = _run(d, has_q=True, tokens=256, chunk=32, valid=32)
    ref, rows, _ = _reference(d["bank"], d["idx"], d["w"], 0, 256, 32, 32)
    ref_q, _, _ = _reference(d["bank"], d["q_idx"], d["q_w"], 0, 256, 32, 32)
    assert rel_max(ref, out) <= 1e-6 and rel_max(ref_q, q_out) <= 1e-6
    assert bits_equal(stash, rows)


def _order_exact(bank, idx, w, groups):
    """float32 result of the kernel's reduction order with ``groups`` row groups.

    Group g accumulates entries g, g + groups, ... with fused multiply-adds, then the
    groups are added in order starting from +0.0. The test values are bounded (|x| in
    [2^-8, 2^8]), so every product and every product-plus-accumulator sum is exact in
    float64, and rounding each float64 step to float32 replays the FMA exactly.
    """
    rows, w = bank[idx].double(), w.double()
    acc = torch.zeros(idx.shape[0], groups, bank.shape[1], dtype=torch.float64, device=bank.device)
    for j in range(idx.shape[1] // groups):
        n = slice(j * groups, (j + 1) * groups)
        acc = (w[:, n, None] * rows[:, n] + acc).float().double()
    out = torch.zeros(idx.shape[0], bank.shape[1], dtype=torch.float32, device=bank.device)
    for g in range(groups):
        out = (out.double() + acc[:, g]).float()
    return out


def _bounded(shape, lo, hi, gen, dtype):
    mag = torch.exp2(torch.empty(shape, device="cuda").uniform_(lo, hi, generator=gen))
    sign = torch.where(torch.rand(shape, device="cuda", generator=gen) < 0.5, -1.0, 1.0)
    return (mag * sign).to(dtype)


@pytest.mark.parametrize("groups", [None, 8, 16], ids=["default", "g8", "g16"])
def test_reduction_order_is_pinned(groups):
    """bf16 head_dim 128, W = R = 64: every output bit follows the documented order, with
    the default and with 8 and 16 row groups."""
    g = _gather()
    tokens, slots = 2048, 1 << 16
    gen = torch.Generator(device="cuda").manual_seed(0 if groups is None else groups)
    bank = _bounded((slots, 128), -4, 2, gen, torch.bfloat16)
    idx = torch.randint(0, slots, (tokens, 64), device="cuda", generator=gen)
    q_idx = torch.randint(0, slots, (tokens, 64), device="cuda", generator=gen)
    kw = _bounded((tokens, 64), -4, 0, gen, torch.bfloat16)
    qw = _bounded((tokens, 64), -4, 0, gen, torch.bfloat16)
    kv_pd = _bounded((tokens, 64), -4, 0, gen, torch.float32)
    geo = g.gather_geometry(128, torch.bfloat16, 64, tokens, num_sms=_num_sms())
    if groups is not None:
        geo = geo._replace(groups=groups, tpc=1, threads=16 * groups)
    for q_args, weights in ((dict(q_idx=q_idx, q_weights=qw), kw), ({}, kv_pd)):
        stash = torch.empty(tokens, 64, 128, device="cuda", dtype=torch.bfloat16)
        out, q_out = (torch.empty(tokens, 128, device="cuda") for _ in range(2))
        q = dict(q_args, q_out=q_out) if q_args else {}
        g.token_gather(bank, idx, weights, stash, out, geo, tokens=tokens, chunk=64, valid=64, **q)
        assert bits_equal(out, _order_exact(bank, idx, weights, geo.groups))
        if q_args:
            assert bits_equal(q_out, _order_exact(bank, q_idx, qw, geo.groups))
        assert bits_equal(stash, bank[idx])


@pytest.mark.parametrize("hd,dtype",
                         [(128, torch.bfloat16), (1024, torch.bfloat16), (128, torch.float32)])
def test_deterministic_and_graph_safe(hd, dtype):
    """Colliding slots, repeated runs and a CUDA-graph replay give identical bytes."""
    g = _gather()
    tokens, W = 512, 64
    d = _inputs(hd, W, W, dtype, rows=tokens, slots=97, seed=1)
    geo = g.gather_geometry(hd, dtype, W, tokens, num_sms=_num_sms())
    stash = torch.empty(tokens, W, hd, device="cuda", dtype=dtype)
    out, q_out = torch.empty(tokens, hd, device="cuda"), torch.empty(tokens, hd, device="cuda")

    def run():
        g.token_gather(d["bank"], d["idx"], d["w"], stash, out, geo, q_idx=d["q_idx"],
                       q_weights=d["q_w"], q_out=q_out, tokens=tokens, chunk=64, valid=64)
        return stash.clone(), out.clone(), q_out.clone()

    first = run()
    assert all(bits_equal(a, b) for a, b in zip(first, run()))
    for t in (stash, out, q_out):
        t.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.token_gather(d["bank"], d["idx"], d["w"], stash, out, geo, q_idx=d["q_idx"],
                       q_weights=d["q_w"], q_out=q_out, tokens=tokens, chunk=64, valid=64)
    graph.replay()
    torch.cuda.synchronize()
    assert all(bits_equal(a, b) for a, b in zip(first, (stash, out, q_out)))


@pytest.mark.slow
def test_bank_beyond_int32_elements():
    """Rows past 2**31 bank elements are addressed with 64-bit offsets."""
    g = _gather()
    hd, slots = 512, (1 << 22) + 64
    bank = torch.zeros(slots, hd, device="cuda", dtype=torch.bfloat16)
    top = torch.arange(slots - 64, slots, device="cuda")
    bank[top] = torch.randn(64, hd, device="cuda").bfloat16()
    idx = top.view(4, 16)
    w = torch.rand(4, 16, device="cuda")
    geo = g.gather_geometry(hd, torch.bfloat16, 16, 4, num_sms=_num_sms())
    stash = torch.empty(4, 16, hd, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(4, hd, device="cuda")
    g.token_gather(bank, idx, w, stash, out, geo, tokens=4, chunk=4, valid=4)
    ref, rows, _ = _reference(bank, idx, w, 0, 4, 4, 4)
    assert rel_max(ref, out) <= 1e-6 and bits_equal(stash, rows)


def test_rejects_unsupported_inputs():
    g = _gather()
    d = _inputs(64, 4, 4, torch.bfloat16, rows=64, slots=128, seed=0)
    geo = g.gather_geometry(64, torch.bfloat16, 4, 64, num_sms=_num_sms())
    stash = torch.empty(64, 4, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(64, 64, device="cuda")
    with pytest.raises(TypeError):
        g.gather_geometry(64, torch.float16, 4, 64, num_sms=_num_sms())
    with pytest.raises(RuntimeError, match="float32"):
        g.token_gather(d["bank"].half(), d["idx"], d["w"], stash.half(), out, geo,
                       tokens=64, chunk=16, valid=16)
    with pytest.raises(RuntimeError, match="chunk"):
        g.token_gather(d["bank"], d["idx"], d["w"], stash, out, geo, tokens=60, chunk=16, valid=16)
    with pytest.raises(RuntimeError, match="row0"):
        g.token_gather(d["bank"], d["idx"], d["w"], stash, out, geo, tokens=64, chunk=16,
                       valid=16, row0=16)
    with pytest.raises(RuntimeError, match="overlap"):
        bank_as_out = d["bank"].view(torch.float32).view(64, 64)
        g.token_gather(d["bank"], d["idx"], d["w"], stash, bank_as_out, geo, tokens=64, chunk=16,
                       valid=16)
    shared = torch.zeros(128 * 64 + 64 * 4 * 64 - 1024, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="overlap"):
        g.token_gather(shared[:128 * 64].view(128, 64), d["idx"], d["w"],
                       shared[-64 * 4 * 64:].view(64, 4, 64), out, geo, tokens=64, chunk=16,
                       valid=16)
    wide = torch.zeros(64, 257, device="cuda", dtype=torch.long)
    with pytest.raises(RuntimeError, match="1..256"):
        g.token_gather(d["bank"], wide, wide.bfloat16(), stash, out, geo, tokens=64, chunk=16,
                       valid=16)
