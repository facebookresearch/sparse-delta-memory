# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Product-key top-k (``product_key_topk``) against a brute-force reference.

The reference implements the documented rule directly: rank each factor by (value, column)
descending, then for each stage enumerate every pair (i, j) of ranks with
(i + 1) * (j + 1) <= c and keep the best c by (score, i + j, i), scores comparing as float32
with -0 below +0 and NaN as -inf. It shares nothing with the kernel's bisection, packing or
sorting networks, so bitwise agreement for every k checks the kernel against the rule. The
two-factor sides cover every factor layout (1, 2, 4, 8 and 16 keys per lane, streamed tiles, a
partial last tile, the largest side); the row counts leave the last block of 8 rows partial.

Factor lists outside the two- to four-factor kernels (one factor, five or more, wider
factors) run on the any-factor kernels; they are checked against the same reference, against
an enumeration of every slot of the full product (the selected scores are the k best of all
slot scores, and the selected slots are those k whenever the k-th and (k + 1)-th best scores
differ), and, forced onto the factor lists of the two- to four-factor kernels, against those
kernels bit for bit, forward and backward.
"""
import math
import os
import subprocess
import sys
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from sdm_testlib import bits_equal, requires_cuda

pytestmark = requires_cuda

SIDES = (1, 2, 3, 8, 16, 17, 33, 65, 100, 129, 256, 257, 512, 1024, 4097, 65535)
MULTI = ((2, 2, 2), (3, 5, 7), (7, 1, 3), (2, 2, 2, 2), (15, 16, 16, 15), (16, 16, 16, 16),
         (12, 12, 20, 20), (32, 32, 32), (40, 3, 5), (300, 2, 2), (256, 256, 256, 256),
         (1, 65535, 65535))
# Factor lists of the any-factor kernels: one factor (streamed past 512 columns, beyond 16-bit
# column ids), five to sixteen factors with widths below k, width-1 factors and wider factors.
ANY = ((1,), (2,), (3,), (17,), (100,), (256,), (257,), (512,), (513,), (1000,), (4097,),
       (65536,), (70000,), (5, 5, 4, 4, 4, 4, 3, 3), (2, 2, 2, 2, 2), (3, 1, 4, 1, 5),
       (7, 1, 3, 2, 2), (1, 1, 1, 1, 1, 9), (4, 4, 4, 4, 4, 4), (16, 16, 2, 2, 2),
       (300, 2, 2, 2, 2), (3, 3, 3, 3, 3, 3, 3, 3), (2,) * 10, (40, 3, 5, 2, 2, 2),
       (65536, 2), (1, 70000), (100000, 3), (1025, 1025, 1025), (1,) * 7)
# Full-product enumerations: F = 1, 2, 3, 4, 5, 6 and 8, mixed widths, widths below k and
# width-1 factors, up to the 57,600 slots of the paper's configurations.
BRUTE = ((1,), (7,), (300,), (57600,), (240, 240), (60, 60), (30, 30, 64), (40, 40, 36),
         (15, 16, 16, 15), (5, 5, 4, 4, 4, 4, 3, 3), (6, 5, 4, 4, 3, 2), (8, 6, 5, 4, 3),
         (3, 1, 4, 1, 5, 9, 2, 6), (2, 3, 1, 2, 3))
BRUTE_K = (1, 2, 3, 7, 16, 63, 64, 65, 128, 200, 256)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
KINDS = ("normal", "integer", "zeros", "special", "gaps")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def topk(raw, factors, k):
    from sparse_delta_memory.product_key_topk import product_key_topk

    return product_key_topk(raw, factors, k)


def topk_anyf(raw, factors, k):
    from sparse_delta_memory.product_key_topk import _product_key_topk_anyf

    return _product_key_topk_anyf(raw, factors, k)


def make_raw(factors, dtype, kind, rows, seed=0):
    """normal: N(0, 1). integer: values in [-4, 4], so exact ties everywhere. zeros: +-0 and
    +-1 only. special: +-NaN, +-inf, +-0, huge values whose sums overflow and subnormals
    sprinkled in, plus an all-NaN, an all--inf and an all-signed-zero row. gaps: the first
    factor is 2**80 times larger or smaller than the others (alternating by row), so they
    vanish in the sum or dominate it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (rows, sum(factors))
    x = torch.randn(shape, device="cuda", generator=g)
    if kind == "integer":
        x = torch.randint(-4, 5, shape, device="cuda", generator=g).float()
    elif kind == "zeros":
        values = torch.tensor([-1.0, -0.0, 0.0, 1.0], device="cuda")
        x = values[torch.randint(0, 4, shape, device="cuda", generator=g)]
    elif kind == "special":
        u = torch.rand(shape, device="cuda", generator=g)
        specials = (float("nan"), -float("nan"), float("inf"), float("-inf"), 0.0, -0.0,
                    3e38, 1e-40)
        for i, v in enumerate(specials):
            x[(u >= 0.04 * i) & (u < 0.04 * (i + 1))] = v
        x[0], x[1] = float("nan"), float("-inf")
        x[2, ::2], x[2, 1::2] = 0.0, -0.0
    elif kind == "gaps":
        x[0::2, :factors[0]] *= 2.0**40
        x[0::2, factors[0]:] *= 2.0**-40
        x[1::2, :factors[0]] *= 2.0**-40
        x[1::2, factors[0]:] *= 2.0**40
    return x.to(dtype)


def ordered_key(x):
    """int64 keys ordering float32 values, -0 below +0 and NaN as -inf."""
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = torch.where(x.isnan(), torch.full_like(bits, 0xFF800000), bits)
    return torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)


def rankings(raw, factors):
    """Per factor: (values, columns) of each row ordered by (value, column) descending."""
    out = []
    for x in raw.float().split(list(factors), dim=1):
        col = torch.arange(x.shape[1], device=x.device)
        order = ((ordered_key(x) << 31) | col).sort(dim=1, descending=True).indices
        out.append((x.gather(1, order), order))
    return out


def reference_topk(raw, factors, k, ranked):
    """(val, lin) of the documented rule, from the factor rankings of raw."""
    val, lin = (t[:, :min(k, factors[0])] for t in ranked[0])
    for f in range(1, len(factors)):
        fval, fcol = (t[:, :min(k, factors[f])] for t in ranked[f])
        n, m = val.shape[1], fval.shape[1]
        c = min(k, n * m)
        i = torch.arange(n, device=raw.device).repeat_interleave(m)
        j = torch.arange(m, device=raw.device).repeat(n)
        keep = (i + 1) * (j + 1) <= c
        i, j = i[keep], j[keep]
        score = val[:, i] + fval[:, j]
        key = (ordered_key(score) << 16) | ((i + j) << 8 | i)
        order = key.sort(dim=1, descending=True).indices[:, :c]
        val = score.gather(1, order).to(raw.dtype).float()
        lin = lin.gather(1, i[order]) * factors[f] + fcol.gather(1, j[order])
    lin, perm = lin.sort(dim=1)
    return val.gather(1, perm).to(raw.dtype), lin


def _check_every_k(raw, factors):
    ranked = rankings(raw, factors)
    slots = 1
    for w in factors:
        slots *= w
    for k in range(1, min(256, slots) + 1):
        val, lin = topk(raw, factors, k)
        ref_val, ref_lin = reference_topk(raw, factors, k, ranked)
        assert bits_equal(lin, ref_lin), f"k={k}: slots differ"
        assert bits_equal(val, ref_val), f"k={k}: values differ"


def full_scores(raw, factors):
    """float32 score of every slot, [rows, prod(factors)] in slot-id order: the factor values
    added in factor order, each partial sum rounded to raw.dtype from the third factor on."""
    parts = raw.float().split(list(factors), dim=1)
    s = parts[0]
    for f, x in enumerate(parts[1:], start=1):
        if f > 1:
            s = s.to(raw.dtype).float()
        s = (s.unsqueeze(-1) + x.unsqueeze(1)).flatten(1)
    return s


# ================================================================== forward selection
def test_the_tie_order_on_a_worked_example():
    # Both factors rank column 1 before column 0 (equal values: larger column first). At k = 2
    # the candidates are the rank pairs (0, 0), (0, 1) and (1, 0), all scoring 2; the tie goes
    # to the larger i + j, then the larger i: (1, 0) = slot (0, 1) and (0, 1) = slot (1, 0).
    raw = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]], device="cuda")
    assert topk(raw, (3, 3), 1)[1].tolist() == [[4]]
    assert topk(raw, (3, 3), 2)[1].tolist() == [[1, 3]]
    assert topk(raw, (3, 3), 4)[1].tolist() == [[0, 1, 3, 4]]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("side", SIDES)
def test_every_k_matches_the_brute_force_reference(side, dtype, kind):
    _check_every_k(make_raw((side, side), dtype, kind, rows=3 if side == 65535 else 9), (side, side))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors", MULTI, ids=["x".join(map(str, f)) for f in MULTI])
def test_more_factors_match_the_brute_force_reference(factors, dtype, kind):
    rows = 3 if max(factors) > 1000 else 9
    _check_every_k(make_raw(factors, dtype, kind, rows=rows, seed=7), factors)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors", ANY, ids=["x".join(map(str, f)) for f in ANY])
def test_any_factor_list_matches_the_brute_force_reference(factors, dtype, kind):
    rows = 3 if max(factors) > 4096 else 9
    _check_every_k(make_raw(factors, dtype, kind, rows=rows, seed=13), factors)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors", BRUTE, ids=["x".join(map(str, f)) for f in BRUTE])
def test_selection_is_an_exact_top_k_of_the_full_product(factors, dtype, kind):
    """Against every slot score of the full product: the selected scores are the kk best of
    all of them (as multisets), the selected slots are exactly the kk best wherever the kk-th
    and (kk + 1)-th best scores differ, and on rows where they tie the documented rule picks
    the slots (the reference above)."""
    raw = make_raw(factors, dtype, kind, rows=16, seed=21)
    key = ordered_key(full_scores(raw, factors))
    slots = key.shape[1]
    ranked = rankings(raw, factors)
    best = key.sort(dim=1, descending=True).values
    for k in BRUTE_K:
        val, lin = topk(raw, factors, k)
        kk = min(k, slots)
        assert lin.shape == (raw.shape[0], kk)
        assert bool((lin[:, 1:] > lin[:, :-1]).all()) and bool((lin >= 0).all())
        got = key.gather(1, lin).sort(dim=1, descending=True).values
        assert torch.equal(got, best[:, :kk]), f"k={k}: not the k best scores"
        assert bits_equal(val, full_scores(raw, factors).gather(1, lin).to(dtype)), f"k={k}"
        if kk < slots:
            unique = best[:, kk - 1] != best[:, kk]
            top = key.topk(kk, dim=1).indices.sort(dim=1).values
            assert torch.equal(lin[unique], top[unique]), f"k={k}: not the k best slots"
        ref_val, ref_lin = reference_topk(raw, factors, k, ranked)
        assert bits_equal(lin, ref_lin) and bits_equal(val, ref_val), f"k={k}: tie order"


@pytest.mark.parametrize("kind", ("normal", "integer", "special"))
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors", [(3,) * 16 + (16,), (1,) * 500 + (2,) * 12],
                         ids=["f17", "f512"])
def test_many_factors_match_the_brute_force_reference(factors, dtype, kind):
    raw = make_raw(factors, dtype, kind, rows=9, seed=29)
    ranked = rankings(raw, factors)
    for k in (1, 2, 7, 64, 100, 256):
        val, lin = topk(raw, factors, k)
        ref_val, ref_lin = reference_topk(raw, factors, k, ranked)
        assert bits_equal(lin, ref_lin) and bits_equal(val, ref_val), f"k={k}"


def test_wide_factors_past_24_bit_columns():
    """Columns at and above 2**24 in a factor after the first (the backward sorts them with
    64-bit keys): forward against the reference, backward against the scatter reference."""
    factors = (2, 2**24 + 5)
    g = torch.Generator(device="cuda").manual_seed(31)
    raw = torch.randn(3, sum(factors), device="cuda", generator=g)
    hot = torch.tensor([7, 2**24 - 1, 2**24, 2**24 + 3, 2**24 + 4], device="cuda")
    raw[:, 2 + hot] += torch.arange(10, 15, device="cuda", dtype=torch.float32)
    raw = raw.requires_grad_(True)
    ranked = rankings(raw.detach(), factors)
    for k in (1, 4, 9, 64):
        val, lin = topk(raw, factors, k)
        ref_val, ref_lin = reference_topk(raw.detach(), factors, k, ranked)
        assert bits_equal(lin, ref_lin) and bits_equal(val, ref_val), f"k={k}"
        assert bool((lin % factors[1] >= 2**24).any())
        gv = torch.randn(val.shape, device="cuda", generator=g)
        (grad,) = torch.autograd.grad(val, raw, gv)
        ref = _scatter(lin, gv.double(), factors)
        bound = _scatter(lin, gv.double().abs(), factors) * lin.shape[1] * 2.0**-24
        assert bool(((grad.double() - ref).abs() <= bound).all()), f"k={k}"


def test_one_factor_of_the_largest_width():
    """One factor of 2**31 - 1 columns (the largest row): the selection, and the backward's
    zero fill and writes up to the last column."""
    width = 2**31 - 1
    raw = torch.zeros(1, width, device="cuda", dtype=torch.bfloat16)
    cols = torch.tensor([5, width - 40, width - 2, width - 1], device="cuda")
    raw[0, cols] = torch.tensor([3.0, 1.0, 4.0, 2.0], device="cuda", dtype=torch.bfloat16)
    raw.requires_grad_(True)
    val, lin = topk(raw, (width,), 4)
    assert lin.tolist() == [cols.tolist()]
    assert val.float().tolist() == [[3.0, 1.0, 4.0, 2.0]]
    gv = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=torch.bfloat16)
    (grad,) = torch.autograd.grad(val, raw, gv)
    assert grad[0, cols].float().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert int(torch.count_nonzero(grad)) == 4


# Factor lists of the two- to four-factor kernels, run on both kernel families (those with at
# most 2**31 - 1 slots, the limit of the any-factor kernels).
SHARED = tuple(f for f in tuple((s, s) for s in SIDES) + MULTI if math.prod(f) < 2**31)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors", SHARED, ids=["x".join(map(str, f)) for f in SHARED])
def test_any_factor_kernels_match_the_two_to_four_factor_kernels(factors, dtype, kind):
    raw = make_raw(factors, dtype, kind, rows=3 if max(factors) > 1000 else 9,
                   seed=17).requires_grad_(True)
    g = torch.Generator(device="cuda").manual_seed(18)
    slots = 1
    for w in factors:
        slots *= w
    for k in range(1, min(256, slots) + 1):
        val, lin = topk(raw, factors, k)
        any_val, any_lin = topk_anyf(raw, factors, k)
        assert bits_equal(lin, any_lin) and bits_equal(val, any_val), f"k={k}: forward"
        if k in (1, 2, 5, 16, 17, 33, 64, 100, 128, 129, 200, 255, 256) or k == slots:
            gv = torch.randn(val.shape, device="cuda", generator=g).to(dtype)
            (grad,) = torch.autograd.grad(val, raw, gv)
            (any_grad,) = torch.autograd.grad(any_val, raw, gv)
            assert bits_equal(grad, any_grad), f"k={k}: backward"


OLD_FWD, OLD_BWD = "product_key_topk_fwd_kernel", "product_key_topk_bwd_kernel"
ANY_FWD, ONE_FWD, ANY_BWD = ("product_key_topk_anyf_fwd_kernel", "product_key_topk_one_fwd_kernel",
                             "product_key_topk_anyf_bwd_kernel")


@pytest.mark.parametrize("side", (16, 256, 1024))
@pytest.mark.parametrize("kind", KINDS)
def test_bf16_and_its_fp32_upcast_select_the_same_slots(side, kind):
    raw = make_raw((side, side), torch.bfloat16, kind, rows=64, seed=1)
    for k in (1, 5, 64, 128, 200, 256):
        v16, l16 = topk(raw, (side, side), k)
        v32, l32 = topk(raw.float(), (side, side), k)
        assert bits_equal(l16, l32), f"k={k}"
        assert bits_equal(v16, v32.to(torch.bfloat16)), f"k={k}"


def test_leading_dims_strides_and_empty_input():
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(2, 3, 5, 64, device="cuda", generator=g)
    val, lin = topk(x, (32, 32), 20)
    flat_val, flat_lin = topk(x.reshape(-1, 64), (32, 32), 20)
    assert val.shape == lin.shape == (2, 3, 5, 20)
    assert bits_equal(val.reshape(-1, 20), flat_val) and bits_equal(lin.reshape(-1, 20), flat_lin)
    wide = torch.randn(40, 100, device="cuda", generator=g)
    for view in (wide[:, 10:74], wide[:, :64].t().contiguous().t()):
        for factors in ((32, 32), (16, 16, 16, 16)):
            v, l = topk(view, factors, 33)
            rv, rl = topk(view.contiguous(), factors, 33)
            assert bits_equal(v, rv) and bits_equal(l, rl)
    v, l = topk(torch.randn(0, 64, device="cuda"), (32, 32), 20)
    assert v.shape == l.shape == (0, 20)


@pytest.mark.parametrize("factors", [(64,), (3,) * 16 + (16,), (2, 2, 2, 2, 56), (60, 1, 1, 1, 1),
                                     (32, 8, 8, 8, 8)],
                         ids=["f1", "f17", "f5", "f5_ones", "f5_mixed"])
def test_any_factor_leading_dims_strides_and_empty_input(factors):
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(2, 3, 5, 64, device="cuda", generator=g)
    val, lin = topk(x, factors, 20)
    flat_val, flat_lin = topk(x.reshape(-1, 64), factors, 20)
    assert val.shape == lin.shape == (2, 3, 5, 20)
    assert bits_equal(val.reshape(-1, 20), flat_val) and bits_equal(lin.reshape(-1, 20), flat_lin)
    wide = torch.randn(40, 100, device="cuda", generator=g)
    for view in (wide[:, 10:74], wide[:, :64].t().contiguous().t()):
        v, l = topk(view, factors, 33)
        rv, rl = topk(view.contiguous(), factors, 33)
        assert bits_equal(v, rv) and bits_equal(l, rl)
    v, l = topk(torch.randn(0, 64, device="cuda"), factors, 20)
    assert v.shape == l.shape == (0, 20)


@pytest.mark.parametrize(
    "raw,factors,k,error,match",
    [
        (lambda: torch.randn(4, 32, device="cuda").half(), (16, 16), 4, RuntimeError,
         "bfloat16 or float32"),
        (lambda: torch.randn(4, 32), (16, 16), 4, RuntimeError, "CUDA"),
        (lambda: torch.randn(4, 31, device="cuda"), (16, 16), 4, RuntimeError, "row width"),
        (lambda: torch.randn(4, 11, device="cuda"), (2, 2, 2, 2, 2), 4, RuntimeError,
         "row width"),
        (lambda: torch.randn(4, 32, device="cuda"), (), 4, RuntimeError, "1 to 512 factors"),
        (lambda: torch.randn(4, 513, device="cuda"), (1,) * 513, 4, RuntimeError,
         "1 to 512 factors"),
        (lambda: torch.randn(1, 2 * 65536, device="cuda"), (65536, 65536), 4, RuntimeError,
         "slots"),
        (lambda: torch.randn(4, 32, device="cuda"), (0, 32), 4, RuntimeError, "factor widths"),
        (lambda: torch.randn(4, 32, device="cuda"), (32, 0, 1), 4, RuntimeError,
         "factor widths"),
        (lambda: torch.randn(1, 131072, device="cuda"), (65535, 65535, 2), 4, RuntimeError,
         "slots"),
        (lambda: torch.randn(4, 32, device="cuda"), (16, 16), 0, RuntimeError, "k must be"),
        (lambda: torch.randn(4, 32, device="cuda"), (16, 16), 257, RuntimeError, "k must be"),
        (lambda: torch.randn(4, 32, device="cuda"), (32,), 257, RuntimeError, "k must be"),
        (lambda: torch.randn(4, 10, device="cuda"), (2,) * 5, 0, RuntimeError, "k must be"),
        (lambda: torch.randn(4, 32, device="cuda").half(), (32,), 4, RuntimeError,
         "bfloat16 or float32"),
        (lambda: torch.randn(4, 32), (32,), 4, RuntimeError, "CUDA"),
        (lambda: torch.randn(4, 32, device="cuda"), (16, 16), 1.5, TypeError, None),
        (lambda: torch.randn(4, 32, device="cuda"), (16.0, 16), 4, TypeError, None),
    ],
    ids=["fp16", "cpu", "width_mismatch", "width_mismatch_anyf", "no_factors",
         "factors_513", "width_65536", "width_0", "width_0_anyf", "index_bits", "k0", "k257",
         "k257_anyf", "k0_anyf", "fp16_anyf", "cpu_anyf", "k_float", "factor_float"],
)
def test_invalid_arguments_raise(raw, factors, k, error, match):
    with pytest.raises(error, match=match):
        topk(raw(), factors, k)


# =========================================================================== backward
def _scatter(lin, gv, factors):
    """float64 scatter-add of gv onto every factor column of each slot."""
    out = torch.zeros(lin.shape[0], sum(factors), dtype=torch.float64, device=lin.device)
    rest, offset = lin, sum(factors)
    for w in reversed(factors):
        offset -= w
        out.scatter_add_(1, offset + rest % w, gv)
        rest = rest // w
    return out


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors,k", [((3, 3), 9), ((16, 16), 7), ((100, 100), 128),
                                       ((256, 256), 1), ((256, 256), 64), ((256, 256), 256),
                                       ((1024, 1024), 100), ((4097, 4097), 200),
                                       ((15, 16, 16, 15), 64), ((3, 5, 7), 100),
                                       ((32, 32, 32), 200), ((256, 256, 256, 256), 256)])
def test_backward_is_deterministic_and_matches_a_scatter_reference(factors, k, dtype):
    raw = make_raw(factors, dtype, "normal", rows=64, seed=3).requires_grad_(True)
    g = torch.Generator(device="cuda").manual_seed(4)
    val, lin = topk(raw, factors, k)
    gv = torch.randn(val.shape, device="cuda", generator=g).to(dtype)
    grads = []
    for _ in range(3):
        raw.grad = None
        topk(raw, factors, k)[0].backward(gv)
        grads.append(raw.grad)
    assert raw.grad.dtype == dtype
    assert all(bits_equal(grads[0], other) for other in grads[1:]), "backward is not deterministic"
    ref = _scatter(lin, gv.double(), factors)
    touched = _scatter(lin, torch.ones_like(lin, dtype=torch.float64), factors) > 0
    # fp32 accumulation of <= kk terms, then one rounding to the output dtype.
    bound = _scatter(lin, gv.double().abs(), factors) * lin.shape[1] * 2.0**-24
    if dtype == torch.bfloat16:
        bound = bound + ref.abs() * 2.0**-8
    got = grads[0].double()
    assert bool(((got - ref).abs() <= bound).all()), "gradient differs from the scatter reference"
    assert bits_equal(grads[0][~touched], torch.zeros_like(grads[0][~touched]))


def _explicit_sum_grad(raw, lin, gv, factors):
    """float64 autograd gradient of sum(gv * score) with respect to raw, where score is the
    explicit sum of the factor values of each selected slot (decoded from lin)."""
    x = raw.detach().double().requires_grad_(True)
    cols, rest = [], lin
    for w in reversed(factors):
        cols.append(rest % w)
        rest = rest // w
    parts = x.split(list(factors), dim=1)
    score = sum(p.gather(1, c) for p, c in zip(parts, reversed(cols)))
    (score * gv.double()).sum().backward()
    return x.grad


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("factors,k", [((1,), 1), ((7,), 7), ((1000,), 64), ((65536,), 256),
                                       ((70000,), 100), ((5, 5, 4, 4, 4, 4, 3, 3), 64),
                                       ((5, 5, 4, 4, 4, 4, 3, 3), 256), ((2, 2, 2, 2, 2), 32),
                                       ((3, 1, 4, 1, 5), 60), ((4,) * 6, 200),
                                       ((1, 1, 1, 1, 1, 9), 9), ((1025, 1025, 1025), 128),
                                       ((65536, 2), 77), ((3,) * 8, 129)])
def test_any_factor_backward_is_deterministic_and_matches_a_float64_autograd_reference(
        factors, k, dtype):
    raw = make_raw(factors, dtype, "normal", rows=64, seed=3).requires_grad_(True)
    g = torch.Generator(device="cuda").manual_seed(4)
    val, lin = topk(raw, factors, k)
    gv = torch.randn(val.shape, device="cuda", generator=g).to(dtype)
    grads = []
    for _ in range(3):
        raw.grad = None
        topk(raw, factors, k)[0].backward(gv)
        grads.append(raw.grad)
    assert raw.grad.dtype == dtype
    assert all(bits_equal(grads[0], other) for other in grads[1:]), "backward is not deterministic"
    ref = _explicit_sum_grad(raw, lin, gv, factors)
    scatter = _scatter(lin, gv.double(), factors)
    assert bool(((ref - scatter).abs() <= 1e-12 * scatter.abs().max()).all())
    touched = _scatter(lin, torch.ones_like(lin, dtype=torch.float64), factors) > 0
    bound = _scatter(lin, gv.double().abs(), factors) * lin.shape[1] * 2.0**-24
    if dtype == torch.bfloat16:
        bound = bound + ref.abs() * 2.0**-8
    got = grads[0].double()
    assert bool(((got - ref).abs() <= bound).all()), "gradient differs from the float64 reference"
    assert bits_equal(grads[0][~touched], torch.zeros_like(grads[0][~touched]))


def test_backward_reads_strided_and_broadcast_cotangents():
    raw = make_raw((256, 256), torch.float32, "normal", rows=64, seed=5).requires_grad_(True)
    val, _ = topk(raw, (256, 256), 64)
    wide = torch.randn(64, 128, device="cuda")
    for cot in (wide[:, ::2], torch.ones((), device="cuda").expand(val.shape)):
        grads = []
        for c in (cot, cot.contiguous()):
            raw.grad = None
            topk(raw, (256, 256), 64)[0].backward(c)
            grads.append(raw.grad)
        assert bits_equal(grads[0], grads[1])


# ================================================================ recompute and graphs
@pytest.mark.parametrize("selective", [False, True], ids=["checkpoint", "sac"])
def test_activation_checkpoint_recompute_selects_the_same_slots(selective):
    raw = make_raw((256, 256), torch.bfloat16, "integer", rows=128, seed=6).requires_grad_(True)
    w = torch.randn(128, 64, device="cuda").to(torch.bfloat16)
    seen = []

    def loss_fn(x):
        val, lin = topk(x, (256, 256), 64)
        seen.append(lin)
        return (val * w).float().sum()

    kwargs = {"use_reentrant": False}
    if selective:
        policy = lambda ctx, op, *args, **kw: CheckpointPolicy.PREFER_RECOMPUTE  # noqa: E731
        kwargs["context_fn"] = partial(create_selective_checkpoint_contexts, policy)
    checkpoint(loss_fn, raw, **kwargs).backward()
    assert len(seen) == 2 and bits_equal(seen[0], seen[1])
    recomputed = raw.grad
    raw.grad = None
    loss_fn(raw).backward()
    assert bits_equal(raw.grad, recomputed)


_GRAPH_SCRIPT = r"""
import torch
from sparse_delta_memory.product_key_topk import product_key_topk

FACTORS = __FACTORS__
raw = torch.randn(64, sum(FACTORS), device="cuda").to(torch.bfloat16).requires_grad_(True)
gv = torch.randn(64, 64, device="cuda").to(torch.bfloat16)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):  # the first call of the process, extension load included
    val, lin = product_key_topk(raw, FACTORS, 64)
    (grad,) = torch.autograd.grad(val, raw, gv)
for _ in range(2):
    with torch.no_grad():
        raw.copy_(torch.randn(64, sum(FACTORS), device="cuda"))
    graph.replay()
    ref_val, ref_lin = product_key_topk(raw, FACTORS, 64)
    (ref_grad,) = torch.autograd.grad(ref_val, raw, gv)
    for got, ref in ((val, ref_val), (lin, ref_lin), (grad, ref_grad)):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
print("GRAPH_OK")
"""


@pytest.mark.parametrize("factors", [(256, 256), (5, 5, 4, 4, 4, 4, 3, 3), (4096,)],
                         ids=["f2", "f8", "f1"])
def test_graph_capture_needs_no_warmup(factors):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([_ROOT, os.environ.get("PYTHONPATH", "")]))
    script = _GRAPH_SCRIPT.replace("__FACTORS__", repr(tuple(factors)))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=900, env=env)
    assert r.returncode == 0 and "GRAPH_OK" in r.stdout, r.stderr[-3000:]


def test_backward_validates_the_factor_widths():
    """The backward checks the factor list it gets too (it is not given the input row)."""
    from sparse_delta_memory.cuda.product_key_topk_anyf_cuda import _module

    lin = torch.zeros(2, 1, dtype=torch.long, device="cuda")
    gv = torch.ones(2, 1, device="cuda")
    for factors, match in (([2**31 - 1, 1, 1], "columns"), ([65536, 65536], "slots"),
                           ([0, 4], "factor widths"), ([], "1 to 512 factors")):
        with pytest.raises(RuntimeError, match=match):
            _module().topk_bwd(lin, gv, factors)
