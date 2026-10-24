# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Exact top-k over all sums of product-key factors, with a sparse backward."""

import math
import operator
from collections.abc import Sequence

import torch

from .cuda.product_key_topk_anyf_cuda import MAX_FACTORS, MAX_SLOTS
from .cuda.product_key_topk_anyf_cuda import _module as _anyf_module
from .cuda.product_key_topk_cuda import _module


def _topk_function(module):
    """The autograd function of the exact top-k kernels of `module` (a cached extension loader), with a sparse backward."""

    class _ProductKeyTopK(torch.autograd.Function):
        @staticmethod
        def forward(ctx, raw, factors, k):
            rows = raw.reshape(-1, raw.shape[-1])
            if rows.stride(-1) != 1:
                rows = rows.contiguous()
            val, lin = module().topk_fwd(rows, factors, k)
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(lin)
            ctx.raw_shape = raw.shape
            ctx.factors = factors
            ctx.mark_non_differentiable(lin)
            out_shape = (*raw.shape[:-1], lin.shape[-1])
            return val.view(out_shape), lin.view(out_shape)

        @staticmethod
        def backward(ctx, grad_val, _grad_lin):
            (lin,) = ctx.saved_tensors
            grad = module().topk_bwd(lin, grad_val.reshape(lin.shape), ctx.factors)
            return grad.view(ctx.raw_shape), None, None
    return _ProductKeyTopK


_ProductKeyTopK = _topk_function(_module)          # cuda/product_key_topk_cuda.py
_ProductKeyTopKAnyF = _topk_function(_anyf_module)  # cuda/product_key_topk_anyf_cuda.py


def _bit_packed(factors: tuple[int, ...]) -> bool:
    """Whether the kernels of ``cuda/product_key_topk_cuda.py`` take ``factors``: two to four
    factors of 1 to 65535 columns, with at most 32 bits of bit-packed slot index."""
    return (2 <= len(factors) <= 4 and all(1 <= f <= 65535 for f in factors)
            and sum((f - 1).bit_length() for f in factors) <= 32)


def check_product_key_factors(factors: Sequence[int]) -> tuple[int, ...]:
    """The factor widths as a tuple of ints, or ``ValueError`` if ``product_key_topk`` cannot
    select over them: it takes 1 to ``MAX_FACTORS`` factors of at least one column each,
    addressing at most ``MAX_SLOTS`` slots with a row width of at most ``MAX_SLOTS`` (or
    any factor list of the two- to four-factor kernels)."""
    if isinstance(factors, (str, bytes)) or not isinstance(factors, Sequence):
        raise ValueError(f"product-key factors must be a sequence of ints, got {factors!r}")
    if any(isinstance(f, bool) for f in factors):
        raise ValueError(f"product-key factors must be ints, got {list(factors)}")
    try:
        dims = tuple(operator.index(f) for f in factors)
    except TypeError:
        raise ValueError(f"product-key factors must be ints, got {list(factors)}") from None
    if not 1 <= len(dims) <= MAX_FACTORS or min(dims) < 1:
        raise ValueError(f"product-key factors must be 1 to {MAX_FACTORS} positive ints, got "
                         f"{list(dims)}")
    if not _bit_packed(dims) and (math.prod(dims) > MAX_SLOTS or sum(dims) > MAX_SLOTS):
        raise ValueError(f"product-key factors {list(dims)} address {math.prod(dims)} slots "
                         f"over {sum(dims)} columns; at most {MAX_SLOTS} of each.")
    return dims


def product_key_topk(
    raw: torch.Tensor, factors: Sequence[int], k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k over all sums of one value from each product-key factor, for any number
    of factors.

    raw [..., sum(factors)] (CUDA, bfloat16 or float32) holds the F >= 1 factors side by side,
    each at least one column wide (see ``check_product_key_factors`` for the limits). Slot
    (c_0, ..., c_{F-1}) scores the float32 sum of its factor values, added in factor order;
    with bfloat16 input and more than two factors, each partial sum is rounded to bfloat16
    before the next factor is added. With one factor, slot c_0 scores raw[..., c_0]. Returns
    (val, lin), each [..., kk] with kk = min(k, prod(factors)) and 1 <= k <= 256:
    lin = (..(c_0 * factors[1] + c_1) * ..) * factors[F-1] + c_{F-1} (int64), strictly
    ascending, and val = the slot's score rounded to raw.dtype. val is differentiable with
    respect to raw; the backward is deterministic.

    Selection rule: scores compare as float32 numbers, with -0 below +0 and NaN counted as
    -inf. Each factor is ranked by value, larger column first on ties; one factor selects its
    first kk ranks. Stage f = 1..F-1 keeps the best c = min(k, n * m) of the pairs (i, j) of a
    rank i < n = min(k, prod(factors[:f])) of the previous stage and a rank j < m =
    min(k, factors[f]) of factor f with (i + 1) * (j + 1) <= c, exact ties going to the
    larger i + j, then the larger i. Each pair left out has c pairs of lower or equal ranks
    scoring at least as much, so the selected scores are an exact top-k. With two factors,
    bfloat16 input and its float32 upcast select the same slots. The kernels never
    synchronize and need no warmup before CUDA-graph capture.

    Two to four factors of at most 65535 columns and 32 bits of bit-packed slot index run on
    the kernels of ``cuda/product_key_topk_cuda.py``; every other factor list runs on those
    of ``cuda/product_key_topk_anyf_cuda.py``, which apply the same rule.
    """
    factors = tuple(operator.index(f) for f in factors)
    if _bit_packed(factors):
        return _ProductKeyTopK.apply(raw, factors, operator.index(k))
    return _ProductKeyTopKAnyF.apply(raw, factors, operator.index(k))


def _product_key_topk_anyf(raw, factors, k):
    """``product_key_topk`` on the any-factor kernels whatever the factor list (tests)."""
    factors = tuple(operator.index(f) for f in factors)
    return _ProductKeyTopKAnyF.apply(raw, factors, operator.index(k))
