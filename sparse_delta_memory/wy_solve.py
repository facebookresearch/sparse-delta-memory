# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# The 16 x 16 tile scheme of the level-1 kernel is adapted from the solve_tril kernel of
# flash-linear-attention (https://github.com/fla-org/flash-linear-attention), which is
# distributed under the MIT License:
#
#   Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
"""Per-chunk WY solve: X = (I + L)^-1 and QX = QK @ X.

For each chunk of C tokens, ``L = tril(beta[:, None] * A, -1)`` is the gated strictly lower
key-key matrix and ``QK`` the causal query-key matrix (diagonal included). ``wy_solve``
returns the unit lower-triangular ``X`` and ``QX``, both dense row-major [N, C, C] float32
with +0.0 above the diagonal. Only the strict lower triangle of ``A`` and the lower
triangle of ``QK`` are read, so the rest may hold anything.

Level 1: one program per 64 x 64 diagonal block (16 x 16 or 32 x 32 when C <= 32). Its
16 x 16 diagonal tiles are inverted by forward substitution and the tiles below them by
matmuls. The same program emits the diagonal blocks of QX and the zeros of its block row.

Level 2 (C > 64): one program per 64-column panel J walks down the blocks below the
diagonal, X[i, J] = -X[i, i] @ sum_{k=J}^{i-1} L[i, k] X[k, J], and accumulates
QX[i, J] = sum_{k=J}^{i} QK[i, k] X[k, J] from the same loads of X. Its matmuls take
16-wide slices of the reduction dimension from memory (a full-float32 dot keeps whole
operand rows in registers, so wider operands spill); X[i, J] is staged in its own output
block so that it can be read back that way.

Rows and columns >= C load as zero, i.e. identity rows of I + L. The leading block of a
lower-triangular inverse is the inverse of the leading block, so any C is exact.

Every dot is full float32 whatever torch.backends.cuda.matmul.allow_tf32 says: the inverse
amplifies input rounding. Launch configurations are pinned, not autotuned: the warp count
changes the rounding, so the table is part of the numerical contract, and a fixed launch
is capture-safe and gives the same bits for a chunk at any N.

The 16 x 16 tile scheme is adapted from the ``solve_tril`` kernel of flash-linear-attention
(MIT License, notice above).
"""

import torch
import triton
import triton.language as tl

from .cuda._device import tbind

# Pinned (num_warps, num_stages) of the two kernels.
_DIAG_CONFIG = (1, 2)
_OFFDIAG_CONFIG = (2, 3)


@triton.jit
def _mm(a, b, acc=None):
    return tl.dot(a, b, acc, input_precision="ieee")


@triton.jit
def _load(base, rows, cols, LIM, C: tl.constexpr):
    """base[rows, cols] with row pitch C; zero outside the leading [LIM, LIM] block."""
    return tl.load(base + rows[:, None] * C + cols[None, :],
                   mask=(rows < LIM)[:, None] & (cols < LIM)[None, :], other=0.0)


@triton.jit
def _store(base, v, rows, cols, LIM, C: tl.constexpr):
    tl.store(base + rows[:, None] * C + cols[None, :], v,
             mask=(rows < LIM)[:, None] & (cols < LIM)[None, :])


@triton.jit
def _tile(base, r0, c0, LIM, C: tl.constexpr):
    o = tl.arange(0, 16)
    return _load(base, r0 + o, c0 + o, LIM, C)


@triton.jit
def _put(base, v, r0, c0, LIM, C: tl.constexpr):
    o = tl.arange(0, 16)
    _store(base, v, r0 + o, c0 + o, LIM, C)


@triton.jit
def _band_beta(beta, r0, LIM):
    o = tl.arange(0, 16)
    return tl.load(beta + r0 + o, mask=r0 + o < LIM, other=0.0)[:, None]


@triton.jit
def _lower(a, beta, r0, c0, LIM, C: tl.constexpr):
    """beta[:, None] * A on the 16 x 16 tile at (r0, c0)."""
    return _tile(a, r0, c0, LIM, C) * _band_beta(beta, r0, LIM)


@triton.jit
def _subst_start(a, beta, r0, LIM, C: tl.constexpr):
    """-L of the diagonal tile at r0: rows 0 and 1 of (I + L)^-1 - I are already final."""
    o = tl.arange(0, 16)
    return -tl.where(o[:, None] > o[None, :], _lower(a, beta, r0, r0, LIM, C), 0.0)


@triton.jit
def _subst_row(b, a, beta, r0, i, LIM, C: tl.constexpr):
    """Forward substitution of row i of the diagonal tile at r0."""
    o = tl.arange(0, 16)
    r = r0 + i
    row = tl.load(a + r * C + r0 + o, mask=(r < LIM) & (r0 + o < LIM), other=0.0)
    row = -tl.where(r0 + o < r, row * tl.load(beta + r, mask=r < LIM, other=0.0), 0.0)
    row += tl.sum(row[:, None] * b, 0)
    return tl.where((o == i)[:, None], row, b)


@triton.jit
def _qk_diag(q, r0, LIM, C: tl.constexpr):
    o = tl.arange(0, 16)
    return tl.where(o[:, None] >= o[None, :], _tile(q, r0, r0, LIM, C), 0.0)


@triton.jit
def _wy_diag_kernel(A, Beta, QK, X, QX, C: tl.constexpr, BD: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    off = tl.program_id(1) * BD
    LIM = C - off
    blk = n * C * C + off * C + off
    a, q, x, qx = A + blk, QK + blk, X + blk, QX + blk
    beta = Beta + n * C + off

    # Diagonal tiles; the independent substitutions are interleaved for latency hiding.
    x11 = _subst_start(a, beta, 0, LIM, C)
    if BD >= 32:
        x22 = _subst_start(a, beta, 16, LIM, C)
    if BD == 64:
        x33 = _subst_start(a, beta, 32, LIM, C)
        x44 = _subst_start(a, beta, 48, LIM, C)
    for i in range(2, 16):
        x11 = _subst_row(x11, a, beta, 0, i, LIM, C)
        if BD >= 32:
            x22 = _subst_row(x22, a, beta, 16, i, LIM, C)
        if BD == 64:
            x33 = _subst_row(x33, a, beta, 32, i, LIM, C)
            x44 = _subst_row(x44, a, beta, 48, i, LIM, C)
    o = tl.arange(0, 16)
    eye = (o[:, None] == o[None, :]).to(tl.float32)

    x11 += eye
    _put(x, x11, 0, 0, LIM, C)
    _put(qx, _mm(_qk_diag(q, 0, LIM, C), x11), 0, 0, LIM, C)
    if BD >= 32:
        x22 += eye
        x21 = -_mm(_mm(x22, _lower(a, beta, 16, 0, LIM, C)), x11)
        _put(x, x21, 16, 0, LIM, C)
        _put(x, x22, 16, 16, LIM, C)
        q22 = _qk_diag(q, 16, LIM, C)
        _put(qx, _mm(q22, x21, _mm(_tile(q, 16, 0, LIM, C), x11)), 16, 0, LIM, C)
        _put(qx, _mm(q22, x22), 16, 16, LIM, C)
    if BD == 64:
        # The association of each product is fixed: it defines the output bits.
        x33 += eye
        x44 += eye
        a32 = _lower(a, beta, 32, 16, LIM, C)
        a42 = _lower(a, beta, 48, 16, LIM, C)
        a43 = _lower(a, beta, 48, 32, LIM, C)
        x32 = -_mm(_mm(x33, a32), x22)
        x43 = -_mm(_mm(x44, a43), x33)
        x31 = -_mm(x33, _mm(_lower(a, beta, 32, 0, LIM, C), x11) + _mm(a32, x21))
        x42 = -_mm(x44, _mm(a42, x22) + _mm(a43, x32))
        x41 = -_mm(x44, _mm(_lower(a, beta, 48, 0, LIM, C), x11) + _mm(a42, x21) + _mm(a43, x31))
        _put(x, x31, 32, 0, LIM, C)
        _put(x, x32, 32, 16, LIM, C)
        _put(x, x33, 32, 32, LIM, C)
        _put(x, x41, 48, 0, LIM, C)
        _put(x, x42, 48, 16, LIM, C)
        _put(x, x43, 48, 32, LIM, C)
        _put(x, x44, 48, 48, LIM, C)
        q32 = _tile(q, 32, 16, LIM, C)
        q33 = _qk_diag(q, 32, LIM, C)
        _put(qx, _mm(q33, x31, _mm(q32, x21, _mm(_tile(q, 32, 0, LIM, C), x11))), 32, 0, LIM, C)
        _put(qx, _mm(q33, x32, _mm(q32, x22)), 32, 16, LIM, C)
        _put(qx, _mm(q33, x33), 32, 32, LIM, C)
        q42 = _tile(q, 48, 16, LIM, C)
        q43 = _tile(q, 48, 32, LIM, C)
        q44 = _qk_diag(q, 48, LIM, C)
        _put(qx, _mm(q44, x41, _mm(q43, x31, _mm(q42, x21, _mm(_tile(q, 48, 0, LIM, C), x11)))),
             48, 0, LIM, C)
        _put(qx, _mm(q44, x42, _mm(q43, x32, _mm(q42, x22))), 48, 16, LIM, C)
        _put(qx, _mm(q44, x43, _mm(q43, x33)), 48, 32, LIM, C)
        _put(qx, _mm(q44, x44), 48, 48, LIM, C)

    # Zeros: the 16 x 16 tiles above the diagonal inside the block, then the rest of the row.
    r = tl.arange(0, BD)
    for c in range(0, LIM, BD):
        cols = c + r
        m = (cols[None, :] // 16 > r[:, None] // 16) & (r[:, None] < LIM) & (cols[None, :] < LIM)
        offs = r[:, None] * C + cols[None, :]
        tl.store(x + offs, 0.0, mask=m)
        tl.store(qx + offs, 0.0, mask=m)


@triton.jit
def _wy_offdiag_kernel(A, Beta, QK, X, QX, C: tl.constexpr, NB: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    base = n * C * C
    a, q, x, qx = A + base, QK + base, X + base, QX + base
    r = tl.arange(0, 64)
    s = tl.arange(0, 16)
    cj = j * 64 + r
    for i in range(j + 1, NB):
        ri = i * 64 + r
        beta = tl.load(Beta + n * C + ri, mask=ri < C, other=0.0)[:, None]
        lx = tl.zeros((64, 64), dtype=tl.float32)
        qx_ij = tl.zeros((64, 64), dtype=tl.float32)
        for k in range(j * 64, i * 64, 16):
            x_kj = _load(x, k + s, cj, C, C)
            lx = _mm(_load(a, ri, k + s, C, C) * beta, x_kj, lx)
            qx_ij = _mm(_load(q, ri, k + s, C, C), x_kj, qx_ij)
        # X[i, j] = -X[i, i] @ lx, with lx staged in the X[i, j] block.
        _store(x, lx, ri, cj, C, C)
        tl.debug_barrier()
        x_ij = tl.zeros((64, 64), dtype=tl.float32)
        for k in range(i * 64, i * 64 + 64, 16):
            x_ij = _mm(_load(x, ri, k + s, C, C), _load(x, k + s, cj, C, C), x_ij)
        tl.debug_barrier()
        _store(x, -x_ij, ri, cj, C, C)
        tl.debug_barrier()
        for k in range(i * 64, i * 64 + 64, 16):
            cols = k + s
            q_ii = tl.where(ri[:, None] >= cols[None, :], _load(q, ri, cols, C, C), 0.0)
            qx_ij = _mm(q_ii, _load(x, cols, cj, C, C), qx_ij)
        _store(qx, qx_ij, ri, cj, C, C)


def wy_solve(A: torch.Tensor, beta: torch.Tensor, QK: torch.Tensor):
    """X = (I + tril(beta[:, None] * A, -1))^-1 and QX = QK @ X for each chunk.

    A, QK: [N, C, C] float32, any C >= 1. beta: N * C float32 values, e.g. [N, C].
    Returns (X, QX), contiguous [N, C, C] float32.
    """
    if A.dim() != 3 or A.shape[1] != A.shape[2] or QK.shape != A.shape:
        raise ValueError(
            f"wy_solve: A and QK must be [N, C, C], got {tuple(A.shape)} and {tuple(QK.shape)}")
    N, C, _ = A.shape
    if beta.numel() != N * C:
        raise ValueError(f"wy_solve: beta must have N * C = {N * C} elements, got {beta.numel()}")
    if not A.dtype == QK.dtype == beta.dtype == torch.float32:
        raise TypeError("wy_solve: A, beta and QK must be float32")
    A, beta, QK = A.contiguous(), beta.reshape(N * C).contiguous(), QK.contiguous()
    X = torch.empty_like(A)
    QX = torch.empty_like(A)
    bd = 16 if C <= 16 else 32 if C <= 32 else 64
    nb = triton.cdiv(C, bd)
    w, s = _DIAG_CONFIG
    tbind(_wy_diag_kernel, (N, nb), A)(A, beta, QK, X, QX, C=C, BD=bd, num_warps=w, num_stages=s)
    if nb > 1:
        w, s = _OFFDIAG_CONFIG
        tbind(_wy_offdiag_kernel, (N, nb - 1), A)(A, beta, QK, X, QX, C=C, NB=nb,
                                                  num_warps=w, num_stages=s)
    return X, QX
