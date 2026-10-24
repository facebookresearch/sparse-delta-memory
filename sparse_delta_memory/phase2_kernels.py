# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Phase-2 kernels of the SDM recurrence: the dual matmul, and the geometry and quantization helpers the CUDA movers
share. (2.0.0 removed this module's Triton entry movers and quantizer: the CUDA movers, ``movers``, are the only route;
the description of the movers below is their shared contract.)

Phase 2 carries the bank across chunks. Step ``ti`` covers chunk ``ti`` of all P streams,
SC = P * C tokens:

  forward   gather -> dual matmul -> DW
  backward  GWR -> baddbmm -> Phase A -> GDS

DW (decay and write) updates the bank rows the step writes and quantizes the snapshot. GWR
(``token_gather`` of the bank gradient) gathers the gradient rows of the write slots. Phase A,
the backward of DW, writes the bank gradient, restores the bank rows and emits three dots per
write entry. GDS (read-side backward) emits one dot per read entry and scatters the reads'
bank gradient. Phase B, the backward through the solve, follows in ``memory_ops``.

Layout. Chunks are time-major (chunk ``ti * P + p``), N = Nt * P of them. Per-entry
tensors are contiguous [N, C, W] (reads: [N, C, R]) and token rows are [N * C, head_dim].
Tokens at or past ``valid_tokens`` in a stream are padding: they touch no bank row and
their outputs are zero.

Each ``make_*`` function checks its operands once and returns ``launch(ti)``. A launch
passes whole tensors plus the step index, so the step loop slices nothing on the host.
Buffers that may hold one step or every step (snapshot, delta_v) are told apart by their
row count.

Movers. A one-warp program owns ENT entries as an [ENT, BLOCK_D] tile and loops over
NUM_D column tiles (``mover_geom``). Steps without padding whose grid covers the entries
exactly run an unmasked variant.

Determinism: every bank (bank-gradient) element has one writer per launch, so every output
is bitwise reproducible. An entry whose slot occurs once in its chunk (is_only, q_only)
writes its row directly. Entries sharing a slot form a group, a contiguous run of Phase 1's
sorted kind row; the group's owner (its last entry, or for wide rows its last few entries,
each owning whole column slices) writes the group's sum (``_group_sums``): the members in
sorted order, in windows of consecutive entries of the kind row, each window reduced as a
tree of separately rounded f32 products (see ``_group_slices``), in f32, with one rounding
at the store. The other members write nothing. GDS adds into the bank gradient, each
element's single writer (a single-entry slot, or a group's owner) rounding once: a group's
owner, and a single-entry slot with a bf16 gradient, read the row, add the f32 sum or term
and store once; with an fp32 gradient a single-entry slot adds with one f32 atomic (its only
add). The lanes that hold copies of a sum agree bit for bit: no sum that crosses lanes
starts from a product fused into its first cross-lane add (``_group_slices``, ``_row_dot``).
The last entry of a slot restores its bank row in Phase A.

"""

from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from .cuda._device import tbind

try:  # Triton 3.7 (the tested version) has the programmatic-dependent-launch trigger
    from triton.language.extra.cuda import gdc_launch_dependents
except ImportError:  # pragma: no cover - without it the trigger is a no-op (same arithmetic)
    @triton.jit
    def gdc_launch_dependents():
        pass

SNAPSHOT_QUANT_MODES = ("none", "fp8", "int4", "bf16")
# "bf16" (float32 banks; bfloat16 banks run it as "none"): bf16 compact snapshot rows on the CUDA kernels only (the
# unquantized forward with bf16 stores, gx_pa2's bf16-snapshot rows); never this module's movers (no fallback).


# ============================================================
# Geometry
# ============================================================

class MoverGeom(NamedTuple):
    ent: int        # entries per program
    block_d: int    # columns per tile
    num_d: int      # column tiles per row


def mover_geom(dtype: torch.dtype, head_dim: int, row_operands: int = 2) -> MoverGeom:
    """Mover tile for bank rows of ``head_dim`` elements of ``dtype``; one warp per program.

    A row is split into 16-byte vectors over up to 32 lanes, each lane holding at most 16
    elements of it per tile; wider rows loop over column tiles. Each lane covers the same
    columns of several rows, about four vectors of each of two row operands, so many bank
    rows are in flight. Kernels with more ``row_operands`` take proportionally fewer rows,
    but at least four when available. A warp spans whole rows, so the per-entry dots are
    warp shuffles.
    """
    vec = 16 // dtype.itemsize
    row_vecs = triton.cdiv(head_dim, vec)
    lanes = min(32, triton.next_power_of_2(row_vecs))
    per_lane = min(16 // vec, triton.next_power_of_2(triton.cdiv(row_vecs, lanes)))
    block_d = lanes * per_lane * vec
    ent = 32 // lanes * max(1, 4 // per_lane)
    ent = min(ent, max(4, ent * 2 // row_operands))
    return MoverGeom(ent, block_d, triton.cdiv(head_dim, block_d))


class DualMatmulGeom(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    dot: str        # tl.dot input_precision


def dual_matmul_geom(chunk_size: int, dtype: torch.dtype) -> DualMatmulGeom:
    """Tiles of the dual matmul: at least two row blocks per chunk up to 64 rows. bf16 banks
    use TF32 dots; fp32 banks follow ``torch.backends.cuda.matmul.allow_tf32``. IEEE dots
    run on the FMA units, where 32-row blocks keep the accumulators in registers."""
    fp32 = dtype == torch.float32
    dot = "ieee" if fp32 and not torch.backends.cuda.matmul.allow_tf32 else "tf32"
    block_m = min(64 if dot == "tf32" else 32, max(16, triton.next_power_of_2(chunk_size) // 2))
    block_k = 16 if chunk_size <= 32 else 32
    return DualMatmulGeom(block_m, 64, block_k, dot)


# ============================================================
# Kernels
# ============================================================


@triton.jit(do_not_specialize=["ti", "valid_tokens"])
def _sdm_dual_matmul(
    x_ptr, qx_ptr,              # [N, C, C] f32: X = (I + diag(beta) A)^-1, QX = QK @ X
    v_ptr, beta_ptr,            # [N * C, HD] bank dtype, [N * C] f32
    ret_ptr, inter_ptr,         # [SC, HD] f32 step rows from the gather
    dv_ptr, dv_step_rows,       # [*, HD] f32 delta_v
    out_ptr,                    # [N * C, HD] readings, bank dtype
    res_ptr,                    # [N * C, HD] f32 residual, or None
    ti, P, valid_tokens,
    C: tl.constexpr, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DOT: tl.constexpr,
    TRIG: tl.constexpr,
):
    """delta_v = X @ u and readings = QX @ u + inter with u = beta * (v - retrieved), for
    chunk p of step ti. The first row block also stores residual = v - retrieved.
    The epilogue's ``inter`` rows are loaded before the k loop (they overlap its loads). TRIG: issue the
    programmatic-dependent-launch trigger first, so the step's write mover, launched as a programmatic
    dependent, can start its setup while this kernel runs (forward._dual_trig_wanted); no effect otherwise.

    Grid: (cdiv(C, BLOCK_M) * cdiv(HD, BLOCK_N) * P,), row blocks first, then column
    blocks, then chunks (one axis, which holds any number of chunks).
    """
    if TRIG:
        gdc_launch_dependents()
    NUM_M: tl.constexpr = (C + BLOCK_M - 1) // BLOCK_M
    NUM_N: tl.constexpr = (HD + BLOCK_N - 1) // BLOCK_N
    pid = tl.program_id(0)
    pid_m = pid % NUM_M
    pid_n = pid // NUM_M % NUM_N
    p = (pid // (NUM_M * NUM_N)).to(tl.int64)
    chunk = ti.to(tl.int64) * P + p
    # 64-bit bases of this chunk's rows; the offsets within a chunk fit 32 bits.
    x_ptr += chunk * C * C
    qx_ptr += chunk * C * C
    v_ptr += chunk * C * HD
    beta_ptr += chunk * C
    ret_ptr += p * C * HD
    inter_ptr += p * C * HD
    dv_ptr += (ti.to(tl.int64) * dv_step_rows + p * C) * HD
    out_ptr += chunk * C * HD
    if res_ptr is not None:
        res_ptr += chunk * C * HD
    rows = tl.minimum(valid_tokens - ti * C, C)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = rn[None, :] < HD
    m_ok = (rm < rows)[:, None]
    in_c = (rm < C)[:, None] & n_ok
    m_rows = rm[:, None] * HD + rn[None, :]
    inter = tl.load(inter_ptr + m_rows, mask=m_ok & n_ok, other=0.0)
    acc_dv = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        k_ok = rk < rows
        tile = k_ok[:, None] & n_ok
        k_rows = rk[:, None] * HD + rn[None, :]
        v = tl.load(v_ptr + k_rows, mask=tile, other=0.0)
        r = tl.load(ret_ptr + k_rows, mask=tile, other=0.0)
        res = v.to(tl.float32) - r
        if res_ptr is not None:
            if pid_m == 0:
                tl.store(res_ptr + k_rows, res, mask=(rk < C)[:, None] & n_ok)
        u = tl.load(beta_ptr + rk, mask=k_ok, other=0.0)[:, None] * res
        x_off = rm[:, None] * C + rk[None, :]
        x_ok = (rm < C)[:, None] & (rk < C)[None, :]
        x = tl.load(x_ptr + x_off, mask=x_ok, other=0.0)
        acc_dv = tl.dot(x, u, acc_dv, input_precision=DOT)
        qx = tl.load(qx_ptr + x_off, mask=x_ok, other=0.0)
        acc_out = tl.dot(qx, u, acc_out, input_precision=DOT)
    tl.store(dv_ptr + m_rows, tl.where(m_ok, acc_dv, 0.0), mask=in_c)
    out = tl.where(m_ok, acc_out + inter, 0.0)
    tl.store(out_ptr + m_rows, out.to(out_ptr.dtype.element_ty), mask=in_c)


# ============================================================
# Movers
# ============================================================


# ============================================================
# Launchers
# ============================================================

def _check(t: torch.Tensor, name: str, numel: int, dtype=None) -> torch.Tensor:
    if not t.is_contiguous() or t.numel() != numel or (dtype is not None and t.dtype != dtype):
        want = f"{numel} contiguous elements" + (f" of {dtype}" if dtype is not None else "")
        raise ValueError(f"{name}: expected {want}, got shape {tuple(t.shape)} "
                         f"stride {t.stride()} {t.dtype}")
    return t


def quant_row(mode: str, head_dim: int) -> tuple[int, torch.dtype]:
    """(row width, dtype) of a quantized snapshot row."""
    if mode == "fp8":
        return head_dim, torch.float8_e4m3fn
    if mode == "int4":
        return (head_dim + 1) // 2, torch.uint8
    raise ValueError(f"snapshot_quant must be one of {SNAPSHOT_QUANT_MODES}, got {mode!r}")


def _step_rows(t: torch.Tensor, name: str, rows: int, steps: int, width: int) -> int:
    """Row stride between steps of ``t``: ``rows`` if it holds every step, 0 if one."""
    if t.is_contiguous() and t.numel() == steps * rows * width:
        return rows
    if t.is_contiguous() and t.numel() == rows * width:
        return 0
    raise ValueError(f"{name}: expected {rows} or {steps * rows} contiguous rows of {width}, "
                     f"got shape {tuple(t.shape)} stride {t.stride()}")


def make_dual_matmul(x, qx, v, beta, retrieved, inter, delta_v, readings, residual=None, *,
                     num_parallel: int, valid_tokens: int) -> Callable[[int], None]:
    """Step launcher of delta_v = X @ u and readings = QX @ u + inter, u = beta * (v - retrieved).

    x, qx: [N, C, C] f32; v, readings: [N * C, hd] (bank dtype); beta: [N * C] f32;
    retrieved, inter: [SC, hd] f32 (this step's gather); delta_v: f32 rows of one step or
    of every step; residual (v - retrieved, f32 [N * C, hd]) is stored when given.
    """
    n, c, _ = x.shape
    hd = v.shape[-1]
    sc = num_parallel * c
    steps = n // num_parallel
    geom = dual_matmul_geom(c, v.dtype)
    for t, name in ((x, "x"), (qx, "qx")):
        _check(t, name, n * c * c, torch.float32)
    for t, name in ((retrieved, "retrieved"), (inter, "inter")):
        _check(t, name, sc * hd, torch.float32)
    _check(v, "v", n * c * hd)
    _check(readings, "readings", n * c * hd, v.dtype)
    _check(beta, "beta", n * c, torch.float32)
    if residual is not None:
        _check(residual, "residual", n * c * hd, torch.float32)
    dv_rows = _step_rows(delta_v, "delta_v", sc, steps, hd)
    kernel = tbind(_sdm_dual_matmul, (triton.cdiv(c, geom.block_m) * triton.cdiv(hd, geom.block_n)
                                      * num_parallel,), v)
    head = (x, qx, v, beta, retrieved, inter, delta_v, dv_rows, readings, residual)
    meta = dict(C=c, HD=hd, BLOCK_M=geom.block_m, BLOCK_N=geom.block_n, BLOCK_K=geom.block_k,
                DOT=geom.dot, TRIG=0, num_warps=4)

    def launch(ti: int) -> None:
        kernel(*head, ti, num_parallel, valid_tokens, **meta)
    return launch


