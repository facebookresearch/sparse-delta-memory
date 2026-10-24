# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The chunked SDM write/read op (``sdm_write_read``) and its autograd function.

Layout. P parallel streams of T tokens ([P, T, *] inputs, or [P * T, *] rows with
``num_parallel=P``) are cut into Nt = cdiv(T, C) chunks of C tokens and staged time-major:
chunk ``ti * P + p`` is chunk ti of stream p. Tokens past a stream's end are padding: they
are staged as zeros, touch no bank row and produce no output.

Forward:
  Phase 1   the memory-independent quantities of every chunk: per-slot log-decays, dedup
            flags, gather weights and the decay-weighted Gram matrices A (writes x writes)
            and QK (reads x writes)
  solve     X = (I + tril(beta * A, -1))^-1 and QX = QK @ X
  Phase 2   Nt sequential steps over the P chunks of a step:
              gather       retrieved and inter (weighted bank rows) and the snapshot
              dual matmul  delta_v = X @ u, readings = QX @ u + inter, u = beta (v - retrieved)
              DW           decays and writes the bank rows (and quantizes the snapshot)
Backward: Phase 2 in reverse order (GWR, baddbmm, Phase A, GDS), then Phase B, the
matmuls through the solve, and the Phase-1 backward.

The forward steps live in forward.py, the backward in backward.py and the mover launchers in movers.py. A call the
native kernels do not take end to end is refused before any work (_admit, 2.0.0: no fallback route).

The bank is advanced in place to its final state; the backward restores it, exactly with
snapshot_quant "none" and to the dequantized snapshot rows with fp8/int4. A call that
needs no backward saves nothing: the snapshot and delta_v are per-step scratch, and the
steps and streams run in bounded slabs and groups (SLAB_BYTES, STEP_BYTES).

Determinism: every output, the final bank and every gradient are bitwise reproducible for
fixed inputs, settings and GPU type (no element receives two concurrent float adds: every bank and
bank-gradient row has one writer per launch and every sum runs in a fixed order).

Also here: the head split/merge row permutes and the per-batch expand of the learned bank
used by the layer. Every op call records its launch geometry (``last_dispatch``).
"""

import contextlib
import functools
import threading
from typing import NamedTuple

import torch
import torch.utils.deterministic  # noqa: F401  (fill_uninitialized_memory, see _no_det_fill)
import triton
import triton.language as tl
from torch.utils._python_dispatch import _disable_current_modes

from .cuda import _devmemo, _device
from .cuda.gather_cuda import gather_geometry, token_gather
from .cuda.phase1_bwd_cuda import phase1_backward
from .cuda.phase1_cuda import Phase1
from . import movers as _movers
from .movers import make_mover_gds
from .phase2_kernels import SNAPSHOT_QUANT_MODES

# The forward steps (forward.py) call these through this module, the one place a test substitutes them.
from .cuda.phase1_cuda import phase1_forward  # noqa: F401
from .movers import make_mover_dw  # noqa: F401
from .phase2_kernels import dual_matmul_geom, make_dual_matmul, mover_geom, quant_row  # noqa: F401
from .wy_solve import wy_solve  # noqa: F401

MIN_CHUNK, MAX_CHUNK = 16, 256
# RTX 30 / 40 / 50 (99 KB of shared memory per block): chunks of at most 128 tokens, refused before any work otherwise.
RTX_MAX_CHUNK = 128
RTX_CHUNK_MSG = ("chunk_size {chunk_size} is not supported on RTX 30 / 40 / 50 GPUs: their 99 KB of shared memory per "
                 "block allow chunk_size <= 128")
DEFAULT_CHUNK = 64  # chunk_size of sdm_write_read, and of a layer whose chunk_size is None (the previous release: 256)
MAX_WIDTH = 256  # write and read slots per token

# Working-set budgets of a forward without backward state. Its steps run in slabs whose
# staged inputs and Phase-1 state fit SLAB_BYTES, unless that leaves fewer than
# SLAB_CHUNKS_PER_SM chunks per SM (Phase 1 runs a block per chunk), and its streams run
# in groups whose per-step snapshot rows fit STEP_BYTES. Its working memory does not grow
# with T.
SLAB_BYTES = 512 << 20
SLAB_CHUNKS_PER_SM = 2
STEP_BYTES = 512 << 20
# The slabs of the forward without backward state hold NOGRAD_SLAB_FACTOR x SLAB_BYTES (read at every call, so a patched
# SLAB_BYTES applies): fewer slabs, less host work per call, larger Phase-1 / WY launches. Slabs are exact (Phase 1, WY,
# the gather geometry pinned by geom_tokens and the DW are per chunk), so the outputs do not depend on the budget.
NOGRAD_SLAB_FACTOR = 2


# ============================================================
# Dispatch record and repeated-slot counter
# ============================================================
# Plain Python values derived from shapes, dtypes and arguments only: recording never
# reads a tensor value, so it never synchronizes with the device.
_DISPATCH: dict = {}


def _record(**decisions) -> None:
    _DISPATCH.update(decisions)


def last_dispatch() -> dict:
    """A copy of the launch decisions of the most recent op call (graph replays do not
    update it)."""
    return dict(_DISPATCH)


def reset_dispatch() -> None:
    _DISPATCH.clear()


# One counter per device, created on its first call (run one eager call before capturing
# a CUDA graph).
_repeat_counters: dict = {}


def _repeat_counter(device) -> torch.Tensor:
    t = _repeat_counters.get(str(device))
    if t is None:
        t = _repeat_counters[str(device)] = torch.zeros(1, device=device, dtype=torch.int64)
    return t


def repeated_write_slots(reset: bool = True) -> int:
    """Number of write entries seen so far that repeat a slot of their own token (the op
    requires distinct write slots per token and decays a repeated slot once).

    Synchronizes with the device.
    """
    total = 0
    for t in _repeat_counters.values():
        total += int(t.item())
        if reset:
            t.zero_()
    return total


# ============================================================
# Stream rows <-> time-major chunk rows
# ============================================================
# Stream rows [P, T] <-> time-major chunk rows [Nt, P, C]: chunk ti * P + p holds tokens
# ti * C .. ti * C + C - 1 of stream p (step ti). Time-major rows past a stream's last token
# are zero.

@triton.jit
def _stage_kernel(src_ptr, dst_ptr, rows, row0, P, T, C, D,
                  ROWS: tl.constexpr, BLOCK: tl.constexpr, TO_TM: tl.constexpr):
    """Copies time-major rows row0 .. row0 + rows - 1 from their stream rows (TO_TM) or
    back, in the destination dtype; grid (cdiv(rows, ROWS), cdiv(D, BLOCK))."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = r < rows
    n = (row0 + r) // C
    t = n // P * C + (row0 + r) % C
    stream_row = n % P * T + t
    ok = live & (t < T)
    if TO_TM:
        src, dst, store = stream_row, r, live
    else:
        src, dst, store = r, stream_row, ok
    cols = (col < D)[None, :]
    x = tl.load(src_ptr + src.to(tl.int64)[:, None] * D + col[None, :],
                mask=ok[:, None] & cols, other=0)
    tl.store(dst_ptr + dst.to(tl.int64)[:, None] * D + col[None, :],
             x.to(dst_ptr.dtype.element_ty), mask=store[:, None] & cols)


def _launch_stage(src, dst, rows, row0, P, T, C, to_time_major):
    D = dst.shape[-1]
    block = min(1024, triton.next_power_of_2(D))
    per = max(1, 2048 // block)
    _device.tbind(_stage_kernel, (triton.cdiv(rows, per), triton.cdiv(D, block)), src)(
        src, dst, rows, row0, P, T, C, D, ROWS=per, BLOCK=block, TO_TM=to_time_major,
        num_warps=4)


def stage_time_major(x: torch.Tensor, P: int, T: int, C: int, dtype: torch.dtype,
                     steps: tuple[int, int] | None = None) -> torch.Tensor:
    """[P * T, ...] stream rows as [n * P * C, D] time-major chunk rows in ``dtype``, for
    the n steps ``range(*steps)`` (default: all cdiv(T, C) of them), with zero rows past
    each stream's end. A view of ``x`` when the layouts coincide (P == 1, no row past the
    end, same dtype)."""
    first, stop = steps or (0, triton.cdiv(T, C))
    x = x.reshape(P * T, -1)
    if P == 1 and stop * C <= T and x.dtype == dtype and x.is_contiguous():
        return x[first * C:stop * C]
    rows = (stop - first) * P * C
    out = torch.empty(rows, x.shape[1], device=x.device, dtype=dtype)
    _launch_stage(x.contiguous(), out, rows, first * P * C, P, T, C, True)
    return out


def unstage_time_major(x: torch.Tensor, P: int, T: int, C: int, dtype: torch.dtype,
                       steps: tuple[int, int] | None = None,
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """The inverse of ``stage_time_major``: writes the time-major rows of ``steps`` into
    their rows of ``out`` ([P * T, D], allocated in ``dtype`` when None) and returns it.
    Returns ``x`` itself when the layouts coincide and no ``out`` is given."""
    first, stop = steps or (0, triton.cdiv(T, C))
    rows = (stop - first) * P * C
    x = x.reshape(rows, -1)
    if out is None:
        if P == 1 and rows == T and x.dtype == dtype:
            return x
        out = torch.empty(P * T, x.shape[1], device=x.device, dtype=dtype)
    _launch_stage(x, out, rows, first * P * C, P, T, C, False)
    return out


# ============================================================
# Elementwise kernels around the backward matmuls
# ============================================================

@triton.jit
def _fused_pre_bmm_residual_kernel(
    residual_ptr, beta_ptr, d_dva_ptr,
    u_ptr,                # [N * C, dim] f32
    grad_v_ptr,           # [N * C, dim] bank dtype
    grad_beta_part_ptr,   # [N * C] f32
    dim: tl.constexpr, BD: tl.constexpr,
):
    """u = beta * residual, grad_v = beta * d_dva and grad_beta_part = <residual, d_dva>,
    with residual = v - retrieved. One program per token."""
    t = tl.program_id(0).to(tl.int64)
    beta = tl.load(beta_ptr + t)
    acc = tl.zeros([1], dtype=tl.float32)
    for ds in range(0, dim, BD):
        do = ds + tl.arange(0, BD)
        m = do < dim
        res = tl.load(residual_ptr + t * dim + do, mask=m, other=0.0)
        dva = tl.load(d_dva_ptr + t * dim + do, mask=m, other=0.0)
        tl.store(u_ptr + t * dim + do, beta * res, mask=m)
        tl.store(grad_v_ptr + t * dim + do, (beta * dva).to(grad_v_ptr.dtype.element_ty),
                 mask=m)
        acc += tl.sum(res * dva)
    tl.store(grad_beta_part_ptr + t, tl.sum(acc))


def _pre_bmm(residual, beta, d_dva, grad_v):
    """Returns (u, grad_beta_part) [N, C, dim] / [N, C, 1] f32 and writes grad_v."""
    N, C, dim = residual.shape
    u = torch.empty_like(residual)
    grad_beta_part = torch.empty(N, C, 1, device=residual.device, dtype=torch.float32)
    _device.tbind(_fused_pre_bmm_residual_kernel, (N * C,), beta)(
        residual, beta, d_dva, u, grad_v, grad_beta_part,
        dim=dim, BD=min(1024, triton.next_power_of_2(dim)))
    return u, grad_beta_part


@triton.jit
def _fused_post_bmm_kernel(
    d_m_ptr, a_ptr, beta_ptr, grad_beta_part_ptr, d_a_ptr, grad_beta_ptr,
    C: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Strict lower triangle of d_m: d_a = beta * d_m and
    grad_beta = grad_beta_part + <A, d_m> per row. One program per (chunk, row)."""
    pid = tl.program_id(0).to(tl.int64)
    batch = pid // C
    row = pid % C
    beta = tl.load(beta_ptr + batch * C + row).to(tl.float32)
    grad_beta = tl.load(grad_beta_part_ptr + batch * C + row).to(tl.float32)
    base = batch * C * C + row * C
    col_offs = tl.arange(0, BLOCK_C)
    for col_start in range(0, C, BLOCK_C):
        col = col_start + col_offs
        m = col < row
        d_m = tl.load(d_m_ptr + base + col, mask=m, other=0.0)
        a = tl.load(a_ptr + base + col, mask=m, other=0.0)
        grad_beta += tl.sum(a * d_m)
        tl.store(d_a_ptr + base + col, tl.where(m, beta * d_m, 0.0), mask=col < C)
    tl.store(grad_beta_ptr + batch * C + row, grad_beta)


def _post_bmm(d_m, a, beta, grad_beta_part, grad_beta):
    """Returns d_a [N, C, C] and writes grad_beta [N * C]."""
    N, C, _ = d_m.shape
    d_a = torch.empty_like(d_m)
    _device.tbind(_fused_post_bmm_kernel, (N * C,), d_m)(
        d_m, a, beta, grad_beta_part, d_a, grad_beta,
        C=C, BLOCK_C=min(256, triton.next_power_of_2(C)), num_warps=4)
    return d_a


# ============================================================
# The chunked write/read
# ============================================================

class _Layout(NamedTuple):
    P: int              # parallel streams
    T: int              # tokens per stream
    C: int              # chunk size
    Nt: int             # chunks per stream (Phase-2 steps)
    W: int              # write slots per token
    R: int              # read slots per token
    G: int              # gate width, 1 or W
    hd: int             # head_dim
    stream_offset: int  # slot offset between streams; 0 when all share one id space
    num_keys: int       # slot ids per stream


def _layout(memory, k_idx, k_val, v, beta, g, q_idx, q_val, chunk_size, num_memory_slots,
            num_parallel) -> _Layout:
    """Validates the call and returns its static layout (once per op call; _admit then checks the route)."""
    if memory.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"memory must be bfloat16 or float32, got {memory.dtype}")
    if memory.ndim != 2 or not memory.is_cuda or not memory.is_contiguous():
        raise ValueError("memory must be a contiguous CUDA tensor [num_slots, head_dim]")
    for name, n in (("chunk_size", chunk_size), ("num_memory_slots", num_memory_slots),
                    ("num_parallel", num_parallel)):
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"{name} must be an int, got {n!r}")
    if not MIN_CHUNK <= chunk_size <= MAX_CHUNK:
        raise ValueError(f"chunk_size must be in {MIN_CHUNK}..{MAX_CHUNK}, got {chunk_size}")
    if num_memory_slots < 0:
        raise ValueError(f"num_memory_slots must be >= 0 (0: the whole bank), got "
                         f"{num_memory_slots}")
    P = num_parallel
    if P < 1:
        raise ValueError(f"num_parallel must be positive, got {P}")
    if k_idx.ndim == 3:
        if k_idx.shape[0] != P:
            raise ValueError(f"3-D inputs must have num_parallel={P} streams along dim 0, "
                             f"got k_idx of shape {tuple(k_idx.shape)}")
        T = k_idx.shape[1]
    else:
        if k_idx.shape[0] % P:
            raise ValueError(f"{k_idx.shape[0]} tokens do not split into num_parallel={P} "
                             "equal streams")
        T = k_idx.shape[0] // P
    if k_idx.dtype not in (torch.int32, torch.int64) or q_idx.dtype not in (torch.int32,
                                                                            torch.int64):
        raise TypeError(f"k_idx and q_idx must be int32 or int64, got {k_idx.dtype} and "
                        f"{q_idx.dtype}")
    dev = memory.get_device()
    for name, t in (("k_idx", k_idx), ("k_val", k_val), ("v", v), ("beta", beta), ("g", g),
                    ("q_idx", q_idx), ("q_val", q_val)):
        if not t.is_cuda or t.get_device() != dev:
            raise ValueError(f"{name} is on {t.device}, memory on {memory.device}")
    num_slots, hd = memory.shape
    if num_slots < 1 or hd < 1:
        raise ValueError(f"memory needs at least one slot and one column, got shape "
                         f"{tuple(memory.shape)}")
    if num_slots >= 2**31:
        raise ValueError(f"memory has {num_slots} rows; slot ids must fit int32")
    W, R, G = k_idx.shape[-1], q_idx.shape[-1], g.shape[-1]
    lead = k_idx.shape[:-1]
    for name, t, width in (("k_val", k_val, W), ("v", v, hd), ("beta", beta, 1),
                           ("g", g, G), ("q_idx", q_idx, R), ("q_val", q_val, R)):
        if t.shape[:-1] != lead or t.shape[-1] != width:
            raise ValueError(f"{name}: expected shape {(*lead, width)}, got {tuple(t.shape)}")
    if T < 1 or not (1 <= W <= MAX_WIDTH and 1 <= R <= MAX_WIDTH):
        raise ValueError(f"need T >= 1 and 1..{MAX_WIDTH} write/read slots per token, "
                         f"got T={T} W={W} R={R}")
    if G not in (1, W):
        raise ValueError(f"g must hold 1 or W={W} gates per token, got {G}")
    Nt = -(-T // chunk_size)
    if P * Nt * chunk_size >= 2**31:
        raise ValueError(f"{P} streams of {T} tokens, padded to chunks of {chunk_size}, "
                         "exceed the 2**31 - 1 token rows the kernels address")
    sph = num_memory_slots or num_slots
    if num_slots % sph:
        raise ValueError(f"num_memory_slots={sph} does not divide the {num_slots} bank slots")
    streams = num_slots // sph
    if P > 1 and streams not in (1, P):
        raise ValueError(
            f"memory holds {streams} streams of {sph} slots but num_parallel={P}: each "
            f"parallel stream needs its own slot range (num_slots == num_parallel * "
            f"num_memory_slots) or one shared bank (num_memory_slots == num_slots).")
    own = P > 1 and streams == P
    return _Layout(P, T, chunk_size, Nt, W, R, G, hd, sph if own else 0,
                   sph if own else num_slots)


# ============================================================
# Admission: fp8 / int4 snapshot shapes and calls without a native route (refused before any work)
# ============================================================
# The quantized DW (mover_fx_cuda) and Phase A (gx_pa2_cuda) take fp8 / int4 snapshot rows only for heads of at most
# QUANT_TAIL_MAX_HEAD_DIM columns in ceil(head_dim / 128) equal slices of 16-byte rows, and int4 rows of float32 heads only
# in whole 8-column groups: the other heads (T0-T3, S4) are refused on saving calls. No-grad calls never quantize.
QUANT_TAIL_MAX_HEAD_DIM = 16384   # the quantized forward's head bound (cuda/mover_fx_cuda._QUANT_MAX_HEAD_DIM)
_DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


class UnsupportedConfigurationError(ValueError):
    """A configuration this release refuses up front, before any work. The message names the reason and what is
    supported; ``tail_class`` holds the class (quant_tail_class's "T0".."T3" / "S4", native_route_class's "H" / "S" /
    "P"; for logs and tests, not in the message)."""

    tail_class = None


QUANT_TAIL_CLASSES = {
    "T0": "more than 16384 columns",
    "T1": "column slices whose rows are not 16-byte multiples",
    "T2": "uneven column slices",
    "T3": "head_dim not a multiple of 4",
}
QUANT_TAIL_MSG = (
    "sparse_delta_memory: snapshot_quant={quant!r} on a {dtype} bank with head_dim {hd} is not supported for calls "
    "that may need a backward ({what}); support is planned for a later release. Supported for such calls: "
    "head_dim <= 16384 that splits into ceil(head_dim / 128) equal column slices whose width is a multiple of 8 "
    "(bfloat16 bank, or int4 on a float32 bank) or of 4 (fp8 on a float32 bank), e.g. 64, 128, 480, 960, 1024, 2048; "
    "snapshot_quant='none' supports every head_dim, and calls without a backward (torch.no_grad(), inference) are not "
    "limited.")
QUANT_S4_MSG = (
    "sparse_delta_memory: snapshot_quant={quant!r} on a {dtype} bank with head_dim {hd} is not supported for calls "
    "that may need a backward (column slices {sw} columns wide, not a multiple of 8); native support is planned for "
    "2.0.1. Supported for such calls: snapshot_quant='fp8' or 'none' with this head_dim, or int4 with a head_dim that "
    "splits into ceil(head_dim / 128) equal column slices whose width is a multiple of 8, e.g. 64, 128, 480, 960, 1024, "
    "2048; calls without a backward (torch.no_grad(), inference) are not limited.")


def quant_tail_class(bank_dtype, head_dim, snapshot_quant, compact_rows=0, device=None):
    """The class of a saving call with ``snapshot_quant`` rows on a ``bank_dtype`` bank of ``head_dim`` columns:
    "T0".."T3" (QUANT_TAIL_CLASSES) or, when the caller passes the call's ``device`` (not read: no GPU is refused), "S4";
    else None. Shapes, dtype and mode only; ``compact_rows`` is converted (int()) but not bounded (64-bit snapshot rows)."""
    if snapshot_quant not in ("fp8", "int4"):
        return None
    hd = int(head_dim)
    if hd > 2 ** 20:
        return None
    int(compact_rows)
    if hd % 4:
        return "T3"
    nsl = -(-hd // 128)
    sw = hd // nsl
    if hd % nsl or sw % 4:
        return "T2"
    es = 2 if bank_dtype == torch.bfloat16 else 4
    if (sw * es) % 16 or (hd * es) % 16:
        return "T1"
    if hd > QUANT_TAIL_MAX_HEAD_DIM:
        return "T0"
    if device is not None and snapshot_quant == "int4" and bank_dtype == torch.float32 and hd >= 1 and sw % 8:
        return "S4"
    return None


def refuse_quant_tail(bank_dtype, head_dim, snapshot_quant, compact_rows=0, device=None) -> None:
    """Raises UnsupportedConfigurationError for a saving call of a quant_tail_class class (same arguments)."""
    cls = quant_tail_class(bank_dtype, head_dim, snapshot_quant, compact_rows, device=device)
    if cls is not None:
        dtype, hd = _DTYPE_NAMES.get(bank_dtype, str(bank_dtype)), int(head_dim)
        err = UnsupportedConfigurationError(
            QUANT_S4_MSG.format(quant=snapshot_quant, dtype=dtype, hd=hd, sw=hd // -(-hd // 128)) if cls == "S4" else
            QUANT_TAIL_MSG.format(quant=snapshot_quant, dtype=dtype, hd=hd, what=QUANT_TAIL_CLASSES[cls]))
        err.tail_class = cls
        raise err


# Calls without a native route (2.0.0: no fallback route): native_route_class composes the predicates the forward and
# the backward route on, so it refuses exactly the calls that took a fallback before; the bounds (2^20 columns, 2^31
# slots) only NAME the class. Modes: no-grad, saving (SDMWriteRead.forward) and training (saving + the CUDA backward).
NATIVE_CLASSES = {
    "H": "head_dim {hd} is above 1,048,576 (2^20 columns)",
    "S": "the bank has {slots} slots (2^31 or more)",
    "P": "no native kernel plan for this shape fits this GPU ({gpu})",
}
NATIVE_MSG = ("sparse_delta_memory: no native route for this call: {what}. Bank {dtype} [{slots}, {hd}], chunk_size {C}, "
              "snapshot_quant {quant}, {mode}. This release has no fallback route (removed in 2.0.0): {remedy}.")
NATIVE_MODES = {"training": "training call", "saving": "saving call (SDMWriteRead.apply without autograd recording)",
                "no-grad": "no-grad call"}
# Each class's own remedy (NATIVE_MSG's {remedy}); a training / saving call whose no-grad route exists also gets
# NATIVE_NOGRAD_OK (checked: native_route_class in no-grad mode is None).
NATIVE_REMEDIES = {
    "H": "use a head_dim of at most 1,048,576",
    "S": "use a bank of fewer than 2^31 slots",
    "P": "use a smaller chunk_size or another head_dim",
}
NATIVE_P_GPU = ", or an H100 / B200 / GB300 GPU (more shared memory per block)"   # P on the A100 / RTX backends
NATIVE_P_QUANT = ", or snapshot_quant='none'"                                     # P of a saving fp8 / int4 call
NATIVE_NOGRAD_OK = {"training": "; the same call without autograd (torch.no_grad(), inference) runs natively",
                    "saving": "; sdm_write_read runs it natively as a no-grad call"}
_NATIVE_MODS = None   # (backward, forward, movers, cuda.mover_da2_cuda): imported once, at the first route lookup


def _native_preds():
    """The routing predicates native_route_class composes, as currently bound (part of route_token)."""
    global _NATIVE_MODS
    if _NATIVE_MODS is None:
        from . import backward as _bw, forward as _fw, movers as _mv
        from .cuda import mover_da2_cuda as _da2
        _NATIVE_MODS = (_bw, _fw, _mv, _da2)
    bw, fw, mv, da2 = _NATIVE_MODS
    return (mv.dw_admits, mv.gds_admits, mv.tail_admits, bw.supported, da2._shape_supported, fw._fxq_rows,
            fw.bf16_route)


def route_token():
    """What a call's admission and route depend on beside its own arguments and shapes: the forced test row and the
    routing predicates as bound (a test that patches one gets fresh answers from the memos keyed on it)."""
    return _device._FORCED, _native_preds()


def native_route_class(memory, C, snapshot_quant="none", compact_rows=0, needs_backward=False, grad_bank_fp32=False,
                       saving=None):
    """None when ``memory`` with chunks of ``C`` runs natively end to end in the given mode, else the NATIVE_CLASSES key
    (``compact_rows`` = P * Nt * C * W; saving=None: saving = needs_backward)."""
    from . import backward as _bw, forward as _fw, movers as _mv
    from .cuda import mover_da2_cuda as _da2
    needs_backward = bool(needs_backward)
    saving = needs_backward or bool(saving)
    C, rows = int(C), int(compact_rows)
    hd, slots = int(memory.shape[-1]), int(memory.shape[0])
    # the forward's DW mode (forward.forward_steps: quant iff save); "bf16" runs as "none" where effective_snapshot_quant says
    q = effective_snapshot_quant(memory, snapshot_quant, needs_backward) if saving else "none"
    qn = "none" if q == "bf16" else q                   # bf16 rows (float32 banks): 'none''s DW and GDS (forward.bf16_route)
    native = _mv.dw_admits(memory, C, qn)               # else movers.make_mover_dw has no native DW (NO_NATIVE_MOVER_MSG)
    if native and saving:
        if q == "bf16":
            # the saving forward's bf16 compact rows (mover_fx_cuda's WR walk) and the backward reading them (gx_pa2's
            # bf16-snapshot rows): bf16_route
            native = _fw.bf16_route(memory, C, rows)
        elif q != "none" or hd % 4 or not _da2._shape_supported(memory, C):
            # compact rows (layout 2) required: a tail head's DW (movers.make_mover_dw: tail_off), backward._supported
            # (hd % 4 != 0) and every quantized backward (backward.sdm_backward) read them; the saving forward writes them
            # iff backward.supported admits the worst grad dtype (forward.forward_steps), fp8 / int4 also need the
            # quantized DW; otherwise it writes layout 0, which no native DW takes for these calls
            native = _bw.supported(memory, q, C=C, grad_f32_possible=True, rows=rows) and (
                q == "none" or _fw._fxq_rows(memory, C, q))
    if native and needs_backward:   # the CUDA backward (backward._supported) and its GDS (movers.make_mover_gds)
        gdt = torch.float32 if (grad_bank_fp32 and memory.dtype == torch.bfloat16) else memory.dtype
        native = (_bw.supported(memory, q, C=C, grad_f32_possible=bool(grad_bank_fp32), rows=rows)
                  and _mv.gds_admits(memory, gdt, C, qn))
    if native:
        return None
    if hd > 2 ** 20:
        return "H"
    if slots >= 2 ** 31:
        return "S"
    return "P"


def refuse_non_native(memory, C, snapshot_quant="none", compact_rows=0, needs_backward=False, grad_bank_fp32=False,
                      saving=None):
    """Raises UnsupportedConfigurationError (``tail_class`` = the NATIVE_CLASSES key) for a call without a native route
    (native_route_class, same arguments); a saving "bf16" call on a float32 bank raises refuse_bf16_without_route's text.
    First the device's class: a GPU outside the supported matrix (_device's RuntimeError), RTX_CHUNK_MSG on RTX."""
    if _devmemo.backend(memory.device) == "rtx" and C > RTX_MAX_CHUNK:
        raise ValueError(RTX_CHUNK_MSG.format(chunk_size=C))
    saving = bool(needs_backward) or bool(saving)
    cls = native_route_class(memory, C, snapshot_quant, compact_rows, needs_backward, grad_bank_fp32, saving)
    if cls is None:
        return
    if saving and effective_snapshot_quant(memory, snapshot_quant, needs_backward) == "bf16":
        refuse_bf16_without_route(memory, C, int(compact_rows))   # bf16 on a float32 bank: its own text first
    hd, slots = int(memory.shape[-1]), int(memory.shape[0])
    gpu = torch.cuda.get_device_name(memory.device) if cls == "P" else ""
    what = NATIVE_CLASSES[cls].format(hd=hd, slots=slots, rows=int(compact_rows), gpu=gpu)
    mode = "training" if needs_backward else "saving" if saving else "no-grad"
    remedy = NATIVE_REMEDIES[cls]
    if cls == "P":
        remedy += (NATIVE_P_QUANT if saving and snapshot_quant != "none" else "") + (
            NATIVE_P_GPU if _device.backend(memory.device) != "modern" else "")
    if mode != "no-grad" and native_route_class(memory, C) is None:
        remedy += NATIVE_NOGRAD_OK[mode]
    err = UnsupportedConfigurationError(NATIVE_MSG.format(
        what=what, dtype=_DTYPE_NAMES.get(memory.dtype, str(memory.dtype)), slots=slots, hd=hd, C=int(C),
        quant=snapshot_quant if saving else "none", mode=NATIVE_MODES[mode], remedy=remedy))
    err.tail_class = cls
    raise err


# The op's admission memo (item 3): the call signatures _admit admitted. Refused calls are never stored.
_NATIVE_MEMO = {}
_NATIVE_MEMO_MAX = 4096
_NATIVE_LOCK = threading.Lock()


def _admit(memory, lay, snapshot_quant="none", needs_backward=False, saving=False, grad_bank_fp32=False) -> None:
    """The op's checks after _layout, once per call signature: a saving call's T0-T3 / S4 refusals, then the native
    route's. Only admitted signatures are stored, so a refused call is checked, and raises, at every call. Returns the
    snapshot mode the call runs (effective_snapshot_quant), stored with the admission."""
    rows = lay.P * lay.Nt * lay.C * lay.W if saving else 0
    key = (memory.device.index, memory.dtype, tuple(memory.shape), lay.C, rows, snapshot_quant, bool(needs_backward),
           bool(saving), bool(grad_bank_fp32), route_token())
    hit = _NATIVE_MEMO.get(key)
    if hit is not None:
        return hit
    if saving:
        refuse_quant_tail(memory.dtype, lay.hd, snapshot_quant, rows, device=memory.device)
    refuse_non_native(memory, lay.C, snapshot_quant, rows, needs_backward, grad_bank_fp32, saving)
    quant = effective_snapshot_quant(memory, snapshot_quant, needs_backward)
    with _NATIVE_LOCK:
        if len(_NATIVE_MEMO) < _NATIVE_MEMO_MAX:
            _NATIVE_MEMO[key] = quant
    return quant


# The token gather's geometry (its row-group split fixes the rounding) is a function of the shapes only, computed for
# this reference SM count instead of the device's, so the bits do not depend on the GPU's SM count.
_DET_GEOM_REF_SMS = 148


def pinned_gather_geometry(head_dim, dtype, width, tokens):
    """Shape-only geometry of the token gathers (forward and backward): REF_SMS is a constant, not the device's."""
    return gather_geometry(head_dim, dtype, width, tokens, num_sms=_DET_GEOM_REF_SMS)


def _num_sms(device_index: int) -> int:
    return _device.num_sms(device_index)       # (cached per device there; a forced test row overrides it)


def _nonempty(t):
    """``t``, or None when it is None or empty (a Phase-1 field that was not requested)."""
    return t if t is not None and t.numel() > 0 else None


_NO_FILL_LOCK = threading.Lock()
_NO_FILL_DEPTH = 0
_NO_FILL_PREV = None


@contextlib.contextmanager
def _no_det_fill():
    """Allocate the op's own buffers without torch's deterministic NaN fill.

    Under ``torch.use_deterministic_algorithms(True)`` torch fills every ``torch.empty`` with
    NaN (``torch.utils.deterministic.fill_uninitialized_memory``, default True). Every buffer
    this module allocates is written by its kernels before anything reads it (allocator-poison
    tested), so the fill only costs bandwidth and launches. Results are unchanged.

    The flag is process-global: a depth counter under a lock keeps it False while any SDM call
    (on any thread, nested or overlapping) is inside, and the outermost exit restores the value
    seen at the first entry. Other threads' allocations in that window skip the fill as well,
    which never changes results.
    """
    global _NO_FILL_DEPTH, _NO_FILL_PREV
    det = torch.utils.deterministic
    with _NO_FILL_LOCK:
        if _NO_FILL_DEPTH == 0:
            _NO_FILL_PREV = det.fill_uninitialized_memory
            det.fill_uninitialized_memory = False
        _NO_FILL_DEPTH += 1
    try:
        yield
    finally:
        with _NO_FILL_LOCK:
            _NO_FILL_DEPTH -= 1
            if _NO_FILL_DEPTH == 0:
                det.fill_uninitialized_memory = _NO_FILL_PREV
                _NO_FILL_PREV = None


@contextlib.contextmanager
def _kernel_context():
    """The context of the op's own work: no torch dispatch mode sees its internal ops (the
    op saves its own backward state, so activation-checkpoint policies have nothing to
    cache there), autocast is off (the backward matmuls keep their float32 inputs), and the
    op's scratch is allocated without the deterministic NaN fill (``_no_det_fill``)."""
    with _disable_current_modes(), torch.autocast("cuda", enabled=False), _no_det_fill():
        yield


def _sdm_forward(memory, k_idx, k_val, v, beta, g, q_idx, q_val, lay: _Layout,
                 snapshot_quant: str, save: bool, grad_bank_fp32=None, info=None):
    """Runs the forward and returns the readings [P * T, hd] and, when ``save``, the
    tensors the backward reads (see ``SDMWriteRead.backward``).

    Without ``save`` the steps run in slabs and the streams in groups (see SLAB_BYTES,
    NOGRAD_SLAB_FACTOR and STEP_BYTES). Both are exact: Phase 1 treats every chunk on its own,
    and streams own disjoint slots. A group runs with global slot ids, which is Phase 1's
    shared-bank mode. grad_bank_fp32: the op's setting (saving calls; None: unknown).
    """
    inputs = (k_idx, k_val, v, beta, g, q_idx, q_val)
    if save:
        _record(streams=lay.P, steps=lay.Nt, slab_steps=lay.Nt, group_streams=lay.P)
        return _forward.forward_steps(memory, *inputs, lay, snapshot_quant, True, (0, lay.Nt),
                                      grad_bank_fp32=grad_bank_fp32, info=info)
    es = memory.element_size()
    group = max(1, min(lay.P, STEP_BYTES // (lay.C * lay.W * lay.hd * es)))
    # About 64 B per slot entry (staged ids and weights, Phase-1 state and scratch), 16 B
    # per element of the chunk matrices (A, QK, X, QX) and two token rows (v, readings).
    per_step = group * lay.C * ((lay.W + lay.R) * 64 + 16 * lay.C + 2 * lay.hd * es)
    fill = -(-SLAB_CHUNKS_PER_SM * _num_sms(memory.device.index) // group)
    slab = min(lay.Nt, max(1, fill, NOGRAD_SLAB_FACTOR * SLAB_BYTES // per_step))
    _record(streams=lay.P, steps=lay.Nt, slab_steps=slab, group_streams=group)
    readings = torch.empty(lay.P * lay.T, lay.hd, device=memory.device, dtype=memory.dtype)
    stream_rows = [t.reshape(lay.P * lay.T, -1) for t in inputs]
    for p0 in range(0, lay.P, group):
        rows = slice(p0 * lay.T, min(lay.P, p0 + group) * lay.T)
        sub = lay if group == lay.P else lay._replace(
            P=rows.stop // lay.T - p0, stream_offset=0, num_keys=memory.shape[0])
        for s0 in range(0, lay.Nt, slab):
            _forward.forward_steps(memory, *(t[rows] for t in stream_rows), sub, "none", False,
                                   (s0, min(lay.Nt, s0 + slab)), readings[rows],
                                   geom_tokens=lay.P * lay.C)
    return readings, None


BF16_ROUTE_MSG = (
    'sparse_delta_memory: snapshot_quant="bf16" on a float32 bank with head_dim {hd}, chunk_size {C} and {rows} compact '
    'snapshot rows has no CUDA route on this device ({dev}); it runs only on the CUDA kernels, with no fallback. '
    'snapshot_quant="none" keeps a float32 snapshot, and calls without a backward (torch.no_grad(), inference) are not '
    'limited.')


def refuse_bf16_without_route(memory, C, rows) -> None:
    """Raises UnsupportedConfigurationError (BF16_ROUTE_MSG) for a saving "bf16" call on a float32 bank (``rows`` = P * Nt
    * C * W compact snapshot rows) without its native CUDA route (forward.bf16_route; no fallback)."""
    if not _forward.bf16_route(memory, C, rows):
        raise UnsupportedConfigurationError(BF16_ROUTE_MSG.format(hd=int(memory.shape[-1]), C=int(C), rows=int(rows),
                                                                  dev=str(getattr(memory, "device", "?"))))


def effective_snapshot_quant(memory, snapshot_quant, may_need_backward=True):
    """The snapshot mode a call runs: "bf16" runs as "none" on a bank that is not float32 (its snapshot rows are bf16
    already: the same rows, the same route) and in a call that cannot need a backward (an unrecorded direct apply runs
    exactly 'none''s call, as a no-grad call does); every other mode as given."""
    if snapshot_quant == "bf16" and (not may_need_backward or getattr(memory, "dtype", None) != torch.float32):
        return "none"
    return snapshot_quant


class SDMWriteRead(torch.autograd.Function):
    """Chunk-wise gated delta-rule write/read with an exact intra-chunk recurrence.

    Per token t (the reference token loop, which the chunked form reproduces up to
    floating-point rounding):
        at_k      = bank[k_idx]
        read      = at_k * exp(g)                  (only the written rows decay)
        retrieved = sum_w k_val_w * read_w
        delta_v   = beta * (v - retrieved)
        bank[k]  += read - at_k + k_val (x) delta_v
        readings  = sum_r q_val_r * bank[q_idx_r]  (after the write)

    Call through ``sdm_write_read``.
    """

    @staticmethod
    def forward(ctx, memory, k_idx, k_val, v, beta, g, q_idx, q_val, chunk_size,
                num_memory_slots, num_parallel, snapshot_quant, grad_bank_fp32=False, lay=None):
        # Autograd records this call (grad mode on and an input requires grad: a backward may follow) iff its node has
        # next edges; cuda_graph's stand-in ctx (no next_functions) serves calls that may need a backward.
        edges = getattr(ctx, "next_functions", None)
        recorded = edges is None or len(edges) > 0
        if lay is None:   # (sdm_write_read passes its own; the graph runner's capture and a direct apply do not)
            lay = _layout(memory, k_idx, k_val, v, beta, g, q_idx, q_val, chunk_size, num_memory_slots, num_parallel)
        # Every call here runs the saving forward (an unrecorded one too); a recorded one also needs the CUDA backward.
        # The mode it runs: "bf16" as "none" on a bank that is not float32 and in an unrecorded call (_admit).
        snapshot_quant = _admit(memory, lay, snapshot_quant, recorded, True, grad_bank_fp32)
        info = {}
        with _kernel_context():
            readings, saved = _sdm_forward(memory, k_idx, k_val, v, beta, g, q_idx, q_val,
                                           lay, snapshot_quant, save=True,
                                           grad_bank_fp32=bool(grad_bank_fp32), info=info)
        ctx.save_for_backward(*saved)
        ctx.lay = lay
        ctx.snapshot_quant = snapshot_quant
        ctx.quant_layout = info.get("quant_layout", 0)   # the compact-snapshot layout written (2: compact rows, 0: per entry)
        ctx.grad_bank_fp32 = bool(grad_bank_fp32)
        ctx.inputs = [(t.shape, t.dtype) for t in (k_val, v, beta, g, q_val)]
        return readings

    @staticmethod
    def backward(ctx, grad_readings):
        # The forward advanced the bank in place and this backward restores it, so a second
        # backward through the same call (retain_graph=True) would start from the restored bank.
        # The cuda_graph runner's stand-in ctx is exempt (cuda_graph._Ctx.repeated_backward_ok):
        # the runner manages that ctx's state itself (see _Ctx).
        if not getattr(ctx, "repeated_backward_ok", False):
            if getattr(ctx, "backward_done", False):
                raise RuntimeError("sdm_write_read: a second backward through one call is not "
                                   "supported (retain_graph=True): the first backward restored the "
                                   "bank in place. Run the forward again, or use the cuda_graph "
                                   "option.")
            ctx.backward_done = True
        saved = ctx.saved_tensors  # runs any pending activation-checkpoint recompute
        with _kernel_context():
            grads = _backward.sdm_backward(ctx, saved, grad_readings)
        return (*grads, None)   # (no gradient for lay; autograd drops it for a call that did not pass lay)




def needs_backward(*tensors: torch.Tensor) -> bool:
    """Whether a call on ``tensors`` may be differentiated: grad mode is on and one of them
    requires grad. This covers a non-reentrant activation-checkpoint forward (which drops
    what is saved) and every recompute. A reentrant checkpoint runs its first forward
    without grad mode and recomputes it before the backward."""
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def sdm_write_read(memory, k_idx, k_val, v, beta, g, q_idx, q_val, *, chunk_size: int = DEFAULT_CHUNK,
                   num_memory_slots: int = 0, num_parallel: int = 1,
                   snapshot_quant: str = "none", key_weighted_decay: bool | None = None,
                   shared_bank: bool = False, grad_bank_fp32: bool = False):
    """Chunked SDM write/read over P parallel streams; returns the readings [P * T, hd].

    Args:
        memory: [num_slots, hd] bfloat16 or float32 bank, advanced in place to its final
            state. The backward restores it: exactly with snapshot_quant "none", and the
            written rows to their dequantized snapshot with "fp8", "int4" or "bf16".
        k_idx, k_val: [P, T, W] (or [P * T, W]) write slots and weights; the W slots of a
            token must be distinct. Indices are int32 or int64 bank rows.
        v: [..., hd] values; beta: [..., 1] write strengths.
        g: [..., 1] per-token log-decay, or [..., W] one per write.
        q_idx, q_val: [..., R] read slots and weights.
        chunk_size: tokens per chunk C, 16..256 (default 64); T need not be a multiple of it.
        num_memory_slots: slots per stream; 0 means one bank shared by all streams. With
            num_parallel streams of their own, stream p owns rows [p * sph, (p + 1) * sph).
        num_parallel: number of streams P; streams must not share slots.
        shared_bank: required for several streams over one shared bank (num_memory_slots 0
            or the bank's size with num_parallel > 1). It asserts that no two streams select
            the same slot anywhere in the call (disjoint slot sets), which the op does not
            check. Two streams that write one slot in the same chunk step lose one of the two
            updates (as in the previous release); any other overlap (a read
            and a write of one slot, or one slot in different steps) loses no update but
            leaves the step-synchronous order: the no-grad path may run one stream's steps
            before another's.
        snapshot_quant: storage of the backward snapshot, "none", "fp8", "int4" or "bf16"
            (bfloat16 rows, round to nearest even: on a float32 bank it halves the snapshot;
            a bfloat16 bank runs it as "none", its snapshot being bf16 already).
        key_weighted_decay: whether g holds one gate per write; None infers it from g.
        grad_bank_fp32: accumulate the backward's bank gradient (and its gathered grad rows) in
            float32 with one cast back to the bank dtype at the end (default False; a no-op for a
            float32 bank). Forward and snapshot are unchanged; the other gradients change only
            through the bank gradient's rounding, which feeds them.

    k_val, v and q_val are used in the bank dtype, beta and g in float32. A call that does
    not need a backward (``needs_backward``) saves nothing, and its working memory does
    not grow with T. The kernels are deterministic: rows that several entries of a chunk write
    (read) are summed by one program per column slice in a fixed order, and so are Phase 1's pair
    sums.
    """
    if snapshot_quant not in SNAPSHOT_QUANT_MODES:
        raise ValueError(f"snapshot_quant must be one of {SNAPSHOT_QUANT_MODES}, "
                         f"got {snapshot_quant!r}")
    if key_weighted_decay is not None and g.shape[-1] != (k_idx.shape[-1]
                                                          if key_weighted_decay else 1):
        raise ValueError(f"key_weighted_decay={key_weighted_decay} does not match g of "
                         f"shape {tuple(g.shape)}")
    args = (memory, k_idx, k_val, v, beta, g, q_idx, q_val, chunk_size, num_memory_slots,
            num_parallel)
    # Validates the call once (the training forward takes this layout), so malformed arguments get their own errors;
    # then the shared-bank opt-in, then the route (_admit, in the forward for a training call).
    lay = _layout(*args)
    if lay.P > 1 and lay.stream_offset == 0 and not shared_bank:
        raise ValueError(
            f"num_parallel={lay.P} streams over one shared bank of {memory.shape[0]} slots: "
            "pass shared_bank=True to assert that the streams select disjoint slots for the "
            "whole call (the op does not check: two streams writing one slot in the same "
            "chunk step lose an update, and any other overlap leaves the step-synchronous "
            "order), or give each stream its own slot range (num_memory_slots = bank slots "
            "// num_parallel).")
    if needs_backward(memory, k_val, v, beta, g, q_val):
        return SDMWriteRead.apply(*args, effective_snapshot_quant(memory, snapshot_quant), grad_bank_fp32, lay)
    _admit(memory, lay)   # the no-grad forward's route, after every check above, before any work
    with torch.no_grad(), _kernel_context():
        return _sdm_forward(*args[:8], lay, snapshot_quant, save=False)[0]


# ============================================================
# Layer I/O: head split / merge and the per-batch bank expand
# ============================================================
#   head_split : dst_row = (b*H + h)*T + t  <-  src_row = (b*T + t)*H + h
#   head_merge : dst_row = (b*T + t)*H + h  <-  src_row = (b*H + h)*T + t
# Whole contiguous rows are copied, so forward and backward are byte-exact
# equivalents of the einops rearranges.
_HEADPERM_ROWS_PER_CTA = 32  # narrow bf16 rows are packed this many per program
_HEADPERM_PACKED_WARPS = 2


@triton.jit
def _bht_permute_kernel(
    src_ptr, dst_ptr,
    H, T, CD,
    BLOCK: tl.constexpr,
    SPLIT: tl.constexpr,   # True: [B,T,H*CD] -> [B*H,T,CD]; False: the inverse
):
    row = tl.program_id(0)
    if SPLIT:
        t = row % T
        bh = row // T
        h = bh % H
        b = bh // H
        srow = (b * T + t) * H + h
    else:
        h = row % H
        bt = row // H
        t = bt % T
        b = bt // T
        srow = (b * H + h) * T + t
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < CD
    v = tl.load(src_ptr + srow.to(tl.int64) * CD + offs, mask=m)
    tl.store(dst_ptr + row.to(tl.int64) * CD + offs, v, mask=m)


@triton.jit
def _bht_permute_packed_kernel(
    src_ptr, dst_ptr,
    H, T, CD, N,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Copies ROWS narrow rows per program; row addresses are computed once per row."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)[None, :]
    valid_rows = rows < N
    if SPLIT:
        t = rows % T
        bh = rows // T
        h = bh % H
        b = bh // H
        src_rows = (b * T + t) * H + h
    else:
        h = rows % H
        bt = rows // H
        t = bt % T
        b = bt // T
        src_rows = (b * H + h) * T + t
    mask = valid_rows & (cols < CD)
    values = tl.load(src_ptr + src_rows.to(tl.int64) * CD + cols, mask=mask)
    tl.store(dst_ptr + rows.to(tl.int64) * CD + cols, values, mask=mask)


def _bht_permute(x, Bb, H, T, CD, split):
    n = Bb * H * T
    with _no_det_fill():
        out = torch.empty(n * CD, device=x.device, dtype=x.dtype)
    BLOCK = min(2048, triton.next_power_of_2(CD))
    rows = _HEADPERM_ROWS_PER_CTA
    if x.dtype == torch.bfloat16 and CD in (64, 128):
        _device.tbind(_bht_permute_packed_kernel, (triton.cdiv(n, rows), triton.cdiv(CD, BLOCK)), x)(
            x, out, H=H, T=T, CD=CD, N=n, BLOCK=BLOCK, ROWS=rows, SPLIT=split,
            num_warps=_HEADPERM_PACKED_WARPS)
    else:
        _device.tbind(_bht_permute_kernel, (n, triton.cdiv(CD, BLOCK)), x)(
            x, out, H=H, T=T, CD=CD, BLOCK=BLOCK, SPLIT=split,
            num_warps=4 if BLOCK >= 1024 else 1)
    return out


class _HeadSplit(torch.autograd.Function):
    """[B, T, H*CD] -> [B*H, T, CD]  (rearrange 'b t (h k) -> (b h) t k')."""

    @staticmethod
    def forward(ctx, x, H):
        Bb, T, HK = x.shape
        CD = HK // H
        ctx.dims = (Bb, H, T, CD)
        return _bht_permute(x.contiguous(), Bb, H, T, CD, True).view(Bb * H, T, CD)

    @staticmethod
    def backward(ctx, g):
        # Through the sibling Function so the gradient stays differentiable.
        return _HeadMerge.apply(g, ctx.dims[1]), None


class _HeadMerge(torch.autograd.Function):
    """[B*H, T, CD] -> [B, T, H*CD]  (rearrange '(b h) t d -> b t (h d)')."""

    @staticmethod
    def forward(ctx, x, H):
        BH, T, CD = x.shape
        Bb = BH // H
        ctx.dims = (Bb, H, T, CD)
        return _bht_permute(x.contiguous(), Bb, H, T, CD, False).view(Bb, T, H * CD)

    @staticmethod
    def backward(ctx, g):
        return _HeadSplit.apply(g, ctx.dims[1]), None


def head_split(x, H):
    """rearrange(x, 'b t (h k) -> (b h) t k', h=H) as a coalesced row copy.

    Uses the reshape/transpose reference when it is a free view (H == 1, or T == 1 as in
    decode), for strided or empty inputs, and when H does not divide the last dimension
    (the reshape then raises like einops).
    """
    if (H == 1 or x.shape[1] == 1 or not x.is_contiguous()
            or x.numel() == 0 or x.shape[2] % H != 0):
        return x.reshape(x.shape[0], x.shape[1], H, -1).transpose(1, 2).reshape(
            x.shape[0] * H, x.shape[1], -1)
    return _HeadSplit.apply(x, H)


def head_merge(x, H):
    """rearrange(x, '(b h) t d -> b t (h d)', h=H) as a coalesced row copy.

    Same reference fallbacks as ``head_split``.
    """
    BH, T, CD = x.shape
    if (H == 1 or T == 1 or not x.is_contiguous()
            or x.numel() == 0 or BH % H != 0):
        return x.reshape(BH // H, H, T, CD).transpose(1, 2).reshape(BH // H, T, H * CD)
    return _HeadMerge.apply(x, H)


@triton.jit
def _sum_batch_kernel(x_ptr, out_ptr,
                      B: tl.constexpr, S, D,
                      stride_xb, stride_xs, stride_xd,
                      stride_os, stride_od,
                      BLOCK_D: tl.constexpr):
    """Sum a [B, S, D] tensor over the batch dim into [S, D].

    Grid = (S, cdiv(D, BLOCK_D)). All offsets are int64 (b * stride_xb exceeds int32 for
    large banks); the pointer is advanced by stride_xb per batch element so the large
    product is never formed.
    """
    s = tl.program_id(0).to(tl.int64)
    d64 = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    d_mask = d64 < D
    ptr = x_ptr + s * stride_xs + d64 * stride_xd
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _b in tl.static_range(B):
        acc += tl.load(ptr, mask=d_mask, other=0.0).to(tl.float32)
        ptr += stride_xb
    out_off = s * stride_os + d64 * stride_od
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=d_mask)


def triton_sum_dim0(x):
    """Sum over dim 0 of a [B, S, D] tensor -> [S, D] (int64-safe, any D)."""
    assert x.ndim == 3 and x.dtype in (torch.bfloat16, torch.float32)
    B, S, D = x.shape
    with _no_det_fill():
        out = torch.empty(S, D, device=x.device, dtype=x.dtype)
    BLOCK_D = min(1024, triton.next_power_of_2(D))
    _device.tbind(_sum_batch_kernel, (S, triton.cdiv(D, BLOCK_D)), x)(
        x, out,
        B=B, S=S, D=D,
        stride_xb=x.stride(0), stride_xs=x.stride(1), stride_xd=x.stride(2),
        stride_os=out.stride(0), stride_od=out.stride(1),
        BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out


@triton.jit
def _bcast_rows_kernel(
    src_ptr, out_ptr, S,
    D: tl.constexpr, B: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per source row: load it once, store it into every batch copy."""
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    m = d < D
    x = tl.load(src_ptr + row * D + d, mask=m)
    for b in tl.static_range(B):
        tl.store(out_ptr + (b * S + row) * D + d, x, mask=m)


class _ExpandMemoryBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, B):
        S, D = param.shape
        if not param.is_contiguous():
            raise ValueError("expand_memory_batch needs a contiguous [S, D] parameter")
        ctx.dims = (B, S, D)
        with _no_det_fill():
            out = torch.empty(B * S, D, dtype=param.dtype, device=param.device)
        _device.tbind(_bcast_rows_kernel, (S,), param)(
            param, out, S, D=D, B=B, BLOCK_D=triton.next_power_of_2(D), num_warps=8)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return triton_sum_dim0(grad_out.view(ctx.dims)), None


def expand_memory_batch(param, B):
    """``param.unsqueeze(0).expand(B, S, D).reshape(B * S, D).clone()``: private, mutable
    per-batch copies of the [S, D] bank (the op advances them in place), reading the
    parameter once; the backward sums the B gradient copies in one pass."""
    return _ExpandMemoryBatch.apply(param, B)


# The forward steps and the backward use this module's helpers and entry points (looked up here at call time).
from . import backward as _backward, forward as _forward  # noqa: E402
