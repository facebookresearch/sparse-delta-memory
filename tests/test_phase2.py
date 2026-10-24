# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Phase-2 movers (``movers``: the CUDA DW and GDS) against float64 references, one step at a time. (2.0.0: the
Triton movers, Phase A and quantizer of ``phase2_kernels`` are gone; the CUDA Phase A is checked through the op, e.g.
tests/test_f64_oracle.py, tests/test_tail_pa.py, tests/test_bf16_entries.py.)

Each case has P streams whose chunks reuse slots, so the movers take both the single-entry
and the repeated-slot path (group sums), and a padded last chunk. Pad tokens carry NaN and point at live
slots, so a read or write through them shows up. Every step starts the kernel and the
reference from the same state, so the gates measure the error of one step.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import torch

from sdm_testlib import (bits_equal, rel_max, rel_rms, reference_dequantize,
                         reference_quantize, requires_cuda)

pytestmark = requires_cuda

FP32_GATE = 1e-6
BF16_RMS, BF16_MAX = 1e-2, 5e-2   # bf16 bank rows: one rounding per store
TF32_GATE = 2e-3                  # tf32 dots of the dual matmul
READINGS_BF16_GATE = 5e-3         # tf32 dots, then a bf16 store

# (head_dim, width, chunk): tails of every tile width, odd chunks and widths, W = 1.
SHAPES = [(64, 4, 16), (65, 3, 17), (129, 5, 16), (961, 2, 16), (1025, 2, 16), (8, 1, 16),
          (128, 64, 16)]
DTYPES = [torch.float32, torch.bfloat16]
DISTS = ["distinct", "hot", "dup"]
# Determinism: ENT > 1 geometries (small head dims), the flagship head dim, odd shapes; chunks
# long enough for groups of several 32-position windows, and (C = 160) for "hot" groups of
# more than 127 members, whose packed counts are clipped.
DET_SHAPES = [(120, 64, 16), (128, 8, 64), (240, 16, 32), (1024, 8, 64), (1025, 3, 17),
              (64, 2, 160)]


def _rows16(hd):
    """The head splits into equal column slices of 16-byte rows in both dtypes (mover_da2_cuda's predicate): the
    per-entry (layout 0) CUDA DW these unit tests drive. Tail heads run only on mover_fx_cuda's bank-direct DW, on the
    layouts the forward tags (tests/test_fx_tails.py, tests/test_tail_bwd.py)."""
    nsl = -(-hd // 128)
    return hd % nsl == 0 and (hd // nsl) % 8 == 0


DW_SHAPES = [s for s in SHAPES if _rows16(s[0])]
DET_DW_SHAPES = [s for s in DET_SHAPES if _rows16(s[0])]
REPEATS = 6


def p2():
    from sparse_delta_memory import phase2_kernels
    return phase2_kernels


def mv():
    """The DW / GDS step-launcher factories of the op (sparse_delta_memory.movers): the CUDA movers (the only route
    since 2.0.0)."""
    from sparse_delta_memory import movers
    return movers


def _da2_gds_built():
    """How many CUDA GDS launchers the op has built (movers.STATS)."""
    return int(mv().STATS.get("da2_gds", 0))


def _dots_after_step0(built_before, hd):
    """The CUDA GDS launcher sums a wide head's per-slice dot partials (head_dim > 128: several
    128-column slices) into d_q_w once, when step 0 runs (cuda/mover_da2_cuda.py make_gds; the
    op's backward reads d_q_w after its last step), where phase2_kernels' launcher writes each
    step's dots at that step. True when the launcher built since ``built_before`` is the CUDA one
    and the head is wide: its dots are checked after step 0."""
    return hd > 128 and _da2_gds_built() > built_before


def assert_close(ref, got, dtype, what):
    """``got`` against the float64 ``ref``: FP32_GATE on rel_max for fp32 data, the bf16
    bounds otherwise."""
    if dtype == torch.float32:
        err = rel_max(ref, got)
        assert err <= FP32_GATE, f"{what}: rel_max {err:.3g} > {FP32_GATE:g}"
    else:
        rms, mx = rel_rms(ref, got), rel_max(ref, got)
        assert rms <= BF16_RMS and mx <= BF16_MAX, f"{what}: rel_rms {rms:.3g}, rel_max {mx:.3g}"


def assert_zero(t, what):
    assert bits_equal(t, torch.zeros_like(t)), f"{what}: padding is not exactly +0.0"


class Case:
    """Random Phase-2 inputs: ``steps`` steps of P chunks of C tokens, time-major (chunk
    ti * P + p). Tokens of a stream at or past ``valid_tokens`` are padding.

    Slot distributions (``dist``): "distinct" draws W (R) distinct slots per token from the
    stream's ``sph`` slots; "hot" also points lanes 0 and 1 of every token at two shared
    slots (groups of C entries, longer than one window) and reads slot 0 of its writes, so
    groups mix K and Q entries; "dup" also repeats each token's lane 0 slot in its last lane
    (repeated K entries of one token, which break the op contract but must stay
    deterministic). The sorted Phase-1 rows, positions and flags are built from the slots
    with torch (``sorted_rows``)."""

    def __init__(self, *, P=2, C=16, steps=3, W=4, R=3, hd=64, dtype=torch.float32, sph=24,
                 pad=5, seed=0, dist="distinct"):
        self.P, self.C, self.steps, self.W, self.R, self.hd = P, C, steps, W, R, hd
        self.dtype = dtype
        self.N, self.SC = steps * P, P * C
        self.valid_tokens = steps * C - pad
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        step = torch.arange(self.N, device="cuda") // P
        self.valid = step[:, None] * C + torch.arange(C, device="cuda") < self.valid_tokens
        self.slots = P * sph
        lo = 3 if dist == "hot" else 0
        self.k_slot = self._slots(W, sph, lo)
        self.q_slot = self._slots(R, sph, lo)
        base = (torch.arange(self.N, device="cuda") % P * sph)[:, None]
        if dist == "hot":
            self.k_slot[..., 0] = base
            if W > 1:
                self.k_slot[..., 1] = base + 1
            self.q_slot[..., 0] = base
            if R > 1:
                self.q_slot[..., 1] = base + 2
        elif dist == "dup":
            self.k_slot[..., -1] = self.k_slot[..., 0]
        elif dist != "distinct":
            raise ValueError(dist)
        self.k_slot = self.k_slot.contiguous()
        self.q_slot = self.q_slot.contiguous()
        self.resort()

    def resort(self):
        """(Re)builds the Phase-1 rows of the current slots: the combined row, its maps and the sorted kind rows."""
        from phase1_reference import sorted_kind_rows_reference
        (self.spay, self.pos_k, self.pos_q, self.q_last, self.q_only, self.is_last,
         self.is_only) = sorted_rows(self)
        self.spay_k, self.spay_q = sorted_kind_rows_reference(self.spay, self.C, self.W, self.R)

    def randn(self, *shape, dtype=None):
        return torch.randn(*shape, device="cuda", generator=self.gen).to(dtype or self.dtype)

    def entry_valid(self, width):
        return self.valid[:, :, None].expand(self.N, self.C, width)

    def rows(self, dtype=torch.float32):
        """[N * C, hd] token rows, NaN on pad tokens."""
        t = self.randn(self.N * self.C, self.hd, dtype=dtype)
        t.view(self.N, self.C, self.hd)[~self.valid] = float("nan")
        return t

    def entries(self, width, lo=None, hi=None, dtype=torch.float32):
        """[N, C, width] per-entry values, normal or uniform in [lo, hi); NaN on pad."""
        shape = (self.N, self.C, width)
        if lo is None:
            t = self.randn(*shape, dtype=dtype)
        else:
            t = (lo + (hi - lo) * torch.rand(*shape, device="cuda", generator=self.gen)).to(dtype)
        t[~self.entry_valid(width)] = float("nan")
        return t

    def step(self, t, ti, width=1):
        """Step ti of a per-entry [N, C, width], per-token [N * C] or row [N * C * width, hd]
        tensor."""
        n = self.SC * width
        return (t if t.dim() == 2 else t.reshape(-1))[ti * n:(ti + 1) * n]

    def sorted_args(self, side):
        """Keyword arguments of the movers for the K ("k") or Q ("q") side."""
        return dict(sorted_k=self.spay_k) if side == "k" else dict(sorted_q=self.spay_q)

    def _slots(self, width, sph, lo=0):
        """Distinct slots per token from slots lo .. sph - 1 of the chunk's stream range."""
        local = torch.rand(self.N, self.C, sph - lo, device="cuda",
                           generator=self.gen).argsort(-1)
        base = (torch.arange(self.N, device="cuda") % self.P * sph + lo)[:, None, None]
        return (local[..., :width] + base).contiguous()


def sorted_rows(case):
    """The Phase-1 sorted payload rows [N, row] of the case's slots (the layout documented in
    cuda/phase1_common.cuh), the group maps (``group_maps_reference``) and the K flags
    is_last / is_only, all from torch ops."""
    from phase1_reference import group_maps_reference
    from sparse_delta_memory.cuda.phase1_cuda import POS_MASK
    N, C, W, R = case.N, case.C, case.W, case.R
    n = C * (W + R)
    row = -(-n // 16) * 16
    dev = case.k_slot.device
    ids = torch.cat([case.k_slot, case.q_slot], 2).to(torch.int64).reshape(N, n)
    t = torch.arange(C, device=dev).view(1, C, 1).expand(N, C, W + R).reshape(N, n)
    j = torch.arange(W + R, device=dev).view(1, 1, -1).expand(N, C, W + R).reshape(N, n)
    q = (j >= W).to(torch.int64)
    invalid = (~case.valid[:, :, None].expand(N, C, W + R)).reshape(N, n).to(torch.int64)
    pay = t << 9 | q << 8 | torch.where(q == 1, j - W, j) | invalid << 17
    key = torch.where(invalid == 1, torch.full_like(ids, 1 << 40), ids)
    order = torch.sort(key, dim=1, stable=True).indices
    pay, key = pay.gather(1, order), key.gather(1, order)
    inv = (pay >> 17) & 1
    prev_key = torch.cat([torch.full_like(key[:, :1], -1), key[:, :-1]], 1)
    prev_inv = torch.cat([torch.ones_like(inv[:, :1]), inv[:, :-1]], 1)
    head = ((key != prev_key) | (inv == 1) | (prev_inv == 1)).to(torch.int64)
    is_k = ((pay >> 8) & 1 == 0) & (inv == 0)
    gid = head.cumsum(1) - 1
    r = torch.arange(n, device=dev).expand(N, n)
    last_k = torch.full((N, n), -1, device=dev, dtype=torch.int64).scatter_reduce(
        1, gid, torch.where(is_k, r, -1), "amax")
    count_k = torch.zeros(N, n, device=dev, dtype=torch.int64).scatter_add(
        1, gid, is_k.to(torch.int64))
    lk = is_k & (last_k.gather(1, gid) == r)
    pay = pay | head << 31 | lk.to(torch.int64) << 30
    full = torch.full((N, row), (1 << 31) | (1 << 17), device=dev, dtype=torch.int64)
    full[:, :n] = pay
    spay = torch.where(full >= 1 << 31, full - (1 << 32), full).to(torch.int32).contiguous()
    pos_k, pos_q, q_last, q_only = group_maps_reference(spay, C, W, R)
    k_pos = (pos_k.reshape(N, C * W) & POS_MASK).to(torch.int64)
    flag_last = lk.gather(1, k_pos)
    flag_only = flag_last & (count_k.gather(1, gid).gather(1, k_pos) == 1)
    return (spay, pos_k.contiguous(), pos_q.contiguous(), q_last, q_only,
            flag_last.to(torch.int8).view(N, C, W), flag_only.to(torch.int8).view(N, C, W))


# ------------------------------------------------------------------------------ DW
def ref_dw(case, bank, snap, dv, slot, ok, cpe, last, only, kv, rex):
    """One DW step in float64 over the step's [SC * W] entries."""
    out = bank.double()
    dv = dv.double()[torch.arange(slot.numel(), device="cuda") // case.W]
    snap, cpe = snap.double(), cpe.double()[:, None]
    write = (kv.double() * rex.double().exp())[:, None] * dv
    store = ok & (only != 0)
    add = ok & (only == 0)
    out[slot[store]] = snap[store] * cpe[store] + write[store]
    corr = torch.where(last[:, None] != 0, cpe - 1.0, 0.0)
    out.index_add_(0, slot[add], (write + corr * snap)[add])
    return out


def dw_inputs(case):
    return (case.entries(case.W, 0.2, 1.0, case.dtype), case.entries(case.W, dtype=case.dtype),
            case.entries(case.W, -1.5, 0.0), case.rows())


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("snapshot_rows", ["step", "all"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd, W, C", DW_SHAPES)
def test_dw_matches_float64(hd, W, C, dtype, snapshot_rows, dist):
    case = Case(hd=hd, W=W, C=C, dtype=dtype, sph=W + 20, dist=dist)
    cpe, k_val, rex, delta_v = dw_inputs(case)
    bank = case.randn(case.slots, hd)
    rows = case.SC * W * (1 if snapshot_rows == "step" else case.steps)
    snapshot = torch.empty(rows, hd, device="cuda", dtype=dtype)
    launch = mv().make_mover_dw(bank, snapshot, delta_v, case.k_slot, cpe, k_val, rex,
                                chunk_size=C, num_parallel=case.P,
                                valid_tokens=case.valid_tokens, **case.sorted_args("k"))
    for ti in range(case.steps):
        slot, ok = case.step(case.k_slot, ti, W), case.step(case.entry_valid(W), ti, W)
        snap = bank[slot].masked_fill(~ok[:, None], float("nan"))
        (snapshot if snapshot_rows == "step" else case.step(snapshot, ti, W)).copy_(snap)
        ref = ref_dw(case, bank, snap, case.step(delta_v, ti), slot, ok,
                     *(case.step(t, ti, W) for t in (cpe, case.is_last, case.is_only, k_val, rex)))
        launch(ti)
        assert_close(ref, bank, dtype, f"bank after step {ti}")
        bank.copy_(ref)


# ------------------------------------------------------------------------------ Phase A
# ------------------------------------------------------------------------------ GDS
def ref_gds(case, grad, bank, gr, slot, ok, qv, logq):
    """One GDS step in float64; returns (grad, d_q_w)."""
    tok = torch.arange(slot.numel(), device="cuda") // case.R
    grad, gr, mem = grad.double(), gr.double()[tok], bank.double()[slot]
    w = (qv.double() * logq.double().exp())[:, None]
    grad.index_add_(0, slot[ok], (w * gr)[ok])
    dot = torch.where(ok, (mem * gr).sum(-1), torch.zeros((), dtype=torch.float64, device="cuda"))
    return grad, dot


@pytest.mark.parametrize("dist", ["distinct", "hot"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd, R, C", SHAPES)
def test_gds_matches_float64(hd, R, C, dtype, dist):
    case = Case(hd=hd, R=R, C=C, dtype=dtype, sph=R + 20, dist=dist)
    q_val, log_q = case.entries(R, dtype=dtype), case.entries(R, -1.5, 0.0)
    grad_readings = case.rows()
    bank, grad = case.randn(case.slots, hd), case.randn(case.slots, hd)
    dots = torch.full((case.N, case.C, R), float("nan"), device="cuda")
    built = _da2_gds_built()
    launch = mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots,
                                 chunk_size=C, num_parallel=case.P, valid_tokens=case.valid_tokens,
                                 **case.sorted_args("q"))
    # The CUDA GDS also takes the tail heads (129 / 961 / 1025: gx_gds_cuda's tail module), whose
    # dots of a wide head land in d_q_w when step 0 runs (_dots_after_step0): every route is checked after the loop.
    deferred, done = _dots_after_step0(built, hd), []
    for ti in reversed(range(case.steps)):
        slot, ok = case.step(case.q_slot, ti, R), case.step(case.entry_valid(R), ti, R)
        ref_grad, ref_dot = ref_gds(case, grad, bank, case.step(grad_readings, ti), slot, ok,
                                    case.step(q_val, ti, R), case.step(log_q, ti, R))
        launch(ti)
        assert_close(ref_grad, grad, dtype, f"grad after step {ti}")
        if not deferred:
            got = case.step(dots, ti, R)
            assert_close(ref_dot, got, torch.float32, f"d_q_w at step {ti}")
            assert_zero(got[~ok], f"d_q_w at step {ti}")
        done.append((ti, ref_dot, ok))
        grad.copy_(ref_grad)
    for ti, ref_dot, ok in done:   # the final d_q_w of every step, after the full step loop
        got = case.step(dots, ti, R)
        assert_close(ref_dot, got, torch.float32, f"d_q_w of step {ti} after the step loop")
        assert_zero(got[~ok], f"d_q_w of step {ti} after the step loop")


# Regression (gx_gds_cuda's tail module takes the tail heads; their wide-head
# dots land in d_q_w when step 0 runs): the fp32 GDS of several steps against float64 on heads of more than one
# 128-column slice, hd % 4 != 0 (129 / 961 / 1025: the tail module) and the even control 1024 (the regular module): the
# grad after every step and the final d_q_w of every step after the full step loop; the modern backend must take gx_gds.
@pytest.mark.parametrize("dist", ["distinct", "hot"])
@pytest.mark.parametrize("hd, R", [(129, 5), (961, 2), (1025, 2), (1024, 2)])
def test_gds_fp32_multi_step_final_dots_match_float64(hd, R, dist):
    from sparse_delta_memory.cuda import gx_gds_cuda as gxm
    from sparse_delta_memory.cuda._device import backend
    nsl, _, tail = gxm.slices(hd)
    assert nsl == -(-hd // 128) > 1 and tail == (hd % 4 != 0)
    case = Case(hd=hd, R=R, C=16, steps=4, dtype=torch.float32, sph=R + 20, seed=7, dist=dist)
    q_val, log_q = case.entries(R), case.entries(R, -1.5, 0.0)
    grad_readings = case.rows()
    bank, grad = case.randn(case.slots, hd), case.randn(case.slots, hd)
    dots = torch.full((case.N, case.C, R), float("nan"), device="cuda")
    built = _da2_gds_built()
    launch = mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots,
                                 chunk_size=case.C, num_parallel=case.P, valid_tokens=case.valid_tokens,
                                 **case.sorted_args("q"))
    if backend(bank.device) == "modern":
        assert _da2_gds_built() == built + 1, mv().STATS
        assert gxm.LAST_POLICY.get("impl") == "gx" and gxm.LAST_POLICY.get("nsl") == nsl, gxm.LAST_POLICY
    done = []
    for ti in reversed(range(case.steps)):
        slot, ok = case.step(case.q_slot, ti, R), case.step(case.entry_valid(R), ti, R)
        ref_grad, ref_dot = ref_gds(case, grad, bank, case.step(grad_readings, ti), slot, ok,
                                    case.step(q_val, ti, R), case.step(log_q, ti, R))
        launch(ti)
        assert_close(ref_grad, grad, torch.float32, f"grad after step {ti}")
        done.append((ti, ref_dot, ok))
        grad.copy_(ref_grad)
    for ti, ref_dot, ok in done:
        got = case.step(dots, ti, R)
        assert_close(ref_dot, got, torch.float32, f"d_q_w of step {ti} after the step loop")
        assert_zero(got[~ok], f"d_q_w of step {ti} after the step loop")


# ------------------------------------------------------------------------------ saturated groups
# Each token writes (reads) `width` of its stream's 2 * width slots, so a slot is hit by about
# half the tokens of a chunk: at C = 256, groups of ~128 members (some past the 127 the packed
# counts hold) that take many windows of the group-sum loop, and a last chunk of 5 tokens with
# groups of 2 or 3. (head_dim, width): tiles of ENT 1 to 16 entries over the two dtypes, and
# head_dim 17 (one element of a row per lane, scalar accesses).
SAT_SHAPES = [(128, 64), (120, 32), (64, 32), (1024, 16), (17, 8)]
SAT_DW_SHAPES = [s for s in SAT_SHAPES if _rows16(s[0])]   # the DW: 16-byte-slice heads (see _rows16)
# The movers round a saturated group's row once (an atomic add per entry would round a bf16 row
# after every add, ~128 roundings per group, past these gates of one rounding).
def assert_saturated_close(ref, got, dtype, what):
    assert_close(ref, got, dtype, what)


def saturated_case(hd, width, dtype, side, P=2, steps=2, pad=251):
    from sparse_delta_memory.cuda.phase1_cuda import POS_COUNT_MAX, POS_SIZE_SHIFT
    kw = dict(W=width, R=4) if side == "k" else dict(W=4, R=width)
    case = Case(P=P, C=256, steps=steps, hd=hd, dtype=dtype, sph=2 * width, pad=pad, seed=3,
                **kw)
    pos = case.pos_k if side == "k" else case.pos_q
    size = ((pos >> POS_SIZE_SHIFT) & POS_COUNT_MAX).reshape(case.N, -1)
    assert int(size.max()) == POS_COUNT_MAX, "no group past the packed counts"
    tail = size[-P:][case.entry_valid(width)[-P:].reshape(P, -1)]
    assert bool(((tail >= 2) & (tail <= 4)).any()), "no small group in the last chunk"
    return case


def quantized_buffers(case, mode):
    if mode == "none":
        return None, None
    entries = case.N * case.C * case.W
    width = case.hd if mode == "fp8" else (case.hd + 1) // 2
    payload = torch.empty(entries, width, device="cuda",
                          dtype=torch.float8_e4m3fn if mode == "fp8" else torch.uint8)
    return payload, torch.empty(entries, device="cuda")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd, width", SAT_DW_SHAPES)
def test_dw_saturated_groups_match_float64(hd, width, dtype):
    mode = "none"   # 2.0.0: no per-entry (layout 0) quantizer; the quantized compact rows: tests/test_quant_rows.py
    case = saturated_case(hd, width, dtype, "k")
    W = case.W
    cpe, k_val, rex, delta_v = dw_inputs(case)
    bank = case.randn(case.slots, hd)
    snapshot = torch.empty(case.SC * W, hd, device="cuda", dtype=dtype)
    payload, scale = quantized_buffers(case, mode)
    launch = mv().make_mover_dw(bank, snapshot, delta_v, case.k_slot, cpe, k_val, rex,
                                payload, scale, chunk_size=case.C,
                                num_parallel=case.P, valid_tokens=case.valid_tokens,
                                snapshot_quant=mode, **case.sorted_args("k"))
    for ti in range(case.steps):
        slot, ok = case.step(case.k_slot, ti, W), case.step(case.entry_valid(W), ti, W)
        snap = bank[slot].masked_fill(~ok[:, None], float("nan"))
        snapshot.copy_(snap)
        ref = ref_dw(case, bank, snap, case.step(delta_v, ti), slot, ok,
                     *(case.step(t, ti, W) for t in (cpe, case.is_last, case.is_only, k_val, rex)))
        launch(ti)
        assert_saturated_close(ref, bank, dtype, f"bank after step {ti}")
        bank.copy_(ref)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd, width", SAT_SHAPES)
def test_gds_saturated_groups_match_float64(hd, width, dtype):
    case = saturated_case(hd, width, dtype, "q")
    R = case.R
    q_val, log_q = case.entries(R, dtype=dtype), case.entries(R, -1.5, 0.0)
    grad_readings = case.rows()
    bank, grad = case.randn(case.slots, hd), case.randn(case.slots, hd)
    dots = torch.full((case.N, case.C, R), float("nan"), device="cuda")
    built = _da2_gds_built()
    launch = mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots,
                                 chunk_size=case.C, num_parallel=case.P,
                                 valid_tokens=case.valid_tokens, **case.sorted_args("q"))
    deferred, pending = _dots_after_step0(built, hd), []
    for ti in reversed(range(case.steps)):
        slot, ok = case.step(case.q_slot, ti, R), case.step(case.entry_valid(R), ti, R)
        ref_grad, ref_dot = ref_gds(case, grad, bank, case.step(grad_readings, ti), slot, ok,
                                    case.step(q_val, ti, R), case.step(log_q, ti, R))
        launch(ti)
        assert_saturated_close(ref_grad, grad, dtype, f"grad after step {ti}")
        if deferred:
            pending.append((ti, ref_dot, ok))
        else:
            got = case.step(dots, ti, R)
            assert_close(ref_dot, got, torch.float32, f"d_q_w at step {ti}")
            assert_zero(got[~ok], f"d_q_w at step {ti}")
        grad.copy_(ref_grad)
    for ti, ref_dot, ok in pending:   # CUDA GDS, wide head: summed when step 0 ran
        got = case.step(dots, ti, R)
        assert_close(ref_dot, got, torch.float32, f"d_q_w at step {ti} (after step 0)")
        assert_zero(got[~ok], f"d_q_w at step {ti} (after step 0)")


# The CUDA GDS (gx_gds_cuda's gx_gds_kernel; its tail module also takes the tail heads):
# launches counted per module entry (regular, tail module).
RACE_CUDA_GDS = ("gx_gds_kernel", "gx_gds_kernel/tail")
RACE_FILTER_CUDA_GDS = "gx_gds_kernel"   # compute-sanitizer --kernel-regex kns=...
# Head dims of the racecheck driver: the saturated shapes, rows of 1, 3 and 17 elements (a lane
# can hold one element of a row: tiles under 64 columns), and 33, 48 and 60 (64-column tiles, on
# both sides of 16-byte vectors);
# (head_dim, 1) also runs with every row buffer starting one element into its allocation
# (unaligned pointers: scalar accesses).
RACE_SHAPES = ([(hd, 0) for hd, _ in SAT_SHAPES] + [(1, 0), (3, 0), (33, 0), (48, 0), (60, 0)]
               + [(17, 1), (60, 1), (64, 1)])


def _offset(t, shift):
    """A copy of ``t`` starting ``shift`` elements into its allocation (``t`` itself when
    ``shift`` is 0); None stays None."""
    if t is None:
        return t
    if not shift:
        return t
    out = torch.empty(t.numel() + shift, dtype=t.dtype, device=t.device)[shift:].view(t.shape)
    return out.copy_(t)


def racecheck_driver():
    """Launches the CUDA GDS once per step on saturated read groups (both dtypes, the head dims of RACE_SHAPES; run
    under compute-sanitizer by ``test_deterministic_movers_pass_racecheck``) and on the probes below. Prints how often
    each entry of RACE_CUDA_GDS was launched."""
    from sparse_delta_memory.cuda import gx_gds_cuda
    launches = dict.fromkeys(RACE_CUDA_GDS, 0)
    for name, mod, entries in ((RACE_CUDA_GDS[0], gx_gds_cuda._module(), ("gx_gds", "gx_gds_bf")),
                               (RACE_CUDA_GDS[1], gx_gds_cuda._tail_module(), ("gx_gds",))):
        for entry in entries:
            def counted_cuda(*args, _run=getattr(mod, entry), _name=name, **kwargs):
                launches[_name] += 1
                return _run(*args, **kwargs)
            setattr(mod, entry, counted_cuda)
    for dtype in DTYPES:
        for hd, shift in RACE_SHAPES:
            qcase = saturated_case(hd, 8, dtype, "q", P=1, steps=2, pad=251)
            gds_in = (_offset(qcase.randn(qcase.slots, hd), shift),
                      _offset(qcase.randn(qcase.slots, hd), shift), _offset(qcase.rows(), shift),
                      qcase.q_slot, qcase.entries(qcase.R, dtype=dtype),
                      qcase.entries(qcase.R, -1.5, 0.0))
            gds = mv().make_mover_gds(*gds_in, torch.empty(qcase.N, qcase.C, qcase.R, device="cuda"),
                                      chunk_size=qcase.C, num_parallel=qcase.P, valid_tokens=qcase.valid_tokens,
                                      **qcase.sorted_args("q"))
            for ti in range(qcase.steps):
                gds(ti)
    # CUDA GDS probes admitted on every backend (chunks of 64: rtx admits tails up to 128, rule 0e): a regular head and the
    # tail module's one- and two-slice heads, so both CUDA counters see launches wherever the driver runs
    for dtype in DTYPES:
        for hd in (64, 17, 129):
            pc = Case(P=2, C=64, steps=2, R=4, hd=hd, dtype=dtype, sph=24, seed=5, dist="hot")
            gds = mv().make_mover_gds(pc.randn(pc.slots, hd), pc.randn(pc.slots, hd), pc.rows(), pc.q_slot,
                                      pc.entries(pc.R, dtype=dtype), pc.entries(pc.R, -1.5, 0.0),
                                      torch.empty(pc.N, pc.C, pc.R, device="cuda"), chunk_size=pc.C,
                                      num_parallel=pc.P, valid_tokens=pc.valid_tokens, **pc.sorted_args("q"))
            for ti in range(pc.steps):
                gds(ti)
    torch.cuda.synchronize()
    print("racecheck launches " + json.dumps(launches), flush=True)
    print("racecheck driver done", flush=True)


def _compute_sanitizer():
    path = shutil.which("compute-sanitizer")
    cand = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "compute-sanitizer")
    return path or (cand if os.path.exists(cand) else None)


@pytest.mark.slow
def test_deterministic_movers_pass_racecheck():
    """No shared-memory hazard in the deterministic movers: compute-sanitizer racecheck over
    ``racecheck_driver`` with the CUDA GDS filter (it runs the tail heads), whose launches must cover every kernel the filter selects. The lanes
    that hold copies of one value must agree to the last bit when a layout conversion stores
    them all to one shared address (racecheck found copies that differed, an FMA fused into
    the first cross-lane add: group sums with fp8 snapshots, per-entry dots at head_dim 3 and
    17)."""
    sanitizer = _compute_sanitizer()
    if sanitizer is None:
        pytest.skip("compute-sanitizer not found")
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [here, os.path.dirname(here), os.environ.get("PYTHONPATH", "")]))
    for filt, kernels in ((RACE_FILTER_CUDA_GDS, RACE_CUDA_GDS),):
        cmd = [sanitizer, "--tool", "racecheck", "--racecheck-report", "hazard",
               "--kernel-regex", f"kns={filt}", "--print-limit", "20", sys.executable, "-c",
               "import test_phase2; test_phase2.racecheck_driver()"]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        out = run.stdout + run.stderr
        assert run.returncode == 0 and "racecheck driver done" in out, out[-6000:]
        launches = json.loads(re.search(r"racecheck launches (\{.*\})", out).group(1))
        assert all(launches[k] > 0 and filt in k for k in kernels), launches
        summaries = re.findall(
            r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?, (\d+) warnings?\)", out)
        print(f"racecheck kns={filt}: summaries {summaries}, launches {launches}", flush=True)
        assert summaries == [("0", "0")], out[-6000:]


# ------------------------------------------------------------------------------ dual matmul
@pytest.mark.parametrize("delta_v_rows", ["step", "all"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd", [64, 65, 1025])
@pytest.mark.parametrize("C", [16, 17, 48, 64, 127, 128, 256])
def test_dual_matmul_matches_float64(C, hd, dtype, delta_v_rows, ieee_fp32):
    case = Case(C=C, hd=hd, dtype=dtype, pad=C // 3 + 1)
    n, sc = case.N * C, case.SC
    x, qx = (case.randn(case.N, C, C, dtype=torch.float32) for _ in range(2))
    v, beta = case.rows(dtype), case.entries(1, 0.1, 1.0).reshape(-1)
    retrieved, inter = (torch.empty(sc, hd, device="cuda") for _ in range(2))
    delta_v = torch.empty(sc if delta_v_rows == "step" else n, hd, device="cuda")
    readings = torch.full((n, hd), float("nan"), device="cuda", dtype=dtype)
    residual = torch.full((n, hd), float("nan"), device="cuda")
    launch = p2().make_dual_matmul(x, qx, v, beta, retrieved, inter, delta_v, readings, residual,
                                   num_parallel=case.P, valid_tokens=case.valid_tokens)
    bounds = {"delta_v": FP32_GATE if dtype == torch.float32 else TF32_GATE,
              "readings": FP32_GATE if dtype == torch.float32 else READINGS_BF16_GATE,
              "residual": FP32_GATE}
    zero = torch.zeros((), dtype=torch.float64, device="cuda")
    for ti in range(case.steps):
        ok = case.step(case.valid.reshape(-1), ti)[:, None]
        retrieved.copy_(case.randn(sc, hd, dtype=torch.float32).masked_fill(~ok, float("nan")))
        inter.copy_(case.randn(sc, hd, dtype=torch.float32).masked_fill(~ok, float("nan")))
        res = torch.where(ok, case.step(v, ti).double() - retrieved.double(), zero)
        u = (torch.where(ok, case.step(beta, ti).double()[:, None], zero) * res).view(case.P, C, hd)
        chunks = slice(ti * case.P, (ti + 1) * case.P)
        refs = {"delta_v": torch.where(ok, torch.bmm(x[chunks].double(), u).reshape(sc, hd), zero),
                "readings": torch.where(ok, torch.bmm(qx[chunks].double(), u).reshape(sc, hd)
                                        + inter.double(), zero),
                "residual": res}
        launch(ti)
        got = {"delta_v": delta_v if delta_v_rows == "step" else case.step(delta_v, ti),
               "readings": case.step(readings, ti), "residual": case.step(residual, ti)}
        for name, ref in refs.items():
            err = rel_max(ref, got[name])
            assert err <= bounds[name], f"{name} at step {ti}: rel_max {err:.3g} > {bounds[name]:g}"
            assert_zero(got[name][~ok[:, 0]], f"{name} at step {ti}")


@pytest.mark.parametrize("hd", [64, 1024])
def test_gds_group_sum_rounds_once(hd):
    """Three reads of one slot add 256, 1 and 0 to a bf16 gradient row holding -256: the
    exact result 1 needs the group's sum to join the row before its single rounding
    (bf16(257) = 256 would leave 0). hd 1024 splits the row over the group's last entries."""
    case = Case(P=1, C=16, steps=1, W=1, R=1, hd=hd, dtype=torch.bfloat16, sph=40, pad=0)
    case.k_slot = (20 + torch.arange(16, device="cuda")).view(1, 16, 1).contiguous()
    q = torch.arange(16, device="cuda") + 2
    q[:3] = 0
    case.q_slot = q.view(1, 16, 1).contiguous()
    case.resort()
    grad = torch.zeros(case.slots, hd, device="cuda", dtype=torch.bfloat16)
    grad[0] = -256.0
    grad_readings = torch.zeros(16, hd, device="cuda")
    grad_readings[0], grad_readings[1] = 256.0, 1.0
    bank = torch.zeros_like(grad)
    q_val = torch.ones(1, 16, 1, device="cuda", dtype=torch.bfloat16)
    log_q = torch.zeros(1, 16, 1, device="cuda")
    dots = torch.empty(1, 16, 1, device="cuda")
    mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots, chunk_size=16,
                        num_parallel=1, valid_tokens=16, **case.sorted_args("q"))(0)
    assert bits_equal(grad[0], torch.ones(hd, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("hd", [64, 1024])
def test_gds_single_entry_rounds_once(hd):
    """Two steps each read slot 0 once (a single-entry slot in its chunk): the backward's
    first launch (step 1) adds -256 to a zero bf16 gradient row, the second (step 0) adds
    0.99609375 * 258 = 256.9921875. The exact row, 0.9921875, is a bf16 value; rounding the
    term to bf16 (256) before adding it, as a bf16 atomic add does, leaves 0.
    hd 1024 covers two column tiles."""
    case = Case(P=1, C=16, steps=2, W=1, R=1, hd=hd, dtype=torch.bfloat16, sph=40, pad=15)
    t = torch.arange(16, device="cuda")
    case.k_slot = (20 + t).view(1, 16, 1).expand(2, 16, 1).contiguous()
    case.q_slot = torch.where(t == 0, 0, 2 + t).view(1, 16, 1).expand(2, 16, 1).contiguous()
    case.resort()
    grad = torch.zeros(case.slots, hd, device="cuda", dtype=torch.bfloat16)
    grad[1] = 7.0  # a row no read touches
    bank = torch.zeros_like(grad)
    bank[0] = 0.5  # makes the dot outputs of the reads of slot 0 nonzero
    grad_readings = torch.zeros(32, hd, device="cuda")
    grad_readings[0], grad_readings[16] = 258.0, -256.0
    q_val = torch.zeros(2, 16, 1, device="cuda", dtype=torch.bfloat16)
    q_val[0, 0], q_val[1, 0] = 0.99609375, 1.0
    log_q = torch.zeros(2, 16, 1, device="cuda")
    dots = torch.full((2, 16, 1), float("nan"), device="cuda")
    built = _da2_gds_built()
    launch = mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots,
                                 chunk_size=16, num_parallel=1,
                                 valid_tokens=case.valid_tokens, **case.sorted_args("q"))
    deferred = _dots_after_step0(built, hd)   # CUDA GDS, hd 1024: step 1's dots are summed at step 0
    launch(1)
    first = torch.zeros_like(grad)
    first[0], first[1] = -256.0, 7.0
    assert torch.equal(grad, first), grad[:2, :4]
    assert deferred or dots[1, 0, 0].item() == -128.0 * hd, dots[:, 0, 0]
    launch(0)
    assert dots[0, 0, 0].item() == 129.0 * hd and dots[1, 0, 0].item() == -128.0 * hd, dots[:, 0, 0]
    expect = first.clone()
    expect[0] = 0.9921875
    assert torch.equal(grad, expect), grad[0, :4]


def test_gds_single_entry_fp32_matches_atomic():
    """With an fp32 gradient a single-entry slot's term is added with one f32 atomic add (the
    element's only add), as in the previous release, so the row is that release's: the term is
    rounded to f32 before the add, never fused with it. (1 + 2**-23) * (1 - 2**-23) rounds to 1, so -1 + term is 0;
    a fused multiply-add would give -2**-46."""
    case = Case(P=1, C=16, steps=1, W=1, R=1, hd=64, sph=40, pad=0)
    t = torch.arange(16, device="cuda")
    case.k_slot = (20 + t).view(1, 16, 1).contiguous()
    case.q_slot = torch.where(t == 0, 0, 2 + t).view(1, 16, 1).contiguous()
    case.resort()
    grad = torch.zeros(case.slots, 64, device="cuda")
    grad[0] = -1.0
    grad_readings = torch.zeros(16, 64, device="cuda")
    grad_readings[0] = 1.0 - 2.0 ** -23
    q_val = torch.zeros(1, 16, 1, device="cuda")
    q_val[0, 0] = 1.0 + 2.0 ** -23
    log_q = torch.zeros(1, 16, 1, device="cuda")
    dots = torch.empty(1, 16, 1, device="cuda")
    mv().make_mover_gds(torch.zeros_like(grad), grad, grad_readings, case.q_slot, q_val, log_q,
                        dots, chunk_size=16, num_parallel=1, valid_tokens=16,
                        **case.sorted_args("q"))(0)
    assert torch.equal(grad, torch.zeros_like(grad)), grad[0, :4]


@pytest.mark.parametrize("chunk", [16, 32])
def test_gds_rounding_through_the_op(chunk):
    """The same two reads through ``sdm_write_read`` against the float64 oracle (bf16 bank,
    beta = g = 0, distinct write slots): with chunk 16 the reads are single-entry slots of two
    steps, with chunk 32 a group of one step. The exact gradient is a bf16 value, so the
    deterministic op must reproduce it exactly."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    from sdm_testlib import oracle_write_read

    P, T, D = 1, 17, 64
    k_idx = (1 + torch.arange(T, device="cuda")).view(P, T, 1)
    q_idx = (18 + torch.arange(T, device="cuda")).view(P, T, 1)
    q_idx[:, 0] = q_idx[:, -1] = 0
    k_val = torch.ones(P, T, 1, device="cuda", dtype=torch.bfloat16)
    q_val = torch.zeros_like(k_val)
    q_val[:, 0], q_val[:, -1] = 0.99609375, 1.0
    v = torch.zeros(P, T, D, device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros(P, T, 1, device="cuda")
    g = torch.zeros_like(beta)
    cotangent = torch.zeros(T, D, device="cuda", dtype=torch.bfloat16)
    cotangent[0], cotangent[-1] = 258.0, -256.0
    memory = torch.zeros(40, D, device="cuda", dtype=torch.bfloat16)
    memory[0] = 0.5  # the reads' gradient of q_val (from the GDS kernel's dots) is nonzero
    d = {"memory": memory, "k_idx": k_idx, "q_idx": q_idx, "k_val": k_val, "q_val": q_val,
         "v": v, "beta": beta, "g": g}
    _, ref, _ = oracle_write_read(d, cotangent.view(P, T, D), sph=40)
    assert ref["grad_memory"][0, 0].item() == 0.9921875
    bank = memory.clone().requires_grad_(True)
    q_leaf = q_val.clone().requires_grad_(True)
    readings = sdm_write_read(bank, k_idx, k_val, v, beta, g, q_idx, q_leaf, chunk_size=chunk)
    grad, grad_q = torch.autograd.grad(readings, (bank, q_leaf), cotangent)
    # The backward ran its GDS launches: the dots the q_val gradient comes from (exact
    # values, 129 * 64 and -128 * 64) match the float64 oracle.
    assert torch.equal(grad_q, ref["grad_q_val"].to(torch.bfloat16)), grad_q.view(-1)[:4]
    assert torch.equal(grad, ref["grad_memory"].to(torch.bfloat16)), grad[0, :4]


@pytest.mark.parametrize("members", [127, 128])
def test_clipped_group_bounds(members):
    """One group of 127 / 128 K (and Q) entries in an fp32 row of 8192 columns: the size field
    reads 127 (clipped), and the row splits over up to 127 workers, so the group's first
    member is a worker too (127 members) and its bounds come from the walk."""
    hd, C = 8192, 128
    case = Case(P=1, C=C, steps=1, W=1, R=1, hd=hd, sph=3 * C, pad=0)
    t = torch.arange(C, device="cuda")
    case.k_slot = torch.where(t < members, 0, 2 + t).view(1, C, 1).contiguous()
    case.q_slot = torch.where(t < members, 1, 2 + C + t).view(1, C, 1).contiguous()
    case.resort()
    cpe, k_val, rex, delta_v = dw_inputs(case)
    bank = case.randn(case.slots, hd)
    snapshot = bank[case.k_slot.view(-1)].contiguous()
    ref = ref_dw(case, bank, snapshot, delta_v, case.k_slot.view(-1),
                 case.entry_valid(1).reshape(-1), *(t.reshape(-1) for t in (
                     cpe, case.is_last, case.is_only, k_val, rex)))
    mv().make_mover_dw(bank, snapshot, delta_v, case.k_slot, cpe, k_val, rex,
                       chunk_size=C, num_parallel=1, valid_tokens=C,
                       **case.sorted_args("k"))(0)
    assert_close(ref, bank, torch.float32, "bank")
    q_val, log_q = case.entries(1), case.entries(1, -1.5, 0.0)
    grad_readings = case.rows()
    grad = case.randn(case.slots, hd)
    dots = torch.empty(1, C, 1, device="cuda")
    ref_grad, _ = ref_gds(case, grad, bank, grad_readings, case.q_slot.view(-1),
                          case.entry_valid(1).reshape(-1), q_val.reshape(-1), log_q.reshape(-1))
    mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots, chunk_size=C,
                        num_parallel=1, valid_tokens=C, **case.sorted_args("q"))(0)
    assert_close(ref_grad, grad, torch.float32, "grad")


# ------------------------------------------------------------------------------ determinism
def repeat_bitwise(launch, ti, state, outs, what):
    """Runs step ti REPEATS times from the same ``state`` (restored before each launch) and
    asserts that ``state`` and ``outs`` come out bitwise identical every time."""
    init = [t.clone() for t in state]
    first = None
    for k in range(REPEATS):
        for t, i in zip(state, init):
            t.copy_(i)
        launch(ti)
        got = [t.clone() for t in state + outs]
        if first is None:
            first = got
        else:
            for n, (a, b) in enumerate(zip(first, got)):
                assert bits_equal(a, b), f"{what}: output {n} differs at repeat {k}, step {ti}"


@pytest.mark.parametrize("pad", [0, 5], ids=["full", "masked"])
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd, W, C", DET_DW_SHAPES)
def test_dw_bitwise_repeat(hd, W, C, dtype, dist, pad):
    mode = "none"   # 2.0.0: no per-entry (layout 0) quantizer; the quantized compact rows: tests/test_quant_rows.py
    case = Case(hd=hd, W=W, C=C, dtype=dtype, sph=W + 4, dist=dist, pad=pad, steps=2)
    cpe, k_val, rex, delta_v = dw_inputs(case)
    bank = case.randn(case.slots, hd)
    snapshot = torch.empty(case.SC * W, hd, device="cuda", dtype=dtype)
    payload = scale = None
    outs = []
    if mode != "none":
        width = hd if mode == "fp8" else (hd + 1) // 2
        payload = torch.zeros(case.N * case.C * W, width, device="cuda",
                              dtype=torch.float8_e4m3fn if mode == "fp8" else torch.uint8)
        scale = torch.zeros(case.N * case.C * W, device="cuda")
        outs = [payload, scale]
    launch = mv().make_mover_dw(bank, snapshot, delta_v, case.k_slot, cpe, k_val, rex,
                                payload, scale, chunk_size=C,
                                num_parallel=case.P, valid_tokens=case.valid_tokens,
                                snapshot_quant=mode, **case.sorted_args("k"))
    for ti in range(case.steps):
        slot, ok = case.step(case.k_slot, ti, W), case.step(case.entry_valid(W), ti, W)
        snap = bank[slot].masked_fill(~ok[:, None], float("nan"))
        snapshot.copy_(snap)
        ref = ref_dw(case, bank, snap, case.step(delta_v, ti), slot, ok,
                     *(case.step(t, ti, W) for t in (cpe, case.is_last, case.is_only, k_val, rex)))
        repeat_bitwise(launch, ti, [bank], outs, "DW")
        assert_close(ref, bank, dtype, f"bank after step {ti}")
        bank.copy_(ref)


@pytest.mark.parametrize("pad", [0, 5], ids=["full", "masked"])
@pytest.mark.parametrize("dist", ["distinct", "hot"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd, R, C", DET_SHAPES)
def test_gds_bitwise_repeat(hd, R, C, dtype, dist, pad):
    # sph leaves room for the W = 4 write lanes too ("hot" draws them from slots 3 and up)
    case = Case(hd=hd, R=R, C=C, dtype=dtype, sph=max(R, 4) + 4, dist=dist, pad=pad, steps=2)
    q_val, log_q = case.entries(R, dtype=dtype), case.entries(R, -1.5, 0.0)
    grad_readings = case.rows()
    bank, grad = case.randn(case.slots, hd), case.randn(case.slots, hd)
    dots = torch.full((case.N, case.C, R), float("nan"), device="cuda")
    launch = mv().make_mover_gds(bank, grad, grad_readings, case.q_slot, q_val, log_q, dots,
                                 chunk_size=C, num_parallel=case.P, valid_tokens=case.valid_tokens,
                                 **case.sorted_args("q"))
    for ti in reversed(range(case.steps)):
        slot, ok = case.step(case.q_slot, ti, R), case.step(case.entry_valid(R), ti, R)
        ref_grad, _ = ref_gds(case, grad, bank, case.step(grad_readings, ti), slot, ok,
                              case.step(q_val, ti, R), case.step(log_q, ti, R))
        repeat_bitwise(launch, ti, [grad], [dots], "GDS")
        assert_close(ref_grad, grad, dtype, f"grad after step {ti}")
        grad.copy_(ref_grad)


def test_sorted_rows_match_phase1():
    """The torch sorted rows of the tests equal Phase 1's (payload bits, positions, flags)."""
    from phase1_reference import sorted_rank_reference
    from sparse_delta_memory.cuda.phase1_cuda import phase1_forward
    for dist in DISTS:
        case = Case(C=17, W=5, R=3, dist=dist, sph=12)
        g = torch.zeros(case.N, case.C, 1, device="cuda")
        kv = torch.ones(case.N, case.C, case.W, device="cuda")
        qv = torch.ones(case.N, case.C, case.R, device="cuda")
        out = phase1_forward(case.k_slot, case.q_slot, kv, qv, g,
                             torch.zeros(1, device="cuda", dtype=torch.int64),
                             num_parallel=case.P, stream_offset=12, num_keys=12,
                             valid_last=case.C - 5)
        assert bits_equal(out.payload, case.spay), dist
        for name in ("pos_k", "pos_q", "q_last", "q_only", "is_last", "is_only"):
            assert torch.equal(getattr(out, name), getattr(case, name)), (dist, name)
        for name, want in (("spay_k", case.spay_k), ("spay_q", case.spay_q),
                           ("spay_rank", sorted_rank_reference(case.spay, case.C, case.W, case.R))):
            got = getattr(out, name)
            assert got is not None, (dist, name)
            assert bits_equal(got.reshape(want.shape), want), (dist, name)
