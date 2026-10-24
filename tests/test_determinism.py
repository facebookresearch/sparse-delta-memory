# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Bitwise determinism of the SDM kernels.

Every bank row has one writer per kernel launch and every sum runs in a fixed order (Phase 1
sorts each chunk's entries by slot; one owner per slot group sums the group in sorted order).
So for fixed inputs, configuration and GPU type, every output, final bank and gradient is
reproducible byte for byte: over repeated calls, over CUDA-graph replays (which equal eager)
and in fresh processes. Each test runs one computation REPEATS times on identical inputs and
compares every tensor bytewise with the first run.

The inputs are collision-heavy, because accumulating in scheduling order (atomics) only
changes results where several entries of a chunk meet on one slot: ``make_op_inputs`` slots
"hot" (a few slots in every token's write and read set, groups of 2 C entries), "trained"
(the repeated-group statistics of trained models' selections), "zipf" (benchmarks/
bench_sdm.py's popularity law), "clustered" (32 slots carry 96% of the mass) and "repeat"
(also repeated slots inside a token). Layers use 256 slots per head with 32 writes and reads
per token. Every case asserts its realized fraction of repeated write entries
(``check_collisions``), so it cannot silently become collision-free. "distinct" inputs, where
no slot occurs twice in a stream, are a control: there is nothing to accumulate, so they are
deterministic with or without atomics, as is the fused decode step (plain stores). The fp32
one-chunk cases isolate Phase 1: their readings are computed before any bank write lands.

Covered: Phase 1 alone (every output, the A and QK triangles included, on 512 chunks); the
op's training path (readings, final bank, restored bank, the six gradients) on fp32 and
bf16 banks, snapshot_quant none / fp8 / int4, chunk sizes 16, 37, 48, 64, 128 and 256,
head_dim 64 to 1024 (one and two column tiles per row in the movers), ragged T (not a
multiple of C, and shorter than C), P streams with their own slot ranges, the shared-bank
mode (num_memory_slots = whole bank, shared_bank=True, disjoint slots per stream) and the
2-D layout, per-token and per-write gates, int32 and int64 ids, repeated slots; the op
without backward state with slabs and stream groups forced small; the layer's training step
(output, final bank, input and parameter gradients) eager and with ``cuda_graph=True``
(equal to eager), under selective activation checkpointing (equal to the plain step), the
cuda_graph option's eager fallback (equal to both; its forward runs the no-grad path, here
split into groups of streams and slabs of steps; with fp8 snapshots a documented exception,
strict xfail), the no-grad prefill (with the eval memory-access statistics) and
token-by-token decode; two fresh processes; a small 3:1 lingua model (sliding-window
attention and SDM layers) whose attention runs on SDPA's math backend, deterministic
matmuls and softmax, with an all-attention control that must itself be deterministic.

The forward that saves backward state and the no-grad forward agree bitwise on the same
inputs (the cuda_graph fallback test splits the no-grad call; the op-level pin keeps it
whole), with one documented exception: snapshot_quant="fp8" below head_dim 512 (a strict
xfail on a float32 bank; on bfloat16 a non-strict one, equality depending on the inputs).

Not covered here: numerics (test_f64_oracle.py, test_autograd_reference.py, ...), and
agreement between chunk sizes (they differ by rounding).

Tiers: the fast tier (``-m "not slow"``, the repo's convention) exercises every axis at
least once; ``-m slow`` adds a grid over dtype x snapshot x chunk size, the widest token
width, a flagship-sized layer (slots drawn from a trained-like law), TF32 matmuls, decode
after a prefill and the graphed model. ``main(["--digests", FILE])`` writes the SHA-256 of
every tensor of the fresh-process cases (the workers of
``test_fresh_processes_reproduce_every_tensor``, and tests/det_mutation_harness.py).
"""
import argparse
import functools
import json
import os
import signal
import subprocess
import sys
import time
import zlib
from typing import NamedTuple

if __name__ == "__main__":  # worker mode: make ``lingua`` and ``sdm_testlib`` importable
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.dirname(_HERE), _HERE]

import pytest  # noqa: E402
import torch  # noqa: E402

from sdm_testlib import (  # noqa: E402
    OP_GRADS,
    OP_INPUTS,
    SelectionSpy,
    assert_quant_tail_refused,
    assert_runs_bitwise,
    bits_equal,
    bits_mismatch,
    fixed_slot_scores,
    layer_inputs,
    make_cotangent,
    make_layer,
    make_op_inputs,
    set_nograd_budgets as _set_budgets,
    rel_max,
    rel_rms,
    requires_cuda,
    slot_groups,
    streams_own_their_slots,
    tensor_digest,
    quant_tail_refusal,
)

REPEATS = 5
SLOW_REPEATS = 8
bf16, fp32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
_NAME = {bf16: "bf16", fp32: "fp32"}


@pytest.fixture(autouse=True)
def _ieee_and_fresh_graphs(request):
    """fp32 banks run IEEE matmuls (unless a test asks for TF32), and every test starts
    and ends without captured graphs."""
    if not torch.cuda.is_available():
        yield
        return
    from sparse_delta_memory.cuda_graph import reset_cuda_graphs

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    reset_cuda_graphs()
    try:
        yield
    finally:
        reset_cuda_graphs()
        torch.backends.cuda.matmul.allow_tf32 = prev


def _seed(text):
    return zlib.crc32(text.encode()) % 2**31


# ================================================================================ op level
class OpCase(NamedTuple):
    id: str
    dtype: torch.dtype
    quant: str        # snapshot_quant (training path only)
    C: int
    T: int
    P: int
    mode: str         # "own": [P, T] streams with their own slot ranges; "shared": the same
    #                   streams over one bank (num_memory_slots = whole bank); "2d": [P * T]
    #                   rows, one stream over the P slot ranges
    slots: str        # make_op_inputs slot distribution
    W: int
    R: int
    hd: int
    sph: int          # slots per stream
    kwd: bool         # one gate per write (key_weighted_decay)
    idx: torch.dtype  # k_idx / q_idx dtype
    budgets: str = "default"  # no-grad path: "default", "tiny" (one step per slab, one
    #                           stream per group) or "two" (two of each)
    gates: str = "moderate"   # make_op_inputs gates


def op_inputs(c):
    """Float inputs in the bank dtype and int ids (int64; ``run_op`` casts them)."""
    d = make_op_inputs(c.P, c.T, c.W, c.R, c.hd, c.sph, c.dtype, seed=_seed(c.id),
                       slots=c.slots, gates=c.gates)
    if c.kwd:
        gen = torch.Generator().manual_seed(_seed(c.id + "/g"))
        d["g"] = torch.empty(c.P, c.T, c.W).uniform_(-0.5, -0.01, generator=gen).to(
            device=d["g"].device, dtype=c.dtype)
    return d


def run_op(d, c, path="train"):
    """One ``sdm_write_read`` call on private copies of ``d``: {name: tensor}.

    path "train": forward + backward; readings, the final bank (after the forward), the bank
    the backward restored and the six gradients. path "nograd": readings and final bank.
    """
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T = c.P, c.T
    kw = dict(chunk_size=c.C, key_weighted_decay=c.kwd)
    if c.mode == "own":
        kw.update(num_memory_slots=c.sph, num_parallel=P)
    elif c.mode == "shared":
        # One bank for all streams, each on its own slot range: what shared_bank asserts.
        assert streams_own_their_slots(d["k_idx"], c.sph), c.id
        assert streams_own_their_slots(d["q_idx"], c.sph), c.id
        kw.update(num_memory_slots=P * c.sph, num_parallel=P, shared_bank=True)
    elif c.mode == "2d":
        kw.update(num_memory_slots=0, num_parallel=1)
    else:
        raise ValueError(c.mode)

    def lay(t):
        return t.reshape(P * T, -1) if c.mode == "2d" else t

    train = path == "train"
    leaves = {n: d[n].detach().clone().requires_grad_(train) for n in OP_INPUTS}
    args = (leaves["memory"], lay(d["k_idx"].to(c.idx)), lay(leaves["k_val"]), lay(leaves["v"]),
            lay(leaves["beta"]), lay(leaves["g"]), lay(d["q_idx"].to(c.idx)), lay(leaves["q_val"]))
    if not train:
        with torch.no_grad():
            readings = sdm_write_read(*args, **kw)
        return {"readings": readings.clone(), "final_bank": leaves["memory"].detach().clone()}
    readings = sdm_write_read(*args, snapshot_quant=c.quant, **kw)
    res = {"readings": readings.detach().clone(), "final_bank": leaves["memory"].detach().clone()}
    grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS],
                                grad_outputs=make_cotangent(readings, "dense"))
    res["restored_bank"] = leaves["memory"].detach().clone()
    res.update(zip(OP_GRADS, grads))
    return res


# Realized fraction of write entries in repeated groups (see sdm_testlib.slot_groups) that a
# collision-heavy case must keep, so that it cannot silently become collision-free. The
# "trained" law must also stay in the band of trained models' selections.
MIN_REPEATED = 0.2
TRAINED_REPEATED = (0.15, 0.9)


def check_collisions(stats, slots, label):
    """The premise of a case: "distinct" inputs have no group of two entries, the others
    repeat at least MIN_REPEATED of their write entries and have read groups too."""
    if slots == "distinct":
        assert stats["longest"] == (1, 1, 1), f"premise: {label} has colliding slots {stats}"
        return
    lo, hi = TRAINED_REPEATED if slots == "trained" else (MIN_REPEATED, 1.0)
    assert lo <= stats["repeated"] <= hi and stats["longest"][1] >= 2, \
        f"premise: {label} repeats {stats['repeated']:.2f} of its writes, not {lo}..{hi}: {stats}"


def _fmt_stats(s):
    return (f"{s['repeated']:.0%} of writes repeated, mean group {s['mean_group']:.1f}, "
            f"longest w/r/all {s['longest']}")


def check_op(c, repeats, path="train"):
    d = op_inputs(c)
    stats = slot_groups(d["k_idx"], d["q_idx"], c.C, c.mode == "2d")
    check_collisions(stats, c.slots, c.id)
    cls = quant_tail_refusal(c.dtype, c.hd, c.quant) if path == "train" else None
    if cls is not None:      # fp8 / int4 rows of a quantized tail head: the training call is refused, every time
        for _ in range(repeats):
            assert_quant_tail_refused(lambda: run_op(d, c, path), cls)
        return
    runs = [run_op(d, c, path) for _ in range(repeats)]
    assert_runs_bitwise(runs, label=f"{c.id} {path} ({_fmt_stats(stats)})")


# Each row exercises several axes; together they cover every value at least once (checked
# by test_the_default_tier_covers_every_axis). The "distinct" rows are the controls.
# The one-chunk fp32 rows (T <= C) see Phase 1 alone in their readings: the bank is read
# before any write lands, so A and QK are the only accumulations upstream of them.
OP_FAST = [
    # id, dtype, quant, C, T, P, mode, slots, W, R, hd, sph, kwd, idx[, budgets]
    OpCase("hot_bf16_C64", bf16, "none", 64, 200, 2, "own", "hot", 16, 16, 64, 512, False, i64),
    OpCase("trained_fp32_C64_i32_flat", fp32, "none", 64, 200, 2, "own",
           "trained", 16, 16, 64, 4096, False, i32, gates="flat"),
    OpCase("zipf_bf16_C37_fp8", bf16, "fp8", 37, 100, 2, "own",
           "zipf", 16, 16, 64, 4096, False, i64),
    OpCase("clustered_fp32_C16_int4_kwd_hd129", fp32, "int4", 16, 53, 3, "own",
           "clustered", 16, 16, 129, 512, True, i32),
    OpCase("hot_bf16_C128_T90_shared", bf16, "none", 128, 90, 2, "shared",
           "hot", 16, 16, 64, 512, False, i64),
    OpCase("trained_bf16_C256_kwd_i32", bf16, "none", 256, 300, 1, "own",
           "trained", 16, 16, 64, 4096, True, i32),
    OpCase("hot_bf16_C48_int4_2d", bf16, "int4", 48, 130, 2, "2d",
           "hot", 16, 16, 64, 512, False, i64),
    OpCase("zipf_fp32_C128_fp8_P4", fp32, "fp8", 128, 257, 4, "own",
           "zipf", 8, 8, 64, 4096, False, i64),
    OpCase("repeat_bf16_C64", bf16, "none", 64, 150, 2, "own",
           "repeat", 16, 16, 64, 512, False, i64),
    OpCase("hot_fp32_C256_T200_one_chunk_flat", fp32, "none", 256, 200, 2, "own",
           "hot", 16, 16, 64, 512, False, i64, gates="flat"),
    OpCase("clustered_fp32_C64_T64_one_chunk_flat", fp32, "none", 64, 64, 2, "own",
           "clustered", 16, 16, 64, 512, False, i32, gates="flat"),
    # head_dim 1024: two column tiles per bank row in the movers
    OpCase("trained_bf16_C64_hd1024", bf16, "none", 64, 150, 1, "own",
           "trained", 16, 16, 1024, 4096, False, i64),
    # one write and one read per token, and unequal widths: other specializations
    OpCase("zipf_bf16_C64_W1R1", bf16, "none", 64, 200, 2, "own",
           "zipf", 1, 1, 64, 128, False, i64),
    OpCase("trained_fp32_C64_W3R48_i32", fp32, "none", 64, 150, 2, "own",
           "trained", 3, 48, 64, 768, False, i32),
    OpCase("zipf_bf16_C37_W48R3_fp8", bf16, "fp8", 37, 150, 2, "own",
           "zipf", 48, 3, 64, 12288, False, i64),
    OpCase("distinct_bf16_C64", bf16, "none", 64, 150, 2, "own",
           "distinct", 16, 16, 64, 8192, False, i64),
    OpCase("distinct_fp32_C37_fp8_i32", fp32, "fp8", 37, 100, 2, "own",
           "distinct", 16, 16, 64, 4096, False, i32),
]

NOGRAD_FAST = [
    OpCase("hot_bf16_C32_tiny_budgets", bf16, "none", 32, 300, 3, "own",
           "hot", 16, 16, 64, 512, False, i64, "tiny"),
    OpCase("zipf_fp32_C64_kwd_two_budgets", fp32, "none", 64, 500, 3, "own",
           "zipf", 16, 16, 64, 4096, True, i32, "two"),
    OpCase("trained_bf16_C37_2d", bf16, "none", 37, 200, 2, "2d",
           "trained", 16, 16, 64, 4096, False, i64),
    OpCase("distinct_bf16_C64_tiny_budgets", bf16, "none", 64, 150, 2, "own",
           "distinct", 16, 16, 64, 8192, False, i64, "tiny"),
]


def _slow_grid():
    """dtype x snapshot x chunk size; the other axes rotate through their values."""
    modes = [("own", 2), ("shared", 2), ("2d", 2), ("own", 1), ("own", 4)]
    out, i = [], 0
    for dtype in (bf16, fp32):
        for quant in ("none", "fp8", "int4"):
            for C in (16, 37, 64, 128, 256):
                mode, P = modes[(i + 2 * (i // 5)) % len(modes)]
                slots = ("hot", "zipf", "clustered", "repeat", "trained")[(i + i // 5) % 5]
                T = 2 * C + 7 if i % 2 == 0 else C - 3
                width = 32 if C <= 64 else 16
                hd = (128, 96, 129, 1024)[i % 4]
                sph = 256 * width if slots in ("zipf", "trained") else 512
                out.append(OpCase(
                    f"{slots}_{_NAME[dtype]}_{quant}_C{C}_T{T}_{mode}{P}_hd{hd}",
                    dtype, quant, C, T, P, mode, slots, width, width, hd, sph, i % 2 == 1,
                    i32 if i % 4 < 2 else i64))
                i += 1
    out.append(OpCase("hot_bf16_W256_R256_hd1024_C256", bf16, "none", 256, 300, 1, "own", "hot",
                      256, 256, 1024, 4096, False, i64))
    return out


OP_SLOW = _slow_grid()


@requires_cuda
@pytest.mark.parametrize("case", OP_FAST, ids=[c.id for c in OP_FAST])
def test_op_training_call_is_bitwise_reproducible(case):
    check_op(case, REPEATS)


@requires_cuda
@pytest.mark.parametrize("case", NOGRAD_FAST, ids=[c.id for c in NOGRAD_FAST])
def test_op_no_grad_call_is_bitwise_reproducible(case, monkeypatch):
    """The forward without backward state, with its slabs of steps and groups of streams
    forced small (several slabs and groups run)."""
    from sparse_delta_memory import memory_ops

    want = _set_budgets(monkeypatch, memory_ops, case)
    check_op(case, REPEATS, path="nograd")
    rec = memory_ops.last_dispatch()
    assert rec["save"] is False
    if want is not None:
        steps = -(-case.T // case.C)
        assert (rec["slab_steps"], rec["group_streams"]) == want and steps > want[0], \
            f"premise: slabs/groups {rec['slab_steps']}/{rec['group_streams']} over {steps} steps"


# The forward that saves backward state and the no-grad forward (same inputs, chunk size and
# split) agree bitwise, except with snapshot_quant="fp8" below head_dim 512 (bfloat16 at 120
# excepted) on phase2_kernels' Triton DW mover: it is compiled per snapshot mode, and its fp8 variant
# spreads a slot group's window over more lanes than the unquantized variant (head_dim 64:
# 8 lanes against 2 on a float32 bank, 8 against 4 on bfloat16), so the window sums
# associate differently and a float32 bank differs in the last bits of some readings and
# final-bank elements. On a
# bfloat16 bank the final rounding hides it here, but only for these inputs, so that xfail
# is not strict. From head_dim 512 both variants put a window's rows on the same lanes: the
# two forwards agree by construction, asserted at head_dim 1024. Each forward is bitwise
# reproducible on its own, and the package README documents the exception. The float32
# xfail is strict: once the two agree, the test fails, so that the README is updated with
# the fix. The CUDA DW mover is such a fix: where it takes the call (movers.dw_admits) the
# strict xfails are conditioned off (_FP8_DOCUMENTED) and the two must agree.
class SavingNoGradMismatch(AssertionError):
    """The saving and the no-grad forward differ, each reproducible, finite and close: the
    only failure the documented fp8 exception may expect."""


def _da2_dw_admits(dtype, hd, chunk):
    """Whether the CUDA DW mover takes a call with this bank dtype, head_dim and chunk size
    on this GPU (movers.dw_admits: mover_da2_cuda.supported / fits, e.g. sm_90+). Its fp8
    variant sums a slot group's window as the unquantized one does, so such calls' saving and
    no-grad forwards (and the cuda_graph option's eager fallback below) agree with fp8
    snapshots too; every call it declines keeps the documented exception of the Triton DW."""
    if not torch.cuda.is_available():
        return False
    from sparse_delta_memory import movers
    bank = torch.empty(64, hd, dtype=dtype, device="cuda")
    return bool(movers.dw_admits(bank, chunk))


def _da2_dw_calls():
    """How many CUDA DW launchers the op has built (movers.STATS)."""
    from sparse_delta_memory import movers
    return int(movers.STATS.get("da2_dw", 0))


# Conditions of the fp8 xfails (pytest.mark.xfail's first argument), for their configurations:
# the save / no-grad pin at head_dim 64 and chunk 128, and the fallback test's heavy layer
# (head_dim 128, chunk 64).
_FP8_DOCUMENTED = not _da2_dw_admits(torch.float32, 64, 128)
_FP8_INPUTS_DOCUMENTED = not _da2_dw_admits(torch.bfloat16, 64, 128)
_FP8_FALLBACK_DOCUMENTED = not _da2_dw_admits(torch.float32, 128, 64)
_FP8 = pytest.mark.xfail(_FP8_DOCUMENTED, strict=True, raises=SavingNoGradMismatch,
                         reason="the forward DW mover's fp8 variant rounds the bank update "
                                "differently (README)")
_FP8_INPUTS = pytest.mark.xfail(_FP8_INPUTS_DOCUMENTED, strict=False, raises=SavingNoGradMismatch,
                                reason="the fp8 DW mover sums a window in another tree at this "
                                       "head_dim; equality depends on the inputs (README)")
SAVE_VS_NOGRAD = (
    [pytest.param(dtype, quant, 64, id=f"{_NAME[dtype]}_{quant}",
                  marks=(_FP8 if dtype is fp32 else _FP8_INPUTS) if quant == "fp8" else ())
     for dtype in (fp32, bf16) for quant in ("none", "int4", "fp8")]
    + [pytest.param(dtype, "fp8", 1024, id=f"{_NAME[dtype]}_fp8_hd1024") for dtype in (fp32, bf16)])


def _forward(d, c, path):
    """Readings and final bank of one forward: the one that saves backward state ("train",
    snapshot_quant ``c.quant``; no backward runs) or the no-grad one."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    leaves = {n: d[n].detach().clone().requires_grad_(path == "train") for n in OP_INPUTS}
    args = (leaves["memory"], d["k_idx"].to(c.idx), leaves["k_val"], leaves["v"],
            leaves["beta"], leaves["g"], d["q_idx"].to(c.idx), leaves["q_val"])
    kw = dict(chunk_size=c.C, key_weighted_decay=c.kwd, num_memory_slots=c.sph,
              num_parallel=c.P)
    if path == "train":
        readings = sdm_write_read(*args, snapshot_quant=c.quant, **kw)
    else:
        with torch.no_grad():
            readings = sdm_write_read(*args, **kw)
    return {"readings": readings.detach().clone(),
            "final_bank": leaves["memory"].detach().clone()}


@requires_cuda
@pytest.mark.parametrize("dtype,quant,hd", SAVE_VS_NOGRAD)
def test_op_saving_forward_is_bitwise_the_no_grad_forward(dtype, quant, hd):
    """Readings and final bank of the forward that saves backward state (snapshot_quant
    ``quant``) and of the no-grad forward, on the same collision-heavy inputs (both run the
    call whole: 4 streams, 3 steps). Each forward must be reproducible, finite and close to
    the other; only then may they differ (SavingNoGradMismatch), and only the documented
    exception expects that."""
    name = f"save_vs_nograd_{_NAME[dtype]}_{quant}" + ("" if hd == 64 else f"_hd{hd}")
    c = OpCase(name, dtype, quant, 128, 257, 4, "own", "zipf", 8, 8, hd, 4096, False, i64)
    d = op_inputs(c)
    check_collisions(slot_groups(d["k_idx"], d["q_idx"], c.C), c.slots, c.id)
    dw0 = _da2_dw_calls()
    saving = [_forward(d, c, "train") for _ in range(2)]
    no_grad = [_forward(d, c, "nograd") for _ in range(2)]
    if quant == "fp8" and hd == 64 and not (_FP8_DOCUMENTED if dtype is fp32
                                             else _FP8_INPUTS_DOCUMENTED):
        assert _da2_dw_calls() >= dw0 + 4, "premise: DA2's DW mover took the four fp8 forwards"
    assert_runs_bitwise(saving, label=f"saving forward {c.id}")
    assert_runs_bitwise(no_grad, label=f"no-grad forward {c.id}")
    want = {"readings": (c.P * c.T * c.hd, dtype), "final_bank": (d["memory"].numel(), dtype)}
    # The documented difference is rounding: relative RMS ~6e-9 on float32 banks, single
    # elements a few units in the last place apart. Bounds far above that, far below damage:
    # relative RMS, and the largest element difference relative to the largest element.
    close_rms, close_max = (1e-6, 1e-5) if dtype is fp32 else (1e-2, 5e-2)
    for k, ref in saving[0].items():
        got = no_grad[0][k]
        assert (ref.shape == got.shape and ref.dtype == got.dtype == want[k][1]
                and ref.numel() == want[k][0]), \
            f"{c.id} {k}: {tuple(ref.shape)} {ref.dtype} vs {tuple(got.shape)} {got.dtype}"
        rms, worst = rel_rms(ref.float(), got.float()), rel_max(ref.float(), got.float())
        assert bool(torch.isfinite(got).all()) and rms < close_rms and worst < close_max, \
            f"{c.id} {k}: the two forwards are far apart (rel_rms {rms:.2e}, rel_max {worst:.2e})"
    diffs = {k: f"{bits_mismatch(v, no_grad[0][k])}, rel_max "
                f"{rel_max(v.float(), no_grad[0][k].float()):.2e}"
             for k, v in saving[0].items() if not bits_equal(v, no_grad[0][k])}
    if diffs:
        raise SavingNoGradMismatch(f"saving vs no-grad forward {c.id}: {diffs}")


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("case", OP_SLOW, ids=[c.id for c in OP_SLOW])
def test_op_grid_is_bitwise_reproducible(case):
    check_op(case, SLOW_REPEATS)
    if case.quant == "none":
        check_op(case, SLOW_REPEATS, path="nograd")


@requires_cuda
@pytest.mark.slow
def test_op_with_tf32_matmuls_is_bitwise_reproducible():
    torch.backends.cuda.matmul.allow_tf32 = True  # restored by the autouse fixture
    check_op(OpCase("hot_fp32_C64_tf32", fp32, "none", 64, 200, 2, "own", "hot", 16, 16, 64,
                    512, False, i64), SLOW_REPEATS)


def test_the_default_tier_covers_every_axis(request):
    """Every axis value of the determinism contract occurs in the default tier, and some
    collision-heavy chunk holds a slot group longer than one warp (32 entries); the tier's
    only expected failures are the documented ones, each expecting only its own exception
    (strict unless equality depends on the inputs), as collected (markers from any level
    included). CPU only."""
    train, nograd = OP_FAST, NOGRAD_FAST
    assert {c.dtype for c in train} == {c.dtype for c in nograd} == {bf16, fp32}
    assert {c.quant for c in train} == {"none", "fp8", "int4"}
    assert {16, 37, 64, 128, 256} <= {c.C for c in train}
    for cases in (train, nograd):
        assert any(c.T > c.C and c.T % c.C for c in cases), "ragged T"
        assert {"own", "2d"} <= {c.mode for c in cases}
        assert any(c.mode == "own" and c.P > 1 for c in cases)
        assert {c.kwd for c in cases} == {c.idx == i32 for c in cases} == {False, True}
        assert "distinct" in {c.slots for c in cases}, "control"
    assert any(c.T < c.C for c in train), "T shorter than one chunk"
    assert any(c.hd >= 512 for c in train), "two column tiles per bank row in the movers"
    assert "shared" in {c.mode for c in train}
    assert {"hot", "zipf", "clustered", "repeat", "trained"} <= {c.slots for c in train}
    assert any(c.T <= c.C and c.dtype == fp32 and c.slots != "distinct" for c in train), \
        "an fp32 one-chunk case (readings downstream of Phase 1 only)"
    assert {c.budgets for c in nograd} >= {"tiny", "two"}
    xfails = [(p.id, m.kwargs) for p in SAVE_VS_NOGRAD for m in p.marks if m.name == "xfail"]
    saving = dict(strict=True, raises=SavingNoGradMismatch, reason=_FP8.kwargs["reason"])
    inputs = dict(strict=False, raises=SavingNoGradMismatch, reason=_FP8_INPUTS.kwargs["reason"])
    assert xfails == [("fp32_fp8", saving), ("bf16_fp8", inputs)], xfails
    assert [(p.id, p.values) for p in SAVE_VS_NOGRAD if p.values[2] >= 512] == [
        ("fp32_fp8_hd1024", (fp32, "fp8", 1024)), ("bf16_fp8_hd1024", (bf16, "fp8", 1024))] \
        and not [p for p in SAVE_VS_NOGRAD if p.values[2] >= 512 and p.marks], \
        "the structural fp8 pins: unmarked, at head_dim 1024"
    pin = test_op_saving_forward_is_bitwise_the_no_grad_forward
    fallback = test_cuda_graph_eager_fallback_with_fp8_snapshots_is_the_documented_exception
    assert not [m for m in getattr(pin, "pytestmark", []) if m.name in ("slow", "xfail", "skip")]
    assert not [m for m in fallback.pytestmark if m.name in ("slow", "skip")]
    expected = {f"{pin.__name__}[fp32_fp8]": saving, f"{pin.__name__}[bf16_fp8]": inputs}
    expected[fallback.__name__] = dict(strict=True, raises=FallbackMismatch,
                                       reason=_FP8_FALLBACK.kwargs["reason"])
    # This module's items, with their markers as collected (from every level); other modules'
    # expected failures are their own (``pytest tests/`` collects them too).
    here = os.path.realpath(__file__)
    for item in request.session.items:
        if os.path.realpath(str(getattr(item, "path", None) or item.fspath)) != here:
            continue
        marks = [(m.args, m.kwargs) for m in item.iter_markers("xfail")]
        if item.name in expected:
            # run, each with its condition (whether the CUDA DW declines that configuration)
            cond = {f"{pin.__name__}[fp32_fp8]": _FP8_DOCUMENTED,
                    f"{pin.__name__}[bf16_fp8]": _FP8_INPUTS_DOCUMENTED,
                    fallback.__name__: _FP8_FALLBACK_DOCUMENTED}[item.name]
            assert marks == [((cond,), expected[item.name])], (item.nodeid, marks)
        else:
            assert not marks, f"{item.nodeid}: unexpected xfail {marks}"
    longest = 0
    for c in train + nograd + OP_SLOW:  # every premise holds (the slow tier's included)
        d = make_op_inputs(c.P, c.T, c.W, c.R, 1, c.sph, torch.float32, seed=_seed(c.id),
                           slots=c.slots, device="cpu")
        stats = slot_groups(d["k_idx"], d["q_idx"], c.C, c.mode == "2d")
        check_collisions(stats, c.slots, c.id)
        if c in train:
            longest = max(longest, stats["longest"][2])
    assert longest > 32, longest


# ================================================================================ Phase 1
@requires_cuda
@pytest.mark.parametrize("dtype", [fp32, bf16], ids=["fp32", "bf16"])
def test_phase1_outputs_are_bitwise_reproducible(dtype):
    """Phase 1 alone on 512 collision-heavy chunks (many blocks for the scheduler to
    reorder): every output, the A and QK triangles included, bitwise over repeats. Token
    pairs share about seven slots, the weights span 24 binades and the gates barely decay,
    so a cell's terms round differently when summed in another order."""
    stats, run = _phase1_case(dtype)
    runs = [run() for _ in range(REPEATS)]
    assert_runs_bitwise(runs, label=f"phase 1 {_NAME[dtype]} ({_fmt_stats(stats)})")


def _phase1_case(dtype):
    """(slot statistics, a function running Phase 1 once) of the Phase-1 repeat test."""
    from sparse_delta_memory.cuda.phase1_cuda import Phase1, phase1_forward

    N, C, W, R, sph = 512, 64, 16, 16, 512
    d = make_op_inputs(1, N * C, W, R, 1, sph, dtype, seed=_seed(f"phase1/{dtype}"),
                       slots="clustered", gates="flat")
    gen = torch.Generator(device="cuda").manual_seed(_seed(f"phase1/{dtype}/mag"))

    def spread(t):
        mag = torch.exp2(torch.randint(-12, 13, t.shape, device="cuda", generator=gen).float())
        return (t.float() * mag).to(dtype)

    k_idx, q_idx = d["k_idx"].view(N, C, W), d["q_idx"].view(N, C, R)
    k_val, q_val = spread(d["k_val"]).view(N, C, W), spread(d["q_val"]).view(N, C, R)
    g = d["g"].float().view(N, C, 1)
    stats = slot_groups(d["k_idx"], d["q_idx"], C)
    check_collisions(stats, "clustered", "phase1")

    def run():
        out = phase1_forward(k_idx, q_idx, k_val, q_val, g,
                             torch.zeros(1, device="cuda", dtype=torch.int64), num_parallel=1,
                             stream_offset=0, num_keys=sph, valid_last=C)
        return {f: t.clone() for f, t in zip(Phase1._fields, out)}

    return stats, run


# ============================================================================= layer level
# 256 slots per head (16 x 16 product keys), 32 writes and reads per token: heavy collisions.
HEAVY = dict(dim=256, num_heads=2, slots_per_head=256, num_writes=32, num_reads=32,
             chunk_size=64, output_gate=True, normalize_readings=True)
LAYER_TRAIN = [
    ("heavy_bf16_T200", bf16, HEAVY, 2, 200),
    ("heavy_fp32_C37_kwd_int4_T150", fp32,
     dict(HEAVY, chunk_size=37, key_weighted_decay=True, snapshot_quant="int4"), 2, 150),
]
LAYER_SLOW = [
    ("heavy_bf16_C128_fp8_T100", bf16, dict(HEAVY, chunk_size=128, snapshot_quant="fp8"),
     2, 100),
    ("heavy_bf16_C256_nobank_T600", bf16,
     dict(HEAVY, chunk_size=256, backprop_on_memory=False), 2, 600),
    ("flagship_bf16_T1024", bf16, dict(dim=2048, num_heads=2, slots_per_head=65536,
                                       num_writes=64, num_reads=64, chunk_size=64,
                                       output_gate=True, normalize_readings=True), 1, 1024),
]


def layer_step(layer, x, gy):
    """Output, final bank, input gradient and every parameter gradient of one training
    step (the bank is cloned before the backward restores it)."""
    for p in layer.parameters():
        p.grad = None
    xi = x.detach().clone().requires_grad_(True)
    out, state = layer(xi)
    res = {"out": out.detach().clone(), "final_bank": state.memory.detach().clone()}
    out.backward(gy)
    res["grad_x"] = xi.grad.detach().clone()
    res.update({"grad." + n: p.grad.detach().clone() for n, p in layer.named_parameters()})
    return res


def prefill_step(layer, x):
    layer.eval()
    with torch.no_grad():
        out, state = layer(x)
    return {"out": out.clone(), "final_bank": state.memory.clone()}


def decode_steps(layer, x, prefill=0):
    """Token-by-token decode from a fresh cache, after a prefill of ``prefill`` tokens."""
    layer.eval()
    with torch.no_grad():
        cache = layer.create_kv_cache(bsz=x.shape[0], seq_len=0, dtype=x.dtype, device="cuda")
        outs = []
        if prefill:
            outs.append(layer(x[:, :prefill], cache=cache)[0])
        for t in range(prefill, x.shape[1]):
            outs.append(layer(x[:, t:t + 1], cache=cache)[0])
    return {"out": torch.cat(outs, 1), "final_bank": cache.memory.clone()}


def _layer_case(label, dtype, cfg, B, T):
    layer = make_layer(seed=_seed(label) % 1000, dtype=dtype, **cfg)
    x, gy = layer_inputs(B, T, cfg["dim"], dtype=dtype, seed=_seed(label + "/x") % 1000)
    return layer, x, gy


class SlotLawSelections:
    """Replaces the product-key top-k of ``layers`` with slots drawn from the popularity law
    1 / (1 + rank)**exponent, sampled once per (writes or reads, shape): every run, and every
    layer sharing this object, selects the same slots. The values are the slots' scores
    of the projections (with product_key_topk's deterministic backward), so the gradients
    still reach them. Used where a layer's
    own selections at initialization collide too rarely (65536 slots per head)."""

    def __init__(self, *layers, exponent, seed=0):
        self.exponent, self.seed, self.idx, self.calls = exponent, seed, {}, {}
        for layer in layers:
            layer._product_keys = functools.partial(self.select, layer)

    def select(self, layer, x, k):
        n = self.calls[id(layer)] = self.calls.get(id(layer), -1) + 1
        side = layer.key_size_per_head // 2
        key = (n % 2, tuple(x.shape), k)
        if key not in self.idx:
            gen = torch.Generator(device=x.device).manual_seed(self.seed + n % 2)
            log_w = -self.exponent * torch.log1p(
                torch.randperm(side * side, device=x.device, generator=gen).float())
            rows = []
            for r0 in range(0, x.shape[0] * x.shape[1], 256):
                u = torch.rand(min(256, x.shape[0] * x.shape[1] - r0), side * side,
                               device=x.device, generator=gen).clamp_min_(1e-20)
                rows.append((log_w - torch.log(-torch.log(u))).topk(k, dim=-1).indices)
            self.idx[key] = torch.cat(rows).sort(dim=-1).values.view(*x.shape[:2], k)
        idx = self.idx[key]
        return fixed_slot_scores(x, idx, (side, side)), idx


@requires_cuda
@pytest.mark.parametrize("dtype", [fp32, bf16], ids=["fp32", "bf16"])
def test_fixed_slot_scores_are_product_key_topk_with_fixed_slots(dtype):
    """The helper behind SlotLawSelections (and the frozen selections of
    test_cuda_graph_option.py) gives product_key_topk's values and gradient for the same
    slots, bitwise, and a gather reference's gradient up to rounding."""
    from sparse_delta_memory.product_key_topk import product_key_topk

    side, k = 64, 32
    gen = torch.Generator(device="cuda").manual_seed(3)
    raw = torch.randn(6, 40, 2 * side, device="cuda", generator=gen).to(dtype)
    gv = torch.randn(6, 40, k, device="cuda", generator=gen).to(dtype)
    ref_raw, fix_raw = raw.clone().requires_grad_(), raw.clone().requires_grad_()
    val, lin = product_key_topk(ref_raw, (side, side), k)
    val.backward(gv)
    got = fixed_slot_scores(fix_raw, lin, (side, side))
    got.backward(gv)
    assert bits_equal(val.detach(), got.detach()) and bits_equal(ref_raw.grad, fix_raw.grad)
    dense = torch.zeros(6, 40, 2 * side, device="cuda", dtype=torch.float64)
    dense.scatter_add_(-1, lin // side, gv.double()).scatter_add_(-1, side + lin % side,
                                                                   gv.double())
    assert torch.allclose(fix_raw.grad.double(), dense, rtol=1e-2, atol=1e-2)


# Layer cases whose selections follow SlotLawSelections (exponent): at 64 writes over 65536
# slots, 0.9 repeats about 43% of a chunk's writes at C = 64, as trained models' selections do.
LAWS = {"flagship_bf16_T1024": 0.9}


def _selection_groups(layer, x, C):
    """``slot_groups`` of the layer's selections on ``x``."""
    prev = layer.__dict__.get("_product_keys")
    spy = SelectionSpy(layer)
    with torch.no_grad():
        layer(x)
    (_, k_idx), (_, q_idx) = spy.take()
    if prev is None:
        del layer._product_keys  # back to the class's method
    else:
        layer._product_keys = prev
    return slot_groups(k_idx, q_idx, C)


def check_layer_train(label, dtype, cfg, B, T, repeats):
    layer, x, gy = _layer_case(label, dtype, cfg, B, T)
    if label in LAWS:
        SlotLawSelections(layer, exponent=LAWS[label], seed=_seed(label))
    stats = _selection_groups(layer, x, cfg["chunk_size"])
    check_collisions(stats, "layer", label)
    runs = [layer_step(layer.train(), x, gy) for _ in range(repeats)]
    assert_runs_bitwise(runs, label=f"layer {label} ({_fmt_stats(stats)})")


@requires_cuda
@pytest.mark.parametrize("label,dtype,cfg,B,T", LAYER_TRAIN, ids=[c[0] for c in LAYER_TRAIN])
def test_layer_training_step_is_bitwise_reproducible(label, dtype, cfg, B, T):
    check_layer_train(label, dtype, cfg, B, T, REPEATS)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("label,dtype,cfg,B,T", LAYER_SLOW, ids=[c[0] for c in LAYER_SLOW])
def test_layer_training_step_is_bitwise_reproducible_slow(label, dtype, cfg, B, T):
    check_layer_train(label, dtype, cfg, B, T, 3)


def check_graph_equals_eager(label, dtype, cfg, B, T, repeats):
    """cuda_graph=True: every step (the capturing one included) equals the eager twin's
    step bitwise, and the eager steps equal each other."""
    eager, x, gy = _layer_case(label, dtype, cfg, B, T)
    graph = make_layer(seed=0, dtype=dtype, cuda_graph=True, **cfg)
    graph.load_state_dict(eager.state_dict())
    if label in LAWS:
        SlotLawSelections(eager, graph, exponent=LAWS[label], seed=_seed(label))
    check_collisions(_selection_groups(eager, x, cfg["chunk_size"]), "layer", label)
    runs_e = [layer_step(eager, x, gy) for _ in range(repeats)]
    runs_g = [layer_step(graph, x, gy) for _ in range(repeats)]
    stats = dict(graph.graph_runner.stats)
    assert stats.get("forward_capture") == 1 and stats.get("backward_replay", 0) >= repeats - 1, \
        f"premise: the graphs did not replay: {stats}"
    assert_runs_bitwise(runs_e, label=f"eager {label}")
    assert_runs_bitwise(runs_g, label=f"graph vs eager {label}", ref=runs_e[0])


@requires_cuda
@pytest.mark.parametrize("dtype", [bf16, fp32], ids=["bf16", "fp32"])
def test_layer_cuda_graph_steps_are_bitwise_eager(dtype):
    check_graph_equals_eager(f"graph_heavy_{_NAME[dtype]}", dtype,
                             dict(HEAVY, chunk_size=48), 2, 130, REPEATS)


@requires_cuda
@pytest.mark.slow
def test_flagship_cuda_graph_steps_are_bitwise_eager():
    label, dtype, cfg, B, T = LAYER_SLOW[-1]
    check_graph_equals_eager(label, dtype, cfg, B, T, 3)


# The cuda_graph option's eager fallback (cuda_graph._EagerSaved: a differentiable call under
# suspend_cuda_graphs(), inside an outer CUDA-graph capture, or one the captured graphs do not
# serve) runs its forward on the op's no-grad path and its backward on a recompute. The
# no-grad path runs big calls in groups of streams and slabs of steps (memory_ops' STEP_BYTES,
# SLAB_BYTES, SLAB_CHUNKS_PER_SM), and these cases make it split: 66 streams of fp32 rows of
# 1024 at the default budgets (groups of 32, 32 and 2 streams), and budgets forced to one
# stream and one step at a time (8 x 8 calls). A group's token gather must still reduce in
# the order of the whole call: each case asserts that a gather geometry taken from a stream
# group's own token count would differ from the whole call's on this GPU (the row-group
# count, which fixes the rounding, depends on the SM count).
FALLBACK = dict(dim=2048, num_heads=2, slots_per_head=4096, num_writes=64, num_reads=64,
                chunk_size=64)
FALLBACK_CASES = [
    # label, dtype, B, T, no-grad budgets
    ("fallback_fp32_B33_T2", fp32, 33, 2, "default"),
    ("fallback_bf16_B4_T512_one_stream_one_step", bf16, 4, 512, "tiny"),
]


@requires_cuda
@pytest.mark.parametrize("label,dtype,B,T,budgets", FALLBACK_CASES,
                         ids=[c[0] for c in FALLBACK_CASES])
def test_cuda_graph_eager_fallback_is_bitwise_eager_and_graph(label, dtype, B, T, budgets,
                                                               monkeypatch):
    """One training step four ways: the eager layer, the cuda_graph=True layer replaying its
    graphs, and that layer's eager fallback under suspend_cuda_graphs() and for a call of
    another shape than the captured one. Output, final bank and every gradient are bitwise
    equal."""
    from sparse_delta_memory import cuda_graph, memory_ops
    from sparse_delta_memory.cuda.gather_cuda import gather_geometry

    if budgets == "tiny":
        monkeypatch.setattr(memory_ops, "SLAB_CHUNKS_PER_SM", 0)
        monkeypatch.setattr(memory_ops, "SLAB_BYTES", 1)
        monkeypatch.setattr(memory_ops, "STEP_BYTES", 1)
    eager, x, gy = _layer_case(label, dtype, FALLBACK, B, T)
    graph = make_layer(seed=0, dtype=dtype, cuda_graph=True, **FALLBACK)
    other = make_layer(seed=0, dtype=dtype, cuda_graph=True, **FALLBACK)
    graph.load_state_dict(eager.state_dict())
    other.load_state_dict(eager.state_dict())
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    C = FALLBACK["chunk_size"]

    def row_groups(streams):
        return gather_geometry(FALLBACK["dim"] // FALLBACK["num_heads"], dtype,
                               max(FALLBACK["num_writes"], FALLBACK["num_reads"]),
                               streams * C, num_sms=sms).groups

    def split():
        """The no-grad path's stream groups for this call, and whether a gather geometry
        taken from a group's own token count would differ from the whole call's."""
        with torch.no_grad():
            eager(x)
        rec = memory_ops.last_dispatch()
        assert rec["save"] is False, rec
        P, group = rec["streams"], rec["group_streams"]
        own = {row_groups(min(group, P - p0)) for p0 in range(0, P, group)}
        return rec, own != {row_groups(P)}

    rec, sensitive = split()
    if not sensitive and budgets == "default":
        # The row-group count depends on the SM count: on this GPU the default groups would
        # gather in the whole call's order, so one stream per group it is.
        monkeypatch.setattr(memory_ops, "STEP_BYTES", 1)
        rec, sensitive = split()
    if not sensitive:
        pytest.skip(f"no stream group of {rec} gathers in another order on {sms} SMs")

    # Each eager fallback enters through CudaGraphRunner.eager (to _EagerSaved).
    fallbacks = []
    entry = cuda_graph.CudaGraphRunner.eager
    monkeypatch.setattr(cuda_graph.CudaGraphRunner, "eager",
                        staticmethod(lambda *a: fallbacks.append(1) or entry(*a)))

    def replays(layer):
        return {k: v for k, v in layer.graph_runner.stats.items() if not k.startswith("eager_")}

    runs_e = [layer_step(eager, x, gy) for _ in range(2)]
    runs_g = [layer_step(graph, x, gy) for _ in range(3)]  # captures, then replays
    st_g = dict(graph.graph_runner.stats)
    assert (st_g.get("forward_capture") == 1 and st_g.get("forward_replay", 0) >= 2
            and st_g.get("backward_replay", 0) >= 1 and not fallbacks
            and not any(k.startswith("eager_") for k in st_g)), f"premise: graph {st_g}"
    with cuda_graph.suspend_cuda_graphs():
        runs_s = [layer_step(graph, x, gy) for _ in range(2)]
    assert len(fallbacks) == 2 and replays(graph) == st_g, \
        f"premise: the suspended steps did not run eager ({len(fallbacks)} fallbacks, " \
        f"{dict(graph.graph_runner.stats)} vs {st_g})"
    x2, gy2 = layer_inputs(B, T + 1, FALLBACK["dim"], dtype=dtype, seed=7)
    layer_step(other, x2, gy2)  # captures another shape
    runs_o = [layer_step(other, x, gy) for _ in range(2)]
    st_o = dict(other.graph_runner.stats)
    assert st_o.get("eager_shape") == 2 and len(fallbacks) == 4, \
        f"premise: the other-shape calls did not run eager ({st_o})"
    assert_runs_bitwise(runs_e, label=f"eager {label}")
    assert_runs_bitwise(runs_g, label=f"graph vs eager {label}", ref=runs_e[0])
    assert_runs_bitwise(runs_s, label=f"suspended fallback vs eager {label}", ref=runs_e[0])
    assert_runs_bitwise(runs_o, label=f"other-shape fallback vs eager {label}", ref=runs_e[0])


class FallbackMismatch(AssertionError):
    """The eager fallback differs from eager, each reproducible, finite and close: the only
    failure the documented fp8 exception of the fallback may expect."""


# The cuda_graph option's eager fallback runs its forward on the no-grad path, and with
# snapshot_quant="fp8" that path's DW mover is not the fp8 variant the saving forward runs
# (test_op_saving_forward_is_bitwise_the_no_grad_forward): the fallback's outputs then differ
# from eager in the last bits, and so do the gradients computed downstream of them. The
# graphed call runs the saving forward and stays bitwise eager. The package README documents
# the exception; the xfail is strict, so that it fails once the two agree, and the README is
# updated with the fix.
_FP8_FALLBACK = pytest.mark.xfail(_FP8_FALLBACK_DOCUMENTED, strict=True, raises=FallbackMismatch,
                                  reason="the eager fallback's no-grad forward does not run "
                                         "the fp8 variant of the DW mover (README)")


@requires_cuda
@_FP8_FALLBACK
def test_cuda_graph_eager_fallback_with_fp8_snapshots_is_the_documented_exception(monkeypatch):
    """fp8 snapshots on a float32 bank (the heavy layer): eager steps are bitwise
    reproducible and the graphed steps bitwise eager; the eager fallback (under
    suspend_cuda_graphs()) is reproducible, finite and close to eager, and only then may it
    differ from eager (FallbackMismatch)."""
    from sparse_delta_memory import cuda_graph

    label, cfg = "fallback_fp8_heavy_fp32", dict(HEAVY, snapshot_quant="fp8")
    eager, x, gy = _layer_case(label, fp32, cfg, 2, 200)
    graph = make_layer(seed=0, dtype=fp32, cuda_graph=True, **cfg)
    graph.load_state_dict(eager.state_dict())
    check_collisions(_selection_groups(eager, x, cfg["chunk_size"]), "layer", label)
    fallbacks = []
    entry = cuda_graph.CudaGraphRunner.eager
    monkeypatch.setattr(cuda_graph.CudaGraphRunner, "eager",
                        staticmethod(lambda *a: fallbacks.append(1) or entry(*a)))

    def replays():
        return {k: v for k, v in graph.graph_runner.stats.items() if not k.startswith("eager_")}

    dw0 = _da2_dw_calls()
    runs_e = [layer_step(eager, x, gy) for _ in range(2)]
    assert _FP8_FALLBACK_DOCUMENTED or _da2_dw_calls() > dw0, "premise: DA2's DW mover ran"
    runs_g = [layer_step(graph, x, gy) for _ in range(3)]  # captures, then replays
    st = replays()
    assert (st.get("forward_capture") == 1 and st.get("forward_replay", 0) >= 2
            and st.get("backward_replay", 0) >= 1 and not fallbacks), f"premise: graph {st}"
    with cuda_graph.suspend_cuda_graphs():
        runs_s = [layer_step(graph, x, gy) for _ in range(2)]
    assert len(fallbacks) == 2 and replays() == st, \
        f"premise: the suspended steps did not run eager ({len(fallbacks)}, {replays()})"
    assert_runs_bitwise(runs_e, label=f"eager {label}")
    assert_runs_bitwise(runs_g, label=f"graph vs eager {label}", ref=runs_e[0])
    assert_runs_bitwise(runs_s, label=f"fallback {label}")
    for k, ref in runs_e[0].items():
        got = runs_s[0][k]
        assert ref.shape == got.shape and ref.dtype == got.dtype == fp32, \
            f"{label} {k}: {tuple(ref.shape)} {ref.dtype} vs {tuple(got.shape)} {got.dtype}"
        rms, worst = rel_rms(ref, got), rel_max(ref, got)
        assert bool(torch.isfinite(got).all()) and rms < 1e-4 and worst < 1e-3, \
            f"{label} {k}: fallback and eager far apart (rel_rms {rms:.2e}, rel_max {worst:.2e})"
    diffs = {k: bits_mismatch(v, runs_s[0][k]) for k, v in runs_e[0].items()
             if not bits_equal(v, runs_s[0][k])}
    if diffs:
        raise FallbackMismatch(f"fallback vs eager {label}: {diffs}")


class _SelectiveCheckpoint(torch.nn.Module):
    """lingua's selective activation checkpointing policy around one layer."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

        pytest.importorskip("lingua.distributed", reason="needs the repository checkout (apps/, lingua/)")
        from lingua.distributed import get_default_policy

        ctx = functools.partial(create_selective_checkpoint_contexts, get_default_policy())
        return checkpoint(lambda u: self.inner(u)[0], x, use_reentrant=False, context_fn=ctx)


@requires_cuda
def test_layer_under_selective_checkpointing_is_bitwise_the_plain_step():
    """The checkpointed step recomputes the same op on the same inputs: its output and
    gradients equal the plain step's bitwise, over repeats."""
    layer, x, gy = _layer_case("sac_heavy_bf16", bf16, HEAVY, 2, 200)
    plain = []
    for _ in range(REPEATS):
        r = layer_step(layer, x, gy)
        del r["final_bank"]
        plain.append(r)
    wrapped = _SelectiveCheckpoint(layer)
    ck = []
    for _ in range(REPEATS):
        for p in layer.parameters():
            p.grad = None
        xi = x.detach().clone().requires_grad_(True)
        out = wrapped(xi)
        out.backward(gy)
        r = {"out": out.detach().clone(), "grad_x": xi.grad.detach().clone()}
        r.update({"grad." + n: p.grad.detach().clone() for n, p in layer.named_parameters()})
        ck.append(r)
    assert_runs_bitwise(plain, label="plain step")
    assert_runs_bitwise(ck, label="selective checkpointing vs plain", ref=plain[0])


@requires_cuda
@pytest.mark.parametrize("dtype", [bf16, fp32], ids=["bf16", "fp32"])
def test_layer_no_grad_prefill_is_bitwise_reproducible(dtype):
    layer, x, _ = _layer_case(f"prefill_heavy_{_NAME[dtype]}", dtype, HEAVY, 2, 500)
    runs = [prefill_step(layer, x) for _ in range(REPEATS)]
    assert_runs_bitwise(runs, label=f"prefill {_NAME[dtype]}")


@requires_cuda
@pytest.mark.parametrize("dtype", [bf16, fp32], ids=["bf16", "fp32"])
def test_eval_memory_access_statistics_are_bitwise_reproducible(dtype):
    """log_memory_access_stats in eval mode: the slot access counts and weights (accumulated
    per slot over many repeated slots; in fp32 a float sum of weights of varied magnitudes
    depends on its order), the logged statistics and the prefill output."""
    layer, x, _ = _layer_case(f"stats_heavy_{dtype}", dtype,
                              dict(HEAVY, log_memory_access_stats=True), 2, 300)
    names = ("_read_idx_counts", "_write_idx_counts", "_read_slot_access_weights",
             "_write_slot_access_weights")
    runs = []
    for _ in range(REPEATS):
        layer.get_memory_access_stats(reset=True)
        res = prefill_step(layer, x)
        res.update({n: getattr(layer, n).clone() for n in names})
        stats = layer.get_memory_access_stats(reset=False)
        res["logged"] = torch.tensor([stats[k] for k in sorted(stats)], dtype=torch.float64)
        runs.append(res)
    assert int(runs[0]["_read_idx_counts"].sum()) > 0, "premise: no statistics were recorded"
    assert_runs_bitwise(runs, label="eval statistics")


@requires_cuda
def test_layer_decode_is_bitwise_reproducible():
    """Control: the fused decode step uses plain stores and fixed-order sums."""
    layer, x, _ = _layer_case("decode_heavy_bf16", bf16, HEAVY, 2, 48)
    runs = [decode_steps(layer, x) for _ in range(3)]
    assert_runs_bitwise(runs, label="decode")


@requires_cuda
@pytest.mark.slow
def test_layer_decode_after_a_prefill_is_bitwise_reproducible():
    layer, x, _ = _layer_case("decode_after_prefill_heavy_bf16", bf16, HEAVY, 2, 400)
    runs = [decode_steps(layer, x, prefill=336) for _ in range(3)]
    assert_runs_bitwise(runs, label="decode after prefill")


# ========================================================================== fresh processes
FRESH_OPS = ("hot_bf16_C64", "trained_fp32_C64_i32_flat")
# The int4 witness is a supported head: clustered_fp32_C16_int4_kwd_hd129 (float32 int4 hd 129, quantized tail class
# T3) is refused on training calls.
WIDE_OPS = ("zipf_bf16_C37_fp8", "hot_bf16_C48_int4_2d", "trained_bf16_C256_kwd_i32",
            "trained_bf16_C64_hd1024", "zipf_bf16_C64_W1R1", "zipf_bf16_C37_W48R3_fp8")
FRESH_LAYER = LAYER_TRAIN[0]


def _kill_tree(pid):
    """SIGKILL a process and its descendants (children first; Linux /proc)."""
    kids = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
                        kids.append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    for k in kids:
        _kill_tree(k)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def fresh_process_digests(wide=False):
    """{case: {tensor: sha256}} of the fresh-process cases: a bf16 and an fp32 op case,
    Phase 1 alone and a layer; ``wide`` adds op cases with fp8 and int4 snapshots, per-write
    gates, chunks 16, 37 and 256 and head_dim 1024 (the cases the mutation harness hashes to
    see whether a mutation changed any value)."""
    ids = FRESH_OPS + (WIDE_OPS if wide else ())
    out = {f"op {c.id}": run_op(op_inputs(c), c) for c in OP_FAST       # (a refused training call computes nothing)
           if c.id in ids and quant_tail_refusal(c.dtype, c.hd, c.quant) is None}
    out["phase1 fp32"] = _phase1_case(fp32)[1]()
    layer, x, gy = _layer_case(*FRESH_LAYER)
    out[f"layer {FRESH_LAYER[0]}"] = layer_step(layer.train(), x, gy)
    return {case: {k: tensor_digest(t) for k, t in res.items()} for case, res in out.items()}


@requires_cuda
def test_fresh_processes_reproduce_every_tensor(tmp_path):
    """The same op, Phase-1 and layer cases in two fresh interpreters (python -B, each with
    its own Triton cache, run concurrently) and in this process give identical SHA-256 per
    tensor."""
    here = fresh_process_digests()
    # The workers import this module under its own name with this process's sys.path, so
    # they find the code under test (and the helpers) exactly as pytest did. They stay in
    # pytest's process group (a harness that kills pytest's group kills them too) and are
    # killed here on any failure.
    path = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    code = f"import {__name__} as m; m.main()"
    procs, rcs = [], []
    deadline = time.monotonic() + 1200
    try:
        for i in range(2):
            env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton{i}"),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=path)
            with open(tmp_path / f"worker{i}.log", "w") as log:
                procs.append(subprocess.Popen(
                    [sys.executable, "-B", "-c", code, "--digests",
                     str(tmp_path / f"digests{i}.json")],
                    env=env, stdout=log, stderr=subprocess.STDOUT))
        for proc in procs:
            rcs.append(proc.wait(timeout=max(1.0, deadline - time.monotonic())))
    finally:
        for proc in procs:
            if proc.poll() is None:
                _kill_tree(proc.pid)
                proc.wait()
    results = []
    for i, rc in enumerate(rcs):
        tail = (tmp_path / f"worker{i}.log").read_text()[-3000:]
        assert rc == 0, f"worker {i} failed (rc={rc}):\n{tail}"
        results.append(json.loads((tmp_path / f"digests{i}.json").read_text()))
    bad = {}
    for case, want in here.items():
        for i, res in enumerate(results):
            diff = sorted(k for k in want if res.get(case, {}).get(k) != want[k])
            if diff:
                bad[f"{case} [process {i}]"] = diff
    assert not bad, f"tensors that differ from this process: {bad}"


# ============================================================================ model level
LM_T, LM_VOCAB = 200, 512
LM_SDM = dict(num_heads=2, slots_per_head=256, num_writes=32, num_reads=32,
              chunk_size=64, output_gate=True, normalize_readings=True)


def build_lm(sdm_at, dtype, cuda_graph=False):
    """A 4-layer apps.main LMTransformer: SDM at ``sdm_at``, sliding-window attention
    (window 64) elsewhere."""
    pytest.importorskip("apps.main.transformer", reason="needs the repository checkout (apps/, lingua/)")
    from apps.main.transformer import LMTransformer, LMTransformerArgs
    from sparse_delta_memory import SparseDeltaMemoryArgs

    torch.manual_seed(0)
    args = LMTransformerArgs(
        dim=256, n_layers=4, n_heads=4, vocab_size=LM_VOCAB, max_seqlen=LM_T,
        swa_at=[i for i in range(4) if i not in sdm_at], sdm_at=list(sdm_at), swa_window=64,
        sdm_args=SparseDeltaMemoryArgs(cuda_graph=cuda_graph, **LM_SDM))
    model = LMTransformer(args)
    model.init_weights()
    return model.to("cuda", dtype)


def lm_step(model, tokens, targets):
    """Loss, every parameter gradient and each SDM layer's output of one training step,
    with attention on SDPA's math backend."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from sparse_delta_memory import SparseDeltaMemory

    outs, hooks = {}, []
    for n, m in model.named_modules():
        if isinstance(m, SparseDeltaMemory):
            hooks.append(m.register_forward_hook(
                lambda mod, args, res, n=n: outs.__setitem__("out." + n, res[0].detach().clone())))
    model.zero_grad(set_to_none=True)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            loss = model(tokens, targets, attn_impl="sdpa")
            loss.backward()
    finally:
        for h in hooks:
            h.remove()
    res = {"loss": loss.detach().clone(), **outs}
    res.update({"grad." + n: p.grad.detach().clone() for n, p in model.named_parameters()})
    return res


def _lm_batch():
    """Tokens and targets. Every token id occurs once in the batch: the embedding backward
    then accumulates nothing (a scatter of repeated ids may add in any order; compiled, it
    does), and SDM's collisions come from its own slot selection."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    tokens = torch.randperm(LM_VOCAB, device="cuda", generator=gen)[:2 * LM_T].view(2, LM_T)
    return tokens, torch.randint(0, LM_VOCAB, (2, LM_T), device="cuda", generator=gen)


@requires_cuda
def test_lingua_model_training_step_is_bitwise_reproducible():
    """A 3:1 stack (sliding-window attention x 3, then SDM): loss, the SDM layer's output
    and every gradient over repeats. Control: the same stack with attention in place of SDM
    is deterministic (else the model level cannot judge SDM)."""
    pytest.importorskip("apps.main.transformer")
    batch = _lm_batch()
    control = build_lm([], bf16)
    runs = [lm_step(control, *batch) for _ in range(REPEATS)]
    assert_runs_bitwise(runs, label="premise: the attention-only control")
    model = build_lm([3], bf16)
    runs = [lm_step(model, *batch) for _ in range(REPEATS)]
    assert any(k.startswith("out.") for k in runs[0])
    assert_runs_bitwise(runs, label="3:1 model")


@requires_cuda
@pytest.mark.slow
def test_graphed_lingua_model_is_bitwise_eager():
    pytest.importorskip("apps.main.transformer")
    batch = _lm_batch()
    eager = build_lm([3], bf16)
    graph = build_lm([3], bf16, cuda_graph=True)
    graph.load_state_dict(eager.state_dict())
    runs_e = [lm_step(eager, *batch) for _ in range(3)]
    runs_g = [lm_step(graph, *batch) for _ in range(3)]
    from sparse_delta_memory import SparseDeltaMemory

    stats = [dict(m.graph_runner.stats) for m in graph.modules()
             if isinstance(m, SparseDeltaMemory)]
    assert stats and all(s.get("forward_replay", 0) >= 2 and s.get("backward_replay", 0) >= 2
                         for s in stats), f"premise: the graphs did not replay: {stats}"
    assert_runs_bitwise(runs_e, label="eager model")
    assert_runs_bitwise(runs_g, label="graphed model vs eager", ref=runs_e[0])


# ================================================================================== worker
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--digests", required=True, help="write the fresh-process digests here")
    ap.add_argument("--wide", action="store_true", help="also the fp8, int4, kwd, C16/37/256 "
                    "and head_dim 1024 op cases")
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(a.digests, "w") as f:
        json.dump(fresh_process_digests(wide=a.wide), f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
