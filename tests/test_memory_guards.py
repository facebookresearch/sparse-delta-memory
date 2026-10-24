# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Memory-safety canary: the SDM op's kernels write only inside their own buffers.

Each buffer the op's kernels write sits in the middle of a larger allocation whose two guard
regions (the buffer's own size, from 64 KiB to 16 MiB each) hold a 16-bit pattern that is a
signalling NaN in bfloat16 and in float32, so even a read-modify-write that adds zero changes
it. After a synchronize at the end of every forward and every backward, every guard
registered so far must still hold its pattern. A kernel that writes past the end of a buffer,
or just before its start, fails with the buffer's allocation site and the bytes it hit.

In-process tests:
- The op's inputs (k_idx, k_val, v, beta, g, q_idx, q_val) and the bank, which the op advances
  in place and the backward restores, are passed as guarded copies.
- Every allocation the package's own code makes while it runs is guarded: torch.empty, zeros,
  ones, full, empty_strided, their _like forms, Tensor.new_empty / new_zeros / new_ones /
  new_full and Tensor.clone are patched for the duration and guard the calls made from the
  package's modules. Those are all the buffers the Triton kernels and token_gather write:
  staged inputs, the solve's outputs, snapshots and quantized snapshots, gathered rows,
  delta_v, residuals, readings, mover outputs, every gradient buffer and the bank gradient.
  They also include the per-device repeated-slot counter (created again under the guards),
  the decode cache's bank and, with cuda_graph=True, the runner's static buffers and the
  graphs' workspaces.
- A fresh buffer's middle holds the pattern too, since torch.empty leaves its contents
  undefined. A kernel that reads memory nothing wrote, inside or outside its buffers, then
  computes on NaN or on other values. So every guarded call must equal the same call without
  guards bit for bit (the kernels are deterministic), and all results must be finite. A guarded
  call must also leave global state alone (TF32, matmul-precision and determinism flags,
  default dtype, grad mode, current device and stream, the CPU and CUDA RNG states (op level),
  CUDA/cuBLAS/Torch/Triton environment variables). The tensors a call returns must not change
  when the op runs again.
- The layer test runs seven configurations (the heavy bf16 and fp32 layers, head_dim 65 with
  three writes, five reads and chunks of 17, and one, three, four and five product-key
  factors), and every product-key selection and decode step must leave its read-only
  arguments unchanged.
- Not covered in-process: Phase 1's C++ extension allocates its outputs and scratch, and the
  Phase-1 backward its gradients, with at::empty, which no Python patch reaches (the test
  prints how many returned blocks happen to sit next to a guarded buffer). The registry keeps
  every guarded buffer alive, which hides lifetime bugs, and a write that skips more than a
  guard's width lands elsewhere. A buffer allocated inside a CUDA-graph capture is filled
  with its pattern by its allocation node in each replay (graphs reuse freed memory, so the
  fill cannot happen once outside the capture): an overrun by a kernel that ran earlier in
  the replay, from an unguarded C++ destination or past its own buffer's guard, is erased.
  Capture changes no kernel's addressing, so the fresh-process and native-allocator tests
  cover the same launches eagerly. PyTorch's own ops (matmuls, copies) are trusted.

Fresh-process test: every allocation PyTorch makes in a fresh interpreter, Phase 1's C++
outputs and scratch, the product-key top-k extension's and cuBLAS workspaces included, goes
through a pluggable allocator (CUDAPluggableAllocator) that gives it its own cudaMalloc with
guards of a per-allocation signalling-NaN pattern, the buffer filled with it too. Each guard
is checked when its buffer is freed and at every checkpoint. The worker runs op cases and an
eager layer's step, prefill and decode. Its results must equal this process's unguarded ones.
Buffers are freed for real here, but every allocation and free synchronizes the device.

Native-allocator test: the caching allocator, the op's own asynchronous sequence and buffer
lifetimes, no guards and no added synchronization. Before the call, a free hole of the exact
size of each allocation the call makes (Phase 1's included) is left between two live
neighbour tensors holding patterns (verified by address), and every other free block is
taken by fillers holding patterns, so that every buffer of the call lands between
neighbours (required). Right after the call, before any synchronization, new tensors are
allocated (reusing the blocks the call freed) and filled. No neighbour, filler or new tensor
may change: this is the situation of a neighbouring library's live tensors. Not covered:
CUDA-graph execution, other streams, and an overrun that stays inside its allocation's
512-byte rounding (it reaches no other tensor; the guard routes see it).

Every route also checks that a call leaves its read-only arguments (the inputs, the
cotangent and, with exact snapshots, the bank the backward restores) unchanged.

Out of scope: an out-of-bounds read whose value is discarded changes no memory and no result,
so none of these routes sees it; compute-sanitizer memcheck, run with
PYTORCH_NO_CUDA_MEMORY_CACHING=1 so that every tensor is its own allocation, does.
"""
import contextlib
import ctypes
import functools
import gc
import glob
import os
import subprocess
import sys
import time
import zlib
from collections import Counter
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from sdm_testlib import (
    OP_GRADS,
    OP_INPUTS,
    SelectionSpy,
    assert_quant_tail_refused,
    bits_equal,
    bits_mismatch,
    layer_inputs,
    make_cotangent,
    make_layer,
    make_op_inputs,
    set_nograd_budgets as _set_budgets,
    requires_cuda,
    slot_groups,
    streams_own_their_slots,
    quant_tail_refusal,
)

pytestmark = requires_cuda
bf16, fp32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64


# ======================================================================== guarded buffers
GUARD_MIN, GUARD_MAX = 64 << 10, 16 << 20  # guard bytes per side: the buffer's size, clamped
ALIGN = 512  # the caching allocator's block unit: guarded buffers keep its alignment
_EMPTY = torch.empty  # the allocator itself, never patched here


def _round(n):
    return -(-n // ALIGN) * ALIGN


def _pattern_bytes(pattern, lo, hi, device):
    """The bytes lo..hi-1 of a buffer filled with the 16-bit ``pattern``."""
    words = torch.full(((hi - lo) // 2 + 2,), pattern, dtype=torch.int16, device=device)
    return words.view(torch.uint8)[lo % 2:lo % 2 + hi - lo]


class _Guarded(NamedTuple):
    whole: torch.Tensor  # uint8: guard, buffer, guard
    lead: int            # bytes before the buffer
    nbytes: int          # the buffer's bytes
    pattern: int         # 16-bit fill
    label: str           # where it was allocated
    graph: object        # None, or the CUDA-graph capture that allocated it ("forward" or
    #                      "backward"): its replays fill and refill it


class Guards:
    """A registry of guarded buffers; ``check`` verifies their guards."""

    def __init__(self):
        self.buffers = []
        self.phase1 = []  # Phase-1 C++ outputs seen while guarding: (function, tensor)
        self.capture = None  # the CUDA-graph runner's capture in progress, if any
        self.checked = Counter()  # buffer checks per capture (None: eager buffers)

    def new(self, shape, stride, dtype, device, label):
        """A tensor of this shape, stride and dtype whose storage is the middle of a guarded
        allocation; the buffer holds the pattern too."""
        span = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        nbytes = span * dtype.itemsize
        lead = _round(min(max(nbytes, GUARD_MIN), GUARD_MAX))
        whole = _EMPTY(_round(2 * lead + nbytes), dtype=torch.uint8, device=device)
        pattern = 0x7F80 | (1 + len(self.buffers) % 63)  # a signalling NaN (bf16, and f32 pairs)
        whole.view(torch.int16).fill_(pattern)
        out = _EMPTY(0, dtype=dtype, device=device)
        out.set_(whole.untyped_storage(), lead // dtype.itemsize, tuple(shape), tuple(stride))
        graph = self.capture if torch.cuda.is_current_stream_capturing() else None
        self.buffers.append(_Guarded(whole, lead, nbytes, pattern, label, graph))
        return out

    def copy_of(self, t, label):
        """A guarded contiguous copy of ``t``."""
        with torch.no_grad():
            out = self.new(t.shape, _EMPTY(t.shape, device="meta").stride(), t.dtype, t.device,
                           label)
            out.copy_(t)
        return out

    def check(self, when, graphs=()):
        """Synchronizes, then asserts that every guard registered so far holds its pattern.

        A buffer allocated during a CUDA-graph capture holds its pattern only once that graph
        has replayed (a capture runs nothing), and graphs that share a memory pool may reuse
        one graph's freed temporaries for the next graph's buffers. So such a buffer is
        checked only when ``graphs`` names its capture: right after that graph replayed."""
        torch.cuda.synchronize()
        bad = []
        for b in self.buffers:
            if b.graph is not None and b.graph not in graphs:
                continue
            self.checked[b.graph] += 1
            end = b.lead + b.nbytes
            for side, lo, hi in (("before", 0, b.lead), ("past", end, b.whole.numel())):
                got = b.whole[lo:hi]
                want = _pattern_bytes(b.pattern, lo, hi, got.device)
                if torch.equal(got, want):
                    continue
                pos = (got != want).nonzero().flatten()
                first, last = int(pos[0]), int(pos[-1])
                where = (f"{hi - last}..{hi - first} bytes before its start" if side == "before"
                         else f"bytes {first}..{last} past its end")
                bad.append(f"{b.label} ({b.nbytes} bytes): {pos.numel()} guard bytes written, "
                           f"{where}; first values {[hex(v) for v in got[pos[:6]].tolist()]}")
        assert not bad, f"{when}: writes outside their buffers:\n  " + "\n  ".join(bad)

    def release(self):
        """Drops every buffer (a failed test's traceback keeps the registry alive)."""
        self.buffers.clear()
        self.phase1.clear()

    def phase1_neighbours(self):
        """(Phase-1 C++ output blocks, how many a guarded buffer follows, how many one
        precedes): an overrun past such a block's end, or before its start, hits a guard."""
        starts = {b.whole.data_ptr() for b in self.buffers}
        ends = {b.whole.data_ptr() + b.whole.numel() for b in self.buffers}
        blocks = after = before = 0
        for _, t in self.phase1:
            if t.numel():
                blocks += 1
                after += t.data_ptr() + _round(t.numel() * t.element_size()) in starts
                before += t.data_ptr() in ends
        return blocks, after, before


_TORCH_ALLOCATORS = ("empty", "zeros", "ones", "full", "empty_strided", "empty_like",
                     "zeros_like", "ones_like", "full_like")
_TENSOR_ALLOCATORS = ("new_empty", "new_zeros", "new_ones", "new_full", "clone")
_UNINITIALISED = ("empty", "empty_strided", "empty_like", "new_empty")


def _package():
    from sparse_delta_memory import memory_ops

    return memory_ops.__name__.rsplit(".", 1)[0]


def _guarding(guards, name, orig, package):
    """``orig``, guarding the CUDA tensors it allocates for the package's modules."""
    fresh = name in _UNINITIALISED

    def alloc(*args, **kwargs):
        out = orig(*args, **kwargs)
        caller = sys._getframe(1)
        module = caller.f_globals.get("__name__", "")
        if (not (module == package or module.startswith(package + ".")) or "out" in kwargs
                or not isinstance(out, torch.Tensor) or not out.is_cuda or out.numel() == 0
                or out.layout != torch.strided or min(out.stride()) < 0
                or 1 + sum((n - 1) * st for n, st in zip(out.shape, out.stride())) != out.numel()):
            return out  # not ours, not CUDA, empty, or not dense
        label = (f"{module.rsplit('.', 1)[-1]}.py:{caller.f_lineno} {name}{list(out.shape)} "
                 f"{str(out.dtype).removeprefix('torch.')}")
        with torch.no_grad():
            t = guards.new(out.shape, out.stride(), out.dtype, out.device, label)
            if not fresh:
                t.copy_(out)
        return t.requires_grad_(out.requires_grad)
    return alloc


def _capturing(guards, capture, orig):
    """``orig`` (a CudaGraphRunner method that may capture), tagging the buffers allocated
    while it runs with ``capture``."""
    def run(*args, **kwargs):
        prev, guards.capture = guards.capture, capture
        try:
            return orig(*args, **kwargs)
        finally:
            guards.capture = prev
    return run


def _recording(guards, name, orig):
    def call(*args, **kwargs):
        out = orig(*args, **kwargs)
        guards.phase1.extend((name, t) for t in out if isinstance(t, torch.Tensor))
        return out
    return call


_MISSING = object()


@contextlib.contextmanager
def guarded(guards):
    """Inside the block every CUDA allocation of the package's code is a guarded buffer of
    ``guards`` (tagged with the CUDA-graph capture that made it, if any), Phase 1's C++
    outputs are recorded, and the repeated-slot counter is created anew (guarded);
    everything is restored afterwards."""
    from sparse_delta_memory import cuda_graph, memory_ops

    package, undo = _package(), []

    def patch(owner, name, value):
        undo.append((owner, name, vars(owner).get(name, _MISSING)))
        setattr(owner, name, value)

    try:
        for name in _TORCH_ALLOCATORS:
            patch(torch, name, _guarding(guards, name, getattr(torch, name), package))
        for name in _TENSOR_ALLOCATORS:
            patch(torch.Tensor, name, _guarding(guards, name, getattr(torch.Tensor, name), package))
        for name in ("phase1_forward", "phase1_backward"):
            patch(memory_ops, name, _recording(guards, name, getattr(memory_ops, name)))
        runner = cuda_graph.CudaGraphRunner
        for name, capture in (("_capture_forward", "forward"), ("_backward_graphs", "backward")):
            patch(runner, name, _capturing(guards, capture, getattr(runner, name)))
        patch(memory_ops, "_repeat_counters", {})
        yield guards
    finally:
        for owner, name, value in reversed(undo):
            if value is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, value)


def _global_state(rng=True):
    """The process state a call must not change (``rng``: with the CPU and CUDA RNG
    states)."""
    try:
        autocast = torch.is_autocast_enabled("cuda")
    except TypeError:
        autocast = torch.is_autocast_enabled()
    env = ("CUBLAS", "CUDA", "CUDNN", "NCCL", "NVIDIA", "PYTORCH", "TORCH", "TRITON")
    return dict(
        tf32=(torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32),
        matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision=(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                           torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        deterministic=(torch.are_deterministic_algorithms_enabled(),
                       torch.is_deterministic_algorithms_warn_only_enabled()),
        cudnn=(torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark),
        default_dtype=torch.get_default_dtype(),
        grad=torch.is_grad_enabled(),
        autocast=autocast,
        device=torch.cuda.current_device(),
        stream=torch.cuda.current_stream().cuda_stream,
        cpu_rng=bytes(torch.get_rng_state().numpy()) if rng else None,
        cuda_rng=bytes(torch.cuda.get_rng_state().numpy()) if rng else None,
        env=sorted((k, v) for k, v in os.environ.items() if k.startswith(env)),
    )


def _assert_same_state(before, when, rng=True):
    after = _global_state(rng)
    changed = sorted(k for k in before if before[k] != after[k])
    assert not changed, f"{when} changed global state: {changed}"


def _releasing(test):
    """Runs ``test``; a failure (a failed check or an error) is reported by its message
    alone, once the test's frames, and the GPU memory they hold, are released (a traceback
    would keep them alive for the rest of the session)."""
    @functools.wraps(test)
    def run(*args, **kwargs):
        try:
            test(*args, **kwargs)
            return
        except Exception as exc:  # failed checks, and errors such as running out of memory
            message = f"{type(exc).__name__}: {exc}"
        gc.collect()
        torch.cuda.empty_cache()
        pytest.fail(message, pytrace=False)
    return run


# ============================================================================== op cases
class Case(NamedTuple):
    id: str
    dtype: torch.dtype
    quant: str        # snapshot_quant (training path)
    C: int
    T: int
    P: int
    mode: str         # "own": slot ranges of their own; "shared": one bank of P * sph rows
    #                   (num_memory_slots = P * sph) and "shared0" (num_memory_slots = 0), both
    #                   with shared_bank=True; "2d": [P * T] rows, one stream over the ranges
    slots: str        # make_op_inputs slot law
    W: int
    R: int
    hd: int
    sph: int          # slots per stream
    kwd: bool         # one gate per write
    idx: torch.dtype  # k_idx / q_idx dtype
    path: str = "train"       # "train" (forward + backward) or "nograd"
    budgets: str = "default"  # no-grad path: "tiny" (one step per slab, one stream per group)
    #                           or "two" (two of each)
    smem: int = 0             # Phase 1's shared-memory limit (SMEM_LIMIT_OVERRIDE); 0: the GPU's


CASES = [
    # id, dtype, quant, C, T, P, mode, slots, W, R, hd, sph, kwd, idx[, path, budgets]
    Case("hot_bf16_C64", bf16, "none", 64, 200, 2, "own", "hot", 16, 16, 64, 512, False, i64),
    Case("trained_fp32_C64_i32", fp32, "none", 64, 200, 2, "own", "trained", 16, 16, 64, 4096,
         False, i32),
    Case("zipf_bf16_C37_fp8", bf16, "fp8", 37, 100, 2, "own", "zipf", 16, 16, 64, 4096, False,
         i64),
    Case("clustered_fp32_C16_int4_kwd_hd129_P3", fp32, "int4", 16, 53, 3, "own", "clustered",
         16, 16, 129, 512, True, i32),
    Case("hot_bf16_C128_T90_shared", bf16, "none", 128, 90, 2, "shared", "hot", 16, 16, 64,
         512, False, i64),
    Case("zipf_fp32_C64_kwd_shared0_P3", fp32, "none", 64, 150, 3, "shared0", "zipf", 8, 8, 64,
         1024, True, i32),
    Case("hot_bf16_C48_int4_2d", bf16, "int4", 48, 130, 2, "2d", "hot", 16, 16, 64, 512, False,
         i64),
    Case("zipf_fp32_C128_fp8_P4", fp32, "fp8", 128, 257, 4, "own", "zipf", 8, 8, 64, 4096,
         False, i64),
    Case("repeat_bf16_C64", bf16, "none", 64, 150, 2, "own", "repeat", 16, 16, 64, 512, False,
         i64),
    Case("trained_bf16_C64_hd1024", bf16, "none", 64, 150, 1, "own", "trained", 16, 16, 1024,
         4096, False, i64),
    Case("zipf_bf16_C64_W1R1", bf16, "none", 64, 200, 2, "own", "zipf", 1, 1, 64, 128, False,
         i64),
    Case("trained_fp32_C64_W3R48_i32", fp32, "none", 64, 150, 2, "own", "trained", 3, 48, 64,
         768, False, i32),
    Case("hot_bf16_C64_W256_R256", bf16, "none", 64, 100, 1, "own", "hot", 256, 256, 64, 4096,
         False, i64),
    # groups of 256 entries, past Phase 1's saturating counts (POS_COUNT_MAX = 127)
    Case("hot_fp32_C256_saturated", fp32, "none", 256, 521, 2, "own", "hot", 8, 8, 64, 512,
         False, i64),
    Case("hot_bf16_C256_saturated_fp8_kwd", bf16, "fp8", 256, 521, 1, "own", "hot", 8, 8, 64,
         512, True, i32),
    Case("trained_fp32_C37_int4_wide", fp32, "int4", 37, 120, 2, "own", "trained", 16, 16, 64,
         4096, False, i32),
    Case("distinct_bf16_C64", bf16, "none", 64, 150, 2, "own", "distinct", 16, 16, 64, 8192,
         False, i64),
    # an incomplete last tile of mover entries (P * C * W not a multiple of the tile), column
    # tails (hd 65 and 1025), every step full (T a multiple of C), the staging views of one
    # stream, 32-bit sort keys (more than 65536 slots per stream), and the widest chunks
    # (C * W = 65536 entries: the four-segment scan and the most sort tiles)
    Case("tails_fp32_C17_W3R5_hd65", fp32, "none", 17, 75, 2, "own", "trained", 3, 5, 65, 256,
         False, i32),
    Case("tails_bf16_C17_W3R5_hd65_fp8", bf16, "fp8", 17, 68, 2, "own", "hot", 3, 5, 65, 256,
         False, i64),
    Case("tails_bf16_C17_T153_W3R5_hd65_P1", bf16, "none", 17, 153, 1, "own", "trained", 3, 5,
         65, 256, False, i32),
    Case("trained_bf16_C64_hd1025", bf16, "none", 64, 100, 1, "own", "trained", 16, 16, 1025,
         4096, False, i64),
    Case("hot_bf16_C64_T256_full", bf16, "none", 64, 256, 2, "own", "hot", 16, 16, 64, 512,
         False, i64),
    Case("trained_fp32_C64_T256_views", fp32, "none", 64, 256, 1, "own", "trained", 16, 16, 64,
         4096, False, i32),
    Case("zipf_bf16_C64_keys32", bf16, "none", 64, 150, 2, "own", "zipf", 16, 16, 64, 131072,
         False, i64),
    Case("hot_bf16_C256_W256_R256_seg4", bf16, "none", 256, 300, 1, "own", "hot", 256, 256, 64,
         4096, False, i64),
    # fp32: Phase 1's unstaged backward and partially staged scan; then the lean pair kernel of
    # GPUs with less shared memory per block (A100: 164 KiB)
    Case("hot_fp32_C256_W256_R256_seg4", fp32, "none", 256, 300, 1, "own", "hot", 256, 256, 64,
         4096, False, i64),
    Case("hot_bf16_C256_W256_R256_lean", bf16, "none", 256, 300, 1, "own", "hot", 256, 256, 64,
         4096, False, i64, smem=163840),
    # the solve's incomplete 64-row blocks (C = 65, 129), Phase 1's 32-bit-key merge over three
    # tiles (16512 entries per chunk, the last tile short), and the vectorized gather's partial
    # column slices (head_dim 120 in both dtypes, 264 in bf16)
    Case("zipf_bf16_C129_W64R64_hd120_keys32", bf16, "none", 129, 263, 1, "own", "zipf", 64, 64,
         120, 131072, False, i64),
    Case("trained_fp32_C65_int4_hd120", fp32, "int4", 65, 140, 2, "own", "trained", 8, 8, 120,
         4096, False, i32),
    Case("hot_bf16_C64_hd264", bf16, "none", 64, 150, 2, "own", "hot", 16, 16, 264, 512, False,
         i64),
    # the forward without backward state, its slabs of steps and groups of streams forced small
    Case("hot_bf16_C32_P3_nograd_tiny", bf16, "none", 32, 300, 3, "own", "hot", 16, 16, 64, 512,
         False, i64, "nograd", "tiny"),
    Case("zipf_fp32_C64_P3_kwd_nograd_two", fp32, "none", 64, 500, 3, "own", "zipf", 16, 16, 64,
         4096, True, i32, "nograd", "two"),
    Case("trained_bf16_C37_2d_nograd", bf16, "none", 37, 200, 2, "2d", "trained", 16, 16, 64,
         4096, False, i64, "nograd"),
    Case("hot_fp32_C64_shared_nograd_tiny", fp32, "none", 64, 200, 2, "shared", "hot", 16, 16,
         64, 512, False, i64, "nograd", "tiny"),
    Case("hot_bf16_C256_saturated_nograd_tiny", bf16, "none", 256, 521, 2, "own", "hot", 8, 8,
         64, 512, False, i64, "nograd", "tiny"),
    Case("hot_bf16_C64_W256_nograd_wide", bf16, "none", 64, 100, 1, "own", "hot", 256, 256, 64,
         4096, False, i32, "nograd"),
]


def _seed(text):
    return zlib.crc32(text.encode()) % 2**31


def _inputs(c):
    d = make_op_inputs(c.P, c.T, c.W, c.R, c.hd, c.sph, c.dtype, seed=_seed(c.id), slots=c.slots)
    if c.kwd:
        gen = torch.Generator().manual_seed(_seed(c.id + "/g"))
        d["g"] = torch.empty(c.P, c.T, c.W).uniform_(-0.5, -0.01, generator=gen).to(
            device=d["g"].device, dtype=c.dtype)
    d["k_idx"], d["q_idx"] = d["k_idx"].to(c.idx), d["q_idx"].to(c.idx)
    return d


def _premise(d, c):
    """The case has the collisions (and saturated groups) its name promises."""
    from sparse_delta_memory.cuda.phase1_cuda import POS_COUNT_MAX

    stats = slot_groups(d["k_idx"], d["q_idx"], c.C, c.mode == "2d")
    if c.slots == "distinct":
        assert stats["longest"] == (1, 1, 1), f"premise: {c.id} collides: {stats}"
    else:
        assert stats["repeated"] >= 0.1 and stats["longest"][1] >= 2, \
            f"premise: {c.id} has too few collisions: {stats}"
    if "saturated" in c.id:
        assert stats["longest"][0] > POS_COUNT_MAX, f"premise: {c.id}: {stats}"


def _configure(mp, c):
    """Applies the case's no-grad budgets, entry width and shared-memory limit through ``mp``
    (a pytest MonkeyPatch); returns the expected (slab_steps, group_streams) or None."""
    from sparse_delta_memory import memory_ops
    from sparse_delta_memory.cuda import phase1_cuda

    expected = _set_budgets(mp, memory_ops, c)
    if c.smem:
        mp.setattr(phase1_cuda, "SMEM_LIMIT_OVERRIDE", c.smem)
        assert phase1_cuda.pairs_config(c.C, c.W, c.R)["lean"], \
            f"premise: {c.id} does not run the lean pair kernel"
    return expected


def _call(d, c, guards=None, checkpoint=None, start=None):
    """One call of the op on copies of ``d``: forward, and with path "train" the backward.
    With ``guards`` the copies are guarded, every allocation of the package is guarded, every
    guard is checked after the forward and after the backward, and the call must leave global
    state alone. ``start(when)`` runs before and ``checkpoint(when)`` after each of the two.
    Returns (results: clones of everything to compare, returned: what the op returned,
    passed: the inputs and cotangent the op was given, for ``_mutated``)."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T, train = c.P, c.T, c.path == "train"
    kw = dict(chunk_size=c.C, key_weighted_decay=c.kwd)
    if c.mode == "own":
        kw.update(num_memory_slots=c.sph, num_parallel=P)
    elif c.mode in ("shared", "shared0"):
        assert streams_own_their_slots(d["k_idx"], c.sph), c.id
        assert streams_own_their_slots(d["q_idx"], c.sph), c.id
        kw.update(num_memory_slots=P * c.sph if c.mode == "shared" else 0, num_parallel=P,
                  shared_bank=True)
    elif c.mode == "2d":
        kw.update(num_memory_slots=0, num_parallel=1)
    else:
        raise ValueError(c.mode)
    if train:
        kw["snapshot_quant"] = c.quant
    if guards is not None:
        checkpoint = guards.check

    def copy(n):
        return guards.copy_of(d[n], f"input {n}") if guards else d[n].detach().clone()

    def lay(t):
        return t.reshape(P * T, -1) if c.mode == "2d" else t

    def scope():
        return guarded(guards) if guards else contextlib.nullcontext()

    leaves = {n: copy(n).requires_grad_(train) for n in OP_INPUTS}
    ids = {n: copy(n) for n in ("k_idx", "q_idx")}
    # The cotangent is made before the forward: its host-to-device copy would wait for it.
    cot = make_cotangent(SimpleNamespace(shape=(P * T, c.hd), device=d["memory"].device,
                                         dtype=c.dtype), "dense") if train else None
    args = (leaves["memory"], lay(ids["k_idx"]), lay(leaves["k_val"]), lay(leaves["v"]),
            lay(leaves["beta"]), lay(leaves["g"]), lay(ids["q_idx"]), lay(leaves["q_val"]))
    state = _global_state()
    if start:
        start(f"{c.id} forward")
    with scope(), torch.set_grad_enabled(train):
        readings = sdm_write_read(*args, **kw)
    if checkpoint:
        checkpoint(f"{c.id} forward")
    if guards:
        _assert_same_state(state, f"{c.id} forward")
    res = {"readings": readings.detach().clone(),
           "final_bank": leaves["memory"].detach().clone()}
    returned = {"readings": readings}
    if train:
        state = _global_state()
        if start:
            start(f"{c.id} backward")
        with scope():
            grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS], cot)
        if checkpoint:
            checkpoint(f"{c.id} backward")
        if guards:
            _assert_same_state(state, f"{c.id} backward")
        res["restored_bank"] = leaves["memory"].detach().clone()
        res.update((n, g.detach().clone()) for n, g in zip(OP_GRADS, grads))
        returned.update(zip(OP_GRADS, grads))
    passed = dict(leaves, **ids, cotangent=cot)
    return res, returned, passed


def _mutated(d, c, res, passed):
    """The read-only arguments of a call whose contents changed: every input but the bank
    (the training path's restored bank too, with exact snapshots), and the cotangent."""
    out = [n for n in ("k_idx", "q_idx", "k_val", "v", "beta", "g", "q_val")
           if not bits_equal(d[n], passed[n].detach())]
    if c.path == "train":
        cot = make_cotangent(SimpleNamespace(shape=(c.P * c.T, c.hd), device=d["memory"].device,
                                             dtype=c.dtype), "dense")
        out += [] if bits_equal(cot, passed["cotangent"]) else ["cotangent"]
        if c.quant == "none" and not bits_equal(d["memory"], res["restored_bank"]):
            out.append("restored bank")
    return out


def _compare(want, got, label):
    """Every result finite and bitwise equal (the kernels are deterministic)."""
    nonfinite = [k for run in (want, got) for k, t in run.items()
                 if t.is_floating_point() and not bool(torch.isfinite(t).all())]
    assert not nonfinite, f"{label}: non-finite results {sorted(set(nonfinite))}"
    bad = {k: bits_mismatch(want[k], got[k]) for k in want if bits_mismatch(want[k], got[k])}
    assert not bad, f"{label}: the guarded call differs from the unguarded one: {bad}"


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
@_releasing
def test_op_writes_stay_inside_their_buffers(case, monkeypatch):
    """Guards around the op's inputs, its bank and every buffer it allocates hold after the
    forward and after the backward; the guarded call equals the unguarded one, leaves global
    state alone and does not change what an earlier call returned."""
    from sparse_delta_memory import memory_ops

    cls = _refused(case)
    if cls is not None:      # fp8 / int4 rows of a quantized tail head: refused before any work (nothing to guard)
        d = _inputs(case)
        assert_quant_tail_refused(lambda: _call(d, case), cls)
        return
    expected_budgets = _configure(monkeypatch, case)
    d = _inputs(case)
    _premise(d, case)
    want, returned, passed = _call(d, case)  # also compiles and loads everything first
    assert not _mutated(d, case, want, passed), _mutated(d, case, want, passed)
    kept = {k: t.detach().clone() for k, t in returned.items()}
    guards = Guards()
    try:
        got, _, passed = _call(d, case, guards)
        changed = _mutated(d, case, got, passed)
        assert not changed, f"the guarded call changed its read-only arguments {changed}"
        internal = [b.label for b in guards.buffers if not b.label.startswith("input ")]
        assert len(internal) >= 10, f"premise: few allocations were guarded: {internal}"
        if expected_budgets:
            dispatch = memory_ops.last_dispatch()
            assert (dispatch["slab_steps"], dispatch["group_streams"]) == expected_budgets, \
                dispatch
        changed = [k for k, t in returned.items() if not bits_equal(kept[k], t)]
        assert not changed, f"a later call changed tensors an earlier call returned: {changed}"
        _compare(want, got, case.id)
        blocks, after, before = guards.phase1_neighbours()
        print(f"{case.id}: {len(guards.buffers)} guarded buffers; {blocks} Phase-1 "
              f"C++ output blocks, {after} followed and {before} preceded by a guarded buffer")
    finally:
        guards.release()


# =========================================================================== layer level
# 256 slots per head (16 x 16 product keys), 32 writes and reads per token: heavy collisions.
HEAVY = dict(dim=256, num_heads=2, slots_per_head=256, num_writes=32, num_reads=32,
             chunk_size=48, output_gate=True, normalize_readings=True)
LAYER_CASES = [
    # label, dtype, configuration
    ("heavy_bf16", bf16, HEAVY),
    ("heavy_fp32", fp32, HEAVY),
    # head_dim 65, three writes and five reads, chunks of 17: column, entry and chunk tails in
    # the layer's permutation, broadcast, write/read and decode kernels
    ("tails_bf16", bf16, dict(HEAVY, dim=130, num_writes=3, num_reads=5, chunk_size=17)),
    # one and five product-key factors: the any-factor top-k extension's two paths; three and
    # four: the bit-packed extension's multi-factor variant
    ("one_factor_bf16", bf16, dict(HEAVY, product_key_factor_dims=[256])),
    ("three_factors_bf16", bf16, dict(HEAVY, product_key_factor_dims=[4, 4, 16])),
    ("four_factors_bf16", bf16, dict(HEAVY, product_key_factor_dims=[4, 4, 4, 4])),
    ("five_factors_bf16", bf16, dict(HEAVY, product_key_factor_dims=[2, 2, 2, 2, 16])),
]


@contextlib.contextmanager
def _checked_selector(layer):
    """Every product-key selection of ``layer`` inside the block must leave its scores
    unchanged; yields the list of the shapes of those that did not."""
    orig, changed = layer._product_keys, []

    def checked(x, num_keys):
        before = x.detach().clone()
        out = orig(x, num_keys)
        if not bits_equal(before, x.detach()):
            changed.append(tuple(x.shape))
        return out

    layer._product_keys = checked
    try:
        yield changed
    finally:
        layer._product_keys = orig


def _layer_step(layer, x, gy, checkpoint=None, label="", start=None, keep=None):
    """Output, final bank, input gradient and every parameter gradient of one training step,
    with ``start`` before and ``checkpoint`` after the forward and the backward. ``keep``
    receives (the results, the tensors the step returned)."""
    for p in layer.parameters():
        p.grad = None
    xi = x.detach().clone().requires_grad_(True)
    if start:
        start(f"{label} forward")
    out, state = layer(xi)
    if checkpoint:
        checkpoint(f"{label} forward")
    res = {"out": out.detach().clone(), "final_bank": state.memory.detach().clone()}
    if start:
        start(f"{label} backward")
    out.backward(gy)
    if checkpoint:
        checkpoint(f"{label} backward")
    res["grad_x"] = xi.grad.detach().clone()
    res.update({"grad." + n: p.grad.detach().clone() for n, p in layer.named_parameters()})
    if keep is not None:
        returned = {"out": out.detach(), "grad_x": xi.grad}
        returned.update({"grad." + n: p.grad for n, p in layer.named_parameters()})
        keep.append((res, returned))
    return res


def _layer_eval(layer, x, checkpoint=None, start=None):
    """A no-grad prefill of ``x``, then three decode steps from a fresh cache; each decode
    step must leave its read-only arguments unchanged."""
    import importlib

    layer_module = importlib.import_module(_package() + ".layer")
    decode, changed = layer_module.sdm_decode_step, []

    def checked_decode(memory, *args):
        before = [a.detach().clone() for a in args]
        out = decode(memory, *args)
        changed.extend(i for i, (a, b) in enumerate(zip(args, before)) if not bits_equal(a, b))
        return out

    layer.eval()
    layer_module.sdm_decode_step = checked_decode
    try:
        with torch.no_grad():
            if start:
                start("prefill and decode")
            out, state = layer(x)
            if checkpoint:
                checkpoint("prefill")
            res = {"prefill": out.clone(), "prefill_bank": state.memory.clone()}
            cache = layer.create_kv_cache(bsz=x.shape[0], seq_len=0, dtype=x.dtype,
                                          device="cuda")
            res["decode"] = torch.cat([layer(x[:, t:t + 1], cache=cache)[0] for t in range(3)], 1)
            res["decode_bank"] = cache.memory.clone()
            if checkpoint:
                checkpoint("decode")
    finally:
        layer_module.sdm_decode_step = decode
        layer.train()
    assert not changed, f"decode steps changed their read-only arguments (positions {changed})"
    return res


@pytest.mark.parametrize("label,dtype,cfg", LAYER_CASES, ids=[c[0] for c in LAYER_CASES])
@_releasing
def test_layer_and_cuda_graph_writes_stay_inside_their_buffers(label, dtype, cfg):
    """The layer with cuda_graph=True under guards for a whole sequence: the capture, replays
    on new inputs (new selections, so new collision patterns), the eager fallback under
    suspend_cuda_graphs, a no-grad prefill and decode steps. Every guard holds after every
    forward and backward, and every step equals the unguarded eager layer's."""
    from sparse_delta_memory.cuda_graph import reset_cuda_graphs, suspend_cuda_graphs

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    reset_cuda_graphs()
    try:
        eager = make_layer(seed=3, dtype=dtype, **cfg)
        graph = make_layer(seed=3, dtype=dtype, cuda_graph=True, **cfg)
        graph.load_state_dict(eager.state_dict())
        steps = [layer_inputs(2, 130, cfg["dim"], dtype=dtype, seed=s) for s in range(4)]
        spy = SelectionSpy(graph)
        guards = Guards()
        replays = [0]

        def checkpoint(when):
            # The forward graph replays in every graphed forward (also the capturing one);
            # the backward graph's buffers are checked right after its own replays.
            n = graph.graph_runner.stats.get("backward_replay", 0)
            guards.check(when, ("forward", "backward") if n > replays[0] else ("forward",))
            replays[0] = n

        got, kept = [], []
        state = _global_state(rng=False)  # a capture may move the CUDA RNG's graph offsets
        with guarded(guards), _checked_selector(graph) as selections_changed:
            for i, (x, gy) in enumerate(steps):
                got.append(_layer_step(graph, x, gy, checkpoint, f"graph step {i}", keep=kept))
                _assert_same_state(state, f"graph step {i}", rng=False)
            with suspend_cuda_graphs():
                got.append(_layer_step(graph, *steps[1], checkpoint, "eager fallback",
                                       keep=kept))
            got.append(_layer_eval(graph, steps[2][0], checkpoint))
            _assert_same_state(state, "eager fallback, prefill and decode", rng=False)
        assert not selections_changed, f"selections changed their scores: {selections_changed}"
        assert all(guards.checked[k] for k in (None, "forward", "backward")), \
            f"premise: eager and captured buffers were all checked: {dict(guards.checked)}"
        stats = dict(graph.graph_runner.stats)
        assert (stats.get("forward_capture") == 1 and stats.get("forward_replay", 0) >= 3
                and stats.get("backward_replay", 0) >= 2), f"premise: no replays: {stats}"
        writes = [idx for _, idx in spy.take()[::2]]  # each forward selects writes, then reads
        assert not any(torch.equal(writes[0], w) for w in writes[1:4]), \
            "premise: the replays select the same slots as the capture"
        changed = [(i, k) for i, (res, returned) in enumerate(kept) for k, t in returned.items()
                   if not bits_equal(res[k], t)]
        assert not changed, f"later steps changed tensors earlier steps returned: {changed}"

        def eager_steps():
            out = [_layer_step(eager, x, gy) for x, gy in steps]
            return out + [_layer_step(eager, *steps[1]), _layer_eval(eager, steps[2][0])]

        want = eager_steps()
        for i, (w, g) in enumerate(zip(want, got)):
            _compare(w, g, f"layer {label} step {i}")
        blocks, after, before = guards.phase1_neighbours()
        print(f"layer {label}: {len(guards.buffers)} guarded buffers "
              f"({dict(Counter(b.graph for b in guards.buffers))} by capture); {blocks} "
              f"Phase-1 C++ output blocks, {after} followed and {before} preceded by a "
              f"guarded buffer; runner {stats}")
    finally:
        if "guards" in locals():
            guards.release()
        reset_cuda_graphs()
        torch.backends.cuda.matmul.allow_tf32 = prev


# ====================================================== every allocation, in a fresh process
# A pluggable CUDA allocator: one cudaMalloc per allocation, padded with guards (the buffer's
# size, 64 KiB to 16 MiB each side) and filled, buffer included, with a per-allocation 16-bit
# signalling NaN; a guard is checked when its buffer is freed and at every checkpoint. Any
# failing CUDA call is recorded as a finding. Single device.
_ALLOCATOR = r"""
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
const size_t kAlign = 512, kMin = 64 << 10, kMax = 16 << 20, kChunk = 4 << 20;
struct Block { char* base; size_t lead, size, total; uint16_t pattern; };
std::mutex mu;
std::unordered_map<void*, Block> live;
std::vector<std::string> found;
std::vector<uint16_t> host;
std::string label = "start";
long long allocations = 0;

size_t round_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

bool ok(cudaError_t err, const char* what) {  // holds mu
  if (err == cudaSuccess) return true;
  found.push_back(label + ": " + what + " failed: " + cudaGetErrorString(err));
  return false;
}

bool fill(char* dst, size_t bytes, uint16_t pattern) {  // holds mu; bytes is even
  if (host.size() != kChunk / 2 || host[0] != pattern) host.assign(kChunk / 2, pattern);
  for (size_t off = 0; off < bytes; off += kChunk) {
    const size_t n = bytes - off < kChunk ? bytes - off : kChunk;
    if (!ok(cudaMemcpy(dst + off, host.data(), n, cudaMemcpyHostToDevice), "cudaMemcpy"))
      return false;
  }
  return true;
}

void check_block(const Block& b, const char* when) {  // holds mu
  const size_t past = b.lead + b.size;
  const size_t offs[2] = {0, past}, lens[2] = {b.lead, b.total - past};
  std::vector<unsigned char> got;
  for (int side = 0; side < 2; ++side) {
    got.resize(lens[side]);
    if (!ok(cudaMemcpy(got.data(), b.base + offs[side], lens[side], cudaMemcpyDeviceToHost),
            "cudaMemcpy of a guard"))
      return;
    size_t first = 0, last = 0, count = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const unsigned char want = (offs[side] + i) % 2 ? b.pattern >> 8 : b.pattern & 0xFF;
      if (got[i] != want) {
        if (!count) first = i;
        last = i;
        ++count;
      }
    }
    if (count) {
      char msg[320];
      if (side == 0)
        snprintf(msg, sizeof msg, "%s %s: a buffer of %zu bytes: %zu guard bytes written, "
                 "%zu..%zu bytes before its start", label.c_str(), when, b.size, count,
                 b.lead - last, b.lead - first);
      else
        snprintf(msg, sizeof msg, "%s %s: a buffer of %zu bytes: %zu guard bytes written, "
                 "bytes %zu..%zu past its end", label.c_str(), when, b.size, count, first, last);
      found.push_back(msg);
    }
  }
}
}  // namespace

extern "C" {
void* sdm_guard_malloc(ssize_t size, int device, cudaStream_t stream) {
  const size_t n = size > 0 ? static_cast<size_t>(size) : 0;
  const size_t lead = round_up(n < kMin ? kMin : (n > kMax ? kMax : n));
  const size_t total = round_up(2 * lead + n);
  std::lock_guard<std::mutex> lock(mu);
  int prev = 0;
  if (!ok(cudaGetDevice(&prev), "cudaGetDevice")) return nullptr;
  if (prev != device && !ok(cudaSetDevice(device), "cudaSetDevice")) return nullptr;
  char* base = nullptr;
  void* user = nullptr;
  const uint16_t pattern = static_cast<uint16_t>(0x7F80 | (1 + allocations % 63));
  if (cudaMalloc(reinterpret_cast<void**>(&base), total) == cudaSuccess) {
    if (ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && fill(base, total, pattern)) {
      user = base + lead;
      live[user] = Block{base, lead, n, total, pattern};
      ++allocations;
    } else {
      cudaFree(base);
    }
  }
  if (prev != device) cudaSetDevice(prev);
  return user;  // nullptr: PyTorch reports the allocation failure
}

void sdm_guard_free(void* ptr, size_t size, int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mu);
  int prev = 0;
  cudaGetDevice(&prev);
  if (prev != device) cudaSetDevice(device);
  ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize before a free");  // any stream's work
  auto it = live.find(ptr);
  if (it == live.end()) {
    found.push_back(label + ": free of an unknown pointer");
  } else {
    check_block(it->second, "at free");
    ok(cudaFree(it->second.base), "cudaFree");
    live.erase(it);
  }
  if (prev != device) cudaSetDevice(prev);
}

void sdm_guard_label(const char* text) {
  std::lock_guard<std::mutex> lock(mu);
  label = text;
}

int sdm_guard_check(const char* when) {
  std::lock_guard<std::mutex> lock(mu);
  if (ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize before a check"))
    for (auto& kv : live) check_block(kv.second, when);
  return static_cast<int>(found.size());
}

int sdm_guard_count() {
  std::lock_guard<std::mutex> lock(mu);
  return static_cast<int>(found.size());
}

const char* sdm_guard_message(int i) {
  std::lock_guard<std::mutex> lock(mu);
  return found[i].c_str();
}

long long sdm_guard_allocations() {
  std::lock_guard<std::mutex> lock(mu);
  return allocations;
}
}
"""

# Cases of the fresh-process test.
FRESH_CASES = ("hot_bf16_C64", "zipf_bf16_C37_fp8", "clustered_fp32_C16_int4_kwd_hd129_P3",
               "zipf_fp32_C64_kwd_shared0_P3", "hot_bf16_C48_int4_2d", "repeat_bf16_C64",
               "trained_bf16_C64_hd1024", "trained_fp32_C64_W3R48_i32", "hot_fp32_C256_saturated",
               "trained_fp32_C37_int4_wide", "distinct_bf16_C64", "tails_fp32_C17_W3R5_hd65",
               "tails_bf16_C17_W3R5_hd65_fp8", "tails_bf16_C17_T153_W3R5_hd65_P1",
               "trained_bf16_C64_hd1025", "trained_fp32_C64_T256_views", "zipf_bf16_C64_keys32",
               "hot_bf16_C256_W256_R256_seg4", "zipf_bf16_C129_W64R64_hd120_keys32",
               "trained_fp32_C65_int4_hd120", "hot_bf16_C64_hd264", "hot_bf16_C32_P3_nograd_tiny",
               "hot_bf16_C256_saturated_nograd_tiny", "hot_fp32_C256_W256_R256_seg4",
               "hot_bf16_C256_W256_R256_lean")


def _refused(c):
    """The class with which the case's training call is refused before any work (fp8 / int4 rows of a quantized tail
    head), or None (no-grad cases never quantize)."""
    return quant_tail_refusal(c.dtype, c.hd, c.quant) if c.path == "train" else None


def _fresh_cases():   # (a refused training call allocates nothing: test_op_writes_stay_inside_their_buffers checks it)
    return [c for c in CASES if c.id in FRESH_CASES and _refused(c) is None]


def _build_allocator(out_dir):
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        pytest.skip("no CUDA toolkit to build the guard allocator")
    src, lib = os.path.join(out_dir, "guard_alloc.cpp"), os.path.join(out_dir, "guard_alloc.so")
    with open(src, "w") as f:
        f.write(_ALLOCATOR)
    # The toolkit's own layout (include/, lib64/) or a conda one (targets/<arch>/include and lib/).
    targets = sorted(glob.glob(os.path.join(CUDA_HOME, "targets", "*")))
    incdir = next((d for d in [os.path.join(CUDA_HOME, "include")] + [os.path.join(t, "include") for t in targets]
                   if os.path.exists(os.path.join(d, "cuda_runtime_api.h"))), os.path.join(CUDA_HOME, "include"))
    libdirs = [os.path.join(CUDA_HOME, "lib64"), os.path.join(CUDA_HOME, "lib")] + [os.path.join(t, "lib")
                                                                                 for t in targets]
    # -lcudart needs the unversioned libcudart.so; otherwise link a versioned one by its path.
    libdir = next((d for d in libdirs if os.path.exists(os.path.join(d, "libcudart.so"))), None)
    if libdir is not None:
        link = [f"-L{libdir}", f"-Wl,-rpath,{libdir}", "-lcudart"]
    else:
        found = sorted(f for d in libdirs for f in glob.glob(os.path.join(d, "libcudart.so.*")))
        link = ([found[-1], f"-Wl,-rpath,{os.path.dirname(found[-1])}"] if found else
                [f"-L{libdirs[0]}", f"-Wl,-rpath,{libdirs[0]}", "-lcudart"])
    cmd = [os.environ.get("CXX", "c++"), "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{incdir}", src, "-o", lib,
           *link]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, f"building the guard allocator failed: {done.stderr[-2000:]}"
    return lib


def _results_for(cases, checkpoint=None, start=None, mutated=None):
    """{case/mode: results} of the unguarded call of every case, with each case's budgets and
    entry width applied (and undone)."""
    out = {}
    for c in cases:
        with pytest.MonkeyPatch.context() as mp:
            _configure(mp, c)
            d = _inputs(c)
            res, _, passed = _call(d, c, checkpoint=checkpoint, start=start)
            if mutated is not None:
                mutated += [f"{c.id} {n}" for n in _mutated(d, c, res, passed)]
        out[c.id] = {k: t.cpu() for k, t in res.items()}
    return out


def _layer_results(checkpoint=None, start=None):
    """{layer/mode: results} of eager layers' training step, no-grad prefill and decode steps
    (product-key top-k, head split and merge, the bank expand, the decode kernel), for every
    configuration of LAYER_CASES; each selection must leave its scores unchanged."""
    out = {}
    for label, dtype, cfg in LAYER_CASES:
        layer = make_layer(seed=3, dtype=dtype, **cfg)
        x, gy = layer_inputs(2, 130, cfg["dim"], dtype=dtype, seed=0)
        with _checked_selector(layer) as changed:
            res = _layer_step(layer, x, gy, checkpoint, f"layer {label}", start)
            res.update(_layer_eval(layer, x, checkpoint, start))
        assert not changed, f"layer {label}: selections changed their scores: {changed}"
        out[f"layer {label}"] = {k: t.cpu() for k, t in res.items()}
    return out


def guarded_process_main():
    """The fresh-process worker: argv = [allocator .so, output file, default dtype]."""
    lib_path, out_path, default_dtype = sys.argv[1:4]
    torch.cuda.memory.change_current_allocator(torch.cuda.memory.CUDAPluggableAllocator(
        lib_path, "sdm_guard_malloc", "sdm_guard_free"))
    torch.set_default_dtype(getattr(torch, default_dtype.removeprefix("torch.")))
    lib = ctypes.CDLL(lib_path)
    lib.sdm_guard_label.argtypes = [ctypes.c_char_p]
    lib.sdm_guard_check.argtypes = [ctypes.c_char_p]
    lib.sdm_guard_message.restype = ctypes.c_char_p
    lib.sdm_guard_allocations.restype = ctypes.c_longlong
    torch.backends.cuda.matmul.allow_tf32 = False

    def start(when):  # names the phase in the messages of the frees and checks that follow
        lib.sdm_guard_label(when.encode())

    def checkpoint(when):
        lib.sdm_guard_check(b"at its end")

    mutated = []
    results = _results_for(_fresh_cases(), checkpoint, start, mutated)
    results.update(_layer_results(checkpoint, start))
    torch.cuda.synchronize()
    lib.sdm_guard_check(b"end")
    found = [lib.sdm_guard_message(i).decode() for i in range(lib.sdm_guard_count())]
    found += [f"{text}: a read-only argument changed" for text in mutated]
    torch.save({"found": found, "allocations": lib.sdm_guard_allocations(),
                "results": results}, out_path)


@_releasing
def test_every_allocation_guarded_in_a_fresh_process(tmp_path):
    """In a fresh interpreter whose every CUDA allocation (Phase 1's C++ outputs, cuBLAS
    workspaces and PyTorch's own included) has guards and starts as 0xFF bytes, no guard is
    written by the op cases nor by an eager layer's step, prefill and decode, and the results
    equal this process's unguarded ones."""
    lib = _build_allocator(str(tmp_path))
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        want = dict(_results_for(_fresh_cases()), **_layer_results())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    # The worker imports this file by its path, with this process's sys.path (where pytest
    # found the code under test and the helpers).
    path = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=path)
    out = tmp_path / "results.pt"
    code = ("import importlib.util as u; "
            f"s = u.spec_from_file_location('sdm_memory_guards_worker', {__file__!r}); "
            "m = u.module_from_spec(s); s.loader.exec_module(m); m.guarded_process_main()")
    t0 = time.monotonic()
    with open(tmp_path / "worker.log", "w") as log:
        proc = subprocess.run([sys.executable, "-B", "-c", code, lib, str(out),
                               str(torch.get_default_dtype())], env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=1800)
    tail = (tmp_path / "worker.log").read_text()[-4000:]
    assert proc.returncode == 0, f"the worker failed (rc={proc.returncode}):\n{tail}"
    data = torch.load(out)
    assert not data["found"], "writes outside their buffers:\n  " + "\n  ".join(data["found"])
    assert data["allocations"] >= 1000, f"premise: {data['allocations']} guarded allocations"
    assert data["results"].keys() == want.keys(), sorted(data["results"].keys() ^ want.keys())
    for key, w in want.items():
        _compare(w, data["results"][key], f"fresh process {key}")
    print(f"fresh process: {data['allocations']} guarded allocations, "
          f"{time.monotonic() - t0:.0f} s")


# ================================================ live neighbours, with the native allocator
_SMALL_POOL = 1 << 20  # the caching allocator serves allocations up to 1 MiB from 2 MiB segments
_OWN_SEGMENT = 10 << 20  # and makes a segment of its own for a miss from 10 MiB on
_NEIGHBOUR = {True: 512, False: (2 << 20) + 512}  # a neighbour in the small / the large pool
NEIGHBOUR_CASES = ("hot_bf16_C64", "trained_fp32_C64_i32", "zipf_bf16_C37_fp8",
                   "clustered_fp32_C16_int4_kwd_hd129_P3", "hot_bf16_C128_T90_shared",
                   "hot_fp32_C256_saturated", "tails_fp32_C17_W3R5_hd65",
                   "tails_bf16_C17_W3R5_hd65_fp8", "tails_bf16_C17_T153_W3R5_hd65_P1",
                   "hot_bf16_C256_W256_R256_seg4", "zipf_bf16_C129_W64R64_hd120_keys32",
                   "trained_fp32_C65_int4_hd120", "hot_bf16_C64_hd264",
                   "hot_fp32_C256_W256_R256_seg4", "hot_bf16_C256_W256_R256_lean",
                   "hot_bf16_C32_P3_nograd_tiny", "zipf_fp32_C64_P3_kwd_nograd_two")


def _traced(fn):
    """``fn()`` and the caching allocator's trace of it (no synchronization inside ``fn``):
    (result, [(address, requested bytes, owner) of each allocation, in order], {addresses
    freed})."""
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(enabled="all", context="alloc", stacks="python",
                                             max_entries=1 << 20, clear_history=True)
    try:
        out = fn()
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    trace = snap["device_traces"][torch.cuda.current_device()]

    def owner(e):  # the innermost frame in the package or the tests
        for f in e.get("frames") or ():
            name = f.get("filename", "")
            if "sparse_delta_memory" in name or "test_memory_guards" in name:
                return f"{os.path.basename(name)}:{f.get('line')} {f.get('name')}"
        return "?"

    allocs = [(e["addr"], e["size"], owner(e)) for e in trace if e["action"] == "alloc"]
    freed = {e["addr"] for e in trace if e["action"] in ("free_requested", "free_completed")}
    return out, allocs, freed


def _free_blocks(device):
    """(address, bytes, pool) of the free blocks of this stream's default pool on ``device``
    that an allocation can use (a large-pool block of 1 MiB or less serves none)."""
    stream = torch.cuda.current_stream(device).cuda_stream
    return [(b["address"], b["size"], seg["segment_type"]) for seg in torch.cuda.memory_snapshot()
            if seg["device"] == device.index and seg["stream"] == stream
            and tuple(seg.get("segment_pool_id", (0, 0))) == (0, 0)  # not a graph's pool
            for b in seg["blocks"] if b["state"] == "inactive"
            and (seg["segment_type"] == "small" or b["size"] > _SMALL_POOL)]


def _fill_free_blocks(device):
    """Fillers, as (tensor, pattern), that take every usable free block."""
    fillers = []
    for _ in range(64):
        free = _free_blocks(device)
        if not free:
            return fillers
        fillers += [_neighbour(min(n, _SMALL_POOL) if pool == "small" else n, device,
                               _patterns(len(fillers) + i)[0])
                    for i, (_, n, pool) in enumerate(free)]
    raise AssertionError(f"premise: free blocks remain after filling: {free[:8]}")


def _neighbour(nbytes, device, pattern):
    t = _EMPTY(nbytes, dtype=torch.uint8, device=device)
    t.view(torch.int16).fill_(pattern)
    return t, pattern


def _changed(tensors):
    """(address, description) of the (tensor, pattern) pairs whose bytes changed."""
    out = []
    for t, pattern in tensors:
        want = _pattern_bytes(pattern, 0, t.numel(), t.device)
        if not torch.equal(t, want):
            pos = (t != want).nonzero().flatten()
            out.append((t.data_ptr(), f"{t.numel()} bytes at {t.data_ptr():#x}: bytes "
                                      f"{int(pos[0])}..{int(pos[-1])} written ({pos.numel()})"))
    return out


def _patterns(k):
    """Two different signalling-NaN patterns (bf16, and f32 pairs)."""
    return 0x7F80 | (1 + k % 63), 0x7F80 | (1 + (k + 1) % 63)


def _sandwich(sizes, device):
    """For each size, a free hole between two live neighbour tensors holding patterns that
    touch it (checked by address), and no other usable free block: (neighbours and fillers
    as (tensor, pattern), {address: bytes} of the holes).

    No free block is left before a neighbour, hole, neighbour triple is allocated, so the
    three come from the last segment's free tail, in order, or from a new segment; a triple
    that does not touch (a new segment started inside it, or a block was not split) stays
    allocated, the free blocks are taken again, and the triple is tried again. From 10 MiB,
    where a miss makes a segment of its own, the triple is carved out of a freed arena of
    its total size, the only free block at that point. Unreachable reference cycles are
    collected first: an earlier call's buffers held by a cycle (the CUDA kernels'
    launch closures keep their tensors as attributes) would otherwise be freed by a later
    garbage collection, in the middle of the sandwich, as free blocks that are not holes."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    kept = _fill_free_blocks(device)
    holes, touching = [], {}
    for i, size in enumerate(sizes):
        nbytes = _round(max(size, 1))
        nb = _NEIGHBOUR[nbytes <= _SMALL_POOL]
        first, second = _patterns(2 * i)
        for _ in range(4):
            if nbytes >= _OWN_SEGMENT:
                arena = [_EMPTY(2 * nb + nbytes, dtype=torch.uint8, device=device)]
                kept += _fill_free_blocks(device)  # the arena's segment's tail
                arena.clear()  # now the only free block
            before = _neighbour(nb, device, first)
            hole = _EMPTY(nbytes, dtype=torch.uint8, device=device)
            after = _neighbour(nb, device, second)
            kept += [before, after]
            if (before[0].data_ptr() + nb == hole.data_ptr()
                    and hole.data_ptr() + nbytes == after[0].data_ptr()):
                touching[hole.data_ptr()] = nbytes
                holes.append(hole)
                break
            hole.view(torch.int16).fill_(first)  # a failed hole stays, as a checked filler
            kept.append((hole, first))
            kept += _fill_free_blocks(device)
        hole = None
    kept += _fill_free_blocks(device)  # the last segments' tails, while the holes are held
    holes = None  # frees the holes
    left = [(a, n, pool) for a, n, pool in _free_blocks(device) if touching.get(a) != n]
    assert not left, f"premise: free blocks other than the holes: {left[:8]}"
    torch.cuda.synchronize()
    return kept, touching


@pytest.mark.parametrize("case", [c for c in CASES if c.id in NEIGHBOUR_CASES and _refused(c) is None],
                         ids=[c.id for c in CASES if c.id in NEIGHBOUR_CASES and _refused(c) is None])
@_releasing
def test_live_neighbours_survive_the_op_with_the_native_allocator(case, monkeypatch):
    """With the caching allocator, the op's own asynchronous sequence and buffer lifetimes, no
    guards and no added synchronization: every allocation of the call (Phase 1's C++
    outputs and scratch included) lands in a hole of its exact size between two live
    neighbour tensors holding patterns, the only free memory left; everything else the
    allocator had free is taken by fillers holding patterns too. Right after the call,
    tensors allocated into the blocks it freed are filled. No neighbour, filler or such
    tensor changes, and neither does any read-only argument of the call."""
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        pytest.skip("assumes the caching allocator's default configuration")
    _configure(monkeypatch, case)
    d = _inputs(case)
    want, allocs, _ = _traced(lambda: _call(d, case)[0])  # also compiles everything
    sizes = [size for _, size, _ in allocs]
    assert sizes, "premise: the allocator traced no allocation"
    kept, holes = _sandwich(sizes, d["memory"].device)

    def call_then_reuse():
        res, _, passed = _call(d, case)
        # Queued behind the call (no synchronization): new tensors in the blocks it freed.
        late = [_neighbour(_round(max(n, 1)), d["memory"].device, _patterns(i)[0])
                for i, n in enumerate(sizes)]
        return res, passed, late

    (got, passed, late), allocs, freed = _traced(call_then_reuse)
    call_allocs = allocs[:len(allocs) - len(late)]  # each late tensor is one allocation
    ends = {}  # every allocation of the call that ended where a neighbour starts
    for i, (addr, n, who) in enumerate(call_allocs):
        ends.setdefault(addr + _round(n), []).append(f"#{i} ({n} bytes, {who})")
    bad = [text + (f", right after allocations {', '.join(ends[addr][:4])}" if addr in ends
                   else "")
           for addr, text in _changed(kept)]
    bad += [f"a tensor allocated after the call, {text}" for _, text in _changed(late)]
    bad += [f"the read-only argument {n}" for n in _mutated(d, case, got, passed)]
    assert not bad, f"{case.id}: live tensors written:\n  " + "\n  ".join(bad)
    missed = [(n, who) for addr, n, who in call_allocs if holes.get(addr) != _round(max(n, 1))]
    assert not missed, f"premise: allocations not between neighbours: {missed[:10]}"
    reused = sum(t.data_ptr() in freed for t, _ in late)
    assert reused, "premise: no tensor allocated after the call reused a block it freed"
    _compare(want, got, f"{case.id} native")
    print(f"{case.id}: {len(call_allocs)} allocations between live neighbours "
          f"(largest {max(n for _, n, _ in call_allocs)} bytes); {reused} tensors allocated "
          f"after the call reused blocks it freed")
