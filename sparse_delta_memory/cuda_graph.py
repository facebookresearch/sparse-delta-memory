# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CUDA-graph execution of the SDM write/read (``SparseDeltaMemoryArgs.cuda_graph``).

A layer replays two graphs, captured once per entry: the forward (batch expand of the initial bank, then
``SDMWriteRead.forward``), captured at the entry's first call, and the backward (``SDMWriteRead.backward``, then the
batch reduction of the bank gradient), captured at its first backward. Projections, top-k and gates stay eager.

Contract:
  * One captured key (shapes, dtypes, op scalars, the TF32 matmul setting) per layer: that of
    its first call that may need a backward. A call with another key runs eager, and so does
    every call of a layer whose capture failed (or whose first backward ran out of memory);
    each case logs one warning per layer. ``reset_cuda_graphs()`` drops the graphs (call it
    between steps) so that the next call captures again. Graphs are never re-captured on
    their own.
  * Inside ``suspend_cuda_graphs()`` or a CUDA-graph capture every call runs eager.
  * A differentiable call saves the same tensors for backward on either path: a token, the
    bank parameter and the inputs. An eager call (``_EagerSaved``) runs its forward again in
    the backward. So under non-reentrant checkpointing the original forward and its
    recompute may take different paths.
  * The graphs, their static buffers and the state of the call they last replayed form an entry. Under non-reentrant
    activation checkpointing (``_region_entries``) the layers of one key in different regions share an entry (each
    backward follows its own recompute); layers of one key in one region take distinct entries ("slots"); a layer
    outside any checkpoint keeps its own.
  * Each forward replay issues a token (a non-reentrant recompute substitutes its own) and makes its call the owner of
    the entry. A backward that does not own its entry first replays its forward from the saved tensors: the same
    result, at the cost of one more forward.
  * Forward graphs share one memory pool per device and capture stream and are captured in call order. Replaying one
    may reuse blocks that hold the state of entries captured after it, so the replay invalidates their tokens; a replay
    on another stream is ordered after the work queued on the pool's stream and before the work queued there next, and
    threads run their calls on the pool's graphs one at a time (``_Pool.lock``). Backward graphs use the pool of their
    forward graph; forward graphs captured after the first backward graph of the shared pool get a private pool.
  * An entry lives while a layer or a pending backward refers to it: until ``reset_cuda_graphs()``, the death of its
    layers, or an out-of-memory first backward, which releases every entry of its pool."""

import contextlib
import itertools
import logging
import threading
import weakref
from collections import Counter

import torch
from torch.autograd.function import once_differentiable
from torch.utils._python_dispatch import _disable_current_modes

from .cache import SDMLayerState
from .memory_ops import (SDMWriteRead, UnsupportedConfigurationError, _layout, expand_memory_batch,
                         refuse_non_native, refuse_quant_tail, sdm_write_read, triton_sum_dim0)
from .phase2_kernels import SNAPSHOT_QUANT_MODES

logger = logging.getLogger(__name__)

_tokens = itertools.count(1)
_pools: dict = {}
_pools_lock = threading.Lock()
# The shared entries by (pool, key, slot), held weakly: by their layers and pending backwards.
_shared: "weakref.WeakValueDictionary" = weakref.WeakValueDictionary()
_shared_lock = threading.Lock()  # every access to _shared (pools insert concurrently)
_runners: "weakref.WeakSet[CudaGraphRunner]" = weakref.WeakSet()
_suspended = 0
# Per thread: checkpoint region pack hook -> the entries its calls used (weak).
_local = threading.local()
# One CUDA-graph capture at a time in the process (torch.cuda.graph), whatever the pool.
_capture_lock = threading.Lock()


def reset_cuda_graphs() -> None:
    """Drop every captured graph, e.g. between steps after a permanent shape change: the next
    call of each layer captures again (also in a layer whose capture failed)."""
    for runner in list(_runners):
        runner.entry = None
        runner.capture_failed = False
    with _pools_lock:
        _pools.clear()
    with _shared_lock:
        _shared.clear()
    _local.__dict__.clear()


@contextlib.contextmanager
def suspend_cuda_graphs():
    """Run the write/read of every layer eager inside the block, e.g. a probe at another
    shape. The backward of those calls may run outside the block."""
    global _suspended
    _suspended += 1
    try:
        yield
    finally:
        _suspended -= 1


def cuda_graphs_suspended() -> bool:
    return _suspended > 0


@contextlib.contextmanager
def _untracked():
    # Selective activation checkpointing caches op outputs in call order, so a recompute must
    # issue the ops of the original forward. The first graphed call also warms up and
    # captures; hiding every op of the runner keeps both sequences empty.
    with _disable_current_modes(), torch.no_grad():
        yield


def _region_entries():
    """The entries used by the graphed calls of the innermost non-reentrant checkpoint region (its saved-tensors pack hook,
    a private torch API), or None outside one or on any error."""
    try:
        hooks = _top_hooks(False)
        pack = hooks[0] if hooks else None
        if not (getattr(pack, "__module__", None) == "torch.utils.checkpoint"
                and str(getattr(pack, "__qualname__", "")).startswith("_checkpoint_hook.")):
            return None
        regions = getattr(_local, "regions", None)
        if regions is None:
            regions = _local.regions = weakref.WeakKeyDictionary()
        used = regions.get(pack)
        if used is None:
            used = regions[pack] = weakref.WeakSet()
        return used
    except Exception:  # noqa: BLE001 - another torch: entries of their own
        return None


_top_hooks = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)


def _capture(fn, pool):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.current_stream()
    try:
        with _capture_lock, _untracked(), torch.cuda.graph(graph, pool=pool,
                                                           capture_error_mode="thread_local"):
            out = fn()
    finally:
        # a failed capture_end() skips torch.cuda.graph's stream restore
        torch.cuda.set_stream(stream)
    return graph, out


def _failure(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:300]


class _Ctx:
    """Stand-in autograd ctx: SDMWriteRead uses only save_for_backward, saved_tensors and
    plain attributes. The runner takes several backwards on one such ctx: the eager warm-up
    of the first backward (whose gradients the first backward returns), the capture right
    after it (which records the kernels without running them), and eager fallbacks that run
    after the call's forward was replayed. SDMWriteRead's second-backward guard therefore
    does not apply to it."""

    repeated_backward_ok = True

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class _Pool:
    def __init__(self, stream):
        self.handle = torch.cuda.graph_pool_handle()
        self.stream = stream  # the capture stream
        self.forward = []  # weak references to the shared forward entries, in capture order
        self.has_backward = False
        self.lock = threading.RLock()  # serializes the threads' calls on the pool's graphs


def _stream_pool(device):
    """The shared pool of ``device`` and its current stream (one stream: replays never overlap)."""
    stream = torch.cuda.current_stream(device)
    with _pools_lock:
        pool = _pools.get((device.index, stream.stream_id))
        if pool is None:
            pool = _pools[(device.index, stream.stream_id)] = _Pool(stream)
    return pool


@contextlib.contextmanager
def _ordered(pool):
    """Runs the block alone among the threads' blocks on ``pool`` and orders it, when it runs
    on another stream than the pool's, after the work queued on the pool's stream and before
    the work queued there next: graphs that share the pool never overlap."""
    with pool.lock:
        current = torch.cuda.current_stream(pool.stream.device)
        if current == pool.stream:
            yield
            return
        current.wait_stream(pool.stream)
        try:
            yield
        finally:
            pool.stream.wait_stream(current)


def _write_read(param, bank_shape, dtype, inputs, scalars, training=False):
    """Readings of the eager op from the initial bank (B copies of ``param``, or zeros), and
    the final bank."""
    if param is None:
        bank = torch.zeros(bank_shape, device=inputs[0].device, dtype=dtype)
    else:
        bank = expand_memory_batch(param, bank_shape[0] // len(param))
    chunk, slots, parallel, quant, *rest = scalars  # rest: (grad_bank_fp32,), part of the graph key
    if training and quant in SNAPSHOT_QUANT_MODES:   # the no-grad op's training refusals (else: the op's ValueError)
        lay = _layout(bank, *inputs, chunk, slots, parallel)   # (SDMWriteRead.forward's checks, in its order)
        rows = lay.P * lay.Nt * lay.C * lay.W
        refuse_quant_tail(dtype, lay.hd, quant, rows, device=bank.device)
        refuse_non_native(bank, lay.C, quant, rows, needs_backward=True, grad_bank_fp32=bool(rest[0]) if rest else False)
    readings = sdm_write_read(bank, *inputs, chunk_size=chunk, num_memory_slots=slots,
                              num_parallel=parallel, snapshot_quant=quant,
                              grad_bank_fp32=bool(rest[0]) if rest else False)
    return readings, bank


def _recomputed_grads(call, param, inputs, grad, needs):
    """Gradients for (param, *inputs) of the readings, by running the forward again with
    autograd; ``needs`` flags the ones to compute."""
    leaves = [None if t is None else t.detach().requires_grad_(n)
              for t, n in zip((param, *inputs), needs)]
    with torch.enable_grad():
        readings, _ = _write_read(leaves[0], call[0], call[1], leaves[1:], call[2])
        want = [t for t in leaves if t is not None and t.requires_grad]
        got = iter(torch.autograd.grad(readings, want, grad, allow_unused=True) if want else ())
    return tuple(next(got) if t is not None and t.requires_grad else None for t in leaves)


class _EagerSaved(torch.autograd.Function):
    """The eager op for a differentiable call that the graphs do not serve. It saves what a
    graphed call saves (a token, the bank parameter and the inputs) and keeps no other state:
    the backward runs the forward again with autograd."""

    @staticmethod
    def forward(ctx, call, param, *inputs):
        with _untracked():
            readings, bank = _write_read(param, call[0], call[1], inputs, call[2], training=True)
        ctx.call = call
        ctx.save_for_backward(torch.tensor(0), param, *inputs)
        ctx.mark_non_differentiable(bank)
        return readings, bank

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _grad_bank):
        _, param, *inputs = ctx.saved_tensors
        return (None, *_recomputed_grads(ctx.call, param, inputs, grad,
                                         ctx.needs_input_grad[1:]))


class _Entry:
    """Static buffers, graphs and state of one captured key, used by the layers of one slot."""

    def __init__(self, runner, key, call, pool, handle, static_param, static_in, ctx, forward,
                 bank, readings):
        self.runners = weakref.WeakSet((runner,))  # the layers that use it
        self.key, self.call, self.pool, self.handle = key, call, pool, handle
        self.static_param, self.static_in, self.ctx = static_param, static_in, ctx
        self.forward, self.bank, self.readings = forward, bank, readings
        self.grad_in = torch.empty_like(self.readings, memory_format=torch.contiguous_format)
        self.backward = self.grad_out = None  # captured on the first backward
        self.backward_eager = False  # set when that capture fails
        self.backward_tf32 = None  # torch.backends.cuda.matmul.allow_tf32 at that capture
        self.index = None  # position in pool.forward when the forward graph is shared
        self.slot = None  # its slot when shared by layers in checkpoint regions (_acquire)
        self.token = None  # the owner: token of the replay whose state is live
        self.released = False

    def release(self):
        """Drops the graphs and static buffers; later backwards of this entry run eager."""
        self.released, self.token = True, None
        self.forward = self.backward = self.ctx = self.bank = self.readings = None
        self.static_param = self.static_in = self.grad_in = self.grad_out = None


_STALE = ("SparseDeltaMemory(cuda_graph=True): the final bank of this replayed call was overwritten "
          "before its state was first read (by a later forward of a layer sharing its CUDA graphs, "
          "of this layer or of a layer captured before it, or by its backward). Read "
          "new_cache.memory right after the call and keep that copy (or a clone), or run the "
          "call under suspend_cuda_graphs().")


class ReplayedState(SDMLayerState):
    """A replayed call's ``new_cache``: the first read of ``memory`` copies the call's final bank (never the static one);
    it raises once a later replay that may write the bank, or the backward, has run."""

    def __init__(self, entry, token, seq_len=0):
        self._entry, self._token, self._copy = weakref.ref(entry), token, None
        self._cache_len = seq_len

    @property
    def memory(self):
        if self._copy is None:
            e = self._entry() if self._entry is not None else None
            if e is None or e.released:
                raise RuntimeError(_STALE)
            with _ordered(e.pool):
                if e.token != self._token or e.released:
                    raise RuntimeError(_STALE)
                self._copy = e.bank.clone()
            self._entry = None
        return self._copy

    @memory.setter
    def memory(self, value):
        self._copy, self._entry = value, None

    def update_(self, new_memory, seq_len=1):
        self.memory = new_memory.detach()  # without reading the replaced bank
        self._cache_len += seq_len
        return self


class CudaGraphRunner:
    """Runs the write/read of one layer from CUDA graphs. ``stats`` counts captures, replays,
    backward-time forward replays (``stale_replay``) and eager fallbacks (``eager_*``)."""

    def __init__(self, name: str = "SparseDeltaMemory"):
        self.name = name
        self.entry = None
        self.capture_failed = False
        self.stats = Counter()
        self._warned = set()
        _runners.add(self)

    def __deepcopy__(self, memo):
        # Graphs cannot be copied; the copy captures its own.
        return CudaGraphRunner(self.name)

    def __reduce__(self):
        # Nor pickled (e.g. torch.save of a whole module); the restored layer captures its own.
        return CudaGraphRunner, (self.name,)

    def note(self, reason: str, message: str) -> None:
        """Counts ``reason`` and logs ``message`` the first time ``reason`` occurs here."""
        self.stats[reason] += 1
        if reason not in self._warned:
            self._warned.add(reason)
            logger.warning("%s(cuda_graph=True): %s", self.name, message)

    def __call__(self, param, bank_shape, dtype, inputs, scalars, seq_len=None):
        """Readings and final bank of ``SDMWriteRead.apply(bank, *inputs, *scalars)`` for a
        call that may need a backward: replayed from the graphs, or ``eager`` (another key
        than the captured one, or a failed capture).

        The bank of shape ``bank_shape`` = [B * S, D] and dtype ``dtype`` starts as B copies
        of ``param`` [S, D], or as zeros when ``param`` is None. A replayed call returns the
        static bank: it holds the final state until the backward rewinds it or a forward
        replay of a layer sharing the entry, or of an entry captured before it, overwrites it.
        With ``seq_len`` (the layer's calls), the second output is the call's state instead:
        a ``ReplayedState`` for a replayed call, which never hands out the static bank.
        """
        if self.capture_failed:
            self.stats["eager_capture_failed"] += 1
            return self._eager_out(seq_len, param, bank_shape, dtype, inputs, scalars)
        key = (None if param is None else (param.shape, param.dtype),
               tuple(bank_shape), dtype,
               tuple((x.shape, x.dtype, x.device) for x in inputs),
               tuple(scalars), torch.backends.cuda.matmul.allow_tf32)
        e = self.entry
        if e is None:
            with _untracked():
                e = self._acquire(key, param, bank_shape, dtype, inputs, scalars)
            if e is None:
                return self._eager_out(seq_len, param, bank_shape, dtype, inputs, scalars)
            self.entry = e
        elif e.key != key:
            self.note("eager_shape",
                      "a call that differs from the captured one (shapes, dtypes, op settings "
                      "or torch.backends.cuda.matmul.allow_tf32) runs eager. After a permanent "
                      "change, call cuda_graph.reset_cuda_graphs() between steps to capture "
                      "the new call.")
            return self._eager_out(seq_len, param, bank_shape, dtype, inputs, scalars)
        elif e.slot is not None:
            used = _region_entries()
            if used is not None:  # the region's later new layers take other slots
                used.add(e)
        with e.pool.lock:
            if not e.released:  # (another thread's out-of-memory backward releases it)
                token = []
                readings = _Graphed.apply(self, e, token, param, *inputs)
                return readings, (e.bank if seq_len is None else
                                  ReplayedState(e, token[0], seq_len))
        self.stats["eager_released"] += 1
        return self._eager_out(seq_len, param, bank_shape, dtype, inputs, scalars)

    @staticmethod
    def eager(param, bank_shape, dtype, inputs, scalars):
        """``__call__`` on the eager path (``_EagerSaved``)."""
        return _EagerSaved.apply((tuple(bank_shape), dtype, tuple(scalars)), param, *inputs)

    def _eager_out(self, seq_len, *call):
        readings, bank = self.eager(*call)
        return readings, (bank if seq_len is None else SDMLayerState(memory=bank, seq_len=seq_len))

    def _acquire(self, key, param, bank_shape, dtype, inputs, scalars):
        """The entry of this layer at its first graphed call: in a checkpoint region, the shared entry of the first slot of
        ``key`` no other call of the region uses; elsewhere its own. None when the capture fails."""
        pool = _stream_pool(inputs[0].device)
        used = _region_entries()
        with pool.lock:
            if used is None:
                return self._capture_forward(key, pool, param, bank_shape, dtype, inputs, scalars)
            slot = 0
            with _shared_lock:
                while (e := _shared.get((pool, key, slot))) is not None and e in used:
                    slot += 1
            if e is None:  # the pool's lock keeps the slot free during the capture
                e = self._capture_forward(key, pool, param, bank_shape, dtype, inputs, scalars)
                if e is None:
                    return None
                e.slot = slot
                with _shared_lock:
                    _shared[(pool, key, slot)] = e
            else:
                e.runners.add(self)
                self.stats["forward_shared"] += 1
            used.add(e)
            return e

    def _capture_forward(self, key, pool, param, bank_shape, dtype, inputs, scalars):
        device = inputs[0].device
        contiguous = torch.contiguous_format

        def failed(exc):
            self.capture_failed = True
            self.note("eager_capture_failed",
                      f"capturing the write/read failed ({_failure(exc)}); this layer runs "
                      "eager until cuda_graph.reset_cuda_graphs().")

        try:
            # The static copies, and an eager call on them that builds extensions, compiles
            # and autotunes kernels and creates lazy state outside the capture. Running out
            # of memory here falls back to eager, which needs less; any other error is the
            # call's own (the eager path raises it too).
            static_param = (None if param is None else
                            param.detach().to(dtype=dtype, memory_format=contiguous, copy=True))
            static_in = tuple(x.detach().clone(memory_format=contiguous) for x in inputs)

            def forward(ctx):
                if static_param is None:
                    bank = torch.zeros(bank_shape, device=device, dtype=dtype)
                else:
                    bank = expand_memory_batch(static_param, bank_shape[0] // len(static_param))
                return bank, SDMWriteRead.forward(ctx, bank, *static_in, *scalars)

            forward(_Ctx())
        except torch.cuda.OutOfMemoryError as exc:
            failed(exc)
            return None
        shared = not pool.has_backward
        handle = pool.handle if shared else torch.cuda.graph_pool_handle()
        ctx = _Ctx()
        try:
            graph, (bank, readings) = _capture(lambda: forward(ctx), handle)
            e = _Entry(self, key, (tuple(bank_shape), dtype, tuple(scalars)), pool, handle,
                       static_param, static_in, ctx, graph, bank, readings)
        except UnsupportedConfigurationError:   # a refusal is the call's own error, never a capture failure
            raise
        except Exception as exc:  # any failure of the capture itself falls back to eager
            failed(exc)
            return None
        if shared:
            e.index = len(pool.forward)
            pool.forward.append(weakref.ref(e))
        self.stats["forward_capture"] += 1
        return e

    def _replay_forward(self, e, param, inputs):
        if param is not None:
            e.static_param.copy_(param)
        for static, x in zip(e.static_in, inputs):
            static.copy_(x)
        e.forward.replay()
        e.token = next(_tokens)
        if e.index is not None:
            for ref in e.pool.forward[e.index + 1:]:
                later = ref()
                if later is not None:
                    later.token = None
        return e.token

    def _backward_body(self, e):
        grads = SDMWriteRead.backward(e.ctx, e.grad_in)
        grad_param = None
        if e.static_param is not None:
            g = grads[0]
            grad_param = triton_sum_dim0(g.view(-1, len(e.static_param), g.shape[1]))
        return (grad_param, *grads[1:1 + len(e.static_in)])

    def _backward(self, e, token, grad, param, inputs, needs):
        abandoned = False
        with _ordered(e.pool):
            if not e.released:
                if e.token != token:
                    self.note("stale_replay",
                              "the graph state of a forward was overwritten before its backward "
                              "(a forward of this layer or of a layer sharing its graphs, or of a "
                              "layer captured before it, ran in between, or the backward runs "
                              "twice), so the backward replays that forward first. Each such "
                              "backward costs one more forward.")
                    with _untracked():
                        self._replay_forward(e, param, inputs)
                e.token = None  # the backward rewinds the bank in place
                e.grad_in.copy_(grad)
                grads = self._backward_graphs(e)
                if grads is not None:
                    # Never hand out graph memory: later replays overwrite it, and autograd
                    # may keep an incoming gradient as .grad.
                    return tuple(None if t is None else t.clone() for t in grads)
                self._abandon(e)
                abandoned = True
        if abandoned:
            torch.cuda.empty_cache()
        else:  # released by an earlier out-of-memory backward
            self.stats["eager_backward"] += 1
        return _recomputed_grads(e.call, param, inputs, grad, needs)

    def _abandon(self, e):
        """The first backward ran out of memory (called under the pool's lock). Drop the graphs
        and buffers of the entry and of every entry sharing its memory pool, so that the pool
        can be freed: the layers that use them run eager from now on (pending backwards from
        their saved inputs). The caller then computes its gradients eagerly."""
        # join this backward's stream before the pool's buffers are freed
        e.pool.stream.wait_stream(torch.cuda.current_stream(e.pool.stream.device))
        entries = [e]
        if e.index is not None:
            entries += [ent for ent in (ref() for ref in e.pool.forward)
                        if ent is not None and ent is not e]
            with _pools_lock:
                for key, pool in list(_pools.items()):
                    if pool is e.pool:
                        del _pools[key]
        with _shared_lock:
            for key, ent in list(_shared.items()):
                if any(ent is x for x in entries):
                    del _shared[key]
        for ent in entries:
            runners = list(ent.runners)
            ent.release()
            for runner in runners:
                runner.capture_failed = True
                if runner.entry is ent:
                    runner.entry = None
                runner.note("eager_capture_failed",
                            "the first backward of this layer, or of a layer that shares its "
                            "graphs or their memory, ran out of memory; this layer runs eager "
                            "until cuda_graph.reset_cuda_graphs().")

    def _backward_graphs(self, e):
        """The backward outputs on the static state, or None when the first backward runs out
        of memory."""
        tf32 = torch.backends.cuda.matmul.allow_tf32
        if e.backward is None and not e.backward_eager:
            # First backward: eager on the static state (the exact result of this call, and
            # the kernels get compiled), then capture.
            try:
                grads = self._backward_body(e)
            except torch.cuda.OutOfMemoryError:
                return None
            try:
                e.backward, e.grad_out = _capture(lambda: self._backward_body(e), e.handle)
            except Exception as exc:  # keep the forward graph; the backward runs eager
                e.backward_eager = True
                self.note("eager_backward",
                          f"capturing the backward failed ({_failure(exc)}); it runs eager.")
            else:
                e.backward_tf32 = tf32
                if e.index is not None:
                    e.pool.has_backward = True
                self.stats["backward_capture"] += 1
        elif e.backward_eager or e.backward_tf32 != tf32:
            # Eager on the static state: the capture failed, or the matmul TF32 setting
            # differs from that of the capture.
            if not e.backward_eager:
                self.note("eager_backward_tf32",
                          "torch.backends.cuda.matmul.allow_tf32 differs from that of the "
                          "backward capture; the backward runs eager.")
            grads = self._backward_body(e)
            self.stats["eager_backward"] += 1
        else:
            e.backward.replay()
            grads = e.grad_out
            self.stats["backward_replay"] += 1
        return grads


class _Graphed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, entry, box, param, *inputs):
        with _untracked(), _ordered(entry.pool):
            token = runner._replay_forward(entry, param, inputs)
            readings = entry.readings.clone()
        box.append(token)  # the caller's ReplayedState
        runner.stats["forward_replay"] += 1
        ctx.runner, ctx.entry = runner, entry
        # The token is a tensor, so that a non-reentrant checkpoint recompute substitutes its
        # own. The bank parameter and inputs let a backward whose state was overwritten
        # replay this forward.
        ctx.save_for_backward(torch.tensor(token), param, *inputs)
        return readings

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        token, param, *inputs = ctx.saved_tensors  # unpacking first runs any pending recompute
        return (None, None, None, *ctx.runner._backward(ctx.entry, token.item(), grad, param,
                                                        inputs, ctx.needs_input_grad[3:]))
