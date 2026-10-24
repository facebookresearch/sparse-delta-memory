# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The CUDA-graph option of the SDM layer (``SparseDeltaMemoryArgs.cuda_graph``, implemented
by ``sparse_delta_memory.cuda_graph``).

The kernels are deterministic (test_determinism.py) and a replay runs the eager kernels on
the same inputs, so every comparison with eager is bitwise. The op-level tests check two
layers sharing the graph pool under every checkpointing schedule (on slots that no two
tokens share, and they still check their premise that eager is bitwise reproducible).

The model-level tests compare a model of small blocks (a bias-free Linear, so selective
activation checkpointing sees aten.mm, then an SDM layer with ``cuda_graph=True``) with a
deep-copied eager twin, bitwise over every output and gradient (``check_steps``). The one
exception re-runs a backward eagerly under another TF32 setting, on buffers at other
addresses where cuBLAS may pick other kernels: it compares at TF32 precision.

Calls the graphs cannot serve run eager (another shape, a call from a cache, a failed capture,
``suspend_cuda_graphs()``), and a backward whose graph state was overwritten replays its
forward first; the tests check that each of these matches eager, bitwise at op level.
"""
import contextlib
import copy
import functools
import io
import logging
import weakref

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import (
    CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts)

from sdm_testlib import (
    assert_close_across, assert_runs_bitwise, bits_equal, fixed_slot_scores, make_op_inputs,
    rel_rms, requires_cuda)

from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs
from sparse_delta_memory.cache import SDMLayerState
from sparse_delta_memory import cuda_graph
from sparse_delta_memory.cuda_graph import (
    CudaGraphRunner, reset_cuda_graphs, suspend_cuda_graphs)
from sparse_delta_memory.memory_ops import SDMWriteRead, expand_memory_batch
from sparse_delta_memory.product_key_topk import product_key_topk

pytestmark = requires_cuda

DIM = 256
K64 = dict(num_heads=2, slots_per_head=4096, num_writes=64, num_reads=64,
           chunk_size=64, output_gate=True, normalize_readings=True)
GEN = dict(num_heads=2, slots_per_head=1024, num_writes=16, num_reads=16, chunk_size=64)
FP32 = dict(num_heads=4, slots_per_head=256, num_writes=8, num_reads=8, chunk_size=32)
# A backward re-run eagerly under TF32 against eager TF32 steps (relative RMS per tensor,
# relative max for tensors of at most TINY_NUMEL elements).
TF32_GATE = dict(rms=2e-4, tiny_max=1e-5)
EAGER_RUNS = 2
F8 = dict(num_heads=2, slots_per_head=57600, product_key_factor_dims=[5, 5, 4, 4, 4, 4, 3, 3],
          num_writes=64, num_reads=48, chunk_size=64, output_gate=True)
F1 = dict(num_heads=4, slots_per_head=1024, product_key_factor_dims=[1024], num_writes=8,
          num_reads=12, chunk_size=32)


class Block(nn.Module):
    def __init__(self, sdm):
        super().__init__()
        self.lin = nn.Linear(sdm.dim, sdm.dim, bias=False)
        self.sdm = sdm

    def forward(self, x):
        h = x + self.lin(x)
        return h + self.sdm(h)[0]


class Model(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.use_reentrant = None  # per-block checkpoint: None (off), False or True

    def forward(self, x):
        for b in self.blocks:
            x = b(x) if self.use_reentrant is None else checkpoint(
                b, x, use_reentrant=self.use_reentrant)
        return x


def lingua_sac(model):
    """Whole-model selective activation checkpointing, as lingua's parallelize_model."""
    pytest.importorskip("lingua.distributed", reason="needs the repository checkout (apps/, lingua/)")
    from lingua.distributed import get_default_policy

    return checkpoint_wrapper(model, context_fn=functools.partial(
        create_selective_checkpoint_contexts, get_default_policy()))


def per_layer(use_reentrant):
    def wrap(model):
        model.use_reentrant = use_reentrant
        return model
    return wrap


class Autocast(nn.Module):
    """Runs ``model`` under bf16 autocast; the backward runs outside it (the AMP recipe)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(x)


def graphed_sdm(cfg, layer_id=0):
    torch.manual_seed(100 + layer_id)
    sdm = SparseDeltaMemory(SparseDeltaMemoryArgs(dim=DIM, cuda_graph=True, **cfg), layer_id)
    sdm.init_weights()
    return sdm


def eager_twin(model):
    twin = copy.deepcopy(model)
    for m in twin.modules():
        if isinstance(m, SparseDeltaMemory):
            m.graph_runner = None
    return twin


def make_pair(cfgs, dtype, wrap=lambda m: m):
    """(eager twin, graphed model, unwrapped graphed model) with identical parameters."""
    graph = Model([Block(graphed_sdm(cfg, i)) for i, cfg in enumerate(cfgs)]).to("cuda", dtype)
    return wrap(eager_twin(graph)), wrap(graph), graph


def inputs_dim(B, T, dim, seed, dtype):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, T, dim, device="cuda", generator=gen)
    gy = torch.randn(B, T, dim, device="cuda", generator=gen) * 0.01
    return x.to(dtype), gy.to(dtype)


def inputs(B, T, seed, dtype):
    return inputs_dim(B, T, DIM, seed, dtype)


def full(t):
    """A copy of ``t``, gathered when it is a sharded DTensor."""
    return (t.full_tensor() if hasattr(t, "full_tensor") else t).detach().clone()


def train_step(model, x, gy, accumulate=False, layer=False):
    """Output, input grad and every parameter grad of one forward + backward (layer: a bare layer, its output)."""
    if not accumulate:
        for p in model.parameters():
            p.grad = None
    xi = x.detach().clone().requires_grad_(True)
    out = model(xi)[0] if layer else model(xi)
    out.backward(gy)
    res = {"out": out.detach().clone(), "grad_x": xi.grad.clone()}
    res.update({"grad." + n: full(p.grad) for n, p in model.named_parameters()})
    return res


def check_steps(eager, graph, batches, dtype, label, graph_runs=2, step=train_step,
                eager_step=None):
    """Every step of ``graph`` equals the steps of ``eager`` (run by ``eager_step``,
    default ``step``) bitwise; dtype "tf32" compares at TF32 precision (TF32_GATE).

    Returns the first graph run of each batch."""
    firsts = []
    for i, batch in enumerate(batches):
        e = [(eager_step or step)(eager, *batch) for _ in range(EAGER_RUNS)]
        g = [step(graph, *batch) for _ in range(graph_runs)]
        if dtype == "tf32":
            assert_close_across(e, g, label=f"{label} {i}", **TF32_GATE)
        else:
            assert_runs_bitwise(e + g, label=f"{label} {i}")
        firsts.append(g[0])
    return firsts


def stats(*models):
    return [dict(m.graph_runner.stats) for model in models for m in model.modules()
            if isinstance(m, SparseDeltaMemory)]


@pytest.fixture(autouse=True)
def fresh_graphs(ieee_fp32):
    reset_cuda_graphs()
    yield
    reset_cuda_graphs()


# (label, dtype, streams, T, W, head_dim, slots per stream, chunk)
OP_CASES = [("fp32_hd64_w8_c32", torch.float32, 4, 128, 8, 64, 4096, 32),
            ("bf16_hd128_w64_c64", torch.bfloat16, 4, 128, 64, 128, 65536, 64)]
# Op calls per step for layers A and B: "fwd X" keeps the output of X, "bwd X" backpropagates
# the latest one. A recompute replaces the original output, which is never backpropagated.
# In "late_capture", B is first captured after a backward graph exists, and the backward of A
# replays while the state of B is live.
OP_SCHEDULES = {
    "no_checkpoint": ["fwd A", "fwd B", "bwd B", "bwd A"],
    "whole_model_recompute": ["fwd A", "fwd B", "fwd A", "fwd B", "bwd B", "bwd A"],
    "per_layer_recompute": ["fwd A", "fwd B", "fwd B", "bwd B", "fwd A", "bwd A"],
    "late_capture": ["fwd A", "bwd A", "fwd B", "fwd A", "bwd A", "bwd B"],
}


def collision_free_inputs(P, T, W, dim, sph, dtype, seed):
    """Op inputs whose write slots, and separately read slots, are distinct across all tokens
    of a stream: no two tokens meet in an atomic accumulation."""
    d = make_op_inputs(P, T, W, W, dim, sph, dtype, seed=seed, weights="softmax")
    gen = torch.Generator().manual_seed(seed)
    for name in ("k_idx", "q_idx"):
        d[name] = torch.stack([
            torch.randperm(sph, generator=gen)[:T * W].view(T, W).sort(-1).values + p * sph
            for p in range(P)]).cuda()
    return d


def op_call(runner, d, C, sph, batch=2):
    """One op call on ``batch`` copies of the bank (eager when ``runner`` is None): readings,
    the bank the op advances in place, a copy of its final state, bank parameter, inputs."""
    P, dim, dtype = d["k_idx"].shape[0], d["v"].shape[-1], d["v"].dtype
    param = d["memory"][:P // batch * sph].clone().requires_grad_()
    names = ("k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")
    ins = [d[n].clone().requires_grad_(d[n].is_floating_point()) for n in names]
    scalars = (C, sph, P, "none", False)  # chunk, slots, streams, snapshot_quant, grad_bank_fp32
    if runner is None:
        bank = expand_memory_batch(param, batch)
        readings = SDMWriteRead.apply(bank, *ins, *scalars)
    else:
        readings, bank = runner(param, (P * sph, dim), dtype, ins, scalars)
    return readings, bank, bank.detach().clone(), param, ins


def op_backward(readings, bank, final_bank, param, ins, gy, retain_graph=False):
    """Readings, final and rewound bank, and the (accumulated) gradients of one call."""
    readings.backward(gy, retain_graph=retain_graph)
    grads = [param.grad] + [x.grad for x in ins if x.is_floating_point()]
    return [readings.detach(), final_bank, bank.detach().clone()] + [g.clone() for g in grads]


@pytest.mark.parametrize("schedule", OP_SCHEDULES)
@pytest.mark.parametrize("label,dtype,P,T,W,dim,sph,C", OP_CASES, ids=[c[0] for c in OP_CASES])
def test_graph_is_bitwise_eager_without_collisions(label, dtype, P, T, W, dim, sph, C, schedule):
    runners = {X: CudaGraphRunner() for X in "AB"}
    for step in range(3):
        data = {X: collision_free_inputs(P, T, W, dim, sph, dtype, seed=10 * step + i)
                for i, X in enumerate("AB")}
        gy = torch.randn(P * T, dim, device="cuda").to(dtype)
        eager, again = ({X: op_backward(*op_call(None, d, C, sph), gy) for X, d in data.items()}
                        for _ in range(2))
        assert all(bits_equal(a, b) for X in data for a, b in zip(eager[X], again[X])), \
            "premise: eager is not bitwise deterministic on these inputs"
        live = {}
        for op in OP_SCHEDULES[schedule]:
            kind, X = op.split()
            if kind == "fwd":
                live[X] = op_call(runners[X], data[X], C, sph)
            else:
                got = op_backward(*live.pop(X), gy)
                assert all(bits_equal(a, b) for a, b in zip(eager[X], got)), f"{op} step {step}"
    for runner in runners.values():
        assert runner.stats["forward_capture"] == runner.stats["backward_capture"] == 1


# Schedules run after a first step that captures A, then B. "A2" is a second call of layer A
# on other inputs; "bwd X retain" keeps the autograd graph for a second backward of X, whose
# gradients accumulate. Each backward that finds its state overwritten replays its forward:
# (schedule, replays per step and layer).
STALE_SCHEDULES = {
    "two_forwards": (["fwd A", "fwd A2", "bwd A2", "bwd A"], {"A": 1}),
    "two_forwards_fifo": (["fwd A", "fwd A2", "bwd A", "bwd A2"], {"A": 2}),
    "earlier_layer_replays": (["fwd B", "fwd A", "bwd B", "bwd A"], {"B": 1}),
    "second_backward": (["fwd A", "bwd A retain", "bwd A"], {"A": 1}),
}


@pytest.mark.parametrize("schedule", STALE_SCHEDULES)
@pytest.mark.parametrize("label,dtype,P,T,W,dim,sph,C", OP_CASES, ids=[c[0] for c in OP_CASES])
def test_overwritten_state_replays_forward_bitwise(label, dtype, P, T, W, dim, sph, C, schedule):
    """A replay of A may reuse the pool blocks that hold the state of B (captured after A),
    a second forward of A overwrites the state of the first, and a backward rewinds the bank:
    each later backward of an overwritten call replays its forward first, bitwise eager."""
    ops, replays = STALE_SCHEDULES[schedule]
    runners = {X: CudaGraphRunner() for X in "AB"}
    steps = 3
    for step in range(steps):
        data = {n: collision_free_inputs(P, T, W, dim, sph, dtype, seed=10 * step + i)
                for i, n in enumerate(("A", "A2", "B"))}
        gy = torch.randn(P * T, dim, device="cuda").to(dtype)
        eager = {n: op_backward(*op_call(None, d, C, sph), gy) for n, d in data.items()}
        live, backwards = {}, {}
        for op in OP_SCHEDULES["no_checkpoint"] if step == 0 else ops:
            kind, name, *retain = op.split()
            if kind == "fwd":
                live[name] = op_call(runners[name[0]], data[name], C, sph)
                continue
            got = op_backward(*live[name], gy, retain_graph=bool(retain))
            backwards[name] = backwards.get(name, 0) + 1
            n = backwards[name]  # gradients accumulate over the backwards of one call
            want = eager[name][:3] + [n * g for g in eager[name][3:]]
            assert all(bits_equal(a, b) for a, b in zip(want, got)), f"{op} step {step}"
    for X, runner in runners.items():
        assert runner.stats["forward_capture"] == runner.stats["backward_capture"] == 1
        assert runner.stats["stale_replay"] == (steps - 1) * replays.get(X, 0)


CASES = [
    ("k64_bf16", [K64, K64], torch.bfloat16, 256),
    ("generic_w16", [GEN, GEN], torch.bfloat16, 256),
    ("fp32_hd64_c32", [FP32, FP32], torch.float32, 128),
    ("kwd_fp8", [dict(GEN, key_weighted_decay=True, snapshot_quant="fp8")], torch.bfloat16, 256),
    ("int4", [dict(GEN, snapshot_quant="int4")], torch.bfloat16, 256),
    ("fp32_kwd_int4", [dict(FP32, key_weighted_decay=True, snapshot_quant="int4")],
     torch.float32, 128),
    ("fp32_bf16snap", [dict(FP32, snapshot_quant="bf16")], torch.float32, 128),
    ("no_learned_bank", [dict(GEN, backprop_on_memory=False)] * 2, torch.bfloat16, 256),
    ("ragged_t", [K64, GEN], torch.bfloat16, 200),
    ("f8_bf16", [F8, F8], torch.bfloat16, 256),
    ("f1_fp32", [F1, F1], torch.float32, 128),
]


@pytest.mark.parametrize("label,cfgs,dtype,T", CASES, ids=[c[0] for c in CASES])
def test_graph_matches_eager(label, cfgs, dtype, T):
    eager, graph, _ = make_pair(cfgs, dtype)
    firsts = check_steps(eager, graph, [inputs(2, T, 10 + s, dtype) for s in range(3)],
                         dtype, label)
    for a, b in zip(firsts, firsts[1:]):
        assert rel_rms(a["out"], b["out"]) > 0.5, "a replay returned a stale output"
    # 3 batches x 2 graph runs: 1 capture + 6 forward replays, 1 capture + 5 backward replays.
    expect = dict(forward_capture=1, forward_replay=6, backward_capture=1, backward_replay=5)
    assert stats(graph) == [expect] * len(cfgs)


# The flagship layer geometry (2 heads of 1024, 64 writes and reads, chunk 64, a learned bank).
DEEP = dict(dim=2048, num_heads=2, slots_per_head=16384, num_writes=64, num_reads=64,
            chunk_size=64, output_gate=True, normalize_readings=True)


def perturb_and_copy(src, dst, step):
    """Changes every weight of ``src`` in place (by 1% of its RMS) and copies it in place into
    ``dst``, whose graphs captured those storages."""
    gen = torch.Generator(device="cuda").manual_seed(1000 + step)
    with torch.no_grad():
        for p, q in zip(src.parameters(), dst.parameters()):
            rms = p.float().pow(2).mean().sqrt().clamp_min(1e-3)
            p.add_((torch.randn(p.shape, device="cuda", generator=gen) * 0.01 * rms).to(p.dtype))
            q.copy_(p)


class FrozenSelections:
    """Replaces the product-key top-k of the SDM layers of ``models`` with the slots that the
    first forward after ``new_batch()`` selected (a layer's writes, then its reads), keeping
    the weights differentiable functions of the projections: every run then reads and writes
    the same slots."""

    def __init__(self, *models):
        self.idx, self.calls = {}, {}
        for m in models:
            for layer in m.modules():
                if isinstance(layer, SparseDeltaMemory):
                    layer._product_keys = functools.partial(self.select, layer)

    def new_batch(self):
        self.idx.clear()

    def select(self, layer, x, k):
        n = self.calls[layer.layer_id] = self.calls.get(layer.layer_id, -1) + 1
        side = layer.key_size_per_head // 2
        key = (layer.layer_id, n % 2)
        if key not in self.idx:
            self.idx[key] = product_key_topk(x, (side, side), k)[1]
        idx = self.idx[key]
        return fixed_slot_scores(x, idx, (side, side)), idx


# Native selections in bf16, frozen ones in both dtypes.
DEEP_CASES = [("fp32_frozen", torch.float32, "frozen"), ("bf16_frozen", torch.bfloat16, "frozen"),
              ("bf16_native", torch.bfloat16, "native")]


@pytest.mark.parametrize("label,dtype,selections", DEEP_CASES, ids=[c[0] for c in DEEP_CASES])
def test_deep_stack_matches_eager_over_steps(label, dtype, selections):
    """Three graphed layers in a row at B = 1, over several steps whose weights change in
    place in between: every output and gradient equals eager bitwise. With native
    selections each layer's exact top-k reads the previous layer's output, so a last-bit
    difference anywhere would flip slots downstream; frozen selections reuse the first
    forward's slots."""
    layers = []
    for i in range(3):
        torch.manual_seed(200 + i)
        layer = SparseDeltaMemory(SparseDeltaMemoryArgs(cuda_graph=True, **DEEP), i)
        layer.init_weights()
        layers.append(Block(layer))
    graph = Model(layers).to("cuda", dtype)
    eager = eager_twin(graph)
    frozen = FrozenSelections(eager, graph) if selections == "frozen" else None
    for s in range(3):
        if frozen:
            frozen.new_batch()
        check_steps(eager, graph, [inputs_dim(1, 1024, DEEP["dim"], 60 + s, dtype)], dtype,
                    f"deep {selections} step {s}")
        perturb_and_copy(eager, graph, s)
    expect = dict(forward_capture=1, forward_replay=6, backward_capture=1, backward_replay=5)
    assert stats(graph) == [expect] * 3


def test_autocast_forward():
    """fp32 parameters, forward under bf16 autocast and backward outside it: the op gets
    bf16 activations with an fp32 bank, and the graphs match eager."""
    eager, graph, inner = make_pair([K64, GEN], torch.float32, Autocast)
    check_steps(eager, graph, [inputs(2, 256, 25 + s, torch.float32) for s in range(2)],
                torch.bfloat16, "autocast")
    expect = dict(forward_capture=1, forward_replay=4, backward_capture=1, backward_replay=3)
    assert stats(inner) == [expect] * 2


SCHEDULES = [("no_checkpoint", lambda m: m, 1), ("lingua_sac", lingua_sac, 2),
             ("per_layer_checkpoint", per_layer(False), 2),
             ("reentrant_checkpoint", per_layer(True), 1)]
MODELS = [("bf16", [K64, K64, GEN], torch.bfloat16, 256),
          ("fp32", [FP32, FP32], torch.float32, 128)]


@pytest.mark.parametrize("model_label,cfgs,dtype,T", MODELS, ids=[m[0] for m in MODELS])
@pytest.mark.parametrize("label,wrap,forwards", SCHEDULES, ids=[s[0] for s in SCHEDULES])
def test_activation_checkpointing(label, wrap, forwards, model_label, cfgs, dtype, T):
    """Per step, a layer replays its backward once and its forward on every call that may
    need a backward: the forward and the recompute of a non-reentrant checkpoint, only the
    recompute of a reentrant one (its first forward runs without grad, eager). Under per-layer
    non-reentrant checkpointing the layers of one shape share one entry: one capture each."""
    eager, graph, inner = make_pair(cfgs, dtype, wrap)
    check_steps(eager, graph, [inputs(2, T, 20 + s, dtype) for s in range(2)], dtype,
                f"{label} {model_label}")
    steps = 4
    expect = dict(forward_capture=1, forward_replay=forwards * steps, backward_capture=1,
                  backward_replay=steps - 1)
    if label != "per_layer_checkpoint":
        assert stats(inner) == [expect] * len(cfgs)
        return
    for group in entry_groups(inner):
        want = dict(forward_capture=1, forward_shared=len(group) - 1, forward_replay=forwards *
                    steps * len(group), backward_capture=1, backward_replay=steps * len(group) - 1)
        assert entry_totals(group) == {k: v for k, v in want.items() if v}
    assert len(entry_groups(inner)) == len({tuple(sorted(c.items())) for c in cfgs})


def sdm_layers(model):
    return [m for m in model.modules() if isinstance(m, SparseDeltaMemory)]


def entry_groups(model):
    """The stats of the SDM layers of ``model``, grouped by the entry they replay."""
    groups = {}
    for m in sdm_layers(model):
        groups.setdefault(id(m.graph_runner.entry), []).append(dict(m.graph_runner.stats))
    return list(groups.values())


def entry_totals(group):
    total = {}
    for st in group:
        for k, v in st.items():
            total[k] = total.get(k, 0) + v
    return total


def _forward_with_mm(forward):
    def patched(ctx, *args):
        readings = forward(ctx, *args)
        eye = torch.eye(readings.shape[-1], device=readings.device, dtype=readings.dtype)
        return readings + 0 * torch.mm(readings, eye)
    return staticmethod(patched)


def test_sac_with_mm_inside_the_op(monkeypatch):
    """An op issuing aten.mm (saved by the SAC policy) still matches eager under SAC.

    Control: without disabling the dispatch modes during capture, the captured mm shifts the
    SAC cache and the capture step's gradients are wrong."""
    monkeypatch.setattr(SDMWriteRead, "forward", _forward_with_mm(SDMWriteRead.forward))
    batches = [inputs(2, 128, 30 + s, torch.float32) for s in range(2)]
    eager, graph, _ = make_pair([FP32, FP32], torch.float32, lingua_sac)
    check_steps(eager, graph, batches, torch.float32, "mm inside the op")

    reset_cuda_graphs()
    monkeypatch.setattr(cuda_graph, "_disable_current_modes", contextlib.nullcontext)
    eager, graph, _ = make_pair([FP32, FP32], torch.float32, lingua_sac)
    e, g = train_step(eager, *batches[0]), train_step(graph, *batches[0])
    assert max(rel_rms(e[k], g[k]) for k in e) > 0.1, "control: SAC misalignment not detected"


def test_gradient_accumulation():
    eager, graph, _ = make_pair([K64, GEN], torch.bfloat16, lingua_sac)

    def two_microbatches(model, a, b):
        train_step(model, *a)
        return train_step(model, *b, accumulate=True)

    batches = [tuple(inputs(2, 256, 40 + 2 * s + j, torch.bfloat16) for j in range(2))
               for s in range(2)]
    check_steps(eager, graph, batches, torch.bfloat16, "accumulation", step=two_microbatches)


def _leaf(x):
    return x.detach().clone().requires_grad_(True)


def _results(models, outs, leaves):
    for m in models:
        assert all(p.grad is not None for p in m.parameters())
    res = {f"out{i}": o.detach().clone() for i, o in enumerate(outs)}
    res.update({f"grad_x{i}": x.grad.clone() for i, x in enumerate(leaves)})
    res.update({f"grad{k}.{n}": full(p.grad) for k, m in enumerate(models)
                for n, p in m.named_parameters()})
    return res


def _zero(models):
    for m in models:
        m.zero_grad(set_to_none=True)


def one_step(models, x, gy):
    (m,) = models
    return train_step(m, x, gy)


def two_forwards(models, x, gy):
    """Two forwards before one backward: two inputs through one model, or one input through
    each of two models, the second model first."""
    _zero(models)
    leaves = [_leaf(x), _leaf(x.flip(1))]
    if len(models) == 1:
        outs = [models[0](xi) for xi in leaves]
    else:
        outs = [models[1](leaves[1]), models[0](leaves[0])]
    torch.autograd.backward(outs, [gy, gy.flip(1)])
    return _results(models, outs, leaves)


def two_backwards(models, x, gy):
    """A second backward of the same graph (retain_graph); gradients accumulate."""
    _zero(models)
    (m,), xi = models, _leaf(x)
    out = m(xi)
    out.backward(gy, retain_graph=True)
    out.backward(gy)
    return _results(models, [out], [xi])


def two_fresh_backwards(models, x, gy):
    """The eager reference of two_backwards: two forwards and backwards of the same input
    accumulating into the same leaves (a second backward of one eager forward would start
    from the bank its first backward restored)."""
    _zero(models)
    (m,), xi = models, _leaf(x)
    out = m(xi)
    out.backward(gy)
    m(xi).backward(gy)
    return _results(models, [out], [xi])


def one_model(cfgs):
    return lambda dtype: (Model([Block(graphed_sdm(c, i)) for i, c in enumerate(cfgs)])
                          .to("cuda", dtype),)


def shared_layer(cfg):
    def build(dtype):
        block = Block(graphed_sdm(cfg))
        return (Model([block, block]).to("cuda", dtype),)  # one SDM layer, called twice
    return build


def two_models(cfgs):
    """Two graphed models; the first captures its layers first."""
    def build(dtype):
        models = tuple(Model([Block(graphed_sdm(c, 10 * k + i)) for i, c in enumerate(cfgs)])
                       .to("cuda", dtype) for k in range(2))
        x, gy = inputs(2, 256, 69, dtype)
        outs = [m(_leaf(x)) for m in models]
        torch.autograd.backward(outs, [gy, gy])
        return models
    return build


# (label, build, graph step, eager step, forward replays at backward time per graph step)
STALE_CASES = [
    ("two_outstanding_forwards", one_model([K64, GEN]), two_forwards, None, 2),
    ("weight_shared_layer", shared_layer(GEN), one_step, None, 1),
    ("interleaved_models", two_models([GEN]), two_forwards, None, 1),
    ("second_backward", one_model([K64, GEN]), two_backwards, two_fresh_backwards, 2),
]


@pytest.mark.parametrize("label,build,step,eager_step,replays", STALE_CASES,
                         ids=[c[0] for c in STALE_CASES])
def test_overwritten_state_replays_forward(label, build, step, eager_step, replays):
    """Schedules whose backward finds the graph state of its call overwritten (two outstanding
    forwards, a layer called twice per forward, a forward of another graphed model in between,
    a second backward) replay that forward first and match eager."""
    graph = build(torch.bfloat16)
    eager = tuple(eager_twin(m) for m in graph)
    before = sum(st.get("stale_replay", 0) for st in stats(*graph))
    batches = [inputs(2, 256, 70 + s, torch.bfloat16) for s in range(2)]
    check_steps(eager, graph, batches, torch.bfloat16, label, step=step, eager_step=eager_step)
    after = stats(*graph)
    assert sum(st.get("stale_replay", 0) for st in after) - before == replays * 2 * len(batches)
    assert not any(k.startswith("eager_") for st in after for k in st)


def _warnings(caplog, text):
    return [r for r in caplog.records if r.name == cuda_graph.__name__ and text in r.getMessage()]


def test_other_shape_runs_eager(caplog):
    """One call per layer is captured. A call of another shape runs eager, with one warning
    per layer, and the captured shape keeps replaying; reset_cuda_graphs() captures anew."""
    eager, graph, _ = make_pair([GEN, GEN], torch.bfloat16)
    check_steps(eager, graph, [inputs(2, 256, 50, torch.bfloat16)], torch.bfloat16, "T=256")
    with caplog.at_level(logging.WARNING, logger=cuda_graph.__name__):
        check_steps(eager, graph, [inputs(2, 128, 51 + s, torch.bfloat16) for s in range(2)],
                    torch.bfloat16, "T=128 eager")
    assert len(_warnings(caplog, "differs from the captured one")) == 2
    check_steps(eager, graph, [inputs(2, 256, 53, torch.bfloat16)], torch.bfloat16,
                "T=256 again")
    expect = dict(forward_capture=1, forward_replay=4, backward_capture=1, backward_replay=3,
                  eager_shape=4)
    assert stats(graph) == [expect] * 2
    reset_cuda_graphs()
    check_steps(eager, graph, [inputs(2, 128, 52, torch.bfloat16)], torch.bfloat16,
                "T=128 after reset")
    assert [st["forward_capture"] for st in stats(graph)] == [2, 2]


def cache_step(layer, x, gy):
    """A differentiable call from a fresh cache (whose bank is a copy of the learned one)."""
    for p in layer.parameters():
        p.grad = None
    cache = layer.create_kv_cache(bsz=x.shape[0], seq_len=0, dtype=x.dtype, device="cuda")
    xi = _leaf(x)
    out, _ = layer(xi, cache=cache)
    out.backward(gy)
    res = {"out": out.detach().clone(), "grad_x": xi.grad.clone()}
    res.update({"grad." + n: p.grad.clone() for n, p in layer.named_parameters()})
    return res


def test_call_with_a_cache_runs_eager(caplog):
    """The graphs start from the layer's own initial bank: a differentiable call from a cache
    runs eager, with one warning, and matches the layer without graphs."""
    layer = graphed_sdm(GEN).to("cuda", torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger=cuda_graph.__name__):
        check_steps(eager_twin(layer), layer,
                    [inputs(2, 256, 97 + s, torch.bfloat16) for s in range(2)],
                    torch.bfloat16, "cache", step=cache_step)
    assert len(_warnings(caplog, "starts from a cache")) == 1
    assert dict(layer.graph_runner.stats) == {"eager_cache": 4}


def _synchronizing(capture):
    """``capture`` of a function that first synchronizes the device, which a capture forbids."""
    return lambda fn, pool: capture(lambda: (torch.cuda.synchronize(), fn())[1], pool)


def _out_of_memory(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("injected")


@pytest.mark.parametrize("phase", ["warmup_oom", "forward", "backward"])
def test_failed_capture_runs_eager(monkeypatch, caplog, phase):
    """A capture that fails leaves its call eager: after a failed forward capture (or an
    out-of-memory error in its warmup on the static copies), every call of the layer until
    reset_cuda_graphs(); after a failed backward capture, the backward."""
    real, captures = cuda_graph._capture, []

    def capture(fn, pool):
        captures.append(fn)
        failing = (len(captures) == 1) == (phase == "forward")
        return (_synchronizing(real) if failing else real)(fn, pool)

    if phase == "warmup_oom":  # the warmup of the graph path only
        monkeypatch.setattr(cuda_graph._Ctx, "save_for_backward", _out_of_memory)
    else:
        monkeypatch.setattr(cuda_graph, "_capture", capture)
    eager, graph, _ = make_pair([GEN], torch.bfloat16)
    batches = [inputs(2, 256, 80 + s, torch.bfloat16) for s in range(2)]
    with caplog.at_level(logging.WARNING, logger=cuda_graph.__name__):
        check_steps(eager, graph, batches, torch.bfloat16, f"failed {phase} capture")
    assert torch.cuda.current_stream() == torch.cuda.default_stream(), "left on the capture stream"
    assert len(_warnings(caplog, "failed")) == 1
    if phase != "backward":
        assert stats(graph) == [{"eager_capture_failed": 4}]
        monkeypatch.undo()
        reset_cuda_graphs()
        check_steps(eager, graph, batches[:1], torch.bfloat16, "after reset")
        assert stats(graph)[0]["forward_capture"] == 1
    else:
        assert stats(graph) == [dict(forward_capture=1, forward_replay=4, eager_backward=4)]


def test_first_backward_out_of_memory_runs_eager(monkeypatch, caplog):
    """When the first backward on the static state runs out of memory, the layer drops its
    graphs, computes that backward eagerly from the saved inputs, and runs eager until
    reset_cuda_graphs()."""
    real, calls = cuda_graph.CudaGraphRunner._backward_body, []

    def backward_body(self, e):
        calls.append(e)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("injected")
        return real(self, e)

    monkeypatch.setattr(cuda_graph.CudaGraphRunner, "_backward_body", backward_body)
    eager, graph, _ = make_pair([GEN, GEN], torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger=cuda_graph.__name__):
        check_steps(eager, graph, [inputs(2, 256, 83 + s, torch.bfloat16) for s in range(2)],
                    torch.bfloat16, "first backward out of memory")
    # The last layer runs its backward first. Both layers share one forward graph pool, so
    # both drop their graphs; the first layer's pending backward runs from its saved inputs.
    assert len(_warnings(caplog, "ran out of memory")) == 2
    assert stats(graph) == [dict(forward_capture=1, forward_replay=1, eager_backward=1,
                                 eager_capture_failed=4),
                            dict(forward_capture=1, forward_replay=1, eager_capture_failed=4)]


# Under non-reentrant checkpointing, a forward and its recompute that take different paths:
# the forward inside suspend_cuda_graphs() and the recompute outside, or the other way round.
# The backward runs the node of the original forward on the tensors the recompute saved.
def _checkpointed(model, x, gy, suspend_forward):
    model.zero_grad(set_to_none=True)
    xi = _leaf(x)
    with suspend_cuda_graphs() if suspend_forward else contextlib.nullcontext():
        out = checkpoint(model, xi, use_reentrant=False)
    with contextlib.nullcontext() if suspend_forward else suspend_cuda_graphs():
        out.backward(gy)
    return _results((model,), [out], [xi])


RECOMPUTE_CASES = [
    # The eager node recomputes its forward: the recompute's replay is not used.
    ("suspended_forward", True,
     dict(forward_capture=1, forward_replay=3, backward_capture=1)),
    # The graphed node gets the eager recompute's token, so it replays its forward first.
    ("suspended_recompute", False,
     dict(forward_capture=1, forward_replay=3, backward_capture=1, stale_replay=2,
          backward_replay=2)),
]


@pytest.mark.parametrize("label,suspend_forward,expect", RECOMPUTE_CASES,
                         ids=[c[0] for c in RECOMPUTE_CASES])
def test_checkpoint_recompute_may_take_the_other_path(label, suspend_forward, expect):
    """A graphed call and an eager one save the same tensors, so a non-reentrant checkpoint
    accepts a recompute that takes the other path, and the gradients match eager."""
    eager, graph, _ = make_pair([GEN], torch.float32)
    train_step(graph, *inputs(2, 256, 92, torch.float32))  # captures
    check_steps(eager, graph, [inputs(2, 256, 93, torch.float32)], torch.float32, label,
                step=functools.partial(_checkpointed, suspend_forward=suspend_forward))
    assert stats(graph) == [expect]


def test_replays_on_two_streams_are_ordered():
    """Layers sharing a pool replay on a side stream and then, with no synchronization, on the
    capture stream: both steps match eager."""
    eager, graph, _ = make_pair([GEN, GEN], torch.bfloat16)
    train_step(graph, *inputs(2, 256, 89, torch.bfloat16))  # captures on this stream
    batches = [inputs(2, 256, 94 + s, torch.bfloat16) for s in range(2)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        first = train_step(graph, *batches[0])
    second = train_step(graph, *batches[1])
    torch.cuda.current_stream().wait_stream(side)
    for run, batch in zip((first, second), batches):
        assert_runs_bitwise([train_step(eager, *batch) for _ in range(EAGER_RUNS)] + [run],
                            label="two streams")


def test_backward_after_a_tf32_change_runs_eager(caplog):
    """A backward graph keeps the matmul precision of its capture: under another allow_tf32
    setting the backward runs eager on the static state (one warning)."""
    eager, graph, _ = make_pair([GEN], torch.float32)
    train_step(graph, *inputs(2, 256, 85, torch.float32))  # captures both, TF32 off

    def tf32_backward(model, x, gy):
        model.zero_grad(set_to_none=True)
        xi = _leaf(x)
        out = model(xi)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            out.backward(gy)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        return _results((model,), [out], [xi])

    # Graph and eager now run TF32 matmuls on buffers at other addresses, which may pick
    # other cuBLAS kernels: compare at TF32 precision.
    with caplog.at_level(logging.WARNING, logger=cuda_graph.__name__):
        check_steps(eager, graph, [inputs(2, 256, 86, torch.float32)], "tf32",
                    "TF32 backward", step=tf32_backward)
    assert len(_warnings(caplog, "allow_tf32 differs")) == 1
    assert stats(graph) == [dict(forward_capture=1, forward_replay=3, backward_capture=1,
                                 eager_backward=2, eager_backward_tf32=2)]


def test_layers_first_called_on_another_stream_use_their_own_pool():
    """Graphs that could replay concurrently share no memory: a layer first called on another
    stream captures into a pool of its own, and still matches eager."""
    eager_a, graph_a, _ = make_pair([GEN], torch.bfloat16)
    eager_b, graph_b, _ = make_pair([GEN], torch.bfloat16)
    batch = inputs(2, 256, 87, torch.bfloat16)
    train_step(graph_a, *batch)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        train_step(graph_b, *batch)
    torch.cuda.current_stream().wait_stream(side)
    (a,), (b,) = ([m.graph_runner.entry for m in g.modules() if isinstance(m, SparseDeltaMemory)]
                  for g in (graph_a, graph_b))
    assert a.pool is not b.pool and a.index == b.index == 0
    check_steps(eager_b, graph_b, [inputs(2, 256, 88, torch.bfloat16)], torch.bfloat16,
                "captured on a side stream")


def test_suspended_calls_run_eager():
    """Inside suspend_cuda_graphs() every call runs eager and captures nothing, so a probe at
    another shape before the first training step leaves that step's shape to be captured."""
    eager, graph, _ = make_pair([GEN, GEN], torch.bfloat16)
    with suspend_cuda_graphs():
        check_steps(eager, graph, [inputs(1, 128, 60, torch.bfloat16)], torch.bfloat16,
                    "suspended probe")
    assert stats(graph) == [{}, {}]
    check_steps(eager, graph, [inputs(2, 256, 61, torch.bfloat16)], torch.bfloat16, "step")
    expect = dict(forward_capture=1, forward_replay=2, backward_capture=1, backward_replay=1)
    assert stats(graph) == [expect] * 2


def test_copies_capture_their_own_graphs():
    """deepcopy and torch.save / torch.load of a model whose layer holds graphs give a fresh
    runner, which captures its own graphs."""
    eager, graph, _ = make_pair([GEN], torch.bfloat16)
    train_step(graph, *inputs(2, 256, 90, torch.bfloat16))
    buf = io.BytesIO()
    torch.save(graph, buf)
    buf.seek(0)
    for clone in (copy.deepcopy(graph), torch.load(buf, weights_only=False)):
        assert stats(clone) == [{}]
        check_steps(eager, clone, [inputs(2, 256, 91, torch.bfloat16)], torch.bfloat16, "copy")
        expect = dict(forward_capture=1, forward_replay=2, backward_capture=1, backward_replay=1)
        assert stats(clone) == [expect]


def lm_step(model, tokens, targets):
    for p in model.parameters():
        p.grad = None
    loss = model(tokens, targets)
    loss.backward()
    res = {"loss": loss.detach().clone()}
    res.update({"grad." + n: full(p.grad) for n, p in model.named_parameters()})
    return res


@pytest.mark.parametrize("cfg,dtype", [(K64, torch.bfloat16), (FP32, torch.float32)],
                         ids=["bf16", "fp32"])
def test_compiled_transformer_with_selective_checkpointing(cfg, dtype):
    """An LMTransformer of SDM layers wrapped as lingua's parallelize_model does (selective
    activation checkpointing, then model.compile()) matches the same model without graphs."""
    pytest.importorskip("apps.main.transformer", reason="needs the repository checkout (apps/, lingua/)")
    from apps.main.transformer import LMTransformer, LMTransformerArgs

    vocab, T = 512, 256
    torch.manual_seed(0)
    args = LMTransformerArgs(dim=DIM, n_layers=2, n_heads=4, vocab_size=vocab, max_seqlen=T,
                             sdm_at="all", sdm_args=SparseDeltaMemoryArgs(cuda_graph=True, **cfg))
    base = LMTransformer(args)
    base.init_weights()
    base = base.to("cuda", dtype)
    eager, graph = eager_twin(base), base

    def wrap(model):
        model = lingua_sac(model)
        model.compile()
        return model

    # Every token id once per batch: the compiled embedding backward scatters repeated ids
    # with atomics (nondeterministic in fp32), which is not what this test is about.
    gen = torch.Generator(device="cuda").manual_seed(7)
    batches = [(torch.randperm(vocab, device="cuda", generator=gen)[:2 * T].view(2, T),
                torch.randint(0, vocab, (2, T), device="cuda", generator=gen))
               for _ in range(2)]
    check_steps(wrap(eager), wrap(graph), batches, dtype, "compiled", step=lm_step)
    expect = dict(forward_capture=1, forward_replay=8, backward_capture=1, backward_replay=3)
    assert stats(graph) == [expect] * 2
    torch._dynamo.reset()


@pytest.fixture
def single_rank_group(tmp_path):
    import torch.distributed as dist

    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'store'}", rank=0,
                            world_size=1)
    yield
    dist.destroy_process_group()


# (label, lingua's selective checkpointing, step, calls per layer and step, of which replayed
# again at backward time)
FSDP_CASES = [("selective", True, one_step, 1, 0), ("no_checkpoint", False, one_step, 1, 0),
              ("two_forwards", False, two_forwards, 2, 1)]


@pytest.mark.parametrize("label,sac,step,calls,stale", FSDP_CASES, ids=[c[0] for c in FSDP_CASES])
def test_fsdp2_single_rank(single_rank_group, label, sac, step, calls, stale):
    """FSDP2 with bf16 parameters and reshard_after_forward: every forward (including a
    recompute) reads a freshly gathered bank parameter, which the graphs copy in, and a
    backward-time replay reads the bank parameter regathered for the backward."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (1,))
    policy = dict(mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    def shard(model):
        for block in model.blocks:
            fully_shard(block, **policy)
        model = fully_shard(model, **policy)
        return lingua_sac(model) if sac else model

    eager, graph, inner = make_pair([K64, GEN], torch.float32, shard)
    check_steps((eager,), (graph,), [inputs(2, 256, 100 + s, torch.float32) for s in range(2)],
                torch.bfloat16, f"fsdp2 {label}", step=step)
    steps = 4
    forwards = steps * calls * (2 if sac else 1)
    expect = dict(forward_capture=1, forward_replay=forwards, backward_capture=1,
                  backward_replay=steps * calls - 1)
    if stale:
        expect["stale_replay"] = steps * stale
    assert stats(inner) == [expect] * 2


# Shared entries (cuda_graph._acquire): the SDM layers of one key whose calls run in different
# non-reentrant checkpoint regions replay ONE set of graphs, static buffers and state. Every
# such model is compared bitwise with its eager twin AND with a graphed twin whose layers keep
# entries of their own (captured with the region detection off: the graphs before sharing).
class LinearBlock(nn.Module):
    """A block without an SDM layer (the attention blocks between the SDM blocks of a hybrid)."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x + self.lin(x)


def save_mm(ctx, op, *args, **kwargs):
    """A selective checkpoint policy that keeps the matmul outputs (as lingua's does)."""
    return CheckpointPolicy.MUST_SAVE if op == torch.ops.aten.mm.default else \
        CheckpointPolicy.PREFER_RECOMPUTE


class Ckpt(nn.Module):
    """``module`` in a non-reentrant checkpoint region of its own (``sac``: with a selective
    policy); ``enabled=False`` calls it directly."""

    def __init__(self, module, sac=False):
        super().__init__()
        self.module, self.sac, self.enabled = module, sac, True

    def forward(self, x):
        if not self.enabled:
            return self.module(x)
        kw = dict(context_fn=functools.partial(create_selective_checkpoint_contexts, save_mm)) \
            if self.sac else {}
        return checkpoint(self.module, x, use_reentrant=False, **kw)


# One checkpoint region per SDM block: plain, with a selective policy, and as a layer-wise SAC
# (torch's checkpoint_wrapper, non-reentrant, around every second block, here the SDM ones).
REGION_WRAPS = {"per_block": Ckpt, "per_block_sac": functools.partial(Ckpt, sac=True),
                "mos_layer_sac": functools.partial(checkpoint_wrapper, preserve_rng_state=False)}


def hybrid(cfgs, dtype, wrap, block=Block):
    """Blocks [Linear, wrap(SDM block 0), Linear, wrap(SDM block 1), ...]."""
    blocks = []
    for i, cfg in enumerate(cfgs):
        blocks += [LinearBlock(), wrap(block(graphed_sdm(cfg, i)))]
    return Model(blocks).to("cuda", dtype)


def private_twin(graph, step, batch, monkeypatch):
    """A graphed copy of ``graph`` whose layers captured entries of their own (its first step,
    returned, runs with the region detection off)."""
    twin = copy.deepcopy(graph)  # fresh runners, the same weights
    with monkeypatch.context() as m:
        m.setattr(cuda_graph, "_region_entries", lambda: None)
        first = step(twin, *batch)
    assert all(st.get("forward_capture") == 1 for st in stats(twin)), "private twin shares"
    return twin, first


def check_three(eager, graph, twin, first, batches, label, step=train_step):
    """Every step of ``graph`` and of ``twin`` equals eager bitwise (3 batches, 2 runs each)."""
    for i, batch in enumerate(batches):
        runs = [step(eager, *batch) for _ in range(EAGER_RUNS)]
        runs += [step(graph, *batch) for _ in range(2)] + [step(twin, *batch) for _ in range(2)]
        assert_runs_bitwise(runs + ([first] if i == 0 else []), label=f"{label} {i}")


def assert_shared(graph, n_keys, steps, calls_per_step=1, stale=0):
    """One entry per key, captured once and replayed by every layer of its key; ``stale``
    backward-time forward replays per call."""
    groups = entry_groups(graph)
    assert len(groups) == n_keys
    for group in groups:
        total = entry_totals(group)
        assert total.get("forward_capture") == 1 and total.get("backward_capture") == 1, total
        assert total.get("forward_shared", 0) == len(group) - 1, total
        assert total.get("stale_replay", 0) == stale * steps * calls_per_step, total
        assert not any(k.startswith("eager_") for k in total), total


SHARED_MODELS = [("bf16", [K64, K64, K64, GEN], torch.bfloat16, 256),
                 ("fp32", [FP32, FP32, FP32], torch.float32, 128)]


@pytest.mark.parametrize("model_label,cfgs,dtype,T", SHARED_MODELS, ids=[m[0] for m in SHARED_MODELS])
@pytest.mark.parametrize("wrap", REGION_WRAPS)
def test_per_block_checkpoint_shares_one_entry_per_key(monkeypatch, wrap, model_label, cfgs,
                                                        dtype, T):
    """Forward of every layer, then the backward in reverse with each block's recompute right
    before it: the layers of one key share one entry, no backward replays its forward, and every
    step equals eager and the graphs without sharing bitwise."""
    graph = hybrid(cfgs, dtype, REGION_WRAPS[wrap])
    eager = eager_twin(graph)
    batches = [inputs(2, T, 110 + s, dtype) for s in range(3)]
    twin, first = private_twin(graph, train_step, batches[0], monkeypatch)
    check_three(eager, graph, twin, first, batches, f"{wrap} {model_label}")
    assert_shared(graph, len({tuple(sorted(c.items())) for c in cfgs}), steps=6)
    per_layer = [st["forward_replay"] for st in stats(graph)]
    assert per_layer == [2 * 6] * len(cfgs), per_layer  # forward + recompute per step


@pytest.mark.parametrize("wrap", ["per_block", "mos_layer_sac"])
def test_per_block_checkpoint_gradient_accumulation(monkeypatch, wrap):
    """Two micro-batches per step (the paper-speed graph runs use grad_acc 2): no stale
    replay, and every step equals eager and the graphs without sharing bitwise."""
    def two_microbatches(model, a, b):
        train_step(model, *a)
        return train_step(model, *b, accumulate=True)

    graph = hybrid([GEN, GEN, GEN], torch.bfloat16, REGION_WRAPS[wrap])
    eager = eager_twin(graph)
    batches = [tuple(inputs(2, 256, 120 + 2 * s + j, torch.bfloat16) for j in range(2))
               for s in range(3)]
    twin, first = private_twin(graph, two_microbatches, batches[0], monkeypatch)
    check_three(eager, graph, twin, first, batches, f"accumulation {wrap}", step=two_microbatches)
    assert_shared(graph, 1, steps=6, calls_per_step=2)


class TwiceBlock(Block):
    """One SDM layer called twice in one block (and so in one checkpoint region)."""

    def forward(self, x):
        h = x + self.lin(x)
        h = h + self.sdm(h)[0]
        return h + self.sdm(h * 0.5)[0]


def weight_shared_blocks(dtype):
    block = Ckpt(Block(graphed_sdm(GEN)))
    return Model([LinearBlock(), block, LinearBlock(), block]).to("cuda", dtype)


def disable_checkpoints(model):
    for m in model.modules():
        if isinstance(m, Ckpt):
            m.enabled = False


# (label, build, graph step, stale replays per step and layer of a shared entry but its last)
ORDER_CASES = [
    ("two_forwards", lambda dt: hybrid([GEN, GEN, GEN], dt, Ckpt), two_forwards, 0),
    ("weight_shared_layer_in_two_regions", weight_shared_blocks, one_step, 0),
    ("layer_called_twice_in_one_region", lambda dt: hybrid([GEN, GEN], dt, Ckpt, TwiceBlock),
     one_step, 1),
]


@pytest.mark.parametrize("label,build,step,stale", ORDER_CASES, ids=[c[0] for c in ORDER_CASES])
def test_shared_entry_call_orders(label, build, step, stale):
    """Call orders on shared entries: two forwards before one backward and a layer called in
    two regions need no extra replay; a layer called twice in one region replays the forward of
    its first call (the second overwrote it), bitwise equal to eager."""
    graph = (build(torch.bfloat16),)
    eager = tuple(eager_twin(m) for m in graph)
    batches = [inputs(2, 256, 130 + s, torch.bfloat16) for s in range(3)]
    check_steps(eager, graph, batches, torch.bfloat16, label, step=step)
    total = entry_totals([st for g in entry_groups(graph[0]) for st in g])
    assert total.get("stale_replay", 0) == stale * 6 * len(sdm_layers(graph[0])), total
    assert total.get("forward_capture") == 1 and not any(k.startswith("eager_") for k in total)
    # Two calls per layer and step, each replayed by its forward and its recompute.
    assert all(st["forward_replay"] == 2 * 2 * 6 for st in stats(*graph)), stats(*graph)


def test_shared_entry_without_checkpoint_replays_overwritten_state(monkeypatch):
    """Layers that shared an entry under per-block checkpointing, then called without it
    (forward of every layer, backward in reverse): each backward but the last finds the state of
    a later layer and replays its own forward first; every step equals eager and the graphs
    without sharing bitwise. A backward that trusted the workspace would get wrong gradients."""
    graph = hybrid([GEN, GEN, GEN], torch.bfloat16, Ckpt)
    eager = eager_twin(graph)
    batches = [inputs(2, 256, 140 + s, torch.bfloat16) for s in range(3)]
    twin, first = private_twin(graph, train_step, batches[0], monkeypatch)
    train_step(graph, *batches[0])  # captures the shared entry under checkpointing
    for model in (graph, eager, twin):
        disable_checkpoints(model)
    check_three(eager, graph, twin, first, batches, "no checkpoint after a shared capture")
    (group,) = entry_groups(graph)
    assert entry_totals(group)["stale_replay"] == 6 * (len(group) - 1)
    assert sum(st.get("stale_replay", 0) for st in stats(twin)) == 0


def test_fsdp2_single_rank_per_block_checkpoint(single_rank_group):
    """FSDP2 (bf16 parameters, reshard_after_forward) around per-block checkpointing: the SDM
    layers share one entry, each recompute reads a freshly gathered bank parameter, and every
    step equals eager bitwise with no stale replay."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (1,))
    policy = dict(mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    def shard(model):
        for block in model.blocks:
            fully_shard(block, **policy)
        return fully_shard(model, **policy)

    graph = hybrid([GEN, GEN, GEN], torch.float32, Ckpt)
    eager = shard(eager_twin(graph))
    graph = shard(graph)
    check_steps((eager,), (graph,), [inputs(2, 256, 150 + s, torch.float32) for s in range(3)],
                torch.bfloat16, "fsdp2 per-block checkpoint", step=one_step)
    assert_shared(graph, 1, steps=6)


@pytest.mark.parametrize("phase", ["warmup_oom", "capture_error", "first_backward_oom"])
def test_shared_entry_failures_run_eager(monkeypatch, caplog, phase):
    """A failed capture of a shared entry leaves the capturing layer eager and the next layer
    of its key captures it; an out-of-memory first backward releases the entry for every layer
    sharing it (pending backwards from their saved inputs). Every step equals eager."""
    calls = []
    if phase == "warmup_oom":
        real_save = cuda_graph._Ctx.save_for_backward

        def save(self, *tensors):  # the first warm-up on the static copies
            calls.append(1)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("injected")
            return real_save(self, *tensors)
        monkeypatch.setattr(cuda_graph._Ctx, "save_for_backward", save)
    elif phase == "capture_error":
        real = cuda_graph._capture

        def capture(fn, pool):
            calls.append(fn)
            return (_synchronizing(real) if len(calls) == 1 else real)(fn, pool)
        monkeypatch.setattr(cuda_graph, "_capture", capture)
    else:
        real_body = cuda_graph.CudaGraphRunner._backward_body

        def backward_body(self, e):
            calls.append(e)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("injected")
            return real_body(self, e)
        monkeypatch.setattr(cuda_graph.CudaGraphRunner, "_backward_body", backward_body)
    graph = hybrid([GEN, GEN, GEN], torch.bfloat16, Ckpt)
    eager = eager_twin(graph)
    with caplog.at_level(logging.WARNING, logger=cuda_graph.__name__):
        check_steps(eager, graph, [inputs(2, 256, 160 + s, torch.bfloat16) for s in range(3)],
                    torch.bfloat16, f"shared {phase}")
    assert torch.cuda.current_stream() == torch.cuda.default_stream(), "left on the capture stream"
    st = stats(graph)
    if phase == "first_backward_oom":
        assert len(_warnings(caplog, "ran out of memory")) == 3
        assert all(s.get("eager_capture_failed", 0) > 0 for s in st), st
        assert all(m.graph_runner.entry is None for m in sdm_layers(graph))
        assert not cuda_graph._shared
    else:
        assert len(_warnings(caplog, "failed")) == 1
        assert st[0] == {"eager_capture_failed": 12}, st  # 6 steps x (forward + recompute)
        assert st[1]["forward_capture"] == 1 and st[2]["forward_shared"] == 1, st
        assert sum(s.get("stale_replay", 0) for s in st) == 0, st


def test_checkpoint_region_detection():
    """Pins the region detection on the installed torch: non-reentrant checkpoints (plain, with a
    selective policy, torch's checkpoint_wrapper) are regions, each call of one region sees the
    same entry set and two regions different ones; no checkpoint, a reentrant one, CPU offload
    and a recompute are not; an error in the private hook API counts as no region."""
    seen = []

    def f(x):
        seen.append(cuda_graph._region_entries())
        return (x * 2).sin()

    def run(fn):
        seen.clear()
        x = torch.randn(64, device="cuda", requires_grad=True)
        fn(x).sum().backward()
        return list(seen)

    class Wrapped(nn.Module):
        def forward(self, x):
            return f(f(x))

    sac = functools.partial(create_selective_checkpoint_contexts, save_mm)
    for label, fn in [("plain", lambda x: checkpoint(f, x, use_reentrant=False)),
                      ("selective", lambda x: checkpoint(f, x, use_reentrant=False, context_fn=sac)),
                      ("checkpoint_wrapper", checkpoint_wrapper(Wrapped(), preserve_rng_state=False))]:
        got = run(fn)
        forward, recompute = got[:len(got) // 2], got[len(got) // 2:]
        assert all(s is not None and s is forward[0] for s in forward), (torch.__version__, label)
        assert all(s is None for s in recompute), label
    a, b, *rest = run(lambda x: checkpoint(f, checkpoint(f, x, use_reentrant=False),
                                           use_reentrant=False))
    assert a is not None and b is not None and a is not b and rest == [None, None]
    assert run(f) == [None]
    assert run(lambda x: checkpoint(f, x, use_reentrant=True)) == [None, None]
    with torch.autograd.graph.save_on_cpu():
        assert run(f) == [None]

    def broken(_):
        raise RuntimeError("private API changed")

    keep = cuda_graph._top_hooks
    try:
        for getter in (broken, None):
            cuda_graph._top_hooks = getter
            assert run(lambda x: checkpoint(f, x, use_reentrant=False)) == [None, None]
    finally:
        cuda_graph._top_hooks = keep


class NestedBlock(nn.Module):
    """SDM layers a, b and c in one checkpoint region, b in a region nested in it."""

    def __init__(self, a, b, c):
        super().__init__()
        self.a, self.b, self.c = a, b, c

    def forward(self, x):
        x = x + self.a(x)[0]
        x = x + checkpoint(lambda t: self.b(t)[0], x, use_reentrant=False)
        return x + self.c(x)[0]


def test_nested_checkpoint_regions():
    """a and c (one key) in an outer region, b (another key) in a region nested between them:
    back in the outer region, c still sees a's entry as used there and takes another slot, so
    no backward replays its forward (with c on a's entry, a's backward would). Every step
    equals eager bitwise. (A layer FIRST called in a nested region cannot know the outer
    region's entries and may share one: that costs one forward, see README.)"""
    layers = [graphed_sdm(cfg, i) for i, cfg in enumerate((GEN, K64, GEN))]
    graph = Model([LinearBlock(), Ckpt(NestedBlock(*layers))]).to("cuda", torch.bfloat16)
    eager = eager_twin(graph)
    check_steps(eager, graph, [inputs(2, 256, 170 + s, torch.bfloat16) for s in range(3)],
                torch.bfloat16, "nested regions")
    a, b, c = (m.graph_runner.entry for m in sdm_layers(graph))
    assert len({id(a), id(b), id(c)}) == 3
    st = stats(graph)
    assert sum(s.get("stale_replay", 0) for s in st) == 0, st
    assert not any(k.startswith("eager_") for s in st for k in s), st


def test_threads_share_entries_safely():
    """Two threads train two models whose layers share entries (captured first on the default
    stream), concurrently on two side streams: the calls on the shared pool are serialized, an
    overwritten state is replayed, and every step equals eager bitwise."""
    import threading

    graphs = [hybrid([GEN, GEN], torch.bfloat16, Ckpt) for _ in range(2)]
    eagers = [eager_twin(g) for g in graphs]
    batches = [[inputs(2, 256, 180 + 10 * k + s, torch.bfloat16) for s in range(3)] for k in range(2)]
    want = [[train_step(e, *b) for b in bs] for e, bs in zip(eagers, batches)]
    for g, bs in zip(graphs, batches):
        train_step(g, *bs[0])  # captures (the second model shares the first one's entry)
    assert len({id(m.graph_runner.entry) for g in graphs for m in sdm_layers(g)}) == 1
    got, errors = [[None] * 3 for _ in range(2)], []

    def run(k):
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                for s, batch in enumerate(batches[k]):
                    got[k][s] = train_step(graphs[k], *batch)
            stream.synchronize()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    assert not any(t.is_alive() for t in threads), "deadlock"
    torch.cuda.synchronize()
    assert not errors, errors
    for k in range(2):
        for s in range(3):
            assert_runs_bitwise([want[k][s], got[k][s]], label=f"thread {k} step {s}")
    assert not any(key.startswith("eager_") for g in graphs for st in stats(g) for key in st)


def test_threads_capture_concurrently():
    """Two threads whose models are first called at the same time on two streams: their
    captures (one pool per stream) run one at a time, and every step equals eager bitwise."""
    import threading

    graphs = [hybrid([GEN, GEN], torch.bfloat16, Ckpt) for _ in range(2)]
    eagers = [eager_twin(g) for g in graphs]
    batches = [[inputs(2, 256, 250 + 10 * k + s, torch.bfloat16) for s in range(3)] for k in range(2)]
    want = [[train_step(e, *b) for b in bs] for e, bs in zip(eagers, batches)]
    got, errors, start = [[None] * 3 for _ in range(2)], [], threading.Barrier(2)

    def run(k):
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.default_stream())
            start.wait()
            with torch.cuda.stream(stream):
                for s, batch in enumerate(batches[k]):
                    got[k][s] = train_step(graphs[k], *batch)
            stream.synchronize()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    assert not any(t.is_alive() for t in threads), "deadlock"
    torch.cuda.synchronize()
    assert not errors, errors
    for k in range(2):
        for s in range(3):
            assert_runs_bitwise([want[k][s], got[k][s]], label=f"thread {k} step {s}")
    assert all(entry_totals(g)["forward_capture"] == 1 for m in graphs for g in entry_groups(m))


# The state of a replayed call (cuda_graph.ReplayedState): a private copy of the call's final
# bank at the first read, never the static bank; a first read after a forward or backward that
# reuses the call's graph memory raises.
def _final_bank(layer, x):
    """The final bank of an eager call (read before its backward restores it)."""
    out, state = layer(_leaf(x))
    return state.memory.detach().clone()


def test_replayed_state_reads_copy_or_raise():
    """First read right after the call: a private copy, bitwise the eager final bank, kept for
    later reads and not rewound by the backward. A first read after the layer's next forward,
    or after the call's backward, raises; update_ and reset act on the state's own tensor."""
    layer = graphed_sdm(GEN).to("cuda", torch.bfloat16)
    eager = eager_twin(layer)
    xs = [inputs(2, 256, 190 + s, torch.bfloat16) for s in range(5)]
    want = [_final_bank(eager, x) for x, _ in xs]
    states = []
    for x, gy in xs[:2]:
        out, st = layer(_leaf(x))
        states.append((out, st))
    (o0, s0), (o1, s1) = states
    assert isinstance(s0, cuda_graph.ReplayedState) and s0.cache_len == 256
    with pytest.raises(RuntimeError, match="new_cache.memory right after the call"):
        s0.memory  # the layer's next forward replayed the graph that wrote it
    copy1 = s1.memory
    assert bits_equal(copy1, want[1]) and s1.memory is copy1 and s1[0] is copy1
    assert copy1.data_ptr() != layer.graph_runner.entry.bank.data_ptr()
    o1.backward(xs[1][1])
    assert bits_equal(s1.memory, want[1]), "the backward rewound the copy"
    out, s2 = layer(_leaf(xs[2][0]))
    out.backward(xs[2][1])
    with pytest.raises(RuntimeError):
        s2.memory  # the backward rewound the static bank
    out, s3 = layer(_leaf(xs[3][0]))
    s3.update_(torch.ones_like(want[3]), seq_len=5)  # replaces it without reading
    assert torch.equal(s3.memory, torch.ones_like(want[3])) and s3.cache_len == 261
    s3.reset()
    assert s3.memory.abs().max() == 0 and s3.cache_len == 0
    out.backward(xs[3][1])
    out, s4 = layer(_leaf(xs[4][0]))
    assert bits_equal(s4.memory, want[4]), "reset of a state changed the graph's bank"


def test_replayed_state_stale_after_a_later_layer():
    """Two layers sharing one entry (one checkpoint region each): A's state read after B's
    forward raises, B's equals eager bitwise, and both gradients equal eager."""
    graph = [graphed_sdm(GEN, i).to("cuda", torch.bfloat16) for i in range(2)]
    eager = [eager_twin(m) for m in graph]
    (xa, ga), (xb, gb) = (inputs(2, 256, 200 + s, torch.bfloat16) for s in range(2))
    want_b = _final_bank(eager[1], xb)

    def run(layers, x, gy):
        leaf = _leaf(x)
        out, st = checkpoint(lambda t: layers(t), leaf, use_reentrant=False)
        return leaf, out, st

    for step in range(2):  # the first step captures
        la, oa, sa = run(graph[0], xa, ga)
        lb, ob, sb = run(graph[1], xb, gb)
        assert graph[0].graph_runner.entry is graph[1].graph_runner.entry
        with pytest.raises(RuntimeError):
            sa.memory
        assert bits_equal(sb.memory, want_b)
        torch.autograd.backward([oa, ob], [ga, gb])
    ea, eb = (run(m, x, g) for m, (x, g) in zip(eager, ((xa, ga), (xb, gb))))
    torch.autograd.backward([ea[1], eb[1]], [ga, gb])
    assert bits_equal(la.grad, ea[0].grad) and bits_equal(lb.grad, eb[0].grad)


def test_replayed_state_read_on_another_stream():
    """A call replayed on one side stream and its state first read on another, with no
    synchronization in between: the copy is ordered after the call's writes."""
    layer = graphed_sdm(GEN).to("cuda", torch.bfloat16)
    eager = eager_twin(layer)
    x, gy = inputs(2, 256, 210, torch.bfloat16)
    want = _final_bank(eager, x)
    layer(_leaf(x))[0].backward(gy)  # captures
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    s1.wait_stream(torch.cuda.current_stream())
    s2.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        with torch.cuda.stream(s1):
            out, st = layer(_leaf(x))
        with torch.cuda.stream(s2):
            got = st.memory
            torch.cuda.current_stream().synchronize()
        assert bits_equal(got, want)
        with torch.cuda.stream(s1):
            out.backward(gy)
    torch.cuda.synchronize()


def test_replayed_state_is_an_opaque_leaf_and_holds_no_entry():
    """pytree, copy.copy, repr and the output traversal of torch.distributed (FSDP / DDP)
    treat the state as a leaf and never read it; it keeps no entry alive (after
    reset_cuda_graphs() its entry dies and a read raises), and dropping a read state frees
    its copy."""
    import gc

    from torch.distributed.utils import _apply_to_tensors
    from torch.utils._pytree import tree_flatten, tree_map

    layer = graphed_sdm(GEN).to("cuda", torch.bfloat16)
    x, gy = inputs(2, 256, 220, torch.bfloat16)
    layer(_leaf(x))[0].backward(gy)  # captures
    out, st = layer(_leaf(x))
    leaves, _ = tree_flatten((out, st))
    assert len(leaves) == 2 and leaves[1] is st
    assert tree_map(lambda t: t, (out, st))[1] is st
    assert _apply_to_tensors(lambda t: t, (out, st))[1] is st
    shallow = copy.copy(st)
    repr(st)
    assert st._copy is None and shallow._copy is None, "a traversal read the state"
    entry = weakref.ref(layer.graph_runner.entry)
    del out, shallow, leaves  # the output's autograd node holds the entry
    reset_cuda_graphs()
    gc.collect()
    assert entry() is None, "the state kept its entry alive"
    with pytest.raises(RuntimeError):
        st.memory
    out, st = layer(_leaf(x))
    copy_ref = weakref.ref(st.memory)
    del out, st
    gc.collect()
    assert copy_ref() is None, "the state's copy outlived it"


def test_no_checkpoint_and_inference_calls_keep_private_entries():
    """Without activation checkpointing every layer keeps an entry of its own (slot None, as
    before sharing existed), and calls without a backward (no_grad, eval, prefill, decode from
    a cache) replay nothing, capture nothing and return plain states."""
    eager, graph, _ = make_pair([GEN, GEN, GEN], torch.bfloat16)
    check_steps(eager, graph, [inputs(2, 256, 230 + s, torch.bfloat16) for s in range(2)],
                torch.bfloat16, "private entries")
    entries = [m.graph_runner.entry for m in sdm_layers(graph)]
    assert len({id(e) for e in entries}) == 3 and all(e.slot is None for e in entries)
    before = stats(graph)
    layer = sdm_layers(graph)[0]
    x, _ = inputs(2, 256, 240, torch.bfloat16)
    with torch.no_grad():
        _, st = layer(x)
        assert type(st) is SDMLayerState
        layer.eval()
        cache = layer.create_kv_cache(bsz=2, seq_len=0, dtype=torch.bfloat16, device="cuda")
        _, cache = layer(x[:, :128], cache=cache)
        for t in range(128, 132):
            _, cache = layer(x[:, t:t + 1], cache=cache)
        layer.train()
    assert type(cache) is SDMLayerState and stats(graph) == before


def test_failed_backward_capture_orders_the_gradients():
    """A backward capture that fails (its capture_end raises) returns the eager gradients on
    the caller's stream, consumed at once on another stream: they equal eager bitwise, and
    the caller's stream is the current one again."""
    real, calls, streams = cuda_graph._capture, [], []

    def capture(fn, pool):  # the 2nd capture is the backward's: make it fail
        calls.append(fn)
        before = torch.cuda.current_stream()
        try:
            return (_synchronizing(real) if len(calls) == 2 else real)(fn, pool)
        finally:
            streams.append((before, torch.cuda.current_stream()))

    layer = graphed_sdm(GEN).to("cuda", torch.bfloat16)
    eager = eager_twin(layer)
    x, gy = inputs(2, 256, 260, torch.bfloat16)
    want = train_step(eager, x, gy, layer=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    orig = cuda_graph._capture
    cuda_graph._capture = capture
    try:
        with torch.cuda.stream(side):
            got = train_step(layer, x, gy, layer=True)
            assert torch.cuda.current_stream() == side, "left on the capture stream"
        torch.cuda.current_stream().wait_stream(side)
    finally:
        cuda_graph._capture = orig
    assert layer.graph_runner.entry.backward_eager and len(streams) == 2
    assert all(b == a for b, a in streams), "a capture left its thread on the capture stream"
    assert_runs_bitwise([want, got], label="failed backward capture")


def test_side_stream_out_of_memory_backward_then_reuse():
    """The first backward runs out of memory on a side stream, its gradient copy delayed
    behind a sleep there. Right after the entry's buffers (allocated on the default stream) are
    released, still inside the teardown, tensors are allocated on the default stream: one of
    them reuses the gradient buffer's memory (at its address, or at an offset inside it where the
    caching allocator merged the freed block with a free neighbour first, as on A100 / sm_80), and
    all keep their values because the default stream joined the side stream before the release.
    The gradients equal eager bitwise."""
    real_body, calls = cuda_graph.CudaGraphRunner._backward_body, []

    def backward_body(self, e):
        calls.append(e)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("injected")
        return real_body(self, e)

    real_release, junk, freed = cuda_graph._Entry.release, [], []

    def release(self):  # allocate on the default stream right after the release, in the window
        if self.grad_in is not None:
            freed.append(self.grad_in.data_ptr())
            freed.append(self.grad_in.numel() * self.grad_in.element_size())   # the freed buffer's bytes
            numel, device = self.grad_in.numel(), self.grad_in.device
        real_release(self)
        if freed and not junk:
            with torch.cuda.stream(torch.cuda.default_stream(device)):
                junk.extend(torch.full((numel,), 7.0, device=device, dtype=torch.bfloat16)
                            for _ in range(32))

    layer = graphed_sdm(GEN).to("cuda", torch.bfloat16)
    eager = eager_twin(layer)
    x, gy = inputs(2, 256, 270, torch.bfloat16)
    want = train_step(eager, x, gy, layer=True)
    layer(_leaf(x))  # captures on the default stream (no backward yet)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    patches = [(cuda_graph.CudaGraphRunner, "_backward_body", backward_body),
               (cuda_graph._Entry, "release", release)]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, f in patches:
        setattr(o, n, f)
    try:
        with torch.cuda.stream(side):
            for p in layer.parameters():
                p.grad = None
            xi = _leaf(x)
            out, _ = layer(xi)
            torch.cuda._sleep(200_000_000)  # the backward's gradient copy waits behind it
            out.backward(gy)
        torch.cuda.synchronize()
    finally:
        for o, n, f in saved:
            setattr(o, n, f)
    got = {"out": out.detach().clone(), "grad_x": xi.grad.clone()}
    got.update({"grad." + n: p.grad.clone() for n, p in layer.named_parameters()})
    # Premise: a junk tensor overlaps the freed gradient buffer [freed[0], freed[0] + freed[1]) (the exact interval
    # test): reuse at the buffer's own address, or at an
    # offset where the allocator merged the freed block with a free neighbour (A100). Fails if none overlaps it.
    assert freed and any(j.data_ptr() < freed[0] + freed[1] and freed[0] < j.data_ptr() + j.numel() * j.element_size()
                         for j in junk), "premise: no reuse"
    assert all(bool((j == 7).all()) for j in junk), "a freed buffer was reused before its last use"
    assert layer.graph_runner.capture_failed
    assert_runs_bitwise([want, got], label="side-stream out-of-memory backward")


def test_pool_lock_serializes_threads():
    """One thread trains (forward + backward) while another keeps replaying the shared entry
    with forward calls that need no backward, from two side streams. Every forward replay and
    every backward (graph replay, stale replay) runs alone among the pool's: with a 20 ms host
    pause injected inside each (which makes any overlap visible), no two intervals overlap,
    whichever thread ran them (the workers' forwards, the autograd engine's device thread for
    the recomputes and backwards)."""
    import threading
    import time

    graphs = [hybrid([GEN, GEN], torch.bfloat16, Ckpt) for _ in range(2)]
    batches = [[inputs(2, 256, 280 + 10 * k + s, torch.bfloat16) for s in range(2)] for k in range(2)]
    for g, bs in zip(graphs, batches):
        train_step(g, *bs[0])  # captures on the default stream (the second model shares the entry)
    assert len({id(m.graph_runner.entry) for g in graphs for m in sdm_layers(g)}) == 1
    Runner = cuda_graph.CudaGraphRunner
    real = {n: getattr(Runner, n) for n in ("_replay_forward", "_backward_graphs")}
    spans, errors, start, done = [], [], threading.Barrier(2), threading.Event()

    def timed(name):
        def run(self, *args):
            t0 = time.perf_counter()
            time.sleep(0.02)
            try:
                return real[name](self, *args)
            finally:
                spans.append((threading.get_ident(), t0, time.perf_counter()))
        return run

    def run(k):
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.default_stream())
            start.wait()
            with torch.cuda.stream(stream):
                if k == 0:  # trains
                    for batch in batches[0]:
                        train_step(graphs[0], *batch)
                else:  # forwards only, until the trainer is done (they replay: inputs need grad)
                    x = batches[1][0][0].detach().clone().requires_grad_(True)
                    for _ in range(400):
                        if done.is_set():
                            break
                        graphs[1](x)
            stream.synchronize()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            if k == 0:
                done.set()

    for n in real:
        setattr(Runner, n, timed(n))
    try:
        threads = [threading.Thread(target=run, args=(k,)) for k in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=900)
    finally:
        for n, f in real.items():
            setattr(Runner, n, f)
    assert not any(t.is_alive() for t in threads), "deadlock"
    assert not errors, errors
    spans.sort(key=lambda s: s[1])
    # the two workers' forwards, the trainer's recomputes and backwards on the autograd device thread
    assert len({s[0] for s in spans}) >= 3 and len(spans) >= 8, spans
    end, overlaps = spans[0][2], 0
    for _, t0, t1 in spans[1:]:
        overlaps += t0 < end
        end = max(end, t1)
    assert overlaps == 0, f"{overlaps} replays / backwards of one pool overlapped another's"
