# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Timing of one SparseDeltaMemory layer, or of its write/read op alone, per process.

The implementation timed is the SDM package of the first checkout on PYTHONPATH
(``sparse_delta_memory``, or ``lingua.sparse_delta_memory`` in the previous release), so two
checkouts are compared by switching PYTHONPATH (give each its own TORCH_EXTENSIONS_DIR and
TRITON_CACHE_DIR).

Modes:
  train   forward + backward of the layer from a fixed upstream gradient: eager (CUDA
          events) and one CUDA graph of the whole step.
  eval    layer.eval() forward under no_grad: eager and graph.
  decode  layer.eval() single-token step with a cache: eager and graph.
  op      the layer's write/read call (``SparseDeltaMemory.gated_write_read``) alone,
          forward + backward, on the inputs the layer hands it: eager, graph, and the GPU
          kernel time of its forward and of its backward from the profiler.

--sac (train mode) runs the layer under lingua's selective activation checkpointing (the
policy of ``distributed.selective_activation_checkpointing``). --sdm-cuda-graph (train mode)
sets ``SparseDeltaMemoryArgs.cuda_graph`` and times eager steps only.

--indices (op mode) replaces the layer's slot selections; values, gates and the bank stay
the layer's own:
  layer        the selections the layer made
  uniform      distinct uniform slots per token
  zipf         slot popularity proportional to 1 / rank, in a random order per stream
  trace:FILE   an .npz with int k_idx [streams, T', W], q_idx [streams, T', R] (slots local
               to a stream's bank, distinct per token, T' >= T) and slots_per_head; streams
               are reused cyclically

Times are medians over --iters. The run refuses to time while another process is on the GPU
(--allow-busy overrides it; the process list is saved either way).

Example:
  PYTHONPATH=. python benchmarks/bench_sdm.py --mode train --hd 128 --W 64 --R 64 --C 64 \\
      --out benchmarks/results/train_hd128.json
"""

import argparse
import functools
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

import torch
from torch.profiler import ProfilerActivity, profile


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "eval", "decode", "op"), default="train")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--hd", type=int, default=128, help="head dim; heads = dim // hd")
    p.add_argument("--W", type=int, default=64, help="writes per token")
    p.add_argument("--R", type=int, default=64, help="reads per token")
    p.add_argument("--C", type=int, default=64,
                   help="chunk size (chunk_size; memory_block_size in the previous release)")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--sph", type=int, default=65536, help="slots per head")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--quant", choices=("none", "fp8", "int4", "bf16"), default="none")
    p.add_argument("--kwd", action="store_true", help="key_weighted_decay")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls")
    p.add_argument("--sac", action="store_true", help="train: selective activation checkpointing")
    p.add_argument("--sdm-cuda-graph", action="store_true",
                   help="train: SparseDeltaMemoryArgs.cuda_graph=True (eager timing only)")
    p.add_argument("--indices", default="layer", help="op mode selections (see above)")
    p.add_argument("--seed", type=int, default=0, help="seed of synthetic selections")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-eager", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.indices not in ("layer", "uniform", "zipf") and not a.indices.startswith("trace:"):
        p.error(f"--indices {a.indices!r}: expected layer, uniform, zipf or trace:FILE")
    if (a.sac or a.sdm_cuda_graph) and a.mode != "train":
        p.error("--sac and --sdm-cuda-graph apply to --mode train")
    if a.sdm_cuda_graph:
        a.no_graph = True
    return a


# ---------------------------------------------------------------------------- timing
def other_gpu_processes():
    """``pid, MiB`` of the other compute processes on this process's GPU, or None if unknown."""
    try:
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        out = subprocess.run(
            ["nvidia-smi", "-i", f"GPU-{uuid}", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60, check=True).stdout
    except (AttributeError, OSError, subprocess.SubprocessError):
        return None
    me = str(os.getpid())
    return [line.strip() for line in out.splitlines()
            if line.strip() and line.split(",")[0].strip() != me]


def require_idle_gpu(a, res):
    others = other_gpu_processes()
    res["gpu_processes"] = others
    if others and not a.allow_busy:
        raise RuntimeError(f"another process is on the GPU: {others}")


def summary(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def event(external=False):
    return torch.cuda.Event(enable_timing=True, external=external)


def eager_times(phases, iters, before=None, alone=None):
    """Per phase, the time from the start of the iteration to the end of that phase (each
    phase gets the previous one's return value), and the host issue time. ``alone``, a list,
    receives per phase the summary of that phase's own time (end of the previous phase to
    its end)."""
    ends, host = [[] for _ in phases], []
    for _ in range(iters):
        if before is not None:
            before()
        evs = [event() for _ in range(len(phases) + 1)]
        h0 = time.perf_counter()
        evs[0].record()
        state = None
        for i, phase in enumerate(phases):
            state = phase(state)
            evs[i + 1].record()
        host.append((time.perf_counter() - h0) * 1e3)
        del state
        torch.cuda.synchronize()
        for i, e in enumerate(evs[1:]):
            ends[i].append(evs[0].elapsed_time(e))
    if alone is not None:
        alone[:] = [summary([b - a for a, b in zip([0.0] * iters if i == 0 else ends[i - 1],
                                                    ends[i])]) for i in range(len(phases))]
    return [summary(x) for x in ends], summary(host)


def capture_graph(fn, warmup, before=None):
    """A CUDA graph of ``fn()``, captured after ``warmup`` runs on a side stream;
    ``before()`` runs eagerly ahead of every warmup run and of the capture."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            if before is not None:
                before()
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if before is not None:
        before()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def replay_times(graph, iters, split=None, before=None, bwd=None):
    """Replay time, and the time between the first two ``split`` events recorded in the
    graph. With a third split event, ``bwd`` (a dict) receives under "ms" the summary of
    the time between the second and the third (inside the graph)."""
    total, part, tail = [], [], []
    for _ in range(iters):
        if before is not None:
            before.replay()
        e0, e1 = event(), event()
        e0.record()
        graph.replay()
        e1.record()
        torch.cuda.synchronize()
        total.append(e0.elapsed_time(e1))
        if split is not None:
            part.append(split[0].elapsed_time(split[1]))
            if len(split) > 2:
                tail.append(split[1].elapsed_time(split[2]))
    if bwd is not None and tail:
        bwd["ms"] = summary(tail)
    return summary(total), (summary(part) if split is not None else None)


def kernel_ms(fn):
    """GPU time of the kernels, memcpys and memsets that ``fn()`` launches, and their count."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    device = [e for e in prof.events() if "CUDA" in str(e.device_type)]
    return out, sum(e.time_range.elapsed_us() for e in device) / 1e3, len(device)


# ------------------------------------------------------------------------------ layer
def dispatch_record():
    """The op's dispatch record, or None for a release without one."""
    from provenance import sdm_package  # this directory is sys.path[0]

    memory_ops = importlib.import_module(sdm_package().__name__ + ".memory_ops")
    record = getattr(memory_ops, "last_dispatch", None)
    return record() if record is not None else None




def build_layer(a, dtype):
    from provenance import chunk_field, sdm_package  # this directory is sys.path[0]

    sdm = sdm_package()
    SparseDeltaMemory, SparseDeltaMemoryArgs = sdm.SparseDeltaMemory, sdm.SparseDeltaMemoryArgs
    torch.manual_seed(1234)
    # cuda_graph is passed only when set, so a release without it also runs.
    option = {"cuda_graph": True} if a.sdm_cuda_graph else {}
    args = SparseDeltaMemoryArgs(
        dim=a.dim, num_heads=a.dim // a.hd, slots_per_head=a.sph,
        num_writes=a.W, num_reads=a.R, **{chunk_field(SparseDeltaMemoryArgs): a.C},
        output_gate=True, normalize_readings=True, backprop_on_memory=True,
        snapshot_quant=a.quant, key_weighted_decay=a.kwd, **option,
    )
    layer = SparseDeltaMemory(args, 0).cuda()
    layer.init_weights()
    return layer.to(dtype)


def layer_inputs(a, dtype, T):
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(a.B, T, a.dim, device="cuda", dtype=dtype, generator=gen)
    gy = torch.randn(a.B, T, a.dim, device="cuda", dtype=dtype, generator=gen) * 0.01
    return x, gy


def gib(nbytes):
    return nbytes / 2**30


def selective_checkpointing(fn):
    """``fn`` under lingua's selective activation checkpointing."""
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

    from lingua.distributed import get_default_policy

    context = functools.partial(create_selective_checkpoint_contexts, get_default_policy())
    return lambda *args: checkpoint(fn, *args, use_reentrant=False, context_fn=context)


def record_weights(layer, res):
    from provenance import sha256_tensors  # this directory is sys.path[0]

    res["weights_sha256"] = sha256_tensors(sorted(layer.state_dict().items()))


def run_train(a, dtype, res):
    layer = build_layer(a, dtype).train()
    record_weights(layer, res)
    x, gy = layer_inputs(a, dtype, a.T)
    x.requires_grad_(True)

    def forward(x):
        return layer(x)[0]

    if a.sac:
        forward = selective_checkpointing(forward)

    def clear():
        layer.zero_grad(set_to_none=True)
        x.grad = None

    def step():
        clear()
        forward(x).backward(gy)

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    res["dispatch"] = dispatch_record()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    res["peak_gib"] = gib(torch.cuda.max_memory_allocated())
    graph = split = None
    if not a.no_graph:
        split = (event(external=True), event(external=True), event(external=True))

        def graph_step():
            split[0].record()
            out = forward(x)
            split[1].record()
            out.backward(gy)
            split[2].record()

        graph = capture_graph(graph_step, warmup=3, before=clear)
    require_idle_gpu(a, res)
    if not a.no_eager:
        alone = []
        (fwd, fwd_bwd), host = eager_times(
            [lambda _: forward(x), lambda out: out.backward(gy)], a.iters, before=clear,
            alone=alone)
        res["eager"] = {"fwd_bwd_ms": fwd_bwd, "fwd_ms": fwd, "bwd_ms": alone[1],
                        "host_issue_ms": host}
    if a.sdm_cuda_graph:  # captures, replays and eager calls of the option
        res["cuda_graph_stats"] = dict(layer.graph_runner.stats)
    if graph is not None:
        for _ in range(a.warmup):
            graph.replay()
        bwd = {}
        total, fwd = replay_times(graph, a.iters, split, bwd=bwd)
        res["graph"] = {"fwd_bwd_ms": total, "fwd_ms": fwd, "bwd_ms": bwd["ms"]}


def run_eval(a, dtype, res):
    layer = build_layer(a, dtype).eval()
    x, _ = layer_inputs(a, dtype, a.T)
    with torch.no_grad():
        for _ in range(a.warmup):
            layer(x)
        torch.cuda.synchronize()
        res["dispatch"] = dispatch_record()
        torch.cuda.reset_peak_memory_stats()
        layer(x)
        torch.cuda.synchronize()
        res["peak_gib"] = gib(torch.cuda.max_memory_allocated())
        graph = None if a.no_graph else capture_graph(lambda: layer(x), warmup=3)
        require_idle_gpu(a, res)
        if not a.no_eager:
            (fwd,), host = eager_times([lambda _: layer(x)[0]], a.iters)
            res["eager"] = {"fwd_ms": fwd, "host_issue_ms": host}
        if graph is not None:
            for _ in range(a.warmup):
                graph.replay()
            res["graph"] = {"fwd_ms": replay_times(graph, a.iters)[0]}


def run_decode(a, dtype, res):
    layer = build_layer(a, dtype).eval()
    x, _ = layer_inputs(a, dtype, 1)
    with torch.no_grad():
        cache = layer.create_kv_cache(a.B, 1, dtype=dtype, device="cuda")

        def step():
            layer(x, cache=cache)

        for _ in range(a.warmup):
            step()
        torch.cuda.synchronize()
        graph = None if a.no_graph else capture_graph(step, warmup=3)
        steps = max(a.iters, 100)
        require_idle_gpu(a, res)
        if not a.no_eager:
            (fwd,), host = eager_times([lambda _: step()], steps)
            res["eager"] = {"step_ms": fwd, "host_issue_ms": host}
        if graph is not None:
            for _ in range(a.warmup):
                graph.replay()
            # One event pair around many replays: a single step is a few microseconds.
            e0, e1 = event(), event()
            e0.record()
            for _ in range(steps):
                graph.replay()
            e1.record()
            torch.cuda.synchronize()
            res["graph"] = {"step_ms": e0.elapsed_time(e1) / steps}


# --------------------------------------------------------------------------------- op
class _Captured(Exception):
    pass


def capture_write_read_inputs(layer, x):
    """The arguments of the layer's write/read call on input ``x`` (memory, k_idx, k_val, v,
    beta, g, q_idx, q_val), cloned."""
    seen = []

    def record(*args):
        seen.extend(t.detach().clone() for t in args)
        raise _Captured

    layer.gated_write_read = record
    try:
        with torch.no_grad():
            layer(x)
    except _Captured:
        pass
    finally:
        del layer.gated_write_read
    if not seen:
        raise RuntimeError("the layer did not call gated_write_read")
    return seen


def synthetic_indices(kind, streams, T, width, sph, gen):
    """[streams, T, width] distinct local slots per token (Gumbel top-k)."""
    rows = max(1, (1 << 26) // sph)
    out = torch.empty(streams, T, width, dtype=torch.int64, device="cuda")
    for s in range(streams):
        log_w = torch.zeros(sph, device="cuda")
        if kind == "zipf":
            rank = torch.randperm(sph, device="cuda", generator=gen)
            log_w = -torch.log1p(rank.float())
        for t0 in range(0, T, rows):
            u = torch.rand(min(rows, T - t0), sph, device="cuda", generator=gen)
            keys = log_w - torch.log(-torch.log(u.clamp_min_(1e-20)))
            out[s, t0:t0 + rows] = keys.topk(width, dim=-1).indices.sort(dim=-1).values
    return out


def frozen_indices(a, streams):
    """(k_idx, q_idx) local slots [streams, T, W] / [streams, T, R] and a description."""
    if a.indices in ("uniform", "zipf"):
        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        return (synthetic_indices(a.indices, streams, a.T, a.W, a.sph, gen),
                synthetic_indices(a.indices, streams, a.T, a.R, a.sph, gen),
                f"{a.indices} seed={a.seed}")
    import numpy as np

    path = a.indices[len("trace:"):]
    with np.load(path) as z:
        k_idx = torch.from_numpy(z["k_idx"]).cuda().long()
        q_idx = torch.from_numpy(z["q_idx"]).cuda().long()
        sph = int(z["slots_per_head"])
    if (k_idx.shape[2], q_idx.shape[2]) != (a.W, a.R) or sph > a.sph or k_idx.shape[1] < a.T:
        raise ValueError(f"{path}: trace has W={k_idx.shape[2]} R={q_idx.shape[2]} "
                         f"slots_per_head={sph} T={k_idx.shape[1]}; the run has W={a.W} "
                         f"R={a.R} slots_per_head={a.sph} T={a.T}")
    pick = torch.arange(streams, device="cuda") % k_idx.shape[0]
    return k_idx[pick, :a.T], q_idx[pick, :a.T], os.path.basename(path)


def run_op(a, dtype, res):
    layer = build_layer(a, dtype).train()
    x, _ = layer_inputs(a, dtype, a.T)
    bank, k_idx, k_val, v, beta, g, q_idx, q_val = capture_write_read_inputs(layer, x)
    del x
    if a.indices == "layer":
        res["indices"] = "the layer's selections"
    else:
        streams = k_idx.shape[0]
        offset = torch.arange(streams, device="cuda").view(-1, 1, 1) * a.sph
        local_k, local_q, res["indices"] = frozen_indices(a, streams)
        k_idx = (local_k + offset).to(k_idx.dtype)
        q_idx = (local_q + offset).to(q_idx.dtype)
    memory = bank.clone().requires_grad_(True)
    leaves = [memory] + [t.requires_grad_(True) for t in (k_val, v, beta, g, q_val)]

    def forward(_=None):
        return layer.gated_write_read(memory, k_idx, k_val, v, beta, g, q_idx, q_val)[0]

    gy = torch.randn(forward().shape, device="cuda", generator=torch.Generator(
        device="cuda").manual_seed(11)).to(dtype) * 0.01

    def copy_bank():
        # The op advances the bank in place; every iteration starts from the same state.
        with torch.no_grad():
            memory.copy_(bank)

    def reset():
        copy_bank()
        for t in leaves:
            t.grad = None

    for _ in range(a.warmup):
        reset()
        forward().backward(gy)
    torch.cuda.synchronize()
    res["dispatch"] = dispatch_record()
    reset()
    torch.cuda.reset_peak_memory_stats()
    forward().backward(gy)
    torch.cuda.synchronize()
    res["peak_gib"] = gib(torch.cuda.max_memory_allocated())

    graph = reset_graph = split = None
    if not a.no_graph:
        split = (event(external=True), event(external=True))

        def graph_step():
            split[0].record()
            out = forward()
            split[1].record()
            out.backward(gy)

        reset_graph = capture_graph(copy_bank, warmup=0)
        graph = capture_graph(graph_step, warmup=3, before=reset)
    require_idle_gpu(a, res)
    if not a.no_eager:
        (fwd, fwd_bwd), host = eager_times(
            [forward, lambda out: out.backward(gy)], a.iters, before=reset)
        res["eager"] = {"fwd_bwd_ms": fwd_bwd, "fwd_ms": fwd, "host_issue_ms": host}
    if graph is not None:
        for _ in range(a.warmup):
            reset_graph.replay()
            graph.replay()
        total, fwd = replay_times(graph, a.iters, split, before=reset_graph)
        res["graph"] = {"fwd_bwd_ms": total, "fwd_ms": fwd}
    del graph
    fwd_ms, bwd_ms, launches = [], [], []
    for _ in range(3):
        reset()
        out, f, nf = kernel_ms(forward)
        _, b, nb = kernel_ms(lambda: out.backward(gy))
        fwd_ms.append(f)
        bwd_ms.append(b)
        launches.append(nf + nb)
    res["kernel_ms"] = {"fwd": statistics.median(fwd_ms), "bwd": statistics.median(bwd_ms),
                        "launches": statistics.median(launches)}
    res["kernel_ms"]["fwd_bwd"] = res["kernel_ms"]["fwd"] + res["kernel_ms"]["bwd"]


# ------------------------------------------------------------------------------- main
def environment():
    props = torch.cuda.get_device_properties(0)
    return {"gpu": props.name, "sm": f"{props.major}.{props.minor}",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.version.split()[0],
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def main(argv=None):
    a = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    dtype = getattr(torch, a.dtype)
    from provenance import checkout_info, public_args  # this directory is sys.path[0]

    res = {"bench": "sdm", "args": public_args(a), "status": "started", "heads": a.dim // a.hd,
           "environment": environment(), "sdm_checkout": checkout_info()}
    runner = {"train": run_train, "eval": run_eval, "decode": run_decode, "op": run_op}[a.mode]
    t0 = time.perf_counter()
    try:
        runner(a, dtype, res)
        res["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="oom", error=str(e)[:2000])
    except Exception as e:
        res.update(status="error", error=f"{type(e).__name__}: {str(e)[:2000]}",
                   traceback=traceback.format_exc()[-4000:])
    res["wall_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    brief = {k: res[k] for k in ("status", "error", "eager", "graph", "kernel_ms", "peak_gib")
             if k in res}
    print(json.dumps(brief, default=str))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
