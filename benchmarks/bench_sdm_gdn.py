# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Model-level timing of Sparse Delta Memory (SDM) against Gated DeltaNet (GDN), one arm per
process.

The model is a pre-norm residual stack of ``--n-blocks`` x [SWA, SWA, SWA, X] with SwiGLU
FFNs, where X is a SparseDeltaMemory layer (``--mixer sdm``) or fla's GatedDeltaNet
(``--mixer gdn``). SWA is causal sliding-window attention through
``torch.compile(flex_attention)``. The loss is mean(out.float() ** 2).

The timed step is forward + backward to every parameter (the optimizer step is excluded
unless ``--optimizer adamw`` adds a fused AdamW step with lr 0, captured in the same graph;
gradient clearing is outside the timed region), eager
(CUDA events) or as a replay of one CUDA graph of the whole step (``--mode graph``).
``--sdm-cuda-graph`` sets ``SparseDeltaMemoryArgs.cuda_graph`` (eager mode only). ``--sac``
wraps the model in lingua's selective activation checkpointing, as
``distributed.selective_activation_checkpointing`` does. The defaults are the flagship
config's layers: SDM with 2 heads of 1024, W = R = 64, C = 64, 65536 slots per head; GDN
with 16 heads, key head_dim 64 and expand_v 2 (the same 1024-wide key projection). The SDM
implementation is the SDM package of the first checkout on PYTHONPATH (``sparse_delta_memory``,
or ``lingua.sparse_delta_memory`` in the previous release). The run refuses to time while
another process is on the GPU (--allow-busy overrides it).

Example:
  PYTHONPATH=. python benchmarks/bench_sdm_gdn.py --mixer sdm --mode graph --batch 2 \\
      --seqlen 8192 --out benchmarks/results/model_sdm_graph.json
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
import zlib
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mixer", choices=["sdm", "gdn"], default="sdm")
    p.add_argument("--mode", choices=["eager", "graph"], default="eager")
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=8192)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--n-blocks", type=int, default=2, help="blocks of [SWA, SWA, SWA, X]")
    p.add_argument("--swa-window", type=int, default=1024)
    p.add_argument("--swa-heads", type=int, default=16)
    # SDM
    p.add_argument("--sdm-heads", type=int, default=2)
    p.add_argument("--slots-per-head", type=int, default=65536)
    p.add_argument("--writes", type=int, default=64)
    p.add_argument("--reads", type=int, default=64)
    p.add_argument("--chunk", type=int, default=64,
                   help="SDM chunk size (chunk_size; memory_block_size in the previous release)")
    p.add_argument("--snapshot-quant", default="none", choices=["none", "fp8", "int4", "bf16"])
    p.add_argument("--sdm-cuda-graph", action="store_true",
                   help="SparseDeltaMemoryArgs.cuda_graph=True (eager mode)")
    # GDN
    p.add_argument("--gdn-heads", type=int, default=0, help="0 = dim // 128")
    p.add_argument("--gdn-head-dim", type=int, default=64)
    p.add_argument("--gdn-expand-v", type=float, default=2.0)
    p.add_argument("--sac", action="store_true", help="selective activation checkpointing")
    p.add_argument("--optimizer", choices=["none", "adamw"], default="none",
                   help="adamw: add a fused AdamW step (lr 0: the full update math, weights "
                        "unchanged, which the run checks)")
    # Run control
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.gdn_heads <= 0:
        a.gdn_heads = max(1, a.dim // 128)
    if a.sdm_cuda_graph and a.mode == "graph":
        p.error("--sdm-cuda-graph applies to eager mode")
    return a


# ----------------------------------------------------------------------------- model


def make_sdm(a, layer_id, dtype):
    from provenance import chunk_field, sdm_package  # this directory is sys.path[0]

    sdm = sdm_package()
    SparseDeltaMemory, SparseDeltaMemoryArgs = sdm.SparseDeltaMemory, sdm.SparseDeltaMemoryArgs

    # cuda_graph is passed only when set, so a release without it also runs.
    option = {"cuda_graph": True} if a.sdm_cuda_graph else {}
    args = SparseDeltaMemoryArgs(
        dim=a.dim, num_heads=a.sdm_heads, slots_per_head=a.slots_per_head,
        num_writes=a.writes, num_reads=a.reads, **{chunk_field(SparseDeltaMemoryArgs): a.chunk},
        output_gate=True, normalize_readings=True, backprop_on_memory=True,
        snapshot_quant=a.snapshot_quant, **option,
    )
    layer = SparseDeltaMemory(args, layer_id).cuda()
    layer.init_weights()
    return layer.to(dtype)


def make_gdn(a, layer_id, dtype):
    from fla.layers import GatedDeltaNet

    layer = GatedDeltaNet(
        hidden_size=a.dim, num_heads=a.gdn_heads, head_dim=a.gdn_head_dim,
        expand_v=a.gdn_expand_v, use_gate=True, use_short_conv=True, conv_size=4,
        mode="chunk", layer_idx=layer_id,
    )
    return layer.cuda().to(dtype)


class Mixer(nn.Module):
    """The first output of an SDM or GDN layer."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)[0]


class SWA(nn.Module):
    """Causal sliding-window multi-head attention (no positional rotation)."""

    def __init__(self, dim, n_heads, block_mask):
        super().__init__()
        self.n_heads = n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.block_mask = block_mask

    def forward(self, x):
        from torch.nn.attention.flex_attention import flex_attention

        B, T, D = x.shape
        q, k, v = (w(x).view(B, T, self.n_heads, -1).transpose(1, 2)
                   for w in (self.wq, self.wk, self.wv))
        o = _compiled(flex_attention)(q, k, v, block_mask=self.block_mask)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


_COMPILED = {}


def _compiled(fn):
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=False)
    return _COMPILED[fn]


def sliding_window_block_mask(T, window):
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    return create_block_mask(mask_mod, None, None, T, T, device="cuda")


class SwiGLU(nn.Module):
    def __init__(self, dim, multiple_of=256):
        super().__init__()
        hidden = multiple_of * -(-int(8 * dim / 3) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, mixer):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        h = x + self.mixer(self.norm1(x))
        return h + self.ffn(self.norm2(h))


def build_model(a, dtype):
    torch.manual_seed(a.seed)
    mask = sliding_window_block_mask(a.seqlen, a.swa_window)
    blocks = []
    for i in range(4 * a.n_blocks):
        if i % 4 < 3:
            mixer = SWA(a.dim, a.swa_heads, mask)
        else:
            mixer = Mixer((make_sdm if a.mixer == "sdm" else make_gdn)(a, i, dtype))
        blocks.append(Block(a.dim, mixer))
    model = nn.Sequential(*blocks).cuda()
    # Every parameter outside the X layers gets the same values in both arms.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".mixer.layer." in name:
                continue
            if "norm" in name:
                p.fill_(1.0)
                continue
            g = torch.Generator(device="cuda").manual_seed(a.seed + zlib.crc32(name.encode()))
            p.normal_(0.0, p.shape[-1] ** -0.5, generator=g)
    model = model.to(dtype).train()
    if a.sac:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )
        from torch.utils.checkpoint import create_selective_checkpoint_contexts

        from lingua.distributed import get_default_policy

        model = checkpoint_wrapper(model, context_fn=functools.partial(
            create_selective_checkpoint_contexts, get_default_policy()))
    return model


def param_counts(model):
    counts = {"x_layers": 0, "sdm_bank": 0, "rest": 0}
    for name, p in model.named_parameters():
        if getattr(p, "_sdm_memory_bank", False):
            counts["sdm_bank"] += p.numel()
        elif ".mixer.layer." in name:
            counts["x_layers"] += p.numel()
        else:
            counts["rest"] += p.numel()
    return counts


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


def summary(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def event(external=False):
    return torch.cuda.Event(enable_timing=True, external=external)


def run(a, res):
    from provenance import sha256_tensors  # this directory is sys.path[0]

    dtype = getattr(torch, a.dtype)
    model = build_model(a, dtype)
    res["params"] = param_counts(model)
    res["x_layers_weights_sha256"] = sha256_tensors(
        sorted((n, p) for n, p in model.named_parameters() if ".mixer.layer." in n))
    opt = None
    if a.optimizer == "adamw":
        # A real fused AdamW step over every parameter (banks included); lr 0 keeps the
        # weights, so every iteration times the same selections (checked at the end).
        opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1, fused=True,
                                capturable=a.mode == "graph")
    gen = torch.Generator(device="cuda").manual_seed(a.seed + 1)
    x = torch.randn(a.batch, a.seqlen, a.dim, device="cuda", generator=gen).to(dtype)

    def clear():
        model.zero_grad(set_to_none=True)

    def forward():
        return model(x).float().pow(2).mean()

    def opt_step():
        if opt is not None:
            opt.step()

    # Graph mode warms up and captures on one side stream: gradient accumulation must not
    # reference the default stream during capture.
    graph = None
    split = (event(external=True), event(external=True), event(external=True))
    stream = torch.cuda.Stream() if a.mode == "graph" else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(a.warmup):
            clear()
            forward().backward()
            opt_step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if a.mode == "graph":
        clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            split[0].record()
            loss = forward()
            split[1].record()
            loss.backward()
            split[2].record()
            opt_step()
        torch.cuda.synchronize()
    res["gpu_processes"] = other_gpu_processes()
    if res["gpu_processes"] and not a.allow_busy:
        raise RuntimeError(f"another process is on the GPU: {res['gpu_processes']}")
    fwd_bwd, fwd, step, host = [], [], [], []
    for _ in range(a.iters):
        e0, e1, e2, e3 = event(), event(), event(), event()
        if graph is None:
            clear()
            h0 = time.perf_counter()
            e0.record()
            loss = forward()
            e1.record()
            loss.backward()
            e2.record()
            opt_step()
            e3.record()
            host.append((time.perf_counter() - h0) * 1e3)
            torch.cuda.synchronize()
            fwd.append(e0.elapsed_time(e1))
            fwd_bwd.append(e0.elapsed_time(e2))
        else:
            e0.record()
            graph.replay()
            e3.record()
            torch.cuda.synchronize()
            fwd.append(split[0].elapsed_time(split[1]))
            fwd_bwd.append(split[0].elapsed_time(split[2]))
        step.append(e0.elapsed_time(e3))
    if opt is None:  # the step is forward + backward; keep the replay-inclusive number
        fwd_bwd = step
    res["timing"] = {"fwd_bwd_ms": summary(fwd_bwd), "fwd_ms": summary(fwd)}
    if opt is not None:
        res["timing"]["step_ms"] = summary(step)
    if host:
        res["timing"]["host_issue_ms"] = summary(host)
    res["x_layers_weights_sha256_after"] = sha256_tensors(
        sorted((n, p) for n, p in model.named_parameters() if ".mixer.layer." in n))
    res["memory"] = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    if a.sdm_cuda_graph:  # captures, replays and eager calls of the option, all SDM layers
        res["cuda_graph_stats"] = dict(sum((m.graph_runner.stats for m in model.modules()
                                            if getattr(m, "graph_runner", None)), Counter()))


def environment():
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name()}
    for mod in ("triton", "fla"):
        try:
            info[mod] = importlib.import_module(mod).__version__
        except Exception:
            info[mod] = None
    return info


def main(argv=None):
    from provenance import public_args  # this directory is sys.path[0]

    a = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"bench": "sdm_gdn", "args": public_args(a), "status": "started",
           "environment": environment()}
    if a.mixer == "sdm":
        from provenance import checkout_info  # this directory is sys.path[0]

        res["sdm_checkout"] = checkout_info()
    try:
        run(a, res)
        res["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="oom", error=str(e)[:2000])
    except Exception as e:
        res.update(status="error", error=f"{type(e).__name__}: {str(e)[:2000]}",
                   traceback=traceback.format_exc()[-4000:])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("status", "error", "timing", "memory", "params")
                      if k in res}, default=str))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
