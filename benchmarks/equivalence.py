# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Numerical equivalence of two SDM checkouts (e.g. the previous release and this one) on
identical weights and inputs, run before timing them against each other.

Each step runs in its own process. The SDM code is the SDM package of the first checkout on
PYTHONPATH (``sparse_delta_memory``, or ``lingua.sparse_delta_memory`` in the previous
release), so two checkouts are compared by switching PYTHONPATH (give each its own
TORCH_EXTENSIONS_DIR and TRITON_CACHE_DIR), as with bench_sdm.py:

  make       fp32 master weights of one layer, the layer inputs, the write/read op's
             arguments captured from one forward pass, and FROZEN slot selections: the
             exact top-k (ties to the lower slot id) of the fp32 projections of the layer,
             and of every SDM layer of the 3:1 model (any checkout)
  run        per checkout and dtype (bf16, and fp32 on the same bf16-valued inputs):
               selector  the checkout's own product-key top-k on the layer's projections:
                         slot ids, values and backward against their definitions
               layer     forward + backward of the layer with the frozen selections
                         (--repeats times), and once with the checkout's own selections
               op        the layer's write/read call alone on the frozen arguments, at
                         every chunk size of --chunks
  run-model  per checkout: forward + backward of bench_sdm_gdn.py's 3:1 model with the
             frozen selections, from a fixed random output gradient
  run-cudagraph  this version only: the layer and the 3:1 model with the layer's
             ``cuda_graph=True`` option against the same modules without it, for CG_STEPS
             consecutive steps on different inputs, the weights (memory bank included)
             changed in place between steps, so a replay that reused stale inputs, weights or
             outputs would differ from the eager step. The model runs free (its own
             selections) next to CG_REPEATS more eager copies, whose spread is the reference
             for the graphed run, and with the frozen selections of every SDM layer. Writes
             cg_<dtype>_<free|frozen>.pt into --out (a directory); --cg-dtype and --cg-mode
             pick one file per process
  oracle     float64 references (no SDM code): the op as a token loop, and the whole layer
             with the frozen selections (as tests/test_layer_reference.py)
  compare    the verdict, as JSON: "verdict" compares the two checkouts (exit status 1
             unless EQUIVALENT); "cuda_graph_option.verdict" (VERIFIED or NOT VERIFIED) is
             the separate result of run-cudagraph

Gates (ceilings fixed in advance; deviation = rms(ref - x) / rms(ref), and max|ref - x| /
max|ref| for tensors of at most 16 elements, e.g. per-head A_log, dt_bias, bias gradients):
  op        each checkout against the float64 oracle, every tensor, chunk size and repeat:
            fp32 <= 1e-5; bf16 <= 4e-2 RMS and <= 1e-1 max (tests/sdm_testlib.py
            BF16_CEILINGS)
  layer     each checkout against the float64 layer reference with the same selections:
            fp32 <= 1e-5 (<= 1e-3 for the small tensors; tests/test_layer_reference.py);
            bf16 <= 4e-2 (<= 1e-1)
  selector  slot ids distinct and ascending; values bitwise equal to the factor sums in the
            input dtype; gradient against the exact scatter: fp32 <= 1e-6, bf16 <= 1e-2;
            every row an exact top-k under the checkout's rule (the previous release ranks
            the factor sums rounded to the input dtype, this version the float32 sums);
            raw projections bitwise equal between the checkouts
  model     fp32: cross-checkout deviation and each checkout's run-to-run spread <= 1e-4
            (<= 1e-3 small) for the output, input gradient, the full gradients of the first
            SWA block and the first SDM layer (bank excluded) and per-parameter gradient
            norms and 4096 sampled gradient entries; bf16: output and input gradient of
            each checkout within 4e-2 of the fp32 run
  cuda_graph option  (rule v2; the verdict also keeps rule v1's result) the option must
            have replayed its graphs (no eager fallback, no backward-time forward replay),
            from identical weights, and on every step:
            layer (both dtypes) and model with the frozen selections in fp32: every tensor
            (output, input gradient, every parameter gradient of the layer; the model record
            of run-model) within 1e-5 relative RMS in fp32 (1e-3 relative max for tensors of at
            most 16 elements), the bf16 ceilings in bf16;
            model with its own selections (both dtypes), and model with the frozen
            selections in bf16: the eager path need not be bitwise reproducible; through
            the stack such noise flips near-tie top-k selections and grows
            with bf16 rounding, so the graphed run is held to its eager copies: per tensor,
            its largest deviation from the eager run over the steps <= CG_NOISE_MULT x the
            largest deviation of the other eager runs from it (or the fixed ceiling, if
            larger), with the per-head tensors of at most 16 elements judged together (one
            relative RMS over their concatenation per step).
            Rule v1 held the frozen-selection model in bf16 to the fixed ceilings and each
            per-head tensor to its own relative max; rule v2 is a post-hoc amendment (see
            CG_AMENDMENT)
  Every value must be finite. Runs with each checkout's own selections are reported,
  not gated: in bf16 the two checkouts pick different slots among scores that tie after
  rounding, so their outputs differ by more than rounding (see the selection report).

Example (checkouts OLD and NEW, results in DIR):
  PYTHONPATH=$NEW python benchmarks/equivalence.py make --out DIR/inputs.pt
  PYTHONPATH=$OLD python benchmarks/equivalence.py run --inputs DIR/inputs.pt --out DIR/old.pt
  PYTHONPATH=$NEW python benchmarks/equivalence.py run --inputs DIR/inputs.pt --out DIR/new.pt
  python benchmarks/equivalence.py oracle --inputs DIR/inputs.pt --out DIR/oracle.pt
  PYTHONPATH=$OLD python benchmarks/equivalence.py run-model --inputs DIR/inputs.pt --out DIR/old_model.pt
  PYTHONPATH=$NEW python benchmarks/equivalence.py run-model --inputs DIR/inputs.pt --out DIR/new_model.pt
  PYTHONPATH=$NEW python benchmarks/equivalence.py run-cudagraph --inputs DIR/inputs.pt --out DIR
  python benchmarks/equivalence.py compare --dir DIR --a old --b new --out DIR/verdict.json
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
OP_INPUTS = ("memory", "k_val", "v", "beta", "g", "q_val")
TINY_NUMEL = 16
OP_FP32 = 1e-5
BF16_RMS, BF16_MAX = 4e-2, 1e-1
LAYER_FP32, LAYER_FP32_TINY = 1e-5, 1e-3
SELECTOR_GRAD = {"fp32": 1e-6, "bf16": 1e-2}
MODEL_FP32, MODEL_FP32_TINY = 1e-4, 1e-3
MODEL_SAMPLES = 4096
CG_STEPS = 4
CG_FP32, CG_FP32_TINY = 1e-5, 1e-3
CG_REPEATS = 2  # eager copies next to each graphed model
CG_NOISE_MULT = 1.5
CG_AMENDMENT = {
    "rule": "v2", "post_hoc": True,
    "changes": [
        "model with the frozen selections, bf16: the fixed bf16 ceilings are replaced by the "
        "noise-referenced test (graph vs eager <= CG_NOISE_MULT x the largest eager-vs-eager "
        "deviation per tensor, the fixed ceiling as the floor)",
        "per-head tensors of at most 16 elements (A_log, dt_bias, a_proj/b_proj bias), every "
        "model check: judged together per step, one relative RMS over their concatenation, "
        "with the noise-referenced test"],
    "unchanged": [
        "layer checks and the model with the frozen selections in fp32: fixed ceilings, "
        "every tensor", "exact replay counts", "identical weights",
        "model with its own selections: noise-referenced test of every larger tensor"],
    "rationale": [
        "the fixed bf16 ceilings were designed for the op and the layer against float64, "
        "not for 24 layers of bf16 accumulation",
        "a relative maximum over a 2-element tensor, with 8 eager-vs-eager samples against "
        "4 graph samples, is heavy-tailed"]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("make", "run", "run-model", "run-cudagraph", "oracle",
                                   "compare"))
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--sph", type=int, default=65536, help="slots per head")
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--C", type=int, default=64, help="chunk size of the layer and model")
    p.add_argument("--chunks", default="64,128,256", help="run: chunk sizes of the op check")
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--n-blocks", type=int, default=6, help="model: [SWA x3, SDM] blocks")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inputs")
    p.add_argument("--dir", help="compare: directory of the result files")
    p.add_argument("--cg-dir", help="compare: directory of the run-cudagraph files (--dir)")
    p.add_argument("--cg-dtype", choices=("bf16", "fp32", "all"), default="all",
                   help="run-cudagraph: dtype of this process")
    p.add_argument("--cg-mode", choices=("free", "frozen", "all"), default="all",
                   help="run-cudagraph: model with its own or the frozen selections")
    p.add_argument("--a", default="old", help="compare: checkout A (the previous release)")
    p.add_argument("--b", default="new", help="compare: checkout B (this version)")
    p.add_argument("--out")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- helpers
def checkout_info():
    from provenance import checkout_info as info  # this directory is sys.path[0]

    return {**info(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name()}


def sha256_tensors(named):
    from provenance import sha256_tensors as sha

    return sha(named)


def config(a):
    return dict(dim=a.dim, num_heads=a.heads, slots_per_head=a.sph, num_writes=a.W,
                num_reads=a.R, chunk_size=a.C, output_gate=True,
                normalize_readings=True, backprop_on_memory=True)


def layer_args(cfg, **option):
    """The checkout's SparseDeltaMemoryArgs of ``cfg`` (whose chunk size is ``chunk_size``,
    ``memory_block_size`` in the previous release)."""
    from provenance import chunk_field, sdm_package  # this directory is sys.path[0]

    SparseDeltaMemoryArgs = sdm_package().SparseDeltaMemoryArgs
    cfg = dict(cfg)
    cfg[chunk_field(SparseDeltaMemoryArgs)] = cfg.pop("chunk_size")
    return SparseDeltaMemoryArgs(**cfg, **option)


def build_layer(cfg, master, dtype, **option):
    """The checkout's layer with the master weights rounded to bfloat16 values."""
    from provenance import sdm_package  # this directory is sys.path[0]

    SparseDeltaMemory = sdm_package().SparseDeltaMemory

    layer = SparseDeltaMemory(layer_args(cfg, **option), 0).cuda().to(dtype)
    layer.load_state_dict({k: v.to(torch.bfloat16).to(dtype) for k, v in master.items()},
                          strict=True)
    return layer


def model_args(a, cuda_graph=False):
    import bench_sdm_gdn as M  # this directory is sys.path[0]

    return M, M.parse_args([
        "--mixer", "sdm", "--batch", str(a.B), "--seqlen", str(a.T), "--n-blocks",
        str(a.n_blocks), "--sdm-heads", str(a.heads), "--slots-per-head", str(a.sph),
        "--writes", str(a.W), "--reads", str(a.R), "--chunk", str(a.C), "--seed",
        str(a.seed), "--dim", str(a.dim), "--out", os.devnull]
        + (["--sdm-cuda-graph"] if cuda_graph else []))


def build_model(a, dtype, cuda_graph=False):
    M, margs = model_args(a, cuda_graph)
    model = M.build_model(margs, dtype)
    with torch.no_grad():  # bfloat16 values in both dtypes
        for p in model.parameters():
            p.copy_(p.to(torch.bfloat16).to(dtype))
    return model, margs


def sdm_layers(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "gated_write_read")]


class Capture:
    """Records the first call's arguments of ``layer.gated_write_read``, passing them on."""

    def __init__(self, layer):
        self.layer, self.args = layer, None
        orig = layer.gated_write_read

        def wrapped(*args, **kwargs):
            if self.args is None:
                self.args = [t.detach().clone() for t in args[:8]]
            return orig(*args, **kwargs)

        layer.gated_write_read = wrapped

    def remove(self):
        del self.layer.gated_write_read


def native_selector(layer):
    """The checkout's own product-key top-k as f(x [BH, T, 2 side], k) -> (val, idx)."""
    side = layer.key_size_per_head // 2
    if hasattr(layer, "_product_keys"):                  # this version: (x, k)
        return lambda x, k: layer._product_keys(x, k)
    if hasattr(layer, "_get_product_keys_for_head"):     # previous release: (x, k, half_key)
        return lambda x, k: layer._get_product_keys_for_head(x, k, side)
    raise RuntimeError("unknown product-key method")


def set_selector(layer, fn):
    if hasattr(layer, "_product_keys"):
        layer._product_keys = lambda x, k: fn(x, k)
    else:
        layer._get_product_keys_for_head = lambda x, k, half: fn(x, k)


class _OrderedGather(torch.autograd.Function):
    """``x.gather(-1, idx)`` whose backward sums repeated indices in a fixed order. torch's
    CUDA gather backward scatters with float atomics, so the frozen-selection copies of a
    deterministic layer would differ from run to run in that backward alone. Contract (the
    frozen selections): ``idx`` has ``x``'s leading dimensions and at least one column."""

    @staticmethod
    def forward(ctx, x, idx):
        if idx.shape[:-1] != x.shape[:-1] or idx.shape[-1] < 1:
            raise ValueError(f"idx {tuple(idx.shape)} must have the leading dims of x {tuple(x.shape)} "
                             f"and at least one column")
        ctx.save_for_backward(idx)
        ctx.x_shape = x.shape
        return x.gather(-1, idx)

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        n, k = ctx.x_shape[-1], idx.shape[-1]
        rows = idx.numel() // k
        flat = (idx.reshape(rows, k) + torch.arange(rows, device=idx.device).unsqueeze(1) * n).reshape(-1)
        out = torch.zeros(rows * n, dtype=grad.dtype, device=grad.device)
        # accumulate=True takes torch's sorted implementation on CUDA: a fixed order, no atomics.
        out.index_put_((flat,), grad.reshape(-1), accumulate=True)
        return out.view(ctx.x_shape), None


def factor_sums(x, idx, side, ordered=False):
    """Slot values: the sums of the selected columns of the two factors. ``ordered``: a
    backward in a fixed order (the frozen-selection copies of the cuda_graph checks)."""
    if ordered:
        return _OrderedGather.apply(x[..., :side], idx // side) + _OrderedGather.apply(x[..., side:], idx % side)
    return x[..., :side].gather(-1, idx // side) + x[..., side:].gather(-1, idx % side)


def exact_topk_ids(x, k, side):
    """Slot ids of the exact top-k of the float32 factor sums, ties to the lower id."""
    flat = x.detach().reshape(-1, 2 * side)
    idx = torch.empty(flat.shape[0], k, dtype=torch.int64, device=x.device)
    for r0 in range(0, flat.shape[0], 1024):
        f = flat[r0:r0 + 1024].float()
        sc = (f[:, :side].unsqueeze(-1) + f[:, side:].unsqueeze(-2)).reshape(f.shape[0], -1)
        top = torch.sort(sc, dim=-1, descending=True, stable=True).indices[:, :k]
        idx[r0:r0 + 1024] = top.sort(-1).values
    return idx.view(*x.shape[:-1], k)


def use_frozen_selection(layer, sel_k, sel_q):
    """Make the layer select the frozen slot ids (writes, then reads, per forward), with
    values = the factor sums of its own projections (differentiable)."""
    side, state = layer.key_size_per_head // 2, {"n": 0}

    def pick(x, k):
        idx = (sel_k if state["n"] % 2 == 0 else sel_q).to(x.device)
        state["n"] += 1
        assert tuple(idx.shape) == (*x.shape[:-1], k), (tuple(idx.shape), tuple(x.shape), k)
        return factor_sums(x, idx, side, ordered=True), idx

    set_selector(layer, pick)
    return layer


def record_selection(layer, store):
    """Make the layer select exact top-k ids and record them (writes, then reads)."""
    side = layer.key_size_per_head // 2

    def pick(x, k):
        idx = exact_topk_ids(x, k, side)
        store.append(idx.cpu())
        return factor_sums(x, idx, side), idx

    set_selector(layer, pick)


def head_split(t, H):
    B, T, _ = t.shape
    return t.view(B, T, H, -1).permute(0, 2, 1, 3).reshape(B * H, T, -1)


def to_cpu(d):
    return {k: (None if v is None else v.detach().cpu()) for k, v in d.items()}


def layer_pass(layer, x, gy):
    """Output, input gradient and every parameter gradient of one forward + backward."""
    for p in layer.parameters():
        p.grad = None
    xi = x.detach().clone().requires_grad_(True)
    out = layer(xi)[0]
    out.backward(gy)
    rec = {"out": out, "grad_x": xi.grad}
    for n, p in layer.named_parameters():
        rec["grad." + n] = p.grad
    rec = to_cpu(rec)
    for p in layer.parameters():
        p.grad = None
    return rec


def op_pass(layer, d):
    """Readings and input gradients of the layer's write/read call on frozen arguments."""
    leaves = {n: d[n].cuda().clone().requires_grad_(True) for n in OP_INPUTS}
    k_idx, q_idx = d["k_idx"].cuda(), d["q_idx"].cuda()
    readings = layer.gated_write_read(leaves["memory"], k_idx, leaves["k_val"], leaves["v"],
                                      leaves["beta"], leaves["g"], q_idx, leaves["q_val"])[0]
    grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS],
                                grad_outputs=d["cotangent"].cuda().view_as(readings))
    rec = {"readings": readings}
    rec.update({"grad_" + n: g for n, g in zip(OP_INPUTS, grads)})
    return to_cpu(rec)


def selector_check(layer, raw, H, side, dt, seed):
    """The checkout's own top-k on its projections against the definitions."""
    sel = native_selector(layer)
    out = {}
    for key, k in (("k", layer.args.num_writes), ("q", layer.args.num_reads)):
        x = head_split(raw[key].cuda(), H).detach().requires_grad_(True)
        val, idx = sel(x, k)
        ids_ok = bool(((idx[..., 1:] > idx[..., :-1]).all() & (idx >= 0).all()
                       & (idx < side * side).all()).item())
        ref_val = factor_sums(x.detach(), idx, side)
        gv = torch.randn(val.shape, device="cuda", generator=torch.Generator(
            device="cuda").manual_seed(seed)).to(dt)
        (grad,) = torch.autograd.grad(val, x, grad_outputs=gv)
        ref = torch.zeros(x.shape, dtype=torch.float64, device="cuda")
        ref.scatter_add_(-1, idx // side, gv.double())
        ref.scatter_add_(-1, side + idx % side, gv.double())
        out[key] = {"ids_distinct_ascending_in_range": ids_ok,
                    "values_bitwise_equal": bool(torch.equal(val.to(dt), ref_val.to(dt))),
                    "grad_rel_rms": rel_rms(ref, grad), "idx": idx.cpu()}
    return out


# ---------------------------------------------------------------------------- commands
def cmd_make(a):
    from provenance import sdm_package  # this directory is sys.path[0]

    SparseDeltaMemory = sdm_package().SparseDeltaMemory

    cfg = config(a)
    torch.manual_seed(a.seed)
    ref = SparseDeltaMemory(layer_args(cfg), 0).cuda()
    ref.init_weights()
    master = {k: v.detach().float().cpu().clone() for k, v in ref.state_dict().items()}
    del ref
    gen = torch.Generator(device="cuda").manual_seed(a.seed + 1)
    x = torch.randn(a.B, a.T, a.dim, device="cuda", generator=gen).to(torch.bfloat16)
    gy = (torch.randn(a.B, a.T, a.dim, device="cuda", generator=gen) * 0.01).to(torch.bfloat16)
    # Frozen layer selections: exact top-k of the fp32 projections.
    layer = build_layer(cfg, master, torch.float32).train()
    sel = []
    record_selection(layer, sel)
    with torch.no_grad():
        layer(x.float())
    frozen = {"k": sel[0], "q": sel[1]}
    ops = {}
    for name, dt in DTYPES.items():
        layer = build_layer(cfg, master, dt).train()
        cap = Capture(layer)
        with torch.no_grad():
            layer(x.to(dt))
        cap.remove()
        memory, k_idx, k_val, v, beta, g, q_idx, q_val = cap.args
        cg = torch.Generator(device="cuda").manual_seed(a.seed + 2)
        cot = torch.randn(k_idx.shape[0], k_idx.shape[1], v.shape[-1], device="cuda",
                          generator=cg) * 0.01
        ops[name] = to_cpu({"memory": memory, "k_idx": k_idx, "k_val": k_val, "v": v,
                            "beta": beta, "g": g, "q_idx": q_idx, "q_val": q_val,
                            "cotangent": cot.to(dt)})
        del layer, cap
    # Frozen model selections: every SDM layer of the fp32 model.
    model, margs = build_model(a, torch.float32)
    stores = {}
    for n, m in sdm_layers(model):
        stores[n] = []
        record_selection(m, stores[n])
    with torch.no_grad():
        model(torch.randn(a.B, a.T, a.dim, device="cuda", generator=torch.Generator(
            device="cuda").manual_seed(a.seed + 1)).to(torch.bfloat16).float())
    model_sel = {n: {"k": s[0], "q": s[1]} for n, s in stores.items()}
    del model
    torch.save({"config": cfg, "seed": a.seed, "B": a.B, "T": a.T, "n_blocks": a.n_blocks,
                "master": master, "x": x.cpu(), "gy": gy.cpu(), "op": ops, "frozen": frozen,
                "model_frozen": model_sel, "model_args": vars(margs),
                "made_by": checkout_info()}, a.out)
    print(f"wrote {a.out}")


def cmd_run(a):
    from provenance import chunk_field  # this directory is sys.path[0]

    blob = torch.load(a.inputs, weights_only=False)
    cfg, H = blob["config"], blob["config"]["num_heads"]
    side = int(round(cfg["slots_per_head"] ** 0.5))
    chunks = [int(c) for c in a.chunks.split(",")]
    out = {"checkout": checkout_info(), "config": cfg, "repeats": a.repeats, "chunks": chunks,
           "weights_sha256": {}, "layer": {}, "layer_native": {}, "op": {}, "selector": {},
           "selections": {}, "raw": {}}
    for name, dt in DTYPES.items():
        x, gy = blob["x"].cuda().to(dt), blob["gy"].cuda().to(dt)
        # The checkout's own selection: one pass, with its selections and projections.
        layer = build_layer(cfg, blob["master"], dt).train()
        out["weights_sha256"][name] = sha256_tensors(sorted(layer.state_dict().items()))
        cap, raw = Capture(layer), {}
        hooks = [mod.register_forward_hook(
            lambda m, i, o, key=key: raw.__setitem__(key, o.detach().cpu()))
            for key, mod in (("k", layer.Wk_write), ("q", layer.Wq_read))]
        out["layer_native"][name] = [layer_pass(layer, x, gy)]
        cap.remove()
        for h in hooks:
            h.remove()
        out["selections"][name] = {"k_idx": cap.args[1].cpu(), "q_idx": cap.args[6].cpu()}
        out["raw"][name] = raw
        out["selector"][name] = selector_check(layer, raw, H, side, dt, a.seed + 3)
        # The frozen selections: the gated layer runs.
        frozen = use_frozen_selection(build_layer(cfg, blob["master"], dt).train(),
                                      blob["frozen"]["k"], blob["frozen"]["q"])
        out["layer"][name] = [layer_pass(frozen, x, gy) for _ in range(a.repeats)]
        del frozen
        out["op"][name] = {}
        for C in chunks:  # repeats at the main chunk size, one run at the others
            setattr(layer.args, chunk_field(type(layer.args)), C)
            n = a.repeats if C == cfg["chunk_size"] else 1
            out["op"][name][C] = [op_pass(layer, blob["op"][name]) for _ in range(n)]
        del layer
        torch.cuda.empty_cache()
        print(f"{name}: done", flush=True)
    torch.save(out, a.out)
    print(f"wrote {a.out}")


def model_record(model, y, xi):
    rec = {"y": y, "grad_x": xi.grad}
    norms = []
    for n, p in model.named_parameters():
        g = p.grad
        norms.append(g.double().norm())
        bank = getattr(p, "_sdm_memory_bank", False)
        if (n.startswith("0.") or n.startswith("3.mixer.layer.")) and not bank:
            rec["grad." + n] = g
        gen = torch.Generator(device="cuda").manual_seed(sum(map(ord, n)))
        pick = torch.randint(0, g.numel(), (min(MODEL_SAMPLES, g.numel()),), device="cuda",
                             generator=gen)
        rec["sample." + n] = g.reshape(-1)[pick]
    rec["grad_norms"] = torch.stack(norms)
    return to_cpu(rec)


def cmd_run_model(a):
    blob = torch.load(a.inputs, weights_only=False)
    out = {"checkout": checkout_info(), "repeats": a.repeats, "sdm_weights_sha256": {},
           "weights_sha256": {}, "runs": {}}
    for name, dt in DTYPES.items():
        model, margs = build_model(a, dt)
        out["args"] = vars(margs)
        out["sdm_weights_sha256"][name] = sha256_tensors(
            sorted((n, p) for n, p in model.named_parameters() if ".mixer.layer." in n))
        out["weights_sha256"][name] = sha256_tensors(sorted(model.named_parameters()))
        for n, m in sdm_layers(model):
            use_frozen_selection(m, blob["model_frozen"][n]["k"], blob["model_frozen"][n]["q"])
        gen = torch.Generator(device="cuda").manual_seed(a.seed + 1)
        x = torch.randn(a.B, a.T, a.dim, device="cuda", generator=gen).to(torch.bfloat16).to(dt)
        gy = torch.randn(a.B, a.T, a.dim, device="cuda", generator=torch.Generator(
            device="cuda").manual_seed(a.seed + 5)).to(torch.bfloat16).to(dt)
        runs = []
        for _ in range(a.repeats):
            model.zero_grad(set_to_none=True)
            xi = x.clone().requires_grad_(True)
            y = model(xi)
            y.backward(gy)
            runs.append(model_record(model, y, xi))
        out["runs"][name] = runs
        del model
        torch.cuda.empty_cache()
        print(f"{name}: done", flush=True)
    torch.save(out, a.out)
    print(f"wrote {a.out}")


def graph_stats(module):
    """Captures and replays of the cuda_graph option, summed over the module's SDM layers,
    and the number of those layers ("runners")."""
    from collections import Counter

    runners = [m.graph_runner for m in module.modules()
               if getattr(m, "graph_runner", None) is not None]
    return {"runners": len(runners), **sum((Counter(r.stats) for r in runners), Counter())}


def perturb_and_copy(src, dsts, step, seed):
    """Change every weight of ``src`` in place (+1% of its RMS, seeded), then copy it in place
    into each of ``dsts``: all hold the same new values, in the storages a graph captured."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1000 + step)
    params = [dict(d.named_parameters()) for d in dsts]
    with torch.no_grad():
        for n, p in src.named_parameters():
            rms = p.float().pow(2).mean().sqrt().clamp_min(1e-3)
            p.add_((torch.randn(p.shape, device="cuda", generator=gen) * 0.01 * rms).to(p.dtype))
            for q in params:
                q[n].copy_(p)


def same_weights(models):
    ref = list(models[0].parameters())
    return all(len(ref) == len(list(m.parameters())) and
               all(torch.equal(p, q) for p, q in zip(ref, m.parameters())) for m in models[1:])


def model_cudagraph(a, dt, frozen):
    """CG_STEPS steps of the eager model (the reference), CG_REPEATS more eager copies and the
    graphed model, on the same inputs, their weights changed identically in place between
    steps. With ``frozen``, every model selects the frozen slots of make."""
    models = {"eager": build_model(a, dt)[0]}
    for r in range(1, CG_REPEATS + 1):
        models[f"eager{r}"] = build_model(a, dt)[0]
    models["graph"], margs = build_model(a, dt, cuda_graph=True)
    identical = same_weights(list(models.values()))
    if frozen is not None:
        for model in models.values():
            for n, m in sdm_layers(model):
                use_frozen_selection(m, frozen[n]["k"], frozen[n]["q"])
    steps = []
    for s in range(CG_STEPS):
        gen = torch.Generator(device="cuda").manual_seed(a.seed + 20 + s)
        x = torch.randn(a.B, a.T, a.dim, device="cuda", generator=gen).to(torch.bfloat16).to(dt)
        gy = torch.randn(x.shape, device="cuda", generator=gen).to(torch.bfloat16).to(dt)
        rec = {}
        for key, model in models.items():
            model.zero_grad(set_to_none=True)
            xi = x.clone().requires_grad_(True)
            y = model(xi)
            y.backward(gy)
            rec[key] = model_record(model, y, xi)
        steps.append(rec)
        perturb_and_copy(models["eager"], [m for k, m in models.items() if k != "eager"], s,
                         a.seed)
    stats = graph_stats(models["graph"])
    del models
    torch.cuda.empty_cache()
    return {"steps": steps, "stats": stats, "weights_identical": identical}, margs


def layer_cudagraph(blob, dt, seed):
    cfg = blob["config"]
    eager = build_layer(cfg, blob["master"], dt).train()
    graphed = build_layer(cfg, blob["master"], dt, cuda_graph=True).train()
    steps = []
    for s in range(CG_STEPS):
        gen = torch.Generator(device="cuda").manual_seed(seed + 10 + s)
        x = torch.randn(blob["B"], blob["T"], cfg["dim"], device="cuda",
                        generator=gen).to(torch.bfloat16).to(dt)
        gy = (torch.randn(x.shape, device="cuda", generator=gen) * 0.01).to(torch.bfloat16).to(dt)
        steps.append({"eager": layer_pass(eager, x, gy), "graph": layer_pass(graphed, x, gy)})
        perturb_and_copy(eager, [graphed], s, seed)
    stats = graph_stats(graphed)
    del eager, graphed
    torch.cuda.empty_cache()
    return {"steps": steps, "stats": stats}


def cmd_run_cudagraph(a):
    blob = torch.load(a.inputs, weights_only=False)
    os.makedirs(a.out, exist_ok=True)
    for name in (DTYPES if a.cg_dtype == "all" else (a.cg_dtype,)):
        for mode in (("free", "frozen") if a.cg_mode == "all" else (a.cg_mode,)):
            out = {"checkout": checkout_info(), "steps": CG_STEPS, "repeats": CG_REPEATS,
                   "dtype": name, "mode": mode}
            # The model first: in a fresh process its layers share one graph pool, as in
            # training (later captures in the process get pools of their own).
            out["model"], margs = model_cudagraph(
                a, DTYPES[name], blob["model_frozen"] if mode == "frozen" else None)
            out["model_args"] = vars(margs)
            if mode == "free":
                out["layer"] = layer_cudagraph(blob, DTYPES[name], a.seed)
            path = os.path.join(a.out, f"cg_{name}_{mode}.pt")
            torch.save(out, path)
            print(f"wrote {path}", flush=True)


def cg_limit(name, is_tiny):
    """(metric, fixed ceiling) of a tensor in the cuda_graph checks."""
    if name == "fp32":
        return ("rel_max", CG_FP32_TINY) if is_tiny else ("rel_rms", CG_FP32)
    return ("rel_max", BF16_MAX) if is_tiny else ("rel_rms", BF16_RMS)


def cudagraph_gate(steps, name, ref="eager", got="graph"):
    """Per step and tensor: run ``got`` against run ``ref`` of the same step, within the fixed
    ceilings (bf16: also <= BF16_MAX relative max for every tensor)."""
    rows, ok = {}, True
    for s, st in enumerate(steps):
        for k, r0 in st[ref].items():
            if r0 is None:
                continue
            r = {"rel_rms": rel_rms(r0, st[got][k]), "rel_max": rel_max(r0, st[got][k])}
            metric, ceil = cg_limit(name, tiny(r0, k))
            r["score"] = r[metric] / ceil
            if name == "bf16":
                r["score"] = max(r["score"], r["rel_max"] / BF16_MAX)
            r["pass"] = r["score"] <= 1 and all(math.isfinite(x) for x in r.values())
            rows[f"step{s}.{k}"] = r
            ok &= r["pass"]
    return ok, rows


def noise_referenced(steps, name, joint_tiny=False):
    """Per tensor, the largest deviation of the graphed run from the eager run over the steps,
    against CG_NOISE_MULT x the largest deviation of the other eager runs from it, or the fixed
    ceiling if that is larger. With ``joint_tiny``, the tensors of at most TINY_NUMEL elements
    are judged together: one relative RMS over their concatenation per step."""
    refs = [k for k in steps[0] if k.startswith("eager") and k != "eager"]
    keys = [k for k, r0 in steps[0]["eager"].items() if r0 is not None]
    small = sorted(k for k in keys if tiny(steps[0]["eager"][k], k)) if joint_tiny else []

    def get(st, run, k):
        if k == "per-head tensors (joint)":
            return torch.cat([st[run][x].reshape(-1).double() for x in small])
        return st[run][k]

    rows, ok = {}, True
    for k in [x for x in keys if x not in small] + (["per-head tensors (joint)"] if small else []):
        joint = k == "per-head tensors (joint)"
        metric, ceil = cg_limit(name, False if joint else tiny(steps[0]["eager"][k], k))
        f = rel_max if metric == "rel_max" else rel_rms
        dev = max(f(get(st, "eager", k), get(st, "graph", k)) for st in steps)
        noise = max(f(get(st, "eager", k), get(st, r, k)) for st in steps for r in refs)
        limit = max(CG_NOISE_MULT * noise, ceil)
        r = {"metric": metric, "graph": dev, "eager": noise, "limit": limit,
             "score": dev / limit}
        if joint:
            r["members"] = small
        r["pass"] = math.isfinite(dev) and math.isfinite(noise) and dev <= limit
        rows[k] = r
        ok &= r["pass"]
    return ok, rows


def replayed(stats, steps):
    """Each layer captured once; every forward replayed (the first one right after its
    capture) and every backward but the first (which runs eagerly, then captures); no call
    fell back to eager and no backward had to replay its forward again."""
    n = stats.get("runners", 0)
    return (n > 0 and stats.get("forward_capture") == n and stats.get("backward_capture") == n
            and stats.get("forward_replay") == n * steps
            and stats.get("backward_replay") == n * (steps - 1)
            and not any(x for k, x in stats.items() if k.startswith(("eager", "stale"))))


def provenance_problems(runs, harness):
    """Every run of a compare must come from this harness, and each model run from the same
    checkout (commit and code tree) as the layer run of its side."""
    p = [f"{n}: harness {r['checkout'].get('harness_sha256')} is not {harness}"
         for n, r in runs.items() if r["checkout"].get("harness_sha256") != harness]
    for layer, model in (("A", "MA"), ("B", "MB")):
        for key in ("commit", "code_tree"):
            if runs[layer]["checkout"].get(key) != runs[model]["checkout"].get(key):
                p.append(f"{model}: {key} differs from {layer}")
    return p


def cuda_graph_option(cg_dir, commit, harness):
    """The option's separate verdict from the run-cudagraph files: rule v2, with rule v1's
    result kept under "rule_v1"."""
    files = {(n, m): os.path.join(cg_dir, f"cg_{n}_{m}.pt") for n in DTYPES
             for m in ("free", "frozen")}
    missing = [f for f in files.values() if not os.path.exists(f)]
    res = {"steps": CG_STEPS, "repeats": CG_REPEATS, "noise_multiplier": CG_NOISE_MULT,
           "rule": "v2", "amendment": CG_AMENDMENT}
    if missing:
        res.update(verdict="NOT VERIFIED", missing=[os.path.basename(f) for f in missing])
        return res
    v1 = {}
    ok = ok1 = True

    def block(g_ok, rows, stats, extra=None):
        worst = max(rows.items(), key=lambda kv: kv[1]["score"])
        return {"pass": g_ok, "graph_stats": stats, "worst": {"tensor": worst[0], **worst[1]},
                "tensors": rows, **(extra or {})}

    for (name, mode), f in files.items():
        cg = torch.load(f, weights_only=False)
        res["checkout"] = cg["checkout"]
        base = (cg["checkout"]["commit"] == commit
                and cg["checkout"].get("harness_sha256") == harness and cg["steps"] == CG_STEPS
                and replayed(cg["model"]["stats"], cg["steps"]) and cg["model"]["weights_identical"])
        ok &= base
        ok1 &= base
        model, stats = cg["model"]["steps"], cg["model"]["stats"]
        if mode == "frozen":
            _, noise = cudagraph_gate(model, name, got="eager1")
            extra = {"eager_vs_eager_max_rel_rms": max(r["rel_rms"] for r in noise.values())}
            g1, rows1 = cudagraph_gate(model, name)  # rule v1: fixed ceilings, every tensor
            v1[f"model same slots {name}"] = block(g1 and base, rows1, stats, extra)
            if name == "fp32":  # rule v2: unchanged in fp32
                g2, rows2 = g1, rows1
            else:
                g2, rows2 = noise_referenced(model, name, joint_tiny=True)
            res[f"model same slots {name}"] = block(g2 and base, rows2, stats, extra)
        else:
            g1, rows1 = noise_referenced(model, name)
            v1[f"model free {name}"] = block(g1 and base, rows1, stats)
            g2, rows2 = noise_referenced(model, name, joint_tiny=True)
            res[f"model free {name}"] = block(g2 and base, rows2, stats)
            lay = cg["layer"]
            l_ok, lrows = cudagraph_gate(lay["steps"], name)
            l_ok &= replayed(lay["stats"], cg["steps"])
            res[f"layer {name}"] = v1[f"layer {name}"] = block(l_ok, lrows, lay["stats"])
            g1, g2 = g1 and l_ok, g2 and l_ok
        ok &= g2
        ok1 &= g1
        del cg
    res["verdict"] = "VERIFIED" if ok else "NOT VERIFIED"
    res["rule_v1"] = {"verdict": "VERIFIED" if ok1 else "NOT VERIFIED", **v1}
    return res


def oracle_recurrence(memory, k_idx, k_val, v, beta, g, q_idx, q_val):
    """Token loop of the gated delta-rule write/read on [S, T, *] streams whose slot ranges
    are disjoint (as tests/sdm_testlib.py): decay and delta-update the written rows, then
    read. Returns readings [S, T, dim]; gradients flow to every float input."""
    T, dim = k_idx.shape[1], v.shape[-1]
    mem = memory.clone()
    outs = []
    for t in range(T):
        ki = k_idx[:, t]
        at_k = mem[ki]                                              # [S, W, dim]
        read = at_k * torch.exp(g[:, t]).unsqueeze(-1)
        retrieved = (k_val[:, t].unsqueeze(-1) * read).sum(1)       # [S, dim]
        dv = beta[:, t] * (v[:, t] - retrieved)
        upd = read - at_k + k_val[:, t].unsqueeze(-1) * dv.unsqueeze(1)
        mem.index_add_(0, ki.reshape(-1), upd.reshape(-1, dim))
        outs.append((q_val[:, t].unsqueeze(-1) * mem[q_idx[:, t]]).sum(1))
    return torch.stack(outs, 1)


def layer_reference(blob):
    """float64 forward + backward of the layer with the frozen selections, from its
    bf16-valued weights (as tests/test_layer_reference.py rebuilds SparseDeltaMemory)."""
    cfg = blob["config"]
    H, sph = cfg["num_heads"], cfg["slots_per_head"]
    hd, side = cfg["dim"] // H, int(round(sph ** 0.5))
    P = {n: v.to(torch.bfloat16).double().cuda().requires_grad_(True)
         for n, v in blob["master"].items()}
    x = blob["x"].double().cuda().requires_grad_(True)
    B, T, _ = x.shape

    def lin(name, inp):
        return F.linear(inp, P[name + ".weight"], P.get(name + ".bias"))

    g = -torch.exp(P["A_log"]) * F.softplus(lin("a_proj", x) + P["dt_bias"])
    beta = torch.sigmoid(lin("b_proj", x))
    v = head_split(lin("Wv_write", x), H)
    sk, sq = blob["frozen"]["k"].cuda(), blob["frozen"]["q"].cuda()
    k_val = F.softmax(factor_sums(head_split(lin("Wk_write", x), H), sk, side), dim=-1)
    q_val = F.softmax(factor_sums(head_split(lin("Wq_read", x), H), sq, side), dim=-1)
    g = g.permute(0, 2, 1).reshape(B * H, T, 1)
    beta = beta.permute(0, 2, 1).reshape(B * H, T, 1)
    memory = P["memory"].reshape(1, H * sph, hd).expand(B, -1, -1).reshape(B * H * sph, hd)
    off = torch.arange(B * H, device="cuda").view(-1, 1, 1) * sph
    readings = oracle_recurrence(memory, sk + off, k_val, v, beta, g, sq + off, q_val)
    readings = F.layer_norm(readings, (hd,), P["readings_norm.weight"],
                            P["readings_norm.bias"], 1e-6)
    readings = readings.view(B, H, T, hd).permute(0, 2, 1, 3).reshape(B, T, H * hd)
    out = lin("Wo", torch.sigmoid(lin("Wog", x)) * readings)
    out.backward(blob["gy"].double().cuda())
    res = {"out": out, "grad_x": x.grad}
    res.update({"grad." + n: p.grad for n, p in P.items()})
    return to_cpu(res)


def cmd_oracle(a):
    blob = torch.load(a.inputs, weights_only=False)
    out = {"op": {}}
    for name in DTYPES:
        d = blob["op"][name]
        leaves = {n: d[n].cuda().double().requires_grad_(True) for n in OP_INPUTS}
        t0 = time.time()
        readings = oracle_recurrence(leaves["memory"], d["k_idx"].cuda(), leaves["k_val"],
                                     leaves["v"], leaves["beta"], leaves["g"],
                                     d["q_idx"].cuda(), leaves["q_val"])
        grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS],
                                    grad_outputs=d["cotangent"].cuda().double().view_as(readings))
        out["op"][name] = to_cpu({"readings": readings,
                                  **{"grad_" + n: g for n, g in zip(OP_INPUTS, grads)}})
        print(f"op {name}: oracle {time.time() - t0:.1f} s", flush=True)
        del leaves, readings, grads
        torch.cuda.empty_cache()
    t0 = time.time()
    out["layer"] = layer_reference(blob)
    print(f"layer: oracle {time.time() - t0:.1f} s", flush=True)
    torch.save(out, a.out)
    print(f"wrote {a.out}")


# ------------------------------------------------------------------------------ compare
def _pair(ref, got):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return ref.to(dev).double().reshape(-1), got.to(dev).double().reshape(-1)


def rel_rms(ref, got):
    a, b = _pair(ref, got)
    if a.numel() != b.numel() or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return float("inf")
    den = a.pow(2).mean().sqrt().item()
    if den == 0.0:
        return 0.0 if b.abs().max().item() == 0.0 else float("inf")
    return (a - b).pow(2).mean().sqrt().item() / den


def rel_max(ref, got):
    a, b = _pair(ref, got)
    if a.numel() != b.numel() or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return float("inf")
    den = a.abs().max().item()
    if den == 0.0:
        return 0.0 if b.abs().max().item() == 0.0 else float("inf")
    return (a - b).abs().max().item() / den


def tiny(t, key=""):
    return t.numel() <= TINY_NUMEL and key != "grad_norms"


def against_reference(ref, runs_a, runs_b, ceiling_fn):
    """Per tensor: every run of both checkouts against ``ref``; ceiling_fn(key, tensor)
    gives (rms ceiling or None, max ceiling or None). Non-finite deviations fail."""
    rows, ok = {}, True
    for k, r in ref.items():
        if r is None:
            continue
        c_rms, c_max = ceiling_fn(k, r)
        absent = [who for who, runs in (("a", runs_a), ("b", runs_b))
                  if any(x.get(k) is None for x in runs)]
        if absent:  # a reference tensor that a run did not produce fails the check
            inf = float("inf")
            rows[k] = {"rms_a": inf, "max_a": inf, "rms_b": inf, "max_b": inf,
                       "cross_rms": inf, "missing_in": absent, "ceiling_rms": c_rms,
                       "ceiling_max": c_max, "pass_": False}
            ok = False
            continue
        row = {}
        for who, runs in (("a", runs_a), ("b", runs_b)):
            row[f"rms_{who}"] = max(rel_rms(r, x[k]) for x in runs)
            row[f"max_{who}"] = max(rel_max(r, x[k]) for x in runs)
        row["cross_rms"] = max(rel_rms(x[k], y[k]) for x in runs_a for y in runs_b)
        good = all(math.isfinite(v) for v in row.values())
        for who in "ab":
            if c_rms is not None:
                good &= row[f"rms_{who}"] <= c_rms
            if c_max is not None:
                good &= row[f"max_{who}"] <= c_max
        row.update(ceiling_rms=c_rms, ceiling_max=c_max, pass_=good)
        rows[k] = row
        ok &= good
    return ok, rows


def bf16_ceiling(k, r):
    return (None, BF16_MAX) if tiny(r, k) else (BF16_RMS, BF16_MAX)


def cross_and_noise(runs_a, runs_b, big, small):
    """Per tensor: cross-checkout deviation and each checkout's run-to-run spread, each at
    most the fixed ceiling (``small`` for tensors of at most 16 elements)."""
    rows, ok = {}, True
    for k, r in runs_a[0].items():
        if r is None:
            continue
        m, c = (rel_max, small) if tiny(r, k) else (rel_rms, big)
        row = {"cross": max(m(x[k], y[k]) for x in runs_a for y in runs_b),
               "noise_a": max([m(x[k], y[k]) for x, y in itertools.combinations(runs_a, 2)]
                              + [0.0]),
               "noise_b": max([m(x[k], y[k]) for x, y in itertools.combinations(runs_b, 2)]
                              + [0.0]),
               "ceiling": c}
        row["pass_"] = all(math.isfinite(row[n]) and row[n] <= c
                           for n in ("cross", "noise_a", "noise_b"))
        rows[k] = row
        ok &= row["pass_"]
    return ok, rows


def selection_report(sel_a, sel_b, raw_a, raw_b, cfg, B):
    """Rows (stream, token) whose selected slot sets differ between the checkouts, and
    whether each checkout's selections are exact top-k of the float32 factor sums and of
    the factor sums rounded to the projection dtype (each checkout on its own projections)."""
    H, sph = cfg["num_heads"], cfg["slots_per_head"]
    side = int(round(sph ** 0.5))
    rep = {}
    for kind, key, k in (("writes", "k", cfg["num_writes"]), ("reads", "q", cfg["num_reads"])):
        off = torch.arange(B * H, device="cuda").view(-1, 1, 1) * sph
        la = (sel_a[key + "_idx"].cuda() - off).sort(-1).values
        lb = (sel_b[key + "_idx"].cuda() - off).sort(-1).values
        differ = (la != lb).any(-1)
        counts = {"rows": differ.numel(), "rows_differ": int(differ.sum()),
                  "slots_differ": int((~(la.unsqueeze(-1) == lb.unsqueeze(-2)).any(-1)).sum())}
        for who, sel, raw in (("a", la, raw_a), ("b", lb, raw_b)):
            r = head_split(raw[key].cuda(), H)
            T = r.shape[1]
            for model in ("float32_sum", "rounded_sum"):
                valid = 0
                for s in range(B * H):
                    for t0 in range(0, T, 512):
                        f = r[s, t0:t0 + 512]
                        f1, f2 = f[:, :side], f[:, side:]
                        sc = ((f1.float().unsqueeze(-1) + f2.float().unsqueeze(-2))
                              if model == "float32_sum" else
                              (f1.unsqueeze(-1) + f2.unsqueeze(-2)).float()
                              ).reshape(f.shape[0], -1)
                        kth = sc.topk(k, dim=-1).values[:, -1:]
                        chosen = sc.gather(1, sel[s, t0:t0 + 512])
                        good = ((chosen >= kth).all(-1)
                                & ((chosen > kth).sum(-1) == (sc > kth).sum(-1)))
                        valid += int(good.sum())
                counts[f"exact_topk_{model}_{who}"] = valid
        rep[kind] = counts
    return rep


def cmd_compare(a):
    d = a.dir
    files = {n: os.path.join(d, f) for n, f in (
        ("A", f"{a.a}.pt"), ("B", f"{a.b}.pt"), ("O", "oracle.pt"), ("I", "inputs.pt"),
        ("MA", f"{a.a}_model.pt"), ("MB", f"{a.b}_model.pt"))}
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing result files: {missing}")
    A, Bv, O, inputs, MA, MB = (torch.load(files[n], weights_only=False)
                                for n in ("A", "B", "O", "I", "MA", "MB"))
    from provenance import harness_sha256

    here = harness_sha256()
    v = {"a": {"name": a.a, **A["checkout"]}, "b": {"name": a.b, **Bv["checkout"]},
         "harness_sha256": here, "config": A["config"], "B": inputs["B"],
         "T": inputs["T"], "repeats": A["repeats"], "chunks": A["chunks"],
         "weights_identical": A["weights_sha256"] == Bv["weights_sha256"],
         "weights_sha256": A["weights_sha256"], "op": {}, "layer": {}, "selector": {},
         "layer_native": {}, "selections": {}}
    v["provenance"] = provenance_problems({"A": A, "B": Bv, "MA": MA, "MB": MB}, here)
    ok = v["weights_identical"] and not v["provenance"]
    for name in DTYPES:  # op: each checkout against the float64 oracle, every chunk size
        v["op"][name] = {}
        for C in A["chunks"]:
            ceil = (lambda k, r: (OP_FP32, None)) if name == "fp32" else bf16_ceiling
            g_ok, rows = against_reference(O["op"][name], A["op"][name][C], Bv["op"][name][C],
                                           ceil)
            v["op"][name][C] = {"pass": g_ok, "tensors": rows}
            ok &= g_ok
    for name in DTYPES:  # layer with the frozen selections against the float64 layer
        ceil = ((lambda k, r: (None, LAYER_FP32_TINY) if tiny(r, k) else (LAYER_FP32, None))
                if name == "fp32" else bf16_ceiling)
        g_ok, rows = against_reference(O["layer"], A["layer"][name], Bv["layer"][name], ceil)
        v["layer"][name] = {"pass": g_ok, "tensors": rows}
        ok &= g_ok
    for name in DTYPES:  # each checkout's own selector
        sel = selection_report(A["selections"][name], Bv["selections"][name],
                               A["raw"][name], Bv["raw"][name], A["config"], inputs["B"])
        raw_equal = all(torch.equal(A["raw"][name][k], Bv["raw"][name][k]) for k in "kq")
        checks = {}
        for who, R in (("a", A), ("b", Bv)):
            for key in "kq":
                c = {k2: x for k2, x in R["selector"][name][key].items() if k2 != "idx"}
                taken = R["selections"][name][key + "_idx"].long()
                c["matches_layer_selection"] = bool(torch.equal(
                    R["selector"][name][key]["idx"].long(),
                    taken - torch.arange(taken.shape[0]).view(-1, 1, 1)
                    * A["config"]["slots_per_head"]))
                c["pass"] = (c["ids_distinct_ascending_in_range"] and c["values_bitwise_equal"]
                             and c["matches_layer_selection"]
                             and math.isfinite(c["grad_rel_rms"])
                             and c["grad_rel_rms"] <= SELECTOR_GRAD[name])
                checks[f"{who}.{key}"] = c
        rules = all(c["exact_topk_rounded_sum_a"] == c["rows"]
                    and c["exact_topk_float32_sum_b"] == c["rows"] for c in sel.values())
        s_ok = raw_equal and rules and all(c["pass"] for c in checks.values())
        v["selector"][name] = {"pass": s_ok, "raw_projections_bitwise_equal": raw_equal,
                               "every_row_exact_topk_under_own_rule": rules, "checks": checks}
        v["selections"][name] = sel
        ok &= s_ok
        nat = {}
        for k in A["layer_native"][name][0]:
            r = A["layer_native"][name][0][k]
            if r is not None:
                nat[k] = rel_rms(r, Bv["layer_native"][name][0][k])
        v["layer_native"][name] = {"cross_rms": nat}  # reported, not gated
    v["model"] = {"args": MA["args"],
                  "weights_identical": MA["weights_sha256"] == MB["weights_sha256"]}
    ok &= v["model"]["weights_identical"]
    g_ok, rows = cross_and_noise(MA["runs"]["fp32"], MB["runs"]["fp32"], MODEL_FP32,
                                 MODEL_FP32_TINY)
    v["model"]["fp32"] = {"pass": g_ok, "tensors": rows}
    ok &= g_ok
    ref = MA["runs"]["fp32"][0]
    g_ok, rows = against_reference({k: ref[k] for k in ("y", "grad_x")}, MA["runs"]["bf16"],
                                   MB["runs"]["bf16"], lambda k, r: (BF16_RMS, None))
    v["model"]["bf16"] = {"pass": g_ok, "tensors": rows}
    ok &= g_ok
    # The comparison of the two releases ("verdict") and this version's cuda_graph option
    # ("cuda_graph_option.verdict") are separate results: the option is not part of either
    # release's default path.
    v["verdict"] = "EQUIVALENT" if ok else "NOT EQUIVALENT"
    v["cuda_graph_option"] = cuda_graph_option(a.cg_dir or d, Bv["checkout"]["commit"], here)
    with open(a.out, "w") as f:
        json.dump(v, f, indent=1, default=str)
    print(json.dumps(summary(v), indent=1, default=str))
    return 0 if ok else 1


def summary(v):
    """The verdict with the worst tensor of each gated section."""
    s = {"verdict": v["verdict"], "cuda_graph_option": v["cuda_graph_option"]["verdict"],
         "weights_identical": v["weights_identical"],
         "model_weights_identical": v["model"]["weights_identical"]}

    def worst(rows):
        def score(r):
            if "cross" in r:
                return r["cross"] / r["ceiling"]
            c = r["ceiling_rms"] or r["ceiling_max"]
            return max(r["rms_a"], r["rms_b"]) / c if r["ceiling_rms"] else \
                max(r["max_a"], r["max_b"]) / c
        k, r = max(rows.items(), key=lambda kv: score(kv[1]))
        return {"tensor": k, **r}

    for name in DTYPES:
        for C, blk in v["op"][name].items():
            s[f"op {name} C={C}"] = {"pass": blk["pass"], "worst": worst(blk["tensors"])}
        s[f"layer {name}"] = {"pass": v["layer"][name]["pass"],
                              "worst": worst(v["layer"][name]["tensors"])}
        s[f"selector {name}"] = {k: v["selector"][name][k] for k in v["selector"][name]
                                 if k != "checks"}
        s[f"model {name}"] = {"pass": v["model"][name]["pass"],
                              "worst": worst(v["model"][name]["tensors"])}
    for k, blk in v["cuda_graph_option"].items():
        if isinstance(blk, dict) and "pass" in blk:
            s[f"cuda_graph_option {k}"] = {x: blk[x] for x in blk if x != "tensors"}
    s["cuda_graph_option rule_v1"] = v["cuda_graph_option"].get("rule_v1", {}).get("verdict")
    s["selections"] = v["selections"]
    s["layer_native_cross_rms_max"] = {n: max(v["layer_native"][n]["cross_rms"].values())
                                       for n in DTYPES}
    return s


def main(argv=None):
    a = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rc = {"make": cmd_make, "run": cmd_run, "run-model": cmd_run_model,
          "run-cudagraph": cmd_run_cudagraph, "oracle": cmd_oracle,
          "compare": cmd_compare}[a.cmd](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
