# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Decode tokens/s of an SDM model: eager decoding against the generator's CUDA-graph decode
step, with greedy token identity.

Builds the model of a shipped config (only its ``model:`` section is read; ``--swa-window``
overrides its SWA window) with random weights (bf16, bytes vocabulary); ``--weights FILE``
loads them from FILE if it exists and writes them there otherwise, so that runs of several
checkouts decode the same model. For each context length: one prefill, then from that same
state (restored in place before each run) WARM + GEN greedy decode steps eagerly
(``generate_next_token``) and through the generator's ``decode_step`` (CUDA graph: the first
step eager, the second captured, then replays). Times the last GEN steps of each run,
including the per-step host read of the token that ``generate()`` does; tokens/s is the
median over --repeats runs per mode. Tokens of all steps and the logits of the untimed steps
must match bitwise between the two modes. A checkout whose generator has no ``decode_step``
(the previous release) is timed eagerly only.

Run from a checkout (``apps/`` and ``lingua/`` on PYTHONPATH); writes one JSON file:

  PYTHONPATH=. python benchmarks/bench_decode.py --config apps/main/configs/sdm_flagship.yaml \\
      --out benchmarks/results/decode.json [--contexts 1024,8192,32768] [--gen 256]

The run refuses to time while another process is on the GPU (``--allow-busy`` overrides).
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

import torch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="a training config; its model: section")
    p.add_argument("--contexts", default="1024,8192,32768")
    p.add_argument("--gen", type=int, default=256, help="timed decode steps per run")
    p.add_argument("--warm", type=int, default=8, help="untimed decode steps before them")
    p.add_argument("--repeats", type=int, default=3, help="runs per mode and context")
    p.add_argument("--swa-window", type=int, help="override the config's model.swa_window")
    p.add_argument("--weights", help="state dict file: loaded if it exists, written otherwise")
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


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


def environment():
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name()}
    try:
        info["triton"] = importlib.import_module("triton").__version__
    except Exception:
        info["triton"] = None
    return info


def mutable_state(gen):
    from lingua.transformer import Attention
    from provenance import sdm_package  # this directory is sys.path[0]

    SparseDeltaMemory = sdm_package().SparseDeltaMemory
    state = [gen.current_tok_id]
    for m in gen.model.modules():
        if isinstance(m, Attention):
            state += [m.kv_cache.k_cache, m.kv_cache.v_cache]
        elif isinstance(m, SparseDeltaMemory):
            state.append(m._gen_cache.memory)
    return state


def run(step, first, warm, n):
    from apps.main.generate import sample_tokens

    tokens, logits, tok = [], [], first
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = step(tok)
        tok = sample_tokens(out.clone(), 0.0)
        tokens.append(tok.item())  # the per-step host read generate() does
        if i < warm:
            logits.append(out.float().cpu())
    torch.cuda.synchronize()
    return tokens, logits, n / (time.perf_counter() - t0)


@torch.inference_mode()
def bench(gen, prompt, warm, n, repeats):
    from apps.main.generate import pack_prompts, sample_tokens

    tokens, lengths = pack_prompts([prompt])
    tokens, lengths = tokens.cuda(), lengths.cuda()
    first = sample_tokens(gen.prefill(tokens.unsqueeze(0), lengths), 0.0)[:, lengths.cumsum(0) - 1]
    saved = [t.clone() for t in mutable_state(gen)]
    steps = {"eager": gen.generate_next_token}
    if hasattr(gen, "decode_step"):  # the CUDA-graph decode step (not in the previous release)
        dt = gen._decode_tensors()
        steps["graph"] = lambda t: gen.decode_step(t, dt)
    out = {}
    for _ in range(repeats):
        for mode, step in steps.items():
            for t, s in zip(mutable_state(gen), saved):
                t.copy_(s)
            toks, logits, rate = run(step, first, warm, n)
            o = out.setdefault(mode, dict(rates=[], tokens=[], logits=[]))
            o["rates"].append(rate)
            o["tokens"].append(toks)
            o["logits"].append(logits)
    for m in gen.model.modules():
        if hasattr(m, "_gen_cache"):
            del m._gen_cache
    return out


def known_keys(cls, d, path=""):
    """``d`` without the keys that the checkout's dataclass ``cls`` (nested dataclasses too)
    does not have, and the dotted names of those keys: the previous release lacks options
    added since, e.g. the training-only ``sdm_args.cuda_graph``."""
    import dataclasses
    import types
    import typing

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
    out, dropped = {}, []
    for k, v in d.items():
        if k not in hints:
            dropped.append(path + k)
            continue
        t = hints[k]
        if typing.get_origin(t) in (typing.Union, types.UnionType):
            args = [x for x in typing.get_args(t) if x is not type(None)]
            t = args[0] if len(args) == 1 else t
        if isinstance(v, dict) and dataclasses.is_dataclass(t):
            v, sub = known_keys(t, v, f"{path}{k}.")
            dropped += sub
        out[k] = v
    return out, dropped


def weights_sha256(model):
    import hashlib

    h = hashlib.sha256()
    for n, t in sorted(model.state_dict().items()):
        t = t.detach().contiguous().cpu()
        h.update(f"{n}:{tuple(t.shape)}:{t.dtype};".encode() + t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkout_relative(path):
    """``path`` relative to the checkout whose lingua/ was imported, else to this harness's
    checkout (the config of a run of the previous release), else its base name."""
    import lingua

    for root in (os.path.dirname(os.path.dirname(os.path.abspath(lingua.__file__))),
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        rel = os.path.relpath(os.path.abspath(path), root)
        if not rel.startswith(".."):
            return rel
    return os.path.basename(path)


def run_all(a, res):
    from omegaconf import OmegaConf

    from apps.main.generate import (PackedCausalTransformerGenerator,
                                    PackedCausalTransformerGeneratorArgs)
    from apps.main.transformer import LMTransformer, LMTransformerArgs
    from lingua.args import dataclass_from_dict
    from lingua.tokenizer import build_tokenizer
    from lingua.transformer import Attention

    cfg = OmegaConf.load(a.config)
    if a.swa_window is not None:
        cfg.model.swa_window = a.swa_window
    tok = build_tokenizer("bytes")
    contexts = [int(c) for c in a.contexts.split(",")]
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    sdm = model_cfg.get("sdm_args")
    if isinstance(sdm, dict) and "chunk_size" in sdm:  # memory_block_size in the previous release
        from provenance import chunk_field, sdm_package  # this directory is sys.path[0]

        SparseDeltaMemoryArgs = sdm_package().SparseDeltaMemoryArgs

        field, c = chunk_field(SparseDeltaMemoryArgs), sdm.pop("chunk_size")
        if field == "chunk_size" or c is not None:  # None: each version's default (this release 64, the previous 256)
            sdm[field] = c
    model_cfg, res["config_keys_ignored"] = known_keys(LMTransformerArgs, model_cfg)
    margs = dataclass_from_dict(LMTransformerArgs, model_cfg, strict=False)
    margs.vocab_size = tok.n_words
    margs.max_seqlen = max(contexts) + a.gen + a.warm + 8
    torch.manual_seed(0)
    model = LMTransformer(margs).cuda()
    model.init_weights()
    if a.weights and os.path.exists(a.weights):
        model.load_state_dict(torch.load(a.weights, map_location="cuda", weights_only=True))
        res["weights"] = "loaded"
    elif a.weights:
        tmp = f"{a.weights}.tmp{os.getpid()}"
        torch.save({k: t.to(torch.bfloat16) for k, t in model.state_dict().items()}, tmp)
        os.replace(tmp, a.weights)
        res["weights"] = "written"
    model = model.to(torch.bfloat16).eval()
    res["swa_window"] = getattr(model, "swa_window", None)
    res["weights_sha256"] = weights_sha256(model)
    g = torch.Generator().manual_seed(0)
    res["rows"] = []
    for ctx in contexts:
        prompt = [tok.bos_id] + torch.randint(0, 256, (ctx - 1,), generator=g).tolist()
        for m in model.modules():  # the generator sizes the KV caches on first use only
            if isinstance(m, Attention) and hasattr(m, "kv_cache"):
                del m.kv_cache
        gargs = PackedCausalTransformerGeneratorArgs(
            temperature=0.0, max_gen_len=a.gen + a.warm + 1, dtype="bf16",
            max_tokens=ctx + a.gen + a.warm + 8)
        gen = PackedCausalTransformerGenerator(gargs, model, tok)
        if ctx == contexts[0]:
            res["gpu_processes"] = other_gpu_processes()
            if res["gpu_processes"] and not a.allow_busy:
                raise RuntimeError(f"another process is on the GPU: {res['gpu_processes']}")
        out = bench(gen, prompt, a.warm, a.gen, a.repeats)
        e = out["eager"]
        row = dict(context=ctx, eager_tokens_per_s=sorted(e["rates"])[len(e["rates"]) // 2],
                   eager_rates=e["rates"])
        if "graph" in out:
            gr = out["graph"]
            # Each repetition of the graph against the eager run from the same state.
            same_tokens = [x == y for x, y in zip(e["tokens"], gr["tokens"])]
            same_logits = [len(x) == len(y) and all(torch.equal(p, q) for p, q in zip(x, y))
                           for x, y in zip(e["logits"], gr["logits"])]
            row.update(graph_tokens_per_s=sorted(gr["rates"])[len(gr["rates"]) // 2],
                       graph_rates=gr["rates"], tokens_identical=all(same_tokens),
                       untimed_logits_bitwise=all(same_logits),
                       tokens_identical_per_repeat=same_tokens,
                       untimed_logits_bitwise_per_repeat=same_logits,
                       graph_stats=dict(gen.decode_graph.stats))
            row["speedup"] = row["graph_tokens_per_s"] / row["eager_tokens_per_s"]
        else:  # eager only: its repetitions from the same state must agree with each other
            row["eager_tokens_identical_across_repeats"] = all(t == e["tokens"][0] for t in e["tokens"])
        res["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_rates")}), flush=True)
        del gen
        torch.cuda.empty_cache()


def main(argv=None):
    from provenance import checkout_info, public_args  # this directory is sys.path[0]

    a = parse_args(argv)
    args = public_args(a)
    res = {"bench": "decode", "args": args, "status": "started", "environment": environment()}
    try:
        args["config"] = checkout_relative(a.config)
        if args.get("weights"):
            args["weights"] = os.path.basename(args["weights"])
        res["sdm_checkout"] = checkout_info()
        run_all(a, res)
        bad = [r["context"] for r in res["rows"]
               if "graph_tokens_per_s" in r and not (r["tokens_identical"] and r["untimed_logits_bitwise"])]
        unstable = [r["context"] for r in res["rows"] if "graph_tokens_per_s" not in r
                    and r.get("eager_tokens_identical_across_repeats") is not True]
        if bad:
            res.update(status="mismatch", error=f"graph decoding differs from eager at contexts {bad}")
        elif unstable:
            res.update(status="mismatch",
                       error=f"eager decoding repetitions differ at contexts {unstable}")
        else:
            res["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="oom", error=str(e)[:2000])
    except Exception as e:
        res.update(status="error", error=f"{type(e).__name__}: {str(e)[:2000]}",
                   traceback=traceback.format_exc()[-4000:])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("status", "error") if k in res}, default=str))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
