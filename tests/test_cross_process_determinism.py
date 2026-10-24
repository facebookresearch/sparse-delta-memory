# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Cross-process + activation-checkpointing determinism (two fresh
processes with the same seed and data must give the same bits; the SDM forward re-run by non-reentrant activation
checkpointing inside the backward must reproduce the first forward bit for bit). Each case runs the SAME short
training loop (a SparseDeltaMemory layer of 2 heads, chunk 128, 64 writes / reads, Adam steps) in fresh Python
processes: a clean one, and one whose CUDA caching allocator was shifted first (blocks of odd sizes kept alive, freed
blocks filled with NaN, so an uninitialized read or an address-dependent choice changes bits or yields NaN); each
process runs the loop twice, with and without activation checkpointing. Every digest (losses, outputs, all parameter
gradients, final parameters) must be equal across the two processes and across both modes."""
import json
import os
import subprocess
import sys

import pytest
import torch

from sdm_testlib import requires_cuda

pytestmark = requires_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))

_CHILD = r"""
import contextlib, hashlib, json, sys
sys.path[:0] = [ROOT, HERE]
import torch
import torch.utils.checkpoint as tuc
from sdm_testlib import tensor_digest
from torch.utils.checkpoint import checkpoint
from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

def digest(ts):
    return hashlib.sha256("|".join(tensor_digest(t.detach()) for t in ts).encode()).hexdigest()

def run(ac, full=False):
    torch.manual_seed(1234)
    args = SparseDeltaMemoryArgs(dim=CFG["dim"], num_heads=2, slots_per_head=CFG["sph"], num_writes=64, num_reads=64,
                                 chunk_size=128, normalize_readings=True, output_gate=True, backprop_on_memory=True,
                                 snapshot_quant=CFG["quant"])
    layer = SparseDeltaMemory(args, layer_id=0).cuda()
    layer.init_weights()
    layer = layer.to(getattr(torch, CFG["dtype"]))
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    reads = []   # the layer output of every forward the layer runs (first forwards and AC recomputes)
    layer.register_forward_hook(lambda m, i, o: reads.append(tensor_digest(o[0].detach())))
    gen = torch.Generator(device="cuda").manual_seed(7)
    out = []
    for step in range(CFG["steps"]):
        x = (torch.randn(CFG["B"], CFG["T"], CFG["dim"], device="cuda", generator=gen)).to(getattr(torch, CFG["dtype"]))
        x.requires_grad_(True)
        fn = lambda t: layer(t)[0]
        # full: no early stop, so the recompute runs the whole layer and its forward hook sees the recomputed output
        with tuc.set_checkpoint_early_stop(False) if full else contextlib.nullcontext():
            y = checkpoint(fn, x, use_reentrant=False) if ac else fn(x)
        loss = (y.float() ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        out.append(digest([loss, y, x.grad] + [p.grad for p in layer.parameters() if p.grad is not None]))
        opt.step()
    out.append(digest(list(layer.parameters())))
    return (out, reads) if full or not ac else out

if DIRTY:
    dirt = [torch.full((n,), float("nan"), device="cuda") for n in (1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26)]
    del dirt
    keep = [torch.empty(n, device="cuda", dtype=torch.uint8) for n in (4100, 77777, 12345, 3 << 20)]
plain, plain_reads = run(False)
full, full_reads = run(True, full=True)
print(json.dumps({"plain": plain, "ac": run(True), "ac_full": full, "reads": {"plain": plain_reads, "ac_full": full_reads}}))
"""

CASES = {   # (model / bank dtype, snapshot_quant, tokens): hybrid-model configurations, scaled down (hd 960);
    # T 500 = 3 chunks of 128 + 116: padding tokens in the last chunk (rows no kernel may leave unwritten)
    "bf16_none": dict(dtype="bfloat16", quant="none", T=512),
    "fp32_none": dict(dtype="float32", quant="none", T=512),
    "fp32_fp8": dict(dtype="float32", quant="fp8", T=512),
    "bf16_none_pad": dict(dtype="bfloat16", quant="none", T=500),
    "fp32_none_pad": dict(dtype="float32", quant="none", T=500),
}


def _child(cfg, dirty):
    code = "ROOT = %r\nHERE = %r\nCFG = %r\nDIRTY = %r\n" % (_ROOT, _HERE, cfg, dirty) + _CHILD
    env = dict(os.environ)
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=_HERE, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", list(CASES))
def test_two_processes_and_the_ac_recompute_give_the_same_bits(case):
    cfg = dict(CASES[case], dim=1920, sph=4096, B=2, steps=3)   # head_dim 960 (2 heads), chunk 128
    clean, dirty = _child(cfg, False), _child(cfg, True)
    assert clean["plain"] == clean["ac"], ("activation checkpointing changed the bits (clean process)", clean)
    assert dirty["plain"] == dirty["ac"], ("activation checkpointing changed the bits (shifted allocator)", dirty)
    assert clean == dirty, ("two processes differ (the second with a shifted, NaN-dirtied allocator)", clean, dirty)
    # R3-2: the SDM forward that activation checkpointing recomputes inside the backward reproduces the first
    # forward bit for bit, read directly on the layer output of every forward (recompute without early stop)
    for label, res in (("clean", clean), ("shifted allocator", dirty)):
        r, steps = res["reads"], cfg["steps"]
        assert res["ac_full"] == res["plain"], ("activation checkpointing without early stop changed the bits", label)
        assert len(r["plain"]) == steps and len(r["ac_full"]) == 2 * steps, (label, r)
        assert r["ac_full"][0::2] == r["plain"], ("the first forward under activation checkpointing differs", label, r)
        assert r["ac_full"][1::2] == r["ac_full"][0::2], ("the recomputed SDM forward differs from the first", label, r)
