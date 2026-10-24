# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Permanent cross-process determinism gate on token-like data. The op runs at a hybrid model's
shape: head_dim 960, chunk 128, 2 streams with their own slots, 64 writes and 64 reads
per token, snapshot 'none', one gate per write (key_weighted_decay). The bank is bfloat16 or float32. The slots are
token-like: Zipf popularity, and the SAME slot rows recurring in consecutive chunks. Most tokens replay the write and
read rows of the token one chunk earlier, and some read exactly the rows that token wrote. So consecutive steps write and
read the same bank rows, with duplicate slots inside every chunk. The last chunk is padded.

Every child process runs, for each bank dtype, a training forward + backward three times and a no-grad forward twice.
The children are fresh Python processes in different allocator and launch states:
- clean (SDM_GATE_PROCS of them, default 4; the release gate runs 12);
- a shifted, NaN-dirtied caching allocator;
- the caching allocator off (PYTORCH_NO_CUDA_MEMORY_CACHING=1);
- serialized launches (CUDA_LAUNCH_BLOCKING=1).
Every digest must be identical across the in-process repeats and across all processes: readings, final bank, restored
bank, the six gradients, and the no-grad readings and bank.
Nothing here reads the GPU class: it is the same gate on every device (H100, GB300)."""
import json
import os
import subprocess
import sys

from sdm_testlib import requires_cuda

pytestmark = requires_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))

_CHILD = r"""
import json, sys
sys.path[:0] = [ROOT, HERE]
import torch
from sdm_testlib import make_cotangent, tensor_digest
from sparse_delta_memory.memory_ops import sdm_write_read

P, C, W, R, HD = 2, 128, 64, 64, 960
T, SPH = CFG["T"], CFG["sph"]
NAMES = ("memory", "k_val", "v", "beta", "g", "q_val")


def token_slots(gen):
    # [P, T, W] / [P, T, R] global ids (stream p owns [p * SPH, (p + 1) * SPH)), distinct and sorted per token. Zipf
    # popularity (1 / (1 + rank) in a random slot order, shared by writes and reads), then chunk by chunk: a token
    # replays the write and read rows of the token one chunk earlier with probability 3/4, and reads that token's WRITE
    # rows with probability 1/4 (R == W).
    ks, qs = [], []
    for p in range(P):
        log_w = -torch.log1p(torch.randperm(SPH, generator=gen).float())

        def rows(n, width):
            u = torch.rand(n, SPH, generator=gen).clamp_min_(1e-20)
            return (log_w - torch.log(-torch.log(u))).topk(width, dim=-1).indices.sort(dim=-1).values

        k, q = rows(T, W), rows(T, R)
        rep = torch.rand(T, generator=gen) < 0.75
        raw = torch.rand(T, generator=gen) < 0.25
        for c0 in range(C, T, C):
            n = min(C, T - c0)
            cur, prev = slice(c0, c0 + n), slice(c0 - C, c0 - C + n)
            k[cur] = torch.where(rep[cur, None], k[prev], k[cur])
            q[cur] = torch.where(raw[cur, None], k[prev], torch.where(rep[cur, None], q[prev], q[cur]))
        ks.append(k + p * SPH)
        qs.append(q + p * SPH)
    return torch.stack(ks), torch.stack(qs)


def inputs(dtype):
    gen = torch.Generator().manual_seed(CFG["seed"])
    k_idx, q_idx = token_slots(gen)
    k_val = torch.softmax(2.0 * torch.randn(P, T, W, generator=gen), dim=-1)
    q_val = torch.softmax(2.0 * torch.randn(P, T, R, generator=gen), dim=-1)
    v = 0.1 * torch.randn(P, T, HD, generator=gen)
    beta = 0.3 * torch.rand(P, T, 1, generator=gen)
    flat = torch.rand(P, T, W, generator=gen) < 0.25            # a quarter of the writes almost without decay
    g = torch.where(flat, torch.empty(P, T, W).uniform_(-1e-3, -1e-5, generator=gen),
                    torch.empty(P, T, W).uniform_(-0.5, -0.01, generator=gen))
    memory = 0.1 * torch.randn(P * SPH, HD, generator=gen)
    f = dict(memory=memory, k_val=k_val, v=v, beta=beta, g=g, q_val=q_val)
    d = {n: t.to(device="cuda", dtype=dtype) for n, t in f.items()}
    d["k_idx"], d["q_idx"] = k_idx.cuda(), q_idx.cuda()
    return d


def call(d, leaves, **kw):
    return sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                          d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=SPH, num_parallel=P,
                          key_weighted_decay=True, **kw)


def train(d):
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in NAMES}
    readings = call(d, leaves, snapshot_quant="none")
    out = {"readings": tensor_digest(readings.detach()), "final_bank": tensor_digest(leaves["memory"].detach())}
    grads = torch.autograd.grad(readings, [leaves[n] for n in NAMES], grad_outputs=make_cotangent(readings, "dense"))
    out["restored_bank"] = tensor_digest(leaves["memory"].detach())
    out.update({"d_" + n: tensor_digest(t) for n, t in zip(NAMES, grads)})
    return out


def nograd(d):
    leaves = {n: d[n].detach().clone() for n in NAMES}
    with torch.no_grad():
        readings = call(d, leaves)
    return {"readings": tensor_digest(readings), "final_bank": tensor_digest(leaves["memory"])}


if DIRTY:   # shift the caching allocator: NaN-filled freed blocks and odd-sized blocks kept alive
    dirt = [torch.full((n,), float("nan"), device="cuda") for n in (1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26)]
    del dirt
    keep = [torch.empty(n, device="cuda", dtype=torch.uint8) for n in (4100, 77777, 12345, 3 << 20)]
res = {}
for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    d = inputs(dtype)
    res[name] = {"train": [train(d) for _ in range(3)], "nograd": [nograd(d) for _ in range(2)]}
    del d
print(json.dumps(res))
"""

_CFG = dict(T=8 * 128 - 11, sph=8192, seed=20261023)   # 8 chunks, the last with 117 valid tokens


def _child(dirty=False, env_extra=None):
    code = "ROOT = %r\nHERE = %r\nCFG = %r\nDIRTY = %r\n" % (_ROOT, _HERE, _CFG, dirty) + _CHILD
    env = dict(os.environ)
    for k in ("CUDA_LAUNCH_BLOCKING", "PYTORCH_NO_CUDA_MEMORY_CACHING"):
        env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=_HERE, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_token_data_gives_the_same_bits_in_every_process_and_allocator_state():
    procs = max(2, int(os.environ.get("SDM_GATE_PROCS", "4")))
    runs = [("clean%d" % i, _child()) for i in range(procs)]
    runs += [("dirty", _child(dirty=True)), ("nocache", _child(env_extra={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})),
             ("blocking", _child(env_extra={"CUDA_LAUNCH_BLOCKING": "1"}))]
    for label, res in runs:   # in-process repeats: every repeat equals the first
        for case, kinds in res.items():
            for kind, reps in kinds.items():
                bad = [(i, n) for i, rep in enumerate(reps) for n in rep if rep[n] != reps[0].get(n)]
                assert not bad, ("in-process repeats differ", label, case, kind, bad)
    ref_label, ref = runs[0]
    bad = [label for label, res in runs[1:] if res != ref]
    assert not bad, ("processes differ from " + ref_label, bad)
