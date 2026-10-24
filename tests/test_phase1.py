# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Phase-1 forward (cuda/phase1_cuda.py) against a float64 reference.

The reference evaluates the Phase-1 definitions directly: it joins the entries of each
(chunk, slot) with torch and sums over the joined pairs in float64. It shares no sort
order, scan or walk with the kernels. ``test_reference_matches_token_loop`` checks it
against a literal token loop on small chunks.
"""
import dataclasses
import itertools
import math
import re

import pytest
import torch

import phase1_reference
from sdm_testlib import (
    bits_equal,
    make_op_inputs,
    profiled_kernel_names,
    reference_dequantize,
    reference_quantize,
    rel_max,
    require_free_memory,
    requires_cuda,
)

pytestmark = requires_cuda

F64 = torch.float64
VALUE_FIELDS = ("a", "qk", "log_k", "log_q", "rex", "cpe", "k_eff", "q_w")
FLAG_FIELDS = ("is_last", "is_only")
GATE_REL_MAX = 1e-6
BF16_REL = 2.0 ** -8


def _p1():
    from sparse_delta_memory.cuda import phase1_cuda
    return phase1_cuda


# ------------------------------------------------------------------------------ inputs
def _distinct_uniform(rows, width, sph, gen):
    """[rows, width] distinct slots in [0, sph), lanes in random order."""
    base = torch.randint(0, sph - width + 1, (rows, width), device="cuda", generator=gen)
    slots = base.sort(-1).values + torch.arange(width, device="cuda")
    return slots.gather(1, torch.rand(rows, width, device="cuda", generator=gen).argsort(-1))


def _zipf(rows, width, sph, gen):
    """Distinct slots with popularity ~ 1/rank (Gumbel top-k)."""
    rank = torch.randperm(sph, device="cuda", generator=gen)
    logw = -torch.log1p(rank.double())
    u = torch.rand(rows, sph, device="cuda", generator=gen, dtype=F64).clamp_min(1e-300)
    return (logw - torch.log(-torch.log(u))).topk(width, dim=-1).indices


def _single_hot(rows, width, sph, gen):
    """Every row holds one shared hot slot at a random lane; the rest are uniform."""
    hot = int(torch.randint(0, sph, (1,), device="cuda", generator=gen))
    rest = _distinct_uniform(rows, width, sph - 1, gen)
    rest = rest + (rest >= hot).long()
    lane = torch.randint(0, width, (rows, 1), device="cuda", generator=gen)
    return rest.scatter(1, lane, hot)


def _sticky(rows, width, sph, gen, keep=0.9):
    """Each row keeps a slot of the previous row with probability ``keep``."""
    out = [_distinct_uniform(1, width, sph, gen)[0]]
    for _ in range(rows - 1):
        row = out[-1].clone()
        fresh = torch.rand(width, device="cuda", generator=gen) >= keep
        while True:
            cand = torch.randint(0, sph, (width,), device="cuda", generator=gen)
            row = torch.where(fresh, cand, row)
            dup = torch.zeros(width, dtype=torch.bool, device="cuda")
            srt, order = row.sort()
            dup[order[1:]] = srt[1:] == srt[:-1]
            if not dup.any():
                break
            fresh = dup
        out.append(row)
    return torch.stack(out)


def _slots(dist, rows, width, sph, gen, k_rows=None):
    if dist == "overlap":
        if width <= k_rows.shape[1]:
            pick = torch.rand(k_rows.shape, device="cuda", generator=gen).argsort(-1)[:, :width]
            return k_rows.gather(1, pick)
        extra = _distinct_uniform(rows, width, sph, gen)
        return torch.cat([k_rows, extra[:, : width - k_rows.shape[1]]], 1) % sph
    return {"uniform": _distinct_uniform, "zipf": _zipf, "single_hot": _single_hot,
            "sticky": _sticky}[dist](rows, width, sph, gen)


@dataclasses.dataclass(frozen=True)
class Case:
    W: int
    R: int
    C: int
    gate: str = "token"          # token | write
    sph: int = 65536
    valid: str = "C"             # C | C-1 | 1
    dist: str = "uniform"        # uniform | zipf | single_hot | sticky | overlap
    dtype: str = "float32"       # float32 | bfloat16
    index: str = "int64"         # int64 | int32
    layout: str = "streams"      # streams (own slot ranges) | shared (one bank) | single
    steps: int = 2
    gate_scale: float = 0.2

    @property
    def P(self):
        return 1 if self.layout == "single" else 2

    @property
    def valid_last(self):
        return {"C": self.C, "C-1": max(1, self.C - 1), "1": 1}[self.valid]

    def id(self):
        return (f"W{self.W}-R{self.R}-C{self.C}-{self.gate}-sph{self.sph}-v{self.valid}-"
                f"{self.dist}-{self.dtype}-{self.index}-{self.layout}")


def make_inputs(case, seed=0):
    """Time-major chunk inputs [N, C, *] and the phase1_forward keyword arguments."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    P, C, W, R, sph = case.P, case.C, case.W, case.R, case.sph
    N = P * case.steps
    k_local = torch.empty(N, C, W, dtype=torch.long, device="cuda")
    q_local = torch.empty(N, C, R, dtype=torch.long, device="cuda")
    for p in range(P):
        k = _slots("uniform" if case.dist == "overlap" else case.dist, case.steps * C, W, sph, gen)
        q = _slots(case.dist, case.steps * C, R, sph, gen, k_rows=k)
        k_local[p::P] = k.view(case.steps, C, W)
        q_local[p::P] = q.view(case.steps, C, R)
    offset = sph if case.layout == "streams" else 0
    stream = (torch.arange(N, device="cuda") % P).view(N, 1, 1)
    idx_dtype = getattr(torch, case.index)
    dtype = getattr(torch, case.dtype)
    gw = W if case.gate == "write" else 1
    return dict(
        k_idx=(k_local + stream * offset).to(idx_dtype),
        q_idx=(q_local + stream * offset).to(idx_dtype),
        k_val=torch.randn(N, C, W, device="cuda", generator=gen).to(dtype),
        q_val=torch.randn(N, C, R, device="cuda", generator=gen).to(dtype),
        g=-torch.rand(N, C, gw, device="cuda", generator=gen) * case.gate_scale,
        kwargs=dict(num_parallel=P, stream_offset=offset, num_keys=sph,
                    valid_last=case.valid_last),
    )


def run_kernel(d):
    repeats = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = _p1().phase1_forward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"], d["g"],
                               repeats, **d["kwargs"])
    return out, repeats


# --------------------------------------------------------------------------- reference
def reference_phase1(k_idx, q_idx, k_val, q_val, g, *, num_parallel, stream_offset,
                     valid_last, **_):
    """Phase-1 outputs in float64 from their definitions (see the module docstring)."""
    N, C, W = k_idx.shape
    R = q_idx.shape[2]
    dev = k_idx.device
    chunk = torch.arange(N, device=dev)
    t_valid = torch.where(chunk >= N - num_parallel, valid_last, C)
    tok = torch.arange(C, device=dev)
    valid = tok.view(1, C) < t_valid.view(N, 1)
    off = (chunk % num_parallel).view(N, 1, 1) * stream_offset
    ks = k_idx.long() - off
    qs = q_idx.long() - off
    # x: the gate on the first lane of each (token, slot); repeats contribute nothing.
    earlier = torch.ones(W, W, dtype=torch.bool, device=dev).tril(-1)
    first = ~((ks.unsqueeze(-1) == ks.unsqueeze(-2)) & earlier).any(-1)
    x = torch.where(first & valid.unsqueeze(-1), g.double().expand(N, C, W), 0.0)

    def entries(slots, vals, width, is_q):
        n_, t_, l_ = torch.meshgrid(chunk, tok, torch.arange(width, device=dev), indexing="ij")
        keep = valid.unsqueeze(-1).expand(N, C, width)
        count = int(keep.sum())
        gates = torch.zeros(count, dtype=F64, device=dev) if is_q else x[keep]
        return dict(chunk=n_[keep], slot=slots[keep], tok=t_[keep], lane=l_[keep],
                    val=vals.double()[keep], x=gates, is_q=torch.full((count,), is_q, device=dev))

    ek, eq = entries(ks, k_val, W, False), entries(qs, q_val, R, True)
    e = {f: torch.cat([ek[f], eq[f]]) for f in ek}
    nk = ek["chunk"].numel()
    # All ordered pairs (u, v) of entries sharing a (chunk, slot).
    _, gid = torch.unique(torch.stack([e["chunk"], e["slot"]]), dim=1, return_inverse=True)
    order = gid.argsort()
    size = torch.bincount(gid)
    start = torch.cumsum(size, 0) - size
    reps = size[gid]
    u = torch.arange(gid.numel(), device=dev).repeat_interleave(reps)
    first = (torch.cumsum(reps, 0) - reps).repeat_interleave(reps)
    member = torch.arange(u.numel(), device=dev) - first
    v = order[start[gid[u]] + member]
    vk = ~e["is_q"][v]
    tu, tv = e["tok"][u], e["tok"][v]
    log = torch.zeros(e["tok"].numel(), dtype=F64, device=dev).index_add_(
        0, u, torch.where(vk & (tv <= tu), e["x"][v], 0.0))
    total = torch.zeros_like(log).index_add_(0, u, torch.where(vk, e["x"][v], 0.0))
    later_k = torch.zeros_like(log).index_add_(
        0, u, (vk & ((tv > tu) | ((tv == tu) & (e["lane"][v] > e["lane"][u])))).double())
    k_count = torch.zeros_like(log).index_add_(0, u, vk.double())

    out = {}
    lk, lq = log[:nk], log[nk:]
    shape_k, shape_q = (N, C, W), (N, C, R)

    def scatter(vals, ent, shape, fill, dtype=F64):
        o = torch.full(shape, fill, dtype=dtype, device=dev)
        o[ent["chunk"], ent["tok"], ent["lane"]] = vals.to(dtype)
        return o

    out["log_k"] = scatter(lk, ek, shape_k, 0.0)
    out["log_q"] = scatter(lq, eq, shape_q, 0.0)
    out["rex"] = scatter(total[:nk] - lk, ek, shape_k, 0.0)
    out["cpe"] = scatter(torch.exp(total[:nk]), ek, shape_k, 1.0)
    out["k_eff"] = scatter(ek["val"] * torch.exp(lk), ek, shape_k, 0.0)
    out["q_w"] = scatter(eq["val"] * torch.exp(lq), eq, shape_q, 0.0)
    out["is_last"] = scatter(later_k[:nk] == 0, ek, shape_k, 0, torch.int8)
    out["is_only"] = scatter(k_count[:nk] == 1, ek, shape_k, 0, torch.int8)
    uq = e["is_q"][u]
    pair_a = vk & ~uq & (tv < tu)
    pair_qk = vk & uq & (tv <= tu)
    term = e["val"][u] * e["val"][v] * torch.exp(log[u] - log[v])
    for name, sel in (("a", pair_a), ("qk", pair_qk)):
        flat = (e["chunk"][u] * C + tu) * C + tv
        out[name] = torch.zeros(N * C * C, dtype=F64, device=dev).index_add_(
            0, flat[sel], term[sel]).view(N, C, C)
    return out


def token_loop_reference(kidx, qidx, kv, qv, g, valid_t):
    """One chunk, literal token loop."""
    C, W = kidx.shape
    R = qidx.shape[1]
    gw = g.double().expand(C, W)
    acc, lk, lq = {}, torch.zeros(C, W, dtype=F64), torch.zeros(C, R, dtype=F64)
    for t in range(valid_t):
        seen = set()
        for w in range(W):
            s = int(kidx[t, w])
            if s not in seen:
                acc[s] = acc.get(s, 0.0) + float(gw[t, w])
                seen.add(s)
            lk[t, w] = acc[s]
        for r in range(R):
            lq[t, r] = acc.get(int(qidx[t, r]), 0.0)
    A, QK = torch.zeros(C, C, dtype=F64), torch.zeros(C, C, dtype=F64)
    for i in range(valid_t):
        for j in range(i + 1):
            for a in range(W):
                for b in range(W):
                    if j < i and kidx[i, a] == kidx[j, b]:
                        A[i, j] += kv[i, a] * kv[j, b] * torch.exp(lk[i, a] - lk[j, b])
            for a in range(R):
                for b in range(W):
                    if qidx[i, a] == kidx[j, b]:
                        QK[i, j] += qv[i, a] * kv[j, b] * torch.exp(lq[i, a] - lk[j, b])
    rex = torch.zeros(C, W, dtype=F64)
    for t in range(valid_t):
        for w in range(W):
            rex[t, w] = acc[int(kidx[t, w])] - lk[t, w]
    return dict(log_k=lk, log_q=lq, rex=rex, a=A, qk=QK)


# ------------------------------------------------------------------------------- cases
_WR = (1, 3, 16, 48, 64, 128, 256)
_CS = (16, 17, 48, 64, 127, 128, 255, 256)
_DISTS = ("uniform", "zipf", "single_hot", "sticky", "overlap")
_ROTATE = dict(gate=("token", "write"), valid=("C", "C-1", "1"),
               dtype=("float32", "bfloat16"), index=("int64", "int32"),
               layout=("streams", "shared", "single"))


def _rotated(i, **fixed):
    """A case whose unspecified factors cycle with ``i``, so every level appears."""
    free = {k: v[i % len(v)] for k, v in _ROTATE.items() if k not in fixed}
    return Case(**{**free, **fixed})


def _grid():
    cases = []
    pairs = [(w, w) for w in _WR] + [(1, 256), (256, 1), (3, 48), (48, 3), (16, 128),
                                     (128, 16), (64, 256), (256, 64)]
    for i, (W, R) in enumerate(pairs):
        cases.append(_rotated(i, W=W, R=R, C=64))
    for i, (C, W) in enumerate(itertools.product(_CS, (3, 64, 256))):
        cases.append(_rotated(i + 1, W=W, R=W if i % 2 else max(1, W // 2), C=C))
    for i, (dist, sph) in enumerate(itertools.product(_DISTS, (512, 65536, 1 << 20))):
        if dist == "zipf" and sph > 65536:
            continue
        cases.append(_rotated(i + 2, W=64, R=48, C=127 if i % 2 else 64, dist=dist, sph=sph))
    for i, (valid, gate) in enumerate(itertools.product(_ROTATE["valid"], _ROTATE["gate"])):
        cases.append(_rotated(i, W=48, R=64, C=17, valid=valid, gate=gate, dist="sticky"))
    cases.append(_rotated(0, W=256, R=256, C=256, sph=512, dist="uniform", steps=1))
    cases.append(_rotated(1, W=256, R=256, C=256, sph=65536, dist="single_hot", steps=1))
    cases.append(_rotated(2, W=64, R=64, C=64, gate_scale=40.0, dist="sticky"))
    return cases


GRID = _grid()


def _check(case, out, ref):
    bad = {}
    for f in VALUE_FIELDS:
        got = getattr(out, f)
        if f in ("cpe", "k_eff", "q_w") and got.dtype == torch.bfloat16:
            err = ((got.double() - ref[f]).abs() - BF16_REL * ref[f].abs()).max().item()
            if err > 1e-30:
                bad[f] = f"bf16 value off by more than 2^-8 relative ({err:.3g})"
            continue
        dev = rel_max(ref[f], got)
        if not dev <= GATE_REL_MAX:
            bad[f] = f"rel_max {dev:.3g}"
    for f in FLAG_FIELDS:
        if not torch.equal(getattr(out, f), ref[f]):
            bad[f] = "flags differ"
    assert not bad, f"{case.id()}: {bad}"


@pytest.mark.parametrize("case", GRID, ids=[c.id() for c in GRID])
def test_forward_matches_reference(case):
    d = make_inputs(case)
    out, repeats = run_kernel(d)
    ref = reference_phase1(**d, **d["kwargs"])
    _check(case, out, ref)
    _check_maps(out, d)
    assert repeats.item() == 0


def test_invalid_entries_are_exactly_neutral():
    case = Case(W=16, R=8, C=48, valid="1", dtype="bfloat16")
    d = make_inputs(case)
    out, _ = run_kernel(d)
    N, v = out.a.shape[0], case.valid_last
    tail = slice(N - case.P, N)
    for f, value in (("log_k", 0.0), ("log_q", 0.0), ("rex", 0.0), ("cpe", 1.0),
                     ("k_eff", 0.0), ("q_w", 0.0), ("is_last", 0), ("is_only", 0)):
        t = getattr(out, f)[tail, v:]
        assert bits_equal(t, torch.full_like(t, value)), f
    for f in ("a", "qk"):
        t = getattr(out, f)[tail]
        assert bits_equal(t[:, v:], torch.zeros_like(t[:, v:])), f
        assert bits_equal(t[:, :, v:], torch.zeros_like(t[:, :, v:])), f


def test_triangles_are_exactly_zero_above_the_diagonal():
    out, _ = run_kernel(make_inputs(Case(W=64, R=64, C=64, dist="single_hot")))
    C = out.a.shape[-1]
    upper = torch.ones(C, C, dtype=torch.bool, device="cuda").triu()
    assert bits_equal(out.a[:, upper], torch.zeros_like(out.a[:, upper]))
    strict = upper.triu(1)
    assert bits_equal(out.qk[:, strict], torch.zeros_like(out.qk[:, strict]))


def _single_pair_decays(gates, pairs, W=16, R=16, sph=65536, seed=0):
    """One chunk of C = len(gates) tokens in which every token writes and reads one hot slot
    (lane 0) whose group straddles the first scan tile boundary. For each (i, j) in `pairs`
    only k_i, k_j and q_i are nonzero, so A[i, j] / (k_i k_j) is the computed decay.
    Returns [(computed decay A, computed decay QK[i, j], QK[i, i] / (q_i k_i), exact decay)]."""
    C = len(gates)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hot = sph // 2
    k_idx = _distinct_uniform(C, W, sph - 1, gen)
    q_idx = _distinct_uniform(C, R, sph - 1, gen)
    k_idx = (k_idx + (k_idx >= hot).long()).view(1, C, W)
    q_idx = (q_idx + (q_idx >= hot).long()).view(1, C, R)
    k_idx[..., 0] = hot
    q_idx[..., 0] = hot
    before = int((k_idx < hot).sum() + (q_idx < hot).sum())
    assert before < 4096 < before + 2 * C, "the hot group must straddle the tile boundary"
    g = torch.tensor(gates, dtype=torch.float32, device="cuda").view(1, C, 1)
    exact_log = torch.cumsum(g.double().view(-1), 0)
    out = []
    for i, j in pairs:
        k_val = torch.zeros(1, C, W, device="cuda")
        q_val = torch.zeros(1, C, R, device="cuda")
        k_val[0, i, 0], k_val[0, j, 0], q_val[0, i, 0] = 1.5, -0.75, 0.5
        rep = torch.zeros(1, dtype=torch.int64, device="cuda")
        r = _p1().phase1_forward(k_idx, q_idx, k_val, q_val, g, rep, num_parallel=1,
                                 stream_offset=0, num_keys=sph, valid_last=C)
        out.append((r.a[0, i, j].item() / (1.5 * -0.75), r.qk[0, i, j].item() / (0.5 * -0.75),
                    r.qk[0, i, i].item() / (0.5 * 1.5),
                    torch.exp(exact_log[i] - exact_log[j]).item()))
    return out


def _ulp32(x):
    x = abs(x)
    return 2.0 ** -149 if x < 2.0 ** -126 else 2.0 ** (math.floor(math.log2(x)) - 23)


def test_decay_after_a_large_prefix_is_ulp_accurate():
    """A -4100 log prefix, then increments near, far below and exactly at zero relative to its
    ulp (4.9e-4): each computed decay is within 4 ulps of exp(sum of gates), never above 1,
    and a Q entry's decay to its own token's K entry is exactly 1."""
    steps = (-1e-3, -3e-7, 0.0, -2.5e-4, -1e-9)
    gates = [-41.0] * 100 + [steps[t % len(steps)] for t in range(156)]
    pairs = [(101, 100), (102, 101), (103, 102), (104, 103), (105, 104), (140, 120), (255, 100),
             (100, 99), (120, 3)]
    decays = _single_pair_decays(gates, pairs)
    for (i, j), (dec_a, dec_qk, own, exact) in zip(pairs, decays):
        for got in (dec_a, dec_qk):
            assert abs(got - exact) <= 4 * _ulp32(exact), ((i, j), got, exact)
            assert got <= 1.0, ((i, j), got)
        assert own == 1.0, ((i, j), own)


def test_decay_near_float32_underflow():
    """Decays from exp(-80) down through the subnormal range to exp(-110), checked as
    |decay - exact| <= 4 * (ulp(|log exact|) + 2^-24) * exact + 4 * 2^-149: a float32
    exponent near 100 is off by about ulp(100) = 7.6e-6, which becomes the decay's relative
    error, and subnormal results add an absolute error of a few 2^-149."""
    gates = [-0.25] * 64 + [-0.87] * 192
    pairs = [(64 + 92, 64), (64 + 100, 64), (64 + 115, 64), (64 + 120, 64), (64 + 127, 64)]
    decays = _single_pair_decays(gates, pairs)
    for (i, j), (dec_a, dec_qk, _, exact) in zip(pairs, decays):
        log = abs(math.log(exact))
        for got in (dec_a, dec_qk):
            bound = 4 * (_ulp32(log) + 2.0 ** -24) * exact + 4 * 2.0 ** -149
            assert abs(got - exact) <= bound, ((i, j), got, exact)


def test_reference_matches_token_loop():
    for case in (Case(W=3, R=2, C=8, sph=12), Case(W=4, R=3, C=8, sph=6, valid="C-1"),
                 Case(W=5, R=4, C=16, sph=20, gate="write"), Case(W=1, R=1, C=4, sph=3)):
        d = make_inputs(case)
        ref = reference_phase1(**d, **d["kwargs"])
        N = d["k_idx"].shape[0]
        off = d["kwargs"]["stream_offset"]
        for c in range(N):
            vt = case.valid_last if c >= N - case.P else case.C
            base = (c % case.P) * off
            loop = token_loop_reference((d["k_idx"][c] - base).cpu(), (d["q_idx"][c] - base).cpu(),
                                        d["k_val"][c].double().cpu(), d["q_val"][c].double().cpu(),
                                        d["g"][c].cpu(), vt)
            for f, want in loop.items():
                got = ref[f][c].cpu()
                assert torch.allclose(got, want, rtol=1e-12, atol=1e-12), (case.id(), c, f)


def test_repeated_write_slots_are_counted_and_decay_once():
    case = Case(W=8, R=4, C=16, sph=32, dtype="float32", layout="single")
    d = make_inputs(case)
    k = d["k_idx"].clone()
    k[0, 3, 5] = k[0, 3, 2]
    k[0, 7, 1] = k[0, 7, 6]
    k[1, 0, 7] = k[1, 0, 0]
    d["k_idx"] = k
    out, repeats = run_kernel(d)
    assert repeats.item() == 3
    _check(case, out, reference_phase1(**d, **d["kwargs"]))


def test_payload_follows_the_sorted_layout():
    case = Case(W=5, R=3, C=17, sph=40, valid="C-1", layout="streams", dist="sticky")
    d = make_inputs(case)
    out, _ = run_kernel(d)
    N, C, W = d["k_idx"].shape
    R = d["q_idx"].shape[2]
    n = C * (W + R)
    assert out.payload.shape == (N, (n + 15) // 16 * 16)
    off = (torch.arange(N, device="cuda") % case.P).view(N, 1, 1) * case.sph
    slots = torch.cat([d["k_idx"] - off, d["q_idx"] - off], 2).reshape(N, n)
    e = torch.arange(n, device="cuda")
    t, j = e // (W + R), e % (W + R)
    last_chunks = (torch.arange(N, device="cuda") >= N - case.P).view(N, 1)
    invalid = (t.view(1, n) >= case.valid_last) & last_chunks
    key = torch.where(invalid, 1 << 40, slots) * n + e
    order = key.argsort(dim=1)
    lane = torch.where(j < W, j, j - W)
    pay = (t << 9) | ((j >= W).long() << 8) | lane | (invalid.long() << 17)
    pay = pay.gather(1, order)
    sk = slots.gather(1, order)
    inv = invalid.gather(1, order)
    head = torch.ones_like(inv)
    head[:, 1:] = (sk[:, 1:] != sk[:, :-1]) | inv[:, 1:] | inv[:, :-1]
    is_k = ((pay >> 8) & 1 == 0) & ~inv
    gid = head.long().cumsum(1)
    later_k = torch.zeros_like(inv)
    for r in range(n - 2, -1, -1):
        later_k[:, r] = (gid[:, r + 1] == gid[:, r]) & (is_k[:, r + 1] | later_k[:, r + 1])
    pay = pay | (head.long() << 31) | ((is_k & ~later_k).long() << 30)
    got = out.payload[:, :n].long() & 0xFFFFFFFF
    assert torch.equal(got, pay)


def _check_maps(out, d):
    """pos_k, pos_q, q_last, q_only, spay_k and spay_q equal group_maps_reference and
    sorted_kind_rows_reference of the sorted payload."""
    _, C, W = d["k_idx"].shape
    R = d["q_idx"].shape[2]
    ref = phase1_reference.group_maps_reference(out.payload, C, W, R)
    ref += phase1_reference.sorted_kind_rows_reference(out.payload, C, W, R)
    ref += (phase1_reference.sorted_rank_reference(out.payload, C, W, R),)
    for f, want in zip(("pos_k", "pos_q", "q_last", "q_only", "spay_k", "spay_q", "spay_rank"),
                       ref):
        assert bits_equal(getattr(out, f), want), f


# ------------------------------------------------------------------ bitwise reproducibility
@dataclasses.dataclass(frozen=True)
class Hot:
    """Collision-heavy chunks: every token draws its slots from a pool of `pool` slot ids
    spread over [0, num_keys), so most groups hold many tokens; repeat_k adds repeated K
    entries of one token (contract violations, counted), dup_q lets a token read one slot
    with several lanes."""
    C: int
    W: int
    R: int
    pool: int
    num_keys: int
    index: str = "int64"
    dtype: str = "float32"
    valid_last: int = 0          # 0: C
    P: int = 2
    steps: int = 2
    repeat_k: bool = False
    dup_q: bool = False
    gate: str = "token"

    def id(self):
        return (f"C{self.C}-W{self.W}-R{self.R}-pool{self.pool}-keys{self.num_keys}-"
                f"{self.index}-{self.dtype}-v{self.valid_last or self.C}-P{self.P}-"
                f"{'rep' if self.repeat_k else 'norep'}-{'dupq' if self.dup_q else 'q'}-"
                f"{self.gate}")


def hot_inputs(case, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    P, C, W, R = case.P, case.C, case.W, case.R
    N = P * case.steps
    rows = N * C
    ids = torch.randperm(case.num_keys, device="cuda", generator=gen)[: case.pool]
    k = ids[torch.rand(rows, case.pool, device="cuda", generator=gen).argsort(-1)[:, :W]]
    if case.dup_q:
        q = ids[torch.randint(0, case.pool, (rows, R), device="cuda", generator=gen)]
    else:
        q = ids[torch.rand(rows, case.pool, device="cuda", generator=gen).argsort(-1)[:, :R]]
    k, q = k.view(N, C, W), q.view(N, C, R)
    if case.repeat_k:
        rep = torch.rand(N, C, W, device="cuda", generator=gen) < 0.1
        rep[..., 0] = False
        k = torch.where(rep, k[..., :1].expand(N, C, W), k)
    stream = (torch.arange(N, device="cuda") % P).view(N, 1, 1) * case.num_keys
    idx_dtype, dtype = getattr(torch, case.index), getattr(torch, case.dtype)
    gw = W if case.gate == "write" else 1
    return dict(
        k_idx=(k + stream).to(idx_dtype), q_idx=(q + stream).to(idx_dtype),
        k_val=torch.randn(N, C, W, device="cuda", generator=gen).to(dtype),
        q_val=torch.randn(N, C, R, device="cuda", generator=gen).to(dtype),
        g=-torch.rand(N, C, gw, device="cuda", generator=gen) * 0.2,
        kwargs=dict(num_parallel=P, stream_offset=case.num_keys, num_keys=case.num_keys,
                    valid_last=case.valid_last or C),
    )


HOT = [
    Hot(16, 8, 8, 10, 1 << 12, repeat_k=True, dup_q=True, steps=3),
    Hot(37, 16, 24, 20, 1 << 20, index="int32", dtype="bfloat16", valid_last=32, P=3,
        dup_q=True, gate="write"),
    Hot(64, 64, 64, 72, 65536, dtype="bfloat16", repeat_k=True, dup_q=True),
    Hot(64, 48, 64, 64, 1 << 17, index="int32", valid_last=1, dup_q=True, gate="write"),
    Hot(128, 32, 48, 40, 1 << 17, index="int32", valid_last=100, repeat_k=True, dup_q=True),
    Hot(17, 3, 48, 50, 65536, dtype="bfloat16", valid_last=9, P=4),
    Hot(256, 64, 64, 80, 4096, dtype="bfloat16", valid_last=200, steps=1, dup_q=True),
    Hot(256, 4, 4, 4, 1 << 18, index="int32", steps=1, repeat_k=True, dup_q=True),
    Hot(200, 16, 16, 18, 1 << 16, valid_last=150, steps=1, gate="write"),
    Hot(256, 160, 160, 2000, 1 << 17, index="int32", dtype="bfloat16", valid_last=250,
        steps=1, repeat_k=True, dup_q=True),     # n > 2^16: maps not staged
    Hot(15, 33, 31, 40, 1 << 10, repeat_k=True, dup_q=True, gate="write"),
    Hot(48, 255, 1, 300, 1 << 16, dtype="bfloat16", valid_last=47, dup_q=True),
    Hot(1, 4, 4, 4, 64, P=3, steps=4, repeat_k=True, dup_q=True),
    Hot(5, 8, 8, 9, 1 << 9, valid_last=2, P=2, steps=3, repeat_k=True, dup_q=True),
]


def _pair_bound_failures(out, d, case):
    """A and QK against the float64 reference, per cell, within the rounding of any order of
    summation: |got - ref| <= (m + 0.8 C + 8) 2^-24 S, where m is the cell's number of terms
    and S the sum of their magnitudes. A term's decay is exact to about 0.8 C 2^-24 relative:
    it is the exp of a difference of compensated log-cumsums of at most C gates in [-0.2, 0]."""
    ref = reference_phase1(**d, **d["kwargs"])
    ab = reference_phase1(**{**d, "k_val": d["k_val"].abs(), "q_val": d["q_val"].abs()},
                          **d["kwargs"])
    count = reference_phase1(**{**d, "k_val": torch.ones_like(d["k_val"]),
                                "q_val": torch.ones_like(d["q_val"]),
                                "g": torch.zeros_like(d["g"])}, **d["kwargs"])
    bad = {}
    for f in ("a", "qk"):
        bound = (count[f] + 0.8 * case.C + 8) * 2.0 ** -24 * ab[f] + 2.0 ** -140
        err = (getattr(out, f).double() - ref[f]).abs()
        if not bool((err <= bound).all()):
            bad[f] = f"max err/bound {(err / bound).max().item():.3g}"
    return bad


@pytest.mark.parametrize("case", HOT, ids=[c.id() for c in HOT])
def test_repeated_calls_are_bitwise_identical(case):
    """Five calls on collision-heavy chunks (long groups, repeated K entries, duplicated Q
    entries) return identical bytes for every output, the group maps equal their reference,
    and the triangles agree with the float64 reference."""
    d = hot_inputs(case)
    runs = [run_kernel(d)[0] for _ in range(5)]
    for r in runs[1:]:
        for f in runs[0]._fields:
            assert bits_equal(getattr(runs[0], f), getattr(r, f)), (case.id(), f)
    _check_maps(runs[0], d)
    bad = _pair_bound_failures(runs[0], d, case)
    assert not bad, f"{case.id()}: {bad}"


@pytest.mark.parametrize("C,W,R", [(2, 256, 2), (1, 254, 2), (1, 255, 2), (1, 256, 2)])
def test_exact_term_counts_with_repeats_and_duplicates(C, W, R):
    """Every K and Q entry of every token on one slot (weights 1, gates 0, so every term is
    exactly 1): A[i, j] = W * W for j < i, QK[i, j] = R * W for j <= i, exactly, across the
    walks' count saturation at 255 (repeated K entries of one token, duplicated Q entries)."""
    k_idx = torch.zeros(1, C, W, dtype=torch.int64, device="cuda")
    q_idx = torch.zeros(1, C, R, dtype=torch.int64, device="cuda")
    rep = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = _p1().phase1_forward(k_idx, q_idx, torch.ones(1, C, W, device="cuda"),
                               torch.ones(1, C, R, device="cuda"),
                               torch.zeros(1, C, 1, device="cuda"), rep, num_parallel=1,
                               stream_offset=0, num_keys=1, valid_last=C)
    lower = torch.ones(C, C, device="cuda").tril()
    assert bits_equal(out.a[0], (lower - torch.eye(C, device="cuda")) * float(W * W))
    assert bits_equal(out.qk[0], lower * float(R * W))
    assert rep.item() == C * (W - 1)


_GRAPH_CASES = [("zipf", make_inputs, Case(W=64, R=64, C=64, dist="zipf", dtype="bfloat16",
                                            steps=4))] + [
    ("hot", hot_inputs, HOT[i]) for i in (2, 6, 9)]  # repeats, dups, split, unstaged maps


@pytest.mark.parametrize("make,case", [c[1:] for c in _GRAPH_CASES],
                         ids=[f"{c[0]}-{c[2].id()}" for c in _GRAPH_CASES])
def test_graph_replay_matches_eager(make, case):
    d = make(case)
    p1 = _p1()
    repeats = torch.zeros(1, dtype=torch.int64, device="cuda")

    def fwd():
        return p1.phase1_forward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"], d["g"],
                                 repeats, **d["kwargs"])

    eager = fwd()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fwd()
    for _ in range(2):
        graph.replay()
        for f in eager._fields:
            assert bits_equal(getattr(eager, f), getattr(captured, f)), f


def _brute_maps(payload, C, W, R):
    """The group maps and the sorted write / read rows by a literal walk over every group
    of every sorted row (CPU)."""
    p1 = _p1()
    N, n = payload.shape[0], C * (W + R)
    pk = torch.zeros(N, C, W, dtype=torch.int64)
    pq = torch.zeros(N, C, R, dtype=torch.int64)
    ql = torch.zeros(N, C, R, dtype=torch.int8)
    qo = torch.zeros(N, C, R, dtype=torch.int8)
    sk = torch.zeros(N, C * W, dtype=torch.int64)
    sq = torch.zeros(N, C * R, dtype=torch.int64)
    rank = torch.zeros(N, (n + 31) // 32, 2, dtype=torch.int64)
    rows = (payload[:, :n].cpu().to(torch.int64) & 0xFFFFFFFF).tolist()
    for c, row in enumerate(rows):
        groups = []
        for r, p in enumerate(row):
            if p >> 31 & 1:
                groups.append([])
            groups[-1].append((r, p))
        # Rank table: write entries before each block of 32 positions, and the block's mask.
        before = 0
        for b in range(rank.shape[1]):
            mask = sum(1 << i for i, p in enumerate(row[32 * b:32 * b + 32]) if not p >> 8 & 1)
            rank[c, b, 0], rank[c, b, 1] = before, mask
            before += bin(mask).count("1")
        # Sorted rows: each kind's entries in row order; a group's run of them is marked
        # first / last (a padding entry is a group of its own).
        for kind, out in ((0, sk), (1, sq)):
            j = 0
            for grp in groups:
                mine = [p for _, p in grp if (p >> 8 & 1) == kind]
                for i, p in enumerate(mine):
                    out[c, j] = ((p & 0x3FFFF) | min(i, 4095) << 18 | (i == 0) << 31 |
                                 (i == len(mine) - 1) << 30)
                    j += 1
            assert j == out.shape[1]
        for grp in groups:
            for kind in (0, 1):
                peers = [r for r, p in grp if (p >> 8 & 1) == kind and not p >> 17 & 1]
                for r, p in grp:
                    lane, isq, tok, inv = p & 0xFF, p >> 8 & 1, p >> 9 & 0xFF, p >> 17 & 1
                    if isq != kind:
                        continue
                    after = sum(1 for x in peers if x > r)
                    value = r if inv else p1.pos_pack(r, after, len(peers))
                    (pq if isq else pk)[c, tok, lane] = value
                    if isq and not inv:
                        ql[c, tok, lane] = after == 0
                        qo[c, tok, lane] = len(peers) == 1
    signed = lambda t: torch.where(t >= 1 << 31, t - (1 << 32), t).to(torch.int32)  # noqa: E731
    return (pk.to(torch.int32), pq.to(torch.int32), ql, qo, signed(sk), signed(sq),
            signed(rank))


def test_group_maps_match_a_literal_walk():
    """Packed positions (peers after, group size, saturating at 127), the read-group flags
    and the sorted write / read rows against a literal walk, including a group of more than
    127 peers and padding tokens."""
    cases = [(hot_inputs, HOT[0]), (hot_inputs, HOT[-1]), (make_inputs, Case(
        W=5, R=3, C=17, sph=40, valid="C-1", layout="streams", dist="sticky")),
        (hot_inputs, Hot(160, 4, 4, 4, 1 << 12, steps=1, P=1, repeat_k=True, dup_q=True))]
    for make, case in cases:
        d = make(case)
        out = run_kernel(d)[0]
        _, C, W = d["k_idx"].shape
        want = _brute_maps(out.payload, C, W, d["q_idx"].shape[2])
        for f, w in zip(("pos_k", "pos_q", "q_last", "q_only", "spay_k", "spay_q", "spay_rank"),
                        want):
            assert bits_equal(getattr(out, f).cpu(), w), (case.id(), f)
    assert int((out.pos_k >> 24).max()) == 127   # a saturated group size was exercised


# ------------------------------------------------------ group outputs from the slot ids
def _signed32(v):
    return v - (1 << 32) if v >= 1 << 31 else v


def _group_outputs_literal(k_rows, q_rows, t_valid, W, R):
    """One chunk's group-derived outputs from its slot ids alone (pure Python lists).

    k_rows / q_rows: [C][W] / [C][R] slots relative to the chunk's stream; t_valid: valid
    tokens. The entries (entry order t * (W + R) + lane, K lanes before Q lanes) are sorted
    stably by slot, padding tokens' entries last in entry order. A group is a run of valid
    entries on one slot; every padding entry is a group of its own. Nothing here reads a
    kernel output.
    """
    C = len(k_rows)
    wr, n = W + R, C * (W + R)
    entries = []                       # (valid, slot, entry order, is_q, token, lane)
    for t in range(C):
        for q, rows, width in ((0, k_rows, W), (1, q_rows, R)):
            for lane in range(width):
                entries.append((t < t_valid, rows[t][lane], t * wr + q * W + lane, q, t, lane))
    entries.sort(key=lambda x: (0, x[1], x[2]) if x[0] else (1, 0, x[2]))
    res = dict(payload=[0] * n, pos_k=[0] * (C * W), pos_q=[0] * (C * R),
               is_last=[0] * (C * W), is_only=[0] * (C * W), q_last=[0] * (C * R),
               q_only=[0] * (C * R), spay_k=[0] * (C * W), spay_q=[0] * (C * R), repeats=0)
    starts = [i for i in range(n) if i == 0 or not entries[i][0] or not entries[i - 1][0]
              or entries[i][1] != entries[i - 1][1]] + [n]
    kind_index = [0, 0]
    for g0, g1 in zip(starts[:-1], starts[1:]):
        grp = entries[g0:g1]
        size = [sum(1 for e in grp if e[0] and e[3] == q) for q in (0, 1)]
        seen = [0, 0]
        k_tokens = set()
        for i, (valid, _, _, q, t, lane) in zip(range(g0, g1), grp):
            bits = lane | q << 8 | t << 9 | (0 if valid else 1) << 17
            o = t * (R if q else W) + lane
            if valid:
                before, after = seen[q], size[q] - seen[q] - 1
                seen[q] += 1
                pos = i | min(after, 127) << 17 | min(size[q], 127) << 24
                elem = bits | min(before, 4095) << 18 | (before == 0) << 31 | (after == 0) << 30
                last, only = int(after == 0), int(after == 0 and size[q] == 1)
                if q == 0:
                    res["repeats"] += t in k_tokens
                    k_tokens.add(t)
            else:
                after, pos, elem, last, only = -1, i, bits | 1 << 31 | 1 << 30, 0, 0
            res["payload"][i] = _signed32(bits | (i == g0) << 31 |
                                          (valid and q == 0 and after == 0) << 30)
            name = ("pos_q", "q_last", "q_only", "spay_q") if q else \
                   ("pos_k", "is_last", "is_only", "spay_k")
            res[name[0]][o], res[name[1]][o], res[name[2]][o] = pos, last, only
            res[name[3]][kind_index[q]] = _signed32(elem)
            kind_index[q] += 1
    rank, before = [], 0
    for b in range(0, n, 32):
        mask = sum(1 << (i - b) for i in range(b, min(b + 32, n)) if entries[i][3] == 0)
        rank.append([before, _signed32(mask)])
        before += bin(mask).count("1")
    res["spay_rank"] = rank
    return res


def _maps_from_slot_ids(d):
    """``_group_outputs_literal`` of every chunk of the inputs, as tensors shaped like the
    kernel outputs (payload: its first C * (W + R) positions), and the total repeat count."""
    kw = d["kwargs"]
    P, offset, valid_last = kw["num_parallel"], kw["stream_offset"], kw["valid_last"]
    k_idx, q_idx = d["k_idx"].cpu().long(), d["q_idx"].cpu().long()
    N, C, W = k_idx.shape
    R = q_idx.shape[2]
    per = [_group_outputs_literal((k_idx[c] - (c % P) * offset).tolist(),
                                  (q_idx[c] - (c % P) * offset).tolist(),
                                  valid_last if c >= N - P else C, W, R) for c in range(N)]
    shapes = dict(payload=(N, -1), pos_k=(N, C, W), pos_q=(N, C, R), is_last=(N, C, W),
                  is_only=(N, C, W), q_last=(N, C, R), q_only=(N, C, R), spay_k=(N, -1),
                  spay_q=(N, -1), spay_rank=(N, -1, 2))
    want = {f: torch.tensor([p[f] for p in per], dtype=torch.int8 if "last" in f or
                            "only" in f else torch.int32).view(shape)
            for f, shape in shapes.items()}
    want["repeats"] = sum(p["repeats"] for p in per)
    return want


def _one_chunk(k, q, valid_last, num_keys):
    """Inputs of one chunk (P = 1) with the given [C, W] / [C, R] slot ids, weights 1 and
    gates 0."""
    C, W = k.shape
    R = q.shape[1]
    return dict(k_idx=k.view(1, C, W).cuda(), q_idx=q.view(1, C, R).cuda(),
                k_val=torch.ones(1, C, W, device="cuda"), q_val=torch.ones(1, C, R, device="cuda"),
                g=torch.zeros(1, C, 1, device="cuda"),
                kwargs=dict(num_parallel=1, stream_offset=0, num_keys=num_keys,
                            valid_last=valid_last))


_SLOT_ID_CASES = {
    **{f"hot-{c.id()}": (lambda c=c: hot_inputs(c)) for c in HOT},
    "sticky-streams": lambda: make_inputs(Case(W=5, R=3, C=17, sph=40, valid="C-1",
                                               layout="streams", dist="sticky")),
    "overlap-shared": lambda: make_inputs(Case(W=48, R=64, C=17, sph=300, valid="1",
                                               layout="shared", dist="overlap")),
    # Three tiles with a two-entry tail, groups of 4097 entries of each kind.
    "one-slot-17x241x241": lambda: _one_chunk(torch.zeros(17, 241, dtype=torch.long),
                                              torch.zeros(17, 241, dtype=torch.long), 17, 1),
    "one-slot-17x241x241-ragged": lambda: _one_chunk(torch.zeros(17, 241, dtype=torch.long),
                                                     torch.zeros(17, 241, dtype=torch.long),
                                                     9, 1),
    # A write group across 2^15 entries (wide scan element; distance saturated).
    "one-slot-129x256x1": lambda: _one_chunk(torch.zeros(129, 256, dtype=torch.long),
                                             torch.zeros(129, 1, dtype=torch.long), 129, 1),
    # 2^17 entries, every write before every read: the read half's rank blocks count 65536.
    "writes-before-reads-256x256x256": lambda: _one_chunk(
        torch.arange(65536).view(256, 256), 65536 + torch.arange(65536).view(256, 256), 256,
        1 << 17),
}


@pytest.mark.parametrize("name", list(_SLOT_ID_CASES))
def test_group_outputs_match_the_slot_ids(name):
    """The sorted payload (order, heads, last-K flags), pos_k / pos_q, the write and read
    group flags, the sorted rows, the rank table and the repeat count against groups rebuilt
    from the raw slot ids (``_maps_from_slot_ids``), not from the payload's head bits."""
    d = _SLOT_ID_CASES[name]()
    if name.startswith("writes-before-reads"):
        optin = getattr(torch.cuda.get_device_properties(0), "shared_memory_per_block_optin",
                        None)
        if optin is not None and optin < 210 * 1024:
            pytest.skip("C = W = R = 256 needs more shared memory per block than this GPU has")
    out, repeats = run_kernel(d)
    want = _maps_from_slot_ids(d)
    n = want["payload"].shape[1]
    assert bits_equal(out.payload[:, :n].cpu(), want["payload"]), "payload"
    for f in ("pos_k", "pos_q", "is_last", "is_only", "q_last", "q_only", "spay_k", "spay_q",
              "spay_rank"):
        assert bits_equal(getattr(out, f).cpu(), want[f]), f
    assert int(repeats) == want["repeats"]


# ----------------------------------------------------- GPUs with less shared memory per block
A100_SMEM = 166912      # cudaDevAttrMaxSharedMemoryPerBlockOptin on A100 (163 KiB)
SMEM_160K = 163840      # 160 KiB, below A100's
_WIDE_C256 = Hot(256, 128, 128, 160, 1 << 16, valid_last=200, steps=1, repeat_k=True, dup_q=True)


def _lean_config(case, limit, monkeypatch):
    """Simulates a GPU granting ``limit`` bytes per block and returns the pair kernel's
    layout for the case there; skips where this GPU cannot run the shape at all."""
    p1 = _p1()
    monkeypatch.setattr(p1, "SMEM_LIMIT_OVERRIDE", limit)
    cfg = p1.pairs_config(case.C, case.W, case.R)
    if cfg["need"] > cfg["limit"]:
        pytest.skip(f"this GPU grants {cfg['limit']} bytes per block, too few for C={case.C}")
    assert cfg["lean"] and cfg["split"], cfg
    return cfg


@pytest.mark.parametrize("limit", [A100_SMEM, SMEM_160K])
@pytest.mark.parametrize("case", [_WIDE_C256, HOT[9]], ids=lambda c: c.id())
def test_low_shared_memory_pairs_are_bitwise_equal(case, limit, monkeypatch):
    """Under A100's shared memory per block, C = 256 with 128 or 160 slots cannot keep the
    pair kernel's count bytes beside a triangle, so the lean kernel runs every walk to its
    group's head instead. That adds the same terms in the same order: every output equals
    the full kernel's bit for bit (where this GPU can run the full kernel)."""
    if _p1().pairs_config(case.C, case.W, case.R)["lean"]:
        pytest.skip("this GPU runs the lean kernel natively: no full kernel to compare with")
    d = hot_inputs(case)
    full, full_repeats = run_kernel(d)
    _lean_config(case, limit, monkeypatch)
    lean, lean_repeats = run_kernel(d)
    for f in full._fields:
        assert bits_equal(getattr(full, f), getattr(lean, f)), f
    assert int(full_repeats) == int(lean_repeats)


@pytest.mark.parametrize("case", [_WIDE_C256, HOT[9]], ids=lambda c: c.id())
def test_low_shared_memory_pairs_match_the_reference(case, monkeypatch):
    """The lean kernel at C = 256 under A100's limit (natively on A100): A and QK against
    the float64 reference within the rounding of any summation order, the group maps and
    sorted rows against theirs."""
    _lean_config(case, A100_SMEM, monkeypatch)
    d = hot_inputs(case)
    out, _ = run_kernel(d)
    _check_maps(out, d)
    assert not _pair_bound_failures(out, d, case)


SMEM_48K = 49152        # the least Phase 1 runs with (every CUDA GPU grants it)


@pytest.mark.parametrize("case", [
    Case(W=48, R=64, C=150, dist="zipf", sph=65536),
    Case(W=64, R=48, C=150, gate="write", valid="C-1", dist="sticky", sph=4096,
         dtype="bfloat16", index="int32"),
], ids=lambda c: c.id())
def test_low_shared_memory_paths_match_the_reference(case, monkeypatch):
    """Under a 48 KiB limit these chunks take the lean pair kernel (and stage fewer scan
    fields): every output matches the float64 reference at the usual tolerances, the group
    maps their reference, and every output the full path's bits."""
    d = make_inputs(case)
    full = run_kernel(d)[0]
    _lean_config(case, SMEM_48K, monkeypatch)
    out, repeats = run_kernel(d)
    _check(case, out, reference_phase1(**d, **d["kwargs"]))
    _check_maps(out, d)
    assert repeats.item() == 0
    for f in full._fields:
        assert bits_equal(getattr(full, f), getattr(out, f)), f


def test_subnormal_pair_sums():
    """Two normal terms whose sum is subnormal: k_val [2^-126, -(2^-126 + 2^-149)] and
    [1, 1] on the same two slots, gates 0, so A[1, 0] = -2^-149. The pair sum keeps it
    exactly (0x80000001); a float atomic add could flush it to a sign-preserving zero
    (0x80000000, PTX atom.add.f32)."""
    k_idx = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64, device="cuda")
    q_idx = torch.tensor([[[0], [1]]], dtype=torch.int64, device="cuda")
    tiny = 2.0 ** -126
    k_val = torch.tensor([[[tiny, -(tiny + 2.0 ** -149)], [1.0, 1.0]]], device="cuda")
    d = dict(k_idx=k_idx, q_idx=q_idx, k_val=k_val, q_val=torch.ones(1, 2, 1, device="cuda"),
             g=torch.zeros(1, 2, 1, device="cuda"),
             kwargs=dict(num_parallel=1, stream_offset=0, num_keys=2, valid_last=2))
    out = run_kernel(d)[0]
    bits = int(out.a[0, 1, 0].view(torch.int32)) & 0xFFFFFFFF
    assert bits == 0x80000001, hex(bits)


@pytest.mark.parametrize("quant", ["none", "fp8", "int4"])
def test_backward_restores_the_bank_and_both_paths_launch_phase1(quant):
    """A saving call of sdm_write_read (forward and backward) restores the bank: exactly
    without snapshot quantization, and the written rows to their dequantized first snapshot
    (the input rows, quantized) with it; every other row, the padding tokens' row 0 among
    them, stays bitwise unchanged. The saving and the no-grad calls launch Phase 1's scan and
    pair kernels."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T, C, W, hd, sph = 2, 150, 64, 32, 64, 4096
    d = make_op_inputs(P, T, W, W, hd, sph, torch.float32, seed=3, slots="hot")
    # Bank row 0 (the id staged for padding tokens) written by no valid entry: swap its id with
    # an unwritten one, so restoring it checks that padding writes nothing.
    written = torch.unique(d["k_idx"].reshape(-1))
    spare = int(torch.isin(torch.arange(sph, device="cuda"), written).logical_not().nonzero()[0])
    for n in ("k_idx", "q_idx"):
        d[n] = torch.where(d[n] == 0, spare, torch.where(d[n] == spare, 0, d[n]))
    assert not bool((d["k_idx"] == 0).any())
    names = ("k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")
    grads = ("memory", "k_val", "v", "beta", "g", "q_val")
    kw = dict(chunk_size=C, num_memory_slots=sph, num_parallel=P, snapshot_quant=quant)

    def saving():
        leaves = {n: d[n].clone().requires_grad_(n in grads) for n in ("memory",) + names}
        out = sdm_write_read(leaves["memory"], *(leaves[n] for n in names), **kw)
        out.square().sum().backward()
        restored = d["memory"].clone()
        rows = torch.unique(d["k_idx"].reshape(-1))
        assert rows.numel() < restored.shape[0]   # untouched rows are checked too
        if quant != "none":
            snap = reference_quantize(d["memory"][rows], quant)
            restored[rows] = reference_dequantize(*snap, quant, restored.dtype, hd)
        assert bits_equal(leaves["memory"].detach(), restored), quant

    def no_grad():
        with torch.no_grad():
            sdm_write_read(d["memory"].clone(), *(d[n] for n in names), **kw)

    for call in (saving, no_grad):
        call()
        _, launched = profiled_kernel_names(call)
        # The pair kernel is p1_pairs_kernel or the hot/cold kernel p1_pairs_hc_kernel (chosen by
        # shapes and arch).
        for kernel in (r"p1_pairs(_hc)?_kernel", "p1_scan_kernel"):
            assert any(re.search(rf"\b{kernel}\b", n) for n in launched), (call.__name__, kernel)


def test_sorted_rows_saturate_the_group_distance():
    """One group of 8192 write and 8192 read entries (every entry on one slot): the sorted
    rows hold every entry once, FIRST only on the first element, LAST only on the last, and
    the distance back to FIRST saturates at 4095 from element 4095 on."""
    C, W, R = 32, 256, 256
    k_idx = torch.zeros(1, C, W, dtype=torch.int64, device="cuda")
    q_idx = torch.zeros(1, C, R, dtype=torch.int64, device="cuda")
    out = _p1().phase1_forward(k_idx, q_idx, torch.ones(1, C, W, device="cuda"),
                               torch.ones(1, C, R, device="cuda"),
                               torch.zeros(1, C, 1, device="cuda"),
                               torch.zeros(1, dtype=torch.int64, device="cuda"), num_parallel=1,
                               stream_offset=0, num_keys=1, valid_last=C)
    ref = phase1_reference.sorted_kind_rows_reference(out.payload, C, W, R)
    for row, want, width, isq in ((out.spay_k, ref[0], W, 0), (out.spay_q, ref[1], R, 1)):
        assert bits_equal(row, want)
        e = row[0].to(torch.int64) & 0xFFFFFFFF
        m = C * width
        assert sorted((((e >> 9) & 0xFF) * width + (e & 0xFF)).tolist()) == list(range(m))
        assert bool((((e >> 8) & 1) == isq).all()) and not bool(((e >> 17) & 1).any())
        j = torch.arange(m, device="cuda")
        assert torch.equal((e >> 31) & 1, (j == 0).to(torch.int64))
        assert torch.equal((e >> 30) & 1, (j == m - 1).to(torch.int64))
        assert torch.equal((e >> 18) & 0xFFF, j.clamp(max=4095))


@pytest.mark.parametrize("case", [HOT[2], Hot(96, 64, 64, 80, 1 << 16, repeat_k=True,
                                                 dup_q=True)], ids=lambda c: c.id())
def test_read_groups_can_be_skipped(case):
    """need_read_groups=False leaves q_last, q_only and spay_q out (None) and every other
    output byte-identical (both shared-memory layouts: every field staged, and per field)."""
    d = hot_inputs(case)
    full = run_kernel(d)[0]
    rep = torch.zeros(1, dtype=torch.int64, device="cuda")
    lean = _p1().phase1_forward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"], d["g"], rep,
                                need_read_groups=False, **d["kwargs"])
    assert lean.q_last is None and lean.q_only is None and lean.spay_q is None
    for f in full._fields:
        if f not in ("q_last", "q_only", "spay_q"):
            assert bits_equal(getattr(full, f), getattr(lean, f)), f


@pytest.mark.slow
def test_more_than_2_31_sorted_entries():
    """N * n > 2^31 entries: every kernel addresses rows with 64-bit offsets. Sampled
    chunks, including both sides of the 2^31 boundary, match the reference."""
    require_free_memory(124)
    P, C, W = 16, 128, 64
    n = C * 2 * W
    steps = (2 ** 31 // n) // P + 1
    N = P * steps
    assert N * n > 2 ** 31
    sph = 65536
    gen = torch.Generator(device="cuda").manual_seed(0)
    stream = (torch.arange(N, device="cuda", dtype=torch.int32) % P).view(N, 1, 1)
    k_idx = torch.randint(0, sph - W + 1, (N, C, W), device="cuda", generator=gen,
                          dtype=torch.int32).sort(-1).values
    k_idx += torch.arange(W, device="cuda", dtype=torch.int32) + stream * sph
    q_idx = torch.randint(0, sph, (N, C, W), device="cuda", generator=gen,
                          dtype=torch.int32) + stream * sph
    k_val = torch.randn(N, C, W, device="cuda", generator=gen, dtype=torch.bfloat16)
    q_val = torch.randn(N, C, W, device="cuda", generator=gen, dtype=torch.bfloat16)
    g = -torch.rand(N, C, 1, device="cuda", generator=gen) * 0.2
    kw = dict(num_parallel=P, stream_offset=sph, num_keys=sph, valid_last=C - 3)
    rep = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = _p1().phase1_forward(k_idx, q_idx, k_val, q_val, g, rep, **kw)
    boundary = 2 ** 31 // n
    for c in (0, boundary - 1, boundary, N - P - 1, N - 1):
        c0 = c - c % P
        sl = slice(c0, c0 + P)
        ref = reference_phase1(k_idx[sl], q_idx[sl], k_val[sl], q_val[sl], g[sl],
                               **{**kw, "valid_last": kw["valid_last"] if c0 == N - P else C})
        for f in ("a", "qk", "log_k", "log_q", "rex"):
            assert rel_max(ref[f], getattr(out, f)[sl]) <= GATE_REL_MAX, (c, f)
