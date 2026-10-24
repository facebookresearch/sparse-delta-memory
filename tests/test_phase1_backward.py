# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Phase-1 backward (cuda/phase1_bwd_cuda.py) against float64 autograd.

The reference rebuilds the Phase-1 outputs in float64 from their definitions, joining the
entries of each (chunk, slot) with torch, and differentiates a weighted sum of them with
autograd. Each product reads its own copy of k_val (and q_val), so autograd also returns
the four pulled sums separately: d_row and d_col (from a), d_kvqk and d_qv (from qk). It
shares no sort, scan, walk or hand-written adjoint with the kernels.

The kernel returns only the total gradients, so the pulled sums are isolated through the
cotangents, which enter linearly: with only d_qk nonzero, grad_k_val = d_kvqk and
grad_q_val = d_qv; with only row i0 of d_a nonzero (in every chunk), grad_k_val is exactly
d_row at token i0 and that row's share of d_col before it, so one run per row gives d_row
and, summed, d_col.

Metric: max|got - ref| <= 1e-6 max|ref|. bf16 outputs (grad_k_val, grad_q_val of bf16 banks)
may also be off by half a bf16 ulp of the reference, i.e. be correctly rounded; the float32
cases check the same arithmetic without rounding. With only d_qk nonzero the gradient of x is
small by construction (a (K, Q) pair moves it only at the K entries strictly between the
two), so that run also checks that pairs with nothing between them stay out of the gate
scans, where their terms would cancel at the scale of the adjoints.
"""
import math

import pytest
import torch

from sdm_testlib import bits_equal, require_free_memory, requires_cuda
from test_phase1 import GRID, Case, _distinct_uniform, _rotated, _ulp32, make_inputs

pytestmark = requires_cuda

F64 = torch.float64
GATE_REL_MAX = 1e-6
COTANGENTS = ("d_a", "d_qk", "d_k_eff", "d_k_pd", "d_q_w")


def _p1():
    from sparse_delta_memory.cuda import phase1_bwd_cuda, phase1_cuda
    return phase1_cuda, phase1_bwd_cuda


def forward(d):
    repeats = torch.zeros(1, dtype=torch.int64, device="cuda")
    return _p1()[0].phase1_forward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"], d["g"],
                                   repeats, **d["kwargs"])


def backward(d, out, cot):
    return _p1()[1].phase1_backward(out.payload, out.log_k, out.log_q, out.rex, d["k_val"],
                                    d["q_val"], d["g"], *(cot[k] for k in COTANGENTS))


def make_cotangents(d, seed=1):
    """Random float32 cotangents, nonzero everywhere: outside the triangles and at invalid
    tokens too, where the backward must ignore them."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    N, C, W = d["k_idx"].shape
    R = d["q_idx"].shape[2]
    shapes = dict(d_a=(N, C, C), d_qk=(N, C, C), d_k_eff=(N, C, W), d_k_pd=(N, C, W),
                  d_q_w=(N, C, R))
    return {k: torch.randn(shapes[k], device="cuda", generator=gen) for k in COTANGENTS}


def token_valid(d):
    """[N, C] mask of the valid tokens."""
    N, C, _ = d["k_idx"].shape
    P, valid_last = d["kwargs"]["num_parallel"], d["kwargs"]["valid_last"]
    t_valid = torch.full((N, 1), C, device="cuda")
    t_valid[N - P:] = valid_last
    return torch.arange(C, device="cuda").view(1, C) < t_valid


# --------------------------------------------------------------------------- reference
def reference_backward(k_idx, q_idx, k_val, q_val, g, cotangent_sets, *, num_parallel,
                       stream_offset, valid_last, **_):
    """Float64 gradients for each dict of cotangents in ``cotangent_sets``: d_row, d_col,
    d_kvqk, d_qv, grad_k_val, grad_q_val and grad_g (the gradient of each write's x)."""
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
    first = ~((ks.unsqueeze(-1) == ks.unsqueeze(-2)) & earlier).any(-1) & valid.unsqueeze(-1)

    def leaf(t):
        return t.detach().double().clone().requires_grad_()

    kv = {name: leaf(k_val) for name in ("row", "col", "qk", "eff", "pd")}
    qv = {name: leaf(q_val) for name in ("qk", "w")}
    x = leaf(torch.where(first, g.double().expand(N, C, W), 0.0))

    keep_k = valid.unsqueeze(-1).expand(N, C, W)
    keep_q = valid.unsqueeze(-1).expand(N, C, R)
    nk, nq = int(keep_k.sum()), int(keep_q.sum())

    def coords(keep, width):
        n_, t_, _ = torch.meshgrid(chunk, tok, torch.arange(width, device=dev), indexing="ij")
        return n_[keep], t_[keep]

    kn, kt = coords(keep_k, W)
    qn, qt = coords(keep_q, R)
    e_chunk, e_tok = torch.cat([kn, qn]), torch.cat([kt, qt])
    e_slot = torch.cat([ks[keep_k], qs[keep_q]])
    is_k = torch.arange(nk + nq, device=dev) < nk
    # Ordered pairs (u, v) of entries sharing a (chunk, slot) in which v is a K entry; the
    # causal ones (v not after u) give the log-cumsums and both products.
    _, gid = torch.unique(torch.stack([e_chunk, e_slot]), dim=1, return_inverse=True)
    order = gid.argsort()
    size = torch.bincount(gid)
    start = torch.cumsum(size, 0) - size
    reps = size[gid]
    u = torch.arange(nk + nq, device=dev).repeat_interleave(reps)
    first_pair = (torch.cumsum(reps, 0) - reps).repeat_interleave(reps)
    member = torch.arange(u.numel(), device=dev) - first_pair
    v = order[start[gid[u]] + member]
    u, v = u[is_k[v]], v[is_k[v]]
    x_e = x[keep_k]
    total = torch.zeros(nk + nq, dtype=F64, device=dev).index_add(0, u, x_e[v])
    causal = e_tok[v] <= e_tok[u]
    u, v = u[causal], v[causal]
    tu, tv = e_tok[u], e_tok[v]
    log = torch.zeros_like(total).index_add(0, u, x_e[v])
    lk, lq, rex = log[:nk], log[nk:], total[:nk] - log[:nk]
    decay = torch.exp(log[u] - log[v])
    cell = (e_chunk[u] * C + tu) * C + tv

    def k_entries(t):
        return torch.cat([t[keep_k], torch.zeros(nq, dtype=F64, device=dev)])

    def q_entries(t):
        return torch.cat([torch.zeros(nk, dtype=F64, device=dev), t[keep_q]])

    def product(pair_sel, left, right):
        terms = left[u] * right[v] * decay
        return torch.zeros(N * C * C, dtype=F64, device=dev).index_add(
            0, cell[pair_sel], terms[pair_sel]).view(N, C, C)

    a = product(is_k[u] & (tv < tu), k_entries(kv["row"]), k_entries(kv["col"]))
    qk = product(~is_k[u] & (tv <= tu), q_entries(qv["qk"]), k_entries(kv["qk"]))
    k_eff = kv["eff"][keep_k] * torch.exp(lk)
    k_pd = kv["pd"][keep_k] * torch.exp(rex)
    q_w = qv["w"][keep_q] * torch.exp(lq)

    leaves = [*kv.values(), *qv.values(), x]
    results = []
    for i, cot in enumerate(cotangent_sets):
        loss = ((cot["d_a"].double() * a).sum() + (cot["d_qk"].double() * qk).sum()
                + (cot["d_k_eff"].double()[keep_k] * k_eff).sum()
                + (cot["d_k_pd"].double()[keep_k] * k_pd).sum()
                + (cot["d_q_w"].double()[keep_q] * q_w).sum())
        grads = torch.autograd.grad(loss, leaves, retain_graph=i + 1 < len(cotangent_sets),
                                    allow_unused=True)
        grads = [torch.zeros_like(t) if gr is None else gr for t, gr in zip(leaves, grads)]
        gk, gq = dict(zip(kv, grads[:5])), dict(zip(qv, grads[5:7]))
        results.append(dict(
            d_row=gk["row"], d_col=gk["col"], d_kvqk=gk["qk"], d_qv=gq["qk"],
            grad_k_val=sum(gk.values()), grad_q_val=sum(gq.values()),
            grad_g=torch.where(first, grads[7], 0.0)))
    return results


def token_loop_backward(kidx, qidx, kv, qv, g, cot, valid_t):
    """One chunk: the Phase-1 outputs as a literal token loop in float64, differentiated by
    autograd (grad_k_val, grad_q_val, grad of the per-write gate)."""
    C, W = kidx.shape
    R = qidx.shape[1]
    kv, qv = kv.double().clone().requires_grad_(), qv.double().clone().requires_grad_()
    gw = g.double().expand(C, W).clone().requires_grad_()
    acc, lk, lq = {}, {}, {}
    for t in range(valid_t):
        seen = set()
        for w in range(W):
            s = int(kidx[t, w])
            if s not in seen:
                acc[s] = acc.get(s, 0.0) + gw[t, w]
                seen.add(s)
            lk[t, w] = acc[s]
        for r in range(R):
            lq[t, r] = acc.get(int(qidx[t, r]), torch.zeros((), dtype=F64))
    loss = torch.zeros((), dtype=F64)
    for i in range(valid_t):
        for j in range(i + 1):
            for a in range(W):
                for b in range(W):
                    if j < i and kidx[i, a] == kidx[j, b]:
                        decay = torch.exp(lk[i, a] - lk[j, b])
                        loss = loss + cot["d_a"][i, j] * kv[i, a] * kv[j, b] * decay
            for a in range(R):
                for b in range(W):
                    if qidx[i, a] == kidx[j, b]:
                        decay = torch.exp(lq[i, a] - lk[j, b])
                        loss = loss + cot["d_qk"][i, j] * qv[i, a] * kv[j, b] * decay
    for t in range(valid_t):
        for w in range(W):
            rex = acc[int(kidx[t, w])] - lk[t, w]
            loss = loss + cot["d_k_eff"][t, w] * kv[t, w] * torch.exp(lk[t, w])
            loss = loss + cot["d_k_pd"][t, w] * kv[t, w] * torch.exp(rex)
        for r in range(R):
            loss = loss + cot["d_q_w"][t, r] * qv[t, r] * torch.exp(lq[t, r])
    gk, gq, gg = torch.autograd.grad(loss, (kv, qv, gw), allow_unused=True)
    zero = lambda t, ref: torch.zeros_like(ref) if t is None else t
    return dict(grad_k_val=zero(gk, kv), grad_q_val=zero(gq, qv), grad_g=zero(gg, gw))


# ------------------------------------------------------------------------------- cases
def _grid():
    cases = list(GRID)
    for i, (W, R, C) in enumerate(((127, 48, 64), (3, 127, 127), (127, 127, 17), (48, 3, 127))):
        cases.append(_rotated(i, W=W, R=R, C=C, dist=("sticky", "zipf")[i % 2]))
    return cases


BWD_GRID = _grid()


def _half_ulp_bf16(x):
    """Half the spacing of bfloat16 numbers at |x| (the largest correct-rounding error)."""
    _, exponent = torch.frexp(x.abs())
    return torch.ldexp(torch.ones_like(x), exponent - 9)


def _mismatch(name, ref, got, scale=None):
    """None when ``got`` matches the float64 ``ref`` within GATE_REL_MAX of ``scale``
    (default max|ref|); bf16 outputs may also be off by their rounding."""
    scale = ref.abs().max().item() if scale is None else scale
    err = (got.double() - ref).abs()
    if got.dtype == torch.bfloat16:
        err = err - _half_ulp_bf16(ref)
    worst = err.max().item() / scale if scale else (0.0 if err.max().item() <= 0 else math.inf)
    return None if worst <= GATE_REL_MAX else f"{name}: error {worst:.3g}"


def _row_by_row(d, out, d_a):
    """The kernel's d_row and d_col for cotangent d_a (all other cotangents zero), from one
    run per row of d_a. Values are passed as float32 so the recovered sums are not rounded
    to bf16 (bf16 values are exact in float32; the pulls are float32 arithmetic either way)."""
    N, C, W = d["k_idx"].shape
    R = d["q_idx"].shape[2]
    fp32 = dict(d, k_val=d["k_val"].float(), q_val=d["q_val"].float())
    zero = {k: torch.zeros(s, device="cuda") for k, s in (
        ("d_qk", (N, C, C)), ("d_k_eff", (N, C, W)), ("d_k_pd", (N, C, W)), ("d_q_w", (N, C, R)))}
    d_row = torch.zeros(N, C, W, dtype=F64, device="cuda")
    d_col = torch.zeros_like(d_row)
    tok = torch.arange(C, device="cuda").view(1, C, 1)
    for i0 in range(C):
        gk, _, _ = backward(fp32, out, {**zero, "d_a": d_a * (tok == i0)})
        d_row[:, i0] = gk[:, i0]
        d_col += torch.where(tok < i0, gk.double(), 0.0)
    return d_row, d_col


@pytest.mark.parametrize("case", BWD_GRID, ids=[c.id() for c in BWD_GRID])
def test_backward_matches_reference(case):
    d = make_inputs(case)
    out = forward(d)
    cot = make_cotangents(d)
    zero = {k: torch.zeros_like(t) for k, t in cot.items()}
    sets = {"all": cot, "a": {**zero, "d_a": cot["d_a"]}, "qk": {**zero, "d_qk": cot["d_qk"]}}
    refs = dict(zip(sets, reference_backward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"],
                                             d["g"], list(sets.values()), **d["kwargs"])))
    grads = {label: backward(d, out, c) for label, c in sets.items()}
    ref, (gk, gq, gg) = refs["all"], grads["all"]
    checks = [("grad_k_val", ref["grad_k_val"], gk), ("grad_q_val", ref["grad_q_val"], gq),
              ("grad_g", ref["grad_g"], gg)]
    ref, (gk, gq, gg) = refs["qk"], grads["qk"]
    checks += [("d_kvqk", ref["d_kvqk"], gk), ("d_qv", ref["d_qv"], gq),
               ("grad_g (d_qk only)", ref["grad_g"], gg)]
    ref, (gk, gq, gg) = refs["a"], grads["a"]
    d_row, d_col = _row_by_row(d, out, cot["d_a"])
    checks += [("d_row", ref["d_row"], d_row), ("d_col", ref["d_col"], d_col),
               ("d_row + d_col", ref["d_row"] + ref["d_col"], gk),
               ("grad_g (d_a only)", ref["grad_g"], gg)]
    bad = [m for m in (_mismatch(*c) for c in checks) if m]
    if grads["a"][1].count_nonzero():
        bad.append("grad_q_val is not zero with only d_a nonzero")
    rebuilt = backward(d, forward(d), cot)
    if not all(bits_equal(a, b) for a, b in zip(grads["all"], rebuilt)):
        bad.append("not bitwise identical after rebuilding the sorted order")
    assert not bad, f"{case.id()}: {bad}"


def test_reference_matches_token_loop():
    for case in (Case(W=3, R=2, C=8, sph=12), Case(W=4, R=3, C=8, sph=6, valid="C-1"),
                 Case(W=5, R=4, C=12, sph=20, gate="write"), Case(W=1, R=1, C=4, sph=3)):
        d = make_inputs(case)
        cot = make_cotangents(d)
        ref = reference_backward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"], d["g"], [cot],
                                 **d["kwargs"])[0]
        N = d["k_idx"].shape[0]
        off = d["kwargs"]["stream_offset"]
        for c in range(N):
            vt = case.valid_last if c >= N - case.P else case.C
            base = (c % case.P) * off
            loop = token_loop_backward(
                (d["k_idx"][c] - base).cpu(), (d["q_idx"][c] - base).cpu(), d["k_val"][c].cpu(),
                d["q_val"][c].cpu(), d["g"][c].cpu(),
                {k: t[c].double().cpu() for k, t in cot.items()}, vt)
            for f, want in loop.items():
                got = ref[f][c].cpu()
                assert torch.allclose(got, want, rtol=1e-10, atol=1e-12), (case.id(), c, f)


def test_invalid_tokens_get_exactly_zero_gradients():
    case = Case(W=16, R=8, C=48, valid="1", dtype="bfloat16", dist="sticky")
    d = make_inputs(case)
    grads = backward(d, forward(d), make_cotangents(d))
    invalid = ~token_valid(d)
    for name, t in zip(("grad_k_val", "grad_q_val", "grad_g"), grads):
        tail = t[invalid]
        assert bits_equal(tail, torch.zeros_like(tail)), name


def test_repeated_write_slots_get_no_gate_gradient():
    case = Case(W=8, R=4, C=16, sph=32, layout="single")
    d = make_inputs(case)
    k = d["k_idx"].clone()
    repeats = ((0, 3, 5, 2), (0, 7, 1, 6), (1, 0, 7, 0))
    for c, t, lane, src in repeats:
        k[c, t, lane] = k[c, t, src]
    d["k_idx"] = k
    cot = make_cotangents(d)
    gk, gq, gg = backward(d, forward(d), cot)
    ref = reference_backward(d["k_idx"], d["q_idx"], d["k_val"], d["q_val"], d["g"], [cot],
                             **d["kwargs"])[0]
    bad = [m for m in (_mismatch("grad_k_val", ref["grad_k_val"], gk),
                       _mismatch("grad_q_val", ref["grad_q_val"], gq),
                       _mismatch("grad_g", ref["grad_g"], gg)) if m]
    assert not bad, bad
    for c, t, lane, src in repeats:  # the later lane of the two repeats the slot
        assert gg[c, t, max(lane, src)].item() == 0.0


# Joint and split triangles, gradients assembled in shared memory or not, long groups that
# cross warp windows.
DETERMINISM_CASES = [
    Case(W=64, R=64, C=64, dist="zipf", dtype="bfloat16", steps=4),
    Case(W=48, R=127, C=127, dist="sticky", gate="write"),
    Case(W=128, R=128, C=256, dist="single_hot", sph=512, steps=1),
]


@pytest.mark.parametrize("case", DETERMINISM_CASES, ids=[c.id() for c in DETERMINISM_CASES])
def test_bitwise_deterministic(case):
    """Five backward runs, and backwards over three forward runs (three rebuilds of the
    sorted order), are bytewise identical."""
    d = make_inputs(case)
    cot = make_cotangents(d)
    out = forward(d)
    first = backward(d, out, cot)
    runs = [backward(d, out, cot) for _ in range(4)]
    runs += [backward(d, forward(d), cot) for _ in range(3)]
    for run in runs:
        for a, b in zip(first, run):
            assert bits_equal(a, b)


def test_graph_replay_matches_eager():
    case = Case(W=64, R=64, C=64, dist="zipf", dtype="bfloat16", steps=4)
    d = make_inputs(case)
    out = forward(d)
    cot = make_cotangents(d)
    eager = backward(d, out, cot)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backward(d, out, cot)
    graph.replay()
    for a, b in zip(eager, captured):
        assert bits_equal(a, b)


def _single_pair_grads(gates, pairs, W=16, R=16, sph=65536, seed=0):
    """One chunk of C = len(gates) tokens in which every token writes and reads one hot slot
    (lane 0) whose group straddles the first scan tile boundary. For each (i, j) only k_i,
    k_j and q_i are nonzero and d_a (d_qk) is one-hot at (i, j), so each gradient of the
    pair is a known value times the computed decay. Returns, per pair, the decays computed
    from both ends of the A pair and of the QK pair, the same-token QK decay (exactly 1),
    and the exact decay."""
    C = len(gates)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hot = sph // 2
    k_idx = _distinct_uniform(C, W, sph - 1, gen)
    q_idx = _distinct_uniform(C, R, sph - 1, gen)
    k_idx = (k_idx + (k_idx >= hot).long()).view(1, C, W)
    q_idx = (q_idx + (q_idx >= hot).long()).view(1, C, R)
    k_idx[..., 0] = hot
    q_idx[..., 0] = hot
    g = torch.tensor(gates, dtype=torch.float32, device="cuda").view(1, C, 1)
    exact_log = torch.cumsum(g.double().view(-1), 0)
    d = dict(k_idx=k_idx, q_idx=q_idx, g=g,
             kwargs=dict(num_parallel=1, stream_offset=0, num_keys=sph, valid_last=C))
    ki, kj, qi = 1.5, -0.75, 0.5
    zero = lambda *s: torch.zeros(1, *s, device="cuda")
    out = []
    for i, j in pairs:
        d["k_val"], d["q_val"] = zero(C, W), zero(C, R)
        d["k_val"][0, i, 0], d["k_val"][0, j, 0], d["q_val"][0, i, 0] = ki, kj, qi
        fwd = forward(d)
        base = dict(d_a=zero(C, C), d_qk=zero(C, C), d_k_eff=zero(C, W), d_k_pd=zero(C, W),
                    d_q_w=zero(C, R))
        a = dict(base, d_a=base["d_a"].index_put((torch.tensor([0]), torch.tensor([i]),
                                                  torch.tensor([j])), torch.tensor(1.0)))
        qk = dict(base, d_qk=base["d_qk"].index_put((torch.tensor([0]), torch.tensor([i]),
                                                     torch.tensor([j])), torch.tensor(1.0)))
        own = dict(base, d_qk=base["d_qk"].index_put((torch.tensor([0]), torch.tensor([i]),
                                                      torch.tensor([i])), torch.tensor(1.0)))
        gk_a, _, _ = backward(d, fwd, a)
        gk_q, gq_q, _ = backward(d, fwd, qk)
        _, gq_own, _ = backward(d, fwd, own)
        out.append((gk_a[0, i, 0].item() / kj, gk_a[0, j, 0].item() / ki,
                    gq_q[0, i, 0].item() / kj, gk_q[0, j, 0].item() / qi,
                    gq_own[0, i, 0].item() / ki,
                    torch.exp(exact_log[i] - exact_log[j]).item()))
    return out


def test_decays_after_a_large_prefix_are_ulp_accurate():
    """The forward's adversarial prefix (see test_phase1): a -4100 log prefix, then
    increments near, far below and exactly at zero relative to its ulp. Every decay the
    backward applies is within 4 ulps of exp(sum of gates) and at most 1, and a Q entry's
    decay to its own token's K entry is exactly 1."""
    steps = (-1e-3, -3e-7, 0.0, -2.5e-4, -1e-9)
    gates = [-41.0] * 100 + [steps[t % len(steps)] for t in range(156)]
    pairs = [(101, 100), (102, 101), (103, 102), (104, 103), (105, 104), (140, 120), (255, 100),
             (100, 99), (120, 3)]
    for (i, j), (*decays, own, exact) in zip(pairs, _single_pair_grads(gates, pairs)):
        for got in decays:
            assert abs(got - exact) <= 4 * _ulp32(exact), ((i, j), got, exact)
            assert got <= 1.0, ((i, j), got)
        assert own == 1.0, ((i, j), own)


@pytest.mark.slow
def test_more_than_2_31_sorted_entries():
    """N * n > 2^31 entries: every row offset is 64-bit. Sampled chunks, including both
    sides of the 2^31 boundary, match the reference."""
    require_free_memory(160)
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
    d = dict(k_idx=k_idx, q_idx=q_idx,
             k_val=torch.randn(N, C, W, device="cuda", generator=gen, dtype=torch.bfloat16),
             q_val=torch.randn(N, C, W, device="cuda", generator=gen, dtype=torch.bfloat16),
             g=-torch.rand(N, C, 1, device="cuda", generator=gen) * 0.2,
             kwargs=dict(num_parallel=P, stream_offset=sph, num_keys=sph, valid_last=C - 3))
    out = forward(d)
    cot = make_cotangents(d)
    grads = backward(d, out, cot)
    del out
    boundary = 2 ** 31 // n
    for c in (0, boundary - 1, boundary, N - P - 1, N - 1):
        c0 = c - c % P
        sl = slice(c0, c0 + P)
        kw = dict(d["kwargs"], valid_last=d["kwargs"]["valid_last"] if c0 == N - P else C)
        ref = reference_backward(k_idx[sl], q_idx[sl], d["k_val"][sl], d["q_val"][sl],
                                 d["g"][sl], [{k: t[sl] for k, t in cot.items()}], **kw)[0]
        for name, got in zip(("grad_k_val", "grad_q_val", "grad_g"), grads):
            assert not _mismatch(name, ref[name], got[sl]), (c, name)
