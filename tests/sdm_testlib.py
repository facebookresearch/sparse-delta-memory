# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared helpers for the SDM tests: input builders, a float64 reference and noise gates.

The reference (``oracle_write_read``) is an unchunked token loop in float64 whose gradients
come from autograd. It shares no kernel, chunking or hand-written adjoint with the code
under test. Streams own disjoint slot ranges, so the loop advances all streams together.

The SDM kernels are deterministic for fixed inputs, configuration and GPU type: every bank
row has one writer per launch and every sum runs in a fixed order (test_determinism.py). So
repeated runs, CUDA-graph replays, the layer's 2-D and 3-D inputs (reshaped before any
compute) and recomputes are compared byte for byte (``assert_runs_bitwise``). Two
implementations of the same math that may round differently (two chunk sizes, the op's
[P * T] layout against [P, T], the no-grad path against the saving forward) are compared
with fixed tolerances measured on the deterministic kernels (``assert_close_across``); a
run-to-run spread is zero and cannot serve as their yardstick.

Also here: the numerically defined gates against the reference (``fp32_gate_failures``,
``bf16_gate_failures``, ``quant_gate_failures``), a torch reference of the snapshot
quantizer, and the profiler helpers behind the kernel-name allowlist checks.
"""
import contextlib
import hashlib
import re

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def require_free_memory(gib):
    """Skips the calling test unless the GPU has ``gib`` GiB free."""
    if torch.cuda.mem_get_info()[0] < gib * 2**30:
        pytest.skip(f"needs {gib} GiB of free GPU memory")

OP_INPUTS = ("memory", "k_val", "v", "beta", "g", "q_val")
OP_GRADS = tuple("grad_" + n for n in OP_INPUTS)


# ----------------------------------------------------------------------------- metrics
def rel_max(ref, got):
    """max|ref - got| / max|ref|, with non-finite or mis-shaped ``got`` scoring inf."""
    a, b = ref.double().reshape(-1), got.double().reshape(-1)
    if a.numel() != b.numel() or not bool(torch.isfinite(b).all()):
        return float("inf")
    den = a.abs().max().item()
    if den == 0.0:
        return 0.0 if b.abs().max().item() == 0.0 else float("inf")
    return (a - b).abs().max().item() / den


def rel_rms(ref, got):
    """rms(ref - got) / rms(ref)."""
    a, b = ref.double().reshape(-1), got.double().reshape(-1)
    if a.numel() != b.numel() or not bool(torch.isfinite(b).all()):
        return float("inf")
    den = a.pow(2).mean().sqrt().item()
    if den == 0.0:
        return 0.0 if b.abs().max().item() == 0.0 else float("inf")
    return (a - b).pow(2).mean().sqrt().item() / den


def _bytes(t):
    """The bytes of ``t`` as a 1-D uint8 tensor (0-d tensors included)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def bits_equal(a, b):
    """Bytewise equality (distinguishes -0.0 from +0.0 and compares NaN payloads)."""
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bytes(a), _bytes(b)))


_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits_mismatch(ref, got):
    """'' when ``got`` is byte-equal to ``ref``, else a short description: the fraction of
    elements whose bytes differ and the relative RMS of the difference."""
    if ref.dtype != got.dtype or ref.shape != got.shape:
        return f"{got.dtype} {tuple(got.shape)} vs {ref.dtype} {tuple(ref.shape)}"
    if bits_equal(ref, got):
        return ""
    view = _INT_VIEW[ref.element_size()]
    a = ref.detach().contiguous().reshape(-1).view(view)
    b = got.detach().contiguous().reshape(-1).view(view)
    frac = (a != b).double().mean().item()
    return f"{frac:.2%} of elements differ, rel_rms {rel_rms(ref.float(), got.float()):.2e}"


def tensor_digest(t):
    """sha256 of a tensor's dtype, shape and bytes."""
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(_bytes(t).cpu().numpy().tobytes())
    return h.hexdigest()


def assert_runs_bitwise(runs, *, label="", ref=None):
    """Every run ({name: tensor}) equals ``ref`` (default: the first run) byte for byte, in
    every tensor. The message names each differing tensor, the runs that differ and how
    much (the evidence of a nondeterministic accumulation)."""
    ref = runs[0] if ref is None else ref
    bad = {}
    for i, run in enumerate(runs):
        if run.keys() != ref.keys():
            bad[f"run {i}"] = f"tensors {sorted(run)} vs {sorted(ref)}"
            continue
        for k, t in run.items():
            if not bits_equal(ref[k], t):
                bad.setdefault(k, []).append(i)
    if bad:
        msg = {k: v if isinstance(v, str) else
               f"runs {v} of {len(runs)}: {bits_mismatch(ref[k], runs[v[0]][k])}"
               for k, v in bad.items()}
        raise AssertionError(f"{label}: not bitwise reproducible: {msg}")


def assert_close_across(runs_a, runs_b, *, rms, tiny_max, tiny_abs=None, label="",
                        overrides=None):
    """Two implementations of the same math that may round differently (two chunk sizes,
    TF32 matmuls at other addresses): each is bitwise reproducible over its own runs, and
    per tensor the two differ by at most ``rms`` in relative RMS, or ``tiny_max`` in
    relative max for tensors of at most TINY_NUMEL elements (sums with heavy cancellation).
    A tiny tensor also passes when its largest absolute difference is at most ``tiny_abs``
    (its sum may cancel to a small value, which a relative bound would magnify).
    ``overrides`` maps tensor names to their own (rms or max) bound. The bounds are fixed
    numbers measured on the deterministic kernels; each call prints its largest deviations
    relative to their bounds (pytest -rP shows them) to keep that calibration visible."""
    a0, b0 = runs_a[0], runs_b[0]
    assert a0.keys() == b0.keys(), f"{label}: tensors {sorted(a0)} vs {sorted(b0)}"
    schema = {k: (a0[k].dtype, tuple(a0[k].shape), b0[k].dtype, tuple(b0[k].shape)) for k in a0
              if a0[k].dtype != b0[k].dtype or a0[k].shape != b0[k].shape}
    assert not schema, f"{label}: dtype/shape differ: {schema}"
    devs, bad = {}, {}
    for k, ref in a0.items():
        metric, bound = (rel_rms, rms) if ref.numel() > TINY_NUMEL else (rel_max, tiny_max)
        bound = (overrides or {}).get(k, bound)
        dev = metric(ref, runs_b[0][k])
        devs[k] = (dev / bound if bound else float("inf"), metric.__name__, dev, bound)
        if (not dev <= bound and tiny_abs is not None and ref.numel() <= TINY_NUMEL
                and bool(torch.isfinite(b0[k]).all())
                and (ref.double() - b0[k].double()).abs().max().item() <= tiny_abs):
            continue
        if not dev <= bound:
            bad[k] = f"{metric.__name__} {dev:.3g} > {bound:g}"
    worst = sorted(devs.items(), key=lambda kv: -kv[1][0])[:4]
    print(f"[cross] {label}: " + "; ".join(f"{k} {m} {d:.3g} (bound {b:g})"
                                         for k, (_, m, d, b) in worst))
    assert_runs_bitwise(runs_a, label=f"{label} [first]")
    assert_runs_bitwise(runs_b, label=f"{label} [second]")
    assert not bad, f"{label}: beyond the cross-implementation bound: {bad}"


# ------------------------------------------------------------------------ op-level inputs
def _unique_rows(rows, width, n, gen, weights=None):
    """``rows`` rows of ``width`` distinct sorted ids in [0, n)."""
    if weights is None:
        keys = torch.rand(rows, n, generator=gen)
        return keys.topk(width, dim=-1).indices.sort(dim=-1).values
    probs = weights.expand(rows, n)
    return torch.multinomial(probs, width, replacement=False, generator=gen).sort(dim=-1).values


def _hot_rows(rows, width, n, gen, hot):
    """Rows of ``width`` distinct sorted ids in [0, n) that all contain the ids ``hot``."""
    keys = torch.rand(rows, n, generator=gen)
    keys[:, hot] = 2.0
    return keys.topk(width, dim=-1).indices.sort(dim=-1).values


def _zipf_rows(rows, width, log_w, gen):
    """Rows of distinct sorted ids drawn without replacement with weights exp(log_w)
    (Gumbel top-k, as benchmarks/bench_sdm.py ``synthetic_indices``)."""
    u = torch.rand(rows, log_w.numel(), generator=gen).clamp_min_(1e-20)
    keys = log_w - torch.log(-torch.log(u))
    return keys.topk(width, dim=-1).indices.sort(dim=-1).values


HOT_SLOTS = 4
# Popularity exponent of the "trained" slot law: 1 / (1 + rank)**0.65 at W = 16 over 4096
# slots puts about 40% of a chunk's write entries in repeated groups at C = 64 and 70% at
# C = 256, close to the selections of trained models.
TRAINED_EXPONENT = 0.65


def _collision_slots(slots, P, T, W, R, sph, gen):
    """(k_idx, q_idx) local ids [P, T, W] / [P, T, R] of the slot distributions of
    ``make_op_inputs`` other than "uniform" and "clustered"."""
    ks, qs = [], []
    for _ in range(P):
        if slots in ("hot", "repeat"):
            hot = torch.randperm(sph, generator=gen)[:HOT_SLOTS]
            k, q = _hot_rows(T, W, sph, gen, hot), _hot_rows(T, R, sph, gen, hot)
            if slots == "repeat":  # every third token repeats a write and a read slot
                k[::3, 1], q[::3, 1] = k[::3, 0], q[::3, 0]
        elif slots in ("zipf", "trained"):
            a = 1.0 if slots == "zipf" else TRAINED_EXPONENT
            log_w = -a * torch.log1p(torch.randperm(sph, generator=gen).float())
            k, q = _zipf_rows(T, W, log_w, gen), _zipf_rows(T, R, log_w, gen)
        elif slots == "distinct":
            if T * (W + R) > sph:
                raise ValueError(f"distinct slots need T * (W + R) <= sph, got sph={sph}")
            perm = torch.randperm(sph, generator=gen)
            k = perm[:T * W].view(T, W).sort(-1).values
            q = perm[T * W:T * (W + R)].view(T, R).sort(-1).values
        else:
            raise ValueError(slots)
        ks.append(k)
        qs.append(q)
    return torch.stack(ks), torch.stack(qs)


def make_op_inputs(P, T, W, R, dim, sph, dtype, *, seed=0, device="cuda",
                   slots="uniform", gates="moderate", betas="moderate", weights="randn"):
    """Inputs of ``sdm_write_read`` in the [P, T, *] stream layout.

    Indices are global (stream p owns [p * sph, (p + 1) * sph)), distinct and ascending
    within a token, like the layer's product-key selection ("repeat" excepted).

    slots:   "uniform"    independent uniform draws per token
             "clustered"  32 hot slots per stream carry 96% of the sampling mass, so every
                          token rewrites almost the same set (heavy cross-token collisions)
             "hot"        HOT_SLOTS slots per stream are in the write and the read set of
                          every token: each chunk has groups of 2 C entries on one slot
             "zipf"       popularity proportional to 1 / (1 + rank) in a random slot order,
                          shared by writes and reads (benchmarks/bench_sdm.py's "zipf")
             "trained"    the same law with exponent TRAINED_EXPONENT: the repeated-group
                          statistics of trained models' selections (at W = 16, 4096 slots)
             "repeat"     "hot", and every third token repeats its first write slot and its
                          first read slot (a write repeat violates the op's contract; the
                          op decays it once and counts it)
             "distinct"   no slot occurs twice in a stream (writes and reads included):
                          nothing to accumulate (needs T * (W + R) <= sph)
    gates:   "moderate"   log-decay g ~ U(-0.5, -0.01)
             "extreme"    per token either ~0 (-1e-4) or very negative (down to -20)
             "flat"       g ~ U(-1e-3, -1e-5): almost no decay, so every earlier write of a
                          slot still contributes at full weight
    betas:   "moderate"   U(0, 0.3);  "extreme"  either ~0 (1e-4) or ~1 (0.9999)
    weights: "randn"      k_val, q_val ~ 0.3 * N(0, 1)
             "softmax"    softmax-normalised rows, as the layer produces
    Values are generated in float32 and cast to ``dtype``; references must upcast the cast
    values so input quantisation is not charged to the kernel.
    """
    gen = torch.Generator().manual_seed(seed)
    if slots in ("uniform", "clustered"):
        wsl = None
        if slots == "clustered":
            hot = torch.randperm(sph, generator=gen)[:32]
            wsl = torch.full((sph,), 0.04 / (sph - 32))
            wsl[hot] = 0.96 / 32
        k_idx = torch.stack([_unique_rows(T, W, sph, gen, wsl) + p * sph for p in range(P)])
        q_idx = torch.stack([_unique_rows(T, R, sph, gen, wsl) + p * sph for p in range(P)])
    else:
        k_idx, q_idx = _collision_slots(slots, P, T, W, R, sph, gen)
        off = torch.arange(P).view(P, 1, 1) * sph
        k_idx, q_idx = k_idx + off, q_idx + off

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen) * scale

    if weights == "randn":
        k_val, q_val = randn(P, T, W, scale=0.3), randn(P, T, R, scale=0.3)
    elif weights == "softmax":
        k_val = torch.softmax(randn(P, T, W, scale=2.0), dim=-1)
        q_val = torch.softmax(randn(P, T, R, scale=2.0), dim=-1)
    else:
        raise ValueError(weights)

    if gates == "moderate":
        g = torch.empty(P, T, 1).uniform_(-0.5, -0.01, generator=gen)
    elif gates == "extreme":
        pick = torch.rand(P, T, 1, generator=gen) < 0.5
        deep = -torch.empty(P, T, 1).uniform_(5.0, 20.0, generator=gen)
        g = torch.where(pick, torch.full_like(deep, -1e-4), deep)
    elif gates == "flat":
        g = torch.empty(P, T, 1).uniform_(-1e-3, -1e-5, generator=gen)
    else:
        raise ValueError(gates)

    if betas == "moderate":
        beta = torch.rand(P, T, 1, generator=gen) * 0.3
    elif betas == "extreme":
        pick = torch.rand(P, T, 1, generator=gen) < 0.5
        beta = torch.where(pick, torch.full_like(g, 1e-4), torch.full_like(g, 0.9999))
    else:
        raise ValueError(betas)

    def cast(t):
        return t.to(device=device, dtype=dtype)

    return dict(
        memory=cast(randn(P * sph, dim, scale=0.1)),
        k_idx=k_idx.to(device), q_idx=q_idx.to(device),
        k_val=cast(k_val), v=cast(randn(P, T, dim, scale=0.1)), beta=cast(beta),
        g=cast(g), q_val=cast(q_val),
    )


def streams_own_their_slots(idx, sph):
    """Whether stream p of the [P, T, W] ids ``idx`` only selects rows [p * sph, (p + 1) *
    sph): the condition under which several streams may share one bank (shared_bank=True)."""
    P = idx.shape[0]
    stream = torch.arange(P, device=idx.device).view(P, *([1] * (idx.dim() - 1)))
    return bool((torch.div(idx, sph, rounding_mode="floor") == stream).all())


def slot_groups(k_idx, q_idx, C, two_d=False):
    """Collision statistics of the chunks of [S, T, W] / [S, T, R] slot ids (one stream of
    S * T tokens when ``two_d``), counting a group as the entries of one chunk on one slot:
    the fraction of write entries in groups of two or more, the mean size of those groups,
    and the longest write, read and combined (write + read) group."""
    k, q = k_idx.cpu(), q_idx.cpu()
    if two_d:
        k, q = k.reshape(1, -1, k.shape[-1]), q.reshape(1, -1, q.shape[-1])
    rep = total = 0
    sizes, longest = [], [0, 0, 0]
    for s in range(k.shape[0]):
        for t0 in range(0, k.shape[1], C):
            kc, qc = k[s, t0:t0 + C].reshape(-1), q[s, t0:t0 + C].reshape(-1)
            counts = [torch.unique(ids, return_counts=True)[1]
                      for ids in (kc, qc, torch.cat([kc, qc]))]
            for i, c in enumerate(counts):
                longest[i] = max(longest[i], int(c.max()))
            many = counts[0][counts[0] > 1]
            rep += int(many.sum())
            total += kc.numel()
            sizes += many.tolist()
    return dict(repeated=rep / total, mean_group=sum(sizes) / max(1, len(sizes)),
                longest=tuple(longest))


def make_cotangent(readings, mode="dense", seed=17):
    """dL/d(readings) in ``readings``' shape with a deliberate memory layout.

    dense        contiguous random values
    dense2       contiguous, different values (a second projection of the Jacobian)
    expanded     all strides zero, what ``readings.sum().backward()`` produces
    noncontig    the ``dense`` values viewed through a stride-2 slice
    """
    shape, dev, dt = readings.shape, readings.device, readings.dtype
    if mode == "expanded":
        out = torch.ones((), device=dev, dtype=dt).expand(shape)
        assert all(s == 0 for s in out.stride())
        return out
    if mode == "noncontig":
        dense = make_cotangent(readings, "dense", seed)
        big = torch.zeros(*shape[:-1], 2 * shape[-1], device=dev, dtype=dt)
        big[..., ::2] = dense
        out = big[..., ::2]
        assert not out.is_contiguous() and torch.equal(out, dense)
        return out
    gen = torch.Generator().manual_seed(seed if mode == "dense" else seed + 6)
    return torch.randn(*shape, generator=gen).to(device=dev, dtype=dt)


# ------------------------------------------------------------------------ the reference
def oracle_write_read(d, cotangent=None, *, device=None, dtype=torch.float64, sph=None):
    """Unchunked token-loop reference of the SDM write/read.

    Per token t and stream s:
        at_k      = mem[k_idx]
        read      = at_k * exp(g)                   decay applies only to written slots
        retrieved = sum_w k_val_w * read_w
        dv        = beta * (v - retrieved)
        mem      += scatter(read - at_k + k_val (x) dv)
        out       = sum_r q_val_r * mem[q_idx_r]    read after the write

    ``d`` holds [S, T, *] streams (2-D [S*T, *] inputs must be reshaped by the caller).
    Returns (readings [S, T, dim], grads or None, final memory).
    """
    dev = device or d["memory"].device
    need_grad = cotangent is not None
    leaves = {n: d[n].detach().to(dev, dtype).clone().requires_grad_(need_grad)
              for n in OP_INPUTS}
    k_idx, q_idx = d["k_idx"].to(dev), d["q_idx"].to(dev)
    S = k_idx.shape[0]
    if sph is not None:
        owner = torch.arange(S, device=dev).view(S, 1, 1)
        assert bool((k_idx // sph == owner).all()) and bool((q_idx // sph == owner).all()), \
            "streams must own disjoint slot ranges"
    srt = k_idx.sort(dim=-1).values
    assert bool((srt[..., 1:] != srt[..., :-1]).all()), "write slots must be distinct per token"

    readings, mem = oracle_recurrence(
        leaves["memory"], k_idx, leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
        q_idx, leaves["q_val"])
    if not need_grad:
        return readings.detach(), None, mem.detach()
    grads = torch.autograd.grad(
        readings, [leaves[n] for n in OP_INPUTS],
        grad_outputs=cotangent.to(dev, dtype).reshape(readings.shape))
    return readings.detach(), dict(zip(OP_GRADS, grads)), mem.detach()


def oracle_recurrence(memory, k_idx, k_val, v, beta, g, q_idx, q_val):
    """Differentiable token loop of ``oracle_write_read`` on [S, T, *] streams.

    Returns (readings [S, T, dim], final memory); gradients flow to every float input.
    """
    T = k_idx.shape[1]
    dim = v.shape[-1]
    # Updated in place: neither the gathers nor index_add_ save the bank for backward,
    # so the tape holds O(T * (W + R) * dim) values instead of a bank copy per token.
    mem = memory.clone()
    outs = []
    for t in range(T):
        ki = k_idx[:, t]
        at_k = mem[ki]                                              # [S, W, dim]
        read = at_k * torch.exp(g[:, t]).unsqueeze(-1)              # g: [S, 1] or [S, W]
        retrieved = (k_val[:, t].unsqueeze(-1) * read).sum(1)       # [S, dim]
        dv = beta[:, t] * (v[:, t] - retrieved)
        upd = read - at_k + k_val[:, t].unsqueeze(-1) * dv.unsqueeze(1)
        mem.index_add_(0, ki.reshape(-1), upd.reshape(-1, dim))
        outs.append((q_val[:, t].unsqueeze(-1) * mem[q_idx[:, t]]).sum(1))
    return torch.stack(outs, 1), mem


def kernel_write_read(d, *, chunk_size, sph, cotangent="dense", layout="3d"):
    """Training op forward + backward on private copies of ``d``.

    layout "3d": [P, T, *] inputs with num_parallel = P.
    layout "2d": [P*T, *] inputs with num_parallel = 1 (one sequential stream order over a
                 bank that still holds P slot ranges).
    Returns (readings in the input layout, grads, cotangent used).
    """
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T = d["k_idx"].shape[:2]
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}

    def lay(t):
        return t if layout == "3d" else t.reshape(P * T, -1)

    readings = sdm_write_read(
        leaves["memory"], lay(d["k_idx"]), lay(leaves["k_val"]), lay(leaves["v"]),
        lay(leaves["beta"]), lay(leaves["g"]), lay(d["q_idx"]), lay(leaves["q_val"]),
        chunk_size=chunk_size, num_memory_slots=sph,
        num_parallel=P if layout == "3d" else 1,
    )
    if cotangent is None:
        return readings.detach(), None, None
    G = make_cotangent(readings, cotangent) if isinstance(cotangent, str) else cotangent
    grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS], grad_outputs=G)
    return readings.detach(), dict(zip(OP_GRADS, grads)), G.detach()


# ------------------------------------------------------------------------- layer helpers
def make_layer(seed=0, dtype=torch.bfloat16, device="cuda", **overrides):
    from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

    cfg = dict(dim=256, num_heads=2, slots_per_head=1024, num_writes=16, num_reads=16,
               chunk_size=64, normalize_readings=True)
    cfg.update(overrides)
    torch.manual_seed(seed)
    layer = SparseDeltaMemory(SparseDeltaMemoryArgs(**cfg), layer_id=0).to(device)
    layer.init_weights()
    return layer.to(dtype)


def layer_inputs(B, T, dim, dtype=torch.bfloat16, seed=1, scale=1.0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, T, dim, device=device, dtype=torch.float32, generator=gen) * scale
    gy = torch.randn(B, T, dim, device=device, dtype=torch.float32, generator=gen) * 0.01
    return x.to(dtype), gy.to(dtype)


def layer_fwd_bwd(layer, x, gy):
    """Output, input grad and every parameter grad of one forward + backward."""
    for p in layer.parameters():
        p.grad = None
    xi = x.detach().clone().requires_grad_(True)
    out, _ = layer(xi)
    out.backward(gy)
    res = {"out": out.detach().clone(), "grad_x": xi.grad.detach().clone()}
    for n, p in layer.named_parameters():
        res["grad." + n] = p.grad.detach().clone()
    for p in layer.parameters():
        p.grad = None
    return res


class _FixedSlotScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, lin, factors):
        cols, rem = [], lin
        for f in reversed(factors):
            cols.append(rem % f)
            rem = rem // f
        score, off = None, 0
        for c, f in zip(reversed(cols), factors):
            v = torch.gather(raw, -1, off + c).float()
            score = v if score is None else score + v
            off += f
        ctx.save_for_backward(lin)
        ctx.factors, ctx.raw_shape = factors, raw.shape
        return score.to(raw.dtype)

    @staticmethod
    def backward(ctx, grad):
        from sparse_delta_memory.cuda.product_key_topk_cuda import _module

        (lin,) = ctx.saved_tensors
        rows = lin.reshape(-1, lin.shape[-1]).contiguous()
        g = _module().topk_bwd(rows, grad.reshape(rows.shape).contiguous(), ctx.factors)
        return g.view(ctx.raw_shape), None, None


def fixed_slot_scores(raw, lin, factors):
    """Scores of the fixed product-key slots ``lin`` ([..., k] int64, ascending, as
    ``product_key_topk`` returns them) from the factor columns of ``raw``, differentiable
    with product_key_topk's deterministic backward. (A gather's backward scatter-adds the
    factor columns that several selected slots share, with atomics.)"""
    return _FixedSlotScores.apply(raw, lin, tuple(factors))


class SelectionSpy:
    """Records (raw scores, selected slots) of every product-key call of a layer: per
    forward, the writes, then the reads."""

    def __init__(self, layer):
        self.calls, self._orig = [], layer._product_keys
        layer._product_keys = self._spy

    def _spy(self, x, k):
        val, idx = self._orig(x, k)
        self.calls.append((x.detach().clone(), idx.detach().clone()))
        return val, idx

    def take(self):
        out, self.calls = self.calls, []
        return out


class ProjectionReplay:
    """cuBLAS's row-count dependence, removed from a decode-vs-full-forward comparison.
    Records every nn.Linear call of a layer's full forward (input and output rows); during the
    token-by-token decode that follows, a decode call whose input row is bit-identical to the recorded row t but whose
    output row is not (the library chose a different kernel for one row than for T rows: A100's cuBLAS does, GB300's
    observed not to) returns the recorded output row instead. The decode then consumes exactly the full forward's
    projections wherever their inputs agree, and every remaining difference is the layer's own (memory rounding, its
    decode step); calls whose inputs differ (the output projection of the readings) are untouched. ``replaced`` counts
    the substituted output rows: 0 on a GPU whose cuBLAS is row-count invariant, where the comparison is the plain
    one.
    Usage: rep = ProjectionReplay(layer); with rep.record(): full forward; with rep.decode(): token loop; rep.close()."""

    def __init__(self, layer):
        self.mods = [(n, m) for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)]
        self.rec, self.mode, self.step, self.replaced, self.calls = {}, None, {}, 0, {}
        self.hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: self._hook(n, inp, out))
                      for n, m in self.mods]

    def _hook(self, name, inp, out):
        x = inp[0]
        if self.mode == "record":
            assert name not in self.rec, f"{name}: called twice in the full forward"
            self.rec[name] = (x.detach().clone(), out.detach().clone())
            return None
        if self.mode != "decode" or name not in self.rec:
            return None
        t = self.step.get(name, 0)
        self.step[name] = t + 1
        xf, yf = self.rec[name]
        if x.shape[:-2] != xf.shape[:-2] or x.shape[-2] != 1 or t >= xf.shape[-2]:
            return None
        xr, yr = xf[..., t:t + 1, :], yf[..., t:t + 1, :]
        if x.dtype != xr.dtype or out.shape != yr.shape or not bits_equal(x, xr) or bits_equal(out, yr):
            return None
        self.replaced += 1
        return yr.clone()

    @contextlib.contextmanager
    def record(self):
        self.mode = "record"
        try:
            yield self
        finally:
            self.mode = None

    @contextlib.contextmanager
    def decode(self):
        self.mode, self.step = "decode", {}
        try:
            yield self
        finally:
            self.mode = None

    def close(self):
        for h in self.hooks:
            h.remove()


# Tensors this small (per-head parameters) are gated with a normwise statistic instead of a
# relative RMS; see assert_close_across.
TINY_NUMEL = 16


# ------------------------------------------------------------ numerically defined gates
GATE_FP32 = 1e-5              # rel_max per tensor; fp32 banks, IEEE matmuls
# bf16 banks: (rel_rms, rel_max) ceilings per tensor. Deviations come from rounding the bank,
# the readings and the gradients to bf16 (rel_rms up to about 2e-2 over the test grid).
BF16_CEILINGS = (4e-2, 1e-1)
# Where written slots take many writes per chunk (24 at W = 256 over 512 slots, C = 48), a
# bank rounded to bf16 after every write deviates by about 0.12, so cases above
# BF16_COLLISION_WRITES mean writes per written slot in a chunk gate the final bank at
# BF16_BANK_COLLISION_CEILINGS. The kernels sum a slot's writes in float32 and round once per
# chunk, so these are upper bounds.
BF16_COLLISION_WRITES = 8.0
BF16_BANK_COLLISION_CEILINGS = (0.2, 0.2)
BF16_PRECISION_RATIO = 50.0   # an fp32 bank on the same inputs is at least this much closer
BF16_PRECISION_FLOOR = 1e-5   # below this a bf16 deviation is already at fp32 level
# Quantized snapshot, rel_rms of the gradients that read it (QUANT_GRADS). fp8 (e4m3) keeps
# 3 mantissa bits per element. int4 keeps 16 levels per row: its per-element error is about
# 14% of the rms of a Gaussian row, and the gate gradients, which sum products of
# snapshot rows with other rows, reach 3x that (0.43 at worst over the test grid).
# A bf16 bank adds its own rounding, so bf16 banks are gated at QUANT_BOUNDS + BF16_CEILINGS[0].
# bf16 (float32 banks; a bf16 bank runs it as "none") keeps 8 significand bits per element (16x fp8's precision):
# its bound is fp8's / 6.
QUANT_BOUNDS = {"fp8": 6e-2, "int4": 5e-1, "bf16": 1e-2}
QUANT_SIGNAL_RATIO = 10.0     # quantized deviation / unquantized one, fp32 banks
QUANT_GRADS = ("grad_k_val", "grad_g", "grad_q_val")


def deviations(ref, got):
    """{name: {"rel_max", "rel_rms"}} of ``got`` against ``ref`` over ref's keys."""
    return {k: {"rel_max": rel_max(ref[k], got[k]), "rel_rms": rel_rms(ref[k], got[k])}
            for k in ref}


def fp32_gate_failures(dev, names=None):
    """Tensors of ``dev`` (from ``deviations``) whose rel_max exceeds GATE_FP32."""
    return {k: f"rel_max {d['rel_max']:.3g} > {GATE_FP32:g}" for k, d in dev.items()
            if (names is None or k in names) and not d["rel_max"] <= GATE_FP32}


def bf16_gate_failures(dev, dev_fp32, names=None, ceilings=None):
    """The bf16 gate, per tensor of ``dev`` (a bf16 bank against the reference):

    1. rel_rms and rel_max within BF16_CEILINGS, or ``ceilings[name]`` when given;
    2. ``dev_fp32`` (an fp32 bank on the same bf16-valued inputs) is at least
       BF16_PRECISION_RATIO times closer in rel_rms, unless the bf16 deviation is
       already below BF16_PRECISION_FLOOR.
    """
    bad = {}
    for k, d in dev.items():
        if names is not None and k not in names:
            continue
        why = []
        rms_ceiling, max_ceiling = (ceilings or {}).get(k, BF16_CEILINGS)
        if not d["rel_rms"] <= rms_ceiling:
            why.append(f"rel_rms {d['rel_rms']:.3g} > {rms_ceiling:g}")
        if not d["rel_max"] <= max_ceiling:
            why.append(f"rel_max {d['rel_max']:.3g} > {max_ceiling:g}")
        r32 = dev_fp32[k]["rel_rms"]
        if not (d["rel_rms"] <= BF16_PRECISION_FLOOR
                or r32 * BF16_PRECISION_RATIO <= d["rel_rms"]):
            why.append(f"fp32 rel_rms {r32:.3g} is not {BF16_PRECISION_RATIO:g}x below "
                       f"{d['rel_rms']:.3g}")
        if why:
            bad[k] = "; ".join(why)
    return bad


def quant_gate_failures(dev, dev_none, mode, fp32_bank):
    """The quantized-snapshot gate on QUANT_GRADS: each rel_rms <= QUANT_BOUNDS[mode], plus
    BF16_CEILINGS[0] for bf16 banks. For fp32 banks, at least one of them must also be
    QUANT_SIGNAL_RATIO times the unquantized run's (``dev_none``), which shows the quantized
    snapshot is the one used; a single gradient can be insensitive when no read hits a row
    restored from the snapshot. Other tensors are gated by the bank's own gate."""
    bound = QUANT_BOUNDS[mode] + (0.0 if fp32_bank else BF16_CEILINGS[0])
    bad = {k: f"rel_rms {dev[k]['rel_rms']:.3g} > {bound:g}" for k in QUANT_GRADS
           if not dev[k]["rel_rms"] <= bound}
    if fp32_bank and not any(dev[k]["rel_rms"] >= QUANT_SIGNAL_RATIO * dev_none[k]["rel_rms"]
                             for k in QUANT_GRADS):
        bad["quantized snapshot"] = (f"no gradient of {QUANT_GRADS} deviates "
                                     f"{QUANT_SIGNAL_RATIO:g}x more than unquantized")
    return bad


# ----------------------------------------------------------------- snapshot quantizer
def reference_quantize(rows, mode):
    """Torch reference of the snapshot quantizer (per-row amax scaling).

    scale = bfloat16(max(amax / 448 (fp8) or amax / 7 (int4), 1e-12)) stored in float32,
    with NaN ignored by the amax. Values are divided by the scale and rounded to bfloat16,
    then cast: fp8 to float8_e4m3fn (NaN -> 0x7f); int4 to round-half-even clamped to
    [-8, 7] (NaN -> 7), element 2i in the high nibble of byte i and a missing odd last
    element encoded as 0. Returns (payload, scale [..., 1]).
    """
    x = rows.float()
    amax = torch.where(x.isnan(), 0.0, x.abs()).amax(-1, keepdim=True)
    scale = torch.clamp(amax / (448.0 if mode == "fp8" else 7.0), min=1e-12)
    scale = scale.to(torch.bfloat16).float()
    q = (x / scale).to(torch.bfloat16).float()
    if mode == "fp8":
        return torch.where(q.isnan(), float("nan"), q).to(torch.float8_e4m3fn), scale
    if mode != "int4":
        raise ValueError(mode)
    nib = torch.where(q.isnan(), 7.0, q.round().clamp(-8.0, 7.0)).to(torch.int32) & 0xF
    if nib.shape[-1] % 2:
        nib = torch.nn.functional.pad(nib, (0, 1))
    return ((nib[..., 0::2] << 4) | nib[..., 1::2]).to(torch.uint8), scale


def reference_dequantize(payload, scale, mode, dtype, dim):
    """Snapshot rows [..., dim] in the bank dtype, as the backward kernels rebuild them:
    dtype(float32(q) * scale)."""
    if mode == "fp8":
        q = payload.to(dtype)
    else:
        nib = torch.stack([payload >> 4, payload & 0xF], dim=-1).flatten(-2)[..., :dim]
        q = (nib.to(torch.int32) - 16 * (nib >= 8).to(torch.int32)).to(dtype)
    return (q.float() * scale).to(dtype)


# --------------------------------------------------------------- kernel-name allowlist
# GPU ops the write/read may launch: its own kernels, cuBLAS matmuls and torch
# elementwise/reduction/copy kernels.
OP_KERNELS = (
    r"^_stage_kernel$",
    r"\bp1_(merge|scan|pairs|bwd_pairs|bwd_gate)_kernel\b",
    r"^_wy_(diag|offdiag)_kernel$",
    r"\btoken_gather_kernel<",
    r"^_sdm_(dual_matmul|mover_dw|mover_phase_a|mover_gds)$",
    r"^_fused_(pre_bmm_residual|post_bmm)_kernel$",
    r"gemm|gemv|cutlass|nvjet|cublasLt::",
    r"<.*\bcublas[A-Z]",  # cuBLAS kernels templated on its own types (scal, dot, gemv reductions)
    r"^void at::native::(vectorized_elementwise|unrolled_elementwise|elementwise|reduce)_kernel<",
    r"^Mem(cpy|set) ",
)
# The CUDA kernels (sm_80+ / sm_90+): Phase 1's hot/cold pair kernel, the movers and record builders
# (cuda/mover_da2_cuda.py), the forward / backward kernels of cuda/pa_dc_cuda.py, and torch.cat in
# their glue.
DET_ONLY_KERNELS = (
    r"\bp1_pairs_hc_kernel\b",
    # Phase 1's keys kernel and parallel-walk pair kernel, the Phase-1 backward variant (cuda/phase1_cuda.py,
    # cuda/phase1_bwd_cuda.py), the read-record builder (cuda/px_qrec_cuda.py), the write movers
    # (cuda/mover_fx_cuda.py), the backward movers (cuda/gx_gds_cuda.py, cuda/gx_pa2_cuda.py; the GEMM of
    # cuda/gx_gemm_cuda.py matches "gemm" above). The dual matmul is _sdm_dual_matmul (above) in every call.
    r"\bp1_(keys_zw|pairs_w2|bwd_pairs_det|bwd_gate_det)_kernel\b",
    r"\bpx_qrec8_records_kernel\b",
    r"\bfx_dw2?_kernel\b",
    r"\bgx_(gds|pa2)_kernel\b",
    r"\bda2_(dot_reduce|dw|gdots|gds2f|gds|gscatter|krec|qrec|quant|segb|sgather|tiles)_kernel\b",
    r"\bdc_(copy_strided|gidx|gwr|gwr_r64|kpos|pa3|pa|pa_reduce|records_pa)_kernel\b",
    r"^void at::native::\(anonymous namespace\)::CatArrayBatchedCopy",
)
OP_KERNELS = OP_KERNELS + DET_ONLY_KERNELS


def nograd_slab_factor(memory_ops):
    """The no-grad forward's slab budget over ``memory_ops.SLAB_BYTES``: memory_ops.NOGRAD_SLAB_FACTOR (1 for a
    package without it)."""
    return getattr(memory_ops, "NOGRAD_SLAB_FACTOR", 1)


def set_nograd_budgets(monkeypatch, memory_ops, c):
    """Forces the no-grad path's slabs and stream groups for case ``c`` (``c.budgets`` "default", "tiny" or "two");
    returns the expected (slab_steps, group_streams) of a call, or None for the default budgets."""
    if c.budgets == "default":
        return None
    monkeypatch.setattr(memory_ops, "SLAB_CHUNKS_PER_SM", 0)
    if c.budgets == "tiny":
        monkeypatch.setattr(memory_ops, "SLAB_BYTES", 1)
        monkeypatch.setattr(memory_ops, "STEP_BYTES", 1)
        return 1, 1
    es = c.dtype.itemsize
    monkeypatch.setattr(memory_ops, "STEP_BYTES", 2 * c.C * c.W * c.hd * es)
    per_step = 2 * c.C * ((c.W + c.R) * 64 + 16 * c.C + 2 * c.hd * es)
    monkeypatch.setattr(memory_ops, "SLAB_BYTES", 2 * per_step)
    # The no-grad forward's slab budget is a multiple of SLAB_BYTES (nograd_slab_factor), so its slabs hold that many
    # more steps.
    return 2 * nograd_slab_factor(memory_ops), 2


def quant_tail_refusal(dtype, head_dim, snapshot_quant, compact_rows=0):
    """The class ("T0".."T3"; on a CUDA host also "S4", the int4 refusal) with which a training call (one that may need
    a backward) of fp8 / int4 snapshot rows of this bank dtype and head_dim is refused before any work
    (memory_ops.quant_tail_class), or None. The device only selects the S4 check (quant_config_class reads no device),
    so it is passed without an index: torch.cuda.current_device() would initialize CUDA, and test modules call this at
    import (test_memory_guards' fresh-process worker swaps the CUDA allocator after importing its module)."""
    from sparse_delta_memory.memory_ops import quant_tail_class
    device = torch.device("cuda") if torch.cuda.is_available() else None
    return quant_tail_class(dtype, head_dim, snapshot_quant, compact_rows, device=device)


def assert_quant_tail_refused(call, tail_class):
    """``call()`` raises UnsupportedConfigurationError (a ValueError) with ``tail_class``; returns the exception."""
    from sparse_delta_memory.memory_ops import UnsupportedConfigurationError
    with pytest.raises(UnsupportedConfigurationError) as err:
        call()
    assert isinstance(err.value, ValueError) and err.value.tail_class == tail_class, (err.value.tail_class, tail_class)
    return err.value


def profiled_kernel_names(fn, *args, **kwargs):
    """Run ``fn`` under torch.profiler; return (its result, the set of GPU op names:
    kernels, memcpys and memsets)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
    names = {e.name for e in prof.events() if "CUDA" in str(e.device_type)}
    return out, names


def kernels_outside(names, allowlist):
    """The names matched by none of the ``allowlist`` regular expressions (re.search)."""
    patterns = [re.compile(p) for p in allowlist]
    return sorted(n for n in names if not any(p.search(n) for p in patterns))
