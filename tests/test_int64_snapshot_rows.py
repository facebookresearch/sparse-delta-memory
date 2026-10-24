# Copyright (c) Meta Platforms, Inc. and affiliates.
"""64-bit snapshot-row indexing."""
import pytest
import torch

from sdm_testlib import bits_mismatch, kernel_write_read, make_op_inputs, requires_cuda
from test_mover_row_types import _assert_same, _inject_special, _phase1, _rows, _sdm

BF, F32 = torch.bfloat16, torch.float32


# ---------------------------------------------------------------------------------------- GPU: dc_pa's WIDE walk
def _dc_pa_runs(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, sph, dbg, flags, pad=0, seed=0, special=False):
    """dc_pa over the reversed step loop on identical inputs, once per dbg flag set in `flags` (OR-ed into dbg)."""
    bw, _, _, _, _, _, dc = _sdm()
    p1, k_idx, _, k_val, _, beta = _phase1(P, Nt, C, W, 4, pool, sph, bank_dtype, seed, C - 5)
    N, n = Nt * P, C * W
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    bpre = p1.dc_bpre.reshape(N, -1) if dbg & 1024 else torch.empty(0, device="cuda", dtype=torch.int32)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    ssr = P * C * W + pad
    snap = _rows(N * n if dbg & 256 else Nt * ssr, hd, bank_dtype, gen)
    dva = torch.randn(N * C, hd, device="cuda", generator=gen)
    bank0 = _rows(P * sph, hd, bank_dtype, gen)
    grad0 = _rows(P * sph, hd, grad_dtype, gen, 0.01)
    if special:
        _inject_special(bank0, sph), _inject_special(grad0, sph), _inject_special(snap, n)
    nsl = dc.slices(hd)
    _, depth, nw = bw.pa_plan(C, bank_dtype, grad_dtype, bank0.device)
    empty = torch.empty(0, device="cuda", dtype=grad_dtype)
    beta_f = beta.reshape(-1).contiguous()
    L = dc.tile_len(C, W, hd, P)
    fn = dc._module().dc_pa
    outs = []
    for flag in flags:
        grad, bank = grad0.clone(), bank0.clone()
        eff = torch.full((nsl, N * n), float("nan"), device="cuda")
        gg = torch.full((nsl, N * n), float("nan"), device="cuda")
        for ti in reversed(range(Nt)):
            fn(grad, bank, snap, ssr, dva, beta_f, kpay, rec, eff.view(-1) if nsl == 1 else eff,
               gg.view(-1) if nsl == 1 else gg, ti, P, C, W, L, nw, depth, dbg | flag, True, bpre, empty)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "bank": bank, "eff": eff, "g": gg})
    return outs


WIDE_CASES = [   # (bank dtype, grad dtype, P, Nt, C, W, hd, pool, dbg, pad)
    (BF, BF, 2, 3, 64, 8, 128, 48, 1792, 0),          # dense compact rows
    (BF, F32, 2, 2, 128, 16, 1024, 96, 1792, 0),      # 8 column slices, f32 grad rows
    (BF, BF, 1, 2, 192, 8, 128, 64, 1792, 0),         # C 192 (dc_pa3's plan: at most 8 warps per CTA)
    (BF, BF, 2, 3, 64, 8, 128, 48, 1792 | 65536, 0),  # the dense rows' L2 bulk prefetch (dbg 65536, NSL 1): from snap_w
    (F32, F32, 2, 3, 64, 8, 64, 48, 512, 0),          # per-entry rows
    (F32, F32, 2, 2, 32, 4, 960, 32, 512, 5),         # per-entry rows, padded steps (ssr = P * C * W + 5)
    (BF, BF, 2, 2, 64, 8, 132, 48, 512, 3),           # slices of 68 bf16 columns (dc_pa driven directly)
    (BF, BF, 2, 2, 64, 8, 132, 48, 768, 0),           # compact rows by entry position
    (F32, F32, 2, 2, 16, 4, 256, 16, 768, 0),
]


@requires_cuda
@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,W,hd,pool,dbg,pad", WIDE_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_W{c[5]}_hd{c[6]}_dbg{c[8]}_pad{c[9]}"
                              for c in WIDE_CASES])
def test_dc_pa_wide_walk_matches_the_32_bit_walk(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, dbg, pad):
    from sparse_delta_memory.cuda import pa_dc_cuda as dc
    ref, got = _dc_pa_runs(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, 4 * pool, dbg, (0, dc.DBG_WIDE), pad=pad)
    _assert_same(ref, got, "dc_pa WIDE vs 32-bit walk")
    assert not ref["grad"].isnan().any() and not ref["eff"][:, :1].isnan().all()


@requires_cuda
def test_dc_pa_wide_walk_special_values():
    from sparse_delta_memory.cuda import pa_dc_cuda as dc
    for bank_dtype, grad_dtype, hd, dbg in ((BF, BF, 320, 1792), (F32, F32, 128, 512), (BF, F32, 132, 768)):
        ref, got = _dc_pa_runs(bank_dtype, grad_dtype, 2, 2, 64, 8, hd, 48, 192, dbg, (0, dc.DBG_WIDE), seed=5,
                               special=True)
        _assert_same(ref, got, f"dc_pa WIDE vs 32-bit walk, special values {bank_dtype} hd {hd} dbg {dbg}")


class _DbgModule:
    """pa_dc_cuda's module recording each dc_pa launch's dbg (with this proxy the C++ step loop declines)."""
    def __init__(self, m, seen):
        self._m, self._seen = m, seen

    def __getattr__(self, k):
        return getattr(self._m, k)

    def dc_pa(self, *a):
        self._seen.append(int(a[17]))
        return self._m.dc_pa(*a)


@requires_cuda
@pytest.mark.parametrize("loop", ["cpp", "python"])
def test_backward_takes_the_wide_dc_pa_walk_from_the_threshold(monkeypatch, loop):
    """dc_pa route (gx_pa2 declined): below pa_dc_cuda._WIDE_SNAP_ROWS no dc_pa plan / launch carries DBG_WIDE, at 0
    every one does, with bitwise equal readings and gradients."""
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import gx_pa2_cuda as gxp, pa_dc_cuda as dc
    monkeypatch.setattr(gxp, "supported", lambda *a, **k: False)
    seen = []
    if loop == "python":
        proxy = _DbgModule(dc._module(), seen)
        monkeypatch.setattr(dc, "_module", lambda: proxy)
    else:
        orig = dc.pa_step_plan

        def spy(*a, **k):
            seen.append(int(a[17]))
            return orig(*a, **k)
        monkeypatch.setattr(dc, "pa_step_plan", spy)
    assert dc._WIDE_SNAP_ROWS == 2 ** 31
    d = make_op_inputs(2, 256, 16, 16, 128, 2048, BF, seed=3)
    runs = []
    for thr in (2 ** 31, 0):
        monkeypatch.setattr(dc, "_WIDE_SNAP_ROWS", thr)
        del seen[:]
        readings, grads, _ = kernel_write_read(d, chunk_size=64, sph=2048)
        torch.cuda.synchronize()
        runs.append((readings, grads, list(seen), bw.LAST_LOOP.get("name")))
    (r1, g1, s1, l1), (r2, g2, s2, l2) = runs
    assert not bits_mismatch(r2, r1) and not any(bits_mismatch(g2[k], g1[k]) for k in g1)
    if loop == "cpp" and "python_step_loop" in (l1, l2):   # (no C++ step-loop plan for this shape here)
        pytest.skip(f"the backward ran its Python step loop here ({l1}, {l2}); the python case checks that loop")
    assert s1 and not any(x & dc.DBG_WIDE for x in s1), s1
    assert s2 and all(x & dc.DBG_WIDE for x in s2) and [x & ~dc.DBG_WIDE for x in s2] == s1, (s1, s2)


# ------------------------------------------------------------------------------------- GPU: mover_fx's snapc map
def _modern_fx():
    from sparse_delta_memory.cuda import mover_fx_cuda as fxm
    if fxm._portable(torch.device("cuda")):
        pytest.skip("the portable side (a100 / rtx) encodes no tensor maps")
    return fxm


@requires_cuda
@pytest.mark.parametrize("dtype,hd,C,W", [(BF, 384, 64, 16), (F32, 256, 64, 8)])
def test_saving_forward_with_the_zeroed_snapc_map_has_the_same_bits(monkeypatch, dtype, hd, C, W):
    """The saving forward of column-sliced heads through mover_fx's DW (fx_maps, not the two-slice walk)."""
    fxm = _modern_fx()
    real = fxm._module()
    seen = []

    class _Maps:
        def __getattr__(self, k):
            return getattr(real, k)

        def fx_maps(self, *a):
            out = real.fx_maps(*a)
            seen.append((int(a[3]), bool(out[0].any())))
            return out
        def fx_dw_plan(self, *a):        # the maps each DW plan is built with (a[13]); 2-D maps: [2, 128] bytes
            if a[13].numel() == 256:
                plans.append(bool(a[13][0].any()))
            return real.fx_dw_plan(*a)
    plans = []
    proxy = _Maps()
    monkeypatch.setattr(fxm, "_module", lambda: proxy)
    monkeypatch.setattr(fxm, "_MAPS", {})       # ONE cache for the three runs (the threshold is in the key)
    assert fxm._WIDE_SNAP_ROWS == 2 ** 31
    P, T, sph = 2, 256, 2048
    d = make_op_inputs(P, T, W, W, hd, sph, dtype, seed=5)
    runs = []
    for thr in (2 ** 31, 0, 2 ** 31):
        monkeypatch.setattr(fxm, "_WIDE_SNAP_ROWS", thr)
        del seen[:], plans[:]
        readings, grads, _ = kernel_write_read(d, chunk_size=C, sph=sph)
        torch.cuda.synchronize()
        runs.append((readings, grads, list(seen), list(plans)))
    (r1, g1, s1, p1), (r2, g2, s2, p2), (r3, g3, s3, p3) = runs
    assert s1 and all(w == 2 ** 31 and enc for w, enc in s1), s1
    assert s2 and all(w == 0 and not enc for w, enc in s2), s2          # a changed threshold re-encodes (no stale hit)
    assert all(w == 2 ** 31 and enc for w, enc in s3), s3               # (hits allowed: then no call at all)
    assert p1 and all(p1) and p2 and not any(p2) and p3 and all(p3), (p1, p2, p3)
    assert {k[-1] for k in fxm._MAPS} == {0, 2 ** 31}, [k[-1] for k in fxm._MAPS]
    for rr, gg in ((r2, g2), (r3, g3)):
        assert not bits_mismatch(rr, r1), "readings"
        for k in g1:
            assert not bits_mismatch(gg[k], g1[k]), f"grad {k}"


# ------------------------------------------------------------------- GPU: a real snapshot past 2**31 rows (slow)
@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("which", ["dc_pa", "gx_pa2"])
def test_phase_a_reads_snapshot_rows_past_2p31(which):
    """A real per-entry snapshot (dbg 512) of 2**31 + P * C * W rows with snap_step_rows 2**31."""
    bw, _, _, _, gxp, _, dc = _sdm()
    hd = 4 if which == "dc_pa" else 8
    P, Nt, C, W, pool, sph = 2, 2, 64, 8, 48, 192
    n = C * W
    big_rows = 2 ** 31 + P * n
    free, _ = torch.cuda.mem_get_info()
    if free < big_rows * hd * 2 + (2 << 30):
        pytest.skip(f"needs {big_rows * hd * 2 / 2 ** 30:.1f} GiB of free device memory for the snapshot")
    p1, k_idx, _, k_val, _, beta = _phase1(P, Nt, C, W, 4, pool, sph, BF, 7, C - 5)
    N = Nt * P
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    gen = torch.Generator(device="cuda").manual_seed(8)
    small = _rows(2 * P * n, hd, BF, gen)                 # step 0's rows, then step 1's
    dva = torch.randn(N * C, hd, device="cuda", generator=gen)
    bank0 = _rows(P * sph, hd, BF, gen)
    grad0 = _rows(P * sph, hd, BF, gen, 0.01)
    beta_f = beta.reshape(-1).contiguous()
    empty_i = torch.empty(0, device="cuda", dtype=torch.int32)
    empty = torch.empty(0, device="cuda", dtype=BF)
    _, depth, nw = bw.pa_plan(C, BF, BF, bank0.device)
    L = dc.tile_len(C, W, hd, P)
    outs = []
    for snap_kind in ("small", "big"):
        if snap_kind == "big":     # rows [P * n, 2**31) are never read (left uninitialized)
            snap = torch.empty(big_rows, hd, device="cuda", dtype=BF)
            snap[:P * n] = small[:P * n]
            snap[2 ** 31:] = small[P * n:]
            ssr = 2 ** 31
        else:
            snap, ssr = small, P * n
        grad, bank = grad0.clone(), bank0.clone()
        eff = torch.full((N * n,), float("nan"), device="cuda")
        gg = torch.full((N * n,), float("nan"), device="cuda")
        if which == "dc_pa":
            fn, w_, d_, L_ = dc._module().dc_pa, nw, depth, L
        else:
            assert gxp.supported(grad, bank, snap, C, 512, depth, nw, False), snap_kind
            fn = gxp.launcher(pdl=False, bank=bank, grad=grad, snap=snap)
            w_, d_ = gxp.plan(grad, bank, C)
            L_ = gxp.tile_len(C, W, hd, P, L, device=bank.device)
        for ti in reversed(range(Nt)):
            fn(grad, bank, snap, ssr, dva, beta_f, kpay, rec, eff, gg, ti, P, C, W, L_, w_, d_, 512, True, empty_i,
               empty)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "bank": bank, "eff": eff, "g": gg})
        del snap
    _assert_same(outs[0], outs[1], f"{which}: rows past 2**31 vs the same rows below")
    assert not outs[0]["grad"].isnan().any() and not outs[0]["eff"].isnan().all()


# --------------------------------------- GPU: the op past 2**31 snapshot rows, forward + backward (slow; INT64_BIG=1)
def _big_inputs(P, T, W, R, hd, sph, dtype, seed):
    """sdm_write_read inputs [P, T, *] built on the GPU in token batches."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    k_idx = torch.empty(P, T, W, device="cuda", dtype=torch.int32)
    q_idx = torch.empty(P, T, R, device="cuda", dtype=torch.int32)
    B = 16384
    for p in range(P):
        for t0 in range(0, T, B):
            t1 = min(T, t0 + B)
            r = torch.rand(t1 - t0, sph, device="cuda", generator=gen)
            k_idx[p, t0:t1] = (r.topk(W, dim=-1).indices.sort(dim=-1).values + p * sph).int()
            q_idx[p, t0:t1] = (r.topk(R, dim=-1, largest=False).indices.sort(dim=-1).values + p * sph).int()
            del r

    def rnd(*shape, scale):
        out = torch.empty(*shape, device="cuda", dtype=dtype)
        for p in range(shape[0]):
            out[p] = (torch.randn(*shape[1:], device="cuda", generator=gen) * scale).to(dtype)
        return out
    k_val, q_val, v = rnd(P, T, W, scale=0.3), rnd(P, T, R, scale=0.3), rnd(P, T, hd, scale=0.1)
    g = (torch.rand(P, T, 1, device="cuda", generator=gen) * -0.49 - 0.01).to(dtype)
    beta = (torch.rand(P, T, 1, device="cuda", generator=gen) * 0.3).to(dtype)
    memory = (torch.randn(P * sph, hd, device="cuda", generator=gen) * 0.1).to(dtype)
    return dict(memory=memory, k_idx=k_idx, k_val=k_val, v=v, beta=beta, g=g, q_idx=q_idx, q_val=q_val)


_LEAVES = ("k_val", "v", "beta", "g", "q_val")


def _op_range(d, t0, t1, mem, C, sph, quant, cot):
    """The training call on tokens [t0, t1) of every stream from bank `mem` (a leaf the forward advances in place):
    (readings [P, t1 - t0, hd], the bank after the forward, grads {name: [P, t1 - t0, *]} incl. "memory"), or, with
    cot None, a no-grad call (returns None; `mem` holds the bank after it)."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    P = d["k_idx"].shape[0]
    sl = slice(t0, t1)
    xs = {n: d[n][:, sl].detach().contiguous() for n in _LEAVES}
    args = lambda: (d["k_idx"][:, sl].contiguous(), xs["k_val"], xs["v"], xs["beta"], xs["g"],  # noqa: E731
                    d["q_idx"][:, sl].contiguous(), xs["q_val"])
    kw = dict(chunk_size=C, num_memory_slots=sph, num_parallel=P, snapshot_quant=quant)
    if cot is None:
        with torch.no_grad():
            sdm_write_read(mem, *args(), **kw)
        return None
    for n in _LEAVES:
        xs[n].requires_grad_(True)
    r = sdm_write_read(mem, *args(), **kw)
    bank = mem.detach().clone()
    gr = torch.autograd.grad(r, [mem] + [xs[n] for n in _LEAVES], grad_outputs=cot)
    hd = r.shape[-1]
    return r.detach().view(P, t1 - t0, hd), bank, dict(zip(("memory",) + _LEAVES, gr))


def _same(a, b):
    """(bitwise equal, the max |a - b| / max |a|) of two tensors (+0.0 and -0.0 count as equal only bitwise-different)."""
    eq = not bits_mismatch(a, b)
    da = (a.float() - b.float()).abs().max().item()
    return eq, da / max(a.float().abs().max().item(), 1e-30)


def _suffix_check(d, Nt, K, C, sph, quant, seed):
    """Full training call over Nt chunks vs the same op on its last K chunks from the bank the first Nt - K chunks leave
    (a no-grad call on them): same P, C, W, hd (the same geometry), so the suffix's readings, the bank after the forward
    and every gradient of the suffix's inputs must be bitwise equal (the suffix's gradients do not depend on earlier
    chunks; its backward restores the same snapshot rows, quantized alike)."""
    P, _, hd = d["v"].shape
    T, s0 = Nt * C, (Nt - K) * C
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cot = (torch.randn(P, T, hd, device="cuda", generator=gen) * 0.1).to(d["v"].dtype)
    mem = d["memory"].clone().requires_grad_(True)
    r, bank, gr = _op_range(d, 0, T, mem, C, sph, quant, cot.view(P * T, hd))
    full = {"readings": r[:, s0:].clone(), "bank": bank}
    full.update({n: gr[n][:, s0:].clone() for n in _LEAVES})
    del r, gr, mem
    torch.cuda.empty_cache()
    mid = d["memory"].clone()
    _op_range(d, 0, s0, mid, C, sph, quant, None)
    mem = mid.clone().requires_grad_(True)
    r, bank, gr = _op_range(d, s0, T, mem, C, sph, quant, cot[:, s0:].contiguous().view(P * (T - s0), hd))
    suf = {"readings": r, "bank": bank}
    suf.update({n: gr[n] for n in _LEAVES})
    return {k: _same(full[k], suf[k]) for k in full}


BIG_CASES = [   # (snapshot_quant, head_dim, bank dtype): fp8 / int4 past 2**31 rows, the hd % 4 tails (class R)
    ("int4", 128, BF), ("int4", 64, BF), ("none", 3, BF), ("none", 5, F32),
]


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("quant,hd,dtype", BIG_CASES, ids=[f"{c[0]}_hd{c[1]}_{str(c[2])[6:]}" for c in BIG_CASES])
def test_op_past_2p31_snapshot_rows_matches_its_suffix(quant, hd, dtype):
    """A training call (forward + backward) whose snapshot holds more than 2**31 rows."""
    import json
    import os
    if os.environ.get("INT64_BIG") != "1":
        pytest.skip("set INT64_BIG=1 (a > 2**31-row snapshot: minutes, up to ~250 GiB of device memory)")
    P, C, W, R, sph, K = 16, 128, 256, 8, 4096, 2
    n = C * W
    Nt = -(-(2 ** 31 + 2 ** 20) // (P * n)) + K             # steps: the last K steps' chunks start past 2**31 + 2**20 rows
    assert (Nt - K) * P * n >= 2 ** 31 + 2 ** 20 and (Nt - K - 1) * P * n < 2 ** 31 + 2 ** 20
    rep = {"quant": quant, "hd": hd, "dtype": str(dtype), "P": P, "C": C, "W": W, "R": R, "Nt": Nt, "K": K,
           "rows": Nt * P * n, "gpu": torch.cuda.get_device_name()}
    torch.cuda.empty_cache()
    d = _big_inputs(P, Nt * C, W, R, hd, sph, dtype, seed=11)
    # method control: the same check at Nt = 6 (rows < 2**31 everywhere), on the first 6 chunks' inputs
    ctl = _suffix_check(d, 6, K, C, sph, quant, seed=12)
    rep["control"] = {k: [bool(v[0]), v[1]] for k, v in ctl.items()}
    assert all(v[0] for v in ctl.values()), ("method control (all rows < 2**31) not bitwise", ctl)
    # footprint: a full training call over Nt_cal steps, measured, extrapolated linearly in entries
    Nt_cal = 64
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cal = _suffix_check(d, Nt_cal, K, C, sph, quant, seed=13)
    torch.cuda.synchronize()
    rep["control64"] = {k: [bool(v[0]), v[1]] for k, v in cal.items()}
    per_step = (torch.cuda.max_memory_allocated() - base) / Nt_cal
    free, total = torch.cuda.mem_get_info()
    need = per_step * Nt * 1.05
    rep.update(per_step_bytes=per_step, need_gib=need / 2 ** 30, free_gib=free / 2 ** 30, total_gib=total / 2 ** 30,
               inputs_gib=base / 2 ** 30)
    print("INT64_BIG " + json.dumps(rep))
    if need > 0.95 * (free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()):
        pytest.skip(f"needs ~{need / 2 ** 30:.0f} GiB beyond the inputs, {free / 2 ** 30:.0f} GiB free: {rep}")
    torch.cuda.reset_peak_memory_stats()
    res = _suffix_check(d, Nt, K, C, sph, quant, seed=14)
    rep["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    rep["big"] = {k: [bool(v[0]), v[1]] for k, v in res.items()}
    print("INT64_BIG " + json.dumps(rep))
    path = os.environ.get("INT64_REPORT")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rep) + "\n")
    assert all(v[0] for v in res.values()), ("past 2**31 rows: not bitwise equal to the suffix's 32-bit run", rep)
