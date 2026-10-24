# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The slim per-step host path: gx_pa2's and gx_gds's bf16 entries (gx_pa2_bf / gx_pa2_bf_pdl, gx_gds_bf).

The backward picks each entry once per backward (gx_pa2_cuda.launcher(bank=, grad=, snap=)) or per factory call
(gx_gds_cuda.make_gds): bf16 bank and grad rows below 2^32 elements (gx_pa2: with bf16 snapshot rows in 16-byte pieces,
the regular module) take the bf16 entries, whose steps carry no row-type, quantization, piece-width or offset-width
selection; everything else takes the generic entries. The bf16 entries must launch exactly what the generic entries
launch for those rows. Observed here without any instrumentation of the production path: every kernel launch of a step
loop is recorded by torch.profiler (CUDA activity) as (launch API, kernel, grid, block, shared memory), and the outputs
are compared bitwise; across every warp count and ring depth the launcher accepts (and the same refusal past the
device's opt-in shared memory), both snapshot readers, both record layouts, plain and programmatic launches; plus the
routing at the exact 2^32-element edges and the whole op. Cases run on every backend the
package serves where the shape is admitted (programmatic launches: the modern backend only).
"""
import itertools
import json

import pytest
import torch

from sdm_testlib import OP_INPUTS, bits_mismatch, make_op_inputs, requires_cuda
from test_mover_row_types import _assert_same, _op_run, _phase1, _rows, _sdm


def _backend():
    if not torch.cuda.is_available():
        return None
    from sparse_delta_memory.cuda import _device
    try:
        return _device.backend(torch.device("cuda"))
    except RuntimeError:
        return None


pytestmark = [requires_cuda, pytest.mark.skipif(_backend() is None, reason="needs a CUDA backend of the package")]
BF = torch.bfloat16
_trace_ids = itertools.count()


def _pdl_ok():
    return _backend() == "modern"


def _cap():
    from sparse_delta_memory.cuda._device import smem_cap
    return smem_cap(torch.device("cuda"))


class _Proxy:
    """An extension module with some entries replaced; counts the calls of every callable it hands out. Like a real
    extension module it hands out ONE object per entry name, so identity tests in the package see what they would see on
    the real module (make_gds's plan passes `gfn is not mod.gx_gds` - "the bf16 entry" - to gx_gds_plan). An entry
    replaced by a str is an alias: the named entry's own object (a module whose bf16 entry IS the generic one)."""

    def __init__(self, m, **over):
        self._m, self._over, self.calls, self.plan_args, self._fns = m, over, {}, [], {}

    def __getattr__(self, k):
        o = self._over.get(k)
        if isinstance(o, str):
            return getattr(self, o)
        f = o or getattr(self._m, k)
        if not callable(f):
            return f
        if k not in self._fns:
            def call(*a, **kw):
                self.calls[k] = self.calls.get(k, 0) + 1
                if k == "gx_gds_plan":          # the entry the plan is made with = its last argument (True: bf16)
                    self.plan_args.append((a, dict(kw)))
                return f(*a, **kw)
            self._fns[k] = call
        return self._fns[k]


def _launches(fn, tmp_path):
    """fn() under torch.profiler (CUDA activity): its kernel launches in issue order as (launch API without its version
    suffix, kernel name, grid, block, dynamic + static shared memory), from the exported trace."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = tmp_path / f"trace_{next(_trace_ids)}.json"
    prof.export_chrome_trace(str(path))
    ev = json.load(open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    kern = {e["args"]["correlation"]: e for e in ev
            if e.get("cat") == "kernel" and e.get("ph") == "X" and "correlation" in e.get("args", {})}
    out = []
    for e in sorted((e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"
                     and e.get("name", "").startswith(("cudaLaunchKernel", "cuLaunchKernel"))), key=lambda e: e["ts"]):
        k = kern.get(e.get("args", {}).get("correlation"))
        assert k is not None, f"a kernel launch call without its kernel record: {e.get('name')} {e.get('args')}"
        a = k["args"]
        grid, block, smem = a.get("grid"), a.get("block"), a.get("shared memory")
        dims_ok = all(isinstance(x, list) and len(x) == 3 and all(isinstance(v, int) and v >= 1 for v in x)
                      for x in (grid, block))
        assert dims_ok and isinstance(smem, int) and smem >= 0, f"incomplete launch record of {k['name']}: {a}"
        out.append((e["name"].split("_v")[0], k["name"], tuple(grid), tuple(block), smem))
    return out


# ------------------------------------------------------------------------------------------------------- gx_pa2 ----
def _pa_inputs(P, Nt, C, W, hd, pool, dbg, seed=0):
    _, _, _, _, gxpa2, _, dc = _sdm()
    p1, k_idx, _, k_val, _, beta = _phase1(P, Nt, C, W, 4, pool, 4 * pool, BF, seed, C - 5)
    N, n = Nt * P, C * W
    kpay = p1.spay_k.reshape(-1, n)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    d = {"P": P, "Nt": Nt, "C": C, "W": W, "hd": hd, "dbg": dbg, "N": N, "n": n, "nsl": dc.slices(hd), "kpay": kpay,
         "rec": dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe),
         "bpre": p1.dc_bpre.reshape(N, -1) if dbg & 1024 else torch.empty(0, device="cuda", dtype=torch.int32),
         "snap": _rows(N * n, hd, BF, gen), "dva": torch.randn(N * C, hd, device="cuda", generator=gen),
         "bank": _rows(P * 4 * pool, hd, BF, gen), "grad": _rows(P * 4 * pool, hd, BF, gen, 0.01),
         "beta": beta.reshape(-1).contiguous(), "empty": torch.empty(0, device="cuda", dtype=BF)}
    d["L"] = gxpa2.tile_len(C, W, hd, P, dc.tile_len(C, W, hd, P), device=d["bank"].device)
    return d


def _pa_run(fn, d, nw, depth, pdl, tmp_path=None):
    """The reversed step loop of fn on fresh copies of d's grad and bank (with a kernel writing only the d_u rows before
    each programmatic launch, as the backward's GEMM): (outputs, launches or None)."""
    grad, bank = d["grad"].clone(), d["bank"].clone()
    eff = torch.full((d["nsl"], d["N"] * d["n"]), float("nan"), device="cuda")
    gg = torch.full((d["nsl"], d["N"] * d["n"]), float("nan"), device="cuda")
    one = d["nsl"] == 1

    def loop():
        for ti in reversed(range(d["Nt"])):
            if pdl:
                d["dva"].add_(0.0)
            fn(grad, bank, d["snap"], d["P"] * d["C"] * d["W"], d["dva"], d["beta"], d["kpay"], d["rec"],
               eff.view(-1) if one else eff, gg.view(-1) if one else gg, ti, d["P"], d["C"], d["W"], d["L"], nw, depth,
               d["dbg"], True, d["bpre"], d["empty"])
    launches = _launches(loop, tmp_path) if tmp_path is not None else (loop() or None)
    torch.cuda.synchronize()
    return {"grad": grad, "bank": bank, "eff": eff, "g": gg}, launches


def _entries(pdl):
    _, _, _, _, gxpa2, _, _ = _sdm()
    m = gxpa2._module()
    return (m.gx_pa2_pdl, m.gx_pa2_bf_pdl) if pdl else (m.gx_pa2, m.gx_pa2_bf)


PLANS = [(nw, depth) for nw in (1, 2, 3, 4, 5, 6, 7, 8, 16) for depth in (8, 16)]


@pytest.mark.parametrize("pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("dbg", [1792, 512], ids=["dense", "per_entry"])
def test_gx_pa2_bf16_entry_launches_every_plan(dbg, pdl, tmp_path):
    """Every warp count (1 .. 8 on the 256-thread instantiations, 16 on the 512-thread ones) and ring depth (8, 16) the
    launcher accepts: the bf16 entry launches the generic entry's kernel with the same API, grid, block and shared
    memory, and writes the same bits (also across plans: the outputs do not depend on the plan); a plan past the
    device's opt-in shared memory is refused by both entries with the same error."""
    if pdl and not _pdl_ok():
        pytest.skip("programmatic launches: the modern backend only")
    _, _, _, _, gxpa2, _, _ = _sdm()
    generic, fast = _entries(pdl)
    cap = _cap()
    first, refused = None, 0
    for C, P, Nt, W, hd, pool in ((64, 2, 2, 8, 128, 48), (256, 1, 2, 4, 256, 64)):
        d = _pa_inputs(P, Nt, C, W, hd, pool, dbg)
        for nw, depth in PLANS:
            if gxpa2.smem(C, nw, depth, 16) > cap:
                errs = []
                for fn in (generic, fast):
                    with pytest.raises(RuntimeError, match="bytes of shared memory exceed") as ei:
                        _pa_run(fn, d, nw, depth, pdl)
                    errs.append(str(ei.value).split("\n")[0])
                assert errs[0] == errs[1], errs
                refused += 1
                continue
            (og, lg), (of, lf) = (_pa_run(fn, d, nw, depth, pdl, tmp_path) for fn in (generic, fast))
            assert lf == lg, f"bf16 entry launches differ (C {C}, nw {nw}, depth {depth}): {lf} vs {lg}"
            pa = [x for x in lg if "gx_pa2_kernel" in x[1]]   # (pdl: the d_u-row kernel runs before each step)
            assert len(pa) == Nt, f"profiler saw {lg} (C {C}, nw {nw}, depth {depth})"
            assert {x[0] for x in pa} == ({"cudaLaunchKernelExC"} if pdl else {"cudaLaunchKernel"}), pa
            ctas = -(-(-(-(C * W) // d["L"])) // nw)
            assert all(x[2] == (P * d["nsl"] * ctas, 1, 1) and x[3] == (32 * nw, 1, 1)
                       and x[4] == gxpa2.smem(C, nw, depth, 16) for x in pa), pa
            _assert_same(og, of, f"gx_pa2_bf vs gx_pa2 (C {C}, nw {nw}, depth {depth}, dbg {dbg}, pdl {pdl})")
            if C == 64:
                if first is None:
                    first = og
                    assert not og["grad"].isnan().any() and not og["eff"][:, :1].isnan().all()
                else:
                    _assert_same(first, og, f"plan independence (nw {nw}, depth {depth})")
    assert refused >= 1   # C 256, 16 warps, depth 16 exceeds every device's opt-in limit


PA_CASES = [   # (P, Nt, C, W, hd, pool): ring depth 16 (C 64), 16 warps (C 192), 8 slices, uneven 16-byte slices
    (2, 3, 64, 8, 128, 48),
    (1, 2, 192, 8, 256, 64),
    (2, 2, 128, 16, 960, 96),
    (2, 2, 32, 4, 280, 32),
]


@pytest.mark.parametrize("pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("dbg", [1792, 512], ids=["dense", "per_entry"])
@pytest.mark.parametrize("P,Nt,C,W,hd,pool", PA_CASES, ids=[f"C{c[2]}_W{c[3]}_hd{c[4]}" for c in PA_CASES])
def test_gx_pa2_bf16_entry_matches_generic(P, Nt, C, W, hd, pool, dbg, pdl, tmp_path):
    """The backward's own plan (gx_pa2_cuda.plan()) on admitted shapes: the launcher hands out the bf16 entry, which
    launches what the generic entry launches and writes the same bits."""
    if pdl and not _pdl_ok():
        pytest.skip("programmatic launches: the modern backend only")
    _, _, _, _, gxpa2, _, _ = _sdm()
    d = _pa_inputs(P, Nt, C, W, hd, pool, dbg)
    if (gxpa2.plan(d["grad"], d["bank"], C) is None
            or not gxpa2.supported(d["grad"], d["bank"], d["snap"], C, dbg, 8, 8, False)):
        pytest.skip(f"C {C} hd {hd}: not admitted on this backend")
    nw, depth = gxpa2.plan(d["grad"], d["bank"], C)
    generic, fast = _entries(pdl)
    assert gxpa2.launcher(pdl=pdl, bank=d["bank"], grad=d["grad"], snap=d["snap"]) is fast
    (og, lg), (of, lf) = (_pa_run(fn, d, nw, depth, pdl, tmp_path) for fn in (generic, fast))
    assert lg and lf == lg, (lf, lg)
    _assert_same(og, of, f"gx_pa2_bf vs gx_pa2 (dbg {dbg}, pdl {pdl})")
    assert not og["grad"].isnan().any() and not og["eff"][:, :1].isnan().all()


# ------------------------------------------------------------------------------------------------------- gx_gds ----
def _gds_run(nw, rec8, pdl, monkeypatch, real_mod, tmp_path, force_generic, P=2, Nt=3, C=64, R=8, hd=960, pool=48,
             bank_dtype=BF, grad_dtype=BF, seed=0):
    """make_gds on identical inputs (warps forced to nw) with gx_gds_cuda._module() replaced by a counting proxy (the
    bf16 entry forced onto the generic one when force_generic), the reversed step loop launched with pdl: (outputs,
    launches, proxy calls)."""
    _, _, _, gxm, *_ = _sdm()
    vl = C - 3
    p1, _, q_idx, _, _, _ = _phase1(P, Nt, C, 4, R, pool, 4 * pool, bank_dtype, seed, vl)
    N, n = Nt * P, C * R
    gen = torch.Generator(device="cuda").manual_seed(seed + 2)
    bank = _rows(P * 4 * pool, hd, bank_dtype, gen)
    grad = _rows(P * 4 * pool, hd, grad_dtype, gen, 0.01)
    gr = torch.randn(Nt * P * C, hd, device="cuda", generator=gen)
    recs = p1.qrec8.reshape(N, n, 2) if rec8 else p1.qrec.reshape(N, n, 4)
    proxy = _Proxy(real_mod, **({"gx_gds_bf": real_mod.gx_gds} if force_generic else {}))
    monkeypatch.setattr(gxm, "_module", lambda: proxy)
    if nw is not None:
        monkeypatch.setattr(gxm, "geometry", lambda *a, **k: (nw, {4: 32, 8: 128, 16: 256}[nw]))
    dqw = torch.full((N, C, R), float("nan"), device="cuda")
    kw = dict(chunk_size=C, num_parallel=P, steps=Nt, q_idx=q_idx, valid_tokens=(Nt - 1) * C + vl, tiles_q=None)
    assert gxm.supported(bank, grad, C, R)
    launch = gxm.make_gds(bank, grad, gr, p1.spay_q, recs, dqw, **kw)
    assert launch.geom["impl"] == "gx" and (nw is None or launch.geom["nw"] == nw)

    def loop():
        for ti in reversed(range(Nt)):
            launch(ti, pdl=pdl)
    launches = _launches(loop, tmp_path)
    torch.cuda.synchronize()
    return {"grad": grad, "d_q_w": dqw}, launches, dict(proxy.calls), dict(launch.geom, P=P)


@pytest.mark.parametrize("pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("rec8", [False, True], ids=["qrec16", "rec8"])
@pytest.mark.parametrize("nw", [4, 8, 16])
def test_gx_gds_bf16_entry_launches(nw, rec8, pdl, monkeypatch, tmp_path):
    """make_gds hands every step of bf16 / bf16 rows to gx_gds_bf; it launches what gx_gds launches (API, kernel, grid,
    block, shared memory: programmatic launches under pdl) and writes the same bits."""
    if pdl and not _pdl_ok():
        pytest.skip("programmatic launches: the modern backend only")
    _, _, _, gxm, *_ = _sdm()
    real = gxm._module()
    (og, lg, cg, geo), (of, lf, cf, _) = (_gds_run(nw, rec8, pdl, monkeypatch, real, tmp_path, force)
                                          for force in (True, False))
    assert cf.get("gx_gds_bf") == 3 and "gx_gds" not in cf, cf
    assert cg.get("gx_gds_bf") == 3, cg     # forced onto the generic entry's code
    assert lg and lf == lg, (lf, lg)
    gds = [x for x in lg if "gx_gds_kernel" in x[1]]
    assert len(gds) == 3 and all(x[2] == (geo["P"] * geo["nsl"] * geo["K"], 1, 1) and x[3] == (32 * nw, 1, 1)
                                 for x in gds), (gds, geo)
    assert {x[0] for x in gds} == ({"cudaLaunchKernelExC"} if pdl else {"cudaLaunchKernel"}), gds
    _assert_same(og, of, f"gx_gds_bf vs gx_gds (nw {nw}, rec8 {rec8}, pdl {pdl})")
    assert not og["d_q_w"].isnan().any()


def test_gx_gds_other_rows_take_the_generic_entry(monkeypatch, tmp_path):
    _, _, _, gxm, *_ = _sdm()
    real = gxm._module()
    for bank_dtype, grad_dtype in ((BF, torch.float32), (torch.float32, torch.float32)):
        _, _, calls, _ = _gds_run(None, True, False, monkeypatch, real, tmp_path, False, Nt=2, hd=128,
                                  bank_dtype=bank_dtype, grad_dtype=grad_dtype)
        assert calls.get("gx_gds", 0) == 2 and "gx_gds_bf" not in calls, (bank_dtype, calls)


# ------------------------------------------------------------------------------------------------------ routing ----
class _Rows:
    """The attributes the entry selection reads (dtype, shape, numel, dim), without memory."""

    def __init__(self, dtype, shape, numel=None):
        self.dtype, self.shape = dtype, tuple(shape)
        self._n = numel if numel is not None else self.shape[0] * self.shape[1]

    def numel(self):
        return self._n

    def dim(self):
        return len(self.shape)


def test_entry_selection():
    _, _, _, gxm, gxpa2, _, _ = _sdm()
    m = gxpa2._module()
    b16 = torch.zeros(64, 128, device="cuda", dtype=BF)
    f32 = torch.zeros(64, 128, device="cuda")
    q8 = torch.zeros(64, 128, device="cuda", dtype=torch.uint8)      # fp8 compact rows (bytes)
    assert gxpa2.launcher(pdl=False, bank=b16, grad=b16, snap=b16) is m.gx_pa2_bf
    assert gxpa2.launcher(pdl=True, bank=b16, grad=b16, snap=b16) is m.gx_pa2_bf_pdl
    for grad, snap in ((f32, b16), (b16, q8), (b16, None), (None, b16)):
        assert gxpa2.launcher(pdl=False, bank=b16, grad=grad, snap=snap) is m.gx_pa2
        assert gxpa2.launcher(pdl=True, bank=b16, grad=grad, snap=snap) is m.gx_pa2_pdl
    assert gxpa2.launcher(pdl=False, bank=f32, grad=f32, snap=f32) is m.gx_pa2
    assert gxpa2.launcher(pdl=False) is m.gx_pa2 and gxpa2.launcher(pdl=True) is m.gx_pa2_pdl
    t16 = torch.zeros(64, 132, device="cuda", dtype=BF)   # 8-byte pieces: the tail module's generic entry
    tm = gxpa2._tail_module()
    assert gxpa2.launcher(pdl=False, bank=t16, grad=t16, snap=t16) is tm.gx_pa2
    assert not hasattr(tm, "gx_pa2_bf") and not hasattr(gxm._tail_module(), "gx_gds_bf")


def test_routing_at_the_2_32_edges():
    """The exact element-count edges of both selections: below 2^32 the bf16 entry, at 2^32 the generic one."""
    _, _, _, gxm, gxpa2, _, _ = _sdm()
    m = gxpa2._module()
    bank, snap = _Rows(BF, (2 ** 25, 128)), _Rows(BF, (64, 128))
    assert gxpa2.launcher(pdl=False, bank=bank, grad=_Rows(BF, (2 ** 25, 128), 2 ** 32 - 1), snap=snap) is m.gx_pa2_bf
    assert gxpa2.launcher(pdl=False, bank=bank, grad=_Rows(BF, (2 ** 25, 128), 2 ** 32), snap=snap) is m.gx_pa2
    assert gxpa2.launcher(pdl=True, bank=bank, grad=_Rows(BF, (2 ** 25, 128), 2 ** 32), snap=snap) is m.gx_pa2_pdl
    assert not gxm.wide_offsets(_Rows(BF, (2 ** 25, 128), 2 ** 32 - 1), BF)
    assert gxm.wide_offsets(_Rows(BF, (2 ** 25, 128), 2 ** 32), BF)
    assert gxm.wide_offsets(_Rows(BF, (64, 128)), torch.float32) and gxm.wide_offsets(_Rows(torch.float32, (64, 128)),
                                                                                      torch.float32)


def _gds_bf_call(bank, grad, hd, gr_dtype=torch.float32):
    i32 = dict(device="cuda", dtype=torch.int32)
    _, _, _, gxm, *_ = _sdm()
    gxm._module().gx_gds_bf(bank, grad, torch.zeros(64, hd, device="cuda", dtype=gr_dtype), 0,
                            torch.zeros(1, 64, 2, **i32), torch.zeros(1, 64, **i32), torch.zeros(1, 9, **i32),
                            torch.zeros(64 * (1 + (hd - 1) // 128), device="cuda"), 0, 1, 64, 1, 1, 8, 0)   # T5: hd's planes


def _gds_plan_replay(mod, plan_call):
    """The planner gx_gds_plan replayed on the op's OWN recorded plan geometry (the bank's shape, the record rows, tiles, dot
    planes and readings of make_gds's call, its ti_max, P, C, R, K and nw) with an f32 grad bank and zero contents: with kbf it
    must run the bf16 entry's host path (gx_gds_run<true>), which refuses that grad bank (its own message), with the op's own
    records and with 8-byte records; without kbf gx_gds's host path (gx_gds_run<false>) plans it (8-byte records + spay_q, as
    that path reads for an f32 grad bank). T5: with the run's own dtypes too (bf16 bank, bf16 grad bank), a same-shape
    non-contiguous bank must reach the bf16 entry's own refusal (gx_gds's host path refuses it with a different message), so a
    kbf selection that also depends on the grad's dtype is seen. Under plan capture nothing launches."""
    a, kw = plan_call
    assert len(a) == 15 and a[-1] is True and not kw, plan_call   # the bf16 run's plan: make_gds's 15 arguments, kbf True
    bank, grad, gr, gr_rows, rec, spay, tb, dots, ti_max, Pp, Cc, Rr, K, nw, _ = a
    i32 = dict(device=bank.device, dtype=torch.int32)
    rec8 = torch.zeros(rec.shape[0], rec.shape[1], 2, **i32)
    sp8 = torch.zeros(tb.shape[0], rec.shape[1], **i32)

    def call(r, s, kbf):
        return mod.gx_gds_plan(torch.zeros_like(bank), torch.zeros(bank.shape, device=bank.device), torch.zeros_like(gr),
                               gr_rows, r, s, torch.zeros_like(tb), torch.zeros_like(dots), ti_max, Pp, Cc, Rr, K, nw, kbf)
    for r, s in ((torch.zeros_like(rec), torch.zeros_like(spay)), (rec8, sp8)):
        with pytest.raises(RuntimeError, match=r"gx_gds \(bf16 entry\): bf16 bank and bf16 grad"):
            call(r, s, True)
    nc = torch.zeros(bank.shape[1], bank.shape[0], device=bank.device, dtype=torch.bfloat16).t()   # T5: non-contiguous
    assert nc.shape == bank.shape and not nc.is_contiguous() and bank.dtype == grad.dtype == torch.bfloat16, grad.dtype
    with pytest.raises(RuntimeError, match=r"gx_gds \(bf16 entry\): bf16 bank and bf16 grad"):
        mod.gx_gds_plan(nc, torch.zeros(bank.shape, device=bank.device, dtype=torch.bfloat16), torch.zeros_like(gr),
                        gr_rows, torch.zeros_like(rec), torch.zeros_like(spay), torch.zeros_like(tb), torch.zeros_like(dots),
                        ti_max, Pp, Cc, Rr, K, nw, True)
    plan = call(rec8, sp8, False)
    assert isinstance(plan, torch.Tensor) and plan.dtype == torch.uint8 and plan.numel() > 0, plan


def test_gx_gds_factory_routing_at_the_2_32_edge(monkeypatch):
    """make_gds's entry selection at the exact element-count edge of bf16 / bf16 rows (hd 128), observed on a real
    launch through the counting proxy: 2^25 - 1 rows (below 2^32 elements: gx_gds_bf), 2^25 rows (exactly 2^32: gx_gds,
    MODE 4, 8-byte records) and 2^25 + 1 rows (above: gx_gds); half of the slots in the bank's last rows."""
    from sdm_testlib import require_free_memory
    require_free_memory(40)
    _, _, _, gxm, *_ = _sdm()
    real = gxm._module()
    hd, P, Nt, C, R, pool = 128, 1, 1, 16, 4, 64
    for rows, want in ((2 ** 25 - 1, "gx_gds_bf"), (2 ** 25, "gx_gds"), (2 ** 25 + 1, "gx_gds")):
        p1, _, q_idx, _, _, _ = _phase1(P, Nt, C, 4, R, pool, rows, BF, 0, C - 3, hi=rows)
        N, n = Nt * P, C * R
        bank = torch.zeros(rows, hd, device="cuda", dtype=BF)
        grad = torch.zeros(rows, hd, device="cuda", dtype=BF)
        assert (bank.numel() < 2 ** 32) == (want == "gx_gds_bf")
        gr = torch.zeros(N * C, hd, device="cuda")
        dqw = torch.empty(N, C, R, device="cuda")
        proxy = _Proxy(real)
        monkeypatch.setattr(gxm, "_module", lambda p=proxy: p)
        launch = gxm.make_gds(bank, grad, gr, p1.spay_q, p1.qrec.reshape(N, n, 4), dqw, chunk_size=C,
                              num_parallel=P, steps=Nt, q_idx=q_idx, valid_tokens=C - 3, tiles_q=None)
        assert launch.geom["impl"] == "gx" and gxm.LAST_POLICY["w64"] == (want == "gx_gds"), dict(gxm.LAST_POLICY)
        launch(0)
        torch.cuda.synchronize()
        assert proxy.calls == {want: 1}, (rows, proxy.calls)
        del bank, grad, launch, p1
        torch.cuda.empty_cache()


def test_gx_gds_bf16_entry_2_32_edge():
    """2^32 elements are refused by the bf16 entry's bound; 2^32 - 4 pass it (and are refused by the next check: the
    float64 readings), so the bound sits exactly at 2^32."""
    free = torch.cuda.mem_get_info()[0]
    if free < 40 * 2 ** 30:
        pytest.skip("needs 2 x 2 x 8 GiB of free device memory")
    for rows, refused in ((2 ** 30, True), (2 ** 30 - 1, False)):
        bank = torch.empty(rows, 4, device="cuda", dtype=BF)
        grad = torch.empty(rows, 4, device="cuda", dtype=BF)
        with pytest.raises(RuntimeError) as ei:
            _gds_bf_call(bank, grad, 4, gr_dtype=torch.float64)
        msg = str(ei.value)
        assert ("2^32 elements or more" in msg) == refused and (refused or "grad_readings f32" in msg), msg[:200]
        del bank, grad
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------- op ----
OP_CASES = [(2, 256, 16, 16, 128, 2048, 64), (1, 384, 8, 8, 960, 4096, 192)]


def _assert_gds_entry(proxy, entry, steps):
    """The backward's GDS used ONLY `entry` ('bf16': gx_gds_bf, 'generic': gx_gds) for all its steps: the C++ step loop
    (planned once with that entry - gx_gds_plan's last argument is True exactly for the bf16 entry - then launched from
    gx_gds_launch_addr, nothing else called) or the per-step Python loop (that entry called once per step, the other never)."""
    if "gx_gds_plan" in proxy.calls:
        from sparse_delta_memory.backward import LAST_LOOP
        assert proxy.calls == {"gx_gds_plan": 1, "gx_gds_launch_addr": 1}, proxy.calls
        assert proxy.plan_args and all(a[0][-1] is (entry == "bf16") for a in proxy.plan_args), proxy.plan_args
        assert LAST_LOOP["name"] == "eager_cpp_step_loop", LAST_LOOP
    else:
        want, other = ("gx_gds_bf", "gx_gds") if entry == "bf16" else ("gx_gds", "gx_gds_bf")
        assert proxy.calls.get(want, 0) == steps and other not in proxy.calls, proxy.calls


@pytest.mark.parametrize("P,T,W,R,hd,sph,C", OP_CASES, ids=[f"C{c[6]}_hd{c[4]}" for c in OP_CASES])
def test_op_bits_with_the_bf16_entries(P, T, W, R, hd, sph, C, monkeypatch):
    """The op (forward + backward): the backward asks for gx_pa2's entry with its bank, grad bank and snapshot rows and
    takes the bf16 entries every step (Phase A and GDS); forced onto the generic entries, the same bits."""
    _, _, _, gxm, gxpa2, _, _ = _sdm()
    probe = torch.empty(1, hd, device="cuda", dtype=BF)
    if gxpa2.plan(probe, probe, C) is None or not gxm.supported(probe, probe, C, R):
        pytest.skip(f"C {C} hd {hd}: not admitted on this backend")
    d = make_op_inputs(P, T, W, R, hd, sph, BF, seed=3)
    real_launcher, real_gds = gxpa2.launcher, gxm._module()
    # make_gds's plan tells the GDS entries apart by identity (its last argument `gfn is not mod.gx_gds`), so the REAL
    # module's bf16 entry must be its own object (the counting proxies below hand out one wrapper per name: they would hide
    # an alias); and the planner must run the entry it is asked for (replayed on the op's own plan after the bf16 run below).
    assert real_gds.gx_gds_bf is not real_gds.gx_gds, "the module's bf16 GDS entry is its generic entry"
    # ... and the REAL module's bf16 entry is the bf16 entry (both packages): it refuses a bf16 bank with an f32 grad bank
    # (its own message), which the generic entry takes - so entries swapped under their names are seen on either path
    with pytest.raises(RuntimeError, match=r"gx_gds \(bf16 entry\): bf16 bank and bf16 grad"):   # T5: at this case's own hd
        _gds_bf_call(torch.zeros(64, hd, device="cuda", dtype=BF), torch.zeros(64, hd, device="cuda"), hd)
    seen = []

    def spy(pdl=False, bank=None, grad=None, snap=None):
        fn = real_launcher(pdl=pdl, bank=bank, grad=grad, snap=snap)
        seen.append(fn)
        return fn
    proxy = _Proxy(real_gds)
    monkeypatch.setattr(gxpa2, "launcher", spy)
    monkeypatch.setattr(gxm, "_module", lambda: proxy)
    fast = _op_run(d, C=C, sph=sph, P=P)
    m = gxpa2._module()
    assert len(seen) == 1 and seen[0] in (m.gx_pa2_bf, m.gx_pa2_bf_pdl), seen
    # The C++ backward step loop: the steps' GDS launches no longer go through the Python module - the launcher is
    # PLANNED once (gx_gds_plan) and the C++ loop launches from its address (gx_gds_launch_addr). The property is unchanged:
    # a bf16 bank uses ONLY the bf16 GDS entry, never the generic one (planned: the plan is made with the bf16 entry, i.e.
    # make_gds's last argument `gfn is not mod.gx_gds` is True; unplanned, the per-step Python loop: as before); forced onto
    # the generic entry (below) it uses ONLY the generic one (planned: last argument False) and writes the same bits.
    _assert_gds_entry(proxy, "bf16", T // C)
    if "gx_gds_plan" in proxy.calls:   # the planned path: its plan's own geometry, replayed (both entries' host paths)
        _gds_plan_replay(real_gds, proxy.plan_args[0])
    generic_proxy = _Proxy(real_gds, gx_gds_bf="gx_gds")     # the bf16 entry IS the generic one (alias)
    monkeypatch.setattr(gxpa2, "launcher", lambda pdl=False, bank=None, grad=None, snap=None:
                        real_launcher(pdl=pdl, bank=bank))
    monkeypatch.setattr(gxm, "_module", lambda: generic_proxy)
    generic = _op_run(d, C=C, sph=sph, P=P)
    _assert_gds_entry(generic_proxy, "generic", T // C)
    for k in fast:
        assert not bits_mismatch(generic[k], fast[k]), k
    assert set(fast) >= {"readings", "bank_bwd"} | {"grad_" + n for n in OP_INPUTS}
