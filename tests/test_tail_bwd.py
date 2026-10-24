# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Native tails, the backward side."""
import os

import pytest
import torch

from sdm_testlib import OP_GRADS, OP_INPUTS, bits_mismatch, make_op_inputs, requires_cuda
from test_mover_row_types import _assert_same, _pa_pair, _phase1, _rows, _sdm
from test_fx_tails import _backend, _chunk_ok, _oracle_gate

pytestmark = [requires_cuda]
BF, F32 = torch.bfloat16, torch.float32


@pytest.fixture
def tails_on(monkeypatch):
    """The shared tail predicate forced on for unquantized calls (kernel-level tests: these kernels alone)."""
    from sparse_delta_memory import movers
    monkeypatch.setattr(movers, "tail_admits", lambda bank, chunk_size, quant="none": quant == "none")
    return monkeypatch


# ------------------------------------------------------------------------------------------------ gx_pa2 PB 2 ----
def _pa_elem_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, seed=0):
    """(dc_pa3 on the head zero-padded to hd4 = 4 ceil(hd / 4) columns, gx_pa2 on the hd % 4 != 0 head)."""
    bw, _, _, _, gxpa2, _, dc = _sdm()
    hd4 = -(-hd // 4) * 4
    sph = 4 * pool
    p1, k_idx, _, k_val, _, beta = _phase1(P, Nt, C, W, 4, pool, sph, bank_dtype, seed, C - 5, sph)
    N, n = Nt * P, C * W
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    bpre = p1.dc_bpre.reshape(N, -1)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    rows = P * sph
    snap_t = _rows(N * n, hd, bank_dtype, gen)
    dva_t = torch.randn(N * C, hd, device="cuda", generator=gen)
    bank_t = _rows(rows, hd, bank_dtype, gen)
    grad_t = _rows(rows, hd, grad_dtype, gen, 0.01)

    def pad(t):
        out = torch.zeros(t.shape[0], hd4, device="cuda", dtype=t.dtype)
        out[:, :hd] = t
        return out
    nsl = dc.slices(hd)
    assert nsl == dc.slices(hd4)
    _, depth, nw = bw.pa_plan(C, bank_dtype, grad_dtype, bank_t.device)
    empty = torch.empty(0, device="cuda", dtype=grad_dtype)
    beta_f = beta.reshape(-1).contiguous()
    outs = []
    for use_gx, (snap, dva, bank0, grad0) in ((False, (pad(snap_t), pad(dva_t), pad(bank_t), pad(grad_t))),
                                              (True, (snap_t, dva_t, bank_t, grad_t))):
        grad, bank = grad0.clone(), bank0.clone()
        eff = torch.full((nsl, N * n), float("nan"), device="cuda")
        gg = torch.full((nsl, N * n), float("nan"), device="cuda")
        h = bank.shape[-1]
        L = dc.tile_len(C, W, h, P)
        nw_k, depth_k = nw, depth
        if use_gx:
            assert gxpa2.supported(grad, bank, snap, C, 1792, depth, nw, False)
            assert gxpa2.needs_tail_module(bank)
            fn = gxpa2.launcher(pdl=False, bank=bank)
            nw_k, depth_k = gxpa2.plan(grad, bank, C)
            L = gxpa2.tile_len(C, W, h, P, L, device=bank.device)
        else:
            fn = dc._module().dc_pa
        for ti in reversed(range(Nt)):
            fn(grad, bank, snap, P * C * W, dva, beta_f, kpay, rec, eff.view(-1) if nsl == 1 else eff,
               gg.view(-1) if nsl == 1 else gg, ti, P, C, W, L, nw_k, depth_k, 1792, True, bpre, empty)
        torch.cuda.synchronize()
        outs.append({"grad": grad[:, :hd].contiguous(), "bank": bank[:, :hd].contiguous(), "eff": eff, "g": gg})
    return outs


ELEM_PA_CASES = [   # (bank dtype, grad dtype, P, Nt, C, W, hd, pool): hd % 4 != 0, element pieces
    (BF, BF, 2, 3, 64, 8, 63, 48),
    (BF, BF, 2, 2, 128, 16, 1, 96),        # one column
    (BF, BF, 2, 2, 64, 8, 129, 64),        # slices 68 / 61
    (BF, BF, 1, 2, 192, 8, 266, 64),       # 16 warps per CTA, slices 92 / 92 / 82
    (BF, BF, 2, 2, 64, 8, 321, 64),        # the hd-320 class's odd neighbour: slices 108 / 108 / 105
    (BF, F32, 2, 3, 64, 8, 66, 48),
    (BF, F32, 2, 2, 32, 4, 1001, 32),      # slices 7 x 128 + 105
    (F32, F32, 2, 3, 64, 8, 3, 48),
    (F32, F32, 2, 2, 128, 16, 130, 96),
    (F32, F32, 2, 2, 32, 4, 1001, 32),
    (BF, BF, 2, 2, 64, 4, 16387, 16),      # 128 slices of 128 + a 3-column last slice
    (F32, F32, 2, 2, 64, 4, 16385, 16),
]


@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,W,hd,pool", ELEM_PA_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_W{c[5]}_hd{c[6]}" for c in ELEM_PA_CASES])
def test_gx_pa2_elem_match_padded_dc_pa3(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, tails_on):
    _chunk_ok(C)
    ref, got = _pa_elem_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool)
    _assert_same(ref, got, "gx_pa2 element pieces vs dc_pa3 on the zero-padded head")
    assert not ref["grad"].isnan().any() and not ref["eff"].isnan().any()


# ------------------------------------------------------------------------------------- the chain under PDL ----
def _gx_elem_runs(bank_dtype, grad_dtype, P, Nt, C, W, R, hd, pool, seed=0):
    """The backward's step chain on one hd % 4 != 0 problem, on SHARED bank / grad tensors."""
    bw, _, _, gxm, gxpa2, _, dc = _sdm()
    from sparse_delta_memory.cuda import gx_gemm_cuda as gxgemm
    sph = 4 * pool
    p1, k_idx, q_idx, k_val, _, beta = _phase1(P, Nt, C, W, R, pool, sph, bank_dtype, seed, C - 5, sph)
    N, n, nq = Nt * P, C * W, C * R
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    bpre = p1.dc_bpre.reshape(N, -1)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    rows = P * sph
    snap = _rows(N * n, hd, bank_dtype, gen)
    du0 = torch.randn(N, C, hd, device="cuda", generator=gen)
    x = torch.randn(N, C, C, device="cuda", generator=gen) * 0.1
    ddv = torch.randn(N * C, hd, device="cuda", generator=gen)
    bank0 = _rows(rows, hd, bank_dtype, gen)
    grad0 = _rows(rows, hd, grad_dtype, gen, 0.01)
    gr = torch.randn(N * C, hd, device="cuda", generator=gen)
    nsl = dc.slices(hd)
    _, depth, nw = bw.pa_plan(C, bank_dtype, grad_dtype, bank0.device)
    empty = torch.empty(0, device="cuda", dtype=grad_dtype)
    beta_f = beta.reshape(-1).contiguous()
    q8 = p1.qrec8.reshape(N, nq, 2)
    outs = []
    for mode in ("plain", "pdl", "graph"):
        pdl = mode != "plain"
        grad, bank, du = grad0.clone(), bank0.clone(), du0.clone()
        eff = torch.full((nsl, N * n), float("nan"), device="cuda")
        gg = torch.full((nsl, N * n), float("nan"), device="cuda")
        dqw = torch.full((N, C, R), float("nan"), device="cuda")
        assert gxgemm.supported(du, x, ddv, C)
        assert gxpa2.supported(grad, bank, snap, C, 1792, depth, nw, False) and gxm.supported(bank, grad, C, R)
        gemm = gxgemm.make_step(du, x, ddv, P)
        assert gemm.pdl_capable
        pa = gxpa2.launcher(pdl=pdl, bank=bank)
        nw_k, depth_k = gxpa2.plan(grad, bank, C)
        L = gxpa2.tile_len(C, W, hd, P, dc.tile_len(C, W, hd, P), device=bank.device)
        gds = gxm.make_gds(bank, grad, gr, p1.spay_q, q8, dqw, chunk_size=C, num_parallel=P, steps=Nt,
                           q_idx=q_idx, valid_tokens=(Nt - 1) * C + C - 5, tiles_q=None)
        assert gds.geom["impl"] == "gx"
        du_rows = du.view(N * C, hd)

        def steps():
            for ti in reversed(range(Nt)):
                gemm(ti, pdl=pdl)
                pa(grad, bank, snap, P * C * W, du_rows, beta_f, kpay, rec, eff.view(-1) if nsl == 1 else eff,
                   gg.view(-1) if nsl == 1 else gg, ti, P, C, W, L, nw_k, depth_k, 1792, True, bpre, empty)
                gds(ti, pdl=pdl)
        if mode == "graph":
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=side):
                    steps()
            torch.cuda.current_stream().wait_stream(side)
            g.replay()
        else:
            steps()
        torch.cuda.synchronize()
        outs.append({"grad": grad, "bank": bank, "eff": eff, "g": gg, "d_u": du, "d_q_w": dqw})
    return outs


@pytest.mark.parametrize("bank_dtype,grad_dtype,hd", [(BF, BF, 63), (BF, F32, 129), (F32, F32, 3), (BF, BF, 266)],
                         ids=["bf16_bf16_hd63", "bf16_f32_hd129", "f32_f32_hd3", "bf16_bf16_hd266"])
def test_elem_pdl_and_graph_replay(bank_dtype, grad_dtype, hd, tails_on):
    """Element pieces in the backward's PDL chain."""
    if _backend() != "modern":
        pytest.skip("programmatic dependent launch: the modern backend only")
    plain, pdl, graph = _gx_elem_runs(bank_dtype, grad_dtype, 2, 3, 64, 8, 8, hd, 32)
    _assert_same(plain, pdl, "element pieces: PDL chain vs plain")
    _assert_same(plain, graph, "element pieces: CUDA-graph replay vs plain")
    assert not plain["d_q_w"].isnan().any() and not plain["eff"].isnan().any()


# --------------------------------------------------------------------------------------------- gx_gds tails ----
def _gds_tail_pair(bank_dtype, grad_dtype, P, Nt, C, R, hd, hd_pad, pool, rec8, seed=0):
    """(gx_gds on the head padded with zero columns to hd_pad = equal slices of the tail's slice width, gx_gds on the
    tail head): the same records, scales and token / bank / grad values in the first hd columns. Half of each stream's
    slots in its last `pool` rows (the last stream's: the bank's final rows, where a row slice could reach past the
    tensor)."""
    _, _, _, gxm, *_ = _sdm()
    vl = C - 3
    p1, _, q_idx, _, _, _ = _phase1(P, Nt, C, 4, R, pool, 4 * pool, bank_dtype, seed, vl, hi=4 * pool)
    N, n = Nt * P, C * R
    gen = torch.Generator(device="cuda").manual_seed(seed + 2)
    rows = P * 4 * pool
    bank_t = _rows(rows, hd, bank_dtype, gen)
    grad_t = _rows(rows, hd, grad_dtype, gen, 0.01)
    gr_t = torch.randn(N * C, hd, device="cuda", generator=gen)

    def pad(t):
        out = torch.zeros(t.shape[0], hd_pad, device="cuda", dtype=t.dtype)
        out[:, :hd] = t
        return out
    q16 = p1.qrec.reshape(N, n, 4)
    q8 = p1.qrec8.reshape(N, n, 2)
    outs = []
    route = getattr(gxm, "h100_f32_to_da2", None)
    for k, (bank, grad0, gr) in enumerate(((pad(bank_t), pad(grad_t), pad(gr_t)), (bank_t, grad_t, gr_t))):
        # The zero-padded REGULAR reference (k == 0) must run on gx_gds itself; on H100, h100_f32_to_da2 sends float32 grad
        # rows of regular heads to mover_da2's GDS by design, so it is switched off for the reference only (the tail runs as
        # shipped).
        if k == 0 and route is not None:
            gxm.h100_f32_to_da2 = lambda *a, **kw: False
        try:
            grad = grad0.clone()
            dqw = torch.full((N, C, R), float("nan"), device="cuda")
            assert gxm.supported(bank, grad, C, R)
            launch = gxm.make_gds(bank, grad, gr, p1.spay_q, q8 if rec8 else q16, dqw, chunk_size=C, num_parallel=P,
                                  steps=Nt, q_idx=q_idx, valid_tokens=(Nt - 1) * C + vl, tiles_q=None)
            assert launch.geom["impl"] == "gx"
            for ti in reversed(range(Nt)):
                launch(ti)
            torch.cuda.synchronize()
        finally:
            if k == 0 and route is not None:
                gxm.h100_f32_to_da2 = route
        outs.append({"grad": grad[:, :hd].contiguous(), "d_q_w": dqw})
    return outs


TAIL_GDS_CASES = [   # (bank dtype, grad dtype, P, Nt, C, R, hd, hd_pad, pool, rec8)
    (BF, BF, 2, 3, 64, 8, 320, 324, 48, False),        # slices 108 / 108 / 104
    (BF, BF, 2, 2, 128, 16, 320, 324, 48, True),       # the hd-320 class at C 128
    (BF, BF, 1, 2, 256, 16, 320, 324, 32, True),       # ... and at C 256
    (BF, BF, 2, 2, 128, 16, 260, 264, 96, True),       # slices 88 / 88 / 84
    (BF, F32, 2, 2, 64, 8, 1000, 1024, 64, True),      # slices 7 x 128 + 104
    (F32, F32, 2, 3, 64, 8, 132, 136, 48, True),       # slices 68 / 64
    (F32, F32, 2, 2, 32, 4, 320, 324, 32, True),
    (BF, F32, 2, 2, 64, 8, 260, 264, 48, True),        # f32 grad rows, slices 88 / 88 / 84
    (BF, F32, 2, 2, 32, 4, 320, 324, 32, True),
    (BF, BF, 2, 2, 64, 4, 16388, 16512, 16, False),    # 128 slices of 128 + 4 columns
    (F32, F32, 2, 2, 64, 4, 16388, 16512, 16, True),
    # hd % 4 != 0 (MODE bit 16, element pieces) vs the regular head padded to equal slices of the same width
    (BF, BF, 2, 3, 64, 8, 63, 64, 48, False),
    (BF, BF, 2, 2, 64, 8, 3, 4, 48, True),
    (BF, BF, 2, 2, 128, 16, 129, 136, 96, True),       # slices 68 / 61
    (BF, BF, 2, 2, 64, 8, 321, 324, 48, True),         # slices 108 / 108 / 105
    (BF, F32, 2, 2, 64, 8, 66, 68, 48, True),
    (BF, F32, 2, 2, 32, 4, 1001, 1024, 32, True),      # slices 7 x 128 + 105
    (F32, F32, 2, 3, 64, 8, 130, 136, 48, True),
    (F32, F32, 2, 2, 32, 4, 1, 4, 32, True),
    (BF, BF, 2, 2, 64, 4, 16387, 16512, 16, False),    # 128 slices of 128 + 3 columns
    (F32, F32, 2, 2, 64, 4, 16385, 16512, 16, True),
]


@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,R,hd,hd_pad,pool,rec8", TAIL_GDS_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_R{c[5]}_hd{c[6]}" for c in TAIL_GDS_CASES])
def test_gx_gds_tails_match_padded_regular(bank_dtype, grad_dtype, P, Nt, C, R, hd, hd_pad, pool, rec8, tails_on):
    _chunk_ok(C)
    _, _, _, gxm, *_ = _sdm()
    nsl, sw, tail = gxm.slices(hd)
    nsl_p, sw_p, tail_p = gxm.slices(hd_pad)
    assert tail and not tail_p and (nsl, sw) == (nsl_p, sw_p), (hd, hd_pad, nsl, sw, nsl_p, sw_p)
    ref, got = _gds_tail_pair(bank_dtype, grad_dtype, P, Nt, C, R, hd, hd_pad, pool, rec8)
    _assert_same(ref, got, "gx_gds tail vs the zero-padded regular head")
    assert not ref["d_q_w"].isnan().any()


def _train(d, C, sph, P, G, gbf32=False, quant="none"):
    """Saving forward + backward of the public op on private leaves; (readings, {grad name: tensor})."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    kw = {"grad_bank_fp32": True} if gbf32 else {}
    if quant != "none":
        kw["snapshot_quant"] = quant
    y = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                       d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph, num_parallel=P, **kw)
    grads = torch.autograd.grad(y, [leaves[n] for n in OP_INPUTS], grad_outputs=G.reshape(y.shape).to(y.dtype))
    torch.cuda.synchronize()
    return y.detach(), dict(zip(OP_GRADS, grads))


class _Route:
    """Route counters of one call: forward DW (movers), backward."""
    def __init__(self):
        from sparse_delta_memory import backward as bw, movers
        self.bw, self.mv = bw, movers
        self.b0, self.m0 = dict(bw.STATS), dict(movers.STATS)

    def delta(self):
        d = {f"bw.{k}": v - self.b0.get(k, 0) for k, v in self.bw.STATS.items() if v != self.b0.get(k, 0)}
        d.update({f"mv.{k}": v - self.m0.get(k, 0) for k, v in self.mv.STATS.items() if v != self.m0.get(k, 0)})
        return d


# (P, T, W, R, hd, sph, C, dtype, gbf32): the shipped hd-320 class (nsl 3 uneven slices, bf16, C 128 / 256, W / R 128 /
# 64; P and slots cut so the float64 oracle stays cheap) and one case per other tail class, f32 grad rows included
E2E = [
    (2, 2 * 128 + 37, 128, 128, 320, 4096, 128, BF, False),
    (2, 2 * 256 + 37, 128, 128, 320, 4096, 256, BF, False),
    (2, 2 * 256 + 37, 64, 64, 320, 4096, 256, BF, False),
    (2, 2 * 128 + 37, 128, 128, 320, 4096, 128, BF, True),
    (2, 2 * 64 + 37, 8, 8, 136, 2048, 64, BF, False),          # (a) equal slices, rows not 16-byte multiples
    (2, 2 * 256 + 37, 8, 8, 36, 2048, 256, BF, True),
    (2, 2 * 128 + 37, 8, 8, 1028, 2048, 128, BF, False),       # (b) uneven slices
    (2, 2 * 64 + 37, 8, 8, 132, 2048, 64, F32, False),
    (2, 2 * 64 + 37, 8, 8, 129, 2048, 64, BF, False),          # (c) hd % 4 != 0
    (2, 2 * 64 + 37, 8, 8, 129, 2048, 64, BF, True),
    (2, 2 * 128 + 37, 8, 8, 961, 2048, 128, BF, False),
    (2, 2 * 256 + 37, 8, 8, 63, 2048, 256, F32, False),
    (2, 2 * 64 + 37, 8, 8, 1, 2048, 64, BF, False),
]
E2E_IDS = [f"hd{e[4]}_{str(e[7])[6:]}{'_gbf32' if e[8] else ''}_C{e[6]}_W{e[2]}" for e in E2E]


@pytest.mark.parametrize("P,T,W,R,hd,sph,C,dtype,gbf32", E2E, ids=E2E_IDS)
def test_backward_tails_native_within_the_oracle_gates(P, T, W, R, hd, sph, C, dtype, gbf32, monkeypatch, ieee_fp32):
    """A tail head's training call takes the native route in both directions."""
    b = _backend()
    _chunk_ok(C)
    import sdm_testlib as tl
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import gx_gds_cuda as gxm
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=31, slots="trained", weights="softmax")
    gen = torch.Generator(device="cuda").manual_seed(32)
    # the cotangent in the readings' dtype, as the op receives it, and the oracle's VJP of exactly that cotangent
    G = (torch.randn(P, T, hd, device="cuda", generator=gen) * 0.01).to(dtype)
    ref_y, ref_g, _ = tl.oracle_write_read(d, G, device="cuda", sph=sph)
    runs = []
    for rep in range(2):
        r = _Route()
        y, gr = _train(d, C, sph, P, G, gbf32)
        dl = r.delta()
        assert dl.get("bw.dc", 0) == 1 and not dl.get("bw.shipped", 0), dl             # the CUDA backward
        assert dl.get("bw.gx_pa2", 0) == 1, dl                                          # Phase A in gx_pa2
        assert not dl.get("mv.shipped_dw", 0) and not dl.get("mv.shipped_gds", 0), dl   # no phase2_kernels mover
        assert gxm.LAST_POLICY.get("impl") == "gx", gxm.LAST_POLICY                     # gx_gds (tail module)
        runs.append((y, gr))
    (y1, g1), (y2, g2) = runs
    assert not bits_mismatch(y1, y2), "readings: repeats differ"
    for k in g1:
        assert not bits_mismatch(g1[k], g2[k]), f"{k}: repeats differ"
    from sparse_delta_memory.memory_ops import UnsupportedConfigurationError
    monkeypatch.setattr(movers, "tail_admits", lambda *a, **k: False)                  # no tail predicate
    r = _Route()
    with pytest.raises(UnsupportedConfigurationError) as err:
        _train(d, C, sph, P, G, gbf32)
    assert err.value.tail_class == "P" and not r.delta(), (err.value.tail_class, r.delta())   # before any work
    report = {"readings": _oracle_gate(y1.view(ref_y.shape), ref_y, dtype)}
    for k in g1:
        report[k] = _oracle_gate(g1[k].reshape(ref_g[k].shape), ref_g[k], dtype)
    print(f"BWD_TAIL {b} hd {hd} {dtype} gbf32 {gbf32} C {C} W {W}: float64-oracle gates " +
          " ".join(f"{k} {v[1]}" for k, v in report.items()))
    bad = {k: v[1] for k, v in report.items() if not v[0]}
    assert not bad, bad


@pytest.mark.parametrize("quant", ["fp8", "int4"])
def test_quantized_tails_are_refused(quant):
    """fp8 / int4 calls of a tail head."""
    _backend()
    from sdm_testlib import assert_quant_tail_refused
    P, T, W, R, hd, sph, C = 2, 2 * 64 + 37, 8, 8, 320, 2048, 64
    d = make_op_inputs(P, T, W, R, hd, sph, BF, seed=33, slots="zipf")
    G = (torch.randn(P, T, hd, device="cuda", generator=torch.Generator(device="cuda").manual_seed(34)) * 0.01).to(BF)
    r = _Route()
    assert_quant_tail_refused(lambda: _train(d, C, sph, P, G, quant=quant), "T2")
    assert not any(r.delta().values()), r.delta()


# ----------------------------------------------------------------------------- the portable side on real hardware ----
# test_backends.py's method: the tail modules built as compute_80 PTX only (8.0+PTX; a modern GPU JIT-compiles the PTX,
# so every SDM_PORTABLE_ARCH branch runs: sdmcp's staging, no bulk copy, no griddepcontrol) against the native builds,
# under the A100 / RTX rows' host decisions (plans, chunk limits, no PDL).
MODERN = ((9, 0), (10, 0), (10, 3))
A100_ROW = ((8, 0), 166912, 108, 41943040)
RTX_ROW = ((8, 6), 101376, 82, 6291456)
_TV = {}


def _tail_variant(module, arch_list):
    """`module`'s tail extension built for TORCH_CUDA_ARCH_LIST=`arch_list` as its own module."""
    from test_backends import _images
    key = (module.__name__, arch_list)
    if key not in _TV:
        from sparse_delta_memory.cuda import _image
        with _image._unchecked_build_env():   # TEST ONLY: a build for another target list (_image's frozen environment)
            prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
            try:
                _TV[key] = module._tail_module.__wrapped__()
            finally:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
        im = _images(_TV[key].__file__)
        cc = torch.cuda.get_device_capability()
        want = ({"sm_80"}, {"sm_80"}) if arch_list == "8.0+PTX" else ({f"sm_{cc[0]}{cc[1]}"}, set())
        assert im == want, (_TV[key].__file__, im, want)
    return _TV[key]


def _force_row(monkeypatch, row):
    from sparse_delta_memory.cuda import _device as d
    cc, optin, sms, l2 = A100_ROW if row == "a100" else RTX_ROW
    monkeypatch.setattr(d, "_FORCED", (tuple(cc), int(optin), sms, l2))
    assert d.backend() == ("a100" if row == "a100" else "rtx")


requires_modern = pytest.mark.skipif(not torch.cuda.is_available() or
                                     tuple(torch.cuda.get_device_capability()) not in MODERN,
                                     reason="needs an H100 / B200 / GB300 (native = modern side)")

PORT_PA_CASES = [   # (row, bank dtype, grad dtype, P, Nt, C, W, hd, pool, elem): the row's plan for each
    ("a100", BF, BF, 2, 2, 128, 8, 320, 64, False),     # the hd-320 class, PB 8
    ("a100", BF, F32, 2, 2, 256, 4, 100, 64, False),
    ("a100", F32, F32, 2, 3, 64, 8, 3, 48, True),        # element pieces, f32 rows
    ("rtx", BF, BF, 2, 2, 128, 8, 12, 48, False),
    ("a100", BF, BF, 2, 3, 64, 8, 63, 48, True),         # element pieces
    ("a100", BF, F32, 2, 2, 128, 8, 321, 64, True),
    ("rtx", F32, F32, 2, 2, 96, 8, 129, 48, True),
    ("rtx", BF, BF, 2, 2, 128, 16, 1, 96, True),
]


@requires_modern
@pytest.mark.parametrize("row,bank_dtype,grad_dtype,P,Nt,C,W,hd,pool,elem", PORT_PA_CASES,
                         ids=[f"{c[0]}_{str(c[1])[6:]}_{str(c[2])[6:]}_C{c[5]}_hd{c[7]}" for c in PORT_PA_CASES])
def test_portable_gx_pa2_tails(row, bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, elem, monkeypatch):
    """gx_pa2's tail module built for the portable side, under the row's plan: the bits of dc_pa3."""
    from sparse_delta_memory.cuda import gx_pa2_cuda as gxpa2
    pa80 = _tail_variant(gxpa2, "8.0+PTX")
    monkeypatch.setattr(gxpa2, "_tail_module", lambda: pa80)
    _force_row(monkeypatch, row)
    from sparse_delta_memory import movers
    monkeypatch.setattr(movers, "tail_admits", lambda bank, chunk_size, quant="none": quant == "none")
    if elem:
        ref, got = _pa_elem_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, seed=7)
    else:
        ref, got = _pa_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, sph=4 * pool, hi=4 * pool, seed=7)
    _assert_same(ref, got, f"gx_pa2 portable tail ({row}) vs dc_pa3")


PORT_GDS_CASES = [   # (row, bank dtype, grad dtype, P, Nt, C, R, hd, pool, rec8)
    ("a100", BF, BF, 2, 3, 128, 8, 320, 48, False),      # MODE 8, 32-bit offsets
    ("a100", BF, F32, 2, 2, 256, 4, 1000, 32, True),     # MODE 12
    ("rtx", F32, F32, 2, 2, 64, 8, 132, 48, True),
    ("a100", BF, BF, 2, 3, 64, 8, 63, 48, False),        # MODE 24
    ("rtx", BF, F32, 2, 2, 128, 8, 129, 48, True),       # MODE 28
    ("a100", F32, F32, 2, 2, 64, 8, 1, 32, True),
]


@requires_modern
@pytest.mark.parametrize("row,bank_dtype,grad_dtype,P,Nt,C,R,hd,pool,rec8", PORT_GDS_CASES,
                         ids=[f"{c[0]}_{str(c[1])[6:]}_{str(c[2])[6:]}_C{c[5]}_hd{c[7]}" for c in PORT_GDS_CASES])
def test_portable_gx_gds_tails(row, bank_dtype, grad_dtype, P, Nt, C, R, hd, pool, rec8, monkeypatch):
    """gx_gds's tail module built for the portable side."""
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import gx_gds_cuda as gxm
    monkeypatch.setattr(movers, "tail_admits", lambda bank, chunk_size, quant="none": quant == "none")
    cc = torch.cuda.get_device_capability()
    native, g80 = _tail_variant(gxm, f"{cc[0]}.{cc[1]}"), _tail_variant(gxm, "8.0+PTX")
    vl = C - 3
    p1, _, q_idx, _, _, _ = _phase1(P, Nt, C, 4, R, pool, 4 * pool, bank_dtype, 9, vl, hi=4 * pool)
    _force_row(monkeypatch, row)
    N, n = Nt * P, C * R
    gen = torch.Generator(device="cuda").manual_seed(10)
    bank = _rows(P * 4 * pool, hd, bank_dtype, gen)
    grad0 = _rows(P * 4 * pool, hd, grad_dtype, gen, 0.01)
    gr = torch.randn(N * C, hd, device="cuda", generator=gen)
    recs = p1.qrec8.reshape(N, n, 2) if rec8 else p1.qrec.reshape(N, n, 4)
    assert gxm.slices(hd)[2] and gxm.supported(bank, grad0, C, R)
    outs = []
    for mod in (native, g80):
        monkeypatch.setattr(gxm, "_tail_module", lambda m=mod: m)
        grad = grad0.clone()
        dqw = torch.full((N, C, R), float("nan"), device="cuda")
        launch = gxm.make_gds(bank, grad, gr, p1.spay_q, recs, dqw, chunk_size=C, num_parallel=P, steps=Nt,
                              q_idx=q_idx, valid_tokens=(Nt - 1) * C + vl, tiles_q=None)
        assert launch.geom["impl"] == "gx" and not launch.geom["pdl_capable"]
        for ti in reversed(range(Nt)):
            launch(ti)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "d_q_w": dqw})
    _assert_same(outs[0], outs[1], f"gx_gds portable tail ({row}) vs native")
    assert not outs[0]["d_q_w"].isnan().any()
