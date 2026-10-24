# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The movers' row types (float32 rows, 64-bit offsets) against the kernels they replace, on the same inputs."""

import pytest
import torch

from sdm_testlib import OP_INPUTS, bits_mismatch, make_op_inputs, requires_cuda

# the fast kernels under test (gx_pa2, gx_gds, mover_fx) are sm_90+ kernels
pytestmark = [requires_cuda, pytest.mark.skipif(not torch.cuda.is_available() or
                                                torch.cuda.get_device_capability()[0] < 9, reason="needs sm_90+")]


def _sdm():
    from sparse_delta_memory import backward, forward, memory_ops
    from sparse_delta_memory.cuda import gx_gds_cuda, gx_pa2_cuda, mover_da2_cuda, pa_dc_cuda
    return backward, forward, memory_ops, gx_gds_cuda, gx_pa2_cuda, mover_da2_cuda, pa_dc_cuda


def _phase1(P, Nt, C, W, R, pool, sph, dtype, seed, valid_last, hi=0):
    """Phase 1 of Nt steps of P chunks (time-major: chunk c is stream c % P)."""
    _, fw, mo, *_ = _sdm()
    from sparse_delta_memory.cuda import phase1_cuda as ph
    gen = torch.Generator(device="cuda").manual_seed(seed)
    N = Nt * P
    stream = (torch.arange(N, device="cuda") % P).view(N, 1, 1) * sph

    def slots(width):
        if hi:
            lo = torch.rand(N * C, pool, device="cuda", generator=gen).topk(width // 2, -1).indices
            up = torch.rand(N * C, pool, device="cuda", generator=gen).topk(width - width // 2, -1).indices
            ids = torch.cat([lo, up + (hi - pool)], -1)
        else:
            ids = torch.rand(N * C, pool, device="cuda", generator=gen).topk(width, -1).indices
        return ids.sort(-1).values.view(N, C, width) + stream

    k_idx, q_idx = slots(W), slots(R)
    k_val = (torch.randn(N, C, W, device="cuda", generator=gen) * 0.3).to(dtype)
    q_val = (torch.randn(N, C, R, device="cuda", generator=gen) * 0.3).to(dtype)
    g = torch.rand(N, C, 1, device="cuda", generator=gen) * -0.49 - 0.01
    beta = torch.rand(N, C, device="cuda", generator=gen) * 0.3
    fields = fw._SAVING_FIELDS | ph.DET_QREC8          # 16-byte qrec AND qrec8 (= its words 1-2), dc_bpre
    p1 = mo.phase1_forward(k_idx.int(), q_idx.int(), k_val, q_val, g, mo._repeat_counter(torch.device("cuda")),
                           num_parallel=P, stream_offset=sph, num_keys=sph, valid_last=valid_last,
                           need_read_groups=True, beta=beta, det_fields=fields)
    return p1, k_idx.int(), q_idx.int(), k_val, q_val, beta


def _rows(n, hd, dtype, gen, scale=0.1):
    out = torch.empty(n, hd, device="cuda", dtype=dtype)
    step = 1 << 20
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        out[r0:r1] = (torch.randn(r1 - r0, hd, device="cuda", generator=gen) * scale).to(dtype)
    return out


def _inject_special(t, stream_rows):
    """Signed zeros, subnormals and non-finite values in the first rows of every stream of t."""
    for r0 in range(0, t.shape[0], stream_rows):
        t[r0] = -0.0
        t[r0 + 1] = 1e-39
        t[r0 + 2, 0] = float("inf")
        t[r0 + 3, 1] = float("-inf")
        t[r0 + 4, 2] = float("nan")
    return t


def _pa_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, sph, seed=0, hi=0, valid_last=None, special=False,
             dbg=1792):
    """(dc_pa3 outputs, gx_pa2 outputs) over the reversed step loop on identical inputs."""
    bw, _, _, _, gxpa2, _, dc = _sdm()
    p1, k_idx, _, k_val, _, beta = _phase1(P, Nt, C, W, 4, pool, sph, bank_dtype, seed,
                                           valid_last or C - 5, hi)
    N, n = Nt * P, C * W
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    bpre = p1.dc_bpre.reshape(N, -1)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    rows = P * sph
    snap = _rows(N * n, hd, bank_dtype, gen)
    dva = torch.randn(N * C, hd, device="cuda", generator=gen)
    bank0 = _rows(rows, hd, bank_dtype, gen)
    grad0 = _rows(rows, hd, grad_dtype, gen, 0.01)
    if special:
        _inject_special(bank0, sph), _inject_special(grad0, sph), _inject_special(snap, n)
    nsl = dc.slices(hd)
    plan = bw.pa_plan(C, bank_dtype, grad_dtype, bank0.device)
    _, depth, nw = plan
    empty = torch.empty(0, device="cuda", dtype=grad_dtype)
    beta_f = beta.reshape(-1).contiguous()
    outs = []
    bpre_arg = bpre if dbg & 1024 else torch.empty(0, device="cuda", dtype=torch.int32)
    for use_gx in (False, True):
        grad, bank = grad0.clone(), bank0.clone()
        eff = torch.full((nsl, N * n), float("nan"), device="cuda")
        gg = torch.full((nsl, N * n), float("nan"), device="cuda")
        L = dc.tile_len(C, W, hd, P)
        w_, d_ = nw, depth
        if use_gx:
            assert gxpa2.supported(grad, bank, snap, C, dbg, depth, nw, False)
            fn = gxpa2.launcher(pdl=False, bank=bank)
            L = gxpa2.tile_len(C, W, hd, P, L, device=bank.device)
            w_, d_ = gxpa2.plan(grad, bank, C)     # its own warps per CTA and ring depth (as the backward passes them)
        else:
            fn = dc._module().dc_pa
        for ti in reversed(range(Nt)):
            fn(grad, bank, snap, P * C * W, dva, beta_f, kpay, rec, eff.view(-1) if nsl == 1 else eff,
               gg.view(-1) if nsl == 1 else gg, ti, P, C, W, L, w_, d_, dbg, True, bpre_arg, empty)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "bank": bank, "eff": eff, "g": gg})
        del grad, bank
    return outs


def _assert_same(a, b, what):
    for k in a:
        bad = bits_mismatch(a[k], b[k])
        assert not bad, f"{what}: {k} differs: {bad}"


PA_CASES = [   # (bank dtype, grad dtype, P, Nt, C, W, hd, pool): f32 rows at 1 / 8 / 2 slices, C 16 .. 192
    (torch.float32, torch.float32, 2, 3, 64, 8, 64, 48),
    (torch.float32, torch.float32, 2, 2, 128, 16, 128, 96),
    (torch.float32, torch.float32, 1, 2, 144, 8, 128, 64),      # 16 warps per CTA (C > 128, fits)
    (torch.float32, torch.float32, 2, 2, 64, 8, 100, 64),       # 100-column slice (f32 16-byte rows only)
    (torch.float32, torch.float32, 2, 2, 32, 4, 960, 32),       # 8 slices
    (torch.float32, torch.float32, 2, 2, 16, 4, 256, 16),
    (torch.bfloat16, torch.float32, 2, 3, 64, 8, 64, 48),
    (torch.bfloat16, torch.float32, 2, 2, 128, 16, 1024, 96),
    (torch.bfloat16, torch.float32, 1, 2, 192, 8, 128, 64),     # 16 warps per CTA
    (torch.bfloat16, torch.float32, 2, 2, 32, 4, 120, 32),
]


@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,W,hd,pool", PA_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_W{c[5]}_hd{c[6]}" for c in PA_CASES])
def test_gx_pa2_row_types_match_dc_pa3(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool):
    ref, got = _pa_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, sph=4 * pool)
    _assert_same(ref, got, "gx_pa2 vs dc_pa3")
    assert not ref["grad"].isnan().any() and not got["eff"][:, :1].isnan().all()


def _gds_pair(bank_dtype, grad_dtype, P, Nt, C, R, hd, pool, sph, nw, rec8, seed=0, hi=0, monkeypatch=None,
              special=False, valid_last=None):
    """(mover_da2 GDS outputs, gx_gds outputs) over the reversed step loop on identical inputs."""
    _, _, _, gxm, _, da2, _ = _sdm()
    vl = C - 3 if valid_last is None else valid_last         # valid tokens of the last chunk of each stream
    p1, _, q_idx, _, q_val, _ = _phase1(P, Nt, C, 4, R, pool, sph, bank_dtype, seed, vl, hi)
    N, n = Nt * P, C * R
    gen = torch.Generator(device="cuda").manual_seed(seed + 2)
    rows = P * sph
    bank = _rows(rows, hd, bank_dtype, gen)
    grad0 = _rows(rows, hd, grad_dtype, gen, 0.01)
    if special:
        _inject_special(bank, sph), _inject_special(grad0, sph)
    gr = torch.randn(Nt * P * C, hd, device="cuda", generator=gen)
    spay_q = p1.spay_q
    q16 = p1.qrec.reshape(N, n, 4)
    q8 = p1.qrec8.reshape(N, n, 2)
    assert torch.equal(q8, q16[:, :, 1:3]), "Phase 1's qrec8 is not qrec's words 1-2"
    if monkeypatch is not None and nw is not None:
        ppw = {4: 32, 8: 128, 16: 256}[nw]
        monkeypatch.setattr(gxm, "geometry", lambda *a, **k: (nw, ppw))
    outs = []
    for use_gx in (False, True):
        grad = grad0.clone()
        dqw = torch.full((N, C, R), float("nan"), device="cuda")
        kw = dict(chunk_size=C, num_parallel=P, steps=Nt, q_idx=q_idx, valid_tokens=(Nt - 1) * C + vl, tiles_q=None)
        if use_gx:
            assert gxm.supported(bank, grad, C, R)
            launch = gxm.make_gds(bank, grad, gr, spay_q, q8 if rec8 else q16, dqw, **kw)
            assert launch.geom["impl"] == "gx"
        else:
            launch = da2.make_gds(bank, grad, gr, spay_q, q16, dqw, **kw)
        for ti in reversed(range(Nt)):
            launch(ti)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "d_q_w": dqw})
    return outs, bank


GDS_CASES = [   # (bank dtype, grad dtype, P, Nt, C, R, hd, pool)
    (torch.float32, torch.float32, 2, 3, 64, 8, 64, 48),
    (torch.float32, torch.float32, 2, 2, 128, 16, 1024, 96),
    (torch.float32, torch.float32, 2, 2, 32, 4, 100, 32),
    (torch.bfloat16, torch.float32, 2, 3, 64, 8, 128, 48),
    (torch.bfloat16, torch.float32, 2, 2, 64, 16, 960, 64),
]


@pytest.mark.parametrize("nw", [4, 8, 16])
@pytest.mark.parametrize("rec8", [False, True], ids=["qrec16", "rec8"])
@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,R,hd,pool", GDS_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_R{c[5]}_hd{c[6]}" for c in GDS_CASES])
def test_gx_gds_row_types_match_da2(bank_dtype, grad_dtype, P, Nt, C, R, hd, pool, rec8, nw, monkeypatch):
    _, _, _, gxm, *_ = _sdm()
    if nw == 16:   # the 16-warp instantiations themselves (H100 plans float32 grad rows at 8 warps: tested below)
        monkeypatch.setattr(gxm, "capability", lambda *a, **k: (10, 0))
    copies = gxm.REC8_COPIES[0]
    (ref, got), _ = _gds_pair(bank_dtype, grad_dtype, P, Nt, C, R, hd, pool, 4 * pool, nw, rec8,
                              monkeypatch=monkeypatch)
    _assert_same(ref, got, "gx_gds vs mover_da2 GDS")
    assert not ref["d_q_w"].isnan().any()
    assert gxm.LAST_POLICY["w64"] and gxm.LAST_POLICY["nw"] == nw
    assert gxm.REC8_COPIES[0] == copies + (0 if rec8 else 1)   # 16-byte records: one copy per factory call


@pytest.mark.parametrize("bank_dtype,grad_dtype", [(torch.float32, torch.float32), (torch.bfloat16, torch.float32)],
                         ids=["f32_f32", "bf16_f32"])
def test_row_types_special_values_match(bank_dtype, grad_dtype, monkeypatch):
    """-0.0, subnormal, +-inf and NaN in bank, grad and snapshot rows: both movers keep the fallbacks' bits."""
    ref, got = _pa_pair(bank_dtype, grad_dtype, 2, 2, 64, 8, 128, 32, 128, special=True)
    _assert_same(ref, got, "gx_pa2 vs dc_pa3 (special values)")
    (ref, got), _ = _gds_pair(bank_dtype, grad_dtype, 2, 2, 64, 8, 128, 32, 128, 8, True, monkeypatch=monkeypatch,
                              special=True)
    _assert_same(ref, got, "gx_gds vs mover_da2 GDS (special values)")


def test_rec8_copy_of_a_single_entry_is_aligned(monkeypatch):
    """C = R = P = steps = 1: words 1-2 of a [1, 1, 4] record are already 'contiguous' at a 4-byte offset."""
    (ref, got), _ = _gds_pair(torch.float32, torch.float32, 1, 1, 1, 1, 64, 4, 16, None, False, valid_last=1)
    _assert_same(ref, got, "gx_gds vs mover_da2 GDS (one entry, 16-byte records)")


# ------------------------------------------------------------------ banks of >= 2^32 elements (64-bit row offsets)
BIG_ROWS = (1 << 22) + 16   # x 1024 columns = 2^32 + 16,384 elements: bf16 8 GiB, f32 16 GiB per bank


@pytest.mark.slow
@pytest.mark.parametrize("bank_dtype,grad_dtype", [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
                                                   (torch.float32, torch.float32)],
                         ids=["bf16_bf16", "bf16_f32", "f32_f32"])
def test_big_bank_movers_match(bank_dtype, grad_dtype):
    """Both backward movers on a bank of 2^32 + 16,384 elements."""
    from sdm_testlib import require_free_memory
    require_free_memory(96)
    _, _, _, gxm, _, _, _ = _sdm()
    ref, got = _pa_pair(bank_dtype, grad_dtype, 1, 2, 64, 16, 1024, 4096, BIG_ROWS, hi=BIG_ROWS)
    _assert_same(ref, got, "gx_pa2 vs dc_pa3 (big bank)")
    del ref, got
    torch.cuda.empty_cache()
    (ref, got), bank = _gds_pair(bank_dtype, grad_dtype, 1, 2, 64, 16, 1024, 4096, BIG_ROWS, None, True,
                                 hi=BIG_ROWS)
    assert bank.numel() >= 2 ** 32 and gxm.LAST_POLICY["w64"]
    _assert_same(ref, got, "gx_gds vs mover_da2 GDS (big bank)")


# ------------------------------------------------------------------ float32 forward DW (mover_fx_cuda) vs today's path
def _op_run(d, *, C, sph, P, no_grad=False, cot_seed=17):
    """sdm_write_read on private copies of d."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    leaves = {n: d[n].detach().clone().requires_grad_(not no_grad) for n in OP_INPUTS}
    args = (leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"], d["q_idx"],
            leaves["q_val"])
    kw = dict(chunk_size=C, num_memory_slots=sph, num_parallel=P)
    if no_grad:
        with torch.no_grad():
            out = sdm_write_read(*args, **kw)
        return {"readings": out, "bank": leaves["memory"]}
    out = sdm_write_read(*args, **kw)
    res = {"readings": out.detach().clone(), "bank_fwd": leaves["memory"].detach().clone()}
    gen = torch.Generator(device="cuda").manual_seed(cot_seed)
    out.backward(torch.randn(out.shape, device="cuda", generator=gen).to(out.dtype))
    res["bank_bwd"] = leaves["memory"].detach()
    for n in OP_INPUTS:
        res["grad_" + n] = leaves[n].grad
    return res


def _decline_fx_f32(monkeypatch):
    """Today's float32 path: mover_fx_cuda declines float32 banks."""
    from sparse_delta_memory.cuda import gx_pa2_cuda as gxp, mover_fx_cuda as fx
    orig = fx.supported
    monkeypatch.setattr(fx, "supported", lambda bank, *a, **k: bank.dtype != torch.float32 and orig(bank, *a, **k))
    orig_pa = gxp.supported
    monkeypatch.setattr(gxp, "supported",
                        lambda grad, bank, *a, **k: bank.dtype != torch.float32 and orig_pa(grad, bank, *a, **k))


FWD_CASES = [   # (P, T, W, R, hd, sph, C, slots): one slice / 8 and 16 warps / column slices / 100-column rows
    (2, 256, 8, 8, 128, 512, 64, "hot"),
    (2, 512, 16, 16, 128, 1024, 256, "zipf"),
    (2, 256, 8, 8, 960, 512, 64, "hot"),
    (1, 512, 64, 64, 64, 4096, 64, "uniform"),
    (2, 192, 8, 8, 100, 256, 32, "hot"),
    (2, 300, 16, 16, 256, 1024, 128, "trained"),
    (2, 512, 16, 16, 256, 1024, 256, "zipf"),          # 16 warps with column slices
]


@pytest.mark.parametrize("no_grad", [False, True], ids=["saving", "nograd"])
@pytest.mark.parametrize("P,T,W,R,hd,sph,C,slots", FWD_CASES,
                         ids=[f"P{c[0]}_T{c[1]}_W{c[2]}_hd{c[4]}_C{c[6]}_{c[7]}" for c in FWD_CASES])
def test_fp32_forward_fx_matches_da2_path(P, T, W, R, hd, sph, C, slots, no_grad, monkeypatch, ieee_fp32):
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    d = make_op_inputs(P, T, W, R, hd, sph, torch.float32, seed=11, slots=slots)
    n0, pa0 = fx.STATS.get("fx_dw_f32", 0), bw.STATS.get("gx_pa2", 0)
    got = _op_run(d, C=C, sph=sph, P=P, no_grad=no_grad)
    torch.cuda.synchronize()
    n1, pa1 = fx.STATS.get("fx_dw_f32", 0), bw.STATS.get("gx_pa2", 0)
    assert n1 > n0, "the float32 forward did not take mover_fx_cuda's DW"
    if not no_grad:
        assert pa1 > pa0, "the float32 backward did not take gx_pa2 (dense compact rows)"
    with monkeypatch.context() as m:
        _decline_fx_f32(m)
        ref = _op_run(d, C=C, sph=sph, P=P, no_grad=no_grad)
        torch.cuda.synchronize()
    assert fx.STATS.get("fx_dw_f32", 0) == n1 and bw.STATS.get("gx_pa2", 0) == pa1   # the reference took today's path
    _assert_same(ref, got, "float32 op: mover_fx DW vs mover_da2 DW + stash")


def test_fp32_forward_fx_special_values(monkeypatch, ieee_fp32):
    """-0.0, subnormal, +-inf and NaN bank rows (and -0.0 / subnormal values)."""
    d = make_op_inputs(2, 256, 8, 8, 128, 1024, torch.float32, seed=13, slots="uniform")
    _inject_special(d["memory"], 1024)
    d["v"][:, ::7] = -0.0
    d["v"][:, 3::11] = 1e-39
    got = _op_run(d, C=64, sph=1024, P=2)
    with monkeypatch.context() as m:
        _decline_fx_f32(m)
        ref = _op_run(d, C=64, sph=1024, P=2)
    _assert_same(ref, got, "float32 op with special values: mover_fx DW vs mover_da2 DW + stash")


def _saved(out):
    """The SDMWriteRead node of a readings tensor (saved tensors: memory, ..., snapshot, scale, *Phase 1 fields)."""
    fn = out.grad_fn
    while fn is not None and type(fn).__name__ != "SDMWriteReadBackward":
        fn = fn.next_functions[0][0]
    assert fn is not None, "no SDMWriteRead node"
    return fn


def test_fp32_compact_rows_equal_the_fallback_stash(monkeypatch, ieee_fp32):
    """Every compact row mover_fx's float32 DW writes."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    P, T, W, R, hd, sph, C = 2, 256, 8, 8, 128, 1024, 64
    d = make_op_inputs(P, T, W, R, hd, sph, torch.float32, seed=17, slots="hot")
    _inject_special(d["memory"], sph)

    def fwd():
        leaves = {k: d[k].detach().clone().requires_grad_(True) for k in OP_INPUTS}
        out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"],
                             leaves["g"], d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph,
                             num_parallel=P)
        fn = _saved(out)
        return fn.saved_tensors[11], fn.quant_layout, fn.saved_tensors[13 + 10]   # snapshot, layout, sorted payload

    snapc, layout, pay = fwd()
    with monkeypatch.context() as m:
        _decline_fx_f32(m)
        stash, slayout, _ = fwd()
    assert layout == 2 and slayout != 2, (layout, slayout)
    n = C * W
    p = pay[:, :C * (W + R)].long().cpu() & 0xFFFFFFFF
    a, b = snapc.view(-1, hd).view(torch.int32).cpu(), stash.view(-1, hd).view(torch.int32).cpu()
    rows = 0
    for c in range(p.shape[0]):
        g, seen = 0, None
        for e in p[c].tolist():
            if e >> 31:
                seen = False                                  # a new group (slot) starts
            if seen or (e >> 8) & 1 or (e >> 17) & 1:
                continue                                      # not the group's first valid write entry
            seen = True
            tok, lane = (e >> 9) & 0xFF, e & 0xFF
            assert torch.equal(a[c * n + g], b[c * n + tok * W + lane]), f"chunk {c} group {g}"
            g, rows = g + 1, rows + 1
    assert rows > 0


def _big_op_inputs(rows, hd, T, W, dtype, seed):
    """One stream over `rows` rows."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pool = 4096

    def slots(width):
        lo = torch.rand(T, pool, device="cuda", generator=gen).topk(width // 2, -1).indices
        hi = torch.rand(T, pool, device="cuda", generator=gen).topk(width - width // 2, -1).indices + (rows - pool)
        return torch.cat([lo, hi], -1).sort(-1).values.view(1, T, width)

    def rn(*shape, scale):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(dtype)

    return dict(memory=_rows(rows, hd, dtype, gen), k_idx=slots(W), q_idx=slots(W), k_val=rn(1, T, W, scale=0.3),
                v=rn(1, T, hd, scale=0.1), beta=(torch.rand(1, T, 1, device="cuda", generator=gen) * 0.3).to(dtype),
                g=(torch.rand(1, T, 1, device="cuda", generator=gen) * -0.49 - 0.01).to(dtype),
                q_val=rn(1, T, W, scale=0.3))


@pytest.mark.slow
def test_big_fp32_forward_fx_matches_da2_path(monkeypatch, ieee_fp32):
    """The float32 op on a bank of 2^32 + 16,384 elements (16 GiB; half of the slots in its last 4096 rows)."""
    from sdm_testlib import require_free_memory
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    require_free_memory(120)
    d = _big_op_inputs(BIG_ROWS, 1024, 512, 16, torch.float32, seed=41)
    assert d["memory"].numel() >= 2 ** 32
    n0 = fx.STATS.get("fx_dw_f32", 0)
    got = _op_run(d, C=64, sph=BIG_ROWS, P=1)
    got = {k: got[k] for k in ("readings", "bank_bwd", "grad_memory", "grad_v", "grad_k_val", "grad_q_val")}
    torch.cuda.synchronize()
    assert fx.STATS.get("fx_dw_f32", 0) > n0
    torch.cuda.empty_cache()
    with monkeypatch.context() as m:
        _decline_fx_f32(m)
        ref = _op_run(d, C=64, sph=BIG_ROWS, P=1)
        ref = {k: ref[k] for k in got}
        torch.cuda.synchronize()
    _assert_same(ref, got, "float32 op on a 2^32-element bank: mover_fx DW vs mover_da2 DW + stash")
