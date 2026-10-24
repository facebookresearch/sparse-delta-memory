# Copyright (c) Meta Platforms, Inc. and affiliates.
"""mover_fx_cuda's portable side."""
import os

import pytest
import torch

from sdm_testlib import bits_mismatch, make_op_inputs, requires_cuda

MODERN = ((9, 0), (10, 0), (10, 3))
A100 = ((8, 0), 166912, 108, 41943040)     # capability, opt-in smem per block, SMs, L2 bytes (rule 0e rows)
RTX = ((8, 6), 101376, 82, 6291456)


def _tb():
    import test_backends as tb      # the image helpers (_native / _portable: cuobjdump-checked builds)
    return tb


def _tmr():
    import test_mover_row_types as tmr   # Phase-1 inputs and row helpers of the movers' row-type tests
    return tmr


def _dev():
    from sparse_delta_memory.cuda import _device
    return _device


def _row(m, row):
    d = _dev()
    cc, optin, sms, l2 = row
    m.setattr(d, "_FORCED", (tuple(cc), int(optin), int(sms), l2))


def _modern_gpu():
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) in MODERN


requires_modern = pytest.mark.skipif(not _modern_gpu(), reason="needs an H100 / B200 / GB300 (native = modern side)")


# (bank dtype, P, Nt, C, W, hd, pool, layout, quant): layout 2 = saving forward (dense compact rows written by the DW),
# 4 = no-grad forward (bank-direct, one-step snapshot); quant = one-slice fp8 / int4 rows (layout 2, by the DW)
DW_CASES = [
    (torch.bfloat16, 2, 3, 64, 8, 128, 48, 2, None),       # one slice, 8 warps
    (torch.bfloat16, 2, 3, 64, 8, 128, 48, 4, None),       # no-grad
    (torch.bfloat16, 2, 2, 256, 4, 128, 96, 2, None),      # 128 KiB of token rows: 16 warps on A100
    (torch.bfloat16, 2, 2, 64, 8, 384, 64, 2, None),       # 3 column slices (fx_dw, T2D)
    (torch.bfloat16, 2, 2, 64, 8, 960, 64, 2, None),       # 8 slices of 120: two-slice walk (fx_dw2)
    (torch.bfloat16, 1, 2, 128, 8, 1024, 64, 4, None),     # fx_dw2, 16 warps (no-grad)
    (torch.bfloat16, 2, 3, 16, 16, 256, 48, 2, None),      # one slice pair, C 16
    (torch.float32, 2, 3, 64, 8, 128, 48, 2, None),        # float32 rows, one slice
    (torch.float32, 2, 2, 64, 8, 960, 64, 2, None),        # float32 column slices
    (torch.float32, 2, 2, 128, 8, 256, 64, 4, None),       # float32 no-grad
    (torch.float32, 2, 2, 256, 4, 128, 96, 2, None),       # float32, 16 warps
    (torch.bfloat16, 2, 3, 64, 8, 128, 48, 2, "fp8"),      # one-slice quantized rows (warp max in the walk)
    (torch.bfloat16, 2, 3, 64, 8, 64, 48, 2, "int4"),
    (torch.float32, 2, 2, 32, 8, 128, 48, 2, "int4"),
]


_FILL = 12352.0   # the unwritten compact rows' value (bf16- and f32-exact; no random or special bank row equals it)


def _inputs(bank_dtype, P, Nt, C, W, hd, pool, layout, quant, seed=0):
    tmr = _tmr()
    vl = C - 3
    p1, _, _, _, _, _ = tmr._phase1(P, Nt, C, W, 4, pool, 4 * pool, bank_dtype, seed, vl)
    n = C * W
    gen = torch.Generator(device="cuda").manual_seed(seed + 7)
    bank0 = tmr._rows(P * 4 * pool, hd, bank_dtype, gen)
    tmr._inject_special(bank0, 4 * pool)
    for r0 in range(0, bank0.shape[0], 4 * pool):   # one large element in one slice; tiny values with one moderate one
        bank0[r0 + 5, hd - 3] = 300.0
        bank0[r0 + 6] = 1e-30
        bank0[r0 + 6, hd // 2 + 1] = -2.5
    every = layout == 2
    dv = torch.randn((Nt if every else 1) * P * C, hd, device="cuda", generator=gen)
    return dict(p1=p1, n=n, bank0=bank0, dv=dv, valid=(Nt - 1) * C + vl)


def _run(fx_mod, row, case, inp, monkeypatch, all_slices=False, pdl=False):
    """The DW launcher of mover_da2_cuda.make_dw."""
    from sparse_delta_memory.cuda import mover_da2_cuda as da2, mover_fx_cuda as fx
    bank_dtype, P, Nt, C, W, hd, pool, layout, quant = case
    n = inp["n"]
    with monkeypatch.context() as m:
        if row is not None:
            _row(m, row)
        m.setattr(fx, "_module", lambda mod=fx_mod: mod)
        if all_slices:      # the all-slices form on any backend (quant_form's choice for heads past the cluster)
            m.setattr(fx, "quant_form", lambda dev, head_dim, q: (q | 4) if q and head_dim > 128 else q)
        bank = inp["bank0"].clone()
        rows = (Nt if (layout == 2 and quant is None) else 1) * P * C
        snap = torch.full((rows, W, hd), _FILL, device="cuda", dtype=bank_dtype)
        snap._dc_snapc_layout = layout
        if layout == 2:
            snap._dc_snapc_by_dw = True
        kw = {}
        if quant is not None:
            kw = dict(snapshot_quant=quant, valid_tokens=inp["valid"],
                      payload=torch.full((Nt * P * n, hd if quant == "fp8" else hd // 2), 0xAB, device="cuda",
                                         dtype=torch.uint8),
                      scale=torch.full((Nt * P * n,), -1.0, device="cuda"))
        p1 = inp["p1"]
        launch = da2.make_dw(bank, inp["dv"], p1.spay_k.reshape(-1, n), p1.krec.reshape(-1, n, 4), chunk_size=C,
                             num_parallel=P, steps=Nt, snapshot=snap, tiles_k=None, segb_k=None, **kw)
        assert launch.fx is not None, "mover_fx did not take the call"
        launch.set_pdl(pdl)
        out = {}
        for ti in range(Nt):
            launch(ti)
            out[f"bank_after_step{ti}"] = bank.clone()
        torch.cuda.synchronize()
        out["bank"] = bank
        if layout == 2 and quant is None:
            out["snapc"] = snap
        if quant is not None:
            out["payload"], out["scale"] = kw["payload"], kw["scale"]
        return out, dict(launch.fx.geom)


@requires_cuda
@requires_modern
@pytest.mark.parametrize("bank_dtype,P,Nt,C,W,hd,pool,layout,quant", DW_CASES,
                         ids=[f"{str(c[0])[6:]}_C{c[3]}_W{c[4]}_hd{c[5]}_L{c[7]}{'_' + c[8] if c[8] else ''}"
                              for c in DW_CASES])
def test_portable_fx_matches_modern(bank_dtype, P, Nt, C, W, hd, pool, layout, quant, monkeypatch):
    """compute_80 fx under the A100 row vs native fx under the GPU's own row: the same bits."""
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    case = (bank_dtype, P, Nt, C, W, hd, pool, layout, quant)
    inp = _inputs(*case)
    native, mod80 = _tb()._native(fx), _tb()._portable(fx)
    a, ga = _run(native, None, case, inp, monkeypatch)
    b, gb = _run(mod80, A100, case, inp, monkeypatch)
    for k in a:
        bad = bits_mismatch(a[k], b[k])
        assert not bad, f"fx portable vs modern: {k} differs: {bad} (geometry modern {ga}, a100 {gb})"
    assert bits_mismatch(a["bank"], inp["bank0"]), "the DW wrote nothing"
    if hd > 128 and hd // -(-hd // 128) * -(-hd // 128) == hd and (-(-hd // 128)) % 2 == 0 and bank_dtype == torch.bfloat16:
        assert gb["w2"], gb   # even slice counts take the two-slice walk on A100 too


# ---------------------------------------------------------------------------------------------------------------------
# The plans (patched rows; no kernel launch)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Route witness of the forced A100 job (P2_ROW=a100: every extension built as compute_80 PTX, the row forced process-wide)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# The portable quantized column-sliced rows (one CTA per tile set over every slice; no clusters)
# ---------------------------------------------------------------------------------------------------------------------

QUANT_CASES = [   # (bank dtype, P, Nt, C, W, hd, pool, quant)
    (torch.bfloat16, 2, 2, 64, 8, 384, 64, "fp8"),       # fx_dw, 3 slices
    (torch.bfloat16, 2, 2, 64, 8, 1024, 64, "int4"),     # fx_dw2, 4 slice pairs (README class)
    (torch.bfloat16, 2, 2, 32, 8, 960, 48, "fp8"),       # fx_dw2, slices of 120 (lanes 30-31 off: the tail)
    (torch.bfloat16, 1, 2, 256, 4, 640, 96, "int4"),     # fx_dw, 5 slices, C 256 (16 warps)
    (torch.float32, 2, 2, 64, 8, 960, 64, "fp8"),        # float32 rows (IEEE division)
    (torch.float32, 2, 2, 16, 8, 256, 48, "int4"),       # float32, 2 slices
    (torch.bfloat16, 1, 2, 64, 8, 2176, 64, "fp8"),      # 17 slices: no modern cluster holds them (reference only)
]


def _a100_class_gpu():
    """An sm_80+ GPU with at least the A100's opt-in shared memory per block (166,912 B): an A100 or a modern part."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        return False
    return torch.cuda.get_device_properties(0).shared_memory_per_block_optin >= 166912


requires_sm80 = pytest.mark.skipif(not _a100_class_gpu(),
                                   reason="needs an A100-class GPU (sm_80+, >= 166,912 B opt-in shared memory per block)")


@requires_cuda
@requires_sm80
@pytest.mark.parametrize("bank_dtype,P,Nt,C,W,hd,pool,quant", QUANT_CASES,
                         ids=[f"{str(c[0])[6:]}_C{c[3]}_hd{c[5]}_{c[7]}" for c in QUANT_CASES])
def test_portable_quant_rows(bank_dtype, P, Nt, C, W, hd, pool, quant, monkeypatch):
    """Under the A100 row (compute_80 fx)."""
    from test_quant_fx import ref_quant
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    qcase = (bank_dtype, P, Nt, C, W, hd, pool, 2, quant)
    rcase = (bank_dtype, P, Nt, C, W, hd, pool, 2, None)
    inp = _inputs(*qcase)
    mod80 = _tb()._portable(fx)
    q, gq = _run(mod80, A100, qcase, inp, monkeypatch)
    raw, _ = _run(mod80, A100, rcase, inp, monkeypatch)
    assert gq.get("all_slices") and gq["quant"] and not gq["cluster"], gq
    assert not bits_mismatch(q["bank"], raw["bank"]), "the quantized walk's bank differs from the quant-none walk's"
    written = q["scale"] != -1.0
    snap = raw["snapc"].reshape(written.numel(), hd)
    fill = torch.tensor(_FILL, dtype=bank_dtype, device="cuda")
    raw_written = (snap != fill).any(-1) | snap.isnan().any(-1)
    assert torch.equal(written, raw_written) and int(written.sum()) > 0, (int(written.sum()), int(raw_written.sum()))
    pay_ref, sc_ref = ref_quant(snap[written].float(), quant)
    assert torch.equal(q["payload"][written].cpu(), pay_ref), "payload differs from the interface's formula"
    assert torch.equal(q["scale"][written].cpu().view(torch.int32), sc_ref.view(torch.int32)), "scales differ"
    if _modern_gpu() and _cluster_form_takes(fx, hd, bank_dtype, C, quant):
        native = _tb()._native(fx)
        m, gm = _run(native, None, qcase, inp, monkeypatch)
        assert gm["cluster"] > 1, gm
        for k in m:
            bad = bits_mismatch(m[k], q[k])
            assert not bad, f"portable vs modern quantized rows: {k} differs: {bad}"


def _cluster_form_takes(fx, hd, dtype, C, quant):
    """The modern cluster form takes this head."""
    q = 1 if quant == "fp8" else 2
    return fx.quant_choice(torch.empty(4, hd, device="cuda", dtype=dtype), C, q) == q


def _digest(out):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(k.encode())
        h.update(out[k].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


DET_CASE = (torch.bfloat16, 2, 2, 64, 8, 960, 64, 2, "int4")   # the portable quantized two-slice walk (all slices)
DET_CASE_F = (torch.float32, 1, 2, 64, 8, 640, 64, 2, "fp8")    # float32 rows, 5 column slices (fx_dw, all slices)


def _det_arm(arm, case, monkeypatch, inp=None):
    """(outputs, inputs) of `case` on one arm."""
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    inp = inp or _inputs(*case)
    if arm == "portable":
        out, g = _run(_tb()._portable(fx), A100, case, inp, monkeypatch)
    else:
        out, g = _run(_tb()._native(fx), None, case, inp, monkeypatch, all_slices=True)
    assert g["all_slices"], g
    return out, inp


@requires_cuda
@requires_sm80
@pytest.mark.parametrize("arm", ["portable", "modern"])
@pytest.mark.parametrize("ci", [0, 1], ids=["bf16_dw2_int4", "f32_dw_fp8"])
def test_all_slices_form_deterministic_and_graph_replay(arm, ci, monkeypatch):
    """The all-slices quantized form on both sides (portable: A100 row, compute_80; modern: native, forced)."""
    import subprocess
    import sys
    from sparse_delta_memory.cuda import mover_da2_cuda as da2, mover_fx_cuda as fx
    if arm == "modern" and not _modern_gpu():
        pytest.skip("the modern arm needs an H100 / B200 / GB300")
    case = (DET_CASE, DET_CASE_F)[ci]
    a, inp = _det_arm(arm, case, monkeypatch)
    b, _ = _det_arm(arm, case, monkeypatch, inp)
    for k in a:
        assert not bits_mismatch(a[k], b[k]), f"repeat: {k}"
    bank_dtype, P, Nt, C, W, hd, pool, layout, quant = case
    n = inp["n"]
    with monkeypatch.context() as m:
        if arm == "portable":
            _row(m, A100)
            m.setattr(fx, "_module", lambda mod=_tb()._portable(fx): mod)
        else:
            m.setattr(fx, "_module", lambda mod=_tb()._native(fx): mod)
            m.setattr(fx, "quant_form", lambda dev, head_dim, q: (q | 4) if q and head_dim > 128 else q)
        bank = inp["bank0"].clone()
        snap = torch.full((P * C, W, hd), _FILL, device="cuda", dtype=bank_dtype)
        snap._dc_snapc_layout, snap._dc_snapc_by_dw = 2, True
        payload = torch.full((Nt * P * n, hd if quant == "fp8" else hd // 2), 0xAB, device="cuda", dtype=torch.uint8)
        scale = torch.full((Nt * P * n,), -1.0, device="cuda")
        p1 = inp["p1"]
        launch = da2.make_dw(bank, inp["dv"], p1.spay_k.reshape(-1, n), p1.krec.reshape(-1, n, 4), chunk_size=C,
                             num_parallel=P, steps=Nt, snapshot=snap, snapshot_quant=quant, payload=payload,
                             scale=scale, valid_tokens=inp["valid"])
        assert launch.fx is not None and launch.fx.geom["all_slices"]
        launch(0)                                   # warm (plan, attributes) outside the capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            for ti in range(Nt):
                launch(ti)
        torch.cuda.current_stream().wait_stream(s)
        bank.copy_(inp["bank0"])
        payload.fill_(0xAB)
        scale.fill_(-1.0)
        g.replay()
        torch.cuda.synchronize()
        for k, v in (("bank", bank), ("payload", payload), ("scale", scale)):
            assert not bits_mismatch(a[k], v), f"graph replay: {k}"
    code = ("import sys, torch, pytest; sys.path[:0] = [%r]; import test_fx_portable as t; "
            "mp = pytest.MonkeyPatch(); out, _ = t._det_arm(%r, (t.DET_CASE, t.DET_CASE_F)[%d], mp); "
            "print('DIGEST', t._digest({k: out[k] for k in ('bank', 'payload', 'scale')}))" % (os.path.dirname(__file__),
                                                                                              arm, ci))
    res = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=1800,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    line = [ln for ln in res.stdout.splitlines() if ln.startswith("DIGEST ")]
    assert res.returncode == 0 and line, res.stderr[-3000:]
    assert line[-1].split()[1] == _digest({k: a[k] for k in ("bank", "payload", "scale")}), "fresh process differs"


@requires_cuda
@requires_modern
@pytest.mark.parametrize("pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("bank_dtype,P,Nt,C,W,hd,pool,quant", QUANT_CASES,
                         ids=[f"{str(c[0])[6:]}_C{c[3]}_hd{c[5]}_{c[7]}" for c in QUANT_CASES])
def test_modern_all_slices_form(bank_dtype, P, Nt, C, W, hd, pool, quant, pdl, monkeypatch):
    """The all-slices form (QUANT 5 / 6) on the modern side."""
    from test_quant_fx import ref_quant
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    qcase = (bank_dtype, P, Nt, C, W, hd, pool, 2, quant)
    inp = _inputs(*qcase)
    native = _tb()._native(fx)
    a, ga = _run(native, None, qcase, inp, monkeypatch, all_slices=True, pdl=pdl)
    assert ga["all_slices"] and not ga["cluster"], ga
    raw, _ = _run(native, None, (bank_dtype, P, Nt, C, W, hd, pool, 2, None), inp, monkeypatch)
    assert not bits_mismatch(a["bank"], raw["bank"])
    written = a["scale"] != -1.0
    snap = raw["snapc"].reshape(written.numel(), hd)
    pay_ref, sc_ref = ref_quant(snap[written].float(), quant)
    assert torch.equal(a["payload"][written].cpu(), pay_ref) and \
        torch.equal(a["scale"][written].cpu().view(torch.int32), sc_ref.view(torch.int32))
    if _cluster_form_takes(fx, hd, bank_dtype, C, quant):
        c, gc = _run(native, None, qcase, inp, monkeypatch, pdl=pdl)
        assert gc["cluster"] > 1, gc
        for k in c:
            assert not bits_mismatch(c[k], a[k]), f"all-slices vs cluster form: {k}"


# ---------------------------------------------------------------------------------------------------------------------
# The public quantized op on column-sliced heads: the producer (portable: the all-slices form 5 / 6; modern: the cluster
# form, or the all-slices form for heads past the cluster, quant_choice) -> gx_pa2's quantized consumer
# ---------------------------------------------------------------------------------------------------------------------

OP_Q_CASES = [   # (bank dtype, grad_bank_fp32, C, hd, quant)
    (torch.bfloat16, False, 64, 384, "fp8"),       # fx_dw, 3 slices
    (torch.bfloat16, True, 64, 384, "int4"),       # grad_bank_fp32 (f32 grad rows)
    (torch.bfloat16, False, 256, 1024, "fp8"),     # two-slice walk (4 pairs), C 256 (the A100 class: 166,912 B plans)
    (torch.bfloat16, True, 256, 1024, "int4"),
    (torch.float32, False, 64, 640, "fp8"),        # float32 rows, 5 slices
    (torch.float32, False, 128, 256, "int4"),
    (torch.bfloat16, False, 64, 2176, "fp8"),      # 17 slices: no cluster holds them (modern: the all-slices form too)
    (torch.bfloat16, True, 64, 2176, "int4"),
]


@requires_cuda
@pytest.mark.parametrize("dtype,gbf32,C,hd,quant", OP_Q_CASES,
                         ids=[f"{str(c[0])[6:]}{'_gbf32' if c[1] else ''}_C{c[2]}_hd{c[3]}_{c[4]}" for c in OP_Q_CASES])
def test_quantized_op_column_sliced(dtype, gbf32, C, hd, quant, monkeypatch):
    """The public op with fp8 / int4 snapshots on column-sliced heads, on this process's backend."""
    import types
    from sdm_testlib import OP_INPUTS
    from test_quant_rows import dequant_ref
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import mover_da2_cuda as da2, mover_fx_cuda as fx
    from sparse_delta_memory.memory_ops import sdm_write_read
    backend = _dev().backend()
    if backend not in ("modern", "a100"):
        pytest.skip("the modern and A100 classes (the RTX class is the P2b fallback's evidence)")
    P, W, R, sph = 2, 8, 8, 512
    d = make_op_inputs(P, 2 * C, W, R, hd, sph, dtype, seed=5)
    q = 1 if quant == "fp8" else 2
    form = fx.quant_form(d["memory"].device, hd, fx.quant_choice(d["memory"], C, q))
    assert form in (q, q | 4), form
    if backend == "a100":
        assert form == q | 4, form                 # no clusters below sm_90: every column-sliced head walks all slices
    geoms, make_dw0 = [], da2.make_dw

    def make_dw_spy(*a, **k):
        launch = make_dw0(*a, **k)
        if k.get("snapshot_quant", "none") != "none":
            geoms.append(dict(launch.fx.geom) if getattr(launch, "fx", None) is not None else None)
        return launch

    cap, bw0 = {}, bw.sdm_backward

    def bw_spy(ctx, saved, gr):
        cap.update(lay=ctx.lay, ql=getattr(ctx, "quant_layout", 0), inputs=ctx.inputs, gr=gr.clone(),
                   saved=[t.clone() if torch.is_tensor(t) else t for t in saved])
        n0, s0 = bw.STATS.get("gx_pa2", 0), bw.STATS["shipped"]
        res = bw0(ctx, saved, gr)
        cap.update(grads=[t.clone() if torch.is_tensor(t) else t for t in res], bank=saved[0].clone(),
                   gx=bw.STATS.get("gx_pa2", 0) - n0, shipped=bw.STATS["shipped"] - s0)
        return res

    monkeypatch.setattr(da2, "make_dw", make_dw_spy)
    monkeypatch.setattr(bw, "sdm_backward", bw_spy)

    def call(save):
        leaves = {n: d[n].detach().clone().requires_grad_(save) for n in OP_INPUTS}
        with torch.set_grad_enabled(save):
            out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                                 d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph, num_parallel=P,
                                 snapshot_quant=quant, grad_bank_fp32=gbf32)
        res = dict(out=out.detach().clone(), bank_fwd=leaves["memory"].detach().clone())
        if save:
            gen = torch.Generator(device="cuda").manual_seed(17)
            out.backward(torch.randn(out.shape, device="cuda", generator=gen).to(out.dtype))
            res.update({f"grad_{n}": leaves[n].grad.detach().clone() for n in OP_INPUTS if leaves[n].grad is not None})
        torch.cuda.synchronize()
        return res

    first = call(True)
    assert geoms and geoms[0] is not None, "the saving forward's DW did not take P1b's producer"
    g0 = geoms[0]
    assert g0["quant"] == form and g0["all_slices"] == (form >= 5), (g0, form)
    assert (g0["cluster"] == 0) if form >= 5 else (g0["cluster"] > 1), (g0, form)
    assert cap["ql"] == 2 and cap["gx"] == 1 and cap["shipped"] == 0, (cap["ql"], cap["gx"], cap["shipped"])
    assert all(f"grad_{n}" in first for n in ("k_val", "v", "beta", "g", "q_val")), sorted(first)
    # the backward on the decoded rows (unquantized layout-2 compact rows): the decode is the only difference
    saved2 = cap["saved"]
    pay, scale = saved2[11], saved2[12]
    assert pay.dtype == (torch.float8_e4m3fn if quant == "fp8" else torch.uint8), pay.dtype
    assert pay.shape[1] == (hd if quant == "fp8" else hd // 2) and scale.dtype == torch.float32, (pay.shape, scale.dtype)
    saved2[11] = dequant_ref(pay, scale, quant, dtype)
    saved2[12] = None                                  # 'none' saves no scale (its layout: ctx2.quant_layout)
    ctx2 = types.SimpleNamespace(lay=cap["lay"], snapshot_quant="none", quant_layout=cap["ql"], grad_bank_fp32=gbf32,
                                 inputs=cap["inputs"])
    grads2 = bw0(ctx2, saved2, cap["gr"])
    for i, (a, b) in enumerate(zip(cap["grads"], grads2)):
        if torch.is_tensor(a):
            bad = bits_mismatch(a, b)
            assert not bad, f"{quant} {dtype} hd {hd}: grad {i} differs from the decoded-rows backward: {bad}"
    bad = bits_mismatch(cap["bank"], saved2[0])
    assert not bad, f"{quant} {dtype} hd {hd}: restored bank differs from the decoded-rows backward's: {bad}"
    # the no-grad forward (bank-direct walk, no snapshot): the same output and bank, bitwise
    nograd = call(False)
    for k in ("out", "bank_fwd"):
        bad = bits_mismatch(first[k], nograd[k])
        assert not bad, f"{quant} {dtype} hd {hd}: saving vs no-grad forward: {k} differs: {bad}"
    # a second saving call repeats every bit
    second = call(True)
    for k in first:
        bad = bits_mismatch(first[k], second[k])
        assert not bad, f"{quant} {dtype} hd {hd}: repeat: {k} differs: {bad}"

