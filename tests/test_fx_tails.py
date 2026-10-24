# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Native tails, the forward side: mover_fx's DW on the tail shapes."""
import math
import os

import pytest
import torch

from sdm_testlib import make_op_inputs, OP_INPUTS

BF, F32 = torch.bfloat16, torch.float32


def _backend():
    """This process's backend: modern natively, or a portable row."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from sparse_delta_memory.cuda import _device
    try:
        b = _device.backend()
    except RuntimeError:
        pytest.skip("not a backend of this package")
    row = os.environ.get("P2_ROW")
    if row:
        assert b == ("a100" if row == "a100" else "rtx"), f"P2_ROW={row} but the backend is {b} (forced row inactive)"
    return b


def _modern_only():
    if _backend() != "modern":
        pytest.skip("modern backend only")


def _portable_only():
    if _backend() == "modern":
        pytest.skip("the portable backends only (a real A100 / RTX, or a forced row)")


def _chunk_ok(C):
    if _backend() == "rtx" and C > 128:
        pytest.skip("RTX rows take chunks of <= 128 tokens (rule 0e)")


def _bits_eq(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _finite(*ts):
    return all(bool(torch.isfinite(t.float()).all()) for t in ts)


def _oracle_gate(cand, ref, dtype):
    """The fixed float64-oracle gates of sdm_testlib, route-independent."""
    from sdm_testlib import BF16_CEILINGS, GATE_FP32, rel_max, rel_rms
    if dtype == torch.float32:
        e = rel_max(ref, cand)
        return e <= GATE_FP32, (e, GATE_FP32)
    rms, mx = rel_rms(ref, cand), rel_max(ref, cand)
    return rms <= BF16_CEILINGS[0] and mx <= BF16_CEILINGS[1], (rms, mx)


def _fwd(d, C, sph, P, quant="none"):
    from sparse_delta_memory.memory_ops import sdm_write_read
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    kw = {} if quant == "none" else {"snapshot_quant": quant}
    y = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                       d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph, num_parallel=P, **kw)
    torch.cuda.synchronize()
    return y.detach(), leaves["memory"].detach()


def _fwd_nograd(d, C, sph, P):
    from sparse_delta_memory.memory_ops import sdm_write_read
    mem = d["memory"].detach().clone()
    with torch.no_grad():
        y = sdm_write_read(mem, d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"],
                           chunk_size=C, num_memory_slots=sph, num_parallel=P)
    torch.cuda.synchronize()
    return y, mem


def _witness(monkeypatch):
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    made = []
    make0 = fx.make_dw_fx
    monkeypatch.setattr(fx, "make_dw_fx", lambda *a, **k: (made.append(dict(k)), make0(*a, **k))[1])
    return made


# ---------------------------------------------------------------------------------------------------------------------
# hd > 16384
# ---------------------------------------------------------------------------------------------------------------------

# (hd, dtype, P, C, T): wide heads up to the new bound 2^20, one and two parallel streams, a partial last chunk, C 256
WIDE = [(16512, BF, 1, 64, 128), (32768, BF, 1, 64, 128), (16512, F32, 1, 64, 128), (16512, BF, 2, 128, 300),
        (16512, BF, 1, 256, 600), (1 << 20, BF, 1, 64, 96)]
WIDE_IDS = [f"hd{w[0]}_{str(w[1])[6:]}_P{w[2]}_C{w[3]}" for w in WIDE]


@pytest.mark.parametrize("hd,dtype,P,C,T", WIDE, ids=WIDE_IDS)
def test_wide_heads_take_fx_with_da2_bits(hd, dtype, P, C, T, monkeypatch):
    """Modern: heads above 16384 take mover_fx's walk with the bits of today's route."""
    _modern_only()
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    W, R, sph = 4, 4, 64
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=11, slots="zipf")
    made = _witness(monkeypatch)
    y_fx, m_fx = _fwd(d, C, sph, P)
    yn_fx, mn_fx = _fwd_nograd(d, C, sph, P)
    assert made, "mover_fx did not take the wide head"
    assert _finite(y_fx, m_fx, yn_fx, mn_fx)
    monkeypatch.setattr(fx, "supported", lambda *a, **k: False)          # today's route: mover_da2_cuda's own DW
    made.clear()
    s0 = movers.STATS["da2_dw"]
    y_da2, m_da2 = _fwd(d, C, sph, P)
    yn_da2, mn_da2 = _fwd_nograd(d, C, sph, P)
    assert not made, "the comparison run took mover_fx"
    assert movers.STATS["da2_dw"] >= s0 + 2, "the comparison run did not take mover_da2_cuda's DW"
    for name, a, b in (("readings", y_fx, y_da2), ("bank", m_fx, m_da2), ("no-grad readings", yn_fx, yn_da2),
                       ("no-grad bank", mn_fx, mn_da2)):
        assert _bits_eq(a, b), name


@pytest.mark.parametrize("case", ["offsets_past_2p31", "bf16_bank_above_4GiB"])
def test_wide_heads_large_offsets(case, monkeypatch):
    """Modern, the index boundaries: (1) hd 2^20, C 256, 9 steps."""
    _modern_only()
    import sdm_testlib as tl
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    if case == "offsets_past_2p31":
        P, C, T, W, R, sph, hd = 1, 256, 9 * 256, 4, 4, 64, 1 << 20
    else:
        P, C, T, W, R, sph, hd = 1, 64, 128, 4, 4, 131072, 16512
    d = make_op_inputs(P, T, W, R, hd, sph, BF, seed=31, slots="zipf")
    if case != "offsets_past_2p31":
        assert d["memory"].numel() * 2 > 2 ** 32, "premise: a bank above 4 GiB"
    made = _witness(monkeypatch)
    y1, m1 = _fwd_nograd(d, C, sph, P)
    assert made, "mover_fx did not take the call"
    assert _finite(y1, m1)
    from sparse_delta_memory.cuda import mover_da2_cuda as da2
    if not (da2.supported(d["memory"], C) and da2.fits(d["memory"], C, "dw")):
        return   # mover_da2_cuda's DW does not take this head: mover_fx's walk is its only route (2.0.0: no Triton DW)
    monkeypatch.setattr(fx, "supported", lambda *a, **k: False)
    made.clear()
    s0 = movers.STATS["da2_dw"]
    y0, m0 = _fwd_nograd(d, C, sph, P)
    assert movers.STATS["da2_dw"] > s0 and not made, "mover_da2_cuda's DW did not take the call"
    assert _bits_eq(y1, y0) and _bits_eq(m1, m0), "mover_fx != mover_da2_cuda's DW"


@pytest.mark.parametrize("hd,dtype,P,C,T", WIDE, ids=WIDE_IDS)
def test_portable_wide_heads(hd, dtype, P, C, T, monkeypatch):
    """A100 / RTX."""
    _portable_only()
    _chunk_ok(C)
    import sdm_testlib as tl
    from sparse_delta_memory import movers
    W, R, sph = 4, 4, 64
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=11, slots="zipf")
    made = _witness(monkeypatch)
    s0 = movers.STATS["shipped_dw"]
    y1, m1 = _fwd(d, C, sph, P)
    y2, m2 = _fwd(d, C, sph, P)
    y3, m3 = _fwd_nograd(d, C, sph, P)
    y4, m4 = _fwd_nograd(d, C, sph, P)
    assert made and movers.STATS["shipped_dw"] == s0, "the wide head did not take mover_fx's walk"
    assert _finite(y1, m1, y3, m3)
    for a, b in ((y1, y2), (m1, m2), (y3, y4), (m3, m4), (y1, y3), (m1, m3)):
        assert _bits_eq(a, b)
    ref_y, _, ref_m = tl.oracle_write_read(d, None, device="cuda", sph=sph)
    gate = 1e-1 if dtype == BF else 1e-5
    e = (tl.rel_max(ref_y, y1.view(ref_y.shape)), tl.rel_max(ref_m, m1.view(ref_m.shape)))
    assert all(math.isfinite(x) and x < gate for x in e), e


@pytest.mark.parametrize("dtype", [BF, F32])
def test_quantized_wide_heads_are_refused(dtype, monkeypatch):
    """fp8 / int4 calls on heads above 16384 have no quantized route on any backend."""
    b = _backend()
    from sdm_testlib import assert_quant_tail_refused
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    for hd in (16512, 32768, 1 << 20):
        bank = torch.empty(64, hd, device="cuda", dtype=dtype)
        for q in (1, 2):
            assert fx.quant_choice(bank, 64, q) == 0 and not fx.quant_supported(bank, 64, q)
        if b != "modern":
            assert movers.dw_admits(bank, 64, "none")
            assert not movers.dw_admits(bank, 64, "fp8") and not movers.dw_admits(bank, 64, "int4")
    d = make_op_inputs(1, 128, 4, 4, 32768, 64, dtype, seed=7, slots="zipf")
    made = _witness(monkeypatch)
    for quant in ("fp8", "int4"):
        s0 = dict(movers.STATS)
        assert_quant_tail_refused(lambda: _fwd(d, 64, 64, 1, quant), "T0")
        assert not made and dict(movers.STATS) == s0, f"a refused ({quant}) wide head reached a mover"
    y, _ = _fwd_nograd(d, 64, 64, 1)
    assert _finite(y)


# ---------------------------------------------------------------------------------------------------------------------
# Tail classes
# ---------------------------------------------------------------------------------------------------------------------

# (a) hd % 4 == 0 with equal slices of SW % 4 == 0 columns whose rows are not 16-byte multiples (bf16);
# (b) UNEVEN: hd % 4 == 0 whose equal slices would not be 4-column groups (128-wide slices, the last one short);
# (c) SCALAR: hd % 4 != 0 (element accesses, CTA-copied token rows)
TAILS = [(12, BF), (20, BF), (36, BF), (124, BF), (136, BF), (200, BF), (132, BF), (196, BF), (320, BF), (1028, BF),
         (132, F32), (320, F32), (63, BF), (125, BF), (129, BF), (961, BF), (63, F32), (129, F32)]


@pytest.mark.parametrize("hd,dtype", TAILS, ids=[f"hd{h}_{str(t)[6:]}" for h, t in TAILS])
def test_tail_classes_take_fx(hd, dtype, monkeypatch, ieee_fp32):
    """A tail head's no-grad forward runs mover_fx's bank-direct walk, deterministically, within the oracle gates."""
    b = _backend()
    import sdm_testlib as tl
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    P, T, W, R, sph, C = 2, 192, 4, 4, 256, 64
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=13, slots="zipf")
    assert not fx.rows16(d["memory"]) and fx.tail_supported(d["memory"], C)
    ref_y, _, ref_m = tl.oracle_write_read(d, None, device="cuda", sph=sph)
    made = _witness(monkeypatch)
    from sparse_delta_memory.memory_ops import UnsupportedConfigurationError
    monkeypatch.setattr(movers, "tail_admits", lambda *a, **k: False)          # no tail predicate: no native route
    with pytest.raises(UnsupportedConfigurationError) as err:
        _fwd_nograd(d, C, sph, P)
    assert err.value.tail_class == "P" and not made, (err.value.tail_class, made)
    monkeypatch.setattr(movers, "tail_admits", lambda bank, chunk_size, quant="none": quant == "none")
    y1, m1 = _fwd_nograd(d, C, sph, P)
    y2, m2 = _fwd_nograd(d, C, sph, P)
    assert made, "mover_fx did not take the tail head"
    assert _bits_eq(y1, y2) and _bits_eq(m1, m2), "repeats differ"
    ok_y, iy = _oracle_gate(y1.view(ref_y.shape), ref_y, dtype)
    ok_m, im = _oracle_gate(m1.view(ref_m.shape), ref_m, dtype)
    print(f"TAIL {b} hd {hd} {dtype}: float64-oracle gates readings {iy} bank {im}")
    assert ok_y and ok_m, (iy, im)


# The saving forward of tail heads (forward only; the predicate and the backward's admission forced): layout 2 with the
# compact rows written by the DW itself (WR), 16-warp launches (C 256 with 128-wide slices), partial last chunks, SCALAR
# lanes with 2 columns in range; saving = no-grad bitwise, repeats bitwise.
@pytest.mark.parametrize("hd,dtype,C,T", [(136, BF, 64, 200), (62, BF, 64, 200), (130, BF, 256, 300),
                                          (1028, BF, 256, 300), (129, F32, 64, 200), (320, F32, 256, 300)])
def test_tail_saving_forward_writes_compact_rows(hd, dtype, C, T, monkeypatch):
    _backend()
    _chunk_ok(C)
    from sparse_delta_memory import backward, movers
    from sparse_delta_memory.memory_ops import sdm_write_read
    P, W, R, sph = 2, 4, 4, 512
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=17, slots="zipf")
    monkeypatch.setattr(movers, "tail_admits", lambda bank, chunk_size, quant="none": quant == "none")
    monkeypatch.setattr(backward, "supported", lambda *a, **k: True)
    monkeypatch.setattr(backward, "admits_at_forward", lambda *a, **k: True)
    plans = _witness(monkeypatch)
    y0, m0 = _fwd_nograd(d, C, sph, P)
    outs = []
    for _ in range(2):
        leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
        out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                             d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph, num_parallel=P)
        torch.cuda.synchronize()
        outs.append((out, leaves["memory"].detach()))
    assert any(k.get("snapw") for k in plans), f"the saving forward's DW did not write the compact rows: {plans}"
    out, mem = outs[0]
    assert _finite(out.detach(), mem)
    assert _bits_eq(out.detach(), y0), "readings: saving != no-grad"
    assert _bits_eq(mem, m0), "bank: saving != no-grad"
    assert _bits_eq(outs[1][0].detach(), out.detach()) and _bits_eq(outs[1][1], mem), "saving repeats differ"
    fn = out.grad_fn
    while fn is not None and type(fn).__name__ != "SDMWriteReadBackward":
        fn = fn.next_functions[0][0]
    raw = fn.saved_tensors[11]
    assert fn.quant_layout == 2, "the saving forward did not save layout-2 compact rows"
    rows = raw.view(-1, hd)
    k = d["k_idx"].cpu()
    for p in range(P):                     # step 0's chunks: the pre-write rows are the initial bank rows
        slots = k[p, :C].unique()
        want = d["memory"][slots.to(d["memory"].device)].cpu()
        got = rows[p * C * W:p * C * W + slots.numel()].cpu()
        key = lambda t: sorted(bytes(r.contiguous().view(torch.uint8).numpy()) for r in t)
        assert key(got) == key(want), f"chunk {p}: compact rows != the groups' pre-write bank rows"

