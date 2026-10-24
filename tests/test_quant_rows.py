# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Quantized compact snapshot rows in gx_pa2_cuda's Phase A."""
import pytest
import torch

from sdm_testlib import bits_mismatch, requires_cuda
from test_mover_row_types import _assert_same, _phase1, _rows

def _gx_pa2_backend():
    """gx_pa2 (its quantized instantiations included) runs on every CUDA backend of the package (modern / a100 / rtx)."""
    if not torch.cuda.is_available():
        return False
    from sparse_delta_memory.cuda._device import cuda_backend
    try:
        return bool(cuda_backend())
    except RuntimeError:   # a GPU outside the package's classes
        return False


pytestmark = [requires_cuda, pytest.mark.skipif(not _gx_pa2_backend(), reason="needs a CUDA backend of the package")]

QMAX = {"fp8": 448.0, "int4": 7.0}


def quantize_ref(rows, quant):
    """The interface's encoder on rows [M, hd] (bank dtype)."""
    x = rows.float()
    inv = torch.tensor(1.0 / QMAX[quant], dtype=torch.float32, device=rows.device)
    scale = torch.clamp(x.abs().amax(-1) * inv, min=1e-12).to(torch.bfloat16).float()
    q = (x / scale[:, None]).to(torch.bfloat16)
    if quant == "fp8":
        return q.to(torch.float8_e4m3fn).view(torch.uint8).contiguous(), scale.contiguous()
    nib = torch.clamp(torch.round(q.float()), -8, 7).to(torch.int32) & 0xF
    return ((nib[:, 0::2] << 4) | nib[:, 1::2]).to(torch.uint8).contiguous(), scale.contiguous()


def dequant_ref(payload, scale, quant, dtype):
    """x' = dtype(f32(q) * f32(dtype(scale))) (gx_pa2's dec4; 2.0.0's phase2_kernels._snapshot_tile, the same)."""
    if quant == "fp8":
        q = payload.view(torch.float8_e4m3fn).to(dtype).float()
    else:
        p = payload.to(torch.int32)
        nib = torch.stack([p >> 4, p & 0xF], -1).reshape(p.shape[0], -1)
        q = torch.where(nib >= 8, nib - 16, nib).float()
    return (q * scale.to(dtype).float()[:, None]).to(dtype).contiguous()


def _payload(kind, quant, M, hd, gen):
    width = hd if quant == "fp8" else hd // 2
    if kind == "all_codes":   # every byte value, NaN codes (fp8 0x7F / 0xFF) excluded: compared bitwise
        p = torch.randint(0, 256, (M, width), device="cuda", generator=gen, dtype=torch.int32)
        p[0, : min(width, 256)] = torch.arange(min(width, 256), device="cuda") % 256
        if quant == "fp8":
            p = torch.where((p & 0x7F) == 0x7F, p ^ 0x01, p)
        return p.to(torch.uint8).contiguous()
    if kind == "nan_codes":   # all_codes plus fp8 NaN codes in every 7th row
        p = _payload("all_codes", quant, M, hd, gen)
        if quant == "fp8":
            p[::7, :4] = torch.tensor([0x7F, 0xFF, 0x7F, 0xFF], dtype=torch.uint8, device="cuda")
        return p
    raise ValueError(kind)


def _scales(M, gen):
    """bf16-valued f32 scales over 12 binades, one exact 1.0, one 1e-12 floor (the encoder's minimum)."""
    s = torch.exp2(torch.randint(-14, -2, (M,), device="cuda", generator=gen).float())
    s = (s * (1 + torch.rand(M, device="cuda", generator=gen))).to(torch.bfloat16).float()
    s[0], s[1 % M] = 1.0, torch.tensor(1e-12).to(torch.bfloat16).float()
    return s.contiguous()


def _quant_pair(bank_dtype, grad_dtype, quant, P, Nt, C, W, hd, pool, sph, seed=0, payload="encoder",
                valid_last=None, hi=0):
    """(unquantized gx_pa2 on dequant_ref rows, quantized gx_pa2 on payload + scales) over the reversed step loop."""
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import gx_pa2_cuda as gxpa2, pa_dc_cuda as dc
    p1, k_idx, _, k_val, _, beta = _phase1(P, Nt, C, W, 4, pool, sph, bank_dtype, seed, valid_last or C - 5, hi)
    N, n = Nt * P, C * W
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    bpre = p1.dc_bpre.reshape(N, -1)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    rows = P * sph
    if payload == "encoder":
        pay, scale = quantize_ref(_rows(N * n, hd, bank_dtype, gen), quant)
    elif payload == "inf_rows":   # rows decoding to +inf (code x scale +inf) among finite rows; positive operands
        width = hd if quant == "fp8" else hd // 2
        pay = torch.full((N * n, width), 0x38 if quant == "fp8" else 0x77, dtype=torch.uint8, device="cuda")
        scale = torch.ones(N * n, device="cuda")
        scale[::3] = float("inf")
    else:
        pay, scale = _payload(payload, quant, N * n, hd, gen), _scales(N * n, gen)
        if payload == "nan_codes":
            scale[5::11] = float("nan")
    snap = dequant_ref(pay, scale, quant, bank_dtype)
    if payload == "inf_rows":
        dva = torch.ones(N * C, hd, device="cuda")
        bank0 = torch.ones(rows, hd, device="cuda", dtype=bank_dtype)
        grad0 = torch.ones(rows, hd, device="cuda", dtype=grad_dtype)
    else:
        dva = torch.randn(N * C, hd, device="cuda", generator=gen)
        bank0 = _rows(rows, hd, bank_dtype, gen)
        grad0 = _rows(rows, hd, grad_dtype, gen, 0.01)
    nsl = dc.slices(hd)
    _, depth, nw = bw.pa_plan(C, bank_dtype, grad_dtype, bank0.device)
    beta_f = beta.reshape(-1).contiguous()
    L = gxpa2.tile_len(C, W, hd, P, dc.tile_len(C, W, hd, P), device=bank0.device)
    fn = gxpa2.launcher(pdl=False)
    outs = []
    for s_arg, q_arg in ((snap, torch.empty(0, device="cuda", dtype=grad_dtype)), (pay, scale)):
        assert gxpa2.supported(grad0, bank0, s_arg, C, 1792, depth, nw, False, scales=q_arg if q_arg.numel() else None)
        grad, bank = grad0.clone(), bank0.clone()
        eff = torch.full((nsl, N * n), float("nan"), device="cuda")
        gg = torch.full((nsl, N * n), float("nan"), device="cuda")
        # gx_pa2's own warps per CTA and ring depth for these rows (as the backward passes them; bits depend on neither)
        w_, d_ = gxpa2.plan(grad0, bank0, C, gxpa2.quant_of(bank0, s_arg))
        for ti in reversed(range(Nt)):
            fn(grad, bank, s_arg, P * C * W, dva, beta_f, kpay, rec, eff.view(-1) if nsl == 1 else eff,
               gg.view(-1) if nsl == 1 else gg, ti, P, C, W, L, w_, d_, 1792, True, bpre, q_arg)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "bank": bank, "eff": eff, "g": gg})
        del grad, bank
    return outs


Q_CASES = [   # (bank dtype, grad dtype, P, Nt, C, W, hd, pool): 32-bit bf16, W64 f32 grad rows, f32; 1 / 2 / 8 slices
    (torch.bfloat16, torch.bfloat16, 2, 3, 64, 8, 128, 48),
    (torch.bfloat16, torch.bfloat16, 2, 2, 16, 8, 64, 24),
    (torch.bfloat16, torch.float32, 2, 2, 128, 16, 128, 96),
    (torch.bfloat16, torch.bfloat16, 1, 2, 192, 8, 256, 160),
    (torch.float32, torch.float32, 2, 3, 64, 8, 64, 48),
    (torch.float32, torch.float32, 2, 2, 64, 8, 1024, 48),
    (torch.bfloat16, torch.float32, 1, 2, 32, 4, 1024, 40),
]


@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("payload", ["encoder", "all_codes"])
@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,W,hd,pool", Q_CASES,
                         ids=lambda v: str(v).replace("torch.", ""))
def test_quantized_rows_match_decoded_rows(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, quant, payload):
    from sparse_delta_memory.cuda import _device
    if _device.backend() == "rtx" and C > 128:
        pytest.skip("the RTX class refuses chunks above 128 (rule 0e)")
    ref, q = _quant_pair(bank_dtype, grad_dtype, quant, P, Nt, C, W, hd, pool, 4 * pool, payload=payload)
    _assert_same(ref, q, f"{quant} {payload} {bank_dtype}/{grad_dtype} C{C} hd{hd}")


def test_quantized_singleton_chunk():
    for quant in ("fp8", "int4"):
        ref, q = _quant_pair(torch.bfloat16, torch.bfloat16, quant, 2, 2, 1, 4, 128, 8, 32, valid_last=1)
        _assert_same(ref, q, f"{quant} C1")


@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("bank_dtype,grad_dtype,hd", [(torch.bfloat16, torch.bfloat16, 64),
                                                      (torch.bfloat16, torch.bfloat16, 144),
                                                      (torch.bfloat16, torch.float32, 64),
                                                      (torch.bfloat16, torch.float32, 144),
                                                      (torch.float32, torch.float32, 64),
                                                      (torch.float32, torch.float32, 144)],
                         ids=lambda v: str(v).replace("torch.", ""))
def test_quantized_off_slice_lanes_with_infinite_scales(bank_dtype, grad_dtype, hd, quant):
    """Slices narrower than the warp's 128 columns (hd < 128; hd 144 = 2 x 72)."""
    ref, q = _quant_pair(bank_dtype, grad_dtype, quant, 2, 2, 64, 8, hd, 48, 192, payload="inf_rows")
    for k in ref:
        assert torch.equal(ref[k].isnan(), q[k].isnan()), f"{quant} hd {hd}: {k} NaN positions differ"
        keep = dict(nan=0.0, posinf=float("inf"), neginf=float("-inf"))
        assert not bits_mismatch(torch.nan_to_num(ref[k], **keep), torch.nan_to_num(q[k], **keep)), \
            f"{quant} hd {hd}: {k} differs"
    assert ref["eff"].isinf().any(), "the fixture must reach infinite planes"


@pytest.mark.parametrize("quant", ["fp8", "int4"])
def test_quantized_nan_codes(quant):
    """NaN payload codes (fp8 0x7F / 0xFF; int4 has none) and NaN scales."""
    ref, q = _quant_pair(torch.bfloat16, torch.bfloat16, quant, 2, 2, 64, 8, 128, 48, 192, payload="nan_codes")
    for k in ref:
        a, b = ref[k], q[k]
        na, nb = torch.isnan(a.float()), torch.isnan(b.float())
        assert torch.equal(na, nb), f"{quant}: {k} NaN positions differ"
        assert na.any() or k == "grad", f"{quant}: {k} carries no NaN (the case is vacuous)"
        assert not bits_mismatch(torch.where(na, 0, a.float()), torch.where(nb, 0, b.float())), f"{quant}: {k}"


BIG_ROWS = (1 << 22) + 16   # x 1024 columns = 2^32 + 16,384 elements (bf16 8 GiB per bank)


@pytest.mark.slow
@pytest.mark.parametrize("quant", ["fp8", "int4"])
def test_quantized_rows_big_bank(quant):
    """A bf16 bank of 2^32 + 16,384 elements (W64 instantiation), half of the slots in its last 4096 rows."""
    from sdm_testlib import require_free_memory
    require_free_memory(64)
    ref, q = _quant_pair(torch.bfloat16, torch.bfloat16, quant, 1, 2, 64, 16, 1024, 4096, BIG_ROWS, hi=BIG_ROWS)
    _assert_same(ref, q, f"{quant} big")


# ------------------------------------------------------------------ the op: fx-q compact rows -> gx_pa2's quantized rows
@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("dtype,gbf32", [(torch.bfloat16, False), (torch.bfloat16, True), (torch.float32, False)],
                         ids=["bf16", "bf16_gbf32", "f32"])
def test_quantized_op_backward_matches_decoded_rows(quant, dtype, gbf32, monkeypatch):
    """The quantized op's CUDA backward equals the same backward on the decoded rows, bitwise."""
    import types
    from sdm_testlib import OP_INPUTS, make_op_inputs
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import _device, mover_fx_cuda as _fxm
    from sparse_delta_memory.memory_ops import sdm_write_read
    if quant == "fp8" and not hasattr(_fxm, "quant_supported") and tuple(_device.capability()) < (8, 9):
        # no fx producer in this tree: fp8 takes the Triton reference backward, whose fp8e4nv needs sm_89+ (A100 / RTX 30
        # have none); with the producer, fp8 rows go through gx_pa2's integer decode on every backend
        pytest.skip("fp8 reference backward needs fp8e4nv (sm_89+) and this tree has no fx producer")
    P, T, W, R, hd, sph, C = 2, 256, 8, 8, 128, 512, 64
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=3)
    cap, orig = {}, bw.sdm_backward

    def spy(ctx, saved, gr):
        cap.update(lay=ctx.lay, ql=getattr(ctx, "quant_layout", 0), inputs=ctx.inputs, gr=gr.clone(),
                   saved=[t.clone() if torch.is_tensor(t) else t for t in saved])
        n0, s0 = bw.STATS.get("gx_pa2", 0), bw.STATS["shipped"]
        res = orig(ctx, saved, gr)
        cap.update(grads=[t.clone() if torch.is_tensor(t) else t for t in res], bank=saved[0].clone(),
                   gx=bw.STATS.get("gx_pa2", 0) - n0, shipped=bw.STATS["shipped"] - s0)
        return res

    monkeypatch.setattr(bw, "sdm_backward", spy)
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                         d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph, num_parallel=P,
                         snapshot_quant=quant, grad_bank_fp32=gbf32)
    gen = torch.Generator(device="cuda").manual_seed(17)
    out.backward(torch.randn(out.shape, device="cuda", generator=gen).to(out.dtype))
    from sparse_delta_memory.cuda import mover_fx_cuda
    if not hasattr(mover_fx_cuda, "quant_supported"):
        # without the producer: the forward saved full metadata and the reference backward ran
        assert cap["ql"] == 0 and cap["gx"] == 0 and cap["shipped"] == 1, (cap["ql"], cap["gx"], cap["shipped"])
        return
    # with the producer: these shapes must take fx-q compact rows and gx_pa2's quantized instantiation, no reference
    assert cap["ql"] == 2 and cap["gx"] == 1 and cap["shipped"] == 0, (cap["ql"], cap["gx"], cap["shipped"])
    saved2 = cap["saved"]
    pay, scale = saved2[11], saved2[12]
    assert pay.dtype == (torch.float8_e4m3fn if quant == "fp8" else torch.uint8), pay.dtype
    assert pay.shape[1] == (hd if quant == "fp8" else hd // 2), pay.shape
    saved2[11] = dequant_ref(pay, scale, quant, dtype)
    saved2[12] = None                                  # 'none' saves no scale (its layout: ctx2.quant_layout)
    ctx2 = types.SimpleNamespace(lay=cap["lay"], snapshot_quant="none", quant_layout=cap["ql"], grad_bank_fp32=gbf32,
                                 inputs=cap["inputs"])
    grads2 = orig(ctx2, saved2, cap["gr"])
    for i, (a, b) in enumerate(zip(cap["grads"], grads2)):
        if torch.is_tensor(a):
            bad = bits_mismatch(a, b)
            assert not bad, f"{quant} {dtype}: grad {i} differs: {bad}"
    bad = bits_mismatch(cap["bank"], saved2[0])
    assert not bad, f"{quant} {dtype}: restored bank differs: {bad}"
