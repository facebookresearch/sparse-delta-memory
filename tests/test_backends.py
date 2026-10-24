# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The backends of the routing predicate (cuda/_device.backend): modern (sm_90 / 100 / 103), a100 (sm_80) and rtx."""
import os
import re
import shutil
import subprocess

import pytest
import torch

from sdm_testlib import bits_mismatch, requires_cuda

MODERN = ((9, 0), (10, 0), (10, 3))
A100 = ((8, 0), 166912, 108, 41943040)     # capability, opt-in smem per block, SMs, L2 bytes (rule 0e rows)
RTX = ((8, 6), 101376, 82, 6291456)        # RTX 3090


def _patch_device(monkeypatch, cc, optin, sms=None, l2=None):
    """One row of the hardware matrix through the state of the package's private forcing hook."""
    d = _dev()
    monkeypatch.setattr(d, "_FORCED", (tuple(cc), int(optin), d._props(d._index(None))[0] if sms is None else sms, l2))


def _dev():
    from sparse_delta_memory.cuda import _device
    return _device


# ---------------------------------------------------------------------------------------------------------------------
# The predicate and the plans (patched properties)
# ---------------------------------------------------------------------------------------------------------------------


UNSUPPORTED_MSG = (r"sparse_delta_memory supports NVIDIA H100 / B200 / GB300 \(compute capability 9\.0 / 10\.0 / 10\.3\), "
                   r"A100 \(8\.0\) and RTX 30 / 40 / 50 \(8\.6 / 8\.9 / 12\.0\); device 0 \(.+\) has compute "
                   r"capability {}\.{} with {} bytes of opt-in shared memory per block")


@requires_cuda
def test_backend_refusals(monkeypatch):
    """Every target outside the matrix (below 8.0, or a capability or shared-memory limit not listed)."""
    d = _dev()
    for cc, optin in (((7, 5), 65536), ((7, 0), 98304), ((8, 7), 166912), ((11, 0), 232448), ((12, 1), 101376),
                      ((9, 0), 101376), ((8, 0), 101376), ((8, 6), 166912), ((12, 0), 232448)):
        _patch_device(monkeypatch, cc, optin)
        with pytest.raises(RuntimeError, match="^" + UNSUPPORTED_MSG.format(cc[0], cc[1], optin) + "$"):
            d.backend(0)


# ---------------------------------------------------------------------------------------------------------------------
# The portable device side: compute_80 PTX builds against the native ones
# ---------------------------------------------------------------------------------------------------------------------

def _modern_gpu():
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) in MODERN


requires_modern = pytest.mark.skipif(not _modern_gpu(), reason="needs an H100 / B200 / GB300 (native = modern side)")
_VARIANTS = {}


def _images(so):
    """(ELF architectures, PTX architectures) embedded in an extension, from cuobjdump (required)."""
    tool = shutil.which("cuobjdump") or (os.path.join(os.environ["CUDA_HOME"], "bin", "cuobjdump")
                                         if os.environ.get("CUDA_HOME") else None)
    if not tool or not os.path.exists(tool):
        pytest.fail("cuobjdump not found (PATH / CUDA_HOME): the image checks are part of this evidence")
    elf = subprocess.run([tool, "--list-elf", so], capture_output=True, text=True, check=False).stdout
    ptx = subprocess.run([tool, "--list-ptx", so], capture_output=True, text=True, check=False).stdout
    return set(re.findall(r"\.(sm_\d+a?)\.cubin", elf)), set(re.findall(r"\.(sm_\d+a?)\.ptx", ptx))


def _variant(module, arch_list):
    """`module`'s extension built for TORCH_CUDA_ARCH_LIST=`arch_list` as its own module."""
    key = (module.__name__, arch_list)
    if key not in _VARIANTS:
        from sparse_delta_memory.cuda import _image
        with _image._unchecked_build_env():   # TEST ONLY: a build for another target list (_image's frozen environment)
            prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
            try:
                _VARIANTS[key] = module._module.__wrapped__()
            finally:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _VARIANTS[key]


def _native(module):
    """The modern arm: built explicitly for this GPU's own architecture only (no PTX); checked to hold that cubin."""
    cc = torch.cuda.get_device_capability()
    mod = _variant(module, f"{cc[0]}.{cc[1]}")
    im = _images(mod.__file__)
    assert im == ({f"sm_{cc[0]}{cc[1]}"}, set()), (mod.__file__, im)
    return mod


def _portable(module):
    """The portable arm: compute_80."""
    mod = _variant(module, "8.0+PTX")
    im = _images(mod.__file__)
    assert im == ({"sm_80"}, {"sm_80"}), (mod.__file__, im)
    return mod


def _same(a, b, what):
    for k in a:
        bad = bits_mismatch(a[k], b[k])
        assert not bad, f"{what}: {k} differs: {bad}"


def _tmr():
    import test_mover_row_types as tmr   # Phase-1 inputs and row helpers of the movers' row-type tests
    return tmr


GDS_CASES = [   # (bank dtype, grad dtype, P, Nt, C, R, hd, pool, nw, special values)
    (torch.bfloat16, torch.bfloat16, 2, 3, 64, 8, 128, 48, 8, False),       # 32-bit offsets, one slice (SW == hd)
    (torch.bfloat16, torch.bfloat16, 2, 2, 256, 4, 128, 96, 16, False),     # token rows > 96 KB: 16 warps
    (torch.bfloat16, torch.bfloat16, 2, 2, 64, 16, 960, 64, 4, False),      # 8 slices of 120 columns (strided rows)
    (torch.bfloat16, torch.bfloat16, 2, 2, 64, 8, 128, 32, 8, True),        # -0.0, subnormal, +-inf, NaN rows
    (torch.bfloat16, torch.float32, 2, 3, 64, 8, 64, 48, 8, False),         # grad_bank_fp32 rows
    (torch.float32, torch.float32, 2, 2, 128, 16, 1024, 96, 16, False),     # f32 rows, 8 slices
    (torch.float32, torch.float32, 2, 2, 32, 4, 100, 32, 4, False),         # 100-column (400-byte) rows
    (torch.float32, torch.float32, 1, 1, 1, 1, 64, 4, 8, False),            # one entry
]


@requires_cuda
@requires_modern
@pytest.mark.parametrize("rec8", [False, True], ids=["qrec16", "rec8"])
@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,R,hd,pool,nw,special", GDS_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_R{c[5]}_hd{c[6]}_nw{c[8]}{'_special' if c[9] else ''}"
                              for c in GDS_CASES])
def test_portable_gx_gds_matches_modern(bank_dtype, grad_dtype, P, Nt, C, R, hd, pool, nw, special, rec8, monkeypatch):
    """gx_gds' portable side (cooperative cp.async staging + CTA barrier) gives the modern kernel's bits."""
    tmr = _tmr()
    from sparse_delta_memory.cuda import gx_gds_cuda as gxm
    vl = max(1, C - 3)
    p1, _, q_idx, _, q_val, _ = tmr._phase1(P, Nt, C, 4, R, pool, 4 * pool, bank_dtype, 0, vl)
    N, n = Nt * P, C * R
    gen = torch.Generator(device="cuda").manual_seed(5)
    bank = tmr._rows(P * 4 * pool, hd, bank_dtype, gen)
    grad0 = tmr._rows(P * 4 * pool, hd, grad_dtype, gen, 0.01)
    if special:
        tmr._inject_special(bank, 4 * pool), tmr._inject_special(grad0, 4 * pool)
    gr = torch.randn(N * C, hd, device="cuda", generator=gen)
    recs = p1.qrec8.reshape(N, n, 2) if rec8 else p1.qrec.reshape(N, n, 4)
    native, mod80 = _native(gxm), _portable(gxm)
    ppw = {4: 32, 8: 128, 16: 256}[nw]
    monkeypatch.setattr(gxm, "geometry", lambda *a, **k: (nw, ppw))
    # make_gds's H100 plan rule (float32 grad rows at (8, 128) / (16, 256) take (8, 64) on sm_90) would
    # replace the forced geometry there; a non-H100 capability keeps the forced warps, so the portable / modern A/B of
    # every instantiation (16-warp float32 rows included) stays covered on H100 too (as test_mover_row_types does).
    monkeypatch.setattr(gxm, "capability", lambda device=None: (10, 0))
    outs = []
    for mod in (native, mod80):
        monkeypatch.setattr(gxm, "_module", lambda m=mod: m)
        grad = grad0.clone()
        dqw = torch.full((N, C, R), float("nan"), device="cuda")
        launch = gxm.make_gds(bank, grad, gr, p1.spay_q, recs, dqw, chunk_size=C, num_parallel=P, steps=Nt,
                              q_idx=q_idx, valid_tokens=(Nt - 1) * C + vl, tiles_q=None)
        assert launch.geom["impl"] == "gx" and launch.geom["nw"] == nw
        for ti in reversed(range(Nt)):
            launch(ti)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "d_q_w": dqw})
    _same(outs[0], outs[1], "gx_gds portable vs modern")
    assert special or not outs[0]["d_q_w"].isnan().any()


PA_CASES = [   # (row, bank dtype, grad dtype, P, Nt, C, W, hd, pool): the row's plan for each
    ("a100", torch.bfloat16, torch.bfloat16, 2, 2, 256, 4, 128, 96),     # 7 warps
    ("a100", torch.bfloat16, torch.float32, 2, 2, 256, 4, 128, 96),      # 5 warps
    ("a100", torch.float32, torch.float32, 2, 2, 256, 4, 128, 96),       # 4 warps, 166,912 B: the A100 limit
    ("a100", torch.bfloat16, torch.bfloat16, 2, 3, 64, 8, 960, 64),      # 8 warps, depth 8, 8 slices
    ("a100", torch.bfloat16, torch.bfloat16, 2, 3, 16, 4, 128, 16),      # depth 16
    ("a100", torch.bfloat16, torch.bfloat16, 1, 2, 160, 8, 128, 64),     # 16 warps
    ("a100", torch.bfloat16, torch.float32, 2, 3, 64, 8, 128, 64),       # 16 warps: f32 grad rows (2 x 8 warps > SM)
    ("a100", torch.float32, torch.float32, 2, 2, 32, 8, 64, 32),         # 16 warps: f32 rows
    ("rtx", torch.bfloat16, torch.bfloat16, 2, 2, 128, 8, 128, 96),      # 7 warps
    ("rtx", torch.float32, torch.float32, 2, 2, 128, 8, 64, 96),         # 4 warps
    ("rtx", torch.bfloat16, torch.float32, 2, 2, 96, 8, 256, 64),        # 7 warps, 2 slices
]


@requires_cuda
@requires_modern
@pytest.mark.parametrize("row,bank_dtype,grad_dtype,P,Nt,C,W,hd,pool", PA_CASES,
                         ids=[f"{c[0]}_{str(c[1])[6:]}_{str(c[2])[6:]}_C{c[5]}_hd{c[7]}" for c in PA_CASES])
def test_portable_gx_pa2_matches_dc_pa3(row, bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, monkeypatch):
    """gx_pa2 built for the portable side, with the row's plan, against dc_pa3 (native): the same bits."""
    tmr = _tmr()
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import gx_pa2_cuda as gxpa2, pa_dc_cuda as dc, phase1_cuda as ph
    dc_native = _native(dc)
    pa80 = _portable(gxpa2)
    p1, k_idx, _, k_val, _, beta = tmr._phase1(P, Nt, C, W, 4, pool, 4 * pool, bank_dtype, 3, C - 5)
    _patch_device(monkeypatch, *(A100 if row == "a100" else RTX))
    N, n = Nt * P, C * W
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    bpre = p1.dc_bpre.reshape(N, -1)
    gen = torch.Generator(device="cuda").manual_seed(4)
    snap = tmr._rows(N * n, hd, bank_dtype, gen)
    dva = torch.randn(N * C, hd, device="cuda", generator=gen)
    bank0 = tmr._rows(P * 4 * pool, hd, bank_dtype, gen)
    grad0 = tmr._rows(P * 4 * pool, hd, grad_dtype, gen, 0.01)
    nsl = dc.slices(hd)
    _, depth, nw = bw.pa_plan(C, bank_dtype, grad_dtype, bank0.device)
    nw_g, depth_g = gxpa2.plan(grad0, bank0, C)
    assert gxpa2.supported(grad0, bank0, snap, C, 1792, depth, nw, False)
    empty = torch.empty(0, device="cuda", dtype=grad_dtype)
    beta_f = beta.reshape(-1).contiguous()
    outs = []
    for use_gx in (False, True):
        grad, bank = grad0.clone(), bank0.clone()
        eff = torch.full((nsl, N * n), float("nan"), device="cuda")
        gg = torch.full((nsl, N * n), float("nan"), device="cuda")
        L = dc.tile_len(C, W, hd, P)
        fn, w_, d_ = dc_native.dc_pa, nw, depth
        if use_gx:
            fn, w_, d_ = pa80.gx_pa2, nw_g, depth_g
            L = gxpa2.tile_len(C, W, hd, P, L, device=bank.device)
        for ti in reversed(range(Nt)):
            fn(grad, bank, snap, P * C * W, dva, beta_f, kpay, rec, eff.view(-1) if nsl == 1 else eff,
               gg.view(-1) if nsl == 1 else gg, ti, P, C, W, L, w_, d_, 1792, True, bpre, empty)
        torch.cuda.synchronize()
        outs.append({"grad": grad, "bank": bank, "eff": eff, "g": gg})
    _same(outs[0], outs[1], f"gx_pa2 portable ({row} plan {nw_g} warps, depth {depth_g}) vs dc_pa3")


@requires_cuda
@requires_modern
@pytest.mark.parametrize("row", ["a100", "rtx"])
@pytest.mark.parametrize("C,W,R", [(64, 16, 16), (128, 32, 8), (256, 8, 8), (16, 64, 64), (128, 112, 112)])
def test_portable_phase1_ordered_matches_native(row, C, W, R):
    """The ordered Phase 1 (modern=False: the A100 / RTX route) at the row's shared-memory limit and SM count on the
    portable side gives the native build's bits at the same limit (the limit picks the pair kernel's lean / split
    forms)."""
    from sparse_delta_memory.cuda import phase1_cuda as ph
    from sparse_delta_memory import forward as fw
    _, limit, sms, _ = A100 if row == "a100" else RTX
    if row == "rtx" and C > 128:
        pytest.skip("RTX parts run chunks of at most 128 tokens")
    native = _native(ph)
    mod80 = _portable(ph)
    P, Nt, sph = 2, 3, 1024
    gen = torch.Generator(device="cuda").manual_seed(C + W)
    N = P * Nt
    stream = (torch.arange(N, device="cuda") % P).view(N, 1, 1) * sph

    def ids(width):
        return (torch.rand(N * C, sph // 4, device="cuda", generator=gen).topk(width, -1).indices.sort(-1).values
                .view(N, C, width) + stream).int()
    k_idx, q_idx = ids(W), ids(R)
    k_val = (torch.randn(N, C, W, device="cuda", generator=gen) * 0.3).bfloat16()
    q_val = (torch.randn(N, C, R, device="cuda", generator=gen) * 0.3).bfloat16()
    g = torch.rand(N, C, 1, device="cuda", generator=gen) * -0.49 - 0.01
    beta = torch.rand(N, C, device="cuda", generator=gen) * 0.3
    fields = fw._SAVING_FIELDS | ph.DET_QREC8
    outs = []
    for mod, modern in ((native, False), (mod80, False)):
        rep = torch.zeros(1, device="cuda", dtype=torch.int64)
        res = mod.p1_forward(k_idx, q_idx, k_val, q_val, g, rep, P, sph, (sph - 1).bit_length(), C - 7, True, limit,
                             beta, 128, fields, modern, sms)
        torch.cuda.synchronize()
        outs.append({f"{i}": (t if t is not None and t.numel() else torch.zeros(0)) for i, t in enumerate(res)}
                    | {"rep": rep})
    _same(outs[0], outs[1], f"ordered Phase 1 portable vs native ({row} limit {limit})")


@requires_cuda
@requires_modern
@pytest.mark.parametrize("dtype,hd,W", [(torch.bfloat16, 1024, 64), (torch.float32, 128, 16), (torch.bfloat16, 120, 8)])
def test_portable_dc_gwr_matches_native(dtype, hd, W):
    from sparse_delta_memory import memory_ops as mo
    from sparse_delta_memory.cuda import pa_dc_cuda as dc
    native, mod80 = _native(dc), _portable(dc)
    P, C, Nt, S = 2, 64, 2, 4096
    SC, NC = P * C, Nt * P * C
    gen = torch.Generator(device="cuda").manual_seed(hd + W)
    bank = (torch.randn(S, hd, device="cuda", generator=gen) * 0.1).to(dtype)
    idx = torch.rand(NC, S, device="cuda", generator=gen).topk(W, -1).indices.int().contiguous()
    w = torch.randn(NC, W, device="cuda", generator=gen)
    dv = torch.randn(NC, hd, device="cuda", generator=gen)
    geo = mo.pinned_gather_geometry(hd, dtype, W, SC)
    outs = []
    for mod in (native, mod80):
        out = torch.full((NC, hd), float("nan"), device="cuda")
        pd = torch.full((geo.slices, NC * W) if geo.slices > 1 else (NC * W,), float("nan"), device="cuda")
        for ti in reversed(range(Nt)):
            mod.dc_gwr(bank, idx, w, out, dv, pd, ti * SC, ti * SC, SC, C, C - (5 if ti == Nt - 1 else 0),
                       geo.vec_bytes, geo.lanes, geo.groups, geo.tpc, torch.empty(0, device="cuda", dtype=torch.int32),
                       torch.empty(0, device="cuda", dtype=dtype), 0)
        torch.cuda.synchronize()
        outs.append({"out": out, "pd": pd})
    _same(outs[0], outs[1], "dc_gwr portable vs native")


@requires_cuda
@requires_modern
@pytest.mark.parametrize("C,hd,P", [(64, 1024, 2), (256, 128, 2), (37, 100, 3)])
def test_portable_gx_gemm_matches_native(C, hd, P):
    from sparse_delta_memory.cuda import gx_gemm_cuda as gg
    native, mod80 = _native(gg), _portable(gg)
    Nt = 2
    N = P * Nt
    gen = torch.Generator(device="cuda").manual_seed(C)
    x = torch.randn(N, C, C, device="cuda", generator=gen)
    dv = torch.randn(N * C, hd, device="cuda", generator=gen)
    du0 = torch.randn(N, C, hd, device="cuda", generator=gen)
    outs = []
    for mod in (native, mod80):
        du = du0.clone()
        for ti in reversed(range(Nt)):
            mod.gx_gemm(du, x, dv, ti, P, 0, 32, 0)
        torch.cuda.synchronize()
        outs.append({"du": du})
    _same(outs[0], outs[1], "gx_gemm portable vs native")


@requires_cuda
@pytest.mark.parametrize("cc,optin", [((7, 5), 65536), ((8, 0), 101376), ((12, 1), 101376)])
def test_unsupported_targets_refused_at_the_op_before_any_work(monkeypatch, cc, optin):
    """The public op on a target outside the matrix: the frozen RuntimeError before any kernel runs."""
    from sparse_delta_memory import memory_ops as mo
    from sdm_testlib import make_op_inputs
    d = make_op_inputs(1, 64, 4, 4, 64, 256, torch.bfloat16, seed=1)
    mem = d["memory"].clone()
    before = mem.clone()
    _patch_device(monkeypatch, cc, optin)
    with pytest.raises(RuntimeError, match="^" + UNSUPPORTED_MSG.format(cc[0], cc[1], optin) + "$"):
        mo.sdm_write_read(mem, d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"],
                          chunk_size=64, num_memory_slots=256, num_parallel=1)
    torch.cuda.synchronize()
    assert torch.equal(mem.view(torch.int16), before.view(torch.int16))


# ---------------------------------------------------------------------------------------------------------------------
# The copy layer on its own (portable passes only: built as compute_80 PTX)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# RTX class refusals and limits (host decisions under the forced row)
# ---------------------------------------------------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("C", [256, 192, 129])
def test_rtx_refuses_chunks_above_128_before_any_work(monkeypatch, C):
    """RTX parts: chunk_size above 128 raises the frozen ValueError naming 128 before any kernel runs."""
    from sparse_delta_memory import memory_ops as mo
    from sdm_testlib import make_op_inputs
    d = make_op_inputs(1, 64, 4, 4, 64, 256, torch.bfloat16, seed=1)
    mem = d["memory"].clone()
    before = mem.clone()
    _patch_device(monkeypatch, *RTX)
    msg = (r"^chunk_size {} is not supported on RTX 30 / 40 / 50 GPUs: their 99 KB of shared memory per block allow "
           r"chunk_size <= 128$").format(C)
    with pytest.raises(ValueError, match=msg):
        mo.sdm_write_read(mem, d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"],
                          chunk_size=C, num_memory_slots=256, num_parallel=1)
    torch.cuda.synchronize()
    assert torch.equal(mem.view(torch.int16), before.view(torch.int16))
    lay = mo._layout(mem, d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"], 128, 256, 1)
    assert lay.C == 128


@requires_cuda
def test_a100_and_modern_keep_chunk_256(monkeypatch):
    from sparse_delta_memory import memory_ops as mo
    from sdm_testlib import make_op_inputs
    d = make_op_inputs(1, 64, 4, 4, 64, 256, torch.bfloat16, seed=1)
    from sparse_delta_memory.cuda import _image
    # the chunk decision of each forced row, on any GPU: the loaded-image guard would (rightly) refuse a forced modern row
    # over a real compute_80 image; the guard itself is tested in tests/test_image_refusal.py
    monkeypatch.setattr(_image, "require_supported_image", lambda device=None: None)
    for row in (A100, ((10, 3), 232448)):
        _patch_device(monkeypatch, *row)
        lay = mo._layout(d["memory"], d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"], 256,
                         256, 1)
        assert lay.C == 256


# ---------------------------------------------------------------------------------------------------------------------
# Route witness (a Triton launch hook turns the fused C++ step loop off and biases instrumented runs)
# ---------------------------------------------------------------------------------------------------------------------

