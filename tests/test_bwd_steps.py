# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The backward's step loop as ONE C++ call."""
import re

import pytest
import torch

from sdm_testlib import OP_INPUTS, bits_mismatch, make_op_inputs, requires_cuda

pytestmark = [requires_cuda]

CASES = [   # (P, T, W, R, hd, sph, C, dtype, grad_bank_fp32, snapshot_quant)
    (2, 256, 16, 16, 128, 4096, 64, torch.bfloat16, False, "none"),
    (2, 200, 16, 16, 128, 4096, 64, torch.bfloat16, False, "none"),       # partial last chunk
    (1, 64, 8, 8, 64, 1024, 64, torch.bfloat16, False, "none"),           # a single step
    (1, 40, 8, 8, 64, 1024, 64, torch.bfloat16, False, "none"),           # a single partial step
    (4, 128, 16, 16, 128, 4096, 64, torch.bfloat16, False, "none"),       # P = 4
    (2, 512, 16, 16, 128, 4096, 32, torch.bfloat16, False, "none"),       # C = 32
    (2, 512, 16, 16, 128, 4096, 128, torch.bfloat16, False, "none"),      # C = 128
    (1, 512, 8, 8, 128, 4096, 256, torch.bfloat16, False, "none"),        # C = 256
    (2, 256, 16, 16, 96, 4096, 64, torch.bfloat16, False, "none"),        # a 96-column head
    (2, 256, 16, 16, 1024, 4096, 64, torch.bfloat16, False, "none"),      # 8 column slices
    (2, 256, 16, 16, 128, 4096, 64, torch.bfloat16, True, "none"),        # fp32 grad bank
    (2, 256, 16, 16, 128, 4096, 64, torch.float32, False, "none"),        # float32 bank
    (2, 256, 16, 16, 130, 4096, 64, torch.bfloat16, False, "none"),       # hd % 4 != 0: tail module
    (2, 256, 16, 16, 132, 4096, 64, torch.bfloat16, False, "none"),       # bf16 rows in 8-byte pieces (tail module)
    (2, 256, 16, 16, 66, 4096, 64, torch.float32, False, "none"),         # float32, hd % 4 != 0
    (2, 256, 16, 16, 20000, 512, 64, torch.float32, False, "none"),       # hd > 16384 (native)
    (2, 256, 16, 16, 128, 4096, 64, torch.bfloat16, False, "fp8"),
    (2, 256, 16, 16, 256, 4096, 64, torch.bfloat16, False, "int4"),
]
IDS = [f"P{c[0]}_T{c[1]}_C{c[6]}_hd{c[4]}_{str(c[7])[6:]}{'_gf32' if c[8] else ''}_{c[9]}" for c in CASES]


def _run(d, P, sph, C, grad_bank_fp32, quant, cot_seed=11):
    """sdm_write_read forward + backward on private copies of d: readings, banks, input gradients."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                         d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph, num_parallel=P,
                         snapshot_quant=quant, grad_bank_fp32=grad_bank_fp32)
    res = {"readings": out.detach().clone(), "bank_fwd": leaves["memory"].detach().clone()}
    gen = torch.Generator(device="cuda").manual_seed(cot_seed)
    out.backward(torch.randn(out.shape, device="cuda", generator=gen).to(out.dtype))
    torch.cuda.synchronize()
    res["bank_bwd"] = leaves["memory"].detach().clone()
    for n in OP_INPUTS:
        res["grad_" + n] = leaves[n].grad
    return res


def _spy(monkeypatch):
    """Records which Phase-A plan each C++ step loop built: ("gx_pa2", pdl variant) or ("dc_pa3", False)."""
    from sparse_delta_memory.cuda import gx_pa2_cuda as gxpa2, pa_dc_cuda as dc
    seen = []
    orig_g, orig_d = gxpa2.step_plan, dc.pa_step_plan

    def g(launch, *a, **k):   # the entry's own launch mode (regular / bf16 entry, regular / tail module)
        e = gxpa2._entry(launch)
        seen.append(("gx_pa2", bool(e is not None and e[1])))
        return orig_g(launch, *a, **k)

    def dd(*a, **k):
        seen.append(("dc_pa3", False))
        return orig_d(*a, **k)
    monkeypatch.setattr(gxpa2, "step_plan", g)
    monkeypatch.setattr(dc, "pa_step_plan", dd)
    return seen


def _qual_spy(monkeypatch):
    """Records the backward's GEMM decisions: gx_gemm_cuda.supported and matches_cublas results, in call order."""
    from sparse_delta_memory.cuda import gx_gemm_cuda as gxgemm
    seen = []
    orig_s, orig_m = gxgemm.supported, gxgemm.matches_cublas

    def s_(*a, **k):
        r = orig_s(*a, **k)
        seen.append(("supported", bool(r)))
        return r

    def m_(*a, **k):
        r = orig_m(*a, **k)
        seen.append(("matches_cublas", bool(r)))
        return r
    monkeypatch.setattr(gxgemm, "supported", s_)
    monkeypatch.setattr(gxgemm, "matches_cublas", m_)
    return seen


def _gemm_domain(P, C, hd):
    """The shapes on which the backward's d_u update must run gx_gemm's qualified GEMM on this device."""
    from sparse_delta_memory.cuda import _device, gx_gemm_cuda as gxgemm
    sms = _device.num_sms(torch.device("cuda"))
    target = 32 * -(-sms // 32)

    def grid(ti):
        return P * -(-C // ti) * -(-hd // 64)
    ti = next((t for t in (16, 32, 48) if grid(t) <= target), 64)
    ours = P >= 2 and C >= 2 and grid(ti) <= sms
    assert ours == (P >= 2 and C >= 2 and gxgemm.one_wave(P, C, hd, torch.device("cuda"))), (P, C, hd, sms, ti)
    return ours


def _compare(got, again, ref):
    for k in ref:
        assert not bits_mismatch(ref[k], got[k]), f"{k}: C++ step loop != per-step loop: {bits_mismatch(ref[k], got[k])}"
        assert not bits_mismatch(got[k], again[k]), f"{k}: C++ step loop repeat differs"


@pytest.mark.parametrize("gemm", ["auto", "cublas"])
@pytest.mark.parametrize("P,T,W,R,hd,sph,C,dtype,gf32,quant", CASES, ids=IDS)
def test_step_loop_cpp_matches_the_per_step_loop(P, T, W, R, hd, sph, C, dtype, gf32, quant, gemm, monkeypatch,
                                                ieee_fp32):
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import _device, gx_gemm_cuda as gxgemm
    if gemm == "cublas":                                       # no qualified GEMM: cuBLAS baddbmm, Phase A plain
        monkeypatch.setattr(bw, "_fp32_matmul_ieee", lambda: False)
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=5)
    from sparse_delta_memory import memory_ops as mo
    if _device.backend(torch.device("cuda")) == "rtx" and C > mo.RTX_MAX_CHUNK:
        # the RTX validator refuses chunks above 128 up front (rule 0e), with its own message
        with pytest.raises(ValueError, match=re.escape(mo.RTX_CHUNK_MSG.format(chunk_size=C))):
            _run(d, P, sph, C, gf32, quant)
        return
    seen = _spy(monkeypatch)
    qual = _qual_spy(monkeypatch)
    dc0, cpp0, py0, gm0 = (bw.STATS["dc"], bw.STATS.get("steps_cpp", 0), bw.STATS.get("steps_python", 0),
                           bw.STATS.get("gx_gemm", 0))
    ds0 = gxgemm.QUAL_STATS["declined_shape"]                  # matches_cublas's speed-rule declines (before any probe)
    got = _run(d, P, sph, C, gf32, quant)
    if bw.STATS["dc"] == dc0:
        # every case is a CUDA-backward case on a CUDA backend (a native decline must fail, not skip)
        assert not _device.cuda_backend(torch.device("cuda")), "a case the CUDA backward should take was declined"
        pytest.skip("no CUDA backend on this device")
    assert bw.STATS.get("steps_cpp", 0) == cpp0 + 1 and bw.STATS.get("steps_python", 0) == py0, \
        "the CUDA backward did not take the C++ step loop"
    assert bw.LAST_LOOP["name"] == "eager_cpp_step_loop", bw.LAST_LOOP
    used_gemm = bw.STATS.get("gx_gemm", 0) > gm0
    shape_declines = gxgemm.QUAL_STATS["declined_shape"] - ds0
    if gemm == "cublas" or P < 2 or C < 2:
        # the backward's prefilters (IEEE fp32 matmuls, P >= 2, C >= 2) keep cuBLAS without any GEMM decision
        assert qual == [] and not used_gemm and shape_declines == 0, (qual, used_gemm, shape_declines)
    elif _gemm_domain(P, C, hd):
        # the qualified GEMM must actually run ("auto" must not pass with nothing qualified): the shape
        # check admits it and the run-time qualification probe accepts it on this device
        assert qual == [("supported", True), ("matches_cublas", True)] and used_gemm and shape_declines == 0, \
            (qual, used_gemm, shape_declines)
    else:
        # outside the speed rule (e.g. C64 hd20000 on GB300) matches_cublas declines before any probe and the
        # backward keeps cuBLAS
        assert qual == [("supported", True), ("matches_cublas", False)] and not used_gemm and shape_declines == 1, \
            (qual, used_gemm, shape_declines)
    assert len(seen) == 1 and seen[0][0] in ("gx_pa2", "dc_pa3"), seen   # one Phase-A plan per backward
    if gemm == "cublas" or seen[0][0] == "dc_pa3":
        assert seen[0][1] is False, (used_gemm, seen)            # plain Phase-A launches
        if gemm == "cublas":
            assert not used_gemm, used_gemm
    else:   # Phase A launches programmatically exactly when the qualified GEMM precedes it on a PDL-capable device
        assert seen[0][1] == (used_gemm and _device.pdl_ok(torch.device("cuda"))), (used_gemm, seen)
    again = _run(d, P, sph, C, gf32, quant)
    with monkeypatch.context() as m:
        m.setattr(bw, "_steps_cpp", lambda **kw: False)        # the per-step loop
        ref = _run(d, P, sph, C, gf32, quant)
    _compare(got, again, ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("ti", [16, 32, 48, 64])
def test_gemm_rows_per_cta_each_specialization(ti, dtype, monkeypatch, ieee_fp32):
    """gx_gemm's four specializations."""
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.cuda import gx_gemm_cuda as gxgemm
    P, T, W, R, hd, sph, C = 2, 256, 16, 16, 128, 4096, 64
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=12)
    default = _run(d, P, sph, C, False, "none")
    monkeypatch.setattr(gxgemm, "_rows", lambda *a, **k: ti)
    # the PRODUCTION factory's geometry: the rows rule also runs inside the qualification (one_wave) and
    # the probe's own factory calls (made with _probe_counts), so the selector's call count is not asserted
    prod, orig_make = [], gxgemm.make_step

    def make(*a, **k):
        step = orig_make(*a, **k)
        if k.get("_probe_counts", a[4] if len(a) > 4 else None) is None:
            prod.append(step.geom["ti_rows"])
        return step
    monkeypatch.setattr(gxgemm, "make_step", make)
    gm0, cpp0 = bw.STATS.get("gx_gemm", 0), bw.STATS.get("steps_cpp", 0)
    got = _run(d, P, sph, C, False, "none")
    if not _device_cuda():
        pytest.skip("no CUDA backend on this device")
    assert bw.STATS.get("gx_gemm", 0) == gm0 + 1 and bw.STATS.get("steps_cpp", 0) == cpp0 + 1 and prod == [ti], \
        (prod, bw.STATS.get("gx_gemm", 0) - gm0)
    with monkeypatch.context() as m:
        m.setattr(bw, "_steps_cpp", lambda **kw: False)        # the per-step loop, same TI
        ref = _run(d, P, sph, C, False, "none")
    for k in ref:
        assert not bits_mismatch(ref[k], got[k]), f"TI {ti} {k}: C++ loop != per-step loop: {bits_mismatch(ref[k], got[k])}"
        assert not bits_mismatch(default[k], got[k]), f"TI {ti} {k}: differs from the default TI's bits"


def _device_cuda():
    from sparse_delta_memory.cuda import _device
    return _device.cuda_backend(torch.device("cuda"))


def test_captured_forward_backward_replays_eager_bits(ieee_fp32):
    """forward + backward captured in one CUDA graph."""
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.memory_ops import sdm_write_read
    P, T, W, R, hd, sph, C = 2, 256, 16, 16, 128, 4096, 64
    dtype = torch.bfloat16
    ref_inputs = [make_op_inputs(P, T, W, R, hd, sph, dtype, seed=s) for s in (21, 22, 23)]
    static = {k: v.detach().clone() for k, v in ref_inputs[0].items()}
    gy = torch.zeros(P * T, hd, device="cuda", dtype=dtype)
    leaves = {n: static[n].requires_grad_(True) for n in OP_INPUTS}

    def step():
        out = sdm_write_read(leaves["memory"], static["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"],
                             leaves["g"], static["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=sph,
                             num_parallel=P)
        out.backward(gy.view(out.shape))
        return out

    def load(d, seed):
        with torch.no_grad():
            for k, v in d.items():
                static[k].copy_(v)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            gy.copy_(torch.randn(gy.shape, device="cuda", generator=gen).to(dtype))
        for n in OP_INPUTS:
            leaves[n].grad = None
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            load(ref_inputs[0], 1)
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    load(ref_inputs[0], 1)
    cpp0 = bw.STATS.get("steps_cpp", 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    assert bw.STATS.get("steps_cpp", 0) == cpp0 + 1, "the captured backward did not take the C++ step loop"
    grads = {n: leaves[n].grad for n in OP_INPUTS}
    for i, d in enumerate(ref_inputs[1:]):
        with torch.no_grad():
            for k, v in d.items():
                static[k].copy_(v)
            gen = torch.Generator(device="cuda").manual_seed(100 + i)
            gy.copy_(torch.randn(gy.shape, device="cuda", generator=gen).to(dtype))
        graph.replay()
        torch.cuda.synchronize()
        got = {"readings": out.detach().clone(), "bank": leaves["memory"].detach().clone(),
               **{"grad_" + n: grads[n].detach().clone() for n in OP_INPUTS}}
        e_leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
        e_out = sdm_write_read(e_leaves["memory"], d["k_idx"], e_leaves["k_val"], e_leaves["v"], e_leaves["beta"],
                               e_leaves["g"], d["q_idx"], e_leaves["q_val"], chunk_size=C, num_memory_slots=sph,
                               num_parallel=P)
        gen = torch.Generator(device="cuda").manual_seed(100 + i)
        e_out.backward(torch.randn(e_out.shape, device="cuda", generator=gen).to(dtype))
        torch.cuda.synchronize()
        ref = {"readings": e_out.detach(), "bank": e_leaves["memory"].detach(),
               **{"grad_" + n: e_leaves[n].grad for n in OP_INPUTS}}
        for k in ref:
            assert not bits_mismatch(ref[k], got[k]), f"replay {i}: {k}: {bits_mismatch(ref[k], got[k])}"


@pytest.mark.slow
def test_big_bank_cpp_loop_matches_the_per_step_loop(monkeypatch, ieee_fp32):
    """A float32 bank of 2^32 + 16,384 elements (half of the slots in its last 4096 rows)."""
    if torch.cuda.get_device_properties(0).total_memory < 160 * 2 ** 30:
        pytest.skip("needs >= 160 GiB of device memory (two runs of a 16 GiB bank with its gradient)")
    import test_mover_row_types as tmr
    from sparse_delta_memory import backward as bw
    from sparse_delta_memory.memory_ops import sdm_write_read
    hd, T, W, C = 128, 256, 8, 64
    rows = (2 ** 32 + 16384) // hd
    d = tmr._big_op_inputs(rows, hd, T, W, torch.float32, 3)

    def run():
        leaves = {n: d[n].detach().clone().requires_grad_(n == "memory") for n in OP_INPUTS}
        out = sdm_write_read(leaves["memory"], d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"],
                             d["q_val"], chunk_size=C, num_memory_slots=rows, num_parallel=1)
        gen = torch.Generator(device="cuda").manual_seed(4)
        out.backward(torch.randn(out.shape, device="cuda", generator=gen))
        torch.cuda.synchronize()
        return {"readings": out.detach(), "bank": leaves["memory"].detach(), "grad_memory": leaves["memory"].grad}
    cpp0 = bw.STATS.get("steps_cpp", 0)
    got = run()
    assert bw.STATS.get("steps_cpp", 0) == cpp0 + 1
    with monkeypatch.context() as m:
        m.setattr(bw, "_steps_cpp", lambda **kw: False)
        ref = run()
    for k in ref:
        assert not bits_mismatch(ref[k], got[k]), f"{k}: {bits_mismatch(ref[k], got[k])}"

