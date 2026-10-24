# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The int4 refusal (class S4; no GPU is refused): int4 rows of regular float32 heads whose ceil(head_dim /
128) equal column slices are 4 mod 8 columns wide are refused on calls that may need a backward, on every GPU, before any
work, with UnsupportedConfigurationError: memory_ops.quant_tail_class(..., device=...) after the quantized-tail classes
T0-T3, raised by refuse_quant_tail at the op's entry (memory_ops._admit).

Static (no GPU): the S4 heads equal gx_pa2's int4 rule on regular float32 heads, the neighbours are not refused,
precedence T0-T3 > S4, the exact messages, device=None yields T0-T3 only. GPU: the op (sdm_write_read, a recorded and an
unrecorded direct apply) refuses S4 with the exact message, zero CUDA kernels, no allocation request and nothing changed
(natively, and on the A100 / RTX rows forced in-process); the same shapes run without a backward and with snapshot_quant
'none'; the neighbours keep their native route; a refusal on a forced row leaves the native calls untouched."""
import pytest
import torch

from sdm_testlib import requires_cuda

BF, F32 = torch.bfloat16, torch.float32

# The user-facing texts (sent to the docs owner verbatim; pinned here).
S4_TEMPLATE = (
    "sparse_delta_memory: snapshot_quant={quant!r} on a {dtype} bank with head_dim {hd} is not supported for calls "
    "that may need a backward (column slices {sw} columns wide, not a multiple of 8); native support is planned for "
    "2.0.1. Supported for such calls: snapshot_quant='fp8' or 'none' with this head_dim, or int4 with a head_dim that "
    "splits into ceil(head_dim / 128) equal column slices whose width is a multiple of 8, e.g. 64, 128, 480, 960, 1024, "
    "2048; calls without a backward (torch.no_grad(), inference) are not limited.")
# The S4 heads up to 1024 (one slice: 4 mod 8 up to 124; then 2..8 slices whose width is 4 mod 8).
S4_UPTO_1024 = (4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136, 152, 168, 184, 200, 216, 232,
                248, 276, 300, 324, 348, 372, 400, 432, 464, 496, 540, 580, 620, 648, 696, 744, 812, 868, 928, 992)
S4_REPS = (4, 12, 36, 100, 136, 200, 300)            # (12, 36, 100, 136, 200, 300) and the smallest
# (capability, opt-in shared memory per block, SMs, L2 bytes) rows of the matrix (QTREF's rows plus RTX 50 and modern).
ROWS = {"a100": ((8, 0), 166912, 108, 41943040), "rtx86": ((8, 6), 101376, 82, 6291456),
        "rtx89": ((8, 9), 101376, 128, 75497472), "rtx120": ((12, 0), 101376, 170, 100663296),
        "h100": ((9, 0), 232448, 132, 52428800), "b200": ((10, 0), 232448, 148, 132120576),
        "gb300": ((10, 3), 232448, 152, 135528448)}
F32_INT4_OK = (8, 16, 64, 128, 256, 480, 960, 1024, 2048, 16384)


def _mo():
    from sparse_delta_memory import memory_ops
    return memory_ops


def _dev():
    from sparse_delta_memory.cuda import _device
    return _device


def _qtref_regular(hd, dtype):
    """QTREF's class None below 2**20 columns (no T0-T3): hd % 4 == 0, equal slices of 16-byte rows, hd <= 16384."""
    nsl = -(-hd // 128)
    if hd % 4 or hd % nsl or (hd // nsl) % 4:
        return False
    es = 2 if dtype == BF else 4
    return (hd // nsl) * es % 16 == 0 and hd * es % 16 == 0 and hd <= 16384


def _s4(dtype, hd, quant):
    """The S4 rule, independently: int4 on a float32 bank, a regular head, slices 4 mod 8 columns wide."""
    return dtype == F32 and quant == "int4" and _qtref_regular(hd, dtype) and (hd // -(-hd // 128)) % 8 == 4


def _gx_pa2_int4_rule(hd):
    """gx_pa2_cuda.admits_rows's slice rule for int4 rows (q = 8), its arithmetic copied (the module needs CUDA)."""
    nsl = -(-hd // 128)
    return not (hd % 8 or hd % nsl or (hd // nsl) % 8 or 4 * (-(-(hd // 4) // nsl)) * nsl != hd)


def _force(monkeypatch, row):
    """The package sees `row` (ROWS) on every device; None: the native device (no forced row)."""
    d = _dev()
    if row is None:
        monkeypatch.setattr(d, "_FORCED", None)
        return
    cc, optin, sms, l2 = ROWS[row]
    monkeypatch.setattr(d, "_FORCED", (tuple(cc), int(optin), sms, l2))


def _s4_msg(hd):
    return S4_TEMPLATE.format(quant="int4", dtype="float32", hd=hd, sw=hd // -(-hd // 128))


CUDA0 = torch.device("cuda", 0)
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------------------------------------- static
def test_s4_is_the_int4_rule_of_regular_float32_heads():
    """S4 (the device is not read: a CPU device here) = int4 on a float32 bank, QTREF-regular head, slices 4 mod 8
    columns wide = exactly the regular float32 heads gx_pa2's int4 rows do not take; 68 heads, the widest 3844, the 43 up
    to 1024 pinned; QTREF's own answer (device=None) for them stays None."""
    mo = _mo()
    hds = list(range(1, 4200)) + list(range(16300, 16600)) + [32768, 2 ** 20 - 4, 2 ** 20, 2 ** 20 + 4]
    s4 = []
    for dt in (BF, F32):
        for hd in hds:
            for q in ("fp8", "int4", "none"):
                want = _s4(dt, hd, q)
                got = mo.quant_tail_class(dt, hd, q, 4096, device=CPU)
                assert got == (mo.quant_tail_class(dt, hd, q, 4096) or ("S4" if want else None)), (dt, hd, q, got)
                if want:
                    s4.append(hd)
                    assert mo.quant_tail_class(dt, hd, q, 4096) is None and not _gx_pa2_int4_rule(hd), hd
                    assert mo.quant_tail_class(dt, hd, q, 2 ** 31, device=CPU) == "S4"   # any row count
            if dt == F32 and _qtref_regular(hd, dt):
                assert _s4(dt, hd, "int4") == (not _gx_pa2_int4_rule(hd)), hd
    assert len(s4) == 68 and max(s4) == 3844
    assert tuple(h for h in s4 if h <= 1024) == S4_UPTO_1024
    assert {12, 36, 100, 136, 200, 300} <= set(s4)


def test_neighbours_are_not_refused():
    """Not S4: hd 8 / 16 / 128 / 1024 (and the other regular float32 int4 heads), int4 on bfloat16 banks (QTREF's T1
    where the rows are not 16 bytes), fp8 and 'none' on the S4 heads."""
    mo = _mo()
    for hd in F32_INT4_OK:
        assert mo.quant_tail_class(F32, hd, "int4", 4096, device=CPU) is None, hd
    for hd in S4_REPS:
        assert mo.quant_tail_class(F32, hd, "fp8", 4096, device=CPU) is None, hd
        assert mo.quant_tail_class(F32, hd, "none", 4096, device=CPU) is None, hd
        assert mo.quant_tail_class(BF, hd, "int4", 4096, device=CPU) == mo.quant_tail_class(BF, hd, "int4", 4096)
        assert mo.quant_tail_class(BF, hd, "int4", 4096) in (None, "T1"), hd
    for hd in (64, 128, 480, 960, 1024, 2048):                     # the message's int4 examples are not refused
        for dt in (BF, F32):
            assert mo.quant_tail_class(dt, hd, "int4", 4096, device=CPU) is None, (dt, hd)


def test_precedence_t_classes_then_s4(monkeypatch):
    """T0-T3 first (QTREF's answer and message on every row), then S4, on every row of the matrix (A100 and RTX
    included); every other quantized head: no class."""
    mo = _mo()
    t_reps = [("T0", BF, 16512), ("T1", BF, 136), ("T2", BF, 320), ("T2", F32, 132), ("T3", BF, 129), ("T3", F32, 65)]
    for row in list(ROWS):
        _force(monkeypatch, row)
        for cls, dt, hd in t_reps:
            for q in ("fp8", "int4"):
                assert mo.quant_tail_class(dt, hd, q, 4096, device=CUDA0) == cls, (row, cls, q)
        for hd in S4_REPS:
            assert mo.quant_tail_class(F32, hd, "int4", 4096, device=CUDA0) == "S4", (row, hd)
        for dt, hd, q in ((BF, 128, "fp8"), (BF, 128, "int4"), (F32, 128, "int4"), (F32, 12, "fp8")):
            assert mo.quant_tail_class(dt, hd, q, 4096, device=CUDA0) is None, (row, dt, hd)
            assert mo.quant_tail_class(dt, hd, "none", 4096, device=CUDA0) is None


def test_the_messages(monkeypatch):
    """The exact texts (UnsupportedConfigurationError, a ValueError; tail_class S4, not in the message); QTREF's text
    unchanged for T0-T3 with the device passed; no refusal for 'none' or a native shape; on the A100 and RTX rows too."""
    mo = _mo()
    assert mo.QUANT_S4_MSG == S4_TEMPLATE
    for row in ("gb300", "a100", "rtx86", "rtx120"):
        _force(monkeypatch, row)
        for hd in S4_REPS + (3844,):
            with pytest.raises(mo.UnsupportedConfigurationError) as err:
                mo.refuse_quant_tail(F32, hd, "int4", 4096, device=CUDA0)
            assert str(err.value) == _s4_msg(hd) and err.value.tail_class == "S4" and isinstance(err.value, ValueError)
            assert "class" not in str(err.value) and "2.0.1" in str(err.value)
        for q in ("fp8", "int4"):
            with pytest.raises(mo.UnsupportedConfigurationError) as err:
                mo.refuse_quant_tail(BF, 129, q, 4096, device=CUDA0)       # T3 first: QTREF's text
            assert err.value.tail_class == "T3" and str(err.value).startswith(
                f"sparse_delta_memory: snapshot_quant={q!r} on a bfloat16 bank with head_dim 129 is not supported")
        mo.refuse_quant_tail(F32, 128, "int4", 4096, device=CUDA0)
        mo.refuse_quant_tail(F32, 12, "fp8", 4096, device=CUDA0)
        mo.refuse_quant_tail(F32, 12, "none", 4096, device=CUDA0)
        mo.refuse_quant_tail(BF, 128, "fp8", 4096, device=CUDA0)
    assert _s4_msg(300).count("column slices 100 columns wide") == 1 and "column slices 68 columns" in _s4_msg(136)


def test_without_the_device_qtref_is_unchanged(monkeypatch):
    """device=None never yields S4, on any row, and refuse_quant_tail without a device refuses T0-T3 only."""
    mo = _mo()
    for row in ("a100", "rtx89", "gb300"):
        _force(monkeypatch, row)
        for dt in (BF, F32):
            for hd in list(range(1, 1100)) + [16384, 16388, 16512]:
                for q in ("fp8", "int4"):
                    got = mo.quant_tail_class(dt, hd, q, 4096)
                    assert got != "S4", (row, dt, hd, q)
                    if got is None:
                        mo.refuse_quant_tail(dt, hd, q, 4096)            # no device: no S4 refusal


# ------------------------------------------------------------------------------------------------------------- GPU
def _modern():
    return tuple(torch.cuda.get_device_capability()) in ((9, 0), (10, 0), (10, 3))


def _assert_refused(call, msg, cls, state):
    """QTREF's _assert_refused with this refusal's text: the call raises ``msg`` (tail_class ``cls``) with ZERO CUDA
    kernels, no allocation request and nothing changed."""
    import gc
    from torch.profiler import ProfilerActivity, profile
    from test_quant_tail_refusal import _allocations
    mo = _mo()
    gc.collect()
    torch.cuda.synchronize()
    n0 = _allocations()
    gc.disable()
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with pytest.raises(mo.UnsupportedConfigurationError) as err:
                call()
            torch.cuda.synchronize()
        n1 = _allocations()
    finally:
        gc.enable()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if getattr(state, "layer", None) is None and getattr(state, "runner_obj", None) is None:   # the op's own entry
        assert kernels == [], kernels[:8]
        assert n1 == n0, f"{n1 - n0} allocation requests during the refused call"
    assert type(err.value) is mo.UnsupportedConfigurationError and err.value.tail_class == cls, (type(err.value), cls)
    assert str(err.value) == msg, str(err.value)
    state.assert_unchanged()


def _setup(monkeypatch, row):
    """Native (row None: a modern GPU) or a row forced in-process (A100 / RTX: S4 refused there too)."""
    from test_quant_tail_refusal import _warm
    if row is None and not _modern():
        pytest.skip("the native cases run on a modern GPU (the A100 / RTX rules are covered by their forced rows)")
    if row is not None:
        _force(monkeypatch, row)
    _warm()


# (row, bank dtype, head_dim, snapshot_quant): every case is an S4 head (no GPU is refused)
OP_CASES = ([(None, F32, hd, "int4") for hd in S4_REPS] + [("rtx86", F32, hd, "int4") for hd in (12, 300)]
            + [("a100", F32, hd, "int4") for hd in (36, 300)])


def _want(row, dt, hd, q):
    return ("S4", _s4_msg(hd))


def _ids(cases):
    return [f"{r or 'native'}_{str(d)[6:]}_hd{h}_{q}" for r, d, h, q in cases]


@requires_cuda
@pytest.mark.parametrize("row,dtype,hd,quant", OP_CASES, ids=_ids(OP_CASES))
@pytest.mark.parametrize("entry", ["op", "apply"])
def test_the_op_refuses_without_work(entry, row, dtype, hd, quant, monkeypatch):
    """sdm_write_read's training call and a recorded direct SDMWriteRead.apply: refused before any work."""
    from test_quant_tail_refusal import _State, _op_call
    _setup(monkeypatch, row)
    call, tensors = _op_call(dtype, hd, quant, direct=entry == "apply")
    cls, msg = _want(row, dtype, hd, quant)
    _assert_refused(call, msg, cls, _State(tensors))


UNREC_CASES = [(None, F32, 12, "int4"), (None, F32, 300, "int4"), ("rtx86", F32, 36, "int4"), ("a100", F32, 12, "int4")]


@requires_cuda
@pytest.mark.parametrize("row,dtype,hd,quant", UNREC_CASES, ids=_ids(UNREC_CASES))
@pytest.mark.parametrize("how", ["no_grad", "no_input_grad"])
def test_an_unrecorded_direct_apply_is_refused_fail_closed(how, row, dtype, hd, quant, monkeypatch):
    """A direct SDMWriteRead.apply that autograd does not record (torch.no_grad(); grad on with no grad input): refused
    before any work like QTREF's classes (fail-closed: no no-grad equality evidence for GPU / S4)."""
    from sdm_testlib import make_op_inputs, OP_INPUTS
    from test_quant_tail_refusal import _State
    mo = _mo()
    _setup(monkeypatch, row)
    P, T, W, R, sph, C = 2, 100, 4, 4, 256, 64
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=11, slots="zipf")
    a = [d[n].detach().clone().requires_grad_(how != "no_input_grad" and n in OP_INPUTS)
         for n in ("memory", "k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")]

    def call():
        if how == "no_grad":
            with torch.no_grad():
                return mo.SDMWriteRead.apply(*a, C, sph, P, quant, False)
        return mo.SDMWriteRead.apply(*a, C, sph, P, quant, False)
    cls, msg = _want(row, dtype, hd, quant)
    _assert_refused(call, msg, cls, _State(a))


@requires_cuda
@pytest.mark.parametrize("hd", [36, 100, 300])
def test_the_s4_heads_run_without_a_backward_or_unquantized(hd):
    """Not refused: the S4 heads without a backward (the layer under no_grad / inference_mode, sdm_write_read without
    grad: nothing is quantized) and with snapshot_quant 'none' (or fp8) on a training call."""
    from test_quant_tail_refusal import _layer, _op_call
    if not _modern():
        pytest.skip("a modern GPU (H100 / B200 / GB300)")
    layer = _layer(F32, hd, "int4")
    x = torch.randn(2, 96, 2 * hd, device="cuda", dtype=F32)
    with torch.no_grad():
        y0, _ = layer(x)
    with torch.inference_mode():
        y1, _ = layer(x)
    assert torch.equal(y0, y1) and bool(torch.isfinite(y0).all())
    call, _ = _op_call(F32, hd, "int4")
    with torch.no_grad():
        assert bool(torch.isfinite(call()).all())
    for q in ("none", "fp8"):
        call, leaves = _op_call(F32, hd, q)
        call().float().sum().backward()
        assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in leaves[1:]), q


NEIGHBOURS = [(F32, h, "int4", h in (8, 128, 960)) for h in (8, 16, 128, 960, 1024)] + [
    (BF, 128, "int4", False), (BF, 1024, "int4", False), (BF, 128, "fp8", False), (F32, 100, "fp8", False)]


@requires_cuda
@pytest.mark.parametrize("dtype,hd,quant,native", NEIGHBOURS,
                         ids=[f"{str(d)[6:]}_hd{h}_{q}" for d, h, q, _ in NEIGHBOURS])
def test_native_neighbours_are_not_refused(dtype, hd, quant, native):
    """H100 / GB300 untouched: the S4 neighbours and regular quantized heads are not refused and train with finite
    gradients; the logged boundary table's accept rows among them (float32 int4 hd 8 / 128 / 960) on the CUDA backward
    (not the reference loop), as QTREF's boundary test asserts for them."""
    from sparse_delta_memory import backward
    from test_quant_tail_refusal import _op_call
    if not _modern():
        pytest.skip("a modern GPU (H100 / B200 / GB300)")
    assert _mo().quant_tail_class(dtype, hd, quant, 4096, device=torch.device("cuda")) is None
    call, leaves = _op_call(dtype, hd, quant)
    call().float().sum().backward()
    if native:
        assert backward.LAST_LOOP["name"] != "reference_python_loop", backward.LAST_LOOP
    assert all(t.grad is not None and bool(torch.isfinite(t.grad.float()).all()) for t in leaves[1:])


@requires_cuda
def test_a_forced_refusal_leaves_the_native_call_untouched(monkeypatch):
    """The memo on the real op: an S4 call refused on a forced RTX 30 row (zero work), then regular quantized calls on
    the native modern GPU train (nothing cached by the refusal; no GPU is refused)."""
    from test_quant_tail_refusal import _State, _op_call
    if not _modern():
        pytest.skip("a modern GPU (H100 / B200 / GB300)")
    _setup(monkeypatch, "rtx86")
    call, tensors = _op_call(F32, 100, "int4")
    _assert_refused(call, _s4_msg(100), "S4", _State(tensors))
    _force(monkeypatch, None)
    for dt, hd, q in ((BF, 128, "fp8"), (F32, 128, "int4")):
        call, leaves = _op_call(dt, hd, q)
        call().float().sum().backward()
        assert all(t.grad is not None for t in leaves[1:]), (dt, hd, q)
