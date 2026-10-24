# Copyright (c) Meta Platforms, Inc. and affiliates.
"""2.0.0: every call runs natively end to end or is refused before any work, with one clear error.

memory_ops.native_route_class composes the routing predicates the forward and the backward use in the mode of the
forward that runs: no-grad (the unquantized DW), saving (SDMWriteRead.forward, recorded or not: the DW in the snapshot's
mode and the snapshot layout the native DW reads) and training (saving plus the CUDA backward and its GDS).
refuse_non_native raises UnsupportedConfigurationError for the calls it declines (``tail_class``: H, S or P); the op's
gate (memory_ops._admit) runs it after every other check of the entry, once per admitted call signature, and at every
call of a refused one. Checked here: the classes, modes and messages; that each step of an admitted call runs exactly one
kernel family per mover and a refused one launches nothing and moves no state (the op, the graph runner); the gate (a
refused call raises at every call, an admitted signature is checked once; one _layout per call); and regressions
(the saving forward of an unrecorded apply, the quantized refusals first, the shared-bank opt-in first)."""
import re

import pytest
import torch

from sdm_testlib import make_op_inputs, requires_cuda

pytestmark = requires_cuda

BF, F32 = torch.bfloat16, torch.float32
A100_ROW = ((8, 0), 166912, 108, 41943040)   # tests/test_quant_config_refusal.py ROWS["a100"]


def _mo():
    from sparse_delta_memory import memory_ops
    return memory_ops


def _view(slots, hd, dtype=BF):
    return torch.empty(1, hd, device="cuda", dtype=dtype).expand(slots, hd)   # shape, dtype, device: no storage


def _launched(fn):
    """The CUDA kernel launches of ``fn()`` (torch.profiler; names in launch order, repeats kept), and its exception
    (or None)."""
    from torch.profiler import ProfilerActivity, profile
    err = None
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        try:
            fn()
        except BaseException as e:   # noqa: BLE001  (returned to the caller)
            err = e
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA], err


def _package(launched):
    return [n for n in launched if re.search(r"(?<![A-Za-z0-9_])(da2_|fx_|gx_|dc_|sdm|phase1|gather)", n)]


def _no_dw(monkeypatch):
    from sparse_delta_memory import movers
    monkeypatch.setattr(movers, "dw_admits", lambda bank, chunk_size, quant="none": False)


MODES = {"no-grad": (False, False), "saving": (False, True), "training": (True, True)}   # (needs_backward, saving)


# ------------------------------------------------------------------------------------------- classes, modes, message
@pytest.mark.parametrize("slots, hd, rows, mode, want", [
    (4096, 2 ** 20 + 4, 4096, "training", "H"),
    (2 ** 31, 128, 4096, "training", "S"),
    (4096, 128, 2 ** 31, "training", "P"),    # 64-bit snapshot rows: the row count names no class
    (4096, 128, 2 ** 31, "saving", "P"),
    (4096, 128, 2 ** 31, "no-grad", "P"),
    (4096, 128, 4096, "training", "P"),
    (4096, 128, 4096, "saving", "P"),
    (4096, 128, 4096, "no-grad", "P"),
])
def test_a_declined_call_is_named_by_its_class(slots, hd, rows, mode, want, monkeypatch):
    mo = _mo()
    _no_dw(monkeypatch)
    nb, sv = MODES[mode]
    bank = _view(slots, hd)
    assert mo.native_route_class(bank, 64, "none", rows, nb, False, sv) == want
    with pytest.raises(mo.UnsupportedConfigurationError) as err:
        mo.refuse_non_native(bank, 64, "none", rows, needs_backward=nb, saving=sv)
    assert err.value.tail_class == want
    text = str(err.value)
    assert text.startswith("sparse_delta_memory: no native route for this call: ") and "removed in 2.0.0" in text, text
    assert mo.NATIVE_MODES[mode] in text and mo.NATIVE_REMEDIES[want] in text, text
    assert "2.0.1" not in text and "without autograd (" not in text, text   # no no-grad route here either


def test_a_training_refusal_names_the_native_no_grad_route(monkeypatch):
    """The remedy of a training (saving) call whose no-grad route exists says so; a no-grad refusal never offers it."""
    mo = _mo()
    from sparse_delta_memory import movers
    bank = _view(4096, 128)
    assert mo.native_route_class(bank, 64) is None
    monkeypatch.setattr(movers, "gds_admits", lambda *a, **k: False)
    with pytest.raises(mo.UnsupportedConfigurationError) as err:
        mo.refuse_non_native(bank, 64, "none", 4096, needs_backward=True)
    assert err.value.tail_class == "P" and str(err.value).endswith(
        mo.NATIVE_REMEDIES["P"] + ("" if _modern() else mo.NATIVE_P_GPU) + mo.NATIVE_NOGRAD_OK["training"] + "."), \
        str(err.value)


def _modern():
    from sparse_delta_memory.cuda import _device
    return _device.backend(torch.device("cuda", torch.cuda.current_device())) == "modern"


def test_the_classes_refuse_nothing_on_their_own():
    """The bounds only name a refused call: a call every predicate admits is native at any of them."""
    mo = _mo()
    from sparse_delta_memory import movers
    bank = _view(4096, 128)
    assert movers.dw_admits(bank, 64) and mo.native_route_class(bank, 64, "none", 2 ** 31 - 1, True) is None
    for rows in (2 ** 31, 2 ** 40):   # 64-bit snapshot rows
        assert mo.native_route_class(bank, 64, "none", rows, True) is None, rows


# ------------------------------------------------------------------- one family per mover per step, or the refusal
# mover -> family -> the CUDA kernel that runs its step (the module's __global__ name): the forward's DW, the backward's
# GDS and Phase A. A step of an admitted call launches exactly one of them per mover, once.
FAMILIES = {
    "dw": {"da2": ("da2_dw_kernel",), "fx": ("fx_dw_kernel", "fx_dw2_kernel")},
    "gds": {"da2": ("da2_gds2f_kernel", "da2_gscatter_kernel"), "gx": ("gx_gds_kernel",)},   # da2: direct or scatter
    "pa": {"dc": ("dc_pa3_kernel",), "gx": ("gx_pa2_kernel",)},
}


def _family_counts(launched):
    out = {}
    for mover, fams in FAMILIES.items():
        out[mover] = {f: sum(1 for n in launched for k in names if re.search(r"(?<![A-Za-z0-9_])" + k + r"(?![A-Za-z0-9_])", n))
                      for f, names in fams.items()}
    return out


P, T, W, R, SPH = 2, 150, 8, 8, 1024     # 3 steps of chunk 64 (the last one partial)


def _op(d, C, train, quant="none"):
    """sdm_write_read on private leaves of ``d``; returns the leaves (the bank the op received is leaves['memory'])."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    names = ("memory", "k_val", "v", "beta", "g", "q_val")
    leaves = {n: d[n].detach().clone().requires_grad_(train) for n in names}
    y = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                       d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=SPH, num_parallel=P,
                       snapshot_quant=quant)
    if train:
        y.float().sum().backward()
    torch.cuda.synchronize()
    return leaves


ROUTE_CASES = [(dt, hd, C) for dt, hd in ((BF, 64), (BF, 128), (BF, 129), (BF, 1024), (F32, 128), (F32, 66))
               for C in (32, 64)]


@pytest.mark.parametrize("train", [False, True], ids=["nograd", "train"])
@pytest.mark.parametrize("dt, hd, C", ROUTE_CASES, ids=[f"{str(c[0])[6:]}_hd{c[1]}_C{c[2]}" for c in ROUTE_CASES])
def test_each_step_runs_exactly_one_family_per_mover(dt, hd, C, train):
    """Every input has exactly one route or an explicit error. An admitted call launches, for
    each of its steps, exactly one family's kernel per mover (the DW in the forward; the GDS and Phase A in a training
    call's backward; a no-grad call has neither), once per step; a declined one is refused (UnsupportedConfigurationError)
    with no package kernel launched. Phase A's family agrees with backward.STATS (gx_pa2)."""
    from sparse_delta_memory import backward as bw
    mo = _mo()
    d = make_op_inputs(P, T, W, R, hd, SPH, dt, seed=7)
    steps = -(-T // C)
    cls = mo.native_route_class(d["memory"], C, "none", P * steps * C * W, train)
    pa0 = bw.STATS.get("gx_pa2", 0)
    launched, err = _launched(lambda: _op(d, C, train))
    if cls is not None:
        assert isinstance(err, mo.UnsupportedConfigurationError) and err.tail_class == cls, err
        assert not _package(launched), launched[:8]
        return
    assert err is None, err
    got = _family_counts(launched)
    for mover in ("dw",) + (("gds", "pa") if train else ()):
        ran = {f: n for f, n in got[mover].items() if n}
        assert len(ran) == 1 and list(ran.values()) == [steps], (mover, got[mover], steps)
    if not train:
        assert not any(got["gds"].values()) and not any(got["pa"].values()), got
    else:
        assert (got["pa"]["gx"] > 0) == (bw.STATS.get("gx_pa2", 0) == pa0 + 1), (got["pa"], bw.STATS.get("gx_pa2"))


# ---------------------------------------------------------------------------------------- the entries, end to end
@pytest.mark.parametrize("train", [False, True], ids=["nograd", "train"])
@pytest.mark.parametrize("hd", [64, 129])
def test_an_admitted_call_runs_and_a_declined_one_launches_nothing(hd, train, monkeypatch):
    from sparse_delta_memory import backward as bw
    mo = _mo()
    d = make_op_inputs(P, T, W, R, hd, SPH, BF, seed=7)
    if mo.native_route_class(d["memory"], 64, "none", P * 3 * 64 * W, train) is None:
        b0 = bw.STATS["dc"]
        _op(d, 64, train)                               # the CUDA kernels, no internal error
        assert bw.STATS["dc"] == b0 + (1 if train else 0)
    _no_dw(monkeypatch)
    p1 = []
    orig = mo.phase1_forward
    monkeypatch.setattr(mo, "phase1_forward", lambda *a, **k: p1.append(1) or orig(*a, **k))
    held = {}

    def call():
        held.update(_op_refused(d, train, held))
    launched, err = _launched(call)
    assert isinstance(err, mo.UnsupportedConfigurationError) and err.tail_class == "P", err
    assert not _package(launched), launched[:8]
    assert not p1, "Phase 1 ran before the refusal"
    assert torch.equal(held["memory"].detach(), held["memory0"]), "the refused call touched the bank it was passed"


def _op_refused(d, train, held):
    """_op with the leaves kept in ``held`` (the bank the op receives, and its bytes before the call)."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    names = ("memory", "k_val", "v", "beta", "g", "q_val")
    leaves = {n: d[n].detach().clone().requires_grad_(train) for n in names}
    held["memory"], held["memory0"] = leaves["memory"], leaves["memory"].detach().clone()
    sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                   d["q_idx"], leaves["q_val"], chunk_size=64, num_memory_slots=SPH, num_parallel=P)
    return held


def _runner_inputs(BH, hd, k_dtype=torch.int64):
    k_idx = torch.randint(0, SPH, (BH, 128, 4), device="cuda").to(k_dtype)
    q_idx = torch.randint(0, SPH, (BH, 128, 4), device="cuda")
    f = lambda *s: torch.rand(*s, device="cuda", dtype=BF, requires_grad=True)   # noqa: E731
    return (k_idx, f(BH, 128, 4), f(BH, 128, hd), f(BH, 128, 1), f(BH, 128, 1), q_idx, f(BH, 128, 4))


def test_the_runner_refuses_at_its_entry(monkeypatch):
    """CudaGraphRunner.__call__ and .eager refuse a call without a native route: the op's refusal (in the capture's
    warm-up; the eager op applies it before it runs), and no entry."""
    from sparse_delta_memory.cuda_graph import CudaGraphRunner
    mo = _mo()
    hd, BH, C = 64, 4, 64
    inputs = _runner_inputs(BH, hd)
    _no_dw(monkeypatch)
    runner = CudaGraphRunner("t")
    for call in (lambda: runner(None, (BH * SPH, hd), BF, inputs, (C, SPH, BH, "none", False)),
                 lambda: CudaGraphRunner.eager(None, (BH * SPH, hd), BF, inputs, (C, SPH, BH, "none", False))):
        launched, err = _launched(call)
        assert isinstance(err, mo.UnsupportedConfigurationError) and err.tail_class == "P", err
        assert runner.entry is None, runner.stats


# ------------------------------------------------------------------------------------------------- regression tests
def _apply_nograd(mem, d, C, quant, P_=P, sph=SPH):
    """An unrecorded direct SDMWriteRead.apply (torch.no_grad()): it runs the SAVING forward, though no backward follows."""
    mo = _mo()
    with torch.no_grad():
        return mo.SDMWriteRead.apply(mem, d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"],
                                     C, sph, P_, quant, False)


@pytest.mark.parametrize("dt, hd, quant", [(BF, 130, "none"), (BF, 128, "fp8"), (F32, 128, "int4")])
def test_blocker_an_unrecorded_apply_is_checked_as_the_saving_forward(dt, hd, quant, monkeypatch):
    """An unrecorded apply runs the saving forward, which writes the snapshot's compact rows (a tail
    head, fp8 / int4) only where backward.supported admits the worst grad dtype, else layout 0, which no native DW takes
    (NO_NATIVE_MOVER_MSG, or the removed per-entry quantizer). With backward.supported declining, the call is refused
    before any work (class P, on every GPU), never an internal error; the same call through sdm_write_read (a no-grad call) is not refused by it."""
    from sparse_delta_memory import backward
    mo = _mo()
    d = make_op_inputs(P, T, W, R, hd, SPH, dt, seed=7)
    monkeypatch.setattr(backward, "supported", lambda *a, **k: False)
    mem = d["memory"].clone()
    launched, err = _launched(lambda: _apply_nograd(mem, d, 64, quant))
    want = "P"
    assert isinstance(err, mo.UnsupportedConfigurationError) and err.tail_class == want, repr(err)
    assert not _package(launched), launched[:8]
    assert torch.equal(mem, d["memory"]), "the refused call touched the bank"
    if want == "P":
        assert mo.NATIVE_MODES["saving"] in str(err), str(err)
    assert mo.native_route_class(d["memory"], 64) == mo.native_route_class(d["memory"], 64, quant, 4096, False, False,
                                                                           False)   # the no-grad mode ignores quant


def test_major_663_an_unrecorded_quantized_apply_keeps_the_quantized_refusal(monkeypatch):
    """The quantized refusals run before
    the native-route refusal: an unrecorded int4 apply of an S4 head (float32, head_dim 36) on the forced A100 row, with backward.supported
    declining (class P), gets QTR's S4 refusal and its text, not the native-route refusal."""
    from sparse_delta_memory import backward
    from sparse_delta_memory.cuda import _device, _image
    mo = _mo()
    monkeypatch.setattr(_device, "_FORCED", A100_ROW)
    dev = torch.cuda.current_device()
    _image.require_supported_image(dev)
    _device.backend(dev)
    monkeypatch.setattr(backward, "supported", lambda *a, **k: False)
    d = make_op_inputs(P, T, W, R, 36, SPH, F32, seed=3)
    mem = d["memory"].clone()
    launched, err = _launched(lambda: _apply_nograd(mem, d, 64, "int4"))
    assert isinstance(err, mo.UnsupportedConfigurationError) and err.tail_class == "S4", repr(err)
    assert str(err) == mo.QUANT_S4_MSG.format(quant="int4", dtype="float32", hd=36, sw=36), str(err)
    assert not _package(launched), launched[:8]


def test_major_392_488_the_op_argument_errors_come_first(monkeypatch):
    """The op's argument checks that guard out-of-bounds access come
    before the native-route refusal (here a call without a native route): float indices (TypeError) and a bank of another head_dim,
    e.g. a mismatched cache (ValueError)."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    d = make_op_inputs(P, T, W, R, 64, SPH, BF, seed=7)
    _no_dw(monkeypatch)
    kw = dict(chunk_size=64, num_memory_slots=SPH, num_parallel=P)
    with pytest.raises(TypeError, match="k_idx and q_idx must be int32 or int64"):
        sdm_write_read(d["memory"], d["k_idx"].float(), d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"],
                       **kw)
    with pytest.raises(ValueError, match=r"v: expected shape"):
        sdm_write_read(torch.zeros(P * SPH, 68, device="cuda", dtype=BF), d["k_idx"], d["k_val"], d["v"], d["beta"],
                       d["g"], d["q_idx"], d["q_val"], **kw)


def test_major_987_the_shared_bank_opt_in_comes_first(monkeypatch):
    """A no-grad call of two streams over one shared bank without shared_bank=True,
    on a call without a native route, gets the shared-bank ValueError, not the native-route refusal."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    d = make_op_inputs(P, T, W, R, 64, SPH, BF, seed=7)
    _no_dw(monkeypatch)
    with torch.no_grad(), pytest.raises(ValueError, match="pass shared_bank=True"):
        sdm_write_read(d["memory"][:SPH].contiguous(), d["k_idx"] % SPH, d["k_val"], d["v"], d["beta"], d["g"],
                       d["q_idx"] % SPH, d["q_val"], chunk_size=64, num_memory_slots=0, num_parallel=P)


# ------------------------------------------------------------------------------------------------ the op's gate (_admit)
def _count(monkeypatch, name):
    """Counts the calls of memory_ops.<name> (the gate and the op look it up at call time)."""
    mo = _mo()
    calls, real = [], getattr(mo, name)
    monkeypatch.setattr(mo, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


@pytest.mark.parametrize("train", [False, True], ids=["nograd", "train"])
def test_a_refused_call_raises_at_every_call_and_an_admitted_one_is_checked_once(train, monkeypatch):
    """Only admitted signatures are stored: the same refused call raises the same error at every call (the route is
    checked each time); an admitted signature runs refuse_non_native once, then no check at all (the memo hit)."""
    mo = _mo()
    monkeypatch.setattr(mo, "_NATIVE_MEMO", {})
    d = make_op_inputs(P, T, W, R, 64, SPH, BF, seed=7)
    checks = _count(monkeypatch, "refuse_non_native")
    with monkeypatch.context() as m:
        _no_dw(m)
        errs = []
        for _ in range(3):
            launched, err = _launched(lambda: _op(d, 64, train))
            assert isinstance(err, mo.UnsupportedConfigurationError) and err.tail_class == "P", err
            assert not _package(launched), launched[:8]
            errs.append(str(err))
        assert len(set(errs)) == 1 and len(checks) == 3, (len(checks), errs[0])
    del checks[:]
    _op(d, 64, train)
    _op(d, 64, train)
    assert len(checks) == 1, len(checks)


@pytest.mark.parametrize("entry", ["nograd", "train", "apply", "apply_nograd"])
def test_one_layout_per_call(entry, monkeypatch):
    """_layout runs once per op call: sdm_write_read hands its layout to the training forward; a direct apply (and the
    graph runner's capture) computes it in the forward."""
    mo = _mo()
    d = make_op_inputs(P, T, W, R, 64, SPH, BF, seed=7)
    _op(d, 64, False)                                   # warm: the signature is admitted
    calls = _count(monkeypatch, "_layout")
    if entry in ("nograd", "train"):
        _op(d, 64, entry == "train")
    else:
        leaves = [d[n].detach().clone().requires_grad_(entry == "apply" and d[n].is_floating_point())
                  for n in ("memory", "k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")]
        with torch.set_grad_enabled(entry == "apply"):
            mo.SDMWriteRead.apply(*leaves, 64, SPH, P, "none", False)
    assert len(calls) == 1, len(calls)
