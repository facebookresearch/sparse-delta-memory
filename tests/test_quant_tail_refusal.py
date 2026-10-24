# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The quantized-tail refusal (memory_ops.quant_tail_class / refuse_quant_tail, UnsupportedConfigurationError): fp8 /
int4 snapshots of the head dims without a quantized route (classes T0-T3) are refused on calls that may need a backward,
at the op's entry (memory_ops._admit), before any work, on every GPU.

Static: the classes against the routes' own predicates; the exported exception and its logged text. GPU (natively, and
the A100 row forced): the classes equal the forward's own native decision; the op (sdm_write_read, a direct apply) and
the layer (eager and with its CUDA-graph runner) refuse every class with the exact text, zero CUDA kernels at the op
and nothing changed; the same shapes run without a backward and unquantized."""
import ast
import os

import pytest
import torch

from sdm_testlib import make_op_inputs, OP_INPUTS, requires_cuda

BF, F32 = torch.bfloat16, torch.float32
MEMORY_OPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sparse_delta_memory",
                          "memory_ops.py")
# The user-facing text: no class label in the message.
LOGGED_TEMPLATE = (
    "sparse_delta_memory: snapshot_quant={quant!r} on a {dtype} bank with head_dim {hd} is not supported for calls "
    "that may need a backward ({what}); support is planned for a later release. Supported for such calls: "
    "head_dim <= 16384 that splits into ceil(head_dim / 128) equal column slices whose width is a multiple of 8 "
    "(bfloat16 bank, or int4 on a float32 bank) or of 4 (fp8 on a float32 bank), e.g. 64, 128, 480, 960, 1024, 2048; "
    "snapshot_quant='none' supports every head_dim, and calls without a backward (torch.no_grad(), inference) are not "
    "limited.")
WHAT = {"T0": "more than 16384 columns", "T1": "column slices whose rows are not 16-byte multiples",
        "T2": "uneven column slices", "T3": "head_dim not a multiple of 4"}
ROWS = {"a100": ((8, 0), 166912, 108, 41943040), "rtx86": ((8, 6), 101376, 82, 6291456),
        "rtx89": ((8, 9), 101376, 128, 75497472)}
# One representative per class (bank dtype, head_dim); T0 is op-level only (its layer would need dim 16512).
REPS = [("T0", BF, 16512), ("T1", BF, 136), ("T2", BF, 320), ("T2", F32, 132), ("T3", BF, 129), ("T3", F32, 65)]
LAYER_REPS = [r for r in REPS if r[0] != "T0"]


def _mo():
    from sparse_delta_memory import memory_ops
    return memory_ops


def _expected(quant, dtype, hd, cls):
    return LOGGED_TEMPLATE.format(quant=quant, dtype={BF: "bfloat16", F32: "float32"}[dtype], hd=hd, what=WHAT[cls])


def test_classes_equal_the_routes_predicates():
    """T1-T3 = the 16-byte-slice predicate's complement (mover_fx_cuda.rows16; mover_da2_cuda._shape_supported_inner
    where it exists), T0 = the quantized forward's head bound, nothing above 2**20, the same class at any compact-row
    count (2**31 and more: 64-bit snapshot rows)."""
    mo = _mo()
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    try:
        from sparse_delta_memory.cuda import mover_da2_cuda as da2
    except ImportError:          # (a tree without it: rows16 alone)
        da2 = None
    assert mo.QUANT_TAIL_MAX_HEAD_DIM == fx._QUANT_MAX_HEAD_DIM == 16384
    hds = list(range(1, 4200)) + list(range(16300, 16600)) + [32768, 65536, 2 ** 20 - 4, 2 ** 20 - 1, 2 ** 20,
                                                              2 ** 20 + 4]
    for dt in (BF, F32):
        for hd in hds:
            meta = torch.empty(1, hd, dtype=dt, device="meta")
            regular = bool(fx.rows16(meta))
            if da2 is not None and hasattr(da2, "_shape_supported_inner"):
                assert regular == bool(da2._shape_supported_inner(meta, 64)), (dt, hd)
            for q in ("fp8", "int4"):
                cls = mo.quant_tail_class(dt, hd, q)
                want_refused = hd <= 2 ** 20 and (not regular or hd > 16384)
                assert (cls is not None) == want_refused, (dt, hd, q, cls)
                assert mo.quant_tail_class(dt, hd, q, 2 ** 31) == cls
            assert mo.quant_tail_class(dt, hd, "none") is None



def test_the_refusal_is_one_exported_value_error_with_the_logged_text():
    import sparse_delta_memory
    mo = _mo()
    E = mo.UnsupportedConfigurationError
    assert issubclass(E, ValueError) and sparse_delta_memory.UnsupportedConfigurationError is E
    assert "UnsupportedConfigurationError" in sparse_delta_memory.__all__
    assert mo.QUANT_TAIL_MSG == LOGGED_TEMPLATE and mo.QUANT_TAIL_CLASSES == WHAT
    for cls, dt, hd in REPS:
        for q in ("fp8", "int4"):
            with pytest.raises(E) as err:
                mo.refuse_quant_tail(dt, hd, q, 4096)
            assert str(err.value) == _expected(q, dt, hd, cls) and err.value.tail_class == cls
            assert "class T" not in str(err.value)
    from sparse_delta_memory.cuda import _image
    assert not issubclass(E, _image.ImageRefusal) and not issubclass(_image.ImageRefusal, E)
    mo.refuse_quant_tail(BF, 1024, "fp8", 4096)          # a native shape: no refusal
    mo.refuse_quant_tail(BF, 320, "none", 4096)


# ------------------------------------------------------------------------------------------------------------- GPU
def _force(monkeypatch, row):
    if row == "native":
        return
    from sparse_delta_memory.cuda import _device as d
    cc, optin, sms, l2 = ROWS[row]
    monkeypatch.setattr(d, "_FORCED", (tuple(cc), int(optin), sms, l2))
    assert d.backend() == ("a100" if row == "a100" else "rtx")


def _native(bank, C, quant):
    """The forward's own decision for an fp8 / int4 saving call (forward.forward_steps): layout 2 (the CUDA DW writes
    the quantized compact rows, the CUDA backward reads them) needs backward.supported + mover_fx_cuda's producer, and the
    CUDA DW at all (movers.dw_admits)."""
    from sparse_delta_memory import backward, forward, movers
    rows = 4096
    return bool(backward.supported(bank, quant, C=C, grad_f32_possible=True, rows=rows)
                and forward._fxq_rows(bank, C, quant) and movers.dw_admits(bank, C, quant))


@requires_cuda
@pytest.mark.parametrize("row", ["native", "a100", "rtx86", "rtx89"])
def test_classes_equal_the_forward_route(row, monkeypatch):
    """Exactness on the real route: a shape is refused (T0-T3, or S4 with the device) iff the forward would not take the
    native quantized route for it (heads above 2**20 excepted: the native-route refusal's). Host predicates only."""
    if row == "native" and tuple(torch.cuda.get_device_capability()) not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("the native row of this test is a modern GPU")
    _force(monkeypatch, row)
    mo = _mo()
    hds = list(range(1, 521)) + [640, 960, 1020, 1024, 1028, 1920, 2048, 2176, 4096, 8192, 16380, 16384, 16388, 16512,
                                 32768, 65536, 2 ** 20 - 4, 2 ** 20]
    chunks = (16, 64, 128) if row.startswith("rtx") else (16, 64, 256)
    bad = []
    for dt in (BF, F32):
        for hd in hds:
            bank = torch.empty(16, hd, device="cuda", dtype=dt)
            for q in ("fp8", "int4"):
                cls = mo.quant_tail_class(dt, hd, q, 4096, device=bank.device)
                for C in chunks:
                    nat = _native(bank, C, q)
                    if cls is not None and nat:
                        bad.append(("refused but native", dt, hd, q, C, cls))
                    if cls is None and not nat and hd <= 2 ** 20:
                        bad.append(("not refused, no native route", dt, hd, q, C))
            del bank
    assert not bad, bad[:20]


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


class _State:
    """What a refused call must leave unchanged: its tensors (and a layer's parameters, a cache's bank); for a call of the
    op itself (no layer, no runner: nothing prepares before the op's entry) also the package's counters, dispatch record
    and graph state."""

    def __init__(self, tensors=(), layer=None, cache=None, runner=None):
        import gc
        from sparse_delta_memory import backward, cuda_graph, forward, movers
        gc.collect()   # pending garbage first: cuda_graph._shared is a weak-value map (earlier tests' entries)
        mo = _mo()
        self.tensors = [(t, t.detach().clone()) for t in tensors if torch.is_tensor(t)]
        if layer is not None:   # (a layer's call may prepare before the op refuses: buffers, the runner's pool)
            self.tensors += [(t, t.detach().clone()) for t in layer.parameters()]
        if cache is not None and torch.is_tensor(getattr(cache, "__dict__", {}).get("memory")):
            self.tensors.append((cache.memory, cache.memory.detach().clone()))
        self.counters = (dict(movers.STATS), dict(forward.STATS), dict(backward.STATS), dict(backward.LAST_LOOP),
                         mo.last_dispatch(), sorted(mo._repeat_counters))
        self.runner_obj, self.layer = runner, layer
        self.cache = cache
        self.graphs = self._graphs(cuda_graph, layer, cache)
        if cache is not None and torch.is_tensor(getattr(cache, "__dict__", {}).get("_copy")):
            self.tensors.append((cache._copy, cache._copy.detach().clone()))

    @staticmethod
    def _graphs(cuda_graph, layer, cache):
        """Module-level graph state (shared static buffers, stream pools and each pool's flags and forward count), the
        cache's length and copied bank (identity), and the layer's one-time warning flag."""
        d = getattr(cache, "__dict__", {}) if cache is not None else {}
        pools = [(repr(k), {a: len(v) if isinstance(v, list) else v for a, v in vars(p).items()
                            if v is None or isinstance(v, (bool, int, list))})
                 for k, p in sorted(cuda_graph._pools.items(), key=lambda kv: repr(kv[0]))]
        return (len(cuda_graph._shared), sorted(map(repr, list(cuda_graph._shared.keys()))), len(cuda_graph._pools),
                sorted(map(repr, list(cuda_graph._pools.keys()))), d.get("_cache_len"), id(d.get("_copy")),
                getattr(layer, "_stats_skip_warned", None), pools)

    def assert_unchanged(self):
        from sparse_delta_memory import backward, forward, movers
        mo = _mo()
        for t, before in self.tensors:
            assert torch.equal(_bytes(t), _bytes(before)), "a tensor changed"
        if self.layer is None and self.runner_obj is None:   # (a layer / runner call prepares before the op refuses)
            from sparse_delta_memory import cuda_graph
            assert (dict(movers.STATS), dict(forward.STATS), dict(backward.STATS), dict(backward.LAST_LOOP),
                    mo.last_dispatch(), sorted(mo._repeat_counters)) == self.counters
            assert self._graphs(cuda_graph, self.layer, self.cache) == self.graphs


def _allocations():
    """The caching allocator's cumulative count of allocation requests on this device (frees of earlier garbage do not
    move it; any allocation does)."""
    return torch.cuda.memory_stats().get("allocation.all.allocated", 0)


def _assert_refused(call, quant, dtype, hd, cls, state):
    """The call raises the logged message (tail_class `cls`) with ZERO CUDA kernels, no allocation request and nothing
    changed."""
    import gc
    from torch.profiler import ProfilerActivity, profile
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
    if state.layer is None and state.runner_obj is None:   # the op's own entry refuses before any work (a layer or
        assert kernels == [], kernels[:8]                   # runner call refuses there, after its own preparation)
        assert n1 == n0, f"{n1 - n0} allocation requests during the refused call"
    assert isinstance(err.value, ValueError) and err.value.tail_class == cls
    assert type(err.value) is mo.UnsupportedConfigurationError, type(err.value)
    assert str(err.value) == _expected(quant, dtype, hd, cls)
    state.assert_unchanged()


def _op_call(dt, hd, quant, direct=False):
    mo = _mo()
    P, T, W, R, sph, C = (2, 100, 4, 4, 256, 64) if hd <= 4096 else (1, 65, 4, 4, 64, 64)   # wide heads: small banks
    d = make_op_inputs(P, T, W, R, hd, sph, dt, seed=11, slots="zipf")
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    args = (leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"], d["q_idx"],
            leaves["q_val"])
    if direct:
        return (lambda: mo.SDMWriteRead.apply(*args, C, sph, P, quant, False)), list(leaves.values())
    return (lambda: mo.sdm_write_read(*args, chunk_size=C, num_memory_slots=sph, num_parallel=P,
                                      snapshot_quant=quant)), list(leaves.values())


def _warm(device=None):
    """The guard's memoized image check and the forced row's first reads, outside the profiled call."""
    from sparse_delta_memory.cuda import _device, _image
    dev = torch.cuda.current_device() if device is None else device
    check = getattr(_image, "require_supported_image", None)
    if check is not None:
        check(dev)
    _device.backend(dev)


@requires_cuda
@pytest.mark.parametrize("row", ["native", "a100"])
@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("cls,dtype,hd", REPS, ids=[f"{c}_{str(d)[6:]}_hd{h}" for c, d, h in REPS])
@pytest.mark.parametrize("entry", ["op", "apply"])
def test_the_op_refuses_without_work(entry, cls, dtype, hd, quant, row, monkeypatch):
    _force(monkeypatch, row)
    _warm()
    call, tensors = _op_call(dtype, hd, quant, direct=entry == "apply")
    _assert_refused(call, quant, dtype, hd, cls, _State(tensors))


def _layer(dtype, hd, quant, **kw):
    from sdm_testlib import make_layer
    cfg = dict(dim=2 * hd, num_heads=2, slots_per_head=1024, num_writes=8, num_reads=8, chunk_size=64,
               snapshot_quant=quant)
    cfg.update(kw)
    return make_layer(seed=3, dtype=dtype, **cfg)


@requires_cuda
@pytest.mark.parametrize("row", ["native", "a100"])
@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("cls,dtype,hd", LAYER_REPS, ids=[f"{c}_{str(d)[6:]}_hd{h}" for c, d, h in LAYER_REPS])
@pytest.mark.parametrize("mode", ["train", "cuda_graph"])
def test_the_layer_refuses_without_work(mode, cls, dtype, hd, quant, row, monkeypatch):
    """The layer's training call, eager and through its CUDA-graph runner: refused at the op's entry, nothing launched
    there and no module state moved."""
    _force(monkeypatch, row)
    _warm()
    layer = _layer(dtype, hd, quant, **({"cuda_graph": True} if mode == "cuda_graph" else {})).train()
    x = (torch.randn(2, 96, 2 * hd, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)) * 0.5
         ).to(dtype)
    _assert_refused(lambda: layer(x), quant, dtype, hd, cls, _State([x], layer=layer))


@requires_cuda
@pytest.mark.parametrize("cls,dtype,hd", LAYER_REPS, ids=[f"{c}_{str(d)[6:]}_hd{h}" for c, d, h in LAYER_REPS])
def test_the_same_shapes_run_without_a_backward_or_unquantized(cls, dtype, hd):
    """Not refused: the same heads without a backward (no_grad, inference_mode: nothing is quantized) and with
    snapshot_quant 'none' on a training call."""
    mo = _mo()
    layer = _layer(dtype, hd, "fp8")
    x = torch.randn(2, 96, 2 * hd, device="cuda", dtype=dtype)
    with torch.no_grad():
        y0, _ = layer(x)
    with torch.inference_mode():
        y1, _ = layer(x)
    assert torch.equal(y0, y1) and bool(torch.isfinite(y0.float()).all())
    call, _ = _op_call(dtype, hd, "none")
    call().float().sum().backward()
    assert mo.quant_tail_class(dtype, hd, "fp8", 4096) == cls


# ------------------------------------------------------------------------------------------ the op's gate (static)
# SDMWriteRead.forward's first statements: the recorded split, the layout (sdm_write_read's, else its own) and the gate,
# before any work. The gate (memory_ops._admit) refuses T0-T3 / S4 (refuse_quant_tail with the device), then the calls
# without a native route, and stores a signature only after both returned.
FWD_HEAD = ("edges = getattr(ctx, 'next_functions', None)", "recorded = edges is None or len(edges) > 0",
            "if lay is None:\n    lay = _layout(memory, k_idx, k_val, v, beta, g, q_idx, q_val, chunk_size, num_memory_slots,"
            " num_parallel)", "snapshot_quant = _admit(memory, lay, snapshot_quant, recorded, True, grad_bank_fp32)")
ADMIT_ORDER = ("hit = _NATIVE_MEMO.get(key)", "if hit is not None:\n    return hit",
               "if saving:\n    refuse_quant_tail(memory.dtype, lay.hd, snapshot_quant, rows, device=memory.device)",
               "refuse_non_native(memory, lay.C, snapshot_quant, rows, needs_backward, grad_bank_fp32, saving)")


def _gate_pins(src):
    """[] or what is wrong with the op's entry in the memory_ops source ``src``."""
    t = ast.parse(src)
    fns = {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)}
    sdm = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "SDMWriteRead")
    fwd = next(n for n in sdm.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    bad = []
    if [ast.unparse(s) for s in fwd.body[:4]] != list(FWD_HEAD):
        bad.append("SDMWriteRead.forward: not the recorded split, the layout and the gate first")
    swr = [ast.unparse(s) for s in fns["sdm_write_read"].body]
    i = next((k for k, s in enumerate(swr) if s.startswith("if needs_backward(")), None)
    if "lay = _layout(*args)" not in swr[:i or 0] or i is None or swr[i].split("\n")[1:] != [
            "    return SDMWriteRead.apply(*args, effective_snapshot_quant(memory, snapshot_quant), grad_bank_fp32, lay)"]:
        bad.append("sdm_write_read: its layout is not handed to the training forward")
    elif swr[i + 1] != "_admit(memory, lay)" or not swr[i + 2].startswith("with torch.no_grad(), _kernel_context():"):
        bad.append("sdm_write_read: the no-grad call is not admitted right before its work")
    adm = [ast.unparse(s) for s in fns["_admit"].body[1:]]   # (after its docstring)
    pos = [adm.index(s) if s in adm else -1 for s in ADMIT_ORDER]
    store = [k for k, s in enumerate(adm) if "_NATIVE_MEMO[key]" in s]
    if -1 in pos or pos != sorted(pos) or store != [len(adm) - 2] or not adm[-2].startswith("with _NATIVE_LOCK:"):
        bad.append("_admit: a signature is not stored after both refusals, and only then")
    return bad


GATE_MUTANTS = {   # negative controls: each breaks the gate's placement and must be detected
    "forward without the gate": ("        snapshot_quant = _admit(memory, lay, snapshot_quant, recorded, True, grad_bank_fp32)\n",
                                 ""),
    "forward gate in a forced mode": ("_admit(memory, lay, snapshot_quant, recorded, True,",
                                      "_admit(memory, lay, snapshot_quant, True, True,"),
    "forward recomputes the layout": ("        if lay is None:   #", "        if True:   #"),
    "the training call loses its layout": ("snapshot_quant), grad_bank_fp32, lay)", "snapshot_quant), grad_bank_fp32)"),
    "no-grad call without the gate": ("    _admit(memory, lay)   #", "    pass   #"),
    "a refused call stored": ("        return hit\n", "        return hit\n    _NATIVE_MEMO[key] = snapshot_quant\n"),
    "the native refusal skipped": ("    refuse_non_native(memory, lay.C, snapshot_quant, rows, needs_backward, grad_bank_fp32, "
                                   "saving)\n", ""),
}


def test_the_op_entry_admits_before_any_work():
    """The gate's placement (static), with its negative controls: the unmutated source passes, every mutant fails."""
    with open(MEMORY_OPS) as f:
        src = f.read()
    assert _gate_pins(src) == [], _gate_pins(src)
    for name, (old, new) in GATE_MUTANTS.items():
        assert src.count(old) == 1, name
        assert _gate_pins(src.replace(old, new)), name
