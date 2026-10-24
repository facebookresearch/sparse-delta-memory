# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Eval and inference paths of the layer and their agreement with training.

The layer routes on whether a call may need a backward (``memory_ops.needs_backward``:
grad mode on and an input requiring grad), not on train/eval mode:
  may need a backward   the chunked op saving its backward state, any T
  otherwise, T == 1     the fused decode step (``decode.sdm_decode_step``)
  otherwise, T > 1      the chunked op without backward state; it runs its steps in slabs
                        and its streams in groups, so its working memory does not grow
                        with T
Every path uses the same product-key selector, so identical projections select
identical slots.

The kernels are deterministic. Under no_grad, train and eval mode take the same no-grad
path, and 2-D and 3-D inputs the same op call: both pairs agree bitwise. Decode and prefill
round differently (the decode rounds the bf16 bank at every token, the prefill at every
chunk): observed at T=256, 4e-3 relative RMS in bf16 and 1e-6 in fp32. The decode-vs-
prefill gates below are set from those numbers.
"""
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from sdm_testlib import (
    OP_KERNELS,
    ProjectionReplay,
    SelectionSpy,
    assert_runs_bitwise,
    bits_equal,
    kernels_outside,
    layer_fwd_bwd,
    layer_inputs,
    make_layer,
    make_op_inputs,
    nograd_slab_factor,
    oracle_write_read,
    profiled_kernel_names,
    rel_max,
    rel_rms,
    requires_cuda,
)

K64 = dict(dim=512, num_heads=4, slots_per_head=4096, num_writes=64, num_reads=64,
           chunk_size=64, output_gate=True)
GENERIC = dict(dim=256, num_heads=2, slots_per_head=1024, num_writes=16, num_reads=16,
               chunk_size=64)
CONFIGS = [("k64", K64), ("generic_k16", GENERIC)]

DECODE_VS_PREFILL = {torch.float32: 1e-5, torch.bfloat16: 1.5e-2}


def _fwd(layer, x, train, cache=None):
    layer.train(train)
    with torch.no_grad():
        return layer(x, cache=cache)[0]


def _decode(layer, x, cache=None):
    layer.eval()
    if cache is None:
        cache = layer.create_kv_cache(bsz=x.shape[0], seq_len=0, dtype=x.dtype, device=x.device)
    outs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            o, cache = layer(x[:, t:t + 1], cache=cache)
            outs.append(o)
    return torch.cat(outs, 1), cache


class _SaveSpy:
    """Record the ``save`` flag of every forward of the chunked op."""

    def __init__(self, monkeypatch):
        from sparse_delta_memory import memory_ops

        self.saves, orig = [], memory_ops._sdm_forward

        def spy(*args, save, **kwargs):
            self.saves.append(save)
            return orig(*args, save=save, **kwargs)

        monkeypatch.setattr(memory_ops, "_sdm_forward", spy)


# ======================================================================== train vs eval
@requires_cuda
@pytest.mark.parametrize("label,cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_train_and_eval_prefill_agree_bitwise_and_select_the_same_slots(label, cfg):
    """Both under no_grad: the same no-grad op path, with the same slabs, stream groups and
    gather geometry (the mode only switches BatchNorm and memory statistics)."""
    layer = make_layer(seed=0, **cfg)
    spy = SelectionSpy(layer)
    x, _ = layer_inputs(2, 256, cfg["dim"])
    train = [_fwd(layer, x, True) for _ in range(3)]
    sel_train = spy.take()[:2]
    evals = [_fwd(layer, x, False) for _ in range(3)]
    sel_eval = spy.take()[:2]
    for (raw_t, idx_t), (raw_e, idx_e) in zip(sel_train, sel_eval):
        assert bits_equal(raw_t, raw_e), "precondition: identical projections"
        assert torch.equal(idx_t, idx_e), "train and eval selected different slots"
    assert_runs_bitwise([{"out": o} for o in train + evals], label=label)


@requires_cuda
@pytest.mark.parametrize("T", [1, 48, 128])
def test_eval_with_grad_gives_the_training_gradients(T, monkeypatch, ieee_fp32):
    """Train/eval mode does not select kernels: an eval-mode forward under grad saves the
    backward state and yields the training step's output and gradients (a single token
    included: it takes the chunked op, not the decode step)."""
    layer = make_layer(seed=0, dtype=torch.float32, **dict(GENERIC, output_gate=True))
    x, gy = layer_inputs(2, T, GENERIC["dim"], dtype=torch.float32, seed=T)
    spy = _SaveSpy(monkeypatch)
    train = layer_fwd_bwd(layer.train(), x, gy)
    evals = layer_fwd_bwd(layer.eval(), x, gy)
    assert spy.saves == [True, True]
    bad = {k: rel_max(train[k], evals[k]) for k in train
           if not rel_max(train[k], evals[k]) < 1e-5}
    assert not bad, bad


# ================================================================ decode vs prefill
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("label,cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_token_by_token_decode_matches_prefill(label, cfg, dtype, ieee_fp32):
    """The decode steps replay the prefill's selections: fp32 projections of one row and of
    T rows differ in the last bits, which can flip a near-tie at the k-th slot (the
    selector itself is layout invariant, see the next tests)."""
    layer = make_layer(seed=0, dtype=dtype, **cfg)
    x, _ = layer_inputs(2, 128, cfg["dim"], dtype=dtype, seed=3)
    select, recorded = layer._product_keys, []

    def record(raw, k):
        recorded.append(select(raw, k))
        return recorded[-1]

    layer._product_keys = record
    prefill = _fwd(layer, x, False)
    steps = iter([(val[:, t:t + 1], idx[:, t:t + 1])
                  for t in range(x.shape[1]) for val, idx in recorded])
    layer._product_keys = lambda raw, k: next(steps)
    decoded, cache = _decode(layer, x)
    assert cache.cache_len == x.shape[1] and next(steps, None) is None
    err = rel_rms(prefill, decoded)
    assert err < DECODE_VS_PREFILL[dtype], f"{label} {dtype}: decode vs prefill {err:.3e}"


@requires_cuda
@pytest.mark.parametrize("label,cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_bf16_decode_matches_prefill_end_to_end(label, cfg):
    """Without selection replay: the decode consumes the prefill's bf16 projection rows wherever their inputs are
    bit-identical (ProjectionReplay: nothing to replay where cuBLAS gives one row the bits of the same row among T rows,
    as on GB300; A100's cuBLAS does not), so the projections, hence the scores, are identical on every GPU; then both
    paths select the same slots and the whole layer agrees."""
    layer = make_layer(seed=0, **cfg)
    x, _ = layer_inputs(2, 128, cfg["dim"], seed=3)
    spy = SelectionSpy(layer)
    rep = ProjectionReplay(layer)
    with rep.record():
        prefill = _fwd(layer, x, False)
    (raw_k, idx_k), (raw_q, idx_q) = spy.take()
    with rep.decode():
        decoded, _ = _decode(layer, x)
    rep.close()
    calls = spy.take()
    for pos, raw_p, idx_p in ((0, raw_k, idx_k), (1, raw_q, idx_q)):
        raw_d = torch.cat([c[0] for c in calls[pos::2]], dim=1)
        idx_d = torch.cat([c[1] for c in calls[pos::2]], dim=1)
        assert bits_equal(raw_d, raw_p), f"precondition: identical bf16 projections ({rep.replaced} rows replayed)"
        assert torch.equal(idx_d, idx_p), "identical scores selected different slots"
    err = rel_rms(prefill, decoded)
    assert err < DECODE_VS_PREFILL[torch.bfloat16], f"{label}: decode vs prefill {err:.3e} ({rep.replaced} replayed)"


@requires_cuda
def test_the_decode_vs_prefill_difference_is_precision_shaped(ieee_fp32):
    errs = {}
    for dtype in (torch.float32, torch.bfloat16):
        layer = make_layer(seed=0, dtype=dtype, **GENERIC)
        x, _ = layer_inputs(2, 128, GENERIC["dim"], dtype=dtype, seed=3)
        errs[dtype] = rel_rms(_fwd(layer, x, False), _decode(layer, x)[0])
    assert errs[torch.bfloat16] > 100 * errs[torch.float32], errs


@requires_cuda
def test_decode_and_prefill_layouts_select_identical_slots():
    """The selector sees [BH, 1, K] rows at decode and [BH, T, K] rows at prefill; on the same
    scores it must return the same slots."""
    layer = make_layer(seed=0, dim=512, num_heads=4, slots_per_head=4096, num_writes=64,
                       num_reads=64, chunk_size=64)
    g = torch.Generator(device="cuda").manual_seed(0)
    raw = torch.randn(8, 96, 128, device="cuda", generator=g).to(torch.bfloat16)
    val_p, idx_p = layer._product_keys(raw, 64)
    for t in (0, 17, 95):
        val_d, idx_d = layer._product_keys(raw[:, t:t + 1].contiguous(), 64)
        assert torch.equal(idx_d[:, 0], idx_p[:, t]) and bits_equal(val_d[:, 0], val_p[:, t])


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_short_prefill_matches_decode_in_the_layer(dtype, ieee_fp32):
    """A prefill shorter than one block (T=48, C=64) through the op's valid-token mask."""
    layer = make_layer(seed=0, dtype=dtype, **GENERIC)
    x, _ = layer_inputs(2, 48, GENERIC["dim"], dtype=dtype, seed=4)
    err = rel_rms(_fwd(layer, x, False), _decode(layer, x)[0])
    assert err < DECODE_VS_PREFILL[dtype], f"short prefill vs decode {err:.3e}"


@requires_cuda
def test_decode_on_a_float32_cache_of_a_bfloat16_layer_matches_prefill(ieee_fp32):
    """Mixed dtypes: inputs are rounded to the bank dtype and the readings cast back."""
    layer = make_layer(seed=0, dtype=torch.bfloat16, **GENERIC).eval()
    x, _ = layer_inputs(2, 96, GENERIC["dim"], seed=9)

    def cache():
        return layer.create_kv_cache(bsz=2, seq_len=0, dtype=torch.float32, device="cuda")

    prefill = _fwd(layer, x, False, cache=cache())
    decoded, state = _decode(layer, x, cache())
    assert prefill.dtype == decoded.dtype == torch.bfloat16
    assert state.memory.dtype == torch.float32
    assert rel_rms(prefill, decoded) < DECODE_VS_PREFILL[torch.bfloat16]


# ========================================================= the op without backward state
def _no_grad_op(d, *, chunk_size, sph, layout="3d"):
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T = d["k_idx"].shape[:2]
    memory = d["memory"].clone()

    def lay(t):
        return t if layout == "3d" else t.reshape(P * T, -1)

    with torch.no_grad():
        out = sdm_write_read(memory, *(lay(d[n]) for n in ("k_idx", "k_val", "v", "beta", "g",
                                                          "q_idx", "q_val")),
                             chunk_size=chunk_size, num_memory_slots=sph,
                             num_parallel=P if layout == "3d" else 1)
    return out.view(P, T, -1), memory


# (P, T, C, W, head_dim, layout)
NO_GRAD_CASES = [(2, T, C, 8, 64, "3d") for T in (2, 37, 100, 2000) for C in (16, 64, 256)]
NO_GRAD_CASES += [(1, 100, 64, 8, 64, "3d"), (2, 100, 48, 8, 64, "2d"),
                  (2, 100, 64, 256, 1024, "3d")]


@requires_cuda
@pytest.mark.parametrize("P,T,C,W,hd,layout", NO_GRAD_CASES,
                         ids=[f"P{c[0]}_T{c[1]}_C{c[2]}_W{c[3]}_hd{c[4]}_{c[5]}"
                              for c in NO_GRAD_CASES])
def test_no_grad_forward_matches_the_float64_reference(P, T, C, W, hd, layout, ieee_fp32):
    """Readings and final bank, including a last chunk that is not full (fp32 bank)."""
    from sparse_delta_memory import memory_ops

    sph = 512 if W < 256 else 1024
    d = make_op_inputs(P, T, W, W, hd, sph, torch.float32, seed=P * T + C + W)
    ref, _, ref_bank = oracle_write_read(d, sph=sph)
    out, bank = _no_grad_op(d, chunk_size=C, sph=sph, layout=layout)
    assert memory_ops.last_dispatch()["save"] is False
    assert rel_max(ref, out) < 1e-5
    assert rel_max(ref_bank, bank) < 1e-5


@requires_cuda
@pytest.mark.parametrize("kwd", [False, True], ids=["gate_per_token", "gate_per_write"])
@pytest.mark.parametrize("layout", ["3d", "2d"])
def test_slab_and_stream_group_budgets_do_not_change_the_result(layout, kwd, monkeypatch,
                                                                ieee_fp32):
    """One step per slab and one stream per group equal the unbounded run."""
    from sparse_delta_memory import memory_ops

    P, T, C, W, hd, sph = 3, 300, 32, 16, 96, 1024
    d = make_op_inputs(P, T, W, W, hd, sph, torch.float32, seed=5)
    if kwd:
        d["g"] = d["g"] * d["k_val"].abs()
    runs = {}
    monkeypatch.setattr(memory_ops, "SLAB_CHUNKS_PER_SM", 0)
    for label, slab_bytes, step_bytes in (("unbounded", 1 << 40, 1 << 40), ("tiny", 1, 1)):
        monkeypatch.setattr(memory_ops, "SLAB_BYTES", slab_bytes)
        monkeypatch.setattr(memory_ops, "STEP_BYTES", step_bytes)
        runs[label] = _no_grad_op(d, chunk_size=C, sph=sph, layout=layout)
        rec = memory_ops.last_dispatch()
        runs[label + "_record"] = (rec["slab_steps"], rec["group_streams"])
    streams = P if layout == "3d" else 1
    steps = -(-(P * T // streams) // C)
    assert runs["unbounded_record"] == (steps, streams)
    assert runs["tiny_record"] == (1, 1)
    for a, b in zip(runs["unbounded"], runs["tiny"]):
        assert rel_max(a, b) < 1e-6


@requires_cuda
def test_grouped_no_grad_calls_gather_with_the_call_s_geometry(monkeypatch):
    """A no-grad call that runs its streams in groups (here 3 streams as groups of 2 and 1)
    sizes every group's gather from the call's full token count (the order the saving call
    reduces in); the dispatch record holds the last group's geometry. The forward computes the
    geometry (memory_ops.pinned_gather_geometry; both names of gather_geometry are watched) for a
    fixed reference SM count (shape-only, memory_ops._DET_GEOM_REF_SMS), never the device's."""
    from sparse_delta_memory import memory_ops
    from sparse_delta_memory.cuda import gather_cuda
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T, C, W, hd, sph = 3, 128, 128, 64, 128, 1024
    d = make_op_inputs(P, T, W, W, hd, sph, torch.bfloat16, seed=7)
    monkeypatch.setattr(memory_ops, "STEP_BYTES", 2 * C * W * hd * 2)   # two streams per group
    pinned = memory_ops._DET_GEOM_REF_SMS
    device_sms = 131         # the device's count: the pinned geometry must not see it
    monkeypatch.setattr(memory_ops, "_num_sms", lambda index: device_sms)
    calls, real = [], gather_cuda.gather_geometry
    assert memory_ops.gather_geometry is real

    def spy(*args, **kw):
        calls.append((args[3], kw.get("num_sms")))
        return real(*args, **kw)

    monkeypatch.setattr(memory_ops, "gather_geometry", spy)
    monkeypatch.setattr(gather_cuda, "gather_geometry", spy)
    with torch.no_grad():
        sdm_write_read(d["memory"].clone(), *(d[n] for n in ("k_idx", "k_val", "v", "beta", "g",
                                                            "q_idx", "q_val")),
                       chunk_size=C, num_memory_slots=sph, num_parallel=P)
    rec = memory_ops.last_dispatch()
    assert rec["save"] is False and rec["group_streams"] == 2
    tokens = [t for t, _ in calls]
    assert tokens == [P * C, P * C]
    assert all(n == pinned for _, n in calls), calls
    assert rec["gather"] == real(hd, torch.bfloat16, W, tokens[-1], num_sms=pinned)._asdict()


@requires_cuda
def test_no_grad_forward_saves_nothing_and_needs_little_memory():
    """At T=4096, W=64, head_dim 128 the working memory (beyond the readings) stays below a
    quarter of the snapshot a backward would save and does not grow at 2T, and only the
    op's kernels run."""
    from sparse_delta_memory import memory_ops
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T, W, hd, sph, C = 32, 4096, 64, 128, 4096, 64
    gen = torch.Generator(device="cuda").manual_seed(0)

    def slots():  # W distinct slots per token: a random start and consecutive rows
        start = torch.randint(0, sph, (P, T, 1), device="cuda", generator=gen)
        own = torch.arange(P, device="cuda").view(P, 1, 1) * sph
        return (start + torch.arange(W, device="cuda")) % sph + own

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).to(torch.bfloat16)

    memory = randn(P * sph, hd, scale=0.1)
    args = (memory, slots(), torch.softmax(randn(P, T, W).float(), -1).bfloat16(),
            randn(P, T, hd, scale=0.1), torch.rand(P, T, 1, device="cuda", generator=gen),
            -torch.rand(P, T, 1, device="cuda", generator=gen), slots(),
            torch.softmax(randn(P, T, W).float(), -1).bfloat16())
    kw = dict(chunk_size=C, num_memory_slots=sph, num_parallel=P)

    def peak(fn, *a):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*a, **kw)
        return torch.cuda.max_memory_allocated() - before - out.nbytes

    with torch.no_grad():
        sdm_write_read(*args, **kw)  # compiles the kernels
        work = peak(sdm_write_read, *args)
        _, kernels = profiled_kernel_names(sdm_write_read, *args, **kw)
        longer = [torch.cat([t, t], 1) if t.dim() == 3 else t for t in args]
        work_2t = peak(sdm_write_read, *longer)
    snapshot = P * T * W * hd * memory.element_size()
    assert memory_ops.last_dispatch()["save"] is False
    # Calibrated for 512 MiB slabs; the no-grad forward runs slabs nograd_slab_factor times as
    # large (1 GiB by default: fewer, larger launches), bounded the same way.
    limit = snapshot / 4 * nograd_slab_factor(memory_ops)
    assert work < limit, f"work {work / 2**20:.0f} MiB, snapshot {snapshot / 2**20:.0f} MiB, limit {limit / 2**20:.0f} MiB"
    assert work_2t < 1.05 * work, f"2T needs {work_2t / 2**20:.0f} vs {work / 2**20:.0f} MiB"
    assert not kernels_outside(kernels, OP_KERNELS)


# ============================================================ activation checkpointing
# (id, layer config, T, reentrant): a ragged T at head_dim 128, and a full last chunk at
# head_dim 64.
CHECKPOINT_CASES = [
    ("non_reentrant", dict(GENERIC, normalize_readings=True), 100, False),
    ("reentrant", dict(GENERIC, normalize_readings=True), 100, True),
    ("non_reentrant_dim128_T128", dict(GENERIC, dim=128, normalize_readings=True), 128, False),
]


@requires_cuda
@pytest.mark.parametrize("cfg,T,reentrant", [c[1:] for c in CHECKPOINT_CASES],
                         ids=[c[0] for c in CHECKPOINT_CASES])
def test_checkpointed_forwards_are_routed_by_grad_state(cfg, T, reentrant, monkeypatch,
                                                        ieee_fp32):
    """A non-reentrant checkpoint runs its forward under grad (it may need a backward); a
    reentrant one runs it without grad and recomputes it under grad. Either way the
    recompute re-expands the learned bank, and the output and every gradient equal those of
    a plain step."""
    layer = make_layer(seed=3, dtype=torch.float32, **cfg)
    x, gy = layer_inputs(2, T, cfg["dim"], dtype=torch.float32, seed=8)
    plain = layer_fwd_bwd(layer, x, gy)

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, t):
            return checkpoint(lambda u: self.inner(u)[0], t, use_reentrant=reentrant), None

    spy = _SaveSpy(monkeypatch)
    xi = x.detach().clone().requires_grad_(True)
    out, _ = Wrapped(layer)(xi)
    assert spy.saves == [not reentrant]
    out.backward(gy)
    assert spy.saves == [not reentrant, True]
    assert rel_max(plain["out"], out.detach()) < 1e-5
    assert rel_max(plain["grad_x"], xi.grad) < 1e-5
    for n, p in layer.named_parameters():
        assert rel_max(plain["grad." + n], p.grad) < 1e-5, n


# ============================================================ cached prefill and layouts
@requires_cuda
def test_prefill_continues_from_a_cache(ieee_fp32):
    """Prefill of [A, B] in one call equals prefill of A, then prefill (or decode) of B from
    the resulting cache."""
    layer = make_layer(seed=0, dtype=torch.float32, **GENERIC)
    x, _ = layer_inputs(2, 224, GENERIC["dim"], dtype=torch.float32, seed=5)
    full = _fwd(layer, x, False)

    cache = layer.create_kv_cache(bsz=2, seq_len=0, dtype=torch.float32, device="cuda")
    first = _fwd(layer, x[:, :128], False, cache=cache)
    assert cache.cache_len == 128
    bank_after_first = cache.memory.clone()
    second = _fwd(layer, x[:, 128:], False, cache=cache)
    assert cache.cache_len == 224
    assert rel_max(full, torch.cat([first, second], 1)) < 1e-5

    cache2 = layer.create_kv_cache(bsz=2, seq_len=128, dtype=torch.float32, device="cuda",
                                   initial_memory=bank_after_first)
    tail, _ = _decode(layer, x[:, 128:160], cache=cache2)
    assert rel_max(full[:, 128:160], tail) < 1e-5


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_2d_input_matches_3d_input(dtype):
    """The layer reshapes [B*T, D] inputs to [B, T, D] first: the same computation, so the
    outputs agree bitwise."""
    layer = make_layer(seed=0, dtype=dtype, **GENERIC)
    x, _ = layer_inputs(2, 96, GENERIC["dim"], dtype=dtype, seed=6)
    freqs_cis = torch.zeros(96, 2, device="cuda")   # only its first dim (seq_len) is used
    for train in (True, False):
        runs3 = [_fwd(layer, x, train) for _ in range(2)]
        layer.train(train)
        with torch.no_grad():
            out2 = layer(x.reshape(2 * 96, -1), freqs_cis=freqs_cis)[0]
        assert out2.shape == (2 * 96, GENERIC["dim"])
        assert_runs_bitwise([{"out": r.reshape(2 * 96, -1)} for r in runs3] + [{"out": out2}],
                            label=f"2d vs 3d train={train}")


# ====================================================================== create_kv_cache
def _cpu_layer(**kw):
    return make_layer(seed=0, dtype=torch.float32, device="cpu", **dict(GENERIC, **kw))


def test_create_kv_cache_shape_dtype_and_seed():
    layer = _cpu_layer()
    H, sph, hd = layer.num_heads, layer.slots_per_head, layer.head_dim
    for want in (torch.bfloat16, torch.float16, torch.float32):
        cache = layer.create_kv_cache(bsz=3, seq_len=0, dtype=want, device="cpu")
        assert cache.memory.dtype == want and cache.cache_len == 0
        assert tuple(cache.memory.shape) == (3 * H * sph, hd)
        assert len(cache) == 1 and cache[0] is cache.memory
    cache = layer.create_kv_cache(bsz=2, seq_len=0, dtype=torch.float32, device="cpu")
    bank = layer.memory.detach().reshape(H * sph, hd)
    for b in range(2):
        assert torch.equal(cache.memory[b * H * sph:(b + 1) * H * sph], bank)
    assert cache.memory.data_ptr() != layer.memory.data_ptr()


def test_create_kv_cache_copies_an_explicit_initial_memory():
    layer = _cpu_layer()
    H, sph, hd = layer.num_heads, layer.slots_per_head, layer.head_dim
    seed = torch.full((2 * H * sph, hd), 7.0)
    cache = layer.create_kv_cache(bsz=2, seq_len=11, dtype=torch.bfloat16, device="cpu",
                                  initial_memory=seed)
    assert cache.cache_len == 11 and cache.memory.dtype == torch.bfloat16
    assert torch.equal(cache.memory.float(), seed)
    seed.zero_()
    assert cache.memory.float().abs().max() == 7.0, "the cache aliases the caller's tensor"
    with pytest.raises(ValueError, match="initial_memory must have shape"):
        layer.create_kv_cache(bsz=3, seq_len=0, dtype=torch.float32, device="cpu",
                              initial_memory=torch.zeros(2 * H * sph, hd))


def test_create_kv_cache_without_a_learned_bank_starts_at_zero():
    layer = _cpu_layer(backprop_on_memory=False)
    assert layer.memory is None
    cache = layer.create_kv_cache(bsz=2, seq_len=0, dtype=torch.float32, device="cpu")
    assert cache.memory.abs().max() == 0
    cache.update_(cache.memory + 1, seq_len=5)
    assert cache.cache_len == 5
    cache.reset()
    assert cache.cache_len == 0 and cache.memory.abs().max() == 0
