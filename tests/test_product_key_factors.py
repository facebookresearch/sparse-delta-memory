# Copyright (c) Meta Platforms, Inc. and affiliates.
"""``SparseDeltaMemoryArgs.product_key_factor_dims``: product keys of any number of factors.

``None`` keeps the two square factors of ``slots_per_head``, and ``[side, side]`` is that same
layer (same parameters, state dict and selections). Any other factor list sizes the key and
query projections to ``sum(dims)`` columns per head and the bank to ``prod(dims)`` rows per
head, and every path (training, prefill, decode) selects with ``product_key_topk`` over those
factors. The whole layer is checked against a float64 torch reference (as
tests/test_layer_reference.py) whose top-k enumerates every slot of the full product, so the
gradients of the key and query projections go through the top-k backward. The ``cuda_graph``
option and whole-step captures with factor lists are in tests/test_cuda_graph_option.py and
tests/test_cuda_graph.py, the kernel allowlist in tests/test_kernel_allowlist.py.
"""
import math

import pytest
import torch
from einops import rearrange
from torch.nn import functional as F

from sdm_testlib import (
    TINY_NUMEL,
    SelectionSpy,
    bits_equal,
    layer_fwd_bwd,
    layer_inputs,
    make_layer,
    oracle_recurrence,
    rel_rms,
    requires_cuda,
)

pytestmark = requires_cuda

F8 = [5, 5, 4, 4, 4, 4, 3, 3]  # 57,600 slots, 32 key columns per head
# (label, dtype, layer overrides): eight factors with widths below k, one factor (the key
# projection scores every slot), five factors with width-1 factors, two factors wider than
# the two- to four-factor kernels take, and a 2-to-4-factor list for comparison.
FACTOR_CONFIGS = [
    ("f8", torch.float32, dict(slots_per_head=57600, product_key_factor_dims=F8,
                               num_writes=16, num_reads=24)),
    ("f1", torch.float32, dict(slots_per_head=2048, product_key_factor_dims=[2048],
                               num_writes=24, num_reads=16)),
    ("f5_ones", torch.float32, dict(slots_per_head=96, product_key_factor_dims=[4, 1, 3, 1, 8],
                                    num_writes=8, num_reads=20)),
    ("f2_wide", torch.float32, dict(slots_per_head=140000, product_key_factor_dims=[70000, 2],
                                    num_writes=16, num_reads=16)),
    ("f3", torch.float32, dict(slots_per_head=3000, product_key_factor_dims=[10, 30, 10],
                               num_writes=16, num_reads=12)),
]
BASE = dict(dim=256, num_heads=2, chunk_size=64, normalize_readings=True,
            output_gate=True)
# As tests/test_layer_reference.py (fp32, TF32 off, against float64).
GATE_REL_RMS = 1e-5
GATE_REL_RMS_TINY = 1e-3
TIE_GAP = 1e-4
DECODE_VS_PREFILL = 1e-5


# ================================================================== configuration
class _ValueSpy:
    """Records (raw scores, values, slots) of every product-key call of a layer."""

    def __init__(self, layer):
        self.calls, self._orig = [], layer._product_keys
        layer._product_keys = self._spy

    def _spy(self, x, k):
        val, idx = self._orig(x, k)
        self.calls.append([t.detach().clone() for t in (x, val, idx)])
        return val, idx


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_square_factor_dims_are_the_default_layer(dtype):
    """[side, side] builds the default layer: same state dict, and the same projections,
    selections (values and slots) and top-k gradients in training, and the same decode."""
    a = make_layer(seed=0, dtype=dtype)
    b = make_layer(seed=0, dtype=dtype, product_key_factor_dims=[32, 32])
    assert a.product_key_factors == b.product_key_factors == (32, 32)
    assert a.key_size_per_head == b.key_size_per_head == 64
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(bits_equal(sa[n], sb[n]) for n in sa)
    x, gy = layer_inputs(2, 96, 256, dtype=dtype)
    spies = [_ValueSpy(a), _ValueSpy(b)]
    layer_fwd_bwd(a, x, gy)
    layer_fwd_bwd(b, x, gy)
    assert len(spies[0].calls) == len(spies[1].calls) == 2
    for ca, cb in zip(spies[0].calls, spies[1].calls):
        assert all(bits_equal(u, w) for u, w in zip(ca, cb))
        # The same top-k backward for the same cotangent.
        raw = [c[0].requires_grad_(True) for c in (ca, cb)]
        k = ca[2].shape[-1]
        outs = [spy._orig(r, k)[0] for spy, r in zip(spies, raw)]
        gv = torch.randn(outs[0].shape, device="cuda", generator=torch.Generator(
            device="cuda").manual_seed(k)).to(dtype)
        grads = [torch.autograd.grad(o, r, gv)[0] for o, r in zip(outs, raw)]
        assert bits_equal(grads[0], grads[1])
    dec = [_decode(layer, x[:, :24])[0] for layer in (a, b)]
    assert bits_equal(dec[0], dec[1])


@pytest.mark.parametrize("dims", [[1024], F8, [16, 8, 8], [7, 1, 3, 2, 2, 1], [32, 32, 1],
                                  [70000, 2]],
                         ids=["f1", "f8", "f3", "f6_ones", "f3_one", "f2_wide"])
def test_factor_dims_size_the_projections_and_the_bank(dims):
    sph = math.prod(dims)
    layer = make_layer(seed=0, dtype=torch.float32, slots_per_head=sph,
                       product_key_factor_dims=dims, num_writes=16, num_reads=16,
                       query_batchnorm=True)
    H, D = 2, 256
    assert layer.product_key_factors == tuple(dims)
    assert layer.key_size_per_head == sum(dims) and layer.key_size == H * sum(dims)
    assert layer.Wq_read.weight.shape == layer.Wk_write.weight.shape == (H * sum(dims), D)
    assert layer.query_bn.num_features == layer.key_bn.num_features == H * sum(dims)
    assert layer.memory.shape == (H * sph, D // H)
    state = layer.state_dict()
    other = make_layer(seed=1, dtype=torch.float32, slots_per_head=sph,
                       product_key_factor_dims=list(dims), num_writes=16, num_reads=16,
                       query_batchnorm=True)
    other.load_state_dict(state, strict=True)
    assert all(bits_equal(state[n].reshape(-1), other.state_dict()[n].reshape(-1)) for n in state)


@pytest.mark.parametrize("dims,sph,match", [
    ([32, 31], 1024, "address 992 slots per head, but slots_per_head=1024"),
    ([], 1024, "1 to 512 positive ints"),
    ([0, 1024], 1024, "1 to 512 positive ints"),
    ([-4, -256], 1024, "1 to 512 positive ints"),
    ([2.0, 512], 1024, "must be ints"),
    ([True, 1024], 1024, "must be ints"),
    ("1024", 1024, "sequence of ints"),
    ([1] * 512 + [1024], 1024, "1 to 512 positive ints"),
    ([65536, 32768], 2**31, "at most 2147483647"),
    ([1], True, "slots_per_head must be a positive int"),
    ([4, 4], 0, "slots_per_head must be a positive int"),
], ids=["product", "empty", "zero", "negative", "float", "bool", "string", "factors_513",
        "slots", "slots_per_head_bool", "slots_per_head_zero"])
def test_invalid_factor_dims_raise(dims, sph, match):
    from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

    with pytest.raises(ValueError, match=match):
        SparseDeltaMemory(SparseDeltaMemoryArgs(dim=256, num_heads=1, slots_per_head=sph, num_writes=1, num_reads=1,
                                                product_key_factor_dims=dims), layer_id=0)


def test_factor_dims_load_from_a_config():
    """The field goes through lingua's OmegaConf config loading (YAML lists and None)."""
    dataclass_from_dict = pytest.importorskip("lingua.args").dataclass_from_dict
    from sparse_delta_memory import SparseDeltaMemoryArgs

    args = dataclass_from_dict(SparseDeltaMemoryArgs,
                               {"slots_per_head": 57600, "product_key_factor_dims": F8})
    assert args.product_key_factor_dims == F8 and args.slots_per_head == 57600
    assert dataclass_from_dict(SparseDeltaMemoryArgs, {}).product_key_factor_dims is None


# ================================================================ float64 reference
def _decode_columns(idx, factors):
    cols, rest = [], idx
    for w in reversed(factors):
        cols.append(rest % w)
        rest = rest // w
    return cols[::-1]


def _dense_product_keys(raw, factors, k, layer_idx):
    """Exact top-k over every slot of the full product (float64), differentiable w.r.t. raw.

    Returns (val, idx) sorted by slot id; rows with a near-tie at the k-th boundary take the
    layer's selection, every other row must match it. val is the explicit sum of the selected
    slot's factor values.
    """
    parts = raw.split(list(factors), dim=-1)
    with torch.no_grad():
        total = parts[0]
        for x in parts[1:]:
            total = (total.unsqueeze(-1) + x.unsqueeze(-2)).flatten(-2)
        kk = min(k, total.shape[-1])
        top = total.topk(kk + 1 if kk < total.shape[-1] else kk, dim=-1)
        ref_idx = top.indices[..., :kk].sort(dim=-1).values
        if kk < total.shape[-1]:
            gap = (top.values[..., kk - 1] - top.values[..., kk]).abs()
            scale = top.values[..., kk - 1].abs().clamp_min(1.0)
            tie = gap <= TIE_GAP * scale
        else:
            tie = torch.zeros(ref_idx.shape[:-1], dtype=torch.bool, device=raw.device)
    agree = (ref_idx == layer_idx).all(-1)
    bad = ~agree & ~tie
    assert not bool(bad.any()), (
        f"layer selected different slots on {int(bad.sum())} rows without a near-tie")
    idx = torch.where(tie.unsqueeze(-1), layer_idx, ref_idx)
    val = sum(p.gather(-1, c) for p, c in zip(parts, _decode_columns(idx, factors)))
    return val, idx


def _reference_fwd_bwd(layer, x, gy, sel):
    """float64 functional forward + backward of ``layer`` on x [B, T, D] (train mode)."""
    P = {n: p.detach().double().clone().requires_grad_(True)
         for n, p in layer.named_parameters()}
    xx = x.detach().double().clone().requires_grad_(True)
    B = xx.shape[0]
    H, sph, hd = layer.num_heads, layer.slots_per_head, layer.head_dim
    a, factors = layer.args, tuple(layer.args.product_key_factor_dims)

    def lin(name, inp):
        return F.linear(inp, P[name + ".weight"], P.get(name + ".bias"))

    g = -torch.exp(P["A_log"]) * F.softplus(lin("a_proj", xx) + P["dt_bias"])
    beta = torch.sigmoid(lin("b_proj", xx))
    split = lambda t: rearrange(t, "b t (h d) -> (b h) t d", h=H)  # noqa: E731
    v = split(lin("Wv_write", xx))
    k_val, k_idx = _dense_product_keys(split(lin("Wk_write", xx)), factors, a.num_writes,
                                       sel.calls[0][1])
    q_val, q_idx = _dense_product_keys(split(lin("Wq_read", xx)), factors, a.num_reads,
                                       sel.calls[1][1])
    k_val = F.softmax(k_val * a.write_act_scale, dim=-1)
    q_val = F.softmax(q_val * a.read_act_scale, dim=-1)
    g = rearrange(g, "b t h -> (b h) t 1")
    beta = rearrange(beta, "b t h -> (b h) t 1")
    memory = P["memory"].reshape(1, H * sph, hd).expand(B, -1, -1).reshape(B * H * sph, hd)
    off = torch.arange(B * H, device=x.device).view(-1, 1, 1) * sph
    readings, _ = oracle_recurrence(memory, k_idx + off, k_val, v, beta, g, q_idx + off, q_val)
    readings = F.layer_norm(readings, (hd,), P["readings_norm.weight"],
                            P["readings_norm.bias"], a.norm_eps)
    readings = rearrange(readings, "(b h) t d -> b t (h d)", h=H)
    readings = torch.sigmoid(lin("Wog", xx)) * readings
    out = lin("Wo", readings)
    out.backward(gy.double())
    res = {"out": out.detach(), "grad_x": xx.grad}
    for n, p in P.items():
        res["grad." + n] = p.grad
    return res


@pytest.mark.parametrize("label,dtype,cfg", FACTOR_CONFIGS, ids=[c[0] for c in FACTOR_CONFIGS])
def test_layer_matches_the_float64_torch_reference(label, dtype, cfg, ieee_fp32):
    layer = make_layer(seed=0, dtype=dtype, **dict(BASE, **cfg))
    x, gy = layer_inputs(2, 100, BASE["dim"], dtype=dtype)
    sel = SelectionSpy(layer)
    got = layer_fwd_bwd(layer, x, gy)
    ref = _reference_fwd_bwd(layer, x, gy, sel)
    devs = {k: (rel_rms(ref[k], got[k]), ref[k].numel()) for k in ref}
    bad = {k: f"{v:.2e}" for k, (v, n) in devs.items()
           if not v < (GATE_REL_RMS if n > TINY_NUMEL else GATE_REL_RMS_TINY)}
    assert not bad, f"{label}: {bad}"
    # The key and query projections are trained through the top-k backward.
    assert float(got["grad.Wk_write.weight"].abs().sum()) > 0
    assert float(got["grad.Wq_read.weight"].abs().sum()) > 0


# ================================================================ prefill and decode
def _decode(layer, x):
    layer.eval()
    cache = layer.create_kv_cache(bsz=x.shape[0], seq_len=0, dtype=x.dtype, device=x.device)
    outs = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            o, cache = layer(x[:, t:t + 1], cache=cache)
            outs.append(o)
    return torch.cat(outs, 1), cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("label,cfg_dtype,cfg", FACTOR_CONFIGS, ids=[c[0] for c in FACTOR_CONFIGS])
def test_every_path_selects_over_the_factor_dims(label, cfg_dtype, cfg, dtype, ieee_fp32):
    """Training, no-grad prefill and decode select with product_key_topk over the factor
    dims; decode replaying the prefill's selections gives the prefill's output."""
    from sparse_delta_memory.product_key_topk import product_key_topk

    layer = make_layer(seed=0, dtype=dtype, **dict(BASE, **cfg))
    dims = tuple(cfg["product_key_factor_dims"])
    ks = (cfg["num_writes"], cfg["num_reads"])
    x, gy = layer_inputs(2, 70, BASE["dim"], dtype=dtype, seed=3)
    spy = SelectionSpy(layer)
    layer_fwd_bwd(layer.train(), x, gy)
    train = spy.take()
    with torch.no_grad():
        prefill = layer.eval()(x)[0]
    calls = spy.take()
    _decode(layer, x)
    decode = spy.take()
    assert len(train) == len(calls) == 2 and len(decode) == 2 * x.shape[1]
    for i, (raw, idx) in enumerate(train + calls + decode):
        assert raw.shape[-1] == sum(dims)
        assert bits_equal(idx, product_key_topk(raw, dims, ks[i % 2])[1]), f"{label} call {i}"
    # Decode replaying the prefill's selections (fp32 projections of one and of T rows may
    # differ in the last bits and flip a near-tie, as in tests/test_eval_paths.py).
    recorded, select = [], layer._product_keys

    def record(raw, k):
        recorded.append(select(raw, k))
        return recorded[-1]

    layer._product_keys = record
    with torch.no_grad():
        prefill = layer.eval()(x)[0]
    steps = iter([(val[:, t:t + 1], idx[:, t:t + 1])
                  for t in range(x.shape[1]) for val, idx in recorded])
    layer._product_keys = lambda raw, k: next(steps)
    decoded, cache = _decode(layer, x)
    assert cache.cache_len == x.shape[1] and next(steps, None) is None
    gate = DECODE_VS_PREFILL if dtype == torch.float32 else 1.5e-2
    err = rel_rms(prefill, decoded)
    assert err < gate, f"{label} {dtype}: decode vs prefill {err:.3e}"
