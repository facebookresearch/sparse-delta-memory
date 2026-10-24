# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Checkpoint compatibility of SparseDeltaMemory with the previous release of the layer.

The layer must expose the same parameters, in the same order (optimizer state is keyed by
parameter order), with the same shapes and the same initialization as before. The expected
values are written out here from the previous implementation rather than imported.
"""
import inspect
import math
import os
import warnings

import pytest
import torch
from torch import nn

from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

# (label, args overrides)
CONFIGS = [
    ("h1_defaults", dict(dim=256, num_heads=1, slots_per_head=1024)),
    ("h4_all_options", dict(dim=256, num_heads=4, slots_per_head=4096, output_gate=True,
                            normalize_readings=True, query_batchnorm=True)),
    ("h2_no_memory", dict(dim=512, num_heads=2, slots_per_head=65536, output_gate=True,
                          backprop_on_memory=False)),
]
IDS = [c[0] for c in CONFIGS]


def _args(**kw):
    return SparseDeltaMemoryArgs(**{"num_writes": 1, "num_reads": 1, **kw})   # the former defaults, explicit


def _expected_parameters(a):
    """Ordered (name, shape) of the previous layer's parameters."""
    H, dim = a.num_heads, a.dim
    key_size = H * 2 * math.isqrt(a.slots_per_head)
    hd = dim // H
    out = [("A_log", (H,)), ("dt_bias", (H,))]
    if a.backprop_on_memory:
        out.append(("memory", (H * a.slots_per_head, hd if H > 1 else dim)))
    out += [("Wq_read.weight", (key_size, dim)), ("Wq_read.bias", (key_size,)),
            ("Wk_write.weight", (key_size, dim)), ("Wk_write.bias", (key_size,)),
            ("Wv_write.weight", (dim, dim)), ("Wv_write.bias", (dim,))]
    if a.query_batchnorm:
        out += [("query_bn.weight", (key_size,)), ("query_bn.bias", (key_size,)),
                ("key_bn.weight", (key_size,)), ("key_bn.bias", (key_size,))]
    out += [("a_proj.weight", (H, dim)), ("a_proj.bias", (H,)),
            ("b_proj.weight", (H, dim)), ("b_proj.bias", (H,)),
            ("Wo.weight", (dim, dim)), ("Wo.bias", (dim,))]
    if a.output_gate:
        out += [("Wog.weight", (dim, dim)), ("Wog.bias", (dim,))]
    if a.normalize_readings:
        out += [("readings_norm.weight", (hd,)), ("readings_norm.bias", (hd,))]
    return out


def _expected_buffers(a):
    if not a.query_batchnorm:
        return []
    ks = a.num_heads * 2 * math.isqrt(a.slots_per_head)
    return [(f"{bn}.{n}", s) for bn in ("query_bn", "key_bn")
            for n, s in (("running_mean", (ks,)), ("running_var", (ks,)),
                         ("num_batches_tracked", ()))]


@pytest.mark.parametrize("label,kw", CONFIGS, ids=IDS)
def test_parameter_names_order_and_shapes_are_unchanged(label, kw):
    a = _args(**kw)
    layer = SparseDeltaMemory(a, layer_id=0)
    got = [(n, tuple(p.shape)) for n, p in layer.named_parameters()]
    assert got == _expected_parameters(a)
    sd = layer.state_dict()
    want_keys = {n for n, _ in _expected_parameters(a)} | {n for n, _ in _expected_buffers(a)}
    assert set(sd) == want_keys, set(sd) ^ want_keys
    for n, s in _expected_buffers(a):
        assert tuple(sd[n].shape) == s
    assert getattr(layer.A_log, "_no_weight_decay", False)
    assert getattr(layer.dt_bias, "_no_weight_decay", False)
    if layer.memory is not None:
        assert getattr(layer.memory, "_sdm_memory_bank", False)
    assert layer.key_size_per_head == 2 * math.isqrt(a.slots_per_head)


def _previous_constructor_parameters(a, seed):
    """The parameters the previous constructor drew from the global RNG, in its order."""
    torch.manual_seed(seed)
    H, dim = a.num_heads, a.dim
    ks = H * 2 * math.isqrt(a.slots_per_head)
    mods = {"Wq_read": nn.Linear(dim, ks), "Wk_write": nn.Linear(dim, ks),
            "Wv_write": nn.Linear(dim, dim), "a_proj": nn.Linear(dim, H),
            "b_proj": nn.Linear(dim, H)}
    A_min, A_max = a.A_init_range
    A_log = torch.log(torch.empty(H).uniform_(A_min, A_max).clamp(min=1e-4))
    dt_min, dt_max = 0.001, 0.1
    dt = torch.exp(torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
                   + math.log(dt_min)).clamp(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    mods["Wo"] = nn.Linear(dim, dim)
    if a.output_gate:
        mods["Wog"] = nn.Linear(dim, dim)
    out = {"A_log": A_log, "dt_bias": dt_bias}
    for name, m in mods.items():
        out[name + ".weight"], out[name + ".bias"] = m.weight.detach(), m.bias.detach()
    return out


@pytest.mark.parametrize("label,kw", CONFIGS, ids=IDS)
def test_constructor_draws_the_same_random_parameters(label, kw):
    a = _args(**kw)
    want = _previous_constructor_parameters(a, seed=123)
    torch.manual_seed(123)
    layer = SparseDeltaMemory(a, layer_id=0)
    got = dict(layer.named_parameters())
    for n, t in want.items():
        assert torch.equal(got[n].detach(), t), n
    if layer.memory is not None:
        assert layer.memory.abs().max() == 0


def _previous_init_weights(layer, init_std=None, factor=1.0):
    """The previous ``init_weights``, step for step (same RNG consumption)."""
    in_std = (init_std or (layer.dim ** (-0.5))) / factor
    for w in [layer.Wq_read, layer.Wk_write, layer.Wv_write, layer.Wo,
              layer.a_proj, layer.b_proj]:
        nn.init.trunc_normal_(w.weight, mean=0.0, std=in_std, a=-3 * in_std, b=3 * in_std)
        if w.bias is not None:
            nn.init.zeros_(w.bias)
    if layer.Wog is not None:
        nn.init.trunc_normal_(layer.Wog.weight, mean=0.0, std=in_std, a=-3 * in_std, b=3 * in_std)
        if layer.Wog.bias is not None:
            nn.init.zeros_(layer.Wog.bias)
    A_min, A_max = layer.args.A_init_range
    with torch.no_grad():
        layer.A_log.uniform_(A_min, A_max).clamp_(min=1e-4).log_()
        layer.dt_bias.uniform_(math.log(0.001), math.log(0.1)).exp_().clamp_(min=1e-4)
        layer.dt_bias.data.copy_(layer.dt_bias.data + torch.log(-torch.expm1(-layer.dt_bias.data)))
    if layer.memory is not None:
        nn.init.trunc_normal_(layer.memory, mean=0.0, std=in_std, a=-3 * in_std, b=3 * in_std)
    if layer.readings_norm is not None:
        layer.readings_norm.reset_parameters()
    for norm in [layer.query_bn, layer.key_bn]:
        if norm is not None:
            norm.reset_parameters()


@pytest.mark.parametrize("init_std,factor", [(None, 1.0), (0.02, 2.0)])
@pytest.mark.parametrize("label,kw", CONFIGS, ids=IDS)
def test_init_weights_matches_the_previous_initialization(label, kw, init_std, factor):
    a = _args(**kw)
    new = SparseDeltaMemory(a, layer_id=0)
    old = SparseDeltaMemory(a, layer_id=0)
    torch.manual_seed(7)
    new.init_weights(init_std=init_std, factor=factor)
    torch.manual_seed(7)
    _previous_init_weights(old, init_std=init_std, factor=factor)
    for (n, p), (_, q) in zip(new.named_parameters(), old.named_parameters()):
        assert torch.equal(p, q), n
    # Sanity of the distributions themselves.
    assert bool((new.A_log.exp() <= a.A_init_range[1]).all())
    dt = torch.nn.functional.softplus(new.dt_bias)
    assert bool((dt > 0.0009).all()) and bool((dt < 0.11).all())


@pytest.mark.parametrize("label,kw", CONFIGS, ids=IDS)
def test_a_previous_format_state_dict_loads_strictly(label, kw):
    a = _args(**kw)
    g = torch.Generator().manual_seed(0)
    sd = {n: torch.randn(s, generator=g) * 0.02 for n, s in _expected_parameters(a)}
    for n, s in _expected_buffers(a):
        sd[n] = torch.zeros(s, dtype=torch.long) if n.endswith("tracked") else torch.ones(s)
    layer = SparseDeltaMemory(a, layer_id=0)
    layer.load_state_dict(sd, strict=True)
    for n, p in layer.named_parameters():
        assert torch.equal(p.detach(), sd[n]), n


@pytest.mark.parametrize("kw,match", [
    (dict(slots_per_head=1000), "perfect square"),
    (dict(slots_per_head=65536 ** 2), "largest product-key bank"),
    (dict(num_heads=3), "divisible"),
    (dict(num_writes=257), "num_writes"),
    (dict(num_reads=0), "num_reads"),
    (dict(num_writes=32, slots_per_head=16), "num_writes"),
    (dict(chunk_size=15), "chunk_size"),
    (dict(chunk_size=257), "chunk_size"),
    (dict(snapshot_quant="fp4"), "snapshot_quant"),
])
def test_invalid_config_is_rejected(kw, match):
    base = dict(dim=256, num_heads=2, slots_per_head=4096)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        SparseDeltaMemory(_args(**base), layer_id=0)


def test_memory_block_size_is_the_former_name_of_chunk_size():
    with pytest.warns(DeprecationWarning, match="memory_block_size"):
        args = _args(dim=256, num_heads=2, slots_per_head=4096, memory_block_size=64)
    assert (args.chunk_size, args.memory_block_size) == (64, None)
    assert SparseDeltaMemory(args, layer_id=0).chunk_size == 64
    with pytest.warns(DeprecationWarning):  # both names with the same value
        assert _args(chunk_size=128, memory_block_size=128).chunk_size == 128
    with pytest.raises(ValueError, match="memory_block_size"):
        _args(chunk_size=64, memory_block_size=128)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="chunk_size"):
        SparseDeltaMemory(_args(dim=256, num_heads=2, slots_per_head=4096,
                                memory_block_size=15), layer_id=0)


def test_chunk_size_none_is_the_op_default():
    from sparse_delta_memory.memory_ops import DEFAULT_CHUNK, sdm_write_read

    layer = SparseDeltaMemory(_args(dim=256, num_heads=2, slots_per_head=4096), layer_id=0)
    assert layer.args.chunk_size is None and layer.chunk_size == DEFAULT_CHUNK == 64
    assert inspect.signature(sdm_write_read).parameters["chunk_size"].default == DEFAULT_CHUNK
    for explicit in (64, 256):  # explicit values are kept, including the former default 256
        layer = SparseDeltaMemory(_args(dim=256, num_heads=2, slots_per_head=4096, chunk_size=explicit),
                                  layer_id=0)
        assert layer.chunk_size == explicit


def test_previous_release_debug_config_loads():
    """The previous release's apps/main/configs/debug_sdm.yaml (with memory_block_size: 64),
    copied verbatim, loads into this version's model and training configs with chunk_size 64,
    and a config saved by this version loads back without the warning."""
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    pytest.importorskip("apps.main.transformer", reason="needs the repository checkout (apps/, lingua/)")
    from apps.main.transformer import LMTransformerArgs
    from lingua.args import dataclass_from_dict, dataclass_to_dict

    path = os.path.join(os.path.dirname(__file__), "data", "debug_sdm_previous_release.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert cfg["model"]["sdm_args"]["memory_block_size"] == 64
    with pytest.warns(DeprecationWarning, match="memory_block_size"):
        model = dataclass_from_dict(LMTransformerArgs, cfg["model"])
    assert (model.sdm_args.chunk_size, model.sdm_args.memory_block_size) == (64, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        again = dataclass_from_dict(LMTransformerArgs, dataclass_to_dict(model))
    assert again.sdm_args.chunk_size == 64
    # the training entry point's loader (apps/main/train.py main)
    train = pytest.importorskip("apps.main.train")
    with pytest.warns(DeprecationWarning, match="memory_block_size"):
        args = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(train.TrainArgs()),
                                                   OmegaConf.load(path)))
    assert args.model.sdm_args.chunk_size == 64


@pytest.mark.parametrize("kw,want", [(dict(), 64), (dict(chunk_size=64), 64), (dict(chunk_size=256), 256),
                                     (dict(memory_block_size=32), 32)])
def test_chunk_size_reaches_the_op(monkeypatch, kw, want):
    """The op gets the resolved chunk size: 64 for the default None (256 before 2.0.0)."""
    import sparse_delta_memory.layer as layer_module

    seen = []

    def fake_write_read(memory, k_idx, k_val, v, beta, g, q_idx, q_val, *, chunk_size, **_):
        seen.append(chunk_size)
        return torch.zeros(k_idx.shape[0] * k_idx.shape[1], v.shape[-1])

    monkeypatch.setattr(layer_module, "sdm_write_read", fake_write_read)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        layer = SparseDeltaMemory(_args(dim=64, num_heads=2, slots_per_head=256, **kw), layer_id=0)
    BH, T, hd = 2, 4, 32
    idx = torch.zeros(BH, T, 1, dtype=torch.long)
    layer.gated_write_read(torch.zeros(BH * 256, hd, requires_grad=True), idx,
                           torch.ones(BH, T, 1, requires_grad=True), torch.zeros(BH, T, hd),
                           torch.ones(BH, T, 1), torch.zeros(BH, T, 1), idx, torch.ones(BH, T, 1))
    assert seen == [want]


def test_grad_bank_fp32_is_opt_in():
    """grad_bank_fp32 defaults to False in the layer args and in the op."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    assert _args(dim=256, num_heads=2, slots_per_head=4096).grad_bank_fp32 is False
    assert inspect.signature(sdm_write_read).parameters["grad_bank_fp32"].default is False


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "apps", "main", "configs")
# The shipped training configs are part of the repository checkout, not of the package.
needs_configs = pytest.mark.skipif(not os.path.isdir(CONFIG_DIR),
                                   reason="needs the repository checkout (apps/main/configs)")


SDM_CONFIGS = sorted(f for f in os.listdir(CONFIG_DIR)
                     if f.endswith(".yaml") and "sdm_args" in open(os.path.join(CONFIG_DIR, f)).read()
                     ) if os.path.isdir(CONFIG_DIR) else []


@needs_configs
def test_the_sdm_configs_are_found():
    assert {"debug_sdm.yaml", "debug_hybrid.yaml", "sdm_flagship.yaml", "sdm_7B.yaml"} <= set(SDM_CONFIGS)


@needs_configs
@pytest.mark.parametrize("name", SDM_CONFIGS)
def test_shipped_configs_load_through_the_training_loader(name):
    """Every shipped SDM training config merges into the training entry point's TrainArgs
    (every key one of the args) without a deprecation warning, and none sets grad_bank_fp32."""
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    train = pytest.importorskip("apps.main.train")
    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, name))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # shipped configs use chunk_size
        merged = OmegaConf.merge(OmegaConf.structured(train.TrainArgs()), cfg)
    assert merged.model.sdm_args.grad_bank_fp32 is False


@pytest.mark.parametrize("kw,want", [(dict(), False), (dict(grad_bank_fp32=True), True),
                                     (dict(memory_block_size=32), False)])
def test_grad_bank_fp32_reaches_the_op(monkeypatch, kw, want):
    """The op gets the layer's grad_bank_fp32 setting (eager path), in a call that binds to the
    op's signature (no argument the op does not take)."""
    import sparse_delta_memory.layer as layer_module

    op = inspect.signature(layer_module.sdm_write_read)
    seen = []

    def fake_write_read(*args, **kwargs):
        op.bind(*args, **kwargs)
        seen.append(kwargs["grad_bank_fp32"])
        k_idx, v = args[1], args[3]
        return torch.zeros(k_idx.shape[0] * k_idx.shape[1], v.shape[-1])

    monkeypatch.setattr(layer_module, "sdm_write_read", fake_write_read)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        layer = SparseDeltaMemory(_args(dim=64, num_heads=2, slots_per_head=256, **kw), layer_id=0)
    BH, T, hd = 2, 4, 32
    idx = torch.zeros(BH, T, 1, dtype=torch.long)
    layer.gated_write_read(torch.zeros(BH * 256, hd, requires_grad=True), idx,
                           torch.ones(BH, T, 1, requires_grad=True), torch.zeros(BH, T, hd),
                           torch.ones(BH, T, 1), torch.zeros(BH, T, 1), idx, torch.ones(BH, T, 1))
    assert seen == [want]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_unsupported_dtype_is_rejected(dtype):
    """float16 / float64: the forward raises and leaves the layer's state unchanged."""
    from sdm_testlib import bits_equal
    layer = SparseDeltaMemory(_args(dim=64, num_heads=2, slots_per_head=256), layer_id=0).to("cuda", dtype)
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        layer(torch.zeros(1, 4, 64, device="cuda", dtype=dtype))
    assert all(bits_equal(before[k], v) for k, v in layer.state_dict().items())
