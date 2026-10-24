# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for explicit per-layer sequence-mixer assignment (attn / swa / sdm).

Runs standalone (no pytest dependency, like tests/smoke_sdm.py) or under pytest. The
`resolve_layer_types` + mask-semantics checks are pure-CPU; the hybrid forward/backward
check needs a GPU (the SDM kernels JIT-compile on first use). Run in the release env:

    PYTHONPATH=. python tests/test_layer_types.py
"""
import os
import sys
from contextlib import contextmanager

import torch

from sparse_delta_memory import SparseDeltaMemoryArgs

try:
    from apps.main.transformer import LMTransformer, LMTransformerArgs, create_causal_mask
    from lingua.transformer import resolve_layer_types
except ModuleNotFoundError:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(_ROOT, "apps")) and os.path.isdir(os.path.join(_ROOT, "lingua")):
        raise  # a checkout whose training code does not import: a real failure
    import pytest  # a copy of tests/ outside the repository: these tests need its apps/ and lingua/
    pytest.skip("needs the repository checkout (apps/, lingua/)", allow_module_level=True)


@contextmanager
def raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None:
            assert match in str(e), f"expected {match!r} in error, got: {e}"
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


def _no_gpu(name):
    """Without a GPU: skip under pytest; print and return True when run standalone."""
    if torch.cuda.is_available():
        return False
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("needs a CUDA GPU")
    print(f"[SKIP] {name} (no GPU)")
    return True


# ------------------------------------------------------------- resolve_layer_types
def test_all_sentinel():
    assert resolve_layer_types(4, "all", [], []) == ["attn"] * 4
    assert resolve_layer_types(3, [], "all", []) == ["swa"] * 3
    assert resolve_layer_types(2, [], [], "all") == ["sdm"] * 2


def test_all_empty_defaults_to_attn():
    # BaseTransformer-level leniency: nothing set -> all attention (legacy behaviour).
    assert resolve_layer_types(5, [], [], []) == ["attn"] * 5
    assert resolve_layer_types(5, "", "", "") == ["attn"] * 5


def test_three_way_partition():
    types = resolve_layer_types(6, [4, 5], [0, 2], [1, 3])
    assert types == ["swa", "sdm", "swa", "sdm", "attn", "attn"]


def test_comma_string_spec():
    assert resolve_layer_types(4, "0,3", "1", "2") == ["attn", "swa", "sdm", "attn"]


def test_gap_raises():
    with raises(ValueError, match="does not cover"):
        resolve_layer_types(4, [0, 1], [2], [])  # layer 3 unassigned


def test_overlap_raises():
    with raises(ValueError, match="overlap"):
        resolve_layer_types(4, [0, 1, 2], [2], [3])  # layer 2 in attn & swa


def test_out_of_range_raises():
    with raises(ValueError, match="out of range"):
        resolve_layer_types(4, [0, 1, 2, 3, 4], [], [])


def test_duplicate_within_spec_raises():
    with raises(ValueError, match="duplicate"):
        resolve_layer_types(4, [0, 0, 1], [2], [3])


def test_rejects_bool_float_and_empty_token():
    with raises(ValueError):
        resolve_layer_types(4, [True, 1], [2], [3])   # bool is not a layer index
    with raises(ValueError):
        resolve_layer_types(4, [0, 1.5], [2], [3])    # non-integer float
    with raises(ValueError, match="empty token"):
        resolve_layer_types(4, "0,,1", [2], [3])      # empty comma token


# ------------------------------------------------------------- SWA head defaults (CPU)
def test_swa_inherits_base_gqa_when_mirroring():
    # SWA mirrors base attention (no swa_n_heads) -> inherits the base GQA n_kv_heads.
    args = LMTransformerArgs(dim=64, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=64,
                             max_seqlen=32, swa_at=[0], attn_at=[1], swa_window=8)
    m = LMTransformer(args)
    assert m.layers[0].attn_type == "swa"
    assert m.layers[0].n_heads == 8 and m.layers[0].n_kv_heads == 4


def test_swa_custom_heads_default_to_mha():
    # Custom swa_n_heads with no swa_n_kv_heads -> MHA (n_kv_heads == swa_n_heads),
    # not the base GQA ratio (which need not divide swa_n_heads).
    args = LMTransformerArgs(dim=64, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=64,
                             max_seqlen=32, swa_at=[0], attn_at=[1], swa_window=8, swa_n_heads=4)
    m = LMTransformer(args)
    assert m.layers[0].n_heads == 4 and m.layers[0].n_kv_heads == 4


def test_attn_gate():
    # Off by default: no w_gate on any attention layer (full/swa/llama unaffected).
    off = LMTransformer(LMTransformerArgs(dim=64, n_layers=1, n_heads=4, vocab_size=64,
                                          max_seqlen=16, attn_at="all"))
    assert not hasattr(off.layers[0].attention, "w_gate")
    # On: w_gate on every attention layer (full + swa); forward+backward finite (CPU).
    m = LMTransformer(LMTransformerArgs(dim=64, n_layers=2, n_heads=4, vocab_size=64,
                                        max_seqlen=16, swa_at=[0], attn_at=[1],
                                        swa_window=4, attn_gate=True))
    m.init_weights()
    assert hasattr(m.layers[0].attention, "w_gate") and hasattr(m.layers[1].attention, "w_gate")
    loss = m(torch.randint(0, 64, (2, 8)), target=torch.randint(0, 64, (2, 8)))
    loss.backward()
    assert torch.isfinite(loss).all() and m.layers[1].attention.w_gate.weight.grad is not None
    print("[PASS] attn_gate off-by-default; on -> gated fwd+bwd finite")


def test_full_attn_gate_gates_only_full_attention():
    # full_attn_gate: w_gate on the full-attention layers only, laid out as attn_gate's.
    kw = dict(dim=64, n_layers=3, n_heads=4, vocab_size=64, max_seqlen=16, swa_at=[0, 2],
              attn_at=[1], swa_window=4)
    plain = LMTransformer(LMTransformerArgs(**kw))
    m = LMTransformer(LMTransformerArgs(**kw, full_attn_gate=True))
    m.init_weights()
    gated = [i for i, layer in enumerate(m.layers) if hasattr(layer.attention, "w_gate")]
    assert gated == [1] and m.layers[1].attention.attn_gate
    assert not m.layers[0].attention.attn_gate and not m.layers[2].attention.attn_gate
    extra = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 64 * 64  # one dim x (n_heads * head_dim) gate
    every = LMTransformer(LMTransformerArgs(**kw, attn_gate=True))
    assert ({k: v.shape for k, v in m.state_dict().items() if "w_gate" in k}
            == {k: v.shape for k, v in every.state_dict().items() if "layers.1." in k and "w_gate" in k})
    loss = m(torch.randint(0, 64, (2, 8)), target=torch.randint(0, 64, (2, 8)))
    loss.backward()
    g = m.layers[1].attention.w_gate.weight.grad
    assert torch.isfinite(loss).all() and g is not None and g.abs().sum() > 0
    print("[PASS] full_attn_gate gates only the full-attention layers")


def test_param_count_excludes_sdm_memory():
    from lingua.metrics import get_num_params
    # backprop_on_memory default True; num_writes / num_reads: the former defaults, explicit
    sdm = SparseDeltaMemoryArgs(num_heads=2, slots_per_head=1024, num_writes=1, num_reads=1)
    m = LMTransformer(LMTransformerArgs(dim=64, n_layers=2, n_heads=4, vocab_size=64,
                                        max_seqlen=16, attn_at=[0], sdm_at=[1], sdm_args=sdm))
    tot = get_num_params(m, exclude_sdm_memory=False)
    act = get_num_params(m, exclude_sdm_memory=True)
    mem = m.layers[1].attention.memory.numel()
    assert mem > 0 and tot - act == mem
    print(f"[PASS] get_num_params excludes SDM memory bank ({mem:,} params)")


# ------------------------------------------------------------- mask semantics
def test_sdpa_causal_is_string():
    assert create_causal_mask(8, "sdpa", None) == "causal"


def test_sdpa_sliding_window_mask():
    S, w = 6, 3
    m = create_causal_mask(S, "sdpa", w, device="cpu")
    assert m.dtype == torch.bool and m.shape == (S, S)
    q = torch.arange(S)
    delta = q[:, None] - q[None, :]
    expected = (delta >= 0) & (delta < w)
    assert torch.equal(m, expected)
    # query 5 attends to the 3 most-recent keys {3,4,5}, not the older {0,1,2}
    assert m[5].tolist() == [False, False, False, True, True, True]
    # strictly causal: no attention to the future
    assert not m.triu(1).any()


# ------------------------------------------------------------- LMTransformer wiring
def test_lmtransformer_requires_explicit():
    args = LMTransformerArgs(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seqlen=32)
    with raises(ValueError, match="explicit layer-type"):
        LMTransformer(args)


def test_swa_without_window_raises():
    args = LMTransformerArgs(
        dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seqlen=32,
        swa_at="all",  # SWA layers but no swa_window
    )
    with raises(AssertionError, match="swa_window"):
        LMTransformer(args)


def test_hybrid_forward_backward():
    if _no_gpu("test_hybrid_forward_backward"):
        return
    torch.manual_seed(0)
    B, T, V, dim, nh = 2, 128, 256, 256, 8  # full-attn head_dim = dim/nh = 32
    sdm = SparseDeltaMemoryArgs(
        num_heads=1, slots_per_head=1024, num_reads=8, num_writes=8,
        chunk_size=64, output_gate=True,
        read_act="Softmax", write_act="Softmax",
    )
    args = LMTransformerArgs(
        dim=dim, n_layers=6, n_heads=nh, vocab_size=V, max_seqlen=T,
        init_base_std=0.02,
        swa_at=[0, 2], attn_at=[4], sdm_at=[1, 3, 5],
        swa_window=16, swa_n_heads=4, swa_n_kv_heads=2, swa_head_dim=64,  # != default 32
        sdm_args=sdm,
    )
    model = LMTransformer(args).cuda()
    model.init_weights()

    # Per-type wiring assertions.
    assert model.layer_types == ["swa", "sdm", "swa", "sdm", "attn", "sdm"]
    assert model.has_swa and model.swa_window == 16
    assert "64" in model.extra_rope  # SWA head_dim differs -> its own RoPE table
    assert model.layers[0].attn_type == "swa" and model.layers[0].n_heads == 4
    assert model.layers[0].head_dim == 64 and model.layers[0].n_kv_heads == 2
    assert model.layers[4].attn_type == "attn" and model.layers[4].head_dim == 32
    assert model.layers[1]._is_sdm

    tok = torch.randint(0, V, (B, T), device="cuda")
    tgt = torch.randint(0, V, (B, T), device="cuda")

    logits = model(tok)
    assert logits.shape == (B, T, V)
    assert torch.isfinite(logits).all()

    loss = model(tok, target=tgt)
    loss.backward()
    bad = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite grads: {bad}"
    print(f"[PASS] hybrid swa/attn/sdm forward+backward, loss={loss.item():.4f}")


def test_generation_with_swa():
    if _no_gpu("test_generation_with_swa"):
        return
    from lingua.tokenizer import build_tokenizer
    from apps.main.generate import (
        PackedCausalTransformerGenerator, PackedCausalTransformerGeneratorArgs,
    )
    tok = build_tokenizer("bytes")
    # SWA + full attention (no SDM) to exercise the windowed prefill/decode masks.
    args = LMTransformerArgs(
        dim=128, n_layers=4, n_heads=8, vocab_size=tok.n_words, max_seqlen=256,
        init_base_std=0.02, swa_at=[0, 2], attn_at=[1, 3],
        swa_window=8, swa_n_heads=4, swa_head_dim=64,
    )
    torch.manual_seed(0)
    model = LMTransformer(args).cuda()
    model.init_weights()
    model = model.to(torch.bfloat16).eval()
    assert model.has_swa and model.swa_window == 8

    gargs = PackedCausalTransformerGeneratorArgs(
        temperature=0.0, max_gen_len=12, max_tokens=128, dtype="bf16",
    )
    gen = PackedCausalTransformerGenerator(gargs, model, tok)
    assert gen.has_swa and gen.swa_window == 8   # no longer blocks SWA
    out, _, _ = gen.generate(["The quick brown fox", "hello"])
    assert isinstance(out, list) and len(out) == 2 and all(isinstance(g, str) for g in out)
    print("[PASS] generation with swa (windowed prefill+decode) ran end-to-end")


def test_generation_with_sdm():
    if _no_gpu("test_generation_with_sdm"):
        return
    from lingua.tokenizer import build_tokenizer
    from apps.main.generate import (
        PackedCausalTransformerGenerator, PackedCausalTransformerGeneratorArgs,
    )
    tok = build_tokenizer("bytes")
    sdm = SparseDeltaMemoryArgs(
        num_heads=1, slots_per_head=256, num_reads=4, num_writes=4, chunk_size=64,
        output_gate=True, read_act="Softmax", write_act="Softmax",
    )
    # Full 3-way hybrid: swa + full attn + sdm, exercising windowed masks AND SDM decode.
    args = LMTransformerArgs(
        dim=128, n_layers=4, n_heads=8, vocab_size=tok.n_words, max_seqlen=128,
        init_base_std=0.02, swa_at=[0], attn_at=[1], sdm_at=[2, 3],
        swa_window=8, swa_n_heads=4, swa_head_dim=64, sdm_args=sdm,
    )
    model = LMTransformer(args).cuda()
    model.init_weights()
    model = model.to(torch.bfloat16).eval()
    gargs = PackedCausalTransformerGeneratorArgs(
        temperature=0.0, max_gen_len=8, max_tokens=64, dtype="bf16",
    )
    gen = PackedCausalTransformerGenerator(gargs, model, tok)
    assert gen.has_sdm and gen.has_swa
    out, _, _ = gen.generate(["The quick brown fox", "hello"])  # processed one at a time
    assert isinstance(out, list) and len(out) == 2 and all(isinstance(g, str) for g in out)
    # decode caches are cleared afterwards -> model safe to reuse for a plain forward
    assert not any(hasattr(m, "_gen_cache") for m in model.modules())
    print("[PASS] generation with swa+attn+sdm hybrid ran end-to-end")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            if t.__name__ != "test_hybrid_forward_backward":
                print(f"[PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
