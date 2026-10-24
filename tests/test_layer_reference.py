# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The whole layer against a float64 torch reference, and its behavior under recompute.

The reference rebuilds ``SparseDeltaMemory.forward`` from the layer's parameters with plain
torch ops: projections, einops head split, a dense product-key top-k, the unchunked token
loop of ``sdm_testlib.oracle_recurrence``, the per-head LayerNorm, einops head merge, the
sigmoid gate and ``Wo``. It shares no kernel, layout helper or adjoint with the layer, so
it checks what the layer computes (e.g. the head split/merge order that a checkpoint
depends on), not only that two layer paths agree.

The only thing taken from the layer is the slot selection on rows where the reference's
own top-k has a near-tie at the k-th boundary (float64 vs fp32 scores may order those
differently); on every other row the selections must be identical.
"""

import pytest
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from sdm_testlib import (
    TINY_NUMEL,
    SelectionSpy,
    layer_fwd_bwd,
    layer_inputs,
    make_layer,
    oracle_write_read,
    oracle_recurrence,
    assert_runs_bitwise,
    rel_rms,
    requires_cuda,
)

pytestmark = requires_cuda

# fp32 with TF32 off against float64. Calibrated over the cases below and two seeds:
# largest relative RMS deviation 9.2e-7 for the output and every gradient with more than
# TINY_NUMEL elements; the per-head gate gradients (A_log, dt_bias, gate biases: sums with
# heavy cancellation) reach 6.6e-5. A wrong head-merge order gives ~1.
GATE_REL_RMS = 1e-5
GATE_REL_RMS_TINY = 1e-3
# Rows whose k-th and (k+1)-th candidate scores are this close may be ordered differently.
TIE_GAP = 1e-4

LAYER_HEADS = [1, 2, 4]


def _ref_layer(heads, seed=0):
    return make_layer(seed=seed, dtype=torch.float32, dim=256, num_heads=heads,
                      slots_per_head=1024, num_writes=16, num_reads=16, chunk_size=64,
                      normalize_readings=True, output_gate=True)


def _dense_product_keys(raw, k, layer_idx):
    """Exact top-k of the outer sum of the two factors (float64), differentiable w.r.t. raw.

    Returns (val, idx) sorted by idx = a * side + b; rows with a near-tie at the k-th
    boundary take the layer's selection, every other row must match it.
    """
    side = raw.shape[-1] // 2
    total = (raw[..., :side].unsqueeze(-1) + raw[..., side:].unsqueeze(-2)).flatten(-2)
    kk = min(k, total.shape[-1])
    with torch.no_grad():
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
    val = raw.gather(-1, idx // side) + raw.gather(-1, side + idx % side)
    return val, idx


def _reference_forward(layer, x, sel):
    """float64 functional forward of ``layer`` on x [B, T, D] (train mode, no cache)."""
    P = {n: p.detach().double().clone().requires_grad_(True)
         for n, p in layer.named_parameters()}
    xx = x.detach().double().clone().requires_grad_(True)
    B, T, _ = xx.shape
    H, sph, hd = layer.num_heads, layer.slots_per_head, layer.head_dim
    a = layer.args

    def lin(name, inp):
        return F.linear(inp, P[name + ".weight"], P.get(name + ".bias"))

    g = -torch.exp(P["A_log"]) * F.softplus(lin("a_proj", xx) + P["dt_bias"])
    beta = torch.sigmoid(lin("b_proj", xx))
    split = lambda t: rearrange(t, "b t (h d) -> (b h) t d", h=H)  # noqa: E731
    v = split(lin("Wv_write", xx))
    k_val, k_idx = _dense_product_keys(split(lin("Wk_write", xx)), a.num_writes, sel.calls[0][1])
    q_val, q_idx = _dense_product_keys(split(lin("Wq_read", xx)), a.num_reads, sel.calls[1][1])
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
    return lin("Wo", readings), xx, P


def _reference_fwd_bwd(layer, x, gy, sel):
    out, xx, P = _reference_forward(layer, x, sel)
    out.backward(gy.double())
    res = {"out": out.detach(), "grad_x": xx.grad}
    for n, p in P.items():
        res["grad." + n] = p.grad
    return res


def _compare(layer, x, gy):
    sel = SelectionSpy(layer)
    got = layer_fwd_bwd(layer, x, gy)
    ref = _reference_fwd_bwd(layer, x, gy, sel)
    return {k: (rel_rms(ref[k], got[k]), ref[k].numel()) for k in ref}


@pytest.mark.parametrize("heads", LAYER_HEADS, ids=[f"h{h}" for h in LAYER_HEADS])
def test_layer_matches_the_float64_torch_reference(heads, ieee_fp32):
    layer = _ref_layer(heads)
    x, gy = layer_inputs(2, 128, 256, dtype=torch.float32)
    devs = _compare(layer, x, gy)
    bad = {k: f"{v:.2e}" for k, (v, n) in devs.items()
           if not v < (GATE_REL_RMS if n > TINY_NUMEL else GATE_REL_RMS_TINY)}
    assert not bad, f"h{heads}: {bad}"


def test_reference_comparison_detects_a_head_merge_permutation(monkeypatch, ieee_fp32):
    """Non-vacuity: merging heads as (d h) instead of (h d) is reported."""
    from sparse_delta_memory import layer as layer_mod

    monkeypatch.setattr(layer_mod, "head_merge",
                        lambda r, H: rearrange(r, "(b h) t d -> b t (d h)", h=H))
    layer = _ref_layer(4)
    x, gy = layer_inputs(2, 128, 256, dtype=torch.float32)
    devs = _compare(layer, x, gy)
    assert devs["out"][0] > 1e-2 and devs["grad.Wo.weight"][0] > 1e-2, devs


# ===================================================== activation-checkpoint recompute
def _ckpt_fwd_bwd(layer, x, x_recompute, gy):
    """Non-reentrant checkpoint whose recompute sees ``x_recompute`` instead of ``x``."""
    calls = {"n": 0}

    def fn(xi):
        calls["n"] += 1
        return layer(xi if calls["n"] == 1 else x_recompute)[0]

    for p in layer.parameters():
        p.grad = None
    out = checkpoint(fn, x.detach().clone(), use_reentrant=False)
    out.backward(gy)
    assert calls["n"] == 2, "the checkpoint did not recompute"
    res = {"grad." + n: p.grad.detach().clone() for n, p in layer.named_parameters()}
    for p in layer.parameters():
        p.grad = None
    return res


def test_checkpoint_recompute_rebuilds_all_backward_state(ieee_fp32):
    """The backward must be the gradient of the recomputed forward.

    The recompute sees a perturbed input (as when a nondeterministic upstream op makes
    recomputed activations differ), which changes about 2% of the selected slots. If any
    backward state came from the original forward instead, the gradient would mix two
    forwards. The recompute runs the plain step's computation on the perturbed input, so
    the gradients must equal the plain ones bitwise.
    """
    layer = _ref_layer(2)
    x, gy = layer_inputs(2, 256, 256, dtype=torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(5)
    x2 = x + 1e-3 * x.std() * torch.randn(x.shape, device="cuda", generator=gen)

    def plain(xx):
        r = layer_fwd_bwd(layer, xx, gy)
        return {k: v for k, v in r.items() if k.startswith("grad.")}

    ref = [plain(x2) for _ in range(2)]
    other = plain(x)
    ck = [_ckpt_fwd_bwd(layer, x, x2, gy) for _ in range(2)]
    # The test can fail: the two valid gradients differ far beyond the last bits.
    assert rel_rms(ref[0]["grad.Wk_write.weight"], other["grad.Wk_write.weight"]) > 1e-3
    assert_runs_bitwise(ref + ck, label="checkpoint recompute vs plain step")


# =========================================================== padded training sequences
PAD_CASES = [
    # (label, dtype, T, layer overrides, relative-RMS gate against float64)
    ("fp32_T100", torch.float32, 100, {}, 1e-5),
    ("fp32_T37", torch.float32, 37, {}, 1e-5),
    # bf16 at head_dim 128, W = R = C = 64; the bank is rounded to bf16 at every chunk
    # write, about 5e-3 relative RMS.
    ("bf16_k64_T2000", torch.bfloat16, 2000,
     dict(dim=512, num_heads=4, slots_per_head=4096, num_writes=64, num_reads=64), 2e-2),
]


@pytest.mark.parametrize("label,dtype,T,overrides,gate", PAD_CASES,
                         ids=[c[0] for c in PAD_CASES])
def test_padded_training_forward_leaves_the_exact_final_memory(label, dtype, T, overrides,
                                                               gate, ieee_fp32):
    """T not a multiple of chunk_size (the op masks the rest of the last chunk): the
    bank the training forward returns (read before backward) equals the reference."""
    cfg = dict(dim=256, num_heads=2, slots_per_head=1024, num_writes=16, num_reads=16,
               chunk_size=64, normalize_readings=True, output_gate=True)
    cfg.update(overrides)
    layer = make_layer(seed=0, dtype=dtype, **cfg)
    captured = {}
    orig = layer.gated_write_read

    def spy(memory, k_idx, k_val, v, beta, g, q_idx, q_val):
        captured.update(memory=memory.detach().clone(), k_idx=k_idx, k_val=k_val.detach(),
                        v=v.detach(), beta=beta.detach(), g=g.detach(), q_idx=q_idx,
                        q_val=q_val.detach())
        return orig(memory, k_idx, k_val, v, beta, g, q_idx, q_val)

    layer.gated_write_read = spy
    x, _ = layer_inputs(2, T, cfg["dim"], dtype=dtype)
    out, state = layer(x)
    final = state.memory.detach().clone()
    assert T % layer.args.chunk_size != 0
    _, _, ref_final = oracle_write_read(captured, sph=layer.slots_per_head)
    err = rel_rms(ref_final, final)
    assert err < gate, f"{label}: final bank {err:.2e}"
    out.float().sum().backward()
    bad = [n for n, p in layer.named_parameters()
           if p.grad is None or not bool(torch.isfinite(p.grad).all())]
    assert not bad, f"{label}: parameters without a finite gradient: {bad}"
