# Copyright (c) Meta Platforms, Inc. and affiliates.
"""snapshot_quant="bf16": the backward snapshot of a float32 bank stored as bfloat16 rows."""
import pytest
import torch

from sdm_testlib import (
    GATE_FP32,
    OP_GRADS,
    OP_INPUTS,
    QUANT_BOUNDS,
    QUANT_GRADS,
    QUANT_SIGNAL_RATIO,
    bits_equal,
    make_cotangent,
    make_op_inputs,
    oracle_write_read,
    rel_max,
    rel_rms,
    requires_cuda,
)

NS, R = 1024, 8
SNAP_FREE = ("grad_memory", "grad_v", "grad_beta")   # gradients the snapshot does not enter


def _run(d, C, quant, G=None, grad_bank_fp32=False):
    """Op forward + backward on private copies of ``d`` ([P, T, *] streams, stream p on rows [p NS, (p + 1) NS))."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    P = d["k_idx"].shape[0]
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}
    out = sdm_write_read(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
                         d["q_idx"], leaves["q_val"], chunk_size=C, num_memory_slots=NS, num_parallel=P,
                         snapshot_quant=quant, grad_bank_fp32=grad_bank_fp32)
    node = out.grad_fn
    saved = node.saved_tensors
    info = dict(quant=node.snapshot_quant, layout=getattr(node, "quant_layout", None),
                snap=saved[11].detach().clone(), scale=None if saved[12] is None else saved[12].detach().clone())
    final = leaves["memory"].detach().clone()
    G = make_cotangent(out, "dense") if G is None else G
    grads = torch.autograd.grad(out, [leaves[n] for n in OP_INPUTS], grad_outputs=G)
    res = {"out": out.detach(), "final": final, "restored": leaves["memory"].detach().clone()}
    res.update(zip(OP_GRADS, grads))
    return res, G, info


def _written_rows_bf16(init, k_idx):
    """The bank a bf16-snapshot backward restores: bf16(x) on every row the call wrote, x elsewhere."""
    want = init.clone()
    rows = k_idx.reshape(-1).unique()
    want[rows] = init[rows].to(torch.bfloat16).to(torch.float32)
    return want


# (P, T, W, head_dim, C): one column slice (64, 128: the quantized DW's in-walk rows), several (256, 1024: the
# cluster form on the modern backend, the all-slices form on the portable one).
SHAPES = [(2, 256, 8, 64, 64), (2, 256, 8, 128, 32), (1, 192, 16, 256, 48), (1, 128, 8, 1024, 32)]


@requires_cuda
@pytest.mark.parametrize("P,T,W,hd,C", SHAPES)
def test_bf16_snapshot_against_float64_oracle(P, T, W, hd, C, ieee_fp32):
    d = make_op_inputs(P, T, W, R, hd, NS, torch.float32, seed=hd + C, weights="softmax")
    base, G, info0 = _run(d, C, "none")
    got, _, info = _run(d, C, "bf16", G)
    ref_out, ref_grads, ref_final = oracle_write_read(d, G.reshape(P, T, hd), sph=NS)
    assert info["quant"] == "bf16" and info["snap"].dtype == torch.bfloat16
    # the forward is the unquantized one, bit for bit; within the fp32 gate of the oracle
    assert bits_equal(got["out"], base["out"]) and bits_equal(got["final"], base["final"])
    assert rel_max(ref_out, got["out"]) < GATE_FP32 and rel_max(ref_final, got["final"]) < GATE_FP32
    for n in SNAP_FREE:
        assert rel_max(ref_grads[n], got[n]) < GATE_FP32, f"{n} depends on the snapshot"
    dev = {n: rel_rms(ref_grads[n], got[n]) for n in QUANT_GRADS}
    dev0 = {n: rel_rms(ref_grads[n], base[n]) for n in QUANT_GRADS}
    bad = {n: e for n, e in dev.items() if not e < QUANT_BOUNDS["bf16"]}
    assert not bad, f"bf16 snapshot gradients beyond {QUANT_BOUNDS['bf16']}: {bad}"
    assert any(dev[n] >= QUANT_SIGNAL_RATIO * dev0[n] for n in QUANT_GRADS), (dev, dev0)
    # the backward restores bf16(x) on the written rows (the unquantized one restores x)
    assert bits_equal(base["restored"], d["memory"])
    assert bits_equal(got["restored"], _written_rows_bf16(d["memory"], d["k_idx"]))


@requires_cuda
@pytest.mark.parametrize("hd,C", [(128, 32), (1024, 32)])
def test_bf16_snapshot_is_exact_on_a_representable_snapshot(hd, C, ieee_fp32):
    """Each chunk writes its own slots, so every snapshot row is an initial bank row."""
    P, T, W = 2, 128, 8
    d = make_op_inputs(P, T, W, R, hd, NS, torch.float32, seed=5, weights="softmax")
    gen = torch.Generator().manual_seed(5)
    pool = NS // (T // C)
    local = torch.rand(P, T, pool, generator=gen).argsort(-1)[..., :W].sort(-1).values
    chunk = (torch.arange(T) // C).view(1, T, 1) * pool
    d["k_idx"] = (local + chunk + torch.arange(P).view(P, 1, 1) * NS).to(d["k_idx"])
    d["memory"] = d["memory"].to(torch.bfloat16).to(torch.float32)
    base, G, _ = _run(d, C, "none")
    got, _, info = _run(d, C, "bf16", G)
    assert info["quant"] == "bf16" and info["layout"] == 2
    for k in ("out", "final", "restored") + OP_GRADS:
        assert bits_equal(base[k], got[k]), k
    assert bits_equal(got["restored"], d["memory"])


@requires_cuda
def test_bf16_snapshot_against_the_bf16_bank_route(ieee_fp32):
    """On bf16-valued inputs: a float32 bank with bf16 snapshots."""
    P, T, W, hd, C = 2, 256, 8, 128, 64
    d16 = make_op_inputs(P, T, W, R, hd, NS, torch.bfloat16, seed=3, weights="softmax")
    d32 = {k: (v.float() if v.is_floating_point() else v) for k, v in d16.items()}
    probe, _, _ = _run(d32, C, "none")
    G = make_cotangent(probe["out"], "dense").to(torch.bfloat16)
    a, _, info_a = _run(d32, C, "bf16", G.float())
    a2, _, _ = _run(d32, C, "bf16", G.float(), grad_bank_fp32=True)
    b, _, info_b = _run(d16, C, "none", G, grad_bank_fp32=True)
    b2, _, info_b2 = _run(d16, C, "bf16", G, grad_bank_fp32=True)
    assert info_a["quant"] == "bf16" and info_b2["quant"] == "none"
    for k in ("out", "final", "restored") + OP_GRADS:
        assert bits_equal(a[k], a2[k]), f"grad_bank_fp32 changed {k} on a float32 bank"
        assert bits_equal(b[k], b2[k]), f"'bf16' on a bf16 bank is not 'none': {k}"
    ref_out, ref_grads, _ = oracle_write_read(d32, G.float().reshape(P, T, hd), sph=NS)
    ref = dict(ref_grads, out=ref_out)
    dev_a = {k: rel_rms(ref[k], a[k]) for k in ref}
    dev_b = {k: rel_rms(ref[k], b[k]) for k in ref}
    worse = {k: (dev_a[k], dev_b[k]) for k in ref if not dev_a[k] <= dev_b[k]}
    assert not worse, f"the bf16-snapshot float32 bank is not closer than the bf16 bank: {worse}"
    for k in ("out",) + SNAP_FREE:
        assert rel_max(ref[k], a[k]) < GATE_FP32, k


@requires_cuda
def test_bf16_snapshot_is_deterministic(ieee_fp32):
    P, T, W, hd, C = 2, 192, 16, 256, 64
    d = make_op_inputs(P, T, W, R, hd, NS, torch.float32, seed=9, slots="hot", weights="softmax")
    a, G, _ = _run(d, C, "bf16")
    b, _, _ = _run(d, C, "bf16", G)
    for k in a:
        assert bits_equal(a[k], b[k]), k


# ------------------------------------------------------------------------- the refusal without a CUDA route (no fallback)
ROUTE_TEMPLATE = (
    'sparse_delta_memory: snapshot_quant="bf16" on a float32 bank with head_dim {hd}, chunk_size {C} and {rows} compact '
    'snapshot rows has no CUDA route on this device ({dev}); it runs only on the CUDA kernels, with no fallback. '
    'snapshot_quant="none" keeps a float32 snapshot, and calls without a backward (torch.no_grad(), inference) are not '
    'limited.')


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16bank"])
@pytest.mark.parametrize("hd", [36, 128, 130])
def test_bf16_without_a_backward_is_none_bitwise(hd, dtype):
    """Calls without a backward are never limited and run 'none'."""
    import warnings
    from sparse_delta_memory.memory_ops import sdm_write_read
    from test_quant_tail_refusal import _layer

    P, T, W, C = 2, 128, 8, 32
    d = make_op_inputs(P, T, W, R, hd, NS, dtype, seed=hd + 1, weights="softmax")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for how in ("no_grad", "no_input_grad"):
            res = {}
            for quant in ("none", "bf16"):
                bank = d["memory"].clone()
                args = (bank, d["k_idx"], d["k_val"], d["v"], d["beta"], d["g"], d["q_idx"], d["q_val"])
                kw = dict(chunk_size=C, num_memory_slots=NS, num_parallel=P, snapshot_quant=quant)
                if how == "no_grad":
                    with torch.no_grad():
                        out = sdm_write_read(*args, **kw)
                else:
                    out = sdm_write_read(*args, **kw)
                assert out.grad_fn is None
                res[quant] = (out, bank)
            assert bits_equal(res["none"][0], res["bf16"][0]) and bits_equal(res["none"][1], res["bf16"][1]), how
        x = (torch.randn(2, 96, 2 * hd, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9)) * 0.5
             ).to(dtype)
        for ctx in (torch.no_grad, torch.inference_mode):
            outs = []
            for quant in ("none", "bf16"):
                layer = _layer(dtype, hd, quant).eval()
                with ctx():
                    outs.append(layer(x)[0])
            assert bits_equal(outs[0], outs[1]), ctx.__name__
    assert not [w for w in rec if "snapshot_quant" in str(w.message)], [str(w.message) for w in rec]


@requires_cuda
def test_bf16_float32_bank_refuses_2_31_snapshot_rows_up_front():
    """snapshot_quant "bf16" on a float32 bank at 2^31 or more compact snapshot rows."""
    from sparse_delta_memory import forward as fw
    from sparse_delta_memory import memory_ops as mo

    dev = torch.device("cuda", torch.cuda.current_device())
    view = torch.empty(4096, 128, dtype=torch.float32, device=dev)
    C, ok = 64, 2 ** 31 - 1
    assert fw.bf16_route(view, C, ok)
    assert mo.native_route_class(view, C, "bf16", ok, needs_backward=True) is None
    mo.refuse_bf16_without_route(view, C, ok)
    mo.refuse_non_native(view, C, "bf16", ok, needs_backward=True)
    for rows in (2 ** 31, 2 ** 31 + 4096, 2 ** 40):
        assert not fw.bf16_route(view, C, rows), rows
        assert mo.native_route_class(view, C, "bf16", rows, needs_backward=True) is not None, rows
        want = ROUTE_TEMPLATE.format(hd=128, C=C, rows=rows, dev=str(dev))
        for how in ("direct", "training"):
            with pytest.raises(mo.UnsupportedConfigurationError) as err:
                if how == "direct":
                    mo.refuse_bf16_without_route(view, C, rows)
                else:
                    mo.refuse_non_native(view, C, "bf16", rows, needs_backward=True)
            assert type(err.value) is mo.UnsupportedConfigurationError and err.value.tail_class is None, (how, rows)
            assert str(err.value) == want, (how, rows, str(err.value))
        # an unrecorded (saving, no backward) call: "bf16" runs as "none", so its admission is exactly 'none''s
        assert mo.native_route_class(view, C, "bf16", rows, saving=True) == mo.native_route_class(
            view, C, "none", rows, saving=True), rows
    # A bfloat16 bank runs "bf16" as "none" (2^31 rows are then 'none''s domain, not this refusal's).
    bview = torch.empty(4096, 128, dtype=torch.bfloat16, device=dev)
    assert mo.native_route_class(bview, C, "bf16", ok, needs_backward=True) == mo.native_route_class(
        bview, C, "none", ok, needs_backward=True)

