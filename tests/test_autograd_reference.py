# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The training op against a float64 autograd reference, in the shared-bank layout.

``test_f64_oracle.py`` drives the op with one bank region per stream ([P, T, *] inputs).
This file covers the other calling convention: 2-D [B*T, *] inputs with ``num_parallel=B``
over one shared bank (``num_memory_slots`` = whole bank), plus chunk-size invariance, the
fp32 output precision, the lossy backward-snapshot quantization modes and slot groups far
longer than a warp (one owner sums each group). The reference is the float64 token loop
in ``sdm_testlib.oracle_write_read``.
"""
import pytest
import torch

from sdm_testlib import (
    GATE_FP32,
    OP_GRADS,
    OP_INPUTS,
    QUANT_BOUNDS,
    assert_close_across,
    bits_equal,
    make_cotangent,
    make_op_inputs,
    oracle_write_read,
    rel_max,
    rel_rms,
    requires_cuda,
    slot_groups,
    streams_own_their_slots,
)

pytestmark = requires_cuda

NS, DIM, W, R = 512, 64, 4, 4
GATE_BF16 = 3e-2    # bf16 deviations from the reference: 3e-3..6e-3
# Relative RMS between two chunk sizes, bf16 bank (each chunk size is bitwise reproducible, so
# a fixed bound): at most 3.4e-3 (grad_beta, 16 vs 128).
CHUNK_SIZE_BF16 = 5e-3


def _shared(d):
    """The streams of ``d`` share one bank of B * NS rows, each on its own NS of them, which
    is what shared_bank=True asserts (the op cannot check it)."""
    assert streams_own_their_slots(d["k_idx"], NS) and streams_own_their_slots(d["q_idx"], NS)
    return dict(num_memory_slots=d["k_idx"].shape[0] * NS, num_parallel=d["k_idx"].shape[0],
                shared_bank=True)


def _run(d, cs, B, snapshot_quant="none", G=None):
    """Op forward + backward on the [B*T, *] layout over one shared bank."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    T = d["k_idx"].shape[1]
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}

    def flat(t):
        return t.reshape(B * T, -1)

    out = sdm_write_read(
        leaves["memory"], flat(d["k_idx"]), flat(leaves["k_val"]), flat(leaves["v"]),
        flat(leaves["beta"]), flat(leaves["g"]), flat(d["q_idx"]), flat(leaves["q_val"]),
        chunk_size=cs, snapshot_quant=snapshot_quant, **_shared(d))
    assert out.shape == (B * T, DIM) and out.dtype == d["v"].dtype
    G = make_cotangent(out, "dense") if G is None else G
    grads = torch.autograd.grad(out, [leaves[n] for n in OP_INPUTS], grad_outputs=G)
    res = {"out": out.detach().view(B, T, DIM), "restored": leaves["memory"].detach()}
    res.update(zip(OP_GRADS, grads))
    return res, G


# (B, T, dtype)
CONFIGS = [(1, 64, torch.float32), (2, 64, torch.float32), (4, 32, torch.float32),
           (1, 64, torch.bfloat16), (2, 64, torch.bfloat16), (4, 32, torch.bfloat16)]


@pytest.mark.parametrize("cs", [16, 32])
@pytest.mark.parametrize("B,T,dtype", CONFIGS,
                         ids=[f"B{b}_T{t}_{str(d).split('.')[-1]}" for b, t, d in CONFIGS])
def test_forward_and_all_gradients_match_the_float64_reference(B, T, dtype, cs, ieee_fp32):
    d = make_op_inputs(B, T, W, R, DIM, NS, dtype, seed=42)
    got, G = _run(d, cs, B)
    ref_out, ref_grads, _ = oracle_write_read(d, G.view(B, T, DIM))
    gate = GATE_FP32 if dtype is torch.float32 else GATE_BF16
    errs = {"out": rel_max(ref_out, got["out"])}
    errs.update({n: rel_max(ref_grads[n], got[n]) for n in OP_GRADS})
    bad = {k: v for k, v in errs.items() if not v < gate}
    assert not bad, f"B={B} T={T} cs={cs} {dtype}: " + ", ".join(
        f"{k}={v:.3e}" for k, v in bad.items())


def test_fp32_output_is_not_rounded_to_bf16(ieee_fp32):
    d = make_op_inputs(1, 64, W, R, DIM, NS, torch.float32, seed=42)
    got, _ = _run(d, 32, 1)
    out = got["out"]
    assert out.dtype == torch.float32
    assert (out - out.to(torch.bfloat16).float()).abs().max() > 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_chunk_size_invariance(dtype, ieee_fp32):
    """Different chunk sizes compute the same recurrence up to floating-point rounding."""
    B, T = 2, 128
    d = make_op_inputs(B, T, W, R, DIM, NS, dtype, seed=3)
    ref, G = _run(d, 16, B)
    if dtype is torch.float32:
        for cs in (32, 64, 128):
            got, _ = _run(d, cs, B, G=G)
            for k in ref:
                assert rel_max(ref[k], got[k]) < GATE_FP32, f"cs={cs} {k}"
        return
    runs16 = [ref] + [_run(d, 16, B, G=G)[0] for _ in range(2)]
    for cs in (32, 64, 128):
        runs = [_run(d, cs, B, G=G)[0] for _ in range(3)]
        # Different chunk sizes round the bf16 bank at different boundaries: each is
        # bitwise reproducible, and they differ by a fixed bound (CHUNK_SIZE_BF16).
        assert_close_across(runs16, runs, rms=CHUNK_SIZE_BF16, tiny_max=CHUNK_SIZE_BF16,
                            label=f"cs=16 vs cs={cs}")


@pytest.mark.parametrize("cs", [64, 256])
def test_long_slot_groups_match_the_float64_reference(cs, ieee_fp32):
    """Slots shared by every token: sdm_testlib.HOT_SLOTS slots in each token's write and read
    set give every chunk groups of cs writes and cs reads, far longer than a warp or one
    window of a group walk (T = 2 * cs + 9: two full chunks and a ragged one). At cs = 256
    the groups also exceed Phase 1's packed group counts, which saturate at POS_COUNT_MAX =
    127: the movers then find a group's bounds by walking its kind row. fp32 bank: readings,
    the final bank and every gradient at GATE_FP32."""
    from sparse_delta_memory.cuda.phase1_cuda import POS_COUNT_MAX
    from sparse_delta_memory.memory_ops import sdm_write_read

    B, T, Wl = 2, 2 * cs + 9, 8
    d = make_op_inputs(B, T, Wl, Wl, DIM, NS, torch.float32, seed=cs, slots="hot")
    longest = slot_groups(d["k_idx"], d["q_idx"], cs)["longest"]
    assert min(longest[:2]) >= cs and (cs < 2 * POS_COUNT_MAX
                                       or min(longest[:2]) > POS_COUNT_MAX), \
        f"premise: longest write/read groups {longest[:2]} at cs={cs}"
    leaves = {n: d[n].detach().clone().requires_grad_(True) for n in OP_INPUTS}

    def flat(t):
        return t.reshape(B * T, -1)

    out = sdm_write_read(
        leaves["memory"], flat(d["k_idx"]), flat(leaves["k_val"]), flat(leaves["v"]),
        flat(leaves["beta"]), flat(leaves["g"]), flat(d["q_idx"]), flat(leaves["q_val"]),
        chunk_size=cs, **_shared(d))
    final = leaves["memory"].detach().clone()
    G = make_cotangent(out, "dense")
    grads = torch.autograd.grad(out, [leaves[n] for n in OP_INPUTS], grad_outputs=G)
    ref_out, ref_grads, ref_bank = oracle_write_read(d, G.view(B, T, DIM))
    errs = {"out": rel_max(ref_out, out.detach().view(B, T, DIM)),
            "final_bank": rel_max(ref_bank, final)}
    errs.update({n: rel_max(ref_grads[n], g) for n, g in zip(OP_GRADS, grads)})
    bad = "  ".join(f"{k}={v:.3e}" for k, v in errs.items() if not v < GATE_FP32)
    assert not bad, f"cs={cs}: {bad}"


# Snapshot quantization stores the backward snapshot of the bank in fp8 or int4. The forward
# is unchanged; only the gradients that read the snapshot values (k_val, g, q_val) degrade.
@pytest.mark.parametrize("mode", ["fp8", "int4"])
def test_snapshot_quantization_only_perturbs_snapshot_gradients(mode, ieee_fp32):
    B, T = 2, 256
    d = make_op_inputs(B, T, 8, 8, DIM, NS, torch.float32, seed=1, weights="softmax")
    base, G = _run(d, 64, B)
    got, _ = _run(d, 64, B, snapshot_quant=mode, G=G)
    ref_out, ref_grads, _ = oracle_write_read(d, G.view(B, T, DIM))
    assert rel_max(ref_out, got["out"]) < GATE_FP32
    for n in ("grad_memory", "grad_v", "grad_beta"):
        assert rel_max(ref_grads[n], got[n]) < GATE_FP32, f"{mode}: {n} depends on the snapshot"
    for n in ("grad_k_val", "grad_g", "grad_q_val"):
        e_q, e_0 = rel_rms(ref_grads[n], got[n]), rel_rms(ref_grads[n], base[n])
        assert e_q < QUANT_BOUNDS[mode], f"{mode}: {n} error {e_q:.3e}"
        assert e_q > 10 * e_0, f"{mode}: {n} unchanged; is the quantized snapshot used?"


@pytest.mark.parametrize("mode", ["fp8", "int4"])
def test_quantized_backward_is_exact_on_a_representable_snapshot(mode, ieee_fp32):
    """Each chunk writes its own slots, so every snapshot row is an initial bank row, and the
    rows are integers times a power of two with the row amax at the largest code, which the
    quantizer represents exactly. The quantized backward then matches the unquantized one
    and restores the bank bitwise."""
    B, T, C = 2, 128, 32
    d = make_op_inputs(B, T, W, R, DIM, NS, torch.float32, seed=5)
    gen = torch.Generator().manual_seed(5)
    pool = NS // (T // C)
    local = torch.rand(B, T, pool, generator=gen).argsort(-1)[..., :W].sort(-1).values
    chunk = (torch.arange(T) // C).view(1, T, 1) * pool
    d["k_idx"] = (local + chunk + torch.arange(B).view(B, 1, 1) * NS).to(d["k_idx"])
    qmax, lo, hi, div = (448, -16, 17, 512) if mode == "fp8" else (7, -7, 8, 8)
    n = torch.randint(lo, hi, (B * NS, DIM), generator=gen).float()
    n[:, 0] = qmax
    exponent = torch.randint(-3, 1, (B * NS, 1), generator=gen).float()
    d["memory"] = (n * torch.exp2(exponent) / div).cuda()
    base, G = _run(d, C, B)
    got, _ = _run(d, C, B, snapshot_quant=mode, G=G)
    for k in ("out",) + OP_GRADS:
        assert rel_max(base[k], got[k]) < GATE_FP32, f"{mode}: {k}"
    assert bits_equal(got["restored"], d["memory"])
