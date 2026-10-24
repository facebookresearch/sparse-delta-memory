# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The training op (forward and all six input gradients) against a float64 reference.

Reference: ``sdm_testlib.oracle_write_read``, an unchunked float64 token loop
differentiated by autograd. It never chunks, never launches a Triton/CUDA kernel and has no
hand-written adjoint, so it cannot share a chunk-grid, layout or adjoint-algebra fault with
the op. The reference itself is validated against central differences.

Every gradient is gated separately with ``rel_max`` (normalised by the reference's own
magnitude, so an all-zero gradient scores 1.0). The fp32 gate is only meaningful with TF32
off (``ieee_fp32`` fixture): fp32-bank matmuls follow ``torch.backends.cuda.matmul.allow_tf32``.

Gate calibration (fp32 IEEE, every shape and cotangent below): largest deviation from the
reference 2.2e-07; repeated runs are bitwise equal (the kernels are deterministic).
GATE_FP32 = 1e-5 is 45x that deviation, and 1000x below the 1e-2 faults injected in the
qualification tests. bf16 deviations at C=64/128 are 3e-3..6e-3; bf16 is only gated at a
gross-structure bar (GATE_BF16), and the discriminator test checks that the bf16 deviation
shrinks by more than 100x in fp32, i.e. that it is precision-shaped.
"""
import inspect
import math

import pytest
import torch

from sdm_testlib import (
    GATE_FP32,
    OP_GRADS,
    OP_INPUTS,
    kernel_write_read,
    make_op_inputs,
    assert_runs_bitwise,
    oracle_write_read,
    rel_max,
    requires_cuda,
)

GATE_BF16 = 1e-1
FAULT_EPS = 1e-2

SPH, DIM = 512, 64

# (label, P, T, W, R, C, layout)
SHAPES = [
    ("P2_T256_C16", 2, 256, 4, 4, 16, "3d"),
    ("P2_T256_C32", 2, 256, 4, 4, 32, "3d"),
    ("P2_T256_C64", 2, 256, 4, 4, 64, "3d"),
    ("P2_T256_C128", 2, 256, 4, 4, 128, "3d"),
    ("P2_T256_C256", 2, 256, 4, 4, 256, "3d"),
    ("P4_T128_C64", 4, 128, 4, 4, 64, "3d"),
    ("P2_T128_W8R4_C64", 2, 128, 8, 4, 64, "3d"),   # W != R
    # A single stream (B * H == 1) takes the P == 1 branches of the forward and backward.
    ("P1_T256_C16", 1, 256, 4, 4, 16, "3d"),
    ("P1_T256_C64", 1, 256, 4, 4, 64, "3d"),
    ("P1_T256_C128", 1, 256, 4, 4, 128, "3d"),
    # Two slot ranges walked as one sequential stream (num_parallel=1, 2-D inputs).
    ("BH2_2d_T128_C64", 2, 128, 4, 4, 64, "2d"),
    # Ragged last chunks, a chunk size that is not a power of two.
    ("P2_T200_C48", 2, 200, 4, 4, 48, "3d"),
    ("BH2_2d_T77_C32", 2, 77, 4, 4, 32, "2d"),
]
SHAPE_IDS = [s[0] for s in SHAPES]
PROD_SHAPES = [s for s in SHAPES if s[0] in ("P2_T256_C64", "P2_T256_C128")]
# The widest write and read sets, over a 4096-slot bank: (label, P, T, W, R, C, layout, sph).
WIDE_SHAPES = [
    ("P2_T256_W256_C128", 2, 256, 256, 64, 128, "3d", 4096),
    ("P2_T96_W64_R256_C32", 2, 96, 64, 256, 32, "3d", 4096),
]
FP32_SHAPES = [s + (SPH,) for s in SHAPES] + WIDE_SHAPES
COTANGENTS = ("dense", "dense2", "expanded", "noncontig")


def _run_pair(P, T, W, R, C, layout, dtype, cotangent="dense", seed=5, kwd=False, sph=SPH):
    if dtype is torch.float32:
        assert torch.backends.cuda.matmul.allow_tf32 is False, "request the ieee_fp32 fixture"
    d = make_op_inputs(P, T, W, R, DIM, sph, dtype, seed=seed)
    if kwd:
        # Per-write log-decay g * |k_val| (the key_weighted_decay form), [P, T, W].
        d["g"] = (d["g"].float() * d["k_val"].float().abs()).to(dtype)
    readings_k, kg, G = kernel_write_read(d, chunk_size=C, sph=sph, cotangent=cotangent,
                                          layout=layout)
    readings_o, og, _ = oracle_write_read(d, G.reshape(P, T, -1), sph=sph)
    return d, readings_k, readings_o, kg, og


def _devs(og, kg):
    return {n: rel_max(og[n], kg[n]) for n in OP_GRADS}


def _offenders(devs, gate):
    return {n: v for n, v in devs.items() if not v < gate}


def _fmt(devs):
    return "  ".join(f"{n.removeprefix('grad_')}={v:.2e}" for n, v in devs.items())


# ================================================================ the reference itself
def _cpu_fixture():
    d = make_op_inputs(2, 24, 4, 4, 16, 64, torch.float64, seed=11, device="cpu")
    G = torch.randn(2, 24, 16, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    return d, G


def test_oracle_is_not_vacuous():
    d, G = _cpu_fixture()
    readings, og, mem_end = oracle_write_read(d, G, sph=64)
    assert torch.isfinite(readings).all() and readings.abs().max() > 1e-6
    assert not torch.allclose(mem_end, d["memory"])
    for n in OP_GRADS:
        assert torch.isfinite(og[n]).all() and og[n].abs().max() > 1e-9, n
    # The swap faults below are only detectable if the swapped gradients differ.
    assert rel_max(og["grad_beta"], og["grad_g"]) > 0.1
    assert rel_max(og["grad_k_val"], og["grad_q_val"]) > 0.1


@pytest.mark.parametrize("name,index", [
    ("beta", (0, 3, 0)), ("g", (1, 5, 0)), ("k_val", (1, 7, 2)),
    ("v", (0, 9, 4)), ("q_val", (1, 11, 1)), ("memory", (17, 5)),
])
def test_oracle_gradients_match_central_differences(name, index):
    d, G = _cpu_fixture()
    _, og, _ = oracle_write_read(d, G, sph=64)
    # Pick a coordinate that the loss depends on (a slot that is actually read).
    if name == "memory":
        index = (int(d["k_idx"][0, 2, 1]), 5)

    def loss(dd):
        r, _, _ = oracle_write_read(dd, None)
        return (r * G).sum().item()

    eps = 1e-6
    dd = {k: v.clone() for k, v in d.items()}
    dd[name][index] += eps
    lp = loss(dd)
    dd[name][index] -= 2 * eps
    lm = loss(dd)
    fd = (lp - lm) / (2 * eps)
    an = og["grad_" + name][index].item()
    assert abs(fd) > 1e-6, f"coordinate {index} of {name} has ~zero gradient"
    assert math.isclose(fd, an, rel_tol=1e-5), f"grad_{name}{index}: autograd {an} vs fd {fd}"


@requires_cuda
def test_oracle_is_device_independent():
    d = make_op_inputs(2, 64, 4, 4, 32, 128, torch.float64, seed=3)
    G = torch.randn(2, 64, 32, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    _, g_cuda, _ = oracle_write_read(d, G.cuda(), sph=128)
    _, g_cpu, _ = oracle_write_read(d, G, device="cpu", sph=128)
    for n in OP_GRADS:
        assert rel_max(g_cpu[n], g_cuda[n].cpu()) < 1e-12, n


# ======================================================================== repeats
@requires_cuda
@pytest.mark.parametrize("label,P,T,W,R,C,layout", SHAPES, ids=SHAPE_IDS)
def test_fp32_repeats_are_bitwise_equal(label, P, T, W, R, C, layout, ieee_fp32):
    """Repeated runs on one input are bitwise equal: the gates below never absorb noise."""
    d = make_op_inputs(P, T, W, R, DIM, SPH, torch.float32, seed=5)
    runs = []
    for _ in range(5):
        r, g, _ = kernel_write_read(d, chunk_size=C, sph=SPH, layout=layout)
        g = dict(g)
        g["out"] = r
        runs.append(g)
    assert_runs_bitwise(runs, label=label)


# ========================================================================= the gates
@requires_cuda
@pytest.mark.parametrize("label,P,T,W,R,C,layout,sph", FP32_SHAPES,
                         ids=[s[0] for s in FP32_SHAPES])
def test_fp32_forward_and_every_gradient_match_the_oracle(label, P, T, W, R, C, layout, sph,
                                                          ieee_fp32):
    _, rk, ro, kg, og = _run_pair(P, T, W, R, C, layout, torch.float32, sph=sph)
    fwd = rel_max(ro, rk.reshape(ro.shape))
    devs = _devs(og, kg)
    assert fwd < GATE_FP32, f"{label}: forward {fwd:.3e}"
    bad = _offenders(devs, GATE_FP32)
    assert not bad, f"{label}: {_fmt(bad)}"


@requires_cuda
@pytest.mark.parametrize("cot", COTANGENTS)
@pytest.mark.parametrize("label,P,T,W,R,C,layout",
                         [s for s in SHAPES if s[0] in ("P2_T256_C64", "P1_T256_C64")],
                         ids=["P2_C64", "P1_C64"])
def test_every_cotangent_layout_matches_the_oracle(label, P, T, W, R, C, layout, cot,
                                                   ieee_fp32):
    _, _, _, kg, og = _run_pair(P, T, W, R, C, layout, torch.float32, cotangent=cot)
    bad = _offenders(_devs(og, kg), GATE_FP32)
    assert not bad, f"{label} cotangent={cot}: {_fmt(bad)}"


@requires_cuda
@pytest.mark.parametrize("C", [32, 64])
def test_fp32_per_write_gate_matches_the_oracle(C, ieee_fp32):
    _, rk, ro, kg, og = _run_pair(2, 128, 4, 4, C, "3d", torch.float32, kwd=True)
    assert rel_max(ro, rk) < GATE_FP32
    bad = _offenders(_devs(og, kg), GATE_FP32)
    assert not bad, f"per-write gate C={C}: {_fmt(bad)}"


@requires_cuda
@pytest.mark.parametrize("dim", [1024, 1536, 2048])
def test_fp32_wide_heads_match_the_oracle(dim, ieee_fp32):
    """Wide heads (single-head layers of width 1024 to 2048): the movers loop over column
    tiles and the gather splits rows into column slices."""
    d = make_op_inputs(2, 128, 8, 8, dim, 256, torch.float32, seed=1, weights="softmax")
    rk, kg, G = kernel_write_read(d, chunk_size=64, sph=256)
    ro, og, _ = oracle_write_read(d, G, sph=256)
    assert rel_max(ro, rk) < GATE_FP32
    bad = _offenders(_devs(og, kg), GATE_FP32)
    assert not bad, f"head_dim {dim}: {_fmt(bad)}"


@requires_cuda
@pytest.mark.parametrize("dim", [129, 961, 1025, 1024])
def test_fp32_multi_step_tail_heads_match_the_oracle(dim, ieee_fp32):
    """Regression (gx_gds_cuda's tail module takes the tail heads; its
    wide-head dots land in d_q_w when step 0 runs): heads of more than one 128-column slice, hd % 4 != 0 (129 / 961 /
    1025) and the even control 1024, over three chunks (T 165 at C 64: a ragged last chunk). The forward and all six
    gradients (grad_q_val and grad_g read the final d_q_w) are within GATE_FP32 of the float64 oracle; on the modern
    backend the backward's GDS is gx_gds with ceil(dim / 128) slices (no phase2_kernels GDS)."""
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import gx_gds_cuda as gxm
    from sparse_delta_memory.cuda._device import backend
    nsl = gxm.slices(dim)[0]
    assert nsl == -(-dim // 128) > 1
    d = make_op_inputs(2, 165, 8, 8, dim, 256, torch.float32, seed=1, weights="softmax")
    s0 = dict(movers.STATS)
    rk, kg, G = kernel_write_read(d, chunk_size=64, sph=256)
    if backend(d["memory"].device) == "modern":
        assert movers.STATS["da2_gds"] > s0["da2_gds"], movers.STATS
        assert movers.STATS["shipped_gds"] == s0["shipped_gds"], movers.STATS
        assert gxm.LAST_POLICY.get("impl") == "gx" and gxm.LAST_POLICY.get("nsl") == nsl, gxm.LAST_POLICY
    ro, og, _ = oracle_write_read(d, G, sph=256)
    assert rel_max(ro, rk) < GATE_FP32
    bad = _offenders(_devs(og, kg), GATE_FP32)
    assert not bad, f"head_dim {dim}: {_fmt(bad)}"


@requires_cuda
@pytest.mark.parametrize("label,P,T,W,R,C,layout", PROD_SHAPES, ids=[s[0] for s in PROD_SHAPES])
def test_bf16_gradients_match_the_oracle_at_a_gross_bar(label, P, T, W, R, C, layout):
    _, rk, ro, kg, og = _run_pair(P, T, W, R, C, layout, torch.bfloat16)
    assert rel_max(ro, rk) < GATE_BF16
    bad = _offenders(_devs(og, kg), GATE_BF16)
    assert not bad, f"{label} bf16: {_fmt(bad)}"


@requires_cuda
@pytest.mark.parametrize("label,P,T,W,R,C,layout", PROD_SHAPES, ids=[s[0] for s in PROD_SHAPES])
def test_deviation_shrinks_when_precision_is_raised(label, P, T, W, R, C, layout, ieee_fp32):
    """A deviation that does not shrink from 8 to 23 mantissa bits is not rounding."""
    _, _, _, kg32, og32 = _run_pair(P, T, W, R, C, layout, torch.float32)
    _, _, _, kg16, og16 = _run_pair(P, T, W, R, C, layout, torch.bfloat16)
    d32, d16 = _devs(og32, kg32), _devs(og16, kg16)
    for n in OP_GRADS:
        ratio = d16[n] / max(d32[n], 1e-300)
        assert ratio > 100.0, f"{n}: bf16 {d16[n]:.3e} vs fp32 {d32[n]:.3e} ({ratio:.1f}x)"


# ======================================================== qualification (non-vacuity)
def _perturb(grads, fault):
    g = dict(grads)
    if fault.startswith("scale_"):
        n = "grad_" + fault[len("scale_"):]
        g[n] = g[n] * (1.0 + FAULT_EPS)
    elif fault == "swap_beta_g":
        g["grad_beta"], g["grad_g"] = g["grad_g"], g["grad_beta"]
    elif fault == "swap_kval_qval":
        g["grad_k_val"], g["grad_q_val"] = g["grad_q_val"], g["grad_k_val"]
    elif fault == "sign_memory":
        g["grad_memory"] = -g["grad_memory"]
    else:
        raise AssertionError(fault)
    return g


FAULTS = ["scale_" + n for n in OP_INPUTS] + ["swap_beta_g", "swap_kval_qval", "sign_memory"]


@requires_cuda
def test_the_gate_fails_on_every_perturbed_oracle(ieee_fp32):
    """Perturbing the reference by FAULT_EPS (or swapping two same-shape gradients) must
    fail the gate, while the unperturbed comparison passes."""
    _, _, _, kg, og = _run_pair(2, 256, 4, 4, 64, "3d", torch.float32)
    assert not _offenders(_devs(og, kg), GATE_FP32), "cannot qualify against a failing baseline"
    for fault in FAULTS:
        caught = _offenders(_devs(_perturb(og, fault), kg), GATE_FP32)
        assert caught, f"fault {fault} was not detected"
        if fault.startswith("scale_"):
            assert list(caught) == ["grad_" + fault[len("scale_"):]], (fault, caught)


@requires_cuda
def test_the_gate_resolves_perturbations_near_its_own_bound(ieee_fp32):
    """Sensitivity: a 10x-gate relative perturbation of any single gradient is caught."""
    _, _, _, kg, og = _run_pair(2, 256, 4, 4, 64, "3d", torch.float32)
    for n in OP_GRADS:
        pert = dict(og)
        pert[n] = og[n] * (1.0 + 10 * GATE_FP32)
        assert n in _offenders(_devs(pert, kg), GATE_FP32), n


def _return_slots(F):
    """Position of each differentiable input in backward's return tuple, from the signature."""
    names = list(inspect.signature(F.forward).parameters)[1:]   # drop ctx
    return {n: names.index(n) for n in OP_INPUTS}


class _ReturnTupleFault:
    """Wrap the real backward of ``F`` and corrupt its return tuple; restore on exit."""

    def __init__(self, F, kind, a, b=None):
        self.F, self.kind, self.a, self.b, self.calls = F, kind, a, b, 0

    def __enter__(self):
        F = self.F
        self.cls, self.orig = F, F.__dict__["backward"]
        orig = self.orig.__func__

        def faulted(ctx, *grads):
            out = list(orig(ctx, *grads))
            self.calls += 1
            if self.kind == "swap":
                out[self.a], out[self.b] = out[self.b], out[self.a]
            elif self.kind == "scale":
                out[self.a] = out[self.a] * (1.0 + FAULT_EPS)
            else:
                out[self.a] = torch.zeros_like(out[self.a])
            return tuple(out)

        F.backward = staticmethod(faulted)
        return self

    def __exit__(self, *exc):
        self.cls.backward = self.orig
        return False


@requires_cuda
@pytest.mark.parametrize("kind,a,b", [
    ("swap", "beta", "g"), ("swap", "k_val", "q_val"), ("scale", "v", None),
    ("zero", "memory", None),
])
def test_a_fault_in_the_real_backward_return_tuple_is_detected(kind, a, b, ieee_fp32):
    from sparse_delta_memory.memory_ops import SDMWriteRead as F

    slots = _return_slots(F)
    d = make_op_inputs(2, 256, 4, 4, DIM, SPH, torch.float32, seed=5)
    with _ReturnTupleFault(F, kind, slots[a], slots[b] if b else None) as inj:
        _, kg_bad, G = kernel_write_read(d, chunk_size=64, sph=SPH)
    assert inj.calls > 0, "the faulted backward never ran"
    _, og, _ = oracle_write_read(d, G, sph=SPH)
    assert _offenders(_devs(og, kg_bad), GATE_FP32), f"{kind} {a} {b} not detected"
    _, kg_ok, G2 = kernel_write_read(d, chunk_size=64, sph=SPH)
    _, og2, _ = oracle_write_read(d, G2, sph=SPH)
    assert not _offenders(_devs(og2, kg_ok), GATE_FP32), "did not return to green"


@requires_cuda
def test_an_internal_backward_fault_is_detected(monkeypatch, ieee_fp32):
    """Perturb the gate gradient of the Phase-1 backward; the reference must see it. The
    CUDA backward (backward.py) calls the Phase-1 backward through
    cuda/phase1_bwd_cuda.phase1_backward_det (the same outputs as memory_ops.phase1_backward,
    which the reference backward calls), so both are perturbed."""
    from sparse_delta_memory import memory_ops
    from sparse_delta_memory.cuda import phase1_bwd_cuda

    targets = [(memory_ops, "phase1_backward"), (phase1_bwd_cuda, "phase1_backward_det")]
    slot = 2
    calls = []

    def perturb(orig):
        def perturbed(*a, **kw):
            out = list(orig(*a, **kw))
            calls.append(1)
            out[slot] = out[slot] * (1.0 + FAULT_EPS)
            return tuple(out)
        return perturbed

    d = make_op_inputs(2, 256, 4, 4, DIM, SPH, torch.float32, seed=5)
    with monkeypatch.context() as m:
        for module, name in targets:
            m.setattr(module, name, perturb(getattr(module, name)))
        _, kg_bad, G = kernel_write_read(d, chunk_size=64, sph=SPH)
    assert calls, f"the backward no longer calls {[name for _, name in targets]}"
    _, og, _ = oracle_write_read(d, G, sph=SPH)
    assert "grad_g" in _offenders(_devs(og, kg_bad), GATE_FP32)
    _, kg_ok, _ = kernel_write_read(d, chunk_size=64, sph=SPH, cotangent=G)
    assert not _offenders(_devs(og, kg_ok), GATE_FP32)


@requires_cuda
def test_a_skipped_group_decay_is_detected(monkeypatch, ieee_fp32):
    """Skip the decay of the rows written several times in a chunk: the forward mover gets a
    cumulative decay of 1 for every entry of a repeated slot, so each slot's one writer
    stores the undecayed snapshot plus its group's writes. The reference must see it in the
    readings. The CUDA DW mover takes each entry's decay from Phase 1's records (krec), so the fault
    also drops them: the mover then builds its records from the faulted cpe (movers.make_mover_dw
    with krec None). 2.0.0: no Triton DW to divert the call to. The fault reads Phase 1's is_only,
    which the native lean field sets omit (DET_FLAGS): the faulted call asks Phase 1 for the full set
    (forward._FULL_FIELDS) and keeps the native routes."""
    from sparse_delta_memory import forward, memory_ops

    orig = memory_ops.make_mover_dw
    calls = []

    def faulted(bank, snapshot, delta_v, slot, cpe, is_last, is_only, *rest, **kw):
        calls.append(int((is_only == 0).sum()))
        cpe = torch.where(is_only != 0, cpe, torch.ones_like(cpe))
        kw["krec"] = None   # the CUDA DW then builds its records from the faulted cpe
        return orig(bank, snapshot, delta_v, slot, cpe, is_last, is_only, *rest, **kw)

    d = make_op_inputs(2, 256, 4, 4, DIM, SPH, torch.float32, seed=5)
    with monkeypatch.context() as m:
        m.setattr(memory_ops, "make_mover_dw", faulted)
        m.setattr(forward, "phase1_fields", lambda *a, **k: forward._FULL_FIELDS)   # is_only (DET_FLAGS) for the fault
        r_bad, _, G = kernel_write_read(d, chunk_size=64, sph=SPH)
    assert calls and max(calls) > 0, "premise: no slot was written twice in a chunk"
    ro, _, _ = oracle_write_read(d, G, sph=SPH)
    assert rel_max(ro, r_bad) > GATE_FP32
    r_ok, _, _ = kernel_write_read(d, chunk_size=64, sph=SPH, cotangent=G)
    assert rel_max(ro, r_ok) < GATE_FP32


def test_the_fault_set_covers_every_gradient():
    for n in OP_INPUTS:
        assert "scale_" + n in FAULTS
    assert {"swap_beta_g", "swap_kval_qval"} <= set(FAULTS)


@requires_cuda
def test_a_missing_pair_in_the_phase1_gram_matrix_is_detected(monkeypatch, ieee_fp32):
    """Drop the write-write pairs of one chunk from the Phase-1 matrix A; the reference must
    see it in the readings."""
    from sparse_delta_memory import memory_ops

    orig = memory_ops.phase1_forward

    def faulted(*args, **kw):
        p1 = orig(*args, **kw)
        p1.a[1].zero_()
        return p1

    d = make_op_inputs(2, 256, 4, 4, DIM, SPH, torch.float32, seed=5, slots="clustered")
    with monkeypatch.context() as m:
        m.setattr(memory_ops, "phase1_forward", faulted)
        r_bad, _, G = kernel_write_read(d, chunk_size=64, sph=SPH)
    ro, _, _ = oracle_write_read(d, G, sph=SPH)
    assert rel_max(ro, r_bad) > GATE_FP32
    r_ok, _, _ = kernel_write_read(d, chunk_size=64, sph=SPH, cotangent=G)
    assert rel_max(ro, r_ok) < GATE_FP32
