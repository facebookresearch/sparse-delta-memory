# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The dense limit of SDM is the gated delta rule (GDN).

With one product-key factor of width d_k, every slot selected (num_writes = num_reads =
d_k) and l2-normalized key and query weights, each token writes and reads every row of its
head's bank, and the bank of a head is the d_k x d_v state of the gated delta rule:

    S_t = a_t S_{t-1} + beta_t k_t (v_t - a_t k_t^T S_{t-1})^T,    o_t = S_t^T q_t,
    a_t = exp(g_t),

where k_t and q_t are the key and query weights scattered back to slot order and o_t is the
reading of token t, taken after its write. Every test compares against ``gdn_reference``,
this recurrence written with float64 matrices and differentiated by autograd; it shares no
kernel, chunking, adjoint or token loop with the code under test.

Op level (``sdm_write_read``, its no-grad path, the CUDA-graph runner and the T = 1 decode
kernel): every token lists its d_k slots in the top-k order of random scores, a new
permutation per token, from a nonzero initial bank; the readings, the final bank and the
six input gradients are checked at chunk sizes 16 to 256 with a ragged last chunk.

Layer level: ``SparseDeltaMemory`` with ``product_key_factor_dims=[d_k]``, l2-normalized
reads and writes, no output gate, no readings LayerNorm and a zero initial bank, against
plain torch GDN on the same weights: q = l2norm(Wq x), k = l2norm(Wk x), v = Wv x,
g = -exp(A_log) * softplus(a_proj(x) + dt_bias), beta = sigmoid(b_proj(x)), output Wo(o).
Training (output and every gradient), the layer's CUDA graphs and decode.

Gates are the suite's. Op: fp32 banks with TF32 off within GATE_FP32 (rel_max) of the
reference; bf16 banks by ``bf16_gate_failures`` (the bf16 ceilings, and an fp32 bank on the
same bf16-valued inputs at least BF16_PRECISION_RATIO times closer), with op_grid's
collision ceilings for the final bank, since every token writes every slot; the decode
kernel within test_decode's 1e-6. Layer: test_layer_reference's relative-RMS gates in fp32,
the bf16 gate in bf16. If ``fla`` is installed, its ``chunk_gated_delta_rule`` (scale 1) is
compared too; the package does not depend on it.
"""
import pytest
import torch
from torch.nn import functional as F

from sdm_testlib import (
    BF16_BANK_COLLISION_CEILINGS,
    BF16_CEILINGS,
    BF16_COLLISION_WRITES,
    OP_GRADS,
    OP_INPUTS,
    TINY_NUMEL,
    assert_runs_bitwise,
    bf16_gate_failures,
    bits_equal,
    deviations,
    fp32_gate_failures,
    layer_fwd_bwd,
    make_layer,
    rel_max,
    rel_rms,
    requires_cuda,
)

pytestmark = requires_cuda

DTYPES = [torch.float32, torch.bfloat16]
DTYPE_IDS = ["fp32", "bf16"]
TENSORS = ("readings", "final_memory") + OP_GRADS
DECODE_GATE_FP32 = 1e-6        # test_decode.GATE: rel_max of the deterministic decode kernel
LAYER_GATE_REL_RMS = 1e-5      # test_layer_reference: fp32 layer against float64
LAYER_GATE_REL_RMS_TINY = 1e-3  # ... per-head gate parameters (sums with heavy cancellation)
# bf16 layer: the per-head gate-parameter gradients (A_log, dt_bias, a_proj / b_proj biases,
# at most TINY_NUMEL elements) are sums over B * T tokens with heavy cancellation that
# autograd forms in bf16 (a run-to-run spread of several percent): up to about 6e-2 rel_rms
# against float64 (the other tensors at most 8e-3; the fp32 layer 1.4e-5). They keep the
# precision-ratio check, under these ceilings.
LAYER_BF16_TINY_CEILINGS = (2e-1, 2e-1)


def l2n(x):
    """The layer's ``l2_normalize``: x / (||x||_2 + 1e-12) over the last dim."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def gdn_reference(S0, k, v, beta, g, q):
    """The gated delta rule with float64 matrices; autograd gives the gradients.

    S0 [N, dk, dv]; k, q [N, T, dk]; v [N, T, dv]; beta, g [N, T] (g: log-decay).
    Returns (o [N, T, dv], S_T).
    """
    S, outs = S0, []
    for t in range(k.shape[1]):
        a = torch.exp(g[:, t]).view(-1, 1, 1)
        kt = k[:, t].unsqueeze(-1)                                    # [N, dk, 1]
        decayed = a * S                                               # a_t S_{t-1}
        error = v[:, t].unsqueeze(1) - kt.transpose(1, 2) @ decayed   # v_t - a_t k_t^T S_{t-1}
        S = decayed + beta[:, t].view(-1, 1, 1) * (kt @ error)        # S_t
        outs.append((q[:, t].unsqueeze(1) @ S).squeeze(1))            # o_t = S_t^T q_t
    return torch.stack(outs, 1), S


# Broken recurrences the gates must reject (``test_the_gates_reject_every_broken_recurrence``).
FAULTS = ("read_before_write", "undecayed_retrieval", "beta_on_v_only")


def broken_gdn(S0, k, v, beta, g, q, fault):
    """``gdn_reference`` with one term broken:
    read_before_write     o_t = S_{t-1}^T q_t
    undecayed_retrieval   the error uses k_t^T S_{t-1} instead of a_t k_t^T S_{t-1}
    beta_on_v_only        S_t = a_t S_{t-1} + k_t (beta_t v_t - a_t k_t^T S_{t-1})^T
    """
    assert fault in FAULTS, fault
    S, outs = S0, []
    for t in range(k.shape[1]):
        a, b = torch.exp(g[:, t]).view(-1, 1, 1), beta[:, t].view(-1, 1, 1)
        kt, vt = k[:, t].unsqueeze(-1), v[:, t].unsqueeze(1)
        decayed = a * S
        retrieved = kt.transpose(1, 2) @ (S if fault == "undecayed_retrieval" else decayed)
        if fault == "beta_on_v_only":
            new = decayed + kt @ (b * vt - retrieved)
        else:
            new = decayed + b * (kt @ (vt - retrieved))
        outs.append((q[:, t].unsqueeze(1) @ (S if fault == "read_before_write" else new))
                    .squeeze(1))
        S = new
    return torch.stack(outs, 1), S


# ================================================================================ op level
def dense_problem(P, T, dk, dv, dtype, seed=0, device="cuda", gates=(-0.7, -0.02),
                  betas=(0.05, 0.95), spread=None):
    """Inputs of ``sdm_write_read`` for P streams of T tokens, each token writing and reading
    all dk slots of its stream in the top-k order of random scores (a new permutation per
    token), with l2-normalized weights, log-decays g ~ U(gates), beta ~ U(betas), a nonzero
    initial bank and a cotangent. Keys and queries are independent random directions, or
    with ``spread`` the directions of one shared random vector per stream plus ``spread``
    times noise (correlated keys: the retrieval k^T S is then a large part of each write).
    Values are rounded to ``dtype`` and kept in float32, so the reference sees what the op
    gets."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    if spread is None:
        k, q = l2n(randn(P, T, dk)), l2n(randn(P, T, dk))
    else:
        center = randn(P, 1, dk)
        k, q = l2n(center + spread * randn(P, T, dk)), l2n(center + spread * randn(P, T, dk))
    k_order = randn(P, T, dk).argsort(-1, descending=True)
    q_order = randn(P, T, dk).argsort(-1, descending=True)
    offset = (torch.arange(P) * dk).view(P, 1, 1)
    d = dict(memory=randn(P * dk, dv) * 0.5, k_idx=k_order + offset,
             k_val=k.gather(-1, k_order), v=randn(P, T, dv),
             beta=torch.empty(P, T, 1).uniform_(*betas, generator=gen),
             g=torch.empty(P, T, 1).uniform_(*gates, generator=gen),
             q_idx=q_order + offset, q_val=q.gather(-1, q_order),
             cotangent=randn(P, T, dv))
    return {n: (t.to(dtype).float() if t.is_floating_point() else t).to(device)
            for n, t in d.items()}


def dense_gdn_inputs(d, leaves):
    """The GDN state and inputs of ``d``: the slot lists scattered back to slot order, the
    bank rows of stream p as its state. ``leaves`` holds the float tensors of OP_INPUTS."""
    P, T, dk = d["k_idx"].shape
    offset = (torch.arange(P, device=d["k_idx"].device) * dk).view(P, 1, 1)
    zeros = torch.zeros(P, T, dk, dtype=leaves["k_val"].dtype, device=d["k_idx"].device)
    k = zeros.scatter(-1, d["k_idx"] - offset, leaves["k_val"])
    q = zeros.scatter(-1, d["q_idx"] - offset, leaves["q_val"])
    S0 = leaves["memory"].view(P, dk, -1)
    return S0, k, leaves["v"], leaves["beta"].squeeze(-1), leaves["g"].squeeze(-1), q


def reference(d, fault=None):
    """``gdn_reference`` on the op's inputs: {readings [P, T, dv], final_memory, grads}
    (``broken_gdn`` with ``fault``)."""
    leaves = {n: d[n].detach().double().clone().requires_grad_(True) for n in OP_INPUTS}
    if fault is None:
        o, S = gdn_reference(*dense_gdn_inputs(d, leaves))
    else:
        o, S = broken_gdn(*dense_gdn_inputs(d, leaves), fault)
    grads = torch.autograd.grad(o, [leaves[n] for n in OP_INPUTS],
                                grad_outputs=d["cotangent"].double())
    return dict(zip(OP_GRADS, grads), readings=o.detach(),
                final_memory=S.detach().reshape(-1, S.shape[-1]))


def op_training(d, chunk_size, dtype):
    """``sdm_write_read`` forward + backward in the bank dtype: (results as float32, the bank
    after the backward, which must be the initial bank again)."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T, dk = d["k_idx"].shape
    leaves = {n: d[n].to(dtype, copy=True).requires_grad_(True) for n in OP_INPUTS}
    readings = sdm_write_read(
        leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"],
        leaves["g"], d["q_idx"], leaves["q_val"], chunk_size=chunk_size,
        num_memory_slots=dk, num_parallel=P, key_weighted_decay=False)
    final = leaves["memory"].detach().clone()   # advanced in place; the backward restores it
    grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS],
                                grad_outputs=d["cotangent"].to(dtype).reshape(readings.shape))
    out = dict(zip(OP_GRADS, grads), readings=readings.detach().reshape(P, T, -1),
               final_memory=final)
    return {k: v.float() for k, v in out.items()}, leaves["memory"].detach()


def op_failures(ref, got, got_fp32, dtype, writes_per_slot, names=TENSORS):
    """The suite's op gates (op_grid.run_case): GATE_FP32 for fp32 banks; for bf16 banks
    ``bf16_gate_failures`` against ``got_fp32`` (an fp32 bank on the same values), with the
    collision ceilings for the final bank above BF16_COLLISION_WRITES writes per slot."""
    dev = deviations({k: ref[k] for k in names}, got)
    if dtype == torch.float32:
        return fp32_gate_failures(dev), dev
    ceilings = ({"final_memory": BF16_BANK_COLLISION_CEILINGS}
                if writes_per_slot > BF16_COLLISION_WRITES and "final_memory" in names
                else None)
    return bf16_gate_failures(dev, deviations({k: ref[k] for k in names}, got_fp32),
                              ceilings=ceilings), dev


def _report(label, dev):
    worst = max(dev, key=lambda k: dev[k]["rel_max"])
    print(f"DENSE_GDN {label} worst={worst} rel_max={dev[worst]['rel_max']:.3e} "
          f"max_rel_rms={max(v['rel_rms'] for v in dev.values()):.3e}")


def test_reference_is_the_sdm_token_loop_in_the_dense_limit():
    """In float64 the GDN matrices and the suite's SDM token loop (``oracle_recurrence``) on
    dense permuted slot lists agree to rounding: readings, final bank and gradients."""
    from sdm_testlib import oracle_recurrence

    d = dense_problem(2, 40, 16, 8, torch.float32, seed=1)
    ref = reference(d)
    leaves = {n: d[n].detach().double().clone().requires_grad_(True) for n in OP_INPUTS}
    o, mem = oracle_recurrence(leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"],
                               leaves["beta"], leaves["g"], d["q_idx"], leaves["q_val"])
    grads = torch.autograd.grad(o, [leaves[n] for n in OP_INPUTS],
                                grad_outputs=d["cotangent"].double())
    got = dict(zip(OP_GRADS, grads), readings=o, final_memory=mem)
    bad = {k: v["rel_max"] for k, v in deviations(ref, got).items() if not v["rel_max"] < 1e-12}
    assert not bad, bad


# (dk, dv, P, T, chunk size, key spread): T is not a multiple of any chunk size.
OP_CASES = [(64, 64, 2, 300, C, None) for C in (16, 64, 128, 256)] + [
    (64, 64, 2, 300, 16, 0.3),    # correlated keys: the delta-rule retrieval dominates
    (64, 64, 2, 300, 256, 0.3),
    (16, 32, 3, 77, 16, None),    # narrow bank, three streams
    (256, 128, 2, 150, 64, None),  # the widest selection, W = R = 256
]
OP_IDS = [f"dk{dk}_dv{dv}_P{P}_T{T}_C{C}" + ("" if s is None else f"_spread{s}")
          for dk, dv, P, T, C, s in OP_CASES]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("dk,dv,P,T,C,spread", OP_CASES, ids=OP_IDS)
def test_op_is_the_gated_delta_rule(dk, dv, P, T, C, spread, dtype, ieee_fp32):
    d = dense_problem(P, T, dk, dv, dtype, seed=dk + C, spread=spread)
    assert not bool((d["k_idx"].diff(dim=-1) > 0).all()), "slot lists should be permuted"
    ref = reference(d)
    got, restored = op_training(d, C, dtype)
    got32 = op_training(d, C, torch.float32)[0] if dtype == torch.bfloat16 else None
    bad, dev = op_failures(ref, got, got32, dtype, writes_per_slot=min(C, T))
    _report(f"op {dtype} dk{dk} C{C} spread{spread}", dev)
    assert bits_equal(restored, d["memory"].to(dtype)), "the backward did not restore the bank"
    assert not bad, f"dk={dk} C={C} {dtype}: {bad}"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_the_gates_reject_every_broken_recurrence(dtype, ieee_fp32):
    """Control first: the op passes against ``gdn_reference``. Then against each
    ``broken_gdn`` (read before the write, retrieval from the undecayed state, beta on v
    only) the same results fail the same gate, by a margin. Correlated keys make the
    retrieval k^T S a large part of each write, so a broken retrieval moves every tensor:
    measured, the worst tensor by 0.9 relative RMS or more and the readings by 0.18 or more
    (random keys in 64 dimensions are nearly orthogonal and give 0.09 and 0.04)."""
    P, T, dk, dv, C = 2, 300, 64, 64, 64
    d = dense_problem(P, T, dk, dv, dtype, seed=13, spread=0.3)
    got = op_training(d, C, dtype)[0]
    got32 = op_training(d, C, torch.float32)[0] if dtype == torch.bfloat16 else None
    bad, _ = op_failures(reference(d), got, got32, dtype, C)
    assert not bad, f"control: {bad}"
    for fault in FAULTS:
        bad, dev = op_failures(reference(d, fault), got, got32, dtype, C)
        _report(f"broken {fault} {dtype}", dev)
        assert bad, f"{fault}: the gate accepts a broken recurrence"
        assert max(v["rel_rms"] for v in dev.values()) > 5 * BF16_CEILINGS[0], (fault, dev)
        assert dev["readings"]["rel_rms"] > 2 * BF16_CEILINGS[0], (fault, dev["readings"])


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_op_forward_without_grad_is_the_gated_delta_rule(dtype, ieee_fp32):
    """The no-grad forward (prefill: slabs of steps, nothing saved)."""
    from sparse_delta_memory.memory_ops import sdm_write_read

    P, T, dk, dv, C = 2, 300, 64, 64, 64
    d = dense_problem(P, T, dk, dv, dtype, seed=5)
    ref = reference(d)

    def run(dt):
        memory = d["memory"].to(dt, copy=True)
        with torch.no_grad():
            out = sdm_write_read(memory, d["k_idx"], d["k_val"].to(dt), d["v"].to(dt),
                                 d["beta"].to(dt), d["g"].to(dt), d["q_idx"],
                                 d["q_val"].to(dt), chunk_size=C, num_memory_slots=dk,
                                 num_parallel=P)
        return {"readings": out.float().view(P, T, dv), "final_memory": memory.float()}

    names = ("readings", "final_memory")
    bad, dev = op_failures(ref, run(dtype), run(torch.float32), dtype, C, names)
    _report(f"op_nograd {dtype}", dev)
    assert not bad, bad


def op_graph_call(runner, d, chunk_size, dtype):
    """``op_training`` through the layer's CUDA-graph runner (``CudaGraphRunner``)."""
    P, T, dk = d["k_idx"].shape
    leaves = {n: d[n].to(dtype, copy=True).requires_grad_(True) for n in OP_INPUTS}
    inputs = (d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"],
              d["q_idx"], leaves["q_val"])
    # The layer's scalars: chunk size, slots per stream, streams, snapshot_quant, grad_bank_fp32
    # (the defaults of sdm_write_read, as in op_training).
    readings, bank = runner(leaves["memory"], tuple(d["memory"].shape), dtype, inputs,
                            (chunk_size, dk, P, "none", False))
    final = bank.detach().clone()   # the static bank, until the backward rewinds it
    grads = torch.autograd.grad(readings, [leaves[n] for n in OP_INPUTS],
                                grad_outputs=d["cotangent"].to(dtype).reshape(readings.shape))
    out = dict(zip(OP_GRADS, grads), readings=readings.detach().reshape(P, T, -1),
               final_memory=final)
    return {k: v.float() for k, v in out.items()}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_op_graph_replay_is_the_gated_delta_rule(dtype, ieee_fp32):
    """The layer's CUDA-graph runner on the op, two calls per input (the first call of the
    first input captures, every call replays), on two inputs: each call passes the reference
    gate and equals eager calls of the same dtype bit for bit (the kernels are deterministic)."""
    from sparse_delta_memory.cuda_graph import CudaGraphRunner, reset_cuda_graphs

    P, T, dk, dv, C = 2, 300, 64, 64, 128
    reset_cuda_graphs()
    runner = CudaGraphRunner("dense_gdn")
    try:
        for seed in (7, 8):
            d = dense_problem(P, T, dk, dv, dtype, seed=seed)
            ref = reference(d)
            got32 = op_training(d, C, torch.float32)[0] if dtype == torch.bfloat16 else None
            graphed = [op_graph_call(runner, d, C, dtype) for _ in range(2)]
            for i, got in enumerate(graphed):
                bad, dev = op_failures(ref, got, got32, dtype, writes_per_slot=C)
                _report(f"op_graph {dtype} seed{seed} call{i}", dev)
                assert not bad, f"seed={seed} call {i} {dtype}: {bad}"
            assert_runs_bitwise(graphed + [op_training(d, C, dtype)[0] for _ in range(2)],
                                label=f"graph vs eager, seed={seed} {dtype}")
        stats = dict(runner.stats)
        assert stats.get("forward_capture") == 1 and stats.get("forward_replay") == 4, stats
        assert stats.get("backward_replay", 0) >= 3, stats
        assert not any(k.startswith("eager") for k in stats), stats
    finally:
        reset_cuda_graphs()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_decode_steps_are_the_gated_delta_rule(dtype):
    """T single-token steps of ``sdm_decode_step`` from the initial bank."""
    from sparse_delta_memory.decode import sdm_decode_step

    P, T, dk, dv = 2, 96, 64, 64
    d = dense_problem(P, T, dk, dv, dtype, seed=9)
    ref = reference(d)
    names = ("readings", "final_memory")

    def decode(dt):
        memory = d["memory"].to(dt, copy=True)
        outs = [sdm_decode_step(memory, *(d[n][:, t:t + 1].to(dt) if d[n].is_floating_point()
                                          else d[n][:, t:t + 1]
                                          for n in ("k_idx", "k_val", "v", "beta", "g",
                                                    "q_idx", "q_val")))
                for t in range(T)]
        return {"readings": torch.cat(outs, 1).float(), "final_memory": memory.float()}

    got = decode(dtype)
    dev = deviations({k: ref[k] for k in names}, got)
    _report(f"decode {dtype}", dev)
    if dtype == torch.float32:
        bad = {k: v["rel_max"] for k, v in dev.items() if not v["rel_max"] <= DECODE_GATE_FP32}
    else:
        bad = bf16_gate_failures(dev, deviations({k: ref[k] for k in names},
                                                 decode(torch.float32)))
    assert not bad, f"{dtype}: {bad}"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_fla_chunk_gated_delta_rule_agrees(dtype, ieee_fp32):
    """fla's GDN kernel (if installed) on the same dense problem: it matches the reference and
    the SDM op within the bf16 ceilings (rel_rms and rel_max), fp32 inputs included: fla
    computes its chunks with TF32 or bf16 matmuls, so no precision-ratio check applies."""
    fla_gdr = pytest.importorskip("fla.ops.gated_delta_rule")
    P, T, dk, dv, C = 2, 300, 64, 64, 64
    d = dense_problem(P, T, dk, dv, dtype, seed=11)
    ref = reference(d)
    S0, k, v, beta, g, q = dense_gdn_inputs(d, {n: d[n] for n in OP_INPUTS})
    cast = lambda t: t.to(dtype).unsqueeze(2).contiguous()  # noqa: E731  [B=P, T, H=1, *]
    o, S = fla_gdr.chunk_gated_delta_rule(
        cast(q), cast(k), cast(v), g.unsqueeze(2).float().contiguous(), cast(beta),
        scale=1.0, initial_state=S0.unsqueeze(1).float().contiguous(),
        output_final_state=True)
    sdm = op_training(d, C, dtype)[0]
    fla = {"readings": o.squeeze(2).float(), "final_memory": S.squeeze(1).reshape(P * dk, dv)}
    bad = {}
    for name in ("readings", "final_memory"):
        for label, a, b in (("fla_vs_reference", ref[name], fla[name]),
                            ("fla_vs_sdm", sdm[name], fla[name])):
            err = (rel_rms(a, b), rel_max(a, b))
            print(f"DENSE_GDN fla {dtype} {label} {name} rel_rms={err[0]:.3e} "
                  f"rel_max={err[1]:.3e}")
            if not (err[0] <= BF16_CEILINGS[0] and err[1] <= BF16_CEILINGS[1]):
                bad[f"{label}.{name}"] = err
    assert not bad, bad


# ============================================================================= layer level
DK, LAYER_T = 64, 150
LAYER_CFG = dict(dim=256, num_heads=2, slots_per_head=DK, product_key_factor_dims=[DK],
                 num_writes=DK, num_reads=DK, read_act="l2_normalize",
                 write_act="l2_normalize", output_gate=False, normalize_readings=False,
                 backprop_on_memory=False, query_batchnorm=False, chunk_size=64)


def dense_layer(dtype, **overrides):
    """The dense-limit layer. Its gates are set so that the decays exp(g) spread over about
    0.05..0.95 and every term of the recurrence matters."""
    layer = make_layer(seed=0, dtype=torch.float32, **{**LAYER_CFG, **overrides})
    with torch.no_grad():
        layer.A_log.copy_(torch.tensor([1.0, 3.0]).log())
        layer.dt_bias.fill_(-1.05)   # softplus(-1.05) = 0.3
    return layer.to(dtype)


def gdn_layer_reference(layer, x, normalize=True):
    """Plain torch GDN in float64 on the layer's weights: (output, input leaf, param leaves).
    ``normalize=False`` drops the l2 normalize of q and k (a broken reference)."""
    P = {n: p.detach().double().clone().requires_grad_(True)
         for n, p in layer.named_parameters()}
    xx = x.detach().double().clone().requires_grad_(True)
    B, T, _ = xx.shape
    H, dk, hd = layer.num_heads, layer.slots_per_head, layer.head_dim

    def lin(name, inp):
        return F.linear(inp, P[name + ".weight"], P[name + ".bias"])

    def heads(t):   # [B, T, H * d] -> [B * H, T, d]
        return t.view(B, T, H, -1).transpose(1, 2).reshape(B * H, T, -1)

    q, k = heads(lin("Wq_read", xx)), heads(lin("Wk_write", xx))
    if normalize:
        q, k = l2n(q), l2n(k)
    v = heads(lin("Wv_write", xx))
    g = -torch.exp(P["A_log"]) * F.softplus(lin("a_proj", xx) + P["dt_bias"])   # [B, T, H]
    beta = torch.sigmoid(lin("b_proj", xx))
    g, beta = (t.transpose(1, 2).reshape(B * H, T) for t in (g, beta))
    S0 = torch.zeros(B * H, dk, hd, dtype=torch.float64, device=x.device)
    o, _ = gdn_reference(S0, k, v, beta, g, q)
    o = o.view(B, H, T, hd).transpose(1, 2).reshape(B, T, H * hd)
    return lin("Wo", o), xx, P


def gdn_layer_fwd_bwd(layer, x, gy, normalize=True):
    out, xx, P = gdn_layer_reference(layer, x, normalize)
    out.backward(gy.double())
    res = {"out": out.detach(), "grad_x": xx.grad}
    res.update({"grad." + n: p.grad for n, p in P.items()})
    return res


def layer_inputs(dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(2, LAYER_T, LAYER_CFG["dim"], device="cuda", generator=gen)
    gy = torch.randn(2, LAYER_T, LAYER_CFG["dim"], device="cuda", generator=gen) * 0.01
    return x.to(dtype), gy.to(dtype)


def layer_failures(ref, got, got_fp32, dtype):
    """fp32: test_layer_reference's relative-RMS gates. bf16: ``bf16_gate_failures`` (the
    ceilings, LAYER_BF16_TINY_CEILINGS for the per-head gate parameters, and the fp32 layer
    on the same bf16-valued weights and inputs at least BF16_PRECISION_RATIO times closer)."""
    dev = deviations(ref, got)
    if dtype == torch.float32:
        return {k: f"rel_rms {v['rel_rms']:.3g}" for k, v in dev.items()
                if not v["rel_rms"] < (LAYER_GATE_REL_RMS if ref[k].numel() > TINY_NUMEL
                                       else LAYER_GATE_REL_RMS_TINY)}, dev
    ceilings = {k: LAYER_BF16_TINY_CEILINGS for k in ref if ref[k].numel() <= TINY_NUMEL}
    return bf16_gate_failures(dev, deviations(ref, got_fp32), ceilings=ceilings), dev


@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_layer_is_the_gated_delta_rule(dtype, cuda_graph, ieee_fp32):
    """Output and every gradient of the dense-limit layer against plain torch GDN. With
    cuda_graph, two steps on new inputs (the first captures, both replay), each also equal bit
    for bit to two eager steps of the same dtype (the kernels are deterministic)."""
    from sparse_delta_memory.cuda_graph import reset_cuda_graphs

    reset_cuda_graphs()
    layer = dense_layer(dtype, cuda_graph=cuda_graph)
    lay32 = dense_layer(torch.bfloat16).float() if dtype == torch.bfloat16 else None
    eager = dense_layer(dtype) if cuda_graph else None
    try:
        for seed in (21, 22) if cuda_graph else (21,):
            x, gy = layer_inputs(dtype, seed)
            got = layer_fwd_bwd(layer, x, gy)
            ref = gdn_layer_fwd_bwd(layer, x, gy)
            got32 = None if lay32 is None else layer_fwd_bwd(lay32, x.float(), gy.float())
            bad, dev = layer_failures(ref, got, got32, dtype)
            _report(f"layer {dtype} graph={cuda_graph} seed{seed}", dev)
            assert not bad, f"{dtype} cuda_graph={cuda_graph} seed={seed}: {bad}"
            if cuda_graph:
                assert_runs_bitwise([got] + [layer_fwd_bwd(eager, x, gy) for _ in range(2)],
                                    label=f"graph vs eager, seed={seed}")
        if cuda_graph:
            stats = dict(layer.graph_runner.stats)
            assert stats.get("forward_capture") == 1 and stats.get("forward_replay") == 2, stats
            assert stats.get("backward_replay", 0) >= 1, stats
            assert not any(k.startswith("eager") for k in stats), stats
    finally:
        reset_cuda_graphs()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_the_layer_gate_rejects_a_reference_without_l2_normalize(dtype, ieee_fp32):
    """Control first: the layer passes against ``gdn_layer_reference``; against the same
    reference without the l2 normalize of q and k it fails the same gate."""
    layer = dense_layer(dtype)
    x, gy = layer_inputs(dtype, seed=24)
    got = layer_fwd_bwd(layer, x, gy)
    got32 = None
    if dtype == torch.bfloat16:
        got32 = layer_fwd_bwd(dense_layer(torch.bfloat16).float(), x.float(), gy.float())
    bad, _ = layer_failures(gdn_layer_fwd_bwd(layer, x, gy), got, got32, dtype)
    assert not bad, f"control: {bad}"
    bad, dev = layer_failures(gdn_layer_fwd_bwd(layer, x, gy, normalize=False), got, got32,
                              dtype)
    _report(f"broken no_l2_normalize {dtype}", dev)
    assert bad and dev["out"]["rel_rms"] > 3 * BF16_CEILINGS[0], dev["out"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_layer_decode_is_the_gated_delta_rule(dtype, ieee_fp32):
    """Token-by-token decode from the layer's (zero) cache against the GDN output."""
    layer = dense_layer(dtype).eval()
    x, _ = layer_inputs(dtype, seed=23)

    def decode(lay, xs, dt):
        cache = lay.create_kv_cache(xs.shape[0], 0, dt, "cuda")
        with torch.no_grad():
            return torch.cat([lay(xs[:, t:t + 1], cache=cache)[0]
                              for t in range(xs.shape[1])], 1).float()

    ref = {"out": gdn_layer_reference(layer, x)[0].detach()}
    dev = deviations(ref, {"out": decode(layer, x, dtype)})
    _report(f"layer_decode {dtype}", dev)
    if dtype == torch.float32:
        bad = {k: v["rel_rms"] for k, v in dev.items() if not v["rel_rms"] < LAYER_GATE_REL_RMS}
    else:
        lay32 = dense_layer(torch.bfloat16).float().eval()
        bad = bf16_gate_failures(dev, deviations(ref, {"out": decode(lay32, x.float(),
                                                                     torch.float32)}))
    assert not bad, f"{dtype}: {bad}"
