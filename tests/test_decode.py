# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The single-token decode kernel (``sdm_decode_step``) against the float64 token loop.

The kernel uses plain stores and fixed-order reductions, so it is deterministic: repeated
runs and CUDA-graph replays are compared bitwise.
"""
import itertools

import pytest
import torch

from sdm_testlib import (
    bf16_gate_failures,
    bits_equal,
    deviations,
    make_op_inputs,
    oracle_write_read,
    require_free_memory,
    requires_cuda,
)
from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

pytestmark = requires_cuda

GATE = 1e-6          # rel_max for fp32 banks
S, SPH = 3, 1024     # streams and slots per stream
ARGS = ("memory", "k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")


def _inputs(W, R, hd, dtype, kwd=False, T=1, seed=0, **kinds):
    """[S, T, *] streams. Half of each token's reads (at least one) hit slots it writes, so
    the read-after-write order is exercised. ``kinds`` go to make_op_inputs."""
    d = make_op_inputs(S, T, W, R, hd, SPH, dtype, seed=seed, **kinds)
    if kwd:
        gen = torch.Generator().manual_seed(seed + 1)
        d["g"] = torch.empty(S, T, W).uniform_(-0.5, -0.01, generator=gen).to("cuda", dtype)
    n = max(1, min(W, R) // 2)
    d["q_idx"][..., :n] = d["k_idx"][..., :n]
    return d


def _step(memory, d, t=0):
    from sparse_delta_memory.decode import sdm_decode_step

    return sdm_decode_step(memory, *(d[k][:, t:t + 1] for k in ARGS[1:]))


def _decode(d):
    """(readings [S, T, hd], final bank): token by token on a private copy of the bank."""
    memory = d["memory"].clone()
    out = torch.cat([_step(memory, d, t) for t in range(d["k_idx"].shape[1])], dim=1)
    return out, memory


def _deviations(d, out, memory):
    """Deviations of the readings and the written rows; unwritten rows keep their bits."""
    ref, _, ref_memory = oracle_write_read(d)
    written = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
    written[d["k_idx"].reshape(-1)] = True
    assert bits_equal(memory[~written], d["memory"][~written]), "an unwritten row changed"
    return deviations({"readings": ref, "written": ref_memory[written]},
                      {"readings": out, "written": memory[written]})


def _fp32_failures(dev):
    return {k: v["rel_max"] for k, v in dev.items() if not v["rel_max"] <= GATE}


def _same(a, b):
    return all(bits_equal(x.reshape(y.shape), y) for x, y in zip(a, b))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("kwd", [False, True], ids=["token_decay", "kwd"])
@pytest.mark.parametrize("hd", [64, 65, 128, 960, 1024, 2048])
@pytest.mark.parametrize("W,R", list(itertools.product([1, 16, 64, 256], [1, 64, 256])))
def test_matches_the_float64_token_loop(W, R, hd, kwd, dtype):
    d = _inputs(W, R, hd, dtype, kwd)
    dev = _deviations(d, *_decode(d))
    if dtype == torch.float32:
        bad = _fp32_failures(dev)
    else:
        # The same bf16-valued inputs on an fp32 bank must be much closer (precision-shaped).
        d32 = {k: t.float() if t.is_floating_point() else t for k, t in d.items()}
        bad = bf16_gate_failures(dev, _deviations(d32, *_decode(d32)))
    assert not bad, f"W={W} R={R} hd={hd} kwd={kwd} {dtype}: {bad}"


@pytest.mark.parametrize("hd,W,R", [(1, 3, 5), (17, 48, 100), (4097, 255, 256)])
def test_any_head_dim_and_lane_count(hd, W, R):
    d = _inputs(W, R, hd, torch.float32, kwd=True)
    bad = _fp32_failures(_deviations(d, *_decode(d)))
    assert not bad, bad


def test_a_token_sequence_follows_the_recurrence():
    """Consecutive steps with hot slots rewritten across tokens, extreme gates (down to
    -20) and betas (near 0 and 1), on strided per-token views of [S, T, *] inputs."""
    d = _inputs(16, 64, 128, torch.float32, T=24, seed=3, slots="clustered",
                gates="extreme", betas="extreme")
    bad = _fp32_failures(_deviations(d, *_decode(d)))
    assert not bad, bad


def test_reads_see_every_write_of_their_step():
    """Reads equal to the writes, at many programs per stream: a read that missed its step's
    write would return the zero bank row instead of k_val * beta * v."""
    d = _inputs(64, 64, 1024, torch.float32, kwd=True, seed=4)
    d["q_idx"] = d["k_idx"].clone()
    d["memory"].zero_()
    d["beta"].fill_(1.0)
    out, memory = _decode(d)
    for _ in range(20):
        assert _same(_decode(d), (out, memory))
    assert not _fp32_failures(_deviations(d, out, memory))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_graph_replays_match_eager_steps_bitwise(dtype):
    from sparse_delta_memory.decode import sdm_decode_step

    T = 4
    d = _inputs(64, 64, 128, dtype, kwd=True, T=T, seed=5, slots="clustered")
    eager_memory = d["memory"].clone()
    eager = [(_step(eager_memory, d, t), eager_memory.clone()) for t in range(T)]
    assert _same(_decode(d), (torch.cat([o for o, _ in eager], 1), eager_memory))

    static = {k: d[k][:, :1].clone() for k in ARGS[1:]}
    memory = d["memory"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sdm_decode_step(memory, *static.values())
    for t in range(T):
        for k, buf in static.items():
            buf.copy_(d[k][:, t:t + 1])
        graph.replay()
        assert _same((out, memory), eager[t]), f"replay {t} differs from the eager step"


def test_inputs_round_to_the_bank_dtype_and_layouts_agree():
    """k_val, v and q_val round to the bank dtype, beta and g stay float32 (as in the chunked
    op); int32 and int64 indices, and 2-D and 3-D inputs, give identical bits."""
    d = _inputs(16, 64, 128, torch.float32, kwd=True)
    bf16_bank = dict(d, memory=d["memory"].bfloat16())
    rounded = dict(bf16_bank, **{k: d[k].bfloat16() for k in ("k_val", "v", "q_val")})
    assert _same(_decode(bf16_bank), _decode(rounded))

    low = {k: t.bfloat16() if t.is_floating_point() else t for k, t in d.items()}
    high = {k: t.float() if t.is_floating_point() else t for k, t in low.items()}
    assert _same(_decode(dict(low, memory=high["memory"])), _decode(high))

    from sparse_delta_memory.decode import sdm_decode_step

    ref = _decode(d)
    assert _same(_decode(dict(d, k_idx=d["k_idx"].int(), q_idx=d["q_idx"].int())), ref)
    memory = d["memory"].clone()
    out = sdm_decode_step(memory, *(d[k].reshape(S, -1) for k in ARGS[1:]))
    assert _same((out, memory), ref)


@pytest.mark.slow
def test_row_offsets_beyond_int32():
    """Slots whose element offset exceeds 2^31 give the same bits as a compact bank."""
    hd = 2048
    d = _inputs(64, 64, hd, torch.bfloat16)
    shift = 2**31 // hd + 1
    far = torch.empty(shift + S * SPH, hd, device="cuda", dtype=torch.bfloat16)
    far[shift:] = d["memory"]
    ref = _decode(d)
    out = _step(far, dict(d, k_idx=d["k_idx"] + shift, q_idx=d["q_idx"] + shift))
    assert bits_equal(out, ref[0]) and bits_equal(far[shift:], ref[1])


@pytest.mark.slow
def test_streams_times_head_dim_beyond_int32():
    """streams * head_dim > 2^31: the last streams give the same bits as run alone."""
    require_free_memory(40)
    streams, hd = 2**31 // 1024 + 1, 1024
    gen = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape):
        return torch.rand(*shape, device="cuda", generator=gen)

    slot = torch.arange(streams, device="cuda").view(streams, 1, 1)  # stream s owns row s
    d = dict(memory=rand(streams, hd) - 0.5, k_idx=slot, k_val=rand(streams, 1, 1),
             v=rand(streams, 1, hd) - 0.5, beta=rand(streams, 1, 1),
             g=-rand(streams, 1, 1), q_idx=slot, q_val=rand(streams, 1, 1))
    last = slice(streams - 4, streams)
    alone = {k: t[last].clone() for k, t in d.items()}
    alone["k_idx"] = alone["q_idx"] = slot[:4]
    ref = _step(alone["memory"], alone)
    out = _step(d["memory"], d)
    assert bits_equal(out[last], ref) and bits_equal(d["memory"][last], alone["memory"])


def test_rejects_unsupported_inputs():
    """The public sdm_decode_step refuses a bad bank dtype, gate shape, leading dims and read width."""
    d = _inputs(16, 16, 64, torch.float32)
    cases = [
        (TypeError, "bfloat16 or float32", dict(memory=d["memory"].half())),
        (ValueError, "last dim 1 or W", dict(g=d["g"].expand(S, 1, 3))),
        (ValueError, "v has shape", dict(v=d["v"][..., :32])),
        (ValueError, "k_val has shape", dict(k_val=d["k_val"][:2])),
        (ValueError, "W and R", dict(q_idx=d["q_idx"].repeat(1, 1, 17),
                                     q_val=d["q_val"].repeat(1, 1, 17))),
    ]
    for error, match, bad in cases:
        with pytest.raises(error, match=match):
            _decode(dict(d, **bad))


def test_layer_decode_equals_forward():
    """The layer decoding through its cache (a prefill, then 1-token steps) matches one full
    forward (moved here from test_layer_types.py, which needs the training code)."""
    torch.manual_seed(0)
    DIM, T, P = 256, 40, 24
    sdm = SparseDeltaMemoryArgs(
        dim=DIM, num_heads=1, slots_per_head=1024, num_reads=8, num_writes=8,
        chunk_size=16, output_gate=True, normalize_readings=True,
        read_act="Softmax", write_act="Softmax",
    )
    layer = SparseDeltaMemory(sdm, 0).cuda()
    layer.init_weights(init_std=0.02)
    layer = layer.float().eval()
    x = torch.randn(1, T, DIM, device="cuda")
    with torch.no_grad():
        out_full, _ = layer(x, cache=None)
        cache = layer.create_kv_cache(bsz=1, seq_len=0, dtype=torch.float32, device="cuda")
        out_pre, cache = layer(x[:, :P], cache=cache)
        chunks = [out_pre]
        for t in range(P, T):
            o, cache = layer(x[:, t:t + 1], cache=cache)
            chunks.append(o)
        out_inc = torch.cat(chunks, dim=1)
    max_delta = (out_full - out_inc).abs().max().item()
    assert max_delta < 1e-3, f"decode != forward: max|Δ|={max_delta:.3e}"
