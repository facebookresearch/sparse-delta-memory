# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Input validation of the two public ops, before any kernel runs: the training op
(``sdm_write_read``) and the single-token decode step (``sdm_decode_step``) refuse operands on
another device than the bank, non-integer slot ids, an empty bank and a negative
``num_memory_slots`` with a TypeError or ValueError, and still accept valid calls."""
import pytest
import torch

from sdm_testlib import make_op_inputs, requires_cuda

pytestmark = requires_cuda

SPH, DIM, W, R, P = 256, 64, 4, 4, 2
NAMES = ("memory", "k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")


def _inputs(T):
    d = make_op_inputs(P, T, W, R, DIM, SPH, torch.float32, seed=3)
    return {n: d[n] for n in NAMES}


def _write_read(a, num_memory_slots=SPH):
    from sparse_delta_memory.memory_ops import sdm_write_read

    with torch.no_grad():
        return sdm_write_read(*(a[n] for n in NAMES), chunk_size=64,
                              num_memory_slots=num_memory_slots, num_parallel=P)


def _decode(a):
    from sparse_delta_memory.decode import sdm_decode_step

    return sdm_decode_step(*(a[n] for n in NAMES))


OPS = {"write_read": (_write_read, 32), "decode": (_decode, 1)}


@pytest.mark.parametrize("op", sorted(OPS))
def test_valid_call_runs(op):
    fn, T = OPS[op]
    a = _inputs(T)
    out = fn(a)
    torch.cuda.synchronize()
    # The training op returns [P * T, head_dim]; the decode step returns readings shaped like v.
    expected = (P * T, DIM) if op == "write_read" else tuple(a["v"].shape)
    assert tuple(out.shape) == expected and torch.isfinite(out).all(), tuple(out.shape)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("name", NAMES[1:])
def test_operand_on_cpu_is_refused(op, name):
    fn, T = OPS[op]
    a = _inputs(T)
    a[name] = a[name].cpu()
    with pytest.raises(ValueError, match=f"{name} is on cpu"):
        fn(a)


@pytest.mark.parametrize("op", sorted(OPS))
def test_bank_on_cpu_is_refused(op):
    fn, T = OPS[op]
    a = _inputs(T)
    a["memory"] = a["memory"].cpu()
    with pytest.raises(ValueError, match="CUDA tensor"):
        fn(a)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA GPUs")
@pytest.mark.parametrize("op", sorted(OPS))
def test_operand_on_another_gpu_is_refused(op):
    fn, T = OPS[op]
    a = _inputs(T)
    other = f"cuda:{(a['memory'].device.index + 1) % torch.cuda.device_count()}"
    a["v"] = a["v"].to(other)
    with pytest.raises(ValueError, match=f"v is on {other}, memory on {a['memory'].device}"):
        fn(a)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("name", ("k_idx", "q_idx"))
def test_float_slot_ids_are_refused(op, name):
    fn, T = OPS[op]
    a = _inputs(T)
    a[name] = a[name].float()
    with pytest.raises(TypeError, match="int32 or int64"):
        fn(a)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("shape", [(0, DIM), (P * SPH, 0)])
def test_empty_bank_is_refused(op, shape):
    fn, T = OPS[op]
    a = _inputs(T)
    a["memory"] = torch.zeros(shape, device="cuda")
    with pytest.raises(ValueError, match="at least one slot and one column"):
        if op == "write_read":
            _write_read(a, num_memory_slots=0)
        else:
            fn(a)


def test_negative_num_memory_slots_is_refused():
    with pytest.raises(ValueError, match="num_memory_slots must be >= 0"):
        _write_read(_inputs(32), num_memory_slots=-SPH)
