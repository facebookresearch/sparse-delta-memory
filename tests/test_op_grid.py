# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The write/read op on the test grid of ``op_grid.py``.

Every case must pass the numerically defined gates against the float64 reference and
launch only kernels of the op's allowlist. The anchors and the pairwise tier run by
default; the strength-3 tier is marked slow.
"""
import pytest
import torch

from op_grid import BASE, case_key, make_inputs, run_case, tier, uncovered
from sdm_testlib import (
    GATE_FP32,
    OP_KERNELS,
    kernels_outside,
    oracle_write_read,
    rel_max,
    requires_cuda,
)

pytestmark = requires_cuda


@pytest.fixture(autouse=True)
def _ieee_fp32(ieee_fp32):
    yield


def _check(case):
    rec = run_case(case, profile=True)
    if rec["status"] == "refused":   # fp8 / int4 rows of a quantized tail head: refused before any work
        assert rec["refused"] in ("T0", "T1", "T2", "T3") and rec["dispatch"] == {}, rec
        return
    assert rec["status"] == "pass", rec.get("error") or rec["failures"]
    assert rec["dispatch"]["save"] is True
    assert not kernels_outside(rec["kernels"], OP_KERNELS)


def test_covering_arrays_cover_every_pair_and_triple():
    assert uncovered(tier("default"), 2) == 0
    assert uncovered(tier("slow"), 3) == 0


@pytest.mark.parametrize("case", tier("anchors") + tier("default"), ids=case_key)
def test_case(case):
    _check(case)


@pytest.mark.slow
@pytest.mark.parametrize("case", tier("slow"), ids=case_key)
def test_slow_case(case):
    _check(case)


# Small cases at the extreme chunk sizes and widths, with colliding slots, cheap enough for
# compute-sanitizer, e.g.
#   compute-sanitizer --tool memcheck python -m pytest tests/test_op_grid.py -k sanitizer
SANITIZER_CASES = [dict(BASE, W=w, R=w, C=c, T="C-1", hd=64, dtype="float32", slots="collide")
                   for c in (16, 64, 256) for w in (1, 64, 256)]


@pytest.mark.parametrize("case", SANITIZER_CASES, ids=case_key)
def test_sanitizer_case(case):
    _check(case)


# fp32 cases for the forward without saved state: ragged T, collisions, per-write gates,
# the shared-bank layout.
NO_GRAD_CASES = [
    dict(BASE, dtype="float32", T="2C+1"),
    dict(BASE, dtype="float32", W=16, R=100, C=17, hd=65, kwd=True, slots="collide", T="4C+3"),
    dict(BASE, dtype="float32", W=48, R=1, C=48, hd=129, layout="2d", slots="hot_boundary"),
]


@pytest.mark.parametrize("case", NO_GRAD_CASES, ids=case_key)
def test_forward_without_grad_matches_the_reference_in_one_or_two_calls(case):
    """Under no_grad the op saves nothing; readings and the final bank still match the
    reference, also when the sequence is split into two calls that carry the bank."""
    from sparse_delta_memory.memory_ops import last_dispatch, sdm_write_read

    d, S, T, sph = make_inputs(case)
    readings, _, final = oracle_write_read(d, sph=sph)
    two_d = case["layout"] == "2d"

    def call(bank, lo, hi):
        def cut(t):
            t = t[:, lo:hi]
            return t.reshape(-1, t.shape[-1]) if two_d else t
        return sdm_write_read(
            bank, *(cut(d[n]) for n in ("k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")),
            chunk_size=case["C"], num_memory_slots=0 if two_d else sph,
            num_parallel=1 if two_d else S, key_weighted_decay=case["kwd"])

    split = case["C"] + 5 if T > case["C"] + 5 else T // 2
    for bounds in ((0, T), (0, split, T)):
        bank = d["memory"].clone()
        with torch.no_grad():
            parts = [call(bank, a, b).reshape(S, b - a, -1)
                     for a, b in zip(bounds[:-1], bounds[1:])]
        assert last_dispatch()["save"] is False
        got = torch.cat(parts, dim=1)
        assert rel_max(readings, got) <= GATE_FP32, bounds
        assert rel_max(final, bank) <= GATE_FP32, bounds
