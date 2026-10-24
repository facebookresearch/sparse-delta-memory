# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tails in Phase A (native backward)."""
import pytest
import torch

from sdm_testlib import requires_cuda
from test_mover_row_types import _assert_same, _pa_pair
from test_fx_tails import _backend, _chunk_ok

pytestmark = [requires_cuda]


TAIL_PA_CASES = [   # (bank dtype, grad dtype, P, Nt, C, W, hd, pool): dc_pa3 slices, 8- or 16-byte bf16 pieces; half of
    # each stream's slots in its last `pool` rows (the last stream's: the bank's final rows)
    (torch.bfloat16, torch.bfloat16, 2, 3, 64, 8, 12, 48),       # 24-byte rows: 8-byte pieces, D 16
    (torch.bfloat16, torch.bfloat16, 2, 2, 128, 16, 100, 96),    # 200-byte rows
    (torch.bfloat16, torch.bfloat16, 2, 2, 64, 8, 320, 64),      # slices 108 / 108 / 104 (216-byte slice rows)
    (torch.bfloat16, torch.bfloat16, 2, 2, 32, 4, 1000, 32),     # slices 7 x 128 + 104 (16-byte pieces)
    (torch.bfloat16, torch.bfloat16, 1, 2, 192, 8, 260, 64),     # slices 88 / 88 / 84, 16 warps per CTA
    (torch.bfloat16, torch.float32, 2, 3, 64, 8, 100, 48),       # f32 grad rows, bf16 snapshot rows in 8-byte pieces
    (torch.bfloat16, torch.float32, 2, 2, 128, 16, 320, 96),
    (torch.float32, torch.float32, 2, 3, 64, 8, 132, 48),        # slices 68 / 64
    (torch.float32, torch.float32, 2, 2, 128, 16, 320, 96),
    (torch.float32, torch.float32, 2, 2, 32, 4, 1000, 32),
    (torch.bfloat16, torch.bfloat16, 2, 2, 64, 4, 16388, 16),    # 128 slices of 128 + a 4-column last slice
    (torch.bfloat16, torch.float32, 2, 2, 64, 4, 16388, 16),
    (torch.float32, torch.float32, 2, 2, 64, 4, 16388, 16),
]


@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,W,hd,pool", TAIL_PA_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_W{c[5]}_hd{c[6]}" for c in TAIL_PA_CASES])
def test_gx_pa2_tails_match_dc_pa3(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool):
    _chunk_ok(C)
    ref, got = _pa_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, sph=4 * pool, hi=4 * pool)
    _assert_same(ref, got, "gx_pa2 tail vs dc_pa3")
    assert not ref["grad"].isnan().any() and not got["eff"][:, :1].isnan().all()


def test_gx_pa2_tail_special_values():
    """-0.0, subnormal, +-inf and NaN in bank, grad and snapshot rows of a tail shape: dc_pa3's bits."""
    _backend()
    for bank_dtype, grad_dtype in ((torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)):
        ref, got = _pa_pair(bank_dtype, grad_dtype, 2, 2, 64, 8, 100, 32, 128, special=True)
        _assert_same(ref, got, f"gx_pa2 tail vs dc_pa3 (special values, {bank_dtype})")


L0_CASES = [   # (bank dtype, grad dtype, P, Nt, C, W, hd, pool): dc_pa3's per-entry snapshot reader (dbg 512)
    (torch.bfloat16, torch.bfloat16, 2, 3, 64, 8, 128, 48),      # regular rows, D 16
    (torch.bfloat16, torch.bfloat16, 2, 2, 128, 16, 1024, 96),   # regular, 8 slices
    (torch.bfloat16, torch.float32, 2, 2, 64, 8, 256, 64),       # regular, f32 grad rows
    (torch.float32, torch.float32, 2, 2, 64, 8, 128, 48),        # regular f32 rows
    (torch.bfloat16, torch.bfloat16, 2, 2, 64, 8, 320, 64),      # tails
    (torch.bfloat16, torch.bfloat16, 2, 3, 64, 8, 12, 48),
    (torch.bfloat16, torch.float32, 2, 2, 128, 16, 320, 96),
    (torch.float32, torch.float32, 2, 3, 64, 8, 132, 48),
    (torch.bfloat16, torch.bfloat16, 1, 2, 192, 8, 260, 64),     # 16 warps per CTA
]


@pytest.mark.parametrize("bank_dtype,grad_dtype,P,Nt,C,W,hd,pool", L0_CASES,
                         ids=[f"{str(c[0])[6:]}_{str(c[1])[6:]}_C{c[4]}_W{c[5]}_hd{c[6]}" for c in L0_CASES])
def test_gx_pa2_entry_snapshot_matches_dc_pa3(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool):
    """dbg 512 (layout 0): gx_pa2 reads each group's snapshot row at dc_pa3's per-entry index; same bits."""
    _chunk_ok(C)
    ref, got = _pa_pair(bank_dtype, grad_dtype, P, Nt, C, W, hd, pool, sph=4 * pool, hi=4 * pool, dbg=512)
    _assert_same(ref, got, "gx_pa2 per-entry snapshot vs dc_pa3")
    assert not ref["grad"].isnan().any() and not got["eff"][:, :1].isnan().all()


def test_gx_pa2_entry_snapshot_special_values():
    _backend()
    for bank_dtype, grad_dtype, hd in ((torch.bfloat16, torch.bfloat16, 320), (torch.float32, torch.float32, 128)):
        ref, got = _pa_pair(bank_dtype, grad_dtype, 2, 2, 64, 8, hd, 48, sph=192, seed=5, special=True, dbg=512)
        _assert_same(ref, got, f"gx_pa2 per-entry snapshot special values {bank_dtype}")

