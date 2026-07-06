# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Triton STABLE bitonic argsort for small arrays (W <= 128).

Uses compound key (key, original_index) to ensure stability:
ties in key are broken by original position (ascending).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_kernel(
    keys_ptr,       # [N, W] int64 input (modified in-place: sorted)
    perm_ptr,       # [N, W] int64 output (permutation indices)
    N: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Stable ascending sort by keys. Ties broken by original index (ascending)."""
    pid = tl.program_id(0)
    if pid >= N:
        return

    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    base = pid * W

    keys = tl.load(keys_ptr + base + offs, mask=mask, other=0x7FFFFFFFFFFFFFFF)
    perm = offs.to(tl.int64)

    size = 1
    while size < BLOCK_W:
        stride = size
        while stride > 0:
            partner = offs ^ stride
            ascending = ((offs & (size << 1)) == 0)

            p_key = tl.load(keys_ptr + base + partner, mask=(partner < W), other=0x7FFFFFFFFFFFFFFF)
            p_perm = tl.load(perm_ptr + base + partner, mask=(partner < W), other=0x7FFFFFFFFFFFFFFF)

            is_lower_idx = offs < partner
            want_smaller = (is_lower_idx & ascending) | ((~is_lower_idx) & (~ascending))

            # STABLE comparison: if keys equal, compare by original index (perm)
            my_is_smaller = (keys < p_key) | ((keys == p_key) & (perm < p_perm))

            should_take_partner = (want_smaller & (~my_is_smaller) & (partner < W)) | \
                                  ((~want_smaller) & my_is_smaller & (partner < W))

            keys = tl.where(should_take_partner, p_key, keys)
            perm = tl.where(should_take_partner, p_perm, perm)

            tl.store(keys_ptr + base + offs, keys, mask=mask)
            tl.store(perm_ptr + base + offs, perm, mask=mask)
            tl.debug_barrier()

            keys = tl.load(keys_ptr + base + offs, mask=mask, other=0x7FFFFFFFFFFFFFFF)
            perm = tl.load(perm_ptr + base + offs, mask=mask, other=0)

            stride = stride >> 1
        size = size << 1


def triton_argsort(keys: torch.Tensor):
    """Stable ascending argsort for small arrays. Returns (sorted_keys, permutation)."""
    N, W = keys.shape
    BLOCK_W = triton.next_power_of_2(W)
    keys_out = keys.clone()
    perm = torch.arange(W, device=keys.device, dtype=torch.int64).unsqueeze(0).expand(N, -1).contiguous()
    _stable_argsort_kernel[(N,)](keys_out, perm, N=N, W=W, BLOCK_W=BLOCK_W, num_warps=1)
    return keys_out, perm
