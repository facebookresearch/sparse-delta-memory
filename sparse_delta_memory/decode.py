# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Single-token (T == 1) SDM step: write then read, one kernel launch for all streams."""

import torch
import triton
import triton.language as tl

from .memory_ops import MAX_WIDTH


@triton.jit
def _sdm_decode_kernel(
    mem_ptr, k_idx_ptr, k_val_ptr, v_ptr, beta_ptr, g_ptr, q_idx_ptr, q_val_ptr, out_ptr,
    hd,
    W: tl.constexpr, R: tl.constexpr, G: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < hd
    bank = mem_ptr.dtype.element_ty

    w = tl.arange(0, BLOCK_W)
    w_mask = w < W
    k_slot = tl.load(k_idx_ptr + s * W + w, mask=w_mask, other=0).to(tl.int64)
    k_val = tl.load(k_val_ptr + s * W + w, mask=w_mask, other=0.0).to(bank).to(tl.float32)
    # G == 1: one gate per token; G == W: one gate per write.
    decay = tl.exp(tl.load(g_ptr + s * G + w % G, mask=w_mask, other=0.0).to(tl.float32))
    # Read slots and weights do not depend on the bank: load them before the barrier.
    r = tl.arange(0, BLOCK_R)
    r_mask = r < R
    q_slot = tl.load(q_idx_ptr + s * R + r, mask=r_mask, other=0).to(tl.int64)
    q_val = tl.load(q_val_ptr + s * R + r, mask=r_mask, other=0.0).to(bank).to(tl.float32)
    k_ptrs = mem_ptr + k_slot[:, None] * hd + cols[None, :]
    k_mask = w_mask[:, None] & col_mask[None, :]
    rows = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

    retrieved = tl.sum((k_val * decay)[:, None] * rows, axis=0)
    v = tl.load(v_ptr + s * hd + cols, mask=col_mask, other=0.0).to(bank).to(tl.float32)
    delta_v = tl.load(beta_ptr + s).to(tl.float32) * (v - retrieved)
    # Write slots are distinct within a stream, so plain stores suffice.
    rows = rows * decay[:, None] + k_val[:, None] * delta_v[None, :]
    tl.store(k_ptrs, rows.to(bank), mask=k_mask)

    # The reads must see this step's writes. Only this program touches these columns of
    # this stream's slots, so a CTA barrier orders the stores before the loads.
    tl.debug_barrier()

    q_rows = tl.load(mem_ptr + q_slot[:, None] * hd + cols[None, :],
                     mask=r_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    readings = tl.sum(q_val[:, None] * q_rows, axis=0)
    tl.store(out_ptr + s * hd + cols, readings.to(bank), mask=col_mask)


def sdm_decode_step(
    memory: torch.Tensor,
    k_idx: torch.Tensor,
    k_val: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    q_idx: torch.Tensor,
    q_val: torch.Tensor,
) -> torch.Tensor:
    """Advance ``memory`` by one token per stream and read it back.

    Per stream, with d = exp(g):
        retrieved   = sum_w k_val[w] * d[w] * memory[k_idx[w]]
        delta_v     = beta * (v - retrieved)
        memory[k_w] = d[w] * memory[k_w] + k_val[w] * delta_v
        readings    = sum_r q_val[r] * memory[q_idx[r]]      (after the write)

    Args:
        memory: [num_slots, head_dim] contiguous bank, bfloat16 or float32; updated in
            place. Each stream's slots must be private to it (the layer adds per-stream
            offsets) and its W write slots distinct; neither is checked.
        k_idx, k_val: [..., W] write slots (int32 or int64) and weights, W <= 256.
        v: [..., head_dim] values. beta: [..., 1] write strengths.
        g: [..., 1] log-decay per token, or [..., W] per write.
        q_idx, q_val: [..., R] read slots and weights, R <= 256.
    All inputs share the leading dims, one entry per stream. As in the chunked op, k_val,
    v and q_val are rounded to the bank dtype, and beta and g are used in float32.

    Returns:
        readings shaped like ``v``, in the bank dtype.
    """
    if memory.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"memory must be bfloat16 or float32, got {memory.dtype}")
    if memory.ndim != 2 or not memory.is_cuda or not memory.is_contiguous():
        raise ValueError("memory must be a contiguous CUDA tensor [num_slots, head_dim]")
    if memory.shape[0] < 1 or memory.shape[1] < 1:
        raise ValueError(f"memory needs at least one slot and one column, got shape "
                         f"{tuple(memory.shape)}")
    if k_idx.dtype not in (torch.int32, torch.int64) or q_idx.dtype not in (torch.int32,
                                                                            torch.int64):
        raise TypeError(f"k_idx and q_idx must be int32 or int64, got {k_idx.dtype} and "
                        f"{q_idx.dtype}")
    dev = memory.get_device()
    for name, t in (("k_idx", k_idx), ("k_val", k_val), ("v", v), ("beta", beta), ("g", g),
                    ("q_idx", q_idx), ("q_val", q_val)):
        if not t.is_cuda or t.get_device() != dev:
            raise ValueError(f"{name} is on {t.device}, memory on {memory.device}")
    hd = memory.shape[1]
    W, R, G = k_idx.shape[-1], q_idx.shape[-1], g.shape[-1]
    if not (1 <= W <= MAX_WIDTH and 1 <= R <= MAX_WIDTH):
        raise ValueError(f"W and R must be in 1..{MAX_WIDTH}, got W={W}, R={R}")
    if G not in (1, W):
        raise ValueError(f"g must have last dim 1 or W={W}, got {G}")
    lead = k_idx.shape[:-1]
    for name, t, n in (("k_val", k_val, W), ("v", v, hd), ("beta", beta, 1), ("g", g, G),
                       ("q_idx", q_idx, R), ("q_val", q_val, R)):
        if t.shape != (*lead, n):
            raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {(*lead, n)}")
    return _decode_launch(memory, k_idx, k_val, v, beta, g, q_idx, q_val)


def _decode_launch(memory, k_idx, k_val, v, beta, g, q_idx, q_val):
    """sdm_decode_step without its argument checks: the layer's decode branch, whose inputs the layer builds and whose
    bank was validated at the prefill (or made by create_kv_cache)."""
    hd = memory.shape[1]
    W, R, G = k_idx.shape[-1], q_idx.shape[-1], g.shape[-1]
    streams = k_idx.shape[:-1].numel()

    def flat(t):
        return t.reshape(streams, t.shape[-1]).contiguous()

    readings = torch.empty(v.shape, device=memory.device, dtype=memory.dtype)
    block_w, block_r = triton.next_power_of_2(W), triton.next_power_of_2(R)
    # Narrow column blocks: many programs keep many row loads in flight, and the step is
    # latency bound. About 128 bytes of bank rows per thread.
    block_d = min(16, triton.next_power_of_2(hd))
    tile_bytes = max(block_w, block_r) * block_d * memory.element_size()
    num_warps = min(4, max(1, tile_bytes // (128 * 32)))
    _sdm_decode_kernel[(streams, triton.cdiv(hd, block_d))](
        memory, flat(k_idx), flat(k_val), flat(v), flat(beta), flat(g), flat(q_idx),
        flat(q_val), readings, hd,
        W=W, R=R, G=G, BLOCK_W=block_w, BLOCK_R=block_r, BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return readings
