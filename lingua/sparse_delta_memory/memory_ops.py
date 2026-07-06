# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Memory controller operations including Triton sparse inner product kernels.

See GATED_MEMCTRL_README.md in this directory for kernel design notes,
performance lessons, and pitfalls (float32 precision, sparse vs dense
tradeoffs, what fusions work and don't).

Contains:
- Triton sparse inner product: batched computation of A[i,j] = sum over
  shared slots of val_row[i,:] * val_col[j,:], with autograd support.
  Used by both the gated and non-gated memory controllers for intra-chunk
  causal attention via sparse key/query matching.
- Fused snapshot + gather kernel for Phase 2 loop optimization.
- Fused scatter write kernel for memory updates.
- Decay kernels (unique-slot and sequential variants).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

# ============================================================
# Snapshot quantization helpers
# ============================================================

def _dequantize_snapshot_int4(packed: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize packed int4 snapshot back to original dtype.

    Args:
        packed: [*, dim//2] uint8 tensor
        scale: [*, 1] float32 per-row scale factors

    Returns:
        x: [*, dim] tensor in out_dtype
    """
    # Unpack high nibble (arithmetic right shift preserves sign for signed int4)
    high = packed.to(torch.int8) >> 4
    # Unpack low nibble (mask then sign-extend)
    low_unsigned = (packed & 0x0F).to(torch.int8)
    low = torch.where(low_unsigned >= 8, low_unsigned - 16, low_unsigned)
    # Interleave and dequantize
    dim_half = packed.shape[-1]
    scale_cast = scale.to(out_dtype)
    result = torch.empty(*packed.shape[:-1], dim_half * 2, dtype=out_dtype, device=packed.device)
    result[..., 0::2] = high.to(out_dtype) * scale_cast
    result[..., 1::2] = low.to(out_dtype) * scale_cast
    return result


def _dequant_snapshot_slice(snap_q, snap_s, snapshot_quant, sl, out_dtype):
    """Dequantize a slice of the quantized snapshot. Used for lazy backward dequant."""
    if snapshot_quant == "fp8":
        return snap_q[sl].to(out_dtype) * snap_s[sl].to(out_dtype)
    else:  # int4
        return _dequantize_snapshot_int4(snap_q[sl], snap_s[sl], out_dtype)

# ============================================================
# Triton sparse inner product kernel
# ============================================================

@triton.jit
def _sparse_inner_product_kernel(
    idx_row_ptr, val_row_ptr,
    idx_col_ptr, val_col_ptr,
    out_ptr,
    C: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
):
    """Sparse inner product via broadcast comparison.

    Each program computes one row: out[ci, i, 0..C-1].
    For each (i, j) pair, broadcasts the W_ROW sparse entries of row i
    against the W_COL sparse entries of row j, accumulating products
    where indices match.
    """
    pid = tl.program_id(0)
    ci = pid // C
    i = pid % C

    row_offset = (ci * C + i) * W_ROW
    offs_r = tl.arange(0, W_ROW)
    idx_i = tl.load(idx_row_ptr + row_offset + offs_r)
    val_i = tl.load(val_row_ptr + row_offset + offs_r)

    out_base = ci * C * C + i * C

    for j in range(C):
        col_offset = (ci * C + j) * W_COL
        offs_c = tl.arange(0, W_COL)
        idx_j = tl.load(idx_col_ptr + col_offset + offs_c)
        val_j = tl.load(val_col_ptr + col_offset + offs_c)

        match = idx_i[:, None] == idx_j[None, :]
        prod = val_i[:, None] * val_j[None, :]
        result = tl.sum(tl.where(match, prod, 0.0))

        tl.store(out_ptr + out_base + j, result)

def sparse_inner_product_gated(
    idx_row: torch.Tensor,       # [num_chunks, C, W_ROW] — must be sorted along W
    val_row: torch.Tensor,       # [num_chunks, C, W_ROW]
    log_cumul_row: torch.Tensor, # [num_chunks, C, W_ROW]
    idx_col: torch.Tensor,       # [num_chunks, C, W_COL] — must be sorted along W
    val_col: torch.Tensor,       # [num_chunks, C, W_COL]
    log_cumul_col: torch.Tensor, # [num_chunks, C, W_COL]
    include_diag: bool = False,  # False: strict i>j (for A), True: inclusive i>=j (for QK)
) -> torch.Tensor:
    """Causal gated sparse inner product using CUDA two-pointer merge.
    Indices must be pre-sorted along the W dimension."""
    from .cuda.sparse_ip_cuda import sparse_ip_gated_fwd_sorted
    causal_mode = 2 if include_diag else 1
    return sparse_ip_gated_fwd_sorted(
        idx_row.contiguous(),
        val_row.contiguous(),
        log_cumul_row.contiguous(),
        idx_col.contiguous(),
        val_col.contiguous(),
        log_cumul_col.contiguous(),
        causal_mode)

# ============================================================

# ============================================================

@triton.jit
def _decay_at_unique_kernel(
    memory_ptr,       # [num_slots, dim]
    unique_k_ptr,     # [N_UNIQUE] unique slot indices
    cumul_ptr,        # [num_slots, 1] total cumulative decay per slot
    N_UNIQUE: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply decay to memory at unique slot positions only.

    memory[slot] *= cumul[slot] for each unique slot.
    No duplicate-index race condition since each slot appears exactly once.
    """
    pid = tl.program_id(0)
    if pid >= N_UNIQUE:
        return
    slot = tl.load(unique_k_ptr + pid)
    decay = tl.load(cumul_ptr + slot)
    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < dim
        val = tl.load(memory_ptr + slot * dim + d_offs, mask=mask, other=0.0)
        tl.store(memory_ptr + slot * dim + d_offs, val * decay, mask=mask)

def decay_at_unique_slots(
    memory: torch.Tensor,        # [num_slots, dim] — modified in-place
    unique_k: torch.Tensor,      # [N_UNIQUE] unique slot indices
    total_cumul: torch.Tensor,   # [num_slots, 1] cumulative decay
):
    """Apply decay to memory at unique slot positions via Triton.

    ~8x faster than PyTorch fancy-index `memory[flat_k] = memory[flat_k] * cumul[flat_k]`
    because it avoids the duplicate-index overhead in PyTorch's scatter.
    """
    N_UNIQUE = unique_k.shape[0]
    dim = memory.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))

    _decay_at_unique_kernel[(N_UNIQUE,)](
        memory,
        unique_k.contiguous().to(torch.int64),
        total_cumul.contiguous().to(memory.dtype),
        N_UNIQUE=N_UNIQUE, dim=dim, BLOCK_D=BLOCK_D,
    )

# ============================================================
# Fused scatter-multiply-add kernel for memory writes
# ============================================================

@triton.jit
def _fused_scatter_write_kernel(
    memory_ptr,       # [num_slots, dim]
    k_idx_ptr,        # [C, W]
    k_val_ptr,        # [C, W]
    delta_v_ptr,      # [C, dim]
    post_decay_ptr,   # [C, W]
    C: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused scatter write: memory[k_idx[t,n]] += k_val[t,n] * delta_v[t,:] * post_decay[t,n]

    Avoids materializing [C, W, dim] intermediate tensor.
    Each program handles one (t, n) pair across dim blocks.
    """
    pid = tl.program_id(0)
    t = pid // W
    n = pid % W

    # Load scalar values for this (t, n)
    idx = tl.load(k_idx_ptr + t * W + n)
    kv = tl.load(k_val_ptr + t * W + n)
    pd = tl.load(post_decay_ptr + t * W + n)
    scale = kv * pd

    # Process dim in blocks
    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < dim

        # Load delta_v[t, d_start:d_start+BLOCK_D]
        dv = tl.load(delta_v_ptr + t * dim + d_offs, mask=mask, other=0.0)

        # Compute contribution and atomically add to memory
        contrib = scale * dv
        tl.atomic_add(memory_ptr + idx * dim + d_offs, contrib, mask=mask, sem="relaxed")

def fused_scatter_write(
    memory: torch.Tensor,      # [num_slots, dim] — modified in-place
    k_idx: torch.Tensor,       # [C, W]
    k_val: torch.Tensor,       # [C, W]
    delta_v: torch.Tensor,     # [C, dim]
    post_decay: torch.Tensor,  # [C, W]
):
    """Fused scatter write: memory[k_idx[t,n]] += k_val[t,n] * delta_v[t,:] * post_decay[t,n]

    Replaces: write_vals = einsum('cn,cd->cnd', k_val, delta_v) * post_decay.unsqueeze(2)
              memory.index_add_(0, flat_k, write_vals.reshape(-1, dim))
    Avoids the [C, W, dim] intermediate tensor.
    """
    C, W = k_idx.shape
    dim = delta_v.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))

    _fused_scatter_write_kernel[(C * W,)](
        memory,
        k_idx.contiguous().to(torch.int64),
        k_val.contiguous().to(memory.dtype),
        delta_v.contiguous().to(memory.dtype),
        post_decay.contiguous().to(memory.dtype),
        C=C, W=W, dim=dim, BLOCK_D=BLOCK_D,
        num_warps=1,
    )

# ============================================================
# Fused snapshot + weighted gather kernel for Phase 2 loop
# ============================================================

@triton.jit
def _fused_snapshot_gather_kernel(
    memory_ptr,         # [num_slots, dim]
    k_idx_ptr,          # [C, W_K] key slot indices for this chunk
    k_weights_ptr,      # [C, W_K] key weights (K_eff = k_val * cumul_k)
    q_idx_ptr,          # [C, W_Q] query slot indices for this chunk
    q_weights_ptr,      # [C, W_Q] query weights (q_val * cumul_q)
    snapshot_ptr,       # [C, W_K, dim] output: memory snapshot at k positions
    retrieved_ptr,      # [C, dim] output: weighted sum of memory at k positions
    inter_ptr,          # [C, dim] output: weighted sum of memory at q positions
    C: tl.constexpr,
    W_K: tl.constexpr,
    W_Q: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACC_DTYPE: tl.constexpr,  # tl.float32 or tl.float64
):
    """Fused snapshot + two weighted gathers from memory.

    For each token t in [0, C):
      1. snapshot[t, n, :] = memory[k_idx[t, n], :] for n in [0, W_K)
      2. retrieved[t, d] = sum_n k_weights[t, n] * memory[k_idx[t, n], d]
      3. inter[t, d] = sum_n q_weights[t, n] * memory[q_idx[t, n], d]

    Grid: (C, num_d_blocks) — one program per (token, dim_tile) pair.
    Uses a 2D grid to increase occupancy: with BLOCK_D=256 and dim=2048,
    this gives C*8 = 1024 programs instead of C=128 (baseline), achieving
    ~2.5x speedup at dim=2048 on H100.
    """
    t = tl.program_id(0)
    d_block = tl.program_id(1)
    d_start = d_block * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    # --- Weighted gather at k positions (retrieved) + snapshot ---
    acc_retrieved = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)
    for n in range(W_K):
        slot = tl.load(k_idx_ptr + t * W_K + n)
        weight = tl.load(k_weights_ptr + t * W_K + n).to(ACC_DTYPE)
        mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0)
        # Store snapshot (in memory dtype)
        tl.store(snapshot_ptr + (t * W_K + n) * dim + d_offs, mem_val, mask=d_mask)
        # Accumulate weighted sum
        acc_retrieved += weight * mem_val.to(ACC_DTYPE)

    tl.store(retrieved_ptr + t * dim + d_offs, acc_retrieved, mask=d_mask)

    # --- Weighted gather at q positions (inter) ---
    acc_inter = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)
    for n in range(W_Q):
        slot = tl.load(q_idx_ptr + t * W_Q + n)
        weight = tl.load(q_weights_ptr + t * W_Q + n).to(ACC_DTYPE)
        mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0)
        acc_inter += weight * mem_val.to(ACC_DTYPE)

    tl.store(inter_ptr + t * dim + d_offs, acc_inter, mask=d_mask)

def fused_snapshot_gather(
    memory: torch.Tensor,       # [num_slots, dim]
    k_idx: torch.Tensor,        # [C, W_K] key slot indices
    k_weights: torch.Tensor,    # [C, W_K] key weights
    q_idx: torch.Tensor,        # [C, W_Q] query slot indices
    q_weights: torch.Tensor,    # [C, W_Q] query weights
    snapshot: torch.Tensor,     # [C, W_K, dim] output (pre-allocated)
    retrieved: torch.Tensor,    # [C, dim] output (pre-allocated)
    inter: torch.Tensor,        # [C, dim] output (pre-allocated)
):
    """Fused snapshot + two weighted gathers. Writes results into pre-allocated output tensors.

    Replaces 3 separate operations:
    1. snapshot = memory[flat_k].reshape(C, W_K, dim)
    2. retrieved = F.embedding_bag(k_idx, memory, per_sample_weights=k_weights, mode='sum')
    3. inter = F.embedding_bag(q_idx, memory, per_sample_weights=q_weights, mode='sum')

    Uses CUDA warp-cooperative kernel for bf16 memory (1.7-2.7x faster than Triton),
    falls back to Triton for other dtypes or if CUDA compilation fails.
    Supports f32 output buffers for retrieved/inter (snapshot always stays in memory dtype).
    """
    if memory.dtype == torch.bfloat16:
        # Use CUDA warp-cooperative gather (vectorized int4 loads, 2D grid)
        # The CUDA kernel handles both bf16 and f32 output dtypes for retrieved/inter.
        try:
            from .cuda.warp_cooperative_gather_cuda import (
                warp_cooperative_gather,
            )
            warp_cooperative_gather(
                memory, k_idx, k_weights, q_idx, q_weights,
                snapshot, retrieved, inter,
                block_threads=128,
            )
            return
        except ImportError:
            pass  # CUDA compilation failed, fall through to Triton

    C, W_K = k_idx.shape
    W_Q = q_idx.shape[1]
    dim = memory.shape[1]
    # Full-dim load when dim fits in registers (≤1024), tiled otherwise.
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    num_d_blocks = (dim + BLOCK_D - 1) // BLOCK_D

    # Use float64 accumulation for float64 memory, float32 otherwise
    acc_dtype = tl.float64 if memory.dtype == torch.float64 else tl.float32

    _fused_snapshot_gather_kernel[(C, num_d_blocks)](
        memory,
        k_idx.contiguous().to(torch.int64),
        k_weights.contiguous().to(memory.dtype),
        q_idx.contiguous().to(torch.int64),
        q_weights.contiguous().to(memory.dtype),
        snapshot,
        retrieved,
        inter,
        C=C, W_K=W_K, W_Q=W_Q, dim=dim, BLOCK_D=BLOCK_D,
        ACC_DTYPE=acc_dtype,
    )

@triton.jit
def _fused_dual_gather_kernel(
    memory_ptr,         # [num_slots, dim]
    k_idx_ptr,          # [C, W_K]
    k_weights_ptr,      # [C, W_K]
    q_idx_ptr,          # [C, W_Q]
    q_weights_ptr,      # [C, W_Q]
    retrieved_ptr,      # [C, dim] output
    inter_ptr,          # [C, dim] output
    C: tl.constexpr,
    W_K: tl.constexpr,
    W_Q: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """Two fused weighted gathers from memory, no snapshot.

    For each token t in [0, C):
      1. retrieved[t, d] = sum_n k_weights[t, n] * memory[k_idx[t, n], d]
      2. inter[t, d] = sum_n q_weights[t, n] * memory[q_idx[t, n], d]

    Grid: (C, num_d_blocks) — one program per (token, dim_tile) pair.
    Uses a 2D grid for higher occupancy (same optimization as snapshot variant).
    """
    t = tl.program_id(0)
    d_block = tl.program_id(1)
    d_start = d_block * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    acc_retrieved = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)
    for n in range(W_K):
        slot = tl.load(k_idx_ptr + t * W_K + n)
        weight = tl.load(k_weights_ptr + t * W_K + n).to(ACC_DTYPE)
        mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0)
        acc_retrieved += weight * mem_val.to(ACC_DTYPE)
    tl.store(retrieved_ptr + t * dim + d_offs, acc_retrieved, mask=d_mask)

    acc_inter = tl.zeros([BLOCK_D], dtype=ACC_DTYPE)
    for n in range(W_Q):
        slot = tl.load(q_idx_ptr + t * W_Q + n)
        weight = tl.load(q_weights_ptr + t * W_Q + n).to(ACC_DTYPE)
        mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0)
        acc_inter += weight * mem_val.to(ACC_DTYPE)
    tl.store(inter_ptr + t * dim + d_offs, acc_inter, mask=d_mask)

def fused_dual_gather(
    memory: torch.Tensor,       # [num_slots, dim]
    k_idx: torch.Tensor,        # [C, W_K]
    k_weights: torch.Tensor,    # [C, W_K]
    q_idx: torch.Tensor,        # [C, W_Q]
    q_weights: torch.Tensor,    # [C, W_Q]
    retrieved: torch.Tensor,    # [C, dim] output (pre-allocated)
    inter: torch.Tensor,        # [C, dim] output (pre-allocated)
):
    """Fused two weighted gathers, no snapshot. For inference (no backward needed).

    Uses a 2D grid (C, num_d_blocks) with BLOCK_D=256 for higher occupancy.
    """
    C, W_K = k_idx.shape
    W_Q = q_idx.shape[1]
    dim = memory.shape[1]
    # Full-dim load when dim fits in registers (≤1024), tiled otherwise.
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    num_d_blocks = (dim + BLOCK_D - 1) // BLOCK_D
    acc_dtype = tl.float64 if memory.dtype == torch.float64 else tl.float32

    _fused_dual_gather_kernel[(C, num_d_blocks)](
        memory,
        k_idx.contiguous().to(torch.int64),
        k_weights.contiguous().to(memory.dtype),
        q_idx.contiguous().to(torch.int64),
        q_weights.contiguous().to(memory.dtype),
        retrieved,
        inter,
        C=C, W_K=W_K, W_Q=W_Q, dim=dim, BLOCK_D=BLOCK_D,
        ACC_DTYPE=acc_dtype,
    )

# ============================================================
# Fused decay + scatter write kernel
# ============================================================

@triton.jit
def _fused_decay_and_write_kernel(
    memory_ptr,         # [num_slots, dim]
    snapshot_ptr,       # [SCW, dim] contiguous (snapshot = pre-decay memory values)
    flat_k_ptr,         # [CW] slot indices
    cumul_ptr,          # [CW] cumul_per_entry
    is_last_ptr,        # [CW] int32: 1 if last entry for this slot
    is_only_ptr,        # [CW] int32: 1 if sole entry for this slot in chunk
    k_val_ptr,          # [C, W] key values for write (any float dtype)
    delta_v_ptr,        # [C, dim] delta values (any float dtype)
    post_decay_ptr,     # [C, W] post-decay factors (any float dtype)
    CW: tl.constexpr,
    C_local: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused decay + scatter write via additive formulation.

    Combines decay and scatter_write into one kernel, halving memory traffic.
    All arithmetic is in f32 regardless of input dtypes. Inputs can be bf16 or f32.

    For is_only entries (~95%): plain store eliminates atomic RMW read:
      memory[slot] = snapshot * cumul + k_val * delta_v * post_decay

    For multi-entry slots: keeps atomic_add for correctness:
      delta = k_val * delta_v * post_decay + is_last*(snapshot*(cumul-1))
      atomic_add(memory[slot], delta)

    Grid: (CW,)
    """
    pid = tl.program_id(0)
    t_local = pid // W
    n = pid % W

    # Load slot index
    slot = tl.load(flat_k_ptr + pid).to(tl.int64)

    # Load write scalars — upcast to f32 for precision
    kv = tl.load(k_val_ptr + t_local * W + n).to(tl.float32)
    pd = tl.load(post_decay_ptr + t_local * W + n).to(tl.float32)
    write_scale = kv * pd

    # Load decay scalars
    cumul_val = tl.load(cumul_ptr + pid).to(tl.float32)
    decay_correction = cumul_val - 1.0

    is_last = tl.load(is_last_ptr + pid)
    is_only = tl.load(is_only_ptr + pid)

    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < dim

        # Write contribution: k_val * delta_v * post_decay (f32)
        dv = tl.load(delta_v_ptr + t_local * dim + d_offs, mask=mask, other=0.0).to(tl.float32)
        write_delta = write_scale * dv

        # Snapshot value (always loaded for both paths)
        snap = tl.load(snapshot_ptr + pid * dim + d_offs, mask=mask, other=0.0).to(tl.float32)

        if is_only != 0:
            # Plain store: snapshot * cumul + write_delta
            # Eliminates redundant HBM read from atomic RMW.
            new_val = snap * cumul_val + write_delta
            tl.store(memory_ptr + slot * dim + d_offs,
                     new_val.to(memory_ptr.dtype.element_ty), mask=mask)
        else:
            # Multi-entry slot: keep atomic for correctness
            delta = write_delta
            delta += tl.where(is_last != 0, snap * decay_correction, 0.0)
            tl.atomic_add(memory_ptr + slot * dim + d_offs, delta, mask=mask, sem="relaxed")

def fused_decay_and_write(
    memory: torch.Tensor,       # [num_slots, dim] — modified in-place
    snapshot: torch.Tensor,     # [C, W, dim] contiguous snapshot (pre-decay values)
    flat_k: torch.Tensor,       # [C*W] slot indices
    cumul_per_entry: torch.Tensor,  # [C*W] per-entry cumulative decay
    is_last: torch.Tensor,     # [C*W] int32 mask (1=last entry for slot)
    k_val: torch.Tensor,       # [C, W] key values (any float dtype)
    delta_v: torch.Tensor,     # [C, dim] delta values (any float dtype)
    post_decay: torch.Tensor,  # [C, W] post-decay factors (any float dtype)
    is_only: torch.Tensor | None = None,  # [C*W] int32 mask (1=sole entry for slot)
):
    """Fused decay + scatter write. Replaces scatter_decay_from_snapshot + fused_scatter_write.

    Uses additive formulation: each entry atomically adds both its write contribution
    and (if it's the last entry for its slot) the decay correction.
    For is_only entries, uses plain store instead of atomic_add (no RMW overhead).
    Accepts any float dtype for k_val, delta_v, post_decay — kernel computes in f32.
    """
    C_local, W = k_val.shape
    CW = C_local * W
    dim = delta_v.shape[-1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    if is_only is None:
        is_only = torch.zeros(CW, device=memory.device, dtype=torch.int32)
    _fused_decay_and_write_kernel[(CW,)](
        memory,
        snapshot.reshape(-1, dim),
        flat_k.contiguous().to(torch.int64),
        cumul_per_entry.contiguous(),
        is_last.to(torch.int32) if is_last.dtype != torch.int32 else is_last,
        is_only.to(torch.int32) if is_only.dtype != torch.int32 else is_only,
        k_val.contiguous(),
        delta_v.contiguous(),
        post_decay.contiguous(),
        CW=CW, C_local=C_local, W=W, dim=dim, BLOCK_D=BLOCK_D,
        num_warps=1,
    )

def sparse_inner_product(
    idx_row: torch.Tensor,   # [num_chunks, C, W_ROW]
    val_row: torch.Tensor,   # [num_chunks, C, W_ROW]
    idx_col: torch.Tensor,   # [num_chunks, C, W_COL]
    val_col: torch.Tensor,   # [num_chunks, C, W_COL]
) -> torch.Tensor:
    """Compute batched sparse inner product using Triton kernel.

    Returns [num_chunks, C, C] where out[ci, i, j] = sum over shared slots
    of val_row[ci, i, :] * val_col[ci, j, :].

    No autograd — use SparseInnerProductAutograd for differentiable version.
    """
    num_chunks, C, W_ROW = idx_row.shape
    W_COL = idx_col.shape[2]
    device = idx_row.device

    out = torch.empty(num_chunks, C, C, device=device, dtype=torch.float32)

    _sparse_inner_product_kernel[(num_chunks * C,)](
        idx_row.contiguous().to(torch.int64),
        val_row.contiguous().to(torch.float32),
        idx_col.contiguous().to(torch.int64),
        val_col.contiguous().to(torch.float32),
        out,
        C=C, W_ROW=W_ROW, W_COL=W_COL,
    )

    return out

def _segmented_cumsum_batched(
    values: torch.Tensor,    # [B, N] values to cumsum
    slots: torch.Tensor,     # [B, N] slot indices (group keys)
    times: torch.Tensor,     # [N] or [B, N] time indices (broadcastable)
    max_time: int | None = None,  # Upper bound on (times * 2).long().max(); avoids GPU->CPU sync
) -> torch.Tensor:
    """Batched per-slot cumulative sum. Each batch element is independent.

    Same semantics as _segmented_cumsum but operates on [B, N] inputs,
    processing each batch element in parallel. Float32 safe since each
    batch element has at most ~16K entries (cumsum magnitude < 8K).

    No chunk_slot_offset needed — batch elements are naturally separated.

    Args:
        max_time: Known upper bound on max(times * 2). When provided, avoids a
            GPU->CPU synchronization (.item()) to compute it from the tensor.
            Callers should pass 2*(C-1) for integer times or 2*C-1 for mixed
            integer/half-integer times.
    """
    B, N = values.shape
    device = values.device

    # Broadcast times to [B, N] if needed
    if times.dim() == 1:
        times = times.unsqueeze(0).expand(B, -1)

    # Sort by (slot, time) within each batch element
    # Multiply times by 2 to preserve half-integer ordering (e.g. 0, 0.5, 1, 1.5 -> 0, 1, 2, 3)
    times_2x = (times * 2).long()
    if max_time is None:
        max_time = int(times_2x.max().item())
    sort_key = slots.long() * (max_time + 1) + times_2x
    sorted_order = sort_key.argsort(dim=-1)  # [B, N]

    sorted_vals = values.gather(-1, sorted_order)
    sorted_slots = slots.gather(-1, sorted_order)
    sorted_times = times.gather(-1, sorted_order)

    # Per-batch cumsum (float32 safe at N ≤ 16K)
    global_cs = sorted_vals.cumsum(dim=-1)

    # Boundary detection: slot changes between adjacent sorted entries
    boundary = torch.ones(B, N, device=device, dtype=torch.bool)
    boundary[:, 1:] = sorted_slots[:, 1:] != sorted_slots[:, :-1]

    # Offset propagation via cummax trick
    boundary_offsets = torch.where(boundary, global_cs - sorted_vals,
                                   torch.zeros_like(global_cs))
    arange = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    boundary_idx = torch.where(boundary, arange,
                               torch.zeros(B, N, device=device, dtype=torch.long))
    boundary_idx_filled, _ = boundary_idx.cummax(dim=-1)
    offsets = boundary_offsets.gather(-1, boundary_idx_filled)

    seg_cs = global_cs - offsets

    # Same-(slot, time) merge: entries at identical (slot, time) get the max cumsum
    # Always run the merge — removing the `if same_st.any()` guard avoids a
    # GPU->CPU sync. The logic is a no-op when there are no duplicates.
    same_st = torch.zeros(B, N, device=device, dtype=torch.bool)
    same_st[:, :-1] = ((sorted_slots[:, :-1] == sorted_slots[:, 1:]) &
                        (sorted_times[:, :-1] == sorted_times[:, 1:]))

    run_end_marker = ~same_st
    run_end_idx = torch.where(run_end_marker, arange,
                              torch.full((B, N), N, device=device, dtype=torch.long))
    run_end_filled, _ = run_end_idx.flip(-1).cummin(dim=-1)
    run_end_filled = run_end_filled.flip(-1)
    seg_cs = seg_cs.gather(-1, run_end_filled.clamp(max=N - 1))

    # Scatter back to original order
    result = torch.empty_like(values)
    result.scatter_(-1, sorted_order, seg_cs)
    return result

def _pad_inputs(k_idx, k_val, v, beta, g, q_idx, q_val, pad):
    """Pad inputs along the time dimension for chunk alignment."""
    if pad > 0:
        return (F.pad(k_idx, (0, 0, 0, pad)), F.pad(k_val, (0, 0, 0, pad)),
                F.pad(v, (0, 0, 0, pad)), F.pad(beta, (0, 0, 0, pad)),
                F.pad(g, (0, 0, 0, pad)), F.pad(q_idx, (0, 0, 0, pad)),
                F.pad(q_val, (0, 0, 0, pad)))
    return k_idx, k_val, v, beta, g, q_idx, q_val

def _sparse_ip_gated_bwd_vals(row_idx, row_val, row_lc, col_idx, col_val, col_lc,
                               dout, C, causal_mode):
    """Backward of sparse_inner_product_gated w.r.t. row_val and col_val.

    Uses CUDA two-pointer kernel. Indices must be pre-sorted along W.
    Supports int32 and int64 idx, all val/lc/dout float32.

    Returns (d_row_val, d_col_val) in float32.
    """
    from .cuda.sparse_ip_cuda import sparse_ip_gated_bwd_sorted
    return sparse_ip_gated_bwd_sorted(
        row_idx.contiguous(), row_val.contiguous(), row_lc.contiguous(),
        col_idx.contiguous(), col_val.contiguous(), col_lc.contiguous(),
        dout.contiguous(), causal_mode)

@triton.jit
def _fused_gather_dot_and_scatter_kernel(
    memory_ptr,         # [num_slots, dim] — table to gather from (READ)
    grad_memory_ptr,    # [num_slots, dim] — table to scatter into (WRITE)
    q_idx_ptr,          # [NC*R] — slot indices
    grad_out_ptr,       # [NC, dim] — vectors to dot with and scatter
    q_weights_ptr,      # [NC*R] — per-entry scalar weights for scatter (f32)
    dot_out_ptr,        # [NC*R] — output dot products (f32)
    NC: tl.constexpr,
    R: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: gather_dot + outer_scatter_add in one pass over q indices.

    For each (t, r):
      1. dot_out[t,r] = sum_d grad_out[t,d] * memory[q_idx[t,r], d]   (READ memory)
      2. grad_memory[q_idx[t,r]] += q_weights[t,r] * grad_out[t,:]     (WRITE grad_memory)

    Shares the grad_out[t,:] load and q_idx lookup between both operations.
    Grid: (NC*R,) — one program per (token, read_entry).
    """
    i = tl.program_id(0)
    t = i // R

    slot = tl.load(q_idx_ptr + i).to(tl.int64)
    w = tl.load(q_weights_ptr + i).to(tl.float32)

    dot = tl.zeros([1], dtype=tl.float32)
    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim

        # Load grad_out (shared by both ops)
        grad_val = tl.load(grad_out_ptr + t * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)

        # Op 1: dot product with memory[slot]
        mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0)
        dot += tl.sum(mem_val.to(tl.float32) * grad_val)

        # Op 2: scatter q_weights * grad_out to grad_memory[slot]
        scatter_val = (w * grad_val).to(grad_memory_ptr.dtype.element_ty)
        tl.atomic_add(grad_memory_ptr + slot * dim + d_offs, scatter_val, mask=d_mask, sem="relaxed")

    tl.store(dot_out_ptr + i, tl.sum(dot))

def fused_gather_dot_and_scatter(
    memory: torch.Tensor,       # [num_slots, dim] — read
    grad_memory: torch.Tensor,  # [num_slots, dim] — write (atomic_add)
    q_idx: torch.Tensor,        # [NC, R] — slot indices
    grad_out: torch.Tensor,     # [NC, dim] — f32
    q_weights: torch.Tensor,    # [NC, R] — f32
    NC: int, R: int, dim: int,
) -> torch.Tensor:
    """Fused gather_dot + outer_scatter_add. Returns [NC, R] f32 dot products."""
    dot_out = torch.empty(NC, R, device=memory.device, dtype=torch.float32)
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    _fused_gather_dot_and_scatter_kernel[(NC * R,)](
        memory, grad_memory,
        q_idx.contiguous().reshape(-1).to(torch.int64),
        grad_out.contiguous(),
        q_weights.contiguous().reshape(-1),
        dot_out,
        NC=NC, R=R, dim=dim, BLOCK_D=BLOCK_D,
        # Adaptive num_warps: with many programs (NC≥1024), fewer warps = more
        # blocks per SM = more distinct random memory addresses in flight.
        # Same policy as the GR kernel (_fused_gather_weighted_reduce_kernel).
        num_warps=1 if NC >= 1024 else 4,
    )
    return dot_out

@torch.no_grad()
def gated_sparse_memory_write_read_inference(
    memory: torch.Tensor,  # [num_slots, dim] — modified in-place
    k_idx: torch.Tensor,   # [B*T, num_writes]
    k_val: torch.Tensor,   # [B*T, num_writes]
    v: torch.Tensor,        # [B*T, dim]
    beta: torch.Tensor,     # [B*T, 1]
    g: torch.Tensor,        # [B*T, 1]
    q_idx: torch.Tensor,   # [B*T, num_reads]
    q_val: torch.Tensor,   # [B*T, num_reads]
    chunk_size: int = 128,
    num_memory_slots: int = 0,
) -> torch.Tensor:
    """Inference-only forward. Same computation as training but:
    - No all_mem_before_decay snapshot (saves B*T*W*dim memory)
    - No all_delta_v buffer
    - No M_sys_inv_T_all (only needed for backward)
    - No ctx/wy_data saving
    """
    B_T, num_writes = k_idx.shape
    num_slots, dim = memory.shape
    num_reads = q_idx.shape[1]
    device = memory.device
    dtype = memory.dtype

    readings = torch.empty(B_T, dim, device=device, dtype=dtype)

    num_chunks = (B_T + chunk_size - 1) // chunk_size
    C = chunk_size

    pad = num_chunks * C - B_T
    k_idx_p, k_val_p, v_p, beta_p, g_p, q_idx_p, q_val_p = _pad_inputs(
        k_idx, k_val, v, beta, g, q_idx, q_val, pad)

    k_idx_ch = k_idx_p.reshape(num_chunks, C, num_writes)
    fwd_compute_dtype = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype
    k_val_ch = k_val_p.reshape(num_chunks, C, num_writes).to(fwd_compute_dtype)
    v_ch = v_p.reshape(num_chunks, C, dim).to(fwd_compute_dtype)
    beta_ch = beta_p.reshape(num_chunks, C, 1).to(fwd_compute_dtype)
    g_ch = g_p.reshape(num_chunks, C, -1).to(fwd_compute_dtype)  # [N, C, 1] or [N, C, W]
    q_idx_ch = q_idx_p.reshape(num_chunks, C, num_reads)
    q_val_ch = q_val_p.reshape(num_chunks, C, num_reads).to(fwd_compute_dtype)

    # Phase 1: batched WY quantities
    flat_k = k_idx_ch.reshape(num_chunks, C * num_writes)
    flat_g = g_ch.expand(num_chunks, C, num_writes).reshape(num_chunks, C * num_writes)
    flat_time = torch.arange(C, device=device).unsqueeze(1).expand(-1, num_writes).reshape(-1)
    log_cumul_k_all = _segmented_cumsum_batched(flat_g, flat_k, flat_time, max_time=2 * (C - 1)).reshape(num_chunks, C, num_writes)
    cumul_k_all = torch.exp(log_cumul_k_all)

    flat_q = q_idx_ch.reshape(num_chunks, C * num_reads)
    flat_q_time = torch.arange(C, device=device).unsqueeze(1).expand(-1, num_reads).reshape(-1).float() + 0.5
    flat_k_time_for_q = torch.arange(C, device=device).unsqueeze(1).expand(-1, num_writes).reshape(-1).float()
    all_slots_q = torch.cat([flat_k, flat_q], dim=-1)
    all_g_q = F.pad(flat_g, (0, C * num_reads))  # [N, C*(W+R)]
    all_time_q = torch.cat([flat_k_time_for_q, flat_q_time])
    all_cumul_q = _segmented_cumsum_batched(all_g_q, all_slots_q, all_time_q, max_time=2 * C - 1)
    log_cumul_q_all = all_cumul_q[:, C * num_writes:].reshape(num_chunks, C, num_reads)
    cumul_q_all = torch.exp(log_cumul_q_all)

    A_all = sparse_inner_product_gated(
        k_idx_ch, k_val_ch, log_cumul_k_all,
        k_idx_ch, k_val_ch, log_cumul_k_all,
    ).to(fwd_compute_dtype)
    QK_all = sparse_inner_product_gated(
        q_idx_ch, q_val_ch, log_cumul_q_all,
        k_idx_ch, k_val_ch, log_cumul_k_all,
        include_diag=True,
    ).to(fwd_compute_dtype)

    # Fused M_sys build (I + beta * A * tril_mask)
    M_sys_all = fused_build_M_sys(beta_ch, A_all)

    rev_time = (C - 1) - flat_time
    rev_log_cumul_k_all = _segmented_cumsum_batched(flat_g, flat_k, rev_time, max_time=2 * (C - 1)).reshape(num_chunks, C, num_writes)
    post_decay_all = torch.exp(rev_log_cumul_k_all - g_ch.expand(num_chunks, C, num_writes))

    # Only 2 solves needed for inference (no M_sys_inv_T)
    # Consolidate: 1 solve → M_sys_inv, then derive delta_v_const and B_matrix
    eye_inf = torch.eye(C, device=device, dtype=fwd_compute_dtype).unsqueeze(0).expand(num_chunks, -1, -1)
    M_sys_inv_all = torch.linalg.solve_triangular(
        M_sys_all, eye_inf, upper=False)
    delta_v_const_all = torch.bmm(M_sys_inv_all, beta_ch * v_ch)
    B_matrix_all = M_sys_inv_all * (-beta_ch).transpose(-1, -2)
    K_eff_all = k_val_ch * cumul_k_all

    # Phase 2 pre-computation (fused dedup + decay)
    decay_per_all = fused_dedup_decay(k_idx_ch, g_ch)
    flat_k_unsq_all = flat_k.unsqueeze(2)
    K_eff_casted_all = K_eff_all.to(dtype)
    q_weights_all = (q_val_ch * cumul_q_all).to(dtype)
    QK_tril_all = torch.tril(QK_all)  # [N, C, C]
    total_cumul = torch.empty(num_slots, 1, device=device, dtype=fwd_compute_dtype)
    retrieved_buf = torch.empty(C, dim, device=device, dtype=fwd_compute_dtype)
    inter_buf = torch.empty(C, dim, device=device, dtype=fwd_compute_dtype)

    # Pre-compute unique slots per chunk via batched sort+diff (~0.2ms total)
    sorted_k, _ = flat_k.sort(dim=-1)  # [N, C*W]
    unique_mask = torch.ones_like(sorted_k, dtype=torch.bool)
    unique_mask[:, 1:] = sorted_k[:, 1:] - sorted_k[:, :-1] > 0
    unique_counts = unique_mask.sum(dim=-1)  # [N]
    # Compact into list of per-chunk unique index tensors
    unique_k_padded = sorted_k[unique_mask].split(unique_counts.tolist())

    # Phase 2: sequential loop — no backward saves, no snapshot needed
    for ci in range(num_chunks):
        chunk_start = ci * C
        chunk_end = min(chunk_start + C, B_T)
        actual_C = chunk_end - chunk_start
        sl = slice(chunk_start, chunk_end)

        total_cumul.fill_(1.0)
        total_cumul.scatter_reduce_(
            0, flat_k_unsq_all[ci], decay_per_all[ci],
            reduce='prod', include_self=True,
        )

        # Inference: fused dual gather (retrieved + inter), no snapshot
        fused_dual_gather(
            memory.data, k_idx_ch[ci], K_eff_casted_all[ci],
            q_idx_ch[ci], q_weights_all[ci],
            retrieved_buf, inter_buf,
        )

        delta_v = delta_v_const_all[ci] + B_matrix_all[ci] @ retrieved_buf

        # Decay at unique slots (Triton kernel, no redundant gather)
        decay_at_unique_slots(memory.data, unique_k_padded[ci], total_cumul)

        fused_scatter_write(
            memory.data, k_idx_ch[ci],
            k_val_ch[ci], delta_v, post_decay_all[ci],
        )

        # Fused: causal_intra = mm(QK, delta_v), then out = (inter + causal_intra).bf16
        causal_intra = torch.mm(QK_tril_all[ci], delta_v)
        fused_add_cast(inter_buf, causal_intra, readings[sl], actual_C)

    return readings


@torch.no_grad()
def gated_sparse_memory_write_read_inference_v2(
    memory: torch.Tensor,  # [num_slots, dim] — modified in-place
    k_idx: torch.Tensor,   # [B*T, num_writes] or [P, T, num_writes]
    k_val: torch.Tensor,   # [B*T, num_writes] or [P, T, num_writes]
    v: torch.Tensor,        # [B*T, dim] or [P, T, dim]
    beta: torch.Tensor,     # [B*T, 1] or [P, T, 1]
    g: torch.Tensor,        # [B*T, 1] or [P, T, 1]
    q_idx: torch.Tensor,   # [B*T, num_reads] or [P, T, num_reads]
    q_val: torch.Tensor,   # [B*T, num_reads] or [P, T, num_reads]
    chunk_size: int = 128,
    num_memory_slots: int = 0,
    num_parallel: int = 1,
) -> torch.Tensor:
    """Inference-only forward with WY-kernel Phase 2 optimizations.

    Combines the best of both worlds:
    - Batched Phase 1 (fused triple cumsum, sparse inner products, batched solve)
    - WY-kernel Phase 2 loop with fused dual_matmul (delta_v + readings in one
      kernel) and fused_decay_and_write (is_only fast-path: ~93% plain stores)
    - Reusable per-chunk snapshot buffer (~20 MB) instead of saving full
      all_mem_before_decay for backward (~24 GB at 1M tokens)
    - No save_for_backward, no wy_data, no delta_v_all/retrieved_all saves

    Memory savings vs WY training kernel: ~26 GB → ~20 MB at L08 1M prefill.
    Speed improvement vs old inference kernel: fused dual_matmul eliminates
    3 kernel launches per chunk, is_only fast-path avoids atomics for ~93% of
    memory updates.
    """
    # Handle 3D inputs [P, T, ...] from num_parallel > 1
    input_was_3d = k_idx.ndim == 3
    if input_was_3d:
        T_seq = k_idx.shape[1]
        k_idx = k_idx.reshape(num_parallel * T_seq, -1)
        k_val = k_val.reshape(num_parallel * T_seq, -1)
        v = v.reshape(num_parallel * T_seq, -1)
        beta = beta.reshape(num_parallel * T_seq, -1)
        g = g.reshape(num_parallel * T_seq, -1)
        q_idx = q_idx.reshape(num_parallel * T_seq, -1)
        q_val = q_val.reshape(num_parallel * T_seq, -1)

    B_T, num_writes = k_idx.shape
    num_slots, dim = memory.shape
    num_reads = q_idx.shape[1]
    device = memory.device
    dtype = memory.dtype

    # Compute slots_per_head for cumsum accumulator optimization (same as training forward)
    if num_memory_slots > 0:
        sph = num_memory_slots
    else:
        sph = num_slots
    BH = num_slots // sph if sph > 0 else 1

    # === Phase 1: Pre-compute all WY quantities batched (no memory dependency) ===
    num_chunks = (B_T + chunk_size - 1) // chunk_size
    C = chunk_size

    pad = num_chunks * C - B_T
    k_idx_p, k_val_p, v_p, beta_p, g_p, q_idx_p, q_val_p = _pad_inputs(
        k_idx, k_val, v, beta, g, q_idx, q_val, pad)

    k_idx_ch = k_idx_p.reshape(num_chunks, C, num_writes)
    fwd_compute_dtype = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype
    k_val_ch = k_val_p.reshape(num_chunks, C, num_writes).to(fwd_compute_dtype)
    v_ch = v_p.reshape(num_chunks, C, dim).to(fwd_compute_dtype)
    beta_ch = beta_p.reshape(num_chunks, C, 1).to(fwd_compute_dtype)
    g_ch = g_p.reshape(num_chunks, C, -1).to(fwd_compute_dtype)
    q_idx_ch = q_idx_p.reshape(num_chunks, C, num_reads)
    q_val_ch = q_val_p.reshape(num_chunks, C, num_reads).to(fwd_compute_dtype)

    # Fused triple cumsum: forward K + cross Q + reverse K in ONE kernel
    flat_k = k_idx_ch.reshape(num_chunks, C * num_writes)
    flat_g = g_ch.expand(num_chunks, C, num_writes).reshape(num_chunks, C * num_writes)
    flat_q = q_idx_ch.reshape(num_chunks, C * num_reads)

    # Strip batch offsets from indices for cumsum (same as training forward)
    if BH > 1:
        chunks_per_batch = num_chunks // BH
        batch_offsets = (torch.arange(num_chunks, device=device) // chunks_per_batch * sph).unsqueeze(1)
        flat_k_local = flat_k - batch_offsets
        flat_q_local = flat_q - batch_offsets
    else:
        flat_k_local = flat_k
        flat_q_local = flat_q

    log_cumul_k_flat, log_cumul_q_flat, rev_log_cumul_k_flat = _fused_triple_cumsum(
        flat_g, flat_k_local, flat_q_local, C, num_writes, num_reads, sph,
    )
    log_cumul_k_all = log_cumul_k_flat.reshape(num_chunks, C, num_writes)
    cumul_k_all = torch.exp(log_cumul_k_all)
    log_cumul_q_all = log_cumul_q_flat.reshape(num_chunks, C, num_reads)
    cumul_q_all = torch.exp(log_cumul_q_all)
    rev_log_cumul_k_all = rev_log_cumul_k_flat.reshape(num_chunks, C, num_writes)
    post_decay_all = torch.exp(rev_log_cumul_k_all - g_ch.expand(num_chunks, C, num_writes))

    A_all = sparse_inner_product_gated(
        k_idx_ch, k_val_ch, log_cumul_k_all,
        k_idx_ch, k_val_ch, log_cumul_k_all,
    ).to(fwd_compute_dtype)
    QK_all = sparse_inner_product_gated(
        q_idx_ch, q_val_ch, log_cumul_q_all,
        k_idx_ch, k_val_ch, log_cumul_k_all,
        include_diag=True,
    ).to(fwd_compute_dtype)

    M_sys_all = fused_build_M_sys(beta_ch, A_all)

    eye = torch.eye(C, device=device, dtype=fwd_compute_dtype).unsqueeze(0).expand(num_chunks, -1, -1)
    M_sys_inv_all = torch.linalg.solve_triangular(M_sys_all, eye, upper=False)

    delta_v_const_all = torch.bmm(M_sys_inv_all, beta_ch * v_ch)
    B_matrix_all = M_sys_inv_all * (-beta_ch).transpose(-1, -2)
    K_eff_all = k_val_ch * cumul_k_all

    # === Phase 2 pre-computation ===
    decay_per_all = fused_dedup_decay(k_idx_ch, g_ch)
    K_eff_casted_all = K_eff_all.to(dtype)
    q_weights_all = (q_val_ch * cumul_q_all).to(dtype)
    QK_tril_all = torch.tril(QK_all)

    # Pre-compute is_last/is_only/cumul_per_entry via sort-based approach
    CW = C * num_writes
    chunk_offsets = torch.arange(num_chunks, device=device, dtype=torch.int32).unsqueeze(1) * num_slots
    sort_keys = (flat_k.to(torch.int32) + chunk_offsets).reshape(-1)
    sorted_keys, sort_perm = sort_keys.sort()

    seg_start = torch.ones(num_chunks * CW, device=device, dtype=torch.int32)
    seg_start[1:] = (sorted_keys[1:] != sorted_keys[:-1]).to(torch.int32)

    is_last_sorted = torch.ones(num_chunks * CW, device=device, dtype=torch.int32)
    is_last_sorted[:-1] = seg_start[1:]
    is_only_sorted = seg_start & is_last_sorted

    seg_id = seg_start.long().cumsum(0) - 1
    n_segs = num_chunks * CW
    decay_sorted = decay_per_all.reshape(-1)[sort_perm]
    seg_product = torch.ones(n_segs, device=device, dtype=fwd_compute_dtype)
    seg_product.scatter_reduce_(0, seg_id, decay_sorted, reduce='prod', include_self=True)
    cumul_sorted = seg_product[seg_id]

    unsort_perm = torch.empty_like(sort_perm)
    unsort_perm[sort_perm] = torch.arange(num_chunks * CW, device=device)

    cumul_per_entry_all = cumul_sorted[unsort_perm].reshape(num_chunks, CW).to(dtype)
    is_last_all = is_last_sorted[unsort_perm].reshape(num_chunks, CW)
    is_only_all = is_only_sorted[unsort_perm].reshape(num_chunks, CW)
    del sort_keys, sorted_keys, sort_perm, seg_start, is_last_sorted, is_only_sorted
    del seg_id, decay_sorted, seg_product, cumul_sorted, unsort_perm

    # Determine parallel batch dimensions
    P = num_parallel if (num_parallel > 1 and pad == 0 and num_chunks % num_parallel == 0) else 1
    Nt = num_chunks // P
    SC = P * C

    inter_buf = torch.empty(SC, dim, device=device, dtype=fwd_compute_dtype)

    # Permute to time-major for P>1
    def _to_time_major(x):
        if P == 1:
            return x
        shape = x.shape
        return x.view(P, Nt, *shape[1:]).permute(1, 0, *range(2, len(shape) + 1)).contiguous().view(shape)

    k_idx_tm = _to_time_major(k_idx_ch)
    K_eff_tm = _to_time_major(K_eff_casted_all)
    q_idx_tm = _to_time_major(q_idx_ch)
    q_weights_tm = _to_time_major(q_weights_all)
    delta_v_const_tm = _to_time_major(delta_v_const_all)
    B_matrix_tm = _to_time_major(B_matrix_all)
    QK_tril_tm = _to_time_major(QK_tril_all)
    flat_k_tm = _to_time_major(flat_k)
    cumul_per_entry_tm = _to_time_major(cumul_per_entry_all)
    is_last_tm = _to_time_major(is_last_all)
    is_only_tm = _to_time_major(is_only_all)
    k_val_tm = _to_time_major(k_val_ch)
    post_decay_tm = _to_time_major(post_decay_all)

    # Pre-compute combined matrices for dual-output matmul
    QB_tm = torch.bmm(QK_tril_tm, B_matrix_tm)
    qk_dvc_tm = torch.bmm(QK_tril_tm, delta_v_const_tm)

    # Reusable per-chunk buffers (~20 MB snapshot vs ~24 GB full save)
    snapshot_chunk_buf = torch.empty(SC, num_writes, dim, device=device, dtype=dtype)
    retrieved_buf = torch.empty(SC, dim, device=device, dtype=fwd_compute_dtype)
    delta_v_buf = torch.empty(P, C, dim, device=device, dtype=fwd_compute_dtype)
    readings_tm = torch.empty(num_chunks * C, dim, device=device, dtype=dtype)

    # Pre-flatten loop metadata
    _fwd_flat_k = flat_k_tm.reshape(-1)
    _fwd_cumul = cumul_per_entry_tm.reshape(-1)
    _fwd_islast = is_last_tm.reshape(-1)
    _fwd_isonly = is_only_tm.reshape(-1)
    _fwd_kval = k_val_tm.reshape(-1, num_writes)
    _fwd_pd = post_decay_tm.reshape(-1, num_writes)

    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    _dm_grid = (triton.cdiv(C, 64), triton.cdiv(dim, 64), P)
    _gather_bt = 128 if SC <= 1024 else 256

    if dtype == torch.bfloat16:
        try:
            from .cuda.warp_cooperative_gather_cuda import warp_cooperative_gather as _wc_gather
        except ImportError:
            _wc_gather = None
    else:
        _wc_gather = None

    # === Phase 2: Loop with reusable snapshot buffer ===
    for ti in range(Nt):
        tm_start = ti * P
        tm_end = tm_start + P
        tok_start = ti * SC
        tok_end = tok_start + SC
        cw_s = ti * SC * num_writes
        cw_e = cw_s + SC * num_writes
        sc_s = ti * SC
        sc_e = sc_s + SC

        # Snapshot gather into reusable buffer + retrieved + inter
        if _wc_gather is not None:
            _wc_gather(memory.data,
                       k_idx_tm[tm_start:tm_end], K_eff_tm[tm_start:tm_end],
                       q_idx_tm[tm_start:tm_end], q_weights_tm[tm_start:tm_end],
                       snapshot_chunk_buf, retrieved_buf, inter_buf,
                       block_threads=_gather_bt)
        else:
            fused_snapshot_gather(memory.data,
                k_idx_tm[tm_start:tm_end].reshape(SC, num_writes),
                K_eff_tm[tm_start:tm_end].reshape(SC, num_writes),
                q_idx_tm[tm_start:tm_end].reshape(SC, num_reads),
                q_weights_tm[tm_start:tm_end].reshape(SC, num_reads),
                snapshot_chunk_buf, retrieved_buf, inter_buf)

        # Dual matmul: delta_v + readings in one kernel
        _fused_dual_matmul_add_kernel[_dm_grid](
            B_matrix_tm[tm_start:tm_end], QB_tm[tm_start:tm_end],
            retrieved_buf.view(P, C, dim),
            delta_v_const_tm[tm_start:tm_end],
            inter_buf.view(P, C, dim),
            qk_dvc_tm[tm_start:tm_end],
            delta_v_buf,
            readings_tm[tok_start:tok_end].view(P, C, dim),
            stride_ap=C * C, stride_bp=C * dim, stride_cp=C * dim,
            M=C, K=C, N=dim,
            BLOCK_M=64, BLOCK_K=min(64, C), BLOCK_N=64,
        )

        # Fused decay + scatter write (is_only fast-path, reusable snapshot)
        _fused_decay_and_write_kernel[(SC * num_writes,)](
            memory.data,
            snapshot_chunk_buf.reshape(-1, dim),
            _fwd_flat_k[cw_s:cw_e],
            _fwd_cumul[cw_s:cw_e],
            _fwd_islast[cw_s:cw_e],
            _fwd_isonly[cw_s:cw_e],
            _fwd_kval[sc_s:sc_e],
            delta_v_buf.reshape(SC, dim),
            _fwd_pd[sc_s:sc_e],
            CW=SC * num_writes, C_local=SC, W=num_writes, dim=dim, BLOCK_D=BLOCK_D,
            num_warps=1,
        )

    # Trim padding and permute back to batch-first
    if P > 1:
        def _to_batch_first(x):
            shape = x.shape
            return x.view(Nt, P, *shape[1:]).permute(1, 0, *range(2, len(shape) + 1)).contiguous().view(shape)
        readings = _to_batch_first(readings_tm.view(num_chunks, C, dim)).view(B_T + pad, dim)[:B_T]
    else:
        readings = readings_tm[:B_T]

    return readings


# ============================================================
# Fast Triton embedding_bag kernels
# Adapted from "Memory Layers at Scale" (Meta, 2024).
# Forward achieves near-peak memory bandwidth (~3 TB/s on H100).
# Backward uses reverse_indices strategy (atomic-free, 6-19x faster
# than PyTorch's index_add_ approach).
# ============================================================

@triton.jit
def _embedding_bag_fwd_kernel(
    out_ptr,              # [B, dim]
    indices_ptr,          # [B, bag_size]
    weight_ptr,           # [K, dim]
    per_sample_weights,   # [B, bag_size]
    dim: tl.constexpr,
    bag_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward: weighted sum of embeddings. One program per batch element.
    Handles arbitrary dim via BLOCK_D loop."""
    out_idx = tl.program_id(axis=0).to(tl.int64)
    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim
        out_value = tl.zeros([BLOCK_D], dtype=tl.float32)
        for bag in range(0, bag_size):
            my_index = tl.load(indices_ptr + out_idx * bag_size + bag).to(tl.int64)
            my_scaling = tl.load(per_sample_weights + out_idx * bag_size + bag)
            my_weight = tl.load(weight_ptr + my_index * dim + d_offs, mask=d_mask, other=0.0)
            out_value = out_value + my_weight.to(tl.float32) * my_scaling
        tl.store(out_ptr + out_idx * dim + d_offs, out_value, mask=d_mask)

def embedding_bag_fwd_triton(
    indices: torch.Tensor, weight: torch.Tensor, per_sample_weights: torch.Tensor
) -> torch.Tensor:
    """Triton forward for embedding_bag (sum mode). Handles any dim."""
    B, bag_size = indices.shape
    K, dim = weight.shape
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    out = torch.empty([B, dim], dtype=weight.dtype, device=weight.device)
    _embedding_bag_fwd_kernel[(B,)](
        out, indices, weight, per_sample_weights,
        dim=dim, bag_size=bag_size, BLOCK_D=BLOCK_D,
        num_warps=1, num_stages=1,
    )
    return out

@triton.jit
def _embedding_bag_bwd_atomics_kernel(
    weight_grad_ptr,              # [K, dim]
    per_sample_weights_grad_ptr,  # [B, bag_size]
    indices_ptr,                  # [B, bag_size]
    weight_ptr,                   # [K, dim]
    per_sample_weights_ptr,       # [B, bag_size]
    gradient_ptr,                 # [B, dim]
    dim: tl.constexpr,
    bag_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward with atomics. One program per batch element. Handles any dim."""
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for bag in range(bag_size):
        my_index = tl.load(indices_ptr + batch_id * bag_size + bag).to(tl.int64)
        my_scaling = tl.load(per_sample_weights_ptr + batch_id * bag_size + bag)
        # Accumulate psw grad as a BLOCK_D vector, sum at the end
        psw_grad_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        for d_start in range(0, dim, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < dim
            gradient = tl.load(gradient_ptr + batch_id * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            my_weight = tl.load(weight_ptr + my_index * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            weight_grad = gradient * my_scaling
            tl.atomic_add(weight_grad_ptr + my_index * dim + d_offs, weight_grad, mask=d_mask, sem="relaxed")
            psw_grad_vec += gradient * my_weight
        tl.store(per_sample_weights_grad_ptr + batch_id * bag_size + bag, tl.sum(psw_grad_vec))

def embedding_bag_bwd_atomics(
    indices: torch.Tensor,
    weight: torch.Tensor,
    per_sample_weights: torch.Tensor,
    gradient: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Atomics-based backward for embedding_bag."""
    B, bag_size = indices.shape
    K, dim = weight.shape
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    weight_grad = torch.zeros_like(weight)
    per_sample_weights_grad = torch.empty_like(per_sample_weights)
    _embedding_bag_bwd_atomics_kernel[(B,)](
        weight_grad, per_sample_weights_grad,
        indices, weight, per_sample_weights, gradient,
        dim=dim, bag_size=bag_size, BLOCK_D=BLOCK_D,
        num_warps=1,
    )
    return weight_grad, per_sample_weights_grad

class FastEmbeddingBag(torch.autograd.Function):
    """Fast embedding_bag using Triton kernels from Memory Layers at Scale.

    Forward: Triton kernel achieving near-peak memory bandwidth.
    Backward: atomics strategy (7x faster than index_add_). For power-of-2
    dims, reverse_indices is even faster (13x) but requires dim adaptation.
    """

    @staticmethod
    def forward(ctx, indices, weight, per_sample_weights):
        ctx.save_for_backward(indices, weight, per_sample_weights)
        return embedding_bag_fwd_triton(indices, weight, per_sample_weights)

    @staticmethod
    def backward(ctx, gradient):
        indices, weight, per_sample_weights = ctx.saved_tensors
        weight_g, psw_g = embedding_bag_bwd_atomics(
            indices, weight, per_sample_weights, gradient,
        )
        return None, weight_g, psw_g

def fast_embedding_bag(
    indices: torch.Tensor,
    weight: torch.Tensor,
    per_sample_weights: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    """Drop-in replacement for custom_embedding_bag_function using fast Triton kernels.

    ~13x faster than PyTorch F.embedding_bag + index_add_ backward.
    Requires num_memory_slots divisible by 8.
    """
    assert mode == "sum"
    return FastEmbeddingBag.apply(indices, weight, per_sample_weights)

# ============================================================
# Triton scatter_add_: drop-in replacement for index_add_
# Uses Triton atomic_add (same approach as embedding_bag backward).
# ============================================================

# ============================================================
# Fused outer-product scatter-add: computes kv[n] * vec[d] and
# atomically adds to target[slot, d] in one pass, avoiding the
# [C, W, dim] intermediate tensor materialization.
# ============================================================

@triton.jit
def _fused_outer_scatter_add_kernel(
    target_ptr,          # [num_slots, dim] — atomic add target
    indices_ptr,         # [C, W] — slot indices
    weights_ptr,         # [C, W] — per-entry scalar weights (float32)
    vectors_ptr,         # [C, dim] — per-token vectors (float32)
    C: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: target[indices[t,n]] += weights[t,n] * vectors[t,:] for all (t,n).
    Grid: (C*W,) — one program per (token, write_entry).
    """
    i = tl.program_id(0)
    t = i // W
    slot = tl.load(indices_ptr + i).to(tl.int64)
    w = tl.load(weights_ptr + i).to(tl.float32)

    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim
        vec = tl.load(vectors_ptr + t * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        val = (w * vec).to(target_ptr.dtype.element_ty)
        tl.atomic_add(target_ptr + slot * dim + d_offs, val, mask=d_mask, sem="relaxed")

def fused_outer_scatter_add(
    target: torch.Tensor,     # [num_slots, dim]
    indices: torch.Tensor,    # [C, W]
    weights: torch.Tensor,    # [C, W] float32
    vectors: torch.Tensor,    # [C, dim] float32
):
    """Fused outer-product + scatter-add."""
    C, W = indices.shape
    dim = target.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    _fused_outer_scatter_add_kernel[(C * W,)](
        target, indices.contiguous().reshape(-1).to(torch.int64),
        weights.contiguous().reshape(-1),
        vectors.contiguous(),
        C=C, W=W, dim=dim, BLOCK_D=BLOCK_D,
        num_warps=1,
    )

# ============================================================
# Triton weighted reduction kernel (replaces bmm for embedding_bag-style ops)
# Computes out[t, d] = sum_n weights[t, n] * mat[t, n, d] for each token t.
# Replaces torch.bmm([C,1,W] @ [C,W,dim]) which launches C independent gemv2T.
# ============================================================

@triton.jit
def _weighted_reduction_kernel(
    weights_ptr,     # [C, W] — per-entry scalar weights (float32)
    mat_ptr,         # [C, W, dim] — per-token per-entry vectors (bf16 or f32)
    out_ptr,         # [C, dim] — output (float32)
    C: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Weighted reduction: out[t, d] = sum_n weights[t, n] * mat[t, n, d].

    Grid: (C, cdiv(dim, BLOCK_D)) — one program per (token, dim_tile).
    Each program loads W scalar weights and W×BLOCK_D matrix elements,
    accumulates the weighted sum in float32, and stores BLOCK_D outputs.
    """
    t = tl.program_id(0)
    d_tile = tl.program_id(1)
    if t >= C:
        return

    d_start = d_tile * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    # Accumulate weighted sum over W entries
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n in range(W):
        w_n = tl.load(weights_ptr + t * W + n).to(tl.float32)
        m_vals = tl.load(mat_ptr + (t * W + n) * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        acc += w_n * m_vals

    tl.store(out_ptr + t * dim + d_offs, acc, mask=d_mask)

def triton_weighted_reduction(
    weights: torch.Tensor,    # [C, W] float32
    mat: torch.Tensor,        # [C, W, dim] any dtype
    out: torch.Tensor = None, # [C, dim] float32 output buffer (optional)
) -> torch.Tensor:
    """Compute out[t, d] = sum_n weights[t, n] * mat[t, n, d].

    Replaces torch.bmm(weights.unsqueeze(1), mat).squeeze(1).
    """
    C, W = weights.shape
    dim = mat.shape[2]
    if out is None:
        out = torch.empty(C, dim, device=weights.device, dtype=torch.float32)
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    grid = (C, triton.cdiv(dim, BLOCK_D))
    _weighted_reduction_kernel[grid](
        weights.contiguous(), mat.contiguous(), out,
        C=C, W=W, dim=dim, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out

# ============================================================
# Fused outer-product scatter-add with per-token vector scale:
# target[indices[t,n]] += weights[t,n] * scale[t] * vectors[t,:]
# Avoids materializing d_retrieved = -beta * d_dva as intermediate.
# ============================================================

# ============================================================
# Fused Phase A backward kernel: gather + decay_write_back + grad_g_decay + undo_writes
# Replaces 4 separate operations (47% of Phase A) with 1 Triton kernel.
# ============================================================

@triton.jit
def fused_phaseA_gather_decay_undo(
    grad_memory: torch.Tensor,        # [num_slots, dim]
    memory: torch.Tensor,             # [num_slots, dim]
    all_mem_before_decay: torch.Tensor,  # [B_T, W, dim] or flat
    flat_k: torch.Tensor,             # [C*W]
    total_cumul: torch.Tensor,        # [num_slots, 1] per-slot OR [C*W] per-entry
    chunk_dw: torch.Tensor,           # [C*W, dim] output buffer
    grad_g_decay: torch.Tensor,       # [C*W] output buffer
    chunk_offset: int,                # offset into all_mem_before_decay
    C: int,
    W: int,
    per_entry_cumul: bool = False,    # If True, total_cumul is [C*W] per-entry
    # Optional: fused contraction outputs (if provided, computes in same kernel)
    d_dv_writes: torch.Tensor = None,   # [SC, dim] f32 — weighted reduction (zeroed before call)
    base_dv_dw: torch.Tensor = None,    # [SC*W] f32 — dot product
    kv_pd: torch.Tensor = None,         # [C*W] f32 — weights for d_dv_writes
    delta_v: torch.Tensor = None,       # [SC, dim] — delta_v for base_dv_dw
):
    """Gather chunk_dw, then fused decay+grad_g+undo+contractions kernel.

    When d_dv_writes and base_dv_dw are provided, computes all 5 outputs in
    a single pass over chunk_dw, eliminating 2 extra HBM reads of the 151 MB buffer.
    """
    CW = C * W
    dim = grad_memory.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))

    # Atomic gather: all reads complete before any writes (fixes race condition)
    torch.index_select(grad_memory, 0, flat_k, out=chunk_dw)

    if per_entry_cumul and d_dv_writes is not None:
        # Fused path: decay + grad_g + undo + d_dv_writes + base_dv_dw
        _fused_phaseA_gather_decay_undo_per_entry_kernel[(CW,)](
            grad_memory, memory.data, all_mem_before_decay,
            flat_k.to(torch.int64), total_cumul,
            chunk_dw, grad_g_decay,
            d_dv_writes, base_dv_dw,
            kv_pd, delta_v,
            chunk_offset=chunk_offset, CW=CW, W=W, dim=dim, BLOCK_D=BLOCK_D,
            num_warps=1,
        )
    elif per_entry_cumul:
        # Legacy path without contractions (for cs=1 backward / additive ops)
        _fused_phaseA_gather_decay_undo_per_entry_kernel[(CW,)](
            grad_memory, memory.data, all_mem_before_decay,
            flat_k.to(torch.int64), total_cumul,
            chunk_dw, grad_g_decay,
            # Dummy pointers — kernel won't write if we pass the same grad_g_decay
            grad_g_decay, grad_g_decay,
            total_cumul, all_mem_before_decay,
            chunk_offset=chunk_offset, CW=CW, W=W, dim=dim, BLOCK_D=BLOCK_D,
            num_warps=1,
        )

@triton.jit
def _fused_phaseA_gather_decay_undo_per_entry_kernel(
    # Inputs
    grad_memory_ptr,       # [num_slots, dim]
    memory_ptr,            # [num_slots, dim]
    snapshot_ptr,          # [B_T, W, dim]
    flat_k_ptr,            # [C*W]
    cumul_per_entry_ptr,   # [C*W] — per-entry cumulative decay
    chunk_dw_ptr,          # [C*W, dim] — pre-gathered chunk_dw (INPUT)
    # Outputs
    grad_g_decay_ptr,      # [C*W] — per-entry grad_g from decay
    d_dv_writes_ptr,       # [SC, dim] f32 — weighted reduction output (atomic_add)
    base_dv_dw_ptr,        # [SC*W] f32 — dot product output (one per entry)
    # Extra inputs for contractions
    kv_pd_ptr,             # [C*W] f32 — k_val * post_decay weights
    delta_v_ptr,           # [SC, dim] — delta_v for dot product
    # Dimensions
    chunk_offset,
    CW: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused Phase A: decay + grad_g + undo + d_dv_writes + base_dv_dw.

    Reads chunk_dw ONCE and computes all outputs in a single pass over dim.
    Eliminates 2 extra HBM reads of the 151 MB chunk_dw buffer per time step.
    """
    i = tl.program_id(0)
    if i >= CW:
        return

    slot = tl.load(flat_k_ptr + i).to(tl.int64)
    cumul = tl.load(cumul_per_entry_ptr + i).to(tl.float32)
    chunk_offset_i64 = chunk_offset.to(tl.int64)
    kv_pd_val = tl.load(kv_pd_ptr + i).to(tl.float32)
    t_local = i // W

    partial_g = tl.zeros([1], dtype=tl.float32)
    partial_dv_dw = tl.zeros([1], dtype=tl.float32)

    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim

        cw_val = tl.load(chunk_dw_ptr + i * dim + d_offs, mask=d_mask, other=0.0)
        cw_f32 = cw_val.to(tl.float32)
        decayed = (cw_f32 * cumul).to(cw_val.dtype)
        tl.store(grad_memory_ptr + slot * dim + d_offs, decayed, mask=d_mask)
        snap_val = tl.load(snapshot_ptr + (chunk_offset_i64 + i) * dim + d_offs, mask=d_mask, other=0.0)
        partial_g += tl.sum(cw_f32 * snap_val.to(tl.float32) * cumul)
        tl.store(memory_ptr + slot * dim + d_offs, snap_val, mask=d_mask)

        # d_dv_writes: atomic_add kv_pd * chunk_dw to per-token output
        # Output [SC, dim] f32 = 4.72 MB — fits in L2, atomics are fast
        tl.atomic_add(d_dv_writes_ptr + t_local * dim + d_offs,
                      (kv_pd_val * cw_f32).to(tl.float32), mask=d_mask, sem="relaxed")

        # base_dv_dw: accumulate dot(delta_v, chunk_dw) in registers
        dv_val = tl.load(delta_v_ptr + t_local * dim + d_offs, mask=d_mask, other=0.0)
        partial_dv_dw += tl.sum(dv_val.to(tl.float32) * cw_f32)

    tl.store(grad_g_decay_ptr + i, tl.sum(partial_g))
    tl.store(base_dv_dw_ptr + i, tl.sum(partial_dv_dw))

@triton.jit
def _fused_phaseA_combined_write_kernel(
    # Inputs
    grad_memory_ptr,       # [num_slots, dim] — atomic_add target
    memory_ptr,            # [num_slots, dim] — write snapshot (undo)
    snapshot_ptr,          # [B_T*W, dim] — all_mem_before_decay
    flat_k_ptr,            # [C*W]
    cumul_per_entry_ptr,   # [C*W] — per-entry cumulative decay
    is_last_ptr,           # [C*W] int32 — 1 if last entry for this slot
    is_only_ptr,           # [C*W] int32 — 1 if sole entry for this slot in chunk
    chunk_dw_ptr,          # [C*W, dim] — pre-gathered chunk_dw (INPUT, bf16)
    d_dva_ptr,             # [SC, dim] f32 — pre-computed d_dva (INPUT)
    kv_ck_ptr,             # [C*W] f32 — k_val * cumul_k weights for scatter
    neg_beta_ptr,          # [C] f32 — -beta per token
    # Outputs
    grad_g_decay_ptr,      # [C*W] — per-entry grad_g from decay
    base_dv_dw_ptr,        # [SC*W] f32 — dot product output
    base_ret_ptr,          # [SC*W] f32 — dot(d_ret, snapshot) per entry (Phase B Step 7)
    # Extra inputs
    delta_v_ptr,           # [SC, dim] — delta_v for base_dv_dw
    # Dimensions
    chunk_offset,
    CW: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused Phase A with combined grad_memory write.

    For entries that are the SOLE entry for their slot (is_only=1, ~93% of entries),
    uses plain tl.store instead of tl.atomic_add — avoids redundant HBM read from
    the atomic RMW since we already have the old value in chunk_dw (gathered by GR).

    For multi-entry slots (is_only=0), keeps atomic_add for correctness:
      if is_last: grad_memory[slot] += chunk_dw*(cumul-1) + kv_ck*(-beta)*d_dva
      else:       grad_memory[slot] += kv_ck*(-beta)*d_dva

    Also computes: grad_g_decay, base_dv_dw, base_ret (fused Phase B Step 7),
    and undo writes.

    base_ret[i] = dot(-beta[t] * d_dva[t,:], snapshot[i,:])
               = neg_beta * sum_d(d_dva[t,d] * snapshot[i,d])
    Fusing this eliminates a separate [NC, 1, dim] @ [NC, W, dim]^T bmm that
    re-reads the entire snapshot buffer (4 GB/layer at level13).
    """
    i = tl.program_id(0)
    if i >= CW:
        return

    slot = tl.load(flat_k_ptr + i).to(tl.int64)
    cumul = tl.load(cumul_per_entry_ptr + i).to(tl.float32)
    is_last = tl.load(is_last_ptr + i)
    is_only = tl.load(is_only_ptr + i)
    chunk_offset_i64 = chunk_offset.to(tl.int64)
    t_local = i // W

    # Scatter contribution scalars
    kv_ck_val = tl.load(kv_ck_ptr + i).to(tl.float32)
    neg_beta_val = tl.load(neg_beta_ptr + t_local).to(tl.float32)
    scatter_scale = kv_ck_val * neg_beta_val

    # Decay correction (only for last entry per slot, used in atomic path)
    decay_corr = tl.where(is_last != 0, cumul - 1.0, 0.0)

    partial_g = tl.zeros([1], dtype=tl.float32)
    partial_dv_dw = tl.zeros([1], dtype=tl.float32)
    partial_ret = tl.zeros([1], dtype=tl.float32)

    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim

        cw_val = tl.load(chunk_dw_ptr + i * dim + d_offs, mask=d_mask, other=0.0)
        cw_f32 = cw_val.to(tl.float32)

        dva_val = tl.load(d_dva_ptr + t_local * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)

        if is_only != 0:
            # Plain store: old_val * cumul + scatter.  old_val = chunk_dw (from GR gather).
            # Eliminates redundant HBM read from atomic RMW.
            new_val = cw_f32 * cumul + scatter_scale * dva_val
            tl.store(grad_memory_ptr + slot * dim + d_offs,
                     new_val.to(grad_memory_ptr.dtype.element_ty),
                     mask=d_mask)
        else:
            # Multi-entry slot: keep atomic for correctness
            combined = cw_f32 * decay_corr + scatter_scale * dva_val
            tl.atomic_add(grad_memory_ptr + slot * dim + d_offs,
                          combined.to(grad_memory_ptr.dtype.element_ty),
                          mask=d_mask, sem="relaxed")

        snap_val = tl.load(snapshot_ptr + (chunk_offset_i64 + i) * dim + d_offs, mask=d_mask, other=0.0)
        partial_g += tl.sum(cw_f32 * snap_val.to(tl.float32) * cumul)
        tl.store(memory_ptr + slot * dim + d_offs, snap_val, mask=d_mask)

        # base_ret: dot(d_dva, snapshot) — fused Phase B Step 7
        # d_ret = -beta * d_dva, so base_ret = neg_beta * dot(d_dva, snapshot)
        # dva_val already loaded above; snap_val already loaded here.
        partial_ret += tl.sum(dva_val * snap_val.to(tl.float32))

        dv_val = tl.load(delta_v_ptr + t_local * dim + d_offs, mask=d_mask, other=0.0)
        partial_dv_dw += tl.sum(dv_val.to(tl.float32) * cw_f32)

    tl.store(grad_g_decay_ptr + i, tl.sum(partial_g))
    tl.store(base_dv_dw_ptr + i, tl.sum(partial_dv_dw))
    tl.store(base_ret_ptr + i, neg_beta_val * tl.sum(partial_ret))

# ============================================================
# Triton weighted reduction: d_dv_writes[t,d] = sum_w weights[t,w] * data[t,w,d]
# Reads data in bf16, weights in f32, accumulates in f32. No materialized f32 copy.
# ============================================================

@triton.jit
def _weighted_reduce_kernel(
    out_ptr,       # [SC, dim] f32 output
    weights_ptr,   # [SC, W] f32 weights
    data_ptr,      # [SC*W, dim] bf16 data (contiguous as [SC, W, dim])
    SC: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """out[t,d] = sum_w weights[t,w] * data[t*W+w, d].

    2D grid: (SC, num_d_blocks) — one program per (token, dim_tile) pair.
    Higher occupancy than 1D grid for large dim (e.g., dim=1920 → 8x more programs).
    """
    t = tl.program_id(0)
    d_block = tl.program_id(1)
    d_start = d_block * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for w in range(W):
        wt = tl.load(weights_ptr + t * W + w).to(tl.float32)
        val = tl.load(data_ptr + (t * W + w) * dim + d_offs, mask=d_mask, other=0.0)
        acc += wt * val.to(tl.float32)
    tl.store(out_ptr + t * dim + d_offs, acc, mask=d_mask)

def weighted_reduce_f32(
    weights: torch.Tensor,   # [SC, W] f32
    data: torch.Tensor,      # [SC*W, dim] bf16
    out: torch.Tensor,       # [SC, dim] f32 (pre-allocated)
):
    """Weighted reduction: out[t,d] = sum_w weights[t,w] * data[t,w,d].
    Reads data in native dtype, accumulates in f32. No copies.
    Uses 2D grid (SC, num_d_blocks) for higher occupancy."""
    SC, W = weights.shape
    dim = out.shape[1]
    BLOCK_D = min(256, triton.next_power_of_2(dim))
    num_d_blocks = (dim + BLOCK_D - 1) // BLOCK_D
    _weighted_reduce_kernel[(SC, num_d_blocks)](
        out, weights, data,
        SC=SC, W=W, dim=dim, BLOCK_D=BLOCK_D,
        num_warps=1,
    )


@triton.jit
def _fused_gather_weighted_reduce_kernel(
    out_ptr,       # [SC, dim] f32 output (weighted reduce result)
    gather_out_ptr,  # [CW, dim] gathered data output (optional, NULL=0 to skip)
    weights_ptr,   # [SC, W] f32 weights
    src_ptr,       # [num_slots, dim] source tensor (bf16 or f32, random access)
    idx_ptr,       # [CW] int64 indices into src_ptr
    SC: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STORE_GATHER: tl.constexpr,  # True to also store raw gathered data
):
    """Fused gather + weighted reduce with optional dual output.

    Output 1 (always): out[t,d] = sum_w weights[t,w] * src[idx[t*W+w], d]
    Output 2 (if STORE_GATHER): gather_out[t*W+w, d] = src[idx[t*W+w], d]

    Replaces separate index_select + weighted_reduce: one gather pass instead of
    index_select writing [CW,dim] intermediate → weighted_reduce reading it.
    All arithmetic in f32 regardless of src dtype.
    """
    t = tl.program_id(0)
    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for w in range(W):
            wt = tl.load(weights_ptr + t * W + w).to(tl.float32)
            slot_idx = tl.load(idx_ptr + t * W + w)
            val = tl.load(src_ptr + slot_idx * dim + d_offs, mask=d_mask, other=0.0)
            val_f32 = val.to(tl.float32)
            acc += wt * val_f32
            if STORE_GATHER:
                tl.store(gather_out_ptr + (t * W + w) * dim + d_offs, val, mask=d_mask)
        tl.store(out_ptr + t * dim + d_offs, acc, mask=d_mask)


def fused_gather_weighted_reduce_f32(
    weights: torch.Tensor,   # [SC, W] f32
    src: torch.Tensor,       # [num_slots, dim] source (bf16/f32, random access)
    idx: torch.Tensor,       # [CW] int64 indices
    out: torch.Tensor,       # [SC, dim] f32 (pre-allocated)
    gather_out: torch.Tensor | None = None,  # [CW, dim] optional gathered data output
):
    """Fused gather + weighted reduce: out[t,d] = sum_w weights[t,w] * src[idx[t*W+w], d].

    If gather_out is provided, also stores the raw gathered data (replaces index_select).
    Replaces torch.index_select(src, 0, idx) + weighted_reduce_f32(weights, gathered, out).
    Saves one kernel launch and avoids a full [CW, dim] read pass.
    """
    SC, W = weights.shape
    dim = out.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    store_gather = gather_out is not None
    # num_warps tuning: the GR kernel loops `for w in range(W)` sequentially,
    # so all warps in a block access the SAME random slot at each iteration.
    # More warps per block → fewer blocks per SM → fewer distinct random addresses
    # in flight → worse MLP. Optimal num_warps depends on program count (SC):
    #   SC≥2048 (level05-08): nw=1 best (64 blocks/SM, each with 1 random addr)
    #   SC≤256  (level13):    nw=16 best (only 2 blocks/SM, need 16 warps for MLP)
    nw = 1 if SC >= 1024 else 16
    _fused_gather_weighted_reduce_kernel[(SC,)](
        out, gather_out if store_gather else src,  # dummy ptr when not storing
        weights, src, idx,
        SC=SC, W=W, dim=dim, BLOCK_D=BLOCK_D,
        STORE_GATHER=store_gather,
        num_warps=nw,
    )

# ============================================================
# Fused elementwise kernels for Phase 1 and Phase 2
# ============================================================

@triton.jit
def _fused_build_M_sys_kernel(
    beta_ptr,      # [N*C] flattened beta values
    A_ptr,         # [N, C, C]
    out_ptr,       # [N, C, C] output: I + beta * A * tril_mask
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,  # next_power_of_2(C) for tl.arange
):
    """Fused M_sys = I + beta * A * tril(mask, -1).

    Each program computes one (chunk, row) of the output matrix.
    Replaces: eye.clone() + beta_ch * A_all * tril_mask which launches
    3+ elementwise kernels (clone, mul, mul, masked_fill, add).
    Grid: (N * C,) — one program per (chunk, row).
    """
    pid = tl.program_id(0)
    ci = pid // C
    i = pid % C

    beta_i = tl.load(beta_ptr + ci * C + i)

    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < C
    a_vals = tl.load(A_ptr + ci * C * C + i * C + cols, mask=col_mask, other=0.0)

    # tril(diagonal=-1): mask where col < row (i.e., j < i)
    tril_mask = cols < i
    # I + beta * A * tril_mask
    result = tl.where(tril_mask, beta_i * a_vals, 0.0)
    # Add identity diagonal
    result = tl.where(cols == i, result + 1.0, result)

    tl.store(out_ptr + ci * C * C + i * C + cols, result, mask=col_mask)

def fused_build_M_sys(
    beta: torch.Tensor,    # [N, C, 1]
    A: torch.Tensor,       # [N, C, C]
) -> torch.Tensor:
    """Fused M_sys = I + beta * A * tril(mask, -1).

    Replaces:
        M_sys = eye(C).expand(N, -1, -1).clone()
        M_sys += beta * A * tril_mask
    with a single Triton kernel launch.
    """
    N, C, _ = A.shape
    out = torch.empty_like(A)
    # beta is [N, C, 1], flatten to [N*C] for kernel
    beta_flat = beta.reshape(N * C)
    BLOCK_C = triton.next_power_of_2(C)
    _fused_build_M_sys_kernel[(N * C,)](
        beta_flat, A, out,
        N=N, C=C, BLOCK_C=BLOCK_C,
    )
    return out

@triton.jit
def _fused_dedup_decay_kernel(
    k_idx_ptr,     # [N, C, W] — slot indices
    g_ptr,         # [N*C] — flattened log gate values
    out_ptr,       # [N*C*W] — output decay_per, masked by dedup
    N_total: tl.constexpr,  # N * C
    C: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,  # next_power_of_2(W) for tl.arange
):
    """Fused is_dup detection + masked decay computation.

    For each (chunk, token) pair, checks if k_idx[t, n] appears at any
    earlier position n' < n (duplicate within same token), and sets
    decay[t, n] = 1.0 if duplicate, exp(g[t]) otherwise.

    Replaces:
        is_dup = (k_idx.unsqueeze(3) == k_idx.unsqueeze(2)).tril(-1).any(3)
        decay_raw = exp(g).expand(N, C, W)
        decay_per = where(~is_dup, decay_raw, ones)
    which allocates [N, C, W, W] bool tensor and launches 5+ kernels.

    Grid: (N * C,) — one program per (chunk, token).
    """
    pid = tl.program_id(0)
    ci = pid // C
    t = pid % C

    g_val = tl.load(g_ptr + ci * C + t)
    decay = tl.exp(g_val)

    base_idx = (ci * C + t) * W
    offs_n = tl.arange(0, BLOCK_W)
    w_mask = offs_n < W
    idx_vals = tl.load(k_idx_ptr + base_idx + offs_n, mask=w_mask, other=-1)

    # For each position n, check if idx_vals[n] matches any earlier position
    # This is O(W^2) per program but W=64 and it's all in registers
    out_base = ci * C * W + t * W
    for n in range(W):
        idx_n = tl.load(k_idx_ptr + base_idx + n)
        # Check if idx_n appears at any position m < n (using offs_n as position vector)
        is_dup = tl.sum(tl.where((offs_n < n) & w_mask & (idx_vals == idx_n), 1, 0))
        val = tl.where(is_dup > 0, 1.0, decay)
        tl.store(out_ptr + out_base + n, val)

@triton.jit
def _fused_dedup_decay_per_write_kernel(
    k_idx_ptr,
    g_ptr,       # [N*C*W] — per-write g values
    out_ptr,
    N_total: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Same as _fused_dedup_decay_kernel but with per-write g values."""
    pid = tl.program_id(0)
    ci = pid // C
    t = pid % C

    base_idx = (ci * C + t) * W
    offs_n = tl.arange(0, BLOCK_W)
    w_mask = offs_n < W
    idx_vals = tl.load(k_idx_ptr + base_idx + offs_n, mask=w_mask, other=-1)

    out_base = ci * C * W + t * W
    g_base = ci * C * W + t * W
    for n in range(W):
        idx_n = tl.load(k_idx_ptr + base_idx + n)
        g_val = tl.load(g_ptr + g_base + n)
        decay = tl.exp(g_val)
        is_dup = tl.sum(tl.where((offs_n < n) & w_mask & (idx_vals == idx_n), 1, 0))
        val = tl.where(is_dup > 0, 1.0, decay)
        tl.store(out_ptr + out_base + n, val)

def fused_dedup_decay(
    k_idx: torch.Tensor,   # [N, C, W]
    g: torch.Tensor,       # [N, C, 1] or [N, C, W]
) -> torch.Tensor:
    """Fused dedup detection + masked decay.

    Returns [N, C*W, 1] tensor where duplicate entries within a token
    have value 1.0 and non-duplicates have value exp(g).

    When g is [N, C, 1], uses scalar g per (chunk, token).
    When g is [N, C, W], uses per-write g values.
    """
    N, C, W = k_idx.shape
    out = torch.empty(N, C * W, 1, device=k_idx.device, dtype=g.dtype)
    BLOCK_W = triton.next_power_of_2(W)

    if g.shape[-1] == W:
        # Per-write g
        g_flat = g.reshape(N * C * W)
        _fused_dedup_decay_per_write_kernel[(N * C,)](
            k_idx.contiguous(), g_flat, out.reshape(-1),
            N_total=N * C, C=C, W=W, BLOCK_W=BLOCK_W,
        )
    else:
        # Scalar g per token
        g_flat = g.reshape(N * C)
        _fused_dedup_decay_kernel[(N * C,)](
            k_idx.contiguous(), g_flat, out.reshape(-1),
            N_total=N * C, C=C, W=W, BLOCK_W=BLOCK_W,
        )
    return out


# ============================================================
# Fused matmul+add Triton kernel for Phase 2 BMMs
# ============================================================
# Replaces cuBLAS bmm + elementwise add with a single fused kernel.
# For the tiny [256,256]@[256,768] matmuls in Phase 2, cuBLAS launch
# overhead dominates (12μs per call at 0.9% MFU). Triton fuses the
# add and avoids the cuBLAS dispatch overhead.

@triton.jit
def _fused_matmul_add_kernel(
    A_ptr,     # [P, M, K] f32 — left matrix
    B_ptr,     # [P, K, N] f32 — right matrix
    bias_ptr,  # [P, M, N] f32 — additive bias
    out_ptr,   # [P, M, N] output
    stride_ap: tl.constexpr,  # A batch stride
    stride_bp: tl.constexpr,  # B batch stride
    stride_cp: tl.constexpr,  # bias/out batch stride
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAST_BF16: tl.constexpr,  # if True, cast output to bf16
):
    """Fused: out = A @ B + bias, optionally cast to bf16.

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), P)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)

    # Batch offsets
    a_base = A_ptr + pid_p * stride_ap
    b_base = B_ptr + pid_p * stride_bp
    c_base = bias_ptr + pid_p * stride_cp
    o_base = out_ptr + pid_p * stride_cp

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N
    mn_mask = m_mask[:, None] & n_mask[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K
        a = tl.load(a_base + rm[:, None] * K + rk[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_base + rk[:, None] * N + rn[None, :], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(c_base + rm[:, None] * N + rn[None, :], mask=mn_mask, other=0.0)
    acc += bias

    # Store
    if CAST_BF16:
        tl.store(o_base + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=mn_mask)
    else:
        tl.store(o_base + rm[:, None] * N + rn[None, :], acc, mask=mn_mask)


def fused_matmul_add(
    A: torch.Tensor,        # [P, M, K] f32
    B: torch.Tensor,        # [P, K, N] f32
    bias: torch.Tensor,     # [P, M, N] f32
    out: torch.Tensor,      # [P, M, N] output (f32 or bf16)
):
    """Fused: out = A @ B + bias.

    Replaces torch.bmm + add with single Triton kernel.
    Output dtype matches out tensor (bf16 if out is bf16, f32 otherwise).
    """
    P, M, K = A.shape
    _, _, N = B.shape
    cast_bf16 = out.dtype == torch.bfloat16

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = min(64, K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), P)

    _fused_matmul_add_kernel[grid](
        A, B, bias, out,
        stride_ap=M * K, stride_bp=K * N, stride_cp=M * N,
        M=M, K=K, N=N,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        CAST_BF16=cast_bf16,
    )


@triton.jit
def _fused_dual_matmul_add_kernel(
    # Two left matrices, shared right matrix
    A1_ptr,     # [P, M, K] — left matrix for output 1 (delta_v)
    A2_ptr,     # [P, M, K] — left matrix for output 2 (readings)
    B_ptr,      # [P, K, N] — shared right matrix (retrieved)
    # Biases
    bias1_ptr,  # [P, M, N] — bias for output 1 (delta_v_const)
    bias2a_ptr, # [P, M, N] — first bias for output 2 (inter)
    bias2b_ptr, # [P, M, N] — second bias for output 2 (qk_dvc)
    # Outputs
    out1_ptr,   # [P, M, N] — f32 output (delta_v)
    out2_ptr,   # [P, M, N] — bf16 output (readings)
    # Strides (all contiguous with same batch stride pattern)
    stride_ap: tl.constexpr,  # A batch stride (M * K)
    stride_bp: tl.constexpr,  # B batch stride (K * N)
    stride_cp: tl.constexpr,  # bias/out batch stride (M * N)
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dual-output fused matmul+add. Loads B once for two matmuls.

    Computes:
        out1 = A1 @ B + bias1              (f32 output)
        out2 = A2 @ B + bias2a + bias2b    (bf16 output)

    Saves one kernel launch and one full B load compared to two separate
    fused_matmul_add calls. Used when both matmuls share the same B (retrieved).

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), P)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)

    # Batch offsets
    a1_base = A1_ptr + pid_p * stride_ap
    a2_base = A2_ptr + pid_p * stride_ap
    b_base = B_ptr + pid_p * stride_bp
    c1_base = bias1_ptr + pid_p * stride_cp
    c2a_base = bias2a_ptr + pid_p * stride_cp
    c2b_base = bias2b_ptr + pid_p * stride_cp
    o1_base = out1_ptr + pid_p * stride_cp
    o2_base = out2_ptr + pid_p * stride_cp

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N
    mn_mask = m_mask[:, None] & n_mask[None, :]

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        k_mask = rk < K
        # Load B tile ONCE — shared between both matmuls
        b = tl.load(b_base + rk[:, None] * N + rn[None, :], mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        # A1 tile for delta_v
        a1 = tl.load(a1_base + rm[:, None] * K + rk[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        acc1 += tl.dot(a1, b)
        # A2 tile for readings
        a2 = tl.load(a2_base + rm[:, None] * K + rk[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        acc2 += tl.dot(a2, b)

    tile_offs = rm[:, None] * N + rn[None, :]

    # Output 1: delta_v (f32)
    bias1 = tl.load(c1_base + tile_offs, mask=mn_mask, other=0.0)
    tl.store(o1_base + tile_offs, acc1 + bias1, mask=mn_mask)

    # Output 2: readings — two biases: inter + qk_dvc
    bias2a = tl.load(c2a_base + tile_offs, mask=mn_mask, other=0.0)
    bias2b = tl.load(c2b_base + tile_offs, mask=mn_mask, other=0.0)
    out2_val = acc2 + bias2a + bias2b
    tl.store(o2_base + tile_offs, out2_val, mask=mn_mask)


def fused_dual_matmul_add(
    A1: torch.Tensor,       # [P, M, K] — B_matrix
    A2: torch.Tensor,       # [P, M, K] — QB (precomputed QK_tril @ B_matrix)
    B: torch.Tensor,        # [P, K, N] — retrieved
    bias1: torch.Tensor,    # [P, M, N] — delta_v_const
    bias2a: torch.Tensor,   # [P, M, N] — inter (from gather)
    bias2b: torch.Tensor,   # [P, M, N] — qk_dvc (precomputed QK_tril @ delta_v_const)
    out1: torch.Tensor,     # [P, M, N] — f32 output (delta_v)
    out2: torch.Tensor,     # [P, M, N] — bf16 output (readings)
):
    """Dual-output matmul+add: out1 = A1@B+bias1 (f32), out2 = A2@B+bias2a+bias2b (bf16).

    Reads B once for both matmuls, saving 1 kernel launch + 1 B-load per call.
    Used in Phase 2 loop where both matmuls share retrieved as their B operand.
    """
    P, M, K = A1.shape
    _, _, N = B.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = min(64, K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), P)

    _fused_dual_matmul_add_kernel[grid](
        A1, A2, B, bias1, bias2a, bias2b, out1, out2,
        stride_ap=M * K, stride_bp=K * N, stride_cp=M * N,
        M=M, K=K, N=N,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )


# ============================================================
# Fused product key decode kernel (T=1 inference fast path)
# ============================================================

@triton.jit
def _fused_pkm_remap_sort_kernel(
    val_ptr,          # [N, K_SUB] f32 — topk scores (unsorted by index)
    inner_idx_ptr,    # [N, K_SUB] int64 — inner indices from final topk
    idx1_ptr,         # [N, K_SUB] int64 — original half-1 indices
    idx2_ptr,         # [N, K_SUB] int64 — original half-2 indices
    out_val_ptr,      # [N, K_SUB] f32 output — sorted by index
    out_idx_ptr,      # [N, K_SUB] int64 output — sorted ascending
    # Workspace for bitonic sort (store-barrier-load pattern)
    sort_key_ptr,     # [N, SORT_BLOCK] int64 workspace
    sort_perm_ptr,    # [N, SORT_BLOCK] int64 workspace
    N: tl.constexpr,
    K_SUB: tl.constexpr,
    HALF_KEY: tl.constexpr,
    SORT_BLOCK: tl.constexpr,   # next_power_of_2(K_SUB)
):
    """Fused: divmod → 2×gather → product_idx → bitonic argsort.

    Replaces 7 PyTorch kernel launches (divmod, 2 gathers, mul+add, sort, gather)
    with a single Triton kernel per PKM call.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    offs = tl.arange(0, SORT_BLOCK)
    k_mask = offs < K_SUB
    base = pid * K_SUB
    sort_base = pid * SORT_BLOCK

    # Load topk results
    inner = tl.load(inner_idx_ptr + base + offs, mask=k_mask, other=0)
    val = tl.load(val_ptr + base + offs, mask=k_mask, other=-1e9)

    # Remap: divmod → gather from original idx1, idx2 → product index
    inner1 = inner // K_SUB
    inner2 = inner % K_SUB
    chosen_i1 = tl.load(idx1_ptr + base + inner1, mask=k_mask, other=0)
    chosen_i2 = tl.load(idx2_ptr + base + inner2, mask=k_mask, other=0)
    final_idx = chosen_i1 * HALF_KEY + chosen_i2

    # Bitonic argsort by final_idx (ascending), carrying val as payload
    # Uses store-barrier-load pattern for cross-thread communication
    sort_key = tl.where(k_mask, final_idx, 0x7FFFFFFFFFFFFFFF)
    sort_perm = offs.to(tl.int64)

    # Store initial state
    tl.store(sort_key_ptr + sort_base + offs, sort_key, mask=(offs < SORT_BLOCK))
    tl.store(sort_perm_ptr + sort_base + offs, sort_perm, mask=(offs < SORT_BLOCK))
    tl.debug_barrier()

    size = 1
    while size < SORT_BLOCK:
        stride = size
        while stride > 0:
            partner = offs ^ stride

            sort_key = tl.load(sort_key_ptr + sort_base + offs, mask=(offs < SORT_BLOCK), other=0x7FFFFFFFFFFFFFFF)
            sort_perm = tl.load(sort_perm_ptr + sort_base + offs, mask=(offs < SORT_BLOCK), other=0x7FFFFFFFFFFFFFFF)
            p_key = tl.load(sort_key_ptr + sort_base + partner, mask=(partner < SORT_BLOCK), other=0x7FFFFFFFFFFFFFFF)
            p_perm = tl.load(sort_perm_ptr + sort_base + partner, mask=(partner < SORT_BLOCK), other=0x7FFFFFFFFFFFFFFF)

            ascending = ((offs & (size << 1)) == 0)
            is_lower = offs < partner
            want_smaller = (is_lower & ascending) | ((~is_lower) & (~ascending))
            my_is_smaller = (sort_key < p_key) | ((sort_key == p_key) & (sort_perm < p_perm))
            should_swap = (want_smaller & (~my_is_smaller) & (partner < SORT_BLOCK)) | \
                          ((~want_smaller) & my_is_smaller & (partner < SORT_BLOCK))

            sort_key = tl.where(should_swap, p_key, sort_key)
            sort_perm = tl.where(should_swap, p_perm, sort_perm)

            tl.store(sort_key_ptr + sort_base + offs, sort_key, mask=(offs < SORT_BLOCK))
            tl.store(sort_perm_ptr + sort_base + offs, sort_perm, mask=(offs < SORT_BLOCK))
            tl.debug_barrier()

            stride = stride >> 1
        size = size << 1

    # Load final sorted state
    sort_key = tl.load(sort_key_ptr + sort_base + offs, mask=(offs < SORT_BLOCK), other=0x7FFFFFFFFFFFFFFF)
    sort_perm = tl.load(sort_perm_ptr + sort_base + offs, mask=(offs < SORT_BLOCK), other=0x7FFFFFFFFFFFFFFF)

    # Permute values by sort order (gather from original val)
    sorted_val = tl.load(val_ptr + base + sort_perm, mask=k_mask, other=0.0)

    # Store sorted outputs
    tl.store(out_idx_ptr + base + offs, sort_key, mask=k_mask)
    tl.store(out_val_ptr + base + offs, sorted_val, mask=k_mask)


def fused_product_key_rest(
    top_val1: torch.Tensor,   # [N, k_sub] — topk values from half-1
    idx1: torch.Tensor,       # [N, k_sub] — topk indices from half-1
    top_val2: torch.Tensor,   # [N, k_sub] — topk values from half-2
    idx2: torch.Tensor,       # [N, k_sub] — topk indices from half-2
    half_key: int,
    k_sub: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused product key completion: outer_sum → topk → remap → argsort.

    Takes the output of two torch.topk calls (one per half-key) and
    produces sorted (val, idx) pairs. Keeps the outer_sum + topk as
    PyTorch (cuDNN-optimized), fuses the remap + argsort into one kernel.

    Returns:
        val: [N, k_sub] sorted values (original shape preserved with T=1 dim)
        idx: [N, k_sub] sorted indices (ascending)
    """
    N = top_val1.shape[0]
    device = top_val1.device

    # Outer sum + topk (3 PyTorch launches — cuDNN optimized for small tensors)
    scores = top_val1.unsqueeze(-1) + top_val2.unsqueeze(-2)  # [N, k_sub, k_sub]
    scores = scores.view(N, -1)                                # [N, k_sub²]
    val, inner_idx = torch.topk(scores, k=k_sub, dim=-1)       # [N, k_sub]

    # Fused remap + argsort (1 Triton kernel replaces 7 PyTorch launches)
    SORT_BLOCK = triton.next_power_of_2(k_sub)

    out_val = torch.empty(N, k_sub, dtype=torch.float32, device=device)
    out_idx = torch.empty(N, k_sub, dtype=torch.int64, device=device)
    sort_key_ws = torch.empty(N, SORT_BLOCK, dtype=torch.int64, device=device)
    sort_perm_ws = torch.empty(N, SORT_BLOCK, dtype=torch.int64, device=device)

    _fused_pkm_remap_sort_kernel[(N,)](
        val.contiguous().to(torch.float32),
        inner_idx.contiguous().to(torch.int64),
        idx1.contiguous().to(torch.int64),
        idx2.contiguous().to(torch.int64),
        out_val, out_idx,
        sort_key_ws, sort_perm_ws,
        N=N, K_SUB=k_sub, HALF_KEY=half_key, SORT_BLOCK=SORT_BLOCK,
        num_warps=1,
    )

    return out_val, out_idx


class OuterAddTopK(torch.autograd.Function):
    """Fused outer-add + topk with sparse backward.

    forward(x1, x2, K_out):
        scores = x1.unsqueeze(-1) + x2.unsqueeze(-2)   # [..., K, K]
        val, idx = topk(scores.flatten(-2), K_out, dim=-1)
        idx1 = idx // K ; idx2 = idx % K
        return val, idx1, idx2

    Backward avoids materializing the [..., K, K] gradient: since only K_out
    entries are selected, the gradient is sparse and can be scattered directly
    into x1 and x2 grad buffers. This eliminates the ~537 MB bf16 reduce_kernel
    that dominates SDM's backward at L13 (4.18 ms per call, called ×4).
    """

    @staticmethod
    def forward(ctx, x1, x2, K_out):
        # x1, x2: [..., K] (must be same shape)
        K = x1.shape[-1]
        scores = x1.unsqueeze(-1) + x2.unsqueeze(-2)   # [..., K, K]
        scores_flat = scores.flatten(-2)               # [..., K*K]
        val, idx = torch.topk(scores_flat, K_out, dim=-1)
        idx1 = idx // K
        idx2 = idx % K
        ctx.save_for_backward(idx1, idx2)
        ctx.K = K
        return val, idx1, idx2

    @staticmethod
    def backward(ctx, grad_val, _grad_idx1, _grad_idx2):
        idx1, idx2 = ctx.saved_tensors
        K = ctx.K
        # grad_x1[..., k] = sum over (k_out: idx1[..., k_out]==k) grad_val[..., k_out]
        # grad_x2 similarly. Use scatter_add along the last dim.
        # idx1, idx2: int64 already, grad_val matches x1/x2 dtype.
        out_shape = list(idx1.shape)
        out_shape[-1] = K
        grad_x1 = torch.zeros(out_shape, dtype=grad_val.dtype, device=grad_val.device)
        grad_x2 = torch.zeros_like(grad_x1)
        grad_x1.scatter_add_(-1, idx1, grad_val)
        grad_x2.scatter_add_(-1, idx2, grad_val)
        return grad_x1, grad_x2, None


def fused_outer_add_topk(x1, x2, K_out):
    """Public wrapper for OuterAddTopK; returns (val, idx1, idx2)."""
    return OuterAddTopK.apply(x1, x2, K_out)


@triton.jit
def _sum_batch_kernel(x_ptr, out_ptr,
                      B: tl.constexpr, S, D: tl.constexpr,
                      stride_xb, stride_xs, stride_od,
                      stride_os,
                      BD: tl.constexpr):
    """One thread block per slot row; sum over B batch dim into output [S, D]."""
    s = tl.program_id(0)
    d_off = tl.arange(0, BD)
    d_mask = d_off < D
    acc = tl.zeros((BD,), dtype=tl.float32)
    for b in tl.static_range(B):
        ptr = x_ptr + b * stride_xb + s * stride_xs + d_off * stride_od
        v = tl.load(ptr, mask=d_mask, other=0.0)
        acc += v.to(tl.float32)
    out_ptr_row = out_ptr + s * stride_os + d_off * stride_od
    tl.store(out_ptr_row, acc.to(out_ptr.dtype.element_ty), mask=d_mask)


def triton_sum_dim0(x):
    """Fast sum over dim 0 of a [B, S, D] tensor. ~5.6x faster than aten.sum
    at L13 sizes (B=4, S=460800, D=1920): 15.7 ms -> 2.8 ms."""
    assert x.ndim == 3 and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    B, S, D = x.shape
    out = torch.empty(S, D, device=x.device, dtype=x.dtype)
    BD = triton.next_power_of_2(D)
    _sum_batch_kernel[(S,)](
        x, out,
        B=B, S=S, D=D,
        stride_xb=x.stride(0), stride_xs=x.stride(1), stride_od=x.stride(2),
        stride_os=out.stride(0),
        BD=BD, num_warps=8,
    )
    return out


class ExpandMemoryBatch(torch.autograd.Function):
    """Equivalent to `param.unsqueeze(0).expand(B, S, D).reshape(B*S, D).clone()`
    but with a fast backward reduce that bypasses aten.sum's slow path.

    Forward: same allocation/copy as the original expand+clone (no savings).
    Backward: triton kernel sums grad [B*S, D] over the batch dim into [S, D],
    avoiding the 15.7 ms aten.sum at L13 (~12 ms kernel-time saving).
    """

    @staticmethod
    def forward(ctx, param, B):
        # param: [S, D] -> [B*S, D] writable contiguous tensor.
        # Allocate fresh and broadcast-copy in one step (single copy, ~7 GB
        # less transient HBM at L13 vs `.expand().reshape().clone()`).
        #
        # History note: on 2026-06-19 this was blamed for a "3x training
        # slowdown" at L13 SFT 128k cp=4 H=2 fp8 16n. That was wrong —
        # the bisection runs had `global_window: 128` (good) vs 2048 (bad)
        # plus an incompatible warm-start checkpoint. Re-tested 2026-06-20
        # with byte-identical configs: 10.67s vs reference 11.7s (faster).
        S, D = param.shape
        ctx.B = B
        ctx.S = S
        ctx.D = D
        # Allocate directly and broadcast-copy in one step
        out = torch.empty(B * S, D, dtype=param.dtype, device=param.device)
        out.view(B, S, D).copy_(param.unsqueeze(0).expand(B, S, D))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        B, S, D = ctx.B, ctx.S, ctx.D
        # grad_out: [B*S, D] is_contiguous=True (came from clone)
        grad_3d = grad_out.view(B, S, D)
        return triton_sum_dim0(grad_3d), None


def expand_memory_batch(param, B):
    """Replacement for `param.unsqueeze(0).expand(B, *).reshape(B*S, D).clone()`
    with a faster backward."""
    return ExpandMemoryBatch.apply(param, B)


# ============================================================
# Fused decode kernels (T=1 inference fast path)
# ============================================================

@triton.jit
def _fused_decode_write_kernel(
    memory_ptr,       # [num_slots, dim] — modified in-place
    k_idx_ptr,        # [BH, W]
    k_val_ptr,        # [BH, W]
    v_ptr,            # [BH, dim]
    beta_ptr,         # [BH, 1]
    g_ptr,            # [BH, 1] or [BH, W] (key_weighted_decay)
    BH: tl.constexpr,
    W: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_WEIGHTED_DECAY: tl.constexpr,
    USE_DELTA_RULE: tl.constexpr,
):
    """Fused decode write step: gather → delta_v → decay + write.

    All BH heads processed in parallel (non-overlapping memory regions).
    PKM guarantees unique write indices per head, so no atomics needed.
    Two-pass: pass 1 gathers + computes delta_v, pass 2 applies decay + write.
    """
    h = tl.program_id(0)
    if h >= BH:
        return

    beta_val = tl.load(beta_ptr + h).to(tl.float32)

    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim

        if USE_DELTA_RULE:
            # Pass 1: gather W slots, compute weighted sum (retrieved)
            acc_retrieved = tl.zeros([BLOCK_D], dtype=tl.float32)
            for n in range(W):
                slot = tl.load(k_idx_ptr + h * W + n)
                weight = tl.load(k_val_ptr + h * W + n).to(tl.float32)
                if KEY_WEIGHTED_DECAY:
                    g_val = tl.load(g_ptr + h * W + n).to(tl.float32)
                else:
                    g_val = tl.load(g_ptr + h).to(tl.float32)
                decay = tl.exp(g_val)
                mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                acc_retrieved += weight * mem_val * decay

            v_tile = tl.load(v_ptr + h * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            delta_v_tile = beta_val * (v_tile - acc_retrieved)
        else:
            # Additive rule: delta_v = beta * v (no retrieval needed)
            v_tile = tl.load(v_ptr + h * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            delta_v_tile = beta_val * v_tile

        # Pass 2: apply decay + write (re-load from memory)
        for n in range(W):
            slot = tl.load(k_idx_ptr + h * W + n)
            weight = tl.load(k_val_ptr + h * W + n).to(tl.float32)
            if KEY_WEIGHTED_DECAY:
                g_val = tl.load(g_ptr + h * W + n).to(tl.float32)
            else:
                g_val = tl.load(g_ptr + h).to(tl.float32)
            decay = tl.exp(g_val)
            mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            new_val = mem_val * decay + weight * delta_v_tile
            tl.store(memory_ptr + slot * dim + d_offs, new_val.to(mem_val.dtype), mask=d_mask)


@triton.jit
def _fused_decode_read_kernel(
    memory_ptr,       # [num_slots, dim]
    q_idx_ptr,        # [BH, R]
    q_val_ptr,        # [BH, R]
    readings_ptr,     # [BH, dim] output
    BH: tl.constexpr,
    R: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused decode read step: gather R slots → weighted sum → reading.

    2D grid (BH, dim_blocks) for maximum occupancy.
    """
    h = tl.program_id(0)
    d_block = tl.program_id(1)
    if h >= BH:
        return

    d_start = d_block * BLOCK_D
    d_offs = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n in range(R):
        slot = tl.load(q_idx_ptr + h * R + n)
        weight = tl.load(q_val_ptr + h * R + n).to(tl.float32)
        mem_val = tl.load(memory_ptr + slot * dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        acc += weight * mem_val

    tl.store(readings_ptr + h * dim + d_offs, acc, mask=d_mask)


def fused_decode_step(
    memory: torch.Tensor,    # [num_slots, dim] — modified in-place
    k_idx: torch.Tensor,     # [BH, W]
    k_val: torch.Tensor,     # [BH, W]
    v: torch.Tensor,         # [BH, dim]
    beta: torch.Tensor,      # [BH, 1]
    g: torch.Tensor,         # [BH, 1] or [BH, W]
    q_idx: torch.Tensor,     # [BH, R]
    q_val: torch.Tensor,     # [BH, R]
    use_delta_rule: bool = True,
    normalize_memory: bool = False,
    key_weighted_decay: bool = False,
) -> torch.Tensor:
    """Fused T=1 decode: write + read in 2 kernel launches (all heads parallel).

    Replaces the BH-iteration Python loop in _step_write_read with
    2 Triton kernels that process all heads simultaneously.
    """
    BH, W = k_idx.shape
    num_slots, dim = memory.shape
    R = q_idx.shape[1]
    device = memory.device
    dtype = memory.dtype
    BLOCK_D = min(1024, triton.next_power_of_2(dim))

    # Kernel B: fused gather → delta_v → decay + write
    _fused_decode_write_kernel[(BH,)](
        memory,
        k_idx.contiguous().to(torch.int64),
        k_val.contiguous().to(dtype),
        v.contiguous().to(dtype),
        beta.contiguous().to(torch.float32),
        g.contiguous().to(torch.float32),
        BH=BH, W=W, dim=dim, BLOCK_D=BLOCK_D,
        KEY_WEIGHTED_DECAY=key_weighted_decay,
        USE_DELTA_RULE=use_delta_rule,
        num_warps=1,
    )

    # Optional: L2-normalize updated memory slots
    if normalize_memory:
        flat_k = k_idx.reshape(-1).to(torch.int64)
        updated = memory.data[flat_k]
        norms = updated.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        memory.data[flat_k] = updated / norms

    # Kernel C: fused gather → weighted sum → reading
    readings = torch.empty(BH, dim, device=device, dtype=dtype)
    num_d_blocks = (dim + BLOCK_D - 1) // BLOCK_D
    _fused_decode_read_kernel[(BH, num_d_blocks)](
        memory,
        q_idx.contiguous().to(torch.int64),
        q_val.contiguous().to(dtype),
        readings,
        BH=BH, R=R, dim=dim, BLOCK_D=BLOCK_D,
    )

    return readings


@triton.jit
def _fused_add_cast_kernel(
    a_ptr,         # [rows, dim] float32 input A
    b_ptr,         # [rows, dim] float32 input B
    out_ptr,       # [actual_rows, dim] bf16/f16 output
    rows: tl.constexpr,
    actual_rows: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused: out = (A + B)[:actual_rows].to(out_dtype).

    Each program handles one row. Fuses the add + slice + dtype cast
    into a single kernel launch, avoiding intermediate tensor allocation.
    Grid: (actual_rows,) — one program per output row.
    """
    row = tl.program_id(0)
    if row >= actual_rows:
        return

    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < dim
        a_val = tl.load(a_ptr + row * dim + d_offs, mask=d_mask, other=0.0)
        b_val = tl.load(b_ptr + row * dim + d_offs, mask=d_mask, other=0.0)
        result = a_val + b_val
        tl.store(out_ptr + row * dim + d_offs, result, mask=d_mask)

def fused_add_cast(
    a: torch.Tensor,        # [rows, dim] float32
    b: torch.Tensor,        # [rows, dim] float32
    out: torch.Tensor,      # [actual_rows, dim] bf16 — pre-allocated output
    actual_rows: int,       # number of valid rows (may be < rows)
):
    """Fused add + cast: out[:actual_rows] = (a + b)[:actual_rows].to(out.dtype).

    Avoids creating a float32 intermediate and separate dtype cast.
    """
    rows = a.shape[0]
    dim = a.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    _fused_add_cast_kernel[(actual_rows,)](
        a, b, out,
        rows=rows, actual_rows=actual_rows, dim=dim, BLOCK_D=BLOCK_D,
    )

@triton.jit
def _fused_pre_bmm_kernel(
    # Inputs [NC*C, dim] (except beta which is [NC*C, 1])
    v_ptr, ret_ptr, beta_ptr, d_dva_ptr,
    # Outputs
    dv_approx_ptr,      # [NC*C, dim] f32
    grad_v_ptr,         # [NC*C, dim] bf16 (cast on store)
    grad_beta_part1_ptr,  # [NC*C, 1] f32 (partial sum over dim)
    # Dims
    NCxC: tl.constexpr, dim: tl.constexpr, BD: tl.constexpr,
    STORE_GRAD_V_BF16: tl.constexpr,
):
    """Fused pre-bmm elementwise: dv_approx, grad_v, grad_beta_part1.

    Grid: (NC*C,) -- one program per (chunk, token).
    Each program processes dim elements, computing:
      residual = v - ret           (not stored — only used internally)
      dv_approx = beta * residual
      grad_v = beta * d_dva                   (cast to bf16)
      grad_beta_part1 = sum(residual * d_dva)  (reduction over dim)
    """
    t = tl.program_id(0)
    if t >= NCxC:
        return

    beta = tl.load(beta_ptr + t)
    acc = tl.zeros([1], dtype=tl.float32)

    for ds in range(0, dim, BD):
        do = ds + tl.arange(0, BD)
        m = do < dim
        v_val = tl.load(v_ptr + t * dim + do, mask=m, other=0.0)
        ret_val = tl.load(ret_ptr + t * dim + do, mask=m, other=0.0)
        dva_val = tl.load(d_dva_ptr + t * dim + do, mask=m, other=0.0)

        res = v_val - ret_val
        tl.store(dv_approx_ptr + t * dim + do, beta * res, mask=m)
        if STORE_GRAD_V_BF16:
            tl.store(grad_v_ptr + t * dim + do, (beta * dva_val).to(tl.bfloat16), mask=m)
        else:
            tl.store(grad_v_ptr + t * dim + do, beta * dva_val, mask=m)
        acc += tl.sum(res * dva_val)

    tl.store(grad_beta_part1_ptr + t, tl.sum(acc))


def fused_pre_bmm(
    v_tm_f32: torch.Tensor,     # [NC, C, dim] f32
    ret_f32: torch.Tensor,      # [NC, C, dim] f32
    beta_tm_f32: torch.Tensor,  # [NC, C, 1] f32
    d_dva_all: torch.Tensor,    # [NC, C, dim] f32
    grad_v_out: torch.Tensor,   # [NC, C, dim] bf16 (pre-allocated output)
) -> tuple:
    """Fused pre-bmm elementwise ops.

    Returns (dv_approx, grad_beta_part1) and writes grad_v in-place.
    Replaces 3 separate [NC,C,dim] elementwise ops + 1 reduce with 1 kernel.
    """
    NC, C, dim = v_tm_f32.shape
    NCxC = NC * C
    dv_approx = torch.empty_like(v_tm_f32)
    grad_beta_part1 = torch.empty(NC, C, 1, device=v_tm_f32.device, dtype=torch.float32)

    BD = min(1024, triton.next_power_of_2(dim))
    store_bf16 = grad_v_out.dtype == torch.bfloat16
    _fused_pre_bmm_kernel[(NCxC,)](
        v_tm_f32, ret_f32, beta_tm_f32.reshape(NCxC), d_dva_all,
        dv_approx, grad_v_out,
        grad_beta_part1.reshape(NCxC),
        NCxC=NCxC, dim=dim, BD=BD,
        STORE_GRAD_V_BF16=store_bf16,
    )
    return dv_approx, grad_beta_part1


@triton.jit
def _fused_pre_bmm_residual_kernel(
    # Inputs [NC*C, dim] (except beta which is [NC*C, 1])
    residual_ptr, beta_ptr, d_dva_ptr,
    # Outputs
    dv_approx_ptr,      # [NC*C, dim] f32
    grad_v_ptr,         # [NC*C, dim] bf16 (cast on store)
    grad_beta_part1_ptr,  # [NC*C, 1] f32 (partial sum over dim)
    # Dims
    NCxC: tl.constexpr, dim: tl.constexpr, BD: tl.constexpr,
    STORE_GRAD_V_BF16: tl.constexpr,
):
    """Fused pre-bmm with pre-computed residual: dv_approx, grad_v, grad_beta_part1.

    Same as _fused_pre_bmm_kernel but takes residual = v - retrieved directly,
    reading 2 [NC,C,dim] tensors instead of 3.
    """
    t = tl.program_id(0)
    if t >= NCxC:
        return

    beta = tl.load(beta_ptr + t)
    acc = tl.zeros([1], dtype=tl.float32)

    for ds in range(0, dim, BD):
        do = ds + tl.arange(0, BD)
        m = do < dim
        res = tl.load(residual_ptr + t * dim + do, mask=m, other=0.0)
        dva_val = tl.load(d_dva_ptr + t * dim + do, mask=m, other=0.0)

        tl.store(dv_approx_ptr + t * dim + do, beta * res, mask=m)
        if STORE_GRAD_V_BF16:
            tl.store(grad_v_ptr + t * dim + do, (beta * dva_val).to(tl.bfloat16), mask=m)
        else:
            tl.store(grad_v_ptr + t * dim + do, beta * dva_val, mask=m)
        acc += tl.sum(res * dva_val)

    tl.store(grad_beta_part1_ptr + t, tl.sum(acc))


def fused_pre_bmm_residual(
    residual_f32: torch.Tensor,  # [NC, C, dim] f32 (v - retrieved, pre-computed in forward)
    beta_tm_f32: torch.Tensor,   # [NC, C, 1] f32
    d_dva_all: torch.Tensor,     # [NC, C, dim] f32
    grad_v_out: torch.Tensor,    # [NC, C, dim] bf16 (pre-allocated output)
) -> tuple:
    """Fused pre-bmm with pre-computed residual.

    Same as fused_pre_bmm but takes residual = v - retrieved directly,
    eliminating one [NC,C,dim] tensor read in the kernel.
    Returns (dv_approx, grad_beta_part1) and writes grad_v in-place.
    """
    NC, C, dim = residual_f32.shape
    NCxC = NC * C
    dv_approx = torch.empty_like(residual_f32)
    grad_beta_part1 = torch.empty(NC, C, 1, device=residual_f32.device, dtype=torch.float32)

    BD = min(1024, triton.next_power_of_2(dim))
    store_bf16 = grad_v_out.dtype == torch.bfloat16
    _fused_pre_bmm_residual_kernel[(NCxC,)](
        residual_f32, beta_tm_f32.reshape(NCxC), d_dva_all,
        dv_approx, grad_v_out,
        grad_beta_part1.reshape(NCxC),
        NCxC=NCxC, dim=dim, BD=BD,
        STORE_GRAD_V_BF16=store_bf16,
    )
    return dv_approx, grad_beta_part1


@triton.jit
def _fused_post_bmm_kernel(
    dM_sys_ptr, A_ptr, beta_ptr, grad_beta_part1_ptr,
    dA_ptr, grad_beta_out_ptr,
    NC: tl.constexpr, C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused: tril_mask + A*dM_lower sum + beta*dM_lower + grad_beta dtype cast.

    Replaces 4 PyTorch ops (tril multiply, A*dM_lower sum, beta*dM_lower, to(bf16)).
    One program per (batch, row). Vectorized over columns.
    """
    pid = tl.program_id(0)
    batch = pid // C
    row = pid % C

    beta_val = tl.load(beta_ptr + batch * C + row).to(tl.float32)
    grad_beta_acc = tl.load(grad_beta_part1_ptr + batch * C + row).to(tl.float32)

    base = batch * C * C + row * C
    col_offs = tl.arange(0, BLOCK_C)
    mask = col_offs < row

    for col_start in range(0, C, BLOCK_C):
        col = col_start + col_offs
        m = mask & (col < C)
        dM_val = tl.load(dM_sys_ptr + base + col, mask=m, other=0.0)
        A_val = tl.load(A_ptr + base + col, mask=m, other=0.0)
        grad_beta_acc += tl.sum(A_val * dM_val)
        tl.store(dA_ptr + base + col, tl.where(m, beta_val * dM_val, 0.0), mask=col < C)
        mask = col_offs + col_start + BLOCK_C < row

    tl.store(grad_beta_out_ptr + batch * C + row, grad_beta_acc)


def fused_post_bmm(
    dM_sys: torch.Tensor,         # [NC, C, C] f32 — dM_sys_all
    A: torch.Tensor,              # [NC, C, C] f32 — wy_A_tm
    beta: torch.Tensor,           # [NC, C, 1] f32 — beta_tm_f32
    grad_beta_part1: torch.Tensor, # [NC, C, 1] f32 — partial grad_beta from pre-bmm
    grad_beta_out: torch.Tensor,  # [NC, C, 1] bf16 — output
) -> torch.Tensor:
    """Fused post-bmm: tril mask + dA + grad_beta sum + dtype cast."""
    NC, C, _ = dM_sys.shape
    dA = torch.empty_like(dM_sys)
    BLOCK_C = min(256, triton.next_power_of_2(C))
    _fused_post_bmm_kernel[(NC * C,)](
        dM_sys, A, beta.reshape(-1), grad_beta_part1.reshape(-1),
        dA, grad_beta_out.reshape(-1),
        NC=NC, C=C, BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return dA


@triton.jit
def _fused_bwd_elementwise_kernel(
    # W-wide inputs [NC*C, W]
    base_ret_ptr, base_dv_dw_ptr,
    kv_ptr, cumul_k_ptr, post_decay_ptr,
    d_kv_A_row_ptr, d_kv_A_col_ptr, d_kv_from_QK_ptr,
    # R-wide inputs [NC*C, R]
    base_go_msq_ptr, qv_ptr, cumul_q_ptr, d_qv_from_QK_ptr,
    # W-wide outputs
    grad_kv_ptr,       # bf16
    d_lck_total_ptr,   # f32
    d_rev_lck_ptr,     # f32
    d_g_from_pd_ptr,   # f32
    # R-wide outputs
    grad_qv_ptr,       # bf16
    d_lcq_total_ptr,   # f32
    # Dims
    NCxC: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
):
    """Fused backward Phase B elementwise chain.

    Fuses 23 separate elementwise ops on [NC*C, W] and [NC*C, R] tensors into
    one kernel.  Grid: (NC*C,) -- one program per (chunk, token).  Each program
    processes W write-entries and R read-entries.
    """
    t = tl.program_id(0)
    if t >= NCxC:
        return

    w_offs = tl.arange(0, W)
    r_offs = tl.arange(0, R)

    # --- Load W-wide inputs ---
    base_ret = tl.load(base_ret_ptr + t * W + w_offs)
    base_dv_dw = tl.load(base_dv_dw_ptr + t * W + w_offs)
    kv = tl.load(kv_ptr + t * W + w_offs)
    cumul_k = tl.load(cumul_k_ptr + t * W + w_offs)
    pd = tl.load(post_decay_ptr + t * W + w_offs)
    d_A_row = tl.load(d_kv_A_row_ptr + t * W + w_offs)
    d_A_col = tl.load(d_kv_A_col_ptr + t * W + w_offs)
    d_kv_QK = tl.load(d_kv_from_QK_ptr + t * W + w_offs)

    # --- Load R-wide inputs ---
    go_msq = tl.load(base_go_msq_ptr + t * R + r_offs)
    qv = tl.load(qv_ptr + t * R + r_offs)
    cumul_q = tl.load(cumul_q_ptr + t * R + r_offs)
    d_qv_QK = tl.load(d_qv_from_QK_ptr + t * R + r_offs)

    # --- Compute W-wide outputs ---
    d_kv_ret = base_ret * cumul_k
    d_lck_ret = base_ret * kv * cumul_k
    d_lck_A = kv * d_A_row - kv * d_A_col
    d_kv_A = d_A_row + d_A_col
    d_lck_QK = -kv * d_kv_QK
    d_kv_wv = base_dv_dw * pd
    d_pd_val = kv * base_dv_dw
    d_rev_lck = d_pd_val * pd
    d_g_pd = -d_pd_val * pd

    grad_kv = d_kv_A + d_kv_QK + d_kv_ret + d_kv_wv
    d_lck_total = d_lck_A + d_lck_QK + d_lck_ret

    # --- Compute R-wide outputs ---
    d_lcq_QK = qv * d_qv_QK
    d_lcq_inter = go_msq * qv * cumul_q
    grad_qv = d_qv_QK + go_msq * cumul_q
    d_lcq_total = d_lcq_QK + d_lcq_inter

    # --- Store outputs ---
    tl.store(grad_kv_ptr + t * W + w_offs, grad_kv)
    tl.store(d_lck_total_ptr + t * W + w_offs, d_lck_total)
    tl.store(d_rev_lck_ptr + t * W + w_offs, d_rev_lck)
    tl.store(d_g_from_pd_ptr + t * W + w_offs, d_g_pd)
    tl.store(grad_qv_ptr + t * R + r_offs, grad_qv)
    tl.store(d_lcq_total_ptr + t * R + r_offs, d_lcq_total)


def fused_bwd_elementwise(
    # W-wide inputs [NC, C, W]
    base_ret_all: torch.Tensor,
    base_dv_dw_all: torch.Tensor,
    wy_k_val_tm: torch.Tensor,
    wy_cumul_k_tm: torch.Tensor,
    wy_post_decay_tm: torch.Tensor,
    d_kv_A_row: torch.Tensor,
    d_kv_A_col: torch.Tensor,
    d_kv_from_QK: torch.Tensor,
    # R-wide inputs [NC, C, R]
    base_go_msq_all: torch.Tensor,
    wy_q_val_tm: torch.Tensor,
    wy_cumul_q_tm: torch.Tensor,
    d_qv_from_QK: torch.Tensor,
    # W-wide outputs (pre-allocated)
    grad_kv_tm: torch.Tensor,       # bf16
    d_lck_total_all: torch.Tensor,   # f32
    d_rev_lck_all: torch.Tensor,     # f32
    d_g_from_pd_all: torch.Tensor,   # f32
    # R-wide outputs (pre-allocated)
    grad_qv_tm: torch.Tensor,       # bf16
    d_lcq_total_all: torch.Tensor,   # f32
    num_writes: int,
    num_reads: int,
):
    """Fused backward Phase B elementwise chain.

    Replaces 23 separate elementwise kernel launches with a single Triton
    kernel. Each program handles one (chunk, token) pair, processing W
    write-entries and R read-entries.
    """
    NCxC = base_ret_all.shape[0] * base_ret_all.shape[1]
    _fused_bwd_elementwise_kernel[(NCxC,)](
        # W-wide inputs (contiguous [NC*C, W] views)
        base_ret_all, base_dv_dw_all,
        wy_k_val_tm, wy_cumul_k_tm, wy_post_decay_tm,
        d_kv_A_row, d_kv_A_col, d_kv_from_QK,
        # R-wide inputs
        base_go_msq_all, wy_q_val_tm, wy_cumul_q_tm, d_qv_from_QK,
        # W-wide outputs
        grad_kv_tm, d_lck_total_all, d_rev_lck_all, d_g_from_pd_all,
        # R-wide outputs
        grad_qv_tm, d_lcq_total_all,
        # Dims
        NCxC=NCxC, W=num_writes, R=num_reads,
    )


class GatedSparseMemoryWriteRead(torch.autograd.Function):
    """Fully-batched chunk-wise write-read with deltanet-style gating and
    exact intra-chunk recurrence via WY triangular solve.

    Steps:
    1. Compute per-slot cumulative log-decay for correct mem_read
    2. Build decay-weighted A = K_fwd @ K_rev^T via sparse matmul
    3. WY solve: (I + beta * A_lower) @ delta_v = delta_v_approx
    4. Post-decay writes to memory for correct M_end state
    5. Output = M_start * cumul_q + causal(QK) @ delta_v_exact

    This makes the output chunk_size-invariant.
    """

    @staticmethod
    def forward(
        ctx,
        memory: torch.Tensor,  # [num_slots, dim]
        k_idx: torch.Tensor,   # [B*T, num_writes]
        k_val: torch.Tensor,   # [B*T, num_writes]
        v: torch.Tensor,        # [B*T, dim]
        beta: torch.Tensor,     # [B*T, 1]
        g: torch.Tensor,        # [B*T, 1]
        q_idx: torch.Tensor,   # [B*T, num_reads]
        q_val: torch.Tensor,   # [B*T, num_reads]
        chunk_size: int = 256,
        intra_chunk: bool = True,  # kept for backward compat, always True
        num_memory_slots: int = 0,
        num_parallel: int = 1,
        normalize_memory: bool = False,  # kept for backward compat, unused
        snapshot_quant: str = "none",
        grad_final_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if chunk_size < 2:
            raise ValueError(
                f"chunk_size must be >= 2, got {chunk_size}. "
                "Token-by-token (chunk_size=1) path has been removed."
            )

        # Handle 3D inputs [P, T, ...] from num_parallel > 1
        input_was_3d = k_idx.ndim == 3
        if input_was_3d:
            P = num_parallel
            T_seq = k_idx.shape[1]
            k_idx = k_idx.reshape(P * T_seq, -1)
            k_val = k_val.reshape(P * T_seq, -1)
            v = v.reshape(P * T_seq, -1)
            beta = beta.reshape(P * T_seq, -1)
            g = g.reshape(P * T_seq, -1)
            q_idx = q_idx.reshape(P * T_seq, -1)
            q_val = q_val.reshape(P * T_seq, -1)

        B_T, num_writes = k_idx.shape
        num_slots, dim = memory.shape
        num_reads = q_idx.shape[1]
        device = memory.device
        dtype = memory.dtype

        # Compute slots_per_head for cumsum accumulator optimization.
        # When multi-head (H>1), each batch element's memory is BH*sph, and
        # k_idx has batch offsets added by the layer. For cumsum, we strip
        # these offsets so the accumulator only needs sph entries (fits L2).
        # num_memory_slots parameter is either sph (when H>1) or num_slots (when H=1).
        if num_memory_slots > 0:
            sph = num_memory_slots  # effective_num_slots = sph passed by layer
        else:
            sph = num_slots  # fallback
        BH = num_slots // sph if sph > 0 else 1

        readings = torch.empty(B_T, dim, device=device, dtype=dtype)

        # === Phase 1: Pre-compute all WY quantities batched (no memory dependency) ===
        num_chunks = (B_T + chunk_size - 1) // chunk_size
        C = chunk_size  # all chunks same size (last may be smaller, handled below)

        pad = num_chunks * C - B_T
        k_idx_p, k_val_p, v_p, beta_p, g_p, q_idx_p, q_val_p = _pad_inputs(
            k_idx, k_val, v, beta, g, q_idx, q_val, pad)

        k_idx_ch = k_idx_p.reshape(num_chunks, C, num_writes)    # [N, C, W]
        # Indices arrive pre-sorted from product key computation
        # Upcast bf16→float32 for stability, but keep float32/float64 as-is
        fwd_compute_dtype = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype
        k_val_ch = k_val_p.reshape(num_chunks, C, num_writes).to(fwd_compute_dtype)
        v_ch = v_p.reshape(num_chunks, C, dim).to(fwd_compute_dtype)
        beta_ch = beta_p.reshape(num_chunks, C, 1).to(fwd_compute_dtype)
        g_ch = g_p.reshape(num_chunks, C, -1).to(fwd_compute_dtype)  # [N, C, 1] or [N, C, W]
        q_idx_ch = q_idx_p.reshape(num_chunks, C, num_reads)
        q_val_ch = q_val_p.reshape(num_chunks, C, num_reads).to(fwd_compute_dtype)

        # === Phase 1: Pre-compute WY quantities (batched where possible) ===

        # Fused triple cumsum: forward K + cross Q + reverse K in ONE kernel
        # Replaces 3 separate cumsum calls, saving 2 kernel launches + redundant work.
        flat_k = k_idx_ch.reshape(num_chunks, C * num_writes)  # [N, C*W]
        flat_g = g_ch.expand(num_chunks, C, num_writes).reshape(num_chunks, C * num_writes)  # [N, C*W]
        flat_q = q_idx_ch.reshape(num_chunks, C * num_reads)  # [N, C*R]

        # Strip batch offsets from indices for cumsum — each chunk's indices
        # map to [0, sph) within one batch element. This reduces the cumsum
        # accumulator from [num_chunks, num_slots] to [num_chunks, sph],
        # fitting L2 cache (400 KB vs 4 MB per row at L05).
        if BH > 1:
            chunks_per_batch = num_chunks // BH
            batch_offsets = (torch.arange(num_chunks, device=device) // chunks_per_batch * sph).unsqueeze(1)
            flat_k_local = flat_k - batch_offsets
            flat_q_local = flat_q - batch_offsets
        else:
            flat_k_local = flat_k
            flat_q_local = flat_q

        log_cumul_k_flat, log_cumul_q_flat, rev_log_cumul_k_flat = _fused_triple_cumsum(
            flat_g, flat_k_local, flat_q_local, C, num_writes, num_reads, sph,
        )
        log_cumul_k_all = log_cumul_k_flat.reshape(num_chunks, C, num_writes)
        cumul_k_all = torch.exp(log_cumul_k_all)
        log_cumul_q_all = log_cumul_q_flat.reshape(num_chunks, C, num_reads)
        cumul_q_all = torch.exp(log_cumul_q_all)
        rev_log_cumul_k_all = rev_log_cumul_k_flat.reshape(num_chunks, C, num_writes)
        post_decay_all = torch.exp(rev_log_cumul_k_all - g_ch.expand(num_chunks, C, num_writes))

        # Per-chunk A and QK via causal gated sparse inner product (Triton)
        A_all = sparse_inner_product_gated(
            k_idx_ch, k_val_ch, log_cumul_k_all,
            k_idx_ch, k_val_ch, log_cumul_k_all,
        ).to(fwd_compute_dtype)
        QK_all = sparse_inner_product_gated(
            q_idx_ch, q_val_ch, log_cumul_q_all,
            k_idx_ch, k_val_ch, log_cumul_k_all,
            include_diag=True,
        ).to(fwd_compute_dtype)

        # Batched M_sys: [N, C, C] — fused Triton kernel (I + beta * A * tril_mask)
        beta_ch_det = beta_ch.detach()
        M_sys_all = fused_build_M_sys(beta_ch_det, A_all)

        # === Phase 1 complete ===

        # Pre-compute delta_v decomposition for Phase 2:
        # Consolidate 3 solve_triangular → 1 solve + derive.
        # M_sys_inv = solve(M_sys, I) reads M_sys once instead of 3 times.
        # Then: delta_v_const = M_sys_inv @ (beta * v)
        #       B_matrix = M_sys_inv * (-beta)  (column scaling)
        #       M_sys_inv_T = M_sys_inv^T
        eye = torch.eye(C, device=device, dtype=fwd_compute_dtype).unsqueeze(0).expand(num_chunks, -1, -1)
        M_sys_inv_all = torch.linalg.solve_triangular(
            M_sys_all, eye, upper=False,
        )  # [N, C, C]

        delta_v_const_all = torch.bmm(
            M_sys_inv_all, (beta_ch_det * v_ch),
        )  # [N, C, dim]

        B_matrix_all = M_sys_inv_all * (-beta_ch_det).transpose(-1, -2)  # [N, C, C] column scaling

        # K_eff for retrieved = einsum('cn,cnd->cd', K_eff, mem_read_raw)
        K_eff_all = (k_val_ch * cumul_k_all).detach()  # [N, C, W]

        M_sys_inv_T_all = M_sys_inv_all.transpose(-1, -2).contiguous()  # [N, C, C]

        # === Phase 2 pre-computation: move all loop-invariant work out ===

        # Pre-compute decay_per for all chunks (fused Triton kernel).
        # Detects duplicate slots within each token and masks them to 1.0,
        # non-duplicates get exp(g). Avoids [N, C, W, W] bool intermediate.
        decay_per_all = fused_dedup_decay(k_idx_ch, g_ch)

        # Pre-compute K_eff weights casted to memory dtype for fused gather
        K_eff_casted_all = K_eff_all.to(dtype)  # [N, C, W]

        # Pre-compute q_weights for all chunks
        q_weights_all = (q_val_ch * cumul_q_all).detach().to(dtype)  # [N, C, R]

        # Pre-compute QK with lower-triangular mask for all chunks
        QK_tril_all = torch.tril(QK_all)  # [N, C, C]

        # === Pre-compute metadata via sort-based approach ===
        # Sort entries by (chunk_id, slot_id) to group same-slot entries contiguously.
        # Then compute is_last/is_only/cumul from adjacent comparisons and segment products.
        # 1.9× faster than scatter_reduce approach (avoids allocating [N×num_slots] temp arrays).
        CW = C * num_writes
        chunk_offsets = torch.arange(num_chunks, device=device, dtype=torch.int32).unsqueeze(1) * num_slots  # [N, 1]
        sort_keys = (flat_k.to(torch.int32) + chunk_offsets).reshape(-1)  # [N*CW] int32
        sorted_keys, sort_perm = sort_keys.sort()

        # Segment boundaries: diff[i] = True where a new (chunk, slot) group starts
        seg_start = torch.ones(num_chunks * CW, device=device, dtype=torch.int32)
        seg_start[1:] = (sorted_keys[1:] != sorted_keys[:-1]).to(torch.int32)

        # is_last in sorted order: next entry starts a new segment (or this is the final entry)
        is_last_sorted = torch.ones(num_chunks * CW, device=device, dtype=torch.int32)
        is_last_sorted[:-1] = seg_start[1:]

        # is_only in sorted order: both first AND last in segment (singleton)
        is_only_sorted = seg_start & is_last_sorted

        # Segment-product for cumul_per_entry: product of decay factors within each (chunk, slot)
        seg_id = seg_start.long().cumsum(0) - 1  # 0-based segment ID
        n_segs = num_chunks * CW  # upper bound avoids GPU→CPU sync (.item())
        decay_sorted = decay_per_all.reshape(-1)[sort_perm]
        seg_product = torch.ones(n_segs, device=device, dtype=fwd_compute_dtype)
        seg_product.scatter_reduce_(0, seg_id, decay_sorted, reduce='prod', include_self=True)
        cumul_sorted = seg_product[seg_id]

        # Unsort: map results back to original entry order
        unsort_perm = torch.empty_like(sort_perm)
        unsort_perm[sort_perm] = torch.arange(num_chunks * CW, device=device)

        cumul_per_entry_all = cumul_sorted[unsort_perm].reshape(num_chunks, CW).to(dtype)
        is_last_all = is_last_sorted[unsort_perm].reshape(num_chunks, CW)
        is_only_all = is_only_sorted[unsort_perm].reshape(num_chunks, CW)
        del sort_keys, sorted_keys, sort_perm, seg_start, is_last_sorted, is_only_sorted
        del seg_id, decay_sorted, seg_product, cumul_sorted, unsort_perm

        # Pre-allocate output buffers for Phase 2 loop
        # Determine parallel batch dimensions
        P = num_parallel if (num_parallel > 1 and pad == 0 and num_chunks % num_parallel == 0) else 1
        Nt = num_chunks // P
        SC = P * C  # "super-chunk" size: tokens processed per time step

        inter_buf = torch.empty(SC, dim, device=device, dtype=fwd_compute_dtype)
        # Save delta_v and retrieved for backward — allocate in time-major order
        # so the Phase 2 loop writes directly without needing a permute copy.
        # Store in f32 (fwd_compute_dtype) to avoid per-step bf16 copy kernels.
        # This eliminates ~64 elementwise CUDA kernels from dtype casts.
        # Backward also benefits: no .to(compute_dtype) upcast needed.
        delta_v_all = torch.empty(num_chunks, C, dim, device=device, dtype=fwd_compute_dtype)
        retrieved_all = torch.empty(num_chunks, C, dim, device=device, dtype=fwd_compute_dtype)

        # === Permute Phase 1 outputs to time-major order for Phase 2 ===
        # Batch-first: [b0t0, b0t1, ..., b1t0, b1t1, ...] shape [P*Nt, C, ...]
        # Time-major:  [t0b0, t0b1, ..., t1b0, t1b1, ...] shape [Nt*P, C, ...]
        # For P=1 this is a no-op (identity permute).
        def _to_time_major(x):
            """Permute [P*Nt, ...] batch-first → [Nt*P, ...] time-major."""
            if P == 1:
                return x
            shape = x.shape
            return x.view(P, Nt, *shape[1:]).permute(1, 0, *range(2, len(shape) + 1)).contiguous().view(shape)

        k_idx_tm = _to_time_major(k_idx_ch)
        K_eff_tm = _to_time_major(K_eff_casted_all)
        q_idx_tm = _to_time_major(q_idx_ch)
        q_weights_tm = _to_time_major(q_weights_all)
        delta_v_const_tm = _to_time_major(delta_v_const_all)
        B_matrix_tm = _to_time_major(B_matrix_all)
        QK_tril_tm = _to_time_major(QK_tril_all)
        flat_k_tm = _to_time_major(flat_k)
        cumul_per_entry_tm = _to_time_major(cumul_per_entry_all)
        is_last_tm = _to_time_major(is_last_all)
        is_only_tm = _to_time_major(is_only_all)
        k_val_tm = _to_time_major(k_val_ch)
        post_decay_tm = _to_time_major(post_decay_all)

        # Pre-permute WY quantities to time-major for backward (avoids
        # redundant _to_time_major copies in backward).  For P=1 these are
        # no-ops (identity).  The batch-first originals are freed below to
        # avoid increasing peak memory.
        A_tm = _to_time_major(A_all)
        QK_tm = _to_time_major(QK_all)
        cumul_k_tm = _to_time_major(cumul_k_all)
        cumul_q_tm = _to_time_major(cumul_q_all)
        log_cumul_k_tm = _to_time_major(log_cumul_k_all)
        log_cumul_q_tm = _to_time_major(log_cumul_q_all)
        M_sys_inv_T_tm = _to_time_major(M_sys_inv_T_all)
        q_val_tm = _to_time_major(q_val_ch)
        beta_tm = _to_time_major(beta_ch)
        del A_all, QK_all, cumul_k_all, cumul_q_all
        del log_cumul_k_all, log_cumul_q_all, M_sys_inv_T_all
        del q_val_ch, beta_ch
        del k_idx_ch, q_idx_ch, k_val_ch, post_decay_all
        del cumul_per_entry_all, is_last_all, is_only_all

        # Pre-compute combined matrices for dual-output matmul in Phase 2:
        # readings = QK_tril @ (B_matrix @ retrieved + delta_v_const) + inter
        #          = (QK_tril @ B_matrix) @ retrieved + QK_tril @ delta_v_const + inter
        #          = QB @ retrieved + qk_dvc + inter
        # This decouples readings from delta_v, enabling a single dual-output kernel
        # that computes both delta_v and readings from the same B (retrieved).
        QB_tm = torch.bmm(QK_tril_tm, B_matrix_tm)  # [Nt*P, C, C]
        qk_dvc_tm = torch.bmm(QK_tril_tm, delta_v_const_tm)  # [Nt*P, C, dim]

        # Allocate all_mem_before_decay and readings directly in time-major order.
        # No permute needed — Phase 2 loop writes directly in time-major order.
        # For P=1, time-major == batch-first, so we reuse the existing tensors.
        total_snap_tokens = Nt * SC if P > 1 else B_T
        if snapshot_quant != "none":
            # Chunk-by-chunk quantization: only allocate bf16 for one chunk,
            # store quantized version for all chunks to avoid full bf16 materialization.
            snapshot_chunk_buf = torch.empty(SC, num_writes, dim, device=device, dtype=dtype)
            if snapshot_quant == "fp8":
                snap_q_tm = torch.empty(total_snap_tokens, num_writes, dim, device=device, dtype=torch.float8_e4m3fn)
                snap_s_tm = torch.empty(total_snap_tokens, num_writes, 1, device=device, dtype=torch.float32)
            else:  # int4
                snap_q_tm = torch.empty(total_snap_tokens, num_writes, dim // 2, device=device, dtype=torch.uint8)
                snap_s_tm = torch.empty(total_snap_tokens, num_writes, 1, device=device, dtype=torch.float32)
        else:
            if P > 1:
                all_mem_before_decay_tm = torch.empty(Nt * SC, num_writes, dim, device=device, dtype=dtype)
            else:
                all_mem_before_decay_tm = torch.empty(B_T, num_writes, dim, device=device, dtype=dtype)
        if P > 1:
            readings_tm = torch.empty(Nt * SC, dim, device=device, dtype=dtype)
        else:
            readings_tm = readings  # reuse — P=1 means time-major == original order

        # === Phase 2: Loop over Nt time steps, processing SC = P*C tokens each ===
        # 3 kernels per iteration: gather, dual_matmul (delta_v + readings), decay_write.
        # Pre-import CUDA gather module and hoist loop-invariant constants
        BLOCK_D = min(1024, triton.next_power_of_2(dim))
        _dm_grid = (triton.cdiv(C, 64), triton.cdiv(dim, 64), P)  # dual matmul grid
        if dtype == torch.bfloat16:
            try:
                from .cuda.warp_cooperative_gather_cuda import warp_cooperative_gather as _wc_gather
            except ImportError:
                _wc_gather = None
        else:
            _wc_gather = None

        # Pre-flatten FWD loop metadata for faster per-step slicing
        _fwd_flat_k = flat_k_tm.reshape(-1)          # [NC * CW]
        _fwd_cumul = cumul_per_entry_tm.reshape(-1)   # [NC * CW]
        _fwd_islast = is_last_tm.reshape(-1)          # [NC * CW]
        _fwd_isonly = is_only_tm.reshape(-1)          # [NC * CW]
        _fwd_kval = k_val_tm.reshape(-1, num_writes)  # [NC * C, W] — for decay kernel
        _fwd_pd = post_decay_tm.reshape(-1, num_writes)  # [NC * C, W]
        _fwd_CW = C * num_writes
        _gather_bt = 128 if SC <= 1024 else 256
        # Pre-flatten output buffers for direct per-step slicing
        _retrieved_flat = retrieved_all.reshape(-1, dim)  # [Nt*SC, dim]
        _delta_v_flat_fwd = delta_v_all.reshape(-1, dim)  # [Nt*SC, dim]

        for ti in range(Nt):
            tm_start = ti * P
            tm_end = tm_start + P
            tok_start = ti * SC
            tok_end = tok_start + SC
            cw_start = ti * _fwd_CW * P  # P chunks × CW entries per chunk — wait, CW = C*W per CHUNK
            # Actually: flat_k_tm is [NC, C*W], reshaped to [NC*C*W]. Per time step ti, P chunks.
            # Entries per step = P * C * W = SC * W = CW (where CW at level05 = 512*64 = 32768)
            cw_s = ti * SC * num_writes
            cw_e = cw_s + SC * num_writes
            sc_s = ti * SC
            sc_e = sc_s + SC

            # Use retrieved_all slice directly as gather output buffer (f32).
            retrieved_dst = _retrieved_flat[tok_start:tok_end]

            if snapshot_quant != "none":
                if _wc_gather is not None:
                    _wc_gather(memory.data,
                               k_idx_tm[tm_start:tm_end], K_eff_tm[tm_start:tm_end],
                               q_idx_tm[tm_start:tm_end], q_weights_tm[tm_start:tm_end],
                               snapshot_chunk_buf, retrieved_dst, inter_buf, block_threads=_gather_bt)
                else:
                    fused_snapshot_gather(memory.data,
                        k_idx_tm[tm_start:tm_end].reshape(SC, num_writes),
                        K_eff_tm[tm_start:tm_end].reshape(SC, num_writes),
                        q_idx_tm[tm_start:tm_end].reshape(SC, num_reads),
                        q_weights_tm[tm_start:tm_end].reshape(SC, num_reads),
                        snapshot_chunk_buf, retrieved_dst, inter_buf)
            else:
                if _wc_gather is not None:
                    _wc_gather(memory.data,
                               k_idx_tm[tm_start:tm_end], K_eff_tm[tm_start:tm_end],
                               q_idx_tm[tm_start:tm_end], q_weights_tm[tm_start:tm_end],
                               all_mem_before_decay_tm[tok_start:tok_end], retrieved_dst, inter_buf,
                               block_threads=_gather_bt)
                else:
                    fused_snapshot_gather(memory.data,
                        k_idx_tm[tm_start:tm_end].reshape(SC, num_writes),
                        K_eff_tm[tm_start:tm_end].reshape(SC, num_writes),
                        q_idx_tm[tm_start:tm_end].reshape(SC, num_reads),
                        q_weights_tm[tm_start:tm_end].reshape(SC, num_reads),
                        all_mem_before_decay_tm[tok_start:tok_end], retrieved_dst, inter_buf)

            # Dual matmul: compute delta_v and readings from retrieved in one kernel.
            # delta_v = B_matrix @ retrieved + delta_v_const          (f32 output)
            # readings = QB @ retrieved + inter + qk_dvc              (bf16 output)
            # Inline dual matmul — skip wrapper overhead
            delta_v_dst = delta_v_all[tm_start:tm_end]  # [P, C, dim] f32
            delta_v_dst_2d = _delta_v_flat_fwd[tok_start:tok_end]  # [SC, dim] alias
            _fused_dual_matmul_add_kernel[_dm_grid](
                B_matrix_tm[tm_start:tm_end], QB_tm[tm_start:tm_end],
                retrieved_dst.view(P, C, dim),
                delta_v_const_tm[tm_start:tm_end],
                inter_buf.view(P, C, dim),
                qk_dvc_tm[tm_start:tm_end],
                delta_v_dst,
                readings_tm[tok_start:tok_end].view(P, C, dim),
                stride_ap=C * C, stride_bp=C * dim, stride_cp=C * dim,
                M=C, K=C, N=dim,
                BLOCK_M=64, BLOCK_K=min(64, C), BLOCK_N=64,
            )

            # Decay + scatter write to memory
            snap_src = snapshot_chunk_buf if snapshot_quant != "none" else all_mem_before_decay_tm[tok_start:tok_end]
            # Inline decay+write kernel — skip wrapper overhead
            _fused_decay_and_write_kernel[(SC * num_writes,)](
                memory.data,
                snap_src.reshape(-1, dim),
                _fwd_flat_k[cw_s:cw_e],
                _fwd_cumul[cw_s:cw_e],
                _fwd_islast[cw_s:cw_e],
                _fwd_isonly[cw_s:cw_e],
                _fwd_kval[sc_s:sc_e],
                delta_v_dst_2d,
                _fwd_pd[sc_s:sc_e],
                CW=SC * num_writes, C_local=SC, W=num_writes, dim=dim, BLOCK_D=BLOCK_D,
                num_warps=1,
            )

            # Quantize snapshot after decay_and_write is done with it
            if snapshot_quant == "fp8":
                max_fp8 = torch.finfo(torch.float8_e4m3fn).max
                amax = snapshot_chunk_buf.abs().amax(dim=-1, keepdim=True).float()
                scale = (amax / max_fp8).clamp(min=1e-12)
                snap_q_tm[tok_start:tok_end] = (snapshot_chunk_buf / scale.to(dtype)).to(torch.float8_e4m3fn)
                snap_s_tm[tok_start:tok_end] = scale
            elif snapshot_quant == "int4":
                amax = snapshot_chunk_buf.abs().amax(dim=-1, keepdim=True).float()
                scale = (amax / 7.0).clamp(min=1e-12)
                x_int = torch.round(snapshot_chunk_buf / scale.to(dtype)).clamp(-8, 7).to(torch.int8)
                even = x_int[..., 0::2] & 0x0F
                odd = x_int[..., 1::2] & 0x0F
                snap_q_tm[tok_start:tok_end] = ((even << 4) | odd).to(torch.uint8)
                snap_s_tm[tok_start:tok_end] = scale

        # === Permute only readings back to batch-first (small tensor, needed for return) ===
        # all_mem_before_decay, delta_v_all, retrieved_all stay time-major — NO copies.
        # Backward indexes them via _bf_chunk_to_tm_token_offset.
        if P > 1:
            def _to_batch_first(x):
                shape = x.shape
                return x.view(Nt, P, *shape[1:]).permute(1, 0, *range(2, len(shape) + 1)).contiguous().view(shape)
            readings = _to_batch_first(readings_tm.view(num_chunks, C, dim)).view(B_T + pad, dim)[:B_T]

        # Compute residual = v - retrieved in-place into retrieved_all.
        # Backward only needs residual (not v and retrieved separately).
        # This saves one [NC, C, dim] f32 tensor read in the backward kernel.
        v_ch_tm = _to_time_major(v_ch)
        torch.sub(v_ch_tm, retrieved_all, out=retrieved_all)  # retrieved_all now holds residual
        residual_all = retrieved_all  # rename for clarity
        del v_ch_tm

        # Save snapshot (quantized or bf16) for backward
        if snapshot_quant != "none":
            ctx.save_for_backward(memory, k_idx, k_val, v, beta, g,
                                  q_idx, q_val, snap_q_tm, snap_s_tm)
        else:
            all_mem_before_decay = all_mem_before_decay_tm
            ctx.save_for_backward(memory, k_idx, k_val, v, beta, g,
                                  q_idx, q_val, all_mem_before_decay)
        ctx.snapshot_quant = snapshot_quant
        ctx.chunk_size = chunk_size
        ctx.num_memory_slots = num_memory_slots
        ctx.input_was_3d = input_was_3d
        ctx.num_parallel = num_parallel
        ctx.T_seq = T_seq if input_was_3d else 0
        ctx.P_fwd = P  # parallel batch P used in forward (for time-major snapshot indexing)
        ctx.Nt_fwd = Nt  # number of time steps in forward
        ctx._grad_final_memory = grad_final_memory
        # Save pre-computed WY quantities for backward
        ctx.wy_data = {
            'A_tm': A_tm, 'QK_tm': QK_tm,
            'cumul_k_tm': cumul_k_tm, 'cumul_q_tm': cumul_q_tm,
            'log_cumul_k_tm': log_cumul_k_tm,
            'log_cumul_q_tm': log_cumul_q_tm,
            'post_decay_tm': post_decay_tm,
            'M_sys_inv_T_tm': M_sys_inv_T_tm,
            'k_idx_tm': k_idx_tm, 'q_idx_tm': q_idx_tm,
            'k_val_tm': k_val_tm, 'q_val_tm': q_val_tm,
            'beta_tm': beta_tm,
            'num_chunks': num_chunks, 'C': C, 'pad': pad,
            'cumul_per_entry_tm': cumul_per_entry_tm,
            'is_last_tm': is_last_tm,
            'is_only_tm': is_only_tm,
            'delta_v_all': delta_v_all,
            'residual_all': residual_all,
        }
        return readings, torch.empty(0, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_readings, grad_delta_v_unused):
        """Fully-batched backward matching the WY-corrected forward."""
        snapshot_quant = ctx.snapshot_quant
        if snapshot_quant != "none":
            (memory, k_idx, k_val, v, beta, g,
             q_idx, q_val, snap_q, snap_s) = ctx.saved_tensors
            all_mem_before_decay = None  # lazy dequant — never materialize full bf16
        else:
            (memory, k_idx, k_val, v, beta, g,
             q_idx, q_val, all_mem_before_decay) = ctx.saved_tensors
            snap_q = snap_s = None
        chunk_size = ctx.chunk_size
        num_memory_slots = ctx.num_memory_slots
        B_T, num_writes = k_idx.shape
        num_slots, dim = memory.shape
        num_reads = q_idx.shape[1]
        device = memory.device
        dtype = memory.dtype

        # Recompute sph/BH for cumsum accumulator optimization (same as forward)
        if num_memory_slots > 0:
            sph = num_memory_slots
        else:
            sph = num_slots
        BH = num_slots // sph if sph > 0 else 1

        # grad_memory: gradient w.r.t. final memory state, propagated backward.
        # With CP, the wrapper sets _grad_final_memory on ctx to propagate
        # gradients from later CP ranks. Without CP, starts at zeros.
        grad_final_mem = getattr(ctx, '_grad_final_memory', None)
        if grad_final_mem is not None:
            grad_memory = grad_final_mem.clone()
        else:
            grad_memory = torch.zeros_like(memory)
        grad_k_val = torch.zeros_like(k_val)
        grad_v = torch.zeros_like(v)
        grad_beta = torch.zeros_like(beta)
        grad_g = torch.zeros_like(g)
        grad_q_val = torch.zeros_like(q_val)

        P_fwd = ctx.P_fwd
        Nt_fwd = ctx.Nt_fwd

        wy = ctx.wy_data

        # Interleaved Phase A + Phase B backward with time-major iteration.
        # Mirrors the forward's time-major structure: iterate by time step,
        # processing P (parallel batch) chunks per step simultaneously.
        # This eliminates the _bf_chunk_to_tm_token_offset indexing hack.
        #
        # Groups are ranges of time steps, processed from latest to earliest.
        # Each group does:
        #   Phase A (sequential by time step): memory ops, capture per-chunk data
        #   Phase B (batched over the group): analytical parameter gradients
        # This ensures earlier groups' chunk_dw includes later groups' grad_mem_read.

        pad = wy.get('pad', 0)
        C_wy = wy['C']
        num_wy_chunks = wy['num_chunks']
        P = P_fwd
        Nt = Nt_fwd
        SC = P * C_wy  # tokens per time step (super-chunk)

        k_idx_p, k_val_p, v_p, beta_p, g_p, q_idx_p, q_val_p = _pad_inputs(
            k_idx, k_val, v, beta, g, q_idx, q_val, pad)

        # WY data from forward is already float32; determine if we need upcasting
        bwd_compute_dtype = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype

        # === Permute padded inputs and grad_readings to time-major order ===
        # Batch-first: [b0t0, b0t1, ..., b1t0, b1t1, ...] shape [P*Nt, C, ...]
        # Time-major:  [t0b0, t0b1, ..., t1b0, t1b1, ...] shape [Nt*P, C, ...]
        # For P=1 this is an identity (no copy).
        def _to_time_major(x):
            """Permute [P*Nt, ...] batch-first -> [Nt*P, ...] time-major."""
            if P == 1:
                return x
            shape = x.shape
            return x.view(P, Nt, *shape[1:]).permute(1, 0, *range(2, len(shape) + 1)).contiguous().view(shape)

        # Permute padded inputs to time-major [Nt*P, C, ...]
        # k_idx and q_idx are reused from saved WY data (same content, avoids 2 permute copies)
        beta_tm_f32 = _to_time_major(beta_p.reshape(num_wy_chunks, C_wy, 1).to(bwd_compute_dtype))
        # q_val permute eliminated: use wy_q_val_tm (f32, from wy_data) directly
        # q_val_tm = _to_time_major(q_val[...])  — no longer needed

        # Permute grad_readings to time-major
        # .contiguous() is critical: grad_readings from .sum().backward() has
        # zero strides (expanded scalar), and Triton kernels use pointer arithmetic
        # that assumes contiguous layout.
        grad_readings_tm = _to_time_major(
            grad_readings[:num_wy_chunks * C_wy].reshape(num_wy_chunks, C_wy, dim).contiguous()
        )
        grad_readings_tm_f32 = grad_readings_tm.to(bwd_compute_dtype)

        # WY quantities are already time-major from forward — no permute needed
        wy_QK_tril_tm = wy['QK_tm']
        wy_cumul_k_tm = wy['cumul_k_tm']
        wy_cumul_q_tm = wy['cumul_q_tm']
        wy_post_decay_tm = wy['post_decay_tm']
        wy_k_val_tm = wy['k_val_tm']
        wy_q_val_tm = wy['q_val_tm']
        wy_beta_tm = wy['beta_tm']
        wy_M_sys_inv_T_tm = wy['M_sys_inv_T_tm']
        wy_A_tm = wy['A_tm']
        wy_log_cumul_k_tm = wy['log_cumul_k_tm']
        wy_log_cumul_q_tm = wy['log_cumul_q_tm']
        k_idx_tm = wy['k_idx_tm']   # reuse for flat_k_tm and k_idx_gb
        q_idx_tm = wy['q_idx_tm']   # reuse for q_idx_gb
        cumul_per_entry_tm = wy['cumul_per_entry_tm']
        is_last_tm = wy['is_last_tm']
        is_only_tm = wy['is_only_tm']

        # delta_v_all and residual_all are already time-major from forward (f32)
        delta_v_tm = wy['delta_v_all']    # [Nt*P, C, dim] f32
        residual_tm = wy['residual_all']  # [Nt*P, C, dim] f32 (v - retrieved)

        # Free wy_data dict — all needed quantities are now in local variables
        del wy
        ctx.wy_data = None
        # Free padded inputs — no longer needed
        del k_idx_p, k_val_p, v_p, beta_p, g_p, q_idx_p, q_val_p

        # flat_k in time-major order [Nt*P, C*W]
        flat_k_tm = k_idx_tm.reshape(num_wy_chunks, C_wy * num_writes)

        # Pre-compute d_dva_readings = M_sys_inv_T @ tril(QK)^T @ grad_readings
        # Fused: compute M_sys_inv_T @ tril(QK)^T first (37.7 MB, fits L2),
        # then multiply with grad_readings. Avoids 302 MB intermediate round-trip.
        QK_tril_T_tm = wy_QK_tril_tm.transpose(-1, -2).contiguous()
        # Pre-convert indices to int32 for SIP backward (overlap with d_dva bmms)
        k_idx_i32 = k_idx_tm.contiguous().to(torch.int32)
        q_idx_i32 = q_idx_tm.contiguous().to(torch.int32)
        d_dva_readings_tm = torch.bmm(
            torch.bmm(wy_M_sys_inv_T_tm, QK_tril_T_tm),
            grad_readings_tm_f32,
        )  # [Nt*P, C, dim]

        # Group size in time steps (each time step has P chunks)
        bwd_group_size_chunks = min(num_wy_chunks, max(1, 128 * 32 * 64 // (C_wy * max(num_writes, num_reads))))
        # Convert to time steps (round up to ensure at least 1 time step per group)
        bwd_group_size_ts = max(1, bwd_group_size_chunks // P)

        # Pre-allocate cumsum accumulators (reused) and batched cumsum inputs
        max_N_gb = bwd_group_size_ts * P
        _cumsum_accum = torch.empty(num_wy_chunks, sph, device=device, dtype=torch.float32)
        _cumsum_accum2 = torch.empty(num_wy_chunks, sph, device=device, dtype=torch.float32)
        _cumsum_accum3 = torch.empty(num_wy_chunks, sph, device=device, dtype=torch.float32)

        # Pre-allocate buffers for deferred Step 11 cumsums (batched after group loop)
        d_lck_total_all = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=torch.float32)
        d_rev_lck_all = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=torch.float32)
        d_lcq_total_all = torch.empty(num_wy_chunks, C_wy, num_reads, device=device, dtype=torch.float32)
        d_g_from_pd_all = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=torch.float32)

        # Phase A outputs for deferred batched Phase B Steps 1-4
        g_width = g.shape[-1]  # 1 or num_writes
        d_dv_writes_all = torch.empty(num_wy_chunks, C_wy, dim, device=device, dtype=torch.float32)
        base_dv_dw_all = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=torch.float32)
        base_ret_all = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=torch.float32)
        base_go_msq_all = torch.empty(num_wy_chunks, C_wy, num_reads, device=device, dtype=torch.float32)
        grad_g_decay_all = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=torch.float32)
        # Pre-allocate d_dva_all: Phase A writes per-step, skips Phase B recomputation
        d_dva_all = torch.empty(num_wy_chunks, C_wy, dim, device=device, dtype=torch.float32)

        # Allocate output gradient buffers in time-major chunk order [Nt*P, C, ...]
        # grad_g_tm needs zeros (accumulated via +=), others are fully overwritten
        grad_kv_tm = torch.empty(num_wy_chunks, C_wy, num_writes, device=device, dtype=dtype)
        grad_qv_tm = torch.empty(num_wy_chunks, C_wy, num_reads, device=device, dtype=dtype)
        grad_v_tm = torch.empty(num_wy_chunks, C_wy, dim, device=device, dtype=dtype)
        grad_beta_tm = torch.empty(num_wy_chunks, C_wy, 1, device=device, dtype=dtype)
        grad_g_tm = torch.zeros(num_wy_chunks, C_wy, g_width, device=device, dtype=dtype)

        # Pre-compute weight products once for ALL chunks (not per-group)
        _kv_pd_all = wy_k_val_tm * wy_post_decay_tm   # [NC, C, W] f32
        _kv_ck_all = wy_k_val_tm * wy_cumul_k_tm       # [NC, C, W] f32
        _neg_beta_all = -wy_beta_tm                     # [NC, C, 1] f32
        _q_weights_all_f32 = wy_q_val_tm * wy_cumul_q_tm  # [NC, C, R] f32

        # Process groups from latest to earliest (in time steps)
        for gb_end_ts in range(Nt, 0, -bwd_group_size_ts):
            gb_start_ts = max(0, gb_end_ts - bwd_group_size_ts)
            # Chunk range in time-major order
            gb_start_chunk = gb_start_ts * P
            gb_end_chunk = gb_end_ts * P
            N_gb = gb_end_chunk - gb_start_chunk  # number of chunks in group
            gb = slice(gb_start_chunk, gb_end_chunk)

            # === Phase A pre-computation: batch everything that doesn't depend on grad_memory ===
            k_idx_gb = k_idx_tm[gb]  # [N_gb, C, W] time-major
            q_idx_gb = q_idx_tm[gb]  # [N_gb, C, R] time-major
            flat_k_gb = flat_k_tm[gb]  # [N_gb, C*W] time-major
            cumul_per_entry_gb = cumul_per_entry_tm[gb]  # [N_gb, C*W] time-major
            is_last_gb = is_last_tm[gb]  # [N_gb, C*W] int32
            is_only_gb = is_only_tm[gb]  # [N_gb, C*W] int32

            # Pre-computed weight products: just slice from global tensors (views, no compute)
            kv_pd_gb = _kv_pd_all[gb]
            kv_ck_gb = _kv_ck_all[gb]
            neg_beta_gb = _neg_beta_all[gb]
            q_weights_gb_f32 = _q_weights_all_f32[gb]
            grad_readings_gb_f32 = grad_readings_tm_f32[gb]  # [N_gb, C, dim]

            # Phase A outputs: views into global buffers for deferred batched Phase B
            d_dv_writes_gb = d_dv_writes_all[gb]
            base_dv_dw_gb = base_dv_dw_all[gb]
            base_ret_gb = base_ret_all[gb]
            base_go_msq_gb = base_go_msq_all[gb]
            grad_g_decay_gb = grad_g_decay_all[gb]
            # Pre-allocate buffers for the SC-sized operations per time step
            grad_g_decay_buf = torch.empty(SC * num_writes, device=device, dtype=torch.float32)
            d_dv_writes_buf = torch.empty(SC, dim, device=device, dtype=torch.float32)
            base_dv_dw_buf = torch.empty(SC * num_writes, device=device, dtype=torch.float32)
            base_ret_buf = torch.empty(SC * num_writes, device=device, dtype=torch.float32)
            chunk_dw_buf = torch.empty(SC * num_writes, dim, device=device, dtype=dtype)  # per-step, reused
            CW = SC * num_writes
            BLOCK_D = min(1024, triton.next_power_of_2(dim))
            _gr_nw = 1 if SC >= 1024 else 16  # GR kernel num_warps (loop-invariant)
            # Pre-flatten metadata to avoid per-step slice+reshape
            flat_k_gb_flat = flat_k_gb.reshape(-1)  # [N_gb * CW]
            cumul_gb_flat = cumul_per_entry_gb.reshape(-1)
            islast_gb_flat = is_last_gb.reshape(-1)
            isonly_gb_flat = is_only_gb.reshape(-1)
            kv_pd_gb_flat = kv_pd_gb.reshape(N_gb * SC // P, num_writes)  # already [N_gb, C, W] → flatten first 2 dims
            kv_ck_gb_flat = kv_ck_gb.reshape(-1)
            neg_beta_gb_flat = neg_beta_gb.reshape(-1)
            q_idx_gb_flat = q_idx_gb.reshape(-1)
            q_weights_gb_flat = q_weights_gb_f32.reshape(-1)
            grad_readings_gb_flat = grad_readings_gb_f32.reshape(N_gb * SC // P, dim)
            base_go_msq_gb_flat = base_go_msq_gb.reshape(-1)
            # Pre-flatten Phase A output buffers (write directly from phaseA kernel)
            base_dv_dw_gb_flat = base_dv_dw_gb.reshape(-1)
            base_ret_gb_flat = base_ret_gb.reshape(-1)
            grad_g_decay_gb_flat = grad_g_decay_gb.reshape(-1)
            # Hoist snapshot reshape out of per-step loop (same every step)
            if snapshot_quant == "none":
                _snap_flat = all_mem_before_decay.reshape(-1, dim)
            # Pre-flatten 3D tensors to 2D for direct integer slicing in per-step loop
            _delta_v_flat = delta_v_tm.reshape(-1, dim)  # [Nt*SC, dim]
            _d_dv_writes_flat = d_dv_writes_all.reshape(-1, dim)  # [Nt*SC, dim]

            for ti in range(gb_end_ts - 1, gb_start_ts - 1, -1):
                ti_local = ti - gb_start_ts
                ch_start = ti_local * P
                ch_end = ch_start + P
                ch_sl = slice(ch_start, ch_end)
                cw_start = ti_local * CW
                cw_end = cw_start + CW
                ch_sl = slice(ch_start, ch_end)

                # Token offset in all_mem_before_decay (time-major, contiguous)
                tok_start = ti * SC
                tok_end = tok_start + SC
                chunk_offset = tok_start * num_writes

                # Gather flat_k for all P chunks at this time step
                flat_k_ts = flat_k_gb_flat[cw_start:cw_end]
                cumul_ts = cumul_gb_flat[cw_start:cw_end]
                is_last_ts = islast_gb_flat[cw_start:cw_end]
                is_only_ts = isonly_gb_flat[cw_start:cw_end]

                # Global chunk indices in time-major order
                tm_chunk_start = ti * P
                tm_chunk_end = tm_chunk_start + P

                # Step 1+2: GR writes directly to d_dv_writes_all (skip intermediate buf + copy)
                d_dv_writes_dst = _d_dv_writes_flat[tok_start:tok_end]
                sc_start = ti_local * SC
                sc_end = sc_start + SC
                _fused_gather_weighted_reduce_kernel[(SC,)](
                    d_dv_writes_dst, chunk_dw_buf,
                    kv_pd_gb_flat[sc_start:sc_end],
                    grad_memory, flat_k_ts,
                    SC=SC, W=num_writes, dim=dim, BLOCK_D=BLOCK_D,
                    STORE_GATHER=True,
                    num_warps=_gr_nw,
                )

                # Step 3: Compute d_dva = d_dva_readings + M_sys_inv_T @ d_dv_writes.
                torch.baddbmm(
                    d_dva_readings_tm[tm_chunk_start:tm_chunk_end],
                    wy_M_sys_inv_T_tm[tm_chunk_start:tm_chunk_end],
                    d_dv_writes_dst.view(P, C_wy, dim),
                    beta=1.0,
                    alpha=1.0,
                    out=d_dva_all[tm_chunk_start:tm_chunk_end],
                )  # [P, C, dim]
                d_dva_buf = d_dva_all[tm_chunk_start:tm_chunk_end].reshape(SC, dim)

                # phaseA writes directly to pre-flattened output views
                if snapshot_quant != "none":
                    tok_sl_ts = slice(ti * SC, (ti + 1) * SC)
                    snap_ts = _dequant_snapshot_slice(snap_q, snap_s, snapshot_quant, tok_sl_ts, dtype)
                    snap_flat = snap_ts.reshape(-1, dim)
                    kernel_chunk_offset = 0
                else:
                    snap_flat = _snap_flat
                    kernel_chunk_offset = chunk_offset
                _fused_phaseA_combined_write_kernel[(CW,)](
                    grad_memory, memory.data, snap_flat,
                    flat_k_ts, cumul_ts, is_last_ts, is_only_ts,
                    chunk_dw_buf, d_dva_buf,
                    kv_ck_gb_flat[cw_start:cw_end],
                    neg_beta_gb_flat[sc_start:sc_end],
                    grad_g_decay_gb_flat[cw_start:cw_end],
                    base_dv_dw_gb_flat[cw_start:cw_end],
                    base_ret_gb_flat[cw_start:cw_end],
                    _delta_v_flat[tok_start:tok_end],
                    chunk_offset=kernel_chunk_offset, CW=CW, W=num_writes, dim=dim, BLOCK_D=BLOCK_D,
                    num_warps=1,
                )

                # All outputs written directly by phaseA kernel — no copies needed

                # Store contractions
                # d_dv_writes already written directly to output by GR kernel
                # base_dv_dw and base_ret written directly by phaseA kernel

                # Fused: gather_dot + q scatter in one pass over q indices
                # Inline GDS kernel — skip wrapper overhead
                qr_start = ti_local * SC * num_reads
                qr_end = qr_start + SC * num_reads
                _fused_gather_dot_and_scatter_kernel[(SC * num_reads,)](
                    memory.data, grad_memory,
                    q_idx_gb_flat[qr_start:qr_end],
                    grad_readings_gb_flat[sc_start:sc_end],
                    q_weights_gb_flat[qr_start:qr_end],
                    base_go_msq_gb_flat[qr_start:qr_end],
                    NC=SC, R=num_reads, dim=dim, BLOCK_D=BLOCK_D,
                    num_warps=1 if SC >= 1024 else 4,
                )

            # Phase B Steps 1-4 deferred — run batched after all Phase A groups.
            # Deferred sum: reduce per-entry grad_g_decay to per-token (once per group, not per step)
            if g_width == 1:
                grad_g_tm[gb] = grad_g_decay_gb.sum(dim=-1, keepdim=True).to(dtype)
            else:
                grad_g_tm[gb] = grad_g_decay_gb.to(dtype)

        # === BATCHED Phase B Steps 1-4: bmms + fused elementwise over ALL chunks ===
        d_dv_all = torch.baddbmm(
            d_dv_writes_all,
            QK_tril_T_tm, grad_readings_tm_f32,
            beta=1.0,
            alpha=1.0,
        )
        # d_dva_all already populated per-step in Phase A (skip redundant bmm)
        # Fused pre-bmm with pre-computed residual: dv_approx + grad_v + grad_beta_part1
        # Uses residual (v - retrieved) saved from forward, reads 2 tensors instead of 3
        delta_v_f32 = delta_v_tm
        dv_approx_all, grad_beta_part1 = fused_pre_bmm_residual(
            residual_tm, beta_tm_f32, d_dva_all, grad_v_tm,
        )
        dM_sys_inv_all = torch.bmm(d_dv_all, dv_approx_all.transpose(-1, -2))
        _dM_tmp = torch.bmm(wy_M_sys_inv_T_tm, dM_sys_inv_all)
        dM_sys_all = torch.baddbmm(
            torch.empty_like(_dM_tmp), _dM_tmp, wy_M_sys_inv_T_tm,
            beta=0.0, alpha=-1.0,
        )
        del _dM_tmp
        # Post-bmm: dM_lower + dA + grad_beta (reverted to PyTorch ops for correctness)
        tril_mask_f = torch.tril(torch.ones(1, C_wy, C_wy, device=device, dtype=torch.float32), diagonal=-1)
        dM_lower_all = dM_sys_all * tril_mask_f
        grad_beta_part1 += (wy_A_tm * dM_lower_all).sum(dim=-1, keepdim=True)
        grad_beta_tm[:] = grad_beta_part1.to(dtype)
        dA_all = beta_tm_f32 * dM_lower_all

        # === BATCHED Phase B Steps 5-6: sparse_ip_bwd over ALL chunks ===
        d_kv_A_row, d_kv_A_col = _sparse_ip_gated_bwd_vals(
            k_idx_i32, wy_k_val_tm, wy_log_cumul_k_tm,
            k_idx_i32, wy_k_val_tm, wy_log_cumul_k_tm,
            dA_all, C_wy, causal_mode=1)

        dQK_all = torch.tril(torch.bmm(grad_readings_tm_f32, delta_v_f32.transpose(-1, -2)))
        d_qv_from_QK, d_kv_from_QK = _sparse_ip_gated_bwd_vals(
            q_idx_i32, wy_q_val_tm, wy_log_cumul_q_tm,
            k_idx_i32, wy_k_val_tm, wy_log_cumul_k_tm,
            dQK_all, C_wy, causal_mode=2)

        # === BATCHED Phase B Steps 7-10 ===
        # Fused: 23 elementwise ops → 1 Triton kernel
        fused_bwd_elementwise(
            base_ret_all, base_dv_dw_all,
            wy_k_val_tm, wy_cumul_k_tm, wy_post_decay_tm,
            d_kv_A_row, d_kv_A_col, d_kv_from_QK,
            base_go_msq_all, wy_q_val_tm, wy_cumul_q_tm, d_qv_from_QK,
            grad_kv_tm, d_lck_total_all, d_rev_lck_all, d_g_from_pd_all,
            grad_qv_tm, d_lcq_total_all,
            num_writes, num_reads,
        )

        # === Batched Step 11: d_g via fused triple cumsum over ALL chunks ===
        # Single kernel launch with 3 accumulators replaces 3 separate launches.
        # All 3 cumsums run simultaneously in one bidirectional+cross pass.
        flat_k_all = k_idx_tm.reshape(num_wy_chunks, C_wy * num_writes)
        flat_q_all = q_idx_tm.reshape(num_wy_chunks, C_wy * num_reads)

        # Strip batch offsets for cumsum (same as forward, adjusted for time-major order)
        if BH > 1:
            if P > 1:
                # Time-major order: [t0b0, t0b1, ..., t1b0, ...] — batch element = i % P
                batch_offsets_bwd = (torch.arange(num_wy_chunks, device=device) % P * sph).unsqueeze(1)
            else:
                # Batch-first order: [b0t0, b0t1, ..., b1t0, ...] — batch element = i // chunks_per_batch
                chunks_per_batch_bwd = num_wy_chunks // BH
                batch_offsets_bwd = (torch.arange(num_wy_chunks, device=device) // chunks_per_batch_bwd * sph).unsqueeze(1)
            flat_k_all_local = flat_k_all - batch_offsets_bwd
            flat_q_all_local = flat_q_all - batch_offsets_bwd
        else:
            flat_k_all_local = flat_k_all
            flat_q_all_local = flat_q_all

        d_g_fwd, d_g_rev, d_g_cq = _fused_bwd_triple_cumsum(
            d_lck_total_all.reshape(num_wy_chunks, C_wy * num_writes),
            d_rev_lck_all.reshape(num_wy_chunks, C_wy * num_writes),
            d_lcq_total_all.reshape(num_wy_chunks, C_wy * num_reads),
            flat_k_all_local, flat_q_all_local,
            C_wy, num_writes, num_reads, sph,
            accum=_cumsum_accum,
            accum2=_cumsum_accum2,
            accum3=_cumsum_accum3,
        )
        d_g_fwd = d_g_fwd.reshape(num_wy_chunks, C_wy, num_writes)
        d_g_rev = d_g_rev.reshape(num_wy_chunks, C_wy, num_writes)
        d_g_cq = d_g_cq.reshape(num_wy_chunks, C_wy, num_writes)

        # Total grad_g = Phase A grad_g_decay + Phase B (4 components)
        grad_g_all = d_g_fwd + d_g_rev + d_g_from_pd_all + d_g_cq
        if g_width == 1:
            grad_g_all = grad_g_all.sum(dim=-1, keepdim=True)
        # grad_g_decay was already stored in grad_g_tm during Phase A
        grad_g_tm += grad_g_all.to(dtype)

        # === Permute gradients from time-major back to batch-first flat order ===
        def _to_batch_first(x):
            """Permute [Nt*P, ...] time-major -> [P*Nt, ...] batch-first."""
            if P == 1:
                return x
            shape = x.shape
            return x.view(Nt, P, *shape[1:]).permute(1, 0, *range(2, len(shape) + 1)).contiguous().view(shape)

        grad_k_val = _to_batch_first(grad_kv_tm).reshape(-1, num_writes)[:B_T]
        grad_g = _to_batch_first(grad_g_tm).reshape(-1, g_width)[:B_T]
        grad_q_val = _to_batch_first(grad_qv_tm).reshape(-1, num_reads)[:B_T]
        grad_v = _to_batch_first(grad_v_tm).reshape(-1, dim)[:B_T]
        grad_beta = _to_batch_first(grad_beta_tm).reshape(-1, 1)[:B_T]

        # Reshape gradients back to 3D if inputs were 3D
        if ctx.input_was_3d:
            P_out = ctx.num_parallel
            T_out = ctx.T_seq
            grad_k_val = grad_k_val.view(P_out, T_out, -1)
            grad_v = grad_v.view(P_out, T_out, -1)
            grad_beta = grad_beta.view(P_out, T_out, -1)
            grad_g = grad_g.view(P_out, T_out, -1)
            grad_q_val = grad_q_val.view(P_out, T_out, -1)

        return (grad_memory, None, grad_k_val, grad_v, grad_beta, grad_g,
                None, grad_q_val, None, None, None, None, None, None, None)

@triton.jit
def _scatter_decay_from_snapshot_kernel(
    memory_ptr,       # [num_slots, dim]
    snapshot_ptr,     # [SCW, dim] contiguous
    flat_k_ptr,       # [P, CW] may be strided
    cumul_ptr,        # [P, CW] may be strided
    stride_fk_b,      # batch stride for flat_k
    stride_cu_b,      # batch stride for cumul
    CW_local: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter decayed snapshot to memory with strided batch support.
    memory[k] = snapshot[i] * cumul[i]. Grid: (P*CW_local,)
    """
    i = tl.program_id(0)
    b = i // CW_local
    i_local = i % CW_local
    slot = tl.load(flat_k_ptr + b * stride_fk_b + i_local).to(tl.int64)
    c = tl.load(cumul_ptr + b * stride_cu_b + i_local)
    for d_start in range(0, dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < dim
        snap = tl.load(snapshot_ptr + i * dim + d_offs, mask=mask, other=0.0)
        tl.store(memory_ptr + slot * dim + d_offs, snap * c, mask=mask)

def scatter_decay_from_snapshot(
    memory: torch.Tensor,      # [num_slots, dim]
    snapshot: torch.Tensor,    # [SC, W, dim] contiguous snapshot
    flat_k: torch.Tensor,     # [C*W] or [P, C*W] — may be strided
    cumul_per_entry: torch.Tensor,  # [C*W] or [P, C*W]
):
    """Apply decay using pre-gathered snapshot. Accepts strided batch inputs."""
    if flat_k.ndim == 2:
        P, CW_local = flat_k.shape
        SCW = P * CW_local
        stride_fk_b = flat_k.stride(0)
        stride_cu_b = cumul_per_entry.stride(0)
    else:
        SCW = flat_k.shape[0]
        CW_local = SCW
        stride_fk_b = stride_cu_b = 0
    dim = memory.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(dim))
    _scatter_decay_from_snapshot_kernel[(SCW,)](
        memory, snapshot.reshape(-1, dim),
        flat_k, cumul_per_entry,
        stride_fk_b=stride_fk_b, stride_cu_b=stride_cu_b,
        CW_local=CW_local, dim=dim, BLOCK_D=BLOCK_D, num_warps=1,
    )

@triton.jit
def _scatter_cumsum_fwd_kernel(
    flat_g_ptr, flat_k_ptr, result_ptr, accum_ptr,
    B_dim: tl.constexpr, C_dim: tl.constexpr, W_dim: tl.constexpr,
    num_slots: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """Forward scatter cumsum: scan time 0 -> C-1. Stores f32 results."""
    bid = tl.program_id(0)
    if bid >= B_dim:
        return
    accum_base = bid * num_slots
    data_base = bid * C_dim * W_dim
    w_offs = tl.arange(0, BLOCK_W)
    mask = w_offs < W_dim
    for t in tl.static_range(C_dim):
        pos = t * W_dim + w_offs
        slots = tl.load(flat_k_ptr + data_base + pos, mask=mask)
        g_vals = tl.load(flat_g_ptr + data_base + pos, mask=mask).to(tl.float32)
        old = tl.atomic_add(accum_ptr + accum_base + slots, g_vals, mask=mask)
        cumul = old + g_vals
        tl.store(result_ptr + data_base + pos, cumul, mask=mask)

@triton.jit
def _scatter_cumsum_rev_kernel(
    flat_g_ptr, flat_k_ptr, result_ptr, accum_ptr,
    B_dim: tl.constexpr, C_dim: tl.constexpr, W_dim: tl.constexpr,
    num_slots: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """Reverse scatter cumsum: scan time C-1 -> 0. Stores f32 results."""
    bid = tl.program_id(0)
    if bid >= B_dim:
        return
    accum_base = bid * num_slots
    data_base = bid * C_dim * W_dim
    w_offs = tl.arange(0, BLOCK_W)
    mask = w_offs < W_dim
    for t_off in tl.static_range(C_dim):
        t = C_dim - 1 - t_off
        pos = t * W_dim + w_offs
        slots = tl.load(flat_k_ptr + data_base + pos, mask=mask)
        g_vals = tl.load(flat_g_ptr + data_base + pos, mask=mask).to(tl.float32)
        old = tl.atomic_add(accum_ptr + accum_base + slots, g_vals, mask=mask)
        cumul = old + g_vals
        tl.store(result_ptr + data_base + pos, cumul, mask=mask)

@triton.jit
def _scatter_cross_cumsum_fwd_kernel(
    flat_g_ptr, flat_k_ptr, flat_q_ptr,
    result_k_ptr, result_q_ptr, accum_ptr,
    B_dim: tl.constexpr, C_dim: tl.constexpr,
    W_dim: tl.constexpr, R_dim: tl.constexpr,
    num_slots: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """Forward cross cumsum: k-scatter at integer times, q-gather at half-integer times."""
    bid = tl.program_id(0)
    if bid >= B_dim:
        return
    accum_base = bid * num_slots
    k_base = bid * C_dim * W_dim
    q_base = bid * C_dim * R_dim
    w_offs = tl.arange(0, BLOCK_W)
    r_offs = tl.arange(0, BLOCK_R)
    w_mask = w_offs < W_dim
    r_mask = r_offs < R_dim
    for t in tl.static_range(C_dim):
        k_pos = t * W_dim + w_offs
        k_slots = tl.load(flat_k_ptr + k_base + k_pos, mask=w_mask)
        g_vals = tl.load(flat_g_ptr + k_base + k_pos, mask=w_mask).to(tl.float32)
        old_k = tl.atomic_add(accum_ptr + accum_base + k_slots, g_vals, mask=w_mask)
        k_cumul = old_k + g_vals
        tl.store(result_k_ptr + k_base + k_pos, k_cumul, mask=w_mask)
        tl.debug_barrier()
        q_pos = t * R_dim + r_offs
        q_slots = tl.load(flat_q_ptr + q_base + q_pos, mask=r_mask)
        q_cumul = tl.load(accum_ptr + accum_base + q_slots, mask=r_mask)
        tl.store(result_q_ptr + q_base + q_pos, q_cumul, mask=r_mask)

@triton.jit
def _scatter_cross_cumsum_rev_kernel(
    d_lcq_ptr, flat_k_ptr, flat_q_ptr,
    result_k_ptr, accum_ptr,
    B_dim: tl.constexpr, C_dim: tl.constexpr,
    W_dim: tl.constexpr, R_dim: tl.constexpr,
    num_slots: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """Reverse cross cumsum for backward d_g computation.

    Scans from t=C-1 to t=0. At each step:
      1. Scatter q values FIRST (q at half-integer time t+0.5 has lower
         reverse_time than k at integer time t)
      2. Then gather k results (k sees q at same t and all later t values)
    """
    bid = tl.program_id(0)
    if bid >= B_dim:
        return
    accum_base = bid * num_slots
    k_base = bid * C_dim * W_dim
    q_base = bid * C_dim * R_dim
    w_offs = tl.arange(0, BLOCK_W)
    r_offs = tl.arange(0, BLOCK_R)
    w_mask = w_offs < W_dim
    r_mask = r_offs < R_dim
    for t_off in tl.static_range(C_dim):
        t = C_dim - 1 - t_off
        # 1. Scatter q values first (lower reverse_time at same t)
        q_pos = t * R_dim + r_offs
        q_slots = tl.load(flat_q_ptr + q_base + q_pos, mask=r_mask)
        q_vals = tl.load(d_lcq_ptr + q_base + q_pos, mask=r_mask).to(tl.float32)
        tl.atomic_add(accum_ptr + accum_base + q_slots, q_vals, mask=r_mask)
        tl.debug_barrier()
        # 2. Then gather k results (sees q at same t and all later t)
        k_pos = t * W_dim + w_offs
        k_slots = tl.load(flat_k_ptr + k_base + k_pos, mask=w_mask)
        k_cumul = tl.load(accum_ptr + accum_base + k_slots, mask=w_mask)
        tl.store(result_k_ptr + k_base + k_pos, k_cumul, mask=w_mask)

def _segmented_cumsum_scatter(
    values: torch.Tensor, slots: torch.Tensor,
    C: int, W: int, num_slots: int, reverse: bool = False,
    accum: torch.Tensor = None,
) -> torch.Tensor:
    """Scatter-based segmented cumsum for structured time layouts. f32 output."""
    B, N = values.shape
    assert N == C * W, f"N={N} != C*W={C*W}"
    device = values.device
    if accum is None:
        accum = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    else:
        accum[:B].zero_()
    result = torch.empty(B, N, device=device, dtype=torch.float32)
    BLOCK_W = triton.next_power_of_2(W)
    kernel = _scatter_cumsum_rev_kernel if reverse else _scatter_cumsum_fwd_kernel
    kernel[(B,)](values, slots.to(torch.int64), result, accum, B, C, W, num_slots, BLOCK_W)
    return result

def _segmented_cross_cumsum_scatter(
    flat_g: torch.Tensor, flat_k: torch.Tensor, flat_q: torch.Tensor,
    C: int, W: int, R: int, num_slots: int,
) -> torch.Tensor:
    """Scatter-based forward cross cumsum (k+q interleaved)."""
    B = flat_g.shape[0]
    device = flat_g.device
    accum = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    result_k = torch.empty(B, C * W, device=device, dtype=flat_g.dtype)
    result_q = torch.empty(B, C * R, device=device, dtype=flat_g.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    BLOCK_R = triton.next_power_of_2(R)
    _scatter_cross_cumsum_fwd_kernel[(B,)](
        flat_g, flat_k.to(torch.int64), flat_q.to(torch.int64),
        result_k, result_q, accum, B, C, W, R, num_slots, BLOCK_W, BLOCK_R)
    return torch.cat([result_k, result_q], dim=-1)

def _segmented_cross_cumsum_rev_scatter(
    d_lcq: torch.Tensor, flat_k: torch.Tensor, flat_q: torch.Tensor,
    C: int, W: int, R: int, num_slots: int,
    accum: torch.Tensor = None,
) -> torch.Tensor:
    """Scatter-based reverse cross cumsum for backward d_g computation. f32 output."""
    B = d_lcq.shape[0]
    device = d_lcq.device
    if accum is None:
        accum = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    else:
        accum[:B].zero_()
    result_k = torch.empty(B, C * W, device=device, dtype=torch.float32)
    BLOCK_W = triton.next_power_of_2(W)
    BLOCK_R = triton.next_power_of_2(R)
    _scatter_cross_cumsum_rev_kernel[(B,)](
        d_lcq, flat_k.to(torch.int64), flat_q.to(torch.int64),
        result_k, accum, B, C, W, R, num_slots, BLOCK_W, BLOCK_R)
    return result_k


# ============================================================
# Fused triple cumsum: forward K + cross Q + reverse K in ONE kernel
# Saves 2 kernel launches and eliminates redundant forward K scan.
# ============================================================

@triton.jit
def _fused_triple_cumsum_kernel(
    flat_g_ptr, flat_k_ptr, flat_q_ptr,
    result_k_ptr, result_q_ptr, result_rev_k_ptr,
    accum_fwd_ptr, accum_rev_ptr,
    B_dim: tl.constexpr, C_dim: tl.constexpr,
    W_dim: tl.constexpr, R_dim: tl.constexpr,
    num_slots: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """Fused triple cumsum: combines forward K, cross Q, and reverse K in one kernel.

    Pass 1 (forward t=0..C-1):
      - Scatter K g-values, gather K cumul → result_k (replaces _scatter_cumsum_fwd)
      - Also gather Q cumul at half-integer times → result_q (replaces cross cumsum)
    Pass 2 (reverse t=C-1..0):
      - Scatter K g-values, gather K cumul → result_rev_k (replaces _scatter_cumsum_rev)

    Uses two separate accumulators (fwd and rev) so they don't interfere.
    Grid: (B_dim,) — one program per batch element (chunk).
    NOTE: Uses range() instead of tl.static_range() for fast compilation.
    """
    bid = tl.program_id(0)
    if bid >= B_dim:
        return

    accum_fwd_base = bid * num_slots
    accum_rev_base = bid * num_slots
    k_base = bid * C_dim * W_dim
    q_base = bid * C_dim * R_dim
    w_offs = tl.arange(0, BLOCK_W)
    r_offs = tl.arange(0, BLOCK_R)
    w_mask = w_offs < W_dim
    r_mask = r_offs < R_dim

    # === Pass 1: Forward scan (produces both K cumul and Q cumul) ===
    for t in range(C_dim):
        # Scatter K g-values to accumulator
        k_pos = t * W_dim + w_offs
        k_slots = tl.load(flat_k_ptr + k_base + k_pos, mask=w_mask)
        g_vals = tl.load(flat_g_ptr + k_base + k_pos, mask=w_mask).to(tl.float32)
        old_k = tl.atomic_add(accum_fwd_ptr + accum_fwd_base + k_slots, g_vals, mask=w_mask)

        # Gather K cumul (integer time) — use return value, no barrier needed
        k_cumul = old_k + g_vals
        tl.store(result_k_ptr + k_base + k_pos, k_cumul, mask=w_mask)

        # Gather Q cumul (half-integer time — needs barrier to see K atomics from all warps)
        tl.debug_barrier()
        q_pos = t * R_dim + r_offs
        q_slots = tl.load(flat_q_ptr + q_base + q_pos, mask=r_mask)
        q_cumul = tl.load(accum_fwd_ptr + accum_fwd_base + q_slots, mask=r_mask)
        tl.store(result_q_ptr + q_base + q_pos, q_cumul, mask=r_mask)

    # === Pass 2: Reverse scan (produces reverse K cumul) ===
    for t_off in range(C_dim):
        t = C_dim - 1 - t_off
        k_pos = t * W_dim + w_offs
        k_slots = tl.load(flat_k_ptr + k_base + k_pos, mask=w_mask)
        g_vals = tl.load(flat_g_ptr + k_base + k_pos, mask=w_mask).to(tl.float32)
        old_rev = tl.atomic_add(accum_rev_ptr + accum_rev_base + k_slots, g_vals, mask=w_mask)
        rev_cumul = old_rev + g_vals
        tl.store(result_rev_k_ptr + k_base + k_pos, rev_cumul, mask=w_mask)


def _fused_triple_cumsum(
    flat_g: torch.Tensor, flat_k: torch.Tensor, flat_q: torch.Tensor,
    C: int, W: int, R: int, num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused triple cumsum: forward K + cross Q + reverse K.

    Returns:
        log_cumul_k: [B, C*W] forward K cumulative sums
        log_cumul_q: [B, C*R] cross Q cumulative sums
        rev_log_cumul_k: [B, C*W] reverse K cumulative sums
    """
    B = flat_g.shape[0]
    device = flat_g.device
    BLOCK_W = triton.next_power_of_2(W)
    BLOCK_R = triton.next_power_of_2(R)

    # Two separate accumulators (zeroed)
    accum_fwd = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    accum_rev = torch.zeros(B, num_slots, device=device, dtype=torch.float32)

    # Output tensors
    result_k = torch.empty(B, C * W, device=device, dtype=torch.float32)
    result_q = torch.empty(B, C * R, device=device, dtype=torch.float32)
    result_rev_k = torch.empty(B, C * W, device=device, dtype=torch.float32)

    _fused_triple_cumsum_kernel[(B,)](
        flat_g, flat_k.to(torch.int64), flat_q.to(torch.int64),
        result_k, result_q, result_rev_k,
        accum_fwd, accum_rev,
        B, C, W, R, num_slots, BLOCK_W, BLOCK_R,
    )

    return result_k, result_q, result_rev_k


# ============================================================
# Fused backward triple cumsum: 3 backward d_g cumsums in ONE kernel
# Combines: reverse K (d_g_fwd), forward K (d_g_rev), reverse cross (d_g_cq)
# Saves 2 kernel launches vs 3 separate calls.
# ============================================================

@triton.jit
def _fused_bwd_triple_cumsum_kernel(
    d_lck_total_ptr, d_rev_lck_ptr, d_lcq_ptr,
    flat_k_ptr, flat_q_ptr,
    result_fwd_ptr, result_rev_ptr, result_cq_ptr,
    accum1_ptr, accum2_ptr, accum3_ptr,
    B_dim: tl.constexpr, C_dim: tl.constexpr,
    W_dim: tl.constexpr, R_dim: tl.constexpr,
    num_slots: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_R: tl.constexpr,
):
    """Fused backward triple cumsum for d_g computation.

    Performs all three backward cumsums in a single pass using 3 accumulators:

    For t = 0..C-1 (forward direction):
      - accum1: scatter d_rev_lck[t] at k-slots, gather -> d_g_rev[t]
    For t_rev = C-1..0 (reverse direction, simultaneous):
      - accum2: scatter d_lck_total[t_rev] at k-slots, gather -> d_g_fwd[t_rev]
      - accum3: scatter d_lcq[t_rev] at q-slots, then gather at k-slots -> d_g_cq[t_rev]

    All 3 accumulators are pre-zeroed by Python (fast CUDA memset).
    Grid: (B_dim,) -- one program per chunk.
    NOTE: Uses range() instead of tl.static_range() for fast compilation.
    """
    bid = tl.program_id(0)
    if bid >= B_dim:
        return

    accum1_base = bid * num_slots
    accum2_base = bid * num_slots
    accum3_base = bid * num_slots
    k_base = bid * C_dim * W_dim
    q_base = bid * C_dim * R_dim
    w_offs = tl.arange(0, BLOCK_W)
    r_offs = tl.arange(0, BLOCK_R)
    w_mask = w_offs < W_dim
    r_mask = r_offs < R_dim

    # Single pass: forward + reverse + reverse-cross all run simultaneously
    for t in range(C_dim):
        t_rev = C_dim - 1 - t

        # Forward: scatter d_rev_lck[t] to accum1, gather -> d_g_rev[t]
        fwd_pos = t * W_dim + w_offs
        fwd_slots = tl.load(flat_k_ptr + k_base + fwd_pos, mask=w_mask)
        fwd_vals = tl.load(d_rev_lck_ptr + k_base + fwd_pos, mask=w_mask).to(tl.float32)
        old_fwd = tl.atomic_add(accum1_ptr + accum1_base + fwd_slots, fwd_vals, mask=w_mask)
        fwd_cumul = old_fwd + fwd_vals
        tl.store(result_rev_ptr + k_base + fwd_pos, fwd_cumul, mask=w_mask)

        # Reverse: scatter d_lck_total[t_rev] to accum2, gather -> d_g_fwd[t_rev]
        rev_pos = t_rev * W_dim + w_offs
        rev_slots = tl.load(flat_k_ptr + k_base + rev_pos, mask=w_mask)
        rev_vals = tl.load(d_lck_total_ptr + k_base + rev_pos, mask=w_mask).to(tl.float32)
        old_rev = tl.atomic_add(accum2_ptr + accum2_base + rev_slots, rev_vals, mask=w_mask)
        rev_cumul = old_rev + rev_vals
        tl.store(result_fwd_ptr + k_base + rev_pos, rev_cumul, mask=w_mask)

        # Reverse cross: scatter d_lcq[t_rev] at q-slots, then gather at k-slots -> d_g_cq[t_rev]
        q_pos = t_rev * R_dim + r_offs
        q_slots = tl.load(flat_q_ptr + q_base + q_pos, mask=r_mask)
        q_vals = tl.load(d_lcq_ptr + q_base + q_pos, mask=r_mask).to(tl.float32)
        tl.atomic_add(accum3_ptr + accum3_base + q_slots, q_vals, mask=r_mask)
        tl.debug_barrier()
        # Gather d_g_cq at k-slots (rev_slots already loaded above)
        cq_cumul = tl.load(accum3_ptr + accum3_base + rev_slots, mask=w_mask)
        tl.store(result_cq_ptr + k_base + rev_pos, cq_cumul, mask=w_mask)


def _fused_bwd_triple_cumsum(
    d_lck_total: torch.Tensor, d_rev_lck: torch.Tensor, d_lcq: torch.Tensor,
    flat_k: torch.Tensor, flat_q: torch.Tensor,
    C: int, W: int, R: int, num_slots: int,
    accum: torch.Tensor = None,
    accum2: torch.Tensor = None,
    accum3: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused backward triple cumsum: reverse K + forward K + reverse cross in one kernel.

    Args:
        d_lck_total: [B, C*W] grad values for reverse k-cumsum (d_g_fwd)
        d_rev_lck: [B, C*W] grad values for forward k-cumsum (d_g_rev)
        d_lcq: [B, C*R] grad values for reverse cross cumsum (d_g_cq)
        flat_k: [B, C*W] k-slot indices
        flat_q: [B, C*R] q-slot indices
        C: chunk size
        W: number of writes
        R: number of reads
        num_slots: total number of memory slots
        accum: optional [>=B, num_slots] pre-allocated accumulator 1
        accum2: optional [>=B, num_slots] pre-allocated accumulator 2
        accum3: optional [>=B, num_slots] pre-allocated accumulator 3

    Returns:
        d_g_fwd: [B, C*W] reverse k-cumsum result
        d_g_rev: [B, C*W] forward k-cumsum result
        d_g_cq: [B, C*W] reverse cross cumsum result
    """
    B = d_lck_total.shape[0]
    device = d_lck_total.device
    BLOCK_W = triton.next_power_of_2(W)
    BLOCK_R = triton.next_power_of_2(R)

    # Three accumulators: zeroed by Python (fast CUDA memset)
    if accum is None:
        _a1 = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    else:
        _a1 = accum
        _a1[:B].zero_()
    if accum2 is None:
        _a2 = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    else:
        _a2 = accum2
        _a2[:B].zero_()
    if accum3 is None:
        _a3 = torch.zeros(B, num_slots, device=device, dtype=torch.float32)
    else:
        _a3 = accum3
        _a3[:B].zero_()

    # Output tensors
    d_g_fwd = torch.empty(B, C * W, device=device, dtype=torch.float32)
    d_g_rev = torch.empty(B, C * W, device=device, dtype=torch.float32)
    d_g_cq = torch.empty(B, C * W, device=device, dtype=torch.float32)

    _fused_bwd_triple_cumsum_kernel[(B,)](
        d_lck_total, d_rev_lck, d_lcq,
        flat_k.to(torch.int64), flat_q.to(torch.int64),
        d_g_fwd, d_g_rev, d_g_cq,
        _a1, _a2, _a3,
        B, C, W, R, num_slots, BLOCK_W, BLOCK_R,
    )

    return d_g_fwd, d_g_rev, d_g_cq
