# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Gated Memory Controller Layer with Deltanet-style gating.

This implements a sparse memory architecture with gated deltanet updates.
The forget gate and input gate are applied selectively only to memory slots
accessed by the sparse keys, enabling efficient gradient flow.

The chunk-wise forward uses the WY representation (triangular solve) to
compute exact intra-chunk deltanet recurrence, making the output
chunk_size-invariant.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .cache import SDMLayerState, l2_normalize
from .memory_stats import SDMStatsMixin
from einops import rearrange
from .memory_ops import (
    fast_embedding_bag,
    GatedSparseMemoryWriteRead,
    gated_sparse_memory_write_read_inference,
    gated_sparse_memory_write_read_inference_v2,
    fused_decode_step,
    fused_product_key_rest,
    fused_outer_add_topk,
    expand_memory_batch,
)

import torch.distributed as dist


def _packed_all_gather(tensors, group):
    """All-gather multiple tensors in a single collective by packing into one buffer.

    Args:
        tensors: list of tensors (may have different shapes/dtypes)
        group: process group

    Returns:
        list of lists: gathered[i] = [tensor_from_rank_0, tensor_from_rank_1, ...]
    """
    world_size = dist.get_world_size(group)

    # Record metadata for unpacking
    metadata = []  # (numel, dtype, shape) per tensor
    total_bytes = 0
    for t in tensors:
        nbytes = t.numel() * t.element_size()
        metadata.append((t.numel(), t.dtype, t.shape, nbytes))
        total_bytes += nbytes

    # Pack into a single byte buffer
    buf = torch.empty(total_bytes, dtype=torch.uint8, device=tensors[0].device)
    offset = 0
    for t, (_, _, _, nbytes) in zip(tensors, metadata):
        buf[offset:offset + nbytes].copy_(t.contiguous().flatten().view(torch.uint8))
        offset += nbytes

    # Single all-gather
    gathered_bufs = [torch.empty_like(buf) for _ in range(world_size)]
    dist.all_gather(gathered_bufs, buf, group=group)

    # Unpack per-rank
    result = [[] for _ in range(len(tensors))]
    for rank_buf in gathered_bufs:
        offset = 0
        for i, (numel, dtype, shape, nbytes) in enumerate(metadata):
            raw = rank_buf[offset:offset + nbytes]
            t = raw.view(dtype)[:numel].reshape(shape).clone()
            result[i].append(t)
            offset += nbytes

    return result


# ============================================================
# Ulysses-style CP for SDM: alltoall on head dimension
# ============================================================
# Instead of sequential broadcasts (cp_world rounds per layer), Ulysses
# shards heads across CP ranks. Each rank processes the FULL sequence
# for H/cp_size heads — no sequential state dependency, no broadcast chain.
# Requires H % cp_size == 0.
#
# Cost: 1 alltoall before + 1 alltoall after SDM kernel per layer.
# Memory saving vs sequential: per-rank memory bank halves (H_local vs H heads).


class _UlyssesScatterHeads(torch.autograd.Function):
    """Scatter heads, gather tokens: [B, T_local, H, D] → [B, T_full, H_local, D]."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        cp = dist.get_world_size(group)
        B, T_local, H, D = x.shape
        assert H % cp == 0, f"H={H} must be divisible by cp={cp}"
        H_local = H // cp
        # Reshape: [B, T_local, cp, H_local, D] then scatter on cp dim, gather on token dim
        x = x.reshape(B, T_local, cp, H_local, D)
        x = x.permute(2, 0, 1, 3, 4).contiguous()  # [cp, B, T_local, H_local, D]
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        # out[r] has rank r's T_local tokens for our H_local heads
        out = out.permute(1, 0, 2, 3, 4).contiguous()  # [B, cp, T_local, H_local, D]
        return out.reshape(B, T_local * cp, H_local, D)

    @staticmethod
    def backward(ctx, grad_out):
        # Reverse: gather heads, scatter tokens
        return _UlyssesGatherHeads.apply(grad_out, ctx.group, None), None


class _UlyssesGatherHeads(torch.autograd.Function):
    """Gather heads, scatter tokens: [B, T_full, H_local, D] → [B, T_local, H, D]."""

    @staticmethod
    def forward(ctx, x, group, _unused):
        ctx.group = group
        cp = dist.get_world_size(group)
        B, T_full, H_local, D = x.shape
        T_local = T_full // cp
        x = x.reshape(B, cp, T_local, H_local, D)
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # [cp, B, T_local, H_local, D]
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        # out: [cp, B, T_local, H_local, D] — dim 0 now indexes head chunks (from different ranks)
        # Permute so head chunks are adjacent: [B, T_local, cp, H_local, D] then merge cp*H_local=H
        out = out.permute(1, 2, 0, 3, 4).contiguous()  # [B, T_local, cp, H_local, D]
        H = cp * H_local
        return out.reshape(B, T_local, H, D)

    @staticmethod
    def backward(ctx, grad_out):
        return _UlyssesScatterHeads.apply(grad_out, ctx.group), None, None


def ulysses_scatter_heads(x: torch.Tensor, group) -> torch.Tensor:
    """[B, T_local, H, D] → [B, T_full, H_local, D] via alltoall."""
    return _UlyssesScatterHeads.apply(x, group)


def ulysses_gather_heads(x: torch.Tensor, group) -> torch.Tensor:
    """[B, T_full, H_local, D] → [B, T_local, H, D] via alltoall."""
    return _UlyssesGatherHeads.apply(x, group, None)


class _GradScale(torch.autograd.Function):
    """Scale gradient by a constant in backward. Forward is identity."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def _cp_gated_write_read(layer, memory, k_idx, k_val, v, beta, g, q_idx, q_val, cp_group, h_local=None, head_start=0):
    """CP-aware gated write-read using broadcast for state passing.

    No input all-gather — each rank only stores its own tokens.
    Sequential rounds of broadcast pass the memory state between ranks.
    Cost: cp_world forward passes (1 autograd + cp_world-1 no_grad) in forward,
    similar in backward. All communication is broadcast (FSDP-safe).

    h_local / head_start: under 2D CP (Ulysses head-split then this sequential
    token-relay) the kernel operates on H_local = num_heads/cp_uly heads taken
    from head_start = cp_uly_rank * H_local. Defaults (None, 0) reproduce the
    pure-sequential behaviour (all heads). These are needed so the backward
    reconstructs the correct per-rank M0 (local-head slice of the learned memory).
    """
    return _CPWriteReadFn.apply(
        memory, k_idx, k_val, v, beta, g, q_idx, q_val, cp_group, layer, h_local, head_start,
    )


class _CPWriteReadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, k_idx, k_val, v, beta, g, q_idx, q_val, cp_group, layer, h_local=None, head_start=0):
        cp_rank = dist.get_rank(cp_group)
        cp_world = dist.get_world_size(cp_group)
        num_slots, dim = memory.shape

        # --- Sequential forward rounds via broadcast ---
        # cp_world rounds. In round r:
        #   - Rank r runs gated_write_read on its local tokens with current M_state
        #   - Rank r broadcasts M_final_r to all ranks
        #   - All other ranks just receive the broadcast
        # Each rank runs forward exactly once (on its own round).
        # Rounds before this rank: no_grad (just state propagation).
        # This rank's round: with autograd.
        # Rounds after: no_grad (state propagation continues for later ranks).
        M_state = memory.clone()
        readings = None
        memory_final = None
        for r in range(cp_world):
            src = dist.get_global_rank(cp_group, r)
            if cp_rank == r:
                # Our turn — run with autograd
                readings, memory_final = layer.gated_write_read(
                    M_state, k_idx, k_val, v, beta, g, q_idx, q_val
                )
                M_state = memory_final.data.clone()
            # All ranks participate in broadcast
            dist.broadcast(M_state.contiguous(), src, group=cp_group)

        # Save for backward
        ctx.save_for_backward(k_idx, k_val, v, beta, g, q_idx, q_val)
        ctx.cp_group = cp_group
        ctx.layer = layer
        ctx.cp_rank = cp_rank
        ctx.cp_world = cp_world
        # 2D CP: kernel runs on h_local heads (from head_start) — see _get_m0.
        ctx.h_local = h_local if h_local is not None else layer.num_heads
        ctx.head_start = head_start
        ctx.mem_shape = memory.shape
        ctx.mem_dtype = memory.dtype
        ctx.mem_device = memory.device

        memory.data.copy_(memory_final.data)
        return readings, memory

    @staticmethod
    def backward(ctx, grad_readings, grad_memory_unused):
        k_idx, k_val, v, beta, g, q_idx, q_val = ctx.saved_tensors
        cp_group = ctx.cp_group
        layer = ctx.layer
        cp_rank = ctx.cp_rank
        cp_world = ctx.cp_world

        BH = k_idx.shape[0]
        H = ctx.h_local
        head_start = ctx.head_start
        sph = layer.slots_per_head
        hd = layer.head_dim

        def _get_m0():
            # Reconstruct the exact per-rank initial memory the forward fed to the
            # kernel. With decay_to_init the forward inits the bank to zeros; else
            # it expands the learned memory. Under 2D CP only the local-head slice
            # [head_start : head_start+H] is used (H = H_local). For pure sequential
            # CP, head_start=0 and H=num_heads → the full bank (unchanged behaviour).
            if layer.memory is not None:
                batch = BH // H if H > 1 else BH
                mem_local = layer.memory.reshape(-1, hd)[
                    head_start * sph:(head_start + H) * sph
                ]
                return mem_local.unsqueeze(0).expand(batch, -1, -1).reshape(BH * sph, hd).clone()
            return torch.zeros(BH * sph, hd, device=ctx.mem_device, dtype=ctx.mem_dtype)

        def _run_backward(init_mem, grad_rd, grad_fm=None):
            """Recompute forward + backward on THIS rank's tokens."""
            mem = init_mem.detach().requires_grad_(True)
            kv = k_val.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            bv = beta.detach().requires_grad_(True)
            gv = g.detach().requires_grad_(True)
            qv = q_val.detach().requires_grad_(True)
            with torch.enable_grad():
                rd, _ = layer.gated_write_read(
                    mem, k_idx, kv, v_, bv, gv, q_idx, qv,
                    grad_final_memory=grad_fm,
                )
            torch.autograd.backward(rd, grad_rd)
            g_mem = mem.grad if mem.grad is not None else torch.zeros_like(mem)
            return g_mem, kv.grad, v_.grad, bv.grad, gv.grad, qv.grad

        # --- Recompute M_init via forward broadcast rounds (same protocol as forward) ---
        M_state = _get_m0()
        M_init = M_state.clone()  # will be overwritten for rank > 0
        with torch.no_grad():
            for r in range(cp_world):
                src = dist.get_global_rank(cp_group, r)
                if cp_rank == r:
                    _, M_state = layer.gated_write_read(
                        M_state, k_idx, k_val, v, beta, g, q_idx, q_val
                    )
                    M_state = M_state.clone()
                # Save state BEFORE this rank's forward as M_init
                if r == cp_rank - 1:
                    pass  # M_init already set from previous broadcast
                dist.broadcast(M_state.contiguous(), src, group=cp_group)
                if r == cp_rank - 1:
                    M_init = M_state.clone()
        if cp_rank == 0:
            M_init = _get_m0()

        # --- First backward (no grad_fm): get this rank's grad_M_init ---
        grad_M_init, g_kv_0, g_v_0, g_b_0, g_g_0, g_qv_0 = _run_backward(
            M_init, grad_readings,
        )

        # --- All-gather grad_M_init (small — one [num_slots, dim] per rank) ---
        grad_M_list = [torch.empty_like(grad_M_init) for _ in range(cp_world)]
        dist.all_gather(grad_M_list, grad_M_init.contiguous(), group=cp_group)

        # --- Compute grad_fm via reverse broadcast rounds ---
        # Reverse sequential: from rank cp_world-1 backward.
        # grad_fm for rank r = accumulated gradient from all later ranks,
        # propagated backward through each intermediate rank's forward.
        #
        # Protocol: cp_world-1 rounds in reverse. In round r (cp_world-1 down to 1):
        #   - incoming = grad_M_list[r] + grad_chain (accumulated from later rounds)
        #   - Rank r-1 runs _run_backward with grad_fm=incoming → gets grad_mem
        #   - Rank r-1 broadcasts grad_mem to all
        #   - grad_chain = grad_mem for next round
        #   - If r-1 == cp_rank, save incoming as this rank's grad_fm

        grad_fm = None
        zero_grad_rd = torch.zeros_like(grad_readings)
        grad_chain = torch.zeros_like(grad_M_init)

        for r in range(cp_world - 1, 0, -1):
            incoming = grad_M_list[r] + grad_chain
            src = dist.get_global_rank(cp_group, r - 1)

            if r - 1 == cp_rank:
                # This is OUR grad_fm — don't need to broadcast, just save it
                grad_fm = incoming

            # Rank r-1 propagates 'incoming' backward through its forward
            result = torch.empty_like(grad_M_init)
            if cp_rank == r - 1:
                result, _, _, _, _, _ = _run_backward(
                    M_init, zero_grad_rd, grad_fm=incoming,
                )
            dist.broadcast(result.contiguous(), src, group=cp_group)
            grad_chain = result

        # --- Second backward WITH grad_fm (if needed) ---
        if grad_fm is not None:
            _, g_kv, g_v, g_b, g_g, g_qv = _run_backward(
                M_init, grad_readings, grad_fm=grad_fm,
            )
        else:
            # Last rank: no grad_fm, reuse first backward
            g_kv, g_v, g_b, g_g, g_qv = g_kv_0, g_v_0, g_b_0, g_g_0, g_qv_0

        # --- Compute memory gradient: chain this rank's grad_M_init back to m0 ---
        # Each rank's grad_memory = dL_rank/d(m0).
        # Rank 0: already at m0, grad_memory = grad_M_init.
        # Rank r>0: chain backward through ranks r-1, ..., 0.
        # All ranks participate in cp_world-1 broadcast rounds.
        #
        # This chain propagates grad_M_init from the LAST rank backward.
        # Intermediate ranks pick up their chain result when the round
        # processes their position.
        chain_val = torch.zeros_like(grad_M_init)
        for i in range(cp_world - 1):
            r = cp_world - 1 - i  # r goes cp_world-1, ..., 1
            # Rank r broadcasts its grad_M_init (+ accumulated chain from later ranks)
            bcast_val = torch.empty_like(grad_M_init)
            if cp_rank == r:
                bcast_val = grad_M_list[r] + chain_val
            dist.broadcast(bcast_val.contiguous(), dist.get_global_rank(cp_group, r), group=cp_group)
            # Rank r-1 runs backward to propagate bcast_val through its forward
            result = torch.empty_like(grad_M_init)
            if cp_rank == r - 1:
                result, _, _, _, _, _ = _run_backward(
                    M_init, zero_grad_rd, grad_fm=bcast_val,
                )
            dist.broadcast(result.contiguous(), dist.get_global_rank(cp_group, r - 1), group=cp_group)
            chain_val = result  # all ranks update — needed for next round

        # After the loop: chain_val = total dL/d(m0) chained from all later ranks.
        # Total memory gradient = grad_M_list[0] + chain_val.
        # Every cp_seq rank (sharing the same heads) computes this identical value.
        #
        # FSDP reduce-scatter SUMS grads over the flattened (dp × cp) group and
        # divides only by dp_size (set_reduce_scatter_divide_factor(dp_size)). The
        # cp_world (= cp_seq) ranks each emit total_grad/cp_world, so FSDP sums
        # them back to total_grad before the /dp_size DP average → exact match vs
        # the no-CP mean-loss baseline. Under 2D CP, different cp_uly ranks own
        # disjoint head slices (zero grad elsewhere), so there is no /cp_uly.
        grad_memory = (grad_M_list[0] + chain_val) / cp_world

        return (grad_memory, None,
                g_kv if g_kv is not None else torch.zeros_like(k_val),
                g_v if g_v is not None else torch.zeros_like(v),
                g_b if g_b is not None else torch.zeros_like(beta),
                g_g if g_g is not None else torch.zeros_like(g),
                None,
                g_qv if g_qv is not None else torch.zeros_like(q_val),
                None, None, None, None)


class ScaledSoftmax(nn.Module):
    """Softmax with logit scaling: softmax(x * scale)."""
    def __init__(self, scale: float = 1.0, dim: int = -1):
        super().__init__()
        self.scale = scale
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x * self.scale, dim=self.dim)


@dataclass
class SparseDeltaMemoryArgs:
    """Configuration for SparseDeltaMemory.

    Layer placement (which layers use SDM) is controlled by the transformer config's
    `sdm_at` list, not here — this dataclass holds only SDM hyper-parameters.
    """
    dim: int = -1
    num_writes: int = 1
    num_reads: int = 1
    slots_per_head: int = 10000  # Per-head memory slots; must be a perfect square (product-key addressing).
    memory_block_size: int = 256  # Kernel chunk size: a speed-only knob, does NOT change results (the kernel is chunk-size invariant).
    read_act: str = "Softmax"
    write_act: str = "Softmax"
    read_act_scale: float = 1.0
    write_act_scale: float = 1.0
    normalize_readings: bool = False  # LayerNorm on per-head readings before the output projection
    norm_eps: float = 1e-6
    # Gating parameters
    dt_bias_init: float = 0.0  # Initial bias for time step
    A_init_range: tuple[float, float] = (0.0, 16.0)  # Range for decay parameter init (uniform random, matching GDN/FLA)
    backprop_on_memory: bool = True  # Learn the initial memory (parametric memory) — key to SDM's pretraining-knowledge gains.
    output_gate: bool = False  # Sigmoid output gate: sigmoid(Wog(x)) * readings
    query_batchnorm: bool = False  # BatchNorm on PKM query/key projections (prevents slot collapse)
    log_memory_access_stats: bool = False  # Log memory access statistics (read/write idx frequency, gate stats)
    log_memory_norms: bool = False  # Log memory norm/max stats (expensive: .abs() copies full memory bank)
    num_heads: int = 1  # Independent memory heads. Each head has (slots_per_head, dim/num_heads) memory.
    key_weighted_decay: bool = False  # Weight forget gate by key values: exp(g * k_val) instead of exp(g)
    snapshot_quant: str = "none"  # Quantize saved snapshot to reduce activation memory: "none" (bf16), "fp8", "int4"


class SparseDeltaMemory(SDMStatsMixin, nn.Module):
    """Memory Controller with Deltanet-style gating.

    This layer implements sparse memory with gated updates:
    - Forget gate g controls exponential decay of memory
    - Input gate beta controls how much new information to integrate
    - Gates only apply to memory slots touched by sparse keys

    The gating mechanism follows the deltanet formulation:
    - g = -A * softplus(a_proj(x) + dt_bias)
    - beta = sigmoid(b_proj(x))
    - memory[idx] *= exp(g)
    - memory[idx] += beta * k @ (v - memory[idx] @ k).T
    """

    def __init__(self, args: SparseDeltaMemoryArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.dim = args.dim
        self.num_heads = args.num_heads

        # Per-head dimensions
        H = self.num_heads
        assert args.dim % H == 0, f"dim={args.dim} must be divisible by num_heads={H}"
        self.head_dim = args.dim // H

        # Memory slots are configured PER HEAD. Total slots = slots_per_head * H².
        # (Product keys give key_size = 2*sqrt(slots); the key space splits across
        # heads, so total = (H * key_size_per_head / 2)² = slots_per_head * H².
        # When H=1 this reduces to total_slots = slots_per_head.)
        self.slots_per_head = args.slots_per_head

        # Product keys (mandatory): slots_per_head must be a perfect square so the two
        # sub-codes of size sqrt(slots_per_head) address exactly slots_per_head slots.
        sph_sqrt = int(round(self.slots_per_head ** 0.5))
        assert sph_sqrt * sph_sqrt == self.slots_per_head, (
            f"slots_per_head must be a perfect square (product keys); got {self.slots_per_head}"
        )
        self.key_size_per_head = 2 * sph_sqrt
        self.key_size = H * self.key_size_per_head

        # Query and Key projections
        self.Wq_read = nn.Linear(args.dim, self.key_size)
        self.Wk_write = nn.Linear(args.dim, self.key_size)
        self.Wv_write = nn.Linear(args.dim, args.dim)

        # BatchNorm on PKM projections (prevents slot collapse by normalizing
        # the query distribution across the batch, ensuring uniform key space usage)
        self.query_bn = nn.BatchNorm1d(self.key_size) if args.query_batchnorm else None
        self.key_bn = nn.BatchNorm1d(self.key_size) if args.query_batchnorm else None

        # Gating projections (per-head when num_heads > 1)
        self.a_proj = nn.Linear(args.dim, H)  # For forget gate
        self.b_proj = nn.Linear(args.dim, H)  # For input gate

        # Decay parameter (A in deltanet/GDN formulation)
        # Initialized as uniform random in [A_min, A_max], stored in log space (matching FLA)
        A_min, A_max = args.A_init_range
        A_init = torch.empty(H, dtype=torch.float32).uniform_(A_min, A_max).clamp(min=1e-4)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.A_log._no_weight_decay = True

        # Time step bias — initialized from inv_softplus(uniform(dt_min, dt_max)) matching FLA/GDN.
        # This gives softplus(a_proj(x) + dt_bias) ≈ dt ≈ 0.001-0.1 at init,
        # so g = -A * dt ≈ -0.01 to -1.6 (moderate decay), avoiding exp overflow in WY.
        import math
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inv_softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Output projection
        self.Wo = nn.Linear(args.dim, args.dim)

        # Output gate (sigmoid gating, like the ungated memctrl's output_gate)
        self.Wog = nn.Linear(args.dim, args.dim) if args.output_gate else None

        # Learnable initial memory (when backprop_on_memory=True)
        # With num_heads > 1: H separate memories of (slots_per_head, head_dim)
        # stored as (H * slots_per_head, head_dim)
        total_mem_slots = H * self.slots_per_head
        mem_dim = self.head_dim if H > 1 else args.dim
        self.memory = (
            nn.Parameter(torch.zeros(total_mem_slots, mem_dim))
            if args.backprop_on_memory
            else None
        )
        # Tag the learned memory bank so model-size / param-count reporting can exclude it:
        # it is SDM's *state* (read sparsely, negligible FLOPs), not an "active" parameter.
        if self.memory is not None:
            self.memory._sdm_memory_bank = True

        # Optional per-head normalization of the readings (matching GDN convention).
        self.readings_norm = (
            nn.LayerNorm(self.head_dim, eps=args.norm_eps) if args.normalize_readings else None
        )

        # Activation functions
        self.read_act = (
            ScaledSoftmax(scale=args.read_act_scale, dim=-1)
            if args.read_act == "Softmax"
            else l2_normalize
            if args.read_act == "l2_normalize"
            else getattr(nn, args.read_act)()
        )
        self.write_act = (
            ScaledSoftmax(scale=args.write_act_scale, dim=-1)
            if args.write_act == "Softmax"
            else l2_normalize
            if args.write_act == "l2_normalize"
            else getattr(nn, args.write_act)()
        )

        # Memory access statistics tracking (opt-in logging; see memory_stats.py).
        self._init_stats(args, H)

    def compute_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forget gate g and input gate beta.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            g: Forget gate in log space [B, T, H]
            beta: Input gate [B, T, H]
        """
        # Compute forget gate: g = -A * softplus(a_proj + dt_bias)
        a = self.a_proj(x)  # [B, T, H]

        A = -torch.exp(self.A_log)  # [H], negative for decay
        g = A * F.softplus(a + self.dt_bias)  # [B, T, H]

        # Compute input gate: beta = sigmoid(b_proj)
        b = self.b_proj(x)  # [B, T, H]

        beta = torch.sigmoid(b)  # [B, T, 1]

        return g, beta

    def _get_product_keys_for_head(
        self,
        x: torch.Tensor,  # [B, T, key_size_per_head]
        num_keys: int,
        half_key: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Product key computation for a single head with its own sub-key space."""
        B, T, _ = x.shape
        q1, q2 = x.chunk(2, dim=-1)  # each [B, T, half_key]

        k_sub = min(num_keys, q1.shape[-1])
        top_val1, idx1 = torch.topk(q1, k=k_sub, dim=-1)
        top_val2, idx2 = torch.topk(q2, k=k_sub, dim=-1)

        if T == 1 and not self.training:
            # Fast decode path: fused remap + argsort (1 kernel vs 7)
            val, idx = fused_product_key_rest(
                top_val1.reshape(B, k_sub), idx1.reshape(B, k_sub),
                top_val2.reshape(B, k_sub), idx2.reshape(B, k_sub),
                half_key, k_sub,
            )
            return (val.unsqueeze(1).to(x.dtype), idx.unsqueeze(1))

        # Fused outer-add + topk with sparse backward — avoids materializing the
        # [B, T, K, K] grad in backward (was ×4 reduce_kernel = ~17ms at L13).
        k_final = min(num_keys, k_sub * k_sub)
        val, inner_idx1, inner_idx2 = fused_outer_add_topk(top_val1, top_val2, k_final)

        chosen_idx1 = torch.gather(idx1, dim=-1, index=inner_idx1)
        chosen_idx2 = torch.gather(idx2, dim=-1, index=inner_idx2)
        idx = chosen_idx1 * half_key + chosen_idx2

        # Sort by index value for efficient two-pointer sparse inner product
        if idx.shape[-1] <= 128:
            from .triton_argsort import triton_argsort
            orig_shape = idx.shape
            idx_sorted, sort_perm = triton_argsort(idx.reshape(-1, orig_shape[-1]))
            idx = idx_sorted.reshape(orig_shape)
            val = val.reshape(-1, orig_shape[-1]).gather(-1, sort_perm).reshape(orig_shape)
        else:
            idx, sort_perm = idx.sort(dim=-1)
            val = val.gather(-1, sort_perm)

        return (val, idx)

    def gated_write_read(
        self,
        memory: torch.Tensor,
        k_idx: torch.Tensor,
        k_val: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        q_idx: torch.Tensor,
        q_val: torch.Tensor,
        grad_final_memory=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunk-wise gated write + read with optional WY intra-chunk correction."""
        B, T, num_writes = k_idx.shape
        num_reads = q_idx.shape[2]
        dim = v.shape[2]

        # Indices are within per-head memory (of size slots_per_head).
        effective_num_slots = self.slots_per_head

        # Chunk size cannot exceed the sequence length: a short sequence (T <
        # memory_block_size) must run as a single chunk, otherwise the chunked kernels
        # over-run their buffers (illegal memory access). This is a no-op in normal
        # training (T >> memory_block_size) and exact by chunk-size invariance.
        chunk_size = min(self.args.memory_block_size, T)

        # Keep batch dim separate — don't flatten BH*T
        # Inputs are [B, T, ...] where B = BH (batch × heads)

        if self.training:
            # Training: autograd Function with full backward support (delta rule)
            readings, _delta_v = GatedSparseMemoryWriteRead.apply(
                memory, k_idx, k_val, v, beta, g, q_idx, q_val,
                chunk_size,
                True,  # intra_chunk: always on (exact WY intra-chunk recurrence)
                effective_num_slots,
                B,  # num_parallel
                False,  # normalize_memory (removed; kernel arg retained as no-op)
                self.args.snapshot_quant,
                grad_final_memory,
            )
        elif T == 1:
            # Fast decode: fused kernels, all heads in parallel
            k_idx_flat = k_idx.reshape(B, num_writes)
            k_val_flat = k_val.reshape(B, num_writes)
            v_flat = v.reshape(B, dim)
            beta_flat = beta.reshape(B, 1)
            g_flat = g.reshape(B, g.shape[-1])
            q_idx_flat = q_idx.reshape(B, num_reads)
            q_val_flat = q_val.reshape(B, num_reads)
            readings = fused_decode_step(
                memory, k_idx_flat, k_val_flat, v_flat,
                beta_flat, g_flat, q_idx_flat, q_val_flat,
                use_delta_rule=True,
                normalize_memory=False,
                key_weighted_decay=self.args.key_weighted_decay,
            )
        elif T <= 64:
            # Inference decode: token-by-token, no backward
            k_idx_flat = k_idx.reshape(B * T, num_writes)
            k_val_flat = k_val.reshape(B * T, num_writes)
            v_flat = v.reshape(B * T, dim)
            beta_flat = beta.reshape(B * T, 1)
            g_flat = g.reshape(B * T, g.shape[-1])
            q_idx_flat = q_idx.reshape(B * T, num_reads)
            q_val_flat = q_val.reshape(B * T, num_reads)
            readings = self._step_write_read(
                memory, k_idx_flat, k_val_flat, v_flat,
                beta_flat, g_flat, q_idx_flat, q_val_flat,
            )
        else:
            # Inference prefill: v2 kernel — WY Phase 2 optimizations (fused
            # dual_matmul + is_only decay fast-path) with reusable per-chunk
            # snapshot buffer (~20 MB) instead of full backward saves (~26 GB).
            readings = gated_sparse_memory_write_read_inference_v2(
                memory, k_idx, k_val, v, beta, g, q_idx, q_val,
                chunk_size,
                effective_num_slots,
                B,  # num_parallel
            )

        return readings.view(B, T, dim), memory

    def _step_write_read(
        self,
        memory: torch.Tensor,
        k_idx: torch.Tensor,
        k_val: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        q_idx: torch.Tensor,
        q_val: torch.Tensor,
    ) -> torch.Tensor:
        """Token-by-token inference write-read (T≤64). No backward, no snapshots."""
        B_T, num_writes = k_idx.shape
        num_slots, dim = memory.shape
        num_reads = q_idx.shape[1]
        device = memory.device
        dtype = memory.dtype

        readings = torch.empty(B_T, dim, device=device, dtype=dtype)

        for t in range(B_T):
            sl = slice(t, t + 1)

            c_k_idx = k_idx[sl]
            c_k_val = k_val[sl]
            c_v = v[sl]
            c_beta = beta[sl]
            c_g = g[sl]
            flat_k = c_k_idx.reshape(-1)

            # Delta rule: delta_v = beta * (v - retrieved)
            mem_read_raw = memory.data[flat_k].reshape(1, num_writes, dim)
            decay = torch.exp(c_g).unsqueeze(2)
            retrieved = torch.einsum('cn,cnd->cd', c_k_val, mem_read_raw * decay)
            delta_v = c_beta * (c_v - retrieved)
            write_vals = torch.einsum('cn,cd->cnd', c_k_val, delta_v)

            # Apply decay + write to memory
            decay_per = torch.exp(c_g).expand(1, num_writes).reshape(-1)
            cumul = torch.ones(num_slots, 1, device=device, dtype=dtype)
            cumul.scatter_reduce_(
                0, flat_k.unsqueeze(1), decay_per.unsqueeze(1),
                reduce='prod', include_self=True,
            )
            memory.data[flat_k] = memory.data[flat_k] * cumul[flat_k]
            memory.data.index_add_(0, flat_k, write_vals.reshape(-1, dim))

            # Read output
            q_flat = q_idx[sl].reshape(-1)
            read_mem = memory.data[q_flat].reshape(1, num_reads, dim)
            readings[sl] = torch.einsum('cn,cnd->cd', q_val[sl], read_mem)

        return readings

    def read(
        self,
        memory: torch.Tensor,
        q_val: torch.Tensor,
        q_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Read from memory using sparse queries (for inference / single-token).

        Args:
            memory: Memory tensor [B * num_slots, dim] (batched)
            q_val: Query strengths [B, T, num_reads]
            q_idx: Query indices [B, T, num_reads] (with batch offsets)

        Returns:
            Reading result [B, T, dim]
        """
        B, T, num_reads = q_idx.shape

        # Use embedding bag for efficient sparse read
        readings = fast_embedding_bag(
            q_idx.view(B * T, num_reads),
            memory,
            q_val.view(B * T, num_reads),
            mode="sum",
        )

        return readings.view(B, T, -1)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: SDMLayerState | None = None,
    ) -> Tuple[torch.Tensor, SDMLayerState | None]:
        """Forward pass with gated memory updates.

        Args:
            x: Input [B, T, D] or [B*T, D]
            freqs_cis: Positional encoding tensor, used to extract seq_len when x is 2D
            mask: Not used, for compatibility
            cache: Optional memory state

        Returns:
            output: Output [B, T, D] or [B*T, D] (matches input)
            new_cache: Updated memory state
        """
        # Handle both 2D and 3D inputs (like mLSTM does)
        if x.ndim == 2:
            # x is [B*T, D], extract seq_len from freqs_cis
            assert freqs_cis is not None, "freqs_cis required when input is 2D"
            seq_len, _ = freqs_cis.shape
            assert x.shape[0] % seq_len == 0, f"{x.shape=} {seq_len=}"
            bsz = x.shape[0] // seq_len
            x = x.reshape(bsz, seq_len, self.args.dim)
            input_was_2d = True
        else:
            input_was_2d = False

        # CP: sequential state passing via broadcast (not send/recv which deadlocks with FSDP,
        # not all-gather which doubles memory and defeats the purpose of CP).
        # All ranks call broadcast simultaneously — no deadlock.
        cp_group = getattr(self, 'cp_group', None)
        cp_ulysses_group = getattr(self, 'cp_ulysses', None)
        _ulysses_active = cp_ulysses_group is not None
        # 2D CP: Ulysses head-split (cp_ulysses) composed with sequential token
        # relay (cp_group). The uly scatter/gather + local-head memory init below
        # run on cp_ulysses; the write/read kernel runs the sequential relay on
        # cp_group instead of a single standard kernel call.
        _both_active = _ulysses_active and (cp_group is not None)

        B, T, D = x.shape
        H = self.num_heads
        sph = self.slots_per_head   # num_memory_slots when H=1
        hd = self.head_dim          # dim when H=1
        BH = B * H

        # Compute gates (shared across heads)
        g, beta = self.compute_gates(x)  # Both [B, T, 1]

        # Log gate statistics (decay factor = exp(g) in [0,1])
        self._record_gate_stats(g, beta)

        # Compute value projection
        v = self.Wv_write(x)  # [B, T, D]

        # Compute product keys — project then chunk into heads.
        raw_k = self.Wk_write(x)  # [B, T, key_size]
        raw_q = self.Wq_read(x)   # [B, T, key_size]
        if self.key_bn is not None:
            raw_k = self.key_bn(raw_k.reshape(B * T, -1)).reshape(B, T, -1)
        if self.query_bn is not None:
            raw_q = self.query_bn(raw_q.reshape(B * T, -1)).reshape(B, T, -1)

        # Reshape to [BH, T, kph] — when H=1 this is just [B, T, key_size]
        kph = self.key_size_per_head
        raw_k = rearrange(raw_k, 'b t (h k) -> (b h) t k', h=H)
        raw_q = rearrange(raw_q, 'b t (h k) -> (b h) t k', h=H)

        # Product keys on [BH, T, kph] → [BH, T, num_writes/reads]
        k_val, k_idx = self._get_product_keys_for_head(raw_k, self.args.num_writes, kph // 2)
        q_val, q_idx = self._get_product_keys_for_head(raw_q, self.args.num_reads, kph // 2)
        q_val = self.read_act(q_val)
        k_val = self.write_act(k_val)

        # Track read index statistics if enabled (skip under Ulysses CP — local heads only)
        self._record_read_stats(q_idx, q_val, BH, H, sph, _ulysses_active)

        # Reshape v and gates to [BH, T, ...]
        v = rearrange(v, 'b t (h d) -> (b h) t d', h=H)
        g = rearrange(g, 'b t h -> (b h) t 1')
        beta = rearrange(beta, 'b t h -> (b h) t 1')

        # Key-weighted decay: exp(g * k_val) instead of exp(g)
        # When enabled, g becomes [BH, T, num_writes] (per-write decay)
        # When disabled, g stays [BH, T, 1] (scalar decay, expanded inside kernel)
        if self.args.key_weighted_decay:
            g = g * k_val  # [BH, T, num_writes]

        # --- Ulysses CP: alltoall on head dim BEFORE memory/offset setup ---
        # Swaps token↔head sharding so each rank gets full T for H_local heads.
        # After alltoall, BH/T/H are redefined for the rest of the function.
        if _ulysses_active:
            cp_size = dist.get_world_size(cp_ulysses_group)
            cp_rank = dist.get_rank(cp_ulysses_group)
            assert H % cp_size == 0, f"Ulysses CP requires num_heads={H} divisible by cp_size={cp_size}"
            H_local = H // cp_size
            T_local = T  # T is already T/cp from upstream token slicing

            def _scatter_heads(x):
                """[B*H, T_local, D] → [B*H_local, T_full, D] via head-scatter alltoall."""
                D = x.shape[-1]
                x = x.reshape(B, H, T_local, D).permute(0, 2, 1, 3)
                x = ulysses_scatter_heads(x, cp_ulysses_group)
                return x.permute(0, 2, 1, 3).reshape(B * H_local, T_local * cp_size, D)

            # alltoall all per-head tensors
            v = _scatter_heads(v)
            k_idx = _scatter_heads(k_idx)
            k_val = _scatter_heads(k_val)
            q_idx = _scatter_heads(q_idx)
            q_val = _scatter_heads(q_val)
            g = _scatter_heads(g)
            beta = _scatter_heads(beta)

            # Redefine dimensions for the rest of forward
            H_orig = H
            T_orig = T
            H = H_local
            T = T_local * cp_size  # T_full
            BH = B * H  # B * H_local

        # Get or create memory [BH * sph, hd]
        if cache is not None:
            memory = cache.memory
        else:
            if self.memory is not None:
                if _ulysses_active:
                    # Only expand local heads' memory.
                    # With mean loss (actual training), FSDP dp+cp gradient
                    # averaging is naturally correct — no _GradScale needed.
                    # Each CP rank has grad for local heads only (zero for others);
                    # mean-loss compensates the head-splitting automatically.
                    head_start = cp_rank * H
                    mem_local = self.memory.reshape(-1, hd)[head_start * sph:(head_start + H) * sph]
                    memory = expand_memory_batch(mem_local, B).reshape(BH * sph, hd)
                else:
                    # Standard: expand all heads.
                    # No gradient scaling needed here — for sequential CP, DDP/FSDP
                    # uses dp+cp mesh and correctly averages gradients. For Ulysses,
                    # DDP uses dp-only mesh (set in get_fsdp_mesh) so no CP averaging.
                    memory = expand_memory_batch(
                        self.memory.reshape(self.memory.shape[0], -1), B
                    ).reshape(BH * sph, hd)
            else:
                memory = torch.zeros(BH * sph, hd, device=x.device, dtype=x.dtype)

        # Batch offsets: each of BH elements gets sph slots
        offset = torch.arange(BH, device=x.device).view(BH, 1, 1) * sph
        k_idx = k_idx + offset
        q_idx = q_idx + offset

        # Track pre-update memory norms (only when log_memory_norms=True; .abs() copies full bank)
        self._record_memory_norms_pre(memory)

        # Gated write-read with BH as batch dim
        if _both_active:
            # 2D CP: after the Ulysses head-scatter above, this rank holds
            # H_local heads over T/cp_seq tokens. Run the sequential relay over
            # cp_group (=cp_seq) so the local heads' memory flows in token order
            # across the cp_seq ranks. cp_rank/H here are the cp_uly rank and
            # H_local, so head_start selects this rank's heads from the learned
            # memory in the backward M0 recompute.
            readings, memory = _cp_gated_write_read(
                self, memory, k_idx, k_val, v, beta, g, q_idx, q_val, cp_group,
                h_local=H, head_start=cp_rank * H,
            )
        elif _ulysses_active:
            # Ulysses: each rank has full T for local heads — standard kernel, no CP wrapper
            readings, memory = self.gated_write_read(
                memory, k_idx, k_val, v, beta, g, q_idx, q_val
            )
        elif cp_group is not None:
            # Sequential CP: broadcast-based state passing
            readings, memory = _cp_gated_write_read(
                self, memory, k_idx, k_val, v, beta, g, q_idx, q_val, cp_group
            )
        else:
            readings, memory = self.gated_write_read(
                memory, k_idx, k_val, v, beta, g, q_idx, q_val
            )

        # Per-head normalization of the readings (matching GDN convention)
        if self.readings_norm is not None:
            readings = self.readings_norm(readings)

        # --- Ulysses CP: alltoall readings back ---
        if _ulysses_active:
            # readings: [B*H_local, T_full, hd] → [B*H_orig, T_orig, hd]
            def _gather_heads(x):
                D = x.shape[-1]
                x = x.reshape(B, H, T, D).permute(0, 2, 1, 3)  # [B, T_full, H_local, D]
                x = ulysses_gather_heads(x, cp_ulysses_group)  # [B, T_orig, H_orig, D]
                return x.permute(0, 2, 1, 3).reshape(B * H_orig, T_orig, D)

            readings = _gather_heads(readings)
            # Restore original H/T for the rest of forward
            H = H_orig
            T = T_orig
            BH = B * H

        # Reshape back: [BH, T, hd] → [B, T, D]
        readings = rearrange(readings, '(b h) t d -> b t (h d)', h=H)

        # Output gate: sigmoid(Wog(x)) * readings
        if self.Wog is not None:
            readings = torch.sigmoid(self.Wog(x)) * readings

        # Output projection
        output = self.Wo(readings)

        # Track write statistics (skip under Ulysses CP — k_idx is local-heads only)
        self._record_write_stats(k_idx, k_val, memory, output, BH, H, sph, _ulysses_active)

        # Restore original shape if input was 2D
        if input_was_2d:
            output = output.reshape(B * T, D)

        # Update cache
        if cache is not None:
            cache.update_(memory, seq_len=T)
            new_cache = cache
        else:
            new_cache = SDMLayerState(memory=memory, seq_len=T)

        return output, new_cache

    def init_weights(
        self, init_std: float | None = None, factor: float = 1.0, width_scaling: float | None = None
    ) -> None:
        """Initialize weights.

        CRITICAL: must reinitialize ALL parameters including bare nn.Parameters.
        The training loop calls model.to_empty(device='cuda') which creates
        uninitialized tensors, then calls init_weights() to initialize them.
        """
        in_init_std = (init_std or (self.dim ** (-0.5))) / factor

        for w in [self.Wq_read, self.Wk_write, self.Wv_write, self.Wo,
                  self.a_proj, self.b_proj]:
            nn.init.trunc_normal_(
                w.weight, mean=0.0, std=in_init_std,
                a=-3 * in_init_std, b=3 * in_init_std
            )
            if w.bias is not None:
                nn.init.zeros_(w.bias)

        # Output gate
        if self.Wog is not None:
            nn.init.trunc_normal_(
                self.Wog.weight, mean=0.0, std=in_init_std,
                a=-3 * in_init_std, b=3 * in_init_std
            )
            if self.Wog.bias is not None:
                nn.init.zeros_(self.Wog.bias)

        # Reinitialize bare nn.Parameters (critical for to_empty)
        H = self.num_heads
        A_min, A_max = self.args.A_init_range
        with torch.no_grad():
            self.A_log.uniform_(A_min, A_max).clamp_(min=1e-4).log_()
            # dt_bias from inv_softplus(uniform(0.001, 0.1)) matching FLA
            import math
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            self.dt_bias.uniform_(math.log(dt_min), math.log(dt_max)).exp_().clamp_(min=dt_init_floor)
            self.dt_bias.data.copy_(self.dt_bias.data + torch.log(-torch.expm1(-self.dt_bias.data)))

        if self.memory is not None:
            nn.init.trunc_normal_(
                self.memory,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )

        if self.readings_norm is not None:
            self.readings_norm.reset_parameters()

        # BatchNorm on queries/keys: always standard init (NOT zero-init,
        # unlike output gate norms — queries must produce nonzero values)
        for norm in [self.query_bn, self.key_bn]:
            if norm is not None:
                norm.reset_parameters()

    def create_kv_cache(
        self,
        bsz: int,
        seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device | None = None,
    ) -> SDMLayerState:
        """Create cache for inference."""
        H = self.num_heads
        sph = self.slots_per_head
        hd = self.head_dim
        BH = bsz * H
        if self.memory is not None:
            # Initialize from learned memory (same as forward's no-cache path)
            # Materialize full tensor if sharded (FSDP DTensor), cast to
            # target dtype, and clone in one step to avoid holding two full copies.
            mem = self.memory
            if hasattr(mem, 'full_tensor'):
                mem = mem.full_tensor()
            memory = mem.unsqueeze(0).expand(bsz, -1, -1).reshape(
                BH * sph, hd
            ).to(dtype=dtype).clone()
            del mem
        else:
            memory = torch.zeros(
                BH * sph, hd, dtype=dtype, device=device
            )
        return SDMLayerState(memory=memory, seq_len=seq_len)
