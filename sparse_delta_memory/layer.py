# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Sparse Delta Memory layer: a gated delta-rule recurrence over a sparse memory bank.

Each token writes to and reads from ``num_writes`` / ``num_reads`` slots of a per-head
memory bank selected by product keys. The forget gate and input gate are applied only
to the slots a token touches.

The chunk-wise training forward uses the WY representation (triangular solve) to
compute the exact intra-chunk delta-rule recurrence, so ``chunk_size`` changes
the output only through floating-point rounding.
"""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from .cache import SDMLayerState, l2_normalize
from .cuda_graph import CudaGraphRunner, cuda_graphs_suspended
from .decode import _decode_launch, sdm_decode_step
from .memory_ops import (DEFAULT_CHUNK, MAX_CHUNK, MAX_WIDTH, MIN_CHUNK, expand_memory_batch, head_merge,
                         head_split, needs_backward, sdm_write_read)
from .memory_stats import SDMStatsMixin
from .phase2_kernels import SNAPSHOT_QUANT_MODES
from .product_key_topk import check_product_key_factors, product_key_topk

logger = logging.getLogger(__name__)

# Largest product-key factor (the selector's column ids are 16 bits wide).
_MAX_PRODUCT_KEY_SIDE = 65535


def _stream_is_capturing(x: torch.Tensor) -> bool:
    """True when the current CUDA stream is capturing a graph (host syncs are illegal)."""
    if not x.is_cuda:
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


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
    `sdm_at` list, not here; this dataclass holds only SDM hyper-parameters.
    """
    dim: int = -1
    num_writes: int = 64
    num_reads: int = 64
    slots_per_head: int = 10000  # Per-head memory slots: side ** 2 for two product-key factors of side <= 65535, or prod(product_key_factor_dims).
    chunk_size: int | None = None  # Kernel chunk size (16..256; None: the op's default, 64): changes results only through floating-point rounding (the intra-chunk recurrence is exact).
    read_act: str = "Softmax"
    write_act: str = "Softmax"
    read_act_scale: float = 1.0
    write_act_scale: float = 1.0
    normalize_readings: bool = False  # LayerNorm on per-head readings before the output projection
    norm_eps: float = 1e-6
    # Gating parameters
    dt_bias_init: float = 0.0  # Unused; kept for config compatibility (dt_bias follows the GDN/FLA init).
    A_init_range: tuple[float, float] = (0.0, 16.0)  # Range for decay parameter init (uniform random, matching GDN/FLA)
    backprop_on_memory: bool = True  # Learn the initial memory (parametric memory).
    output_gate: bool = False  # Sigmoid output gate: sigmoid(Wog(x)) * readings
    query_batchnorm: bool = False  # BatchNorm on PKM query/key projections (prevents slot collapse)
    log_memory_access_stats: bool = False  # Log memory access statistics in eval mode (read/write idx frequency, gate stats)
    log_memory_norms: bool = False  # Log memory norm/max stats (expensive: .abs() copies full memory bank)
    num_heads: int = 1  # Independent memory heads. Each head has (slots_per_head, dim/num_heads) memory.
    key_weighted_decay: bool = False  # Weight forget gate by key values: exp(g * k_val) instead of exp(g)
    snapshot_quant: str = "none"  # Quantize saved snapshot to reduce activation memory: "none" (bank dtype), "fp8", "int4", "bf16" (halves it on float32 banks; "none" on bf16 banks)
    # Replay the write/read of calls that may need a backward from CUDA graphs (see
    # SparseDeltaMemory.forward; under non-reentrant activation checkpointing the layers of one
    # shape share them). Removes host launch overhead; costs static buffers.
    # Calls the graphs cannot serve run eager. On in the debug and flagship configs.
    cuda_graph: bool = False
    # Accumulate the backward's bank gradient (and its gathered grad rows) in float32, cast back to the bank dtype
    # once at the end (no-op for a float32 bank). Forward and snapshot are unchanged; the other gradients
    # differ only through the bank gradient's rounding, which feeds them.
    grad_bank_fp32: bool = False
    # Widths of the per-head product-key factors, any number F >= 1 of them, whose product
    # must equal slots_per_head; the key and query projections then have sum(widths) columns
    # per head. None: two factors of side sqrt(slots_per_head) (a perfect square).
    product_key_factor_dims: list[int] | None = None
    # Former name of chunk_size, still accepted (configs and checkpoints of the previous
    # release): mapped to chunk_size with a DeprecationWarning.
    memory_block_size: int | None = None

    def __post_init__(self):
        if self.memory_block_size is None:
            return
        if self.chunk_size is not None and self.chunk_size != self.memory_block_size:
            raise ValueError(f"chunk_size={self.chunk_size!r} and memory_block_size="
                             f"{self.memory_block_size!r} (its former name) differ; set "
                             f"chunk_size only.")
        warnings.warn("SparseDeltaMemoryArgs.memory_block_size is deprecated: use chunk_size.",
                      DeprecationWarning, stacklevel=3)
        self.chunk_size, self.memory_block_size = self.memory_block_size, None

    def __setstate__(self, state):
        # Unpickling skips __post_init__. Args pickled by the previous release hold
        # memory_block_size (default 256 there, stored explicitly) and no chunk_size: map them as construction does,
        # so such layers keep their chunk size.
        self.__dict__.update(state)
        self.__post_init__()


def _product_key_side(args: SparseDeltaMemoryArgs) -> int:
    """Validate the product-key geometry and return the side of its two square factors."""
    sph = args.slots_per_head
    if isinstance(sph, bool) or not isinstance(sph, int) or sph < 1:
        raise ValueError(f"slots_per_head must be a positive int, got {sph!r}.")
    # Slot a * side + b for factor columns a, b < side: only a square bank is addressable.
    side = math.isqrt(sph)
    if side * side != sph:
        raise ValueError(
            f"slots_per_head={sph} is not a perfect square, so the two-factor product key "
            f"would leave {sph - side * side} slots per head unaddressable. Use "
            f"{side * side} or {(side + 1) ** 2}.")
    if side > _MAX_PRODUCT_KEY_SIDE:
        raise ValueError(f"slots_per_head={sph} exceeds the largest product-key bank, "
                         f"{_MAX_PRODUCT_KEY_SIDE}**2.")
    return side


def _product_key_factors(args: SparseDeltaMemoryArgs) -> tuple[int, ...]:
    """Validate the product-key geometry and return the per-head factor widths: those of
    ``product_key_factor_dims``, or two square factors when it is None."""
    dims = args.product_key_factor_dims
    if dims is None:
        side = _product_key_side(args)
        return (side, side)
    sph = args.slots_per_head
    if isinstance(sph, bool) or not isinstance(sph, int) or sph < 1:
        raise ValueError(f"slots_per_head must be a positive int, got {sph!r}.")
    try:
        dims = check_product_key_factors(dims)
    except ValueError as e:
        raise ValueError(f"product_key_factor_dims: {e}") from None
    if math.prod(dims) != sph:
        raise ValueError(
            f"product_key_factor_dims={list(dims)} address {math.prod(dims)} slots per head, "
            f"but slots_per_head={sph}; set slots_per_head to their product.")
    return dims


def _check_args(args: SparseDeltaMemoryArgs) -> None:
    """Reject settings outside the supported range with the name of the config field."""
    def is_int(n):
        return isinstance(n, int) and not isinstance(n, bool)

    for name in ("num_writes", "num_reads"):
        n = getattr(args, name)
        if not is_int(n) or not 1 <= n <= min(MAX_WIDTH, args.slots_per_head):
            raise ValueError(f"{name} must be an int in 1..min({MAX_WIDTH}, slots_per_head="
                             f"{args.slots_per_head}), got {n!r}.")
    c = args.chunk_size
    if c is not None and (not is_int(c) or not MIN_CHUNK <= c <= MAX_CHUNK):
        raise ValueError(f"chunk_size must be None or an int in {MIN_CHUNK}..{MAX_CHUNK}, "
                         f"got {c!r}.")
    if args.snapshot_quant not in SNAPSHOT_QUANT_MODES:
        raise ValueError(f"snapshot_quant must be one of {SNAPSHOT_QUANT_MODES}, "
                         f"got {args.snapshot_quant!r}.")


class SparseDeltaMemory(SDMStatsMixin, nn.Module):
    """Sparse memory with delta-rule (DeltaNet-style) gated updates.

    - Forget gate g controls exponential decay of memory
    - Input gate beta controls how much new information to integrate
    - Gates only apply to memory slots touched by sparse keys

    Per token:
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
        if H < 1:
            raise ValueError(f"num_heads={H} must be >= 1.")
        if args.dim % H != 0:
            raise ValueError(f"dim={args.dim} must be divisible by num_heads={H}.")
        self.head_dim = args.dim // H

        # Each head owns a bank of slots_per_head rows; the layer bank is
        # num_heads * slots_per_head rows of head_dim.
        self.slots_per_head = args.slots_per_head
        self.product_key_factors = _product_key_factors(args)
        self.key_size_per_head = sum(self.product_key_factors)
        _check_args(args)
        self.key_size = H * self.key_size_per_head
        total_mem_slots = H * self.slots_per_head
        # The kernels address bank rows with int32 slot ids; the op checks the batch
        # multiple B * total_mem_slots at runtime.
        if total_mem_slots >= 2**31:
            raise ValueError(
                f"num_heads * slots_per_head = {total_mem_slots} must be below 2**31, the "
                f"int32 slot-index range of the kernels.")

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

        # Time step bias, initialized from inv_softplus(uniform(dt_min, dt_max)) matching FLA/GDN,
        # so softplus(a_proj(x) + dt_bias) is about 0.001-0.1 at init and g = -A * dt stays
        # moderate (no exp overflow in WY).
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

        # Output gate (sigmoid gating)
        self.Wog = nn.Linear(args.dim, args.dim) if args.output_gate else None

        # Learnable initial memory (when backprop_on_memory=True): H memories of
        # (slots_per_head, head_dim) stored as (H * slots_per_head, head_dim).
        self.memory = (
            nn.Parameter(torch.zeros(total_mem_slots, self.head_dim))
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
        self._stats_skip_warned = False
        self._decode_key = None   # the bank layout sdm_decode_step last checked (gated_write_read)

        self.graph_runner = (CudaGraphRunner(f"SparseDeltaMemory(layer_id={layer_id})")
                             if args.cuda_graph else None)

    def compute_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forget gate g and input gate beta.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            g: Forget gate in log space [B, T, H]
            beta: Input gate [B, T, H]
        """
        a = self.a_proj(x)  # [B, T, H]
        A = -torch.exp(self.A_log)  # [H], negative for decay
        g = A * F.softplus(a + self.dt_bias)  # [B, T, H]
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, H]
        return g, beta

    def _product_keys(
        self,
        x: torch.Tensor,  # [BH, T, key_size_per_head]
        num_keys: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head product-key top-k.

        Returns (val, idx) of shape [BH, T, num_keys], idx local to the head's bank and
        ascending: (..(c_0 * n_1 + c_1) * ..) * n_{F-1} + c_{F-1} for the columns c_f of the
        factors of widths n_f (a * side + b for the default two square factors). Training,
        prefill and decode all select with the same rule.
        """
        return product_key_topk(x, self.product_key_factors, num_keys)

    @property
    def chunk_size(self) -> int:
        """The write/read op's chunk size: ``args.chunk_size``, or the op's default (64)."""
        return DEFAULT_CHUNK if self.args.chunk_size is None else self.args.chunk_size

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gated write + read of all streams; returns (readings [BH, T, head_dim], memory).

        Inputs are [BH, T, ...] with per-stream slot offsets already added to the
        indices; ``memory`` is [BH * slots_per_head, head_dim] and is advanced in place.
        A call that may need a backward (``needs_backward``: grad mode on and an input
        requiring grad, in train and eval mode alike) saves the backward state, so run
        inference under ``torch.no_grad()`` or ``torch.inference_mode()``. Any other call
        saves nothing, and a single token then takes the fused decode step.
        """
        B, T, _ = k_idx.shape
        dim = v.shape[2]
        if T == 1 and not needs_backward(memory, k_val, v, beta, g, q_val):
            key = (B, memory.shape, memory.stride(), memory.dtype, memory.device, v.device)
            if key == self._decode_key:   # the bank sdm_decode_step checked last: no check per token
                readings = _decode_launch(memory, k_idx, k_val, v, beta, g, q_idx, q_val)
            else:                         # a new or swapped bank: the slot rows, the public step's checks, then the memo
                if memory.shape[:1] != (B * self.slots_per_head,):
                    raise ValueError(f"SparseDeltaMemory: the cache memory has shape {tuple(memory.shape)}; this call "
                                     f"needs {B * self.slots_per_head} rows ({B} streams x {self.slots_per_head} slots)")
                readings = sdm_decode_step(memory, k_idx, k_val, v, beta, g, q_idx, q_val)
                self._decode_key = key
        else:
            readings = sdm_write_read(
                memory, k_idx, k_val, v, beta, g, q_idx, q_val,
                chunk_size=self.chunk_size,
                num_memory_slots=self.slots_per_head,
                num_parallel=B,
                snapshot_quant=self.args.snapshot_quant,
                key_weighted_decay=self.args.key_weighted_decay,
                grad_bank_fp32=self.args.grad_bank_fp32,
            )
        return readings.view(B, T, dim), memory

    def _graphed_write_read(self, x, param, cache, inputs):
        """``gated_write_read`` of the initial bank through the layer's graph runner, or None
        when the call runs plain eager: it needs no backward, or starts from a cache. The
        runner replays the graphs, or runs eager inside a graph capture or
        ``suspend_cuda_graphs()``, for another call than the captured one, or after a failed
        capture; either way the call saves the same tensors for backward. Returns the
        readings and the call's state (a ``cuda_graph.ReplayedState`` for a replayed call)."""
        bank = param if cache is None else cache.memory
        if not needs_backward(*(inputs if bank is None else (bank, *inputs))):
            return None
        if cache is not None:
            # The graphs start from the layer's own initial bank.
            self.graph_runner.note("eager_cache",
                                   "a differentiable call that starts from a cache runs eager.")
            return None
        BH, T, _ = inputs[0].shape
        call = (param, (BH * self.slots_per_head, self.head_dim),
                x.dtype if param is None else param.dtype, inputs,
                (self.chunk_size, self.slots_per_head, BH,
                 self.args.snapshot_quant, self.args.grad_bank_fp32))
        if _stream_is_capturing(x) or cuda_graphs_suspended():
            readings, memory = self.graph_runner.eager(*call)
            state = SDMLayerState(memory=memory, seq_len=T)
        else:
            readings, state = self.graph_runner(*call, seq_len=T)
        return readings.view(BH, T, self.head_dim), state

    # The kernels sit behind an autograd.Function with Triton/CUDA launches; dynamo
    # tracing into them only adds compile time. The surrounding model still compiles.
    @torch.compiler.disable
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

        The bank is advanced in place. When the call may need a backward (see
        ``gated_write_read``), the backward restores it to its initial state (with
        ``snapshot_quant`` fp8/int4, the written rows to their dequantized snapshot), so the
        returned state and a passed-in ``cache`` of an eager call hold the final bank only
        until backward runs; clone ``new_cache.memory`` before backward to carry it into the
        next segment. (With ``cuda_graph``, the option's eager fallback keeps its final bank,
        and a replayed call's state returns a copy at its first read, see below.)

        With ``cuda_graph``, a call that may need a backward replays CUDA graphs (see
        cuda_graph.py): the layer's own, or under non-reentrant activation checkpointing a set
        it may share with layers of its shape, whose first call captures the forward graph and
        whose first backward the backward graph. A call of another shape, a
        call with a ``cache``, a call inside ``suspend_cuda_graphs()`` or a capture, and every
        call after a failed capture run eager (``reset_cuda_graphs()`` drops the graphs after
        a permanent change). A replayed call's ``new_cache`` (``cuda_graph.ReplayedState``)
        never exposes the graphs' static bank: the first read of its ``memory`` returns a copy
        of the call's final bank, and raises if a later forward of this layer, of a layer
        sharing its graphs or of a layer captured before it, or the backward, ran since.
        """
        if x.ndim == 2:
            # x is [B*T, D], extract seq_len from freqs_cis
            seq_len, _ = freqs_cis.shape
            bsz = x.shape[0] // seq_len
            x = x.reshape(bsz, seq_len, self.args.dim)
            input_was_2d = True
        else:
            input_was_2d = False

        B, T, D = x.shape
        H = self.num_heads
        sph = self.slots_per_head
        hd = self.head_dim
        BH = B * H

        # Statistics read tensor values on the host, so they are collected only in eval
        # mode and never during CUDA-graph capture.
        stats_on = (self.args.log_memory_access_stats and not self.training
                    and not _stream_is_capturing(x))
        if self.args.log_memory_access_stats and self.training and not self._stats_skip_warned:
            self._stats_skip_warned = True
            logger.warning(
                "SparseDeltaMemory(layer_id=%s): log_memory_access_stats is collected in "
                "eval mode only; nothing is recorded while training.", self.layer_id)

        g, beta = self.compute_gates(x)  # [B, T, H] each
        if stats_on:
            self._record_gate_stats(g, beta)

        v = self.Wv_write(x)  # [B, T, D]

        # Product keys: project, then split into heads.
        raw_k = self.Wk_write(x)  # [B, T, key_size]
        raw_q = self.Wq_read(x)   # [B, T, key_size]
        if self.key_bn is not None:
            raw_k = self.key_bn(raw_k.reshape(B * T, -1)).reshape(B, T, -1)
        if self.query_bn is not None:
            raw_q = self.query_bn(raw_q.reshape(B * T, -1)).reshape(B, T, -1)
        raw_k = head_split(raw_k, H)  # [BH, T, key_size_per_head]
        raw_q = head_split(raw_q, H)

        k_val, k_idx = self._product_keys(raw_k, self.args.num_writes)
        q_val, q_idx = self._product_keys(raw_q, self.args.num_reads)
        q_val = self.read_act(q_val)
        k_val = self.write_act(k_val)

        if stats_on:
            self._record_read_stats(q_idx, q_val, BH, H, sph)

        v = head_split(v, H)  # [BH, T, hd]
        g = rearrange(g, 'b t h -> (b h) t 1')
        beta = rearrange(beta, 'b t h -> (b h) t 1')

        # Key-weighted decay: exp(g * k_val) instead of exp(g), per write.
        if self.args.key_weighted_decay:
            g = g * k_val  # [BH, T, num_writes]

        # Stream offsets: stream (b, h) owns slots [(b*H + h) * sph, (b*H + h + 1) * sph).
        offset = torch.arange(BH, device=x.device).view(BH, 1, 1) * sph
        k_idx = k_idx + offset
        q_idx = q_idx + offset
        inputs = (k_idx, k_val, v, beta, g, q_idx, q_val)

        param = None if self.memory is None else self.memory.reshape(self.memory.shape[0], -1)
        graphed = new_cache = None
        if self.graph_runner is not None:
            graphed = self._graphed_write_read(x, param, cache, inputs)
        if graphed is not None:
            readings, new_cache = graphed
            if stats_on:  # norms of the initial bank: B copies of the parameter, or zeros
                with torch.no_grad():
                    self._record_memory_norms_pre(
                        torch.zeros(BH * sph, hd, device=x.device, dtype=x.dtype)
                        if param is None else expand_memory_batch(param, B).reshape(BH * sph, hd))
        else:
            # Memory [BH * sph, hd]: private per-sequence copies, advanced in place.
            if cache is not None:
                memory = cache.memory
            elif param is not None:
                memory = expand_memory_batch(param, B).reshape(BH * sph, hd)
            else:
                memory = torch.zeros(BH * sph, hd, device=x.device, dtype=x.dtype)
            if stats_on:
                self._record_memory_norms_pre(memory)
            readings, memory = self.gated_write_read(memory, *inputs)
        # A float32 cache bank yields float32 readings; the rest runs in the compute dtype.
        readings = readings.to(v.dtype)

        # Per-head normalization of the readings (matching GDN convention)
        if self.readings_norm is not None:
            readings = self.readings_norm(readings)

        readings = head_merge(readings, H)  # [B, T, D]

        # Output gate: sigmoid(Wog(x)) * readings
        if self.Wog is not None:
            readings = torch.sigmoid(self.Wog(x)) * readings

        output = self.Wo(readings)

        if stats_on:
            self._record_write_stats(k_idx, k_val, memory if graphed is None else new_cache.memory,
                                     output, BH, H, sph)

        if input_was_2d:
            output = output.reshape(B * T, D)

        if cache is not None:
            cache.update_(memory, seq_len=T)
            new_cache = cache
        elif graphed is None:
            new_cache = SDMLayerState(memory=memory, seq_len=T)

        return output, new_cache

    def init_weights(
        self, init_std: float | None = None, factor: float = 1.0, width_scaling: float | None = None
    ) -> None:
        """Initialize weights.

        Must reinitialize ALL parameters including bare nn.Parameters: the training loop
        calls model.to_empty(device='cuda'), which leaves them uninitialized.
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

        if self.Wog is not None:
            nn.init.trunc_normal_(
                self.Wog.weight, mean=0.0, std=in_init_std,
                a=-3 * in_init_std, b=3 * in_init_std
            )
            if self.Wog.bias is not None:
                nn.init.zeros_(self.Wog.bias)

        A_min, A_max = self.args.A_init_range
        with torch.no_grad():
            self.A_log.uniform_(A_min, A_max).clamp_(min=1e-4).log_()
            # dt_bias from inv_softplus(uniform(0.001, 0.1)) matching FLA
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

        # BatchNorm on queries/keys: standard init (queries must produce nonzero values).
        for norm in [self.query_bn, self.key_bn]:
            if norm is not None:
                norm.reset_parameters()

    def create_kv_cache(
        self,
        bsz: int,
        seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device | None = None,
        initial_memory: torch.Tensor | None = None,
    ) -> SDMLayerState:
        """Create the inference state.

        Args:
            initial_memory: optional [bsz * num_heads * slots_per_head, head_dim] bank to
                start from (e.g. the state after a previous segment). It is copied.
                Defaults to the learned initial memory, or zeros without one. A state
                produced by a differentiable forward is valid only until backward (see
                ``forward``).
        """
        H = self.num_heads
        sph = self.slots_per_head
        hd = self.head_dim
        BH = bsz * H
        if initial_memory is not None:
            expected = (BH * sph, hd)
            if tuple(initial_memory.shape) != expected:
                raise ValueError(
                    f"initial_memory must have shape {expected}, got "
                    f"{tuple(initial_memory.shape)}.")
            memory = initial_memory.to(dtype=dtype, device=device).clone()
        elif self.memory is not None:
            # Materialize a sharded (DTensor) parameter, then cast and clone in one step.
            # The parameter may live on another device, so `device` is applied explicitly.
            mem = self.memory
            if hasattr(mem, 'full_tensor'):
                mem = mem.full_tensor()
            memory = mem.unsqueeze(0).expand(bsz, -1, -1).reshape(
                BH * sph, hd
            ).to(dtype=dtype, device=device).clone()
            del mem
        else:
            memory = torch.zeros(BH * sph, hd, dtype=dtype, device=device)
        return SDMLayerState(memory=memory, seq_len=seq_len)
