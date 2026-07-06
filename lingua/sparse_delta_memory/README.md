# Gated Memory Controller — Kernel Design Notes

## Architecture Overview

The gated memory controller uses a sparse memory bank with deltanet-style gating.
Each forward pass has two phases:

1. **Phase 1 (batched)**: Compute WY correction quantities — segmented cumsums for per-slot decay, sparse inner products for A/QK matrices, triangular solve for delta_v decomposition.
2. **Phase 2 (sequential loop)**: Per-chunk memory reads, decays, and writes. Each chunk depends on the previous chunk's memory state.

## Key Lessons Learned

### Float32 Cumsum Precision on GPU

**Problem**: When computing per-slot cumulative sums, we batch all chunks together (6.8M elements) to avoid Python loop overhead. The global `cumsum()` on GPU in float32 accumulates errors of ±16 at magnitudes of -19M. This corrupts log-decay values from negative (correct) to positive, causing `exp(positive) >> 1` in the A matrix and NaN via WY solve amplification.

**Root cause**: GPU parallel reduction for cumsum uses a different summation order than CPU sequential cumsum. The parallel tree reduction introduces more rounding error at large element counts.

**Solution**: Use `_segmented_cumsum_batched` which keeps a batch dimension — each chunk (8192 elements) cumsums independently. Float32 cumsum over 8192 elements has precision ±0.0005, which is negligible. No float64 needed.

**Beware**: Any operation that accumulates millions of float32 values on GPU will have this issue. Always prefer batched operations over flattened ones when possible.

### Sparse vs Dense Memory Operations

The memory bank is [num_slots × dim] where num_slots = (dim/4)² (36K to 262K slots). Operations on this table are fundamentally different from dense GDN:

- **Dense GDN**: State is [qk_dim × v_dim] ≈ [64 × 256] per head. Fits in SRAM. FLA puts the entire chunk loop inside a Triton kernel.
- **Sparse memctrl**: State is 57MB-1GB. Random gather/scatter to a huge table. Cannot fit in SRAM. Each operation needs massive parallelism (thousands of programs), not sequential processing.

**Consequence**: Triton kernels that loop over chunks internally (like FLA) are 25x SLOWER for sparse memory because they serialize the parallel gather/scatter operations into ~32 dim-block programs.

### What Works for Kernel Fusion

| Fusion | Result | Why |
|--------|--------|-----|
| 3 gathers → 1 Triton kernel (fused_snapshot_gather) | ✅ 12% faster | Same memory access pattern, saves 2 kernel launches per chunk |
| Gathers + cuBLAS matmuls → 1 Triton kernel | ❌ 10x slower | tl.dot with scratch buffers can't match cuBLAS for [128,128]@[128,dim] |
| Scalar matmul loops in Triton | ❌ 25x slower | 128×128 iterations vs tensor core WMMA |
| Full Phase 2 loop in Triton | ❌ 25x slower | Serializes all operations into few programs |
| CUDA graphs for Phase 2 loop | ❌ 1.5x slower | D2D copies into static buffers cost more than Python dispatch |
| Sequential per-entry decay in Triton | ❌ 8x slower | Only 32 dim-block programs vs 8000 parallel programs |
| Reuse single gather for snapshot + retrieved (gather+einsum vs embedding_bag+gather) | ❌ 0.81-0.86x | embedding_bag fuses gather+weighted reduction without materializing [C,W,dim] intermediate (32MB at dim=2048) |
| Fuse snapshot + decay into one kernel | ❌ race condition | Duplicate indices in flat_k cause some snapshots to see already-decayed values. Instead: `_decay_at_unique_kernel` on unique slots only (decay 8.2x faster, total -5.5% at dim=2048) |

**Rule of thumb**: For sparse memory operations, keep each kernel maximally parallel (one program per slot or entry) and accept the Python loop overhead between chunks. Fuse only operations that share the same access pattern.

### WY Triangular Solve Stability

`M_sys = I + beta * A * tril_mask` where A is the sparse inner product matrix. The triangular solve `M_sys^{-1} @ b` can amplify by `(1 + max(beta*A))^C` where C is chunk size.

- With correct A values (≤ 1 from Cauchy-Schwarz): amplification is bounded
- With corrupted A values (from cumsum precision bug): amplification reaches 1e12 → NaN

The fix is ensuring A values are correct (via the batched cumsum), not clamping the solve output.

### Inference Kernel Design

At inference, skip the autograd Function entirely:
- **Decode (T ≤ 64)**: `_step_write_read` — token-by-token, no WY, no backward tensors
- **Prefill (T > 64)**: `_chunk_write_read_inference` — WY chunks, no snapshot/delta_v saves

Decode went from 4400µs → 780µs (5.6x) by avoiding autograd overhead.

### BatchNorm on PKM Queries

BatchNorm1d after the query/key projections prevents slot collapse (from 100% → 0.2% utilization). But:
- Must NOT be zero-initialized (even with `zero_init_gating=True` for the output gate)
- The standard `reset_parameters()` (weight=1, bias=0) is correct for BN

### Memory Usage

The main memory consumer is `all_mem_before_decay`: [B*T, W, dim] snapshot tensor saved for backward. At level06 (B=12, T=8192, W=64, dim=1536): **18 GB per layer**. Use activation checkpointing to keep only 1 layer's snapshot at a time.

## File Organization

- `gated_memory_controller_layer.py`: Main module — autograd Function, layer class, inference kernels, segmented cumsum
- `memory_controller_ops.py`: Triton kernels — sparse inner products, fused gather, scatter write, decay
- `memory_controller_layer.py`: Original (non-gated) memory controller (separate implementation)

## Speed Benchmarks

All measurements on single H100 80GB, bf16, dim=768, ns=36864, W=R=64.
Timings are mean of 20 runs after 5 warmup iterations.

### 2025-03-02 — after code cleanup refactor

Refactored `GatedSparseMemoryWriteRead.forward` (270→248 lines) and `.backward`
(432→365 lines). Extracted `_pad_inputs` / `_sparse_ip_gated_bwd_vals` helpers,
pre-allocated Phase A buffers (replacing dict-list + torch.stack), vectorized
scatter-back, eliminated redundant second backward loop. All correctness tests
pass (8/8). Speedup comes from removing Python-level overhead in the backward.

| B | T | cs | chunks | fwd (ms) | bwd (ms) | total (ms) |
|---|---|----|--------|----------|----------|------------|
| 1 | 8192 | 32 | 256 | 71.6 | 101.5 | 173.1 |
| 1 | 8192 | 64 | 128 | 36.5 | 72.3 | 108.8 |
| 1 | 8192 | 128 | 64 | 24.3 | 63.9 | 88.2 |
| 2 | 8192 | 32 | 512 | 128.5 | 201.0 | 329.5 |
| 2 | 8192 | 64 | 256 | 66.7 | 132.0 | 198.7 |
| 2 | 8192 | 128 | 128 | 39.1 | 126.9 | 166.1 |
| 4 | 8192 | 32 | 1024 | 252.6 | 398.5 | 651.1 |
| 4 | 8192 | 64 | 512 | 130.7 | 265.4 | 396.1 |
| 4 | 8192 | 128 | 256 | 75.7 | 254.6 | 330.3 |
| 4 | 4096 | 32 | 512 | 133.6 | 202.5 | 336.0 |
| 4 | 4096 | 64 | 256 | 67.9 | 133.3 | 201.2 |
| 4 | 4096 | 128 | 128 | 41.6 | 127.3 | 168.8 |

Compared to pre-refactor (same hardware, same seed):

| Config | Before (ms) | After (ms) | Speedup |
|--------|-------------|------------|---------|
| B=1, T=8192, cs=32 | 187.2 | 173.1 | 7.5% |
| B=1, T=8192, cs=64 | 112.1 | 108.8 | 2.9% |
| B=1, T=8192, cs=128 | 89.3 | 88.2 | 1.2% |
| B=1, T=4096, cs=32 | 95.5 | 87.3 | 8.6% |
| B=1, T=4096, cs=64 | 58.0 | 51.9 | 10.5% |
| B=4, T=4096, cs=32 | 363.5 | 333.8 | 8.2% |
| B=4, T=4096, cs=64 | 218.7 | 202.6 | 7.4% |

Backward is consistently ~12-15% faster (pre-allocated buffers, vectorized
scatter-back, no torch.stack). Forward is ~2-5% faster (removed duplicate
flat_k tensor, _pad_inputs helper).

### 2026-03-05 — Sparse (Gated MemCtrl) vs Dense (GDN) speed comparison

Compares the sparse PKM memory controller against dense GDN at matched state
sizes. All measurements: single H100 80GB, bf16, dim=2048, B=2, T=8192.

**Setup**:
- Gated MemCtrl: ns=262144, W=R=64
- GDN: use_short_conv=False (disabled for fair comparison), mode=chunk

**Gated Memory Controller** (state = 536.9M elements, 1.07 GB):

| cs | fwd (ms) | bwd (ms) | total (ms) |
|----|----------|----------|------------|
| 128 | 68.7 | 206.1 | 274.8 |
| 256 | 58.3 | 217.6 | 275.8 |

**GDN (Dense Deltanet)** — scaling state via qk_dim (up to 256, FLA kernel limit),
v_dim_factor, and n_heads:

| Config | State (M) | fwd (ms) | bwd (ms) | total (ms) |
|--------|-----------|----------|----------|------------|
| 6h qk=64 vf=2 (standard) | 0.05 | 1.1 | 1.6 | 2.7 |
| 6h qk=128 vf=2 | 0.20 | 1.4 | 2.3 | 3.7 |
| 6h qk=256 vf=2 (max qk) | 0.79 | 2.6 | 5.3 | 7.9 |
| 6h qk=256 vf=4 | 1.57 | 4.1 | 8.5 | 12.6 |
| 6h qk=256 vf=8 | 3.15 | 7.1 | 14.8 | 21.9 |
| 6h qk=256 vf=16 | 6.29 | 13.6 | 28.2 | 41.7 |
| 12h qk=256 vf=2 | 1.57 | 4.8 | 9.1 | 13.9 |
| 24h qk=256 vf=2 | 3.15 | 9.1 | 17.4 | 26.5 |
| 12h qk=256 vf=8 | 6.29 | 14.2 | 29.5 | 43.7 |
| 24h qk=256 vf=4 | 6.29 | 15.4 | 32.3 | 47.7 |
| 36h qk=256 vf=4 | 9.44 | 22.9 | 49.1 | 72.0 |
| 48h qk=256 vf=4 | 12.58 | 30.5 | 65.4 | 95.9 |
| 64h+ | >16.8 | — | — | CUDA crash |

**FLA kernel limits**: qk_dim ≤ 256, total value_dim ≤ ~65K. Cannot run GDN at
536.9M state to match MemCtrl. Max achievable: ~12.6M (2.3% of MemCtrl).

**Conclusions**:
1. GDN scales linearly with state size (~7.5 ms/M at dim=2048, B=2).
   Linear extrapolation to 536.9M: **~4000 ms** — roughly **15× slower** than
   MemCtrl's 275 ms at the same state capacity.
2. At 12.6M state (2.3% of MemCtrl's 536.9M), GDN already takes 96 ms.
   MemCtrl handles 43× more state for only 2.9× the time.
3. The sparse advantage grows with dim: at dim=768 the ratio was ~1.4×,
   at dim=2048 it's ~15× (extrapolated). This is because MemCtrl scales
   as O(T × W × dim) while dense GDN scales as O(T × K × V) where K×V
   is the full state size.
4. Projection costs are even more extreme at dim=2048: matching 536.9M state
   via GDN would need absurd projection sizes (dim→millions), while MemCtrl
   projections remain small (2048→1024 for PKM keys).

### 2026-03-05 — Forward/Backward profiling breakdown

Detailed per-section timing of the gated memory controller layer.
All measurements: single H100 80GB, bf16, dim=2048, ns=262144, W=R=64, B=2, T=8192.

**Forward profile (cs=128, wall=68.3ms):**

| Section | ms | % |
|---------|-----|-----|
| Layer projections (gates, PKM keys, Wv, Wq, output gate, Wo) | 8.6 | 12.6% |
| Phase 1: segmented cumsums (cumul_k + cumul_q) | 1.7 | 2.5% |
| Phase 1: sparse inner products (A, QK) | 0.6 | 0.9% |
| Phase 1: M_sys build + post_decay cumsum | 0.6 | 0.9% |
| Phase 1: solve_triangular (delta_v_const, B_matrix, M_sys_inv_T) | 1.6 | 2.4% |
| Phase 2: pre-computation (dedup mask, cast, alloc) | 0.7 | 1.0% |
| **Phase 2: sequential loop** | **54.3** | **79.5%** |
| **TOTAL** | **68.3** | |

**Forward profile (cs=256, wall=57.8ms):**

| Section | ms | % |
|---------|-----|-----|
| Layer projections | 8.6 | 14.8% |
| Phase 1: segmented cumsums | 1.7 | 2.9% |
| Phase 1: sparse inner products (A, QK) | 1.1 | 2.0% |
| Phase 1: M_sys + post_decay | 0.7 | 1.3% |
| Phase 1: solve_triangular | 2.1 | 3.7% |
| Phase 2: pre-computation | 0.7 | 1.2% |
| **Phase 2: sequential loop** | **42.7** | **73.9%** |
| **TOTAL** | **57.8** | |

**Backward profile (cs=128, wall=205.7ms):**

| Section | ms | %wall |
|---------|-----|------|
| Phase A: sequential memory ops (grad_memory gather/scatter, undo writes) | 125.2 | 60.9% |
| Phase B: d_delta_v, solve adjoint, dM_sys, dA, d_beta | 35.8 | 17.4% |
| Phase B: sparse IP backward (CUDA kernels for dA, dQK) | 0.5 | 0.3% |
| Phase B: _phase_b_grad_ops (compiled grad computations) | 31.5 | 15.3% |
| Phase B: segmented cumsums for d_g | 5.8 | 2.8% |
| Phase B: scatter gradients back | 0.5 | 0.2% |
| Layer projection backward | 5.0 | 2.4% |
| **TOTAL** | **205.7** | |

**Backward profile (cs=256, wall=217.4ms):**

| Section | ms | %wall |
|---------|-----|------|
| Phase A: sequential memory ops | 118.6 | 54.6% |
| Phase B: d_delta_v, solve adjoint, dM_sys, dA, d_beta | 36.7 | 16.9% |
| Phase B: sparse IP backward | 0.7 | 0.3% |
| Phase B: _phase_b_grad_ops | 48.3 | 22.2% |
| Phase B: segmented cumsums for d_g | 6.0 | 2.7% |
| Phase B: scatter gradients back | 0.5 | 0.2% |
| Layer projection backward | 5.0 | 2.3% |
| **TOTAL** | **217.4** | |

**Key takeaways:**
1. **Phase 2 sequential loop dominates forward** (74–80%): scatter_reduce for decay,
   fused_snapshot_gather, B_matrix @ retrieved matmul, memory decay write,
   fused_scatter_write, QK @ delta_v matmul — all sequential per-chunk.
2. **Phase A dominates backward** (55–61%): sequential reverse loop gathering
   grad_memory, undoing writes, computing analytical grad_mem_read. At 119–125ms
   this is the single largest cost in the entire fwd+bwd.
3. **Layer projection backward is small** (~5ms, 2.3–2.4%) and constant across
   chunk sizes, as expected.
4. **Phase 1 (batched WY)** is cheap in forward (~5ms, ~8%) — cumsums, sparse IPs,
   and triangular solves are all batched and GPU-parallel.
5. **Phase B grad_ops grows with cs**: 31.5ms at cs=128 → 48.3ms at cs=256,
   because the compiled ops work on larger [256×dim] matrices per group.
6. **cs=256 vs cs=128**: forward loop goes from 54.3→42.7ms (−21%) with fewer
   chunks. Backward Phase A drops slightly (125→119ms) but grad_ops grows,
   resulting in a net slower backward (217 vs 206ms).

### 2026-03-05 — Fast Triton embedding_bag kernels (from Memory Layers at Scale)

Integrated Triton embedding_bag kernels adapted from the "Memory Layers at Scale"
paper (Meta, 2024). The forward kernel achieves near-peak memory bandwidth (~3 TB/s
on H100). The backward uses atomics (7× faster than index_add_) with a
reverse_indices variant available for power-of-2 dims (13× faster).

Replaces `custom_embedding_bag_function` → `fast_embedding_bag` in the read path
and non-gated memory controller. The gated memory controller's Phase 2 loop
already uses a custom Triton `fused_snapshot_gather` kernel for reads, so it's
not affected by this change.

**Benchmark** (H100, bf16, bag_size=64):

| Config | old (ms) | new (ms) | speedup |
|--------|----------|----------|---------|
| ns=36K dim=768 BT=8K | 7.7 | 1.0 | 7.7× |
| ns=36K dim=768 BT=16K | 15.2 | 1.8 | 8.3× |
| ns=262K dim=2048 BT=8K | 21.5 | 3.7 | 5.8× |
| ns=262K dim=2048 BT=16K | 49.9 | 6.9 | 7.2× |

Kernels are in `memory_controller_ops.py`: `fast_embedding_bag`, `FastEmbeddingBag`,
`embedding_bag_fwd_triton`, `embedding_bag_bwd_atomics`, `embedding_bag_bwd_reverse_indices`.

Also added `triton_scatter_add_` (Triton atomics drop-in for `index_add_`) and applied
it to Phase A backward's two scatter-accumulate calls. However, benchmarking showed:

- **Forward reads**: `fused_snapshot_gather` (our existing Triton kernel) is already
  18–27% faster than 3 separate calls (1 gather + 2 `embedding_bag_fwd_triton`),
  because the fused kernel avoids extra kernel launches and reuses loaded memory.
- **Backward scatter**: `triton_scatter_add_` shows no measurable speedup over
  `index_add_` in Phase A because each call only scatters C×W=8192 entries (~32 MB)
  — too small to be bandwidth-bound. The 7× speedup from the embedding_bag benchmark
  was at 524K entries where PyTorch's serial accumulation is the bottleneck.

### 2026-03-06 — Speed comparison: GDN vs MemCtrl at matched qk/value dimensions

Comparison with GDN using the same effective key dimension (qk=64 = W=64 writes)
and total value dimension (2048 = model dim). This matches the per-token FLOP count
of the recurrence (both do W=64 gathers of dim=2048 vectors per token).

All measurements: H100, bf16, dim=2048, T=8192.

**NOTE (2026-05-14)**: The MemCtrl numbers below are from 2026-03-05, BEFORE the
March 24 optimization pass (76% speedup at dim=768, ~15% at dim=1536). The GDN
numbers are unchanged. At dim=768, the gap dropped from 25× to ~13× after optimizations.
At dim=2048 the MemCtrl numbers need re-benchmarking; estimated current gap is ~10×
based on March 14 intermediate benchmarks (B=2 total: 121ms at cs=256) and further
March 24 gains (~15% at large dims). **End-to-end training** slowdown (full model,
3:1 hybrid with 16 SWA + 5 SDM layers) is only **1.6× at 8B scale** (3.15s vs 2.01s
per iter) and **2.0× at 1.4B** (1.56s vs 0.79s).

| Model | B | State | value_dim | fwd+bwd | vs MemCtrl (Mar 5) |
|---|---|---|---|---|---|
| GDN 8h qk=64 vf=4 | 2 | 131K | 2048 | 3.8ms | 35× faster |
| GDN 8h qk=64 vf=4 | 8 | 131K | 2048 | 12.7ms | 42× faster |
| GDN 16h qk=64 vf=2 | 2 | 131K | 2048 | 4.2ms | 32× faster |
| GDN 16h qk=64 vf=2 | 8 | 131K | 2048 | 15.2ms | 35× faster |
| GDN 32h qk=64 vf=1 | 2 | 131K | 2048 | 5.3ms | 25× faster |
| GDN 32h qk=64 vf=1 | 8 | 131K | 2048 | 19.6ms | 27× faster |
| **MemCtrl W=64 (Mar 5)** | **2** | **538M** | **2048** | **133ms** | **1×** |
| **MemCtrl W=64 (Mar 5)** | **8** | **538M** | **2048** | **532ms** | **1×** |
| **MemCtrl W=64 (Mar 14, cs=256)** | **2** | **538M** | **2048** | **121ms** | — |

**Key insight**: GDN is faster per layer at matched qk/value dims (estimated ~10× with
current code), but MemCtrl has 4,100× more state (538M vs 131K elements). Per unit of
state, MemCtrl is ~400× more efficient. The paper reports ~10× lower per-kernel MFU
for SDM vs the highly optimized GDN FLA kernel.

**Why the gap exists**: GDN's state ([qk, head_v] = [64, 256] per head) fits entirely
in GPU registers/SRAM. FLA's Triton kernel processes ALL chunks inside a single kernel
launch with state in registers. MemCtrl's state (262K × 2048 = 1 GB) lives in HBM —
every access is a random HBM read/write with ~400ns latency.

**End-to-end training overhead** (from training logs, all paper models at 16k seq_len):

| Scale | SDM iter | GDN iter | FullAttn iter | SDM/GDN | SDM/FullAttn |
|---|---|---|---|---|---|
| 1.4B (L08, 64 GPUs) | 1.56s | 0.79s | 0.69s | 2.0× | 2.3× |
| 8B (L13, 256 GPUs) | 3.15s | 2.01s | 1.83s | 1.6× | 1.7× |

The per-layer gap is amortized by the 3:1 hybrid architecture (16 SWA + 5 SDM layers)
and shrinks with scale as linear projections (identical across architectures) dominate.

### SPEED TARGET

**Goal: get as close as possible to 1× GDN speed at equivalent per-token FLOPs**
(qk=64, vdim=2048). Current per-kernel gap: ~10× (paper estimate). End-to-end: 1.6× at 8B.

The gap is fundamental: GDN's 131K-element state fits entirely in GPU SRAM
(registers), so the recurrence runs at compute speed. MemCtrl's 538M-element
state (4100× larger) lives in HBM, so every access pays ~400ns random read
latency.

Closing the gap requires architectural changes:
- Reduce memory table size (fewer slots, lower dim, compression, quantization)
- Change addressing to enable coalesced/cached access patterns
- Approximate the recurrence to enable parallel-over-slots processing
- Hardware-aware tiling keeping hot slots in L2 cache across chunks
- Hybrid approaches: small fast state (SRAM) + large slow state (HBM)

### 2026-03-24 — FLA-style bf16 compute + time-first layout + copy elimination

Comprehensive optimization pass. All measurements: single H100 80GB, bf16, dim=768,
ns=36864, W=R=64, B=2, T=8192. Timings are Q1 of 20 runs after warmup.

**Optimizations applied (cumulative):**

1. **cuda_sync**: Eliminated 18 GPU→CPU syncs per fwd+bwd (`.max().item()`, `.any()`)
2. **misc_cleanup**: In-place `tril_()`, `F.pad`, removed redundant `.contiguous()`
3. **save_cumul**: Save `cumul_per_entry_all` from forward, skip backward recomputation
4. **save_deltav**: Save `delta_v` from forward, eliminate backward `solve_triangular`
5. **save_retrieved**: Save `retrieved` from forward, eliminate backward bmm recomputation
6. **hidden_copy_elim**: Pre-compute contiguous group copies, remove redundant `.float()` casts
7. **FLA-style bf16**: All tensors bf16 in global memory, f32 only inside kernel accumulators.
   Modified CUDA kernels (sparse_ip, warp_cooperative_gather) to output bf16.
   Modified `_segmented_cumsum_batched` to upcast internally and return native dtype.
   Triton kernels upcast gate values internally via `.to(tl.float32)`.
8. **Time-first layout**: Changed 4D output tensors from `[P, Nt, C, ...]` (batch-first) to
   `[Nt, P, C, ...]` (time-first). Eliminates 128 strided copy kernels in forward Phase 2
   (kernel outputs write directly to contiguous `[ti]` slices).
9. **Phase B cast caching**: Pre-computed `tril_mask_f32`, cached `wy_MsysInvT_f32`,
   removed no-op `.to(native_dtype)` in `_phase_b_grad_ops`, bf16 bmm with f32 result
   instead of f32 inputs.
10. **Hoisted backward transposes**: 11 per-group `.permute().contiguous()` copies hoisted
    before the group loop (44 copies → 11 copies once).

**Results at dim=768, ns=36864, B=2, T=8192:**

| Version | cs=128 fwd | cs=128 bwd | cs=128 total | cs=128 peak | cs=256 fwd | cs=256 bwd | cs=256 total | cs=256 peak |
|---|---|---|---|---|---|---|---|---|
| 2026-03-02 baseline | 39.1 | 126.9 | 166.1 | — | — | — | — | — |
| 2026-03-24 start-of-session | 18.3 | 38.2 | 56.5 | 4.75 GB | 16.7 | 34.6 | 51.2 | 4.91 GB |
| + cuda_sync + misc_cleanup | 16.5 | 34.8 | 51.4 | 4.40 GB | 16.8 | 32.5 | 49.3 | 4.52 GB |
| + save_cumul/deltav/retrieved | 16.6 | 31.9 | 48.6 | 4.83 GB | 16.9 | 31.5 | 48.4 | 4.92 GB |
| + FLA-style bf16 | 15.8 | 32.0 | 47.8 | 4.51 GB | 16.1 | 30.9 | 47.0 | 4.59 GB |
| + time-first + cached casts | 13.1 | 27.6 | 40.7 | 3.42 GB | 13.6 | 25.9 | 39.5 | 3.50 GB |
| **Total speedup vs session start** | **-28%** | **-28%** | **-28%** | **-1.33 GB** | **-19%** | **-25%** | **-23%** | **-1.41 GB** |
| **Total speedup vs 2026-03-02** | **-66%** | **-78%** | **-76%** | — | — | — | — | — |

**Results at dim=1536, ns=147456, B=1, T=8192, cs=256:**

| Version | fwd (ms) | bwd (ms) | total (ms) | peak |
|---|---|---|---|---|
| 2026-03-24 session start | 13.6 | 23.4 | 37.0 | 7.11 GB |
| + all optimizations | 11.5 | 19.9 | 31.4 | 5.26 GB |
| **Speedup** | **-15%** | **-15%** | **-15%** | **-1.85 GB** |

**What each optimization contributed (at cs=256, dim=768):**

| Optimization | Time saved | Memory saved | How |
|---|---|---|---|
| cuda_sync | -1.9 ms | 0 | Eliminated `.max().item()` GPU→CPU syncs |
| save_deltav | -1.0 ms | +42 MB | Saved delta_v from fwd, eliminated bwd solve_triangular |
| save_retrieved | -0.4 ms | +48 MB | Saved retrieved from fwd, eliminated bwd bmm recompute |
| FLA-style bf16 | -1.4 ms | -330 MB | bf16 global memory, f32 only in kernel accumulators |
| Time-first layout | -6.7 ms | -1.09 GB | Eliminated 128 fwd strided copies + 44 bwd group copies |
| Phase B casts | -0.7 ms | 0 | Cached f32 casts, removed no-ops |

