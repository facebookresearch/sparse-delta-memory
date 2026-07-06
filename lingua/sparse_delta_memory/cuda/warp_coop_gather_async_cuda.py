# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Warp-cooperative gather kernel with cp.async prefetching and double-buffering.

Variant of the warp-cooperative gather that uses cp.async (global->shared memory
bypass) with double-buffering: while computing on buffer A, prefetch next data
into buffer B. Requires SM 80+ (Ampere/Hopper).

Each thread block (1024 threads = 32 warps) processes one token (one row of C).
The 32 warps cooperatively load from 32 different memory slots simultaneously
via cp.async, then reduce across warps in shared memory.

Compiled via torch.utils.cpp_extension.load_inline.
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

// ============================================================
// SM 80+ check (Ampere required for cp.async)
// ============================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// OK — cp.async available
#else
#error "This kernel requires SM 80+ (Ampere or later) for cp.async support."
#endif

// ============================================================
// cp.async intrinsics via PTX inline assembly
// ============================================================

// Copy 16 bytes (8 bf16 elements) from global to shared memory,
// bypassing registers (cache-global policy).
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "n"(16)
        : "memory"
    );
}

// Commit outstanding cp.async operations into a group.
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Wait until at most N groups remain in-flight.
// n must be a compile-time constant for best performance.
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(N >= 0 && N <= 1, "Only N=0 or N=1 supported");
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    } else {
        asm volatile("cp.async.wait_group 1;" ::: "memory");
    }
}

// ============================================================
// bf16 <-> float helpers
// ============================================================
__device__ __forceinline__ float bf16_load_as_float(const at::BFloat16* ptr, int idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
}

__device__ __forceinline__ void store_as_bf16(at::BFloat16* ptr, int idx, float val) {
    reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(val);
}


// ============================================================
// Warp-cooperative gather kernel with cp.async double-buffering
//
// Grid:  (C,)        — one block per token
// Block: BLOCK_THREADS = 1024 = 32 warps
// TILE_D: 256 (bf16 elements processed per dim tile)
//
// Shared memory layout (double-buffered):
//   buf[0]: [NUM_WARPS * TILE_D] bf16 = 32*256*2 = 16 KB
//   buf[1]: [NUM_WARPS * TILE_D] bf16 = 16 KB
//   reduction: [TILE_D] float = 1 KB
// Total: ~33 KB
//
// For each dim tile (d_tile = 0, TILE_D, 2*TILE_D, ...):
//   Process W_K slots in groups of NUM_WARPS (32).
//   Each slot group is prefetched via cp.async, double-buffered.
//
//   Per slot group:
//     1. Wait for current buffer
//     2. Prefetch next group into alternate buffer
//     3. For each warp's slot: store snapshot, accumulate weighted sum
//     4. Warp-level reduction + cross-warp reduction in smem
//
//   After all k-slot groups: process W_Q slots for inter output.
// ============================================================

template<int BLOCK_THREADS, int TILE_D>
__global__ void warp_coop_gather_async_kernel(
    const at::BFloat16* __restrict__ memory,      // [num_slots, dim]
    const int64_t*      __restrict__ k_idx,        // [C, W_K]
    const float*        __restrict__ k_weights,    // [C, W_K]
    const int64_t*      __restrict__ q_idx,        // [C, W_Q]
    const float*        __restrict__ q_weights,    // [C, W_Q]
    at::BFloat16*       __restrict__ snapshot,     // [C, W_K, dim] OUTPUT
    float*              __restrict__ retrieved,    // [C, dim] OUTPUT
    float*              __restrict__ inter,        // [C, dim] OUTPUT
    int C, int W_K, int W_Q, int dim)
{
    static_assert(BLOCK_THREADS == 1024, "Kernel designed for 1024 threads (32 warps)");
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 32
    constexpr int EPL = TILE_D / 32;               // elements per lane = 8

    const int token = blockIdx.x;  // which token (0..C-1)
    if (token >= C) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory: 2 bf16 buffers + 1 float reduction buffer
    // Layout: [2][NUM_WARPS][TILE_D] bf16, then [TILE_D] float
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    // buf[b] starts at smem_bf16[b * NUM_WARPS * TILE_D]
    // reduction buffer starts after both bf16 buffers
    float* reduction_smem = reinterpret_cast<float*>(
        smem_raw + 2 * NUM_WARPS * TILE_D * sizeof(__nv_bfloat16));

    // Pointers to this token's index/weight arrays
    const int64_t* my_k_idx = k_idx + token * W_K;
    const float*   my_k_wt  = k_weights + token * W_K;
    const int64_t* my_q_idx = q_idx + token * W_Q;
    const float*   my_q_wt  = q_weights + token * W_Q;

    // Number of slot groups for k and q
    int num_k_groups = (W_K + NUM_WARPS - 1) / NUM_WARPS;
    int num_q_groups = (W_Q + NUM_WARPS - 1) / NUM_WARPS;

    // ============================================================
    // Iterate over dim tiles
    // ============================================================
    for (int d_start = 0; d_start < dim; d_start += TILE_D) {
        int tile_d = min(TILE_D, dim - d_start);  // actual tile width (may be < TILE_D at end)

        // Zero the reduction buffer for this tile (retrieved accumulator)
        // Each thread zeros EPL elements (or fewer for partial tiles)
        for (int i = tid; i < TILE_D; i += BLOCK_THREADS) {
            if (i < tile_d) {
                reduction_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // ========================================================
        // Phase 1: Process k-slots (snapshot + retrieved)
        // ========================================================
        int buf_idx = 0;

        // Prefetch first k-slot group into buf[0]
        {
            int slot_group = 0;
            int slot_n = slot_group * NUM_WARPS + warp_id;
            if (slot_n < W_K) {
                int64_t slot = my_k_idx[slot_n];
                const __nv_bfloat16* gmem_row = reinterpret_cast<const __nv_bfloat16*>(memory)
                    + slot * dim + d_start;
                __nv_bfloat16* smem_dst = smem_bf16 + buf_idx * NUM_WARPS * TILE_D
                    + warp_id * TILE_D;
                // Each lane issues cp.async for its EPL elements (16 bytes = 8 bf16)
                for (int e = 0; e < EPL; e += 8) {
                    int elem_off = lane_id * EPL + e;
                    if (elem_off < tile_d) {
                        cp_async_cg_16(&smem_dst[elem_off], &gmem_row[elem_off]);
                    }
                }
            }
            // If slot_n >= W_K, this warp issues no cp.async (its smem region is unused)
        }
        cp_async_commit_group();

        for (int slot_group = 0; slot_group < num_k_groups; slot_group++) {
            bool has_next = (slot_group + 1) < num_k_groups;

            // CORRECT double-buffering: prefetch BEFORE wait.
            // This ensures 2 groups are in flight, so wait_group<1>
            // forces the OLDER group (current) to complete while the
            // NEWER group (next) may remain in flight.
            if (has_next) {
                int next_buf = 1 - buf_idx;
                int next_slot_n = (slot_group + 1) * NUM_WARPS + warp_id;
                if (next_slot_n < W_K) {
                    int64_t slot = my_k_idx[next_slot_n];
                    const __nv_bfloat16* gmem_row = reinterpret_cast<const __nv_bfloat16*>(memory)
                        + slot * dim + d_start;
                    __nv_bfloat16* smem_dst = smem_bf16 + next_buf * NUM_WARPS * TILE_D
                        + warp_id * TILE_D;
                    for (int e = 0; e < EPL; e += 8) {
                        int elem_off = lane_id * EPL + e;
                        if (elem_off < tile_d) {
                            cp_async_cg_16(&smem_dst[elem_off], &gmem_row[elem_off]);
                        }
                    }
                }
                cp_async_commit_group();
                cp_async_wait_group<1>();  // 2 in flight → older (current) completes
            } else {
                cp_async_wait_group<0>();  // last group, wait for all
            }
            __syncthreads();

            // ---- COMPUTE on current buffer ----

            // Each warp processes its own slot from the current buffer.
            // warp_id's slot index in the global k-slot array:
            int slot_n = slot_group * NUM_WARPS + warp_id;

            if (slot_n < W_K) {
                float weight = my_k_wt[slot_n];
                __nv_bfloat16* src = smem_bf16 + buf_idx * NUM_WARPS * TILE_D
                    + warp_id * TILE_D;

                // Snapshot store + weighted accumulation
                // Each lane handles EPL contiguous elements
                float local_acc[EPL];
                #pragma unroll
                for (int e = 0; e < EPL; e++) {
                    local_acc[e] = 0.0f;
                }

                #pragma unroll
                for (int e = 0; e < EPL; e++) {
                    int elem_off = lane_id * EPL + e;
                    if (elem_off < tile_d) {
                        __nv_bfloat16 val_bf16 = src[elem_off];
                        float fval = __bfloat162float(val_bf16);

                        // Store snapshot to global memory
                        int snap_idx = (token * W_K + slot_n) * dim + d_start + elem_off;
                        reinterpret_cast<__nv_bfloat16*>(snapshot)[snap_idx] = val_bf16;

                        // Weighted value for retrieved accumulation
                        local_acc[e] = weight * fval;
                    }
                }

                // Accumulate this warp's weighted values into the shared reduction buffer.
                // We need to reduce across all NUM_WARPS warps. Use atomicAdd into
                // the shared float reduction buffer.
                #pragma unroll
                for (int e = 0; e < EPL; e++) {
                    int elem_off = lane_id * EPL + e;
                    if (elem_off < tile_d) {
                        atomicAdd(&reduction_smem[elem_off], local_acc[e]);
                    }
                }
            }

            __syncthreads();  // ensure all warps done before buffer swap
            buf_idx = 1 - buf_idx;
        }

        // Write retrieved output for this tile from reduction buffer
        for (int i = tid; i < tile_d; i += BLOCK_THREADS) {
            retrieved[token * dim + d_start + i] = reduction_smem[i];
        }
        __syncthreads();

        // ========================================================
        // Phase 2: Process q-slots (inter output)
        // ========================================================

        // Zero reduction buffer for inter
        for (int i = tid; i < TILE_D; i += BLOCK_THREADS) {
            if (i < tile_d) {
                reduction_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        buf_idx = 0;

        // Prefetch first q-slot group
        {
            int slot_n = warp_id;  // slot_group=0
            if (slot_n < W_Q) {
                int64_t slot = my_q_idx[slot_n];
                const __nv_bfloat16* gmem_row = reinterpret_cast<const __nv_bfloat16*>(memory)
                    + slot * dim + d_start;
                __nv_bfloat16* smem_dst = smem_bf16 + buf_idx * NUM_WARPS * TILE_D
                    + warp_id * TILE_D;
                for (int e = 0; e < EPL; e += 8) {
                    int elem_off = lane_id * EPL + e;
                    if (elem_off < tile_d) {
                        cp_async_cg_16(&smem_dst[elem_off], &gmem_row[elem_off]);
                    }
                }
            }
        }
        cp_async_commit_group();

        for (int slot_group = 0; slot_group < num_q_groups; slot_group++) {
            bool has_next = (slot_group + 1) < num_q_groups;

            // Prefetch BEFORE wait (same fix as k-slot loop)
            if (has_next) {
                int next_buf = 1 - buf_idx;
                int next_slot_n = (slot_group + 1) * NUM_WARPS + warp_id;
                if (next_slot_n < W_Q) {
                    int64_t slot = my_q_idx[next_slot_n];
                    const __nv_bfloat16* gmem_row = reinterpret_cast<const __nv_bfloat16*>(memory)
                        + slot * dim + d_start;
                    __nv_bfloat16* smem_dst = smem_bf16 + next_buf * NUM_WARPS * TILE_D
                        + warp_id * TILE_D;
                    for (int e = 0; e < EPL; e += 8) {
                        int elem_off = lane_id * EPL + e;
                        if (elem_off < tile_d) {
                            cp_async_cg_16(&smem_dst[elem_off], &gmem_row[elem_off]);
                        }
                    }
                }
                cp_async_commit_group();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();

            // COMPUTE on current buffer (q-slots, no snapshot needed)
            int slot_n = slot_group * NUM_WARPS + warp_id;

            if (slot_n < W_Q) {
                float weight = my_q_wt[slot_n];
                __nv_bfloat16* src = smem_bf16 + buf_idx * NUM_WARPS * TILE_D
                    + warp_id * TILE_D;

                #pragma unroll
                for (int e = 0; e < EPL; e++) {
                    int elem_off = lane_id * EPL + e;
                    if (elem_off < tile_d) {
                        float fval = __bfloat162float(src[elem_off]);
                        atomicAdd(&reduction_smem[elem_off], weight * fval);
                    }
                }
            }

            __syncthreads();
            buf_idx = 1 - buf_idx;
        }

        // Write inter output for this tile
        for (int i = tid; i < tile_d; i += BLOCK_THREADS) {
            inter[token * dim + d_start + i] = reduction_smem[i];
        }
        __syncthreads();

    }  // end dim tile loop
}


// ============================================================
// C++ wrapper
// ============================================================
void warp_coop_gather_async(
    torch::Tensor memory,      // [num_slots, dim] bf16
    torch::Tensor k_idx,       // [C, W_K] int64
    torch::Tensor k_weights,   // [C, W_K] float32
    torch::Tensor q_idx,       // [C, W_Q] int64
    torch::Tensor q_weights,   // [C, W_Q] float32
    torch::Tensor snapshot,    // [C, W_K, dim] bf16 — OUTPUT
    torch::Tensor retrieved,   // [C, dim] float32 — OUTPUT
    torch::Tensor inter        // [C, dim] float32 — OUTPUT
)
{
    TORCH_CHECK(memory.dtype() == torch::kBFloat16, "memory must be bf16");
    TORCH_CHECK(k_idx.dtype() == torch::kInt64, "k_idx must be int64");
    TORCH_CHECK(k_weights.dtype() == torch::kFloat32, "k_weights must be float32");
    TORCH_CHECK(q_idx.dtype() == torch::kInt64, "q_idx must be int64");
    TORCH_CHECK(q_weights.dtype() == torch::kFloat32, "q_weights must be float32");
    TORCH_CHECK(snapshot.dtype() == torch::kBFloat16, "snapshot must be bf16");
    TORCH_CHECK(retrieved.dtype() == torch::kFloat32, "retrieved must be float32");
    TORCH_CHECK(inter.dtype() == torch::kFloat32, "inter must be float32");

    int C = k_idx.size(0);
    int W_K = k_idx.size(1);
    int W_Q = q_idx.size(1);
    int dim = memory.size(1);

    constexpr int BLOCK_THREADS = 1024;
    constexpr int TILE_D = 256;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 32

    // Shared memory: 2 bf16 buffers + 1 float reduction buffer
    // = 2 * NUM_WARPS * TILE_D * sizeof(bf16) + TILE_D * sizeof(float)
    size_t smem_bytes = 2 * NUM_WARPS * TILE_D * sizeof(__nv_bfloat16)
                      + TILE_D * sizeof(float);

    // Ensure we can use this much shared memory
    cudaFuncSetAttribute(
        warp_coop_gather_async_kernel<BLOCK_THREADS, TILE_D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(C);
    dim3 block(BLOCK_THREADS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    warp_coop_gather_async_kernel<BLOCK_THREADS, TILE_D><<<grid, block, smem_bytes, stream>>>(
        memory.data_ptr<at::BFloat16>(),
        k_idx.data_ptr<int64_t>(),
        k_weights.data_ptr<float>(),
        q_idx.data_ptr<int64_t>(),
        q_weights.data_ptr<float>(),
        snapshot.data_ptr<at::BFloat16>(),
        retrieved.data_ptr<float>(),
        inter.data_ptr<float>(),
        C, W_K, W_Q, dim);
}
"""

_CPP_SOURCE = r"""
void warp_coop_gather_async(
    torch::Tensor memory,
    torch::Tensor k_idx,
    torch::Tensor k_weights,
    torch::Tensor q_idx,
    torch::Tensor q_weights,
    torch::Tensor snapshot,
    torch::Tensor retrieved,
    torch::Tensor inter);
"""

# Compile once, cached by PyTorch
_warp_coop_async_cuda = None


def _get_cuda_module():
    global _warp_coop_async_cuda
    if _warp_coop_async_cuda is None:
        _warp_coop_async_cuda = load_inline(
            name="warp_coop_gather_async",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["warp_coop_gather_async"],
            verbose=False,
        )
    return _warp_coop_async_cuda


def warp_coop_gather_async(
    memory: torch.Tensor,      # [num_slots, dim] bf16
    k_idx: torch.Tensor,       # [C, W_K] int64
    k_weights: torch.Tensor,   # [C, W_K] float32
    q_idx: torch.Tensor,       # [C, W_Q] int64
    q_weights: torch.Tensor,   # [C, W_Q] float32
    snapshot: torch.Tensor,    # [C, W_K, dim] bf16 -- OUTPUT
    retrieved: torch.Tensor,   # [C, dim] float32 -- OUTPUT
    inter: torch.Tensor,       # [C, dim] float32 -- OUTPUT
):
    """Warp-cooperative gather with cp.async prefetching and double-buffering.

    Gathers from a bf16 memory bank at sparse indices, producing:
      - snapshot[t, n, :] = memory[k_idx[t, n], :]                       (bf16 copy)
      - retrieved[t, :] = sum_n k_weights[t, n] * memory[k_idx[t, n], :] (float32 weighted sum)
      - inter[t, :] = sum_n q_weights[t, n] * memory[q_idx[t, n], :]     (float32 weighted sum)

    Uses cp.async (SM 80+) to asynchronously load memory rows from global to shared
    memory, with double-buffering to overlap loads with computation.

    All output tensors must be pre-allocated by the caller.
    """
    mod = _get_cuda_module()
    mod.warp_coop_gather_async(
        memory.contiguous(),
        k_idx.contiguous(),
        k_weights.contiguous(),
        q_idx.contiguous(),
        q_weights.contiguous(),
        snapshot,
        retrieved,
        inter,
    )
