# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CUDA warp-cooperative gather kernel for fused snapshot + weighted gather.

Replaces the Triton `_fused_snapshot_gather_kernel` with a CUDA kernel that
uses multiple warps within a block to load from different memory slots
simultaneously, increasing memory-level parallelism.

Design: Grid (C, num_d_tiles), Block configurable (128/256/512/1024 threads).
Each block handles one (token, dim_tile) pair. Within a block, N warps
simultaneously load from N different memory slots, giving Nx more outstanding
random memory requests than sequential loading.

Compiled via torch.utils.cpp_extension.load_inline (same pattern as sparse_ip_cuda.py).
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

// ============================================================
// Warp-cooperative gather kernel — 2D grid, configurable block size
//
// Grid:  (C, num_d_tiles)
// Block: BLOCK_THREADS (template param: 128, 256, 512, or 1024)
// TILE_D: 256 (elements per dim tile)
//
// Each block processes one (token, dim_tile) pair.
// NUM_WARPS = BLOCK_THREADS/32 warps load from different memory slots
// simultaneously, then reduce across warps via shared memory.
// ============================================================

template<int BLOCK_THREADS, int TILE_D>
__global__ void warp_cooperative_gather_bf16_kernel(
    const at::BFloat16* __restrict__ memory,       // [num_slots, dim]
    const int64_t*      __restrict__ k_idx,        // [P, C, W_K] strided
    const at::BFloat16* __restrict__ k_weights,    // [P, C, W_K] strided
    const int64_t*      __restrict__ q_idx,        // [P, C, W_Q] strided
    const at::BFloat16* __restrict__ q_weights,    // [P, C, W_Q] strided
    at::BFloat16*       __restrict__ snapshot,     // [SC, W_K, dim] contiguous
    at::BFloat16*       __restrict__ retrieved,    // [SC, dim] contiguous
    at::BFloat16*       __restrict__ inter,        // [SC, dim] contiguous
    int SC, int C_local, int W_K, int W_Q, int dim,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b)
{
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
    constexpr int EPL = TILE_D / WARP_SIZE;  // elements per lane

    extern __shared__ float smem[];  // [NUM_WARPS * TILE_D]

    const int t = blockIdx.x;       // global token index [0, SC)
    const int d_tile = blockIdx.y;  // dim tile index
    if (t >= SC) return;

    // Batch offset for strided inputs
    const int b = t / C_local;
    const int t_local = t % C_local;

    const int d_start = d_tile * TILE_D;
    if (d_start >= dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const __nv_bfloat16* k_wt_raw = reinterpret_cast<const __nv_bfloat16*>(k_weights);
    const __nv_bfloat16* q_wt_raw = reinterpret_cast<const __nv_bfloat16*>(q_weights);
    const __nv_bfloat16* mem_raw = reinterpret_cast<const __nv_bfloat16*>(memory);

    // === K-gather: snapshot + retrieved ===
    float local_acc[EPL];
    #pragma unroll
    for (int i = 0; i < EPL; i++) local_acc[i] = 0.0f;

    __nv_bfloat16* snap_raw = reinterpret_cast<__nv_bfloat16*>(snapshot);
    const bool can_vectorize = (d_start + TILE_D <= dim);

    for (int slot_group = 0; slot_group < W_K; slot_group += NUM_WARPS) {
        int n = slot_group + warp_id;
        if (n < W_K) {
            int64_t slot = k_idx[b * stride_k_b + t_local * W_K + n];
            float weight = __bfloat162float(k_wt_raw[b * stride_kw_b + t_local * W_K + n]);
            int64_t base_mem = slot * dim + d_start + lane_id * EPL;
            int64_t base_snap = ((int64_t)t * W_K + n) * dim + d_start + lane_id * EPL;

            if (can_vectorize) {
                // Vectorized path: load 8 bf16 (16 bytes) as int4
                int4 packed = *reinterpret_cast<const int4*>(&mem_raw[base_mem]);
                // Vectorized snapshot store
                *reinterpret_cast<int4*>(&snap_raw[base_snap]) = packed;
                // Extract and accumulate
                __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    local_acc[i] += weight * __bfloat162float(vals[i]);
                }
            } else {
                // Scalar fallback for partial dim tiles
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    int d = d_start + lane_id * EPL + i;
                    if (d < dim) {
                        float val = __bfloat162float(mem_raw[slot * dim + d]);
                        snap_raw[((int64_t)t * W_K + n) * dim + d] = __float2bfloat16(val);
                        local_acc[i] += weight * val;
                    }
                }
            }
        }
    }

    // Write to smem for cross-warp reduction
    #pragma unroll
    for (int i = 0; i < EPL; i++) {
        smem[warp_id * TILE_D + lane_id * EPL + i] = local_acc[i];
    }
    __syncthreads();

    // Reduce: threads cooperatively sum across all warps
    // Use strided loop to handle BLOCK_THREADS < TILE_D
    for (int pos = tid; pos < TILE_D; pos += BLOCK_THREADS) {
        int d = d_start + pos;
        if (d < dim) {
            float sum = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                sum += smem[w * TILE_D + pos];
            }
            reinterpret_cast<__nv_bfloat16*>(retrieved)[t * dim + d] = __float2bfloat16(sum);
        }
    }
    __syncthreads();

    // === Q-gather: inter ===
    #pragma unroll
    for (int i = 0; i < EPL; i++) local_acc[i] = 0.0f;

    for (int slot_group = 0; slot_group < W_Q; slot_group += NUM_WARPS) {
        int n = slot_group + warp_id;
        if (n < W_Q) {
            int64_t slot = q_idx[b * stride_q_b + t_local * W_Q + n];
            float weight = __bfloat162float(q_wt_raw[b * stride_qw_b + t_local * W_Q + n]);
            int64_t base_mem = slot * dim + d_start + lane_id * EPL;

            if (can_vectorize) {
                int4 packed = *reinterpret_cast<const int4*>(&mem_raw[base_mem]);
                __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    local_acc[i] += weight * __bfloat162float(vals[i]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    int d = d_start + lane_id * EPL + i;
                    if (d < dim) {
                        local_acc[i] += weight * __bfloat162float(mem_raw[slot * dim + d]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < EPL; i++) {
        smem[warp_id * TILE_D + lane_id * EPL + i] = local_acc[i];
    }
    __syncthreads();

    for (int pos = tid; pos < TILE_D; pos += BLOCK_THREADS) {
        int d = d_start + pos;
        if (d < dim) {
            float sum = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                sum += smem[w * TILE_D + pos];
            }
            reinterpret_cast<__nv_bfloat16*>(inter)[t * dim + d] = __float2bfloat16(sum);
        }
    }
}

// ============================================================
// f32-output variant: retrieved and inter are float32,
// snapshot stays bf16 (memory dtype for backward).
// ============================================================

template<int BLOCK_THREADS, int TILE_D>
__global__ void warp_cooperative_gather_f32out_kernel(
    const at::BFloat16* __restrict__ memory,       // [num_slots, dim]
    const int64_t*      __restrict__ k_idx,        // [P, C, W_K] strided
    const at::BFloat16* __restrict__ k_weights,    // [P, C, W_K] strided
    const int64_t*      __restrict__ q_idx,        // [P, C, W_Q] strided
    const at::BFloat16* __restrict__ q_weights,    // [P, C, W_Q] strided
    at::BFloat16*       __restrict__ snapshot,     // [SC, W_K, dim] contiguous — bf16
    float*              __restrict__ retrieved,    // [SC, dim] contiguous — f32
    float*              __restrict__ inter,        // [SC, dim] contiguous — f32
    int SC, int C_local, int W_K, int W_Q, int dim,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b)
{
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;
    constexpr int EPL = TILE_D / WARP_SIZE;  // elements per lane

    extern __shared__ float smem[];  // [NUM_WARPS * TILE_D]

    const int t = blockIdx.x;       // global token index [0, SC)
    const int d_tile = blockIdx.y;  // dim tile index
    if (t >= SC) return;

    // Batch offset for strided inputs
    const int b = t / C_local;
    const int t_local = t % C_local;

    const int d_start = d_tile * TILE_D;
    if (d_start >= dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const __nv_bfloat16* k_wt_raw = reinterpret_cast<const __nv_bfloat16*>(k_weights);
    const __nv_bfloat16* q_wt_raw = reinterpret_cast<const __nv_bfloat16*>(q_weights);
    const __nv_bfloat16* mem_raw = reinterpret_cast<const __nv_bfloat16*>(memory);

    // === K-gather: snapshot (bf16) + retrieved (f32) ===
    float local_acc[EPL];
    #pragma unroll
    for (int i = 0; i < EPL; i++) local_acc[i] = 0.0f;

    __nv_bfloat16* snap_raw = reinterpret_cast<__nv_bfloat16*>(snapshot);
    const bool can_vectorize = (d_start + TILE_D <= dim);

    for (int slot_group = 0; slot_group < W_K; slot_group += NUM_WARPS) {
        int n = slot_group + warp_id;
        if (n < W_K) {
            int64_t slot = k_idx[b * stride_k_b + t_local * W_K + n];
            float weight = __bfloat162float(k_wt_raw[b * stride_kw_b + t_local * W_K + n]);
            int64_t base_mem = slot * dim + d_start + lane_id * EPL;
            int64_t base_snap = ((int64_t)t * W_K + n) * dim + d_start + lane_id * EPL;

            if (can_vectorize) {
                // Vectorized path: load 8 bf16 (16 bytes) as int4
                int4 packed = *reinterpret_cast<const int4*>(&mem_raw[base_mem]);
                // Vectorized snapshot store (bf16)
                *reinterpret_cast<int4*>(&snap_raw[base_snap]) = packed;
                // Extract and accumulate
                __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    local_acc[i] += weight * __bfloat162float(vals[i]);
                }
            } else {
                // Scalar fallback for partial dim tiles
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    int d = d_start + lane_id * EPL + i;
                    if (d < dim) {
                        float val = __bfloat162float(mem_raw[slot * dim + d]);
                        snap_raw[((int64_t)t * W_K + n) * dim + d] = __float2bfloat16(val);
                        local_acc[i] += weight * val;
                    }
                }
            }
        }
    }

    // Write to smem for cross-warp reduction
    #pragma unroll
    for (int i = 0; i < EPL; i++) {
        smem[warp_id * TILE_D + lane_id * EPL + i] = local_acc[i];
    }
    __syncthreads();

    // Reduce: write f32 directly to retrieved
    for (int pos = tid; pos < TILE_D; pos += BLOCK_THREADS) {
        int d = d_start + pos;
        if (d < dim) {
            float sum = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                sum += smem[w * TILE_D + pos];
            }
            retrieved[t * dim + d] = sum;  // f32 store
        }
    }
    __syncthreads();

    // === Q-gather: inter (f32) ===
    #pragma unroll
    for (int i = 0; i < EPL; i++) local_acc[i] = 0.0f;

    for (int slot_group = 0; slot_group < W_Q; slot_group += NUM_WARPS) {
        int n = slot_group + warp_id;
        if (n < W_Q) {
            int64_t slot = q_idx[b * stride_q_b + t_local * W_Q + n];
            float weight = __bfloat162float(q_wt_raw[b * stride_qw_b + t_local * W_Q + n]);
            int64_t base_mem = slot * dim + d_start + lane_id * EPL;

            if (can_vectorize) {
                int4 packed = *reinterpret_cast<const int4*>(&mem_raw[base_mem]);
                __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    local_acc[i] += weight * __bfloat162float(vals[i]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < EPL; i++) {
                    int d = d_start + lane_id * EPL + i;
                    if (d < dim) {
                        local_acc[i] += weight * __bfloat162float(mem_raw[slot * dim + d]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < EPL; i++) {
        smem[warp_id * TILE_D + lane_id * EPL + i] = local_acc[i];
    }
    __syncthreads();

    for (int pos = tid; pos < TILE_D; pos += BLOCK_THREADS) {
        int d = d_start + pos;
        if (d < dim) {
            float sum = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                sum += smem[w * TILE_D + pos];
            }
            inter[t * dim + d] = sum;  // f32 store
        }
    }
}

// ============================================================
// C++ wrappers — dispatch block sizes
// ============================================================

template<int BLOCK_THREADS>
void launch_kernel(
    torch::Tensor memory, torch::Tensor k_idx, torch::Tensor k_weights,
    torch::Tensor q_idx, torch::Tensor q_weights,
    torch::Tensor snapshot, torch::Tensor retrieved, torch::Tensor inter,
    int SC, int C_local, int W_K, int W_Q, int dim,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b)
{
    constexpr int TILE_D = 256;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    int num_d_tiles = (dim + TILE_D - 1) / TILE_D;
    int smem_bytes = NUM_WARPS * TILE_D * sizeof(float);

    cudaFuncSetAttribute(
        warp_cooperative_gather_bf16_kernel<BLOCK_THREADS, TILE_D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(SC, num_d_tiles);
    dim3 block(BLOCK_THREADS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    warp_cooperative_gather_bf16_kernel<BLOCK_THREADS, TILE_D><<<grid, block, smem_bytes, stream>>>(
        memory.data_ptr<at::BFloat16>(),
        k_idx.data_ptr<int64_t>(),
        k_weights.data_ptr<at::BFloat16>(),
        q_idx.data_ptr<int64_t>(),
        q_weights.data_ptr<at::BFloat16>(),
        snapshot.data_ptr<at::BFloat16>(),
        retrieved.data_ptr<at::BFloat16>(),
        inter.data_ptr<at::BFloat16>(),
        SC, C_local, W_K, W_Q, dim,
        stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
}

void warp_cooperative_gather(
    torch::Tensor memory,
    torch::Tensor k_idx, torch::Tensor k_weights,
    torch::Tensor q_idx, torch::Tensor q_weights,
    torch::Tensor snapshot, torch::Tensor retrieved, torch::Tensor inter,
    int block_threads,
    int C_local,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b)
{
    TORCH_CHECK(memory.dtype() == torch::kBFloat16, "memory must be bf16");
    TORCH_CHECK(k_idx.dtype() == torch::kInt64, "k_idx must be int64");
    TORCH_CHECK(k_weights.dtype() == torch::kBFloat16, "k_weights must be bf16");

    int SC = k_idx.size(0);
    int W_K = k_idx.size(1);
    int W_Q = q_idx.size(1);
    int dim = memory.size(1);

    switch (block_threads) {
        case 64:
            launch_kernel<64>(memory, k_idx, k_weights, q_idx, q_weights,
                              snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                              stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 128:
            launch_kernel<128>(memory, k_idx, k_weights, q_idx, q_weights,
                               snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                               stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 256:
            launch_kernel<256>(memory, k_idx, k_weights, q_idx, q_weights,
                               snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                               stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 512:
            launch_kernel<512>(memory, k_idx, k_weights, q_idx, q_weights,
                               snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                               stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 1024:
            launch_kernel<1024>(memory, k_idx, k_weights, q_idx, q_weights,
                                snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                                stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        default:
            TORCH_CHECK(false, "block_threads must be 64, 128, 256, 512, or 1024");
    }
}

// ============================================================
// f32-output launch helper + wrapper
// ============================================================

template<int BLOCK_THREADS>
void launch_kernel_f32out(
    torch::Tensor memory, torch::Tensor k_idx, torch::Tensor k_weights,
    torch::Tensor q_idx, torch::Tensor q_weights,
    torch::Tensor snapshot, torch::Tensor retrieved, torch::Tensor inter,
    int SC, int C_local, int W_K, int W_Q, int dim,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b)
{
    constexpr int TILE_D = 256;
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    int num_d_tiles = (dim + TILE_D - 1) / TILE_D;
    int smem_bytes = NUM_WARPS * TILE_D * sizeof(float);

    cudaFuncSetAttribute(
        warp_cooperative_gather_f32out_kernel<BLOCK_THREADS, TILE_D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(SC, num_d_tiles);
    dim3 block(BLOCK_THREADS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    warp_cooperative_gather_f32out_kernel<BLOCK_THREADS, TILE_D><<<grid, block, smem_bytes, stream>>>(
        memory.data_ptr<at::BFloat16>(),
        k_idx.data_ptr<int64_t>(),
        k_weights.data_ptr<at::BFloat16>(),
        q_idx.data_ptr<int64_t>(),
        q_weights.data_ptr<at::BFloat16>(),
        snapshot.data_ptr<at::BFloat16>(),
        retrieved.data_ptr<float>(),
        inter.data_ptr<float>(),
        SC, C_local, W_K, W_Q, dim,
        stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
}

void warp_cooperative_gather_f32out(
    torch::Tensor memory,
    torch::Tensor k_idx, torch::Tensor k_weights,
    torch::Tensor q_idx, torch::Tensor q_weights,
    torch::Tensor snapshot, torch::Tensor retrieved, torch::Tensor inter,
    int block_threads,
    int C_local,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b)
{
    TORCH_CHECK(memory.dtype() == torch::kBFloat16, "memory must be bf16");
    TORCH_CHECK(k_idx.dtype() == torch::kInt64, "k_idx must be int64");
    TORCH_CHECK(k_weights.dtype() == torch::kBFloat16, "k_weights must be bf16");
    TORCH_CHECK(retrieved.dtype() == torch::kFloat32, "retrieved must be f32 for f32out variant");
    TORCH_CHECK(inter.dtype() == torch::kFloat32, "inter must be f32 for f32out variant");
    TORCH_CHECK(snapshot.dtype() == torch::kBFloat16, "snapshot must be bf16");

    int SC = k_idx.size(0);
    int W_K = k_idx.size(1);
    int W_Q = q_idx.size(1);
    int dim = memory.size(1);

    switch (block_threads) {
        case 64:
            launch_kernel_f32out<64>(memory, k_idx, k_weights, q_idx, q_weights,
                              snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                              stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 128:
            launch_kernel_f32out<128>(memory, k_idx, k_weights, q_idx, q_weights,
                               snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                               stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 256:
            launch_kernel_f32out<256>(memory, k_idx, k_weights, q_idx, q_weights,
                               snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                               stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 512:
            launch_kernel_f32out<512>(memory, k_idx, k_weights, q_idx, q_weights,
                               snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                               stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        case 1024:
            launch_kernel_f32out<1024>(memory, k_idx, k_weights, q_idx, q_weights,
                                snapshot, retrieved, inter, SC, C_local, W_K, W_Q, dim,
                                stride_k_b, stride_kw_b, stride_q_b, stride_qw_b);
            break;
        default:
            TORCH_CHECK(false, "block_threads must be 64, 128, 256, 512, or 1024");
    }
}
"""

_CPP_SOURCE = r"""
void warp_cooperative_gather(
    torch::Tensor memory, torch::Tensor k_idx, torch::Tensor k_weights,
    torch::Tensor q_idx, torch::Tensor q_weights,
    torch::Tensor snapshot, torch::Tensor retrieved, torch::Tensor inter,
    int block_threads, int C_local,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b);

void warp_cooperative_gather_f32out(
    torch::Tensor memory, torch::Tensor k_idx, torch::Tensor k_weights,
    torch::Tensor q_idx, torch::Tensor q_weights,
    torch::Tensor snapshot, torch::Tensor retrieved, torch::Tensor inter,
    int block_threads, int C_local,
    int64_t stride_k_b, int64_t stride_kw_b,
    int64_t stride_q_b, int64_t stride_qw_b);
"""

_warp_gather_cuda = None


def _get_cuda_module():
    global _warp_gather_cuda
    if _warp_gather_cuda is None:
        _warp_gather_cuda = load_inline(
            name="warp_cooperative_gather",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["warp_cooperative_gather", "warp_cooperative_gather_f32out"],
            verbose=False,
        )
    return _warp_gather_cuda


def warp_cooperative_gather(
    memory: torch.Tensor,
    k_idx: torch.Tensor,       # [SC, W_K] or [P, C, W_K] — may be strided
    k_weights: torch.Tensor,
    q_idx: torch.Tensor,       # [SC, W_Q] or [P, C, W_Q]
    q_weights: torch.Tensor,
    snapshot: torch.Tensor,    # [SC, W_K, dim] contiguous output (always bf16)
    retrieved: torch.Tensor,   # [SC, dim] contiguous output (bf16 or f32)
    inter: torch.Tensor,       # [SC, dim] contiguous output (bf16 or f32)
    block_threads: int = 1024,
):
    """Warp-cooperative fused snapshot + two weighted gathers.
    Accepts strided [P, C, W] batch inputs — no .contiguous() needed.
    Supports f32 output for retrieved/inter (uses f32out kernel variant)."""
    mod = _get_cuda_module()

    # Determine which kernel variant to use based on output dtype
    use_f32out = (retrieved.dtype == torch.float32 and inter.dtype == torch.float32)

    if k_idx.ndim == 3:
        P, C_local, W_K = k_idx.shape
        SC = P * C_local
        # Strides in ELEMENTS (not bytes)
        stride_k_b = k_idx.stride(0)
        stride_kw_b = k_weights.stride(0)
        stride_q_b = q_idx.stride(0)
        stride_qw_b = q_weights.stride(0)
        # Reshape to 2D for CUDA (keeps same data pointer, just changes size metadata)
        k_idx_2d = k_idx.as_strided((SC, k_idx.shape[2]), (k_idx.stride(1), k_idx.stride(2)))
        kw_2d = k_weights.as_strided((SC, k_weights.shape[2]), (k_weights.stride(1), k_weights.stride(2)))
        q_idx_2d = q_idx.as_strided((SC, q_idx.shape[2]), (q_idx.stride(1), q_idx.stride(2)))
        qw_2d = q_weights.as_strided((SC, q_weights.shape[2]), (q_weights.stride(1), q_weights.stride(2)))
    else:
        SC = k_idx.shape[0]
        C_local = SC
        stride_k_b = stride_kw_b = stride_q_b = stride_qw_b = 0
        k_idx_2d = k_idx.to(torch.int64)
        kw_2d = k_weights.to(memory.dtype)
        q_idx_2d = q_idx.to(torch.int64)
        qw_2d = q_weights.to(memory.dtype)

    if use_f32out:
        mod.warp_cooperative_gather_f32out(
            memory,
            k_idx_2d, kw_2d,
            q_idx_2d, qw_2d,
            snapshot, retrieved, inter,
            block_threads, C_local,
            stride_k_b, stride_kw_b, stride_q_b, stride_qw_b,
        )
    else:
        mod.warp_cooperative_gather(
            memory,
            k_idx_2d, kw_2d,
            q_idx_2d, qw_2d,
            snapshot, retrieved, inter,
            block_threads, C_local,
            stride_k_b, stride_kw_b, stride_q_b, stride_qw_b,
        )
