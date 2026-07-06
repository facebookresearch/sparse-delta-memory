# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CUDA fused scatter-write kernel using native bf16 atomicAdd + vectorized loads.

Replaces the Triton `_fused_scatter_write_kernel` in memory_controller_ops.py.
Uses native `atomicAdd(__nv_bfloat16*, __nv_bfloat16)` (hardware-accelerated on
SM 8.0+ / CUDA 11.8+) combined with int4 vectorized loads for delta_v.

Operation: memory[k_idx[t,n], :] += k_val[t,n] * delta_v[t,:] * post_decay[t,n]

Grid: (C*W,) — one block per (t, n) entry.
Block: BLOCK_DIM threads cooperate over the dim dimension in a strided pattern.

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
// Fused scatter-write kernel — native bf16 atomicAdd + int4 loads
//
// Grid:  (C*W,) — one block per (t, n) entry
// Block: BLOCK_DIM threads cooperate over dim in strided pattern
//
// Each thread loads 8 bf16 values at a time via int4 (128-bit),
// converts to float, multiplies by scale, converts back to bf16,
// and uses native atomicAdd(__nv_bfloat16*) for each element.
// ============================================================

template<int BLOCK_DIM>
__global__ void fused_scatter_write_native_kernel(
    __nv_bfloat16*       __restrict__ memory,      // [num_slots, dim]
    const int64_t*       __restrict__ k_idx,       // [C * W]
    const __nv_bfloat16* __restrict__ k_val,       // [C * W]
    const __nv_bfloat16* __restrict__ delta_v,     // [C, dim]
    const __nv_bfloat16* __restrict__ post_decay,  // [C * W]
    int CW, int W, int dim)
{
    int entry = blockIdx.x;
    if (entry >= CW) return;

    int t = entry / W;

    // Load scalar values — shared by all threads in block
    int64_t slot = k_idx[entry];
    float kv = __bfloat162float(k_val[entry]);
    float pd = __bfloat162float(post_decay[entry]);
    float scale = kv * pd;

    if (scale == 0.0f) return;

    // Base pointers
    const __nv_bfloat16* dv_row = delta_v + (int64_t)t * dim;
    __nv_bfloat16* mem_row = memory + slot * (int64_t)dim;

    // Each thread processes elements at stride BLOCK_DIM * 8 (8 elements per int4 load)
    // Total elements per thread per pass: 8
    int total_groups = dim / 8;  // number of int4-sized groups (dim is multiple of 8 for all configs)

    for (int g = threadIdx.x; g < total_groups; g += BLOCK_DIM) {
        int d = g * 8;

        // Vectorized load: 8 bf16 = 16 bytes = int4
        int4 packed_dv = *reinterpret_cast<const int4*>(&dv_row[d]);
        __nv_bfloat16* dv_vals = reinterpret_cast<__nv_bfloat16*>(&packed_dv);

        // Convert, scale, and atomicAdd each element using native bf16 atomic
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            __nv_bfloat16 contrib = __float2bfloat16(scale * __bfloat162float(dv_vals[i]));
            atomicAdd(&mem_row[d + i], contrib);
        }
    }

    // Handle remainder if dim is not divisible by 8
    int remainder_start = total_groups * 8;
    for (int d = remainder_start + threadIdx.x; d < dim; d += BLOCK_DIM) {
        __nv_bfloat16 dv_val = dv_row[d];
        __nv_bfloat16 contrib = __float2bfloat16(scale * __bfloat162float(dv_val));
        atomicAdd(&mem_row[d], contrib);
    }
}

// ============================================================
// C++ wrapper
// ============================================================

template<int BLOCK_DIM>
void launch_kernel(
    torch::Tensor memory, torch::Tensor k_idx, torch::Tensor k_val,
    torch::Tensor delta_v, torch::Tensor post_decay,
    int CW, int W, int dim)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_scatter_write_native_kernel<BLOCK_DIM><<<CW, BLOCK_DIM, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(memory.data_ptr<at::BFloat16>()),
        k_idx.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(k_val.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(delta_v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(post_decay.data_ptr<at::BFloat16>()),
        CW, W, dim);
}

void fused_scatter_write_cuda(
    torch::Tensor memory,
    torch::Tensor k_idx,
    torch::Tensor k_val,
    torch::Tensor delta_v,
    torch::Tensor post_decay,
    int W,
    int block_dim)
{
    TORCH_CHECK(memory.dtype() == torch::kBFloat16, "memory must be bf16");
    TORCH_CHECK(k_idx.dtype() == torch::kInt64, "k_idx must be int64");
    TORCH_CHECK(k_val.dtype() == torch::kBFloat16, "k_val must be bf16");
    TORCH_CHECK(delta_v.dtype() == torch::kBFloat16, "delta_v must be bf16");
    TORCH_CHECK(post_decay.dtype() == torch::kBFloat16, "post_decay must be bf16");

    int CW = k_idx.size(0);
    int dim = memory.size(1);

    switch (block_dim) {
        case 32:  launch_kernel<32>(memory, k_idx, k_val, delta_v, post_decay, CW, W, dim); break;
        case 64:  launch_kernel<64>(memory, k_idx, k_val, delta_v, post_decay, CW, W, dim); break;
        case 128: launch_kernel<128>(memory, k_idx, k_val, delta_v, post_decay, CW, W, dim); break;
        case 256: launch_kernel<256>(memory, k_idx, k_val, delta_v, post_decay, CW, W, dim); break;
        case 512: launch_kernel<512>(memory, k_idx, k_val, delta_v, post_decay, CW, W, dim); break;
        default: TORCH_CHECK(false, "block_dim must be 32, 64, 128, 256, or 512");
    }
}
"""

_CPP_SOURCE = r"""
void fused_scatter_write_cuda(
    torch::Tensor memory, torch::Tensor k_idx, torch::Tensor k_val,
    torch::Tensor delta_v, torch::Tensor post_decay, int W, int block_dim);
"""

_scatter_write_cuda = None


def _get_cuda_module():
    global _scatter_write_cuda
    if _scatter_write_cuda is None:
        _scatter_write_cuda = load_inline(
            name="scatter_write_native_bf16",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["fused_scatter_write_cuda"],
            verbose=False,
        )
    return _scatter_write_cuda


def fused_scatter_write(
    memory: torch.Tensor,      # [num_slots, dim] bf16
    k_idx: torch.Tensor,       # [C, W]
    k_val: torch.Tensor,       # [C, W]
    delta_v: torch.Tensor,     # [C, dim]
    post_decay: torch.Tensor,  # [C, W]
    block_dim: int = 128,
):
    """Fused scatter write using native bf16 atomicAdd + int4 vectorized loads.

    Drop-in replacement for the Triton `fused_scatter_write` in memory_controller_ops.py.
    """
    C, W = k_idx.shape
    mod = _get_cuda_module()
    mod.fused_scatter_write_cuda(
        memory.contiguous(),
        k_idx.contiguous().reshape(-1).to(torch.int64),
        k_val.contiguous().to(memory.dtype).reshape(-1),
        delta_v.contiguous().to(memory.dtype),
        post_decay.contiguous().to(memory.dtype).reshape(-1),
        W,
        block_dim,
    )
