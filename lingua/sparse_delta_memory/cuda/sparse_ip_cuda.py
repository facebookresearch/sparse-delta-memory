# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CUDA two-pointer sparse inner product kernel.

Compiled via torch.utils.cpp_extension.load_inline.
Assumes indices are sorted along the W dimension.
Supports float32 and bfloat16 val/lc inputs (accumulation in float32).
Supports int32 and int64 indices (int32 preferred — 15% faster due to halved index data).
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <c10/cuda/CUDAStream.h>

// Helper: load a scalar as float regardless of storage type
template<typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int idx) {
    return static_cast<float>(ptr[idx]);
}

template<>
__device__ __forceinline__ float load_as_float(const at::BFloat16* ptr, int idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
}

// Helper: store float to output (always float32 output)
__device__ __forceinline__ void store_float(float* ptr, int idx, float val) {
    ptr[idx] = val;
}

// ============================================================
// Forward kernel: out[ci, i, j] = merge_ip(row[ci,i], col[ci,j])
// One thread per (ci, i, j) triplet.
// Templated on both scalar_t (val/lc dtype) and idx_t (index dtype).
// ============================================================
template<typename scalar_t, typename idx_t>
__global__ void sparse_ip_gated_fwd_sorted_kernel(
    const idx_t*    __restrict__ idx_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ val_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ lc_row,    // [N, C, W_ROW]
    const idx_t*    __restrict__ idx_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ val_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ lc_col,    // [N, C, W_COL]
    float*          __restrict__ out,       // [N, C, C]
    int N, int C, int W_ROW, int W_COL, int causal_mode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * C;
    if (tid >= total) return;

    int ci = tid / (C * C);
    int rem = tid % (C * C);
    int i = rem / C;
    int j = rem % C;

    bool causal = (causal_mode == 1) ? (i > j) : (i >= j);
    if (!causal) {
        out[tid] = 0.0f;
        return;
    }

    int row_off = (ci * C + i) * W_ROW;
    int col_off = (ci * C + j) * W_COL;

    float acc = 0.0f;
    int pa = 0, pb = 0;
    while (pa < W_ROW && pb < W_COL) {
        idx_t ia = idx_row[row_off + pa], ib = idx_col[col_off + pb];
        if (ia == ib) {
            float vr = load_as_float(val_row, row_off + pa);
            float vc = load_as_float(val_col, col_off + pb);
            float lr = load_as_float(lc_row, row_off + pa);
            float lc = load_as_float(lc_col, col_off + pb);
            acc += vr * vc * expf(lr - lc);
            pa++; pb++;
        } else if (ia < ib) {
            pa++;
        } else {
            pb++;
        }
    }
    out[tid] = acc;
}

// ============================================================
// Forward kernel with shared memory for row data.
// One BLOCK per (ci, i) row.  Block size = C threads.
// Thread j handles the (ci, i, j) output element.
// Row data (idx, val, lc) is cooperatively loaded into shared memory,
// eliminating redundant global memory reads across threads in a row.
// ============================================================
template<typename scalar_t, typename idx_t>
__global__ void sparse_ip_gated_fwd_smem_kernel(
    const idx_t*    __restrict__ idx_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ val_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ lc_row,    // [N, C, W_ROW]
    const idx_t*    __restrict__ idx_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ val_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ lc_col,    // [N, C, W_COL]
    float*          __restrict__ out,       // [N, C, C]
    int N, int C, int W_ROW, int W_COL, int causal_mode)
{
    // One block = one (ci, i) pair
    int block_id = blockIdx.x;
    int ci = block_id / C;
    int i  = block_id % C;
    int j  = threadIdx.x;  // each thread handles one column j

    // Shared memory layout: [W_ROW idx_t] [W_ROW float] [W_ROW float]
    extern __shared__ char smem_raw[];
    idx_t* s_idx  = (idx_t*)smem_raw;
    float* s_val  = (float*)(s_idx + W_ROW);
    float* s_lc   = s_val + W_ROW;

    // Cooperative load of row (ci, i) data into shared memory
    int row_off = (ci * C + i) * W_ROW;
    for (int w = j; w < W_ROW; w += blockDim.x) {
        s_idx[w] = idx_row[row_off + w];
        s_val[w] = load_as_float(val_row, row_off + w);
        s_lc[w]  = load_as_float(lc_row, row_off + w);
    }
    __syncthreads();

    if (j >= C) return;

    // Output index
    int out_idx = ci * C * C + i * C + j;

    // Causal check: threads in upper triangle write zero
    bool causal = (causal_mode == 1) ? (i > j) : (i >= j);
    if (!causal) {
        out[out_idx] = 0.0f;
        return;
    }

    // Two-pointer merge: row from shared mem, col from global
    int col_off = (ci * C + j) * W_COL;

    float acc = 0.0f;
    int pa = 0, pb = 0;
    while (pa < W_ROW && pb < W_COL) {
        idx_t ia = s_idx[pa];
        idx_t ib = idx_col[col_off + pb];
        if (ia == ib) {
            float vr = s_val[pa];
            float vc = load_as_float(val_col, col_off + pb);
            float lr = s_lc[pa];
            float lc_v = load_as_float(lc_col, col_off + pb);
            acc += vr * vc * expf(lr - lc_v);
            pa++; pb++;
        } else if (ia < ib) {
            pa++;
        } else {
            pb++;
        }
    }
    out[out_idx] = acc;
}

// ============================================================
// Backward: atomic two-pointer merge. 1 thread per (ci, i, j).
// Requires zero-initialized grad arrays.
// Templated on both scalar_t (val/lc dtype) and idx_t (index dtype).
// ============================================================
template<typename scalar_t, typename idx_t>
__global__ void sparse_ip_gated_bwd_atomic_kernel(
    const float*    __restrict__ dout,      // [N, C, C]
    const idx_t*    __restrict__ idx_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ val_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ lc_row,    // [N, C, W_ROW]
    const idx_t*    __restrict__ idx_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ val_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ lc_col,    // [N, C, W_COL]
    float*          __restrict__ dval_row,  // [N, C, W_ROW]
    float*          __restrict__ dval_col,  // [N, C, W_COL]
    int N, int C, int W_ROW, int W_COL, int causal_mode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * C;
    if (tid >= total) return;

    int ci = tid / (C * C);
    int rem = tid % (C * C);
    int i = rem / C;
    int j = rem % C;

    bool causal = (causal_mode == 1) ? (i > j) : (i >= j);
    if (!causal) return;

    float gout = dout[tid];
    if (gout == 0.0f) return;

    int row_off = (ci * C + i) * W_ROW;
    int col_off = (ci * C + j) * W_COL;

    int pa = 0, pb = 0;
    while (pa < W_ROW && pb < W_COL) {
        idx_t ia = idx_row[row_off + pa], ib = idx_col[col_off + pb];
        if (ia == ib) {
            float vr = load_as_float(val_row, row_off + pa);
            float vc = load_as_float(val_col, col_off + pb);
            float lr = load_as_float(lc_row, row_off + pa);
            float lc = load_as_float(lc_col, col_off + pb);
            float E = expf(lr - lc);
            atomicAdd(&dval_row[row_off + pa], gout * vc * E);
            atomicAdd(&dval_col[col_off + pb], gout * vr * E);
            pa++; pb++;
        } else if (ia < ib) {
            pa++;
        } else {
            pb++;
        }
    }
}

// ============================================================
// Backward kernel with shared memory for row data.
// One BLOCK per (ci, i) row.  Block size = C threads.
// Thread j handles the (ci, i, j) gradient contribution.
// Row data is cooperatively loaded into shared memory.
// Row grad (dval_row) is accumulated in shared memory and written once.
// Col grad (dval_col) still uses atomicAdd to global memory.
// ============================================================
template<typename scalar_t, typename idx_t>
__global__ void sparse_ip_gated_bwd_smem_kernel(
    const float*    __restrict__ dout,      // [N, C, C]
    const idx_t*    __restrict__ idx_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ val_row,   // [N, C, W_ROW]
    const scalar_t* __restrict__ lc_row,    // [N, C, W_ROW]
    const idx_t*    __restrict__ idx_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ val_col,   // [N, C, W_COL]
    const scalar_t* __restrict__ lc_col,    // [N, C, W_COL]
    float*          __restrict__ dval_row,  // [N, C, W_ROW]
    float*          __restrict__ dval_col,  // [N, C, W_COL]
    int N, int C, int W_ROW, int W_COL, int causal_mode)
{
    // One block = one (ci, i) pair
    int block_id = blockIdx.x;
    int ci = block_id / C;
    int i  = block_id % C;
    int j  = threadIdx.x;

    // Shared memory layout:
    //   s_idx:      [W_ROW] idx_t
    //   s_val:      [W_ROW] float
    //   s_lc:       [W_ROW] float
    //   s_dval_row: [W_ROW] float  (for reducing row gradients)
    extern __shared__ char smem_raw[];
    idx_t* s_idx      = (idx_t*)smem_raw;
    float* s_val      = (float*)(s_idx + W_ROW);
    float* s_lc       = s_val + W_ROW;
    float* s_dval_row = s_lc + W_ROW;

    // Cooperative load of row (ci, i) data and zero-init dval accumulator
    int row_off = (ci * C + i) * W_ROW;
    for (int w = j; w < W_ROW; w += blockDim.x) {
        s_idx[w]      = idx_row[row_off + w];
        s_val[w]      = load_as_float(val_row, row_off + w);
        s_lc[w]       = load_as_float(lc_row, row_off + w);
        s_dval_row[w] = 0.0f;
    }
    __syncthreads();

    if (j < C) {
        // Causal check
        bool causal = (causal_mode == 1) ? (i > j) : (i >= j);

        if (causal) {
            int dout_idx = ci * C * C + i * C + j;
            float gout = dout[dout_idx];

            if (gout != 0.0f) {
                int col_off = (ci * C + j) * W_COL;

                int pa = 0, pb = 0;
                while (pa < W_ROW && pb < W_COL) {
                    idx_t ia = s_idx[pa];
                    idx_t ib = idx_col[col_off + pb];
                    if (ia == ib) {
                        float vr = s_val[pa];
                        float vc = load_as_float(val_col, col_off + pb);
                        float lr = s_lc[pa];
                        float lc_v = load_as_float(lc_col, col_off + pb);
                        float E = expf(lr - lc_v);
                        // Accumulate row grad via atomicAdd in shared memory
                        atomicAdd(&s_dval_row[pa], gout * vc * E);
                        // Col grad still goes to global memory
                        atomicAdd(&dval_col[col_off + pb], gout * vr * E);
                        pa++; pb++;
                    } else if (ia < ib) {
                        pa++;
                    } else {
                        pb++;
                    }
                }
            }
        }
    }

    __syncthreads();

    // Write accumulated row grads to global memory (one write per W element)
    for (int w = j; w < W_ROW; w += blockDim.x) {
        // Use atomicAdd for safety — multiple blocks could share row_off
        // (but in practice, each (ci, i) pair maps to a unique row, so direct write works)
        dval_row[row_off + w] = s_dval_row[w];
    }
}

// ============================================================
// C++ wrappers — dispatch float32/bfloat16 × int32/int64
// ============================================================
#define DISPATCH_FLOAT_OR_BF16(TENSOR, NAME, ...)                            \
    if (TENSOR.scalar_type() == at::ScalarType::Float) {                     \
        using scalar_t = float;                                              \
        __VA_ARGS__;                                                         \
    } else if (TENSOR.scalar_type() == at::ScalarType::BFloat16) {           \
        using scalar_t = at::BFloat16;                                       \
        __VA_ARGS__;                                                         \
    } else {                                                                 \
        TORCH_CHECK(false, NAME " only supports float32 and bfloat16");      \
    }

// Dispatch helper for index dtype (int32 or int64)
#define DISPATCH_IDX(IDX_TENSOR, NAME, ...)                                  \
    if (IDX_TENSOR.scalar_type() == at::ScalarType::Int) {                   \
        using idx_t = int32_t;                                               \
        __VA_ARGS__;                                                         \
    } else if (IDX_TENSOR.scalar_type() == at::ScalarType::Long) {           \
        using idx_t = int64_t;                                               \
        __VA_ARGS__;                                                         \
    } else {                                                                 \
        TORCH_CHECK(false, NAME " indices must be int32 or int64");          \
    }

torch::Tensor sparse_ip_gated_fwd_sorted(
    torch::Tensor idx_row, torch::Tensor val_row, torch::Tensor lc_row,
    torch::Tensor idx_col, torch::Tensor val_col, torch::Tensor lc_col,
    int causal_mode)
{
    int N = idx_row.size(0);
    int C = idx_row.size(1);
    int W_ROW = idx_row.size(2);
    int W_COL = idx_col.size(2);

    auto out = torch::empty({N, C, C}, torch::TensorOptions().dtype(torch::kFloat32).device(val_row.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_IDX(idx_row, "sparse_ip_fwd", {
        DISPATCH_FLOAT_OR_BF16(val_row, "sparse_ip_fwd", {
            if (C >= 64) {
                // Shared memory kernel: 1 block per (ci, i) row
                int grid = N * C;
                int block_sz = C;  // one thread per column j
                // smem: W_ROW * (sizeof(idx_t) + 2 * sizeof(float))
                size_t smem_bytes = W_ROW * (sizeof(idx_t) + 2 * sizeof(float));
                sparse_ip_gated_fwd_smem_kernel<scalar_t, idx_t><<<grid, block_sz, smem_bytes, stream>>>(
                    idx_row.data_ptr<idx_t>(),
                    val_row.data_ptr<scalar_t>(), lc_row.data_ptr<scalar_t>(),
                    idx_col.data_ptr<idx_t>(),
                    val_col.data_ptr<scalar_t>(), lc_col.data_ptr<scalar_t>(),
                    out.data_ptr<float>(), N, C, W_ROW, W_COL, causal_mode);
            } else {
                // Original flat kernel for small C
                int total = N * C * C;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                sparse_ip_gated_fwd_sorted_kernel<scalar_t, idx_t><<<blocks, threads, 0, stream>>>(
                    idx_row.data_ptr<idx_t>(),
                    val_row.data_ptr<scalar_t>(), lc_row.data_ptr<scalar_t>(),
                    idx_col.data_ptr<idx_t>(),
                    val_col.data_ptr<scalar_t>(), lc_col.data_ptr<scalar_t>(),
                    out.data_ptr<float>(), N, C, W_ROW, W_COL, causal_mode);
            }
        });
    });

    return out;
}

std::vector<torch::Tensor> sparse_ip_gated_bwd_sorted(
    torch::Tensor idx_row, torch::Tensor val_row, torch::Tensor lc_row,
    torch::Tensor idx_col, torch::Tensor val_col, torch::Tensor lc_col,
    torch::Tensor dout, int causal_mode)
{
    int N = idx_row.size(0);
    int C = idx_row.size(1);
    int W_ROW = idx_row.size(2);
    int W_COL = idx_col.size(2);

    // Zero-initialized for atomic accumulation
    auto dval_row = torch::zeros({N, C, W_ROW}, torch::TensorOptions().dtype(torch::kFloat32).device(val_row.device()));
    auto dval_col = torch::zeros({N, C, W_COL}, torch::TensorOptions().dtype(torch::kFloat32).device(val_col.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_IDX(idx_row, "sparse_ip_bwd", {
        DISPATCH_FLOAT_OR_BF16(val_row, "sparse_ip_bwd", {
            if (C >= 64) {
                // Shared memory kernel: 1 block per (ci, i) row
                int grid = N * C;
                int block_sz = C;
                // smem: W_ROW * (sizeof(idx_t) + 2*sizeof(float) + sizeof(float))
                // last term is s_dval_row accumulator
                size_t smem_bytes = W_ROW * (sizeof(idx_t) + 3 * sizeof(float));
                sparse_ip_gated_bwd_smem_kernel<scalar_t, idx_t><<<grid, block_sz, smem_bytes, stream>>>(
                    dout.data_ptr<float>(),
                    idx_row.data_ptr<idx_t>(), val_row.data_ptr<scalar_t>(), lc_row.data_ptr<scalar_t>(),
                    idx_col.data_ptr<idx_t>(), val_col.data_ptr<scalar_t>(), lc_col.data_ptr<scalar_t>(),
                    dval_row.data_ptr<float>(), dval_col.data_ptr<float>(),
                    N, C, W_ROW, W_COL, causal_mode);
            } else {
                // Original flat kernel for small C
                int total = N * C * C;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                sparse_ip_gated_bwd_atomic_kernel<scalar_t, idx_t><<<blocks, threads, 0, stream>>>(
                    dout.data_ptr<float>(),
                    idx_row.data_ptr<idx_t>(), val_row.data_ptr<scalar_t>(), lc_row.data_ptr<scalar_t>(),
                    idx_col.data_ptr<idx_t>(), val_col.data_ptr<scalar_t>(), lc_col.data_ptr<scalar_t>(),
                    dval_row.data_ptr<float>(), dval_col.data_ptr<float>(),
                    N, C, W_ROW, W_COL, causal_mode);
            }
        });
    });

    return {dval_row, dval_col};
}

// ============================================================
// Fused gather + decay scatter for Phase A backward.
// Replaces 2 separate kernels: gather chunk_dw, scatter decayed back.
// Does NOT include the grad_g_decay reduction (kept as PyTorch op).
//
// One thread per (cw, d) element:
//   chunk_dw_out[cw, d] = grad_memory[flat_k[cw], d]           (gather)
//   grad_memory[flat_k[cw], d] *= total_cumul[flat_k[cw]]      (decay in-place)
// ============================================================
template<typename scalar_t>
__global__ void fused_gather_decay_scatter_kernel(
    scalar_t*           __restrict__ grad_memory,   // [num_slots, dim] — read & written
    const int64_t*      __restrict__ flat_k,        // [C*W]
    const float*        __restrict__ total_cumul,    // [num_slots, 1]
    scalar_t*           __restrict__ chunk_dw_out,  // [C*W, dim]
    int CW, int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= CW * dim) return;

    int cw = tid / dim;
    int d = tid % dim;

    int64_t slot = flat_k[cw];
    int gm_off = slot * dim + d;

    float gm_val = load_as_float(grad_memory, gm_off);
    float cumul = total_cumul[slot];

    // Store original to chunk_dw output
    chunk_dw_out[cw * dim + d] = static_cast<scalar_t>(gm_val);

    // Write decayed value back
    grad_memory[gm_off] = static_cast<scalar_t>(gm_val * cumul);
}

// C++ wrapper
torch::Tensor fused_gather_decay_scatter(
    torch::Tensor grad_memory,    // [num_slots, dim] — modified in-place
    torch::Tensor flat_k,         // [C*W] int64
    torch::Tensor total_cumul,    // [num_slots, 1] f32
    int C, int W, int dim)
{
    int CW = C * W;
    auto chunk_dw = torch::empty({C, W, dim}, grad_memory.options());

    int total = CW * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_FLOAT_OR_BF16(grad_memory, "fused_gather_decay_scatter", {
        fused_gather_decay_scatter_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_memory.data_ptr<scalar_t>(),
            flat_k.data_ptr<int64_t>(),
            total_cumul.data_ptr<float>(),
            chunk_dw.data_ptr<scalar_t>(),
            CW, dim);
    });

    return chunk_dw;
}
"""

_CPP_SOURCE = r"""
torch::Tensor sparse_ip_gated_fwd_sorted(
    torch::Tensor idx_row, torch::Tensor val_row, torch::Tensor lc_row,
    torch::Tensor idx_col, torch::Tensor val_col, torch::Tensor lc_col,
    int causal_mode);

std::vector<torch::Tensor> sparse_ip_gated_bwd_sorted(
    torch::Tensor idx_row, torch::Tensor val_row, torch::Tensor lc_row,
    torch::Tensor idx_col, torch::Tensor val_col, torch::Tensor lc_col,
    torch::Tensor dout, int causal_mode);

torch::Tensor fused_gather_decay_scatter(
    torch::Tensor grad_memory, torch::Tensor flat_k,
    torch::Tensor total_cumul, int C, int W, int dim);
"""

# Compile once, cached by PyTorch
_sparse_ip_cuda = None

def _get_cuda_module():
    global _sparse_ip_cuda
    if _sparse_ip_cuda is None:
        _sparse_ip_cuda = load_inline(
            name="sparse_ip_sorted",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["sparse_ip_gated_fwd_sorted", "sparse_ip_gated_bwd_sorted",
                        "fused_gather_decay_scatter"],
            verbose=False,
        )
    return _sparse_ip_cuda


def sparse_ip_gated_fwd_sorted(idx_row, val_row, lc_row, idx_col, val_col, lc_col, causal_mode):
    """Forward: sorted two-pointer sparse inner product. Supports f32 and bf16."""
    mod = _get_cuda_module()
    return mod.sparse_ip_gated_fwd_sorted(
        idx_row.contiguous(), val_row.contiguous(), lc_row.contiguous(),
        idx_col.contiguous(), val_col.contiguous(), lc_col.contiguous(),
        causal_mode)


def sparse_ip_gated_bwd_sorted(idx_row, val_row, lc_row, idx_col, val_col, lc_col, dout, causal_mode):
    """Backward: atomic two-pointer sparse inner product gradient. Supports f32 and bf16."""
    mod = _get_cuda_module()
    return mod.sparse_ip_gated_bwd_sorted(
        idx_row.contiguous(), val_row.contiguous(), lc_row.contiguous(),
        idx_col.contiguous(), val_col.contiguous(), lc_col.contiguous(),
        dout.contiguous(), causal_mode)


def fused_gather_decay_scatter(grad_memory, flat_k, total_cumul, C, W, dim):
    """Fused gather + decay scatter for Phase A backward.

    Replaces gather chunk_dw + scatter decayed back with 1 kernel.
    grad_memory is modified in-place. Returns chunk_dw [C, W, dim].
    Supports bf16 and f32.
    """
    mod = _get_cuda_module()
    return mod.fused_gather_decay_scatter(
        grad_memory, flat_k.contiguous(), total_cumul.contiguous(), C, W, dim)
