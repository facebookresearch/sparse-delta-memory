# Copyright (c) Meta Platforms, Inc. and affiliates.
"""gather_cuda: the token gather kernel: per-token weighted sums of random bank rows, plus a copy of the rows.

For every token ``t`` of a launch and each operand ``o`` (one or two slot lists)

    out_o[t]    = sum_n weights_o[t, n] * bank[idx_o[t, n]]     (float32 accumulation)
    stash[t, n] = bank[idx_0[t, n]]                            (raw rows of operand 0)

The forward uses two operands: write slots weighted by ``k_eff`` give ``retrieved`` and the
snapshot, read slots weighted by ``q_w`` give ``inter``. The backward uses one: bank-gradient
rows at the write slots weighted by ``write_w`` give ``d_delta_v``, and the stash feeds the
per-entry kernels that follow.

The loads are random rows, so the kernel is latency bound and its geometry keeps rows in
flight. A row is read as 16-byte vectors by a row group of ``lanes`` threads (8, 16 or 32);
rows longer than one row group are split into column slices along the grid's y axis. The
entries of a token are spread over ``groups`` row groups: group ``g`` takes entries
``g, g + groups, ...`` in order and loads 4 (bf16) or 8 (fp32) rows across operands before
using any. Group partials are summed in a fixed order, so results are deterministic and
depend on ``groups`` only.

Tokens come in chunks of ``chunk``; positions ``>= valid`` within a chunk are padding: they
load nothing and produce zero outputs and zero stash rows. Launches use the current stream
and allocate nothing, so they are CUDA-graph safe.

The source also holds the no-stash variant that gather_ns_cuda builds (preprocessor variant GATHER_NS, plus GATHER_L2
for its L2 cache hints); see that module."""

import functools
from typing import NamedTuple

import torch

from ._build import load_cuda_extension

_MAX_THREADS = 512  # kMaxThreads, the kernel's launch bound

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#if defined(GATHER_NS)
#include <cstring>
#endif
#include <algorithm>
#include <limits>

namespace {

constexpr int kMaxThreads = 512;

#if defined(GATHER_NS)
#define FX_LDB(p) __ldcg(p)   // bank rows a grid that may still be finishing (PDL) wrote: coherent path
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define FX_PDL_WAIT_LAUNCH() asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory")
#else
#define FX_PDL_WAIT_LAUNCH() ((void)0)
#endif
#endif
#if defined(GATHER_L2)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ uint64_t fx_pol_ef() {   // L2 evict-first policy (pure: CSE'd by the compiler)
    uint64_t p;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
template <typename V> __device__ __forceinline__ V fx_ld_ef(const V* p) { return FX_LDB(p); }
template <> __device__ __forceinline__ uint4 fx_ld_ef<uint4>(const uint4* p) {   // L2-only, evict-first
    uint4 v;
    asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(fx_pol_ef()) : "memory");
    return v;
}
template <typename V> __device__ __forceinline__ void fx_st_ef(V* p, const V& v) { *p = v; }
template <> __device__ __forceinline__ void fx_st_ef<uint4>(uint4* p, const uint4& v) {
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(fx_pol_ef()) : "memory");
}
#else
template <typename V> __device__ __forceinline__ V fx_ld_ef(const V* p) { return FX_LDB(p); }
template <typename V> __device__ __forceinline__ void fx_st_ef(V* p, const V& v) { *p = v; }
#endif
#endif
template <int B> struct Bits;
template <> struct Bits<2> { using type = uint16_t; };
template <> struct Bits<4> { using type = uint32_t; };
template <> struct Bits<16> { using type = uint4; };

// Elements are raw bits: uint16_t is bfloat16 (the high half of a float), uint32_t float.
__device__ __forceinline__ float to_float(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ float to_float(uint32_t b) { return __uint_as_float(b); }

struct Operand {
    const void* idx;    // [rows, width] int32 or int64 bank rows
    const void* w;      // [rows, width] bfloat16 or float32 weights
    float* out;         // [rows, hd]
    int width;
};

struct Args {
    const void* bank;   // [slots, hd]
    void* stash;        // [rows, width of operand 0, hd]
    Operand op[2];
    int64_t row0, stash_row0, out_row0;   // first row of this launch in idx/w, stash, out
    int hd, vpr;                          // row length, vectors per row
    int tokens, chunk, valid;
    int per_group;                        // entries per row group, ceil(max width / groups)
    bool idx64, w_bf16;
#if defined(GATHER_NS)
    const int* kpos;     // [rows, width of operand 0]: FIRST position in the chunk's sorted row, or -1
    int crow;            // rows per chunk of the compact snapshot (C * W)
#endif
};

// Every input is read-only for the kernel's lifetime, so loads take the non-coherent path.
__device__ __forceinline__ int load_slot(const void* p, int64_t i, bool idx64) {
    return idx64 ? static_cast<int>(__ldg(static_cast<const long long*>(p) + i))
                 : __ldg(static_cast<const int*>(p) + i);
}

__device__ __forceinline__ float load_weight(const void* p, int64_t i, bool bf16) {
    return bf16 ? to_float(__ldg(static_cast<const unsigned short*>(p) + i))
                : __ldg(static_cast<const float*>(p) + i);
}

template <int N>
__device__ __forceinline__ void store_floats(float* dst, const float (&v)[N]) {
    if constexpr (N % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < N; i += 4)
            *reinterpret_cast<float4*>(dst + i) = make_float4(v[i], v[i + 1], v[i + 2], v[i + 3]);
    } else {
        #pragma unroll
        for (int i = 0; i < N; ++i) dst[i] = v[i];
    }
}

// Grid (token blocks, column slices); block (LPR lanes, groups, tokens per CTA). A column
// slice is LPR vectors wide. Threads linearise x first, so a row group is LPR consecutive
// lanes. Row group g accumulates entries g, g + groups, ... in order and the group partials
// are summed in order, so the rounding depends on groups only.
template <typename E, int VB, int LPR, bool HAS_Q>
__global__ void __launch_bounds__(kMaxThreads) token_gather_kernel(const Args a) {
    using V = typename Bits<VB>::type;
    constexpr int VEC = VB / sizeof(E);
    constexpr int NOP = HAS_Q ? 2 : 1;
    // Rows in flight per row group, across operands: 4 for bf16 and 8 for fp32, whose
    // accumulators take half the registers per 16-byte vector.
    constexpr int STEP = (sizeof(E) == 4 ? 8 : 4) / NOP;
    constexpr int SLICE = LPR * VEC;    // elements per column slice
    extern __shared__ float4 smem[];
    float* partial = reinterpret_cast<float*>(smem);   // [NOP][tpc][groups][SLICE] if groups > 1

    const int lane = threadIdx.x, group = threadIdx.y, groups = blockDim.y;
    const int t = blockIdx.x * blockDim.z + threadIdx.z;
    const bool in_range = t < a.tokens;
    const bool live = in_range && (a.valid == a.chunk || t % a.chunk < a.valid);
    const int v = blockIdx.y * LPR + lane;
    const bool lane_on = v < a.vpr;
    const int64_t col = int64_t(v) * VEC;
    const int64_t row = a.row0 + t;
    const E* bank = static_cast<const E*>(a.bank);
#if defined(GATHER_NS)
    E* stash = a.stash == nullptr ? nullptr : (a.kpos != nullptr
        ? static_cast<E*>(a.stash) + ((a.row0 + t) / a.chunk) * a.crow * a.hd + col
        : static_cast<E*>(a.stash) + (a.stash_row0 + t) * a.op[0].width * a.hd + col);
#else
    E* stash = static_cast<E*>(a.stash) + (a.stash_row0 + t) * a.op[0].width * a.hd + col;
#endif

    float acc[NOP][VEC];
    #pragma unroll
    for (int o = 0; o < NOP; ++o)
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc[o][i] = 0.0f;

    for (int jb = 0; jb < a.per_group; jb += LPR) {
        // Lane l reads the slot and weight of entry jb + l; loads broadcast them.
#if defined(GATHER_NS)
        int kp_l = -1;
        if (a.kpos != nullptr) {
            const int n = group + groups * (jb + lane);
            const bool ok = live && jb + lane < a.per_group && n < a.op[0].width;
            kp_l = ok ? __ldg(a.kpos + row * a.op[0].width + n) : -1;
        }
#endif
        int slot_l[NOP];
        float w_l[NOP];
        #pragma unroll
        for (int o = 0; o < NOP; ++o) {
            const int n = group + groups * (jb + lane);
            const bool ok = live && jb + lane < a.per_group && n < a.op[o].width;
            const int64_t i = row * a.op[o].width + n;
            slot_l[o] = ok ? load_slot(a.op[o].idx, i, a.idx64) : 0;
            w_l[o] = ok ? load_weight(a.op[o].w, i, a.w_bf16) : 0.0f;
        }
#if defined(GATHER_NS)
        if (jb == 0) FX_PDL_WAIT_LAUNCH();   // PDL: the previous grid (DW) has completed; before any bank read
#endif
        const int batch = min(LPR, a.per_group - jb);
        for (int s0 = 0; s0 < batch; s0 += STEP) {
            // Addresses first, then all loads: issuing a stash store between two loads would
            // stall on the first load and serialise the rest.
            const V* src[NOP][STEP];
            bool load[NOP][STEP];
            float w[NOP][STEP];
            #pragma unroll
            for (int s = 0; s < STEP; ++s) {
                const int n = group + groups * (jb + s0 + s);
                #pragma unroll
                for (int o = 0; o < NOP; ++o) {
                    const int slot = __shfl_sync(0xffffffffu, slot_l[o], s0 + s, LPR);
                    w[o][s] = __shfl_sync(0xffffffffu, w_l[o], s0 + s, LPR);
                    src[o][s] = reinterpret_cast<const V*>(bank + int64_t(slot) * a.hd + col);
                    load[o][s] = live && lane_on && s0 + s < batch && n < a.op[o].width;
                }
            }
            V r[NOP][STEP];
            #pragma unroll
            for (int s = 0; s < STEP; ++s)
                #pragma unroll
#if defined(GATHER_NS)
#if defined(GATHER_L2)
                for (int o = 0; o < NOP; ++o) r[o][s] = load[o][s] ? (o == 1 ? fx_ld_ef(src[o][s]) : FX_LDB(src[o][s])) : V{};   // read rows evict-first
#else
                for (int o = 0; o < NOP; ++o) r[o][s] = load[o][s] ? FX_LDB(src[o][s]) : V{};
#endif
#else
                for (int o = 0; o < NOP; ++o) r[o][s] = load[o][s] ? __ldg(src[o][s]) : V{};
#endif
            #pragma unroll
            for (int s = 0; s < STEP; ++s) {
                const int n = group + groups * (jb + s0 + s);
#if defined(GATHER_NS)
                if (a.kpos != nullptr) {
                    const int kj = __shfl_sync(0xffffffffu, kp_l, s0 + s, LPR);
                    if (kj >= 0 && in_range && lane_on && s0 + s < batch && n < a.op[0].width)
#if defined(GATHER_L2)
                        fx_st_ef(reinterpret_cast<V*>(stash + int64_t(kj) * a.hd), r[0][s]);   // evict-first
#else
                        *reinterpret_cast<V*>(stash + int64_t(kj) * a.hd) = r[0][s];
#endif
                } else if (a.stash != nullptr && in_range && lane_on && s0 + s < batch && n < a.op[0].width)
#else
                if (in_range && lane_on && s0 + s < batch && n < a.op[0].width)
#endif
                    *reinterpret_cast<V*>(stash + int64_t(n) * a.hd) = r[0][s];
                // Missing entries have weight 0 and a zero row, so their update adds an exact +0.
                #pragma unroll
                for (int o = 0; o < NOP; ++o) {
                    const E* e = reinterpret_cast<const E*>(&r[o][s]);
                    #pragma unroll
                    for (int i = 0; i < VEC; ++i)
                        acc[o][i] = fmaf(w[o][s], to_float(e[i]), acc[o][i]);
                }
            }
        }
    }

#if defined(GATHER_NS)
    FX_PDL_WAIT_LAUNCH();   // PDL: also when no entry loop ran (no store, no exit before the wait)
#endif
    if (groups == 1) {
        if (in_range && lane_on) {
            #pragma unroll
            for (int o = 0; o < NOP; ++o)
                store_floats(a.op[o].out + (a.out_row0 + t) * a.hd + col, acc[o]);
        }
        return;
    }
    #pragma unroll
    for (int o = 0; o < NOP; ++o) {
        float* dst = partial + ((o * blockDim.z + threadIdx.z) * groups + group) * SLICE;
        store_floats(dst + lane * VEC, acc[o]);
    }
    __syncthreads();
    // Sum the group partials of each column in order.
    const int tid = lane + LPR * (group + groups * threadIdx.z);
    const int width = min(SLICE, a.hd - int(blockIdx.y) * SLICE);
    for (int q = tid; q < blockDim.z * width; q += LPR * groups * blockDim.z) {
        const int z = q / width, c = q - z * width;
        const int tok = blockIdx.x * blockDim.z + z;
        if (tok >= a.tokens) continue;
        #pragma unroll
        for (int o = 0; o < NOP; ++o) {
            const float* src = partial + (o * blockDim.z + z) * groups * SLICE + c;
            float sum = 0.0f;
            for (int g = 0; g < groups; ++g) sum += src[g * SLICE];
            a.op[o].out[(a.out_row0 + tok) * a.hd + blockIdx.y * SLICE + c] = sum;
        }
    }
}

#if defined(GATHER_NS)
static thread_local int fx_pdl_next = 0;   // set by token_gather_dc for its own launch
struct FxGatherCap {          // what launch() would launch (plan capture)
    const void* fn;
    Args a;
    dim3 grid, block;
    size_t smem;
    int set;
};
static thread_local FxGatherCap* fx_cap = nullptr;   // non-null only inside token_gather_dc_plan
#endif
template <typename E, int VB, int LPR, bool HAS_Q>
void launch(const Args& a, dim3 grid, dim3 block, size_t smem) {
#if defined(GATHER_NS)
    if (fx_cap != nullptr) {   // plan capture: record, do not launch
        fx_cap->fn = reinterpret_cast<const void*>(token_gather_kernel<E, VB, LPR, HAS_Q>);
        fx_cap->a = a;
        fx_cap->grid = grid;
        fx_cap->block = block;
        fx_cap->smem = smem;
        fx_cap->set = 1;
        return;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    if (fx_pdl_next) {   // programmatic stream serialization (the caller guarantees the previous launch wrote nothing read before the wait)
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream.stream();
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, token_gather_kernel<E, VB, LPR, HAS_Q>, a));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
#else
    auto stream = at::cuda::getCurrentCUDAStream();
#endif
    token_gather_kernel<E, VB, LPR, HAS_Q><<<grid, block, smem, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename E, bool HAS_Q>
void dispatch(const Args& a, int vec_bytes, dim3 grid, dim3 block, size_t smem) {
    if (vec_bytes != 16) return launch<E, sizeof(E), 32, HAS_Q>(a, grid, block, smem);
    switch (block.x) {
        case 8: return launch<E, 16, 8, HAS_Q>(a, grid, block, smem);
        case 16: return launch<E, 16, 16, HAS_Q>(a, grid, block, smem);
        default: return launch<E, 16, 32, HAS_Q>(a, grid, block, smem);
    }
}

}  // namespace

#if defined(GATHER_NS)
void token_gather_dc(
#else
void token_gather(
#endif
    at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor stash, at::Tensor out,
    std::optional<at::Tensor> q_idx, std::optional<at::Tensor> q_w, std::optional<at::Tensor> q_out,
    int64_t row0, int64_t stash_row0, int64_t out_row0,
    int64_t tokens, int64_t chunk, int64_t valid,
#if defined(GATHER_NS)
    int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc,
    std::optional<at::Tensor> kpos, int64_t crow, int64_t pdl)
#else
    int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc)
#endif
{
    const bool has_q = q_idx.has_value();
#if defined(GATHER_NS)
    TORCH_CHECK(pdl == 0 || pdl == 1, "token_gather_dc: pdl must be 0 or 1");
#endif
    TORCH_CHECK(has_q == q_w.has_value() && has_q == q_out.has_value(),
                "token_gather: q_idx, q_weights and q_out go together");
    TORCH_CHECK(bank.is_cuda() && bank.dim() == 2 && bank.is_contiguous(),
                "token_gather: bank must be a contiguous CUDA tensor [slots, head_dim]");
    TORCH_CHECK(bank.scalar_type() == at::kBFloat16 || bank.scalar_type() == at::kFloat,
                "token_gather: bank must be bfloat16 or float32, got ", bank.scalar_type());
    TORCH_CHECK(bank.size(0) <= std::numeric_limits<int>::max(),
                "token_gather: the bank must have fewer than 2^31 rows");
    TORCH_CHECK(stash.scalar_type() == bank.scalar_type(),
                "token_gather: stash must match the bank dtype");
    TORCH_CHECK(tokens >= 0 && tokens <= std::numeric_limits<int>::max() && chunk >= 1
                && tokens % chunk == 0 && valid >= 0 && valid <= chunk,
                "token_gather: need tokens = k * chunk and 0 <= valid <= chunk");
    const int64_t hd = bank.size(1), es = bank.element_size();

    Args a{};
    a.bank = bank.data_ptr();
    a.idx64 = idx.scalar_type() == at::kLong;
    a.w_bf16 = w.scalar_type() == at::kBFloat16;
    const auto set_operand = [&](int o, const at::Tensor& oi, const at::Tensor& ow,
                                 const at::Tensor& oo) {
        for (const at::Tensor* x : {&oi, &ow, &oo})
            TORCH_CHECK(x->device() == bank.device() && x->is_contiguous(),
                        "token_gather: every tensor must be contiguous and on the bank's device");
        TORCH_CHECK(oi.scalar_type() == (a.idx64 ? at::kLong : at::kInt),
                    "token_gather: indices must be int32 or int64, the same for both operands");
        TORCH_CHECK(ow.scalar_type() == (a.w_bf16 ? at::kBFloat16 : at::kFloat),
                    "token_gather: weights must be bfloat16 or float32, the same for both "
                    "operands");
        TORCH_CHECK(oo.scalar_type() == at::kFloat, "token_gather: outputs must be float32");
        TORCH_CHECK(ow.dim() >= 1 && oi.sizes() == ow.sizes(),
                    "token_gather: indices and weights must match");
        const int64_t width = ow.size(-1);
        TORCH_CHECK(width >= 1 && width <= 256,
                    "token_gather: widths must be in 1..256, got ", width);
        TORCH_CHECK(row0 >= 0 && row0 + tokens <= ow.numel() / width,
                    "token_gather: rows [row0, row0 + tokens) exceed the index rows");
        TORCH_CHECK(oo.numel() % hd == 0 && out_row0 >= 0 && out_row0 + tokens <= oo.numel() / hd,
                    "token_gather: rows [out_row0, out_row0 + tokens) exceed the output rows");
        a.op[o] = {oi.data_ptr(), ow.data_ptr(), oo.data_ptr<float>(), static_cast<int>(width)};
    };
    set_operand(0, idx, w, out);
    if (has_q) set_operand(1, *q_idx, *q_w, *q_out);
    const int64_t stash_row = a.op[0].width * hd;
#if defined(GATHER_NS)
    TORCH_CHECK(stash.numel() == 0 || (stash.device() == bank.device() && stash.is_contiguous()
                && stash.numel() % stash_row == 0 && stash_row0 >= 0
                && stash_row0 + tokens <= stash.numel() / stash_row),
#else
    TORCH_CHECK(stash.device() == bank.device() && stash.is_contiguous()
                && stash.numel() % stash_row == 0 && stash_row0 >= 0
                && stash_row0 + tokens <= stash.numel() / stash_row,
#endif
                "token_gather: stash must be contiguous [rows, W, head_dim] and cover "
                "[stash_row0, stash_row0 + tokens)");

    TORCH_CHECK(vec_bytes == 16 || vec_bytes == es,
                "token_gather: vec_bytes must be 16 or the element size");
    TORCH_CHECK((hd * es) % vec_bytes == 0, "token_gather: rows are not a whole number of vectors");
    TORCH_CHECK(lanes == 8 || lanes == 16 || lanes == 32,
                "token_gather: lanes must be 8, 16 or 32");
    TORCH_CHECK(vec_bytes == 16 || lanes == 32, "token_gather: element vectors need 32 lanes");
    // Rows are read through the non-coherent path, so nothing the kernel writes may alias the bank.
    const auto address = [](const at::Tensor& x) {
        return reinterpret_cast<uintptr_t>(x.data_ptr());
    };
    const auto overlaps_bank = [&](const at::Tensor& x) {
        return address(x) < address(bank) + bank.nbytes()
            && address(bank) < address(x) + x.nbytes();
    };
#if defined(GATHER_NS)
    TORCH_CHECK((stash.numel() == 0 || !overlaps_bank(stash)) && !overlaps_bank(out) && !(has_q && overlaps_bank(*q_out)),
#else
    TORCH_CHECK(!overlaps_bank(stash) && !overlaps_bank(out) && !(has_q && overlaps_bank(*q_out)),
#endif
                "token_gather: stash and outputs must not overlap the bank");
    const auto aligned = [&](const at::Tensor& x) { return address(x) % 16 == 0; };
#if defined(GATHER_NS)
    TORCH_CHECK(vec_bytes != 16 || (aligned(bank) && (stash.numel() == 0 || aligned(stash)) && aligned(out)
#else
    TORCH_CHECK(vec_bytes != 16 || (aligned(bank) && aligned(stash) && aligned(out)
#endif
                                    && (!has_q || aligned(*q_out))),
                "token_gather: 16-byte vectors need 16-byte aligned bank, stash and outputs");
    const int64_t threads = lanes * groups * tpc;
    TORCH_CHECK(groups >= 1 && tpc >= 1 && threads % 32 == 0 && threads <= kMaxThreads,
                "token_gather: inconsistent geometry");
    const int64_t slice = lanes * vec_bytes / es;
    const size_t smem = groups > 1 ? (has_q ? 2 : 1) * tpc * groups * slice * sizeof(float) : 0;
    TORCH_CHECK(smem <= 48 * 1024,
                "token_gather: geometry needs ", smem, " bytes of shared memory");
    if (tokens == 0) return;

#if defined(GATHER_NS)
    a.stash = stash.numel() ? stash.data_ptr() : nullptr;
    TORCH_CHECK(!kpos.has_value() || (kpos->scalar_type() == at::kInt && kpos->is_contiguous() &&
                kpos->device() == bank.device() && kpos->numel() == idx.numel() && stash.numel() > 0 &&
                crow >= 1 && crow == chunk * a.op[0].width && stash.numel() % (crow * hd) == 0 &&
                row0 % chunk == 0 && ((row0 + tokens + chunk - 1) / chunk) * crow * hd <= stash.numel()),
                "token_gather_dc: kpos int32 like idx, and a compact snapshot of [chunks * crow, hd]");
    a.kpos = kpos.has_value() ? kpos->data_ptr<int>() : nullptr;
    a.crow = static_cast<int>(crow);
#else
    a.stash = stash.data_ptr();
#endif
    a.row0 = row0;
    a.stash_row0 = stash_row0;
    a.out_row0 = out_row0;
    a.hd = static_cast<int>(hd);
    a.vpr = static_cast<int>(hd * es / vec_bytes);
    a.tokens = static_cast<int>(tokens);
    a.chunk = static_cast<int>(chunk);
    a.valid = static_cast<int>(valid);
    const int width = has_q ? std::max(a.op[0].width, a.op[1].width) : a.op[0].width;
    a.per_group = static_cast<int>((width + groups - 1) / groups);

    const c10::cuda::CUDAGuard guard(bank.device());
    const dim3 grid((tokens + tpc - 1) / tpc, (hd + slice - 1) / slice);
    const dim3 block(lanes, groups, tpc);
    const int vb = static_cast<int>(vec_bytes);
#if defined(GATHER_NS)
    fx_pdl_next = static_cast<int>(pdl);   // read by launch() of this call only (every call sets it first)
#endif
    if (bank.scalar_type() == at::kBFloat16) {
        if (has_q) dispatch<uint16_t, true>(a, vb, grid, block, smem);
        else dispatch<uint16_t, false>(a, vb, grid, block, smem);
    } else {
        if (has_q) dispatch<uint32_t, true>(a, vb, grid, block, smem);
        else dispatch<uint32_t, false>(a, vb, grid, block, smem);
    }
}
#if defined(GATHER_NS)


// Plan: the checked launch of token_gather_dc for the no-snapshot call (stash empty, kpos absent) at the LARGEST row0
// a step will use (every check that depends on row0 is monotonic in it: 0 <= row0, row0 + tokens <= index rows; the
// stash / kpos checks are vacuous without a stash), captured instead of launched. A step re-checks 0 <= row0 <= row0_max
// and 0 <= valid <= chunk, then launches the captured kernel with Args.row0 / Args.valid replaced. The plan holds raw
// pointers: the caller keeps every tensor alive and unresized while it uses the plan.
namespace {
struct FxGatherPlan {
    uint64_t magic;
    int64_t row0_max;
    int32_t device;
    int32_t blocks;      // 0: nothing to launch (tokens == 0)
    int64_t chunk;       // the planned call's chunk (valid's bound, also for an empty plan)
    const void* fn;
    Args a;
    dim3 grid, block;
    size_t smem;
};
constexpr uint64_t kGatherPlanMagic = 0x4e414c50475846ull;
}  // namespace

at::Tensor token_gather_dc_plan(
    at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor stash, at::Tensor out,
    std::optional<at::Tensor> q_idx, std::optional<at::Tensor> q_w, std::optional<at::Tensor> q_out,
    int64_t row0_max, int64_t out_row0, int64_t tokens, int64_t chunk,
    int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc)
{
    TORCH_CHECK(stash.numel() == 0, "token_gather_dc_plan: the planned gather writes no compact rows (empty stash)");
    TORCH_CHECK(row0_max >= 0, "token_gather_dc_plan: row0_max range");
    TORCH_CHECK(fx_cap == nullptr, "token_gather_dc_plan: nested plan capture");
    FxGatherCap cap{};
    fx_cap = &cap;
    try {
        token_gather_dc(bank, idx, w, stash, out, q_idx, q_w, q_out, row0_max, 0, out_row0, tokens, chunk, chunk,
                        vec_bytes, lanes, groups, tpc, std::nullopt, 0, 0);
    } catch (...) {
        fx_cap = nullptr;
        throw;
    }
    fx_cap = nullptr;
    FxGatherPlan pl;
    std::memset(&pl, 0, sizeof(pl));
    pl.magic = kGatherPlanMagic;
    pl.row0_max = row0_max;
    pl.device = static_cast<int32_t>(bank.get_device());
    pl.chunk = chunk;
    pl.blocks = cap.set ? 1 : 0;
    if (cap.set) {
        pl.fn = cap.fn;
        pl.a = cap.a;
        pl.grid = cap.grid;
        pl.block = cap.block;
        pl.smem = cap.smem;
    }
    at::Tensor t = at::empty({static_cast<int64_t>(sizeof(FxGatherPlan))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(t.data_ptr<uint8_t>(), &pl, sizeof(pl));
    return t;
}

void token_gather_dc_step(at::Tensor plan, int64_t row0, int64_t valid, int64_t pdl)
{
    TORCH_CHECK(plan.device().is_cpu() && plan.scalar_type() == at::kByte && plan.is_contiguous() &&
                plan.numel() == static_cast<int64_t>(sizeof(FxGatherPlan)), "token_gather_dc_step: not a gather plan");
    FxGatherPlan pl;
    std::memcpy(&pl, plan.data_ptr<uint8_t>(), sizeof(pl));
    TORCH_CHECK(pl.magic == kGatherPlanMagic, "token_gather_dc_step: not a gather plan (magic)");
    TORCH_CHECK(pdl == 0 || pdl == 1, "token_gather_dc_step: pdl must be 0 or 1");
    TORCH_CHECK(row0 >= 0 && row0 <= pl.row0_max, "token_gather_dc_step: row0 ", row0, " out of [0, ", pl.row0_max, "]");
    TORCH_CHECK(valid >= 0 && valid <= pl.chunk, "token_gather_dc_step: valid out of [0, chunk]");
    if (pl.blocks == 0) return;
    pl.a.row0 = row0;
    pl.a.valid = static_cast<int>(valid);
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(pl.device));
    auto stream = at::cuda::getCurrentCUDAStream();
    void* args[] = {(void*)&pl.a};
    if (pdl == 1) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = pl.grid;
        cfg.blockDim = pl.block;
        cfg.dynamicSmemBytes = pl.smem;
        cfg.stream = stream.stream();
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, pl.fn, args));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    C10_CUDA_CHECK(cudaLaunchKernel(pl.fn, pl.grid, pl.block, args, pl.smem, stream.stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// token_gather_dc_step as a plain C++ entry point for mover_fx's fused deterministic step loop (one call per forward
// slab). Same checks and the same launch as token_gather_dc_step, on the given stream; returns 0, a cudaError_t, or a
// negative check code (-1 magic, -2 row0, -3 valid, -4 pdl, -5 device, -6 size). On a CUDA error the error state is
// consumed.
namespace {
int64_t fx_gather_plan_launch(const void* plan_bytes, int64_t nbytes, int64_t row0, int64_t valid, int64_t pdl,
                              void* stream) {
    if (nbytes != static_cast<int64_t>(sizeof(FxGatherPlan))) return -6;   // exact-size handshake
    FxGatherPlan pl;
    std::memcpy(&pl, plan_bytes, sizeof(pl));
    if (pl.magic != kGatherPlanMagic) return -1;
    if (!(pdl == 0 || pdl == 1)) return -4;
    if (row0 < 0 || row0 > pl.row0_max) return -2;
    if (valid < 0 || valid > pl.chunk) return -3;
    if (pl.blocks == 0) return 0;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev != pl.device) {
        (void)cudaGetLastError();
        return -5;
    }
    pl.a.row0 = row0;
    pl.a.valid = static_cast<int>(valid);
    void* args[] = {(void*)&pl.a};
    cudaError_t e;
    if (pdl == 1) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = pl.grid;
        cfg.blockDim = pl.block;
        cfg.dynamicSmemBytes = pl.smem;
        cfg.stream = static_cast<cudaStream_t>(stream);
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        e = cudaLaunchKernelExC(&cfg, pl.fn, args);
    } else
    {
        e = cudaLaunchKernel(pl.fn, pl.grid, pl.block, args, pl.smem, static_cast<cudaStream_t>(stream));
    }
    if (e == cudaSuccess) e = cudaGetLastError();   // = C10_CUDA_KERNEL_LAUNCH_CHECK's probe
    else (void)cudaGetLastError();
    return static_cast<int64_t>(e);
}
}  // namespace

int64_t token_gather_dc_launch_addr() { return reinterpret_cast<int64_t>(&fx_gather_plan_launch); }
#endif
"""

_CPP_SOURCE = r"""
#include <optional>
#if defined(GATHER_NS)
void token_gather_dc(
#else
void token_gather(
#endif
    at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor stash, at::Tensor out,
    std::optional<at::Tensor> q_idx, std::optional<at::Tensor> q_w, std::optional<at::Tensor> q_out,
    int64_t row0, int64_t stash_row0, int64_t out_row0,
    int64_t tokens, int64_t chunk, int64_t valid,
#if defined(GATHER_NS)
    int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc,
    std::optional<at::Tensor> kpos, int64_t crow, int64_t pdl);
#else
    int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc);
#endif
#if defined(GATHER_NS)

at::Tensor token_gather_dc_plan(
    at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor stash, at::Tensor out,
    std::optional<at::Tensor> q_idx, std::optional<at::Tensor> q_w, std::optional<at::Tensor> q_out,
    int64_t row0_max, int64_t out_row0, int64_t tokens, int64_t chunk,
    int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc);
void token_gather_dc_step(at::Tensor plan, int64_t row0, int64_t valid, int64_t pdl);
int64_t token_gather_dc_launch_addr();
#endif
"""

@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_token_gather",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=("token_gather",),
        extra_cuda_cflags=("-O3",),
    )


class GatherGeometry(NamedTuple):
    """Launch geometry of one token_gather call (see the module docstring)."""

    vec_bytes: int  # bytes per vector load: 16, or one element for rows not a multiple of 16 B
    lanes: int  # threads per row group; a column slice is lanes vectors wide
    slices: int  # column slices per row (grid y)
    groups: int  # row groups per token; fixes the rounding
    tpc: int  # tokens per CTA
    threads: int  # threads per CTA


_ELEMENT_BYTES = {torch.bfloat16: 2, torch.float32: 4}
_WARPS_PER_SM = 24  # grid size that entry splitting aims for, in warps per SM


def _pow2_ceil(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pow2_floor(x: int) -> int:
    return 1 << (x.bit_length() - 1)


def gather_geometry(
    head_dim: int, dtype: torch.dtype, width: int, tokens: int, *, num_sms: int
) -> GatherGeometry:
    """Geometry for ``tokens`` tokens of ``width`` = max(W, R) entries of ``head_dim``.

    A row group has one lane per 16-byte vector of the row (8, 16 or 32 lanes; longer rows
    are cut into 32-vector column slices). Entries are split over more row groups, keeping at
    least four per group, until the grid holds about ``_WARPS_PER_SM`` warps per SM, and a
    CTA takes enough tokens to reach 128 threads.
    """
    if dtype not in _ELEMENT_BYTES:
        raise TypeError(f"token_gather supports bfloat16 and float32 banks, got {dtype}")
    es = _ELEMENT_BYTES[dtype]
    vec_bytes = 16 if head_dim * es % 16 == 0 else es
    vpr = head_dim * es // vec_bytes
    lanes = max(8, _pow2_ceil(vpr)) if vec_bytes == 16 and vpr <= 16 else 32
    slices = -(-vpr // lanes)
    max_groups = _pow2_floor(max(1, width // 4))
    groups = 1
    while (groups < max_groups and 2 * lanes * groups <= _MAX_THREADS
           and tokens * slices * groups * lanes < 32 * _WARPS_PER_SM * num_sms):
        groups *= 2
    tpc = max(1, 128 // (lanes * groups))
    return GatherGeometry(vec_bytes, lanes, slices, groups, tpc, lanes * groups * tpc)


def token_gather(
    bank: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    stash: torch.Tensor,
    out: torch.Tensor,
    geometry: GatherGeometry,
    *,
    q_idx: torch.Tensor | None = None,
    q_weights: torch.Tensor | None = None,
    q_out: torch.Tensor | None = None,
    tokens: int,
    chunk: int,
    valid: int,
    row0: int = 0,
    stash_row0: int = 0,
    out_row0: int = 0,
) -> None:
    """Gather ``tokens`` consecutive token rows starting at ``row0``, in place.

    The bank must not be written while the launch runs, and no output may overlap it.

    Args:
        bank: [slots, head_dim] bfloat16 or float32, contiguous.
        idx, weights: [..., W] bank rows (int32 or int64, not bounds checked) and weights
            (bfloat16 or float32), read at rows ``row0 + t``.
        stash: [..., W, head_dim] in the bank dtype; receives the raw rows at row
            ``stash_row0 + t``.
        out: [..., head_dim] float32; receives the weighted sums at row ``out_row0 + t``.
        geometry: from ``gather_geometry``.
        q_idx, q_weights, q_out: optional second operand [..., R] (no stash); same
            dtypes as the first, output rows at ``out_row0 + t``.
        tokens: tokens in this launch, a multiple of ``chunk``.
        chunk, valid: token ``t`` is padding when ``t % chunk >= valid``.
        row0, stash_row0, out_row0: first rows of this launch in each tensor, so that
            per-step launches need no slicing.
    """
    _module().token_gather(
        bank, idx, weights, stash, out, q_idx, q_weights, q_out,
        row0, stash_row0, out_row0, tokens, chunk, valid,
        geometry.vec_bytes, geometry.lanes, geometry.groups, geometry.tpc,
    )

def tail_supported(bank, chunk_size, quant="none") -> bool:
    """movers.tail_admits hook (native tails): this gather runs the call's tail class
    (rows that are not 16-byte multiples take element-sized vectors, see gather_cuda.gather_geometry: arch-neutral code)
    on every backend of _device.backend (modern, a100; rtx with chunks of <= 128 tokens, rule 0e), so a tail class takes
    the same route everywhere. Unquantized snapshots only."""
    import torch as _t
    # snapshot_quant bf16 (float32 banks): the DW writes the compact rows; this gather is the unquantized one
    if quant not in ("none", "bf16") or not bank.is_cuda or bank.dtype not in (_t.bfloat16, _t.float32):
        return False
    from ._device import backend
    try:
        b = backend(bank.device)
    except RuntimeError:
        return False
    return b in ("modern", "a100") or (b == "rtx" and chunk_size <= 128)
