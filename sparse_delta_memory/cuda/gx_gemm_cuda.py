# Copyright (c) Meta Platforms, Inc. and affiliates.
"""gx_gemm_cuda: the deterministic backward's per-step d_u update, torch.baddbmm(d_u_ti, x_t_ti, d_dv_ti, out=d_u_ti) with
x_t = x.transpose(1, 2), as one small FP32 kernel with a FIXED reduction order:

  d_u[n, i, c] = d_u[n, i, c] + acc,   acc = fmaf(x[n, j, i], d_dv[n, j, c], acc) for j = 0, 1, .., C-1 from acc = +0

for every chunk n = ti * P + p of step ti, row i < C and column c < hd. Each output element is read, computed and
written by ONE thread with exactly that chain, so the result does not depend on the tiling, the launch geometry or the
run, and the in-place update is race-free.

That chain and the single rounding of d_u + acc are what cuBLAS's FP32 batched GEMM computed for the backward's call
on every configuration tested (allow_tf32 off; batch P >= 2, C >= 2), but that is not a cuBLAS guarantee, so the kernel
is qualified at run time (matches_cublas: per key covering device / arch, cuBLAS version, shapes, strides, pointer
alignment classes, dtypes, precision / BLAS / workspace settings; 6 input patterns incl. zeros, large magnitudes,
denormals, Inf / NaN), and the backward uses it only where the two agree bit for bit; everywhere else, and during a
CUDA graph capture of a configuration not qualified yet, it keeps cuBLAS. Either way the outputs are cuBLAS's bits.
matches_cublas first applies a speed rule from shapes only (one_wave: the launch grid fits one wave at one CTA per SM);
multi-wave shapes keep cuBLAS without a probe.

PDL (per launch opt-in by the caller, step(ti, pdl=True); the modern backend only): the kernel's x block and its d_u
values (d_u(ti) is written only by this kernel after the backward's initial bmm) are read before griddepcontrol.wait;
d_dv (the previous kernel's output, the step's GWR) after it; griddepcontrol.launch_dependents follows the wait. The
portable device passes (sm_80 / 86 / 89 / 120 ...: SDM_PORTABLE_ARCH) compile without griddepcontrol and run plain
launches."""

import functools
import os
import threading
import time

import torch

from ._build import _target_architectures, load_cuda_extension
from ._device import (backend, capability as _device_capability, cuda_backend, headers, num_sms,
                      pdl_ok as device_pdl_ok, smem_cap, smem_per_sm)

_CPP = r"""
void gx_gemm(at::Tensor du, at::Tensor x, at::Tensor dv, int64_t ti, int64_t P, int64_t pdl, int64_t ti_rows,
             int64_t smem_min);
void gx_gemm_attrs(at::Tensor out);
at::Tensor gx_gemm_plan(at::Tensor du, at::Tensor x, at::Tensor dv, int64_t ti_max, int64_t P, int64_t ti_rows,
                        int64_t smem_min);
int64_t gx_gemm_launch_addr();
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include "sdm_copy.cuh"
#include "sdm_plancap.cuh"

#ifndef GX_GEMM_TI
#define GX_GEMM_TI 16       // only the instance gx_gemm_attrs reports (rows per CTA are chosen per launch)
#endif

namespace {

constexpr int TC = 64;               // columns c per CTA
constexpr int RC = 4;                // columns per thread (one float4 of the staged d_dv row)
constexpr int TX = TC / RC;          // 16 threads across the columns
constexpr int NT = 256;
constexpr int TY = NT / TX;          // 16 thread rows

#ifdef SDM_PORTABLE_ARCH
#define GX_GEMM_SM90(...)
#else
#define GX_GEMM_SM90(...) __VA_ARGS__
#endif

// The staging copies are asynchronous (cp.async), so all of a thread's global loads are in flight at once; past C / hd
// they zero-fill (src-size 0 reads nothing). 16-byte copies (.cg: L2 only) when the rows are 16-byte aligned; x otherwise
// in 4-byte copies (.ca); d_dv otherwise with plain L2 loads. The caller passes a valid global address as `src` for
// zero-filled vectors (the tensor's base; never read: src-size 0), so no out-of-array pointer is ever formed. cp.async
// needs sm_80+; pre-sm_80 compile passes trap instead (supported() admits sm_80+ only, so they are never launched).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define GX_GEMM_PRE_SM80 1
#endif
__device__ __forceinline__ void cp16(float* dst, const float* src, int bytes) {
#ifdef GX_GEMM_PRE_SM80
    __trap();
#else
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void cp4(float* dst, const float* src, bool valid) {
#ifdef GX_GEMM_PRE_SM80
    __trap();
#else
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 4 : 0) : "memory");
#endif
}
__device__ __forceinline__ void cp_wait_all() {
#ifdef GX_GEMM_PRE_SM80
    __trap();
#else
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

// Grid: P * nib * ncb CTAs (b: cb = b % ncb, ib = b / ncb % nib, p = rest). Shared memory: xs [C][TI] = x[n, j, i0 + ii]
// (zero past C), ds [C][TC] = d_dv[n, j, c0 + cc] (zero past hd).
// TI (rows per CTA: 16, 32, 48 or 64) is a template parameter chosen per launch; PRE (always true): the thread's d_u
// values are read before griddepcontrol.wait into registers (d_u(ti) is written only by this kernel: the backward's
// initial bmm wrote it long before the step loop; nothing in flight writes it) and the epilogue stores float4s. The
// chain per element and the single rounding of d_u + acc do not depend on TI.
template <int TI, bool PRE>
__global__ void __launch_bounds__(NT) gx_gemm_kernel(float* __restrict__ du, const float* __restrict__ x,
                                                     const float* __restrict__ dv, int ti, int P, int C, int hd,
                                                     int nib, int ncb) {
    constexpr int RI = TI / TY;          // rows per thread
    static_assert(TI % TY == 0 && RI >= 1 && RI <= 4, "TI: 16, 32, 48 or 64");
    extern __shared__ __align__(16) float sm[];
    float* xs = sm;
    float* ds = sm + C * TI;
    unsigned b = blockIdx.x;
    const int cb = static_cast<int>(b % static_cast<unsigned>(ncb));
    b /= static_cast<unsigned>(ncb);
    const int ib = static_cast<int>(b % static_cast<unsigned>(nib));
    const int p = static_cast<int>(b / static_cast<unsigned>(nib));
    const int64_t n = static_cast<int64_t>(ti) * P + p;
    const int i0 = ib * TI, c0 = cb * TC;
    const float* xn = x + n * C * C;                              // x[n] [C][C]: x[n, j, i] at j * C + i
    const float* dvn = dv + n * static_cast<int64_t>(C) * hd;     // d_dv rows of chunk n [C][hd]
    float* dun = du + n * static_cast<int64_t>(C) * hd;           // d_u rows of chunk n [C][hd]
    // x: a forward tensor (staged before the wait under PDL). Rows of C floats: 16-byte vectors when C % 4 == 0 (i0 and
    // ii are multiples of 4, so a vector is entirely valid or entirely past C).
    if ((C & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & 15u) == 0) {
#pragma unroll 4
        for (int v = threadIdx.x; v < C * (TI / 4); v += NT) {
            const int j = v / (TI / 4), ii = 4 * (v - j * (TI / 4));
            const bool ok = i0 + ii < C;
            cp16(xs + j * TI + ii, ok ? xn + static_cast<int64_t>(j) * C + i0 + ii : x, ok ? 16 : 0);
        }
    } else {
#pragma unroll 4
        for (int e = threadIdx.x; e < C * TI; e += NT) {
            const int j = e / TI, ii = e - j * TI;
            const bool ok = i0 + ii < C;
            cp4(xs + e, ok ? xn + static_cast<int64_t>(j) * C + i0 + ii : x, ok);
        }
    }
    const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
    // PRE: this thread's d_u values (rows ty + TY * r, columns c0 + tx * RC + q), before the wait. Vector path when
    // hd % 4 == 0 and d_u is 16-byte aligned (a float4 is then entirely valid or entirely past hd).
    const bool vec_u = (hd & 3) == 0 && (reinterpret_cast<uintptr_t>(du) & 15u) == 0;
    float4 u0[PRE ? RI : 1];
    {
#pragma unroll
        for (int r = 0; r < RI; ++r) {
            const int i = i0 + ty + TY * r;
            const int c = c0 + tx * RC;
            u0[r] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < C && c < hd) {                       // pointers formed inside the row only
                const float* o = dun + static_cast<int64_t>(i) * hd + c;
                if (vec_u) {
                    u0[r] = __ldcg(reinterpret_cast<const float4*>(o));   // hd % 4 == 0: the 4 floats are in the row
                } else {
                    u0[r].x = __ldcg(o);
                    if (c + 1 < hd) u0[r].y = __ldcg(o + 1);
                    if (c + 2 < hd) u0[r].z = __ldcg(o + 2);
                    if (c + 3 < hd) u0[r].w = __ldcg(o + 3);
                }
            }
        }
    }
    GX_GEMM_SM90(asm volatile("griddepcontrol.wait;" ::: "memory"));              // the previous kernel: complete
    GX_GEMM_SM90(asm volatile("griddepcontrol.launch_dependents;" ::: "memory"));
    // d_dv: the previous kernel's output (after the wait), through L2 only. 16-byte vectors when hd % 4 == 0 (c0, cc are
    // multiples of 4: a vector is entirely valid or entirely past hd); otherwise plain L2 loads.
    if ((hd & 3) == 0 && (reinterpret_cast<uintptr_t>(dv) & 15u) == 0) {
#pragma unroll 4
        for (int v = threadIdx.x; v < C * (TC / 4); v += NT) {
            const int j = v / (TC / 4), cc = 4 * (v - j * (TC / 4));
            const bool ok = c0 + cc < hd;
            cp16(ds + j * TC + cc, ok ? dvn + static_cast<int64_t>(j) * hd + c0 + cc : dv, ok ? 16 : 0);
        }
    } else {
        for (int e = threadIdx.x; e < C * TC; e += NT) {
            const int j = e / TC, cc = e - j * TC;
            ds[e] = (c0 + cc < hd) ? __ldcg(dvn + static_cast<int64_t>(j) * hd + c0 + cc) : 0.f;
        }
    }
    cp_wait_all();                                                // this thread's copies landed; the barrier publishes
    __syncthreads();                                              // every thread's copies and stores to the CTA
    float acc[RI][RC];
#pragma unroll
    for (int r = 0; r < RI; ++r) {
#pragma unroll
        for (int q = 0; q < RC; ++q) acc[r][q] = 0.f;
    }
    // The fixed chain: j = 0, 1, .., C - 1 for every element (rows ty + TY * r, columns tx * RC + q of the tile).
#pragma unroll 4
    for (int j = 0; j < C; ++j) {
        const float4 d4 = *reinterpret_cast<const float4*>(ds + j * TC + tx * RC);
#pragma unroll
        for (int r = 0; r < RI; ++r) {
            const float xv = xs[j * TI + ty + TY * r];
            acc[r][0] = fmaf(xv, d4.x, acc[r][0]);
            acc[r][1] = fmaf(xv, d4.y, acc[r][1]);
            acc[r][2] = fmaf(xv, d4.z, acc[r][2]);
            acc[r][3] = fmaf(xv, d4.w, acc[r][3]);
        }
    }
#pragma unroll
    for (int r = 0; r < RI; ++r) {
        const int i = i0 + ty + TY * r;
        if (i >= C) continue;
        {
            const int c = c0 + tx * RC;
            if (c >= hd) continue;                       // the thread's 4 columns are all past hd
            float* o = dun + static_cast<int64_t>(i) * hd + c;
            const float4 v = make_float4(u0[r].x + acc[r][0], u0[r].y + acc[r][1], u0[r].z + acc[r][2],
                                         u0[r].w + acc[r][3]);   // single rounding of each f32 sum
            if (vec_u) {
                *reinterpret_cast<float4*>(o) = v;
            } else {
                o[0] = v.x;
                if (c + 1 < hd) o[1] = v.y;
                if (c + 2 < hd) o[2] = v.z;
                if (c + 3 < hd) o[3] = v.w;
            }
        }
    }
}

using GemmK = void (*)(float*, const float*, const float*, int, int, int, int, int, int);

void set_smem(GemmK kern, size_t bytes) {   // cudaFuncSetAttribute is per (kernel, device)
    static std::mutex mu;
    static std::map<std::pair<const void*, int>, size_t> done;
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_pair(reinterpret_cast<const void*>(kern), dev);
    auto it = done.find(key);
    if (it != done.end() && it->second >= bytes) return;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    done[key] = bytes;
}

GemmK pick(int ti_rows) {
    switch (ti_rows) {
        case 16: return &gx_gemm_kernel<16, true>;
        case 32: return &gx_gemm_kernel<32, true>;
        case 48: return &gx_gemm_kernel<48, true>;
        default: return &gx_gemm_kernel<64, true>;
    }
}

}  // namespace

void gx_gemm(at::Tensor du, at::Tensor x, at::Tensor dv, int64_t ti, int64_t P, int64_t pdl, int64_t ti_rows,
             int64_t smem_min) {
    const at::cuda::CUDAGuard guard(du.device());
    TORCH_CHECK(ti_rows == 16 || ti_rows == 32 || ti_rows == 48 || ti_rows == 64, "gx_gemm: rows per CTA 16, 32, 48, 64");
    const int TI = static_cast<int>(ti_rows);
    const GemmK kern = pick(TI);
    TORCH_CHECK(du.scalar_type() == at::kFloat && x.scalar_type() == at::kFloat && dv.scalar_type() == at::kFloat,
                "gx_gemm: f32 d_u, x, d_dv");
    TORCH_CHECK(du.is_contiguous() && x.is_contiguous() && dv.is_contiguous(), "gx_gemm: contiguous operands");
    // Raw pointers read the storage: refuse lazy negative / conjugate views (materialized values only).
    TORCH_CHECK(!du.is_neg() && !x.is_neg() && !dv.is_neg() && !du.is_conj() && !x.is_conj() && !dv.is_conj(),
                "gx_gemm: lazy negative / conjugate views are not supported (materialize them first)");
    TORCH_CHECK(x.dim() == 3 && x.size(1) == x.size(2) && du.dim() == 3 && du.size(0) == x.size(0) &&
                du.size(1) == x.size(1), "gx_gemm: x [N, C, C], d_u [N, C, hd]");
    const int64_t N = x.size(0), C = x.size(1), hd = du.size(2);
    TORCH_CHECK(C >= 1 && C <= 256 && hd >= 1 && hd < (int64_t(1) << 20), "gx_gemm: 1 <= C <= 256, hd");
    TORCH_CHECK(dv.numel() == du.numel(), "gx_gemm: d_dv [N * C, hd] like d_u");
    // Bounds by division before any product (no signed overflow for any int64 ti / P).
    TORCH_CHECK(N >= 1 && N < (int64_t(1) << 31) && P >= 1 && P <= N && ti >= 0 && ti < N / P,
                "gx_gemm: step out of range (need 1 <= P <= N < 2^31 and 0 <= ti < N / P)");
    TORCH_CHECK(x.device() == du.device() && dv.device() == du.device(), "gx_gemm: operands on one device");
    // d_u (written in place) must not overlap the operands it is computed from.
    {
        auto lo_hi = [](const at::Tensor& t) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(t.data_ptr());
            return std::make_pair(a, a + static_cast<uintptr_t>(t.numel()) * sizeof(float));
        };
        const auto u = lo_hi(du), xr = lo_hi(x), vr = lo_hi(dv);
        TORCH_CHECK(!(u.first < xr.second && xr.first < u.second) && !(u.first < vr.second && vr.first < u.second),
                    "gx_gemm: d_u overlaps x or d_dv (in-place update needs disjoint operands)");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(du.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(dv.data_ptr()) % 4 == 0, "gx_gemm: 4-byte aligned operands");
    const int nib = static_cast<int>((C + TI - 1) / TI), ncb = static_cast<int>((hd + TC - 1) / TC);
    const int64_t blocks64 = P * static_cast<int64_t>(nib) * ncb;
    TORCH_CHECK(blocks64 <= 0x7FFFFFFF, "gx_gemm: grid");
    const int blocks = static_cast<int>(blocks64);
    TORCH_CHECK(smem_min >= 0 && smem_min <= (int64_t(1) << 20), "gx_gemm: smem_min");
    const size_t smem = std::max(static_cast<size_t>(C) * (TI + TC) * sizeof(float), static_cast<size_t>(smem_min));
    int dev = 0, optin = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK(smem <= static_cast<size_t>(optin), "gx_gemm: ", smem, " bytes of shared memory exceed ", optin);
    if (blocks == 0) return;
    set_smem(kern, smem);
    auto st = at::cuda::getCurrentCUDAStream();
    float* dup = du.data_ptr<float>();
    const float* xp = x.data_ptr<float>();
    const float* dvp = dv.data_ptr<float>();
    const int ti_ = static_cast<int>(ti), P_ = static_cast<int>(P), C_ = static_cast<int>(C), hd_ = static_cast<int>(hd);
    // plan capture: records this launch; gx_gemm_plan_launch issues it per step, plain or PDL as below
    if (sdm_plancap::take(kern, dim3(blocks), dim3(NT), smem, dup, xp, dvp, ti_, P_, C_, hd_, nib, ncb)) return;
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(NT);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, dup, xp, dvp, ti_, P_, C_, hd_, nib, ncb));
        return;
    }
    kern<<<blocks, NT, smem, st>>>(dup, xp, dvp, ti_, P_, C_, hd_, nib, ncb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gx_gemm_attrs(at::Tensor out) {   // [4] int64 CPU: regs, local bytes, static smem, max threads
    TORCH_CHECK(out.device().is_cpu() && out.scalar_type() == at::kLong && out.numel() == 4, "gx_gemm_attrs: int64 [4]");
    cudaFuncAttributes a{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, pick(GX_GEMM_TI)));
    auto o = out.accessor<int64_t, 1>();
    o[0] = a.numRegs;
    o[1] = static_cast<int64_t>(a.localSizeBytes);
    o[2] = static_cast<int64_t>(a.sharedSizeBytes);
    o[3] = a.maxThreadsPerBlock;
}

// ---- the backward's C++ step loop (bwd_steps_cuda): gx_gemm as a per-backward plan ------------------------------------
// gx_gemm_plan runs gx_gemm ONCE under plan capture at step ti_max (every check runs; its step bound ti < N / P covers
// every ti <= ti_max) and records the launch. gx_gemm_plan_launch (a plain C++ entry point; address from
// gx_gemm_launch_addr) sets the step parameter ti and issues the recorded launch: with programmatic stream serialization
// when pdl != 0, else plain -- gx_gemm's choice for the same pdl. The plan holds raw pointers into d_u, x and d_dv.
namespace {
constexpr uint64_t kGemmPlanMagic = 0x314e414c504d4547ull;
constexpr int kGemmTiParam = 3;     // (du, x, dv, ti, P, C, hd, nib, ncb)
struct GemmPlan {
    uint64_t magic;
    int32_t device;
    int32_t pad;
    int64_t ti_max;
    sdm_plancap::Launch cap;
};

int64_t gx_gemm_plan_launch(const void* plan_bytes, int64_t nbytes, int64_t ti, int64_t pdl, void* stream) {
    GemmPlan pl;
    if (!sdm_plancap::from_bytes(pl, plan_bytes, nbytes, kGemmPlanMagic)) return -1;
    if (ti < 0 || ti > pl.ti_max) return -2;
    if (pl.cap.set == 0) return 0;                 // gx_gemm launched nothing (empty grid)
    if (sdm_plancap::current_device() != pl.device) return -5;
    int* tip = sdm_plancap::param<int>(pl.cap, kGemmTiParam);
    if (tip == nullptr) return -7;
    *tip = static_cast<int>(ti);
    return sdm_plancap::launch(pl.cap, pdl != 0, static_cast<cudaStream_t>(stream));
}
}  // namespace

at::Tensor gx_gemm_plan(at::Tensor du, at::Tensor x, at::Tensor dv, int64_t ti_max, int64_t P, int64_t ti_rows,
                        int64_t smem_min) {
    TORCH_CHECK(ti_max >= 0 && ti_max < (int64_t(1) << 31), "gx_gemm_plan: ti_max");
    GemmPlan pl;
    std::memset(&pl, 0, sizeof(pl));
    {
        sdm_plancap::Scope scope(&pl.cap);
        gx_gemm(du, x, dv, ti_max, P, 0, ti_rows, smem_min);
    }
    pl.magic = kGemmPlanMagic;
    pl.device = static_cast<int32_t>(du.get_device());
    pl.ti_max = ti_max;
    TORCH_CHECK(pl.cap.set == 0 || (pl.cap.nparams == 9 && sdm_plancap::param<int>(pl.cap, kGemmTiParam) != nullptr),
                "gx_gemm_plan: parameters");
    return sdm_plancap::to_tensor(pl);
}

int64_t gx_gemm_launch_addr() { return reinterpret_cast<int64_t>(&gx_gemm_plan_launch); }
"""

def _one_wave_ctas(device=None) -> int:
    """CTAs of one wave at the default one-CTA-per-SM cap: the SM count (GB300: 152)."""
    return num_sms(device)


def _grid_target(device=None) -> int:
    """CTAs per launch the rows rule aims at: the SM count rounded up to whole 32-CTA groups (GB300: 160)."""
    return 32 * -(-num_sms(device) // 32)


def one_wave(P: int, C: int, hd: int, device=None) -> bool:
    """Speed rule (shapes only): the launch grid at the rows rule's rows per CTA fits one wave of _one_wave_ctas(device)
    CTAs at the default one-CTA-per-SM cap. On multi-wave grids the kernel would serialize waves of single CTAs per SM,
    while cuBLAS packs several."""
    ti = _rows(P, C, hd, device)
    return P * (-(-C // ti)) * (-(-hd // 64)) <= _one_wave_ctas(device)


def _rows(P: int = 1, C: int = 64, hd: int = 1024, device=None) -> int:
    """Rows per CTA (outputs do not depend on it): the smallest of 16, 32, 48, 64 whose grid P * ceil(C / TI) *
    ceil(hd / 64) has at most _grid_target(device) CTAs, else 64."""
    target = _grid_target(device)
    for ti in (16, 32, 48):
        if P * (-(-C // ti)) * (-(-hd // 64)) <= target:
            return ti
    return 64


def _cap() -> int:
    """Max CTAs of this kernel per SM (1) under PDL, enforced by padding the dynamic shared memory request to more than
    1 / (cap + 1) of the SM's shared memory (_device.smem_per_sm: 228 KB on H100 / B200 / GB300; the padding is never
    read): a PDL launch behind a kernel that is still finishing then spreads its CTAs one per SM instead of packing up
    to 8 onto the few SMs the predecessor's last wave frees. Clamped to the device's opt-in shared memory per block.
    Plain launches (no PDL: the portable backends) are not padded. Outputs do not depend on it."""
    return 1


@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_gx_gemm",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("gx_gemm", "gx_gemm_attrs", "gx_gemm_plan", "gx_gemm_launch_addr"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo"),
    )


def supported(d_u, x, d_dv, chunk_size: int) -> bool:
    """The backward's check, once per backward (shapes, dtypes, layout only; the policy is the caller's): f32 contiguous
    d_u [N, C, hd], x [N, C, C] and d_dv (N * C * hd elements) on one CUDA device, 1 <= C <= 256, and the kernel's shared
    memory (C * (TI + 64) * 4 bytes) within the device's opt-in limit."""
    ts = (d_u, x, d_dv)
    if not all(torch.is_tensor(t) and t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in ts):
        return False
    if any(t.is_neg() or t.is_conj() for t in ts):   # the kernel reads storage (no lazy views)
        return False
    if len({t.device for t in ts}) != 1 or x.dim() != 3 or d_u.dim() != 3:
        return False
    if not cuda_backend(d_u.device):                 # cp.async: sm_80+ (PDL on the modern backend only)
        return False
    if any(t.data_ptr() % 4 for t in ts):            # the native float-alignment check, mirrored
        return False
    N, C = x.shape[0], x.shape[1]
    if not 1 <= N < 2 ** 31:                         # no empty step set (make_step / the native entry need N >= 1)
        return False
    if x.shape[2] != C or C != chunk_size or not 1 <= C <= 256 or d_u.shape[0] != N or d_u.shape[1] != C:
        return False
    hd = d_u.shape[2]
    if not 1 <= hd < 2 ** 20 or d_dv.numel() != d_u.numel() or N >= 2 ** 31:
        return False
    if _overlaps(d_u, x) or _overlaps(d_u, d_dv):      # the in-place update needs disjoint operands
        return False
    # the launch grid (P * ceil(C / TI) * ceil(hd / 64) CTAs, P <= N) must fit the native int limit for any
    # TI (the largest grid: TI 16), and the shared memory C * (TI + 64) * 4 bytes the opt-in limit for any TI (64).
    if N * (-(-C // 16)) * (-(-hd // 64)) > 0x7FFFFFFF:
        return False
    return C * (64 + 64) * 4 <= smem_cap(d_u.device)


def _overlaps(a, b) -> bool:
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


PDL_LAUNCHES = [0, 0]   # launches without / with programmatic dependent launch since import (tests, diagnostics)
LAST_GEOM = {}          # the last production make_step's launch geometry (tests, diagnostics)
PROBE_LAUNCHES = [0, 0] # the qualification probe's launches (kept out of PDL_LAUNCHES)
_DEV = {}               # device index -> (compute capability, name): read once (the key is computed per backward)

_QUAL = {}                    # qualification key -> bool (this kernel == the production cuBLAS call, bitwise)
_QUAL_ERR = {}                # key -> exceptions seen (transient failures are retried up to 3 times, then cached False)
_QUAL_LOCK = threading.Lock()
QUAL_STATS = {"lookups": 0, "probes": 0, "equal": 0, "differ": 0, "errors": 0, "capture_skips": 0, "declined_key": 0,
              "declined_shape": 0, "key_s": [], "lock_s": [], "build_s": [], "probe_s": [], "failed_probe_s": []}
_ALIGN = 256                  # pointer alignment classes kept distinct in the key and replicated exactly by the probe
_LIB = {}


def _blas_libs():
    """(libcublas path, version, libcublasLt path, Lt version) of the libraries this process actually loaded (from
    /proc/self/maps: exactly one distinct path of each, else None; versions via cublasGetProperty / cublasLtGetVersion),
    plus the libcublas ctypes handle for the native getters. Cached per process; anything unreadable -> None."""
    if "v" in _LIB:
        return _LIB["v"]
    out = None
    try:
        import ctypes
        torch.cuda.current_blas_handle()            # cuBLAS is initialized (and its libraries mapped) from here on
        blas, lt = set(), set()
        with open("/proc/self/maps") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 6:
                    base = os.path.basename(parts[5])
                    if base.startswith("libcublasLt.so"):
                        lt.add(os.path.realpath(parts[5]))
                    elif base.startswith("libcublas.so"):
                        blas.add(os.path.realpath(parts[5]))
        if len(blas) == 1 and len(lt) <= 1:          # an ambiguous library set -> no key
            bpath = next(iter(blas))
            lib = ctypes.CDLL(bpath)
            ver = []
            for prop in (0, 1, 2):                 # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                v = ctypes.c_int(-1)
                if lib.cublasGetProperty(ctypes.c_int(prop), ctypes.byref(v)) != 0 or v.value < 0:
                    raise RuntimeError("cublasGetProperty")
                ver.append(v.value)
            ltpath, ltver = "", -1
            if lt:
                ltpath = next(iter(lt))
                ltlib = ctypes.CDLL(ltpath)
                ltlib.cublasLtGetVersion.restype = ctypes.c_size_t
                ltver = int(ltlib.cublasLtGetVersion())
            out = (bpath, tuple(ver), ltpath, ltver, lib)
    except Exception:  # noqa: BLE001
        out = None
    _LIB["v"] = out
    return out


# Native cuBLAS handle state that can change the algorithm cuBLAS picks or its arithmetic: read with the
# library's getters on the handle torch uses for the current stream. A getter the loaded library does not export is
# recorded as absent (that feature does not exist there); one that exists but fails -> no key (decline).
_NATIVE_GETTERS = ("cublasGetMathMode", "cublasGetAtomicsMode", "cublasGetPointerMode_v2", "cublasGetSmCountTarget",
                   "cublasGetEmulationStrategy", "cublasGetEmulationSpecialValuesSupport",
                   "cublasGetFixedPointEmulationMantissaControl", "cublasGetFixedPointEmulationMaxMantissaBitCount",
                   "cublasGetFixedPointEmulationMantissaBitOffset")


def _native_state(lib):
    """(handle pointer, {getter: value | 'absent'}, cuBLAS workspace bytes, cuBLASLt workspace bytes) of the current
    stream's torch cuBLAS handle, or None if anything is unreadable."""
    try:
        import ctypes
        h = torch.cuda.current_blas_handle()
        vals = []
        for fn in _NATIVE_GETTERS:
            f = getattr(lib, fn, None)
            if f is None:
                vals.append((fn, "absent"))
                continue
            v = ctypes.c_int64(0)                  # every getter writes a 32-bit enum / int (zero-extended here)
            if f(ctypes.c_void_p(h), ctypes.byref(v)) != 0:
                return None
            vals.append((fn, int(v.value) & 0xFFFFFFFF))
        ws = getattr(torch._C, "_cuda_getCublasWorkspaceSize", None)
        wsl = getattr(torch._C, "_cuda_getCublasLtWorkspaceSize", None)
        if ws is None or wsl is None:              # the effective workspace policy is unreadable -> decline
            return None
        return (int(h), tuple(vals), int(ws()), int(wsl()))
    except Exception:  # noqa: BLE001
        return None


# The environment variables that can change the production GEMM's algorithm or bits, read once (torch / cuBLAS
# read them once per process). Read sites: torch 2.13.0 sources; cuBLAS: its documentation.
_BLAS_ENV = (
    "CUBLAS_WORKSPACE_CONFIG",           # torch CublasHandlePool.cpp:313 (parseChosenWorkspaceSize), kept in
                                         # getChosenWorkspaceSize's function-local static (:422); cuBLAS: handle creation
    "CUBLASLT_WORKSPACE_SIZE",           # torch CublasHandlePool.cpp:383, kept in getCUDABlasLtWorkspaceSize's static (:448)
    "TORCH_CUBLASLT_UNIFIED_WORKSPACE",  # torch CublasHandlePool.cpp:358 / :372 (function-local static)
    "TORCH_CUBLAS_WORKSPACE_CACHE",      # torch CublasHandlePool.cpp:359 / :362-363 (function-local static)
    "TORCH_BLAS_PREFER_CUBLASLT",        # torch aten/src/ATen/Context.h:547-555 (at::Context member initializer: once)
    "TORCH_BLAS_PREFER_HIPBLASLT",       # torch Context.h:549 / :552 (the alias, same initializer)
    "DISABLE_ADDMM_CUDA_LT",             # torch aten/src/ATen/native/cuda/Blas.cpp:177 (function-local static)
    "NVIDIA_TF32_OVERRIDE",              # cuBLAS / CUDA math libraries: TF32 off, read at library initialization
    "CUBLAS_EMULATE_SINGLE_PRECISION",   # cuBLAS 12.9+: FP32 emulation, read at handle creation (torch pools handles)
    "CUBLAS_EMULATION_STRATEGY",         # cuBLAS 12.9+: the emulation strategy, read at handle creation
)
# Deliberately not read: cuBLAS's logging-only variables (CUBLAS_LOGINFO_DBG, CUBLAS_LOGDEST_DBG, CUBLASLT_LOG_*), which
# change no algorithm and no bit.
_BLAS_ENV_VALUES = None              # _BLAS_ENV's (name, value or None) pairs, read once (_blas_env)
_BLAS_ENV_LOCK = threading.Lock()


def _blas_env():
    """_BLAS_ENV's values, read once per process (torch and cuBLAS read them only at initialization)."""
    global _BLAS_ENV_VALUES
    vals = _BLAS_ENV_VALUES
    if vals is None:
        with _BLAS_ENV_LOCK:
            if _BLAS_ENV_VALUES is None:
                _BLAS_ENV_VALUES = tuple((k, os.environ.get(k)) for k in _BLAS_ENV)
            vals = _BLAS_ENV_VALUES
    return vals


def _blas_settings():
    """Everything torch / cuBLAS read when choosing and running the production GEMM that this process can change, as a
    hashable tuple; None when any of it is unreadable (the caller then declines: any doubt -> cuBLAS). The environment
    part is _BLAS_ENV's values as read once (_blas_env): no environment read per call."""
    try:
        new = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
        env = _blas_env()
        return (bool(torch._C._get_cublas_allow_tf32()), torch.get_float32_matmul_precision(), str(new),
                str(torch.backends.cuda.preferred_blas_library()),
                bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
                bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                torch.are_deterministic_algorithms_enabled(), env)
    except Exception:  # noqa: BLE001  (unreadable state -> decline)
        return None


def _align_classes(d_u, x, d_dv, P, steps):
    """The distinct (d_u, x, d_dv) step-pointer residues modulo _ALIGN over all steps (each operand's residue sequence
    repeats with period <= _ALIGN / 4, so 64 steps cover every combination)."""
    C, hd = x.shape[1], d_u.shape[2]
    bu, bx, bv = d_u.data_ptr(), x.data_ptr(), d_dv.data_ptr()
    su, sx = P * C * hd * 4, P * C * C * 4
    return tuple(sorted({((bu + ti * su) % _ALIGN, (bx + ti * sx) % _ALIGN, (bv + ti * su) % _ALIGN)
                         for ti in range(min(steps, _ALIGN // 4))}))


def _qual_key(d_u, x, d_dv, P):
    """The qualification key: device (index, compute capability, name), the routing profile and code image, the loaded
    cuBLAS / cuBLASLt (paths + versions), torch and CUDA runtime versions, the production call's shapes, dtypes, strides
    (step views) and pointer alignment classes, every precision / BLAS setting (_blas_settings) and the native handle
    state (_native_state: torch's handle for the current stream, its math / atomics / pointer modes, SM-count target,
    FP32-emulation settings, and the effective cuBLAS / cuBLASLt workspace sizes). None if any part is unreadable."""
    libs = _blas_libs()
    st = _blas_settings()
    if libs is None or st is None:
        return None
    nat = _native_state(libs[4])
    if nat is None:
        return None
    dev = d_u.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    N, C, hd = x.shape[0], x.shape[1], d_u.shape[2]
    steps = N // P
    x_t = x.transpose(1, 2)
    views = (tuple(d_u.stride()), tuple(x_t.stride()), tuple(d_dv.view(N, C, hd).stride()))
    dv = _DEV.get(idx)
    if dv is None:
        dv = _DEV.setdefault(idx, (torch.cuda.get_device_capability(idx), torch.cuda.get_device_name(idx)))
    # LAST field: the routing profile and the code image's build inputs (a qualification never crosses either)
    arch = _target_architectures()
    prof = (_device_capability(idx), backend(idx), arch["TORCH_CUDA_ARCH_LIST"], tuple(map(tuple, arch["capabilities"])))
    return (idx, dv[0], dv[1], libs[:4], torch.__version__,
            torch.version.cuda, int(P), int(C), int(hd), (d_u.dtype, x.dtype, d_dv.dtype), views,
            _align_classes(d_u, x, d_dv, P, steps), st, nat, prof)


def _probe_inputs(P, C, hd, dev, pattern, seed):
    """One step's operands (x [P, C, C], d_dv [P * C, hd], d_u [P, C, hd]) on the CPU, deterministic, in one of the
    qualification patterns: normal (with signed zeros), zeros, large magnitudes, denormals, cancellation, specials."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(P, C, C, generator=g) * 0.25
    dv = torch.randn(P * C, hd, generator=g)
    du = torch.randn(P, C, hd, generator=g)
    if pattern == "normal":
        x.view(-1)[::7] = -0.0
        dv.view(-1)[::11] = 0.0
        du.view(-1)[::13] = -0.0
    elif pattern == "zeros":
        x.mul_((torch.rand(P, C, C, generator=g) < 0.5).float())
        x[x == 0] = -0.0
        dv.mul_((torch.rand(P * C, hd, generator=g) < 0.5).float())
        du[torch.rand(P, C, hd, generator=g) < 0.5] = -0.0
    elif pattern == "large":
        for t in (x, dv, du):
            t.mul_(torch.exp2(torch.randint(-60, 61, t.shape, generator=g).float()))
        x.view(-1)[::17] = 3.0e38
        dv.view(-1)[::19] = -3.0e38
    elif pattern == "denormal":
        x.mul_(1e-20)
        dv.mul_(1e-19)
        du.mul_(1e-38)
        du.view(-1)[::5] = 1e-41
    elif pattern == "cancel":
        x.copy_(torch.sign(torch.randn(P, C, C, generator=g)) * (1.0 + torch.rand(P, C, C, generator=g)))
        dv.mul_(1e4)
        v3 = dv.view(P, C, hd)
        h = C // 2
        if h:
            v3[:, 1:2 * h:2, :] = -v3[:, 0:2 * h:2, :] * (1.0 + 1e-3 * torch.randn(P, h, hd, generator=g))
        du.mul_(1e-3)
    elif pattern == "special":
        pats = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 1e-41, 3.4e38, -3.4e38])
        for t, k in ((dv, 64), (x, 16), (du, 32)):
            fl = t.view(-1)
            k = min(k, fl.numel())
            fl[torch.randperm(fl.numel(), generator=g)[:k]] = pats[torch.arange(k) % pats.numel()]
    return x.to(dev), dv.to(dev), du.to(dev)


PATTERNS = ("normal", "zeros", "large", "denormal", "cancel", "special")


def _at(n, res, dev):
    """A contiguous float32 view of n elements whose data pointer is `res` (mod _ALIGN) past an _ALIGN boundary."""
    buf = torch.empty(n + _ALIGN // 4 * 2, device=dev, dtype=torch.float32)
    off = ((res - buf.data_ptr()) % _ALIGN) // 4
    return buf[off:off + n]


def _probe(P, C, hd, dev, classes):
    """True iff, for every alignment class and every pattern, this kernel and the production cuBLAS call
    (torch.baddbmm(u, x_t, d_dv, out=u) on [P, C, *] step views: u, d_dv contiguous, x_t = x.transpose(1, 2)) give
    bitwise identical results, under the current settings."""
    seed = 20261020 + C * 131 + hd
    for ru, rx, rv in classes:
        for k, pat in enumerate(PATTERNS):
            x0, dv0, du0 = _probe_inputs(P, C, hd, dev, pat, seed + 7919 * k)
            x = _at(P * C * C, rx, dev).view(P, C, C)
            dv = _at(P * C * hd, rv, dev).view(P * C, hd)
            ref = _at(P * C * hd, ru, dev).view(P, C, hd)
            out = _at(P * C * hd, ru, dev).view(P, C, hd)
            x.copy_(x0)
            dv.copy_(dv0)
            ref.copy_(du0)
            out.copy_(du0)
            torch.baddbmm(ref, x.transpose(1, 2), dv.view(P, C, hd), out=ref)
            make_step(out, x, dv, P, _probe_counts=PROBE_LAUNCHES)(0, pdl=False)
            if not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
                return False
    return True


def matches_cublas(d_u, x, d_dv, num_parallel: int) -> bool:
    """Run-time qualification: True iff this kernel reproduces the backward's production cuBLAS call bit for bit for THIS
    backward's operands, as far as anything that can change cuBLAS's algorithm or its arithmetic is concerned. The
    verdict is cached per key (_qual_key: device / arch / name, loaded cuBLAS and cuBLASLt paths + versions, torch /
    CUDA versions, P, C, hd, dtypes, step-view strides, step-pointer alignment classes mod 256, every precision / BLAS
    setting incl. the cuBLAS workspace variables, the native state of torch's cuBLAS handle and the effective workspace
    sizes). A miss runs _probe: every alignment class of this backward replicated exactly, 6 input patterns (normal with
    signed zeros, zeros, large magnitudes, denormals, cancellation, Inf / NaN), every output bit compared. Any doubt ->
    False (cuBLAS): unreadable key, probe exception, a key change during the probe, a capture in progress on a miss.
    Everything runs on the OPERANDS' device (torch.cuda.device(d_u.device)): the native handle state read, the capture
    check, the probe and the re-read concern the device the production call runs on, whatever the ambient current
    device. The lookup runs once per backward (in its factory), never per step. First the speed rule (one_wave, shapes
    only): a multi-wave shape returns False at once (no key, no probe), so the backward keeps cuBLAS there."""
    try:
        P0, C0, hd0 = int(num_parallel), int(x.shape[1]), int(d_u.shape[2])
        fast = one_wave(P0, C0, hd0, d_u.device)
    except Exception:  # noqa: BLE001
        fast = False
    if not fast:
        with _QUAL_LOCK:
            QUAL_STATS["lookups"] += 1
            QUAL_STATS["declined_shape"] += 1
        return False
    t_key = time.perf_counter()
    try:
        dev_ctx = torch.cuda.device(d_u.device)
    except Exception:  # noqa: BLE001
        return False
    with dev_ctx:
        try:
            P = int(num_parallel)
            key = _qual_key(d_u, x, d_dv, P)
        except Exception:  # noqa: BLE001
            key = None
        t_key = time.perf_counter() - t_key
        t_lock = time.perf_counter()
        with _QUAL_LOCK:
            t_lock = time.perf_counter() - t_lock
            QUAL_STATS["lookups"] += 1
            if len(QUAL_STATS["key_s"]) < 64:
                QUAL_STATS["key_s"].append(round(t_key, 6))
                QUAL_STATS["lock_s"].append(round(t_lock, 6))
            if key is None:
                QUAL_STATS["declined_key"] += 1
                return False
            hit = _QUAL.get(key)
            if hit is not None:
                return hit
            try:
                capturing = torch.cuda.is_current_stream_capturing()
            except Exception:  # noqa: BLE001
                capturing = True
            if capturing:
                QUAL_STATS["capture_skips"] += 1
                return False
            t0 = time.perf_counter()
            try:
                tb = time.perf_counter()
                _module()                              # first use: JIT build / load (timed apart from the probe)
                QUAL_STATS["build_s"].append(round(time.perf_counter() - tb, 6))
                t0 = time.perf_counter()
                ok = _probe(P, x.shape[1], d_u.shape[2], d_u.device, key[11])
                torch.cuda.synchronize(d_u.device)
                if _qual_key(d_u, x, d_dv, P) != key:    # settings changed during the probe -> not a valid verdict
                    raise RuntimeError("qualification key changed during the probe")
            except Exception:  # noqa: BLE001
                QUAL_STATS["failed_probe_s"].append(round(time.perf_counter() - t0, 6))
                QUAL_STATS["errors"] += 1
                n = _QUAL_ERR.get(key, 0) + 1
                _QUAL_ERR[key] = n
                if n >= 3:
                    _QUAL[key] = False
                return False
            QUAL_STATS["probe_s"].append(round(time.perf_counter() - t0, 6))
            _QUAL[key] = ok
            QUAL_STATS["probes"] += 1
            QUAL_STATS["equal" if ok else "differ"] += 1
            return ok


def make_step(d_u, x, d_dv, num_parallel: int, _probe_counts=None):
    """The per-step update d_u(ti) += x(ti)^T d_dv(ti) for the backward's reverse loop: step(ti, pdl=False). The caller
    opts in to PDL per launch (it vouches that the kernel right before writes nothing the kernel reads before its wait:
    x and d_u(ti), which the backward's initial bmm wrote before the step loop and only this kernel writes afterwards);
    step.pdl_capable says whether this build can. (_probe_counts: the qualification probe's own launch counter, so that
    PDL_LAUNCHES / LAST_GEOM describe production launches only.)"""
    mod = _module()
    counts = PDL_LAUNCHES if _probe_counts is None else _probe_counts
    P = int(num_parallel)
    du3 = d_u
    dv = d_dv
    pdl_ok = device_pdl_ok(d_u.device)
    N = x.shape[0]
    if not 1 <= P <= N:
        raise ValueError(f"gx_gemm: need 1 <= num_parallel <= N chunks; got P {P}, N {N}")
    lim = N // P
    ti_rows = _rows(P, x.shape[1], d_u.shape[2], d_u.device)   # once per factory call (once per backward)
    cap = _cap()
    # > 1 / (cap + 1) of the SM: at most cap CTAs (programmatic launches only)
    smem_min = 0 if cap == 0 or not pdl_ok else smem_per_sm(d_u.device) // (cap + 1) + 1024
    if smem_min:
        smem_min = min(smem_min, smem_cap(d_u.device))

    def step(ti: int, pdl: bool = False) -> None:
        if not 0 <= int(ti) < lim:
            raise ValueError(f"gx_gemm: step {ti} outside [0, {lim})")
        use = 1 if (pdl and pdl_ok) else 0
        mod.gx_gemm(du3, x, dv, int(ti), P, use, ti_rows, smem_min)
        counts[use] += 1

    def plan(nsteps: int, pdl: bool = False):
        """(plan, launch address, use) of steps 0 .. nsteps - 1 for bwd_steps_cuda; use: 1 = PDL."""
        p = mod.gx_gemm_plan(du3, x, dv, int(nsteps) - 1, P, ti_rows, smem_min)
        return p, int(mod.gx_gemm_launch_addr()), (1 if (pdl and pdl_ok) else 0)

    def count(n: int, use: int) -> None:
        counts[use] += int(n)

    step.pdl_capable = pdl_ok
    step.plan = plan
    step.count = count
    step.geom = dict(ti_rows=ti_rows, cap=cap, smem_min=smem_min)
    if _probe_counts is None:
        LAST_GEOM.clear()
        LAST_GEOM.update(step.geom)
    return step
