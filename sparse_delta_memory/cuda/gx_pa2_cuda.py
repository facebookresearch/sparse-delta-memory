# Copyright (c) Meta Platforms, Inc. and affiliates.
"""gx_pa2_cuda: Phase A of the deterministic backward (the write side of the backward step), bitwise equal to pa_dc_cuda's
dc_pa3_kernel on its common configurations (chosen once per backward; anything else keeps dc_pa3).

Row types (bank / snapshot / grad): bf16 / bf16 / bf16, bf16 / bf16 / f32 (grad_bank_fp32), f32 / f32 / f32 and f32 /
bf16 / f32 (snapshot_quant bf16; the restore stores f32(x'), exact). Grad banks of 2^32 elements or more and float32 rows
take the W64 instantiations (64-bit row offsets). Tails (unquantized, tail_supported): slice rows that are not 16-byte
multiples, uneven slices and hd % 4 != 0 heads run in sdm_gx_pa2_tail (8-byte or element pieces; columns past the head
hold zeros), with dc_pa3's arithmetic.

Same walk, arithmetic, ring contents and consumption order as dc_pa3_kernel<EG, EB, 8, false, false, false>, so every
output has dc_pa3's bits. Only the scheduling differs: a lane-parallel round pump (16-byte cp.async row copies into the
ring, one commit group per round), 32-bit LAST-store offsets below 2^32 elements, ring depth 16 or 8 and 8 or 16 warps
per CTA (plan()), the token-row staging wait after the first pump, and its own tile length (outputs do not depend on it).
Launch modes: gx_pa2 triggers its dependents at its first instruction; gx_pa2_pdl (after gx_gemm_cuda's GEMM) is itself a
programmatic launch whose token-row reads and writes all follow griddepcontrol.wait."""

import functools

import types

import torch

from ._build import load_cuda_extension
from ._device import RESERVED_SMEM, headers, num_sms
# device-class reads memoized per device ordinal and forced row (values = _device's)
from ._devmemo import backend, capability, cuda_backend, smem_cap, smem_per_sm
from .gx_gds_cuda import ROW_HELPERS

_CPP = r"""
#include <torch/extension.h>
void gx_pa2(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc);
void gx_pa2_pdl(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc);
int64_t gx_pa2_static_smem(bool bf_bank, bool g32, bool w64, int64_t dev, int64_t quant);
void gx_pa2_bf(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc);
void gx_pa2_bf_pdl(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc);
at::Tensor gx_pa2_plan(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti_max, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc, bool pdl_launch, bool kbf);
int64_t gx_pa2_launch_addr();
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include "sdm_copy.cuh"
#include "sdm_plancap.cuh"

namespace {

constexpr uint32_t kFirst = 0x80000000u;
constexpr uint32_t kLast = 0x40000000u;
constexpr uint32_t kInvalid = 0x20000u;
constexpr int kDistShift = 18;
constexpr uint32_t kDistMax = 4095u;
constexpr unsigned kFull = 0xFFFFFFFFu;
#ifndef GX_PA2_THREADS
#define GX_PA2_THREADS 256
#endif
#define GX_PA2_MINB 2
// griddepcontrol is a modern-pass instruction. The portable device passes (SDM_PORTABLE_ARCH, sdm_copy.cuh: sm_80 / 86
// / 89 / 120 ...) compile without it, as gx_gds_cuda does: the trigger is dropped (dependents then launch at completion,
// the default) and the wait traps (the host requests mode 1 on the modern backend only: _device.pdl_ok).
#ifdef SDM_PORTABLE_ARCH
#define GX_PA2_GDC_TRIGGER()
#define GX_PA2_GDC_WAIT() __trap()
#else
#define GX_PA2_GDC_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define GX_PA2_GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#endif

@ROW_HELPERS@
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(src_bytes) : "memory");
}
// One 16-byte piece of a ring row (nb valid source bytes, zero fill): one 16-byte copy, or (PB 8: rows whose stride or
// slice start is a multiple of 8 bytes only) two 8-byte copies of the same bytes.
template <int PB>
__device__ __forceinline__ void cp_piece(void* dst, const void* src, int nb) {
    if constexpr (PB == 16) {
        cp_async16(dst, src, nb);
    } else {
        cp_async8(dst, src, min(nb, 8));
        cp_async8(static_cast<char*>(dst) + 8, static_cast<const char*>(src) + 8, max(nb - 8, 0));
    }
}
// Element pieces (PB 2: hd % 4 != 0, grad / bank / snapshot rows aligned to their element only): nb valid source bytes of
// a 16-byte piece, zeros past them; float32 elements by 4-byte copies, bf16 elements by plain loads into shared memory
// (the warp synchronizes before any lane reads a ring entry, as for the copies).
template <typename E>
__device__ __forceinline__ void cp_piece_e(void* dst, const void* src, int nb) {
    if constexpr (sizeof(E) == 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            cp_async4(static_cast<float*>(dst) + j, static_cast<const float*>(src) + (4 * j < nb ? j : 0),
                      4 * j < nb ? 4 : 0);
    } else {
        const unsigned short* sp = static_cast<const unsigned short*>(src);
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t lo = 4 * j < nb ? __ldg(sp + 2 * j) : 0u;
            const uint32_t hi = 4 * j + 2 < nb ? __ldg(sp + 2 * j + 1) : 0u;
            w[j] = lo | (hi << 16);
        }
        *static_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}
// A lane's 4 float32 token columns in element pieces (PB 2): 4-byte copies of the nv valid columns, zeros past them.
__device__ __forceinline__ void cp_tok4_elem(float4* dst, const float* src, int nv) {
#pragma unroll
    for (int j = 0; j < 4; ++j) cp_async4(reinterpret_cast<float*>(dst) + j, src + (j < nv ? j : 0), j < nv ? 4 : 0);
}
// Element stores of a lane's first nv columns (PB 2): rounded values (st4's rounding per element) and raw elements.
__device__ __forceinline__ void st_elems(__nv_bfloat16* p, float4 v, int nv) {
    const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (j < nv) p[j] = __float2bfloat16_rn(f[j]);
}
__device__ __forceinline__ void st_elems(float* p, float4 v, int nv) {
    const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (j < nv) p[j] = f[j];
}
__device__ __forceinline__ void st_raw_elems(__nv_bfloat16* p, uint2 r, int nv) {
    const uint32_t w[2] = {r.x, r.y};
    unsigned short* q = reinterpret_cast<unsigned short*>(p);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (j < nv) q[j] = static_cast<unsigned short>(w[j >> 1] >> (16 * (j & 1)));
}
__device__ __forceinline__ void st_raw_elems(float* p, float4 r, int nv) { st_elems(p, r, nv); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_dyn(int pending) {   // 0..7 (larger: waits for 7, stricter)
    switch (pending) {
#define GX_W(k) case k: cp_wait<k>(); return;
        GX_W(0) GX_W(1) GX_W(2) GX_W(3) GX_W(4) GX_W(5) GX_W(6)
#undef GX_W
        default: break;
    }
    cp_wait<7>();
}
// FIRST position of the group holding position j (dc_pa3's snap_back, verbatim). Warp-uniform.
__device__ __forceinline__ int snap_back(const int32_t* __restrict__ kp, int j, int n, int lane) {
    if (j <= 0) return 0;
    if (j >= n) return n;
    const uint32_t p = static_cast<uint32_t>(__ldg(kp + j));
    if (p & kFirst) return j;
    const uint32_t d = (p >> kDistShift) & kDistMax;
    if (d < kDistMax) return j - static_cast<int>(d);
    for (int base = j - static_cast<int>(kDistMax); ; base -= 32) {
        const int q = base - lane;
        const bool f = q <= 0 || (static_cast<uint32_t>(__ldg(kp + max(q, 0))) & kFirst);
        const unsigned m = __ballot_sync(kFull, f);
        if (m) return max(0, base - (__ffs(m) - 1));
    }
}
__device__ __forceinline__ float tr_comb(float a, float b, int off, int lane, int width) {
    const bool up = (lane & off) != 0;
    const float snd = up ? a : b;
    const float kp = up ? b : a;
    return __fadd_rn(kp, __shfl_xor_sync(kFull, snd, off, width));
}
__device__ __forceinline__ int4 load_rec(const int4* __restrict__ r, int j, int e) {
    if (j < e) return __ldg(r + j);
    return make_int4(static_cast<int32_t>(kFirst | kLast | kInvalid), 0, 0, 0);
}

// Grid: P * NSL * ctas blocks (block b: cta = b % ctas, slice = b / ctas % NSL, p = rest), as dc_pa3.
// D: ring depth in groups per warp (8 or 16; staging only: the outputs do not depend on it). NT: max threads per CTA
// (256: up to 8 warps, GX_PA2_MINB CTAs per SM; 512: 16 warps, 1 CTA per SM; <= 128 registers either way).
// EG / EB / ES: grad-row / bank / snapshot element types (bf16 / bf16 / bf16, f32 / bf16 / bf16, f32 / f32 / f32, or
// f32 / f32 / bf16: bf16 snapshot rows of a float32 bank, snapshot_quant bf16; ES == EB otherwise). W64: 64-bit row element
// offsets (grad rows of the pump, LAST stores); every row type but bf16 / bf16 takes them. One type-generic body
// (ROW_HELPERS: Row4, cvt4, st4, st_raw, row_off) except the pump's round, whose generic form changes the bf16 / 32-bit
// instantiations' register allocation (measured, sm_103 + sm_90): those keep the bf16 round (if constexpr (!W64)).
// QUANT 1 / 2: the snapshot rows are the quantized compact rows (fp8 / int4 payload in `snap`, f32 row scales in the
// one trailing argument QX = const float*): each lane copies the payload word holding its 4 columns (4-byte cp.async)
// and decodes it at the group's FIRST (dec4); the LAST store restores the decoded row x'. QUANT 0: no trailing argument.
// PB: bytes per copy of bf16 rows: 16, or 8 for the tail instantiations (QUANT 0: dc_pa3's slices of rows whose stride
// or slice start is a multiple of 8 bytes only: hd % 8 == 4 or slices of 4 mod 8 columns; float32 rows stay 16-byte
// pieces: hd % 4 == 0), or 2 (QUANT 0, hd % 4 != 0: every row type in element pieces, cp_piece_e; token rows by
// 4-byte copies; LAST stores of the lane's nv valid columns). The arithmetic is the same for every PB.
// ssr < 0: dc_pa3's dense compact snapshot (dbg 1792: row chunk * n + the group's valid-FIRST ordinal, from bpre);
// ssr >= 0 (QUANT 0): dc_pa3's per-entry snapshot (dbg 512, the forward's layout 0: row ti * ssr + p * C * W + tok * W
// + w of the group's FIRST record, ssr = snap_step_rows). Index arithmetic only: the rows read are dc_pa3's.
template <typename EG, typename EB, typename ES, bool W64, int QUANT, int D, int NT, int PB, typename... QX>
__global__ void __launch_bounds__(NT, (NT >= 512 ? 1 : GX_PA2_MINB)) gx_pa2_kernel(
    EG* __restrict__ grad, EB* __restrict__ bank, const ES* __restrict__ snap,
    const float* __restrict__ dva, const float* __restrict__ beta, const int32_t* __restrict__ kpay,
    const int4* __restrict__ rec, float* __restrict__ out_eff, float* __restrict__ out_g,
    int64_t plane, int ti, int P, int C, int W, int n, int hd, int NSL, int SLW, int ctas, int L,
    const int32_t* __restrict__ bpre, int64_t ssr, int pdl_mode, QX... qscales) {
    static_assert(D == 8 || D == 16, "ring depth 8 or 16");
    static_assert(PB == 16 || ((PB == 8 || PB == 2) && QUANT == 0),
                  "16-byte row pieces, or 8-byte / element pieces of unquantized tail rows");
    static_assert((QUANT == 0) == (sizeof...(QX) == 0), "quantized instantiations take the row scales");
    static_assert(std::is_same<ES, EB>::value ||
                      (QUANT == 0 && std::is_same<EG, float>::value && std::is_same<EB, float>::value &&
                       std::is_same<ES, __nv_bfloat16>::value),
                  "a snapshot type other than the bank's: bf16 rows of a float32 bank (snapshot_quant bf16), unquantized, "
                  "float32 grad rows");
    using RowG = typename Row4<EG>::T;    // a lane's 4 columns of a grad row (ring entry)
    using RowS = typename Row4<ES>::T;    // ... of a snapshot row (the bank's type, or bf16 rows of a float32 bank)
    constexpr int kPG = sizeof(RowG) / 4, kPS = sizeof(RowS) / 4;   // 16-byte pieces of a lane-row: 2 bf16, 4 f32
    static_assert(W64 || (std::is_same<EG, __nv_bfloat16>::value && std::is_same<EB, __nv_bfloat16>::value),
                  "float32 rows use 64-bit row offsets");
    // Mode 0 (plain launch, after the step's cuBLAS baddbmm, which never triggers early): everything before this
    // kernel in the stream is complete when it starts, so the next kernel may launch now. Safe for a kernel launched
    // with programmatic stream serialization that reads only data written before this kernel ahead of its
    // griddepcontrol.wait (full completion + visibility of this kernel): gx_gds_cuda's GDS with pdl=True. Mode 1
    // (launched with programmatic stream serialization after gx_gemm_cuda's GEMM): the trigger follows this kernel's wait.
    if (!pdl_mode) {
        GX_PA2_GDC_TRIGGER();
    }
    extern __shared__ float4 smem[];
    const int nw = blockDim.x >> 5;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned lt_mask = (1u << lane) - 1u;
    float4* tok_s = smem;                                                          // [C][32]
    float* beta_s = reinterpret_cast<float*>(tok_s + C * 32);                      // [C]
    RowG* const gring_all = reinterpret_cast<RowG*>(beta_s + ((C + 3) & ~3));      // [nw][D][32] grad rows
    RowG* const gring = gring_all + warp * D * 32;
    using RowR = std::conditional_t<QUANT == 0, RowS, uint32_t>;   // ring entry: raw row, or the lane's payload word
    RowR* const sring_all = reinterpret_cast<RowR*>(gring_all + nw * D * 32);      // [nw][D][32] snapshot rows
    RowR* const sring = sring_all + warp * D * 32;
    int2* const tab = reinterpret_cast<int2*>(sring_all + nw * D * 32) + warp * 64;   // [64] {grad row, snapshot row}
    [[maybe_unused]] float* const scl =                                                 // [nw][D] row scales (QUANT)
        QUANT ? reinterpret_cast<float*>(reinterpret_cast<int2*>(sring_all + nw * D * 32) + nw * 64) + warp * D : nullptr;
    int b = blockIdx.x;
    const int cta = b % ctas; b /= ctas;
    const int sl = b % NSL;
    const int p = b / NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * P + p;
    const int64_t tok0 = chunk * C;
    const int col0 = sl * SLW + 4 * lane;
    const bool on = 4 * lane < SLW && col0 < hd;
    // PB 2: the lane's valid columns (1..3 in the head's last group; 0 off the slice / head). A lane on the slice has
    // col0 + 4 <= the slice's end (SLW % 4 == 0), so only the head bounds it.
    [[maybe_unused]] const int nv = (PB == 2 && on) ? min(4, hd - col0) : 0;
    {
        const float* rb = dva + tok0 * hd;
        if (!pdl_mode) {   // mode 1: d_u's rows are the previous kernel's output: staged after the wait (below)
            for (int t = warp; t < C; t += nw) {
                if constexpr (PB == 2) {
                    cp_tok4_elem(&tok_s[t * 32 + lane], rb + static_cast<int64_t>(t) * hd + (on ? col0 : 0), nv);
                } else {
                    cp_async16(&tok_s[t * 32 + lane], rb + static_cast<int64_t>(t) * hd + (on ? col0 : 0), on ? 16 : 0);
                }
            }
            cp_commit();
        }
        for (int i = threadIdx.x; i < C; i += blockDim.x) beta_s[i] = beta[tok0 + i];
    }
    const int32_t* kp = kpay + chunk * n;
    const int4* rp = rec + chunk * n;
    const int tile = cta * nw + warp;
    const int s = snap_back(kp, tile * L, n, lane);
    const int e = snap_back(kp, (tile + 1) * L, n, lane);
    // Round pump: lane L copies the 16-byte pieces at slice-row bytes 128 q + pa, pa = 16 * (L & 7), of the round's row
    // L >> 3 (q < 2 for bf16 rows, < 4 for float32 rows); vb / vbs = valid bytes of a grad / snapshot slice row; src
    // bytes clamp(vb - offset, 0, 16) (0: zero fill).
    const int pa = 16 * (lane & 7);
    const int vb = static_cast<int>(sizeof(EG)) * min(SLW, hd - sl * SLW);
    const int nba = max(0, min(16, vb - pa));
    const int nbb = max(0, min(16, vb - 128 - pa));
    const int vbs = static_cast<int>(sizeof(ES)) * min(SLW, hd - sl * SLW);
    int nbg[kPG], nbs[kPS];
#pragma unroll
    for (int q = 0; q < kPG; ++q) nbg[q] = max(0, min(16, vb - 128 * q - pa));
#pragma unroll
    for (int q = 0; q < kPS; ++q) nbs[q] = max(0, min(16, vbs - 128 * q - pa));
    const EG* const gsl = grad + sl * SLW;
    // The snapshot row 0 of this chunk: ordinal 0's (dense compact) or the step's first entry row (per-entry, QUANT 0).
    const ES* const ssl = snap + (ssr < 0 ? chunk * static_cast<int64_t>(n)
                                          : static_cast<int64_t>(ti) * ssr + static_cast<int64_t>(p) * C * W) * hd +
                          sl * SLW;
    [[maybe_unused]] const int qpitch = QUANT == 1 ? hd : hd / 2;                  // payload bytes per row
    [[maybe_unused]] const uint8_t* const qsl = QUANT == 0 ? nullptr :
        reinterpret_cast<const uint8_t*>(snap) + chunk * static_cast<int64_t>(n) * qpitch +
        (QUANT == 1 ? sl * SLW + 4 * lane : (sl * SLW) / 2 + 4 * (lane >> 1));     // the lane's word of ordinal 0
    [[maybe_unused]] const float* qsc = nullptr;
    if constexpr (QUANT != 0) qsc = (qscales, ...) + chunk * static_cast<int64_t>(n);
    float* const oeff = out_eff + (NSL == 1 ? 0 : static_cast<int64_t>(sl) * plane);
    float* const og = out_g + (NSL == 1 ? 0 : static_cast<int64_t>(sl) * plane);
    const int64_t obase = chunk * (static_cast<int64_t>(C) * W);
    int4 w0 = load_rec(rp, s + lane, e);
    int4 w1 = load_rec(rp, s + 32 + lane, e);
    int4 w2 = load_rec(rp, s + 64 + lane, e);
    int jb = s, issued = 0, seg = 0, avail = 0;
    unsigned rbits = 0u;   // bit j: a round starts at FIRST ordinal (issued - 1 - j)
    // Dense snapshot ordinal of the tile's first valid FIRST (block prefix + valid FIRSTs in [32 * (s / 32), s)).
    int gw = 0;
    if (ssr < 0) {
        const int nb = (n + 31) >> 5;
        const int bs = s >> 5;
        const uint32_t pq = (32 * bs + lane < s) ? static_cast<uint32_t>(kp[32 * bs + lane]) : 0u;
        gw = bpre[chunk * (nb + 1) + bs] + __popc(__ballot_sync(kFull, (pq & kFirst) && !(pq & kInvalid)));
    }
    // Registers a record window's in-range FIRSTs (dc_pa3's pend / gi / si of that window) in the table.
#define PA2_REG(WR, J0)                                                                        \
    {                                                                                          \
        const uint32_t py_ = static_cast<uint32_t>(WR.x);                                      \
        const unsigned fm_ = __ballot_sync(kFull, (py_ & kFirst) && (J0) + lane < e);          \
        const unsigned vf_ = fm_ & __ballot_sync(kFull, !(py_ & kInvalid));                    \
        if ((fm_ >> lane) & 1u) {                                                              \
            const bool v_ = ((vf_ >> lane) & 1u) != 0u;                                        \
            const int o_ = ssr < 0 ? gw + __popc(vf_ & lt_mask)                                 \
                                   : static_cast<int>((py_ >> 9) & 0xFFu) * W + static_cast<int>(py_ & 0xFFu); \
            tab[(avail + __popc(fm_ & lt_mask)) & 63] = make_int2(v_ ? WR.y : -1, v_ ? o_ : -1); \
        }                                                                                      \
        avail += __popc(fm_);                                                                  \
        gw += __popc(vf_);                                                                     \
    }
    // Issues rounds of up to 4 groups while groups are registered and ring slots are free (dc_pa3: issued < seg + D).
#define PA2_PUMP()                                                                             \
    if (issued < avail && issued < seg + D) {                                                  \
        __syncwarp();                                                                          \
        do {                                                                                   \
            const int k_ = min(min(avail - issued, seg + D - issued), 4);                      \
            const int r_ = lane >> 3;                                                          \
            if (r_ < k_) {                                                                     \
                const int2 t_ = tab[(issued + r_) & 63];                                       \
                if constexpr (!W64 && QUANT == 0 && PB == 16) {   /* bf16 round (bf16 SASS) */      \
                const int ro_ = ((issued + r_) & (D - 1)) * 256 + pa;                          \
                const bool v_ = t_.x >= 0;                                                     \
                const EG* gr_ = gsl + static_cast<uint32_t>(v_ ? t_.x : 0) * static_cast<uint32_t>(hd); \
                const ES* sr_ = ssl + static_cast<int64_t>(v_ ? t_.y : 0) * hd;                \
                char* gd_ = reinterpret_cast<char*>(gring) + ro_;                              \
                char* sd_ = reinterpret_cast<char*>(sring) + ro_;                              \
                const bool a_ = v_ && nba > 0, b_ = v_ && nbb > 0;                             \
                cp_async16(gd_, a_ ? reinterpret_cast<const char*>(gr_) + pa : reinterpret_cast<const char*>(gsl), \
                           a_ ? nba : 0);                                                      \
                cp_async16(gd_ + 128, b_ ? reinterpret_cast<const char*>(gr_) + 128 + pa :     \
                                           reinterpret_cast<const char*>(gsl), b_ ? nbb : 0);  \
                cp_async16(sd_, a_ ? reinterpret_cast<const char*>(sr_) + pa : reinterpret_cast<const char*>(gsl), \
                           a_ ? nba : 0);                                                      \
                cp_async16(sd_ + 128, b_ ? reinterpret_cast<const char*>(sr_) + 128 + pa :     \
                                           reinterpret_cast<const char*>(gsl), b_ ? nbb : 0);  \
                } else {                                                                       \
                const int ro_ = ((issued + r_) & (D - 1)) * (32 * static_cast<int>(sizeof(RowG))) + pa; \
                const bool v_ = t_.x >= 0;                                                     \
                const EG* gr_ = gsl + row_off<W64>(static_cast<uint32_t>(v_ ? t_.x : 0), hd); \
                const ES* sr_ = ssl + static_cast<int64_t>(v_ ? t_.y : 0) * hd;                \
                const int rs_ = ((issued + r_) & (D - 1)) * (32 * static_cast<int>(sizeof(RowS))) + pa; \
                char* gd_ = reinterpret_cast<char*>(gring) + ro_;                              \
                char* sd_ = reinterpret_cast<char*>(sring) + rs_;                              \
                bool og_[kPG], os_[kPS];                                                       \
                _Pragma("unroll") for (int q_ = 0; q_ < kPG; ++q_) og_[q_] = v_ && nbg[q_] > 0; \
                if constexpr (QUANT == 0) {                                                    \
                _Pragma("unroll") for (int q_ = 0; q_ < kPS; ++q_) os_[q_] = v_ && nbs[q_] > 0; \
                }                                                                              \
                if constexpr (PB == 2) {   /* element pieces (QUANT 0): grad and snapshot rows */ \
                _Pragma("unroll") for (int q_ = 0; q_ < kPG; ++q_)                             \
                    cp_piece_e<EG>(gd_ + 128 * q_, og_[q_] ?                                   \
                        reinterpret_cast<const char*>(gr_) + 128 * q_ + pa : reinterpret_cast<const char*>(gsl), \
                        og_[q_] ? nbg[q_] : 0);                                                \
                _Pragma("unroll") for (int q_ = 0; q_ < kPS; ++q_)                             \
                    cp_piece_e<ES>(sd_ + 128 * q_, os_[q_] ?                                   \
                        reinterpret_cast<const char*>(sr_) + 128 * q_ + pa : reinterpret_cast<const char*>(gsl), \
                        os_[q_] ? nbs[q_] : 0);                                                \
                } else {                                                                       \
                _Pragma("unroll") for (int q_ = 0; q_ < kPG; ++q_)                             \
                    cp_piece<(sizeof(EG) == 2 ? PB : 16)>(gd_ + 128 * q_, og_[q_] ?            \
                        reinterpret_cast<const char*>(gr_) + 128 * q_ + pa : reinterpret_cast<const char*>(gsl), \
                        og_[q_] ? nbg[q_] : 0);                                                \
                if constexpr (QUANT == 0) {                                                    \
                _Pragma("unroll") for (int q_ = 0; q_ < kPS; ++q_)                             \
                    cp_piece<(sizeof(ES) == 2 ? PB : 16)>(sd_ + 128 * q_, os_[q_] ?            \
                        reinterpret_cast<const char*>(sr_) + 128 * q_ + pa : reinterpret_cast<const char*>(gsl), \
                        os_[q_] ? nbs[q_] : 0);                                                \
                }                                                                              \
                }                                                                              \
                }                                                                              \
            }                                                                                  \
            if constexpr (QUANT != 0) {   /* every lane: its payload word of each round row; lane j: row j's scale */ \
                for (int j_ = 0; j_ < k_; ++j_) {                                              \
                    const int2 tq_ = tab[(issued + j_) & 63];                                  \
                    const bool vq_ = tq_.x >= 0;                                               \
                    cp_async4(&sring[((issued + j_) & (D - 1)) * 32 + lane], vq_ && on ?      \
                              qsl + static_cast<int64_t>(tq_.y) * qpitch : reinterpret_cast<const uint8_t*>(snap), \
                              vq_ && on ? 4 : 0);                                              \
                }                                                                              \
                if (lane < k_) {                                                               \
                    const int2 tq_ = tab[(issued + lane) & 63];                                \
                    cp_async4(&scl[(issued + lane) & (D - 1)], qsc + (tq_.x >= 0 ? tq_.y : 0), tq_.x >= 0 ? 4 : 0); \
                }                                                                              \
            }                                                                                  \
            cp_commit();                                                                       \
            rbits = (rbits << k_) | (1u << (k_ - 1));                                          \
            issued += k_;                                                                      \
        } while (issued < avail && issued < seg + D);                                          \
    }
    PA2_REG(w0, s)
    PA2_REG(w1, s + 32)
    PA2_PUMP();
    if (pdl_mode) {
        // Mode 1: everything above reads data written two or more launches back (Phase 1's payloads / records / block
        // prefixes, the snapshot, beta, and G rows last written by the previous step's GDS, complete before this step's
        // GWR passed its own wait). Then the previous kernel's output, d_u's token rows, into the same smem bytes as mode 0's
        // cp.async staging (zeros past the slice): cp.async in one group newer than the rounds (see below; rbits counts
        // rounds only).
        GX_PA2_GDC_WAIT();
        GX_PA2_GDC_TRIGGER();
        const float* rb = dva + tok0 * hd;
        // The same bytes as mode 0's staging (zero fill past the slice), copied L2 -> shared memory without a register
        // round trip, every row in flight at once; one commit group, newer than the pre-wait rounds. Waiting for it
        // completes every older group too (all pre-wait rounds: the walk's later waits then only over-wait).
        for (int t = warp; t < C; t += nw) {
            if constexpr (PB == 2) {
                cp_tok4_elem(&tok_s[t * 32 + lane], rb + static_cast<int64_t>(t) * hd + (on ? col0 : 0), nv);
            } else {
                cp_async16(&tok_s[t * 32 + lane], rb + static_cast<int64_t>(t) * hd + (on ? col0 : 0), on ? 16 : 0);
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
    } else {
        cp_wait_dyn(__popc(rbits));
        __syncthreads();
    }

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 gdw = make_float4(0.f, 0.f, 0.f, 0.f), sn = make_float4(0.f, 0.f, 0.f, 0.f);
    RowS sn_raw{};
    float gp = 0.f;
    while (jb < e) {
        const unsigned inr = __ballot_sync(kFull, jb + lane < e);
        const uint32_t pay_l = static_cast<uint32_t>(w0.x);
        const unsigned fm = __ballot_sync(kFull, (pay_l & kFirst) != 0) & inr;
        const unsigned lm = __ballot_sync(kFull, (pay_l & kLast) != 0) & inr;
        const unsigned vm = __ballot_sync(kFull, !(pay_l & kInvalid)) & inr;
        const int tko_l = static_cast<int>(((pay_l >> 9) & 0xFF) * 32);
        float l3e = 0.f, l3g = 0.f, l4e = 0.f, l4g = 0.f, dot_mine = 0.f, g_mine = 0.f;
        for (int sb = 0; sb < 4; ++sb) {
            float pe[8], pg[8];
            const unsigned fsb = (fm >> (sb * 8)) & 0xFFu;
            if (fsb) {
                PA2_PUMP();
                // FIRST ordinals < seg + popc(fsb) must have landed: rounds starting at or after it may stay pending.
                const int ahead = issued - seg - __popc(fsb);
                cp_wait_dyn(ahead > 0 ? __popc(rbits & ((1u << ahead) - 1u)) : 0);
                __syncwarp();
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int k = sb * 8 + u;
                const int tk = __shfl_sync(kFull, tko_l, k);
                const float sck = __int_as_float(__shfl_sync(kFull, w0.z, k));
                const float4 x = tok_s[tk + lane];
                if ((fsb >> u) & 1u) {
                    acc = make_float4(0.f, 0.f, 0.f, 0.f);
                    const int rs = (seg & (D - 1)) * 32 + lane;
                    gdw = cvt4(gring[rs]);
                    if constexpr (QUANT == 0) {
                        sn_raw = sring[rs];
                        sn = cvt4(sn_raw);
                    } else {
                        sn = dec4<QUANT, EB>(sring[rs], lane, scl[seg & (D - 1)], sn_raw);
                        // a lane off the slice holds zeros (the decoded-rows run's zero-filled lanes) whatever the row's
                        // scale: an infinite or NaN scale times its zero-filled payload word would be NaN
                        if (!on) {
                            sn = make_float4(0.f, 0.f, 0.f, 0.f);
                            sn_raw = RowS{};
                        }
                    }
                    ++seg;
                    gp = __fmul_rn(gdw.x, sn.x);
                    gp = __fmaf_rn(gdw.y, sn.y, gp);
                    gp = __fmaf_rn(gdw.z, sn.z, gp);
                    gp = __fmaf_rn(gdw.w, sn.w, gp);
                }
                acc.x = __fmaf_rn(sck, x.x, acc.x);
                acc.y = __fmaf_rn(sck, x.y, acc.y);
                acc.z = __fmaf_rn(sck, x.z, acc.z);
                acc.w = __fmaf_rn(sck, x.w, acc.w);
                float part = __fmul_rn(x.x, sn.x);
                part = __fmaf_rn(x.y, sn.y, part);
                part = __fmaf_rn(x.z, sn.z, part);
                part = __fmaf_rn(x.w, sn.w, part);
                pe[u] = part;
                pg[u] = gp;
                if ((lm & vm) >> k & 1u) {
                    const uint32_t slot = static_cast<uint32_t>(__shfl_sync(kFull, w0.y, k));
                    const float cpe_l = __int_as_float(__shfl_sync(kFull, w0.w, k));
                    if (on) {
                        float4 o;
                        o.x = __fmaf_rn(gdw.x, cpe_l, acc.x);
                        o.y = __fmaf_rn(gdw.y, cpe_l, acc.y);
                        o.z = __fmaf_rn(gdw.z, cpe_l, acc.z);
                        o.w = __fmaf_rn(gdw.w, cpe_l, acc.w);
                        const auto off = row_off<W64>(slot, hd) + static_cast<uint32_t>(col0);
                        if constexpr (PB == 2) {
                            st_elems(grad + off, o, nv);
                            if constexpr (std::is_same<ES, EB>::value) {
                                st_raw_elems(bank + off, sn_raw, nv);
                            } else {   // bf16 snapshot rows of a float32 bank: f32(x'), exact (sn)
                                st_elems(bank + off, sn, nv);
                            }
                        } else if constexpr (std::is_same<ES, EB>::value) {
                            st4(grad + off, o);
                            st_raw(bank + off, sn_raw);
                        } else {   // bf16 snapshot rows of a float32 bank: the restore stores f32(x'), exact (sn)
                            st4(grad + off, o);
                            st4(bank + off, sn);
                        }
                    }
                }
            }
            // levels 0-2 (xor 1, 2, 4) over the sub-batch's 8 entries: lane l -> entry sb*8 + (l & 7)
#pragma unroll
            for (int i = 0; i < 4; ++i) pe[i] = tr_comb(pe[2 * i], pe[2 * i + 1], 1, lane, 32);
#pragma unroll
            for (int i = 0; i < 2; ++i) pe[i] = tr_comb(pe[2 * i], pe[2 * i + 1], 2, lane, 32);
            const float ve = tr_comb(pe[0], pe[1], 4, lane, 32);
            float vg;
            if (fsb) {
#pragma unroll
                for (int i = 0; i < 4; ++i) pg[i] = tr_comb(pg[2 * i], pg[2 * i + 1], 1, lane, 32);
#pragma unroll
                for (int i = 0; i < 2; ++i) pg[i] = tr_comb(pg[2 * i], pg[2 * i + 1], 2, lane, 32);
                vg = tr_comb(pg[0], pg[1], 4, lane, 32);
            } else {
                // No FIRST in these 8 positions: every pg[u] is the same gp (dc_pa3's octet butterfly, same bits).
                vg = gp;
                vg = __fadd_rn(vg, __shfl_xor_sync(kFull, vg, 1));
                vg = __fadd_rn(vg, __shfl_xor_sync(kFull, vg, 2));
                vg = __fadd_rn(vg, __shfl_xor_sync(kFull, vg, 4));
            }
            // levels 3-4 (xor 8, 16) across sub-batches
            if (!(sb & 1)) {
                l3e = ve; l3g = vg;
            } else {
                const float te = tr_comb(l3e, ve, 8, lane, 32);
                const float tg = tr_comb(l3g, vg, 8, lane, 32);
                if (!(sb & 2)) {
                    l4e = te; l4g = tg;
                } else {
                    dot_mine = tr_comb(l4e, te, 16, lane, 32);
                    g_mine = tr_comb(l4g, tg, 16, lane, 32);
                }
            }
        }
        if ((inr >> lane) & 1u) {   // lane l: position jb + l
            const int tok = (pay_l >> 9) & 0xFF;
            const int64_t o = obase + static_cast<int64_t>(tok) * W + (pay_l & 0xFF);
            const bool valid = !(pay_l & kInvalid);
            if (NSL == 1) {
                oeff[o] = valid ? -beta_s[tok] * dot_mine : 0.f;
                og[o] = valid ? __int_as_float(w0.w) * g_mine : 0.f;
            } else {
                oeff[o] = valid ? dot_mine : 0.f;
                og[o] = valid ? g_mine : 0.f;
            }
        }
        jb += 32;
        w0 = w1;
        w1 = w2;
        w2 = load_rec(rp, jb + 64 + lane, e);
        __syncwarp();          // table entries 64 ordinals back were read by earlier rounds (other lanes)
        PA2_REG(w1, jb + 32)
        PA2_PUMP();
    }
    cp_wait<0>();
#undef PA2_PUMP
#undef PA2_REG
}

template <typename K>
void set_smem(K kernel, size_t bytes) {
    static std::mutex mu;
    static std::map<std::pair<int, const void*>, size_t> done;
    std::lock_guard<std::mutex> g(mu);
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    auto key = std::make_pair(dev, reinterpret_cast<const void*>(kernel));
    auto it = done.find(key);
    if (it != done.end() && it->second >= bytes) return;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(bytes)));
    done[key] = bytes;
}

int optin_of(int dev) {   // cached per device (the launcher runs once per chunk step)
    static std::mutex mu;
    static std::map<int, int> cache;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(dev);
    if (it != cache.end()) return it->second;
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    cache[dev] = optin;
    return optin;
}

}  // namespace

// dc_pa's exact signature (a drop-in for the backward's per-step call); only dc_pa3's configuration in the package's backward.
namespace {
size_t static_smem_of(const void* kern, int dev) {   // a kernel's static shared memory (cached; none today: extern only)
    static std::mutex mu;
    static std::map<std::pair<int, const void*>, size_t> cache;
    std::lock_guard<std::mutex> g(mu);
    const auto key = std::make_pair(dev, kern);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    cudaFuncAttributes fa;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));   // the current device: dev (the callers' guard)
    cache[key] = fa.sharedSizeBytes;
    return fa.sharedSizeBytes;
}

// The 8-warp D 8 instantiation of a row type and quantization mode (its static shared memory: the tests' hook). The
// tail module (GX_PA2_TAILS_ONLY) holds only unquantized tail instantiations: 8-byte pieces of bf16 rows (PB 8; float32
// rows: their element-piece kernel, PB 2, the module's only float32 / float32 family).
#if defined(GX_PA2_TAILS_ONLY)
template <int Q>
const void* kern8_of(bool w64, bool bf_bank, bool g32) {
    TORCH_CHECK(Q == 0, "gx_pa2 (tail module): unquantized rows only");
    return !w64 ? reinterpret_cast<const void*>(&gx_pa2_kernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, false, 0, 8, 256, 8>)
         : !bf_bank ? reinterpret_cast<const void*>(&gx_pa2_kernel<float, float, float, true, 0, 8, 256, 2>)
         : g32 ? reinterpret_cast<const void*>(&gx_pa2_kernel<float, __nv_bfloat16, __nv_bfloat16, true, 0, 8, 256, 8>)
               : reinterpret_cast<const void*>(&gx_pa2_kernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, 0, 8, 256, 8>);
}
#else
template <int Q>
const void* kern8_of(bool w64, bool bf_bank, bool g32) {
#define GX_PA2_K8(EG_, EB_, W64_)                                                                                    \
    (Q == 0 ? reinterpret_cast<const void*>(&gx_pa2_kernel<EG_, EB_, EB_, W64_, 0, 8, 256, 16>)                      \
            : reinterpret_cast<const void*>(&gx_pa2_kernel<EG_, EB_, EB_, W64_, (Q == 0 ? 1 : Q), 8, 256, 16, const float*>))
    return !w64 ? GX_PA2_K8(__nv_bfloat16, __nv_bfloat16, false)
         : !bf_bank ? GX_PA2_K8(float, float, true)
         : g32 ? GX_PA2_K8(float, __nv_bfloat16, true) : GX_PA2_K8(__nv_bfloat16, __nv_bfloat16, true);
#undef GX_PA2_K8
}
#endif

// The launch of one instantiation family (row types EG / EB, offset width W64, quantization, bf16 row pieces PB) with
// the checked geometry: the bf16 / bf16 / 32-bit family has D 8 / 16 x 256 / 512 threads, the W64 families D 8 x 256 /
// 512 threads.
template <typename EG, typename EB, bool W64, int QUANT, int PB, typename ES = EB>
void gx_pa2_go(int D, int nw_e, size_t smem, int blocks, bool pdl_launch, at::Tensor& grad, at::Tensor& bank,
               at::Tensor& snap, at::Tensor& dva, at::Tensor& beta, at::Tensor& kpay, at::Tensor& rec,
               at::Tensor& out_eff, at::Tensor& out_g, int64_t plane, int64_t ti, int64_t P, int64_t C, int64_t W,
               int n, int hd, int nsl, int slw, int ctas, int64_t L, at::Tensor& bpre, int64_t ssr,
               const float* qscales) {
    const int threads = nw_e * 32;
    auto go = [&](auto kern, auto... qx) {   // qx: the quantized instantiations' row scales (none otherwise)
        set_smem(kern, smem);
        // plan capture: the launch below, with the mode argument its PDL / plain branch passes
        if (sdm_plancap::take(kern, dim3(blocks), dim3(threads), smem,
                reinterpret_cast<EG*>(grad.data_ptr()), reinterpret_cast<EB*>(bank.data_ptr()),
                reinterpret_cast<const ES*>(snap.data_ptr()), dva.data_ptr<float>(), beta.data_ptr<float>(),
                kpay.data_ptr<int32_t>(), reinterpret_cast<const int4*>(rec.data_ptr<int32_t>()),
                out_eff.data_ptr<float>(), out_g.data_ptr<float>(), plane, static_cast<int>(ti), static_cast<int>(P),
                static_cast<int>(C), static_cast<int>(W), n, hd, nsl, slw, ctas, static_cast<int>(L),
                bpre.data_ptr<int32_t>(), ssr, pdl_launch ? 1 : 0, qx...)) {
            return;
        }
        if (pdl_launch) {
            // Mode 1: programmatic stream serialization (a programmatic edge under graph capture): the kernel may
            // start while the previous kernel (gx_gemm_cuda's GEMM) finishes; its griddepcontrol.wait orders every
            // dependent access.
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(blocks);
            cfg.blockDim = dim3(threads);
            cfg.dynamicSmemBytes = smem;
            cfg.stream = at::cuda::getCurrentCUDAStream();
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
            C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern,
                reinterpret_cast<EG*>(grad.data_ptr()), reinterpret_cast<EB*>(bank.data_ptr()),
                reinterpret_cast<const ES*>(snap.data_ptr()), static_cast<const float*>(dva.data_ptr<float>()),
                static_cast<const float*>(beta.data_ptr<float>()), static_cast<const int32_t*>(kpay.data_ptr<int32_t>()),
                reinterpret_cast<const int4*>(rec.data_ptr<int32_t>()),
                out_eff.data_ptr<float>(), out_g.data_ptr<float>(), plane, static_cast<int>(ti), static_cast<int>(P),
                static_cast<int>(C), static_cast<int>(W), n, hd, nsl, slw, ctas, static_cast<int>(L),
                static_cast<const int32_t*>(bpre.data_ptr<int32_t>()), ssr, 1, qx...));
            return;
        }
        kern<<<blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<EG*>(grad.data_ptr()), reinterpret_cast<EB*>(bank.data_ptr()),
            reinterpret_cast<const ES*>(snap.data_ptr()), dva.data_ptr<float>(), beta.data_ptr<float>(),
            kpay.data_ptr<int32_t>(), reinterpret_cast<const int4*>(rec.data_ptr<int32_t>()),
            out_eff.data_ptr<float>(), out_g.data_ptr<float>(), plane, static_cast<int>(ti), static_cast<int>(P),
            static_cast<int>(C), static_cast<int>(W), n, hd, nsl, slw, ctas, static_cast<int>(L),
            bpre.data_ptr<int32_t>(), ssr, 0, qx...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if constexpr (QUANT != 0) {
        go(nw_e == 16 ? &gx_pa2_kernel<EG, EB, ES, W64, QUANT, 8, 512, PB, const float*>
                      : &gx_pa2_kernel<EG, EB, ES, W64, QUANT, 8, 256, PB, const float*>, qscales);
    } else if constexpr (W64) {
        go(nw_e == 16 ? &gx_pa2_kernel<EG, EB, ES, W64, 0, 8, 512, PB> : &gx_pa2_kernel<EG, EB, ES, W64, 0, 8, 256, PB>);
    } else {
        go(nw_e == 16
               ? (D == 16 ? &gx_pa2_kernel<EG, EB, ES, W64, 0, 16, 512, PB> : &gx_pa2_kernel<EG, EB, ES, W64, 0, 8, 512, PB>)
               : (D == 16 ? &gx_pa2_kernel<EG, EB, ES, W64, 0, 16, 256, PB> : &gx_pa2_kernel<EG, EB, ES, W64, 0, 8, 256, PB>));
    }
}

// KBF: the bf16 entry (gx_pa2_bf / gx_pa2_bf_pdl, regular module only), picked once per backward by
// launcher(bank=, grad=, snap=) for bf16 bank, grad bank and (unquantized) snapshot rows in 16-byte pieces below 2^32
// grad elements: the row type, the quantization mode, the piece width and the offset width are compile-time, so each
// step runs the plain host path (its checks, the caller's plan validated against the opt-in limit, one launch; no
// row-type, quantization, piece-width or offset-width selection per step). !KBF (gx_pa2 / gx_pa2_pdl): every row type
// and mode of the module. The same kernels, geometry and arguments for the rows both take.
template <bool KBF>
void gx_pa2_run(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
                at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
                at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
                int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc, bool pdl_launch) {
    (void)pipe;             // pipe: unused (dc_pa3's snapshot readers below)
    const at::cuda::CUDAGuard guard(bank.device());
    TORCH_CHECK(snap_step_rows >= 0, "gx_pa2: snapshot rows");          // dc_pa's check
    // the quantized compact rows: `snap` uint8 [rows, hd] (fp8 e4m3) or [rows, hd / 2] (int4), their f32 row scales
    // in dc_pa's gdwc slot (empty for unquantized rows). The bf16 entry: unquantized rows (0 at compile time; the
    // snapshot's dtype is checked with the other row types below).
    const int quant = KBF ? 0
                          : snap.scalar_type() == at::kByte ? (snap.dim() == 2 && snap.size(1) == bank.size(1) ? 1 : 2)
                                                            : 0;
    // dbg 1792: the dense compact rows (chunk * n + valid-FIRST ordinal); dbg 512 (unquantized rows): one snapshot row
    // per entry (the forward's layout 0), dc_pa3's row ti * snap_step_rows + p * C * W + tok * W + w of a group's FIRST.
    const bool erows = dbg == 512 && !quant;
    TORCH_CHECK((dbg == (256 | 512 | 1024) || erows) && (depth == 8 || (depth == 16 && !quant)) &&
                (quant ? gdwc.scalar_type() == at::kFloat && gdwc.dim() == 1 && gdwc.numel() == snap.size(0) &&
                         gdwc.is_contiguous() && gdwc.device() == bank.device() &&
                         reinterpret_cast<uintptr_t>(gdwc.data_ptr()) % 4 == 0
                       : gdwc.numel() == 0),
                "gx_pa2: dc_pa3's dense compact (dbg 1792) or, for unquantized rows, per-entry (dbg 512) snapshot (ring "
                "depth 8, or 16 for unquantized rows; gdwc: empty, or the f32 scales of quantized rows)");
    TORCH_CHECK(ti >= 0 && P >= 1 && C >= 1 && W >= 1, "gx_pa2: ti >= 0, P, C, W >= 1");
    // Row types (dc_pa's): bf16 bank + bf16 snapshot with bf16 or f32 grad rows, f32 bank + f32 snapshot + f32 grad
    // (the bf16 entry: bf16 bank, grad bank and snapshot, compile-time).
    const bool bf_bank = KBF || bank.scalar_type() == at::kBFloat16;
    const bool g32 = !KBF && grad.scalar_type() == at::kFloat;
    // bf16 snapshot rows of a float32 bank (snapshot_quant bf16: the quantized forward's compact rows with a scale of 1.0,
    // read here as unquantized bf16 rows); float32 grad rows; the restore stores f32(x') into the bank.
    const bool s16 = !KBF && !quant && bank.scalar_type() == at::kFloat && snap.scalar_type() == at::kBFloat16;
    if constexpr (KBF) {
        TORCH_CHECK(bank.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kBFloat16 &&
                    snap.scalar_type() == at::kBFloat16, "gx_pa2 (bf16 entry): bf16 bank, grad bank and snapshot");
    } else {
        TORCH_CHECK((bf_bank || bank.scalar_type() == at::kFloat) &&
                    (quant || s16 || snap.scalar_type() == bank.scalar_type()) &&
                    (g32 || grad.scalar_type() == at::kBFloat16) && (bf_bank || g32),
                    "gx_pa2: bf16 bank and snapshot with a bf16 or f32 grad bank, or f32 bank and grad bank with an f32 "
                    "or bf16 snapshot");
    }
    TORCH_CHECK(grad.dim() == 2 && bank.dim() == 2 && grad.sizes() == bank.sizes(), "gx_pa2: grad and bank [slots, hd]");
#if defined(GX_PA2_TAILS_ONLY)
    constexpr bool kTailMod = true;
#else
    constexpr bool kTailMod = false;
#endif
    // head_dim (read once): a multiple of 4; the tail module also takes unquantized heads of hd % 4 != 0 (element
    // pieces, the tail bound hd <= 2^20). Checked before narrowing to int.
    const int64_t hd64 = bank.size(1);
    TORCH_CHECK(hd64 > 0 && hd64 < (int64_t(1) << 31) &&
                (hd64 % 4 == 0 || (kTailMod && !quant && hd64 <= (int64_t(1) << 20))),
                "gx_pa2: head_dim a multiple of 4 in [4, 2^31) (unquantized heads of hd % 4 != 0: the tail module, "
                "hd <= 2^20)");
    const int hd = static_cast<int>(hd64);
    const int n = static_cast<int>(kpay.size(1));
    TORCH_CHECK(kpay.dim() == 2 && C <= 256 && n == C * W, "gx_pa2: C <= 256 and kpay [N, C * W]");
    // dc_pa3's column slices: SLW = 4 ceil(ceil(hd / 4) / NSL) columns (the last slice may be narrower); ceil(hd / 4)
    // without int overflow (= hd / 4 for hd % 4 == 0).
    const int nsl = 1 + (hd - 1) / 128;
    const int slw = 4 * (((1 + (hd - 1) / 4) + nsl - 1) / nsl);
    // Quantized rows: 16-byte slice rows of equal slices (8 bf16 or 4 f32 columns; a bf16 snapshot ring sets the rule
    // with f32 grad rows). Unquantized rows: hd % 4 == 0: bf16 rows in 16-byte pieces when every row stride and slice
    // start is a multiple of 16 bytes (hd % 8 == 0 and SLW % 8 == 0), else in 8-byte pieces (the tail module); float32
    // rows (hd % 4 == 0) are always 16-byte pieces; hd % 4 != 0: element pieces of every row type (PB 2, the tail
    // module).
    const int q = bf_bank ? 8 : 4;
    TORCH_CHECK(!quant || (hd % q == 0 && slw * nsl == hd && slw % q == 0),
                "gx_pa2: quantized rows need head_dim a multiple of ", q, " split into equal slices of a multiple of ", q,
                " columns");
    if constexpr (KBF) {   // 16-byte pieces: every row stride and slice start a multiple of 16 bytes
        TORCH_CHECK(hd % 8 == 0 && slw % 8 == 0, "gx_pa2 (bf16 entry): head_dim and slice width multiples of 8");
    }
    // bf16 snapshot rows of a float32 bank: dc_pa3's dense compact rows only; their pieces follow the bf16 rule (16 bytes
    // when every bf16 row stride and slice start is a multiple of 16 bytes, else 8-byte or element pieces: the tail module)
    TORCH_CHECK(!s16 || dbg == (256 | 512 | 1024),
                "gx_pa2: bf16 snapshot rows of a float32 bank need dc_pa3's dense compact rows (dbg 1792)");
    const int pb = KBF ? 16 : hd % 4 != 0 ? 2 : ((!bf_bank && !s16) || (hd % 8 == 0 && slw % 8 == 0)) ? 16 : 8;
    // int4 payload words are shared by lane pairs: rows and slices of whole 8-column groups.
    TORCH_CHECK(quant != 2 || (snap.dim() == 2 && 2 * snap.size(1) == hd && hd % 8 == 0 && slw % 8 == 0),
                "gx_pa2: int4 rows need head_dim and slice widths that are multiples of 8");
    const int64_t snap_rows = quant ? snap.size(0) : snap.numel() / hd;
    // 64-bit row offsets: grad banks of >= 2^32 elements and every row type with float32 rows (the bf16 entry: never).
    if constexpr (KBF) {
        TORCH_CHECK(grad.numel() < (int64_t(1) << 32), "gx_pa2 (bf16 entry): grad bank of 2^32 elements or more");
    }
    const bool w64 = !KBF && (!bf_bank || g32 || grad.numel() >= (int64_t(1) << 32));
    TORCH_CHECK(dva.scalar_type() == at::kFloat && dva.is_contiguous(), "gx_pa2: d_dva f32 contiguous");
    TORCH_CHECK(rec.scalar_type() == at::kInt && rec.numel() == kpay.numel() * 4, "gx_pa2: rec");
    TORCH_CHECK(kpay.scalar_type() == at::kInt, "gx_pa2: kpay int32");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "gx_pa2: beta f32");
    // Warps per CTA and ring depth: the caller's plan (gx_pa2_cuda.plan(): 1 .. 8 warps on the 256-thread
    // instantiations or 16 on the 512-thread ones, ring depth 16 only without 64-bit offsets); a regrouping of the same
    // tiles and a staging depth, so the outputs depend on neither. Validated here against the device.
    TORCH_CHECK(L >= 1 && L <= n && ((nw >= 1 && nw * 32 <= GX_PA2_THREADS) || nw == 16),
                "gx_pa2: 1 <= L <= C*W, 1 <= nw <= 8 or nw = 16");
    const int nw_e = static_cast<int>(nw), D = static_cast<int>(depth);
    TORCH_CHECK(!(w64 && D == 16), "gx_pa2: the 64-bit-offset instantiations have ring depth 8");
    // Snapshot rows: any count (the kernel reads chunk-local int32 rows from a 64-bit chunk base, ssl / qsl / qsc).
    TORCH_CHECK(grad.size(0) < (int64_t(1) << 31), "gx_pa2: row indices must fit 31 bits");
    TORCH_CHECK(grad.is_contiguous() && bank.is_contiguous() && snap.is_contiguous() && kpay.is_contiguous() &&
                rec.is_contiguous() && beta.is_contiguous() && out_eff.is_contiguous() && out_g.is_contiguous(),
                "gx_pa2: contiguous operands");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(bank.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(snap.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(dva.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "gx_pa2: 16-byte aligned row operands");
    TORCH_CHECK(erows || (bpre.scalar_type() == at::kInt && bpre.is_contiguous() &&
                          bpre.numel() == kpay.size(0) * ((kpay.size(1) + 31) / 32 + 1)), "gx_pa2: bpre [N, nb + 1]");
    const int optin = optin_of(bank.device().index());
    // ring bytes per lane-row: grad row + snapshot row (quantized: the payload word), plus one f32 scale per ring row
    const size_t row_bytes = (g32 ? sizeof(float4) : sizeof(uint2)) +
                             (quant ? sizeof(uint32_t) : (bf_bank || s16) ? sizeof(uint2) : sizeof(float4));
    auto smem_of = [&](int d, int w) {
        return static_cast<size_t>(C) * 32 * sizeof(float4) + ((C + 3) & ~3) * sizeof(float) +
               static_cast<size_t>(w) * d * 32 * row_bytes + static_cast<size_t>(w) * 64 * sizeof(int2) +
               (quant ? static_cast<size_t>(w) * d * sizeof(float) : 0);
    };
    const int64_t tiles64 = (static_cast<int64_t>(n) + L - 1) / L;
    const int64_t ctas64 = (tiles64 + nw_e - 1) / nw_e;
    TORCH_CHECK(ctas64 * nw_e * L < (int64_t(1) << 31) && P * nsl * ctas64 < (int64_t(1) << 31),
                "gx_pa2: tile geometry overflows 32 bits");
    TORCH_CHECK(kpay.size(0) >= (ti + 1) * P && kpay.size(0) < (int64_t(1) << 31) &&
                dva.numel() >= (ti + 1) * P * C * hd && beta.numel() >= (ti + 1) * P * C,
                "gx_pa2: step ", ti, " out of range");
    TORCH_CHECK((quant || snap.numel() % hd == 0) &&
                    (erows ? ti * snap_step_rows + P * C * W : (ti + 1) * P * static_cast<int64_t>(n)) <= snap_rows,
                "gx_pa2: the snapshot does not cover step ", ti);   // dc_pa's need_rows for each reader
    const int64_t planes = nsl == 1 ? 1 : nsl;
    TORCH_CHECK(out_eff.numel() == out_g.numel() && out_eff.numel() % planes == 0 &&
                out_eff.numel() / planes >= (ti + 1) * P * n && out_eff.scalar_type() == at::kFloat &&
                out_g.scalar_type() == at::kFloat, "gx_pa2: output planes do not cover step ", ti);
    for (const at::Tensor* t : {&grad, &snap, &dva, &beta, &kpay, &rec, &out_eff, &out_g})
        TORCH_CHECK(t->device() == bank.device(), "gx_pa2: all operands on the bank's device");
    TORCH_CHECK(erows || bpre.device() == bank.device(), "gx_pa2: all operands on the bank's device");
    const int64_t ssr = erows ? snap_step_rows : -1;
    const int64_t plane = out_eff.numel() / planes;
    const int ctas = static_cast<int>(ctas64);
    const int blocks = static_cast<int>(P * nsl * ctas64);
    const size_t smem = smem_of(D, nw_e);
    TORCH_CHECK(smem <= static_cast<size_t>(optin), "gx_pa2: ", smem, " bytes of shared memory exceed ", optin);
    if (blocks == 0) return;
    const float* qs = quant ? gdwc.data_ptr<float>() : nullptr;
#if defined(GX_PA2_TAILS_ONLY)
    // The tail module (sdm_gx_pa2_tail): unquantized rows in 8-byte pieces (bf16 rows, PB 8) or element pieces (hd % 4
    // != 0, every row type, PB 2). Its instantiations live apart from the regular ones: in one module they change the
    // bf16 / 32-bit kernels' sm_103 SASS (measured, job 2016051).
    static_assert(!KBF, "gx_pa2: the bf16 entry is the regular module's");
    TORCH_CHECK(!quant && (pb == 2 || ((bf_bank || s16) && pb == 8)),
                "gx_pa2 (tail module): unquantized rows that need 8-byte (bf16) or element pieces only");
#define GX_PA2_GO(EG_, EB_, W64_, PB_)                                                                               \
    gx_pa2_go<EG_, EB_, W64_, 0, PB_>(D, nw_e, smem, blocks, pdl_launch, grad, bank, snap, dva, beta, kpay, rec,     \
                                      out_eff, out_g, plane, ti, P, C, W, n, hd, nsl, slw, ctas, L, bpre, ssr, qs)
    if (pb == 2) {   // hd % 4 != 0: element pieces of every row type
        if (!w64) {
            GX_PA2_GO(__nv_bfloat16, __nv_bfloat16, false, 2);
        } else if (s16) {   // bf16 snapshot rows of a float32 bank
            gx_pa2_go<float, float, true, 0, 2, __nv_bfloat16>(D, nw_e, smem, blocks, pdl_launch, grad, bank, snap, dva,
                                                               beta, kpay, rec, out_eff, out_g, plane, ti, P, C, W, n,
                                                               hd, nsl, slw, ctas, L, bpre, ssr, qs);
        } else if (!bf_bank) {
            GX_PA2_GO(float, float, true, 2);
        } else if (g32) {
            GX_PA2_GO(float, __nv_bfloat16, true, 2);
        } else {
            GX_PA2_GO(__nv_bfloat16, __nv_bfloat16, true, 2);
        }
    } else if (!w64) {
        GX_PA2_GO(__nv_bfloat16, __nv_bfloat16, false, 8);
    } else if (s16) {   // bf16 snapshot rows of a float32 bank in 8-byte pieces (float32 grad rows: 16-byte pieces)
        gx_pa2_go<float, float, true, 0, 8, __nv_bfloat16>(D, nw_e, smem, blocks, pdl_launch, grad, bank, snap, dva, beta,
                                                           kpay, rec, out_eff, out_g, plane, ti, P, C, W, n, hd, nsl,
                                                           slw, ctas, L, bpre, ssr, qs);
    } else if (g32) {
        GX_PA2_GO(float, __nv_bfloat16, true, 8);
    } else {
        GX_PA2_GO(__nv_bfloat16, __nv_bfloat16, true, 8);
    }
#undef GX_PA2_GO
#else
    if constexpr (KBF) {
        gx_pa2_go<__nv_bfloat16, __nv_bfloat16, false, 0, 16>(D, nw_e, smem, blocks, pdl_launch, grad, bank, snap, dva,
                                                               beta, kpay, rec, out_eff, out_g, plane, ti, P, C, W, n,
                                                               hd, nsl, slw, ctas, L, bpre, ssr, qs);
    } else {
        TORCH_CHECK(pb == 16, "gx_pa2: bf16 rows in 8-byte pieces run in the tail module (launcher(bank=...))");
        auto rows = [&](auto qc) {   // the row-type dispatch, per quantization mode
            constexpr int Q = decltype(qc)::value;
#define GX_PA2_GO(EG_, EB_, W64_)                                                                                    \
            gx_pa2_go<EG_, EB_, W64_, Q, 16>(D, nw_e, smem, blocks, pdl_launch, grad, bank, snap, dva, beta, kpay,     \
                                             rec, out_eff, out_g, plane, ti, P, C, W, n, hd, nsl, slw, ctas, L, bpre,  \
                                             ssr, qs)
            if (!w64) {
                GX_PA2_GO(__nv_bfloat16, __nv_bfloat16, false);
            } else if (s16) {   // bf16 snapshot rows of a float32 bank (unquantized: quant == 0 when s16)
                if constexpr (Q == 0) {
                    gx_pa2_go<float, float, true, 0, 16, __nv_bfloat16>(D, nw_e, smem, blocks, pdl_launch, grad, bank,
                                                                        snap, dva, beta, kpay, rec, out_eff, out_g,
                                                                        plane, ti, P, C, W, n, hd, nsl, slw, ctas, L,
                                                                        bpre, ssr, qs);
                }
            } else if (!bf_bank) {
                GX_PA2_GO(float, float, true);
            } else if (g32) {
                GX_PA2_GO(float, __nv_bfloat16, true);
            } else {
                GX_PA2_GO(__nv_bfloat16, __nv_bfloat16, true);
            }
#undef GX_PA2_GO
        };
        if (quant == 1) {
            rows(std::integral_constant<int, 1>{});
        } else if (quant == 2) {
            rows(std::integral_constant<int, 2>{});
        } else {
            rows(std::integral_constant<int, 0>{});
        }
    }
#endif
}
}  // namespace

// dc_pa's exact signature (a drop-in for the backward's per-step call): a plain launch (mode 0).
void gx_pa2(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc) {
    gx_pa2_run<false>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti, P, C, W, L, nw, depth,
                      dbg, pipe, bpre, gdwc, false);
}

// The same with programmatic stream serialization (mode 1). Only for a caller whose previous kernel writes nothing
// this kernel reads before its wait (the backward: gx_gemm_cuda's GEMM).
void gx_pa2_pdl(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
                at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
                at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
                int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc) {
    gx_pa2_run<false>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti, P, C, W, L, nw, depth,
                      dbg, pipe, bpre, gdwc, true);
}

#if !defined(GX_PA2_TAILS_ONLY)
// The bf16 entry (bf16 bank, grad bank and snapshot rows in 16-byte pieces below 2^32 grad elements): plain (mode 0)
// and programmatic (mode 1) launches; the same kernels, geometry and arguments as gx_pa2 / gx_pa2_pdl for those rows.
void gx_pa2_bf(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc) {
    gx_pa2_run<true>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti, P, C, W, L, nw, depth,
                     dbg, pipe, bpre, gdwc, false);
}
void gx_pa2_bf_pdl(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
            at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
            at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
            int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc) {
    gx_pa2_run<true>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti, P, C, W, L, nw, depth,
                     dbg, pipe, bpre, gdwc, true);
}
#endif

// ---- the backward's C++ step loop (bwd_steps_cuda): gx_pa2 as a per-backward plan -------------------------------------
// gx_pa2_plan runs the per-step entry's own host path ONCE under plan capture at step ti_max -- gx_pa2_run<kbf> with
// pdl_launch: gx_pa2 (false, false), gx_pa2_pdl (false, true), gx_pa2_bf (true, false), gx_pa2_bf_pdl (true, true); the
// bf16 entries exist in the regular module only -- so every check runs (its step bounds cover every ti <= ti_max) and its
// launch is recorded, with the mode argument the PDL / plain branch passes. gx_pa2_plan_launch (a plain C++ entry point; address from gx_pa2_launch_addr) sets the step
// parameter ti and issues the recorded launch: with programmatic stream serialization when planned for gx_pa2_pdl, else
// plain. Its pdl argument is ignored: the launcher variant is fixed by the plan.
namespace {
constexpr uint64_t kPa2PlanMagic = 0x314e414c50324150ull;
constexpr int kPa2TiParam = 10;     // (grad, bank, snap, dva, beta, kpay, rec, out_eff, out_g, plane, ti, ...)
struct Pa2Plan {
    uint64_t magic;
    int32_t device;
    int32_t pdl_launch;
    int32_t kbf;
    int32_t pad;
    int64_t ti_max;
    sdm_plancap::Launch cap;
};

int64_t gx_pa2_plan_launch(const void* plan_bytes, int64_t nbytes, int64_t ti, int64_t pdl, void* stream) {
    (void)pdl;
    Pa2Plan pl;
    if (!sdm_plancap::from_bytes(pl, plan_bytes, nbytes, kPa2PlanMagic)) return -1;
    if (ti < 0 || ti > pl.ti_max) return -2;
    if (pl.cap.set == 0) return 0;                 // empty grid: gx_pa2_run launched nothing
    if (sdm_plancap::current_device() != pl.device) return -5;
    int* tip = sdm_plancap::param<int>(pl.cap, kPa2TiParam);
    if (tip == nullptr) return -7;
    *tip = static_cast<int>(ti);
    return sdm_plancap::launch(pl.cap, pl.pdl_launch != 0, static_cast<cudaStream_t>(stream));
}
}  // namespace

at::Tensor gx_pa2_plan(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
                       at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
                       at::Tensor out_g, int64_t ti_max, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
                       int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc, bool pdl_launch,
                       bool kbf) {
    TORCH_CHECK(ti_max >= 0 && ti_max < (int64_t(1) << 31), "gx_pa2_plan: ti_max");
    Pa2Plan pl;
    std::memset(&pl, 0, sizeof(pl));
    {
        sdm_plancap::Scope scope(&pl.cap);
#if defined(GX_PA2_TAILS_ONLY)
        TORCH_CHECK(!kbf, "gx_pa2_plan: the bf16 entry is the regular module's");
        gx_pa2_run<false>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti_max, P, C, W, L,
                          nw, depth, dbg, pipe, bpre, gdwc, pdl_launch);
#else
        if (kbf) {
            gx_pa2_run<true>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti_max, P, C, W, L,
                             nw, depth, dbg, pipe, bpre, gdwc, pdl_launch);
        } else {
            gx_pa2_run<false>(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti_max, P, C, W,
                              L, nw, depth, dbg, pipe, bpre, gdwc, pdl_launch);
        }
#endif
    }
    pl.magic = kPa2PlanMagic;
    pl.device = static_cast<int32_t>(bank.get_device());
    pl.pdl_launch = pdl_launch ? 1 : 0;
    pl.kbf = kbf ? 1 : 0;
    pl.ti_max = ti_max;
    TORCH_CHECK(pl.cap.set == 0 || sdm_plancap::param<int>(pl.cap, kPa2TiParam) != nullptr, "gx_pa2_plan: parameters");
    return sdm_plancap::to_tensor(pl);
}

int64_t gx_pa2_launch_addr() { return reinterpret_cast<int64_t>(&gx_pa2_plan_launch); }


// Tests (gx_pa2_cuda.plan() assumes it is 0): the static shared memory of the D 8 / 256-thread kernel of one row-type
// family and quantization mode on device dev.
int64_t gx_pa2_static_smem(bool bf_bank, bool g32, bool w64, int64_t dev, int64_t quant) {
    TORCH_CHECK((bf_bank || g32) && (w64 || (bf_bank && !g32)) && quant >= 0 && quant <= 2,
                "gx_pa2_static_smem: a row type, offset width and quantization mode gx_pa2 runs");
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev));
    const void* k8 = quant == 1 ? kern8_of<1>(w64, bf_bank, g32) : quant == 2 ? kern8_of<2>(w64, bf_bank, g32)
                                : kern8_of<0>(w64, bf_bank, g32);
    return static_cast<int64_t>(static_smem_of(k8, static_cast<int>(dev)));
}
""".replace("@ROW_HELPERS@\n", ROW_HELPERS.lstrip("\n"))

@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_gx_pa2",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("gx_pa2", "gx_pa2_pdl", "gx_pa2_bf", "gx_pa2_bf_pdl", "gx_pa2_static_smem", "gx_pa2_plan",
                   "gx_pa2_launch_addr"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo"),
    )


@functools.cache
def _tail_module():
    """Tails: the same source with only the unquantized tail instantiations (8-byte pieces of bf16 rows, PB 8; element
    pieces for hd % 4 != 0, PB 2), in their own module so the regular instantiations' SASS stays identical."""
    return load_cuda_extension(
        "sdm_gx_pa2_tail",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("gx_pa2", "gx_pa2_pdl", "gx_pa2_static_smem", "gx_pa2_plan", "gx_pa2_launch_addr"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo", "-DGX_PA2_TAILS_ONLY"),
    )


def needs_tail_module(bank, snap=None) -> bool:
    """Rows the launcher copies in element pieces (hd % 4 != 0) or, for bf16 rows, in 8-byte pieces: the launcher's pb rule."""
    hd = bank.shape[-1]
    nsl = -(-hd // 128)
    slw = 4 * (-(-(-(-hd // 4)) // nsl))
    bf_rows = bank.dtype == torch.bfloat16 or (snap is not None and snap.dtype == torch.bfloat16)
    return hd % 4 != 0 or (bf_rows and not (hd % 8 == 0 and slw % 8 == 0))


ROW_TYPES = ((torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.float32))
# quant_of() of bf16 snapshot rows on a float32 bank: read as unquantized bf16 rows (QUANT 0, float32 grad rows)
S16 = 3


def _regular(hd: int, bank_dtype) -> bool:
    """16-byte slice rows of equal slices: the shapes outside the tail classes."""
    nsl = -(-hd // 128)
    q = 8 if bank_dtype == torch.bfloat16 else 4
    return not (hd % q or hd % nsl or (hd // nsl) % q or 4 * (-(-(hd // 4) // nsl)) * nsl != hd)


def tail_supported(bank, chunk_size, quant="none") -> bool:
    """Tail hook (movers.tail_admits): Phase A of an unquantized call on ``bank`` (every grad dtype it may use) takes
    gx_pa2 for the call's head on every backend (rtx: chunks of <= 128 tokens). Shapes, dtypes and the device only."""
    if quant not in ("none", 0, "bf16") or not bank.is_cuda or bank.dim() != 2 or bank.dtype not in (torch.bfloat16, torch.float32):
        return False
    if quant == "bf16" and bank.dtype != torch.float32:   # bf16 snapshot rows: float32 banks (S16 rows)
        return False
    try:
        b = backend(bank.device)
    except RuntimeError:
        return False
    if not (b in ("modern", "a100") or (b == "rtx" and chunk_size <= 128)) or not (1 <= chunk_size <= 256):
        return False
    hd = bank.shape[-1]
    if not (1 <= hd <= 2 ** 20 and bank.shape[0] < 2 ** 31):     # hd bound: the forward's
        return False
    gdts = (torch.bfloat16, torch.float32) if bank.dtype == torch.bfloat16 else (torch.float32,)
    q = S16 if quant == "bf16" else 0
    return all(plan(types.SimpleNamespace(dtype=g, numel=bank.numel), bank, chunk_size, q) is not None for g in gdts)


def s16_admits(bank, C: int) -> bool:
    """snapshot_quant bf16 on a float32 bank (backward.supported): gx_pa2 takes Phase A of its bf16 compact rows (admits_rows,
    or the tail module's pieces: tail_supported)."""
    return bool(admits_rows(bank, torch.float32, C, S16) or tail_supported(bank, C, "bf16"))


def smem(C: int, nw: int, depth: int, row_bytes: int, quant: int = 0) -> int:
    """Dynamic shared memory of a launch (the launcher's formula; no static shared memory): token rows, beta, grad + snapshot
    rings [nw][depth][32] of row_bytes per lane-row, record tables [nw][64] int2 and, for quantized rows, the scale ring."""
    return C * 512 + ((C + 3) // 4) * 16 + nw * depth * 32 * row_bytes + nw * 512 + (nw * depth * 4 if quant else 0)


def plan(grad, bank, C: int, quant: int = 0):
    """(warps per CTA, ring depth) for chunks of C tokens on these rows, or None when one warp does not fit (footprint and
    class limits only): 16 warps where they fit and C > 128 or float32 grad rows need them (not float32 banks on H100 at
    C <= 128), else the most of 8 .. 1; depth 16 at C <= 64 on bf16 rows below 2^32 elements when two CTAs fit, else 8."""
    g32 = grad.dtype == torch.float32
    rb = (16 if g32 else 8) + (8 if quant == S16 else 4 if quant else 8 if bank.dtype == torch.bfloat16 else 16)
    w64 = not (bank.dtype == torch.bfloat16 and grad.dtype == torch.bfloat16) or grad.numel() >= 2 ** 32
    cap, per_sm = smem_cap(bank.device), smem_per_sm(bank.device)
    quant = 0 if quant == S16 else quant          # bf16 snapshot rows: unquantized rows (no scale ring), 8 bytes a lane-row
    nw = next((w for w in (8, 7, 6, 5, 4, 3, 2, 1) if smem(C, w, 8, rb, quant) <= cap), None)
    if nw is None:
        return None
    # float32 grad rows: 16 warps, except float32 banks on H100 (one 16-warp wave is slower there). Speed only.
    if (C > 128 or (g32 and 2 * (smem(C, nw, 8, rb, quant) + RESERVED_SMEM) > per_sm
                    and (bank.dtype == torch.bfloat16 or capability(bank.device) != (9, 0)))) \
            and smem(C, 16, 8, rb, quant) <= cap:
        nw = 16
    two16 = 2 * (smem(C, nw, 16, rb, quant) + RESERVED_SMEM) <= per_sm
    return nw, (16 if C <= 64 and not w64 and not quant and two16 else 8)


def quant_of(bank, snap) -> int:
    """0: unquantized compact rows; 1 / 2: fp8 / int4 rows; S16: bf16 rows of a float32 bank; -1: neither."""
    if snap.dtype == bank.dtype:
        return 0
    if bank.dtype == torch.float32 and snap.dtype == torch.bfloat16:
        return S16
    if snap.dtype != torch.uint8 or snap.dim() != 2 or bank.dim() != 2:
        return -1
    hd = bank.shape[-1]
    return 1 if snap.shape[1] == hd else 2 if 2 * snap.shape[1] == hd else -1


def admits_rows(bank, grad_dtype, C: int, quant: int = 0, nw: int = 8) -> bool:
    """The shape / dtype / backend part of supported() (no snapshot tensor yet: backward.supported, for the forward): row types
    bf16 / bf16, bf16 / f32, f32 / f32; 16-byte slice rows; int4 rows of whole 8-column groups; bank rows < 2^31; and a plan."""
    if not (1 <= C <= 256 and bank.is_cuda and bank.dim() == 2 and (bank.dtype, grad_dtype) in ROW_TYPES):
        return False
    if not cuda_backend(bank.device):
        return False
    if quant == S16 and not (bank.dtype == torch.float32 and grad_dtype == torch.float32):
        return False                              # bf16 snapshot rows: float32 bank and grad rows only
    hd = bank.shape[-1]
    nsl = -(-hd // 128)
    q = 8 if bank.dtype == torch.bfloat16 or quant in (2, S16) else 4
    if hd % q or hd % nsl or (hd // nsl) % q or 4 * (-(-(hd // 4) // nsl)) * nsl != hd:
        return False
    if bank.shape[0] >= 2 ** 31:
        return False
    grad_rows = types.SimpleNamespace(dtype=grad_dtype, numel=bank.numel)   # plan() reads the grad rows' dtype and size
    return plan(grad_rows, bank, C, quant) is not None


def supported(grad, bank, snap, C: int, dbg: int, depth: int, nw: int, gdc: bool, scales=None) -> bool:
    """The backward's choice, once per backward: True -> gx_pa2 runs every step instead of dc_pa (same bits): admits_rows for
    the grad bank's dtype plus the operands themselves (quantized rows: their f32 row scales in dc_pa's gdwc slot;
    unquantized rows also dc_pa3's per-entry snapshot and the tail shapes)."""
    if dbg not in (256 | 512 | 1024, 512) or depth != 8 or gdc or not (1 <= nw <= 8):
        return False
    quant = quant_of(bank, snap)
    if quant < 0 or grad.shape != bank.shape or (quant and dbg != (256 | 512 | 1024)):
        return False
    if quant in (1, 2) and not (scales is not None and scales.dtype == torch.float32 and scales.dim() == 1
                      and scales.numel() == snap.shape[0] and scales.is_contiguous() and scales.device == bank.device
                      and snap.is_contiguous() and snap.device == bank.device):
        return False
    if quant == 0 and bank.dim() == 2 and not _regular(bank.shape[-1], bank.dtype):
        if not (bank.is_cuda and (bank.dtype, grad.dtype) in ROW_TYPES and 1 <= C <= 256 and cuda_backend(bank.device)
                and tail_supported(bank, C) and plan(grad, bank, C) is not None):
            return False
    elif quant == S16 and bank.dim() == 2 and not _regular(bank.shape[-1], torch.bfloat16):
        # bf16 snapshot slices that are not 16-byte rows: the tail module
        if not (bank.is_cuda and bank.dtype == torch.float32 and grad.dtype == torch.float32 and 1 <= C <= 256
                and cuda_backend(bank.device) and tail_supported(bank, C, "bf16") and plan(grad, bank, C, S16) is not None):
            return False
    elif not admits_rows(bank, grad.dtype, C, quant, nw):
        return False
    if any(t.data_ptr() % 16 for t in (grad, bank, snap)):
        return False
    if quant == S16 and not (snap.dim() >= 2 and snap.shape[-1] == bank.shape[-1] and snap.is_contiguous()
                             and snap.device == bank.device):   # the forward's every-step [N * C, W, hd] compact rows
        return False
    return True   # any snapshot row count: chunk-local int32 rows from a 64-bit chunk base (the kernel's ssl / qsl)

def _wave_ctas(device=None) -> int:
    """Resident 8-warp CTAs of one wave: two per SM (GB300: 304)."""
    return 2 * num_sms(device)


def tile_len(C: int, W: int, hd: int, P: int, L_dc: int, device=None) -> int:
    """The tile length the backward passes to this kernel, from shapes and the device's SM count (outputs do not depend
    on it). pa_dc_cuda's tile_len (L_dc) is kept unless its grid would leave SMs idle: with 8-warp CTAs (C <= 128), when
    P * slices * ceil(ceil(C * W / L_dc) / 8) CTAs are fewer than one wave of _wave_ctas(device) (2 resident CTAs per
    SM), the step gets that many CTAs instead: ctas = wave // (P * slices) per (chunk, slice), L = ceil(C * W /
    (8 * ctas)), at least 16."""
    wave = _wave_ctas(device)
    n = int(C) * int(W)
    if C > 128 or n <= 0 or P < 1 or L_dc < 1 or hd <= 0:   # hd <= 0 guarded
        return L_dc
    nsl = -(-int(hd) // 128)
    if P * nsl * (-(-(-(-n // L_dc)) // 8)) >= wave:
        return L_dc
    ctas = max(1, wave // (P * nsl))
    return min(L_dc, max(16, -(-n // (8 * ctas))))   # never above L_dc (<= C * W)


def launcher(pdl: bool = False, bank=None, grad=None, snap=None):
    """The per-step PA callable with dc_pa's signature (the C++ entry itself). pdl=True: a programmatic launch, only after a
    kernel that writes nothing it reads before its griddepcontrol.wait (gx_gemm_cuda's GEMM). bank / grad / snap pick the
    module and entry once per backward (bf16 rows in 16-byte pieces below 2^32 grad elements: gx_pa2_bf / gx_pa2_bf_pdl;
    anything else the generic entry: the same kernels and bits)."""
    tail = bank is not None and needs_tail_module(bank, snap)
    mod = _tail_module() if tail else _module()
    if (not tail and bank is not None and grad is not None and snap is not None and bank.dtype == torch.bfloat16
            and grad.dtype == torch.bfloat16 and snap.dtype == torch.bfloat16 and grad.numel() < 2 ** 32):
        return mod.gx_pa2_bf_pdl if pdl else mod.gx_pa2_bf
    return mod.gx_pa2_pdl if pdl else mod.gx_pa2


_ENTRIES = (("gx_pa2", False, False), ("gx_pa2_pdl", True, False), ("gx_pa2_bf", False, True),
            ("gx_pa2_bf_pdl", True, True))


def _entry(fn):
    """(module, pdl_launch, kbf) of a launcher() result, or None (only modules already built are inspected)."""
    for load in (_module, _tail_module):
        if not load.cache_info().currsize:
            continue
        m = load()
        for name, pdl_launch, kbf in _ENTRIES:
            if getattr(m, name, None) is fn:
                return m, pdl_launch, kbf
    return None


def is_launcher(fn) -> bool:
    """Whether `fn` is one of launcher()'s per-step entries (step_plan can plan it)."""
    return _entry(fn) is not None


def step_plan(launch, grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti_max, P, C, W, L, nw,
              depth, dbg, pipe, bpre, gdwc):
    """(plan, launch address) of the per-step PA entry `launch` for steps 0 .. ti_max of the backward's C++ step loop
    (bwd_steps_cuda): gx_pa2_plan in the entry's own module, on the entry's own host path."""
    m, pdl_launch, kbf = _entry(launch)
    p = m.gx_pa2_plan(grad, bank, snap, int(snap_step_rows), dva, beta, kpay, rec, out_eff, out_g, int(ti_max), int(P),
                      int(C), int(W), int(L), int(nw), int(depth), int(dbg), bool(pipe), bpre, gdwc, pdl_launch, kbf)
    return p, int(m.gx_pa2_launch_addr())
