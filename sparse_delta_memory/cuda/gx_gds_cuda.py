# Copyright (c) Meta Platforms, Inc. and affiliates.
"""gx_gds_cuda: the read-side backward mover (GDS) of the deterministic backward as ONE sorted walk per tile, bitwise
equal to mover_da2_cuda's GDS.

What it computes, per chunk of one step, over Phase 1's sorted read records (qrec: {spay_q element, absolute slot,
scale = __fmul_rn(f32(q_val), expf(log_q)), 0}; one slot group = consecutive positions FIRST..LAST):

  grad[slot]  = round(gb + acc),  acc = fmaf(sc_m, dy[t_m], acc) over the group's members in sorted order from 0,
                gb = the grad row before this launch (one rounding: mover_da2_cuda's read-modify-write update);
  d_q_w[t, r] = <dy[t], bank[slot]>: per lane the chain __fmaf_rn(x.w, m.w, __fmaf_rn(x.z, m.z, __fmaf_rn(x.y, m.y,
                __fmul_rn(x.x, m.x)))) over its 4 columns, summed over the lanes in the xor-16, 8, 4, 2, 1 order
                (mover_da2_cuda's 8-entry reduce-scatter); column slices of a wide head go to planes summed in slice
                order at step 0.

These are mover_da2_cuda's formulas, so the outputs are bitwise its GDS's. What changes is the row supply; the walk
has no shared-memory rings and no waits:
  * at tile start every valid group's bank row (needed from its FIRST) and grad row (needed at its LAST) of the first
    two record batches is prefetched to L2 (prefetch.global.L2 per 128-byte line, the LSU path: no TMA engine
    serialization), then the rows of batch bi + 1 at batch bi;
  * the rows enter registers 8 positions ahead: position j of a sub-batch is refilled with position j of the next
    sub-batch right after it is consumed (16 row registers per lane);
  * per-entry dots are computed from the group's bank row held in registers (one bank row load per GROUP instead of
    one per read entry), so there is one role and one launch per step.
Tiles: Phase 1's tiles_q (L = 128, forward-snapped) or mover_da2_cuda's build_tiles; every group lies in one tile, so
outputs do not depend on the tiling, the warps per CTA or the prefetch distance. Records a provider built at factory time
are read as tensor records (the step's GDS may then launch with PDL in captured backwards too).
Scope: bf16 bank with bf16 or f32 grad rows, f32 bank with f32 grad rows; every row type but bf16 / bf16, and bf16 banks
of >= 2^32 elements, take the 64-bit-offset instantiations. Shapes outside admits() take mover_da2_cuda's make_gds on the
modern backend; elsewhere the caller routes them (movers.gds_admits). On a100 / rtx (the portable side of sdm_copy.cuh)
the token rows are staged with cooperative 16-byte cp.async copies and a CTA barrier, without programmatic launch;
everything else is the same code, so the outputs are the modern kernel's, bit for bit."""

import functools

import torch

from ._build import load_cuda_extension
from ._device import headers, num_sms, prev_pow2
# device-class reads memoized per device ordinal and forced row (values = _device's)
from ._devmemo import backend, capability, cuda_backend, pdl_ok as device_pdl_ok
from . import mover_da2_cuda as da2

# Device helpers of the movers' row types (Row4, cvt4, st4 / st_raw, zero rows, row_off / rec_val / rec_row), shared by
# gx_gds_cuda, gx_pa2_cuda and mover_fx_cuda (inserted into each source).
ROW_HELPERS = r"""
template <typename E> struct Row4;
template <> struct Row4<__nv_bfloat16> { using T = uint2; };     // 4 bf16
template <> struct Row4<float> { using T = float4; };            // 4 f32
__device__ __forceinline__ float4 cvt4(uint2 v) {
    return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xFFFF0000u),
                       __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xFFFF0000u));
}
__device__ __forceinline__ float4 cvt4(float4 v) { return v; }
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void st4(__nv_bfloat16* p, float4 v) {
    *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf2(v.x, v.y), pack_bf2(v.z, v.w));
}
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
template <typename E> __device__ __forceinline__ void st_raw(E* p, typename Row4<E>::T v) {
    *reinterpret_cast<typename Row4<E>::T*>(p) = v;
}
template <typename T> __device__ __forceinline__ T make_zero_row();
template <> __device__ __forceinline__ uint2 make_zero_row<uint2>() { return make_uint2(0u, 0u); }
template <> __device__ __forceinline__ float4 make_zero_row<float4>() { return make_float4(0.f, 0.f, 0.f, 0.f); }
template <bool W64> __device__ __forceinline__ auto row_off(uint32_t row, int hd) {
    if constexpr (W64) return static_cast<int64_t>(row) * hd; else return row * static_cast<uint32_t>(hd);
}
template <bool W64> __device__ __forceinline__ uint32_t rec_val(uint32_t slot, int hd) {
    if constexpr (W64) return slot; else return slot * static_cast<uint32_t>(hd);
}
template <bool W64> __device__ __forceinline__ uint64_t rec_row(uint32_t v, int hd) {
    if constexpr (W64) return static_cast<uint64_t>(v) * static_cast<uint32_t>(hd); else return v;
}
// Quantized snapshot rows: QUANT 1 = fp8 e4m3 bytes, 2 = int4 signed nibbles (element 2i in the high
// nibble of byte i), one f32 scale per row; decoded x' = E(f32(q) * f32(E(scale))) (phase2_kernels._snapshot_tile).
// Integer decode only (no fp8 intrinsics): the same code on every architecture.
__device__ __forceinline__ float e4m3_f32(uint32_t b) {
    const uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : static_cast<float>(m) * 0x1p-9f;
    if ((b & 0x7Fu) == 0x7Fu) v = __uint_as_float(0x7FC00000u);   // S.1111.111: NaN (e4m3 has no infinity)
    return (b & 0x80u) ? -v : v;
}
__device__ __forceinline__ float nib_f32(uint32_t n) { return static_cast<float>(static_cast<int>(n ^ 8u) - 8); }
// A lane's 4 columns of a quantized row: w = the payload word copied for it (fp8: its 4 bytes; int4: the 4-byte word
// of its lane pair, its 2 bytes at byte 2 * (lane & 1)); raw = x' in the bank dtype (what the backward restores).
template <int QUANT, typename E>
__device__ __forceinline__ float4 dec4(uint32_t w, int lane, float scale, typename Row4<E>::T& raw) {
    float4 q;
    if constexpr (QUANT == 1) {
        q = make_float4(e4m3_f32(w & 0xFFu), e4m3_f32((w >> 8) & 0xFFu), e4m3_f32((w >> 16) & 0xFFu), e4m3_f32(w >> 24));
    } else {
        const uint32_t h = (lane & 1) ? (w >> 16) : (w & 0xFFFFu);
        q = make_float4(nib_f32((h >> 4) & 0xFu), nib_f32(h & 0xFu), nib_f32((h >> 12) & 0xFu), nib_f32((h >> 8) & 0xFu));
    }
    if constexpr (std::is_same<E, __nv_bfloat16>::value) {
        const float s = __bfloat162float(__float2bfloat16_rn(scale));
        raw = make_uint2(pack_bf2(__fmul_rn(q.x, s), __fmul_rn(q.y, s)), pack_bf2(__fmul_rn(q.z, s), __fmul_rn(q.w, s)));
        return cvt4(raw);
    } else {
        raw = make_float4(__fmul_rn(q.x, scale), __fmul_rn(q.y, scale), __fmul_rn(q.z, scale), __fmul_rn(q.w, scale));
        return raw;
    }
}
"""

# fp8 / int4 snapshot rows (briefs/w3_interface.md), shared by the producer (mover_fx_cuda's quantized DW) and the
# consumer's decode (needs <cuda_fp8.h>). One row = the hd columns of one pre-write bank row, in f32; the arithmetic of
# 2.0.0's per-entry quantizers (da2_quant_kernel, phase2_kernels' _store_quantized): amax = fmaxf over |x| from +0 (NaN ignored, inf kept), scale = f32(bf16(fmaxf(amax * f32(1 / qmax), 1e-12))),
# q = bf16(x / scale) (IEEE division), then e4m3 RN satfinite (Q 1) or nibble = clamp(rint(q), -8, 7) & 0xF with element
# 2i in the high nibble of byte i (Q 2).
QUANT_HELPERS = r"""
__device__ __forceinline__ float q_amax4(float4 x) {
    return fmaxf(fmaxf(fmaxf(fmaxf(0.f, fabsf(x.x)), fabsf(x.y)), fabsf(x.z)), fabsf(x.w));
}
// = q_amax4(cvt4(v)) for 4 bf16 values, on packed pairs (__hmax2: a NaN operand yields the other; both NaN: NaN, which
// the final fmaxf from +0 ignores)
__device__ __forceinline__ float q_amax_bf4(uint2 v) {
    const __nv_bfloat162 m = __hmax2(__habs2(*reinterpret_cast<const __nv_bfloat162*>(&v.x)),
                                     __habs2(*reinterpret_cast<const __nv_bfloat162*>(&v.y)));
    return fmaxf(fmaxf(0.f, __low2float(m)), __high2float(m));
}
// max over the warp of non-negative, non-NaN floats (their bits order as unsigned integers)
__device__ __forceinline__ float q_warp_max(float a) {
    return __uint_as_float(__reduce_max_sync(0xFFFFFFFFu, __float_as_uint(a)));
}
template <int Q> __device__ __forceinline__ float q_scale(float amax) {
    const float qinv = (Q == 1) ? static_cast<float>(1.0 / 448.0) : static_cast<float>(1.0 / 7.0);
    return __bfloat162float(__float2bfloat16_rn(fmaxf(amax * qinv, 1e-12f)));
}
// q = bf16(x / sc). EXACT: __fdiv_rn. Else (x a bf16 value of a row whose scale is sc; y = __frcp_rn(sc)): the product
// with one FMA correction, signed zeros kept; equal through the final byte / nibble to the EXACT form for every such
// (x, sc) (exhaustive proof: tests/test_quant_fx.py::test_quant_division_exhaustive).
template <bool EXACT> __device__ __forceinline__ float q_div(float x, float sc, float y) {
    float q;
    if constexpr (EXACT) {
        q = __fdiv_rn(x, sc);
    } else {
        q = x * y;
        if (q != 0.f) q = fmaf(fmaf(-sc, q, x), y, q);
    }
    return __bfloat162float(__float2bfloat16_rn(q));
}
__device__ __forceinline__ uint32_t q_nib(float q) {
    return static_cast<uint32_t>(static_cast<int>(fmaxf(fminf(rintf(q), 7.f), -8.f))) & 0xFu;
}
// Columns col .. col + 3 (col % 4 == 0) of one quantized row qrow (hd bytes for fp8, hd / 2 for int4).
template <int Q, bool EXACT> __device__ __forceinline__ void q_store4(uint8_t* qrow, int64_t col, float4 x, float sc,
                                                                      float y) {
    const float a = q_div<EXACT>(x.x, sc, y), b = q_div<EXACT>(x.y, sc, y);
    const float c = q_div<EXACT>(x.z, sc, y), d = q_div<EXACT>(x.w, sc, y);
    if constexpr (Q == 1) {   // e4m3 pairs: the low byte holds the first column
        *reinterpret_cast<uint32_t*>(qrow + col) =
            static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3)) |
            (static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3)) << 16);
    } else {
        *reinterpret_cast<uint16_t*>(qrow + col / 2) =
            static_cast<uint16_t>((q_nib(a) << 4) | q_nib(b) | (q_nib(c) << 12) | (q_nib(d) << 8));
    }
}
"""

_CPP = r"""
void gx_gds(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor rec, at::Tensor spay,
            at::Tensor tb, at::Tensor dots, int64_t ti, int64_t P, int64_t C, int64_t R, int64_t K, int64_t nw,
            int64_t pdl);
void gx_gds_bf(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor rec, at::Tensor spay,
            at::Tensor tb, at::Tensor dots, int64_t ti, int64_t P, int64_t C, int64_t R, int64_t K, int64_t nw,
            int64_t pdl);
at::Tensor gx_gds_plan(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor rec,
                       at::Tensor spay, at::Tensor tb, at::Tensor dots, int64_t ti_max, int64_t P, int64_t C, int64_t R,
                       int64_t K, int64_t nw, bool kbf);
int64_t gx_gds_launch_addr();
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
constexpr unsigned kFull = 0xFFFFFFFFu;

#define GX_MINB 2           // launch-bounds min resident CTAs per SM
// Row loads of data the previous kernel writes (bank / grad rows): coherent L2 loads (ld.global.cg), so
// nothing depends on the non-coherent read-only path across the programmatic edge.
#define GX_LD(T, p) __ldcg(reinterpret_cast<const T*>(p))

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// The portable side (SDM_PORTABLE_ARCH, sdm_copy.cuh: sm_80 / 86 / 89 / 120 ...) stages the token rows with sdmcp's
// cooperative cp.async copies and publishes them with a CTA barrier (stage_tok_rows_spread, mbar_wait_dep): no barrier
// object, no griddepcontrol (GX_SM90_OR_NONE: nothing; the host never launches this kernel programmatically there) and
// no bulk copy (GX_SM90_ONLY traps there: never reached).
#ifdef SDM_PORTABLE_ARCH
#define GX_PRE_SM90 1
#define GX_SM90_ONLY(...) __trap()
#define GX_SM90_OR_NONE(...)
#else
#define GX_SM90_ONLY(...) __VA_ARGS__
#define GX_SM90_OR_NONE(...) __VA_ARGS__
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    GX_SM90_OR_NONE(asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count)
                              : "memory"));
}
__device__ __forceinline__ void fence_mbar_init() {
    GX_SM90_OR_NONE(asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"));
}
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* bar, uint32_t bytes) {
    GX_SM90_OR_NONE(asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
                              "r"(bytes) : "memory"));
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    GX_SM90_ONLY(asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory"));
}
// Spins until the phase of parity `parity` of `bar` completed; returns 0. No memory clobber: the caller adds the
// result to the address of every read that must follow the wait (as in mover_da2_cuda). Portable side: every thread
// waits for its own token-row copies, then the CTA barrier publishes them (every thread of the CTA reaches both calls).
__device__ __forceinline__ uint32_t mbar_wait_dep(uint64_t* bar, uint32_t parity) {
#ifdef GX_PRE_SM90
    (void)bar;
    (void)parity;
    sdmcp::publish_cta();
    return 0u;
#else
    uint32_t ok;
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " GX_WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " @!p bra GX_WAIT;\n"
        " mov.u32 %0, 0;\n"
        "}\n"
        : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity));
    return ok;
#endif
}
// L2 prefetch of [p, p + bytes): one prefetch per 128-byte line (a performance hint only, no semantic effect).
__device__ __forceinline__ void prefetch_l2(const void* p, int bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uintptr_t end = a + static_cast<uintptr_t>(bytes);
    for (uintptr_t q = a & ~static_cast<uintptr_t>(127); q < end; q += 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(q));
    }
}

@ROW_HELPERS@

// Read records, two layouts with the same content (Phase 1's records or px_qrec_cuda's per-step builder):
// REC 0: qrec  uint4 {spay_q element, absolute slot, scale bits, 0} per sorted read position;
// REC 1: rec8  uint2 {absolute slot, scale bits} + the spay_q element (int32) of the same position.
// Both are returned as {pay, slot, scale, 0}; past the tile end: a padding group of its own.
template <int REC> struct RecSrc;
template <> struct RecSrc<0> {
    const uint4* r;
    const int32_t* pay;   // unused
    __device__ __forceinline__ uint4 ld(int pos, int e) const {
        if (pos < e) return __ldg(r + pos);
        return make_uint4(kFirst | kLast | kInvalid, 0u, 0u, 0u);
    }
};
template <> struct RecSrc<1> {
    const uint2* r;
    const int32_t* pay;
    __device__ __forceinline__ uint4 ld(int pos, int e) const {
        if (pos < e) {
            const uint2 v = __ldg(r + pos);
            return make_uint4(static_cast<uint32_t>(__ldg(pay + pos)), v.x, v.y, 0u);
        }
        return make_uint4(kFirst | kLast | kInvalid, 0u, 0u, 0u);
    }
};

// The per-row copies dealt round-robin over the warps (row t by warp t % NW): every warp issues ~C / NW of them
// (each bulk copy is a uniform-datapath instruction, serialized over the active lanes of a warp).
// Portable side: every thread of the CTA copies its share of the C rows (SW floats each, row stride hd: 16-byte
// pieces, aligned by the host checks) with cp.async and commits its group; mbar_wait_dep publishes them.
__device__ __forceinline__ void stage_tok_rows_spread(float* tok_s, const float* rows, int C, int SW, int hd,
                                                      uint64_t* tok_bar, int warp, int lane, int nw) {
#ifdef GX_PRE_SM90
    (void)tok_bar;
    (void)warp;
    (void)lane;
    sdmcp::copy_rows16(tok_s, SW * 4, rows, static_cast<int64_t>(hd) * 4, C, SW * 4, static_cast<int>(threadIdx.x),
                       nw * 32);
    sdmcp::commit();
#else
    if (threadIdx.x == 0) mbar_arrive_tx(tok_bar, static_cast<uint32_t>(C * SW * 4));
    if (SW == hd) {
        if (threadIdx.x == 0) bulk_g2s(tok_s, rows, static_cast<uint32_t>(C * SW * 4), tok_bar);
    } else {
        for (int t = warp + nw * lane; t < C; t += nw * 32) {
            bulk_g2s(tok_s + t * SW, rows + static_cast<int64_t>(t) * hd, static_cast<uint32_t>(SW * 4), tok_bar);
        }
    }
#endif
}

// Tails (dc_pa3's slices, the last one narrower): the slice's wl valid columns of each token row (wl % 4 == 0),
// into rows of SW floats (the columns past wl are never read: lanes past the slice read lane 0's columns).
// Portable side: sdmcp's cooperative 16-byte cp.async copies of the wl valid columns (wl % 4 == 0: 16-byte pieces at
// 16-byte aligned row starts, hd % 4 == 0), committed per thread and published by mbar_wait_dep, as the regular staging.
__device__ __forceinline__ void stage_tok_rows_tail(float* tok_s, const float* rows, int C, int SW, int wl, int hd,
                                                    uint64_t* tok_bar, int warp, int lane, int nw) {
#ifdef GX_PRE_SM90
    (void)tok_bar;
    (void)warp;
    (void)lane;
    sdmcp::copy_rows16(tok_s, SW * 4, rows, static_cast<int64_t>(hd) * 4, C, wl * 4, static_cast<int>(threadIdx.x),
                       nw * 32);
    sdmcp::commit();
#else
    if (threadIdx.x == 0) mbar_arrive_tx(tok_bar, static_cast<uint32_t>(C * wl * 4));
    for (int t = warp + nw * lane; t < C; t += nw * 32) {
        bulk_g2s(tok_s + t * SW, rows + static_cast<int64_t>(t) * hd, static_cast<uint32_t>(wl * 4), tok_bar);
    }
#endif
}

// Tails with hd % 4 != 0 (MODE bit 16; rows aligned to 4 bytes only): the slice's wl valid columns of each token row
// by plain loads, the rest of the row's SW columns zeroed (a lane's last column group may be partial), one arrival.
__device__ __forceinline__ void stage_tok_rows_elem(float* tok_s, const float* rows, int C, int SW, int wl, int hd,
                                                    uint64_t* tok_bar) {
    for (int i = threadIdx.x; i < C * SW; i += blockDim.x) {
        const int t = i / SW, c = i - t * SW;
        tok_s[i] = c < wl ? __ldg(rows + static_cast<int64_t>(t) * hd + c) : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) mbar_arrive_tx(tok_bar, 0u);
}
// MODE bit 16: a lane's 4 row columns as elements, the first nv of them (zeros past them); stores of the first nv.
__device__ __forceinline__ uint2 ld_elems_row(const __nv_bfloat16* p, int nv) {
    const unsigned short* q = reinterpret_cast<const unsigned short*>(p);
    uint32_t w0 = 0u, w1 = 0u;
    if (nv > 0) w0 = static_cast<uint32_t>(__ldcg(q));
    if (nv > 1) w0 |= static_cast<uint32_t>(__ldcg(q + 1)) << 16;
    if (nv > 2) w1 = static_cast<uint32_t>(__ldcg(q + 2));
    if (nv > 3) w1 |= static_cast<uint32_t>(__ldcg(q + 3)) << 16;
    return make_uint2(w0, w1);
}
__device__ __forceinline__ float4 ld_elems_row(const float* p, int nv) {
    return make_float4(nv > 0 ? __ldcg(p) : 0.f, nv > 1 ? __ldcg(p + 1) : 0.f, nv > 2 ? __ldcg(p + 2) : 0.f,
                       nv > 3 ? __ldcg(p + 3) : 0.f);
}
__device__ __forceinline__ void st_elems_row(__nv_bfloat16* p, float4 v, int nv) {   // st4's rounding per element
    if (nv > 0) p[0] = __float2bfloat16_rn(v.x);
    if (nv > 1) p[1] = __float2bfloat16_rn(v.y);
    if (nv > 2) p[2] = __float2bfloat16_rn(v.z);
    if (nv > 3) p[3] = __float2bfloat16_rn(v.w);
}
__device__ __forceinline__ void st_elems_row(float* p, float4 v, int nv) {
    if (nv > 0) p[0] = v.x;
    if (nv > 1) p[1] = v.y;
    if (nv > 2) p[2] = v.z;
    if (nv > 3) p[3] = v.w;
}

struct Geo {
    int P, C, n, hd, NSL, SW, K, TU, tok_bytes;
};

// Grid: P * NSL * K CTAs of NW warps (CTA b: cr = b % K, slice = b / K % NSL, p = rest); warp w walks tile cr * NW + w
// of chunk ti * P + p. Shared memory: token rows [C][SW] f32 | per-warp record batches [NW][2][32] uint4 | tok_bar.
// Min resident CTAs per SM for the launch bounds: GX_MINB CTAs of 8 warps, i.e. 8 * GX_MINB warps per SM whatever NW.
// Clamped to 1 .. 32 / nw CTAs (<= 1024 threads per SM keeps >= 64 registers per thread: the row pipeline's 32 alone).
constexpr int gx_minb(int nw) {
    return (8 * GX_MINB) / nw < 1 ? 1 : ((8 * GX_MINB) / nw > 32 / nw ? 32 / nw : (8 * GX_MINB) / nw);
}

// rec: the records (REC layout) of chunks rec_chunk0 .. (all chunks: 0; one step's P chunks: ti * P), spay: spay_q of
// all chunks (REC 1 only).
// MODE = role | 4 * W64 | 8 * TAIL. Role 0 = the full walk (grad RMW + dots), the only role compiled; kDot / kRmw keep
// the two roles' code apart. W64 (MODE bit 4): 64-bit row element offsets (banks of >= 2^32 elements, and every
// float32 row type): the record batch keeps the slot and each row access forms uint64(slot) * uint32(hd),
// mover_da2_cuda's GDS arithmetic; without it (bf16 bank + bf16 grad) the batch keeps the 32-bit slot * hd. TAIL
// (MODE bit 8): dc_pa3's slices of SW = 4 ceil(ceil(hd / 4) / NSL) columns, the last one narrower (uneven
// slices); the same arithmetic per column and lane. ELEM (MODE bit 16, with 8; hd % 4 != 0): rows aligned to their
// element only, so token rows, row loads and grad stores go by elements (zeros past the head's last column).
template <typename E, typename G, int NW, int REC, int MODE>
__global__ void __launch_bounds__(NW * 32, gx_minb(NW)) gx_gds_kernel(
    const E* __restrict__ bank, G* __restrict__ grad, const float* __restrict__ gr, int64_t gr_step_rows,
    const void* __restrict__ rec, const int32_t* __restrict__ spay, int64_t rec_chunk0,
    const int32_t* __restrict__ tb, float* __restrict__ dots, int64_t plane, int ti, Geo g, int R) {
    static_assert(MODE >= 0 && (MODE & ~24) <= 6 && (MODE & 3) <= 2 && ((MODE & 16) == 0 || (MODE & 8) != 0),
                  "MODE: role (0 full, 1 grad RMW only, 2 dots only) | 4 (64-bit row offsets) | 8 (tail slices) | 16 "
                  "(element pieces, with 8)");
    constexpr bool kW64 = (MODE & 4) != 0;
    constexpr bool kTail = (MODE & 8) != 0;
    constexpr bool kElem = (MODE & 16) != 0;
    static_assert(kW64 || (std::is_same<E, __nv_bfloat16>::value && std::is_same<G, __nv_bfloat16>::value),
                  "float32 rows use 64-bit row offsets");
    constexpr bool kDot = (MODE & 3) != 1;   // bank rows + d_q_w dots
    constexpr bool kRmw = (MODE & 3) != 2;   // grad rows + the single-writer read-modify-write
    using RowT = typename Row4<E>::T;
    using RowG = typename Row4<G>::T;
    extern __shared__ __align__(128) unsigned char smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned ub = blockIdx.x;   // unsigned index math: blocks < 2^31 and K, NSL >= 1 (host checks), same values
    const int cr = static_cast<int>(ub % static_cast<unsigned>(g.K));
    ub /= static_cast<unsigned>(g.K);
    const int sl = static_cast<int>(ub % static_cast<unsigned>(g.NSL));
    const int p = static_cast<int>(ub / static_cast<unsigned>(g.NSL));
    const int64_t chunk = static_cast<int64_t>(ti) * g.P + p;
    const int SW = g.SW, C = g.C, hd = g.hd;
    float* tok_s = reinterpret_cast<float*>(smem);
    uint4* rec_s = reinterpret_cast<uint4*>(smem + g.tok_bytes) + warp * 64;     // [2][32]: current | next batch
    uint64_t* tok_bar = reinterpret_cast<uint64_t*>(smem + g.tok_bytes + NW * 64 * sizeof(uint4));
    const int32_t* tbc = tb + chunk * (g.TU + 1);
    const int tile = cr * NW + warp;
    const int s = __ldg(tbc + tile), e = __ldg(tbc + tile + 1);   // read-only input: in flight across the barrier
    if (threadIdx.x == 0) mbar_init(tok_bar, 1);
    fence_mbar_init();
    __syncthreads();
    if constexpr (kElem) {
        stage_tok_rows_elem(tok_s, gr + (static_cast<int64_t>(ti) * gr_step_rows + static_cast<int64_t>(p) * C) * hd +
                                       static_cast<int64_t>(sl) * SW, C, SW, min(SW, hd - sl * SW), hd, tok_bar);
    } else if constexpr (kTail) {
        stage_tok_rows_tail(tok_s, gr + (static_cast<int64_t>(ti) * gr_step_rows + static_cast<int64_t>(p) * C) * hd +
                                       static_cast<int64_t>(sl) * SW, C, SW, min(SW, hd - sl * SW), hd, tok_bar, warp,
                            lane, NW);
    } else {
        stage_tok_rows_spread(tok_s, gr + (static_cast<int64_t>(ti) * gr_step_rows + static_cast<int64_t>(p) * C) * hd +
                                         static_cast<int64_t>(sl) * SW, C, SW, hd, tok_bar, warp, lane, NW);
    }
    RecSrc<REC> kr;
    if constexpr (REC == 0) {
        kr.r = reinterpret_cast<const uint4*>(rec) + (chunk - rec_chunk0) * g.n;
        kr.pay = nullptr;
    } else {
        kr.r = reinterpret_cast<const uint2*>(rec) + (chunk - rec_chunk0) * g.n;
        kr.pay = spay + chunk * g.n;
    }
    const int c4 = 4 * lane;
    // Lanes past the slice's columns (hd 120 / 960: lanes 30, 31; hd 64: lanes 16-31) mirror lane 0's columns for
    // their row LOADS (same addresses as lane 0's loads in the same instruction: unguarded, no extra traffic); their
    // grad stores stay predicated off by `on` (one writer per address) and their dot partials are zeroed.
    const bool on = c4 < SW && sl * SW + c4 < hd;
    const int coff = on ? c4 : 0;
    // ELEM: valid columns at the lane's address (off lanes mirror lane 0's; the head's last group may be partial)
    const int nvl = kElem ? min(4, hd - (sl * SW + coff)) : 4;
    const E* const bank_col = bank + static_cast<int64_t>(sl) * SW + coff;
    G* const grad_col = grad + static_cast<int64_t>(sl) * SW + coff;
    float* const dots_sl = dots + static_cast<int64_t>(sl) * plane + chunk * (static_cast<int64_t>(C) * R);

    // Records: batch bi + 1 .. bi + 3 in registers (lane = position), batch bi and bi + 1 staged in rec_s.
    uint4 r0 = kr.ld(s + lane, e);
    uint4 r1 = kr.ld(s + 32 + lane, e);
    uint4 r2 = kr.ld(s + 64 + lane, e);
    uint4 r3 = kr.ld(s + 96 + lane, e);
    // The lane holding a valid FIRST prefetches its group's bank row slice, the lane holding a valid LAST its grad row
    // slice (each slot has one group per chunk and one writer per launch, so reading it early is safe). TAIL: the slice's
    // valid columns only (the last slice is narrower; a row slice never reaches past its row).
    int pfw = SW;
    if constexpr (kTail) pfw = min(SW, hd - sl * SW);
    auto pf_b = [&](const uint4& r) {
        if (kDot && (r.x & (kFirst | kInvalid)) == kFirst) {
            prefetch_l2(bank + static_cast<uint64_t>(r.y) * static_cast<uint32_t>(hd) + static_cast<int64_t>(sl) * SW,
                        pfw * static_cast<int>(sizeof(E)));
        }
    };
    auto pf_g = [&](const uint4& r) {
        if (kRmw && (r.x & (kLast | kInvalid)) == kLast) {
            prefetch_l2(grad + static_cast<uint64_t>(r.y) * static_cast<uint32_t>(hd) + static_cast<int64_t>(sl) * SW,
                        pfw * static_cast<int>(sizeof(G)));
        }
    };
    // rec_s entry: {token row byte offset in the slice's token rows, scale bits, row element offset slot * hd (bank and
    // grad rows share it; < 2^32, checked on the host) or with W64 the slot, d_q_w index (token * R + lane; bit 31 =
    // invalid)}.
    auto stage = [&](int buf, const uint4& r) {
        const uint32_t px = r.x;
        const uint32_t tk = (px >> 9) & 0xFFu;
        const uint32_t oidx = (tk * static_cast<uint32_t>(R) + (px & 0xFFu)) | ((px & kInvalid) ? 0x80000000u : 0u);
        rec_s[buf * 32 + lane] = make_uint4(static_cast<uint32_t>(min(static_cast<int>(tk), C - 1) * SW) * 4u, r.z,
                                            rec_val<kW64>(r.y, hd), oidx);
    };
    // Uniform masks of a batch: valid FIRSTs (bank row loads), any FIRST (group starts: acc and row reset; FIRST(j) is
    // exactly LAST(j - 1) or the tile start, so this equals mover_da2_cuda's reset after every LAST), valid LASTs (grad
    // updates).
    auto mask_vf = [&](const uint4& r) { return __ballot_sync(kFull, (r.x & (kFirst | kInvalid)) == kFirst); };
    auto mask_af = [&](const uint4& r) { return __ballot_sync(kFull, (r.x & kFirst) != 0u); };
    auto mask_vl = [&](const uint4& r) { return __ballot_sync(kFull, (r.x & (kLast | kInvalid)) == kLast); };
    // valid LASTs whose grad row is loaded (= vl; kept separate from vl for the refill masks)
    auto mask_vg = [&](const uint4& r) { return __ballot_sync(kFull, (r.x & (kLast | kInvalid)) == kLast); };

    stage(0, r0);
    uint32_t vf0 = mask_vf(r0), af0 = mask_af(r0), vl0 = mask_vl(r0), vg0 = mask_vg(r0);
    __syncwarp();
    // Everything above reads data written two or more launches back (tile bounds, records, grad_readings' token rows:
    // inputs of the backward; the host launches with PDL only for tensor records and f32 readings). L2 hints for the
    // first batches may run ahead of the previous kernel (prefetch.global.L2 is a hint; L2 is the point of coherence).
    {
        pf_b(r0);
        pf_g(r0);
        pf_b(r1);
        pf_g(r1);
    }
    GX_SM90_OR_NONE(asm volatile("griddepcontrol.wait;" ::: "memory"));              // previous kernel complete + visible
    GX_SM90_OR_NONE(asm volatile("griddepcontrol.launch_dependents;" ::: "memory"));  // only after this kernel's own wait
    // Row registers of the current sub-batch's positions: mb[j] = bank row if position j is a valid FIRST, gb[j] = grad
    // row if it is a valid LAST (zeros otherwise). Filled here for the tile's first sub-batch.
    RowT mb[8];
    RowG gb[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint64_t ro = rec_row<kW64>(rec_s[j].z, hd);
        if constexpr (kElem) {
            if constexpr (kDot) mb[j] = ((vf0 >> j) & 1u) ? ld_elems_row(bank_col + ro, nvl) : make_zero_row<RowT>();
            if constexpr (kRmw) gb[j] = ((vg0 >> j) & 1u) ? ld_elems_row(grad_col + ro, nvl) : make_zero_row<RowG>();
        } else {
        if constexpr (kDot) mb[j] = ((vf0 >> j) & 1u) ? GX_LD(RowT, bank_col + ro) : make_zero_row<RowT>();
        if constexpr (kRmw) gb[j] = ((vg0 >> j) & 1u) ? GX_LD(RowG, grad_col + ro) : make_zero_row<RowG>();
        }
    }
    const uint32_t ztok = mbar_wait_dep(tok_bar, 0u);
    const unsigned char* tok = reinterpret_cast<const unsigned char*>(tok_s + ztok + coff);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 mrow = make_float4(0.f, 0.f, 0.f, 0.f);    // the current group's bank row
    const int nb = (e - s + 31) >> 5;
    for (int bi = 0; bi < nb; ++bi) {
        if (bi > 0) {
            r1 = r2;
            r2 = r3;
            r3 = kr.ld(s + 32 * (bi + 3) + lane, e);
            {                                         // batch bi + 1
                pf_b(r1);
                pf_g(r1);
            }
        }
        __syncwarp();                                 // every lane is done with buffer (bi + 1) & 1 (batch bi - 1)
        stage((bi + 1) & 1, r1);
        const uint32_t vf1 = mask_vf(r1), af1 = mask_af(r1), vl1 = mask_vl(r1), vg1 = mask_vg(r1);
        __syncwarp();
        const uint4* rc = rec_s + (bi & 1) * 32;
        const uint4* rn = rec_s + ((bi + 1) & 1) * 32;
        const int pos0 = s + 32 * bi;
        const int kend = min(32, e - pos0);
#pragma unroll 1
        for (int k8 = 0; k8 < kend; k8 += 8) {
            // The next sub-batch: the rest of this batch, else the next batch's first 8 positions, else none.
            const uint4* rq_n = rc;
            uint32_t fn8 = 0u, ln8 = 0u;
            if (k8 + 8 < kend) {                      // ln8: grad rows to load (valid LASTs)
                rq_n = rc + k8 + 8;
                fn8 = vf0 >> (k8 + 8);
                ln8 = vg0 >> (k8 + 8);
            } else if (bi + 1 < nb) {
                rq_n = rn;
                fn8 = vf1;
                ln8 = vg1;
            }
            const uint4* rq = rc + k8;
            const uint32_t a8 = af0 >> k8, l8 = vl0 >> k8;
            float pd[8];
            // Refill position j with the next sub-batch's rows (8 positions ahead): predicated loads, no branch.
            // (A macro, not a lambda: mb / gb must stay register arrays indexed by unrolled constants.)
#define GX_REFILL(j)                                                                                           \
            {                                                                                                  \
                const uint64_t no = rec_row<kW64>(rq_n[j].z, hd);                                              \
                if constexpr (kElem) {                                                                         \
                    if constexpr (kDot) mb[j] = ((fn8 >> (j)) & 1u) ? ld_elems_row(bank_col + no, nvl) : make_zero_row<RowT>(); \
                    if constexpr (kRmw) gb[j] = ((ln8 >> (j)) & 1u) ? ld_elems_row(grad_col + no, nvl) : make_zero_row<RowG>(); \
                } else {                                                                                       \
                if constexpr (kDot) mb[j] = ((fn8 >> (j)) & 1u) ? GX_LD(RowT, bank_col + no) : make_zero_row<RowT>(); \
                if constexpr (kRmw) gb[j] = ((ln8 >> (j)) & 1u) ? GX_LD(RowG, grad_col + no) : make_zero_row<RowG>(); \
                }                                                                                              \
            }
            if ((a8 & l8 & 0xFFu) == 0xFFu) {
                // Every position starts a group (any FIRST) and ends it as a valid LAST: exactly the general body
                // below with both branches taken (mrow = cvt4(mb[j]), acc = +0, then the same fmaf / dot / sum /
                // store), written without branches; acc and mrow end as position 7's, as there.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint4 rv = rq[j];
                    const float4 x = *reinterpret_cast<const float4*>(tok + rv.x);   // rv.x: byte offset
                    const float4 m = cvt4(mb[j]);
                    const float sc = __uint_as_float(rv.y);
                    float4 a;
                    a.x = fmaf(sc, x.x, 0.f);
                    a.y = fmaf(sc, x.y, 0.f);
                    a.z = fmaf(sc, x.z, 0.f);
                    a.w = fmaf(sc, x.w, 0.f);
                    pd[j] = on ? __fmaf_rn(x.w, m.w, __fmaf_rn(x.z, m.z, __fmaf_rn(x.y, m.y, __fmul_rn(x.x, m.x))))
                               : 0.f;
                    const float4 gv = cvt4(gb[j]);
                    float4 o;
                    o.x = gv.x + a.x;
                    o.y = gv.y + a.y;
                    o.z = gv.z + a.z;
                    o.w = gv.w + a.w;
                    if constexpr (kElem) {
                        if (on) st_elems_row(grad_col + rec_row<kW64>(rv.z, hd), o, nvl);
                    } else {
                    if (on) st4(grad_col + rec_row<kW64>(rv.z, hd), o);   // off lanes: no store (they mirror lane 0's address)
                    }
                    GX_REFILL(j)
                    if (j == 7) {
                        mrow = m;
                        acc = a;
                    }
                }
            } else
            {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint4 rr = rq[j];                                       // {x, y, z, w} in one read
                    const float4 x = *reinterpret_cast<const float4*>(tok + rr.x);   // rr.x: byte offset
                    if ((a8 >> j) & 1u) {
                        mrow = cvt4(mb[j]);
                        acc = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                    const float sc = __uint_as_float(rr.y);
                    acc.x = fmaf(sc, x.x, acc.x);
                    acc.y = fmaf(sc, x.y, acc.y);
                    acc.z = fmaf(sc, x.z, acc.z);
                    acc.w = fmaf(sc, x.w, acc.w);
                    pd[j] = on ? __fmaf_rn(x.w, mrow.w, __fmaf_rn(x.z, mrow.z, __fmaf_rn(x.y, mrow.y,
                                                                                          __fmul_rn(x.x, mrow.x))))
                               : 0.f;
                    if ((l8 >> j) & 1u) {
                        const float4 gv = cvt4(gb[j]);
                        float4 o;
                        o.x = gv.x + acc.x;
                        o.y = gv.y + acc.y;
                        o.z = gv.z + acc.z;
                        o.w = gv.w + acc.w;
                        if constexpr (kElem) {
                            if (on) st_elems_row(grad_col + rec_row<kW64>(rr.z, hd), o, nvl);
                        } else {
                        if (on) st4(grad_col + rec_row<kW64>(rr.z, hd), o);   // off lanes: no store (they mirror lane 0's address)
                        }
                    }
                    GX_REFILL(j)
                }
            }
#undef GX_REFILL
            // Reduce-scatter of the 8 partial dots: lane L ends with entry (L >> 2) & 7 summed over all lanes, in the
            // xor-16, 8, 4, 2, 1 order of a butterfly (mover_da2_cuda's tree, so d_q_w matches it bit for bit).
            if constexpr (kDot) {
            const bool h4 = (lane >> 4) & 1, h3 = (lane >> 3) & 1, h2 = (lane >> 2) & 1;
            float a4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float keep = h4 ? pd[i + 4] : pd[i];
                const float send = h4 ? pd[i] : pd[i + 4];
                a4[i] = keep + __shfl_xor_sync(kFull, send, 16);
            }
            float a2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float keep = h3 ? a4[i + 2] : a4[i];
                const float send = h3 ? a4[i] : a4[i + 2];
                a2[i] = keep + __shfl_xor_sync(kFull, send, 8);
            }
            float a1;
            {
                const float keep = h2 ? a2[1] : a2[0];
                const float send = h2 ? a2[0] : a2[1];
                a1 = keep + __shfl_xor_sync(kFull, send, 4);
            }
            a1 += __shfl_xor_sync(kFull, a1, 2);
            a1 += __shfl_xor_sync(kFull, a1, 1);
            const int kk = k8 + ((lane >> 2) & 7);
            if ((lane & 3) == 0 && pos0 + kk < e) {
                const uint32_t w = rc[kk].w;
                dots_sl[w & 0x7FFFFFFFu] = (w & 0x80000000u) ? 0.f : a1;
            }
            }
        }
        vf0 = vf1;
        af0 = af1;
        vl0 = vl1;
        vg0 = vg1;
    }
    (void)mbar_wait_dep(tok_bar, 0u);   // no CTA exits with its token-row copy in flight
}

template <typename K>
void set_smem(K kernel, size_t bytes) {   // cudaFuncSetAttribute is per device: key on (kernel, device)
    static std::mutex mu;
    static std::map<std::pair<const void*, int>, size_t> done;
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_pair(reinterpret_cast<const void*>(kernel), dev);
    auto it = done.find(key);
    if (it != done.end() && it->second >= bytes) return;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(bytes)));
    done[key] = bytes;
}

int slices_of(int hd) { return 1 + (hd - 1) / 128; }

// A plain launch, or (pdl) one with programmatic stream serialization (captured as a programmatic graph edge): the
// kernel may start while the previous kernel in the stream runs; its griddepcontrol.wait orders it.
template <typename K, typename... Args>
void gx_launch(K kernel, int blocks, int threads, size_t smem, cudaStream_t st, bool pdl, Args... args) {
    if (sdm_plancap::take(kernel, dim3(blocks), dim3(threads), smem, args...)) return;   // plan capture
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(threads);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
        return;
    }
    kernel<<<blocks, threads, smem, st>>>(args...);
}

// KBF: the bf16 entry (gx_gds_bf, regular module only), picked once per factory call (make_gds) for bf16 bank + bf16
// grad below 2^32 elements: the row type and offset width are compile-time, so each step runs the plain host path (no
// row-type or offset-width selection per step). !KBF (gx_gds): every row type of the module. The same kernels,
// geometry and arguments for the rows both take.
template <bool KBF>
void gx_gds_run(const at::Tensor& bank, const at::Tensor& grad, const at::Tensor& gr, int64_t gr_step_rows,
                const at::Tensor& rec, const at::Tensor& spay, const at::Tensor& tb, const at::Tensor& dots, int64_t ti,
                int64_t P, int64_t C, int64_t R, int64_t K, int64_t nw, int64_t pdl) {
    const at::cuda::CUDAGuard guard(bank.device());
    // Row types: bf16 bank + bf16 grad (32-bit row offsets below 2^32 elements), bf16 bank + f32 grad (the
    // grad_bank_fp32 option), f32 bank + f32 grad; every instantiation other than bf16 / bf16 / 32-bit offsets is a
    // MODE 4 one (64-bit row offsets) and reads 8-byte records (rec8 + spay_q).
    bool bf_bank = true, g32 = false, w64 = false, elem = false;
    int hd = 0, NSL = 0, SW = 0;
    if constexpr (KBF) {
        // The plain validation sequence of bf16 / bf16 rows in equal 16-byte slices (no tail rows reach this
        // entry: make_gds picks it only for the regular module's rows). Bank rows < 2^30 follow from < 2^32 elements
        // and hd % 4 == 0 (the records' 31-bit slots).
        TORCH_CHECK(bank.dim() == 2 && bank.is_contiguous() && grad.is_contiguous() && grad.sizes() == bank.sizes() &&
                    bank.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kBFloat16,
                    "gx_gds (bf16 entry): bf16 bank and bf16 grad [slots, hd], contiguous");
        TORCH_CHECK(bank.numel() < (int64_t(1) << 32), "gx_gds (bf16 entry): bank of 2^32 elements or more");
        hd = static_cast<int>(bank.size(1));   // < 2^32 elements: a head of >= 2^31 narrows to <= 0 (refused next)
        TORCH_CHECK(hd > 0 && bank.size(0) > 0, "gx_gds: empty bank or head_dim >= 2^31");
        NSL = slices_of(hd);
        TORCH_CHECK(hd % NSL == 0, "gx_gds: head_dim split");
        SW = hd / NSL;
        TORCH_CHECK((hd % 4) == 0 && (SW % 4) == 0, "gx_gds: head_dim / slice width multiples of 4");
    } else {
        bf_bank = bank.scalar_type() == at::kBFloat16;
        g32 = grad.scalar_type() == at::kFloat;
        TORCH_CHECK(bank.dim() == 2 && bank.is_contiguous() && grad.is_contiguous() && grad.sizes() == bank.sizes() &&
                    (bf_bank || bank.scalar_type() == at::kFloat) && (g32 || grad.scalar_type() == at::kBFloat16) &&
                    (bf_bank || g32),
                    "gx_gds: bf16 bank + bf16 / f32 grad, or f32 bank + f32 grad, [slots, hd], contiguous");
        TORCH_CHECK(bank.size(1) > 0 && bank.size(1) < (int64_t(1) << 31) && bank.size(0) > 0,
                    "gx_gds: empty bank or head_dim >= 2^31");   // checked before narrowing to int
#if defined(GX_GDS_TAILS_ONLY)
        TORCH_CHECK(bank.size(1) <= (int64_t(1) << 20), "gx_gds (tail module): head_dim <= 2^20 (the tail-module bound)");
#endif
        hd = static_cast<int>(bank.size(1));
        TORCH_CHECK(bank.size(0) <= (int64_t(1) << 31) - 1, "gx_gds: bank rows must fit the records' 31-bit slots");
        w64 = !bf_bank || g32 || bank.numel() >= (int64_t(1) << 32);
        NSL = slices_of(hd);
        // Equal slices of hd / NSL columns, or (tails: hd % NSL != 0 or a slice width that is not a multiple of 4)
        // dc_pa3's slices of SW = 4 ceil(ceil(hd / 4) / NSL) columns, the last one narrower (MODE bit 8, 8 warps per
        // CTA; hd % 4 != 0, always a tail: element pieces, MODE bit 16). Tails run in the tail module
        // (sdm_gx_gds_tail).
        const bool tail = hd % NSL != 0 || (hd / NSL) % 4 != 0;
        elem = hd % 4 != 0;
        SW = tail ? 4 * ((1 + (hd - 1) / 4 + NSL - 1) / NSL) : hd / NSL;   // ceil(hd / 4) without int overflow
        TORCH_CHECK((SW % 4) == 0, "gx_gds: slice width a multiple of 4");
        TORCH_CHECK(!tail || nw == 8, "gx_gds: tail slices take 8 warps per CTA");
#if defined(GX_GDS_TAILS_ONLY)
        TORCH_CHECK(tail, "gx_gds (tail module): tail slices only");
#else
        TORCH_CHECK(!tail, "gx_gds: tail slices run in the tail module (gx_gds_cuda._tail_module)");
#endif
    }
    TORCH_CHECK(gr.scalar_type() == at::kFloat && gr.is_contiguous(), "gx_gds: grad_readings f32 contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bank.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(gr.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(dots.data_ptr()) % 4 == 0, "gx_gds: 16-byte alignment (bank, grad, readings)");
    TORCH_CHECK(dots.is_contiguous() && dots.scalar_type() == at::kFloat, "gx_gds: dots f32 contiguous");
    TORCH_CHECK(tb.is_contiguous() && tb.scalar_type() == at::kInt && tb.dim() == 2, "gx_gds: tiles int32 [N, TU + 1]");
    TORCH_CHECK(nw == 4 || nw == 8 || nw == 16, "gx_gds: 4, 8 or 16 warps per CTA");
    TORCH_CHECK(C >= 1 && C <= 256 && R >= 1 && R <= 256, "gx_gds: chunk / read width");
    const int64_t N = tb.size(0);
    const int n = static_cast<int>(C * R);
    // Records: int32 [Nr, n, 4] (qrec) or [Nr, n, 2] (rec8, + spay_q [N, n]); Nr = N (every chunk) or P (this step).
    TORCH_CHECK(rec.scalar_type() == at::kInt && rec.is_contiguous() && rec.dim() == 3 && rec.size(1) == n &&
                (rec.size(2) == 4 || rec.size(2) == 2) && (rec.size(0) == N || rec.size(0) == P),
                "gx_gds: records int32 [N or P, C * R, 4 (qrec) or 2 (rec8)]");
    const int REC = rec.size(2) == 4 ? 0 : 1;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(rec.data_ptr()) % (REC == 0 ? 16 : 8) == 0, "gx_gds: record alignment");
    if constexpr (!KBF) {
        TORCH_CHECK(!w64 || REC == 1, "gx_gds: the float32-row / 64-bit-offset instantiations read 8-byte records "
                    "(rec8 + spay_q)");
    }
    TORCH_CHECK(REC == 0 || (spay.scalar_type() == at::kInt && spay.is_contiguous() && spay.numel() == N * n &&
                             reinterpret_cast<uintptr_t>(spay.data_ptr()) % 4 == 0),
                "gx_gds: rec8 needs spay_q int32 [N, C * R]");
    const int64_t rec_chunk0 = rec.size(0) == N ? 0 : ti * P;
    const int64_t plane = N * static_cast<int64_t>(n);
    TORCH_CHECK(dots.numel() == plane * NSL, "gx_gds: dots planes");
    const int TU = static_cast<int>(tb.size(1)) - 1;
    TORCH_CHECK(TU == K * nw, "gx_gds: tiles");
    TORCH_CHECK(ti >= 0 && P >= 1 && (ti + 1) * P <= N && N < (int64_t(1) << 31), "gx_gds: step out of range");
    TORCH_CHECK(gr_step_rows >= 0 && gr.numel() >= (ti * gr_step_rows + P * C) * hd, "gx_gds: grad_readings too small");
    for (const at::Tensor* t : {&grad, &gr, &rec, &tb, &dots})
        TORCH_CHECK(t->device() == bank.device(), "gx_gds: all operands on the bank's device");
    TORCH_CHECK(REC == 0 || spay.device() == bank.device(), "gx_gds: spay_q on the bank's device");
    Geo g;
    g.P = static_cast<int>(P);
    g.C = static_cast<int>(C);
    g.n = n;
    g.hd = hd;
    g.NSL = NSL;
    g.SW = SW;
    g.K = static_cast<int>(K);
    g.TU = TU;
    g.tok_bytes = ((static_cast<int>(C) * SW * 4 + 64) + 127) / 128 * 128;
    const size_t smem = static_cast<size_t>(g.tok_bytes) + static_cast<size_t>(nw) * 64 * sizeof(uint4) + 16;
    TORCH_CHECK(K >= 1 && K <= (int64_t(1) << 20), "gx_gds: tiles per chunk");
    const int64_t blocks64 = P * static_cast<int64_t>(NSL) * K;   // full product in 64 bits before narrowing
    TORCH_CHECK(blocks64 <= static_cast<int64_t>(0x7FFFFFFF), "gx_gds: grid of ", blocks64, " CTAs exceeds 2^31 - 1");
    const int blocks = static_cast<int>(blocks64);
    if (blocks == 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
    const int32_t* sp = REC == 1 ? spay.data_ptr<int32_t>() : nullptr;
#define GX_L(E_, G_, NW_, REC_, MODE_)                                                        \
    {                                                                                         \
        auto kf = gx_gds_kernel<E_, G_, NW_, REC_, MODE_>;                                    \
        set_smem(kf, smem);                                                                   \
        gx_launch(kf, blocks, NW_ * 32, smem, stream, pdl != 0,                               \
            reinterpret_cast<const E_*>(bank.data_ptr()),                                     \
            reinterpret_cast<G_*>(grad.data_ptr()), static_cast<const float*>(gr.data_ptr<float>()), \
            gr_step_rows, static_cast<const void*>(rec.data_ptr()), sp, rec_chunk0,           \
            static_cast<const int32_t*>(tb.data_ptr<int32_t>()), dots.data_ptr<float>(), plane, \
            static_cast<int>(ti), g, static_cast<int>(R));                                    \
    }
#define GX_LN(E_, G_, REC_, MODE_) { if (nw == 4) GX_L(E_, G_, 4, REC_, MODE_) else if (nw == 16) GX_L(E_, G_, 16, REC_, MODE_) else GX_L(E_, G_, 8, REC_, MODE_) }
#if defined(GX_GDS_TAILS_ONLY)
    // Tail slices, 8 warps: MODE 8 (| 4 W64) for hd % 4 == 0, MODE 24 (| 4) element pieces for hd % 4 != 0. Their
    // own module: the regular module's instantiations stay exactly the regular set.
    static_assert(!KBF, "gx_gds: the bf16 entry is the regular module's");
    if (!elem) {
        if (!w64) {
            if (REC == 0) GX_L(__nv_bfloat16, __nv_bfloat16, 8, 0, 8) else GX_L(__nv_bfloat16, __nv_bfloat16, 8, 1, 8)
        } else if (!bf_bank) {
            GX_L(float, float, 8, 1, 12)
        } else if (g32) {
            GX_L(__nv_bfloat16, float, 8, 1, 12)
        } else {
            GX_L(__nv_bfloat16, __nv_bfloat16, 8, 1, 12)
        }
    } else if (!w64) {
        if (REC == 0) GX_L(__nv_bfloat16, __nv_bfloat16, 8, 0, 24) else GX_L(__nv_bfloat16, __nv_bfloat16, 8, 1, 24)
    } else if (!bf_bank) {
        GX_L(float, float, 8, 1, 28)
    } else if (g32) {
        GX_L(__nv_bfloat16, float, 8, 1, 28)
    } else {
        GX_L(__nv_bfloat16, __nv_bfloat16, 8, 1, 28)
    }
#else
    (void)elem;
    if (KBF || !w64) {
        if (REC == 0) GX_LN(__nv_bfloat16, __nv_bfloat16, 0, 0) else GX_LN(__nv_bfloat16, __nv_bfloat16, 1, 0)
    } else if constexpr (!KBF) {
        if (!bf_bank) {
            GX_LN(float, float, 1, 4)
        } else if (g32) {
            GX_LN(__nv_bfloat16, float, 1, 4)
        } else {
            GX_LN(__nv_bfloat16, __nv_bfloat16, 1, 4)
        }
    }
#endif
#undef GX_LN
#undef GX_L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void gx_gds(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor rec, at::Tensor spay,
            at::Tensor tb, at::Tensor dots, int64_t ti, int64_t P, int64_t C, int64_t R, int64_t K, int64_t nw,
            int64_t pdl) {
    gx_gds_run<false>(bank, grad, gr, gr_step_rows, rec, spay, tb, dots, ti, P, C, R, K, nw, pdl);
}

#if !defined(GX_GDS_TAILS_ONLY)
// The bf16 entry: the same kernels, geometry and arguments as gx_gds for bf16 / bf16 rows below 2^32 elements.
void gx_gds_bf(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor rec, at::Tensor spay,
            at::Tensor tb, at::Tensor dots, int64_t ti, int64_t P, int64_t C, int64_t R, int64_t K, int64_t nw,
            int64_t pdl) {
    gx_gds_run<true>(bank, grad, gr, gr_step_rows, rec, spay, tb, dots, ti, P, C, R, K, nw, pdl);
}
#endif

// ---- the backward's C++ step loop (bwd_steps_cuda): gx_gds as a per-backward plan -------------------------------------
// gx_gds_plan runs the per-step entry's own host path (gx_gds_run<kbf>: gx_gds, or the regular module's bf16 entry
// gx_gds_bf) with pdl 0 ONCE under plan capture at step ti_max: every check runs (its step bounds cover every
// ti <= ti_max) and the launch is recorded. gx_gds_plan_launch (a plain C++ entry point; address from gx_gds_launch_addr)
// sets the step parameters (ti; rec_chunk0 = ti * P for per-step records, else 0, as gx_gds computes it) and issues the
// recorded launch on the given stream (the caller's current stream, as gx_gds uses): with programmatic stream
// serialization when pdl != 0, else plain -- gx_launch's choice for the same pdl.
namespace {
constexpr uint64_t kGdsPlanMagic = 0x314e414c50534447ull;
constexpr int kGdsRecChunk0Param = 6;   // (bank, grad, gr, gr_step_rows, rec, spay, rec_chunk0, tb, dots, plane, ti, g, R)
constexpr int kGdsTiParam = 10;
struct GdsPlan {
    uint64_t magic;
    int32_t device;
    int32_t rec_per_step;
    int32_t kbf;
    int32_t pad;
    int64_t ti_max, P;
    sdm_plancap::Launch cap;
};

int64_t gx_gds_plan_launch(const void* plan_bytes, int64_t nbytes, int64_t ti, int64_t pdl, void* stream) {
    GdsPlan pl;
    if (!sdm_plancap::from_bytes(pl, plan_bytes, nbytes, kGdsPlanMagic)) return -1;
    if (ti < 0 || ti > pl.ti_max) return -2;
    if (pl.cap.set == 0) return 0;                 // gx_gds launched nothing (empty grid)
    if (sdm_plancap::current_device() != pl.device) return -5;
    int* tip = sdm_plancap::param<int>(pl.cap, kGdsTiParam);
    int64_t* r0p = sdm_plancap::param<int64_t>(pl.cap, kGdsRecChunk0Param);
    if (tip == nullptr || r0p == nullptr) return -7;
    *tip = static_cast<int>(ti);
    *r0p = pl.rec_per_step ? ti * pl.P : 0;
    return sdm_plancap::launch(pl.cap, pdl != 0, static_cast<cudaStream_t>(stream));
}
}  // namespace

at::Tensor gx_gds_plan(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor rec,
                       at::Tensor spay, at::Tensor tb, at::Tensor dots, int64_t ti_max, int64_t P, int64_t C, int64_t R,
                       int64_t K, int64_t nw, bool kbf) {
    TORCH_CHECK(ti_max >= 0 && ti_max < (int64_t(1) << 31), "gx_gds_plan: ti_max");
    GdsPlan pl;
    std::memset(&pl, 0, sizeof(pl));
    {
        sdm_plancap::Scope scope(&pl.cap);
#if defined(GX_GDS_TAILS_ONLY)
        TORCH_CHECK(!kbf, "gx_gds_plan: the bf16 entry is the regular module's");
        gx_gds_run<false>(bank, grad, gr, gr_step_rows, rec, spay, tb, dots, ti_max, P, C, R, K, nw, 0);
#else
        if (kbf) {
            gx_gds_run<true>(bank, grad, gr, gr_step_rows, rec, spay, tb, dots, ti_max, P, C, R, K, nw, 0);
        } else {
            gx_gds_run<false>(bank, grad, gr, gr_step_rows, rec, spay, tb, dots, ti_max, P, C, R, K, nw, 0);
        }
#endif
    }
    pl.magic = kGdsPlanMagic;
    pl.device = static_cast<int32_t>(bank.get_device());
    pl.rec_per_step = (rec.dim() == 3 && rec.size(0) != tb.size(0)) ? 1 : 0;   // gx_gds: rec_chunk0 = (Nr == N) ? 0 : ti * P
    pl.kbf = kbf ? 1 : 0;
    pl.ti_max = ti_max;
    pl.P = P;
    TORCH_CHECK(pl.cap.set == 0 || (pl.cap.nparams == 13 && sdm_plancap::param<int>(pl.cap, kGdsTiParam) != nullptr &&
                                    sdm_plancap::param<int64_t>(pl.cap, kGdsRecChunk0Param) != nullptr),
                "gx_gds_plan: parameters");
    return sdm_plancap::to_tensor(pl);
}

int64_t gx_gds_launch_addr() { return reinterpret_cast<int64_t>(&gx_gds_plan_launch); }

""".replace("@ROW_HELPERS@\n", ROW_HELPERS.lstrip("\n"))

@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_gx_gds",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("gx_gds", "gx_gds_bf", "gx_gds_plan", "gx_gds_launch_addr"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo"),
    )


@functools.cache
def _tail_module():
    """Tails: the same source with only the tail instantiations (MODE 8 / 24), built on first use."""
    return load_cuda_extension(
        "sdm_gx_gds_tail",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("gx_gds", "gx_gds_plan", "gx_gds_launch_addr"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo", "-DGX_GDS_TAILS_ONLY"),
    )


def _wave_warps(device=None) -> int:
    """Resident warps of one wave: 16 per SM at 128 registers (GB300: 2432)."""
    return 16 * num_sms(device)


def _target_warps(device=None) -> int:
    """Warps a step aims for: the largest power of two within one wave (GB300: 2048)."""
    return prev_pow2(_wave_warps(device))


def geometry(num_parallel: int = 0, nsl: int = 1, n: int = 0, chunk: int = 0, sw: int = 0, device=None) -> tuple:
    """(nw, ppw): warps per CTA and positions per warp tile, from shapes and the device's SM count (the outputs depend on
    neither: the walk's arithmetic is per sorted group, and tiles snap to group starts). Two shape regimes first:
      (a) the slice's token rows exceed 96 KB (chunk * sw * 4 B, e.g. C 256 with 128-column slices: one CTA per SM):
          16 warps, 256-position tiles;
      (b) read width n / chunk >= 256: 4 warps, 32-position tiles.
    Otherwise 8 warps and ppw 128, halved (floor 32) while a step would launch fewer than _target_warps(device) warps,
    i.e. num_parallel * nsl * ceil(n / ppw) < target. When that still launches fewer than _wave_warps(device) warps
    (16 resident warps per SM at 128 registers), the step's warps fill one wave evenly instead: 4-warp CTAs (4 per SM:
    the same 16 warps), K = wave // (4 * num_parallel * nsl) CTAs per (chunk, slice) (at most ceil(n / 64): tiles of
    >= 16 positions), ppw = ceil(n / (4 K)); 8-warp CTAs would leave some SMs one CTA short."""
    if chunk > 0:
        if chunk * sw * 4 > 96 * 1024:
            return 16, 256
        if n >= 256 * chunk:
            return 4, 32
    nw = 8
    target = _target_warps(device)
    ppw = 128
    while ppw > 32 and num_parallel * nsl * (-(-n // ppw)) < target:
        ppw //= 2
    wave = _wave_warps(device)
    if n > 0 and num_parallel >= 1 and nsl >= 1 and num_parallel * nsl * nw * (-(-n // (nw * ppw))) < wave:
        k = max(1, min(wave // (4 * num_parallel * nsl), -(-n // 64)))
        return 4, -(-n // (4 * k))
    return nw, ppw


def supported(bank: torch.Tensor, grad_bank: torch.Tensor, chunk_size: int, width: int) -> bool:
    """Shape / dtype / arch gate (never the data): a bf16 bank with bf16 or f32 grad rows, or an f32 bank with f32 grad
    rows, on every CUDA backend (sm_80+), rows of whole 4-element vectors, bank rows < 2^31, and the token rows of a slice
    fitting the device's opt-in shared memory."""
    return admits(bank, grad_bank.dtype, chunk_size, width)


ROW_TYPES = ((torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.float32))


def wide_offsets(bank: torch.Tensor, grad_dtype) -> bool:
    """The call takes a MODE-4 instantiation (64-bit row offsets): float32 rows, or bf16 banks of >= 2^32 elements."""
    return not (bank.dtype == torch.bfloat16 and grad_dtype == torch.bfloat16) or bank.numel() >= 2 ** 32


def slices(hd: int) -> tuple:
    """(nsl, sw, tail): equal slices of hd / nsl columns (a multiple of 4), or (tail) dc_pa3's uneven slices."""
    nsl = -(-hd // 128)
    if hd % nsl == 0 and (hd // nsl) % 4 == 0:
        return nsl, hd // nsl, False
    return nsl, 4 * (-(-(-(-hd // 4)) // nsl)), True


def _fits(bank, chunk_size: int, width: int, sw: int, nsl: int, tail: bool) -> bool:
    nw = 8 if tail else geometry(0, nsl, chunk_size * width, chunk_size, sw, bank.device)[0]   # the launch's warps
    tok = (chunk_size * sw * 4 + 64 + 127) // 128 * 128
    return tok + nw * 64 * 16 + 16 <= da2._smem_limit(bank.device)


def tail_supported(bank, chunk_size, quant="none") -> bool:
    """Tail hook (movers.tail_admits): this GDS takes the call's head on every backend (rtx: chunks <= 128); it never
    reads snapshot payloads, so ``quant`` does not matter."""
    if not bank.is_cuda or bank.dim() != 2 or bank.dtype not in (torch.bfloat16, torch.float32):
        return False
    try:
        b = backend(bank.device)
    except RuntimeError:
        return False
    if not (b in ("modern", "a100") or (b == "rtx" and chunk_size <= 128)) or not (1 <= chunk_size <= 256):
        return False
    hd = bank.shape[-1]
    if not (1 <= hd <= 2 ** 20) or bank.shape[0] >= 2 ** 31:   # hd bound: the forward's
        return False
    nsl, sw, tail = slices(hd)
    return _fits(bank, chunk_size, 256, sw, nsl, tail)


def admits(bank: torch.Tensor, grad_dtype, chunk_size: int, width: int, quant="none") -> bool:
    """supported() with the grad bank's dtype instead of the tensor (callers deciding before it exists); tails via
    movers.tail_admits."""
    if bank.dim() != 2 or (bank.dtype, grad_dtype) not in ROW_TYPES or not bank.is_cuda:
        return False
    if not cuda_backend(bank.device):
        return False
    hd = bank.shape[-1]
    if not 0 < hd < 2 ** 31:                        # the native launchers take hd as int
        return False
    if not (1 <= chunk_size <= 256) or not (1 <= width <= 256):
        return False
    if bank.shape[0] >= 2 ** 31:                    # the records' slots are 31-bit
        return False
    nsl, sw, tail = slices(hd)
    if tail:
        from .. import movers
        if not movers.tail_admits(bank, chunk_size, quant):
            return False
    return _fits(bank, chunk_size, width, sw, nsl, tail)


LAST_POLICY = {}
# 16-byte records copied to the 8-byte layout for a MODE-4 instantiation (once per factory call; tests, diagnostics)
REC8_COPIES = [0]


def _factory_records(qrec):
    """The records of a provider that built ALL of them with ONE plain launch on the current stream when the backward
    factory ran (px_qrec_cuda.build_qrec8_steps under CUDA-graph capture or for Nt == 1, which the dispatch layer wraps
    as a record holder with events None and no release hook) are complete before the backward's step loop starts (its
    first dc_gwr is a plain launch, and nothing writes them afterwards), exactly like tensor records built at factory
    time (Phase 1's, or mover_da2_cuda.build_qrec): they are returned as that tensor, so that GDS may launch with PDL
    (its pre-wait reads of the records follow that plain-launch boundary). The kernel reads the same tensor either way
    (the holder's get(ti) returns (records, None) for every step), so outputs are unchanged. Any other provider
    (per-step events: records written on a side stream during the loop; a release hook; no `events` attribute) is
    returned unchanged and keeps plain launches."""
    if qrec is None or torch.is_tensor(qrec):
        return qrec
    rec = getattr(qrec, "rec8", None)
    if getattr(qrec, "events", 0) is None and getattr(qrec, "release", None) is None and torch.is_tensor(rec):
        return rec
    return qrec


def records_ok(qrec, spay_q) -> bool:
    """The record inputs (any of them has the same content per sorted read position):
          * qrec  int32 [N, C * R, 4] {spay_q element, absolute slot, scale bits, 0} (Phase 1 / mover_da2_cuda.build_qrec);
          * rec8  int32 [N, C * R, 2] {absolute slot, scale bits} (scale = __fmul_rn(f32(q_val), expf(log_q)); padding:
            0, 0), with the flags / token / lane read from spay_q;
          * a provider with get(ti) -> (records of step ti or of every chunk as int32 [P or N, C * R, 2 or 4], CUDA event
            or None) and optionally release(ti, event): GDS(ti) waits on the event before it reads the records and calls
            release with an event recorded right after its launch (the records may be overwritten once it completed)."""
    if qrec is None:
        return False
    if hasattr(qrec, "get") and callable(qrec.get):
        return True
    if not torch.is_tensor(qrec) or qrec.dtype != torch.int32 or not qrec.is_contiguous():
        return False
    n = spay_q.shape[-1]
    N = spay_q.numel() // n
    if qrec.numel() == N * n * 4:
        return qrec.data_ptr() % 16 == 0
    if qrec.numel() == N * n * 2:
        return qrec.data_ptr() % 8 == 0
    return False


def make_gds(bank, grad_bank, grad_readings, spay_q, qrec, d_q_w, *, chunk_size, num_parallel,
             steps, q_idx=None, valid_tokens=None, tiles_q=None):
    """Step launcher of this GDS mover; the same arguments and outputs as mover_da2_cuda.make_gds (which it delegates to
    outside its scope, keeping that kernel's bits there; it needs the 16-byte qrec). `qrec`: see records_ok()."""
    from_provider = qrec is not None and not torch.is_tensor(qrec)
    qrec = _factory_records(qrec)                  # factory-time records of a provider = tensor records
    from_provider = from_provider and torch.is_tensor(qrec)
    n = spay_q.shape[-1]
    r = n // chunk_size
    if not supported(bank, grad_bank, chunk_size, r) or not records_ok(qrec, spay_q):
        # mover_da2_cuda's GDS (its kernel and bits) for the shapes this one declines, on the caller's 16-byte records
        # (movers.make_mover_gds). gds_admits reads no read width: the admission is checked again here.
        grad_ok = grad_bank.dtype == bank.dtype or (bank.dtype == torch.bfloat16 and grad_bank.dtype == torch.float32)
        if not (da2.supported(bank, chunk_size) and grad_ok and da2.fits(bank, chunk_size, "gds", grad_bank.dtype)):
            raise ValueError(f"make_gds: the mover_da2 route admits its own GDS inputs only (bank bf16 / f32 [slots, hd] "
                             f"on sm_90+, grad = bank dtype or f32 with a bf16 bank, fitting shapes); got bank "
                             f"{bank.dtype} {tuple(bank.shape)}, grad {grad_bank.dtype}")
        LAST_POLICY.clear()
        LAST_POLICY["impl"] = "da2"
        inner = da2.make_gds(bank, grad_bank, grad_readings, spay_q, qrec.view(-1, n, 4), d_q_w,
                             chunk_size=chunk_size, num_parallel=num_parallel, steps=steps, q_idx=q_idx,
                             valid_tokens=valid_tokens, tiles_q=tiles_q)

        def launch_da2(ti: int) -> None:
            inner(ti)

        launch_da2.geom = dict(inner.geom, impl="da2")
        return launch_da2
    hd = bank.shape[-1]
    sc = num_parallel * chunk_size
    if grad_readings.numel() == steps * sc * hd:
        gr_rows = sc
    elif grad_readings.numel() == sc * hd:
        gr_rows = 0
    else:
        raise ValueError("grad_readings: unexpected size")
    nsl, sw, tail = slices(hd)
    nw, ppw = geometry(int(num_parallel), nsl, n, int(chunk_size), sw, bank.device)
    if tail:
        nw = 8                                     # the tail instantiations' warps (the launcher checks)
    elif grad_bank.dtype == torch.float32 and capability(bank.device) == (9, 0) and (nw, ppw) in ((8, 128), (16, 256)):
        # H100, float32 grad rows: 8-warp CTAs of 64-position tiles (16 warps starve the concurrent dc_gwr). Speed only.
        nw, ppw = 8, 64
    K = max(1, -(-n // (nw * ppw)))
    spay2 = spay_q.reshape(-1, n)
    N = spay2.shape[0]
    P = int(num_parallel)
    if P < 1 or N >= 2 ** 31:
        raise ValueError(f"make_gds: P >= 1 and N < 2^31 record chunks needed; got P {P}, N {N}")
    # the steps plan() covers (exact int64 products, ti < 2^31)
    lim = N // P if gr_rows == 0 else min(N // P, int(steps))
    tb = da2._tiles_or_build(spay2, tiles_q, K * nw) if ppw == 128 else None   # Phase 1's tiles use L = 128
    if tb is None:
        tb = da2.build_tiles(spay2, K * nw)
    provider = qrec if not torch.is_tensor(qrec) else None
    w64 = wide_offsets(bank, grad_bank.dtype)
    if provider is None:
        rec_t = qrec.view(N, n, 4) if qrec.numel() == N * n * 4 else qrec.view(N, n, 2)
        if w64 and rec_t.shape[-1] == 4:
            # MODE 4 reads 8-byte records = words 1-2 of the 16-byte ones (one copy per factory call; direct callers only)
            rec_t = rec_t[:, :, 1:3].clone(memory_format=torch.contiguous_format)   # own storage: 8-byte aligned
            REC8_COPIES[0] += 1
    dqw = d_q_w.view(-1)
    part = torch.empty(nsl * dqw.numel(), device=bank.device, dtype=torch.float32) if nsl > 1 else dqw
    gr_conv = grad_readings.dtype != torch.float32
    gr = grad_readings if not gr_conv else torch.empty(grad_readings.shape, device=grad_readings.device,
                                                       dtype=torch.float32)
    mod = _tail_module() if tail else _module()
    # the entry, once per factory call: bf16 / bf16 rows below 2^32 elements in the regular module take the bf16 one
    gfn = mod.gx_gds_bf if not (tail or w64) else mod.gx_gds
    dmod = da2._module() if nsl > 1 else None
    LAST_POLICY.clear()
    # PDL is opt-in PER LAUNCH by the caller (launch(ti, pdl=True)), which vouches that the kernel right before this
    # launch writes none of the kernel's pre-wait inputs (tile bounds, records, grad_readings rows): the backward step
    # (gwr -> baddbmm -> PA -> GDS) does; direct calls (tests) get plain launches. Only possible with tensor records (a
    # provider's records may come from the kernel right before) and f32 readings (else gr is converted by the launch
    # right before), and only on the modern backend (the portable side has no griddepcontrol).
    pdl_ok = provider is None and not gr_conv and device_pdl_ok(bank.device)
    LAST_POLICY.update(impl="gx", nw=nw, ppw=ppw, K=K, nsl=nsl,
                       pdl_capable=pdl_ok,
                       records="provider" if provider is not None else ("qrec" if rec_t.shape[-1] == 4 else "rec8"),
                       factory_records=from_provider, w64=w64,
                       rows=(str(bank.dtype).split(".")[-1], str(grad_bank.dtype).split(".")[-1]))
    release = getattr(provider, "release", None) if provider is not None else None

    def launch(ti: int, pdl: bool = False) -> None:
        if gr_conv:
            gr.copy_(grad_readings)
        if provider is not None:
            recs, ev = provider.get(ti)
            cur = torch.cuda.current_stream(bank.device)
            if ev is not None:
                cur.wait_event(ev)
            # The records are used on this stream (whatever stream produced them, event or not): the caching
            # allocator must not hand their storage out again before this launch completed, even if the provider
            # drops its reference. (A no-op when they were allocated on this stream, e.g. under graph capture.)
            if recs.is_cuda:
                recs.record_stream(cur)
            recs = recs if recs.dim() == 3 else recs.view(recs.shape[0], n, -1)
            if w64 and recs.shape[-1] != 2:
                raise ValueError("make_gds: this call's 64-bit-offset GDS reads 8-byte records; the provider returned "
                                 f"{tuple(recs.shape)}")
        else:
            recs = rec_t
        use_pdl = 1 if (pdl and pdl_ok) else 0
        gfn(bank, grad_bank, gr, gr_rows, recs, spay2, tb, part, ti, num_parallel, chunk_size, r, K, nw, use_pdl)
        if release is not None:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(bank.device))
            release(ti, done)
        if nsl > 1 and ti == 0:
            dmod.da2_dot_reduce(part, dqw, nsl)

    def plan(nsteps: int, pdl: bool = False):
        """(plan, launch address, use) of steps 0 .. nsteps - 1 for bwd_steps_cuda, or None when a step's launch is more than
        one gx_gds launch; use: 1 = PDL. done() runs launch(0)'s slice-plane reduce."""
        if provider is not None or gr_conv or not 1 <= int(nsteps) <= lim:
            return None
        p = mod.gx_gds_plan(bank, grad_bank, gr, gr_rows, rec_t, spay2, tb, part, int(nsteps) - 1, num_parallel,
                            chunk_size, r, K, nw, gfn is not mod.gx_gds)   # the entry each launch() runs
        return p, int(mod.gx_gds_launch_addr()), (1 if (pdl and pdl_ok) else 0)

    def done() -> None:
        if nsl > 1:
            dmod.da2_dot_reduce(part, dqw, nsl)

    launch.geom = dict(nw=nw, ppw=ppw, K=K, nsl=nsl, impl="gx", pdl_capable=pdl_ok)
    launch.pdl_capable = pdl_ok
    launch.join = None   # nothing to join (callers test for it)
    launch.plan = plan
    launch.done = done
    return launch
