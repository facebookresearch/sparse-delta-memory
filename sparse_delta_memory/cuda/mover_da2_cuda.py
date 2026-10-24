# Copyright (c) Meta Platforms, Inc. and affiliates.
"""mover_da2_cuda: the deterministic Phase-2 movers that walk Phase 1's sorted entries: DW (the forward's decay and
write) and GDS (the read-side backward).

DW computes, per slot group,

  bank[slot] = fma(base, cpe_last, acc),  acc = fma(sc_m, dv[tok_m], acc) over the group's members in
  Phase-1 sorted order starting from 0, sc = __fmul_rn(f32(k_val), expf(rex)), base = the pre-write bank row,

the arithmetic of 2.0.0's Triton DW (phase2_kernels), its bitwise reference then. The walk keeps global latency off its
critical path:
  * Slot rows (base) are prefetched into a per-warp shared-memory ring of RS rows in group order (per-lane cp.async;
    TMA bulk copies for runs of a dense buffer): a batch start fills the ring, and every completed group issues the
    next pending one. The walk only waits for copies that have long completed.
  * Records (krec, Phase 1's record format: {spay element, absolute slot, f32 scale, f32 cpe}) are one 16-byte load
    per lane per 32 positions, three batches ahead, broadcast by shuffles.
  * Token rows: one bulk copy per token row slice into shared memory.
  * The walk is unrolled over 32 positions and has no shared-memory stores and no memory clobbers, so the
    compiler can hoist the token-row loads of later positions over the (uniform) FIRST / LAST branches.
Tiles: warp tile t of a chunk = [tb[t], tb[t+1]) with tb[t] the group FIRST nearest to t*n/TU (ties: back).
Every group lies in one tile and is summed in sorted order by one warp, so the output does not depend on the
tiling (TU, warps per CTA, CTAs per chunk, ring depth). The geometry comes from shapes only.

GDS (da2_gds2) computes the read entries' dots in token order (no atomics) and scatters the bank gradient in sorted
order, one single-rounding read-modify-write per (slot, element) and launch (a single writer per element). Both roles
run in one fused launch (8 warps per block) or in two launches.

The saving forward (dense compact snapshot) and the no-grad forward hand their DW to mover_fx_cuda's bank-direct kernel
when it supports the shape (see make_dw)."""

import functools

import torch

from ._build import load_cuda_extension
from ._device import backend, smem_cap

_CPP = r"""
void da2_krec(at::Tensor kpay, at::Tensor k_idx, at::Tensor k_val, at::Tensor rex, at::Tensor cpe,
              at::Tensor krec);
void da2_tiles(at::Tensor kpay, at::Tensor tb);
void da2_dw(at::Tensor bank, at::Tensor dv, int64_t dv_step_rows, at::Tensor krec, at::Tensor tb,
            int64_t ti, int64_t P, int64_t C, int64_t K, int64_t nw, int64_t rs,
            at::Tensor snapc, at::Tensor segbuf, at::Tensor segb, int64_t sb_step, int64_t sf_full);
void da2_segb(at::Tensor kpay, at::Tensor tb, at::Tensor segb);
void da2_qrec(at::Tensor qpay, at::Tensor q_idx, at::Tensor q_val, at::Tensor logq, at::Tensor qrec);
void da2_dot_reduce(at::Tensor part, at::Tensor out, int64_t nsl);
void da2_gds2(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor qrec, at::Tensor tb,
              at::Tensor q_idx, at::Tensor dots, int64_t ti, int64_t valid, int64_t P, int64_t C, int64_t R, int64_t K,
              int64_t nw, int64_t which, int64_t tpw);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

namespace {

constexpr uint32_t kFirst = 0x80000000u;
constexpr uint32_t kLast = 0x40000000u;
constexpr uint32_t kInvalid = 0x20000u;
constexpr int kDistShift = 18;
constexpr uint32_t kDistMax = 4095u;
constexpr unsigned kFull = 0xFFFFFFFFu;

__device__ __forceinline__ float f_of(float v) { return v; }
__device__ __forceinline__ float f_of(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// These kernels need sm_90+ (TMA bulk copies, mbarrier transaction counts, PDL); a JIT build for an arch list that
// also names older GPUs (8.0, mixed A100/H100) compiles these helpers to traps there, and the launchers never run on
// cc < 9 (supported() gates on the device's capability).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#define DA2_PRE_SM90 1
#define DA2_SM90_ONLY(...) __trap()
#else
#define DA2_SM90_ONLY(...) __VA_ARGS__
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    DA2_SM90_ONLY(asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count)
                               : "memory"));
}
__device__ __forceinline__ void fence_mbar_init() {
    DA2_SM90_ONLY(asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"));
}
__device__ __forceinline__ void fence_proxy_async() {
    DA2_SM90_ONLY(asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"));
}
// One arrival that also announces `bytes` of transactions (count-1 barriers: the phase completes
// when this arrival happened AND every announced byte landed, in either order).
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* bar, uint32_t bytes) {
    DA2_SM90_ONLY(asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
                               "r"(bytes) : "memory"));
}
// TMA bulk copy global -> this CTA's shared memory, completion counted on `bar`.
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    DA2_SM90_ONLY(asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory"));
}
#define DA2_SC_MINB 3       // min resident blocks per SM for the GDS scatter (registers <= 64K / (3 * 256))
#define DA2_DOTS_MINB 3     // min resident blocks per SM for the token-order dots (latency-bound gather)
constexpr int kRecPad = 40;  // record batch slots per warp: 32 + prefetch overrun (sentinels)
template <int BY>
__device__ __forceinline__ void cp_async_lane(void* dst, const void* src, int src_bytes) {
    if constexpr (BY == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                     "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_u32(dst)), "l"(src),
                     "n"(BY), "r"(src_bytes));
    }
}
// Per-lane cp.async rings complete through commit groups (per thread, FIFO): a row's copy is one group; waiting for
// row q while `issued` rows were committed = wait until at most issued - 1 - q groups are pending (a form that
// compute-sanitizer's racecheck models).
__device__ __forceinline__ void cp_async_commit() {
    DA2_SM90_ONLY(asm volatile("cp.async.commit_group;\n" ::: "memory"));
}
template <int N>
__device__ __forceinline__ void cp_async_wait_n() {
    DA2_SM90_ONLY(asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"));
}
__device__ __forceinline__ void cp_async_wait_upto(int n) {   // 0 <= n; n >= 31 waits for at most 31 pending
    switch (n) {
        case 0: cp_async_wait_n<0>(); break;   case 1: cp_async_wait_n<1>(); break;
        case 2: cp_async_wait_n<2>(); break;   case 3: cp_async_wait_n<3>(); break;
        case 4: cp_async_wait_n<4>(); break;   case 5: cp_async_wait_n<5>(); break;
        case 6: cp_async_wait_n<6>(); break;   case 7: cp_async_wait_n<7>(); break;
        case 8: cp_async_wait_n<8>(); break;   case 9: cp_async_wait_n<9>(); break;
        case 10: cp_async_wait_n<10>(); break; case 11: cp_async_wait_n<11>(); break;
        case 12: cp_async_wait_n<12>(); break; case 13: cp_async_wait_n<13>(); break;
        case 14: cp_async_wait_n<14>(); break; case 15: cp_async_wait_n<15>(); break;
        case 16: cp_async_wait_n<16>(); break; case 17: cp_async_wait_n<17>(); break;
        case 18: cp_async_wait_n<18>(); break; case 19: cp_async_wait_n<19>(); break;
        case 20: cp_async_wait_n<20>(); break; case 21: cp_async_wait_n<21>(); break;
        case 22: cp_async_wait_n<22>(); break; case 23: cp_async_wait_n<23>(); break;
        case 24: cp_async_wait_n<24>(); break; case 25: cp_async_wait_n<25>(); break;
        case 26: cp_async_wait_n<26>(); break; case 27: cp_async_wait_n<27>(); break;
        case 28: cp_async_wait_n<28>(); break; case 29: cp_async_wait_n<29>(); break;
        case 30: cp_async_wait_n<30>(); break; default: cp_async_wait_n<31>(); break;
    }
}
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint64_t* bar) {
    DA2_SM90_ONLY(asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar))));
}
// Spins until the phase of parity `parity` of `bar` completed; returns 0. Deliberately NO memory
// clobber: the caller adds the result to the address of every read that must follow the wait, so
// only those reads are ordered after it (acquire semantics of try_wait) and everything else stays
// free to move.
__device__ __forceinline__ uint32_t mbar_wait_dep(uint64_t* bar, uint32_t parity) {
#ifdef DA2_PRE_SM90
    __trap();
    return 0u;
#else
    uint32_t ok;
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " DA2_WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " @!p bra DA2_WAIT;\n"
        " mov.u32 %0, 0;\n"
        "}\n"
        : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity));
    return ok;
#endif
}

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
template <typename T> __device__ __forceinline__ T make_zero_row();
template <> __device__ __forceinline__ uint2 make_zero_row<uint2>() { return make_uint2(0u, 0u); }
template <> __device__ __forceinline__ float4 make_zero_row<float4>() {
    return make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ __forceinline__ uint4 ld_rec(const uint4* __restrict__ kr, int pos, int e) {
    if (pos < e) return __ldg(kr + pos);
    return make_uint4(kFirst | kLast | kInvalid, 0u, 0u, 0x3F800000u);   // padding: own group
}
// Valid group FIRSTs of a record batch (one bit per lane).
__device__ __forceinline__ uint32_t first_mask(const uint4& r) {
    return __ballot_sync(kFull, (r.x & (kFirst | kInvalid)) == kFirst);
}

// ---------------------------------------------------------------------------------------------
// Records (Phase-1 record contract): per sorted write position {spay_k element, absolute slot,
// __fmul_rn(f32(k_val), expf(rex)), f32(cpe)}; padding: slot 0, scale 0, cpe 1.
template <typename IdxT, typename ValT>
__global__ void __launch_bounds__(256) da2_krec_kernel(
    const int32_t* __restrict__ kpay, const IdxT* __restrict__ k_idx, const ValT* __restrict__ kv,
    const float* __restrict__ rex, const ValT* __restrict__ cpe, uint4* __restrict__ krec,
    int64_t total, int n, int W) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t chunk = i / n;
    const uint32_t pay = static_cast<uint32_t>(kpay[i]);
    uint32_t slot = 0u, sc = 0u, c = 0x3F800000u;
    if (!(pay & kInvalid)) {
        const int64_t o = chunk * n + static_cast<int64_t>((pay >> 9) & 0xFF) * W + (pay & 0xFF);
        slot = static_cast<uint32_t>(static_cast<int32_t>(k_idx[o]));
        sc = __float_as_uint(__fmul_rn(f_of(kv[o]), expf(rex[o])));
        c = __float_as_uint(f_of(cpe[o]));
    }
    krec[i] = make_uint4(pay, slot, sc, c);
}

// Tile bounds: tb[chunk][t] = the group FIRST nearest to floor(t * n / TU) (ties: the earlier one),
// t = 0..TU (tb[0] = 0, tb[TU] = n). Monotone in t, so tiles are well formed (possibly empty).
__global__ void __launch_bounds__(256) da2_tiles_kernel(
    const int32_t* __restrict__ kpay, int32_t* __restrict__ tb, int64_t total, int n, int TU) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t chunk = i / (TU + 1);
    const int t = static_cast<int>(i % (TU + 1));
    const int j = static_cast<int>(static_cast<int64_t>(t) * n / TU);
    const int32_t* kp = kpay + chunk * n;
    int r;
    if (j <= 0) {
        r = 0;
    } else if (j >= n) {
        r = n;
    } else {
        const uint32_t pj = static_cast<uint32_t>(kp[j]);
        if (pj & kFirst) {
            r = j;
        } else {
            const uint32_t d = (pj >> kDistShift) & kDistMax;
            int back;
            if (d < kDistMax) {
                back = j - static_cast<int>(d);
            } else {
                back = j - 1;
                while (back > 0 && !(static_cast<uint32_t>(kp[back]) & kFirst)) --back;
            }
            int fwd = j + 1;
            while (fwd < n && !(static_cast<uint32_t>(kp[fwd]) & kFirst)) ++fwd;
            r = (fwd - j < j - back) ? fwd : back;
        }
    }
    tb[i] = r;
}

// Segment ordinals (dense walk): segb[chunk][t] = valid group FIRSTs before tb[chunk][t], t = 0..TU
// (segb[TU] = the chunk's group count). One block per chunk.
__global__ void __launch_bounds__(256) da2_segb_kernel(
    const int32_t* __restrict__ kpay, const int32_t* __restrict__ tb, int32_t* __restrict__ segb, int n, int TU) {
    extern __shared__ int32_t cnt_s[];
    const int64_t chunk = blockIdx.x;
    const int32_t* kp = kpay + chunk * n;
    const int32_t* tbc = tb + chunk * (TU + 1);
    for (int t = threadIdx.x; t < TU; t += blockDim.x) {
        const int s = tbc[t], e = tbc[t + 1];
        int c = 0;
        for (int j = s; j < e; ++j) c += (static_cast<uint32_t>(__ldg(kp + j)) & (kFirst | kInvalid)) == kFirst;
        cnt_s[t] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int t = 0; t < TU; ++t) {
            const int c = cnt_s[t];
            cnt_s[t] = acc;
            acc += c;
        }
        cnt_s[TU] = acc;
    }
    __syncthreads();
    for (int t = threadIdx.x; t <= TU; t += blockDim.x) segb[chunk * (TU + 1) + t] = cnt_s[t];
}

// Read records (Phase-1 record contract): per sorted read position {spay_q element, absolute slot,
// __fmul_rn(f32(q_val), expf(log_q)), 0}; padding: slot 0, scale 0.
template <typename IdxT, typename ValT>
__global__ void __launch_bounds__(256) da2_qrec_kernel(
    const int32_t* __restrict__ qpay, const IdxT* __restrict__ q_idx, const ValT* __restrict__ qv,
    const float* __restrict__ logq, uint4* __restrict__ qrec, int64_t total, int n, int R) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t chunk = i / n;
    const uint32_t pay = static_cast<uint32_t>(qpay[i]);
    uint32_t slot = 0u, sc = 0u;
    if (!(pay & kInvalid)) {
        const int64_t o = chunk * n + static_cast<int64_t>((pay >> 9) & 0xFF) * R + (pay & 0xFF);
        slot = static_cast<uint32_t>(static_cast<int32_t>(q_idx[o]));
        sc = __float_as_uint(__fmul_rn(f_of(qv[o]), expf(logq[o])));
    }
    qrec[i] = make_uint4(pay, slot, sc, 0u);
}

// d_q_w[e] = ((part[0][e] + part[1][e]) + part[2][e]) + ...: the column slices of a wide head, in order.
__global__ void __launch_bounds__(256) da2_dot_reduce_kernel(
    const float* __restrict__ part, float* __restrict__ out, int64_t total, int NSL) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float s = part[i];
    for (int k = 1; k < NSL; ++k) s = __fadd_rn(s, part[static_cast<int64_t>(k) * total + i]);
    out[i] = s;
}

// ---------------------------------------------------------------------------------------------
// DW. Grid: P * NSL * K blocks (block b: cr = b % K, slice = b / K % NSL, p = rest); warp w of
// block cr walks tile cr * NW + w of its chunk. Shared memory: token rows [C][SW] f32 | rings
// [NW][RS][RB] | record batches [NW][32] uint4 | mbarriers [1 + NW * RS].
// Token rows [C, SW] of a slice into shared memory (row stride SW floats), all completing on tok_bar (its expected
// bytes are set here): ONE bulk copy when the slice is the whole row (SW == hd, the C rows are contiguous), else one
// copy per row. (Per-row copies issued by divergent threads serialize through an elect loop: avoid when possible.)
__device__ __forceinline__ void stage_tok_rows(float* tok_s, const float* rows, int C, int SW, int hd,
                                               uint64_t* tok_bar, int nthreads) {
    if (threadIdx.x == 0) mbar_arrive_tx(tok_bar, static_cast<uint32_t>(C * SW * 4));
    if (SW == hd) {
        if (threadIdx.x == 0) bulk_g2s(tok_s, rows, static_cast<uint32_t>(C * SW * 4), tok_bar);
    } else {
        for (int t = threadIdx.x; t < C; t += nthreads) {
            bulk_g2s(tok_s + t * SW, rows + static_cast<int64_t>(t) * hd, static_cast<uint32_t>(SW * 4), tok_bar);
        }
    }
}

struct Geo {
    int P, C, n, hd, NSL, SW, K, TU, tok_bytes;
    int sf_full;   // SF source layout: 0 slice-major dense buffer, 1 full rows chunk * n + g (dense snapc)
};

template <typename E, int NW, int RS, int MODE = 0>
__global__ void __launch_bounds__(NW * 32, (NW <= 8 ? 2 : 1)) da2_dw_kernel(
    E* __restrict__ bank, const float* __restrict__ dv, int64_t dv_step_rows,
    const uint4* __restrict__ krec, const int32_t* __restrict__ tb, int ti, Geo g,
    long long* __restrict__ stats, E* __restrict__ snapc, const E* __restrict__ segbuf,
    const int32_t* __restrict__ segb, int sb_step) {
    // MODE & 32 (SF, the dense walk): the tile's base rows are rows sb_chunk * n + segb[tile] + q of the dense buffer
    // segbuf (q = the group's index in the tile): issue order = group order, no records.
    constexpr bool SF = (MODE & 32) != 0;
    // The dense walk streams its rows in TMA bulk RUNS; the per-row ring (no SF) is cp.async.
    // CPR: the cp.async per-row ring is the active ring.
    constexpr bool CPR = !SF;
    // snapc: compact snapshot rows chunk * n + j_FIRST <- the group's raw pre-write bank row, or null (the launchers
    // pass null: no row is written).
    using RowT = typename Row4<E>::T;
    static_assert((RS & (RS - 1)) == 0 && RS <= 32, "RS: power of two <= 32");
    extern __shared__ __align__(128) unsigned char smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int b = blockIdx.x;
    const int cr = b % g.K;
    b /= g.K;
    const int sl = b % g.NSL;
    const int p = b / g.NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * g.P + p;
    const int SW = g.SW, C = g.C, hd = g.hd;
    constexpr int BY = 4 * static_cast<int>(sizeof(E));                // bytes of a lane's 4 columns
    const int RB = SW * static_cast<int>(sizeof(E));                   // row slice bytes (16 | RB)
    const int RBS = CPR ? 32 * BY : RB;                                // ring row stride
    float* tok_s = reinterpret_cast<float*>(smem);
    unsigned char* rings = smem + g.tok_bytes;
    unsigned char* ring = rings + static_cast<size_t>(warp) * RS * RBS;
    uint4* rec_s = reinterpret_cast<uint4*>(rings + static_cast<size_t>(NW) * RS * RBS) + warp * kRecPad;
    uint64_t* bars = reinterpret_cast<uint64_t*>(rings + static_cast<size_t>(NW) * RS * RBS +
                                                 static_cast<size_t>(NW) * kRecPad * sizeof(uint4));
    if (lane < kRecPad - 32) rec_s[32 + lane] = make_uint4(0u, 0u, 0u, 0u);
    uint64_t* tok_bar = bars;
    uint64_t* my_bars = bars + 1 + warp * RS;
    if (threadIdx.x == 0) mbar_init(tok_bar, 1);
    if (lane < RS) mbar_init(my_bars + lane, CPR ? 32u : 1u);
    fence_mbar_init();
    __syncthreads();

    {   // Token rows of the slice, all on tok_bar.
        const float* rows = dv + (static_cast<int64_t>(ti) * dv_step_rows +
                                  static_cast<int64_t>(p) * C) * hd + static_cast<int64_t>(sl) * SW;
        stage_tok_rows(tok_s, rows, C, SW, hd, tok_bar, NW * 32);
    }

    const int32_t* tbc = tb + chunk * (g.TU + 1);
    const int tile = cr * NW + warp;
    const int s = __ldg(tbc + tile), e = __ldg(tbc + tile + 1);
    const uint4* kr = krec + chunk * g.n;
    const int c4 = 4 * lane;
    const bool on = c4 < SW && sl * SW + c4 < hd;
    const int coff = on ? c4 : 0;                                       // safe column for loads
    const int ring_off = CPR ? lane * BY : coff * static_cast<int>(sizeof(E));
    E* const bank_col = bank + static_cast<int64_t>(sl) * SW + c4;      // stores only when `on`
    const E* const bank_src = bank + static_cast<int64_t>(sl) * SW;     // slice start of a row

    uint4 r0 = ld_rec(kr, s + lane, e);
    uint4 r1 = ld_rec(kr, s + 32 + lane, e);
    uint4 r2 = ld_rec(kr, s + 64 + lane, e);
    uint4 r3 = ld_rec(kr, s + 96 + lane, e);
    // Not-yet-issued valid FIRSTs of the batches held in r0, r1, r2 (issue order = segment order).
    uint32_t rm0 = 0u, rm1 = 0u, rm2 = 0u;
    if constexpr (!SF) {
        rm0 = first_mask(r0);
        rm1 = first_mask(r1);
        rm2 = first_mask(r2);
    }
    int done = 0;      // segments of the tile completed (emitted)
    int issued = 0;    // segments whose slot row was issued (always >= done)
    int nseg = 0;      // SF: the tile's group count
    const E* sf_src = nullptr;
    if constexpr (SF) {
        const int64_t tix = chunk * (g.TU + 1) + tile;
        const int r0_ = __ldg(segb + tix);
        nseg = __ldg(segb + tix + 1) - r0_;
        const int64_t sbc = static_cast<int64_t>(ti) * sb_step + p;
        sf_src = g.sf_full ? segbuf + (sbc * g.n + r0_) * static_cast<int64_t>(hd) + static_cast<int64_t>(sl) * SW
                           : segbuf + ((sbc * g.NSL + sl) * g.n + r0_) * static_cast<int64_t>(SW);
    }
    const int sf_rs = g.sf_full ? hd : SW;   // SF row stride (elements); runs need contiguous rows (sf_rs == SW)
    // SF: rows go in RUNS (one bulk copy per contiguous run of groups, split at the ring end), one mbarrier per
    // run (at its first slot). run_mask: slots where an issued, unwaited run starts; ph_mask: per-slot barrier
    // phase; rows <= ready_until are present.
    constexpr bool RUNS = SF;
    uint32_t run_mask = 0u, ph_mask = 0u;
    int ready_until = -1;
    auto issue_runs = [&](int lim) {   // warp-uniform: rows [issued, lim) (lim <= done + RS)
        while (issued < lim) {
            const int s0 = issued & (RS - 1);
            const int k = sf_rs == SW ? min(lim - issued, RS - s0) : 1;
            if (lane == 0) {
                mbar_arrive_tx(my_bars + s0, static_cast<uint32_t>(k * RB));
                bulk_g2s(ring + s0 * RBS, sf_src + static_cast<int64_t>(issued) * sf_rs, static_cast<uint32_t>(k * RB),
                         my_bars + s0);
            }
            run_mask |= 1u << s0;
            issued += k;
        }
    };
    // zrun: the sum of every run wait's dependency token so far (all 0 at run time). Ring reads of rows covered
    // by waited runs add zrun to their address, so none can be scheduled before the wait of its run.
    uint32_t zrun = 0u;
    auto wait_run = [&](int q) -> uint32_t {   // q: a run start > ready_until; returns the updated zrun
        const int s0 = q & (RS - 1);
        zrun += mbar_wait_dep(my_bars + s0, (ph_mask >> s0) & 1u);
        ph_mask ^= 1u << s0;
        run_mask &= ~(1u << s0);
        const uint64_t m2 = static_cast<uint64_t>(run_mask) | (static_cast<uint64_t>(run_mask) << RS);
        const uint32_t rot = static_cast<uint32_t>(m2 >> (s0 + 1)) & static_cast<uint32_t>((1ull << RS) - 1ull);
        ready_until = (rot ? q + __ffs(rot) : issued) - 1;
        return zrun;
    };
    auto sf_issue = [&](int q) {   // SF: group q's base row (warp-uniform call; TMA: lane 0 issues)
        const int ri = q & (RS - 1);
        const E* row = sf_src + static_cast<int64_t>(q) * sf_rs;
        if constexpr (CPR) {
            __syncwarp();   // slot handoff: every lane's reads of this slot's previous row precede the refill
            cp_async_lane<BY>(ring + ri * RBS + lane * BY, on ? row + c4 : row, on ? BY : 0);
            cp_async_commit();
        } else {
            if (lane == 0) {
                mbar_arrive_tx(my_bars + ri, static_cast<uint32_t>(RB));
                bulk_g2s(ring + ri * RBS, row, static_cast<uint32_t>(RB), my_bars + ri);
            }
        }
    };

    // Source row of a group: bank[slot] (key = slot).
    const E* const row_src = bank_src;
    int bpos = s;                 // sorted position of r0's first lane (r1: + 32, r2: + 64)
    auto issue_seg = [&](int q, int key) {   // TMA: the calling lane issues group q's base row
        const int ri = q & (RS - 1);
        uint64_t* bar = my_bars + ri;
        mbar_arrive_tx(bar, static_cast<uint32_t>(RB));
        bulk_g2s(ring + ri * RBS, row_src + static_cast<int64_t>(key) * hd, static_cast<uint32_t>(RB), bar);
    };
    auto issue_all = [&](int q, int key) {   // LDGSTS: every lane copies its 4 columns (warp-uniform key)
        const int ri = q & (RS - 1);
        const E* row = row_src + static_cast<int64_t>(key) * hd;
        __syncwarp();   // slot handoff: every lane's reads of this slot's previous row precede the refill
        cp_async_lane<BY>(ring + ri * RBS + lane * BY, on ? row + c4 : row, on ? BY : 0);
        cp_async_commit();
    };
    // Issue the next FIRSTs of batch mask `m` (records r, first lane at sorted position p0) while the ring has room.
    auto issue_from = [&](uint32_t& m, const uint4& r, int p0) {
        const int avail = done + RS - issued;
        if (avail <= 0 || m == 0u) return;
        const uint32_t lt = (1u << lane) - 1u;
        const bool mine = ((m >> lane) & 1u) && __popc(m & lt) < avail;
        const uint32_t tm = __ballot_sync(kFull, mine);
        if constexpr (CPR) {
            uint32_t t = tm;
            int q = issued;
            while (t) {
                const int li = __ffs(t) - 1;
                t &= t - 1u;
                issue_all(q++, __shfl_sync(kFull, static_cast<int32_t>(r.y), li));
            }
        } else {
            if (mine) issue_seg(issued + __popc(tm & lt), static_cast<int32_t>(r.y));
        }
        issued += __popc(tm);
        m &= ~tm;
    };
    // After a group completed (a ring slot freed): issue the next pending group, if its record is held.
    auto pump = [&]() {
        if (issued >= done + RS) return;
        const uint32_t mm = rm0 ? rm0 : (rm1 ? rm1 : rm2);
        if (mm == 0u) return;
        const int jb = rm0 ? 0 : (rm1 ? 1 : 2);
        const int li = __ffs(mm) - 1;
        const int32_t sl_ = static_cast<int32_t>(jb == 0 ? r0.y : (jb == 1 ? r1.y : r2.y));
        if constexpr (CPR) {
            issue_all(issued, __shfl_sync(kFull, sl_, li));
        } else {
            fence_proxy_async();
            __syncwarp();
            if (lane == li) issue_seg(issued, sl_);
        }
        const uint32_t bit = 1u << li;
        if (jb == 0) rm0 &= ~bit; else if (jb == 1) rm1 &= ~bit; else rm2 &= ~bit;
        ++issued;
    };
    auto top_up = [&]() {
        if constexpr (RUNS) {
            issue_runs(min(done + RS, nseg));
        } else if constexpr (SF) {
            const int lim = min(done + RS, nseg);
            while (issued < lim) sf_issue(issued++);
        } else {
            issue_from(rm0, r0, bpos);
            if (rm0 == 0u) issue_from(rm1, r1, bpos + 32);
            if (rm0 == 0u && rm1 == 0u) issue_from(rm2, r2, bpos + 64);
        }
    };
    auto sf_pump = [&]() {
        if constexpr (RUNS) {
            const int lim = min(done + RS, nseg);
            if (lim - issued >= RS / 2 || (lim == nseg && lim > issued)) {
                fence_proxy_async();
                __syncwarp();
                issue_runs(lim);
            }
        } else if (issued < min(done + RS, nseg)) {
            if constexpr (!CPR) {
                fence_proxy_async();
                __syncwarp();
            }
            sf_issue(issued++);
        }
    };
    __syncwarp();
    top_up();
    const uint32_t ztok = mbar_wait_dep(tok_bar, 0u);
    const float* tok = tok_s + ztok + coff;

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    int jf = s;                                   // position of the current group's FIRST
    E* const snap_row0 = snapc == nullptr ? nullptr : snapc + chunk * static_cast<int64_t>(g.n) * hd +
                                                       static_cast<int64_t>(sl) * SW + c4;
    const int nb = (e - s + 31) >> 5;
    for (int bi = 0; bi < nb; ++bi) {
        // The previous batch's walk is done with rec_s / ring. TMA ring: every lane orders its generic reads
        // of ring slots before the async-proxy refill (fence, THEN the warp sync that hands slots over).
        if constexpr (!CPR) fence_proxy_async();
        __syncwarp();
        if (bi > 0) {
            bpos = s + 32 * bi;
            r0 = r1;
            r1 = r2;
            r2 = r3;
            r3 = ld_rec(kr, s + 32 * (bi + 3) + lane, e);
            if constexpr (!SF) {
                rm0 = rm1;
                rm1 = rm2;
                rm2 = first_mask(r2);
            }
            top_up();
        }
        // Per-batch precompute, vectorized over the lanes (lane i = position s + 32 * bi + i): the token row
        // offset and the uniform FIRST / LAST masks, so the walk has no per-position decode and only uniform
        // branches. rec_s[i] = {token row offset (floats), scale, slot, cpe}.
        const uint32_t vfirst = __ballot_sync(kFull, (r0.x & (kFirst | kInvalid)) == kFirst);
        const uint32_t vlast = __ballot_sync(kFull, (r0.x & (kLast | kInvalid)) == kLast);
        // SF: no FIRST test. Groups (valid ones and the invalid entries' own groups) tile the walk contiguously,
        // so the accumulator restarts right after EVERY LAST (vend), valid or not (an invalid entry's product,
        // even NaN from a non-finite token row, never reaches the next group); only valid LASTs emit (vlast).
        const uint32_t vend = SF ? __ballot_sync(kFull, (r0.x & kLast) != 0u) : 0u;
        if constexpr (SF) {
            rec_s[lane] = make_uint4(static_cast<uint32_t>(min(static_cast<int>((r0.x >> 9) & 0xFFu), C - 1) * SW),
                                     r0.z, r0.y, r0.w);
        } else {
            rec_s[lane] = make_uint4(static_cast<uint32_t>(min(static_cast<int>((r0.x >> 9) & 0xFFu), C - 1) * SW),
                                     r0.z, r0.y, r0.w);
        }
        // Rows of the groups this batch completes: [done, done + nlast); the ones already issued
        // are waited for here, so the walk itself waits only for a group issued late (slow path).
        const int nlast = __popc(vlast);
        const int wend = min(done + nlast, issued);
        uint32_t zr = 0u;
        if constexpr (RUNS) {
            for (int q = max(done, ready_until + 1); q < wend; q = ready_until + 1) (void)wait_run(q);
            zr = zrun;
        } else {
            if constexpr (CPR) {
                if (wend > done) cp_async_wait_upto(issued - wend);   // rows [done, wend) landed (FIFO groups)
            } else {
                for (int q = done; q < wend; ++q) zr += mbar_wait_dep(my_bars + (q & (RS - 1)),
                                                                      static_cast<uint32_t>(q / RS) & 1u);
            }
        }
        __syncwarp();
        const uint4* rq = rec_s + zr;
        RowT nbase = make_zero_row<RowT>();
        if (!RUNS && done < wend) {
            nbase = *reinterpret_cast<const RowT*>(ring + (done & (RS - 1)) * RBS + zr + ring_off);
        }
        const int pos0 = s + 32 * bi;
        const int kend = min(32, e - pos0);            // positions of this batch inside the tile
#pragma unroll 1
        for (int k8 = 0; k8 < kend; k8 += 8) {
            // All 8 positions' records and token rows first (16 independent shared loads in flight).
            uint2 rr[8];
            float4 xx[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) rr[j] = *reinterpret_cast<const uint2*>(rq + k8 + j);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                xx[j] = *reinterpret_cast<const float4*>(tok + rr[j].x);
            }
            const uint32_t f8 = vfirst >> k8, l8 = vlast >> k8, e8 = vend >> k8;
            // RUNS (SF): every row this sub-batch emits is made present ONCE here (issue + wait, one code site;
            // normally already covered by the batch-start wait), so the 8 unrolled emits carry no slow path.
            uint32_t zs = 0u;
            if constexpr (RUNS) {
                const int nl = __popc(l8 & 0xFFu);
                if (nl > 0 && done + nl - 1 > ready_until) {
                    if (done + nl > issued) {
                        fence_proxy_async();
                        __syncwarp();
                        issue_runs(min(done + RS, nseg));
                    }
                    while (ready_until < done + nl - 1) (void)wait_run(ready_until + 1);
                }
                zs = zrun;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (!SF && (f8 & (1u << j))) {
                    acc = make_float4(0.f, 0.f, 0.f, 0.f);
                    jf = pos0 + k8 + j;
                    if (done >= issued) {
                        // Slow path (the ring was full at the batch start): issue this group's row now.
                        const int32_t key_f = static_cast<int32_t>(rq[k8 + j].z);
                        if constexpr (CPR) {
                            issue_all(done, key_f);
                        } else {
                            fence_proxy_async();
                            __syncwarp();
                            if (lane == 0) issue_seg(done, key_f);
                        }
                        issued = done + 1;
                        rm0 &= ~(1u << (k8 + j));
                    }
                }
                const float sc = __uint_as_float(rr[j].y);
                acc.x = fmaf(sc, xx[j].x, acc.x);
                acc.y = fmaf(sc, xx[j].y, acc.y);
                acc.z = fmaf(sc, xx[j].z, acc.z);
                acc.w = fmaf(sc, xx[j].w, acc.w);
                if ((SF ? e8 : l8) & (1u << j)) {
                  if (!SF || (l8 & (1u << j))) {
                    const uint2 sw2 = *reinterpret_cast<const uint2*>(
                        reinterpret_cast<const unsigned char*>(rq + k8 + j) + 8);   // {slot, cpe}
                    const int slot = static_cast<int>(sw2.x);
                    const float cpe = __uint_as_float(sw2.y);
                    RowT bv = nbase;
                    if constexpr (RUNS) {
                        bv = *reinterpret_cast<const RowT*>(ring + (done & (RS - 1)) * RBS + zs + ring_off);
                    } else if (done >= wend) {   // issued in this batch (slow path): wait here
                        if constexpr (SF) {
                            if (done >= issued) {
                                if constexpr (!CPR) {
                                    fence_proxy_async();
                                    __syncwarp();
                                }
                                sf_issue(done);
                                issued = done + 1;
                            }
                        }
                        const int ri = done & (RS - 1);
                        uint32_t z = 0u;
                        if constexpr (CPR) {
                            cp_async_wait_upto(issued - done - 1);
                        } else {
                            z = mbar_wait_dep(my_bars + ri, static_cast<uint32_t>(done / RS) & 1u);
                        }
                        bv = *reinterpret_cast<const RowT*>(ring + ri * RBS + z + ring_off);
                    }
                    const float4 base = cvt4(bv);
                    float4 o;
                    o.x = fmaf(base.x, cpe, acc.x);
                    o.y = fmaf(base.y, cpe, acc.y);
                    o.z = fmaf(base.z, cpe, acc.z);
                    o.w = fmaf(base.w, cpe, acc.w);
                    if (on) st4(bank_col + static_cast<uint64_t>(static_cast<uint32_t>(slot)) * static_cast<uint32_t>(hd), o);
                    if (snap_row0 != nullptr && on) {
                        *reinterpret_cast<RowT*>(snap_row0 + static_cast<int64_t>(jf) * hd) = bv;
                    }
                    ++done;
                    if (!RUNS && done < wend) {
                        nbase = *reinterpret_cast<const RowT*>(ring + (done & (RS - 1)) * RBS + zr + ring_off);
                    }
                    if constexpr (!RUNS) {
                        if constexpr (SF) sf_pump(); else pump();
                    }
                  }
                    if constexpr (SF) {   // after any LAST: the next group starts at the next position
                        acc = make_float4(0.f, 0.f, 0.f, 0.f);
                        jf = pos0 + k8 + j + 1;
                    }
                }
            }
            if constexpr (RUNS) {   // slots freed by this sub-batch's groups: refill
                if (l8 & 0xFFu) sf_pump();
            }
        }
    }
    // Every thread observes the token rows before the CTA may exit (no copy left in flight).
    (void)mbar_wait_dep(tok_bar, 0u);
}

// ---------------------------------------------------------------------------------------------
// GDS, split by data need: (1) dots in TOKEN order, no atomics: d_q_w[t, r] = <gr[t], bank[slot]> per slice with
// the per-lane chain fma(x.w, m.w, fma(x.z, m.z, fma(x.y, m.y, x.x * m.x))) and the xor-16/8/4/2/1 reduce-scatter;
// (2) the grad scatter in SORTED order with no slot-row reads. RMW (single rounding, no subnormal flush): at each
// valid LAST, grad[slot] = round(grad[slot] + acc) by a plain load + store; the walk is the row's only writer in the
// launch, so this is race-free and deterministic. The launchers instantiate RMW = true only.
template <typename G, int NW, bool RMW>
__device__ __forceinline__ void gscatter_role(
    G* __restrict__ grad, const float* __restrict__ gr, int64_t gr_step_rows, const uint4* __restrict__ qrec,
    const int32_t* __restrict__ tb, int ti, const Geo& g, int b, unsigned char* smem) {
    using RowG = typename Row4<G>::T;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int cr = b % g.K;
    b /= g.K;
    const int sl = b % g.NSL;
    const int p = b / g.NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * g.P + p;
    const int SW = g.SW, C = g.C, hd = g.hd;
    float* tok_s = reinterpret_cast<float*>(smem);
    uint4* rec_s = reinterpret_cast<uint4*>(smem + g.tok_bytes) + warp * 32;
    uint64_t* tok_bar = reinterpret_cast<uint64_t*>(smem + g.tok_bytes + NW * 32 * sizeof(uint4));
    if (threadIdx.x == 0) mbar_init(tok_bar, 1);
    fence_mbar_init();
    __syncthreads();
    {
        const float* rows = gr + (static_cast<int64_t>(ti) * gr_step_rows + static_cast<int64_t>(p) * C) * hd +
                            static_cast<int64_t>(sl) * SW;
        stage_tok_rows(tok_s, rows, C, SW, hd, tok_bar, NW * 32);
    }
    const int32_t* tbc = tb + chunk * (g.TU + 1);
    const int tile = cr * NW + warp;
    const int s = __ldg(tbc + tile), e = __ldg(tbc + tile + 1);
    const uint4* kr = qrec + chunk * g.n;
    const int c4 = 4 * lane;
    const bool on = c4 < SW && sl * SW + c4 < hd;
    const int coff = on ? c4 : 0;
    G* const grad_col = grad + static_cast<int64_t>(sl) * SW + c4;
    // Records 4 batches ahead (a batch walks in a fraction of one global-load latency).
    uint4 r0 = ld_rec(kr, s + lane, e);
    uint4 r1 = ld_rec(kr, s + 32 + lane, e);
    uint4 r2 = ld_rec(kr, s + 64 + lane, e);
    uint4 r3 = ld_rec(kr, s + 96 + lane, e);
    const uint32_t ztok = mbar_wait_dep(tok_bar, 0u);
    const float* tok = tok_s + ztok + coff;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const int nb = (e - s + 31) >> 5;
    for (int bi = 0; bi < nb; ++bi) {
        __syncwarp();
        if (bi > 0) {
            r0 = r1;
            r1 = r2;
            r2 = r3;
            r3 = ld_rec(kr, s + 32 * (bi + 3) + lane, e);
        }
        // No FIRST test (see the DW's dense walk): acc restarts after EVERY LAST (vend, valid or invalid group); only
        // valid LASTs add to the gradient (vlast).
        const uint32_t vlast = __ballot_sync(kFull, (r0.x & (kLast | kInvalid)) == kLast);
        const uint32_t vend = __ballot_sync(kFull, (r0.x & kLast) != 0u);
        rec_s[lane] = make_uint4(static_cast<uint32_t>(min(static_cast<int>((r0.x >> 9) & 0xFFu), C - 1) * SW),
                                 r0.z, r0.y, 0u);
        if constexpr (RMW) {
            // The NEXT batch's group rows (records r1, one lane per position) to L2 a batch ahead of their
            // read-modify-write (each (slot, row) has one group per launch: no other writer, so reading early is safe).
            if ((r1.x & (kLast | kInvalid)) == kLast) {
                const char* rp = reinterpret_cast<const char*>(grad + static_cast<uint64_t>(r1.y) * static_cast<uint32_t>(hd) +
                                                               static_cast<int64_t>(sl) * SW);
                for (int off = 0; off < SW * static_cast<int>(sizeof(G)); off += 128) {
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(rp + off));
                }
            }
            if (bi == 0 && (vlast >> lane) & 1u) {   // the first batch's rows (nothing prefetched them)
                const char* rp = reinterpret_cast<const char*>(grad + static_cast<uint64_t>(r0.y) * static_cast<uint32_t>(hd) +
                                                               static_cast<int64_t>(sl) * SW);
                for (int off = 0; off < SW * static_cast<int>(sizeof(G)); off += 128) {
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(rp + off));
                }
            }
        }
        __syncwarp();
        const int pos0 = s + 32 * bi;
        const int kend = min(32, e - pos0);
        // RMW: the grad rows of a sub-batch's valid LASTs are loaded into registers one sub-batch ahead.
        RowG gc[RMW ? 8 : 1];
        auto load_g = [&](RowG (&g)[RMW ? 8 : 1], int kb) {
            if constexpr (RMW) {
                const uint32_t lb = vlast >> kb;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    g[j] = make_zero_row<RowG>();
                    if (on && ((lb >> j) & 1u)) {
                        const int slot = static_cast<int>(rec_s[kb + j].z);
                        g[j] = *reinterpret_cast<const RowG*>(
                            grad_col + static_cast<uint64_t>(static_cast<uint32_t>(slot)) * static_cast<uint32_t>(hd));
                    }
                }
            }
        };
        if constexpr (RMW) load_g(gc, 0);
#pragma unroll 1
        for (int k8 = 0; k8 < kend; k8 += 8) {
            RowG gn[RMW ? 8 : 1];
            if constexpr (RMW) {
                if (k8 + 8 < kend) load_g(gn, k8 + 8);
            }
            uint2 rr[8];
            float4 xx[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) rr[j] = *reinterpret_cast<const uint2*>(rec_s + k8 + j);
#pragma unroll
            for (int j = 0; j < 8; ++j) xx[j] = *reinterpret_cast<const float4*>(tok + rr[j].x);
            const uint32_t l8 = vlast >> k8, e8 = vend >> k8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float sc = __uint_as_float(rr[j].y);
                acc.x = fmaf(sc, xx[j].x, acc.x);
                acc.y = fmaf(sc, xx[j].y, acc.y);
                acc.z = fmaf(sc, xx[j].z, acc.z);
                acc.w = fmaf(sc, xx[j].w, acc.w);
                if (e8 & (1u << j)) {
                    if (l8 & (1u << j)) {
                        const int slot = static_cast<int>(rec_s[k8 + j].z);
                        G* const gp = grad_col + static_cast<uint64_t>(static_cast<uint32_t>(slot)) * static_cast<uint32_t>(hd);
                        if constexpr (RMW) {
                            if (on) {
                                const float4 gb = cvt4(gc[j]);
                                float4 o;
                                o.x = gb.x + acc.x;
                                o.y = gb.y + acc.y;
                                o.z = gb.z + acc.z;
                                o.w = gb.w + acc.w;
                                st4(gp, o);
                            }
                        }
                    }
                    acc = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            if constexpr (RMW) {
                if (k8 + 8 < kend) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) gc[j] = gn[j];
                }
            }
        }
    }
    (void)mbar_wait_dep(tok_bar, 0u);
}

template <typename G, int NW, bool RMW>
__global__ void __launch_bounds__(NW * 32, (NW <= 8 ? DA2_SC_MINB : 1)) da2_gscatter_kernel(
    G* __restrict__ grad, const float* __restrict__ gr, int64_t gr_step_rows, const uint4* __restrict__ qrec,
    const int32_t* __restrict__ tb, int ti, Geo g) {
    extern __shared__ __align__(128) unsigned char smem[];
    gscatter_role<G, NW, RMW>(grad, gr, gr_step_rows, qrec, tb, ti, g, blockIdx.x, smem);
    // A no-op unless this grid is a programmatic dependent launch (the launchers never make one): the wait would let
    // it complete only after the grid launched before it has completed.
#ifndef DA2_PRE_SM90
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

template <typename E, typename IdxT, int NW>
__device__ __forceinline__ void gdots_role(
    const E* __restrict__ bank, const float* __restrict__ gr, int64_t gr_step_rows, const IdxT* __restrict__ q_idx,
    float* __restrict__ dots, int64_t plane, int ti, int valid, int TG, const Geo& g, int R, int b, int TPW) {
    using RowT = typename Row4<E>::T;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tg = b % TG;
    b /= TG;
    const int sl = b % g.NSL;
    const int p = b / g.NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * g.P + p;
    const int SW = g.SW, C = g.C, hd = g.hd;
  // TPW tokens per warp (a block covers NW * TPW consecutive tokens): fewer, longer dot blocks.
  for (int it = 0; it < TPW; ++it) {
    const int t = (tg * TPW + it) * NW + warp;
    if (t >= C) break;                                                    // warp-uniform
    const int c4 = 4 * lane;
    const bool on = c4 < SW && sl * SW + c4 < hd;
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
    if (on) {
        x = *reinterpret_cast<const float4*>(
            gr + (static_cast<int64_t>(ti) * gr_step_rows + static_cast<int64_t>(p) * C + t) * hd +
            static_cast<int64_t>(sl) * SW + c4);
    }
    const IdxT* qi = q_idx + (chunk * C + t) * static_cast<int64_t>(R);
    const E* bcol = bank + static_cast<int64_t>(sl) * SW + (on ? c4 : 0);
    float* drow = dots + static_cast<int64_t>(sl) * plane + (chunk * C + t) * static_cast<int64_t>(R);
    const bool tvalid = t < valid;
    // Per 32 entries: one coalesced load of their slot ids (lane j holds entry r32 + j; absolute slots fit int32),
    // then 4 sub-batches of 8 entries with the next sub-batch's bank rows loaded before this one is reduced.
    auto load8 = [&](RowT (&m)[8], int myslot, int r32, int s8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {   // padding tokens (t >= valid) touch neither q_idx nor the bank
            const int jj = s8 + j;
            const int slot = __shfl_sync(kFull, myslot, jj & 31);
            const bool ld = tvalid && r32 + jj < R && jj < 32;
            m[j] = (on && ld) ? *reinterpret_cast<const RowT*>(bcol + static_cast<int64_t>(slot) * hd)
                              : make_zero_row<RowT>();
        }
    };
    // One 8-entry sub-batch: per-lane chain, xor-16/8/4/2/1 reduce-scatter, store entry `base + (lane >> 2) & 7`.
    auto dot8 = [&](const RowT (&m)[8], int base) {
        float pd[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 mr = cvt4(m[j]);
            pd[j] = on ? __fmaf_rn(x.w, mr.w, __fmaf_rn(x.z, mr.z, __fmaf_rn(x.y, mr.y, __fmul_rn(x.x, mr.x)))) : 0.f;
        }
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
        const int rr_ = base + ((lane >> 2) & 7);
        if ((lane & 3) == 0 && rr_ < R) drow[rr_] = tvalid ? a1 : 0.f;
    };
    if (R <= 64) {
        // Common case: all <= 64 slot ids up front (two coalesced loads), then ONE pipeline over the token's
        // sub-batches (the next sub-batch's rows in flight while this one reduces), no stall between 32-groups.
        const int myA = (tvalid && lane < R) ? static_cast<int>(__ldg(qi + lane)) : 0;
        const int myB = (tvalid && 32 + lane < R) ? static_cast<int>(__ldg(qi + 32 + lane)) : 0;
        RowT mc[8], mn[8];
        load8(mc, myA, 0, 0);
#pragma unroll
        for (int s8 = 0; s8 < 64; s8 += 8) {
            if (s8 < R) {
                if (s8 + 8 < R) {
                    if (s8 + 8 < 32) load8(mn, myA, 0, s8 + 8);
                    else load8(mn, myB, 32, s8 + 8 - 32);
                }
                dot8(mc, s8);
#pragma unroll
                for (int j = 0; j < 8; ++j) mc[j] = mn[j];
            }
        }
    } else {
    for (int r32 = 0; r32 < R; r32 += 32) {
        const int rj = r32 + lane;
        const int myslot = (tvalid && rj < R) ? static_cast<int>(__ldg(qi + rj)) : 0;
        const int ng = min(32, R - r32);
        RowT mc[8], mn[8];
        load8(mc, myslot, r32, 0);
#pragma unroll
        for (int s8 = 0; s8 < 32; s8 += 8) {
            if (s8 < ng) {
                if (s8 + 8 < ng) load8(mn, myslot, r32, s8 + 8);
                dot8(mc, r32 + s8);
#pragma unroll
                for (int j = 0; j < 8; ++j) mc[j] = mn[j];
            }
        }
    }
    }
  }
}

template <typename E, typename IdxT, int NW>
__global__ void __launch_bounds__(NW * 32, DA2_DOTS_MINB) da2_gdots_kernel(
    const E* __restrict__ bank, const float* __restrict__ gr, int64_t gr_step_rows, const IdxT* __restrict__ q_idx,
    float* __restrict__ dots, int64_t plane, int ti, int valid, int TG, Geo g, int R, int TPW) {
    // launch_dependents: a no-op unless the next launch is programmatic (the launchers never make one).
#ifndef DA2_PRE_SM90
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    gdots_role<E, IdxT, NW>(bank, gr, gr_step_rows, q_idx, dots, plane, ti, valid, TG, g, R, blockIdx.x, TPW);
}

// Both GDS roles in ONE launch (no per-step fork/join): blocks [0, sblocks) scatter (they use the
// dynamic shared memory: token rows), the rest compute dots (no shared memory). 8 warps per block for both roles.
template <typename E, typename G, typename IdxT, bool RMW>
__global__ void __launch_bounds__(256, 2) da2_gds2f_kernel(
    const E* __restrict__ bank, G* __restrict__ grad, const float* __restrict__ gr, int64_t gr_step_rows,
    const uint4* __restrict__ qrec, const int32_t* __restrict__ tb, const IdxT* __restrict__ q_idx,
    float* __restrict__ dots, int64_t plane, int ti, int valid, int TG, Geo g, int R, int sblocks, int TPW) {
    extern __shared__ __align__(128) unsigned char smem[];
    // Roles interleaved over blockIdx in proportion (block b is a scatter block iff floor((b+1) S / T) > floor(b S / T),
    // S = sblocks, T = gridDim.x), so every wave mixes both roles instead of running all scatter blocks first.
    const int64_t b = blockIdx.x, T = gridDim.x, S = sblocks;
    const int64_t s0 = (b * S) / T, s1 = ((b + 1) * S) / T;
    if (s1 > s0) {
        gscatter_role<G, 8, RMW>(grad, gr, gr_step_rows, qrec, tb, ti, g, static_cast<int>(s0), smem);
    } else {
        gdots_role<E, IdxT, 8>(bank, gr, gr_step_rows, q_idx, dots, plane, ti, valid, TG, g, R,
                               static_cast<int>(b - s0), TPW);
    }
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

int slices_of(int hd) { return (hd + 127) / 128; }

}  // namespace

void da2_krec(at::Tensor kpay, at::Tensor k_idx, at::Tensor k_val, at::Tensor rex, at::Tensor cpe,
              at::Tensor krec) {
    const at::cuda::CUDAGuard guard(kpay.device());
    TORCH_CHECK(kpay.dim() == 2 && kpay.is_contiguous() && kpay.scalar_type() == at::kInt,
                "da2_krec: kpay must be a contiguous int32 [N, C * W]");
    TORCH_CHECK(k_idx.dim() == 3 && k_idx.is_contiguous() && k_val.is_contiguous() &&
                rex.is_contiguous() && cpe.is_contiguous(), "da2_krec: contiguous [N, C, W] inputs");
    TORCH_CHECK(rex.scalar_type() == at::kFloat, "da2_krec: rex must be float32");
    TORCH_CHECK(krec.is_contiguous() && krec.scalar_type() == at::kInt && krec.numel() == kpay.numel() * 4 &&
                reinterpret_cast<uintptr_t>(krec.data_ptr()) % 16 == 0,
                "da2_krec: krec must be a contiguous, 16-byte aligned int32 [N, C * W, 4]");
    const int64_t N = kpay.size(0), n = kpay.size(1);
    const int W = static_cast<int>(k_idx.size(2));
    TORCH_CHECK(k_idx.size(0) == N && k_idx.size(1) * W == n, "da2_krec: shape mismatch");
    TORCH_CHECK(cpe.scalar_type() == k_val.scalar_type(), "da2_krec: cpe dtype");
    const int64_t total = N * n;
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>((total + 255) / 256);
    uint4* out = reinterpret_cast<uint4*>(krec.data_ptr<int32_t>());
#define DA2_REC(IDX, VAL)                                                                   \
    da2_krec_kernel<IDX, VAL><<<blocks, 256, 0, stream>>>(                                  \
        kpay.data_ptr<int32_t>(), k_idx.data_ptr<IDX>(),                                    \
        reinterpret_cast<const VAL*>(k_val.data_ptr()), rex.data_ptr<float>(),              \
        reinterpret_cast<const VAL*>(cpe.data_ptr()), out, total, static_cast<int>(n), W)
    const bool i64 = k_idx.scalar_type() == at::kLong;
    const bool bf = k_val.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bf || k_val.scalar_type() == at::kFloat, "da2_krec: k_val bf16 or f32");
    TORCH_CHECK(i64 || k_idx.scalar_type() == at::kInt, "da2_krec: k_idx int32 or int64");
    if (i64 && bf) { DA2_REC(int64_t, __nv_bfloat16); }
    else if (i64) { DA2_REC(int64_t, float); }
    else if (bf) { DA2_REC(int32_t, __nv_bfloat16); }
    else { DA2_REC(int32_t, float); }
#undef DA2_REC
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void da2_tiles(at::Tensor kpay, at::Tensor tb) {
    const at::cuda::CUDAGuard guard(kpay.device());
    TORCH_CHECK(kpay.dim() == 2 && kpay.is_contiguous() && kpay.scalar_type() == at::kInt,
                "da2_tiles: kpay contiguous int32 [N, n]");
    TORCH_CHECK(tb.dim() == 2 && tb.is_contiguous() && tb.scalar_type() == at::kInt &&
                tb.size(0) == kpay.size(0) && tb.size(1) >= 2, "da2_tiles: tb int32 [N, TU + 1]");
    const int64_t N = kpay.size(0);
    const int n = static_cast<int>(kpay.size(1));
    const int TU = static_cast<int>(tb.size(1)) - 1;
    const int64_t total = N * (TU + 1);
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    da2_tiles_kernel<<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(
        kpay.data_ptr<int32_t>(), tb.data_ptr<int32_t>(), total, n, TU);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void da2_segb(at::Tensor kpay, at::Tensor tb, at::Tensor segb) {
    const at::cuda::CUDAGuard guard(kpay.device());
    TORCH_CHECK(kpay.dim() == 2 && kpay.is_contiguous() && kpay.scalar_type() == at::kInt, "da2_segb: kpay");
    TORCH_CHECK(tb.dim() == 2 && tb.is_contiguous() && tb.scalar_type() == at::kInt && tb.size(0) == kpay.size(0),
                "da2_segb: tb");
    TORCH_CHECK(segb.sizes() == tb.sizes() && segb.scalar_type() == at::kInt && segb.is_contiguous(), "da2_segb: segb");
    const int64_t N = kpay.size(0);
    const int TU = static_cast<int>(tb.size(1)) - 1;
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    da2_segb_kernel<<<static_cast<int>(N), 256, (TU + 1) * sizeof(int32_t), stream>>>(
        kpay.data_ptr<int32_t>(), tb.data_ptr<int32_t>(), segb.data_ptr<int32_t>(),
        static_cast<int>(kpay.size(1)), TU);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void da2_dw(at::Tensor bank, at::Tensor dv, int64_t dv_step_rows, at::Tensor krec, at::Tensor tb,
            int64_t ti, int64_t P, int64_t C, int64_t K, int64_t nw, int64_t rs,
            at::Tensor snapc, at::Tensor segbuf, at::Tensor segb, int64_t sb_step, int64_t sf_full) {
    const at::cuda::CUDAGuard guard(bank.device());
    TORCH_CHECK(bank.dim() == 2 && bank.is_contiguous(), "da2_dw: bank [slots, hd] contiguous");
    const int hd = static_cast<int>(bank.size(1));
    const int NSL = slices_of(hd);
    TORCH_CHECK(hd % NSL == 0, "da2_dw: head_dim ", hd, " does not split into ", NSL, " slices");
    const int SW = hd / NSL;
    const bool bf = bank.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bf || bank.scalar_type() == at::kFloat, "da2_dw: bank must be bf16 or f32");
    const int es = bf ? 2 : 4;
    TORCH_CHECK((SW * es) % 16 == 0 && (hd * es) % 16 == 0 && SW % 4 == 0,
                "da2_dw: slice rows must be 16-byte multiples (hd ", hd, ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bank.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(dv.data_ptr()) % 16 == 0, "da2_dw: 16-byte alignment");
    TORCH_CHECK(dv.scalar_type() == at::kFloat && dv.is_contiguous(), "da2_dw: dv f32 contiguous");
    TORCH_CHECK(C >= 1 && C <= 256, "da2_dw: C <= 256");
    TORCH_CHECK(krec.dim() == 3 && krec.size(2) == 4 && krec.is_contiguous() && krec.scalar_type() == at::kInt &&
                reinterpret_cast<uintptr_t>(krec.data_ptr()) % 16 == 0, "da2_dw: krec [N, n, 4] int32, 16-byte aligned");
    const int n = static_cast<int>(krec.size(1));
    const int TU = static_cast<int>(tb.size(1)) - 1;
    TORCH_CHECK(TU == K * nw, "da2_dw: tile table holds ", TU, " tiles, the grid needs ", K * nw);
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
    g.sf_full = 0;
    g.sf_full = static_cast<int>(sf_full);
    const bool sf_launch = segb.numel() > 0;   // SF: TMA bulk-run rows (SW * es)
    const size_t ring_row = !sf_launch ? 32 * 4 * es : SW * es;
    const size_t smem = static_cast<size_t>(g.tok_bytes) + static_cast<size_t>(nw) * rs * ring_row +
                        static_cast<size_t>(nw) * kRecPad * 16 + 8 * (1 + static_cast<size_t>(nw) * rs);
    TORCH_CHECK(smem <= 227 * 1024, "da2_dw: ", smem, " bytes of shared memory per CTA");
    const int blocks = static_cast<int>(P) * NSL * static_cast<int>(K);
    if (blocks == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* kr = reinterpret_cast<const uint4*>(krec.data_ptr<int32_t>());
    long long* stats_ptr = nullptr;
    void* snapc_ptr = nullptr;
    if (snapc.numel() > 0) {
        TORCH_CHECK(snapc.is_contiguous() && snapc.scalar_type() == bank.scalar_type() &&
                    snapc.numel() == krec.size(0) * static_cast<int64_t>(n) * hd,
                    "da2_dw: snapc must be the full per-entry snapshot [N * C * W, hd] in the bank dtype");
        snapc_ptr = snapc.data_ptr();
    }
    const bool sf = segb.numel() > 0;
    const void* segbuf_ptr = nullptr;
    const int32_t* segb_ptr = nullptr;
    if (sf) {
        TORCH_CHECK(bf, "da2_dw: the S/F split is compiled for bf16 banks only");
        TORCH_CHECK(segb.scalar_type() == at::kInt && segb.is_contiguous() && segb.numel() == tb.numel(),
                    "da2_dw: segb int32 shaped like tb");
        TORCH_CHECK(segbuf.scalar_type() == bank.scalar_type() && segbuf.is_contiguous() &&
                    segbuf.numel() >= (sb_step == 0 ? P : krec.size(0)) * static_cast<int64_t>(n) * hd &&
                    reinterpret_cast<uintptr_t>(segbuf.data_ptr()) % 16 == 0, "da2_dw: segbuf");
        segbuf_ptr = segbuf.data_ptr();
        segb_ptr = segb.data_ptr<int32_t>();
    }
#define DA2_DWM(E, NW_, RS_, M_)                                                              \
    {                                                                                         \
        auto kf = da2_dw_kernel<E, NW_, RS_, M_>;                                             \
        set_smem(kf, smem);                                                                   \
        kf<<<blocks, NW_ * 32, smem, stream>>>(reinterpret_cast<E*>(bank.data_ptr()),          \
                                               dv.data_ptr<float>(), dv_step_rows, kr,          \
                                               tb.data_ptr<int32_t>(), static_cast<int>(ti), g, \
                                               stats_ptr, reinterpret_cast<E*>(snapc_ptr),      \
                                               reinterpret_cast<const E*>(segbuf_ptr), segb_ptr, \
                                               static_cast<int>(sb_step));                      \
    }
#define DA2_DW(E, NW_, RS_) DA2_DWM(E, NW_, RS_, 0)
#define DA2_DWSF(NW_, RS_) DA2_DWM(__nv_bfloat16, NW_, RS_, 32)
#define DA2_DW_E(E)                                                                           \
    if (nw == 8 && rs == 16) DA2_DW(E, 8, 16)                                                 \
    else if (nw == 8 && rs == 32) DA2_DW(E, 8, 32)                                            \
    else if (nw == 16 && rs == 16) DA2_DW(E, 16, 16)                                          \
    else if (nw == 16 && rs == 32) DA2_DW(E, 16, 32)                                          \
    else TORCH_CHECK(false, "da2_dw: unsupported (nw, rs) = (", nw, ", ", rs, ")");
    if (sf) {
        if (nw == 8 && rs == 16) DA2_DWSF(8, 16)
        else if (nw == 8 && rs == 32) DA2_DWSF(8, 32)
        else if (nw == 16 && rs == 16) DA2_DWSF(16, 16)
        else if (nw == 16 && rs == 32) DA2_DWSF(16, 32)
        else TORCH_CHECK(false, "da2_dw: unsupported (nw, rs) = (", nw, ", ", rs, ")");
    }
    else if (bf) { DA2_DW_E(__nv_bfloat16) }
    else { DA2_DW_E(float) }
#undef DA2_DW_E
#undef DA2_DWSF
#undef DA2_DW
#undef DA2_DWM
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void da2_qrec(at::Tensor qpay, at::Tensor q_idx, at::Tensor q_val, at::Tensor logq, at::Tensor qrec) {
    const at::cuda::CUDAGuard guard(qpay.device());
    TORCH_CHECK(qpay.dim() == 2 && qpay.is_contiguous() && qpay.scalar_type() == at::kInt,
                "da2_qrec: qpay contiguous int32 [N, C * R]");
    TORCH_CHECK(q_idx.dim() == 3 && q_idx.is_contiguous() && q_val.is_contiguous() && logq.is_contiguous(),
                "da2_qrec: contiguous [N, C, R] inputs");
    TORCH_CHECK(logq.scalar_type() == at::kFloat, "da2_qrec: log_q float32");
    TORCH_CHECK(qrec.is_contiguous() && qrec.scalar_type() == at::kInt && qrec.numel() == qpay.numel() * 4 &&
                reinterpret_cast<uintptr_t>(qrec.data_ptr()) % 16 == 0,
                "da2_qrec: qrec contiguous, 16-byte aligned int32 [N, C * R, 4]");
    const int64_t N = qpay.size(0), n = qpay.size(1);
    const int R = static_cast<int>(q_idx.size(2));
    TORCH_CHECK(q_idx.size(0) == N && q_idx.size(1) * R == n, "da2_qrec: shape mismatch");
    const int64_t total = N * n;
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>((total + 255) / 256);
    uint4* out = reinterpret_cast<uint4*>(qrec.data_ptr<int32_t>());
#define DA2_QREC(IDX, VAL)                                                                  \
    da2_qrec_kernel<IDX, VAL><<<blocks, 256, 0, stream>>>(                                  \
        qpay.data_ptr<int32_t>(), q_idx.data_ptr<IDX>(),                                    \
        reinterpret_cast<const VAL*>(q_val.data_ptr()), logq.data_ptr<float>(), out, total, \
        static_cast<int>(n), R)
    const bool i64 = q_idx.scalar_type() == at::kLong;
    const bool bf = q_val.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bf || q_val.scalar_type() == at::kFloat, "da2_qrec: q_val bf16 or f32");
    TORCH_CHECK(i64 || q_idx.scalar_type() == at::kInt, "da2_qrec: q_idx int32 or int64");
    if (i64 && bf) { DA2_QREC(int64_t, __nv_bfloat16); }
    else if (i64) { DA2_QREC(int64_t, float); }
    else if (bf) { DA2_QREC(int32_t, __nv_bfloat16); }
    else { DA2_QREC(int32_t, float); }
#undef DA2_QREC
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void da2_gds2(at::Tensor bank, at::Tensor grad, at::Tensor gr, int64_t gr_step_rows, at::Tensor qrec, at::Tensor tb,
              at::Tensor q_idx, at::Tensor dots, int64_t ti, int64_t valid, int64_t P, int64_t C, int64_t R, int64_t K,
              int64_t nw, int64_t which, int64_t tpw) {
    // which: 3 = both roles (two launches), 4 = both in ONE launch.
    const at::cuda::CUDAGuard guard(bank.device());
    TORCH_CHECK(which == 3 || which == 4, "da2_gds2: which must be 3 (two launches) or 4 (one fused launch)");
    TORCH_CHECK(bank.dim() == 2 && bank.is_contiguous() && grad.is_contiguous() && grad.sizes() == bank.sizes() &&
                (grad.scalar_type() == bank.scalar_type() ||
                 (bank.scalar_type() == at::kBFloat16 && grad.scalar_type() == at::kFloat)),
                "da2_gds2: bank / grad [slots, hd]; grad dtype = bank dtype, or f32 with a bf16 bank");
    const int hd = static_cast<int>(bank.size(1));
    const int NSL = slices_of(hd);
    TORCH_CHECK(hd % NSL == 0, "da2_gds2: head_dim split");
    const int SW = hd / NSL;
    const bool bf = bank.scalar_type() == at::kBFloat16;
    const bool g32 = grad.scalar_type() == at::kFloat;
    TORCH_CHECK(gr.scalar_type() == at::kFloat && gr.is_contiguous(), "da2_gds2: grad_readings f32");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bank.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(gr.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(dots.data_ptr()) % 4 == 0 &&
                reinterpret_cast<uintptr_t>(qrec.data_ptr()) % 16 == 0 && qrec.is_contiguous() &&
                qrec.scalar_type() == at::kInt, "da2_gds2: 16-byte alignment (bank, grad, readings, qrec)");
    TORCH_CHECK((hd % 4) == 0 && (SW % 4) == 0, "da2_gds2: head_dim / slice width multiples of 4");
    TORCH_CHECK(q_idx.is_contiguous() && (q_idx.scalar_type() == at::kInt || q_idx.scalar_type() == at::kLong),
                "da2_gds2: q_idx int32 / int64 contiguous");
    const int n = static_cast<int>(qrec.size(1));
    const int64_t plane = qrec.size(0) * static_cast<int64_t>(n);
    TORCH_CHECK(dots.numel() == plane * NSL, "da2_gds2: dots planes");
    const int TU = static_cast<int>(tb.size(1)) - 1;
    TORCH_CHECK(TU == K * nw, "da2_gds2: tiles");
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
    g.sf_full = 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* qr = reinterpret_cast<const uint4*>(qrec.data_ptr<int32_t>());
    // (2) scatter, sorted order
    const size_t smem = static_cast<size_t>(g.tok_bytes) + static_cast<size_t>(nw) * 32 * 16 + 16;
    const int sblocks = static_cast<int>(P) * NSL * static_cast<int>(K);
    if (which == 4) {
        TORCH_CHECK(nw == 8, "da2_gds2: the fused launch uses 8 warps per block");
        // Dot blocks inherit the scatter's shared-memory reservation in the fused launch: make them few and long
        // (TPW tokens per warp).
        const int TPWf = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tpw, 16)));
        const int TGf = static_cast<int>((C + 8 * TPWf - 1) / (8 * TPWf));
        const int total = sblocks + static_cast<int>(P) * NSL * TGf;
        if (total == 0) return;
        const bool i64 = q_idx.scalar_type() == at::kLong;
#define DA2_GF(E_, G_, I_)                                                                    \
        {                                                                                     \
            auto kf = da2_gds2f_kernel<E_, G_, I_, true>;                                     \
            set_smem(kf, smem);                                                               \
            kf<<<total, 256, smem, stream>>>(reinterpret_cast<const E_*>(bank.data_ptr()),     \
                reinterpret_cast<G_*>(grad.data_ptr()), gr.data_ptr<float>(), gr_step_rows, qr, \
                tb.data_ptr<int32_t>(), q_idx.data_ptr<I_>(), dots.data_ptr<float>(), plane,     \
                static_cast<int>(ti), static_cast<int>(valid), TGf, g, static_cast<int>(R), sblocks, TPWf); \
        }
        if (bf && g32) { if (i64) DA2_GF(__nv_bfloat16, float, int64_t) else DA2_GF(__nv_bfloat16, float, int32_t) }
        else if (bf) { if (i64) DA2_GF(__nv_bfloat16, __nv_bfloat16, int64_t) else DA2_GF(__nv_bfloat16, __nv_bfloat16, int32_t) }
        else { if (i64) DA2_GF(float, float, int64_t) else DA2_GF(float, float, int32_t) }
#undef DA2_GF
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
#define DA2_GSC(G_, NW_)                                                                      \
    {                                                                                         \
        auto kf = da2_gscatter_kernel<G_, NW_, true>;                                         \
        set_smem(kf, smem);                                                                   \
        kf<<<sblocks, NW_ * 32, smem, stream>>>(reinterpret_cast<G_*>(grad.data_ptr()), gr.data_ptr<float>(), \
                                                gr_step_rows, qr, tb.data_ptr<int32_t>(), static_cast<int>(ti), g); \
    }
    if (sblocks > 0 && (which & 1)) {
        if (g32) { if (nw == 16) DA2_GSC(float, 16) else DA2_GSC(float, 8) }
        else { if (nw == 16) DA2_GSC(__nv_bfloat16, 16) else DA2_GSC(__nv_bfloat16, 8) }
    }
#undef DA2_GSC
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // (1) dots, token order: one warp per token, 8 warps per CTA
    const int TG = static_cast<int>((C + 7) / 8);
    const int dblocks = static_cast<int>(P) * NSL * TG;
#define DA2_GD(E_, I_)                                                                        \
    da2_gdots_kernel<E_, I_, 8><<<dblocks, 256, 0, stream>>>(                                 \
        reinterpret_cast<const E_*>(bank.data_ptr()), gr.data_ptr<float>(), gr_step_rows,     \
        q_idx.data_ptr<I_>(), dots.data_ptr<float>(), plane, static_cast<int>(ti), static_cast<int>(valid), TG, g, \
        static_cast<int>(R), 1);
    if (dblocks > 0 && (which & 2)) {
        const bool i64 = q_idx.scalar_type() == at::kLong;
        if (bf) { if (i64) { DA2_GD(__nv_bfloat16, int64_t) } else { DA2_GD(__nv_bfloat16, int32_t) } }
        else { if (i64) { DA2_GD(float, int64_t) } else { DA2_GD(float, int32_t) } }
    }
#undef DA2_GD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void da2_dot_reduce(at::Tensor part, at::Tensor out, int64_t nsl) {
    const at::cuda::CUDAGuard guard(part.device());
    const int64_t total = out.numel();
    TORCH_CHECK(part.is_contiguous() && out.is_contiguous() && part.numel() == total * nsl &&
                part.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "da2_dot_reduce");
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    da2_dot_reduce_kernel<<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(
        part.data_ptr<float>(), out.data_ptr<float>(), total, static_cast<int>(nsl));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_mover_da2",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("da2_krec", "da2_tiles", "da2_dw", "da2_qrec", "da2_dot_reduce", "da2_segb", "da2_gds2"),
        extra_cuda_cflags=("-O3", "-lineinfo"),
    )


# ------------------------------------------------------------------ geometry (shapes only)

_GEOMS = {(8, 16), (8, 32), (16, 16), (16, 32)}


_SMEM_MAX = 227 * 1024   # GB300 / H100 opt-in per block; fits() narrows it to the device's own limit


def _smem_limit(device) -> int:
    """The device class's opt-in shared memory per block (never the grid or the geometry)."""
    return smem_cap(device)


def _smem_bytes(chunk_size, head_dim, row_es, nw, rs):
    """Shared memory of a DW (row_es = (bank es,)) or GDS (row_es = (bank es, grad es)) CTA."""
    nsl = -(-head_dim // 128)
    sw = head_dim // nsl
    tok = (chunk_size * sw * 4 + 64 + 127) // 128 * 128
    slot = sum(32 * 4 * es for es in row_es)
    return tok + nw * rs * slot + nw * 40 * 16 + 8 * (1 + nw * rs)


def _fit(pairs, want, chunk_size, head_dim, row_es, limit=_SMEM_MAX):
    """`want` if its shared memory fits, else the first compiled pair (largest first) that fits."""
    if _smem_bytes(chunk_size, head_dim, row_es, *want) <= limit:
        return want
    for nw, rs in sorted(pairs, key=lambda x: -(x[0] * x[1])):
        if _smem_bytes(chunk_size, head_dim, row_es, nw, rs) <= limit:
            return nw, rs
    raise ValueError(f"no mover geometry fits shared memory (C={chunk_size}, hd={head_dim})")


def dw_geometry(chunk_size: int, width: int, head_dim: int, bank_dtype: torch.dtype, limit: int = _SMEM_MAX):
    """(nw, rs, ppw): warps per CTA, ring rows per warp, nominal positions per warp tile: (8, 16, 128), or the
    largest compiled (nw, rs) that fits shared memory. From the shapes only (never the SM count); the output does not
    depend on any of them (see the module docstring)."""
    es = bank_dtype.itemsize
    nw, rs = _fit(_GEOMS, (8, 16), chunk_size, head_dim, (es,), limit)
    return nw, rs, 128


def fits(bank: torch.Tensor, chunk_size: int, kind: str, grad_dtype=None) -> bool:
    """Whether some compiled geometry of the `kind` ("dw" | "gds") mover fits shared memory."""
    es = bank.element_size()
    if kind == "dw":
        pairs, row_es = _GEOMS, (es,)
    else:
        esg = grad_dtype.itemsize if grad_dtype is not None else es
        pairs, row_es = _GDS_GEOMS, (es, esg)
    try:
        _fit(pairs, min(pairs), chunk_size, bank.shape[-1], row_es, _smem_limit(bank.device))
        return True
    except ValueError:
        return False


def supported(bank: torch.Tensor, chunk_size: int) -> bool:
    """Shapes, dtype and architecture these movers handle (the modern backend: TMA bulk copies and mbarrier
    transaction counts); anything else takes phase2_kernels' launchers."""
    if not bank.is_cuda or backend(bank.device) != "modern":
        return False
    return _shape_supported(bank, chunk_size)


def _shape_supported(bank: torch.Tensor, chunk_size: int) -> bool:
    # Slot ids travel as signed int32 in the records and ring addressing: banks of >= 2^31 rows take phase2_kernels' path.
    if bank.dim() != 2 or bank.shape[0] >= 2 ** 31:
        return False
    return _shape_supported_inner(bank, chunk_size)


def _shape_supported_inner(bank: torch.Tensor, chunk_size: int) -> bool:
    hd = bank.shape[-1]
    nsl = -(-hd // 128)
    if hd % nsl:
        return False
    sw = hd // nsl
    es = bank.element_size()
    return (bank.dtype in (torch.bfloat16, torch.float32) and sw % 4 == 0 and (sw * es) % 16 == 0
            and (hd * es) % 16 == 0 and chunk_size <= 256)


def build_krec(spay_k, k_idx, k_val, rex, cpe):
    """krec [N, C * W, 4] int32 per sorted write position (the Phase-1 record contract)."""
    n = spay_k.shape[-1]
    spay_k = spay_k.reshape(-1, n)
    krec = torch.empty(spay_k.shape[0], n, 4, device=spay_k.device, dtype=torch.int32)
    _module().da2_krec(spay_k, k_idx.contiguous(), k_val.contiguous(), rex.contiguous(),
                       cpe.contiguous(), krec)
    return krec


def build_tiles(spay_k, tiles: int):
    n = spay_k.shape[-1]
    spay_k = spay_k.reshape(-1, n)
    tb = torch.empty(spay_k.shape[0], tiles + 1, device=spay_k.device, dtype=torch.int32)
    _module().da2_tiles(spay_k, tb)
    return tb



def _tiles_or_build(spay, tiles, TU, fill_last=False):
    """Tile bounds [N, TU + 1] int32: Phase 1's FIRST-snapped tiles (tiles[t] = first FIRST >= t * L, last
        entry n; or a per-tile count prefix with fill_last) padded to TU + 1 with their last entry, when given and not
        longer than TU + 1; else None (the caller builds its own). Any monotone FIRST-snapped bounds give the same bits."""
    if tiles is None or tiles.numel() == 0:
        return None
    t2 = tiles.reshape(-1, tiles.shape[-1])
    if t2.shape[0] != spay.reshape(-1, spay.shape[-1]).shape[0] or t2.shape[1] > TU + 1 or t2.dtype != torch.int32:
        return None
    if t2.shape[1] == TU + 1:
        return t2.contiguous()
    pad = t2[:, -1:].expand(-1, TU + 1 - t2.shape[1])
    return torch.cat([t2, pad], dim=1).contiguous()

def build_qrec(spay_q, q_idx, q_val, log_q):
    """qrec [N, C * R, 4] int32 per sorted read position {spay_q element, abs slot, scale, 0}."""
    n = spay_q.shape[-1]
    spay_q = spay_q.reshape(-1, n)
    qrec = torch.empty(spay_q.shape[0], n, 4, device=spay_q.device, dtype=torch.int32)
    _module().da2_qrec(spay_q, q_idx.contiguous(), q_val.contiguous(), log_q.contiguous(), qrec)
    return qrec


_GDS_GEOMS = {(8, 16), (8, 8), (16, 16), (16, 8)}


def gds_geometry(chunk_size: int, width: int, head_dim: int, bank_dtype: torch.dtype, grad_dtype=None,
                 limit: int = _SMEM_MAX):
    """(nw, rs, ppw) of the GDS mover: (8, 8, 128), or the largest compiled (nw, rs) that fits shared memory; shapes
    only."""
    es = torch.tensor([], dtype=bank_dtype).element_size()
    esg = torch.tensor([], dtype=grad_dtype or bank_dtype).element_size()
    nw, rs = _fit(_GDS_GEOMS, (8, 8), chunk_size, head_dim, (es, esg), limit)
    return nw, rs, 128


def make_gds(bank, grad_bank, grad_readings, spay_q, qrec, d_q_w, *, chunk_size, num_parallel,
             steps, q_idx=None, valid_tokens=None, tiles_q=None):
    """Step launcher of the GDS mover. The backward walks the steps in reverse, so the slice
        partials of a wide head's dots are summed (in slice order) when step 0 runs."""
    hd = bank.shape[-1]
    sc = num_parallel * chunk_size
    if grad_readings.numel() == steps * sc * hd:
        gr_rows = sc
    elif grad_readings.numel() == sc * hd:
        gr_rows = 0
    else:
        raise ValueError("grad_readings: unexpected size")
    n = spay_q.shape[-1]
    r = n // chunk_size
    nsl = -(-hd // 128)
    nw, rs, ppw = gds_geometry(chunk_size, r, hd, bank.dtype, grad_bank.dtype, _smem_limit(bank.device))
    K = max(1, -(-n // (nw * ppw)))
    tb = _tiles_or_build(spay_q, tiles_q, K * nw) if ppw == 128 else None   # Phase 1's tiles use L = 128
    if tb is None:
        tb = build_tiles(spay_q, K * nw)
    dqw = d_q_w.view(-1)
    part = torch.empty(nsl * dqw.numel(), device=bank.device, dtype=torch.float32) if nsl > 1 else dqw
    # Non-f32 readings (direct callers; the package's backward stages f32): converted at every launch into one f32
    # buffer, so a caller (or the aligned-scratch refresh) updating the readings between steps is always seen.
    gr_conv = grad_readings.dtype != torch.float32
    gr = grad_readings if not gr_conv else torch.empty(grad_readings.shape, device=grad_readings.device,
                                                       dtype=torch.float32)
    mod = _module()

    # Dots in token order + the grad scatter in sorted order with the single-writer RMW update (da2_gds2: one
    # rounding per element); one fused launch with 8 warps per block, else two launches.
    if q_idx is None or valid_tokens is None:
        raise ValueError("make_gds: q_idx [N, C, R] and valid_tokens are required (the token-order dots read them)")
    qix = q_idx.reshape(-1, chunk_size, r).contiguous()
    fused = nw == 8

    def launch(ti: int) -> None:
        if gr_conv:
            gr.copy_(grad_readings)
        valid = max(0, min(chunk_size, valid_tokens - ti * chunk_size))
        mod.da2_gds2(bank, grad_bank, gr, gr_rows, qrec, tb, qix, part, ti, valid, num_parallel, chunk_size, r, K, nw,
                     4 if fused else 3, 1)
        if nsl > 1 and ti == 0:
            mod.da2_dot_reduce(part, dqw, nsl)

    launch.geom = dict(nw=nw, rs=rs, ppw=ppw, K=K, nsl=nsl)
    return launch


def make_dw(bank, delta_v, spay_k, krec, *, chunk_size, num_parallel, steps, snapshot=None, tiles_k=None,
            segb_k=None, snapshot_quant="none", payload=None, scale=None, valid_tokens=None):
    """Step launcher of the DW mover. delta_v: f32 rows of one step or of every step. A snapshot tagged by the forward
    (snapshot._dc_snapc_layout) selects the base-row source: layout 2 (saving forward: the dense compact rows, row
    chunk * n + g of each valid group) or layout 4 (no-grad forward: the bank itself, mover_fx_cuda's bank-direct
    kernel only); otherwise the walk reads bank[slot] and the gather writes the per-entry stash. snapshot_quant fp8 /
    int4: mover_fx_cuda's quantized bank-direct DW writes the compact rows (layout 2, by the DW); a quantized call
    without them raises (2.0.0: memory_ops.refuse_non_native refuses it up front). The launcher announces what it reads
    and writes (reads_snapc, writes_snapc)."""
    hd = bank.shape[-1]
    sc = num_parallel * chunk_size
    if delta_v.numel() == steps * sc * hd:
        dv_rows = sc
    elif delta_v.numel() == sc * hd:
        dv_rows = 0
    else:
        raise ValueError(f"delta_v: expected {sc} or {steps * sc} rows of {hd}")
    n = spay_k.shape[-1]
    # This module's DW kernels run on the modern backend only; elsewhere (a100 / rtx) mover_fx_cuda's portable walk takes
    # every call (movers.dw_admits / _fx_layout) with Phase 1's tile set (8 warps, L = 128) as the default tiles.
    modern = backend(bank.device) == "modern"
    tail = not _shape_supported(bank, chunk_size)   # tail: only mover_fx_cuda's DW runs it (checked below)
    if modern:
        try:
            nw, rs, ppw = dw_geometry(chunk_size, n // chunk_size, hd, bank.dtype, _smem_limit(bank.device))
        except ValueError:
            if not tail:
                raise
            nw, rs, ppw = 8, 16, 128              # a tail never runs this module's DW: nominal tiles for mover_fx
    else:
        nw, rs, ppw = 8, 16, 128
    K = max(1, -(-n // (nw * ppw)))
    tb = _tiles_or_build(spay_k, tiles_k, K * nw) if ppw == 128 else None   # Phase 1's tiles use L = 128
    segb_dp = _tiles_or_build(spay_k, segb_k, K * nw) if tb is not None else None
    if tb is None:
        tb = build_tiles(spay_k, K * nw)
    mod = _module()

    snapc = torch.empty(0, device=bank.device, dtype=bank.dtype)
    # snapshot_quant bf16 on a float32 bank (the forward passes "none" here): the every-step compact rows are bf16, tagged
    # layout 2 and written by the DW (by_dw); only mover_fx_cuda's bank-direct WR walk writes them (bf16 stores, RN).
    s16 = (snapshot is not None and bank.dtype == torch.float32 and snapshot.dtype == torch.bfloat16
           and snapshot_quant == "none" and getattr(snapshot, "_dc_snapc_layout", 0) == 2
           and bool(getattr(snapshot, "_dc_snapc_by_dw", False)))
    full_snap = (snapshot is not None and (snapshot.dtype == bank.dtype or s16) and snapshot.is_contiguous()
                 and snapshot.data_ptr() % 16 == 0 and snapshot_quant == "none"
                 and snapshot.numel() == steps * sc * (n // chunk_size) * hd)
    # float32 banks: mover_fx_cuda's float32 instantiations (the one-slice bank-direct walk, 8 / 16 warps) serve layouts
    # 4 and 2 when they take the call (fx32: the predicate of the launcher loop below with nw_fx); every other float32
    # call keeps this module's DW and the gather's per-entry stash (layout 0), as before.
    fx32 = False
    if bank.dtype == torch.float32 and snapshot is not None and snapshot_quant == "none":
        from . import mover_fx_cuda as _fx
        nw_fx32, _ = _fx.tile_geometry(num_parallel, hd, n, chunk_size, 8, bank.device)
        fx32 = _fx.supported(bank, chunk_size, nw_fx32, True)
        if (tail or s16) and not fx32:            # tails, bf16 rows: any bank-direct warp count that fits (s16_supported)
            fx32 = any(_fx.supported(bank, chunk_size, w, True) for w in (8, 16))
    fx_dtype = bank.dtype == torch.bfloat16 or fx32
    # Layout 4 (no-grad forward, bank-direct), tagged by the writer: the forward writes no compact rows; mover_fx_cuda's
    # DW reads each valid group's base row from bank[slot] (the pre-step row). Only that kernel reads the bank this way:
    # when it does not run, the launcher announces 0 (this module's DW and the gather's per-step stash).
    step_bank = (fx_dtype and snapshot is not None
                 and getattr(snapshot, "_dc_snapc_layout", 0) == 4 and snapshot.dtype == bank.dtype
                 and snapshot.is_contiguous() and snapshot.data_ptr() % 16 == 0 and snapshot_quant == "none"
                 and snapshot.numel() == sc * (n // chunk_size) * hd)
    # Dense compact snapshot (saving forward, writer-tagged layout 2: row chunk * n + g of each valid group, g its
    # ordinal): the dense walk reads its base rows there (group order); DW writes no snapc.
    dense = (fx_dtype and full_snap and not step_bank
             and getattr(snapshot, "_dc_snapc_layout", 0) == 2)
    sf = dense
    noint = torch.empty(0, device=bank.device, dtype=torch.int32)
    sb_step, sf_full = 0, 0
    if dense:
        segb = segb_dp if segb_dp is not None else build_segb(spay_k, tb)
        segbuf, sb_step, sf_full = snapshot, num_parallel, 1
        snapc = snapshot[:0]                      # no DW-side snapc writes
    else:
        segb, segbuf = noint, snapc[:0]
    # snapshot_quant fp8 / int4 (training forward only): the DW updates the bank exactly as without quantization
    # (the no-grad forward never quantizes, so both forwards run the same DW arithmetic).
    if snapshot_quant == "bf16":   # bf16 rows: the forward passes "none" with its bf16 compact buffer (no per-entry route)
        raise RuntimeError("make_dw: snapshot_quant bf16 has no per-entry route (the forward passes 'none')")
    qmode = {"none": 0, "fp8": 1, "int4": 2}[snapshot_quant]
    if qmode and (snapshot is None or payload is None or scale is None or valid_tokens is None):
        raise ValueError("make_dw: snapshot_quant needs snapshot, payload, scale and valid_tokens")
    # fp8 / int4 compact rows (saving forward, writer-tagged layout 2 + by-DW, eager and captured): mover_fx_cuda's
    # quantized bank-direct DW writes each valid group's pre-write row quantized to payload / scale (row chunk * n + g).
    # It is the only quantized writer (2.0.0, A4: the per-entry quantizer da2_quant is gone): any other quantized call
    # raises below; the op refuses such calls before any work (memory_ops.refuse_non_native: forward._fxq_rows).
    qdense = False
    qform = 0                                     # the kernels' form: 1 / 2 cluster, 5 / 6 all slices (quant_choice)
    if (qmode and getattr(snapshot, "_dc_snapc_layout", 0) == 2 and getattr(snapshot, "_dc_snapc_by_dw", False)
            and snapshot.dtype == bank.dtype and snapshot.is_contiguous() and snapshot.data_ptr() % 16 == 0):
        from . import mover_fx_cuda as _fx
        qform = _fx.quant_choice(bank, chunk_size, qmode)
        qdense = qform != 0
    if qmode and not qdense:
        raise RuntimeError("make_dw: a quantized call needs mover_fx_cuda's quantized compact rows (snapshot layout 2 "
                           "written by the DW, quant_choice != 0); memory_ops.refuse_non_native refuses the others")
    fxq = dict(quant=qform, payload=payload, scale=scale) if qdense else {}

    # mover_fx_cuda's DW (8-row base-row units, 2D tensor-map boxes for column-sliced heads) when it supports the
    # shape: bitwise this module's dense walk, with the same records, tiles and segb.
    fx_launch = None
    # Bank-direct DW for layout 4 and for the dense layout (the dense rows the backward needs are still written; the
    # DW reads the same values from the bank).
    # Off the modern backend no other CUDA DW exists: a call without one of mover_fx_cuda's layouts (layout 0: the gather
    # writes the per-entry stash) runs its bank-direct walk too (the no-grad forward's DW, hence the same bank update).
    stash0 = not modern and not (step_bank or dense or qdense)
    bankb = step_bank or dense or qdense or stash0
    # mover_fx_cuda's snapc argument: the dense rows (layout 2), or for layout 4 the one-step snapshot (shape-checked
    # and tensor-map anchor only: the bank-direct DW reads no snapshot rows). This module's own arguments (segbuf /
    # sb_step) are left untouched.
    fx_snap, fx_sb = (snapshot, 0) if step_bank else (segbuf if dense else None, sb_step)
    if qdense:                                    # the one-step snapshot anchors the maps; rows of every step
        fx_snap, fx_sb = snapshot, num_parallel
    if stash0:                                    # shape-checked only: the bank-direct walk reads no snapshot rows
        fx_snap, fx_sb = snapshot, 0
    # The bank-direct DW also writes the dense compact rows of the saving forward (layout 2) when the forward asked for
    # it on the snapshot tensor (snapshot._dc_snapc_by_dw, set by the writer only) and the kernel runs; the launcher
    # announces it (writes_snapc = 2) and the forward's gather then writes no compact rows.
    snapw = dense and bankb and bool(getattr(snapshot, "_dc_snapc_by_dw", False))
    if dense or step_bank or qdense or stash0:
        from . import mover_fx_cuda as _fx
        # mover_fx_cuda's tile geometry: shorter warp tiles (ppw 64 / 32) when a step has few warps, 16-warp CTAs
        # when the token rows are large. Speed only: every tile is cut at FIRST boundaries, so each group stays whole in
        # one warp and is summed in sorted order (the same bits for any tiles). Phase 1's L = 128 tiles serve ppw 128
        # (padded); other tile lengths are built here (tiles + segb, once per factory call).
        nw_fx, ppw_fx = _fx.tile_geometry(num_parallel, bank.shape[-1], n, chunk_size,
                                          nw if bank.dtype == torch.bfloat16 else 8, bank.device,
                                          quant=qform)
        # Two column slices per warp (fx_dw2_kernel; even slice counts; bf16 banks) when its geometry fits: tiles of
        # ppw2 positions, built here (FIRST-snapped like every tile set: the same bits for any tiles).
        g2 = (_fx.tile_geometry_w2(num_parallel, bank.shape[-1], n, chunk_size, _fx._smem_limit(bank.device),
                                   bank.device, quant=qform)
              if bankb and bank.dtype == torch.bfloat16 and _fx.rows16(bank)   # tails: one slice per warp
              and _fx.supported(bank, chunk_size, 8, True) else None)
        if g2 is not None:
            nw_c, ppw_c = g2
            K_fx = max(1, -(-n // (nw_c * ppw_c)))
            if qdense:                            # quantized: one wave of co-resident clusters (speed only)
                K_fx = _fx.quant_k(bank, chunk_size, qform, True, nw_c, num_parallel, n, K_fx)
            tb_fx = build_tiles(spay_k, K_fx * nw_c)
            segb_fx = build_segb(spay_k, tb_fx) if snapw or qdense else tb_fx
            fx_launch = _fx.make_dw_fx(bank, delta_v, krec, tb_fx, segb_fx, fx_snap, chunk_size=chunk_size,
                                       num_parallel=num_parallel, dv_rows=dv_rows, sb_step=fx_sb, nw=nw_c, K=K_fx,
                                       bankb=bankb, snapw=snapw, w2=True, **fxq)
            fx_launch.geom = dict(fx_launch.geom, ppw=ppw_c)
        for nw_c in (() if g2 is not None else dict.fromkeys((nw_fx, nw) + ((8, 16) if qdense or tail or s16 else ()))):
            if not _fx.supported(bank, chunk_size, nw_c, bankb) or (qdense and not _fx.quant_geometry(bank, chunk_size, nw_c, qform)):
                continue
            K_fx = max(1, -(-n // (nw_c * ppw_fx)))
            if qdense:
                K_fx = _fx.quant_k(bank, chunk_size, qform, False, nw_c, num_parallel, n, K_fx)
            no_segb = bankb and not snapw and not qdense   # the bank-direct DW never reads segb (shape-checked only); writing the dense rows uses its ordinals
            if K_fx * nw_c == K * nw and ppw_fx == ppw:
                tb_fx = tb
                segb_fx = tb if no_segb else (segb if dense else build_segb(spay_k, tb))
            else:
                tb_fx = _tiles_or_build(spay_k, tiles_k, K_fx * nw_c) if ppw_fx == 128 else None
                segb_fx = (_tiles_or_build(spay_k, segb_k, K_fx * nw_c) if tb_fx is not None and not no_segb
                           else None)
                if tb_fx is None:
                    tb_fx = build_tiles(spay_k, K_fx * nw_c)
                if no_segb:
                    segb_fx = tb_fx
                elif segb_fx is None:
                    segb_fx = build_segb(spay_k, tb_fx)
            fx_launch = _fx.make_dw_fx(bank, delta_v, krec, tb_fx, segb_fx, fx_snap, chunk_size=chunk_size,
                                       num_parallel=num_parallel, dv_rows=dv_rows, sb_step=fx_sb, nw=nw_c, K=K_fx,
                                       bankb=bankb, snapw=snapw, **fxq)
            fx_launch.geom = dict(fx_launch.geom, ppw=ppw_fx)
            break
    if fx32 and (dense or step_bank) and fx_launch is None:   # the loop's own predicate admitted it: unreachable
        raise RuntimeError("make_dw: mover_fx_cuda declined a float32 call its admission accepted")
    if s16 and (fx_launch is None or not snapw):  # bf16 compact rows: only that walk writes them (s16_supported admitted it)
        raise RuntimeError("make_dw: bf16 compact snapshot rows need mover_fx_cuda's bank-direct WR walk, which declined")
    if qdense and fx_launch is None:              # quant_supported admitted it (16 warps fit): unreachable
        raise RuntimeError("make_dw: mover_fx_cuda declined a quantized call its admission accepted")
    if not modern and fx_launch is None:          # mover_fx_cuda.admits accepted it, and no other CUDA DW exists there
        raise RuntimeError("make_dw: mover_fx_cuda declined a call on the a100 / rtx backend that its admission accepted")
    if tail and fx_launch is None:               # tail shape: this module's DW cannot run it
        raise RuntimeError("make_dw: a tail shape (non-16-byte slices) needs mover_fx_cuda's bank-direct DW")

    def launch(ti: int) -> None:
        if fx_launch is not None:
            fx_launch(ti)
            return
        mod.da2_dw(bank, delta_v, dv_rows, krec, tb, ti, num_parallel, chunk_size, K, nw, rs,
                   snapc, segbuf, segb, sb_step, sf_full)

    launch.geom = dict(nw=nw, rs=rs, ppw=ppw, K=K, sf=sf, dense=dense, step_bank=step_bank,
                       fx=fx_launch.geom if fx_launch is not None else None)
    # Handshake with the forward (it writes compact rows only when the DW launcher announces it reads them, in the
    # announced layout): 2 = dense group ordinals, 4 = the bank-direct DW reads the bank, 0 = the DW reads the bank and
    # the gather writes the per-entry stash as usual.
    launch.reads_snapc = 2 if dense or (qdense and fx_launch is not None) else (4 if step_bank and fx_launch is not None else 0)
    # Programmatic dependent launches of mover_fx_cuda's DW steps (the caller asserts the launch-order rule, see
    # mover_fx_cuda.make_dw_fx); this module's kernels never launch that way.
    launch.set_pdl = fx_launch.set_pdl if fx_launch is not None else (lambda on: None)
    launch.fx = fx_launch                  # mover_fx_cuda's DW launcher (its plan feeds the fused deterministic step loop), or None
    # 2 = this launcher writes the dense compact rows (layout 2) of every step itself, raw or quantized (fx quant)
    # (the forward's gather must not).
    launch.writes_snapc = 2 if fx_launch is not None and (fx_launch.geom.get("snapw") or fx_launch.geom.get("quant")) else 0
    return launch


def build_segb(spay, tb):
    """segb [N, TU + 1] int32: valid group FIRSTs before each tile bound (the dense walk's ordinals)."""
    n = spay.shape[-1]
    spay = spay.reshape(-1, n)
    segb = torch.empty_like(tb)
    _module().da2_segb(spay, tb, segb)
    return segb
