# Copyright (c) Meta Platforms, Inc. and affiliates.
"""pa_dc_cuda: the deterministic backward's Phase A (the backward of DW) and its GWR gather ("pa": Phase A).

Backward step ti (backward.py, deterministic path):
  GWR  (dc_gwr)  d_delta_v[t] = sum_n write_w[t,n] * grad[slot[t,n]]   (same order as token_gather)
                 d_k_pd[t,n]  = <delta_v[t], grad[slot[t,n]]>           (no grad_rows stash)
  PA   (dc_pa)   per write group (slot s, sorted members m):
                 grad[s]  = gdw * cpe_last + sum_m sc_m * d_dva[tok_m],  sc = -beta*kv*exp(log_k)
                 bank[s]  = snap                                         (restore)
                 grad_g_decay[e] = cpe_e * <gdw, snap>,  d_k_eff[e] = -beta_e * <d_dva[tok_e], snap>
gdw = grad[s] before this step's PA (read once per group), snap = the snapshot row of the group's FIRST
member (all members' snapshot rows are the same pre-write row).

PA geometry: block = (stream-chunk p, column slice, CTA range); warp = tile of sorted positions whose
bounds snap back to a group FIRST (no group crosses a tile; L from shapes only); lane = 4 columns of a
slice of SLW = 4*ceil(ceil(hd/4)/ceil(hd/128)) columns (hd 120 -> 1 x 120, hd 960 -> 8 x 120).
Per position: one 16-byte smem load of the token row, 4 FMAs into the group accumulator, 4 for the
d_k_eff partial. Per-entry dots: an incremental binary-counter transposed reduction over 32 positions
(level k pairs lanes l, l^(1<<k)); every entry is summed over the same lane tree, so its value does
not depend on the tiling, the warps per CTA or the ring depth. Per group the grad row and snapshot row
arrive through a per-warp cp.async ring D groups ahead; <gdw, snap> is one xor allreduce per group.
With several slices the per-entry dots are written as slice partial planes and summed in slice order
by one reduce launch per op call (dc_pa_reduce)."""

import functools

import torch

from ._build import load_cuda_extension
from ._device import headers

# dc_pa's WIDE walk (64-bit snapshot rows: chunk-local int32 rows from a 64-bit block base; the same bits) runs for
# snapshots of 2**31 rows or more, which dc_pa decides by itself, or with the host-only dbg bit DBG_WIDE, which the
# backward sets on the dc_pa route from _WIDE_SNAP_ROWS snapshot rows. Tests set _WIDE_SNAP_ROWS = 0 to run the wide walk
# at small sizes (as phase2_kernels._WIDE_ENTRIES).
_WIDE_SNAP_ROWS = 2 ** 31
DBG_WIDE = 1 << 18

_CPP = r"""
void dc_records_pa(at::Tensor kpay, at::Tensor k_idx, at::Tensor k_val, at::Tensor logk,
                   at::Tensor beta, at::Tensor cpe, at::Tensor rec);
void dc_copy_raw(at::Tensor dst, at::Tensor src);
bool dc_overlaps(at::Tensor t);
void dc_pa(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
           at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
           at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
           int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc);
void dc_pa_reduce(at::Tensor part_eff, at::Tensor part_g, at::Tensor part_pd, at::Tensor beta,
                  at::Tensor cpe, at::Tensor d_k_eff, at::Tensor grad_g, at::Tensor d_k_pd,
                  int64_t P, int64_t C, int64_t W, int64_t T);
void dc_gwr(at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor out, at::Tensor dv,
            at::Tensor pd, int64_t row0, int64_t out_row0, int64_t tokens, int64_t chunk,
            int64_t valid, int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc,
            at::Tensor kpos, at::Tensor gdwc, int64_t pdl);
at::Tensor dc_gwr_plan(at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor out, at::Tensor dv, at::Tensor pd,
                       int64_t row_step, int64_t out_row_step, int64_t tokens, int64_t chunk, int64_t valid_tokens,
                       int64_t ti_max, int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc,
                       at::Tensor kpos, at::Tensor gdwc);
int64_t dc_gwr_launch_addr();
at::Tensor dc_pa_plan(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
                      at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
                      at::Tensor out_g, int64_t ti_max, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
                      int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc);
int64_t dc_pa_launch_addr();
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>
#include <mutex>
#include <map>
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
#define DC_THREADS 256
#define DC_MINB 2

__device__ __forceinline__ float f_of(float v) { return v; }
__device__ __forceinline__ float f_of(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float4 to_f4(uint2 v) {
    return make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xFFFF0000u),
                       __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xFFFF0000u));
}
__device__ __forceinline__ float4 to_f4(float4 v) { return v; }
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
template <typename E> struct RowOf;
template <> struct RowOf<__nv_bfloat16> { using T = uint2; };    // 4 bf16
template <> struct RowOf<float> { using T = float4; };           // 4 f32
__device__ __forceinline__ void store4(__nv_bfloat16* p, float4 v) {
    *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf2(v.x, v.y), pack_bf2(v.z, v.w));
}
__device__ __forceinline__ void store4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void store_raw(__nv_bfloat16* p, uint2 v) { *reinterpret_cast<uint2*>(p) = v; }
__device__ __forceinline__ void store_raw(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
template <int BYTES>
__device__ __forceinline__ void cp_async(void* dst, const void* src, int src_bytes) {
    if (BYTES == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
                     "l"(src), "r"(src_bytes) : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_u32(dst)),
                     "l"(src), "n"(BYTES), "r"(src_bytes) : "memory");
    }
}
// Range prefetch into L2 (modern device passes): a performance hint only.
__device__ __forceinline__ void l2_prefetch(const void* p, int bytes) {
#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
// Waits until at most `pending` (0..7) of this thread's newest cp.async groups are outstanding; larger values
// wait for 7 (stricter, still correct). The ring depth is at most 8, so pending <= 7 here.
__device__ __forceinline__ void cp_wait_dyn(int pending) {
    switch (pending) {
#define DC_W(k) case k: cp_wait<k>(); return;
        DC_W(0) DC_W(1) DC_W(2) DC_W(3) DC_W(4) DC_W(5) DC_W(6)
#undef DC_W
        default: break;
    }
    cp_wait<7>();
}

// FIRST position of the group holding position j. Warp-uniform; all lanes call it.
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

// Sum of one value over the 32 lanes in a fixed xor tree; every lane ends with the same bits.
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
}

__device__ __forceinline__ float warp_allsum(float v) {
#pragma unroll
    for (int k = 1; k < 32; k <<= 1) v = __fadd_rn(v, __shfl_xor_sync(kFull, v, k));
    return v;
}

// One level of the transposed reduction: lanes with bit `off` clear keep entry a, the others entry b;
// each adds its partner's copy of the entry it keeps.
__device__ __forceinline__ float tr_comb(float a, float b, int off, int lane, int width) {
    const bool up = (lane & off) != 0;
    const float snd = up ? a : b;
    const float kp = up ? b : a;
    return __fadd_rn(kp, __shfl_xor_sync(kFull, snd, off, width));
}

// Pushes the partial of entry k (0 <= k < 2^LEVELS, warp-uniform) into the counter c[]; after the
// last entry, returns the reduced dot of entry `lane % 2^LEVELS` in `out`.
template <int LEVELS>
__device__ __forceinline__ void tr_push(float v, int k, float (&c)[5], float& out, int lane) {
    constexpr int WID = 1 << LEVELS;
    if (!(k & 1)) { c[0] = v; return; }
    v = tr_comb(c[0], v, 1, lane, WID);
    if (LEVELS == 1) { out = v; return; }
    if (!(k & 2)) { c[1] = v; return; }
    v = tr_comb(c[1], v, 2, lane, WID);
    if (LEVELS == 2) { out = v; return; }
    if (!(k & 4)) { c[2] = v; return; }
    v = tr_comb(c[2], v, 4, lane, WID);
    if (LEVELS == 3) { out = v; return; }
    if (!(k & 8)) { c[3] = v; return; }
    v = tr_comb(c[3], v, 8, lane, WID);
    if (LEVELS == 4) { out = v; return; }
    if (!(k & 16)) { c[4] = v; return; }
    out = tr_comb(c[4], v, 16, lane, WID);
}

// Full transposed reduction of p[0..2^LEVELS) (entry k's partial in p[k], every lane): lane l ends with the
// sum over the 2^LEVELS-lane segment of entry l % 2^LEVELS, over the same lane tree as tr_push (level k pairs
// lanes l, l^(1<<k)), so the bits equal tr_push's. All shuffles of a level are independent (ILP).
template <int LEVELS>
__device__ __forceinline__ float tr_full(float (&p)[32], int lane) {
    constexpr int WID = 1 << LEVELS;
#pragma unroll
    for (int lev = 0; lev < LEVELS; ++lev) {
        const int off = 1 << lev;
        const int cnt = WID >> (lev + 1);       // pairs at this level
#pragma unroll
        for (int i = 0; i < cnt; ++i) p[i] = tr_comb(p[2 * i], p[2 * i + 1], off, lane, WID);
    }
    return p[0];
}

// ------------------------------------------------------------------------------------------------
// Records: per sorted write position {pay, abs slot, f32 -beta*(kv*exp(log_k)), f32 cpe}.
template <typename IdxT, typename ValT>
__global__ void __launch_bounds__(256) dc_records_pa_kernel(
    const int32_t* __restrict__ kpay, const IdxT* __restrict__ k_idx, const ValT* __restrict__ kv,
    const float* __restrict__ logk, const float* __restrict__ beta, const ValT* __restrict__ cpe,
    int4* __restrict__ rec, int64_t total, int n, int C, int W) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t chunk = i / n;
    const uint32_t pay = static_cast<uint32_t>(kpay[i]);
    const int tok = (pay >> 9) & 0xFF;
    const int64_t o = chunk * n + static_cast<int64_t>(tok) * W + (pay & 0xFF);
    int32_t slot = 0;
    float sc = 0.0f, c = 1.0f;
    if (!(pay & kInvalid)) {
        slot = static_cast<int32_t>(k_idx[o]);
        sc = -beta[chunk * C + tok] * (f_of(kv[o]) * expf(logk[o]));
        c = f_of(cpe[o]);
    }
    rec[i] = make_int4(static_cast<int32_t>(pay), slot, __float_as_int(sc), __float_as_int(c));
}

__device__ __forceinline__ int4 load_rec(const int4* __restrict__ r, int j, int e) {
    if (j < e) return __ldg(r + j);
    return make_int4(static_cast<int32_t>(kFirst | kLast | kInvalid), 0, 0, 0);
}

// ------------------------------------------------------------------------------------------------
// Phase A. Grid: P * NSL * ctas blocks (block b: cta = b % ctas, slice = b / ctas % NSL, p = rest).
// A 32-position window is walked as 4 sub-batches of 8 positions, each fully unrolled (static u): FIRST / LAST / valid
// come from warp-uniform ballot masks, the token offset and scale of position k from one shuffle each of the window
// registers, and the per-entry dot partials and the group <gdw, snap> partials go into static register arrays, reduced
// per sub-batch over lane levels 0-2 (xor 1, 2, 4) and across sub-batches over levels 3-4 (xor 8, 16): the lane tree of
// tr_push / warp_allsum.
// EG: grad-row dtype (the bank's, or f32 with the optional f32 grad bank); EB: bank / snapshot dtype. The launchers
// instantiate D = 8 and V4 = NS = TR = false. WIDE (dc_pa: snapshots of 2^31 rows or more, or the host-only dbg bit
// kDcWide): the block's 64-bit base snapshot row is folded into the snapshot pointer and the walk's snapshot rows are
// chunk-local int32 (gx_pa2's scheme); the same rows are read, so the outputs are the same bits. Without WIDE the
// snapshot rows are global int32, as before.
template <typename EG, typename EB, int D, bool V4, bool NS, bool TR, bool WIDE>
__global__ void __launch_bounds__(DC_THREADS, DC_MINB) dc_pa3_kernel(
    EG* __restrict__ grad, EB* __restrict__ bank, const EB* __restrict__ snap, int64_t snap_step_rows,
    const float* __restrict__ dva, const float* __restrict__ beta, const int32_t* __restrict__ kpay,
    const int4* __restrict__ rec, float* __restrict__ out_eff, float* __restrict__ out_g,
    int64_t plane, int ti, int P, int C, int W, int n, int hd, int NSL, int SLW, int ctas, int L, int flags,
    const int32_t* __restrict__ bpre, const EG* __restrict__ gdwc) {
    using RowG = typename RowOf<EG>::T;
    using RowS = typename RowOf<EB>::T;
    constexpr int kRowG = static_cast<int>(sizeof(RowG));
    constexpr int kRowS = static_cast<int>(sizeof(RowS));
    extern __shared__ float4 smem[];
    const bool snapc = (flags & 256) != 0;
    const bool dense = (flags & 1024) != 0;    // compact rows by valid-group ordinal (needs snapc and bpre)
    const bool nofetch = (flags & 8192) != 0;  // dc_pa's dbg bit 8192 (never set by the package's backward): no slot-row fetch (wrong results; timing only)
    const bool gdc = (flags & 32768) != 0;     // gdw rows from the per-step dense gdwc (row p * n + ordinal); dbg bit 32768, never set by the package's backward
    const bool l2d = (flags & 65536) != 0;     // L2 bulk prefetch of the next window's dense rows (NSL == 1); dbg bit 65536, never set by the package's backward
    const bool ordi = dense || gdc;            // track the dense valid-group ordinal
    const bool nostore = (flags & 16384) != 0; // dc_pa's dbg bit 16384 (never set by the package's backward): no grad / bank row stores (wrong results; timing only)
    const int nw = blockDim.x >> 5;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned lt_mask = (1u << lane) - 1u;
    float4* tok_s = smem;                                                         // [C][32] (not with NS)
    float* beta_s = reinterpret_cast<float*>(tok_s + C * 32);                     // [C]
    RowG* const gring_all = reinterpret_cast<RowG*>(beta_s + ((C + 3) & ~3));        // [nw][D][32] grad rows
    RowG* const gring = gring_all + warp * D * 32;
    RowS* const sring_all = reinterpret_cast<RowS*>(gring_all + nw * D * 32);          // [nw][D][32] snap
    RowS* const sring = sring_all + warp * D * 32;
    int b = blockIdx.x;
    const int cta = b % ctas; b /= ctas;
    const int sl = b % NSL;
    const int p = b / NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * P + p;
    const int64_t tok0 = chunk * C;
    const int col0 = sl * SLW + 4 * lane;
    const bool on = 4 * lane < SLW && col0 < hd;
    {
        const float* rb = dva + tok0 * hd;
        for (int t = warp; t < C; t += nw) {
            cp_async<16>(&tok_s[t * 32 + lane], rb + static_cast<int64_t>(t) * hd + (on ? col0 : 0),
                         on ? 16 : 0);
        }
        cp_commit();
        for (int i = threadIdx.x; i < C; i += blockDim.x) beta_s[i] = beta[tok0 + i];
        cp_wait<0>();
        __syncthreads();
    }
    const int32_t* kp = kpay + chunk * n;
    const int4* rp = rec + chunk * n;
    const int tile = cta * nw + warp;
    const int s = snap_back(kp, tile * L, n, lane);
    const int e = snap_back(kp, (tile + 1) * L, n, lane);
    const EB* const snap_col = snap + (on ? col0 : 0);
    EG* const grad_col = grad + (on ? col0 : 0);
    const EG* const gsrc_col = gdc ? gdwc + (on ? col0 : 0) : grad_col;   // where the pump reads gdw rows
    const int64_t gbase = gdc ? static_cast<int64_t>(p) * n : 0;
    const int64_t snap_base = snapc ? chunk * n : static_cast<int64_t>(ti) * snap_step_rows +
                                                  static_cast<int64_t>(p) * C * W;
    // WIDE: the block's base row in the pointers (64-bit), the rows below chunk-local; else snap itself (global rows).
    const EB* const snap_w = WIDE ? snap + snap_base * hd : snap;
    const EB* const snap_colw = WIDE ? snap_col + snap_base * hd : snap_col;
    float* const oeff = out_eff + (NSL == 1 ? 0 : static_cast<int64_t>(sl) * plane);
    float* const og = out_g + (NSL == 1 ? 0 : static_cast<int64_t>(sl) * plane);
    const int64_t obase = chunk * (static_cast<int64_t>(C) * W);
    int4 w0 = load_rec(rp, s + lane, e);
    int4 w1 = load_rec(rp, s + 32 + lane, e);
    int4 w2 = load_rec(rp, s + 64 + lane, e);
    int jb = s, issued = 0, seg = 0;
    // Pump state, two record windows: bit u of pend0 (pend1) = position jb+u (jb+32+u) is a group FIRST whose
    // rows are not issued yet; rdm = non-padding bits; gi/si = this lane's grad slot / snapshot row (32-bit; WIDE:
    // relative to snap_w, sbase64 = 0: a chunk-local row below n + 64 for each of the three readers).
    const int64_t sbase64 = WIDE ? int64_t(0) : snap_base;
#define DC3_FIRSTS(WR, J0) __ballot_sync(kFull, (static_cast<uint32_t>(WR.x) & kFirst) && (J0) + lane < e)
#define DC3_RDS(WR) __ballot_sync(kFull, !(static_cast<uint32_t>(WR.x) & kInvalid))
#define DC3_SROW(WR, J0) static_cast<int>(snapc ? sbase64 + (J0) + lane :                       \
        sbase64 + static_cast<int64_t>((static_cast<uint32_t>(WR.x) >> 9) & 0xFF) * W +          \
        (static_cast<uint32_t>(WR.x) & 0xFF))
    unsigned pend0 = DC3_FIRSTS(w0, s), pend1 = DC3_FIRSTS(w1, s + 32);
    unsigned rdm0 = DC3_RDS(w0), rdm1 = DC3_RDS(w1);
    int gi0 = w0.y, gi1 = w1.y;
    // Dense rows: ordinal of the tile's first valid FIRST = block prefix + valid FIRSTs in [32 * (s / 32), s);
    // then a running count per window (vf = the window's in-tile valid FIRSTs, captured before the pump eats it).
    int gw0 = 0;
    if (ordi) {
        const int nb = (n + 31) >> 5;
        const int bs = s >> 5;
        const uint32_t pq = (32 * bs + lane < s) ? static_cast<uint32_t>(kp[32 * bs + lane]) : 0u;
        gw0 = bpre[chunk * (nb + 1) + bs] + __popc(__ballot_sync(kFull, (pq & kFirst) && !(pq & kInvalid)));
    }
    unsigned vf1 = pend1 & rdm1;
    int gw1 = gw0 + __popc(pend0 & rdm0);
#define DC3_SROWD(VF, G) static_cast<int>(sbase64 + (G) + __popc((VF) & lt_mask))
    int si0 = dense ? DC3_SROWD(pend0 & rdm0, gw0) : DC3_SROW(w0, s);
    int si1 = dense ? DC3_SROWD(vf1, gw1) : DC3_SROW(w1, s + 32);
#define DC3_GROWD(VF, G) static_cast<int>(gbase + (G) + __popc((VF) & lt_mask))
    if (gdc) {
        gi0 = DC3_GROWD(pend0 & rdm0, gw0);
        gi1 = DC3_GROWD(vf1, gw1);
    }
    // Dense rows of a window's groups are contiguous: one L2 bulk prefetch per array, one window ahead.
    const bool pf_rows = l2d && NSL == 1 && (static_cast<int64_t>(hd) * sizeof(EG)) % 16 == 0 &&
                         (static_cast<int64_t>(hd) * sizeof(EB)) % 16 == 0;
#define DC3_L2D(VF, G)                                                                         \
    if (pf_rows && lane == 0 && (VF) != 0u) {                                                  \
        const int cnt_ = __popc(VF);                                                           \
        if (gdc) l2_prefetch(gdwc + (gbase + (G)) * hd, cnt_ * hd * static_cast<int>(sizeof(EG))); \
        if (dense) l2_prefetch(snap_w + (sbase64 + (G)) * hd, cnt_ * hd * static_cast<int>(sizeof(EB))); \
    }
    DC3_L2D(pend0 & rdm0, gw0)
    DC3_L2D(vf1, gw1)
#define DC3_PUMP()                                                                             \
    while (!nofetch && issued < seg + D && (pend0 | pend1)) {                                  \
        const bool lo_ = pend0 != 0u;                                                          \
        const int u_ = __ffs(lo_ ? pend0 : pend1) - 1;                                         \
        const int g_ = __shfl_sync(kFull, lo_ ? gi0 : gi1, u_);                                \
        const int r_ = __shfl_sync(kFull, lo_ ? si0 : si1, u_);                                \
        const bool rd_ = on && ((((lo_ ? rdm0 : rdm1) >> u_) & 1u) != 0u);                     \
        const int rs_ = (issued & (D - 1)) * 32 + lane;                                        \
        cp_async<kRowG>(gring + rs_, gsrc_col + (rd_ ? static_cast<int64_t>(g_) * hd : 0),     \
                        rd_ ? kRowG : 0);                                                      \
        cp_async<kRowS>(sring + rs_, snap_colw + (rd_ ? static_cast<int64_t>(r_) * hd : 0),    \
                        rd_ ? kRowS : 0);                                                      \
        cp_commit();                                                                           \
        ++issued;                                                                              \
        if (lo_) pend0 &= pend0 - 1u; else pend1 &= pend1 - 1u;                                \
    }
    DC3_PUMP();

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
            // The ring holds D >= 8 groups ahead, so after this pump every FIRST of the next 8 positions has
            // been issued; one wait covers them all (no pump or wait inside the unrolled body).
            const unsigned fsb = (fm >> (sb * 8)) & 0xFFu;
            if (fsb && !nofetch) {
                DC3_PUMP();
                cp_wait_dyn(issued - seg - __popc(fsb));
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int k = sb * 8 + u;
                const int tk = __shfl_sync(kFull, tko_l, k);
                const float sck = __int_as_float(__shfl_sync(kFull, w0.z, k));
                const float4 x = tok_s[tk + lane];
                if ((fsb >> u) & 1u) {
                    acc = make_float4(0.f, 0.f, 0.f, 0.f);
                    const int rs = (seg % D) * 32 + lane;
                    if (!nofetch) {
                        gdw = to_f4(gring[rs]);
                        sn_raw = sring[rs];
                    }
                    sn = to_f4(sn_raw);
                    ++seg;
                    gp = __fmul_rn(gdw.x, sn.x);
                    gp = __fmaf_rn(gdw.y, sn.y, gp);
                    gp = __fmaf_rn(gdw.z, sn.z, gp);
                    gp = __fmaf_rn(gdw.w, sn.w, gp);
                }
                // No valid branch: a padding entry is a singleton group (FIRST|LAST|invalid) with scale 0 and
                // zero-filled ring rows; its row is never stored and its outputs are forced to +0 below.
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
                    const int64_t slot = __shfl_sync(kFull, w0.y, k);
                    const float cpe_l = __int_as_float(__shfl_sync(kFull, w0.w, k));
                    if (on) {
                        float4 tot = acc;
                        float4 o;
                        o.x = __fmaf_rn(gdw.x, cpe_l, tot.x);
                        o.y = __fmaf_rn(gdw.y, cpe_l, tot.y);
                        o.z = __fmaf_rn(gdw.z, cpe_l, tot.z);
                        o.w = __fmaf_rn(gdw.w, cpe_l, tot.w);
                        if (!nostore) {
                            store4(grad + slot * hd + col0, o);
                            store_raw(bank + slot * hd + col0, sn_raw);
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
            float vg = 0.f;
            if (fsb) {
#pragma unroll
                for (int i = 0; i < 4; ++i) pg[i] = tr_comb(pg[2 * i], pg[2 * i + 1], 1, lane, 32);
#pragma unroll
                for (int i = 0; i < 2; ++i) pg[i] = tr_comb(pg[2 * i], pg[2 * i + 1], 2, lane, 32);
                vg = tr_comb(pg[0], pg[1], 4, lane, 32);
            } else {
                // No FIRST in these 8 positions: every pg[u] is the same gp, so the transposed levels 0-2
                // reduce to the octet butterfly of gp (the same lane tree, the same bits).
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
        pend0 = pend1; rdm0 = rdm1; gi0 = gi1; si0 = si1;
        pend1 = DC3_FIRSTS(w1, jb + 32); rdm1 = DC3_RDS(w1); gi1 = w1.y;
        gw1 += __popc(vf1);
        vf1 = pend1 & rdm1;
        si1 = dense ? DC3_SROWD(vf1, gw1) : DC3_SROW(w1, jb + 32);
        if (gdc) gi1 = DC3_GROWD(vf1, gw1);
        DC3_L2D(vf1, gw1)
        DC3_PUMP();
    }
    cp_wait<0>();
#undef DC3_PUMP
#undef DC3_FIRSTS
#undef DC3_RDS
#undef DC3_SROW
#undef DC3_SROWD
#undef DC3_GROWD
#undef DC3_L2D
}

// Sums slice partial planes in slice order: d_k_eff = -beta * sum, grad_g = cpe * sum, d_k_pd = sum.
template <typename ValT>
__global__ void __launch_bounds__(256) dc_pa_reduce_kernel(
    const float* __restrict__ pe, const float* __restrict__ pg, const float* __restrict__ pp,
    const float* __restrict__ beta, const ValT* __restrict__ cpe, float* __restrict__ d_k_eff,
    float* __restrict__ grad_g, float* __restrict__ d_k_pd, int64_t total, int ne, int ng, int np,
    int P, int C, int W, int64_t T) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / W;                 // chunk * C + tok
    const int64_t chunk = row / C;
    const int tok = static_cast<int>(row - chunk * C);
    const bool valid = (chunk / P) * C + tok < T;
    if (ne > 0) {
        float s = pe[i];
        for (int k = 1; k < ne; ++k) s = __fadd_rn(s, pe[static_cast<int64_t>(k) * total + i]);
        d_k_eff[i] = valid ? -beta[row] * s : 0.f;
        float g = pg[i];
        for (int k = 1; k < ng; ++k) g = __fadd_rn(g, pg[static_cast<int64_t>(k) * total + i]);
        grad_g[i] = valid ? f_of(cpe[i]) * g : 0.f;
    }
    if (np > 0) {
        float s = pp[i];
        for (int k = 1; k < np; ++k) s = __fadd_rn(s, pp[static_cast<int64_t>(k) * total + i]);
        d_k_pd[i] = valid ? s : 0.f;
    }
}

// ------------------------------------------------------------------------------------------------
// GWR of the deterministic backward: token_gather with one operand and no stash, plus d_k_pd per entry. The
// d_delta_v accumulation is token_gather's (same row groups, same order, same partial sums).
template <int B> struct Bits;
template <> struct Bits<2> { using type = uint16_t; };
template <> struct Bits<4> { using type = uint32_t; };
template <> struct Bits<16> { using type = uint4; };
__device__ __forceinline__ float to_float(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ float to_float(uint32_t b) { return __uint_as_float(b); }

struct GArgs {
    const void* bank; const void* idx; const void* w; float* out; const float* dv; float* pd;
    int64_t row0, out_row0, plane;
    int hd, vpr, tokens, chunk, valid, per_group, width;
    bool idx64, w_bf16;
    const int* kpos;   // optional [rows, width]: dense valid-group ordinal of a FIRST entry, else -1
    void* gdwc;        // with kpos: per-step compact grad rows, row (chunk - row0 / chunk) * crow + ordinal
    int crow;
};
__device__ __forceinline__ int load_slot(const void* p, int64_t i, bool idx64) {
    return idx64 ? static_cast<int>(__ldg(static_cast<const long long*>(p) + i))
                 : __ldg(static_cast<const int*>(p) + i);
}
__device__ __forceinline__ float load_weight(const void* p, int64_t i, bool bf16) {
    return bf16 ? to_float(__ldg(static_cast<const unsigned short*>(p) + i))
                : __ldg(static_cast<const float*>(p) + i);
}

// Programmatic dependent launch: the deterministic backward launches step ti's GWR (ti < last) right after step ti +
// 1's GDS, which triggers its dependents once its own wait has returned. The delta_v / slot / weight / kpos loads
// (saved by the forward, written before the backward's step loop) run before griddepcontrol.wait; every grad-bank row
// is read after it through the coherent L2 path (__ldcg: the GDS has just written them), and no store precedes it.
// Right after its wait the grid lets its dependents launch (the cuBLAS baddbmm that follows may be a programmatic
// launch; a no-op otherwise). Arithmetic and order are unchanged.
#define FX_LDG_ROW(p) __ldcg(p)
#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)   // a modern device pass (sdm_copy.cuh); PDL: modern only
#define FX_PDL_WAIT_LAUNCH() asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory")
#else
#define FX_PDL_WAIT_LAUNCH() ((void)0)
#endif

template <typename E, int VB, int LPR>
__global__ void __launch_bounds__(512) dc_gwr_kernel(const GArgs a) {
    using V = typename Bits<VB>::type;
    constexpr int VEC = VB / sizeof(E);
    constexpr int STEP = (sizeof(E) == 4 ? 8 : 4);
    constexpr int SLICE = LPR * VEC;
    constexpr int PADW = LPR + 4;      // dot-buffer row stride (floats): 16-byte rows, conflict-free reads
    extern __shared__ float4 smem[];
    float* partial = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x, group = threadIdx.y, groups = blockDim.y;
    // Per row group: [LPR entries][PADW] f32 dot partials (entry k's partial of lane l at k * PADW + l).
    float* dbuf = partial + (groups > 1 ? blockDim.z * groups * SLICE : 0) +
                  (threadIdx.z * groups + group) * (LPR * PADW + LPR);   // +LPR skews row-group banks
    const int t = blockIdx.x * blockDim.z + threadIdx.z;
    const bool in_range = t < a.tokens;
    const bool live = in_range && (a.valid == a.chunk || t % a.chunk < a.valid);
    const int v = blockIdx.y * LPR + lane;
    const bool lane_on = v < a.vpr;
    const int64_t col = int64_t(v) * VEC;
    const int64_t row = a.row0 + t;
    const E* bank = static_cast<const E*>(a.bank);
    float* pd = a.pd + (gridDim.y == 1 ? 0 : static_cast<int64_t>(blockIdx.y) * a.plane);
    float dvr[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i)
        dvr[i] = (live && lane_on) ? __ldg(a.dv + row * a.hd + col + i) : 0.0f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    for (int jb = 0; jb < a.per_group; jb += LPR) {
        const int nl = group + groups * (jb + lane);
        const bool okl = live && jb + lane < a.per_group && nl < a.width;
        const int64_t il = row * a.width + nl;
        const int slot_l = okl ? load_slot(a.idx, il, a.idx64) : 0;
        const float w_l = okl ? load_weight(a.w, il, a.w_bf16) : 0.0f;
        const int kp_l = (a.kpos != nullptr && okl) ? __ldg(a.kpos + il) : -1;
        if (jb == 0) FX_PDL_WAIT_LAUNCH();   // PDL: the previous grid (GDS) has completed; before any grad-bank read
        const int batch = min(LPR, a.per_group - jb);
        for (int s0 = 0; s0 < LPR; s0 += STEP) {
          if (s0 >= batch) {
#pragma unroll
            for (int s = 0; s < STEP; ++s) dbuf[(s0 + s) * PADW + lane] = 0.f;
          } else {
            const V* src[STEP];
            bool load[STEP];
            float w[STEP];
#pragma unroll
            for (int s = 0; s < STEP; ++s) {
                const int n = group + groups * (jb + s0 + s);
                const int slot = __shfl_sync(kFull, slot_l, (s0 + s) & (LPR - 1), LPR);
                w[s] = __shfl_sync(kFull, w_l, (s0 + s) & (LPR - 1), LPR);
                src[s] = reinterpret_cast<const V*>(bank + int64_t(slot) * a.hd + col);
                load[s] = live && lane_on && s0 + s < batch && n < a.width;
            }
            V r[STEP];
#pragma unroll
            for (int s = 0; s < STEP; ++s) r[s] = load[s] ? FX_LDG_ROW(src[s]) : V{};
            if (a.kpos != nullptr) {   // gdwc: the raw grad row of each valid group FIRST, dense by ordinal
#pragma unroll
                for (int s = 0; s < STEP; ++s) {
                    const int kj = __shfl_sync(kFull, kp_l, (s0 + s) & (LPR - 1), LPR);
                    if (kj >= 0 && load[s])
                        *reinterpret_cast<V*>(static_cast<E*>(a.gdwc) +
                                              ((row / a.chunk - a.row0 / a.chunk) * a.crow + kj) *
                                                  static_cast<int64_t>(a.hd) + col) = r[s];
                }
            }
#pragma unroll
            for (int s = 0; s < STEP; ++s) {
                const E* el = reinterpret_cast<const E*>(&r[s]);
#pragma unroll
                for (int i = 0; i < VEC; ++i) acc[i] = fmaf(w[s], to_float(el[i]), acc[i]);
                float pdp = __fmul_rn(dvr[0], to_float(el[0]));
#pragma unroll
                for (int i = 1; i < VEC; ++i) pdp = __fmaf_rn(dvr[i], to_float(el[i]), pdp);
                dbuf[(s0 + s) * PADW + lane] = pdp;
            }
          }
        }
        __syncwarp();
        // Lane l sums entry l's LPR partials pairwise in lane-index order: the xor tree of tr_push/tr_full.
        float tv[LPR];
#pragma unroll
        for (int j = 0; j < LPR; j += 4) {
            const float4 q = *reinterpret_cast<const float4*>(dbuf + lane * PADW + j);
            tv[j] = q.x; tv[j + 1] = q.y; tv[j + 2] = q.z; tv[j + 3] = q.w;
        }
#pragma unroll
        for (int wdt = LPR / 2; wdt >= 1; wdt >>= 1) {
#pragma unroll
            for (int j = 0; j < wdt; ++j) tv[j] = __fadd_rn(tv[2 * j], tv[2 * j + 1]);
        }
        const float dot_mine = tv[0];
        __syncwarp();
        if (in_range && okl) pd[il] = dot_mine;
        else if (in_range && jb + lane < a.per_group && nl < a.width) pd[il] = 0.0f;
    }

    FX_PDL_WAIT_LAUNCH();   // PDL: also when no entry loop ran (no store, no exit before the wait)
    if (groups == 1) {
        if (in_range && lane_on) {
#pragma unroll
            for (int i = 0; i < VEC; i += (VEC % 4 == 0 ? 4 : 1)) {
                if constexpr (VEC % 4 == 0) {
                    *reinterpret_cast<float4*>(a.out + (a.out_row0 + t) * a.hd + col + i) =
                        make_float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
                } else {
                    a.out[(a.out_row0 + t) * a.hd + col + i] = acc[i];
                }
            }
        }
        return;
    }
    {
        float* dst = partial + (threadIdx.z * groups + group) * SLICE;
#pragma unroll
        for (int i = 0; i < VEC; ++i) dst[lane * VEC + i] = acc[i];
    }
    __syncthreads();
    const int tid = lane + LPR * (group + groups * threadIdx.z);
    const int width = min(SLICE, a.hd - int(blockIdx.y) * SLICE);
    for (int q = tid; q < blockDim.z * width; q += LPR * groups * blockDim.z) {
        const int z = q / width, c = q - z * width;
        const int tok = blockIdx.x * blockDim.z + z;
        if (tok >= a.tokens) continue;
        const float* src = partial + z * groups * SLICE + c;
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) sum += src[g * SLICE];
        a.out[(a.out_row0 + tok) * a.hd + blockIdx.y * SLICE + c] = sum;
    }
}

// A 64-register entry point (dc_gwr_r64_kernel) for the 256-thread launches of the bf16 / 16-byte / 32-lane instance:
// at 72 registers only 3 such CTAs fit an SM, at <= 64 registers 4 do (a grid of 512 CTAs then runs in one wave on
// GB300). dc_gwr_body_cap is dc_gwr_kernel's body copied verbatim except (1) the signature, (2) STEP = STEPX rows in
// flight (DC_GWR_CAP_STEP = 2: 63 registers and no spill; 4 would spill at 64), (3) the zero-fill bound zlim, which
// makes a smaller STEP process exactly STEP0's entries (padding included): every output element sees the same
// operations in the same order.
#ifndef DC_GWR_CAP_STEP
#define DC_GWR_CAP_STEP 2
#endif
template <typename E, int VB, int LPR, int STEPX>
__device__ __forceinline__ void dc_gwr_body_cap(const GArgs a) {
    using V = typename Bits<VB>::type;
    constexpr int VEC = VB / sizeof(E);
    constexpr int STEP0 = (sizeof(E) == 4 ? 8 : 4);
    constexpr int STEP = STEPX ? STEPX : STEP0;   // rows in flight (STEPX divides STEP0)
    static_assert(STEP >= 1 && (STEP0 % STEP) == 0, "STEPX must divide the default STEP");
    constexpr int SLICE = LPR * VEC;
    constexpr int PADW = LPR + 4;      // dot-buffer row stride (floats): 16-byte rows, conflict-free reads
    extern __shared__ float4 smem[];
    float* partial = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x, group = threadIdx.y, groups = blockDim.y;
    // Per row group: [LPR entries][PADW] f32 dot partials (entry k's partial of lane l at k * PADW + l).
    float* dbuf = partial + (groups > 1 ? blockDim.z * groups * SLICE : 0) +
                  (threadIdx.z * groups + group) * (LPR * PADW + LPR);   // +LPR skews row-group banks
    const int t = blockIdx.x * blockDim.z + threadIdx.z;
    const bool in_range = t < a.tokens;
    const bool live = in_range && (a.valid == a.chunk || t % a.chunk < a.valid);
    const int v = blockIdx.y * LPR + lane;
    const bool lane_on = v < a.vpr;
    const int64_t col = int64_t(v) * VEC;
    const int64_t row = a.row0 + t;
    const E* bank = static_cast<const E*>(a.bank);
    float* pd = a.pd + (gridDim.y == 1 ? 0 : static_cast<int64_t>(blockIdx.y) * a.plane);
    float dvr[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i)
        dvr[i] = (live && lane_on) ? __ldg(a.dv + row * a.hd + col + i) : 0.0f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    for (int jb = 0; jb < a.per_group; jb += LPR) {
        const int nl = group + groups * (jb + lane);
        const bool okl = live && jb + lane < a.per_group && nl < a.width;
        const int64_t il = row * a.width + nl;
        const int slot_l = okl ? load_slot(a.idx, il, a.idx64) : 0;
        const float w_l = okl ? load_weight(a.w, il, a.w_bf16) : 0.0f;
        const int kp_l = (a.kpos != nullptr && okl) ? __ldg(a.kpos + il) : -1;
        if (jb == 0) FX_PDL_WAIT_LAUNCH();   // PDL: the previous grid (GDS) has completed; before any grad-bank read
        const int batch = min(LPR, a.per_group - jb);
        // STEP0 computes entries batch .. ceil(batch / STEP0) * STEP0 - 1 as padding (zero rows, zero weight)
        // and zero-fills the rest; a smaller STEP pads to the same zlim, so acc and dbuf see STEP0's operations exactly.
        const int zlim = (batch + STEP0 - 1) / STEP0 * STEP0;
        for (int s0 = 0; s0 < LPR; s0 += STEP) {
          if (s0 >= zlim) {
#pragma unroll
            for (int s = 0; s < STEP; ++s) dbuf[(s0 + s) * PADW + lane] = 0.f;
          } else {
            const V* src[STEP];
            bool load[STEP];
            float w[STEP];
#pragma unroll
            for (int s = 0; s < STEP; ++s) {
                const int n = group + groups * (jb + s0 + s);
                const int slot = __shfl_sync(kFull, slot_l, (s0 + s) & (LPR - 1), LPR);
                w[s] = __shfl_sync(kFull, w_l, (s0 + s) & (LPR - 1), LPR);
                src[s] = reinterpret_cast<const V*>(bank + int64_t(slot) * a.hd + col);
                load[s] = live && lane_on && s0 + s < batch && n < a.width;
            }
            V r[STEP];
#pragma unroll
            for (int s = 0; s < STEP; ++s) r[s] = load[s] ? FX_LDG_ROW(src[s]) : V{};
            if (a.kpos != nullptr) {   // gdwc: the raw grad row of each valid group FIRST, dense by ordinal
#pragma unroll
                for (int s = 0; s < STEP; ++s) {
                    const int kj = __shfl_sync(kFull, kp_l, (s0 + s) & (LPR - 1), LPR);
                    if (kj >= 0 && load[s])
                        *reinterpret_cast<V*>(static_cast<E*>(a.gdwc) +
                                              ((row / a.chunk - a.row0 / a.chunk) * a.crow + kj) *
                                                  static_cast<int64_t>(a.hd) + col) = r[s];
                }
            }
#pragma unroll
            for (int s = 0; s < STEP; ++s) {
                const E* el = reinterpret_cast<const E*>(&r[s]);
#pragma unroll
                for (int i = 0; i < VEC; ++i) acc[i] = fmaf(w[s], to_float(el[i]), acc[i]);
                float pdp = __fmul_rn(dvr[0], to_float(el[0]));
#pragma unroll
                for (int i = 1; i < VEC; ++i) pdp = __fmaf_rn(dvr[i], to_float(el[i]), pdp);
                dbuf[(s0 + s) * PADW + lane] = pdp;
            }
          }
        }
        __syncwarp();
        // Lane l sums entry l's LPR partials pairwise in lane-index order: the xor tree of tr_push/tr_full.
        float tv[LPR];
#pragma unroll
        for (int j = 0; j < LPR; j += 4) {
            const float4 q = *reinterpret_cast<const float4*>(dbuf + lane * PADW + j);
            tv[j] = q.x; tv[j + 1] = q.y; tv[j + 2] = q.z; tv[j + 3] = q.w;
        }
#pragma unroll
        for (int wdt = LPR / 2; wdt >= 1; wdt >>= 1) {
#pragma unroll
            for (int j = 0; j < wdt; ++j) tv[j] = __fadd_rn(tv[2 * j], tv[2 * j + 1]);
        }
        const float dot_mine = tv[0];
        __syncwarp();
        if (in_range && okl) pd[il] = dot_mine;
        else if (in_range && jb + lane < a.per_group && nl < a.width) pd[il] = 0.0f;
    }

    FX_PDL_WAIT_LAUNCH();   // PDL: also when no entry loop ran (no store, no exit before the wait)
    if (groups == 1) {
        if (in_range && lane_on) {
#pragma unroll
            for (int i = 0; i < VEC; i += (VEC % 4 == 0 ? 4 : 1)) {
                if constexpr (VEC % 4 == 0) {
                    *reinterpret_cast<float4*>(a.out + (a.out_row0 + t) * a.hd + col + i) =
                        make_float4(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
                } else {
                    a.out[(a.out_row0 + t) * a.hd + col + i] = acc[i];
                }
            }
        }
        return;
    }
    {
        float* dst = partial + (threadIdx.z * groups + group) * SLICE;
#pragma unroll
        for (int i = 0; i < VEC; ++i) dst[lane * VEC + i] = acc[i];
    }
    __syncthreads();
    const int tid = lane + LPR * (group + groups * threadIdx.z);
    const int width = min(SLICE, a.hd - int(blockIdx.y) * SLICE);
    for (int q = tid; q < blockDim.z * width; q += LPR * groups * blockDim.z) {
        const int z = q / width, c = q - z * width;
        const int tok = blockIdx.x * blockDim.z + z;
        if (tok >= a.tokens) continue;
        const float* src = partial + z * groups * SLICE + c;
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) sum += src[g * SLICE];
        a.out[(a.out_row0 + tok) * a.hd + blockIdx.y * SLICE + c] = sum;
    }
}

template <typename E, int VB, int LPR>
__global__ void __launch_bounds__(256, 4) dc_gwr_r64_kernel(const GArgs a) {
    dc_gwr_body_cap<E, VB, LPR, DC_GWR_CAP_STEP>(a);
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

}  // namespace

void dc_records_pa(at::Tensor kpay, at::Tensor k_idx, at::Tensor k_val, at::Tensor logk,
                   at::Tensor beta, at::Tensor cpe, at::Tensor rec) {
    const at::cuda::CUDAGuard guard(kpay.device());
    const int64_t N = kpay.size(0), n = kpay.size(1);
    const int C = static_cast<int>(k_idx.size(1)), W = static_cast<int>(k_idx.size(2));
    TORCH_CHECK(k_idx.size(0) == N && static_cast<int64_t>(C) * W == n, "dc_records_pa: shapes");
    TORCH_CHECK(rec.numel() == N * n * 4 && rec.scalar_type() == at::kInt && rec.is_contiguous() &&
                (reinterpret_cast<uintptr_t>(rec.data_ptr()) & 15) == 0, "dc_records_pa: rec");
    for (const at::Tensor* t : {&k_idx, &k_val, &logk, &beta, &cpe, &rec})
        TORCH_CHECK(t->device() == kpay.device(), "dc_records_pa: all operands on one device");
    TORCH_CHECK(cpe.scalar_type() == k_val.scalar_type() &&
                (k_val.scalar_type() == at::kBFloat16 || k_val.scalar_type() == at::kFloat),
                "dc_records_pa: k_val / cpe bf16 or f32 (the same)");
    TORCH_CHECK(logk.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat &&
                (k_idx.scalar_type() == at::kInt || k_idx.scalar_type() == at::kLong),
                "dc_records_pa: log_k / beta f32, k_idx int32/int64");
    TORCH_CHECK(k_idx.is_contiguous() && k_val.is_contiguous() && logk.is_contiguous() && beta.is_contiguous() &&
                cpe.is_contiguous() && kpay.is_contiguous() && kpay.scalar_type() == at::kInt,
                "dc_records_pa: contiguous operands");
    TORCH_CHECK(k_val.numel() == N * n && logk.numel() == N * n && cpe.numel() == N * n &&
                beta.numel() == N * C, "dc_records_pa: operand sizes");
    const int64_t total = N * n;
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>((total + 255) / 256);
#define DC_RECA(IDX, VAL)                                                                    \
    dc_records_pa_kernel<IDX, VAL><<<blocks, 256, 0, stream>>>(                              \
        kpay.data_ptr<int32_t>(), k_idx.data_ptr<IDX>(),                                     \
        reinterpret_cast<const VAL*>(k_val.data_ptr()), logk.data_ptr<float>(),              \
        beta.data_ptr<float>(), reinterpret_cast<const VAL*>(cpe.data_ptr()),                \
        reinterpret_cast<int4*>(rec.data_ptr<int32_t>()), total, static_cast<int>(n), C, W)
    const bool i64 = k_idx.scalar_type() == at::kLong;
    const bool bf = k_val.scalar_type() == at::kBFloat16;
    if (i64 && bf) { DC_RECA(int64_t, __nv_bfloat16); }
    else if (i64) { DC_RECA(int64_t, float); }
    else if (bf) { DC_RECA(int32_t, __nv_bfloat16); }
    else { DC_RECA(int32_t, float); }
#undef DC_RECA
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Raw device copy of a contiguous tensor into a tensor of the same shape and any strides (no autograd version bump
// on dst): the aligned-scratch bank restore of the backward writes the caller's saved bank (which a saved-tensor
// unpack hook may hand over as a strided view), like the other raw kernels.
struct StridedDst {
    int64_t size[6];
    int64_t stride[6];
    int nd;
};
template <typename T>
__global__ void __launch_bounds__(256) dc_copy_strided_kernel(T* __restrict__ dst, const T* __restrict__ src,
                                                              int64_t n, StridedDst g) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t rem = i, off = 0;
        for (int d = g.nd - 1; d >= 0; --d) {
            const int64_t q = rem / g.size[d];
            off += (rem - q * g.size[d]) * g.stride[d];
            rem = q;
        }
        dst[off] = src[i];
    }
}

// Self-overlap of a strided view (some two elements share an address). Exact for views with at most 2 dimensions of
// size > 1 (the bank is 2-D): with strides a <= b and sizes na, nb, elements collide iff b/g <= na-1 and a/g <= nb-1,
// g = gcd(a, b); any size > 1 dimension of stride 0 collides. Higher ranks: non-overlap is accepted only when proven
// by the sorted-extent test (each stride exceeds the reach of the smaller ones); otherwise reported as overlapping.
static bool view_overlaps(const at::Tensor& t) {
    std::vector<std::pair<int64_t, int64_t>> d;   // (stride, size) of the dimensions of size > 1
    for (int64_t k = 0; k < t.dim(); ++k) {
        if (t.size(k) > 1) {
            if (t.stride(k) == 0) return true;
            d.emplace_back(t.stride(k) < 0 ? -t.stride(k) : t.stride(k), t.size(k));
        }
    }
    if (d.size() <= 1) return false;
    std::sort(d.begin(), d.end());
    if (d.size() == 2) {
        const int64_t a = d[0].first, na = d[0].second, b = d[1].first, nb = d[1].second;
        const int64_t g = std::gcd(a, b);
        return (b / g) <= na - 1 && (a / g) <= nb - 1;
    }
    int64_t reach = 0;
    for (const auto& sn : d) {
        if (sn.first <= reach) return true;
        reach += sn.first * (sn.second - 1);
    }
    return false;
}

bool dc_overlaps(at::Tensor t) { return !t.is_contiguous() && view_overlaps(t); }

void dc_copy_raw(at::Tensor dst, at::Tensor src) {
    const at::cuda::CUDAGuard guard(dst.device());
    TORCH_CHECK(dst.is_cuda() && src.device() == dst.device() && dst.scalar_type() == src.scalar_type() &&
                dst.sizes() == src.sizes() && src.is_contiguous(),
                "dc_copy_raw: same device, dtype and shape; a contiguous source");
    if (dst.numel() == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (dst.is_contiguous()) {
        C10_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), dst.nbytes(), cudaMemcpyDeviceToDevice,
                                       stream));
        return;
    }
    TORCH_CHECK(!view_overlaps(dst),
                "dc_copy_raw: some elements of the destination refer to a single memory location (a self-overlapping "
                "view cannot hold a bank)");
    TORCH_CHECK(dst.dim() <= 6, "dc_copy_raw: at most 6 dimensions");
    StridedDst g{};
    g.nd = static_cast<int>(dst.dim());
    for (int d = 0; d < g.nd; ++d) {
        g.size[d] = dst.size(d);
        g.stride[d] = dst.stride(d);
    }
    const int64_t n = dst.numel();
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>((n + 255) / 256, int64_t(1) << 20));
    if (dst.element_size() == 2) {
        dc_copy_strided_kernel<uint16_t><<<blocks, 256, 0, stream>>>(
            static_cast<uint16_t*>(dst.data_ptr()), static_cast<const uint16_t*>(src.data_ptr()), n, g);
    } else if (dst.element_size() == 4) {
        dc_copy_strided_kernel<uint32_t><<<blocks, 256, 0, stream>>>(
            static_cast<uint32_t*>(dst.data_ptr()), static_cast<const uint32_t*>(src.data_ptr()), n, g);
    } else {
        TORCH_CHECK(false, "dc_copy_raw: 2- or 4-byte elements");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dbg bit 1 << 18 (host-only, never passed to the kernel's flags): the WIDE walk (64-bit snapshot rows) at any size.
constexpr int64_t kDcWide = int64_t(1) << 18;

void dc_pa(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
           at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
           at::Tensor out_g, int64_t ti, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
           int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc) {
    const at::cuda::CUDAGuard guard(bank.device());
    const bool dense = (dbg & 1024) != 0;
    const bool gdc = (dbg & 32768) != 0;
    TORCH_CHECK(!gdc || ((dbg & 512) && depth == 8 && gdwc.scalar_type() == grad.scalar_type() &&
                         gdwc.is_contiguous() && gdwc.device() == bank.device() &&
                         gdwc.numel() >= P * kpay.size(1) * bank.size(1) &&
                         bpre.scalar_type() == at::kInt && bpre.is_contiguous() &&
                         bpre.device() == bank.device() &&
                         bpre.numel() == kpay.size(0) * ((kpay.size(1) + 31) / 32 + 1) &&
                         (reinterpret_cast<uintptr_t>(gdwc.data_ptr()) % 16) == 0),
                "dc_pa: gdwc needs the dense walk, depth 8, a [P * C * W, hd] grad-dtype buffer and bpre");
    (void)pipe;
    TORCH_CHECK((dbg & 512) && !(dbg & (2048 | 4096 | 131072)) && depth == 8,
                "dc_pa: only the dense walk is built (dbg bit 512, depth 8)");
    TORCH_CHECK(ti >= 0 && P >= 1 && C >= 1 && W >= 1, "dc_pa: ti >= 0, P, C, W >= 1");
    TORCH_CHECK(!dense || ((dbg & 256) && (dbg & 512) && depth == 8 && bpre.scalar_type() == at::kInt &&
                           bpre.is_contiguous() &&
                           bpre.numel() == kpay.size(0) * ((kpay.size(1) + 31) / 32 + 1) &&
                           bpre.device() == bank.device()),
                "dc_pa: the dense compact snapshot needs the dense walk, the compact flag and bpre [N, nb + 1]");
    const int hd = static_cast<int>(bank.size(1));
    const int n = static_cast<int>(kpay.size(1));
    TORCH_CHECK(hd % 4 == 0, "dc_pa: head_dim must be a multiple of 4, got ", hd);
    TORCH_CHECK(C <= 256 && n == C * W, "dc_pa: C <= 256 and kpay [N, C * W]");
    TORCH_CHECK(dva.scalar_type() == at::kFloat && dva.is_contiguous(), "dc_pa: d_dva f32");
    TORCH_CHECK(snap.scalar_type() == bank.scalar_type(), "dc_pa: snapshot dtype must be the bank's");
    const bool g32 = grad.scalar_type() == at::kFloat && bank.scalar_type() == at::kBFloat16;
    TORCH_CHECK(grad.scalar_type() == bank.scalar_type() || (g32 && (dbg & 512)),
                "dc_pa: grad must have the bank dtype (or f32 with a bf16 bank)");
    TORCH_CHECK(rec.scalar_type() == at::kInt && rec.numel() == kpay.numel() * 4, "dc_pa: rec");
    TORCH_CHECK(L > 0 && nw > 0 && nw <= 32, "dc_pa: L > 0 and 0 < nw <= 32");
    // Snapshots of 2^31 rows or more take the WIDE instantiation (chunk-local int32 snapshot rows from a 64-bit block
    // base); so does any size with the host-only dbg bit kDcWide (tests: the same bits as the 32-bit walk). Grad / bank
    // rows (the records' slots) stay 31-bit.
    const bool wide = (dbg & kDcWide) != 0 || snap.numel() / hd >= (int64_t(1) << 31);
    TORCH_CHECK(grad.size(0) < (int64_t(1) << 31), "dc_pa: row indices must fit 31 bits");
    TORCH_CHECK(grad.is_contiguous() && bank.is_contiguous() && snap.is_contiguous() &&
                kpay.is_contiguous() && rec.is_contiguous() && beta.is_contiguous() &&
                out_eff.is_contiguous() && out_g.is_contiguous(), "dc_pa: contiguous operands");
    const int64_t ralign = bank.scalar_type() == at::kBFloat16 ? 8 : 16;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(grad.data_ptr()) % (g32 ? 16 : ralign) == 0 &&
                reinterpret_cast<uintptr_t>(bank.data_ptr()) % ralign == 0 &&
                reinterpret_cast<uintptr_t>(snap.data_ptr()) % ralign == 0 &&
                reinterpret_cast<uintptr_t>(dva.data_ptr()) % 16 == 0,
                "dc_pa: row operands must be aligned to their vector size");
    const int nsl = (hd + 127) / 128;
    const int slw = 4 * (((hd / 4) + nsl - 1) / nsl);
    TORCH_CHECK(L >= 1 && L <= n && nw >= 1 && nw <= 32, "dc_pa: need 1 <= L <= C*W and 1 <= nw <= 32");
    const int64_t tiles64 = (static_cast<int64_t>(n) + L - 1) / L;
    const int64_t ctas64 = (tiles64 + nw - 1) / nw;
    TORCH_CHECK(ctas64 * nw * L < (int64_t(1) << 31) && P * nsl * ctas64 < (int64_t(1) << 31),
                "dc_pa: tile geometry overflows 32 bits");
    const int ctas = static_cast<int>(ctas64);
    const int blocks = static_cast<int>(P * nsl * ctas64);
    const int threads = static_cast<int>(nw) * 32;
    TORCH_CHECK(threads <= DC_THREADS, "dc_pa: at most ", DC_THREADS, " threads per CTA");
    TORCH_CHECK(kpay.size(0) >= (ti + 1) * P && dva.numel() >= (ti + 1) * P * C * hd &&
                beta.numel() >= (ti + 1) * P * C, "dc_pa: step ", ti, " out of range");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(rec.data_ptr()) & 15) == 0, "dc_pa: records 16-byte aligned");
    TORCH_CHECK(grad.dim() == 2 && bank.dim() == 2 && grad.size(0) == bank.size(0) && grad.size(1) == hd,
                "dc_pa: grad and bank [slots, hd]");
    TORCH_CHECK(snap_step_rows >= 0 && snap.numel() % hd == 0, "dc_pa: snapshot rows");
    {
        const int64_t need_rows = (dbg & 256) ? (ti + 1) * P * n : ti * snap_step_rows + P * C * W;
        TORCH_CHECK(need_rows * hd <= snap.numel(), "dc_pa: the snapshot does not cover step ", ti);
        const int64_t planes = nsl == 1 ? 1 : nsl;
        TORCH_CHECK(out_eff.numel() == out_g.numel() && out_eff.numel() % planes == 0 &&
                    out_eff.numel() / planes >= (ti + 1) * P * n && out_eff.scalar_type() == at::kFloat &&
                    out_g.scalar_type() == at::kFloat, "dc_pa: output planes do not cover step ", ti);
        TORCH_CHECK(rec.numel() >= (ti + 1) * P * n * 4, "dc_pa: records do not cover step ", ti);
    }
    for (const at::Tensor* t : {&grad, &snap, &dva, &beta, &kpay, &rec, &out_eff, &out_g})
        TORCH_CHECK(t->device() == bank.device(), "dc_pa: all operands on the bank's device");
    const int64_t plane = out_eff.numel() / (nsl == 1 ? 1 : nsl);
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool bf = bank.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bf || bank.scalar_type() == at::kFloat, "dc_pa: bank must be bf16 or f32");
    const size_t head = static_cast<size_t>(C) * 32 * sizeof(float4) + ((C + 3) & ~3) * sizeof(float);
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                          bank.device().index()));
#define DC_PL3(EG, EB, DD)                                                                    \
    {                                                                                         \
        auto k3 = wide ? &dc_pa3_kernel<EG, EB, DD, false, false, false, true>                \
                       : &dc_pa3_kernel<EG, EB, DD, false, false, false, false>;              \
        const size_t smem3 = head + static_cast<size_t>(nw) * (DD) * 32 *                     \
                                        (sizeof(RowOf<EG>::T) + sizeof(RowOf<EB>::T));        \
        TORCH_CHECK(smem3 <= static_cast<size_t>(optin), "dc_pa: ", smem3,                    \
                    " bytes of shared memory exceed the device opt-in limit ", optin);        \
        set_smem(k3, smem3);                                                                  \
        if (sdm_plancap::take(k3, dim3(blocks), dim3(threads), smem3,   /* plan capture */  \
            reinterpret_cast<EG*>(grad.data_ptr()), reinterpret_cast<EB*>(bank.data_ptr()),   \
            reinterpret_cast<const EB*>(snap.data_ptr()), snap_step_rows,                     \
            dva.data_ptr<float>(), beta.data_ptr<float>(), kpay.data_ptr<int32_t>(),          \
            reinterpret_cast<const int4*>(rec.data_ptr<int32_t>()),                           \
            out_eff.data_ptr<float>(), out_g.data_ptr<float>(), plane,                        \
            static_cast<int>(ti), static_cast<int>(P), static_cast<int>(C),                   \
            static_cast<int>(W), n, hd, nsl, slw, ctas, static_cast<int>(L),                  \
            static_cast<int>(dbg & (256 | 1024 | 8192 | 16384 | 32768 | 65536)),              \
            (dense || gdc) ? bpre.data_ptr<int32_t>() : nullptr,                              \
            gdc ? reinterpret_cast<const EG*>(gdwc.data_ptr()) : nullptr)) {                  \
            return;                                                                           \
        }                                                                                     \
        k3<<<blocks, threads, smem3, stream>>>(                                               \
            reinterpret_cast<EG*>(grad.data_ptr()), reinterpret_cast<EB*>(bank.data_ptr()),   \
            reinterpret_cast<const EB*>(snap.data_ptr()), snap_step_rows,                     \
            dva.data_ptr<float>(), beta.data_ptr<float>(), kpay.data_ptr<int32_t>(),          \
            reinterpret_cast<const int4*>(rec.data_ptr<int32_t>()),                           \
            out_eff.data_ptr<float>(), out_g.data_ptr<float>(), plane,                        \
            static_cast<int>(ti), static_cast<int>(P), static_cast<int>(C),                   \
            static_cast<int>(W), n, hd, nsl, slw, ctas, static_cast<int>(L),                  \
            static_cast<int>(dbg & (256 | 1024 | 8192 | 16384 | 32768 | 65536)),              \
            (dense || gdc) ? bpre.data_ptr<int32_t>() : nullptr,                              \
            gdc ? reinterpret_cast<const EG*>(gdwc.data_ptr()) : nullptr);                    \
    }
    if (g32) {
        DC_PL3(float, __nv_bfloat16, 8)
    } else if (bf) {
        DC_PL3(__nv_bfloat16, __nv_bfloat16, 8)
    } else {
        DC_PL3(float, float, 8)
    }
#undef DC_PL3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the backward's C++ step loop (bwd_steps_cuda): dc_pa (Phase A, dc_pa3) as a per-backward plan --------------------
// dc_pa_plan runs dc_pa ONCE under plan capture at step ti_max: every check runs (its step bounds grow with ti, so they hold
// for every ti <= ti_max) and its one launch is recorded. dc_pa_plan_launch (a plain C++ entry point; address from
// dc_pa_launch_addr) sets the step parameter ti and issues the recorded launch plain, as dc_pa launches (no PDL variant;
// its pdl argument must be 0). The plan holds raw pointers: the caller keeps the operands alive while it is used.
namespace {
constexpr uint64_t kPaPlanMagic = 0x314e414c50334150ull;
constexpr int kPaTiParam = 11;      // (grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, plane, ti, ...)
struct PaPlan {
    uint64_t magic;
    int32_t device;
    int32_t pad;
    int64_t ti_max;
    sdm_plancap::Launch cap;
};

int64_t dc_pa_plan_launch(const void* plan_bytes, int64_t nbytes, int64_t ti, int64_t pdl, void* stream) {
    PaPlan pl;
    if (!sdm_plancap::from_bytes(pl, plan_bytes, nbytes, kPaPlanMagic)) return -1;
    if (pdl != 0) return -4;
    if (ti < 0 || ti > pl.ti_max) return -2;
    if (pl.cap.set == 0) return 0;                 // dc_pa launched nothing
    if (sdm_plancap::current_device() != pl.device) return -5;
    int* tip = sdm_plancap::param<int>(pl.cap, kPaTiParam);
    if (tip == nullptr) return -7;
    *tip = static_cast<int>(ti);
    return sdm_plancap::launch(pl.cap, false, static_cast<cudaStream_t>(stream));
}
}  // namespace

at::Tensor dc_pa_plan(at::Tensor grad, at::Tensor bank, at::Tensor snap, int64_t snap_step_rows,
                      at::Tensor dva, at::Tensor beta, at::Tensor kpay, at::Tensor rec, at::Tensor out_eff,
                      at::Tensor out_g, int64_t ti_max, int64_t P, int64_t C, int64_t W, int64_t L, int64_t nw,
                      int64_t depth, int64_t dbg, bool pipe, at::Tensor bpre, at::Tensor gdwc) {
    TORCH_CHECK(ti_max >= 0 && ti_max < (int64_t(1) << 31), "dc_pa_plan: ti_max");
    PaPlan pl;
    std::memset(&pl, 0, sizeof(pl));
    {
        sdm_plancap::Scope scope(&pl.cap);
        dc_pa(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti_max, P, C, W, L, nw, depth,
              dbg, pipe, bpre, gdwc);
    }
    pl.magic = kPaPlanMagic;
    pl.device = static_cast<int32_t>(bank.get_device());
    pl.ti_max = ti_max;
    TORCH_CHECK(pl.cap.set == 0 || (pl.cap.nparams == 24 && sdm_plancap::param<int>(pl.cap, kPaTiParam) != nullptr),
                "dc_pa_plan: parameters");
    return sdm_plancap::to_tensor(pl);
}

int64_t dc_pa_launch_addr() { return reinterpret_cast<int64_t>(&dc_pa_plan_launch); }

void dc_pa_reduce(at::Tensor part_eff, at::Tensor part_g, at::Tensor part_pd, at::Tensor beta,
                  at::Tensor cpe, at::Tensor d_k_eff, at::Tensor grad_g, at::Tensor d_k_pd,
                  int64_t P, int64_t C, int64_t W, int64_t T) {
    const at::cuda::CUDAGuard guard(d_k_eff.device());
    const int64_t total = d_k_eff.numel();
    if (total == 0) return;
    const int ne = static_cast<int>(part_eff.numel() / total);
    const int np = static_cast<int>(part_pd.numel() / total);
    TORCH_CHECK(part_eff.numel() == ne * total && part_g.numel() == part_eff.numel() &&
                part_pd.numel() == np * total && grad_g.numel() == total && d_k_pd.numel() == total &&
                beta.numel() * W >= total && cpe.numel() >= total && P >= 1 && C >= 1 && W >= 1 &&
                total % (C * W) == 0, "dc_pa_reduce: operand sizes / geometry");
    for (const at::Tensor* t : {&part_eff, &part_g, &part_pd, &beta, &d_k_eff, &grad_g, &d_k_pd})
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous() && t->device() == d_k_eff.device(),
                    "dc_pa_reduce: f32 contiguous operands on one device");
    TORCH_CHECK((cpe.scalar_type() == at::kFloat || cpe.scalar_type() == at::kBFloat16) && cpe.is_contiguous() &&
                cpe.device() == d_k_eff.device(), "dc_pa_reduce: cpe");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>((total + 255) / 256);
#define DC_RED(VAL)                                                                           \
    dc_pa_reduce_kernel<VAL><<<blocks, 256, 0, stream>>>(                                     \
        part_eff.data_ptr<float>(), part_g.data_ptr<float>(), part_pd.data_ptr<float>(),      \
        beta.data_ptr<float>(), reinterpret_cast<const VAL*>(cpe.data_ptr()),                 \
        d_k_eff.data_ptr<float>(), grad_g.data_ptr<float>(), d_k_pd.data_ptr<float>(), total, \
        ne, ne, np, static_cast<int>(P), static_cast<int>(C), static_cast<int>(W), T)
    if (cpe.scalar_type() == at::kBFloat16) { DC_RED(__nv_bfloat16); } else { DC_RED(float); }
#undef DC_RED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dc_gwr(at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor out, at::Tensor dv,
            at::Tensor pd, int64_t row0, int64_t out_row0, int64_t tokens, int64_t chunk,
            int64_t valid, int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc,
            at::Tensor kpos, at::Tensor gdwc, int64_t pdl) {
    const at::cuda::CUDAGuard guard(bank.device());
    TORCH_CHECK(pdl == 0 || pdl == 1, "dc_gwr: pdl must be 0 or 1");
    const int hd = static_cast<int>(bank.size(1));
    const int es = static_cast<int>(bank.element_size());
    TORCH_CHECK(out.scalar_type() == at::kFloat && dv.scalar_type() == at::kFloat &&
                pd.scalar_type() == at::kFloat, "dc_gwr: out, delta_v, d_k_pd f32");
    TORCH_CHECK(idx.dim() == 2 && w.sizes() == idx.sizes(), "dc_gwr: idx/w [rows, width]");
    TORCH_CHECK(idx.scalar_type() == at::kInt || idx.scalar_type() == at::kLong, "dc_gwr: idx int32/int64");
    TORCH_CHECK(w.scalar_type() == at::kFloat || w.scalar_type() == at::kBFloat16, "dc_gwr: weights f32/bf16");
    TORCH_CHECK(out.scalar_type() == at::kFloat && dv.scalar_type() == at::kFloat && pd.scalar_type() == at::kFloat,
                "dc_gwr: f32 outputs and delta_v");
    TORCH_CHECK(chunk >= 1 && tokens >= 0 && tokens % chunk == 0 && valid >= 0 && valid <= chunk,
                "dc_gwr: need tokens = k * chunk and 0 <= valid <= chunk");
    {
        const auto addr = [](const at::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()); };
        TORCH_CHECK(vec_bytes != 16 || (addr(bank) % 16 == 0 && addr(out) % 16 == 0 && addr(dv) % 4 == 0),
                    "dc_gwr: 16-byte vectors need 16-byte aligned bank and output");
        TORCH_CHECK(!(addr(out) < addr(bank) + bank.nbytes() && addr(bank) < addr(out) + out.nbytes()) &&
                    !(addr(pd) < addr(bank) + bank.nbytes() && addr(bank) < addr(pd) + pd.nbytes()),
                    "dc_gwr: the outputs must not overlap the bank");
    }
    TORCH_CHECK(vec_bytes == 16 || vec_bytes == es, "dc_gwr: vec_bytes must be 16 or the element size");
    TORCH_CHECK(vec_bytes != 16 || (hd * es) % 16 == 0, "dc_gwr: 16-byte vectors need hd*es % 16 == 0");
    TORCH_CHECK(lanes == 8 || lanes == 16 || lanes == 32, "dc_gwr: lanes must be 8, 16 or 32");
    TORCH_CHECK(vec_bytes == 16 || lanes == 32, "dc_gwr: element vectors use 32 lanes");
    TORCH_CHECK(groups >= 1 && tpc >= 1 && lanes * groups * tpc <= 512, "dc_gwr: block too large");
    TORCH_CHECK(bank.is_contiguous() && idx.is_contiguous() && w.is_contiguous() && out.is_contiguous() &&
                dv.is_contiguous() && pd.is_contiguous(), "dc_gwr: contiguous operands");
    TORCH_CHECK(row0 >= 0 && row0 + tokens <= idx.size(0) && out_row0 >= 0 &&
                (out_row0 + tokens) * bank.size(1) <= out.numel() &&
                (row0 + tokens) * bank.size(1) <= dv.numel(), "dc_gwr: rows out of range");
    for (const at::Tensor* t : {&idx, &w, &out, &dv, &pd})
        TORCH_CHECK(t->device() == bank.device(), "dc_gwr: all operands on the bank's device");
    if (tokens == 0) return;
    GArgs a;
    a.bank = bank.data_ptr(); a.idx = idx.data_ptr(); a.w = w.data_ptr();
    a.out = out.data_ptr<float>(); a.dv = dv.data_ptr<float>(); a.pd = pd.data_ptr<float>();
    a.row0 = row0; a.out_row0 = out_row0;
    a.hd = hd; a.vpr = static_cast<int>(hd * es / vec_bytes);
    a.tokens = static_cast<int>(tokens); a.chunk = static_cast<int>(chunk);
    a.valid = static_cast<int>(valid);
    a.width = static_cast<int>(idx.size(1));
    a.per_group = static_cast<int>((a.width + groups - 1) / groups);
    a.idx64 = idx.scalar_type() == at::kLong;
    a.w_bf16 = w.scalar_type() == at::kBFloat16;
    const int slices = static_cast<int>((a.vpr + lanes - 1) / lanes);
    a.plane = idx.numel();
    TORCH_CHECK(pd.numel() == a.plane * (slices == 1 ? 1 : slices), "dc_gwr: d_k_pd planes");
    a.kpos = nullptr; a.gdwc = nullptr; a.crow = 0;
    if (kpos.numel() > 0) {
        const int64_t crow = chunk * a.width;
        TORCH_CHECK(kpos.scalar_type() == at::kInt && kpos.is_contiguous() && kpos.numel() == idx.numel() &&
                    kpos.device() == bank.device(), "dc_gwr: kpos int32 like idx");
        TORCH_CHECK(gdwc.scalar_type() == bank.scalar_type() && gdwc.is_contiguous() &&
                    gdwc.device() == bank.device() && row0 % chunk == 0 &&
                    gdwc.numel() >= (tokens / chunk) * crow * hd &&
                    (reinterpret_cast<uintptr_t>(gdwc.data_ptr()) % 16) == 0,
                    "dc_gwr: gdwc [tokens / chunk * chunk * width, hd] in the grad dtype, 16-byte aligned");
        {
            const auto addr = [](const at::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()); };
            TORCH_CHECK(!(addr(gdwc) < addr(bank) + bank.nbytes() && addr(bank) < addr(gdwc) + gdwc.nbytes()),
                        "dc_gwr: gdwc must not overlap the gathered rows");
        }
        a.kpos = kpos.data_ptr<int>(); a.gdwc = gdwc.data_ptr(); a.crow = static_cast<int>(crow);
    }
    const dim3 grid(static_cast<unsigned>((tokens + tpc - 1) / tpc), static_cast<unsigned>(slices));
    const dim3 block(static_cast<unsigned>(lanes), static_cast<unsigned>(groups),
                     static_cast<unsigned>(tpc));
    const size_t smem = (groups > 1 ? static_cast<size_t>(tpc) * groups * lanes *
                                          (vec_bytes / es) * sizeof(float) : 0) +
                        static_cast<size_t>(tpc) * groups * (lanes * (lanes + 4) + lanes) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool bf = bank.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bf || bank.scalar_type() == at::kFloat, "dc_gwr: bank bf16 or f32");
    // pdl = 1: programmatic stream serialization (the caller asserts the launch-order rule: the launch right before is
    // the next step's GDS, which wrote none of the pre-wait inputs).
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
#define DC_G(E, VB, LPR)                                                                      \
    { auto kf = dc_gwr_kernel<E, VB, LPR>; set_smem(kf, smem);                                \
      if (sdm_plancap::take(kf, grid, block, smem, a)) return;   /* plan capture */     \
      if (pdl == 1) { C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kf, a)); }                      \
      else { kf<<<grid, block, smem, stream>>>(a); } }
    // Elements are raw bits, as in token_gather: uint16_t = bfloat16, uint32_t = float.
    // 256-thread CTAs of the bf16 / 16-byte / 32-lane instance launch the 64-register entry point (same body).
    if (bf && vec_bytes == 16 && lanes == 32 && lanes * groups * tpc == 256) {
        { auto kf = dc_gwr_r64_kernel<uint16_t, 16, 32>; set_smem(kf, smem);
          if (sdm_plancap::take(kf, grid, block, smem, a)) return;   // plan capture
          if (pdl == 1) { C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kf, a)); }
          else { kf<<<grid, block, smem, stream>>>(a); } }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    if (vec_bytes != 16) {
        TORCH_CHECK(lanes == 32, "dc_gwr: element vectors use 32 lanes");
        if (bf) DC_G(uint16_t, 2, 32) else DC_G(uint32_t, 4, 32)
    } else if (bf) {
        if (lanes == 8) DC_G(uint16_t, 16, 8)
        else if (lanes == 16) DC_G(uint16_t, 16, 16)
        else DC_G(uint16_t, 16, 32)
    } else {
        if (lanes == 8) DC_G(uint32_t, 16, 8)
        else if (lanes == 16) DC_G(uint32_t, 16, 16)
        else DC_G(uint32_t, 16, 32)
    }
#undef DC_G
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the backward's C++ step loop (bwd_steps_cuda): dc_gwr as a per-backward plan -------------------------------------
// dc_gwr_plan runs dc_gwr ONCE under plan capture at the loop's largest step ti_max (row0 = ti_max * row_step, out_row0 =
// ti_max * out_row_step, valid = min(chunk, valid_tokens - ti_max * chunk)): every check of dc_gwr runs (its row bounds
// grow with the step, so they hold for every ti <= ti_max), and its one launch is recorded. dc_gwr_plan_launch (a plain
// C++ entry point; address from dc_gwr_launch_addr) sets row0 / out_row0 / valid of step ti in the recorded GArgs and
// issues the recorded launch: plain (pdl 0) or with programmatic stream serialization (pdl 1), as dc_gwr does for the same
// pdl. The plan holds raw pointers: the caller keeps the operands alive and unresized while it is used.
namespace {
constexpr uint64_t kGwrPlanMagic = 0x314e414c50525747ull;
struct GwrPlan {
    uint64_t magic;
    int32_t device;
    int32_t pad;
    int64_t ti_max, row_step, out_row_step, chunk, valid_tokens;
    sdm_plancap::Launch cap;
};

int64_t dc_gwr_plan_launch(const void* plan_bytes, int64_t nbytes, int64_t ti, int64_t pdl, void* stream) {
    GwrPlan pl;
    if (!sdm_plancap::from_bytes(pl, plan_bytes, nbytes, kGwrPlanMagic)) return -1;
    if (pdl != 0 && pdl != 1) return -4;
    if (ti < 0 || ti > pl.ti_max) return -2;
    if (pl.cap.set == 0) return 0;                 // dc_gwr launched nothing (tokens == 0)
    if (sdm_plancap::current_device() != pl.device) return -5;
    GArgs* a = sdm_plancap::param<GArgs>(pl.cap, 0);
    if (a == nullptr) return -7;
    a->row0 = ti * pl.row_step;
    a->out_row0 = ti * pl.out_row_step;
    a->valid = static_cast<int>(std::min(pl.chunk, pl.valid_tokens - ti * pl.chunk));
    return sdm_plancap::launch(pl.cap, pdl == 1, static_cast<cudaStream_t>(stream));
}
}  // namespace

at::Tensor dc_gwr_plan(at::Tensor bank, at::Tensor idx, at::Tensor w, at::Tensor out, at::Tensor dv, at::Tensor pd,
                       int64_t row_step, int64_t out_row_step, int64_t tokens, int64_t chunk, int64_t valid_tokens,
                       int64_t ti_max, int64_t vec_bytes, int64_t lanes, int64_t groups, int64_t tpc,
                       at::Tensor kpos, at::Tensor gdwc) {
    TORCH_CHECK(ti_max >= 0 && ti_max < (int64_t(1) << 31) && row_step >= 0 && out_row_step >= 0 && chunk >= 1 &&
                valid_tokens > ti_max * chunk && valid_tokens <= (ti_max + 1) * chunk,
                "dc_gwr_plan: steps 0..ti_max must each hold 1..chunk valid tokens");
    TORCH_CHECK(kpos.numel() == 0 || row_step % chunk == 0, "dc_gwr_plan: kpos needs chunk-aligned row steps");
    GwrPlan pl;
    std::memset(&pl, 0, sizeof(pl));
    {
        sdm_plancap::Scope scope(&pl.cap);
        dc_gwr(bank, idx, w, out, dv, pd, ti_max * row_step, ti_max * out_row_step, tokens, chunk,
               std::min(chunk, valid_tokens - ti_max * chunk), vec_bytes, lanes, groups, tpc, kpos, gdwc, 0);
    }
    pl.magic = kGwrPlanMagic;
    pl.device = static_cast<int32_t>(bank.get_device());
    pl.ti_max = ti_max;
    pl.row_step = row_step;
    pl.out_row_step = out_row_step;
    pl.chunk = chunk;
    pl.valid_tokens = valid_tokens;
    TORCH_CHECK(pl.cap.set == 0 || (pl.cap.nparams == 1 && sdm_plancap::param<GArgs>(pl.cap, 0) != nullptr),
                "dc_gwr_plan: GArgs");
    return sdm_plancap::to_tensor(pl);
}

int64_t dc_gwr_launch_addr() { return reinterpret_cast<int64_t>(&dc_gwr_plan_launch); }
"""


def gwr_tail_supported(bank, chunk_size, quant="none") -> bool:
    """Tail hook (movers.tail_admits): the backward's GWR (dc_gwr) runs the call's tail class on every backend (rtx: chunks
    of <= 128 tokens): it reads grad rows with memory_ops.pinned_gather_geometry (any head width), never snapshot rows."""
    from ._device import backend
    if not bank.is_cuda or bank.dim() != 2 or bank.dtype not in (torch.bfloat16, torch.float32):
        return False
    try:
        b = backend(bank.device)
    except RuntimeError:
        return False
    return ((b in ("modern", "a100") or (b == "rtx" and chunk_size <= 128)) and 1 <= chunk_size <= 256
            and 1 <= bank.shape[-1] <= 2 ** 20 and bank.shape[0] < 2 ** 31)


@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_pa_dc",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("dc_records_pa", "dc_copy_raw", "dc_overlaps", "dc_pa", "dc_pa_reduce", "dc_gwr", "dc_gwr_plan",
                   "dc_gwr_launch_addr", "dc_pa_plan", "dc_pa_launch_addr"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo"),
    )


def gwr_plan(bank, idx, w, out, dv, pd, *, row_step, out_row_step, tokens, chunk, valid_tokens, ti_max, vec_bytes,
             lanes, groups, tpc, kpos, gdwc):
    """(plan, launch address) of dc_gwr for steps 0 .. ti_max of the backward's C++ step loop (bwd_steps_cuda)."""
    m = _module()
    plan = m.dc_gwr_plan(bank, idx, w, out, dv, pd, int(row_step), int(out_row_step), int(tokens), int(chunk),
                         int(valid_tokens), int(ti_max), int(vec_bytes), int(lanes), int(groups), int(tpc), kpos, gdwc)
    return plan, int(m.dc_gwr_launch_addr())


def pa_step_plan(grad, bank, snap, snap_step_rows, dva, beta, kpay, rec, out_eff, out_g, ti_max, P, C, W, L, nw, depth,
                 dbg, pipe, bpre, gdwc):
    """(plan, launch address) of dc_pa for steps 0 .. ti_max of the backward's C++ step loop (bwd_steps_cuda)."""
    m = _module()
    plan = m.dc_pa_plan(grad, bank, snap, int(snap_step_rows), dva, beta, kpay, rec, out_eff, out_g, int(ti_max), int(P),
                        int(C), int(W), int(L), int(nw), int(depth), int(dbg), bool(pipe), bpre, gdwc)
    return plan, int(m.dc_pa_launch_addr())


def slices(head_dim: int) -> int:
    return -(-head_dim // 128)


def tile_len(chunk_size: int, width: int, head_dim: int, num_parallel: int) -> int:
    """Sorted positions per warp tile from SHAPES only: 256, halved while the step would hold fewer
    than 4096 tiles (2048 when the head is split into column slices), floor 32. Results do not depend on it."""
    nsl = slices(head_dim)
    work = num_parallel * nsl * chunk_size * width
    target = 4096 if nsl == 1 else 2048
    tl = 256
    while tl > 32 and work // tl < target:
        tl //= 2
    return max(1, min(tl, chunk_size * width))


def build_records_pa(spay_k, k_idx, k_val, log_k, beta, cpe):
    """int32 [N, C * W, 4] records per sorted write position: pay, slot, sc bits, cpe bits."""
    n = spay_k.shape[-1]
    spay_k = spay_k.reshape(-1, n)
    rec = torch.empty(spay_k.shape[0], n, 4, device=spay_k.device, dtype=torch.int32)
    _module().dc_records_pa(spay_k, k_idx.contiguous(), k_val.contiguous(), log_k.contiguous(),
                            beta.contiguous().view(-1), cpe.contiguous(), rec)
    return rec
