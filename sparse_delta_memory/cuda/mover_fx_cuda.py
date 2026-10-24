# Copyright (c) Meta Platforms, Inc. and affiliates.
"""mover_fx_cuda: the deterministic forward's bank-direct DW kernels (Phase-2 decay and write, bf16 or f32 bank) and its
fused step loop.

Same arithmetic, bit for bit, as mover_da2_cuda's dense walk (da2_dw_kernel MODE 32, sf_full = 1), which was itself
bitwise 2.0.0's Triton DW (phase2_kernels):

  per group (sorted order): acc = 0; acc = fma(sc_m, dv[tok_m], acc) over the members in sorted order;
  bank[slot] = bf16_rn(fma(base, cpe_last, acc)) (a float32 bank: the fma result itself), base = the group's
  pre-write bank row;
  acc restarts after EVERY LAST (valid or not); only valid LASTs store.

float32 banks (fx_dw_kernel<float, ...>: one column slice per warp, 64-bit row offsets) compute mover_da2_cuda's float32
walk: the same per-group chain, base = the group's pre-step bank row, stored unrounded.

How it differs (speed only):
  * Base rows are read from bank[slot] itself (bank-direct): within one launch every group owns its slot, so no other
    emit writes it, and the read precedes the group's own store in the same thread.
  * Token rows: one 1D bulk copy (one slice) or one 2D tensor-map box {SW, C} (column slices) instead of C per-row
    copies.
  * The walk: per 32-position batch one STS.128 per lane of {token offset, scale, row byte offset, cpe}; per
    8-position sub-batch the records and token rows are loaded first, and a sub-batch without any LAST runs a
    branch-free FMA path; the emit is one code path with 32-bit row offsets (64-bit for banks over 4 GiB).
  * fx_dw2_kernel takes two column slices per warp (even slice counts), so records, masks and addressing serve 8
    columns instead of 4.
  * In the saving forward the kernel can also write the dense compact snapshot rows (raw copies of the base rows it
    reads), so the gather need not write them.
Tiles, records (krec), the tile group prefix (segb) and the dense row layout are mover_da2_cuda's / Phase 1's; every
group is summed in sorted order by one warp, so the output does not depend on the geometry.

fx_fwd_steps runs the deterministic forward's step loop (gather, dual matmul, DW of every step) in one host call."""

import functools

import torch

from ._build import load_cuda_extension
from .gx_gds_cuda import QUANT_HELPERS, ROW_HELPERS
from ._device import RESERVED_SMEM, headers, num_sms, prev_pow2
from ._devmemo import (backend as _device_backend, pdl_ok as _device_pdl_ok, sm90 as _sm90, smem_cap as _device_smem_cap,
                       smem_per_sm as _device_smem_per_sm)

_CPP = r"""
at::Tensor fx_maps(at::Tensor snapc, at::Tensor dv, int64_t C, int64_t wide_rows);
at::Tensor fx_maps2(at::Tensor snapc, at::Tensor dv, int64_t C);
at::Tensor fx_dw_plan(at::Tensor bank, at::Tensor dv, int64_t dv_step_rows, at::Tensor krec, at::Tensor tb,
                      at::Tensor segb, at::Tensor snapc, int64_t sb_step, int64_t P, int64_t C, int64_t K, int64_t nw,
                      int64_t nu, at::Tensor maps, int64_t bankb, int64_t quant, at::Tensor payload, at::Tensor scale,
                      int64_t portable, int64_t smem_cap);
void fx_dw_step(at::Tensor plan, int64_t ti, int64_t pdl);
int64_t fx_q_capacity(int64_t quant, int64_t w2, int64_t bf, int64_t nw, int64_t t2d, int64_t allon, int64_t cluster,
                      int64_t smem);
int64_t fx_tri_desc_size();
void fx_fwd_steps(at::Tensor gplan, int64_t gather_addr, at::Tensor tdesc, at::Tensor dwplan, int64_t t0, int64_t t1,
                  int64_t SC, int64_t C, int64_t tokens, int64_t pdl_gather, int64_t pdl_dw);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include "sdm_copy.cuh"

namespace {

constexpr uint32_t kFirst = 0x80000000u;
constexpr uint32_t kLast = 0x40000000u;
constexpr uint32_t kInvalid = 0x20000u;
constexpr unsigned kFull = 0xFFFFFFFFu;
constexpr int kUnitRows = 8;      // dense rows per unit (one TMA, one wait)
constexpr int kTileRows = 128;    // valid groups per warp tile, every fx tile set (FIRST-snapped, <= 128 positions apart)
constexpr int64_t kMaxHeadDim = int64_t(1) << 20;   // host bound: row bytes, TMA coordinates and slice offsets stay
                                                   // 32-bit safe (hd * 4 < 2^23); Python supported() gates it

// The portable side (SDM_PORTABLE_ARCH, sdm_copy.cuh: sm_80 / 86 / 89 / 120 ...): no barrier object (FX_SM90_OR_NONE:
// nothing), the token rows staged by every thread with sdmcp's cp.async copies and published by a CTA barrier, no
// griddepcontrol, no clusters; the bulk / tensor copies and transaction waits (FX_SM90) trap there (never reached).
#ifdef SDM_PORTABLE_ARCH
#define FX_PRE_SM90 1
#define FX_SM90(...) __trap()
#define FX_SM90_OR_NONE(...)
#else
#define FX_SM90(...) __VA_ARGS__
#define FX_SM90_OR_NONE(...) __VA_ARGS__
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    FX_SM90_OR_NONE(asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory"));
}
__device__ __forceinline__ void fence_mbar_init() {
    FX_SM90_OR_NONE(asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"));
}
__device__ __forceinline__ void fence_proxy_async() {
    FX_SM90_OR_NONE(asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"));
}
// Invalidates an mbarrier before its memory is initialized again (the all-slices walk re-runs the prologue per slice).
__device__ __forceinline__ void mbar_inval(uint64_t* bar) {
    FX_SM90_OR_NONE(asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory"));
}
// One arrival that also announces `bytes` of transactions (count-1 barriers).
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* bar, uint32_t bytes) {
    FX_SM90(asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
                         "r"(bytes) : "memory"));
}
// 1D TMA bulk copy global -> this CTA's shared memory (16-byte aligned, multiple of 16 bytes).
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    FX_SM90(asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory"));
}
// 16-byte cp.async global -> this thread's shared memory (L2 only, .cg); completion: cp_async_wait_all (PF path).
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    FX_SM90(asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src) : "memory"));
}
// Waits for every cp.async this thread issued (commit + wait_group 0); the caller's __syncwarp then makes the copied
// bytes visible to the other lanes.
__device__ __forceinline__ void cp_async_wait_all() {
    FX_SM90(asm volatile("cp.async.wait_all;\n" ::: "memory"));
}
// 2D tensor-map TMA box -> shared memory (128-byte aligned destination); c0 = column, c1 = row.
__device__ __forceinline__ void tma2d_g2s(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    FX_SM90(asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
        : "memory"));
}
// Spins until phase `parity` of `bar` completed; returns 0. No memory clobber: the caller adds the result to the
// address of every read that must follow the wait (acquire semantics of try_wait), as in mover_da2_cuda.
__device__ __forceinline__ uint32_t mbar_wait_dep(uint64_t* bar, uint32_t parity) {
#ifdef FX_PRE_SM90
    __trap();
    return 0u;
#else
    uint32_t ok;
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " FX_WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " @!p bra FX_WAIT;\n"
        " mov.u32 %0, 0;\n"
        "}\n"
        : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity));
    return ok;
#endif
}

// Thread-block cluster (sm_90+) of a quantized multi-slice walk: the CTAs of every column slice of the same tiles
// (cluster rank = slice) exchange their per-row partial amax through distributed shared memory.
__device__ __forceinline__ void cluster_arrive() {   // release
    FX_SM90(asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory"));
}
__device__ __forceinline__ void cluster_wait() {     // acquire
    FX_SM90(asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"));
}
// The float at shared address p of CTA `rank` of this cluster.
__device__ __forceinline__ float ld_peer(const float* p, uint32_t rank) {
#ifdef FX_PRE_SM90
    __trap();
    return 0.f;
#else
    uint32_t a;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(a) : "r"(smem_u32(p)), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(a) : "memory");
    return v;
#endif
}

@ROW_HELPERS@
__device__ __forceinline__ uint4 ld_rec(const uint4* __restrict__ kr, int pos, int e) {
    if (pos < e) return __ldg(kr + pos);
    return make_uint4(kFirst | kLast | kInvalid, 0u, 0u, 0x3F800000u);   // padding: own (invalid) group
}
// The bank row of a record-batch entry. Banks <= 4 GiB keep the 32-bit BYTE offset in the batch; BIG banks (> 4 GiB,
// e.g. 4 heads x head dim 960 x 921,600 slots) keep the row INDEX and widen it to a 64-bit byte offset here.
template <bool BIG>
__device__ __forceinline__ unsigned char* fx_row(unsigned char* lane_base, uint32_t rec, uint32_t row_bytes) {
    if constexpr (BIG) {
        return lane_base + static_cast<uint64_t>(rec) * row_bytes;
    } else {
        return lane_base + rec;
    }
}

__device__ __forceinline__ void fma4(float4& acc, uint32_t scb, const float4& x) {
    const float sc = __uint_as_float(scb);
    acc.x = fmaf(sc, x.x, acc.x);
    acc.y = fmaf(sc, x.y, acc.y);
    acc.z = fmaf(sc, x.z, acc.z);
    acc.w = fmaf(sc, x.w, acc.w);
}
// Tails, SCALAR (hd % 4 != 0, variant bit 32): bank and compact rows are only element-aligned, so a lane touches its
// first nv (<= 4) columns element by element: the same values and roundings as the 4-wide accesses.
template <typename E>
__device__ __forceinline__ typename Row4<E>::T ld_row_n(const unsigned char* p, int nv) {
    if constexpr (std::is_same<E, __nv_bfloat16>::value) {
        const unsigned short* q = reinterpret_cast<const unsigned short*>(p);
        const uint32_t a = nv > 0 ? __ldcg(q) : 0u, b = nv > 1 ? __ldcg(q + 1) : 0u;
        const uint32_t c = nv > 2 ? __ldcg(q + 2) : 0u, d = nv > 3 ? __ldcg(q + 3) : 0u;
        return make_uint2(a | (b << 16), c | (d << 16));
    } else {
        const float* q = reinterpret_cast<const float*>(p);
        return make_float4(nv > 0 ? __ldcg(q) : 0.f, nv > 1 ? __ldcg(q + 1) : 0.f, nv > 2 ? __ldcg(q + 2) : 0.f,
                           nv > 3 ? __ldcg(q + 3) : 0.f);
    }
}
__device__ __forceinline__ void st4_n(__nv_bfloat16* p, float4 v, int nv) {
    const uint32_t lo = pack_bf2(v.x, v.y), hi = pack_bf2(v.z, v.w);
    unsigned short* q = reinterpret_cast<unsigned short*>(p);
    if (nv > 0) q[0] = static_cast<unsigned short>(lo);
    if (nv > 1) q[1] = static_cast<unsigned short>(lo >> 16);
    if (nv > 2) q[2] = static_cast<unsigned short>(hi);
    if (nv > 3) q[3] = static_cast<unsigned short>(hi >> 16);
}
__device__ __forceinline__ void st4_n(float* p, float4 v, int nv) {
    if (nv > 0) p[0] = v.x;
    if (nv > 1) p[1] = v.y;
    if (nv > 2) p[2] = v.z;
    if (nv > 3) p[3] = v.w;
}
__device__ __forceinline__ void st_raw_n(__nv_bfloat16* p, uint2 v, int nv) {
    unsigned short* q = reinterpret_cast<unsigned short*>(p);
    if (nv > 0) q[0] = static_cast<unsigned short>(v.x);
    if (nv > 1) q[1] = static_cast<unsigned short>(v.x >> 16);
    if (nv > 2) q[2] = static_cast<unsigned short>(v.y);
    if (nv > 3) q[3] = static_cast<unsigned short>(v.y >> 16);
}
__device__ __forceinline__ void st_raw_n(float* p, float4 v, int nv) { st4_n(p, v, nv); }

@QUANT_HELPERS@
// Quantized walks (QUANT 1 fp8 / 2 int4), before the walk: the valid LASTs of the warp's tile [s, e) in order, kQB at a
// time: f(sl, n, k0) with sl[j] (j < n) the slot of the tile's valid group k0 + j (sl[j >= n]: a repeat of sl[n - 1]).
// Each record batch's valid slots are compacted through scr (the warp's 32 record words, free before the walk), so every
// call carries kQB rows whatever the share of valid positions. Returns the tile's valid groups.
constexpr int kQB = 8;
template <typename F>
__device__ __forceinline__ int q_batches(const uint4* __restrict__ kr, int s, int e, int lane, uint32_t* scr, F f) {
    int k = 0;
    for (int pos = s; pos < e; pos += 32) {
        const uint4 r = ld_rec(kr, pos + lane, e);
        const bool v = (r.x & (kLast | kInvalid)) == kLast;
        const uint32_t ml = __ballot_sync(kFull, v);
        const int nv = __popc(ml);
        if (v) scr[__popc(ml & ((1u << lane) - 1u))] = r.y;
        __syncwarp();
#pragma unroll 1
        for (int b0 = 0; b0 < nv; b0 += kQB) {
            const int n = min(kQB, nv - b0);
            uint32_t sl[kQB];
#pragma unroll
            for (int j = 0; j < kQB; ++j) sl[j] = scr[b0 + min(j, n - 1)];
            f(sl, n, k);
            k += n;
        }
        __syncwarp();   // this batch's reads precede the next batch's writes
    }
    return k;
}

struct FxGeo {
    int P, C, n, hd, NSL, SW, K, TU;
    int tok_bytes;        // shared bytes of the token rows (128-aligned)
    uint32_t row_bytes;   // hd * sizeof(element): bytes of one bank row
};

// A quantized walk's rows (QUANT_HELPERS), before the walk, over q_batches: load(sl, n, v) reads the lane's columns of
// the pre-write rows of slots sl[j < n] into v (a V); amax(v, a) gives each lane's |max| a[j] over them (0 for inactive
// columns) -> the warp's max in shared memory after tok_bar ({amax, scale} per valid group of the CTA's tiles, from the
// first tile's dense ordinal); CL (column slices, sm_90+: the CTAs of every slice of the same tiles form one cluster):
// barrier, row amax = max of the peers' partials (DSMEM; exact, any order); scale -> qw[k].y, the writer stores it at
// qsc_row0[k]; then store(v, n, k0, qw) writes the lane's quantized columns of the rows k0 + j (the first batch from
// registers, the others re-read). Bank reads, the quantized rows and their scales only (no other kernel of the step
// touches them). CL: the caller waits on the cluster before it exits.
template <int Q, int NW, bool CL, typename V, typename Ld, typename Am, typename St>
__device__ __forceinline__ void q_rows(unsigned char* smem, const FxGeo& g, const int32_t* __restrict__ segb, int64_t tix0,
                                       int g0, const uint4* __restrict__ kr, int s, int e, int lane, uint32_t* scr,
                                       bool writer, float* __restrict__ qsc_row0, Ld load, Am amax, St store) {
    const int ord0 = __ldg(segb + tix0);
    if (__ldg(segb + tix0 + NW) - ord0 > NW * kTileRows) __trap();   // the shared-memory bound of every fx tile set
    float2* const qw = reinterpret_cast<float2*>(smem + g.tok_bytes + NW * 32 * sizeof(uint4) + 16) + (g0 - ord0);
    V held;                                   // the tile's first batch (n_held rows), kept for the stores
    int n_held = 0;
    const int k = q_batches(kr, s, e, lane, scr, [&](const uint32_t* sl, int n, int k0) {
        V v;
        load(sl, n, v);
        float a[kQB];
        amax(v, a);
#pragma unroll
        for (int j = 0; j < kQB; ++j) {
            if (j < n) {
                const float m = q_warp_max(a[j]);
                if (lane == 0) qw[k0 + j].x = m;
            }
        }
        if (k0 == 0) {
            held = v;
            n_held = n;
        }
    });
    if constexpr (CL) {
        cluster_arrive();
        cluster_wait();
    } else {
        __syncwarp();
    }
    for (int i = lane; i < k; i += 32) {
        float m = qw[i].x;
        if constexpr (CL) {
            float pv[16];
#pragma unroll
            for (int r = 0; r < 16; ++r) pv[r] = r < g.NSL ? ld_peer(&qw[i].x, static_cast<uint32_t>(r)) : 0.f;
#pragma unroll
            for (int r = 0; r < 16; ++r) m = fmaxf(m, pv[r]);
        }
        qw[i].y = q_scale<Q>(m);
        if (writer) qsc_row0[i] = qw[i].y;
    }
    __syncwarp();
    if constexpr (CL) cluster_arrive();     // this CTA's reads of its peers' rows are done
    if (n_held > 0) store(held, n_held, 0, qw);
    if (k > n_held) {
        q_batches(kr, s, e, lane, scr, [&](const uint32_t* sl, int n, int k0) {
            if (k0 > 0) {
                V v;
                load(sl, n, v);
                store(v, n, k0, qw);
            }
        });
    }
}

// The all-slices quantized form (QUANT 5 fp8 / 6 int4, q = QUANT & 3; every backend: the portable side's only column-
// sliced quantized form (no clusters there), and the modern form for heads past the 16-CTA cluster): ONE CTA per tile set
// walks every slice of its tiles (grid P * K, the kernels' slice passes), so before any of its walks writes a bank row
// each warp quantizes its own tile's valid groups over ALL hd columns (q_batches order = the dense ordinals, as q_rows):
// per kQB-row batch the nsl slices of SW columns are streamed twice (pass 1: the lane's |max| over its 4 columns of every
// slice -> q_warp_max -> q_scale, lane 0 stores the scale once per row; pass 2: q_store4 per slice), O(1) registers per
// row, no shared memory beyond scr (the warp's record words, as q_batches), no barrier: the warp is the only writer of its
// tile's slots and walks them afterwards. Lanes past the slice width (4 * lane >= SW: the final tail) are off. The max is
// exact, so the scales and payload bytes are those of the cluster form (EXACT = !bf16, y = __frcp_rn(sc), as q_rows).
// Plain coherent loads (the rows the previous step's DW wrote). Bank reads, quantized rows and scales only.
template <int Q, bool EXACT, typename E>
__device__ __forceinline__ void q_rows_cta(const E* __restrict__ bank, const FxGeo& g, int nsl, const uint4* __restrict__ kr,
                                           int s, int e, int lane, uint32_t* scr, uint8_t* __restrict__ qpay,
                                           float* __restrict__ qsc) {
    using RowT = typename Row4<E>::T;
    const bool on = 4 * lane < g.SW;
    const int c4 = on ? 4 * lane : 0;
    const int64_t qrow = Q == 1 ? g.hd : g.hd / 2;
    q_batches(kr, s, e, lane, scr, [&](const uint32_t* sl, int n, int k0) {
        float a[kQB];
#pragma unroll
        for (int j = 0; j < kQB; ++j) a[j] = 0.f;
#pragma unroll 1
        for (int c = 0; c < nsl; ++c) {
            const int64_t col = static_cast<int64_t>(c) * g.SW + c4;
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                if (j < n && on) {
                    a[j] = fmaxf(a[j], q_amax4(cvt4(__ldcg(reinterpret_cast<const RowT*>(
                                                    bank + static_cast<int64_t>(sl[j]) * g.hd + col)))));
                }
            }
        }
        float sc[kQB];
#pragma unroll
        for (int j = 0; j < kQB; ++j) {
            sc[j] = 1.f;
            if (j < n) {                     // warp-uniform: every lane takes part in the warp max
                sc[j] = q_scale<Q>(q_warp_max(a[j]));
                if (lane == 0) qsc[k0 + j] = sc[j];
            }
        }
#pragma unroll 1
        for (int c = 0; c < nsl; ++c) {
            const int64_t col = static_cast<int64_t>(c) * g.SW + c4;
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                if (j < n && on) {
                    const RowT v = __ldcg(reinterpret_cast<const RowT*>(bank + static_cast<int64_t>(sl[j]) * g.hd + col));
                    q_store4<Q, EXACT>(qpay + static_cast<int64_t>(k0 + j) * qrow, col, cvt4(v), sc[j],
                                       EXACT ? 0.f : __frcp_rn(sc[j]));
                }
            }
        }
    });
}

// Grid: P * NSL * K CTAs (CTA b: cr = b % K, slice = b / K % NSL, p = rest); warp w walks tile cr * NW + w of its chunk.
// Shared memory: token rows [C][SW] f32 | rings [NW][NU][8 rows][SW] bf16 | record batch [NW][32] uint4 |
// mbarriers [1 + NW * NU].
// BANKB (bank-direct, the only mode the launchers instantiate): no rings, no segb; the base row of each valid LAST is
// read from bank[slot] itself (the pre-step row: within one launch every group owns its slot, so no other emit writes
// it; the read precedes the group's own store in the same thread). Same values as the dense snapshot rows (raw copies
// of those bank rows), hence the same arithmetic and bits. Shared memory: token rows | record batch | tok_bar.
// Without BANKB the base rows stream from the dense snapshot through the TMA rings; PF (BANKB only) cp.async's a
// batch's base rows into per-warp buffers [NW][32 rows][RB] before its walk. Neither is instantiated.
// WR (BANKB only): the DW also writes the saving forward's dense compact snapshot (layout 2): the base row of each valid
// group (its pre-write bank row, which BANKB reads anyway) goes to snapc row (ti * sb_step + p) * n + g, g = the group's
// dense ordinal in the chunk = segb[tile] + valid LASTs before it in the tile. Raw copies: the same bytes the
// gather would write there, so the backward is unchanged; the gather then writes no compact rows.
// E: the bank / snapshot element type (ROW_HELPERS: Row4, cvt4, st4, st_raw). bf16: one RN rounding per stored element;
// float: the fmaf result is stored as it is (float32 rows, 16 bytes per lane), always with 64-bit row offsets (BIG).
// QUANT (1 fp8, 2 int4; BANKB, !PF, !WR, BIG): the saving forward's compact rows, quantized (QUANT_HELPERS): the row of
// each valid group (WR's row index and base-row registers) goes to qpay ([rows, hd] fp8 bytes or [rows, hd / 2] int4)
// with its per-row scale in qsc[row]; qpay / qsc travel in the first two words of map_rows (a bank-direct walk never
// reads that map), so every instantiation keeps one parameter list. One slice (!T2D): the warp holds the whole row and
// quantizes it at its valid LAST (warp max). Column slices: the CTAs of the slices of the same tiles form one cluster
// (launched slice-fastest) and q_rows writes the rows before the PDL wait (overlapping the dual matmul).
// SE: the compact snapshot's element type: E, or bf16 rows of a float32 bank's WR walk (snapshot_quant bf16: each group's
// pre-write row rounded to bf16, RN, at its dense row; the bank update is the same walk and arithmetic as with SE = E).
template <typename E, int NW, int NU, bool T2D, bool BANKB, bool PF, bool WR, bool BIG, int QUANT = 0, typename SE = E>
__global__ void __launch_bounds__(NW * 32, (NW <= 8 ? 2 : 1)) fx_dw_kernel(
    E* __restrict__ bank, const float* __restrict__ dv, int64_t dv_step_rows,
    const uint4* __restrict__ krec, const int32_t* __restrict__ tb, const int32_t* __restrict__ segb,
    SE* __restrict__ snapc, int64_t sb_step, int ti, FxGeo g,
    const __grid_constant__ CUtensorMap map_rows, const __grid_constant__ CUtensorMap map_tok) {
    static_assert(std::is_same<SE, E>::value ||
                      (std::is_same<E, float>::value && std::is_same<SE, __nv_bfloat16>::value && BANKB && WR && !PF &&
                       (QUANT & 7) == 0),
                  "fx_dw_kernel: a compact snapshot type other than the bank's: bf16 rows of a float32 bank's WR walk");
    static_assert(BANKB || !PF, "fx_dw_kernel: the batch prefetch (PF) is a bank-direct (BANKB) mode");
    static_assert(BANKB || !WR, "fx_dw_kernel: the compact-snapshot write (WR) is a bank-direct (BANKB) mode");
    static_assert(!(PF && WR), "fx_dw_kernel: WR is not combined with the batch prefetch");
    static_assert((BANKB && !PF) || !BIG, "fx_dw_kernel: banks of > 4 GiB use the bank-direct walk without prefetch");
    constexpr bool kBf = std::is_same<E, __nv_bfloat16>::value;
    static_assert(kBf || (BANKB && !PF && BIG), "fx_dw_kernel: float32 banks: the bank-direct walk, 64-bit offsets");
#if defined(FX_PRE_SM90)
    static_assert(BANKB && !PF, "fx_dw_kernel: the portable side runs the bank-direct walk without prefetch only");
#endif
    using RowT = typename Row4<E>::T;             // a lane's 4 columns of a bank row
    // QUANT = variant bits: bits 0-2 the quant form (0 none; 1 / 2 the cluster form (column slices: the CTAs of every slice
    // of the same tiles, sm_90+) / in-walk quantization (one slice); 5 / 6 the all-slices form (q_rows_cta: one CTA per
    // tile set, grid P * K, the slices walked in turn); q = QF & 3), bit 4 UNEVEN (tails: 128-wide column slices, the
    // last one short)
    constexpr int QF = QUANT & 7;
    constexpr bool kQC = QF == 1 || QF == 2;       // the cluster form (column slices) / in-walk quantization (one slice)
    constexpr bool kAllSl = QF >= 5;               // the all-slices form
    constexpr bool kUneven = (QUANT & 16) != 0;
    constexpr bool kScalar = (QUANT & 32) != 0;   // tails: hd % 4 != 0 (element accesses, CTA-copied token rows)
    static_assert(QF == 0 || (BANKB && !PF && !WR && BIG), "fx_dw_kernel: quantized rows: bank-direct, 64-bit offsets");
    static_assert(!kAllSl || T2D, "fx_dw_kernel: the all-slices quantized form is for column-sliced heads");
    static_assert(!kUneven || (BANKB && !PF && T2D && QF == 0), "fx_dw_kernel: uneven slices: bank-direct, unquantized");
    static_assert(!kScalar || (BANKB && !PF && QF == 0 && !kUneven), "fx_dw_kernel: SCALAR: bank-direct, unquantized");
    extern __shared__ __align__(128) unsigned char smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int b = blockIdx.x;
    if constexpr (kQC && T2D) b = b / g.NSL % g.K + g.K * (b % g.NSL + g.NSL * (b / g.NSL / g.K));   // cluster order
    if constexpr (kAllSl) b = b / g.K * g.NSL * g.K + b % g.K;   // all slices: grid P * K, from slice 0
fx_slice:   // kAllSl: the next slice of this CTA's tiles restarts here (the end of the kernel)
    const int cr = b % g.K;
    b /= g.K;
    const int sl = b % g.NSL;
    const int p = b / g.NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * g.P + p;
    const int SW = g.SW, C = g.C;
    const int RB = SW * static_cast<int>(sizeof(E));   // bytes of a row slice (multiple of 16)
    const int UB = kUnitRows * RB;                 // bytes of a ring unit (multiple of 128)
    float* tok_s = reinterpret_cast<float*>(smem);
    constexpr int NUR = BANKB ? 0 : NU;           // ring units per warp held in shared memory
    const int PFB = PF ? 32 * RB : 0;              // PF: bytes of a warp's prefetch buffer (32 rows of RB)
    unsigned char* const rings = smem + g.tok_bytes;
    unsigned char* const ring = rings + static_cast<size_t>(warp) * NUR * UB;
    unsigned char* const pfb = rings + static_cast<size_t>(warp) * PFB;   // PF: row j = position pos0 + j of the batch
    uint4* const rec_s = reinterpret_cast<uint4*>(rings + static_cast<size_t>(NW) * NUR * UB +
                                                  static_cast<size_t>(NW) * PFB) + warp * 32;
    uint64_t* const bars = reinterpret_cast<uint64_t*>(rings + static_cast<size_t>(NW) * NUR * UB +
                                                       static_cast<size_t>(NW) * PFB +
                                                       static_cast<size_t>(NW) * 32 * sizeof(uint4));
    uint64_t* const tok_bar = bars;
    uint64_t* const ubar = bars + 1 + warp * NUR;
    if (threadIdx.x == 0) mbar_init(tok_bar, 1);
    if (lane < NUR) mbar_init(ubar + lane, 1);
    fence_mbar_init();
    __syncthreads();

    // Token rows of the slice, all on tok_bar: [C][SW] f32 (row stride SW floats).
    const int64_t tok_row0 = static_cast<int64_t>(ti) * dv_step_rows + static_cast<int64_t>(p) * C;
    // SCALAR: no TMA (the token rows' stride hd * 4 is not 16-byte aligned): the CTA copies [C][SW] with plain loads,
    // zero past the slice's columns (SW = 128, or hd rounded up to 4 for one slice), then a CTA barrier.
    auto tok_copy_scalar = [&]() {
        const int swe_t = min(SW, g.hd - sl * SW);
        const float* const src = dv + tok_row0 * g.hd + static_cast<int64_t>(sl) * SW;
        for (int i = threadIdx.x; i < C * SW; i += NW * 32) {
            const int r = i / SW, c = i - r * SW;
            tok_s[i] = c < swe_t ? __ldcg(src + static_cast<int64_t>(r) * g.hd + c) : 0.f;
        }
        __syncthreads();
    };
#if defined(FX_PRE_SM90)
    // Portable: every thread copies its share of the C token rows (the slice's SW floats of each delta_v row; one slice:
    // SW == hd; UNEVEN: the last slice's own columns, the walk reads no column past them) with cp.async and commits its
    // group; the token wait below publishes them (no exit precedes it). SCALAR (hd % 4 != 0: rows are not 16-byte
    // aligned): the CTA copies them with plain loads and a CTA barrier (tok_copy_scalar).
    if constexpr (kScalar) {
        tok_copy_scalar();
    } else {
        const int swt = kUneven ? min(SW, g.hd - sl * SW) : SW;
        sdmcp::copy_rows16(tok_s, SW * 4, dv + tok_row0 * g.hd + static_cast<int64_t>(sl) * SW, static_cast<int64_t>(g.hd) * 4,
                           C, swt * 4, static_cast<int>(threadIdx.x), NW * 32);
        sdmcp::commit();
    }
#endif

    const int64_t tix = chunk * (g.TU + 1) + cr * NW + warp;
    const int s = __ldg(tb + tix), e = __ldg(tb + tix + 1);
    const int g0 = (BANKB && !WR && QF == 0) ? 0 : __ldg(segb + tix);   // WR / QUANT: the tile's first dense ordinal
    const int nseg = BANKB ? 0 : __ldg(segb + tix + 1) - g0;   // valid groups of the tile = emits = dense rows (ring)
    const int nunits = (nseg + kUnitRows - 1) / kUnitRows;
    const int64_t drow0 = (static_cast<int64_t>(ti) * sb_step + p) * g.n + g0;   // dense row of the tile's group 0
    // Unit u = dense rows [drow0 + 8u, drow0 + 8u + 8) clipped to the tile, into ring slot u % NU, completing on
    // ubar[u % NU] (k-th use of a slot = phase k). Lane 0 only.
    const CUtensorMap* const mrows = &map_rows;    // generic address of the grid-constant parameter
    auto issue_unit = [&](int u) {
        uint64_t* bar = ubar + (u % NU);
        unsigned char* dst = ring + (u % NU) * UB;
        const int k = min(kUnitRows, nseg - u * kUnitRows);
        const int64_t row = drow0 + static_cast<int64_t>(u) * kUnitRows;
        mbar_arrive_tx(bar, static_cast<uint32_t>(k * RB));
        if constexpr (T2D) {
            if (k == kUnitRows) {
                tma2d_g2s(dst, mrows, sl * SW, static_cast<int>(row), bar);
            } else {
                for (int i = 0; i < k; ++i) {
                    bulk_g2s(dst + i * RB, snapc + (row + i) * g.hd + static_cast<int64_t>(sl) * SW,
                             static_cast<uint32_t>(RB), bar);
                }
            }
        } else {
            bulk_g2s(dst, snapc + row * g.hd, static_cast<uint32_t>(k * RB), bar);   // one slice: contiguous rows
        }
    };

    const uint4* const kr = krec + chunk * g.n;
    const int c4 = 4 * lane;
    const int swe = (kUneven || kScalar) ? min(SW, g.hd - sl * SW) : SW;   // UNEVEN / SCALAR: the last slice is short
    const bool on = c4 < swe;
    const int nv = kScalar ? min(4, swe - c4) : 4;   // SCALAR: the lane's columns in range (<= 0 when off)
    const int coff = on ? c4 : 0;                  // safe column for token loads
    const int ring_off = (on ? lane : 0) * static_cast<int>(sizeof(RowT));   // the lane's 4 columns in a ring row
    // coff, not c4: an inactive lane (c4 >= SW) never forms a pointer past the bank; its stores are off.
    unsigned char* const bank_lane = reinterpret_cast<unsigned char*>(bank + static_cast<int64_t>(sl) * SW + coff);
    // PF: the slice's column 0 (16-byte aligned: bank, hd * 2 and SW * 2 are, host-checked); copy lane chunk c16.
    const unsigned char* const bank_sl = reinterpret_cast<const unsigned char*>(bank + static_cast<int64_t>(sl) * SW);
    const int c16 = (lane & 15) * 16;
    // WR: this lane's 4 columns of the slice in a compact-snapshot row r: element r * hd + snap_col (written iff on).
    const int64_t snap_col = static_cast<int64_t>(sl) * SW + coff;

    uint4 r0 = ld_rec(kr, s + lane, e);
    uint4 r1 = ld_rec(kr, s + 32 + lane, e);
    uint4 r2 = ld_rec(kr, s + 64 + lane, e);
    uint4 r3 = ld_rec(kr, s + 96 + lane, e);
    if (lane == 0) {
        for (int u = 0; u < NU && u < nunits; ++u) issue_unit(u);
    }
    // QUANT && T2D: the quantized rows before the PDL wait (they read the bank rows the previous step's DW wrote, three
    // launches back, and write only the quantized rows and their scales), overlapping the dual matmul.
    uint8_t* const qpay = QF != 0 ? reinterpret_cast<uint8_t*>(map_rows.opaque[0]) + drow0 * ((QF & 3) == 1 ? g.hd : g.hd / 2)
                                  : nullptr;
    float* const qsc = QF != 0 ? reinterpret_cast<float*>(map_rows.opaque[1]) + drow0 : nullptr;
    if constexpr (kAllSl) {   // every slice of these rows is written by this CTA: quantize them whole before slice 0's walk
        if (sl == 0) q_rows_cta<QF & 3, !kBf>(bank, g, g.NSL, kr, s, e, lane, reinterpret_cast<uint32_t*>(rec_s), qpay, qsc);
    }
    if constexpr (kQC && T2D) {
        struct QV { RowT v[kQB]; };
        q_rows<QF, NW, true, QV>(smem, g, segb, chunk * (g.TU + 1) + cr * NW, g0, kr, s, e, lane,
                                    reinterpret_cast<uint32_t*>(rec_s), sl == 0, qsc,
                                    [&](const uint32_t* sla, int n, QV& q) {
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                q.v[j] = make_zero_row<RowT>();
                if (j < n) q.v[j] = __ldcg(reinterpret_cast<const RowT*>(fx_row<BIG>(bank_lane, sla[j], g.row_bytes)));
            }
        }, [&](const QV& q, float* a) {
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                if constexpr (kBf) {
                    a[j] = on ? q_amax_bf4(q.v[j]) : 0.f;
                } else {
                    a[j] = on ? q_amax4(cvt4(q.v[j])) : 0.f;
                }
            }
        }, [&](const QV& q, int n, int k0, const float2* qw) {
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                if (j < n && on) {
                    const float sc = qw[k0 + j].y;
                    q_store4<QF, !kBf>(qpay + static_cast<int64_t>(k0 + j) * ((QF & 3) == 1 ? g.hd : g.hd / 2), snap_col,
                                          cvt4(q.v[j]), sc, kBf ? __frcp_rn(sc) : 0.f);
                }
            }
        });
    }
#if !defined(FX_PRE_SM90)
    // PDL: everything above reads only data written two or more launches back: Phase 1's records, tile bounds and segb
    // (built before the step loop), the ring units' dense snapshot rows (written by this step's gather, which completed
    // before the dual matmul launched) and, QUANT, the bank rows (the previous step's DW). The token rows are the dual
    // matmul's output (the launch right before): every thread waits for that grid's completion and memory first (no exit
    // precedes this point; the only global writes before it are QUANT's rows and scales, which no other launch of the
    // step reads or writes), then thread 0 copies them. Later reads of predecessor-written data are coherent: token rows by TMA, bank rows by
    // __ldcg / cp.async.cg. Dependents (the next step's gather, which waits for this grid in turn) may launch from here.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr (kScalar) {
        tok_copy_scalar();
    } else if (threadIdx.x == 0) {
        mbar_arrive_tx(tok_bar, static_cast<uint32_t>(C * SW * 4));
        if constexpr (T2D) {
            tma2d_g2s(tok_s, &map_tok, sl * SW, static_cast<int>(tok_row0), tok_bar);
        } else {
            bulk_g2s(tok_s, dv + tok_row0 * g.hd, static_cast<uint32_t>(C * SW * 4), tok_bar);   // SW == hd
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    int issued_u = min(NU, nunits);                // units issued so far (warp-uniform)
    // PF: the token rows are waited for inside batch 0, after its base-row copies are issued (the two latencies overlap).
#if defined(FX_PRE_SM90)
    if constexpr (!kScalar) sdmcp::publish_cta();  // every copier's token rows landed; the CTA barrier publishes them
    const uint32_t ztok = 0u;                      // (SCALAR: tok_copy_scalar ended with its CTA barrier)
#else
    const uint32_t ztok = (PF || kScalar) ? 0u : mbar_wait_dep(tok_bar, 0u);   // SCALAR: copied above (barrier)
#endif
    const float* tok = tok_s + ztok + coff;
    uint32_t zu = 0u;                              // sum of the unit waits' tokens (0): ring reads depend on it
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    int done = 0;                                  // groups emitted so far (= dense row of the next emit)
    int roff = 0;                                  // ring byte offset of row `done` (set at each unit start)
    const int nb = (e - s + 31) >> 5;
    for (int bi = 0; bi < nb; ++bi) {
        if (bi > 0) {
            r0 = r1;
            r1 = r2;
            r2 = r3;
            r3 = ld_rec(kr, s + 32 * (bi + 3) + lane, e);
        }
        const uint32_t x = r0.x;
        const uint32_t ml = __ballot_sync(kFull, (x & (kLast | kInvalid)) == kLast);   // valid LASTs: emit
        const uint32_t me = __ballot_sync(kFull, (x & kLast) != 0u);                   // every LAST: restart
        const uint32_t tokoff = static_cast<uint32_t>(min(static_cast<int>((x >> 9) & 0xFFu), C - 1) * SW);
        // {token offset, scale, row bytes (BIG: row index), cpe}
        rec_s[lane] = make_uint4(tokoff, r0.z, BIG ? r0.y : r0.y * g.row_bytes, r0.w);
        if constexpr (PF) {
            // Base rows of this batch's valid LASTs (ml), bank[slot] -> pfb row j (position pos0 + j): lanes 0-15 copy
            // row 2i, lanes 16-31 row 2i + 1, 16 bytes per lane per row. The previous batch's reads of pfb precede
            // these writes (its closing __syncwarp; the cp.async asm carries a memory clobber). The bytes are the ones
            // BANKB's __ldcg reads later in the walk: no store of this launch writes a valid LAST's slot except that
            // group's own emit, which follows this read (one group per slot per launch: the single-writer rule).
            const uint32_t rbl = r0.y * g.row_bytes;     // this lane's row bytes (= rec_s[lane].z)
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int j = 2 * i + (lane >> 4);
                const uint32_t rbj = __shfl_sync(kFull, rbl, j);
                if (c16 < RB && ((ml >> j) & 1u)) cp_async16(pfb + j * RB + c16, bank_sl + rbj + c16);
            }
            if (bi == 0) tok = tok_s + mbar_wait_dep(tok_bar, 0u) + coff;   // token rows (dependency token)
            cp_async_wait_all();                          // this lane's copies landed; the __syncwarp shares them
        }
        __syncwarp();
        const int pos0 = s + 32 * bi;
        const int kend = min(32, e - pos0);
#pragma unroll 1
        for (int k8 = 0; k8 < kend; k8 += 8) {
            uint2 rr[8];
            float4 xx[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) rr[j] = *reinterpret_cast<const uint2*>(rec_s + k8 + j);
#pragma unroll
            for (int j = 0; j < 8; ++j) xx[j] = *reinterpret_cast<const float4*>(tok + rr[j].x);
            const uint32_t e8 = (me >> k8) & 0xFFu;
            if (e8 == 0u) {
#pragma unroll
                for (int j = 0; j < 8; ++j) fma4(acc, rr[j].y, xx[j]);
            } else if (BANKB) {
                // Bank-direct: the walk's ops in the walk's order (acc = fma(sc, x, acc) per position; at a valid LAST
                // o = fma(base, cpe, acc), one bf16 RN rounding, store; +0 after EVERY LAST); base = bank[slot] read
                // first for every valid LAST of the sub-batch (distinct slots: no store of this sub-batch aliases a
                // load). Inactive lanes (c4 >= SW) read column 0 of the slice (in bounds) and never store.
                const uint32_t l8 = (ml >> k8) & 0xFFu;
                RowT bv[8];
                uint2 rc[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    rc[j] = *reinterpret_cast<const uint2*>(
                        reinterpret_cast<const unsigned char*>(rec_s + k8 + j) + 8);   // {row bytes, cpe}
                    bv[j] = make_zero_row<RowT>();
                    if (l8 & (1u << j)) {
                        if constexpr (PF) {
                            bv[j] = *reinterpret_cast<const RowT*>(pfb + (k8 + j) * RB + ring_off);   // prefetched
                        } else if constexpr (kScalar) {
                            bv[j] = ld_row_n<E>(fx_row<BIG>(bank_lane, rc[j].x, g.row_bytes), nv);
                        } else {
                            bv[j] = __ldcg(reinterpret_cast<const RowT*>(fx_row<BIG>(bank_lane, rc[j].x, g.row_bytes)));
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    fma4(acc, rr[j].y, xx[j]);
                    if (l8 & (1u << j)) {
                        const float cpe = __uint_as_float(rc[j].y);
                        const float4 base = cvt4(bv[j]);
                        float4 o;
                        o.x = fmaf(base.x, cpe, acc.x);
                        o.y = fmaf(base.y, cpe, acc.y);
                        o.z = fmaf(base.z, cpe, acc.z);
                        o.w = fmaf(base.w, cpe, acc.w);
                        if constexpr (kScalar) {
                            if (on) st4_n(reinterpret_cast<E*>(fx_row<BIG>(bank_lane, rc[j].x, g.row_bytes)), o, nv);
                        } else {
                            if (on) st4(reinterpret_cast<E*>(fx_row<BIG>(bank_lane, rc[j].x, g.row_bytes)), o);
                        }
                        if constexpr (WR) {
                            // the group's dense row: ordinal done + (valid LASTs of this sub-batch before position j)
                            const int64_t row = drow0 + done + __popc(l8 & ((1u << j) - 1u));
                            if constexpr (!std::is_same<SE, E>::value) {   // bf16 rows of a float32 bank: RN
                                if constexpr (kScalar) {
                                    if (on) st4_n(snapc + (row * g.hd + snap_col), base, nv);
                                } else {
                                    if (on) st4(snapc + (row * g.hd + snap_col), base);
                                }
                            } else if constexpr (kScalar) {
                                if (on) st_raw_n(snapc + (row * g.hd + snap_col), bv[j], nv);   // raw row
                            } else {
                                if (on) st_raw(snapc + (row * g.hd + snap_col), bv[j]);   // raw row
                            }
                        }
                        if constexpr (kQC && !T2D) {   // one slice: the row, quantized with the warp's amax
                            const int kq = done + __popc(l8 & ((1u << j) - 1u));
                            const float sc = q_scale<QF>(q_warp_max(on ? q_amax4(base) : 0.f));
                            if (lane == 0) qsc[kq] = sc;
                            if (on) q_store4<QF, !kBf>(qpay + static_cast<int64_t>(kq) * ((QF & 3) == 1 ? g.hd : g.hd / 2),
                                                          snap_col, base, sc, kBf ? __frcp_rn(sc) : 0.f);
                        }
                    }
                    if (e8 & (1u << j)) acc = make_float4(0.f, 0.f, 0.f, 0.f);
                }
                if constexpr (WR || (kQC && !T2D)) done += __popc(l8);
            } else {
                const uint32_t l8 = (ml >> k8) & 0xFFu;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    fma4(acc, rr[j].y, xx[j]);
                    if (e8 & (1u << j)) {
                        if (l8 & (1u << j)) {
                            if ((done & (kUnitRows - 1)) == 0) {   // first row of unit u: refill, then wait
                                const int u = done / kUnitRows;
                                if (u >= 1 && u - 1 + NU < nunits) {
                                    // Slot of unit u - 1 is free: every lane's generic reads of it precede the
                                    // async-proxy refill (fence, THEN the warp sync that hands the slot over).
                                    fence_proxy_async();
                                    __syncwarp();
                                    if (lane == 0) issue_unit(u - 1 + NU);
                                    ++issued_u;
                                }
                                zu += mbar_wait_dep(ubar + (u % NU), static_cast<uint32_t>(u / NU) & 1u);
                                roff = (u % NU) * UB;
                            }
                            const RowT bv = *reinterpret_cast<const RowT*>(ring + roff + zu + ring_off);
                            roff += RB;
                            const uint2 rc = *reinterpret_cast<const uint2*>(
                                reinterpret_cast<const unsigned char*>(rec_s + k8 + j) + 8);   // {row bytes, cpe}
                            const float cpe = __uint_as_float(rc.y);
                            const float4 base = cvt4(bv);
                            float4 o;
                            o.x = fmaf(base.x, cpe, acc.x);
                            o.y = fmaf(base.y, cpe, acc.y);
                            o.z = fmaf(base.z, cpe, acc.z);
                            o.w = fmaf(base.w, cpe, acc.w);
                            if (on) st4(reinterpret_cast<E*>(bank_lane + rc.x), o);
                            ++done;
                        }
                        acc = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                }
            }
        }
        __syncwarp();   // this batch's rec_s reads precede the next batch's STS
    }
    // Drain (defensive; normally empty: every issued unit holds rows the walk emitted, so it was waited for): no copy
    // into this CTA's shared memory may still be in flight when the CTA exits.
    for (int u = (done + kUnitRows - 1) / kUnitRows; u < issued_u; ++u) {
        (void)mbar_wait_dep(ubar + (u % NU), static_cast<uint32_t>(u / NU) & 1u);
    }
    if constexpr (PF) {
        if (nb == 0) (void)mbar_wait_dep(tok_bar, 0u);   // empty tile: the token-row TMA lands before the CTA can exit
    }
    if constexpr (kQC && T2D) cluster_wait();     // no peer reads this CTA's rows any more
    if constexpr (kAllSl) {
        if (sl + 1 < g.NSL) {    // CTA-uniform: the next slice of the same tiles
            fence_proxy_async(); // this slice's generic reads of the token rows precede the next slice's async-proxy copy
            __syncthreads();     // every warp is done with this slice (token rows, record batches, bank rows)
            if (threadIdx.x == 0) mbar_inval(tok_bar);   // the prologue initializes it again
            b = (p * g.NSL + sl + 1) * g.K + cr;
            goto fx_slice;
        }
    }
}

// fx_dw2_kernel: the bank-direct DW (fx_dw_kernel with BANKB, never PF) with TWO column slices per warp: lanes 0-15
// take slice 2 * sp, lanes 16-31 slice 2 * sp + 1 of slice pair sp, 8 columns per lane in two groups of 4: columns [4
// l', 4 l' + 4) and [64 + 4 l', 64 + 4 l' + 4) of the half's slice (l' = lane & 15; conflict-free LDS.128 of the token
// rows, 128-byte coalesced 8-byte global accesses per half). One warp walks each position for both slices: the same
// records, masks and branches for all 32 lanes, and per column the BANKB walk's ops in its order (acc = fma(sc, x, acc)
// per position; at a valid LAST o = fma(base, cpe, acc), one bf16 RN rounding, store; acc = +0 after EVERY LAST; base =
// bank[slot] read before the group's own store, the single-writer rule), hence the bits of fx_dw_kernel<NW, *, true,
// true, false, WR, BIG>. Records, masks and addressing are shared by 8 columns instead of 4 (about half the
// instructions per column; the walk is issue-bound). Token rows: one 2D box {2 SW, C} per CTA ([C][2 SW] f32, map from
// fx_maps2). Grid P * (NSL / 2) * K (g.NSL = slice pairs). Shared memory: token rows | record batch [NW][32] uint4 |
// tok_bar. ALLON: SW == 128 (every column group of every lane in range). QUANT (!WR, BIG): the quantized rows as in
// fx_dw_kernel (q_rows before the PDL wait; one cluster = the CTAs of every slice pair of the same tiles).
template <int NW, bool WR, bool BIG, bool ALLON, int QUANT = 0>
__global__ void __launch_bounds__(NW * 32, (NW <= 8 ? 2 : 1)) fx_dw2_kernel(
    __nv_bfloat16* __restrict__ bank, const float* __restrict__ dv, int64_t dv_step_rows,
    const uint4* __restrict__ krec, const int32_t* __restrict__ tb, const int32_t* __restrict__ segb,
    __nv_bfloat16* __restrict__ snapc, int64_t sb_step, int ti, FxGeo g,
    const __grid_constant__ CUtensorMap map_rows, const __grid_constant__ CUtensorMap map_tok) {
    constexpr int QF = QUANT & 7;                  // variant bits as in fx_dw_kernel (no tail bits on this walk)
    constexpr bool kQC = QF == 1 || QF == 2;       // the cluster form
    constexpr bool kAllSl = QF >= 5;               // the all-slices form (as fx_dw_kernel's): slice pairs walked in turn
    static_assert(QF == 0 || (!WR && BIG), "fx_dw2_kernel: quantized rows: 64-bit offsets, no raw rows");
    static_assert((QUANT & 48) == 0, "fx_dw2_kernel: no tail variants (tails take fx_dw_kernel)");
    extern __shared__ __align__(128) unsigned char smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int b = blockIdx.x;
    if constexpr (kQC) b = b / g.NSL % g.K + g.K * (b % g.NSL + g.NSL * (b / g.NSL / g.K));   // cluster order
    if constexpr (kAllSl) b = b / g.K * g.NSL * g.K + b % g.K;   // all slices: grid P * K, from slice pair 0
fx2_slice:   // kAllSl: the next slice pair restarts here (the end of the kernel)
    const int cr = b % g.K;
    b /= g.K;
    const int sp = b % g.NSL;                       // slice pair
    const int p = b / g.NSL;
    const int64_t chunk = static_cast<int64_t>(ti) * g.P + p;
    const int SW = g.SW, C = g.C;
    const int RW = 2 * SW;                          // floats of a token row in shared memory (both slices)
    float* tok_s = reinterpret_cast<float*>(smem);
    uint4* const rec_s = reinterpret_cast<uint4*>(smem + g.tok_bytes) + warp * 32;
    uint64_t* const tok_bar =
        reinterpret_cast<uint64_t*>(smem + g.tok_bytes + static_cast<size_t>(NW) * 32 * sizeof(uint4));
    if (threadIdx.x == 0) mbar_init(tok_bar, 1);
    fence_mbar_init();
    __syncthreads();

    // Token rows of the slice pair, all on tok_bar: [C][2 SW] f32.
    const int64_t tok_row0 = static_cast<int64_t>(ti) * dv_step_rows + static_cast<int64_t>(p) * C;
#if defined(FX_PRE_SM90)
    // Portable: the slice pair's 2 SW floats of each token row, copied by every thread (as fx_dw_kernel's).
    sdmcp::copy_rows16(tok_s, RW * 4, dv + tok_row0 * g.hd + static_cast<int64_t>(sp) * RW, static_cast<int64_t>(g.hd) * 4,
                       C, RW * 4, static_cast<int>(threadIdx.x), NW * 32);
    sdmcp::commit();
#endif

    const int64_t tix = chunk * (g.TU + 1) + cr * NW + warp;
    const int s = __ldg(tb + tix), e = __ldg(tb + tix + 1);
    const int g0 = (WR || QF != 0) ? __ldg(segb + tix) : 0;      // WR / QUANT: the tile's first dense ordinal
    const int64_t drow0 = (static_cast<int64_t>(ti) * sb_step + p) * g.n + g0;   // WR: dense row of the tile's group 0
    const uint4* const kr = krec + chunk * g.n;
    const int h = lane >> 4;                        // the lane's slice: 2 sp + h
    const int lc = 4 * (lane & 15);                 // first column of the lane's group 0 (group 1: lc + 64)
    const bool on0 = ALLON || lc < SW;
    const bool on1 = ALLON || lc + 64 < SW;
    // Safe columns: an inactive group reads column 0 of the slice (in bounds) and never stores.
    const int c0 = on0 ? lc : 0, c1 = on1 ? lc + 64 : 0;
    const int64_t col = static_cast<int64_t>(2 * sp + h) * SW;   // element offset of the lane's slice in a row
    unsigned char* const bank0 = reinterpret_cast<unsigned char*>(bank + col + c0);
    unsigned char* const bank1 = reinterpret_cast<unsigned char*>(bank + col + c1);
    const int64_t snap0 = col + c0, snap1 = col + c1;            // WR: element offsets in a compact row

    uint4 r0 = ld_rec(kr, s + lane, e);
    uint4 r1 = ld_rec(kr, s + 32 + lane, e);
    uint4 r2 = ld_rec(kr, s + 64 + lane, e);
    uint4 r3 = ld_rec(kr, s + 96 + lane, e);
    // QUANT: the quantized rows before the PDL wait (as fx_dw_kernel's), overlapping the dual matmul.
    if constexpr (kAllSl) {   // the whole rows (2 slices per pair), once, before slice pair 0's walk
        if (sp == 0) {
            q_rows_cta<QF & 3, false>(bank, g, 2 * g.NSL, kr, s, e, lane, reinterpret_cast<uint32_t*>(rec_s),
                                      reinterpret_cast<uint8_t*>(map_rows.opaque[0]) + drow0 * ((QF & 3) == 1 ? g.hd : g.hd / 2),
                                      reinterpret_cast<float*>(map_rows.opaque[1]) + drow0);
        }
    }
    if constexpr (kQC) {
        uint8_t* const qpay = reinterpret_cast<uint8_t*>(map_rows.opaque[0]) + drow0 * ((QF & 3) == 1 ? g.hd : g.hd / 2);
        struct QV { uint2 v0[kQB], v1[kQB]; };
        q_rows<QF, NW, true, QV>(smem, g, segb, chunk * (g.TU + 1) + cr * NW, g0, kr, s, e, lane,
                                    reinterpret_cast<uint32_t*>(rec_s), sp == 0,
                                    reinterpret_cast<float*>(map_rows.opaque[1]) + drow0,
                                    [&](const uint32_t* sla, int n, QV& q) {
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                q.v0[j] = make_uint2(0u, 0u);
                q.v1[j] = make_uint2(0u, 0u);
                if (j < n) {
                    q.v0[j] = __ldcg(reinterpret_cast<const uint2*>(fx_row<BIG>(bank0, sla[j], g.row_bytes)));
                    q.v1[j] = __ldcg(reinterpret_cast<const uint2*>(fx_row<BIG>(bank1, sla[j], g.row_bytes)));
                }
            }
        }, [&](const QV& q, float* a) {
#pragma unroll
            for (int j = 0; j < kQB; ++j) a[j] = fmaxf(on0 ? q_amax_bf4(q.v0[j]) : 0.f, on1 ? q_amax_bf4(q.v1[j]) : 0.f);
        }, [&](const QV& q, int n, int k0, const float2* qw) {
#pragma unroll
            for (int j = 0; j < kQB; ++j) {
                if (j < n) {
                    const float sc = qw[k0 + j].y, y = __frcp_rn(sc);
                    uint8_t* const qrow = qpay + static_cast<int64_t>(k0 + j) * ((QF & 3) == 1 ? g.hd : g.hd / 2);
                    if (on0) q_store4<QF, false>(qrow, snap0, cvt4(q.v0[j]), sc, y);
                    if (on1) q_store4<QF, false>(qrow, snap1, cvt4(q.v1[j]), sc, y);
                }
            }
        });
    }
#if !defined(FX_PRE_SM90)
    // PDL (as in fx_dw_kernel): everything above reads data written two or more launches back (tile bounds, segb,
    // records; QUANT: bank rows); the token rows are the dual matmul's output: every thread waits for that grid first (no
    // exit precedes this point; before it only QUANT's rows and scales are written), then thread 0 copies them; the walk
    // reads bank rows after this wait (__ldcg). Dependents (the
    // next step's gather, which waits for this grid in turn) may launch from here.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (threadIdx.x == 0) {
        mbar_arrive_tx(tok_bar, static_cast<uint32_t>(C * RW * 4));
        tma2d_g2s(tok_s, &map_tok, sp * RW, static_cast<int>(tok_row0), tok_bar);
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
#if defined(FX_PRE_SM90)
    sdmcp::publish_cta();                           // every copier's token rows landed; the CTA barrier publishes them
    const uint32_t ztok = 0u;
#else
    const uint32_t ztok = mbar_wait_dep(tok_bar, 0u);
#endif
    const float* const tok0 = tok_s + ztok + h * SW + c0;       // the lane's token columns (group 0 / group 1)
    const float* const tok1 = tok_s + ztok + h * SW + c1;
    float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f);    // running group sums: group 0 / group 1 columns
    float4 a1 = make_float4(0.f, 0.f, 0.f, 0.f);
    int done = 0;                                   // WR: groups emitted so far (dense ordinal of the next emit)
    const int nb = (e - s + 31) >> 5;
    for (int bi = 0; bi < nb; ++bi) {
        if (bi > 0) {
            r0 = r1;
            r1 = r2;
            r2 = r3;
            r3 = ld_rec(kr, s + 32 * (bi + 3) + lane, e);
        }
        const uint32_t x = r0.x;
        const uint32_t ml = __ballot_sync(kFull, (x & (kLast | kInvalid)) == kLast);   // valid LASTs: emit
        const uint32_t me = __ballot_sync(kFull, (x & kLast) != 0u);                   // every LAST: restart
        const uint32_t tokoff = static_cast<uint32_t>(min(static_cast<int>((x >> 9) & 0xFFu), C - 1) * RW);
        // {token row offset (floats), scale, row bytes (BIG: row index), cpe}
        rec_s[lane] = make_uint4(tokoff, r0.z, BIG ? r0.y : r0.y * g.row_bytes, r0.w);
        __syncwarp();
        const int pos0 = s + 32 * bi;
        const int kend = min(32, e - pos0);
#pragma unroll 1
        for (int k8 = 0; k8 < kend; k8 += 8) {
            const uint32_t e8 = (me >> k8) & 0xFFu;
            if (e8 == 0u) {
                // No LAST in the sub-batch: the running sums only.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint2 rt = *reinterpret_cast<const uint2*>(rec_s + k8 + j);   // {token offset, scale}
                    fma4(a0, rt.y, *reinterpret_cast<const float4*>(tok0 + rt.x));
                    fma4(a1, rt.y, *reinterpret_cast<const float4*>(tok1 + rt.x));
                }
                continue;
            }
            // Base rows (bank[slot]) of the sub-batch's valid LASTs, read first (distinct slots: no store of this
            // sub-batch aliases a load).
            const uint32_t l8 = (ml >> k8) & 0xFFu;
            uint2 bv0[8], bv1[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                bv0[j] = make_uint2(0u, 0u);
                bv1[j] = make_uint2(0u, 0u);
                if (l8 & (1u << j)) {
                    const uint32_t rref = rec_s[k8 + j].z;
                    bv0[j] = __ldcg(reinterpret_cast<const uint2*>(fx_row<BIG>(bank0, rref, g.row_bytes)));
                    bv1[j] = __ldcg(reinterpret_cast<const uint2*>(fx_row<BIG>(bank1, rref, g.row_bytes)));
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 rc = rec_s[k8 + j];             // {token offset, scale, row ref, cpe}
                fma4(a0, rc.y, *reinterpret_cast<const float4*>(tok0 + rc.x));
                fma4(a1, rc.y, *reinterpret_cast<const float4*>(tok1 + rc.x));
                if (l8 & (1u << j)) {
                    const float cpe = __uint_as_float(rc.w);
                    const float4 b0 = cvt4(bv0[j]), b1 = cvt4(bv1[j]);
                    float4 o0, o1;
                    o0.x = fmaf(b0.x, cpe, a0.x);
                    o0.y = fmaf(b0.y, cpe, a0.y);
                    o0.z = fmaf(b0.z, cpe, a0.z);
                    o0.w = fmaf(b0.w, cpe, a0.w);
                    o1.x = fmaf(b1.x, cpe, a1.x);
                    o1.y = fmaf(b1.y, cpe, a1.y);
                    o1.z = fmaf(b1.z, cpe, a1.z);
                    o1.w = fmaf(b1.w, cpe, a1.w);
                    if (on0) *reinterpret_cast<uint2*>(fx_row<BIG>(bank0, rc.z, g.row_bytes)) = make_uint2(pack_bf2(o0.x, o0.y), pack_bf2(o0.z, o0.w));
                    if (on1) *reinterpret_cast<uint2*>(fx_row<BIG>(bank1, rc.z, g.row_bytes)) = make_uint2(pack_bf2(o1.x, o1.y), pack_bf2(o1.z, o1.w));
                    if constexpr (WR) {
                        // the group's dense row: ordinal done + (valid LASTs of this sub-batch before position j)
                        const int64_t row = drow0 + done + __popc(l8 & ((1u << j) - 1u));
                        if (on0) *reinterpret_cast<uint2*>(snapc + (row * g.hd + snap0)) = bv0[j];   // raw rows
                        if (on1) *reinterpret_cast<uint2*>(snapc + (row * g.hd + snap1)) = bv1[j];
                    }
                }
                if (e8 & (1u << j)) {
                    a0 = make_float4(0.f, 0.f, 0.f, 0.f);
                    a1 = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            if constexpr (WR) done += __popc(l8);
        }
        __syncwarp();   // this batch's rec_s reads precede the next batch's STS
    }
    if constexpr (kQC) cluster_wait();     // no peer reads this CTA's rows any more
    if constexpr (kAllSl) {
        if (sp + 1 < g.NSL) {    // CTA-uniform: the next slice pair of the same tiles
            fence_proxy_async(); // this pair's generic reads of the token rows precede the next pair's async-proxy copy
            __syncthreads();     // every warp is done with this slice pair
            if (threadIdx.x == 0) mbar_inval(tok_bar);   // the prologue initializes it again
            b = (p * g.NSL + sp + 1) * g.K + cr;
            goto fx2_slice;
        }
    }
}

// cudaFuncSetAttribute is per device: key on (kernel, device).
template <typename K>
void set_smem(K kernel, size_t bytes) {
    static std::mutex mu;
    static std::map<std::pair<const void*, int>, size_t> done;
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_pair(reinterpret_cast<const void*>(kernel), dev);
    auto it = done.find(key);
    if (it != done.end() && it->second >= bytes) return;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    done[key] = bytes;
}

int slices_of(int hd) { return (hd + 127) / 128; }
// Tails: a head whose equal slices would not be 4-column groups takes 128-wide slices, the last one short (UNEVEN,
// variant bit 16: the bank-direct one-slice-per-warp walk only).
// SCALAR (variant bit 32): hd % 4 != 0: element accesses, token rows copied by the CTA; 128-wide slices (the last one
// short), or one slice padded to hd rounded up to 4 (the token rows' shared-memory row stride).
bool scalar_of(int hd) { return hd % 4 != 0; }
bool uneven_of(int hd) {
    const int n = slices_of(hd);
    return !scalar_of(hd) && n > 1 && (hd % n != 0 || (hd / n) % 4 != 0);
}
int slice_width(int hd) {
    if (scalar_of(hd)) return slices_of(hd) > 1 ? 128 : (hd + 3) / 4 * 4;
    return uneven_of(hd) ? 128 : hd / slices_of(hd);
}

// cuTensorMapEncodeTiled from the driver the process already loaded (no -lcuda at build time).
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encode_tiled() {
    static EncodeTiled fn = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) h = dlopen("libcuda.so.1", RTLD_NOW);
        if (h != nullptr) fn = reinterpret_cast<EncodeTiled>(dlsym(h, "cuTensorMapEncodeTiled"));
    });
    return fn;
}

// A 2D map over a row-major [rows, cols] tensor of element size es, box {box_cols, box_rows}.
void encode_2d(CUtensorMap* m, void* base, CUtensorMapDataType dt, int es, uint64_t cols, uint64_t rows,
               uint32_t box_cols, uint32_t box_rows) {
    EncodeTiled fn = encode_tiled();
    TORCH_CHECK(fn != nullptr, "fx_maps: cuTensorMapEncodeTiled not found in libcuda.so.1");
    std::memset(m, 0, sizeof(CUtensorMap));
    const cuuint64_t dims[2] = {cols, rows};
    const cuuint64_t strides[1] = {cols * static_cast<uint64_t>(es)};
    const cuuint32_t box[2] = {box_cols, box_rows};
    const cuuint32_t estr[2] = {1u, 1u};
    const CUresult r = fn(m, dt, 2, base, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                          CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "fx_maps: cuTensorMapEncodeTiled failed (", static_cast<int>(r), ")");
}

}  // namespace

// CPU uint8 [2, 128]: the 2D maps for column-sliced heads (hd > 128): dense snapc rows box {SW, 8} (the bank dtype: bf16
// or f32) and the token rows box {SW, C} (f32 delta_v). Empty for one-slice heads (1D bulk copies, no maps).
// wide_rows (<= 2^31): from this many snapc rows the snapc slot is left zeroed (its 32-bit row coordinates end at 2^31).
// Only the ring walks read that map and the launchers instantiate the bank-direct walk only (BANKB: no ring units), so
// the outputs do not change; below wide_rows the map is encoded as before. The caller passes 2^31 (tests: 0).
at::Tensor fx_maps(at::Tensor snapc, at::Tensor dv, int64_t C, int64_t wide_rows) {
    TORCH_CHECK(wide_rows >= 0 && wide_rows <= (int64_t(1) << 31), "fx_maps: wide_rows must be in [0, 2^31]");
    TORCH_CHECK(snapc.dim() >= 1 && snapc.size(-1) >= 1 && snapc.size(-1) <= kMaxHeadDim,
                "fx_maps: head_dim must be in [1, ", kMaxHeadDim, "]");   // before any int arithmetic
    const int hd = static_cast<int>(snapc.size(-1));
    const int NSL = slices_of(hd);
    if (NSL == 1) return at::empty({0}, at::TensorOptions().dtype(at::kByte));
    if (scalar_of(hd)) {                          // SCALAR: the walk copies its token rows itself (no TMA maps)
        TORCH_CHECK(C >= 1 && C <= 256, "fx_maps: unsupported chunk");
        return at::zeros({2, 128}, at::TensorOptions().dtype(at::kByte));
    }
    const int SW = slice_width(hd);               // UNEVEN heads: box 128; TMA zero-fills the columns past hd
    const bool bf = snapc.scalar_type() == at::kBFloat16;
    const int es = bf ? 2 : 4;
    // rows16: the snapc map (ring units) needs 16-byte slice rows and rows; the token map only hd % 4 == 0, SW % 4 == 0
    // (tails: the bank-direct walks, which never read the snapc map, get a zeroed one)
    const bool rows16 = !uneven_of(hd) && (SW * es) % 16 == 0 && (hd * es) % 16 == 0;
    TORCH_CHECK((hd % NSL == 0 || uneven_of(hd)) && SW % 4 == 0 && hd % 4 == 0 && C >= 1 && C <= 256,
                "fx_maps: unsupported head_dim / chunk");
    TORCH_CHECK(snapc.is_cuda() && (bf || snapc.scalar_type() == at::kFloat) && snapc.is_contiguous() &&
                reinterpret_cast<uintptr_t>(snapc.data_ptr()) % 16 == 0, "fx_maps: snapc bf16 / f32 contiguous aligned");
    TORCH_CHECK(dv.is_cuda() && dv.scalar_type() == at::kFloat && dv.is_contiguous() && dv.size(-1) == hd &&
                reinterpret_cast<uintptr_t>(dv.data_ptr()) % 16 == 0, "fx_maps: delta_v f32 contiguous aligned");
    const int64_t srows = snapc.numel() / hd, drows = dv.numel() / hd;
    // the token map's rows are token rows (< 2^31: memory_ops' token-row bound); the snapc map is encoded below wide_rows
    TORCH_CHECK(drows < (int64_t(1) << 31), "fx_maps: row coordinates must fit int32");
    at::Tensor out = at::zeros({2, 128}, at::TensorOptions().dtype(at::kByte));
    const c10::cuda::CUDAGuard guard(snapc.device());
    alignas(64) CUtensorMap m;
    if (rows16 && srows < wide_rows) {
        encode_2d(&m, snapc.data_ptr(), bf ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32, es, hd,
                  srows, SW, kUnitRows);
        std::memcpy(out.data_ptr<uint8_t>(), &m, 128);
    }
    encode_2d(&m, dv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, hd, drows, SW, static_cast<uint32_t>(C));
    std::memcpy(out.data_ptr<uint8_t>() + 128, &m, 128);
    return out;
}

// CPU uint8 [2, 128] for the two-slice walk (fx_dw2_kernel): the token rows of a slice PAIR, box {2 SW, C}
// (f32 delta_v), in both slots (that kernel reads no compact rows; the first slot only keeps a valid descriptor).
// Needs an even number of column slices (hd = 2 * pairs * SW, SW <= 128).
at::Tensor fx_maps2(at::Tensor snapc, at::Tensor dv, int64_t C) {
    TORCH_CHECK(dv.dim() >= 1 && dv.size(-1) >= 1 && dv.size(-1) <= kMaxHeadDim,
                "fx_maps2: head_dim must be in [1, ", kMaxHeadDim, "]");
    const int hd = static_cast<int>(dv.size(-1));
    const int NSL = slices_of(hd);
    TORCH_CHECK(NSL >= 2 && NSL % 2 == 0, "fx_maps2: the two-slice walk needs an even number of column slices");
    const int SW = hd / NSL;
    TORCH_CHECK(hd % NSL == 0 && (SW * 2) % 16 == 0 && C >= 1 && C <= 256, "fx_maps2: unsupported head_dim / chunk");
    TORCH_CHECK(snapc.size(-1) == hd, "fx_maps2: snapc rows of head_dim");
    TORCH_CHECK(dv.is_cuda() && dv.scalar_type() == at::kFloat && dv.is_contiguous() &&
                reinterpret_cast<uintptr_t>(dv.data_ptr()) % 16 == 0, "fx_maps2: delta_v f32 contiguous aligned");
    const int64_t drows = dv.numel() / hd;
    TORCH_CHECK(drows < (int64_t(1) << 31), "fx_maps2: row coordinates must fit int32");
    at::Tensor out = at::empty({2, 128}, at::TensorOptions().dtype(at::kByte));
    const c10::cuda::CUDAGuard guard(dv.device());
    alignas(64) CUtensorMap m;
    encode_2d(&m, dv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, hd, drows, static_cast<uint32_t>(2 * SW),
              static_cast<uint32_t>(C));
    std::memcpy(out.data_ptr<uint8_t>(), &m, 128);
    std::memcpy(out.data_ptr<uint8_t>() + 128, &m, 128);
    return out;
}

// Everything the DW launch checks and derives, computed once (fx_plan_make); a launch then needs only the step index
// (fx_launch_plan). fx_dw_plan / fx_dw_step expose that split to Python: the checks and the argument binding run once
// per factory call instead of once per step.
struct FxPlan {
    uint64_t magic;                 // kPlanMagic: a plan tensor made by fx_dw_plan of this module
    int64_t ti_max;                 // every step index in [0, ti_max] passes the step checks (< 0: none does)
    int32_t device;
    int32_t blocks;                 // 0: nothing to launch (no CTAs, or an empty bank)
    int32_t threads;
    int32_t cluster;                // CTAs per thread-block cluster (quantized column-sliced walks); 0: no cluster
    int32_t portable;               // 1: the portable side (A100 / RTX rows): no tensor maps, no PDL, no cluster
    uint64_t smem;
    const void* fn;                 // the fx_dw_kernel instantiation (its smem attribute set for `device`)
    void* bank;                     // bf16 or f32 rows (the kernel instantiation's element type)
    const float* dv;
    int64_t dv_step_rows;
    const uint4* krec;
    const int32_t* tb;
    const int32_t* segb;
    void* snapc;                    // the bank's element type
    int64_t sb_step;
    FxGeo g;
    CUtensorMap mrows;              // 64-byte aligned by its type
    CUtensorMap mtok;
};
constexpr uint64_t kPlanMagic = 0x464750314e414c50ull;

// The quantized instantiation of a plan (QUANT 1 fp8 / 2 int4; bank-direct, 64-bit row offsets, no raw rows; nw 8 / 16).
// quant 1 / 2: the cluster form; 5 / 6: the all-slices form (column-sliced heads only: t2d or the two-slice walk).
const void* fx_q_fn(int64_t quant, bool w2, bool bf, int64_t nw, bool t2d, bool allon) {
    TORCH_CHECK((quant == 1 || quant == 2 || ((quant == 5 || quant == 6) && (t2d || w2))) && (nw == 8 || nw == 16) &&
                (bf || !w2), "fx_dw: no quantized instantiation");
#define FX_QE(E_, NW_, Q_) (t2d ? reinterpret_cast<const void*>(fx_dw_kernel<E_, NW_, 2, true, true, false, false, true, Q_>) \
                                : reinterpret_cast<const void*>(fx_dw_kernel<E_, NW_, 2, false, true, false, false, true, Q_>))
#define FX_Q2(NW_, Q_) (allon ? reinterpret_cast<const void*>(fx_dw2_kernel<NW_, false, true, true, Q_>) \
                              : reinterpret_cast<const void*>(fx_dw2_kernel<NW_, false, true, false, Q_>))
#define FX_QN(Q_) (w2 ? (nw == 8 ? FX_Q2(8, Q_) : FX_Q2(16, Q_)) \
                      : bf ? (nw == 8 ? FX_QE(__nv_bfloat16, 8, Q_) : FX_QE(__nv_bfloat16, 16, Q_)) \
                           : (nw == 8 ? FX_QE(float, 8, Q_) : FX_QE(float, 16, Q_)))
// the all-slices form exists for column-sliced heads only (T2D = true; one-slice heads quantize in the walk)
#define FX_QA(E_, NW_, Q_) reinterpret_cast<const void*>(fx_dw_kernel<E_, NW_, 2, true, true, false, false, true, Q_>)
#define FX_QAN(Q_) (w2 ? (nw == 8 ? FX_Q2(8, Q_) : FX_Q2(16, Q_)) \
                       : bf ? (nw == 8 ? FX_QA(__nv_bfloat16, 8, Q_) : FX_QA(__nv_bfloat16, 16, Q_)) \
                            : (nw == 8 ? FX_QA(float, 8, Q_) : FX_QA(float, 16, Q_)))
    switch (quant) {
        case 1: return FX_QN(1);
        case 2: return FX_QN(2);
        case 5: return FX_QAN(5);
        default: return FX_QAN(6);
    }
#undef FX_QAN
#undef FX_QA
#undef FX_QN
#undef FX_Q2
#undef FX_QE
}

// Clusters of `cluster` CTAs of fn (smem bytes, `threads` each; > 8: the non-portable size) that can be co-resident on
// the current device (cudaOccupancyMaxActiveClusters): 0 = they never fit.
int64_t cluster_capacity(const void* fn, int64_t cluster, size_t smem, int threads) {
    set_smem(fn, smem);
    if (cluster > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = static_cast<unsigned>(cluster);
    at[0].val.clusterDim.y = at[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(cluster));
    cfg.blockDim = dim3(static_cast<unsigned>(threads));
    cfg.dynamicSmemBytes = smem;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, fn, &cfg));
    return n;
}

// A portable plan's validation (A100 / RTX rows, decided in Python from _device.backend): the device runs the portable
// image of fn (its PTX target is not sm_90 / 100 / 103), so the kernel takes the cp.async side the plan is made for (no
// tensor maps, no PDL wait). Cached per (function, device); never called for modern plans.
bool fx_portable_image(const void* fn) {
    static std::mutex mu;
    static std::map<std::pair<const void*, int>, bool> done;
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_pair(fn, dev);
    auto it = done.find(key);
    if (it != done.end()) return it->second;
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, fn));
    const bool portable = a.ptxVersion != 90 && a.ptxVersion != 100 && a.ptxVersion != 103;
    done[key] = portable;
    return portable;
}

// All the launch checks except the step's, the geometry, the shared memory size, the kernel instantiation (and its
// smem attribute). The caller holds a CUDAGuard on bank's device. portable (Python: _device.backend() != "modern"):
// no tensor maps (maps empty), no programmatic launch, no cluster; smem_cap: the class's opt-in bytes per block
// (_device.smem_cap; modern 232,448).
// bankb: 1 = bank-direct, 3 = bank-direct + compact-snapshot write, 4 / 5 = 1 / 3 with two column slices per warp
// (fx_dw2_kernel); the checks refuse other values. quant 1 (fp8) / 2 (int4), bankb 1 / 4 only: the compact rows go
// quantized to payload ([>= N * n, hd] or [>= N * n, hd / 2] bytes) and scale (f32 [>= N * n]); snapc only anchors the maps.
FxPlan fx_plan_make(at::Tensor bank, at::Tensor dv, int64_t dv_step_rows, at::Tensor krec, at::Tensor tb,
                    at::Tensor segb, at::Tensor snapc, int64_t sb_step, int64_t P, int64_t C, int64_t K, int64_t nw,
                    int64_t nu, at::Tensor maps, int64_t bankb, int64_t quant, at::Tensor payload, at::Tensor scale,
                    int64_t portable, int64_t smem_cap) {
    TORCH_CHECK((portable == 0 || portable == 1) && smem_cap > 0 && smem_cap <= 227 * 1024,
                "fx_dw: portable must be 0 or 1 and smem_cap in (0, 227 KiB]");
    TORCH_CHECK(bankb == 1 || bankb == 3 || bankb == 4 || bankb == 5, "fx_dw: bankb must be 1 (bank-direct), "
                "3 (bank-direct + compact-snapshot write), 4 (1, two slices per warp) or 5 (3, two slices per warp)");
    // bankb 3 / 5: the DW writes the dense compact rows of every step's chunks (the every-step snapshot only).
    const bool wr = bankb == 3 || bankb == 5;
    const bool w2 = bankb == 4 || bankb == 5;      // fx_dw2_kernel (two column slices per warp)
    TORCH_CHECK(!wr || (sb_step == P && P >= 1), "fx_dw: the compact-snapshot write needs sb_step == P");
    TORCH_CHECK(quant == 0 || ((quant == 1 || quant == 2 || quant == 5 || quant == 6) && (bankb == 1 || bankb == 4) &&
                               sb_step == P && P >= 1),
                "fx_dw: quantized compact rows (quant 1 / 2) need bankb 1 or 4 and sb_step == P");
    // bf16 banks, or f32 banks (the one-slice-per-warp bank-direct walk, nw 8 / 16, 64-bit row offsets).
    const bool bf = bank.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bank.dim() == 2 && bank.is_contiguous() && (bf || bank.scalar_type() == at::kFloat),
                "fx_dw: bank [slots, hd] bf16 / f32 contiguous");
    TORCH_CHECK(bf || (!w2 && (nw == 8 || nw == 16)), "fx_dw: a float32 bank takes the one-slice walk with 8 or 16 warps");
    const int es = bf ? 2 : 4;
    // Banks of > 4 GiB run the bank-direct walk with 64-bit row offsets (BIG instantiations); the record's slot
    // field is 32-bit (rows < 2^31 checked below), the 32-bit byte-offset kernels serve every bf16 bank <= 4 GiB
    // unchanged. float32 banks always take the BIG instantiations.
    const bool big = !bf || static_cast<uint64_t>(bank.numel()) * 2 > (uint64_t(1) << 32);
    TORCH_CHECK(!big || (bankb != 0 && bankb != 2),
                "fx_dw: a bank of > 4 GiB needs the bank-direct walk without prefetch (bankb 1, 3, 4 or 5)");
    // Every value narrowed to int below, and every product, is bounded here first (64-bit arithmetic).
    constexpr int64_t kI32 = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(bank.size(1) >= 1 && bank.size(1) <= kMaxHeadDim, "fx_dw: head_dim must be in [1, ", kMaxHeadDim, "]");
    TORCH_CHECK(bank.size(0) <= kI32, "fx_dw: bank rows must fit the records' 32-bit slot field");
    TORCH_CHECK(P >= 0 && P <= kI32 && K >= 0 && K <= kI32 && nw >= 1 && nw <= 32, "fx_dw: P, K, nw out of range");
    // n (positions per chunk) <= 2^30: the kernel's position arithmetic (s + 32 * (bi + 3) + lane <= n + 127, e - s + 31)
    // stays in int; Phase 1 emits n <= 65536. The tile table's column count is bounded before any cast, so
    // TU = cols - 1 and the kernel's TU + 1 fit int.
    TORCH_CHECK(krec.dim() == 3 && krec.size(1) <= (int64_t(1) << 30) && tb.dim() == 2 && tb.size(1) >= 1 &&
                tb.size(1) <= kI32, "fx_dw: record / tile table sizes out of range");
    TORCH_CHECK(dv_step_rows >= 0 && dv_step_rows <= dv.numel() && (sb_step == 0 || sb_step == P),
                "fx_dw: dv_step_rows / sb_step out of range (sb_step is 0 or P)");
    const int hd = static_cast<int>(bank.size(1));
    const int NSL = slices_of(hd);
    const bool uneven = uneven_of(hd);           // tails: 128-wide slices, the last one short
    const bool scalar = scalar_of(hd);           // tails: hd % 4 != 0
    TORCH_CHECK(hd % NSL == 0 || uneven || scalar, "fx_dw: head_dim ", hd, " does not split into ", NSL, " slices");
    const int SW = slice_width(hd);
    // 16-byte slice rows and rows (ring units, snapc maps, the two-slice walk); the bank-direct one-slice-per-warp walk
    // (bankb 1 / 3) needs only 8-byte (bf16) / 16-byte (f32) lane vectors: hd % 4 == 0 (tails)
    // rows16: 16-byte slice rows and rows (every regular plan); anything else is a tail (classes a / b / c)
    const bool rows16 = !uneven && !scalar && (SW * es) % 16 == 0 && (hd * es) % 16 == 0;
    TORCH_CHECK(rows16 || ((bankb == 1 || bankb == 3) && quant == 0 && (nw == 8 || nw == 16)),
                "fx_dw: tail shapes (rows that are not 16-byte multiples, uneven slices, hd % 4 != 0): the unquantized "
                "bank-direct one-slice-per-warp walk, 8 or 16 warps");
    TORCH_CHECK(SW % 4 == 0 && SW <= 128 && (hd % 4 == 0 || scalar) && (rows16 || bankb == 1 || bankb == 3),
                "fx_dw: slice rows must be 16-byte multiples, or hd % 4 == 0 on the bank-direct walk (hd ", hd, ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bank.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(dv.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(snapc.data_ptr()) % 16 == 0, "fx_dw: 16-byte alignment");
    TORCH_CHECK(dv.scalar_type() == at::kFloat && dv.is_contiguous() && dv.device() == bank.device(),
                "fx_dw: dv f32 contiguous");
    TORCH_CHECK(C >= 1 && C <= 256, "fx_dw: C <= 256");
    TORCH_CHECK(krec.dim() == 3 && krec.size(2) == 4 && krec.is_contiguous() && krec.scalar_type() == at::kInt &&
                reinterpret_cast<uintptr_t>(krec.data_ptr()) % 16 == 0 && krec.device() == bank.device(),
                "fx_dw: krec [N, n, 4] int32, 16-byte aligned");
    const int n = static_cast<int>(krec.size(1));
    const int TU = static_cast<int>(tb.size(1) - 1);
    TORCH_CHECK(tb.dim() == 2 && tb.size(0) == krec.size(0) && tb.scalar_type() == at::kInt && tb.is_contiguous(),
                "fx_dw: tb [N, TU + 1] int32");
    TORCH_CHECK(static_cast<int64_t>(TU) == K * nw, "fx_dw: tile table holds ", TU, " tiles, the grid needs ", K * nw);
    TORCH_CHECK(segb.scalar_type() == at::kInt && segb.is_contiguous() && segb.sizes() == tb.sizes(),
                "fx_dw: segb int32 shaped like tb");
    TORCH_CHECK(tb.device() == bank.device() && segb.device() == bank.device(), "fx_dw: tb / segb on the bank's device");
    if (wr) {   // WR: the compact rows are written: they must not alias the bank or delta_v
        const auto lo = [](const at::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()); };
        const auto overlap = [&](const at::Tensor& a, const at::Tensor& b) {
            return lo(a) < lo(b) + b.nbytes() && lo(b) < lo(a) + a.nbytes();
        };
        TORCH_CHECK(!overlap(snapc, bank) && !overlap(snapc, dv),
                    "fx_dw: the compact snapshot must not overlap the bank or delta_v");
    }
    // bf16 compact rows of a float32 bank (snapshot_quant bf16): written by the WR walk (bankb 3), rounded to bf16
    const bool s16 = wr && quant == 0 && bank.scalar_type() == at::kFloat && snapc.scalar_type() == at::kBFloat16;
    TORCH_CHECK((snapc.scalar_type() == bank.scalar_type() || s16) && snapc.is_contiguous() && snapc.device() == bank.device() &&
                snapc.size(-1) == hd &&
                (quant != 0 || n == 0 || (sb_step == 0 ? P : krec.size(0)) <= snapc.numel() / hd / n),   // no overflow
                "fx_dw: snapc [rows, hd] in the bank dtype (or bf16 rows of a float32 bank's WR walk) holding the dense "
                "rows of every chunk read");
    if (quant != 0) {   // per-entry-sized quantized rows, never aliasing the bank / delta_v
        const auto lo = [](const at::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()); };
        const auto apart = [&](const at::Tensor& a, const at::Tensor& b) {
            return lo(a) >= lo(b) + b.nbytes() || lo(b) >= lo(a) + a.nbytes();
        };
        TORCH_CHECK(payload.is_contiguous() && payload.element_size() == 1 && payload.device() == bank.device() &&
                    payload.dim() == 2 && payload.size(1) == ((quant & 3) == 1 ? hd : hd / 2) &&
                    payload.size(0) >= krec.size(0) * n && lo(payload) % 16 == 0 && scale.is_contiguous() &&
                    scale.scalar_type() == at::kFloat && scale.device() == bank.device() &&
                    scale.numel() >= krec.size(0) * n, "fx_dw: payload [>= N * n, hd or hd / 2] bytes, 16-byte aligned, "
                    "and scale f32 [>= N * n], contiguous on the bank's device");
        TORCH_CHECK(apart(payload, bank) && apart(payload, dv) && apart(scale, bank) && apart(scale, dv) &&
                    apart(payload, scale), "fx_dw: payload / scale must not overlap the bank, delta_v or each other");
    }
    // The step checks (0 <= ti <= INT32_MAX, (ti + 1) * P <= N, ti * dv_step_rows + P * C <= delta_v rows) as one range:
    // every ti in [0, ti_max] passes them (64-bit, division forms: no overflow); fx_launch_plan checks ti against it.
    const int64_t dv_rows_total = dv.numel() / hd;
    TORCH_CHECK(P * C <= dv_rows_total, "fx_dw: delta_v rows");
    int64_t ti_max = kI32;
    if (P > 0) ti_max = std::min(ti_max, krec.size(0) / P - 1);
    if (dv_step_rows > 0) ti_max = std::min(ti_max, (dv_rows_total - P * C) / dv_step_rows);
    const bool t2d = NSL > 1;
    TORCH_CHECK(!w2 || (NSL >= 2 && NSL % 2 == 0), "fx_dw: the two-slice walk needs an even number of column slices");
    alignas(64) CUtensorMap mrows, mtok;
    std::memset(&mrows, 0, sizeof(mrows));
    std::memset(&mtok, 0, sizeof(mtok));
    if (portable) {   // the portable side addresses the token rows through dv itself: no tensor map is read
        TORCH_CHECK(maps.numel() == 0, "fx_dw: a portable plan takes no tensor maps");
        TORCH_CHECK(quant < 5 ? !(quant != 0 && t2d) : true, "fx_dw: the portable side's quantized column-sliced rows take "
                    "the all-slices form (quant 5 / 6; no clusters)");
    } else if (t2d) {
        TORCH_CHECK(maps.device().is_cpu() && maps.scalar_type() == at::kByte && maps.numel() == 256 &&
                    maps.is_contiguous(), "fx_dw: column-sliced heads need fx_maps(snapc, delta_v, C)");
        std::memcpy(&mrows, maps.data_ptr<uint8_t>(), 128);
        std::memcpy(&mtok, maps.data_ptr<uint8_t>() + 128, 128);
    }
    FxGeo g;
    g.P = static_cast<int>(P);
    g.C = static_cast<int>(C);
    g.n = n;
    g.hd = hd;
    g.NSL = w2 ? NSL / 2 : NSL;        // two-slice walk: slice PAIRS
    g.SW = SW;
    g.K = static_cast<int>(K);
    g.TU = TU;
    g.tok_bytes = (static_cast<int>(C) * SW * (w2 ? 2 : 1) * 4 + 127) / 128 * 128;
    g.row_bytes = static_cast<uint32_t>(hd) * static_cast<uint32_t>(es);
    const size_t ub = static_cast<size_t>(kUnitRows) * SW * es;
    const size_t nur = bankb ? 0 : static_cast<size_t>(nu);   // BANKB: no rings (must match the kernel's NUR)
    const size_t pfb = bankb == 2 ? static_cast<size_t>(32) * SW * es : 0;   // PF buffer per warp (kernel's PFB)
    const size_t smem0 = static_cast<size_t>(g.tok_bytes) + static_cast<size_t>(nw) * nur * ub +
                        static_cast<size_t>(nw) * pfb +
                        static_cast<size_t>(nw) * 32 * 16 + 8 * (1 + static_cast<size_t>(nw) * nur);
    // quantized column-sliced walks: one cluster per tile range (every slice), {amax, scale} per valid group (q_rows)
    // quant 5 / 6 (all slices): one CTA per tile set walks every slice (grid P * K), no cluster, no {amax, scale} rows
    const bool all_sl = quant >= 5;
    TORCH_CHECK(!all_sl || t2d, "fx_dw: the all-slices quantized form needs a column-sliced head");
    const int cluster = (quant != 0 && !all_sl && t2d) ? g.NSL : 0;
    TORCH_CHECK(cluster <= 16, "fx_dw: quantized rows: ", cluster, " CTAs per cluster (at most 16)");
    const size_t smem = smem0 + (cluster != 0 ? 8 + static_cast<size_t>(nw) * kTileRows * 8 : 0);
    TORCH_CHECK(smem <= static_cast<size_t>(smem_cap), "fx_dw: ", smem, " bytes of shared memory per CTA (cap ", smem_cap, ")");
    TORCH_CHECK(P == 0 || K == 0 || P * g.NSL <= kI32 / K, "fx_dw: grid of P * NSL * K blocks exceeds int32");
    const int blocks = static_cast<int>(all_sl ? P * K : P * g.NSL * K);
    FxPlan pl;
    std::memset(&pl, 0, sizeof(pl));
    pl.magic = kPlanMagic;
    pl.ti_max = ti_max;
    pl.device = static_cast<int32_t>(bank.get_device());
    pl.smem = smem;
    pl.bank = bank.data_ptr();
    pl.dv = dv.data_ptr<float>();
    pl.dv_step_rows = dv_step_rows;
    pl.krec = reinterpret_cast<const uint4*>(krec.data_ptr<int32_t>());
    pl.tb = tb.data_ptr<int32_t>();
    pl.segb = segb.data_ptr<int32_t>();
    pl.snapc = snapc.data_ptr();
    pl.sb_step = sb_step;
    pl.g = g;
    std::memcpy(&pl.mrows, &mrows, sizeof(mrows));
    std::memcpy(&pl.mtok, &mtok, sizeof(mtok));
    if (quant != 0) {   // the quantized walks' output pointers, in the rows map they never read (see fx_dw_kernel)
        std::memset(&pl.mrows, 0, sizeof(pl.mrows));
        pl.mrows.opaque[0] = reinterpret_cast<uint64_t>(payload.data_ptr());
        pl.mrows.opaque[1] = reinterpret_cast<uint64_t>(scale.data_ptr());
    }
    pl.cluster = cluster;
    pl.portable = static_cast<int32_t>(portable);
    // Nothing to write (no CTAs, or an empty bank: no row any record could name): no launch, so the kernel never forms
    // a bank pointer outside an empty array; no instantiation is chosen either.
    pl.blocks = (blocks == 0 || bank.numel() == 0) ? 0 : blocks;
    if (pl.blocks == 0) return pl;
    if (quant != 0) {
        pl.fn = fx_q_fn(quant, w2, bf, nw, t2d, SW == 128);
        pl.threads = static_cast<int32_t>(nw) * 32;
        set_smem(pl.fn, smem);
        TORCH_CHECK(cluster <= 1 || cluster_capacity(pl.fn, cluster, smem, pl.threads) > 0, "fx_dw: clusters of ", cluster,
                    " CTAs with ", smem, " bytes of shared memory each do not fit this device");
        TORCH_CHECK(!portable || fx_portable_image(pl.fn), "fx_dw: a portable (A100 / RTX) plan, but this device runs the "
                    "modern image of the kernel (build the extensions for the row's architecture)");
        return pl;
    }
#define FX_DW(NW_, NU_, T2D_, BK_, PF_, WR_, BIG_) FX_DWE(__nv_bfloat16, NW_, NU_, T2D_, BK_, PF_, WR_, BIG_)
#define FX_DWE(E_, NW_, NU_, T2D_, BK_, PF_, WR_, BIG_)                                           \
    {                                                                                             \
        auto kf = fx_dw_kernel<E_, NW_, NU_, T2D_, BK_, PF_, WR_, BIG_>;                          \
        set_smem(kf, smem);                                                                       \
        pl.fn = reinterpret_cast<const void*>(kf);                                                \
        pl.threads = NW_ * 32;                                                                    \
    }
// Tails: UNEVEN (variant 16) column-sliced bank-direct walks.
#define FX_DWV(E_, NW_, WR_, BIG_)                                                                \
    {                                                                                             \
        auto kf = fx_dw_kernel<E_, NW_, 2, true, true, false, WR_, BIG_, 16>;                     \
        set_smem(kf, smem);                                                                       \
        pl.fn = reinterpret_cast<const void*>(kf);                                                \
        pl.threads = NW_ * 32;                                                                    \
    }
#define FX_DW_UB(NW_) { if (wr) { if (big) FX_DWV(__nv_bfloat16, NW_, true, true) else FX_DWV(__nv_bfloat16, NW_, true, false) } \
                        else { if (big) FX_DWV(__nv_bfloat16, NW_, false, true) else FX_DWV(__nv_bfloat16, NW_, false, false) } }
#define FX_DW_UF(NW_) { if (wr) FX_DWV(float, NW_, true, true) else FX_DWV(float, NW_, false, true) }
// Tails: SCALAR (variant 32) bank-direct walks, one slice (T2D false) or column slices.
#define FX_DWS(E_, NW_, T2D_, WR_, BIG_)                                                          \
    {                                                                                             \
        auto kf = fx_dw_kernel<E_, NW_, 2, T2D_, true, false, WR_, BIG_, 32>;                     \
        set_smem(kf, smem);                                                                       \
        pl.fn = reinterpret_cast<const void*>(kf);                                                \
        pl.threads = NW_ * 32;                                                                    \
    }
#define FX_DW_SB2(NW_, T2D_) { if (wr) { if (big) FX_DWS(__nv_bfloat16, NW_, T2D_, true, true) else FX_DWS(__nv_bfloat16, NW_, T2D_, true, false) } \
                               else { if (big) FX_DWS(__nv_bfloat16, NW_, T2D_, false, true) else FX_DWS(__nv_bfloat16, NW_, T2D_, false, false) } }
#define FX_DW_SB(NW_) { if (t2d) FX_DW_SB2(NW_, true) else FX_DW_SB2(NW_, false) }
#define FX_DW_SF2(NW_, T2D_) { if (wr) FX_DWS(float, NW_, T2D_, true, true) else FX_DWS(float, NW_, T2D_, false, true) }
#define FX_DW_SF(NW_) { if (t2d) FX_DW_SF2(NW_, true) else FX_DW_SF2(NW_, false) }
#define FX_W2(NW_, WR_, BIG_, ALLON_)                                                            \
    {                                                                                             \
        auto kf = fx_dw2_kernel<NW_, WR_, BIG_, ALLON_>;                                          \
        set_smem(kf, smem);                                                                       \
        pl.fn = reinterpret_cast<const void*>(kf);                                                \
        pl.threads = NW_ * 32;                                                                    \
    }
#define FX_W2_A(NW_, WR_, BIG_) { if (SW == 128) FX_W2(NW_, WR_, BIG_, true) else FX_W2(NW_, WR_, BIG_, false) }
#define FX_W2_N(NW_) { if (wr) { if (big) FX_W2_A(NW_, true, true) else FX_W2_A(NW_, true, false) } \
                       else { if (big) FX_W2_A(NW_, false, true) else FX_W2_A(NW_, false, false) } }
#define FX_DW_B(NW_) { if (big) { if (t2d) FX_DW(NW_, 2, true, true, false, false, true) else FX_DW(NW_, 2, false, true, false, false, true) } \
                       else { if (t2d) FX_DW(NW_, 2, true, true, false, false, false) else FX_DW(NW_, 2, false, true, false, false, false) } }
#define FX_DW_W(NW_) { if (big) { if (t2d) FX_DW(NW_, 2, true, true, false, true, true) else FX_DW(NW_, 2, false, true, false, true, true) } \
                       else { if (t2d) FX_DW(NW_, 2, true, true, false, true, false) else FX_DW(NW_, 2, false, true, false, true, false) } }
// float32 banks, bf16 compact rows (snapshot_quant bf16): the WR walks (one slice / column slices, UNEVEN 16, SCALAR 32)
// with bf16 stores of the dense rows, 64-bit row offsets.
#define FX_S16(NW_, T2D_, V_)                                                                     \
    {                                                                                             \
        auto kf = fx_dw_kernel<float, NW_, 2, T2D_, true, false, true, true, V_, __nv_bfloat16>;  \
        set_smem(kf, smem);                                                                       \
        pl.fn = reinterpret_cast<const void*>(kf);                                                \
        pl.threads = NW_ * 32;                                                                    \
    }
#define FX_S16N(NW_) { if (scalar) { if (t2d) FX_S16(NW_, true, 32) else FX_S16(NW_, false, 32) }  \
                       else if (uneven) FX_S16(NW_, true, 16)                                       \
                       else { if (t2d) FX_S16(NW_, true, 0) else FX_S16(NW_, false, 0) } }
// float32 banks: the bank-direct walk, 64-bit row offsets, with (WR) or without the compact-snapshot write.
#define FX_DW_F(NW_) { if (wr) { if (t2d) FX_DWE(float, NW_, 2, true, true, false, true, true) else FX_DWE(float, NW_, 2, false, true, false, true, true) } \
                       else { if (t2d) FX_DWE(float, NW_, 2, true, true, false, false, true) else FX_DWE(float, NW_, 2, false, true, false, false, true) } }
    if (s16) {         // fx_dw_kernel<float, NW, 2, T2D, true, false, true, true, 0 / 16 / 32, bf16> (nw 8 / 16)
        TORCH_CHECK(nw == 8 || nw == 16, "fx_dw: bf16 compact rows of a float32 bank: 8 or 16 warps");
        if (nw == 8) FX_S16N(8) else FX_S16N(16)
    }
    else if (scalar) {      // fx_dw_kernel<E, NW, 2, T2D, true, false, WR, BIG, 32> (nw 8 / 16; float32: BIG)
        if (bf) { if (nw == 8) FX_DW_SB(8) else FX_DW_SB(16) }
        else { if (nw == 8) FX_DW_SF(8) else FX_DW_SF(16) }
    }
    else if (uneven) {      // fx_dw_kernel<E, NW, 2, true, true, false, WR, BIG, 16> (nw 8 / 16; float32: BIG)
        if (bf) { if (nw == 8) FX_DW_UB(8) else FX_DW_UB(16) }
        else { if (nw == 8) FX_DW_UF(8) else FX_DW_UF(16) }
    }
    else if (!bf) {  // fx_dw_kernel<float, NW, 2, T2D, true, false, WR, true> (nw 8 / 16)
        if (nw == 8) FX_DW_F(8)
        else FX_DW_F(16)
    }
    else if (w2) {   // fx_dw2_kernel<NW, WR, BIG, ALLON> (nw 8 / 16)
        if (nw == 8) FX_W2_N(8)
        else if (nw == 16) FX_W2_N(16)
        else TORCH_CHECK(false, "fx_dw: unsupported two-slice nw = ", nw);
    }
    else if (bankb == 3) {
        if (nw == 8) FX_DW_W(8)
        else if (nw == 16) FX_DW_W(16)
        else if (nw == 4) FX_DW_W(4)
        else TORCH_CHECK(false, "fx_dw: unsupported bank-direct compact-write nw = ", nw);
    }
    else {
        if (nw == 8) FX_DW_B(8)
        else if (nw == 16) FX_DW_B(16)
        else if (nw == 4) FX_DW_B(4)
        else TORCH_CHECK(false, "fx_dw: unsupported bank-direct nw = ", nw);
    }
#undef FX_S16N
#undef FX_S16
#undef FX_DW_SF
#undef FX_DW_SF2
#undef FX_DW_SB
#undef FX_DW_SB2
#undef FX_DWS
#undef FX_DW_UF
#undef FX_DW_UB
#undef FX_DWV
#undef FX_W2_N
#undef FX_W2_A
#undef FX_W2
#undef FX_DW_F
#undef FX_DW_W
#undef FX_DW_B
#undef FX_DW
#undef FX_DWE
    TORCH_CHECK(!portable || fx_portable_image(pl.fn), "fx_dw: a portable (A100 / RTX) plan, but this device runs the "
                "modern image of the kernel (build the extensions for the row's architecture)");
    return pl;
}

// One step of a plan on the current stream of the plan's device (the caller holds a CUDAGuard on it): the step check,
// then the launch with the plan's arguments (the kernel's parameter order) and ti. pdl = 1: programmatic stream
// serialization, i.e. the grid may launch before the previous kernel of the stream completes; the kernel waits for it
// (griddepcontrol.wait) before reading its output. The caller asserts that the kernel's pre-wait reads were written
// two or more launches back (the deterministic forward: the dual matmul precedes every step's DW).
void fx_launch_plan(const FxPlan& pl, int64_t ti, int64_t pdl) {
    TORCH_CHECK(ti >= 0 && ti <= pl.ti_max, "fx_dw: step ", ti, " out of range [0, ", pl.ti_max, "]");
    TORCH_CHECK(pdl == 0 || pdl == 1, "fx_dw: pdl must be 0 or 1");
    TORCH_CHECK(pdl == 0 || pl.portable == 0, "fx_dw: no programmatic launch on the portable side (A100 / RTX)");
    if (pl.blocks == 0) return;
    alignas(64) FxPlan a = pl;          // the argument copies the launch reads
    int tii = static_cast<int>(ti);     // ti <= ti_max <= INT32_MAX
    void* args[] = {(void*)&a.bank, (void*)&a.dv, (void*)&a.dv_step_rows, (void*)&a.krec, (void*)&a.tb,
                    (void*)&a.segb, (void*)&a.snapc, (void*)&a.sb_step, (void*)&tii, (void*)&a.g, (void*)&a.mrows,
                    (void*)&a.mtok};
    auto stream = at::cuda::getCurrentCUDAStream();
    if (pdl == 1 || a.cluster > 1) {
        cudaLaunchAttribute attr[2];
        unsigned na = 0;
        if (pdl == 1) {
            attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[na].val.programmaticStreamSerializationAllowed = 1;
            ++na;
        }
        if (a.cluster > 1) {   // quantized column-sliced walks: the CTAs of every slice of the same tiles (slice-fastest)
            attr[na].id = cudaLaunchAttributeClusterDimension;
            attr[na].val.clusterDim.x = static_cast<unsigned>(a.cluster);
            attr[na].val.clusterDim.y = 1;
            attr[na].val.clusterDim.z = 1;
            ++na;
        }
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(static_cast<unsigned>(a.blocks));
        cfg.blockDim = dim3(static_cast<unsigned>(a.threads));
        cfg.dynamicSmemBytes = static_cast<size_t>(a.smem);
        cfg.stream = stream.stream();
        cfg.attrs = attr;
        cfg.numAttrs = na;
        C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, a.fn, args));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    C10_CUDA_CHECK(cudaLaunchKernel(a.fn, dim3(static_cast<unsigned>(a.blocks)), dim3(static_cast<unsigned>(a.threads)),
                                    args, static_cast<size_t>(a.smem), stream.stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// CPU uint8 [sizeof(FxPlan)]: the DW's checks and launch arguments for every step of one factory call. The plan holds raw
// pointers into bank / dv / krec / tb / segb / snapc: the caller keeps those tensors alive and unresized while it is used.
at::Tensor fx_dw_plan(at::Tensor bank, at::Tensor dv, int64_t dv_step_rows, at::Tensor krec, at::Tensor tb,
                      at::Tensor segb, at::Tensor snapc, int64_t sb_step, int64_t P, int64_t C, int64_t K, int64_t nw,
                      int64_t nu, at::Tensor maps, int64_t bankb, int64_t quant, at::Tensor payload, at::Tensor scale,
                      int64_t portable, int64_t smem_cap) {
    const at::cuda::CUDAGuard guard(bank.device());
    const FxPlan pl = fx_plan_make(bank, dv, dv_step_rows, krec, tb, segb, snapc, sb_step, P, C, K, nw, nu, maps, bankb,
                                   quant, payload, scale, portable, smem_cap);
    at::Tensor out = at::empty({static_cast<int64_t>(sizeof(FxPlan))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(out.data_ptr<uint8_t>(), &pl, sizeof(FxPlan));
    return out;
}

// Admission of a quantized plan on the current device: clusters that can be co-resident (> 0: launchable).
int64_t fx_q_capacity(int64_t quant, int64_t w2, int64_t bf, int64_t nw, int64_t t2d, int64_t allon, int64_t cluster,
                      int64_t smem) {
    TORCH_CHECK(cluster >= 1 && cluster <= 16 && smem > 0 && smem <= 227 * 1024, "fx_q_capacity: cluster / smem");
    return cluster_capacity(fx_q_fn(quant, w2 != 0, bf != 0, nw, t2d != 0, allon != 0), cluster, static_cast<size_t>(smem),
                            static_cast<int>(nw) * 32);
}

void fx_dw_step(at::Tensor plan, int64_t ti, int64_t pdl) {
    TORCH_CHECK(plan.device().is_cpu() && plan.scalar_type() == at::kByte && plan.is_contiguous() &&
                plan.numel() == static_cast<int64_t>(sizeof(FxPlan)), "fx_dw_step: not an fx_dw_plan tensor");
    alignas(64) FxPlan pl;
    std::memcpy(&pl, plan.data_ptr<uint8_t>(), sizeof(FxPlan));
    TORCH_CHECK(pl.magic == kPlanMagic, "fx_dw_step: not an fx_dw_plan tensor (magic)");
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(pl.device));
    fx_launch_plan(pl, ti, pdl);
}

// ---- The deterministic forward's step loop in ONE call (eager host cost) ------------------------------------------
// For ti in [t0, t1): (1) the gather plan step (sdm_token_gather_dc's C++ entry point, address passed in), (2) the Triton
// dual matmul exactly as Triton 3.7's CudaLauncher issues it for this kernel (cuLaunchKernelEx; grid (grid_x, 1, 1) x
// num_ctas 1, block 32 * num_warps, the kernel's dynamic shared memory, no launch attributes since the kernel has no
// PDL / cooperative / cluster launch; parameters = its non-constexpr arguments in signature order + the null global and
// profile scratch pointers; only ti changes per step), (3) this module's DW plan step. Same kernels, same arguments,
// same order, same stream as the Python step loop: same launches, same bits. The descriptor is built and checked in
// Python (mover_fx_cuda.tri_desc) from the CompiledKernel Triton's launcher would use.
using LaunchKernelEx = CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**);
LaunchKernelEx launch_kernel_ex() {
    static LaunchKernelEx fn = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) h = dlopen("libcuda.so.1", RTLD_NOW);
        if (h != nullptr) fn = reinterpret_cast<LaunchKernelEx>(dlsym(h, "cuLaunchKernelEx"));
    });
    return fn;
}
constexpr int kTriMaxParams = 24;
constexpr uint64_t kTriMagic = 0x3154584644585446ull;
struct FxTriDesc {                   // Python packs it with struct "<QQiIIIii24i24Q" (checked against sizeof)
    uint64_t magic;
    uint64_t function;               // CUfunction (CompiledKernel.function)
    int32_t device;
    uint32_t grid_x;
    uint32_t block_x;                // 32 * num_warps
    uint32_t shared;                 // metadata.shared
    int32_t nparams;                 // kernel parameters incl. the two trailing scratch pointers
    int32_t ti_slot;                 // the 32-bit parameter that carries ti
    int32_t sizes[kTriMaxParams];    // 4 or 8 bytes
    uint64_t values[kTriMaxParams];  // little-endian values (pointers, ints; the scratch pointers 0)
};
using GatherLaunch = int64_t (*)(const void*, int64_t, int64_t, int64_t, int64_t, void*);

int64_t fx_tri_desc_size() { return static_cast<int64_t>(sizeof(FxTriDesc)); }

void fx_fwd_steps(at::Tensor gplan, int64_t gather_addr, at::Tensor tdesc, at::Tensor dwplan, int64_t t0, int64_t t1,
                  int64_t SC, int64_t C, int64_t tokens, int64_t pdl_gather, int64_t pdl_dw) {
    TORCH_CHECK(gplan.device().is_cpu() && gplan.scalar_type() == at::kByte && gplan.is_contiguous() && gplan.numel() > 0,
                "fx_fwd_steps: gather plan");
    TORCH_CHECK(gather_addr != 0, "fx_fwd_steps: gather entry point");
    TORCH_CHECK(tdesc.device().is_cpu() && tdesc.scalar_type() == at::kByte && tdesc.is_contiguous() &&
                tdesc.numel() == static_cast<int64_t>(sizeof(FxTriDesc)), "fx_fwd_steps: not a dual descriptor");
    TORCH_CHECK(dwplan.device().is_cpu() && dwplan.scalar_type() == at::kByte && dwplan.is_contiguous() &&
                dwplan.numel() == static_cast<int64_t>(sizeof(FxPlan)), "fx_fwd_steps: not an fx_dw_plan tensor");
    alignas(64) FxPlan dw;
    std::memcpy(&dw, dwplan.data_ptr<uint8_t>(), sizeof(FxPlan));
    TORCH_CHECK(dw.magic == kPlanMagic, "fx_fwd_steps: not an fx_dw_plan tensor (magic)");
    FxTriDesc td;
    std::memcpy(&td, tdesc.data_ptr<uint8_t>(), sizeof(FxTriDesc));
    TORCH_CHECK(td.magic == kTriMagic && td.function != 0 && td.nparams >= 2 && td.nparams <= kTriMaxParams &&
                td.ti_slot >= 0 && td.ti_slot < td.nparams - 2 && td.sizes[td.ti_slot] == 4 && td.grid_x >= 1 &&
                td.block_x >= 32 && td.block_x <= 1024 && td.block_x % 32 == 0, "fx_fwd_steps: bad dual descriptor");
    for (int i = 0; i < td.nparams; ++i) TORCH_CHECK(td.sizes[i] == 4 || td.sizes[i] == 8, "fx_fwd_steps: param size");
    TORCH_CHECK(td.device == dw.device, "fx_fwd_steps: the dual matmul and the DW plan are on different devices");
    TORCH_CHECK(t0 >= 0 && t0 <= t1 && t1 - 1 <= dw.ti_max && t1 - 1 <= std::numeric_limits<int32_t>::max() &&
                C >= 1 && SC >= 0 && tokens >= 0 && (pdl_gather == 0 || pdl_gather == 1) && (pdl_dw == 0 || pdl_dw == 1),
                "fx_fwd_steps: step range / arguments");
    TORCH_CHECK(t1 <= t0 || tokens > (t1 - 1) * C, "fx_fwd_steps: every step must hold valid tokens");
    LaunchKernelEx lk = launch_kernel_ex();
    TORCH_CHECK(lk != nullptr, "fx_fwd_steps: cuLaunchKernelEx not found in libcuda.so.1");
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dw.device));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const GatherLaunch gather = reinterpret_cast<GatherLaunch>(gather_addr);
    const uint8_t* gp = gplan.data_ptr<uint8_t>();
    uint64_t vals[kTriMaxParams];
    void* params[kTriMaxParams];
    std::memcpy(vals, td.values, sizeof(vals));
    for (int i = 0; i < td.nparams; ++i) params[i] = &vals[i];
    CUlaunchConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.gridDimX = td.grid_x;
    cfg.gridDimY = 1;
    cfg.gridDimZ = 1;
    cfg.blockDimX = td.block_x;
    cfg.blockDimY = 1;
    cfg.blockDimZ = 1;
    cfg.sharedMemBytes = td.shared;
    cfg.hStream = reinterpret_cast<CUstream>(stream);
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    for (int64_t ti = t0; ti < t1; ++ti) {
        const int64_t valid = std::min(C, tokens - ti * C);
        const int64_t rc = gather(gp, gplan.numel(), ti * SC, valid, (pdl_gather && ti > 0) ? 1 : 0,
                                  static_cast<void*>(stream));
        TORCH_CHECK(rc == 0, "fx_fwd_steps: gather step ", ti, " failed (", rc, ")");
        vals[td.ti_slot] = static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(ti)));
        const CUresult r = lk(&cfg, reinterpret_cast<CUfunction>(td.function), params, nullptr);
        TORCH_CHECK(r == CUDA_SUCCESS, "fx_fwd_steps: dual matmul step ", ti, " failed (CUresult ", static_cast<int>(r), ")");
        fx_launch_plan(dw, ti, pdl_dw);
    }
}
""".replace("@ROW_HELPERS@\n", ROW_HELPERS.lstrip("\n")).replace("@QUANT_HELPERS@\n", QUANT_HELPERS.lstrip("\n"))

_SMEM_MAX = 227 * 1024
_MAX_HEAD_DIM = 1 << 20   # = kMaxHeadDim (every row / element offset in the kernels is 64-bit)
# fp8 / int4 rows: heads above the old 16384 bound have no quantized writer (QT class T0: refused up front) until the
# quantized-tails bundle admits them with its own tests
_QUANT_MAX_HEAD_DIM = 16384
# Compact snapshot rows from which fx_maps leaves its snapc map zeroed (never read: the bank-direct walk only; its 32-bit
# row coordinates end at 2**31): 2**31. Tests set 0 (the zeroed map at small sizes: the same bits).
_WIDE_SNAP_ROWS = 2 ** 31


def pdl_ok(device):
    """PDL may be requested for launches on `device` (a torch.device) iff it is on the modern backend
    (_device.pdl_ok: sm_90 / 100 / 103): the portable device passes compile the griddepcontrol hooks away, so a
    programmatic launch is never requested elsewhere (RTX 50 included). Device only: never the bits."""
    return _device_pdl_ok(device)


def fwd_pdl_ok(device):
    """pdl_ok for the forward kernels (the DW here, token_gather_dc)."""
    return pdl_ok(device)


def gwr_pdl_ok(device):
    """pdl_ok for pa_dc_cuda's dc_gwr (its PDL build reads G through the coherent path after its wait)."""
    return pdl_ok(device)


@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_mover_fx",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("fx_maps", "fx_maps2", "fx_dw_plan", "fx_dw_step", "fx_q_capacity", "fx_tri_desc_size",
                   "fx_fwd_steps"),
        headers=headers(),
        extra_cuda_cflags=("-O3", "-lineinfo"),
        extra_ldflags=("-ldl",),
    )


def slice_width(head_dim) -> int:
    """Columns per slice (= the C++ slice_width): hd / nsl when that is a multiple of 4, else 128 (tails)."""
    nsl = -(-head_dim // 128)
    if head_dim % 4:                       # SCALAR (hd % 4 != 0): 128-wide slices, or one slice padded to 4 columns
        return 128 if nsl > 1 else -(-head_dim // 4) * 4
    if nsl > 1 and (head_dim % nsl or (head_dim // nsl) % 4):
        return 128
    return head_dim // nsl


def smem_bytes(chunk_size, head_dim, nw, nu, bankb=False, pf=False, w2=False, quant=0):
    nsl = -(-head_dim // 128)
    sw = slice_width(head_dim)
    tok = (chunk_size * sw * (2 if w2 else 1) * 4 + 127) // 128 * 128   # two-slice walk: token rows of a slice pair
    nur = 0 if bankb else nu                      # bank-direct: no rings
    pfb = 32 * sw * 2 if (bankb and pf) else 0    # prefetch buffer per warp (32 rows)
    qrows = 8 + nw * _TILE_ROWS * 8 if (quant in (1, 2) and nsl > 1) else 0   # the cluster form: {amax, scale} per group
    return tok + nw * nur * 8 * sw * 2 + nw * pfb + nw * 32 * 16 + 8 * (1 + nw * nur) + qrows


_TILE_ROWS = 128   # valid groups per warp tile of every fx tile set (kTileRows)


# Host cost: tensor maps are reused when both tensors' address, shape, strides and dtype, the chunk, the device and
# the encoder match (a map encodes only those). The cache never iterates (cleared when full). Device lookups are not
# cached.
_MAPS: dict = {}


def _smem_limit(device):
    """The device class's opt-in shared memory per block (never the bits)."""
    return _device_smem_cap(device)


def geometry(chunk_size, head_dim, nw, limit=_SMEM_MAX, bankb=False, quant=0):
    """(nw, 2) or None: the bank-direct walk (bankb True) has no ring units, so nu is a placeholder (2); None when nw is
    not 4 / 8 / 16, when its shared memory does not fit `limit`, or without bankb (only the bank-direct walk is
    launched). Shapes and the device's smem limit only."""
    if not bankb:
        return None   # only the bank-direct walk is launched
    if quant and nw not in (8, 16):
        return None
    return (nw, 2) if nw in (4, 8, 16) and smem_bytes(chunk_size, head_dim, nw, 2, True, quant=quant) <= limit else None


def _portable(device) -> bool:
    """`device` runs the portable side (a100 / rtx); False on the modern backend (raises outside the matrix)."""
    return _device_backend(device) != "modern"


def quant_form(device, head_dim, quant):
    """The kernels' QUANT for fp8 (1) / int4 (2) rows: quant (cluster form) or quant | 4 (all slices; always on a100 / rtx
    for column-sliced heads)."""
    if not quant:
        return 0
    if -(-head_dim // 128) > 1 and _portable(device):
        return quant | 4
    return quant
def rows16(bank) -> bool:
    """The head splits into equal 16-byte slice rows and 16-byte rows (tails do not)."""
    hd = bank.shape[-1]
    nsl = -(-hd // 128)
    es = bank.element_size()
    return hd % nsl == 0 and (hd // nsl * es) % 16 == 0 and (hd * es) % 16 == 0


def tail_supported(bank, chunk_size, quant="none") -> bool:
    """movers.tail_admits hook (tails): this DW runs the call's tail class with the bank-direct one-slice-per-warp walk,
    8 or 16 warps, on every backend. Unquantized snapshots; bf16 snapshots (float32 banks) are the same walk with bf16 stores."""
    if quant not in ("none", "bf16") or not bank.is_cuda or rows16(bank):
        return False
    if quant == "bf16" and bank.dtype != torch.float32:
        return False
    return any(supported(bank, chunk_size, nw, True) for nw in (8, 16))


def s16_supported(bank, chunk_size) -> bool:
    """snapshot_quant bf16 on a float32 bank: this DW's bank-direct WR walk (8 or 16 warps) writes the dense compact rows as
    bf16 (RN) at their dense ordinals."""
    return (bank.is_cuda and bank.dtype == torch.float32 and bank.dim() == 2
            and any(supported(bank, chunk_size, nw, True) for nw in (8, 16)))


def supported(bank, chunk_size, nw, bankb=False):
    """Shapes this DW runs (bf16 bank, or f32 bank on the bank-direct walk with 8 / 16 warps; every backend of
    _device.backend: modern, or the portable side on a100 / rtx; 16-byte slice rows, bank < 4 GiB unless bank-direct,
    smem fits the device's class); else mover_da2_cuda's kernel (modern) or phase2_kernels' (elsewhere)."""
    if not bank.is_cuda or bank.dtype not in (torch.bfloat16, torch.float32) or bank.dim() != 2:
        return False
    if bank.dtype == torch.float32 and not (bankb and nw in (8, 16)):
        return False
    try:
        _device_backend(bank.device)
    except RuntimeError:
        return False
    hd = bank.shape[-1]
    sw = slice_width(hd)
    es = bank.element_size()
    if sw % 4 or chunk_size > 256 or hd > _MAX_HEAD_DIM:
        return False
    if not rows16(bank) and not (bankb and nw in (8, 16)):   # tails (incl. hd % 4 != 0): the bank-direct walk, 8 / 16 warps
        return False
    # Banks of > 4 GiB only through the bank-direct walk (64-bit row offsets; rows < 2^31: 32-bit record slots)
    if (bank.numel() * es > 2 ** 32 and not bankb) or bank.shape[0] >= 2 ** 31:
        return False
    return geometry(chunk_size, hd, nw, _smem_limit(bank.device), bankb) is not None


def _warp_target(device=None) -> int:
    """Warps a DW step aims for: the largest power of two within one wave of 16 resident warps per SM (GB300: 2048);
    the geometry never changes bits."""
    return prev_pow2(16 * num_sms(device))


def tile_geometry(num_parallel, head_dim, n, chunk_size, nw_default, device=None, quant=0):
    """(nw, ppw): warps per CTA and positions per warp tile of this DW for a step of num_parallel chunks of n sorted
    positions. ppw 128 (Phase 1's tile length), halved to 64 / 32 while the step would launch fewer than
    _warp_target(device) warps (small steps: shorter per-warp chains); nw 16 when the chunk's token rows take >= 96 KiB
    of shared memory (more warps share them), else nw_default. Shapes only, speed only (bits do not depend on the
    tiles: every tile is cut at FIRST boundaries and each group is summed by one warp in sorted order)."""
    nsl = -(-head_dim // 128)
    sw = head_dim // nsl
    ppw = 128
    target = _warp_target(device)
    # all-slices quant form: tiles down to 8 positions, 1 on H100; any FIRST-snapped tile set: same bits
    nsl_w, ppw_min = (1, 1 if _sm90(device) else 8) if quant_form(device, head_dim, quant) >= 5 else (nsl, 32)
    while ppw > ppw_min and num_parallel * nsl_w * -(-n // ppw) < target:
        ppw //= 2
    if _portable(device):   # A100 / RTX: 16 warps when two 8-warp CTAs do not fit one SM (the same 16 warps per SM)
        nw = 16 if 2 * (smem_bytes(chunk_size, head_dim, 8, 2, True) + RESERVED_SMEM) > _device_smem_per_sm(device) \
            else nw_default
    else:
        nw = 16 if chunk_size * sw * 4 >= 96 * 1024 else nw_default
    return nw, ppw


def admits(bank, chunk_size):
    """Off the modern backend (a100 / rtx), the forward's DW admission (movers.dw_admits): the bank-direct walk with 8 warps
    takes calls on ``bank`` with chunks of ``chunk_size`` tokens (16-byte slice rows; tails: movers.tail_admits)."""
    return rows16(bank) and supported(bank, chunk_size, 8, True)


_TRI_FMT = "<QQiIIIii24i24Q"
_TRI_MAGIC = 0x3154584644585446


def tri_desc(ck, arg_names, args, grid0, varying="ti"):
    """The launch descriptor (CPU uint8, FxTriDesc) of a compiled Triton kernel `ck` for the argument tuple
        `args` (JITFunction.run's bound_args, in arg_names order) and grid (grid0, 1, 1), as Triton 3.7's CudaLauncher
        would launch it; `varying` names the one 32-bit int argument the fused loop sets per step. None when the launch is
        outside what fx_fwd_steps replicates (cluster / cooperative / PDL launch, scratch buffers, tensor descriptors,
        other argument types, too many parameters, a non-CUDA or unloaded kernel, a Triton other than 3.7.x)."""
    import struct
    import triton
    if not str(getattr(triton, "__version__", "")).startswith("3.7."):
        return None
    md = ck.metadata
    if (getattr(md, "num_ctas", 1) != 1 or getattr(md, "launch_cooperative_grid", False)
            or getattr(md, "launch_pdl", False) or getattr(md, "global_scratch_size", 0)
            or getattr(md, "profile_scratch_size", 0) or getattr(md, "tensordesc_meta", None)):
        return None
    sig = list(ck.src.signature.items())
    if len(sig) != len(arg_names) or len(args) != len(arg_names):
        return None
    sizes, values, slot = [], [], -1
    for (key, ty), name, val in zip(sig, arg_names, args):
        if isinstance(key, str) and key != name:
            return None
        if ty == "constexpr":
            continue
        if not isinstance(ty, str):
            return None
        if ty.startswith("*"):
            if not (isinstance(val, torch.Tensor) and val.is_cuda):
                return None
            sizes.append(8)
            values.append(val.data_ptr())
        elif ty in ("i32", "i64"):
            if not isinstance(val, int) or isinstance(val, bool):
                return None
            lim = 2 ** 31 if ty == "i32" else 2 ** 63
            if not -lim <= val < lim:
                return None
            sizes.append(4 if ty == "i32" else 8)
            values.append(val & (2 ** (32 if ty == "i32" else 64) - 1))
            if name == varying:
                if ty != "i32":
                    return None
                slot = len(sizes) - 1
        else:
            return None
    sizes += [8, 8]                       # global / profile scratch pointers (both null: sizes are 0)
    values += [0, 0]
    if slot < 0 or len(sizes) > 24 or not ck.function:
        return None
    sizes += [0] * (24 - len(sizes))
    values += [0] * (24 - len(values))
    block = 32 * md.num_warps
    raw = struct.pack(_TRI_FMT, _TRI_MAGIC, ck.function, torch.cuda.current_device(), grid0, block, md.shared,
                      len([z for z in sizes if z]), slot, *sizes, *values)
    if len(raw) != _module().fx_tri_desc_size():
        return None
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


_W2_PPW = (16, 32, 64)      # two-slice walk: positions per warp tile (half the one-slice walk's 32 / 64 / 128)
STATS = {"fx_dw2_wave": 0}  # two-slice DW factory calls whose grid _w1_ppw enlarged (counted only when K grows)


def _dw2_wave_ctas(device=None) -> int:
    """Resident 8-warp two-slice DW CTAs of one wave: two per SM (GB300: 304; speed only)."""
    return 2 * num_sms(device)


def tile_geometry_w2(num_parallel, head_dim, n, chunk_size, limit=_SMEM_MAX, device=None, quant=0):
    """Two column slices per warp (fx_dw2_kernel): (nw, ppw) or None when the shape does not take it (an odd number of
    slices, i.e. head_dim <= 128 or 3 / 5 / 7 slices; the token rows of a slice pair do not fit). nw 8 while two
    8-warp CTAs fit an SM's shared memory (<= 112 KiB each), else 16 (one CTA, the same 16 warps per SM). ppw 64,
    halved to 32 / 16 while the step would launch fewer than _warp_target(device) warps (the one-slice rule's warps at
    twice the columns per warp), then _w1_ppw (fill one resident wave). Shapes only, speed only: bits do not depend on
    the tiles or the column split."""
    nsl = -(-head_dim // 128)
    if nsl < 2 or nsl % 2 or head_dim % nsl:
        return None
    pairs = nsl // 2
    qf = quant_form(device, head_dim, quant)     # 1 / 2: the cluster form; 5 / 6: all slices (no cluster, no smem rows)
    # two 8-warp CTAs per SM (modern: <= 112 KiB each; A100 / RTX: the residency rule of tile_geometry), else 16 warps
    two8 = (2 * (smem_bytes(chunk_size, head_dim, 8, 2, True, w2=True) + RESERVED_SMEM) <= _device_smem_per_sm(device)
            if _portable(device) else smem_bytes(chunk_size, head_dim, 8, 2, True, w2=True, quant=qf) <= 112 * 1024)
    if two8 and smem_bytes(chunk_size, head_dim, 8, 2, True, w2=True, quant=qf) <= limit:   # within limit
        nw = 8
    elif smem_bytes(chunk_size, head_dim, 16, 2, True, w2=True, quant=qf) <= limit:
        nw = 16
    else:
        return None
    if qf in (1, 2) and not q_fits(device, quant, True, True, nw, True, head_dim // nsl == 128, pairs,
                                   smem_bytes(chunk_size, head_dim, nw, 2, True, w2=True, quant=qf)):
        return None
    # all-slices form: tiles down to 1 position on H100; any FIRST-snapped tile set: same bits
    units, ppw_min = (num_parallel, 1 if _sm90(device) else 8) if qf >= 5 else (num_parallel * pairs, _W2_PPW[0])
    ppw = _W2_PPW[-1]
    target = _warp_target(device)
    while ppw > ppw_min and units * -(-n // ppw) < target:
        ppw //= 2
    ppw = _w1_ppw(units, n, nw, ppw, device)
    return nw, ppw


def _w1_ppw(units, n, nw, ppw, device=None):
    """Wave fill: when the two-slice DW's grid (units = chunks x slice pairs, times K tiles per unit) fills at least half
    but not all of ONE resident wave of CTAs (_dw2_wave_ctas(device) for 8-warp CTAs, half for 16-warp CTAs), raise K to
    wave // units (positions per warp tile ceil(n / (nw K))): every resident slot gets a CTA and each warp walks a
    shorter tile, still one wave. Shapes only, speed only: any monotone FIRST-snapped tile set gives the same bits (tile
    bounds snap to group FIRSTs, so every group is summed whole by one warp in sorted order). The caller's K = ceil(n /
    (nw ppw)) <= wave // units, so the grid stays within one wave. Counted in STATS["fx_dw2_wave"] per factory call
    whose K this raised; when ceil rounding would leave K unchanged (e.g. n <= nw), the caller's ppw is returned and
    nothing is counted."""
    if n <= 0 or units <= 0:
        return ppw
    wave_ctas = _dw2_wave_ctas(device)
    wave = wave_ctas if nw <= 8 else wave_ctas // 2
    k_now = max(1, -(-n // (nw * ppw)))
    if not wave <= 2 * units * k_now:          # below half a wave (tiny grids): unchanged tiles
        return ppw
    k_wave = wave // units
    if k_wave <= k_now:
        return ppw
    ppw_w = max(1, -(-n // (nw * k_wave)))
    if -(-n // (nw * ppw_w)) <= k_now:          # rounding leaves K unchanged: unchanged tiles, not counted
        return ppw
    STATS["fx_dw2_wave"] = STATS.get("fx_dw2_wave", 0) + 1
    return ppw_w


_QFITS: dict = {}


def q_capacity(device, quant, w2, bf, nw, t2d, allon, cluster, smem):
    """Co-resident clusters of a quantized plan on `device` (cudaOccupancyMaxActiveClusters; 0: none), cached."""
    idx = torch.device(device).index if device is not None else None
    idx = torch.cuda.current_device() if idx is None else idx          # the device the query runs on
    key = (idx, quant, w2, bf, nw, t2d, allon, cluster, smem)
    n = _QFITS.get(key)
    if n is None:
        n = 0
        if 1 < cluster <= 16:
            try:
                with torch.cuda.device(idx):
                    n = int(_module().fx_q_capacity(quant, int(w2), int(bf), nw, int(t2d), int(allon), cluster, smem))
            except RuntimeError:
                n = 0
        _QFITS[key] = n
    return n


def q_fits(device, quant, w2, bf, nw, t2d, allon, cluster, smem):
    """A quantized plan's clusters fit `device` (no cluster: always)."""
    return cluster <= 1 or q_capacity(device, quant, w2, bf, nw, t2d, allon, cluster, smem) > 0


def quant_k(bank, chunk_size, quant, w2, nw, num_parallel, n, K):
    """Tiles per chunk of a quantized cluster plan: K, lowered so that the P * K clusters fit one wave when they would
    overflow it by less than one (speed only: any FIRST-snapped tile set gives the same bits)."""
    hd = bank.shape[-1]
    nsl = -(-hd // 128)
    cluster = nsl // 2 if w2 else nsl
    if cluster <= 1 or quant_form(bank.device, hd, quant) >= 5:   # (the all-slices form: no clusters)
        return K
    cap = q_capacity(bank.device, quant, w2, bank.dtype == torch.bfloat16, nw, True, hd // nsl == 128, cluster,
                     smem_bytes(chunk_size, hd, nw, 2, True, w2=w2, quant=quant))
    k_min = max(1, -(-n // (nw * _TILE_ROWS)))
    if num_parallel * K <= cap or num_parallel * K > 2 * cap or cap // num_parallel < k_min:
        return K
    STATS["fx_dw_q_wave"] = STATS.get("fx_dw_q_wave", 0) + 1
    return cap // num_parallel


def quant_geometry(bank, chunk_size, nw, quant):
    """The one-slice-per-warp quantized walk with nw warps fits (shared memory; column slices: one cluster of <= 16)."""
    hd = bank.shape[-1]
    nsl = -(-hd // 128)
    qf = quant_form(bank.device, hd, quant)
    if qf >= 5:          # the all-slices form: no cluster, no {amax, scale} rows, any number of slices
        return geometry(chunk_size, hd, nw, _smem_limit(bank.device), True, qf) is not None
    if nsl > 16 or geometry(chunk_size, hd, nw, _smem_limit(bank.device), True, quant) is None:
        return False
    return q_fits(bank.device, quant, False, bank.dtype == torch.bfloat16, nw, nsl > 1, hd // nsl == 128, nsl,
                  smem_bytes(chunk_size, hd, nw, 2, True, quant=quant))


def _quant_form_fits(bank, chunk_size, q):
    """A quantized plan of form q (1 / 2 cluster, 5 / 6 all slices) fits, in mover_da2_cuda.make_dw's order: the two-slice
    walk, else the one-slice walk with 8 or 16 warps."""
    hd = bank.shape[-1]
    if (bank.dtype == torch.bfloat16 and supported(bank, chunk_size, 8, True)
            and tile_geometry_w2(1, hd, 1, chunk_size, _smem_limit(bank.device), bank.device, q) is not None):
        return True
    return any(supported(bank, chunk_size, nw, True) and quant_geometry(bank, chunk_size, nw, q) for nw in (8, 16))


def quant_choice(bank, chunk_size, quant):
    """The kernels' quantized form for fp8 (1) / int4 (2) rows, 0 when none fits (the op then refuses the call): the
    cluster form, else (modern) the all-slices form (quant | 4: same rows, bits and launch)."""
    if quant not in (1, 2) or not bank.is_cuda or not rows16(bank):   # quantized tails: not taken
        return 0
    if bank.shape[-1] > _QUANT_MAX_HEAD_DIM:      # (QT class T0: heads above the old bound, with its own tests)
        return 0
    if _quant_form_fits(bank, chunk_size, quant):
        return quant
    if -(-bank.shape[-1] // 128) > 1 and not _portable(bank.device) and _quant_form_fits(bank, chunk_size, quant | 4):
        return quant | 4
    return 0


def quant_supported(bank, chunk_size, quant):
    """fp8 (1) / int4 (2) compact rows from this DW (= mover_da2_cuda.make_dw's choice, quant_choice)."""
    return quant_choice(bank, chunk_size, quant) != 0


_NOQ = torch.empty(0, dtype=torch.uint8)   # placeholder payload / scale of unquantized plans (never read)


def make_dw_fx(bank, delta_v, krec, tb, segb, snapc, *, chunk_size, num_parallel, dv_rows, sb_step, nw, K,
               bankb=False, snapw=False, w2=False, quant=0, payload=None, scale=None):
    """Step launcher of this DW over mover_da2_cuda's records / tiles / segb and a dense snapc ([rows, hd] view): row
    (ti * sb_step + p) * n + g of chunk p's g-th valid group (sb_step = P: the saving forward's every-step snapshot;
    0: a one-step buffer, the no-grad forward). bankb (required): base rows from bank[slot]; snapc / segb are not read
    (still shape-checked; snapc also anchors the column-sliced token-row map). snapw (sb_step = P only): the DW writes
    those dense rows (each valid group's pre-write bank row, raw) at its segb ordinal, so the forward's gather need not;
    segb must then be the tiles' valid-group prefix. quant 1 (fp8) / 2 (int4), sb_step = P, not snapw: the DW writes those
    rows quantized instead, to payload ([N * n, hd] or [N * n, hd / 2] one-byte rows) and scale (f32 [N * n])."""
    hd = bank.shape[-1]
    limit = _smem_limit(bank.device)
    quant = quant_form(bank.device, hd, quant) if quant in (1, 2) else quant   # the kernels' QUANT (5 / 6: all slices)
    nw, nu = (nw, 2) if w2 else geometry(chunk_size, hd, nw, limit, bankb, quant)
    # 64-bit row offsets (bank-direct only, no prefetch): bf16 banks of > 4 GiB, every float32 bank
    big = bank.dtype == torch.float32 or bank.numel() * 2 > 2 ** 32
    if bank.dtype == torch.float32:
        STATS["fx_dw_f32"] = STATS.get("fx_dw_f32", 0) + 1   # float32 DW factory calls (engagement counter)
    snap2 = snapc.view(-1, hd)
    mod = _module()
    dv2 = delta_v.view(-1, hd)
    # The portable side (a100 / rtx) reads the token rows through delta_v itself: no tensor maps are encoded there.
    portable = _portable(bank.device)
    if portable:
        maps = _NOQ
    else:
        encode = mod.fx_maps2 if w2 else mod.fx_maps     # two-slice walk: the token-row box of a slice pair
        wide = 0 if w2 else min(max(int(_WIDE_SNAP_ROWS), 0), 2 ** 31)   # fx_maps: the snapc map's row threshold
        mkey = (encode, snap2.data_ptr(), tuple(snap2.shape), tuple(snap2.stride()), snap2.dtype, dv2.data_ptr(),
                tuple(dv2.shape), tuple(dv2.stride()), dv2.dtype, chunk_size, snap2.get_device(), wide)
        maps = _MAPS.get(mkey)
        if maps is None:
            # empty for one-slice heads
            maps = encode(snap2, dv2, chunk_size) if w2 else encode(snap2, dv2, chunk_size, wide)
            if len(_MAPS) > 64:
                _MAPS.clear()
            _MAPS[mkey] = maps

    # DW mode: 1 bank-direct, 3 bank-direct + compact-snapshot write; 4 / 5 = 1 / 3 with two column slices per warp
    # (fx_dw2_kernel)
    bk = 3 if snapw else 1
    if w2:
        bk = 5 if snapw else 4

    # The launch checks run once per factory call; each step passes only the plan and ti (same kernel, same arguments,
    # same launch). The plan holds raw pointers: the launcher keeps every tensor it names alive.
    qpay = payload.view(-1, hd if (quant & 3) == 1 else hd // 2) if quant else _NOQ
    plan = mod.fx_dw_plan(bank, delta_v, dv_rows, krec, tb, segb, snap2, sb_step, num_parallel, chunk_size, K, nw,
                          nu, maps, bk, quant, qpay, scale.view(-1) if quant else _NOQ, int(portable), limit)
    if quant:
        STATS["fx_dw_q"] = STATS.get("fx_dw_q", 0) + 1   # quantized-row DW factory calls (reachability counter)
    step = mod.fx_dw_step
    pdl = [0]

    def launch(ti: int) -> None:
        step(plan, ti, pdl[0])

    def set_pdl(on: bool) -> None:
        # Programmatic dependent launches for every step of this factory (sm_90+ only; else a no-op). The CALLER asserts
        # the rule the kernel's pre-wait code relies on: the launch right before each step's DW is that step's dual matmul,
        # launched after the step's gather (the deterministic forward).
        pdl[0] = 1 if (on and fwd_pdl_ok(bank.device)) else 0
    launch.set_pdl = set_pdl
    launch.fx_plan = plan              # the fused deterministic step loop launches this plan (with fx_pdl[0])
    launch.fx_pdl = pdl
    launch._fx_keep = (bank, delta_v, krec, tb, segb, snap2, maps, plan, qpay, scale)

    nsl = -(-hd // 128)
    launch.geom = dict(nw=nw, nu=0 if bankb else nu, K=K, fx=True, bankb=bankb, pf=False, snapw=snapw, big=big or bool(quant),
                       w2=w2, quant=quant, cluster=(nsl // 2 if w2 else nsl) if quant in (1, 2) and nsl > 1 else 0,
                       all_slices=quant >= 5)
    return launch
