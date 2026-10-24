// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Device building blocks shared by the Phase-1 forward and backward kernels.
//
// Sorted layout
// -------------
// A chunk holds C tokens. Token t writes W slots (K entries) and reads R slots (Q entries).
// Its n = C * (W + R) entries are enumerated e = t * (W + R) + j: K entry (t, j) for
// j < W, then Q entry (t, j - W). Each chunk is one row of the sorted arrays [N, row],
// row = n rounded up to a multiple of 16 (entries [n, row) are unused). The row holds the
// chunk's entries stably sorted by key = slot id relative to the chunk's stream, so each
// group (the maximal run of one slot) is ordered by (token, K before Q, lane).
// Entries of tokens at or past the chunk's valid length are invalid: they take the
// largest key, sort after every valid entry and are each a group of their own.
//
// Payload of a sorted entry:
//   bits  0-7   lane (K: 0..W-1, Q: 0..R-1)
//   bit   8     Q entry
//   bits  9-16  token
//   bit  17     invalid
//   bit  30     last K entry of its group (set by the forward scan)
//   bit  31     first entry of its group (set by the forward scan)
//
// x is the log-decay an entry contributes: the gate of its token (of its lane with
// per-write gates) on the first K entry of each (slot, token); zero on Q entries, invalid
// entries and repeated K entries of one token (repeats break the distinct-slot contract
// and are counted). An entry's log-cumsum is the inclusive sum of x over its group
// prefix, so a Q entry sees the K entries of its own token. Log-cumsums are compensated
// sums: the decay between two entries of a group is the exp of their difference, and the
// compensation keeps its error at the rounding of the difference rather than of the
// (possibly much larger) cumsums.
#pragma once

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>

namespace sdm_p1 {

constexpr uint32_t kLaneMask = 0xFFu;
constexpr uint32_t kIsQ = 1u << 8;
constexpr int kTokenShift = 9;
constexpr uint32_t kInvalid = 1u << 17;
constexpr uint32_t kLastK = 1u << 30;
constexpr uint32_t kHead = 1u << 31;
constexpr int kRowAlign = 16;
constexpr float kLog2e = 1.4426950408889634f;

// A thread's consecutive row entries, moved with 16-byte accesses. Rows are padded to a
// multiple of kRowAlign entries, so a thread's items never straddle a row end.
template <typename T, int ITEMS>
__device__ __forceinline__ void load_items(const T* src, T (&dst)[ITEMS]) {
    constexpr int kVecs = sizeof(dst) / sizeof(uint4);
    uint4 v[kVecs];
#pragma unroll
    for (int i = 0; i < kVecs; ++i) v[i] = reinterpret_cast<const uint4*>(src)[i];
    memcpy(dst, v, sizeof(dst));
}

template <typename T, int ITEMS>
__device__ __forceinline__ void store_items(T* dst, const T (&src)[ITEMS]) {
    constexpr int kVecs = sizeof(src) / sizeof(uint4);
    uint4 v[kVecs];
    memcpy(v, src, sizeof(src));
#pragma unroll
    for (int i = 0; i < kVecs; ++i) reinterpret_cast<uint4*>(dst)[i] = v[i];
}

__device__ __forceinline__ int lane_of(uint32_t p) { return p & kLaneMask; }
__device__ __forceinline__ int token_of(uint32_t p) { return (p >> kTokenShift) & 0xFF; }
__device__ __forceinline__ bool is_head(uint32_t p) { return p & kHead; }
__device__ __forceinline__ bool is_k(uint32_t p) { return !(p & (kIsQ | kInvalid)); }
__device__ __forceinline__ bool is_q(uint32_t p) { return (p & (kIsQ | kInvalid)) == kIsQ; }

// A K entry that repeats the slot of an earlier K entry of the same token. Such entries
// are adjacent in sorted order, directly after the token's first K entry of that slot.
__device__ __forceinline__ bool repeats_k(uint32_t prev, uint32_t cur) {
    return !is_head(cur) && is_k(cur) && is_k(prev) && token_of(prev) == token_of(cur);
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Segmented-scan element: a compensated sum of x (value `x + c`, where c collects the
// rounding error of every addition) and a K count in bits 0-30; bit 31 starts a segment.
// Compensation matters because exponentiating a log-sum turns its absolute error into a
// relative one.
struct SegSum {
    float x;
    float c;
    uint32_t kf;
};

__device__ __forceinline__ SegSum seg_combine(SegSum a, SegSum b) {
    if (b.kf & kHead) return b;
    const float s = a.x + b.x;
    const float bb = s - a.x;
    const float err = (a.x - (s - bb)) + (b.x - bb);
    return {s, a.c + b.c + err, a.kf + b.kf};
}

__device__ __forceinline__ float seg_value(SegSum s) { return s.x + s.c; }
__device__ __forceinline__ uint32_t seg_count(SegSum s) { return s.kf & ~kHead; }

template <bool REVERSE>
__device__ __forceinline__ SegSum shfl_prev(SegSum v, int delta) {
    if (REVERSE) {
        return {__shfl_down_sync(0xFFFFFFFFu, v.x, delta),
                __shfl_down_sync(0xFFFFFFFFu, v.c, delta),
                __shfl_down_sync(0xFFFFFFFFu, v.kf, delta)};
    }
    return {__shfl_up_sync(0xFFFFFFFFu, v.x, delta), __shfl_up_sync(0xFFFFFFFFu, v.c, delta),
            __shfl_up_sync(0xFFFFFFFFu, v.kf, delta)};
}

// Shared state of block_scan_exclusive (blocks of up to 1024 threads).
struct BlockScanSmem {
    SegSum warp_total[32];
    SegSum warp_prefix[32];
    SegSum block_total;
};

// Segmented scan across a block. Scan order runs over threads in increasing index, or in
// decreasing index when REVERSE, and each thread passes the combined value of its own
// items in that order. Returns the combination of everything before the thread, starting
// from `carry` (earlier tiles), and advances `carry` past this tile. Every thread of the
// block must call it.
template <int THREADS, bool REVERSE>
__device__ SegSum block_scan_exclusive(SegSum own, SegSum& carry, BlockScanSmem& smem) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rank = REVERSE ? 31 - lane : lane;
    SegSum inclusive = own;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const SegSum other = shfl_prev<REVERSE>(inclusive, d);
        if (rank >= d) inclusive = seg_combine(other, inclusive);
    }
    const SegSum before = shfl_prev<REVERSE>(inclusive, 1);
    if (rank == 31) smem.warp_total[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        // Lane l scans the totals of the warps in scan order, seeded with the carry.
        const int w = REVERSE ? kWarps - 1 - lane : lane;
        SegSum v = lane < kWarps ? smem.warp_total[w] : SegSum{0.0f, 0.0f, 0u};
#pragma unroll
        for (int d = 1; d < kWarps; d <<= 1) {
            const SegSum other = shfl_prev<false>(v, d);
            if (lane >= d) v = seg_combine(other, v);
        }
        const SegSum earlier = shfl_prev<false>(v, 1);
        if (lane < kWarps) smem.warp_prefix[w] = lane == 0 ? carry : seg_combine(carry, earlier);
        if (lane == kWarps - 1) smem.block_total = seg_combine(carry, v);
    }
    __syncthreads();
    carry = smem.block_total;
    const SegSum prefix = smem.warp_prefix[warp];
    return rank == 0 ? prefix : seg_combine(prefix, before);
}

// The log-cumsum of `s` in log2 units, as a compensated pair hi + lo whose low part also
// carries the rounding error of the product.
__device__ __forceinline__ void log2_parts(SegSum s, float& hi, float& lo) {
    hi = s.x * kLog2e;
    lo = fmaf(s.c, kLog2e, fmaf(s.x, kLog2e, -hi));
}

// log2 of the decay between entries p < r of one group, exp(sum of x over (p, r]), from
// their log-cumsums in log2 units held as compensated pairs hi + lo.
__device__ __forceinline__ float log2_decay(float hi_r, float lo_r, float hi_p, float lo_p) {
    return (hi_r - hi_p) + (lo_r - lo_p);
}

}  // namespace sdm_p1
