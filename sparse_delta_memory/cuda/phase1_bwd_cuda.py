# Copyright (c) Meta Platforms, Inc. and affiliates.
"""phase1_bwd_cuda: the backward of Phase 1 (phase1_cuda): gradients of k_val, q_val and the gates.

It reads the forward's sorted payload, so it neither sorts nor searches. With D(p, r) the
decay between entries p < r of one group and i, j the tokens of the two entries:

  d_row[r]   = sum over earlier K p of d_a[i, j] v_p D(p, r)      (K entry r)
  d_col[r]   = sum over later K s of d_a[j, i] v_s D(r, s)        (K entry r)
  d_kvqk[r]  = sum over later Q s of d_qk[j, i] v_s D(r, s)       (K entry r)
  d_qv[r]    = sum over earlier K p of d_qk[i, j] v_p D(p, r)     (Q entry r)

Every sum has one writer and a fixed order, so the backward is deterministic. The
per-entry chain then gives the value gradients and the adjoints of the log-cumsums and of
rex, and two segmented scans turn those into the gradient of each entry's log-decay x:
the adjoints of the log-cumsums at or after it (log-cumsums are inclusive sums of x) plus
the adjoints of rex before it (rex is an exclusive reverse sum).

Kernels, one block per chunk: the pair kernel recomputes the log-cumsums and pulls the
sums (one block per product when the two triangles do not fit together), and the gate
kernel runs the chain and the scans. Launches use the current stream, allocate through
the caching allocator and do not synchronize, so they can be captured in a CUDA graph.

The source also holds the deterministic-mode variant of both kernels (preprocessor variant P1_BWD_DET): the same sums
in the same order, hence the same bits, with more latency hidden: the recompute fetches the entries' values before its
block scan, and the pulls broadcast the before / after batches of a window's first and last group from shared memory
(one 16-byte load per entry instead of four shuffles) when that buffer costs no resident block. The extension compiles
the source twice, without and with P1_BWD_DET, as two units of one module; phase1_backward_det takes the variant at the
chunk sizes where it is faster."""

from __future__ import annotations

import functools
import os

import torch

from ._build import load_cuda_extension
from ._device import smem_cap

__all__ = ["phase1_backward", "phase1_backward_det"]

_HEADER = "phase1_common.cuh"

_ARGS = """(
    at::Tensor payload, at::Tensor log_k, at::Tensor log_q, at::Tensor rex, at::Tensor k_val,
    at::Tensor q_val, at::Tensor g, at::Tensor d_a, at::Tensor d_qk, at::Tensor d_k_eff,
    at::Tensor d_k_pd, at::Tensor d_q_w, int64_t smem_cap);"""
_CPP_SOURCE = "std::vector<at::Tensor> p1_backward" + _ARGS + "\nstd::vector<at::Tensor> p1_backward_det" + _ARGS + "\n"

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "phase1_common.cuh"

using namespace sdm_p1;

constexpr int kThreads = 512;
constexpr int kItems = 8;
constexpr int kTile = kThreads * kItems;
static_assert(kRowAlign % kItems == 0, "row padding");
// Shared memory of a block, so that two fit on an SM.
constexpr size_t kMaxSmemBytes = 112 * 1024;
constexpr uint32_t kFull = 0xFFFFFFFFu;

// The pair kernel keeps a copy of the payload whose bits 18-26 hold the number of K
// entries carrying an x at or before the entry in its group.
constexpr int kCountShift = 18;
__device__ __forceinline__ int count_of(uint32_t p) { return (p >> kCountShift) & 0x1FF; }

// Position of d_a[i, j], j < i, in a packed strictly lower triangle, and of d_qk[i, j],
// j <= i, in a packed lower triangle.
__device__ __forceinline__ int strict_at(int i, int j) { return i * (i - 1) / 2 + j; }
__device__ __forceinline__ int lower_at(int i, int j) { return i * (i + 1) / 2 + j; }

// 2^x, flushing subnormal results to zero: such decays are far below the precision of
// the sums they enter.
__device__ __forceinline__ float exp2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Compensated running sum (value s + c): a pulled sum can have hundreds of terms whose
// cancellation the gate scans would otherwise amplify. Terms arrive as plain partial sums
// of at most 32.
struct CompSum {
    float s = 0.0f;
    float c = 0.0f;

    __device__ __forceinline__ void add(float t) {
        const float sum = s + t;
        const float bb = sum - s;
        c += (s - (sum - bb)) + (t - bb);
        s = sum;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};

// Sorted-order rows [N, row] of the pulled sums: `grad`, their part of the value gradient,
// and `log`, their part of the log-cumsum adjoint (divided by the value). One plane per
// block kind of the pair kernel.
struct Plane {
    float* grad;
    float* log;
};
struct Planes {
    Plane p[2];
};

// A sorted entry as the pulls read it: payload, log-cumsum in log2 units (hi + lo), value.
#if defined(P1_BWD_DET)
#define Entry EntryDet    // the deterministic-mode unit's own types (both units are linked into one extension)
#define Puller PullerDet
struct __align__(16) Entry {
#else
struct Entry {
#endif
    uint32_t pay;
    float hi;
    float lo;
    float v;
};

__device__ __forceinline__ Entry shfl_entry(const Entry& e, int src) {
    return {__shfl_sync(kFull, e.pay, src), __shfl_sync(kFull, e.hi, src),
            __shfl_sync(kFull, e.lo, src), __shfl_sync(kFull, e.v, src)};
}
__device__ __forceinline__ Entry shfl_entry_up(const Entry& e, int d) {
    return {__shfl_up_sync(kFull, e.pay, d), __shfl_up_sync(kFull, e.hi, d),
            __shfl_up_sync(kFull, e.lo, d), __shfl_up_sync(kFull, e.v, d)};
}

// One entry's view of its group, pairing it with earlier K entries (its row of d_a or d_qk
// at `row`) and, for a K entry, with later entries (column i of d_a or d_qk). `tri` holds
// the packed d_a triangle, then the d_qk one at `qk`. Weights are d * decay, or 0 for a
// pair the block does not pull. A pair with no x between its entries (a Q entry and the
// last K entry of its slot at or before its token) has decay exactly 1 whatever the gates,
// so its term enters the value gradient but not the log-cumsum adjoint, where it would
// cancel against itself.
struct Puller {
    Entry own;
    const float* tri;
    int qk;
    int row;           // offset of this entry's row, when `pulls_back`
    bool pulls_back;   // a K entry pulling from d_a, or a Q entry pulling from d_qk
    bool pulls_later;  // a K entry
    bool do_a, do_qk;

    __device__ __forceinline__ float weight_earlier(const Entry& e, bool pair) const {
        const int j = token_of(e.pay);
        if (!pair || !pulls_back || !is_k(e.pay) || (!is_q(own.pay) && j == token_of(own.pay))) {
            return 0.0f;
        }
        return tri[row + j] * exp2_ftz(log2_decay(own.hi, own.lo, e.hi, e.lo));
    }

    __device__ __forceinline__ float weight_later(const Entry& e, bool pair) const {
        const int i = token_of(own.pay), j = token_of(e.pay);
        const bool q = is_q(e.pay);
        if (!pair || !pulls_later || (q ? !do_qk : !do_a || j == i)) return 0.0f;
        return tri[q ? qk + lower_at(j, i) : strict_at(j, i)] *
               exp2_ftz(log2_decay(e.hi, e.lo, own.hi, own.lo));
    }
};

// One block per chunk, or per (chunk, product) when `split`: even blocks pull from d_a
// into plane 0, odd blocks from d_qk into plane 1. The block first recomputes the chunk's
// sorted log-cumsums with the forward's scan, the values and the counted payloads, into its
// slice of `work`.
// Each warp then pulls for windows of 32 consecutive entries, one per lane, in a fixed
// order. A pair inside the window is visited once, by its later entry, which sends the
// earlier entry its share through a shuffle. The part of the window's first (last) group
// before (after) the window is read in coalesced batches of 32 and broadcast. The pulls
// are latency bound, hence three blocks per SM.
template <typename ValT>
#if defined(P1_BWD_DET)
__global__ void __launch_bounds__(kThreads, 3) p1_bwd_pairs_det_kernel(
#else
__global__ void __launch_bounds__(kThreads, 3) p1_bwd_pairs_kernel(
#endif
    const uint32_t* __restrict__ pay, const float* __restrict__ g,
    const ValT* __restrict__ k_val, const ValT* __restrict__ q_val,
    const float* __restrict__ d_a, const float* __restrict__ d_qk, float* work,
#if defined(P1_BWD_DET)
    Planes planes, int C, int W, int R, int row, int g_width, int g_lane_stride, bool split, int sb_off) {
    // 16-byte aligned base (the batch buffer at tri + sb_off holds 16-byte Entry records).
    extern __shared__ __align__(16) float tri[];
#else
    Planes planes, int C, int W, int R, int row, int g_width, int g_lane_stride, bool split) {
    extern __shared__ float tri[];
#endif
    __shared__ BlockScanSmem scan_smem;
    const int c = split ? blockIdx.x >> 1 : blockIdx.x;
    const bool do_a = !split || !(blockIdx.x & 1);
    const bool do_qk = !split || (blockIdx.x & 1);
    const int n = C * (W + R);
    const int tiles = (n + kTile - 1) / kTile;
    const int64_t base = static_cast<int64_t>(c) * row;
    const uint32_t* pay_row = pay + base;
    const float* g_chunk = g + static_cast<int64_t>(c) * C * g_width;
    const ValT* kv_chunk = k_val + static_cast<int64_t>(c) * C * W;
    const ValT* qv_chunk = q_val + static_cast<int64_t>(c) * C * R;
    float* hi_row = work + static_cast<int64_t>(blockIdx.x) * 4 * row;
    float* lo_row = hi_row + row;
    float* v_row = lo_row + row;
    uint32_t* pc_row = reinterpret_cast<uint32_t*>(v_row + row);
    const int qk = do_a ? C * (C - 1) / 2 : 0;

    SegSum carry{0.0f, 0.0f, 0u};
    for (int tile = 0; tile < tiles; ++tile) {
        const int r0 = tile * kTile + threadIdx.x * kItems;
        uint32_t p[kItems];
        if (r0 < row) load_items(pay_row + r0, p);
        const uint32_t prev = r0 > 0 && r0 <= n ? pay_row[r0 - 1] : kInvalid;
        SegSum item[kItems];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            if (r0 + k >= n) p[k] = kInvalid | kHead;
            const bool carries = is_k(p[k]) && !repeats_k(k ? p[k - 1] : prev, p[k]);
            const float x = carries
                ? g_chunk[token_of(p[k]) * g_width + lane_of(p[k]) * g_lane_stride] : 0.0f;
            item[k] = {x, 0.0f, (carries ? 1u : 0u) | (p[k] & kHead)};
        }
#if defined(P1_BWD_DET)
        // The values are gathered before the block scan (their latency overlaps it).
        float hi[kItems], lo[kItems], v[kItems];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const uint32_t pk = p[k];
            v[k] = is_k(pk) ? to_float(kv_chunk[token_of(pk) * W + lane_of(pk)])
                 : is_q(pk) ? to_float(qv_chunk[token_of(pk) * R + lane_of(pk)]) : 0.0f;
        }
#endif
        SegSum own = item[0];
#pragma unroll
        for (int k = 1; k < kItems; ++k) own = seg_combine(own, item[k]);
        SegSum run = block_scan_exclusive<kThreads, false>(own, carry, scan_smem);
#if !defined(P1_BWD_DET)
        float hi[kItems], lo[kItems], v[kItems];
#endif
        uint32_t pc[kItems];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            run = seg_combine(run, item[k]);
            log2_parts(run, hi[k], lo[k]);
            const uint32_t pk = p[k];
            pc[k] = pk | seg_count(run) << kCountShift;
#if !defined(P1_BWD_DET)
            v[k] = is_k(pk) ? to_float(kv_chunk[token_of(pk) * W + lane_of(pk)])
                 : is_q(pk) ? to_float(qv_chunk[token_of(pk) * R + lane_of(pk)]) : 0.0f;
#endif
        }
        if (r0 < row) {
            store_items(hi_row + r0, hi);
            store_items(lo_row + r0, lo);
            store_items(v_row + r0, v);
            store_items(pc_row + r0, pc);
        }
    }
    const int64_t mat = static_cast<int64_t>(c) * C * C;
    for (int e = threadIdx.x; e < C * C; e += kThreads) {
        const int i = e / C;
        const int j = e - i * C;
        if (do_a && j < i) tri[strict_at(i, j)] = d_a[mat + e];
        if (do_qk && j <= i) tri[qk + lower_at(i, j)] = d_qk[mat + e];
    }
    __syncthreads();

    const Plane out = split && (blockIdx.x & 1) ? planes.p[1] : planes.p[0];
    const int lane = threadIdx.x & 31;
    auto load = [&](int r) {
        return r < row ? Entry{pc_row[r], hi_row[r], lo_row[r], v_row[r]}
                       : Entry{kInvalid | kHead, 0.0f, 0.0f, 0.0f};
    };
    for (int w0 = (threadIdx.x >> 5) * 32; w0 < row; w0 += kThreads) {
        const int r = w0 + lane;
        const Entry own = load(r);
        const int i = token_of(own.pay);
        const bool q = is_q(own.pay);
        const Puller pull{own, tri, qk, q ? qk + lower_at(i, 0) : strict_at(i, 0),
                          q ? do_qk : is_k(own.pay) && do_a, is_k(own.pay), do_a, do_qk};
        // Lanes of this entry's group within the window: [first, last]; first < 0 (last > 31)
        // when the group starts before (ends after) the window.
        const uint32_t heads = __ballot_sync(kFull, is_head(own.pay));
        const uint32_t upto = heads & (kFull >> (31 - lane));
        const uint32_t after = heads & ~(kFull >> (31 - lane));
        const int first = upto ? 31 - __clz(upto) : -1;
        const int last = after ? __ffs(after) - 2 : 32;

        CompSum back, fwd;
        float exact = 0.0f;
        // Pairs inside the window: lane r visits (r - d, r) and sends lane r - d its share.
        const int span = pull.pulls_back ? lane - max(first, 0) : 0;
        const int reach = __reduce_max_sync(kFull, span);
        float back_part = 0.0f, fwd_part = 0.0f;
        for (int d = 1; d <= reach; ++d) {
            const Entry e = shfl_entry_up(own, d);
            const float w = pull.weight_earlier(e, d <= span);
            const bool is_exact = count_of(e.pay) == count_of(own.pay);
            const float mine = w * e.v, theirs = w * own.v;
            if (is_exact) {
                exact += mine;
            } else {
                back_part += mine;
            }
            const float got = __shfl_down_sync(kFull, is_exact ? 0.0f : theirs, d);
            const float got_exact = __shfl_down_sync(kFull, is_exact ? theirs : 0.0f, d);
            if (lane + d < 32) {
                fwd_part += got;
                exact += got_exact;
            }
        }
        back.add(back_part);
        fwd.add(fwd_part);

#if defined(P1_BWD_DET)
        // A warp's current before / after batch (32 entries) in dynamic shared memory after the triangles, when the host
        // found room for it without losing resident blocks (sb_off >= 0); else the shuffles.
        Entry* const sb = sb_off >= 0
            ? reinterpret_cast<Entry*>(reinterpret_cast<char*>(tri) + sb_off) + (threadIdx.x >> 5) * 32 : nullptr;
#endif
        if (!(heads & 1u)) {
            const bool part = first < 0;
            for (int p0 = w0 - 32;; p0 -= 32) {
                const Entry batch = load(p0 + lane);
                const uint32_t bh = __ballot_sync(kFull, is_head(batch.pay));
                const int stop = bh ? 31 - __clz(bh) : 0;
#if defined(P1_BWD_DET)
                if (sb != nullptr) {
                    __syncwarp();   // the previous batch's broadcasts are done
                    sb[lane] = batch;
                    __syncwarp();
                }
#endif
                float part_sum = 0.0f;
                for (int s = 31; s >= stop; --s) {
#if defined(P1_BWD_DET)
                    // one broadcast load instead of four shuffles (same values)
                    const Entry e = sb != nullptr ? sb[s] : shfl_entry(batch, s);
#else
                    const Entry e = shfl_entry(batch, s);
#endif
                    const float t = pull.weight_earlier(e, part) * e.v;
                    if (count_of(e.pay) == count_of(own.pay)) {
                        exact += t;
                    } else {
                        part_sum += t;
                    }
                }
                back.add(part_sum);
                if (bh) break;
            }
        }
        if (w0 + 32 < n && !is_head(pay_row[w0 + 32])) {
            const bool part = last > 31;
            for (int p0 = w0 + 32; p0 < n; p0 += 32) {
                const Entry batch = load(p0 + lane);
                const uint32_t bh = __ballot_sync(kFull, is_head(batch.pay));
                const int stop = bh ? __ffs(bh) - 1 : 32;
#if defined(P1_BWD_DET)
                if (sb != nullptr) {
                    __syncwarp();
                    sb[lane] = batch;
                    __syncwarp();
                }
#endif
                float part_sum = 0.0f;
                for (int s = 0; s < stop; ++s) {
#if defined(P1_BWD_DET)
                    const Entry e = sb != nullptr ? sb[s] : shfl_entry(batch, s);
#else
                    const Entry e = shfl_entry(batch, s);
#endif
                    const float t = pull.weight_later(e, part) * e.v;
                    if (count_of(e.pay) == count_of(own.pay)) {
                        exact += t;
                    } else {
                        part_sum += t;
                    }
                }
                fwd.add(part_sum);
                if (bh) break;
            }
        }
        if (r < row) {
            out.grad[base + r] = back.value() + fwd.value() + exact;
            out.log[base + r] = back.value() - fwd.value();
        }
    }
}

// Per-entry outputs of one chunk: [C, W] K fields, [C, R] Q fields.
template <typename ValT>
struct ChunkGrads {
    ValT* grad_kv;
    ValT* grad_qv;
    float* grad_g;

    __host__ __device__ static size_t bytes(int C, int W, int R) {
        return (sizeof(float) + sizeof(ValT)) * C * W + sizeof(ValT) * C * R;
    }

    // The tile laid out in `mem`, widest field first so that each field stays aligned.
    __device__ static ChunkGrads carve(char* mem, int C, int W, int R) {
        ChunkGrads o;
        o.grad_g = reinterpret_cast<float*>(mem);
        o.grad_kv = reinterpret_cast<ValT*>(o.grad_g + C * W);
        o.grad_qv = o.grad_kv + C * W;
        return o;
    }

    __device__ ChunkGrads at(int64_t kw, int64_t qr) const {
        return {grad_kv + kw, grad_qv + qr, grad_g + kw};
    }

    // Copies this tile to `dst` with the whole block.
    __device__ void copy_to(const ChunkGrads& dst, int cw, int cr) const {
        for (int i = threadIdx.x; i < cw; i += blockDim.x) {
            dst.grad_kv[i] = grad_kv[i];
            dst.grad_g[i] = grad_g[i];
        }
        for (int i = threadIdx.x; i < cr; i += blockDim.x) dst.grad_qv[i] = grad_qv[i];
    }
};

// Per-entry forward values and gradients of the forward outputs, [N, C, W] or [N, C, R].
template <typename ValT>
struct EntryInputs {
    const ValT* k_val;
    const ValT* q_val;
    const float* log_k;
    const float* log_q;
    const float* rex;
    const float* d_k_eff;
    const float* d_k_pd;
    const float* d_q_w;
};

// One block per chunk. Pass 1 runs forward over the row: the per-entry chain (value
// gradients, adjoint of the log-cumsum `dl` and of rex `dr`) and the exclusive scan of dr.
// Pass 2 runs backwards: the inclusive reverse scan of dl, which completes the gradient of
// each K entry's x. The last tile of pass 1 stays in registers for pass 2; the others park
// dl and the scan of dr in the plane-0 rows they consumed. The chunk's gradients are
// assembled in shared memory when they fit (`staged`) and written out contiguously.
template <typename ValT>
#if defined(P1_BWD_DET)
__global__ void __launch_bounds__(kThreads, 2) p1_bwd_gate_det_kernel(
#else
__global__ void __launch_bounds__(kThreads, 2) p1_bwd_gate_kernel(
#endif
    const uint32_t* __restrict__ pay, Planes planes, int num_planes, EntryInputs<ValT> in,
    ChunkGrads<ValT> out, int C, int W, int R, int row, bool staged) {
    extern __shared__ __align__(16) char stage_mem[];
    __shared__ BlockScanSmem scan_smem;
    const int c = blockIdx.x;
    const int n = C * (W + R);
    const int tiles = (n + kTile - 1) / kTile;
    const int64_t base = static_cast<int64_t>(c) * row;
    const int64_t kw = static_cast<int64_t>(c) * C * W;
    const int64_t qr = static_cast<int64_t>(c) * C * R;
    const uint32_t* pay_row = pay + base;
    float* park_dl = planes.p[0].grad + base;
    float* park_dr = planes.p[0].log + base;
    const ChunkGrads<ValT> chunk_out = out.at(kw, qr);
    const ChunkGrads<ValT> dst = staged ? ChunkGrads<ValT>::carve(stage_mem, C, W, R) : chunk_out;

    uint32_t p[kItems];
    float dl[kItems];
    float before[kItems];
    SegSum carry{0.0f, 0.0f, 0u};
    for (int tile = 0; tile < tiles; ++tile) {
        const int r0 = tile * kTile + threadIdx.x * kItems;
        float pg[kItems] = {}, pl[kItems] = {};  // pulled sums: grad and log parts
        if (r0 < row) {
            load_items(pay_row + r0, p);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                if (s == num_planes) break;
                float a[kItems], b[kItems];
                load_items(planes.p[s].grad + base + r0, a);
                load_items(planes.p[s].log + base + r0, b);
#pragma unroll
                for (int k = 0; k < kItems; ++k) {
                    pg[k] += a[k];
                    pl[k] += b[k];
                }
            }
        }
        SegSum item[kItems];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            if (r0 + k >= n) p[k] = kInvalid | kHead;
            const uint32_t pk = p[k];
            const bool valid = !(pk & kInvalid);
            float grad = 0.0f, dr = 0.0f;
            dl[k] = 0.0f;
            if (r0 + k >= n) {
            } else if (pk & kIsQ) {
                const int o = token_of(pk) * R + lane_of(pk);
                if (valid) {
                    const float w = in.d_q_w[qr + o] * expf(in.log_q[qr + o]);
                    grad = pg[k] + w;
                    dl[k] = to_float(in.q_val[qr + o]) * (pl[k] + w);
                }
                dst.grad_qv[o] = from_float<ValT>(grad);
            } else {
                const int o = token_of(pk) * W + lane_of(pk);
                if (valid) {
                    const float kv = to_float(in.k_val[kw + o]);
                    const float eff = in.d_k_eff[kw + o] * expf(in.log_k[kw + o]);
                    const float post = in.d_k_pd[kw + o] * expf(in.rex[kw + o]);
                    grad = pg[k] + eff + post;
                    dl[k] = kv * (pl[k] + eff);
                    dr = kv * post;
                }
                dst.grad_kv[o] = from_float<ValT>(grad);
            }
            item[k] = {dr, 0.0f, pk & kHead};
        }
        SegSum own = item[0];
#pragma unroll
        for (int k = 1; k < kItems; ++k) own = seg_combine(own, item[k]);
        SegSum run = block_scan_exclusive<kThreads, false>(own, carry, scan_smem);
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            before[k] = (item[k].kf & kHead) ? 0.0f : seg_value(run);
            run = seg_combine(run, item[k]);
        }
        if (tile < tiles - 1 && r0 < row) {
            store_items(park_dl + r0, dl);
            store_items(park_dr + r0, before);
        }
    }

    carry = {0.0f, 0.0f, 0u};
    for (int tile = tiles - 1; tile >= 0; --tile) {
        const int r0 = tile * kTile + threadIdx.x * kItems;
        if (tile < tiles - 1 && r0 < row) {
            load_items(pay_row + r0, p);
            load_items(park_dl + r0, dl);
            load_items(park_dr + r0, before);
        }
        const uint32_t next = r0 + kItems < n ? pay_row[r0 + kItems] : kHead;
        const uint32_t prev = r0 > 0 && r0 <= n ? pay_row[r0 - 1] : kInvalid;
        SegSum item[kItems];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            if (r0 + k >= n) p[k] = kInvalid | kHead;
            item[k] = {dl[k], 0.0f, (k + 1 < kItems ? p[k + 1] : next) & kHead};
        }
        SegSum own = item[kItems - 1];
#pragma unroll
        for (int k = kItems - 2; k >= 0; --k) own = seg_combine(own, item[k]);
        SegSum run = block_scan_exclusive<kThreads, true>(own, carry, scan_smem);
#pragma unroll
        for (int k = kItems - 1; k >= 0; --k) {
            run = seg_combine(run, item[k]);
            const uint32_t pk = p[k];
            if (r0 + k >= n || (pk & kIsQ)) continue;
            const bool owns_x = is_k(pk) && !repeats_k(k ? p[k - 1] : prev, pk);
            dst.grad_g[token_of(pk) * W + lane_of(pk)] =
                owns_x ? run.x + (run.c + before[k]) : 0.0f;
        }
    }
    if (staged) {
        __syncthreads();
        dst.copy_to(chunk_out, C * W, C * R);
    }
}

#if defined(P1_BWD_DET)
// The pair kernel's dynamic shared memory plan. The 8 KB batch buffer (one 16-byte Entry per thread) goes at the
// 16-byte aligned end of the triangles only when the CUDA occupancy calculator (allocation granularity, the per-block
// reservation, the static shared memory and the registers included) gives at least as many resident blocks per SM with
// it as without it, and the block still fits the device's opt-in maximum. Speed only: both give the same bits.
struct PairsDetConfig {
    bool use_sb;
    size_t sb_off, dyn_bytes;
    int blocks_tri, blocks_sb;
};

#endif
template <typename ValT>
#if defined(P1_BWD_DET)
PairsDetConfig pairs_det_config(size_t tri_bytes, int64_t smem_cap) {
    PairsDetConfig cfg{false, (tri_bytes + 15) / 16 * 16, tri_bytes, 0, 0};
    const size_t sb_dyn = cfg.sb_off + sizeof(Entry) * kThreads;
    int dev = 0, optin = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    optin = static_cast<int>(std::min<int64_t>(optin, smem_cap));   // the caller's plan limit (_device.smem_cap)
    cudaFuncAttributes fa;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1_bwd_pairs_det_kernel<ValT>));
    if (sb_dyn + fa.sharedSizeBytes > static_cast<size_t>(optin)) return cfg;
    // The calculator honours the kernel's dynamic smem limit: raise it to the larger size before both queries.
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1_bwd_pairs_det_kernel<ValT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sb_dyn)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cfg.blocks_tri, p1_bwd_pairs_det_kernel<ValT>,
                                                                 kThreads, tri_bytes));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cfg.blocks_sb, p1_bwd_pairs_det_kernel<ValT>,
                                                                 kThreads, sb_dyn));
    if (cfg.blocks_sb > 0 && cfg.blocks_sb >= cfg.blocks_tri) {
        cfg.use_sb = true;
        cfg.dyn_bytes = sb_dyn;
    }
    return cfg;
}

template <typename ValT>
std::vector<at::Tensor> run_backward_det(
#else
std::vector<at::Tensor> run_backward(
#endif
    const at::Tensor& payload, const at::Tensor& log_k, const at::Tensor& log_q,
    const at::Tensor& rex, const at::Tensor& k_val, const at::Tensor& q_val,
    const at::Tensor& g, const at::Tensor& d_a, const at::Tensor& d_qk,
    const at::Tensor& d_k_eff, const at::Tensor& d_k_pd, const at::Tensor& d_q_w, int64_t smem_cap) {
    const int64_t N = k_val.size(0);
    const int C = k_val.size(1), W = k_val.size(2), R = q_val.size(2);
    const int row = payload.size(1);
    const auto f32 = k_val.options().dtype(at::kFloat);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* pay = reinterpret_cast<const uint32_t*>(payload.data_ptr<int32_t>());

    const size_t strict_bytes = sizeof(float) * C * (C - 1) / 2;
    const size_t lower_bytes = sizeof(float) * C * (C + 1) / 2;
    const bool split = strict_bytes + lower_bytes > kMaxSmemBytes;
    const int num_planes = split ? 2 : 1;
    const size_t tri_bytes = split ? lower_bytes : strict_bytes + lower_bytes;
    const int blocks = N * num_planes;
    auto work = at::empty({blocks, 4, row}, f32);
    auto sums = at::empty({num_planes, 2, N, row}, f32);
    Planes planes{};
    for (int s = 0; s < num_planes; ++s) {
        planes.p[s] = {sums[s][0].data_ptr<float>(), sums[s][1].data_ptr<float>()};
    }

#if defined(P1_BWD_DET)
    // The shared-memory batch buffers (8 KB) go after the triangles only when they cost no resident block (speed only:
    // both paths give the same bits); residency from the CUDA occupancy calculator (see pairs_det_config).
    const PairsDetConfig cfg = pairs_det_config<ValT>(tri_bytes, smem_cap);
    const bool use_sb = cfg.use_sb;
    const size_t sb_off = cfg.sb_off;
    const size_t dyn_bytes = cfg.dyn_bytes;
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1_bwd_pairs_det_kernel<ValT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dyn_bytes)));
#else
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1_bwd_pairs_kernel<ValT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, tri_bytes));
#endif
    const int g_width = g.size(2);
#if defined(P1_BWD_DET)
    p1_bwd_pairs_det_kernel<ValT><<<blocks, kThreads, dyn_bytes, stream>>>(
#else
    p1_bwd_pairs_kernel<ValT><<<blocks, kThreads, tri_bytes, stream>>>(
#endif
        pay, g.data_ptr<float>(), reinterpret_cast<const ValT*>(k_val.data_ptr()),
        reinterpret_cast<const ValT*>(q_val.data_ptr()), d_a.data_ptr<float>(),
        d_qk.data_ptr<float>(), work.data_ptr<float>(), planes, C, W, R, row, g_width,
#if defined(P1_BWD_DET)
        g_width == 1 ? 0 : 1, split, use_sb ? static_cast<int>(sb_off) : -1);
#else
        g_width == 1 ? 0 : 1, split);
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto grad_kv = at::empty_like(k_val);
    auto grad_qv = at::empty_like(q_val);
    auto grad_g = at::empty({N, C, W}, f32);
    const ChunkGrads<ValT> out{reinterpret_cast<ValT*>(grad_kv.data_ptr()),
                               reinterpret_cast<ValT*>(grad_qv.data_ptr()),
                               grad_g.data_ptr<float>()};
    const EntryInputs<ValT> in{
        reinterpret_cast<const ValT*>(k_val.data_ptr()),
        reinterpret_cast<const ValT*>(q_val.data_ptr()), log_k.data_ptr<float>(),
        log_q.data_ptr<float>(), rex.data_ptr<float>(), d_k_eff.data_ptr<float>(),
        d_k_pd.data_ptr<float>(), d_q_w.data_ptr<float>()};
    const size_t stage_bytes = ChunkGrads<ValT>::bytes(C, W, R);
    // The outputs are staged in shared memory where they fit: at most kMaxSmemBytes (the tuned threshold) and at most
    // the device's opt-in limit (the caller's _device.smem_cap) minus the gate kernel's static shared memory (RTX parts:
    // bf16, C = 128, W = R = 112 needs 114,688 B > 101,376 B). Speed only: both paths give the same bits.
    cudaFuncAttributes gate_attr{};
#if defined(P1_BWD_DET)
    C10_CUDA_CHECK(cudaFuncGetAttributes(&gate_attr, p1_bwd_gate_det_kernel<ValT>));
#else
    C10_CUDA_CHECK(cudaFuncGetAttributes(&gate_attr, p1_bwd_gate_kernel<ValT>));
#endif
    const size_t gate_cap = smem_cap > static_cast<int64_t>(gate_attr.sharedSizeBytes)
                                ? static_cast<size_t>(smem_cap) - gate_attr.sharedSizeBytes : 0;
    const bool stage_out = stage_bytes <= std::min(kMaxSmemBytes, gate_cap);
    const size_t gate_smem = stage_out ? stage_bytes : 0;
#if defined(P1_BWD_DET)
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1_bwd_gate_det_kernel<ValT>,
#else
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1_bwd_gate_kernel<ValT>,
#endif
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, gate_smem));
#if defined(P1_BWD_DET)
    p1_bwd_gate_det_kernel<ValT><<<N, kThreads, gate_smem, stream>>>(
#else
    p1_bwd_gate_kernel<ValT><<<N, kThreads, gate_smem, stream>>>(
#endif
        pay, planes, num_planes, in, out, C, W, R, row, stage_out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_kv, grad_qv, grad_g};
}

#if defined(P1_BWD_DET)
std::vector<at::Tensor> p1_backward_det(
#else
std::vector<at::Tensor> p1_backward(
#endif
    at::Tensor payload, at::Tensor log_k, at::Tensor log_q, at::Tensor rex, at::Tensor k_val,
    at::Tensor q_val, at::Tensor g, at::Tensor d_a, at::Tensor d_qk, at::Tensor d_k_eff,
    at::Tensor d_k_pd, at::Tensor d_q_w, int64_t smem_cap) {
    TORCH_CHECK(k_val.dim() == 3 && q_val.dim() == 3,
#if defined(P1_BWD_DET)
                "p1_backward_det: expected k_val [N, C, W] and q_val [N, C, R]");
#else
                "p1_backward: expected k_val [N, C, W] and q_val [N, C, R]");
#endif
    const int64_t N = k_val.size(0), C = k_val.size(1), W = k_val.size(2), R = q_val.size(2);
    TORCH_CHECK(N >= 1 && C >= 1 && C <= 256 && W >= 1 && W <= 256 && R >= 1 && R <= 256,
#if defined(P1_BWD_DET)
                "p1_backward_det: needs chunk size 1..256 and 1..256 write/read slots per token");
#else
                "p1_backward: needs chunk size 1..256 and 1..256 write/read slots per token");
#endif
    const int64_t row = (C * (W + R) + kRowAlign - 1) / kRowAlign * kRowAlign;
    auto is = [](const at::Tensor& t, std::initializer_list<int64_t> shape) {
        return t.sizes() == at::IntArrayRef(shape);
    };
    TORCH_CHECK(q_val.size(0) == N && q_val.size(1) == C && is(payload, {N, row}) &&
                is(log_k, {N, C, W}) && is(rex, {N, C, W}) && is(d_k_eff, {N, C, W}) &&
                is(d_k_pd, {N, C, W}) && is(log_q, {N, C, R}) && is(d_q_w, {N, C, R}) &&
                is(d_a, {N, C, C}) && is(d_qk, {N, C, C}) && g.dim() == 3 && g.size(0) == N &&
                g.size(1) == C && (g.size(2) == 1 || g.size(2) == W),
#if defined(P1_BWD_DET)
                "p1_backward_det: inconsistent shapes");
#else
                "p1_backward: inconsistent shapes");
#endif
    TORCH_CHECK(k_val.scalar_type() == q_val.scalar_type() &&
                (k_val.scalar_type() == at::kFloat || k_val.scalar_type() == at::kBFloat16),
#if defined(P1_BWD_DET)
                "p1_backward_det: k_val and q_val must both be float32 or bfloat16");
    TORCH_CHECK(payload.scalar_type() == at::kInt, "p1_backward_det: payload must be int32");
#else
                "p1_backward: k_val and q_val must both be float32 or bfloat16");
    TORCH_CHECK(payload.scalar_type() == at::kInt, "p1_backward: payload must be int32");
#endif
    for (const auto* t : {&log_k, &log_q, &rex, &g, &d_a, &d_qk, &d_k_eff, &d_k_pd, &d_q_w}) {
        TORCH_CHECK(t->scalar_type() == at::kFloat,
#if defined(P1_BWD_DET)
                    "p1_backward_det: log-cumsums, rex, gates and gradients must be float32");
#else
                    "p1_backward: log-cumsums, rex, gates and gradients must be float32");
#endif
    }
    for (const auto* t : {&payload, &log_k, &log_q, &rex, &k_val, &q_val, &g, &d_a, &d_qk,
                          &d_k_eff, &d_k_pd, &d_q_w}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->device() == k_val.device(),
#if defined(P1_BWD_DET)
                    "p1_backward_det: inputs must be contiguous tensors on one CUDA device");
#else
                    "p1_backward: inputs must be contiguous tensors on one CUDA device");
#endif
    }
    const c10::cuda::CUDAGuard guard(k_val.device());
    return k_val.scalar_type() == at::kFloat
#if defined(P1_BWD_DET)
        ? run_backward_det<float>(payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk, d_k_eff,
#else
        ? run_backward<float>(payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk, d_k_eff,
#endif
                              d_k_pd, d_q_w, smem_cap)
#if defined(P1_BWD_DET)
        : run_backward_det<__nv_bfloat16>(payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk,
#else
        : run_backward<__nv_bfloat16>(payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk,
#endif
                                      d_k_eff, d_k_pd, d_q_w, smem_cap);
}
"""


@functools.cache
def _module():
    with open(os.path.join(os.path.dirname(__file__), _HEADER), encoding="utf-8") as f:
        header = f.read()
    return load_cuda_extension(
        "sdm_phase1_bwd",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=(_CUDA_SOURCE, "#define P1_BWD_DET 1\n" + _CUDA_SOURCE),
        functions=("p1_backward", "p1_backward_det"),
        headers={_HEADER: header},
        extra_cuda_cflags=("-O3",),
    )


def phase1_backward(
    payload: torch.Tensor,
    log_k: torch.Tensor,
    log_q: torch.Tensor,
    rex: torch.Tensor,
    k_val: torch.Tensor,
    q_val: torch.Tensor,
    g: torch.Tensor,
    d_a: torch.Tensor,
    d_qk: torch.Tensor,
    d_k_eff: torch.Tensor,
    d_k_pd: torch.Tensor,
    d_q_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of the Phase-1 outputs with respect to k_val, q_val and the gates.

    Args:
        payload, log_k, log_q, rex: the forward's outputs of the same name.
        k_val, q_val, g: the forward's inputs.
        d_a, d_qk: [N, C, C] float32 gradients of ``a`` and ``qk``; entries outside their
            triangles are ignored.
        d_k_eff: [N, C, W] float32 gradient of ``k_eff``.
        d_k_pd: [N, C, W] float32 gradient of the post-decayed write weight
            ``k_val * exp(rex)``.
        d_q_w: [N, C, R] float32 gradient of ``q_w``.

    Returns grad_k_val and grad_q_val in the dtype of ``k_val``, and grad_g [N, C, W]
    float32: the gradient of each write's log-decay x (sum it over W for per-token gates).
    Entries that contribute no x (repeated slots of a token, invalid tokens) get zero.
    """
    return tuple(_module().p1_backward(
        payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk, d_k_eff, d_k_pd, d_q_w, smem_cap(k_val.device)))


# Chunk sizes outside [_DET_MIN_C, _DET_MAX_C] take phase1_backward, which gives the same bits and is the faster kernel
# there. A shape-only rule; it never changes a bit.
_DET_MIN_C = 128
_DET_MAX_C = 128


def phase1_backward_det(payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk, d_k_eff, d_k_pd, d_q_w):
    """phase1_backward (same arguments, same outputs, same bits) with the deterministic-mode variant of the kernels at
    the chunk sizes where it is faster."""
    if not (_DET_MIN_C <= log_k.shape[1] <= _DET_MAX_C):
        return phase1_backward(payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk, d_k_eff, d_k_pd, d_q_w)
    return tuple(_module().p1_backward_det(
        payload, log_k, log_q, rex, k_val, q_val, g, d_a, d_qk, d_k_eff, d_k_pd, d_q_w, smem_cap(k_val.device)))
