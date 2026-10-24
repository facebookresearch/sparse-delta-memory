# Copyright (c) Meta Platforms, Inc. and affiliates.
"""product_key_topk_anyf_cuda: CUDA kernels of the product-key top-k for any factor list (``product_key_topk``).

``product_key_topk`` sends two to four factors that the kernels of ``product_key_topk_cuda.py``
accept (each factor at most 65535 columns, at most 32 bits of bit-packed slot index) to those
kernels, and every other factor list here: one factor, five or more, or wider factors. Both
apply the same selection rule, so on two to four factors these kernels return the same bits
(the tests check it). One warp handles one row.

One factor (``product_key_topk_one_fwd_kernel``): the top ``kk = min(k, N)`` columns by (value,
column) descending. Each half-warp keeps the top ``KMAX = max(16, pow2ceil(kk))`` keys of half
of the 256-column tiles with the bitonic networks of the other kernels (a tile that cannot
beat the current kk-th key is skipped), the two halves merge, and the selection is sorted by
column.

Two or more factors (``product_key_topk_anyf_fwd_kernel``): the stages of
``product_key_topk_cuda.py``, for any number of factors. Stage f merges the running list
(factor 0's ranking at first) with the top ``min(k, width)`` keys of factor f: the candidates
are the pairs of a running rank i and a factor rank j with ``(i + 1) * (j + 1) <= c``, a
warp-wide bisection finds the c-th best score, and exact ties at the boundary go to the
larger i + j, then the larger i. The differences from those kernels: factor keys are 64-bit
(the column takes the low 32 bits, so any width works); slot ids are mixed-radix
(``lin * width + column``, so there is no 32-bit packing limit); the per-stage counts are
computed in the kernel from the widths; a stage of a factor at most 8 wide scans a table of
the candidate pairs with ``j < 8`` only; and the running list carries the input values
themselves, NaN included, so the last stage's sums are the returned values (a sum involving
NaN or -inf ranks as -inf either way, so the ranking is unchanged) and no slot is decoded
again.

Backward (``product_key_topk_anyf_bwd_kernel``): one warp per row zero-fills its gradient row,
then writes each touched column once with a segmented sum in a fixed order (factor 0's
columns are runs of the ascending slot ids; the other factors sort (column, position) with
64-bit keys). The summation order is that of ``product_key_topk_cuda.py``, so the result is
deterministic and needs no atomics."""
import functools

from ._build import load_cuda_extension
from .product_key_topk_cuda import TOPK_COMMON

# Largest number of factors (the widths travel in the kernel parameters) and of slots or
# row width (32-bit slot ids and column offsets).
MAX_FACTORS = 512
MAX_SLOTS = 2**31 - 1

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kWarps = 8;  // rows per block
constexpr int kMaxK = 256;
constexpr int kMaxFactors = 512;
constexpr int64_t kMaxSlots = 2147483647;  // slots and row width: 32-bit ids and offsets
constexpr unsigned kFull = 0xffffffffu;

constexpr int kNarrow = 8;  // widest factor served by the narrow pair table

// Number of rank pairs (i, j) with j < J and (i + 1) * (j + 1) <= k.
constexpr int num_pairs(int k, int J) {
    int n = 0;
    for (int j = 0; j < J && j < k; ++j) n += k / (j + 1);
    return n;
}

// Rank pairs (i, j) with j < J and (i + 1) * (j + 1) <= kMaxK, encoded i << 8 | j and ordered
// by (i + 1) * (j + 1): the pairs with (i + 1) * (j + 1) <= k are the first num_pairs(k, J).
// J = kMaxK holds every pair (the table of product_key_topk_cuda.py); J = kNarrow holds
// those of a factor at most kNarrow wide, which a stage of such a factor scans instead.
template <int J> struct PairTable {
    uint16_t code[num_pairs(kMaxK, J)];
};

template <int J> constexpr PairTable<J> make_pair_table() {
    PairTable<J> t{};
    int n = 0;
    for (int p = 1; p <= kMaxK; ++p)
        for (int d = 1; d <= p; ++d)
            if (p % d == 0 && p / d <= J) t.code[n++] = uint16_t((d - 1) << 8 | (p / d - 1));
    return t;
}

__device__ const PairTable<kMaxK> kPairs = make_pair_table<kMaxK>();
__device__ const PairTable<kNarrow> kNarrowPairs = make_pair_table<kNarrow>();

template <int J> __device__ __forceinline__ uint32_t pair_code(int c);
template <> __device__ __forceinline__ uint32_t pair_code<kMaxK>(int c) { return kPairs.code[c]; }
template <> __device__ __forceinline__ uint32_t pair_code<kNarrow>(int c) {
    return kNarrowPairs.code[c];
}

struct FwdProblem {
    int64_t rows, row_stride;
    int nf, k, kk, slots;      // kk = min(k, slots)
    int width[kMaxFactors];
};

struct BwdProblem {
    int64_t rows, gv_row_stride, gv_col_stride;
    int nf, kk, slots, row_width;
    int width[kMaxFactors];
};

""" + TOPK_COMMON + r"""

// Factor sort key: value (NaN as -inf), then larger column first; the low 32 bits hold the
// column. For bfloat16 it ranks exactly like the 32-bit keys of product_key_topk_cuda.py.
using FKey = uint64_t;
template <typename T> __device__ __forceinline__ FKey make_key(T v, int col) {
    return uint64_t(ordered(to_float(v))) << 32 | uint32_t(col);
}
__device__ __forceinline__ uint32_t key_col(FKey k) { return uint32_t(k); }

template <typename T, int M>
__device__ __forceinline__ void load_tile(const T* f, int width, int64_t base, int sub,
                                          FKey (&t)[M]) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
        const int64_t c = base + r * 16 + sub;
        t[r] = c < width ? make_key(f[c], int(c)) : FKey(0);
    }
}

// Top kf keys of a factor, per half-warp (each half-warp may read its own factor); tiles
// are streamed up to the warp-uniform bound.
template <typename T, int G, int M, bool kStream>
__device__ __forceinline__ void factor_top(const T* f, int width, int kf, int bound, int lane,
                                           FKey (&q)[M]) {
    const int sub = lane & 15;
    load_tile<T, M>(f, width, 0, sub, q);
    top_sorted<G, M>(q, lane);
    if constexpr (kStream) {
        const int src = (lane & ~(G - 1)) | (kf - 1) / M;
        for (int64_t base = 16 * M; base < bound; base += 16 * M) {
            FKey t[M];
            load_tile<T, M>(f, width, base, sub, t);
            const FKey thr = __shfl_sync(kFull, register_at(q, (kf - 1) % M), src);
            bool hit = false;
#pragma unroll
            for (int r = 0; r < M; ++r) hit |= t[r] > thr;
            if (!__any_sync(kFull, hit)) continue;
            top_sorted<G, M>(t, lane);
            merge_top<G, M>(q, t, lane, G - 1);
        }
    }
}

// Top kf keys of one factor over the whole warp: half-warp h takes the tiles h, h + 2, ...
// of 16 * M columns, then the halves merge, so every group of G lanes ends with the G * M
// largest keys of the factor, sorted descending.
template <typename T, int G, int M, bool kStream>
__device__ __forceinline__ void single_factor_top(const T* f, int width, int kf, int lane,
                                                  FKey (&q)[M]) {
    constexpr int kTile = 16 * M;
    const int half = lane >> 4, sub = lane & 15;
    load_tile<T, M>(f, width, int64_t(half) * kTile, sub, q);
    top_sorted<G, M>(q, lane);
    if constexpr (kStream) {
        const int src = (lane & ~(G - 1)) | (kf - 1) / M;
        for (int64_t base = 2 * kTile; base < width; base += 2 * kTile) {
            FKey t[M];
            load_tile<T, M>(f, width, base + half * kTile, sub, t);
            const FKey thr = __shfl_sync(kFull, register_at(q, (kf - 1) % M), src);
            bool hit = false;
#pragma unroll
            for (int r = 0; r < M; ++r) hit |= t[r] > thr;
            if (!__any_sync(kFull, hit)) continue;
            top_sorted<G, M>(t, lane);
            merge_top<G, M>(q, t, lane, G - 1);
        }
    }
    merge_top<G, M>(q, q, lane, 16 | (G - 1));
}

template <int N, typename Pred> __device__ __forceinline__ int warp_count(Pred pred) {
    int n = 0;
#pragma unroll
    for (int t = 0; t < N; ++t) n += pred(t);
    return __reduce_add_sync(kFull, n);
}

__device__ __forceinline__ int warp_exclusive_sum(int v, int lane) {
    int s = v;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const int y = __shfl_up_sync(kFull, s, d);
        if (lane >= d) s += y;
    }
    return s - v;
}

// Writes the selection of one row sorted by slot id, from (slot id << 32 | value bits) keys.
template <typename T, int MQ>
__device__ __forceinline__ void store_selection(const uint64_t* sel, int kk, int64_t row,
                                                int lane, T* out_val, int64_t* out_lin) {
    uint64_t out[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        out[r] = e < kk ? sel[e] : ~uint64_t(0);
    }
    bitonic_sort<false, 32, MQ>(out, lane);
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        if (e < kk) {
            out_lin[row * kk + e] = int64_t(out[r] >> 32);
            out_val[row * kk + e] = from_float<T>(__uint_as_float(uint32_t(out[r])));
        }
    }
}

// One factor: the top kk columns by (value, column) descending.
template <typename T, int M, bool kStream, int KMAX>
__global__ __launch_bounds__(kWarps * 32) void product_key_topk_one_fwd_kernel(
    const T* __restrict__ raw, T* __restrict__ out_val, int64_t* __restrict__ out_lin,
    FwdProblem p) {
    constexpr int MQ = KMAX < 32 ? 1 : KMAX / 32;     // selected slots per lane
    constexpr int G = KMAX / M < 16 ? KMAX / M : 16;  // lanes per group holding the top R
    __shared__ uint64_t sel[kWarps][KMAX];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = int64_t(blockIdx.x) * kWarps + warp;
    if (row >= p.rows) return;
    const T* x = raw + row * p.row_stride;
    FKey q[M];
    single_factor_top<T, G, M, kStream>(x, p.width[0], p.kk, lane, q);
    if (lane < G) {
#pragma unroll
        for (int r = 0; r < M; ++r) {
            const int e = lane * M + r;
            if (e < p.kk) {
                const uint32_t c = key_col(q[r]);
                sel[warp][e] = uint64_t(c) << 32 | __float_as_uint(to_float(x[c]));
            }
        }
    }
    __syncwarp();
    store_selection<T, MQ>(sel[warp], p.kk, row, lane, out_val, out_lin);
}

template <int KMAX> struct Scratch {
    float val[2][KMAX];  // running scores by rank, factor values by rank
    uint32_t lin[KMAX];  // running slot ids by rank
    uint32_t col[KMAX];  // factor columns by rank
    // Selected (score, payload) keys of a stage; (slot id, value) keys of the last one.
    uint64_t order[KMAX];
};

// One stage: the best kk of the candidate pairs of a running rank i < n (s.val[0], s.lin)
// and a factor rank j < m (s.val[1], s.col), every candidate taken from table J. The last
// stage leaves its selection in s.order as (slot id << 32 | value bits), an intermediate one
// the running list of the next stage, its scores rounded to T.
template <typename T, int KMAX, int J>
__device__ __forceinline__ void run_stage(Scratch<KMAX>& s, int n, int m, int kk,
                                          uint32_t width, bool last, int lane) {
    constexpr int MQ = KMAX < 32 ? 1 : KMAX / 32;      // selected slots per lane
    constexpr int NC = (num_pairs(KMAX, J) + 31) / 32;  // candidates per lane
    constexpr int NP = num_pairs(kMaxK, J);

    // Score keys of the candidate rank pairs; lane l holds pairs l, l + 32, ... and key 0
    // marks a non-candidate (every real key is above it).
    uint32_t key[NC];
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        const int c = t * 32 + lane;
        key[t] = 0;
        if (c < NP) {
            const uint32_t code = pair_code<J>(c);
            const int i = code >> 8, j = code & 0xff;
            if ((i + 1) * (j + 1) <= kk && i < n && j < m)
                key[t] = ordered(__fadd_rn(s.val[0][i], s.val[1][j]));
        }
    }

    // Bisect for the kk-th best key, keeping count(key >= lo) >= kk > count(key > hi).
    // The pair (lo_i - 1, lo_j - 1) with lo_i * lo_j == kk scores at most the kk-th best
    // (kk <= 256, so lo_i <= 16).
    int lo_i = 0, lo_j = 0;
#pragma unroll
    for (int a = 1; a <= 16; ++a) {
        if (a * a <= kk && kk % a == 0 && a <= n && kk / a <= m) {
            lo_i = a;
            lo_j = kk / a;
        }
    }
    uint32_t lo = lo_i ? ordered(__fadd_rn(s.val[0][lo_i - 1], s.val[1][lo_j - 1])) : 0;
    uint32_t hi = ordered(__fadd_rn(s.val[0][0], s.val[1][0]));
    int n_lo = 0;
    while (lo < hi) {
        const uint32_t mid = hi - (hi - lo) / 2;
        const int cnt = warp_count<NC>([&](int t) { return key[t] >= mid; });
        if (cnt < kk) {
            hi = mid - 1;
        } else {
            lo = mid;
            n_lo = cnt;
            if (cnt == kk) break;
        }
    }
    if (n_lo == 0) n_lo = warp_count<NC>([&](int t) { return key[t] >= lo; });

    // Re-key by final rank class: 1 above the boundary, 2 + payload when tied with it
    // (payload 0xffff - ((i + j) << 8 | i): larger i + j, then larger i, first), ~0 below.
    // Select key <= rho.
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        const uint32_t code = key[t] == lo ? pair_code<J>(t * 32 + lane) : 0;
        const uint32_t payload = 0xffffu - (((code >> 8) + (code & 0xff)) << 8 | code >> 8);
        key[t] = key[t] > lo ? 1u : key[t] == lo ? 2u + payload : ~0u;
    }
    uint32_t rho = 0x10001u;
    if (n_lo > kk) {
        uint32_t plo = 2;
        while (plo < rho) {  // smallest rho with count(key <= rho) >= kk
            const uint32_t mid = plo + (rho - plo) / 2;
            const int cnt = warp_count<NC>([&](int t) { return key[t] <= mid; });
            if (cnt < kk) {
                plo = mid + 1;
            } else {
                rho = mid;
                if (cnt == kk) break;
            }
        }
    }
    int n_sel = 0;
#pragma unroll
    for (int t = 0; t < NC; ++t) n_sel += key[t] <= rho;
    int pos = warp_exclusive_sum(n_sel, lane);
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        if (key[t] <= rho) {
            const uint32_t code = pair_code<J>(t * 32 + lane);
            const int i = code >> 8, j = code & 0xff;
            const float v = __fadd_rn(s.val[0][i], s.val[1][j]);
            s.order[pos++] = last ? uint64_t(s.lin[i] * width + s.col[j]) << 32 | __float_as_uint(v)
                                  : uint64_t(ordered(v)) << 32 | uint32_t((i + j) << 8 | i);
        }
    }
    __syncwarp();
    if (last) return;

    // Sort the selection by (score, payload) into the running list of the next stage.
    uint64_t o[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        o[r] = e < kk ? s.order[e] : 0;
    }
    bitonic_sort<true, 32, MQ>(o, lane);
    float nv[MQ];
    uint32_t nl[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int ij = int(o[r] >> 8) & 0xff, i = int(o[r]) & 0xff, j = ij - i;
        nv[r] = round_to<T>(__fadd_rn(s.val[0][i], s.val[1][j]));
        nl[r] = s.lin[i] * width + s.col[j];
    }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        if (e < kk) {
            s.val[0][e] = nv[r];
            s.lin[e] = nl[r];
        }
    }
    __syncwarp();
}

// Two or more factors, combined in order. The running list and the factor ranks hold the
// input values themselves: a NaN propagates through the sums (so the last stage's value is
// the slot's score with the NaN of an input, as recomputing it from the row would give),
// and since every sum involving a NaN or -inf ranks as -inf, the ranking is that of the
// documented rule.
template <typename T, int M, bool kStream, int KMAX>
__global__ __launch_bounds__(kWarps * 32) void product_key_topk_anyf_fwd_kernel(
    const T* __restrict__ raw, T* __restrict__ out_val, int64_t* __restrict__ out_lin,
    FwdProblem p) {
    constexpr int MQ = KMAX < 32 ? 1 : KMAX / 32;     // selected slots per lane
    constexpr int G = KMAX / M < 16 ? KMAX / M : 16;  // lanes per group holding the top R
    __shared__ Scratch<KMAX> scratch[kWarps];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = int64_t(blockIdx.x) * kWarps + warp;
    if (row >= p.rows) return;
    Scratch<KMAX>& s = scratch[warp];
    const T* x = raw + row * p.row_stride;
    const int half = lane >> 4, sub = lane & 15;

    int n = min(p.k, p.width[0]);  // running list length: min(k, prod(width[:f]))
    int base = p.width[0];         // first column of factor f
    for (int f = 1; f < p.nf; ++f) {
        const int width = p.width[f];
        const int m = min(p.k, width);
        const int kk = int(min(int64_t(p.k), int64_t(n) * m));
        // Top m of factor f (and the top n of factor 0 in stage 1), by (value desc, column
        // desc). Stage 1 sorts factors 0 and 1 in the two half-warps.
        {
            const bool own = half || f > 1;
            const T* fx = own ? x + base : x;
            const int bound = f > 1 ? width : max(p.width[0], width);
            const int fk = own ? m : n;
            FKey q[M];
            factor_top<T, G, M, kStream>(fx, own ? width : p.width[0], fk, bound, lane, q);
            if (sub < G && (half || f == 1)) {
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const int e = sub * M + r;
                    if (e < fk) {
                        const uint32_t c = key_col(q[r]);
                        s.val[half][e] = to_float(fx[c]);
                        if (half) s.col[e] = c;
                        else s.lin[e] = c;
                    }
                }
            }
        }
        __syncwarp();
        const bool last = f + 1 == p.nf;
        if (m <= kNarrow)
            run_stage<T, KMAX, kNarrow>(s, n, m, kk, uint32_t(width), last, lane);
        else
            run_stage<T, KMAX, kMaxK>(s, n, m, kk, uint32_t(width), last, lane);
        n = kk;
        base += width;
    }
    store_selection<T, MQ>(s.order, p.kk, row, lane, out_val, out_lin);
}


template <typename T, int MQ>
__global__ __launch_bounds__(kWarps * 32) void product_key_topk_anyf_bwd_kernel(
    const int64_t* __restrict__ lin, const T* __restrict__ grad_val, T* __restrict__ grad_raw,
    BwdProblem p) {
    __shared__ float stage[kWarps][32 * MQ];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = int64_t(blockIdx.x) * kWarps + warp;
    if (row >= p.rows) return;
    const int kk = p.kk;
    T* out = grad_raw + row * p.row_width;
    for (int64_t c = lane; c < p.row_width; c += 32) out[c] = from_float<T>(0.f);

    uint32_t id[MQ];  // slot ids
    float g[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        id[r] = 0;
        g[r] = 0.f;
        if (e < kk) {
            id[r] = uint32_t(lin[row * kk + e]);
            g[r] = to_float(grad_val[row * p.gv_row_stride + e * p.gv_col_stride]);
            stage[warp][e] = g[r];
        }
    }
    __syncwarp();
    // lin is ascending, so each factor-0 column is a contiguous run; the other factors sort
    // (column, position) first.
    uint32_t div = uint32_t(p.slots);
    int col_base = 0;
    for (int f = 0; f < p.nf; ++f) {
        const uint32_t w = uint32_t(p.width[f]);
        div /= w;  // product of the later widths
        uint32_t c[MQ];
        float v[MQ];
        if (f == 0) {
#pragma unroll
            for (int r = 0; r < MQ; ++r) {
                const int e = lane * MQ + r;
                c[r] = e >= kk ? 0xffffffffu : id[r] / div;
                v[r] = g[r];
            }
        } else {
            uint64_t sk[MQ];
#pragma unroll
            for (int r = 0; r < MQ; ++r) {
                const int e = lane * MQ + r;
                sk[r] = e >= kk ? ~uint64_t(0) : uint64_t(id[r] / div % w) << 32 | uint32_t(e);
            }
            bitonic_sort<false, 32, MQ>(sk, lane);
#pragma unroll
            for (int r = 0; r < MQ; ++r) {
                const bool pad = sk[r] == ~uint64_t(0);
                c[r] = pad ? 0xffffffffu : uint32_t(sk[r] >> 32);
                v[r] = pad ? 0.f : stage[warp][uint32_t(sk[r])];
            }
        }
        store_run_sums<MQ>(c, v, lane, w, out + col_base);
        col_base += int(w);
    }
}

int pow2_ceil(int64_t x) {
    int p = 1;
    while (p < x) p *= 2;
    return p;
}

template <typename T, int M, bool kStream, int KMAX, bool kOne>
void launch_fwd(const at::Tensor& raw, at::Tensor& val, at::Tensor& lin, const FwdProblem& p) {
    const auto grid = (p.rows + kWarps - 1) / kWarps;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const T* x = reinterpret_cast<const T*>(raw.data_ptr());
    T* v = reinterpret_cast<T*>(val.data_ptr());
    if constexpr (kOne)
        product_key_topk_one_fwd_kernel<T, M, kStream, KMAX><<<grid, kWarps * 32, 0, stream>>>(
            x, v, lin.data_ptr<int64_t>(), p);
    else
        product_key_topk_anyf_fwd_kernel<T, M, kStream, KMAX><<<grid, kWarps * 32, 0, stream>>>(
            x, v, lin.data_ptr<int64_t>(), p);
}

template <typename T, int M, bool kStream, bool kOne>
void dispatch_fwd(int kmax, const at::Tensor& raw, at::Tensor& val, at::Tensor& lin,
                  const FwdProblem& p) {
    switch (kmax) {
        case 16: return launch_fwd<T, M, kStream, 16, kOne>(raw, val, lin, p);
        case 32: return launch_fwd<T, M, kStream, 32, kOne>(raw, val, lin, p);
        case 64: return launch_fwd<T, M, kStream, 64, kOne>(raw, val, lin, p);
        case 128: return launch_fwd<T, M, kStream, 128, kOne>(raw, val, lin, p);
        default: return launch_fwd<T, M, kStream, 256, kOne>(raw, val, lin, p);
    }
}

// One factor: M = 16 keys per lane (each half-warp then holds R = KMAX keys), streamed past
// the first two tiles. More factors: M = 1 or 2 up to 32 columns and streamed M = 16 beyond.
template <typename T>
void dispatch_fwd(int max_width, int kmax, const at::Tensor& raw, at::Tensor& val,
                  at::Tensor& lin, const FwdProblem& p) {
    if (p.nf == 1) {
        if (max_width > 2 * 16 * 16) return dispatch_fwd<T, 16, true, true>(kmax, raw, val, lin, p);
        return dispatch_fwd<T, 16, false, true>(kmax, raw, val, lin, p);
    }
    if (max_width <= 16) return dispatch_fwd<T, 1, false, false>(kmax, raw, val, lin, p);
    if (max_width <= 32) return dispatch_fwd<T, 2, false, false>(kmax, raw, val, lin, p);
    return dispatch_fwd<T, 16, true, false>(kmax, raw, val, lin, p);
}

template <typename T, int MQ>
void launch_bwd(const at::Tensor& lin, const at::Tensor& grad_val, at::Tensor& grad,
                const BwdProblem& p) {
    product_key_topk_anyf_bwd_kernel<T, MQ>
        <<<(p.rows + kWarps - 1) / kWarps, kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            lin.data_ptr<int64_t>(), reinterpret_cast<const T*>(grad_val.data_ptr()),
            reinterpret_cast<T*>(grad.data_ptr()), p);
}

template <typename T>
void dispatch_bwd(int mq, const at::Tensor& lin, const at::Tensor& grad_val, at::Tensor& grad,
                  const BwdProblem& p) {
    switch (mq) {
        case 1: return launch_bwd<T, 1>(lin, grad_val, grad, p);
        case 2: return launch_bwd<T, 2>(lin, grad_val, grad, p);
        case 4: return launch_bwd<T, 4>(lin, grad_val, grad, p);
        default: return launch_bwd<T, 8>(lin, grad_val, grad, p);
    }
}

void check_dtype(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat, name,
                " must be bfloat16 or float32, got ", t.scalar_type());
}

// Validates the factor widths and fills the factor fields of a problem; returns the row
// width (the sum of the widths). The slot count goes to p.slots.
template <typename P>
int64_t set_factors(P& p, const std::vector<int64_t>& factors) {
    const int64_t nf = int64_t(factors.size());
    TORCH_CHECK(nf >= 1 && nf <= kMaxFactors, "need 1 to ", kMaxFactors, " factors, got ", nf);
    p.nf = int(nf);
    int64_t sum = 0, slots = 1;
    for (int f = 0; f < p.nf; ++f) {
        const int64_t w = factors[f];
        TORCH_CHECK(w >= 1 && w <= kMaxSlots, "factor widths must be in [1, ", kMaxSlots,
                    "], got ", w);
        p.width[f] = int(w);
        sum += w;
        slots *= w;
        TORCH_CHECK(slots <= kMaxSlots, "the factors address more than ", kMaxSlots, " slots");
        TORCH_CHECK(sum <= kMaxSlots, "the factor widths sum to more than ", kMaxSlots,
                    " columns");
    }
    p.slots = int(slots);
    return sum;
}

}  // namespace

std::vector<at::Tensor> topk_fwd(const at::Tensor& raw, std::vector<int64_t> factors,
                                 int64_t k) {
    TORCH_CHECK(raw.is_cuda() && raw.dim() == 2 && raw.stride(1) == 1,
                "raw must be a CUDA [rows, sum(factors)] tensor with unit column stride");
    check_dtype(raw, "raw");
    FwdProblem p{};
    const int64_t row_width = set_factors(p, factors);
    TORCH_CHECK(row_width == raw.size(1), "the factor widths must sum to the row width ",
                raw.size(1), ", got ", row_width);
    TORCH_CHECK(k >= 1 && k <= kMaxK, "k must be in [1, ", kMaxK, "], got ", k);
    p.rows = raw.size(0);
    p.row_stride = raw.stride(0);
    p.k = int(k);
    p.kk = int(std::min<int64_t>(k, p.slots));
    int max_width = 0;
    for (int f = 0; f < p.nf; ++f) max_width = std::max(max_width, p.width[f]);
    auto val = at::empty({p.rows, p.kk}, raw.options());
    auto lin = at::empty({p.rows, p.kk}, raw.options().dtype(at::kLong));
    if (p.rows == 0) return {val, lin};
    const c10::cuda::CUDAGuard guard(raw.device());
    const int kmax = std::max(16, pow2_ceil(p.kk));
    if (raw.scalar_type() == at::kFloat)
        dispatch_fwd<float>(max_width, kmax, raw, val, lin, p);
    else
        dispatch_fwd<__nv_bfloat16>(max_width, kmax, raw, val, lin, p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {val, lin};
}

at::Tensor topk_bwd(const at::Tensor& lin, const at::Tensor& grad_val,
                    std::vector<int64_t> factors) {
    TORCH_CHECK(lin.is_cuda() && lin.dim() == 2 && lin.is_contiguous() &&
                lin.scalar_type() == at::kLong, "lin must be a contiguous CUDA int64 matrix");
    TORCH_CHECK(grad_val.device() == lin.device() && grad_val.sizes() == lin.sizes(),
                "grad_val must match lin's shape and device");
    check_dtype(grad_val, "grad_val");
    BwdProblem p{};
    const int64_t row_width = set_factors(p, factors);
    TORCH_CHECK(lin.size(1) >= 1 && lin.size(1) <= std::min<int64_t>(kMaxK, p.slots),
                "inconsistent factors or selection width");
    p.rows = lin.size(0);
    p.gv_row_stride = grad_val.stride(0);
    p.gv_col_stride = grad_val.stride(1);
    p.kk = int(lin.size(1));
    p.row_width = int(row_width);
    auto grad = at::empty({p.rows, row_width}, grad_val.options());
    if (p.rows == 0) return grad;
    const c10::cuda::CUDAGuard guard(lin.device());
    const int mq = pow2_ceil((p.kk + 31) / 32);
    if (grad_val.scalar_type() == at::kFloat)
        dispatch_bwd<float>(mq, lin, grad_val, grad, p);
    else
        dispatch_bwd<__nv_bfloat16>(mq, lin, grad_val, grad, p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
"""

_CPP_SOURCE = (
    "std::vector<at::Tensor> topk_fwd(const at::Tensor& raw, std::vector<int64_t> factors, "
    "int64_t k);\n"
    "at::Tensor topk_bwd(const at::Tensor& lin, const at::Tensor& grad_val, "
    "std::vector<int64_t> factors);\n"
)


@functools.cache
def _module():
    """The compiled extension (built on first use)."""
    return load_cuda_extension(
        "sdm_product_key_topk_anyf",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=("topk_fwd", "topk_bwd"),
        extra_cuda_cflags=("-O3",),
    )
