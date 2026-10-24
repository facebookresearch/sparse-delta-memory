# Copyright (c) Meta Platforms, Inc. and affiliates.
"""product_key_topk_cuda: CUDA kernels of the product-key top-k (``product_key_topk.product_key_topk``).

One warp handles one row. The F = 2..4 factors are combined in order: stage f merges the
running list (factor 0's ranking at first) with factor f. Two-factor kernels run one stage;
the three- and four-factor kernels loop over the stages. Forward:

1. Factors. A half-warp keeps the top ``R = min(KMAX, 16 * M)`` keys of a factor, where
   ``KMAX = pow2ceil(kk)`` (at least 16), with a bitonic network over registers (M keys per
   lane): groups of ``R / M`` lanes sort, then merge pairwise until every group holds the top
   ``R``. Factors wider than 256 columns stream 256-column tiles through the same merge.
   Stage 1 sorts factors 0 and 1 in the two half-warps, later stages one factor each.
2. Stage. The candidates are the pairs of a running rank i < n and a factor rank j < m with
   ``(i + 1) * (j + 1) <= c = min(k, n * m)``; any other pair has at least c pairs of lower
   or equal ranks, none of them scoring less. A warp-wide bisection over the candidates'
   score keys finds the c-th best score, then resolves an exactly tied boundary by the rank
   payload ``(i + j) << 8 | i``, larger first.
3. Output. An intermediate stage sorts its selection by (score, payload) into the next
   running list, rounding each score to bfloat16 for bfloat16 input. The last stage sorts its
   selection by slot index (bit-packed columns) and recomputes each value from the input row.

Backward: one warp per row zero-fills its gradient row, then writes each touched column once
with a segmented sum in a fixed order, so it is deterministic and needs no atomics."""
import functools

from ._build import load_cuda_extension

# Device helpers shared with product_key_topk_anyf_cuda.py (spliced into both CUDA sources, inside their anonymous
# namespace): storage <-> float conversions, the order-preserving key map, the bitonic compare-exchange networks and
# the backward's run sums.
TOPK_COMMON = r"""
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

template <typename T> __device__ __forceinline__ float round_to(float x) {
    return to_float(from_float<T>(x));
}

// Order-preserving map of a float to an unsigned key (-0 below +0), with NaN ranked as -inf.
__device__ __forceinline__ uint32_t ordered(float x) {
    const uint32_t b = x != x ? 0xff800000u : __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// One compare-exchange stage of a bitonic network over the G * M keys that G consecutive
// lanes hold in blocked order: element (lane % G) * M + r is register r.
template <bool kDescending, int G, int M, typename K>
__device__ __forceinline__ void bitonic_stage(K (&x)[M], int lane, int size, int stride) {
    const int g = lane % G;
    if (stride >= M) {
        const bool upper = g & (stride / M);
        const bool desc = ((g * M & size) == 0) == kDescending;
#pragma unroll
        for (int r = 0; r < M; ++r) {
            const K y = __shfl_xor_sync(kFull, x[r], stride / M);
            x[r] = ((desc != upper) == (y > x[r])) ? y : x[r];
        }
    } else {
#pragma unroll
        for (int r = 0; r < M; ++r) {
            if (r & stride) continue;
            const bool desc = (((g * M + r) & size) == 0) == kDescending;
            const K a = x[r], b = x[r + stride];
            const bool swap = desc == (b > a);
            x[r] = swap ? b : a;
            x[r + stride] = swap ? a : b;
        }
    }
}

template <bool kDescending, int G, int M, typename K>
__device__ __forceinline__ void bitonic_sort(K (&x)[M], int lane) {
#pragma unroll
    for (int size = 2; size <= G * M; size *= 2)
#pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2)
            bitonic_stage<kDescending, G, M>(x, lane, size, stride);
}

// q <- the G * M largest of q and of t as held by lane ^ mask, where the mask maps a G-lane
// group onto its mirror image in the partner group. Both inputs are sorted descending.
template <int G, int M, typename K>
__device__ __forceinline__ void merge_top(K (&q)[M], const K (&t)[M], int lane, int mask) {
    K y[M];
#pragma unroll
    for (int r = 0; r < M; ++r) y[r] = __shfl_xor_sync(kFull, t[M - 1 - r], mask);
#pragma unroll
    for (int r = 0; r < M; ++r) q[r] = y[r] > q[r] ? y[r] : q[r];
#pragma unroll
    for (int stride = G * M / 2; stride > 0; stride /= 2)
        bitonic_stage<true, G, M>(q, lane, G * M, stride);
}

// Every group of G lanes of a half-warp ends with the G * M largest of the half-warp's
// 16 * M keys, sorted descending.
template <int G, int M, typename K>
__device__ __forceinline__ void top_sorted(K (&x)[M], int lane) {
    bitonic_sort<true, G, M>(x, lane);
#pragma unroll
    for (int d = G; d < 16; d *= 2) merge_top<G, M>(x, x, lane, d | (G - 1));
}

template <int M, typename K> __device__ __forceinline__ K register_at(const K (&x)[M], int r) {
    K v = x[0];
#pragma unroll
    for (int i = 1; i < M; ++i) v = i == r ? x[i] : v;
    return v;
}

// Loads 16 * M columns of a factor starting at `base`; columns past `width` get key 0, below
// every real key. The load order is irrelevant because the keys are sorted next.

// For keys ascending in blocked order, writes the sum of v over each run of equal keys to
// out[key] in a fixed summation order. Keys >= limit are padding.
template <int MQ, typename T>
__device__ __forceinline__ void store_run_sums(const uint32_t (&key)[MQ], const float (&v)[MQ],
                                               int lane, uint32_t limit, T* out) {
    float s[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) s[r] = (r > 0 && key[r] == key[r - 1]) ? s[r - 1] + v[r] : v[r];
    // Sum of the run holding each lane's last key over lanes <= lane (segmented scan).
    const bool joins = __shfl_up_sync(kFull, key[MQ - 1], 1) == key[0] && lane > 0;
    bool head = !(joins && key[0] == key[MQ - 1]);
    float tail = s[MQ - 1];
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const float t = __shfl_up_sync(kFull, tail, d);
        const bool h = __shfl_up_sync(kFull, head, d);
        if (lane >= d) {
            if (!head) tail = t + tail;
            head = head || h;
        }
    }
    const float carry = __shfl_up_sync(kFull, tail, 1);
    const uint32_t next = __shfl_down_sync(kFull, key[0], 1);
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const bool last = r + 1 < MQ ? key[r + 1] != key[r] : (lane == 31 || next != key[r]);
        if (last && key[r] < limit)
            out[key[r]] = from_float<T>(joins && key[r] == key[0] ? carry + s[r] : s[r]);
    }
}
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kWarps = 8;  // rows per block
constexpr int kMaxK = 256;
constexpr int kMaxFactors = 4;
constexpr int kMaxWidth = 65535;
constexpr unsigned kFull = 0xffffffffu;

constexpr int num_pairs(int k) {
    int n = 0;
    for (int i = 1; i <= k; ++i) n += k / i;
    return n;
}

// Rank pairs (i, j) with (i + 1) * (j + 1) <= kMaxK, encoded i << 8 | j and ordered by
// (i + 1) * (j + 1): the pairs with (i + 1) * (j + 1) <= k are the first num_pairs(k).
struct PairTable {
    uint16_t code[num_pairs(kMaxK)];
};

constexpr PairTable make_pair_table() {
    PairTable t{};
    int n = 0;
    for (int p = 1; p <= kMaxK; ++p)
        for (int d = 1; d <= p; ++d)
            if (p % d == 0) t.code[n++] = uint16_t((d - 1) << 8 | (p / d - 1));
    return t;
}

__device__ const PairTable kPairs = make_pair_table();

struct FwdProblem {
    int64_t rows, row_stride;
    int nf, kk;
    int width[kMaxFactors], offset[kMaxFactors];
    int kf[kMaxFactors];     // min(k, width)
    int count[kMaxFactors];  // running list length after stage f: min(k, prod(width[:f + 1]))
    int lo_i[kMaxFactors], lo_j[kMaxFactors];  // stage f: lo_i * lo_j == count[f], or 0
    int bits[kMaxFactors], pos[kMaxFactors];   // column bit field of the packed slot index
};

struct BwdProblem {
    int64_t rows, gv_row_stride, gv_col_stride;
    int nf, kk, row_width;
    int width[kMaxFactors], offset[kMaxFactors];
    int bits[kMaxFactors], pos[kMaxFactors];  // column bit field of the packed slot index
};

""" + TOPK_COMMON + r"""

__device__ __forceinline__ uint32_t unordered_bits(uint32_t u) {
    return (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
}

// Bit field [pos, pos + bits) of a packed slot index; an empty field may sit at pos 32.
__device__ __forceinline__ uint32_t field(uint32_t packed, int pos, int bits) {
    return uint32_t(uint64_t(packed) >> pos) & ((1u << bits) - 1);
}

// Factor sort key: value, then larger column first; the low 16 bits hold the column. value()
// decodes the value with NaN as -inf, whose sums rank exactly like those of the raw values.
template <typename T> struct FactorKey;
template <> struct FactorKey<float> {
    using type = uint64_t;
    __device__ static type make(float v, int col) {
        return uint64_t(ordered(v)) << 32 | uint32_t(col);
    }
    __device__ static float value(type k) { return __uint_as_float(unordered_bits(k >> 32)); }
};
template <> struct FactorKey<__nv_bfloat16> {
    using type = uint32_t;
    __device__ static type make(__nv_bfloat16 v, int col) {
        return (ordered(__bfloat162float(v)) & 0xffff0000u) | uint32_t(col);
    }
    __device__ static float value(type k) {
        return __uint_as_float(unordered_bits(k & 0xffff0000u) & 0xffff0000u);
    }
};

template <typename T, int M>
__device__ __forceinline__ void load_tile(const T* f, int width, int base, int sub,
                                          typename FactorKey<T>::type (&t)[M]) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
        const int c = base + r * 16 + sub;
        t[r] = c < width ? FactorKey<T>::make(f[c], c) : 0;
    }
}

// Top kf keys of a factor, per half-warp (each half-warp may read its own factor); tiles
// are streamed up to the warp-uniform bound.
template <typename T, int G, int M, bool kStream>
__device__ __forceinline__ void factor_top(const T* f, int width, int kf, int bound, int lane,
                                           typename FactorKey<T>::type (&q)[M]) {
    using K = typename FactorKey<T>::type;
    const int sub = lane & 15;
    load_tile<T, M>(f, width, 0, sub, q);
    top_sorted<G, M>(q, lane);
    if constexpr (kStream) {
        const int src = (lane & ~(G - 1)) | (kf - 1) / M;
        for (int base = 16 * M; base < bound; base += 16 * M) {
            K t[M];
            load_tile<T, M>(f, width, base, sub, t);
            const K thr = __shfl_sync(kFull, register_at(q, (kf - 1) % M), src);
            bool hit = false;
#pragma unroll
            for (int r = 0; r < M; ++r) hit |= t[r] > thr;
            if (!__any_sync(kFull, hit)) continue;
            top_sorted<G, M>(t, lane);
            merge_top<G, M>(q, t, lane, G - 1);
        }
    }
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

template <int KMAX> struct Scratch {
    float val[2][KMAX];  // running scores by rank, factor values by rank
    uint32_t lin[KMAX];  // running packed slot indices by rank
    uint16_t col[KMAX];  // factor columns by rank
    union {
        uint32_t sel[KMAX];        // last stage: selected packed slot indices
        uint64_t order[KMAX];      // other stages: selected (score, payload) keys
    };
};

// kMulti: three or four factors; otherwise two, in one stage.
template <typename T, int M, bool kStream, int KMAX, bool kMulti>
__global__ __launch_bounds__(kWarps * 32) void product_key_topk_fwd_kernel(
    const T* __restrict__ raw, T* __restrict__ out_val, int64_t* __restrict__ out_lin,
    FwdProblem p) {
    using FK = FactorKey<T>;
    using K = typename FK::type;
    constexpr int MQ = KMAX < 32 ? 1 : KMAX / 32;     // selected slots per lane
    constexpr int G = KMAX / M < 16 ? KMAX / M : 16;  // lanes per group holding the top R
    constexpr int NC = (num_pairs(KMAX) + 31) / 32;   // candidates per lane
    constexpr int NP = num_pairs(kMaxK);
    constexpr int NF = kMulti ? kMaxFactors : 2;      // bound of the unrolled factor loops
    __shared__ Scratch<KMAX> scratch[kWarps];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = int64_t(blockIdx.x) * kWarps + warp;
    if (row >= p.rows) return;
    Scratch<KMAX>& s = scratch[warp];
    const T* x = raw + row * p.row_stride;
    const int half = lane >> 4, sub = lane & 15;

    for (int f = 1; f < p.nf; ++f) {
        // 1. Top kf of factor f (and of factor 0 in stage 1), by (value desc, column desc).
        {
            const int fi = half || f > 1 ? f : 0;
            const int bound = f > 1 ? p.width[f] : max(p.width[0], p.width[1]);
            K q[M];
            factor_top<T, G, M, kStream>(x + p.offset[fi], p.width[fi], p.kf[fi], bound, lane, q);
            if (sub < G && (half || f == 1)) {
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const int e = sub * M + r;
                    if (e < p.kf[fi]) {
                        s.val[half][e] = FK::value(q[r]);
                        if (half) s.col[e] = uint16_t(q[r] & 0xffffu);
                        else s.lin[e] = uint32_t(q[r] & 0xffffu);
                    }
                }
            }
        }
        __syncwarp();
        const int n = p.count[f - 1], m = p.kf[f], kk = p.count[f];

        // 2. Score keys of the candidate rank pairs; lane l holds pairs l, l + 32, ... and key 0
        // marks a non-candidate (every real key is above it).
        uint32_t key[NC];
#pragma unroll
        for (int t = 0; t < NC; ++t) {
            const int c = t * 32 + lane;
            key[t] = 0;
            if (c < NP) {
                const int i = kPairs.code[c] >> 8, j = kPairs.code[c] & 0xff;
                if ((i + 1) * (j + 1) <= kk && i < n && j < m)
                    key[t] = ordered(__fadd_rn(s.val[0][i], s.val[1][j]));
            }
        }

        // Bisect for the kk-th best key, keeping count(key >= lo) >= kk > count(key > hi).
        uint32_t lo = p.lo_i[f] ? ordered(__fadd_rn(s.val[0][p.lo_i[f] - 1],
                                                     s.val[1][p.lo_j[f] - 1])) : 0;
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
            const uint32_t code = key[t] == lo ? kPairs.code[t * 32 + lane] : 0;
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
        const bool last = !kMulti || f + 1 == p.nf;
#pragma unroll
        for (int t = 0; t < NC; ++t) {
            if (key[t] <= rho) {
                const int code = kPairs.code[t * 32 + lane], i = code >> 8, j = code & 0xff;
                if (last) {
                    s.sel[pos++] = s.lin[i] << p.bits[f] | s.col[j];
                } else {
                    const uint32_t score = ordered(__fadd_rn(s.val[0][i], s.val[1][j]));
                    s.order[pos++] = uint64_t(score) << 32 | uint32_t((i + j) << 8 | i);
                }
            }
        }
        __syncwarp();
        if (last) break;

        // 3a. Sort the selection by (score, payload) into the running list of stage f + 1.
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
            nl[r] = s.lin[i] << p.bits[f] | s.col[j];
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

    // 3b. Sort by slot index; recompute each value from the input row, factor by factor.
    const int kk = p.kk;
    uint32_t out[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        out[r] = e < kk ? s.sel[e] : 0xffffffffu;
    }
    bitonic_sort<false, 32, MQ>(out, lane);
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        if (e < kk) {
            int64_t lin = uint64_t(out[r]) >> p.pos[0];
            float v = to_float(x[lin]);
#pragma unroll
            for (int f = 1; f < NF; ++f) {
                if (kMulti && f >= p.nf) break;
                const int c = field(out[r], p.pos[f], p.bits[f]);
                const float y = to_float(x[p.offset[f] + c]);
                lin = lin * p.width[f] + c;
                v = __fadd_rn(f > 1 ? round_to<T>(v) : v, y);
            }
            out_lin[row * kk + e] = lin;
            out_val[row * kk + e] = from_float<T>(v);
        }
    }
}


template <typename T, int MQ, bool kMulti>
__global__ __launch_bounds__(kWarps * 32) void product_key_topk_bwd_kernel(
    const int64_t* __restrict__ lin, const T* __restrict__ grad_val, T* __restrict__ grad_raw,
    BwdProblem p) {
    constexpr int NF = kMulti ? kMaxFactors : 2;
    __shared__ float stage[kWarps][32 * MQ];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = int64_t(blockIdx.x) * kWarps + warp;
    if (row >= p.rows) return;
    const int kk = p.kk;
    T* out = grad_raw + row * p.row_width;
    for (int c = lane; c < p.row_width; c += 32) out[c] = from_float<T>(0.f);

    uint32_t packed[MQ];  // bit-packed columns
    float g[MQ];
#pragma unroll
    for (int r = 0; r < MQ; ++r) {
        const int e = lane * MQ + r;
        packed[r] = 0;
        g[r] = 0.f;
        if (e < kk) {
            uint32_t rest = uint32_t(lin[row * kk + e]);
#pragma unroll
            for (int f = NF - 1; f > 0; --f) {
                if (kMulti && f >= p.nf) continue;
                const uint32_t q = rest / p.width[f];
                packed[r] |= (rest - q * p.width[f]) << p.pos[f];
                rest = q;
            }
            packed[r] |= uint32_t(uint64_t(rest) << p.pos[0]);
            g[r] = to_float(grad_val[row * p.gv_row_stride + e * p.gv_col_stride]);
            stage[warp][e] = g[r];
        }
    }
    __syncwarp();
    // lin is ascending, so each factor-0 column is a contiguous run; the other factors sort
    // (column, position) first.
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        if (kMulti && f >= p.nf) break;
        uint32_t c[MQ];
        float v[MQ];
#pragma unroll
        for (int r = 0; r < MQ; ++r) {
            const int e = lane * MQ + r;
            const uint32_t col = field(packed[r], p.pos[f], p.bits[f]);
            c[r] = e >= kk ? 0xffffffffu : f ? col << 8 | e : col;
            v[r] = g[r];
        }
        if (f) {
            bitonic_sort<false, 32, MQ>(c, lane);
#pragma unroll
            for (int r = 0; r < MQ; ++r) {
                v[r] = c[r] == 0xffffffffu ? 0.f : stage[warp][c[r] & 0xff];
                c[r] >>= 8;
            }
        }
        store_run_sums<MQ>(c, v, lane, p.width[f], out + p.offset[f]);
    }
}

int pow2_ceil(int64_t x) {
    int p = 1;
    while (p < x) p *= 2;
    return p;
}

int bit_width(int64_t x) {
    int b = 0;
    while ((int64_t(1) << b) <= x) ++b;
    return b;
}

template <typename T, int M, bool kStream, int KMAX, bool kMulti>
void launch_fwd(const at::Tensor& raw, at::Tensor& val, at::Tensor& lin, const FwdProblem& p) {
    product_key_topk_fwd_kernel<T, M, kStream, KMAX, kMulti>
        <<<(p.rows + kWarps - 1) / kWarps, kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const T*>(raw.data_ptr()), reinterpret_cast<T*>(val.data_ptr()),
            lin.data_ptr<int64_t>(), p);
}

template <typename T, int M, bool kStream, bool kMulti>
void dispatch_fwd(int kmax, const at::Tensor& raw, at::Tensor& val, at::Tensor& lin,
                  const FwdProblem& p) {
    switch (kmax) {
        case 16: return launch_fwd<T, M, kStream, 16, kMulti>(raw, val, lin, p);
        case 32: return launch_fwd<T, M, kStream, 32, kMulti>(raw, val, lin, p);
        case 64: return launch_fwd<T, M, kStream, 64, kMulti>(raw, val, lin, p);
        case 128: return launch_fwd<T, M, kStream, 128, kMulti>(raw, val, lin, p);
        default: return launch_fwd<T, M, kStream, 256, kMulti>(raw, val, lin, p);
    }
}

// Two factors pick M = pow2ceil(ceil(width / 16)) keys per lane (streamed past 256 columns);
// more factors use M = 1 or 2 up to 32 columns and streamed M = 16 beyond.
template <typename T>
void dispatch_fwd(int max_width, int kmax, const at::Tensor& raw, at::Tensor& val,
                  at::Tensor& lin, const FwdProblem& p) {
    if (p.nf > 2) {
        if (max_width <= 16) return dispatch_fwd<T, 1, false, true>(kmax, raw, val, lin, p);
        if (max_width <= 32) return dispatch_fwd<T, 2, false, true>(kmax, raw, val, lin, p);
        return dispatch_fwd<T, 16, true, true>(kmax, raw, val, lin, p);
    }
    switch (std::min(16, pow2_ceil((max_width + 15) / 16))) {
        case 1: return dispatch_fwd<T, 1, false, false>(kmax, raw, val, lin, p);
        case 2: return dispatch_fwd<T, 2, false, false>(kmax, raw, val, lin, p);
        case 4: return dispatch_fwd<T, 4, false, false>(kmax, raw, val, lin, p);
        case 8: return dispatch_fwd<T, 8, false, false>(kmax, raw, val, lin, p);
        default:
            if (max_width > 256) return dispatch_fwd<T, 16, true, false>(kmax, raw, val, lin, p);
            return dispatch_fwd<T, 16, false, false>(kmax, raw, val, lin, p);
    }
}

template <typename T, int MQ, bool kMulti>
void launch_bwd(const at::Tensor& lin, const at::Tensor& grad_val, at::Tensor& grad,
                const BwdProblem& p) {
    product_key_topk_bwd_kernel<T, MQ, kMulti>
        <<<(p.rows + kWarps - 1) / kWarps, kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            lin.data_ptr<int64_t>(), reinterpret_cast<const T*>(grad_val.data_ptr()),
            reinterpret_cast<T*>(grad.data_ptr()), p);
}

template <typename T, bool kMulti>
void dispatch_bwd(int mq, const at::Tensor& lin, const at::Tensor& grad_val, at::Tensor& grad,
                  const BwdProblem& p) {
    switch (mq) {
        case 1: return launch_bwd<T, 1, kMulti>(lin, grad_val, grad, p);
        case 2: return launch_bwd<T, 2, kMulti>(lin, grad_val, grad, p);
        case 4: return launch_bwd<T, 4, kMulti>(lin, grad_val, grad, p);
        default: return launch_bwd<T, 8, kMulti>(lin, grad_val, grad, p);
    }
}

template <typename T>
void dispatch_bwd(int mq, const at::Tensor& lin, const at::Tensor& grad_val, at::Tensor& grad,
                  const BwdProblem& p) {
    if (p.nf > 2) return dispatch_bwd<T, true>(mq, lin, grad_val, grad, p);
    return dispatch_bwd<T, false>(mq, lin, grad_val, grad, p);
}

void check_dtype(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat, name,
                " must be bfloat16 or float32, got ", t.scalar_type());
}

// Validates the factor widths and fills the factor fields of a problem; returns the number
// of slots (< 2^32).
template <typename P>
int64_t set_factors(P& p, const std::vector<int64_t>& factors, int64_t row_width) {
    p.nf = int(factors.size());
    TORCH_CHECK(p.nf >= 2 && p.nf <= kMaxFactors, "need 2 to ", kMaxFactors, " factors, got ",
                p.nf);
    int64_t sum = 0, slots = 1;
    int bits = 0;
    for (int f = p.nf - 1; f >= 0; --f) {
        const int64_t w = factors[f];
        TORCH_CHECK(w >= 1 && w <= kMaxWidth, "factor widths must be in [1, ", kMaxWidth,
                    "], got ", w);
        p.width[f] = int(w);
        p.bits[f] = bit_width(w - 1);
        p.pos[f] = bits;
        bits += p.bits[f];
        sum += w;
        slots *= w;
    }
    TORCH_CHECK(sum == row_width, "the factor widths must sum to the row width ", row_width,
                ", got ", sum);
    TORCH_CHECK(bits <= 32, "the factors need ", bits, " bits of slot index (at most 32)");
    for (int f = 1; f < p.nf; ++f) p.offset[f] = p.offset[f - 1] + p.width[f - 1];
    return slots;
}

}  // namespace

std::vector<at::Tensor> topk_fwd(const at::Tensor& raw, std::vector<int64_t> factors,
                                 int64_t k) {
    TORCH_CHECK(raw.is_cuda() && raw.dim() == 2 && raw.stride(1) == 1,
                "raw must be a CUDA [rows, sum(factors)] tensor with unit column stride");
    check_dtype(raw, "raw");
    FwdProblem p{};
    const int64_t slots = set_factors(p, factors, raw.size(1));
    TORCH_CHECK(k >= 1 && k <= kMaxK, "k must be in [1, ", kMaxK, "], got ", k);
    p.rows = raw.size(0);
    p.row_stride = raw.stride(0);
    p.kk = int(std::min<int64_t>(k, slots));
    int max_width = 0;
    for (int f = 0; f < p.nf; ++f) {
        max_width = std::max(max_width, p.width[f]);
        p.kf[f] = int(std::min<int64_t>(k, p.width[f]));
        p.count[f] = f ? int(std::min<int64_t>(k, int64_t(p.count[f - 1]) * p.kf[f])) : p.kf[0];
        if (f == 0) continue;
        for (int a = 1; a * a <= p.count[f]; ++a) {
            if (p.count[f] % a == 0 && a <= p.count[f - 1] && p.count[f] / a <= p.kf[f]) {
                p.lo_i[f] = a;
                p.lo_j[f] = p.count[f] / a;
            }
        }
    }
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
    int64_t row_width = 0;
    for (const int64_t w : factors) row_width += w;
    const int64_t slots = set_factors(p, factors, row_width);
    TORCH_CHECK(lin.size(1) >= 1 && lin.size(1) <= std::min<int64_t>(kMaxK, slots),
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
        "sdm_product_key_topk",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=("topk_fwd", "topk_bwd"),
        extra_cuda_cflags=("-O3",),
    )
