# Copyright (c) Meta Platforms, Inc. and affiliates.
"""phase1_cuda: Phase 1 of the chunked SDM recurrence, the memory-independent quantities of every chunk.

For each chunk the write (K) and read (Q) entries of all tokens are sorted stably by slot
(``phase1_common.cuh`` documents the sorted layout). Every quantity is then a segmented
scan over one slot group or a walk inside it:

  log_k, log_q   per-slot log-cumsum of the gates at each K / Q entry (a Q entry sees the
                 K entries of its own token)
  rex            sum of the gates of the later K entries of the slot in the chunk
  cpe            exp(total gate of the slot in the chunk)
  is_last        last K entry of its slot; is_only: the slot's only K entry
  k_eff, q_w     k_val * exp(log_k) and q_val * exp(log_q)
  a              A[i, j] = sum over shared slots of k_i k_j exp(log_k_i - log_k_j), j < i
  qk             QK[i, j] = sum over shared slots of q_i k_j exp(log_q_i - log_k_j), j <= i
  pos_k, pos_q   the sorted position of every K / Q entry, packed with its peers after it
                 and in its group (see pos_pack)
  q_last, q_only last valid Q entry of its slot; q_only: the slot's only valid Q entry

On request (``det_fields``) the scan also writes the Phase-2 movers' per-entry records and tile indices (krec, kspa,
qrec, qrec8, tiles_k / tiles_q / segb_k, dc_kpos / dc_bpre), in the sorted order of each kind.

Kernels, one block per chunk unless noted: the sort keys (one block per 8192-entry tile,
sorted on chip; longer rows are completed by stable merge rounds), a two-pass segmented
scan, and the pair sums, which accumulate the triangles in shared memory: the ordered walk
(p1_pairs_kernel), its parallel-walk form (p1_pairs_w2_kernel: the same terms in the same order
per cell) and, on sm_90+, the hot/cold kernel (p1_pairs_hc_kernel: slot groups with many K
entries are summed densely). Launches use the current stream, allocate through the caching
allocator and do not synchronize, so they can be captured in a CUDA graph.

Shared memory: the pair kernel bounds each walk by a count byte per entry kept on chip. On a
GPU that cannot hold those bytes beside a triangle (A100 at C = 256 with many slots; see
``pairs_config``), the walks run to their group heads instead: past an entry's last earlier
K entry its group holds only read entries, so the terms and their order, and hence every
output bit, are the same. Shapes whose triangles alone do not fit are refused up front.

Determinism: for fixed inputs, shapes and GPU type every output is bitwise reproducible
(repeated calls, fresh processes, graph replay). The sort is stable and the scans combine in
a fixed tree. Each cell of ``a`` and ``qk`` has exactly one writer thread, which adds the
cell's terms in an order fixed by the input; no atomics touch a sum. The peer counts packed
into ``pos_k`` / ``pos_q`` saturate at 127 (``POS_COUNT_MAX``), and the walks' K-entry budget
at 255; a saturated count only makes a walk run to its group's head, so results never depend
on saturation."""

from __future__ import annotations

import functools
import os
from typing import NamedTuple, Optional

import torch

from ._build import load_cuda_extension
from ._device import backend, headers as device_headers, num_sms, smem_cap

__all__ = ["Phase1", "phase1_forward"]

# Shared memory per block (bytes) that Phase 1 may assume; None: the GPU's opt-in maximum (_device.smem_cap, so a
# forced backend row's patched limit reaches the planner). Lowering it simulates a GPU with less shared memory.
SMEM_LIMIT_OVERRIDE: Optional[int] = None

_HEADER = "phase1_common.cuh"

_CPP_SOURCE = r"""
std::vector<at::Tensor> p1_forward(
    at::Tensor k_idx, at::Tensor q_idx, at::Tensor k_val, at::Tensor q_val, at::Tensor g,
    at::Tensor repeats, int64_t num_parallel, int64_t stream_offset, int64_t key_bits,
    int64_t valid_last, bool read_groups, int64_t smem_limit,
    at::Tensor beta, int64_t rec_L, int64_t det_fields, bool modern, int64_t sms);
std::vector<int64_t> p1_pairs_config(int64_t C, int64_t W, int64_t R, int64_t smem_limit);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

#include <cub/block/block_radix_sort.cuh>

#include "phase1_common.cuh"
#include "sdm_copy.cuh"

using namespace sdm_p1;

constexpr int kKeysThreads = 512;
constexpr int kKeysItems = 16;
constexpr int kKeysTile = kKeysThreads * kKeysItems;
constexpr int kTileSortBits = 4;
constexpr int kMergeThreads = 256;
constexpr int kMergeItems = 16;
constexpr int kScanThreads = 512;
constexpr int kScanItems = 8;
constexpr int kScanTile = kScanThreads * kScanItems;
constexpr int kPairsThreads = 512;
static_assert(kRowAlign % kKeysItems == 0 && kRowAlign % kMergeItems == 0 &&
              kRowAlign % kScanItems == 0 && kRowAlign % 4 == 0, "row padding");
// The scan's blocks per SM and staging budget (its tiles, hence its sums, do not depend on
// them), and the size up to which one block holds both triangles.
constexpr int kDetScanMinBlocks = 1;
constexpr size_t kDetMaxStageBytes = 216 * 1024;
constexpr size_t kMaxJointTriangleBytes = 116 * 1024;

// One block per tile of kKeysTile entries of a row: builds the keys and payloads and sorts
// them stably on chip (LSD radix). Invalid entries take the largest key. Rows longer than a
// tile are then completed by p1_merge_kernel. Block 0 also zeroes the count of chunks without
// a hot mark (see p1_pairs_w2_kernel).
template <typename IdxT, typename KeyT>
__global__ void __launch_bounds__(kKeysThreads) p1_keys_zw_kernel(
    const IdxT* __restrict__ k_idx, const IdxT* __restrict__ q_idx, KeyT* __restrict__ keys,
    uint32_t* __restrict__ pay, int C, int W, int R, int row, int P, int64_t stream_offset,
    int first_partial, int valid_last, int key_bits, uint32_t* __restrict__ zero_word) {
    using Sort = cub::BlockRadixSort<KeyT, kKeysThreads, kKeysItems, uint32_t, kTileSortBits>;
    __shared__ typename Sort::TempStorage sort_smem;
    // Zeroes that count (Records::w2_elig) for the scan launched after this kernel (stream order puts this store
    // before every scan block's atomicAdd), so no separate memset is needed.
    if (zero_word != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *zero_word = 0u;
    const int c = blockIdx.x;
    const int wr = W + R;
    const int n = C * wr;
    const int t_valid = c >= first_partial ? valid_last : C;
    const int64_t base = static_cast<int64_t>(c) * row;
    const int64_t id_base = static_cast<int64_t>(c % P) * stream_offset;
    const int64_t tok_base = static_cast<int64_t>(c) * C;
    const int r0 = blockIdx.y * kKeysTile + threadIdx.x * kKeysItems;
    KeyT key[kKeysItems];
    uint32_t p[kKeysItems];
    // The (token, column) of entry r0 by one division; every next item advances the column, wrapping to the next
    // token. When W and R are multiples of kKeysItems (shapes only), a thread's kKeysItems entries are kKeysItems
    // consecutive K (or Q) columns of one token: their ids load as 16-byte vectors.
    int t = r0 / wr;
    int j = r0 - t * wr;
    if (W % kKeysItems == 0 && R % kKeysItems == 0 && r0 < n) {
        const bool q = j >= W;
        const int lane0 = q ? j - W : j;
        const IdxT* src = q ? q_idx + (tok_base + t) * R + lane0 : k_idx + (tok_base + t) * W + lane0;
        IdxT ids[kKeysItems];
        if ((reinterpret_cast<uintptr_t>(src) & 15u) == 0u) {
            load_items(src, ids);
        } else {
#pragma unroll
            for (int k = 0; k < kKeysItems; ++k) ids[k] = src[k];
        }
        const bool valid = t < t_valid;
#pragma unroll
        for (int k = 0; k < kKeysItems; ++k) {
            key[k] = valid ? static_cast<KeyT>(static_cast<int64_t>(ids[k]) - id_base) : static_cast<KeyT>(~KeyT(0));
            p[k] = static_cast<uint32_t>(t) << kTokenShift | (q ? kIsQ : 0u) |
                   static_cast<uint32_t>(lane0 + k) | (valid ? 0u : kInvalid);
        }
    } else {
#pragma unroll
        for (int k = 0; k < kKeysItems; ++k) {
            const int e = r0 + k;
            key[k] = static_cast<KeyT>(~KeyT(0));
            p[k] = kInvalid;
            if (e < n) {
                const bool q = j >= W;
                const int lane = q ? j - W : j;
                const int64_t id = q ? static_cast<int64_t>(q_idx[(tok_base + t) * R + lane])
                                     : static_cast<int64_t>(k_idx[(tok_base + t) * W + lane]);
                const bool valid = t < t_valid;
                if (valid) key[k] = static_cast<KeyT>(id - id_base);
                p[k] = static_cast<uint32_t>(t) << kTokenShift | (q ? kIsQ : 0u) |
                       static_cast<uint32_t>(lane) | (valid ? 0u : kInvalid);
            }
            if (++j == wr) {
                j = 0;
                ++t;
            }
        }
    }
    Sort(sort_smem).Sort(key, p, 0, key_bits);
    if (r0 < row) {
        store_items(keys + base + r0, key);
        store_items(pay + base + r0, p);
    }
}

// Merges adjacent sorted runs of `run` entries of every row into runs of 2 * run. Ties take
// the earlier run first, so the sort stays stable. Each thread finds where its outputs
// start on the merge path by bisection, then merges kMergeItems outputs.
template <typename KeyT>
__global__ void __launch_bounds__(kMergeThreads) p1_merge_kernel(
    const KeyT* __restrict__ keys_in, const uint32_t* __restrict__ pay_in,
    KeyT* __restrict__ keys_out, uint32_t* __restrict__ pay_out, int n, int row, int run) {
    const int o0 = (blockIdx.y * kMergeThreads + threadIdx.x) * kMergeItems;
    if (o0 >= n) return;
    const int64_t base = static_cast<int64_t>(blockIdx.x) * row;
    const int start = o0 / (2 * run) * (2 * run);
    const int len_a = min(run, n - start);
    const int len_b = max(0, min(run, n - start - run));
    const KeyT* a = keys_in + base + start;
    const KeyT* b = a + len_a;
    const int d = o0 - start;
    int i = max(0, d - len_b);
    int hi = min(d, len_a);
    while (i < hi) {
        const int mid = (i + hi) >> 1;
        if (b[d - mid - 1] < a[mid]) {
            hi = mid;
        } else {
            i = mid + 1;
        }
    }
    int j = d - i;
    KeyT key[kMergeItems];
    uint32_t p[kMergeItems];
#pragma unroll
    for (int k = 0; k < kMergeItems; ++k) {
        const bool from_a = j >= len_b || (i < len_a && !(b[j] < a[i]));
        const int src = from_a ? i++ : len_a + j++;
        key[k] = a[min(src, len_a + len_b - 1)];
        p[k] = pay_in[base + start + min(src, len_a + len_b - 1)];
    }
    if (o0 < row) {
        store_items(keys_out + base + o0, key);
        store_items(pay_out + base + o0, p);
    }
}

// Output fields of the scan, per chunk: [C, W] K fields and [C, R] Q fields. kQFlags is the
// staged form of q_last (bit 0) and q_only (bit 1); positions are staged packed (int32).
enum Field : int {
    kLogK, kRex, kCpe, kKEff, kIsLast, kIsOnly, kPosK,
    kLogQ, kQW, kPosQ, kQFlags,
    kDcKpos,   // dc_kpos ([C, W] int32, entry order), staged like the other K fields
    kFields
};

// The output tensors.
template <typename ValT>
struct Outputs {
    float* log_k;
    float* rex;
    ValT* cpe;
    ValT* k_eff;
    int8_t* is_last;
    int8_t* is_only;
    int32_t* pos_k;
    float* log_q;
    ValT* q_w;
    int32_t* pos_q;
    int8_t* q_last;
    int8_t* q_only;
    int32_t* spay_k;     // [C * W] per chunk
    int32_t* spay_q;     // [C * R] per chunk, or null
    int2* spay_rank;     // [ceil(C * (W + R) / 32)] per chunk
};

// Mover records (optional: krec == nullptr skips them), in the
// sorted K-only / Q-only row order of spay_k / spay_q:
// krec[j] = {spay_k element, absolute slot, kv * exp(rex), cpe as the ValT value widened}
// kspa[j] = -beta[token] * (kv * exp(log_k))
// qrec[j] = {spay_q element, absolute slot, q * exp(log_q), 0}
// tiles_k / tiles_q[t] = the first FIRST index >= t L of the K-only / Q-only row (n if none),
// t < ceil(n / L); the last entry is n. No atomics: the entry at index t L writes it.
// segb_k[t] = valid group FIRSTs (K-only row) before t L, which equals the count before
// tiles_k[t]; the last entry is the chunk's valid group count.
// dc_kpos[entry] = g(j) for a valid FIRST K entry at K-only index j, else -1 (entry order),
// g(j) = valid FIRSTs before j; dc_bpre[b] = g(32 b), the last entry the chunk's total.
// dc_rec[j] = {spay_k element, absolute slot, kspa value, cpe as the ValT value widened} (never requested).
// qrec8[j] = {absolute slot, q * exp(log_q)} (qrec's words 1-2; the element comes from spay_q).
// Padding entries: slot 0, scales +0, cpe 1. A null pointer skips its field.
// The indices are written in the scan loop by the entry at the boundary index, from
// exact per-entry peer counts (next FIRST = j + peers after + 1) and a count of the valid LAST
// write entries before j (= the valid groups that ended before j); no bitmasks, no index pass.
struct Records {
    int4* krec;
    float* kspa;
    int4* qrec;
    int2* qrec8;      // {absolute slot, q * exp(log_q)} per sorted read position (DET_QREC8)
    int* tiles_k;
    int* tiles_q;
    int* segb_k;
    int* dc_kpos;
    int* dc_bpre;
    int4* dc_rec;
    const float* beta;
    int L;
    int l_shift;      // log2(L) when L is a power of two, else -1
    int P;
    int64_t stream_offset;
    bool need_slot;   // a requested record holds the slot: pass 2 fetches the keys
    uint32_t* hot;    // per chunk 1 if a group holds >= hot_g valid K entries, else 0 (or null)
    int hot_g;
    uint32_t* w2_claim;   // work counter of p1_pairs_w2_kernel, zeroed here (block 0) for its launch after the scan
    uint32_t* w2_elig;    // chunks without a hot mark (p1_pairs_w2_kernel's work beside the hot/cold kernel); zeroed by p1_keys_zw_kernel
};

// Compile-time field sets of the specialized scan instantiations (FS = the call's det_fields, one of the dispatch
// layer's masks): the record code of a field whose bit is not in FS is compiled out (such a call has no destination
// for it: checked on the host before the instantiation is chosen). FS = 0: the generic kernel (every pointer checked
// at run time). The bits mirror the DET_* values (static_assert at their host definitions).
constexpr uint32_t kFsSpayK = 8, kFsSpayQ = 16, kFsKrec = 32, kFsKspa = 64, kFsQrec = 128, kFsTiles = 256,
                   kFsDcGidx = 512, kFsDcRec = 1024, kFsQrec8 = 2048;
template <uint32_t FS>
__device__ __forceinline__ constexpr bool fs_has(uint32_t bit) { return FS == 0u || (FS & bit) != 0u; }

// The fields a block assembles in shared memory (bit f of mask) at byte offset[f], to write
// them out contiguously; the others are scattered to global memory directly.
struct StageLayout {
    uint32_t mask;
    int offset[kFields];

    __host__ __device__ bool staged(Field f) const { return (mask >> f) & 1u; }
};

// Copies n elements from shared to global memory with the whole block, 16 bytes per access
// when both sides are 16-byte aligned (any tail element by element).
template <typename T>
__device__ __forceinline__ void block_copy(T* dst, const T* src, int n) {
    constexpr int kVec = 16 / sizeof(T);
    int done = 0;
    if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0) {
        const int vecs = n / kVec;
        for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
            reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
        }
        done = vecs * kVec;
    }
    for (int i = done + threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// Splits staged Q flags into q_last (bit 0) and q_only (bit 1).
__device__ __forceinline__ void block_copy_flags(int8_t* last, int8_t* only, const uint8_t* src,
                                                 int n) {
    int done = 0;
    if (((reinterpret_cast<uintptr_t>(last) | reinterpret_cast<uintptr_t>(only) |
          reinterpret_cast<uintptr_t>(src)) & 15) == 0) {
        const int vecs = n / 16;
        for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
            const uint4 v = reinterpret_cast<const uint4*>(src)[i];
            constexpr uint32_t kOnes = 0x01010101u;
            reinterpret_cast<uint4*>(last)[i] =
                make_uint4(v.x & kOnes, v.y & kOnes, v.z & kOnes, v.w & kOnes);
            reinterpret_cast<uint4*>(only)[i] = make_uint4(
                (v.x >> 1) & kOnes, (v.y >> 1) & kOnes, (v.z >> 1) & kOnes, (v.w >> 1) & kOnes);
        }
        done = vecs * 16;
    }
    for (int i = done + threadIdx.x; i < n; i += blockDim.x) {
        last[i] = src[i] & 1;
        only[i] = src[i] >> 1;
    }
}

// Stores one output element of a chunk: to its staged tile, or to the tensor `dst` (already
// offset to the chunk). ALL: every field the kernel writes is staged.
// A field outside the layout with a null destination is skipped (det_fields).
// The fields stage_layout stages in every ALL layout (`every`); kPosK / kPosQ / kDcKpos are staged there exactly
// when they have a destination (checked on the host for ALL launches).
__device__ __forceinline__ constexpr bool all_always_staged(Field f) {
    return f == kCpe || f == kKEff || f == kQW || f == kLogK || f == kRex || f == kLogQ;
}

template <bool ALL, typename T, typename S = T>
__device__ __forceinline__ void put(const StageLayout& st, char* stage, Field f, T* dst, int o,
                                    T v) {
    if constexpr (ALL) {
        // Under ALL every field with a destination is staged: there is no global-store path.
        if (all_always_staged(f) || st.staged(f)) reinterpret_cast<S*>(stage + st.offset[f])[o] = static_cast<S>(v);
    } else {
        if (st.staged(f)) {
            reinterpret_cast<S*>(stage + st.offset[f])[o] = static_cast<S>(v);
        } else if (dst != nullptr) {
            dst[o] = v;
        }
    }
}

template <typename T, typename S = T>
__device__ __forceinline__ void unstage(const StageLayout& st, const char* stage, Field f, T* dst,
                                        int n) {
    if (st.staged(f) && dst != nullptr) block_copy(dst, reinterpret_cast<const S*>(stage + st.offset[f]), n);
}

// put() of one field of a write / read pair (fk / fq: compile-time constants at every call site), the pair member
// chosen per lane by q, without a dynamically indexed stage offset: exactly put<ALL>(st, stage, q ? fq : fk, q ? dq : dk, o, v).
template <bool ALL, typename T, typename S = T>
__device__ __forceinline__ void put_kq(const StageLayout& st, char* stage, bool q, Field fk, Field fq, T* dk, T* dq, int o,
                                       T v) {
    const bool sk = ALL ? (all_always_staged(fk) || st.staged(fk)) : st.staged(fk);
    const bool sq = ALL ? (all_always_staged(fq) || st.staged(fq)) : st.staged(fq);
    const int off = q ? st.offset[fq] : st.offset[fk];
    if (q ? sq : sk) {
        reinterpret_cast<S*>(stage + off)[o] = static_cast<S>(v);
    } else if constexpr (!ALL) {
        T* const d = q ? dq : dk;
        if (d != nullptr) d[o] = v;
    }
}

// Packed pos_k / pos_q (see pos_pack in the Python module): the position in the chunk's sorted
// row, then the peers after the entry in its group and the peers in the group, saturating.
// Peers are the valid K (for pos_k) or valid Q (for pos_q) entries of the group.
constexpr int kPosBits = 17;
constexpr uint32_t kPosMask = (1u << kPosBits) - 1;
constexpr int kPosAfterShift = kPosBits;
constexpr int kPosSizeShift = kPosBits + 7;
constexpr uint32_t kPosCountMax = 127;
// Elements of the sorted write / read rows: the payload's entry bits (lane, Q, token,
// invalid), the distance back to the group's first element of the row above them
// (saturating), LAST in bit 30 (kLastK's place) and FIRST in bit 31 (kHead's place).
constexpr uint32_t kSortedEntryMask = 0x3FFFFu;
constexpr int kSortedDistShift = 18;
constexpr uint32_t kSortedDistMax = 4095;
static_assert(256 * 512 <= (1 << kPosBits), "positions fit kPosBits");

// SegSum with a Q count: kf holds the K count and the segment start (as in SegSum), qf the
// count of valid Q entries. Counts add exactly (at most 2^16 per row); the float part
// combines exactly as seg_combine, so the sums are those of the shared scan bit for bit.
struct Seg4 {
    float x;
    float c;
    uint32_t kf;
    uint32_t qf;
};

__device__ __forceinline__ Seg4 seg4_combine(Seg4 a, Seg4 b) {
    if (b.kf & kHead) return b;
    const float s = a.x + b.x;
    const float bb = s - a.x;
    const float err = (a.x - (s - bb)) + (b.x - bb);
    return {s, a.c + b.c + err, a.kf + b.kf, a.qf + b.qf};
}

__device__ __forceinline__ float seg4_value(Seg4 s) { return s.x + s.c; }
__device__ __forceinline__ uint32_t seg4_kcount(Seg4 s) { return s.kf & ~kHead; }

template <bool REVERSE>
__device__ __forceinline__ Seg4 shfl4_prev(Seg4 v, int delta) {
    if (REVERSE) {
        return {__shfl_down_sync(0xFFFFFFFFu, v.x, delta),
                __shfl_down_sync(0xFFFFFFFFu, v.c, delta),
                __shfl_down_sync(0xFFFFFFFFu, v.kf, delta),
                __shfl_down_sync(0xFFFFFFFFu, v.qf, delta)};
    }
    return {__shfl_up_sync(0xFFFFFFFFu, v.x, delta), __shfl_up_sync(0xFFFFFFFFu, v.c, delta),
            __shfl_up_sync(0xFFFFFFFFu, v.kf, delta), __shfl_up_sync(0xFFFFFFFFu, v.qf, delta)};
}

struct BlockScanSmem4 {
    Seg4 warp_total[32];
    Seg4 warp_prefix[32];
    Seg4 block_total;
};

// block_scan_exclusive (phase1_common.cuh) on Seg4: the same order of float operations.
template <int THREADS, bool REVERSE>
__device__ Seg4 block_scan4(Seg4 own, Seg4& carry, BlockScanSmem4& smem) {
    constexpr int kWarps = THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rank = REVERSE ? 31 - lane : lane;
    Seg4 inclusive = own;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const Seg4 other = shfl4_prev<REVERSE>(inclusive, d);
        if (rank >= d) inclusive = seg4_combine(other, inclusive);
    }
    const Seg4 before = shfl4_prev<REVERSE>(inclusive, 1);
    if (rank == 31) smem.warp_total[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        const int w = REVERSE ? kWarps - 1 - lane : lane;
        Seg4 v = lane < kWarps ? smem.warp_total[w] : Seg4{0.0f, 0.0f, 0u, 0u};
#pragma unroll
        for (int d = 1; d < kWarps; d <<= 1) {
            const Seg4 other = shfl4_prev<false>(v, d);
            if (lane >= d) v = seg4_combine(other, v);
        }
        const Seg4 earlier = shfl4_prev<false>(v, 1);
        if (lane < kWarps) smem.warp_prefix[w] = lane == 0 ? carry : seg4_combine(carry, earlier);
        if (lane == kWarps - 1) smem.block_total = seg4_combine(carry, v);
    }
    __syncthreads();
    carry = smem.block_total;
    const Seg4 prefix = smem.warp_prefix[warp];
    return rank == 0 ? prefix : seg4_combine(prefix, before);
}

// The scan element and its block scan: SegSum (phase1_common.cuh) with the K count in bits
// 0-14 and the valid-Q count in bits 15-29 of kf, when no group can hold 2^15 entries of a
// kind (C * W and C * R below 2^15: every count then fits its field and the combine stays
// seg_combine's single add); Seg4 otherwise.
// Thread block clusters (sm_90+) for the scan of short rows: CTA r of a cluster of kScanCl CTAs takes
// tile r of its chunk (rows of exactly kScanCl tiles); the carries of the two block scans pass between the CTAs through
// distributed shared memory, so every operation, and hence every output bit, is the single-CTA kernel's.
constexpr int kScanCl = 2;
// Chunks at most this many: two CTAs per chunk then still fit one wave. prev_pow2(SMs / 2): 64 on GB300 (152 SMs).
inline int scan_cl_max_chunks(int sms) {
    if (sms < 2) return 0;
    int p = 1;
    while (2 * p <= sms / 2) p *= 2;
    return p;
}
__device__ __forceinline__ uint32_t cl_rank() {
#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)   // a modern device pass (sdm_copy.cuh)
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
#else
    return 0u;
#endif
}
// Cluster barrier halves: every thread of every CTA of the cluster arrives once per phase (release), then waits
// (acquire) before its next arrive. Called block-uniformly (non-aligned forms: a warp may arrive after a divergent branch).
__device__ __forceinline__ void cl_arrive() {
#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)
    asm volatile("barrier.cluster.arrive.release;" ::: "memory");
#else
    __trap();
#endif
}
__device__ __forceinline__ void cl_wait() {
#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
#else
    __trap();
#endif
}
// The generic address of shared variable `p` in CTA `rank` of this cluster.
template <typename T>
__device__ __forceinline__ T* cl_map(T* p, uint32_t rank) {
#if defined(__CUDA_ARCH__) && !defined(SDM_PORTABLE_ARCH)
    uint64_t out;
    asm volatile("mapa.u64 %0, %1, %2;" : "=l"(out) : "l"(reinterpret_cast<uint64_t>(p)), "r"(rank));
    return reinterpret_cast<T*>(out);
#else
    __trap();
    return p;
#endif
}
__device__ __forceinline__ uint32_t dynamic_smem_bytes() {
    uint32_t b;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(b));
    return b;
}
// The bitwise OR of two values of one type (merges two staging copies of which at most one is non-zero).
template <typename T>
__device__ __forceinline__ T or_bits(T x, T y) {
    using U = typename std::conditional<sizeof(T) == 4, uint32_t,
                                        typename std::conditional<sizeof(T) == 2, uint16_t, uint8_t>::type>::type;
    static_assert(sizeof(T) == sizeof(U), "or_bits: 1, 2 or 4 bytes");
    U a, b;
    memcpy(&a, &x, sizeof(T));
    memcpy(&b, &y, sizeof(T));
    a |= b;
    T r;
    memcpy(&r, &a, sizeof(T));
    return r;
}
// block_copy of the OR of two staging copies (local `a`, the other CTA's `b`): part `part` of `parts` of the range.
template <typename T>
__device__ __forceinline__ void block_copy_or(T* dst, const T* a, const T* b, int n, int part, int parts) {
    constexpr int kVec = 16 / sizeof(T);
    if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 15) == 0) {
        const int vecs = n / kVec;
        const int per = (vecs + parts - 1) / parts;
        const int v0 = min(part * per, vecs), v1 = min(v0 + per, vecs);
        for (int i = v0 + static_cast<int>(threadIdx.x); i < v1; i += blockDim.x) {
            const uint4 x = reinterpret_cast<const uint4*>(a)[i];
            const uint4 y = reinterpret_cast<const uint4*>(b)[i];
            reinterpret_cast<uint4*>(dst)[i] = make_uint4(x.x | y.x, x.y | y.y, x.z | y.z, x.w | y.w);
        }
        if (part == parts - 1) {
            for (int i = vecs * kVec + static_cast<int>(threadIdx.x); i < n; i += blockDim.x) dst[i] = or_bits(a[i], b[i]);
        }
        return;
    }
    const int per = (n + parts - 1) / parts;
    const int e0 = min(part * per, n), e1 = min(e0 + per, n);
    for (int i = e0 + static_cast<int>(threadIdx.x); i < e1; i += blockDim.x) dst[i] = or_bits(a[i], b[i]);
}
// block_scan_exclusive (phase1_common.cuh) with get_carry(carry) called by every thread between the warp scans and
// the carry's first use: the same float operations in the same order, wherever the carry comes from.
template <int THREADS, bool REVERSE, typename GetCarry>
__device__ SegSum block_scan_exclusive_cl(SegSum own, SegSum& carry, BlockScanSmem& smem, GetCarry get_carry) {
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
    get_carry(carry);
    if (warp == 0) {
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

template <typename E>
struct Elem;

template <>
struct Elem<SegSum> {
    using Smem = BlockScanSmem;
    static constexpr uint32_t kQOne = 1u << 15;
    __device__ static SegSum make(float x, bool k, bool q, bool head) {
        return {x, 0.0f, (k ? 1u : 0u) | (q ? kQOne : 0u) | (head ? kHead : 0u)};
    }
    __device__ static SegSum zero() { return {0.0f, 0.0f, 0u}; }
    __device__ static SegSum combine(SegSum a, SegSum b) { return seg_combine(a, b); }
    __device__ static uint32_t kcount(SegSum s) { return s.kf & (kQOne - 1); }
    __device__ static uint32_t qcount(SegSum s) { return (s.kf >> 15) & (kQOne - 1); }
    template <bool REVERSE>
    __device__ static SegSum scan(SegSum own, SegSum& carry, Smem& smem) {
        return block_scan_exclusive<kScanThreads, REVERSE>(own, carry, smem);
    }
    template <bool REVERSE, typename GetCarry>
    __device__ static SegSum scan_cl(SegSum own, SegSum& carry, Smem& smem, GetCarry get_carry) {
        return block_scan_exclusive_cl<kScanThreads, REVERSE>(own, carry, smem, get_carry);
    }
};

template <>
struct Elem<Seg4> {
    using Smem = BlockScanSmem4;
    __device__ static Seg4 make(float x, bool k, bool q, bool head) {
        return {x, 0.0f, (k ? 1u : 0u) | (head ? kHead : 0u), q ? 1u : 0u};
    }
    __device__ static Seg4 zero() { return {0.0f, 0.0f, 0u, 0u}; }
    __device__ static Seg4 combine(Seg4 a, Seg4 b) { return seg4_combine(a, b); }
    __device__ static uint32_t kcount(Seg4 s) { return seg4_kcount(s); }
    __device__ static uint32_t qcount(Seg4 s) { return s.qf; }
    template <bool REVERSE>
    __device__ static Seg4 scan(Seg4 own, Seg4& carry, Smem& smem) {
        return block_scan4<kScanThreads, REVERSE>(own, carry, smem);
    }
};

// Group flags from packed positions (staged mode ALL): last = a peer with no peer after it,
// only = the group's one peer. Entries without peers (invalid) have size 0.
__device__ __forceinline__ void block_flags(int8_t* last, int8_t* only, const int32_t* pos, int n) {
    int done = 0;
    if (((reinterpret_cast<uintptr_t>(last) | reinterpret_cast<uintptr_t>(only)) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(pos) & 15) == 0) {
        const int vecs = n / 4;
        for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
            const int4 p = reinterpret_cast<const int4*>(pos)[i];
            const uint32_t q[4] = {static_cast<uint32_t>(p.x), static_cast<uint32_t>(p.y),
                                   static_cast<uint32_t>(p.z), static_cast<uint32_t>(p.w)};
            uint32_t l = 0, o = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t size = (q[k] >> kPosSizeShift) & kPosCountMax;
                const uint32_t after = (q[k] >> kPosAfterShift) & kPosCountMax;
                l |= static_cast<uint32_t>(size != 0 && after == 0) << (8 * k);
                o |= static_cast<uint32_t>(size == 1) << (8 * k);
            }
            reinterpret_cast<uint32_t*>(last)[i] = l;
            reinterpret_cast<uint32_t*>(only)[i] = o;
        }
        done = vecs * 4;
    }
    for (int i = done + threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t q = static_cast<uint32_t>(pos[i]);
        const uint32_t size = (q >> kPosSizeShift) & kPosCountMax;
        last[i] = size != 0 && ((q >> kPosAfterShift) & kPosCountMax) == 0;
        only[i] = size == 1;
    }
}

// Scan configuration (compile time).
constexpr bool kScanHoistVals = true;    // gather the entries' values before the block scan
constexpr bool kStreamRecords = true;    // records, sorted rows and indices with evict-first stores
// Thread carry: pass 1 hands a tile that leaves registers to pass 2 as each thread's reverse block-scan
// prefix (16 B per thread instead of a 16-B record per entry); pass 2 reloads the flagged payloads and
// recomputes x, the group ends and the reverse item loop from that prefix: the same values, bit for bit.
constexpr bool kScanThreadCarry = true;
// Passes of the pass-2 item loop (1: fully unrolled); the same bits for any value.
constexpr int kScanHalves = 2;
static_assert(kScanItems % kScanHalves == 0 && (kScanItems / kScanHalves) % 2 == 0, "pass width: whole value pairs");

// A store of a record / index element that is read back much later (by another kernel).
template <typename T>
__device__ __forceinline__ void st_rec(T* ptr, T v) {
    if (kStreamRecords) {
        __stcs(ptr, v);
    } else {
        *ptr = v;
    }
}

// The values of a thread's kScanItems entries, kept packed between the gather and their use.
template <typename ValT>
struct ValPack;

template <>
struct ValPack<float> {
    float w[kScanItems];
    __device__ __forceinline__ void set(int k, float v) { w[k] = v; }
    __device__ __forceinline__ float get(int k) const { return w[k]; }
    template <int S>   // values k + S move to slot k
    __device__ __forceinline__ void rotate() {
#pragma unroll
        for (int k = 0; k + S < kScanItems; ++k) w[k] = w[k + S];
    }
};

template <>
struct ValPack<__nv_bfloat16> {
    uint32_t w[kScanItems / 2];
    __device__ __forceinline__ void set(int k, __nv_bfloat16 v) {
        const uint32_t b = __bfloat16_as_ushort(v);
        if (k % 2 == 0) {
            w[k / 2] = b;
        } else {
            w[k / 2] |= b << 16;
        }
    }
    __device__ __forceinline__ __nv_bfloat16 get(int k) const {
        return __ushort_as_bfloat16(static_cast<unsigned short>(w[k / 2] >> (16 * (k % 2))));
    }
    template <int S>   // values k + S move to slot k (S even: whole words)
    __device__ __forceinline__ void rotate() {
        static_assert(S % 2 == 0, "whole words");
#pragma unroll
        for (int k = 0; k + S / 2 < kScanItems / 2; ++k) w[k] = w[k + S / 2];
    }
};

// One block per chunk. Pass 1 walks the row backwards in tiles: group boundaries, x, the
// sum of x after each entry (rex), the last-K flag and every entry's peers after it; it
// writes the payload flags back to the row. Pass 2
// walks forwards: the inclusive log-cumsum, every per-entry output, the packed positions
// (peers before + after give the group size), and the records {payload, hi, lo, value} of
// the pair kernel. The last tile of pass 1 stays in registers for pass 2. The output fields
// that fit in shared memory (`stage`) are assembled there and written out contiguously; the
// others are scattered to global memory directly. ALL: every per-entry field and both
// position maps are staged, and the flags are derived from the packed positions on the way
// out. Pass 2 also writes the sorted write / read rows and their rank table: a plain count
// of write (read) entries gives each entry its index there, and the group counts it already
// has give FIRST (none before), LAST (none after) and the distance back to FIRST.
// CL = kScanCl: a cluster of CL CTAs per chunk (CTA r: tile r; rows of exactly CL tiles; SegSum only).
// FS != 0: the compile-time field set (see kFsKrec ...); FS = 0: the generic kernel.
template <typename KeyT, typename ValT, bool ALL, typename E, int CL = 1, uint32_t FS = 0>
__global__ void __launch_bounds__(kScanThreads, kDetScanMinBlocks) p1_scan_kernel(
    const KeyT* __restrict__ keys, uint32_t* __restrict__ pay, uint4* __restrict__ rec,
    const float* __restrict__ g, const ValT* __restrict__ k_val, const ValT* __restrict__ q_val,
    Outputs<ValT> out, StageLayout stage, unsigned long long* __restrict__ repeats, int C, int W,
    int R, int row, int g_width, int g_lane_stride, Records recs, uint4* __restrict__ tcarry) {
    extern __shared__ __align__(16) char stage_mem[];
    using EL = Elem<E>;
    static_assert(sizeof(E) <= sizeof(uint4), "a scan element fits 16 bytes");
    __shared__ typename EL::Smem scan_smem;
    // Per-warp plain counts of a tile's write / read entries (low / high 16 bits), for the
    // entries' indices in the sorted rows; double-buffered by tile.
    __shared__ uint32_t kind_total[2][kScanThreads / 32];
    // Whether a group of the chunk holds >= recs.hot_g valid K entries (recs.hot).
    __shared__ uint32_t s_hot;
    static_assert(CL == 1 || (CL == kScanCl && std::is_same<E, SegSum>::value), "clusters: kScanCl CTAs, SegSum");
    // CL > 1: what the neighbouring CTAs hand over (written remotely before their arrive): pass 1's carry
    // from CTA rank + 1; pass 2's carry and the kind counts before this tile from CTA rank - 1.
    __shared__ E cl_carry1, cl_carry2;
    __shared__ uint32_t cl_counts[3];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rk = CL > 1 ? static_cast<int>(cl_rank()) : 0;
    const int c = CL > 1 ? static_cast<int>(blockIdx.x) / CL : static_cast<int>(blockIdx.x);
    const int n = C * (W + R);
    const int tiles = (n + kScanTile - 1) / kScanTile;
    if (threadIdx.x == 0) s_hot = 0u;   // ordered before pass 2 by pass 1's barriers
    if (recs.w2_claim != nullptr && c == 0 && rk == 0 && threadIdx.x == 0) *recs.w2_claim = 0u;
    if constexpr (CL > 1) {
        // Every staged element has exactly one writer CTA: both copies start zeroed and merge by OR on the way out
        // (ordered before the first put by pass 1's barriers).
        const uint32_t sb = dynamic_smem_bytes();
        for (uint32_t i = threadIdx.x; i < sb / 16; i += blockDim.x) reinterpret_cast<uint4*>(stage_mem)[i] = make_uint4(0u, 0u, 0u, 0u);
        for (uint32_t i = sb / 16 * 16 + threadIdx.x; i < sb; i += blockDim.x) stage_mem[i] = 0;
    }
    const int64_t base = static_cast<int64_t>(c) * row;
    const int64_t cw = static_cast<int64_t>(c) * C * W;
    const int64_t cr = static_cast<int64_t>(c) * C * R;
    const KeyT* key_row = keys + base;
    uint32_t* pay_row = pay + base;
    uint4* rec_row = rec + base;
    const float* g_chunk = g + static_cast<int64_t>(c) * C * g_width;
    const ValT* kv_chunk = k_val + cw;
    const ValT* qv_chunk = q_val + cr;
    uint32_t p[kScanItems];
    float x[kScanItems];
    float rx[kScanItems];
    // Peers after each item (exact, at most 65535), 16 bits per item: pass 1 to pass 2 of the
    // tile that stays in registers; the other tiles carry them in their records (exact counts, so
    // pass 2 finds the next FIRST of a kind row without bitmasks).
    uint32_t aft[kScanItems / 2];
    unsigned repeat_count = 0;

    E carry = EL::zero();
    for (int tile = tiles - 1; tile >= 0; --tile) {
        if (CL > 1 && tile != rk) continue;   // block-uniform: CTA rk takes tile rk
        const int r0 = tile * kScanTile + threadIdx.x * kScanItems;
        KeyT key[kScanItems];
        if (r0 < row) {
            load_items(key_row + r0, key);
            load_items(pay_row + r0, p);
        }
#pragma unroll
        for (int k = 0; k < kScanItems; ++k) {
            if (r0 + k >= n) {
                key[k] = 0;
                p[k] = kInvalid;
            }
        }
        const bool has_prev = r0 > 0 && r0 <= n;
        const KeyT prev_key = has_prev ? key_row[r0 - 1] : KeyT(0);
        const uint32_t prev_pay = has_prev ? pay_row[r0 - 1] : kInvalid;
        const int r_next = r0 + kScanItems;
        const KeyT next_key = r_next < n ? key_row[r_next] : KeyT(0);
        const uint32_t next_pay = r_next < n ? pay_row[r_next] : kInvalid;
        if constexpr (CL > 1) {
            // Cluster barrier 0: every CTA of the cluster runs (its shared memory may be addressed from here on) and has
            // read its neighbours' boundary entries before any CTA writes its flags into the row.
            cl_arrive();
            cl_wait();
        }

#pragma unroll
        for (int k = 0; k < kScanItems; ++k) {
            const KeyT kp = k ? key[k - 1] : prev_key;
            const uint32_t pp = k ? p[k - 1] : prev_pay;
            const bool head = r0 + k == 0 || ((p[k] | pp) & kInvalid) || key[k] != kp;
            p[k] |= head ? kHead : 0u;
            const bool rep = repeats_k(pp, p[k]);
            repeat_count += rep;
            x[k] = is_k(p[k]) && !rep
                ? g_chunk[token_of(p[k]) * g_width + lane_of(p[k]) * g_lane_stride] : 0.0f;
        }
        // Bit k: entry k is the last of its group (its successor starts a group).
        uint32_t ends = 0;
#pragma unroll
        for (int k = 0; k < kScanItems; ++k) {
            const bool next_head = k + 1 < kScanItems
                ? is_head(p[k + 1])
                : ((next_pay | p[k]) & kInvalid) || next_key != key[k];
            ends |= (next_head ? 1u : 0u) << k;
        }
        // The scan element of entry k (built where needed: fewer live registers).
        auto item = [&](int k) {
            return EL::make(x[k], is_k(p[k]), is_q(p[k]), (ends >> k) & 1u);
        };
        E own = item(kScanItems - 1);
#pragma unroll
        for (int k = kScanItems - 2; k >= 0; --k) own = EL::combine(own, item(k));
        E run;
        if constexpr (CL > 1) {
            if (rk < CL - 1) {   // the carry = the block total of tile rk + 1 (barrier 1)
                run = EL::template scan_cl<true>(own, carry, scan_smem, [&](E& cy) {
                    cl_arrive();
                    cl_wait();
                    cy = cl_carry1;
                });
            } else {
                run = EL::template scan<true>(own, carry, scan_smem);
            }
            if (rk > 0) {   // hand this tile's total to CTA rk - 1; barrier 1 (waited in pass 2)
                if (threadIdx.x == 0) *cl_map(&cl_carry1, static_cast<uint32_t>(rk - 1)) = carry;
                cl_arrive();
            }
        } else {
            run = EL::template scan<true>(own, carry, scan_smem);
        }
        const bool thread_carry = CL == 1 && kScanThreadCarry && tcarry != nullptr;
        if (thread_carry && tile > 0) {   // thread carry: this thread's prefix for pass 2
            uint4 t = make_uint4(0u, 0u, 0u, 0u);
            memcpy(&t, &run, sizeof(E));
            tcarry[(static_cast<int64_t>(c) * tiles + tile) * kScanThreads + threadIdx.x] = t;
        }
#pragma unroll
        for (int k = 0; k < kScanItems / 2; ++k) aft[k] = 0u;
#pragma unroll
        for (int k = kScanItems - 1; k >= 0; --k) {
            const E after = ((ends >> k) & 1u) ? EL::zero() : run;
            rx[k] = after.x + after.c;
            if (is_k(p[k]) && EL::kcount(after) == 0) p[k] |= kLastK;
            const uint32_t peers =
                is_k(p[k]) ? EL::kcount(after) : is_q(p[k]) ? EL::qcount(after) : 0u;
            aft[k / 2] |= min(peers, 0xFFFFu) << (16 * (k % 2));
            run = EL::combine(run, item(k));
        }
        if (r0 < row) {
            store_items(pay_row + r0, p);
            // Pass 2 reloads the payload, peers after, x and rex of the tiles that leave
            // registers from their records (thread carry: from the payloads and the thread's prefix).
            if (CL == 1 && tile > 0 && !thread_carry) {
#pragma unroll
                for (int k = 0; k < kScanItems; ++k) {
                    rec_row[r0 + k] = make_uint4(p[k], (aft[k / 2] >> (16 * (k % 2))) & 0xFFFFu,
                                                 __float_as_uint(x[k]), __float_as_uint(rx[k]));
                }
            }
        }
        __syncthreads();
    }
    repeat_count = __reduce_add_sync(0xFFFFFFFFu, repeat_count);
    if ((threadIdx.x & 31) == 0 && repeat_count) atomicAdd(repeats, repeat_count);

    carry = EL::zero();
    // Write / read entries and valid LAST write entries before the tile (all threads keep them).
    uint32_t kind_k = 0, kind_q = 0, kind_vl = 0;
    // The records' slot base of this chunk (read after the first block scan's barriers).
    __shared__ long long s_id_base;
    if (threadIdx.x == 0) s_id_base = static_cast<long long>(c % max(recs.P, 1)) * recs.stream_offset;
    // Per-warp counts of a tile's valid LAST write entries (double-buffered by tile, like kind_total).
    __shared__ uint32_t vl_total[2][kScanThreads / 32];
    const int nk = C * W, nq = C * R;
    const int nwk = (nk + 31) >> 5;
    const int L = max(recs.L, 1);
    const int ntk = (nk + L - 1) / L, ntq = (nq + L - 1) / L;
    const int64_t tk_base = static_cast<int64_t>(c) * (ntk + 1);
    const int64_t tq_base = static_cast<int64_t>(c) * (ntq + 1);
    const int64_t bp_base = static_cast<int64_t>(c) * (nwk + 1);
    // Index j of a kind row starts tile t of length L (shifts when L is a power of two).
    auto tile_at = [&](uint32_t j, int& t) -> bool {
        if (recs.l_shift >= 0) {
            if (j & ((1u << recs.l_shift) - 1u)) return false;
            t = static_cast<int>(j >> recs.l_shift);
            return true;
        }
        if (j % static_cast<uint32_t>(L)) return false;
        t = static_cast<int>(j / static_cast<uint32_t>(L));
        return true;
    };
#define P1_OR_NULL(p, off) ((p) != nullptr ? (p) + (off) : nullptr)
    bool hot_mine = false;   // an item of this thread meets the hot test (a group with >= recs.hot_g valid K entries)
    for (int tile = 0; tile < tiles; ++tile) {
        if (CL > 1 && tile != rk) continue;   // block-uniform: the tile pass 1 left in registers
        const int r0 = tile * kScanTile + threadIdx.x * kScanItems;
        if (CL == 1 && tile > 0 && r0 < row && kScanThreadCarry && tcarry != nullptr) {
            // Thread carry: pass 1's flagged payloads and this thread's reverse prefix; x, the group ends
            // and the reverse item loop exactly as pass 1 computed them.
            load_items(pay_row + r0, p);
            const uint4 t = tcarry[(static_cast<int64_t>(c) * tiles + tile) * kScanThreads + threadIdx.x];
            E run;
            memcpy(&run, &t, sizeof(E));
            const uint32_t prev_pay = r0 > 0 && r0 <= n ? pay_row[r0 - 1] : kInvalid;
            const int r_next = r0 + kScanItems;
            const bool last_ends = r_next < n ? is_head(pay_row[r_next]) : true;
#pragma unroll
            for (int k = 0; k < kScanItems; ++k) {
                if (r0 + k >= n) p[k] = kInvalid | kHead;
            }
            uint32_t ends = 0;
#pragma unroll
            for (int k = 0; k < kScanItems; ++k) {
                const uint32_t pp = k ? p[k - 1] : prev_pay;   // flags do not affect repeats_k's prev
                x[k] = is_k(p[k]) && !repeats_k(pp, p[k])
                    ? g_chunk[token_of(p[k]) * g_width + lane_of(p[k]) * g_lane_stride] : 0.0f;
                const bool next_head = k + 1 < kScanItems ? is_head(p[k + 1]) : last_ends;
                ends |= (next_head ? 1u : 0u) << k;
            }
            auto ritem = [&](int k) {
                return EL::make(x[k], is_k(p[k]), is_q(p[k]), (ends >> k) & 1u);
            };
#pragma unroll
            for (int k = 0; k < kScanItems / 2; ++k) aft[k] = 0u;
#pragma unroll
            for (int k = kScanItems - 1; k >= 0; --k) {
                const E after = ((ends >> k) & 1u) ? EL::zero() : run;
                rx[k] = after.x + after.c;
                const uint32_t peers =
                    is_k(p[k]) ? EL::kcount(after) : is_q(p[k]) ? EL::qcount(after) : 0u;
                aft[k / 2] |= min(peers, 0xFFFFu) << (16 * (k % 2));
                run = EL::combine(run, ritem(k));
            }
        } else if (CL == 1 && tile > 0 && r0 < row) {
#pragma unroll
            for (int k = 0; k < kScanItems / 2; ++k) aft[k] = 0u;
#pragma unroll
            for (int k = 0; k < kScanItems; ++k) {
                const uint4 e = rec_row[r0 + k];
                p[k] = e.x;
                aft[k / 2] |= e.y << (16 * (k % 2));
                x[k] = __uint_as_float(e.z);
                rx[k] = __uint_as_float(e.w);
            }
        }
#pragma unroll
        for (int k = 0; k < kScanItems; ++k) {
            if (r0 + k >= n) {
                p[k] = kInvalid | kHead;
                x[k] = 0.0f;
            }
        }
        // The keys (record slots) and the entries' values are fetched before the block scan, so their
        // latency overlaps its barriers.
        KeyT key[kScanItems];
#pragma unroll
        for (int k = 0; k < kScanItems; ++k) key[k] = KeyT(0);   // defined even when no record holds a slot
        if (recs.need_slot && r0 < row) load_items(key_row + r0, key);
        ValPack<ValT> vals;
        if (kScanHoistVals) {
#pragma unroll
            for (int k = 0; k < kScanItems; ++k) {
                const uint32_t pk = p[k];
                ValT raw = from_float<ValT>(0.0f);
                if (r0 + k < n && !(pk & kInvalid)) {
                    raw = (pk & kIsQ) ? qv_chunk[token_of(pk) * R + lane_of(pk)]
                                      : kv_chunk[token_of(pk) * W + lane_of(pk)];
                }
                vals.set(k, raw);
            }
        }
        auto item = [&](int k) { return EL::make(x[k], is_k(p[k]), is_q(p[k]), p[k] & kHead); };
        // Each entry's index in its kind's sorted row: a plain count of the row's write (read)
        // entries before it, padding included; and the valid LAST write entries before it (the
        // valid groups that ended before it). The warp totals are published here and read
        // after the block scan below, whose barriers order them (double-buffered by tile).
        uint32_t kbits = 0, nqi = 0, nvl = 0;
#pragma unroll
        for (int k = 0; k < kScanItems; ++k) {
            if (r0 + k < n) {
                if (p[k] & kIsQ) {
                    ++nqi;
                } else {
                    kbits |= 1u << k;
                    nvl += (p[k] & kLastK) ? 1u : 0u;   // kLastK: valid K entries only
                }
            }
        }
        const uint32_t own_kind = __popc(kbits) | nqi << 16;
        uint32_t incl_kind = own_kind, incl_vl = nvl;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, incl_kind, d);
            const uint32_t other_vl = __shfl_up_sync(0xFFFFFFFFu, incl_vl, d);
            if (lane >= d) {
                incl_kind += other;
                incl_vl += other_vl;
            }
        }
        if (lane == 31) {
            kind_total[tile & 1][warp] = incl_kind;
            vl_total[tile & 1][warp] = incl_vl;
        }
        E own = item(0);
#pragma unroll
        for (int k = 1; k < kScanItems; ++k) own = EL::combine(own, item(k));
        E run;
        if constexpr (CL > 1) {
            if (rk > 0) {   // the carry and kind counts of tile rk - 1 (barrier 2)
                cl_wait();   // barrier 1
                run = EL::template scan_cl<false>(own, carry, scan_smem, [&](E& cy) {
                    cl_arrive();
                    cl_wait();
                    cy = cl_carry2;
                    kind_k = cl_counts[0];
                    kind_q = cl_counts[1];
                    kind_vl = cl_counts[2];
                });
            } else {
                run = EL::template scan<false>(own, carry, scan_smem);
            }
        } else {
            run = EL::template scan<false>(own, carry, scan_smem);
        }
        uint32_t kind_before = incl_kind - own_kind, kind_tile;
        uint32_t vl_before = incl_vl - nvl, vl_tile;
        {
            // The warps' totals through one 16-lane warp scan (lane w holds warp w's totals) instead of a 16-step loop
            // in every thread: integer sums of the same terms, so the same values.
            constexpr int kWarpsT = kScanThreads / 32;
            static_assert(kWarpsT <= 32 && (kWarpsT & (kWarpsT - 1)) == 0, "one warp total per lane");
            uint32_t sk = lane < kWarpsT ? kind_total[tile & 1][lane] : 0u;
            uint32_t sv = lane < kWarpsT ? vl_total[tile & 1][lane] : 0u;
#pragma unroll
            for (int d = 1; d < kWarpsT; d <<= 1) {
                const uint32_t ok = __shfl_up_sync(0xFFFFFFFFu, sk, d);
                const uint32_t ov = __shfl_up_sync(0xFFFFFFFFu, sv, d);
                if (lane >= d) {
                    sk += ok;
                    sv += ov;
                }
            }
            const int src = warp > 0 ? warp - 1 : 0;   // warp-uniform
            const uint32_t bk = __shfl_sync(0xFFFFFFFFu, sk, src);
            const uint32_t bv = __shfl_sync(0xFFFFFFFFu, sv, src);
            kind_before += warp > 0 ? bk : 0u;
            vl_before += warp > 0 ? bv : 0u;
            kind_tile = __shfl_sync(0xFFFFFFFFu, sk, kWarpsT - 1);
            vl_tile = __shfl_sync(0xFFFFFFFFu, sv, kWarpsT - 1);
        }
        uint32_t jk = kind_k + (kind_before & 0xFFFFu), jq = kind_q + (kind_before >> 16);
        uint32_t vlb = kind_vl + vl_before;   // valid LAST write entries before this thread's items
        kind_k += kind_tile & 0xFFFFu;
        kind_q += kind_tile >> 16;
        kind_vl += vl_tile;
        if constexpr (CL > 1) {
            if (rk < CL - 1) {   // hand the carry and the counts through this tile to CTA rk + 1; barrier 2 (waited at the end)
                if (threadIdx.x == 0) {
                    const uint32_t nb = static_cast<uint32_t>(rk + 1);
                    *cl_map(&cl_carry2, nb) = carry;
                    uint32_t* cn = cl_map(cl_counts, nb);
                    cn[0] = kind_k;
                    cn[1] = kind_q;
                    cn[2] = kind_vl;
                }
                cl_arrive();
            }
        }
        // Rank table: per 32 positions (four threads' items), the write entries before them
        // and the bitmask of write entries among them.
        {
            uint32_t word = kbits;
            word |= __shfl_down_sync(0xFFFFFFFFu, kbits, 1) << 8;
            word |= __shfl_down_sync(0xFFFFFFFFu, kbits, 2) << 16;
            word |= __shfl_down_sync(0xFFFFFFFFu, kbits, 3) << 24;
            if ((lane & 3) == 0 && r0 < n && out.spay_rank != nullptr) {
                out.spay_rank[static_cast<int64_t>(c) * ((n + 31) >> 5) + (r0 >> 5)] =
                    make_int2(static_cast<int>(jk), static_cast<int>(word));
            }
        }
        // The item loop runs as kScanHalves passes of IH = kScanItems / kScanHalves unrolled items (the same items in
        // the same order, the same operations): before the next pass the remaining items rotate into slots 0..IH-1, so every
        // array index stays a compile-time constant (registers) while the loop body is emitted once per pass width (a smaller
        // instruction footprint: the scan's cost on a cold L2 is mostly instruction fetch).
        constexpr int IH = kScanItems / kScanHalves;
#pragma unroll 1
        for (int h = 0; h < kScanHalves; ++h) {
            const int r0h = r0 + h * IH;
#pragma unroll
            for (int k = 0; k < IH; ++k) {
                // Peers of the group before this entry.
                const bool first = p[k] & kHead;
                const uint32_t k_before = first ? 0u : EL::kcount(run);
                const uint32_t q_before = first ? 0u : EL::qcount(run);
                // The hot test of every item is OR-ed per thread; the block votes once after the loop (same s_hot).
                hot_mine |= is_k(p[k]) && k_before + 1u >= static_cast<uint32_t>(recs.hot_g);
                run = EL::combine(run, item(k));
                const float hi = run.x * kLog2e;
                const float lo = fmaf(run.c, kLog2e, fmaf(run.x, kLog2e, -hi));
                float v = 0.0f;
                if (r0h + k < n) {
                    const uint32_t pk = p[k];
                    const bool valid = !(pk & kInvalid);
                    const int o = token_of(pk) * ((pk & kIsQ) ? R : W) + lane_of(pk);
                    const float fwd = run.x + run.c;
                    const int r = r0h + k;
                    const uint32_t after = (aft[k / 2] >> (16 * (k % 2))) & 0xFFFFu;   // exact
                    const uint32_t after_sat = min(after, kPosCountMax);
                    const uint32_t before = (pk & kIsQ) ? q_before : k_before;
                    const int32_t packed = static_cast<int32_t>(
                        valid ? static_cast<uint32_t>(r) | after_sat << kPosAfterShift |
                                    min(before + 1 + after_sat, kPosCountMax) << kPosSizeShift
                              : static_cast<uint32_t>(r));
                    // The entry's element of its kind's sorted row (a padding entry is a group of
                    // its own: FIRST | LAST, distance 0).
                    const int32_t sorted = static_cast<int32_t>(
                        (pk & kSortedEntryMask) |
                        (valid ? min(before, kSortedDistMax) << kSortedDistShift |
                                     (before == 0 ? kHead : 0u) | (after == 0 ? kLastK : 0u)
                               : kHead | kLastK));
                    const bool kind_first = static_cast<uint32_t>(sorted) & kHead;
                    // The sorted elements (spay_k / spay_q), the mover records and the tile / group
                    // indices, in the kinds' sorted orders, from the values this item already holds.
                    // The work both kinds do (value, exp of the log-cumsum, sorted element, tile index, slot, log-cumsum
                    // and scaled value) runs once per item for the whole warp, the kind choosing only addresses (nearly
                    // every warp holds both kinds). Per entry the same operations on the same operands.
                    const bool isq = (pk & kIsQ) != 0u;
                    v = valid ? (kScanHoistVals ? to_float(vals.get(k)) : to_float(isq ? qv_chunk[o] : kv_chunk[o])) : 0.0f;
                    const float ef = expf(fwd);
                    const uint32_t jx = isq ? jq : jk;         // the entry's index in its kind's sorted row
                    const int64_t kb = isq ? cr : cw;          // the chunk's base in its kind's [C, W] / [C, R] arrays
                    // g = valid group FIRSTs before jk = valid LASTs before jk, plus this entry's
                    // own group when it started earlier (a valid entry that is not its FIRST). Write entries only.
                    const int g = static_cast<int>(vlb) + ((valid && k_before != 0) ? 1 : 0);
                    {
                        int32_t* const sp = isq ? (fs_has<FS>(kFsSpayQ) ? out.spay_q : nullptr)
                                                : (fs_has<FS>(kFsSpayK) ? out.spay_k : nullptr);
                        if (sp != nullptr) st_rec(sp + kb + jx, sorted);
                    }
                    if (fs_has<FS>(kFsTiles)) {
                        // The first FIRST at or after jx: jx itself, or the one after its group (+ segb_k for write entries).
                        int* const tp = isq ? recs.tiles_q : recs.tiles_k;
                        int* const sb = isq ? nullptr : recs.segb_k;
                        int t;
                        if ((tp != nullptr || sb != nullptr) && tile_at(jx, t)) {
                            const int64_t tb = isq ? tq_base : tk_base;
                            if (tp != nullptr) {
                                st_rec(tp + tb + t, kind_first ? static_cast<int>(jx)
                                                               : min(static_cast<int>(jx + after + 1), isq ? nq : nk));
                            }
                            if (sb != nullptr) st_rec(sb + tb + t, g);
                        }
                    }
                    const int64_t slot = valid ? static_cast<int64_t>(key[k]) + s_id_base : 0;
                    put_kq<ALL>(stage, stage_mem, isq, kLogK, kLogQ, out.log_k + cw, out.log_q + cr, o, fwd);
                    put_kq<ALL>(stage, stage_mem, isq, kKEff, kQW, out.k_eff + cw, out.q_w + cr, o, from_float<ValT>(v * ef));
                    put_kq<ALL>(stage, stage_mem, isq, kPosK, kPosQ, P1_OR_NULL(out.pos_k, cw), P1_OR_NULL(out.pos_q, cr), o,
                                packed);
                    jq += isq ? 1u : 0u;
                    if (isq) {
                        if ((fs_has<FS>(kFsQrec) && recs.qrec != nullptr) || (fs_has<FS>(kFsQrec8) && recs.qrec8 != nullptr)) {
                            const float qs = valid ? __fmul_rn(v, ef) : 0.0f;
                            if (fs_has<FS>(kFsQrec) && recs.qrec != nullptr) {
                                st_rec(recs.qrec + cr + jx, make_int4(sorted, static_cast<int>(slot), __float_as_int(qs), 0));
                            }
                            if (fs_has<FS>(kFsQrec8) && recs.qrec8 != nullptr) {
                                st_rec(recs.qrec8 + cr + jx, make_int2(static_cast<int>(slot), __float_as_int(qs)));
                            }
                        }
                        if (!ALL && out.q_last != nullptr) {
                            const bool q_last = valid && after == 0;
                            const bool q_only = q_last && q_before == 0;
                            if (stage.staged(kQFlags)) {
                                reinterpret_cast<uint8_t*>(stage_mem + stage.offset[kQFlags])[o] =
                                    (q_last ? 1u : 0u) | (q_only ? 2u : 0u);
                            } else {
                                out.q_last[cr + o] = q_last;
                                out.q_only[cr + o] = q_only;
                            }
                        }
                    } else {
                        const bool last = pk & kLastK;
                        const ValT cpe_v = from_float<ValT>(expf(seg_value(
                            seg_combine(SegSum{run.x, run.c, 0u}, SegSum{rx[k], 0.0f, 0u}))));
                        if (fs_has<FS>(kFsDcGidx) && recs.dc_bpre != nullptr && (jk & 31u) == 0u) st_rec(recs.dc_bpre + bp_base + (jk >> 5), g);
                        if (fs_has<FS>(kFsDcGidx)) put<ALL>(stage, stage_mem, kDcKpos, P1_OR_NULL(recs.dc_kpos, cw), o,
                                 (valid && k_before == 0) ? g : -1);
                        vlb += last ? 1u : 0u;
                        if ((fs_has<FS>(kFsKrec) && recs.krec != nullptr) || (fs_has<FS>(kFsKspa) && recs.kspa != nullptr) ||
                            (fs_has<FS>(kFsDcRec) && recs.dc_rec != nullptr)) {
                            const float cpe_w = valid ? to_float(cpe_v) : 1.0f;
                            if (fs_has<FS>(kFsKrec) && recs.krec != nullptr) {
                                st_rec(recs.krec + cw + jk, make_int4(sorted, static_cast<int>(slot),
                                                                      __float_as_int(valid ? __fmul_rn(v, expf(rx[k])) : 0.0f),
                                                                      __float_as_int(cpe_w)));
                            }
                            if ((fs_has<FS>(kFsKspa) && recs.kspa != nullptr) || (fs_has<FS>(kFsDcRec) && recs.dc_rec != nullptr)) {
                                const float ps = valid ? __fmul_rn(-recs.beta[static_cast<int64_t>(c) * C + token_of(pk)],
                                                                   __fmul_rn(v, ef)) : 0.0f;
                                if (fs_has<FS>(kFsKspa) && recs.kspa != nullptr) st_rec(recs.kspa + cw + jk, ps);
                                if (fs_has<FS>(kFsDcRec) && recs.dc_rec != nullptr) {
                                    st_rec(recs.dc_rec + cw + jk, make_int4(sorted, static_cast<int>(slot), __float_as_int(ps),
                                                                            __float_as_int(cpe_w)));
                                }
                            }
                        }
                        ++jk;
                        put<ALL>(stage, stage_mem, kRex, out.rex + cw, o, rx[k]);
                        put<ALL>(stage, stage_mem, kCpe, out.cpe + cw, o, cpe_v);
                        if (!ALL) {
                            put<false, int8_t>(stage, stage_mem, kIsLast, P1_OR_NULL(out.is_last, cw), o, last);
                            put<false, int8_t>(stage, stage_mem, kIsOnly, P1_OR_NULL(out.is_only, cw), o,
                                               last && k_before == 0);
                        }
                    }
                }
                if (r0 < row) {
                    rec_row[r0h + k] = make_uint4(p[k], __float_as_uint(hi), __float_as_uint(lo),
                                                 __float_as_uint(v));
                }
            }
            if (h + 1 < kScanHalves) {
#pragma unroll
                for (int k = 0; k + IH < kScanItems; ++k) {
                    p[k] = p[k + IH];
                    x[k] = x[k + IH];
                    rx[k] = rx[k + IH];
                    key[k] = key[k + IH];
                }
#pragma unroll
                for (int k = 0; k + IH / 2 < kScanItems / 2; ++k) aft[k] = aft[k + IH / 2];
                vals.template rotate<IH>();
            }
        }
    }
    // One vote per thread for the whole pass (before every reader of s_hot: the barrier below / cluster barrier 3).
    if (recs.hot != nullptr && __any_sync(0xFFFFFFFFu, hot_mine) && lane == 0) atomicOr(&s_hot, 1u);   // block-uniform
    // The closing entries of the index rows: n, and the chunk's valid group count (cluster: the last CTA's counts).
    if (threadIdx.x == 0 && rk == CL - 1) {
        if (recs.tiles_k != nullptr) st_rec(recs.tiles_k + tk_base + ntk, nk);
        if (recs.segb_k != nullptr) st_rec(recs.segb_k + tk_base + ntk, static_cast<int>(kind_vl));
        if (recs.tiles_q != nullptr) st_rec(recs.tiles_q + tq_base + ntq, nq);
        if (recs.dc_bpre != nullptr) st_rec(recs.dc_bpre + bp_base + nwk, static_cast<int>(kind_vl));
    }
    int32_t* const pos_k_c = P1_OR_NULL(out.pos_k, cw);
    int32_t* const pos_q_c = P1_OR_NULL(out.pos_q, cr);
    int8_t* const is_last_c = P1_OR_NULL(out.is_last, cw);
    int8_t* const is_only_c = P1_OR_NULL(out.is_only, cw);
    int32_t* const dc_kpos_c = P1_OR_NULL(recs.dc_kpos, cw);
#undef P1_OR_NULL
    if constexpr (CL > 1) {
        if (rk < CL - 1) cl_wait();   // barrier 2
        cl_arrive();                  // barrier 3: every CTA's staged elements, flags and records are final
        cl_wait();
        if (recs.hot != nullptr && threadIdx.x == 0 && rk == 0) {
            uint32_t h = s_hot;
            for (int r = 1; r < CL; ++r) h |= *cl_map(&s_hot, static_cast<uint32_t>(r));
            recs.hot[c] = h;
            if (h == 0u && recs.w2_elig != nullptr) atomicAdd(recs.w2_elig, 1u);
        }
        if (stage.mask) {
            const char* other = cl_map(stage_mem, static_cast<uint32_t>(rk ^ 1));
            const int ncw = C * W, ncr = C * R;
            auto un = [&](Field f, auto* dst, int cnt) {
                using T = typename std::remove_pointer<decltype(dst)>::type;
                if (stage.staged(f) && dst != nullptr) {
                    block_copy_or(dst, reinterpret_cast<const T*>(stage_mem + stage.offset[f]),
                                  reinterpret_cast<const T*>(other + stage.offset[f]), cnt, rk, CL);
                }
            };
            un(kLogK, out.log_k + cw, ncw);
            un(kRex, out.rex + cw, ncw);
            un(kCpe, out.cpe + cw, ncw);
            un(kKEff, out.k_eff + cw, ncw);
            un(kPosK, pos_k_c, ncw);
            un(kDcKpos, dc_kpos_c, ncw);
            un(kLogQ, out.log_q + cr, ncr);
            un(kQW, out.q_w + cr, ncr);
            un(kPosQ, pos_q_c, ncr);
            if (ALL) {
                // the flags from the merged packed positions (block_flags' rule), element by element
                auto fl = [&](int8_t* last, int8_t* only, Field f, int cnt) {
                    const int32_t* pa = reinterpret_cast<const int32_t*>(stage_mem + stage.offset[f]);
                    const int32_t* pb = reinterpret_cast<const int32_t*>(other + stage.offset[f]);
                    const int per = (cnt + CL - 1) / CL;
                    const int e0 = min(rk * per, cnt), e1 = min(e0 + per, cnt);
                    for (int i = e0 + static_cast<int>(threadIdx.x); i < e1; i += blockDim.x) {
                        const uint32_t q = static_cast<uint32_t>(pa[i] | pb[i]);
                        const uint32_t size = (q >> kPosSizeShift) & kPosCountMax;
                        last[i] = size != 0 && ((q >> kPosAfterShift) & kPosCountMax) == 0;
                        only[i] = size == 1;
                    }
                };
                if (is_last_c != nullptr) fl(is_last_c, is_only_c, kPosK, ncw);
                if (out.q_last != nullptr) fl(out.q_last + cr, out.q_only + cr, kPosQ, ncr);
            } else {
                un(kIsLast, is_last_c, ncw);
                un(kIsOnly, is_only_c, ncw);
                if (stage.staged(kQFlags)) {
                    const uint8_t* fa = reinterpret_cast<const uint8_t*>(stage_mem + stage.offset[kQFlags]);
                    const uint8_t* fb = reinterpret_cast<const uint8_t*>(other + stage.offset[kQFlags]);
                    const int per = (ncr + CL - 1) / CL;
                    const int e0 = min(rk * per, ncr), e1 = min(e0 + per, ncr);
                    for (int i = e0 + static_cast<int>(threadIdx.x); i < e1; i += blockDim.x) {
                        const uint8_t v = fa[i] | fb[i];
                        out.q_last[cr + i] = v & 1;
                        out.q_only[cr + i] = v >> 1;
                    }
                }
            }
        }
        cl_arrive();   // barrier 4: no CTA leaves while another may still read its shared memory
        cl_wait();
        return;
    }
    if (stage.mask || recs.hot != nullptr) __syncthreads();
    if (recs.hot != nullptr && threadIdx.x == 0) {
        recs.hot[c] = s_hot;
        if (s_hot == 0u && recs.w2_elig != nullptr) atomicAdd(recs.w2_elig, 1u);
    }
    if (stage.mask) {
        const int ncw = C * W, ncr = C * R;
        unstage(stage, stage_mem, kLogK, out.log_k + cw, ncw);
        unstage(stage, stage_mem, kRex, out.rex + cw, ncw);
        unstage(stage, stage_mem, kCpe, out.cpe + cw, ncw);
        unstage(stage, stage_mem, kKEff, out.k_eff + cw, ncw);
        unstage(stage, stage_mem, kPosK, pos_k_c, ncw);
        unstage(stage, stage_mem, kDcKpos, dc_kpos_c, ncw);
        unstage(stage, stage_mem, kLogQ, out.log_q + cr, ncr);
        unstage(stage, stage_mem, kQW, out.q_w + cr, ncr);
        unstage(stage, stage_mem, kPosQ, pos_q_c, ncr);
        if (ALL) {
            if (is_last_c != nullptr) {
                const auto* pos_k = reinterpret_cast<const int32_t*>(stage_mem + stage.offset[kPosK]);
                block_flags(is_last_c, is_only_c, pos_k, ncw);
            }
            if (out.q_last != nullptr) {
                const auto* pos_q =
                    reinterpret_cast<const int32_t*>(stage_mem + stage.offset[kPosQ]);
                block_flags(out.q_last + cr, out.q_only + cr, pos_q, ncr);
            }
        } else {
            unstage(stage, stage_mem, kIsLast, is_last_c, ncw);
            unstage(stage, stage_mem, kIsOnly, is_only_c, ncw);
            if (stage.staged(kQFlags)) {
                const auto* flags =
                    reinterpret_cast<const uint8_t*>(stage_mem + stage.offset[kQFlags]);
                block_copy_flags(out.q_last + cr, out.q_only + cr, flags, ncr);
            }
        }
    }
}

// One block per chunk (and per product when the two triangles do not fit together).
// A[i, j] sums one term per K entry of token i and earlier K entry of token j < i in its
// group, QK[i, j] one per Q entry of token i and earlier K entry of token j <= i. Each cell
// has one writer thread that adds its terms in an order fixed by the input, so no atomics
// touch the sums:
// 1. A segmented scan over the sorted row counts, for every entry, the valid K entries
// before it in its group (a byte, saturating at 255) and marks the entries that have
// any (a bit per (token, lane); integer OR is order-free). The others add nothing.
// 2. One thread per (row i, triangle) walks the marked entries of token i in lane order:
// from the nearest earlier entry towards the group head, kWalkBatch records per load,
// until it has met that many K entries.
// 3. The triangles are written out.
// Each term is rounded (v * v_pos * decay) and added with a separate rounding (__fadd_rn: never
// contracted into an FMA).
constexpr int kWalkBatch = 4;
constexpr int kPairItems = 16;      // sorted entries per thread per tile of phase 1
constexpr uint32_t kKBeforeMax = 255;

template <bool LEAN>
__global__ void __launch_bounds__(kPairsThreads) p1_pairs_kernel(
    const uint32_t* __restrict__ pay, const uint4* __restrict__ rec,
    const int32_t* __restrict__ pos_k, const int32_t* __restrict__ pos_q, float* __restrict__ a,
    float* __restrict__ qk, int C, int W, int R, int row, bool split,
    const uint32_t* __restrict__ w2_flags) {
    // After the parallel-walk kernel (below), only the blocks it declined are computed here.
    if (w2_flags != nullptr && w2_flags[blockIdx.x] == 0u) return;
    extern __shared__ float tri[];
    __shared__ uint32_t warp_total[kPairsThreads / 32];
    __shared__ uint32_t warp_prefix[kPairsThreads / 32];
    const int c = split ? blockIdx.x >> 1 : blockIdx.x;
    const bool do_a = !split || !(blockIdx.x & 1);
    const bool do_qk = !split || (blockIdx.x & 1);
    const int n = C * (W + R);
    const int tri_size = C * (C + 1) / 2;
    const int ntri = split ? 1 : 2;
    const int qk_off = split ? 0 : tri_size;
    // Marks (rows of wk / rk words per token), then the K entries before each (token, lane)
    // in its group: the K rows (A), then the Q rows (QK). LEAN keeps no count bytes.
    const int wk = do_a ? (W + 31) >> 5 : 0;
    const int rk = do_qk ? (R + 31) >> 5 : 0;
    uint32_t* mask_k = reinterpret_cast<uint32_t*>(tri + ntri * tri_size);
    uint32_t* mask_q = mask_k + C * wk;
    uint8_t* kb_k = reinterpret_cast<uint8_t*>(mask_q + C * rk);
    uint8_t* kb_q = kb_k + (do_a ? C * W : 0);
    const int kb_bytes = LEAN ? 0 : (do_a ? C * W : 0) + (do_qk ? C * R : 0);
    const int kb_words = C * (wk + rk) + (kb_bytes + 3) / 4;
    for (int i = threadIdx.x; i < ntri * tri_size + kb_words; i += blockDim.x) {
        reinterpret_cast<uint32_t*>(tri)[i] = 0u;
    }
    __syncthreads();

    const int64_t base = static_cast<int64_t>(c) * row;
    const uint32_t* pay_row = pay + base;
    const uint4* rec_row = rec + base;
    // Phase 1. Scan element: a K count in bits 0-30, bit 31 = a segment (group) starts here.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    uint32_t carry = 0;
    for (int t0 = 0; t0 < n; t0 += blockDim.x * kPairItems) {
        const int r0 = t0 + threadIdx.x * kPairItems;
        uint32_t p[kPairItems];
        if (r0 < n) load_items(pay_row + r0, p);   // rows are padded to kRowAlign >= 16
        uint32_t own = 0;
#pragma unroll
        for (int k = 0; k < kPairItems; ++k) {
            if (r0 + k >= n) p[k] = kHead | kInvalid;
            own = is_head(p[k]) ? (kHead | (is_k(p[k]) ? 1u : 0u)) : own + (is_k(p[k]) ? 1u : 0u);
        }
        // Exclusive scan of the threads' elements (combine: a later segment start wins).
        uint32_t incl = own;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, incl, d);
            if (lane >= d && !(incl & kHead)) incl = (other & ~kHead) + incl | (other & kHead);
        }
        if (lane == 31) warp_total[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            uint32_t v = lane < warps ? warp_total[lane] : 0u;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, v, d);
                if (lane >= d && !(v & kHead)) v = (other & ~kHead) + v | (other & kHead);
            }
            const uint32_t earlier = __shfl_up_sync(0xFFFFFFFFu, v, 1);
            if (lane < warps) {
                uint32_t before = carry;
                if (lane > 0) before = (earlier & kHead) ? earlier : (carry & ~kHead) + earlier;
                warp_prefix[lane] = before;
            }
            // Lane 0 read warp_total[0] above: the warp's lanes are ordered only by this barrier
            // (shuffles carry no memory ordering), so the overwrite comes after it.
            __syncwarp();
            if (lane == warps - 1) warp_total[0] = (v & kHead) ? v : (carry & ~kHead) + v;
        }
        __syncthreads();
        const uint32_t wpre = warp_prefix[warp];
        const uint32_t lpre = __shfl_up_sync(0xFFFFFFFFu, incl, 1);
        uint32_t run = lane == 0 ? wpre : ((lpre & kHead) ? lpre : (wpre & ~kHead) + lpre);
        carry = warp_total[0];
        __syncthreads();   // warp_total is rewritten by the next tile
        uint32_t count = run & ~kHead;   // K entries before r0 in r0's group (if r0 continues it)
#pragma unroll
        for (int k = 0; k < kPairItems; ++k) {
            const uint32_t pk = p[k];
            if (is_head(pk)) count = 0;
            if (r0 + k < n && count > 0) {
                const int o = token_of(pk) * ((pk & kIsQ) ? R : W) + lane_of(pk);
                const uint8_t kb = static_cast<uint8_t>(min(count, kKBeforeMax));
                const uint32_t bit = 1u << (lane_of(pk) & 31);
                if (is_k(pk) && do_a) {
                    if (!LEAN) kb_k[o] = kb;
                    atomicOr(mask_k + token_of(pk) * wk + (lane_of(pk) >> 5), bit);
                }
                if (is_q(pk) && do_qk) {
                    if (!LEAN) kb_q[o] = kb;
                    atomicOr(mask_q + token_of(pk) * rk + (lane_of(pk) >> 5), bit);
                }
            }
            count += is_k(pk) ? 1u : 0u;
        }
    }
    __syncthreads();

    // Phase 2. Units: (row, A) and (row, QK) interleaved, or the rows of the block's triangle.
    const int units = (do_a && do_qk) ? 2 * C : C;
    for (int u = threadIdx.x; u < units; u += blockDim.x) {
        const bool q = (do_a && do_qk) ? (u & 1) : do_qk;
        const int i = (do_a && do_qk) ? u >> 1 : u;
        float* trow = tri + (q ? qk_off : 0) + i * (i + 1) / 2;
        const int words = q ? rk : wk;
        const uint32_t* mrow = q ? mask_q + i * rk : mask_k + i * wk;
        const uint8_t* kbrow = q ? kb_q + i * R : kb_k + i * W;
        const int32_t* prow = q ? pos_q + (static_cast<int64_t>(c) * C + i) * R
                                : pos_k + (static_cast<int64_t>(c) * C + i) * W;
        for (int m = 0; m < words; ++m) {
            for (uint32_t bits = mrow[m]; bits; bits &= bits - 1) {
                const int e = m * 32 + __ffs(bits) - 1;
                // LEAN: no count, so the walk runs to the group's head; past the entry's last
                // earlier K entry there are only read entries, so it adds the same terms.
                const uint32_t need = LEAN ? kKBeforeMax : kbrow[e];
                const int r = prow[e] & kPosMask;
                const uint4 own = rec_row[r];
                const float hi = __uint_as_float(own.y), lo = __uint_as_float(own.z);
                const float v = __uint_as_float(own.w);
                uint32_t met = 0;
                bool done = false;
                for (int top = r - 1; !done; top -= kWalkBatch) {
                    // The records of the next kWalkBatch earlier positions, loaded together; the
                    // ones past the walk's end (clamped at 0) are not used.
                    uint4 rb[kWalkBatch];
    #pragma unroll
                    for (int k = 0; k < kWalkBatch; ++k) rb[k] = rec_row[max(top - k, 0)];
    #pragma unroll
                    for (int k = 0; k < kWalkBatch; ++k) {
                        if (done) break;
                        const uint32_t pp = rb[k].x;
                        if (is_k(pp)) {
                            const int j = token_of(pp);
                            if (q || j != i) {
                                const float term =
                                    v * __uint_as_float(rb[k].w) *
                                    exp2f(log2_decay(hi, lo, __uint_as_float(rb[k].y),
                                                     __uint_as_float(rb[k].z)));
                                trow[j] = __fadd_rn(trow[j], term);
                            }
                            ++met;
                        }
                        done = is_head(pp) || (need < kKBeforeMax && met == need);
                    }
                }
            }
        }
    }
    __syncthreads();

    float* tri_a = tri;
    float* tri_qk = tri + qk_off;
    const int64_t out = static_cast<int64_t>(c) * C * C;
    for (int e = threadIdx.x; e < C * C; e += blockDim.x) {
        const int i = e / C;
        const int j = e - i * C;
        const int t = i * (i + 1) / 2 + j;
        if (do_a) a[out + e] = j < i ? tri_a[t] : 0.0f;
        if (do_qk) qk[out + e] = j <= i ? tri_qk[t] : 0.0f;
    }
}

// ---------------------------------------------------------------------------------------
// Parallel walks (p1_pairs_w2_kernel): the ordered pair kernel's sums with the SAME bits, its walks done in parallel.
// The ordered kernel adds, per cell (i, j), the terms of the marked entries of token i in lane
// order, each entry's walk nearest-first, every term with __fadd_rn onto 0.0f; one thread
// per (row, triangle) does all of it serially (a chain of dependent global loads). This kernel keeps
// every term and the per-cell order, and only moves the walks out of that chain:
// 1. as the ordered kernel: per sorted position the K entries before it in its group, a
// count byte per (token, lane) of the block's triangles; and per 32 lanes of a (row,
// triangle) the sum of those bytes (integer shared-memory atomics: order-free).
// 2. an exclusive scan of the word sums in (row, triangle, lane) order gives every marked
// entry its slots: one per K entry its walk meets, in the walk's order.
// 3. a sweep over the sorted row: every marked entry walks (all threads in parallel) exactly
// as the ordered kernel does and stores each term, with its column, in its next slot (an
// A walk's own-token K entries take a slot too; step 4 skips them, as the ordered kernel).
// 4. one thread per (row, triangle) adds its slots in order: per cell exactly the ordered
// kernel's sequence of __fadd_rn.
// 5. the triangles are written out.
// A block whose walks need more slots than its buffer holds, or that holds a count of 255 or
// more (the ordered kernel then walks to the group head), DECLINES: it writes no output and
// marks its block in `flags`; the fallback kernel launched after it (the ordered kernel)
// computes exactly those blocks. Whether a block declines never changes a bit.
constexpr int kW2Threads = 512;
constexpr uint32_t kW2Declined = 2u;
// With the hot/cold kernel: chunks whose groups all hold fewer than kW2HcG valid K entries go to the
// parallel-walk kernel. kW2HcG must be at most kHcG: the hot/cold kernel has no hot candidate on such a chunk,
// so both give the ordered walk's bits; the value only moves work between the two kernels.
constexpr int kW2HcG = 8;

// The sum of the count bytes row[from, to), to - from < 32.
__device__ __forceinline__ uint32_t w2_kb_sum(const uint8_t* row, int from, int to) {
    uint32_t s = 0;
    const uint8_t* p = row + from;
    int cnt = to - from;
    if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0u) {
        // Whole words, then the low bytes of the next one (the bytes past `to` it reads are later count
        // bytes or the count region's padding: written before the first barrier only).
        const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
        const int q = cnt >> 2;
        for (int t = 0; t < q; ++t) s = __dp4a(w[t], 0x01010101u, s);
        if (cnt & 3) s = __dp4a(w[q] & ((1u << (8 * (cnt & 3))) - 1u), 0x01010101u, s);
        return s;
    }
    for (int t = 0; t < cnt; ++t) s += p[t];
    return s;
}

// One work item b (a block of the ordered kernel's grid: chunk b, or chunk b / 2 and one triangle
// when split) with the whole block; returns block-uniformly (also when it declines).
__device__ __forceinline__ void w2_item(
    int b, char* w2mem, uint32_t* warp_total, uint32_t* warp_prefix, uint32_t& s_sat,
    const uint32_t* __restrict__ pay, const uint4* __restrict__ rec, float* __restrict__ a,
    float* __restrict__ qk, uint32_t* __restrict__ flags, int C, int W, int R, int row,
    bool split, int cap) {
    const int c = split ? b >> 1 : b;
    const bool do_a = !split || !(b & 1);
    const bool do_qk = !split || (b & 1);
    const int n = C * (W + R);
    const int tri_size = C * (C + 1) / 2;
    const int ntri = split ? 1 : 2;
    const int qk_off = split ? 0 : tri_size;
    const int wk = do_a ? (W + 31) >> 5 : 0;   // count-byte words (32 lanes) per row, A / QK
    const int rk = do_qk ? (R + 31) >> 5 : 0;
    const int wpr = wk + rk;
    const int nwords = C * wpr;
    const int nwm = (n + 31) >> 5;   // marked-entry bit words (one bit per sorted position)
    // Layout: triangles | word sums -> slot bases [nwords + 1] | marks [nwm] | terms [cap] | count bytes |
    // columns [cap]
    float* tri = reinterpret_cast<float*>(w2mem);
    uint32_t* wsum = reinterpret_cast<uint32_t*>(tri + ntri * tri_size);
    uint32_t* markw = wsum + nwords + 1;
    float* terms = reinterpret_cast<float*>(markw + nwm);
    uint8_t* kb_k = reinterpret_cast<uint8_t*>(terms + cap);
    uint8_t* kb_q = kb_k + (do_a ? C * W : 0);
    const int kb_bytes = (do_a ? C * W : 0) + (do_qk ? C * R : 0);
    // Columns start at the next 4-byte boundary: w2_kb_sum's masked last-word read of a count row stays
    // within the count bytes and their padding (all written before the first barrier only).
    uint8_t* tcol = kb_k + ((kb_bytes + 3) & ~3);
    for (int i = threadIdx.x; i < ntri * tri_size + nwords + 1 + nwm; i += blockDim.x) {
        reinterpret_cast<uint32_t*>(tri)[i] = 0u;
    }
    for (int i = threadIdx.x; i < (kb_bytes + 3) / 4; i += blockDim.x) {
        reinterpret_cast<uint32_t*>(kb_k)[i] = 0u;
    }
    if (threadIdx.x == 0) s_sat = 0u;
    __syncthreads();

    const int64_t base = static_cast<int64_t>(c) * row;
    const uint32_t* pay_row = pay + base;
    const uint4* rec_row = rec + base;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    // 1. Counts: the ordered kernel's phase 1 (same scan, same counts), plus the word sums.
    bool sat = false;
    uint32_t carry = 0;
    for (int t0 = 0; t0 < n; t0 += blockDim.x * kPairItems) {
        const int r0 = t0 + threadIdx.x * kPairItems;
        uint32_t p[kPairItems];
        if (r0 < n) load_items(pay_row + r0, p);   // rows are padded to kRowAlign >= 16
        uint32_t own = 0;
#pragma unroll
        for (int k = 0; k < kPairItems; ++k) {
            if (r0 + k >= n) p[k] = kHead | kInvalid;
            own = is_head(p[k]) ? (kHead | (is_k(p[k]) ? 1u : 0u)) : own + (is_k(p[k]) ? 1u : 0u);
        }
        uint32_t incl = own;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, incl, d);
            if (lane >= d && !(incl & kHead)) incl = (other & ~kHead) + incl | (other & kHead);
        }
        if (lane == 31) warp_total[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            uint32_t v = lane < warps ? warp_total[lane] : 0u;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, v, d);
                if (lane >= d && !(v & kHead)) v = (other & ~kHead) + v | (other & kHead);
            }
            const uint32_t earlier = __shfl_up_sync(0xFFFFFFFFu, v, 1);
            if (lane < warps) {
                uint32_t before = carry;
                if (lane > 0) before = (earlier & kHead) ? earlier : (carry & ~kHead) + earlier;
                warp_prefix[lane] = before;
            }
            __syncwarp();
            if (lane == warps - 1) warp_total[0] = (v & kHead) ? v : (carry & ~kHead) + v;
        }
        __syncthreads();
        const uint32_t wpre = warp_prefix[warp];
        const uint32_t lpre = __shfl_up_sync(0xFFFFFFFFu, incl, 1);
        uint32_t run = lane == 0 ? wpre : ((lpre & kHead) ? lpre : (wpre & ~kHead) + lpre);
        carry = warp_total[0];
        __syncthreads();   // warp_total is rewritten by the next tile
        uint32_t count = run & ~kHead;   // K entries before r0 in r0's group (if r0 continues it)
        uint32_t mk16 = 0;               // this thread's marked positions (r0 + k)
#pragma unroll
        for (int k = 0; k < kPairItems; ++k) {
            const uint32_t pk = p[k];
            if (is_head(pk)) count = 0;
            if (r0 + k < n && count > 0) {
                const int i = token_of(pk), l = lane_of(pk);
                const uint32_t kb = min(count, kKBeforeMax);
                if (is_k(pk) && do_a) {
                    kb_k[i * W + l] = static_cast<uint8_t>(kb);
                    atomicAdd(wsum + i * wpr + (l >> 5), kb);
                    sat |= count >= kKBeforeMax;
                    mk16 |= 1u << k;
                }
                if (is_q(pk) && do_qk) {
                    kb_q[i * R + l] = static_cast<uint8_t>(kb);
                    atomicAdd(wsum + i * wpr + wk + (l >> 5), kb);
                    sat |= count >= kKBeforeMax;
                    mk16 |= 1u << k;
                }
            }
            count += is_k(pk) ? 1u : 0u;
        }
        // r0 is a multiple of 16: an even thread and the next one hold one 32-position mark word.
        const uint32_t mk_hi = __shfl_down_sync(0xFFFFFFFFu, mk16, 1);
        if (!(threadIdx.x & 1) && (r0 >> 5) < nwm) markw[r0 >> 5] = mk16 | mk_hi << 16;
    }
    if (sat) atomicOr(&s_sat, 1u);
    __syncthreads();

    // 2. Slot bases: exclusive scan of the word sums in word order; wsum[nwords] = the total.
    {
        const int per = (nwords + static_cast<int>(blockDim.x) - 1) / static_cast<int>(blockDim.x);
        const int w0 = min(static_cast<int>(threadIdx.x) * per, nwords), w1 = min(w0 + per, nwords);
        uint32_t s = 0;
        for (int w = w0; w < w1; ++w) s += wsum[w];
        uint32_t incl = s;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, incl, d);
            if (lane >= d) incl += other;
        }
        if (lane == 31) warp_total[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            const uint32_t v = lane < warps ? warp_total[lane] : 0u;
            uint32_t vi = v;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, vi, d);
                if (lane >= d) vi += other;
            }
            if (lane < warps) warp_prefix[lane] = vi - v;
        }
        __syncthreads();
        uint32_t at = warp_prefix[warp] + incl - s;
        for (int w = w0; w < w1; ++w) {
            const uint32_t x = wsum[w];
            wsum[w] = at;
            at += x;
        }
        if (threadIdx.x == blockDim.x - 1) wsum[nwords] = at;
        __syncthreads();
    }
    if (s_sat != 0u || wsum[nwords] > static_cast<uint32_t>(cap)) {   // block-uniform
        if (threadIdx.x == 0) flags[b] = kW2Declined;
        return;
    }
    if (threadIdx.x == 0) flags[b] = 0u;

    // 3. Walks, in parallel: each marked entry fills its slots in its walk's order. A warp takes
    // 1024 sorted positions (32 mark words) at a time and deals their marked entries to its lanes
    // in sorted order, 32 per round (no per-entry filtering; neighbouring lanes, neighbouring
    // records). Which thread walks an entry never changes what it writes.
    for (int w0 = warp * 32; w0 < nwm; w0 += warps * 32) {
        const int wl = w0 + lane;
        const uint32_t bits = wl < nwm ? markw[wl] : 0u;
        uint32_t incl = __popc(bits);
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t other = __shfl_up_sync(0xFFFFFFFFu, incl, d);
            if (lane >= d) incl += other;
        }
        const uint32_t excl = incl - __popc(bits);
        const uint32_t total = __shfl_sync(0xFFFFFFFFu, incl, 31);
        for (uint32_t m0 = 0; m0 < total; m0 += 32) {
            const uint32_t m = m0 + lane;
            // The word holding marked entry m: the number of lanes whose inclusive count is <= m.
            int o = 0;
#pragma unroll
            for (int step = 16; step > 0; step >>= 1) {
                if (__shfl_sync(0xFFFFFFFFu, incl, o + step - 1) <= m) o += step;
            }
            const uint32_t ob = __shfl_sync(0xFFFFFFFFu, bits, o & 31);
            const uint32_t oe = __shfl_sync(0xFFFFFFFFu, excl, o & 31);
            if (m >= total) continue;
            // Its bit: the (m - oe)-th set bit of the word.
            uint32_t q = m - oe, bb = ob;
            int bit = 0;
#pragma unroll
            for (int width = 16; width > 0; width >>= 1) {
                const uint32_t c = __popc(bb & ((1u << width) - 1u));
                if (q >= c) {
                    q -= c;
                    bb >>= width;
                    bit += width;
                }
            }
            const int r = (w0 + o) * 32 + bit;
            const uint32_t pk = pay_row[r];
            const bool kq = is_q(pk) && do_qk;
            const int i = token_of(pk), l = lane_of(pk);
            const uint8_t* kbrow = kq ? kb_q + i * R : kb_k + i * W;
            const uint32_t need = kbrow[l];
            uint32_t s = wsum[i * wpr + (kq ? wk : 0) + (l >> 5)] + w2_kb_sum(kbrow, l & ~31, l);
            const uint4 own = rec_row[r];
            const float hi = __uint_as_float(own.y), lo = __uint_as_float(own.z);
            const float v = __uint_as_float(own.w);
            uint32_t met = 0;
            bool done = false;
            for (int top = r - 1; !done; top -= kWalkBatch) {
                uint4 rb[kWalkBatch];
#pragma unroll
                for (int u = 0; u < kWalkBatch; ++u) rb[u] = rec_row[max(top - u, 0)];
#pragma unroll
                for (int u = 0; u < kWalkBatch; ++u) {
                    if (done) break;
                    const uint32_t pp = rb[u].x;
                    if (is_k(pp)) {
                        const int j = token_of(pp);
                        float term = 0.0f;   // an A walk's own-token K entry: met, not added
                        if (kq || j != i) {
                            term = v * __uint_as_float(rb[u].w) *
                                   exp2f(log2_decay(hi, lo, __uint_as_float(rb[u].y),
                                                    __uint_as_float(rb[u].z)));
                        }
                        terms[s] = term;
                        tcol[s] = static_cast<uint8_t>(j);
                        ++s;
                        ++met;
                    }
                    done = is_head(pp) || met == need;
                }
            }
        }
    }
    __syncthreads();

    // 4. Per (row, triangle): its slots in order (lane order, each walk nearest first).
    const int units = (do_a && do_qk) ? 2 * C : C;
    for (int u = threadIdx.x; u < units; u += blockDim.x) {
        const bool q = (do_a && do_qk) ? (u & 1) : do_qk;
        const int i = (do_a && do_qk) ? u >> 1 : u;
        float* trow = tri + (q ? qk_off : 0) + i * (i + 1) / 2;
        const int w_beg = i * wpr + (q ? wk : 0);
        const uint32_t s1 = wsum[w_beg + (q ? rk : wk)];
        for (uint32_t s = wsum[w_beg]; s < s1; ++s) {
            const int col = tcol[s];
            if (q || col != i) trow[col] = __fadd_rn(trow[col], terms[s]);   // A: j == i is not a pair
        }
    }
    __syncthreads();

    // 5. Out (as the ordered kernel).
    float* tri_a = tri;
    float* tri_qk = tri + qk_off;
    const int64_t out = static_cast<int64_t>(c) * C * C;
    for (int e = threadIdx.x; e < C * C; e += blockDim.x) {
        const int i = e / C;
        const int j = e - i * C;
        const int t = i * (i + 1) / 2 + j;
        if (do_a) a[out + e] = j < i ? tri_a[t] : 0.0f;
        if (do_qk) qk[out + e] = j <= i ? tri_qk[t] : 0.0f;
    }
}

// Persistent: a fixed grid (from the shapes) claims work items in order through `claim` (zeroed by
// the scan kernel before this launch); hc_mode skips the chunks the scan marked hot (the hot/cold
// kernel computes them). Which block computes an item never changes its bits.
constexpr int kW2Grid = 1024;
// In hc_mode this kernel runs only when at least this many chunks lack a hot mark (else the hot/cold kernel
// computes them all). It never changes a bit.
constexpr int kW2MinItems = 256;
__global__ void __launch_bounds__(kW2Threads, 2) p1_pairs_w2_kernel(
    const uint32_t* __restrict__ pay, const uint4* __restrict__ rec, float* __restrict__ a,
    float* __restrict__ qk, uint32_t* __restrict__ flags, uint32_t* __restrict__ claim, int items,
    int C, int W, int R, int row, bool split, int cap, bool hc_mode, const uint32_t* __restrict__ elig,
    int min_items) {
    // hc_mode: with fewer than min_items unmarked chunks, this kernel would only add its one-wave latency to the pair
    // phase (the hot/cold kernel fills its waves with them at little cost): hand them all to that kernel (flag
    // kW2Declined; it gives the same bits on a chunk without a hot candidate) and claim nothing. The count is final:
    // the scan ran first. Not when no chunk is marked (count == items): the hot/cold kernel then has nothing else to
    // do, and its one-CTA-per-SM cold path is slower on all-cold short rows. Which chunks this kernel takes never
    // changes a bit.
    if (hc_mode && elig != nullptr && *elig < static_cast<uint32_t>(min(min_items, items))) {   // grid-uniform
        for (int b = blockIdx.x * kW2Threads + threadIdx.x; b < items; b += gridDim.x * kW2Threads)
            if (flags[b] == 0u) flags[b] = kW2Declined;
        return;
    }
    extern __shared__ __align__(16) char w2mem[];
    __shared__ uint32_t warp_total[kW2Threads / 32];
    __shared__ uint32_t warp_prefix[kW2Threads / 32];
    __shared__ uint32_t s_sat;
    __shared__ int s_item;
    for (;;) {
        if (threadIdx.x == 0) {
            const uint32_t got = atomicAdd(claim, 1u);
            s_item = got < static_cast<uint32_t>(items) ? static_cast<int>(got) : -1;
        }
        __syncthreads();
        const int b = s_item;
        if (b < 0) break;   // block-uniform
        if (!(hc_mode && flags[b] != 0u)) {   // block-uniform (one flag per item)
            w2_item(b, w2mem, warp_total, warp_prefix, s_sat, pay, rec, a, qk, flags, C, W, R, row, split, cap);
        }
        __syncthreads();   // the item's shared memory and s_item are reused by the next one
    }
}

// ---------------------------------------------------------------------------------------
// Pair sums, hot/cold kernel (p1_pairs_hc_kernel): hot slots dense, cold slots by a packed walk.
// One block per chunk, kHcThreads threads, C <= kHcMaxC, rows <= kHcMaxRow, compute capability
// >= 9.0 (the layout needs up to ~210 KB of shared memory); dispatch depends on shapes only.
// 1. Classification (ballots over 32-position rows, carries across rows / warps / tiles): for
// each sorted position the K entries and entries since its group's head, K ranks, the K
// records (hi, lo, v, token) staged in K-rank order, and (token, lane) -> position. Groups
// with >= kHcG K entries are hot candidates, in sorted order.
// 2. Hot slots, kHcKB per batch (one warp each), validated or DEMOTED to the exact walk
// (non-finite values, nonzero |v| outside [2^-100, 2^32], a factor below FLT_MIN, a
// positive gate, a repeated K or Q slot of one token). A slot whose log2 span is <= 2 takes
// one reference; others take per-(slot, 16-token block) references ref(I) = the slot's
// log-cumsum just before block I:
// a  = v_K 2^(L - ref(I))     aq = v_Q 2^(L_Q - ref(I))     (rows, <= |v|)
// bd = v_K 2^(ref(I) - L)     (columns inside the row's block; only if the block's own
// log2 range is <= 2, else the block is WIDE)
// bo = v_K 2^(ref(J+1) - L)   (columns of earlier blocks, <= |v|)
// D[I][J] = 2^(ref(I) - ref(J+1)) <= 1 for J < I.
// Every thread owns fixed 4x4 tiles of A and QK in registers and adds, slot by slot in
// sorted-slot order, a*bd (J == I) or a*(D*bo) (J < I), as __fadd_rn(acc, __fmul_rn(a, b))
// (kHcRoundedProducts; false: fmaf). This order and factorization differ from the
// ordered kernel's: results are not bitwise equal to it. QK tiles skip row
// blocks without Q members of the slot (exact zeros). The QK diagonal (a Q entry and its own
// token's K entry, decay exactly 1) is summed as v_Q*v_K per token in slot order. A WIDE
// block's own pairs (its entries' nearest earlier K members in the block) take step 3.
// 3. Exact walk (cold slots, demoted slots, wide blocks' own pairs): a warp owns a row
// (token i, A or QK); one lane per (entry, earlier K member) pair, the term computed as
// the ordered kernel does (v * v_p * exp2f(decay), rounded products), added after the
// dense sum in (entry lane, nearest member first) order; lanes of one cell are added in
// lane order by one lane. Rows go to warps dynamically: a row is only touched by its warp.
// 4. The triangles are written out (zeros above the diagonal).
// Every cell's additions happen in an order fixed by the input, so results are bitwise
// reproducible and independent of scheduling; the geometry is fixed at compile time.
// Compile-time configuration (the dispatch depends on shapes only).
constexpr int kHcG = 16;                     // hot threshold: groups with >= kHcG K entries go dense
constexpr bool kHcRoundedProducts = true;    // dense terms as __fadd_rn(acc, __fmul_rn(a, b)) (false: fmaf)
constexpr bool kHcProfiling = false;         // true: per-chunk phase clocks into `prof` when the host passes a buffer (it passes none)
constexpr int kHcColdRows = 1;               // exact walk: rows a warp walks in lockstep
constexpr size_t kHcSmemBudget = 227 * 1024 - 4096;   // dynamic shared memory the hot/cold layout may take
constexpr int kHcThreads = 512;              // 16 warps: 4 per register partition -> 128 registers
constexpr int kHcWarps = kHcThreads / 32;
constexpr int kHcBlk = 16;
constexpr int kHcMaxC = 128;
constexpr int kHcMaxNB = kHcMaxC / kHcBlk;   // 8
constexpr int kHcKB = kHcWarps;              // hot slots per batch (one warp each)
constexpr int kHcMaxRow = 32752;               // entries per group then fit 15 bits
constexpr int kHcMaxCand = 1024;
constexpr uint8_t kHcHot = 0x80, kHcCand = 0x40, kHcCount = 0x3F;
// Accuracy: a factor 2^(L - ref) carries an exponent error of about ulp(|L - ref|), so the
// row and column factors of a block's own pairs (and of a single-reference slot) keep their
// exponents within kHcMaxRange; cross-block factors have exponents bounded by the pair's own
// decay, as in the direct evaluation. Values: nonzero |v| in [2^-100, 2^32], and a factor of a
// nonzero value below FLT_MIN demotes the slot. A cross-block product D*bo may still round below
// FLT_MIN; the term it carries, a * D * bo, is then below max|a| * FLT_MIN <= 2^32 * 2^-126 =
// 2^-94 in absolute value (an absolute bound, not a relative one).
constexpr float kHcMaxRange = 2.0f;          // log2 units
constexpr float kHcSingleRange = 2.0f;       // log2 units: one reference per slot
constexpr float kHcMaxAbsV = 4294967296.0f;  // 2^32
constexpr float kHcMinAbsV = 7.888609052210118e-31f;   // 2^-100

// Shared-memory layout (byte offsets), computed on the host.
struct HcLayout {
    int kbh;      // uint8 [row]: hot / candidate bits, cold K-before count
    int kmask;    // uint32 [row / 32]: K entries among 32 positions
    int kbef;     // uint32 [row / 32]: K entries before them
    int kst;      // float [3][C * W]: hi, lo, v of each K entry, in sorted (K rank) order
    int ktok;     // uint8 [C * W]: its token
    int posmap;   // uint16 [C * W + C * R]: sorted position of each (token, lane) K, then Q entry
    int cand;     // uint32 [kHcMaxCand]: (start | end << 16) of each candidate group
    int refs;     // float2 [kHcKB][kHcMaxNB + 1]: references of the batch's slots
    int valid;    // uint32 [kHcKB]
    int dtab;     // float [kHcKB][kHcMaxNB][kHcMaxNB]
    int uni;      // union: panels (5 x kHcKB x kHcMaxC floats) / triangles (2 x tri floats)
    int diag;     // float [kHcMaxC]: exact hot QK diagonal per token
    int misc;     // uint32 [64]: scan scratch, counters
    int total;
};

__device__ __forceinline__ uint32_t hc_combine(uint32_t a, uint32_t b) {
    // bits 0-15: K entries, bits 16-30: entries, bit 31: a group starts here
    return (b & kHead) ? b : a + b;
}

// Exclusive block scan (segmented, hc_combine) of one element per thread, plus a plain
// exclusive block sum of a second value; carries advance across tiles.
__device__ __forceinline__ void hc_block_scan(uint32_t own, uint32_t own_sum, uint32_t& excl,
                                              uint32_t& excl_sum, uint32_t& carry,
                                              uint32_t& carry_sum, uint32_t* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t incl = own, incl_sum = own_sum;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t o = __shfl_up_sync(0xFFFFFFFFu, incl, d);
        const uint32_t os = __shfl_up_sync(0xFFFFFFFFu, incl_sum, d);
        if (lane >= d) {
            incl = hc_combine(o, incl);
            incl_sum += os;
        }
    }
    if (lane == 31) {
        scratch[warp] = incl;
        scratch[32 + warp] = incl_sum;
    }
    __syncthreads();
    if (warp == 0) {
        uint32_t v = lane < kHcWarps ? scratch[lane] : 0u;
        uint32_t vs = lane < kHcWarps ? scratch[32 + lane] : 0u;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t o = __shfl_up_sync(0xFFFFFFFFu, v, d);
            const uint32_t os = __shfl_up_sync(0xFFFFFFFFu, vs, d);
            if (lane >= d) {
                v = hc_combine(o, v);
                vs += os;
            }
        }
        const uint32_t ev = __shfl_up_sync(0xFFFFFFFFu, v, 1);
        const uint32_t evs = __shfl_up_sync(0xFFFFFFFFu, vs, 1);
        __syncwarp();
        if (lane < kHcWarps) {
            scratch[lane] = lane == 0 ? carry : hc_combine(carry, ev);
            scratch[32 + lane] = carry_sum + (lane == 0 ? 0u : evs);
        }
        if (lane == kHcWarps - 1) {
            scratch[62] = hc_combine(carry, v);
            scratch[63] = carry_sum + vs;
        }
    }
    __syncthreads();
    const uint32_t wpre = scratch[warp], wpre_s = scratch[32 + warp];
    const uint32_t lpre = __shfl_up_sync(0xFFFFFFFFu, incl, 1);
    const uint32_t lpre_s = __shfl_up_sync(0xFFFFFFFFu, incl_sum, 1);
    excl = lane == 0 ? wpre : hc_combine(wpre, lpre);
    excl_sum = wpre_s + (lane == 0 ? 0u : lpre_s);
    carry = scratch[62];
    carry_sum = scratch[63];
    __syncthreads();   // scratch is rewritten by the next call
}

__device__ __forceinline__ int hc_krank(const uint32_t* kmask, const uint32_t* kbef, int r) {
    return static_cast<int>(kbef[r >> 5]) + __popc(kmask[r >> 5] & ((1u << (r & 31)) - 1u));
}

// The last candidate whose start is <= r (r lies in a candidate group).
__device__ __forceinline__ int hc_cand_of(const uint32_t* cand, int nc, int r) {
    int lo = 0, hi = nc - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (static_cast<int>(cand[mid] & 0xFFFFu) <= r) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// One CTA per SM (its shared memory is ~210 KB at C = 128). 16 warps = 4 per 16K-register
// partition, so 128 registers per thread (18 warps would be 5/5/4/4: at most 96, and spills).
#define HC_KERNEL_QUAL __launch_bounds__(kHcThreads, 1)
__global__ void HC_KERNEL_QUAL p1_pairs_hc_kernel(
    const uint32_t* __restrict__ pay, const uint4* __restrict__ rec,
    const int32_t* __restrict__ pos_k, const int32_t* __restrict__ pos_q, float* __restrict__ a,
    float* __restrict__ qk, int C, int W, int R, int row, int G, HcLayout L,
    long long* __restrict__ prof, const uint32_t* __restrict__ w2_flags) {
    // In hc_mode, after the parallel-walk kernel: only the hot chunks and the ones it declined are computed here.
    if (w2_flags != nullptr && w2_flags[blockIdx.x] == 0u) return;
    extern __shared__ __align__(16) char smem[];
    uint8_t* kbh = reinterpret_cast<uint8_t*>(smem + L.kbh);
    uint32_t* kmask = reinterpret_cast<uint32_t*>(smem + L.kmask);
    uint32_t* kbef = reinterpret_cast<uint32_t*>(smem + L.kbef);
    float* kst_hi = reinterpret_cast<float*>(smem + L.kst);
    float* kst_lo = kst_hi + C * W;
    float* kst_v = kst_lo + C * W;
    uint8_t* ktok = reinterpret_cast<uint8_t*>(smem + L.ktok);
    uint16_t* posmap = reinterpret_cast<uint16_t*>(smem + L.posmap);
    uint32_t* cand = reinterpret_cast<uint32_t*>(smem + L.cand);
    float2* refs = reinterpret_cast<float2*>(smem + L.refs);
    uint32_t* valid = reinterpret_cast<uint32_t*>(smem + L.valid);
    float* dtab = reinterpret_cast<float*>(smem + L.dtab);
    float* uni = reinterpret_cast<float*>(smem + L.uni);
    float* diag = reinterpret_cast<float*>(smem + L.diag);
    uint32_t* misc = reinterpret_cast<uint32_t*>(smem + L.misc);
    __shared__ uint32_t wtot[kHcWarps][3];
    __shared__ uint32_t wpre[kHcWarps][3];

    const int c = blockIdx.x;
    const int n = C * (W + R);
    const int NB = (C + kHcBlk - 1) / kHcBlk;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t lt_mask = (1u << lane) - 1u;
    const int64_t base = static_cast<int64_t>(c) * row;
    const uint32_t* pay_row = pay + base;
    const uint4* rec_row = rec + base;
    long long t_last = 0, t_acc[12] = {};
    auto tick = [&](int slot) {
        if (kHcProfiling && prof != nullptr && tid == 0) {
            const long long now = clock64();
            if (slot >= 0) t_acc[slot] += now - t_last;
            t_last = now;
        }
    };
    tick(-1);

    // ---- 1. classification. A warp takes 1024 consecutive positions as 32 rows of 32 lanes
    // (coalesced loads). Group counts come from ballots: per lane, the K entries and entries
    // since its group's head; carries run across rows, then warps (hc_combine), then tiles.
    constexpr int kWarpSpan = 1024;
    constexpr int kTileSpan = kHcWarps * kWarpSpan;
    uint32_t* ctmp = reinterpret_cast<uint32_t*>(uni);   // per-warp candidate staging
    constexpr int kCtmp = kWarpSpan / 8 + 2;             // per warp (G >= 8)
    uint32_t* hbs = ctmp + kHcWarps * kCtmp;              // [warp][row] head ballots of a tile
    const uint32_t le_mask = lt_mask | (1u << lane);
    uint32_t tcarry = 0u, tk = 0u, tcand = 0u;            // carries over tiles
    for (int t0 = 0; t0 < n; t0 += kTileSpan) {
        const int wbase = t0 + warp * kWarpSpan;
        // pass A: warp totals (ballots only); row head ballots kept for pass B; the records
        // pass B reads are prefetched into L2 meanwhile
        uint32_t ck = 0u, ce_ = 0u, wk = 0u;
        bool seen = false;
#pragma unroll 1
        for (int k0 = 0; k0 < 32; k0 += 16) {
            uint32_t pv[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int ru = wbase + 32 * (k0 + u) + lane;
                pv[u] = ru < n ? pay_row[ru] : (kHead | kInvalid);
                if (ru < n) asm volatile("prefetch.global.L2 [%0];" ::"l"(rec_row + ru));
            }
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                const uint32_t p = pv[u];
                const uint32_t hb = __ballot_sync(0xFFFFFFFFu, (p & kHead) != 0u);
                const uint32_t kbl = __ballot_sync(0xFFFFFFFFu, is_k(p));
                if (lane == 0) hbs[warp * 32 + k0 + u] = hb;
                if (hb) {
                    const int lh = 31 - __clz(hb);
                    ck = __popc(kbl >> lh);
                    ce_ = 32u - lh;
                    seen = true;
                } else {
                    ck += __popc(kbl);
                    ce_ += 32u;
                }
                wk += __popc(kbl);
            }
        }
        if (lane == 0) {
            wtot[warp][0] = ck | (ce_ << 16) | (seen ? kHead : 0u);
            wtot[warp][1] = wk;
        }
        __syncthreads();
        tick(8);
        if (tid == 0) {
            uint32_t cc = tcarry, ckk = tk;
            for (int w = 0; w < kHcWarps; ++w) {
                wpre[w][0] = cc; wpre[w][1] = ckk;
                cc = hc_combine(cc, wtot[w][0]);
                ckk += wtot[w][1];
            }
            misc[4] = cc; misc[5] = ckk;
        }
        __syncthreads();
        // pass B: per position (records: payload, hi, lo, v)
        ck = wpre[warp][0] & 0xFFFFu;
        ce_ = (wpre[warp][0] >> 16) & 0x7FFFu;
        uint32_t krun = wpre[warp][1];
        uint32_t ncand = 0u;
        // next position's head flag for lane 31 of the warp's last row: the next warp's first
        // row, or the next tile's first position (one global load per tile)
        uint32_t nxt_head = 1u;
        if (warp + 1 < kHcWarps) {
            nxt_head = hbs[(warp + 1) * 32] & 1u;
        } else if (wbase + kWarpSpan < n) {
            nxt_head = (pay_row[wbase + kWarpSpan] & kHead) ? 1u : 0u;
        }
#pragma unroll 1
        for (int k0 = 0; k0 < 32; k0 += 8) {
        uint4 ev[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int ru = wbase + 32 * (k0 + u) + lane;
            ev[u] = ru < n ? rec_row[ru] : make_uint4(kHead | kInvalid, 0u, 0u, 0u);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int k = k0 + u;
            const int r = wbase + 32 * k + lane;
            const uint4 er = ev[u];
            const uint32_t p = er.x;
            const bool isk = is_k(p);
            const uint32_t hb = __ballot_sync(0xFFFFFFFFu, (p & kHead) != 0u);
            const uint32_t kball = __ballot_sync(0xFFFFFFFFu, isk);
            const uint32_t hm = hb & le_mask;
            uint32_t kb, eb;
            if (hm) {
                const int h = 31 - __clz(hm);
                kb = __popc(kball & lt_mask & ~((1u << h) - 1u));
                eb = static_cast<uint32_t>(lane - h);
            } else {
                kb = ck + __popc(kball & lt_mask);
                eb = ce_ + static_cast<uint32_t>(lane);
            }
            if (hb) {
                const int lh = 31 - __clz(hb);
                ck = __popc(kball >> lh);
                ce_ = 32u - lh;
            } else {
                ck += __popc(kball);
                ce_ += 32u;
            }
            const int kr = static_cast<int>(krun) + __popc(kball & lt_mask);
            if (lane == 0 && wbase + 32 * k < n) {
                kmask[(wbase + 32 * k) >> 5] = kball;
                kbef[(wbase + 32 * k) >> 5] = krun;
            }
            krun += __popc(kball);
            uint32_t nx = __shfl_down_sync(0xFFFFFFFFu, p, 1);
            if (lane == 31) nx = ((k + 1 < 32 ? hbs[warp * 32 + k + 1] & 1u : nxt_head) != 0u) ? kHead : 0u;
            bool is_cand = false;
            if (r < n) {
                kbh[r] = static_cast<uint8_t>(min(kb, 63u));
                if (isk) {
                    kst_hi[kr] = __uint_as_float(er.y);
                    kst_lo[kr] = __uint_as_float(er.z);
                    kst_v[kr] = __uint_as_float(er.w);
                    ktok[kr] = static_cast<uint8_t>(token_of(p));
                }
                posmap[(p & kIsQ) ? C * W + token_of(p) * R + lane_of(p) : token_of(p) * W + lane_of(p)] =
                    static_cast<uint16_t>(r);
                const bool last = (nx & kHead) != 0u || r + 1 >= n;
                is_cand = last && kb + (isk ? 1u : 0u) >= static_cast<uint32_t>(G);
            }
            const uint32_t cb = __ballot_sync(0xFFFFFFFFu, is_cand);
            if (is_cand) {
                const uint32_t at = ncand + __popc(cb & lt_mask);
                if (at < static_cast<uint32_t>(kCtmp)) {
                    ctmp[warp * kCtmp + at] = static_cast<uint32_t>(r - static_cast<int>(eb)) |
                                              static_cast<uint32_t>(r + 1) << 16;
                }
            }
            ncand += __popc(cb);
        }
        }
        if (lane == 0) wtot[warp][2] = ncand;
        __syncthreads();
        tick(9);
        if (tid == 0) {
            uint32_t cc = tcand;
            for (int w = 0; w < kHcWarps; ++w) { wpre[w][2] = cc; cc += wtot[w][2]; }
            misc[6] = cc;
        }
        __syncthreads();
        {
            const uint32_t m = min(wtot[warp][2], static_cast<uint32_t>(kCtmp));
            for (uint32_t t = lane; t < m; t += 32) {
                const uint32_t at = wpre[warp][2] + t;
                if (at < static_cast<uint32_t>(kHcMaxCand)) cand[at] = ctmp[warp * kCtmp + t];
            }
        }
        tcarry = misc[4]; tk = misc[5]; tcand = misc[6];
        __syncthreads();
        tick(10);
    }
    if (tid == 0) {
        misc[0] = min(tcand, static_cast<uint32_t>(kHcMaxCand));
        misc[8] = 0u;   // cold-walk row counters (A, QK)
        misc[9] = 0u;
        misc[10] = 0u;  // cold pairs walked (diagnostics)
        misc[11] = 0u;  // cold rows with work
        misc[12] = 0u;  // cold batches (32 pairs)
        misc[13] = 0u;  // same-cell groups (> 1 lane)
        misc[14] = 0u;  // lanes in same-cell groups
    }
    for (int t = tid; t < C; t += kHcThreads) diag[t] = 0.0f;
    __syncthreads();
    const int nc = static_cast<int>(misc[0]);
    for (int g = warp; g < nc; g += kHcWarps) {
        const int s0 = cand[g] & 0xFFFFu, s1 = cand[g] >> 16;
        for (int r = s0 + lane; r < s1; r += 32) kbh[r] |= kHcCand;
    }
    tick(0);

    // ---- 2. hot slots, batch by batch (one warp builds one slot's columns; no block barrier
    // until the dense step). A slot whose log-cumsum range over the chunk is at most
    // kHcSingleRange takes one reference (its first entry): a = v 2^(L - ref) <= |v|,
    // b = v 2^(ref - L) <= |v| 2^range, term a*b. Other slots take 16-token block references
    // (validation walk first).
    // 4x4 output tiles: threads 0-447 take the 28 off-diagonal 16x16 block pairs (16 each);
    // 448-495 the 6 below-diagonal 4x4 tiles of each diagonal block; 496-511 two diagonal
    // 4x4 tiles each (10 live cells per tile).
    // Dual threads keep their two diagonal tiles' 20 live cells in acc[x][y] (D0: y <= x; D1: the 6
    // upper slots) and ex[4] (D1's last row), so every thread needs only 40 accumulators.
    float acc_a[4][4], acc_q[4][4], ex_a[4], ex_q[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        ex_a[x] = ex_q[x] = 0.0f;
#pragma unroll
        for (int y = 0; y < 4; ++y) acc_a[x][y] = acc_q[x][y] = 0.0f;
    }
    int TI = 0, TJ = 0, row0 = 0, col0 = 0, row0b = 0;
    bool dual = false;
    if (tid < 448) {
        const int ob = tid >> 4, st = tid & 15;
        TI = 1;
        while (TI * (TI + 1) / 2 <= ob) ++TI;
        TJ = ob - TI * (TI - 1) / 2;
        row0 = TI * kHcBlk + (st >> 2) * 4;
        col0 = TJ * kHcBlk + (st & 3) * 4;
    } else if (tid < 496) {
        const int d = (tid - 448) / 6, u = (tid - 448) % 6;   // (1,0) (2,0) (2,1) (3,0) (3,1) (3,2)
        const int sr = u < 1 ? 1 : (u < 3 ? 2 : 3);
        const int sc = u < 1 ? 0 : (u < 3 ? u - 1 : u - 3);
        TI = TJ = d;
        row0 = d * kHcBlk + sr * 4;
        col0 = d * kHcBlk + sc * 4;
    } else {
        const int d = (tid - 496) >> 1, p = (tid - 496) & 1;
        TI = TJ = d;
        dual = true;
        row0 = col0 = d * kHcBlk + 8 * p;   // diagonal tiles (2p, 2p) and (2p+1, 2p+1)
        row0b = row0 + 4;
    }
    const bool tile_live = TI < NB;
    const bool tdiag = TI == TJ;
    float* pa = uni;
    float* paq = pa + kHcKB * kHcMaxC;
    float* pbd = paq + kHcKB * kHcMaxC;
    float* pbo = pbd + kHcKB * kHcMaxC;
    float* pdg = pbo + kHcKB * kHcMaxC;
    constexpr int kU = 4;   // entries per lane in flight in the group walks (128 per round; = the cache)
    __syncthreads();
    for (int b0 = 0; b0 < nc; b0 += kHcKB) {
        const int nb = min(kHcKB, nc - b0);
        if (warp < kHcKB) {
            const int s = warp;
            float* ca = pa + s * kHcMaxC;
            float* caq = paq + s * kHcMaxC;
            float* cbd = pbd + s * kHcMaxC;
            float* cbo = pbo + s * kHcMaxC;
            float* cdg = pdg + s * kHcMaxC;
            auto zero_cols = [&]() {
                for (int t = lane; t < kHcMaxC; t += 32) {
                    ca[t] = 0.0f; caq[t] = 0.0f; cbd[t] = 0.0f; cbo[t] = 0.0f; cdg[t] = 0.0f;
                }
            };
            zero_cols();
            if (s < nb) {
                const int g = b0 + s;
                const int s0 = cand[g] & 0xFFFFu, s1 = cand[g] >> 16;
                float2* rf = refs + s * (kHcMaxNB + 1);
                // Groups of at most 32 * kU entries are loaded once and kept in registers for
                // both walks (one global round trip per slot).
                constexpr int kUc = 4;   // cached entries per lane (128 per group)
                const bool cached = s1 - s0 <= 32 * kUc;
                uint4 ec[kUc];
#pragma unroll
                for (int k = 0; k < kUc; ++k) {
                    const int r = s0 + 32 * k + lane;
                    ec[k] = (cached && r < s1) ? rec_row[r] : make_uint4(kInvalid, 0u, 0u, 0u);
                }
                float fhi, flo, lhi = 0.0f, llo = 0.0f;
                if (cached) {
                    fhi = __shfl_sync(0xFFFFFFFFu, __uint_as_float(ec[0].y), 0);
                    flo = __shfl_sync(0xFFFFFFFFu, __uint_as_float(ec[0].z), 0);
                    const int kl = (s1 - 1 - s0) >> 5, ll = (s1 - 1 - s0) & 31;
#pragma unroll
                    for (int k = 0; k < kUc; ++k) {
                        const float hy = __shfl_sync(0xFFFFFFFFu, __uint_as_float(ec[k].y), ll);
                        const float hz = __shfl_sync(0xFFFFFFFFu, __uint_as_float(ec[k].z), ll);
                        if (k == kl) { lhi = hy; llo = hz; }
                    }
                } else {
                    const uint4 ef = rec_row[s0], el = rec_row[s1 - 1];
                    fhi = __uint_as_float(ef.y); flo = __uint_as_float(ef.z);
                    lhi = __uint_as_float(el.y); llo = __uint_as_float(el.z);
                }
                const float span = log2_decay(fhi, flo, lhi, llo);
                const bool single = span <= kHcSingleRange;   // false for NaN
                bool bad = false;
                uint32_t wmask_l = 0u;   // wide blocks of this slot
                uint32_t qm_l = 0u;      // row blocks holding this slot's Q members
                if (!single) {
                    // block references + validation
                    for (int I = lane; I <= NB; I += 32) rf[I] = make_float2(0.0f, 0.0f);
                    __syncwarp();
                    uint32_t cprev = kHead | kInvalid;
                    float chi = 0.0f, clo = 0.0f;
                    for (int r1 = s0; r1 < s1; r1 += 32 * kU) {
                        uint4 e[kU];
#pragma unroll
                        for (int k = 0; k < kU; ++k) {
                            const int r = r1 + 32 * k + lane;
                            e[k] = (cached && k < kUc) ? ec[k < kUc ? k : 0]
                                                       : (r < s1 ? rec_row[r] : make_uint4(kInvalid, 0u, 0u, 0u));
                        }
#pragma unroll
                        for (int k = 0; k < kU; ++k) {
                            if (r1 + 32 * k >= s1) break;
                            const int r = r1 + 32 * k + lane;
                            const float hi = __uint_as_float(e[k].y), lo = __uint_as_float(e[k].z);
                            uint32_t pp = __shfl_up_sync(0xFFFFFFFFu, e[k].x, 1);
                            float phi = __shfl_up_sync(0xFFFFFFFFu, hi, 1);
                            float plo = __shfl_up_sync(0xFFFFFFFFu, lo, 1);
                            if (lane == 0) { pp = cprev; phi = chi; plo = clo; }
                            uint32_t nxp = __shfl_down_sync(0xFFFFFFFFu, e[k].x, 1);
                            const uint32_t nx0 = __shfl_sync(0xFFFFFFFFu, e[k + 1 < kU ? k + 1 : k].x, 0);
                            const bool has_next = r + 1 < s1;
                            if (lane == 31 && has_next) nxp = k + 1 < kU ? nx0 : pay_row[r + 1];
                            if (r < s1) {
                                const int tok = token_of(e[k].x);
                                const bool first = r == s0;
                                const float dd = first ? hi + lo : log2_decay(hi, lo, phi, plo);
                                bad |= !(dd <= 0.0f) || !isfinite(hi) || !isfinite(lo);
                                (void)pp;
                                const int blk = tok / kHcBlk;
                                const int nblk = has_next ? token_of(nxp) / kHcBlk : NB;
                                for (int I = blk + 1; I <= nblk; ++I) rf[I] = make_float2(hi, lo);
                            }
                            cprev = __shfl_sync(0xFFFFFFFFu, e[k].x, 31);
                            chi = __shfl_sync(0xFFFFFFFFu, hi, 31);
                            clo = __shfl_sync(0xFFFFFFFFu, lo, 31);
                        }
                    }
                    __syncwarp();
                    // Blocks whose range exceeds kHcMaxRange are WIDE: their own pairs go to the
                    // exact walk (below), every cross-block pair stays dense.
                    for (int I = lane; I < NB; I += 32) {
                        const float rg = log2_decay(rf[I].x, rf[I].y, rf[I + 1].x, rf[I + 1].y);
                        bad |= !(rg == rg);   // NaN
                        if (!(rg <= kHcMaxRange)) wmask_l |= 1u << I;
                    }
                    bad = __any_sync(0xFFFFFFFFu, bad);
                    wmask_l = __reduce_or_sync(0xFFFFFFFFu, wmask_l);
                    // D table of this slot
                    float* dt = dtab + s * kHcMaxNB * kHcMaxNB;
                    for (int IJ = lane; IJ < kHcMaxNB * kHcMaxNB; IJ += 32) {
                        const int I = IJ / kHcMaxNB, J = IJ - I * kHcMaxNB;
                        float dv = 0.0f;
                        if (!bad && J < I && I < NB) dv = exp2f(log2_decay(rf[I].x, rf[I].y, rf[J + 1].x, rf[J + 1].y));
                        dt[IJ] = dv;
                    }
                } else if (lane == 0) {
                    rf[0] = make_float2(fhi, flo);
                }
                __syncwarp();
                if (!bad) {
                    // panel walk with the per-entry checks; entries of WIDE blocks get their
                    // within-block earlier-K count for the exact walk instead of diagonal factors
                    uint32_t cprev = kHead | kInvalid;
                    float cv = 0.0f, chi = 0.0f, clo = 0.0f;
                    uint32_t ccnt = 0u;
                    int cblk = -1;
                    for (int r1 = s0; r1 < s1; r1 += 32 * kU) {
                        uint4 e[kU];
#pragma unroll
                        for (int k = 0; k < kU; ++k) {
                            const int r = r1 + 32 * k + lane;
                            e[k] = (cached && k < kUc) ? ec[k < kUc ? k : 0]
                                                       : (r < s1 ? rec_row[r] : make_uint4(kInvalid, 0u, 0u, 0u));
                        }
#pragma unroll
                        for (int k = 0; k < kU; ++k) {
                            if (r1 + 32 * k >= s1) break;
                            const int r = r1 + 32 * k + lane;
                            const float hi = __uint_as_float(e[k].y), lo = __uint_as_float(e[k].z),
                                        v = __uint_as_float(e[k].w);
                            uint32_t pp = __shfl_up_sync(0xFFFFFFFFu, e[k].x, 1);
                            float pv = __shfl_up_sync(0xFFFFFFFFu, v, 1);
                            float phi = __shfl_up_sync(0xFFFFFFFFu, hi, 1);
                            float plo = __shfl_up_sync(0xFFFFFFFFu, lo, 1);
                            if (lane == 0) { pp = cprev; pv = cv; phi = chi; plo = clo; }
                            const bool inr = r < s1;
                            const int blkk = inr ? (single ? 0 : token_of(e[k].x) / kHcBlk) : 0x7FFF;
                            int pblk = __shfl_up_sync(0xFFFFFFFFu, blkk, 1);
                            if (lane == 0) pblk = cblk;
                            const uint32_t kbl = __ballot_sync(0xFFFFFFFFu, inr && is_k(e[k].x));
                            const uint32_t hbl = __ballot_sync(0xFFFFFFFFu, inr && blkk != pblk);
                            const uint32_t hmm = hbl & le_mask;
                            uint32_t cbk;
                            if (hmm) {
                                const int h = 31 - __clz(hmm);
                                cbk = __popc(kbl & lt_mask & ~((1u << h) - 1u));
                            } else {
                                cbk = ccnt + __popc(kbl & lt_mask);
                            }
                            if (hbl) {
                                ccnt = __popc(kbl >> (31 - __clz(hbl)));
                            } else {
                                ccnt += __popc(kbl);
                            }
                            cblk = __shfl_sync(0xFFFFFFFFu, blkk, 31);
                            if (inr) {
                                const uint32_t px = e[k].x;
                                const bool kk = is_k(px), qq = is_q(px);
                                const int tok = token_of(px);
                                const bool first = r == s0;
                                bad |= !(isfinite(hi) && isfinite(lo) && isfinite(v));
                                bad |= !(fabsf(v) <= kHcMaxAbsV) || (v != 0.0f && !(fabsf(v) >= kHcMinAbsV));
                                const float dd = first ? (single ? 0.0f : hi + lo) : log2_decay(hi, lo, phi, plo);
                                bad |= !(dd <= 0.0f);
                                // A token's repeated K (or Q) entry of the slot demotes it; it writes nothing.
                                const bool dup = !first && token_of(pp) == tok &&
                                                 ((kk && is_k(pp)) || (qq && is_q(pp)));
                                bad |= dup;
                                const int I = single ? 0 : tok / kHcBlk;
                                const float2 r0f = rf[I];
                                const bool wide = !single && ((wmask_l >> I) & 1u);
                                if (dup) {
                                } else if (kk) {
                                    // A nonzero value whose factor falls below FLT_MIN (subnormal or
                                    // zero: a wide block's row factor can) demotes the slot: the product
                                    // would lose a term the exact walk keeps.
                                    const float fa = __fmul_rn(v, exp2f(log2_decay(hi, lo, r0f.x, r0f.y)));
                                    ca[tok] = fa;
                                    float fmin = fabsf(fa);
                                    if (!wide) {
                                        const float fbd = __fmul_rn(v, exp2f(log2_decay(r0f.x, r0f.y, hi, lo)));
                                        cbd[tok] = fbd;
                                        fmin = fminf(fmin, fabsf(fbd));
                                    }
                                    if (!single) {
                                        const float2 r1f = rf[I + 1];
                                        const float fbo = __fmul_rn(v, exp2f(log2_decay(r1f.x, r1f.y, hi, lo)));
                                        cbo[tok] = fbo;
                                        fmin = fminf(fmin, fabsf(fbo));
                                    }
                                    bad |= v != 0.0f && !(fmin >= FLT_MIN);
                                } else if (qq) {
                                    qm_l |= 1u << (tok / kHcBlk);
                                    const float faq = __fmul_rn(v, exp2f(log2_decay(hi, lo, r0f.x, r0f.y)));
                                    caq[tok] = faq;
                                    bad |= v != 0.0f && !(fabsf(faq) >= FLT_MIN);
                                    if (!wide && !first && is_k(pp) && token_of(pp) == tok) cdg[tok] = __fmul_rn(v, pv);
                                }
                                if (kk || qq) kbh[r] = static_cast<uint8_t>(kHcHot | (wide ? min(cbk, 63u) : 0u));
                            }
                            cprev = __shfl_sync(0xFFFFFFFFu, e[k].x, 31);
                            cv = __shfl_sync(0xFFFFFFFFu, v, 31);
                            chi = __shfl_sync(0xFFFFFFFFu, hi, 31);
                            clo = __shfl_sync(0xFFFFFFFFu, lo, 31);
                        }
                    }
                    bad = __any_sync(0xFFFFFFFFu, bad);
                    qm_l = __reduce_or_sync(0xFFFFFFFFu, qm_l);
                }
                if (bad) {
                    __syncwarp();
                    zero_cols();
                    for (int r = s0 + lane; r < s1; r += 32) kbh[r] = kHcCand;   // demoted: full walk
                }
                if (lane == 0) valid[s] = bad ? 0u : ((single ? 3u : 1u) | (wmask_l << 8) | (qm_l << 16));
            }
        }
        __syncthreads();
        tick(2);
        // dense accumulation, slot by slot (single-reference slots: b = bd for every tile)
        // Branch-free: demoted slots and the diagonal columns of wide blocks are zero in the
        // panels, so they add exact zeros. Two slots per iteration, added in slot order.
        if (tile_live) {
            auto slot_fma = [&](float (&aa)[4][4], float (&aq)[4][4], const float4 bvr, const float dsc,
                                const float4 av, const float* aqp, const bool qon) {
                const float bc[4] = {__fmul_rn(bvr.x, dsc), __fmul_rn(bvr.y, dsc),
                                     __fmul_rn(bvr.z, dsc), __fmul_rn(bvr.w, dsc)};
                const float ar[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y) {
                        if constexpr (kHcRoundedProducts) {
                            aa[x][y] = __fadd_rn(aa[x][y], __fmul_rn(ar[x], bc[y]));
                        } else {
                            aa[x][y] = fmaf(ar[x], bc[y], aa[x][y]);
                        }
                    }
                if (qon) {   // the slot has Q members in this row block (else aq is zero here)
                    const float4 aqv = *reinterpret_cast<const float4*>(aqp);
                    const float qr[4] = {aqv.x, aqv.y, aqv.z, aqv.w};
#pragma unroll
                    for (int x = 0; x < 4; ++x)
#pragma unroll
                        for (int y = 0; y < 4; ++y) {
                            if constexpr (kHcRoundedProducts) {
                                aq[x][y] = __fadd_rn(aq[x][y], __fmul_rn(qr[x], bc[y]));
                            } else {
                                aq[x][y] = fmaf(qr[x], bc[y], aq[x][y]);
                            }
                        }
                }
            };
            // This thread's per-slot choice (bit s: column factor bo with the D scale), read once.
            uint32_t dmask = 0u, qmask = 0u;
            for (int s2 = 0; s2 < nb; ++s2) {
                const uint32_t f = valid[s2];
                if (!tdiag) dmask |= ((f & 0xFFu) == 1u ? 1u : 0u) << s2;
                qmask |= ((f >> (16 + TI)) & 1u) << s2;
            }
            // dual threads: diagonal tiles D0 (rows/cols row0..+3) and D1 (row0b..+3), bd columns
            auto dual_slot = [&](const int s2, const float4 b0v, const float4 a0v) {
                auto acc1 = [&](float& c, const float x, const float y) {
                    if constexpr (kHcRoundedProducts) {
                        c = __fadd_rn(c, __fmul_rn(x, y));
                    } else {
                        c = fmaf(x, y, c);
                    }
                };
                const float4 b1v = *reinterpret_cast<const float4*>(pbd + s2 * kHcMaxC + row0b);
                const float4 a1v = *reinterpret_cast<const float4*>(pa + s2 * kHcMaxC + row0b);
                const float B0[4] = {b0v.x, b0v.y, b0v.z, b0v.w}, B1[4] = {b1v.x, b1v.y, b1v.z, b1v.w};
                auto cells = [&](float (&c)[4][4], float (&e)[4], const float (&A0)[4], const float (&A1)[4]) {
#pragma unroll
                    for (int x = 0; x < 4; ++x)
#pragma unroll
                        for (int y = 0; y <= x; ++y) acc1(c[x][y], A0[x], B0[y]);
                    acc1(c[0][1], A1[0], B1[0]);
                    acc1(c[0][2], A1[1], B1[0]);
                    acc1(c[0][3], A1[1], B1[1]);
                    acc1(c[1][2], A1[2], B1[0]);
                    acc1(c[1][3], A1[2], B1[1]);
                    acc1(c[2][3], A1[2], B1[2]);
                    acc1(e[0], A1[3], B1[0]);
                    acc1(e[1], A1[3], B1[1]);
                    acc1(e[2], A1[3], B1[2]);
                    acc1(e[3], A1[3], B1[3]);
                };
                {
                    const float A0[4] = {a0v.x, a0v.y, a0v.z, a0v.w}, A1[4] = {a1v.x, a1v.y, a1v.z, a1v.w};
                    cells(acc_a, ex_a, A0, A1);
                }
                if ((qmask >> s2) & 1u) {
                    const float4 q0v = *reinterpret_cast<const float4*>(paq + s2 * kHcMaxC + row0);
                    const float4 q1v = *reinterpret_cast<const float4*>(paq + s2 * kHcMaxC + row0b);
                    const float Q0[4] = {q0v.x, q0v.y, q0v.z, q0v.w}, Q1[4] = {q1v.x, q1v.y, q1v.z, q1v.w};
                    cells(acc_q, ex_q, Q0, Q1);
                }
            };
            int s = 0;
            for (; s + 1 < nb; s += 2) {
                const bool d0 = (dmask >> s) & 1u, d1 = (dmask >> (s + 1)) & 1u;
                const float4 b0 = *reinterpret_cast<const float4*>((d0 ? pbo : pbd) + s * kHcMaxC + col0);
                const float4 b1 = *reinterpret_cast<const float4*>((d1 ? pbo : pbd) + (s + 1) * kHcMaxC + col0);
                const float x0 = d0 ? dtab[(s * kHcMaxNB + TI) * kHcMaxNB + TJ] : 1.0f;
                const float x1 = d1 ? dtab[((s + 1) * kHcMaxNB + TI) * kHcMaxNB + TJ] : 1.0f;
                const float4 a0 = *reinterpret_cast<const float4*>(pa + s * kHcMaxC + row0);
                const float4 a1 = *reinterpret_cast<const float4*>(pa + (s + 1) * kHcMaxC + row0);
                if (!dual) {
                    slot_fma(acc_a, acc_q, b0, x0, a0, paq + s * kHcMaxC + row0, (qmask >> s) & 1u);
                    slot_fma(acc_a, acc_q, b1, x1, a1, paq + (s + 1) * kHcMaxC + row0, (qmask >> (s + 1)) & 1u);
                } else {
                    dual_slot(s, b0, a0);
                    dual_slot(s + 1, b1, a1);
                }
            }
            if (s < nb) {
                const bool d0 = (dmask >> s) & 1u;
                const float4 b0 = *reinterpret_cast<const float4*>((d0 ? pbo : pbd) + s * kHcMaxC + col0);
                const float x0 = d0 ? dtab[(s * kHcMaxNB + TI) * kHcMaxNB + TJ] : 1.0f;
                const float4 a0 = *reinterpret_cast<const float4*>(pa + s * kHcMaxC + row0);
                if (!dual) {
                    slot_fma(acc_a, acc_q, b0, x0, a0, paq + s * kHcMaxC + row0, (qmask >> s) & 1u);
                } else {
                    dual_slot(s, b0, a0);
                }
            }
        }
        for (int t = tid; t < C; t += kHcThreads) {
            float dsum = diag[t];
            for (int s = 0; s < nb; ++s) dsum = __fadd_rn(dsum, pdg[s * kHcMaxC + t]);
            diag[t] = dsum;
        }
        __syncthreads();
        tick(3);
    }

    // ---- 3./4./5. per triangle: dense sums -> shared triangle, cold walk, out
    float* tri = uni;
    const int64_t out = static_cast<int64_t>(c) * C * C;
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        const bool q = pass == 1;
        if (tile_live) {
            auto put = [&](const int i, const int j, const float va, const float vq) {
                if (i < C && j <= i) tri[i * (i + 1) / 2 + j] = q ? (i == j ? diag[i] : vq) : va;
            };
            if (!dual) {
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y < 4; ++y) put(row0 + x, col0 + y, acc_a[x][y], acc_q[x][y]);
            } else {
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int y = 0; y <= x; ++y) put(row0 + x, row0 + y, acc_a[x][y], acc_q[x][y]);
                put(row0b + 0, row0b + 0, acc_a[0][1], acc_q[0][1]);
                put(row0b + 1, row0b + 0, acc_a[0][2], acc_q[0][2]);
                put(row0b + 1, row0b + 1, acc_a[0][3], acc_q[0][3]);
                put(row0b + 2, row0b + 0, acc_a[1][2], acc_q[1][2]);
                put(row0b + 2, row0b + 1, acc_a[1][3], acc_q[1][3]);
                put(row0b + 2, row0b + 2, acc_a[2][3], acc_q[2][3]);
#pragma unroll
                for (int y = 0; y < 4; ++y) put(row0b + 3, row0b + y, ex_a[y], ex_q[y]);
            }
        }
        __syncthreads();
        tick(4);
        const int width = q ? R : W;
        // A row's entries in lane order: two consecutive entries per lane per 64-entry chunk.
        // A warp walks kHcColdRows rows in LOCKSTEP (every step is issued for all its rows
        // back to back, so their latency chains overlap); rows go to warps dynamically and a
        // row is only ever touched by its warp, in its own fixed order (pairs in index order,
        // same-cell lanes in lane order): the result does not depend on the grouping.
        constexpr int NR = kHcColdRows;
        for (;;) {
            int ib = 0;
            if (lane == 0) ib = static_cast<int>(atomicAdd(misc + 8 + pass, static_cast<unsigned>(NR)));
            ib = __shfl_sync(0xFFFFFFFFu, ib, 0);
            if (ib >= C) break;
            for (int e0 = 0; e0 < width; e0 += 64) {
                const int ea = e0 + 2 * lane, eb = ea + 1;
                int ra[NR], rb[NR], cea[NR], ceb[NR], kra[NR], krb[NR], incl[NR], exa[NR], tot[NR];
                float ha[NR], la[NR], va[NR], hb[NR], lb[NR], vb[NR];
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    ra[r] = 0;
                    rb[r] = 0;
                    if (ib + r < C) {
                        const uint16_t* prow = posmap + (q ? C * W : 0) + (ib + r) * width;
                        if (ea < width) ra[r] = prow[ea];
                        if (eb < width) rb[r] = prow[eb];
                    }
                }
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    const bool live = ib + r < C;
                    const uint8_t ba = (live && ea < width) ? kbh[ra[r]] : kHcHot;
                    const uint8_t bbv = (live && eb < width) ? kbh[rb[r]] : kHcHot;
                    int ca = 0, cb = 0, ka = 0, kb = 0;
                    if (ba & kHcHot) {
                        ca = ba & kHcCount;   // wide block of a hot slot: its own earlier K members
                        if (ca) ka = hc_krank(kmask, kbef, ra[r]);
                    } else if (ba & kHcCand) {
                        ka = hc_krank(kmask, kbef, ra[r]);
                        ca = ka - hc_krank(kmask, kbef, static_cast<int>(cand[hc_cand_of(cand, nc, ra[r])] & 0xFFFFu));
                    } else {
                        ca = ba & kHcCount;
                        if (ca) ka = hc_krank(kmask, kbef, ra[r]);
                    }
                    if (bbv & kHcHot) {
                        cb = bbv & kHcCount;
                        if (cb) kb = hc_krank(kmask, kbef, rb[r]);
                    } else if (bbv & kHcCand) {
                        kb = hc_krank(kmask, kbef, rb[r]);
                        cb = kb - hc_krank(kmask, kbef, static_cast<int>(cand[hc_cand_of(cand, nc, rb[r])] & 0xFFFFu));
                    } else {
                        cb = bbv & kHcCount;
                        if (cb) kb = hc_krank(kmask, kbef, rb[r]);
                    }
                    cea[r] = ca; ceb[r] = cb; kra[r] = ka; krb[r] = kb;
                    incl[r] = ca + cb;
                }
                // The entries' records are fetched before the prefix sum (QK rows read them from global
                // memory), so their latency overlaps it; no record when no pairs.
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    ha[r] = la[r] = va[r] = hb[r] = lb[r] = vb[r] = 0.0f;
                    if (q) {
                        if (cea[r]) {
                            const uint4 o = rec_row[ra[r]];
                            ha[r] = __uint_as_float(o.y); la[r] = __uint_as_float(o.z); va[r] = __uint_as_float(o.w);
                        }
                        if (ceb[r]) {
                            const uint4 o = rec_row[rb[r]];
                            hb[r] = __uint_as_float(o.y); lb[r] = __uint_as_float(o.z); vb[r] = __uint_as_float(o.w);
                        }
                    } else {
                        if (cea[r]) { ha[r] = kst_hi[kra[r]]; la[r] = kst_lo[kra[r]]; va[r] = kst_v[kra[r]]; }
                        if (ceb[r]) { hb[r] = kst_hi[krb[r]]; lb[r] = kst_lo[krb[r]]; vb[r] = kst_v[krb[r]]; }
                    }
                }
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
#pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        const int o = __shfl_up_sync(0xFFFFFFFFu, incl[r], d);
                        if (lane >= d) incl[r] += o;
                    }
                }
                int nmax = 0;
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    tot[r] = __shfl_sync(0xFFFFFFFFu, incl[r], 31);
                    exa[r] = incl[r] - (cea[r] + ceb[r]);
                    nmax = max(nmax, tot[r]);
                    if (kHcProfiling && prof != nullptr && lane == 0 && tot[r] > 0) {
                        atomicAdd(misc + 10, static_cast<uint32_t>(tot[r]));
                        atomicAdd(misc + 11, 1u);
                        atomicAdd(misc + 12, static_cast<uint32_t>((tot[r] + 31) / 32));
                    }
                }
                if (nmax == 0) continue;
                // 32 pairs per row per step: pair pb0 + lane of every row
                for (int pb0 = 0; pb0 < nmax; pb0 += 32) {
                    const int pidx = pb0 + lane;
                    int Ln[NR];
#pragma unroll
                    for (int r = 0; r < NR; ++r) Ln[r] = 0;
#pragma unroll
                    for (int step = 16; step >= 1; step >>= 1) {
#pragma unroll
                        for (int r = 0; r < NR; ++r) {
                            const int v = __shfl_sync(0xFFFFFFFFu, incl[r], Ln[r] + step - 1);
                            if (v <= pidx) Ln[r] += step;
                        }
                    }
                    float term[NR];
                    int jj[NR];
                    bool act[NR];
#pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        const int L = min(Ln[r], 31);
                        const int exL = __shfl_sync(0xFFFFFFFFu, exa[r], L);
                        const int ceaL = __shfl_sync(0xFFFFFFFFu, cea[r], L);
                        const int kraL = __shfl_sync(0xFFFFFFFFu, kra[r], L);
                        const int krbL = __shfl_sync(0xFFFFFFFFu, krb[r], L);
                        const float haL = __shfl_sync(0xFFFFFFFFu, ha[r], L);
                        const float laL = __shfl_sync(0xFFFFFFFFu, la[r], L);
                        const float vaL = __shfl_sync(0xFFFFFFFFu, va[r], L);
                        const float hbL = __shfl_sync(0xFFFFFFFFu, hb[r], L);
                        const float lbL = __shfl_sync(0xFFFFFFFFu, lb[r], L);
                        const float vbL = __shfl_sync(0xFFFFFFFFu, vb[r], L);
                        const bool inA = pidx < exL + ceaL;
                        const int m = inA ? pidx - exL : pidx - exL - ceaL;
                        const int kre = inA ? kraL : krbL;
                        const float hel = inA ? haL : hbL, lel = inA ? laL : lbL, vel = inA ? vaL : vbL;
                        act[r] = false;
                        jj[r] = 0;
                        term[r] = 0.0f;
                        if (pidx < tot[r]) {
                            const int mk = kre - 1 - m;
                            const int j = ktok[mk];
                            jj[r] = j;
                            act[r] = q || j != ib + r;
                            if (act[r]) {
                                term[r] = __fmul_rn(__fmul_rn(vel, kst_v[mk]),
                                                    exp2f(log2_decay(hel, lel, kst_hi[mk], kst_lo[mk])));
                            }
                        }
                    }
                    // commit row by row (rows are distinct); lanes of one cell: the lowest lane
                    // adds the cell's terms in lane order, fetched by full-warp shuffles
#pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        if (pb0 >= tot[r]) continue;   // warp-uniform
                        float* trow = tri + (ib + r) * (ib + r + 1) / 2;
                        const uint32_t am = __ballot_sync(0xFFFFFFFFu, act[r]);
                        uint32_t peers = 0u;
                        if (act[r]) peers = __match_any_sync(am, jj[r]);
                        const bool leader = act[r] && (peers & lt_mask) == 0u;
                        const int gsz = leader ? __popc(peers) : 0;
                        const int gmax = __reduce_max_sync(0xFFFFFFFFu, static_cast<unsigned>(gsz));
                        if (kHcProfiling && prof != nullptr && gsz > 1) {
                            atomicAdd(misc + 13, 1u);
                            atomicAdd(misc + 14, static_cast<uint32_t>(gsz));
                        }
                        // The leader is the lowest lane of `peers`; it walks the other lanes of its cell in
                        // increasing lane order by clearing the lowest set bit.
                        float vv = 0.0f;
                        uint32_t rest = 0u;
                        if (leader) {
                            vv = __fadd_rn(trow[jj[r]], term[r]);
                            rest = peers & (peers - 1u);
                        }
                        for (int k = 1; k < gmax; ++k) {
                            const int src = rest ? __ffs(rest) - 1 : lane;
                            const float t = __shfl_sync(0xFFFFFFFFu, term[r], src);
                            if (rest) {
                                vv = __fadd_rn(vv, t);
                                rest &= rest - 1u;
                            }
                        }
                        if (leader) trow[jj[r]] = vv;
                        __syncwarp();
                    }
                }
            }
            // A row is final once its own warp has walked it (the dense sums were put before the walk and
            // no other warp touches it): write it out now, overlapped with the other warps' walks.
            __syncwarp();
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                const int i = ib + r;
                if (i >= C) break;
                const float* trw = tri + i * (i + 1) / 2;
                float* drow = (q ? qk : a) + out + static_cast<int64_t>(i) * C;
                const int lim = q ? i + 1 : i;   // live cells j < lim (zeros above the diagonal)
                // Consecutive lanes read consecutive words (no bank conflicts) and store 128 B per warp.
                for (int j = lane; j < C; j += 32) drow[j] = j < lim ? trw[j] : 0.0f;
            }
        }
        __syncthreads();   // the next pass rewrites tri
        tick(5);
        tick(6);
    }
    if (kHcProfiling && prof != nullptr && tid == 0) {
        for (int k = 0; k < 7; ++k) prof[static_cast<int64_t>(c) * 16 + k] = t_acc[k];
        prof[static_cast<int64_t>(c) * 16 + 7] = misc[10];
        for (int k = 8; k < 12; ++k) prof[static_cast<int64_t>(c) * 16 + k] = t_acc[k];
        for (int k = 11; k < 15; ++k) prof[static_cast<int64_t>(c) * 16 + k + 1] = misc[k];
    }
}

// Shared-memory layout of the hot/cold kernel for a chunk shape, or total = 0 when it does not apply.
HcLayout hc_layout(int C, int W, int R, int row, int G) {
    HcLayout L{};
    if (C > kHcMaxC || row > kHcMaxRow || G < 8 || G > 64 ||
        static_cast<int64_t>(C) * W / G > kHcMaxCand) return L;
    auto take = [&](int bytes) {
        const int at = (L.total + 15) / 16 * 16;
        L.total = at + bytes;
        return at;
    };
    L.kbh = take(row);
    L.kmask = take(row / 32 * 4 + 4);
    L.kbef = take(row / 32 * 4 + 4);
    L.kst = take(3 * C * W * 4);
    L.ktok = take(C * W);
    L.posmap = take((C * W + C * R) * 2);
    L.cand = take(kHcMaxCand * 4);
    L.refs = take(kHcKB * (kHcMaxNB + 1) * 8);
    L.valid = take(kHcKB * 4);
    L.dtab = take(kHcKB * kHcMaxNB * kHcMaxNB * 4);
    L.uni = take(std::max(5 * kHcKB * kHcMaxC * 4, C * (C + 1) / 2 * 4));
    L.diag = take(kHcMaxC * 4);
    L.misc = take(64 * 4);
    return L;
}

struct Args {
    at::Tensor k_idx, q_idx, k_val, q_val, g, repeats;
    int P;
    int64_t stream_offset;
    int key_bits;
    int valid_last;
    bool read_groups;
    int64_t smem_limit;   // shared memory per block to assume (0: the device's opt-in maximum)
    at::Tensor beta;      // [N, C] f32 or empty: needed by kspa
    int rec_L;            // tile length of tiles_k / tiles_q / segb_k
    int64_t det_fields;   // optional outputs to compute (kDet* bits)
    bool modern;          // the routing predicate's modern backend (_device.backend()): hot/cold and cluster scans
    int sms;              // the device's SM count (_device.num_sms()): the cluster scan's chunk cutoff
};

// det_fields bits (mirrored as DET_* in the Python module).
constexpr int64_t kDetPos = 1;        // pos_k, pos_q
constexpr int64_t kDetSpayRank = 2;   // spay_rank
constexpr int64_t kDetFlags = 4;      // is_last, is_only, q_last, q_only
constexpr int64_t kDetSpayK = 8;      // spay_k
constexpr int64_t kDetSpayQ = 16;     // spay_q
constexpr int64_t kDetKrec = 32;      // krec
constexpr int64_t kDetKspa = 64;      // kspa (needs beta)
constexpr int64_t kDetQrec = 128;     // qrec
constexpr int64_t kDetTiles = 256;    // tiles_k, tiles_q, segb_k (tile length rec_L)
constexpr int64_t kDetDcGidx = 512;   // dc_kpos, dc_bpre
constexpr int64_t kDetDcRec = 1024;   // dc_rec (refused: no caller requests it)
constexpr int64_t kDetQrec8 = 2048;   // qrec8 (compact qrec for the read-side backward mover)
constexpr int64_t kDetAll = 4095;
static_assert(kFsSpayK == kDetSpayK && kFsSpayQ == kDetSpayQ && kFsKrec == kDetKrec && kFsKspa == kDetKspa &&
              kFsQrec == kDetQrec && kFsTiles == kDetTiles && kFsDcGidx == kDetDcGidx && kFsDcRec == kDetDcRec &&
              kFsQrec8 == kDetQrec8, "the scan flag bits mirror the det_fields bits");

// Shared memory per block that Phase 1 may use: the device's opt-in maximum, or less when the
// caller says so (tests simulate GPUs with less shared memory this way).
size_t smem_limit_of(int device, int64_t override_bytes) {
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    size_t limit = static_cast<size_t>(optin);
    if (override_bytes > 0) limit = std::min(limit, static_cast<size_t>(override_bytes));
    return limit;
}

// The least shared memory per block Phase 1 runs with: the 48 KiB every CUDA GPU grants
// without opting in, which holds the sort's static scratch.
constexpr size_t kMinSmemLimit = 48 * 1024;

// Refuses a limit below kMinSmemLimit, naming the shape, the need and the limit.
void check_smem_floor(const char* who, int64_t C, int64_t W, int64_t R, size_t limit) {
    TORCH_CHECK(limit >= kMinSmemLimit, who, ": C=", C, ", W=", W, ", R=", R, " needs ",
                kMinSmemLimit, " bytes of shared memory per block (the sort's scratch); this GPU "
                "grants ", limit);
}

// Static shared memory of a pair kernel instantiation (its scan scratch), as compiled.
template <bool LEAN>
size_t pairs_static_smem() {
    static const size_t bytes = [] {
        cudaFuncAttributes attr;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, p1_pairs_kernel<LEAN>));
        return attr.sharedSizeBytes;
    }();
    return bytes;
}

// Layout of the pair kernel: both triangles in one block when they fit beside the marks and
// count bytes (within kMaxJointTriangleBytes), else one block per triangle (split). LEAN when
// even a split block cannot hold the count bytes on this GPU (e.g. A100 at C = 256 with 128
// slots): marks only, walks to the group heads (same terms, same order, same bits).
struct PairsConfig {
    bool split;
    bool lean;
    size_t smem;   // dynamic shared memory per block
    size_t need;   // dynamic plus static: what the GPU must grant a block
};

PairsConfig pairs_config(int C, int W, int R, size_t limit) {
    const size_t tri = sizeof(float) * C * (C + 1) / 2;
    const size_t cw = static_cast<size_t>(C) * W, cr = static_cast<size_t>(C) * R;
    const size_t mk = (W + 31) / 32 * 4 * C, mq = (R + 31) / 32 * 4 * C;   // mark bytes
    const size_t joint_limit = std::min(kMaxJointTriangleBytes, limit > 4096 ? limit - 4096 : 0);
    PairsConfig p;
    p.split = 2 * tri + mk + mq + cw + cr > joint_limit;
    p.smem = (p.split ? tri + std::max(mk + cw, mq + cr) : 2 * tri + mk + mq + cw + cr) + 4;
    p.need = p.smem + pairs_static_smem<false>();
    p.lean = p.need > limit;   // only when split: a joint layout fits within limit - 4096
    if (p.lean) {
        p.split = true;
        p.smem = tri + std::max(mk, mq) + 4;
        p.need = p.smem + pairs_static_smem<true>();
    }
    return p;
}

// The parallel-walk kernel's layout beside the ordered kernel's (the same split, so both index their blocks
// alike): its triangles, word sums, count bytes and as many term slots (a float and a column byte each) as fit,
// aiming at two resident blocks per SM. smem 0: no room (the kernel does not run). How many slots there are only
// decides which blocks decline: never a bit.
struct W2Config {
    int cap;       // term slots per block
    size_t smem;   // dynamic shared memory per block (0: the kernel does not run)
};

size_t w2_static_smem() {
    static const size_t bytes = [] {
        cudaFuncAttributes attr;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, p1_pairs_w2_kernel));
        return attr.sharedSizeBytes;
    }();
    return bytes;
}

static_assert(kW2HcG >= 1 && kW2HcG <= kHcG, "the two-slice pair kernel takes only chunks without a hot candidate");
constexpr size_t kW2TwoPerSm = 112 * 1024;   // two blocks per SM on GPUs with 228 KB per SM
constexpr int kW2MinCap = 1024;

W2Config w2_config(int C, int W, int R, bool split, size_t limit) {
    W2Config w{0, 0};
    const size_t tri = sizeof(float) * C * (C + 1) / 2;
    const size_t wk = (W + 31) / 32, rk = (R + 31) / 32;
    const size_t words = split ? C * std::max(wk, rk) : C * (wk + rk);
    const size_t kbb = split ? static_cast<size_t>(C) * std::max(W, R) : static_cast<size_t>(C) * (W + R);
    const size_t nwm = (static_cast<size_t>(C) * (W + R) + 31) / 32;   // mark words
    // count bytes rounded up to 4 (their zeroing and padding; the columns start after it)
    const size_t fixed = (split ? 1 : 2) * tri + 4 * (words + 1) + 4 * nwm + (kbb + 3) / 4 * 4;
    const size_t st = w2_static_smem();
    if (limit < st + fixed + 5 * static_cast<size_t>(kW2MinCap)) return w;
    const size_t avail = limit - st;
    const size_t want_cap = std::max<size_t>(4096, static_cast<size_t>(C) * (W + R) / 3);
    size_t smem = std::min(fixed + 5 * want_cap, avail);
    if (smem > kW2TwoPerSm && fixed + 5 * 4096 <= kW2TwoPerSm) smem = kW2TwoPerSm;
    const size_t cap = (smem - fixed) / 5;
    w.cap = static_cast<int>(cap);
    w.smem = fixed + 5 * cap;
    return w;
}

// The output fields a scan block stages within its budget (a staged field turns one
// scattered global store per element into a shared-memory store and a contiguous copy).
// First choice: every per-entry field and both position maps (`all`; the flags then derive
// from the packed positions). Otherwise greedily by element size, the flags as fields.
StageLayout stage_layout(int C, int W, int R, size_t val_bytes, bool read_groups,
                         size_t budget, bool* all, size_t* bytes, bool want_pos, bool want_flags,
                         bool want_gidx) {
    struct Spec {
        Field f;
        size_t elem, count;
    };
    const size_t cw = static_cast<size_t>(C) * W, cr = static_cast<size_t>(C) * R;
    auto place = [&](std::vector<Spec> fields, bool greedy) {
        std::stable_sort(fields.begin(), fields.end(),
                         [](const Spec& x, const Spec& y) { return x.elem < y.elem; });
        StageLayout st{};
        size_t end = 0;
        for (const Spec& x : fields) {
            const size_t at = (end + 15) / 16 * 16;
            if (at + x.elem * x.count > budget) {
                if (!greedy) return std::make_pair(StageLayout{}, size_t{0});
                continue;
            }
            st.mask |= 1u << x.f;
            st.offset[x.f] = static_cast<int>(at);
            end = at + x.elem * x.count;
        }
        return std::make_pair(st, end);
    };
    // ALL derives the flags from the staged positions, so it stages them for either.
    std::vector<Spec> every = {{kCpe, val_bytes, cw}, {kKEff, val_bytes, cw}, {kQW, val_bytes, cr},
                               {kLogK, 4, cw}, {kRex, 4, cw}, {kLogQ, 4, cr}};
    if (want_pos || want_flags) {
        every.push_back({kPosK, 4, cw});
        every.push_back({kPosQ, 4, cr});
    }
    if (want_gidx) every.push_back({kDcKpos, 4, cw});   // dc_kpos from the scan loop
    auto full = place(every, false);
    *all = full.first.mask != 0;
    if (!*all) {
        // A fixed field order (ties of element size keep it), minus the fields not requested.
        std::vector<Spec> fields;
        if (want_flags) {
            fields.push_back({kIsLast, 1, cw});
            fields.push_back({kIsOnly, 1, cw});
            if (read_groups) fields.push_back({kQFlags, 1, cr});
        }
        fields.push_back({kCpe, val_bytes, cw});
        fields.push_back({kKEff, val_bytes, cw});
        fields.push_back({kQW, val_bytes, cr});
        if (want_pos) {
            fields.push_back({kPosK, 4, cw});
            fields.push_back({kPosQ, 4, cr});
        }
        fields.push_back({kLogK, 4, cw});
        fields.push_back({kRex, 4, cw});
        fields.push_back({kLogQ, 4, cr});
        if (want_gidx) fields.push_back({kDcKpos, 4, cw});
        full = place(fields, true);
    }
    *bytes = full.second;
    return full.first;
}

// The specialized instantiations exist for 16-bit keys and bfloat16 values only (the deterministic path's types).
template <typename KeyT, typename ValT>
constexpr bool kScanFsTypes = std::is_same<KeyT, uint16_t>::value && std::is_same<ValT, __nv_bfloat16>::value;

template <typename KeyT, typename ValT, int CL>
auto scan_fs_kernel(uint32_t fs) -> decltype(&p1_scan_kernel<KeyT, ValT, true, SegSum, CL, 0u>) {
    if constexpr (kScanFsTypes<KeyT, ValT>) {
        if (fs == 296u) return p1_scan_kernel<KeyT, ValT, true, SegSum, CL, 296u>;
        if (fs == 952u) return p1_scan_kernel<KeyT, ValT, true, SegSum, CL, 952u>;
        if (fs == 2872u) return p1_scan_kernel<KeyT, ValT, true, SegSum, CL, 2872u>;
    }
    TORCH_CHECK(false, "p1_forward: internal error: no field-set scan instantiation for ", fs);
    return nullptr;
}

// The `cl`-CTA cluster form of `kernel` can run here: the device advertises cluster launches and the occupancy
// calculator gives the launch (clusters of `cl` CTAs of `threads` threads, `smem` dynamic bytes) at least one resident
// cluster. Cached per (device, kernel, smem).
template <typename K>
bool cluster_launch_ok(K kernel, int cl, int threads, size_t smem, int device) {
    static std::mutex mu;
    static std::map<std::tuple<int, const void*, size_t>, bool> done;
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_tuple(device, reinterpret_cast<const void*>(kernel), smem);
    const auto it = done.find(key);
    if (it != done.end()) return it->second;
    int advertised = 0, clusters = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&advertised, cudaDevAttrClusterLaunch, device));
    if (advertised) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(static_cast<unsigned>(cl));
        cfg.blockDim = dim3(static_cast<unsigned>(threads));
        cfg.dynamicSmemBytes = smem;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = static_cast<unsigned>(cl);
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &cfg) != cudaSuccess) {
            (void)cudaGetLastError();   // the query is not supported here: no cluster form
            clusters = 0;
        }
    }
    done[key] = clusters > 0;
    return clusters > 0;
}

template <typename IdxT, typename KeyT, typename ValT>
std::vector<at::Tensor> run_forward(const Args& a) {
    const int64_t N = a.k_idx.size(0);
    const int C = a.k_idx.size(1), W = a.k_idx.size(2), R = a.q_idx.size(2);
    const int n = C * (W + R);
    const int row = (n + kRowAlign - 1) / kRowAlign * kRowAlign;
    const auto opts = a.k_val.options();
    const auto f32 = opts.dtype(at::kFloat);
    const auto stream = at::cuda::getCurrentCUDAStream();
    // Refused up front, before any launch: shapes whose pair sums do not fit this GPU.
    const size_t limit = smem_limit_of(a.k_idx.get_device(), a.smem_limit);
    check_smem_floor("p1_forward", C, W, R, limit);
    const PairsConfig pc = pairs_config(C, W, R, limit);
    TORCH_CHECK(pc.need <= limit, "p1_forward: C=", C, ", W=", W, ", R=", R, " needs ", pc.need,
                " bytes of shared memory per block for the pair sums; this GPU grants ", limit);

    // Rows longer than one tile are merged between two buffers.
    const int tiles = (n + kKeysTile - 1) / kKeysTile;
    at::Tensor keys[2], pays[2];
    for (int i = 0; i < (tiles == 1 ? 1 : 2); ++i) {
        keys[i] = at::empty({N, row}, opts.dtype(sizeof(KeyT) == 2 ? at::kShort : at::kInt));
        pays[i] = at::empty({N, row}, opts.dtype(at::kInt));
    }
    auto key_ptr = [&](int i) { return reinterpret_cast<KeyT*>(keys[i].data_ptr()); };
    auto pay_ptr = [&](int i) { return reinterpret_cast<uint32_t*>(pays[i].data_ptr()); };

    // Hot/cold dispatch: from the shapes only (C, W, R and the compile-time kHcG; a hot slot needs kHcG
    // distinct tokens) and, per architecture, the modern backend (the caller's routing predicate: sm_90 / 100 / 103
    // with 227 KB of opt-in shared memory per block; the device's compute capability >= 9.0 as a guard). Elsewhere
    // (A100, RTX parts) the ordered kernel below runs (it reads the position maps, which are then computed whatever
    // det_fields says). Decided before the keys launch, whose first block zeroes the unmarked-chunk count.
    const HcLayout hl = C >= kHcG ? hc_layout(C, W, R, row, kHcG) : HcLayout{};
    int hc_cc_major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&hc_cc_major, cudaDevAttrComputeCapabilityMajor, a.k_idx.get_device()));
    const bool modern = a.modern && hc_cc_major >= 9;
    const bool use_hc = hl.total > 0 && static_cast<size_t>(hl.total) <= kHcSmemBudget && modern;
    // With the hot/cold kernel, the chunks without a hot group (no group of >= kW2HcG valid K entries; kW2HcG <=
    // kHcG, so the hot/cold kernel has no candidate there and its result is the ordered walk's, bit for bit) go to the
    // parallel-walk kernel: the scan marks the hot chunks, the parallel-walk kernel computes the others, the hot/cold
    // kernel the marked ones and those the parallel-walk kernel declined. Its item counts (and the claims past them, up to
    // + kW2Grid) must stay below 2^31.
    const bool w2_items_ok = N <= (int64_t{1} << 29);
    const W2Config w2h = use_hc && w2_items_ok ? w2_config(C, W, R, false, limit) : W2Config{0, 0};
    at::Tensor w2h_flags;   // [N] hot marks / parallel-walk results, then its work counter and the unmarked-chunk count
    if (w2h.smem > 0) w2h_flags = at::empty({N + 2}, opts.dtype(at::kInt));   // the count is zeroed by p1_keys_zw_kernel

    p1_keys_zw_kernel<IdxT, KeyT><<<dim3(N, tiles), kKeysThreads, 0, stream>>>(
        a.k_idx.data_ptr<IdxT>(), a.q_idx.data_ptr<IdxT>(), key_ptr(0), pay_ptr(0), C, W, R,
        row, a.P, a.stream_offset, N - a.P, a.valid_last, a.key_bits,
        w2h.smem > 0 ? reinterpret_cast<uint32_t*>(w2h_flags.data_ptr<int32_t>()) + N + 1 : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int cur = 0;
    const int merge_blocks = (n + kMergeThreads * kMergeItems - 1) / (kMergeThreads * kMergeItems);
    for (int run = kKeysTile; run < n; run *= 2, cur ^= 1) {
        p1_merge_kernel<KeyT><<<dim3(N, merge_blocks), kMergeThreads, 0, stream>>>(
            key_ptr(cur), pay_ptr(cur), key_ptr(cur ^ 1), pay_ptr(cur ^ 1), n, row, run);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int w2_min = kW2MinItems;
    // The parallel-walk kernel for the ordered kernel (shapes without the hot/cold kernel): decided here so the scan can zero its work counter.
    const W2Config w2 = use_hc || pc.lean || !w2_items_ok ? W2Config{0, 0} : w2_config(C, W, R, pc.split, limit);
    // narrowed only when the parallel-walk kernel runs (N <= 2^29: at most 2^30 items); 0 otherwise (never used then).
    const int w2_items = w2.smem > 0 ? static_cast<int>(N * (pc.split ? 2 : 1)) : 0;
    at::Tensor w2_flags;    // [items] parallel-walk results, then its work counter
    if (w2.smem > 0) w2_flags = at::empty({w2_items + 1}, opts.dtype(at::kInt));

    // Which optional outputs to compute (det_fields); the others are None, or empty
    // tensors for the fields Phase1 always carries (is_last, is_only, the records).
    const int64_t df = a.det_fields;
    const bool want_pos = (df & kDetPos) || !use_hc;
    const bool want_flags = (df & kDetFlags) != 0;
    const bool want_rank = (df & kDetSpayRank) != 0;
    const bool want_krec = (df & kDetKrec) != 0, want_kspa = (df & kDetKspa) != 0;
    const bool want_qrec = a.read_groups && (df & kDetQrec);
    const bool want_tiles = (df & kDetTiles) != 0, want_gidx = (df & kDetDcGidx) != 0;
    const bool want_qrec8 = a.read_groups && (df & kDetQrec8);
    // dc_kpos reads the write row's entries back from spay_k or the records' first word.
    const bool want_spk = (df & kDetSpayK) || (want_gidx && !want_krec);
    const bool want_spq = a.read_groups && (df & kDetSpayQ);

    // Records {payload, hi, lo, value} of the sorted entries for the pair walks.
    auto rec = at::empty({N, row, 4}, opts.dtype(at::kInt));
    auto log_k = at::empty({N, C, W}, f32);
    auto log_q = at::empty({N, C, R}, f32);
    auto rex = at::empty({N, C, W}, f32);
    auto cpe = at::empty({N, C, W}, opts);
    auto is_last = at::empty({want_flags ? N : 0, C, W}, opts.dtype(at::kChar));
    auto is_only = at::empty({want_flags ? N : 0, C, W}, opts.dtype(at::kChar));
    auto k_eff = at::empty({N, C, W}, opts);
    auto q_w = at::empty({N, C, R}, opts);
    at::Tensor pos_k, pos_q, spay_k, spay_rank, q_last, q_only, spay_q;  // undefined (None) unless needed
    if (want_pos) {
        pos_k = at::empty({N, C, W}, opts.dtype(at::kInt));
        pos_q = at::empty({N, C, R}, opts.dtype(at::kInt));
    }
    if (want_spk) spay_k = at::empty({N, C * W}, opts.dtype(at::kInt));
    if (want_rank) spay_rank = at::empty({N, (n + 31) / 32, 2}, opts.dtype(at::kInt));
    if (a.read_groups && want_flags) {
        q_last = at::empty({N, C, R}, opts.dtype(at::kChar));
        q_only = at::empty({N, C, R}, opts.dtype(at::kChar));
    }
    if (want_spq) spay_q = at::empty({N, C * R}, opts.dtype(at::kInt));
    auto ptr_or_null = [](const at::Tensor& t) -> void* { return t.defined() ? t.data_ptr() : nullptr; };
    const Outputs<ValT> out{log_k.data_ptr<float>(),
                            rex.data_ptr<float>(),
                            reinterpret_cast<ValT*>(cpe.data_ptr()),
                            reinterpret_cast<ValT*>(k_eff.data_ptr()),
                            want_flags ? is_last.data_ptr<int8_t>() : nullptr,
                            want_flags ? is_only.data_ptr<int8_t>() : nullptr,
                            static_cast<int32_t*>(ptr_or_null(pos_k)),
                            log_q.data_ptr<float>(),
                            reinterpret_cast<ValT*>(q_w.data_ptr()),
                            static_cast<int32_t*>(ptr_or_null(pos_q)),
                            static_cast<int8_t*>(ptr_or_null(q_last)),
                            static_cast<int8_t*>(ptr_or_null(q_only)),
                            static_cast<int32_t*>(ptr_or_null(spay_k)),
                            static_cast<int32_t*>(ptr_or_null(spay_q)),
                            static_cast<int2*>(ptr_or_null(spay_rank))};
    // The staging budget, within what the device grants one block beside the static scratch
    // (dc_kpos is a staged field).
    const size_t budget = std::min(kDetMaxStageBytes, limit - 4096);
    size_t stage_smem = 0;
    bool all = false;
    const StageLayout stage = stage_layout(C, W, R, sizeof(ValT), a.read_groups, budget, &all,
                                           &stage_smem, want_pos, want_flags, want_gidx);
    // The narrow scan element when no group can count 2^15 entries of a kind (see Elem).
    const bool narrow = C * W < (1 << 15) && C * R < (1 << 15);
    const auto scan = !narrow ? p1_scan_kernel<KeyT, ValT, false, Seg4>
                      : all   ? p1_scan_kernel<KeyT, ValT, true, SegSum>
                              : p1_scan_kernel<KeyT, ValT, false, SegSum>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(scan, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        stage_smem));
    const int g_width = a.g.size(2);
    // Empty (not None) unless requested, so every record output is a tensor.
    const auto i32 = opts.dtype(at::kInt);
    at::Tensor krec = at::empty({0}, i32), kspa = at::empty({0}, f32), qrec = at::empty({0}, i32),
               tiles_k = at::empty({0}, i32), tiles_q = at::empty({0}, i32), segb_k = at::empty({0}, i32),
               dc_kpos = at::empty({0}, i32), dc_bpre = at::empty({0}, i32), dc_rec = at::empty({0}, i32),
               qrec8 = at::empty({0}, i32);
    Records recs{};
    recs.L = 1;
    recs.l_shift = 0;
    recs.P = 1;
    if (want_krec || want_kspa || want_qrec || want_tiles || want_gidx || want_qrec8) {
        if (want_kspa) {
            TORCH_CHECK(a.beta.defined() && a.beta.numel() > 0, "p1_forward: kspa needs beta");
            TORCH_CHECK(a.beta.scalar_type() == at::kFloat && a.beta.numel() == N * C && a.beta.is_cuda() &&
                        a.beta.is_contiguous(), "p1_forward: beta must be contiguous float32 [N, C]");
            TORCH_CHECK(a.beta.device() == a.k_idx.device(), "p1_forward: beta must be on the inputs' device");
        }
        TORCH_CHECK(!want_tiles || (a.rec_L >= 1 && a.rec_L <= (1 << 20)),
                    "p1_forward: tiles need a tile length in [1, 2^20]");
        // Records hold the absolute slot id (key + stream * stream_offset) in 32 bits; the bound is
        // checked without overflow: (P - 1) * stream_offset + 2^key_bits - 1 <= INT32_MAX.
        if (want_krec || want_qrec || want_qrec8) {
            const int64_t imax = std::numeric_limits<int32_t>::max();
            const int64_t kmax = (int64_t{1} << a.key_bits) - 1;
            const bool ok = a.stream_offset >= 0 && kmax <= imax &&
                            (a.P <= 1 || a.stream_offset <= (imax - kmax) / static_cast<int64_t>(a.P - 1));
            TORCH_CHECK(ok, "p1_forward: records need absolute slot ids below 2^31");
        }
        const int L = want_tiles ? a.rec_L : 1;
        const int64_t ntk = (static_cast<int64_t>(C) * W + L - 1) / L;
        const int64_t ntq = (static_cast<int64_t>(C) * R + L - 1) / L;
        if (want_krec) krec = at::empty({N, C * W, 4}, i32);
        if (want_kspa) kspa = at::empty({N, C * W}, f32);
        if (want_qrec) qrec = at::empty({N, C * R, 4}, i32);
        if (want_tiles) {
            tiles_k = at::empty({N, ntk + 1}, i32);
            segb_k = at::empty({N, ntk + 1}, i32);
            if (a.read_groups) tiles_q = at::empty({N, ntq + 1}, i32);
        }
        if (want_gidx) {
            dc_kpos = at::empty({N, C, W}, i32);
            dc_bpre = at::empty({N, (C * W + 31) / 32 + 1}, i32);
        }
        if (want_qrec8) qrec8 = at::empty({N, C * R, 2}, i32);
        auto p32 = [](const at::Tensor& t) { return t.numel() > 0 ? t.data_ptr<int32_t>() : nullptr; };
        int l_shift = -1;
        for (int b = 0; b < 31; ++b) {
            if ((1 << b) == L) l_shift = b;
        }
        recs = Records{reinterpret_cast<int4*>(p32(krec)), want_kspa ? kspa.data_ptr<float>() : nullptr,
                       reinterpret_cast<int4*>(p32(qrec)), reinterpret_cast<int2*>(p32(qrec8)),
                       p32(tiles_k), p32(tiles_q), p32(segb_k),
                       p32(dc_kpos), p32(dc_bpre), reinterpret_cast<int4*>(p32(dc_rec)),
                       want_kspa ? a.beta.data_ptr<float>() : nullptr, L, l_shift, a.P,
                       a.stream_offset, want_krec || want_qrec || want_qrec8};
    }
    if (w2h.smem > 0) {
        recs.hot = reinterpret_cast<uint32_t*>(w2h_flags.data_ptr<int32_t>());
        recs.hot_g = kW2HcG;
        recs.w2_claim = recs.hot + N;
        recs.w2_elig = recs.hot + N + 1;
    } else if (w2.smem > 0) {
        recs.w2_claim = reinterpret_cast<uint32_t*>(w2_flags.data_ptr<int32_t>()) + w2_items;
    }
    // Thread carry: per-thread reverse prefixes of the tiles that leave registers (16 B per thread and tile).
    const int scan_tiles = (n + kScanTile - 1) / kScanTile;
    // Short rows of exactly kScanCl tiles run the scan as clusters of kScanCl CTAs per chunk when the chunks fit one
    // wave, on the modern backend, where the device advertises cluster launches and the occupancy calculator gives the
    // launch at least one resident cluster (shapes and architecture only; CL = 1 otherwise, same bits).
    const int sms = a.sms;
    const bool scan_cl_shape = narrow && scan_tiles == kScanCl && modern && N <= scan_cl_max_chunks(sms);
    // The dispatch layer's masks (no-grad 296, saving 952 / 2872) run instantiations with that compile-time field set
    // when every field outside the mask has no destination in this call (else: the generic kernel).
    uint32_t fs = 0;
    {
        const uint32_t m = static_cast<uint32_t>(df);
        if (kScanFsTypes<KeyT, ValT> && narrow && all && (m == 296u || m == 952u || m == 2872u)) {
            auto ok = [&](uint32_t bit, bool present) { return (m & bit) != 0u || !present; };
            if (ok(kFsSpayK, out.spay_k != nullptr) && ok(kFsSpayQ, out.spay_q != nullptr) &&
                ok(kFsKrec, recs.krec != nullptr) && ok(kFsKspa, recs.kspa != nullptr) && ok(kFsQrec, recs.qrec != nullptr) &&
                ok(kFsQrec8, recs.qrec8 != nullptr) &&
                ok(kFsTiles, recs.tiles_k != nullptr || recs.tiles_q != nullptr || recs.segb_k != nullptr) &&
                ok(kFsDcGidx, recs.dc_kpos != nullptr || recs.dc_bpre != nullptr) && ok(kFsDcRec, recs.dc_rec != nullptr)) {
                fs = m;
            }
        }
    }
    // put<ALL> has no global-store path: under ALL, every staged-capable field with a destination must be staged.
    TORCH_CHECK(!all || ((out.pos_k == nullptr || stage.staged(kPosK)) && (out.pos_q == nullptr || stage.staged(kPosQ)) &&
                         (recs.dc_kpos == nullptr || stage.staged(kDcKpos))),
                "p1_forward: internal error: an ALL stage layout misses a requested field");
    auto scan2 = all ? p1_scan_kernel<KeyT, ValT, true, SegSum, kScanCl>
                     : p1_scan_kernel<KeyT, ValT, false, SegSum, kScanCl>;
    if (fs != 0u) scan2 = scan_fs_kernel<KeyT, ValT, kScanCl>(fs);
    const bool scan_cl = scan_cl_shape && cluster_launch_ok(scan2, kScanCl, kScanThreads, stage_smem,
                                                            a.k_idx.get_device());
    const at::Tensor tcarry = at::empty({kScanThreadCarry && scan_tiles > 1 && !scan_cl ? N : 0, scan_tiles, kScanThreads, 4},
                                        opts.dtype(at::kInt));
    if (scan_cl) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(scan2, cudaFuncAttributeMaxDynamicSharedMemorySize, stage_smem));
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(static_cast<unsigned>(N * kScanCl));
        cfg.blockDim = dim3(kScanThreads);
        cfg.dynamicSmemBytes = stage_smem;
        cfg.stream = stream;
        cudaLaunchAttribute cl_attr[1];
        cl_attr[0].id = cudaLaunchAttributeClusterDimension;
        cl_attr[0].val.clusterDim.x = kScanCl;
        cl_attr[0].val.clusterDim.y = 1;
        cl_attr[0].val.clusterDim.z = 1;
        cfg.attrs = cl_attr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg, scan2, key_ptr(cur), pay_ptr(cur), reinterpret_cast<uint4*>(rec.data_ptr()), a.g.data_ptr<float>(),
            reinterpret_cast<const ValT*>(a.k_val.data_ptr()), reinterpret_cast<const ValT*>(a.q_val.data_ptr()), out,
            stage, reinterpret_cast<unsigned long long*>(a.repeats.data_ptr<int64_t>()), C, W, R, row, g_width,
            g_width == 1 ? 0 : 1, recs, static_cast<uint4*>(nullptr)));
    } else {
        auto scan1 = scan;
        if (fs != 0u) {
            scan1 = scan_fs_kernel<KeyT, ValT, 1>(fs);
            C10_CUDA_CHECK(cudaFuncSetAttribute(scan1, cudaFuncAttributeMaxDynamicSharedMemorySize, stage_smem));
        }
        scan1<<<N, kScanThreads, stage_smem, stream>>>(
            key_ptr(cur), pay_ptr(cur), reinterpret_cast<uint4*>(rec.data_ptr()), a.g.data_ptr<float>(),
            reinterpret_cast<const ValT*>(a.k_val.data_ptr()),
            reinterpret_cast<const ValT*>(a.q_val.data_ptr()), out, stage,
            reinterpret_cast<unsigned long long*>(a.repeats.data_ptr<int64_t>()), C, W, R, row,
            g_width, g_width == 1 ? 0 : 1, recs,
            tcarry.numel() > 0 ? reinterpret_cast<uint4*>(tcarry.data_ptr<int32_t>()) : nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto A = at::empty({N, C, C}, f32);
    auto QK = at::empty({N, C, C}, f32);
    if (use_hc) {
        {
            int optin = 0;
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  a.k_idx.get_device()));
            TORCH_CHECK(static_cast<size_t>(hl.total) + 4096 <= static_cast<size_t>(optin),
                        "p1_forward: the hot-candidate pair kernel needs ", hl.total + 4096,
                        " bytes of shared memory per block; this device grants ", optin);
            const auto hck = p1_pairs_hc_kernel;
            C10_CUDA_CHECK(cudaFuncSetAttribute(hck, cudaFuncAttributeMaxDynamicSharedMemorySize, hl.total));
            long long* prof = nullptr;
            uint32_t* w2f = nullptr;
            if (w2h.smem > 0) {
                w2f = reinterpret_cast<uint32_t*>(w2h_flags.data_ptr<int32_t>());
                C10_CUDA_CHECK(cudaFuncSetAttribute(p1_pairs_w2_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, w2h.smem));
                p1_pairs_w2_kernel<<<static_cast<int>(std::min<int64_t>(N, kW2Grid)), kW2Threads, w2h.smem, stream>>>(
                    pay_ptr(cur), reinterpret_cast<const uint4*>(rec.data_ptr()), A.data_ptr<float>(),
                    QK.data_ptr<float>(), w2f, w2f + N, static_cast<int>(N), C, W, R, row, false, w2h.cap, true,
                    w2f + N + 1, w2_min);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            hck<<<N, kHcThreads, hl.total, stream>>>(
                pay_ptr(cur), reinterpret_cast<const uint4*>(rec.data_ptr()),
                static_cast<const int32_t*>(ptr_or_null(pos_k)), static_cast<const int32_t*>(ptr_or_null(pos_q)),
                A.data_ptr<float>(), QK.data_ptr<float>(), C, W, R, row, kHcG, hl, prof, w2f);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return {A,     QK,  log_k,     log_q, rex,   cpe,    is_last, is_only, k_eff,
                    q_w,   pays[cur], pos_k, pos_q, q_last, q_only, spay_k, spay_q, spay_rank,
                    krec, kspa, qrec, tiles_k, tiles_q, segb_k, dc_kpos, dc_bpre, dc_rec, qrec8};
        }
    }
    // The parallel-walk kernel first (the same sums, walks in parallel) when its layout fits; the ordered kernel
    // then computes only the blocks it declined (w2_flags), else every block.
    if (w2.smem > 0) {
        uint32_t* f = reinterpret_cast<uint32_t*>(w2_flags.data_ptr<int32_t>());
        C10_CUDA_CHECK(cudaFuncSetAttribute(p1_pairs_w2_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, w2.smem));
        p1_pairs_w2_kernel<<<std::min(w2_items, kW2Grid), kW2Threads, w2.smem, stream>>>(
            pay_ptr(cur), reinterpret_cast<const uint4*>(rec.data_ptr()), A.data_ptr<float>(),
            QK.data_ptr<float>(), f, f + w2_items, w2_items, C, W, R, row, pc.split, w2.cap, false, nullptr, 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // A block holds its triangles, a mark bit and (unless lean) a count byte per (token, lane)
    // of its K (A) or Q (QK) rows; see pairs_config.
    const auto pairs = pc.lean ? p1_pairs_kernel<true> : p1_pairs_kernel<false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(pairs, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        pc.smem));
    // One thread per (row, triangle) unit.
    const int pair_threads = ((pc.split ? C : 2 * C) + 31) / 32 * 32;
    pairs<<<N * (pc.split ? 2 : 1), pair_threads, pc.smem, stream>>>(
        pay_ptr(cur), reinterpret_cast<const uint4*>(rec.data_ptr()), pos_k.data_ptr<int32_t>(),
        pos_q.data_ptr<int32_t>(), A.data_ptr<float>(), QK.data_ptr<float>(), C, W, R, row,
        pc.split, w2.smem > 0 ? reinterpret_cast<const uint32_t*>(w2_flags.data_ptr<int32_t>()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {A,     QK,  log_k,     log_q, rex,   cpe,    is_last, is_only, k_eff,
            q_w,   pays[cur], pos_k, pos_q, q_last, q_only, spay_k, spay_q, spay_rank,
            krec, kspa, qrec, tiles_k, tiles_q, segb_k, dc_kpos, dc_bpre, dc_rec, qrec8};
}

template <typename T>
struct Tag {
    using type = T;
};

// The pair kernel's layout for a chunk shape: {split, lean, dynamic shared memory per block,
// the limit it was chosen for, dynamic plus static} (for tests and diagnostics).
std::vector<int64_t> p1_pairs_config(int64_t C, int64_t W, int64_t R, int64_t smem_limit) {
    TORCH_CHECK(C >= 1 && C <= 256 && W >= 1 && W <= 256 && R >= 1 && R <= 256,
                "p1_pairs_config: needs chunk size 1..256 and 1..256 write/read slots per token");
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    const size_t limit = smem_limit_of(device, smem_limit);
    check_smem_floor("p1_pairs_config", C, W, R, limit);
    const PairsConfig p = pairs_config(C, W, R, limit);
    return {p.split, p.lean, static_cast<int64_t>(p.smem), static_cast<int64_t>(limit),
            static_cast<int64_t>(p.need)};
}

std::vector<at::Tensor> p1_forward(
    at::Tensor k_idx, at::Tensor q_idx, at::Tensor k_val, at::Tensor q_val, at::Tensor g,
    at::Tensor repeats, int64_t num_parallel, int64_t stream_offset, int64_t key_bits,
    int64_t valid_last, bool read_groups, int64_t smem_limit,
    at::Tensor beta, int64_t rec_L, int64_t det_fields, bool modern, int64_t sms) {
    TORCH_CHECK(k_idx.dim() == 3 && q_idx.dim() == 3 && k_val.sizes() == k_idx.sizes() &&
                q_val.sizes() == q_idx.sizes() && g.dim() == 3,
                "p1_forward: expected k_idx/k_val [N, C, W], q_idx/q_val [N, C, R], g [N, C, G]");
    const int64_t N = k_idx.size(0), C = k_idx.size(1), W = k_idx.size(2), R = q_idx.size(2);
    TORCH_CHECK(q_idx.size(0) == N && q_idx.size(1) == C && g.size(0) == N && g.size(1) == C &&
                (g.size(2) == 1 || g.size(2) == W), "p1_forward: inconsistent shapes");
    TORCH_CHECK(N >= 1 && C >= 1 && C <= 256 && W >= 1 && W <= 256 && R >= 1 && R <= 256,
                "p1_forward: needs chunk size 1..256 and 1..256 write/read slots per token");
    TORCH_CHECK(num_parallel >= 1 && N % num_parallel == 0,
                "p1_forward: num_parallel must divide the chunk count");
    TORCH_CHECK(valid_last >= 1 && valid_last <= C, "p1_forward: valid_last must be in 1..C");
    TORCH_CHECK(key_bits >= 1 && key_bits <= 32, "p1_forward: key_bits must be in 1..32");
    TORCH_CHECK(k_idx.scalar_type() == q_idx.scalar_type() &&
                (k_idx.scalar_type() == at::kInt || k_idx.scalar_type() == at::kLong),
                "p1_forward: indices must be int32 or int64");
    TORCH_CHECK(k_val.scalar_type() == q_val.scalar_type() &&
                (k_val.scalar_type() == at::kFloat || k_val.scalar_type() == at::kBFloat16),
                "p1_forward: k_val and q_val must both be float32 or bfloat16");
    TORCH_CHECK(g.scalar_type() == at::kFloat && repeats.scalar_type() == at::kLong &&
                repeats.numel() == 1, "p1_forward: g must be float32, repeats int64 [1]");
    for (const auto* t : {&k_idx, &q_idx, &k_val, &q_val, &g, &repeats}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->device() == k_idx.device(),
                    "p1_forward: inputs must be contiguous tensors on one CUDA device");
    }
    const c10::cuda::CUDAGuard guard(k_idx.device());
    const Args args{k_idx, q_idx, k_val, q_val, g, repeats, static_cast<int>(num_parallel),
                    stream_offset, static_cast<int>(key_bits), static_cast<int>(valid_last),
                    read_groups, smem_limit, beta,
                    static_cast<int>(std::min<int64_t>(std::max<int64_t>(rec_L, -1), int64_t{1} << 21)),
                    det_fields, modern, static_cast<int>(sms)};
    TORCH_CHECK(det_fields >= 0 && det_fields <= kDetAll && (det_fields & kDetDcRec) == 0,
                "p1_forward: det_fields has unknown bits");
    auto by_value = [&](auto idx, auto key) {
        using I = typename decltype(idx)::type;
        using K = typename decltype(key)::type;
        return k_val.scalar_type() == at::kFloat ? run_forward<I, K, float>(args)
                                                 : run_forward<I, K, __nv_bfloat16>(args);
    };
    auto by_key = [&](auto idx) {
        return key_bits <= 16 ? by_value(idx, Tag<uint16_t>{}) : by_value(idx, Tag<uint32_t>{});
    };
    return k_idx.scalar_type() == at::kInt ? by_key(Tag<int32_t>{}) : by_key(Tag<int64_t>{});
}
"""


class Phase1(NamedTuple):
    """Phase-1 outputs for N chunks of C tokens with W write and R read slots per token.

    ``cpe``, ``k_eff`` and ``q_w`` have the dtype of ``k_val``; flags are int8. Entries of
    invalid tokens are neutral: zero logs, sums and weights, cpe 1, flags 0, and zero rows
    and columns in ``a`` and ``qk``. Fields not requested (``need_read_groups``, ``det_fields``)
    are None, except is_last / is_only and the record fields, which are then empty tensors.
    """

    a: torch.Tensor        # [N, C, C] float32, strictly lower triangular
    qk: torch.Tensor       # [N, C, C] float32, lower triangular
    log_k: torch.Tensor    # [N, C, W] float32
    log_q: torch.Tensor    # [N, C, R] float32
    rex: torch.Tensor      # [N, C, W] float32
    cpe: torch.Tensor      # [N, C, W]
    is_last: torch.Tensor  # [N, C, W] int8
    is_only: torch.Tensor  # [N, C, W] int8
    k_eff: torch.Tensor    # [N, C, W]
    q_w: torch.Tensor      # [N, C, R]
    payload: torch.Tensor  # [N, row] int32, sorted payloads (see phase1_common.cuh)
    pos_k: Optional[torch.Tensor]  # [N, C, W] int32, packed position + group role of each K entry
    pos_q: Optional[torch.Tensor]  # [N, C, R] int32, packed position + group role of each Q entry
    q_last: Optional[torch.Tensor]  # [N, C, R] int8, last valid Q entry of its slot group
    q_only: Optional[torch.Tensor]  # [N, C, R] int8, the only valid Q entry of its slot group
    # The chunk's write / read entries in sorted order (reference: tests/phase1_reference.py).
    spay_k: Optional[torch.Tensor] = None   # [N, C * W] int32
    spay_q: Optional[torch.Tensor] = None   # [N, C * R] int32
    # Per 32 positions of the sorted row: write entries before them, bitmask of write entries
    # among them (reference: tests/phase1_reference.py).
    spay_rank: Optional[torch.Tensor] = None   # [N, ceil(C * (W + R) / 32), 2] int32
    # Mover records and indices (per det_fields; empty tensors when not
    # requested; see Records in the CUDA source). L = rec_tile.
    krec: Optional[torch.Tensor] = None        # [N, C * W, 4] int32
    kspa: Optional[torch.Tensor] = None        # [N, C * W] float32
    qrec: Optional[torch.Tensor] = None        # [N, C * R, 4] int32
    tiles_k: Optional[torch.Tensor] = None     # [N, ceil(C * W / L) + 1] int32
    tiles_q: Optional[torch.Tensor] = None     # [N, ceil(C * R / L) + 1] int32
    segb_k: Optional[torch.Tensor] = None      # [N, ceil(C * W / L) + 1] int32
    dc_kpos: Optional[torch.Tensor] = None     # [N, C, W] int32
    dc_bpre: Optional[torch.Tensor] = None     # [N, ceil(C * W / 32) + 1] int32
    dc_rec: Optional[torch.Tensor] = None      # [N, C * W, 4] int32
    qrec8: Optional[torch.Tensor] = None       # [N, C * R, 2] int32 (DET_QREC8)


# det_fields bits of phase1_forward: which outputs to compute. Unrequested
# Optional fields are None; is_last / is_only and the record fields are then empty tensors.
DET_POS = 1          # pos_k, pos_q (always computed when the ordered pair kernel runs)
DET_SPAY_RANK = 2    # spay_rank
DET_FLAGS = 4        # is_last, is_only, q_last, q_only
DET_SPAY_K = 8       # spay_k
DET_SPAY_Q = 16      # spay_q
DET_KREC = 32        # krec: {spay_k element, slot, kv * exp(rex), cpe}
DET_KSPA = 64        # kspa: -beta * kv * exp(log_k) (needs beta)
DET_QREC = 128       # qrec: {spay_q element, slot, q * exp(log_q), 0}
DET_TILES = 256      # tiles_k, tiles_q, segb_k (tile length rec_tile, default 128)
DET_DC_GIDX = 512    # dc_kpos, dc_bpre
DET_QREC8 = 2048     # qrec8: {slot, q * exp(log_q)} per sorted read position (= qrec words 1-2)
DET_SHIPPED = DET_POS | DET_SPAY_RANK | DET_FLAGS | DET_SPAY_K | DET_SPAY_Q


# Packing of pos_k / pos_q: the entry's position in its chunk's sorted row, and how its slot
# group looks from it. "Peers" are the entries of the group that write the same bank row: the
# K entries for pos_k, the valid Q entries for pos_q. The two counts saturate at
# POS_COUNT_MAX, which is safe for any decision bounded by it (a group is split over at most
# 32 workers) while the exact size always comes from the walk itself.
POS_BITS = 17                      # positions 0..131071 (C * (W + R) <= 256 * 512)
POS_MASK = (1 << POS_BITS) - 1
POS_AFTER_SHIFT = POS_BITS         # peers after this entry in the group, saturating
POS_SIZE_SHIFT = POS_BITS + 7      # peers in the group, saturating
POS_COUNT_MAX = 127


def pos_pack(pos, after, size):
    """The packed value of ``Phase1.pos_k`` / ``pos_q`` (torch tensors or ints)."""
    after = after.clamp(max=POS_COUNT_MAX) if torch.is_tensor(after) else min(after, POS_COUNT_MAX)
    size = size.clamp(max=POS_COUNT_MAX) if torch.is_tensor(size) else min(size, POS_COUNT_MAX)
    return pos | (after << POS_AFTER_SHIFT) | (size << POS_SIZE_SHIFT)


# Sorted write-only / read-only rows (spay_k / spay_q): a chunk's C * W write (C * R read)
# entries in the order of its sorted payload row, so every slot group's entries of that kind
# are consecutive; padding tokens' entries come last. Each element keeps the payload's entry
# bits (lane 0-7, Q 8, token 9-16, invalid 17) and places it in its group:
SPAY_ENTRY_MASK = 0x3FFFF
SPAY_DIST_SHIFT = 18        # bits 18-29: distance back to the group's first element in this
SPAY_DIST_MAX = 4095        # row, saturating (0 for the first element and for padding)
# A group is the valid entries of the row's kind with the same bank row; a padding-token
# entry is a group of its own (FIRST | LAST | invalid) and writes or reads nothing. Always
# FIRST(j) = (j == 0) or LAST(j - 1).


@functools.cache
def _module():
    with open(os.path.join(os.path.dirname(__file__), _HEADER), encoding="utf-8") as f:
        header = f.read()
    return load_cuda_extension(
        "sdm_phase1",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=("p1_forward", "p1_pairs_config"),
        headers={_HEADER: header, **device_headers()},
        extra_cuda_cflags=("-O3",),
    )


def phase1_forward(
    k_idx: torch.Tensor,
    q_idx: torch.Tensor,
    k_val: torch.Tensor,
    q_val: torch.Tensor,
    g: torch.Tensor,
    repeats: torch.Tensor,
    *,
    num_parallel: int,
    stream_offset: int,
    num_keys: int,
    valid_last: int,
    need_read_groups: bool = True,
    beta: Optional[torch.Tensor] = None,
    rec_tile: int = 0,
    det_fields: Optional[int] = None,
) -> Phase1:
    """Phase 1 of N = num_parallel * steps time-major chunks.

        Args:
            k_idx, q_idx: [N, C, W] / [N, C, R] int32 or int64 slot ids. Chunk c belongs to
                stream c % num_parallel, whose ids lie in [s * stream_offset,
                s * stream_offset + num_keys). The K slots of one token must be distinct.
            k_val, q_val: [N, C, W] / [N, C, R] float32 or bfloat16 weights.
            g: [N, C, 1] or [N, C, W] float32 log-gates (per token or per write).
            repeats: int64 [1] counter, incremented by the number of K entries that repeat a
                slot of their token.
            valid_last: valid tokens in the last chunk of each stream (1..C); the last
                num_parallel chunks are the streams' last chunks.
            need_read_groups: compute q_last, q_only and spay_q (the read side of the backward
                needs them); when False they are None and nothing is allocated for them.
            beta: [N, C] float32, needed by kspa.
            rec_tile: tile length L of tiles_k / tiles_q / segb_k (default 128 when requested).
            det_fields: DET_* bits of the optional outputs to compute. Default: DET_SHIPPED,
                plus DET_KREC | DET_KSPA | DET_QREC | DET_TILES when beta is given.

        Every output is bitwise reproducible for fixed inputs, shapes and GPU type.
"""
    key_bits = max(1, (num_keys - 1).bit_length())
    if det_fields is None:
        det_fields = DET_SHIPPED | ((DET_KREC | DET_KSPA | DET_QREC | DET_TILES) if beta is not None else 0)
    if (det_fields & DET_TILES) and rec_tile <= 0:
        rec_tile = 128
    return Phase1(*_module().p1_forward(
        k_idx, q_idx, k_val, q_val, g, repeats, num_parallel, stream_offset, key_bits,
        valid_last, need_read_groups, SMEM_LIMIT_OVERRIDE or smem_cap(k_idx.device),
        beta if beta is not None else torch.empty(0, device=k_idx.device),
        int(rec_tile), int(det_fields), backend(k_idx.device) == "modern", num_sms(k_idx.device)))


def pairs_config(C: int, W: int, R: int) -> dict:
    """How the pair kernel runs chunks of C tokens with W / R slots on the current GPU (under
    ``SMEM_LIMIT_OVERRIDE``): split (one block per triangle), lean (no count bytes: walks run
    to the group heads), its dynamic shared memory per block, the limit used, and need
    (dynamic plus static: the shape runs when need <= limit)."""
    split, lean, smem, limit, need = _module().p1_pairs_config(C, W, R, SMEM_LIMIT_OVERRIDE or smem_cap())
    return dict(split=bool(split), lean=bool(lean), smem=smem, limit=limit, need=need)
