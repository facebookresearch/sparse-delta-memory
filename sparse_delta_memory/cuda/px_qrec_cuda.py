# Copyright (c) Meta Platforms, Inc. and affiliates.
"""px_qrec_cuda: compact read records (qrec8) for the deterministic backward's read-side mover (GDS), built on a
side stream, one chunk step at a time.

qrec8[c, j] = {absolute slot, f32 bits of __fmul_rn(f32(q_val), expf(log_q))} for sorted read position j of chunk c
(the order of Phase1.spay_q[c]); padding and invalid entries are {0, 0}. The records are bitwise equal to words 1-2 of
Phase 1's qrec (DET_QREC) and to Phase 1's own qrec8 (DET_QREC8): the same inputs (the saved q_idx, q_val, log_q) and
the same expression.

build_qrec8_steps(...) returns (qrec8 int32 [Nt*P, C*R, 2], events). events[ti] is recorded once rows
[ti*P, (ti+1)*P) are written (every step is queued up front, in backward order Nt-1..0), and the consumer
(gx_gds_cuda) waits on events[ti] before step ti. events is None when one launch on the current stream builds every
step (CUDA graph capture, or Nt == 1).

Why the side-stream path is safe:
- Fork: side.wait_stream(cur) orders every side launch after the work already queued on the current stream (the inputs
  are complete; qrec8, allocated on the current stream just before, is not used by queued work).
- Join: the consumer waits on events[ti] on the current stream before step ti, the only reader of those rows. Every
  step is waited on, so all side work is complete, as seen from the current stream, once the backward loop has run.
- Allocator: qrec8, every input and every contiguous copy made here are marked record_stream(side), so the caching
  allocator keeps their blocks from reuse until the side work queued so far has completed. No host synchronization."""

from __future__ import annotations

import functools

import torch

from ._build import load_cuda_extension

__all__ = ["build_qrec8", "build_qrec8_steps"]

_CPP_SOURCE = r"""
void px_qrec8(at::Tensor spay_q, at::Tensor q_idx, at::Tensor q_val, at::Tensor log_q, at::Tensor out,
              int64_t c0, int64_t c1);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {
constexpr uint32_t kInvalid = 1u << 17;
__device__ __forceinline__ float f_of(float v) { return v; }
__device__ __forceinline__ float f_of(__nv_bfloat16 v) { return __bfloat162float(v); }

// One thread per sorted read position of chunks [c0, c1): the records of qrec words 1-2.
template <typename IdxT, typename ValT>
__global__ void __launch_bounds__(256) px_qrec8_records_kernel(
    const int32_t* __restrict__ spay_q, const IdxT* __restrict__ q_idx, const ValT* __restrict__ q_val,
    const float* __restrict__ log_q, int2* __restrict__ out, int64_t first, int64_t total, int n, int R) {
    const int64_t i = first + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t chunk = i / n;
    const uint32_t e = static_cast<uint32_t>(spay_q[i]);
    int slot = 0;
    float qs = 0.0f;
    if (!(e & kInvalid)) {
        const int64_t o = chunk * n + static_cast<int64_t>((e >> 9) & 0xFFu) * R + (e & 0xFFu);
        slot = static_cast<int>(q_idx[o]);
        qs = __fmul_rn(f_of(q_val[o]), expf(log_q[o]));
    }
    __stcs(out + i, make_int2(slot, __float_as_int(qs)));
}
}  // namespace

void px_qrec8(at::Tensor spay_q, at::Tensor q_idx, at::Tensor q_val, at::Tensor log_q, at::Tensor out,
              int64_t c0, int64_t c1) {
    const c10::cuda::CUDAGuard guard(spay_q.device());
    TORCH_CHECK(spay_q.dim() == 2 && spay_q.scalar_type() == at::kInt && spay_q.is_contiguous(), "px_qrec8: spay_q");
    const int64_t N = spay_q.size(0), n = spay_q.size(1);
    TORCH_CHECK(q_idx.dim() == 3 && q_idx.size(0) == N && q_idx.size(1) * q_idx.size(2) == n &&
                (q_idx.scalar_type() == at::kInt || q_idx.scalar_type() == at::kLong) && q_idx.is_contiguous(),
                "px_qrec8: q_idx [N, C, R] int32/int64");
    TORCH_CHECK(q_val.sizes() == q_idx.sizes() && q_val.is_contiguous() &&
                (q_val.scalar_type() == at::kFloat || q_val.scalar_type() == at::kBFloat16), "px_qrec8: q_val");
    TORCH_CHECK(log_q.sizes() == q_idx.sizes() && log_q.is_contiguous() && log_q.scalar_type() == at::kFloat,
                "px_qrec8: log_q");
    TORCH_CHECK(out.numel() == N * n * 2 && out.scalar_type() == at::kInt && out.is_contiguous() &&
                (reinterpret_cast<uintptr_t>(out.data_ptr()) & 7) == 0, "px_qrec8: out [N, n, 2] int32");
    for (const at::Tensor* t : {&q_idx, &q_val, &log_q, &out})
        TORCH_CHECK(t->device() == spay_q.device(), "px_qrec8: one device");
    TORCH_CHECK(0 <= c0 && c0 <= c1 && c1 <= N, "px_qrec8: chunk range");
    const int64_t first = c0 * n, total = c1 * n;
    if (total == first) return;
    const int R = static_cast<int>(q_idx.size(2));
    const int blocks = static_cast<int>((total - first + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* o2 = reinterpret_cast<int2*>(out.data_ptr<int32_t>());
#define PX_Q8(IDX, VAL)                                                                                   \
    px_qrec8_records_kernel<IDX, VAL><<<blocks, 256, 0, stream>>>(                                        \
        spay_q.data_ptr<int32_t>(), q_idx.data_ptr<IDX>(), reinterpret_cast<const VAL*>(q_val.data_ptr()),  \
        log_q.data_ptr<float>(), o2, first, total, static_cast<int>(n), R)
    const bool i64 = q_idx.scalar_type() == at::kLong, bf = q_val.scalar_type() == at::kBFloat16;
    if (i64 && bf) { PX_Q8(int64_t, __nv_bfloat16); }
    else if (i64) { PX_Q8(int64_t, float); }
    else if (bf) { PX_Q8(int32_t, __nv_bfloat16); }
    else { PX_Q8(int32_t, float); }
#undef PX_Q8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_px_qrec",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=("px_qrec8",),
        extra_cuda_cflags=("-O3",),
    )


def build_qrec8(spay_q, q_idx, q_val, log_q):
    """qrec8 int32 [N, C*R, 2] in one launch on the current stream (see the module docstring)."""
    n = spay_q.shape[-1]
    sq = spay_q.reshape(-1, n).contiguous()
    N = sq.shape[0]
    out = torch.empty(N, n, 2, device=sq.device, dtype=torch.int32)
    _module().px_qrec8(sq, q_idx.reshape(N, -1, q_idx.shape[-1]).contiguous(),
                       q_val.reshape(N, -1, q_val.shape[-1]).contiguous(),
                       log_q.reshape(N, -1, log_q.shape[-1]).contiguous(), out, 0, N)
    return out


_SIDE = {}


def build_qrec8_steps(spay_q, q_idx, q_val, log_q, *, P, Nt):
    """(qrec8 int32 [Nt*P, C*R, 2], events or None): records built per chunk step on a side stream, steps queued in
    backward order; events[ti] marks rows [ti*P, (ti+1)*P) as written. Under CUDA graph capture on the inputs' device,
    or for Nt == 1, one launch on the current stream builds every step (events None). Same bits either way. Slots are
    narrowed to int32 as in Phase 1's qrec: the forward that produced spay_q / log_q has checked the absolute-slot bound."""
    n = spay_q.shape[-1]
    N = spay_q.numel() // n
    if N != Nt * P:
        raise ValueError(f"build_qrec8_steps: {N} chunks != Nt * P = {Nt} * {P}")
    dev = spay_q.device
    with torch.cuda.device(dev):
        if Nt <= 1 or torch.cuda.is_current_stream_capturing():
            return build_qrec8(spay_q, q_idx, q_val, log_q), None
        sq = spay_q.reshape(-1, n).contiguous()
        qi = q_idx.reshape(N, -1, q_idx.shape[-1]).contiguous()
        qv = q_val.reshape(N, -1, q_val.shape[-1]).contiguous()
        lq = log_q.reshape(N, -1, log_q.shape[-1]).contiguous()
        cur = torch.cuda.current_stream(dev)
        side = _SIDE.get(dev)
        if side is None:
            side = _SIDE.setdefault(dev, torch.cuda.Stream(device=dev))
        out = torch.empty(N, n, 2, device=dev, dtype=torch.int32)
        mod = _module()
        side.wait_stream(cur)
        events = [None] * Nt
        with torch.cuda.stream(side):
            for ti in reversed(range(Nt)):
                mod.px_qrec8(sq, qi, qv, lq, out, ti * P, (ti + 1) * P)
                ev = torch.cuda.Event()
                ev.record(side)
                events[ti] = ev
        for t in (out, sq, qi, qv, lq):
            t.record_stream(side)
    return out, events
