# Copyright (c) Meta Platforms, Inc. and affiliates.
"""bwd_steps_cuda: the deterministic backward's chunk-step loop in ONE C++ call: every launcher planned once per backward
(sdm_plancap.cuh), then the same kernels, parameters, launch attributes and order on the same stream."""

import functools

import torch

from ._build import load_cuda_extension

_CPP = r"""
void bwd_steps(at::Tensor gwr_plan, int64_t gwr_addr, at::Tensor gemm_plan, int64_t gemm_addr,
               at::Tensor pa_plan, int64_t pa_addr, at::Tensor gds_plan, int64_t gds_addr, int64_t nt,
               int64_t gwr_pdl, int64_t gemm_use, int64_t gds_use, int64_t device,
               at::Tensor d_u4, at::Tensor x_t4, at::Tensor d_dv4);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>

namespace {
// A planned launcher's step entry point: (plan bytes, plan size, step, pdl, cudaStream_t) -> 0 or an error code.
using PlanLaunch = int64_t (*)(const void*, int64_t, int64_t, int64_t, void*);

void check_plan(const at::Tensor& p, const char* what) {
    TORCH_CHECK(p.device().is_cpu() && p.scalar_type() == at::kByte && p.dim() == 1 && p.is_contiguous() &&
                p.numel() > 0, "bwd_steps: the ", what, " plan must be a contiguous 1-D CPU uint8 tensor");
}
}  // namespace

// For ti = nt - 1 .. 0: gwr(ti) [PDL iff ti < nt - 1 and gwr_pdl], the d_u update (the GEMM plan with gemm_use, or cuBLAS
// baddbmm_out when gemm_addr == 0), pa(ti) [its mode is fixed by its plan], gds(ti) [gds_use]: the backward loop's calls,
// in its order, on the current stream of `device` (the stream each per-step call uses). A failing launch raises; the
// launches already issued stay issued, as with a failing per-step call.
void bwd_steps(at::Tensor gwr_plan, int64_t gwr_addr, at::Tensor gemm_plan, int64_t gemm_addr,
               at::Tensor pa_plan, int64_t pa_addr, at::Tensor gds_plan, int64_t gds_addr, int64_t nt,
               int64_t gwr_pdl, int64_t gemm_use, int64_t gds_use, int64_t device,
               at::Tensor d_u4, at::Tensor x_t4, at::Tensor d_dv4) {
    check_plan(gwr_plan, "gwr");
    check_plan(pa_plan, "pa");
    check_plan(gds_plan, "gds");
    TORCH_CHECK(gwr_addr != 0 && pa_addr != 0 && gds_addr != 0, "bwd_steps: launcher entry points");
    TORCH_CHECK(nt >= 1 && nt <= std::numeric_limits<int32_t>::max(), "bwd_steps: steps");
    TORCH_CHECK((gwr_pdl == 0 || gwr_pdl == 1) && (gemm_use == 0 || gemm_use == 1) && (gds_use == 0 || gds_use == 1),
                "bwd_steps: pdl flags 0 / 1");
    const bool blas = gemm_addr == 0;
    if (blas) {
        TORCH_CHECK(d_u4.dim() == 4 && x_t4.dim() == 4 && d_dv4.dim() == 4 && d_u4.size(0) >= nt && x_t4.size(0) >= nt &&
                    d_dv4.size(0) >= nt, "bwd_steps: cuBLAS step views [Nt, P, *, *]");
    } else {
        check_plan(gemm_plan, "gemm");
    }
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    void* const st = static_cast<void*>(stream);
    const PlanLaunch gwr = reinterpret_cast<PlanLaunch>(gwr_addr);
    const PlanLaunch gemm = blas ? nullptr : reinterpret_cast<PlanLaunch>(gemm_addr);
    const PlanLaunch pa = reinterpret_cast<PlanLaunch>(pa_addr);
    const PlanLaunch gds = reinterpret_cast<PlanLaunch>(gds_addr);
    const void* gwp = gwr_plan.data_ptr();
    const void* gmp = blas ? nullptr : gemm_plan.data_ptr();
    const void* pap = pa_plan.data_ptr();
    const void* gdp = gds_plan.data_ptr();
    const int64_t gwn = gwr_plan.numel(), gmn = blas ? 0 : gemm_plan.numel(), pan = pa_plan.numel(),
                  gdn = gds_plan.numel();
    for (int64_t ti = nt - 1; ti >= 0; --ti) {
        int64_t rc = gwr(gwp, gwn, ti, (ti < nt - 1 && gwr_pdl) ? 1 : 0, st);
        TORCH_CHECK(rc == 0, "bwd_steps: dc_gwr step ", ti, " failed (", rc, ")");
        if (blas) {
            at::Tensor u = d_u4.select(0, ti);
            at::baddbmm_out(u, u, x_t4.select(0, ti), d_dv4.select(0, ti));
        } else {
            rc = gemm(gmp, gmn, ti, gemm_use, st);
            TORCH_CHECK(rc == 0, "bwd_steps: gx_gemm step ", ti, " failed (", rc, ")");
        }
        rc = pa(pap, pan, ti, 0, st);
        TORCH_CHECK(rc == 0, "bwd_steps: gx_pa2 step ", ti, " failed (", rc, ")");
        rc = gds(gdp, gdn, ti, gds_use, st);
        TORCH_CHECK(rc == 0, "bwd_steps: gx_gds step ", ti, " failed (", rc, ")");
    }
}
"""


@functools.cache
def _module():
    return load_cuda_extension(
        "sdm_bwd_steps",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=("bwd_steps",),
        extra_cuda_cflags=("-O3",),
    )


def run(gwr, gemm, pa, gds, *, steps, gwr_pdl, gemm_use, gds_use, device, blas_views=None):
    """Issue steps steps - 1 .. 0 (gwr / pa / gds / gemm: (plan, address); gemm None: cuBLAS on blas_views)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if gemm is None:
        gplan, gaddr = torch.empty(0, dtype=torch.uint8), 0
        d_u4, x_t4, d_dv4 = blas_views
    else:
        gplan, gaddr = gemm
        d_u4 = x_t4 = d_dv4 = torch.empty(0, device=device, dtype=torch.float32)
    _module().bwd_steps(gwr[0], int(gwr[1]), gplan, int(gaddr), pa[0], int(pa[1]), gds[0], int(gds[1]), int(steps),
                        int(gwr_pdl), int(gemm_use), int(gds_use), int(idx), d_u4, x_t4, d_dv4)
