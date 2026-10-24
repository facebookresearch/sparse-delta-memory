# Copyright (c) Meta Platforms, Inc. and affiliates.
"""_image: PTX-only builds for older GPUs are refused on Hopper / Blackwell GPUs, at every CUDA extension load
(_build.load_cuda_extension), before any kernel of the extension runs. The modern routes (H100 / B200 / GB300) use device
features only the modern passes compile (sdm_copy.cuh); the image the runtime loaded for a probe kernel, built like every
extension, must be one (cudaFuncGetAttributes().ptxVersion). ImageRefusal is a BaseException: no `except Exception`
fallback catches it. An RTX-class device warns once instead."""

import contextlib
import os
import threading
import warnings

import torch


class UntestedHardwareWarning(UserWarning):
    """sparse_delta_memory runs on a GPU class this release has not tested (RTX: compute capability 8.6 / 8.9 / 12.0): the
    results are unverified. Issued once per process per device, when the first CUDA extension loads."""


UNTESTED_MESSAGE = ("sparse_delta_memory {version} has not been tested on RTX-class GPUs: device {index} ({name}, compute "
                    "capability {major}.{minor}); results are unverified. Tested: A100 / H100 / B200 / GB300 (compute "
                    "capability 8.0 / 9.0 / 10.0 / 10.3).")


class ImageRefusal(BaseException):
    """The package refuses to run the CUDA image it loaded (sparse_delta_memory.cuda._image). A BaseException, so that no
    `except Exception` fallback reroutes it: catch it by name. The message names the image, the device and the fix."""


MODERN_PTX = (90, 100, 103)   # ptxVersion of the device passes without SDM_PORTABLE_ARCH (sdm_copy.cuh)
ENV_KEYS = ("TORCH_CUDA_ARCH_LIST", "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT", "NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS")
PROBE_BASE = "sdm_image_probe"
MESSAGE = ("sparse_delta_memory: PTX-only builds for older GPUs are refused on Hopper / Blackwell GPUs. The CUDA image "
           "loaded on device {index} ({name}, compute capability {major}.{minor}) was compiled for compute_{ptx} "
           "(cudaFuncGetAttributes ptxVersion {ptx}; {env}). Rebuild with TORCH_CUDA_ARCH_LIST including your GPU "
           "(9.0 for H100, 10.0 for B200, 10.3 for GB300), or unset it, and start a new process.")

_PROBE = {}       # build environment -> the probe extension built under it
_PTX = {}         # (compute capability, build environment) -> the probe's loaded ptxVersion
_OK = set()       # (device ordinal, forced test row, build environment) checked
_WARNED = set()   # (device ordinal, forced test row) given the RTX warning
_TLS = threading.local()   # .depth: _unchecked_build_env in this thread (test only)


def build_env() -> tuple:
    """What the probe's and every extension's images depend on: ENV_KEYS and the toolkit (_build.toolkit)."""
    from ._build import toolkit
    return (*(os.environ.get(k) for k in ENV_KEYS), *toolkit())


def env_summary() -> str:
    return ", ".join(f"{k}={os.environ.get(k)!r}" for k in ENV_KEYS)


_PROBE_CPP = "int64_t sdm_image_ptx();"
_PROBE_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void sdm_image_probe_kernel(int* p) { if (p != nullptr) *p = 0; }
int64_t sdm_image_ptx() {
    cudaFuncAttributes attr{};
    const cudaError_t err = cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(sdm_image_probe_kernel));
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        TORCH_CHECK(false, "sdm_image_ptx: cudaFuncGetAttributes failed: ", cudaGetErrorString(err));
    }
    return static_cast<int64_t>(attr.ptxVersion);
}
"""


def probe_ptx(index: int, env: tuple) -> int:
    """The ptxVersion of the image the runtime loads on device `index` for the probe kernel built under build environment
    `env` (as for every extension built under it), once per (compute capability, environment)."""
    real = tuple(int(c) for c in torch.cuda.get_device_capability(index))
    if (real, env) not in _PTX:
        if env not in _PROBE:
            from ._build import load_cuda_extension
            _PROBE[env] = load_cuda_extension(PROBE_BASE, cpp_sources=_PROBE_CPP, cuda_sources=_PROBE_SOURCE,
                                              functions=("sdm_image_ptx",))
        with torch.cuda.device(index):
            _PTX[real, env] = int(_PROBE[env].sdm_image_ptx())
    return _PTX[real, env]


def require_supported_image(device=None) -> None:
    """Refuses (ImageRefusal) a device whose routes are the modern ones (a forced test row included) while the image the
    runtime loads there is not a modern pass; an RTX-class device warns once. `device` None = every visible device, the
    current one first. Run by _build after every extension load; memoized per (device, forced row, build environment: an
    extension built under another environment is checked again, with a probe built under it), refusals never."""
    if getattr(_TLS, "depth", 0):
        return
    from . import _device
    env = build_env()
    if device is None:
        cur = torch.cuda.current_device()
        devices = [cur] + [j for j in range(torch.cuda.device_count()) if j != cur]
    else:
        devices = [_device._index(device)]
    for i in devices:
        key = (i, _device._FORCED, env)
        if key in _OK:
            continue
        routes = _device.backend(i)
        if routes == "modern" and probe_ptx(i, env) not in MODERN_PTX:
            major, minor = (int(c) for c in torch.cuda.get_device_capability(i))
            raise ImageRefusal(MESSAGE.format(index=i, name=torch.cuda.get_device_name(i), major=major, minor=minor,
                                              ptx=probe_ptx(i, env), env=env_summary()))
        if routes == "rtx" and key[:2] not in _WARNED:
            from .. import __version__
            cc = tuple(_device.capability(i))
            warnings.warn(UNTESTED_MESSAGE.format(version=__version__, index=i, name=torch.cuda.get_device_name(i),
                                                  major=cc[0], minor=cc[1]), UntestedHardwareWarning, stacklevel=2)
            _WARNED.add(key[:2])
        _OK.add(key)


@contextlib.contextmanager
def _unchecked_build_env():
    """TEST ONLY: every visible device is checked first; inside, this thread's builds (a test's own module for another
    target list, called directly, never through the package's routes) are not checked."""
    require_supported_image()
    _TLS.depth = getattr(_TLS, "depth", 0) + 1
    try:
        yield
    finally:
        _TLS.depth -= 1
