# Copyright (c) Meta Platforms, Inc. and affiliates.
"""_device: the routing predicate (backend(): which kernels and plans serve a device) and the device properties that the
launch-shape rules scale with: SM count and L2 size, cached per device index.

The launch-shape rules were tuned on GB300 (152 SMs, 135528448-byte L2), where they reduce to the tuned constants. They
choose launch shapes only, never the arithmetic, so no output depends on them."""

import contextlib
import functools
import os
import threading

import torch


def _index(device) -> int:
    """CUDA device index of `device` (None, an int, a string or a torch.device; unindexed = the current device)."""
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    d = device if isinstance(device, torch.device) else torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


@functools.lru_cache(maxsize=None)
def _props(index: int):
    p = torch.cuda.get_device_properties(index)
    return int(p.multi_processor_count), int(p.L2_cache_size)


def num_sms(device=None) -> int:
    """Streaming multiprocessors of `device` (default: the current device)."""
    return _FORCED[2] if _FORCED else _props(_index(device))[0]


def l2_bytes(device=None) -> int:
    """L2 cache size in bytes of `device` (default: the current device)."""
    return _FORCED[3] if _FORCED and _FORCED[3] is not None else _props(_index(device))[1]


def prev_pow2(x: int) -> int:
    """Largest power of two <= x (x >= 1)."""
    return 1 << (int(x).bit_length() - 1)


# The ONE routing predicate (the supported hardware matrix; no knob). "modern": H100 / B200 / GB300 (compute capability
# 9.0 / 10.0 / 10.3 with at least 232,448 bytes of opt-in shared memory per block): the fast kernels as tuned (TMA, PDL,
# clusters, hot/cold Phase 1). "a100": A100 (8.0, 166,912 bytes): the same kernels through the portable (cp.async) side of
# their arch guards, no PDL, no clusters, the ordered Phase 1, A100 tile plans. "rtx": RTX 30 / 40 / 50 (8.6 / 8.9 / 12.0,
# 101,376 bytes): the same portable code at RTX limits (a native sm_120 build compiles the same device branches as sm_86
# / 89: sdm_copy.cuh). Every other target (below 8.0, or a capability or shared-memory limit not listed): one clear error
# naming the supported GPUs. Read from torch's device properties only, so a test that patches them forces a row.
MODERN_CAPS = ((9, 0), (10, 0), (10, 3))
MODERN_SMEM = 232448
A100_CAPS, A100_SMEM = ((8, 0),), 166912
RTX_CAPS, RTX_SMEM = ((8, 6), (8, 9), (12, 0)), 101376
UNSUPPORTED = ("sparse_delta_memory supports NVIDIA H100 / B200 / GB300 (compute capability 9.0 / 10.0 / 10.3), A100 (8.0) "
               "and RTX 30 / 40 / 50 (8.6 / 8.9 / 12.0); device {i} ({name}) has compute capability {major}.{minor} with "
               "{smem} bytes of opt-in shared memory per block")

# The device-side mirror of backend() lives in sdm_copy.cuh (SDM_PORTABLE_ARCH: every device pass but sm_90 / 100 / 103),
# with the portable copy layer; kernels that guard on the architecture include it (headers()).
_COPY_HEADER = "sdm_copy.cuh"
_PLAN_HEADER = "sdm_plancap.cuh"   # host-side launch capture of the backward's step plans (bwd_steps_cuda)


@functools.cache
def headers() -> dict:
    """{file name: text} of the shared headers for load_cuda_extension(headers=...)."""
    out = {}
    for name in (_COPY_HEADER, _PLAN_HEADER):
        with open(os.path.join(os.path.dirname(__file__), name), encoding="utf-8") as f:
            out[name] = f.read()
    return out


def capability(device=None) -> tuple:
    """Compute capability (major, minor) of `device`."""
    return _FORCED[0] if _FORCED else tuple(torch.cuda.get_device_capability(_index(device)))


def _optin(device=None) -> int:
    """The device's own opt-in shared memory per block (bytes): the classification input of backend()."""
    return _FORCED[1] if _FORCED else int(torch.cuda.get_device_properties(_index(device)).shared_memory_per_block_optin)


def backend(device=None) -> str:
    """'modern' | 'a100' | 'rtx' for `device` (see MODERN_CAPS); any other target raises RuntimeError(UNSUPPORTED)."""
    i = _index(device)
    cc, optin = capability(i), _optin(i)
    if cc in MODERN_CAPS and optin >= MODERN_SMEM:
        return "modern"
    if cc in A100_CAPS and optin == A100_SMEM:
        return "a100"
    if cc in RTX_CAPS and optin == RTX_SMEM:
        return "rtx"
    raise RuntimeError(UNSUPPORTED.format(i=i, name=torch.cuda.get_device_name(i), major=cc[0], minor=cc[1], smem=optin))


def cuda_backend(device=None) -> bool:
    """The package's CUDA kernels serve `device`: True on every backend (elsewhere, backend()'s clear error)."""
    return bool(backend(device))


def pdl_ok(device=None) -> bool:
    """Programmatic dependent launch may be requested on `device`: the modern backend only."""
    return backend(device) == "modern"


def tbind(kernel, grid, ref):
    """kernel[grid] for an SDM Triton launch (`ref`: one of its CUDA operands)."""
    return kernel[grid]


# Shared memory per block every plan of a class assumes (modern: 232,448 on any modern part; A100 / RTX exact).
CLASS_SMEM = {"modern": MODERN_SMEM, "a100": A100_SMEM, "rtx": RTX_SMEM}


def smem_cap(device=None) -> int:
    """Opt-in shared memory per block the plans of `device`'s class use (bytes; CLASS_SMEM)."""
    return CLASS_SMEM[backend(device)]


def smem_per_sm(device=None) -> int:
    """Shared memory per SM of `device`'s class (bytes): smem_cap() plus the 1 KB reserved per block."""
    return smem_cap(device) + RESERVED_SMEM


RESERVED_SMEM = 1024   # shared memory the system reserves per resident block (sm_80+)

# _force_row overrides capability(), _optin(), num_sms() and l2_bytes() for every device (process-wide; contexts nest).
_FORCED = None
_FORCE_STACK = []
_FORCE_LOCK = threading.Lock()


@contextlib.contextmanager
def _force_row(capability, smem_optin, sm_count, l2_bytes=None):
    """TEST ONLY: while active, every device reports this capability, opt-in shared memory, SM count and (given) L2, so a
    test reproduces a hardware-matrix row on another GPU. Contexts must nest (an exit out of order raises)."""
    global _FORCED
    entry = (tuple(int(c) for c in capability), int(smem_optin), int(sm_count),
             None if l2_bytes is None else int(l2_bytes))
    token = object()
    with _FORCE_LOCK:
        _FORCE_STACK.append((token, entry))
        _FORCED = entry
    top = True
    try:
        yield
    finally:
        with _FORCE_LOCK:
            top = bool(_FORCE_STACK) and _FORCE_STACK[-1][0] is token
            _FORCE_STACK[:] = [e for e in _FORCE_STACK if e[0] is not token]
            _FORCED = _FORCE_STACK[-1][1] if _FORCE_STACK else None
    if not top:   # outside the finally block: an exception in flight (e.g. ImageRefusal) is never replaced
        raise RuntimeError("_force_row: forcing contexts must nest (this one exited out of order)")
