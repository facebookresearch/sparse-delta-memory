# Copyright (c) Meta Platforms, Inc. and affiliates.
"""_build: lazy, content-addressed JIT builds of the package's CUDA extensions.

The build directory and the imported module are keyed only by the extension name. A
process that loses the build lock imports whatever shared object the winner produced
without checking that it came from the same sources. Every extension here is therefore
named ``<base>_<sha256(build inputs)[:16]>``, where the build inputs are the sources,
the headers, the exported bindings, all compiler flags, the torch/CUDA/Python versions,
the CUDA toolkit location and the target GPU architectures. Identical inputs share one
cache entry; anything else gets its own directory, so a stale ``.so`` is never reused
and two source trees can share one ``TORCH_EXTENSIONS_DIR``.

Each entry of ``cuda_sources`` is its own translation unit, so ninja compiles them in
parallel (``MAX_JOBS`` caps the job count). ``headers`` are written next to the sources,
so any unit can ``#include "<name>"``. The toolkit's CCCL directory (CUB, Thrust,
libcu++) is on the include path.

Builds happen on the first call, never at import. Call each op once eagerly (for
example in a warmup step) before capturing a CUDA graph so that no compilation is
triggered during capture."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import warnings
from collections.abc import Mapping, Sequence
from types import ModuleType

import torch
from torch.utils import cpp_extension

from . import _image

__all__ = ["load_cuda_extension"]

_LOCK = threading.Lock()


def _as_tuple(value: str | Sequence[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _require_toolchain() -> str:
    """Return CUDA_HOME, or raise a readable error if the toolkit is unusable."""
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "The SDM CUDA kernels need a CUDA build of PyTorch and a visible GPU."
        )
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise RuntimeError(
            "The SDM CUDA kernels are JIT-compiled and need the CUDA toolkit, but "
            "PyTorch could not locate it. Set CUDA_HOME to a toolkit that matches "
            f"torch.version.cuda={torch.version.cuda} (it must contain bin/nvcc)."
        )
    nvcc = os.environ.get("PYTORCH_NVCC") or os.path.join(cuda_home, "bin", "nvcc")
    if not (os.path.exists(nvcc) or shutil.which(nvcc)):
        raise RuntimeError(
            f"nvcc was not found at {nvcc!r}. Set CUDA_HOME to a full CUDA toolkit "
            f"matching torch.version.cuda={torch.version.cuda}."
        )
    return cuda_home


def toolkit() -> tuple:
    """The toolkit an extension build uses (cpp_extension.CUDA_HOME, PYTORCH_NVCC): part of _image's build environment."""
    return cpp_extension.CUDA_HOME, os.environ.get("PYTORCH_NVCC")


def _cccl_include_paths(cuda_home: str) -> tuple[str, ...]:
    """CUDA 13 moved CUB/Thrust/libcu++ to ``include/cccl``; older toolkits keep them
    directly under ``include``, which PyTorch already adds."""
    path = os.path.join(cuda_home, "include", "cccl")
    return (path,) if os.path.isdir(path) else ()


def _target_architectures() -> dict[str, object]:
    """The inputs PyTorch uses to choose the -gencode flags."""
    capabilities = sorted(
        {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    )
    return {
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "capabilities": [list(c) for c in capabilities],
    }


def _header_items(headers: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    items = tuple(sorted((headers or {}).items()))
    for file_name, _ in items:
        if not file_name or os.path.basename(file_name) != file_name:
            raise ValueError(f"header names must be plain file names, got {file_name!r}")
    return items


def _extension_name(
    base: str,
    *,
    cpp_sources: str | Sequence[str],
    cuda_sources: str | Sequence[str],
    functions: Sequence[str],
    headers: Mapping[str, str] | None = None,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
) -> str:
    """Return ``<base>_<16 hex digits>`` identifying one exact build."""
    if not base.isidentifier():
        raise ValueError(f"extension base name must be an identifier, got {base!r}")
    cuda_home = _require_toolchain()
    identity = {
        "sources": {"cpp": _as_tuple(cpp_sources), "cuda": _as_tuple(cuda_sources)},
        "headers": _header_items(headers),
        "functions": tuple(functions),
        "flags": {
            "cflags": tuple(extra_cflags),
            "cuda_cflags": tuple(extra_cuda_cflags),
            "ldflags": tuple(extra_ldflags),
            "include_paths": tuple(extra_include_paths),
        },
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.implementation.cache_tag,
        "cuda_home": os.path.realpath(cuda_home),
        "arch": _target_architectures(),
    }
    nvcc_env = {k: os.environ[k] for k in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS") if os.environ.get(k)}
    if nvcc_env:   # nvcc's environment flags shape the targets: a build under them is its own build (_image)
        identity["nvcc_env"] = nvcc_env
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return f"{base}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _warn_if_locked(build_dir: str) -> None:
    """A build killed mid-compile leaves a lock file that makes later loads wait forever."""
    lock = os.path.join(build_dir, "lock")
    if os.path.exists(lock):
        warnings.warn(
            f"{lock} exists. If no other process is currently building this "
            "extension, delete that file; otherwise this load will wait for it.",
            stacklevel=3,
        )


def _write_if_changed(path: str, text: str) -> str:
    """Write atomically, and leave an identical file untouched so ninja skips it."""
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == text:
                return path
    except FileNotFoundError:
        pass
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def _bindings_source(cpp_sources: tuple[str, ...], functions: tuple[str, ...]) -> str:
    """The C++ unit holding the declarations and the pybind11 module."""
    defs = [f'  m.def("{f}", torch::wrap_pybind_function({f}), "{f}");' for f in functions]
    return "\n".join(
        ["#include <torch/extension.h>", *cpp_sources,
         "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {", *defs, "}", ""]
    )


def load_cuda_extension(
    name: str,
    *,
    cpp_sources: str | Sequence[str],
    cuda_sources: str | Sequence[str],
    functions: Sequence[str],
    headers: Mapping[str, str] | None = None,
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
    verbose: bool = False,
) -> ModuleType:
    """Build (on first use) and import a JIT-compiled CUDA extension.

    Args:
        name: identifier prefix of the extension; the content hash is appended.
        cpp_sources: C++ declarations of the bound functions.
        cuda_sources: CUDA translation unit(s); each string is compiled separately.
        functions: names to expose through pybind11; each must be declared in
            ``cpp_sources``. The loaded module is checked to export all of them.
        headers: ``{file name: text}`` of headers the units include as
            ``#include "<file name>"``.
        extra_cflags / extra_cuda_cflags / extra_ldflags / extra_include_paths:
            forwarded to the compiler and linker and included in the content hash.
        verbose: forwarded to ``torch.utils.cpp_extension.load``.

    Returns the imported module. Callers cache it: each call hashes the inputs and loads
    the extension.
    """
    cpp_sources, cuda_sources = _as_tuple(cpp_sources), _as_tuple(cuda_sources)
    functions = tuple(functions)
    header_items = _header_items(headers)
    if not functions:
        raise ValueError("load_cuda_extension needs at least one bound function")
    if not cuda_sources:
        raise ValueError("load_cuda_extension needs at least one CUDA source")
    full_name = _extension_name(
        name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        headers=headers,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
    )
    with _LOCK:
        build_dir = cpp_extension._get_build_directory(full_name, verbose)
        _warn_if_locked(build_dir)
        for file_name, text in header_items:
            _write_if_changed(os.path.join(build_dir, file_name), text)
        sources = [_write_if_changed(os.path.join(build_dir, "main.cpp"),
                                     _bindings_source(cpp_sources, functions))]
        sources += [_write_if_changed(os.path.join(build_dir, f"unit{i}.cu"), text)
                    for i, text in enumerate(cuda_sources)]
        module = cpp_extension.load(
            name=full_name,
            sources=sources,
            extra_cflags=list(extra_cflags),
            extra_cuda_cflags=list(extra_cuda_cflags),
            extra_ldflags=list(extra_ldflags),
            extra_include_paths=[*_cccl_include_paths(cpp_extension.CUDA_HOME),
                                 *extra_include_paths],
            build_directory=build_dir,
            verbose=verbose,
            with_cuda=True,
        )
        missing = [f for f in functions if not callable(getattr(module, f, None))]
        if missing:
            raise RuntimeError(
                f"CUDA extension {full_name!r} loaded from "
                f"{getattr(module, '__file__', None)!r} does not export {missing}. "
                "The cached build is incomplete; delete its directory and retry."
            )
    if name != _image.PROBE_BASE:
        _image.require_supported_image()   # before any kernel of the extension runs
    return module
