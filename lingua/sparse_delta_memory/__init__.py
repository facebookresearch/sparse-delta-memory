# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Deprecated location of the Sparse Delta Memory package (v1's ``lingua.sparse_delta_memory``).

v2 moved the package to the top-level ``sparse_delta_memory`` (``pip install sparse-delta-memory``;
in this repository, ``sparse_delta_memory/``). This alias keeps v1-style imports working in a
checkout: ``lingua.sparse_delta_memory`` and ``lingua.sparse_delta_memory.<module>`` are the same
module objects as ``sparse_delta_memory`` and ``sparse_delta_memory.<module>``, never second copies
(which would duplicate the autograd functions and break ``isinstance``). Importing it warns once
per process. The modules v1 had and v2 removed raise ImportError. Not part of the pip package.
"""
import importlib
import importlib.abc
import pkgutil
import sys
import warnings

import sparse_delta_memory as _new

_OLD = __name__
_NEW = _new.__name__
_REMOVED = ("triton_argsort", "cuda.scatter_write_cuda", "cuda.sparse_ip_cuda",
            "cuda.warp_coop_gather_async_cuda", "cuda.warp_cooperative_gather_cuda")

warnings.warn(f"{_OLD} is deprecated: v2 moved the package to the top-level {_NEW} "
              f"(import {_NEW}; pip install sparse-delta-memory)", FutureWarning, stacklevel=2)

# Every module of the package, imported once (importing builds no CUDA extension) and registered
# under its old name as well, so an import through either name finds the same object.
for _info in pkgutil.walk_packages(_new.__path__, _NEW + "."):
    importlib.import_module(_info.name)
for _name, _module in list(sys.modules.items()):
    if _name == _NEW or _name.startswith(_NEW + "."):
        sys.modules[_OLD + _name[len(_NEW):]] = _module


class _OldNames(importlib.abc.MetaPathFinder):
    """Any other name under the old prefix: a clear ImportError, never a second copy of a module.
    A removed module raises ImportError under either prefix: ``from lingua.sparse_delta_memory
    import triton_argsort`` asks for ``sparse_delta_memory.triton_argsort`` (the alias is the new
    package object), and a plain ImportError, unlike ModuleNotFoundError, reaches the caller."""

    def find_spec(self, name, path=None, target=None):
        for prefix in (_OLD, _NEW):
            if name.startswith(prefix + ".") and name[len(prefix) + 1:] in _REMOVED:
                raise ImportError(f"{_OLD}.{name[len(prefix) + 1:]} was removed in v2: the chunked "
                                  f"write/read op of {_NEW}.memory_ops replaces its kernels",
                                  name=name)
        if not name.startswith(_OLD + "."):
            return None
        rel = name[len(_OLD) + 1:]
        raise ModuleNotFoundError(f"No module named {name!r} ({_OLD} is an alias of {_NEW}, "
                                  f"which has no module {rel!r})", name=name)


sys.meta_path.insert(0, _OldNames())
sys.modules[_OLD] = _new
