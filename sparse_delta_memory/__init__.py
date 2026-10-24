# Copyright (c) Meta Platforms, Inc. and affiliates.

from .cuda_graph import reset_cuda_graphs, suspend_cuda_graphs
from .layer import SparseDeltaMemory, SparseDeltaMemoryArgs
from .cuda._image import UntestedHardwareWarning
from .memory_ops import UnsupportedConfigurationError

__all__ = ["SparseDeltaMemory", "SparseDeltaMemoryArgs", "UnsupportedConfigurationError", "UntestedHardwareWarning",
           "reset_cuda_graphs", "suspend_cuda_graphs"]

# The single source of the version: pyproject.toml reads it (distribution metadata), and it holds
# for a checkout or a vendored copy as well.
__version__ = "2.0.0"
