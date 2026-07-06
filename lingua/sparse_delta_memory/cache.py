# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Minimal cache/state classes for SparseDeltaMemory.

Extracted (self-contained) from the internal memory-controller layer so this
open-source package has no dependency on the original framework. ``Cache`` is a
tiny ABC; ``SDMLayerState`` holds the memory bank plus the running sequence
length for inference/decode.
"""

from abc import ABC, abstractmethod

import torch


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize a tensor along a given dimension."""
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class Cache(ABC):
    """Base class for all caches. A model's full cache is a ``list[Cache]``."""

    @abstractmethod
    def __getitem__(self, *args, **kwargs) -> torch.Tensor | int:
        pass

    @property
    @abstractmethod
    def cache_len(self) -> int:
        pass

    @abstractmethod
    def update_(self, *args, **kwargs) -> "Cache":
        """Update the cache in place."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the state."""
        pass


class SDMLayerState(Cache):
    """Inference/decode state for a ``SparseDeltaMemory`` layer: the memory bank
    plus the number of tokens consumed so far."""

    def __init__(self, memory: torch.Tensor, seq_len: int = 0) -> None:
        self.memory = memory
        self._cache_len = seq_len

    def __getitem__(self, key_idx: int) -> torch.Tensor | None:
        if key_idx == 0:
            return self.memory
        raise KeyError

    def __len__(self) -> int:
        return 1

    @property
    def cache_len(self) -> int:
        return self._cache_len

    def reset(self) -> None:
        self.memory.zero_()
        self._cache_len = 0

    def update_(self, new_memory: torch.Tensor, seq_len: int = 1) -> Cache:
        assert self.memory is not None
        self.memory = new_memory.detach()
        self._cache_len += seq_len
        return self
