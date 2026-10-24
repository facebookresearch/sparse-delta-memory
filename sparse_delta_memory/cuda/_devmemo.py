"""Memoized device-class reads: _device's backend() and capability() serve exactly the originals' values (or raise their
exceptions, never cached) from one bounded memo keyed by (read, device ordinal, forced test row), so a call makes no
device query after the device's first one."""
from . import _device as _dev_mod
from ._device import CLASS_SMEM, RESERVED_SMEM

_DEVCLS: dict = {}   # ("backend" | "capability", device ordinal, _device._FORCED) -> the original's value (successful reads only)


def _memo(read, idx, compute):
    """compute(idx) memoized under (read, idx, _device._FORCED); an exception propagates and is never cached."""
    hit = _DEVCLS.get((read, idx, _dev_mod._FORCED))
    if hit is None:
        with _dev_mod._FORCE_LOCK:
            key = (read, idx, _dev_mod._FORCED)
            hit = _DEVCLS.get(key)
            if hit is None:
                hit = compute(idx)
                if len(_DEVCLS) >= 64:
                    _DEVCLS.clear()
                _DEVCLS[key] = hit
    return hit


def _raw_capability(idx) -> tuple:
    return tuple(_dev_mod.capability(idx))


def backend(device=None) -> str:
    """_device.backend(device), memoized."""
    return _memo("backend", _dev_mod._index(device), _dev_mod.backend)


def capability(device=None) -> tuple:
    """_device.capability(device) on every row, supported or not (a forced row's directly), memoized."""
    forced = _dev_mod._FORCED
    if forced:
        return forced[0]
    return _memo("capability", _dev_mod._index(device), _raw_capability)


def cuda_backend(device=None) -> bool:
    """_device.cuda_backend(device), memoized."""
    return bool(backend(device))


def pdl_ok(device=None) -> bool:
    """_device.pdl_ok(device), memoized."""
    return backend(device) == "modern"


def smem_cap(device=None) -> int:
    """_device.smem_cap(device), memoized."""
    return CLASS_SMEM[backend(device)]


def smem_per_sm(device=None) -> int:
    """_device.smem_per_sm(device), memoized."""
    return smem_cap(device) + RESERVED_SMEM


def sm90(device=None) -> bool:
    """`device` is an sm_90 part (H100): capability(device) == (9, 0), memoized."""
    return capability(device) == (9, 0)
