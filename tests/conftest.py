# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared pytest configuration for the SDM test suite."""
import os
import sys

# From a checkout, test the checkout: its sparse_delta_memory/, lingua/ and apps/ come first on
# sys.path even when pytest is launched without PYTHONPATH set. A copy of tests/ outside a
# checkout tests the installed sparse_delta_memory (the checkout-only tests then skip).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")


@pytest.fixture(autouse=True)
def _fresh_native_route_memo():
    """memory_ops.native_route_class memoizes per shape and the identities of the routing predicates it calls
    (2.0.0); a test that patches a helper below them (e.g. mover_fx_cuda.supported) must not see an earlier test's
    entry, so every test starts with an empty memo. Production never patches: the memo only saves the lookups."""
    mo = sys.modules.get("sparse_delta_memory.memory_ops")
    for name in ("_NATIVE_MEMO",):
        memo = getattr(mo, name, None)
        if memo is not None:
            memo.clear()
    yield


@pytest.fixture
def ieee_fp32():
    """Disable TF32 matmuls for the duration of a test: the fp32-bank kernels and the
    backward's cuBLAS matmuls then run in full float32, as the float64 gates assume."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
