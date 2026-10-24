# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Standalone GPU smoke test for SparseDeltaMemory (same checks as test_smoke_sdm.py).

    PYTHONPATH=. python tests/smoke_sdm.py
"""
from test_smoke_sdm import main

if __name__ == "__main__":
    main()
