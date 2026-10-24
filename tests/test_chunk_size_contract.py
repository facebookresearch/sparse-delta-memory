# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Chunk-size contract of the training op: any chunk size in 16..256 runs correctly for any
number of tokens per stream (shorter than a chunk, ragged last chunk), and every other chunk
size is refused with a ValueError before any kernel runs.

"Correct" is measured against the unchunked float64 reference, which is independent of the
chunk size by construction (comparing chunk sizes with each other cannot see a fault they
share). The running cases execute in a separate interpreter, so an illegal memory access
fails the test instead of poisoning the CUDA context of the rest of the suite.
"""
import os
import subprocess
import sys

import pytest
import torch

from sdm_testlib import GATE_FP32, requires_cuda

pytestmark = requires_cuda

SPH, DIM, W, R, P = 512, 64, 4, 4, 2

# (tokens per stream T, chunk_size C)
RUNS = [(256, 16), (256, 32), (256, 64), (256, 128), (256, 256),
        (128, 256),                 # T < C
        (5, 48),                    # T < C, C not a power of two
        (192, 128), (96, 64),       # ragged last chunk
        (256, 96), (100, 17)]       # C not a power of two, ragged last chunk
REFUSED = [(256, 512), (256, 257), (256, 15), (256, 3), (256, 1)]

_HERE = os.path.dirname(os.path.abspath(__file__))


def _worker():
    torch.backends.cuda.matmul.allow_tf32 = False
    from sdm_testlib import kernel_write_read, make_op_inputs, oracle_write_read, rel_max

    for T, C in RUNS:
        d = make_op_inputs(P, T, W, R, DIM, SPH, torch.float32, seed=5)
        readings, grads, G = kernel_write_read(d, chunk_size=C, sph=SPH)
        torch.cuda.synchronize()
        ro, og, _ = oracle_write_read(d, G, sph=SPH)
        worst = max([rel_max(ro, readings)] + [rel_max(og[n], grads[n]) for n in og])
        print(f"REL {T} {C} {worst:.6e}", flush=True)


def test_every_supported_chunk_size_matches_the_float64_reference():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(_HERE), _HERE, os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], capture_output=True,
                       text=True, timeout=1800, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    rel = {(int(t), int(c)): float(v) for _, t, c, v in
           (line.split() for line in r.stdout.splitlines() if line.startswith("REL "))}
    assert sorted(rel) == sorted(RUNS)
    bad = {k: v for k, v in rel.items() if not v < GATE_FP32}
    assert not bad, f"(T, C): rel_max from the float64 reference: {bad}"


@pytest.mark.parametrize("T,C", REFUSED, ids=[f"T{t}_C{c}" for t, c in REFUSED])
def test_unsupported_chunk_sizes_are_refused(T, C):
    from sdm_testlib import kernel_write_read, make_op_inputs

    d = make_op_inputs(P, T, W, R, DIM, SPH, torch.float32, seed=5)
    with pytest.raises(ValueError, match="chunk_size"):
        kernel_write_read(d, chunk_size=C, sph=SPH)


if __name__ == "__main__":
    _worker()
