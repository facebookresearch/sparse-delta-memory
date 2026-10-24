# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Per-chunk WY solve (``wy_solve``): accuracy against float64 and cuBLAS, exact structure,
golden bits at C = 64, full output coverage, independence of the batch size and CUDA-graph
replay."""
import hashlib

import pytest
import torch

from sdm_testlib import GATE_FP32, bits_equal, rel_max, requires_cuda

pytestmark = requires_cuda

CHUNKS = [1, 2, 5, 16, 17, 24, 32, 48, 64, 96, 127, 128, 160, 255, 256, 512]
BLOCK_EDGES = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256]


def _solve(A, beta, QK):
    from sparse_delta_memory.wy_solve import wy_solve
    return wy_solve(A, beta, QK)


def _inputs(C, seed, scale=0.1, n=None, beta_max=1.0):
    """Gated key-key matrix A (garbage on and above the diagonal), beta and a dense QK."""
    n = n or (24 if C <= 256 else 6)
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(n, C, C, device="cuda", generator=g) * (scale / (C / 64) ** 0.5)
    beta = torch.rand(n, C, device="cuda", generator=g) * (beta_max - 0.1) + 0.1
    QK = torch.randn(n, C, C, device="cuda", generator=g)
    return A, beta, QK


def _float64(A, beta, QK):
    C = A.shape[-1]
    eye = torch.eye(C, device=A.device, dtype=torch.float64)
    L = torch.tril(beta.double()[..., None] * A.double(), -1)
    X = torch.linalg.solve_triangular(eye + L, eye.expand(A.shape[0], C, C), upper=False)
    return X, torch.tril(QK.double()) @ X


def _cublas(A, beta, QK):
    C = A.shape[-1]
    eye = torch.eye(C, device=A.device)
    X = torch.linalg.solve_triangular(eye + torch.tril(beta[..., None] * A, -1),
                                      eye.expand_as(A), upper=False)
    return X, torch.bmm(torch.tril(QK), X)


def _max_err(got, ref):
    return (got.double() - ref).abs().max().item()


def _normwise_err(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("scale", [0.1, 0.3])
@pytest.mark.parametrize("C", CHUNKS)
def test_as_accurate_as_cublas(C, scale, ieee_fp32):
    """Max error within 2x of fp32 cuBLAS on fixed small batches.

    A maximum over a small batch is noisy: on other seeds an independent fp32 solver (LAPACK)
    also exceeds 2x cuBLAS. The normwise test below is the stable criterion.
    """
    A, beta, QK = _inputs(C, seed=C, scale=scale)
    outputs = zip(("X", "QX"), _solve(A, beta, QK), _float64(A, beta, QK), _cublas(A, beta, QK))
    for name, got, ref, cublas in outputs:
        err, bound = _max_err(got, ref), 2.0 * _max_err(cublas, ref)
        assert err <= bound, f"C={C} {name}: max error {err:.3e} vs 2 x cuBLAS {bound:.3e}"


@pytest.mark.parametrize("C", BLOCK_EDGES)
def test_normwise_error_at_production_size(C, ieee_fp32):
    """65536 rows, beta up to 2 (entries of X in the hundreds): normwise error within 2x of
    fp32 cuBLAS, and the largest error within the fp32 gate relative to the largest entry."""
    A, beta, QK = _inputs(C, seed=C + 7, scale=0.3, n=65536 // C, beta_max=2.0)
    outputs = zip(("X", "QX"), _solve(A, beta, QK), _float64(A, beta, QK), _cublas(A, beta, QK))
    for name, got, ref, cublas in outputs:
        err, bound = _normwise_err(got, ref), 2.0 * _normwise_err(cublas, ref)
        assert err <= bound, f"C={C} {name}: normwise error {err:.3e} vs 2 x cuBLAS {bound:.3e}"
        assert rel_max(ref, got) <= GATE_FP32, f"C={C} {name}: rel_max {rel_max(ref, got):.3e}"


@pytest.mark.parametrize("C", sorted(set(CHUNKS + BLOCK_EDGES)))
def test_exact_unit_lower_triangular_structure(C):
    X, QX = _solve(*_inputs(C, seed=C + 1))
    upper = torch.triu(torch.ones(C, C, device="cuda", dtype=torch.bool), diagonal=1)
    for name, out in (("X", X), ("QX", QX)):
        assert out.is_contiguous()
        zeros = torch.zeros_like(out[:, upper])
        assert bits_equal(out[:, upper], zeros), f"{name} upper triangle is not +0.0"
    diag = X.diagonal(dim1=-2, dim2=-1).reshape(-1)
    assert bits_equal(diag, torch.ones_like(diag)), "diagonal of X is not exactly 1.0"


@pytest.mark.parametrize("C", [17, 64, 96, 256])
def test_reads_only_the_lower_triangles(C):
    A, beta, QK = _inputs(C, seed=C + 2)
    upper_A = torch.triu(torch.ones(C, C, device="cuda", dtype=torch.bool))
    upper_QK = torch.triu(upper_A, diagonal=1)
    nan = float("nan")
    got = _solve(A.masked_fill(upper_A, nan), beta, QK.masked_fill(upper_QK, nan))
    for want, out in zip(_solve(A, beta, QK), got):
        assert bits_equal(want, out)


def _hashed_unit(n, salt):
    """n values in [-1, 1) from an integer hash: exact in float32 and platform independent."""
    h = (torch.arange(n, dtype=torch.int64) + salt) * 2654435761 % 2**32
    h = (h ^ (h >> 13)) * 1274126177 % 2**32
    return (h >> 8).to(torch.float32) / 2**23 - 1.0


def _golden_stack():
    import triton
    return (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)
            and triton.__version__.startswith("3.7"))


@pytest.mark.skipif(not _golden_stack(), reason="the digest is for sm_103 with Triton 3.7")
def test_c64_golden_bits():
    """X at C = 64 is pinned bit for bit under the pinned launch configuration."""
    N, C = 16, 64
    A = (0.25 * _hashed_unit(N * C * C, 1)).reshape(N, C, C).cuda()
    beta = (0.5 + 0.25 * _hashed_unit(N * C, 2)).reshape(N, C).cuda()
    QK = _hashed_unit(N * C * C, 3).reshape(N, C, C).cuda()
    X, _ = _solve(A, beta, QK)
    digest = hashlib.sha256(X.cpu().numpy().tobytes()).hexdigest()
    assert digest == "2bcb2ea30214bbe922f1ebbe537bb37b1170fceda2d91a0dc99d3017a67ed2ae"


@pytest.mark.parametrize("C", [17, 64, 96, 128, 256])
def test_every_output_element_is_written_and_nothing_stale_is_read(C):
    """The outputs land in recycled NaN-filled memory: level 1 must write every element
    (there is no memset) and level 2 must read back only blocks it has written."""
    A, beta, QK = _inputs(C, seed=C + 6, n=64)
    want = _solve(A, beta, QK)
    poison = [torch.full_like(A, float("nan")) for _ in want]
    recycled = {p.data_ptr() for p in poison}
    del poison
    got = _solve(A, beta, QK)
    assert {t.data_ptr() for t in got} == recycled, "the outputs did not reuse the poisoned memory"
    for w, g in zip(want, got):
        assert bits_equal(w, g)


@pytest.mark.parametrize("C", [17, 64, 128, 256])
def test_bits_do_not_depend_on_the_batch_size(C):
    A, beta, QK = _inputs(C, seed=C + 3, n=300)
    full = _solve(A, beta, QK)
    part = _solve(A[100:107], beta[100:107], QK[100:107])
    for f, p in zip(full, part):
        assert bits_equal(f[100:107], p)


@pytest.mark.parametrize("C", [16, 64, 256])
def test_graph_replay_matches_eager(C):
    A, beta, QK = _inputs(C, seed=C + 4, n=64)
    _solve(A, beta, QK)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _solve(A, beta, QK)
    fresh = _inputs(C, seed=C + 5, n=64)
    for static, new in zip((A, beta, QK), fresh):
        static.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    for want, got in zip(_solve(*fresh), captured):
        assert bits_equal(want, got)


def test_rejects_bad_inputs():
    A, beta, QK = _inputs(32, seed=0, n=4)
    with pytest.raises(TypeError, match="float32"):
        _solve(A.bfloat16(), beta, QK)
    with pytest.raises(ValueError, match=r"\[N, C, C\]"):
        _solve(A, beta, QK[:, :16, :16])
    with pytest.raises(ValueError, match=r"N \* C"):
        _solve(A, beta[:, :16], QK)
