# Copyright (c) Meta Platforms, Inc. and affiliates.
"""fp8 / int4 compact snapshot rows written by mover_fx_cuda's quantized DW."""
import pytest
import torch

from sdm_testlib import OP_INPUTS, make_op_inputs, requires_cuda

pytestmark = [requires_cuda, pytest.mark.skipif(not torch.cuda.is_available() or
                                                torch.cuda.get_device_capability()[0] < 9, reason="needs sm_90+")]
QMODE = {"fp8": 1, "int4": 2}


def ref_quant(rows, quant):
    """The interface's encoding of f32 rows [m, hd] on the CPU, with the GPU's semantics where IEEE leaves room."""
    x = rows.cpu().float()
    amax = torch.where(torch.isnan(x), torch.zeros_like(x), x.abs()).amax(dim=1)
    qinv = torch.tensor(1.0 / (448.0 if quant == "fp8" else 7.0), dtype=torch.float32)
    sc = torch.maximum(amax * qinv, torch.tensor(1e-12, dtype=torch.float32)).to(torch.bfloat16).float()
    q = (x / sc[:, None]).to(torch.bfloat16).float()
    q = torch.where(torch.isnan(q), torch.full_like(q, float("nan")).abs(), q)   # after the bf16 round trip
    if quant == "fp8":
        return q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8), sc
    nib = torch.where(torch.isnan(q), torch.full_like(q, 7.0), q.round().clamp(-8.0, 7.0)).to(torch.int32) & 0xF
    return ((nib[:, 0::2] << 4) | nib[:, 1::2]).to(torch.uint8), sc


def _node(out):
    fn = out.grad_fn
    while fn is not None and type(fn).__name__ != "SDMWriteReadBackward":
        fn = fn.next_functions[0][0]
    assert fn is not None, "no SDMWriteRead node"
    return fn


def _fwd(d, C, sph, P, quant, no_grad=False):
    """(readings, bank after the forward, the op's autograd node or None) of sdm_write_read on private copies."""
    from sparse_delta_memory.memory_ops import sdm_write_read
    leaves = {n: d[n].detach().clone().requires_grad_(not no_grad) for n in OP_INPUTS}
    args = (leaves["memory"], d["k_idx"], leaves["k_val"], leaves["v"], leaves["beta"], leaves["g"], d["q_idx"],
            leaves["q_val"])
    kw = dict(chunk_size=C, num_memory_slots=sph, num_parallel=P, snapshot_quant=quant)
    if no_grad:
        with torch.no_grad():
            return sdm_write_read(*args, **kw), leaves["memory"].detach(), None
    out = sdm_write_read(*args, **kw)
    return out.detach(), leaves["memory"].detach(), _node(out)


def _valid_rows(d, C, P, T):
    """Row indices of the valid compact rows."""
    k = d["k_idx"].cpu()
    W = k.shape[-1]
    rows = []
    for ti in range(-(-T // C)):
        for p in range(P):
            c = ti * P + p
            groups = k[p, ti * C:min(T, ti * C + C)].unique().numel()
            rows.append(torch.arange(c * C * W, c * C * W + groups))
    return torch.cat(rows)


# fixtures whose quantized DW must be the one-slice-per-warp 16-warp walk on a full-size device (227 KiB opt-in)
PIN_NW16 = {(384, 256, torch.bfloat16), (960, 256, torch.float32)}


def _native_q(hd, dtype, C, quant):
    """The forward's choice for a quantized call (forward.phase1_fields: movers.dw_admits, then mover_fx's admission)."""
    from sparse_delta_memory import movers
    from sparse_delta_memory.cuda import mover_fx_cuda as fx
    bank = torch.empty(64, hd, device="cuda", dtype=dtype)
    return movers.dw_admits(bank, C) and fx.quant_supported(bank, C, QMODE[quant])


def _admit_quant(monkeypatch):
    """The forward-only producer tests: backward.supported admits fp8 / int4 (the consumer is another item)."""
    from sparse_delta_memory import backward as bw
    orig = bw.supported
    monkeypatch.setattr(bw, "supported", lambda m, quant, *a, **k: orig(m, "none", *a, **k))


def _special(d, hd):
    """Signed zeros, subnormals, +-inf, NaN, one large element in a single slice and a row of tiny values."""
    mem = d["memory"]
    for p in range(d["k_idx"].shape[0]):
        r = d["k_idx"][p, 0, :7].tolist()
        mem[r[0]] = -0.0
        mem[r[1]] = 1e-39
        mem[r[2], 0] = float("inf")
        mem[r[3], 1] = float("-inf")
        mem[r[4], 2] = float("nan")
        mem[r[5], hd - 3] = 300.0
        mem[r[6], :] = 1e-30
        mem[r[6], hd // 2 + 1] = -2.5


CASES = [   # (P, T, W, R, hd, sph, C, dtype, slots): walk / cluster
    (2, 200, 8, 8, 128, 512, 64, torch.bfloat16, "hot"),           # one slice, warp max
    (2, 200, 8, 8, 256, 512, 64, torch.bfloat16, "zipf"),          # fx_dw2, one slice pair (no cluster attribute)
    (2, 150, 8, 8, 240, 512, 32, torch.bfloat16, "hot"),           # fx_dw2, one pair, SW 120
    (2, 300, 16, 8, 64, 1024, 256, torch.bfloat16, "zipf"),        # one slice, SW 64, C 256 (8 warps on GB300)
    (1, 300, 16, 8, 384, 1024, 256, torch.bfloat16, "zipf"),       # fx_dw column slices, 16 warps (C 256), cluster 3
    (1, 300, 16, 8, 960, 1024, 256, torch.float32, "hot"),         # float32 column slices, 16 warps, cluster 8
    (2, 150, 8, 8, 384, 512, 32, torch.bfloat16, "hot"),           # fx_dw column slices, cluster 3
    (2, 200, 8, 8, 1024, 512, 64, torch.bfloat16, "uniform"),      # fx_dw2 (README class), cluster 4
    (2, 300, 16, 8, 960, 1024, 128, torch.bfloat16, "trained"),    # fx_dw2 16 warps (sa4_h2 class), SW 120
    (1, 128, 8, 8, 2048, 256, 64, torch.bfloat16, "hot"),          # fx_dw2, cluster 8
    (1, 128, 8, 8, 1920, 256, 64, torch.bfloat16, "hot"),          # fx_dw, cluster 15 (non-portable)
    (2, 200, 8, 8, 128, 512, 64, torch.float32, "hot"),            # float32, one slice
    (2, 200, 8, 8, 960, 512, 64, torch.float32, "zipf"),           # float32 column slices, cluster 8
    # Modern wide rows: beyond the cluster form, the all-slices form (quant_choice -> 5 / 6; P1's producer)
    (1, 128, 8, 8, 2176, 256, 64, torch.bfloat16, "hot"),          # 17 slices (odd): fx_dw would need 17 cluster CTAs
    (1, 128, 8, 8, 4224, 256, 64, torch.bfloat16, "zipf"),         # 33 slices
    (1, 96, 8, 8, 8192, 256, 32, torch.bfloat16, "hot"),           # 64 slices: 32 slice pairs
    (1, 128, 8, 8, 2304, 256, 256, torch.bfloat16, "zipf"),        # C 256: no two-slice room, 18 > 16 slices
    (1, 128, 8, 8, 2176, 256, 64, torch.float32, "hot"),           # float32, 17 slices
]
# modern side: these take the all-slices form (beyond any cluster); the others take the form quant_choice picks (the
# cluster form wherever its capacity admits it: every one of them on GB300)
WIDE = {(2176, 64, torch.bfloat16), (4224, 64, torch.bfloat16), (8192, 32, torch.bfloat16), (2304, 256, torch.bfloat16),
        (2176, 64, torch.float32)}


@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("P,T,W,R,hd,sph,C,dtype,slots", CASES,
                         ids=[f"P{c[0]}_T{c[1]}_hd{c[4]}_C{c[6]}_{str(c[7])[6:]}_{c[8]}" for c in CASES])
def test_quant_rows_match_reference(P, T, W, R, hd, sph, C, dtype, slots, quant, monkeypatch, ieee_fp32):
    from sparse_delta_memory.cuda import mover_da2_cuda as da2, mover_fx_cuda as fx
    _admit_quant(monkeypatch)
    if not _native_q(hd, dtype, C, quant):
        pytest.skip(f"hd {hd}: not native on this device (cluster room / shared memory)")
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=7, slots=slots)
    _special(d, hd)
    r0, b0, n0 = _fwd(d, C, sph, P, "none")
    raw = n0.saved_tensors[11]
    assert n0.quant_layout == 2, "the quant-none forward did not write compact rows"
    first = raw.view(-1, hd)[:C * W].float()                  # chunk 0's rows hold the special pre-write rows
    assert torch.isnan(first).any() and torch.isinf(first).any() and (first.abs().amax(1) == 300.0).any()
    assert ((first == 0) & torch.signbit(first)).all(dim=1).any(), "no -0.0 row among the compact rows"
    rn, bn, _ = _fwd(d, C, sph, P, "none", no_grad=True)
    q0 = fx.STATS.get("fx_dw_q", 0)
    geoms = []
    make0 = fx.make_dw_fx

    def make_rec(*a, **k):                                    # record the quantized DW's geometry
        launch = make0(*a, **k)
        if k.get("quant"):
            geoms.append(dict(launch.geom))
        return launch
    monkeypatch.setattr(fx, "make_dw_fx", make_rec)
    assert not hasattr(da2._module(), "da2_quant"), "2.0.0 removed the per-entry quantizer (A4)"
    rq, bq, nq = _fwd(d, C, sph, P, quant)
    rq2, bq2, nq2 = _fwd(d, C, sph, P, quant)                 # repeat
    monkeypatch.undo()
    assert fx.STATS.get("fx_dw_q", 0) >= q0 + 2 and nq.quant_layout == 2, "the quantized DW did not run"
    if (hd, C, dtype) in PIN_NW16 and torch.cuda.get_device_properties(0).shared_memory_per_block_optin >= 227 * 1024:
        assert geoms and all(g["nw"] == 16 and not g["w2"] for g in geoms), geoms
    if not fx._portable(torch.device("cuda")) and -(-hd // 128) > 1:   # modern: the form quant_choice picks
        form = fx.quant_choice(torch.empty(64, hd, device="cuda", dtype=dtype), C, QMODE[quant])
        assert (hd, C, dtype) not in WIDE or form >= 5, ("a wide head must take the all-slices form", form)
        nsl = -(-hd // 128)
        cl = nsl // 2 if (dtype == torch.bfloat16 and nsl % 2 == 0) else nsl
        if (hd, C, dtype) not in WIDE and cl <= 8 and \
                torch.cuda.get_device_properties(0).shared_memory_per_block_optin >= 227 * 1024:
            assert form == QMODE[quant], ("a portable-size cluster must keep the cluster form", form)
        assert geoms and all(g["all_slices"] == (form >= 5) for g in geoms), (form, geoms)
    for name, a, b in (("readings", r0, rq), ("bank", b0, bq), ("readings nograd", rn, rq),
                       ("bank nograd", bn, bq), ("readings repeat", rq, rq2), ("bank repeat", bq, bq2)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    pay, sc = nq.saved_tensors[11].view(torch.uint8), nq.saved_tensors[12]
    pay2, sc2 = nq2.saved_tensors[11].view(torch.uint8), nq2.saved_tensors[12]
    rows = _valid_rows(d, C, P, T).to(pay.device)
    want_p, want_s = ref_quant(raw.view(-1, hd)[rows], quant)
    got_p, got_s = pay[rows].cpu(), sc[rows].cpu()
    assert torch.equal(got_s.view(torch.int32), want_s.view(torch.int32)), "scales != reference"
    bad = (got_p != want_p).any(dim=1).nonzero().flatten()
    assert bad.numel() == 0, f"payload != reference in {bad.numel()} rows, first {rows[bad[:4]].tolist()}"
    assert torch.equal(pay[rows], pay2[rows]) and torch.equal(sc[rows].view(torch.int32), sc2[rows].view(torch.int32))


@pytest.mark.parametrize("quant", ["fp8", "int4"])
@pytest.mark.parametrize("hd,C,dtype", [(128, 64, torch.bfloat16), (1024, 64, torch.bfloat16),
                                        (384, 32, torch.bfloat16), (960, 64, torch.float32)])
def test_quant_rows_graph_replay_equals_eager(hd, C, dtype, quant, monkeypatch, ieee_fp32):
    """The DW writes the quantized rows in captured calls too."""
    from sparse_delta_memory import memory_ops as mo
    _admit_quant(monkeypatch)
    if not _native_q(hd, dtype, C, quant):
        pytest.skip(f"hd {hd}: not native on this device (cluster room / shared memory)")
    P, T, W, R, sph = 2, 200, 8, 8, 512
    d = make_op_inputs(P, T, W, R, hd, sph, dtype, seed=3, slots="hot")
    mem0 = d["memory"]
    mem = mem0.clone()
    ins = [d[n] for n in ("k_idx", "k_val", "v", "beta", "g", "q_idx", "q_val")]
    lay = mo._layout(mem, *ins, C, sph, P)

    def run():
        info = {}
        with mo._kernel_context():
            out, saved = mo._sdm_forward(mem, *ins, lay, quant, save=True, grad_bank_fp32=False, info=info)
        assert info["quant_layout"] == 2
        return out, saved[11], saved[12]

    mem.copy_(mem0)
    r_e, p_e, s_e = (t.clone() for t in run())
    b_e = mem.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mem.copy_(mem0)
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    mem.copy_(mem0)
    with torch.cuda.graph(graph):
        r_c, p_c, s_c = run()
    rows = _valid_rows(d, C, P, T).to(mem.device)
    for rep in range(2):                                       # two replays, each from the initial bank
        mem.copy_(mem0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(r_c.view(torch.uint8), r_e.view(torch.uint8)), rep
        assert torch.equal(mem.view(torch.uint8), b_e.view(torch.uint8)), rep
        assert torch.equal(p_c.view(torch.uint8)[rows], p_e.view(torch.uint8)[rows]), rep
        assert torch.equal(s_c[rows].view(torch.int32), s_e[rows].view(torch.int32)), rep


_PROOF = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
@QUANT_HELPERS@
// For row scale sc[i] (largest amax amax[i]): every bf16 x with |x| <= amax[i] or x NaN (every x for sc = inf), as x and
// -x: the fast division's bytes / nibbles == the EXACT (IEEE division) form's.
template <int Q>
__global__ void proof(const float* sc, const float* amax, int nsc, unsigned long long* bad, unsigned int* first,
                      unsigned long long* tested) {
    const int si = blockIdx.y;
    const unsigned xb = blockIdx.x * blockDim.x + threadIdx.x;   // grid 256 x 256: every bf16 pattern, full warps
    const float s = sc[si], x = __uint_as_float(xb << 16);
    const bool in = x != x || fabsf(x) <= amax[si] || s == __int_as_float(0x7f800000);
    const unsigned n = __popc(__ballot_sync(0xFFFFFFFFu, in));
    if ((threadIdx.x & 31) == 0) atomicAdd(tested, static_cast<unsigned long long>(n));
    if (!in) return;
    const float4 v = make_float4(x, -x, x, -x);
    uint32_t f = 0u, e = 0u;
    q_store4<Q, false>(reinterpret_cast<uint8_t*>(&f), 0, v, s, __frcp_rn(s));
    q_store4<Q, true>(reinterpret_cast<uint8_t*>(&e), 0, v, s, 0.f);
    if (f != e && atomicAdd(bad, 1ull) == 0ull) {
        first[0] = xb; first[1] = __float_as_uint(s); first[2] = f; first[3] = e;
    }
}
void run(torch::Tensor sc, torch::Tensor amax, torch::Tensor bad, torch::Tensor first, int64_t q) {
    const int nsc = static_cast<int>(sc.numel());
    const dim3 grid(256, nsc);
    auto* b = reinterpret_cast<unsigned long long*>(bad.data_ptr<int64_t>());   // [0] mismatches, [1] pairs tested
    auto* f = reinterpret_cast<unsigned int*>(first.data_ptr<int32_t>());
    if (q == 1) proof<1><<<grid, 256>>>(sc.data_ptr<float>(), amax.data_ptr<float>(), nsc, b, f, b + 1);
    else proof<2><<<grid, 256>>>(sc.data_ptr<float>(), amax.data_ptr<float>(), nsc, b, f, b + 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@pytest.mark.parametrize("quant", ["fp8", "int4"])
def test_quant_division_exhaustive(quant):
    """The bf16-bank division of QUANT_HELPERS."""
    from torch.utils.cpp_extension import load_inline
    from sparse_delta_memory.cuda.gx_gds_cuda import QUANT_HELPERS
    mod = load_inline("sdm_quant_div_proof", cpp_sources="void run(torch::Tensor sc, torch::Tensor amax, torch::Tensor "
                      "bad, torch::Tensor first, int64_t q);", cuda_sources=_PROOF.replace("@QUANT_HELPERS@", QUANT_HELPERS),
                      functions=["run"], extra_cuda_cflags=["-O3"])
    a = (torch.arange(0, 0x7F81, dtype=torch.int32) << 16).view(torch.float32)      # +0 .. +inf
    qinv = torch.tensor(1.0 / (448.0 if quant == "fp8" else 7.0), dtype=torch.float32)
    sc = torch.maximum(a * qinv, torch.tensor(1e-12, dtype=torch.float32)).to(torch.bfloat16).float()
    vals, counts = torch.unique_consecutive(sc, return_counts=True)                  # sc is monotone in amax
    amax = a[torch.cumsum(counts, 0) - 1]
    assert vals.numel() < 65536 and torch.isinf(vals[-1])
    bad = torch.zeros(2, dtype=torch.int64, device="cuda")
    first = torch.zeros(4, dtype=torch.int32, device="cuda")
    mod.run(vals.cuda(), amax.cuda(), bad, first, QMODE[quant])
    torch.cuda.synchronize()
    # the pairs the domain holds: per finite scale, x = +-0 .. +-A(sc) (2 * (bits(A) + 1) patterns) and the 254 NaN
    # patterns; every pattern for sc = inf
    abits = (amax[:-1].view(torch.int32) >> 16).long()
    want = int((2 * (abits + 1) + 254).sum()) + 65536
    nbad, tested = bad.tolist()
    assert tested == want and want > 2 ** 29, (tested, want)                 # ~8.2e8 (fp8) / 8.4e8 (int4) pairs
    assert nbad == 0, f"{nbad} mismatches; first (x bits, sc bits, fast, exact) = {first.tolist()}"
