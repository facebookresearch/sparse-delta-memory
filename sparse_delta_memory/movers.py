# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The decay-and-write (DW) and read-side backward (GDS) step launchers of the op.

``make_mover_dw`` and ``make_mover_gds`` run the CUDA movers (cuda/mover_fx_cuda.py and cuda/mover_da2_cuda.py for the
DW; cuda/gx_gds_cuda.py and cuda/mover_da2_cuda.py for the GDS) wherever ``dw_admits`` / ``gds_admits`` hold. There is
no other route (2.0.0): memory_ops.refuse_non_native refuses every call they do not take before any work, so a launcher
asked for one raises (NO_NATIVE_MOVER_MSG, an internal error).
Phase 1's records (krec, qrec / qrec8, tile bounds) are passed by the caller; without them the CUDA movers build their
own (the same values). Every mover gives each bank (or bank-gradient) row one writer per launch and sums a slot
group's entries in a fixed order, so its results are bitwise reproducible.
"""

import torch

from .cuda import mover_da2_cuda as da2
from .cuda._device import backend

STATS = {"da2_dw": 0, "shipped_dw": 0, "da2_gds": 0, "shipped_gds": 0, "p1_krec": 0, "p1_qrec": 0}


# Tail classes (shapes outside the 16-byte-slice predicate): native only when EVERY kernel's hook admits them.
_TAIL_HOOKS = (("mover_fx_cuda", "tail_supported"), ("gather_cuda", "tail_supported"),
               ("gather_ns_cuda", "tail_supported"), ("gx_pa2_cuda", "tail_supported"),
               ("gx_gds_cuda", "tail_supported"), ("pa_dc_cuda", "gwr_tail_supported"))


NO_NATIVE_MOVER_MSG = ("sparse_delta_memory: internal error: no native {kind} mover for bank {dtype} {shape}, chunk_size {C}, "
                       "snapshot_quant {quant} (2.0.0 removed the Triton movers; memory_ops.refuse_non_native should have "
                       "refused this call); please report it")


def tail_admits(bank, chunk_size, quant="none") -> bool:
    """The ONE tail predicate: every kernel of the call admits this tail class (the AND of _TAIL_HOOKS); forward and
    backward admissions read it."""
    import importlib
    return all(getattr(importlib.import_module(f".cuda.{mod}", __package__), fn)(bank, chunk_size, quant)
               for mod, fn in _TAIL_HOOKS)


def dw_admits(bank, chunk_size, quant="none") -> bool:
    """The CUDA DW takes calls on ``bank`` with chunks of ``chunk_size`` tokens: mover_da2_cuda's admission on the
    modern backend, mover_fx_cuda's on a100 / rtx; on every backend also tail_admits."""
    if bank.is_cuda and backend(bank.device) != "modern":
        from .cuda import mover_fx_cuda
        # fp8 / int4 rows above the old 16384 head bound: not admitted (QT class T0)
        fx_ok = mover_fx_cuda.admits(bank, chunk_size) and (
            quant == "none" or bank.shape[-1] <= mover_fx_cuda._QUANT_MAX_HEAD_DIM)
        return fx_ok or tail_admits(bank, chunk_size, quant)
    return (da2.supported(bank, chunk_size) and da2.fits(bank, chunk_size, "dw")) or tail_admits(bank, chunk_size, quant)


def gds_admits(bank, grad_dtype, chunk_size, quant="none") -> bool:
    """The CUDA GDS takes calls on ``bank`` with a ``grad_dtype`` gradient bank (the bank's, or float32 for a bf16 bank)
    and chunks of ``chunk_size`` tokens: gx_gds_cuda's GDS (modern: mover_da2_cuda's for the shapes it declines); tail_admits."""
    grad_ok = grad_dtype == bank.dtype or (bank.dtype == torch.bfloat16 and grad_dtype == torch.float32)
    if not grad_ok or not bank.is_cuda:
        return False
    if backend(bank.device) != "modern":
        from .cuda import gx_gds_cuda
        return gx_gds_cuda.admits(bank, grad_dtype, chunk_size, 1, quant) or tail_admits(bank, chunk_size, quant)
    return ((da2.supported(bank, chunk_size) and da2.fits(bank, chunk_size, "gds", grad_dtype))
            or tail_admits(bank, chunk_size, quant))


def read_records(p1):
    """Phase 1's read records for the GDS: the 16-byte qrec, or, when the forward asked Phase 1 for the compact form
    only (DET_QREC8: int32 [N, C * R, 2] = qrec's words 1-2, {absolute slot, scale bits}), qrec8, which
    gx_gds_cuda's GDS reads with spay_q (same bits); None when Phase 1 emitted neither. make_mover_gds builds 16-byte
    records itself whenever gx_gds_cuda's GDS declines a call that got qrec8 (only speed lost)."""
    q = p1.qrec
    if q is not None and q.numel() > 0:
        return q
    q8 = p1.qrec8
    return q8 if q8 is not None and q8.numel() > 0 else None


def make_mover_dw(bank, snapshot, delta_v, slot, cpe, k_val, rex, payload=None, scale=None, *, chunk_size,
                  num_parallel, valid_tokens, snapshot_quant="none", sorted_k=None, krec=None, tiles_k=None,
                  segb_k=None):
    """The DW step launcher (the CUDA DW; a call dw_admits declines raises NO_NATIVE_MOVER_MSG: the op refuses it up
    front). krec, tiles_k, segb_k: Phase 1's records and tile bounds for these entries (``phase1_forward(..., beta=...)``),
    or None."""
    # krec: Phase 1's write records, [N, C*W, 4] int32; when given they replace the mover's own record build.
    # Quantized snapshots (fp8 / int4) stay on the CUDA DW, which also quantizes them; the bank update is unchanged.
    # Tails run only on mover_fx_cuda's bank-direct DW (layouts 2 / 4); on layout 0 they raise below.
    if snapshot_quant == "bf16":   # bf16 rows: the forward passes "none" with its bf16 compact buffer (no other route)
        raise RuntimeError("make_mover_dw: snapshot_quant bf16 runs as the unquantized DW with bf16 compact rows")
    tail_off = (sorted_k is not None and not da2._shape_supported(bank, chunk_size)
                and getattr(snapshot, "_dc_snapc_layout", 0) not in (2, 4))
    if sorted_k is None or tail_off or not dw_admits(bank, chunk_size, snapshot_quant):
        # 2.0.0 (A4): phase2_kernels' Triton DW is gone; memory_ops.refuse_non_native refuses such calls up front.
        raise RuntimeError(NO_NATIVE_MOVER_MSG.format(kind="DW", shape=tuple(bank.shape), dtype=bank.dtype, C=chunk_size,
                                                      quant=snapshot_quant))
    STATS["da2_dw"] += 1
    n = sorted_k.shape[-1]
    sorted_k = sorted_k.reshape(-1, n)
    w = slot.shape[-1]
    steps = slot.numel() // (num_parallel * chunk_size * w)
    if krec is not None and krec.numel() == sorted_k.numel() * 4 and krec.is_contiguous() \
            and krec.data_ptr() % 16 == 0 and krec.dtype == torch.int32:   # read as uint4: 16-byte aligned
        STATS["p1_krec"] += 1
        krec = krec.reshape(-1, n, 4)
    else:
        krec = da2.build_krec(sorted_k, slot, k_val, rex, cpe)
    # (a misaligned snapshot only disables the compact-snapshot read inside make_dw: it is never copied)
    return _with_aligned(
        lambda bk, dv: da2.make_dw(bk, dv, sorted_k, krec, chunk_size=chunk_size, num_parallel=num_parallel,
                                   steps=steps, snapshot=snapshot, tiles_k=tiles_k, segb_k=segb_k,
                                   snapshot_quant=snapshot_quant, payload=payload, scale=scale,
                                   valid_tokens=valid_tokens),
        (bank, delta_v), outs={0})


class _StepRecords:
    """Record provider of gx_gds_cuda's GDS over px_qrec_cuda's output: the 8-byte read records of every chunk and
    one CUDA event per step (or None)."""

    def __init__(self, rec8, events):
        self.rec8, self.events = rec8, events

    def get(self, ti):
        return self.rec8, (self.events[ti] if self.events is not None else None)


def _px_qrec_provider(sorted_q, slot, q_val, log_q, num_parallel, steps):
    """px_qrec_cuda's per-step side-stream builder of the 8-byte read records: build_qrec8_steps(spay_q, q_idx, q_val,
    log_q, *, P, Nt) -> (int32 [Nt * P, C * R, 2], list of Nt events or None); GDS(ti) waits on events[ti]. Same bits
    as Phase 1's qrec words 1-2."""
    from .cuda import px_qrec_cuda
    rec8, events = px_qrec_cuda.build_qrec8_steps(sorted_q, slot, q_val, log_q, P=num_parallel, Nt=steps)
    STATS["px_qrec8"] = STATS.get("px_qrec8", 0) + 1
    return _StepRecords(rec8, events)


def make_mover_gds(bank, grad_bank, grad_readings, slot, q_val, log_q, d_q_w, *, chunk_size, num_parallel,
                   valid_tokens, sorted_q=None, qrec=None, tiles_q=None, snapshot_quant="none"):
    """The GDS step launcher: gx_gds_cuda's mover (one sorted walk per tile), which delegates to mover_da2_cuda's (the
    same bits) outside its scope; a call gds_admits declines raises NO_NATIVE_MOVER_MSG (the op refuses it up front). qrec, tiles_q: Phase 1's read
    records (``read_records``) and tile bounds for these entries, or None. snapshot_quant: the op's mode (the shared
    tail predicate admits tail classes per mode: fp8 / int4 tails are refused up front)."""
    if sorted_q is None or not gds_admits(bank, grad_bank.dtype, chunk_size, snapshot_quant):
        # 2.0.0 (A4): phase2_kernels' Triton GDS is gone; memory_ops.refuse_non_native refuses such calls up front.
        raise RuntimeError(NO_NATIVE_MOVER_MSG.format(kind="GDS", shape=tuple(bank.shape), dtype=bank.dtype, C=chunk_size,
                                                      quant=snapshot_quant))
    STATS["da2_gds"] += 1
    n = sorted_q.shape[-1]
    sorted_q = sorted_q.reshape(-1, n)
    r = slot.shape[-1]
    steps = slot.numel() // (num_parallel * chunk_size * r)
    from .cuda import gx_gds_cuda as gxm
    if gxm.supported(bank, grad_bank, chunk_size, r):
        # gx_gds_cuda's mover reads Phase 1's qrec (16 B) or qrec8 (8 B and spay_q) as given; with neither,
        # px_qrec_cuda's per-step side-stream builder (records of all steps and one event per step).
        if gxm.records_ok(qrec, sorted_q):
            STATS["gx_qrec_in"] = STATS.get("gx_qrec_in", 0) + 1
        else:
            qrec = _px_qrec_provider(sorted_q, slot, q_val, log_q, num_parallel, steps)
    elif qrec is not None and torch.is_tensor(qrec) and qrec.numel() == sorted_q.numel() * 4 and \
            qrec.is_contiguous() and qrec.data_ptr() % 16 == 0 and qrec.dtype == torch.int32:   # Phase 1's records
        STATS["p1_qrec"] += 1
        qrec = qrec.reshape(-1, n, 4)
    else:
        qrec = da2.build_qrec(sorted_q, slot, q_val, log_q)
    return _with_aligned(
        lambda bk, gb, gr_: gxm.make_gds(bk, gb, gr_, sorted_q, qrec, d_q_w, chunk_size=chunk_size,
                                         num_parallel=num_parallel, steps=steps, q_idx=slot,
                                         valid_tokens=valid_tokens, tiles_q=tiles_q),
        (bank, grad_bank, grad_readings), outs={1})


def _aligned16(*ts):
    return all(t is None or t.numel() == 0 or t.data_ptr() % 16 == 0 for t in ts)


def _with_aligned(build, tensors, outs):
    """Dispatch never depends on memory layout: when one of `tensors` is not 16-byte aligned (a
    view), the CUDA launcher runs on aligned scratch copies instead. Before every launch the scratch is refreshed from
    the caller's tensor (other kernels may change it between steps); after it, the tensors named in `outs` are copied
    back. build(*aligned) returns the launcher. Same arithmetic, so the same bits as with aligned inputs."""
    if _aligned16(*tensors):
        return build(*tensors)
    STATS["aligned_scratch"] = STATS.get("aligned_scratch", 0) + 1
    scratch = [t if (t is None or t.numel() == 0 or t.data_ptr() % 16 == 0) else t.clone(memory_format=torch.contiguous_format)
               for t in tensors]
    launch = build(*scratch)
    pairs = [(t, sc) for t, sc in zip(tensors, scratch) if sc is not t]
    back = [(t, sc) for i, (t, sc) in enumerate(zip(tensors, scratch)) if sc is not t and i in outs]

    def run(ti):
        for t, sc in pairs:
            sc.copy_(t)
        launch(ti)
        for t, sc in back:
            t.copy_(sc)
    for attr in ("reads_snapc", "writes_snapc", "geom"):
        if hasattr(launch, attr):
            setattr(run, attr, getattr(launch, attr))
    return run
