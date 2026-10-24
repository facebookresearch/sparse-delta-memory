# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The backward of the chunked write/read on the CUDA kernels (``sdm_backward``).

Per step in reverse order: the gather-weighted read of the gradient rows (dc_gwr), the d_u update, Phase A (the
write-side backward and bank restore) and the read-side backward (``movers.make_mover_gds``); then Phase B, the matmuls
through the solve and the Phase-1 backward. Calls it does not support (``supported``) are refused at the forward
(memory_ops.refuse_non_native, 2.0.0): there is no other backward. memory_ops' entry points are looked up there at call time."""

import torch

from . import memory_ops as mo, movers
from .cuda import pa_dc_cuda as dc, phase1_bwd_cuda as _p1bwd
from .cuda._device import cuda_backend, smem_cap

STATS = {"dc": 0, "shipped": 0}
# The last backward's step loop: "eager_cpp_step_loop" (one C++ call) or "python_step_loop" (diagnostics).
LAST_LOOP = {"name": None}
QUANT_CODES = {"fp8": 1, "int4": 2}   # gx_pa2_cuda's QUANT of each snapshot_quant mode


def pa_plan(C, bank_dtype, grad_dtype, device):
    """(configuration bits, depth, warps per CTA) of Phase A's kernel, or None when even one warp does not fit the
    device's opt-in shared memory. The footprint is the launcher's exact formula."""
    depth, nw = 8, 8
    optin = smem_cap(device)
    rb = 16 if bank_dtype == torch.float32 else 8
    gb = 16 if grad_dtype == torch.float32 else 8
    head = C * 512 + ((C + 3) // 4) * 16

    def need(w):
        return head + w * depth * 32 * (gb + rb)
    while nw > 1 and need(nw) > optin:
        nw //= 2                                  # results do not depend on the warps per CTA
    if need(nw) > optin:
        return None
    return 512, depth, nw


def compact_rows(lay) -> int:
    """Rows of the call's compact snapshot buffer: Nt * P * C * W (forward and backward compute the same)."""
    return int(lay.Nt) * int(lay.P) * int(lay.C) * int(lay.W)


def supported(memory, quant, C=None, grad_f32_possible=False, rows=None):
    """The CUDA backward runs for this op (evaluated by the forward too, before it writes a compact snapshot):
    dtype/layout/device checks plus a shared-memory plan for the worst grad dtype the backward may use. fp8 / int4 also
    need the compact snapshot's row count (`rows`, compact_rows(lay); any count >= 0: gx_pa2 reads 64-bit snapshot rows),
    so the forward writes quantized compact rows only where the backward's gx_pa2 reads them; unknown -> no."""
    if C is not None and memory.is_cuda:
        gdt = torch.float32 if (grad_f32_possible and memory.dtype == torch.bfloat16) else memory.dtype
        if pa_plan(C, memory.dtype, gdt, memory.device) is None:
            return False
    # Shape / dtype / arch only: no alignment or memory-layout condition (the forward already requires a
    # contiguous, vector-aligned bank; the backward's launchers raise rather than switch paths). head_dim % 4 != 0 (tails:
    # gx_pa2 / gx_gds element pieces) only where the shared tail predicate admits the call and the compact
    # snapshot's row count (`rows`) is known (any count: gx_pa2 reads 64-bit snapshot rows): Phase A has no dc_pa3
    # fallback for those heads, so a call gx_pa2 would refuse is refused up front (memory_ops.refuse_non_native; unknown
    # rows -> no).
    dev_ok = memory.dtype in (torch.bfloat16, torch.float32) and memory.is_cuda and cuda_backend(memory.device)
    ta = getattr(movers, "tail_admits", None)      # (a tree without the shared predicate: no tails)
    base = dev_ok and (memory.shape[-1] % 4 == 0 or (
        C is not None and rows is not None and 0 <= rows and ta is not None and ta(memory, C, quant)))
    if quant == "none" or not base:
        return base and quant == "none"
    # fp8 / int4: gx_pa2's quantized instantiations read the compact rows mover_fx_cuda's quantized DW writes;
    # the calls they do not take are refused up front (slice rows that are not 16-byte rows). One
    # predicate for both sides: the forward reads this one too (lean Phase-1 metadata, compact rows), so it also needs
    # the producer (mover_fx_cuda.quant_supported).
    from .cuda import gx_pa2_cuda, mover_fx_cuda
    producer = getattr(mover_fx_cuda, "quant_supported", None)
    if quant == "bf16":
        # bf16 compact rows of a float32 bank (memory_ops runs "bf16" as "none" on other banks): written by mover_fx_cuda's
        # bank-direct WR walk as bf16 rows (s16_supported; the unquantized forward), read by gx_pa2's bf16-snapshot rows
        # (regular module, or its tail module for slices that are not 16-byte bf16 rows: s16_admits) restoring the float32
        # bank with float32 grad rows (grad_bank_fp32 is a no-op on float32 banks). The same heads as 'none' (base above).
        s16 = getattr(gx_pa2_cuda, "s16_admits", None)
        prod = getattr(mover_fx_cuda, "s16_supported", None)
        return (C is not None and memory.dtype == torch.float32 and s16 is not None and prod is not None
                and rows is not None and 0 <= rows < 2 ** 31 and bool(s16(memory, C)) and bool(prod(memory, C)))
    gdt = torch.float32 if (grad_f32_possible and memory.dtype == torch.bfloat16) else memory.dtype
    code = QUANT_CODES.get(quant)
    return (C is not None and code is not None and producer is not None and rows is not None and 0 <= rows
            and gx_pa2_cuda.admits_rows(memory, gdt, C, code) and bool(producer(memory, C, code)))


def _supported(ctx, memory, C, saved=None):
    ok = supported(memory, ctx.snapshot_quant, C=C, grad_f32_possible=bool(getattr(ctx, "grad_bank_fp32", False)),
                   rows=compact_rows(ctx.lay))
    if ok and saved is not None and memory.shape[-1] % 4 and ctx.snapshot_quant in ("none", "bf16"):
        # head_dim % 4 != 0: only where gx_pa2 runs (dense compact snapshot, layout 2, 16-byte aligned)
        snap = saved[11]
        ok = (ctx.quant_layout == 2 and snap is not None
              and snap.numel() == compact_rows(ctx.lay) * memory.shape[-1] and snap.data_ptr() % 16 == 0)
    return ok


def admits_at_forward(memory, C, snapshot_quant, rows=None):
    """At forward time: the backward of this call runs the CUDA kernels whatever grad dtype it uses (float32 with
    grad_bank_fp32 on a bfloat16 bank): ``supported`` with the worst grad dtype (and the compact row count `rows`),
    and the CUDA GDS (movers.gds_admits) for every such grad dtype."""
    gdts = (memory.dtype, torch.float32) if memory.dtype == torch.bfloat16 else (memory.dtype,)
    return (supported(memory, snapshot_quant, C=C, grad_f32_possible=True, rows=rows)
            and all(movers.gds_admits(memory, g, C, snapshot_quant) for g in gdts))


def _fp32_matmul_ieee() -> bool:
    """A cheap prefilter of the run-time qualification (gx_gemm_cuda.matches_cublas): float32 matmuls in full IEEE
    float32 (cuBLAS's effective TF32 switch off and no reduced-precision request). Unreadable precision state ->
    False (keep cuBLAS)."""
    try:
        if torch._C._get_cublas_allow_tf32():
            return False
        new = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
        if new is not None and str(new) not in ("ieee", "none"):
            return False
        return torch.get_float32_matmul_precision() == "highest"
    except RuntimeError:
        return False


def _steps_cpp(*, Nt, T, C, P, SC, W, L, nw, depth, dbg, pipe, grad_memory, memory, snap_arg, snap_rows, d_u,
               d_u_rows, x_t, d_delta_v, delta_v, beta, kpay, rec, eff_out, g_out, bpre, pa_q, gdwc, k_rows, write_w,
               pd_out, geo, empty_i, gemm_step, gemm_pdl, pa_fn, gds, gds_pdl, gwr_pdl):
    """sdm_backward's step loop as one C++ call (bwd_steps_cuda): True when it ran; False, with nothing launched, when a
    per-step launcher has no plan (the Python loop runs). Same launches as the loop."""
    if Nt < 1:                                    # no step (T = 0): nothing to launch, as the loop
        STATS["steps_cpp"] = STATS.get("steps_cpp", 0) + 1
        LAST_LOOP["name"] = "eager_cpp_step_loop"
        return True
    from .cuda import gx_pa2_cuda as gxpa2
    gds_plan = getattr(gds, "plan", None)
    pa_dc3 = pa_fn is dc._module().dc_pa
    if gds_plan is None or not (gxpa2.is_launcher(pa_fn) or pa_dc3) or (
            gemm_step is not None and getattr(gemm_step, "plan", None) is None):
        STATS["steps_python"] = STATS.get("steps_python", 0) + 1
        LAST_LOOP["name"] = "python_step_loop"
        return False
    gd = gds_plan(Nt, pdl=gds_pdl)
    if gd is None:
        STATS["steps_python"] = STATS.get("steps_python", 0) + 1
        LAST_LOOP["name"] = "python_step_loop"
        return False
    from .cuda import bwd_steps_cuda
    gw = dc.gwr_plan(grad_memory, k_rows, write_w, d_delta_v, delta_v, pd_out, row_step=SC, out_row_step=SC,
                     tokens=SC, chunk=C, valid_tokens=T, ti_max=Nt - 1, vec_bytes=geo.vec_bytes, lanes=geo.lanes,
                     groups=geo.groups, tpc=geo.tpc, kpos=empty_i, gdwc=gdwc)
    if pa_dc3:                                    # dc_pa3 (Phase A for layouts gx_pa2 does not read): plain launches
        pa = dc.pa_step_plan(grad_memory, memory, snap_arg, snap_rows, d_u_rows, beta, kpay, rec, eff_out, g_out,
                             Nt - 1, P, C, W, L, nw, depth, dbg, pipe, bpre, pa_q)
    else:
        pa = gxpa2.step_plan(pa_fn, grad_memory, memory, snap_arg, snap_rows, d_u_rows, beta, kpay, rec, eff_out,
                             g_out, Nt - 1, P, C, W, L, nw, depth, dbg, pipe, bpre, pa_q)
    hd = d_u.shape[-1]
    if gemm_step is not None:
        gplan, gaddr, guse = gemm_step.plan(Nt, pdl=gemm_pdl)
        gm, views = (gplan, gaddr), None
    else:
        gm, guse = None, 0
        views = (d_u.view(Nt, P, C, hd), x_t.unflatten(0, (Nt, P)), d_delta_v.view(Nt, P, C, hd))
    bwd_steps_cuda.run(gw, gm, pa, gd[:2], steps=Nt, gwr_pdl=1 if gwr_pdl else 0, gemm_use=guse, gds_use=gd[2],
                       device=grad_memory.device, blas_views=views)
    if gemm_step is not None:
        gemm_step.count(Nt, guse)
    gds.done()
    STATS["steps_cpp"] = STATS.get("steps_cpp", 0) + 1
    LAST_LOOP["name"] = "eager_cpp_step_loop"
    return True


NO_NATIVE_BACKWARD_MSG = ("SDM backward: internal error: this call has no native backward, but its forward was not refused "
                          "(memory_ops.refuse_non_native); please report it with the call's shapes, dtypes and GPU")


def sdm_backward(ctx, saved, grad_readings):
    """SDMWriteRead's backward: the CUDA kernels (``_supported`` admits every call memory_ops.refuse_non_native lets
    through; 2.0.0 removed the reference backward): a call it declines raises NO_NATIVE_BACKWARD_MSG, an internal
    error, never a slow path."""
    memory = saved[0]
    if ctx.snapshot_quant == "bf16" and (not _supported(ctx, memory, ctx.lay.C, saved) or ctx.quant_layout != 2):
        raise RuntimeError("SDM backward: bf16 compact snapshot rows that the CUDA backward cannot read (no fallback)")
    if not _supported(ctx, memory, ctx.lay.C, saved) or (ctx.snapshot_quant != "none" and ctx.quant_layout != 2):
        # 2.0.0: no reference backward; memory_ops.refuse_non_native refused such calls at the forward (internal error)
        raise RuntimeError(NO_NATIVE_BACKWARD_MSG)
    STATS["dc"] += 1
    lay = ctx.lay
    P, T, C, Nt, W, R, hd = lay.P, lay.T, lay.C, lay.Nt, lay.W, lay.R, lay.hd
    N, SC = Nt * P, P * C
    NC = N * C
    (memory, k_idx, q_idx, k_val, q_val, g, beta, x, qx, delta_v, residual,
     snapshot, scale, *p1) = saved
    p1 = mo.Phase1(*p1)
    dtype, dev, f32 = memory.dtype, memory.device, torch.float32
    # A legal saved bank may be offset (e.g. +2 bytes with a scalar-gather head) or non-contiguous (a saved-tensor
    # unpack hook, e.g. activation offloading, may return an equal-valued strided view): run on an aligned contiguous
    # scratch copy and write the restored bank back with a raw (untracked) copy at the end. The restore lands in the
    # tensor the backward received (the unpacked one), as with the reference backward; if an unpack hook returns a
    # tensor that does not alias the caller's bank, the caller's bank is not restored (the same as there). Same bits.
    bank_user = None
    if not memory.is_contiguous() or memory.data_ptr() % 16:
        if dc._module().dc_overlaps(memory):       # before any work: a view whose elements share addresses
            raise RuntimeError("SDM backward: the saved bank is a self-overlapping view (some of its elements refer to a single "
                               "memory location); it cannot be restored in place, as with Tensor.copy_")
        bank_user = memory
        memory = torch.empty(memory.shape, device=dev, dtype=dtype)
        memory.copy_(bank_user)

    gr = mo.stage_time_major(grad_readings, P, T, C, f32)
    if not gr.is_contiguous() or gr.data_ptr() % 16:
        # A legal autograd layout (P == 1 makes staging a view of grad_readings, which may be offset or strided):
        # use aligned contiguous scratch, never raise. Same values, same bits.
        gr = gr.contiguous() if not gr.is_contiguous() else gr.clone()
    # Optional f32 grad bank (grad_bank_fp32, bf16 banks, default OFF): grad_memory and the gathered grad
    # rows stay f32 inside the backward; one rounding to the bank dtype at the end.
    grad_f32 = bool(getattr(ctx, "grad_bank_fp32", False)) and dtype == torch.bfloat16
    grad_memory = torch.zeros(memory.shape, device=dev, dtype=f32 if grad_f32 else dtype)
    d_u = torch.bmm(qx.transpose(1, 2), gr.view(N, C, hd))
    write_w = p1.rex.exp().mul_(k_val).view(NC, W)
    d_delta_v = torch.empty(NC, hd, device=dev, dtype=f32)
    grad_g_decay = torch.empty(N, C, W, device=dev, dtype=f32)
    d_k_pd = torch.empty_like(grad_g_decay)
    d_k_eff = torch.empty_like(grad_g_decay)
    d_q_w = torch.empty(N, C, R, device=dev, dtype=f32)

    geo = mo.pinned_gather_geometry(hd, grad_memory.dtype, W, SC)
    gslices = geo.slices
    nsl = dc.slices(hd)
    empty = torch.empty(0, device=dev, dtype=f32)
    empty_i = torch.empty(0, device=dev, dtype=torch.int32)
    part_pd = (torch.empty(gslices, NC * W, device=dev, dtype=f32) if gslices > 1 else empty)
    part_eff = (torch.empty(nsl, NC * W, device=dev, dtype=f32) if nsl > 1 else empty)
    part_g = (torch.empty(nsl, NC * W, device=dev, dtype=f32) if nsl > 1 else empty)
    pd_out = part_pd if gslices > 1 else d_k_pd
    eff_out = part_eff if nsl > 1 else d_k_eff
    g_out = part_g if nsl > 1 else grad_g_decay

    n = p1.spay_k.shape[-1]
    kpay = p1.spay_k.reshape(-1, n)
    rec = dc.build_records_pa(kpay, k_idx, k_val, p1.log_k, beta, p1.cpe)
    L = dc.tile_len(C, W, hd, P)
    quant = QUANT_CODES.get(ctx.snapshot_quant, 0)   # fp8 / int4 here: compact rows (payload, scale) of the fx DW
    snap_rows = SC * W if (snapshot.shape[0] == NC * W if quant else snapshot.numel() == NC * W * hd) else 0
    # The snapshot layout the forward wrote decides the reader.
    layout = ctx.quant_layout
    plan = pa_plan(C, dtype, grad_memory.dtype, dev)
    if plan is None:                               # admitted by _supported, so this cannot happen
        raise RuntimeError("SDM backward: no write-side mover plan fits the device's shared memory")
    vbits, depth, nw = plan
    dbg = vbits
    pipe = True                                    # (the kernels' pipe argument)
    mod = dc._module()
    d_u_rows = d_u.view(NC, hd)
    snap_arg = snapshot.view(torch.uint8) if quant else snapshot   # fp8 payloads are float8_e4m3fn tensors: bytes
    bpre = torch.empty(0, device=dev, dtype=torch.int32)
    if layout and snap_rows:
        dbg |= 256                                     # the forward's gather wrote the compact snapshot
        if layout == 2:
            dbg |= 1024                                # dense rows (valid-group ordinal)
    gdwc = torch.empty(0, device=dev, dtype=grad_memory.dtype)   # (the kernels' dense grad-row argument: unused)
    if dbg & 1024:
        bpre = p1.dc_bpre.reshape(N, -1)

    gds = mo.make_mover_gds(memory, grad_memory, gr, q_idx, q_val, p1.log_q, d_q_w, chunk_size=C, num_parallel=P,
                            valid_tokens=T, sorted_q=p1.spay_q, qrec=movers.read_records(p1),
                            tiles_q=mo._nonempty(p1.tiles_q), snapshot_quant=ctx.snapshot_quant)
    k_rows = k_idx.view(NC, W)
    x_t = x.transpose(1, 2)
    # Phase A runs on gx_pa2_cuda where it applies (the same bits as pa_dc_cuda's kernel, fewer instructions per
    # group), chosen once per backward from shapes and dtypes.
    # The per-step update d_u(ti) += x(ti)^T d_dv(ti) runs on gx_gemm_cuda: one FP32 kernel with a fixed per-element
    # order (fmaf over j ascending from +0, then one rounded add of d_u), so the step's Phase A can follow it with
    # programmatic dependent launch. Used only where it reproduces this call's cuBLAS bits: the prefilters (IEEE fp32
    # matmuls, P >= 2, C >= 2) and the run-time qualification matches_cublas() (this kernel against the torch.baddbmm
    # call on random data, once per configuration and settings; cuBLAS during a capture of an unqualified
    # configuration). Either way: cuBLAS's bits.
    gemm_step = None
    if P >= 2 and C >= 2 and _fp32_matmul_ieee():
        from .cuda import gx_gemm_cuda as gxgemm
        if gxgemm.supported(d_u, x, d_delta_v, C) and gxgemm.matches_cublas(d_u, x, d_delta_v, P):
            gemm_step = gxgemm.make_step(d_u, x, d_delta_v, P)
            STATS["gx_gemm"] = STATS.get("gx_gemm", 0) + 1
    # With programmatic dependent launch, the GEMM launches behind the GWR (it reads x and its d_u values before its
    # wait) and Phase A behind the GEMM (its setup and first loads overlap the GEMM). The GEMM's CTAs are capped at one
    # per SM (shared-memory padding), so an early launch spreads them over the SMs instead of packing them onto the few
    # the GWR's last wave frees.
    gemm_pdl = gemm_step is not None and bool(gemm_step.pdl_capable)
    pa_fn = mod.dc_pa
    from .cuda import gx_pa2_cuda as gxpa2
    pa_q = scale if quant else gdwc                # the quantized rows' scales, in dc_pa's gdwc slot
    if gxpa2.supported(grad_memory, memory, snap_arg, C, dbg, depth, nw, False, scales=scale if quant else None):
        pa_fn = gxpa2.launcher(pdl=gemm_pdl, bank=memory, grad=grad_memory, snap=snap_arg)   # once per backward
        # its warps per CTA and ring depth (outputs depend on neither), for these rows (quantized rows: their footprint)
        nw, depth = gxpa2.plan(grad_memory, memory, C, gxpa2.quant_of(memory, snap_arg))
        # one wave on small grids (outputs do not depend on L)
        L = gxpa2.tile_len(C, W, hd, P, L, device=memory.device)
        STATS["gx_pa2"] = STATS.get("gx_pa2", 0) + 1
    elif quant:   # the forward wrote these rows only where backward.supported admits gx_pa2's quantized instantiations
        raise RuntimeError("SDM backward: quantized compact snapshot rows that gx_pa2 cannot read")
    elif ctx.snapshot_quant == "bf16":   # bf16 rows: the forward wrote them only where gx_pa2 reads them (dc_pa3 cannot)
        raise RuntimeError("SDM backward: bf16 compact snapshot rows that gx_pa2 cannot read")
    elif hd % 4:  # _supported admitted it for gx_pa2 only (dc_pa3 takes hd % 4 == 0); nothing written yet
        raise RuntimeError("SDM backward: head_dim % 4 != 0 needs gx_pa2's Phase A, which declined this call")
    elif snap_arg.numel() // hd >= dc._WIDE_SNAP_ROWS:   # dc_pa3: its 64-bit snapshot rows (host-only dbg bit; dc_pa
        dbg |= dc.DBG_WIDE                                 # takes them by itself from 2**31 rows; tests force them)
    gds_pdl = bool(getattr(gds, "pdl_capable", False))
    # Step ti's GWR (ti < Nt - 1) launches right after step ti + 1's GDS with programmatic dependent launch (sm_90+).
    from .cuda import mover_fx_cuda as _fxm_b   # whether dc_gwr carries the launch hooks
    gwr_pdl = _fxm_b.gwr_pdl_ok(grad_memory.device)
    # The step loop below runs as ONE C++ call (bwd_steps_cuda) whenever every per-step launcher has a plan.
    cpp = _steps_cpp(Nt=Nt, T=T, C=C, P=P, SC=SC, W=W, L=L, nw=nw, depth=depth, dbg=dbg, pipe=pipe,
                     grad_memory=grad_memory, memory=memory, snap_arg=snap_arg, snap_rows=snap_rows, d_u=d_u,
                     d_u_rows=d_u_rows, x_t=x_t, d_delta_v=d_delta_v, delta_v=delta_v, beta=beta, kpay=kpay, rec=rec,
                     eff_out=eff_out, g_out=g_out, bpre=bpre, pa_q=pa_q, gdwc=gdwc, k_rows=k_rows, write_w=write_w,
                     pd_out=pd_out, geo=geo, empty_i=empty_i, gemm_step=gemm_step, gemm_pdl=gemm_pdl, pa_fn=pa_fn,
                     gds=gds, gds_pdl=gds_pdl, gwr_pdl=gwr_pdl)
    steps = () if cpp else zip(d_u.view(Nt, P, C, hd).unbind(0), x_t.unflatten(0, (Nt, P)).unbind(0),
                               d_delta_v.view(Nt, P, C, hd).unbind(0))
    for ti, (d_u_ti, x_t_ti, d_dv_ti) in reversed(list(enumerate(steps))):
        valid = min(C, T - ti * C)
        mod.dc_gwr(grad_memory, k_rows, write_w, d_delta_v, delta_v, pd_out, ti * SC, ti * SC,
                   SC, C, valid, geo.vec_bytes, geo.lanes, geo.groups, geo.tpc,
                   empty_i, gdwc,
                   1 if (ti < Nt - 1 and gwr_pdl) else 0)
        if gemm_step is not None:
            gemm_step(ti, pdl=gemm_pdl)
        else:
            torch.baddbmm(d_u_ti, x_t_ti, d_dv_ti, out=d_u_ti)
        pa_fn(grad_memory, memory, snap_arg, snap_rows, d_u_rows, beta, kpay, rec, eff_out,
              g_out, ti, P, C, W, L, nw, depth, dbg, pipe, bpre, pa_q)
        # The GDS launches with programmatic dependent launch after Phase A (which triggers at its start, or after its
        # own wait when it follows the GEMM); the kernel right before writes none of the GDS's pre-wait inputs (tile
        # bounds, records, grad_readings rows).
        if gds_pdl:
            gds(ti, pdl=True)
        else:
            gds(ti)
    if nsl > 1 or gslices > 1:
        mod.dc_pa_reduce(part_eff, part_g, part_pd, beta, p1.cpe, d_k_eff, grad_g_decay, d_k_pd,
                         P, C, W, T)

    d_delta_v = d_delta_v.view(N, C, hd)
    torch.baddbmm(d_delta_v, p1.qk.transpose(1, 2), gr.view(N, C, hd), out=d_delta_v)
    grad_v = torch.empty(N, C, hd, device=dev, dtype=dtype)
    grad_beta = torch.empty(NC, device=dev, dtype=f32)
    u, grad_beta_u = mo._pre_bmm(residual.view(N, C, hd), beta, d_u, grad_v)
    d_x = torch.bmm(d_delta_v, u.transpose(1, 2))
    d_m = torch.baddbmm(d_x, torch.bmm(x_t, d_x), x_t, beta=0.0, alpha=-1.0, out=d_x)
    d_a = mo._post_bmm(d_m, p1.a, beta, grad_beta_u, grad_beta)
    d_qk = torch.bmm(gr.view(N, C, hd), delta_v.view(N, C, hd).transpose(1, 2))
    # The Phase-1 backward, with its faster variant at some chunk sizes (the same bits).
    grad_k_val, grad_q_val, grad_g = _p1bwd.phase1_backward_det(
        p1.payload, p1.log_k, p1.log_q, p1.rex, k_val, q_val, g, d_a, d_qk, d_k_eff,
        d_k_pd, d_q_w)
    grad_g += grad_g_decay
    if lay.G == 1:
        grad_g = grad_g.sum(-1)

    grads = [mo.unstage_time_major(t, P, T, C, dt).view(shape) for t, (shape, dt) in
             zip((grad_k_val, grad_v, grad_beta, grad_g, grad_q_val), ctx.inputs)]
    grad_k_val, grad_v, grad_beta, grad_g, grad_q_val = grads
    if grad_f32:
        grad_memory = grad_memory.to(dtype)
    if bank_user is not None:
        # The restored bank into the caller's (offset) tensor through a raw copy: like the reference backward's raw
        # restore, it does not bump the saved tensor's autograd version (a retained graph can run backward again).
        dc._module().dc_copy_raw(bank_user, memory)
    return (grad_memory, None, grad_k_val, grad_v, grad_beta, grad_g, None, grad_q_val,
            None, None, None, None, None)
