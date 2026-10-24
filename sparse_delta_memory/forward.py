# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The forward of the chunked write/read over a range of steps (``forward_steps``).

Phase 1 with the per-call field set (``phase1_fields``) and the solve, then per step the gather, the dual matmul and the
decay-and-write (``movers.make_mover_dw``), with a compact-snapshot handshake between the gather and the DW.
memory_ops' entry points (phase1_forward, token_gather, make_dual_matmul, make_mover_dw, ...) are looked up there at
call time."""

import threading

import torch

from . import backward, memory_ops as mo, movers
from .cuda import _devmemo, gather_ns_cuda
from .cuda import phase1_cuda as ph

STATS = {}

def _dual_trig_wanted(fx_geom, num_parallel, head_dim, chunk_size, pdl, device_index):
    """Whether to launch the dual matmul with its early trigger (TRIG): only when the DW follows it with programmatic
    dependent launch (pdl), the chunk has at most 64 tokens and the DW's grid fits in one wave of resident CTAs
    (num_parallel * column slices * K CTAs <= 2 CTAs of 8 warps per SM of the call's device, half that for 16-warp
    CTAs). The DW's setup then overlaps the dual; with longer chunks or a grid of several waves, starting the DW early
    costs more than it saves. Speed only: the kernel body, hence the bits, are the same either way."""
    if not pdl or fx_geom is None:
        return False
    if chunk_size > 64:
        return False
    nw, K = int(fx_geom.get("nw", 8)), int(fx_geom.get("K", 0))
    nsl = -(-head_dim // 128)
    ctas = num_parallel * (nsl // 2 if fx_geom.get("w2") else nsl) * K   # the two-slice walk: one CTA per slice pair
    wave = 2 * mo._num_sms(device_index)   # resident 8-warp DW CTAs per wave (speed only)
    return 0 < ctas <= (wave if nw <= 8 else wave // 2)


# Phase-1 fields (phase1_forward(det_fields=...)) per call. Lean sets only when every consumer that can run takes the
# CUDA kernels: the DW (movers.dw_admits); for a saving forward also the backward (backward.admits_at_forward: its own
# admission with the worst grad dtype, and the CUDA GDS for every grad dtype it may use). No-grad: SPAY_K | KREC | TILES
# (296); saving adds SPAY_Q | QREC | DC_GIDX (952), with the 8-byte read records instead of the 16-byte ones (QREC8)
# when gx_gds_cuda's GDS mover, which reads them, will take the backward's GDS (_qrec8_ok). Otherwise the full set (the
# shipped fields, the mover records and DC_GIDX: 1023): since 2.0.0 only a saving call whose backward admission at
# forward time (every grad dtype it may use) declines while memory_ops.refuse_non_native admitted it with its actual grad
# dtype; every other call this function would give the full set is refused up front.
_NOGRAD_FIELDS = ph.DET_SPAY_K | ph.DET_KREC | ph.DET_TILES
_SAVING_FIELDS = _NOGRAD_FIELDS | ph.DET_SPAY_Q | ph.DET_QREC | ph.DET_DC_GIDX
_SAVING_FIELDS_QREC8 = (_SAVING_FIELDS & ~ph.DET_QREC) | ph.DET_QREC8
_FULL_FIELDS = (ph.DET_SHIPPED | ph.DET_KREC | ph.DET_KSPA | ph.DET_QREC | ph.DET_TILES | ph.DET_DC_GIDX)


def _qrec8_ok(memory, C, R, grad_bank_fp32):
    """gx_gds_cuda's GDS mover will take this saving call's GDS (the CUDA GDS admission already passed):
    gx_gds_cuda.admits for the call's bank, grad dtype (float32 with grad_bank_fp32 on a bfloat16 bank), C and R.
    grad_bank_fp32 None (grad dtype unknown) -> False."""
    if grad_bank_fp32 is None:
        return False
    from .cuda import gx_gds_cuda
    return bool(gx_gds_cuda.admits(memory, torch.float32 if grad_bank_fp32 else memory.dtype, C, R))


def _fxq_rows(memory, C, quant):
    """fp8 / int4: mover_fx_cuda's quantized DW writes this call's compact rows (eager and captured calls); bf16 (float32
    banks): its bank-direct WR walk writes them as bf16 rows (mover_fx_cuda.s16_supported)."""
    from .cuda import mover_fx_cuda
    if quant == "bf16":
        return mover_fx_cuda.s16_supported(memory, C)
    return mover_fx_cuda.quant_supported(memory, C, {"fp8": 1, "int4": 2}[quant])


# snapshot_quant "bf16" on a float32 bank runs on every CUDA backend (an RTX GPU gets the image
# guard's one-time untested-hardware warning); a bfloat16 bank runs "bf16" as "none" (memory_ops.effective_snapshot_quant).
def bf16_backend_ok(device) -> bool:
    """snapshot_quant "bf16" on a float32 bank may run on this device's class: every CUDA backend (read through _devmemo)."""
    try:
        return bool(_devmemo.backend(device))
    except RuntimeError:          # a device outside the CUDA backends
        return False


def bf16_route(memory, C, rows) -> bool:
    """snapshot_quant bf16 (float32 banks) has its native CUDA route for this call: the unquantized DW on the CUDA movers
    (dw_admits as "none"), mover_fx_cuda's bank-direct WR walk writing the bf16 compact rows (s16_supported) and the CUDA
    backward reading them (backward.supported "bf16": gx_pa2's bf16-snapshot rows, the GDS / GWR of "none"). Shapes,
    dtypes and the device; rows = backward.compact_rows(lay). Anything else is refused (no fallback route)."""
    return bool(memory.is_cuda and memory.dim() == 2 and memory.dtype == torch.float32
                and bf16_backend_ok(memory.device)
                and movers.dw_admits(memory, C, "none")
                and backward.supported(memory, "bf16", C=C, grad_f32_possible=True, rows=rows)
                and _fxq_rows(memory, C, "bf16"))


# A bf16 saving forward's admission, memoized per signature (phase1_fields' saving checks and
# forward_steps' compact-snapshot layout evaluated the bf16 rows' producer / reader predicates, mover_fx_cuda.s16_supported
# and gx_pa2_cuda.s16_admits, on every call). Admitted signatures only: a declined call evaluates them as before (and
# memory_ops.refuse_non_native refuses such a call up front).
_BF16_SAVING = {}
_BF16_SAVING_MAX = 4096
_BF16_SAVING_LOCK = threading.Lock()


def bf16_saving_admitted(memory, C, rows) -> bool:
    """A bf16 saving forward (snapshot_quant "bf16", float32 bank; rows = backward.compact_rows(lay)) is admitted: the
    unquantized CUDA DW (movers.dw_admits "none"), the CUDA backward reading the bf16 compact rows with its GDS at every
    grad dtype (backward.admits_at_forward "bf16"), mover_fx_cuda's WR walk writing them (_fxq_rows "bf16") and the
    compact layout's condition (backward.supported "bf16", worst grad dtype): phase1_fields' saving checks and
    forward_steps' layout 2. Memoized per signature, keyed by the device ordinal, dtype, shape, C, rows,
    memory_ops.route_token() (the forced test row, the routing predicates as bound) and
    backward.admits_at_forward as bound. ONLY admitted signatures are stored (at most _BF16_SAVING_MAX, under
    _BF16_SAVING_LOCK): a warm call evaluates no predicate; a declined one evaluates them every time."""
    if not (memory.is_cuda and memory.dim() == 2 and memory.dtype == torch.float32 and rows is not None):
        return False
    key = (memory.device.index, memory.dtype, tuple(memory.shape), int(C), int(rows), mo.route_token(),
           backward.admits_at_forward)
    if key in _BF16_SAVING:
        return True
    ok = bool(movers.dw_admits(memory, C, "none") and backward.admits_at_forward(memory, C, "bf16", rows)
              and _fxq_rows(memory, C, "bf16")
              and backward.supported(memory, "bf16", C=C, grad_f32_possible=True, rows=rows))
    if ok and len(_BF16_SAVING) < _BF16_SAVING_MAX:
        with _BF16_SAVING_LOCK:
            if len(_BF16_SAVING) < _BF16_SAVING_MAX:
                _BF16_SAVING[key] = True
    return ok


def phase1_fields(memory, C, snapshot_quant, save, R, grad_bank_fp32=None, rows=None):
    """The det_fields of this call's Phase 1 (see above). rows: the compact snapshot's row count
    (backward.compact_rows(lay)), which fp8 / int4 need for the backward's admission. A bf16 saving call takes its
    admission from bf16_saving_admitted (memoized per signature); a declined one, and every other call, as below."""
    dq = "none" if snapshot_quant == "bf16" else snapshot_quant   # bf16 rows: the unquantized DW (bf16 stores)
    if not (save and snapshot_quant == "bf16" and bf16_saving_admitted(memory, C, rows)):
        if not (memory.is_cuda and memory.dim() == 2) or not movers.dw_admits(memory, C, dq):
            return _FULL_FIELDS
        if not save:
            return _NOGRAD_FIELDS
        if not backward.admits_at_forward(memory, C, snapshot_quant, rows):
            return _FULL_FIELDS
        if snapshot_quant != "none" and not _fxq_rows(memory, C, snapshot_quant):
            return _FULL_FIELDS
    if _qrec8_ok(memory, C, R, grad_bank_fp32):
        STATS["saving_qrec8"] = STATS.get("saving_qrec8", 0) + 1
        return _SAVING_FIELDS_QREC8
    STATS["saving_qrec16"] = STATS.get("saving_qrec16", 0) + 1
    return _SAVING_FIELDS


def _det_dual(x, qx, v, beta, retrieved, inter, delta_v, readings, residual, num_parallel, valid_tokens, trig=False):
    """The dual matmul's step launcher. memory_ops.make_dual_matmul runs first (its checks, and its launcher kept as the
    fallback); then the same Triton kernel for the same arguments, grid and launch options is compiled or fetched once
    (JITFunction.warmup with exactly the launcher's arguments: ti and valid_tokens are do_not_specialize, so one
    compiled kernel serves every step), and each step calls that kernel's own launcher (CompiledKernel.run) with the
    arguments JITFunction.run would pass, on the current stream, skipping JITFunction.run's per-launch binding,
    specialization key and cache lookup. Same kernel, grid, arguments and stream: same launches, same bits.
    Registered launch hooks are honoured as JITFunction.run does (launch metadata and hook objects passed to the
    launcher); pre-run hooks, a stages-inspection hook, a Triton without these entry points or any error while binding
    keep the fallback launcher. The fused C++ step loop (no Python hooks) runs only with Triton's own, empty hook chains
    (hooks_idle). trig: the kernel's early-trigger constexpr (TRIG; same arithmetic)."""
    launch = mo.make_dual_matmul(x, qx, v, beta, retrieved, inter, delta_v, readings, residual,
                                 num_parallel=num_parallel, valid_tokens=valid_tokens)
    try:
        import triton
        from triton import knobs
        from triton.runtime.driver import driver as _tdriver
        from . import phase2_kernels as _p2
        kern = _p2._sdm_dual_matmul
        # Bind only phase2_kernels' own kernel (a substituted memory_ops.make_dual_matmul keeps its own launcher).
        if mo.make_dual_matmul is not _p2.make_dual_matmul:
            return launch
        trig = bool(trig)

        def state():
            # Everything JITFunction.run reads besides the arguments that could change what or how it launches:
            # hooks, the debug / instrumentation options and the stages-inspection hook (they enter the cache key),
            # the current device (its stream). Checked again before EVERY bound launch.
            return (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                    tuple(getattr(kern, "pre_run_hooks", None) or ()), bool(kern.debug or knobs.runtime.debug),
                    knobs.compilation.instrumentation_mode, knobs.runtime.add_stages_inspection_hook,
                    _tdriver.active.get_current_device())
        st0 = state()
        # Pre-run hooks (JITFunction.run calls them before every launch) or a stages-inspection hook (it enters the
        # cache key): the fallback launcher. Launch hooks need no refusal: the bound launch passes JITFunction.run's
        # launch_metadata and hook objects (HookChain in Triton 3.7) to the launcher exactly as run does.
        if st0[2] or st0[5] is not None:
            return launch
        n, c, _ = x.shape
        hd = v.shape[-1]
        geom = _p2.dual_matmul_geom(c, v.dtype)                          # = make_dual_matmul's
        dv_rows = _p2._step_rows(delta_v, "delta_v", num_parallel * c, n // num_parallel, hd)
        grid0 = triton.cdiv(c, geom.block_m) * triton.cdiv(hd, geom.block_n) * num_parallel
        head = (x, qx, v, beta, retrieved, inter, delta_v, dv_rows, readings, residual)
        meta = dict(C=c, HD=hd, BLOCK_M=geom.block_m, BLOCK_N=geom.block_n, BLOCK_K=geom.block_k, DOT=geom.dot,
                    TRIG=1 if trig else 0)
        ck = kern.warmup(*head, 0, num_parallel, valid_tokens, grid=(grid0,), **meta, num_warps=4)
        if ck is None:
            return launch
        run = ck.run                                                       # loads the binary (CompiledKernel._init_handles)
        fn, pm = ck.function, ck.packed_metadata
        tail = (num_parallel, valid_tokens, c, hd, geom.block_m, geom.block_n, geom.block_k, geom.dot, meta["TRIG"])
        dev = _tdriver.active.get_current_device()
        get_stream = _tdriver.active.get_current_stream
        gvals = kern.used_global_vals
        missing = object()
    except Exception:  # noqa: BLE001 - any binding problem keeps the fallback launcher
        return launch

    def valid() -> bool:
        """The bound launch equals JITFunction.run's: same runtime state as at binding and the kernel's used globals
        unchanged (JITFunction.run raises on a changed global: the fallback launcher is then called and raises)."""
        if state() != st0:
            return False
        for (name, _), (val, gdict) in gvals.items():
            if gdict.get(name, missing) != val:
                return False
        return True

    grid = (grid0,)                       # the grid JITFunction.run receives from make_dual_matmul's kernel[grid]

    def bound(ti: int) -> None:
        if not valid():
            STATS["fx_dual_fallback"] = STATS.get("fx_dual_fallback", 0) + 1
            launch(ti)                    # the fallback launcher (JITFunction.run with its own checks and hooks)
            return
        # = JITFunction.run's launch statements: launch_metadata = kernel.launch_metadata(grid, stream, *bound_args);
        #   kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
        #              knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args)
        stream = get_stream(dev)
        args = (*head, ti, *tail)
        run(grid0, 1, 1, stream, fn, pm, ck.launch_metadata(grid, stream, *args),
            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

    hook_chain = getattr(knobs, "HookChain", None)

    def hooks_idle() -> bool:
        """No launch hook registered: each hook is None or EXACTLY Triton's own HookChain with no callbacks (a custom hook
        object may act in __call__ whatever its attributes). Only then does the C++ step loop, which calls no
        Python hook, launch exactly what JITFunction.run would. Checked at every fused call (hooks may be added later)."""
        for h in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
            if h is None:
                continue
            if hook_chain is None or type(h) is not hook_chain or h.calls:
                return False
        return True
    bound.shipped = launch
    bound.valid = valid
    bound.hooks_idle = hooks_idle
    # The same launch as a descriptor for the fused C++ step loop (mover_fx_cuda.fx_fwd_steps), when the kernel's launch
    # is one fx_fwd_steps replicates and the kernel was loaded on the bank's device; else None (Python loop).
    bound.tri_desc = None
    # Only without registered launch hooks.
    if dev == x.device.index and hooks_idle():
        try:
            from .cuda import mover_fx_cuda as _fxm
            bound.tri_desc = _fxm.tri_desc(ck, list(kern.arg_names), (*head, 0, *tail), grid0)
        except Exception:  # noqa: BLE001 - no descriptor: the Python step loop
            bound.tri_desc = None
    STATS["fx_dual_bound"] = STATS.get("fx_dual_bound", 0) + 1          # engagement counters (per forward call)
    if trig:
        STATS["fx_dual_trig"] = STATS.get("fx_dual_trig", 0) + 1        # bound with the early trigger
    if bound.tri_desc is not None:
        STATS["fx_dual_desc"] = STATS.get("fx_dual_desc", 0) + 1
    return bound


def forward_steps(memory, k_idx, k_val, v, beta, g, q_idx, q_val, lay, snapshot_quant, save, steps,
                  out=None, geom_tokens=0, grad_bank_fp32=None, info=None):
    """The forward over the steps ``range(*steps)``: stages their chunks, runs Phase 1 and the solve on them, then the
    steps, and writes their readings into ``out`` (or a new [P * T, hd] tensor). Returns the readings and the saved
    tensors (or None). The token gather's geometry is the pinned shape-only one (memory_ops.pinned_gather_geometry),
    the same as the backward's, so forward and backward agree and the bits do not depend on the SM count; it comes from
    the call's full token count (geom_tokens), so the groups and slabs of a no-grad call reduce in the same order as a
    saving call. grad_bank_fp32: the op's setting (saving calls; None: unknown). Only the engagement counters
    (STATS) and the dispatch record are module state; the computation keeps none (thread-safe, reentrant). info (a
    dict, saving calls): receives quant_layout, the compact-snapshot layout written (2: compact rows, row chunk * C * W +
    group ordinal, in the payload / scale slots for fp8 / int4; 0: per-entry rows)."""
    fields = phase1_fields(memory, lay.C, snapshot_quant, bool(save), lay.R, grad_bank_fp32,
                           rows=backward.compact_rows(lay))
    P, T, C, W, R, hd = lay.P, lay.T, lay.C, lay.W, lay.R, lay.hd
    Nt, SC = steps[1] - steps[0], P * C
    N = Nt * P
    NC = N * C
    tokens = T - steps[0] * C
    dtype, dev, f32 = memory.dtype, memory.device, torch.float32

    def stage(t, dt):
        return mo.stage_time_major(t, P, T, C, dt, steps)

    idx_dtype = torch.int32 if k_idx.dtype == q_idx.dtype == torch.int32 else torch.int64
    k_idx = stage(k_idx, idx_dtype).view(N, C, W)
    q_idx = stage(q_idx, idx_dtype).view(N, C, R)
    k_val = stage(k_val, dtype).view(N, C, W)
    q_val = stage(q_val, dtype).view(N, C, R)
    g = stage(g, f32).view(N, C, lay.G)
    beta = stage(beta, f32).view(NC)
    v = stage(v, dtype)

    p1 = mo.phase1_forward(k_idx, q_idx, k_val, q_val, g, mo._repeat_counter(dev),
                           num_parallel=P, stream_offset=lay.stream_offset,
                           num_keys=lay.num_keys, valid_last=min(C, tokens - (Nt - 1) * C),
                           need_read_groups=save, beta=beta.view(N, C), det_fields=fields)
    # Phase 1's write records go to this call's DW only: the saved Phase-1 tuple keeps an empty krec (the backward
    # never reads it; 16 B per write entry).
    krec = mo._nonempty(p1.krec)
    if krec is not None:
        p1 = p1._replace(krec=p1.krec.new_empty(0))   # fresh tensor: no view keeping the storage alive
    x, qx = mo.wy_solve(p1.a, beta, p1.qk)

    quant = snapshot_quant if save else "none"
    crow = backward.compact_rows(lay)   # the backward's admission sees the same row count (any count: 64-bit rows)
    # snapshot_quant bf16 (float32 banks): the unquantized forward ('none' DW: the bank-direct WR walk) with a bf16
    # every-step compact snapshot, which that walk writes as bf16 rows (RN) in eager and captured calls; the backward reads
    # them with gx_pa2's bf16-snapshot rows. No other route: memory_ops.refuse_non_native refused a call without it up
    # front (memoized: no per-call predicate here); the DW and the backward raise if their kernels decline these rows.
    bf16_rows = quant == "bf16"
    dw_quant = "none" if bf16_rows else quant     # the DW's (and the saved tensors') snapshot mode
    full_snapshot = save and dw_quant == "none"
    snapshot = torch.empty(NC if full_snapshot else SC, W, hd, device=dev,
                           dtype=torch.bfloat16 if bf16_rows else dtype)
    payload = scale = None
    if dw_quant != "none":
        width, qdtype = mo.quant_row(quant, hd)
        payload = torch.empty(NC * W, width, device=dev, dtype=qdtype)
        scale = torch.empty(NC * W, device=dev, dtype=f32)
    retrieved = torch.empty(SC, hd, device=dev, dtype=f32)
    inter = torch.empty_like(retrieved)
    delta_v = torch.empty(NC if save else SC, hd, device=dev, dtype=f32)
    residual = torch.empty(NC, hd, device=dev, dtype=f32) if save else None
    readings = torch.empty(NC, hd, device=dev, dtype=dtype)

    geo = mo.pinned_gather_geometry(hd, dtype, max(W, R), geom_tokens or SC)
    # Compact snapshot, writer-authoritative handshake: the forward declares the layout it will write on the snapshot
    # tensor (attribute _dc_snapc_layout: 2 = one row per valid write group, in group order; 4 = no compact rows, the DW
    # reads base rows from the bank) before the DW launcher is built; a compact-reading DW (the CUDA DW) reads compact
    # rows only for that declared layout and announces it (launcher attribute reads_snapc). The forward then writes
    # compact rows iff the DW announces exactly the declared layout; announced 0 -> the reference gather writes the full
    # per-entry stash. Any other combination raises. The written layout is returned in info["quant_layout"].
    # (bf16 rows: the memoized admission, which includes this backward.supported; a declined bf16 call evaluates it)
    want = 2 if full_snapshot and ((bf16_rows and bf16_saving_admitted(memory, C, crow))
                                   or backward.supported(memory, quant, C=C, grad_f32_possible=True, rows=crow)) else 0
    if (save and dw_quant in ("fp8", "int4") and backward.supported(memory, quant, C=C, grad_f32_possible=True, rows=crow)
            and _fxq_rows(memory, C, quant)):
        want = 2                                  # fp8 / int4 compact rows, quantized by the DW
    if want:
        snapshot._dc_snapc_layout = want
        # Ask the DW to write the compact rows itself; it announces it (writes_snapc) and the gather then writes none
        # (the same raw rows at the same places, see below). Same bits either way. Only in eager calls, not while a CUDA
        # graph is being captured: in graph replays it does not pay; in eager calls it lets the saving forward take the
        # gather plan and the fused step loop.
        # the capture state of the bank's device's current stream (the launches go there)
        dev_idx = memory.device.index
        if dev_idx is None or dev_idx == torch.cuda.current_device():
            capturing = torch.cuda.is_current_stream_capturing()
        else:
            with torch.cuda.device(dev_idx):
                capturing = torch.cuda.is_current_stream_capturing()
        if not capturing or quant != "none":     # quantized rows: written by the DW in captured calls too
            snapshot._dc_snapc_by_dw = True
    # The no-grad forward declares layout 4 (no compact rows): a DW that supports it reads each group's base row from
    # the bank itself, and the gather writes no stash.
    want_step = 0
    if not save:
        want_step = 4
        snapshot._dc_snapc_layout = want_step
    dw = mo.make_mover_dw(memory, snapshot, delta_v, k_idx, p1.cpe, k_val, p1.rex, payload, scale, chunk_size=C,
                          num_parallel=P, valid_tokens=tokens, snapshot_quant=dw_quant, sorted_k=p1.spay_k, krec=krec,
                          tiles_k=mo._nonempty(p1.tiles_k), segb_k=mo._nonempty(p1.segb_k))
    k_rows, k_eff = k_idx.view(NC, W), p1.k_eff.view(NC, W)
    q_rows, q_w = q_idx.view(NC, R), p1.q_w.view(NC, R)
    if (getattr(getattr(dw, "fx", None), "geom", None) or {}).get("w2"):
        STATS["fx_dw2"] = STATS.get("fx_dw2", 0) + 1     # the DW runs the two-slice walk (engagement counter)
    # The dual's step launcher is built after the DW's (no step is launched by either factory), so the early-trigger
    # rule can read the DW's grid.
    dual = _det_dual(x, qx, v, beta, retrieved, inter, delta_v, readings, residual, P, tokens,
                     trig=_dual_trig_wanted(getattr(getattr(dw, "fx", None), "geom", None), P, hd, C,
                                            gather_ns_cuda.pdl_ok(memory.device), memory.device.index))
    # Programmatic dependent launch: every step's DW launches right after that step's dual matmul (launched after
    # the step's gather), and every step's gather for ti >= 1 right after the previous step's DW, so what either reads
    # before its wait (records, tiles, compact rows; slots, weights, kpos: written before this loop or by the step's
    # gather) was written at least two launches earlier.
    pdl = gather_ns_cuda.pdl_ok(memory.device)
    if pdl and hasattr(dw, "set_pdl"):
        dw.set_pdl(True)
    mo._record(save=save, chunk=C, gather=geo._asdict(),
               dual_mm=mo.dual_matmul_geom(C, dtype)._asdict(), mover=mo.mover_geom(dtype, hd)._asdict())
    announced = int(getattr(dw, "reads_snapc", 0) or 0)
    if announced not in (0, want if save else want_step):
        raise RuntimeError(f"SDM forward: the write mover reads compact snapshot layout {announced} but the forward declared "
                           f"{want if save else want_step} (full snapshot={full_snapshot})")
    layout = announced
    # A DW that writes the compact rows itself (writes_snapc == 2, every step) replaces the gather's compact-row stores;
    # the rows, their places and the saved layout are unchanged.
    dw_writes = int(getattr(dw, "writes_snapc", 0) or 0)
    if bf16_rows and (dw_writes != 2 or layout != 2):   # bf16 rows: written by the DW only (the gather writes bank rows)
        raise RuntimeError(f"SDM forward: bf16 compact snapshot rows need the write mover to write them (layout "
                           f"{layout}, writes {dw_writes})")
    if dw_writes and (dw_writes != 2 or layout != 2 or not save):
        raise RuntimeError(f"SDM forward: the write mover writes compact snapshot layout {dw_writes} but the forward uses layout {layout} "
                           f"(save={save})")
    snapc_rows = layout == 2                      # compact rows exist (layout 4: none, no kpos)
    gather_snapc = snapc_rows and not dw_writes   # ... and the gather writes them
    kpos = None
    if gather_snapc:
        kpos = p1.dc_kpos.view(NC, W)
    snap2 = snapshot.view(-1, hd)                 # one view per call, not one per step
    # L2 hints in the gather (gather_ns_cuda.l2_hint: shapes only), decided once per call, only when the call's gather
    # runs gather_ns_cuda's module: layout 4 or DW-written compact rows (the no-stash gather), or compact rows written
    # by the gather (layout 0 runs the reference token_gather). Step tokens = geom_tokens or SC: the caller's tokens per
    # step (geom_tokens = the caller's P * C in the no-grad group and slab loop, whose sub-layouts cover fewer streams;
    # 0 elsewhere), so every sub-call of one op call takes the same decision.
    gl2 = bool((layout == 4 or dw_writes or gather_snapc)
               and gather_ns_cuda.l2_hint(R, geom_tokens or SC, memory.shape[0], hd * memory.element_size(),
                                          device=memory.device))
    if gl2:
        STATS["fx_gather_l2"] = STATS.get("fx_gather_l2", 0) + 1      # engagement (per call using the hinted gather)
    gplan = gstep = None                          # the gather's plan (the no-stash gather only)
    if layout == 4 or dw_writes:
        # The gather's checks and arguments once per call (a plan); each step passes row0 and valid only
        # (gather_ns_cuda.token_gather_dc's kernel and arguments: the same launches).
        snap0 = torch.empty(0, device=memory.device, dtype=memory.dtype)
        g4 = (geo.vec_bytes, geo.lanes, geo.groups, geo.tpc)
        if Nt > 0:
            gmod = gather_ns_cuda._module(gl2)
            gplan = gmod.token_gather_dc_plan(memory, k_rows, k_eff, snap0, retrieved, q_rows, q_w, inter,
                                              (Nt - 1) * SC, 0, SC, C, *g4)
            gstep = gmod.token_gather_dc_step
    # The whole step loop as one C++ call (the gather plan's step, the Triton dual through its compiled function, the DW
    # plan's step: the same kernels, arguments, order and stream as the loop below) when all three exist: the no-stash
    # gather plan, the dual's launch descriptor and the DW's plan launcher.
    fx_dw = getattr(dw, "fx", None)
    tdesc = getattr(dual, "tri_desc", None)
    if (Nt > 0 and gplan is not None and tdesc is not None and fx_dw is not None
            and getattr(fx_dw, "fx_plan", None) is not None and (layout == 4 or dw_writes) and dual.valid()
            and dual.hooks_idle()):
        from .cuda import mover_fx_cuda as _fxm
        STATS["fx_fused"] = STATS.get("fx_fused", 0) + 1      # fused step-loop calls (engagement counter)
        _fxm._module().fx_fwd_steps(gplan, gather_ns_cuda._module(gl2).token_gather_dc_launch_addr(), tdesc,
                                    fx_dw.fx_plan, 0, Nt, SC, C, tokens, 1 if pdl else 0, fx_dw.fx_pdl[0])
        Nt_loop = 0
    else:
        Nt_loop = Nt
    for ti in range(Nt_loop):
        if layout == 4 or dw_writes:
            # = gather_ns_cuda.token_gather_ns(memory, k_rows, k_eff, retrieved, geo, q_idx=q_rows, q_weights=q_w,
            #   q_out=inter, tokens=SC, chunk=C, valid=min(C, tokens - ti * C), row0=ti * SC) (snapc None, kpos None)
            gstep(gplan, ti * SC, min(C, tokens - ti * C), 1 if (pdl and ti > 0) else 0)
        elif gather_snapc:
            gather_ns_cuda.token_gather_ns(memory, k_rows, k_eff, retrieved, geo, q_idx=q_rows,
                                           q_weights=q_w, q_out=inter, tokens=SC, chunk=C,
                                           valid=min(C, tokens - ti * C), row0=ti * SC,
                                           snapc=snap2, kpos=kpos, crow=C * W, pdl=pdl and ti > 0, l2=gl2)
        else:
            mo.token_gather(memory, k_rows, k_eff, snapshot, retrieved, geo, q_idx=q_rows,
                            q_weights=q_w, q_out=inter, tokens=SC, chunk=C,
                            valid=min(C, tokens - ti * C), row0=ti * SC,
                            stash_row0=ti * SC if full_snapshot else 0)
        dual(ti)
        dw(ti)
    readings = mo.unstage_time_major(readings, P, T, C, dtype, steps, out)
    if not save:
        return readings, None
    if info is not None:
        info["quant_layout"] = layout
    kept = (payload, scale) if dw_quant != "none" else (snapshot, None)
    # dc_kpos is forward-only (the gather's map)
    return readings, (memory, k_idx, q_idx, k_val, q_val, g, beta, x, qx, delta_v, residual,
                      *kept, *p1._replace(k_eff=None, q_w=None, dc_kpos=None))
