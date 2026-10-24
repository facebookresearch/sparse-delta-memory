# Copyright (c) Meta Platforms, Inc. and affiliates.
"""gather_ns_cuda: the no-stash token gather of the deterministic forward ("ns": no stash), built from gather_cuda's
source with GATHER_NS defined.

Compared with token_gather it skips the per-entry stash (or, given ``snapc`` and ``kpos``, stores the raw write rows of
the valid group FIRSTs into the compact snapshot), loads an entry batch's kpos before its slots and weights, can run as
a programmatic dependent launch, and offers a plan / step entry pair for the step loop. Every accumulation (retrieved,
inter, d_delta_v) is token_gather's code, so the outputs are bit-identical to token_gather with the same geometry."""

import functools

import torch

from . import gather_cuda
from ._build import load_cuda_extension
from ._device import l2_bytes

# Variants of gather_cuda's source selected by GATHER_NS:
# - Entry loads: an entry batch's kpos load is issued before its slot / weight loads, so the first slot broadcast does
#   not wait behind it. The same loads of the same values.
# - Programmatic dependent launch: the deterministic forward launches each step's gather (steps >= 1) right after the
#   previous step's DW, which triggers its dependents once its own wait has returned. The gather's slot / weight / kpos
#   loads (Phase 1 outputs and staged indices, written before the step loop) run before griddepcontrol.wait; every bank
#   row is read after it, through the coherent L2 path (__ldcg: the DW has just written them), and no store precedes it.
#   Right after its wait the grid lets its own dependents launch (griddepcontrol.launch_dependents; a no-op unless the
#   next launch is programmatic). Arithmetic and order are unchanged.
# - Plan / step: token_gather_dc_plan runs token_gather_dc's checks and argument set-up once (a dry run whose launch()
#   records the kernel instantiation, Args, grid, block and shared memory instead of launching); token_gather_dc_step(
#   plan, row0, valid, pdl) then launches that kernel with only row0 / valid changed. Same kernel, same arguments.
# - L2 hints (GATHER_L2 also defined): the step's write-slot rows (operand 0) should stay in L2 for the DW that follows
#   (it reads each written group's base row bank[slot] right after this grid), so the read-slot rows (operand 1,
#   read-only in the step) are loaded with an L2 evict-first policy, and so are the compact snapshot rows stored (read
#   by the backward much later). Cache policy only: the same loads and stores of the same values. 16-byte vectors
#   (uint4) take the hinted instructions (sm_80+); other vector types and older architectures keep the plain ones. The
#   hinted build is used only when l2_hint (below) finds the step's reads sparse enough.


def _source(l2=False):
    """The no-stash gather source: gather_cuda's single source with GATHER_NS defined (the no-stash stores and checks,
    the kpos-first entry loads, the PDL hooks, the plan entry points) and, for l2, GATHER_L2 (the L2 hints)."""
    return "#define GATHER_NS 1\n" + ("#define GATHER_L2 1\n" if l2 else "") + gather_cuda._CUDA_SOURCE


def _cpp():
    return "#define GATHER_NS 1\n" + gather_cuda._CPP_SOURCE


L2_FETCH = [0]   # number of _module(l2=True) calls (the L2-hinted build returned)
# The read-row evict-first hint costs in-step L2 reuse when a step's reads overflow L2 and helps sparse, small steps.
# Shape rule: the step's read density R * (tokens per step) / (bank rows) is at most _L2_DMAX (num, den), and its
# read-entry bytes R * (tokens per step) * row_bytes are at most _l2_emax(device).
_L2_DMAX = (1, 10)
# The read-entry bound scales with the device's L2: 176 MiB on GB300, whose L2 is 135528448 bytes (exact there).
_L2_EMAX_GB300 = 176 << 20
_L2_GB300 = 135528448


def _l2_emax(device=None) -> int:
    return _L2_EMAX_GB300 * l2_bytes(device) // _L2_GB300


def l2_hint(read_width, step_tokens, bank_rows, row_bytes, device=None):
    """True iff the step's read density read_width * step_tokens / bank_rows is at most _L2_DMAX and its read-entry bytes
    read_width * step_tokens * row_bytes are at most _l2_emax(device) (device: the bank's, default the current device).
    Shapes and L2 size only; the deterministic forward calls it once per call with its tokens per step. Python ints: no
    overflow."""
    num, den = _L2_DMAX
    reads = read_width * step_tokens
    ok = (read_width > 0 and step_tokens > 0 and bank_rows > 0 and row_bytes > 0
          and reads * den <= num * bank_rows and reads * row_bytes <= _l2_emax(device))
    return ok


def _module(l2=False):
    """The module for the caller's L2 decision l2 (True: the L2-hinted source, chosen by the deterministic forward with
    l2_hint; False: the unhinted source). One cached build per l2. _module.cache_clear() clears every build (the
    functools.cache API callers may rely on)."""
    m = _build(bool(l2))
    if l2:
        L2_FETCH[0] += 1
    return m


@functools.cache
def _build(l2):
    return load_cuda_extension(
        "sdm_token_gather_dc",
        cpp_sources=_cpp(),
        cuda_sources=_source(l2),
        functions=("token_gather_dc", "token_gather_dc_plan", "token_gather_dc_step", "token_gather_dc_launch_addr"),
        extra_cuda_cflags=("-O3",),
    )


_module.cache_clear = _build.cache_clear


def pdl_ok(device):
    """PDL may be requested for launches on `device` iff it is sm_90+ (the hooks compile away below sm_90, so a
    programmatic launch there could overlap its predecessor unguarded: never request it)."""
    from .mover_fx_cuda import fwd_pdl_ok
    return fwd_pdl_ok(device)


def token_gather_ns(bank, idx, weights, out, geometry, *, q_idx=None, q_weights=None, q_out=None,
                    tokens, chunk, valid, row0=0, out_row0=0, snapc=None, kpos=None, crow=0, pdl=False, l2=False):
    """token_gather without the per-entry stash (outputs bit-identical to token_gather's). With ``snapc`` and ``kpos`` it
    stores the gathered raw write rows of the valid group FIRSTs into the compact snapshot instead: row
    (chunk * crow + kpos[entry]) for kpos >= 0. pdl: a programmatic dependent launch; the caller asserts that the launch
    right before this one wrote none of the slots, weights or kpos (the deterministic forward: the previous step's DW).
    l2: the caller's l2_hint decision."""
    if snapc is None:
        snapc = torch.empty(0, device=bank.device, dtype=bank.dtype)
        kpos = None
    _module(l2).token_gather_dc(bank, idx, weights, snapc, out, q_idx, q_weights, q_out, row0,
                              row0 if kpos is not None else 0, out_row0, tokens, chunk, valid,
                              geometry.vec_bytes, geometry.lanes, geometry.groups, geometry.tpc,
                              kpos, crow, 1 if pdl else 0)

def tail_supported(bank, chunk_size, quant="none") -> bool:
    """movers.tail_admits hook (native tails): this gather runs the call's tail class
    (rows that are not 16-byte multiples take element-sized vectors, see gather_cuda.gather_geometry: arch-neutral code)
    on every backend of _device.backend (modern, a100; rtx with chunks of <= 128 tokens, rule 0e), so a tail class takes
    the same route everywhere. Unquantized snapshots only."""
    import torch as _t
    # snapshot_quant bf16 (float32 banks): the DW writes the compact rows; this gather is the unquantized one
    if quant not in ("none", "bf16") or not bank.is_cuda or bank.dtype not in (_t.bfloat16, _t.float32):
        return False
    from ._device import backend
    try:
        b = backend(bank.device)
    except RuntimeError:
        return False
    return b in ("modern", "a100") or (b == "rtx" and chunk_size <= 128)
