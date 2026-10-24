# Changelog

## 2.0.0

New Sparse Delta Memory kernels: one chunked write/read op for all supported shapes (replacing
v1's per-shape kernels), deterministic, and faster than v1 (README, Speed). For v1's
configurations (two square product-key factors), parameter names and shapes are unchanged and v1
state dicts load with `strict=True`.

### Upgrading from v1
- Imports: `from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs` (top-level
  package; in a checkout, `lingua.sparse_delta_memory` is a deprecated alias with the same module
  objects and one `FutureWarning` per process; the pip package does not include it).
- v1 configs and checkpoints load unchanged; `memory_block_size` is now `chunk_size` (the old name
  is still accepted, with a `DeprecationWarning`; setting both to different values raises).
- New defaults: `chunk_size` 64 (it was 256 in 1.0.0; set `chunk_size: 256`
  to keep it; the shipped 1.0.0 configs and pickled 1.0.0 layer arguments set it) and `num_writes` /
  `num_reads` 64 (were 1). The chunk size changes outputs only through floating-point rounding.
- Outputs are not bitwise those of v1: the new kernels round differently, and exact top-k ties go to
  the larger rank sum, then the larger first rank (slot score: the float32 sum of the factor
  scores).

### Package
- pip package: `pip install sparse-delta-memory` (2.0.0) installs the top-level package
  `sparse_delta_memory` (Python >= 3.12; pins the tested `torch==2.13.0` and `triton==3.7.1`:
  install the CUDA 13.0 build of torch first; also installs `ninja` and `einops`). The CUDA
  extensions compile on first use and need a CUDA toolkit matching `torch.version.cuda` and a host
  C++ compiler. The training code (`lingua/`, `apps/`), benchmarks and tests stay in this
  repository.

### Deterministic
- Outputs, the final bank and every gradient are bitwise identical across repeated calls, fresh
  processes and CUDA-graph replays, for fixed inputs, settings and GPU type (v1 used float atomics);
  not across GPU types, library builds, rank counts or `chunk_size` values (a different rounding by
  design). See README, Determinism.
- `grad_bank_fp32` (opt-in): the backward accumulates the bank gradient in float32 and casts it back
  once.
- Known limitations: with a cold extension cache the first call compiles the CUDA extensions
  (about 6 minutes for a no-grad call, 10 for a training call on a GB300 node).

### New
- Any number of product-key factors of any widths (`product_key_factor_dims`); exact product-key
  top-k with a deterministic sparse backward.
- CUDA graphs of the write/read (`cuda_graph`; on in `debug_sdm.yaml`, `debug_hybrid.yaml` and
  `sdm_flagship.yaml`), with `suspend_cuda_graphs()` and `reset_cuda_graphs()`; layers of one shape
  in different checkpoint regions share one set of graphs, and a replayed call's state copies its
  final bank at the first read (README, CUDA graphs).
- A single-token decode kernel; generation decodes SDM models from a CUDA graph of the decode step
  (`sdm_decode_cuda_graph`, default `true`).
- bf16 snapshots on a float32 bank (`snapshot_quant="bf16"`): training saves the backward snapshot
  in bfloat16, half the size of `none`'s float32 snapshot, on the same CUDA kernels as `none` for
  every head dim (the forward is `none`'s, bit for bit, measured on H100 and GB300; the gradients of
  `k_val`, the gates and `q_val` read the bfloat16-rounded rows). On a bfloat16 bank,
  `snapshot_quant="bf16"` changes nothing: bit-identical to `"none"` (measured on H100 and GB300). A
  training call these kernels do not take, including one whose snapshot reaches `2**31` rows, raises
  `UnsupportedConfigurationError` before the op does any work.
- `benchmarks/` (layer, model and decoding speed against v1) and a larger test suite.

### Behavior changes
- Calls are routed on whether they may need a backward, not on train/eval mode.
- float16 raises. Out-of-range settings (for example more writes or reads per token than
  `slots_per_head`) raise at construction instead of being capped.
- Decoding with a cache bank smaller than the configured slots now raises instead of reading out of
  bounds.
- A direct `sdm_write_read` call that runs several streams over one shared bank must pass
  `shared_bank=True`, which asserts that the streams' slots are disjoint.
- No fallback paths: one native kernel family per supported GPU class, and an explicit error
  otherwise. Refusals are raised at the entry of the write/read op (`sdm_write_read`); in a layer
  call, the projections, top-k and bank copy (with `cuda_graph`, also input copies) run first.
  Training calls (calls that may need a backward) that v1 accepted but that have no native route in
  2.0.0 raise `UnsupportedConfigurationError` (a `ValueError`) before the op does any work, naming
  the reason (by code reading):
  - int4 snapshots on a float32 bank whose `ceil(head_dim / 128)` equal column slices are 4 mod 8
    columns wide (68 head dims, e.g. 12, 36, 100, 136, 200, 300): use `snapshot_quant="fp8"` or
    `"none"`; native support is planned for 2.0.1.
  - fp8 / int4 snapshots for the head dims without a quantized route (head_dim above 16384, not a
    multiple of 4, or not splitting into `ceil(head_dim / 128)` equal column slices of 16-byte
    rows), on every GPU: use `snapshot_quant="none"`; support is planned for a later release.
  - head_dim above `2**20`, on every GPU and also without a backward (the native kernels take
    head dims up to `2**20`; v1 did not check it): use a smaller head_dim.
  Also by code reading: on RTX GPUs, `chunk_size` above 128 raises a `ValueError` before the op does
  any work, with or without a backward (v1 accepted it, with 256 as its default): use `chunk_size`
  <= 128 there. Calls without a backward (`torch.no_grad()`, inference) run as
  `snapshot_quant="none"`, so none of the snapshot refusals apply to them. Training whose kernel
  plans fit and decode are not refused on any supported GPU; RTX GPUs are not tested yet: each gets
  a warning when the CUDA extensions load (Known limitations).
- The WY solve always uses IEEE float32; the dual matmul of bfloat16 banks uses TF32; other
  float32 matmuls follow `torch.backends.cuda.matmul.allow_tf32`. Autocast is off inside the op.

### Configs
- `sdm_flagship.yaml` and `sdm_7B.yaml`: 64 writes and reads per token, the paper's setting
  (previously 128); `sdm_flagship.yaml`: chunk 64
  (previously 256).
- SWA window 128, the paper's, in the 1.4B and 7B pairs (previously 1024); `baseline_1.4B.yaml`
  gates the output of its full-attention layers (`full_attn_gate`).

### Removed
- The per-shape kernels and their fallbacks, `triton_argsort`, the context-parallel code (no
  config enabled it) and `SNAPSHOT_QUANT.md`.
- `GatedSparseMemoryWriteRead` (use `sparse_delta_memory.memory_ops.sdm_write_read`),
  `fused_decode_step` (use `sparse_delta_memory.decode.sdm_decode_step`), `fast_embedding_bag` (no
  replacement: the op gathers its own rows) and `SparseDeltaMemory.read` (reads happen inside the
  write/read call).

### Known limitations
- A second backward through one SDM call (`retain_graph=True`) raises a RuntimeError (the first
  backward restores the bank in place), except on calls the `cuda_graph` option serves (not a
  differentiable call that starts from a cache).
- Host-side checks: the GPU-build check runs once per device and build environment when the CUDA
  extensions load, never per call; the argument checks run on every op call and the refusal checks
  once per device and call signature (a later call is one memo lookup).
- Every supported GPU class runs the default `snapshot_quant="none"` on a bfloat16 bank, with no
  warning except on RTX GPUs (below). A100-class GPUs (sm_80: A100, A30, A800) run every snapshot
  mode: tested on A100 with this release (training with unquantized, fp8 and int4 snapshots, and
  bf16 snapshots on a float32 bank). RTX GPUs (sm_86: RTX 30,
  A10, A40, A16, A2; sm_89: RTX 40, L4, L40, L40S, RTX 6000 Ada; sm_120: RTX 50, RTX PRO 6000
  Blackwell) are designed to run on the portable CUDA kernels (chunk_size <= 128); not tested on
  RTX hardware: each RTX GPU gets one warning per process when the CUDA extensions first load,
  whatever the snapshot mode (by code reading). A100, H100, B200 and GB300 GPUs never warn. The
  warning is `sparse_delta_memory.UntestedHardwareWarning` (a `UserWarning`); silence it with
  `warnings.filterwarnings("ignore", category=sparse_delta_memory.UntestedHardwareWarning)`.
  Inference on these GPUs runs the unquantized path whatever `snapshot_quant` is set (measured on
  A100; for bf16 snapshots, by code reading).
- Planned for 2.0.1: opt-in per-layer activation checkpointing.
- Tested with torch 2.13.0, triton 3.7.1 and CUDA 13.0 on NVIDIA GB300; the test suite also on
  H100.

## 1.0.0: initial public release (2026-07-06)
