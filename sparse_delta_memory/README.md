# Sparse Delta Memory: layer and kernels

`SparseDeltaMemory` (`layer.py`) is a sequence mixer over a per-head memory bank of
`slots_per_head` rows, addressed by product keys and updated with the gated delta rule.
This file describes the implementation; the repository's README
(https://github.com/facebookresearch/sparse-delta-memory) covers the model, the training code and
the results.

## Install

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # first, the tested CUDA build of PyTorch
pip install sparse-delta-memory
```

```python
import torch
from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

args = SparseDeltaMemoryArgs(dim=2048, num_heads=2, slots_per_head=65536,
                             num_writes=64, num_reads=64, chunk_size=64)
layer = SparseDeltaMemory(args, layer_id=0).cuda()
layer.init_weights()
layer = layer.to(torch.bfloat16)
x = torch.randn(2, 4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y, _ = layer(x)                       # [batch, tokens, dim]; the bank advances in place
y.float().square().mean().backward()  # deterministic kernels (see Determinism)
```

Linux and an NVIDIA GPU (tested on GB300 and H100), Python >= 3.12. The CUDA extensions compile on
first use and need a CUDA toolkit matching `torch.version.cuda` and a host C++ compiler (see Build
requirements). License: CC-BY-NC 4.0.

## What the layer computes

Per token `t` and head, the key and query projections give the scores of the product-key
factors: by default two factors of `side = sqrt(slots_per_head)` entries each, or any number
`F >= 1` of factors with the widths `n_0, ..., n_{F-1}` of `product_key_factor_dims` (their
product is `slots_per_head`, and each projection has `sum(n_f)` columns per head). The exact
top-k over all sums of one score from each factor selects `num_writes` write slots
`(k_idx, k_val)` and `num_reads` read slots `(q_idx, q_val)`, weighted by `write_act` /
`read_act`; slot `(c_0, ..., c_{F-1})` is bank row `(..(c_0 * n_1 + c_1) * ..) * n_{F-1} +
c_{F-1}` (`a * side + b` for the two default factors). With `g_t = -exp(A_log) * softplus(a_proj(x_t) + dt_bias)` and
`beta_t = sigmoid(b_proj(x_t))`, only the touched slots change:

```
M[s]   <- exp(g_t) * M[s]                          for s in k_idx_t
r_t     = sum_n k_val[n] * M[k_idx[n]]
M[s_n] <- M[s_n] + k_val[n] * beta_t * (v_t - r_t)
y_t     = sum_m q_val[m] * M[q_idx[m]]
```

`y` then goes through an optional per-head LayerNorm (`normalize_readings`), an optional
sigmoid output gate and `Wo`. With `key_weighted_decay` the decay is `exp(g_t * k_val[n])`
per write; with `backprop_on_memory` the initial bank is a learned parameter.

The op (`memory_ops.sdm_write_read`) runs the recurrence in chunks of
`C = chunk_size` tokens:

- **Phase 1** (all chunks at once, independent of the bank): the write and read entries of
  each chunk are sorted by slot; segmented scans give the per-slot log-decays, and walks
  inside each slot group build the decay-weighted Gram matrices `A` (writes x writes,
  strictly causal) and `QK` (reads x writes, causal).
- **Solve**: `X = (I + tril(beta * A, -1))^-1` and `QX = QK @ X` per chunk.
- **Phase 2** (sequential over chunks, the `B * num_heads` streams side by side): a gather
  of bank rows, one dual matmul producing the corrected writes and the readings, and a
  decay-and-write into the bank. The backward walks the chunks in reverse.

The chunked form is algebraically equivalent to the token loop, so `chunk_size`
changes the output only through floating-point rounding: the order of accumulation and,
with a bfloat16 bank, the points where bank rows are rounded. A sequence need not fill its
last chunk: tokens past the end of a stream are masked out. Every sum runs in a fixed order
and every bank row has one writer per launch, so results are bitwise reproducible run to run
(see Determinism).

## Files

`layer.py` (the layer), `memory_ops.py` (the op, `sdm_write_read`), `decode.py` (the single-token
step, `sdm_decode_step`, no GPU check) and `cuda/` (the CUDA kernels); the repository README lists
every module.

## Execution

All supported shapes run the same recurrence and source kernels (compile-time specializations; the
launch geometry and a few paths vary with the shape). Newer kernels serve the calls they admit,
chosen by shape, dtype and GPU architecture only (Limits, in Determinism); the others run
another deterministic kernel of the same step, with the same guarantees. The layer routes each call
on whether it may need a backward (grad mode is on and an input requires grad), in train and eval
mode alike:

| Call | Path |
|---|---|
| may need a backward | the chunked op, saving what its backward reads |
| no backward, T > 1 | the chunked op without saved state: the snapshot and `delta_v` are per-step scratch and the chunks run in bounded slabs, so working memory does not grow with T |
| no backward, T == 1 | one decode launch (write and read of all streams); its arguments are checked only when the bank's layout changes |

Run inference under `torch.no_grad()` or `torch.inference_mode()`. An eval-mode call under
grad computes the training gradients.

Kernels of the op. Forward: a staging copy to time-major chunks (skipped when the input already has
that layout), Phase 1, the solve, then per chunk step the token gather, the dual matmul and the
decay-and-write (which also quantizes the snapshot). Backward: per chunk step, in reverse, the
gather of the bank gradient, one `baddbmm`, the write-side mover and the read-side mover; then the
solve and Phase-1 backward. Steps of `2**31` entries or more use 64-bit entry indices in the write
mover, and compact snapshots of `2**31` rows or more 64-bit row offsets in the backward.

`memory_ops.last_dispatch()` returns the launch decisions of the latest op call (updated on eager
calls and captures, never by graph replays; it reads no tensor values).

## Supported inputs and limits

No fallback paths: one native kernel family per supported GPU class, and an explicit error otherwise.

| | Supported | Otherwise |
|---|---|---|
| `num_writes`, `num_reads` | 1..256, and at most `slots_per_head` | `ValueError` at construction |
| `chunk_size` | `None` (the op's default, 64) or any integer in 16..256; at most 128 on RTX GPUs | `ValueError` at construction (on RTX GPUs, above 128: `ValueError` in the forward, before the op does any work) |
| `slots_per_head` | a perfect square, `side <= 65535`; with `product_key_factor_dims`, their product | `ValueError` at construction |
| `product_key_factor_dims` | `None`, or 1 to 512 positive ints (at most `2**31 - 1` columns in total) | `ValueError` at construction |
| `snapshot_quant` | `none`, `bf16`, `fp8`, `int4`; it only affects what training saves for the backward: inference (calls without a backward) is identical to `snapshot_quant="none"` | `ValueError` at construction |
| `snapshot_quant` `fp8` / `int4` on a call that may need a backward | head_dim <= 16384 that splits into `ceil(head_dim / 128)` equal column slices whose width is a multiple of 8 (bfloat16 bank) or of 4 (float32 bank), e.g. 64, 128, 480, 960, 1024, 2048; int4 snapshots on a float32 bank need slices whose width is a multiple of 8: slices 4 mod 8 columns wide (e.g. head_dim 12, 36, 100, 136, 200, 300) are refused (support is planned for 2.0.1). | `UnsupportedConfigurationError` (a `ValueError`) in the forward, before the op does any work, naming the reason; calls without a backward and `snapshot_quant` `none` are not limited |
| bf16 snapshots on a float32 bank (`snapshot_quant="bf16"`) on a call that may need a backward | the snapshot is saved in bfloat16, half the size of `none`'s float32 snapshot, on the same CUDA kernels as `none` for every head dim (by code reading). E.g. one SDM layer of `sdm_7B.yaml` (4 heads of 1024 columns, 64 writes per token) with a float32 bank at 128K tokens and batch 1 saves 137 GB of snapshots with `none` (float32), 69 GB with `bf16`, 34 GB with `fp8` and 17 GB with `int4` (computed as tokens x writes x head dims x bytes per element, not measured; fp8 / int4 add one float32 scale per row, about 0.1 GB). On a bfloat16 bank, `snapshot_quant="bf16"` changes nothing: bit-identical to `"none"` (measured on H100 and GB300) | `UnsupportedConfigurationError` (a `ValueError`) before the op does any work, for a call these kernels do not take (including snapshots of `2**31` rows or more); calls without a backward are not limited |
| `snapshot_quant` on A100 / RTX GPUs | every supported GPU class runs the default `snapshot_quant` `none` on a bfloat16 bank, with no warning except on RTX GPUs. A100-class (sm_80: A100, A30, A800): every mode; tested on A100 with this release (training with unquantized, fp8 and int4 snapshots, and bf16 snapshots on a float32 bank). RTX (sm_86: RTX 30, A10, A40, A16, A2; sm_89: RTX 40, L4, L40, L40S, RTX 6000 Ada; sm_120: RTX 50, RTX PRO 6000 Blackwell): every mode, designed to run on the portable CUDA kernels (chunk_size <= 128); not tested on RTX hardware: each RTX GPU gets one `sparse_delta_memory.UntestedHardwareWarning` per process when the CUDA extensions first load, whatever the snapshot mode (by code reading); A100, H100, B200 and GB300 GPUs never emit it. Inference on these GPUs runs the unquantized path whatever `snapshot_quant` is set (measured on A100; for bf16 snapshots, by code reading) | the warning is a `UserWarning`: silence it with `warnings.filterwarnings("ignore", category=sparse_delta_memory.UntestedHardwareWarning)`; the limits of the other rows (quantized tail head dims, int4 slices 4 mod 8, head_dim above `2**20`, `chunk_size` above 128 on RTX GPUs) apply as stated there |
| head_dim (`dim / num_heads`) | up to `2**20`; tested and tuned for 64..2048 | `UnsupportedConfigurationError` (a `ValueError`) before the op does any work, on every call |
| compute and bank dtype | bfloat16, float32 | an error in the forward (float16, float64) |
| `B * num_heads * slots_per_head` | below `2**31` bank rows (int32 slot ids) | `ValueError` |
| `B * num_heads * T`, T rounded up to a multiple of `chunk_size` | below `2**31` token rows | `ValueError` in the forward |
| compact snapshot rows of a training call (about `B * num_heads * T * num_writes`) | any, with 64-bit row offsets from `2**31` rows (tested on GB300 at 2,149,580,800 rows with a bfloat16 bank, `snapshot_quant` `none` and a head_dim that is not a multiple of 4; other head dims, float32 banks and fp8 / int4 snapshots by code reading), except bf16 snapshots on a float32 bank: below `2**31` | `UnsupportedConfigurationError` (a `ValueError`) before the op does any work, for bf16 snapshots on a float32 bank at `2**31` rows or more |

Also supported: `key_weighted_decay`, training forward and backward, no-grad prefill with
bounded working memory, decode, and continuing from a cache.

**Input contract.** Each token's W write slots must be distinct, and every slot index must lie
inside the bank. Out-of-range indices are undefined behaviour and are not checked per call
(checking would need a device sync). Repeated write slots are handled deterministically (the slot
decays once), counted on the device, and not refused. The layer's top-k produces distinct,
in-range slots; the count is read with `memory_ops.repeated_write_slots()`.

## Determinism

Every output, the final bank and every gradient are **bitwise identical** across repeated calls in
one process, across fresh processes and with CUDA graphs on or off, for fixed inputs, settings and
GPU type. It covers float32 and bfloat16 banks, every `snapshot_quant` mode, every chunk size,
ragged sequences, collision-heavy selections, the training forward, the no-grad prefill and decode.

Version 1 added colliding slots with float atomics, in arrival order, so repeated runs on
identical inputs differed in the last bits, and the next layer's exact top-k turned that into
different selections.

How. Every bank and bank-gradient row has ONE writer per launch (per column slice, where a long row
is split), and every sum runs in a fixed order: Phase 1 sorts each chunk's entries by slot (stably),
so a slot's entries form one group; `A`, `QK`, the write mover and the backward's movers sum each
group in a fixed order and store each row (or cell) once. Everything else was already fixed-order
(segmented scans, the gather's group partials, the WY solve, the Phase-1 backward, the dual matmul,
the decode step, product-key top-k), and no kernel is autotuned.

Also bitwise, and tested, for the chunked op (more than one token; a single-token no-grad call takes
the fused decode step, which rounds the updated bank before reading it): the CUDA-graph option's
eager fallback (a suspended capture, an outer capture, or a call the graphs do not serve) against
both graph replays and plain eager; and the training forward against the no-grad path of the same
tokens. Both hold for every snapshot mode, fp8 included, on the calls the write mover of
`cuda/mover_da2_cuda.py` (or `cuda/mover_fx_cuda.py`, which computes the same bits) serves: compute capability 9.0 or newer, a bfloat16 or float32 bank whose
head dim splits into equal slices of at most 128 columns (each a multiple of 4 columns and of 16
bytes), and a geometry that fits shared memory. Other calls run the other deterministic write
mover, whose fp8 variant sums a slot group's window in another order than the unquantized one at
head dims below 512 (except 120 on bfloat16 banks): there, with fp8 snapshots, the training forward
and the no-grad path (and the eager fallback against eager and graph) can differ in the last bits
of repeated-slot bank rows and of what is read from them. Each path is bitwise reproducible on its
own.

What is NOT promised: different GPU types, library builds or FSDP rank counts (cuBLAS and NCCL
choose other algorithms and reduction trees); a different `chunk_size`, which is a different
rounding by design; and the bits of version 1, which agree only to rounding (fixed-order group sums
versus atomic arrival order) where both select the same slots (Product-key top-k: ties).

Memory. The kernels keep Phase 1's group maps, sorted rows and per-entry records for the backward.
Peak allocated memory: a flagship layer step (2 heads of 1024, `W = R = C = 64`) 12.32 GiB at
B1 x 32K, 12.82 GiB at B2 x 16K, 13.82 GiB at B4 x 8K; the 24-layer 3:1 model 124.8 GiB at B1 x 32K,
126.3 GiB at B2 x 16K, 129.4 GiB at B4 x 8K (`benchmarks/release_compare_gb300.json`).

Limits. The newer deterministic kernels are chosen by shape, dtype and GPU architecture only,
never by the data. The Phase-1 pair kernel and the two movers need compute capability 9.0 or
newer and a geometry that fits shared memory; the forward steps and write-side backward need 8.0
or newer, `snapshot_quant="none"`, a bfloat16 or float32 bank and a head dim that is a multiple
of 4. Other calls run the other deterministic kernels: also bitwise reproducible, but slower.
Parallel streams must use disjoint slots for the whole call. The layer
gives each stream its own slot range; a direct `sdm_write_read` call that runs several streams over
ONE shared bank (`num_memory_slots` 0 or the bank's size with `num_parallel > 1`) is refused unless
it passes `shared_bank=True`, which asserts disjointness. The op does not check the slots
themselves. Two streams that write one slot in the same chunk step lose one of the two updates;
any other overlap (a read and a write of one slot, or one
slot in different steps) loses no update but leaves the step-synchronous order, since the no-grad
path may run one stream's steps before another's.

Determinism outside SDM. This repository's `attn_impl="sdpa"` and `flex_attention` paths, the
compiled FFN and RMSNorm, and NCCL all-reduce / reduce-scatter on one node measured bitwise on the
tested stack. A sliding-window FlashAttention-2 backward is bitwise only because a short window
gives each query block at most two atomic `dq` adds; with a wide window or full attention use the
library's deterministic backward (+15% on the attention call at window 128). A compiled embedding
backward over repeated token ids is NOT deterministic in float32 (inductor lowers it to atomic
scatter-add); the eager one is. Different rank counts are not guaranteed to reproduce bitwise.

## Precision

- Phase 1 accumulates in float32; `A`, `QK`, `X` and `QX` are float32.
- The WY solve always uses IEEE float32 dots (Triton `input_precision="ieee"`), whatever
  `torch.backends.cuda.matmul.allow_tf32` says: the inverse amplifies the rounding of its
  inputs. Its launch configuration is pinned, because the warp count changes the rounding.
  Its bits are pinned only for the tested stack (below).
- Every other float32 matmul follows `torch.backends.cuda.matmul.allow_tf32`: the dual
  matmul of float32 banks and the backward's cuBLAS matmuls. The dual matmul of bfloat16
  banks uses TF32.
- Autocast is disabled inside the op, so its matmuls keep their float32 operands under an
  enclosing `torch.autocast`, in eager mode and in CUDA graphs alike.
- `k_val`, `v` and `q_val` are rounded to the bank dtype, `beta` and `g` are used in float32.
  Gathers and movers accumulate in float32. A bank or bank-gradient row is written once per
  chunk: one entry's update is stored once, several entries are summed in float32 in sorted order
  and stored once. The readings come out in the bank dtype (the layer casts them to the compute
  dtype), so a float32 cache on a bfloat16 layer works.
- With fp8 or int4 snapshots, or bf16 snapshots on a float32 bank, some gradients are approximate
  (see the configuration section).
- A bfloat16 bank rounds a slot once per chunk that writes it. With realistic
  banks this is a few writes per slot; where slots take tens of writes per chunk (e.g. 256
  writes per token over a 512-slot bank) the final bank deviates from float64 by up to about
  12% of its largest value. Use a float32 bank for such regimes.

## Product-key top-k

Slot `(a, b)` scores `fp32(A[a]) + fp32(B[b])`, the sum rounded once to float32 (bfloat16
scores and their float32 upcast select the same slots). Scores compare as float32 numbers
with `-0` below `+0` and NaN counted as `-inf`. Each factor is ranked by value, larger
column first on ties; let `i` and `j` be the ranks of `A[a]` and `B[b]`. The candidates are
the pairs with `(i + 1) * (j + 1) <= k` (any other pair has at least `k` pairs of lower or
equal ranks that score at least as much), and exact ties among them go to the larger
`i + j`, then the larger `i`. The returned indices are ascending; the backward is
deterministic. `product_key_topk` combines any number of factors, in order, with the same
rule at each stage (see its docstring), and the layer uses the factors of
`product_key_factor_dims` (two by default). One factor selects its `k` best columns, larger
column first on ties.

Two to four factors of at most 65535 columns and 32 bits of slot index (every configuration of
version 1) run on `cuda/product_key_topk_cuda.py`, every other factor list (up to 512
factors) on `cuda/product_key_topk_anyf_cuda.py`; both give the same bits on the factor lists they
share (tested). Selection time grows with the number of factors and with `k`, ranking time with a
factor's width.

This tie order differs from the previous release of this repository, which selects the
same score values but may resolve exact ties to other slots. With bfloat16 projections exact
ties are common: at least one selected slot differed in 35% to 37% of rows on trained
projections at `k = 64`, and in 73% of rows on randomly initialized ones at `k = 128`. With
float32 projections ties are rare (1 row in 8192 at `k = 128`).

The dense limit: with one factor of width `d_k` and every slot selected
(`product_key_factor_dims = [d_k]`, `slots_per_head = num_writes = num_reads = d_k`, so
`d_k <= 256`), `l2_normalize` as `write_act` and `read_act`, and `key_weighted_decay` off,
each token writes and reads every row of its head's bank, which becomes the `d_k x d_v` state
of the gated delta rule (the recurrence of Gated DeltaNet). Without output gate or readings
LayerNorm and from a zero initial bank, the layer then computes that recurrence.
`tests/test_dense_limit_gdn.py` checks the op (training, no-grad, CUDA-graph and decode
paths, also from a nonzero initial bank) and the layer against a float64 reference of it,
with `snapshot_quant` `none`: they agree up to floating-point rounding (fp8 and int4 snapshots,
and bf16 snapshots on a float32 bank, make the gradients approximate).

## CUDA graphs

After the first calls, the training forward and backward do not synchronize with the host
(check with `torch.cuda.set_sync_debug_mode("error")`), so a whole training step can be
captured with `torch.cuda.graph`. Run the step eagerly at the exact shapes first: the first
calls build the CUDA extensions, compile the Triton kernels and create a small counter per
device.

```python
# layer: a SparseDeltaMemory converted to bf16, in train mode.
static_x = torch.randn(B, T, args.dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):          # eager warmup at the captured shapes
    for _ in range(3):
        out, _ = layer(static_x)
        out.float().square().mean().backward()
torch.cuda.current_stream().wait_stream(s)
layer.zero_grad(set_to_none=True)   # the graph owns the gradients
static_x.grad = None
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out, _ = layer(static_x)
    static_out.float().square().mean().backward()
# Each step: copy new data into the static input and replay.
with torch.no_grad():
    static_x.copy_(new_x)
graph.replay()   # static_out and every .grad now hold this step
```

A graph is valid for one set of shapes, dtypes and layer settings. The statistics options
(`log_memory_access_stats`, `log_memory_norms`) read values on the host; they are collected
in eval mode only and skipped during a capture. `forward` is wrapped in
`@torch.compiler.disable`: `torch.compile` compiles the rest of the model and runs this layer
eagerly. The decode step can be captured the same way; `apps/main/generate.py` does so for
SDM models (`sdm_decode_cuda_graph`, default true): the first decode step runs eager, the next
is captured, and every later step of every prompt replays it (greedy decoding is
token-identical to eager, the logits bitwise). It is captured again when a tensor it reads
moved or changed shape (checked once per prompt) or a matmul or attention-backend setting
changed; steps under autocast, and all steps after a failed capture, run eager.

### The `cuda_graph` option

`cuda_graph: true` makes each layer replay its write/read (the bank expand, the op forward
and backward, and the bank-gradient reduction) from two CUDA graphs, captured at the first
call that may need a backward (the forward) and at its backward (the backward), once per
layer or, under activation checkpointing, once per set of layers sharing them (see Memory
below). Projections, top-k and gates stay eager, so
FSDP, activation checkpointing (including lingua's selective policy) and `torch.compile` of
the surrounding model work unchanged. The field defaults to false. `debug_sdm.yaml`,
`debug_hybrid.yaml` and `sdm_flagship.yaml` set it; `sdm_7B.yaml` leaves it off (chunk 256 leaves
few launches to save). In a config of your
own, add `cuda_graph: true` under `model.sdm_args` (or pass `model.sdm_args.cuda_graph=true`).

The graphs serve one fixed call per layer, and every other call runs eager, so the option
changes no result beyond floating-point rounding:
- Calls without a backward (evaluation, prefill, decode) run eager.
- A differentiable call that the graphs do not serve runs eager as well, and saves for its
  backward what a replayed call saves (its bank parameter and inputs): its backward runs its
  forward again. So under non-reentrant checkpointing a forward and its recompute may take
  different paths. These are:
  - a call of another shape than the captured one (or under another
    `torch.backends.cuda.matmul.allow_tf32` setting), with a warning once per layer. After a
    permanent change, call `cuda_graph.reset_cuda_graphs()` between steps to capture the new
    shape; graphs are never re-captured on their own;
  - every call inside `with suspend_cuda_graphs():`, which `apps/main/train.py` uses for its
    probe (a smaller batch);
  - every call inside a CUDA-graph capture of your own (with a whole-step capture, leave the
    option off: the captured backward would run each forward again);
  - every call of a layer whose capture failed or whose first backward ran out of memory,
    until `reset_cuda_graphs()` (one warning).
- A differentiable call that starts from a `cache` runs eager (one warning).
- A backward whose graph state was overwritten before it ran (two forwards of a layer before
  one backward, a layer called twice per forward, a forward of another graphed model or of a
  layer sharing the graphs in between, a second backward with `retain_graph=True`) first
  replays its forward from the saved inputs: same gradients, one more forward per such
  backward. A checkpoint recompute right before its backward needs no extra replay (a layer
  called twice in one checkpoint region still replays the forward of its first call).
- The state a replayed call returns (`new_cache`) never exposes the graphs' static bank: its
  `memory` is a private copy of the call's final bank, taken at the first read, which is valid
  until the next SDM forward or backward that reuses the call's graph memory (a later forward
  of the layer, of a layer sharing its graphs or of a layer captured before it, or the call's
  backward). A first read after that raises a `RuntimeError` naming the fix: read the state
  right after the call and keep that copy (or a clone), or run the call under
  `suspend_cuda_graphs()`. After the backward, the state of a plain eager call (option off)
  holds the restored initial bank; a copy read earlier from a replayed call, like the bank of
  the option's eager fallback, keeps the final bank.
- Memory: without activation checkpointing each layer keeps static input, output and gradient
  buffers and its graph state for the whole step (eager keeps that state too). Under
  non-reentrant activation checkpointing (`torch.utils.checkpoint`, also with a selective
  policy), layers of one shape whose calls run in different checkpoint regions share one set
  of graphs, buffers and state, since each backward follows its own recompute: with one
  region per block, the option adds about one layer's state and buffers in all. Layers of one
  shape in the same region keep one set each, because the region's recompute runs them all
  before their backwards (eager then holds all their states too); lingua's selective policy
  wraps the whole model in one region. A layer's sharing is set by its first graphed call
  (until `reset_cuda_graphs()`); a layer first called in a region nested in another may share
  the state of an earlier layer of the outer region, which then replays its forward once more.
  Calls from several threads on graphs that share memory run one at a time.
- Layers first called on different CUDA streams capture into different memory pools; a
  replay on another stream than its capture's is ordered after the work queued on that one.

The option removes host launch overhead: 1.04x to 1.09x for the flagship layer at 32K tokens per
step on GB300 (`benchmarks/RESULTS.md`). In the 24-layer 3:1 model it adds 7.7 GiB to the peak
allocated memory (124.8 to 132.6 GiB at B1 x 32K). A whole-step capture of your own (above)
removes the overhead of the whole model and needs no option.

## Activation checkpointing and the bank state

The op keeps everything its backward reads in `save_for_backward`, and it hides its internal
ops from torch dispatch modes, so selective activation checkpointing neither caches nor
intercepts them: under non-reentrant checkpointing the op is recomputed as a whole, which
costs one extra forward of every SDM layer, and the gradient is that of the recomputed
forward. Reentrant checkpointing runs the first forward without grad, which saves nothing.
The same opacity applies to every dispatch mode: `FlopCounterMode` does not count the op,
and fake or meta tensors are not supported.

The forward advances the bank in place and the backward restores it in place: exactly with
`snapshot_quant` `none`; with fp8 or int4, the written rows come back as their dequantized
snapshot, not their original bits (a data-dependent error, about 2.5% and 10% relative RMS
on those rows for random test inputs); with bf16 snapshots on a float32 bank, as their bfloat16
rounding.
For a call that may need a backward, the
returned `SDMLayerState` (and a passed-in cache) therefore holds the final bank only until
backward runs; clone `state.memory` before backward to carry it to the next segment.
Inference calls are unaffected. For the same reason, a second backward through one op call
(`retain_graph=True`, then another backward) would return a wrong gradient, so it raises a
RuntimeError.
The `cuda_graph` option's calls are the exception: their second backward first replays the
forward and is correct (not a differentiable call that starts from a cache, which runs the
plain op).

## Configuration: `SparseDeltaMemoryArgs` (`model.sdm_args` in the configs)

| Field | Meaning |
|---|---|
| `dim`, `num_heads` | model width and number of heads; head_dim = `dim / num_heads` |
| `slots_per_head` | bank rows per head, `side ** 2` for two product-key factors of `side` entries |
| `product_key_factor_dims` | per-head product-key factor widths, any number of them (`None`: two factors of `sqrt(slots_per_head)`); their product must equal `slots_per_head` |
| `num_writes`, `num_reads` | slots written / read per token and head (top-k width; default 64) |
| `chunk_size` | chunk size `C` of the op (`None`: 64; speed; results change only by floating-point rounding; tuned configs set 128 or 256 explicitly). `memory_block_size`, its former name, is still accepted |
| `read_act`, `write_act`, `*_act_scale` | slot-weight activation: `Softmax` (scaled), `l2_normalize`, or a `torch.nn` activation name |
| `normalize_readings`, `output_gate` | per-head LayerNorm on readings; sigmoid output gate |
| `A_init_range` | decay initialization range; `dt_bias` follows GDN / FLA (`dt_bias_init` is not used) |
| `backprop_on_memory`, `query_batchnorm` | learn the initial bank; BatchNorm on the key and query projections |
| `key_weighted_decay` | per-write decay `exp(g * k_val)` |
| `snapshot_quant` | storage of the backward snapshot: `none`, `bf16`, `fp8`, `int4` |
| `log_memory_access_stats`, `log_memory_norms` | opt-in eval-mode statistics |
| `cuda_graph` | replay the write/read of differentiable calls from per-layer CUDA graphs (on in the debug and flagship configs) |
| `grad_bank_fp32` | opt-in (default `false`): the backward accumulates the bank gradient in float32 and casts it back once; no effect on a float32 bank |

`snapshot_quant`: the training forward saves, for every token and write slot, the bank row
read before decay (`B * T * num_writes * head_dim` elements per head group). `fp8` stores it
as `float8_e4m3fn` and `int4` as packed 4-bit values, each with one scale per row (a
bfloat16-rounded value stored as float32), instead of the bank dtype. The forward computes the same
values (its write mover is specialized per snapshot mode, so with `fp8` below head_dim 512 the bank
update can round differently in the last bits; see Determinism); the gradients of
`k_val`, the gates and `q_val` read the dequantized rows and are approximate, and the
backward restores the written bank rows from them. `bf16` on a float32 bank stores the rows
rounded to bfloat16 (round to nearest even), half the size of the float32 snapshot, on the same
CUDA kernels as `none` (the forward is `none`'s, bit for bit, measured on H100 and GB300; the same
three gradients read the bfloat16 rows). On a bfloat16 bank, `snapshot_quant="bf16"` changes
nothing: bit-identical to `"none"` (measured on H100 and GB300), whose snapshot is bfloat16 already.

## Changes from the previous release

See `CHANGELOG.md` in the repository. For every configuration of the previous release that this
version accepts (two square product-key factors: `product_key_factor_dims` left at `None`),
parameter names and shapes are unchanged and previous state dicts load with `strict=True`; other
factor lists can change the width of the key and query projections (and of their BatchNorm with
`query_batchnorm`).

## Build requirements

- A CUDA GPU, a CUDA build of PyTorch, a CUDA toolkit matching `torch.version.cuda`
  (`CUDA_HOME` must contain `bin/nvcc`, or set `PYTORCH_NVCC`) with its CUB headers, a host
  C++ compiler and `ninja`.
- The CUDA extensions (Phase 1 and its backward, the gathers, the movers and the backward's
  kernels, the product-key top-k) are built with `torch.utils.cpp_extension` on first use, never
  at import, and cached in `TORCH_EXTENSIONS_DIR`. With a cold cache the first call compiles them:
  about 6 minutes for a no-grad or eval call and 10 minutes for a training call on a GB300 node
  (each process that finds them cached loads them in well under a second).
- Builds are named `<base>_<16 hex digits>`, hashing sources, bindings, flags,
  torch/CUDA/Python versions, `CUDA_HOME` and the target GPU architectures, so a stale build
  is never reused and different source trees can share one cache directory.
- A build killed mid-compile leaves a `lock` file in its build directory that later loads
  wait on; the loader warns, and the file can be deleted if no build is running.
- Tested with torch 2.13.0 (CUDA 13.0), triton 3.7.1 and Python 3.12 on NVIDIA GB300
  (sm_103); the test suite also passes on H100 (sm_90). Other GPUs are untested.

## Benchmarks

Speed against the previous release: the repository README (Speed) and `benchmarks/RESULTS.md` (all
tables, setup and spreads); records: `benchmarks/release_compare_gb300.json`; scripts:
`benchmarks/README.md`.
