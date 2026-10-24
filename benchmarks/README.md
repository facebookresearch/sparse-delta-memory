# SDM benchmarks

Each script times one configuration per process and writes one JSON file. The SDM code timed
is the `sparse_delta_memory` first on `PYTHONPATH`, so two checkouts are compared by
switching `PYTHONPATH` (give each its own `TORCH_EXTENSIONS_DIR` and `TRITON_CACHE_DIR`).
Both scripts refuse to time while another process uses the GPU (`--allow-busy` overrides).

- `bench_sdm.py`: one `SparseDeltaMemory` layer.
  - `--mode train`: forward + backward, eager and as one CUDA graph of the step. `--sac`
    wraps the layer in lingua's selective activation checkpointing; `--sdm-cuda-graph` turns
    on the layer's `cuda_graph` option (eager timing only; the JSON records its capture and
    replay counts).
  - `--mode eval`: no-grad prefill.
  - `--mode decode`: single-token steps with a cache.
  - `--mode op`: the layer's write/read call alone, with the GPU kernel time of its forward
    and backward. `--indices uniform|zipf|trace:FILE` replaces the layer's slot selections
    (the same selections for every checkout).
- `bench_sdm_gdn.py`: a model of `--n-blocks` x [SWA, SWA, SWA, X] blocks with X = SDM or
  fla's `GatedDeltaNet` (`pip install flash-linear-attention`), forward + backward of a
  scalar loss, eager or as one CUDA graph. SWA runs through `torch.compile(flex_attention)`.
  `--sdm-cuda-graph` turns on the layer's `cuda_graph` option (with its capture and replay
  counts in the JSON); `--sac` wraps the model in lingua's selective activation
  checkpointing.
- `bench_decode.py`: decoding of a shipped config's model (random weights, bf16, greedy,
  batch 1) at several context lengths: tokens/s of eager decoding against the generator's
  CUDA-graph decode step from the same prefill state, with the tokens and untimed logits
  checked bitwise between the two in every repetition (the record's status is `mismatch`
  otherwise). `--swa-window` overrides the config's window; `--weights FILE` loads the
  weights from FILE, or writes them there if it does not exist, so that two checkouts decode
  the same model (sha256 in the record). A checkout without the decode step (the previous
  release) is timed eagerly only; it gets the config's `sdm_args.chunk_size` as its
  `memory_block_size`, and config keys its code does not know (the training option
  `sdm_args.cuda_graph`) are ignored and listed in the record.
- `summarize.py`: markdown tables of a results directory; `--baseline DIR` adds the ratio to
  the runs with the same arguments in another directory.
- `equivalence.py`: compares two checkouts before they are timed against each other, on
  identical weights and inputs (training path, forward + backward, bf16 and fp32): the
  write/read op on identical arguments and the whole layer with the same slot selections,
  each against a float64 reference with fixed error ceilings; each checkout's own
  product-key top-k against its definition; the 3:1 model across checkouts; and this
  version's `cuda_graph` option against its eager path over consecutive steps on new inputs
  and weights. The release verdict is EQUIVALENT only if every check of the two checkouts
  passes (exit status 1 otherwise); it is an equivalence given the same selections, and it
  reports where the two selectors pick different slots; it is also NOT EQUIVALENT when a run
  comes from another harness, or a model run from another checkout than the layer run of its
  side. The `cuda_graph` option's check is a separate verdict, VERIFIED or NOT VERIFIED (it
  requires this version's commit and the same harness), which decides whether `timing` runs
  and `release_tables.py` expects the option's cells.
- `run_release_compare.sh OLD NEW OUT {equivalence|warm|timing}`: the previous release
  against this version in the 3:1 setting (the top-level README's speed plot, and the tables
  of `RESULTS.md`). `timing` refuses to run without an EQUIVALENT verdict for the same two
  commits and harness files, then times the layer and the model at 32K tokens per step
  (B x T = 1 x 32K, 2 x 16K, 4 x 8K; the model at SWA window 128) eager and as a whole-step
  CUDA graph for both versions and with this version's `cuda_graph` option, and the
  decoding of both versions (at SWA window 128, from one set of weights written by the warm
  pass): rounds of fresh
  processes, the whole-step graphs one lane per GPU, then the host-bound runs (eager, the
  option, decoding) one lane per CPU socket, pinned to the socket's CPUs and memory with
  nothing else on it; a cell's processes run back to back on one lane, and the cells rotate
  over the lanes between rounds (`ONLY` restricts the run to some cells).
- `release_tables.py OUT`: the README tables of a `run_release_compare.sh` results
  directory, after checking that the equivalence verdict is EQUIVALENT and of the records'
  two commits and harness, that every expected record is there, matches its cell and ran
  the same code as the others, and that the two versions' runs differ only in the checkout;
  `--host-rounds quiet-socket` merges the host-bound cells over the rounds of the CPU
  socket(s) whose host-bound runs all spread by at most 10%, and `quiet-socket-or-uncontended`
  falls back, when no socket does, to the median over the processes within 1.15x of each
  cell's fastest (`RESULTS.md` says which rule the README's run used); `--attach
  KEY=FILE` stores the JSON record of a check made outside these scripts; `--json FILE` also
  collects every per-process record and the equivalence verdict (`release_compare_gb300.json` is
  the run behind the README's plot and `RESULTS.md`; `harness` in it has the hash of the
  scripts it was measured with).
- `plot_release.py FILE --out PREFIX`: the README's speed plot (`speedup_gb300.png`) from that
  JSON: the layer, the model and decoding, each group with the previous release at 1.0x, this
  version eager and with CUDA graphs, and the time above each bar (CPU-noisy cells hatched,
  with their range over the runs in the JSON's `attachments.cross_runs`); `--table FILE` also
  writes the plotted values; needs matplotlib.

`bench_sdm.py --mode train` reports the forward, the backward (`bwd_ms`: the end of the
forward to the end of the backward) and their sum; `bench_sdm_gdn.py --optimizer adamw`
adds a fused AdamW step over every parameter to the timed step (`step_ms`). Both record the
checkout they imported (`sdm_checkout`: its commit, the git tree id and status of its SDM code
directory, `sparse_delta_memory/` (`lingua/` in the previous release), and the harness hash; no paths) and a sha256 of the SDM weights, so runs of two checkouts can be checked to start
from the same weights.

`RESULTS.md` has the numbers measured with these scripts on a GB300.

Both default to dim 2048, W = R = 64, C = 64, 65536 slots per head, B = 2, T = 8192, bf16.
`bench_sdm_gdn.py` defaults to the flagship config's SDM layer (2 heads of 1024) and a GDN
layer with the same 1024-wide key projection (16 heads, key head_dim 64, `expand_v` 2), with a
1024-token SWA window (`--swa-window`; the shipped flagship config uses 128);
`bench_sdm.py` defaults to 16 heads of 128 (`--hd 1024` gives the flagship layer).

```bash
PYTHONPATH=. python benchmarks/bench_sdm.py --mode train --out benchmarks/results/train.json
PYTHONPATH=. python benchmarks/bench_sdm.py --mode op --indices zipf --out benchmarks/results/op_zipf.json
PYTHONPATH=. python benchmarks/bench_sdm_gdn.py --mixer sdm --mode graph --out benchmarks/results/sdm.json
PYTHONPATH=. python benchmarks/bench_sdm_gdn.py --mixer gdn --mode graph --out benchmarks/results/gdn.json
python benchmarks/summarize.py benchmarks/results/
```

Timings are medians over `--iters` iterations of one process. Eager times of small shapes are
bound by host launch overhead and vary with the host load; compare them over several fresh
processes in alternating order. This version's SDM kernels are deterministic, so its outputs
repeat bit for bit; the previous release's SDM accumulates with atomics and its outputs differ from
run to run in the last bits.
