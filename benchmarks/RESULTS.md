# Measured results

## This version against the previous release (GB300)

| | |
|---|---|
| records | the run behind the top-level README's speed plot (*Speed*); every per-process record, the equivalence verdict and the checks below are in `release_compare_gb300.json` |
| previous release | the first public version of this repository (commit `183e7df`); its kernels accumulate colliding slots with float atomics; no CUDA-graph option for SDM training (its kernel notes list CUDA graphs for the Phase 2 loop as 1.5x slower), so it always runs eagerly |
| this version | deterministic kernels (outputs, final bank and gradients bitwise reproducible; *Determinism* in `sparse_delta_memory/README.md`), timed eagerly and with the new layer option `cuda_graph`: each SDM layer replays its write/read, forward and backward, from CUDA graphs, while projections, top-k and gates stay eager (*CUDA graphs* in `sparse_delta_memory/README.md`) |
| whole-step graph | both versions are also timed with the whole training step captured as one CUDA graph by the benchmark script (GPU time without host launch overhead; not a feature of either release) |
| software | the same benchmark scripts, `PYTHONPATH` switched between the two checkouts, each with its own kernel build caches (torch 2.13, CUDA 13.0, triton 3.7.1, bf16, `allow_tf32` off) |
| 3:1 model (`bench_sdm_gdn.py`) | 24 layers, 6 × [SWA, SWA, SWA, SDM], at the flagship width, with the SDM layer below; SWA with 16 heads of 128, as in `sdm_flagship.yaml`, at its 128-token window (the paper's); causal `flex_attention` without rotary embeddings, SwiGLU FFNs, no embedding or output layer, loss = mean of the squared output |
| steps | layer step: the forward and backward pass (no optimizer); model step: the forward pass, the backward pass to every parameter, and a fused AdamW update of every parameter, memory banks included (lr 0, so the weights stay fixed; zeroing the gradients is not timed) |
| shapes | layer and model at 32K tokens per step: B × T = 1 × 32K, 2 × 16K and 4 × 8K; one GPU per process; tokens/s/GPU = B × T / step time; speedup = the previous release's time / this version's time |
| run | bitwise-identical weights (sha256) and identical inputs for both versions; four exclusive 4-GPU NVIDIA GB300 nodes, one per B × T (layer and model) and one for decoding, each running both versions of its cells, timed while the equivalence check below ran on the same commits and harness |
| statistic | median over its node's 6 rounds of fresh processes, each process reporting its median over 20 iterations (30 for the model) after 10 warmup iterations; a first pass compiles the kernels before any timing |
| concurrency | whole-step CUDA graphs (GPU-bound): four processes at a time, one per GPU; host-bound runs (eager, `cuda_graph=True` and decoding): two at a time, one per CPU socket, each pinned to all 72 cores of its socket and to its memory, with nothing else running on that socket; in each round both versions of a cell run back to back on one GPU, alternating which goes first, and the cells rotate over the GPUs between rounds |
| host-bound rule | `release_tables.py --host-rounds quiet-socket-or-uncontended`: neither socket lane (CPU socket k of all four nodes) kept its host-bound cells within 10% over its rounds (spreads 30%, 65%), so per cell, version and timed metric (layer: forward + backward; model: step; decoding: tokens/s per context), the median over both sockets' processes within 1.15x of the cell's fastest process; a cell's other columns (forward / backward split, peak memory) come from the same processes; graph-only cells use all 6 rounds; `release_compare_gb300.json` keeps every record with its socket, and each node's cells (`attachments.timing_runs`) |
| † CPU-noisy | 4 plotted quantities had fewer than 3 such processes; the plot draws them hatched, at the median over the processes of the runs of their code (the previous release: seven earlier runs and this run; this version: this run), with a whisker from the smallest to the largest of those runs' medians when there are at least two; the tables give this run's median over all its processes |
| decoding | the `sdm_flagship.yaml` model at its 128-token SWA window, one set of random weights for both versions (bf16, greedy, batch 1); tokens/s over 256 steps from the same prefill state, the median of 3 runs per process and of the uncontended processes (`benchmarks/bench_decode.py`); the previous release decodes eagerly only (`generate_next_token`; it ignores the config's training option `cuda_graph`); this version decodes eagerly and with the generator's CUDA-graph decode step (`sdm_decode_cuda_graph`), which gives the same tokens as its eager decoding; the two versions' tokens are not compared with each other (their bf16 slot selections differ, *Equivalence* below); both decode the same number of steps with the same shapes |
| spread | of a cell's per-process medians ((max - min) / median, over the processes used): at most 1.9% for the layer as a whole-step graph and 1.6% for the model; 6.7% (layer) and 1.3% (model) with the `cuda_graph` option; up to 15.6% (layer) and 34.5% (model) eager; 49.9% (eager) and 14.7% (CUDA graph) for decoding; per round, the eager speedup of a layer row ranges from 2.03x to 7.72x, of a model row from 1.38x to 4.05x, of eager decoding from 0.95x to 1.81x |

The SDM layer is the paper's configuration at the width of
`sdm_flagship.yaml`:
- dim 2048, 2 heads of 1024 (two product-key factors of 256, so 65,536 slots per head);
- W = R = 64, chunk size C = 64;
- softmax read and write weights, output gate, LayerNorm on the readings, learned initial
  memory.

**The plot's values** (ms; speedup over the previous release; † CPU-noisy, see above):

| | previous release, eager | this version, eager | this version, CUDA graphs |
|---|---|---|---|
| layer, 1 × 32K | 210 | 36.4 (**5.78x**) | 33.5 (**6.27x**) |
| layer, 2 × 16K | 117 | 31.3 (**3.75x**) | 30.0 (**3.91x**) |
| layer, 4 × 8K | 68.8 | 30.5 (**2.26x**) | 29.4 (**2.34x**) |
| model, 1 × 32K | 1460 †, runs 1369-1548 | 403 (**3.63x**) | 392 (**3.73x**) |
| model, 2 × 16K | 727 | 373 (**1.95x**) | 368 (**1.98x**) |
| model, 4 × 8K | 532 | 372 (**1.43x**) | 368 (**1.45x**) |
| decoding, 1K | 31.7 | 24.3 (**1.30x**) | 3.35 (**9.47x**) |
| decoding, 8K | 30.8 | 30.2 (**1.02x**) † | 3.50 (**8.81x**) |
| decoding, 32K | 31.1 †, runs 27.5-36.5 | 28.7 (**1.08x**) † | 4.01 (**7.74x**) |

*Layer, eager* (backward timed from forward end to backward end), previous release → this version:

| B x T | forward ms | backward ms | fwd + bwd ms | tokens/s (fwd + bwd) |
|---|---|---|---|---|
| 1 x 32K | 88.4 → 15.0 (**5.88x**) | 121.7 → 21.4 (**5.70x**) | 210.3 → 36.4 (**5.78x**) | 156k → 900k |
| 2 x 16K | 48.6 → 12.3 (**3.97x**) | 68.9 → 19.0 (**3.63x**) | 117.3 → 31.3 (**3.75x**) | 279k → 1048k |
| 4 x 8K | 30.1 → 11.8 (**2.54x**) | 38.5 → 18.6 (**2.07x**) | 68.8 → 30.5 (**2.26x**) | 476k → 1075k |

*Layer, whole-step CUDA graph,* both versions:

| B x T | forward ms | backward ms | fwd + bwd ms | tokens/s (fwd + bwd) |
|---|---|---|---|---|
| 1 x 32K | 34.0 → 11.7 (**2.92x**) | 40.1 → 20.9 (**1.91x**) | 74.1 → 32.6 (**2.27x**) | 442k → 1005k |
| 2 x 16K | 27.7 → 10.3 (**2.69x**) | 30.3 → 18.8 (**1.61x**) | 58.1 → 29.1 (**1.99x**) | 564k → 1125k |
| 4 x 8K | 25.1 → 10.1 (**2.50x**) | 24.9 → 18.4 (**1.35x**) | 50.1 → 28.5 (**1.76x**) | 654k → 1151k |

*Model, every run:*

| SWA window | B x T | run | forward + backward ms | step ms | tokens/s/GPU | peak GiB |
|---|---|---|---|---|---|---|
| 128 | 1 x 32K | previous release, eager | 1361.7 | 1368.7 | 24k | 128 |
| 128 | 1 x 32K | this version, eager | 395.8 | 402.8 | 81k | 125 |
| 128 | 1 x 32K | this version, `cuda_graph=True` | 384.6 | 391.7 | 84k | 133 |
| 128 | 1 x 32K | previous release, whole-step graph | 624.9 | 631.9 | 52k | 128 |
| 128 | 1 x 32K | this version, whole-step graph | 379.3 | 386.4 | 85k | 125 |
| 128 | 2 x 16K | previous release, eager | 719.6 | 726.6 | 45k | 130 |
| 128 | 2 x 16K | this version, eager | 365.6 | 372.7 | 88k | 126 |
| 128 | 2 x 16K | this version, `cuda_graph=True` | 360.5 | 367.6 | 89k | 134 |
| 128 | 2 x 16K | previous release, whole-step graph | 526.8 | 533.8 | 61k | 130 |
| 128 | 2 x 16K | this version, whole-step graph | 355.7 | 362.7 | 90k | 126 |
| 128 | 4 x 8K | previous release, eager | 525.0 | 532.1 | 62k | 133 |
| 128 | 4 x 8K | this version, eager | 364.6 | 371.7 | 88k | 129 |
| 128 | 4 x 8K | this version, `cuda_graph=True` | 360.7 | 367.7 | 89k | 137 |
| 128 | 4 x 8K | previous release, whole-step graph | 484.4 | 491.4 | 67k | 133 |
| 128 | 4 x 8K | this version, whole-step graph | 355.6 | 362.6 | 90k | 129 |

*Decoding* (conditions above), tokens/s (speedup over the previous release):

| context | previous release, eager | this version, eager | this version, CUDA-graph decode step |
|---|---|---|---|
| 1K | 32 | 41 (**1.30x**) | 299 (**9.5x**) |
| 8K | 32 | 33 (**1.02x**) | 286 (**8.8x**) |
| 32K | 27 | 35 (**1.27x**) | 249 (**9.1x**) |

*Equivalence* (`benchmarks/equivalence.py`, before timing; identical weights and inputs, training
path, B = 1, T = 8K; verdict in `benchmarks/release_compare_gb300.json`) and *tests*:

| check | result |
|---|---|
| same slots (fixed to the same exact top-k): whole layer of each version vs a float64 rebuild of the layer (output, input gradient, gradients of every tensor of more than 16 elements) | relative RMS error at most 3.3e-6 in fp32 and 1.7e-2 in bf16 |
| same slots: per-head gate tensors (`A_log`, `dt_bias`, the `a_proj` and `b_proj` biases) | within 3.4e-5 (fp32) and 9.7e-2 (bf16) relative max error |
| same slots: write/read op on identical arguments vs a float64 token loop, at C = 64, 128 and 256 | at most 2.9e-6 (previous release) and 2.0e-7 (this version) in fp32, and 5.9e-3 (both) in bf16 |
| same slots: the 24-layer 3:1 model (SWA window 1024) | the two versions agree to 1.1e-5 relative RMS in fp32 for the output and the input gradient; parameter gradients, compared through per-parameter norms and up to 4,096 sampled entries per parameter, within 1e-4 relative RMS (1e-3 relative max for tensors of at most 16 elements) |
| slot selection | differs: each version's product-key top-k is exact under its own scoring (the previous release ranks the bf16-rounded sum of the two factor scores, this version the float32 sum); in fp32 the same slots except one out of 2,097,152 (exact ties); in bf16 they choose differently among scores that tie after rounding: 1.4% of the slots differ, 61% of the (head, token) rows have at least one different slot of their 64; this version's selection is an exact top-k of the float32 scores on every row, the previous release's on 49% of them |
| own selections, bf16 | the layer outputs differ by 11% relative RMS (at most 1.7e-2 with the same slots), so most of it comes from selecting other slots: in bf16 the two versions are not the same function; the timed operations (W, R, C, shapes) are the same |
| `cuda_graph: true` vs eager | every recorded tensor bitwise equal (2,439 tensors; `equivalence.cuda_graph_option`, the rule of `equivalence.py run-cudagraph` and `compare`): 4 consecutive training steps on different inputs, every weight (memory bank included) changed in place between steps, so a replay that reused stale inputs, weights or outputs would fail; one layer with fixed slot selections and the 3:1 model with fixed and its own, fp32 and bf16, against eager runs of the same steps |
| shim | the fixed-selection blocks use a fixed-order gather backward (`index_put_` with `accumulate=True`) instead of `torch.gather`'s CUDA backward (float atomics) |
| not covered by the check | B × T = 1 × 32K, 2 × 16K and 4 × 8K (the check runs B = 1, T = 8K), the model with the 128-token window (the check uses 1024), the whole-step graph and AdamW paths, eval and decoding, snapshot quantization, activation checkpointing and distributed runs |
| tests, this release's package | whole fast tier on A100: 3,526 passed, 278 skipped; targeted suites on GB300 and H100; failures only in tests that need `wandb` or `sentencepiece` |

**To reproduce** (results go to `results/release/`; the plot needs matplotlib):

```bash
git worktree add ../sdm-previous 183e7df       # the previous release
bash benchmarks/run_release_compare.sh ../sdm-previous . results/release equivalence
bash benchmarks/run_release_compare.sh ../sdm-previous . results/release timing
# (timing runs the cuda_graph=True cells only under a VERIFIED option verdict, VERDICT=...;
#  the run above skipped that check, running the equivalence step alongside: VERIFIED)
# the tables above: timing with REPEATS=6 on four nodes, ONLY='*_B1_T32768*', '*_B2_T16384*',
#  '*_B4_T8192*', 'decode_*', records merged (lane k = CPU socket k of every node)
# --host-rounds as for the tables above (all: every round of every cell)
python benchmarks/release_tables.py results/release --host-rounds quiet-socket-or-uncontended --json results/release/release_compare.json
python benchmarks/plot_release.py results/release/release_compare.json --out benchmarks/speedup_gb300
```
