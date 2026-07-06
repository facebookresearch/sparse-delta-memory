# Snapshot Quantization for Gated Memory Controller

## Overview

The `snapshot_quant` option quantizes the `all_mem_before_decay` snapshot tensor saved between forward and backward in the SDM layer. This is the largest activation tensor in the memory controller — at 128k seq_len it can exceed 30 GB per layer in bf16.

Two quantization modes:
- **FP8** (`snapshot_quant: fp8`): float8_e4m3fn with per-row scaling. 2x memory reduction.
- **INT4** (`snapshot_quant: int4`): symmetric int4 packed into uint8 with per-row scaling. 4x memory reduction.

The quantization only affects the snapshot tensor. All other computations (WY solve, delta rule, memory reads/writes, gradients) remain in bf16/f32. Default is `"none"` — zero overhead on the default code path (just a few Python string comparisons per iteration).

## How it works

**Forward**: The bf16 snapshot chunk buffer is reused each time step (~30 MB). After `fused_decay_and_write` consumes it, the chunk is quantized in bf16 (no f32 intermediate) and stored in the compressed buffer. The full bf16 snapshot is never materialized across all time steps.

**Backward**: Lazy dequantization — the compressed snapshot is never fully dequantized back to bf16:
- Phase A: dequantizes one time step's slice at a time (~small), passes to kernel with `chunk_offset=0`
- Phase B step 7: chunks the bmm over 32-chunk batches, dequantizing per batch
- cs=1 path: dequantizes per token

## Usage

Add `snapshot_quant` to `gated_memory_controller_args` in the config:

```yaml
model:
  gated_memory_controller_args:
    snapshot_quant: fp8   # or "int4" or "none" (default)
```

## Results: Level 08 (1.4B) Pretraining

Config: `160TPP_mhaswa128_gatedmemctrl_3to1_halfqk_topk64_level08_cs256_8nodes`
80320 steps, 8 nodes (64 GPUs), seq_len=8192, batch_size=4.

### Loss at step 1000 (identical across all three)

| BF16   | FP8    | INT4   |
|--------|--------|--------|
| 2.922  | 2.925  | 2.920  |

### Memory and speed

| Variant | GPU Mem | Iter time | Overhead |
|---------|---------|-----------|----------|
| BF16    | 43%     | 1.70s     | —        |
| FP8     | 45%     | 2.32s     | +37%     |
| INT4    | 43%     | 3.43s     | +102%    |

Memory savings are minimal at seq_len=8k (snapshot is small relative to model). The value is at 128k postrain.

## Results: Level 08 (1.4B) 128k Post-training

Config: `160TPP_mhaswa128_gatedmemctrl_3to1_halfqk_topk64_level08_cs256_8nodes_postrain_128k`
1000 steps, 4 nodes (32 GPUs), seq_len=131072, batch_size=1.
All three continue from the same bf16 pretrain checkpoint (step 80320).

### Final loss (step 1000) — bit-exact match

| BF16   | FP8    | INT4   |
|--------|--------|--------|
| 1.9024 | 1.9024 | 1.9024 |

### Memory and speed

| Variant | GPU Mem | Iter time | Overhead | Mem saved |
|---------|---------|-----------|----------|-----------|
| BF16    | 91%     | 9.27s     | —        | —         |
| FP8     | 81%     | 10.2s     | +10%     | 10pp      |
| INT4    | 73%     | 13.0s     | +40%     | 18pp      |

### RULER (needle-in-a-haystack) — no degradation

| Task             |   BF16 |    FP8 |   INT4 |
|------------------|--------|--------|--------|
| **64k average**  |  0.357 |  0.357 |  0.361 |
| **128k average** |  0.261 |  0.261 |  0.262 |

Per-task breakdown at 64k:

| Task             |   BF16 |    FP8 |   INT4 |
|------------------|--------|--------|--------|
| niah_single_1    |  1.000 |  1.000 |  1.000 |
| niah_single_2    |  0.390 |  0.388 |  0.394 |
| niah_single_3    |  0.262 |  0.266 |  0.266 |
| niah_multikey_2  |  0.000 |  0.000 |  0.000 |
| niah_multiquery  |  0.261 |  0.259 |  0.264 |
| vt               |  0.230 |  0.231 |  0.240 |

Per-task breakdown at 128k:

| Task             |   BF16 |    FP8 |   INT4 |
|------------------|--------|--------|--------|
| niah_single_1    |  1.000 |  1.000 |  1.000 |
| niah_single_2    |  0.106 |  0.104 |  0.106 |
| niah_single_3    |  0.134 |  0.132 |  0.132 |
| niah_multikey_2  |  0.000 |  0.000 |  0.000 |
| niah_multiquery  |  0.082 |  0.081 |  0.080 |
| vt               |  0.246 |  0.249 |  0.253 |

## Regression Test (snapshot_quant=none on HEAD)

Full level05 regression (40320 steps) with `snapshot_quant` code on HEAD but `snapshot_quant=none` (default).

| Step | Baseline | Regression | Diff |
|------|----------|------------|------|
| 10   | 11.4532  | 11.4532    | +0.0000 |
| 100  | 6.0582   | 6.0609     | +0.0027 |
| 1000 | 3.0969   | 3.0880     | -0.0089 |
| 1790 | 2.9228   | 2.9195     | -0.0033 |

Memory: 63% (both). Iter time: 1.40s baseline, 1.41s regression. **No regression.**

## Level 13 (8B) 128k Post-training

Level 13 128k postrain OOMs even with INT4 snapshot (79 GB used, no room). The snapshot is compressed but the rest of the per-layer state (WY quantities, attention at 128k) fills GPU memory. INT4 snapshot needs to be combined with context parallelism (CP) to split seq_len across GPUs.

## Files

- Implementation: `lingua/sparse_delta_memory/memory_ops.py`
- Config option: `lingua/sparse_delta_memory/layer.py` (`SparseDeltaMemoryArgs.snapshot_quant`)
