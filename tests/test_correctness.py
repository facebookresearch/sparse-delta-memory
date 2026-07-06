# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Correctness tests for the SparseDeltaMemory layer.

Ported from the standalone SDM repo. Tests:
1. Forward + backward run without error at various configs
2. Chunk-size invariance: outputs match across different memory_block_size values
3. Inference cache: token-by-token decode matches single-pass forward
4. Multi-head: H>1 configs run correctly
5. Deterministic weights: same seed produces same parameters

Note: SDM uses atomic operations (atomicAdd) in its Triton/CUDA kernels, so outputs
are non-deterministic across runs (typically ~1e-3 rel diff in bf16). Thresholds gate
cross-result diffs against the measured run-to-run noise floor. Run in the release env:

    PYTHONPATH=. python tests/test_correctness.py
"""
import torch

from lingua.sparse_delta_memory import (
    SparseDeltaMemory as SDMLayer,
    SparseDeltaMemoryArgs as SDMLayerArgs,
)
from lingua.sparse_delta_memory.cache import SDMLayerState as SDMState


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def stable_config(**overrides):
    # slots_per_head replaces the standalone's total num_memory_slots. For H=1 the layer
    # has slots_per_head slots; the correctness properties below are independent of the
    # exact count.
    defaults = dict(
        dim=128, slots_per_head=1024, num_writes=16, num_reads=16,
        memory_block_size=64, num_heads=1,
        log_memory_access_stats=False, normalize_readings=True,
    )
    defaults.update(overrides)
    return defaults


def make_layer(seed=42, **kw):
    torch.manual_seed(seed)
    cfg = stable_config(**kw)
    layer = SDMLayer(SDMLayerArgs(**cfg), layer_id=0).cuda().bfloat16()
    layer.init_weights()
    return layer, cfg


def _max_diff(a, b):
    return (a - b).abs().max().item()


def _self_noise(layer, x, n=3):
    """Atomics non-determinism floor: max pairwise abs diff over n identical runs."""
    outs = [layer(x.clone())[0] for _ in range(n)]
    noise = 0.0
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            noise = max(noise, _max_diff(outs[i], outs[j]))
    return noise


# Cross-result diffs must stay within this multiple of the measured atomics noise floor.
NOISE_K = 3.0
# Absolute floor so a (near-)deterministic case can't make the bound ~0.
ATOL = 1e-4
# Relative floor for CROSS-kernel comparisons (train-vs-inference, full-vs-token).
REL_FLOOR = 0.012


def test_forward_backward():
    print("=== Forward + backward ===")
    configs = [
        dict(num_heads=1),
        dict(num_heads=2),
        dict(memory_block_size=128),
        dict(key_weighted_decay=True),
        dict(snapshot_quant="fp8"),
    ]
    for extra in configs:
        label = ", ".join(f"{k}={v}" for k, v in extra.items())
        try:
            layer, cfg = make_layer(**extra)
            layer.train()
            x = torch.randn(2, 64, cfg["dim"], device="cuda", dtype=torch.bfloat16) * 0.1
            x.requires_grad_(True)
            out, cache = layer(x)
            out.sum().backward()

            out_ok = out.isfinite().all().item()
            grads_ok = all(
                p.grad is not None and p.grad.isfinite().all().item()
                for p in layer.parameters() if p.requires_grad
            )
            x_grad_ok = x.grad is not None and x.grad.isfinite().all().item()
            check(label, out_ok and grads_ok and x_grad_ok)
        except Exception as e:
            check(label, False, str(e))
    print()


def test_chunk_size_invariance():
    """Output should be nearly identical for different chunk sizes (within atomics noise)."""
    print("=== Chunk-size invariance ===")
    for H in [1, 2]:
        layer_a, _ = make_layer(num_heads=H, memory_block_size=64)
        layer_b, _ = make_layer(num_heads=H, memory_block_size=128)
        layer_b.load_state_dict(layer_a.state_dict())
        layer_a.train(); layer_b.train()

        x = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        out_a, _ = layer_a(x.clone())
        out_b, _ = layer_b(x.clone())

        self_noise = _self_noise(layer_a, x, n=3)
        cross_diff = _max_diff(out_a, out_b)
        scale = max(out_a.abs().max().item(), 1e-10)
        bound = max(NOISE_K * self_noise, ATOL)

        check(
            f"H={H}: cs=64 vs cs=128",
            cross_diff <= bound,
            f"cross={cross_diff:.2e}, {NOISE_K:g}*noise={bound:.2e}, "
            f"rel={cross_diff/scale:.2e}",
        )
    print()


def test_inference_cache():
    """Token-by-token decode must match single-pass within atomics tolerance."""
    print("=== Inference cache ===")
    for H in [1, 2]:
        layer, _ = make_layer(num_heads=H)
        layer.eval()

        torch.manual_seed(99)
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16) * 0.1

        with torch.no_grad():
            out_full, _ = layer(x)

            cache = None
            outs = []
            for t in range(64):
                o, cache = layer(x[:, t:t+1], cache=cache)
                outs.append(o)
            out_tok = torch.cat(outs, dim=1)

        with torch.no_grad():
            self_noise = _self_noise(layer, x, n=3)
        diff = _max_diff(out_full, out_tok)
        scale = max(out_full.abs().max().item(), 1e-10)
        bound = max(NOISE_K * self_noise, REL_FLOOR * scale, ATOL)
        check(
            f"H={H}: full vs token-by-token",
            diff <= bound,
            f"diff={diff:.2e}, bound={bound:.2e}, rel={diff/scale:.2e}",
        )
    print()


def test_train_vs_inference():
    """Training-mode forward must match eval-mode forward on identical weights+input
    (T>64 so the chunked path is exercised)."""
    print("=== Train forward vs inference forward ===")
    for H in [1, 2]:
        layer, cfg = make_layer(num_heads=H, memory_block_size=64)
        torch.manual_seed(7)
        x = torch.randn(2, 256, cfg["dim"], device="cuda", dtype=torch.bfloat16) * 0.1
        with torch.no_grad():
            layer.train()
            out_train = layer(x.clone())[0]
            train_noise = _self_noise(layer, x, n=3)
            layer.eval()
            out_inf = layer(x.clone())[0]
            inf_noise = _self_noise(layer, x, n=3)
        diff = _max_diff(out_train, out_inf)
        noise = max(train_noise, inf_noise)
        scale = max(out_train.abs().max().item(), 1e-10)
        bound = max(NOISE_K * noise, REL_FLOOR * scale, ATOL)
        check(
            f"H={H}: train vs inference forward (T=256)",
            diff <= bound,
            f"diff={diff:.2e} (rel={diff/scale:.2e}), bound={bound:.2e} "
            f"(train_noise={train_noise:.2e}, inf_noise={inf_noise:.2e})",
        )
    print()


def test_create_cache():
    """create_kv_cache should produce a valid SDMLayerState."""
    print("=== Cache creation ===")
    layer, _ = make_layer()
    cache = layer.create_kv_cache(bsz=2, seq_len=0, dtype=torch.bfloat16, device="cuda")
    check("returns SDMLayerState", isinstance(cache, SDMState))
    check("memory shape", cache.memory.shape == (2 * 1024, 128), str(cache.memory.shape))
    check("cache_len=0", cache.cache_len == 0)
    print()


def test_2d_input():
    """2D input [B*T, D] should work (seq_len taken from freqs_cis' first dim)."""
    print("=== 2D input ===")
    layer, _ = make_layer()
    layer.eval()
    x_3d = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    x_2d = x_3d.reshape(64, 128)
    # SDM ignores RoPE values; only freqs_cis.shape[0] (=seq_len) is used for 2D inputs.
    freqs_cis = torch.zeros(32, 2, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out_3d, _ = layer(x_3d)
        out_2d, _ = layer(x_2d, freqs_cis=freqs_cis)

    out_3d_flat = out_3d.reshape(64, 128)
    diff = (out_3d_flat - out_2d).abs().max().item()
    scale = max(out_3d_flat.abs().max().item(), 1e-10)
    check("2D matches 3D", diff / scale < 0.02, f"rel={diff/scale:.2e}")
    print()


def test_weight_init_determinism():
    """Same seed must produce identical parameters."""
    print("=== Weight init determinism ===")
    layer_a, _ = make_layer(seed=123)
    layer_b, _ = make_layer(seed=123)

    all_match = True
    for (n1, p1), (n2, p2) in zip(layer_a.named_parameters(), layer_b.named_parameters()):
        if not torch.equal(p1, p2):
            all_match = False
            break
    check("same seed -> same params", all_match)
    print()


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}\n")

    test_forward_backward()
    test_chunk_size_invariance()
    test_inference_cache()
    test_train_vs_inference()
    test_create_cache()
    test_2d_input()
    test_weight_init_determinism()

    print(f"Results: {PASS} passed, {FAIL} failed")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
